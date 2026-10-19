r"""
Bivariate conditioning algorithm for approximating Gaussian probabilities,
see [Genz2016numerical]_ and [Trinh2015bivariate]_
(``botorch.utils.probability.mvnxpb``).

The solver runs in Rust: ``MVNXPB`` keeps BoTorch's state (``step``,
``perm``, ``bounds``, ``piv_chol``, ``plug_ins``, ``log_prob``,
``log_prob_extra``) as arrays of the caller's type and every operation
(``solve``, ``select_pivot``, ``pivot_``, ``augment``) processes the
flattened batch in Rust and replaces the arrays by new ones (BoTorch
modifies some of them in place). ``solve()`` of a freshly constructed
instance whose torch covariance or bounds require gradients returns log
probabilities with exact gradients (forward-mode dual numbers through the
whole algorithm; BoTorch's are wrong for batches whose members pivot
differently); partial solves, pivots and augmentations are not
differentiable. Solving a complete problem again returns the log
probabilities (BoTorch raises ``UnboundLocalError`` for an odd number of
variables).

.. [Trinh2015bivariate]
    G. Trinh and A. Genz. Bivariate conditioning approximations for
    multivariate normal probabilities. Statistics and Computing, 2015.

.. [Genz2016numerical]
    A. Genz and G. Trinh. Numerical Computation of Multivariate Normal Probabilities
    using Bivariate Conditioning. Monte Carlo and Quasi-Monte Carlo Methods, 2016.

.. [Gibson1994monte]
    GJ. Gibson, CA Galsbey, and DA Elston. Monte Carlo evaluation of multivariate normal
    integrals and sensitivity to variate ordering. Advances in Numerical Methods and
    Applications. 1994.
"""

from __future__ import annotations

import importlib
import math
import warnings
from typing import Any, TypedDict, cast

from botorch_rs import _botorch_rs as _rs
from botorch_rs import _ops
from botorch_rs._arrays import (
    ArrayKind,
    first_kind,
    from_flat,
    is_torch_tensor,
    kind_of,
    to_flat,
)
from botorch_rs.utils.probability.linalg import PivotedCholesky


class mvnxpbState(TypedDict):
    step: int
    perm: Any
    bounds: Any
    piv_chol: PivotedCholesky
    plug_ins: Any
    log_prob: Any
    log_prob_extra: Any


def _mvnxpb_function() -> Any:
    fn = _ops._FUNCTIONS.get("mvnxpb")
    if fn is not None:
        return fn
    torch: Any = importlib.import_module("torch")

    class _MvnxpbLogProb(torch.autograd.Function):
        @staticmethod
        def forward(ctx: Any, cov: Any, bounds: Any) -> Any:  # type: ignore[override]
            *batch, n = cov.shape[:-1]
            nb = math.prod(batch)
            kind = ArrayKind("torch", cov.dtype, cov.device)
            values, gc, gb = _rs.mvnxpb_log_prob_with_grad(
                to_flat(cov)[0], to_flat(bounds)[0], nb, n
            )
            ctx.gc = from_flat(gc, tuple(cov.shape), kind)
            ctx.gb = from_flat(gb, tuple(bounds.shape), kind)
            return from_flat(values, tuple(batch), kind)

        @staticmethod
        def backward(ctx: Any, grad_output: Any) -> Any:  # type: ignore[override]
            g = grad_output[..., None, None]
            return g * ctx.gc, g * ctx.gb

    _ops._FUNCTIONS["mvnxpb"] = _MvnxpbLogProb
    return _MvnxpbLogProb


class MVNXPB:
    r"""An algorithm for approximating Gaussian probabilities ``P(X \in bounds)``, where
    ``X ~ N(0, covariance_matrix)``.
    """

    step: int
    perm: Any
    bounds: Any
    piv_chol: PivotedCholesky
    plug_ins: Any
    log_prob: Any
    log_prob_extra: Any

    def __init__(self, covariance_matrix: Any, bounds: Any) -> None:
        r"""Initializes an MVNXPB instance.

        Args:
            covariance_matrix: Covariance matrices of shape ``batch_shape x [n, n]``.
            bounds: Tensor of lower and upper bounds, ``batch_shape x [n, 2]``. These
                bounds are standardized internally and clipped to STANDARDIZED_RANGE.
        """
        kind = first_kind(covariance_matrix, bounds)
        cshape = _ops.shape_of(covariance_matrix)
        bshape = _ops.shape_of(bounds)
        n = cshape[-1]
        batch = tuple(cshape[:-2])
        from botorch_rs._arrays import broadcast_shapes

        full = broadcast_shapes(batch, tuple(bshape[:-2])) or batch
        cov = _ops.expand(covariance_matrix, (*full, n, n))
        bds = _ops.expand(bounds, (*full, n, 2))
        nb = math.prod(full)
        state = _rs.MvnxpbState.standardized(to_flat(cov)[0], to_flat(bds)[0], nb, n)
        self._kind = kind
        self._set_state(state, full)
        # The (differentiable) inputs of a fresh solve.
        self._source: tuple[Any, Any] | None = (cov, bds)

    @classmethod
    def build(
        cls,
        step: int,
        perm: Any,
        bounds: Any,
        piv_chol: PivotedCholesky,
        plug_ins: Any,
        log_prob: Any,
        log_prob_extra: Any = None,
    ) -> MVNXPB:
        r"""Creates an MVNXPB instance from raw arguments. Unlike MVNXPB.__init__,
        this methods does not preprocess or copy terms.

        Args:
            step: Integer used to track the solver's progress.
            bounds: Tensor of lower and upper bounds, ``batch_shape x [n, 2]``.
            piv_chol: A PivotedCholesky instance for the system.
            plug_ins: Tensor of plug-in estimators used to update lower and upper bounds
                on random variables that have yet to be integrated out.
            log_prob: Tensor of log probabilities.
            log_prob_extra: Tensor of conditional log probabilities for the next random
                variable. Used when integrating over an odd number of random variables.
        """
        new = cls.__new__(cls)
        new.step = step
        new.perm = perm
        new.bounds = bounds
        new.piv_chol = piv_chol
        new.plug_ins = plug_ins
        new.log_prob = log_prob
        new.log_prob_extra = log_prob_extra
        new._kind = kind_of(bounds)
        new._source = None
        return new

    # -- conversion to and from the Rust state -------------------------------

    def _set_state(self, state: Any, batch: tuple[int, ...]) -> None:
        kind = self._kind
        n = state.n
        self.step = state.step
        self.perm = _ops.int_array(state.perm, (*batch, n), kind)
        self.bounds = from_flat(state.bounds, (*batch, n, 2), kind)
        diag = state.diag
        self.piv_chol = PivotedCholesky(
            step=state.pc_step,
            tril=from_flat(state.tril, (*batch, n, n), kind),
            perm=_ops.int_array(state.pc_perm, (*batch, n), kind),
            diag=None if diag is None else from_flat(diag, (*batch, n), kind),
        )
        self.plug_ins = from_flat(state.plug_ins, (*batch, n), kind)
        self.log_prob = from_flat(state.log_prob, batch, kind)
        extra = state.log_prob_extra
        self.log_prob_extra = None if extra is None else from_flat(extra, batch, kind)

    def _state(self) -> tuple[Any, tuple[int, ...]]:
        shape = _ops.shape_of(self.bounds)
        batch, n = tuple(shape[:-2]), shape[-2]
        nb = math.prod(batch)

        def flat(x: Any, trailing: tuple[int, ...]) -> Any:
            return to_flat(_ops.expand(x, (*batch, *trailing)))[0]

        pc = self.piv_chol
        state = _rs.MvnxpbState(
            n,
            nb,
            int(self.step),
            int(pc.step),
            _ops.ints(_ops.expand(self.perm, (*batch, n))),
            flat(self.bounds, (n, 2)),
            flat(pc.tril, (n, n)),
            _ops.ints(_ops.expand(pc.perm, (*batch, n))),
            None if pc.diag is None else flat(pc.diag, (n,)),
            flat(self.plug_ins, (n,)),
            flat(self.log_prob, ()),
            None if self.log_prob_extra is None else flat(self.log_prob_extra, ()),
        )
        return state, batch

    # -- the solver --------------------------------------------------------------

    def solve(self, num_steps: int | None = None, eps: float = 1e-10) -> Any:
        r"""Runs the MVNXPB solver instance for a fixed number of steps.

        Calculates a bivariate conditional approximation to P(X \in bounds), where
        X ~ N(0, Σ). For details, see [Genz2016numerical] or [Trinh2015bivariate]_.
        """
        if self.step > self.piv_chol.step:
            raise ValueError("Invalid state: solver ran ahead of matrix decomposition.")
        source = getattr(self, "_source", None)
        state, batch = self._state()
        state.solve(None if not num_steps else int(num_steps), float(eps))
        self._set_state(state, batch)
        self._source = None
        if (
            source is not None
            and not num_steps
            and eps == 1e-10
            and _ops._requires_grad(list(source))
        ):
            torch = importlib.import_module("torch")
            cov, bds = source
            dtype = self.log_prob.dtype
            cov = cov if cov.dtype == dtype else cov.to(dtype)
            bds = torch.as_tensor(bds, dtype=dtype) if not is_torch_tensor(bds) else bds
            self.log_prob = _mvnxpb_function().apply(cov, bds)
        return self.log_prob

    def select_pivot(self) -> Any:
        r"""GGE variable prioritization strategy from [Gibson1994monte]_.

        Returns the index of the random variable least likely to satisfy its bounds
        when conditioning on the previously integrated random variables ``X[:t - 1]``
        attaining the values of plug-in estimators ``y[:t - 1]``. Equivalently,
        ```
        argmin_{i = t, ..., n} P(X[i] \in bounds[i] | X[:t-1] = y[:t -1]),
        ```
        where ``t`` denotes the current step."""
        state, batch = self._state()
        return _ops.int_array(state.select_pivot(), batch, self._kind)

    def pivot_(self, pivot: Any) -> None:
        r"""Swap random variables at ``pivot`` and ``step`` positions."""
        if self.step > self.piv_chol.step:
            raise ValueError
        state, batch = self._state()
        piv = _ops.ints(_ops.expand(pivot, batch) if batch else pivot)
        state.pivot(piv)
        self._set_state(state, batch)
        self._source = None

    def __getitem__(self, key: Any) -> MVNXPB:
        return self.build(
            step=self.step,
            perm=_ops.getitem(self.perm, key),
            bounds=_ops.getitem(self.bounds, key),
            piv_chol=self.piv_chol[key],
            plug_ins=_ops.getitem(self.plug_ins, key),
            log_prob=_ops.getitem(self.log_prob, key),
            log_prob_extra=(
                None if self.log_prob_extra is None else _ops.getitem(self.log_prob_extra, key)
            ),
        )

    def concat(self, other: MVNXPB, dim: int) -> MVNXPB:
        if not isinstance(other, MVNXPB):
            raise TypeError(f"Expected `other` to be {type(self)} typed but was {type(other)}.")
        batch_ndim = len(_ops.shape_of(self.log_prob))
        if dim > batch_ndim or dim < -batch_ndim:
            raise ValueError(f"`dim={dim}` is not a valid batch dimension.")
        state_dict: dict[str, Any] = dict(self.asdict())
        for key, _other in other.asdict().items():
            _self = state_dict.get(key)
            if _self is None and _other is None:
                continue
            if (_self is None) != (_other is None) or (
                isinstance(_self, PivotedCholesky) != isinstance(_other, PivotedCholesky)
            ):
                raise TypeError(
                    f"Concatenation failed: `self.{key}` has type {type(_self)}, "
                    f"but `other.{key}` is of type {type(_self)}."
                )
            if isinstance(_self, PivotedCholesky):
                state_dict[key] = _self.concat(cast(PivotedCholesky, _other), dim=dim)
            elif isinstance(_self, int):
                if _self != _other:
                    raise ValueError(
                        f"Concatenation failed: `self.{key}` does not equal `other.{key}`."
                    )
            else:
                state_dict[key] = _ops.concat((_self, _other), dim=dim)
        return self.build(**state_dict)

    def expand(self, *sizes: int) -> MVNXPB:
        state_dict: dict[str, Any] = dict(self.asdict())
        state_dict["piv_chol"] = state_dict["piv_chol"].expand(*sizes)
        for name, ndim in {
            "bounds": 2,
            "perm": 1,
            "plug_ins": 1,
            "log_prob": 0,
            "log_prob_extra": 0,
        }.items():
            src = state_dict[name]
            if src is not None:
                shape = _ops.shape_of(src)
                trailing = tuple(shape[-ndim:]) if ndim else ()
                state_dict[name] = _ops.expand(src, (*sizes, *trailing))
        return self.build(**state_dict)

    def augment(
        self,
        covariance_matrix: Any,
        bounds: Any,
        cross_covariance_matrix: Any,
        disable_pivoting: bool = False,
        jitter: float | None = None,
        max_tries: int | None = None,
    ) -> MVNXPB:
        r"""Augment an ``n``-dimensional MVNXPB instance to include ``m``
        additional random variables.
        """
        n = _ops.shape_of(self.perm)[-1]
        if n != self.piv_chol.step:
            raise NotImplementedError("Augmentation of incomplete solutions not implemented yet.")
        state, batch = self._state()
        m = _ops.shape_of(covariance_matrix)[-1]

        def flat(x: Any, trailing: tuple[int, ...]) -> Any:
            return to_flat(_ops.expand(x, (*batch, *trailing)))[0]

        from botorch_rs.exceptions.errors import NotPSDError

        try:
            new, recovered = state.augment(
                flat(covariance_matrix, (m, m)),
                flat(bounds, (m, 2)),
                flat(cross_covariance_matrix, (m, n)),
                m,
                bool(disable_pivoting),
                None if jitter is None else float(jitter),
                None if max_tries is None else int(max_tries),
            )
        except NotPSDError as e:
            warnings.warn(
                "Joint covariance matrix not positive definite, attempting recovery.",
                stacklevel=2,
            )
            raise NotPSDError(
                "Matrix not positive definite after repeatedly adding jitter up to "
                f"{_jitter_reached(jitter, max_tries):.1e}. Original error on first attempt: {e}"
            ) from None
        if recovered:
            warnings.warn(
                "Joint covariance matrix not positive definite, attempting recovery.",
                stacklevel=2,
            )
        out = MVNXPB.__new__(MVNXPB)
        out._kind = self._kind
        out._source = None
        out._set_state(new, batch)
        return out

    def detach(self) -> MVNXPB:
        state_dict: dict[str, Any] = dict(self.asdict())
        for key, obj in state_dict.items():
            if isinstance(obj, PivotedCholesky):
                state_dict[key] = obj.detach()
            elif obj is not None and not isinstance(obj, int):
                state_dict[key] = _ops.detach(obj)
        return self.build(**state_dict)

    def clone(self) -> MVNXPB:
        state_dict: dict[str, Any] = dict(self.asdict())
        for key, obj in state_dict.items():
            if isinstance(obj, PivotedCholesky):
                state_dict[key] = obj.clone()
            elif obj is not None and not isinstance(obj, int):
                state_dict[key] = _ops.clone(obj)
        return self.build(**state_dict)

    def asdict(self) -> mvnxpbState:
        return mvnxpbState(
            step=self.step,
            perm=self.perm,
            bounds=self.bounds,
            piv_chol=self.piv_chol,
            plug_ins=self.plug_ins,
            log_prob=self.log_prob,
            log_prob_extra=self.log_prob_extra,
        )


def _jitter_reached(jitter: float | None, max_tries: int | None) -> float:
    j = 1e-8 if jitter is None else jitter
    tries = 3 if max_tries is None else max_tries
    return j * 10 ** (tries - 1)


__all__ = ["MVNXPB", "mvnxpbState"]

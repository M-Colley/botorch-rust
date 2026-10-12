"""Cost-aware utilities (``botorch.acquisition.cost_aware``): the trade-off
between the improvement in utility of a candidate set and the cost of
evaluating it, used by cost-aware and multi-fidelity acquisition functions."""

from __future__ import annotations

from typing import Any

from botorch_rs import _botorch_rs as _rs
from botorch_rs._arrays import Flat, first_kind, from_pair, is_torch_tensor, to_flat
from botorch_rs.exceptions.errors import BotorchTensorDimensionError
from botorch_rs.models.cost import cost_model_handle


def _mask_arg(mask: Any) -> tuple[list[bool], list[int]] | None:
    """An evaluation mask as ``(data, shape)``."""
    if mask is None:
        return None
    if is_torch_tensor(mask):
        return [bool(v) for v in mask.reshape(-1).tolist()], list(mask.shape)
    rows = [[bool(v) for v in row] for row in (mask.tolist() if hasattr(mask, "tolist") else mask)]
    return [v for r in rows for v in r], [len(rows), len(rows[0]) if rows else 0]


def _utility_args(X: Any, deltas: Any) -> tuple[Flat, Flat]:
    x = to_flat(X, "X")
    if len(x[1]) == 2:
        x = (x[0], (1, *x[1]))
    if len(x[1]) < 3:
        raise BotorchTensorDimensionError(
            f"Expected X to be `batch_shape x q x d`, got shape {x[1]}."
        )
    return x, to_flat(deltas, "deltas")


class CostAwareUtility:
    """Abstract base class of the cost-aware utilities."""

    _rs_utility: Any

    def forward(self, X: Any, deltas: Any, sampler: Any = None) -> Any:
        raise NotImplementedError  # pragma: no cover

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.forward(*args, **kwargs)


class GenericCostAwareUtility(CostAwareUtility):
    r"""A cost-aware utility given by a callable ``cost(X, deltas)`` that
    maps the candidates (``batch_shape x q x d``) and the improvements
    (``num_fantasies x batch_shape``) to the utilities (on torch tensors).

    Inside botorch_rs's acquisition functions the callable is evaluated from
    Rust with gradients from ``torch.autograd``.
    """

    def __init__(self, cost: Any) -> None:
        self._cost_callable = cost
        self._rs_utility = _rs.CostAwareUtility.generic(cost)

    def forward(self, X: Any, deltas: Any, sampler: Any = None) -> Any:
        """``cost(X, deltas)`` (the sampler is ignored)."""
        return self._cost_callable(X, deltas)


class InverseCostWeightedUtility(CostAwareUtility):
    r"""A cost-aware utility that weights the improvement by the inverse cost:
    with the cost ``c`` of a candidate set (the costs of its points summed
    over the q-batch, after ``cost_objective``), a positive improvement
    ``delta`` becomes ``delta / c`` and a non-positive one ``delta * c``;
    with ``log=True``, a log improvement becomes ``delta - log(c)``.

    ``cost_model`` is a botorch_rs cost model (``AffineFidelityCostModel``,
    ``FixedCostModel``), a botorch_rs GP (its posterior mean is the cost) or
    another model callable on torch tensors (e.g. BoTorch's
    ``GenericDeterministicModel``). Without ``cost_objective``, the costs of a
    single-output cost model are used as they are and those of several
    outputs are summed. ``use_mean=False`` (samples of the cost) is
    supported for deterministic cost models, whose samples are their values.
    """

    def __init__(
        self,
        cost_model: Any,
        use_mean: bool = True,
        cost_objective: Any = None,
        log: bool = False,
    ) -> None:
        from botorch_rs.acquisition.objective import GenericMCObjective, IdentityMCObjective

        self.cost_model = cost_model
        handle = cost_model_handle(cost_model)
        if cost_objective is None:
            self.cost_objective: Any = (
                IdentityMCObjective()
                if handle.num_outputs == 1
                else GenericMCObjective(lambda Y, X=None: Y.sum(dim=-1))
            )
        else:
            self.cost_objective = cost_objective
        self._use_mean = use_mean
        self._log = log
        self._rs_utility = _rs.CostAwareUtility.inverse_cost_weighted(
            handle, bool(use_mean), cost_objective, bool(log)
        )

    def forward(
        self,
        X: Any,
        deltas: Any,
        sampler: Any = None,
        X_evaluation_mask: Any = None,
    ) -> Any:
        """The cost-weighted utilities (``num_fantasies x batch_shape``) of the
        improvements ``deltas`` at the candidate sets ``X``."""
        if not self._use_mean and sampler is None:
            raise RuntimeError("Must provide `sampler` if `use_mean=False`")
        x, d = _utility_args(X, deltas)
        mask = _mask_arg(X_evaluation_mask)
        kind = first_kind(deltas, X)
        torch_grad = False
        if is_torch_tensor(X) or is_torch_tensor(deltas):
            import torch

            torch_grad = torch.is_grad_enabled() and (
                getattr(X, "requires_grad", False) or getattr(deltas, "requires_grad", False)
            )
        if torch_grad:
            return _utility_autograd(self._rs_utility, X, deltas, mask)
        values, _ = self._rs_utility.forward(x, d, mask)
        return from_pair(values, kind)


def _utility_autograd(rs: Any, X: Any, deltas: Any, mask: Any) -> Any:
    """The utilities connected to torch's autograd graph (gradients from
    Rust)."""
    import torch

    from botorch_rs._arrays import kind_of

    X_t = torch.as_tensor(X)
    d_t = torch.as_tensor(deltas)

    class _Fn(torch.autograd.Function):
        @staticmethod
        def forward(ctx: Any, Xa: Any, da: Any) -> Any:  # type: ignore[override]
            ctx.save_for_backward(Xa, da)
            x, d = _utility_args(Xa, da)
            values, _ = rs.forward(x, d, mask)
            return from_pair(values, kind_of(da))

        @staticmethod
        def backward(ctx: Any, grad_output: Any) -> Any:  # type: ignore[override]
            Xa, da = ctx.saved_tensors
            x, d = _utility_args(Xa, da)
            _, grads = rs.forward(x, d, mask, to_flat(grad_output.contiguous()))
            dbar, xbar = grads
            gx = from_pair(xbar, kind_of(Xa)).reshape(Xa.shape)
            return gx, from_pair(dbar, kind_of(da))

    return _Fn.apply(X_t, d_t)


__all__ = ["CostAwareUtility", "GenericCostAwareUtility", "InverseCostWeightedUtility"]

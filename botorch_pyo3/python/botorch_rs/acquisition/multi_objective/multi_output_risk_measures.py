r"""Multi-output risk measures
(``botorch.acquisition.multi_objective.multi_output_risk_measures``): MC
multi-output objectives that map the posterior samples of the ``n_w``
perturbations of every candidate (``sample_shape x batch_shape x (q * n_w) x
m``) to a multi-output risk measure (``sample_shape x batch_shape x q' x
m'``), for robust multi-objective Bayesian optimization (Daulton et al.,
*Robust Multi-Objective Bayesian Optimization Under Input Noise*, ICML 2022).

``MVaR`` (the multivariate Value-at-Risk, Prékopa 2012), ``IndependentCVaR``,
``IndependentVaR``, ``MultiOutputWorstCase`` and ``MultiOutputExpectation``
are evaluated in Rust, natively inside qEHVI, qLogEHVI, qNEHVI and qLogNEHVI
(as their ``objective``) and when called directly (with exact gradients
through torch's autograd). ``MARS`` (MVaR approximation based on random
Chebyshev scalarizations, Daulton et al., 2022) is a single-output objective
for the single-objective acquisition functions (e.g. qLogNEI).

Differences from BoTorch main (see ``docs/drafts/mo-robust.md``): tied
alpha-level points are kept (meta-pytorch/botorch#3392), ``MVaR`` with
``expectation=True`` has gradients (BoTorch returns NaN), and inside an
acquisition function the MVaR sets of every candidate set are padded on
their own (BoTorch pads to the largest set of the whole t-batch, which makes
qLogEHVI and qLogNEHVI values depend on the other candidates of the batch).

Example:
    >>> model = SingleTaskGP(train_X, train_Y, input_transform=InputPerturbation(w_set))
    >>> objective = MVaR(n_w=w_set.shape[0], alpha=0.9)
    >>> acqf = qNoisyExpectedHypervolumeImprovement(
    ...     model, ref_point, train_X, objective=objective
    ... )
"""

from __future__ import annotations

import importlib
import warnings
from collections.abc import Callable
from typing import Any

from botorch_rs import _botorch_rs as _rs
from botorch_rs import _ops
from botorch_rs._arrays import first_kind, from_flat, from_pair, is_torch_tensor, kind_of, to_flat
from botorch_rs.acquisition import _native_objective
from botorch_rs.acquisition.multi_objective.objective import (
    IdentityMCMultiOutputObjective,
    MCMultiOutputObjective,
    mo_objective_arg,
)
from botorch_rs.acquisition.risk_measures import (
    _NATIVE_FORWARDS,
    CVaR,
    RiskMeasureMCObjective,
    VaR,
    _class_names,
)
from botorch_rs.exceptions.errors import BotorchError, UnsupportedError
from botorch_rs.exceptions.warnings import BotorchWarning

_MVAR_MAPPING_AUTOGRAD: Any = None


def _mapping_autograd() -> Any:
    global _MVAR_MAPPING_AUTOGRAD
    if _MVAR_MAPPING_AUTOGRAD is None:
        torch: Any = importlib.import_module("torch")

        class _Mapping(torch.autograd.Function):
            @staticmethod
            def forward(ctx: Any, prepared: Any, mvars: Any) -> Any:  # type: ignore[override]
                ctx.prepared = to_flat(prepared, "prepared_samples")
                ctx.mvars = to_flat(mvars, "mvars")
                ctx.kind = kind_of(prepared)
                return from_pair(_rs.mvar_make_differentiable(ctx.prepared, ctx.mvars), ctx.kind)

            @staticmethod
            def backward(ctx: Any, gradient: Any) -> tuple[Any, None]:  # type: ignore[override]
                values = _rs.mvar_make_differentiable(
                    ctx.prepared, ctx.mvars, to_flat(gradient, "adjoint")
                )
                return from_pair(values, ctx.kind), None

        _MVAR_MAPPING_AUTOGRAD = _Mapping
    return _MVAR_MAPPING_AUTOGRAD


class MultiOutputRiskMeasureMCObjective(RiskMeasureMCObjective, MCMultiOutputObjective):
    r"""Base class of the multi-output risk measures: objectives transforming
    the multi-output posterior samples of the ``n_w`` points of every
    candidate into samples of a multi-output risk measure."""

    def _default_preprocessing(self) -> Any:
        return IdentityMCMultiOutputObjective()

    def _preprocessing_spec(self) -> tuple[Any, ...] | None:
        """The preprocessing function for the extension (``None``: all
        outputs)."""
        pre = self.preprocessing_function
        if type(pre) is IdentityMCMultiOutputObjective and pre._outcomes is None:
            return None
        if isinstance(pre, MCMultiOutputObjective) or (
            "MCMultiOutputObjective" in _class_names(pre)
        ):
            return mo_objective_arg(pre)
        if not callable(pre):
            raise BotorchError("`preprocessing_function` must be callable.")
        return ("callable", pre)

    def _prepare_samples(self, samples: Any) -> Any:
        r"""BoTorch's ``_prepare_samples`` (torch tensors): the preprocessed
        samples as ``sample_shape x batch_shape x q x n_w x m'``."""
        samples = self.preprocessing_function(samples)
        return samples.view(*samples.shape[:-2], -1, self.n_w, samples.shape[-1])

    def _mo_spec(self) -> tuple[Any, ...]:
        """Computed in Rust unless a subclass overrides ``forward``."""
        if self._rs is not None and type(self).forward in _NATIVE_FORWARDS:
            return ("native", self._rs.objective())
        return ("callable", self)


class MultiOutputExpectation(MultiOutputRiskMeasureMCObjective):
    r"""The multi-output expectation risk measure: the mean of every output
    over the ``n_w`` points of every candidate."""

    def __init__(
        self,
        n_w: int,
        preprocessing_function: Callable[[Any], Any] | None = None,
    ) -> None:
        super().__init__(n_w=n_w, preprocessing_function=preprocessing_function)
        self._rs = _rs.MultiOutputRiskMeasure.expectation(self.n_w, self._preprocessing_spec())


class IndependentCVaR(CVaR, MultiOutputRiskMeasureMCObjective):
    r"""The multi-output Conditional Value-at-Risk with every output
    evaluated independently (the CVaR of each output; the result may not be
    attainable by any single perturbation)."""

    def _build(self) -> Any:
        return _rs.MultiOutputRiskMeasure.independent_cvar(
            float(self.alpha), self.n_w, self._preprocessing_spec()
        )


class IndependentVaR(IndependentCVaR):
    r"""The multi-output Value-at-Risk with every output evaluated
    independently."""

    def _build(self) -> Any:
        return _rs.MultiOutputRiskMeasure.independent_var(
            float(self.alpha), self.n_w, self._preprocessing_spec()
        )


class MultiOutputWorstCase(MultiOutputRiskMeasureMCObjective):
    r"""The multi-output worst-case risk measure: the smallest value of every
    output over the ``n_w`` points of every candidate."""

    def __init__(
        self,
        n_w: int,
        preprocessing_function: Callable[[Any], Any] | None = None,
    ) -> None:
        super().__init__(n_w=n_w, preprocessing_function=preprocessing_function)
        self._rs = _rs.MultiOutputRiskMeasure.worst_case(self.n_w, self._preprocessing_spec())


class MVaR(MultiOutputRiskMeasureMCObjective):
    r"""The multivariate Value-at-Risk (Prékopa, 2012): the non-dominated
    points ``z`` such that ``P(Y >= z) >= alpha`` over the ``n_w`` points of
    every candidate (each dominates an ``alpha`` fraction of the samples).

    The MVaR set of every candidate and sample is padded (by repeating its
    first point) to the largest set over all candidates and samples (or to
    ``n_w`` with ``pad_to_n_w``), so the result is ``sample_shape x
    batch_shape x (q * k') x m``; with ``expectation=True`` the mean of the
    set is returned (Cousin and Di Bernardino, 2013), ``sample_shape x
    batch_shape x q x m``. Gradients flow to the samples that attain the
    MVaR components (BoTorch's ``make_differentiable``).
    """

    _verify_output_shape = False

    def __init__(
        self,
        n_w: int,
        alpha: float,
        expectation: bool = False,
        preprocessing_function: Callable[[Any], Any] | None = None,
        *,
        pad_to_n_w: bool = False,
        filter_dominated: bool = True,
        use_counting: bool = False,
    ) -> None:
        r"""The multivariate Value-at-Risk.

        Args:
            n_w: The size of the ``w_set`` to calculate the risk measure over.
            alpha: The risk level of MVaR, float in ``(0.0, 1.0]``.
            expectation: Return the expectation of the MVaR set instead of the
                set.
            preprocessing_function: A preprocessing function to apply to the
                samples before computing the risk measure.
            pad_to_n_w: Pad every MVaR set to ``n_w`` points instead of the
                size of the largest set.
            filter_dominated: Return only the non-dominated alpha-level points.
            use_counting: Use BoTorch's counting algorithm
                (``get_mvar_set_via_counting``), which orders the points
                differently.
        """
        super().__init__(n_w=n_w, preprocessing_function=preprocessing_function)
        if not 0 < alpha <= 1:
            raise ValueError("`alpha` must be in (0.0, 1.0]")
        self.alpha = alpha
        self.expectation = bool(expectation)
        self.pad_to_n_w = bool(pad_to_n_w)
        self.filter_dominated = bool(filter_dominated)
        self.use_counting = bool(use_counting)
        self._rs = _rs.MultiOutputRiskMeasure.mvar(
            self.n_w,
            float(alpha),
            self.expectation,
            self._preprocessing_spec(),
            self.pad_to_n_w,
            self.filter_dominated,
            self.use_counting,
        )

    def make_differentiable(self, prepared_samples: Any, mvars: Any) -> Any:
        """Map each MVaR component to the mean of matching sample components.

        Exact native VJPs divide gradient equally among ties. Unattained
        components retain upstream's NaN value and zero sample gradients.
        The supplied MVaR values select matches and do not receive gradients.
        """
        if is_torch_tensor(prepared_samples) and prepared_samples.requires_grad:
            torch = importlib.import_module("torch")
            if torch.is_grad_enabled():
                return _mapping_autograd().apply(prepared_samples, mvars)
        return from_pair(
            _rs.mvar_make_differentiable(
                to_flat(prepared_samples, "prepared_samples"), to_flat(mvars, "mvars")
            ),
            kind_of(prepared_samples),
        )

    def _mvar_sets(self, Y: Any, use_counting: bool) -> list[Any]:
        kind = kind_of(Y)
        data, shape = to_flat(Y, "Y")
        if len(shape) != 3 or shape[-2] != self.n_w:
            raise BotorchError(
                f"Expected Y of shape `batch x n_w x m` with n_w = {self.n_w}, got {shape}."
            )
        return [from_pair(s, kind) for s in self._rs.mvar_set((data, shape), use_counting)]

    def get_mvar_set_via_counting(self, Y: Any) -> list[Any]:
        r"""The MVaR set of every batch element of ``Y`` (``batch x n_w x m``)
        with BoTorch's counting algorithm."""
        return self._mvar_sets(Y, True)

    def get_mvar_set_vectorized(self, Y: Any) -> list[Any]:
        r"""The MVaR set of every batch element of ``Y`` (``batch x n_w x m``)
        with BoTorch's vectorized algorithm."""
        return self._mvar_sets(Y, False)


class MARS(VaR, MultiOutputRiskMeasureMCObjective):
    r"""MVaR approximation based on random scalarizations (Daulton et al.,
    2022): the VaR of the Chebyshev scalarization ``min_t w_t (y_t - r_t)``
    of the outputs normalized with bounds from the MVaR of ``baseline_Y``
    (``r``: the normalized reference point, if given). One set of weights
    approximates one MVaR point; the result is a single-output objective.

    Set the baseline with ``set_baseline_Y`` before using the objective.
    """

    _is_mo: bool = False

    def __init__(
        self,
        alpha: float,
        n_w: int,
        chebyshev_weights: Any,
        baseline_Y: Any = None,
        ref_point: Any = None,
        preprocessing_function: Callable[[Any], Any] | None = None,
    ) -> None:
        r"""Transform the posterior samples to samples of a risk measure.

        Args:
            alpha: The risk level, float in ``(0.0, 1.0]``.
            n_w: The size of the perturbation set.
            chebyshev_weights: The (non-negative) weights of the Chebyshev
                scalarization.
            baseline_Y: An ``n' x m``-dim array of baseline outcomes for the
                normalization bounds (see ``set_baseline_Y``).
            ref_point: An optional MVaR reference point for the normalization
                bounds.
            preprocessing_function: A preprocessing function to apply to the
                samples before computing the risk measure.
        """
        self._chebyshev_objective: Any = None
        self._kind = first_kind(chebyshev_weights, baseline_Y, ref_point)
        self._weights = _weights_arg(chebyshev_weights)
        self._ref = None if ref_point is None else [float(v) for v in to_flat(ref_point)[0]]
        self._baseline = None if baseline_Y is None else to_flat(baseline_Y, "baseline_Y")
        if self._baseline is not None and len(self._baseline[1]) != 2:
            raise UnsupportedError("Batched Y is not supported.")
        super().__init__(alpha=alpha, n_w=n_w, preprocessing_function=preprocessing_function)
        self.ref_point = ref_point
        self.mvar = MVaR(n_w=self.n_w, alpha=self.alpha)

    def _build(self) -> Any:
        return _rs.RiskMeasure.mars(
            float(self.alpha),
            self.n_w,
            self._weights,
            self._baseline,
            self._ref,
            self._preprocessing_spec(),
        )

    def _mo_spec(self) -> tuple[Any, ...]:
        raise UnsupportedError(
            "MARS is a single-output objective (the VaR of a Chebyshev scalarization); use "
            "it with single-objective acquisition functions such as "
            "qLogNoisyExpectedImprovement."
        )

    def forward(self, samples: Any, X: Any = None) -> Any:
        """The VaR of the Chebyshev scalarizations (``sample_shape x
        batch_shape x q``)."""
        if self._baseline is None:
            raise RuntimeError(
                "baseline_Y must be set before constructing the Chebyshev objective."
            )
        return _native_objective.evaluate(self._rs, samples)

    @property
    def chebyshev_objective(self) -> Callable[[Any, Any], Any]:
        """Cached normalized Chebyshev scalarization, before VaR reduction."""
        if self._chebyshev_objective is None:
            if self._baseline is None:
                raise RuntimeError(
                    "baseline_Y must be set before constructing the Chebyshev objective."
                )
            bounds = self._rs.normalization_bounds
            assert bounds is not None
            values, shape = bounds
            m = shape[-1]
            lower = list(values[:m])
            upper = list(values[m:])
            reference = None if self.ref_point is None else list(to_flat(self.ref_point)[0])

            def objective(Y: Any, X: Any = None) -> Any:
                samples = self.preprocessing_function(Y)
                kind = kind_of(samples)
                if not _ops.shape_of(samples) or _ops.shape_of(samples)[-1] != m:
                    raise BotorchError(
                        f"Chebyshev scalarization expects {m} preprocessed outcomes."
                    )
                lo = from_flat(lower, (m,), kind)
                hi = from_flat(upper, (m,), kind)
                width = _ops.apply("sub_raw", hi, lo)
                normalized = _ops.apply("div_raw", _ops.apply("sub_raw", samples, lo), width)
                if reference is not None:
                    point = from_flat(reference, (m,), kind)
                    point = _ops.apply("div_raw", _ops.apply("sub_raw", point, lo), width)
                    normalized = _ops.apply("sub_raw", normalized, point)
                weights = from_flat(self._weights, (len(self._weights),), kind)
                return _ops.reduce("min", _ops.apply("mul_raw", normalized, weights), -1)

            self._chebyshev_objective = objective
        result: Callable[[Any, Any], Any] = self._chebyshev_objective
        return result

    @property
    def chebyshev_weights(self) -> Any:
        """The weights of the Chebyshev scalarization."""
        return from_flat(self._weights, (len(self._weights),), self._kind)

    @chebyshev_weights.setter
    def chebyshev_weights(self, chebyshev_weights: Any) -> None:
        weights = _weights_arg(chebyshev_weights)
        native = self._rs.with_chebyshev_weights(weights)
        self._weights, self._rs = weights, native
        self._chebyshev_objective = None

    @property
    def baseline_Y(self) -> Any:
        """The baseline outcomes of the normalization bounds (``None`` until
        set)."""
        if self._baseline is None:
            return None
        return from_pair(self._baseline, self._kind)

    @baseline_Y.setter
    def baseline_Y(self, baseline_Y: Any) -> None:
        self._chebyshev_objective = None
        if baseline_Y is None:
            self._baseline = None
            self._rs = self._build()
            return
        flat = to_flat(baseline_Y, "baseline_Y")
        if len(flat[1]) != 2:
            raise UnsupportedError("Batched Y is not supported.")
        self._kind = first_kind(baseline_Y) if kind_of(baseline_Y).lib != "list" else self._kind
        self._rs = self._rs.with_baseline_y(flat)
        self._baseline = flat

    def set_baseline_Y(self, model: Any, X_baseline: Any, Y_samples: Any = None) -> None:
        r"""Sets ``baseline_Y`` to the non-dominated MVaR points of the
        posterior means of the ``n_w`` perturbations of every point of
        ``X_baseline`` (``n x d``) under ``model`` (a botorch_rs model with an
        ``InputPerturbation``), or of the predictions ``Y_samples``
        (``(n * n_w) x m``) if given."""
        if Y_samples is None:
            if model is None or X_baseline is None:
                raise BotorchError("`model` and `X_baseline` are required without `Y_samples`.")
            self._kind = first_kind(X_baseline, self._kind)
            rs = self._rs.with_baseline_from_model(
                model._acquisition_model(None), to_flat(X_baseline, "X_baseline")
            )
        else:
            if model is not None or X_baseline is not None:
                warnings.warn(
                    "`model` and `X_baseline` are ignored when `Y_samples` is "
                    "provided to `MARS.set_baseline_Y`.",
                    BotorchWarning,
                    stacklevel=2,
                )
            self._kind = first_kind(Y_samples, self._kind)
            rs = self._rs.with_baseline_from_samples(to_flat(Y_samples, "Y_samples"))
        self._rs = rs
        self._chebyshev_objective = None
        y = rs.baseline_y
        self._baseline = None if y is None else (y[0], tuple(y[1]))

    @property
    def normalization_bounds(self) -> Any:
        """The ``2 x m`` normalization bounds of the scalarization (a
        botorch_rs extension; ``None`` until the baseline is set)."""
        bounds = self._rs.normalization_bounds
        return None if bounds is None else from_pair(bounds, self._kind)

    @staticmethod
    def _get_Y_normalization_bounds(Y: Any, ref_point: Any = None) -> Any:
        r"""BoTorch's normalization bounds (``2 x m``) of the outcomes ``Y``
        (``n x m``): the nadir and ideal points of the Pareto front of ``Y``
        (relative to ``ref_point`` when some Pareto point is better than it)."""
        ref = None if ref_point is None else [float(v) for v in to_flat(ref_point)[0]]
        return from_pair(
            _rs.mars_y_normalization_bounds(to_flat(Y, "Y"), ref), first_kind(Y, ref_point)
        )


def _weights_arg(weights: Any) -> list[float]:
    data, shape = to_flat(weights, "chebyshev_weights")
    if len(shape) != 1:
        raise UnsupportedError("Batched weights are not supported in MARS.")
    values = [float(w) for w in data]
    if any(w < 0 for w in values):
        raise UnsupportedError("Negative weights are not supported in MARS.")
    return values


_NATIVE_FORWARDS.add(MARS.forward)


def mo_risk_measure_from_botorch(objective: Any) -> MultiOutputRiskMeasureMCObjective | None:
    """The botorch_rs equivalent of one of BoTorch's multi-output risk
    measures, or ``None`` for other objects (including subclasses)."""
    names = [
        c.__name__
        for c in type(objective).__mro__
        if getattr(c, "__module__", "").startswith("botorch.acquisition.")
    ]
    name = type(objective).__name__
    if name not in names:
        return None
    pre = getattr(objective, "preprocessing_function", None)
    n_w = int(getattr(objective, "n_w", 0))
    if name == "MVaR":
        return MVaR(
            n_w=n_w,
            alpha=float(objective.alpha),
            expectation=bool(objective.expectation),
            preprocessing_function=pre,
            pad_to_n_w=bool(objective.pad_to_n_w),
            filter_dominated=bool(objective.filter_dominated),
            use_counting=bool(objective.use_counting),
        )
    if name in ("IndependentCVaR", "IndependentVaR"):
        cls = IndependentCVaR if name == "IndependentCVaR" else IndependentVaR
        return cls(alpha=float(objective.alpha), n_w=n_w, preprocessing_function=pre)
    if name == "MultiOutputWorstCase":
        return MultiOutputWorstCase(n_w=n_w, preprocessing_function=pre)
    if name == "MultiOutputExpectation":
        return MultiOutputExpectation(n_w=n_w, preprocessing_function=pre)
    return None


__all__ = [
    "IndependentCVaR",
    "IndependentVaR",
    "MARS",
    "MVaR",
    "MultiOutputExpectation",
    "MultiOutputRiskMeasureMCObjective",
    "MultiOutputWorstCase",
]

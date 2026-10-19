"""Analytic acquisition functions (``botorch.acquisition.analytic``), ``q = 1``.

Values are BoTorch's to ~1e-10 (``best_f`` / ``beta`` and constraint bounds are
used as exact float64 values; BoTorch rounds Python floats passed for them to
float32). Multi-output models require a ``posterior_transform``, except for
the constrained acquisition functions, which read several outputs.
"""

from __future__ import annotations

from typing import Any

from botorch_rs import _botorch_rs as _rs
from botorch_rs._arrays import as_float
from botorch_rs.acquisition.acquisition import AcquisitionFunction, scalar_arg
from botorch_rs.acquisition.objective import PosteriorTransform
from botorch_rs.exceptions.errors import UnsupportedError
from botorch_rs.exceptions.warnings import legacy_ei_numerics_warning


class AnalyticAcquisitionFunction(AcquisitionFunction):
    """Base class of the analytic acquisition functions (one point per
    candidate set, i.e. ``X`` of shape ``batch_shape x 1 x d``)."""

    def __init__(
        self,
        model: Any,
        posterior_transform: Any = None,
        allow_multi_output: bool = False,
    ) -> None:
        super().__init__(model)
        if posterior_transform is None:
            if not allow_multi_output and model.num_outputs != 1:
                raise UnsupportedError(
                    "Must specify a posterior transform when using a multi-output model."
                )
        elif not isinstance(posterior_transform, PosteriorTransform):
            raise UnsupportedError("AnalyticAcquisitionFunctions only support PosteriorTransforms.")
        self.posterior_transform = posterior_transform
        self._rs_model = self._model_handle(posterior_transform)

    def _build(self, **kwargs: Any) -> None:
        self._rs = _rs.AcquisitionFunction.analytic(type(self)._rs_name, self._rs_model, **kwargs)

    _rs_name: str = ""

    def set_X_pending(self, X_pending: Any = None) -> None:
        raise UnsupportedError("Analytic acquisition functions do not account for X_pending yet.")


class _ImprovementBased(AnalyticAcquisitionFunction):
    def __init__(
        self,
        model: Any,
        best_f: Any,
        posterior_transform: Any = None,
        maximize: bool = True,
    ) -> None:
        super().__init__(model, posterior_transform)
        self.best_f = best_f
        self.maximize = maximize
        self._build(best_f=scalar_arg(best_f, "best_f"), maximize=bool(maximize))


class ExpectedImprovement(_ImprovementBased):
    r"""Single-outcome Expected Improvement,
    ``EI(x) = E[max(f(x) - best_f, 0)]`` (BoTorch recommends
    :class:`LogExpectedImprovement`)."""

    _rs_name = "ExpectedImprovement"

    def __init__(
        self,
        model: Any,
        best_f: Any,
        posterior_transform: Any = None,
        maximize: bool = True,
    ) -> None:
        legacy_ei_numerics_warning(legacy_name=type(self).__name__)
        super().__init__(model, best_f, posterior_transform, maximize)


class LogExpectedImprovement(_ImprovementBased):
    r"""Logarithm of single-outcome Expected Improvement, computed in a
    numerically robust way (Ament et al., 2023)."""

    _rs_name = "LogExpectedImprovement"
    _log = True


class ProbabilityOfImprovement(_ImprovementBased):
    r"""Single-outcome Probability of Improvement ``P(f(x) >= best_f)``."""

    _rs_name = "ProbabilityOfImprovement"


class LogProbabilityOfImprovement(_ImprovementBased):
    r"""Logarithm of single-outcome Probability of Improvement."""

    _rs_name = "LogProbabilityOfImprovement"
    _log = True


class UpperConfidenceBound(AnalyticAcquisitionFunction):
    r"""Single-outcome Upper Confidence Bound ``mean + sqrt(beta) * std``."""

    _rs_name = "UpperConfidenceBound"

    def __init__(
        self,
        model: Any,
        beta: Any,
        posterior_transform: Any = None,
        maximize: bool = True,
    ) -> None:
        super().__init__(model, posterior_transform)
        self.beta = beta
        self.maximize = maximize
        self._build(beta=scalar_arg(beta, "beta"), maximize=bool(maximize))


class PosteriorMean(AnalyticAcquisitionFunction):
    r"""The posterior mean (negated if ``maximize=False``)."""

    _rs_name = "PosteriorMean"

    def __init__(self, model: Any, posterior_transform: Any = None, maximize: bool = True) -> None:
        super().__init__(model, posterior_transform)
        self.maximize = maximize
        self._build(maximize=bool(maximize))


class PosteriorStandardDeviation(AnalyticAcquisitionFunction):
    r"""The posterior standard deviation (negated if ``maximize=False``)."""

    _rs_name = "PosteriorStandardDeviation"

    def __init__(self, model: Any, posterior_transform: Any = None, maximize: bool = True) -> None:
        super().__init__(model, posterior_transform)
        self.maximize = maximize
        self._build(maximize=bool(maximize))


def _output_index(index: Any, m: int, what: str) -> int:
    i = int(index)
    if not -m <= i < m:
        raise ValueError(f"{what} {i} is out of range for a model with {m} outputs.")
    return i % m


def _bound(value: Any, name: str) -> float | None:
    return None if value is None else as_float(value, name)


class _ConstrainedExpectedImprovementBase(AnalyticAcquisitionFunction):
    """Shared constructor of the constrained analytic acquisition functions."""

    def __init__(
        self,
        model: Any,
        best_f: Any,
        objective_index: int,
        constraints: dict[int, tuple[Any, Any]],
        maximize: bool = True,
    ) -> None:
        super().__init__(model=model, allow_multi_output=True)
        self.posterior_transform = None
        self.maximize = maximize
        self.objective_index = objective_index
        self.best_f = best_f
        self.constraints = constraints
        if len(constraints) == 0:
            raise ValueError("There must be at least one constraint.")
        if objective_index in constraints:
            raise ValueError("Output corresponding to objective should not be a constraint.")
        m = model.num_outputs
        obj = _output_index(objective_index, m, "objective_index")
        cons = []
        for k, bounds in constraints.items():
            lower, upper = bounds
            cons.append(
                (
                    _output_index(k, m, "constraint index"),
                    _bound(lower, f"constraints[{k}][0]"),
                    _bound(upper, f"constraints[{k}][1]"),
                )
            )
        self._rs = _rs.AcquisitionFunction.constrained(
            type(self).__name__,
            self._rs_model,
            scalar_arg(best_f, "best_f"),
            obj,
            cons,
            bool(maximize),
        )


class LogConstrainedExpectedImprovement(_ConstrainedExpectedImprovementBase):
    r"""Log Constrained Expected Improvement (feasibility-weighted):
    ``log EI(x) + sum_i log P(lower_i <= y_i(x) <= upper_i)`` for the
    objective output ``objective_index`` and independent constraint outputs
    (``constraints = {i: (lower, upper)}``, ``None`` for an unbounded side),
    computed in log space (Ament et al., 2023)."""

    _log = True

    def __init__(
        self,
        model: Any,
        best_f: Any,
        objective_index: int,
        constraints: dict[int, tuple[Any, Any]],
        maximize: bool = True,
    ) -> None:
        super().__init__(model, best_f, objective_index, constraints, maximize)


class ConstrainedExpectedImprovement(_ConstrainedExpectedImprovementBase):
    r"""Constrained Expected Improvement (feasibility-weighted):
    ``EI(x) * prod_i P(lower_i <= y_i(x) <= upper_i)`` (BoTorch recommends
    :class:`LogConstrainedExpectedImprovement`)."""

    def __init__(
        self,
        model: Any,
        best_f: Any,
        objective_index: int,
        constraints: dict[int, tuple[Any, Any]],
        maximize: bool = True,
    ) -> None:
        legacy_ei_numerics_warning(legacy_name=type(self).__name__)
        super().__init__(model, best_f, objective_index, constraints, maximize)


def _constraints_arg(constraints: dict[int, tuple[Any, Any]], m: int) -> list[Any]:
    cons = []
    for k, bounds in constraints.items():
        lower, upper = bounds
        cons.append(
            (
                _output_index(k, m, "constraint index"),
                _bound(lower, f"constraints[{k}][0]"),
                _bound(upper, f"constraints[{k}][1]"),
            )
        )
    return cons


class LogProbabilityOfFeasibility(AnalyticAcquisitionFunction):
    r"""Log Probability of Feasibility: the logarithm of the analytic
    probability of feasibility of independent Gaussian outputs,
    ``LogPF(x) = sum_i log(P(y_i in [lower_i, upper_i]))`` for
    ``constraints = {i: (lower, upper)}`` (``None`` for an unbounded side);
    ``q = 1``. The model may have several outputs."""

    _log = True

    def __init__(self, model: Any, constraints: dict[int, tuple[Any, Any]]) -> None:
        super().__init__(model=model, allow_multi_output=True)
        self.posterior_transform = None
        self.constraints = constraints
        if len(constraints) == 0:
            raise ValueError("There must be at least one constraint.")
        self._rs = _rs.log_probability_of_feasibility(
            self._rs_model, _constraints_arg(constraints, model.num_outputs)
        )


class ScalarizedPosteriorMean(AnalyticAcquisitionFunction):
    r"""Scalarized Posterior Mean: the posterior means of the ``q`` points of
    a candidate set weighted by ``weights`` (shape ``q``); pass ``-weights``
    to minimize the scalarized posterior mean."""

    def __init__(self, model: Any, weights: Any, posterior_transform: Any = None) -> None:
        from botorch_rs._arrays import to_flat

        super().__init__(model, posterior_transform)
        self.weights = weights
        data, shape = to_flat(weights, "weights")
        if len(shape) != 1:
            raise UnsupportedError("`weights` must be a one-dimensional tensor of shape `q`.")
        self._rs = _rs.scalarized_posterior_mean(self._rs_model, [float(v) for v in data])


class qAnalyticProbabilityOfImprovement(AnalyticAcquisitionFunction):
    r"""Approximate, single-outcome batch Probability of Improvement using
    MVNXPB.

    This implementation uses MVNXPB, a bivariate conditioning algorithm for
    approximating ``P(a <= Y <= b)`` for multivariate normal ``Y`` (Trinh &
    Genz, 2015). This (analytic) approximate q-PI is given by ``approx-qPI(X) =
    P(max Y >= best_f) = 1 - P(Y < best_f), Y ~ f(X), X = (x_1,...,x_q)``,
    where ``P(Y < best_f)`` is estimated using MVNXPB. Gradients are exact.
    """

    def __init__(
        self,
        model: Any,
        best_f: Any,
        posterior_transform: Any = None,
        maximize: bool = True,
    ) -> None:
        super().__init__(model, posterior_transform)
        self.maximize = maximize
        self.best_f = best_f
        self._rs = _rs.q_analytic_probability_of_improvement(
            self._rs_model, scalar_arg(best_f, "best_f"), bool(maximize)
        )


def _check_noisy_ei_model(model: Any) -> None:
    from botorch_rs.models.gp_regression import SingleTaskGP

    message = (
        "Only single-output `SingleTaskGP` models with known observation noise "
        "are currently supported for fantasy-based NEI & LogNEI."
    )
    if not isinstance(model, SingleTaskGP):
        raise UnsupportedError(f"{message} Model is not a `SingleTaskGP`.")
    if not model._rs.is_fixed_noise:
        raise UnsupportedError(
            f"{message} Model likelihood is not a `FixedNoiseGaussianLikelihood`."
        )
    if model.num_outputs != 1:
        raise UnsupportedError(f"{message} Model has {model.num_outputs} outputs.")


class _NoisyExpectedImprovementBase(AnalyticAcquisitionFunction):
    def _init_noisy(
        self,
        model: Any,
        X_observed: Any,
        num_fantasies: int,
        maximize: bool,
        sampler: Any,
        posterior_transform: Any,
        log: bool,
    ) -> None:
        from botorch_rs._arrays import first_kind, from_flat, to_flat
        from botorch_rs.acquisition.objective import transform_arg
        from botorch_rs.sampling.normal import NormalMCSampler, SobolQMCNormalSampler

        _check_noisy_ei_model(model)
        AcquisitionFunction.__init__(self, model)
        if posterior_transform is not None and not isinstance(
            posterior_transform, PosteriorTransform
        ):
            raise UnsupportedError("AnalyticAcquisitionFunctions only support PosteriorTransforms.")
        if sampler is None:
            sampler = SobolQMCNormalSampler(sample_shape=(int(num_fantasies),))
        elif not isinstance(sampler, NormalMCSampler):
            raise UnsupportedError("`sampler` must be a botorch_rs SobolQMCNormalSampler or IIDNormalSampler.")
        self.posterior_transform = posterior_transform
        self.maximize = maximize
        self.num_fantasies = int(num_fantasies)
        self.sampler = sampler
        flat = to_flat(X_observed, "X_observed")
        self._rs, best = _rs.noisy_expected_improvement(
            model._acquisition_model(None),
            flat,
            self.num_fantasies,
            bool(maximize),
            sampler._rs,
            log,
            transform_arg(posterior_transform, 1),
        )
        self.best_f = from_flat(best, (len(best),), first_kind(X_observed))


class NoisyExpectedImprovement(_NoisyExpectedImprovementBase):
    r"""Single-outcome Noisy Expected Improvement (via fantasies): the expected
    improvement averaged over ``num_fantasies`` noiseless fantasy models,
    each trained on ``X_observed`` with targets sampled from the posterior
    there, with the best fantasized target as ``best_f``.

    Requires a ``SingleTaskGP`` with known observation noise (``train_Yvar``).
    The fantasies are drawn with a ``SobolQMCNormalSampler`` of
    ``num_fantasies`` samples (random seed, as in BoTorch); the keyword-only
    ``sampler`` (a botorch_rs extension) gives explicit base samples.

    NOTE: It is strongly recommended to use ``LogNoisyExpectedImprovement``
    instead of regular ``NEI``.
    """

    def __init__(
        self,
        model: Any,
        X_observed: Any,
        num_fantasies: int = 20,
        maximize: bool = True,
        *,
        sampler: Any = None,
    ) -> None:
        _check_noisy_ei_model(model)
        legacy_ei_numerics_warning(legacy_name=type(self).__name__)
        self._init_noisy(model, X_observed, num_fantasies, maximize, sampler, None, False)


class LogNoisyExpectedImprovement(_NoisyExpectedImprovementBase):
    r"""Single-outcome Log Noisy Expected Improvement (via fantasies): the
    ``logmeanexp`` over noiseless fantasy models (see
    :class:`NoisyExpectedImprovement`) of the log expected improvement,
    computed in a numerically robust way (Ament et al., 2023)."""

    _log = True

    def __init__(
        self,
        model: Any,
        X_observed: Any,
        num_fantasies: int = 20,
        maximize: bool = True,
        posterior_transform: Any = None,
        *,
        sampler: Any = None,
    ) -> None:
        self._init_noisy(
            model, X_observed, num_fantasies, maximize, sampler, posterior_transform, True
        )


__all__ = [
    "AnalyticAcquisitionFunction",
    "ConstrainedExpectedImprovement",
    "ExpectedImprovement",
    "LogConstrainedExpectedImprovement",
    "LogExpectedImprovement",
    "LogNoisyExpectedImprovement",
    "LogProbabilityOfFeasibility",
    "LogProbabilityOfImprovement",
    "NoisyExpectedImprovement",
    "PosteriorMean",
    "PosteriorStandardDeviation",
    "ProbabilityOfImprovement",
    "ScalarizedPosteriorMean",
    "UpperConfidenceBound",
    "qAnalyticProbabilityOfImprovement",
]

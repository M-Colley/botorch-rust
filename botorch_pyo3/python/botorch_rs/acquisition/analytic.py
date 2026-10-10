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


__all__ = [
    "AnalyticAcquisitionFunction",
    "ConstrainedExpectedImprovement",
    "ExpectedImprovement",
    "LogConstrainedExpectedImprovement",
    "LogExpectedImprovement",
    "LogProbabilityOfImprovement",
    "PosteriorMean",
    "PosteriorStandardDeviation",
    "ProbabilityOfImprovement",
    "UpperConfidenceBound",
]

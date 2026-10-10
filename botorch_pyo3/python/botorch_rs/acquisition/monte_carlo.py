"""Monte-Carlo batch acquisition functions (``botorch.acquisition.monte_carlo``).

The posterior over ``[X; X_pending]`` is sampled with fixed quasi-MC base
samples (``SobolQMCNormalSampler``, 512 samples by default), which makes the
acquisition functions smooth, deterministic sample-average approximations.
Multi-output models are sampled jointly over all outputs (BoTorch's
``sample_shape x q x m`` base samples) and reduced to one value per sample and
point by an MC ``objective``, or scalarized beforehand by a
``posterior_transform``, as in BoTorch. Outcome ``constraints`` (callables
``samples -> slack``, feasible iff ``<= 0``) weight the utilities with a
sigmoid-smoothed feasibility indicator of temperature ``eta``.
"""

from __future__ import annotations

from typing import Any

from botorch_rs import _botorch_rs as _rs
from botorch_rs._arrays import kind_of
from botorch_rs.acquisition.acquisition import (
    AcquisitionFunction,
    pending_arg,
    scalar_arg,
)
from botorch_rs.acquisition.objective import (
    IdentityMCObjective,
    constraints_arg,
    is_constrained_objective,
    objective_arg,
)
from botorch_rs.exceptions.errors import UnsupportedError
from botorch_rs.exceptions.warnings import legacy_ei_numerics_warning
from botorch_rs.sampling.normal import SobolQMCNormalSampler

#: BoTorch's default number of MC samples.
DEFAULT_MC_SAMPLES = 512


def check_constrained_objective(objective: Any, constraints: Any) -> None:
    """BoTorch's error for a ``ConstrainedMCObjective`` together with
    ``constraints``."""
    if constraints is not None and is_constrained_objective(objective):
        raise ValueError(
            "ConstrainedMCObjective as well as constraints passed to constructor."
            "Choose one or the other, preferably the latter."
        )


class MCAcquisitionFunction(AcquisitionFunction):
    """Base class of the Monte-Carlo acquisition functions."""

    def __init__(
        self,
        model: Any,
        sampler: SobolQMCNormalSampler | None = None,
        objective: Any = None,
        posterior_transform: Any = None,
        X_pending: Any = None,
    ) -> None:
        super().__init__(model)
        if objective is None and model.num_outputs != 1:
            if posterior_transform is None:
                raise UnsupportedError(
                    "Must specify an objective or a posterior transform when using "
                    "a multi-output model."
                )
            if not getattr(posterior_transform, "scalarize", False):
                raise UnsupportedError(
                    "If using a multi-output model without an objective, "
                    "posterior_transform must scalarize the output."
                )
        self._objective = objective_arg(objective)
        self.objective = IdentityMCObjective() if objective is None else objective
        self.posterior_transform = posterior_transform
        self._rs_model = self._model_handle(posterior_transform)
        if sampler is None:
            sampler = SobolQMCNormalSampler(sample_shape=(DEFAULT_MC_SAMPLES,))
        elif not isinstance(sampler, SobolQMCNormalSampler):
            raise UnsupportedError("`sampler` must be a botorch_rs SobolQMCNormalSampler.")
        self.sampler = sampler
        self._pending = pending_arg(X_pending, self._rs_model.dim)
        if X_pending is not None:
            self._pending_kind = kind_of(X_pending)
        self._constraints: list[Any] | None = None
        self._eta: Any = None
        self._fat: Any = False

    def _set_constraints(self, constraints: Any, eta: Any, fat: Any = False) -> dict[str, Any]:
        """Stores BoTorch's ``_constraints``, ``_eta`` and ``_fat`` and returns
        the constraint arguments for the extension."""
        cons, etas = constraints_arg(constraints, eta)
        self._constraints = None if constraints is None else list(constraints)
        self._eta = eta
        self._fat = fat
        return {"constraints": cons, "eta": etas}

    def _build(self, name: str, **kwargs: Any) -> None:
        self._rs = _rs.AcquisitionFunction.monte_carlo(
            name,
            self._rs_model,
            self.sampler._rs,
            self._pending,
            objective=self._objective,
            **kwargs,
        )


class qExpectedImprovement(MCAcquisitionFunction):
    r"""MC-based batch Expected Improvement,
    ``qEI(X) = E[max(max_j y_j - best_f, 0)]`` (BoTorch recommends
    :class:`~botorch_rs.acquisition.logei.qLogExpectedImprovement`).

    With outcome ``constraints``, the improvement of every sample and point is
    weighted by the product of ``sigmoid(-constraint(samples) / eta)``.
    """

    def __init__(
        self,
        model: Any,
        best_f: Any,
        sampler: SobolQMCNormalSampler | None = None,
        objective: Any = None,
        posterior_transform: Any = None,
        X_pending: Any = None,
        constraints: Any = None,
        eta: Any = 1e-3,
    ) -> None:
        legacy_ei_numerics_warning(legacy_name=type(self).__name__)
        check_constrained_objective(objective, constraints)
        super().__init__(model, sampler, objective, posterior_transform, X_pending)
        self.best_f = best_f
        kwargs = self._set_constraints(constraints, eta)
        self._build("qExpectedImprovement", best_f=scalar_arg(best_f, "best_f"), **kwargs)


class qProbabilityOfImprovement(MCAcquisitionFunction):
    r"""MC-based batch Probability of Improvement with a sigmoid-smoothed
    indicator of temperature ``tau`` (optionally weighted by the smoothed
    feasibility of outcome ``constraints``)."""

    def __init__(
        self,
        model: Any,
        best_f: Any,
        sampler: SobolQMCNormalSampler | None = None,
        objective: Any = None,
        posterior_transform: Any = None,
        X_pending: Any = None,
        tau: float = 1e-3,
        constraints: Any = None,
        eta: Any = 1e-3,
    ) -> None:
        check_constrained_objective(objective, constraints)
        super().__init__(model, sampler, objective, posterior_transform, X_pending)
        self.best_f = best_f
        self.tau = tau
        kwargs = self._set_constraints(constraints, eta)
        self._build(
            "qProbabilityOfImprovement",
            best_f=scalar_arg(best_f, "best_f"),
            tau=scalar_arg(tau, "tau"),
            **kwargs,
        )


class qSimpleRegret(MCAcquisitionFunction):
    r"""MC-based batch Simple Regret ``E[max_j y_j]``."""

    def __init__(
        self,
        model: Any,
        sampler: SobolQMCNormalSampler | None = None,
        objective: Any = None,
        posterior_transform: Any = None,
        X_pending: Any = None,
    ) -> None:
        super().__init__(model, sampler, objective, posterior_transform, X_pending)
        self._build("qSimpleRegret")


class qUpperConfidenceBound(MCAcquisitionFunction):
    r"""MC-based batch Upper Confidence Bound (Wilson et al., 2017)."""

    def __init__(
        self,
        model: Any,
        beta: float,
        sampler: SobolQMCNormalSampler | None = None,
        objective: Any = None,
        posterior_transform: Any = None,
        X_pending: Any = None,
    ) -> None:
        super().__init__(model, sampler, objective, posterior_transform, X_pending)
        self.beta = beta
        self._build("qUpperConfidenceBound", beta=scalar_arg(beta, "beta"))


__all__ = [
    "MCAcquisitionFunction",
    "qExpectedImprovement",
    "qProbabilityOfImprovement",
    "qSimpleRegret",
    "qUpperConfidenceBound",
]

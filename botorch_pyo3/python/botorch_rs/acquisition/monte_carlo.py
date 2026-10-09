"""Monte-Carlo batch acquisition functions (``botorch.acquisition.monte_carlo``).

The posterior over ``[X; X_pending]`` is sampled with fixed quasi-MC base
samples (``SobolQMCNormalSampler``, 512 samples by default), which makes the
acquisition functions smooth, deterministic sample-average approximations.
"""

from __future__ import annotations

from typing import Any

from botorch_rs import _botorch_rs as _rs
from botorch_rs._arrays import kind_of
from botorch_rs.acquisition.acquisition import (
    AcquisitionFunction,
    check_unsupported,
    pending_arg,
    scalar_arg,
)
from botorch_rs.exceptions.errors import UnsupportedError
from botorch_rs.exceptions.warnings import legacy_ei_numerics_warning
from botorch_rs.sampling.normal import SobolQMCNormalSampler

#: BoTorch's default number of MC samples.
DEFAULT_MC_SAMPLES = 512


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
        check_unsupported(objective=objective, posterior_transform=posterior_transform)
        super().__init__(model)
        if sampler is None:
            sampler = SobolQMCNormalSampler(sample_shape=(DEFAULT_MC_SAMPLES,))
        elif not isinstance(sampler, SobolQMCNormalSampler):
            raise UnsupportedError("`sampler` must be a botorch_rs SobolQMCNormalSampler.")
        self.sampler = sampler
        self._pending = pending_arg(X_pending, model._rs.dim)
        if X_pending is not None:
            self._pending_kind = kind_of(X_pending)

    def _build(self, name: str, **kwargs: Any) -> None:
        self._rs = _rs.AcquisitionFunction.monte_carlo(
            name, self.model._rs, self.sampler._rs, self._pending, **kwargs
        )


class qExpectedImprovement(MCAcquisitionFunction):
    r"""MC-based batch Expected Improvement,
    ``qEI(X) = E[max(max_j y_j - best_f, 0)]`` (BoTorch recommends
    :class:`~botorch_rs.acquisition.logei.qLogExpectedImprovement`)."""

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
        check_unsupported(constraints=constraints)
        super().__init__(model, sampler, objective, posterior_transform, X_pending)
        self.best_f = best_f
        self._build("qExpectedImprovement", best_f=scalar_arg(best_f, "best_f"))


class qProbabilityOfImprovement(MCAcquisitionFunction):
    r"""MC-based batch Probability of Improvement with a sigmoid-smoothed
    indicator of temperature ``tau``."""

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
        check_unsupported(constraints=constraints)
        super().__init__(model, sampler, objective, posterior_transform, X_pending)
        self.best_f = best_f
        self.tau = tau
        self._build(
            "qProbabilityOfImprovement",
            best_f=scalar_arg(best_f, "best_f"),
            tau=scalar_arg(tau, "tau"),
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

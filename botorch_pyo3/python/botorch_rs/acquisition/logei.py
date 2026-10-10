"""Monte-Carlo LogEI acquisition functions (``botorch.acquisition.logei``;
Ament et al., *Unexpected Improvements to Expected Improvement for Bayesian
Optimization*, NeurIPS 2023)."""

from __future__ import annotations

from typing import Any

from botorch_rs import _botorch_rs as _rs
from botorch_rs._arrays import from_pair, kind_of, to_flat
from botorch_rs.acquisition.acquisition import check_unsupported, pending_arg, scalar_arg
from botorch_rs.acquisition.monte_carlo import MCAcquisitionFunction
from botorch_rs.exceptions.errors import BotorchTensorDimensionError
from botorch_rs.sampling.normal import SobolQMCNormalSampler

#: Default temperature of the smooth ReLU.
TAU_RELU = 1e-6
#: Default temperature of the smooth maximum over the ``q`` points.
TAU_MAX = 1e-2

# Number of samples used by ``prune_inferior_points`` (BoTorch's default).
_PRUNE_SAMPLES = 2048
_PRUNE_SEED_SALT = 0x5DEECE66D


class qLogExpectedImprovement(MCAcquisitionFunction):
    r"""MC-based batch Log Expected Improvement: a smooth, numerically robust
    approximation of ``log E[max(max_j y_j - best_f, 0)]`` (fat-tailed
    ``fatplus`` / ``fatmax`` by default)."""

    _log = True

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
        fat: bool = True,
        tau_max: float = TAU_MAX,
        tau_relu: float = TAU_RELU,
    ) -> None:
        check_unsupported(constraints=constraints)
        super().__init__(model, sampler, objective, posterior_transform, X_pending)
        self.best_f = best_f
        self.fat = fat
        self.tau_max = tau_max
        self.tau_relu = tau_relu
        self._build(
            "qLogExpectedImprovement",
            best_f=scalar_arg(best_f, "best_f"),
            fat=bool(fat),
            tau_max=scalar_arg(tau_max, "tau_max"),
            tau_relu=scalar_arg(tau_relu, "tau_relu"),
        )


class qLogNoisyExpectedImprovement(MCAcquisitionFunction):
    r"""MC-based batch Log Noisy Expected Improvement: a smooth approximation of
    ``log E[max(max_j y_j - max_i y_i^baseline, 0)]`` over joint posterior
    samples of the candidates and the (pruned) baseline points.

    botorch_rs always uses the plain joint computation (BoTorch's
    ``cache_root=False``), which equals BoTorch's default ``cache_root=True``
    in distribution; the ``cache_root`` argument is accepted and ignored. The
    baseline is pruned with ``prune_inferior_points`` (2048 Sobol samples)
    seeded deterministically from the sampler's seed.
    """

    _log = True

    def __init__(
        self,
        model: Any,
        X_baseline: Any,
        sampler: SobolQMCNormalSampler | None = None,
        objective: Any = None,
        posterior_transform: Any = None,
        X_pending: Any = None,
        constraints: Any = None,
        eta: Any = 1e-3,
        fat: bool = True,
        prune_baseline: bool = True,
        cache_root: bool | None = None,
        tau_max: float = TAU_MAX,
        tau_relu: float = TAU_RELU,
        marginalize_dim: int | None = None,
        incremental: bool = True,
    ) -> None:
        check_unsupported(constraints=constraints, marginalize_dim=marginalize_dim)
        super().__init__(model, sampler, objective, posterior_transform, None)
        baseline = to_flat(X_baseline, "X_baseline")
        if len(baseline[1]) != 2:
            raise BotorchTensorDimensionError(
                f"Expected X_baseline to be an `n x d` tensor, got shape {baseline[1]}."
            )
        self._baseline_kind = kind_of(X_baseline)
        pending = None
        if X_pending is not None:
            pending = pending_arg(X_pending, self._rs_model.dim)
            self._pending_kind = kind_of(X_pending)
        self.fat = fat
        self.prune_baseline = prune_baseline
        self.cache_root = cache_root
        self.tau_max = tau_max
        self.tau_relu = tau_relu
        self.incremental = incremental
        self._rs = _rs.AcquisitionFunction.q_log_noisy_expected_improvement(
            self._rs_model,
            baseline,
            self.sampler._rs,
            pending,
            fat=bool(fat),
            prune_baseline=bool(prune_baseline),
            prune_num_samples=_PRUNE_SAMPLES,
            prune_max_frac=1.0,
            prune_seed=(self.sampler.seed ^ _PRUNE_SEED_SALT) & 0xFFFFFFFFFFFFFFFF,
            tau_max=scalar_arg(tau_max, "tau_max"),
            tau_relu=scalar_arg(tau_relu, "tau_relu"),
            incremental=bool(incremental),
        )

    @property
    def X_baseline(self) -> Any:
        """The incumbent points: the (pruned) baseline and, in incremental mode,
        the pending points."""
        return from_pair(self._rs.full_x_baseline(), self._baseline_kind)

    @property
    def _X_baseline(self) -> Any:
        """The (pruned) baseline points."""
        return from_pair(self._rs.x_baseline(), self._baseline_kind)


__all__ = ["TAU_MAX", "TAU_RELU", "qLogExpectedImprovement", "qLogNoisyExpectedImprovement"]

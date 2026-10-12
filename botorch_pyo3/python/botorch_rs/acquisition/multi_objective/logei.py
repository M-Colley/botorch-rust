"""Log-space hypervolume improvement (``botorch.acquisition.multi_objective.logei``;
Ament et al., *Unexpected Improvements to Expected Improvement for Bayesian
Optimization*, NeurIPS 2023): qLogEHVI and qLogNEHVI, BoTorch's recommended
multi-objective acquisition function.

The values are BoTorch's ``_compute_log_qehvi`` (the formulas of its
pure-Python path, computed in Rust with exact gradients). BoTorch's optional
fused C++ kernel (``_FusedLogAreas``) evaluates the same formulas with
reassociated arithmetic and agrees with them to rounding; botorch_rs has no
such switch.
"""

from __future__ import annotations

from typing import Any

from botorch_rs import _botorch_rs as _rs
from botorch_rs.acquisition.acquisition import check_unsupported
from botorch_rs.acquisition.logei import TAU_MAX, TAU_RELU
from botorch_rs.acquisition.multi_objective.base import (
    MultiObjectiveMCAcquisitionFunction,
    fat_bool,
)
from botorch_rs.acquisition.multi_objective.monte_carlo import _EHVIBase, _NEHVIBase
from botorch_rs.sampling.normal import SobolQMCNormalSampler
from botorch_rs.utils.multi_objective.hypervolume import SubsetIndexCachingMixin


class qLogExpectedHypervolumeImprovement(_EHVIBase):
    r"""Parallel Log Expected Hypervolume Improvement supporting ``m >= 2``
    outcomes: a smooth, numerically robust approximation of the logarithm of
    qEHVI (``log_fatplus`` / ``fatmin`` with ``fat=True``, else
    ``log_softplus`` / ``smooth_amin``).

    Example:
        >>> model = SingleTaskGP(train_X, train_Y)
        >>> ref_point = [0.0, 0.0]
        >>> acq = qLogExpectedHypervolumeImprovement(model, ref_point, partitioning)
        >>> value = acq(test_X)
    """

    _log = True

    def __init__(
        self,
        model: Any,
        ref_point: Any,
        partitioning: Any,
        sampler: SobolQMCNormalSampler | None = None,
        objective: Any = None,
        constraints: Any = None,
        X_pending: Any = None,
        eta: Any = 1e-2,
        fat: bool = True,
        tau_relu: float = TAU_RELU,
        tau_max: float = TAU_MAX,
    ) -> None:
        fat_ = fat_bool(fat)
        ref = self._setup_ehvi(
            model,
            ref_point,
            partitioning,
            sampler,
            objective,
            constraints,
            X_pending,
            eta,
            fat,
            "The dimensionality of the reference point must match the number of "
            "outcomes. Got ref_point with {} elements, but expected {}.",
        )
        self.tau_relu = tau_relu
        self.tau_max = tau_max
        self._rs = _rs.q_expected_hypervolume_improvement(
            True,
            self._rs_model,
            ref,
            self._cells,
            self.sampler._rs,
            self._pending,
            objective=self._mo_objective,
            fat=fat_,
            tau_relu=float(tau_relu),
            tau_max=float(tau_max),
            **self._constraint_args,
        )


class qLogNoisyExpectedHypervolumeImprovement(_NEHVIBase, qLogExpectedHypervolumeImprovement):
    r"""q-Log Noisy Expected Hypervolume Improvement supporting ``m >= 2``
    outcomes: the log-space version of qNEHVI (BoTorch's recommended
    multi-objective acquisition function).

    See :class:`~botorch_rs.acquisition.multi_objective.monte_carlo.qNoisyExpectedHypervolumeImprovement`
    for the baseline decompositions, ``cache_root``, pending points and
    pruning. For non-incremental NEHVI, the improvement of the cached pending
    points is added in log space.

    Example:
        >>> model = SingleTaskGP(train_X, train_Y)
        >>> ref_point = [0.0, 0.0]
        >>> acq = qLogNoisyExpectedHypervolumeImprovement(model, ref_point, train_X)
        >>> value = acq(test_X)
    """

    _log = True

    def __init__(
        self,
        model: Any,
        ref_point: Any,
        X_baseline: Any,
        sampler: SobolQMCNormalSampler | None = None,
        objective: Any = None,
        constraints: Any = None,
        X_pending: Any = None,
        eta: Any = 1e-3,
        prune_baseline: bool = False,
        alpha: float = 0.0,
        cache_pending: bool = True,
        max_iep: int = 0,
        incremental_nehvi: bool = True,
        cache_root: bool | None = None,
        tau_relu: float = TAU_RELU,
        tau_max: float = 1e-3,
        fat: bool = True,
        marginalize_dim: int | None = None,
    ) -> None:
        check_unsupported(marginalize_dim=marginalize_dim)
        fat_ = fat_bool(fat)
        MultiObjectiveMCAcquisitionFunction.__init__(
            self, model, sampler, objective, constraints, eta
        )
        SubsetIndexCachingMixin.__init__(self)
        self.tau_relu = tau_relu
        self.tau_max = tau_max
        self._setup_nehvi(
            True,
            model,
            ref_point,
            X_baseline,
            X_pending,
            fat_,
            prune_baseline,
            alpha,
            cache_pending,
            max_iep,
            incremental_nehvi,
            cache_root,
            tau_relu,
            tau_max,
        )
        self.fat = fat


__all__ = ["qLogExpectedHypervolumeImprovement", "qLogNoisyExpectedHypervolumeImprovement"]

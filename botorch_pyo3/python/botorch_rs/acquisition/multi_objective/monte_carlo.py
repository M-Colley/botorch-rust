"""Monte-Carlo hypervolume improvement
(``botorch.acquisition.multi_objective.monte_carlo``): qEHVI (Daulton et al.,
NeurIPS 2020) and qNEHVI (Daulton et al., NeurIPS 2021). BoTorch recommends
their log versions (:mod:`botorch_rs.acquisition.multi_objective.logei`)."""

from __future__ import annotations

from typing import Any

from botorch_rs import _botorch_rs as _rs
from botorch_rs._arrays import first_kind, from_flat, kind_of, to_flat
from botorch_rs.acquisition.acquisition import check_unsupported, pending_arg
from botorch_rs.acquisition.multi_objective.base import (
    MultiObjectiveMCAcquisitionFunction,
    fat_bool,
    partitioning_cells,
    ref_point_arg,
)
from botorch_rs.exceptions.errors import UnsupportedError
from botorch_rs.exceptions.warnings import legacy_ei_numerics_warning
from botorch_rs.sampling.normal import SobolQMCNormalSampler
from botorch_rs.utils.multi_objective.hypervolume import (
    NoisyExpectedHypervolumeMixin,
    SubsetIndexCachingMixin,
)

# Salt of the seed of the baseline pruning, derived from the sampler's seed.
_PRUNE_SEED_SALT = 0x6E656876695F7072
_PRUNE_SAMPLES = 2048


class _EHVIBase(MultiObjectiveMCAcquisitionFunction, SubsetIndexCachingMixin):
    """Shared construction of qEHVI and qLogEHVI."""

    _log_name = False

    def _setup_ehvi(
        self,
        model: Any,
        ref_point: Any,
        partitioning: Any,
        sampler: SobolQMCNormalSampler | None,
        objective: Any,
        constraints: Any,
        X_pending: Any,
        eta: Any,
        fat: Any,
        message: str,
    ) -> list[float]:
        ref = ref_point_arg(ref_point)
        num_outcomes = int(partitioning.num_outcomes)
        if len(ref) != num_outcomes:
            raise ValueError(message.format(len(ref), num_outcomes))
        super().__init__(model, sampler, objective, constraints, eta, X_pending)
        SubsetIndexCachingMixin.__init__(self)
        cells, shape = partitioning_cells(partitioning)
        self._cells = (cells, shape)
        try:
            pareto = partitioning.pareto_Y
        except Exception:  # noqa: BLE001 - no outcomes yet
            pareto = None
        self._kind = first_kind(pareto, ref_point)
        self.ref_point = from_flat(ref, (len(ref),), self._kind)
        self.fat = fat
        return ref

    def _cell_bounds(self, which: int) -> Any:
        data, shape = self._cells
        half = len(data) // 2
        return from_flat(data[which * half : (which + 1) * half], shape[1:], self._kind)

    @property
    def cell_lower_bounds(self) -> Any:
        """The lower vertices of the hypercells (``num_cells x m``)."""
        return self._cell_bounds(0)

    @property
    def cell_upper_bounds(self) -> Any:
        """The upper vertices of the hypercells (``num_cells x m``)."""
        return self._cell_bounds(1)


class qExpectedHypervolumeImprovement(_EHVIBase):
    r"""q-Expected Hypervolume Improvement supporting ``m >= 2`` outcomes
    (Daulton et al., 2020): the expected (feasibility-weighted) hypervolume
    improvement of the candidates over a box decomposition of the space not
    dominated by the current Pareto front, computed with the
    inclusion-exclusion principle over the subsets of the ``q`` candidates.

    Example:
        >>> model = SingleTaskGP(train_X, train_Y)
        >>> ref_point = [0.0, 0.0]
        >>> qEHVI = qExpectedHypervolumeImprovement(model, ref_point, partitioning)
        >>> qehvi = qEHVI(test_X)
    """

    def __init__(
        self,
        model: Any,
        ref_point: Any,
        partitioning: Any,
        sampler: SobolQMCNormalSampler | None = None,
        objective: Any = None,
        constraints: Any = None,
        X_pending: Any = None,
        eta: Any = 1e-3,
        fat: bool = False,
    ) -> None:
        legacy_ei_numerics_warning(legacy_name=type(self).__name__)
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
            "The length of the reference point must match the number of outcomes. "
            "Got ref_point with {} elements, but expected {}.",
        )
        self._rs = _rs.q_expected_hypervolume_improvement(
            False,
            self._rs_model,
            ref,
            self._cells,
            self.sampler._rs,
            self._pending,
            objective=self._mo_objective,
            fat=fat_bool(fat),
            **self._constraint_args,
        )


def _cache_root_arg(cache_root: Any) -> bool:
    """BoTorch's ``cache_root`` (``None``: supported, i.e. ``True`` for the
    botorch_rs models)."""
    return True if cache_root is None else bool(cache_root)


class _NEHVIBase(NoisyExpectedHypervolumeMixin):
    """Shared construction of qNEHVI and qLogNEHVI."""

    _rs: Any
    _rs_model: Any
    _mo_objective: tuple[Any, ...] | None
    _constraint_args: dict[str, Any]
    _pending_kind: Any
    sampler: SobolQMCNormalSampler

    def _setup_nehvi(
        self,
        log: bool,
        model: Any,
        ref_point: Any,
        X_baseline: Any,
        X_pending: Any,
        fat: Any,
        prune_baseline: bool,
        alpha: float,
        cache_pending: bool,
        max_iep: int,
        incremental_nehvi: bool,
        cache_root: Any,
        tau_relu: float,
        tau_max: float,
    ) -> None:
        ref = ref_point_arg(ref_point)
        if len(ref) < 2:
            raise ValueError(
                "NoisyExpectedHypervolumeMixin supports m>=2 outcomes "
                f"but ref_point has length {len(ref)}, which is smaller than 2."
            )
        baseline = to_flat(X_baseline, "X_baseline")
        if len(baseline[1]) > 2:
            raise UnsupportedError(
                f"NoisyExpectedHypervolumeMixin does not support batched "
                f"X_baseline. Expected 2 dims, got {len(baseline[1])}."
            )
        self._baseline_kind = kind_of(X_baseline)
        pending = pending_arg(X_pending, self._rs_model.dim)
        if X_pending is not None:
            self._pending_kind = kind_of(X_pending)
        self.ref_point = from_flat(ref, (len(ref),), self._baseline_kind)
        self.alpha = alpha
        self.cache_pending = cache_pending
        self._max_iep = max_iep
        self.incremental_nehvi = incremental_nehvi
        self._cache_root = _cache_root_arg(cache_root)
        self.fat = fat
        self._rs, self._rs_nehvi = _rs.q_noisy_expected_hypervolume_improvement(
            log,
            self._rs_model,
            ref,
            baseline,
            self.sampler._rs,
            pending,
            objective=self._mo_objective,
            fat=fat_bool(fat),
            prune_baseline=bool(prune_baseline),
            prune_num_samples=_PRUNE_SAMPLES,
            prune_seed=(self.sampler.seed ^ _PRUNE_SEED_SALT) & 0xFFFFFFFFFFFFFFFF,
            alpha=float(alpha),
            cache_pending=bool(cache_pending),
            max_iep=int(max_iep),
            incremental_nehvi=bool(incremental_nehvi),
            cache_root=self._cache_root,
            tau_relu=float(tau_relu),
            tau_max=float(tau_max),
            **self._constraint_args,
        )


class qNoisyExpectedHypervolumeImprovement(_NEHVIBase, qExpectedHypervolumeImprovement):
    r"""q-Noisy Expected Hypervolume Improvement supporting ``m >= 2``
    outcomes (Daulton et al., 2021): the expected hypervolume improvement of
    the candidates over the Pareto front of the (noisy) baseline under each
    joint posterior sample, with one box decomposition per MC sample.

    With ``cache_root`` (BoTorch's default), new points are sampled from
    low-rank updates of the cached root of the baseline covariance;
    ``cache_root=False`` samples the joint posterior. Pending points are
    cached in the decompositions as in BoTorch (``cache_pending``,
    ``max_iep``); the base samples of previously decomposed points are kept
    when pending points are appended (meta-pytorch/botorch#3392). With
    ``prune_baseline``, the baseline is pruned with 2048 samples seeded from
    the sampler's seed (BoTorch draws a random seed).

    Example:
        >>> model = SingleTaskGP(train_X, train_Y)
        >>> ref_point = [0.0, 0.0]
        >>> qNEHVI = qNoisyExpectedHypervolumeImprovement(model, ref_point, train_X)
        >>> qnehvi = qNEHVI(test_X)
    """

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
        fat: bool = False,
        prune_baseline: bool = False,
        alpha: float = 0.0,
        cache_pending: bool = True,
        max_iep: int = 0,
        incremental_nehvi: bool = True,
        cache_root: bool | None = None,
        marginalize_dim: int | None = None,
    ) -> None:
        legacy_ei_numerics_warning(legacy_name=type(self).__name__)
        check_unsupported(marginalize_dim=marginalize_dim)
        MultiObjectiveMCAcquisitionFunction.__init__(
            self, model, sampler, objective, constraints, eta
        )
        SubsetIndexCachingMixin.__init__(self)
        self._setup_nehvi(
            False,
            model,
            ref_point,
            X_baseline,
            X_pending,
            fat,
            prune_baseline,
            alpha,
            cache_pending,
            max_iep,
            incremental_nehvi,
            cache_root,
            1e-6,
            1e-3,
        )


__all__ = ["qExpectedHypervolumeImprovement", "qNoisyExpectedHypervolumeImprovement"]

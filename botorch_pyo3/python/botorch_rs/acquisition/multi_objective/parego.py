"""qLogNParEGO (``botorch.acquisition.multi_objective.parego``): qLogNEI of
an augmented Chebyshev scalarization of the objectives (Knowles, 2006;
Daulton et al., 2020; Ament et al., 2023)."""

from __future__ import annotations

from typing import Any

from botorch_rs import _botorch_rs as _rs
from botorch_rs._arrays import from_flat, from_pair, kind_of, to_flat
from botorch_rs.acquisition.acquisition import pending_arg
from botorch_rs.acquisition.logei import TAU_MAX, TAU_RELU, qLogNoisyExpectedImprovement
from botorch_rs.acquisition.multi_objective.base import (
    MultiObjectiveMCAcquisitionFunction,
    fat_bool,
)
from botorch_rs.acquisition.objective import GenericMCObjective
from botorch_rs.sampling.normal import SobolQMCNormalSampler
from botorch_rs.utils.multi_objective.scalarization import ChebyshevScalarization

# Salts of the seeds of the random weights and of the baseline pruning,
# derived from the sampler's seed.
_WEIGHTS_SEED_SALT = 0x7061726567305F77
_PRUNE_SEED_SALT = 0x5DEECE66D
_PRUNE_SAMPLES = 2048


class qLogNParEGO(qLogNoisyExpectedImprovement, MultiObjectiveMCAcquisitionFunction):
    r"""q-LogNParEGO supporting ``m >= 2`` outcomes: qLogNEI of the augmented
    Chebyshev scalarization of the objectives (``alpha = 0.05``), normalized
    by the objective values of the posterior mean at the baseline. The
    scalarization is evaluated natively.

    Explicit ``scalarization_weights`` reproduce BoTorch exactly. Without
    them, the weights are drawn uniformly from the simplex with botorch_rs's
    generator, seeded from the sampler's seed (BoTorch uses torch's global
    generator). Like qLogNEI, the computation is BoTorch's
    ``cache_root=False``; ``cache_root`` is accepted and ignored.

    This implementation assumes maximization of all objectives; use an
    ``objective`` or negative ``scalarization_weights`` to minimize outputs.
    """

    _log = True

    def __init__(
        self,
        model: Any,
        X_baseline: Any,
        scalarization_weights: Any = None,
        sampler: SobolQMCNormalSampler | None = None,
        objective: Any = None,
        constraints: Any = None,
        X_pending: Any = None,
        eta: Any = 1e-3,
        fat: bool = True,
        prune_baseline: bool = False,
        cache_root: bool | None = None,
        tau_relu: float = TAU_RELU,
        tau_max: float = TAU_MAX,
        incremental: bool = True,
    ) -> None:
        MultiObjectiveMCAcquisitionFunction.__init__(
            self, model, sampler, objective, constraints, eta
        )
        baseline = to_flat(X_baseline, "X_baseline")
        self._baseline_kind = kind_of(X_baseline)
        pending = pending_arg(X_pending, self._rs_model.dim)
        if X_pending is not None:
            self._pending_kind = kind_of(X_pending)
        weights = None
        if scalarization_weights is not None:
            weights = [float(w) for w in to_flat(scalarization_weights, "weights")[0]]
        self.fat = fat
        self.prune_baseline = prune_baseline
        self.cache_root = cache_root
        self.tau_max = tau_max
        self.tau_relu = tau_relu
        self.incremental = incremental
        self._warned_infeasible = False
        seed = self.sampler.seed
        self._rs, w, y_baseline = _rs.q_log_nparego(
            self._rs_model,
            baseline,
            scalarization_weights=weights,
            weights_seed=(seed ^ _WEIGHTS_SEED_SALT) & 0xFFFFFFFFFFFFFFFF,
            sampler=self.sampler._rs,
            objective=self._mo_objective,
            x_pending=pending,
            fat=fat_bool(fat),
            prune_baseline=bool(prune_baseline),
            prune_num_samples=_PRUNE_SAMPLES,
            prune_seed=(seed ^ _PRUNE_SEED_SALT) & 0xFFFFFFFFFFFFFFFF,
            tau_relu=float(tau_relu),
            tau_max=float(tau_max),
            incremental=bool(incremental),
            **self._constraint_args,
        )
        kind = self._baseline_kind
        self._org_objective = self.objective
        self.scalarization_weights = (
            scalarization_weights
            if scalarization_weights is not None
            else from_flat(w, (len(w),), kind)
        )
        self.Y_baseline = from_pair(y_baseline, kind)
        self.chebyshev_scalarization = ChebyshevScalarization(
            from_flat(w, (len(w),), kind), self.Y_baseline
        )
        org, cheb = self._org_objective, self.chebyshev_scalarization
        self.objective = GenericMCObjective(lambda samples, X=None: cheb(org(samples, X=X), X=X))


__all__ = ["qLogNParEGO"]

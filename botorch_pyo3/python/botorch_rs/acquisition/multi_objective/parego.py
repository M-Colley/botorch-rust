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
from botorch_rs.exceptions.errors import UnsupportedError
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

    On an ensemble model (e.g. a ``ModelListGP`` of fully Bayesian GPs), the
    scalarization of all members is normalized by the objective values of
    the members' posterior means at the baseline averaged over the members
    (BoTorch's mean over ``MCMC_DIM``), and the baseline is pruned under it
    (see ``_prepare_ensemble``).
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
        *,
        _y_baseline: Any = None,
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
            y_baseline=None if _y_baseline is None else to_flat(_y_baseline, "Y_baseline"),
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

    @classmethod
    def _prepare_ensemble(
        cls, init: Any, model: Any, members: list[Any], arguments: dict[str, Any], model_name: str
    ) -> None:
        """The arguments of the members of an ensemble (see
        ``_init_on_ensemble``): BoTorch's ``Y_baseline`` of an ensemble is the
        mean over the members (``MCMC_DIM``) of the objective values of their
        posterior means, which normalizes one scalarization for all members;
        the baseline is pruned with the samples of this scalarization
        averaged over the members (``prune_inferior_points`` with
        ``marginalize_dim``)."""
        from botorch_rs.acquisition.multi_objective.utils import _prune_ensemble_scalarized

        probe = {**arguments, "prune_baseline": False}
        weights = arguments.get("scalarization_weights")
        ys = []
        for member in members:
            tmp = object.__new__(cls)
            init(tmp, **{**probe, model_name: member})
            ys.append(to_flat(tmp.Y_baseline, "Y_baseline"))
            if weights is None:
                weights = tmp.scalarization_weights
        shape = tuple(ys[0][1])
        y = [sum(v[0][i] for v in ys) / len(ys) for i in range(len(ys[0][0]))]
        kind = kind_of(arguments["X_baseline"])
        y_baseline = from_flat(y, shape, kind)
        arguments["scalarization_weights"] = weights
        arguments["_y_baseline"] = y_baseline
        if arguments.get("prune_baseline", False):
            if arguments.get("objective") is not None or arguments.get("constraints"):
                raise UnsupportedError(
                    "Pruning the baseline of an ensemble model with an objective or "
                    "constraints is not supported by botorch_rs; pass prune_baseline=False."
                )
            cheb = ChebyshevScalarization(weights, y_baseline)
            arguments["X_baseline"] = _prune_ensemble_scalarized(
                model,
                arguments["X_baseline"],
                lambda values, shape: cheb._rs.evaluate((values, shape))[0],
                members=members,
            )
            arguments["prune_baseline"] = False


__all__ = ["qLogNParEGO"]

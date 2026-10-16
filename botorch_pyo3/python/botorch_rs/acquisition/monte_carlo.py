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

import math
import warnings
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
from botorch_rs.sampling.stochastic_samplers import mc_sampler_arg

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
        self.sampler = mc_sampler_arg(sampler, self._rs_model, (DEFAULT_MC_SAMPLES,))
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


class qLowerConfidenceBound(qUpperConfidenceBound):
    r"""MC-based batched lower confidence bound.

    This acquisition function is useful for confident/risk-averse decision
    making. This acquisition function is intended to be maximized as with
    qUpperConfidenceBound, but the qLowerConfidenceBound will be pessimistic
    in the face of uncertainty and lead to conservative candidates.
    """

    def __init__(
        self,
        model: Any,
        beta: float,
        sampler: SobolQMCNormalSampler | None = None,
        objective: Any = None,
        posterior_transform: Any = None,
        X_pending: Any = None,
    ) -> None:
        MCAcquisitionFunction.__init__(
            self, model, sampler, objective, posterior_transform, X_pending
        )
        self.beta = beta
        self._build("qLowerConfidenceBound", beta=scalar_arg(beta, "beta"))


class qPosteriorStandardDeviation(MCAcquisitionFunction):
    r"""MC-based batch Posterior Standard Deviation, an acquisition function
    for pure exploration: ``E[max_j sqrt(pi / 2) |y_j - m_j|]`` with the
    sample means ``m_j`` (optionally weighted by the smoothed feasibility of
    outcome ``constraints``)."""

    def __init__(
        self,
        model: Any,
        sampler: SobolQMCNormalSampler | None = None,
        objective: Any = None,
        posterior_transform: Any = None,
        X_pending: Any = None,
        constraints: Any = None,
        eta: Any = 1e-3,
    ) -> None:
        check_constrained_objective(objective, constraints)
        super().__init__(model, sampler, objective, posterior_transform, X_pending)
        self._scale = math.sqrt(math.pi / 2)
        kwargs = self._set_constraints(constraints, eta)
        self._build("qPosteriorStandardDeviation", **kwargs)


#: The number of samples and the seed salt of the pruning of the baseline
#: (seeded from the sampler; BoTorch's pruning sampler is unseeded).
_PRUNE_SAMPLES = 2048
_PRUNE_SEED_SALT = 0x5DEECE66D

_CACHE_ROOT_MESSAGE = (
    "`cache_root` is only supported for GPyTorchModels that are not MultiTask models and "
    "don't produce a TransformedPosterior. Got a model of type {}. Setting `cache_root = False`."
)


def supports_cache_root(model: Any) -> bool:
    r"""Checks if a model supports the cache_root functionality (BoTorch's
    ``supports_cache_root``): the models of a ``ModelListGP`` must, and
    multi-task (``LCEMGP``) and deterministic models do not."""
    from botorch_rs.models.contextual_multioutput import LCEMGP
    from botorch_rs.models.gp_regression import SingleTaskGP
    from botorch_rs.models.model_list_gp_regression import ModelListGP

    if isinstance(model, ModelListGP):
        return all(supports_cache_root(m) for m in model.models)
    if isinstance(model, LCEMGP):
        return False
    return isinstance(model, SingleTaskGP)


class qNoisyExpectedImprovement(MCAcquisitionFunction):
    r"""MC-based batch Noisy Expected Improvement.

    This function does not assume a ``best_f`` is known (which would require
    noiseless observations). Instead, it uses samples from the joint
    posterior over the ``q`` test points and previously observed points. The
    improvement over previously observed points is computed for each sample
    and averaged.

    ``qNEI(X) = E(max(max Y - max Y_baseline, 0))``, where
    ``(Y, Y_baseline) ~ f((X, X_baseline)), X = (x_1,...,x_q)``

    With ``cache_root`` (the default for GP models), the baseline is sampled
    once with the base samples of the sampler for its points, and the
    candidates by a low-rank update of the cached root of the baseline
    covariance with base samples whose baseline rows are the baseline's, as
    in BoTorch. The baseline is pruned with ``prune_inferior_points`` (2048
    Sobol samples seeded from the sampler's seed). With outcome
    ``constraints``, the incumbent of every sample is the best feasible
    baseline objective; if some sample has no feasible baseline point, a
    lower bound of the objective is estimated at 32 random points of the
    (10% expanded) bounding box of the baseline (seeded from the sampler;
    explicit points with the keyword-only ``lower_bound_points``).

    NOTE: It is strongly recommended to use ``qLogNoisyExpectedImprovement``
    instead of regular ``qNEI``.
    """

    def __init__(
        self,
        model: Any,
        X_baseline: Any,
        sampler: SobolQMCNormalSampler | None = None,
        objective: Any = None,
        posterior_transform: Any = None,
        X_pending: Any = None,
        prune_baseline: bool = True,
        cache_root: bool | None = None,
        constraints: Any = None,
        eta: Any = 1e-3,
        marginalize_dim: int | None = None,
        *,
        lower_bound_points: Any = None,
    ) -> None:
        from botorch_rs._arrays import to_flat
        from botorch_rs.acquisition.acquisition import check_unsupported
        from botorch_rs.exceptions.errors import BotorchTensorDimensionError

        legacy_ei_numerics_warning(legacy_name=type(self).__name__)
        check_unsupported(marginalize_dim=marginalize_dim)
        check_constrained_objective(objective, constraints)
        super().__init__(model, sampler, objective, posterior_transform, X_pending)
        if cache_root is None:
            cache_root = supports_cache_root(model)
        elif cache_root and not supports_cache_root(model):
            warnings.warn(_CACHE_ROOT_MESSAGE.format(type(model)), RuntimeWarning, stacklevel=2)
            cache_root = False
        self._cache_root = bool(cache_root)
        baseline = to_flat(X_baseline, "X_baseline")
        if len(baseline[1]) != 2:
            raise BotorchTensorDimensionError(
                f"Expected X_baseline to be an `n x d` tensor, got shape {baseline[1]}."
            )
        self._baseline_kind = kind_of(X_baseline)
        self.prune_baseline = prune_baseline
        self._warned_infeasible = False
        kwargs = self._set_constraints(constraints, eta)
        lb = (
            None
            if lower_bound_points is None
            else to_flat(lower_bound_points, "lower_bound_points")
        )
        self._rs, self._state = _rs.q_noisy_expected_improvement(
            self._rs_model,
            baseline,
            self.sampler._rs,
            self._pending,
            objective=self._objective,
            prune_baseline=bool(prune_baseline),
            prune_num_samples=_PRUNE_SAMPLES,
            prune_max_frac=1.0,
            prune_seed=(self.sampler.seed ^ _PRUNE_SEED_SALT) & 0xFFFFFFFFFFFFFFFF,
            cache_root=self._cache_root,
            lower_bound_points=lb,
            **kwargs,
        )

    def _after_evaluation(self) -> None:
        # BoTorch warns when some sample has no feasible baseline point.
        if not self._warned_infeasible and self._state.infeasible_lower_bound() is not None:
            from botorch_rs.exceptions.warnings import BotorchWarning

            self._warned_infeasible = True
            warnings.warn(
                "When all training points are infeasible, it is better to use "
                "q(Log)ProbabilityOfFeasibility.",
                BotorchWarning,
                stacklevel=3,
            )

    @property
    def X_baseline(self) -> Any:
        """The (pruned) baseline points (``n x d``)."""
        from botorch_rs._arrays import from_pair

        return from_pair(self._state.x_baseline(), self._baseline_kind)

    @property
    def lower_bound_points(self) -> Any:
        """The points at which the objective's lower bound for infeasible
        baseline points is estimated (``k x d``)."""
        from botorch_rs._arrays import from_pair

        return from_pair(self._state.lower_bound_points(), self._baseline_kind)


__all__ = [
    "MCAcquisitionFunction",
    "qExpectedImprovement",
    "qLowerConfidenceBound",
    "qNoisyExpectedImprovement",
    "qPosteriorStandardDeviation",
    "qProbabilityOfImprovement",
    "qSimpleRegret",
    "qUpperConfidenceBound",
    "supports_cache_root",
]

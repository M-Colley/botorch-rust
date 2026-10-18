"""Monte-Carlo LogEI acquisition functions (``botorch.acquisition.logei``;
Ament et al., *Unexpected Improvements to Expected Improvement for Bayesian
Optimization*, NeurIPS 2023)."""

from __future__ import annotations

import warnings
from typing import Any

from botorch_rs import _botorch_rs as _rs
from botorch_rs._arrays import from_pair, kind_of, to_flat
from botorch_rs.acquisition.acquisition import check_unsupported, pending_arg, scalar_arg
from botorch_rs.acquisition.monte_carlo import (
    SampleReducingMCAcquisitionFunction,
    check_constrained_objective,
)
from botorch_rs.acquisition.objective import IdentityMCObjective, fat_arg, is_constrained_objective
from botorch_rs.exceptions.errors import BotorchError, BotorchTensorDimensionError
from botorch_rs.exceptions.warnings import BotorchWarning
from botorch_rs.sampling.normal import SobolQMCNormalSampler

#: Default temperature of the smooth ReLU.
TAU_RELU = 1e-6
#: Default temperature of the smooth maximum over the ``q`` points.
TAU_MAX = 1e-2

# Number of samples used by ``prune_inferior_points`` (BoTorch's default).
_PRUNE_SAMPLES = 2048
_PRUNE_SEED_SALT = 0x5DEECE66D


def _check_log_objective(objective: Any) -> None:
    if is_constrained_objective(objective):
        raise BotorchError(
            "Log-Improvement should not be used with `ConstrainedMCObjective`."
            "Please pass the `constraints` directly to the constructor of the "
            "acquisition function."
        )


def _logmeanexp(X: Any, dim: Any) -> Any:
    """BoTorch's ``logmeanexp`` (the sample reduction of the log-improvement
    acquisition functions)."""
    from botorch_rs.acquisition._torch_reductions import logmeanexp

    return logmeanexp(X, dim=dim)


class _SmoothMax:
    """BoTorch's ``partial(fatmax if fat else smooth_amax, tau=tau_max)``
    (the q-reduction of the log-improvement acquisition functions)."""

    def __init__(self, fat: bool, tau: float) -> None:
        self.fat = fat
        self.tau = tau

    def __call__(self, X: Any, dim: Any) -> Any:
        from botorch_rs.acquisition._torch_reductions import fatmax, smooth_amax

        fn = fatmax if self.fat else smooth_amax
        return fn(X, dim=dim, tau=self.tau)


class LogImprovementMCAcquisitionFunction(SampleReducingMCAcquisitionFunction):
    r"""
    Abstract base class for Monte-Carlo-based batch LogEI acquisition functions.

    Python subclasses implement ``_sample_forward`` (the log-utilities of
    every sample and point) and are evaluated through torch like the
    subclasses of :class:`SampleReducingMCAcquisitionFunction`, with
    BoTorch's ``logmeanexp`` sample reduction and ``fatmax`` /
    ``smooth_amax`` q-reduction.
    """

    _log: bool = True

    def __init__(
        self,
        model: Any,
        sampler: SobolQMCNormalSampler | None = None,
        objective: Any = None,
        posterior_transform: Any = None,
        X_pending: Any = None,
        constraints: Any = None,
        eta: Any = 1e-3,
        fat: bool = True,
        tau_max: float = TAU_MAX,
    ) -> None:
        r"""Constructor of the base class for LogEI acquisition functions.

        Args:
            model: A fitted model.
            sampler: The sampler used to draw base samples.
            objective: The MCAcquisitionObjective under which the samples are
                evaluated. Defaults to ``IdentityMCObjective()``.
            posterior_transform: A PosteriorTransform (optional).
            X_pending: A ``batch_shape, m x d``-dim Tensor of ``m`` design points
                that have points that have been submitted for function evaluation
                but have not yet been evaluated.
            constraints: A list of constraint callables which map a Tensor
                of posterior samples of dimension
                ``sample_shape x batch-shape x q x m``-dim to a
                ``sample_shape x batch-shape x q``-dim Tensor. The
                associated constraints
                are satisfied if ``constraint(samples) < 0``.
            eta: Temperature parameter(s) governing the smoothness of
                the sigmoid approximation to the constraint indicators.
                See the docs of ``compute_(log_)constraint_indicator``
                for more details on this parameter.
            fat: Toggles the logarithmic / linear asymptotic behavior of the smooth
                approximation to the ReLU.
            tau_max: Temperature parameter controlling the sharpness of the
                approximation to the ``max`` operator over the ``q`` candidate points.
        """
        _check_log_objective(objective)
        super().__init__(
            model=model,
            sampler=sampler,
            objective=objective,
            posterior_transform=posterior_transform,
            X_pending=X_pending,
            sample_reduction=_logmeanexp,
            q_reduction=_SmoothMax(bool(fat), tau_max),
            constraints=constraints,
            eta=eta,
            fat=fat,
        )
        self.tau_max = tau_max


class qLogExpectedImprovement(LogImprovementMCAcquisitionFunction):
    r"""MC-based batch Log Expected Improvement: a smooth, numerically robust
    approximation of ``log E[max(max_j y_j - best_f, 0)]`` (fat-tailed
    ``fatplus`` / ``fatmax`` by default).

    With outcome ``constraints``, the log of the smoothed feasibility
    indicator (``log_fatmoid(-c / eta)`` with ``fat=True``, else
    ``log(sigmoid(-c / eta))``) is added to the log-improvement of every sample
    and point. ``fat`` may also be a list with one entry per constraint
    (``True``, ``False``, or ``None`` for constraints that return
    probabilities of feasibility), as in BoTorch.
    """

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
        _check_log_objective(objective)
        check_constrained_objective(objective, constraints)
        super().__init__(model, sampler, objective, posterior_transform, X_pending)
        self.best_f = best_f
        self.fat = fat
        self.tau_max = tau_max
        self.tau_relu = tau_relu
        fat_bool, fat_list = fat_arg(fat)
        kwargs = self._set_constraints(constraints, eta, fat)
        self._build(
            "qLogExpectedImprovement",
            best_f=scalar_arg(best_f, "best_f"),
            fat=fat_bool,
            tau_max=scalar_arg(tau_max, "tau_max"),
            tau_relu=scalar_arg(tau_relu, "tau_relu"),
            fat_list=fat_list,
            **kwargs,
        )


class qLogNoisyExpectedImprovement(LogImprovementMCAcquisitionFunction):
    r"""MC-based batch Log Noisy Expected Improvement: a smooth approximation of
    ``log E[max(max_j y_j - max_i y_i^baseline, 0)]`` over joint posterior
    samples of the candidates and the (pruned) baseline points.

    botorch_rs always uses the plain joint computation (BoTorch's
    ``cache_root=False``), which equals BoTorch's default ``cache_root=True``
    in distribution; the ``cache_root`` argument is accepted and ignored. The
    baseline is pruned with ``prune_inferior_points`` (2048 Sobol samples)
    seeded deterministically from the sampler's seed.

    With outcome ``constraints``, the incumbent of every sample is the best
    *feasible* baseline objective (``compute_best_feasible_objective``); if
    some sample has no feasible baseline point, infeasible points get a lower
    bound of the objective estimated, as in BoTorch, from the posterior at 32
    random points of the (10% expanded) bounding box of the baseline. botorch_rs
    seeds these points from the sampler (BoTorch draws them with ``torch.rand``);
    they can be given explicitly with the keyword-only ``lower_bound_points``
    (a botorch_rs extension) and are returned by :attr:`lower_bound_points`.
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
        *,
        lower_bound_points: Any = None,
    ) -> None:
        check_unsupported(marginalize_dim=marginalize_dim)
        _check_log_objective(objective)
        check_constrained_objective(objective, constraints)
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
        self._warned_infeasible = False
        fat_bool, fat_list = fat_arg(fat)
        kwargs = self._set_constraints(constraints, eta, fat)
        lb = (
            None
            if lower_bound_points is None
            else to_flat(lower_bound_points, "lower_bound_points")
        )
        self._rs = _rs.AcquisitionFunction.q_log_noisy_expected_improvement(
            self._rs_model,
            baseline,
            self.sampler._rs,
            pending,
            fat=fat_bool,
            prune_baseline=bool(prune_baseline),
            prune_num_samples=_PRUNE_SAMPLES,
            prune_max_frac=1.0,
            prune_seed=(self.sampler.seed ^ _PRUNE_SEED_SALT) & 0xFFFFFFFFFFFFFFFF,
            tau_max=scalar_arg(tau_max, "tau_max"),
            tau_relu=scalar_arg(tau_relu, "tau_relu"),
            incremental=bool(incremental),
            objective=self._objective,
            fat_list=fat_list,
            lower_bound_points=lb,
            **kwargs,
        )

    def _after_evaluation(self) -> None:
        # BoTorch warns when some sample has no feasible baseline point.
        if not self._warned_infeasible and self._rs.infeasible_lower_bound() is not None:
            self._warned_infeasible = True
            warnings.warn(
                "When all training points are infeasible, it is better to use "
                "q(Log)ProbabilityOfFeasibility.",
                BotorchWarning,
                stacklevel=3,
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

    @property
    def lower_bound_points(self) -> Any:
        """The points at which the objective's lower bound for infeasible
        baseline points is estimated (``k x d``)."""
        return from_pair(self._rs.lower_bound_points(), self._baseline_kind)


class qLogProbabilityOfFeasibility(LogImprovementMCAcquisitionFunction):
    r"""MC-based batch Log Probability of Feasibility: a smooth approximation
    of ``log P(constraint_i(f(X)) <= 0 for all i)``, the log of the average
    over samples of the smoothed maximum over the ``q`` points of the smoothed
    feasibility indicators (``fat`` and ``tau_max`` as in qLogEI). The
    ``objective`` is not used (kept for BoTorch compatibility)."""

    _log = True

    def __init__(
        self,
        model: Any,
        constraints: Any,
        sampler: SobolQMCNormalSampler | None = None,
        objective: Any = None,
        posterior_transform: Any = None,
        X_pending: Any = None,
        eta: Any = 1e-3,
        fat: bool = True,
        tau_max: float = TAU_MAX,
    ) -> None:
        # Like BoTorch, the objective is replaced by the identity.
        super().__init__(model, sampler, IdentityMCObjective(), posterior_transform, X_pending)
        self.fat = fat
        self.tau_max = tau_max
        fat_bool, fat_list = fat_arg(fat)
        kwargs = self._set_constraints(constraints, eta, fat)
        if kwargs["constraints"] is None:
            raise TypeError("qLogProbabilityOfFeasibility requires `constraints`.")
        self._rs = _rs.AcquisitionFunction.q_log_probability_of_feasibility(
            self._rs_model,
            kwargs["constraints"],
            kwargs["eta"],
            self.sampler._rs,
            self._pending,
            fat=fat_bool,
            tau_max=scalar_arg(tau_max, "tau_max"),
            fat_list=fat_list,
        )


__all__ = [
    "TAU_MAX",
    "TAU_RELU",
    "LogImprovementMCAcquisitionFunction",
    "qLogExpectedImprovement",
    "qLogNoisyExpectedImprovement",
    "qLogProbabilityOfFeasibility",
]

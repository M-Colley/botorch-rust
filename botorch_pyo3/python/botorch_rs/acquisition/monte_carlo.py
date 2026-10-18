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
from collections.abc import Callable
from typing import Any, Protocol

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


class SampleReductionProtocol(Protocol):
    """For static type check of SampleReducingMCAcquisitionFunction's mc_reduction."""

    @staticmethod
    def __call__(X: Any, *, dim: Any) -> Any:
        pass  # pragma: no cover


def _defines_sample_forward(cls: type) -> bool:
    """Whether ``_sample_forward`` of ``cls`` is defined outside of botorch_rs
    (a Python subclass whose utilities are evaluated through torch)."""
    for c in cls.__mro__:
        if "_sample_forward" in c.__dict__:
            return not c.__module__.startswith("botorch_rs.")
    return False


class SampleReducingMCAcquisitionFunction(MCAcquisitionFunction):
    r"""MC-based batch acquisition function that reduces across samples and implements
    a general treatment of outcome constraints.

    This class's ``forward`` computes the - possibly constrained - acquisition
    value by (1) computing the unconstrained utility for each MC sample using
    ``_sample_forward``, (2) weighing the utility values by the constraint
    indicator per MC sample, and (3) reducing (e.g. averaging) the weighted
    utility values over the MC dimension.

    NOTE: Do *NOT* override the ``forward`` method, unless you have thought
    about it well.

    ``forward`` is implemented generically to incorporate constraints in a
    principled way, and takes care of reducing over the Monte Carlo and batch
    dimensions via the ``sample_reduction`` and ``q_reduction`` arguments, which
    default to ``torch.mean`` and ``torch.max``, respectively.

    In order to implement a custom SampleReducingMCAcquisitionFunction, we only
    need to implement the ``_sample_forward(obj: Tensor) -> Tensor`` method,
    which maps objective samples to acquisition utility values without reducing
    the Monte Carlo and batch (i.e. q) dimensions (see details in the docstring
    of ``_sample_forward``).

    In botorch_rs, the posterior samples of a Python subclass come from Rust
    (with the sampler's base samples, shared by the t-batches) as torch
    tensors whose gradients with respect to ``X`` are exact; the objective,
    ``_sample_forward``, the constraint indicator and the reductions are
    evaluated with torch, so gradients come from ``torch.autograd`` (as for a
    ``GenericMCObjective``). ``optimize_acqf`` evaluates such subclasses from
    Rust through these torch computations. The acquisition functions of
    botorch_rs that derive from this class (``qExpectedImprovement``, ...)
    are evaluated natively.
    """

    _log: bool = False  # whether the acquisition utilities are in log-space

    def __init__(
        self,
        model: Any,
        sampler: SobolQMCNormalSampler | None = None,
        objective: Any = None,
        posterior_transform: Any = None,
        X_pending: Any = None,
        sample_reduction: SampleReductionProtocol | None = None,
        q_reduction: SampleReductionProtocol | None = None,
        constraints: list[Callable[[Any], Any]] | None = None,
        eta: Any = 1e-3,
        fat: list[bool | None] | bool = False,
    ) -> None:
        r"""Constructor of SampleReducingMCAcquisitionFunction.

        Args:
            model: A fitted model.
            sampler: The sampler used to draw base samples (a botorch_rs
                ``SobolQMCNormalSampler``; 512 samples by default).
            objective: The MCAcquisitionObjective under which the samples are
                evaluated. Defaults to ``IdentityMCObjective()``.
                NOTE: ``ConstrainedMCObjective`` for outcome constraints is
                deprecated in favor of passing the ``constraints`` directly to this
                constructor.
            posterior_transform: A ``PosteriorTransform`` (optional).
            X_pending: A ``batch_shape, m x d``-dim Tensor of ``m`` design points
                that have points that have been submitted for function evaluation
                but have not yet been evaluated.
            sample_reduction: A callable that takes in a ``sample_shape x
                batch_shape`` Tensor of acquisition utility values, a keyword-argument
                ``dim`` that specifies the sample dimensions to reduce over, and
                returns a ``batch_shape``-dim Tensor of acquisition values
                (default: ``torch.mean``).
            q_reduction: A callable that takes in a ``sample_shape x batch_shape
                x q`` Tensor of acquisition utility values, a keyword-argument ``dim``
                that specifies the q dimension to reduce over (i.e. -1), and returns
                a ``sample_shape x batch_shape``-dim Tensor of acquisition values
                (default: ``torch.amax``).
            constraints: A list of constraint callables which map a Tensor of
                posterior samples of dimension ``sample_shape x batch-shape x q x
                m``-dim to a ``sample_shape x batch-shape x q``-dim Tensor. The
                associated constraints are considered satisfied if the output is less
                than zero. NOTE: Constraint-weighting is only compatible with
                non-negative acquistion utilities, e.g. all improvement-based
                acquisition functions.
            eta: Temperature parameter(s) governing the smoothness of the sigmoid
                approximation to the constraint indicators. For more details, on this
                parameter, see the docs of
                ``compute_smoothed_feasibility_indicator``.
            fat: Whether to apply a fat-tailed smooth approximation to the feasibility
                indicator or the canonical sigmoid approximation. For more details,
                on this parameter, see the docs of
                ``compute_smoothed_feasibility_indicator``.
        """
        check_constrained_objective(objective, constraints)
        super().__init__(
            model=model,
            sampler=sampler,
            objective=objective,
            posterior_transform=posterior_transform,
            X_pending=X_pending,
        )
        self._sample_reduction_fn = sample_reduction
        self._q_reduction_fn = q_reduction
        self._constraints = constraints
        self._eta = eta
        self._fat = fat
        self._X_pending_torch: Any = None
        if X_pending is not None and type(self)._python_forward():
            self._X_pending_torch = _as_float64_tensor(X_pending)

    @classmethod
    def _python_forward(cls) -> bool:
        """Whether the class is evaluated through torch (it defines
        ``_sample_forward`` outside of botorch_rs)."""
        return _defines_sample_forward(cls)

    @property
    def sample_shape(self) -> tuple[int, ...]:
        """The sample shape of the sampler."""
        return tuple(self.sampler.sample_shape)

    def _sample_reduction(self, X: Any) -> Any:
        import torch

        fn = torch.mean if self._sample_reduction_fn is None else self._sample_reduction_fn
        return fn(X, dim=tuple(range(len(self.sample_shape))))

    def _q_reduction(self, X: Any) -> Any:
        import torch

        fn = torch.amax if self._q_reduction_fn is None else self._q_reduction_fn
        return fn(X, dim=-1)

    def __getattr__(self, name: str) -> Any:
        # The extension handle of a Python subclass: a callable acquisition
        # function evaluating ``_torch_forward`` (built on first use).
        if name == "_rs" and type(self)._python_forward():
            from botorch_rs._torch_bridge import require_torch

            require_torch()
            handle = _rs.callable_acquisition_function(self._torch_forward, int(self._rs_model.dim))
            self.__dict__["_rs"] = handle
            return handle
        raise AttributeError(f"{type(self).__name__!r} object has no attribute {name!r}")

    def _build(self, name: str, **kwargs: Any) -> None:
        if type(self)._python_forward():
            # A Python subclass of a botorch_rs acquisition function that
            # overrides ``_sample_forward``: evaluated through torch.
            return
        super()._build(name, **kwargs)

    def set_X_pending(self, X_pending: Any = None) -> None:
        """Informs the acquisition function about pending points (``m x d``);
        ``None`` removes them."""
        if not type(self)._python_forward():
            super().set_X_pending(X_pending)
            return
        pending_arg(X_pending, int(self._rs_model.dim))
        if X_pending is not None:
            self._pending_kind = kind_of(X_pending)
            self._X_pending_torch = _as_float64_tensor(X_pending)
        else:
            self._X_pending_torch = None

    @property
    def X_pending(self) -> Any:
        """The pending points (``m x d``), or ``None``."""
        if not type(self)._python_forward():
            return MCAcquisitionFunction.X_pending.fget(self)  # type: ignore[attr-defined]
        if self._X_pending_torch is None:
            return None
        from botorch_rs._arrays import from_flat, to_flat

        data, shape = to_flat(self._X_pending_torch, "X_pending")
        return from_flat(data, shape, self._pending_kind)

    def forward(self, X: Any) -> Any:
        r"""Computes the acquisition value associated with the input ``X``. Weighs
        the acquisition utility values by smoothed constraint indicators if
        ``constraints`` was passed to the constructor of the class. Applies
        ``self.sample_reduction`` and ``self.q_reduction`` to reduce over the Monte
        Carlo and batch (q) dimensions.

        Args:
            X: A ``batch_shape x q x d`` Tensor of t-batches with ``q`` ``d``-dim
                design points each.

        Returns:
            A Tensor with shape ``batch_shape'``, where ``batch_shape'`` is the
            broadcasted batch shape of model and input ``X``.
        """
        if not type(self)._python_forward():
            return super().forward(X)
        from botorch_rs._arrays import from_flat, is_torch_tensor

        if is_torch_tensor(X):
            X3 = X if X.dim() > 2 else X.unsqueeze(0)
            return self._torch_forward(X3)
        kind = kind_of(X)
        Xt = _as_float64_tensor(X)
        out = self._torch_forward(Xt if Xt.dim() > 2 else Xt.unsqueeze(0)).detach()
        return from_flat(out.reshape(-1).tolist(), tuple(out.shape), kind)

    def _torch_forward(self, X: Any) -> Any:
        non_reduced_acqval = self._non_reduced_forward(X=X)
        return self._sample_reduction(self._q_reduction(non_reduced_acqval))

    def _non_reduced_forward(self, X: Any) -> Any:
        """Compute the constrained acquisition values at the MC-sample, q level.

        Args:
            X: A ``batch_shape x q x d`` Tensor of t-batches with ``q`` ``d``-dim
                design points each.

        Returns:
            A Tensor with shape ``sample_sample x batch_shape x q``.
        """
        from botorch_rs.acquisition.utils import repeat_to_match_aug_dim

        samples, obj = self._get_samples_and_objectives(X)
        samples = repeat_to_match_aug_dim(target_tensor=samples, reference_tensor=obj)
        acqval = self._sample_forward(obj)  # ``sample_sample x batch_shape x q``
        return self._apply_constraints(acqval=acqval, samples=samples)

    def _get_samples_and_objectives(self, X: Any) -> tuple[Any, Any]:
        """Computes posterior samples and objective values at input ``X`` (with
        the pending points appended).

        Args:
            X: A ``batch_shape x q x d``-dim Tensor of model inputs.

        Returns:
            A two-tuple ``(samples, obj)``, where ``samples`` is a tensor of posterior
            samples with shape ``sample_shape x batch_shape x q x m``, and ``obj`` is a
            tensor of MC objective values with shape ``sample_shape x batch_shape x q``.
        """
        import torch

        if self._X_pending_torch is not None:
            pending = self._X_pending_torch.to(X)
            X = torch.cat([X, pending.expand(*X.shape[:-2], *pending.shape)], dim=-2)
        samples = _posterior_samples(self, X)
        obj = self.objective(samples=samples, X=X)
        return samples, obj

    def _sample_forward(self, obj: Any) -> Any:
        """Evaluates the acquisition utility per MC sample based on objective value obj.
        Should utilize the result of ``set_X_pending`` as needed to account for pending
        function evaluations.

        Args:
            obj: A ``sample_shape x batch_shape x q``-dim Tensor of MC objective values.

        Returns:
            A ``sample_shape x batch_shape x q``-dim Tensor of acquisition
                utility values.
        """
        raise NotImplementedError  # pragma: no cover

    def _apply_constraints(self, acqval: Any, samples: Any) -> Any:
        """Multiplies the acquisition utility by constraint indicators.

        Args:
            acqval: ``sample_shape x batch_shape x q``-dim acquisition utility values.
            samples: ``sample_shape x batch_shape x q x m``-dim posterior samples.

        Returns:
            A ``sample_shape x batch_shape x q``-dim Tensor of acquisition
                utility values multiplied by a smoothed constraint indicator
                per sample.
        """
        if self._constraints is not None:
            from botorch_rs.acquisition._torch_reductions import (
                compute_smoothed_feasibility_indicator,
            )

            if not self._log and (acqval < 0).any():
                raise ValueError(
                    "Constraint-weighting requires unconstrained "
                    "acquisition values to be non-negative."
                )
            ind = compute_smoothed_feasibility_indicator(
                constraints=self._constraints,
                samples=samples,
                eta=self._eta,
                log=self._log,
                fat=self._fat,
            )
            acqval = acqval.add(ind) if self._log else acqval.mul(ind)
        return acqval


def _as_float64_tensor(x: Any) -> Any:
    import torch

    from botorch_rs._arrays import is_torch_tensor, to_flat

    if is_torch_tensor(x):
        return x.detach().to(torch.float64)
    data, shape = to_flat(x, "X")
    return torch.tensor(list(data), dtype=torch.float64).reshape(shape)


def _posterior_samples(acqf: Any, X: Any) -> Any:
    """The posterior samples (``sample_shape x batch_shape x n x m``) of the
    points ``X`` (``batch_shape x n x d``) from the sampler's base samples,
    differentiable with respect to ``X`` (exact vector-Jacobian products
    from Rust)."""
    import torch

    from botorch_rs._arrays import ArrayKind, from_pair, to_flat

    handle = acqf._rs_model
    batch = tuple(X.shape[:-2])
    X3 = X.reshape(-1, *X.shape[-2:])
    n = X3.shape[-2] * int(handle.points_per_input)
    m = int(handle.num_outputs)
    base = acqf.sampler._rs.base_samples(n * m)
    kind = ArrayKind("torch", X.dtype, X.device)

    class _Samples(torch.autograd.Function):
        @staticmethod
        def forward(ctx: Any, Xin: Any) -> Any:  # type: ignore[override]
            ctx.save_for_backward(Xin)
            return from_pair(handle.posterior_samples(to_flat(Xin.detach()), base), kind)

        @staticmethod
        def backward(ctx: Any, grad: Any) -> Any:  # type: ignore[override]
            (Xin,) = ctx.saved_tensors
            g = _rs.posterior_samples_vjp(
                handle, to_flat(Xin.detach()), base, to_flat(grad.contiguous())
            )
            return from_pair(g, kind)

    samples = _Samples.apply(X3)  # S x b x n x m
    shape = (*acqf.sample_shape, *batch, *samples.shape[-2:])
    return samples.reshape(shape)


class qExpectedImprovement(SampleReducingMCAcquisitionFunction):
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


class qProbabilityOfImprovement(SampleReducingMCAcquisitionFunction):
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


class qSimpleRegret(SampleReducingMCAcquisitionFunction):
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


class qUpperConfidenceBound(SampleReducingMCAcquisitionFunction):
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
        SampleReducingMCAcquisitionFunction.__init__(
            self, model, sampler, objective, posterior_transform, X_pending
        )
        self.beta = beta
        self._build("qLowerConfidenceBound", beta=scalar_arg(beta, "beta"))


class qPosteriorStandardDeviation(SampleReducingMCAcquisitionFunction):
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
    multi-task (``MultiTaskGP``, ``LCEMGP``, ``KroneckerMultiTaskGP``) and
    deterministic models do not."""
    from botorch_rs.models.gp_regression import SingleTaskGP
    from botorch_rs.models.model_list_gp_regression import ModelListGP
    from botorch_rs.models.multitask import KroneckerMultiTaskGP, MultiTaskGP

    if isinstance(model, ModelListGP):
        return all(supports_cache_root(m) for m in model.models)
    if isinstance(model, (MultiTaskGP, KroneckerMultiTaskGP)):
        return False
    return isinstance(model, SingleTaskGP)


class qNoisyExpectedImprovement(SampleReducingMCAcquisitionFunction):
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
    "SampleReducingMCAcquisitionFunction",
    "SampleReductionProtocol",
    "qExpectedImprovement",
    "qLowerConfidenceBound",
    "qNoisyExpectedImprovement",
    "qPosteriorStandardDeviation",
    "qProbabilityOfImprovement",
    "qSimpleRegret",
    "qUpperConfidenceBound",
    "supports_cache_root",
]

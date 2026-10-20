"""Device-resident acquisition functions for the explicit Torch model backend.

Imported only after a model opts into that backend. Formulae follow the MIT
licensed BoTorch analytic and scalar MC acquisitions; inference always calls
the port's model.posterior. No upstream acquisition or model is delegated to.
"""

from __future__ import annotations

import inspect
import math
import warnings
from typing import Any

import torch
from torch import Tensor, nn

from botorch_rs.exceptions.errors import UnsupportedError
from botorch_rs.exceptions.warnings import BotorchWarning, legacy_ei_numerics_warning
from botorch_rs.utils.transforms import is_ensemble

_ANALYTIC = frozenset(
    {
        "ExpectedImprovement",
        "LogExpectedImprovement",
        "ProbabilityOfImprovement",
        "LogProbabilityOfImprovement",
        "UpperConfidenceBound",
        "PosteriorMean",
        "PosteriorStandardDeviation",
    }
)
_MC = frozenset(
    {
        "qExpectedImprovement",
        "qLogExpectedImprovement",
        "qProbabilityOfImprovement",
        "qUpperConfidenceBound",
        "qLowerConfidenceBound",
        "qSimpleRegret",
        "qPosteriorStandardDeviation",
        "qNoisyExpectedImprovement",
        "qLogNoisyExpectedImprovement",
    }
)
_NOISY = frozenset({"qNoisyExpectedImprovement", "qLogNoisyExpectedImprovement"})
_MULTI_OBJECTIVE = frozenset(
    {
        "qExpectedHypervolumeImprovement",
        "qLogExpectedHypervolumeImprovement",
        "qNoisyExpectedHypervolumeImprovement",
        "qLogNoisyExpectedHypervolumeImprovement",
    }
)
_LOOKAHEAD = frozenset({"qKnowledgeGradient", "qMultiFidelityKnowledgeGradient"})
_EXTRA_ANALYTIC = frozenset(
    {
        "ConstrainedExpectedImprovement",
        "LogConstrainedExpectedImprovement",
        "LogProbabilityOfFeasibility",
        "ScalarizedPosteriorMean",
    }
)
_WRAPPERS = frozenset({"FixedFeatureAcquisitionFunction", "ProjectedAcquisitionFunction"})
_ENTROPY = frozenset(
    {
        "qMaxValueEntropy",
        "qLowerBoundMaxValueEntropy",
        "qMultiFidelityMaxValueEntropy",
        "qMultiFidelityLowerBoundMaxValueEntropy",
        "qJointEntropySearch",
        "qPredictiveEntropySearch",
        "qMultiObjectivePredictiveEntropySearch",
        "qLowerBoundMultiObjectiveJointEntropySearch",
        "qLowerBoundMultiObjectiveMaxValueEntropySearch",
    }
)
_TYPES: dict[type, type] = {}
_LOG_2PI = math.log(2 * math.pi)


def tensor_acquisition(public_class: type) -> Any:
    """Allocate a graph-native subtype without invoking a native constructor."""
    if (
        public_class.__name__
        not in _ANALYTIC
        | _MC
        | _MULTI_OBJECTIVE
        | _LOOKAHEAD
        | _EXTRA_ANALYTIC
        | _WRAPPERS
        | _ENTROPY
        or not public_class.__module__.startswith("botorch_rs.acquisition.")
    ):
        raise UnsupportedError(f"{public_class.__name__} does not yet support backend='torch'.")
    subtype = _TYPES.get(public_class)
    if subtype is None:
        mixin: type = _TensorAcquisition
        if public_class.__name__ in _MULTI_OBJECTIVE:
            from botorch_rs.acquisition._torch_backend_multi_objective import (
                TensorMultiObjectiveAcquisition,
                TensorNoisyMultiObjectiveAcquisition,
            )

            mixin = (
                TensorNoisyMultiObjectiveAcquisition
                if "Noisy" in public_class.__name__
                else TensorMultiObjectiveAcquisition
            )
        elif public_class.__name__ in _LOOKAHEAD:
            from botorch_rs.acquisition._torch_backend_knowledge_gradient import (
                TensorKnowledgeGradient,
            )

            mixin = TensorKnowledgeGradient
        elif public_class.__name__ in _EXTRA_ANALYTIC:
            from botorch_rs.acquisition._torch_backend_analytic import TensorAnalyticExtra

            mixin = TensorAnalyticExtra
        elif public_class.__name__ in _WRAPPERS:
            from botorch_rs.acquisition._torch_backend_wrappers import (
                TensorFixedFeature,
                TensorProjected,
            )

            mixin = (
                TensorFixedFeature
                if public_class.__name__ == "FixedFeatureAcquisitionFunction"
                else TensorProjected
            )
        elif public_class.__name__ in _ENTROPY:
            if "PredictiveEntropySearch" in public_class.__name__:
                from botorch_rs.acquisition._torch_backend_predictive_entropy import (
                    TensorPredictiveEntropy,
                )

                mixin = TensorPredictiveEntropy
            elif "MultiObjective" in public_class.__name__:
                from botorch_rs.acquisition._torch_backend_mo_entropy import (
                    TensorMultiObjectiveEntropy,
                )

                mixin = TensorMultiObjectiveEntropy
            elif public_class.__name__ == "qJointEntropySearch":
                from botorch_rs.acquisition._torch_backend_joint_entropy import (
                    TensorJointEntropySearch,
                )

                mixin = TensorJointEntropySearch
            else:
                from botorch_rs.acquisition._torch_backend_entropy import TensorMaxValueEntropy

                mixin = TensorMaxValueEntropy
        name = f"Torch{public_class.__name__}"
        subtype = type(
            name,
            (mixin, public_class),
            {"__module__": __name__, "_public_class": public_class},
        )
        _TYPES[public_class] = subtype
        globals()[name] = subtype
    return object.__new__(subtype)


def _as_tensor(value: Any, like: Tensor) -> Tensor:
    return torch.as_tensor(value, dtype=like.dtype, device=like.device)


def _log_ei(u: Tensor) -> Tensor:
    """Stable log(phi(u) + u Phi(u)), including negative tails and grad-grad."""
    if u.dtype not in (torch.float32, torch.float64):
        raise TypeError("LogExpectedImprovement requires float32 or float64.")
    upper_u = u.masked_fill(u < -1, -1)
    phi = (-0.5 * upper_u.square() - 0.5 * _LOG_2PI).exp()
    upper = (phi + upper_u * torch.special.ndtr(upper_u)).log()
    limit = -1e6 if u.dtype == torch.float64 else -1e3
    lower_u = u.masked_fill(u > -1, -1)
    eps_u = lower_u.masked_fill(u < limit, limit)
    w = (torch.special.erfcx(-(2**-0.5) * eps_u) * eps_u.abs()).log()
    w = w + math.log(math.pi / 2) / 2
    # Mask both branches before evaluating them, avoiding 0 * NaN in backward.
    near = w.masked_fill(w < -math.log(2), -math.log(2))
    far = w.masked_fill(w >= -math.log(2), -math.log(2))
    log_one_minus = torch.where(w < -math.log(2), (-far.exp()).log1p(), (-near.expm1()).log())
    lower = (
        -0.5 * u.square()
        - 0.5 * _LOG_2PI
        + torch.where(u > limit, log_one_minus, -2 * lower_u.abs().log())
    )
    return torch.where(u > -1, upper, lower)


def _log_softplus(value: Tensor, tau: Tensor, fat: bool) -> Tensor:
    scaled = value / tau
    if fat:
        return (
            tau * (torch.nn.functional.softplus(scaled) + 0.1 * (1 / (1 + scaled.square())))
        ).log()
    lower = -15 if value.dtype == torch.float32 else -35
    upper = 16 if value.dtype == torch.float32 else 32
    masked = scaled.masked_fill(scaled < lower, lower)
    regular = torch.nn.functional.softplus(masked, threshold=upper).log()
    return tau.log() + torch.where(scaled > lower, regular, scaled)


def _logmeanexp(value: Tensor, dimensions: tuple[int, ...]) -> Tensor:
    return torch.logsumexp(value, dim=dimensions) - math.log(
        math.prod(value.shape[d] for d in dimensions)
    )


def _average_ensemble(value: Tensor, model: Any, log: bool = False) -> Tensor:
    """Match BoTorch's final-axis mixture reduction without merging t-batches."""
    if not is_ensemble(model):
        return value
    return _logmeanexp(value, (-1,)) if log else value.mean(dim=-1)


class _TensorAcquisition(nn.Module):
    """Mixin preceding the public acquisition class in the dispatched MRO."""

    _tensor_backend = "torch"
    _public_class: type
    best_f: Tensor
    beta: Tensor
    tau: Tensor
    tau_max: Tensor
    tau_relu: Tensor
    sampler: Any
    objective: Any
    _X_pending: Tensor | None
    _X_baseline: Tensor
    _lower_bound_points: Tensor | None

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        nn.Module.__init__(self)
        name = self._public_class.__name__
        kwargs = dict(kwargs)
        detach_baseline = kwargs.pop("detach_baseline", True) if name in _NOISY else True
        bound = inspect.signature(inspect.getattr_static(self._public_class, "__init__")).bind(
            self, *args, **kwargs
        )
        bound.apply_defaults()
        values = dict(bound.arguments)
        values.pop("self")
        self._name = name
        self._tensor_backend = "torch"
        self._is_analytic = name in _ANALYTIC
        self._log = name.startswith("Log") or name.startswith("qLog")
        self.model = values["model"]
        self.posterior_transform = values.get("posterior_transform")
        self.maximize = bool(values.get("maximize", True))
        self.detach_baseline = bool(detach_baseline)
        for field in ("best_f", "beta", "tau", "tau_max", "tau_relu"):
            if field in values:
                value = values[field]
                dtype = (
                    None
                    if isinstance(value, Tensor) or field in {"best_f", "tau"} or self._is_analytic
                    else torch.float64
                )
                self.register_buffer(field, torch.as_tensor(value, dtype=dtype))
        self.fat = values.get("fat", self._log)
        self._fat = self.fat
        self._constraints = values.get("constraints")
        # Official scalar temperatures first use the default Torch dtype;
        # retain that rounding before conversion to the sample dtype/device.
        self.register_buffer("_eta", torch.as_tensor(values.get("eta", 1e-3)))
        self.register_buffer("_X_pending", None)
        for temperature in ("tau", "tau_relu", "tau_max"):
            if temperature in values and not (getattr(self, temperature) > 0).all():
                raise ValueError(f"{temperature} must be positive.")
        if "beta" in values and not (self.beta >= 0).all():
            raise ValueError("beta must be non-negative.")
        if name in {"ExpectedImprovement", "qExpectedImprovement", "qNoisyExpectedImprovement"}:
            legacy_ei_numerics_warning(legacy_name=name)
        if self._is_analytic:
            if self.model.num_outputs != 1 and self.posterior_transform is None:
                raise UnsupportedError(
                    "A multi-output analytic model requires a posterior transform."
                )
            return
        from botorch_rs.acquisition.objective import IdentityMCObjective, is_constrained_objective
        from botorch_rs.sampling.normal import SobolQMCNormalSampler

        objective = values.get("objective")
        if self._constraints is not None and is_constrained_objective(objective):
            raise ValueError("ConstrainedMCObjective and constraints cannot both be provided.")
        if self._log and is_constrained_objective(objective):
            raise ValueError("Log-improvement requires constraints passed separately.")
        if objective is None and self.model.num_outputs != 1:
            if not getattr(self.posterior_transform, "scalarize", False):
                raise UnsupportedError(
                    "A multi-output model requires an objective or scalarizing posterior transform."
                )
        self.objective = IdentityMCObjective() if objective is None else objective
        self.sampler = values.get("sampler")
        if self.sampler is None:
            self.sampler = SobolQMCNormalSampler(sample_shape=(512,))
        self.set_X_pending(values.get("X_pending"))
        if name in _NOISY:
            self._init_noisy(values)

    @property
    def sample_shape(self) -> Any:
        return self.sampler.sample_shape

    @property
    def X_pending(self) -> Any:
        return self._X_pending

    def set_X_pending(self, X_pending: Any = None) -> None:
        if self._is_analytic:
            raise UnsupportedError(
                "Analytic acquisition functions do not account for X_pending yet."
            )
        if X_pending is not None:
            if not isinstance(X_pending, Tensor) or X_pending.ndim < 2:
                raise ValueError("X_pending must be a tensor with at least two dimensions.")
            if X_pending.requires_grad:
                warnings.warn(
                    "Pending points are copied without gradients.", BotorchWarning, stacklevel=2
                )
            X_pending = X_pending.detach().clone()
        self._X_pending = X_pending

    def _candidates_tensor(self, X: Any) -> Tensor:
        if not isinstance(X, Tensor):
            raise TypeError("Acquisitions for backend='torch' require Torch tensor candidates.")
        if X.ndim < 2:
            raise ValueError("X must have shape batch_shape x q x d (at least two dimensions).")
        if self._is_analytic and X.shape[-2] != 1:
            raise ValueError("Analytic acquisitions require q=1.")
        X = X.unsqueeze(0) if X.ndim == 2 else X
        if self._X_pending is not None and not (self._name in _NOISY and self.incremental):
            pending = self._X_pending.to(X)
            batch = torch.broadcast_shapes(X.shape[:-2], pending.shape[:-2])
            X = torch.cat(
                (X.expand(*batch, *X.shape[-2:]), pending.expand(*batch, *pending.shape[-2:])),
                dim=-2,
            )
        return X

    def _posterior(self, X: Tensor) -> Any:
        return self.model.posterior(X=X, posterior_transform=self.posterior_transform)

    def _mean_and_sigma(
        self, X: Tensor, compute_sigma: bool = True, min_var: float = 1e-12
    ) -> tuple[Tensor, Tensor | None]:
        posterior = self._posterior(X)
        if self._name in {
            "ExpectedImprovement",
            "LogExpectedImprovement",
            "ProbabilityOfImprovement",
            "LogProbabilityOfImprovement",
        }:
            from botorch_rs.posteriors.gpytorch import GPyTorchPosterior

            if not isinstance(posterior, GPyTorchPosterior):
                raise UnsupportedError(
                    "Analytic EI/PI requires a Gaussian posterior; use an MC acquisition for nonlinear outcome transforms."
                )
        mean = posterior.mean.squeeze(-2)
        sigma = (
            posterior.variance.clamp_min(min_var).sqrt().reshape(mean.shape)
            if compute_sigma
            else None
        )
        return mean, sigma

    def _analytic_forward(self, X: Tensor) -> Tensor:
        name = self._name
        mean, sigma = self._mean_and_sigma(X, compute_sigma=name != "PosteriorMean")
        if mean.shape[-1] != 1:
            raise UnsupportedError("The posterior transform must produce one output.")
        sign = 1 if self.maximize else -1
        if name == "PosteriorMean":
            return (mean * sign).squeeze(-1)
        assert sigma is not None
        if name == "PosteriorStandardDeviation":
            return (sigma * sign).squeeze(-1)
        if name == "UpperConfidenceBound":
            beta = _as_tensor(self.beta, mean).unsqueeze(-1)
            return (sign * mean + beta.sqrt() * sigma).squeeze(-1)
        best = _as_tensor(self.best_f, mean).unsqueeze(-1)
        u = sign * (mean - best) / sigma
        if name == "ProbabilityOfImprovement":
            return torch.special.ndtr(u).squeeze(-1)
        if name == "LogProbabilityOfImprovement":
            return torch.special.log_ndtr(u).squeeze(-1)
        if name == "LogExpectedImprovement":
            return (_log_ei(u) + sigma.log()).squeeze(-1)
        phi = (-0.5 * u.square() - 0.5 * _LOG_2PI).exp()
        return (sigma * (phi + u * torch.special.ndtr(u))).squeeze(-1)

    def _draw(self, X: Tensor) -> tuple[Tensor, Tensor]:
        samples = self.sampler(self._posterior(X))
        objective = self.objective(samples=samples, X=X)
        if not isinstance(objective, Tensor) or objective.shape[-1] != samples.shape[-2]:
            raise RuntimeError(
                "The MC objective must return sample_shape x batch_shape x q tensors."
            )
        if samples.shape[0] != objective.shape[0]:
            if objective.shape[0] % samples.shape[0]:
                raise RuntimeError("An augmented objective must repeat the first sample dimension.")
            samples = samples.repeat(
                objective.shape[0] // samples.shape[0], *([1] * (samples.ndim - 1))
            )
        return samples, objective

    def _constraint_indicator(self, samples: Tensor, log: bool) -> Tensor:
        result = torch.zeros_like(samples[..., 0])
        constraints = self._constraints or []
        eta = _as_tensor(self._eta, samples).expand(len(constraints))
        if not (eta > 0).all():
            raise ValueError("eta must be positive.")
        fats = self.fat if isinstance(self.fat, list) else [self.fat] * len(constraints)
        if len(fats) != len(constraints):
            raise ValueError("The fat list must match the number of constraints.")
        for constraint, temperature, fat in zip(constraints, eta, fats, strict=True):
            value = constraint(samples)
            if fat is None:
                contribution = value.log()
            elif fat:
                scaled = -value / temperature
                shift = math.sqrt(1 / 3)
                sigmoid = torch.where(
                    scaled < 0,
                    (2 / 3) / (1 + (scaled - shift).square()),
                    1 - (2 / 3) / (1 + (scaled + shift).square()),
                )
                contribution = sigmoid.log()
            else:
                contribution = torch.nn.functional.logsigmoid(-value / temperature)
            result = result + contribution
        return result if log else result.exp()

    def _utilities(self, objective: Tensor, best_f: Tensor | None = None) -> Tensor:
        name = self._name
        if name == "qSimpleRegret":
            return objective
        if name in {
            "qUpperConfidenceBound",
            "qLowerConfidenceBound",
            "qPosteriorStandardDeviation",
        }:
            center = objective.mean(dim=0)
            deviation = (objective - center).abs()
            if name == "qPosteriorStandardDeviation":
                return deviation * math.sqrt(math.pi / 2)
            factor = (_as_tensor(self.beta, objective) * math.pi / 2).sqrt().unsqueeze(-1)
            if name == "qLowerConfidenceBound":
                factor = -factor
            return center + factor * deviation
        if best_f is None:
            best_f = _as_tensor(self.best_f, objective)
        difference = objective - best_f.unsqueeze(-1)
        if name == "qProbabilityOfImprovement":
            return (difference / _as_tensor(self.tau, difference)).sigmoid()
        if self._log:
            return _log_softplus(difference, _as_tensor(self.tau_relu, difference), bool(self.fat))
        return difference.clamp_min(0)

    def _reduce(self, utilities: Tensor) -> Tensor:
        sample_dims = tuple(range(len(self.sample_shape)))
        if is_ensemble(self.model):
            sample_dims += (-1,)
        if self._log:
            from botorch_rs.acquisition._torch_reductions import fatmax, smooth_amax

            tau = _as_tensor(self.tau_max, utilities)
            q_values = (fatmax if self.fat else smooth_amax)(utilities, dim=-1, tau=tau)
            return _logmeanexp(q_values, sample_dims)
        return utilities.amax(dim=-1).mean(dim=sample_dims)

    def _apply_constraints(self, acqval: Tensor, samples: Tensor) -> Tensor:
        if self._constraints is None:
            return acqval
        if not self._log and (acqval < 0).any():
            raise ValueError(
                "Constraint-weighting requires unconstrained acquisition values to be non-negative."
            )
        indicator = self._constraint_indicator(samples, self._log)
        return acqval + indicator if self._log else acqval * indicator

    def _sample_forward(self, obj: Tensor) -> Tensor:
        return self._utilities(
            obj, best_f=self.compute_best_f(obj) if self._name in _NOISY else None
        )

    def _get_samples_and_objectives(self, X: Tensor) -> tuple[Tensor, Tensor]:
        if self._name not in _NOISY:
            return self._draw(X)
        baseline = self.X_baseline.to(X)
        batch = torch.broadcast_shapes(baseline.shape[:-2], X.shape[:-2])
        joint = torch.cat(
            (baseline.expand(*batch, *baseline.shape[-2:]), X.expand(*batch, *X.shape[-2:])), dim=-2
        )
        samples, objective = self._draw(joint)
        n = baseline.shape[-2]
        self.baseline_samples, self.baseline_obj = samples[..., :n, :], objective[..., :n]
        self._last_best_f = self._best_baseline(self.baseline_samples, self.baseline_obj)
        return samples[..., n:, :], objective[..., n:]

    def _non_reduced_forward(self, X: Tensor) -> Tensor:
        samples, objective = self._get_samples_and_objectives(X)
        return self._apply_constraints(self._sample_forward(objective), samples)

    def forward(self, X: Any) -> Tensor:
        X = self._candidates_tensor(X)
        if self._is_analytic:
            return _average_ensemble(self._analytic_forward(X), self.model, self._log)
        return self._reduce(self._non_reduced_forward(X))

    def value_and_grad(self, X: Tensor) -> tuple[Tensor, Tensor]:
        with torch.enable_grad():
            variable = X if X.requires_grad else X.detach().requires_grad_(True)
            value = self(variable)
            gradient = torch.autograd.grad(
                value.sum(), variable, create_graph=True, retain_graph=True, allow_unused=True
            )[0]
            if gradient is None:
                gradient = variable * 0
        return value, gradient

    def _init_noisy(self, values: dict[str, Any]) -> None:
        baseline = values["X_baseline"]
        if not isinstance(baseline, Tensor) or baseline.ndim != 2 or baseline.shape[-2] == 0:
            raise ValueError("X_baseline must be a nonempty n x d Torch tensor.")
        marginalize = values.get("marginalize_dim")
        if marginalize is not None and not (is_ensemble(self.model) and marginalize == -3):
            raise UnsupportedError(
                "Torch noisy EI supports automatic ensemble marginalization or marginalize_dim=-3 for an ensemble model."
            )
        self.incremental = bool(values.get("incremental", False))
        self.cache_root = self._cache_root = False
        if values.get("cache_root"):
            warnings.warn(
                "backend='torch' currently uses exact full-joint noisy sampling; "
                "setting cache_root=False.",
                RuntimeWarning,
                stacklevel=3,
            )
        self.prune_baseline = bool(values.get("prune_baseline", True))
        self.register_buffer("_lower_bound_points", values.get("lower_bound_points"))
        self.register_buffer("_X_baseline", baseline)
        if self.prune_baseline:
            self._prune_baseline()

    @property
    def X_baseline(self) -> Tensor:
        baseline = self._X_baseline
        if self.incremental and self._X_pending is not None:
            pending = self._X_pending.to(baseline)
            batch = torch.broadcast_shapes(baseline.shape[:-2], pending.shape[:-2])
            baseline = torch.cat(
                (
                    baseline.expand(*batch, *baseline.shape[-2:]),
                    pending.expand(*batch, *pending.shape[-2:]),
                ),
                dim=-2,
            )
        return baseline

    @property
    def lower_bound_points(self) -> Tensor | None:
        return self._lower_bound_points

    def _feasible(self, samples: Tensor) -> Tensor:
        result = torch.ones_like(samples[..., 0], dtype=torch.bool)
        for constraint in self._constraints or []:
            result = result & (constraint(samples) <= 0)
        return result

    def _prune_baseline(self) -> None:
        from botorch_rs.sampling.normal import SobolQMCNormalSampler

        ensemble = is_ensemble(self.model)
        if self.model.batch_shape and not (ensemble and len(self.model.batch_shape) == 1):
            raise UnsupportedError("Noisy baseline pruning requires an unbatched Torch model.")
        sampler = SobolQMCNormalSampler((2048,), seed=int(getattr(self.sampler, "seed", 0)))
        with torch.no_grad():
            samples = sampler(self._posterior(self._X_baseline))
            objective = self.objective(samples=samples, X=self._X_baseline)
            feasible = self._feasible(samples)
            if ensemble:
                objective = objective.mean(dim=-2)
                feasible = feasible.to(objective).mean(dim=-2).round().bool()
            if not feasible.any():
                # The reference fallback ranks each member's probability of
                # feasibility when no point has majority ensemble feasibility.
                values = torch.zeros_like(samples[..., 0])
                for constraint in self._constraints or []:
                    values = values + torch.nn.functional.logsigmoid(-constraint(samples) / 1e-3)
            else:
                values = torch.where(feasible, objective, objective.amin() - 1)
            indices = values.argmax(dim=-1).unique()
        self._X_baseline = self._X_baseline.index_select(-2, indices)

    def _infeasible_lower_bound(self) -> Tensor:
        baseline = self._X_baseline
        if self._lower_bound_points is None:
            generator = torch.Generator(device=baseline.device)
            generator.manual_seed(int(getattr(self.sampler, "seed", 0)) ^ 0x5DEECE66D)
            # These auxiliary cost-estimation points are fixed after generation,
            # as when explicitly supplied; do not cache a consumed autograd graph.
            low, high = baseline.detach().amin(dim=-2), baseline.detach().amax(dim=-2)
            width = high - low
            uniform = torch.rand(
                (32, baseline.shape[-1]),
                dtype=baseline.dtype,
                device=baseline.device,
                generator=generator,
            )
            self._lower_bound_points = low - 0.1 * width + 1.2 * width * uniform
        points = self._lower_bound_points
        posterior = self._posterior(points)
        deviation = posterior.variance.clamp_min(0).sqrt() * 6
        lower = torch.minimum(
            self.objective(samples=posterior.mean - deviation, X=points),
            self.objective(samples=posterior.mean + deviation, X=points),
        ).amin()
        return lower.clamp_max(0)

    def _best_baseline(self, samples: Tensor, objective: Tensor) -> Tensor:
        if self._constraints is None:
            result = objective.amax(dim=-1)
        else:
            feasible = self._feasible(samples)
            if feasible.any(dim=-1).all():
                lower: Tensor | float = -torch.inf
            else:
                lower = self._infeasible_lower_bound()
            result = torch.where(feasible, objective, lower).amax(dim=-1)
        return result.detach() if self.detach_baseline else result

    def compute_best_f(self, obj: Tensor) -> Tensor:
        best = getattr(self, "_last_best_f", None)
        if best is None:
            raise RuntimeError("compute_best_f requires a completed noisy forward evaluation.")
        n_sample_dims = len(self.sample_shape)
        shape = (
            *best.shape[:n_sample_dims],
            *((1,) * (obj.ndim - best.ndim - 1)),
            *best.shape[n_sample_dims:],
        )
        return best.reshape(shape).to(obj)

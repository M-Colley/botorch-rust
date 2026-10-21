"""Device-native Gaussian max-value entropy search and GIBBON."""

from __future__ import annotations

import copy
import inspect
import math
from typing import Any

import torch
from torch import Tensor, nn

from botorch_rs import _botorch_rs as _rs
from botorch_rs.acquisition._torch_backend_acquisition import _TensorAcquisition
from botorch_rs.exceptions.errors import UnsupportedError
from botorch_rs.posteriors._torch_backend import TorchGPyTorchPosterior
from botorch_rs.posteriors.gpytorch import GPyTorchPosterior
from botorch_rs.sampling.normal import NormalMCSampler, SobolQMCNormalSampler

CLAMP_LB = 1e-8
LOG_2PI = math.log(2 * math.pi)


def _gaussian(model: Any, X: Tensor, transform: Any = None, noise: bool = False) -> Any:
    posterior = model.posterior(X, observation_noise=noise, posterior_transform=transform)
    if (
        isinstance(posterior, GPyTorchPosterior)
        and posterior.__dict__.get("_distribution_adapter") is not None
    ):
        posterior = TorchGPyTorchPosterior(posterior.mean, posterior.distribution.covariance_matrix)
    if not isinstance(posterior, TorchGPyTorchPosterior) or posterior.mean.shape[-1] != 1:
        raise UnsupportedError("Device entropy search requires a scalar Gaussian posterior.")
    return posterior


def sample_maximums(
    model: Any,
    candidate_set: Tensor,
    num_samples: int,
    posterior_transform: Any = None,
    maximize: bool = True,
    *,
    gumbel: bool,
    seed: int | None = None,
    uniforms: Any = None,
    base_samples: Any = None,
) -> Tensor:
    """Posterior maxima with no SciPy, NumPy or inference-data host transfers."""
    posterior = _gaussian(model, candidate_set, posterior_transform)
    if num_samples < 1 or candidate_set.shape[-2] < 1:
        raise ValueError("Maximum sampling requires positive sample and candidate counts.")
    sign = 1.0 if maximize else -1.0
    generator = (
        None if seed is None else torch.Generator(device=candidate_set.device).manual_seed(seed)
    )
    if not gumbel:
        if base_samples is None:
            base_samples = torch.randn(
                (num_samples,) + posterior.base_sample_shape,
                dtype=posterior.dtype,
                device=posterior.device,
                generator=generator,
            )
        else:
            base_samples = torch.as_tensor(
                base_samples, dtype=posterior.dtype, device=posterior.device
            )
        samples = sign * posterior.rsample_from_base_samples((num_samples,), base_samples).squeeze(
            -1
        )
        result = samples.amax(-1)
        return result.unsqueeze(-1) if result.ndim == 1 else result
    mean = sign * posterior.mean.squeeze(-1)
    sigma = posterior.variance.squeeze(-1).clamp_min(CLAMP_LB).sqrt()
    batch = mean.shape[:-1]
    mean, sigma = mean.reshape(-1, mean.shape[-1]), sigma.reshape(-1, sigma.shape[-1])
    probabilities = mean.new_tensor([0.25, 0.5, 0.75]).log().unsqueeze(-1)
    with torch.no_grad():
        lower = (mean - 3 * sigma).amin(-1).expand(3, -1).clone()
        upper = (mean + 5 * sigma).amax(-1).expand(3, -1).clone()
        width = (upper - lower).clamp_min(torch.finfo(mean.dtype).eps)
        # The reference's fixed bracket can fail for a very large candidate
        # set. Expanding it keeps the same quantile law when that occurs.
        for _ in range(12):
            low_cdf = torch.special.log_ndtr((lower.unsqueeze(-1) - mean) / sigma).sum(-1)
            high_cdf = torch.special.log_ndtr((upper.unsqueeze(-1) - mean) / sigma).sum(-1)
            lower = torch.where(low_cdf > probabilities, lower - width, lower)
            upper = torch.where(high_cdf < probabilities, upper + width, upper)
            width = width * 2
        for _ in range(64):
            middle = (lower + upper) * 0.5
            log_cdf = torch.special.log_ndtr((middle.unsqueeze(-1) - mean) / sigma).sum(-1)
            lower, upper = (
                torch.where(log_cdf < probabilities, middle, lower),
                torch.where(log_cdf < probabilities, upper, middle),
            )
        quantiles = (lower + upper) * 0.5
    # Optional standalone calls retain implicit quantile derivatives. The
    # acquisition constructs its sampled maxima under no_grad, like BoTorch.
    for _ in range(4):
        z = (quantiles.unsqueeze(-1) - mean) / sigma
        log_cdf = torch.special.log_ndtr(z)
        derivative = ((-0.5 * z.square() - 0.5 * LOG_2PI - log_cdf).exp() / sigma).sum(-1)
        quantiles = quantiles - (log_cdf.sum(-1) - probabilities) / derivative
    # Preserve the posterior precision, as the native CPU sampler does. The
    # pinned reference accidentally rounds SciPy roots through its default dtype.
    q25, q50, q75 = quantiles.unbind(0)
    scale = (q25 - q75) / (math.log(math.log(4 / 3)) - math.log(math.log(4)))
    location = q50 + scale * math.log(math.log(2))
    if uniforms is None:
        uniforms = torch.rand(
            (num_samples, 1), dtype=mean.dtype, device=mean.device, generator=generator
        )
    else:
        uniforms = torch.as_tensor(uniforms, dtype=mean.dtype, device=mean.device).reshape(
            num_samples, 1
        )
    eps = torch.finfo(mean.dtype).eps
    uniforms = uniforms.clamp(eps, 1 - eps)
    return (location - scale * (-uniforms.log()).log()).reshape(num_samples, *(batch or (1,)))


def _clone_sampler(source: Any) -> Any:
    if not isinstance(source, NormalMCSampler):
        return copy.deepcopy(source)
    sampler_type: Any = type(source)
    result = sampler_type(source.sample_shape, seed=source.seed)
    if source.base_samples is not None:
        if isinstance(result, SobolQMCNormalSampler):
            result._base_samples = source.base_samples.clone()
        else:
            result.base_samples = source.base_samples.clone()
        for name in ("_base_shape", "_tensor_sampler_key", "_base_samples_explicit"):
            if hasattr(source, name):
                setattr(result, name, getattr(source, name))
    return result


class TensorMaxValueEntropy(_TensorAcquisition):
    """MES / GIBBON and their multi-fidelity trace/cost compositions."""

    candidate_set: Tensor
    _base_max_values: Tensor
    project: Any
    expand: Any
    cost_aware_utility: Any

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        nn.Module.__init__(self)
        bound = inspect.signature(inspect.getattr_static(self._public_class, "__init__")).bind(
            self, *args, **kwargs
        )
        bound.apply_defaults()
        values = dict(bound.arguments)
        self._name = self._public_class.__name__
        self._tensor_backend, self._is_analytic, self._log = "torch", False, False
        self._lower_bound = "LowerBound" in self._name
        self._multi_fidelity = "MultiFidelity" in self._name
        self._init_model = self.model = values["model"]
        if self._lower_bound and self.model.batch_shape:
            raise UnsupportedError("Device GIBBON currently requires an initially unbatched model.")
        self.posterior_transform = values["posterior_transform"]
        if self.model.num_outputs != 1:
            raise UnsupportedError("Max-value entropy requires a single-output model.")
        self.maximize = bool(values["maximize"])
        self.weight = 1.0 if self.maximize else -1.0
        self.num_mv_samples = int(values["num_mv_samples"])
        self.use_gumbel, self._max_seed = bool(values["use_gumbel"]), values["seed"]
        if self.num_mv_samples < 1:
            raise ValueError("num_mv_samples must be positive.")
        self.project = values.get("project", lambda X: X)
        self.expand = values.get("expand", lambda X: X)
        if self._lower_bound and self._multi_fidelity and self.expand is not None:
            raise UnsupportedError("Multi-fidelity GIBBON does not support trace observations.")
        candidate = values["candidate_set"]
        if not isinstance(candidate, Tensor) or candidate.ndim != 2:
            raise ValueError("candidate_set must be an n x d Tensor.")
        if self._multi_fidelity:
            candidate = self.project(candidate)
        train = values.get("train_inputs")
        if train is None:
            train = getattr(self.model, "_train_X", None)
        if train is None and hasattr(self.model, "train_inputs"):
            train = self.model.train_inputs[0]
        self._dimension = candidate.shape[-1]
        from botorch_rs.acquisition.max_value_entropy_search import _candidate_train_inputs

        train = _candidate_train_inputs(self.model, train, self._dimension)
        self._validate_points(candidate, "candidate_set")
        if train is not None:
            self._validate_points(train, "train_inputs")
            candidate = torch.cat((candidate, train), 0)
        self.register_buffer("candidate_set", candidate)
        self.register_buffer("_base_max_values", None)
        self.register_buffer("_X_pending", None)
        # MaxValueBase draws before qMES allocates observation/fantasy seeds.
        self._sample_max_values(self.num_mv_samples, values["X_pending"])
        unused_seed = 0 if self._lower_bound and not self._multi_fidelity else None
        self.sampler = values.get("sampler")
        if self.sampler is None:
            observation_seed = (
                unused_seed
                if unused_seed is not None
                else int(torch.randint(0, 1_000_000, (1,)).item())
            )
            self.sampler = SobolQMCNormalSampler(
                (values.get("num_y_samples", 128),), seed=observation_seed
            )
        self.fantasies_sampler = values.get("fantasies_sampler")
        if self.fantasies_sampler is None:
            fantasy_seed = (
                unused_seed
                if unused_seed is not None
                else int(torch.randint(0, 1_000_000, (1,)).item())
            )
            self.fantasies_sampler = SobolQMCNormalSampler(
                (values.get("num_fantasies", 16),), seed=fantasy_seed
            )
        if len(self.sampler.sample_shape) != 1 or len(self.fantasies_sampler.sample_shape) != 1:
            raise UnsupportedError("MES samplers require one sample dimension.")
        self.num_y_samples, self.num_fantasies = (
            self.sampler.sample_shape[0],
            self.fantasies_sampler.sample_shape[0],
        )
        if not self._lower_bound and self.num_y_samples < 1:
            raise ValueError("MES requires a positive observation sample count.")
        self._cost_sampler = None
        self.cost_aware_utility = values.get("cost_aware_utility")
        if self._multi_fidelity and self.cost_aware_utility is None:
            from botorch_rs.acquisition.cost_aware import InverseCostWeightedUtility
            from botorch_rs.models.cost import AffineFidelityCostModel

            self.cost_aware_utility = InverseCostWeightedUtility(AffineFidelityCostModel())
        if self._multi_fidelity:
            # Multifidelity MES redraws after allocating its sampler seeds.
            self._sample_max_values(self.num_mv_samples, values["X_pending"])
        self.set_X_pending(values["X_pending"])

    def _validate_points(self, X: Any, name: str) -> None:
        like = next(self._init_model.parameters(), X)
        if not isinstance(X, Tensor) or X.ndim != 2 or X.shape[-1] != self._dimension:
            raise ValueError(f"{name} must be an n x d Tensor.")
        if X.device != like.device or X.dtype != like.dtype:
            raise ValueError(f"{name} must share the model's device and dtype.")

    @property
    def posterior_max_values(self) -> Tensor:
        return self._base_max_values

    @posterior_max_values.setter
    def posterior_max_values(self, values: Tensor) -> None:
        batch = tuple(self._init_model.batch_shape) or (1,)
        try:
            compatible = torch.broadcast_shapes(values.shape[1:], batch) == batch
        except RuntimeError:
            compatible = False
        if values.ndim < 2 or not compatible or not bool(torch.isfinite(values).all()):
            raise ValueError(
                "posterior_max_values must be finite with shape samples x model_batch."
            )
        self._base_max_values = values

    def _model_batch_rank(self) -> int:
        return max(1, len(self.model.batch_shape))

    def _model_batch_points(self, X: Tensor) -> Tensor:
        return X.reshape(*X.shape[:-2], *((1,) * self._model_batch_rank()), *X.shape[-2:])

    def _sample_max_values(self, num_samples: int, X_pending: Any = None) -> None:
        with torch.no_grad():
            candidates = self.candidate_set
            if X_pending is not None:
                self._validate_points(X_pending, "X_pending")
                candidates = torch.cat((candidates, X_pending), 0)
            if self._multi_fidelity:
                candidates = self.project(candidates)
            self.posterior_max_values = sample_maximums(
                self._init_model,
                candidates,
                num_samples,
                self.posterior_transform,
                self.maximize,
                gumbel=self.use_gumbel,
                seed=self._max_seed,
            )

    def set_X_pending(self, X_pending: Any = None) -> None:
        if X_pending is not None:
            self._validate_points(X_pending, "X_pending")
            X_pending = X_pending.detach().clone()
        self._X_pending = X_pending
        self.model = (
            self._init_model
            if X_pending is None or self._lower_bound
            else self._init_model.fantasize(X_pending, self.fantasies_sampler)
        )

    @property
    def cost_sampler(self) -> Any:
        if self._cost_sampler is None:
            self._cost_sampler = _clone_sampler(self.fantasies_sampler)
        return self._cost_sampler

    def _compute_information_gain(
        self, X: Tensor, mean_M: Tensor, variance_M: Tensor, covar_mM: Tensor
    ) -> Tensor:
        return (
            self._gibbon(X, mean_M, variance_M, covar_mM)
            if self._lower_bound
            else self._mes(X, mean_M, variance_M, covar_mM)
        )

    def _mes(self, X: Tensor, mean_M: Tensor, variance_M: Tensor, covar_mM: Tensor) -> Tensor:
        posterior = _gaussian(
            self.model, self._model_batch_points(X), self.posterior_transform, True
        )
        mean = self.weight * posterior.mean.squeeze(-1)
        covariance = posterior.distribution.covariance_matrix
        root = _rs.tensor_psd_safe_cholesky(covariance)
        samples = self.weight * self.sampler(posterior).squeeze(-1)
        conditional_weight = torch.cholesky_solve(covar_mM.unsqueeze(-1), root).transpose(-1, -2)
        conditional_mean = (conditional_weight @ (samples - mean).unsqueeze(-1)).squeeze(
            -1
        ).squeeze(-1) + mean_M
        conditional_variance = variance_M - (conditional_weight @ covar_mM.unsqueeze(-1)).squeeze(
            -1
        ).squeeze(-1)
        conditional_std = conditional_variance.clamp_min(CLAMP_LB).sqrt()
        maximum = (
            self.posterior_max_values.to(X)
            .reshape(
                self.posterior_max_values.shape[0],
                *((1,) * (mean_M.ndim - self.posterior_max_values.ndim + 1)),
                *self.posterior_max_values.shape[1:],
            )
            .unsqueeze(1)
        )
        cdf_conditional = torch.special.ndtr(
            (maximum - conditional_mean) / conditional_std
        ).clamp_min(CLAMP_LB)
        cdf = torch.special.ndtr((maximum - mean_M) / variance_M.sqrt()).clamp_min(CLAMP_LB)
        difference = samples - mean
        whitened = torch.linalg.solve_triangular(
            root, difference.unsqueeze(-1), upper=False
        ).squeeze(-1)
        log_det = 2 * root.diagonal(dim1=-2, dim2=-1).log().sum(-1)
        log_density = (
            -0.5 * (X.shape[-2] * LOG_2PI + log_det + whitened.square().sum(-1))
        ).unsqueeze(0)
        entropy = 0.5 * (X.shape[-2] * (1 + LOG_2PI) + log_det)
        ratio = cdf_conditional / cdf
        h1 = -ratio * (ratio.log() + log_density)
        h0 = -log_density
        first, zero = h1.mean((0, 1)), h0.mean((0, 1))
        cov = ((h1 - first) * (h0 - zero)).mean((0, 1))
        if h0.shape[0] * h0.shape[1] <= 1:
            # Unbiased variance with one observation has undefined gradients,
            # even if its NaN value is masked afterwards. Do not evaluate it.
            coefficient = torch.zeros_like(entropy)
        else:
            product = h0.var((0, 1)) * h1.var((0, 1))
            positive = product > 0
            denominator = torch.where(positive, product, 1).sqrt()
            coefficient = torch.where(positive, cov / denominator, 0)
        gain = entropy - first + coefficient * (zero - entropy)
        rank = self._model_batch_rank()
        return gain.movedim(tuple(range(-rank, 0)), tuple(range(rank))).flatten(0, rank - 1)

    def _gibbon(self, X: Tensor, mean_M: Tensor, variance_M: Tensor, covar_mM: Tensor) -> Tensor:
        noisy = _gaussian(self.model, X, self.posterior_transform, True)
        variance = noisy.variance.clamp_min(CLAMP_LB).squeeze(-1)
        z = (self.posterior_max_values.to(X).transpose(0, 1) - mean_M) / variance_M.sqrt()
        ratio = (-0.5 * z.square() - 0.5 * LOG_2PI).exp() / torch.special.ndtr(z).clamp_min(
            CLAMP_LB
        )
        squared_correlation = covar_mM.squeeze(-1).square() / (variance * variance_M)
        quality = (
            (-0.5 * (1 - squared_correlation * ratio * (z + ratio)).clamp_min(CLAMP_LB).log())
            .mean(-1)
            .unsqueeze(0)
        )
        if self.X_pending is None or self.X_pending.shape[-2] == 0:
            return quality
        pending = self.X_pending.expand(*X.shape[:-2], *self.X_pending.shape[-2:])
        covariance = _gaussian(
            self.model, torch.cat((X, pending), -2), self.posterior_transform, True
        ).distribution.covariance_matrix
        cross, block = covariance[..., :1, 1:], covariance[..., 1:, 1:]
        root = _rs.tensor_psd_safe_cholesky(block)
        inv_quad = (cross @ torch.cholesky_solve(cross.transpose(-1, -2), root)).squeeze(-1)
        eps = 1e-6 if X.dtype == torch.float32 else 1e-12
        conditional = variance.clamp_min(inv_quad * (1 + eps)) - inv_quad
        repulsion = 0.5 * (conditional.log() - variance.log()).squeeze(-1).unsqueeze(0)
        return quality + repulsion

    def forward(self, X: Tensor) -> Tensor:
        if not isinstance(X, Tensor) or X.ndim < 2 or X.shape[-2] != 1:
            raise ValueError("Max-value entropy candidates must have shape batch x 1 x d.")
        X = X.unsqueeze(0) if X.ndim == 2 else X
        evaluated = self.expand(X) if self._multi_fidelity and self.expand is not None else X
        if self._multi_fidelity:
            target = self.project(X)
            joint = self._model_batch_points(torch.cat((evaluated, target), -2))
            posterior = _gaussian(self.model, joint, self.posterior_transform)
            mean = self.weight * posterior.mean[..., -1, 0]
            variance = posterior.variance[..., -1, 0].clamp_min(CLAMP_LB)
            cross = posterior.distribution.covariance_matrix[..., :-1, -1]
        else:
            posterior = _gaussian(self.model, self._model_batch_points(X), self.posterior_transform)
            mean = self.weight * posterior.mean[..., 0, 0]
            variance = posterior.variance[..., 0, 0].clamp_min(CLAMP_LB)
            cross = variance.unsqueeze(-1)
        gain = self._compute_information_gain(evaluated, mean, variance, cross)
        if self._multi_fidelity:
            gain = self.cost_aware_utility(X=X, deltas=gain, sampler=self.cost_sampler)
        return gain.mean(0)

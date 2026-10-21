"""Whitened sparse variational GPs with device-resident Torch graphs."""

from __future__ import annotations

import copy
import math
import weakref
from typing import Any

import torch
from torch import Tensor, nn

from botorch_rs import _botorch_rs as _rs
from botorch_rs.exceptions.errors import UnsupportedError
from botorch_rs.mlls import VariationalELBO
from botorch_rs.models._torch_backend_kronecker import (
    TorchMultitaskGaussianLikelihood,
    tensor_kron,
)
from botorch_rs.models._torch_backend_modules import (
    build_input_transform,
    build_kernel,
    build_likelihood,
    build_mean,
    build_outcome_transform,
    module_log_prior,
)
from botorch_rs.models.approximate_gp import SingleTaskVariationalGP
from botorch_rs.posteriors._torch_backend import TorchGPyTorchPosterior, independent_covariance
from botorch_rs.variational import CholeskyVariationalDistribution, VariationalStrategy


def _dense(value: Any) -> Tensor:
    return value if isinstance(value, Tensor) else value.to_dense()


class _TorchCholeskyDistribution(nn.Module):
    def __init__(self, m: int, batch_shape: Any, like: Tensor, source: Any) -> None:
        super().__init__()
        self.num_inducing_points = m
        self.batch_shape = torch.Size(batch_shape)
        self.mean_init_std = 1e-3 if source is None else source.mean_init_std
        self.variational_mean = nn.Parameter(like.new_zeros(*batch_shape, m))
        self.chol_variational_covar = nn.Parameter(
            torch.eye(m, dtype=like.dtype, device=like.device).expand(*batch_shape, m, m).clone()
        )

    def initialize_variational_distribution(self, prior_dist: Any = None) -> None:
        with torch.no_grad():
            self.variational_mean.copy_(
                torch.randn_like(self.variational_mean) * self.mean_init_std
            )
            self.chol_variational_covar.copy_(
                torch.eye(
                    self.num_inducing_points,
                    dtype=self.variational_mean.dtype,
                    device=self.variational_mean.device,
                )
            )

    def forward(self) -> Any:
        factor = self.chol_variational_covar.tril()
        return TorchGPyTorchPosterior(
            self.variational_mean.unsqueeze(-1), factor @ factor.mT
        ).distribution


class _TorchVariationalStrategy(nn.Module):
    _tensor_backend = "torch"
    _owner_ref: Any
    inducing_points: Tensor
    variational_params_initialized: Tensor
    _variational_distribution: _TorchCholeskyDistribution

    def __init__(self, owner: Any, points: Tensor, distribution: Any, learn: bool) -> None:
        super().__init__()
        object.__setattr__(self, "_owner_ref", weakref.ref(owner))
        self.learn_inducing_locations = learn
        if learn:
            self.inducing_points = nn.Parameter(points.detach().clone())
        else:
            self.register_buffer("inducing_points", points.clone())
        self._variational_distribution = distribution
        self.register_buffer(
            "variational_params_initialized",
            torch.tensor(0, dtype=torch.uint8, device=points.device),
        )
        self._jitter_val: float | None = None

    @property
    def jitter_val(self) -> float:
        return (
            (1e-4 if self.inducing_points.dtype == torch.float32 else 1e-6)
            if self._jitter_val is None
            else self._jitter_val
        )

    @jitter_val.setter
    def jitter_val(self, value: float) -> None:
        if not math.isfinite(value) or value <= 0:
            raise ValueError("Variational jitter must be finite and positive.")
        self._jitter_val = value

    @property
    def variational_distribution(self) -> Any:
        return self._variational_distribution()

    @property
    def prior_distribution(self) -> Any:
        mean = torch.zeros_like(self._variational_distribution.variational_mean)
        n = mean.shape[-1]
        return TorchGPyTorchPosterior(
            mean.unsqueeze(-1),
            torch.eye(n, dtype=mean.dtype, device=mean.device).expand(*mean.shape[:-1], n, n),
        ).distribution

    def _initialize(self) -> None:
        if not bool(self.variational_params_initialized):
            self._variational_distribution.initialize_variational_distribution()
            self.variational_params_initialized.fill_(1)

    def kl_divergence(self) -> Tensor:
        mean = self._variational_distribution.variational_mean
        factor = self._variational_distribution.chol_variational_covar.tril()
        return 0.5 * (
            mean.square().sum(-1)
            + factor.square().sum((-1, -2))
            - mean.shape[-1]
            - 2 * factor.diagonal(dim1=-2, dim2=-1).abs().log().sum(-1)
        )

    def forward(self, X: Tensor, prior: bool = False) -> Any:
        owner = self._owner_ref()
        if prior:
            return owner._prior_output(X, multitask=owner.num_outputs > 1)
        self._initialize()
        return owner._variational_output(X)


class _TorchIndependentStrategy(nn.Module):
    base_variational_strategy: Any

    def __init__(self, strategy: Any) -> None:
        super().__init__()
        self.base_variational_strategy = strategy

    def kl_divergence(self) -> Tensor:
        return self.base_variational_strategy.kl_divergence().sum(-1)

    def forward(self, X: Tensor, prior: bool = False) -> Any:
        return self.base_variational_strategy(X, prior=prior)


class _TorchVariationalInner(nn.Module):
    _tensor_backend = "torch"
    _outer_ref: Any
    variational_strategy: Any
    mean_module: Any
    covar_module: Any

    def __init__(self, owner: Any, covariance: Any, mean: Any, strategy: Any) -> None:
        super().__init__()
        object.__setattr__(self, "_outer_ref", weakref.ref(owner))
        self.variational_strategy = strategy
        self.mean_module = mean
        self.covar_module = covariance

    @property
    def _outer(self) -> Any:
        return self._outer_ref()

    @property
    def train_inputs(self) -> tuple[Tensor]:
        return (self._outer._training_inputs(),)

    @property
    def train_targets(self) -> Tensor:
        target = self._outer._training_targets()
        return target.squeeze(-1) if self._outer.num_outputs == 1 else target

    def __call__(self, X: Tensor, prior: bool = False) -> Any:
        return self.variational_strategy(X, prior=prior)

    def forward(self, X: Tensor) -> Any:
        return self._outer._prior_output(X, multitask=False)


class _TorchMultitaskNoise(TorchMultitaskGaussianLikelihood):
    def __init__(self, source: Any, like: Tensor) -> None:
        super().__init__(source.num_tasks, source.rank, (), like, source)

    def noise_for(self, n: int) -> Tensor:
        diagonal = self.noise_matrix().diagonal(dim1=-2, dim2=-1)
        return diagonal.unsqueeze(-1).expand(*diagonal.shape, n)

    def forward(self, distribution: Any, noise: Tensor | None = None, **kwargs: Any) -> Any:
        if kwargs:
            raise UnsupportedError(f"Unsupported multitask likelihood arguments: {sorted(kwargs)}.")
        mean = distribution if isinstance(distribution, Tensor) else distribution.mean
        n = mean.shape[-2]
        variances = self.noise_for(n).transpose(-1, -2) if noise is None else noise
        if isinstance(distribution, Tensor):
            return torch.distributions.Independent(
                torch.distributions.Normal(mean, variances.sqrt()), 1
            )
        covariance = distribution.covariance_matrix
        if getattr(distribution, "_interleaved", True):
            m = self.num_tasks
            covariance = torch.einsum(
                "...itju->...tiuj", covariance.reshape(*covariance.shape[:-2], n, m, n, m)
            ).reshape(*covariance.shape[:-2], m * n, m * n)
        addition = (
            tensor_kron(self.noise_matrix(), torch.eye(n, device=mean.device, dtype=mean.dtype))
            if noise is None
            else torch.diag_embed(variances.transpose(-1, -2).flatten(-2))
        )
        covariance = covariance + addition
        return TorchGPyTorchPosterior(mean, covariance).distribution


class _TorchVariationalOutput:
    """Saved q(f), with O(NM) marginals and lazily materialized covariance."""

    _interleaved = True

    def __init__(
        self,
        owner: Any,
        mean: Tensor,
        variance: Tensor,
        covariance: Any,
        prior: bool = False,
        independent: bool = False,
    ) -> None:
        self._owner = owner
        self._mean = mean
        self._variance = variance
        self._covariance = covariance
        self._cached_blocks: Tensor | None = None
        self._is_prior = prior
        self._independent = independent
        self._multitask = owner.num_outputs > 1 and not independent
        if not prior:
            self._kl = owner.model.variational_strategy.kl_divergence()
            self._log_prior = owner._log_prior()
            self._noise = owner.likelihood.noise_for(mean.shape[-1]).clone()

    @property
    def mean(self) -> Tensor:
        return self._mean.transpose(-1, -2) if self._multitask else self._mean

    @property
    def variance(self) -> Tensor:
        minimum = 1e-6 if self._variance.dtype == torch.float32 else 1e-10
        value = self._variance.clamp_min(minimum)
        return value.transpose(-1, -2) if self._multitask else value

    @property
    def loc(self) -> Tensor:
        return self.mean.flatten(-2) if self._multitask else self.mean

    @property
    def batch_shape(self) -> torch.Size:
        return self._variance.shape[:-2] if self._multitask else self._variance.shape[:-1]

    @property
    def event_shape(self) -> torch.Size:
        return self.mean.shape[-2:] if self._multitask else self.mean.shape[-1:]

    def _blocks(self) -> Tensor:
        if self._cached_blocks is None:
            self._cached_blocks = self._covariance()
        return self._cached_blocks

    @property
    def covariance_matrix(self) -> Tensor:
        if not self._multitask:
            return self._blocks()
        cov = independent_covariance(self._blocks())
        m, n = self._mean.shape[-2:]
        return torch.einsum("...tiuj->...itju", cov.reshape(*cov.shape[:-2], m, n, m, n)).reshape(
            *cov.shape[:-2], m * n, m * n
        )

    def posterior(self) -> TorchGPyTorchPosterior:
        mean = self.mean if self._multitask else self.mean.unsqueeze(-1)
        covariance = independent_covariance(self._blocks()) if self._multitask else self._blocks()
        return TorchGPyTorchPosterior(mean, covariance)

    def rsample(self, sample_shape: Any = ()) -> Tensor:
        samples = self.posterior().rsample(sample_shape)
        return samples if self._multitask else samples.squeeze(-1)


class TorchSingleTaskVariationalGP(nn.Module, SingleTaskVariationalGP):
    _tensor_backend = "torch"
    _train_X: Tensor
    _train_Y: Tensor | None
    model: Any
    likelihood: Any
    input_transform: Any
    outcome_transform: Any

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor | None = None,
        likelihood: Any = None,
        num_outputs: int = 1,
        learn_inducing_points: bool = True,
        covar_module: Any = None,
        mean_module: Any = None,
        variational_distribution: Any = None,
        variational_strategy: Any = VariationalStrategy,
        inducing_points: Any = None,
        inducing_point_allocator: Any = None,
        outcome_transform: Any = None,
        input_transform: Any = None,
        *,
        backend: str = "torch",
    ) -> None:
        from botorch_rs.likelihoods import GaussianLikelihood, MultitaskGaussianLikelihood
        from botorch_rs.models._torch_backend import TorchSingleTaskGP
        from botorch_rs.models.utils.inducing_point_allocators import GreedyVarianceReduction

        nn.Module.__init__(self)
        if (
            not isinstance(train_X, Tensor)
            or train_X.ndim < 2
            or train_X.dtype not in (torch.float32, torch.float64)
            or not bool(torch.isfinite(train_X).all())
        ):
            raise ValueError(
                "Torch variational training inputs must be finite float32/float64 Tensors (...,n,d)."
            )
        if train_Y is not None:
            TorchSingleTaskGP._validate_data(train_X, train_Y)
            num_outputs = train_Y.shape[-1]
        elif outcome_transform is not None:
            raise ValueError("An outcome transform requires training targets.")
        if num_outputs <= 0:
            raise ValueError("num_outputs must be positive.")
        if variational_strategy is not VariationalStrategy:
            raise UnsupportedError(
                "Torch variational models currently support the whitened VariationalStrategy only."
            )
        if (
            variational_distribution is not None
            and type(variational_distribution) is not CholeskyVariationalDistribution
        ):
            raise UnsupportedError(
                "Torch variational models currently support CholeskyVariationalDistribution only."
            )
        self._inducing_point_allocator = (
            GreedyVarianceReduction()
            if inducing_point_allocator is None
            else inducing_point_allocator
        )
        if not callable(getattr(self._inducing_point_allocator, "allocate_inducing_points", None)):
            raise TypeError("inducing_point_allocator must implement allocate_inducing_points.")
        self._desired_num_outputs = int(num_outputs)
        self.register_buffer("_train_X", train_X)
        self.register_buffer("_train_Y", train_Y)
        self.input_transform = build_input_transform(input_transform, train_X.shape[-1], train_X)
        self.outcome_transform = (
            None
            if train_Y is None
            else build_outcome_transform(outcome_transform, num_outputs, train_Y)
        )
        covariance = build_kernel(covar_module, train_X.shape[-1], self._aug_batch_shape, train_X)
        mean = build_mean(mean_module, self._aug_batch_shape, train_X)
        if likelihood is None and num_outputs > 1:
            likelihood = MultitaskGaussianLikelihood(num_outputs)
        if num_outputs > 1:
            if (
                type(likelihood) is not MultitaskGaussianLikelihood
                or likelihood.num_tasks != num_outputs
            ):
                raise UnsupportedError(
                    "Multiple variational outputs require a MultitaskGaussianLikelihood with the matching number of tasks."
                )
            self.likelihood = _TorchMultitaskNoise(likelihood, train_X)
        else:
            if likelihood is not None and type(likelihood) is not GaussianLikelihood:
                raise UnsupportedError(
                    "The Torch variational backend currently supports Gaussian observations only."
                )
            self.likelihood = build_likelihood(likelihood, None, self.batch_shape, train_X)
        transformed = self._training_inputs()
        points = (
            self._inducing_point_allocator.allocate_inducing_points(
                inputs=transformed,
                covar_module=covariance,
                num_inducing=(
                    int(0.25 * train_X.shape[-2]) if inducing_points is None else inducing_points
                ),
                input_batch_shape=self.batch_shape,
            )
            if inducing_points is None or isinstance(inducing_points, int)
            else inducing_points
        )
        if (
            not isinstance(points, Tensor)
            or points.ndim < 2
            or points.shape[-1] != train_X.shape[-1]
            or not points.shape[-2]
            or points.dtype != train_X.dtype
            or points.device != train_X.device
            or not bool(torch.isfinite(points).all())
        ):
            raise ValueError(
                "Inducing points must be finite nonempty Tensors with the training device, dtype and feature dimension."
            )
        point_batch = points.shape[:-2]
        if num_outputs > 1 and points.ndim != train_X.ndim + 1:
            point_batch = point_batch + (1,)
        if torch.broadcast_shapes(point_batch, self._aug_batch_shape) != self._aug_batch_shape:
            raise ValueError(
                "Inducing-point batch dimensions must broadcast to the augmented model batch shape."
            )
        if variational_distribution is not None and (
            variational_distribution.num_inducing_points != points.shape[-2]
            or tuple(variational_distribution.batch_shape) != tuple(self._aug_batch_shape)
        ):
            raise ValueError(
                "Variational distribution dimensions must match the inducing points and augmented model batch shape."
            )
        distribution = _TorchCholeskyDistribution(
            points.shape[-2], self._aug_batch_shape, train_X, variational_distribution
        )
        strategy = _TorchVariationalStrategy(self, points, distribution, learn_inducing_points)
        self.model = _TorchVariationalInner(
            self,
            covariance,
            mean,
            _TorchIndependentStrategy(strategy) if num_outputs > 1 else strategy,
        )

    @property
    def batch_shape(self) -> Any:
        return self._train_X.shape[:-2]

    @property
    def _aug_batch_shape(self) -> Any:
        return self.batch_shape + ((self.num_outputs,) if self.num_outputs > 1 else ())

    @property
    def _strategy(self) -> Any:
        strategy = self.model.variational_strategy
        return strategy.base_variational_strategy if self.num_outputs > 1 else strategy

    def _augment(self, X: Tensor) -> Tensor:
        return X.unsqueeze(-3) if self.num_outputs > 1 else X

    def _fit_transforms(self) -> None:
        for transform, values in (
            (self.input_transform, self._train_X),
            (self.outcome_transform, self._train_Y),
        ):
            if transform is not None:
                if (
                    values is not None
                    and values.requires_grad
                    and torch.is_grad_enabled()
                    and getattr(transform, "_tensor_backend", None) == "torch"
                ):
                    transform.train()
                    transform(values)
                transform.eval()

    def _training_inputs(self) -> Tensor:
        self._fit_transforms()
        if self.input_transform is None:
            return self._train_X
        return getattr(self.input_transform, "preprocess_transform", self.input_transform)(
            self._train_X
        )

    def _ensure_initialized(self) -> None:
        self._strategy._initialize()

    def init_inducing_points(self, inputs: Tensor) -> Tensor:
        points = self._inducing_point_allocator.allocate_inducing_points(
            inputs=inputs,
            covar_module=self.model.covar_module,
            num_inducing=self._strategy.inducing_points.shape[-2],
            input_batch_shape=self.batch_shape,
        )
        if points.shape != self._strategy.inducing_points.shape:
            raise ValueError("Reallocated inducing points must retain their shape.")
        with torch.no_grad():
            self._strategy.inducing_points.copy_(points)
            self._strategy.variational_params_initialized.fill_(0)
        return points

    def __deepcopy__(self, memo: dict[int, Any]) -> Any:
        new = object.__new__(type(self))
        memo[id(self)] = new
        for name, value in self.__dict__.items():
            new.__dict__[name] = value.copy() if name == "_buffers" else copy.deepcopy(value, memo)
        object.__setattr__(new.model, "_outer_ref", weakref.ref(new))
        object.__setattr__(new._strategy, "_owner_ref", weakref.ref(new))
        return new

    def _training_targets(self) -> Tensor:
        if self._train_Y is None:
            raise ValueError("Training targets were not supplied; pass targets to VariationalELBO.")
        self._fit_transforms()
        return (
            self._train_Y
            if self.outcome_transform is None
            else self.outcome_transform(self._train_Y)[0]
        )

    def _log_prior(self) -> Any:
        result = self._train_X.new_zeros(())
        for module in (
            self.model.covar_module,
            self.model.mean_module,
            self.likelihood,
            self.input_transform,
        ):
            if module is not None:
                value = module_log_prior(module, self._aug_batch_shape)
                result = result + (value.sum() if isinstance(value, Tensor) else value)
        return result

    def _prior_output(self, X: Tensor, multitask: bool) -> Any:
        augmented = self._augment(X)
        mean = self.model.mean_module(augmented).clone()
        covariance = _dense(self.model.covar_module(augmented, augmented))
        if multitask:
            mean = mean.expand(covariance.shape[:-1])
        return _TorchVariationalOutput(
            self,
            mean,
            covariance.diagonal(dim1=-2, dim2=-1),
            lambda: covariance,
            prior=True,
            independent=not multitask,
        )

    def _variational_output(self, X: Tensor) -> Any:
        X = X.clone()
        augmented = self._augment(X)
        strategy = self._strategy
        points = strategy.inducing_points
        inducing = points if points.ndim == self._train_X.ndim + 1 else self._augment(points)
        kernel = self.model.covar_module
        M, N = points.shape[-2], X.shape[-2]
        jitter = strategy.jitter_val
        eye = torch.eye(M, dtype=X.dtype, device=X.device)
        root = _rs.tensor_psd_safe_cholesky(_dense(kernel(inducing, inducing)) + jitter * eye)
        interpolation = torch.linalg.solve_triangular(
            root, _dense(kernel(inducing, augmented)), upper=False
        )
        distribution = strategy._variational_distribution
        factor = distribution.chol_variational_covar.tril()
        middle = factor @ factor.mT - eye
        mean = self.model.mean_module(augmented) + (
            interpolation.mT @ distribution.variational_mean.unsqueeze(-1)
        ).squeeze(-1)
        update = middle @ interpolation
        variance_update = (interpolation * update).sum(-2)
        variance = _dense(kernel(augmented, augmented, diag=True)) + jitter + variance_update
        parameters = {name: value.clone() for name, value in kernel.named_parameters()}
        buffers = {name: value.clone() for name, value in kernel.named_buffers()}
        diagonal_only = self.model.training and self.num_outputs == 1 and N > M

        def covariance() -> Tensor:
            prior = _dense(
                torch.func.functional_call(kernel, (parameters, buffers), (augmented, augmented))
            )
            delta = (
                torch.diag_embed(variance_update) if diagonal_only else interpolation.mT @ update
            )
            return prior + delta + jitter * torch.eye(N, dtype=X.dtype, device=X.device)

        return _TorchVariationalOutput(self, mean, variance, covariance)

    def forward(self, X: Tensor) -> Any:
        if self.training and self.input_transform is not None:
            X = self.input_transform(X)
        return self.model(X)

    def posterior(
        self,
        X: Tensor,
        output_indices: Any = None,
        observation_noise: Any = False,
        posterior_transform: Any = None,
        **kwargs: Any,
    ) -> Any:
        if kwargs:
            raise UnsupportedError(
                f"Unsupported variational posterior arguments: {sorted(kwargs)}."
            )
        if (
            not isinstance(X, Tensor)
            or X.dtype != self._train_X.dtype
            or X.device != self._train_X.device
            or X.ndim < 2
            or X.shape[-1] != self._train_X.shape[-1]
        ):
            raise ValueError(
                "Query inputs must match the model's device, dtype and feature dimension."
            )
        self.eval()
        self._fit_transforms()
        transformed = X if self.input_transform is None else self.input_transform(X)
        output = self.model(transformed)
        blocks = output._blocks()
        covariance = independent_covariance(blocks) if self.num_outputs > 1 else blocks
        if observation_noise is not False:
            if observation_noise is True and self.num_outputs > 1:
                covariance = covariance + tensor_kron(
                    self.likelihood.noise_matrix(),
                    torch.eye(X.shape[-2], device=X.device, dtype=X.dtype),
                )
            else:
                noise = (
                    self.likelihood.noise_for(X.shape[-2])
                    if observation_noise is True
                    else observation_noise.movedim(-1, -2)
                    if self.num_outputs > 1
                    else observation_noise.squeeze(-1)
                )
                covariance = covariance + torch.diag_embed(
                    noise.flatten(-2) if self.num_outputs > 1 else noise
                )
        mean = output.mean if self.num_outputs > 1 else output.mean.unsqueeze(-1)
        posterior = TorchGPyTorchPosterior(mean, covariance)
        transform = self.outcome_transform
        if output_indices is not None:
            posterior = posterior.subset_output(output_indices)
            if transform is not None:
                transform = transform.subset_output(output_indices)
        if transform is not None:
            posterior = transform.untransform_posterior(posterior, X=X)
        return posterior if posterior_transform is None else posterior_transform(posterior, X=X)


class TorchVariationalELBO(nn.Module, VariationalELBO):
    _tensor_backend = "torch"

    def __init__(
        self,
        likelihood: Any,
        model: Any,
        num_data: int,
        beta: float = 1,
        combine_terms: bool = True,
    ) -> None:
        nn.Module.__init__(self)
        if (
            not isinstance(model, _TorchVariationalInner)
            or likelihood is not model._outer.likelihood
        ):
            raise UnsupportedError(
                "Torch VariationalELBO requires this variational model's inner model and likelihood."
            )
        if not math.isfinite(num_data) or num_data <= 0 or not math.isfinite(beta) or beta < 0:
            raise ValueError("num_data must be positive and beta nonnegative, both finite.")
        self.likelihood = likelihood
        self.model = model
        self.num_data, self.beta, self.combine_terms = num_data, beta, combine_terms

    def forward(
        self, approximate_dist_f: Any = None, target: Tensor | None = None, **kwargs: Any
    ) -> Any:
        if kwargs:
            raise UnsupportedError(f"Unsupported Torch ELBO arguments: {sorted(kwargs)}.")
        owner = self.model._outer
        if approximate_dist_f is None:
            approximate_dist_f = self.model(self.model.train_inputs[0])
            target = self.model.train_targets if target is None else target
        if (
            not isinstance(approximate_dist_f, _TorchVariationalOutput)
            or approximate_dist_f._owner is not owner
            or approximate_dist_f._is_prior
        ):
            raise UnsupportedError(
                "VariationalELBO requires q(f) from this model, not a prior law."
            )
        if target is None:
            raise TypeError("missing target values")
        output = approximate_dist_f
        mean, variance = (
            output._mean,
            output._variance.clamp_min(1e-6 if output._mean.dtype == torch.float32 else 1e-10),
        )
        if owner.num_outputs == 1:
            y = target.squeeze(-1) if target.ndim == mean.ndim + 1 else target
        else:
            y = target.transpose(-1, -2)
        if y.shape != mean.shape:
            raise ValueError("ELBO targets must match the q(f) event and batch shapes.")
        log_likelihood = -0.5 * (
            ((y - mean).square() + variance) / output._noise
            + output._noise.log()
            + math.log(2 * math.pi)
        ).mean(-1)
        if owner.num_outputs > 1:
            log_likelihood = log_likelihood.sum(-1)
        kl = output._kl * (self.beta / self.num_data)
        prior = torch.zeros_like(log_likelihood) + output._log_prior / self.num_data
        return log_likelihood - kl + prior if self.combine_terms else (log_likelihood, kl, prior)

    def value_and_grad(self, raw_parameters: Any = None) -> tuple[float, list[float]]:
        """Summed ELBO and current trainable parameter derivatives.

        The returned flat gradient follows ``named_parameters()``; frozen
        parameters contribute zeros. Optimizers should use actual Parameters
        rather than this detached diagnostic extension.
        """
        if raw_parameters is not None:
            raise UnsupportedError(
                "The Torch ELBO value_and_grad evaluates current Parameters; "
                "assign parameter values explicitly before evaluating."
            )
        if not self.combine_terms:
            raise UnsupportedError("value_and_grad requires combine_terms=True.")
        parameters = tuple(self.parameters())
        trainable = tuple(parameter for parameter in parameters if parameter.requires_grad)
        value = self().sum()
        gradients = (
            torch.autograd.grad(value, trainable, allow_unused=True)
            if trainable and value.requires_grad
            else (None,) * len(trainable)
        )
        by_identity = dict(zip(map(id, trainable), gradients, strict=True))
        parts = []
        for parameter in parameters:
            gradient = by_identity.get(id(parameter))
            parts.append(
                (torch.zeros_like(parameter) if gradient is None else gradient).reshape(-1)
            )
        flat = torch.cat(parts).detach().cpu().tolist() if parts else []
        return float(value.detach()), flat

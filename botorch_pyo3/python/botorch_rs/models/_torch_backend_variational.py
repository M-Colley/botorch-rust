"""Whitened sparse variational GPs with device-resident Torch graphs."""

from __future__ import annotations

import copy
import functools
import math
import sys
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
from botorch_rs.variational import (
    CholeskyVariationalDistribution,
    DeltaVariationalDistribution,
    MeanFieldVariationalDistribution,
    UnwhitenedVariationalStrategy,
    VariationalStrategy,
)


def _dense(value: Any) -> Tensor:
    return value if isinstance(value, Tensor) else value.to_dense()


def _distribution_family(source: Any) -> str:
    if source is None:
        return "cholesky"
    families = {
        "CholeskyVariationalDistribution": CholeskyVariationalDistribution,
        "MeanFieldVariationalDistribution": MeanFieldVariationalDistribution,
        "DeltaVariationalDistribution": DeltaVariationalDistribution,
    }
    gpytorch = sys.modules.get("gpytorch")
    if gpytorch is not None and isinstance(source, gpytorch.variational._VariationalDistribution):
        # Keep supplied modules themselves: Natural distributions define a
        # deliberate natural-gradient VJP, and subclasses may add parameters
        # or change their forward law.
        return "external"
    for name, own_class in families.items():
        if type(source) is own_class:
            return name.removesuffix("VariationalDistribution").lower()
    raise UnsupportedError(
        "Torch variational models require a port Cholesky/MeanField/Delta specification "
        "or a GPyTorch VariationalDistribution module."
    )


def _distribution_parameter(source: Any, name: str, default: Tensor) -> nn.Parameter:
    value = default if source is None else getattr(source, name)
    if isinstance(value, nn.Parameter):
        # A caller's actual module retains the same leaves, like the reference.
        return value
    return nn.Parameter(torch.as_tensor(value, dtype=default.dtype, device=default.device).clone())


class _TorchBaseDistribution(nn.Module):
    def __init__(self, m: int, batch_shape: Any, like: Tensor, source: Any) -> None:
        super().__init__()
        if isinstance(source, nn.Module):
            source.to(like)
        self.num_inducing_points = m
        self.batch_shape = torch.Size(batch_shape)
        self.mean_init_std = 1e-3 if source is None else source.mean_init_std
        self.variational_mean = _distribution_parameter(
            source, "variational_mean", like.new_zeros(*batch_shape, m)
        )

    def initialize_variational_distribution(self, prior_dist: Any = None) -> None:
        with torch.no_grad():
            prior_mean = 0 if prior_dist is None else prior_dist.mean
            self.variational_mean.copy_(
                prior_mean + torch.randn_like(self.variational_mean) * self.mean_init_std
            )

    def covariance_matrix(self) -> Tensor:
        raise NotImplementedError

    def sample_with_base(self, noise: Tensor) -> Tensor:
        raise NotImplementedError

    def kl_divergence(self) -> Tensor:
        raise NotImplementedError


class _TorchCholeskyDistribution(_TorchBaseDistribution):
    def __init__(self, m: int, batch_shape: Any, like: Tensor, source: Any) -> None:
        super().__init__(m, batch_shape, like, source)
        self.chol_variational_covar = _distribution_parameter(
            source,
            "chol_variational_covar",
            torch.eye(m, dtype=like.dtype, device=like.device).expand(*batch_shape, m, m),
        )

    def initialize_variational_distribution(self, prior_dist: Any = None) -> None:
        super().initialize_variational_distribution(prior_dist)
        with torch.no_grad():
            self.chol_variational_covar.copy_(
                torch.eye(
                    self.num_inducing_points,
                    dtype=self.variational_mean.dtype,
                    device=self.variational_mean.device,
                )
                if prior_dist is None
                else _rs.tensor_psd_safe_cholesky(prior_dist.covariance_matrix)
            )

    def forward(self) -> Any:
        return TorchGPyTorchPosterior(
            self.variational_mean.unsqueeze(-1), self.covariance_matrix()
        ).distribution

    def covariance_matrix(self) -> Tensor:
        factor = self.chol_variational_covar.tril()
        return factor @ factor.mT

    def sample_with_base(self, noise: Tensor) -> Tensor:
        return self.variational_mean + (
            self.chol_variational_covar.tril() @ noise.unsqueeze(-1)
        ).squeeze(-1)

    def kl_divergence(self) -> Tensor:
        factor = self.chol_variational_covar.tril()
        return 0.5 * (
            self.variational_mean.square().sum(-1)
            + factor.square().sum((-1, -2))
            - self.num_inducing_points
            - 2 * factor.diagonal(dim1=-2, dim2=-1).abs().log().sum(-1)
        )


class _TorchMeanFieldDistribution(_TorchBaseDistribution):
    def __init__(self, m: int, batch_shape: Any, like: Tensor, source: Any) -> None:
        super().__init__(m, batch_shape, like, source)
        self._variational_stddev = _distribution_parameter(
            source, "_variational_stddev", like.new_ones(*batch_shape, m)
        )

    @property
    def variational_stddev(self) -> Tensor:
        return self._variational_stddev.abs().clamp_min(1e-8)

    def initialize_variational_distribution(self, prior_dist: Any = None) -> None:
        super().initialize_variational_distribution(prior_dist)
        with torch.no_grad():
            if prior_dist is None:
                self._variational_stddev.fill_(1)
            else:
                self._variational_stddev.copy_(prior_dist.stddev)

    def covariance_matrix(self) -> Tensor:
        return torch.diag_embed(self._variational_stddev.square())

    def forward(self) -> Any:
        return TorchGPyTorchPosterior(
            self.variational_mean.unsqueeze(-1), self.covariance_matrix()
        ).distribution

    def sample_with_base(self, noise: Tensor) -> Tensor:
        return self.variational_mean + self._variational_stddev.abs() * noise

    def kl_divergence(self) -> Tensor:
        variance = self._variational_stddev.square()
        return 0.5 * (self.variational_mean.square() + variance - 1 - variance.log()).sum(-1)


class _TorchPointMass(torch.distributions.Distribution):
    arg_constraints: dict[str, Any] = {}
    has_rsample = True

    def __init__(self, value: Tensor) -> None:
        self.v = value
        super().__init__(value.shape, torch.Size(), validate_args=False)

    @property
    def mean(self) -> Tensor:
        return self.v

    @property
    def variance(self) -> Tensor:
        return torch.zeros_like(self.v)

    def rsample(self, sample_shape: Any = ()) -> Tensor:
        return self.v.expand(*sample_shape, *self.v.shape)

    def log_prob(self, value: Tensor) -> Tensor:
        return (value == self.v).to(value.dtype).log()


class _TorchDeltaDistribution(_TorchBaseDistribution):
    def covariance_matrix(self) -> Tensor:
        return torch.diag_embed(torch.zeros_like(self.variational_mean))

    def forward(self) -> Any:
        return _TorchPointMass(self.variational_mean)

    def sample_with_base(self, noise: Tensor) -> Tensor:
        return self.variational_mean.expand_as(noise)

    def kl_divergence(self) -> Tensor:
        # GPyTorch's Delta/MVN registration defines the MAP regularizer as
        # minus the prior log density at the single variational particle.
        return 0.5 * (self.variational_mean.square() + math.log(2 * math.pi)).sum(-1)


def _distribution_moments(distribution: Any) -> tuple[Tensor, Tensor, bool]:
    """Evaluate the actual supplied law, including its registered autograd VJP."""
    if isinstance(distribution, _TorchBaseDistribution):
        return (
            distribution.variational_mean.clone(),
            distribution.covariance_matrix(),
            isinstance(distribution, _TorchDeltaDistribution),
        )
    import gpytorch

    value = distribution()
    if isinstance(value, gpytorch.distributions.MultivariateNormal):
        return value.mean.clone(), value.covariance_matrix, False
    if isinstance(value, gpytorch.distributions.Delta):
        return value.mean.clone(), torch.diag_embed(torch.zeros_like(value.mean)), True
    raise RuntimeError(
        f"Invalid variational distribution ({type(value)}). Expected a MultivariateNormal "
        "or a Delta distribution."
    )


def sample_variational_with_base(distribution: Any, noise: Tensor) -> Tensor:
    """Use a supplied distribution's root and sample semantics, without replacing it."""
    if isinstance(distribution, _TorchBaseDistribution):
        return distribution.sample_with_base(noise)
    import gpytorch

    value = distribution()
    if isinstance(value, gpytorch.distributions.MultivariateNormal):
        return value.rsample(base_samples=noise)
    if isinstance(value, gpytorch.distributions.Delta):
        return value.mean.expand_as(noise)
    raise RuntimeError(f"Invalid variational distribution ({type(value)}).")


class _TorchVariationalStrategy(nn.Module):
    _tensor_backend = "torch"
    _owner_ref: Any
    inducing_points: Tensor
    variational_params_initialized: Tensor
    _variational_distribution: Any

    def __init__(
        self, owner: Any, points: Tensor, distribution: Any, learn: bool, whitened: bool = True
    ) -> None:
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
        self.whitened = whitened
        self._kl_prior_jitter = 1e-3

    @property
    def jitter_val(self) -> float:
        return (
            (1e-4 if self.inducing_points.dtype == torch.float32 else 1e-6)
            if self._jitter_val is None
            else self._jitter_val
        )

    @jitter_val.setter
    def jitter_val(self, value: float) -> None:
        if not math.isfinite(value) or value < 0:
            raise ValueError("Variational jitter must be finite and nonnegative.")
        self._jitter_val = value

    @property
    def variational_distribution(self) -> Any:
        return self._variational_distribution()

    @property
    def prior_distribution(self) -> Any:
        if not self.whitened:
            return self._inducing_prior(1e-3)
        distribution = self._variational_distribution
        n = distribution.num_inducing_points
        mean = self.inducing_points.new_zeros(*distribution.batch_shape, n)
        return TorchGPyTorchPosterior(
            mean.unsqueeze(-1),
            torch.eye(n, dtype=mean.dtype, device=mean.device).expand(*mean.shape[:-1], n, n),
        ).distribution

    def _initialize(self) -> None:
        if not bool(self.variational_params_initialized):
            distribution = self._variational_distribution
            prior = self.prior_distribution
            if not isinstance(distribution, _TorchBaseDistribution):
                import gpytorch

                prior = gpytorch.distributions.MultivariateNormal(
                    prior.mean, prior.covariance_matrix
                )
            distribution.initialize_variational_distribution(
                None
                if self.whitened and isinstance(distribution, _TorchBaseDistribution)
                else prior
            )
            self.variational_params_initialized.fill_(1)

    def train(self, mode: bool = True) -> Any:
        if (self.training and not mode) or mode:
            self._kl_prior_jitter = 1e-3
        return super().train(mode)

    def _inducing_prior(self, jitter: float) -> Any:
        owner = self._owner_ref()
        points = self.inducing_points
        inducing = points if points.ndim == owner._train_X.ndim + 1 else owner._augment(points)
        mean = owner.model.mean_module(inducing)
        covariance = _dense(owner.model.covar_module(inducing, inducing))
        covariance = covariance + jitter * torch.eye(
            points.shape[-2], dtype=points.dtype, device=points.device
        )
        return TorchGPyTorchPosterior(mean.unsqueeze(-1), covariance).distribution

    def kl_divergence(self, moments: tuple[Tensor, Tensor, bool] | None = None) -> Tensor:
        distribution = self._variational_distribution
        if self.whitened and isinstance(distribution, _TorchBaseDistribution):
            return distribution.kl_divergence()
        mean, covariance, delta_distribution = (
            _distribution_moments(distribution) if moments is None else moments
        )
        if not self.whitened:
            prior = self._inducing_prior(self._kl_prior_jitter)
            delta = mean - prior.mean
            root = _rs.tensor_psd_safe_cholesky(prior.covariance_matrix)
            quadratic = (delta * torch.cholesky_solve(delta.unsqueeze(-1), root).squeeze(-1)).sum(
                -1
            )
            logdet = 2 * root.diagonal(dim1=-2, dim2=-1).log().sum(-1)
            if delta_distribution:
                return 0.5 * (quadratic + logdet + delta.shape[-1] * math.log(2 * math.pi))
            trace = torch.cholesky_solve(covariance, root).diagonal(dim1=-2, dim2=-1).sum(-1)
            return 0.5 * (
                quadratic
                + trace
                - delta.shape[-1]
                + logdet
                - torch.linalg.slogdet(covariance).logabsdet
            )
        quadratic = mean.square().sum(-1)
        if delta_distribution:
            return 0.5 * (quadratic + mean.shape[-1] * math.log(2 * math.pi))
        return 0.5 * (
            quadratic
            + covariance.diagonal(dim1=-2, dim2=-1).sum(-1)
            - mean.shape[-1]
            - torch.linalg.slogdet(covariance).logabsdet
        )

    def forward(self, X: Tensor, prior: bool = False) -> Any:
        owner = self._owner_ref()
        if prior:
            return owner._prior_output(X, multitask=owner.num_outputs > 1)
        if self.training:
            self._kl_prior_jitter = 1e-3
        self._initialize()
        return owner._variational_output(X)


class _TorchIndependentStrategy(nn.Module):
    base_variational_strategy: Any

    def __init__(self, strategy: Any) -> None:
        super().__init__()
        self.base_variational_strategy = strategy

    def kl_divergence(self, moments: tuple[Tensor, Tensor, bool] | None = None) -> Tensor:
        return self.base_variational_strategy.kl_divergence(moments).sum(-1)

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

    def named_variational_parameters(self) -> Any:
        distribution = self._outer._strategy._variational_distribution
        leaves = {id(value) for value in distribution.parameters()}
        return ((name, value) for name, value in self.named_parameters() if id(value) in leaves)

    def variational_parameters(self) -> Any:
        return (value for _, value in self.named_variational_parameters())

    def named_hyperparameters(self) -> Any:
        leaves = {id(value) for value in self.variational_parameters()}
        return ((name, value) for name, value in self.named_parameters() if id(value) not in leaves)

    def hyperparameters(self) -> Any:
        return (value for _, value in self.named_hyperparameters())


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
        variational_moments: tuple[Tensor, Tensor, bool] | None = None,
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
            self._kl = owner.model.variational_strategy.kl_divergence(variational_moments)
            self._log_prior = owner._log_prior()
            if getattr(owner.likelihood, "_non_gaussian", False):
                self._expected_log_prob = owner.likelihood.snapshot_expected_log_prob()
            else:
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
        gpytorch = sys.modules.get("gpytorch")
        strategy_options: dict[str, Any] = {}
        if isinstance(variational_strategy, functools.partial):
            strategy_options = dict(variational_strategy.keywords or {})
            if variational_strategy.args or "model" in strategy_options:
                raise TypeError("A variational strategy factory must leave its model unbound.")
            variational_strategy = variational_strategy.func
            # SingleTaskVariationalGP supplies these keyword arguments itself,
            # overriding partial defaults just as the source constructor does.
            for name in ("inducing_points", "variational_distribution", "learn_inducing_locations"):
                strategy_options.pop(name, None)
            if set(strategy_options) - {"jitter_val"}:
                raise TypeError(
                    f"Unexpected variational strategy arguments: {sorted(strategy_options)}."
                )
        strategies = {VariationalStrategy: True, UnwhitenedVariationalStrategy: False}
        if gpytorch is not None:
            strategies[gpytorch.variational.VariationalStrategy] = True
            strategies[gpytorch.variational.UnwhitenedVariationalStrategy] = False
        if variational_strategy not in strategies:
            raise UnsupportedError(
                "Torch variational models support VariationalStrategy and UnwhitenedVariationalStrategy."
            )
        whitened = strategies[variational_strategy]
        distribution_family = _distribution_family(variational_distribution)
        if getattr(variational_distribution, "_owner", None) is not None:
            raise UnsupportedError(
                "This variational distribution already belongs to a native model; create a new specification."
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
        from botorch_rs.likelihoods import BernoulliLikelihood
        from botorch_rs.models._torch_backend_variational_likelihoods import (
            TorchBernoulliLikelihood,
            TorchExternalGaussianLikelihood,
        )

        bernoulli = type(likelihood) is BernoulliLikelihood or (
            gpytorch is not None and type(likelihood) is gpytorch.likelihoods.BernoulliLikelihood
        )
        external_gaussian = (
            gpytorch is not None and type(likelihood) is gpytorch.likelihoods.GaussianLikelihood
        )
        if bernoulli:
            if num_outputs != 1:
                raise UnsupportedError(
                    "Probit variational observations currently require a scalar output."
                )
            self.likelihood = TorchBernoulliLikelihood(likelihood, train_X)
        elif external_gaussian:
            if num_outputs != 1:
                raise UnsupportedError(
                    "Independent multiple outputs require multitask observation noise."
                )
            self.likelihood = TorchExternalGaussianLikelihood(likelihood, train_X)
        elif likelihood is None and num_outputs > 1:
            likelihood = MultitaskGaussianLikelihood(num_outputs)
            self.likelihood = _TorchMultitaskNoise(likelihood, train_X)
        elif num_outputs > 1:
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
                    "The Torch variational backend supports Gaussian and scalar Bernoulli probit observations."
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
        if distribution_family == "external":
            distribution = variational_distribution.to(train_X)
        else:
            distribution_class = {
                "cholesky": _TorchCholeskyDistribution,
                "meanfield": _TorchMeanFieldDistribution,
                "delta": _TorchDeltaDistribution,
            }[distribution_family]
            distribution = distribution_class(
                points.shape[-2], self._aug_batch_shape, train_X, variational_distribution
            )
        strategy = _TorchVariationalStrategy(
            self, points, distribution, learn_inducing_points, whitened=whitened
        )
        if strategy_options.get("jitter_val") is not None:
            strategy.jitter_val = float(strategy_options["jitter_val"])
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
        from botorch_rs.models._pathwise_context import training_outcome_transform

        for transform, values in (
            (self.input_transform, self._train_X),
            (training_outcome_transform(self), self._train_Y),
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
        from botorch_rs.models._pathwise_context import training_outcome_transform

        if self._train_Y is None:
            raise ValueError("Training targets were not supplied; pass targets to VariationalELBO.")
        self._fit_transforms()
        outcome = training_outcome_transform(self)
        return self._train_Y if outcome is None else outcome(self._train_Y)[0]

    def _log_prior(self) -> Any:
        result = self._train_X.new_zeros(())
        for module in (
            self.model.covar_module,
            self.model.mean_module,
            self.likelihood,
            self.input_transform,
            self._strategy._variational_distribution,
        ):
            if module is not None:
                # VariationalELBO globally sums each registered prior before
                # broadcasting the scalar to its loss batches. Do not expand
                # shared scalar priors to augmented batches before this sum.
                value = module_log_prior(module, ())
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
        if not self._strategy.whitened:
            return self._unwhitened_output(X)
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
        moments = _distribution_moments(distribution)
        variational_mean, variational_covariance, _ = moments
        middle = variational_covariance - eye
        mean = self.model.mean_module(augmented) + (
            interpolation.mT @ variational_mean.unsqueeze(-1)
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

        return _TorchVariationalOutput(
            self, mean, variance, covariance, variational_moments=moments
        )

    def _unwhitened_output(self, X: Tensor) -> Any:
        X = X.clone()
        augmented = self._augment(X)
        strategy = self._strategy
        distribution = strategy._variational_distribution
        points = strategy.inducing_points
        inducing = points if points.ndim == self._train_X.ndim + 1 else self._augment(points)
        moments = _distribution_moments(distribution)
        variational_mean, covariance_q, delta_distribution = moments
        if torch.equal(augmented, inducing):
            if delta_distribution:
                raise RuntimeError(
                    "The reference unwhitened Delta strategy has no covariance at its inducing points."
                )
            return _TorchVariationalOutput(
                self,
                variational_mean,
                covariance_q.diagonal(dim1=-2, dim2=-1),
                lambda: covariance_q,
                variational_moments=moments,
            )
        kernel, mean_module = self.model.covar_module, self.model.mean_module
        m = points.shape[-2]
        kzz = _dense(kernel(inducing, inducing)) + strategy.jitter_val * torch.eye(
            m, dtype=X.dtype, device=X.device
        )
        root = _rs.tensor_psd_safe_cholesky(kzz)
        kzx = _dense(kernel(inducing, augmented))
        interpolation = torch.cholesky_solve(kzx, root)
        mean = mean_module(augmented) + (
            interpolation.mT @ (variational_mean - mean_module(inducing)).unsqueeze(-1)
        ).squeeze(-1)
        residual = _dense(kernel(augmented, augmented, diag=True)) - (kzx * interpolation).sum(-2)
        training = self.model.training
        variance = (residual.clamp_min(0) if training else residual) + (
            interpolation * (covariance_q @ interpolation)
        ).sum(-2)
        # The pinned strategy uses the conditional prior during training,
        # while its uncached evaluation prior uses add_jitter(1e-3).
        strategy._kl_prior_jitter = strategy.jitter_val if training else 1e-3
        parameters = {name: value.clone() for name, value in kernel.named_parameters()}
        buffers = {name: value.clone() for name, value in kernel.named_buffers()}

        def covariance() -> Tensor:
            correction = interpolation.mT @ covariance_q @ interpolation
            if training:
                return correction + torch.diag_embed(residual.clamp_min(0))
            prior = _dense(
                torch.func.functional_call(kernel, (parameters, buffers), (augmented, augmented))
            )
            return correction + prior - kzx.mT @ interpolation

        return _TorchVariationalOutput(
            self, mean, variance, covariance, variational_moments=moments
        )

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
        if getattr(self.likelihood, "_non_gaussian", False) and observation_noise is not False:
            if observation_noise is not True:
                raise UnsupportedError(
                    "Bernoulli observation noise is its likelihood, not a Gaussian covariance override."
                )
            if output_indices is not None:
                raise UnsupportedError(
                    "Observed Bernoulli posteriors do not support output selection."
                )
            from botorch_rs.posteriors.torch import TorchPosterior

            result: Any = TorchPosterior(self.likelihood(output))
            if self.outcome_transform is not None:
                result = self.outcome_transform.untransform_posterior(result, X=X)
            return result if posterior_transform is None else posterior_transform(result, X=X)
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
        if hasattr(output, "_expected_log_prob"):
            log_likelihood = output._expected_log_prob(y, mean, variance).mean(-1)
        else:
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

"""Port-owned scalar variational observation laws on the Tensor device."""

from __future__ import annotations

import math
import warnings
from typing import Any

import torch
from torch import Tensor, nn

from botorch_rs.exceptions.errors import UnsupportedError
from botorch_rs.models._torch_backend_modules import module_log_prior
from botorch_rs.posteriors._torch_backend import TorchGPyTorchPosterior


def _expected_probit(
    target: Tensor, mean: Tensor, variance: Tensor, locations: Tensor, weights: Tensor
) -> Tensor:
    if bool(target.eq(-1).any()):
        warnings.warn(
            "BernoulliLikelihood.expected_log_prob expects observations with labels in {0, 1}. "
            "Observations with labels in {-1, 1} are deprecated.",
            DeprecationWarning,
            stacklevel=3,
        )
        labels = target
    else:
        labels = target * 2 - 1
    shape = (-1,) + (1,) * mean.ndim
    values = mean + (2 * variance).sqrt() * locations.reshape(shape)
    return (torch.special.log_ndtr(values * labels) * weights.reshape(shape)).sum(0) / math.sqrt(
        math.pi
    )


class _TorchHermite(nn.Module):
    locations: Tensor
    weights: Tensor

    def __init__(self, source: Any, like: Tensor) -> None:
        super().__init__()
        if hasattr(source, "quadrature"):
            locations = source.quadrature.locations.to(like).clone()
            weights = source.quadrature.weights.to(like).clone()
        else:
            n = source.num_locs
            off_diagonal = (torch.arange(1, n, device=like.device, dtype=torch.double) / 2).sqrt()
            jacobi = torch.diag(off_diagonal, diagonal=1) + torch.diag(off_diagonal, diagonal=-1)
            locations, eigenvectors = torch.linalg.eigh(jacobi)
            weights = eigenvectors[0].square() * math.sqrt(math.pi)
            # GPyTorch constructs these constants in the default dtype before
            # moving a likelihood to its final device/dtype.
            locations = locations.to(torch.get_default_dtype()).to(like)
            weights = weights.to(torch.get_default_dtype()).to(like)
        self.num_locs = locations.numel()
        self.register_buffer("locations", locations, persistent=False)
        self.register_buffer("weights", weights, persistent=False)


class TorchBernoulliLikelihood(nn.Module):
    _tensor_backend = "torch"
    _non_gaussian = True
    has_analytic_marginal = True

    def __init__(self, source: Any, like: Tensor) -> None:
        super().__init__()
        self.quadrature = _TorchHermite(source, like)

    def forward(self, function: Any, **kwargs: Any) -> Any:
        if kwargs:
            raise UnsupportedError(f"Unsupported probit likelihood arguments: {sorted(kwargs)}.")
        if isinstance(function, Tensor):
            link = function
        else:
            link = function.mean / (1 + function.variance).sqrt()
        return torch.distributions.Bernoulli(probs=torch.special.ndtr(link))

    def marginal(self, function: Any, **kwargs: Any) -> Any:
        return self(function, **kwargs)

    def log_marginal(self, observations: Tensor, function: Any, **kwargs: Any) -> Tensor:
        return self.marginal(function, **kwargs).log_prob(observations)

    def expected_log_prob(self, observations: Tensor, function: Any, **kwargs: Any) -> Tensor:
        if kwargs:
            raise UnsupportedError(f"Unsupported probit expectation arguments: {sorted(kwargs)}.")
        return _expected_probit(
            observations,
            function.mean,
            function.variance,
            self.quadrature.locations,
            self.quadrature.weights,
        )

    def snapshot_expected_log_prob(self) -> Any:
        locations, weights = self.quadrature.locations.clone(), self.quadrature.weights.clone()
        return lambda target, mean, variance: _expected_probit(
            target, mean, variance, locations, weights
        )


class TorchExternalGaussianLikelihood(nn.Module):
    """Keep supplied Gaussian-noise leaves and evaluate the device law here."""

    _tensor_backend = "torch"

    def __init__(self, source: Any, like: Tensor) -> None:
        super().__init__()
        source.to(like)
        self.noise_covar = source.noise_covar
        self.batch_shape = self.noise_covar.raw_noise.shape[:-1]

    @property
    def noise(self) -> Tensor:
        return self.noise_covar.noise

    @noise.setter
    def noise(self, value: Any) -> None:
        self.noise_covar.noise = value

    @property
    def raw_noise(self) -> Any:
        return self.noise_covar.raw_noise

    def noise_for(self, n: int) -> Tensor:
        return self.noise.expand(*self.batch_shape, n)

    def log_prior(self) -> Tensor:
        value = module_log_prior(self.noise_covar, self.batch_shape)
        return torch.zeros_like(self.noise[..., 0]) + value

    def raw_parameter_bounds(self) -> dict[str, tuple[Any, Any]]:
        constraint = self.noise_covar.raw_noise_constraint
        if constraint.enforced:
            return {}
        return {"noise_covar.raw_noise": (constraint.lower_bound, constraint.upper_bound)}

    def forward(self, function: Any, noise: Any = None, **kwargs: Any) -> Any:
        if kwargs:
            raise UnsupportedError(f"Unsupported Gaussian likelihood arguments: {sorted(kwargs)}.")
        mean = function if isinstance(function, Tensor) else function.mean
        variance = self.noise_for(mean.shape[-1]) if noise is None else noise
        if isinstance(function, Tensor):
            return torch.distributions.Normal(mean, variance.sqrt())
        covariance = function.covariance_matrix + torch.diag_embed(variance)
        return TorchGPyTorchPosterior(mean.unsqueeze(-1), covariance).distribution

"""Public fixed Gaussian likelihoods without a GPyTorch runtime dependency."""

from __future__ import annotations

import copy
import math
import sys
import warnings
from typing import Any, cast

import torch
from torch import Tensor, nn

from botorch_rs.constraints import GreaterThan
from botorch_rs.exceptions.warnings import InputDataWarning, NumericsWarning
from botorch_rs.likelihoods import FixedNoiseGaussianLikelihood
from botorch_rs.models._torch_backend_modules import (
    TorchConstraint,
    TorchFixedNoiseLikelihood,
    _TorchGaussianNoise,
    _TorchModule,
)


class _FixedNoise(nn.Module):
    noise: Tensor

    def __init__(self, noise: Tensor) -> None:
        super().__init__()
        settings = sys.modules.get("gpytorch.settings")
        minimum = 1e-4 if noise.dtype == torch.float32 else 1e-6
        if settings is not None:
            minimum = settings.min_fixed_noise.value(noise.dtype)
        if bool((noise < minimum).any()):
            warnings.warn(
                f"Very small noise values detected. Rounding small noise values up to {minimum}.",
                NumericsWarning,
                stacklevel=3,
            )
            noise = noise.clamp_min(minimum)
        # Like FixedGaussianNoise, fixed observations are data, rather than a
        # persistent buffer. Keep their original data graph and checkpoint keys.
        self.noise = noise

    def _apply(self, function: Any, recurse: bool = True) -> _FixedNoise:
        if not isinstance(self.noise, nn.Parameter):
            self.noise = function(self.noise)
        return super()._apply(function, recurse=recurse)

    def forward(
        self, *params: Any, shape: Any = None, noise: Tensor | None = None, **kwargs: Any
    ) -> Tensor:
        if noise is not None:
            return torch.diag_embed(noise)
        if shape is None:
            value = params[0] if isinstance(params[0], Tensor) else params[0][0]
            shape = value.shape if value.ndim == 1 else value.shape[:-1]
        selected = self.noise if shape[-1] == self.noise.shape[-1] else self.noise.new_zeros(*shape)
        return torch.diag_embed(selected)

    def initialize(self, **kwargs: Any) -> _FixedNoise:
        if set(kwargs) != {"noise"}:
            raise AttributeError("Fixed Gaussian noise initialization accepts only noise.")
        self.noise = kwargs["noise"]
        return self


class _AdditionalNoise(_TorchGaussianNoise):
    raw_noise_constraint: Any

    def __init__(self, like: Tensor, batch_shape: Any, prior: Any, constraint: Any) -> None:
        _TorchModule.__init__(self, like)
        self.batch_shape = torch.Size(batch_shape)
        # Preserve the source's default-dtype bound before explicit migration.
        constraint = GreaterThan(1e-4) if constraint is None else constraint
        self.raw_noise_constraint = (
            constraint
            if isinstance(constraint, nn.Module)
            else TorchConstraint(constraint, torch.empty(0)).to(like)
        )
        self.raw_noise = nn.Parameter(like.new_zeros(*self.batch_shape, 1))
        if isinstance(prior, nn.Module):
            self.add_module("noise_prior", prior)
            self._prior_parameters["noise_prior"] = "noise", 1
        else:
            self._register_prior("noise_prior", prior, "noise", 1)

    def forward(self, *params: Any, shape: Any = None, **kwargs: Any) -> Tensor:
        if shape is None:
            value = params[0] if isinstance(params[0], Tensor) else params[0][0]
            shape = value.shape if value.ndim == 1 else value.shape[:-1]
        batch = torch.broadcast_shapes(shape[:-1], self.batch_shape)
        return torch.diag_embed(self.noise.expand(*batch, shape[-1]))


class TorchPublicFixedNoiseLikelihood(TorchFixedNoiseLikelihood, FixedNoiseGaussianLikelihood):
    batch_shape: Any
    _tensor_backend = "torch"
    _fixed_noise_component = True

    def __init__(
        self,
        noise: Tensor,
        learn_additional_noise: bool | None = False,
        batch_shape: Any = (),
        **kwargs: Any,
    ) -> None:
        _TorchModule.__init__(self, noise)
        self.batch_shape = torch.Size(batch_shape or ())
        self.noise_covar = _FixedNoise(noise)
        self.second_noise_covar = (
            _AdditionalNoise(
                noise, self.batch_shape, kwargs.get("noise_prior"), kwargs.get("noise_constraint")
            )
            if learn_additional_noise
            else None
        )

    @property
    def noise_covar(self) -> _FixedNoise:
        return cast(_FixedNoise, self._modules["noise_covar"])

    @noise_covar.setter
    def noise_covar(self, value: _FixedNoise) -> None:
        nn.Module.__setattr__(self, "noise_covar", value)

    @property
    def noise(self) -> Tensor:
        return self.noise_covar.noise + self.second_noise

    @noise.setter
    def noise(self, value: Tensor) -> None:
        self.noise_covar.initialize(noise=value)

    @property
    def second_noise(self) -> Tensor | float:
        return 0.0 if self.second_noise_covar is None else self.second_noise_covar.noise

    @second_noise.setter
    def second_noise(self, value: Any) -> None:
        if self.second_noise_covar is None:
            raise RuntimeError("Setting secondary noise requires learn_additional_noise=True.")
        self.second_noise_covar.noise = value

    def noise_for(self, n: int, noise: Tensor | None = None) -> Tensor:
        if noise is None:
            if self.noise_covar.noise.shape[-1] == n:
                noise = self.noise_covar.noise
            else:
                noise = self.noise_covar.noise.new_zeros(n)
                if self.second_noise_covar is None:
                    warnings.warn(
                        "Fixed-noise event size does not match; unspecified noise is a no-op.",
                        InputDataWarning,
                        stacklevel=3,
                    )
        return noise + self.second_noise

    def _shaped_noise_covar(self, base_shape: Any, *params: Any, **kwargs: Any) -> Tensor:
        shape = base_shape
        if params:
            value = params[0] if isinstance(params[0], Tensor) else params[0][0]
            shape = value.shape if value.ndim == 1 else value.shape[:-1]
        return torch.diag_embed(self.noise_for(shape[-1], kwargs.get("noise")))

    def forward(self, distribution: Any, *params: Any, **kwargs: Any) -> Any:
        if isinstance(distribution, Tensor):
            noise = self._shaped_noise_covar(distribution.shape, *params, **kwargs).diagonal(
                dim1=-2, dim2=-1
            )
            return torch.distributions.Normal(distribution, noise.sqrt())
        return self.marginal(distribution, *params, **kwargs)

    def marginal(self, distribution: Any, *params: Any, **kwargs: Any) -> Any:
        mean = distribution.mean
        covariance = distribution.covariance_matrix + self._shaped_noise_covar(
            mean.shape, *params, **kwargs
        )
        # Actual caller Gaussian classes retain their public distribution API.
        if type(distribution).__module__.startswith(("gpytorch.", "torch.distributions")):
            return type(distribution)(mean, covariance)
        from botorch_rs.posteriors._torch_backend import TorchGPyTorchPosterior

        return TorchGPyTorchPosterior(mean.unsqueeze(-1), covariance).distribution

    def expected_log_prob(
        self, target: Tensor, distribution: Any, *params: Any, **kwargs: Any
    ) -> Tensor:
        noise = self._shaped_noise_covar(distribution.mean.shape, *params, **kwargs).diagonal(
            dim1=-2, dim2=-1
        )
        return -0.5 * (
            (target - distribution.mean).square() / noise
            + distribution.variance / noise
            + noise.log()
            + math.log(2 * math.pi)
        )

    def log_marginal(
        self, target: Tensor, distribution: Any, *params: Any, **kwargs: Any
    ) -> Tensor:
        noise = self._shaped_noise_covar(distribution.mean.shape, *params, **kwargs).diagonal(
            dim1=-2, dim2=-1
        )
        return torch.distributions.Normal(
            distribution.mean, (distribution.variance + noise).clamp_min(1e-8).sqrt()
        ).log_prob(target)

    def get_fantasy_likelihood(self, **kwargs: Any) -> TorchPublicFixedNoiseLikelihood:
        if "noise" not in kwargs:
            raise RuntimeError("FixedNoiseGaussianLikelihood.fantasize requires a `noise` kwarg")
        old, new = self.noise_covar.noise, kwargs["noise"]
        batch = torch.broadcast_shapes(old.shape[:-1], new.shape[:-1])
        result = copy.deepcopy(self, {id(self.noise_covar): None})
        result.noise_covar = _FixedNoise(
            torch.cat((old.expand(*batch, old.shape[-1]), new.expand(*batch, new.shape[-1])), -1)
        )
        return result

"""MAP SAAS models with device-resident coupled priors and complete graphs."""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

from botorch_rs.exceptions.errors import UnsupportedError
from botorch_rs.kernels import MaternKernel, ScaleKernel
from botorch_rs.models._torch_backend import TorchSingleTaskGP
from botorch_rs.models._torch_backend_fully_bayesian import TorchGaussianMixturePosterior
from botorch_rs.models._torch_backend_modules import (
    TorchCompositeKernel,
    TorchScaleKernel,
    TorchStationaryKernel,
    build_input_transform,
)
from botorch_rs.models.map_saas import (
    AdditiveMapSaasSingleTaskGP,
    EnsembleMapSaasSingleTaskGP,
    get_gaussian_likelihood_with_gamma_prior,
    get_mean_module_with_normal_prior,
)
from botorch_rs.models.utils.gpytorch_modules import get_gaussian_likelihood_with_lognormal_prior
from botorch_rs.posteriors._torch_backend import TorchGPyTorchPosterior
from botorch_rs.utils.constraints import LogTransformedInterval
from botorch_rs.utils.types import DEFAULT


def _scale(base: TorchStationaryKernel, like: Tensor) -> TorchScaleKernel:
    # These descriptors initialize only the scale's metadata. In particular,
    # CUDA active_dims must not auto-convert them into a second Tensor tree;
    # the configured SAAS base below retains its own priors and parameter leaves.
    source = ScaleKernel(
        MaternKernel(
            ard_num_dims=base.ard_num_dims, active_dims=base.active_dims, backend="native"
        ),
        outputscale_constraint=LogTransformedInterval(0.01, 1e4, initial_value=10),
        batch_shape=base.batch_shape,
        backend="native",
    )
    return TorchScaleKernel(source, base, like)


def _base(
    d: int, batch: Any, like: Tensor, tau: Any = None, active_dims: Any = None
) -> TorchStationaryKernel:
    source = MaternKernel(
        nu=2.5, ard_num_dims=d, batch_shape=batch, active_dims=active_dims, backend="native"
    )
    return TorchStationaryKernel(source, like).configure_saas_prior(tau)


def _transform(input_transform: Any, X: Tensor) -> tuple[Any, int]:
    transform = build_input_transform(input_transform, X.shape[-1], X)
    with torch.no_grad():
        if transform is None:
            d = X.shape[-1]
        else:
            forward = getattr(transform, "preprocess_transform", transform)
            d = forward(X).shape[-1]
    return transform, d


def _sample_tau(batch_shape: Any, like: Tensor) -> Tensor:
    # Device-local random samples: no training datum or continuous parameter
    # is serialized through the CPU/f64 parameter bridge.
    return torch.distributions.HalfCauchy(like.new_tensor(0.1)).sample(batch_shape)


def build_additive_saas_kernel(
    d: int, num_taus: int, batch_shape: Any, like: Tensor, active_dims: Any = None
) -> TorchCompositeKernel:
    if type(num_taus) is not int or num_taus < 1:
        raise ValueError("num_taus must be a positive integer.")
    return TorchCompositeKernel(
        [
            _scale(_base(d, batch_shape, like, _sample_tau(batch_shape, like), active_dims), like)
            for _ in range(num_taus)
        ],
        "add",
        like,
    )


def get_torch_map_saas_model(
    train_X: Tensor,
    train_Y: Tensor,
    train_Yvar: Tensor | None = None,
    input_transform: Any = None,
    outcome_transform: Any = None,
    tau: Any = None,
) -> TorchSingleTaskGP:
    TorchSingleTaskGP._validate_data(train_X, train_Y, train_Yvar)
    batch = train_X.shape[:-2] + ((train_Y.shape[-1],) if train_Y.shape[-1] > 1 else ())
    transform, d = _transform(input_transform, train_X)
    kernel = _scale(_base(d, batch, train_X, tau), train_X)
    return TorchSingleTaskGP(
        train_X,
        train_Y,
        train_Yvar,
        covar_module=kernel,
        mean_module=get_mean_module_with_normal_prior(batch),
        likelihood=get_gaussian_likelihood_with_gamma_prior(batch) if train_Yvar is None else None,
        input_transform=transform,
        outcome_transform=outcome_transform,
    )


class TorchAdditiveMapSaasSingleTaskGP(TorchSingleTaskGP, AdditiveMapSaasSingleTaskGP):
    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        train_Yvar: Tensor | None = None,
        outcome_transform: Any = DEFAULT,
        input_transform: Any = None,
        num_taus: int = 4,
        *,
        backend: str = "torch",
    ) -> None:
        TorchSingleTaskGP._validate_data(train_X, train_Y, train_Yvar)
        batch = train_X.shape[:-2] + ((train_Y.shape[-1],) if train_Y.shape[-1] > 1 else ())
        transform, d = _transform(input_transform, train_X)
        TorchSingleTaskGP.__init__(
            self,
            train_X,
            train_Y,
            train_Yvar,
            mean_module=get_mean_module_with_normal_prior(batch),
            covar_module=build_additive_saas_kernel(d, num_taus, batch, train_X),
            likelihood=get_gaussian_likelihood_with_lognormal_prior(batch)
            if train_Yvar is None
            else None,
            input_transform=transform,
            outcome_transform=outcome_transform,
        )


class TorchEnsembleMapSaasSingleTaskGP(TorchSingleTaskGP, EnsembleMapSaasSingleTaskGP):
    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        train_Yvar: Tensor | None = None,
        num_taus: int = 4,
        taus: Any = None,
        outcome_transform: Any = DEFAULT,
        input_transform: Any = None,
        *,
        backend: str = "torch",
    ) -> None:
        TorchSingleTaskGP._validate_data(train_X, train_Y, train_Yvar)
        if train_X.ndim != 2 or train_Y.shape[-1] != 1:
            raise UnsupportedError("Ensemble MAP SAAS needs unbatched inputs and one output.")
        if type(num_taus) is not int or num_taus < 1:
            raise ValueError("num_taus must be a positive integer.")
        batch = torch.Size([num_taus])
        taus = (
            _sample_tau(batch, train_X)
            if taus is None
            else torch.as_tensor(taus, device=train_X.device, dtype=train_X.dtype)
        )
        if taus.shape != batch:
            raise ValueError(f"Expected taus shape {batch}, got {taus.shape}.")
        transform, d = _transform(input_transform, train_X)
        kernel = _scale(_base(d, batch, train_X, taus), train_X)
        TorchSingleTaskGP.__init__(
            self,
            train_X.unsqueeze(0).expand(num_taus, *train_X.shape).clone(),
            train_Y.unsqueeze(0).expand(num_taus, *train_Y.shape).clone(),
            None
            if train_Yvar is None
            else train_Yvar.unsqueeze(0).expand(num_taus, *train_Yvar.shape).clone(),
            mean_module=get_mean_module_with_normal_prior(batch),
            covar_module=kernel,
            likelihood=get_gaussian_likelihood_with_gamma_prior(batch)
            if train_Yvar is None
            else None,
            input_transform=transform,
            outcome_transform=outcome_transform,
        )
        if self.outcome_transform is not None and not getattr(
            self.outcome_transform, "_is_linear", False
        ):
            raise UnsupportedError(
                "A MAP SAAS Gaussian mixture requires linear outcome transforms."
            )
        self.register_buffer("taus", taus)

    def posterior(
        self,
        X: Tensor,
        output_indices: Any = None,
        observation_noise: Any = False,
        posterior_transform: Any = None,
        **kwargs: Any,
    ) -> TorchGaussianMixturePosterior:
        if isinstance(observation_noise, Tensor):
            observation_noise = observation_noise.unsqueeze(-3)
        posterior = TorchSingleTaskGP.posterior(
            self,
            X.unsqueeze(-3),
            output_indices=output_indices,
            observation_noise=observation_noise,
            posterior_transform=posterior_transform,
            **kwargs,
        )
        if not isinstance(posterior, TorchGPyTorchPosterior):
            raise UnsupportedError("A MAP SAAS mixture requires Gaussian component posteriors.")
        return TorchGaussianMixturePosterior(
            posterior.mean, posterior.distribution.covariance_matrix
        )

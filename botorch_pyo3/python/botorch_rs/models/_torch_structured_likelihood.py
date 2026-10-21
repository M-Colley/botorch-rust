"""Actual Gaussian observation laws for structured Tensor models."""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

from botorch_rs.models._torch_gaussian_likelihood import (
    gaussian_marginal,
    is_gpytorch_gaussian,
)


def validate_scalar_likelihood(likelihood: Any) -> None:
    """Accept scalar Gaussian modules without replacing their registered leaves."""
    from botorch_rs.models._torch_backend_modules import (
        TorchFixedNoiseLikelihood,
        TorchGaussianLikelihood,
    )

    if not is_gpytorch_gaussian(likelihood) and not isinstance(
        likelihood, (TorchGaussianLikelihood, TorchFixedNoiseLikelihood)
    ):
        raise TypeError("Structured exact inference requires a Gaussian likelihood module.")


def scalar_observed_law(
    likelihood: Any,
    mean: Tensor,
    covariance: Tensor,
    inputs: tuple[Tensor, ...],
    *,
    params: tuple[Any, ...] | None = None,
    kwargs: dict[str, Any] | None = None,
    training: bool = False,
) -> tuple[Tensor, Tensor]:
    """Keep a supplied scalar marginal, its inputs and observation overrides."""
    arguments = ((list(inputs),) if training else inputs) if params is None else params
    return gaussian_marginal(
        likelihood,
        mean,
        covariance,
        inputs[0],
        params=arguments,
        kwargs=kwargs,
    )


def _canonical_multitask(likelihood: Any) -> bool:
    from botorch_rs.models._torch_backend_kronecker import TorchMultitaskGaussianLikelihood

    if isinstance(likelihood, TorchMultitaskGaussianLikelihood):
        return type(likelihood).forward is TorchMultitaskGaussianLikelihood.forward
    if not is_gpytorch_gaussian(likelihood):
        return False
    from gpytorch.likelihoods.multitask_gaussian_likelihood import (
        MultitaskGaussianLikelihood,
        _MultitaskGaussianLikelihoodBase,
    )

    return (
        type(likelihood).marginal is MultitaskGaussianLikelihood.marginal
        and type(likelihood)._shaped_noise_covar
        is _MultitaskGaussianLikelihoodBase._shaped_noise_covar
    )


def multitask_observed_law(
    model: Any,
    mean: Tensor,
    covariance: Tensor,
    inputs: Tensor,
    *,
    params: tuple[Any, ...] | None = None,
    kwargs: dict[str, Any] | None = None,
    training: bool = False,
) -> tuple[Tensor, Tensor]:
    """A point-major multitask marginal, including actual module overrides."""
    from botorch_rs.models._torch_backend_kronecker import tensor_kron

    likelihood = model.likelihood
    if _canonical_multitask(likelihood) and not kwargs:
        # Construct the canonical covariance using ordinary Tensor algebra so
        # covariance-factor Hessians do not enter lazy-operator custom VJPs.
        eye = torch.eye(mean.shape[-2], dtype=mean.dtype, device=mean.device)
        return mean, covariance + tensor_kron(eye, model._noise_matrix())
    arguments = (([inputs],) if training else (inputs,)) if params is None else params
    if is_gpytorch_gaussian(likelihood):
        from gpytorch.distributions import MultitaskMultivariateNormal
        from linear_operator.operators import DenseLinearOperator

        distribution = MultitaskMultivariateNormal(
            mean, DenseLinearOperator(covariance), interleaved=True
        )
        observed = likelihood(distribution, *arguments, **(kwargs or {}))
        observed_mean, observed_covariance = observed.mean, observed.covariance_matrix
        if not getattr(observed, "_interleaved", True):
            q, m = mean.shape[-2:]
            index = torch.arange(q * m, device=mean.device).reshape(m, q).T.flatten()
            observed_covariance = observed_covariance.index_select(-2, index).index_select(
                -1, index
            )
    else:
        from botorch_rs.posteriors._torch_backend import TorchGPyTorchPosterior

        q, m = mean.shape[-2:]
        index = torch.arange(q * m, device=mean.device).reshape(q, m).T.flatten()
        output_major = covariance.index_select(-2, index).index_select(-1, index)
        observed = likelihood(
            TorchGPyTorchPosterior(mean, output_major).distribution,
            *arguments,
            **(kwargs or {}),
        )
        observed_mean = observed.mean
        covariance = observed.covariance_matrix
        index = torch.arange(q * m, device=mean.device).reshape(m, q).T.flatten()
        observed_covariance = covariance.index_select(-2, index).index_select(-1, index)
    if not isinstance(observed_mean, Tensor) or not isinstance(observed_covariance, Tensor):
        raise TypeError("A Gaussian marginal must return Tensor mean and covariance_matrix.")
    if (
        observed_mean.shape[-2:] != mean.shape[-2:]
        or observed_covariance.shape[-2:] != covariance.shape[-2:]
    ):
        raise ValueError("A Gaussian marginal must preserve the structured event shape.")
    return observed_mean, observed_covariance

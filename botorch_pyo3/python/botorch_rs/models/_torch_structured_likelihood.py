"""Actual Gaussian observation laws for structured Tensor models."""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

from botorch_rs.models._torch_gaussian_likelihood import (
    gaussian_marginal,
    is_fixed_noise,
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
    if params is None and is_fixed_noise(likelihood):
        # A structured event has one observation per grid cell, rather than
        # one per feature row. Let the actual event size select fixed noise.
        arguments = ()
    if is_gpytorch_gaussian(likelihood):
        from gpytorch.distributions import MultivariateNormal
        from linear_operator.operators import DenseLinearOperator

        from botorch_rs.models._torch_variational_external_likelihood import (
            posterior_from_gaussian,
        )

        observed = likelihood(
            MultivariateNormal(mean, DenseLinearOperator(covariance)),
            *arguments,
            **(kwargs or {}),
        )
        posterior = posterior_from_gaussian(observed)
        return posterior.mean.squeeze(-1), posterior._covariance_tensor
    return gaussian_marginal(
        likelihood,
        mean,
        covariance,
        inputs[0],
        params=() if params is None else params,
        kwargs=kwargs,
    )


def _canonical_multitask(likelihood: Any) -> bool:
    from botorch_rs.models._torch_backend_kronecker import TorchMultitaskGaussianLikelihood

    if isinstance(likelihood, TorchMultitaskGaussianLikelihood):
        return (
            getattr(likelihood.forward, "__func__", None)
            is TorchMultitaskGaussianLikelihood.forward
            and type(likelihood).__call__ is torch.nn.Module.__call__
            and not likelihood._forward_hooks
            and not likelihood._forward_pre_hooks
        )
    if not is_gpytorch_gaussian(likelihood):
        return False
    from gpytorch.likelihoods import Likelihood
    from gpytorch.likelihoods.multitask_gaussian_likelihood import (
        MultitaskGaussianLikelihood,
        _MultitaskGaussianLikelihoodBase,
    )

    return (
        getattr(likelihood.marginal, "__func__", None) is MultitaskGaussianLikelihood.marginal
        and getattr(likelihood._shaped_noise_covar, "__func__", None)
        is _MultitaskGaussianLikelihoodBase._shaped_noise_covar
        and type(likelihood).__call__ is Likelihood.__call__
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
        from botorch_rs.models._torch_variational_external_likelihood import (
            posterior_from_gaussian,
        )

        posterior = posterior_from_gaussian(observed)
        observed_mean = posterior.mean
        q, m = mean.shape[-2:]
        index = torch.arange(q * m, device=mean.device).reshape(m, q).T.flatten()
        observed_covariance = posterior._covariance_tensor.index_select(-2, index).index_select(
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

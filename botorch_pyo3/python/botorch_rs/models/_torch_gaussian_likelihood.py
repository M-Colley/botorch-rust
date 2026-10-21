"""Explicit Gaussian likelihood interoperability without diagonal approximations."""

from __future__ import annotations

from importlib import import_module
from typing import Any

import torch

from botorch_rs.posteriors._torch_backend import TorchGPyTorchPosterior


def is_gpytorch_gaussian(likelihood: Any) -> bool:
    return any(
        base.__name__ == "_GaussianLikelihoodBase"
        and base.__module__ == "gpytorch.likelihoods.gaussian_likelihood"
        for base in type(likelihood).__mro__
    )


def is_fixed_noise(likelihood: Any) -> bool:
    from botorch_rs.models._torch_backend_modules import TorchFixedNoiseLikelihood

    return isinstance(likelihood, TorchFixedNoiseLikelihood) or any(
        base.__name__ == "FixedNoiseGaussianLikelihood"
        and base.__module__ == "gpytorch.likelihoods.gaussian_likelihood"
        for base in type(likelihood).__mro__
    )


def uses_marginal(likelihood: Any) -> bool:
    from botorch_rs.models._torch_backend_modules import (
        TorchFixedNoiseLikelihood,
        TorchGaussianLikelihood,
    )

    return is_gpytorch_gaussian(likelihood) or (
        isinstance(likelihood, (TorchGaussianLikelihood, TorchFixedNoiseLikelihood))
        and type(likelihood).forward
        not in (TorchGaussianLikelihood.forward, TorchFixedNoiseLikelihood.forward)
    )


def condition_fixed_likelihood(
    model: Any, X: torch.Tensor, Y: torch.Tensor, noise: Any, **kwargs: Any
) -> Any:
    """Append actual fixed noise without mutating the supplied likelihood."""
    from botorch_rs.models._numpyro_fitting import _copy_transform
    from botorch_rs.models._torch_backend import TorchSingleTaskGP
    from botorch_rs.models._torch_fully_bayesian_loader import _copy_model

    if noise is None:
        raise ValueError("Fixed-noise conditioning requires explicit noise.")
    if (
        not isinstance(noise, torch.Tensor)
        or noise.dtype != model._train_X.dtype
        or noise.device != model._train_X.device
        or not bool(torch.isfinite(noise).all() & (noise >= 0).all())
    ):
        raise ValueError(
            "Conditioning noise must be finite, nonnegative and use the model dtype/device."
        )
    if torch.broadcast_shapes(noise.shape, Y.shape) != Y.shape:
        raise ValueError("Conditioning noise must broadcast to targets.")
    stage = _copy_model(model)
    stage.likelihood = _copy_transform(model.likelihood)
    # Additional inferred noise remains a separate learned component. The
    # fixed likelihood's fantasy protocol appends only the fixed component.
    fixed = (
        model.likelihood.noise_covar.noise
        if is_gpytorch_gaussian(model.likelihood)
        else model.likelihood.noise
    )
    fixed = fixed.transpose(-1, -2) if model.num_outputs > 1 else fixed.unsqueeze(-1)
    if model.outcome_transform is not None:
        fixed = model.outcome_transform.untransform(
            torch.zeros_like(fixed), fixed, X=model.transform_inputs(model._train_X)
        )[1]
    stage._train_Yvar, stage._is_custom_likelihood = fixed, False
    conditioned = TorchSingleTaskGP.condition_on_observations(stage, X, Y, noise=noise, **kwargs)
    fixed = conditioned._training_values()[2]
    conditioned.likelihood.noise = fixed
    conditioned._is_custom_likelihood = True
    return conditioned


def gaussian_marginal(
    likelihood: Any,
    mean: torch.Tensor,
    covariance: torch.Tensor,
    inputs: torch.Tensor,
    *,
    training: bool = False,
    noise: torch.Tensor | None = None,
    params: tuple[Any, ...] | None = None,
    kwargs: dict[str, Any] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Evaluate the actual supplied marginal on its Gaussian prior and inputs."""
    options = {} if kwargs is None else kwargs.copy()
    if noise is not None:
        options["noise"] = noise.expand_as(mean)
    if is_gpytorch_gaussian(likelihood):
        distribution_type = import_module("gpytorch.distributions").MultivariateNormal
        operator_type = import_module("linear_operator.operators").DenseLinearOperator
        distribution = distribution_type(mean, operator_type(covariance))
        arguments = (([inputs] if training else inputs),) if params is None else params
        observed = likelihood(distribution, *arguments, **options)
    else:
        distribution = TorchGPyTorchPosterior(mean.unsqueeze(-1), covariance).distribution
        observed = likelihood(distribution, *(params or ()), **options)
    observed_mean = getattr(observed, "mean", None)
    observed_covar = getattr(observed, "covariance_matrix", None)
    if not isinstance(observed_mean, torch.Tensor) or not isinstance(observed_covar, torch.Tensor):
        raise TypeError(
            "A Gaussian likelihood marginal must return Tensor mean and covariance_matrix."
        )
    if (
        observed_mean.shape[-1:] != mean.shape[-1:]
        or observed_covar.shape[-2:] != covariance.shape[-2:]
    ):
        raise ValueError("A Gaussian likelihood marginal must preserve the prior event shape.")
    torch.broadcast_shapes(
        mean.shape[:-1], covariance.shape[:-2], observed_mean.shape[:-1], observed_covar.shape[:-2]
    )
    return observed_mean, observed_covar

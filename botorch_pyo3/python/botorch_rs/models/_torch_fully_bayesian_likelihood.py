"""Actual Gaussian marginals from explicitly returned likelihood modules."""

from __future__ import annotations

from importlib import import_module
from typing import Any

import torch

from botorch_rs import _botorch_rs as _rs
from botorch_rs.posteriors._torch_backend import TorchGPyTorchPosterior


def gaussian_marginal(
    likelihood: Any,
    mean: torch.Tensor,
    covariance: torch.Tensor,
    inputs: torch.Tensor,
    *,
    training: bool = False,
    noise: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply the supplied Gaussian law, including its full noise operator.

    Optional GPyTorch distribution/operator interoperability is selected only
    for an explicitly supplied GPyTorch likelihood. No upstream GP is used.
    """
    gpytorch = any(
        base.__name__ == "_GaussianLikelihoodBase"
        and base.__module__ == "gpytorch.likelihoods.gaussian_likelihood"
        for base in type(likelihood).__mro__
    )
    kwargs = {} if noise is None else {"noise": noise.expand_as(mean)}
    if gpytorch:
        distribution_type = import_module("gpytorch.distributions").MultivariateNormal
        operator_type = import_module("linear_operator.operators").DenseLinearOperator

        # A lazy dense operator avoids an unnecessary Cholesky of a possibly
        # rank-deficient latent prior (e.g. a linear kernel with q > d).
        distribution = distribution_type(mean, operator_type(covariance))
        observed = likelihood(distribution, [inputs] if training else inputs, **kwargs)
    else:
        distribution = TorchGPyTorchPosterior(mean.unsqueeze(-1), covariance).distribution
        observed = likelihood(distribution, **kwargs)
    return observed.mean, observed.covariance_matrix


def posterior_tensors(model: Any, X: torch.Tensor, observation_noise: Any) -> tuple[Any, Any]:
    """Condition using the actual returned Gaussian likelihood's covariance."""
    if not isinstance(X, torch.Tensor) or X.ndim < 2 or X.shape[-1] != model._train_X.shape[-1]:
        raise ValueError("X must be a Tensor with shape (..., q, d).")
    if X.dtype != model._train_X.dtype or X.device != model._train_X.device:
        raise ValueError("X must share the model's dtype and device; use model.to(X).")
    model.eval()
    train_X, targets, _ = model._training_values()
    query_X = model._augment_X(model.transform_inputs(X))
    blocks = getattr(model, "_prior_blocks", None)
    if blocks is None:
        train_mean, train_cov = model.mean_module(train_X), model._covariance(train_X, train_X)
        cross = model._covariance(train_X, query_X)
        query_mean, query_cov = model.mean_module(query_X), model._covariance(query_X, query_X)
    else:
        train_mean, train_cov, cross, query_mean, query_cov = blocks(train_X, query_X)
    observed_mean, observed_cov = gaussian_marginal(
        model.likelihood, train_mean, train_cov, train_X, training=True
    )
    factor, alpha = _rs.tensor_exact_gp_factor(
        targets, observed_cov, observed_mean, torch.zeros_like(observed_mean)
    )
    mean, covariance = _rs.tensor_exact_gp_predict(factor, alpha, cross, query_cov, query_mean)
    noise = None
    if isinstance(observation_noise, torch.Tensor):
        noise = model._augment_Y(observation_noise)
    elif observation_noise is True:
        if getattr(model, "_fb_custom_fixed_noise", False):
            noise = model.likelihood.noise.mean(-1, keepdim=True)
    elif observation_noise is False:
        return mean, covariance
    else:
        raise TypeError("observation_noise must be bool or a Tensor.")
    return gaussian_marginal(model.likelihood, mean, covariance, query_X, noise=noise)


__all__ = ["gaussian_marginal", "posterior_tensors"]

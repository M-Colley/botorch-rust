"""Actual Gaussian marginals from explicitly returned likelihood modules."""

from __future__ import annotations

from typing import Any

import torch

from botorch_rs import _botorch_rs as _rs
from botorch_rs.models._torch_gaussian_likelihood import gaussian_marginal, is_fixed_noise
from botorch_rs.posteriors._torch_backend import TorchGPyTorchPosterior


def posterior_tensors(
    model: Any, X: torch.Tensor, observation_noise: Any, *, inputs_transformed: bool = False
) -> tuple[Any, Any]:
    """Condition using the actual returned Gaussian likelihood's covariance."""
    if (
        not isinstance(X, torch.Tensor)
        or X.ndim < 2
        or (not inputs_transformed and X.shape[-1] != model._train_X.shape[-1])
    ):
        raise ValueError("X must be a Tensor with shape (..., q, d).")
    if X.dtype != model._train_X.dtype or X.device != model._train_X.device:
        raise ValueError("X must share the model's dtype and device; use model.to(X).")
    model.eval()
    train_X, targets, _ = model._training_values()
    if inputs_transformed and X.shape[-1] != train_X.shape[-1]:
        raise ValueError("Evaluation inputs must use the transformed feature dimension.")
    query_X = X if inputs_transformed else model._augment_X(model.transform_inputs(X))
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
        if getattr(model, "_fb_custom_fixed_noise", False) or is_fixed_noise(model.likelihood):
            noise = model.likelihood.noise.mean(-1, keepdim=True)
    elif observation_noise is False:
        return mean, covariance
    else:
        raise TypeError("observation_noise must be bool or a Tensor.")
    return gaussian_marginal(model.likelihood, mean, covariance, query_X, noise=noise)


def multitask_posterior(
    model: Any,
    X: torch.Tensor,
    output_indices: Any = None,
    observation_noise: Any = False,
    posterior_transform: Any = None,
    **kwargs: Any,
) -> Any:
    """Long-format task conditioning with an actual returned Gaussian likelihood."""
    from botorch_rs.exceptions.errors import UnsupportedError

    if kwargs:
        raise UnsupportedError(f"Unsupported posterior arguments: {sorted(kwargs)}.")
    if (
        not isinstance(X, torch.Tensor)
        or X.ndim < 2
        or X.dtype != model._train_X.dtype
        or X.device != model._train_X.device
    ):
        raise ValueError("X must be a Tensor with the model's dtype/device and shape (...,q,d).")
    if isinstance(observation_noise, torch.Tensor):
        raise NotImplementedError("Tensor observation_noise is not supported by MultiTaskGP.")
    if not isinstance(observation_noise, bool):
        raise TypeError("observation_noise must be bool.")
    full, outputs, points = model._query_inputs(X, output_indices)
    model.eval()
    raw_train, targets, fixed_noise = model._training_values()
    transformed = model.transform_inputs(full)
    if not torch.equal(transformed[..., model._task_feature], full[..., model._task_feature]):
        raise UnsupportedError("Input transforms must preserve categorical task labels.")
    train_mean, train_cov, cross, query_mean, query_cov = model._task_prior_blocks(
        raw_train, transformed
    )
    observed_mean, observed_cov = gaussian_marginal(
        model.likelihood,
        train_mean,
        train_cov,
        raw_train,
        training=True,
        noise=(
            None
            if getattr(model, "_fb_custom_likelihood", False)
            else model._training_noise(targets, fixed_noise)
        ),
    )
    factor, alpha = _rs.tensor_exact_gp_factor(
        targets, observed_cov, observed_mean, torch.zeros_like(observed_mean)
    )
    mean, covariance = _rs.tensor_exact_gp_predict(
        factor,
        alpha,
        cross,
        query_cov,
        query_mean,
    )
    if observation_noise:
        from botorch_rs.models._torch_backend_modules import TorchFixedNoiseLikelihood

        fixed = getattr(model, "_fb_custom_fixed_noise", False) or isinstance(
            model.likelihood, TorchFixedNoiseLikelihood
        )
        noise = (
            model._fixed_noise_at(
                model._mapped_inputs(transformed),
                model.likelihood.noise
                if getattr(model, "_fb_custom_likelihood", False)
                else fixed_noise,
            )
            if fixed
            else None
        )
        mean, covariance = gaussian_marginal(
            model.likelihood, mean, covariance, transformed, noise=noise
        )
    posterior = TorchGPyTorchPosterior(
        mean.reshape(*mean.shape[:-1], outputs, points).transpose(-1, -2), covariance
    )
    if model.outcome_transform is not None:
        outcome = model.outcome_transform
        if outputs > 1:
            subset = getattr(outcome, "subset_output", None)
            if subset is None:
                raise UnsupportedError(
                    "Multi-output outcome transforms must support subset_output."
                )
            outcome = subset([0] * outputs)
        posterior = outcome.untransform_posterior(posterior, X=X)
    return posterior if posterior_transform is None else posterior_transform(posterior, X=X)


__all__ = ["gaussian_marginal", "posterior_tensors", "multitask_posterior"]

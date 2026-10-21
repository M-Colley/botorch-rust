"""Transactional adoption of explicit fully Bayesian module-loading hooks."""

from __future__ import annotations

import copy
from typing import Any

import torch
from torch import nn

from botorch_rs.exceptions.errors import UnsupportedError
from botorch_rs.models._numpyro_fitting import (
    _adopt_transform,
    _commit_transform_identities,
    _copy_transform,
)
from botorch_rs.models._torch_backend_modules import (
    TorchFixedNoiseLikelihood,
    TorchGaussianLikelihood,
    TorchInputChain,
    build_input_transform,
    build_kernel,
    build_likelihood,
    build_mean,
)

_BUILTIN = {
    "botorch_rs.models.fully_bayesian",
    "botorch_rs.models.fully_bayesian_multitask",
}
_LOAD_HOOKS = (
    "load_mcmc_samples",
    "_mean_module",
    "_build_mean_module",
    "_likelihood",
    "_warping_function",
    "_get_covar_module",
    "_build_multitask_covariance",
)


def has_custom_loader(pyro: Any) -> bool:
    return any(
        (method := getattr(pyro, name, None)) is not None
        and getattr(method, "__module__", "") not in _BUILTIN
        for name in _LOAD_HOOKS
    )


def _module_device(module: nn.Module, like: torch.Tensor) -> None:
    # Do not silently call Module.to(): returned modules can already belong
    # to the caller, and failed loading must not mutate their parameters.
    for value in (*module.parameters(), *module.buffers()):
        if value.device != like.device or (value.is_floating_point() and value.dtype != like.dtype):
            raise ValueError("Returned modules must use the training dtype and device.")


def _copy_model(model: Any) -> Any:
    result = copy.copy(model)
    result._modules = model._modules.copy()
    result._parameters = model._parameters.copy()
    result._buffers = model._buffers.copy()
    result._prediction_cache, result._transform_fit_versions = None, {}
    return result


def _validation_module(module: nn.Module) -> nn.Module:
    """Isolate mutable buffers without breaking supplied parameter graphs."""
    result = copy.copy(module)
    result._parameters = module._parameters.copy()
    result._buffers = {
        name: None if value is None else value.clone() for name, value in module._buffers.items()
    }
    result._modules = {
        name: None if child is None else _validation_module(child)
        for name, child in module._modules.items()
    }
    return result


def _gaussian_likelihood(module: nn.Module) -> bool:
    if isinstance(module, (TorchGaussianLikelihood, TorchFixedNoiseLikelihood)):
        return type(module).forward in (
            TorchGaussianLikelihood.forward,
            TorchFixedNoiseLikelihood.forward,
        )
    # Explicitly supplied GPyTorch Gaussian modules are optional Torch
    # interoperability. A noise attribute alone does not define a Gaussian.
    return (
        any(
            base.__name__ == "_GaussianLikelihoodBase"
            and base.__module__ == "gpytorch.likelihoods.gaussian_likelihood"
            for base in type(module).__mro__
        )
        and getattr(getattr(type(module), "marginal", None), "__module__", "")
        == "gpytorch.likelihoods.gaussian_likelihood"
    )


def load_custom_modules(model: Any, samples: dict[str, Any]) -> None:
    """Validate a custom scalar Gaussian module tuple before changing the model."""
    if not samples:
        raise ValueError("MCMC samples must be a nonempty dictionary.")
    like = model._train_X
    count = None
    for name, value in samples.items():
        if (
            not isinstance(value, torch.Tensor)
            or not value.ndim
            or not value.shape[0]
            or value.device != like.device
            or value.dtype != like.dtype
            or not bool(torch.isfinite(value).all())
        ):
            raise ValueError(f"MCMC {name} must have finite rows and the training dtype/device.")
        if count is None:
            count = value.shape[0]
        elif count != value.shape[0]:
            raise ValueError("All MCMC sample sites must have the same leading row count.")
    assert count is not None
    batch = torch.Size([count])
    stage = _copy_model(model)
    original_pyro = model.pyro_model
    stage.pyro_model = copy.copy(original_pyro)
    user = _copy_transform(model._fb_user_input_transform)
    object.__setattr__(stage, "_fb_user_input_transform", user)
    stage.input_transform = user
    stage.outcome_transform = _copy_transform(model.outcome_transform)
    stage._mcmc_count, stage._initializing, stage._fb_custom_likelihood = None, True, False
    X, Y, fixed = stage._training_values()
    # The program trace receives detached snapshots separately. Loader code
    # is allowed to retain the live transformed training-data graph.
    stage.pyro_model.set_inputs(X, Y.unsqueeze(-1), None if fixed is None else fixed.unsqueeze(-1))
    returned = stage.pyro_model.load_mcmc_samples(mcmc_samples=samples)
    if not isinstance(returned, (tuple, list)) or len(returned) != 4:
        raise ValueError(
            "load_mcmc_samples must return mean, covariance, likelihood and input transform."
        )
    mean, covariance, likelihood, transform = returned
    for module in (mean, covariance, likelihood, transform):
        if isinstance(module, nn.Module):
            _module_device(module, like)
    mean_source, likelihood_source = mean, likelihood
    mean = mean if isinstance(mean, nn.Module) else build_mean(mean, batch, like)
    for name, value in getattr(mean_source, "_pending", {}).items():
        setattr(mean, name, like.new_tensor(value).reshape(batch))
    covariance = (
        covariance
        if isinstance(covariance, nn.Module)
        else build_kernel(covariance, like.shape[-1], batch, like)
    )
    transforms = {} if user is None else {"user": user}
    if transform is not None:
        transform = (
            transform
            if isinstance(transform, nn.Module)
            else build_input_transform(transform, like.shape[-1], like)
        )
        transforms["sample"] = transform
    stage.input_transform = TorchInputChain(transforms, like) if transforms else None
    stage._mcmc_count, stage._initializing = count, False
    if isinstance(likelihood, nn.Module):
        if not _gaussian_likelihood(likelihood):
            raise UnsupportedError(
                "Returned likelihoods must implement a supported Gaussian marginal."
            )
        stage._fb_custom_likelihood = True
    else:
        # Built-in descriptor loaders use the original live fixed noise;
        # avoid evaluating a caller-returned transform before validation.
        if fixed is not None:
            fixed = fixed.expand(count, fixed.shape[-1])
        likelihood = build_likelihood(likelihood, fixed, batch, like)
        for name, value in getattr(likelihood_source, "_pending", {}).items():
            setattr(likelihood, name, like.new_tensor(value).reshape(count, 1))
    stage._fb_custom_fixed_noise = isinstance(likelihood, TorchFixedNoiseLikelihood) or any(
        base.__name__ == "FixedNoiseGaussianLikelihood"
        and base.__module__ == "gpytorch.likelihoods.gaussian_likelihood"
        for base in type(likelihood).__mro__
    )
    stage.mean_module, stage.covar_module, stage.likelihood = mean, covariance, likelihood
    # Validation owns its mutable module/buffer state. The actual returned
    # modules and their Parameter identities are retained for successful adoption.
    validation = _copy_model(stage)
    validation._modules = {
        name: None if child is None else _validation_module(child)
        for name, child in stage._modules.items()
    }
    with torch.no_grad():
        train_X, train_Y, fixed = validation._training_values()
        mean_value = validation.mean_module(train_X)
        covariance_value = validation._covariance(train_X, train_X)
        expected = (count, like.shape[-2])
        if mean_value.shape != expected or covariance_value.shape != (*expected, like.shape[-2]):
            raise ValueError(
                "Returned mean/kernel must expose one scalar GP batch per MCMC sample."
            )
        if validation._fb_custom_likelihood:
            from botorch_rs.models._torch_fully_bayesian_likelihood import gaussian_marginal

            _, observed = gaussian_marginal(
                validation.likelihood, mean_value, covariance_value, train_X, training=True
            )
            noise = observed - covariance_value
            if not bool(torch.isfinite(noise).all()):
                raise ValueError("Returned Gaussian noise covariance must be finite and PSD.")
            tolerance = 100 * torch.finfo(noise.dtype).eps * noise.abs().amax((-2, -1)).clamp_min(1)
            symmetric = (noise - noise.transpose(-1, -2)).abs() <= tolerance[..., None, None]
            if not bool(symmetric.all()) or not bool(
                (torch.linalg.eigvalsh(noise) >= -tolerance[..., None]).all()
            ):
                raise ValueError("Returned Gaussian noise covariance must be finite and PSD.")
        else:
            noise_value = validation._training_noise(train_Y, fixed)
            if not bool(torch.isfinite(noise_value).all() & (noise_value >= 0).all()):
                raise ValueError(
                    "Returned Gaussian observation noise must be finite and nonnegative."
                )
        posterior = validation.posterior(like[: min(2, like.shape[-2])], observation_noise=True)
        if posterior.mean.shape[-3] != count or not bool(
            torch.isfinite(posterior.mean).all() & torch.isfinite(posterior.variance).all()
        ):
            raise ValueError("Returned modules must define finite scalar Gaussian MCMC components.")
    # Commit validation's fitted buffers and evaluation flags only after all
    # checks succeeded. Parameters remain the original returned objects.
    for name, module in stage._modules.items():
        if module is not None:
            _adopt_transform(module, validation._modules[name])
    stage._prediction_cache, stage._transform_fit_versions = None, {}
    _commit_transform_identities(model, stage)
    original_pyro.__dict__.clear()
    original_pyro.__dict__.update(stage.pyro_model.__dict__)
    stage.pyro_model = original_pyro
    model.__dict__.clear()
    model.__dict__.update(stage.__dict__)


__all__ = ["has_custom_loader", "load_custom_modules"]

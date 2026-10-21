"""Preserve actual external outcome callbacks while refreshing calibration graphs."""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor, nn

_TYPES: dict[type[Any], type[Any]] = {}
_METHODS = ("forward", "untransform", "untransform_posterior")
_HOOKS = ("_forward_hooks", "_forward_pre_hooks", "_backward_hooks", "_backward_pre_hooks")


def needs_external_protocol(module: nn.Module) -> bool:
    """Canonical conversion must not discard overrides, callbacks, or user state."""
    if getattr(type(module), "_torch_external_outcome_factory", False):
        return True
    for child in module.modules():
        cls = type(child)
        canonical = (
            cls.__module__ == "botorch.models.transforms.outcome"
            and cls.__name__
            in {
                "Standardize",
                "StratifiedStandardize",
                "Log",
                "Power",
                "Bilog",
                "ChainedOutcomeTransform",
            }
        ) or (
            cls.__module__ == "botorch.models.higher_order_gp"
            and cls.__name__ == "FlattenedStandardize"
        )
        if not canonical or any(getattr(child, name, None) for name in _HOOKS):
            return True
        if any(name in child.__dict__ for name in _METHODS):
            return True
        parameters = {name for name, value in child._parameters.items() if value is not None}
        expected_parameters = {"power"} if cls.__name__ == "Power" else set()
        if parameters - expected_parameters:
            return True
        expected_buffers = (
            {"means", "stdvs", "_stdvs_sq", "_is_trained", "strata_mapping"}
            if "Standardize" in cls.__name__
            else set()
        )
        if set(child._buffers) - expected_buffers:
            return True
        if child._modules and cls.__name__ != "ChainedOutcomeTransform":
            return True
    return False


def _fit(self: Any, Y: Tensor, X: Tensor | None = None) -> Any:
    modes = [(module, module.training) for module in self.modules()]
    try:
        self.train()
        self(Y, X=X)
    finally:
        for module, mode in modes:
            module.train(mode)
    return self


def _reduce(self: Any, protocol: int) -> Any:
    return rebuild, (self._external_outcome_type,), self.__getstate__()


def _source_standardize(original_type: type[Any]) -> bool:
    return any(
        cls.__module__ == "botorch.models.transforms.outcome" and cls.__name__ == "Standardize"
        for cls in original_type.__mro__
    )


def _untransform_posterior(self: Any, posterior: Any, X: Tensor | None = None) -> Any:
    from botorch_rs.posteriors._torch_backend import TorchGPyTorchPosterior

    method = self._external_outcome_type.untransform_posterior
    if not isinstance(posterior, TorchGPyTorchPosterior):
        return method(self, posterior, X=X)
    # Supplied BoTorch callbacks expect their own Gaussian envelope. GP
    # training and prediction remain in the port's differentiable Tensor law.
    from botorch.posteriors.gpytorch import GPyTorchPosterior
    from gpytorch.distributions import MultitaskMultivariateNormal, MultivariateNormal
    from linear_operator.operators import DenseLinearOperator

    mean, covariance = posterior.mean, DenseLinearOperator(posterior._covariance_tensor)
    distribution = (
        MultivariateNormal(mean.squeeze(-1), covariance)
        if mean.shape[-1] == 1
        else MultitaskMultivariateNormal(mean, covariance, interleaved=False)
    )
    source_type = GPyTorchPosterior
    if getattr(posterior, "_is_gaussian_mixture", False):
        from botorch.posteriors.fully_bayesian import GaussianMixturePosterior

        source_type = GaussianMixturePosterior
    return _port_posterior(method(self, source_type(distribution), X=X))


def _port_posterior(posterior: Any) -> Any:
    from botorch.posteriors.gpytorch import GPyTorchPosterior
    from botorch.posteriors.transformed import TransformedPosterior as SourceTransformedPosterior

    from botorch_rs.posteriors.transformed import TransformedPosterior

    if isinstance(posterior, GPyTorchPosterior):
        from botorch_rs.models._torch_variational_external_likelihood import posterior_from_gaussian

        result = posterior_from_gaussian(posterior.distribution)
        if type(posterior).__name__ == "GaussianMixturePosterior":
            from botorch_rs.models._torch_backend_fully_bayesian import (
                TorchGaussianMixturePosterior,
            )

            result = TorchGaussianMixturePosterior(result.mean, result._covariance_tensor)
        return result
    if type(posterior) in (SourceTransformedPosterior, TransformedPosterior):
        return TransformedPosterior(
            _port_posterior(posterior._posterior),
            posterior._sample_transform,
            posterior._mean_transform,
            posterior._variance_transform,
        )
    return posterior


def module_type(original_type: type[Any]) -> type[Any]:
    if original_type not in _TYPES:
        attributes: dict[str, Any] = {
            "__module__": __name__,
            "_external_outcome_type": original_type,
            "_torch_external_outcome_factory": True,
            "_tensor_backend": "torch",
            "__reduce_ex__": _reduce,
        }
        if not callable(getattr(original_type, "fit", None)):
            attributes["fit"] = _fit
            # Model constructors supply input context to every outcome. The
            # fitting protocol must preserve it for custom numerical callbacks.
            attributes["_requires_X"] = True
        if _source_standardize(original_type):
            attributes["untransform_posterior"] = _untransform_posterior
        _TYPES[original_type] = type(
            f"TorchFitted{original_type.__name__}", (original_type,), attributes
        )
    return _TYPES[original_type]


def rebuild(original_type: type[Any]) -> Any:
    return object.__new__(module_type(original_type))


def external_outcome_module(
    source: nn.Module, like: Tensor | None, train_Y: Tensor | None, train_X: Tensor | None
) -> nn.Module:
    """Add only the fit protocol, retaining object, Parameter and hook identity."""
    if not getattr(type(source), "_torch_external_outcome_factory", False):
        source.__class__ = module_type(type(source))
        for child in source.children():
            if callable(getattr(child, "untransform_posterior", None)):
                external_outcome_module(child, like, None, None)
    if like is not None:
        source.to(like)
    if train_Y is not None:
        source.train()
        with torch.no_grad():
            source(train_Y, X=train_X)
    return source.eval()

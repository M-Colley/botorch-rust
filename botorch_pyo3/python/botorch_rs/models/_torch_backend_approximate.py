"""Torch protocol adapter for a caller's actual approximate GP module."""

from __future__ import annotations

import copy
import math
from itertools import chain
from typing import Any

import torch
from torch import Tensor, nn

from botorch_rs.exceptions.errors import UnsupportedError
from botorch_rs.likelihoods import GaussianLikelihood, MultitaskGaussianLikelihood
from botorch_rs.models._torch_backend_kronecker import TorchMultitaskGaussianLikelihood
from botorch_rs.models._torch_backend_modules import TorchGaussianLikelihood
from botorch_rs.models._torch_backend_variational import TorchSingleTaskVariationalGP
from botorch_rs.models.approximate_gp import ApproximateGPyTorchModel, _SingleTaskVariationalGP
from botorch_rs.posteriors._torch_backend import TorchGPyTorchPosterior, TorchMultivariateNormal
from botorch_rs.posteriors.gpytorch import GPyTorchPosterior
from botorch_rs.posteriors.torch import TorchPosterior


def _like(module: nn.Module) -> Tensor:
    return next(
        (
            value
            for value in chain(module.parameters(), module.buffers())
            if value.is_floating_point()
        ),
        torch.empty(0),
    )


def _tensor_gaussian(distribution: Any) -> TorchGPyTorchPosterior:
    if isinstance(distribution, TorchMultivariateNormal):
        return distribution._p
    if not isinstance(distribution, torch.distributions.MultivariateNormal):
        raise TypeError("A Gaussian likelihood requires a multivariate normal distribution.")
    mean, covariance = distribution.mean, distribution.covariance_matrix
    if len(distribution.event_shape) == 1:
        mean = mean.unsqueeze(-1)
    elif len(distribution.event_shape) == 2 and getattr(distribution, "_interleaved", True):
        q, m = mean.shape[-2:]
        covariance = (
            covariance.reshape(*covariance.shape[:-2], q, m, q, m)
            .transpose(-3, -4)
            .transpose(-1, -2)
        )
        covariance = covariance.reshape(*covariance.shape[:-4], q * m, q * m)
    return TorchGPyTorchPosterior(mean, covariance)


class _ScalarLikelihood(TorchGaussianLikelihood):
    def expected_log_prob(self, target: Tensor, function: Any, **kwargs: Any) -> Tensor:
        noise = kwargs.pop("noise", None)
        if kwargs:
            raise UnsupportedError(f"Unsupported Gaussian likelihood arguments: {sorted(kwargs)}.")
        noise = self.noise_for(function.mean.shape[-1], noise)
        return -0.5 * (
            ((target - function.mean).square() + function.variance) / noise
            + noise.log()
            + math.log(2 * math.pi)
        )


class _MultitaskLikelihood(TorchMultitaskGaussianLikelihood):
    def forward(self, distribution: Any, *inputs: Any, **kwargs: Any) -> Any:
        return super().forward(_tensor_gaussian(distribution).distribution, *inputs, **kwargs)

    def expected_log_prob(self, target: Tensor, function: Any, **kwargs: Any) -> Tensor:
        if kwargs:
            raise UnsupportedError(f"Unsupported multitask likelihood arguments: {sorted(kwargs)}.")
        noise = self.noise_matrix().diagonal(dim1=-1, dim2=-2).unsqueeze(-2)
        return (
            -0.5
            * (
                ((target - function.mean).square() + function.variance) / noise
                + noise.log()
                + math.log(2 * math.pi)
            )
        ).sum(-1)


def _likelihood(source: Any, outputs: int, like: Tensor) -> nn.Module:
    if isinstance(source, nn.Module):
        return source
    if source is None:
        source = (
            GaussianLikelihood() if outputs == 1 else MultitaskGaussianLikelihood(num_tasks=outputs)
        )
    if type(source) is GaussianLikelihood:
        return _ScalarLikelihood(source, like)
    if type(source) is MultitaskGaussianLikelihood:
        return _MultitaskLikelihood(
            source.num_tasks, source.rank, source.batch_shape, like, source=source
        )
    raise TypeError(
        "likelihood must be a Torch module or a supported port Gaussian likelihood specification."
    )


def _input_transform(source: Any, like: Tensor) -> nn.Module:
    from botorch_rs.models._backend_selection import native_port_descriptor_type
    from botorch_rs.models._torch_backend_modules import TorchInputChain, TorchNormalize, TorchWarp
    from botorch_rs.models.transforms.input import ChainedInputTransform, Normalize, Warp

    source_type = native_port_descriptor_type(source) or type(source)
    if source_type is Normalize:
        result: nn.Module = TorchNormalize(source, source._d, like)
    elif source_type is Warp:
        result = TorchWarp(source, source._d, like)
    elif source_type is ChainedInputTransform:
        result = TorchInputChain(
            {name: _input_transform(child, like) for name, child in source.items()}, like
        )
    else:
        raise UnsupportedError(
            "Attach a Torch input transform module, or a port Normalize, Warp or ChainedInputTransform specification."
        )
    return result.train(source.training)


def _outcome_transform(source: Any, like: Tensor, outputs: int) -> nn.Module:
    from botorch_rs.models._backend_selection import native_port_descriptor_type
    from botorch_rs.models._torch_backend_modules import (
        TorchElementwiseOutcome,
        TorchOutcomeChain,
        TorchStandardize,
    )
    from botorch_rs.models.transforms.outcome import (
        Bilog,
        ChainedOutcomeTransform,
        Log,
        Power,
        Standardize,
    )

    source_type = native_port_descriptor_type(source) or type(source)
    if source_type is Standardize:
        if source._m != outputs:
            raise ValueError("Standardize's m must equal the declared number of outputs.")
        result: Any = TorchStandardize(
            source._m, source._batch_shape, like, source._outputs, source._min_stdv
        )
        result.means = torch.as_tensor(source.means, dtype=like.dtype, device=like.device).clone()
        result.stdvs = torch.as_tensor(source.stdvs, dtype=like.dtype, device=like.device).clone()
        result._is_trained = source._is_trained
    elif source_type in (Log, Bilog, Power):
        result = TorchElementwiseOutcome(source, outputs)
    elif source_type is ChainedOutcomeTransform:
        result = TorchOutcomeChain(
            {name: _outcome_transform(child, like, outputs) for name, child in source.items()}
        )
    else:
        raise UnsupportedError(
            "Attach a Torch outcome transform module, or a port Standardize, Log, Bilog, Power or ChainedOutcomeTransform specification."
        )
    return result.train(source.training)


class TorchApproximateGPyTorchModel(nn.Module, ApproximateGPyTorchModel):
    """Preserve a supplied module, its inference law, state and parameter leaves."""

    _tensor_backend = "torch"
    model: Any
    likelihood: Any

    def __init__(
        self,
        model: nn.Module,
        likelihood: Any = None,
        num_outputs: int = 1,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        nn.Module.__init__(self)
        if not isinstance(model, nn.Module):
            raise TypeError("model must be a Torch module.")
        if isinstance(num_outputs, bool) or not isinstance(num_outputs, int) or num_outputs < 1:
            raise ValueError("num_outputs must be a positive integer.")
        self.model = model
        self.likelihood = _likelihood(likelihood, num_outputs, _like(model))
        self._desired_num_outputs = num_outputs

    def __setattr__(self, name: str, value: Any) -> None:
        if (
            name in ("input_transform", "outcome_transform")
            and value is not None
            and not isinstance(value, nn.Module)
        ):
            like = _like(self.model)
            value = (
                _input_transform(value, like)
                if name == "input_transform"
                else _outcome_transform(value, like, self.num_outputs)
            )
        nn.Module.__setattr__(self, name, value)

    @property
    def batch_shape(self) -> Any:
        batch = getattr(self.model, "batch_shape", None)
        if batch is not None:
            return torch.Size(batch)
        batch = getattr(self.model, "_input_batch_shape", None)
        if batch is not None:
            return torch.Size(batch)
        strategy = getattr(self.model, "variational_strategy", None)
        base = getattr(strategy, "base_variational_strategy", strategy)
        distribution = getattr(base, "_variational_distribution", None)
        if distribution is None:
            distribution = getattr(strategy, "variational_distribution", None)
        shape = getattr(distribution, "batch_shape", None)
        if shape is None:
            raise NotImplementedError(
                "The supplied model must expose batch_shape or variational-distribution batch metadata."
            )
        task_dim = getattr(strategy, "task_dim", None)
        if task_dim is None:
            task_dim = getattr(strategy, "latent_dim", None)
        if task_dim is not None:
            shape = list(shape)
            del shape[task_dim]
        return torch.Size(shape)

    def forward(self, X: Tensor) -> Any:
        return self.model(self.transform_inputs(X) if self.training else X)

    def posterior(
        self,
        X: Tensor,
        output_indices: Any = None,
        observation_noise: bool = False,
        posterior_transform: Any = None,
        **kwargs: Any,
    ) -> Any:
        if kwargs:
            raise TypeError(f"Unexpected posterior arguments: {sorted(kwargs)}.")
        if output_indices is not None:
            raise NotImplementedError(
                f"{self.__class__.__name__}.posterior does not support output indices."
            )
        if not isinstance(observation_noise, bool):
            raise TypeError("observation_noise must be a bool for this approximate-model wrapper.")
        self.eval()
        X = self.transform_inputs(X)
        if self.num_outputs > 1 and X.ndim > 2:
            X = X.unsqueeze(-3).expand(*X.shape[:-2], self.num_outputs, *X.shape[-2:])
        distribution = self.model(X)
        if observation_noise:
            distribution = self.likelihood(distribution)
        if isinstance(distribution, TorchMultivariateNormal):
            posterior: Any = distribution._p
        elif isinstance(distribution, torch.distributions.MultivariateNormal) and hasattr(
            distribution, "lazy_covariance_matrix"
        ):
            posterior = GPyTorchPosterior(distribution=distribution)
        else:
            posterior = TorchPosterior(distribution=distribution)
        transform = getattr(self, "outcome_transform", None)
        if transform is not None:
            posterior = transform.untransform_posterior(posterior, X=X)
        return (
            posterior
            if posterior_transform is None
            else posterior_transform(posterior=posterior, X=X)
        )

    def __deepcopy__(self, memo: dict[int, Any]) -> Any:
        duplicate = object.__new__(type(self))
        memo[id(self)] = duplicate
        duplicate.__dict__ = copy.deepcopy(self.__dict__, memo)
        return duplicate


class TorchDefaultApproximateGPyTorchModel(TorchSingleTaskVariationalGP):
    """The generic default constructor, using the port's actual variational owner."""

    _has_train_inputs = False

    def __init__(
        self,
        model: Any = None,
        likelihood: Any = None,
        num_outputs: int = 1,
        *args: Any,
        backend: str = "torch",
        **kwargs: Any,
    ) -> None:
        from inspect import signature

        if model is not None:
            raise TypeError("This constructor builds the default approximate model.")
        if isinstance(num_outputs, bool) or not isinstance(num_outputs, int) or num_outputs < 1:
            raise ValueError("num_outputs must be a positive integer.")
        arguments = signature(_SingleTaskVariationalGP.__init__).bind(
            None, *args, num_outputs=num_outputs, **kwargs
        )
        arguments.apply_defaults()
        options = dict(arguments.arguments)
        options.pop("self")
        # The generic inner constructor documents train_Y as unused. In
        # particular, it must not infer num_outputs or register target graphs.
        options.pop("train_Y")
        if likelihood is None:
            likelihood = (
                GaussianLikelihood()
                if num_outputs == 1
                else MultitaskGaussianLikelihood(num_tasks=num_outputs)
            )
        TorchSingleTaskVariationalGP.__init__(self, likelihood=likelihood, **options)

    __setattr__ = TorchApproximateGPyTorchModel.__setattr__

    def _fit_transforms(self) -> None:
        # Unlike SingleTaskVariationalGP, the generic wrapper never fits
        # attached transforms to stored constructor data. Their actual
        # train/eval forward calls own their updates.
        pass

    def posterior(
        self,
        X: Tensor,
        output_indices: Any = None,
        observation_noise: Any = False,
        posterior_transform: Any = None,
        **kwargs: Any,
    ) -> Any:
        if output_indices is not None:
            raise NotImplementedError(
                f"{self.__class__.__name__}.posterior does not support output indices."
            )
        if not isinstance(observation_noise, bool):
            raise TypeError("observation_noise must be a bool for this approximate-model wrapper.")
        return super().posterior(
            X,
            observation_noise=observation_noise,
            posterior_transform=posterior_transform,
            **kwargs,
        )

"""Supplied scalar likelihood protocols over the port's variational Gaussian."""

from __future__ import annotations

from typing import Any

import torch
from gpytorch.distributions import MultitaskMultivariateNormal, MultivariateNormal
from gpytorch.likelihoods import _OneDimensionalLikelihood
from linear_operator.operators import DenseLinearOperator, DiagLinearOperator
from torch import Tensor, nn

from botorch_rs import _botorch_rs as _rs


class _LikelihoodGaussian(MultivariateNormal):
    """The supplied likelihood sees its expected type and a full sampling graph."""

    def rsample(self, sample_shape: Any = (), base_samples: Any = None) -> Tensor:
        if base_samples is None:
            base_samples = self.get_base_samples(torch.Size(sample_shape))
        root = _rs.tensor_psd_safe_cholesky(self.covariance_matrix)
        return self.mean + (root @ base_samples.unsqueeze(-1)).squeeze(-1)


class _LikelihoodMultitaskGaussian(MultitaskMultivariateNormal):
    def rsample(self, sample_shape: Any = (), base_samples: Any = None) -> Tensor:
        if base_samples is None:
            base_samples = self.get_base_samples(torch.Size(sample_shape))
        root = _rs.tensor_psd_safe_cholesky(self.covariance_matrix)
        centered = (root @ base_samples.flatten(-2).unsqueeze(-1)).squeeze(-1)
        return self.mean + centered.reshape(*centered.shape[:-1], *self.event_shape)


def gaussian(function: Any, *, diagonal: bool = False) -> Any:
    """Supply a Gaussian distribution without invoking an upstream GP model."""
    multitask = bool(getattr(function, "_multitask", False))
    variance = function.variance.flatten(-2) if multitask else function.variance
    covariance = (
        DiagLinearOperator(variance)
        if diagonal
        else DenseLinearOperator(function.covariance_matrix)
    )
    cls: Any = _LikelihoodMultitaskGaussian if multitask else _LikelihoodGaussian
    return cls(function.mean, covariance)


class _ExpectedLogProbability(nn.Module):
    likelihood: Any

    def __init__(self, likelihood: nn.Module, multitask: bool) -> None:
        super().__init__()
        self.likelihood = likelihood
        self.multitask = multitask

    def forward(
        self,
        target: Tensor,
        mean: Tensor,
        variance: Tensor,
        covariance: Any = None,
        **kwargs: Any,
    ) -> Tensor:
        # Scalar likelihood quadrature needs only the marginal variances;
        # an N x N training covariance is unnecessary for this ELBO term.
        cls: Any = _LikelihoodGaussian
        if self.multitask:
            mean, variance = mean.transpose(-1, -2), variance.transpose(-1, -2).flatten(-2)
            cls = _LikelihoodMultitaskGaussian
        distribution = cls(
            mean,
            DiagLinearOperator(variance) if covariance is None else DenseLinearOperator(covariance),
        )
        return self.likelihood.expected_log_prob(target, distribution, **kwargs)


def snapshot_expected_log_prob(likelihood: nn.Module, function: Any) -> Any:
    """Capture likelihood Tensor state while retaining its original leaf graph."""
    evaluator = _ExpectedLogProbability(likelihood, function._multitask)
    parameters = {name: value.clone() for name, value in evaluator.named_parameters()}
    buffers = {name: value.clone() for name, value in evaluator.named_buffers()}

    def expected(target: Tensor, mean: Tensor, variance: Tensor, **kwargs: Any) -> Tensor:
        covariance = (
            None
            if isinstance(likelihood, _OneDimensionalLikelihood)
            else function.covariance_matrix
        )
        return torch.func.functional_call(
            evaluator, (parameters, buffers), (target, mean, variance, covariance), kwargs
        )

    return expected

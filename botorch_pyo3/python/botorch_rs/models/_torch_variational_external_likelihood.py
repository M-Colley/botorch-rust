"""Supplied scalar likelihood protocols over the port's variational Gaussian."""

from __future__ import annotations

from typing import Any

import torch
from gpytorch.distributions import MultivariateNormal
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


def gaussian(function: Any, *, diagonal: bool = False) -> Any:
    """Supply a Gaussian distribution without invoking an upstream GP model."""
    covariance = (
        DiagLinearOperator(function.variance)
        if diagonal
        else DenseLinearOperator(function.covariance_matrix)
    )
    return _LikelihoodGaussian(function.mean, covariance)


class _ExpectedLogProbability(nn.Module):
    likelihood: Any

    def __init__(self, likelihood: nn.Module) -> None:
        super().__init__()
        self.likelihood = likelihood

    def forward(
        self, target: Tensor, mean: Tensor, variance: Tensor, covariance: Any = None
    ) -> Tensor:
        # Scalar likelihood quadrature needs only the marginal variances;
        # an N x N training covariance is unnecessary for this ELBO term.
        distribution = _LikelihoodGaussian(
            mean,
            DiagLinearOperator(variance) if covariance is None else DenseLinearOperator(covariance),
        )
        return self.likelihood.expected_log_prob(target, distribution)


def snapshot_expected_log_prob(likelihood: nn.Module, function: Any) -> Any:
    """Capture likelihood Tensor state while retaining its original leaf graph."""
    evaluator = _ExpectedLogProbability(likelihood)
    parameters = {name: value.clone() for name, value in evaluator.named_parameters()}
    buffers = {name: value.clone() for name, value in evaluator.named_buffers()}

    def expected(target: Tensor, mean: Tensor, variance: Tensor) -> Tensor:
        covariance = (
            None if isinstance(likelihood, _OneDimensionalLikelihood) else function._blocks()
        )
        return torch.func.functional_call(
            evaluator, (parameters, buffers), (target, mean, variance, covariance)
        )

    return expected

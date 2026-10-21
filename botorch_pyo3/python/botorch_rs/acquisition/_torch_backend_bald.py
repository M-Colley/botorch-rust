"""Device-resident Gaussian-mixture disagreement information gain."""

from __future__ import annotations

import inspect
import math
import warnings
from typing import Any

import torch
from torch import Tensor, nn

from botorch_rs.acquisition._torch_backend_acquisition import _TensorAcquisition
from botorch_rs.acquisition.bayesian_active_learning import (
    FULLY_BAYESIAN_ERROR_MSG,
    NEGATIVE_INFOGAIN_WARNING,
)
from botorch_rs.exceptions.errors import UnsupportedError
from botorch_rs.utils.transforms import is_fully_bayesian


class TensorBayesianActiveLearningByDisagreement(_TensorAcquisition):
    """BALD using joint component laws and common device base normals."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        from botorch_rs.sampling.normal import SobolQMCNormalSampler

        nn.Module.__init__(self)
        bound = inspect.signature(inspect.getattr_static(self._public_class, "__init__")).bind(
            self, *args, **kwargs
        )
        bound.apply_defaults()
        values = bound.arguments
        self.model = values["model"]
        if not is_fully_bayesian(self.model):
            raise RuntimeError(FULLY_BAYESIAN_ERROR_MSG)
        self._name = self._public_class.__name__
        self._is_analytic, self._log = False, False
        self.posterior_transform = values["posterior_transform"]
        self.sampler = values["sampler"]
        if self.sampler is None:
            self.sampler = SobolQMCNormalSampler((512,))
        self.set_X_pending(values["X_pending"])

    def forward(self, X: Tensor) -> Tensor:
        X = self._candidates_tensor(X)
        posterior = self.model.posterior(
            X, observation_noise=True, posterior_transform=self.posterior_transform
        )
        if not getattr(posterior, "_is_gaussian_mixture", False):
            raise UnsupportedError("BALD requires a Gaussian-mixture posterior.")
        mean = posterior.mean
        covariance = posterior.distribution.covariance_matrix
        samples = self.sampler(posterior)
        count, q, outputs = mean.shape[-3:]
        batch = mean.shape[:-3]
        sample_count = math.prod(self.sample_shape)
        if samples.shape != (*self.sample_shape, *batch, count, q, outputs):
            raise ValueError("The sampler must preserve the posterior's MCMC and event axes.")
        # Every native Tensor Gaussian uses output-major dense covariance.
        # Flatten both the means and samples in that same order, without
        # changing the posterior distribution's event-layout metadata.
        flat_mean = mean.transpose(-1, -2).reshape(*batch, count, q * outputs)
        draws = samples.reshape(sample_count, *batch, count, q, outputs)
        flat_draws = draws.transpose(-1, -2).reshape(sample_count, *batch, count, q * outputs)
        distribution = torch.distributions.MultivariateNormal(
            flat_mean, covariance_matrix=covariance
        )
        # Leading component axis indexes the generator of each draw; the
        # trailing component axis indexes the density being evaluated.
        cross_draws = flat_draws.movedim(-2, 0).unsqueeze(-2)
        log_prob = distribution.log_prob(cross_draws)
        mixture_entropy = -(torch.logsumexp(log_prob, dim=-1) - math.log(count)).mean((0, 1))
        conditional_entropy = -distribution.log_prob(flat_draws).mean((0, -1))
        information_gain = mixture_entropy - conditional_entropy
        if bool((information_gain < 0).any()):
            warnings.warn(NEGATIVE_INFOGAIN_WARNING, RuntimeWarning, stacklevel=2)
        return information_gain

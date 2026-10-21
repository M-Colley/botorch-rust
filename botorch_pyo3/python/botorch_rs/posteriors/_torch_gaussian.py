"""Read a supplied Gaussian posterior without changing its derivative graph."""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

from botorch_rs.exceptions.errors import UnsupportedError
from botorch_rs.posteriors._torch_backend import TorchGPyTorchPosterior


def gaussian_posterior(posterior: Any) -> TorchGPyTorchPosterior:
    if isinstance(posterior, TorchGPyTorchPosterior):
        return posterior
    distribution = getattr(posterior, "distribution", None)
    mean = getattr(posterior, "mean", None)
    if (
        not isinstance(distribution, torch.distributions.MultivariateNormal)
        or not isinstance(mean, Tensor)
        or mean.ndim < 2
    ):
        raise UnsupportedError("Device entropy requires a Gaussian posterior with q x m means.")
    covariance = distribution.covariance_matrix
    q, m = mean.shape[-2:]
    if covariance.shape[-2:] != (q * m, q * m):
        raise UnsupportedError("The Gaussian posterior mean and covariance event shapes disagree.")
    if m > 1 and getattr(distribution, "_interleaved", True):
        # The port stores all q locations by output. A supplied multitask
        # distribution may instead interleave its outputs at each location.
        indices = torch.arange(q * m, device=mean.device).reshape(q, m).mT.reshape(-1)
        covariance = covariance.index_select(-2, indices).index_select(-1, indices)
    return TorchGPyTorchPosterior(mean, covariance)

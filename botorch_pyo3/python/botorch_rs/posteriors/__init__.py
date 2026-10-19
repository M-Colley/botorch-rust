"""Posteriors (``botorch.posteriors``)."""

from botorch_rs.posteriors.gpytorch import (
    GPyTorchPosterior,
    MultitaskMultivariateNormal,
    MultivariateNormal,
    Posterior,
    scalarize_posterior,
    scalarize_posterior_gpytorch,
)
from botorch_rs.posteriors.higher_order import HigherOrderGPPosterior
from botorch_rs.posteriors.multitask import MultitaskGPPosterior
from botorch_rs.posteriors.posterior_list import PosteriorList
from botorch_rs.posteriors.torch import TorchPosterior
from botorch_rs.posteriors.transformed import TransformedPosterior

__all__ = [
    "GPyTorchPosterior",
    "HigherOrderGPPosterior",
    "MultitaskGPPosterior",
    "MultitaskMultivariateNormal",
    "MultivariateNormal",
    "Posterior",
    "PosteriorList",
    "TransformedPosterior",
    "TorchPosterior",
    "scalarize_posterior",
    "scalarize_posterior_gpytorch",
]

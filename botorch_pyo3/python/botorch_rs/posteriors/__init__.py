"""Posteriors (``botorch.posteriors``)."""

from botorch_rs.posteriors.gpytorch import GPyTorchPosterior, MultivariateNormal, Posterior

__all__ = ["GPyTorchPosterior", "MultivariateNormal", "Posterior"]

"""Preference acquisitions over device-resident, differentiable utility laws."""

from __future__ import annotations

import inspect
import math
from typing import Any

import torch
from torch import Tensor, nn

from botorch_rs.acquisition._torch_backend_acquisition import (
    _average_ensemble,
    _TensorAcquisition,
)
from botorch_rs.exceptions.errors import UnsupportedError


class TensorPreferenceAcquisition(_TensorAcquisition):
    previous_winner: Any
    std_noise: Any

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        from botorch_rs.acquisition.objective import IdentityMCObjective
        from botorch_rs.sampling.normal import SobolQMCNormalSampler

        nn.Module.__init__(self)
        bound = inspect.signature(inspect.getattr_static(self._public_class, "__init__")).bind(
            self, *args, **kwargs
        )
        bound.apply_defaults()
        values = bound.arguments
        self._name, self._is_analytic = self._public_class.__name__, False
        self._log = False
        self.model = values["pref_model"]
        if self.model.num_outputs != 1:
            raise UnsupportedError("Preference acquisitions require scalar utility models.")
        self.outcome_model = values.get("outcome_model")
        if self.outcome_model is not None and not callable(self.outcome_model):
            raise TypeError("outcome_model must be a callable Tensor model.")
        self.posterior_transform = values.get("posterior_transform")
        objective = values.get("objective")
        self.objective = IdentityMCObjective() if objective is None else objective
        self.sampler = values.get("sampler")
        if self._name == "qExpectedUtilityOfBestOption" and self.sampler is None:
            self.sampler = SobolQMCNormalSampler(sample_shape=torch.Size([512]))
        self.set_X_pending(values.get("X_pending"))
        winner = values.get("previous_winner")
        like = self.model.datapoints
        if like is None:
            raise UnsupportedError("Preference acquisitions require fitted training datapoints.")
        if winner is not None:
            winner = torch.as_tensor(winner, device=like.device, dtype=like.dtype)
            if winner.ndim not in (1, 2) or (winner.ndim == 2 and winner.shape[-2] != 1):
                raise ValueError("previous_winner must contain a single utility-input point.")
            winner = winner.reshape(1, -1)
            if winner.shape[-1] != like.shape[-1] or not bool(torch.isfinite(winner).all()):
                raise ValueError(
                    "previous_winner must be finite and match the preference dimension."
                )
        self.register_buffer("previous_winner", winner)
        samples = values.get("num_samples", 1024)
        self.num_samples = 1024 if samples is None else samples
        if type(self.num_samples) is not int or self.num_samples < 1:
            raise ValueError("num_samples must be a positive integer.")
        noise = values.get("std_noise")
        noise = torch.as_tensor(
            0.0 if noise is None else noise, dtype=like.dtype, device=like.device
        )
        if not bool(torch.isfinite(noise).all()) or bool((noise < 0).any()):
            raise ValueError("std_noise must contain finite nonnegative values.")
        self.register_buffer("std_noise", noise)

    def _prepare(self, X: Tensor) -> Tensor:
        if not isinstance(X, Tensor) or X.ndim < 2:
            raise ValueError("X must be a Tensor with shape (...,q,d).")
        if not bool(torch.isfinite(X).all()):
            raise ValueError("Candidate inputs must be finite.")
        if self._name == "AnalyticExpectedUtilityOfBestOption":
            expected = 2 if self.previous_winner is None else 1
            if X.shape[-2] != expected:
                raise UnsupportedError("Analytic EUBO requires q=2, or q=1 with a previous winner.")
        elif self._name == "PairwiseBayesianActiveLearningByDisagreement" and X.shape[-2] != 2:
            raise AssertionError("Pairwise BALD requires q=2.")
        X = X.unsqueeze(0) if X.ndim == 2 else X
        if self._X_pending is not None:
            pending = self._X_pending.to(X)
            batch = torch.broadcast_shapes(X.shape[:-2], pending.shape[:-2])
            X = torch.cat(
                (X.expand(*batch, *X.shape[-2:]), pending.expand(*batch, *pending.shape[-2:])), -2
            )
        Y = X if self.outcome_model is None else self.outcome_model(X)
        if not isinstance(Y, Tensor):
            raise TypeError("outcome_model must return a Tensor retaining its computation graph.")
        if self.previous_winner is not None:
            winner = self.previous_winner.to(Y)
            Y = torch.cat((Y, winner.expand(*Y.shape[:-2], 1, Y.shape[-1])), -2)
        return Y

    def set_X_pending(self, X_pending: Any = None) -> None:
        if self._name != "qExpectedUtilityOfBestOption":
            if X_pending is not None:
                raise UnsupportedError("Only qEUBO supports pending preference candidates.")
            self._X_pending = None
            return
        super().set_X_pending(X_pending)

    def forward(self, X: Tensor) -> Tensor:
        Y = self._prepare(X)
        posterior = self.model.posterior(Y, posterior_transform=self.posterior_transform)
        if self._name == "qExpectedUtilityOfBestOption":
            samples = self.sampler(posterior)
            objective = self.objective(samples=samples, X=Y)
            if not isinstance(objective, Tensor) or objective.shape[-1] != Y.shape[-2]:
                raise ValueError("The preference objective must return (...,q) Tensor values.")
            # This family follows the reference's first sample-axis reduction.
            return _average_ensemble(objective.max(-1).values.mean(0), self.model)
        mean = posterior.mean.squeeze(-1)
        covariance = posterior.distribution.covariance_matrix
        difference = mean[..., 0] - mean[..., 1]
        variance = covariance[..., 0, 0] + covariance[..., 1, 1] - 2 * covariance[..., 0, 1]
        if self._name == "AnalyticExpectedUtilityOfBestOption":
            sigma = variance.clamp_min(1e-8).sqrt()
            u = difference / sigma
            density = (-0.5 * u.square() - 0.5 * math.log(2 * math.pi)).exp()
            value = sigma * (density + u * torch.special.ndtr(u))
            if self.previous_winner is None:
                value = value + mean[..., 1]
            return _average_ensemble(value, self.model)
        variance = variance + 2 * self.std_noise
        sigma = variance.clamp_min(1e-8).sqrt()
        probability = torch.special.ndtr(difference / (variance + 1).sqrt())
        entropy = torch.distributions.Bernoulli(probs=probability).entropy()
        normals = torch.randn((self.num_samples, *difference.shape), dtype=Y.dtype, device=Y.device)
        probabilities = torch.special.ndtr(difference + sigma * normals)
        conditional_entropy = torch.distributions.Bernoulli(probs=probabilities).entropy().mean(0)
        return _average_ensemble(entropy - conditional_entropy, self.model)

"""Constrained analytic acquisitions and weighted posterior means on Torch."""

from __future__ import annotations

import inspect
from typing import Any

import torch
from torch import Tensor, nn

from botorch_rs.acquisition._torch_backend_acquisition import _average_ensemble, _log_ei
from botorch_rs.exceptions.errors import UnsupportedError
from botorch_rs.exceptions.warnings import legacy_ei_numerics_warning


def log_normal_interval(lower: Tensor, upper: Tensor) -> Tensor:
    """Log normal interval probability, evaluated in the accurate negative tail."""
    reverse = upper.abs() > lower.abs()
    a = torch.where(reverse, -upper, lower)
    b = torch.where(reverse, -lower, upper)
    log_a, log_b = torch.special.log_ndtr(a), torch.special.log_ndtr(b)
    return log_b + (-(log_a - log_b).expm1()).log()


class TensorAnalyticExtra(nn.Module):
    """Mixin retaining Gaussian moment, threshold, and constraint graphs."""

    _public_class: type
    model: Any
    posterior_transform: Any
    weights: Tensor
    best_f: Tensor

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        nn.Module.__init__(self)
        bound = inspect.signature(inspect.getattr_static(self._public_class, "__init__")).bind(
            self, *args, **kwargs
        )
        bound.apply_defaults()
        values = dict(bound.arguments)
        self._tensor_backend = "torch"
        self._name = self._public_class.__name__
        self.model = values["model"]
        self.posterior_transform = values.get("posterior_transform")
        self.maximize = values.get("maximize", True)
        if self._name in {"ScalarizedPosteriorMean", "qAnalyticProbabilityOfImprovement"}:
            if self.model.num_outputs != 1 and self.posterior_transform is None:
                raise UnsupportedError("A multi-output model requires a posterior transform.")
            if self._name == "qAnalyticProbabilityOfImprovement":
                self.register_buffer("best_f", torch.as_tensor(values["best_f"]))
                return
            weights = torch.as_tensor(values["weights"])
            if weights.ndim != 1:
                raise ValueError("weights must be a one-dimensional tensor.")
            self.register_buffer("weights", weights)
            return
        self.constraints = values["constraints"]
        self.objective_index = int(values.get("objective_index", 0))
        if not self.constraints:
            raise ValueError("There must be at least one constraint.")
        m = self.model.num_outputs
        if not -m <= self.objective_index < m:
            raise ValueError("The objective output index is out of range.")
        self.objective_index %= m
        self._constraint_entries: list[tuple[int, str | None, str | None]] = []
        for position, (index, (lower, upper)) in enumerate(self.constraints.items()):
            if not -m <= index < m:
                raise ValueError("A constraint output index is out of range.")
            index %= m
            if index == self.objective_index and "ExpectedImprovement" in self._name:
                raise ValueError("Output corresponding to objective should not be a constraint.")
            if lower is None and upper is None:
                raise ValueError("A constraint requires at least one finite bound.")
            lo_name = None if lower is None else f"_lower_{position}"
            hi_name = None if upper is None else f"_upper_{position}"
            if lo_name is not None:
                self.register_buffer(lo_name, torch.as_tensor(lower))
            if hi_name is not None:
                self.register_buffer(hi_name, torch.as_tensor(upper))
            if (
                lower is not None
                and upper is not None
                and not bool(torch.as_tensor(lower) < torch.as_tensor(upper))
            ):
                raise ValueError("Lower bounds must be less than upper bounds.")
            self._constraint_entries.append((index, lo_name, hi_name))
        if "best_f" in values:
            self.register_buffer("best_f", torch.as_tensor(values["best_f"]))
        if self._name == "ConstrainedExpectedImprovement":
            legacy_ei_numerics_warning(legacy_name=self._name)

    def set_X_pending(self, X_pending: Any = None) -> None:
        raise UnsupportedError("Analytic acquisition functions do not account for X_pending yet.")

    def _compute_log_prob_feas(self, means: Tensor, sigmas: Tensor) -> Tensor:
        result = means.new_zeros(means.shape[:-1])
        for index, lo_name, hi_name in self._constraint_entries:
            mean, sigma = means[..., index], sigmas[..., index]
            if lo_name is None:
                assert hi_name is not None
                upper = getattr(self, hi_name).to(mean)
                probability = torch.special.log_ndtr((upper - mean) / sigma)
            elif hi_name is None:
                lower = getattr(self, lo_name).to(mean)
                probability = torch.special.log_ndtr((mean - lower) / sigma)
            else:
                lower = (getattr(self, lo_name).to(mean) - mean) / sigma
                upper = (getattr(self, hi_name).to(mean) - mean) / sigma
                probability = log_normal_interval(lower, upper)
            result = result + probability
        return result

    def forward(self, X: Tensor) -> Tensor:
        if not isinstance(X, Tensor) or X.ndim < 2:
            raise ValueError("X must have shape batch_shape x q x d.")
        X = X.unsqueeze(0) if X.ndim == 2 else X
        if (
            self._name not in {"ScalarizedPosteriorMean", "qAnalyticProbabilityOfImprovement"}
            and X.shape[-2] != 1
        ):
            raise ValueError("Constrained analytic acquisitions require q=1.")
        posterior = self.model.posterior(X, posterior_transform=self.posterior_transform)
        if self._name == "ScalarizedPosteriorMean":
            if posterior.mean.shape[-1] != 1 or X.shape[-2] != self.weights.numel():
                raise ValueError("The posterior must have one output and q must match weights.")
            return _average_ensemble(posterior.mean.squeeze(-1) @ self.weights.to(X), self.model)
        from botorch_rs.posteriors.gpytorch import GPyTorchPosterior

        if not isinstance(posterior, GPyTorchPosterior):
            raise UnsupportedError(
                "Analytic Gaussian constraints require a Gaussian posterior; use an MC acquisition for nonlinear outcome transforms."
            )
        if self._name == "qAnalyticProbabilityOfImprovement":
            from botorch_rs.utils.probability.mvnxpb import MVNXPB

            if posterior.mean.shape[-1] != 1:
                raise UnsupportedError("The posterior transform must produce one output.")
            centered = self.best_f.to(X).unsqueeze(-1) - posterior.distribution.mean
            infinite = torch.full_like(centered, -torch.inf if self.maximize else torch.inf)
            bounds = torch.stack(
                (infinite, centered) if self.maximize else (centered, infinite), dim=-1
            )
            value = -MVNXPB(posterior.distribution.covariance_matrix, bounds).solve().expm1()
            return _average_ensemble(value, self.model)
        means = posterior.mean.squeeze(-2)
        sigmas = posterior.variance.clamp_min(1e-12).sqrt().squeeze(-2)
        log_probability = self._compute_log_prob_feas(means, sigmas)
        if self._name == "LogProbabilityOfFeasibility":
            return _average_ensemble(log_probability, self.model, log=True)
        mean, sigma = means[..., self.objective_index], sigmas[..., self.objective_index]
        sign = 1 if self.maximize else -1
        u = sign * (mean - self.best_f.to(mean)) / sigma
        log_improvement = _log_ei(u) + sigma.log() + log_probability
        log = self._name.startswith("Log")
        value = log_improvement if log else log_improvement.exp()
        return _average_ensemble(value, self.model, log)

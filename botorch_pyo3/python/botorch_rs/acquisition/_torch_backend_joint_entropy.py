"""Device-resident joint entropy search with exact conditional Gaussian scaling."""

from __future__ import annotations

import inspect
import math
from typing import Any

import torch
from torch import Tensor, nn

from botorch_rs.acquisition._torch_backend_acquisition import _TensorAcquisition
from botorch_rs.acquisition._torch_backend_entropy import _gaussian
from botorch_rs.exceptions.errors import UnsupportedError
from botorch_rs.sampling.normal import SobolQMCNormalSampler

CLAMP_LB = torch.finfo(torch.float32).eps
LOG_2PI = math.log(2 * math.pi)


class TensorJointEntropySearch(_TensorAcquisition):
    """LB and MC JES, retaining the port's corrected marginal-entropy law."""

    optimal_inputs: Tensor
    optimal_outputs: Tensor

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        nn.Module.__init__(self)
        bound = inspect.signature(inspect.getattr_static(self._public_class, "__init__")).bind(
            self, *args, **kwargs
        )
        bound.apply_defaults()
        values = bound.arguments
        model = values["model"]
        if getattr(model, "batch_shape", ()):
            raise UnsupportedError("JES currently requires an initially unbatched model.")
        if getattr(model, "_is_fully_bayesian", False):
            raise UnsupportedError("JES conditioning on fully Bayesian mixtures is not supported.")
        inputs, outputs = values["optimal_inputs"], values["optimal_outputs"]
        if (
            not isinstance(inputs, Tensor)
            or not isinstance(outputs, Tensor)
            or inputs.ndim != 2
            or outputs.ndim != 2
            or inputs.shape[0] == 0
            or outputs.shape != (inputs.shape[0], model.num_outputs)
            or inputs.device != outputs.device
            or inputs.dtype != outputs.dtype
        ):
            raise ValueError(
                "Optimal inputs and outputs must be matching nonempty n x d/m tensors."
            )
        self._name = "qJointEntropySearch"
        self._is_analytic = False
        self.initial_model = self.model = model
        self.posterior_transform = values["posterior_transform"]
        if model.num_outputs > 1 and self.posterior_transform is None:
            raise UnsupportedError("Multi-output JES requires a scalarizing posterior transform.")
        self.estimation_type = values["estimation_type"]
        if self.estimation_type not in ("LB", "MC"):
            raise ValueError("Estimation type must be 'LB' or 'MC'.")
        self.condition_noiseless = bool(values["condition_noiseless"])
        self.num_samples = inputs.shape[0]
        self.register_buffer("optimal_inputs", inputs.unsqueeze(-2))
        self.register_buffer("optimal_outputs", outputs.unsqueeze(-2))
        self.register_buffer("_X_pending", None)
        self.sampler = values["sampler"]
        self._default_sample_shape = torch.Size((values["num_samples"],))
        if len(self.sample_shape) != 1 or self.sample_shape[0] < 1:
            raise UnsupportedError("JES requires one positive Monte Carlo sample dimension.")
        self.set_X_pending(values["X_pending"])
        noise = torch.full_like(self.optimal_outputs, 1e-4) if self.condition_noiseless else None
        children = getattr(model, "models", None)
        # condition_on_observations takes raw locations. Passing transformed
        # locations here applies Normalize / Warp twice in the reference.
        self.conditional_model = model.condition_on_observations(
            self.optimal_inputs if children is None else [self.optimal_inputs] * len(children),
            self.optimal_outputs,
            noise=noise,
        )

    @property
    def sample_shape(self) -> Any:
        return self._default_sample_shape if self.sampler is None else self.sampler.sample_shape

    def get_posterior_samples(self, posterior: Any) -> Tensor:
        if self.sampler is None:
            seed = int(torch.randint(0, 1_000_000, (1,)).item())
            self.sampler = SobolQMCNormalSampler(self._default_sample_shape, seed=seed)
        return self.sampler(posterior)

    @property
    def optimal_output_values(self) -> Tensor:
        if self.posterior_transform is None:
            return self.optimal_outputs
        return self.posterior_transform.evaluate(
            self.optimal_outputs, X=self.optimal_inputs
        ).unsqueeze(-1)

    def _statistics(self, X: Tensor) -> tuple[Any, Any, Tensor]:
        initial = _gaussian(self.initial_model, X, self.posterior_transform, noise=True)
        conditional = _gaussian(
            self.conditional_model, X.unsqueeze(-3), self.posterior_transform, noise=True
        )
        latent = _gaussian(
            self.conditional_model, X.unsqueeze(-3), self.posterior_transform
        ).variance
        return initial, conditional, latent

    def _compute_lower_bound_information_gain(
        self, X: Tensor, return_parts: bool = False
    ) -> Tensor:
        initial, conditional, latent = self._statistics(X)
        threshold = (self.optimal_output_values - conditional.mean) / latent.sqrt()
        ratio = (-0.5 * (threshold.square() + LOG_2PI)).exp() / torch.special.ndtr(
            threshold
        ).clamp_min(CLAMP_LB)
        truncated = latent * (1 - (threshold + ratio) * ratio).clamp_min(CLAMP_LB)
        truncated = truncated + conditional.variance - latent
        entropy = 0.5 * truncated.log().sum((-1, -2)).mean(-1)
        return (
            0.5 * torch.linalg.slogdet(initial.distribution.covariance_matrix).logabsdet - entropy
        )

    def _compute_monte_carlo_variables(self, posterior: Any) -> tuple[Tensor, Tensor]:
        samples = self.get_posterior_samples(posterior)
        variance = posterior.variance.clamp_min(CLAMP_LB)
        standardized = (samples - posterior.mean) / variance.sqrt()
        # JES sums marginal conditional entropies, not q copies of a joint
        # log density. This also makes independent points additive.
        log_probability = -0.5 * (standardized.square() + variance.log() + LOG_2PI)
        return samples, log_probability

    def _compute_monte_carlo_information_gain(
        self, X: Tensor, return_parts: bool = False
    ) -> Tensor:
        initial, conditional, latent = self._statistics(X)
        variance = conditional.variance.clamp_min(CLAMP_LB)
        samples, log_probability = self._compute_monte_carlo_variables(conditional)
        standardized = (samples - conditional.mean) / variance.sqrt()
        rho = (latent / variance).sqrt()
        threshold = (self.optimal_output_values - conditional.mean) / latent.sqrt()
        conditional_z = (threshold - rho * standardized) / (1 - rho.square()).clamp_min(
            CLAMP_LB
        ).sqrt()
        ratio = torch.special.ndtr(conditional_z).clamp_min(CLAMP_LB) / torch.special.ndtr(
            threshold
        ).clamp_min(CLAMP_LB)
        entropy = -(ratio * (ratio.log() + log_probability)).mean(0).sum((-1, -2)).mean(-1)
        initial_entropy = (
            0.5 * torch.linalg.slogdet(initial.distribution.covariance_matrix).logabsdet
        )
        return initial_entropy + 0.5 * (1 + LOG_2PI) * X.shape[-2] - entropy

    def forward(self, X: Tensor) -> Tensor:
        X = self._candidates_tensor(X)
        if self.estimation_type == "LB":
            return self._compute_lower_bound_information_gain(X)
        return self._compute_monte_carlo_information_gain(X)

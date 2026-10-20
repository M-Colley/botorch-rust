"""Explicit mathematical corrections to the official JES fixture oracle."""

from typing import Any

import torch
from botorch.acquisition import joint_entropy_search as bt_jes


def correct_jes_monte_carlo(acquisition: Any) -> None:
    """Apply explicit mathematical corrections to the pinned JES oracle.

    b0dd01be divides the conditional CDF by its variance rather than its
    standard deviation. Its q>1 density/constants and transformed optimal
    outputs are also corrected here. This is a corrected-reference comparison,
    not an assertion of agreement with the unmodified implementation.
    """
    import math
    import types

    def information_gain(self: Any, X: Any, return_parts: bool = False) -> Any:
        initial = self.initial_model.posterior(
            X, observation_noise=True, posterior_transform=self.posterior_transform
        )
        posterior = self.conditional_model.posterior(
            X.unsqueeze(-3), observation_noise=True, posterior_transform=self.posterior_transform
        )
        latent = self.conditional_model.posterior(
            X.unsqueeze(-3), posterior_transform=self.posterior_transform
        ).variance
        variance = posterior.variance.clamp_min(bt_jes.CLAMP_LB)
        samples = self.get_posterior_samples(posterior)
        standardized = (samples - posterior.mean) / variance.sqrt()
        rho = (latent / variance).sqrt()
        threshold = (self.optimal_output_values - posterior.mean) / latent.sqrt()
        conditional_z = (threshold - rho * standardized) / (1 - rho.square()).clamp_min(
            bt_jes.CLAMP_LB
        ).sqrt()
        ratio = torch.special.ndtr(conditional_z).clamp_min(bt_jes.CLAMP_LB) / torch.special.ndtr(
            threshold
        ).clamp_min(bt_jes.CLAMP_LB)
        log_density = -0.5 * (standardized.square() + variance.log() + math.log(2 * math.pi))
        conditional_entropy = -(ratio * (ratio.log() + log_density)).mean(0).sum((-1, -2))
        initial_entropy = 0.5 * torch.linalg.slogdet(initial.mvn.covariance_matrix).logabsdet
        initial_entropy = initial_entropy + bt_jes.MC_ADD_TERM * X.shape[-2]
        return initial_entropy - conditional_entropy.mean(-1)

    acquisition._compute_monte_carlo_information_gain = types.MethodType(
        information_gain, acquisition
    )

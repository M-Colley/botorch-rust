"""Model utilities (``botorch.models.utils``)."""

from botorch_rs.models.utils.gpytorch_modules import (
    get_covar_module_with_dim_scaled_prior,
    get_gaussian_likelihood_with_gamma_prior,
    get_gaussian_likelihood_with_lognormal_prior,
    get_matern_kernel_with_gamma_prior,
)
from botorch_rs.models.utils.priors import BetaPrior

__all__ = [
    "BetaPrior",
    "get_covar_module_with_dim_scaled_prior",
    "get_gaussian_likelihood_with_gamma_prior",
    "get_gaussian_likelihood_with_lognormal_prior",
    "get_matern_kernel_with_gamma_prior",
]

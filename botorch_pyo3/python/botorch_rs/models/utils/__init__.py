"""Model utilities (``botorch.models.utils``)."""

from botorch_rs.models.utils.assorted import (
    _make_X_full,
    add_output_dim,
    check_min_max_scaling,
    check_no_nans,
    check_standardization,
    consolidate_duplicates,
    detect_duplicates,
    extract_targets_and_noise_single_output,
    fantasize,
    get_task_value_remapping,
    gpt_posterior_settings,
    mod_batch_shape,
    multioutput_to_batch_mode_transform,
    validate_input_scaling,
)
from botorch_rs.models.utils.gpytorch_modules import (
    get_covar_module_with_dim_scaled_prior,
    get_gaussian_likelihood_with_gamma_prior,
    get_gaussian_likelihood_with_lognormal_prior,
    get_matern_kernel_with_gamma_prior,
)
from botorch_rs.models.utils.priors import BetaPrior

__all__ = [
    "BetaPrior",
    "_make_X_full",
    "add_output_dim",
    "check_min_max_scaling",
    "check_no_nans",
    "check_standardization",
    "consolidate_duplicates",
    "detect_duplicates",
    "extract_targets_and_noise_single_output",
    "fantasize",
    "get_covar_module_with_dim_scaled_prior",
    "get_gaussian_likelihood_with_gamma_prior",
    "get_gaussian_likelihood_with_lognormal_prior",
    "get_matern_kernel_with_gamma_prior",
    "get_task_value_remapping",
    "gpt_posterior_settings",
    "mod_batch_shape",
    "multioutput_to_batch_mode_transform",
    "validate_input_scaling",
]

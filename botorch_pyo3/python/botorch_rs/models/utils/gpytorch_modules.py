"""BoTorch's default GP modules (``botorch.models.utils.gpytorch_modules``)."""

from __future__ import annotations

import math
from typing import Any

from botorch_rs import _botorch_rs as _rs
from botorch_rs.constraints import GreaterThan
from botorch_rs.kernels import (
    Kernel,
    MaternKernel,
    RBFKernel,
    _check_batch_shape,
    _kernel_from_spec,
)
from botorch_rs.likelihoods import GaussianLikelihood
from botorch_rs.priors import LogNormalPrior

#: Lower bound on the inferred noise variance of the default likelihoods.
MIN_INFERRED_NOISE_LEVEL = 1e-4
SQRT2 = math.sqrt(2.0)
SQRT3 = math.sqrt(3.0)


def get_covar_module_with_dim_scaled_prior(
    ard_num_dims: int,
    batch_shape: Any = None,
    use_rbf_kernel: bool = True,
    active_dims: Any = None,
    nu: float = 2.5,
    ard: bool = True,
) -> Kernel:
    """An RBF or Matern kernel with the dimension-scaled lengthscale prior
    ``LogNormal(sqrt(2) + log(d) / 2, sqrt(3))`` of Hvarfner et al. (2024),
    BoTorch's ``SingleTaskGP`` default."""
    if not use_rbf_kernel and nu not in {0.5, 1.5, 2.5}:
        raise ValueError(f"Expected `nu` to be 0.5, 1.5, or 2.5, got {nu}.")
    shape = _check_batch_shape(batch_shape)
    if (use_rbf_kernel or nu == 2.5) and ard:
        spec = _rs.KernelSpec.dim_scaled(ard_num_dims, use_rbf_kernel)
        return _kernel_from_spec(
            spec,
            lambda d: _rs.KernelSpec.dim_scaled(d, use_rbf_kernel),
            shape,
            active_dims=active_dims,
        )
    if int(ard_num_dims) <= 0:
        raise ValueError(f"ard_num_dims must be positive, got {ard_num_dims}.")
    lengthscale_prior = LogNormalPrior(SQRT2 + 0.5 * math.log(int(ard_num_dims)), SQRT3)
    lengthscale_constraint = GreaterThan(
        2.5e-2, transform=None, initial_value=lengthscale_prior.mode
    )
    common_kwargs: dict[str, Any] = {
        "ard_num_dims": int(ard_num_dims) if ard else None,
        "batch_shape": shape,
        "active_dims": active_dims,
        "lengthscale_prior": lengthscale_prior,
        "lengthscale_constraint": lengthscale_constraint,
    }
    if use_rbf_kernel:
        return RBFKernel(**common_kwargs)
    return MaternKernel(nu=nu, **common_kwargs)


def get_matern_kernel_with_gamma_prior(ard_num_dims: int, batch_shape: Any = None) -> Kernel:
    """BoTorch's legacy kernel: ``ScaleKernel(MaternKernel(nu=2.5,
    lengthscale_prior=GammaPrior(3.0, 6.0)), outputscale_prior=GammaPrior(2.0, 0.15))``."""
    shape = _check_batch_shape(batch_shape)
    spec = _rs.KernelSpec.matern_gamma(ard_num_dims)
    return _kernel_from_spec(spec, _rs.KernelSpec.matern_gamma, shape)


def get_gaussian_likelihood_with_lognormal_prior(batch_shape: Any = None) -> GaussianLikelihood:
    """BoTorch's default likelihood: noise prior ``LogNormal(-4, 1)``, noise
    constrained to at least 1e-4 (bound imposed by the optimizer)."""
    return GaussianLikelihood._from_spec(
        _rs.GaussianLikelihoodSpec.lognormal(), _check_batch_shape(batch_shape)
    )


def get_gaussian_likelihood_with_gamma_prior(batch_shape: Any = None) -> GaussianLikelihood:
    """BoTorch's legacy likelihood: noise prior ``Gamma(1.1, 0.05)``."""
    return GaussianLikelihood._from_spec(
        _rs.GaussianLikelihoodSpec.gamma(), _check_batch_shape(batch_shape)
    )


__all__ = [
    "MIN_INFERRED_NOISE_LEVEL",
    "SQRT2",
    "SQRT3",
    "get_covar_module_with_dim_scaled_prior",
    "get_gaussian_likelihood_with_gamma_prior",
    "get_gaussian_likelihood_with_lognormal_prior",
    "get_matern_kernel_with_gamma_prior",
]

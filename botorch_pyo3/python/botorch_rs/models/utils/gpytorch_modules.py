"""BoTorch's default GP modules (``botorch.models.utils.gpytorch_modules``)."""

from __future__ import annotations

from typing import Any

from botorch_rs import _botorch_rs as _rs
from botorch_rs.kernels import Kernel, _check_batch_shape, _kernel_from_spec
from botorch_rs.likelihoods import GaussianLikelihood

#: Lower bound on the inferred noise variance of the default likelihoods.
MIN_INFERRED_NOISE_LEVEL = 1e-4


def get_covar_module_with_dim_scaled_prior(
    ard_num_dims: int,
    batch_shape: Any = None,
    use_rbf_kernel: bool = True,
    active_dims: Any = None,
) -> Kernel:
    """An RBF (or Matern-5/2) kernel with the dimension-scaled lengthscale prior
    ``LogNormal(sqrt(2) + log(d) / 2, sqrt(3))`` of Hvarfner et al. (2024),
    BoTorch's ``SingleTaskGP`` default."""
    shape = _check_batch_shape(batch_shape)
    spec = _rs.KernelSpec.dim_scaled(ard_num_dims, use_rbf_kernel)
    return _kernel_from_spec(
        spec,
        lambda d: _rs.KernelSpec.dim_scaled(d, use_rbf_kernel),
        shape,
        active_dims=active_dims,
    )


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
    "get_covar_module_with_dim_scaled_prior",
    "get_gaussian_likelihood_with_gamma_prior",
    "get_gaussian_likelihood_with_lognormal_prior",
    "get_matern_kernel_with_gamma_prior",
]

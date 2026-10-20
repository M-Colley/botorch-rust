"""Optional Gaussian distribution interop, separate from native GP inference."""

from __future__ import annotations

from contextlib import ExitStack
from typing import Any


class GaussianDistributionAdapter:
    """Preserve an explicitly supplied Gaussian distribution's Torch graph."""

    def __init__(self, distribution: Any) -> None:
        import torch

        if not isinstance(distribution, torch.distributions.MultivariateNormal):
            raise TypeError("distribution must be a Torch or GPyTorch multivariate normal.")
        self.distribution: Any = distribution
        self.is_multitask = len(distribution.event_shape) == 2
        if len(distribution.event_shape) not in (1, 2):
            raise ValueError("A Gaussian posterior requires one or two event dimensions.")
        self.is_gpytorch = hasattr(distribution, "lazy_covariance_matrix")
        # Register the optional distribution hierarchy only after the caller
        # explicitly supplies one; package import must remain Torch-free.
        from botorch_rs.sampling.get_sampler import GetSampler, _get_sampler_mvn

        normal_type = torch.distributions.MultivariateNormal
        if (normal_type,) not in GetSampler.funcs:
            GetSampler.register(normal_type)(_get_sampler_mvn)

    @property
    def mean(self) -> Any:
        mean = self.distribution.mean
        return mean if self.is_multitask else mean.unsqueeze(-1)

    @property
    def variance(self) -> Any:
        variance = self.distribution.variance
        return variance if self.is_multitask else variance.unsqueeze(-1)

    @property
    def base_sample_shape(self) -> Any:
        distribution = self.distribution
        return distribution.batch_shape + getattr(
            distribution, "base_sample_shape", distribution.event_shape
        )

    def extended_shape(self, sample_shape: Any = ()) -> Any:
        import torch

        return torch.Size(sample_shape) + self.mean.shape

    def rsample(self, sample_shape: Any = None, base_samples: Any = None) -> Any:
        import torch

        shape = torch.Size([1]) if sample_shape is None else torch.Size(sample_shape)
        distribution = self.distribution
        if base_samples is not None:
            if base_samples.shape[: len(shape)] != shape:
                raise RuntimeError(
                    "`sample_shape` disagrees with shape of `base_samples`. "
                    f"Got sample_shape={shape} and base_samples.shape={base_samples.shape}."
                )
            base_samples = base_samples.expand(shape + self.base_sample_shape)
            if self.is_multitask and not distribution._interleaved:
                # Public normals are point-major; the non-interleaved root is
                # output-major. GPyTorch then views the normals in event shape.
                base_samples = base_samples.transpose(-1, -2).reshape(base_samples.shape)
        with ExitStack() as context:
            if self.is_gpytorch:
                from linear_operator import settings

                if settings._fast_covar_root_decomposition.is_default():
                    context.enter_context(settings._fast_covar_root_decomposition(False))
                kwargs = {} if base_samples is None else {"base_samples": base_samples}
                samples = distribution.rsample(sample_shape=shape, **kwargs)
            elif base_samples is None:
                samples = distribution.rsample(sample_shape=shape)
            else:
                samples = distribution.loc + torch.matmul(
                    distribution.scale_tril, base_samples.unsqueeze(-1)
                ).squeeze(-1)
        return samples if self.is_multitask else samples.unsqueeze(-1)

    def marginal(self, value: Any, density: bool) -> Any:
        import torch

        value = torch.as_tensor(value, dtype=self.mean.dtype, device=self.mean.device)
        if value.numel() > 1:
            return torch.stack([self.marginal(v, density) for v in value])
        normal = torch.distributions.Normal(self.mean, self.variance.sqrt())
        return normal.log_prob(value).exp() if density else normal.icdf(value)


def scalarize_distribution(posterior: Any, weights: Any, offset: float) -> tuple[Any, Any]:
    """Affine moments of a supplied Gaussian law, retaining distribution graphs."""
    import torch

    from botorch_rs.posteriors.gpytorch import _validate_scalarize_inputs

    _validate_scalarize_inputs(weights, posterior._m)
    mean = posterior.mean
    weights = torch.as_tensor(weights, dtype=mean.dtype, device=mean.device)
    covariance = posterior.distribution.covariance_matrix
    q, m = mean.shape[-2:]
    transformed_mean = mean.matmul(weights) + offset
    if posterior._is_mt and posterior.distribution._interleaved:
        covariance = covariance.reshape(*mean.shape[:-2], q, m, q, m)
        transformed_covariance = torch.einsum("...itju,t,u->...ij", covariance, weights, weights)
    else:
        covariance = covariance.reshape(*mean.shape[:-2], m, q, m, q)
        transformed_covariance = torch.einsum("...tiuj,t,u->...ij", covariance, weights, weights)
    return transformed_mean, transformed_covariance


def mixture_quantile(posterior: Any, value: Any) -> Any:
    """Bisection of a supplied Gaussian mixture, matching its component layout."""
    import torch

    mean, variance = posterior.mean, posterior.variance
    value = torch.as_tensor(value, dtype=mean.dtype, device=mean.device)
    if value.numel() > 1:
        return torch.stack([mixture_quantile(posterior, v) for v in value])
    if not bool((value > 0) & (value < 1)):
        raise ValueError("value is expected to be in the range (0, 1).")
    normal = torch.distributions.Normal(mean, variance.sqrt())
    bounds = normal.icdf(value)
    if mean.shape[-3] == 1:
        return bounds.squeeze(-3)
    padding = 1e-4 if mean.dtype == torch.float32 else 1e-6
    low, high = bounds.amin(-3) - padding, bounds.amax(-3) + padding
    center = (low + high) / 2
    for _ in range(32):
        cdf = normal.cdf(center.unsqueeze(-3)).mean(-3)
        high = torch.where(cdf > value, center, high)
        low = torch.where(cdf > value, low, center)
        center = (low + high) / 2
        if bool((normal.cdf(center.unsqueeze(-3)).mean(-3) - value).abs().amax() <= 1e-6):
            break
    return center

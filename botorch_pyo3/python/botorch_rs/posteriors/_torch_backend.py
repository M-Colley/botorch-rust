"""Gaussian laws whose moments and samples retain the complete Torch graph."""

from __future__ import annotations

from typing import Any

import torch

from botorch_rs._arrays import kind_of
from botorch_rs.posteriors.gpytorch import GPyTorchPosterior


def independent_covariance(blocks: Any) -> Any:
    """Output-major covariance from independent ``... x m x q x q`` blocks."""
    m, q = blocks.shape[-3:-1]
    identity = torch.eye(m, dtype=blocks.dtype, device=blocks.device)
    return torch.einsum("...tij,tu->...tiuj", blocks, identity).reshape(
        *blocks.shape[:-3], m * q, m * q
    )


class TorchGPyTorchPosterior(GPyTorchPosterior):
    """Dense Gaussian posterior with device-resident, differentiable moments.

    Covariance uses output-major order, as the independent-output native path
    does. Public means and base samples use ``... x q x m`` order.
    """

    _tensor_backend = "torch"

    def __init__(self, mean: Any, covariance: Any) -> None:
        self._mean_tensor = mean
        self._covariance_tensor = covariance
        self._batch = torch.Size(mean.shape[:-2])
        self._q, self._m = mean.shape[-2:]
        if covariance.shape[-2:] != (self._q * self._m,) * 2:
            raise ValueError("Covariance does not match the posterior event shape.")
        self._kind = kind_of(mean)
        self._rs = None
        self._interleaved_mt = False
        self._base = False
        self._transform = None

    @property
    def _is_mt(self) -> bool:
        return self._m > 1

    @property
    def mean(self) -> Any:
        return self._mean_tensor

    @property
    def variance(self) -> Any:
        diagonal = self._covariance_tensor.diagonal(dim1=-2, dim2=-1)
        minimum = 1e-6 if self.dtype == torch.float32 else 1e-10
        return diagonal.reshape(*self._batch, self._m, self._q).transpose(-1, -2).clamp_min(minimum)

    @property
    def distribution(self) -> Any:
        return TorchMultivariateNormal(self)

    @property
    def mvn(self) -> Any:
        return self.distribution

    @property
    def base_sample_shape(self) -> Any:
        return self._batch + ((self._q, self._m) if self._is_mt else (self._q,))

    def rsample(self, sample_shape: Any = None) -> Any:
        shape = torch.Size([1] if sample_shape is None else sample_shape)
        normals = torch.randn(shape + self.base_sample_shape, dtype=self.dtype, device=self.device)
        return self.rsample_from_base_samples(shape, normals)

    def rsample_from_base_samples(self, sample_shape: Any, base_samples: Any) -> Any:
        shape = torch.Size(sample_shape)
        if not isinstance(base_samples, torch.Tensor):
            base_samples = torch.as_tensor(base_samples, dtype=self.dtype, device=self.device)
        if base_samples.shape[: len(shape)] != shape:
            raise RuntimeError("sample_shape disagrees with base_samples.shape.")
        normals = base_samples.expand(shape + self.base_sample_shape)
        if self._is_mt:
            normals = normals.transpose(-1, -2).reshape(*shape, *self._batch, self._m * self._q)
        root = torch.linalg.cholesky(self._covariance_tensor)
        centered = (root @ normals.unsqueeze(-1)).squeeze(-1)
        mean = self.mean.transpose(-1, -2).reshape(*self._batch, self._m * self._q)
        return (mean + centered).reshape(*shape, *self._batch, self._m, self._q).transpose(-1, -2)

    def quantile(self, value: Any) -> Any:
        value = torch.as_tensor(value, dtype=self.dtype, device=self.device)
        normal = torch.distributions.Normal(self.mean, self.variance.sqrt())
        return (
            normal.icdf(value)
            if value.numel() == 1
            else torch.stack([normal.icdf(v) for v in value])
        )

    def density(self, value: Any) -> Any:
        value = torch.as_tensor(value, dtype=self.dtype, device=self.device)
        normal = torch.distributions.Normal(self.mean, self.variance.sqrt())
        return (
            normal.log_prob(value).exp()
            if value.numel() == 1
            else torch.stack([normal.log_prob(v).exp() for v in value])
        )

    def scalarize(self, weights: Any, offset: Any = 0.0) -> TorchGPyTorchPosterior:
        weights = torch.as_tensor(weights, dtype=self.dtype, device=self.device)
        if weights.ndim != 1 or weights.shape[0] != self._m:
            raise RuntimeError("Output shape not equal to that of weights.")
        cov = self._covariance_tensor.reshape(*self._batch, self._m, self._q, self._m, self._q)
        covariance = torch.einsum("...tiuj,t,u->...ij", cov, weights, weights)
        mean = (self.mean @ weights + offset).unsqueeze(-1)
        return TorchGPyTorchPosterior(mean, covariance)

    def __repr__(self) -> str:
        return f"TorchGPyTorchPosterior(batch_shape={tuple(self._batch)}, q={self._q}, m={self._m})"


class TorchMultivariateNormal:
    """Distribution view of a Tensor-backend posterior."""

    _interleaved = False
    has_rsample = True

    def __init__(self, posterior: TorchGPyTorchPosterior) -> None:
        self._p = posterior

    @property
    def mean(self) -> Any:
        return self._p.mean if self._p._is_mt else self._p.mean.squeeze(-1)

    @property
    def loc(self) -> Any:
        return self._p.mean.transpose(-1, -2).reshape(*self.batch_shape, -1)

    @property
    def variance(self) -> Any:
        return self._p.variance if self._p._is_mt else self._p.variance.squeeze(-1)

    @property
    def stddev(self) -> Any:
        return self.variance.sqrt()

    @property
    def covariance_matrix(self) -> Any:
        return self._p._covariance_tensor

    @property
    def scale_tril(self) -> Any:
        return torch.linalg.cholesky(self.covariance_matrix)

    @property
    def batch_shape(self) -> Any:
        return self._p._batch

    @property
    def event_shape(self) -> Any:
        return torch.Size((self._p._q, self._p._m) if self._p._is_mt else (self._p._q,))

    @property
    def base_sample_shape(self) -> Any:
        return self.event_shape

    @property
    def num_tasks(self) -> int:
        return self._p._m

    def rsample(self, sample_shape: Any = (), base_samples: Any = None) -> Any:
        samples = (
            self._p.rsample(sample_shape)
            if base_samples is None
            else self._p.rsample_from_base_samples(sample_shape, base_samples)
        )
        return samples if self._p._is_mt else samples.squeeze(-1)

    def sample(self, sample_shape: Any = ()) -> Any:
        with torch.no_grad():
            return self.rsample(sample_shape)

    def log_prob(self, value: Any) -> Any:
        if self._p._is_mt:
            value = value.transpose(-1, -2).reshape(*value.shape[:-2], -1)
        return torch.distributions.MultivariateNormal(
            self.loc, covariance_matrix=self.covariance_matrix
        ).log_prob(value)

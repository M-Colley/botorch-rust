"""Device-resident finite-ensemble moments and selected-member samples."""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import Tensor

from botorch_rs._arrays import kind_of
from botorch_rs.posteriors.ensemble import EnsemblePosterior, _shape_tuple


class TorchEnsemblePosterior(EnsemblePosterior):
    _tensor_backend = "torch"
    _weights: Any

    def __init__(self, values: Tensor, weights: Any = None) -> None:
        if values.ndim < 3:
            raise ValueError("Values has to be at least three-dimensional.")
        self.values = values
        self._shape = tuple(values.shape)
        self._kind = kind_of(values)
        self._weights = (
            None
            if weights is None
            else torch.as_tensor(weights, dtype=values.dtype, device=values.device)
        )
        self._mixture_dims = list(range(values.ndim - 2))

    @property
    def batch_shape(self) -> Any:
        return self.values.shape[:-3]

    def _extended_shape(self, sample_shape: Any = ()) -> Any:
        return torch.Size(_shape_tuple(sample_shape)) + self.batch_shape + self.values.shape[-2:]

    @property
    def weights(self) -> Tensor:
        # Build a fresh normalization graph on every call. Reusing a cached
        # non-leaf normalization consumes its graph after the first backward.
        if self._weights is None:
            return self.values.new_ones((self.ensemble_size,)) / self.ensemble_size
        return self._weights / self._weights.sum(dim=-1, keepdim=True)

    @property
    def mixture_weights(self) -> Tensor:
        if self._weights is None:
            return self.values.new_ones(self.values.shape[:-2]) / self.mixture_size
        expanded = self._weights.expand(self.values.shape[:-2])
        return expanded / expanded.sum(dim=self.mixture_dims, keepdim=True)

    @property
    def mean(self) -> Tensor:
        return (self.values * self.weights[..., None, None]).sum(dim=-3)

    @property
    def variance(self) -> Tensor:
        if self.ensemble_size == 1:
            return self.values.squeeze(-3) * 0
        weights = self.weights
        numerator = (
            weights[..., None, None] * (self.values - self.mean.unsqueeze(-3)).square()
        ).sum(dim=-3)
        # Each model batch has its own reliability-weight denominator.
        return numerator / (1 - weights.square().sum(dim=-1))[..., None, None]

    @property
    def mixture_mean(self) -> Tensor:
        return (self.values * self.mixture_weights[..., None, None]).sum(dim=self.mixture_dims)

    @property
    def mixture_variance(self) -> Tensor:
        weights = self.mixture_weights
        numerator = (weights[..., None, None] * (self.values - self.mixture_mean).square()).sum(
            dim=self.mixture_dims
        )
        return numerator / (1 - weights.square().sum())

    def rsample(self, sample_shape: Any = None) -> Tensor:
        shape = torch.Size(_shape_tuple(sample_shape) or (1,))
        if self.values.numel() == 0:
            return self.values.new_empty(self._extended_shape(shape))
        indices = draw_indices(self, shape)
        return self.rsample_from_base_samples(shape, indices)

    def rsample_from_base_samples(self, sample_shape: Any, base_samples: Any) -> Tensor:
        shape = torch.Size(_shape_tuple(sample_shape))
        indices = torch.as_tensor(base_samples, device=self.device)
        if tuple(indices.shape) != tuple(shape + self.batch_shape):
            raise ValueError(
                f"Sample_shape={shape + self.batch_shape} does not match "
                f"the leading dimensions of base_samples.shape={indices.shape}."
            )
        if self.batch_shape:
            batch = math.prod(self.batch_shape)
            values = self.values.reshape(batch, *self.values.shape[-3:])
            return values[
                torch.arange(batch, device=self.device), indices.reshape(*shape, batch)
            ].reshape(self._extended_shape(shape))
        return self.values[indices]


def draw_indices(
    posterior: TorchEnsemblePosterior, sample_shape: Any, generator: Any = None
) -> Tensor:
    """The Categorical/Multinomial sample law with an optional local RNG."""
    shape = torch.Size(sample_shape)
    count = math.prod(shape)
    weights = posterior.mixture_weights
    if count == 0 or math.prod(posterior.batch_shape) == 0:
        return torch.empty(shape + posterior.batch_shape, dtype=torch.long, device=posterior.device)
    # Categorical.sample uses precisely this row-wise multinomial layout.
    indices = torch.multinomial(
        weights.reshape(-1, posterior.ensemble_size), count, True, generator=generator
    )
    return indices.mT.reshape(shape + posterior.batch_shape)

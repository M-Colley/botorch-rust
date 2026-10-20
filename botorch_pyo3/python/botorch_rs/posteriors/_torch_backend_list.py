"""Batch-axis permutation for non-Gaussian device posterior components."""

from __future__ import annotations

from typing import Any

import torch

from botorch_rs.posteriors.gpytorch import Posterior


class MoveMCMCAxisPosterior(Posterior):
    """Move a leading component batch axis behind the ordinary batch axes.

    Evaluating an ordinary model with a leading MCMC query batch keeps captured
    outcome-transform coefficients in their original batch layout. This view
    permutes its moments, normals and samples without reconstructing transform
    closures or converting any values to host buffers.
    """

    _tensor_backend = "torch"

    def __init__(self, posterior: Any) -> None:
        self._source = posterior
        shape = torch.Size(posterior._extended_shape(()))
        self._batch = shape[1:-2] + shape[:1]
        self._q, self._m = shape[-2:]
        source_batch = len(shape) - 2
        self._base_event = tuple(posterior.base_sample_shape[source_batch:])
        self._kind = posterior._kind

    @property
    def batch_shape(self) -> Any:
        return self._batch

    @property
    def base_sample_shape(self) -> Any:
        return self._batch + self._base_event

    @property
    def batch_range(self) -> tuple[int, int]:
        return (0, -len(self._base_event))

    @property
    def mean(self) -> Any:
        return self._source.mean.movedim(0, -3)

    @property
    def variance(self) -> Any:
        return self._source.variance.movedim(0, -3)

    @property
    def device(self) -> Any:
        return self._source.device

    @property
    def dtype(self) -> Any:
        return self._source.dtype

    def _extended_shape(self, sample_shape: Any = ()) -> Any:
        return torch.Size(sample_shape) + self._batch + (self._q, self._m)

    def rsample_from_base_samples(self, sample_shape: Any, base_samples: Any) -> Any:
        shape = torch.Size(sample_shape)
        normals = base_samples.expand(shape + self.base_sample_shape)
        axis = len(shape) + len(self._batch) - 1
        normals = normals.movedim(axis, len(shape))
        values = self._source.rsample_from_base_samples(shape, normals)
        return values.movedim(len(shape), -3)

    def rsample(self, sample_shape: Any = None) -> Any:
        shape = torch.Size([1] if sample_shape is None else sample_shape)
        normals = torch.randn(shape + self.base_sample_shape, dtype=self.dtype, device=self.device)
        return self.rsample_from_base_samples(shape, normals)

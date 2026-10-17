"""Ensemble posteriors (``botorch.posteriors.ensemble``), used in conjunction
with ensemble models."""

from __future__ import annotations

import importlib
import math
import random
from typing import Any

from botorch_rs._arrays import LIST, from_flat, kind_of, to_flat
from botorch_rs.posteriors.gpytorch import Posterior


def _shape_tuple(sample_shape: Any) -> tuple[int, ...]:
    if sample_shape is None:
        return ()
    if isinstance(sample_shape, int):
        return (sample_shape,)
    return tuple(int(s) for s in sample_shape)


class EnsemblePosterior(Posterior):
    r"""Ensemble posterior, that should be used for ensemble models that compute
    eagerly a finite number of samples per X value as for example a deep ensemble
    or a random forest.

    The quantities are computed from ``values`` (``(b) x s x q x m``) with
    the array type of ``values``.
    """

    def __init__(self, values: Any, weights: Any = None) -> None:
        r"""
        Args:
            values: Values of the samples produced by this posterior as
                a ``(b) x s x q x m`` tensor where ``m`` is the output size of the
                model and ``s`` is the ensemble size.
            weights: Optional weights for the ensemble members as a tensor of shape
                ``(s,)``. If None, uses uniform weights.
        """
        data, shape = to_flat(values, "values")
        if len(shape) < 3:
            raise ValueError("Values has to be at least three-dimensional.")
        self.values = values
        self._data = list(data)
        self._shape = tuple(shape)
        self._kind = kind_of(values)
        self._weights = None if weights is None else [float(v) for v in to_flat(weights)[0]]
        self._mixture_dims = list(range(len(self._shape) - 2))

    # -- shapes ----------------------------------------------------------

    @property
    def ensemble_size(self) -> int:
        r"""The size of the ensemble"""
        return self._shape[-3]

    @property
    def mixture_size(self) -> int:
        r"""The total number of elements in the mixture dimensions"""
        return math.prod(self._shape[:-2])

    @property
    def batch_shape(self) -> tuple[int, ...]:
        return self._shape[:-3]

    @property
    def mixture_dims(self) -> list[int]:
        r"""The mixture dimensions of the posterior (all but the last two)."""
        return self._mixture_dims

    @property
    def dtype(self) -> Any:
        return getattr(self.values, "dtype", None)

    @property
    def device(self) -> Any:
        return getattr(self.values, "device", None)

    def _extended_shape(self, sample_shape: Any = ()) -> tuple[int, ...]:
        return (*_shape_tuple(sample_shape), *self._shape[:-3], *self._shape[-2:])

    # -- weights ---------------------------------------------------------

    def _normalized_weights(self) -> list[float]:
        s = self.ensemble_size
        if self._weights is None:
            return [1.0 / s] * s
        total = sum(self._weights)
        return [w / total for w in self._weights]

    def _normalized_mixture_weights(self) -> list[float]:
        n = self.mixture_size
        s = self.ensemble_size
        if self._weights is None:
            return [1.0 / n] * n
        expanded = [self._weights[i % s] for i in range(n)]
        total = sum(expanded)
        return [w / total for w in expanded]

    @property
    def weights(self) -> Any:
        r"""The weights of the individual models in the ensemble.
        uniformly weighted by default."""
        return from_flat(self._normalized_weights(), (self.ensemble_size,), self._kind)

    @property
    def mixture_weights(self) -> Any:
        r"""The weights of the individual models in the ensemble, normalized
        over the ensemble and batch dimensions of the model."""
        return from_flat(self._normalized_mixture_weights(), self._shape[:-2], self._kind)

    # -- moments ---------------------------------------------------------

    def _blocks(self) -> tuple[int, int, int]:
        """``(batch size, ensemble size, q * m)``."""
        qm = self._shape[-2] * self._shape[-1]
        return math.prod(self._shape[:-3]), self._shape[-3], qm

    def _mean(self) -> list[float]:
        b, s, qm = self._blocks()
        w = self._normalized_weights()
        return [
            sum(self._data[(i * s + k) * qm + j] * w[k] for k in range(s))
            for i in range(b)
            for j in range(qm)
        ]

    @property
    def mean(self) -> Any:
        r"""The mean of the posterior as a ``(b) x n x m``-dim Tensor."""
        return from_flat(self._mean(), (*self._shape[:-3], *self._shape[-2:]), self._kind)

    @property
    def variance(self) -> Any:
        r"""The variance of the posterior as a ``(b) x n x m``-dim Tensor:
        the unbiased weighted sample variance
        ``sum_i w_i (x_i - mu)^2 / (1 - sum_i w_i^2)`` across the ensemble."""
        b, s, qm = self._blocks()
        out_shape = (*self._shape[:-3], *self._shape[-2:])
        if s == 1:
            return from_flat([0.0] * (b * qm), out_shape, self._kind)
        w = self._normalized_weights()
        mean = self._mean()
        denom = 1 - sum(x * x for x in w)
        out = [
            sum(w[k] * (self._data[(i * s + k) * qm + j] - mean[i * qm + j]) ** 2 for k in range(s))
            / denom
            for i in range(b)
            for j in range(qm)
        ]
        return from_flat(out, out_shape, self._kind)

    def _mixture_mean(self) -> list[float]:
        n = self.mixture_size
        qm = self._shape[-2] * self._shape[-1]
        w = self._normalized_mixture_weights()
        return [sum(self._data[i * qm + j] * w[i] for i in range(n)) for j in range(qm)]

    @property
    def mixture_mean(self) -> Any:
        r"""The mixture mean of the posterior as a ``n x m``-dim Tensor."""
        return from_flat(self._mixture_mean(), self._shape[-2:], self._kind)

    @property
    def mixture_variance(self) -> Any:
        r"""The mixture variance of the posterior as a ``n x m``-dim Tensor
        (the unbiased weighted sample variance over all mixture dimensions)."""
        n = self.mixture_size
        qm = self._shape[-2] * self._shape[-1]
        w = self._normalized_mixture_weights()
        mean = self._mixture_mean()
        denom = 1 - sum(x * x for x in w)
        out = [
            sum(w[i] * (self._data[i * qm + j] - mean[j]) ** 2 for i in range(n)) / denom
            for j in range(qm)
        ]
        return from_flat(out, self._shape[-2:], self._kind)

    # -- sampling ----------------------------------------------------------

    def rsample(self, sample_shape: Any = None) -> Any:
        r"""Sample from the posterior: the values of ensemble members drawn
        with the mixture weights (from torch's generator if torch is
        installed, as BoTorch's ``Multinomial``; else Python's ``random``)."""
        shape = _shape_tuple(sample_shape) or (1,)
        if not self._data:
            return from_flat([], self._extended_shape(shape), self._kind)
        n = math.prod(shape)
        probs = self._normalized_mixture_weights()
        batch = self.batch_shape
        nb = math.prod(batch)
        s = self.ensemble_size
        try:
            torch = importlib.import_module("torch")
        except ImportError:
            idx = [
                random.choices(range(s), weights=probs[b * s : (b + 1) * s])[0]
                for _ in range(n)
                for b in range(nb)
            ]
        else:
            p = torch.tensor(probs, dtype=torch.float64).reshape(*self._shape[:-2])
            sampled = torch.distributions.Multinomial(probs=p).sample(torch.Size(shape))
            idx = [int(v) for v in sampled.argmax(dim=-1).reshape(-1).tolist()]
        base = from_flat([float(i) for i in idx], (*shape, *batch), LIST)
        return self.rsample_from_base_samples(shape, base)

    def rsample_from_base_samples(self, sample_shape: Any, base_samples: Any) -> Any:
        r"""Sample from the posterior using base samples: the values of the
        ensemble members with the indices ``base_samples`` (shape
        ``sample_shape x batch_shape``, e.g. from an ``IndexSampler``)."""
        shape = _shape_tuple(sample_shape)
        idx, ishape = to_flat(base_samples, "base_samples")
        expected = (*shape, *self.batch_shape)
        if tuple(ishape) != expected:
            raise ValueError(
                f"Sample_shape={expected} does not match "
                f"the leading dimensions of base_samples.shape={tuple(ishape)}."
            )
        b, s, qm = self._blocks()
        n = math.prod(shape)
        out: list[float] = []
        for t in range(n):
            for i in range(b):
                k = int(idx[t * b + i])
                out.extend(self._data[(i * s + k) * qm : (i * s + k + 1) * qm])
        return from_flat(out, self._extended_shape(shape), self._kind)

    def __repr__(self) -> str:
        return f"EnsemblePosterior(values_shape={self._shape})"


__all__ = ["EnsemblePosterior"]

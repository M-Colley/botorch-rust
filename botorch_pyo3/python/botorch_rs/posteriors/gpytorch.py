"""Gaussian posteriors (``botorch.posteriors.gpytorch``).

Posteriors are evaluated lazily: ``mean`` and ``variance`` only need the
marginal distribution of each point (linear in the number of points), while
``distribution.covariance_matrix`` and sampling compute the joint posterior of
every ``q``-batch.
"""

from __future__ import annotations

import math
import random
from typing import Any

from botorch_rs._arrays import ArrayKind, Flat, from_flat, to_flat
from botorch_rs.exceptions.errors import BotorchTensorDimensionError


def _as_shape(sample_shape: Any) -> tuple[int, ...]:
    if sample_shape is None:
        return (1,)
    if isinstance(sample_shape, int):
        return (sample_shape,)
    return tuple(int(s) for s in sample_shape)


def _standard_normal(n: int, kind: ArrayKind) -> list[float]:
    """``n`` i.i.d. standard normal draws from the random number generator of
    the caller's array library (so that ``torch.manual_seed`` and
    ``numpy.random.seed`` apply)."""
    if kind.lib == "torch":
        import torch

        return torch.randn(n, dtype=torch.float64).tolist()
    if kind.lib == "numpy":
        import numpy as np

        return np.random.standard_normal(n).tolist()
    return [random.gauss(0.0, 1.0) for _ in range(n)]


class Posterior:
    """Base class of the posteriors."""


class GPyTorchPosterior(Posterior):
    """The Gaussian posterior of a single-output model at ``X``
    (``batch_shape x q x d``), in the original outcome space."""

    def __init__(self, model: Any, X: Flat, kind: ArrayKind, observation_noise: bool) -> None:
        self._model = model
        self._rs = model._rs
        self._x = X
        self._kind = kind
        self._observation_noise = observation_noise
        shape = X[1]
        self._batch = tuple(shape[:-2])
        self._q = shape[-2]
        self._marginals: tuple[list[float], list[float]] | None = None
        self._joint: tuple[Flat, Flat] | None = None

    # -- lazily computed quantities -------------------------------------

    def _get_marginals(self) -> tuple[list[float], list[float]]:
        if self._marginals is None:
            (mean, _), (var, _) = self._rs.posterior_marginals(self._x, self._observation_noise)
            self._marginals = (mean, var)
        return self._marginals

    def _get_joint(self) -> tuple[Flat, Flat]:
        if self._joint is None:
            self._joint = self._rs.posterior_joint(self._x, self._observation_noise)
        return self._joint

    def _out(self, data: list[float], shape: tuple[int, ...]) -> Any:
        return from_flat(data, shape, self._kind)

    # -- BoTorch's Posterior API -----------------------------------------

    @property
    def mean(self) -> Any:
        """The posterior mean, shape ``batch_shape x q x 1``."""
        return self._out(self._get_marginals()[0], (*self._batch, self._q, 1))

    @property
    def variance(self) -> Any:
        """The posterior variance (clamped at 1e-10), shape ``batch_shape x q x 1``."""
        return self._out(self._get_marginals()[1], (*self._batch, self._q, 1))

    @property
    def distribution(self) -> MultivariateNormal:
        """The joint normal distribution of each ``q``-batch."""
        return MultivariateNormal(self)

    @property
    def mvn(self) -> MultivariateNormal:
        return self.distribution

    @property
    def batch_shape(self) -> tuple[int, ...]:
        return self._batch

    @property
    def base_sample_shape(self) -> tuple[int, ...]:
        """The shape of the base samples of one posterior sample."""
        return (*self._batch, self._q)

    def _extended_shape(self, sample_shape: Any = ()) -> tuple[int, ...]:
        return (*_as_shape(sample_shape or ()), *self._batch, self._q, 1)

    def rsample(self, sample_shape: Any = None) -> Any:
        """Posterior samples, shape ``sample_shape x batch_shape x q x 1``.

        The base samples are drawn with the caller's array library (torch,
        NumPy or Python's ``random``)."""
        sample_shape = _as_shape(sample_shape)
        n = math.prod(sample_shape)
        nb = math.prod(self._batch)
        z = _standard_normal(n * nb * self._q, self._kind)
        out = self._rs.posterior_samples(self._x, (z, [n, nb, self._q]), self._observation_noise)
        return self._out(out[0], (*sample_shape, *self._batch, self._q, 1))

    def rsample_from_base_samples(self, sample_shape: Any, base_samples: Any) -> Any:
        """Posterior samples ``mean + L z`` from explicit base samples ``z`` of
        shape ``sample_shape x base_sample_shape`` (batch dimensions of size 1
        are broadcast; a trailing output dimension of size 1 is allowed)."""
        sample_shape = _as_shape(sample_shape)
        data, zshape = to_flat(base_samples, "base_samples")
        k = len(sample_shape)
        if tuple(zshape[:k]) != sample_shape:
            raise RuntimeError(
                "`sample_shape` disagrees with shape of `base_samples`. "
                f"Got sample_shape={sample_shape} and base_samples.shape={zshape}."
            )
        rest = tuple(zshape[k:])
        if len(rest) == len(self._batch) + 2 and rest[-1] == 1:
            rest = rest[:-1]
        n = math.prod(sample_shape)
        if not rest or rest[-1] != self._q:
            raise BotorchTensorDimensionError(
                f"base samples of shape {zshape} do not match q = {self._q} points."
            )
        if math.prod(rest[:-1]) == 1:
            z: Flat = (data, (n, self._q))
        elif rest[:-1] == self._batch:
            z = (data, (n, math.prod(self._batch), self._q))
        else:
            raise BotorchTensorDimensionError(
                f"base samples of shape {zshape} do not match the batch shape {self._batch}."
            )
        out = self._rs.posterior_samples(self._x, z, self._observation_noise)
        return self._out(out[0], (*sample_shape, *self._batch, self._q, 1))

    def __repr__(self) -> str:
        return f"GPyTorchPosterior(batch_shape={self._batch}, q={self._q})"


class MultivariateNormal:
    """The joint normal distribution of a posterior's ``q``-batches
    (``posterior.distribution``; no output dimension)."""

    def __init__(self, posterior: GPyTorchPosterior) -> None:
        self._p = posterior

    @property
    def mean(self) -> Any:
        """Shape ``batch_shape x q``."""
        p = self._p
        return p._out(p._get_marginals()[0], (*p._batch, p._q))

    @property
    def variance(self) -> Any:
        """Marginal variances (clamped at 1e-10), shape ``batch_shape x q``."""
        p = self._p
        return p._out(p._get_marginals()[1], (*p._batch, p._q))

    @property
    def stddev(self) -> Any:
        p = self._p
        return p._out([math.sqrt(v) for v in p._get_marginals()[1]], (*p._batch, p._q))

    @property
    def covariance_matrix(self) -> Any:
        """Joint covariances, shape ``batch_shape x q x q``."""
        p = self._p
        _, cov = p._get_joint()
        return p._out(cov[0], tuple(cov[1]))

    @property
    def batch_shape(self) -> tuple[int, ...]:
        return self._p._batch

    @property
    def event_shape(self) -> tuple[int, ...]:
        return (self._p._q,)

    def rsample(self, sample_shape: Any = None, base_samples: Any = None) -> Any:
        """Samples of shape ``sample_shape x batch_shape x q``."""
        p = self._p
        if base_samples is None:
            out = p.rsample(sample_shape)
        else:
            out = p.rsample_from_base_samples(_as_shape(sample_shape), base_samples)
        data, shape = to_flat(out, "samples")
        return p._out(data, shape[:-1])

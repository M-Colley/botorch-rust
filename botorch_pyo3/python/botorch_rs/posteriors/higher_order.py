"""The posterior of a ``HigherOrderGP`` (``botorch.posteriors.higher_order``)."""

from __future__ import annotations

import math
import random
from typing import Any

from botorch_rs._arrays import ArrayKind, from_flat, kind_of, to_flat
from botorch_rs.exceptions.errors import BotorchTensorDimensionError
from botorch_rs.posteriors.gpytorch import Posterior


def _randn(n: int, kind: ArrayKind) -> list[float]:
    if kind.lib == "torch":
        import torch

        return [float(v) for v in torch.randn(n, dtype=torch.float64).tolist()]
    if kind.lib == "numpy":
        import numpy as np

        return [float(v) for v in np.random.standard_normal(n)]
    return [random.gauss(0.0, 1.0) for _ in range(n)]


class _DiagonalNormal:
    """``posterior.distribution``: BoTorch's ``MultivariateNormal`` of the
    marginals (mean and variance of shape ``batch x q x target_shape``)."""

    def __init__(self, posterior: HigherOrderGPPosterior) -> None:
        self._p = posterior

    @property
    def mean(self) -> Any:
        return self._p.mean

    @property
    def variance(self) -> Any:
        return self._p.variance

    @property
    def stddev(self) -> Any:
        data, shape = to_flat(self._p.variance, "variance")
        return from_flat([math.sqrt(v) for v in data], tuple(shape), self._p._kind)


class HigherOrderGPPosterior(Posterior):
    r"""
    Posterior class for a Higher order Gaussian process model [Zhe2019hogp]_. Extends
    the standard GPyTorch posterior class by overwriting the rsample method.
    The posterior variance is handled internally by the HigherOrderGP model.
    HOGP is a tensorized GP model so the posterior covariance grows to be extremely
    large, but is highly structured, which means that we can exploit Kronecker
    identities to sample from the posterior using Matheron's rule as described in
    [Doucet2010sampl]_.

    In botorch_rs the posterior is evaluated in Rust (``source`` is the
    extension model); ``X`` are the raw-space points (``batch x q x d``).
    """

    def __init__(
        self,
        distribution: Any = None,
        joint_covariance_matrix: Any = None,
        train_train_covar: Any = None,
        test_train_covar: Any = None,
        train_targets: Any = None,
        output_shape: Any = None,
        num_outputs: int = 1,
        *,
        source: Any = None,
        X: Any = None,
        kind: Any = None,
        target_shape: Any = None,
        num_train: Any = None,
        observation_noise: bool = False,
        original: bool = True,
        noise_scale: float = 1.0,
    ) -> None:
        if distribution is not None:
            from botorch_rs.posteriors._structured import SuppliedMatheron

            if any(
                value is None
                for value in (
                    joint_covariance_matrix,
                    train_train_covar,
                    test_train_covar,
                    train_targets,
                    output_shape,
                )
            ):
                raise TypeError(
                    "Supplied HOGP posteriors require all covariance operators, targets, and output_shape."
                )
            self._supplied = SuppliedMatheron(
                distribution,
                joint_covariance_matrix,
                train_train_covar,
                test_train_covar,
                train_targets,
                output_shape,
            )
            self._kind = kind_of(distribution.mean)
            self._is_mt = True
            self.num_outputs = num_outputs
            return
        if source is None or X is None or kind is None or target_shape is None or num_train is None:
            raise TypeError(
                "Provide a distribution and operators, or a native HOGP source and query."
            )
        self._rs = source.copy()
        self._x = X
        self._kind = kind
        self._target_shape = tuple(target_shape)
        self._n = int(num_train)
        self._noise = bool(observation_noise)
        self._original = bool(original)
        self._noise_scale = float(noise_scale)
        self.num_outputs = num_outputs
        shape = X[1]
        self._batch = tuple(shape[:-2])
        self._q = shape[-2]
        self._d = shape[-1]
        self._S = math.prod(self._target_shape)
        self._mv: tuple[list[float], list[float]] | None = None
        self._is_mt = True

    @property
    def output_shape(self) -> tuple[int, ...]:
        """``X.shape[:-1] + target_shape``."""
        if (adapter := self.__dict__.get("_supplied")) is not None:
            return adapter.output_shape
        return (*self._batch, self._q, *self._target_shape)

    def _batches(self) -> list[tuple[list[float], tuple[int, int]]]:
        data = self._x[0]
        size = self._q * self._d
        nb = math.prod(self._batch)
        return [(list(data[b * size : (b + 1) * size]), (self._q, self._d)) for b in range(nb)]

    def _mean_variance(self) -> tuple[list[float], list[float]]:
        if self._mv is None:
            mean: list[float] = []
            var: list[float] = []
            for xb in self._batches():
                (m, _), (v, _) = self._rs.mean_variance(xb, self._noise, self._original)
                mean.extend(m)
                var.extend(v)
            self._mv = (mean, var)
        return self._mv

    @property
    def mean(self) -> Any:
        """The posterior mean, shape ``batch x q x target_shape``."""
        if (adapter := self.__dict__.get("_supplied")) is not None:
            return adapter.mean
        return from_flat(self._mean_variance()[0], self.output_shape, self._kind)

    @property
    def variance(self) -> Any:
        """The marginal posterior variances (BoTorch's
        ``make_posterior_variances``; with ``observation_noise``, plus the
        noise), shape ``batch x q x target_shape``."""
        if (adapter := self.__dict__.get("_supplied")) is not None:
            return adapter.variance
        return from_flat(self._mean_variance()[1], self.output_shape, self._kind)

    @property
    def distribution(self) -> _DiagonalNormal:
        if (adapter := self.__dict__.get("_supplied")) is not None:
            return adapter.distribution
        return _DiagonalNormal(self)

    @property
    def mvn(self) -> _DiagonalNormal:
        return self.distribution

    @property
    def dtype(self) -> Any:
        return self._kind.dtype

    @property
    def device(self) -> Any:
        return self._kind.device

    @property
    def batch_shape(self) -> tuple[int, ...]:
        if (adapter := self.__dict__.get("_supplied")) is not None:
            return adapter.batch_shape
        return self._batch

    @property
    def base_sample_shape(self) -> tuple[int, ...]:
        r"""The shape of a base sample used for constructing posterior samples:
        ``(n + q)·S`` joint prior draws and ``n·S`` training noise draws
        (BoTorch's ``n + 2 n_train`` layout; with observation noise, ``q·S``
        draws of the test noise follow, a botorch_rs extension)."""
        if (adapter := self.__dict__.get("_supplied")) is not None:
            return adapter.base_sample_shape
        total = (self._n + self._q) * self._S + self._n * self._S
        if self._noise:
            total += self._q * self._S
        return (*self._batch, total)

    @property
    def batch_range(self) -> tuple[int, int]:
        r"""The t-batch range (``(0, -1)``)."""
        return (0, -1)

    def _extended_shape(self, sample_shape: Any = ()) -> tuple[int, ...]:
        return (*tuple(sample_shape or ()), *self.output_shape)

    def rsample_from_base_samples(self, sample_shape: Any, base_samples: Any) -> Any:
        r"""Sample from the posterior (with gradients) using base samples, by
        Matheron's rule: base samples of shape ``sample_shape x
        base_sample_shape`` (batch dimensions of size 1 are broadcast).

        Returns:
            Samples of shape ``sample_shape x batch x q x target_shape``.
        """
        if (adapter := self.__dict__.get("_supplied")) is not None:
            return adapter.rsample(sample_shape, base_samples)
        sample_shape = tuple(int(s) for s in sample_shape)
        data, shape = to_flat(base_samples, "base_samples")
        k = len(sample_shape)
        if tuple(shape[:k]) != sample_shape:
            raise RuntimeError("sample_shape disagrees with shape of base_samples.")
        total = self.base_sample_shape[-1]
        if not shape or shape[-1] != total:
            raise BotorchTensorDimensionError(
                "The base samples are not compatible with base sample shape. "
                f"Received base samples of shape {tuple(shape)}, expected "
                f"{(*sample_shape, *self.base_sample_shape)}."
            )
        ns = math.prod(sample_shape)
        bshape = tuple(shape[k:-1])
        nb = math.prod(self._batch)
        if math.prod(bshape) not in (1, nb):
            raise BotorchTensorDimensionError(
                f"base samples of shape {tuple(shape)} do not match the batch shape {self._batch}."
            )
        per_batch = math.prod(bshape)
        joint = total - (self._q * self._S if self._noise else 0)
        out: list[float] = []
        results = []
        for b, xb in enumerate(self._batches()):
            rows = []
            for s in range(ns):
                off = (s * per_batch + (b if per_batch > 1 else 0)) * total
                rows.append(data[off : off + total])
            base = ([v for r in rows for v in r[:joint]], [ns, joint])
            samples, _ = self._rs.samples(xb, base, self._original)
            samples = list(samples)
            if self._noise:
                sd = math.sqrt(self._rs.noise[0]) * self._noise_scale
                qs = self._q * self._S
                for s in range(ns):
                    extra = rows[s][joint:]
                    for j in range(qs):
                        samples[s * qs + j] += sd * extra[j]
            results.append(samples)
        qs = self._q * self._S
        for s in range(ns):
            for b in range(nb):
                out.extend(results[b][s * qs : (s + 1) * qs])
        return from_flat(out, (*sample_shape, *self.output_shape), self._kind)

    def rsample(self, sample_shape: Any = None) -> Any:
        r"""Sample from the posterior (base samples from the caller's array
        library's generator)."""
        if (adapter := self.__dict__.get("_supplied")) is not None:
            return adapter.rsample(sample_shape)
        if sample_shape is None:
            sample_shape = (1,)
        sample_shape = tuple(int(s) for s in sample_shape)
        shape = (*sample_shape, *self.base_sample_shape)
        z = _randn(math.prod(shape), self._kind)
        return self.rsample_from_base_samples(sample_shape, from_flat(z, shape, self._kind))

    def __repr__(self) -> str:
        return f"HigherOrderGPPosterior(output_shape={self.output_shape})"

    def quantile(self, value: Any) -> Any:
        from botorch_rs.posteriors._structured import marginal

        return marginal(self, value, density=False)

    def density(self, value: Any) -> Any:
        from botorch_rs.posteriors._structured import marginal

        return marginal(self, value, density=True)

    def __getattr__(self, name: str) -> Any:
        adapter = self.__dict__.get("_supplied")
        if adapter is None:
            raise AttributeError(name)
        return getattr(adapter, name)


__all__ = ["HigherOrderGPPosterior"]

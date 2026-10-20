"""Structured posterior of the native latent Kronecker GP."""

from __future__ import annotations

import math
from typing import Any

from botorch_rs._arrays import broadcast_shapes, from_flat, kind_of, to_flat
from botorch_rs.exceptions.errors import BotorchTensorDimensionError
from botorch_rs.kernels import _batch_matrix, _indices
from botorch_rs.posteriors.gpytorch import Posterior
from botorch_rs.posteriors.higher_order import _randn


class _GridDistribution:
    def __init__(self, posterior: LatentKroneckerGPPosterior) -> None:
        self.posterior = posterior

    @property
    def mean(self) -> Any:
        p = self.posterior
        return p._query_out("mean", p._moments()[0], (*p._batch, p._q * p._tasks))

    @property
    def variance(self) -> Any:
        p = self.posterior
        return p._query_out("variance", p._diagonal(), (*p._batch, p._q * p._tasks))

    @property
    def covariance_matrix(self) -> Any:
        p = self.posterior
        return p._query_out(
            "covariance", p._moments()[1], (*p._batch, p._q * p._tasks, p._q * p._tasks)
        )


class LatentKroneckerGPPosterior(Posterior):
    """Exact moments and Matheron samples on a query X x T Cartesian grid.

    Mean, variance and samples consistently use batch x q x t shapes and
    original outcome units. Base draws use the official layout of full
    training-grid prior, observed training noise and conditional test prior.
    """

    def __init__(self, model: Any, distribution: Any = None, X: Any = None, T: Any = None) -> None:
        # Preserve the older native-only (model, X, T) call alongside the
        # public BoTorch (model, distribution, X, T) constructor.
        if T is None and X is not None and distribution is not None:
            from botorch_rs import _botorch_rs as _rs

            if isinstance(getattr(model, "_rs", None), _rs.LatentKroneckerGP):
                distribution, X, T = None, distribution, X
        if X is None or T is None:
            raise TypeError("LatentKroneckerGPPosterior requires model, X, and T.")
        self._external_distribution = distribution
        if distribution is not None:
            from botorch_rs.posteriors._distribution import GaussianDistributionAdapter

            GaussianDistributionAdapter(distribution)
            from botorch_rs import _botorch_rs as _rs

            if not isinstance(getattr(model, "_rs", None), _rs.LatentKroneckerGP):
                from botorch_rs.posteriors._structured import SuppliedLatent

                adapter = SuppliedLatent(model, distribution, X, T)
                self._supplied = adapter
                self.model, self.X, self.T = model, X, T
                self._kind = kind_of(X)
                self._batch = adapter.output_batch_shape
                self.batch_shape = adapter.batch_shape
                self.output_batch_shape = self._batch
                self._q, self._tasks = X.shape[-2], T.shape[-2]
                self._is_mt = True
                return
            supplied_batch = broadcast_shapes(tuple(X.shape[:-2]), tuple(T.shape[:-2]))
            if supplied_batch is None or tuple(distribution.mean.shape) != (
                *supplied_batch,
                X.shape[-2],
                T.shape[-2],
            ):
                raise BotorchTensorDimensionError(
                    "distribution.mean must have batch x q x t shape."
                )
        self.model, self.X, self.T = model, X, T
        self._rs = model._rs.copy()
        self._kind = kind_of(X)
        self._query_inputs = X.clone() if self._kind.lib == "torch" else None
        self._query_times = T.clone() if self._kind.lib == "torch" and hasattr(T, "clone") else None
        self._is_mt = True
        self._x, self._t = to_flat(X, "X"), to_flat(T, "T")
        if (
            len(self._x[1]) < 2
            or len(self._t[1]) < 2
            or self._t[1][-1] != 1
            or self._x[1][-1] != self._rs.dim
        ):
            raise BotorchTensorDimensionError("Expected X batch x q x d and T batch x t x 1.")
        self._q, self._tasks = self._x[1][-2], self._t[1][-2]
        if self._q == 0 or self._tasks == 0:
            raise BotorchTensorDimensionError("Feature and time query grids must be nonempty.")
        batch = broadcast_shapes(tuple(self._x[1][:-2]), tuple(self._t[1][:-2]))
        if batch is None:
            raise BotorchTensorDimensionError("Feature and time query batches do not broadcast.")
        self._batch = tuple(batch)
        self.batch_shape = model.batch_shape
        self.output_batch_shape = self._batch
        self._cache: tuple[list[float], list[float]] | None = None

    def _batches(self) -> list[Any]:
        return [
            (_batch_matrix(self._x, index), _batch_matrix(self._t, index))
            for index in _indices(self._batch)
        ]

    def _moments(self) -> tuple[list[float], list[float]]:
        if self._cache is None:
            means, covariances = [], []
            for x, t in self._batches():
                (mean, _), (covariance, _) = self._rs.posterior_grid(x, t)
                means.extend(mean)
                covariances.extend(covariance)
            self._cache = means, covariances
        return self._cache

    def _diagonal(self) -> list[float]:
        width = self._q * self._tasks
        covariance = self._moments()[1]
        return [
            max(covariance[b * width * width + i * width + i], 1e-10)
            for b in range(math.prod(self._batch))
            for i in range(width)
        ]

    @property
    def mean(self) -> Any:
        if self._external_distribution is not None:
            return self._external_distribution.mean
        return self._query_out("mean", self._moments()[0], (*self._batch, self._q, self._tasks))

    @property
    def variance(self) -> Any:
        if self._external_distribution is not None:
            return self._external_distribution.variance
        return self._query_out("variance", self._diagonal(), (*self._batch, self._q, self._tasks))

    def _query_out(self, mode: str, values: Any, shape: Any, normals: Any = None) -> Any:
        from botorch_rs.posteriors._structured_query_autograd import output

        return output(self, mode, values, shape, normals)

    @property
    def dtype(self) -> Any:
        return getattr(self.X, "dtype", None)

    @property
    def device(self) -> Any:
        return getattr(self.X, "device", None)

    @property
    def distribution(self) -> _GridDistribution:
        if self._external_distribution is not None:
            return self._external_distribution
        return _GridDistribution(self)

    @property
    def mvn(self) -> _GridDistribution:
        return self.distribution

    @property
    def base_sample_shape(self) -> tuple[int, ...]:
        if (adapter := self.__dict__.get("_supplied")) is not None:
            return adapter.base_sample_shape
        full = self._rs.num_train * self.model.num_outputs
        return (full + self._rs.num_observed + self._q * self._tasks,)

    @property
    def batch_range(self) -> tuple[int, int]:
        return 0, -1

    def _extended_shape(self, sample_shape: Any = ()) -> tuple[int, ...]:
        return (*tuple(sample_shape), *self._batch, self._q, self._tasks)

    def rsample_from_base_samples(self, sample_shape: Any, base_samples: Any) -> Any:
        if (adapter := self.__dict__.get("_supplied")) is not None:
            return adapter.rsample(sample_shape, base_samples)
        sample_shape = tuple(sample_shape)
        data, shape = to_flat(base_samples, "base_samples")
        k = len(sample_shape)
        if tuple(shape[:k]) != sample_shape:
            raise RuntimeError("sample_shape disagrees with the shape of base_samples.")
        total = self.base_sample_shape[-1]
        if not shape or shape[-1] != total:
            raise BotorchTensorDimensionError(f"Expected {total} base samples per draw.")
        ns, nb = math.prod(sample_shape), math.prod(self._batch)
        per_batch = math.prod(shape[k:-1])
        if per_batch not in (1, nb):
            raise BotorchTensorDimensionError("Base sample batches do not match the posterior.")
        results = []
        for b, (x, t) in enumerate(self._batches()):
            rows = [
                v
                for sample in range(ns)
                for v in data[
                    (sample * per_batch + (b if per_batch > 1 else 0)) * total : (
                        sample * per_batch + (b if per_batch > 1 else 0) + 1
                    )
                    * total
                ]
            ]
            samples, _ = self._rs.samples(x, t, (rows, (ns, total)))
            results.append(samples)
        width = self._q * self._tasks
        values = [
            v
            for sample in range(ns)
            for result in results
            for v in result[sample * width : (sample + 1) * width]
        ]
        return self._query_out(
            "samples",
            values,
            self._extended_shape(sample_shape),
            (list(data), (ns, per_batch, total)),
        )

    def rsample(self, sample_shape: Any = None) -> Any:
        if (adapter := self.__dict__.get("_supplied")) is not None:
            return adapter.rsample(sample_shape)
        sample_shape = (1,) if sample_shape is None else tuple(sample_shape)
        shape = (*sample_shape, *self.base_sample_shape)
        return self.rsample_from_base_samples(
            sample_shape, from_flat(_randn(math.prod(shape), self._kind), shape, self._kind)
        )

    def quantile(self, value: Any) -> Any:
        from botorch_rs.posteriors._structured import marginal

        return marginal(self, value, density=False)

    def density(self, value: Any) -> Any:
        from botorch_rs.posteriors._structured import marginal

        return marginal(self, value, density=True)


__all__ = ["LatentKroneckerGPPosterior"]

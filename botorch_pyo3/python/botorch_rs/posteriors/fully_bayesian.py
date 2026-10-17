"""Posteriors of ensembles of GPs (``botorch.posteriors.fully_bayesian``):
``GaussianMixturePosterior``, the posterior of BoTorch's
``EnsembleMapSaasSingleTaskGP``."""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

from botorch_rs._arrays import from_flat, to_flat
from botorch_rs.posteriors.gpytorch import GPyTorchPosterior

MCMC_DIM = -3  # Location of the MCMC batch dimension
TOL = 1e-6  # Bisection tolerance


def batched_bisect(
    f: Callable[[list[float]], list[float]],
    target: float,
    bounds: list[list[float]],
    tol: float = TOL,
    max_steps: int = 32,
) -> list[float]:
    r"""Batched bisection with a fixed number of steps.

    Args:
        f: Target function that takes a list of ``b`` values and returns ``b``
            values.
        target: Scalar target value of type float.
        bounds: Lower and upper bounds, ``[lower, upper]`` with ``b`` values each.
        tol: We terminate if all elements are within ``tol`` of the ``target``.
        max_steps: Maximum number of bisection steps.

    Returns:
        ``b`` values ``x`` such that ``f(x) = target``.
    """
    lo, hi = list(bounds[0]), list(bounds[1])
    f1, f2 = f(lo), f(hi)
    if not all(a <= target <= b for a, b in zip(f1, f2, strict=True)):
        raise ValueError("The target is not contained in the interval specified by the bounds")
    center = [(a + b) / 2 for a, b in zip(lo, hi, strict=True)]
    f_center = f(center)
    for _ in range(max_steps):
        for i, fc in enumerate(f_center):
            if fc > target:
                hi[i] = center[i]
            else:
                lo[i] = center[i]
        center = [(a + b) / 2 for a, b in zip(lo, hi, strict=True)]
        f_center = f(center)
        if max(abs(fc - target) for fc in f_center) <= tol:
            return center
    return center


def _norm_cdf(x: float, mu: float, sd: float) -> float:
    return 0.5 * math.erfc(-(x - mu) / (sd * math.sqrt(2.0)))


def _norm_icdf(p: float, mu: float, sd: float) -> float:
    from statistics import NormalDist

    return NormalDist(mu, sd).inv_cdf(p)


class GaussianMixturePosterior(GPyTorchPosterior):
    r"""A Gaussian mixture posterior.

    The MCMC batch dimension that corresponds to the models in the mixture is located
    at ``MCMC_DIM`` (defined at the top of this file). Note that while each MCMC sample
    corresponds to a Gaussian posterior, the posterior is rather a mixture of Gaussian
    distributions.

    ``mean``, ``variance``, ``distribution`` and samples are those of the
    components (``batch_shape x num_models x q x m``); ``mixture_mean``,
    ``mixture_variance`` and ``mixture_covariance_matrix`` those of the
    mixture.
    """

    @classmethod
    def _from_posterior(cls, posterior: GPyTorchPosterior) -> GaussianMixturePosterior:
        """The mixture posterior of a posterior whose batch shape ends with
        the components."""
        obj = cls.__new__(cls)
        obj.__dict__.update(posterior.__dict__)
        return obj

    def _flat(self, values: Any) -> tuple[list[float], tuple[int, ...]]:
        data, shape = to_flat(values, "values")
        return list(data), tuple(shape)

    def _components(self) -> tuple[int, int, int]:
        """``(outer batch size, number of components, q * m)``."""
        k = self._batch[-1] if self._batch else 1
        outer = math.prod(self._batch[:-1]) if self._batch else 1
        return outer, k, self._q * self._m

    @property
    def mixture_mean(self) -> Any:
        r"""The posterior mean for the mixture of models."""
        mean, _ = self._flat(self.mean)
        outer, k, qm = self._components()
        out = [
            sum(mean[(b * k + j) * qm + i] for j in range(k)) / k
            for b in range(outer)
            for i in range(qm)
        ]
        return from_flat(out, (*self._batch[:-1], self._q, self._m), self._kind)

    @property
    def mixture_variance(self) -> Any:
        r"""The posterior variance for the mixture of models."""
        mean, _ = self._flat(self.mean)
        var, _ = self._flat(self.variance)
        outer, k, qm = self._components()
        out = []
        for b in range(outer):
            for i in range(qm):
                idx = [(b * k + j) * qm + i for j in range(k)]
                t1 = sum(var[t] for t in idx) / k
                t2 = sum(mean[t] ** 2 for t in idx) / k
                t3 = -((sum(mean[t] for t in idx) / k) ** 2)
                out.append(t1 + t2 + t3)
        return from_flat(out, (*self._batch[:-1], self._q, self._m), self._kind)

    @property
    def mixture_covariance_matrix(self) -> Any:
        r"""The posterior covariance matrix for the mixture of models
        (single-output posteriors: ``batch_shape x q x q``)."""
        cov, _ = self._flat(self.distribution.covariance_matrix)
        mean, _ = self._flat(self.mean)
        outer, k, qm = self._components()
        mixture, _ = self._flat(self.mixture_mean)
        out = []
        for b in range(outer):
            for r in range(qm):
                for c in range(qm):
                    t1 = sum(cov[(b * k + j) * qm * qm + r * qm + c] for j in range(k)) / k
                    t2 = (
                        sum(
                            (mean[(b * k + j) * qm + r] - mixture[b * qm + r])
                            * (mean[(b * k + j) * qm + c] - mixture[b * qm + c])
                            for j in range(k)
                        )
                        / k
                    )
                    out.append(t1 + t2)
        return from_flat(out, (*self._batch[:-1], qm, qm), self._kind)

    def quantile(self, value: Any) -> Any:
        r"""Compute the posterior quantiles for the mixture of models (shape
        ``value_shape x batch_shape x q x m`` without the component
        dimension)."""
        values, vshape = self._flat(value)
        mean, _ = self._flat(self.mean)
        var, _ = self._flat(self.variance)
        outer, k, qm = self._components()
        results: list[float] = []
        sds = [math.sqrt(x) for x in var]
        for v in values:
            if v <= 0 or v >= 1:
                raise ValueError("value is expected to be in the range (0, 1).")
            if k == 1:
                results.extend(_norm_icdf(v, m, s) for m, s in zip(mean, sds, strict=True))
                continue
            low, high = [], []
            for b in range(outer):
                for i in range(qm):
                    icdf = [
                        _norm_icdf(v, mean[(b * k + j) * qm + i], sds[(b * k + j) * qm + i])
                        for j in range(k)
                    ]
                    low.append(min(icdf) - TOL)
                    high.append(max(icdf) + TOL)

            def mixture_cdf(x: list[float]) -> list[float]:
                out = []
                for b in range(outer):
                    for i in range(qm):
                        t = b * qm + i
                        out.append(
                            sum(
                                _norm_cdf(
                                    x[t], mean[(b * k + j) * qm + i], sds[(b * k + j) * qm + i]
                                )
                                for j in range(k)
                            )
                            / k
                        )
                return out

            results.extend(batched_bisect(f=mixture_cdf, target=v, bounds=[low, high]))
        shape = (*self._batch[:-1], self._q, self._m)
        if vshape and math.prod(vshape) > 1:
            shape = (len(values), *shape)
        return from_flat(results, shape, self._kind)

    @property
    def batch_range(self) -> tuple[int, int]:
        r"""The t-batch range."""
        return (0, -2) if self._is_mt else (0, -1)

    def __repr__(self) -> str:
        return f"GaussianMixturePosterior(batch_shape={self._batch}, q={self._q}, m={self._m})"


__all__ = ["MCMC_DIM", "GaussianMixturePosterior", "batched_bisect"]

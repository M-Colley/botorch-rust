"""Posteriors of ensembles of GPs (``botorch.posteriors.fully_bayesian``):
``GaussianMixturePosterior``, the posterior of BoTorch's
``EnsembleMapSaasSingleTaskGP``."""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
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
        (``batch_shape x q x q``; ``batch_shape x (q m) x (q m)`` in the
        output-major order of the covariance for several outputs, where
        BoTorch fails: see the draft of phase 10a)."""
        cov, _ = self._flat(self.distribution.covariance_matrix)
        mean, _ = self._flat(self.mean)
        outer, k, qm = self._components()
        mixture, _ = self._flat(self.mixture_mean)
        q, m = self._q, self._m

        def at(r: int) -> int:
            # The mean index (point-major) of covariance row r (output-major).
            return r if m == 1 else (r % q) * m + r // q

        out = []
        for b in range(outer):
            for r in range(qm):
                for c in range(qm):
                    t1 = sum(cov[(b * k + j) * qm * qm + r * qm + c] for j in range(k)) / k
                    t2 = (
                        sum(
                            (mean[(b * k + j) * qm + at(r)] - mixture[b * qm + at(r)])
                            * (mean[(b * k + j) * qm + at(c)] - mixture[b * qm + at(c)])
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


def _stack(parts: Sequence[tuple[Any, Any]], outer: int) -> list[float]:
    """Stacks arrays of shape ``outer x tail`` (flat) along a new axis after
    the ``outer`` (flattened) leading dimensions."""
    tail = len(parts[0][0]) // max(outer, 1) if outer else 0
    out: list[float] = []
    for o in range(outer):
        for data, _ in parts:
            out.extend(data[o * tail : (o + 1) * tail])
    return out


class _StackedMixturePosterior(GaussianMixturePosterior):
    """The Gaussian mixture of the posteriors of separate models (the MCMC
    samples of a fully Bayesian multi-task GP, which botorch_rs keeps as one
    ``MultiTaskGP`` per sample), stacked at ``MCMC_DIM``: BoTorch's
    ``GaussianMixturePosterior`` of the batched model."""

    def __init__(self, posteriors: list[GPyTorchPosterior], kind: Any) -> None:
        p0 = posteriors[0]
        self._parts = posteriors
        self._kind = kind
        self._outer = tuple(p0._batch)
        self._batch = (*self._outer, len(posteriors))
        self._q = p0._q
        self._m = p0._m
        self._marginals: Any = None
        self._joint: Any = None

    def _get_marginals(self) -> tuple[list[float], list[float]]:
        if self._marginals is None:
            nb = math.prod(self._outer)
            parts = [p._get_marginals() for p in self._parts]
            mean = _stack([(m, ()) for m, _ in parts], nb)
            var = _stack([(v, ()) for _, v in parts], nb)
            self._marginals = (mean, var)
        return self._marginals

    def _get_joint(self) -> Any:
        if self._joint is None:
            nb = math.prod(self._outer)
            parts = [p._get_joint() for p in self._parts]
            mean = _stack([(list(m[0]), ()) for m, _ in parts], nb)
            cov = _stack([(list(c[0]), ()) for _, c in parts], nb)
            cshape = tuple(parts[0][1][1])
            mshape = tuple(parts[0][0][1])
            k = len(self._outer)
            self._joint = (
                (mean, (*mshape[:k], len(parts), *mshape[k:])),
                (cov, (*cshape[:k], len(parts), *cshape[k:])),
            )
        return self._joint

    def _samples(self, z: Any) -> list[float]:
        """Samples of every component from the same base samples ``z``
        (``n x q m``, shared by the batch, as BoTorch's samplers collapse the
        batch dimensions)."""
        data, shape = z
        n = shape[0]
        nb = math.prod(self._outer)
        parts = [(p._samples((data, (n, shape[-1]))), ()) for p in self._parts]
        return _stack(parts, n * nb)

    def rsample(self, sample_shape: Any = None) -> Any:
        """Posterior samples, shape ``sample_shape x batch_shape x q x m``,
        with base samples shared by the components."""
        from botorch_rs.posteriors.gpytorch import _as_shape, _standard_normal

        sample_shape = _as_shape(sample_shape)
        n = math.prod(sample_shape)
        qm = self._q * self._m
        z = _standard_normal(n * qm, self._kind)
        data = self._samples((z, (n, qm)))
        return self._out(data, (*sample_shape, *self._batch, self._q, self._m))

    def rsample_from_base_samples(self, sample_shape: Any, base_samples: Any) -> Any:
        """Posterior samples from base samples ``sample_shape x 1 ... x q``
        (``x m``) shared by the batch and the components."""
        from botorch_rs.posteriors.gpytorch import _as_shape

        sample_shape = _as_shape(sample_shape)
        data, zshape = to_flat(base_samples, "base_samples")
        n = math.prod(sample_shape)
        qm = self._q * self._m
        if len(data) != n * qm:
            raise RuntimeError(
                f"botorch_rs expects base samples shared by the batch and the mixture components "
                f"({n * qm} values), got shape {tuple(zshape)}."
            )
        out = self._samples((list(data), (n, qm)))
        return self._out(out, (*sample_shape, *self._batch, self._q, self._m))

    def __repr__(self) -> str:
        return f"GaussianMixturePosterior(batch_shape={self._batch}, q={self._q}, m={self._m})"


__all__ = ["MCMC_DIM", "GaussianMixturePosterior", "batched_bisect"]

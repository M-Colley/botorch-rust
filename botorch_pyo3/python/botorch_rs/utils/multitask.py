r"""
Helpers for multitask modeling (``botorch.utils.multitask``).
"""

from __future__ import annotations

import math
from typing import Any

from botorch_rs import _botorch_rs as _rs
from botorch_rs._arrays import ArrayKind, from_flat, to_flat
from botorch_rs.posteriors.gpytorch import MultitaskMultivariateNormal


class DenseMultivariateNormal:
    """A multivariate normal distribution with an explicit mean (``batch_shape
    x n``) and covariance (``batch_shape x n x n``), with the attributes of
    GPyTorch's ``MultivariateNormal`` that botorch_rs uses."""

    def __init__(self, mean: Any, covariance_matrix: Any, kind: ArrayKind) -> None:
        self._mean = mean
        self._cov = covariance_matrix
        self._kind = kind

    @property
    def mean(self) -> Any:
        return self._mean

    loc = mean

    @property
    def covariance_matrix(self) -> Any:
        return self._cov

    lazy_covariance_matrix = covariance_matrix

    @property
    def variance(self) -> Any:
        data, shape = to_flat(self._cov, "covariance_matrix")
        n = shape[-1]
        nb = math.prod(shape[:-2])
        diag = [float(data[b * n * n + i * n + i]) for b in range(nb) for i in range(n)]
        return from_flat(diag, shape[:-1], self._kind)

    @property
    def stddev(self) -> Any:
        data, shape = to_flat(self.variance, "variance")
        return from_flat([math.sqrt(v) for v in data], shape, self._kind)

    @property
    def batch_shape(self) -> tuple[int, ...]:
        return tuple(to_flat(self._mean, "mean")[1][:-1])

    @property
    def event_shape(self) -> tuple[int, ...]:
        return tuple(to_flat(self._mean, "mean")[1][-1:])

    def rsample(self, sample_shape: Any = (), base_samples: Any = None) -> Any:
        """Samples ``sample_shape x batch_shape x n`` (``mean + L z`` with the
        lower Cholesky factor ``L``; ``z`` from ``base_samples`` of that
        shape, else drawn with torch's generator for torch arrays and
        Python's ``random`` otherwise)."""
        import random
        import sys

        sample_shape = tuple(int(v) for v in sample_shape)
        S = math.prod(sample_shape)
        mean, mshape = to_flat(self._mean, "mean")
        cov, _ = to_flat(self._cov, "covariance_matrix")
        n = mshape[-1]
        nb = math.prod(mshape[:-1])
        L, _ = _rs.psd_safe_cholesky(list(cov), nb, n)
        total = S * nb * n
        if base_samples is not None:
            z = [float(v) for v in to_flat(base_samples, "base_samples")[0]]
        elif self._kind.lib == "torch":
            torch = sys.modules["torch"]
            z = torch.randn(total, dtype=torch.float64).tolist()
        else:
            z = [random.gauss(0.0, 1.0) for _ in range(total)]
        out = []
        for s in range(S):
            for b in range(nb):
                zz = z[(s * nb + b) * n : (s * nb + b + 1) * n]
                for i in range(n):
                    acc = float(mean[b * n + i])
                    for k in range(i + 1):
                        acc += float(L[b * n * n + i * n + k]) * zz[k]
                    out.append(acc)
        return from_flat(out, (*sample_shape, *mshape), self._kind)

    def __repr__(self) -> str:
        return f"DenseMultivariateNormal(batch_shape={self.batch_shape}, event_shape={self.event_shape})"


def separate_mtmvn(mvn: Any) -> list[Any]:
    """
    Separate a MTMVN into a list of MVNs, where covariance across data within each task
    are preserved, while covariance across task are dropped.

    For a botorch_rs ``MultitaskMultivariateNormal`` the results are
    ``DenseMultivariateNormal``\\ s; GPyTorch's distributions give GPyTorch
    ``MultivariateNormal``\\ s, as in BoTorch.
    """
    if isinstance(mvn, MultitaskMultivariateNormal):
        p = mvn._p
        (mean, mshape), (cov, cshape) = p._get_joint()
        batch = tuple(cshape[:-2])
        nb = math.prod(batch)
        q, m = p._q, p._m
        N = q * m
        out = []
        for c in range(m):
            mu = [float(mean[(b * q + r) * m + c]) for b in range(nb) for r in range(q)]
            block = [
                float(cov[b * N * N + (c * q + r) * N + c * q + k])
                for b in range(nb)
                for r in range(q)
                for k in range(q)
            ]
            out.append(
                DenseMultivariateNormal(
                    from_flat(mu, (*batch, q), p._kind),
                    from_flat(block, (*batch, q, q), p._kind),
                    p._kind,
                )
            )
        return out
    import importlib

    torch = importlib.import_module("torch")
    gpytorch_mvn = importlib.import_module("gpytorch.distributions.multivariate_normal")
    to_linear_operator = importlib.import_module("linear_operator").to_linear_operator
    full_covar = mvn.lazy_covariance_matrix
    num_data, num_tasks = mvn.mean.shape[-2:]
    mvns = []
    for c in range(num_tasks):
        if mvn._interleaved:
            task_indices = torch.arange(c, num_data * num_tasks, num_tasks, device=full_covar.device)
            task_covar = full_covar[..., task_indices, :]
            task_covar = task_covar[..., :, task_indices]
        else:
            start = c * num_data
            end = start + num_data
            task_covar = full_covar[..., start:end, start:end]
        mvns.append(
            gpytorch_mvn.MultivariateNormal(mvn.mean[..., c], to_linear_operator(task_covar))
        )
    return mvns


__all__ = ["DenseMultivariateNormal", "separate_mtmvn"]

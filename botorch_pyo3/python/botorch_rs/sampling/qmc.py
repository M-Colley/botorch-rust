r"""
Quasi Monte-Carlo sampling from Normal distributions (``botorch.sampling.qmc``).

The uniform points come from torch's scrambled ``SobolEngine`` if torch is
available (so that the samples equal BoTorch's for the same seed), else from
botorch_rs's scrambled Sobol sequence (Owen scrambling; same distribution,
different points). The Box-Muller and inverse transforms and the root of the
covariance (Cholesky, or the eigendecomposition for a singular covariance)
are computed in Rust.

References:

.. [Pages2018numprob]
    G. Pages. Numerical Probability: An Introduction with Applications to
    Finance. Universitext. Springer International Publishing, 2018.
"""

from __future__ import annotations

import importlib
import math
import sys
from typing import Any

from botorch_rs import _botorch_rs as _rs
from botorch_rs import _ops
from botorch_rs._arrays import ArrayKind, from_flat, from_pair, kind_of, to_flat


def _default_kind(dtype: Any) -> ArrayKind:
    torch = sys.modules.get("torch")
    if dtype is None:
        if torch is not None:
            return ArrayKind("torch", torch.get_default_dtype())
        from botorch_rs._arrays import LIST

        return LIST
    from botorch_rs._arrays import kind_from_dtype

    return kind_from_dtype(dtype)


class _RustSobol:
    """Scrambled Sobol points of botorch_rs (without torch)."""

    def __init__(self, dimension: int, seed: int | None) -> None:
        self.dimension = dimension
        self.seed = seed
        self.num_generated = 0

    def draw(self, n: int) -> list[float]:
        # Points num_generated .. num_generated + n of the sequence.
        total = self.num_generated + n
        data, _ = _rs.draw_sobol_samples(
            ([0.0] * self.dimension + [1.0] * self.dimension, (2, self.dimension)),
            total,
            1,
            self.seed,
        )
        out = list(data)[self.num_generated * self.dimension :]
        self.num_generated = total
        return out


class NormalQMCEngine:
    r"""Engine for qMC sampling from a Multivariate Normal ``N(0, I_d)``.

    By default, this implementation uses Box-Muller transformed Sobol samples
    following pg. 123 in [Pages2018numprob]_. To use the inverse transform
    instead, set ``inv_transform=True``.

    Example:
        >>> engine = NormalQMCEngine(3)
        >>> samples = engine.draw(16)
    """

    def __init__(self, d: int, seed: int | None = None, inv_transform: bool = False) -> None:
        r"""Engine for drawing qMC samples from a multivariate normal ``N(0, I_d)``.

        Args:
            d: The dimension of the samples.
            seed: The seed with which to seed the random number generator of the
                underlying SobolEngine.
            inv_transform: If True, use inverse transform instead of Box-Muller.
        """
        self._d = d
        self._seed = seed
        self._inv_transform = inv_transform
        if inv_transform:
            sobol_dim = d
        else:
            # to apply Box-Muller, we need an even number of dimensions
            sobol_dim = 2 * math.ceil(d / 2)
        self._sobol_dim = sobol_dim
        torch = sys.modules.get("torch")
        if torch is None:
            try:
                torch = importlib.import_module("torch")
            except ImportError:
                torch = None
        if torch is not None:
            self._sobol_engine: Any = torch.quasirandom.SobolEngine(
                dimension=sobol_dim, scramble=True, seed=seed
            )
        else:
            self._sobol_engine = _RustSobol(sobol_dim, seed)

    def _uniform(self, n: int, kind: ArrayKind) -> list[float]:
        if isinstance(self._sobol_engine, _RustSobol):
            return self._sobol_engine.draw(n)
        torch = sys.modules["torch"]
        dtype = kind.dtype if kind.lib == "torch" and kind.dtype is not None else torch.float64
        return self._sobol_engine.draw(n, dtype=dtype).reshape(-1).tolist()

    def draw(self, n: int = 1, out: Any = None, dtype: Any = None) -> Any:
        r"""Draw ``n`` qMC samples from the standard Normal.

        Args:
            n: The number of samples to draw. As a best practice, use powers of 2.
            out: An option output tensor. If provided, draws are put into this
                tensor, and the function returns None.
            dtype: The desired torch data type (ignored if ``out`` is provided).
                If None, uses ``torch.get_default_dtype()``.

        Returns:
            A ``n x d`` tensor of samples if ``out=None`` and ``None`` otherwise.
        """
        kind = _default_kind(dtype)
        if out is not None:
            kind = kind_of(out)
        samples_tf = self._draw_kind(n, kind)
        if out is None:
            return samples_tf
        if kind.lib == "torch":
            out.copy_(samples_tf)
        else:
            out[...] = samples_tf
        return None

    def _draw_kind(self, n: int, kind: ArrayKind) -> Any:
        uniform = self._uniform(n, kind)
        data, _ = _rs.normal_qmc_transform(
            uniform,
            n,
            self._sobol_dim,
            self._d,
            bool(self._inv_transform),
            _ops.is_single_precision(kind),
        )
        return from_flat(data, (n, self._d), kind)


class MultivariateNormalQMCEngine:
    r"""Engine for qMC sampling from a multivariate Normal ``N(\mu, \Sigma)``.

    By default, this implementation uses Box-Muller transformed Sobol samples
    following pg. 123 in [Pages2018numprob]_. To use the inverse transform
    instead, set ``inv_transform=True``.

    Example:
        >>> mean = torch.tensor([1.0, 2.0])
        >>> cov = torch.tensor([[1.0, 0.25], [0.25, 2.0]])
        >>> engine = MultivariateNormalQMCEngine(mean, cov)
        >>> samples = engine.draw(16)
    """

    def __init__(
        self,
        mean: Any,
        cov: Any,
        seed: int | None = None,
        inv_transform: bool = False,
    ) -> None:
        r"""Engine for qMC sampling from a multivariate Normal ``N(\mu, \Sigma)``.

        Args:
            mean: The mean vector.
            cov: The covariance matrix.
            seed: The seed with which to seed the random number generator of the
                underlying SobolEngine.
            inv_transform: If True, use inverse transform instead of Box-Muller.
        """
        cshape = _ops.shape_of(cov)
        mshape = _ops.shape_of(mean)
        # validate inputs
        if not cshape[0] == cshape[1]:
            raise ValueError("Covariance matrix is not square.")
        if not mshape[0] == cshape[0]:
            raise ValueError("Dimension mismatch between mean and covariance.")
        self._mean = mean
        self._kind = kind_of(mean)
        self._normal_engine = NormalQMCEngine(d=mshape[0], seed=seed, inv_transform=inv_transform)
        try:
            root = _rs.mvn_qmc_root(to_flat(cov, "cov"), _ops.is_single_precision(kind_of(cov)))
        except ValueError as e:
            raise ValueError(str(e)) from None
        self._corr_matrix = from_pair(root, kind_of(cov))

    def draw(self, n: int = 1, out: Any = None) -> Any:
        r"""Draw ``n`` qMC samples from the multivariate Normal.

        Args:
            n: The number of samples to draw. As a best practice, use powers of 2.
            out: An option output tensor. If provided, draws are put into this
                tensor, and the function returns None.

        Returns:
            A ``n x d`` tensor of samples if ``out=None`` and ``None`` otherwise.
        """
        kind = kind_of(out) if out is not None else self._kind
        base_samples = self._normal_engine._draw_kind(n, kind)
        corr = to_flat(self._corr_matrix)
        mean = to_flat(self._mean)[0]
        b, _ = to_flat(base_samples)
        d = corr[1][0]
        out_data = [
            math.fsum(float(b[i * d + k]) * float(corr[0][k * d + j]) for k in range(d))
            + float(mean[j])
            for i in range(n)
            for j in range(d)
        ]
        qmc_samples = from_flat(out_data, (n, d), kind)
        if out is None:
            return qmc_samples
        if kind.lib == "torch":
            out.copy_(qmc_samples)
        else:
            out[...] = qmc_samples
        return None


__all__ = ["MultivariateNormalQMCEngine", "NormalQMCEngine"]

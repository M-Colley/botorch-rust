"""Hyperparameter priors (``gpytorch.priors``).

Univariate priors (``LogNormalPrior``, ``GammaPrior``, ``NormalPrior``,
``UniformPrior``, ``HalfCauchyPrior``, ``HalfNormalPrior``,
``SmoothedBoxPrior``; BoTorch's
``BetaPrior`` is in :mod:`botorch_rs.models.utils.priors`) on the actual
(constrained) values of a hyperparameter, and the LKJ priors on correlation
and covariance matrices (``LKJCholeskyFactorPrior``, ``LKJPrior``,
``LKJCovariancePrior``, e.g. on the task covariance of an ``IndexKernel``).

Like GPyTorch, which stores prior parameters given as Python floats in
float32, the parameters are rounded to single precision (so that log
probabilities match BoTorch's models to double precision; where torch
evaluates parts of a prior in single precision, those parts are emulated).
Exceptions: ``HalfCauchyPrior`` is evaluated in double precision (torch
evaluates it entirely in float32 after casting its argument), and the
normalizing constant of the LKJ priors in double precision from the
float32-rounded ``eta`` (torch keeps ``eta`` as a float32 tensor).

Priors that torch validates (``UniformPrior``, ``HalfCauchyPrior``,
``HalfNormalPrior``) raise a ``ValueError`` for arguments outside their
support, as in GPyTorch.
"""

from __future__ import annotations

import importlib
import math
import struct
import sys
from typing import Any

from botorch_rs import _botorch_rs as _rs
from botorch_rs._arrays import as_float, from_flat, kind_of, to_flat
from botorch_rs.exceptions.errors import UnsupportedError


class Prior:
    """Base class of the priors."""

    _rs: _rs.Prior

    def log_prob(self, x: Any) -> Any:
        """The log density at ``x`` (elementwise for arrays)."""
        data, shape = to_flat(x, "x")
        for v in data:
            if not self._rs.in_support(v):
                raise ValueError(
                    f"Expected value argument ({v}) to be within the support of the "
                    f"distribution {type(self).__name__}, but found invalid values."
                )
        return from_flat([self._rs.log_prob(v) for v in data], shape, kind_of(x))

    @property
    def mode(self) -> float:
        """The mode of the distribution."""
        return self._rs.mode


def _check_transform(transform: Any) -> None:
    if transform is not None:
        raise UnsupportedError("Prior transforms are not supported by botorch_rs.")


def _is_double(value: Any) -> bool:
    """Whether ``value`` is a double-precision tensor or array (which GPyTorch
    keeps in its dtype)."""
    return str(getattr(value, "dtype", "")) in ("torch.float64", "float64")


def _float32(*values: Any) -> bool:
    """Whether GPyTorch would store prior parameters given as Python floats
    in single precision: torch's default dtype (float32 unless changed with
    ``torch.set_default_dtype``; float32 without torch). Parameters given as
    double-precision tensors (or arrays) keep their precision."""
    if any(_is_double(v) for v in values):
        return False
    # (Without torch imported, its default dtype is float32: torch is not
    # imported here.)
    torch = sys.modules.get("torch")
    if torch is None:
        return True
    return bool(torch.get_default_dtype() == torch.float32)


def _torch_log(prior: _rs.Prior, value: float) -> _rs.Prior:
    """``prior`` with torch's float32 ``log(value)`` (MKL's single-precision
    ``log`` is not always correctly rounded), if torch is installed."""
    try:
        torch = importlib.import_module("torch")
    except ImportError:
        return prior
    if not _float32():
        return prior
    return prior.with_torch_log(float(torch.tensor(value, dtype=torch.float32).log().item()))


class LogNormalPrior(Prior):
    """Log-normal prior: ``log(x) ~ Normal(loc, scale**2)``."""

    def __init__(
        self,
        loc: Any,
        scale: Any,
        validate_args: bool | None = None,
        transform: Any = None,
    ) -> None:
        _check_transform(transform)
        self.loc = as_float(loc, "loc")
        self.scale = as_float(scale, "scale")
        f32 = _float32(loc, scale)
        self._rs = _rs.Prior.log_normal(self.loc, self.scale, f32)
        if f32:
            self._rs = _torch_log(self._rs, self.scale)

    def __repr__(self) -> str:
        return f"LogNormalPrior(loc={self.loc}, scale={self.scale})"


class GammaPrior(Prior):
    """Gamma prior with density ``rate**c / Gamma(c) x**(c-1) exp(-rate x)``."""

    def __init__(
        self,
        concentration: Any,
        rate: Any,
        validate_args: bool | None = False,
        transform: Any = None,
    ) -> None:
        _check_transform(transform)
        self.concentration = as_float(concentration, "concentration")
        self.rate = as_float(rate, "rate")
        self._rs = _rs.Prior.gamma(self.concentration, self.rate, _float32(concentration, rate))

    def __repr__(self) -> str:
        return f"GammaPrior(concentration={self.concentration}, rate={self.rate})"


class NormalPrior(Prior):
    """Normal prior ``Normal(loc, scale**2)``."""

    def __init__(
        self,
        loc: Any,
        scale: Any,
        validate_args: bool | None = False,
        transform: Any = None,
    ) -> None:
        _check_transform(transform)
        self.loc = as_float(loc, "loc")
        self.scale = as_float(scale, "scale")
        self._rs = _rs.Prior.normal(self.loc, self.scale, _float32(loc, scale))

    def __repr__(self) -> str:
        return f"NormalPrior(loc={self.loc}, scale={self.scale})"


class UniformPrior(Prior):
    """Uniform prior on ``[a, b)``."""

    def __init__(
        self,
        a: Any,
        b: Any,
        validate_args: bool | None = None,
        transform: Any = None,
    ) -> None:
        _check_transform(transform)
        self.low = as_float(a, "a")
        self.high = as_float(b, "b")
        f32 = _float32(a, b)
        self._rs = _rs.Prior.uniform(self.low, self.high, f32)
        if f32:
            import struct

            def r(v: float) -> float:
                return float(struct.unpack("f", struct.pack("f", v))[0])

            # torch: log(high - low) of the float32 bounds, in float32.
            self._rs = _torch_log(self._rs, r(r(self.high) - r(self.low)))

    def __repr__(self) -> str:
        return f"UniformPrior(low={self.low}, high={self.high})"


class HalfCauchyPrior(Prior):
    """Half-Cauchy prior with density ``2 / (pi scale (1 + (x / scale)**2))``
    on ``x >= 0``."""

    def __init__(
        self, scale: Any, validate_args: bool | None = None, transform: Any = None
    ) -> None:
        _check_transform(transform)
        self.scale = as_float(scale, "scale")
        self._rs = _rs.Prior.half_cauchy(self.scale, _float32(scale))

    def __repr__(self) -> str:
        return f"HalfCauchyPrior(scale={self.scale})"


class HalfNormalPrior(Prior):
    """Half-normal prior with density ``2 Normal(x | 0, scale**2)`` on ``x >= 0``."""

    def __init__(
        self, scale: Any, validate_args: bool | None = None, transform: Any = None
    ) -> None:
        _check_transform(transform)
        self.scale = as_float(scale, "scale")
        f32 = _float32(scale)
        self._rs = _rs.Prior.half_normal(self.scale, f32)
        if f32:
            self._rs = _torch_log(self._rs, self.scale)

    def __repr__(self) -> str:
        return f"HalfNormalPrior(scale={self.scale})"


# ----------------------------------------------------------------------
# LKJ priors
# ----------------------------------------------------------------------


def _cholesky(m: list[list[float]]) -> list[list[float]]:
    n = len(m)
    lower = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1):
            s = m[i][j] - sum(lower[i][k] * lower[j][k] for k in range(j))
            if i == j:
                if s <= 0:
                    raise ValueError("Input is not a valid correlation matrix")
                lower[i][i] = math.sqrt(s)
            else:
                lower[i][j] = s / lower[j][j]
    return lower


def _matrices(x: Any, n: int) -> tuple[list[list[list[float]]], tuple[int, ...]]:
    data, shape = to_flat(x, "X")
    if len(shape) < 2 or shape[-1] != n or shape[-2] != n:
        raise ValueError(f"Correlation matrix is not of size n={n}")
    count = len(data) // (n * n)
    mats = [
        [[float(data[b * n * n + i * n + j]) for j in range(n)] for i in range(n)]
        for b in range(count)
    ]
    return mats, tuple(shape[:-2])


class LKJCholeskyFactorPrior(Prior):
    """LKJ prior on the Cholesky factor ``L`` of an ``n x n`` correlation
    matrix with concentration ``eta``
    (``gpytorch.priors.LKJCholeskyFactorPrior``)."""

    def __init__(
        self, n: int, eta: Any, validate_args: bool = False, transform: Any = None
    ) -> None:
        _check_transform(transform)
        if not isinstance(n, int) or n < 1:
            raise ValueError("n must be a positive integer")
        self.n = n
        # torch stores eta as a tensor: a float in torch's default dtype, a
        # tensor in its own dtype. The normalizing constant is evaluated in
        # double precision (torch: in eta's dtype).
        dtype = getattr(eta, "dtype", None)
        single = _float32() if dtype is None else str(dtype) in ("torch.float32", "float32")
        self.eta = float(as_float(eta, "eta"))
        if single:
            self.eta = struct.unpack("f", struct.pack("f", self.eta))[0]

    def _log_normalizer(self) -> float:
        dm1 = self.n - 1
        alpha = self.eta + 0.5 * dm1
        denominator = math.lgamma(alpha) * dm1
        numerator = dm1 * (dm1 - 1) / 4 * math.log(math.pi) + sum(
            math.lgamma(alpha - 0.5 - i / 2) for i in range(dm1)
        )
        pi_constant = 0.5 * dm1 * math.log(math.pi)
        return pi_constant + numerator - denominator

    def _log_prob_cholesky(self, lower: list[list[float]]) -> float:
        n = self.n
        total = 0.0
        for i in range(1, n):
            order = 2 * (self.eta - 1) + n - (i + 1)
            total += order * math.log(lower[i][i])
        return total - self._log_normalizer()

    def log_prob(self, x: Any) -> Any:
        """The log density of the Cholesky factors ``x`` (``... x n x n``)."""
        mats, batch = _matrices(x, self.n)
        values = [self._log_prob_cholesky(m) for m in mats]
        return from_flat(values, batch, kind_of(x))

    @property
    def mode(self) -> float:  # pragma: no cover - not defined for matrices
        raise NotImplementedError


class LKJPrior(LKJCholeskyFactorPrior):
    """LKJ prior on ``n x n`` correlation matrices (``gpytorch.priors.LKJPrior``)."""

    def log_prob(self, x: Any) -> Any:
        """The log density of the correlation matrices ``x`` (``... x n x n``)."""
        mats, batch = _matrices(x, self.n)
        values = []
        for m in mats:
            if any(abs(m[i][i] - 1) >= 1e-6 for i in range(self.n)):
                raise ValueError("Input is not a valid correlation matrix")
            values.append(self._log_prob_cholesky(_cholesky(m)))
        return from_flat(values, batch, kind_of(x))


class LKJCovariancePrior(LKJPrior):
    """An LKJ prior on the correlations of an ``n x n`` covariance matrix and
    ``sd_prior`` on its marginal standard deviations
    (``gpytorch.priors.LKJCovariancePrior``)."""

    def __init__(self, n: int, eta: Any, sd_prior: Prior, validate_args: bool = False) -> None:
        if not isinstance(sd_prior, Prior) or isinstance(sd_prior, LKJCholeskyFactorPrior):
            raise ValueError("sd_prior must be an instance of Prior")
        if not isinstance(n, int):
            raise ValueError("n must be an integer")
        super().__init__(n, eta)
        self.sd_prior = sd_prior
        self.correlation_prior = LKJPrior(n, eta)

    def log_prob(self, x: Any) -> Any:
        """The log densities of the covariance matrices ``x`` (``... x n x n``):
        the LKJ log density of the correlations plus the log densities of the
        standard deviations.

        Deviation: GPyTorch returns ``... x n`` values for a univariate
        ``sd_prior`` (the correlation term plus the log density of each
        standard deviation), whose sum, which a model's MLL adds, counts the
        correlation term ``n`` times."""
        mats, batch = _matrices(x, self.n)
        values = []
        for m in mats:
            if any(m[i][i] < 0 for i in range(self.n)):
                raise ValueError("Variance(s) cannot be negative")
            flat = [v for row in m for v in row]
            values.append(
                _rs.lkj_covariance_log_prob((flat, (self.n, self.n)), self.eta, self.sd_prior._rs)
            )
        return from_flat(values, batch, kind_of(x))


class SmoothedBoxPrior(Prior):
    r"""A smoothed approximation of a uniform prior on ``[a, b]``
    (``gpytorch.priors.SmoothedBoxPrior``): full support on the reals, with
    the log density ``log N(d(x) | 0, sigma**2) - log(1 + (b - a) /
    (sqrt(2 pi) sigma))``, where ``d(x)`` is the distance of ``x`` from the
    box. Its default ``sigma`` is 0.01. Only scalar ``a``, ``b`` and
    ``sigma`` are supported."""

    def __init__(
        self,
        a: Any,
        b: Any,
        sigma: Any = 0.01,
        validate_args: bool = False,
        transform: Any = None,
    ) -> None:
        _check_transform(transform)
        self.a = as_float(a, "a")
        self.b = as_float(b, "b")
        self.sigma = as_float(sigma, "sigma")
        if not self.a < self.b:
            raise ValueError("must have that a < b (element-wise)")
        self._rs = _rs.Prior.smoothed_box(self.a, self.b, self.sigma, _float32())

    def __repr__(self) -> str:
        return f"SmoothedBoxPrior(a={self.a}, b={self.b}, sigma={self.sigma})"


def _from_rs(prior: _rs.Prior | None) -> Prior | None:
    """A Python prior object wrapping an extension prior."""
    if prior is None:
        return None
    params = list(prior.parameters)
    kind = prior.kind
    obj: Prior
    if kind == "LogNormal":
        obj = LogNormalPrior.__new__(LogNormalPrior)
        obj.loc, obj.scale = params
    elif kind == "Gamma":
        obj = GammaPrior.__new__(GammaPrior)
        obj.concentration, obj.rate = params
    elif kind == "Normal":
        obj = NormalPrior.__new__(NormalPrior)
        obj.loc, obj.scale = params
    elif kind == "Uniform":
        obj = UniformPrior.__new__(UniformPrior)
        obj.low, obj.high = params
    elif kind == "HalfCauchy":
        obj = HalfCauchyPrior.__new__(HalfCauchyPrior)
        (obj.scale,) = params
    elif kind == "HalfNormal":
        obj = HalfNormalPrior.__new__(HalfNormalPrior)
        (obj.scale,) = params
    elif kind == "SmoothedBox":
        obj = SmoothedBoxPrior.__new__(SmoothedBoxPrior)
        obj.a, obj.b, obj.sigma = params
    else:
        from botorch_rs.models.utils.priors import BetaPrior

        obj = BetaPrior.__new__(BetaPrior)
        obj._a, obj._b = params
        obj._kind = kind_of(0.0)
        obj._batch_shape = ()
        obj._validate_args = False
        obj._transform = None
    obj._rs = prior
    return obj


__all__ = [
    "GammaPrior",
    "HalfCauchyPrior",
    "HalfNormalPrior",
    "LKJCholeskyFactorPrior",
    "LKJCovariancePrior",
    "LKJPrior",
    "LogNormalPrior",
    "NormalPrior",
    "Prior",
    "SmoothedBoxPrior",
    "UniformPrior",
]

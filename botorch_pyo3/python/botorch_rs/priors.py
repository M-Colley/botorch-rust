"""Hyperparameter priors (``gpytorch.priors``).

Like GPyTorch, which stores prior parameters given as Python floats in
float32, the parameters are rounded to single precision (so that log
probabilities match BoTorch's models to double precision).
"""

from __future__ import annotations

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
        return from_flat([self._rs.log_prob(v) for v in data], shape, kind_of(x))

    @property
    def mode(self) -> float:
        """The mode of the distribution."""
        return self._rs.mode


def _check_transform(transform: Any) -> None:
    if transform is not None:
        raise UnsupportedError("Prior transforms are not supported by botorch_rs.")


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
        self._rs = _rs.Prior.log_normal(self.loc, self.scale)

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
        self._rs = _rs.Prior.gamma(self.concentration, self.rate)

    def __repr__(self) -> str:
        return f"GammaPrior(concentration={self.concentration}, rate={self.rate})"


def _from_rs(prior: _rs.Prior | None) -> Prior | None:
    """A Python prior object wrapping an extension prior."""
    if prior is None:
        return None
    a, b = prior.parameters
    obj: Prior
    if prior.kind == "LogNormal":
        obj = LogNormalPrior.__new__(LogNormalPrior)
        obj.loc, obj.scale = a, b
    else:
        obj = GammaPrior.__new__(GammaPrior)
        obj.concentration, obj.rate = a, b
    obj._rs = prior
    return obj


__all__ = ["GammaPrior", "LogNormalPrior", "Prior"]

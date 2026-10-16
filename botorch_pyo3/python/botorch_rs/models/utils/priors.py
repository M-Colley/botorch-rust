"""BoTorch's priors (``botorch.models.utils.priors``): ``BetaPrior``."""

from __future__ import annotations

from typing import Any

from botorch_rs import _botorch_rs as _rs
from botorch_rs._arrays import as_float
from botorch_rs.priors import Prior, _check_transform, _float32


class BetaPrior(Prior):
    """Beta prior with density ``x**(alpha - 1) (1 - x)**(beta - 1) / B(alpha, beta)``
    on ``[0, 1]``, parameterized by ``concentration1`` (``alpha``) and
    ``concentration0`` (``beta``); a prior on task correlations (e.g. of a
    ``PositiveIndexKernel``)."""

    def __init__(
        self,
        concentration1: float,
        concentration0: float,
        validate_args: bool = False,
        transform: Any = None,
    ) -> None:
        _check_transform(transform)
        self.concentration1 = as_float(concentration1, "concentration1")
        self.concentration0 = as_float(concentration0, "concentration0")
        self._rs = _rs.Prior.beta(self.concentration1, self.concentration0, _float32())

    def __repr__(self) -> str:
        return (
            f"BetaPrior(concentration1={self.concentration1}, "
            f"concentration0={self.concentration0})"
        )


__all__ = ["BetaPrior"]

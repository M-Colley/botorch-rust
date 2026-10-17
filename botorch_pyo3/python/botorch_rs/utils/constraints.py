"""Parameter constraints of BoTorch (``botorch.utils.constraints``):
``LogTransformedInterval`` and ``NonTransformedInterval``."""

from __future__ import annotations

import sys
from typing import Any

from botorch_rs import _botorch_rs as _rs
from botorch_rs._arrays import as_float
from botorch_rs.constraints import Interval
from botorch_rs.priors import _float32


def _torch_log32(value: float) -> float | None:
    """torch's float32 ``log(value)`` (as GPyTorch computes the raw initial
    value of a constraint created from Python floats), or ``None`` without
    torch or in double-precision mode."""
    torch = sys.modules.get("torch")
    if torch is None:
        return None
    if not _float32():
        return None
    return float(torch.log(torch.tensor(value, dtype=torch.float32)).item())


class NonTransformedInterval(Interval):
    """Modification of the GPyTorch interval class that does not apply
    transformations: the bounds are passed to the optimizer (L-BFGS-B), so
    that parameters can reach the bounds exactly.
    """

    def __init__(
        self,
        lower_bound: Any,
        upper_bound: Any,
        initial_value: Any = None,
    ) -> None:
        """Constructor of the NonTransformedInterval class.

        Args:
            lower_bound: The lower bound of the interval.
            upper_bound: The upper bound of the interval.
            initial_value: The initial value of the parameter.
        """
        super().__init__(
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            transform=None,
            inv_transform=None,
            initial_value=initial_value,
        )


class LogTransformedInterval(Interval):
    """Modification of the GPyTorch interval class.

    The Interval class in GPyTorch will map the parameter to the range [0, 1] before
    applying the inverse transform. LogTransformedInterval skips this step to avoid
    numerical issues, and applies the log transform directly to the parameter values.
    GPyTorch automatically recognizes that the bound constraint have not been applied
    yet, and passes the bounds to the optimizer instead, which then optimizes
    log(parameter) under the constraints log(lower) <= log(parameter) <= log(upper).
    """

    def __init__(
        self,
        lower_bound: float,
        upper_bound: float,
        initial_value: float | None = None,
    ) -> None:
        """Constructor of the LogTransformedInterval class.

        Args:
            lower_bound: The lower bound of the interval.
            upper_bound: The upper bound of the interval.
            initial_value: The initial value of the parameter.
        """
        init = None if initial_value is None else as_float(initial_value, "initial_value")
        f32 = _float32()
        raw = _torch_log32(init) if init is not None and f32 else None
        try:
            self._rs = _rs.Constraint.log_transformed_interval(
                as_float(lower_bound, "lower_bound"),
                as_float(upper_bound, "upper_bound"),
                init,
                f32,
                raw,
            )
        except ValueError as e:
            raise ValueError(str(e)) from None

    @property
    def initial_value_untransformed(self) -> float | None:
        """The initial value in the constrained space."""
        return self._rs.initial_value


__all__ = ["LogTransformedInterval", "NonTransformedInterval"]

"""Parameter constraints (``gpytorch.constraints``).

``transform`` is GPyTorch's softplus (the default) or ``None``: with ``None``
the constraint is not enforced through a transform and the bound is instead
imposed by the optimizer, as in BoTorch's default models.
"""

from __future__ import annotations

from typing import Any

from botorch_rs import _botorch_rs as _rs
from botorch_rs._arrays import as_float
from botorch_rs.exceptions.errors import UnsupportedError

#: Marker for GPyTorch's default transform (``torch.nn.functional.softplus``).
softplus = "softplus"


def _transform_name(transform: Any) -> str | None:
    if transform is None:
        return None
    if transform == softplus or getattr(transform, "__name__", None) == "softplus":
        return "softplus"
    raise UnsupportedError(
        f"Only the softplus transform (or None) is supported, got {transform!r}."
    )


class Interval:
    """Base class of the constraints."""

    _rs: _rs.GreaterThan

    @property
    def lower_bound(self) -> float:
        return self._rs.lower_bound

    @property
    def upper_bound(self) -> float:
        return float("inf")

    @property
    def enforced(self) -> bool:
        """Whether the constraint is enforced through a transform."""
        return self._rs.enforced

    @property
    def initial_value(self) -> float | None:
        return self._rs.initial_value

    def transform(self, raw: float) -> float:
        """Maps a raw value to the constrained value."""
        return self._rs.transform(as_float(raw, "raw"))

    def inverse_transform(self, value: float) -> float:
        """Maps a constrained value to the raw value."""
        return self._rs.inverse_transform(as_float(value, "value"))

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.lower_bound:.3e})"


class GreaterThan(Interval):
    """``value >= lower_bound``."""

    def __init__(
        self,
        lower_bound: Any,
        transform: Any = softplus,
        inv_transform: Any = None,
        initial_value: Any = None,
    ) -> None:
        init = None if initial_value is None else as_float(initial_value, "initial_value")
        self._rs = _rs.GreaterThan(
            as_float(lower_bound, "lower_bound"), _transform_name(transform), init
        )


class Positive(GreaterThan):
    """``value >= 0`` (enforced with softplus by default)."""

    def __init__(
        self,
        transform: Any = softplus,
        inv_transform: Any = None,
        initial_value: Any = None,
    ) -> None:
        super().__init__(0.0, transform, inv_transform, initial_value)


def _from_rs(constraint: _rs.GreaterThan) -> GreaterThan:
    obj = GreaterThan.__new__(GreaterThan)
    obj._rs = constraint
    return obj


__all__ = ["GreaterThan", "Interval", "Positive", "softplus"]

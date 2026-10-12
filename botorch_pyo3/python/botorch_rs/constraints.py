"""Parameter constraints (``gpytorch.constraints``).

``GreaterThan`` / ``Positive`` (lower bounds) and ``Interval`` (lower and
upper bounds). An enforced constraint maps an unconstrained raw parameter
through a transform: GPyTorch's ``softplus`` for ``GreaterThan``
(``softplus(raw) + lower_bound``) and ``sigmoid`` for ``Interval``
(``sigmoid(raw) * (upper - lower) + lower``). With ``transform=None`` the
constraint is not enforced: the raw value is the value, and the optimizer
keeps it within the bounds (L-BFGS-B box bounds in ``fit_gpytorch_mll``), as
in BoTorch's default models and in the task kernel of ``LCEMGP``.

Like GPyTorch, bounds given as Python floats are stored in single precision,
and the raw initial value (``initial_value``) is computed from them in single
precision.
"""

from __future__ import annotations

from typing import Any

from botorch_rs import _botorch_rs as _rs
from botorch_rs._arrays import as_float
from botorch_rs.exceptions.errors import UnsupportedError

#: Marker for GPyTorch's ``softplus`` transform (``torch.nn.functional.softplus``).
softplus = "softplus"
#: Marker for GPyTorch's ``sigmoid`` transform (``torch.sigmoid``).
sigmoid = "sigmoid"
#: Marker for GPyTorch's ``inv_softplus`` (``gpytorch.utils.transforms``).
inv_softplus = "inv_softplus"
#: Marker for GPyTorch's ``inv_sigmoid`` (``gpytorch.utils.transforms``).
inv_sigmoid = "inv_sigmoid"

_TRANSFORMS = {"softplus": "softplus", "Softplus": "softplus", "sigmoid": "sigmoid"}
_INVERSES = {"inv_softplus": "softplus", "inv_sigmoid": "sigmoid"}


def _name(fn: Any) -> str | None:
    if isinstance(fn, str):
        return fn
    return getattr(fn, "__name__", None) or type(fn).__name__


def _transform_name(transform: Any) -> str | None:
    """``"softplus"``, ``"sigmoid"`` or ``None`` for a GPyTorch transform."""
    if transform is None:
        return None
    name = _TRANSFORMS.get(_name(transform) or "")
    if name is None:
        raise UnsupportedError(
            f"Only the softplus and sigmoid transforms (or None) are supported, got {transform!r}."
        )
    return name


def _inverse_name(inv_transform: Any) -> str | None:
    """The transform whose inverse ``inv_transform`` is (``None`` for ``None``)."""
    if inv_transform is None:
        return None
    name = _INVERSES.get(_name(inv_transform) or "")
    if name is None:
        raise UnsupportedError(
            "Only the inv_softplus and inv_sigmoid inverse transforms are supported, got "
            f"{inv_transform!r}."
        )
    return name


class Interval:
    """``lower_bound <= value <= upper_bound``
    (``gpytorch.constraints.Interval``).

    With the default ``sigmoid`` transform the raw parameter is unbounded;
    with ``transform=None`` the bounds are box bounds of the optimizer.
    ``initial_value`` is in the constrained space; like GPyTorch, a kernel or
    likelihood created with this constraint starts at it.
    """

    _rs: _rs.Constraint

    def __init__(
        self,
        lower_bound: Any,
        upper_bound: Any,
        transform: Any = sigmoid,
        inv_transform: Any = inv_sigmoid,
        initial_value: Any = None,
    ) -> None:
        t = _transform_name(transform)
        # GPyTorch derives the inverse from the transform only if
        # ``inv_transform=None`` is passed explicitly.
        inv = _inverse_name(inv_transform) if inv_transform is not None else t
        init = None if initial_value is None else as_float(initial_value, "initial_value")
        try:
            self._rs = _rs.Constraint.interval(
                as_float(lower_bound, "lower_bound"),
                as_float(upper_bound, "upper_bound"),
                t,
                inv,
                init,
            )
        except ValueError as e:  # GPyTorch raises ValueError for invalid bounds
            raise ValueError(str(e)) from None

    @property
    def lower_bound(self) -> float:
        return self._rs.lower_bound

    @property
    def upper_bound(self) -> float:
        return self._rs.upper_bound

    @property
    def enforced(self) -> bool:
        """Whether the constraint is enforced through a transform."""
        return self._rs.enforced

    @property
    def initial_value(self) -> float | None:
        """The raw initial value (GPyTorch's ``constraint.initial_value``:
        the inverse transform of the given ``initial_value``), or ``None``."""
        return self._rs.initial_raw_value

    def transform(self, raw: Any) -> float:
        """Maps a raw value to the constrained value."""
        return self._rs.transform(as_float(raw, "raw"))

    def inverse_transform(self, value: Any) -> float:
        """Maps a constrained value to the raw value."""
        return self._rs.inverse_transform(as_float(value, "value"))

    def check(self, value: Any) -> bool:
        """Whether ``value`` satisfies the bounds."""
        v = as_float(value, "value")
        return self.lower_bound <= v <= self.upper_bound

    def check_raw(self, raw: Any) -> bool:
        """Whether the transform of ``raw`` satisfies the bounds."""
        return self.check(self.transform(raw))

    def __iter__(self) -> Any:
        yield self.lower_bound
        yield self.upper_bound

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.lower_bound:.3E}, {self.upper_bound:.3E})"


class GreaterThan(Interval):
    """``value >= lower_bound``."""

    def __init__(
        self,
        lower_bound: Any,
        transform: Any = softplus,
        inv_transform: Any = None,
        initial_value: Any = None,
    ) -> None:
        name = _transform_name(transform)
        if name == "sigmoid":
            raise UnsupportedError(
                "GreaterThan supports the softplus transform (or None) in botorch_rs."
            )
        init = None if initial_value is None else as_float(initial_value, "initial_value")
        self._rs = _rs.Constraint.greater_than(as_float(lower_bound, "lower_bound"), name, init)

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.lower_bound:.3E})"


class Positive(GreaterThan):
    """``value >= 0`` (enforced with softplus by default)."""

    def __init__(
        self,
        transform: Any = softplus,
        inv_transform: Any = None,
        initial_value: Any = None,
    ) -> None:
        super().__init__(0.0, transform, inv_transform, initial_value)

    def __repr__(self) -> str:
        return "Positive()"


def _from_rs(constraint: _rs.Constraint) -> Interval:
    """A Python constraint object wrapping an extension constraint."""
    cls: type[Interval] = GreaterThan if constraint.kind == "GreaterThan" else Interval
    obj = cls.__new__(cls)
    obj._rs = constraint
    return obj


__all__ = [
    "GreaterThan",
    "Interval",
    "Positive",
    "inv_sigmoid",
    "inv_softplus",
    "sigmoid",
    "softplus",
]

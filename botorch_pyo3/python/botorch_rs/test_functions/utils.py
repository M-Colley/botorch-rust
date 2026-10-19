"""Utilities of the test functions (``botorch.test_functions.utils``)."""

from __future__ import annotations

from typing import Any

from botorch_rs import _botorch_rs as _rs
from botorch_rs._arrays import from_flat, kind_of, to_flat


def round_nearest(X: Any, increment: float, bounds: tuple[float, float] | None) -> Any:
    r"""Rounds the input tensor to the nearest multiple of ``increment``.

    Args:
        X: The input to be rounded.
        increment: The increment to round to.
        bounds: An optional tuple of two floats representing the lower and upper
            bounds on ``X``. If provided, this will round to the nearest multiple
            of ``increment`` that lies within the bounds.

    Returns:
        The rounded input.

    Values are rounded half to even (``torch.round``).
    """
    data, shape = to_flat(X, "X")
    out = _rs.round_nearest(
        data,
        float(increment),
        None if bounds is None else (float(bounds[0]), float(bounds[1])),
    )
    return from_flat(out, shape, kind_of(X))


__all__ = ["round_nearest"]

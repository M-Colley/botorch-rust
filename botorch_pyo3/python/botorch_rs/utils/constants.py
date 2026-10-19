"""Scalar constants (``botorch.utils.constants``)."""

from __future__ import annotations

from collections.abc import Iterator
from functools import cache
from numbers import Number
from typing import Any

from botorch_rs._arrays import ArrayKind, from_flat, kind_from_dtype, kind_of


@cache
def _cached(values: Any, kind: ArrayKind) -> Any:
    if isinstance(values, Number):
        return from_flat([float(values)], (), kind)  # type: ignore[arg-type]
    return tuple(from_flat([float(v)], (), kind) for v in values)


def get_constants(
    values: Number | Iterator[Number],
    device: Any = None,
    dtype: Any = None,
) -> Any:
    r"""Returns scalar-valued Tensors containing each of the given constants.
    Used to expedite tensor operations involving scalar arithmetic. Note that
    the returned Tensors should not be modified in-place.

    0-dimensional torch tensors for a torch ``dtype`` (or if torch has been
    imported), NumPy scalars for a NumPy dtype, else floats; cached per
    values, dtype and device (``values`` must be hashable, e.g. a number or
    a tuple).
    """
    kind = kind_from_dtype(dtype, device)
    key: Any = values if isinstance(values, Number) else tuple(values)
    return _cached(key, kind)


def get_constants_like(
    values: Number | Iterator[Number],
    ref: Any,
) -> Any:
    """``get_constants`` with the dtype and device of the array ``ref``."""
    kind = kind_of(ref)
    key: Any = values if isinstance(values, Number) else tuple(values)
    return _cached(key, kind)


__all__ = ["get_constants", "get_constants_like"]

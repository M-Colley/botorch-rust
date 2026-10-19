"""Bound layout helpers for independent batched L-BFGS-B problems."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any


def translate_bounds_for_lbfgsb(
    lower_bounds: Any, upper_bounds: Any, num_features: int, q: int
) -> list[tuple[Any, Any]]:
    """Expand scalar/per-feature bounds to the flattened ``q x d`` layout."""
    expanded = []
    for bounds in (lower_bounds, upper_bounds):
        if bounds is None:
            values = [None] * num_features
        elif isinstance(bounds, Iterable):
            values = list(bounds)
        else:
            values = [bounds] * num_features
        if len(values) == num_features:
            values *= q
        if len(values) != num_features * q:
            raise AssertionError(f"Instead got {len(values)} != {num_features} * {q}.")
        expanded.append(values)
    return list(zip(*expanded, strict=True))


__all__ = ["translate_bounds_for_lbfgsb"]

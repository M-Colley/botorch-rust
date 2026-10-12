"""Pareto dominance (``botorch.utils.multi_objective.pareto``).

``is_non_dominated`` is computed in Rust with the loop algorithm that BoTorch
uses for large inputs; it returns BoTorch's mask for every input size,
including which of several identical points is kept (the first). Points with
a NaN in any outcome are dominated and do not affect other points, as
documented; BoTorch's main branch only masks out points whose outcomes are
*all* NaN (fixed in meta-pytorch/botorch#3392).
"""

from __future__ import annotations

from typing import Any

from botorch_rs import _botorch_rs as _rs
from botorch_rs._arrays import bool_array, kind_of, to_flat

#: BoTorch's memory limit of its all-pairs comparison (not used here: the
#: mask is the same for every input size).
MAX_BYTES = 5e6


def is_non_dominated(Y: Any, maximize: bool = True, deduplicate: bool = True) -> Any:
    r"""Computes the non-dominated front.

    Args:
        Y: A ``(batch_shape) x n x m``-dim array of outcomes. Points with a NaN
            in any outcome are treated as dominated.
        maximize: If True, assume maximization (default).
        deduplicate: A boolean indicating whether to only return unique
            points on the Pareto frontier (the first of several identical
            points).

    Returns:
        A ``(batch_shape) x n``-dim boolean array (a bool tensor, a NumPy bool
        array or nested lists of bools) indicating whether each point is
        non-dominated.
    """
    data, shape = to_flat(Y, "Y")
    if len(shape) < 2:
        raise IndexError("tuple index out of range")
    mask, out_shape = _rs.is_non_dominated((data, shape), bool(maximize), bool(deduplicate))
    return bool_array(mask, out_shape, kind_of(Y))


__all__ = ["MAX_BYTES", "is_non_dominated"]

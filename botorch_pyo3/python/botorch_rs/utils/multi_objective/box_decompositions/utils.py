"""Utilities for box decomposition algorithms
(``botorch.utils.multi_objective.box_decompositions.utils``): local upper
bounds (Lacour et al., 2017, Algorithm 1), cell bounds (Equation 2), the
direct two-objective decompositions and the padding of batched Pareto
frontiers. All are computed in Rust.
"""

from __future__ import annotations

from typing import Any

from botorch_rs import _botorch_rs as _rs
from botorch_rs._arrays import first_kind, from_pair, to_flat


def _expand_ref_point(ref_point: Any, batch_shape: Any) -> Any:
    r"""Expand reference point to the proper batch_shape.

    Args:
        ref_point: A ``(batch_shape) x m``-dim array containing the reference
            point.
        batch_shape: The batch shape.

    Returns:
        A ``batch_shape x m``-dim array containing the expanded reference
        point.
    """
    out = _rs.expand_ref_point(to_flat(ref_point, "ref_point"), [int(b) for b in batch_shape])
    return from_pair(out, first_kind(ref_point))


def _pad_batch_pareto_frontier(
    Y: Any,
    ref_point: Any,
    is_pareto: bool = False,
    feasibility_mask: Any = None,
) -> Any:
    r"""Get a batch Pareto frontier by padding the pareto frontier with
    repeated points.

    This assumes maximization.

    Args:
        Y: A ``(batch_shape) x n x m``-dim array of points
        ref_point: a ``(batch_shape) x m``-dim array containing the reference
            point
        is_pareto: a boolean indicating whether the points in Y are already
            non-dominated.
        feasibility_mask: A ``(batch_shape) x n``-dim array of booleans
            indicating whether each point is feasible.

    Returns:
        A ``(batch_shape) x max_num_pareto x m``-dim array of padded Pareto
        frontiers.
    """
    mask = None
    if feasibility_mask is not None:
        mask = [bool(v) for v in to_flat(feasibility_mask, "feasibility_mask")[0]]
    out = _rs.pad_batch_pareto_frontier(
        to_flat(Y, "Y"), to_flat(ref_point, "ref_point"), bool(is_pareto), mask
    )
    return from_pair(out, first_kind(Y, ref_point))


def compute_local_upper_bounds(U: Any, Z: Any, z: Any) -> tuple[Any, Any]:
    r"""Compute local upper bounds.

    Note: this assumes minimization.

    This uses the incremental algorithm (Alg. 1) from Lacour et al. (2017).

    Args:
        U: A ``n x m``-dim array containing the local upper bounds.
        Z: A ``n x m x m``-dim array containing the defining points.
        z: A ``m``-dim array containing the new point.

    Returns:
        2-element tuple containing:

        - A new ``n' x m``-dim array local upper bounds.
        - A ``n' x m x m``-dim array containing the defining points.
    """
    u, zz = _rs.compute_local_upper_bounds(to_flat(U, "U"), to_flat(Z, "Z"), to_flat(z, "z")[0])
    kind = first_kind(U, Z, z)
    return from_pair(u, kind), from_pair(zz, kind)


def get_partition_bounds(Z: Any, U: Any, ref_point: Any) -> Any:
    r"""Get the cell bounds given the local upper bounds and the defining
    points.

    This implements Equation 2 in Lacour et al. (2017).

    Args:
        Z: A ``n x m x m``-dim array containing the defining points. The first
            dimension corresponds to u_idx, the second dimension corresponds
            to j, and Z[u_idx, j] is the set of definining points Z^j(u) where
            u = U[u_idx].
        U: A ``n x m``-dim array containing the local upper bounds.
        ref_point: A ``m``-dim array containing the reference point.

    Returns:
        A ``2 x num_cells x m``-dim array containing the lower and upper
        vertices bounding each hypercell.
    """
    out = _rs.get_partition_bounds(
        to_flat(Z, "Z"), to_flat(U, "U"), to_flat(ref_point, "ref_point")[0]
    )
    return from_pair(out, first_kind(U, Z, ref_point))


def update_local_upper_bounds_incremental(new_pareto_Y: Any, U: Any, Z: Any) -> tuple[Any, Any]:
    r"""Update the current local upper with the new pareto points.

    This assumes minimization.

    Args:
        new_pareto_Y: A ``n x m``-dim array containing the new Pareto points.
        U: A ``n' x m``-dim array containing the local upper bounds.
        Z: A ``n x m x m``-dim array containing the defining points.

    Returns:
        2-element tuple containing:

        - A new ``n' x m``-dim array local upper bounds.
        - A ``n' x m x m``-dim array containing the defining points
    """
    u, zz = _rs.update_local_upper_bounds_incremental(
        to_flat(new_pareto_Y, "new_pareto_Y"), to_flat(U, "U"), to_flat(Z, "Z")
    )
    kind = first_kind(U, Z, new_pareto_Y)
    return from_pair(u, kind), from_pair(zz, kind)


def compute_non_dominated_hypercell_bounds_2d(pareto_Y_sorted: Any, ref_point: Any) -> Any:
    r"""Compute an axis-aligned partitioning of the non-dominated space for 2
    objectives.

    Args:
        pareto_Y_sorted: A ``(batch_shape) x n_pareto x 2``-dim array of
            pareto outcomes that are sorted by the 0th dimension in
            increasing order. All points must be better than the reference
            point.
        ref_point: A ``(batch_shape) x 2``-dim reference point.

    Returns:
        A ``2 x (batch_shape) x n_pareto + 1 x m``-dim array of cell bounds.
    """
    out = _rs.compute_non_dominated_hypercell_bounds_2d(
        to_flat(pareto_Y_sorted, "pareto_Y_sorted"), to_flat(ref_point, "ref_point")
    )
    return from_pair(out, first_kind(pareto_Y_sorted, ref_point))


def compute_dominated_hypercell_bounds_2d(pareto_Y_sorted: Any, ref_point: Any) -> Any:
    r"""Compute an axis-aligned partitioning of the dominated space for
    2-objectives.

    Args:
        pareto_Y_sorted: A ``(batch_shape) x n_pareto x 2``-dim array of
            pareto outcomes that are sorted by the 0th dimension in
            increasing order.
        ref_point: A ``2``-dim reference point.

    Returns:
        A ``2 x (batch_shape) x n_pareto x m``-dim array of cell bounds.
    """
    out = _rs.compute_dominated_hypercell_bounds_2d(
        to_flat(pareto_Y_sorted, "pareto_Y_sorted"), to_flat(ref_point, "ref_point")
    )
    return from_pair(out, first_kind(pareto_Y_sorted, ref_point))


__all__ = [
    "_expand_ref_point",
    "_pad_batch_pareto_frontier",
    "compute_dominated_hypercell_bounds_2d",
    "compute_local_upper_bounds",
    "compute_non_dominated_hypercell_bounds_2d",
    "get_partition_bounds",
    "update_local_upper_bounds_incremental",
]

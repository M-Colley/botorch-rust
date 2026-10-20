"""Device-resident exact box geometry for differentiable hypervolume.

The general case ports Lacour's local upper bound algorithm and the two-pass
non-dominated decomposition used by BoTorch's FastNondominatedPartitioning.
All boundary values and their graphs remain in Torch; scalar reads are limited
to integer sizes used to compact padded search zones.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor


def nondominated_mask(Y: Tensor, ref: Tensor, feasible: Tensor | None = None) -> Tensor:
    """Maximization Pareto membership, retaining the first feasible duplicate."""
    reference = ref.unsqueeze(-2) if ref.ndim == Y.ndim - 1 else ref
    valid = (Y > reference).all(dim=-1)
    if feasible is not None:
        valid = valid & feasible
    a, b = Y.unsqueeze(-3), Y.unsqueeze(-2)
    dominates = (a >= b).all(-1) & (a > b).any(-1)
    duplicate = (a == b).all(-1)
    indices = torch.arange(Y.shape[-2], device=Y.device)
    duplicate = duplicate & (indices.unsqueeze(-2) < indices.unsqueeze(-1))
    return valid & ~((dominates | duplicate) & valid.unsqueeze(-2)).any(dim=-1)


def _compact(U: Tensor, Z: Tensor, active: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    # A ragged batch has one device-side mask per search zone. Read only the
    # integer output width, never an outcome/covariance/acquisition value.
    width = max(int(active.sum(-1).amax().item()), 1)
    indices = active.to(torch.int8).argsort(dim=-1, descending=True, stable=True)[..., :width]
    m = U.shape[-1]
    U = U.gather(-2, indices.unsqueeze(-1).expand(*indices.shape, m))
    Z = Z.gather(-3, indices.unsqueeze(-1).unsqueeze(-1).expand(*indices.shape, m, m))
    return U, Z, active.gather(-1, indices)


def _local_upper_bounds(
    U: Tensor, Z: Tensor, active: Tensor, point: Tensor, valid: Tensor
) -> tuple[Tensor, Tensor, Tensor]:
    """One vectorized Lacour insertion across all sample/model/input batches."""
    m = U.shape[-1]
    dominates = (U > point.unsqueeze(-2)).all(-1) & active & valid.unsqueeze(-1)
    upper, defining, masks = [U], [Z], [active & ~dominates]
    for j in range(m):
        others = [r for r in range(m) if r != j]
        maximum = Z[..., others, j].amax(dim=-1)
        include = dominates & (point[..., j].unsqueeze(-1) >= maximum)
        coordinate = point[..., j].unsqueeze(-1).unsqueeze(-1).expand(*U.shape[:-1], 1)
        candidate = torch.cat((U[..., :j], coordinate, U[..., j + 1 :]), -1)
        rows = [point.unsqueeze(-2).expand_as(U) if r == j else Z[..., r, :] for r in range(m)]
        upper.append(candidate)
        defining.append(torch.stack(rows, -2))
        masks.append(include)
    return _compact(torch.cat(upper, -2), torch.cat(defining, -3), torch.cat(masks, -1))


def partition_nd(
    Y: Tensor, ref: Tensor, feasible: Tensor | None = None
) -> tuple[Tensor, Tensor, Tensor]:
    """Exact non-dominated boxes for m>=3, padded and masked across all batches."""
    if Y.shape[-1] < 3:
        raise ValueError("partition_nd requires at least three objectives.")
    batch, n, m = Y.shape[:-2], Y.shape[-2], Y.shape[-1]
    batch_count = math.prod(batch)
    points = Y.reshape(batch_count, n, m)
    reference = ref.expand(*batch, m).reshape(batch_count, m)
    valid = nondominated_mask(
        points, reference, None if feasible is None else feasible.reshape(batch_count, n)
    )
    width = int(valid.sum(-1).amax().item())
    order = valid.to(torch.int8).argsort(dim=-1, descending=True, stable=True)[..., :width]
    points = points.gather(-2, order.unsqueeze(-1).expand(*order.shape, m))
    valid = valid.gather(-1, order)
    diagonal = torch.eye(m, dtype=torch.bool, device=Y.device).reshape(1, 1, m, m)
    U = -reference.unsqueeze(-2)
    Z = torch.where(diagonal, U.unsqueeze(-2), -torch.inf)
    active = torch.ones((batch_count, 1), dtype=torch.bool, device=Y.device)
    for index in range(width):
        U, Z, active = _local_upper_bounds(U, Z, active, -points[:, index], valid[:, index])
    # The negative local upper bounds are a Pareto front for a second
    # minimization problem with ideal=ref and reference=(+inf,...,+inf).
    second_points, second_valid = -U, active
    U = torch.full((batch_count, 1, m), torch.inf, dtype=Y.dtype, device=Y.device)
    Z = torch.where(diagonal, U.unsqueeze(-2), reference[:, None, None, :])
    active = torch.ones((batch_count, 1), dtype=torch.bool, device=Y.device)
    for index in range(second_points.shape[-2]):
        U, Z, active = _local_upper_bounds(
            U, Z, active, second_points[:, index], second_valid[:, index]
        )
    lower_coordinates = [Z[..., 0, 0]]
    upper_coordinates = [torch.full_like(U[..., 0], torch.inf)]
    for j in range(1, m):
        lower_coordinates.append(Z[..., :j, j].amax(-1))
        upper_coordinates.append(U[..., j])
    lower = torch.stack(lower_coordinates, -1)
    upper = torch.stack(upper_coordinates, -1)
    active = active & (upper > lower).all(-1)
    # Inactive padded cells must have finite harmless values before nonlinear
    # smoothing, preventing unselected infinities from contaminating gradients.
    lower = torch.where(active.unsqueeze(-1), lower, reference.unsqueeze(-2))
    upper = torch.where(active.unsqueeze(-1), upper, reference.unsqueeze(-2) + 1)
    cells = lower.shape[-2]
    return (
        lower.reshape(*batch, cells, m),
        upper.reshape(*batch, cells, m),
        active.reshape(*batch, cells),
    )

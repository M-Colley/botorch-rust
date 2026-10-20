"""Tensor optimization utility algebra with device-resident data and graphs."""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

from botorch_rs.exceptions.errors import BotorchError


def columnwise_clamp(
    X: Tensor, lower: Any = None, upper: Any = None, raise_on_violation: bool = False
) -> Tensor:
    if lower is None and upper is None:
        return X
    lo = None if lower is None else torch.as_tensor(lower).expand_as(X).to(X)
    hi = None if upper is None else torch.as_tensor(upper).expand_as(X).to(X)
    if lo is not None and hi is not None and bool((lo > hi).any()):
        raise ValueError("Lower bounds cannot exceed upper bounds.")
    result = X.clamp(lo, hi)
    if raise_on_violation and not X.allclose(result):
        raise BotorchError("Original values are out of bounds.")
    return result


def evaluate_feasibility(
    X: Tensor,
    inequality: Any = None,
    equality: Any = None,
    nonlinear: Any = None,
    tolerance: float | None = None,
) -> Tensor:
    from botorch_rs.optim.parameter_constraints import get_constraint_tolerance

    if X.ndim < 2:
        raise ValueError("X must be at least two-dimensional (batch_shape x q x d).")
    tolerance = get_constraint_tolerance(X.dtype) if tolerance is None else tolerance
    feasible = torch.ones(X.shape[:-2], dtype=torch.bool, device=X.device)
    for collection, is_equality in ((inequality, False), (equality, True)):
        for index, coefficients, rhs in collection or []:
            indices = torch.as_tensor(index, device=X.device)
            if indices.dtype not in (
                torch.int8,
                torch.int16,
                torch.int32,
                torch.int64,
                torch.uint8,
            ):
                raise ValueError("Constraint indices must be integer tensors.")
            coefficients = torch.as_tensor(coefficients, dtype=X.dtype, device=X.device)
            rhs = torch.as_tensor(rhs, dtype=X.dtype, device=X.device).reshape(())
            if indices.ndim == 1:
                residual = (X[..., indices.long()] * coefficients).sum(-1) - rhs
            elif indices.ndim == 2 and indices.shape[-1] == 2:
                residual = (X[..., indices[:, 0].long(), indices[:, 1].long()] * coefficients).sum(
                    -1
                ) - rhs
            else:
                raise ValueError("Constraint indices must have shape k or k x 2.")
            satisfied = residual.abs() <= tolerance if is_equality else residual >= -tolerance
            feasible &= satisfied.all(-1) if indices.ndim == 1 else satisfied
    for function, intrapoint in nonlinear or []:
        feasible &= nonlinear_is_feasible(function, intrapoint, X, tolerance)
    return feasible


def nonlinear_is_feasible(
    function: Any, intrapoint: bool, X: Tensor, tolerance: float | None = None
) -> Tensor:
    from botorch_rs.optim.parameter_constraints import get_constraint_tolerance

    if X.ndim < 2:
        raise ValueError("X must have shape batch_shape x q x d.")
    tolerance = get_constraint_tolerance(X.dtype) if tolerance is None else tolerance
    points = X.reshape(-1, X.shape[-1]) if intrapoint else X.reshape(-1, *X.shape[-2:])
    if points.shape[0] == 0:
        return torch.ones(X.shape[:-2], dtype=torch.bool, device=X.device)
    values = torch.stack(
        [
            torch.as_tensor(function(point), dtype=X.dtype, device=X.device).reshape(())
            for point in points
        ]
    )
    if intrapoint:
        return (values >= -tolerance).reshape(*X.shape[:-2], X.shape[-2]).all(-1)
    return (values >= -tolerance).reshape(X.shape[:-2])

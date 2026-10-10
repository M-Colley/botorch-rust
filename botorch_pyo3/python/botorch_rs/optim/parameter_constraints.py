"""Parameter constraints of the acquisition-function optimization
(``botorch.optim.parameter_constraints``).

Linear constraints use BoTorch's format: a tuple ``(indices, coefficients,
rhs)`` encodes ``sum_i X[indices[i]] * coefficients[i] >= rhs`` (inequality)
or ``= rhs`` (equality). With a one-dimensional ``indices`` the constraint
applies to every point of a q-batch (intra-point); with a two-dimensional
``indices`` of ``(point, feature)`` rows it couples the points of the q-batch
(inter-point). Nonlinear inequality constraints are tuples ``(callable,
is_intrapoint)`` with ``callable(x) >= 0`` on torch tensors (``x`` of shape
``d`` for intra-point, ``q x d`` for inter-point constraints); their gradients
come from torch.autograd.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from botorch_rs import _botorch_rs as _rs
from botorch_rs._arrays import as_float, bool_array, kind_of, to_flat

#: Constraint tolerance for float64 candidates (BoTorch's
#: ``get_constraint_tolerance(torch.double)``).
CONSTRAINT_TOLERANCE = 1e-8

#: A linear constraint as passed to the extension.
LinearArg = tuple[Any, list[float], float]


def linear_constraints_arg(constraints: Any, name: str) -> list[LinearArg] | None:
    """BoTorch's linear constraints as ``(indices, coefficients, rhs)`` with
    integer (nested) lists of indices and float coefficients."""
    if constraints is None:
        return None
    if not isinstance(constraints, Sequence):
        raise ValueError(f"`{name}` must be a list of (indices, coefficients, rhs) tuples.")
    out: list[LinearArg] = []
    for constraint in constraints:
        try:
            indices, coefficients, rhs = constraint
        except (TypeError, ValueError) as e:
            raise ValueError(
                f"`{name}` must be a list of (indices, coefficients, rhs) tuples."
            ) from e
        idata, ishape = to_flat(indices, "indices")
        ints = [int(v) for v in idata]
        if any(i != v for i, v in zip(ints, idata, strict=True)):
            raise ValueError("Constraint indices must be integers.")
        idx: Any
        if len(ishape) == 1:
            idx = ints
        elif len(ishape) == 2:
            c = ishape[1]
            idx = [ints[r * c : (r + 1) * c] for r in range(ishape[0])]
        else:
            raise ValueError(
                "Linear constraints supported only on individual candidates and across "
                "q-batches, not across general batch shapes."
            )
        cdata, _ = to_flat(coefficients, "coefficients")
        out.append((idx, [float(v) for v in cdata], as_float(rhs, "rhs")))
    return out


def nonlinear_constraints_arg(constraints: Any) -> list[tuple[Callable[..., Any], bool]] | None:
    """BoTorch's nonlinear inequality constraints ``(callable, is_intrapoint)``
    (validated with BoTorch's messages)."""
    if constraints is None:
        return None
    out = []
    for constraint in constraints:
        if not isinstance(constraint, tuple):
            raise ValueError(f"A nonlinear constraint has to be a tuple, got {type(constraint)}.")
        if len(constraint) != 2:
            raise ValueError(
                "A nonlinear constraint has to be a tuple of length 2, "
                f"got length {len(constraint)}."
            )
        nlc, is_intrapoint = constraint
        if not callable(nlc):
            raise ValueError(f"A nonlinear constraint must be callable, got {type(nlc)}.")
        out.append((nlc, bool(is_intrapoint)))
    if out:
        from botorch_rs._torch_bridge import require_torch

        require_torch()
    return out


def evaluate_feasibility(
    X: Any,
    inequality_constraints: list[tuple[Any, Any, float]] | None = None,
    equality_constraints: list[tuple[Any, Any, float]] | None = None,
    nonlinear_inequality_constraints: list[tuple[Callable[..., Any], bool]] | None = None,
    tolerance: float | None = None,
) -> Any:
    r"""Whether each candidate set of ``X`` (``batch x q x d``) satisfies the
    constraints within ``tolerance`` (default ``1e-8``): ``const(X) >= rhs -
    tolerance`` for inequality, ``|const(X) - rhs| < tolerance`` for equality
    and ``const(X) >= -tolerance`` for nonlinear constraints.

    Returns:
        A boolean array of shape ``batch`` (a bool tensor for torch inputs).
    """
    flat = to_flat(X, "X")
    if len(flat[1]) < 2:
        raise ValueError(f"X must be at least 2-dimensional (`batch x q x d`), got {flat[1]}.")
    feasible, batch = _rs.evaluate_feasibility(
        flat,
        linear_constraints_arg(inequality_constraints, "inequality_constraints"),
        linear_constraints_arg(equality_constraints, "equality_constraints"),
        nonlinear_constraints_arg(nonlinear_inequality_constraints),
        tolerance,
    )
    return bool_array(feasible, batch, kind_of(X))


__all__ = ["CONSTRAINT_TOLERANCE", "evaluate_feasibility"]

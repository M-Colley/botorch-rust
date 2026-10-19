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

import importlib
import math
import sys
from collections.abc import Callable, Sequence
from typing import Any

from botorch_rs import _botorch_rs as _rs
from botorch_rs import _ops
from botorch_rs._arrays import as_float, bool_array, from_flat, kind_of, to_flat
from botorch_rs.exceptions.errors import UnsupportedError

#: Constraint tolerance for float64 candidates (BoTorch's
#: ``get_constraint_tolerance(torch.double)``).
CONSTRAINT_TOLERANCE = 1e-8

#: A linear constraint as passed to the extension.
LinearArg = tuple[Any, list[float], float]


def get_constraint_tolerance(dtype: Any) -> float:
    """Return the float64/float32/float16 constraint tolerance."""
    name = "float64" if dtype is float else getattr(dtype, "__name__", str(dtype))
    name = name.replace("torch.", "")
    if name in {"float64", "float32", "float16"}:
        return {"float64": 1e-8, "float32": 1e-6, "float16": 1e-4}[name]
    raise ValueError(f"Unsupported dtype {dtype}.")


def _input_tolerance(X: Any) -> float:
    dtype = _ops.dtype_name(X)
    return CONSTRAINT_TOLERANCE if dtype is None else get_constraint_tolerance(dtype)


def make_scipy_bounds(X: Any, lower_bounds: Any = None, upper_bounds: Any = None) -> Any:
    """Expand column bounds into SciPy's flattened ``Bounds`` representation."""
    if lower_bounds is None and upper_bounds is None:
        return None
    np = importlib.import_module("numpy")
    Bounds = importlib.import_module("scipy.optimize").Bounds
    from botorch_rs.optim.utils.numpy_utils import as_ndarray

    shape = _ops.shape_of(X)

    def expand(value: Any, lower: bool) -> Any:
        if value is None:
            return np.full(math.prod(shape), -np.inf if lower else np.inf)
        if kind_of(X).lib == "torch" and kind_of(value).lib == "list":
            value = sys.modules["torch"].as_tensor(value)
        return np.broadcast_to(as_ndarray(value), shape).reshape(-1).copy()

    return Bounds(lb=expand(lower_bounds, True), ub=expand(upper_bounds, False), keep_feasible=True)


def _constraint_shape(shapeX: Any) -> tuple[int, int, int]:
    shape = tuple(int(value) for value in shapeX)
    if len(shape) not in (2, 3):
        raise UnsupportedError(
            f"`shapeX` must be `(b) x q x d` (at least two-dimensional). It is {shape}."
        )
    if len(shape) == 2:
        shape = (1, *shape)
    return shape[0], shape[1], shape[2]


def make_scipy_linear_constraints(
    shapeX: Any,
    inequality_constraints: Any = None,
    equality_constraints: Any = None,
) -> list[Any]:
    """Build current BoTorch's sparse inequality and equality constraint blocks.

    Returns at most two ``scipy.optimize.LinearConstraint`` objects. This
    follows upstream main; BoTorch 0.18.1 instead returns per-row dictionaries.
    """
    if not inequality_constraints and not equality_constraints:
        return []
    np = importlib.import_module("numpy")
    sparse = importlib.import_module("scipy.sparse")
    LinearConstraint = importlib.import_module("scipy.optimize").LinearConstraint
    b, q, d = _constraint_shape(shapeX)
    output = []
    for constraints, equality, name in (
        (inequality_constraints, False, "inequality_constraints"),
        (equality_constraints, True, "equality_constraints"),
    ):
        rows: list[int] = []
        columns: list[int] = []
        values: list[float] = []
        lower: list[float] = []
        upper: list[float] = []
        for indices, coefficients, rhs in linear_constraints_arg(constraints, name) or []:
            inter = bool(indices and isinstance(indices[0], list))
            if len(coefficients) != len(indices):
                raise ValueError("Constraint indices and coefficients must have the same length.")
            if inter:
                if any(len(index) != 2 for index in indices):
                    raise ValueError("Inter-point indices must have shape (n, 2).")
                if any(not 0 <= point < q for point, _ in indices):
                    raise RuntimeError(f"Index out of bounds for {q}-batch")
                if any(not 0 <= feature < d for _, feature in indices):
                    raise RuntimeError(f"Index out of bounds for {d}-dim parameter tensor")
                offsets = [
                    [batch * q * d + point * d + feature for point, feature in indices]
                    for batch in range(b)
                ]
            else:
                if any(not 0 <= feature < d for feature in indices):
                    raise RuntimeError(f"Index out of bounds for {d}-dim parameter tensor")
                offsets = [
                    [(batch * q + point) * d + feature for feature in indices]
                    for batch in range(b)
                    for point in range(q)
                ]
            for indices_flat in offsets:
                rows.extend([len(lower)] * len(coefficients))
                columns.extend(indices_flat)
                values.extend(coefficients)
                lower.append(rhs)
                upper.append(rhs if equality else np.inf)
        if lower:
            matrix = sparse.coo_array(
                (
                    np.asarray(values, dtype=np.float64),
                    (np.asarray(rows, dtype=np.int64), np.asarray(columns, dtype=np.int64)),
                ),
                shape=(len(lower), b * q * d),
            )
            output.append(LinearConstraint(matrix, np.asarray(lower), np.asarray(upper)))
    return output


def _reshape(X: Any, shape: tuple[int, ...]) -> Any:
    if kind_of(X).lib != "list":
        return X.reshape(shape)
    return from_flat(to_flat(X)[0], shape, kind_of(X))


def nonlinear_constraint_is_feasible(
    nonlinear_inequality_constraint: Callable[..., Any],
    is_intrapoint: bool,
    x: Any,
    tolerance: float | None = None,
) -> Any:
    """Evaluate a nonlinear constraint for every point or complete q-batch."""
    shape = _ops.shape_of(x)
    if len(shape) < 2:
        raise ValueError("x must have shape batch_shape x q x d.")
    tolerance = _input_tolerance(x) if tolerance is None else tolerance
    batches = math.prod(shape[:-2])
    flat = _reshape(x, (batches, *shape[-2:]))
    result = []
    for batch in range(batches):
        points = _ops.getitem(flat, batch)
        if is_intrapoint:
            feasible = all(
                as_float(nonlinear_inequality_constraint(_ops.getitem(points, point)), "constraint")
                >= -tolerance
                for point in range(shape[-2])
            )
        else:
            feasible = as_float(nonlinear_inequality_constraint(points), "constraint") >= -tolerance
        result.append(feasible)
    return bool_array(result, shape[:-2], kind_of(x))


def make_scipy_nonlinear_inequality_constraints(
    nonlinear_inequality_constraints: Any,
    f_np_wrapper: Callable[..., Any],
    x0: Any,
    shapeX: Any,
    validate_feasibility: bool = True,
) -> list[dict[str, Any]]:
    """Create scalar SciPy constraint callbacks with a shared value/gradient cache."""
    np = importlib.import_module("numpy")
    shape = _constraint_shape(shapeX)
    initial = _reshape(x0, shape)
    output = []

    def callbacks(function: Callable[..., Any]) -> tuple[Callable[..., Any], Callable[..., Any]]:
        cache: dict[str, Any] = {}

        def evaluate(x: Any) -> tuple[Any, Any]:
            if "x" not in cache or not np.array_equal(cache["x"], x):
                value, grad = f_np_wrapper(x, f=function)
                cache.update(x=np.array(x, copy=True), value=value, grad=grad)
            return cache["value"], cache["grad"]

        return lambda x: evaluate(x)[0], lambda x: evaluate(x)[1]

    def selected(function: Callable[..., Any], batch: int, point: int | None) -> Callable[..., Any]:
        def evaluate(x: Any) -> Any:
            return function(
                _ops.getitem(_reshape(x, shape), batch if point is None else (batch, point))
            )

        return evaluate

    for constraint in nonlinear_inequality_constraints:
        if not isinstance(constraint, tuple):
            raise ValueError(f"A nonlinear constraint has to be a tuple, got {type(constraint)}.")
        if len(constraint) != 2:
            raise ValueError(
                f"A nonlinear constraint has to be a tuple of length 2, got length {len(constraint)}."
            )
        function, intra = constraint
        if not callable(function):
            raise ValueError("A nonlinear constraint must be callable.")
        if validate_feasibility:
            feasible = nonlinear_constraint_is_feasible(function, bool(intra), initial)
            if not all(to_flat(feasible)[0]):
                raise ValueError(
                    "`batch_initial_conditions` must satisfy the non-linear inequality constraints."
                )
        for batch in range(shape[0]):
            for point in range(shape[1]) if intra else [None]:
                objective, gradient = callbacks(selected(function, batch, point))
                output.append({"type": "ineq", "fun": objective, "jac": gradient})
    return output


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
    constraints within ``tolerance`` (default based on the input dtype): ``const(X) >= rhs -
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
        _input_tolerance(X) if tolerance is None else tolerance,
    )
    return bool_array(feasible, batch, kind_of(X))


__all__ = [
    "CONSTRAINT_TOLERANCE",
    "evaluate_feasibility",
    "get_constraint_tolerance",
    "make_scipy_bounds",
    "make_scipy_linear_constraints",
    "make_scipy_nonlinear_inequality_constraints",
    "nonlinear_constraint_is_feasible",
]

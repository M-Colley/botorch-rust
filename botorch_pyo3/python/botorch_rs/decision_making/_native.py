"""Native FPBI adapters for lists and NumPy arrays, without Torch/SciPy."""

from __future__ import annotations

import importlib
import itertools
import math
import random
from typing import Any

from botorch_rs import _botorch_rs as _rs
from botorch_rs._arrays import bool_array, from_flat, kind_of, to_flat

from ._types import FPBIGeometry, FPBIResult, SolverDiagnostic


def _positive_int(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer.")
    return value


def _indices(values: list[int], shape: tuple[int, ...], kind: Any) -> Any:
    from . import _reshape_indices

    if kind.lib == "numpy":
        np = importlib.import_module("numpy")
        return np.asarray(values, dtype=np.int64).reshape(shape)
    return _reshape_indices(values, shape)


def _geometry(records, batch, m, kind, n_steps, maximize, focus):
    def field(name, tail, matrix=False):
        values = []
        for record in records:
            part = record[name][0] if matrix else record[name]
            values.extend(part)
        return from_flat(values, (*batch, *tail), kind)

    return FPBIGeometry(
        utopia=field("utopia", (m,)),
        nadir=field("nadir", (m,)),
        normalized_extremes=field("normalized_extremes", (m, m), True),
        active_objectives=bool_array(
            [v > 0 for r in records for v in r["ranges"]], (*batch, m), kind
        ),
        farthest_pair=_indices([v for r in records for v in r["farthest_pair"]], (*batch, 2), kind),
        center=field("center", (m,)),
        focus=from_flat(focus, (*batch, m), kind),
        direction=field("search_direction", (m,)),
        basis=field("basis", (m, m - 1), True),
        anchors=field("grid", (n_steps ** (m - 1), m), True),
        degenerate=bool_array([not any(v > 0 for v in r["ranges"]) for r in records], batch, kind),
        used_fallback_basis=bool_array(
            [r["used_degenerate_fallback"] for r in records], batch, kind
        ),
        n_steps=n_steps,
        maximize=maximize,
    )


def fpbi_geometry(
    extreme_Y: Any,
    *,
    n_steps: int = 10,
    focus: Any = None,
    maximize: bool = True,
    max_rays: int = 10000,
) -> FPBIGeometry:
    """Construct equations 16--26 from individual objective extremes.

    ``extreme_Y`` has shape ``(..., m, m)`` and ``focus`` is a shared length-m
    point in normalized cost space (default zero), broadcasting across the
    extreme batches. The grid is guarded by
    ``max_rays``. List/NumPy calculations run in native double precision.
    """
    _positive_int(n_steps, "n_steps")
    _positive_int(max_rays, "max_rays")
    if not isinstance(maximize, bool):
        raise TypeError("maximize must be a bool.")
    values, shape = to_flat(extreme_Y, "extreme_Y")
    if len(shape) < 2 or shape[-2] != shape[-1] or shape[-1] < 2 or not math.prod(shape[:-2]):
        raise ValueError("extreme_Y must have nonempty shape (..., m, m), m >= 2.")
    m, batch = shape[-1], shape[:-2]
    if n_steps ** (m - 1) > max_rays:
        raise ValueError("FPBI grid exceeds max_rays.")
    f, fs = ([0.0] * m, (m,)) if focus is None else to_flat(focus, "focus")
    if not fs or fs[-1] != m or len(fs) - 1 > len(batch):
        raise ValueError("focus must broadcast to (...,m) in normalized cost coordinates.")
    padded = (1,) * (len(batch) - len(fs) + 1) + tuple(fs[:-1])
    if any(a not in (1, b) for a, b in zip(padded, batch, strict=True)):
        raise ValueError("focus must broadcast to (...,m) in normalized cost coordinates.")
    focus_rows = []
    for coords in itertools.product(*(range(b) for b in batch)):
        row = 0
        for coordinate, size in zip(coords, padded, strict=True):
            row = row * size + (0 if size == 1 else coordinate)
        focus_rows.append(list(f[row * m : (row + 1) * m]))
    records = [
        _rs.decision_fpbi_geometry(
            (values[b * m * m : (b + 1) * m * m], (m, m)),
            n_steps,
            focus_rows[b],
            maximize,
            max_rays,
        )
        for b in range(math.prod(batch))
    ]
    return _geometry(
        records,
        batch,
        m,
        kind_of(extreme_Y),
        n_steps,
        maximize,
        [v for row in focus_rows for v in row],
    )


def sample_fpbi(
    objective,
    bounds: Any,
    *,
    n_steps: int = 10,
    focus: Any = None,
    maximize: bool = True,
    num_restarts: int = 4,
    seed: int = 0,
    maxiter: int = 300,
    feasibility_tol: float = 1e-6,
    initial_X: Any = None,
    inequality_constraints=None,
    equality_constraints=None,
    max_rays: int = 10000,
) -> FPBIResult:
    """Generate a continuous FPBI front with native finite-difference SLSQP.

    Objective and constraint callbacks receive a single-row ``(1,d)`` input
    of the bounds' array kind. Objectives return ``(1,m)``. Inequalities are
    nonnegative and equalities zero. This is a multistart local solver; its
    predictions and extrema are not certified global optima.
    """
    for value, name in (
        (n_steps, "n_steps"),
        (num_restarts, "num_restarts"),
        (maxiter, "maxiter"),
        (max_rays, "max_rays"),
    ):
        _positive_int(value, name)
    if not isinstance(maximize, bool):
        raise TypeError("maximize must be a bool.")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a nonnegative integer.")
    if (
        isinstance(feasibility_tol, bool)
        or not math.isfinite(feasibility_tol)
        or feasibility_tol <= 0
    ):
        raise ValueError("feasibility_tol must be finite and positive.")
    values, shape = to_flat(bounds, "bounds")
    if len(shape) != 2 or shape[0] != 2 or not shape[1]:
        raise ValueError("bounds must have nonempty shape (2,d).")
    d, kind = shape[1], kind_of(bounds)
    lower, upper = list(values[:d]), list(values[d:])
    if any(not math.isfinite(v) for v in values) or any(
        a >= b for a, b in zip(lower, upper, strict=True)
    ):
        raise ValueError("bounds must be finite and have upper > lower.")
    if any(not math.isfinite(b - a) for a, b in zip(lower, upper, strict=True)):
        raise ValueError("bound widths must be finite; rescale input coordinates.")
    rng = random.Random(seed)
    starts = [a / 2 + b / 2 for a, b in zip(lower, upper, strict=True)]
    for _ in range(num_restarts - 1):
        starts.extend(rng.uniform(a, b) for a, b in zip(lower, upper, strict=True))
    count = num_restarts
    if initial_X is not None:
        initial, initial_shape = to_flat(initial_X, "initial_X")
        if len(initial_shape) != 2 or initial_shape[1] != d or not initial_shape[0]:
            raise ValueError("initial_X must have nonempty shape (n,d).")
        if any(
            not math.isfinite(v) or not lower[j % d] <= v <= upper[j % d]
            for j, v in enumerate(initial)
        ):
            raise ValueError("initial_X must be finite and lie inside bounds.")
        starts = list(initial) + starts
        count += initial_shape[0]
    f = None
    if focus is not None:
        f, fs = to_flat(focus, "focus")
        if len(fs) != 1:
            raise ValueError("focus must have shape (m,).")
        f = list(f)

    def callback(fn, *, outcomes=False):
        if fn is None:
            return None

        def evaluate(x):
            result, result_shape = to_flat(fn(from_flat(x, (1, d), kind)), "callback result")
            if outcomes and (len(result_shape) != 2 or result_shape[0] != 1 or result_shape[1] < 2):
                raise ValueError("objective must return shape (1,m) with m >= 2.")
            return list(result)

        return evaluate

    step, stopping = 1e-6, min(feasibility_tol / 10, 1e-9)
    if kind.lib == "numpy":
        np = importlib.import_module("numpy")
        dtype = kind.dtype or np.float64
        if np.dtype(dtype).itemsize < 4:
            raise TypeError("SLSQP sampling requires float32 or float64 bounds.")
        eps = float(np.finfo(dtype).eps)
        step, stopping = max(step, math.sqrt(eps)), max(stopping, 2 * eps)
    raw = _rs.decision_fpbi_front(
        callback(objective, outcomes=True),
        (values, shape),
        (starts, (count, d)),
        n_steps,
        f,
        maximize,
        maxiter,
        stopping,
        callback(inequality_constraints),
        callback(equality_constraints),
        feasibility_tol,
        step,
        max_rays,
    )
    m = raw["Y"][1][-1]
    focus_values = [0.0] * m if f is None else f
    geometry = _geometry([raw["geometry"]], (), m, kind, n_steps, maximize, focus_values)
    diagnostics = tuple(
        SolverDiagnostic(
            "extreme" if s["extreme"] else "ray",
            s["index"],
            -1,
            s["mode"] == 0,
            s["success"] and s["feasible"],
            s["mode"],
            s["message"],
            s["max_violation"],
            s["kappa"],
        )
        for s in raw["statuses"]
    )
    ray_success = [False] * (n_steps ** (m - 1))
    for s in raw["statuses"]:
        if not s["extreme"] and s["success"] and s["feasible"]:
            ray_success[s["index"]] = True
    # Flat axes pad the plane with zero columns, so some grid anchors repeat.
    # Native solves each unique anchor once; share its status with duplicates.
    anchor_values = raw["geometry"]["grid"][0]
    first_anchor: dict[tuple[float, ...], int] = {}
    for i in range(len(ray_success)):
        anchor = tuple(anchor_values[i * m : (i + 1) * m])
        if anchor in first_anchor:
            ray_success[i] = ray_success[first_anchor[anchor]]
        else:
            first_anchor[anchor] = i
    return FPBIResult(
        X=from_flat(raw["X"][0], raw["X"][1], kind),
        Y=from_flat(raw["Y"][0], raw["Y"][1], kind),
        extreme_X=from_flat(raw["extreme_X"][0], raw["extreme_X"][1], kind),
        extreme_Y=from_flat(raw["extreme_Y"][0], raw["extreme_Y"][1], kind),
        geometry=geometry,
        diagnostics=diagnostics,
        ray_succeeded=bool_array(ray_success, (len(ray_success),), kind),
    )

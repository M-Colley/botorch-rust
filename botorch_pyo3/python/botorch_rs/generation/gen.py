"""Candidate generation utilities (``botorch.generation.gen``).

``gen_candidates_scipy`` and ``gen_candidates_torch`` optimize botorch_rs
acquisition functions natively (independent L-BFGS-B runs, BoTorch's serial
L-BFGS-B mode, SLSQP with constraints, and Adam with BoTorch's
exponential-moving-average stopping criterion).
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Callable, Mapping
from typing import Any

from botorch_rs import _botorch_rs as _rs
from botorch_rs._arrays import ArrayKind, first_kind, from_flat, kind_of, to_flat
from botorch_rs.exceptions.errors import UnsupportedError
from botorch_rs.exceptions.warnings import OptimizationWarning
from botorch_rs.optim.parameter_constraints import (
    linear_constraints_arg,
    nonlinear_constraints_arg,
)


def _bounds_arg(
    lower_bounds: Any, upper_bounds: Any, d: int
) -> tuple[list[float], tuple[int, int]]:
    """``2 x d`` bounds from scalars, ``d`` arrays or ``None`` (infinite)."""

    def values(b: Any, default: float) -> list[float]:
        if b is None:
            return [default] * d
        data = to_flat(b, "bounds")[0]
        if len(data) == 1:
            return [float(data[0])] * d
        if len(data) != d:
            raise ValueError(f"Bounds must have 1 or {d} values, got {len(data)}.")
        return [float(v) for v in data]

    return values(lower_bounds, -math.inf) + values(upper_bounds, math.inf), (2, d)


def _fixed_arg(fixed_features: Mapping[int, Any] | None, d: int) -> dict[int, float] | None:
    if not fixed_features:
        return None
    out: dict[int, float] = {}
    for k, v in fixed_features.items():
        data, shape = to_flat(v, f"fixed_features[{k}]")
        if shape:
            raise UnsupportedError(
                "botorch_rs supports scalar fixed features in gen_candidates (no values per "
                "t-batch)."
            )
        out[int(k) % d] = float(data[0])
    return out


def _values_out(values: list[float], ics_shape: tuple[int, ...], kind: Any) -> Any:
    if len(ics_shape) == 2:
        return from_flat(values[:1], (1,), kind)
    return from_flat(values, tuple(ics_shape[:-2]), kind)


def gen_candidates_scipy(
    initial_conditions: Any,
    acquisition_function: Any,
    lower_bounds: Any = None,
    upper_bounds: Any = None,
    inequality_constraints: list[tuple[Any, Any, float]] | None = None,
    equality_constraints: list[tuple[Any, Any, float]] | None = None,
    nonlinear_inequality_constraints: list[tuple[Callable[..., Any], bool]] | None = None,
    options: dict[str, Any] | None = None,
    fixed_features: Mapping[int, Any] | None = None,
    timeout_sec: float | None = None,
    use_parallel_mode: bool | None = None,
) -> tuple[Any, Any]:
    r"""Generate a set of candidates with L-BFGS-B or SLSQP (BoTorch's
    ``gen_candidates_scipy``), natively.

    Without constraints and with the options of BoTorch's parallel path,
    every initial condition is optimized by an independent L-BFGS-B run
    (``use_parallel_mode``); otherwise (``timeout_sec``, other options,
    ``use_parallel_mode=False``) the initial conditions are optimized as one
    problem (or groups of ``max_optimization_problem_aggregation_size``).
    With constraints, SLSQP optimizes all initial conditions as one problem.

    Args:
        initial_conditions: Starting points, shape ``(b) x q x d``.
        acquisition_function: A botorch_rs acquisition function.
        lower_bounds: Minimum values for each column of initial_conditions.
        upper_bounds: Maximum values for each column of initial_conditions.
        inequality_constraints: Linear inequality constraints.
        equality_constraints: Linear equality constraints.
        nonlinear_inequality_constraints: Nonlinear inequality constraints.
        options: Options of the optimizer (``method``, ``maxiter``, ...).
        fixed_features: Features fixed to scalar values.
        timeout_sec: Timeout in seconds.
        use_parallel_mode: Force (True) or forbid (False) the parallel mode.

    Returns:
        The generated candidates (the shape of ``initial_conditions``) and
        the acquisition value of each t-batch.
    """
    from botorch_rs.optim.optimize import _as_native, _parse_options

    kind = first_kind(initial_conditions)
    ics = to_flat(initial_conditions, "initial_conditions")
    shape = tuple(ics[1])
    if use_parallel_mode and len(shape) not in (2, 3):
        raise UnsupportedError(
            f"Initial conditions must have either 2 or 3 dimensions. Received shape {shape}"
        )
    d = shape[-1]
    native = _as_native(acquisition_function, d)
    ineq = linear_constraints_arg(inequality_constraints, "inequality_constraints")
    eq = linear_constraints_arg(equality_constraints, "equality_constraints")
    nonlinear = nonlinear_constraints_arg(nonlinear_inequality_constraints)
    constrained = bool(ineq or eq or nonlinear)
    opts = dict(options or {})
    opts.setdefault("maxiter", 2000)
    kwargs, _, _ = _parse_options(
        opts,
        constrained=constrained,
        timeout_sec=timeout_sec,
        torch_gen=False,
        warn_unknown=True,
    )
    fast = "lbfgsb_aggregation" not in kwargs and not constrained
    if use_parallel_mode and not fast:
        raise NotImplementedError(
            "Not using the parallel implementation of l-bfgs-b, as: constraints, a timeout or "
            "options outside the parallel path are given"
        )
    aggregation = kwargs.pop("lbfgsb_aggregation", None)
    if use_parallel_mode is False and aggregation is None and not constrained:
        aggregation = 2**62
        kwargs["maxfun"] = kwargs.get("maxfun", 15000)
    method = kwargs.pop("method", None)
    timeout = kwargs.pop("timeout_sec", timeout_sec)
    ics3 = ics if len(shape) == 3 else (ics[0], (1, *shape))
    candidates, values, failures = _rs.gen_candidates_scipy_py(
        native._rs,
        ics3,
        _bounds_arg(lower_bounds, upper_bounds, d),
        fixed_features=_fixed_arg(fixed_features, d),
        aggregation=aggregation,
        use_slsqp=method == "SLSQP" or constrained,
        timeout_sec=None if timeout is None else float(timeout),
        inequality_constraints=ineq,
        equality_constraints=eq,
        nonlinear_inequality_constraints=nonlinear,
        **kwargs,
    )
    for message in failures:
        warnings.warn(
            f"Optimization failed within `scipy.optimize.minimize` with status 2 and message "
            f"{message}.",
            OptimizationWarning,
            stacklevel=2,
        )
    native._after_evaluation()
    X = from_flat(candidates[0], shape, kind)
    return X, _values_out(list(values), shape, kind)


def gen_candidates_torch(
    initial_conditions: Any,
    acquisition_function: Any,
    lower_bounds: Any = None,
    upper_bounds: Any = None,
    optimizer: Any = None,
    options: dict[str, Any] | None = None,
    callback: Callable[[int, Any, Any], Any] | None = None,
    fixed_features: Mapping[int, Any] | None = None,
    timeout_sec: float | None = None,
) -> tuple[Any, Any]:
    r"""Generate a set of candidates using Adam (BoTorch's
    ``gen_candidates_torch`` with its default ``torch.optim.Adam``),
    natively.

    Every iteration evaluates the negated sum of the acquisition values of
    the clamped candidates and its gradient, takes an Adam step (learning
    rate 0.025 unless ``options["optimizer_options"]["lr"]``) and checks
    BoTorch's ``ExpMAStoppingCriterion(**options["stopping_criterion_options"])``
    on the loss; all t-batches stop together. With fixed features, the free
    features are clamped to their own bounds (BoTorch fails to broadcast the
    bounds).

    Args:
        initial_conditions: Starting points, shape ``(b) x q x d``.
        acquisition_function: A botorch_rs acquisition function.
        lower_bounds: Minimum values for each column of initial_conditions.
        upper_bounds: Maximum values for each column of initial_conditions.
        optimizer: ``None`` or ``torch.optim.Adam`` (native); other torch
            optimizers run BoTorch's loop with torch.
        options: ``optimizer_options``, ``stopping_criterion_options`` and
            the deprecated ``maxiter``.
        callback: Called after every evaluation with the iteration, the loss
            and the gradient (torch tensors).
        fixed_features: Features fixed to scalar values.
        timeout_sec: Timeout in seconds.

    Returns:
        The generated candidates (the shape of ``initial_conditions``) and
        the acquisition value of each t-batch.
    """
    from botorch_rs.optim.optimize import _as_native, _torch_options

    kind = first_kind(initial_conditions)
    ics = to_flat(initial_conditions, "initial_conditions")
    shape = tuple(ics[1])
    d = shape[-1]
    native = _as_native(acquisition_function, d)
    if optimizer is not None and getattr(optimizer, "__name__", None) != "Adam":
        return _gen_candidates_torch_loop(
            initial_conditions,
            native,
            lower_bounds,
            upper_bounds,
            optimizer,
            options,
            callback,
            fixed_features,
            timeout_sec,
        )
    torch_kwargs = _torch_options(dict(options or {}), {})
    py_callback = None
    if callback is not None:
        import torch

        def py_callback(i: int, loss: float, data: Any, gshape: list[int]) -> None:
            grad = from_flat(data, tuple(gshape), ArrayKind("torch", torch.float64))
            callback(i, torch.tensor(loss, dtype=torch.float64), grad)

    ics3 = ics if len(shape) == 3 else (ics[0], (1, *shape))
    candidates, values, _ = _rs.gen_candidates_torch_py(
        native._rs,
        ics3,
        _bounds_arg(lower_bounds, upper_bounds, d),
        fixed_features=_fixed_arg(fixed_features, d),
        callback=py_callback,
        timeout_sec=None if timeout_sec is None else float(timeout_sec),
        **torch_kwargs,
    )
    native._after_evaluation()
    X = from_flat(candidates[0], shape, kind)
    return X, _values_out(list(values), shape, kind)


def _gen_candidates_torch_loop(
    initial_conditions: Any,
    acquisition_function: Any,
    lower_bounds: Any,
    upper_bounds: Any,
    optimizer: Any,
    options: dict[str, Any] | None,
    callback: Any,
    fixed_features: Mapping[int, Any] | None,
    timeout_sec: float | None,
) -> tuple[Any, Any]:
    """BoTorch's ``gen_candidates_torch`` loop with a torch optimizer (on the
    natively computed values and gradients)."""
    import time

    import torch

    from botorch_rs.optim.stopping import ExpMAStoppingCriterion
    from botorch_rs.optim.utils.acquisition_utils import fix_features

    start_time = time.monotonic()
    options = options or {}
    X0 = torch.as_tensor(initial_conditions, dtype=torch.float64)
    d = X0.shape[-1]
    free = [i for i in range(d) if not fixed_features or i not in fixed_features]

    def clamp(X: Any) -> Any:
        lo = None if lower_bounds is None else torch.as_tensor(lower_bounds, dtype=X.dtype)
        hi = None if upper_bounds is None else torch.as_tensor(upper_bounds, dtype=X.dtype)
        if lo is not None and lo.ndim > 0 and lo.numel() == d and X.shape[-1] != d:
            lo = lo[free]
        if hi is not None and hi.ndim > 0 and hi.numel() == d and X.shape[-1] != d:
            hi = hi[free]
        return X.clamp(lo, hi) if lo is not None or hi is not None else X

    params = clamp(X0)[..., free].clone().requires_grad_(True)
    optimizer_options = dict(options.get("optimizer_options", {}) or {})
    stopping_options = dict(options.get("stopping_criterion_options", {}) or {})
    if "maxiter" in options:
        warnings.warn(
            "Passing 'maxiter' directly in options is deprecated. "
            "Please use options['stopping_criterion_options']['maxiter'] instead.",
            DeprecationWarning,
            stacklevel=3,
        )
        stopping_options.setdefault("maxiter", options["maxiter"])
    optimizer_options.setdefault("lr", 0.025)
    opt = optimizer(params=[params], **optimizer_options)
    criterion = ExpMAStoppingCriterion(**stopping_options)
    i, stop = 0, False
    while not stop:
        i += 1
        with torch.no_grad():
            X = clamp(params).requires_grad_(True)
        loss = -acquisition_function(
            fix_features(X, fixed_features, replace_current_value=False)
        ).sum()
        grad = torch.autograd.grad(loss, X)[0]
        if callback:
            callback(i, loss, grad)

        def assign_grad(grad: Any = grad, loss: Any = loss) -> Any:
            opt.zero_grad()
            params.grad = grad
            return loss

        opt.step(assign_grad)
        stop = criterion(loss.detach())
        if timeout_sec is not None and time.monotonic() - start_time > timeout_sec:
            stop = True
    with torch.no_grad():
        final = fix_features(clamp(params), fixed_features, replace_current_value=False)
        values = acquisition_function(final)
    kind = kind_of(initial_conditions)
    return from_flat(final.reshape(-1).tolist(), tuple(final.shape), kind), from_flat(
        values.reshape(-1).tolist(), tuple(values.shape), kind
    )


def get_best_candidates(batch_candidates: Any, batch_values: Any) -> Any:
    r"""Extract the best (q-batch) candidate from a batch of candidates
    (BoTorch's ``get_best_candidates``; ``torch.argmax`` semantics).

    Args:
        batch_candidates: A ``b x q x d`` array of ``b`` q-batch candidates, or
            a ``b x d`` array of ``b`` single-point candidates.
        batch_values: An array with ``b`` elements containing the value of the
            respective candidate (higher is better).

    Returns:
        A ``q x d`` (or ``d``) array of the candidates with the highest value.
    """
    values = [float(v) for v in to_flat(batch_values, "batch_values")[0]]
    best = next((i for i, v in enumerate(values) if v != v), None)
    if best is None:
        best = max(range(len(values)), key=lambda i: (values[i], -i))
    data, shape = to_flat(batch_candidates, "batch_candidates")
    per = math.prod(shape[1:])
    return from_flat(
        data[best * per : (best + 1) * per], tuple(shape[1:]), kind_of(batch_candidates)
    )


__all__ = ["gen_candidates_scipy", "gen_candidates_torch", "get_best_candidates"]

"""Core abstractions and generic optimizers (``botorch.optim.core``).

:class:`OptimizationResult` and :class:`OptimizationStatus` are BoTorch's.
:func:`scipy_minimize` and :func:`torch_minimize` are BoTorch's generic
drivers for closures on torch tensors (they require SciPy / torch); the
native fitting routines of :mod:`botorch_rs.optim.fit` run the same loops in
Rust for botorch_rs models.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from enum import Enum, auto
from itertools import count
from sys import maxsize
from time import monotonic
from typing import Any

_LBFGSB_MAXITER_MAXFUN_REGEX = re.compile(  # regex for maxiter and maxfun messages
    "TOTAL NO. (of|OF) " + "(ITERATIONS REACHED LIMIT|(f AND g|F,G) EVALUATIONS EXCEEDS LIMIT)"
)


class OptimizationStatus(int, Enum):
    """The status of an optimization (BoTorch's ``OptimizationStatus``)."""

    RUNNING = auto()  # incomplete
    SUCCESS = auto()  # optimizer converged
    FAILURE = auto()  # terminated abnormally
    STOPPED = auto()  # stopped due to user provided criterion


@dataclass(slots=True)
class OptimizationResult:
    """The summary of an optimization (BoTorch's ``OptimizationResult``)."""

    step: int
    fval: float | int
    status: OptimizationStatus
    runtime: float | None = None
    message: str | None = None


def _result(out: tuple[int, float, int, float | None, str | None]) -> OptimizationResult:
    """An :class:`OptimizationResult` from the extension's tuple."""
    step, fval, status, runtime, message = out
    return OptimizationResult(
        step=int(step),
        fval=float(fval),
        status=OptimizationStatus(int(status)),
        runtime=runtime,
        message=message,
    )


def scipy_minimize(
    closure: Any,
    parameters: dict[str, Any],
    bounds: dict[str, tuple[float | None, float | None]] | None = None,
    callback: Callable[[dict[str, Any], OptimizationResult], None] | None = None,
    x0: Any = None,
    method: str = "L-BFGS-B",
    options: dict[str, Any] | None = None,
    timeout_sec: float | None = None,
) -> OptimizationResult:
    r"""Generic ``scipy.optimize.minimize``-based optimization routine
    (BoTorch's ``scipy_minimize``) for a closure on torch tensors or an
    :class:`~botorch_rs.optim.closures.NdarrayOptimizationClosure`.

    Args:
        closure: Callable that returns a tensor and an iterable of gradient
            tensors, or an ``NdarrayOptimizationClosure``.
        parameters: A dictionary of tensors to be optimized.
        bounds: A dictionary mapping parameter names to lower and upper bounds.
        callback: A callable taking ``parameters`` and an OptimizationResult.
        x0: An optional initialization vector.
        method: Solver type, passed along to ``scipy.optimize.minimize``.
        options: Dictionary of solver options.
        timeout_sec: Timeout in seconds after which the optimization stops
            (returning the best solution found so far).

    Returns:
        An OptimizationResult summarizing the final state of the run.
    """
    import numpy as np
    from scipy.optimize import minimize

    from botorch_rs.optim.closures import NdarrayOptimizationClosure

    start_time = monotonic()
    wrapped = (
        closure
        if isinstance(closure, NdarrayOptimizationClosure)
        else NdarrayOptimizationClosure(closure, parameters)
    )
    bounds_np = None
    if bounds is not None:
        lower, upper = [], []
        for name, p in parameters.items():
            lo, hi = bounds.get(name, (None, None))
            size = int(np.prod(getattr(p, "shape", ()) or (1,)))
            lower.extend([-np.inf if lo is None else float(lo)] * size)
            upper.extend([np.inf if hi is None else float(hi)] * size)
        bounds_np = list(zip(lower, upper, strict=True))
    counter = count(1)

    class _Timeout(Exception):
        def __init__(self, x: Any) -> None:
            self.x = x

    def wrapped_callback(x: Any) -> None:
        if callback is not None:
            result = OptimizationResult(
                step=next(counter),
                fval=float(wrapped(x)[0]),
                status=OptimizationStatus.RUNNING,
                runtime=monotonic() - start_time,
            )
            callback(parameters, result)
        if timeout_sec is not None and monotonic() - start_time > timeout_sec:
            raise _Timeout(np.array(x, copy=True))

    x_start = wrapped.state if x0 is None else np.asarray(x0, dtype=np.float64)
    try:
        raw = minimize(
            wrapped,
            x_start,
            jac=True,
            bounds=bounds_np,
            method=method,
            options=options,
            callback=wrapped_callback,
        )
        x_final, fun, nit = raw.x, raw.fun, raw.nit
        msg = raw.message if isinstance(raw.message, str) else raw.message.decode("ascii")
        success = bool(raw.success)
    except _Timeout as e:
        x_final = e.x
        fun = float(wrapped(x_final)[0])
        nit = next(counter) - 1
        msg = f"Optimization timed out after {monotonic() - start_time} seconds."
        success = False
    wrapped.state = np.asarray(x_final)
    if success:
        status = OptimizationStatus.SUCCESS
    else:
        status = (
            OptimizationStatus.STOPPED
            if _LBFGSB_MAXITER_MAXFUN_REGEX.search(msg) or "Optimization timed out after" in msg
            else OptimizationStatus.FAILURE
        )
    return OptimizationResult(
        fval=float(fun), step=int(nit), status=status, message=msg, runtime=monotonic() - start_time
    )


def torch_minimize(
    closure: Callable[[], tuple[Any, Sequence[Any]]],
    parameters: dict[str, Any],
    bounds: dict[str, tuple[float | None, float | None]] | None = None,
    callback: Callable[[dict[str, Any], OptimizationResult], None] | None = None,
    optimizer: Any = None,
    scheduler: Any = None,
    step_limit: int | None = None,
    timeout_sec: float | None = None,
    stopping_criterion: Any = None,
) -> OptimizationResult:
    r"""Generic ``torch.optim``-based optimization routine (BoTorch's
    ``torch_minimize``) for a closure that sets the ``grad`` attributes of
    the torch tensors in ``parameters``.

    Args:
        closure: Callable that returns a tensor and an iterable of gradient
            tensors.
        parameters: A dictionary of tensors to be optimized.
        bounds: An optional dictionary of bounds for elements of
            ``parameters``.
        callback: A callable taking ``parameters`` and an OptimizationResult.
        optimizer: A ``torch.optim.Optimizer`` instance or a factory that
            takes a list of parameters (default ``torch.optim.Adam``).
        scheduler: A learning-rate scheduler or a factory taking the
            optimizer.
        step_limit: Maximum number of optimization steps.
        timeout_sec: Timeout in seconds before terminating the loop.
        stopping_criterion: A StoppingCriterion for the optimization loop.

    Returns:
        An OptimizationResult summarizing the final state of the run.
    """
    import torch

    start_time = monotonic()
    if step_limit is None:
        if stopping_criterion is None and timeout_sec is None:
            raise RuntimeError("No termination conditions were given.")
        step_limit = maxsize
    if optimizer is None:
        optimizer = torch.optim.Adam
    if not isinstance(optimizer, torch.optim.Optimizer):
        optimizer = optimizer(list(parameters.values()))
    if scheduler is not None and not isinstance(scheduler, torch.optim.lr_scheduler.LRScheduler):
        scheduler = scheduler(optimizer)
    if stopping_criterion is not None and hasattr(stopping_criterion, "reset"):
        stopping_criterion.reset()
    _bounds = {} if bounds is None else {n: b for n, b in bounds.items() if n in parameters}
    result: OptimizationResult
    step = 0
    for step in range(1, step_limit + 1):
        fval = closure()[0].detach()
        runtime = monotonic() - start_time
        result = OptimizationResult(
            step=step, fval=fval.cpu().item(), status=OptimizationStatus.RUNNING, runtime=runtime
        )
        if stopping_criterion and stopping_criterion(fval):
            result.status = OptimizationStatus.STOPPED
            result.message = "`torch_minimize` stopped due to `stopping_criterion`."
        if timeout_sec is not None and runtime >= timeout_sec:
            result.status = OptimizationStatus.STOPPED
            result.message = f"`torch_minimize` stopped due to timeout after {runtime} seconds."
        if callback:
            callback(parameters, result)
        if result.status != OptimizationStatus.RUNNING:
            break
        optimizer.step()
        for name, (lower, upper) in _bounds.items():
            parameters[name].data = parameters[name].clamp(min=lower, max=upper)
        if scheduler:
            scheduler.step()
    if result.status != OptimizationStatus.RUNNING:
        return replace(result, runtime=monotonic() - start_time)
    return OptimizationResult(
        step=step,
        fval=closure()[0].detach().cpu().item(),
        status=OptimizationStatus.STOPPED,
        runtime=monotonic() - start_time,
        message=f"`torch_minimize` stopped after reaching step_limit={step_limit}.",
    )


__all__ = ["OptimizationResult", "OptimizationStatus", "scipy_minimize", "torch_minimize"]

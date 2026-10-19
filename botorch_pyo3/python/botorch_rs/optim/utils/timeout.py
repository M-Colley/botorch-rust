"""Synchronous SciPy optimization with a timeout checked after each iteration."""

from __future__ import annotations

import importlib
import time
import warnings
from collections.abc import Callable
from contextlib import nullcontext
from typing import Any

from botorch_rs.exceptions.errors import OptimizationTimeoutError


def minimize_with_timeout(
    fun: Callable[..., Any],
    x0: Any,
    args: tuple[Any, ...] = (),
    method: Any = None,
    jac: Any = None,
    hess: Any = None,
    hessp: Any = None,
    bounds: Any = None,
    constraints: Any = (),
    tol: float | None = None,
    callback: Any = None,
    options: dict[str, Any] | None = None,
    timeout_sec: float | None = None,
) -> Any:
    """Forward SciPy's minimize arguments and stop at the first overdue callback.

    The return value on timeout contains the last iterate and ``success=False``.
    Scalar objectives and ``(value, gradient)`` objectives are both supported.
    This runs synchronously: a single slow objective evaluation cannot be
    preempted, and no worker continues after this function returns.
    """
    optimize = importlib.import_module("scipy.optimize")
    iterations = 0
    wrapped_callback = callback
    if timeout_sec is not None:
        start = time.monotonic()

        def timeout_callback(xk: Any) -> bool:
            nonlocal iterations
            iterations += 1
            runtime = time.monotonic() - start
            if runtime > timeout_sec:
                raise OptimizationTimeoutError(current_x=xk.copy(), runtime=runtime)
            return False

        if callback is None:
            wrapped_callback = timeout_callback
        elif callable(method):
            raise NotImplementedError("Custom callable not supported for `method` argument.")
        elif method == "trust-constr":

            def wrapped_callback(xk: Any, state: Any) -> Any:
                return callback(xk, state) or timeout_callback(xk)

        else:

            def wrapped_callback(xk: Any) -> None:
                timeout_callback(xk)
                callback(xk)

    try:
        threadpool = importlib.import_module("threadpoolctl").threadpool_limits
        blas_context = threadpool(limits=1, user_api="blas")
    except ImportError:
        # Limiting BLAS is a performance workaround, not an optimizer requirement.
        blas_context = nullcontext()
    try:
        with warnings.catch_warnings(), blas_context:
            warnings.filterwarnings("error", message="Method .* cannot handle")
            return optimize.minimize(
                fun=fun,
                x0=x0,
                args=args,
                method=method,
                jac=jac,
                hess=hess,
                hessp=hessp,
                bounds=bounds,
                constraints=constraints,
                tol=tol,
                callback=wrapped_callback,
                options=options,
            )
    except OptimizationTimeoutError as error:
        value = fun(error.current_x, *args)
        if jac is True:
            value = value[0]
        return optimize.OptimizeResult(
            fun=value,
            x=error.current_x,
            nit=iterations,
            success=False,
            status=1,
            message=f"Optimization timed out after {error.runtime} seconds.",
        )


__all__ = ["minimize_with_timeout"]

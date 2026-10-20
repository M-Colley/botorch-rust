"""Temporarily reduce real optimizer work in native and interoperability tests."""

from __future__ import annotations

import importlib
import inspect
from collections.abc import Callable, Generator
from contextlib import ExitStack, contextmanager
from dataclasses import replace
from functools import wraps
from typing import Any
from unittest import mock


def _limited(
    function: Callable[..., Any],
    *,
    samples: bool = False,
    iterations: bool = False,
    inner_samples: bool = False,
    options: bool = False,
) -> Callable[..., Any]:
    signature = inspect.signature(function)
    extra_name = next(
        (
            name
            for name, parameter in signature.parameters.items()
            if parameter.kind is inspect.Parameter.VAR_KEYWORD
        ),
        None,
    )

    @wraps(function)
    def limited(*args: Any, **kwargs: Any) -> Any:
        bound = signature.bind(*args, **kwargs)

        def set_argument(name: str, value: Any) -> None:
            if name in signature.parameters:
                bound.arguments[name] = value
            elif extra_name is not None:
                bound.arguments.setdefault(extra_name, {})[name] = value
            else:
                raise TypeError(f"{function.__name__} has no {name!r} option.")

        extra = bound.arguments.get(extra_name, {}) if extra_name is not None else {}
        if samples and extra.get("batch_initial_conditions") is None:
            set_argument("num_restarts", 2)
            set_argument("raw_samples", 4)
        if iterations:
            set_argument("maxiter", 2)
        if inner_samples:
            set_argument("num_inner_restarts", 2)
            set_argument("raw_inner_samples", 4)
        if options:
            settings = dict(bound.arguments.get("options") or {})
            settings["maxiter"] = 2
            set_argument("options", settings)
        return function(*bound.args, **bound.kwargs)

    return limited


@contextmanager
def mock_optimize_context_manager(force: bool = False) -> Generator[None, None, None]:
    """Run actual optimizers with two iterations and minimal generated starts.

    This port reaches its native optimizer directly. The context therefore
    patches the native binding entry points in addition to generic Python
    optimizers, rather than mocking unused SciPy aliases. Exiting restores all
    functions and mixed-optimizer defaults, including when the body raises.
    """
    from botorch_rs import _botorch_rs as rs

    core = importlib.import_module("botorch_rs.optim.core")
    fit = importlib.import_module("botorch_rs.optim.fit")
    mixed = importlib.import_module("botorch_rs.optim.optimize_mixed")
    timeout = importlib.import_module("botorch_rs.optim.utils.timeout")
    patches = []
    with ExitStack() as stack:

        def patch(module: Any, name: str, **limits: Any) -> None:
            original = getattr(module, name)
            replacement = _limited(original, **limits)
            patched = stack.enter_context(mock.patch.object(module, name, wraps=replacement))
            # Preserve signatures for nested contexts and introspection.
            patched.__signature__ = inspect.signature(original)
            patches.append(patched)

        patch(rs, "optimize_acqf", samples=True, iterations=True, inner_samples=True)
        patch(rs, "gen_candidates_scipy_py", iterations=True)
        patch(rs, "gen_batch_initial_conditions", samples=True)
        patch(rs, "fit_gpytorch_mll_scipy_py", iterations=True)
        patch(core, "scipy_minimize", options=True)
        patch(fit, "scipy_minimize", options=True)
        patch(timeout, "minimize_with_timeout", options=True)
        try:
            scipy_optimize = importlib.import_module("scipy.optimize")
        except ImportError:
            pass  # Native-only environments do not need SciPy for this context.
        else:
            # Generic scipy_minimize imports this function locally. Patching
            # here also reaches aliases imported before entering the context.
            patch(scipy_optimize, "minimize", options=True)
        # Constants control actual native mixed-optimizer options. Tracking the
        # problem factory records their use even if no continuous step occurs.
        original_problem = mixed._problem

        @wraps(original_problem)
        def small_problem(opt_inputs: Any, *args: Any, **kwargs: Any) -> Any:
            settings = dict(opt_inputs.options or {})
            for name in ("maxiter_alternating", "maxiter_discrete", "maxiter_continuous"):
                settings.setdefault(name, 1)
            return original_problem(replace(opt_inputs, options=settings), *args, **kwargs)

        patches.append(
            stack.enter_context(mock.patch.object(mixed, "_problem", wraps=small_problem))
        )
        for name in ("MAX_ITER_ALTER", "MAX_ITER_DISCRETE", "MAX_ITER_CONT"):
            stack.enter_context(mock.patch.object(mixed, name, 1))
        yield
    if not force and all(patch.call_count == 0 for patch in patches):
        raise AssertionError(
            "No mocks were called in the context manager. Please remove unused "
            "mock_optimize_context_manager()."
        )


def mock_optimize(f: Callable[..., Any]) -> Callable[..., Any]:
    """Decorate a test with :func:`mock_optimize_context_manager`."""

    @wraps(f)
    def inner(*args: Any, **kwargs: Any) -> Any:
        with mock_optimize_context_manager():
            return f(*args, **kwargs)

    return inner


__all__ = ["mock_optimize", "mock_optimize_context_manager"]

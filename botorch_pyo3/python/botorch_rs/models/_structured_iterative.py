"""Scoped solver configuration shared by native and Tensor structured models."""

from __future__ import annotations

import math
import sys
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class IterativeSettings:
    tol: float = 0.01
    max_iter: int = 10000
    covar_root_decomposition: bool = False
    log_prob: bool = True
    solves: bool = True
    max_cholesky_size: int = 800
    num_trace_samples: int = 10
    max_lanczos_quadrature_iterations: int = 20


_current: ContextVar[IterativeSettings | None] = ContextVar("botorch_rs_iterative", default=None)


def current() -> IterativeSettings | None:
    return _current.get()


def setting(name: str, default: Any) -> Any:
    """Respect optional caller GPyTorch settings without importing a runtime."""
    module = sys.modules.get("gpytorch.settings")
    descriptor = None if module is None else getattr(module, name, None)
    return default if descriptor is None else descriptor.value()


@contextmanager
def use_iterative_methods(
    tol: float = 0.01,
    max_iter: int = 10000,
    covar_root_decomposition: bool = False,
    log_prob: bool = True,
    solves: bool = True,
) -> Iterator[None]:
    if not math.isfinite(tol) or tol <= 0 or max_iter < 1:
        raise ValueError("Iterative tolerance and maximum iterations must be positive.")
    config = IterativeSettings(
        tol,
        max_iter,
        covar_root_decomposition,
        log_prob,
        solves,
        setting("max_cholesky_size", 800),
        setting("num_trace_samples", 10),
        setting("max_lanczos_quadrature_iterations", 20),
    )
    token = _current.set(config)
    try:
        with ExitStack() as stack:
            # Supplied GPyTorch modules see the same settings as in the source.
            # The port's own solvers use the configuration above independently.
            settings = sys.modules.get("gpytorch.settings")
            if settings is not None:
                stack.enter_context(
                    settings.fast_computations(
                        covar_root_decomposition=covar_root_decomposition,
                        log_prob=log_prob,
                        solves=solves,
                    )
                )
                stack.enter_context(settings.cg_tolerance(tol))
                stack.enter_context(settings.max_cg_iterations(max_iter))
            yield
    finally:
        _current.reset(token)


__all__ = ["IterativeSettings", "current", "setting", "use_iterative_methods"]

"""Fantasy sampling may stop training-cache graphs while keeping query paths."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any

_detach_cache: ContextVar[bool] = ContextVar(
    "botorch_fantasy_detach_prediction_cache", default=False
)


@contextmanager
def prediction_cache_gradients(propagate: bool) -> Iterator[None]:
    token = _detach_cache.set(not propagate)
    try:
        yield
    finally:
        _detach_cache.reset(token)


def prediction_factor(factor: Any, alpha: Any) -> tuple[Any, Any]:
    return (factor.detach(), alpha.detach()) if _detach_cache.get() else (factor, alpha)

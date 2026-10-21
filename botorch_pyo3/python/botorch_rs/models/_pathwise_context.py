"""Keep functional training-data transforms while suppressing path outputs."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any

_SAVED_OUTCOMES: ContextVar[dict[int, Any] | None] = ContextVar(
    "pathwise_training_outcomes", default=None
)


def training_outcome_transform(model: Any) -> Any:
    """Training algebra still uses its fitted transform during path construction."""
    return getattr(model, "outcome_transform", (_SAVED_OUTCOMES.get() or {}).get(id(model)))


@contextmanager
def suppress_path_outcome(model: Any) -> Iterator[None]:
    """Match the reference callback protocol, restoring state even on exceptions."""
    from botorch_rs.utils.context_managers import delattr_ctx

    saved = _SAVED_OUTCOMES.get() or {}
    token = _SAVED_OUTCOMES.set({**saved, id(model): training_outcome_transform(model)})
    try:
        with delattr_ctx(model, "outcome_transform"):
            yield
    finally:
        _SAVED_OUTCOMES.reset(token)

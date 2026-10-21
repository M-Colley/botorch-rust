"""Small dispatch predicate shared by native and caller-owned Tensor MLLs."""

from __future__ import annotations

from typing import Any


def is_tensor_mll(module: Any) -> bool:
    return (
        getattr(module, "_tensor_backend", None) == "torch"
        or getattr(getattr(module, "model", module), "_tensor_backend", None) == "torch"
    )

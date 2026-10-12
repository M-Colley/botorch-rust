"""Helpers shared by the multi-objective utilities (not part of the API)."""

from __future__ import annotations

import importlib
import sys
from collections.abc import Sequence
from typing import Any

from botorch_rs._arrays import ArrayKind, from_flat


def shape_repr(x: Any, shape: Sequence[int]) -> str:
    """A shape as BoTorch prints it: ``torch.Size([...])`` for torch inputs,
    else a tuple."""
    torch = sys.modules.get("torch")
    if torch is not None and isinstance(x, torch.Tensor):
        return str(torch.Size(tuple(shape)))
    return str(tuple(shape))


def size(shape: Sequence[int], kind: ArrayKind) -> Any:
    """A shape as ``torch.Size`` for torch results, else as a tuple."""
    if kind.lib == "torch":
        torch = sys.modules.get("torch") or importlib.import_module("torch")
        return torch.Size(tuple(shape))
    return tuple(shape)


def long_array(values: Sequence[int], kind: ArrayKind) -> Any:
    """Indices as an int64 tensor or NumPy array, or a list of ints."""
    if kind.lib == "torch":
        torch = sys.modules.get("torch") or importlib.import_module("torch")
        t = torch.tensor(list(values), dtype=torch.long)
        if kind.device is not None and kind.device.type != "cpu":
            t = t.to(kind.device)
        return t
    if kind.lib == "numpy":
        np = sys.modules.get("numpy") or importlib.import_module("numpy")
        return np.asarray(list(values), dtype=np.int64)
    return [int(v) for v in values]


def scalar_zero(kind: ArrayKind) -> Any:
    """BoTorch's ``torch.tensor(0.0)`` (torch's default dtype) for the
    hypervolume of a decomposition without outcomes."""
    if kind.lib == "torch":
        torch = sys.modules.get("torch") or importlib.import_module("torch")
        t = torch.tensor(0.0)
        if kind.device is not None and kind.device.type != "cpu":
            t = t.to(kind.device)
        return t
    return from_flat([0.0], (), kind)

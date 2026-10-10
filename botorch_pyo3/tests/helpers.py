"""Helpers shared by the botorch_rs tests (array conversion, test data)."""

from __future__ import annotations

import importlib
import importlib.util
import math
import os
import random
from typing import Any

import pytest

HAS_NUMPY = importlib.util.find_spec("numpy") is not None
HAS_TORCH = importlib.util.find_spec("torch") is not None

KINDS = ["list"] + (["numpy"] if HAS_NUMPY else []) + (["torch"] if HAS_TORCH else [])


def as_kind(x: Any, kind: str) -> Any:
    """Converts a nested list of floats to the given array type (float64)."""
    if kind == "list":
        return x
    if kind == "numpy":
        import numpy as np

        return np.asarray(x, dtype=np.float64)
    import torch

    return torch.tensor(x, dtype=torch.float64)


def to_list(x: Any) -> Any:
    """Converts an array (or a float) back to nested lists of floats."""
    if hasattr(x, "tolist"):
        return x.tolist()
    return x


def check_kind(x: Any, kind: str) -> None:
    """Asserts that ``x`` has the array type ``kind`` (float64)."""
    if kind == "list":
        assert isinstance(x, (list, float)), type(x)
    elif kind == "numpy":
        import numpy as np

        assert isinstance(x, (np.ndarray, np.floating)), type(x)
        assert x.dtype == np.float64
    else:
        import torch

        assert isinstance(x, torch.Tensor), type(x)
        assert x.dtype == torch.float64


def shape(x: Any) -> tuple[int, ...]:
    """The shape of an array, nested list or float."""
    if hasattr(x, "shape"):
        return tuple(x.shape)
    out = []
    while isinstance(x, list):
        out.append(len(x))
        x = x[0] if x else None
    return tuple(out)


def flatten(x: Any) -> list[float]:
    x = to_list(x)
    if isinstance(x, list):
        return [v for item in x for v in flatten(item)]
    return [float(x)]


def make_data(n: int = 12, d: int = 2, seed: int = 0, noise: float = 0.05) -> tuple[list, list]:
    """Deterministic training data on [0, 1]^d: a smooth function plus noise."""
    rng = random.Random(seed)
    X = [[rng.random() for _ in range(d)] for _ in range(n)]
    Y = [[math.sin(6.0 * x[0]) + sum(x[1:]) ** 2 + noise * rng.gauss(0.0, 1.0)] for x in X]
    return X, Y


def assert_close(a: Any, b: Any, rtol: float = 1e-9, atol: float = 1e-12) -> None:
    fa, fb = flatten(a), flatten(b)
    assert len(fa) == len(fb)
    for u, v in zip(fa, fb, strict=True):
        assert abs(u - v) <= atol + rtol * max(abs(u), abs(v)), (u, v)


def importorskip_botorch() -> None:
    """Skips the calling test (module) unless BoTorch is importable. With
    ``BOTORCH_RS_REQUIRE_BOTORCH=1`` (CI's parity job) a missing BoTorch is an
    error instead of a skip."""
    if os.environ.get("BOTORCH_RS_REQUIRE_BOTORCH"):
        importlib.import_module("botorch.fit")
    pytest.importorskip("botorch.fit")


class ParityLog:
    """Worst observed error per compared quantity (printed in the pytest
    terminal summary, see ``conftest.py``)."""

    worst: dict[str, float] = {}

    @classmethod
    def record(cls, quantity: str, err: float) -> None:
        cls.worst[quantity] = max(cls.worst.get(quantity, 0.0), float(err))

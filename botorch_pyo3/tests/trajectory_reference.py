"""Control the sampling grid when comparing the native trajectory spline.

PyTorch's CPU linspace may contract ``end - step * i`` into an FMA on Linux.
The native test function uses the portable two-operation float32 expression.
Only the reference's default 1000-point smooth-interpolation grid is replaced;
its original geometry, SciPy spline and cost calculation remain unchanged.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any
from unittest.mock import patch


@contextmanager
def portable_trajectory_grid(enabled: bool = True) -> Iterator[None]:
    if not enabled:
        yield
        return

    import torch

    original = torch.linspace

    def linspace(start: Any, end: Any, steps: int, *args: Any, **kwargs: Any) -> Any:
        if (
            start == 0
            and end == 1
            and steps == 1000
            and not args
            and not kwargs
            and torch.get_default_dtype() == torch.float32
        ):
            indices = torch.arange(steps, dtype=torch.float32, device="cpu")
            step = torch.tensor(1.0, dtype=torch.float32, device="cpu") / (steps - 1)
            left = step * indices
            right_product = step * (steps - indices - 1)
            right = 1.0 - right_product
            return torch.where(indices < steps // 2, left, right)
        return original(start, end, steps, *args, **kwargs)

    with patch.object(torch, "linspace", linspace):
        yield

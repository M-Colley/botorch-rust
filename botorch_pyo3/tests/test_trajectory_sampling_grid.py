"""Separate CPU float32 grid contraction from trajectory spline accuracy."""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

import pytest

from helpers import importorskip_botorch

importorskip_botorch()
torch = pytest.importorskip("torch")

from botorch.test_functions.synthetic import TrajectoryPlanning as Reference  # noqa: E402
from trajectory_reference import portable_trajectory_grid  # noqa: E402

from botorch_rs.test_functions.synthetic import TrajectoryPlanning  # noqa: E402
from parity_helpers import rel_err  # noqa: E402


@pytest.fixture(autouse=True)
def default_float32() -> Any:
    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.float32)
    try:
        yield
    finally:
        torch.set_default_dtype(previous)


def test_portable_grid_context_changes_only_the_default_smooth_sampling_grid() -> None:
    original = torch.linspace
    other = original(0, 1, 7)
    double = original(0, 1, 1000, dtype=torch.double)
    with portable_trajectory_grid():
        grid = torch.linspace(0, 1, 1000)
        # The two float32 roundings give this value; an FMA gives 0x3f03759f.
        assert grid.view(torch.int32)[513].item() == 0x3F0375A0
        assert torch.equal(torch.linspace(0, 1, 7), other)
        assert torch.equal(torch.linspace(0, 1, 1000, dtype=torch.double), double)
    assert torch.linspace is original


@pytest.mark.parametrize(
    "kwargs,ci_error",
    [({"dim": 8}, 2.749579202248941e-7), ({"dim": 6, "max_step_size": 0.2}, 6.8784293081513e-8)],
)
def test_fused_grid_explains_raw_ci_difference_and_portable_grid_preserves_parity(
    kwargs: dict[str, Any], ci_error: float
) -> None:
    reference = Reference(**kwargs)
    native = TrajectoryPlanning(**kwargs)
    X = torch.rand(
        9, reference.dim, generator=torch.Generator().manual_seed(40), dtype=torch.double
    )
    actual = native.evaluate_true(X)
    original = torch.linspace

    def fused_grid(start: Any, end: Any, steps: int, *args: Any, **kw: Any) -> Any:
        if start == 0 and end == 1 and steps == 1000 and not args and not kw:
            indices = torch.arange(steps, dtype=torch.double)
            step = (torch.tensor(1.0, dtype=torch.float32) / (steps - 1)).double()
            # Double intermediates exactly represent these float32 products;
            # the final cast emulates the single rounding of a float32 FMA.
            return torch.where(
                indices < steps // 2, step * indices, 1.0 - step * (steps - indices - 1)
            ).float()
        return original(start, end, steps, *args, **kw)

    with patch.object(torch, "linspace", fused_grid):
        contracted = reference.evaluate_true(X)
    assert rel_err(actual, contracted) == pytest.approx(ci_error, rel=1e-7, abs=1e-15)

    with portable_trajectory_grid():
        expected = reference.evaluate_true(X)
        forward_expected = reference(X)
    torch.testing.assert_close(actual, expected, rtol=1e-9, atol=1e-12)
    torch.testing.assert_close(native.evaluate_true(X.reshape(3, 3, -1)), expected.reshape(3, 3))
    torch.testing.assert_close(native(X), forward_expected, rtol=1e-9, atol=1e-12)

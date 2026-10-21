"""Validate a proposed upstream fix without changing the pinned checkout."""

from __future__ import annotations

import itertools
import math
import os
import subprocess
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

pytestmark = pytest.mark.filterwarnings(
    "ignore:The last_dim_is_batch argument is deprecated:DeprecationWarning"
)


def _module(path: Path) -> ModuleType:
    module = ModuleType("isolated_" + path.stem)
    module.__file__ = str(path)
    # Compile text directly: loading the official file must not create pyc files.
    exec(compile(path.read_text(encoding="utf-8"), str(path), "exec"), module.__dict__)
    return module


@pytest.fixture(scope="module")
def reference(tmp_path_factory: Any) -> Any:
    pytest.importorskip("botorch")
    root = Path(
        os.environ.get(
            "BOTORCH_RS_REFERENCE_ROOT", Path(__file__).resolve().parents[2] / ".reference/botorch"
        )
    )
    relative = Path("botorch/models/kernels/categorical.py")
    source = root / relative
    if not source.is_file():
        pytest.skip("Requires official BoTorch b0dd01be6 via BOTORCH_RS_REFERENCE_ROOT")
    patch = (
        Path(__file__).resolve().parents[2] / "docs/patches/botorch-categorical-feature-axis.patch"
    )
    original = source.read_bytes()
    checked = subprocess.run(
        ["git", "-C", str(root), "apply", "--check", str(patch)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert checked.returncode == 0, checked.stderr
    isolated = tmp_path_factory.mktemp("categorical-feature-patch")
    copied = isolated / relative
    copied.parent.mkdir(parents=True)
    copied.write_bytes(original)
    applied = subprocess.run(
        ["git", "-C", str(isolated), "apply", str(patch)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert applied.returncode == 0, applied.stderr
    assert source.read_bytes() == original
    return SimpleNamespace(
        original=_module(source).CategoricalKernel,
        patched=_module(copied).CategoricalKernel,
    )


def _expected(kernel: Any, x: Any, y: Any) -> Any:
    import torch

    batch = torch.broadcast_shapes(x.shape[:-2], y.shape[:-2], kernel.batch_shape)
    n, d = x.shape[-2:]
    m = y.shape[-2]
    a, b = x.expand(*batch, n, d), y.expand(*batch, m, d)
    lengthscales = kernel.lengthscale.detach().expand(*batch, 1, d)
    output = torch.empty(*batch, d, n, m, dtype=x.dtype)
    for index in itertools.product(*(range(size) for size in batch)):
        for feature, row, column in itertools.product(range(d), range(n), range(m)):
            mismatch = float(a[index + (row, feature)] != b[index + (column, feature)])
            scale = float(lengthscales[index + (0, feature)])
            output[index + (feature, row, column)] = math.exp(-mismatch / scale)
    return output


@pytest.mark.parametrize("batched", [False, True])
@pytest.mark.parametrize("equal_size", [False, True])
@pytest.mark.parametrize("active", [None, [2, 0]])
def test_patched_categorical_matches_independent_feature_covariance(
    reference: Any, batched: bool, equal_size: bool, active: list[int] | None
) -> None:
    torch = pytest.importorskip("torch")
    x = torch.tensor([[0, 1, 0], [1, 0, 2]], dtype=torch.double)
    y = torch.tensor([[1, 0, 2], [2, 1, 0], [0, 0, 1]], dtype=torch.double)
    if equal_size:
        y = y[:2]
    if batched:
        x = torch.stack((x, x + 1)).unsqueeze(-3)
        y = torch.stack((y, y.flip(-2), y + 2))
    d = 3 if active is None else len(active)
    kernel = reference.patched(
        ard_num_dims=d, active_dims=active, batch_shape=torch.Size((2, 1) if batched else ())
    ).double()
    scales = torch.tensor([0.4, 0.7, 1.2][:d], dtype=torch.double)
    kernel.lengthscale = (
        torch.stack((scales, scales * 1.7)).reshape(2, 1, 1, d) if batched else scales
    )
    selected_x, selected_y = (x, y) if active is None else (x[..., active], y[..., active])
    expected = _expected(kernel, selected_x, selected_y)
    actual = kernel(x, y, last_dim_is_batch=True).to_dense()
    torch.testing.assert_close(actual, expected, rtol=1e-14, atol=1e-14)
    torch.testing.assert_close(
        kernel(x, y).to_dense(), expected.log().mean(-3).exp(), rtol=1e-14, atol=1e-14
    )
    if equal_size:
        torch.testing.assert_close(
            # GPyTorch1.15.2 __call__ incorrectly extracts another diagonal
            # when feature_count == point_count; test this kernel's branch.
            kernel.forward(selected_x, selected_y, diag=True, last_dim_is_batch=True),
            expected.diagonal(dim1=-2, dim2=-1),
            rtol=1e-14,
            atol=1e-14,
        )


def test_original_categorical_swaps_unequal_and_asymmetric_point_axes(reference: Any) -> None:
    import torch

    x = torch.tensor([[0.0], [1.0]], dtype=torch.double)
    y = torch.tensor([[1.0], [2.0], [0.0]], dtype=torch.double)
    kernel = reference.original().double()
    expected = _expected(kernel, x, y)
    assert kernel.forward(x, y, last_dim_is_batch=True).shape == (1, 3, 2)
    torch.testing.assert_close(
        kernel.forward(x, y[:2], last_dim_is_batch=True),
        _expected(kernel, x, y[:2]).transpose(-2, -1),
    )
    assert not torch.allclose(
        kernel.forward(x, y[:2], last_dim_is_batch=True), _expected(kernel, x, y[:2])
    )
    assert expected.shape == (1, 2, 3)


def test_gpytorch_composite_lazy_feature_axis_reproduction() -> None:
    torch = pytest.importorskip("torch")
    gpytorch = pytest.importorskip("gpytorch")
    if gpytorch.__version__ != "1.15.2":
        pytest.skip("Verified lazy-shape reproduction is pinned to GPyTorch 1.15.2")
    x = torch.tensor([[0.0, 0.2, 0.7], [0.8, 0.4, 0.1]], dtype=torch.double)
    y = torch.tensor([[0.1, 0.3, 0.9], [0.4, 0.5, 0.2], [0.7, 0.6, 0.3]], dtype=torch.double)
    kernel = (
        gpytorch.kernels.RBFKernel(active_dims=[0, 2])
        + gpytorch.kernels.RBFKernel(active_dims=[1, 2])
    ).double()
    expected = sum(child(x, y, last_dim_is_batch=True).to_dense() for child in kernel.kernels)
    with pytest.raises(RuntimeError, match="expected shape"):
        kernel(x, y, last_dim_is_batch=True).to_dense()
    with gpytorch.settings.lazily_evaluate_kernels(False):
        actual = kernel(x, y, last_dim_is_batch=True).to_dense()
    assert actual.shape == (2, 2, 3)
    torch.testing.assert_close(actual, expected)


def test_gpytorch_diag_wrapper_mistakes_feature_axis_for_point_axis() -> None:
    torch = pytest.importorskip("torch")
    gpytorch = pytest.importorskip("gpytorch")
    if gpytorch.__version__ != "1.15.2":
        pytest.skip("Verified diagonal-wrapper reproduction is pinned to GPyTorch 1.15.2")
    x = torch.tensor([[0.1, 0.2, 0.7], [0.8, 0.4, 0.1], [0.5, 0.6, 0.9]], dtype=torch.double)
    y = x.flip(-2) + 0.05
    kernel = gpytorch.kernels.RBFKernel(ard_num_dims=3).double()
    expected = kernel(x, y, last_dim_is_batch=True).to_dense().diagonal(dim1=-2, dim2=-1)
    correct = kernel.forward(x, y, diag=True, last_dim_is_batch=True)
    torch.testing.assert_close(correct, expected)
    assert correct.shape == (3, 3)
    incorrect = kernel(x, y, diag=True, last_dim_is_batch=True)
    assert incorrect.shape == (3,)
    torch.testing.assert_close(incorrect, expected.diagonal())

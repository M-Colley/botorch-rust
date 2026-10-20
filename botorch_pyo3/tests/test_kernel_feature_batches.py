"""Feature batch covariance matrices use native kernel views and ARD slices."""

from __future__ import annotations

import pytest

from botorch_rs import _ops
from botorch_rs import kernels as rs
from botorch_rs._arrays import to_flat
from helpers import as_kind as _as_kind
from helpers import assert_close


def array(value, kind):
    return _as_kind(value.tolist() if kind == "list" else value, kind)


def kernels(name, batch_shape=()):
    torch = pytest.importorskip("torch")
    bt = pytest.importorskip("gpytorch.kernels")
    kwargs = {"batch_shape": torch.Size(batch_shape)}

    def one(module):
        if name == "sum":
            return module.RBFKernel(ard_num_dims=3, **kwargs) + module.ScaleKernel(
                module.MaternKernel(nu=1.5, ard_num_dims=3, **kwargs), **kwargs
            )
        if name == "product_active":
            return module.RBFKernel(ard_num_dims=3, active_dims=[2, 0, 1], **kwargs) * (
                module.LinearKernel(active_dims=[1], **kwargs)
            )
        if name == "scale_active":
            return module.ScaleKernel(
                module.RBFKernel(ard_num_dims=2, active_dims=[2, 0], **kwargs), **kwargs
            )
        options = dict(kwargs)
        if name in ("RBFKernel", "MaternKernel", "RQKernel", "PeriodicKernel"):
            options["ard_num_dims"] = 3
        if name == "MaternKernel":
            options["nu"] = 1.5
        if name == "PolynomialKernel":
            options["power"] = 3
        return getattr(module, name)(**options)

    actual, expected = one(rs), one(bt).to(dtype=torch.double)
    reference_parameters = dict(expected.named_parameters())
    for name, value in actual.named_parameters():
        _, shape = to_flat(value)
        n = reference_parameters[name].numel()
        raw = (torch.arange(n, dtype=torch.double) * 0.17 - 0.3).reshape(shape)
        actual.initialize(**{name: raw})
        with torch.no_grad():
            reference_parameters[name].copy_(raw)
    return actual, expected


@pytest.mark.parametrize(
    "name",
    [
        "RBFKernel",
        "MaternKernel",
        "RQKernel",
        "PeriodicKernel",
        "LinearKernel",
        "PolynomialKernel",
        "CosineKernel",
        "sum",
        "product_active",
        "scale_active",
    ],
)
@pytest.mark.parametrize("diag", [False, True])
def test_feature_covariance_matches_official(kind, name, diag):
    torch = pytest.importorskip("torch")
    actual, expected = kernels(name)
    x = torch.tensor(
        [[0.1, 0.8, 0.3], [0.6, 0.2, 0.7], [0.4, 0.1, 0.9], [0.85, 0.6, 0.45]],
        dtype=torch.double,
    )
    y = x.flip(0) + 0.05
    a = actual(array(x, kind), array(y, kind), diag=diag, last_dim_is_batch=True)
    b = expected(x, y, diag=diag, last_dim_is_batch=True)
    if not diag:
        b = b.to_dense()
    assert _ops.shape_of(a) == tuple(b.shape)
    assert_close(a, b, rtol=2e-12, atol=2e-12)


@pytest.mark.parametrize("name", ["RBFKernel", "sum", "scale_active"])
@pytest.mark.parametrize("diag", [False, True])
def test_feature_batches_broadcast_input_and_kernel_batches(kind, name, diag):
    torch = pytest.importorskip("torch")
    actual, expected = kernels(name, (3,))
    x = torch.linspace(0.1, 0.9, 24, dtype=torch.double).reshape(2, 1, 4, 3)
    y = torch.linspace(0.2, 0.8, 36, dtype=torch.double).reshape(3, 4, 3)
    a = actual(array(x, kind), array(y, kind), diag=diag, last_dim_is_batch=True)
    b = expected(x, y, diag=diag, last_dim_is_batch=True)
    if not diag:
        b = b.to_dense()
    assert _ops.shape_of(a) == tuple(b.shape)
    assert_close(a, b, rtol=3e-12, atol=3e-12)


@pytest.mark.parametrize("feature_batch", [False, True])
@pytest.mark.parametrize("case", ["plain", "sum", "scaled_sum", "scaled", "nested_scale"])
def test_explicit_single_ard_dimension_validates_actual_call_boundaries(case, feature_batch):
    torch = pytest.importorskip("torch")
    bt = pytest.importorskip("gpytorch.kernels")

    def make(module):
        base = module.RBFKernel(ard_num_dims=1)
        if case == "plain":
            return base
        if case == "sum":
            return base + module.RBFKernel()
        if case == "scaled_sum":
            return module.ScaleKernel(base + module.RBFKernel())
        if case == "scaled":
            return module.ScaleKernel(base)
        return module.ScaleKernel(module.ScaleKernel(base, ard_num_dims=1))

    actual, expected = make(rs), make(bt).double()
    x = torch.tensor([[0.1, 0.8, 0.3], [0.6, 0.2, 0.7]], dtype=torch.double)
    if case in ("plain", "sum", "scaled_sum"):
        with pytest.raises(RuntimeError, match="dimensionality"):
            expected(x, last_dim_is_batch=feature_batch).to_dense()
        with pytest.raises(RuntimeError, match="dimensionality"):
            actual(x, last_dim_is_batch=feature_batch)
    else:
        assert_close(
            actual(x, last_dim_is_batch=feature_batch),
            expected(x, last_dim_is_batch=feature_batch).to_dense(),
            rtol=2e-12,
            atol=2e-12,
        )


@pytest.mark.parametrize("feature_batch", [False, True])
def test_mutated_scale_selection_and_unequal_raw_input_widths(kind, feature_batch):
    torch = pytest.importorskip("torch")
    bt = pytest.importorskip("gpytorch.kernels")
    actual = rs.ScaleKernel(rs.RBFKernel(ard_num_dims=2, active_dims=[2, 0]))
    expected = bt.ScaleKernel(bt.RBFKernel(ard_num_dims=2, active_dims=[2, 0])).double()
    actual.active_dims = (1, 0)
    expected.active_dims = torch.tensor([1, 0])
    x = torch.tensor([[0.1, 0.8, 0.3], [0.6, 0.2, 0.7]], dtype=torch.double)
    y = torch.tensor([[0.4, 0.1, 0.9, 5.0], [0.3, 0.5, 0.7, 6.0]], dtype=torch.double)
    a = actual(array(x, kind), array(y, kind), last_dim_is_batch=feature_batch)
    b = expected(x, y, last_dim_is_batch=feature_batch).to_dense()
    assert_close(a, b, rtol=2e-12, atol=2e-12)


@pytest.mark.parametrize("feature_batch", [False, True])
@pytest.mark.parametrize("diag", [False, True])
def test_empty_input_batch_retains_event_and_feature_axes(kind, feature_batch, diag):
    torch = pytest.importorskip("torch")
    if kind == "list":
        pytest.skip("Nested lists cannot encode trailing axes after an empty batch.")
    actual, expected = kernels("RBFKernel")
    x = torch.empty(0, 4, 3, dtype=torch.double)
    a = actual(array(x, kind), diag=diag, last_dim_is_batch=feature_batch)
    b = expected(x, diag=diag, last_dim_is_batch=feature_batch)
    if not diag:
        b = b.to_dense()
    assert _ops.shape_of(a) == tuple(b.shape)


@pytest.mark.parametrize("equal_size", [False, True])
def test_categorical_feature_cross_covariance_preserves_point_order(kind, equal_size):
    torch = pytest.importorskip("torch")
    from botorch_rs.models.kernels.categorical import CategoricalKernel

    kernel = CategoricalKernel(ard_num_dims=2)
    lengthscale = torch.tensor([[0.4, 0.8]], dtype=torch.double)
    kernel.lengthscale = lengthscale
    x = torch.tensor([[0.0, 1.0], [1.0, 0.0]], dtype=torch.double)
    y = torch.tensor([[1.0, 1.0], [2.0, 0.0], [0.0, 2.0]], dtype=torch.double)
    if equal_size:
        y = y[:2]
    expected = torch.exp(
        -((x.unsqueeze(-2) != y.unsqueeze(-3)) / lengthscale.unsqueeze(-2)).movedim(-1, -3)
    )
    actual = kernel(array(x, kind), array(y, kind), last_dim_is_batch=True)
    assert _ops.shape_of(actual) == tuple(expected.shape)
    assert_close(actual, expected, rtol=2e-12, atol=2e-12)


def test_equal_feature_and_point_counts_keep_diagonal_feature_axis():
    torch = pytest.importorskip("torch")
    actual, expected = kernels("RBFKernel")
    x = torch.tensor([[0.1, 0.8, 0.3], [0.6, 0.2, 0.7], [0.4, 0.1, 0.9]], dtype=torch.double)
    # GPyTorch's generic diag wrapper mistakenly takes a second diagonal when
    # D==N. The full covariance's actual diagonal is the reference instead.
    full = expected(x, last_dim_is_batch=True).to_dense()
    result = actual(x, diag=True, last_dim_is_batch=True)
    torch.testing.assert_close(result, full.diagonal(dim1=-2, dim2=-1), rtol=2e-12, atol=2e-12)
    assert result.shape == (3, 3)


@pytest.mark.parametrize("feature_batch", [False, True])
def test_empty_input_batch_still_validates_diagonal_point_counts(feature_batch):
    torch = pytest.importorskip("torch")
    kernel, _ = kernels("RBFKernel")
    with pytest.raises(RuntimeError, match="same number of points"):
        kernel(
            torch.empty(0, 2, 3, dtype=torch.double),
            torch.empty(0, 3, 3, dtype=torch.double),
            diag=True,
            last_dim_is_batch=feature_batch,
        )

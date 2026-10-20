"""Official-source numerical parity and native empirical utility derivatives."""

from __future__ import annotations

import importlib.util
import math
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from botorch_rs.exceptions.errors import UnsupportedError
from botorch_rs.models.empirical_gps import utils as rs
from helpers import KINDS, as_kind, assert_close, check_kind, shape, to_list


@pytest.fixture(scope="module")
def official() -> object:
    pytest.importorskip("botorch")
    root = Path(
        os.environ.get(
            "BOTORCH_RS_REFERENCE_ROOT", Path(__file__).resolve().parents[2] / ".reference/botorch"
        )
    )
    path = root / "botorch/models/empirical_gps/utils.py"
    if not path.is_file():
        pytest.skip("Requires official empirical GP utility source via BOTORCH_RS_REFERENCE_ROOT")
    spec = importlib.util.spec_from_file_location("official_empirical_utils", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("kind", KINDS)
def test_covariance_centering_interpolation_and_metadata(kind: str) -> None:
    curves = as_kind([[1.0, 2.0, 4.0], [3.0, 0.0, 2.0]], kind)
    mean, centered = rs.center_curves(curves)
    assert_close(mean, [2.0, 1.0, 3.0])
    assert_close(centered, [[-1.0, 1.0, 1.0], [1.0, -1.0, -1.0]])
    cov = rs.compute_sample_covariance(centered, None, 2)
    assert_close(cov, [[1, -1, -1], [-1, 1, 1], [-1, 1, 1]])
    assert_close(rs.compute_sample_covariance(centered, None, 2, diag=True), [1, 1, 1])
    assert_close(
        rs.compute_sample_covariance(centered, None, 2, correction=1),
        [[2, -2, -2], [-2, 2, 2], [-2, 2, 2]],
    )
    interpolant = rs.LinearInterpolation1D(as_kind([1.0, 0.0, 0.4], kind), curves)
    query = as_kind([[0.0, 0.2], [0.7, 1.0]], kind)
    values = interpolant(query)
    assert shape(values) == (2, 2, 2)
    assert_close(to_list(values)[0], [[2.0, 3.0], [2.5, 1.0]])
    check_kind(values, kind)
    assert set(interpolant.state_dict()) == {"_x", "_y", "_bounds_error", "_fill_value"}
    with pytest.raises(ValueError, match="interpolation range"):
        interpolant(as_kind([-0.1], kind))
    fill = rs.LinearInterpolation1D(
        as_kind([0.0, 1.0], kind), as_kind([1.0, 2.0], kind), bounds_error=False, fill_value=8.0
    )
    assert_close(fill(as_kind([-0.2, 0.5, 1.2], kind)), [8.0, 1.5, 8.0])
    a = rs.ExperimentDataset(as_kind([[0.8], [0.2]], kind), as_kind([[1.0], [2.0]], kind))
    b = rs.ExperimentDataset(as_kind([[0.2], [0.9]], kind), as_kind([[3.0], [4.0]], kind))
    unique = rs.build_unique_inputs([a, b], as_kind([[0.4], [0.8]], kind))
    assert_close(unique.X_all, [[0.2], [0.4], [0.8], [0.9]])
    assert to_list(unique.experiment_indices[0]) == [2, 0]
    assert to_list(unique.experiment_indices[1]) == [0, 3]
    assert to_list(unique.forward_indices) == [1, 2]
    obj = SimpleNamespace()
    rs.instantiate_ard(obj, 3, curve_weights=as_kind([0.5, 1.0, 2.0], kind))
    assert obj.ard and to_list(obj.curve_weights) == [0.5, 1.0, 2.0]
    assert obj.curve_weights_constraint.lower_bound == 0.0
    replicated = as_kind([[[[0.1], [0.4]], [[0.1], [0.4]]]] * 3, kind)
    assert_close(rs.extract_slice_for_interp(replicated, 2), [[0.1, 0.4]] * 3)


@pytest.mark.parametrize("method", ["svd", "eigh"])
@pytest.mark.parametrize("rows,cols", [(9, 3), (3, 8)])
def test_official_orthogonal_basis_gram_invariant(
    official: object, method: str, rows: int, cols: int
) -> None:
    import torch

    A = torch.randn(
        2, rows, cols, generator=torch.Generator().manual_seed(918), dtype=torch.float64
    )
    a, b = rs.compute_orthogonal_basis(A, method), official.compute_orthogonal_basis(A, method)
    assert a.shape == b.shape == (2, min(rows, cols), cols)
    torch.testing.assert_close(
        a.transpose(-1, -2) @ a, b.transpose(-1, -2) @ b, rtol=1e-10, atol=1e-10
    )
    with pytest.raises(UnsupportedError, match="decomposition gradients"):
        rs.compute_orthogonal_basis(A.requires_grad_(), method, backend="native")


def test_official_covariance_and_interpolation_values_and_gradients(official: object) -> None:
    import torch

    generator = torch.Generator().manual_seed(913)
    for diag in (False, True):
        left = torch.randn(2, 1, 4, 3, generator=generator, dtype=torch.float64, requires_grad=True)
        right = torch.randn(
            1, 3, 4, 3 if diag else 2, generator=generator, dtype=torch.float64, requires_grad=True
        )
        value = rs.compute_sample_covariance(left, right, 4, diag=diag, correction=1)
        expected = official.compute_sample_covariance(left, right, 4, diag=diag, correction=1)
        weights = torch.randn(value.shape, generator=generator, dtype=torch.float64)
        ga = torch.autograd.grad((value * weights).sum(), (left, right))
        gb = torch.autograd.grad((expected * weights).sum(), (left, right))
        torch.testing.assert_close(value, expected, rtol=1e-12, atol=1e-12)
        for a, b in zip(ga, gb, strict=True):
            torch.testing.assert_close(a, b, rtol=1e-12, atol=1e-12)
    x = torch.tensor([0.0, 0.3, 0.8, 1.0], dtype=torch.float64, requires_grad=True)
    y = torch.randn(2, 3, 4, generator=generator, dtype=torch.float64, requires_grad=True)
    q = torch.tensor([[0.1, 0.5], [0.7, 0.9]], dtype=torch.float64, requires_grad=True)
    value = rs._interp1d_torch(x, y, q)
    expected = official._interp1d_torch(x, y, q)
    weights = torch.randn(value.shape, generator=generator, dtype=torch.float64)
    ga = torch.autograd.grad((value * weights).sum(), (x, y, q))
    gb = torch.autograd.grad((expected * weights).sum(), (x, y, q))
    torch.testing.assert_close(value, expected, rtol=1e-12, atol=1e-12)
    for a, b in zip(ga, gb, strict=True):
        torch.testing.assert_close(a, b, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("vectorized", [False, True])
@pytest.mark.parametrize("use_svd", [False, True, None])
def test_official_basis_and_mean_interpolants(
    official: object, vectorized: bool, use_svd: bool | None
) -> None:
    import torch

    generator = torch.Generator().manual_seed(401)
    X = torch.tensor([[0.0], [0.3], [0.7], [1.0]], dtype=torch.float64)
    Y = torch.randn(12, 4, 2, generator=generator, dtype=torch.float64)
    Q = torch.tensor([[0.1, 0.4], [0.6, 0.9]], dtype=torch.float64)
    fa, na, ma = rs.build_mean_interpolant(X, Y)
    fb, nb, mb = official.build_mean_interpolant(X, Y)
    assert na == nb == 2
    torch.testing.assert_close(ma, mb)
    torch.testing.assert_close(fa(Q), fb(Q))
    fa, na, sa = rs.build_basis_interpolant(
        X, Y, ard=False, use_svd=use_svd, vectorize_outputs=vectorized
    )
    fb, nb, sb = official.build_basis_interpolant(
        X, Y, ard=False, use_svd=use_svd, vectorize_outputs=vectorized
    )
    assert na == nb and sa == sb
    ba, bb = rs.compute_basis_matrix(fa, Q, 2), official.compute_basis_matrix(fb, Q, 2)
    # Basis signs/rotations may differ; full cross-output Gram is invariant.
    if vectorized:
        ba, bb = (
            fa(Q).permute(1, 0, 2, 3).reshape(na, -1),
            fb(Q).permute(1, 0, 2, 3).reshape(nb, -1),
        )
    torch.testing.assert_close(
        ba.transpose(-1, -2) @ ba, bb.transpose(-1, -2) @ bb, rtol=1e-10, atol=1e-10
    )
    weights = torch.ones(na, dtype=torch.float64, requires_grad=True)
    value = rs.compute_basis_matrix(fa, Q, 2, weights)
    assert torch.autograd.grad(value.square().sum(), weights)[0].abs().sum() > 0


def test_official_psd_projection_and_shrinkage_gradients(official: object) -> None:
    import torch

    a = torch.tensor([[-0.3, 0.4], [0.4, 1.2]], dtype=torch.float64)
    target = torch.tensor([[1.0, 0.2], [0.2, 2.0]], dtype=torch.float64, requires_grad=True)
    p = torch.randn(
        2, 2, generator=torch.Generator().manual_seed(394), dtype=torch.float64, requires_grad=True
    )
    symmetric = a + (p + p.T) * 0.02
    value, expected = rs.project_psd(symmetric, 0.1), official.project_psd(symmetric, 0.1)
    torch.testing.assert_close(value, expected)
    ga = torch.autograd.grad(value.square().sum(), p, retain_graph=True)[0]
    gb = torch.autograd.grad(expected.square().sum(), p, retain_graph=True)[0]
    torch.testing.assert_close(ga, gb, rtol=1e-9, atol=1e-10)
    alpha = torch.tensor(0.35, dtype=torch.float64, requires_grad=True)
    value = rs.trace_matched_shrinkage(symmetric, target, alpha)
    expected = official.trace_matched_shrinkage(symmetric, target, alpha)
    torch.testing.assert_close(value, expected)
    ga = torch.autograd.grad(value.square().sum(), (p, target, alpha), retain_graph=True)
    gb = torch.autograd.grad(expected.square().sum(), (p, target, alpha))
    for a, b in zip(ga, gb, strict=True):
        torch.testing.assert_close(a, b, rtol=1e-10, atol=1e-10)


@pytest.mark.parametrize("floor,gap", [(1e-10, 5e-14), (0.0, 0.0)])
def test_psd_clipping_boundary_gradients_are_symmetric(floor: float, gap: float) -> None:
    import torch

    if gap:
        value = torch.diag(torch.tensor([floor + gap, floor - gap], dtype=torch.float64))
        weights = torch.tensor([[1.0, 2.0], [2.0, 1.0]], dtype=torch.float64)
        expected = torch.tensor([[1.0, 1.0], [1.0, 0.0]], dtype=torch.float64)
    else:
        value = torch.diag(torch.tensor([0.0, 1.0], dtype=torch.float64))
        weights = expected = torch.eye(2, dtype=torch.float64)
    value.requires_grad_()
    projected = rs.project_psd(value, floor)
    actual = torch.autograd.grad((projected * weights).sum(), value)[0]
    eigenvalues, eigenvectors = torch.linalg.eigh(value)
    reference = (eigenvectors * eigenvalues.clamp_min(floor)) @ eigenvectors.T
    reference_gradient = torch.autograd.grad((reference * weights).sum(), value)[0]
    torch.testing.assert_close(actual, expected, rtol=1e-9, atol=1e-12)
    torch.testing.assert_close(actual, reference_gradient, rtol=1e-9, atol=1e-12)
    torch.testing.assert_close(actual, actual.T, rtol=0.0, atol=0.0)


def test_shrinkage_trace_clamp_boundary_gradient(official: object) -> None:
    import torch

    covariance = torch.tensor([[2e-12]], dtype=torch.float64, requires_grad=True)
    target = torch.tensor([[1e-12]], dtype=torch.float64, requires_grad=True)
    actual = rs.trace_matched_shrinkage(covariance, target, 0.3)
    reference = official.trace_matched_shrinkage(covariance, target, 0.3)
    gradients = torch.autograd.grad(actual.sum(), (covariance, target))
    reference_gradients = torch.autograd.grad(reference.sum(), (covariance, target))
    for value, expected in zip(gradients, reference_gradients, strict=True):
        torch.testing.assert_close(value, expected, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(gradients[0], torch.ones_like(covariance))
    torch.testing.assert_close(gradients[1], torch.zeros_like(target))


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("correction", [0, 1])
def test_synthetic_em_curves_reproduce_prior(kind: str, correction: int) -> None:
    mu = as_kind([2.0, -1.0, 0.3], kind)
    cov = as_kind([[1.0, 0.3, 0.1], [0.3, 0.7, -0.1], [0.1, -0.1, 0.4]], kind)
    curves = rs.em_prior_to_basis_curves(mu, cov, correction=correction)
    assert shape(curves) == (4, 3, 1)
    mean, centered = rs.center_curves(curves, curve_dim=0)
    assert_close(mean, [[2.0], [-1.0], [0.3]])
    centered = as_kind([[v[0] for v in row] for row in to_list(centered)], kind)
    assert_close(rs.compute_sample_covariance(centered, None, 4, correction=correction), cov)
    rank_one = rs.em_prior_to_basis_curves(mu, cov, num_modes=1)
    assert shape(rank_one) == (2, 3, 1)


def test_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="Unknown method"):
        rs.compute_orthogonal_basis([[1.0]], "unknown")
    with pytest.raises(ValueError, match="greater than correction"):
        rs.compute_sample_covariance([[1.0]], None, 1, correction=1)
    with pytest.raises(ValueError, match="strictly increasing"):
        rs.LinearInterpolation1D([0.0, 0.0], [1.0, 2.0])
    with pytest.raises(ValueError, match="alpha"):
        rs.trace_matched_shrinkage([[1.0]], [[1.0]], 1.1)
    with pytest.raises(ValueError, match="degenerate"):
        rs.em_prior_to_basis_curves([1.0], [[0.0]])
    with pytest.raises(ValueError, match="3-dim"):
        rs.validate_historical_curves_3d([[1.0]])
    with pytest.raises(UnsupportedError, match="input_transform"):
        rs.validate_no_transforms(object(), None, "EmpiricalGP")
    with pytest.raises(ValueError, match="empty"):
        rs.build_unique_inputs([], None)
    rs._interp1d_raise_out_of_bounds_error([0.2], [0.0, 1.0])
    with pytest.raises(ValueError, match="below"):
        rs._interp1d_raise_out_of_bounds_error([-1.0], [0.0, 1.0])


def test_parameter_registration_and_em_mean_gradient() -> None:
    import torch

    module = torch.nn.Module()
    rs.instantiate_ard(module, 3, dtype=torch.float64)
    assert module.curve_weights.requires_grad
    assert dict(module.named_parameters())["curve_weights"].shape == (3,)
    mu = torch.tensor([1.0, 2.0], dtype=torch.float64, requires_grad=True)
    curves = rs.em_prior_to_basis_curves(mu, torch.eye(2, dtype=torch.float64))
    torch.testing.assert_close(torch.autograd.grad(curves.sum(), mu)[0], torch.full_like(mu, 3.0))


def test_gpytorch_parameter_container_and_infinite_constraint_bounds() -> None:
    import torch
    from gpytorch.module import Module

    from botorch_rs.utils.constraints import NonTransformedInterval

    module = Module()
    rs.instantiate_ard(module, 3, dtype=torch.float64)
    assert module.curve_weights.requires_grad and module.ard
    constraint = module.curve_weights_constraint
    assert not constraint.enforced
    assert constraint.check(torch.tensor([0.0, 1.0]))
    assert not constraint.check(torch.tensor([-1.0, 1.0]))
    assert set(module.state_dict()) == {
        "curve_weights",
        "curve_weights_constraint.lower_bound",
        "curve_weights_constraint.upper_bound",
    }
    for lower, upper in [(0.0, math.inf), (-math.inf, 1.0), (-math.inf, math.inf)]:
        constraint = NonTransformedInterval(lower, upper)
        assert not constraint.enforced
        assert constraint.lower_bound == lower and constraint.upper_bound == upper
        assert tuple(constraint._rs.raw_bounds) == (lower, upper)
        assert constraint.transform(0.7) == constraint.inverse_transform(0.7) == 0.7


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("values", [1.0, [1.0], []])
def test_interpolation_invalid_value_shape_has_clear_error(kind: str, values: object) -> None:
    with pytest.raises(ValueError, match="final dimension matching"):
        rs.LinearInterpolation1D(as_kind([0.0, 1.0], kind), as_kind(values, kind))


def test_interpolation_nan_propagates_even_with_finite_fill(official: object) -> None:
    import torch

    x = torch.tensor([0.0, 1.0], dtype=torch.float64)
    y = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float64)
    q = torch.tensor([math.nan, -0.1, 0.5, 1.1], dtype=torch.float64)
    actual = rs.LinearInterpolation1D(x, y, bounds_error=False, fill_value=8.0)(q)
    expected = official._interp1d_torch(x, y, q, bounds_error=False, fill_value=8.0)
    torch.testing.assert_close(actual, expected, equal_nan=True)


@pytest.mark.parametrize(
    "name,arguments",
    [
        ("empirical_project_psd_vjp", (([1.0, 0.0, 0.0, 1.0], [2, 2]), ([1.0], [2, 2]))),
        ("empirical_interp", (([0.0, 1.0], [2]), ([1.0, 2.0], [2]), ([0.4], [99]))),
        ("empirical_interp", (([0.0, 1.0], [99]), ([1.0, 2.0], [2]), ([0.4], [1]))),
        ("empirical_interp", (([0.0, 1.0], [2]), ([1.0, 2.0], [99]), ([0.4], [1]))),
        (
            "empirical_interp_vjp",
            (([0.0, 1.0], [2]), ([1.0, 2.0], [2]), ([0.4], [1]), ([1.0], [99])),
        ),
        (
            "empirical_covariance_vjp",
            (([1.0, 2.0], [1, 2]), ([1.0, 2.0], [1, 2]), ([1.0], [2, 2]), 1),
        ),
        (
            "empirical_covariance_vjp",
            (([1.0, 2.0], [1, 2]), ([1.0, 2.0], [1, 2]), ([1.0] * 4, [4]), 1),
        ),
        ("empirical_covariance", (([1.0, 2.0], [2, 1, 1, 1]), ([1.0, 2.0], [1, 2, 1, 1]), 1)),
        ("empirical_em_basis", (([1.0], [99]), ([1.0], [1, 1]))),
        ("empirical_orthogonal_basis", (([], [2**63, 2]),)),
    ],
)
def test_low_level_array_shapes_are_validated(name: str, arguments: tuple) -> None:
    from botorch_rs import _botorch_rs

    # These used to panic in Rust or return data inconsistent with its shape.
    with pytest.raises(Exception, match="shape|dimensions|dimension"):
        getattr(_botorch_rs, name)(*arguments)

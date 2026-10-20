"""Public empirical utilities execute on device with complete Tensor graphs."""

# ruff: noqa: E402
from __future__ import annotations

import copy

import pytest

pytest.importorskip("torch")

import test_empirical_gp_utils as reference_helpers
import torch
from test_torch_em_empirical_gp import DEVICES

from botorch_rs.models.empirical_gps import utils


@pytest.fixture(scope="module")
def official_utils():
    return reference_helpers.official.__wrapped__()


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("method", ["svd", "eigh"])
@pytest.mark.parametrize("rows,cols", [(8, 3), (3, 8)])
def test_orthogonal_basis_gram_mixed_hessians(device, method, rows, cols):
    torch.manual_seed(224)
    A = torch.randn(2, rows, cols, dtype=torch.double, device=device, requires_grad=True)
    basis = utils.compute_orthogonal_basis(A, method)
    assert basis.shape == (2, min(rows, cols), cols)
    assert basis.device == A.device
    actual, expected = basis.mT @ basis, A.mT @ A
    torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-10)
    # Arbitrary non-invariant column weighting detects dropped basis terms.
    weight = torch.randn_like(actual)
    ga = torch.autograd.grad((actual * weight).square().sum(), A, create_graph=True)[0]
    ge = torch.autograd.grad((expected * weight).square().sum(), A, create_graph=True)[0]
    torch.testing.assert_close(ga, ge, rtol=1e-9, atol=1e-9)
    ha = torch.autograd.grad(ga.square().sum(), A)[0]
    he = torch.autograd.grad(ge.square().sum(), A)[0]
    torch.testing.assert_close(ha, he, rtol=1e-8, atol=1e-7)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("diag", [False, True])
def test_sample_covariance_broadcast_and_centering_hessians(device, diag):
    torch.manual_seed(771)
    Y = torch.randn(4, 3, dtype=torch.double, device=device, requires_grad=True)
    mean, centered = utils.center_curves(Y, 0)
    torch.testing.assert_close(mean, Y.mean(0))
    torch.testing.assert_close(centered, Y - Y.mean(0))
    U1 = centered.reshape(1, 4, 3)
    U2 = torch.randn(2, 4, 3 if diag else 2, dtype=Y.dtype, device=device, requires_grad=True)
    actual = utils.compute_sample_covariance(U1, U2, 7, diag=diag, correction=1)
    expected = ((U1 * U2).sum(-2) if diag else U1.mT @ U2) / 6
    torch.testing.assert_close(actual, expected)
    ga = torch.autograd.grad(actual.square().sum(), (Y, U2), create_graph=True)
    ge = torch.autograd.grad(expected.square().sum(), (Y, U2), create_graph=True)
    for a, b in zip(ga, ge, strict=True):
        torch.testing.assert_close(a, b)
    ha = torch.autograd.grad(sum(v.square().sum() for v in ga), (Y, U2))
    he = torch.autograd.grad(sum(v.square().sum() for v in ge), (Y, U2))
    for a, b in zip(ha, he, strict=True):
        torch.testing.assert_close(a, b)


@pytest.mark.parametrize("device", DEVICES)
def test_interpolation_repeated_backwards_knots_values_queries_and_module_state(device):
    x = torch.tensor([0.8, 0.0, 0.35, 1.0], device=device, dtype=torch.double, requires_grad=True)
    y = torch.tensor(
        [[0.2, 0.1, 0.7, 0.5], [0.4, 0.8, 0.3, 0.9]],
        device=device,
        dtype=torch.double,
        requires_grad=True,
    )
    query = torch.tensor(
        [[0.15, 0.52], [0.67, 0.91]], device=device, dtype=torch.double, requires_grad=True
    )
    interpolant = utils.LinearInterpolation1D(x, y)
    assert isinstance(interpolant, torch.nn.Module)
    assert set(interpolant.state_dict()) == {"_x", "_y", "_bounds_error", "_fill_value"}
    for _ in range(2):
        actual = interpolant(query)
        order = x.argsort()
        knots, values = x[order], y[:, order]
        indices = (torch.searchsorted(knots, query) - 1).clamp(0, 2)
        w = (query - knots[indices]) / (knots[indices + 1] - knots[indices])
        expected = values[:, indices] * (1 - w) + values[:, indices + 1] * w
        torch.testing.assert_close(actual, expected)
        ga = torch.autograd.grad(actual.square().sum(), (x, y, query), create_graph=True)
        ge = torch.autograd.grad(expected.square().sum(), (x, y, query), create_graph=True)
        for a, b in zip(ga, ge, strict=True):
            torch.testing.assert_close(a, b)
        ha = torch.autograd.grad(sum(v.square().sum() for v in ga), (x, y, query))
        he = torch.autograd.grad(sum(v.square().sum() for v in ge), (x, y, query))
        for a, b in zip(ha, he, strict=True):
            torch.testing.assert_close(a, b)
    clone = utils.LinearInterpolation1D(x.detach(), torch.zeros_like(y))
    clone.load_state_dict(interpolant.state_dict())
    torch.testing.assert_close(clone(query), interpolant(query))
    clone.to(dtype=torch.float32)
    assert clone(query.float()).dtype == torch.float32
    fill = utils.LinearInterpolation1D(x.detach(), y.detach(), bounds_error=False, fill_value=0.1)
    outside = query.new_tensor([-1.0, float("nan"), 2.0])
    result = fill(outside)
    assert torch.isnan(result[:, 1]).all()
    torch.testing.assert_close(result[:, 0], query.new_full((2,), float(torch.tensor(0.1))))
    # Saving before another forward must include current raw source data.
    with torch.no_grad():
        y.add_(0.2)
    state = interpolant.state_dict()
    torch.testing.assert_close(state["_y"], y[:, x.argsort()])
    # A migration must first adopt explicit changes to the public tables.
    with torch.no_grad():
        clone._y.fill_(2.0)
    clone.to(dtype=torch.double)
    torch.testing.assert_close(clone(query), torch.full_like(clone(query), 2.0))


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("compressed", [False, True])
def test_mean_and_basis_interpolant_end_to_end_history_hessians(device, compressed):
    torch.manual_seed(285)
    X = torch.tensor(
        [[0.0], [0.3], [0.7], [1.0]], device=device, dtype=torch.double, requires_grad=True
    )
    Y = torch.randn(9, 4, 2, dtype=X.dtype, device=device, requires_grad=True)
    Q = X.new_tensor([[0.1, 0.5], [0.6, 0.9]], requires_grad=True)
    mean_f, outputs, mean = utils.build_mean_interpolant(X, Y)
    assert outputs == 2
    torch.testing.assert_close(mean, Y.mean(0).mT)
    basis_f, curves, used = utils.build_basis_interpolant(
        X, Y, ard=False, use_svd=compressed, vectorize_outputs=True
    )
    assert used is compressed
    weights = X.new_ones(curves, requires_grad=True)
    basis = utils.compute_basis_matrix(basis_f, Q, 2, curve_weights=weights)
    value = (
        utils.compute_sample_covariance(basis, None, 9).square().sum() + mean_f(Q).square().sum()
    )
    grads = torch.autograd.grad(value, (X, Y, Q, weights), create_graph=True)
    second = torch.autograd.grad(sum(v.square().sum() for v in grads), (X, Y, Q, weights))
    assert all(torch.isfinite(v).all() for v in (*grads, *second))


@pytest.mark.parametrize("device", DEVICES)
def test_unique_metadata_psd_shrinkage_and_synthesized_curves(device):
    X = torch.tensor(
        [[0.4, 0.2], [0.1, 0.5]], device=device, dtype=torch.double, requires_grad=True
    )
    forward = X.new_tensor([[0.1, 0.5], [0.9, 0.6]], requires_grad=True)
    result = utils.build_unique_inputs([utils.ExperimentDataset(X, X[:, :1])], forward)
    assert result.X_all.device.type == device
    assert not result.X_all.requires_grad  # discrete, fixed membership metadata
    torch.testing.assert_close(result.X_all[result.experiment_indices[0]], X)
    torch.testing.assert_close(result.X_all[result.forward_indices], forward)
    assert result.experiment_indices[0].dtype == torch.long
    covariance = torch.diag(X.new_tensor([-0.4, -0.4, 1.2])).requires_grad_()
    projected = utils.project_psd(covariance, 0.1)
    torch.testing.assert_close(projected, torch.diag(X.new_tensor([0.1, 0.1, 1.2])))
    gradient = torch.autograd.grad(projected.square().sum(), covariance, create_graph=True)[0]
    assert torch.isfinite(
        torch.autograd.grad(gradient.square().sum(), covariance, retain_graph=True)[0]
    ).all()
    target = (torch.eye(3, device=device, dtype=X.dtype) + 0.2).requires_grad_()
    alpha = X.new_tensor(0.3, requires_grad=True)
    blended = utils.trace_matched_shrinkage(projected, target, alpha)
    grads = torch.autograd.grad(
        blended.square().sum(), (covariance, target, alpha), create_graph=True
    )
    assert all(torch.isfinite(v).all() for v in grads)
    mu = X.new_tensor([0.2, 0.4, 0.6], requires_grad=True)
    Sigma = X.new_tensor([[1.1, 0.2, 0.1], [0.2, 0.8, 0.3], [0.1, 0.3, 0.6]], requires_grad=True)
    curves = utils.em_prior_to_basis_curves(mu, Sigma, correction=1)
    average, centered = utils.center_curves(curves, curve_dim=0)
    reconstructed = utils.compute_sample_covariance(
        centered.squeeze(-1), None, curves.shape[0], correction=1
    )
    torch.testing.assert_close(average.squeeze(-1), mu)
    torch.testing.assert_close(reconstructed, Sigma, rtol=1e-10, atol=1e-10)
    loss = average.square().sum() + reconstructed.square().sum()
    first = torch.autograd.grad(loss, (mu, Sigma), create_graph=True)
    second = torch.autograd.grad(sum(v.square().sum() for v in first), (mu, Sigma))
    assert all(torch.isfinite(v).all() for v in second)


@pytest.mark.parametrize("device", DEVICES)
def test_ard_registration_and_constraint_checks_stay_on_device(device):
    from gpytorch.module import Module

    module = Module()
    utils.instantiate_ard(module, 3, dtype=torch.double, device=device)
    assert module.curve_weights.device.type == device
    assert module.curve_weights.requires_grad
    assert module.curve_weights_constraint.check(module.curve_weights)
    assert not module.curve_weights_constraint.check(-module.curve_weights)
    old = copy.deepcopy(module.state_dict())
    module.curve_weights.data.fill_(0.7)
    module.load_state_dict(old)
    torch.testing.assert_close(module.curve_weights, torch.ones_like(module.curve_weights))


@pytest.mark.parametrize("device", DEVICES)
def test_pinned_official_wide_eigh_artificial_nullspace_gradient_bug(device, official_utils):
    A = torch.zeros(3, 8, dtype=torch.double, device=device)
    A[:, :3] = torch.diag(A.new_tensor([1.0, 2.0, 3.0]))
    A.requires_grad_()
    original = official_utils.compute_orthogonal_basis(A, "eigh")
    original_gradient = torch.autograd.grad((original.mT @ original).square().sum(), A)[0]
    assert not torch.isfinite(original_gradient).all()
    basis = utils.compute_orthogonal_basis(A, "eigh")
    gradient = torch.autograd.grad((basis.mT @ basis).square().sum(), A, create_graph=True)[0]
    expected = 4 * A @ A.mT @ A
    torch.testing.assert_close(gradient, expected)
    actual_hessian = torch.autograd.grad(gradient.square().sum(), A)[0]
    expected_hessian = torch.autograd.grad(expected.square().sum(), A)[0]
    torch.testing.assert_close(actual_hessian, expected_hessian)


@pytest.mark.parametrize("device", DEVICES)
def test_pinned_official_sorted_interpolation_reuses_construction_graph(device, official_utils):
    x = torch.tensor([1.0, 0.0, 0.4], dtype=torch.double, device=device, requires_grad=True)
    y = torch.tensor([[0.5, 0.1, 0.8]], dtype=torch.double, device=device, requires_grad=True)
    query = x.new_tensor([0.2, 0.7])
    original = official_utils.LinearInterpolation1D(x, y)
    torch.autograd.grad(original(query).sum(), (x, y))
    with pytest.raises(RuntimeError, match="backward through the graph a second time"):
        torch.autograd.grad(original(query).sum(), (x, y))
    repaired = utils.LinearInterpolation1D(x, y)
    for _ in range(2):
        gradients = torch.autograd.grad(repaired(query).square().sum(), (x, y), create_graph=True)
        assert all(torch.isfinite(value).all() for value in gradients)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("assume_sorted", [False, True])
@pytest.mark.parametrize("save_before_update", [False, True])
def test_live_sorted_interpolation_sources_survive_external_updates(
    device, assume_sorted, save_before_update
):
    order = [0, 1, 2] if assume_sorted else [1, 2, 0]
    x = torch.tensor(
        [0.0, 0.4, 1.0] if assume_sorted else [1.0, 0.0, 0.4],
        dtype=torch.double,
        device=device,
        requires_grad=True,
    )
    y = x.new_tensor([[0.2, 0.7, -0.1]], requires_grad=True)
    query = x.new_tensor([0.2, 0.7], requires_grad=True)
    interpolant = utils.LinearInterpolation1D(x, y, assume_sorted=assume_sorted)
    # Public state is a snapshot. Mutating caller-owned live source tensors
    # must not be mistaken for an explicit replacement of those state tables.
    for iteration in range(2):
        if save_before_update:
            interpolant.state_dict()
        with torch.no_grad():
            x.add_(0.01)
            y.add_(0.03)
        xs, ys = x[order], y[..., order]
        expected = torch.stack(
            [
                ys[..., 0] + (ys[..., 1] - ys[..., 0]) * (query[0] - xs[0]) / (xs[1] - xs[0]),
                ys[..., 1] + (ys[..., 2] - ys[..., 1]) * (query[1] - xs[1]) / (xs[2] - xs[1]),
            ],
            -1,
        )
        result = interpolant(query)
        torch.testing.assert_close(result, expected)
        gradients = torch.autograd.grad(result.square().sum(), (x, y, query), create_graph=True)
        reference = torch.autograd.grad(expected.square().sum(), (x, y, query), create_graph=True)
        for actual, wanted in zip(gradients, reference, strict=True):
            torch.testing.assert_close(actual, wanted)
        second = torch.autograd.grad(sum(g.square().sum() for g in gradients), (x, y, query))
        reference_second = torch.autograd.grad(
            sum(g.square().sum() for g in reference), (x, y, query)
        )
        for actual, wanted in zip(second, reference_second, strict=True):
            torch.testing.assert_close(actual, wanted)

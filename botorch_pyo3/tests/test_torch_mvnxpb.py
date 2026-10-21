"""Stateful multivariate integration preserves device and complete graphs."""

# ruff: noqa: E402
from __future__ import annotations

from contextlib import ExitStack
from unittest.mock import patch

import pytest

torch = pytest.importorskip("torch")

from botorch_rs.utils.probability.linalg import PivotedCholesky, augment_cholesky
from botorch_rs.utils.probability.mvnxpb import MVNXPB

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


def _problem(device, n=4, dtype=torch.double, batch=2):
    generator = torch.Generator().manual_seed(31 + n)
    factor = torch.randn(batch, n, n, dtype=dtype, generator=generator).to(device)
    covariance = factor @ factor.transpose(-1, -2) + torch.eye(n, device=device, dtype=dtype) * 0.8
    lower = -torch.linspace(0.2, 1.5, n, device=device, dtype=dtype)
    upper = torch.linspace(0.8, 1.3, n, device=device, dtype=dtype)
    bounds = torch.stack((lower, upper), -1).expand(batch, n, 2).clone()
    bounds[1] = bounds[1].flip(0)
    return covariance.requires_grad_(), bounds.requires_grad_()


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("n", [1, 2, 3, 4, 5, 6])
def test_mvnxpb_batched_values_and_memberwise_reference_first_gradients(device, dtype, n):
    reference = pytest.importorskip("botorch.utils.probability.mvnxpb")
    covariance, bounds = _problem(device, n, dtype)
    solver = MVNXPB(covariance, bounds)
    actual = solver.solve()
    actual_gradients = torch.autograd.grad(actual.sum(), (covariance, bounds), create_graph=True)
    expected_values = []
    expected_covariance = []
    expected_bounds = []
    for i in range(covariance.shape[0]):
        # The reference's batched in-place pivot autograd is known incorrect;
        # solve each reference member separately, keeping the source unmodified.
        c = covariance[i].detach().requires_grad_()
        b = bounds[i].detach().requires_grad_()
        value = reference.MVNXPB(c, b).solve()
        gc, gb = torch.autograd.grad(value, (c, b))
        expected_values.append(value)
        expected_covariance.append(gc)
        expected_bounds.append(gb)
    tolerance = 3e-4 if dtype == torch.float32 else 2e-8
    for a, b in (
        (actual, torch.stack(expected_values)),
        (actual_gradients[0], torch.stack(expected_covariance)),
        (actual_gradients[1], torch.stack(expected_bounds)),
    ):
        torch.testing.assert_close(a, b, atol=tolerance, rtol=tolerance)
    second = torch.autograd.grad(
        sum(g.square().sum() for g in actual_gradients), (covariance, bounds)
    )
    assert all(bool(g.isfinite().all()) and g.device == covariance.device for g in second)
    torch.testing.assert_close(solver.solve(), actual)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("partial", [False, True])
def test_mvnxpb_complete_and_resumed_mixed_gradgradcheck(device, partial):
    covariance, bounds = _problem(device, n=3, batch=2)
    covariance, bounds = covariance[0], bounds[0]

    def function(c, b):
        solver = MVNXPB(c, b)
        if partial:
            solver.solve(num_steps=1)
        return solver.solve()

    assert torch.autograd.gradcheck(function, (covariance, bounds), atol=3e-5, rtol=3e-4)
    assert torch.autograd.gradgradcheck(function, (covariance, bounds), atol=3e-5, rtol=3e-4)


@pytest.mark.parametrize("device", DEVICES)
def test_partial_snapshots_batch_views_and_pivot_graphs_do_not_mutate_old_forward(device):
    covariance, bounds = _problem(device, n=5)
    solver = MVNXPB(covariance, bounds)
    initial_tril = solver.piv_chol.tril
    partial = solver.solve(num_steps=2)
    saved_tril = solver.piv_chol.tril
    assert solver.piv_chol.tril is not initial_tril
    solver.pivot_(torch.tensor([4, 3], device=device))
    full = solver.solve()
    assert solver.piv_chol.tril is not saved_tril
    partial_gradient = torch.autograd.grad(partial.sum(), (covariance, bounds), retain_graph=True)
    full_gradient = torch.autograd.grad(full.sum(), (covariance, bounds), create_graph=True)
    second = torch.autograd.grad(sum(g.square().sum() for g in full_gradient), (covariance, bounds))
    assert all(bool(g.isfinite().all()) for g in (*partial_gradient, *full_gradient, *second))
    views = solver[:1].concat(solver[1:], dim=0)
    torch.testing.assert_close(views.log_prob, full)
    torch.testing.assert_close(solver.clone().log_prob, full)
    assert not solver.detach().log_prob.requires_grad


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("disable_pivoting", [False, True])
def test_augmented_solver_reference_state_and_full_gradient_graph(device, disable_pivoting):
    reference = pytest.importorskip("botorch.utils.probability.mvnxpb")
    covariance, bounds = _problem(device, n=5)
    initial = MVNXPB(covariance[..., :2, :2], bounds[..., :2, :])
    initial.solve()
    cross = covariance[..., 2:, :2].gather(-1, initial.perm.unsqueeze(-2).expand(2, 3, 2))
    augmented = initial.augment(
        covariance[..., 2:, 2:], bounds[..., 2:, :], cross, disable_pivoting=disable_pivoting
    )
    actual = augmented.solve()
    expected = []
    for i in range(2):
        init = reference.MVNXPB(covariance[i, :2, :2].detach(), bounds[i, :2].detach())
        init.solve()
        result = init.augment(
            covariance[i, 2:, 2:].detach(),
            bounds[i, 2:].detach(),
            cross[i].detach(),
            disable_pivoting=disable_pivoting,
        )
        expected.append(result.solve())
    torch.testing.assert_close(actual, torch.stack(expected), atol=2e-10, rtol=2e-10)
    first = torch.autograd.grad(actual.sum(), (covariance, bounds), create_graph=True)
    second = torch.autograd.grad(sum(g.square().sum() for g in first), (covariance, bounds))
    assert all(bool(g.isfinite().all()) for g in (*first, *second))


@pytest.mark.parametrize("device", DEVICES)
def test_mvnxpb_and_partial_cholesky_no_host_conversions_even_without_grad(device):
    covariance, bounds = _problem(device, n=4)
    with ExitStack() as stack:
        for name in ("cpu", "numpy", "tolist"):
            stack.enter_context(
                patch.object(torch.Tensor, name, side_effect=AssertionError("host transfer"))
            )
        solver = MVNXPB(covariance, bounds)
        value = solver.solve()
        first = torch.autograd.grad(value.sum(), (covariance, bounds), create_graph=True)
        second = torch.autograd.grad(sum(g.square().sum() for g in first), (covariance, bounds))
        with torch.no_grad():
            pc = PivotedCholesky(
                0, covariance.tril(), torch.arange(4, device=device).expand(2, 4).clone()
            )
            pc.pivot_(torch.tensor([3, 2], device=device))
            pc.update_()
            factor = torch.linalg.cholesky(covariance)
            augmented = augment_cholesky(
                factor[..., :2, :2], covariance[..., 2:, 2:], Kba=covariance[..., 2:, :2]
            )
            torch.testing.assert_close(augmented, factor)
    assert all(bool(g.isfinite().all()) for g in (*first, *second))


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("n", [2, 3, 4])
def test_repeated_gaussian_variables_reduce_to_their_interval_intersection(device, n):
    covariance = torch.ones(n, n, device=device, dtype=torch.double)
    bounds = torch.tensor([[-1.2, 1.1]] * n, device=device, dtype=torch.double)
    bounds[-1] = bounds.new_tensor([-0.7, 0.8])
    value = MVNXPB(covariance, bounds).solve()
    expected = (torch.special.ndtr(bounds[-1, 1]) - torch.special.ndtr(bounds[-1, 0])).log()
    torch.testing.assert_close(value, expected, atol=1e-12, rtol=1e-12)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_strongly_correlated_one_sided_bounds_have_finite_full_graph(device, dtype):
    n = 4
    covariance = (
        torch.full((n, n), 0.98, device=device, dtype=dtype)
        + 0.02 * torch.eye(n, device=device, dtype=dtype)
    ).requires_grad_()
    upper = torch.linspace(-0.2, 0.8, n, device=device, dtype=dtype, requires_grad=True)
    bounds = torch.stack((torch.full_like(upper, -torch.inf), upper), -1)
    value = MVNXPB(covariance, bounds).solve()
    first = torch.autograd.grad(value, (covariance, upper), create_graph=True)
    second = torch.autograd.grad(sum(g.square().sum() for g in first), (covariance, upper))
    assert bool(value.isfinite())
    assert all(bool(g.isfinite().all()) for g in (*first, *second))

"""Bivariate Gaussian integrators retain CPU/CUDA higher-order graphs."""

# ruff: noqa: E402
from __future__ import annotations

import importlib
import math
from contextlib import ExitStack
from unittest.mock import patch

import pytest

torch = pytest.importorskip("torch")

from botorch_rs.utils.probability.bvn import bvn, bvnmom, bvnu

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


def _density(r, h, k):
    complement = 1 - r.square()
    return (-(h.square() + k.square() - 2 * r * h * k) / (2 * complement)).exp() / (
        2 * math.pi * complement.sqrt()
    )


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_bivariate_values_first_and_second_graphs_match_reference(device, dtype):
    pytest.importorskip("botorch")
    reference = importlib.import_module("botorch.utils.probability.bvn")
    r = torch.tensor(
        [-0.999, -0.96, -0.8, -0.5, -0.1, 0.1, 0.4, 0.8, 0.96, 0.999],
        device=device,
        dtype=dtype,
        requires_grad=True,
    )
    h = torch.linspace(-0.3, 0.7, r.numel(), device=device, dtype=dtype).requires_grad_()
    k = torch.linspace(0.2, -0.4, r.numel(), device=device, dtype=dtype).requires_grad_()
    actual, expected = bvnu(r, h, k), reference.bvnu(r, h, k)
    a = torch.autograd.grad(actual.sum(), (r, h, k), create_graph=True)
    b = torch.autograd.grad(expected.sum(), (r, h, k), create_graph=True)
    aa = torch.autograd.grad(sum(g.square().sum() for g in a), (r, h, k))
    bb = torch.autograd.grad(sum(g.square().sum() for g in b), (r, h, k))
    tolerance = 3e-4 if dtype == torch.float32 else 3e-10
    torch.testing.assert_close(actual, expected, atol=tolerance, rtol=tolerance)
    for ao, bo in ((a, b), (aa, bb)):
        for ag, bg in zip(ao, bo, strict=True):
            torch.testing.assert_close(ag, bg, atol=tolerance, rtol=tolerance)
    assert actual.device == r.device and actual.dtype == r.dtype


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("correlation", [-0.98, -0.5, 0.0, 0.4, 0.98])
def test_correlation_gradient_matches_plackett_density_and_gradgrad(device, correlation):
    r = torch.tensor([correlation], device=device, dtype=torch.double, requires_grad=True)
    h = torch.tensor([0.4], device=device, dtype=torch.double, requires_grad=True)
    k = torch.tensor([-0.2], device=device, dtype=torch.double, requires_grad=True)
    probability = bvnu(r, h, k)
    gradient = torch.autograd.grad(probability.sum(), r, create_graph=True)[0]
    density = _density(r, h, k)
    torch.testing.assert_close(gradient, density, atol=2e-12, rtol=2e-10)
    a = torch.autograd.grad(gradient.sum(), (r, h, k), retain_graph=True)
    b = torch.autograd.grad(density.sum(), (r, h, k))
    for actual, expected in zip(a, b, strict=True):
        torch.testing.assert_close(actual, expected, atol=2e-10, rtol=2e-9)
    assert torch.autograd.gradcheck(bvnu, (r, h, k))
    assert torch.autograd.gradgradcheck(bvnu, (r, h, k))


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("correlation", [-0.97, -0.3, 0.0, 0.6, 0.97])
def test_box_and_conditional_moments_gradcheck(device, correlation):
    arguments = tuple(
        torch.tensor([v], device=device, dtype=torch.double, requires_grad=True)
        for v in (correlation, -0.8, -0.6, 0.5, 0.9)
    )

    def function(*values):
        probability = bvn(*values)
        moments = bvnmom(*values, p=probability)
        return torch.stack((probability, *moments), -1)

    assert torch.autograd.gradcheck(function, arguments, atol=2e-5, rtol=2e-4)
    assert torch.autograd.gradgradcheck(function, arguments, atol=2e-5, rtol=2e-4)


@pytest.mark.parametrize("device", DEVICES)
def test_infinite_bounds_singular_values_and_no_host_transfers(device):
    r = torch.tensor(
        [0.0, 0.5, -0.97, 1.0, -1.0], device=device, dtype=torch.double, requires_grad=True
    )
    h = torch.tensor(
        [-torch.inf, 0.1, -0.2, 0.1, -0.2], device=device, dtype=torch.double, requires_grad=True
    )
    k = torch.tensor(
        [0.3, torch.inf, -torch.inf, -0.4, 0.4],
        device=device,
        dtype=torch.double,
        requires_grad=True,
    )
    with ExitStack() as stack:
        for name in ("cpu", "numpy", "tolist"):
            stack.enter_context(
                patch.object(torch.Tensor, name, side_effect=AssertionError("host transfer"))
            )
        value = bvnu(r, h, k)
        gradients = torch.autograd.grad(value.sum(), (r, h, k), create_graph=True)
        second = torch.autograd.grad(sum(g.square().sum() for g in gradients), (r, h, k))
    expected = torch.stack(
        (
            torch.special.ndtr(-k[0]),
            r.new_tensor(0),
            torch.special.ndtr(-h[2]),
            torch.special.ndtr(-h[3]),
            (torch.special.ndtr(-k[4]) - torch.special.ndtr(h[4])).clamp_min(0),
        )
    )
    torch.testing.assert_close(value, expected)
    assert all(bool(g.isfinite().all()) for g in (*gradients, *second))


def test_unmodified_reference_independence_shortcut_drops_correlation_gradient():
    pytest.importorskip("botorch")
    reference = importlib.import_module("botorch.utils.probability.bvn")
    r = torch.tensor([0.0], dtype=torch.double, requires_grad=True)
    h = torch.tensor([0.4], dtype=torch.double, requires_grad=True)
    k = torch.tensor([-0.2], dtype=torch.double, requires_grad=True)
    expected_density = _density(r, h, k)
    source_value = reference.bvnu(r, h, k)
    source_gradient = torch.autograd.grad(source_value.sum(), r, allow_unused=True)[0]
    assert source_gradient is None
    step = 1e-5
    finite_difference = (
        reference.bvnu(r.detach() + step, h, k) - reference.bvnu(r.detach() - step, h, k)
    ) / (2 * step)
    actual_gradient = torch.autograd.grad(bvnu(r, h, k).sum(), r)[0]
    torch.testing.assert_close(actual_gradient, expected_density)
    torch.testing.assert_close(actual_gradient, finite_difference, atol=1e-10, rtol=1e-9)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("correlation,other", [(0.98, 0.4), (-0.98, -0.4)])
def test_equal_signed_bounds_have_smooth_correct_hessians(device, correlation, other):
    args = tuple(
        torch.tensor([v], device=device, dtype=torch.double, requires_grad=True)
        for v in (correlation, 0.4, other)
    )
    assert torch.autograd.gradcheck(bvnu, args)
    assert torch.autograd.gradgradcheck(bvnu, args)
    r, h, k = args
    dh = torch.autograd.grad(bvnu(*args).sum(), h, create_graph=True)[0]
    mixed = torch.autograd.grad(dh.sum(), k)[0]
    torch.testing.assert_close(mixed, _density(r, h, k), atol=1e-12, rtol=1e-12)


@pytest.mark.parametrize("device", DEVICES)
def test_bivariate_forward_mode_and_vmap_keep_analytic_derivatives(device):
    r = torch.tensor([0.98, 0.4], device=device, dtype=torch.double)
    h = torch.tensor([0.4, -0.2], device=device, dtype=torch.double)
    k = torch.tensor([0.4, 0.3], device=device, dtype=torch.double)
    value, tangent = torch.func.jvp(
        bvnu, (r, h, k), (torch.ones_like(r), torch.zeros_like(h), torch.zeros_like(k))
    )
    torch.testing.assert_close(tangent, _density(r, h, k))
    torch.testing.assert_close(torch.vmap(bvnu)(r, h, k), value)
    jacobian = torch.func.jacrev(bvnu, argnums=0)(r, h, k)
    torch.testing.assert_close(jacobian, _density(r, h, k).diag())


def test_unmodified_reference_equal_bound_hessian_fails_density_and_finite_differences():
    pytest.importorskip("botorch")
    reference = importlib.import_module("botorch.utils.probability.bvn")
    r, h, k = (torch.tensor([v], dtype=torch.double, requires_grad=True) for v in (0.98, 0.4, 0.4))
    first = torch.autograd.grad(reference.bvnu(r, h, k).sum(), h, create_graph=True)[0]
    mixed = torch.autograd.grad(first.sum(), k)[0]
    correct = _density(r, h, k)
    assert bool((mixed - correct).abs() > 1e-3)
    step = 1e-5
    samples = []
    for sign in (-1, 1):
        value = reference.bvnu(r.detach(), h, k.detach() + sign * step)
        samples.append(torch.autograd.grad(value.sum(), h)[0])
    finite_difference = (samples[1] - samples[0]) / (2 * step)
    torch.testing.assert_close(finite_difference, correct, atol=2e-8, rtol=2e-8)

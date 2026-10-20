"""Repeated linear-constraint terms add, independently of source indexed assignment."""

import numpy as np
import pytest
import torch
from test_torch_acq_discrete import DEVICES, no_host

from botorch_rs import _botorch_rs as native
from botorch_rs._arrays import to_flat
from botorch_rs.optim.parameter_constraints import project_to_equality_constraints


def closed_form(X, coefficients, rhs, scalar=False):
    if scalar:
        row = torch.stack((2 * coefficients, X.new_zeros(()), coefficients))
    else:
        row = torch.stack((coefficients[0] + coefficients[1], X.new_zeros(()), coefficients[2]))
    residual = rhs - (X * row).sum(-1, keepdim=True)
    return X + residual * row / row.square().sum()


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("scalar", [False, True])
def test_device_duplicate_sum_values_all_derivatives_and_declared_feasibility(device, scalar):
    X = torch.tensor(
        [[0.2, 0.3, 0.8], [0.6, 0.4, 0.1]], dtype=torch.double, device=device, requires_grad=True
    )
    coefficients = X.new_tensor(0.7 if scalar else [1.2, -0.7, 0.8], requires_grad=True)
    rhs = X.new_tensor(0.4, requires_grad=True)
    index = torch.tensor([0, 0, 2], device=device)

    def actual(x, coefficient, right):
        return project_to_equality_constraints(x, [(index, coefficient, right)])

    expected = closed_form(X, coefficients, rhs, scalar)
    expected_gradients = torch.autograd.grad(expected.square().sum(), (X, coefficients, rhs))
    with no_host():
        projected = actual(X, coefficients, rhs)
        gradients = torch.autograd.grad(
            projected.square().sum(), (X, coefficients, rhs), create_graph=True
        )
        lhs = (projected[..., index] * coefficients).sum(-1)
        assert torch.autograd.gradcheck(actual, (X, coefficients, rhs), fast_mode=True)
        assert torch.autograd.gradgradcheck(actual, (X, coefficients, rhs), fast_mode=True)
    torch.testing.assert_close(projected, expected, atol=1e-13, rtol=1e-13)
    torch.testing.assert_close(lhs, rhs.expand_as(lhs), atol=1e-13, rtol=1e-13)
    for actual_g, expected_g in zip(gradients, expected_gradients, strict=True):
        torch.testing.assert_close(actual_g, expected_g, atol=1e-12, rtol=1e-12)


@pytest.mark.parametrize("kind", ["list", "numpy", "torch"])
@pytest.mark.parametrize("scalar", [False, True])
def test_array_duplicate_sum_projection_matches_independent_closed_form(kind, scalar):
    X = torch.tensor(
        [[[0.2, 0.3, 0.8], [0.6, 0.4, 0.1]], [[0.7, 0.5, 0.2], [0.1, 0.2, 0.9]]], dtype=torch.double
    )
    coefficient = torch.tensor(0.7 if scalar else [1.2, -0.7, 0.8], dtype=torch.double)
    convert = {
        "list": lambda tensor: tensor.tolist(),
        "numpy": lambda tensor: np.asarray(tensor),
        "torch": lambda tensor: tensor,
    }[kind]
    actual = project_to_equality_constraints(
        convert(X), [(convert(torch.tensor([0, 0, 2])), convert(coefficient), 0.4)]
    )
    torch.testing.assert_close(
        torch.as_tensor(actual),
        closed_form(X, coefficient, X.new_tensor(0.4), scalar),
        atol=1e-12,
        rtol=1e-12,
        check_dtype=False,
    )


def test_native_duplicate_sum_projection_vjp_matches_formula_and_all_finite_differences():
    X = torch.tensor([[0.2, 0.3, 0.8], [0.6, 0.4, 0.1]], dtype=torch.double, requires_grad=True)
    coefficients = X.new_tensor([1.2, -0.7, 0.8], requires_grad=True)
    rhs = X.new_tensor(0.4, requires_grad=True)
    adjoint = X.new_tensor([[0.7, -0.3, 1.1], [0.4, 0.8, -0.2]])
    expected = closed_form(X, coefficients, rhs)
    expected_gradients = torch.autograd.grad((expected * adjoint).sum(), (X, coefficients, rhs))

    def project(x, coefficient, right):
        data, shape = native.project_linear_equalities(
            to_flat(x), [([0, 0, 2], coefficient.detach().tolist(), float(right.detach()))]
        )
        return torch.tensor(data, dtype=X.dtype).reshape(shape)

    actual = project(X, coefficients, rhs)
    (gx, shape), gc, gr = native.equality_projection_vjp(
        to_flat(X),
        [([0, 0, 2], coefficients.detach().tolist(), float(rhs.detach()))],
        to_flat(adjoint),
    )
    gradients = (
        torch.tensor(gx, dtype=X.dtype).reshape(shape),
        torch.tensor(gc[0], dtype=X.dtype),
        X.new_tensor(gr[0]),
    )
    torch.testing.assert_close(actual, expected, atol=1e-13, rtol=1e-13)
    for actual_g, expected_g in zip(gradients, expected_gradients, strict=True):
        torch.testing.assert_close(actual_g, expected_g, atol=1e-12, rtol=1e-12)
    arguments = (X.detach(), coefficients.detach(), rhs.detach())
    step = 1e-6
    for argument_index, argument in enumerate(arguments):
        for element in range(argument.numel()):
            perturbation = torch.zeros_like(argument).reshape(-1)
            perturbation[element] = step
            perturbation = perturbation.reshape(argument.shape)
            plus, minus = list(arguments), list(arguments)
            plus[argument_index] = argument + perturbation
            minus[argument_index] = argument - perturbation
            numerical = ((project(*plus) - project(*minus)) * adjoint).sum() / (2 * step)
            torch.testing.assert_close(
                gradients[argument_index].reshape(-1)[element], numerical, atol=1e-9, rtol=1e-9
            )

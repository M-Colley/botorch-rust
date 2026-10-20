"""Device equality projection has actual complete point/coefficient/RHS graphs."""

import pytest
import torch
from test_torch_acq_discrete import DEVICES, no_host

from botorch_rs.optim.parameter_constraints import (
    evaluate_feasibility,
    project_to_equality_constraints,
)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.double])
@pytest.mark.parametrize("batch", [(), (2,), (2, 3)])
@pytest.mark.parametrize("scalar", [False, True])
def test_device_projection_matches_reference_values_and_every_first_derivative(
    device, dtype, batch, scalar
):
    from botorch.optim.parameter_constraints import project_to_equality_constraints as reference

    X = (
        torch.arange(6, dtype=dtype, device=device)
        .reshape(2, 3)
        .expand(*batch, 2, 3)
        .clone()
        .requires_grad_()
    )
    coefficients = X.new_tensor(1.3 if scalar else [1.3, -0.7], requires_grad=True)
    rhs = X.new_tensor(0.8, requires_grad=True)
    equality = [(torch.tensor([0, 2], device=device), coefficients, rhs)]
    adjoint = X.detach().square() + 0.3
    expected = reference(X, equality)
    expected_gradients = torch.autograd.grad((expected * adjoint).sum(), (X, coefficients, rhs))
    with no_host():
        actual = project_to_equality_constraints(X, equality)
        gradients = torch.autograd.grad(
            (actual * adjoint).sum(), (X, coefficients, rhs), create_graph=True
        )
    tolerance = 2e-5 if dtype == torch.float32 else 1e-11
    torch.testing.assert_close(actual, expected, atol=tolerance, rtol=tolerance)
    for actual_g, expected_g in zip(gradients, expected_gradients, strict=True):
        torch.testing.assert_close(actual_g, expected_g, atol=tolerance, rtol=tolerance)
    assert actual.device == X.device and actual.dtype == dtype


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("duplicate", [False, True])
def test_device_projection_complete_gradcheck_and_gradgradcheck(device, duplicate):
    X = torch.tensor(
        [[[0.2, 0.3, 0.8], [0.6, 0.4, 0.1]]], dtype=torch.double, device=device, requires_grad=True
    )
    coefficients = X.new_tensor([1.2, -0.7, 0.8] if duplicate else [1.2, -0.7], requires_grad=True)
    second_coefficients = X.new_tensor([0.6, 1.1], requires_grad=True)
    rhs = X.new_tensor([0.4, 0.7], requires_grad=True)
    index = torch.tensor([0, 0, 2] if duplicate else [0, 2], device=device)
    other = torch.tensor([1, 2], device=device)

    def evaluate(x, first, second, right):
        return project_to_equality_constraints(
            x, [(index, first, right[0]), (other, second, right[1])]
        )

    with no_host():
        assert torch.autograd.gradcheck(
            evaluate, (X, coefficients, second_coefficients, rhs), fast_mode=True
        )
        assert torch.autograd.gradgradcheck(
            evaluate, (X, coefficients, second_coefficients, rhs), fast_mode=True
        )
    if duplicate:
        # Both occurrences contribute to the same summed matrix column.
        value = evaluate(X, coefficients, second_coefficients, rhs)
        gradient = torch.autograd.grad(value.square().sum(), coefficients)[0]
        assert gradient[0] == gradient[1] and gradient[0] != 0


@pytest.mark.parametrize("device", DEVICES)
def test_device_projection_identity_empty_batch_validation_and_singular_system(device):
    X = torch.tensor([[0.2, 0.3]], dtype=torch.double, device=device)
    assert project_to_equality_constraints(X, []) is X
    assert (
        project_to_equality_constraints(
            X, [(torch.tensor([[0, 0]], device=device), X.new_tensor([1.0]), 0.3)]
        )
        is X
    )
    empty = X.new_empty(0, 2, 2)
    with no_host():
        assert project_to_equality_constraints(empty, [([0], [1.0], 0.3)]).shape == empty.shape
    for index in ([-1], [2], [0.2]):
        with pytest.raises(ValueError):
            project_to_equality_constraints(X, [(index, [1.0], 0.3)])
    with pytest.raises(RuntimeError, match="singular"):
        project_to_equality_constraints(X, [([0], [1.0], 0.3), ([0], [1.0], 0.4)])


@pytest.mark.parametrize("device", DEVICES)
def test_feasibility_equality_tolerance_matches_source_strict_boundary(device):
    from botorch.optim.parameter_constraints import evaluate_feasibility as reference

    X = torch.tensor([[0.5]], dtype=torch.double, device=device)
    constraints = [(torch.tensor([0], device=device), X.new_tensor([1.0]), 0.0)]
    with no_host():
        actual = evaluate_feasibility(X, equality_constraints=constraints, tolerance=0.5)
    torch.testing.assert_close(
        actual, reference(X, equality_constraints=constraints, tolerance=0.5)
    )
    assert not actual

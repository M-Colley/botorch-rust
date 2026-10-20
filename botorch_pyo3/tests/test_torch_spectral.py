"""Independent finite differences at repeated mixed PSD clipping eigenspaces."""

import pytest
import torch

from botorch_rs.models.empirical_gps._torch_em import psd_projection
from botorch_rs.models.empirical_gps._torch_spectral import smooth_psd_floor

DEVICES = ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("rotated", [False, True])
def test_repeated_mixed_psd_floor_gradcheck_and_gradgradcheck(device, rotated):
    diagonal = torch.tensor([-1.0, -1.0, 2.0, 2.0], dtype=torch.double, device=device)
    value = torch.diag(diagonal)
    if rotated:
        generator = torch.Generator(device=device).manual_seed(41)
        basis, _ = torch.linalg.qr(
            torch.randn(4, 4, dtype=torch.double, device=device, generator=generator)
        )
        value = basis @ value @ basis.mT
    else:
        basis = torch.eye(4, dtype=torch.double, device=device)
    value.requires_grad_()
    expected = basis @ torch.diag(diagonal.clamp_min(0.1)) @ basis.mT
    torch.testing.assert_close(psd_projection(value, 0.1), expected)

    def function(matrix):
        return psd_projection(matrix, 0.1)

    assert torch.autograd.gradcheck(function, (value,), atol=1e-5, rtol=2e-4)
    assert torch.autograd.gradgradcheck(function, (value,), atol=2e-5, rtol=3e-4)


@pytest.mark.parametrize("device", DEVICES)
def test_float32_repeated_mixed_psd_floor_and_third_derivatives(device):
    value = torch.diag(torch.tensor([-1.0, -1.0, 2.0, 2.0], device=device)).requires_grad_()
    weights = torch.arange(1, 17, device=device, dtype=value.dtype).reshape(4, 4)
    result = psd_projection(value, 0.1)
    assert result.dtype == value.dtype and result.device == value.device
    first = torch.autograd.grad((result * weights).sum(), value, create_graph=True)[0]
    second = torch.autograd.grad(first.square().sum(), value, create_graph=True)[0]
    third = torch.autograd.grad(second.square().sum(), value)[0]
    assert all(torch.isfinite(tensor).all() for tensor in (first, second, third))


def _stable_loewner_vjp(value, gradient, floor):
    """Independent divided differences; no absolute or Sylvester algebra."""
    symmetric = (value.double() + value.double().mT) * 0.5
    eigenvalues, vectors = torch.linalg.eigh(symmetric)
    positive = eigenvalues > floor
    left, right = positive.unsqueeze(-1), positive.unsqueeze(-2)
    gap = (eigenvalues.unsqueeze(-1) - eigenvalues.unsqueeze(-2)).abs()
    numerator = torch.maximum(eigenvalues.unsqueeze(-1), eigenvalues.unsqueeze(-2)) - floor
    mixed = left ^ right
    divided_difference = torch.where(
        left & right,
        1.0,
        torch.where(mixed, numerator / torch.where(mixed, gap, 1.0), 0.0),
    )
    symmetric_gradient = (gradient.double() + gradient.double().mT) * 0.5
    return vectors @ (divided_difference * (vectors.mT @ symmetric_gradient @ vectors)) @ vectors.mT


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.double])
@pytest.mark.parametrize("rotation", ["none", "hadamard", "random"])
@pytest.mark.parametrize("small", [0.0, -1e-8])
def test_tiny_floor_mixed_zero_positive_vjp_and_hvp_match_stable_oracle(
    device, dtype, rotation, small
):
    from test_torch_acq_discrete import no_host

    if rotation == "hadamard":
        basis = (
            torch.tensor(
                [[1, 1, 1, 1], [1, -1, 1, -1], [1, 1, -1, -1], [1, -1, -1, 1]],
                dtype=torch.double,
                device=device,
            )
            * 0.5
        )
    elif rotation == "random":
        generator = torch.Generator(device=device).manual_seed(41)
        basis, _ = torch.linalg.qr(
            torch.randn(4, 4, dtype=torch.double, device=device, generator=generator)
        )
    else:
        basis = torch.eye(4, dtype=torch.double, device=device)
    diagonal = basis.new_tensor([small, small, 1.0, 2.0])
    value = (basis @ torch.diag(diagonal) @ basis.mT).to(dtype).requires_grad_()
    gradient = value.new_tensor(
        [[0.7, 0.3, -0.5, 0.2], [0.3, -0.8, 0.4, 0.6], [-0.5, 0.4, 1.1, 0.9], [0.2, 0.6, 0.9, -0.2]]
    )
    cross_block = basis.new_tensor(
        [[0.0, 0.0, 0.3, -0.4], [0.0, 0.0, 0.2, 0.5], [0.3, 0.2, 0.0, 0.0], [-0.4, 0.5, 0.0, 0.0]]
    )
    direction = (basis @ cross_block @ basis.mT).to(dtype)
    floor, step = 1e-10, 1e-5
    plus = value.detach().double() + step * direction.double()
    minus = value.detach().double() - step * direction.double()
    # A generic 1e-5 perturbation would cross a 1e-10 floor and would not
    # approximate the local Hessian. Cross-block directions shift zero
    # eigenvalues negative only at O(step^2); explicitly check that neither
    # finite-difference endpoint changes clipping membership.
    positive = torch.linalg.eigvalsh(value.detach().double()) > floor
    assert torch.equal(torch.linalg.eigvalsh(plus) > floor, positive)
    assert torch.equal(torch.linalg.eigvalsh(minus) > floor, positive)
    expected = _stable_loewner_vjp(value.detach(), gradient, floor)
    numerical_hvp = (
        _stable_loewner_vjp(plus, gradient, floor) - _stable_loewner_vjp(minus, gradient, floor)
    ) / (2 * step)
    with no_host():
        output = smooth_psd_floor(value, floor)
        actual = torch.autograd.grad((output * gradient).sum(), value, create_graph=True)[0]
        hvp = (
            torch.autograd.grad((actual * direction).sum(), value)[0]
            if actual.requires_grad
            else torch.zeros_like(value)
        )
    tolerance = 1e-7 if dtype == torch.float32 else 1e-9
    torch.testing.assert_close(actual.double(), expected, atol=tolerance, rtol=tolerance)
    torch.testing.assert_close(hvp.double(), numerical_hvp, atol=tolerance, rtol=tolerance)
    assert output.device == value.device and output.dtype == value.dtype


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.double])
def test_gap_projector_mixed_batch_and_single_side_graphs(device, dtype):
    diagonal = torch.tensor(
        [[-1.0, -1.0, -2.0, -2.0], [0.2, 0.2, 1.0, 2.0], [0.0, 0.0, 1.0, 2.0]],
        dtype=dtype,
        device=device,
    )
    matrices = torch.diag_embed(diagonal).requires_grad_()
    gradient = torch.arange(1, 17, dtype=dtype, device=device).reshape(4, 4)
    output = smooth_psd_floor(matrices, 1e-10)
    torch.testing.assert_close(output, torch.diag_embed(diagonal.clamp_min(1e-10)))
    actual = torch.autograd.grad((output * gradient).sum(), matrices)[0]
    expected = _stable_loewner_vjp(matrices.detach(), gradient, 1e-10)
    torch.testing.assert_close(
        actual.double(),
        expected,
        atol=1e-6 if dtype == torch.float32 else 1e-12,
        rtol=1e-6 if dtype == torch.float32 else 1e-12,
    )
    for tensor in (matrices[:1].detach().requires_grad_(), matrices[1:2].detach().requires_grad_()):
        result = smooth_psd_floor(tensor, 1e-10)
        assert result.requires_grad
        derivative = torch.autograd.grad(result.square().sum(), tensor)[0]
        expected_derivative = (
            torch.zeros_like(tensor)
            if bool((tensor.detach().diagonal(dim1=-2, dim2=-1) < 1e-10).all())
            else 2 * tensor
        )
        torch.testing.assert_close(derivative, expected_derivative)


@pytest.mark.parametrize("device", DEVICES)
def test_gap_projector_retains_explicit_nondifferentiable_threshold_error(device):
    value = torch.diag(
        torch.tensor([0.1, -1.0, 2.0], dtype=torch.double, device=device)
    ).requires_grad_()
    with pytest.raises(ValueError, match="clipping threshold"):
        smooth_psd_floor(value, 0.1)

"""Independent finite differences at repeated mixed PSD clipping eigenspaces."""

import pytest
import torch

from botorch_rs.models.empirical_gps._torch_em import psd_projection

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

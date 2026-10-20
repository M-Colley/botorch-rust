"""Detached MLL diagnostics must not alter the live model's parameters or graph."""

import pytest
import torch

from botorch_rs.mlls import ExactMarginalLogLikelihood
from botorch_rs.models import SingleTaskGP

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("outputs", [1, 2])
@pytest.mark.parametrize("data_grad", [False, True])
def test_mll_diagnostic_values_gradients_and_raw_vector_leave_live_state_intact(
    device, outputs, data_grad
):
    X = torch.tensor(
        [[0.04], [0.27], [0.56], [0.81]], dtype=torch.double, device=device, requires_grad=data_grad
    )
    Y = torch.cat(((X.detach() * 5.3).sin(), (X.detach() * 3.1).cos()), dim=-1)[
        ..., :outputs
    ].requires_grad_(data_grad)
    model = SingleTaskGP(X, Y, backend="torch")
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    parameters = tuple(model.parameters())
    originals = [parameter.detach().clone() for parameter in parameters]
    value = mll()
    expected = torch.autograd.grad(value.sum(), parameters, retain_graph=True)
    expected_flat = torch.cat([gradient.reshape(-1) for gradient in expected])
    # The extension evaluates derivatives even when public leaves are frozen.
    model.requires_grad_(False)
    actual_value, gradient = mll.value_and_grad()
    torch.testing.assert_close(value.detach(), X.new_tensor(actual_value))
    torch.testing.assert_close(expected_flat, X.new_tensor(gradient))
    raw = torch.cat([parameter.detach().reshape(-1) for parameter in parameters])
    changed = raw.clone()
    changed[0] += 0.02
    changed_value, _ = mll.value_and_grad(changed)
    assert torch.isfinite(X.new_tensor(changed_value)).all()
    epsilon = 1e-6
    direction = torch.zeros_like(raw)
    direction[0] = epsilon
    plus = X.new_tensor(mll.value_and_grad(raw + direction)[0]).sum()
    minus = X.new_tensor(mll.value_and_grad(raw - direction)[0]).sum()
    torch.testing.assert_close(
        (plus - minus) / (2 * epsilon), expected_flat[0], atol=2e-6, rtol=2e-6
    )
    for parameter, original in zip(parameters, originals, strict=True):
        torch.testing.assert_close(parameter, original)
        assert parameter.grad is None
        assert not parameter.requires_grad
    model.requires_grad_(True)
    value.sum().backward()
    assert all(parameter.grad is not None for parameter in parameters)

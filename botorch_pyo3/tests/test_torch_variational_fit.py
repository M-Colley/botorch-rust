"""Public Tensor ELBO types and actual full/minibatch optimizer integration."""

# ruff: noqa: E402
from __future__ import annotations

from functools import partial

import pytest

torch = pytest.importorskip("torch")
gpytorch = pytest.importorskip("gpytorch")

from test_torch_variational_backend import DEVICES, _models

from botorch_rs.exceptions.errors import UnsupportedError
from botorch_rs.fit import fit_gpytorch_mll
from botorch_rs.mlls import VariationalELBO
from botorch_rs.optim.closures import get_loss_closure, get_loss_closure_with_grads
from botorch_rs.optim.fit import fit_gpytorch_mll_scipy, fit_gpytorch_mll_torch
from botorch_rs.optim.utils.model_utils import get_parameters


def _loader(inputs, targets, batch_size):
    return torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(inputs.detach(), targets.detach()),
        batch_size=batch_size,
        shuffle=False,
    )


@pytest.mark.parametrize("device", DEVICES)
def test_torch_elbo_public_type_parameter_protocol_and_diagnostic_gradient(device):
    model, _, _, _ = _models(device, 1)
    mll = VariationalELBO(model.likelihood, model.model, num_data=8)
    assert isinstance(mll, VariationalELBO)
    assert isinstance(mll, torch.nn.Module)
    names = dict(mll.named_parameters())
    assert names.keys() == get_parameters(mll).keys()
    assert not list(mll.named_parameters(recurse=False))
    assert list(dict(mll.named_parameters(prefix="loss"))) == ["loss." + n for n in names]
    assert len(list(mll.named_parameters(remove_duplicate=False))) >= len(names)
    frozen = mll.model.mean_module.raw_constant
    frozen.requires_grad_(False)
    trainable = [p for p in names.values() if p.requires_grad]
    value = mll().sum()
    expected = torch.autograd.grad(value, trainable)
    gradients_by_identity = dict(zip(map(id, trainable), expected, strict=True))
    flat = torch.cat(
        [
            gradients_by_identity[id(parameter)].detach().reshape(-1)
            if parameter.requires_grad
            else torch.zeros_like(parameter).reshape(-1)
            for parameter in names.values()
        ]
    )
    actual, gradient = mll.value_and_grad()
    assert actual == pytest.approx(value.item(), abs=1e-12)
    torch.testing.assert_close(flat.cpu(), torch.tensor(gradient, dtype=flat.dtype))
    with pytest.raises(UnsupportedError, match="current Parameters"):
        mll.value_and_grad([0])
    mll.requires_grad_(True)
    (-mll()).backward()
    assert all(p.grad is not None for p in mll.parameters())
    mll.zero_grad(set_to_none=True)
    assert all(p.grad is None for p in mll.parameters())


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("outputs", [1, 2])
@pytest.mark.parametrize("batch_size", [3, 8])
def test_external_elbo_closure_cycles_actual_batches_and_matches_reference_gradients(
    device, outputs, batch_size
):
    model, reference, inputs, targets = _models(device, outputs)
    mll = VariationalELBO(model.likelihood, model.model, num_data=8)
    ref_mll = gpytorch.mlls.VariationalELBO(reference.likelihood, reference.model, num_data=8)
    loader = _loader(inputs[0], targets[0].squeeze(-1) if outputs == 1 else targets[0], batch_size)
    closure = get_loss_closure_with_grads(mll, dict(mll.named_parameters()), loader)
    batches = list(loader)
    ref_parameters = dict(ref_mll.named_parameters())
    assert ref_parameters.keys() == dict(mll.named_parameters()).keys()
    for index in range(len(batches) * 2 + 1):
        x, y = batches[index % len(batches)]
        expected = -ref_mll(reference.model(x), y).sum()
        ref_gradients = dict(
            zip(
                ref_parameters,
                torch.autograd.grad(expected, tuple(ref_parameters.values())),
                strict=True,
            )
        )
        actual, gradients = closure()
        torch.testing.assert_close(actual, expected, atol=1e-10, rtol=1e-9)
        for name, actual_gradient in zip(dict(mll.named_parameters()), gradients, strict=True):
            torch.testing.assert_close(actual_gradient, ref_gradients[name], atol=2e-9, rtol=2e-8)
        for parameter, gradient in zip(mll.parameters(), gradients, strict=True):
            assert parameter.grad is gradient


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("outputs", [1, 2])
@pytest.mark.parametrize("batch_size", [3, 8])
@pytest.mark.parametrize("high_level", [False, True])
def test_external_elbo_fit_really_steps_adam_like_reference(
    device, outputs, batch_size, high_level
):
    model, reference, inputs, targets = _models(device, outputs)
    mll = VariationalELBO(model.likelihood, model.model, num_data=8)
    ref_mll = gpytorch.mlls.VariationalELBO(reference.likelihood, reference.model, num_data=8)
    loader = _loader(inputs[0], targets[0].squeeze(-1) if outputs == 1 else targets[0], batch_size)
    parameters = dict(mll.named_parameters())
    initial = {n: p.detach().clone() for n, p in parameters.items()}
    closure = get_loss_closure_with_grads(mll, parameters, loader)
    options = dict(
        step_limit=4,
        stopping_criterion=None,
        optimizer=partial(torch.optim.Adam, lr=0.005),
    )
    if high_level:
        assert (
            fit_gpytorch_mll(
                mll, closure=closure, optimizer=fit_gpytorch_mll_torch, optimizer_kwargs=options
            )
            is mll
        )
        assert not mll.training
        assert not model.model.training
        step, actual_loss = mll._fit_result["nit"], mll._fit_result["fval"]
    else:
        result = fit_gpytorch_mll_torch(mll, closure=closure, **options)
        step, actual_loss = result.step, result.fval
    assert step == 4
    optimizer = torch.optim.Adam(ref_mll.parameters(), lr=0.005)
    batches = list(loader)
    for index in range(4):
        x, y = batches[index % len(batches)]
        optimizer.zero_grad()
        loss = -ref_mll(reference.model(x), y).sum()
        loss.backward()
        optimizer.step()
    x, y = batches[4 % len(batches)]
    expected_loss = -ref_mll(reference.model(x), y).sum()
    assert actual_loss == pytest.approx(expected_loss.item(), abs=2e-9)
    assert any(not torch.equal(initial[name], value) for name, value in parameters.items())
    expected = dict(ref_mll.named_parameters())
    for name, value in parameters.items():
        torch.testing.assert_close(value, expected[name], atol=2e-9, rtol=2e-8)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("high_level", [False, True])
def test_tensor_elbo_default_scipy_fit_and_frozen_parameter(device, high_level):
    model, _, _, _ = _models(device, 1)
    mll = VariationalELBO(model.likelihood, model.model, num_data=8)
    fixed = mll.model.mean_module.raw_constant
    fixed.requires_grad_(False)
    original = fixed.detach().clone()
    initial = mll().detach()
    if high_level:
        assert fit_gpytorch_mll(mll, optimizer_kwargs={"options": {"maxiter": 3}}) is mll
        assert not mll.training
    else:
        result = fit_gpytorch_mll_scipy(mll, options={"maxiter": 3})
        assert result.step > 0
    assert mll().detach() > initial
    torch.testing.assert_close(fixed, original, atol=0, rtol=0)


def test_tensor_external_elbo_loader_validation_and_repeated_empty_epochs():
    model, _, inputs, targets = _models("cpu", 1)
    mll = VariationalELBO(model.likelihood, model.model, num_data=8)
    empty = _loader(inputs[0][:0], targets[0][:0], 3)
    closure = get_loss_closure(mll, empty)
    for _ in range(2):
        with pytest.raises(ValueError, match="at least one minibatch"):
            closure()
    with pytest.raises(TypeError, match="batch of tensors"):
        get_loss_closure(mll, [inputs[0]])()
    with pytest.raises(ValueError, match="inputs and targets"):
        get_loss_closure(mll, [(inputs[0],)])()
    with pytest.raises(UnsupportedError, match="closure keyword"):
        get_loss_closure(mll, [(inputs[0], targets[0])])(unknown=True)
    terms = VariationalELBO(model.likelihood, model.model, num_data=8, combine_terms=False)
    assert isinstance(terms, VariationalELBO)
    with pytest.raises(UnsupportedError, match="combine_terms"):
        get_loss_closure(terms, [(inputs[0], targets[0])])

# ruff: noqa: E402
"""External variational minibatches use native ELBO values and gradients."""

from __future__ import annotations

import pytest

pytest.importorskip("torch")
pytest.importorskip("botorch")


import torch
from approximate_helpers import Case, build_pair, elbo_pair, set_random_parameters
from botorch.optim.closures.model_closures import get_loss_closure as reference_loss
from botorch.optim.closures.model_closures import (
    get_loss_closure_with_grads as reference_loss_and_grads,
)
from torch.utils.data import DataLoader, TensorDataset

from botorch_rs.exceptions.errors import UnsupportedError
from botorch_rs.mlls import ExactMarginalLogLikelihood
from botorch_rs.models import SingleTaskGP
from botorch_rs.optim.closures import (
    NdarrayOptimizationClosure,
    get_loss_closure,
    get_loss_closure_with_grads,
)


def twins(outputs=1, transforms=False):
    case = Case(
        "minibatch",
        n=9,
        m=outputs,
        d=2,
        inducing=3,
        input_tf=transforms,
        outcome_tf=transforms,
        seed=73,
    )
    reference, native, _, _ = build_pair(case)
    set_random_parameters(reference, native, seed=11)
    a, b = elbo_pair(reference, native, num_data=37, beta=0.6)
    reference.train()
    native.train()
    # DataLoader follows GPyTorch's inner variational model protocol: its
    # inputs/targets are already transformed by the outer BoTorch wrapper.
    dataset = TensorDataset(*a.model.train_inputs, a.model.train_targets)
    return a, b, DataLoader(dataset, batch_size=4, shuffle=False)


@pytest.mark.parametrize("outputs", [1, 2])
@pytest.mark.parametrize("transforms", [False, True])
def test_values_follow_actual_batches_and_restart_each_epoch(outputs, transforms):
    a, b, loader = twins(outputs, transforms)
    expected, actual = reference_loss(a, loader), get_loss_closure(b, loader)
    train_x = b.model.train_inputs[0].clone()
    train_y = b.model.train_targets.clone()
    values = []
    for _ in range(7):
        va, vb = expected(), actual()
        torch.testing.assert_close(vb, va.detach(), atol=1e-9, rtol=1e-9)
        values.append(float(vb))
    assert abs(values[0] - values[1]) > 1e-3
    assert values[0] == pytest.approx(values[3], abs=1e-12)
    assert values[0] == pytest.approx(values[6], abs=1e-12)
    torch.testing.assert_close(b.model.train_inputs[0], train_x)
    torch.testing.assert_close(b.model.train_targets, train_y)


@pytest.mark.parametrize("outputs", [1, 2])
@pytest.mark.parametrize("transforms", [False, True])
def test_selected_native_parameter_gradients_match_official_minibatches(outputs, transforms):
    a, b, loader = twins(outputs, transforms)
    params_a, params_b = dict(a.named_parameters()), dict(b.named_parameters())
    assert params_a.keys() == params_b.keys()
    # Include likelihood, inducing locations, kernel and variational state.
    selected = list(params_b)[::2]
    ca = reference_loss_and_grads(a, {name: params_a[name] for name in selected}, loader)
    cb = get_loss_closure_with_grads(b, {name: params_b[name] for name in selected}, loader)
    losses, gradients = [], []
    for _ in range(4):
        value_a, grad_a = ca()
        value_b, grad_b = cb()
        assert value_b == pytest.approx(float(value_a), abs=1e-9, rel=1e-9)
        for actual, expected in zip(grad_b, grad_a, strict=True):
            expected = torch.zeros_like(actual) if expected is None else expected
            torch.testing.assert_close(actual, expected, atol=1e-8, rtol=1e-8)
        losses.append(value_b)
        gradients.append(torch.cat([value.reshape(-1) for value in grad_b]))
    assert abs(losses[0] - losses[1]) > 1e-3
    assert not torch.allclose(gradients[0], gradients[1])
    torch.testing.assert_close(gradients[0], gradients[3], atol=1e-12, rtol=1e-12)


def test_ndarray_closure_updates_real_native_state_then_evaluates_external_batch():
    _, mll, loader = twins()
    parameters = dict(mll.named_parameters())
    closure = get_loss_closure_with_grads(mll, parameters, loader)
    array_closure = NdarrayOptimizationClosure(closure, parameters)
    original = array_closure.state.copy()
    value, gradient = array_closure()
    norm = float(gradient @ gradient) ** 0.5
    direction = gradient / max(norm, 1.0)
    step = 1e-6
    changed = original + step * direction
    array_closure.state = changed
    assert not torch.equal(torch.tensor(original), torch.tensor(array_closure.state))
    # A fresh closure starts at the same first minibatch, enabling a directional
    # finite difference independent of the native parameter-gradient code.
    batch_loss = get_loss_closure(mll, loader)
    actual = float(batch_loss())
    directional = (actual - float(value)) / step
    assert directional == pytest.approx(float(gradient @ direction), rel=1e-3)


@pytest.mark.parametrize(
    "loader,message",
    [
        ([], "at least one"),
        ([torch.ones(2)], "batch of tensors"),
        ([(torch.ones(2, 2),)], "inputs and targets"),
    ],
)
def test_invalid_batches_reject_explicitly(loader, message):
    _, mll, _ = twins()
    with pytest.raises((ValueError, TypeError), match=message):
        get_loss_closure(mll, loader)()


def test_empty_loader_rejects_before_upstream_busy_loop():
    class EmptyLoader:
        def __init__(self):
            self.iterations = 0

        def __iter__(self):
            self.iterations += 1
            if self.iterations > 20:
                raise RuntimeError("busy-loop guard")
            return iter(())

    a, b, _ = twins()
    native_loader, reference_loader = EmptyLoader(), EmptyLoader()
    with pytest.raises(ValueError, match="at least one"):
        get_loss_closure(b, native_loader)()
    assert native_loader.iterations == 2
    # Safely detect the upstream infinite empty-epoch loop without a hanging
    # subprocess: this guard interrupts it after 20 iterations.
    with pytest.raises(RuntimeError, match="busy-loop guard"):
        reference_loss(a, reference_loader)()
    assert reference_loader.iterations == 21


def test_external_uncombined_terms_reject_explicitly():
    _, mll, loader = twins()
    mll.combine_terms = False
    with pytest.raises(UnsupportedError, match="combine_terms=True"):
        get_loss_closure(mll, loader)


def test_native_gradient_closure_rejects_ignored_kwargs():
    _, mll, loader = twins()
    closure = get_loss_closure_with_grads(mll, dict(mll.named_parameters()), loader)
    with pytest.raises(UnsupportedError, match="unexpected"):
        closure(unexpected=True)


def test_internal_variational_gradient_closure_uses_elbo_instead_of_exact_mll():
    a, b, _ = twins()
    ca = reference_loss_and_grads(a, dict(a.named_parameters()))
    cb = get_loss_closure_with_grads(b, dict(b.named_parameters()))
    va, ga = ca()
    vb, gb = cb()
    assert vb == pytest.approx(float(va), abs=1e-9, rel=1e-9)
    for actual, expected in zip(gb, ga, strict=True):
        expected = torch.zeros_like(actual) if expected is None else expected
        torch.testing.assert_close(actual, expected, atol=1e-8, rtol=1e-8)


def test_exact_gp_external_data_remains_explicitly_unsupported():
    model = SingleTaskGP(
        torch.tensor([[0.0], [1.0]], dtype=torch.double),
        torch.tensor([[0.0], [1.0]], dtype=torch.double),
    )
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    with pytest.raises(UnsupportedError, match="backend='torch'"):
        get_loss_closure(mll, [(torch.ones(2, 1), torch.ones(2))])

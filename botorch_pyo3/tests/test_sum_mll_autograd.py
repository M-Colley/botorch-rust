"""Broadcast weighted native child MLL adjoints through summed likelihoods."""

from __future__ import annotations

import pytest

from helpers import as_kind, assert_close, check_kind, importorskip_botorch

importorskip_botorch()
torch = pytest.importorskip("torch")

from botorch.models import ModelListGP as BtModelListGP  # noqa: E402
from gpytorch.mlls import SumMarginalLogLikelihood as BtSumMLL  # noqa: E402
from test_posterior_parameter_gradients import model_pair  # noqa: E402

from botorch_rs.mlls import (  # noqa: E402
    ExactMarginalLogLikelihood,
    SumMarginalLogLikelihood,
)
from botorch_rs.models import ModelListGP, SingleTaskGP  # noqa: E402


def unique_parameters(model):
    return tuple({id(parameter): parameter for parameter in model.parameters()}.values())


def paired_mlls(specs):
    pairs = [model_pair(*spec) for spec in specs]
    actual = ModelListGP(*(pair[0] for pair in pairs))
    reference = BtModelListGP(*(pair[1] for pair in pairs))
    return (
        SumMarginalLogLikelihood(actual.likelihood, actual),
        BtSumMLL(reference.likelihood, reference),
    )


def reference_value(mll):
    mll.train()
    return mll(mll.model(*mll.model.train_inputs), mll.model.train_targets)


@pytest.mark.parametrize(
    "first,second",
    [
        ((1, (), False, False, False), (1, (), True, False, True)),
        ((2, (), False, False, False), (1, (), True, False, True)),
        ((2, (2,), True, False, True), (1, (2,), False, False, False)),
        ((3, (2, 1), False, False, False), (1, (2, 1), True, False, True)),
        ((1, (), True, True, False), (1, (), False, False, True)),
        ((1, (2,), False, True, False), (2, (2,), True, False, True)),
    ],
)
def test_weighted_sum_mll_matches_official_child_broadcast_gradients(first, second):
    actual_mll, expected_mll = paired_mlls([first, second])
    actual, expected = actual_mll(), reference_value(expected_mll)
    torch.testing.assert_close(actual, expected, rtol=2e-7, atol=2e-8)
    weights = torch.linspace(-0.7, 1.1, actual.numel(), dtype=actual.dtype).reshape(actual.shape)
    actual_grads = torch.autograd.grad(
        (actual * weights).sum(), unique_parameters(actual_mll.model)
    )
    expected_grads = torch.autograd.grad(
        (expected * weights).sum(), unique_parameters(expected_mll.model)
    )
    for result, want in zip(actual_grads, expected_grads, strict=True):
        torch.testing.assert_close(result, want, rtol=6e-7, atol=6e-8)


@pytest.mark.parametrize("alias", ["model", "mean_parameter"])
def test_sum_mll_accumulates_repeated_models_and_shared_parameter_leaves(alias):
    actual_mll, expected_mll = paired_mlls(
        [(1, (2,), False, False, False), (1, (2,), True, False, False)]
    )
    actual_models, expected_models = actual_mll.model.models, expected_mll.model.models
    if alias == "model":
        actual = ModelListGP(actual_models[0], actual_models[1], actual_models[0])
        reference = BtModelListGP(expected_models[0], expected_models[1], expected_models[0])
        actual_mll = SumMarginalLogLikelihood(actual.likelihood, actual)
        expected_mll = BtSumMLL(reference.likelihood, reference)
    else:
        # The persistent-leaf registry also permits an identical leaf to be
        # reached through two distinct native snapshots. Both adjoints must
        # accumulate, as they do for an explicitly shared Torch parameter.
        shared = actual_models[0].mean_module.raw_constant
        state = actual_models[1]._sync_torch_parameters()
        state.entries = [
            (name, shared if name == "mean_module.raw_constant" else parameter)
            for name, parameter in state.entries
        ]
        expected_models[1].mean_module.raw_constant = expected_models[0].mean_module.raw_constant
    actual, expected = actual_mll(), reference_value(expected_mll)
    weights = torch.tensor([0.3, -0.8], dtype=torch.double)
    torch.testing.assert_close(actual, expected)
    result = torch.autograd.grad((actual * weights).sum(), unique_parameters(actual_mll.model))
    want = torch.autograd.grad((expected * weights).sum(), unique_parameters(expected_mll.model))
    for gradient, expected_gradient in zip(result, want, strict=True):
        torch.testing.assert_close(gradient, expected_gradient, rtol=6e-7, atol=6e-8)


def test_sum_mll_backward_uses_child_snapshots_after_parameter_updates():
    mll, _ = paired_mlls([(2, (2,), False, False, False), (1, (2,), True, False, True)])
    value = mll.forward()
    weights = torch.tensor([[0.3, -0.7], [1.2, 0.4]], dtype=torch.double)
    leaves = unique_parameters(mll.model)
    expected = torch.autograd.grad((value * weights).sum(), leaves, retain_graph=True)
    for child in mll.model.models:
        child.mean_module.constant = 0.4
        tuple(child.parameters())
        with torch.no_grad():
            next(
                parameter
                for name, parameter in child.named_parameters()
                if name.startswith("covar_module.")
            ).add_(0.02)
    assert not torch.equal(mll(), value)
    actual = torch.autograd.grad((value * weights).sum(), leaves)
    for result, want in zip(actual, expected, strict=True):
        torch.testing.assert_close(result, want)


def test_sum_mll_freezing_no_grad_and_external_adam():
    mll, _ = paired_mlls([(1, (), False, False, False), (1, (), True, False, True)])
    leaves = unique_parameters(mll.model)
    leaves[0].requires_grad_(False)
    optimizer = torch.optim.Adam(leaves[1:], lr=0.001)
    losses = []
    for _ in range(4):
        optimizer.zero_grad()
        loss = -mll()
        losses.append(loss.item())
        loss.backward()
        optimizer.step()
    assert losses[-1] < losses[0]
    assert leaves[0].grad is None
    with torch.no_grad():
        assert not mll().requires_grad
    for parameter in leaves:
        parameter.requires_grad_(False)
    assert not mll().requires_grad


@pytest.mark.parametrize("kind", ["list", "numpy"])
def test_sum_mll_preserves_non_torch_types_and_values(kind):
    x = [[0.1], [0.4], [0.7], [0.9]]
    y = [[0.2, 0.4], [0.5, 0.3], [0.4, 0.6], [0.8, 0.7]]
    children = [
        SingleTaskGP(as_kind(x, kind), as_kind(target, kind), outcome_transform=None)
        for target in (y, [[row[0]] for row in y])
    ]
    model = ModelListGP(*children)
    result = SumMarginalLogLikelihood(model.likelihood, model)()
    check_kind(result, kind)
    first, second = [ExactMarginalLogLikelihood(m.likelihood, m)() for m in children]
    assert_close(result, [(float(value) + float(second)) / 2 for value in first])


def test_sum_mll_keeps_incompatible_shape_error():
    mll, _ = paired_mlls([(2, (), False, False, False), (3, (), True, False, True)])
    with pytest.raises(RuntimeError, match="shapes that cannot be broadcast"):
        mll()

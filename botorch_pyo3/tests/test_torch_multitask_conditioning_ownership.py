"""Multitask fantasy parameters have independent source-compatible ownership."""

# ruff: noqa: E402
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from test_torch_backend_multitask import DEVICES, data, paired_models
from test_torch_input_model_conditioning import MeanSampler
from test_torch_multitask_conditioning import observations


def _gradient(value, leaves):
    result = torch.autograd.grad(value, leaves, retain_graph=True, allow_unused=True)
    return tuple(
        torch.zeros_like(leaf) if part is None else part
        for leaf, part in zip(leaves, result, strict=True)
    )


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("fixed", [False, True])
def test_conditioned_multitask_hyperparameters_are_independent_like_source(device, fixed):
    X, Y, Q = data(device)
    noise = torch.full_like(Y, 0.025) if fixed else None
    actual, source = paired_models(X, Y, -1, noise, outcome_transform=None)
    extra = observations(X, -1)
    values = X.new_tensor([[0.3], [0.7]])
    additional = torch.full_like(values, 0.035) if fixed else None
    original = actual.posterior(Q[0]).mean.detach().clone()
    source_original = source.posterior(Q[0]).mean.detach().clone()
    result = actual.condition_on_observations(extra, values, noise=additional)
    reference = source.condition_on_observations(extra, values, noise=additional)
    before, after = dict(actual.named_parameters()), dict(result.named_parameters())
    assert before.keys() == after.keys()
    for name, parameter in before.items():
        assert after[name] is not parameter and after[name].data_ptr() != parameter.data_ptr()
        torch.testing.assert_close(after[name], parameter)
    source_before, source_after = (
        dict(source.named_parameters()),
        dict(reference.named_parameters()),
    )
    assert source_before.keys() == source_after.keys()
    for name in source_before:
        assert source_before[name] is not source_after[name]
    with torch.no_grad():
        for parameter in result.parameters():
            parameter.add_(0.05)
        for parameter in reference.parameters():
            parameter.add_(0.05)
    torch.testing.assert_close(actual.posterior(Q[0]).mean, original)
    torch.testing.assert_close(source.posterior(Q[0]).mean, source_original)
    result.to(dtype=torch.float32)
    assert all(parameter.dtype == torch.double for parameter in actual.parameters())


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("propagate", [False, True])
def test_multitask_fantasy_original_and_fantasy_gradient_leaves_match_source(device, propagate):
    settings = pytest.importorskip("botorch.settings")
    X, Y, _ = data(device)
    X, Y = X.requires_grad_(), Y.requires_grad_()
    actual, source = paired_models(X, Y, outcome_transform=None)
    extra = observations(X, -1).requires_grad_()
    query = X.new_tensor([[0.24, 0.0], [0.57, 1.0]], requires_grad=True)
    left, right = MeanSampler(), MeanSampler()
    result = actual.fantasize(extra, left, propagate_grads=propagate)
    reference = source.fantasize(extra, right, propagate_grads=propagate)
    assert left.shape == right.shape
    reference.prediction_strategy = None
    with settings.propagate_grads(True):
        a, e = result.posterior(query), reference.posterior(query)
    torch.testing.assert_close(a.mean, e.mean, atol=3e-8, rtol=3e-8)
    torch.testing.assert_close(
        a.distribution.covariance_matrix, e.distribution.covariance_matrix, atol=3e-8, rtol=3e-8
    )
    original, copied = dict(actual.named_parameters()), dict(result.named_parameters())
    source_original, source_copied = (
        dict(source.named_parameters()),
        dict(reference.named_parameters()),
    )
    assert original.keys() == copied.keys() == source_original.keys() == source_copied.keys()
    left_leaves = (X, Y, extra, query, *original.values(), *copied.values())
    right_leaves = (
        X,
        Y,
        extra,
        query,
        *(source_original[name] for name in original),
        *(source_copied[name] for name in copied),
    )
    actual_gradient = _gradient(a.mean.sin().sum() + 0.1 * a.variance.square().sum(), left_leaves)
    source_gradient = _gradient(e.mean.sin().sum() + 0.1 * e.variance.square().sum(), right_leaves)
    for value, expected in zip(actual_gradient, source_gradient, strict=True):
        torch.testing.assert_close(value, expected, atol=3e-7, rtol=3e-6)

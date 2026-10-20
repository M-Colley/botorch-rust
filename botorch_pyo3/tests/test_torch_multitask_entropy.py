# ruff: noqa: E402
"""MES/GIBBON use selected-task designs without appending categorical columns."""

import pytest

pytest.importorskip("torch")


import torch
from test_torch_acq_discrete import DEVICES, no_host
from test_torch_backend_multitask import paired_models

from botorch_rs.acquisition import max_value_entropy_search as port
from botorch_rs.models import MultiTaskGP
from botorch_rs.sampling.normal import SobolQMCNormalSampler

NAMES = ["qMaxValueEntropy", "qLowerBoundMaxValueEntropy"]


def data(device, task_feature):
    points = torch.tensor(
        [[0.08, 0.2], [0.2, 0.3], [0.4, 0.6], [0.63, 0.5], [0.77, 0.8], [0.94, 0.9]],
        dtype=torch.double,
        device=device,
    )
    labels = points.new_tensor([[2.0], [5.0], [2.0], [5.0], [2.0], [5.0]])
    index = task_feature % 3
    X = torch.cat((points[:, :index], labels, points[:, index:]), -1)
    Y = (5.2 * points[:, :1]).sin() + 0.3 * (labels == 5)
    candidate = points.new_tensor([[0.15, 0.2], [0.5, 0.4], [0.85, 0.8]])
    return X, Y, points, candidate


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize("task_feature", [-1, 0, 1])
@pytest.mark.parametrize("full", [False, True])
def test_multitask_entropy_matches_corrected_source_and_retains_query_graph(
    device, name, task_feature, full
):
    from botorch.acquisition import max_value_entropy_search as reference
    from botorch.sampling.normal import SobolQMCNormalSampler as ReferenceSampler

    X, Y, continuous, candidate = data(device, task_feature)
    actual_model, expected_model = paired_models(
        X,
        Y,
        task_feature,
        noise=torch.full_like(Y, 0.03),
        output_tasks=[5],
        all_tasks=[2, 5],
        outcome_transform=None,
    )
    if full:
        index = task_feature % 3
        candidate = torch.cat(
            (candidate[:, :index], torch.full_like(candidate[:, :1], 5.0), candidate[:, index:]), -1
        )
    kwargs = dict(num_mv_samples=4, use_gumbel=False)
    if name == "qMaxValueEntropy":
        kwargs.update(num_y_samples=12, num_fantasies=3)
    # On unmodified official b0dd, automatic baseline concatenation fails for
    # task-free designs. This explicit baseline is the proposed source repair.
    expected = getattr(reference, name)(
        expected_model,
        candidate,
        train_inputs=X if full else continuous,
        **kwargs,
    )
    with no_host():
        actual = getattr(port, name)(actual_model, candidate, seed=37, **kwargs)
    expected_baseline = torch.cat((candidate, X if full else continuous), 0)
    torch.testing.assert_close(actual.candidate_set, expected_baseline)
    if name == "qMaxValueEntropy":
        actual.sampler = SobolQMCNormalSampler((12,), seed=19)
        expected.sampler = ReferenceSampler(torch.Size((12,)), seed=19)
    maxima = X.new_tensor([[0.8], [1.0], [1.3], [1.6]])
    actual.posterior_max_values = expected.posterior_max_values = maxima
    query = candidate[:2].unsqueeze(-2).clone().requires_grad_()
    reference_value = expected(query)
    reference_gradient = torch.autograd.grad(reference_value.sum(), query)[0]
    with no_host():
        value = actual(query)
        parameters = tuple(actual_model.parameters())
        gradients = torch.autograd.grad(value.sum(), (query, *parameters), create_graph=True)
        second = torch.autograd.grad(sum(g.square().sum() for g in gradients), query)[0]
    torch.testing.assert_close(value, reference_value, atol=1e-8, rtol=1e-7)
    torch.testing.assert_close(gradients[0], reference_gradient, atol=1e-6, rtol=1e-5)
    assert all(torch.isfinite(g).all() for g in gradients) and torch.isfinite(second).all()
    if full:
        assert not gradients[0][..., task_feature % 3].any()


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("name", NAMES)
def test_multitask_entropy_source_constructor_failure_and_explicit_train_baselines(device, name):
    from botorch.acquisition import max_value_entropy_search as reference

    X, Y, continuous, candidate = data(device, 1)
    model, source = paired_models(
        X, Y, 1, noise=torch.full_like(Y, 0.03), output_tasks=[5], outcome_transform=None
    )
    with pytest.raises(RuntimeError, match="Sizes of tensors must match"):
        getattr(reference, name)(source, candidate)
    kwargs = dict(num_mv_samples=3, use_gumbel=False, seed=13)
    with no_host():
        full = getattr(port, name)(model, candidate, train_inputs=X, **kwargs)
        reduced = getattr(port, name)(model, candidate, train_inputs=continuous, **kwargs)
    torch.testing.assert_close(full.candidate_set, reduced.candidate_set)
    torch.testing.assert_close(full.posterior_max_values, reduced.posterior_max_values)


@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize("kind", ["list", "numpy", "torch"])
def test_native_multitask_entropy_explicit_long_format_baseline_matches_default(name, kind):
    import numpy as np

    X, Y, _, candidate = data("cpu", 1)
    model = MultiTaskGP(X.tolist(), Y.tolist(), 1, output_tasks=[5], outcome_transform=None)
    convert = {
        "list": lambda x: x.tolist(),
        "numpy": lambda x: np.asarray(x),
        "torch": lambda x: x,
    }[kind]
    kwargs = dict(num_mv_samples=3, use_gumbel=False, seed=13)
    if name == "qMaxValueEntropy":
        kwargs["sampler"] = SobolQMCNormalSampler((16,), seed=19)
    automatic = getattr(port, name)(model, convert(candidate), **kwargs)
    explicit = getattr(port, name)(model, convert(candidate), train_inputs=convert(X), **kwargs)
    maxima = [[0.8], [1.1], [1.5]]
    automatic.posterior_max_values = explicit.posterior_max_values = maxima
    query = convert(candidate[:2].unsqueeze(-2))
    torch.testing.assert_close(torch.as_tensor(automatic(query)), torch.as_tensor(explicit(query)))

"""Correlated long-format conditioning, fantasies and observation-noise graphs."""

from unittest.mock import patch

import pytest
import torch
from test_torch_backend_multitask import DEVICES, data, paired_models

from botorch_rs.models import MultiTaskGP
from botorch_rs.sampling.normal import IIDNormalSampler


def observations(X, task_feature):
    continuous = X.new_tensor([0.3, 0.6])
    tasks = X.new_tensor([0.0, 1.0])
    return torch.stack((continuous, tasks) if task_feature == -1 else (tasks, continuous), -1)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("task_feature", [-1, 0])
@pytest.mark.parametrize("fixed", [False, True])
def test_conditioned_law_matches_official_with_broadcast_observations(device, task_feature, fixed):
    X, Y, Q = data(device, task_feature=task_feature)
    noise = torch.full_like(Y, 0.025) if fixed else None
    actual, expected = paired_models(X, Y, task_feature, noise, outcome_transform=None)
    new_X = observations(X, task_feature)
    new_Y = X.new_tensor([[[0.8], [0.3]], [[0.6], [0.5]], [[0.4], [0.1]]])
    extra_noise = torch.full_like(new_Y, 0.035) if fixed else None
    actual.posterior(new_X)
    expected.posterior(new_X)
    conditioned = actual.condition_on_observations(new_X, new_Y, noise=extra_noise)
    if fixed:
        # The unmodified fixed-noise fantasy shortcut fails to expand its
        # covariance blocks to the target batch. Independent reference updates
        # provide the same conditional laws without that broken fast path.
        with pytest.raises(RuntimeError, match="same number of dimensions"):
            expected.condition_on_observations(new_X, new_Y, noise=extra_noise)
        means, covariances = [], []
        for i in range(3):
            _, member = paired_models(X, Y, task_feature, noise, outcome_transform=None)
            member.posterior(new_X)
            updated = member.condition_on_observations(new_X, new_Y[i], noise=extra_noise[i])
            law = updated.posterior(Q[0])
            means.append(law.mean)
            covariances.append(law.distribution.covariance_matrix)
        expected_mean, expected_covariance = torch.stack(means), torch.stack(covariances)
    else:
        reference = expected.condition_on_observations(new_X, new_Y, noise=extra_noise)
        law = reference.posterior(Q[0].unsqueeze(0))
        expected_mean, expected_covariance = law.mean, law.distribution.covariance_matrix
    assert isinstance(conditioned, MultiTaskGP) and conditioned.batch_shape == (3,)
    query = Q[0].unsqueeze(0)
    pa = conditioned.posterior(query)
    torch.testing.assert_close(pa.mean, expected_mean, atol=2e-8, rtol=2e-8)
    torch.testing.assert_close(
        pa.distribution.covariance_matrix, expected_covariance, atol=2e-8, rtol=2e-8
    )
    assert conditioned._train_X.shape == (3, 8, 2)
    assert actual._train_X.shape == (6, 2)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("fixed", [False, True])
def test_conditioning_noise_and_data_keep_full_mixed_second_derivatives(device, fixed):
    X, Y, Q = data(device)
    X, Y = X.requires_grad_(), Y.requires_grad_()
    noise = torch.full_like(Y, 0.025, requires_grad=True) if fixed else None
    model = MultiTaskGP(X, Y, -1, noise, outcome_transform=None, backend="torch")
    new_X = observations(X, -1).requires_grad_()
    new_Y = X.new_tensor([[0.8], [0.3]], requires_grad=True)
    new_noise = torch.full_like(new_Y, 0.035, requires_grad=True)
    query = Q[0].requires_grad_()
    conditioned = model.condition_on_observations(new_X, new_Y, noise=new_noise)
    arguments = (X, Y, new_X, new_Y, new_noise, query, *model.parameters())
    if noise is not None:
        arguments = (*arguments, noise)

    def loss():
        posterior = conditioned.posterior(query)
        return (
            0.4 * posterior.mean.square().sum()
            + 0.2 * posterior.distribution.covariance_matrix.square().sum()
        )

    first = torch.autograd.grad(loss(), arguments, create_graph=True)
    directions = [torch.ones_like(t) * 0.07 for t in arguments]
    directions[0][..., -1] = 0
    directions[2][..., -1] = 0
    second = torch.autograd.grad(
        sum((g * v).sum() for g, v in zip(first, directions, strict=True)), arguments
    )

    # Rebuild conditioning for every perturbed leaf: appended training data are
    # graph-connected concatenations, rather than mutable aliases of these leaves.
    def first_at_current():
        updated = model.condition_on_observations(new_X, new_Y, noise=new_noise)
        posterior = updated.posterior(query)
        value = (
            0.4 * posterior.mean.square().sum()
            + 0.2 * posterior.distribution.covariance_matrix.square().sum()
        )
        return torch.autograd.grad(value, arguments)

    plus_minus = []
    step = 1e-5
    for sign in (1, -1):
        with torch.no_grad():
            for value, direction in zip(arguments, directions, strict=True):
                value.add_(direction, alpha=sign * step)
        plus_minus.append(first_at_current())
        with torch.no_grad():
            for value, direction in zip(arguments, directions, strict=True):
                value.add_(direction, alpha=-sign * step)
    for actual, positive, negative in zip(second, *plus_minus, strict=True):
        assert bool(torch.isfinite(actual).all())
        torch.testing.assert_close(actual, (positive - negative) / (2 * step), atol=3e-6, rtol=3e-5)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("fixed", [False, True])
def test_fantasy_sample_axes_and_batched_fixed_predictive_noise_stay_on_device(device, fixed):
    X, Y, Q = data(device)
    original_noise = (
        X.new_tensor([[0.01], [0.02], [0.03], [0.04], [0.05], [0.06]]) if fixed else None
    )
    model = MultiTaskGP(X, Y, -1, original_noise, backend="torch")
    new_X = observations(X, -1).requires_grad_()
    with (
        patch.object(torch.Tensor, "cpu", side_effect=AssertionError("host copy")),
        patch.object(torch.Tensor, "tolist", side_effect=AssertionError("host copy")),
        patch.object(torch.Tensor, "numpy", side_effect=AssertionError("host copy")),
    ):
        fantasy = model.fantasize(new_X, IIDNormalSampler((2, 3), seed=13), propagate_grads=True)
        assert fantasy.batch_shape == (2, 3)
        query = Q[0].requires_grad_()
        posterior = fantasy.posterior(query, observation_noise=True)
        assert posterior.mean.shape == (2, 3, 2, 2)
        first = torch.autograd.grad(
            posterior.mean.square().sum() + posterior.variance.sum(),
            (new_X, query),
            create_graph=True,
        )
        second = torch.autograd.grad(sum(g.square().sum() for g in first), (new_X, query))
        assert all(bool(torch.isfinite(g).all()) for g in second)
    assert posterior.mean.device == X.device
    if fixed:
        _, _, transformed = fantasy._training_values()
        torch.testing.assert_close(fantasy.likelihood.noise, transformed)
        assert model.likelihood.noise.shape == (6,)
        mapped = fantasy._mapped_inputs(fantasy._train_X)
        assert transformed.shape == (2, 3, 8)
        for task in (0, 1):
            full_query = new_X.new_tensor([[0.45, float(task)]])
            predicted = fantasy._fixed_noise_at(fantasy._mapped_inputs(full_query), transformed)
            mask = mapped[..., -1] == task
            expected = transformed.masked_select(mask).reshape(2, 3, 4).mean(-1, keepdim=True)
            torch.testing.assert_close(predicted, expected)


@pytest.mark.parametrize("device", DEVICES)
def test_noncontiguous_task_conditioning_and_fixed_noise_errors(device):
    X, Y, Q = data(device, labels=(2, 5))
    model = MultiTaskGP(X, Y, -1, torch.full_like(Y, 0.03), backend="torch")
    new_X = X.new_tensor([[0.45, 5.0]])
    with pytest.raises(ValueError, match="requires noise"):
        model.condition_on_observations(new_X, Y[:1])
    conditioned = model.condition_on_observations(new_X, Y[:1], noise=torch.full_like(Y[:1], 0.02))
    assert conditioned.posterior(Q[0]).mean.shape == (2, 2)
    with pytest.raises(ValueError, match="invalid task"):
        model.condition_on_observations(
            X.new_tensor([[0.45, 4.0]]), Y[:1], noise=torch.full_like(Y[:1], 0.02)
        )


@pytest.mark.parametrize("device", DEVICES)
def test_empty_fantasies_and_conditioning_previously_unobserved_task(device):
    X, Y, Q = data(device)
    model = MultiTaskGP(X, Y, -1, all_tasks=[0, 1, 2], backend="torch")
    model.posterior(Q[0])
    conditioned = model.condition_on_observations(X.new_tensor([[0.45, 2.0]]), Y[:1])
    assert conditioned._unobserved_task_indices.numel() == 0
    assert model._unobserved_task_indices.numel() == 1
    assert conditioned.posterior(Q[0], output_indices=[2]).mean.shape == (2, 1)
    empty = model.fantasize(X.new_empty(0, 2), IIDNormalSampler((2, 3), seed=7))
    assert empty.batch_shape == (2, 3)
    torch.testing.assert_close(
        empty.posterior(Q[0]).mean, model.posterior(Q[0]).mean.expand(2, 3, 2, 3)
    )

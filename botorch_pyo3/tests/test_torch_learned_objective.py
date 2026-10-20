"""Device learned utilities and the joint noisy-improvement probability law."""

# ruff: noqa: E402
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from test_torch_pairwise_gp import DEVICES, _pair

from botorch_rs.acquisition import logei, monte_carlo
from botorch_rs.acquisition.objective import LearnedObjective
from botorch_rs.models.gp_regression import SingleTaskGP
from botorch_rs.models.pairwise_gp import PairwiseGP
from botorch_rs.sampling.normal import IIDNormalSampler


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("batched", [False, True])
def test_learned_samples_and_first_query_derivative_independent_reference(device, batched):
    pytest.importorskip("botorch")
    from botorch.acquisition.objective import LearnedObjective as Reference

    models, _, data = _pair(device, "probit", batch=batched)
    results = []
    for i, model in enumerate(models):
        objective = (Reference if i else LearnedObjective)(
            model, sample_shape=torch.Size([4]), seed=82
        )
        samples = (
            torch.linspace(0.1, 0.9, 3 * 2 * 2, dtype=data[i].dtype, device=device)
            .reshape(3, 1, 2, 2)
            .requires_grad_()
        )
        # Pinned source rank-only broadcasting requires explicitly expanding
        # singleton query batches. That view preserves the original adjoint.
        evaluated = samples.expand(3, 2, 2, 2) if i and batched else samples
        utilities = objective(evaluated)
        gradient = torch.autograd.grad(utilities.square().sum(), samples)[0]
        results.append((utilities, gradient))
    for actual, expected in zip(*results, strict=True):
        torch.testing.assert_close(actual, expected, atol=3e-8, rtol=3e-7)


def _models(device):
    x = torch.tensor(
        [[0.1, 0.2], [0.3, 0.8], [0.7, 0.4], [0.9, 0.7]],
        dtype=torch.double,
        device=device,
        requires_grad=True,
    )
    y = torch.stack((x.detach()[:, 0].sin(), x.detach()[:, 1].square()), -1).requires_grad_()
    noise = torch.full_like(y, 0.03, requires_grad=True)
    outcome = SingleTaskGP(x, y, noise, outcome_transform=None, backend="torch")
    preference_x = torch.tensor(
        [[0.1, 0.3], [0.4, 0.2], [0.7, 0.8], [0.3, 0.6]],
        dtype=x.dtype,
        device=device,
        requires_grad=True,
    )
    comparisons = torch.tensor([[0, 1], [2, 1], [2, 3], [3, 0]], device=device)
    preference = PairwiseGP(preference_x, comparisons, backend="torch")
    return outcome, preference, (x, y, noise, preference_x)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("name", ["qNoisyExpectedImprovement", "qLogNoisyExpectedImprovement"])
def test_joint_learned_noisy_improvement_cache_request_and_complete_graphs(device, name):
    outcome, preference, data = _models(device)
    query = data[0].new_tensor([[0.25, 0.45], [0.6, 0.3]], requires_grad=True)
    cls = getattr(logei if name.startswith("qLog") else monte_carlo, name)
    results = []
    for cached in (False, True):
        objective = LearnedObjective(preference, sample_shape=torch.Size([4]), seed=82)
        kwargs = dict(
            model=outcome,
            X_baseline=data[0][:2],
            objective=objective,
            cache_root=cached,
            prune_baseline=False,
            sampler=IIDNormalSampler(torch.Size([8]), seed=83),
        )
        if cached:
            with pytest.warns(RuntimeWarning, match="full-joint"):
                acquisition = cls(**kwargs)
        else:
            acquisition = cls(**kwargs)
        value = acquisition(query)
        assert acquisition.baseline_obj.shape[0] == 32
        assert not acquisition._cache_root
        targets = (
            *data,
            query,
            *(p for p in outcome.parameters() if p.requires_grad),
            *(p for p in preference.parameters() if p.requires_grad),
        )
        first = torch.autograd.grad(value.sum(), targets, create_graph=True)
        second = torch.autograd.grad(sum(g.square().sum() for g in first), targets)
        assert all(bool(torch.isfinite(t).all()) for t in (value, *first, *second))
        results.append((value, *first, *second))
    for left, right in zip(*results, strict=True):
        torch.testing.assert_close(left, right, atol=0, rtol=0)
    if name == "qNoisyExpectedImprovement":
        duplicate = acquisition(data[0][:1])
        # Only the small numerical covariance-root jitter can separate an
        # exactly repeated candidate and baseline draw under the joint law.
        assert float(duplicate.detach()) < 0.002


@pytest.mark.parametrize("device", DEVICES)
def test_multiple_preference_sample_dimensions_and_no_native_handle(device, monkeypatch):
    _, preference, data = _models(device)
    objective = LearnedObjective(preference, sample_shape=torch.Size([2, 3]), seed=82)

    def forbidden(*args, **kwargs):
        raise AssertionError("Torch learned utilities must not flatten into a native model handle.")

    monkeypatch.setattr(preference, "_acquisition_model", forbidden)
    samples = (
        data[0].new_tensor([[[[0.2, 0.4], [0.7, 0.3]]]]).expand(4, 2, 2, 2).clone().requires_grad_()
    )
    values = objective(samples)
    assert values.shape == (24, 2, 2)
    first = torch.autograd.grad(values.square().sum(), samples, create_graph=True)[0]
    second = torch.autograd.grad(first.square().sum(), samples)[0]
    assert bool(torch.isfinite(second).all())


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("batched", [False, True])
def test_default_pruning_augmented_utilities_and_independent_backwards(device, batched):
    outcome, preference, data = _models(device)
    if batched:
        previous = preference
        preference = PairwiseGP(
            previous.datapoints.unsqueeze(0).expand(2, -1, -1).clone(),
            previous.comparisons.unsqueeze(0).expand(2, -1, -1).clone(),
            backend="torch",
        )
    objective = LearnedObjective(preference, sample_shape=(4,), seed=82)
    acquisition = logei.qLogNoisyExpectedImprovement(
        outcome,
        data[0][:2],
        objective=objective,
        cache_root=False,
        sampler=IIDNormalSampler((8,), seed=83),
    )
    query = data[0].new_tensor([[0.25, 0.45]], requires_grad=True)
    targets = (*data, query, *(p for p in preference.parameters() if p.requires_grad))
    results = []
    for _ in range(2):
        value = acquisition(query)
        assert value.shape == ((2,) if batched else (1,))
        first = torch.autograd.grad(value.sum(), targets, create_graph=True)
        second = torch.autograd.grad(sum(g.square().sum() for g in first), targets)
        results.append((value, *first, *second))
    for left, right in zip(*results, strict=True):
        torch.testing.assert_close(left, right, atol=0, rtol=0)

"""Registered outcome buffers control native inverse maps after fitting."""

# ruff: noqa: E402
from __future__ import annotations

import copy

import pytest

torch = pytest.importorskip("torch")

from botorch_rs import _botorch_rs as _rs
from botorch_rs.acquisition.analytic import ExpectedImprovement
from botorch_rs.models.contextual_multioutput import LCEMGP
from botorch_rs.models.gp_regression import SingleTaskGP
from botorch_rs.models.higher_order_gp import HigherOrderGP
from botorch_rs.models.latent_kronecker_gp import LatentKroneckerGP
from botorch_rs.models.multitask import KroneckerMultiTaskGP, MultiTaskGP
from botorch_rs.models.transforms.outcome import ChainedOutcomeTransform, Log, Standardize


def _model(family):
    x = torch.tensor([[0.1], [0.35], [0.7], [0.9]], dtype=torch.double)
    y = torch.tensor([[1.2], [1.7], [2.4], [3.1]], dtype=torch.double)
    if family in {"multitask", "lcem"}:
        x = torch.cat((x, torch.tensor([[0.0], [1.0], [0.0], [1.0]])), -1)
        cls = MultiTaskGP if family == "multitask" else LCEMGP
        return cls(x, y, task_feature=1, backend="native"), x[:2]
    if family in {"kronecker", "higher", "latent"}:
        y = torch.cat((y, y * 1.4 - 0.2), -1)
        cls = {
            "kronecker": KroneckerMultiTaskGP,
            "higher": HigherOrderGP,
            "latent": LatentKroneckerGP,
        }[family]
        if family == "latent":
            return cls(x, x.new_tensor([[0.0], [1.0]]), y, backend="native"), x[:2]
        if family == "kronecker":
            return cls(x, y, outcome_transform=Standardize(2), backend="native"), x[:2]
        return cls(x, y, backend="native"), x[:2]
    if family.startswith("batched"):
        x = x.expand(2, 4, 1).clone()
        y = y.expand(2, 4, 1).clone() + y.new_tensor([0.0, 0.6])[:, None, None]
    if family in {"multioutput", "batched_multioutput"}:
        y = torch.cat((y, y * 1.4 - 0.2), -1)
    return SingleTaskGP(x, y, backend="native"), x[..., :2, :]


@pytest.mark.parametrize(
    "family",
    [
        "scalar",
        "multioutput",
        "batched",
        "batched_multioutput",
        "multitask",
        "lcem",
        "kronecker",
        "higher",
        "latent",
    ],
)
@pytest.mark.parametrize("mutation", ["means", "stdvs", "load_state_dict"])
def test_native_inverse_statistics_follow_registered_state_without_refitting(family, mutation):
    assert hasattr(_rs.SingleTaskGP, "_set_fitted_outcome_state")
    model, query = _model(family)
    before = model.posterior(query)
    old_mean, old_variance = before.mean.clone(), before.variance.clone()
    transform = model.outcome_transform
    old_offset, old_scale = transform.means.clone(), transform.stdvs.clone()
    targets = model.train_targets.clone()
    clone = copy.deepcopy(model)
    original = {key: value.clone() for key, value in transform.state_dict().items()}
    if mutation == "load_state_dict":
        changed = {key: value.clone() for key, value in original.items()}
        changed["means"].add_(23.0)
        changed["stdvs"].mul_(1.6)
        changed["_stdvs_sq"].copy_(changed["stdvs"].square())
        transform.load_state_dict(changed)
    elif mutation == "means":
        dict(model.named_buffers())["outcome_transform.means"].add_(23.0)
    else:
        dict(model.named_buffers())["outcome_transform.stdvs"].mul_(1.6)
        transform._stdvs_sq.copy_(transform.stdvs.square())
    expected_mean = (old_mean - old_offset) / old_scale * transform.stdvs + transform.means
    expected_variance = old_variance * (transform.stdvs / old_scale).square()
    after = model.posterior(query)
    torch.testing.assert_close(after.mean, expected_mean, atol=3e-10, rtol=3e-10)
    torch.testing.assert_close(after.variance, expected_variance, atol=3e-10, rtol=3e-10)
    torch.testing.assert_close(model.train_targets, targets, atol=0, rtol=0)
    torch.testing.assert_close(clone.posterior(query).mean, old_mean, atol=3e-10, rtol=3e-10)
    transform.load_state_dict(original)
    restored = model.posterior(query)
    torch.testing.assert_close(restored.mean, old_mean, atol=3e-10, rtol=3e-10)
    torch.testing.assert_close(restored.variance, old_variance, atol=3e-10, rtol=3e-10)


@pytest.mark.parametrize("batch", [False, True])
def test_native_fitted_chain_preserves_inner_statistics_when_one_buffer_changes(batch):
    x = torch.tensor([[0.1], [0.35], [0.7], [0.9]], dtype=torch.double)
    y = torch.tensor([[1.2], [1.7], [2.4], [3.1]], dtype=torch.double)
    if batch:
        x, y = x.expand(2, 4, 1), y.expand(2, 4, 1)
    shape = (2,) if batch else ()
    outcome = ChainedOutcomeTransform(
        log=Log(),
        first=Standardize(1, batch_shape=shape),
        second=Standardize(1, batch_shape=shape),
    )
    model = SingleTaskGP(x, y, outcome_transform=outcome, backend="native")
    query = x[..., :2, :]
    before = model.posterior(query).mean.clone()
    inner = outcome.second.means.clone()
    outcome.first.means.add_(0.4)
    # The outer log inverse exponentiates the shift in its standardized input.
    expected = before * torch.exp(torch.full_like(before, 0.4))
    torch.testing.assert_close(model.posterior(query).mean, expected, atol=3e-10, rtol=3e-10)
    torch.testing.assert_close(outcome.second.means, inner, atol=0, rtol=0)


def test_native_acquisition_snapshots_refresh_after_outcome_state_changes():
    model, query = _model("scalar")
    query = query[:1].clone().requires_grad_()
    acquisition = ExpectedImprovement(model, best_f=2.0)
    old = acquisition(query).detach()
    model.outcome_transform.means.add_(3.0)
    with pytest.raises(RuntimeError, match="changed|stale|modified"):
        acquisition(query)
    refreshed = ExpectedImprovement(model, best_f=5.0)
    torch.testing.assert_close(refreshed(query), old)
    assert torch.isfinite(torch.autograd.grad(refreshed(query).sum(), query)[0]).all()


def test_unmodified_source_buffer_mutation_has_the_same_native_prediction_law():
    source = pytest.importorskip("botorch.models.gp_regression")
    model, query = _model("scalar")
    x = model.train_inputs[0]
    y = x.new_tensor([[1.2], [1.7], [2.4], [3.1]])
    reference = source.SingleTaskGP(x, y)
    reference.posterior(query)
    model.posterior(query)
    for candidate in (model, reference):
        dict(candidate.named_buffers())["outcome_transform.means"].fill_(100.0)
        candidate.outcome_transform.stdvs.mul_(1.3)
    actual, expected = model.posterior(query), reference.posterior(query)
    torch.testing.assert_close(actual.mean, expected.mean, atol=3e-10, rtol=3e-10)
    torch.testing.assert_close(actual.variance, expected.variance, atol=3e-10, rtol=3e-10)


@pytest.mark.parametrize("family", ["scalar", "multitask", "lcem"])
def test_conditioning_synchronizes_changed_buffers_before_transforming_new_targets(family):
    model, query = _model(family)
    model.posterior(query)
    oracle = copy.deepcopy(model)
    for candidate in (model, oracle):
        candidate.outcome_transform.means.add_(5.0)
        candidate.outcome_transform.stdvs.mul_(1.2)
        candidate.outcome_transform._stdvs_sq.copy_(candidate.outcome_transform.stdvs.square())
    x = query.new_tensor([[0.27]] if family == "scalar" else [[0.27, 0.0]])
    test = query.new_tensor([[0.61]] if family == "scalar" else [[0.61, 1.0]])
    y, noise = query.new_tensor([[1.6]]), query.new_tensor([[0.07]])
    joint = oracle.posterior(torch.cat((x, test)))
    mean, covariance = joint.mean[:, 0], joint.distribution.covariance_matrix
    inverse = (
        covariance[0, 0] + noise.squeeze() * oracle.outcome_transform.stdvs.square().squeeze()
    ).reciprocal()
    expected_mean = mean[1] + covariance[1, 0] * inverse * (y.squeeze() - mean[0])
    expected_variance = covariance[1, 1] - covariance[1, 0] * inverse * covariance[0, 1]
    actual = model.condition_on_observations(x, y, noise=noise).posterior(test)
    torch.testing.assert_close(actual.mean.squeeze(), expected_mean, atol=3e-10, rtol=3e-10)
    torch.testing.assert_close(actual.variance.squeeze(), expected_variance, atol=3e-10, rtol=3e-10)


def test_lcem_fantasy_batch_synchronizes_every_member_inverse_map():
    model, query = _model("lcem")
    model.posterior(query)
    x = query.new_tensor([[[0.27, 0.0]], [[0.47, 1.0]]])
    y = query.new_tensor([[[1.6]], [[2.1]]])
    fantasy = model.condition_on_observations(x, y)
    before = fantasy.posterior(query).mean.clone()
    fantasy.outcome_transform.means.add_(8.0)
    torch.testing.assert_close(fantasy.posterior(query).mean, before + 8.0)


@pytest.mark.parametrize("multitask", [False, True])
def test_fully_bayesian_registered_buffers_refresh_all_mcmc_members(multitask):
    from test_torch_fully_bayesian_multitask import _data, _samples

    from botorch_rs.models.fully_bayesian import SaasFullyBayesianSingleTaskGP
    from botorch_rs.models.fully_bayesian_multitask import SaasFullyBayesianMultiTaskGP

    x, y, noise = _data("cpu")
    if not multitask:
        x = x[:, [0, 2]]
    samples = _samples(x)
    if not multitask:
        samples = {
            key: value
            for key, value in samples.items()
            if key not in {"latent_features", "task_lengthscale", "c0", "c1"}
        }
        samples["mean"] = samples["mean"][:, 0]
    cls = SaasFullyBayesianMultiTaskGP if multitask else SaasFullyBayesianSingleTaskGP
    outcome = ChainedOutcomeTransform(first=Standardize(1), second=Standardize(1))
    model = cls(
        x,
        y,
        train_Yvar=noise,
        outcome_transform=outcome,
        backend="native",
        **({"task_feature": 1, "rank": 2} if multitask else {}),
    )
    model.load_mcmc_samples(samples)
    query = x.new_tensor([[0.2, 0.7], [0.6, 0.4]])
    before = model.posterior(query).mean.clone()
    model.outcome_transform.first.means.add_(7.0)
    torch.testing.assert_close(model.posterior(query).mean, before + 7.0)

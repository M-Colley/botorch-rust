"""Independent nested ensemble integrations, with full data and parameter graphs."""

# ruff: noqa: E402
from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")
from test_torch_pairwise_gp import DEVICES

from botorch_rs.acquisition import logei, monte_carlo
from botorch_rs.acquisition.objective import LearnedObjective
from botorch_rs.models import ModelListGP, SingleTaskGP
from botorch_rs.models.fully_bayesian import SaasFullyBayesianSingleTaskGP
from botorch_rs.sampling.normal import IIDNormalSampler


def _loaded(x, y, count):
    noise = torch.full_like(y, 0.04, requires_grad=True)
    model = SaasFullyBayesianSingleTaskGP(x, y, noise, outcome_transform=None, backend="torch")
    model.load_mcmc_samples(
        {
            "mean": torch.linspace(-0.2, 0.25, count, dtype=x.dtype, device=x.device),
            "lengthscale": torch.linspace(
                0.3, 0.9, count * x.shape[-1], dtype=x.dtype, device=x.device
            ).reshape(count, -1),
            "outputscale": torch.linspace(0.7, 1.3, count, dtype=x.dtype, device=x.device),
        }
    )
    return model.eval(), noise


def _models(device, nested=False, batch=False):
    x = torch.tensor(
        [[0.1, 0.2], [0.3, 0.8], [0.7, 0.4], [0.9, 0.7]],
        dtype=torch.double,
        device=device,
        requires_grad=True,
    )
    y = torch.tensor(
        [[0.2, 0.7], [0.6, 0.3], [0.5, 0.8], [0.9, 0.4]],
        dtype=x.dtype,
        device=device,
        requires_grad=True,
    )
    if nested:
        a, na = _loaded(x, y[:, :1], 2)
        b, nb = _loaded(x, y[:, 1:], 2)
        outcome = ModelListGP(a, b)
        data = [x, y, na, nb]
    else:
        xx, yy = (
            (x.unsqueeze(0).expand(2, -1, -1), y.unsqueeze(0).expand(2, -1, -1))
            if batch
            else (x, y)
        )
        noise = torch.full_like(yy, 0.04, requires_grad=True)
        outcome = SingleTaskGP(xx, yy, noise, outcome_transform=None, backend="torch").eval()
        data = [x, y, noise]
    px = torch.tensor(
        [[0.1, 0.3], [0.4, 0.2], [0.7, 0.8], [0.3, 0.6]],
        dtype=x.dtype,
        device=device,
        requires_grad=True,
    )
    py = torch.tensor(
        [[0.2], [0.7], [0.8], [0.3]], dtype=x.dtype, device=device, requires_grad=True
    )
    preference, pn = _loaded(px, py, 3)
    return outcome, preference, (*data, px, py, pn)


def _explicit_member_utility(objective, samples):
    # The model covariance law is already independently checked against official
    # models. Here each preference member is sampled separately using explicit
    # matrix Cholesky and common normals, without LearnedObjective.forward.
    posterior = objective.pref_model.posterior(samples)
    objective.sampler._construct_base_samples(posterior)
    axes = len(objective.sampler.sample_shape)
    normals = objective.sampler.base_samples.expand(
        *objective.sampler.sample_shape, *posterior.base_sample_shape
    )
    pieces = []
    for member in range(posterior.mean.shape[-3]):
        mean = posterior.mean[..., member, :, 0]
        covariance = posterior.distribution.covariance_matrix[..., member, :, :]
        draw = mean + (
            torch.linalg.cholesky(covariance) @ normals[..., member, :].unsqueeze(-1)
        ).squeeze(-1)
        pieces.append(draw.reshape(-1, *draw.shape[axes + 1 :]))
    return pieces


def _manual_reduction(acquisition, member, baseline_n=0):
    if baseline_n:
        best = member[..., :baseline_n].amax(-1)
        if acquisition.detach_baseline:
            best = best.detach()
        values = member[..., baseline_n:]
    else:
        best = (
            torch.as_tensor(acquisition.best_f, dtype=member.dtype, device=member.device)
            if hasattr(acquisition, "best_f")
            else None
        )
        values = member
    name = acquisition._name
    if name == "qSimpleRegret":
        utility = values
    elif name in {"qUpperConfidenceBound", "qLowerConfidenceBound", "qPosteriorStandardDeviation"}:
        center = values.mean(0)
        deviation = (values - center).abs()
        if name == "qPosteriorStandardDeviation":
            utility = deviation * math.sqrt(math.pi / 2)
        else:
            factor = torch.as_tensor(
                acquisition.beta, dtype=member.dtype, device=member.device
            ).sqrt() * math.sqrt(math.pi / 2)
            utility = (
                center
                + (-1 if name == "qLowerConfidenceBound" else 1) * factor.unsqueeze(-1) * deviation
            )
    elif name == "qProbabilityOfImprovement":
        utility = ((values - best.unsqueeze(-1)) / acquisition.tau.to(values)).sigmoid()
    elif acquisition._log:
        from botorch.acquisition.logei import _log_improvement

        utility = _log_improvement(values, best, acquisition.tau_relu.to(values), acquisition.fat)
    else:
        utility = (values - best.unsqueeze(-1)).clamp_min(0)
    if acquisition._log:
        from botorch.utils.safe_math import fatmax

        return torch.logsumexp(
            fatmax(utility, dim=-1, tau=acquisition.tau_max.to(values)), dim=0
        ) - math.log(utility.shape[0])
    return utility.amax(-1).mean(0)


FAMILIES = [
    "qExpectedImprovement",
    "qLogExpectedImprovement",
    "qProbabilityOfImprovement",
    "qSimpleRegret",
    "qUpperConfidenceBound",
    "qLowerConfidenceBound",
    "qPosteriorStandardDeviation",
    "qNoisyExpectedImprovement",
    "qLogNoisyExpectedImprovement",
]


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("nested,batch", [(False, False), (False, True), (True, False)])
@pytest.mark.parametrize("name", FAMILIES)
def test_all_scalar_mc_integrate_preference_members_after_nonlinear_utilities(
    device, nested, batch, name
):
    outcome, preference, data = _models(device, nested, batch)
    objective = LearnedObjective(preference, sample_shape=(2,), seed=31)
    cls = getattr(logei if "Log" in name else monte_carlo, name)
    kwargs = dict(model=outcome, objective=objective, sampler=IIDNormalSampler((4,), seed=32))
    noisy = "Noisy" in name
    if noisy:
        kwargs.update(X_baseline=data[0][:2], prune_baseline=False, cache_root=False)
    elif "Improvement" in name:
        kwargs["best_f"] = data[0].new_tensor([0.15, 0.25]) if batch else 0.2
    if "ConfidenceBound" in name:
        kwargs["beta"] = data[0].new_tensor([0.3, 0.5]) if batch else 0.4
    acquisition = cls(**kwargs)
    query = data[0].new_tensor([[[0.25, 0.45], [0.6, 0.3]]], requires_grad=True)
    actual = acquisition(query)
    joint = torch.cat((data[0][:2].unsqueeze(0), query), -2) if noisy else query
    samples = acquisition.sampler(outcome.posterior(joint))
    member_values = torch.stack(
        [
            _manual_reduction(acquisition, m, 2 if noisy else 0)
            for m in _explicit_member_utility(objective, samples)
        ],
        -1,
    )
    expected = (
        torch.logsumexp(member_values, -1) - math.log(3)
        if acquisition._log
        else member_values.mean(-1)
    )
    if nested:
        expected = (
            torch.logsumexp(expected, -1) - math.log(2) if acquisition._log else expected.mean(-1)
        )
    torch.testing.assert_close(actual, expected, atol=3e-11, rtol=3e-9)
    assert actual.shape == ((2,) if batch else (1,))
    # Every configuration checks the query gradient. Representative nonlinear
    # noisy cases additionally check all data/parameter Hessian-vector graphs.
    targets = (query,)
    if name in {"qLogNoisyExpectedImprovement", "qUpperConfidenceBound"} and not batch:
        targets = (
            *data,
            query,
            *(p for p in outcome.parameters() if p.requires_grad),
            *(p for p in preference.parameters() if p.requires_grad),
        )
    first = torch.autograd.grad(actual.sum(), targets, create_graph=True, retain_graph=True)
    oracle = torch.autograd.grad(expected.sum(), targets, create_graph=True, retain_graph=True)
    for a, b in zip(first, oracle, strict=True):
        torch.testing.assert_close(a, b, atol=3e-9, rtol=3e-7)
    second = torch.autograd.grad(sum(g.square().sum() for g in first), targets, retain_graph=True)
    oracle_second = torch.autograd.grad(sum(g.square().sum() for g in oracle), targets)
    for a, b in zip(second, oracle_second, strict=True):
        torch.testing.assert_close(a, b, atol=3e-8, rtol=3e-6)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("nested", [False, True])
@pytest.mark.parametrize("mode", ["direct", "legacy", "penalized"])
def test_default_pruning_constraints_composition_and_independent_full_backwards(
    device, nested, mode
):
    from botorch_rs.acquisition.objective import ConstrainedMCObjective
    from botorch_rs.acquisition.penalized import L1PenaltyObjective, PenalizedMCObjective

    outcome, preference, data = _models(device, nested)
    learned = LearnedObjective(preference, sample_shape=(2, 3), seed=31)

    def constraint(samples):
        return samples[..., 0] - 1.5

    objective = learned
    extra = {}
    cls = logei.qLogNoisyExpectedImprovement
    if mode == "direct":
        extra["constraints"] = [constraint]
        extra["eta"] = 0.15
    elif mode == "legacy":
        objective = ConstrainedMCObjective(learned, [constraint], infeasible_cost=3.0, eta=0.15)
        cls = monte_carlo.qNoisyExpectedImprovement
    else:
        objective = PenalizedMCObjective(learned, L1PenaltyObjective(data[0].new_zeros(2)), 0.02)
    acquisition = cls(
        outcome,
        data[0][:3],
        objective=objective,
        cache_root=False,
        sampler=IIDNormalSampler((4,), seed=32),
        detach_baseline=False,
        **extra,
    )
    query = data[0].new_tensor([[[0.25, 0.45], [0.6, 0.3]]], requires_grad=True)
    variables = (
        *data,
        query,
        *(p for p in outcome.parameters() if p.requires_grad),
        *(p for p in preference.parameters() if p.requires_grad),
    )
    results = []
    for _ in range(2):
        value = acquisition(query)
        assert value.shape == (1,)
        first = torch.autograd.grad(value.sum(), variables, create_graph=True)
        second = torch.autograd.grad(sum(g.square().sum() for g in first), variables)
        assert all(torch.isfinite(g).all() for g in (value, *first, *second))
        results.append((value, *first, *second))
    for a, b in zip(*results, strict=True):
        torch.testing.assert_close(a, b, atol=0, rtol=0)
    assert 0 < acquisition.X_baseline.shape[-2] <= 3


@pytest.mark.parametrize("device", DEVICES)
def test_legacy_constraint_forwards_X_and_aligns_exact_preference_member_feasibility(device):
    from botorch_rs.acquisition.objective import ConstrainedMCObjective, GenericMCObjective

    outcome, preference, data = _models(device)
    learned = LearnedObjective(preference, sample_shape=(2, 3), seed=31)
    samples = IIDNormalSampler((4,), seed=32)(outcome.posterior(data[0][:2]))
    query = data[0].new_tensor([[0.2, 0.3], [0.7, 0.5]], requires_grad=True)
    seen = []

    def utility(values, X=None):
        seen.append(X)
        return learned(values) + X.sum(-1).unsqueeze(-2)

    # Callable adapters can explicitly declare their ensemble axis protocol.
    utility._tensor_preference_ensemble = True
    wrapped = GenericMCObjective(utility)
    objective = ConstrainedMCObjective(
        wrapped, [lambda values: values[..., 1] - 0.5], eta=0.15, infeasible_cost=3.0
    )
    actual = objective(samples, X=query)
    raw = learned(samples) + query.sum(-1).unsqueeze(-2)
    feasibility = torch.nn.functional.logsigmoid(-((samples[..., 1] - 0.5) / 0.15)).exp()
    feasibility = feasibility.repeat(6, *([1] * (feasibility.ndim - 1))).unsqueeze(-2)
    expected = (raw + 3.0).clamp_min(0) * feasibility - 3.0
    torch.testing.assert_close(actual, expected, atol=3e-14, rtol=3e-12)
    assert seen == [query]
    first = torch.autograd.grad(actual.sum(), query, create_graph=True)[0]
    oracle = torch.autograd.grad(expected.sum(), query, create_graph=True)[0]
    torch.testing.assert_close(first, oracle)


@pytest.mark.parametrize("device", DEVICES)
def test_all_infeasible_learned_baseline_lower_bound_has_explicit_sample_axis(device):
    outcome, preference, data = _models(device)
    objective = LearnedObjective(preference, sample_shape=(2,), seed=31)
    acquisition = logei.qLogNoisyExpectedImprovement(
        outcome,
        data[0][:2],
        objective=objective,
        cache_root=False,
        prune_baseline=False,
        constraints=[lambda values: values[..., 0] * 0 + 1.0],
        eta=0.3,
        sampler=IIDNormalSampler((4,), seed=32),
        detach_baseline=False,
    )
    query = data[0].new_tensor([[[0.25, 0.45]]], requires_grad=True)
    value = acquisition(query)
    assert value.shape == (1,) and torch.isfinite(value).all()
    first = torch.autograd.grad(value.sum(), query, create_graph=True)[0]
    second = torch.autograd.grad(first.square().sum(), query)[0]
    assert torch.isfinite(first).all() and torch.isfinite(second).all()


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize(
    "name", ["qLogNoisyExpectedImprovement", "qUpperConfidenceBound", "qSimpleRegret"]
)
def test_three_ordinary_preference_batches_cross_two_outcome_members(device, name):
    outcome, _, data = _models(device, nested=True)
    data = data[:4]
    px = data[0].detach().clone().requires_grad_()
    py = torch.tensor(
        [[0.2], [0.7], [0.8], [0.3]], dtype=px.dtype, device=device, requires_grad=True
    )
    offset = px.new_tensor([-0.2, 0.1, 0.4]).reshape(3, 1, 1)
    noise = torch.full((3, 4, 1), 0.035, dtype=px.dtype, device=device, requires_grad=True)
    preference = SingleTaskGP(
        px.unsqueeze(0).expand(3, -1, -1),
        py + offset,
        noise,
        outcome_transform=None,
        backend="torch",
    ).eval()
    objective = LearnedObjective(preference, sample_shape=(2,), seed=31)
    cls = getattr(logei if "Log" in name else monte_carlo, name)
    kwargs = dict(model=outcome, objective=objective, sampler=IIDNormalSampler((4,), seed=32))
    noisy = "Noisy" in name
    if noisy:
        kwargs.update(
            X_baseline=data[0][:2], prune_baseline=False, cache_root=False, detach_baseline=False
        )
    if "ConfidenceBound" in name:
        kwargs["beta"] = 0.4
    acquisition = cls(**kwargs)
    query = data[0].new_tensor([[[0.25, 0.45], [0.6, 0.3]]], requires_grad=True)
    actual = acquisition(query)
    joint = torch.cat((data[0][:2].unsqueeze(0), query), -2) if noisy else query
    samples = acquisition.sampler(outcome.posterior(joint))
    explicit = []
    for member in range(2):
        posterior = preference.posterior(samples[..., member, :, :])
        normals = objective.sampler.base_samples.reshape(2, 4, 1, joint.shape[-2]).expand(
            2, 4, 3, joint.shape[-2]
        )
        draws = posterior.mean[..., 0] + (
            torch.linalg.cholesky(posterior.distribution.covariance_matrix) @ normals.unsqueeze(-1)
        ).squeeze(-1)
        explicit.append(
            _manual_reduction(acquisition, draws.reshape(8, 3, joint.shape[-2]), 2 if noisy else 0)
        )
    expected = torch.stack(explicit, -1)
    expected = (
        torch.logsumexp(expected, -1) - math.log(2) if acquisition._log else expected.mean(-1)
    )
    assert actual.shape == (3,)
    torch.testing.assert_close(actual, expected, atol=3e-11, rtol=3e-9)
    variables = (
        *data,
        px,
        py,
        noise,
        query,
        *(p for p in outcome.parameters() if p.requires_grad),
        *preference.parameters(),
    )
    first = torch.autograd.grad(actual.sum(), variables, create_graph=True, retain_graph=True)
    oracle = torch.autograd.grad(expected.sum(), variables, create_graph=True, retain_graph=True)
    for a, b in zip(first, oracle, strict=True):
        torch.testing.assert_close(a, b, atol=3e-9, rtol=3e-7)
    second = torch.autograd.grad(sum(g.square().sum() for g in first), variables, retain_graph=True)
    oracle_second = torch.autograd.grad(sum(g.square().sum() for g in oracle), variables)
    for a, b in zip(second, oracle_second, strict=True):
        torch.testing.assert_close(a, b, atol=3e-8, rtol=3e-6)


def test_objective_ensemble_context_resets_after_exception_and_nested_calls():
    from botorch_rs.acquisition.objective import (
        GenericMCObjective,
        _tensor_objective_context,
        _tensor_outcome_ensemble,
    )

    outcome, _, data = _models("cpu", nested=True)

    def fail(samples, X=None):
        assert _tensor_outcome_ensemble.get()
        with _tensor_objective_context(False):
            assert not _tensor_outcome_ensemble.get()
        assert _tensor_outcome_ensemble.get()
        raise RuntimeError("caller objective failed")

    acquisition = monte_carlo.qSimpleRegret(
        outcome, objective=GenericMCObjective(fail), sampler=IIDNormalSampler((2,), seed=32)
    )
    assert not _tensor_outcome_ensemble.get()
    with pytest.raises(RuntimeError, match="caller objective failed"):
        acquisition(data[0][:1])
    assert not _tensor_outcome_ensemble.get()


@pytest.mark.parametrize("device", DEVICES)
def test_penalty_query_batch_broadcasts_over_both_independent_member_axes(device):
    from botorch_rs.acquisition.objective import _tensor_objective_context
    from botorch_rs.acquisition.penalized import L1PenaltyObjective, PenalizedMCObjective

    outcome, preference, data = _models(device, nested=True)
    query = data[0].new_tensor(
        [[[0.25, 0.45], [0.6, 0.3]], [[0.35, 0.25], [0.7, 0.4]], [[0.45, 0.3], [0.6, 0.5]]],
        requires_grad=True,
    )
    samples = IIDNormalSampler((4,), seed=32)(outcome.posterior(query))
    learned = LearnedObjective(preference, sample_shape=(2,), seed=31)
    penalty = L1PenaltyObjective(data[0].new_zeros(2))
    objective = PenalizedMCObjective(learned, penalty, 0.02)
    with _tensor_objective_context(True):
        actual = objective(samples, X=query)
        expected = learned(samples) - 0.02 * penalty(query).unsqueeze(-2).unsqueeze(-2)
    assert actual.shape == (8, 3, 2, 3, 2)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    first = torch.autograd.grad(actual.sum(), query, create_graph=True)[0]
    second = torch.autograd.grad(first.square().sum(), query)[0]
    assert torch.isfinite(first).all() and torch.isfinite(second).all()

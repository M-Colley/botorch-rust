"""Preference-member selection inside native scalar MC acquisition workers."""

# ruff: noqa: E402

from __future__ import annotations

import pytest

from helpers import KINDS, as_kind, assert_close, importorskip_botorch

importorskip_botorch()
torch = pytest.importorskip("torch")

from botorch.acquisition import logei as bt_log
from botorch.acquisition import monte_carlo as bt_mc
from botorch.acquisition.objective import LearnedObjective as BtLearned
from mo_acquisition_helpers import bt_sampler, rs_sampler, synced_draws
from test_batched_learned_objective import models as preference_models
from test_learned_objective_mc import models as outcome_models

from botorch_rs import _torch_bridge
from botorch_rs.acquisition import logei as rs_log
from botorch_rs.acquisition import monte_carlo as rs_mc
from botorch_rs.acquisition.objective import LearnedObjective as RsLearned


def acquisition_pair(name, batch, constrained=False, pending=False, prune=False):
    bm, rm, _, _, baseline = outcome_models()
    bp, rp = preference_models(batch)
    bo = BtLearned(bp, sample_shape=torch.Size([3]), seed=194)
    ro = RsLearned(rp, sample_shape=(3,), seed=194)
    kwargs = {}
    if "Noisy" in name:
        kwargs.update(X_baseline=baseline, prune_baseline=prune, cache_root=False)
    elif "Improvement" in name:
        kwargs["best_f"] = 0.2
    if "ConfidenceBound" in name:
        kwargs["beta"] = 0.3
    if constrained:
        kwargs.update(constraints=[lambda y: y[..., 0] - 0.5], eta=0.1)
    if pending:
        kwargs["X_pending"] = torch.tensor([[0.45, 0.55]], dtype=torch.float64)
    bc = getattr(bt_log if "Log" in name else bt_mc, name)
    rc = getattr(rs_log if "Log" in name else rs_mc, name)
    reference_kwargs = dict(kwargs)
    if "Noisy" in name:
        # Official constructor pruning does not insert a preference-member
        # axis into the sampled outcomes; its batched PairwiseGP then fails.
        reference_kwargs["prune_baseline"] = False
    with synced_draws(85):
        ba = bc(bm, sampler=bt_sampler(8), objective=bo, **reference_kwargs)
    ra = rc(rm, sampler=rs_sampler(8, 85, 20), objective=ro, **kwargs)
    return ba, ra


@pytest.mark.parametrize(
    "name",
    [
        "qExpectedImprovement",
        "qLogExpectedImprovement",
        "qProbabilityOfImprovement",
        "qSimpleRegret",
        "qUpperConfidenceBound",
        "qLowerConfidenceBound",
        "qPosteriorStandardDeviation",
        "qNoisyExpectedImprovement",
        "qLogNoisyExpectedImprovement",
    ],
)
@pytest.mark.parametrize(
    "pref_batch,candidate_batch", [((2,), (2,)), ((2,), (3, 2)), ((2, 1), (2, 3))]
)
def test_native_mc_dispatch_matches_official(name, pref_batch, candidate_batch):
    ba, ra = acquisition_pair(name, pref_batch)
    query = torch.rand(
        *candidate_batch,
        2,
        2,
        generator=torch.Generator().manual_seed(206),
        dtype=torch.float64,
        requires_grad=True,
    )
    with synced_draws(85):
        expected = ba(query)
    eg = torch.autograd.grad(expected.sum(), query)[0]
    actual, gradient = ra.value_and_grad(query.detach())
    torch.testing.assert_close(actual, expected, rtol=2e-6, atol=2e-7)
    torch.testing.assert_close(gradient, eg, rtol=3e-5, atol=3e-6)
    if "Noisy" in name:
        obj = torch.zeros(24, *candidate_batch, 2, dtype=torch.float64)
        assert_close(ra.compute_best_f(obj), ba.compute_best_f(obj), rtol=2e-6, atol=2e-7)


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("name", ["qLogExpectedImprovement", "qLogNoisyExpectedImprovement"])
def test_duplicate_candidate_sets_select_different_preference_members(kind, name):
    ba, ra = acquisition_pair(name, (2,), constrained=True, pending=True)
    base = torch.tensor([[0.25, 0.45], [0.65, 0.55]], dtype=torch.float64)
    query = base.expand(2, 2, 2).clone().requires_grad_()
    with synced_draws(85):
        expected = ba(query)
    eg = torch.autograd.grad(expected.sum(), query)[0]
    actual, gradient = ra.value_and_grad(as_kind(query.detach().tolist(), kind))
    assert abs(float(expected[0] - expected[1])) > 1e-3
    assert_close(actual, expected, rtol=2e-6, atol=2e-7)
    assert_close(gradient, eg, rtol=3e-5, atol=3e-6)


def test_native_mc_member_query_graph_folds_explicit_input_expansion():
    ba, ra = acquisition_pair("qLogExpectedImprovement", (2, 1), constrained=True)
    base = torch.tensor([[[[0.25, 0.45], [0.65, 0.55]]]], dtype=torch.float64, requires_grad=True)
    query = base.expand(2, 3, 2, 2)
    actual = ra(query)
    with synced_draws(85):
        expected = ba(query)
    ga = torch.autograd.grad(actual.sum(), base)[0]
    ge = torch.autograd.grad(expected.sum(), base)[0]
    torch.testing.assert_close(actual, expected, rtol=2e-6, atol=2e-7)
    torch.testing.assert_close(ga, ge, rtol=3e-5, atol=3e-6)


def test_default_pruning_uses_union_of_preference_members():
    from botorch_rs.acquisition.utils import prune_inferior_points

    bm, rm, _, _, baseline = outcome_models()
    bp, rp = preference_models((2,))
    bo = BtLearned(bp, sample_shape=torch.Size([3]), seed=194)
    ro = RsLearned(rp, sample_shape=(3,), seed=194)
    with synced_draws(85):
        outcomes = bt_sampler(64)(bm.posterior(baseline))
    # Explicitly preserve the outcome-sample axis when broadcasting members.
    outcomes = outcomes.unsqueeze(1).expand(-1, 2, -1, -1)
    utilities = bo(outcomes)
    feasible = (outcomes[..., 0] <= 0.5).repeat(3, 1, 1)
    winners = utilities.masked_fill(~feasible, -torch.inf).argmax(-1).unique()
    pruned = prune_inferior_points(
        rm,
        baseline,
        objective=ro,
        constraints=[lambda y: y[..., 0] - 0.5],
        sampler=rs_sampler(64, 85, 20),
    )
    torch.testing.assert_close(pruned, baseline[winners])

    _, acquisition = acquisition_pair(
        "qLogNoisyExpectedImprovement", (2,), constrained=True, prune=True
    )
    query = torch.tensor([[[0.25, 0.45], [0.65, 0.55]]], dtype=torch.float64).expand(2, 2, 2)
    value, gradient = acquisition.value_and_grad(query)
    assert torch.isfinite(value).all()
    assert torch.isfinite(gradient).all()
    assert 1 <= len(acquisition.X_baseline) <= 4


@pytest.mark.parametrize("name", ["qLogExpectedImprovement", "qLogNoisyExpectedImprovement"])
def test_batched_member_native_query_gradient_matches_finite_differences(name):
    _, acquisition = acquisition_pair(name, (2,), constrained=True, pending=True)
    query = torch.tensor(
        [[[0.25, 0.45], [0.65, 0.55]], [[0.35, 0.65], [0.8, 0.3]]],
        dtype=torch.float64,
    )
    _, gradient = acquisition.value_and_grad(query)
    for index in range(query.numel()):
        plus, minus = query.clone(), query.clone()
        plus.reshape(-1)[index] += 1e-6
        minus.reshape(-1)[index] -= 1e-6
        finite = (acquisition(plus).sum() - acquisition(minus).sum()) / 2e-6
        torch.testing.assert_close(gradient.reshape(-1)[index], finite, rtol=3e-5, atol=3e-6)


def test_missing_candidate_member_axis_and_native_optimization_are_explicit():
    from botorch_rs.optim import optimize_acqf

    _, acquisition = acquisition_pair("qLogExpectedImprovement", (2,))
    with pytest.raises(ValueError, match="candidate batch"):
        acquisition([[0.25, 0.45], [0.65, 0.55]])
    with pytest.raises(NotImplementedError, match="member dispatch"):
        optimize_acqf(
            acquisition,
            torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.float64),
            q=1,
            num_restarts=1,
            raw_samples=4,
        )


def test_python_callback_context_restores_after_nested_failure():
    assert _torch_bridge.objective_context() is None

    def inner(y, X=None):
        assert _torch_bridge.objective_context() == (([5], 2),)
        raise ValueError("nested failure")

    def outer(y, X=None):
        assert _torch_bridge.objective_context() == (([2, 3], 4),)
        with pytest.raises(ValueError, match="nested failure"):
            _torch_bridge.forward(inner, "objective", [0.2], (1, 1, 1), None, False, ([5], 2))
        assert _torch_bridge.objective_context() == (([2, 3], 4),)
        return y[..., 0]

    _torch_bridge.forward(outer, "objective", [0.2], (1, 1, 1), None, False, ([2, 3], 4))
    assert _torch_bridge.objective_context() is None

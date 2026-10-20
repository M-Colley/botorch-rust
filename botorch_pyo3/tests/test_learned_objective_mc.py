"""Preference samples inside native MC reductions and their query adjoints."""

# ruff: noqa: E402

from __future__ import annotations

import pytest

from helpers import KINDS, as_kind, assert_close, importorskip_botorch

importorskip_botorch()
torch = pytest.importorskip("torch")

from botorch.acquisition import logei as bt_log
from botorch.acquisition import monte_carlo as bt_mc
from botorch.acquisition.objective import LearnedObjective as BtLearned
from botorch.models import SingleTaskGP as BtGP
from botorch.models.deterministic import AffineDeterministicModel as BtAffine
from mo_acquisition_helpers import bt_sampler, rs_sampler, synced_draws
from preference_helpers import Case, build_pair, make_data

from botorch_rs.acquisition import logei as rs_log
from botorch_rs.acquisition import monte_carlo as rs_mc
from botorch_rs.acquisition.objective import LearnedObjective as RsLearned
from botorch_rs.models import SingleTaskGP as RsGP
from botorch_rs.models.deterministic import AffineDeterministicModel as RsAffine


def models(deterministic=False):
    x = torch.tensor([[0.1, 0.2], [0.3, 0.8], [0.7, 0.4], [0.9, 0.7]], dtype=torch.float64)
    y = torch.stack((x[:, 0].sin(), x[:, 1].square()), -1)
    noise = torch.full_like(y, 0.03)
    bmodel, rmodel = BtGP(x, y, noise), RsGP(x, y, noise)
    if deterministic:
        a = torch.tensor([[0.7], [-0.4]], dtype=torch.float64)
        bpref, rpref = BtAffine(a), RsAffine(a)
    else:
        case = Case(n=7, d=2, m=8)
        px, pc = make_data(case, 82)
        bpref, rpref = build_pair(case, px, pc, seed=83)
    return bmodel, rmodel, bpref, rpref, x


def acquisitions(name, *, deterministic=False, constrained=False, cache=False, pending=False):
    bm, rm, bp, rp, baseline = models(deterministic)
    bo = BtLearned(bp, sample_shape=None if deterministic else torch.Size([3]), seed=84)
    ro = RsLearned(rp, sample_shape=None if deterministic else (3,), seed=84)
    kwargs = {}
    if "Noisy" in name:
        kwargs.update(X_baseline=baseline, prune_baseline=False, cache_root=cache)
    elif "Improvement" in name or "ProbabilityOfImprovement" in name:
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
    if "Noisy" in name and not deterministic:
        # The official cached path loses the preference GP joint law.
        reference_kwargs["cache_root"] = False
    with synced_draws(85):
        ba = bc(bm, sampler=bt_sampler(8), objective=bo, **reference_kwargs)
    ra = rc(rm, sampler=rs_sampler(8, 85, 20), objective=ro, **kwargs)
    return ba, ra


NAMES = [
    "qExpectedImprovement",
    "qLogExpectedImprovement",
    "qProbabilityOfImprovement",
    "qSimpleRegret",
    "qUpperConfidenceBound",
    "qLowerConfidenceBound",
    "qPosteriorStandardDeviation",
]


@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize("deterministic", [False, True])
def test_native_learned_mc_values_and_adjoints_match_official(name, deterministic):
    ba, ra = acquisitions(name, deterministic=deterministic)
    query = torch.tensor(
        [[[0.25, 0.45], [0.65, 0.55]], [[0.35, 0.65], [0.8, 0.3]]],
        dtype=torch.float64,
        requires_grad=True,
    )
    with synced_draws(85):
        expected = ba(query)
    eg = torch.autograd.grad(expected.sum(), query)[0]
    actual, ag = ra.value_and_grad(query.detach())
    torch.testing.assert_close(actual, expected, rtol=1e-7, atol=2e-8)
    torch.testing.assert_close(ag, eg, rtol=2e-6, atol=2e-7)


@pytest.mark.parametrize("name", ["qExpectedImprovement", "qLogExpectedImprovement"])
def test_expanded_preference_samples_repeat_constraints_and_pending_points(name):
    ba, ra = acquisitions(name, constrained=True, pending=True)
    x = torch.tensor([[[0.25, 0.45], [0.65, 0.55]]], dtype=torch.float64, requires_grad=True)
    with synced_draws(85):
        expected = ba(x)
    eg = torch.autograd.grad(expected.sum(), x)[0]
    actual, gradient = ra.value_and_grad(x.detach())
    torch.testing.assert_close(actual, expected, rtol=1e-7, atol=2e-8)
    torch.testing.assert_close(gradient, eg, rtol=2e-6, atol=2e-7)


@pytest.mark.parametrize("deterministic", [False, True])
@pytest.mark.parametrize("cache", [False, True])
@pytest.mark.parametrize("constrained", [False, True])
@pytest.mark.parametrize("name", ["qNoisyExpectedImprovement", "qLogNoisyExpectedImprovement"])
def test_noisy_learned_mc_joint_and_cached_roots(name, constrained, cache, deterministic):
    ba, ra = acquisitions(name, constrained=constrained, cache=cache, deterministic=deterministic)
    assert ra._cache_root == (cache and deterministic)
    x = torch.tensor([[[0.25, 0.45], [0.65, 0.55]]], dtype=torch.float64, requires_grad=True)
    with synced_draws(85):
        expected = ba(x)
    eg = torch.autograd.grad(expected.sum(), x)[0]
    actual, gradient = ra.value_and_grad(x.detach())
    torch.testing.assert_close(actual, expected, rtol=2e-6, atol=2e-7)
    torch.testing.assert_close(gradient, eg, rtol=3e-5, atol=3e-6)
    obj = torch.zeros(8 if deterministic else 24, 1, 2, dtype=torch.float64)
    assert_close(ra.compute_best_f(obj), ba.compute_best_f(obj), rtol=1e-7, atol=2e-8)


@pytest.mark.parametrize("kind", KINDS)
def test_native_learned_mc_query_gradient_matches_finite_differences(kind):
    _, acquisition = acquisitions("qLogExpectedImprovement", constrained=True)
    x = [[[0.25, 0.45], [0.65, 0.55]]]
    _, actual = acquisition.value_and_grad(as_kind(x, kind))
    finite = torch.zeros(1, 2, 2, dtype=torch.float64)
    for p in range(2):
        for c in range(2):
            xp = torch.tensor(x, dtype=torch.float64)
            xm = xp.clone()
            xp[0, p, c] += 1e-6
            xm[0, p, c] -= 1e-6
            finite[0, p, c] = (
                float(
                    torch.as_tensor(
                        acquisition(as_kind(xp.tolist(), kind)), dtype=torch.float64
                    ).sum()
                )
                - float(
                    torch.as_tensor(
                        acquisition(as_kind(xm.tolist(), kind)), dtype=torch.float64
                    ).sum()
                )
            ) / 2e-6
    assert_close(actual, finite, rtol=2e-5, atol=2e-6)


def test_deterministic_learned_objective_preserves_standalone_query_graph():
    _, _, bp, rp, _ = models(True)
    x = torch.rand(3, 2, 2, dtype=torch.float64, requires_grad=True)
    actual, expected = RsLearned(rp)(x), BtLearned(bp)(x)
    ga = torch.autograd.grad(actual.square().sum(), x)[0]
    ge = torch.autograd.grad(expected.square().sum(), x)[0]
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(ga, ge)


def test_fully_bayesian_outcome_members_share_preference_draws():
    from test_parity_10a_fully_bayesian import _fb_list_pair

    bm, rm, _ = _fb_list_pair(91)
    _, _, bp, rp, _ = models()
    ba = bt_log.qLogExpectedImprovement(
        bm,
        best_f=0.2,
        objective=BtLearned(bp, sample_shape=torch.Size([3]), seed=84),
        sampler=bt_sampler(8),
    )
    ra = rs_log.qLogExpectedImprovement(
        rm,
        best_f=0.2,
        objective=RsLearned(rp, sample_shape=(3,), seed=84),
        sampler=rs_sampler(8, 85, 20),
    )
    x = torch.tensor(
        [[[0.25, 0.45, 0.35], [0.65, 0.55, 0.7]]], dtype=torch.float64, requires_grad=True
    )
    with synced_draws(85):
        expected = ba(x)
    eg = torch.autograd.grad(expected.sum(), x)[0]
    actual, gradient = ra.value_and_grad(x.detach())
    torch.testing.assert_close(actual, expected, rtol=2e-6, atol=2e-7)
    torch.testing.assert_close(gradient, eg, rtol=3e-5, atol=3e-6)


def test_stochastic_preference_cache_preserves_joint_law_at_a_baseline_point():
    bm, rm, _, _, baseline = models()
    px = torch.rand(7, 2, generator=torch.Generator().manual_seed(82), dtype=torch.float64)
    comps = torch.tensor([[0, 1], [2, 3], [4, 5], [5, 6], [0, 4], [3, 6]])
    bp, rp = build_pair(Case(n=7, d=2, m=6), px, comps, seed=83)
    references = []
    for cache in (False, True):
        with synced_draws(85):
            ba = bt_mc.qNoisyExpectedImprovement(
                bm,
                baseline[:2],
                objective=BtLearned(bp, sample_shape=torch.Size([32]), seed=84),
                sampler=bt_sampler(64),
                cache_root=cache,
                prune_baseline=False,
            )
            references.append(float(ba(baseline[:1]).detach()))
    with pytest.warns(RuntimeWarning, match="joint baseline"):
        ra = rs_mc.qNoisyExpectedImprovement(
            rm,
            baseline[:2],
            objective=RsLearned(rp, sample_shape=(32,), seed=84),
            sampler=rs_sampler(64, 85, 20),
            cache_root=True,
            prune_baseline=False,
        )
    actual = float(ra(baseline[:1]))
    # A repeated baseline point cannot materially improve on that baseline.
    # The small nonzero full-joint values are covariance-root jitter.
    assert references[0] < 0.002
    assert references[1] > 0.05
    assert actual < 0.002
    assert abs(actual - references[0]) < 0.0001


def test_legacy_constrained_learned_objective_repeats_feasibility_samples():
    from botorch.acquisition.objective import GenericMCObjective as BtGeneric

    from botorch_rs.acquisition.objective import ConstrainedMCObjective as RsConstrained

    bm, rm, bp, rp, _ = models()
    bo = BtLearned(bp, sample_shape=torch.Size([3]), seed=84)
    ro = RsLearned(rp, sample_shape=(3,), seed=84)

    def reference_objective(y, X=None):
        utility = bo(y)
        probability = torch.sigmoid(-(y[..., 0] - 0.5) / 0.1).repeat(3, 1, 1)
        return (utility + 1).clamp_min(0) * probability - 1

    ba = bt_mc.qSimpleRegret(bm, sampler=bt_sampler(8), objective=BtGeneric(reference_objective))
    ra = rs_mc.qSimpleRegret(
        rm,
        sampler=rs_sampler(8, 85, 20),
        objective=RsConstrained(ro, [lambda y: y[..., 0] - 0.5], infeasible_cost=1, eta=0.1),
    )
    x = torch.tensor([[[0.25, 0.45], [0.65, 0.55]]], dtype=torch.float64, requires_grad=True)
    with synced_draws(85):
        expected = ba(x)
    eg = torch.autograd.grad(expected.sum(), x)[0]
    actual, gradient = ra.value_and_grad(x.detach())
    torch.testing.assert_close(actual, expected, rtol=2e-6, atol=2e-7)
    torch.testing.assert_close(gradient, eg, rtol=3e-5, atol=3e-6)


def test_multidimensional_preference_draws_are_flattened_before_native_reduction():
    from botorch.acquisition.objective import GenericMCObjective as BtGeneric

    bm, rm, bp, rp, _ = models()
    bo = BtLearned(bp, sample_shape=torch.Size([2, 3]), seed=84)
    # Correct the official standalone layout to the documented P*S axis.
    ba = bt_mc.qSimpleRegret(
        bm,
        sampler=bt_sampler(8),
        objective=BtGeneric(lambda y, X=None: bo(y).reshape(-1, *y.shape[1:-1])),
    )
    ra = rs_mc.qSimpleRegret(
        rm,
        sampler=rs_sampler(8, 85, 20),
        objective=RsLearned(rp, sample_shape=(2, 3), seed=84),
    )
    x = torch.tensor([[[0.25, 0.45], [0.65, 0.55]]], dtype=torch.float64, requires_grad=True)
    with synced_draws(85):
        expected = ba(x)
    eg = torch.autograd.grad(expected.sum(), x)[0]
    actual, gradient = ra.value_and_grad(x.detach())
    torch.testing.assert_close(actual, expected, rtol=2e-6, atol=2e-7)
    torch.testing.assert_close(gradient, eg, rtol=3e-5, atol=3e-6)


def test_default_noisy_pruning_handles_augmented_objective_samples():
    _, rm, _, rp, baseline = models()
    acquisition = rs_log.qLogNoisyExpectedImprovement(
        rm,
        baseline,
        objective=RsLearned(rp, sample_shape=(3,), seed=84),
        sampler=rs_sampler(8, 85, 20),
        constraints=[lambda y: y[..., 0] - 0.5],
        eta=0.1,
    )
    x = torch.tensor([[[0.25, 0.45], [0.65, 0.55]]], dtype=torch.float64)
    value, gradient = acquisition.value_and_grad(x)
    assert not acquisition._cache_root
    assert torch.isfinite(value).all()
    assert torch.isfinite(gradient).all()
    assert 1 <= len(acquisition.X_baseline) <= len(baseline)


def test_explicit_generic_deterministic_preference_needs_no_native_dim_hint():
    from botorch.models.deterministic import GenericDeterministicModel as BtGenericModel

    from botorch_rs.models.deterministic import GenericDeterministicModel as RsGenericModel

    bm, rm, _, _, _ = models()

    def utility(y):
        return y.square().sum(dim=-1, keepdim=True)

    ba = bt_mc.qSimpleRegret(
        bm, sampler=bt_sampler(8), objective=BtLearned(BtGenericModel(utility))
    )
    ra = rs_mc.qSimpleRegret(
        rm, sampler=rs_sampler(8, 85, 20), objective=RsLearned(RsGenericModel(utility))
    )
    x = torch.tensor([[[0.25, 0.45], [0.65, 0.55]]], dtype=torch.float64, requires_grad=True)
    with synced_draws(85):
        expected = ba(x)
    eg = torch.autograd.grad(expected.sum(), x)[0]
    actual, gradient = ra.value_and_grad(x.detach())
    torch.testing.assert_close(actual, expected, rtol=2e-6, atol=2e-7)
    torch.testing.assert_close(gradient, eg, rtol=3e-5, atol=3e-6)

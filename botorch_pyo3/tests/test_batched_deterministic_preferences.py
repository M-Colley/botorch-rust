"""Native member views for deterministic posterior-mean preferences."""

# ruff: noqa: E402

from __future__ import annotations

import math

import pytest

from helpers import KINDS, as_kind, assert_close, importorskip_botorch

importorskip_botorch()
torch = pytest.importorskip("torch")

from botorch.acquisition import logei as bt_log
from botorch.acquisition import monte_carlo as bt_mc
from botorch.acquisition.objective import LearnedObjective as BtLearned
from botorch.models import SingleTaskGP as BtGP
from botorch.models.deterministic import PosteriorMeanModel as BtMean
from botorch.models.transforms.input import Normalize as BtNormalize
from mo_acquisition_helpers import bt_sampler, rs_sampler, synced_draws
from test_batched_learned_objective import models as preference_models
from test_learned_objective_mc import models as outcome_models

from botorch_rs.acquisition import logei as rs_log
from botorch_rs.acquisition import monte_carlo as rs_mc
from botorch_rs.acquisition.objective import LearnedObjective as RsLearned
from botorch_rs.models import SingleTaskGP as RsGP
from botorch_rs.models.deterministic import PosteriorMeanModel as RsMean
from botorch_rs.models.transforms.input import Normalize as RsNormalize


def deterministic_preferences(batch, family, nested=False):
    if family == "pairwise":
        bp, rp = preference_models(batch)
    else:
        x = torch.rand(
            *batch, 6, 2, generator=torch.Generator().manual_seed(211), dtype=torch.float64
        )
        y = (2 * x[..., :1]).sin() - x[..., 1:].square()
        noise = torch.full_like(y, 0.03)
        upper = (
            (1.0 + 0.3 * torch.arange(math.prod(batch), dtype=torch.float64))
            .reshape(*batch, 1)
            .expand(*batch, 2)
        )
        bounds = torch.stack((torch.zeros_like(upper), upper), dim=-2)
        bp = BtGP(x, y, noise, input_transform=BtNormalize(2, bounds=bounds, batch_shape=batch))
        rp = RsGP(x, y, noise, input_transform=RsNormalize(2, bounds=bounds, batch_shape=batch))
    bp, rp = BtMean(bp), RsMean(rp)
    if nested:
        bp, rp = BtMean(bp), RsMean(rp)
    return BtLearned(bp), RsLearned(rp)


@pytest.mark.parametrize("family", ["pairwise", "regression"])
@pytest.mark.parametrize("batch", [(2,), (2, 1)])
@pytest.mark.parametrize("kind", KINDS)
def test_native_mean_member_samples_and_query_graph_match_official(family, batch, kind):
    bo, ro = deterministic_preferences(batch, family, nested=True)
    query = torch.rand(
        3,
        *batch,
        2,
        2,
        generator=torch.Generator().manual_seed(212),
        dtype=torch.float64,
        requires_grad=True,
    )
    expected = bo(query)
    actual = ro(as_kind(query.detach().tolist(), kind))
    assert_close(actual, expected, rtol=2e-7, atol=2e-8)
    actual_graph = ro(query)
    eg = torch.autograd.grad(expected.sum(), query)[0]
    ag = torch.autograd.grad(actual_graph.sum(), query)[0]
    torch.testing.assert_close(ag, eg, rtol=2e-6, atol=2e-7)
    assert ro.sampler is None


NAMES = [
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


@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize("family", ["pairwise", "regression"])
@pytest.mark.parametrize("batch", [(2,), (2, 1)])
def test_native_mc_with_batched_posterior_mean_preferences(name, family, batch):
    bm, rm, _, _, baseline = outcome_models()
    bo, ro = deterministic_preferences(batch, family)
    kwargs = {}
    if "Noisy" in name:
        kwargs.update(X_baseline=baseline, prune_baseline=False, cache_root=False)
    elif "Improvement" in name:
        kwargs["best_f"] = 0.2
    if "ConfidenceBound" in name:
        kwargs["beta"] = 0.3
    ba = getattr(bt_log if "Log" in name else bt_mc, name)(
        bm, sampler=bt_sampler(8), objective=bo, **kwargs
    )
    ra = getattr(rs_log if "Log" in name else rs_mc, name)(
        rm, sampler=rs_sampler(8, 85, 20), objective=ro, **kwargs
    )
    candidate_batch = (3, 2) if batch == (2,) else (2, 3)
    query = torch.rand(
        *candidate_batch,
        2,
        2,
        generator=torch.Generator().manual_seed(213),
        dtype=torch.float64,
        requires_grad=True,
    )
    with synced_draws(85):
        expected = ba(query)
    eg = torch.autograd.grad(expected.sum(), query)[0]
    actual, ag = ra.value_and_grad(query.detach())
    torch.testing.assert_close(actual, expected, rtol=2e-6, atol=2e-7)
    torch.testing.assert_close(ag, eg, rtol=3e-5, atol=3e-6)


@pytest.mark.parametrize("family", ["pairwise", "regression"])
def test_deterministic_member_gradients_and_full_joint_cache_policy(family):
    _, model, _, _, baseline = outcome_models()
    _, objective = deterministic_preferences((2,), family)
    with pytest.warns(RuntimeWarning, match="stochastic or batched"):
        acquisition = rs_log.qLogNoisyExpectedImprovement(
            model,
            X_baseline=baseline,
            sampler=rs_sampler(8, 85, 20),
            objective=objective,
            cache_root=True,
            prune_baseline=False,
        )
    assert not acquisition._cache_root
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

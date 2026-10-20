"""Shared sampling and pending-state APIs remain connected to native state."""

from __future__ import annotations

import pytest

from botorch_rs.acquisition.active_learning import PairwiseMCPosteriorVariance
from botorch_rs.acquisition.max_value_entropy_search import (
    qMultiFidelityLowerBoundMaxValueEntropy,
    qMultiFidelityMaxValueEntropy,
)
from botorch_rs.acquisition.multi_objective.hypervolume_knowledge_gradient import (
    qHypervolumeKnowledgeGradient,
    qMultiFidelityHypervolumeKnowledgeGradient,
)
from botorch_rs.acquisition.multi_objective.logei import qLogNoisyExpectedHypervolumeImprovement
from botorch_rs.acquisition.multi_objective.monte_carlo import qNoisyExpectedHypervolumeImprovement
from botorch_rs.acquisition.objective import IdentityMCObjective
from botorch_rs.acquisition.preference import PairwiseBayesianActiveLearningByDisagreement
from botorch_rs.models import ModelListGP, SingleTaskGP
from botorch_rs.posteriors.posterior_list import PosteriorList
from botorch_rs.sampling.normal import IIDNormalSampler
from botorch_rs.utils.multi_objective.hypervolume import NoisyExpectedHypervolumeMixin
from helpers import KINDS, as_kind, assert_close, shape, to_list


def models(kind):
    x = as_kind([[0.1], [0.4], [0.8]], kind)
    y = as_kind([[0.2], [0.7], [0.5]], kind)
    a = SingleTaskGP(x, y, outcome_transform=None)
    b = SingleTaskGP(x, as_kind([[0.8], [0.3], [0.6]], kind), outcome_transform=None)
    return x, a, b


@pytest.mark.parametrize("kind", KINDS)
def test_pairwise_acquisitions_share_configured_and_lazy_sampling(kind):
    _, model, _ = models(kind)
    posterior = model.posterior(as_kind([[0.3], [0.6]], kind))
    acquisition = PairwiseMCPosteriorVariance(
        model, IdentityMCObjective(), IIDNormalSampler((4,), seed=7)
    )
    assert tuple(acquisition.sample_shape) == (4,)
    samples = acquisition.get_posterior_samples(posterior)
    assert shape(samples) == (4, 2, 1)
    assert_close(samples, acquisition.sampler(posterior))
    bald = PairwiseBayesianActiveLearningByDisagreement(model, num_samples=4)
    assert tuple(bald.sample_shape) == (512,)
    assert bald.sampler is None
    samples = bald.get_posterior_samples(posterior)
    assert bald.sampler is not None
    assert shape(samples) == (512, 2, 1)
    assert_close(samples, bald.get_posterior_samples(posterior))


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize(
    "cls", [qHypervolumeKnowledgeGradient, qMultiFidelityHypervolumeKnowledgeGradient]
)
def test_hypervolume_knowledge_gradient_list_sampling_protocol(kind, cls):
    _, a, b = models(kind)
    kwargs = (
        {"target_fidelities": {0: 1.0}} if cls is qMultiFidelityHypervolumeKnowledgeGradient else {}
    )
    acquisition = cls(
        ModelListGP(a, b), as_kind([-1.0, -1.0], kind), num_fantasies=2, num_pareto=1, **kwargs
    )
    posterior = PosteriorList(*(model.posterior(as_kind([[0.3], [0.6]], kind)) for model in (a, b)))
    assert tuple(acquisition.sample_shape) == (2,)
    samples = acquisition.get_posterior_samples(posterior)
    assert shape(samples) == (2, 2, 2)
    assert_close(samples, acquisition.sampler(posterior))


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize(
    "cls", [qMultiFidelityMaxValueEntropy, qMultiFidelityLowerBoundMaxValueEntropy]
)
def test_mes_cost_sampler_is_cached_independent_copy(kind, cls):
    x, model, _ = models(kind)
    acquisition = cls(model, x, num_fantasies=2, num_mv_samples=2, num_y_samples=4, seed=8)
    posterior = model.posterior(as_kind([[0.3]], kind))
    acquisition.fantasies_sampler(posterior)
    original = to_list(acquisition.fantasies_sampler.base_samples)
    sampler = acquisition.cost_sampler
    assert sampler is acquisition.cost_sampler
    assert sampler is not acquisition.fantasies_sampler
    assert sampler.sample_shape == acquisition.fantasies_sampler.sample_shape
    assert_close(sampler.base_samples, original)
    sampler(model.posterior(x))
    assert_close(acquisition.fantasies_sampler.base_samples, original)
    assert shape(sampler.base_samples) != shape(acquisition.fantasies_sampler.base_samples)


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize(
    "cls", [qNoisyExpectedHypervolumeImprovement, qLogNoisyExpectedHypervolumeImprovement]
)
def test_nehvi_mixin_methods_sample_and_update_native_pending_cache(kind, cls):
    x, a, b = models(kind)
    model = ModelListGP(a, b)
    acquisition = cls(
        model,
        as_kind([-1.0, -1.0], kind),
        x,
        sampler=IIDNormalSampler((8,), seed=11),
        prune_baseline=False,
        max_iep=0,
    )
    query = as_kind([[0.3]], kind)
    assert tuple(acquisition.sample_shape) == (8,)
    posterior = model.posterior(query)
    assert shape(NoisyExpectedHypervolumeMixin.get_posterior_samples(acquisition, posterior)) == (
        8,
        1,
        2,
    )
    NoisyExpectedHypervolumeMixin.set_X_pending(acquisition, as_kind([[0.6]], kind))
    assert shape(acquisition.X_baseline)[-2] == 4
    assert shape(acquisition(query)) == (1,)
    NoisyExpectedHypervolumeMixin.set_X_pending(acquisition, None)
    assert shape(acquisition.X_baseline)[-2] == 3
    assert shape(acquisition(query)) == (1,)

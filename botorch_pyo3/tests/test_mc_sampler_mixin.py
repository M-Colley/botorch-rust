"""Public MC sampling APIs and posterior-driven lazy sampler selection."""

from __future__ import annotations

from typing import Any

import pytest

from botorch_rs.acquisition import MCSamplerMixin, qExpectedImprovement, qLogExpectedImprovement
from botorch_rs.acquisition.acquisition import AcquisitionFunction
from botorch_rs.acquisition.bayesian_active_learning import qBayesianActiveLearningByDisagreement
from botorch_rs.acquisition.joint_entropy_search import qJointEntropySearch
from botorch_rs.acquisition.knowledge_gradient import qKnowledgeGradient
from botorch_rs.acquisition.max_value_entropy_search import qMaxValueEntropy
from botorch_rs.acquisition.monte_carlo import MCAcquisitionFunction
from botorch_rs.acquisition.multi_objective.base import MultiObjectiveMCAcquisitionFunction
from botorch_rs.acquisition.multi_objective.joint_entropy_search import (
    LowerBoundMultiObjectiveEntropySearch,
)
from botorch_rs.acquisition.multi_step_lookahead import qMultiStepLookahead
from botorch_rs.models import SingleTaskGP
from botorch_rs.models.transforms.outcome import Log
from botorch_rs.posteriors.ensemble import EnsemblePosterior
from botorch_rs.posteriors.posterior_list import PosteriorList
from botorch_rs.posteriors.transformed import TransformedPosterior
from botorch_rs.sampling import IIDNormalSampler, ListSampler, MCSampler, SobolQMCNormalSampler
from botorch_rs.sampling.index_sampler import IndexSampler
from helpers import as_kind, assert_close, check_kind, flatten, shape


class _SmallMixin(MCSamplerMixin):
    _default_sample_shape = (4,)


def _posterior(kind: str) -> Any:
    X = as_kind([[0.0], [0.3], [0.8], [1.0]], kind)
    Y = as_kind([[0.0], [1.0], [-0.2], [0.1]], kind)
    return SingleTaskGP(X, Y).posterior(as_kind([[0.2], [0.6]], kind))


def test_mixin_is_lazy_and_reuses_native_sampler(kind: str) -> None:
    mixin = _SmallMixin()
    assert mixin.sampler is None and mixin.sample_shape == (4,)
    posterior = _posterior(kind)
    first = mixin.get_posterior_samples(posterior)
    sampler = mixin.sampler
    assert isinstance(sampler, SobolQMCNormalSampler)
    check_kind(first, kind)
    assert shape(first) == (4, 2, 1)
    assert_close(first, mixin.get_posterior_samples(posterior), atol=0.0, rtol=0.0)
    assert mixin.sampler is sampler
    assert MCSamplerMixin().sample_shape == (512,)


def test_explicit_sampler_and_multidimensional_sample_shape(kind: str) -> None:
    sampler = SobolQMCNormalSampler((2, 3), seed=5)
    mixin = MCSamplerMixin(sampler=sampler)
    posterior = _posterior(kind)
    assert mixin.sample_shape == (2, 3)
    samples = mixin.get_posterior_samples(posterior)
    assert shape(samples) == (2, 3, 2, 1)
    assert_close(samples, sampler(posterior), atol=0.0, rtol=0.0)


def test_lazy_ensemble_and_list_sampler_selection(kind: str) -> None:
    ensemble = EnsemblePosterior(as_kind([[[1.0], [2.0]], [[3.0], [4.0]]], kind))
    mixin = _SmallMixin()
    samples = mixin.get_posterior_samples(ensemble)
    assert isinstance(mixin.sampler, IndexSampler)
    check_kind(samples, kind)
    assert shape(samples) == (4, 2, 1)
    assert all(v in (1.0, 2.0, 3.0, 4.0) for v in flatten(samples))
    mixture = PosteriorList(_posterior(kind), _posterior(kind))
    mixin = _SmallMixin()
    samples = mixin.get_posterior_samples(mixture)
    assert isinstance(mixin.sampler, ListSampler)
    assert all(isinstance(s, IIDNormalSampler) for s in mixin.sampler.samplers)
    check_kind(samples, kind)
    assert shape(samples) == (4, 2, 2)


def test_list_sampler_handles_index_and_normal_components(kind: str) -> None:
    ensemble = EnsemblePosterior(as_kind([[[1.0], [2.0]], [[3.0], [4.0]]], kind))
    normal = _posterior(kind)
    index = IndexSampler((4,), seed=2)
    gaussian = SobolQMCNormalSampler((4,), seed=2)
    sampler = ListSampler(index, gaussian)
    actual = sampler(PosteriorList(ensemble, normal))
    a, b = flatten(index(ensemble)), flatten(gaussian(normal))
    assert flatten(actual) == pytest.approx([v for pair in zip(a, b, strict=True) for v in pair])
    with pytest.raises(ValueError, match="one sampler per"):
        sampler(PosteriorList(normal))


def test_transformed_and_native_nonlinear_posteriors(kind: str) -> None:
    base = _posterior(kind)
    generic = TransformedPosterior(base, sample_transform=lambda values: values)
    sampler = SobolQMCNormalSampler((4,), seed=4)
    mixin = MCSamplerMixin(sampler)
    assert_close(mixin.get_posterior_samples(generic), sampler(base), atol=0.0, rtol=0.0)
    X = as_kind([[0.0], [0.3], [0.8], [1.0]], kind)
    Y = as_kind([[1.0], [2.0], [0.5], [0.8]], kind)
    posterior = SingleTaskGP(X, Y, outcome_transform=Log()).posterior(as_kind([[0.2], [0.6]], kind))
    samples = mixin.get_posterior_samples(posterior)
    check_kind(samples, kind)
    assert shape(samples) == (4, 2, 1)
    assert all(v > 0 for v in flatten(samples))


def test_mc_base_samples_and_objectives(kind: str) -> None:
    X = as_kind([[0.0], [0.3], [0.8], [1.0]], kind)
    Y = as_kind([[0.0], [1.0], [-0.2], [0.1]], kind)
    sampler = SobolQMCNormalSampler((4,), seed=3)
    acqf = MCAcquisitionFunction(SingleTaskGP(X, Y), sampler=sampler)
    Xt = as_kind([[0.2], [0.6]], kind)
    samples, objectives = acqf._get_samples_and_objectives(Xt)
    check_kind(samples, kind)
    check_kind(objectives, kind)
    assert shape(samples) == (4, 2, 1)
    assert shape(objectives) == (4, 2)
    assert flatten(samples) == flatten(objectives)


@pytest.mark.parametrize(
    "cls",
    [
        MCAcquisitionFunction,
        MultiObjectiveMCAcquisitionFunction,
        qExpectedImprovement,
        qLogExpectedImprovement,
        qKnowledgeGradient,
        qMultiStepLookahead,
        qBayesianActiveLearningByDisagreement,
        qJointEntropySearch,
        qMaxValueEntropy,
        LowerBoundMultiObjectiveEntropySearch,
    ],
)
def test_sampling_api_inherited_by_mc_acquisition_families(cls: type) -> None:
    assert issubclass(cls, MCSamplerMixin)
    assert callable(cls.get_posterior_samples)
    assert isinstance(cls.sample_shape, property)
    assert not issubclass(AcquisitionFunction, MCSamplerMixin)


def test_custom_sampler_is_respected() -> None:
    class CustomSampler(MCSampler):
        sample_shape = (7,)

        def __call__(self, *, posterior: Any) -> Any:
            return posterior

    sampler = CustomSampler()
    marker = object()
    mixin = MCSamplerMixin(sampler)
    assert mixin.sample_shape == (7,)
    assert mixin.get_posterior_samples(marker) is marker
    assert mixin.sampler is sampler


def test_mc_base_respects_subclass_default_sample_shape() -> None:
    class SmallMC(MCAcquisitionFunction):
        _default_sample_shape = (2, 3)

    acqf = SmallMC(SingleTaskGP([[0.0], [1.0]], [[0.0], [1.0]]))
    assert acqf.sample_shape == (2, 3)
    assert shape(acqf.get_posterior_samples(acqf.model.posterior([[0.5]]))) == (2, 3, 1, 1)

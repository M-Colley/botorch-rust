"""Native scalarized mixture information gain uses paired ModelList components."""

# ruff: noqa: E402
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("botorch")
pytest.importorskip("jax")
pytest.importorskip("numpyro")

from botorch.acquisition.bayesian_active_learning import (
    qBayesianActiveLearningByDisagreement as ReferenceBALD,
)
from botorch.acquisition.objective import ScalarizedPosteriorTransform as ReferenceScalarize
from botorch.models import ModelListGP as ReferenceModelList
from botorch.sampling import normal as reference_samplers
from fully_bayesian_helpers import model_pair

from botorch_rs.acquisition.bayesian_active_learning import qBayesianActiveLearningByDisagreement
from botorch_rs.acquisition.objective import ScalarizedPosteriorTransform
from botorch_rs.exceptions.errors import UnsupportedError
from botorch_rs.models import ModelListGP
from botorch_rs.sampling import normal as port_samplers


def _models(family="matern", singleton=False):
    pairs = [
        model_pair(family, n=6, num=count, seed=140 + i, fixed=True, transforms=True)
        for i, count in enumerate((1 if singleton else 3, 3))
    ]
    models = ModelListGP(*(p[1] for p in pairs)), ReferenceModelList(*(p[0] for p in pairs))
    x = pairs[0][2]
    low, high = x.min(0).values, x.max(0).values
    query = low + (high - low) * x.new_tensor(
        [[[0.2, 0.3, 0.5], [0.7, 0.4, 0.6]], [[0.3, 0.6, 0.8], [0.5, 0.4, 0.7]]]
    )
    pending = low + (high - low) * x.new_tensor([[0.4, 0.7, 0.3]])
    return models, query, pending


@pytest.mark.parametrize("family", ["matern", "linear"])
@pytest.mark.parametrize("singleton", [False, True])
@pytest.mark.parametrize("pending", [False, True])
@pytest.mark.parametrize("sampler_name", ["IIDNormalSampler", "SobolQMCNormalSampler"])
def test_native_scalarized_model_list_bald_matches_reference(
    family, singleton, pending, sampler_name
):
    models, query, pending_x = _models(family, singleton)
    weights = query.new_tensor([0.3, 0.7])
    normal = torch.linspace(-1.4, 1.7, 16 * (2 + int(pending)), dtype=query.dtype).reshape(
        16, 1, 1, 2 + int(pending)
    )
    results = []
    for model, cls, transform, samplers in zip(
        models,
        (qBayesianActiveLearningByDisagreement, ReferenceBALD),
        (ScalarizedPosteriorTransform, ReferenceScalarize),
        (port_samplers, reference_samplers),
        strict=True,
    ):
        sampler = getattr(samplers, sampler_name)(torch.Size([16]), seed=29)
        sampler.base_samples = normal.clone()
        acq = cls(
            model,
            sampler=sampler,
            posterior_transform=transform(weights),
            X_pending=pending_x if pending else None,
        )
        variable = query.clone().requires_grad_()
        value = acq(variable)
        gradient = torch.autograd.grad(value.sum(), variable)[0]
        if cls is qBayesianActiveLearningByDisagreement:
            actual, actual_gradient = acq.value_and_grad(query)
            torch.testing.assert_close(actual, value, atol=0, rtol=0)
            torch.testing.assert_close(actual_gradient, gradient, atol=2e-13, rtol=2e-12)
        results.append((value, gradient))
    for actual, expected in zip(*results, strict=True):
        torch.testing.assert_close(actual, expected, atol=3e-10, rtol=3e-8)


def test_native_bald_iid_seed_cache_updates_and_array_query_parity():
    models, query, _ = _models()
    weights = query.new_tensor([0.3, 0.7])
    sampler = port_samplers.IIDNormalSampler((16,), seed=29)
    ref_sampler = reference_samplers.IIDNormalSampler(torch.Size([16]), seed=29)
    acq = qBayesianActiveLearningByDisagreement(
        models[0], sampler, ScalarizedPosteriorTransform(weights)
    )
    reference = ReferenceBALD(models[1], ref_sampler, ReferenceScalarize(weights))
    for count in (2, 1, 2):
        candidate = query[..., :count, :].clone().requires_grad_()
        value = acq(candidate)
        expected = reference(candidate)
        torch.testing.assert_close(value, expected, atol=3e-10, rtol=3e-8)
        torch.testing.assert_close(
            sampler.base_samples, ref_sampler.base_samples, atol=2e-14, rtol=2e-13
        )
    normal = sampler.base_samples.detach().clone().neg()
    sampler.base_samples, ref_sampler.base_samples = normal, normal.clone()
    expected = reference(query)
    for kind in (lambda x: x, lambda x: x.numpy(), lambda x: x.tolist()):
        value, gradient = acq.value_and_grad(kind(query))
        torch.testing.assert_close(
            torch.as_tensor(value, dtype=query.dtype), expected, atol=3e-10, rtol=3e-8
        )
        assert torch.isfinite(torch.as_tensor(gradient)).all()
    sampler.base_samples = normal.requires_grad_()
    with pytest.raises(UnsupportedError, match="base samples require backend='torch'"):
        acq(query)


def test_native_multioutput_bald_requires_scalarization_before_native_inference():
    models, _, _ = _models()
    with pytest.raises(UnsupportedError, match="scalarizing posterior transform"):
        qBayesianActiveLearningByDisagreement(models[0])

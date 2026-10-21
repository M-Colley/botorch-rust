"""Posterior moments and draws retain the parameter state at construction."""

from __future__ import annotations

import pytest

from botorch_rs.models import ModelListGP, MultiTaskGP, SingleTaskGP
from botorch_rs.models.transforms.outcome import Log, Standardize
from botorch_rs.posteriors.transformed import TransformedPosterior
from helpers import KINDS, as_kind, assert_close, to_list


def identity_transform(posterior):
    return TransformedPosterior(
        posterior,
        sample_transform=lambda samples: samples,
        mean_transform=lambda mean, variance: mean,
        variance_transform=lambda mean, variance: variance,
    )


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("outcome", ["none", "standardize", "log"])
@pytest.mark.parametrize("nested", [False, True])
def test_single_task_snapshot_preserves_lazy_moments_and_transformed_samples(
    kind: str, outcome: str, nested: bool
) -> None:
    transform = {"none": None, "standardize": Standardize(m=1), "log": Log()}[outcome]
    model = SingleTaskGP(
        as_kind([[0.1], [0.4], [0.8]], kind),
        as_kind([[1.0], [2.0], [3.0]], kind),
        outcome_transform=transform,
    )
    query = as_kind([[0.3], [0.6]], kind)
    posterior, lazy = model.posterior(query), model.posterior(query)
    if nested:
        posterior, lazy = identity_transform(posterior), identity_transform(lazy)
    mean, variance = to_list(posterior.mean), to_list(posterior.variance)
    base = as_kind([[0.0, 0.0], [0.2, -0.3]], kind)
    draws = to_list(posterior.rsample_from_base_samples((2,), base))
    model.likelihood.noise = 100.0
    assert_close(posterior.mean, mean)
    assert_close(posterior.variance, variance)
    assert_close(lazy.mean, mean)
    assert_close(lazy.variance, variance)
    assert_close(posterior.rsample_from_base_samples((2,), base), draws)
    assert_close(lazy.rsample_from_base_samples((2,), base), draws)
    assert to_list(model.posterior(query).mean) != mean


@pytest.mark.parametrize("kind", KINDS)
def test_model_list_snapshot_preserves_child_parameter_states(kind: str) -> None:
    x = as_kind([[0.1], [0.4], [0.8]], kind)
    a = SingleTaskGP(x, as_kind([[1.0], [2.0], [3.0]], kind), outcome_transform=None)
    b = SingleTaskGP(x, as_kind([[0.5], [-0.2], [0.3]], kind), outcome_transform=None)
    model = ModelListGP(a, b)
    query = as_kind([[0.3]], kind)
    posterior, lazy = model.posterior(query), model.posterior(query)
    mean = to_list(posterior.mean)
    base = as_kind([[[0.2, -0.3]]], kind)
    draws = to_list(posterior.rsample_from_base_samples((1,), base))
    a.likelihood.noise, b.likelihood.noise = 100.0, 200.0
    assert_close(posterior.mean, mean)
    assert_close(lazy.mean, mean)
    assert_close(posterior.rsample_from_base_samples((1,), base), draws)


@pytest.mark.parametrize("kind", KINDS)
def test_multitask_immutable_handle_remains_valid(kind: str) -> None:
    model = MultiTaskGP(
        as_kind([[0.1, 0.0], [0.4, 0.0], [0.8, 0.0], [0.1, 1.0], [0.4, 1.0], [0.8, 1.0]], kind),
        as_kind([[0.1], [0.4], [0.7], [0.2], [0.5], [0.8]], kind),
        task_feature=-1,
        outcome_transform=None,
    )
    query = as_kind([[0.3]], kind)
    posterior, lazy = model.posterior(query), model.posterior(query)
    mean = to_list(posterior.mean)
    base = as_kind([[[0.2, -0.3]]], kind)
    draws = to_list(posterior.rsample_from_base_samples((1,), base))
    model.likelihood.noise = 100.0
    assert_close(lazy.mean, mean)
    assert_close(posterior.rsample_from_base_samples((1,), base), draws)


@pytest.mark.parametrize("kind", KINDS)
def test_fully_bayesian_mixture_retains_component_snapshot(kind: str) -> None:
    from botorch_rs.models.fully_bayesian import SaasFullyBayesianSingleTaskGP

    model = SaasFullyBayesianSingleTaskGP(
        as_kind([[0.1], [0.4], [0.8]], kind), as_kind([[1.0], [2.0], [3.0]], kind)
    )
    model.load_mcmc_samples(model.pyro_model.get_dummy_mcmc_samples(2))
    query = as_kind([[0.3]], kind)
    posterior, lazy = model.posterior(query), model.posterior(query)
    mean = to_list(posterior.mean)
    base = as_kind([[[0.2], [-0.3]]], kind)
    draws = to_list(posterior.rsample_from_base_samples((1,), base))
    model.likelihood.noise = 100.0
    assert_close(lazy.mean, mean)
    assert_close(posterior.rsample_from_base_samples((1,), base), draws)
    assert to_list(model.posterior(query).mean) != mean

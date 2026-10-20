"""Actual device KG/HVKG starts and independent heuristic/reference laws."""

from unittest.mock import patch

import pytest
import torch
from test_torch_acq_discrete import DEVICES, no_host
from test_torch_hypervolume_knowledge_gradient import build as build_hv
from test_torch_hypervolume_knowledge_gradient import options as hv_options
from test_torch_knowledge_gradient import model_data

from botorch_rs.acquisition.analytic import PosteriorMean
from botorch_rs.acquisition.knowledge_gradient import qKnowledgeGradient
from botorch_rs.acquisition.multi_objective.hypervolume_knowledge_gradient import (
    qHypervolumeKnowledgeGradient,
    qMultiFidelityHypervolumeKnowledgeGradient,
)
from botorch_rs.models import SingleTaskGP
from botorch_rs.optim import _torch_kg_initializers as helper
from botorch_rs.optim.initializers import (
    gen_one_shot_hvkg_initial_conditions,
    gen_one_shot_kg_initial_conditions,
    gen_value_function_initial_conditions,
)
from botorch_rs.optim.optimize import optimize_acqf
from botorch_rs.sampling.normal import SobolQMCNormalSampler


def kg(device, fantasies=3, batched=False, reference=False):
    X, Y = model_data(device)
    if batched:
        X = X.expand(2, *X.shape).clone()
        Y = torch.stack((Y, Y * 0.7))
    if reference:
        from botorch.acquisition.knowledge_gradient import qKnowledgeGradient as ReferenceKG
        from botorch.models import SingleTaskGP as ReferenceGP
        from botorch.sampling.normal import SobolQMCNormalSampler as ReferenceSampler

        return ReferenceKG(
            ReferenceGP(X, Y, outcome_transform=None),
            num_fantasies=fantasies,
            sampler=ReferenceSampler(torch.Size([fantasies]), seed=13),
        )
    return qKnowledgeGradient(
        SingleTaskGP(X, Y, backend="torch", outcome_transform=None),
        num_fantasies=fantasies,
        sampler=SobolQMCNormalSampler((fantasies,), seed=13),
    )


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("batch", [(), (2,)])
def test_boltzmann_optimizer_resampling_independent_formula(device, batch):
    candidates = torch.linspace(0.1, 0.9, 4, dtype=torch.double, device=device).reshape(4, 1, 1)
    values = candidates[:, 0, 0].sin() * 5
    if batch:
        candidates = candidates[:, None].expand(4, 2, 1, 1)
        values = torch.stack((values, -values), -1)
    standardized = (values - values.mean(0)) / values.std(0)
    weights = (1.7 * standardized).exp().movedim(0, -1)
    torch.manual_seed(19)
    indices = (
        torch.multinomial(weights.reshape(-1, 4), 11, replacement=True)
        .reshape(*batch, 11)
        .movedim(-1, 0)
    )
    expected = candidates.gather(0, indices[..., None, None].expand(11, *batch, 1, 1))
    torch.manual_seed(19)
    with no_host():
        actual = helper._resample(candidates, values, 11, torch.Size(batch), None, 1.7)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("device", DEVICES)
def test_kg_tail_placement_matches_official_heuristic_with_identical_raw_optima(device):
    from botorch.optim.initializers import gen_one_shot_kg_initial_conditions as reference

    actual_acq, reference_acq = kg(device, fantasies=4), kg(device, fantasies=4, reference=True)
    bounds = torch.tensor([[0.0], [1.0]], dtype=torch.double, device=device)
    initial = torch.linspace(0.02, 0.98, 3 * 6, dtype=torch.double, device=device).reshape(3, 6, 1)
    optima = bounds.new_tensor([[[0.2]], [[0.5]], [[0.8]]])
    values = bounds.new_tensor([0.1, 0.4, 0.2])
    options = {"frac_random": 0.5, "eta": 1.7}
    with (
        patch(
            "botorch.optim.initializers.gen_batch_initial_conditions", return_value=initial.clone()
        ),
        patch("botorch.optim.optimize.optimize_acqf", return_value=(optima, values)),
    ):
        torch.manual_seed(19)
        expected = reference(reference_acq, bounds, 2, 3, 8, options=options)
    with (
        patch.object(helper, "gen_initial_conditions", return_value=initial.clone()),
        patch.object(helper, "_inner", return_value=(optima, values)),
        no_host(),
    ):
        torch.manual_seed(19)
        actual = gen_one_shot_kg_initial_conditions(actual_acq, bounds, 2, 3, 8, options=options)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(actual[:, :4], initial[:, :4])


@pytest.mark.parametrize("device", DEVICES)
def test_single_kg_fantasy_keeps_random_tail_without_zero_length_assignment(device):
    acquisition = kg(device, fantasies=1)
    bounds = torch.tensor([[0.0], [1.0]], dtype=torch.double, device=device)
    initial = bounds.new_tensor([[[0.2], [0.3]], [[0.6], [0.8]]])
    with (
        patch.object(helper, "gen_initial_conditions", return_value=initial),
        patch.object(helper, "_inner", side_effect=AssertionError("unneeded inner optimization")),
        no_host(),
    ):
        actual = gen_one_shot_kg_initial_conditions(acquisition, bounds, 1, 2, 8)
    assert actual is initial


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("batched", [False, True])
def test_kg_real_device_inner_optimization_and_outer_default_dispatch(device, batched):
    acquisition = kg(device, fantasies=2, batched=batched)
    bounds = torch.tensor([[0.0], [1.0]], dtype=torch.double, device=device)
    options = {
        "seed": 13,
        "num_inner_restarts": 2,
        "raw_inner_samples": 8,
        "maxiter": 4,
        "frac_random": 0.5,
    }
    recorded = []

    def retain(initial_conditions, acquisition_function, **kwargs):
        recorded.append(initial_conditions)
        return initial_conditions, acquisition_function(initial_conditions)

    with no_host():
        initial = gen_one_shot_kg_initial_conditions(acquisition, bounds, 1, 2, 8, options=options)
        candidate, value = optimize_acqf(
            acquisition,
            bounds,
            1,
            2,
            8,
            options=options,
            gen_candidates=retain,
            return_full_tree=True,
        )
    shape = (2, 2, 3, 1) if batched else (2, 3, 1)
    assert initial.shape == shape and recorded[0].shape == shape
    assert bool(torch.isfinite(candidate).all() & torch.isfinite(value).all())
    assert acquisition.X_pending is None


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("multifidelity", [False, True])
def test_hvkg_real_device_inner_search_and_fixed_fidelity_starts(device, multifidelity):
    X, Y = model_data(device, outputs=2)
    if multifidelity:
        X = torch.cat((X, X.square()), -1)
    cls = (
        qMultiFidelityHypervolumeKnowledgeGradient
        if multifidelity
        else qHypervolumeKnowledgeGradient
    )
    extra = {"target_fidelities": {1: 1.0}} if multifidelity else {}
    acquisition = cls(
        build_hv(X, Y),
        ref_point=X.new_tensor([-1.4, -1.4]),
        use_posterior_mean=True,
        **hv_options(fantasies=2),
        **extra,
    )
    bounds = torch.stack((X.new_zeros(X.shape[-1]), X.new_ones(X.shape[-1])))
    options = {
        "seed": 13,
        "num_inner_restarts": 2,
        "raw_inner_samples": 6,
        "maxiter": 4,
        "frac_random": 0.5,
    }
    with no_host():
        initial = gen_one_shot_hvkg_initial_conditions(
            acquisition, bounds, 1, 2, 6, options=options
        )
        scores = acquisition(initial)
    assert initial.shape == (2, 5, X.shape[-1])
    assert bool(torch.isfinite(scores).all())
    if multifidelity:
        assert bool((initial[:, 1:, 1] == 1.0).all(-1).any())
    assert acquisition.X_pending is None


@pytest.mark.parametrize("device", DEVICES)
def test_fantasy_value_initializer_retains_batch_law_and_device(device):
    acquisition = kg(device, fantasies=2)
    model = acquisition.model
    X = torch.tensor([[0.24], [0.67]], dtype=torch.double, device=device)
    fantasy = model.fantasize(X, SobolQMCNormalSampler((2,), seed=17))
    value = PosteriorMean(fantasy)
    bounds = X.new_tensor([[0.0], [1.0]])
    with no_host():
        initial = gen_value_function_initial_conditions(
            value,
            bounds,
            2,
            8,
            model,
            options={"seed": 13, "num_inner_restarts": 2, "raw_inner_samples": 8, "maxiter": 4},
        )
    assert initial.shape == (2, 2, 1, 1)
    assert bool(torch.isfinite(value(initial)).all())


@pytest.mark.parametrize("device", DEVICES)
def test_hvkg_batched_current_optima_and_conditional_fantasies(device):
    X, Y = model_data(device, outputs=2)
    X = X.expand(2, *X.shape).clone()
    Y = torch.stack((Y, Y * 0.7))
    acquisition = qHypervolumeKnowledgeGradient(
        build_hv(X, Y),
        ref_point=X.new_tensor([-1.4, -1.4]),
        use_posterior_mean=True,
        **hv_options(fantasies=2),
    )
    bounds = X.new_tensor([[0.0], [1.0]])
    with no_host():
        initial = gen_one_shot_hvkg_initial_conditions(
            acquisition,
            bounds,
            1,
            2,
            6,
            options={
                "seed": 13,
                "num_inner_restarts": 2,
                "raw_inner_samples": 6,
                "maxiter": 3,
                "frac_random": 0.5,
            },
        )
        scores = acquisition(initial)
    assert initial.shape == (2, 2, 5, 1)
    assert scores.shape == (2, 2)
    assert bool(torch.isfinite(scores).all())


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("count", [0, 2])
def test_hvkg_optimized_restart_quota_matches_independent_layout(device, count):
    X, Y = model_data(device, outputs=2)
    acquisition = qHypervolumeKnowledgeGradient(
        build_hv(X, Y), ref_point=X.new_tensor([-1.4, -1.4]), **hv_options(fantasies=2)
    )
    bounds = X.new_tensor([[0.0], [1.0]])
    initial = torch.linspace(0.1, 0.9, 3 * 5, dtype=X.dtype, device=X.device).reshape(3, 5, 1)
    optima = X.new_tensor([[[0.2], [0.3]], [[0.7], [0.8]]])
    values = X.new_tensor([0.1, 0.4])
    calls = []

    def conditions(acq, bounds, q, restarts, raw, fixed, options, **kwargs):
        if "fixed_X_fantasies" in kwargs:
            fantasies = kwargs["fixed_X_fantasies"]
            calls.append(fantasies)
            return torch.cat((bounds.new_full((1, 1, 1), 0.42), fantasies.unsqueeze(0)), -2)
        return initial.clone()

    fraction = 0.99 if count == 0 else 0.4
    with (
        patch.object(helper, "gen_initial_conditions", side_effect=conditions),
        patch.object(helper, "_inner", return_value=(optima, values)) as optimizer,
        no_host(),
    ):
        result = gen_one_shot_hvkg_initial_conditions(
            acquisition, bounds, 1, 3, 8, options={"frac_random": fraction, "seed": 17}
        )
    assert len(calls) == count
    assert optimizer.call_count == (1 if count else 0)
    if count == 0:
        torch.testing.assert_close(result, initial)
    else:
        assert int((result[:, 0, 0] == 0.42).sum()) == count
        for fantasies in calls:
            blocks = fantasies.reshape(2, 2, 1)
            assert bool(
                ((blocks[:, None, :, :] == optima[None, :, :, :]).all((-1, -2))).any(-1).all()
            )

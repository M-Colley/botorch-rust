"""Device pruning, batched noisy baselines and live constraint temperatures."""

# ruff: noqa: E402
from __future__ import annotations

import warnings
from unittest.mock import patch

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("botorch")

from botorch.acquisition.utils import prune_inferior_points as reference_prune
from botorch.utils.objective import compute_feasibility_indicator as reference_feasible
from botorch.utils.objective import compute_smoothed_feasibility_indicator as reference_smooth
from test_torch_backend_acquisition import FixedSampler, TensorModel, acquisition, derivatives

from botorch_rs.acquisition.utils import prune_inferior_points
from botorch_rs.utils.objective import (
    apply_constraints,
    compute_feasibility_indicator,
    compute_smoothed_feasibility_indicator,
)

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("fat", [False, True, None])
@pytest.mark.parametrize("log", [False, True])
def test_constraint_temperatures_keep_full_device_gradients(device, fat, log):
    samples = (
        torch.linspace(-0.6, 0.8, 12, dtype=torch.double, device=device)
        .reshape(3, 2, 2)
        .requires_grad_()
    )
    eta = samples.new_tensor([0.2, 0.4], requires_grad=True)
    constraints = (
        [lambda s: s[..., 0].sigmoid(), lambda s: s[..., 1].sigmoid()]
        if fat is None
        else [lambda s: s[..., 0] - 0.1, lambda s: s[..., 1] + 0.2]
    )
    expected = reference_smooth(constraints, samples, eta, log=log, fat=fat)
    ref_first = torch.autograd.grad(
        expected.sum(), (samples, eta), create_graph=True, allow_unused=True
    )
    ref_second = torch.autograd.grad(
        sum(g.square().sum() for g in ref_first if g is not None), (samples, eta), allow_unused=True
    )
    with (
        patch.object(torch.Tensor, "cpu", side_effect=AssertionError("host transfer")),
        patch.object(torch.Tensor, "numpy", side_effect=AssertionError("host transfer")),
        patch.object(torch.Tensor, "tolist", side_effect=AssertionError("host transfer")),
    ):
        actual = compute_smoothed_feasibility_indicator(constraints, samples, eta, log=log, fat=fat)
        first = torch.autograd.grad(
            actual.sum(), (samples, eta), create_graph=True, allow_unused=True
        )
        second = torch.autograd.grad(
            sum(g.square().sum() for g in first if g is not None), (samples, eta), allow_unused=True
        )
        weighted = apply_constraints(
            samples[..., 0], constraints, samples, infeasible_cost=1, eta=eta
        )
        assert weighted.device == samples.device
    for own, ref in [
        (actual, expected),
        *zip(first, ref_first, strict=True),
        *zip(second, ref_second, strict=True),
    ]:
        assert (own is None) == (ref is None)
        if own is not None:
            torch.testing.assert_close(own, ref, atol=2e-10, rtol=2e-9)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("shape,dim", [((5, 2, 3, 2), -3), ((5, 2, 3, 2), 1), ((2, 3, 2), -3)])
def test_boolean_feasibility_marginalization_stays_on_device(device, shape, dim):
    samples = torch.linspace(-0.3, 0.7, torch.Size(shape).numel(), device=device).reshape(shape)
    constraints = [lambda s: s[..., 0] - 0.3]
    expected = reference_feasible(constraints, samples, dim)
    with patch.object(torch.Tensor, "tolist", side_effect=AssertionError("host transfer")):
        actual = compute_feasibility_indicator(constraints, samples, dim)
    torch.testing.assert_close(actual, expected)
    assert actual.device == samples.device


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("infeasible", [False, True])
def test_public_pruning_matches_reference_with_common_samples_and_live_selected_inputs(
    device, infeasible
):
    model = TensorModel(device)
    x = model.train_X
    constraints = [lambda s: s[..., 0].abs() + 0.1] if infeasible else None
    kwargs = dict(sampler=FixedSampler(4, device, (31,)), constraints=constraints)
    expected = reference_prune(model, x, **kwargs)
    with (
        patch.object(torch.Tensor, "cpu", side_effect=AssertionError("host transfer")),
        patch.object(torch.Tensor, "tolist", side_effect=AssertionError("host transfer")),
    ):
        actual = prune_inferior_points(model, x, **kwargs)
        gradient = torch.autograd.grad(actual.sum(), x)[0]
    torch.testing.assert_close(actual, expected)
    assert gradient.sum() == actual.numel()


@pytest.mark.parametrize("device", DEVICES)
def test_public_pruning_supports_explicit_nonensemble_model_batch_marginalization(device):
    from botorch.models import SingleTaskGP as ReferenceGP

    from botorch_rs.models import SingleTaskGP

    x = torch.tensor([[0.03], [0.29], [0.58], [0.87]], device=device, dtype=torch.double)
    y = torch.cat(((5 * x).sin(), (3 * x).cos()), dim=-1).transpose(-1, -2).unsqueeze(-1)
    own = SingleTaskGP(x.expand(2, 4, 1), y, backend="torch")
    reference = ReferenceGP(x.expand(2, 4, 1), y)
    kwargs = dict(sampler=FixedSampler(4, device, (41,)), marginalize_dim=-3)
    torch.testing.assert_close(
        prune_inferior_points(own, x, **kwargs), reference_prune(reference, x, **kwargs)
    )
    noisy = acquisition("qLogNoisyExpectedImprovement")(
        own, X_baseline=x, sampler=FixedSampler(4, device), marginalize_dim=-3, cache_root=False
    )
    assert noisy.X_baseline.ndim == 2 and len(noisy.X_baseline) > 0


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("name", ["qNoisyExpectedImprovement", "qLogNoisyExpectedImprovement"])
@pytest.mark.parametrize("multibatch", [False, True])
def test_batched_noisy_baselines_match_complete_reference_graphs(device, name, multibatch):
    model = TensorModel(device)
    prefix = (2, 1) if multibatch else (2,)
    query_prefix = (2, 3) if multibatch else (2,)
    baseline = model.train_X.detach()[::2].expand(*prefix, 2, 1).clone().requires_grad_()
    query = (
        torch.linspace(
            0.17, 0.83, torch.Size(query_prefix).numel() * 2, device=device, dtype=torch.double
        )
        .reshape(*query_prefix, 2, 1)
        .requires_grad_()
    )
    kwargs = dict(
        X_baseline=baseline,
        sampler=FixedSampler(4, device),
        cache_root=False,
        prune_baseline=False,
        marginalize_dim=0,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        own = acquisition(name)(model, **kwargs)
        reference = acquisition(name, True)(model, **kwargs)
    actual, expected = own(query), reference(query)
    torch.testing.assert_close(actual, expected, atol=2e-10, rtol=2e-9)
    for own_order, ref_order in zip(
        derivatives(actual, query, model), derivatives(expected, query, model), strict=True
    ):
        for a, b in zip(own_order, ref_order, strict=True):
            torch.testing.assert_close(a, b, atol=2e-8, rtol=2e-7)
    own_baseline = torch.autograd.grad(actual.sum(), baseline)[0]
    ref_baseline = torch.autograd.grad(expected.sum(), baseline)[0]
    torch.testing.assert_close(own_baseline, ref_baseline, atol=2e-8, rtol=2e-7)


@pytest.mark.parametrize("device", DEVICES)
def test_batched_infeasible_cost_uses_each_baseline_batch_bounding_box(device):
    from botorch.acquisition.utils import get_infeasible_cost

    model = TensorModel(device)
    baseline = model.train_X.detach()[::2].expand(2, 3, 2, 1).clone()
    baseline[1] += 0.04
    own = acquisition("qLogNoisyExpectedImprovement")(
        model,
        X_baseline=baseline,
        sampler=FixedSampler(4, device),
        prune_baseline=False,
        cache_root=False,
    )
    lower = own._infeasible_lower_bound()
    points = own.lower_bound_points
    assert points.shape == (2, 3, 32, 1)
    low = baseline.amin(dim=-2, keepdim=True)
    high = baseline.amax(dim=-2, keepdim=True)
    assert bool((points >= low - 0.1 * (high - low)).all())
    assert bool((points <= high + 0.1 * (high - low)).all())
    expected = -get_infeasible_cost(points, model, objective=own.objective)
    torch.testing.assert_close(lower, expected.reshape(()))

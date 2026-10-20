"""Actual candidate search using device-native acquisition derivatives."""

import functools
from unittest.mock import patch

import pytest
import torch

from botorch_rs.acquisition.analytic import LogExpectedImprovement, PosteriorMean
from botorch_rs.acquisition.knowledge_gradient import qKnowledgeGradient
from botorch_rs.acquisition.monte_carlo import qSimpleRegret
from botorch_rs.generation.gen import gen_candidates_scipy, gen_candidates_torch
from botorch_rs.models import SingleTaskGP
from botorch_rs.optim import optimize_acqf
from botorch_rs.sampling.normal import SobolQMCNormalSampler

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


class Quadratic(torch.nn.Module):
    _tensor_backend = "torch"

    def forward(self, X):
        return -(X - 0.73).square().sum(dim=(-2, -1))


@pytest.mark.parametrize("device", DEVICES)
def test_torch_optimizer_iterates_match_reference_and_stay_on_device(device):
    from botorch.generation.gen import gen_candidates_torch as reference

    initial = torch.tensor([[[0.1, 0.2]], [[0.8, 0.4]]], dtype=torch.double, device=device)
    options = {"optimizer_options": {"lr": 0.05}, "stopping_criterion_options": {"maxiter": 20}}
    expected = reference(
        initial.clone(), Quadratic(), lower_bounds=0.0, upper_bounds=1.0, options=options
    )
    with (
        patch.object(torch.Tensor, "cpu", side_effect=AssertionError("host transfer")),
        patch.object(torch.Tensor, "tolist", side_effect=AssertionError("host transfer")),
        patch.object(torch.Tensor, "numpy", side_effect=AssertionError("host transfer")),
    ):
        actual = gen_candidates_torch(
            initial, Quadratic(), lower_bounds=0.0, upper_bounds=1.0, options=options
        )
    torch.testing.assert_close(actual[0], expected[0], atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(actual[1], expected[1], atol=1e-12, rtol=1e-12)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("generator", [gen_candidates_scipy, gen_candidates_torch])
def test_optimize_selects_restarts_fixed_features_and_recomputes_postprocessing(device, generator):
    bounds = torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.double, device=device)
    options = {"seed": 13, "maxiter": 80}
    if generator is gen_candidates_torch:
        options = {"seed": 13, "stopping_criterion_options": {"maxiter": 100, "n_window": 100}}
    candidates, values = optimize_acqf(
        Quadratic(),
        bounds,
        q=2,
        num_restarts=4,
        raw_samples=24,
        gen_candidates=generator,
        fixed_features={1: 0.4},
        options=options,
        return_best_only=False,
    )
    assert candidates.shape == (4, 2, 2)
    assert values.shape == (4,)
    torch.testing.assert_close(candidates[..., 1], candidates.new_full((4, 2), 0.4))
    torch.testing.assert_close(
        candidates[..., 0], candidates.new_full((4, 2), 0.73), atol=0.015, rtol=0
    )
    torch.testing.assert_close(values, Quadratic()(candidates))
    best, value = optimize_acqf(
        Quadratic(),
        bounds,
        q=1,
        num_restarts=3,
        raw_samples=16,
        gen_candidates=generator,
        options=options,
        post_processing_func=lambda Z: Z.round(),
    )
    torch.testing.assert_close(value, Quadratic()(best.unsqueeze(0)).squeeze(0))
    assert best.device == bounds.device


@pytest.mark.parametrize("device", DEVICES)
def test_slsqp_linear_interpoint_and_nonlinear_constraints(device):
    initial = torch.full((2, 2, 2), 0.2, dtype=torch.double, device=device)
    inequalities = [
        (torch.tensor([[0, 0], [1, 0]], device=device), initial.new_tensor([-1.0, -1.0]), -1.0)
    ]
    equalities = [(torch.tensor([1], device=device), initial.new_tensor([1.0]), 0.2)]
    nonlinear = [(lambda Z: 0.26 - Z[0].square(), True)]
    candidates, values = gen_candidates_scipy(
        initial,
        Quadratic(),
        0.0,
        1.0,
        inequality_constraints=inequalities,
        equality_constraints=equalities,
        nonlinear_inequality_constraints=nonlinear,
        options={"maxiter": 100, "ftol": 1e-12},
    )
    torch.testing.assert_close(candidates[..., 0], initial.new_full((2, 2), 0.5), atol=1e-7, rtol=0)
    torch.testing.assert_close(candidates[..., 1], initial.new_full((2, 2), 0.2), atol=1e-8, rtol=0)
    torch.testing.assert_close(values, Quadratic()(candidates))


@pytest.mark.parametrize("device", DEVICES)
def test_gp_candidates_improve_and_keep_model_parameter_grads_clean(device):
    X = torch.linspace(0, 1, 5, dtype=torch.double, device=device).unsqueeze(-1)
    model = SingleTaskGP(X, (5 * X).sin(), backend="torch")
    acq = PosteriorMean(model)
    initial = X.new_tensor([[[0.85]], [[0.95]]])
    before = acq(initial).max()
    bounds = X.new_tensor([[0.0], [1.0]])
    candidates, value = optimize_acqf(
        acq, bounds, q=1, num_restarts=2, batch_initial_conditions=initial, options={"maxiter": 35}
    )
    assert value > before
    torch.testing.assert_close(value, acq(candidates).squeeze(0))
    assert all(parameter.grad is None for parameter in model.parameters())
    candidate = candidates.detach().requires_grad_(True)
    value, gradient = acq.value_and_grad(candidate)
    assert gradient.requires_grad
    assert torch.isfinite(torch.autograd.grad(gradient.sum(), candidate)[0]).all()


@pytest.mark.parametrize("device", DEVICES)
def test_sequential_pending_restored_and_kg_augmented_candidates(device):
    X = torch.linspace(0, 1, 5, dtype=torch.double, device=device).unsqueeze(-1)
    model = SingleTaskGP(X, (5 * X).sin(), backend="torch")
    bounds = X.new_tensor([[0.0], [1.0]])
    acq = qSimpleRegret(model, sampler=SobolQMCNormalSampler((8,), seed=3), X_pending=X[:1])
    pending = acq.X_pending.clone()
    candidates, values = optimize_acqf(
        acq,
        bounds,
        q=2,
        num_restarts=2,
        raw_samples=8,
        sequential=True,
        gen_candidates=gen_candidates_torch,
        options={"maxiter": 3, "seed": 5},
    )
    assert candidates.shape == (2, 1)
    assert values.shape == (2,)
    torch.testing.assert_close(acq.X_pending, pending)
    kg = qKnowledgeGradient(model, num_fantasies=2, sampler=SobolQMCNormalSampler((2,), seed=11))
    arguments = dict(
        acq_function=kg,
        bounds=bounds,
        q=1,
        num_restarts=2,
        raw_samples=6,
        gen_candidates=functools.partial(gen_candidates_torch, optimizer=torch.optim.SGD),
        options={"maxiter": 2, "seed": 19},
    )
    full, _ = optimize_acqf(**arguments, return_full_tree=True)
    extracted, _ = optimize_acqf(**arguments)
    assert full.shape == (3, 1)
    torch.testing.assert_close(extracted, full[:1])


@pytest.mark.parametrize("device", DEVICES)
def test_logei_torch_search_no_host_value_roundtrip(device):
    X = torch.linspace(0, 1, 5, dtype=torch.double, device=device).unsqueeze(-1)
    model = SingleTaskGP(X, (5 * X).sin(), backend="torch")
    acq = LogExpectedImprovement(model, best_f=0.7)
    initial = X.new_tensor([[[0.2]], [[0.8]]])
    with (
        patch.object(torch.Tensor, "cpu", side_effect=AssertionError("host transfer")),
        patch.object(torch.Tensor, "tolist", side_effect=AssertionError("host transfer")),
        patch.object(torch.Tensor, "numpy", side_effect=AssertionError("host transfer")),
    ):
        candidates, values = gen_candidates_torch(
            initial, acq, 0.0, 1.0, options={"stopping_criterion_options": {"maxiter": 4}}
        )
    assert torch.isfinite(values).all()
    assert candidates.device == X.device


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("generator", [gen_candidates_scipy, gen_candidates_torch])
def test_none_fixed_features_retain_each_restart_values_and_invalid_fixed_bounds_raise(
    device, generator
):
    initial = torch.tensor([[[0.1, 0.2]], [[0.4, 0.8]]], dtype=torch.double, device=device)
    options = (
        {"maxiter": 30}
        if generator is gen_candidates_scipy
        else {"stopping_criterion_options": {"maxiter": 30}}
    )
    candidates, _ = generator(
        initial, Quadratic(), 0.0, 1.0, fixed_features={1: None}, options=options
    )
    torch.testing.assert_close(candidates[..., 1], initial[..., 1])
    assert (candidates[..., 0] - 0.73).abs().sum() < (initial[..., 0] - 0.73).abs().sum()
    for fixed in ({1: 1.2}, {3: 0.5}):
        with pytest.raises(ValueError, match="Fixed feature"):
            generator(initial, Quadratic(), 0.0, 1.0, fixed_features=fixed, options=options)
    per_restart = initial.new_tensor([0.3, 0.6])
    candidates, _ = generator(
        initial, Quadratic(), 0.0, 1.0, fixed_features={1: per_restart}, options=options
    )
    torch.testing.assert_close(candidates[..., 1], per_restart[:, None])

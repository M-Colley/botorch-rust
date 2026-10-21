# ruff: noqa: E402
"""Gaussian analytic constraints, live thresholds, and weighted q means."""

import pytest

pytest.importorskip("torch")


import importlib
from unittest.mock import patch

import torch

from botorch_rs.acquisition._torch_backend_analytic import log_normal_interval
from botorch_rs.models import SingleTaskGP

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize(
    "name",
    [
        "ConstrainedExpectedImprovement",
        "LogConstrainedExpectedImprovement",
        "LogProbabilityOfFeasibility",
    ],
)
@pytest.mark.parametrize("maximize", [False, True])
def test_constrained_values_and_query_hessians_match_reference(device, name, maximize):
    pytest.importorskip("botorch")
    from botorch.models import SingleTaskGP as ReferenceGP

    X = torch.tensor([[0.03], [0.29], [0.58], [0.87]], device=device, dtype=torch.double)
    Y = torch.cat(((5 * X).sin(), (3 * X).cos(), X.square(), 1 - X), dim=-1)
    query = X.new_tensor([[[0.19]], [[0.71]]], requires_grad=True)
    constraints = {1: (None, 0.8), 2: (0.1, None), 3: (0.05, 0.7)}
    arguments = {"constraints": constraints}
    if name != "LogProbabilityOfFeasibility":
        arguments.update(best_f=0.4, objective_index=0, maximize=maximize)
    cls = getattr(importlib.import_module("botorch_rs.acquisition.analytic"), name)
    pytest.importorskip("botorch")
    ref_cls = getattr(importlib.import_module("botorch.acquisition.analytic"), name)
    actual = cls(SingleTaskGP(X, Y, backend="torch"), **arguments)(query)
    expected = ref_cls(ReferenceGP(X, Y), **arguments)(query)
    torch.testing.assert_close(actual, expected, atol=3e-8, rtol=3e-8)
    gradient = torch.autograd.grad(actual.sum(), query, create_graph=True)[0]
    ref_gradient = torch.autograd.grad(expected.sum(), query, create_graph=True)[0]
    torch.testing.assert_close(gradient, ref_gradient, atol=2e-6, rtol=2e-7)
    hessian = torch.autograd.grad(gradient.sum(), query)[0]
    ref_hessian = torch.autograd.grad(ref_gradient.sum(), query)[0]
    torch.testing.assert_close(hessian, ref_hessian, atol=2e-5, rtol=2e-6)


@pytest.mark.parametrize("device", DEVICES)
def test_constraints_and_best_value_keep_full_device_graphs(device):
    from botorch_rs.acquisition.analytic import LogConstrainedExpectedImprovement

    X = torch.tensor(
        [[0.03], [0.29], [0.58], [0.87]], device=device, dtype=torch.double, requires_grad=True
    )
    Y = torch.cat(((5 * X.detach()).sin(), (3 * X.detach()).cos()), dim=-1).requires_grad_()
    query = X.new_tensor([[0.21]], requires_grad=True)
    model = SingleTaskGP(X, Y, backend="torch")
    lower = X.new_tensor(0.1, requires_grad=True)
    upper = X.new_tensor(0.9, requires_grad=True)
    best = X.new_tensor(0.4, requires_grad=True)

    def evaluate(a, b, c):
        return LogConstrainedExpectedImprovement(
            model, best_f=c, objective_index=0, constraints={1: (a, b)}
        )(query)

    with (
        patch.object(torch.Tensor, "cpu", side_effect=AssertionError("host transfer")),
        patch.object(torch.Tensor, "numpy", side_effect=AssertionError("host transfer")),
        patch.object(torch.Tensor, "tolist", side_effect=AssertionError("host transfer")),
    ):
        leaves = (X, Y, query, lower, upper, best, *model.parameters())
        gradients = torch.autograd.grad(
            evaluate(lower, upper, best).sum(), leaves, create_graph=True
        )
        second = torch.autograd.grad(sum(g.square().sum() for g in gradients), leaves)
    assert all(torch.isfinite(g).all() for g in second)
    assert torch.autograd.gradcheck(evaluate, (lower, upper, best))
    assert torch.autograd.gradgradcheck(evaluate, (lower, upper, best))


@pytest.mark.parametrize("device", DEVICES)
def test_scalarized_mean_weights_follow_parameter_updates_and_full_graphs(device):
    from botorch_rs.acquisition.analytic import ScalarizedPosteriorMean

    X = torch.tensor([[0.03], [0.29], [0.58], [0.87]], device=device, dtype=torch.double)
    model = SingleTaskGP(X, (5 * X).sin(), backend="torch")
    query = X.new_tensor([[0.21], [0.65]], requires_grad=True)
    weights = X.new_tensor([0.2, -0.6], requires_grad=True)
    acq = ScalarizedPosteriorMean(model, weights)
    with torch.no_grad():
        weights.add_(0.1)
    actual = acq(query)
    torch.testing.assert_close(
        actual, model.posterior(query.unsqueeze(0)).mean.squeeze(-1) @ weights
    )
    leaves = (query, weights, *model.parameters())
    gradient = torch.autograd.grad(actual.sum(), leaves, create_graph=True)
    second = torch.autograd.grad(sum(g.square().sum() for g in gradient), leaves, allow_unused=True)
    assert all(g is None or torch.isfinite(g).all() for g in second)


@pytest.mark.parametrize("device", DEVICES)
def test_log_interval_extreme_tail_values_and_second_derivatives(device):
    pytest.importorskip("botorch")
    from botorch.utils.probability.utils import log_prob_normal_in

    lower = torch.tensor(
        [-40.0, -10.01, -0.1, 9.99, 40.0], dtype=torch.double, device=device, requires_grad=True
    )
    upper = lower.detach().clone().add_(0.02).requires_grad_()
    actual = log_normal_interval(lower, upper)
    torch.testing.assert_close(actual, log_prob_normal_in(lower, upper), atol=2e-11, rtol=2e-11)
    assert torch.autograd.gradcheck(log_normal_interval, (lower, upper), atol=2e-5)
    assert torch.autograd.gradgradcheck(log_normal_interval, (lower, upper), atol=2e-5)


@pytest.mark.parametrize("device", DEVICES)
def test_nongaussian_outcome_transform_rejects_gaussian_closed_form_acquisitions(device):
    from botorch_rs.acquisition.analytic import (
        LogConstrainedExpectedImprovement,
        LogExpectedImprovement,
        PosteriorMean,
    )
    from botorch_rs.acquisition.logei import qLogExpectedImprovement
    from botorch_rs.exceptions.errors import UnsupportedError
    from botorch_rs.models.transforms.outcome import Log
    from botorch_rs.sampling.normal import SobolQMCNormalSampler

    X = torch.tensor([[0.03], [0.29], [0.58], [0.87]], dtype=torch.double, device=device)
    model = SingleTaskGP(X, (3 * X).exp(), backend="torch", outcome_transform=Log())
    query = X.new_tensor([[0.21]], requires_grad=True)
    with pytest.raises(UnsupportedError, match="Gaussian posterior"):
        LogExpectedImprovement(model, best_f=1.0)(query)
    # The actual transformed mean and sampled improvement remain valid.
    assert torch.isfinite(PosteriorMean(model)(query)).all()
    value = qLogExpectedImprovement(model, best_f=1.0, sampler=SobolQMCNormalSampler((8,), seed=7))(
        query
    )
    gradient = torch.autograd.grad(value.sum(), query, create_graph=True)[0]
    assert torch.isfinite(torch.autograd.grad(gradient.sum(), query)[0]).all()
    multi = SingleTaskGP(
        X, torch.cat(((3 * X).exp(), (2 * X).exp()), -1), backend="torch", outcome_transform=Log()
    )
    with pytest.raises(UnsupportedError, match="Gaussian posterior"):
        LogConstrainedExpectedImprovement(
            multi, best_f=1.0, objective_index=0, constraints={1: (0.1, None)}
        )(query)

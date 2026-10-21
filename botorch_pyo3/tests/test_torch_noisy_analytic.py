"""Noiseless fantasy laws and analytic noisy improvement on CPU and CUDA."""

# ruff: noqa: E402
from __future__ import annotations

import warnings
from unittest.mock import patch

import pytest

torch = pytest.importorskip("torch")
gpytorch = pytest.importorskip("gpytorch")
pytest.importorskip("botorch")

from botorch.acquisition.analytic import LogNoisyExpectedImprovement as ReferenceLogNEI
from botorch.acquisition.analytic import NoisyExpectedImprovement as ReferenceNEI
from botorch.models import SingleTaskGP as ReferenceGP
from botorch.models.transforms.outcome import Standardize as ReferenceStandardize
from test_torch_backend_acquisition import FixedSampler

from botorch_rs.acquisition.analytic import LogNoisyExpectedImprovement, NoisyExpectedImprovement
from botorch_rs.models import SingleTaskGP
from botorch_rs.models.transforms.outcome import Standardize

DEVICES = [
    "cpu",
    pytest.param(
        "cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
    ),
]


def _models(device, standardized):
    x = torch.tensor([[0.05], [0.35], [0.72], [0.95]], device=device, dtype=torch.double)
    y = torch.tensor([[0.12], [0.62], [-0.17], [0.31]], device=device, dtype=torch.double)
    models = []
    for reference in (False, True):
        kernel = gpytorch.kernels.RBFKernel().to(x)
        kernel.lengthscale = 0.3
        mean = gpytorch.means.ConstantMean().to(x)
        mean.constant = 0.2
        kwargs = {} if reference else {"backend": "torch"}
        outcome = (
            (ReferenceStandardize(m=1) if reference else Standardize(m=1)) if standardized else None
        )
        models.append(
            (ReferenceGP if reference else SingleTaskGP)(
                x,
                y,
                train_Yvar=torch.full_like(y, 0.02),
                covar_module=kernel,
                mean_module=mean,
                outcome_transform=outcome,
                **kwargs,
            )
        )
    return x, *models


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("log", [False, True])
@pytest.mark.parametrize("standardized", [False, True])
@pytest.mark.parametrize("maximize", [False, True])
def test_noisy_analytic_fantasy_law_value_and_query_hessian(device, log, standardized, maximize):
    x, model, ref_model = _models(device, standardized)
    observed = x[::2]
    own_sampler = FixedSampler(len(observed), device, sample_shape=(5,))
    ref_sampler = FixedSampler(len(observed), device, sample_shape=(5,))
    cls, reference_cls = (
        (LogNoisyExpectedImprovement, ReferenceLogNEI)
        if log
        else (NoisyExpectedImprovement, ReferenceNEI)
    )
    with (
        warnings.catch_warnings(),
        patch("botorch.sampling.normal.SobolQMCNormalSampler", return_value=ref_sampler),
    ):
        warnings.simplefilter("ignore")
        actual = cls(model, observed, num_fantasies=5, maximize=maximize, sampler=own_sampler)
        reference = reference_cls(ref_model, observed, num_fantasies=5, maximize=maximize)
    torch.testing.assert_close(actual.best_f, reference.best_f, rtol=2e-10, atol=2e-11)
    query = x.new_tensor([[[0.21]], [[0.56]], [[0.82]]], requires_grad=True)
    own_distribution = actual.model.posterior(query.unsqueeze(-3))
    ref_distribution = reference.model.posterior(query.unsqueeze(-3))
    torch.testing.assert_close(own_distribution.mean, ref_distribution.mean, rtol=2e-9, atol=2e-10)
    torch.testing.assert_close(
        own_distribution.variance, ref_distribution.variance, rtol=2e-8, atol=2e-10
    )
    values = actual(query), reference(query)
    torch.testing.assert_close(*values, rtol=3e-8, atol=3e-10)
    gradients = [
        torch.autograd.grad(value.sum(), query, create_graph=True, retain_graph=True)[0]
        for value in values
    ]
    torch.testing.assert_close(*gradients, rtol=3e-7, atol=3e-9)
    seconds = [
        torch.autograd.grad(gradient.square().sum(), query, retain_graph=True)[0]
        for gradient in gradients
    ]
    torch.testing.assert_close(*seconds, rtol=3e-6, atol=3e-8)
    assert values[0].device == query.device


@pytest.mark.parametrize("device", DEVICES)
def test_noisy_analytic_observed_query_and_fantasy_parameter_graphs(device):
    x, model, _ = _models(device, False)
    observed = x[::2].clone().requires_grad_()
    acquisition = LogNoisyExpectedImprovement(
        model, observed, num_fantasies=4, sampler=FixedSampler(2, device, sample_shape=(4,))
    )
    query = x.new_tensor([[0.42]], requires_grad=True)
    with (
        patch.object(torch.Tensor, "cpu", side_effect=AssertionError("host transfer")),
        patch.object(torch.Tensor, "numpy", side_effect=AssertionError("host transfer")),
        patch.object(torch.Tensor, "tolist", side_effect=AssertionError("host transfer")),
    ):
        value = acquisition(query)
        leaves = (query, observed, *model.parameters(), *acquisition.model.parameters())
        gradients = torch.autograd.grad(value.sum(), leaves, create_graph=True, allow_unused=True)
        assert all(g is not None and torch.isfinite(g).all() for g in gradients)
        second = torch.autograd.grad(
            sum(g.square().sum() for g in gradients if g is not None), leaves, allow_unused=True
        )
        assert all(g is None or torch.isfinite(g).all() for g in second)

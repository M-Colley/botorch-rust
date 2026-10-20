"""Immutable variational laws and synchronized logical multitask noise views."""

from __future__ import annotations

import copy

import pytest

from helpers import KINDS, as_kind, assert_close, importorskip_botorch, make_data

importorskip_botorch()
torch = pytest.importorskip("torch")

from approximate_helpers import (  # noqa: E402
    CASES,
    build_pair,
    elbo_pair,
    set_random_parameters,
    strategy,
)
from botorch.models import SingleTaskVariationalGP as BtSVGP  # noqa: E402

from botorch_rs.models import SingleTaskVariationalGP  # noqa: E402

pytestmark = pytest.mark.filterwarnings("ignore")


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("outputs", [1, 2])
@pytest.mark.parametrize("materialize", [None, "mean", "variance", "covariance_matrix"])
def test_variational_output_snapshots_native_source_and_query(kind, outputs, materialize):
    x, y = make_data(n=9, d=2, seed=7)
    y = [[row[0] + 0.2 * t for t in range(outputs)] for row in y]
    model = SingleTaskVariationalGP(as_kind(x, kind), as_kind(y, kind), inducing_points=3)
    query = as_kind(x, kind)
    output = model.model(query)
    reference = copy.deepcopy(model).model(as_kind(x, kind))
    if materialize is not None:
        getattr(output, materialize)
    expected = [getattr(reference, name) for name in ("mean", "variance", "covariance_matrix")]
    model.model.mean_module.constant = 0.7
    model.model.covar_module.lengthscale = 0.9
    strategy(model)._variational_distribution.variational_mean = 0.3
    with torch.no_grad():
        query[0][0] = 0.95
    for name, want in zip(("mean", "variance", "covariance_matrix"), expected, strict=True):
        assert_close(getattr(output, name), want, rtol=0, atol=0)


@pytest.mark.parametrize("case", [CASES[3], CASES[5]], ids=lambda case: case.name)
@pytest.mark.parametrize("minibatch", [False, True])
@pytest.mark.parametrize("materialize", [None, "mean", "covariance_matrix"])
@pytest.mark.parametrize("mutation", ["native", "leaves"])
def test_elbo_from_saved_output_keeps_one_law_and_original_leaves(
    case, minibatch, materialize, mutation
):
    reference, model, _, _ = build_pair(case)
    set_random_parameters(reference, model, 17)
    expected_mll, mll = elbo_pair(reference, model, 29, beta=0.6)
    x, y = model.model.train_inputs[0].clone(), model.model.train_targets.clone()
    rx, ry = reference.model.train_inputs[0], reference.model.train_targets
    if minibatch:
        x, y, rx, ry = x[::3], y[::3], rx[::3], ry[::3]
    output, expected_output = model.model(x), reference.model(rx)
    leaves = tuple(mll.parameters())
    if materialize is not None:
        getattr(output, materialize)
    before = [getattr(expected_output, name).detach().clone() for name in ("mean", "variance")]
    if mutation == "native":
        model.model.mean_module.constant = 0.5
        kernel = model.model.covar_module
        getattr(kernel, "base_kernel", kernel).lengthscale = 0.9
    else:
        with torch.no_grad():
            for leaf in leaves:
                leaf.add_(0.03)
    x.add_(0.2)
    actual, expected = mll(output, y), expected_mll(expected_output, ry)
    torch.testing.assert_close(actual, expected, rtol=3e-8, atol=3e-9)
    ll, kl, lp = mll._terms(output, y)
    torch.testing.assert_close(actual, torch.tensor(ll - kl + lp, dtype=actual.dtype))
    # Further mutations after loss creation must not affect its native adjoint.
    y.add_(2.0)
    with torch.no_grad():
        for leaf in leaves:
            leaf.add_(0.01)
    gradients = torch.autograd.grad(actual, leaves)
    expected_gradients = torch.autograd.grad(expected, tuple(expected_mll.parameters()))
    for gradient, want in zip(gradients, expected_gradients, strict=True):
        torch.testing.assert_close(gradient, want, rtol=3e-7, atol=3e-8)
    for name, want in zip(("mean", "variance"), before, strict=True):
        torch.testing.assert_close(getattr(output, name), want, rtol=3e-8, atol=3e-9)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_multitask_logical_noise_views_sync_and_have_native_constraint_gradients(dtype):
    x = torch.tensor([[0.1], [0.4], [0.7], [0.9]], dtype=dtype)
    y = torch.cat([torch.sin(x * 3), torch.cos(x * 2)], dim=-1)
    model = SingleTaskVariationalGP(x, y, inducing_points=x[:2])
    reference = BtSVGP(x, y, inducing_points=x[:2])
    actual_lik, expected_lik = model.likelihood, reference.likelihood
    with torch.no_grad():
        for likelihood in (actual_lik, expected_lik):
            likelihood.raw_noise.fill_(0.27)
            likelihood.raw_task_noises.copy_(torch.tensor([-0.4, 0.1], dtype=dtype))
    torch.testing.assert_close(actual_lik.noise, expected_lik.noise)
    torch.testing.assert_close(actual_lik.task_noises, expected_lik.task_noises)
    weights = torch.tensor([-0.7, 1.2], dtype=dtype)
    actual = actual_lik.noise.sum() * 0.3 + (actual_lik.task_noises * weights).sum()
    expected = expected_lik.noise.sum() * 0.3 + (expected_lik.task_noises * weights).sum()
    gradients = torch.autograd.grad(actual, (actual_lik.raw_noise, actual_lik.raw_task_noises))
    wants = torch.autograd.grad(expected, (expected_lik.raw_noise, expected_lik.raw_task_noises))
    for gradient, want in zip(gradients, wants, strict=True):
        torch.testing.assert_close(gradient, want)

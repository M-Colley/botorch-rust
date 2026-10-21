"""Correlated variational task-noise laws and their full parameter graphs."""

# ruff: noqa: E402
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
gpytorch = pytest.importorskip("gpytorch")
pytest.importorskip("botorch")

from botorch.models.approximate_gp import SingleTaskVariationalGP as ReferenceGP

from botorch_rs.kernels import RBFKernel
from botorch_rs.likelihoods import MultitaskGaussianLikelihood
from botorch_rs.mlls import VariationalELBO
from botorch_rs.models.approximate_gp import SingleTaskVariationalGP

DEVICES = [
    "cpu",
    pytest.param(
        "cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
    ),
]


def _models(device, rank):
    x = torch.linspace(0.1, 0.9, 12, device=device, dtype=torch.double).reshape(6, 2)
    y = torch.stack((x.sum(-1).sin(), x.sum(-1).cos(), x.prod(-1)), -1)
    actual = SingleTaskVariationalGP(
        x,
        y,
        likelihood=MultitaskGaussianLikelihood(3, rank=rank),
        covar_module=RBFKernel(ard_num_dims=2, batch_shape=(3,)),
        inducing_points=x[[0, 2, 5]],
        backend="torch",
    )
    expected = ReferenceGP(
        x,
        y,
        likelihood=gpytorch.likelihoods.MultitaskGaussianLikelihood(3, rank=rank).to(x),
        covar_module=gpytorch.kernels.RBFKernel(ard_num_dims=2, batch_shape=(3,)).to(x),
        inducing_points=x[[0, 2, 5]],
    ).to(x)
    for model in (actual, expected):
        model.likelihood.noise = 0.12
        model.model.covar_module.lengthscale = 0.5
        with torch.no_grad():
            if rank:
                model.likelihood.task_noise_covar_factor.copy_(
                    torch.linspace(0.1, 0.4, 3 * rank, device=device, dtype=x.dtype).reshape(
                        3, rank
                    )
                )
            else:
                model.likelihood.task_noises = 0.07
            strategy = model.model.variational_strategy.base_variational_strategy
            strategy.variational_params_initialized.fill_(1)
            strategy._variational_distribution.variational_mean.fill_(0.15)
            strategy._variational_distribution.chol_variational_covar.mul_(0.8)
    return actual, expected, x, y


def _output_major(covariance, n, m):
    return covariance.reshape(n, m, n, m).permute(1, 0, 3, 2).reshape(n * m, n * m)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("rank", [0, 1, 3])
def test_multitask_noise_elbo_and_all_first_parameter_derivatives(device, rank):
    actual, expected, x, y = _models(device, rank)
    values = [
        VariationalELBO(actual.likelihood, actual.model, num_data=11)(actual.model(x), y),
        gpytorch.mlls.VariationalELBO(expected.likelihood, expected.model, num_data=11)(
            expected.model(x), y
        ),
    ]
    torch.testing.assert_close(*values, atol=2e-11, rtol=2e-10)
    named = [dict(model.named_parameters()) for model in (actual, expected)]
    assert named[0].keys() == named[1].keys()
    gradients = [
        dict(
            zip(
                parameters,
                torch.autograd.grad(value.sum(), tuple(parameters.values()), create_graph=True),
                strict=True,
            )
        )
        for value, parameters in zip(values, named, strict=True)
    ]
    for name in named[0]:
        torch.testing.assert_close(gradients[0][name], gradients[1][name], atol=2e-9, rtol=2e-8)
    noise_names = [name for name in named[0] if name.startswith("likelihood.")]
    second = [
        torch.autograd.grad(
            sum(grads[name].sum() for name in noise_names),
            [parameters[name] for name in noise_names],
        )
        for grads, parameters in zip(gradients, named, strict=True)
    ]
    for a, b in zip(*second, strict=True):
        torch.testing.assert_close(a, b, atol=2e-9, rtol=2e-8)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("rank", [0, 1, 3])
def test_noisy_variational_posterior_and_public_likelihood_keep_cross_task_noise(device, rank):
    actual, expected, x, _ = _models(device, rank)
    query = x[[1, 4]].clone().requires_grad_()
    actual.eval()
    expected.eval()
    a = actual.posterior(query, observation_noise=True)
    b = expected.posterior(query, observation_noise=True)
    torch.testing.assert_close(a.mean, b.mean, atol=2e-10, rtol=2e-9)
    torch.testing.assert_close(
        a.distribution.covariance_matrix,
        _output_major(b.distribution.covariance_matrix, 2, 3),
        atol=2e-10,
        rtol=2e-9,
    )
    if rank:
        assert bool((a.distribution.covariance_matrix[:2, 2:4].abs() > 0).any())
    target = a.mean.new_tensor([[0.1, 0.3, -0.2], [0.4, -0.3, 0.5]])
    conditionals = [model.likelihood(target) for model in (actual, expected)]
    assert conditionals[0].event_shape == conditionals[1].event_shape == (3,)
    torch.testing.assert_close(
        conditionals[0].log_prob(target + 0.2),
        conditionals[1].log_prob(target + 0.2),
        atol=2e-12,
        rtol=2e-11,
    )
    marginals = [model.likelihood(model.model(query)) for model in (actual, expected)]
    torch.testing.assert_close(
        marginals[0].covariance_matrix,
        _output_major(marginals[1].covariance_matrix, 2, 3),
        atol=2e-10,
        rtol=2e-9,
    )
    targets = [
        (query, model.likelihood.raw_noise)
        + (
            (model.likelihood.task_noise_covar_factor,)
            if rank
            else (model.likelihood.raw_task_noises,)
        )
        for model in (actual, expected)
    ]
    values = [
        a.mean.square().sum() + a.distribution.covariance_matrix.square().sum(),
        b.mean.square().sum() + b.distribution.covariance_matrix.square().sum(),
    ]
    gradients = [
        torch.autograd.grad(value, leaves, create_graph=True)
        for value, leaves in zip(values, targets, strict=True)
    ]
    for lhs, rhs in zip(*gradients, strict=True):
        torch.testing.assert_close(lhs, rhs, atol=2e-8, rtol=3e-7)
    second = torch.autograd.grad(sum(g.sum() for g in gradients[0]), targets[0])
    assert all(bool(torch.isfinite(value).all()) for value in second)

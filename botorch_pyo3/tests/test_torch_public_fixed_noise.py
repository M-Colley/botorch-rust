"""Direct fixed-noise modules preserve source Gaussian laws and data graphs."""

# ruff: noqa: E402
from __future__ import annotations

import copy
import io

import pytest

torch = pytest.importorskip("torch")

from botorch_rs.likelihoods import FixedNoiseGaussianLikelihood
from botorch_rs.models import SingleTaskGP
from botorch_rs.posteriors._torch_backend import TorchGPyTorchPosterior

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


def values(device, batch):
    noise = torch.tensor([0.03, 0.05, 0.07], dtype=torch.double, device=device)
    if batch:
        noise = torch.stack((noise, noise * 1.3))
    return noise


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("batch", [False, True])
@pytest.mark.parametrize("additional", [False, True])
def test_source_public_state_migration_copy_pickle_and_fantasy(device, batch, additional):
    gpytorch = pytest.importorskip("gpytorch")
    noise = values(device, batch)
    actual = FixedNoiseGaussianLikelihood(noise, additional, noise.shape[:-1]).to(noise)
    source = gpytorch.likelihoods.FixedNoiseGaussianLikelihood(
        noise, additional, noise.shape[:-1]
    ).to(noise)
    assert isinstance(actual, (torch.nn.Module, FixedNoiseGaussianLikelihood))
    assert actual.state_dict().keys() == source.state_dict().keys()
    for name, value in actual.state_dict().items():
        torch.testing.assert_close(value, source.state_dict()[name])
    torch.testing.assert_close(actual.noise, source.noise)
    hooks = []
    actual.register_forward_hook(lambda module, args, output: hooks.append(module))
    actual(torch.zeros_like(noise))
    assert hooks == [actual]
    actual._forward_hooks.clear()
    if additional:
        actual.second_noise = 0.2
        source.second_noise = 0.2
    for candidate in (copy.deepcopy(actual),):
        torch.testing.assert_close(candidate.noise, actual.noise)
        assert {id(p) for p in candidate.parameters()}.isdisjoint(
            id(p) for p in actual.parameters()
        )
    stream = io.BytesIO()
    torch.save(actual, stream)
    stream.seek(0)
    loaded = torch.load(stream, weights_only=False)
    torch.testing.assert_close(loaded.noise, actual.noise)
    torch.testing.assert_close(actual.float().noise, source.float().noise)
    added = noise.new_full((*noise.shape[:-1], 2), 0.04).float()
    af, sf = actual.get_fantasy_likelihood(noise=added), source.get_fantasy_likelihood(noise=added)
    torch.testing.assert_close(af.noise, sf.noise)
    assert af.noise.shape[-1] == 5


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("batch", [False, True])
@pytest.mark.parametrize("additional", [False, True])
def test_source_marginal_pointwise_laws_and_all_active_leaf_hessians(device, batch, additional):
    gpytorch = pytest.importorskip("gpytorch")
    pairs = []
    for kind in (FixedNoiseGaussianLikelihood, gpytorch.likelihoods.FixedNoiseGaussianLikelihood):
        noise = values(device, batch).requires_grad_()
        mean = (
            torch.linspace(-0.4, 0.5, 3, dtype=noise.dtype, device=device)
            .expand_as(noise)
            .clone()
            .requires_grad_()
        )
        variance = noise.new_full(noise.shape, 0.4).requires_grad_()
        model = kind(noise, additional, noise.shape[:-1]).to(noise)
        prior = gpytorch.distributions.MultivariateNormal(mean, torch.diag_embed(variance))
        observed = model(prior)
        target = torch.zeros_like(mean)
        objective = observed.mean.square().sum() + observed.covariance_matrix.square().sum()
        objective = (
            objective
            + model.expected_log_prob(target, prior).sum()
            + model.log_marginal(target, prior).sum()
        )
        leaves = (noise, mean, variance, *model.parameters())
        gradient = torch.autograd.grad(objective, leaves, create_graph=True)
        hvp = torch.autograd.grad(sum(g.sum() for g in gradient), leaves)
        pairs.append((observed, gradient, hvp))
    for left, right in zip(pairs[0][1:], pairs[1][1:], strict=True):
        for a, b in zip(left, right, strict=True):
            torch.testing.assert_close(a, b, atol=2e-9, rtol=2e-9)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("additional", [False, True])
def test_independent_explicit_noise_keeps_additional_component_and_full_graph(device, additional):
    noise = values(device, False).requires_grad_()
    override = noise.new_tensor([0.04, 0.02]).requires_grad_()
    mean = noise.new_tensor([0.2, -0.3]).requires_grad_()
    prior = TorchGPyTorchPosterior(
        mean.unsqueeze(-1), torch.eye(2, dtype=noise.dtype, device=device)
    ).distribution
    likelihood = FixedNoiseGaussianLikelihood(noise, additional)
    actual = likelihood(prior, noise=override).covariance_matrix
    expected = prior.covariance_matrix + torch.diag_embed(override + likelihood.second_noise)
    torch.testing.assert_close(actual, expected)
    assert torch.autograd.grad(actual.sum(), noise, allow_unused=True)[0] is None
    leaves = (override, *likelihood.parameters())
    a = torch.autograd.grad(actual.square().sum(), leaves, create_graph=True)
    b = torch.autograd.grad(expected.square().sum(), leaves, create_graph=True)
    for x, y in zip(a, b, strict=True):
        torch.testing.assert_close(x, y)
    ah = torch.autograd.grad(sum(x.sum() for x in a), leaves)
    bh = torch.autograd.grad(sum(x.sum() for x in b), leaves)
    for x, y in zip(ah, bh, strict=True):
        torch.testing.assert_close(x, y)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("additional", [False, True])
def test_scalar_model_adopts_actual_module_and_conditions_fixed_component_once(device, additional):
    X = torch.tensor([[0.1], [0.45], [0.9]], dtype=torch.double, device=device)
    Y = (X * 4).sin()
    noise = values(device, False).requires_grad_()
    likelihood = FixedNoiseGaussianLikelihood(noise, additional)
    model = SingleTaskGP(X, Y, likelihood=likelihood, outcome_transform=None, backend="torch")
    assert model.likelihood is likelihood
    query = X.new_tensor([[0.3], [0.7]], requires_grad=True)
    posterior = model.posterior(query)
    assert torch.isfinite(posterior.mean).all()
    gx, gn = torch.autograd.grad(posterior.mean.sum(), (query, noise), create_graph=True)
    assert torch.isfinite(torch.autograd.grad(gx.sum() + gn.sum(), (query, noise))[0]).all()
    new_noise = X.new_full((2, 1), 0.06).requires_grad_()
    conditioned = model.condition_on_observations(
        query, X.new_tensor([[0.4], [0.2]]), noise=new_noise
    )
    torch.testing.assert_close(
        conditioned.likelihood.noise_covar.noise, torch.cat((noise, new_noise.squeeze(-1)))
    )
    torch.testing.assert_close(conditioned.likelihood.second_noise, likelihood.second_noise)
    assert {id(p) for p in conditioned.likelihood.parameters()}.isdisjoint(
        id(p) for p in likelihood.parameters()
    )
    result = conditioned.posterior(X.new_tensor([[0.6]])).mean
    assert torch.isfinite(torch.autograd.grad(result.sum(), new_noise)[0]).all()

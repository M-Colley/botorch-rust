# ruff: noqa: E402
"""Higher-order Tensor GPs retain the complete correlated grid law and graph."""

import pytest

pytest.importorskip("torch")


from unittest.mock import patch

import torch

from botorch_rs.exceptions.errors import UnsupportedError
from botorch_rs.models._torch_backend_kronecker import tensor_kron
from botorch_rs.models.higher_order_gp import HigherOrderGP
from botorch_rs.sampling.normal import IIDNormalSampler

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


def data(device, dtype=torch.double, batch=False):
    X = torch.tensor([[0.1], [0.45], [0.9]], dtype=dtype, device=device)
    Y = (X * 3 + torch.arange(6, dtype=dtype, device=device)).sin().reshape(3, 2, 3)
    Q = X.new_tensor([[0.25], [0.7]])
    if batch:
        X = X.expand(2, -1, -1).clone()
        Y = Y.expand(2, -1, -1, -1).clone()
        Y[1] = Y[1] * 1.2 + 0.1
        Q = Q.expand(2, -1, -1).clone()
    return X, Y, Q


def paired(X, Y, **kwargs):
    pytest.importorskip("botorch")
    from botorch.models.higher_order_gp import HigherOrderGP as Reference

    a = HigherOrderGP(X, Y, backend="torch", **kwargs)
    b = Reference(X, Y, **kwargs).to(X)
    actual, expected = dict(a.named_parameters()), dict(b.named_parameters())
    assert actual.keys() == expected.keys()
    with torch.no_grad():
        for name in actual:
            actual[name].copy_(expected[name])
    return a, b


def reference_law(model, X, Y, Q, observation_noise=False):
    # Construct from the actual official modules. Its eval forward detaches latent
    # covariance and its public posterior reports only marginal test covariance.
    if hasattr(model, "outcome_transform"):
        model.outcome_transform.train()
        Y = model.outcome_transform(Y)[0]
        model.outcome_transform.eval()
    latent = None
    for module, coordinates in zip(model.covar_modules[1:], model.latent_parameters, strict=True):
        value = module(coordinates, coordinates).to_dense()
        latent = value if latent is None else tensor_kron(latent, value)
    kernel = model.covar_modules[0]
    ktt = tensor_kron(kernel(X, X).to_dense(), latent)
    ktq = tensor_kron(kernel(X, Q).to_dense(), latent)
    kqq = tensor_kron(kernel(Q, Q).to_dense(), latent)
    count = Y.flatten(start_dim=X.ndim - 2).shape[-1]
    noise = model.likelihood.noise
    observed = ktt + torch.diag_embed(noise.expand(*ktt.shape[:-2], count))
    alpha = torch.linalg.solve(observed, Y.flatten(start_dim=X.ndim - 2).unsqueeze(-1))
    mean = (ktq.transpose(-1, -2) @ alpha).squeeze(-1)
    covariance = kqq - ktq.transpose(-1, -2) @ torch.linalg.solve(observed, ktq)
    if observation_noise:
        covariance = covariance + torch.diag_embed(
            noise.expand(*covariance.shape[:-2], covariance.shape[-1])
        )
    if hasattr(model, "outcome_transform"):
        mean = mean * model.outcome_transform.stdvs.squeeze(
            -1
        ) + model.outcome_transform.means.squeeze(-1)
        covariance = covariance * model.outcome_transform.stdvs.square()
    return mean.reshape(*mean.shape[:-1], Q.shape[-2], *Y.shape[X.ndim - 1 :]), covariance, observed


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.double])
@pytest.mark.parametrize("batch", [False, True])
@pytest.mark.parametrize("standardize", [False, True])
def test_full_grid_law_and_exact_mll(device, dtype, batch, standardize):
    X, Y, Q = data(device, dtype, batch)
    kwargs = {} if standardize else {"outcome_transform": None}
    a, b = paired(X, Y, **kwargs)
    actual = a.posterior(Q)
    mean, covariance, observed = reference_law(b, X, Y, Q)
    tol = 7e-4 if dtype == torch.float32 else 2e-8
    torch.testing.assert_close(actual.mean, mean, atol=tol, rtol=tol)
    torch.testing.assert_close(
        actual.distribution.covariance_matrix, covariance, atol=tol, rtol=tol
    )
    torch.testing.assert_close(
        actual.variance.flatten(-3), covariance.diagonal(dim1=-2, dim2=-1), atol=tol, rtol=tol
    )
    variance = a.make_posterior_variances(actual.joint_covariance_matrix)
    if standardize:
        variance = variance * a.outcome_transform.stdvs.square().squeeze(-1)
    torch.testing.assert_close(variance, actual.variance.flatten(-3), atol=tol, rtol=tol)
    assert actual.mean.device == X.device
    assert actual.base_sample_shape == X.shape[:-2] + (48,)
    target = b.train_targets
    prior = sum(
        prior.log_prob(closure(module)).reshape(*X.shape[:-2], -1).sum(-1)
        for _, module, prior, closure, _ in b.named_priors()
    )
    value = torch.distributions.MultivariateNormal(
        torch.zeros_like(target), covariance_matrix=observed
    ).log_prob(target)
    torch.testing.assert_close(a._exact_mll(), (value + prior) / 18, atol=tol, rtol=tol)


@pytest.mark.parametrize("device", DEVICES)
def test_first_parameter_training_and_query_gradients_and_mixed_hvp(device):
    X, Y, Q = [t.requires_grad_() for t in data(device)]
    a, b = paired(X, Y, outcome_transform=None)
    actual = a.posterior(Q)
    mean, covariance, _ = reference_law(b, X, Y, Q)
    names = sorted(dict(a.named_parameters()))
    aa = (X, Y, Q, *(dict(a.named_parameters())[n] for n in names))
    bb = (X, Y, Q, *(dict(b.named_parameters())[n] for n in names))

    def loss(mean, cov):
        return mean.square().sum() * 0.3 + cov.square().sum() * 0.7

    ga = torch.autograd.grad(
        loss(actual.mean, actual.distribution.covariance_matrix), aa, create_graph=True
    )
    gb = torch.autograd.grad(loss(mean, covariance), bb)
    for left, right in zip(ga, gb, strict=True):
        torch.testing.assert_close(left, right, atol=4e-7, rtol=4e-7)
    direction = [torch.full_like(v, 0.01) for v in aa]
    second = torch.autograd.grad(sum((g * d).sum() for g, d in zip(ga, direction, strict=True)), aa)
    step = 1e-5
    values = []
    for sign in (1, -1):
        with torch.no_grad():
            for v, d in zip(aa, direction, strict=True):
                v.add_(d, alpha=sign * step)
        p = a.posterior(Q)
        values.append(torch.autograd.grad(loss(p.mean, p.distribution.covariance_matrix), aa))
        with torch.no_grad():
            for v, d in zip(aa, direction, strict=True):
                v.add_(d, alpha=-sign * step)
    for value, plus, minus in zip(second, *values, strict=True):
        torch.testing.assert_close(value, (plus - minus) / (2 * step), atol=3e-5, rtol=3e-4)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("noise", [False, True])
def test_matheron_correlated_samples_mean_covariance_and_higher_derivatives(device, noise):
    X, Y, Q = [t.requires_grad_() for t in data(device)]
    a = HigherOrderGP(X, Y, backend="torch")
    p = a.posterior(Q, observation_noise=noise)
    assert p.base_sample_shape == (48 + (12 if noise else 0),)
    shape = torch.Size([2, 3])
    normal = torch.randn(shape + p.base_sample_shape, dtype=X.dtype, device=X.device)
    with (
        patch.object(torch.Tensor, "cpu", side_effect=AssertionError("host copy")),
        patch.object(torch.Tensor, "tolist", side_effect=AssertionError("host copy")),
    ):
        sample = p.rsample_from_base_samples(shape, normal)
        inputs = (X, Y, Q, *a.parameters())
        grad = torch.autograd.grad(sample.square().sum(), inputs, create_graph=True)
        second = torch.autograd.grad(sum(g.square().sum() for g in grad), inputs)
    assert sample.shape == (2, 3, 2, 2, 3)
    assert all(torch.isfinite(v).all() for v in (*grad, *second))
    with torch.no_grad():
        sample = p.rsample(torch.Size([16000])).flatten(-3)
        torch.testing.assert_close(sample.mean(0), p.mean.flatten(-3), atol=0.04, rtol=0.05)
        covariance = (sample - sample.mean(0)).T @ (sample - sample.mean(0)) / (sample.shape[0] - 1)
        torch.testing.assert_close(
            covariance, p.distribution.covariance_matrix, atol=0.045, rtol=0.12
        )


@pytest.mark.parametrize("device", DEVICES)
def test_conditioning_fantasy_batches_and_original_transform_statistics(device):
    X, Y, Q = data(device)
    a = HigherOrderGP(X, Y, backend="torch")
    original = a.posterior(Q)
    values = original.mean.detach().expand(3, -1, -1, -1)
    updated = a.condition_on_observations(Q, values)
    assert updated.train_inputs[0].shape == (3, 5, 1)
    assert updated.posterior(Q).mean.shape == (3, 2, 2, 3)
    assert a._train_Y.shape == (3, 2, 3)
    torch.testing.assert_close(updated.outcome_transform.means, a.outcome_transform.means)
    fantasy = a.fantasize(
        Q.requires_grad_(), IIDNormalSampler(torch.Size([2]), seed=4), propagate_grads=True
    )
    loss = fantasy.posterior(Q).mean.square().sum()
    first = torch.autograd.grad(loss, Q, create_graph=True)[0]
    assert torch.isfinite(torch.autograd.grad(first.square().sum(), Q)[0]).all()


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("latent_init", ["default", "gp"])
def test_multidimensional_latents_frozen_configuration_and_gp_priors(device, latent_init):
    X, Y, Q = data(device)
    a = HigherOrderGP(
        X,
        Y,
        backend="torch",
        num_latent_dims=[2, 1],
        latent_init=latent_init,
        learn_latent_pars=False,
    )
    assert a.latent_parameters[0].shape == (2, 2)
    assert not any(v.requires_grad for v in a.latent_parameters)
    assert torch.isfinite(a._exact_mll()).all()
    a._exact_mll().sum().backward()
    assert a.covar_modules[1].raw_lengthscale.grad is not None
    assert torch.isfinite(a.posterior(Q).rsample()).all()


@pytest.mark.parametrize("device", DEVICES)
def test_explicit_unsupported_and_input_validation(device):
    X, Y, Q = data(device)
    a = HigherOrderGP(X, Y, backend="torch", covar_modules=None)
    with pytest.raises(UnsupportedError, match="overrides"):
        a.posterior(Q, output_indices=[0])
    with pytest.raises(UnsupportedError, match="configured"):
        a.condition_on_observations(Q, Y[:2], noise=Y[:2].square())
    with pytest.raises(ValueError, match="targets"):
        a.condition_on_observations(Q, Y)
    with pytest.raises(ValueError, match="num_latent_dims"):
        HigherOrderGP(X, Y, backend="torch", num_latent_dims=[1])


@pytest.mark.parametrize("device", DEVICES)
def test_standardization_input_transform_custom_kernels_and_mll_higher_derivatives(device):
    pytest.importorskip("gpytorch")
    from gpytorch.kernels import RBFKernel
    from gpytorch.likelihoods import GaussianLikelihood

    from botorch_rs.models.transforms.input import Normalize

    X, Y, Q = [t.requires_grad_() for t in data(device, batch=True)]
    kernels = [RBFKernel(ard_num_dims=1, batch_shape=torch.Size([2])).to(X) for _ in range(3)]
    likelihood = GaussianLikelihood(batch_shape=torch.Size([2])).to(X)
    model = HigherOrderGP(
        X,
        Y,
        covar_modules=kernels,
        likelihood=likelihood,
        input_transform=Normalize(1),
        backend="torch",
    )
    assert model.covar_modules[0] is kernels[0]
    assert model.likelihood is likelihood
    # Native Torch RBF supports Hessians; the external GPyTorch RBF itself may
    # dispatch to a custom backward with a narrower derivative contract.
    loss = model._exact_mll().sum() + model.posterior(Q).mean.square().sum()
    inputs = (X, Y, Q, *model.latent_parameters)
    first = torch.autograd.grad(loss, inputs, create_graph=True)
    assert all(torch.isfinite(v).all() for v in first)
    torch.testing.assert_close(
        model.outcome_transform.untransform(model.outcome_transform(Y)[0])[0], Y
    )
    assert model.outcome_transform.means.shape == (2, 1, 1)


@pytest.mark.parametrize("device", DEVICES)
def test_constant_grid_standardization_mll_and_all_parameter_hessians(device):
    X, Y, Q = data(device)
    Y = torch.ones_like(Y).requires_grad_()
    model = HigherOrderGP(X, Y, backend="torch")
    inputs = (Y, *model.parameters())
    first = torch.autograd.grad(model._exact_mll().sum(), inputs, create_graph=True)
    second = torch.autograd.grad(sum(value.square().sum() for value in first), inputs)
    assert all(torch.isfinite(value).all() for value in (*first, *second))
    assert torch.isfinite(model.posterior(Q).variance).all()


@pytest.mark.parametrize("device", DEVICES)
def test_gaussian_latent_prior_density_matches_actual_official_prior(device):
    X, Y, _ = data(device)
    a, b = paired(X, Y, latent_init="gp", outcome_transform=None)
    actual = a._log_prior()
    expected = sum(
        prior.log_prob(closure(module)).sum() for _, module, prior, closure, _ in b.named_priors()
    )
    torch.testing.assert_close(actual, expected, atol=2e-8, rtol=2e-8)

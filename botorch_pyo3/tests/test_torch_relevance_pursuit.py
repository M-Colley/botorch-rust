"""Exact sparse variances, full Tensor graphs, and genuine support search."""

import copy
from unittest.mock import patch

import pytest
import torch

from botorch_rs.exceptions.errors import UnsupportedError
from botorch_rs.exceptions.warnings import InputDataWarning
from botorch_rs.fit import fit_gpytorch_mll
from botorch_rs.likelihoods import GaussianLikelihood
from botorch_rs.mlls import ExactMarginalLogLikelihood
from botorch_rs.models import SingleTaskGP
from botorch_rs.models.likelihoods.sparse_outlier_noise import (
    SparseOutlierGaussianLikelihood,
    SparseOutlierNoise,
)
from botorch_rs.models.relevance_pursuit import (
    backward_relevance_pursuit,
    forward_relevance_pursuit,
    get_posterior_over_support,
    initialize_dense_parameters,
)
from botorch_rs.models.robust_relevance_pursuit_model import RobustRelevancePursuitSingleTaskGP
from botorch_rs.priors import NormalPrior

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


def data(device, batch=False, outputs=1, grad=False):
    X = torch.tensor([[0.05], [0.3], [0.6], [0.95]], dtype=torch.double, device=device)
    Y = torch.cat([(X * (i + 2)).sin() for i in range(outputs)], -1)
    Y[2] += 1.1
    Q = X.new_tensor([[0.2], [0.8]])
    if batch:
        X = X.expand(2, -1, -1).clone()
        Y = Y.expand(2, -1, -1).clone()
        Y[1] += 0.2
        Q = Q.expand(2, -1, -1).clone()
    return tuple(t.requires_grad_(grad) for t in (X, Y, Q))


def model_for(X, Y, convex=True, support=None, prior=False):
    likelihood = SparseOutlierGaussianLikelihood(
        GaussianLikelihood(),
        X.shape[-2],
        [2, 0] if support is None else support,
        convex_parameterization=convex,
        rho_prior=NormalPrior(0.0, 1.0) if prior else None,
    )
    model = SingleTaskGP(X, Y, likelihood=likelihood, outcome_transform=None, backend="torch")
    with torch.no_grad():
        model.likelihood.noise_covar.raw_rho.fill_(0.18)
        model.likelihood.noise_covar.base_noise.noise = X.new_tensor(0.07)
        model.covar_module.lengthscale = X.new_tensor(0.28)
        model.mean_module.constant = X.new_tensor(0.35)
    return model


def dense_oracle(model, X, Y, Q):
    """Independent dense RBF/Schur law with the same actual leaves."""
    outputs = Y.shape[-1]
    if outputs > 1:
        X = X.unsqueeze(-3).expand(*X.shape[:-2], outputs, *X.shape[-2:])
        Q = Q.unsqueeze(-3).expand(*Q.shape[:-2], outputs, *Q.shape[-2:])
        Y = Y.transpose(-1, -2)
    else:
        Y = Y.squeeze(-1)
    ls = model.covar_module.lengthscale

    def kernel(A, B):
        return (
            -0.5 * ((A.unsqueeze(-2) - B.unsqueeze(-3)) / ls.unsqueeze(-2)).square().sum(-1)
        ).exp()

    mean = model.mean_module.constant.unsqueeze(-1)
    ktt, ktq, kqq = kernel(X, X), kernel(X, Q), kernel(Q, Q)
    sparse = model.likelihood.noise_covar
    raw = sparse.raw_rho
    rho = raw.new_zeros(*raw.shape[:-1], sparse.dim).index_copy(
        -1, raw.new_tensor(sparse.support, dtype=torch.long), raw
    )
    if sparse.convex_parameterization:
        rho = rho / (1 - rho)
    base = sparse.base_noise.noise
    noise = base + rho * (1 + base) if sparse.convex_parameterization else base + rho
    observed = ktt + noise.diag_embed()
    centered = Y - mean
    alpha = torch.linalg.solve(observed, centered.unsqueeze(-1))
    posterior_mean = mean + (ktq.transpose(-1, -2) @ alpha).squeeze(-1)
    covariance = kqq - ktq.transpose(-1, -2) @ torch.linalg.solve(observed, ktq)
    mll = (
        torch.distributions.MultivariateNormal(mean.expand_as(Y), observed).log_prob(Y)
        / Y.shape[-1]
    )
    if outputs > 1:
        posterior_mean = posterior_mean.transpose(-1, -2)
        covariance = (
            torch.block_diag(*covariance.unbind(-3))
            if covariance.ndim == 3
            else torch.stack([torch.block_diag(*c.unbind(-3)) for c in covariance])
        )
    else:
        posterior_mean = posterior_mean.unsqueeze(-1)
    return posterior_mean, covariance, mll


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("convex", [True, False])
@pytest.mark.parametrize("batch", [False, True])
@pytest.mark.parametrize("outputs", [1, 2])
def test_exact_sparse_law_and_mll(device, convex, batch, outputs):
    X, Y, Q = data(device, batch, outputs)
    model = model_for(X, Y, convex)
    expected_mean, expected_cov, expected_mll = dense_oracle(model, X, Y, Q)
    actual = model.posterior(Q)
    torch.testing.assert_close(actual.mean, expected_mean)
    torch.testing.assert_close(actual.distribution.covariance_matrix, expected_cov)
    torch.testing.assert_close(
        ExactMarginalLogLikelihood(model.likelihood, model)(),
        expected_mll + model._log_prior() / X.shape[-2],
    )
    assert actual.mean.device.type == device


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("convex", [True, False])
def test_full_first_and_second_training_query_parameter_derivatives(device, convex):
    X, Y, Q = data(device, grad=True)
    model = model_for(X, Y, convex)
    variables = (X, Y, Q, *model.parameters())
    posterior = model.posterior(Q)
    actual = (
        posterior.mean.square().sum()
        + posterior.distribution.covariance_matrix.square().sum()
        + ExactMarginalLogLikelihood(model.likelihood, model)()
    )
    mean, covariance, mll = dense_oracle(model, X, Y, Q)
    expected = (
        mean.square().sum() + covariance.square().sum() + mll + model._log_prior() / X.shape[-2]
    )
    ga = torch.autograd.grad(actual, variables, create_graph=True, retain_graph=True)
    gb = torch.autograd.grad(expected, variables, create_graph=True, retain_graph=True)
    directions = [
        torch.linspace(0.1, 0.8, v.numel(), device=v.device).reshape_as(v) for v in variables
    ]
    for a, b in zip(ga, gb, strict=True):
        torch.testing.assert_close(a, b, rtol=2e-8, atol=2e-8)
    ha = torch.autograd.grad(
        sum((g * d).sum() for g, d in zip(ga, directions, strict=True)), variables
    )
    hb = torch.autograd.grad(
        sum((g * d).sum() for g, d in zip(gb, directions, strict=True)), variables
    )
    for a, b in zip(ha, hb, strict=True):
        torch.testing.assert_close(a, b, rtol=2e-7, atol=2e-7)


@pytest.mark.parametrize("device", DEVICES)
def test_official_unbatched_values_and_prior(device):
    from botorch.models import SingleTaskGP as RefGP
    from botorch.models.likelihoods.sparse_outlier_noise import (
        SparseOutlierGaussianLikelihood as RefLikelihood,
    )
    from gpytorch.likelihoods import GaussianLikelihood as RefBase
    from gpytorch.mlls import ExactMarginalLogLikelihood as RefMLL
    from gpytorch.priors import NormalPrior as RefPrior

    X, Y, Q = data(device)
    actual = model_for(X, Y, prior=True)
    likelihood = RefLikelihood(RefBase().noise_covar, len(X), [2, 0], rho_prior=RefPrior(0.0, 1.0))
    expected = RefGP(X, Y, likelihood=likelihood, outcome_transform=None).to(X)
    source, target = dict(actual.named_parameters()), dict(expected.named_parameters())
    assert source.keys() == target.keys()
    with torch.no_grad():
        for name in source:
            target[name].copy_(source[name])
    a, b = actual.posterior(Q), expected.posterior(Q)
    torch.testing.assert_close(a.mean, b.mean)
    torch.testing.assert_close(a.distribution.covariance_matrix, b.distribution.covariance_matrix)
    expected.train()
    reference = RefMLL(expected.likelihood, expected)(expected(X), Y.squeeze(-1), X)
    torch.testing.assert_close(ExactMarginalLogLikelihood(actual.likelihood, actual)(), reference)
    a, b = (
        actual.posterior(X, observation_noise=True),
        expected.posterior(X, observation_noise=True),
    )
    torch.testing.assert_close(a.variance, b.variance)
    torch.testing.assert_close(a.distribution.covariance_matrix, b.distribution.covariance_matrix)


@pytest.mark.parametrize("device", DEVICES)
def test_batched_support_atomicity_trainability_and_direct_likelihood(device):
    from gpytorch.likelihoods import GaussianLikelihood as RefBase

    base = RefBase(batch_shape=torch.Size([2])).to(device=device, dtype=torch.double).noise_covar
    sparse = SparseOutlierNoise(base, 4, [2, 0], batch_shape=(2,))
    assert isinstance(sparse, torch.nn.Module)
    sparse.raw_rho.requires_grad_(False)
    sparse.set_sparse_parameter(
        torch.tensor([[0.1, 0.2], [0.3, 0.4]], device=device, dtype=torch.double)
    )
    sparse.to_dense()
    torch.testing.assert_close(
        sparse.raw_rho, sparse.raw_rho.new_tensor([[0.2, 0, 0.1, 0], [0.4, 0, 0.3, 0]])
    )
    sparse.to_sparse().expand_support([3]).contract_support([0])
    assert sparse.support == [2, 3] and not sparse.raw_rho.requires_grad
    old = sparse.raw_rho
    for invalid in ([1, 1], [4], [True], [1.2], [2]):
        with pytest.raises(ValueError):
            sparse.expand_support(invalid)
        assert sparse.raw_rho is old and sparse.support == [2, 3]
    for method in (forward_relevance_pursuit, backward_relevance_pursuit):
        with pytest.raises(UnsupportedError, match="ranking"):
            method(sparse, None)
        assert sparse.raw_rho is old and sparse.support == [2, 3]


@pytest.mark.parametrize("device", DEVICES)
def test_query_noise_match_and_cache_clone(device):
    X, Y, Q = data(device)
    model = model_for(X, Y)
    latent = model.posterior(X).distribution.covariance_matrix
    noisy = model.posterior(X, observation_noise=True).distribution.covariance_matrix
    expected_noise = model._sparse_noise(X, Y.squeeze(-1), latent, None)
    torch.testing.assert_close(noisy - latent, expected_noise.diag_embed())
    novel = X + 0.01
    with pytest.warns(InputDataWarning, match="not applied"):
        noisy = model.posterior(novel, observation_noise=True)
    latent = model.posterior(novel)
    base = model.likelihood.noise_covar.base_noise.noise
    torch.testing.assert_close(
        noisy.distribution.covariance_matrix - latent.distribution.covariance_matrix,
        torch.diag_embed(base.expand(len(X))),
    )
    cache = model.likelihood.noise_covar._cached_train_inputs
    assert cache.data_ptr() != X.data_ptr()
    with torch.no_grad():
        X.add_(0.03)
    assert not torch.equal(cache, X)


@pytest.mark.parametrize("device", DEVICES)
def test_loo_centering_and_gradient_probe_restore(device):
    X, Y, Q = data(device)
    model = model_for(X, Y)
    noise = model.likelihood.noise_covar
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    covariance = model._covariance(X, X)
    observed = covariance + model._sparse_noise(X, Y.squeeze(-1), covariance, None).diag_embed()
    inverse = observed.inverse()
    alpha = inverse @ (Y.squeeze(-1) - model.mean_module(X))
    expected = (
        (alpha / inverse.diagonal()).square() - inverse.diagonal().reciprocal() - noise.rho
    ).clamp_min(0)[~noise.is_active]
    torch.testing.assert_close(noise._optimal_rhos(mll), expected)
    original = noise.raw_rho
    original.requires_grad_(False)
    result = noise._sparse_parameter_gradient(mll)
    assert result.device.type == device and noise.raw_rho is original
    assert not original.requires_grad and noise.is_sparse
    with patch.object(model, "_exact_mll", side_effect=RuntimeError("injected")):
        with pytest.raises(RuntimeError, match="injected"):
            noise._sparse_parameter_gradient(mll)
    assert noise.raw_rho is original and not original.requires_grad and noise.is_sparse


@pytest.mark.parametrize("device", DEVICES)
def test_actual_support_search_fit_and_model_comparison(device):
    X, Y, _ = data(device)
    model = RobustRelevancePursuitSingleTaskGP(
        X, Y, backend="torch", outcome_transform=None, cache_model_trace=True
    )
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_mll(mll, numbers_of_outliers=[1, 0], optimizer_kwargs={"options": {"maxiter": 3}})
    assert mll.model is model and mll.likelihood is model.likelihood
    assert not mll.training and not model.training
    assert len(model.model_trace) == 3
    sizes, probabilities = get_posterior_over_support(
        SparseOutlierNoise, model.model_trace, prior_mean_of_support=1.0
    )
    torch.testing.assert_close(sizes, X.new_tensor([4, 1, 0]))
    torch.testing.assert_close(probabilities, model.bmc_probabilities)
    torch.testing.assert_close(probabilities.sum(), X.new_tensor(1.0))
    assert probabilities.device.type == device
    best = model.model_trace[int(probabilities.argmax())]
    torch.testing.assert_close(model.posterior(X).mean, best.posterior(X).mean)
    original = model.likelihood
    with patch(
        "botorch_rs.models.relevance_pursuit.backward_relevance_pursuit",
        side_effect=RuntimeError("injected"),
    ):
        with pytest.raises(RuntimeError, match="injected"):
            fit_gpytorch_mll(mll, numbers_of_outliers=[0])
    assert mll.model is model and mll.likelihood is original


@pytest.mark.parametrize("device", DEVICES)
def test_public_factory_copy_and_initialize(device):
    X, Y, _ = data(device, grad=True)
    model = RobustRelevancePursuitSingleTaskGP(X, Y, outcome_transform=None)
    assert model._tensor_backend == "torch"
    assert isinstance(model, RobustRelevancePursuitSingleTaskGP)
    noise = SparseOutlierNoise._from_model(model)
    noise.expand_support([2])
    copied = copy.deepcopy(model)
    assert copied.likelihood.noise_covar.support == [2]
    assert copied.likelihood.noise_covar.raw_rho is not noise.raw_rho
    initialize_dense_parameters(model)
    assert bool(torch.isfinite(model.covar_module.lengthscale).all())
    conditioned = model.condition_on_observations(X[:1], Y[:1])
    assert conditioned.likelihood.noise_covar.dim == len(X) + 1
    assert conditioned.likelihood.noise_covar.support == [2]
    assert conditioned.likelihood.noise_covar.raw_rho is noise.raw_rho


@pytest.mark.parametrize("device", DEVICES)
def test_standalone_torch_likelihood_and_registered_base_priors(device):
    from botorch.utils.constraints import NonTransformedInterval
    from gpytorch.distributions import MultivariateNormal
    from gpytorch.likelihoods import GaussianLikelihood as RefBase
    from gpytorch.priors import NormalPrior as RefPrior

    X, Y, _ = data(device)
    base = RefBase(noise_prior=RefPrior(0.0, 1.0)).to(X).noise_covar
    rho_prior = RefPrior(0.0, 1.0).to(X)
    likelihood = SparseOutlierGaussianLikelihood(
        base,
        len(X),
        [2],
        rho_prior=rho_prior,
        rho_constraint=NonTransformedInterval(0.0, 0.95, initial_value=0.1),
    )
    assert isinstance(likelihood, torch.nn.Module)
    model = SingleTaskGP(X, Y, likelihood=likelihood, outcome_transform=None)
    assert model.likelihood is likelihood and model._tensor_backend == "torch"
    K = model._covariance(X, X)
    mean = model.mean_module(X)
    law = likelihood(MultivariateNormal(mean, K), X)
    rho = likelihood.noise_covar.rho
    expected_noise = base.noise + (K.diagonal() + base.noise) * rho
    torch.testing.assert_close(law.covariance_matrix, K + expected_noise.diag_embed())
    expected_prior = base.noise_prior.log_prob(base.noise).sum() + rho_prior.log_prob(rho).sum()
    torch.testing.assert_close(likelihood.log_prior(), expected_prior)
    gradient = torch.autograd.grad(
        likelihood.log_prior(), (base.raw_noise, likelihood.noise_covar.raw_rho)
    )
    assert all(bool(torch.isfinite(g).all()) for g in gradient)
    # Constraint-aware fit traverses both the outlier parameters and an actual
    # registered-prior GPyTorch noise module without treating it as detached.
    fit_gpytorch_mll(
        ExactMarginalLogLikelihood(likelihood, model), optimizer_kwargs={"options": {"maxiter": 1}}
    )
    assert bool((likelihood.noise_covar.raw_rho <= 0.95).all())
    assert bool((likelihood.noise_covar.raw_rho >= 0).all())


@pytest.mark.parametrize("device", DEVICES)
def test_custom_base_likelihood_identity_and_fixed_support_batches(device):
    from gpytorch.likelihoods import GaussianLikelihood as RefBase

    X, Y, Q = data(device, batch=True)
    base = RefBase(batch_shape=torch.Size([2])).to(X)
    model = RobustRelevancePursuitSingleTaskGP(
        X, Y, likelihood=base, backend="torch", outcome_transform=None
    )
    assert model.likelihood.noise_covar.base_noise is base.noise_covar
    model.likelihood.noise_covar.expand_support([2])
    output = model.posterior(Q)
    gradients = torch.autograd.grad(
        output.mean.sum(), (base.raw_noise, model.likelihood.noise_covar.raw_rho), create_graph=True
    )
    assert all(bool(torch.isfinite(g).all()) for g in gradients)
    noise = model.likelihood.noise_covar
    original = noise.raw_rho
    with pytest.raises(UnsupportedError, match="ranking"):
        fit_gpytorch_mll(ExactMarginalLogLikelihood(model.likelihood, model))
    assert noise.raw_rho is original and noise.support == [2]


@pytest.mark.parametrize("device", DEVICES)
def test_fixed_noise_transform_graphs_and_gradient_probe_dense_state(device):
    from botorch_rs.models.transforms.input import Normalize

    X, Y, Q = data(device, grad=True)
    Yvar = X.new_full(Y.shape, 0.04).requires_grad_()
    model = RobustRelevancePursuitSingleTaskGP(
        X, Y, Yvar, backend="torch", input_transform=Normalize(1)
    )
    sparse = model.likelihood.noise_covar
    sparse.expand_support([2])
    with torch.no_grad():
        sparse.raw_rho.fill_(0.12)
    posterior = model.posterior(Q)
    base = torch.tensor([[-0.4, 0.6], [0.3, -0.2]], device=device, dtype=X.dtype).reshape(2, 2, 1)
    samples = posterior.rsample_from_base_samples(torch.Size([2]), base)
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    loss = samples.square().sum() + mll()
    variables = (X, Y, Yvar, Q, sparse.raw_rho)
    gradients = torch.autograd.grad(loss, variables, create_graph=True)
    hessian = torch.autograd.grad(sum(g.square().sum() for g in gradients), variables)
    assert all(
        g.device.type == device and bool(torch.isfinite(g).all()) for g in (*gradients, *hessian)
    )
    sparse.to_dense()
    original = sparse.raw_rho
    original.requires_grad_(False)
    sparse._sparse_parameter_gradient(mll)
    assert sparse.raw_rho is original and not original.requires_grad and not sparse.is_sparse
    with pytest.warns(InputDataWarning, match="not applied"):
        noisy = model.posterior(Q, observation_noise=True)
    latent = model.posterior(Q)
    # FixedGaussianNoise at a different event count contributes zero, not a
    # CPU/float32 ZeroLinearOperator or an unrelated mean training variance.
    torch.testing.assert_close(noisy.variance, latent.variance)


@pytest.mark.parametrize("device", DEVICES)
def test_torch_optimizer_search_and_scalar_custom_support_prior(device):
    from botorch_rs.optim.fit import fit_gpytorch_mll_torch

    X, Y, _ = data(device)
    model = model_for(X, Y, support=[])
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    _, trace = forward_relevance_pursuit(
        model.likelihood.noise_covar,
        mll,
        sparsity_levels=[0, 1],
        optimizer=fit_gpytorch_mll_torch,
        optimizer_kwargs={"step_limit": 2, "stopping_criterion": None},
    )
    assert len(trace) >= 1

    def prior(size):
        assert size.ndim == 0
        return -0.3 * size

    sizes, probabilities = get_posterior_over_support(
        SparseOutlierNoise, trace, log_support_prior=prior
    )
    values = torch.stack([ExactMarginalLogLikelihood(m.likelihood, m)() for m in trace])
    torch.testing.assert_close(probabilities, (values - 0.3 * sizes).softmax(0))


@pytest.mark.parametrize("device", DEVICES)
def test_float32_graphs_are_finite(device):
    X, Y, Q = (v.float().requires_grad_() for v in data(device))
    model = model_for(X, Y)
    posterior = model.posterior(Q)
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    variables = (X, Y, Q, model.likelihood.noise_covar.raw_rho)
    gradients = torch.autograd.grad(
        posterior.mean.square().sum() + posterior.variance.sum() + mll(),
        variables,
        create_graph=True,
    )
    hessian = torch.autograd.grad(sum(g.square().sum() for g in gradients), variables)
    assert all(
        g.dtype == torch.float32 and g.device.type == device and bool(torch.isfinite(g).all())
        for g in (*gradients, *hessian)
    )

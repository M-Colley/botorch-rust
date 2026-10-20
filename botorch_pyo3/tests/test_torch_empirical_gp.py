# ruff: noqa: E402
"""Empirical Tensor covariances preserve their actual cross-output law."""

import pytest

pytest.importorskip("torch")


import torch

from botorch_rs.mlls import ExactMarginalLogLikelihood
from botorch_rs.models.empirical_gps import (
    EmpiricalOneDimensionalGP,
    EmpiricalOneDimensionalKernel,
    EmpiricalOneDimensionalMean,
    MultiOutputEmpiricalOneDimensionalGP,
    MultiOutputEmpiricalOneDimensionalKernel,
    MultiOutputEmpiricalOneDimensionalMean,
)
from botorch_rs.sampling.normal import IIDNormalSampler

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


def data(device, curves=4, outputs=2, grad=False, batch=False):
    generator = torch.Generator(device=device).manual_seed(314)
    HX = torch.linspace(0.0, 1.0, 5, device=device, dtype=torch.double).unsqueeze(-1)
    HY = torch.randn(curves, 5, outputs, generator=generator, device=device, dtype=torch.double)
    X = HX.new_tensor([[0.12], [0.47], [0.87]])
    Y = torch.randn(3, outputs, generator=generator, device=device, dtype=torch.double)
    Q = HX.new_tensor([[0.31], [0.71]])
    if batch:
        X = X.expand(3, -1, -1).clone()
        Y = Y.expand(3, -1, -1).clone()
        Y[1] += 0.2
        Q = Q.expand(3, -1, -1).clone()
    return tuple(v.requires_grad_(grad) for v in (X, Y, Q, HX, HY))


def classes(correlated, reference=False):
    if reference:
        if correlated:
            from botorch.models.empirical_gps.multioutput_empirical_1d_gp import (
                MultiOutputEmpiricalOneDimensionalGP as GP,
            )
            from botorch.models.empirical_gps.multioutput_empirical_1d_gp import (
                MultiOutputEmpiricalOneDimensionalKernel as Kernel,
            )
            from botorch.models.empirical_gps.multioutput_empirical_1d_gp import (
                MultiOutputEmpiricalOneDimensionalMean as Mean,
            )
        else:
            from botorch.models.empirical_gps.empirical_1d_gp import (
                EmpiricalOneDimensionalGP as GP,
            )
            from botorch.models.empirical_gps.empirical_1d_gp import (
                EmpiricalOneDimensionalKernel as Kernel,
            )
            from botorch.models.empirical_gps.empirical_1d_gp import (
                EmpiricalOneDimensionalMean as Mean,
            )
        return GP, Kernel, Mean
    return (
        (
            MultiOutputEmpiricalOneDimensionalGP,
            MultiOutputEmpiricalOneDimensionalKernel,
            MultiOutputEmpiricalOneDimensionalMean,
        )
        if correlated
        else (EmpiricalOneDimensionalGP, EmpiricalOneDimensionalKernel, EmpiricalOneDimensionalMean)
    )


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("correlated", [False, True])
@pytest.mark.parametrize("use_svd", [False, True])
@pytest.mark.parametrize("ard", [False, True])
def test_public_means_kernels_match_official(device, correlated, use_svd, ard):
    _, _, Q, HX, HY = data(device, curves=12)
    _, Kernel, Mean = classes(correlated)
    _, RefKernel, RefMean = classes(correlated, True)
    actual = Kernel(HX, HY, ard=ard, use_svd=use_svd, correction=1, backend="torch")
    expected = RefKernel(HX, HY, ard=ard, use_svd=use_svd, correction=1).to(HX)
    if ard:
        with torch.no_grad():
            weights = torch.linspace(
                0.2, 1.3, actual.curve_weights.numel(), device=device, dtype=HX.dtype
            )
            actual.curve_weights.copy_(weights)
            expected.curve_weights.copy_(weights)
    query = Q if correlated else Q.unsqueeze(-3).expand(2, -1, -1)
    torch.testing.assert_close(actual(query), expected.forward(query, query), rtol=2e-7, atol=2e-8)
    torch.testing.assert_close(
        actual(query, diag=True), expected.forward(query, query, diag=True), rtol=2e-7, atol=2e-8
    )
    torch.testing.assert_close(Mean(HX, HY, backend="torch")(query), RefMean(HX, HY)(query))
    assert actual(query).device.type == device


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("correlated", [False, True])
@pytest.mark.parametrize("batch", [False, True])
@pytest.mark.parametrize("fixed", [False, True])
def test_exact_models_official_values_and_public_training_distribution(
    device, correlated, batch, fixed
):
    from gpytorch.mlls import ExactMarginalLogLikelihood as RefMLL

    X, Y, Q, HX, HY = data(device, batch=batch)
    GP, _, _ = classes(correlated)
    RefGP, _, _ = classes(correlated, True)
    noise = torch.full_like(Y, 0.04) if fixed else None
    actual = GP(X, Y, HX, HY, train_Yvar=noise, ard=True, backend="torch")
    expected = RefGP(X, Y, HX, HY, train_Yvar=noise, ard=True).to(X)
    a, b = dict(actual.named_parameters()), dict(expected.named_parameters())
    assert a.keys() == b.keys()
    with torch.no_grad():
        for name in a:
            a[name].copy_(b[name])
    for observation_noise in (False, True):
        p, r = (
            actual.posterior(Q, observation_noise=observation_noise),
            expected.posterior(Q, observation_noise=observation_noise),
        )
        torch.testing.assert_close(p.mean, r.mean, rtol=2e-7, atol=2e-7)
        torch.testing.assert_close(p.variance, r.variance, rtol=2e-7, atol=2e-7)
        covariance = r.distribution.covariance_matrix
        if correlated:
            order = torch.arange(4, device=device).reshape(2, 2).T.flatten()
            covariance = covariance.index_select(-2, order).index_select(-1, order)
        torch.testing.assert_close(
            p.distribution.covariance_matrix, covariance, rtol=2e-7, atol=2e-7
        )
    actual.train()
    expected.train()
    a_mll = ExactMarginalLogLikelihood(actual.likelihood, actual)
    b_mll = RefMLL(expected.likelihood, expected)
    prior = actual(*actual.train_inputs)
    torch.testing.assert_close(a_mll(), a_mll(prior, actual.train_targets))
    torch.testing.assert_close(
        a_mll(),
        b_mll(expected(*expected.train_inputs), expected.train_targets),
        rtol=2e-7,
        atol=2e-7,
    )


def oracle(model, X, Y, Q, HX, HY, correlated):
    def interpolate(query, table):
        x = HX.squeeze(-1)
        query = query.squeeze(-1)
        indices = torch.searchsorted(x, query).clamp(1, x.numel() - 1)
        right = (query - x[indices - 1]) / (x[indices] - x[indices - 1])
        weight = torch.nn.functional.one_hot(indices, x.numel()) * right.unsqueeze(-1)
        weight = weight + torch.nn.functional.one_hot(indices - 1, x.numel()) * (
            1 - right
        ).unsqueeze(-1)
        return torch.einsum("qp,cpm->cmq", weight, table)

    centered = HY - HY.mean(0, keepdim=True)
    weights = model._base_kernel.curve_weights if correlated else model.covar_module.curve_weights
    U = interpolate(X, centered) * weights[:, None, None]
    V = interpolate(Q, centered) * weights[:, None, None]
    mt = interpolate(X, HY.mean(0, keepdim=True)).squeeze(0)
    mq = interpolate(Q, HY.mean(0, keepdim=True)).squeeze(0)
    if correlated:
        U, V = U.transpose(-1, -2).flatten(-2), V.transpose(-1, -2).flatten(-2)
        mt, mq, target = mt.T.flatten(), mq.T.flatten(), Y.flatten()
        ktt, ktq, kqq = U.T @ U, U.T @ V, V.T @ V
    else:
        U, V = U.movedim(1, 0), V.movedim(1, 0)
        target = Y.T
        ktt, ktq, kqq = U.transpose(-1, -2) @ U, U.transpose(-1, -2) @ V, V.transpose(-1, -2) @ V
    ktt, ktq, kqq = (k / HY.shape[0] for k in (ktt, ktq, kqq))
    observed = ktt + model.likelihood.noise.expand_as(target).diag_embed()
    residual = target - mt
    mean = mq + (
        ktq.transpose(-1, -2) @ torch.linalg.solve(observed, residual.unsqueeze(-1))
    ).squeeze(-1)
    covariance = kqq - ktq.transpose(-1, -2) @ torch.linalg.solve(observed, ktq)
    loss = mean.square().sum() + covariance.square().sum()
    loss = (
        loss
        + (
            torch.distributions.MultivariateNormal(mt, observed).log_prob(target)
            + model._log_prior()
        ).sum()
        / target.shape[-1]
    )
    return loss


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("correlated", [False, True])
def test_historical_training_query_weight_and_noise_mixed_hessians(device, correlated):
    X, Y, Q, HX, HY = data(device, grad=True)
    GP, _, _ = classes(correlated)
    model = GP(X, Y, HX, HY, ard=True, backend="torch")
    posterior = model.posterior(Q)
    actual = (
        posterior.mean.square().sum()
        + posterior.distribution.covariance_matrix.square().sum()
        + ExactMarginalLogLikelihood(model.likelihood, model)().sum()
    )
    expected = oracle(model, X, Y, Q, HX, HY, correlated)
    variables = (X, Y, Q, HX, HY, *model.parameters())
    ga = torch.autograd.grad(actual, variables, create_graph=True, retain_graph=True)
    gb = torch.autograd.grad(expected, variables, create_graph=True, retain_graph=True)
    for a, b in zip(ga, gb, strict=True):
        torch.testing.assert_close(a, b, rtol=1e-6, atol=1e-6)
    ha = torch.autograd.grad(sum(g.square().sum() for g in ga), variables)
    hb = torch.autograd.grad(sum(g.square().sum() for g in gb), variables)
    for a, b in zip(ha, hb, strict=True):
        torch.testing.assert_close(a, b, rtol=2e-6, atol=2e-6)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("correlated", [False, True])
def test_condition_fantasies_and_samples(device, correlated):
    X, Y, Q, HX, HY = data(device)
    GP, _, _ = classes(correlated)
    model = GP(X, Y, HX, HY, backend="torch")
    extra = Y.new_tensor([[0.4, 0.6], [0.5, 0.7]])
    conditioned = model.condition_on_observations(Q, extra)
    rebuilt = GP(torch.cat((X, Q)), torch.cat((Y, extra)), HX, HY, backend="torch")
    with torch.no_grad():
        for name, param in rebuilt.named_parameters():
            param.copy_(dict(model.named_parameters())[name])
    torch.testing.assert_close(conditioned.posterior(Q).mean, rebuilt.posterior(Q).mean)
    torch.testing.assert_close(conditioned.posterior(Q).variance, rebuilt.posterior(Q).variance)
    fantasy = model.fantasize(Q, IIDNormalSampler(torch.Size([2]), seed=31))
    assert fantasy.posterior(Q).mean.shape == (2, 2, 2)
    query = Q.clone().requires_grad_()
    posterior = model.posterior(query)
    normals = torch.randn(3, 2, 2, dtype=Q.dtype, device=device)
    samples = posterior.rsample_from_base_samples(torch.Size([3]), normals)
    gradient = torch.autograd.grad(samples.square().sum(), query, create_graph=True)[0]
    hessian = torch.autograd.grad(gradient.square().sum(), query)[0]
    assert bool(torch.isfinite(hessian).all()) and samples.device.type == device


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("correlated", [False, True])
def test_compressed_basis_historical_hessians(device, correlated):
    _, _, Q, HX, HY = data(device, curves=12, grad=True)
    _, Kernel, _ = classes(correlated)
    _, RefKernel, _ = classes(correlated, True)
    actual = Kernel(HX, HY, use_svd=True, backend="torch")
    reference = RefKernel(HX, HY, use_svd=True)
    query = Q if correlated else Q.unsqueeze(-3).expand(2, -1, -1)
    ga = torch.autograd.grad(actual(query).square().sum(), (HX, HY, Q), create_graph=True)
    gb = torch.autograd.grad(
        reference.forward(query, query).square().sum(), (HX, HY, Q), create_graph=True
    )
    for a, b in zip(ga, gb, strict=True):
        torch.testing.assert_close(a, b, rtol=1e-6, atol=1e-6)
    ha = torch.autograd.grad(sum(g.square().sum() for g in ga), (HX, HY, Q))
    hb = torch.autograd.grad(sum(g.square().sum() for g in gb), (HX, HY, Q))
    for a, b in zip(ha, hb, strict=True):
        torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("correlated", [False, True])
def test_native_descriptor_conversion_and_additive_base_kernel(device, correlated):
    from botorch_rs.fit import fit_gpytorch_mll
    from botorch_rs.kernels import RBFKernel, ScaleKernel

    X, Y, Q, HX, HY = data(device)
    GP, Kernel, Mean = classes(correlated)
    source_mean = Mean(HX, HY * 1.2 + 0.4, backend="native")
    source_kernel = Kernel(HX, HY * 0.7, ard=True, backend="native")
    actual = GP(
        X,
        Y,
        HX,
        HY,
        mean_module=source_mean,
        covar_module=source_kernel,
        base_covar_module=ScaleKernel(RBFKernel()),
        ard=True,
        backend="torch",
    )
    expected = GP(
        X,
        Y,
        HX,
        HY,
        mean_module=Mean(HX, HY * 1.2 + 0.4, backend="torch"),
        covar_module=Kernel(HX, HY * 0.7, ard=True, backend="torch"),
        base_covar_module=ScaleKernel(RBFKernel()),
        ard=True,
        backend="torch",
    )
    torch.testing.assert_close(actual.posterior(Q).mean, expected.posterior(Q).mean)
    torch.testing.assert_close(actual.posterior(Q).variance, expected.posterior(Q).variance)
    fit_gpytorch_mll(
        ExactMarginalLogLikelihood(actual.likelihood, actual),
        optimizer_kwargs={"options": {"maxiter": 2}},
    )
    assert all(bool(torch.isfinite(p).all()) for p in actual.parameters())
    if not correlated:
        subset = actual.subset_output([1])
        torch.testing.assert_close(subset.posterior(Q).mean, actual.posterior(Q).mean[..., 1:2])


@pytest.mark.parametrize("device", DEVICES)
def test_correlated_explicit_noise_broadcast_and_upstream_lazy_kernel_defect(device):
    from types import MethodType

    X, Y, Q, HX, HY = data(device)
    GP, _, _ = classes(True)
    _, RefKernel, _ = classes(True, True)
    model = GP(X, Y, HX, HY, backend="torch")
    latent = model.posterior(Q)
    for noise in (torch.tensor(0.1), torch.tensor([0.1, 0.2]), torch.tensor([[0.1], [0.2]])):
        noisy = model.posterior(Q, observation_noise=noise)
        torch.testing.assert_close(noisy.variance - latent.variance, noise.to(Q).expand(2, 2))
    broken = RefKernel(HX, HY)
    with pytest.raises(RuntimeError, match="expected shape"):
        broken(Q, Q).to_dense()
    # Forward already computes the correct correlated matrix. This instance-only
    # repair proves the missing lazy event-size metadata without modifying the
    # pinned checkout or delegating any port model computation to it.
    broken.num_outputs_per_input = MethodType(lambda self, x1, x2: self.num_outputs, broken)
    torch.testing.assert_close(broken(Q, Q).to_dense(), broken.forward(Q, Q))


@pytest.mark.parametrize("device", DEVICES)
def test_explicit_conditioning_noise_and_custom_gaussian_likelihood(device):
    from gpytorch.likelihoods import GaussianLikelihood

    X, Y, Q, HX, HY = data(device)
    source = GaussianLikelihood().to(X)
    model = MultiOutputEmpiricalOneDimensionalGP(X, Y, HX, HY, likelihood=source, backend="torch")
    extra = Y.new_tensor([[0.4, 0.6], [0.5, 0.7]])
    added_noise = Y.new_tensor([[0.08, 0.09], [0.1, 0.11]])
    conditional = model.condition_on_observations(Q, extra, noise=added_noise)
    full_X = torch.cat((X, Q))
    full_Y = torch.cat((Y, extra))
    full_noise = torch.cat((source.noise.expand_as(Y), added_noise))
    rebuilt = MultiOutputEmpiricalOneDimensionalGP(
        full_X, full_Y, HX, HY, train_Yvar=full_noise, backend="torch"
    )
    torch.testing.assert_close(conditional.posterior(Q).mean, rebuilt.posterior(Q).mean)
    torch.testing.assert_close(conditional.posterior(Q).variance, rebuilt.posterior(Q).variance)
    assert model.likelihood is source

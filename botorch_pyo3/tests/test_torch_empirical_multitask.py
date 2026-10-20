# ruff: noqa: E402
"""Paired historical curves define a genuine correlated task-indexed GP."""

import pytest

pytest.importorskip("torch")


import torch

from botorch_rs.fit import fit_gpytorch_mll
from botorch_rs.mlls import ExactMarginalLogLikelihood
from botorch_rs.models.empirical_gps import (
    MultiTaskEmpiricalOneDimensionalGP,
    MultiTaskEmpiricalOneDimensionalKernel,
    MultiTaskEmpiricalOneDimensionalMean,
)
from botorch_rs.sampling.normal import IIDNormalSampler

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


def data(device, task_feature=1, grad=False):
    generator = torch.Generator(device=device).manual_seed(621)
    H = [
        torch.linspace(0, 1, 5, dtype=torch.double, device=device).unsqueeze(-1),
        torch.linspace(0.5, 1.5, 4, dtype=torch.double, device=device).unsqueeze(-1),
    ]
    curves = [torch.randn(6, len(x), generator=generator, dtype=x.dtype, device=device) for x in H]
    X = H[0].new_tensor([[0.12, 0], [0.62, 1], [0.87, 0]])
    Y = H[0].new_tensor([[0.2], [0.7], [-0.1]])
    Q = H[0].new_tensor([[0.31, 0], [1.1, 1]])
    if task_feature == 0:
        X = X.flip(-1)
        Q = Q.flip(-1)
    for value in (X, Y, Q, *H, *curves):
        value.requires_grad_(grad)
    return X, Y, Q, H, curves


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("task_feature", [0, 1])
@pytest.mark.parametrize("fixed", [False, True])
def test_official_full_joint_posterior_mll_and_standalone_modules(device, task_feature, fixed):
    from botorch.models.empirical_gps.multitask_empirical_1d_gp import (
        MultiTaskEmpiricalOneDimensionalGP as RefGP,
    )
    from botorch.models.empirical_gps.multitask_empirical_1d_gp import (
        MultiTaskEmpiricalOneDimensionalKernel as RefKernel,
    )
    from botorch.models.empirical_gps.multitask_empirical_1d_gp import (
        MultiTaskEmpiricalOneDimensionalMean as RefMean,
    )
    from gpytorch.mlls import ExactMarginalLogLikelihood as RefMLL

    X, Y, Q, H, curves = data(device, task_feature)
    noise = torch.full_like(Y, 0.03) if fixed else None
    actual = MultiTaskEmpiricalOneDimensionalGP(
        X, Y, task_feature, H, curves, noise, backend="torch"
    )
    expected = RefGP(X, Y, task_feature, H, curves, noise).to(X)
    for query in (Q, X.new_tensor([[0.61], [0.81]])):
        for noisy in (False, True):
            a, b = (
                actual.posterior(query, observation_noise=noisy),
                expected.posterior(query, observation_noise=noisy),
            )
            torch.testing.assert_close(a.mean, b.mean, rtol=2e-7, atol=2e-7)
            torch.testing.assert_close(a.variance, b.variance, rtol=2e-7, atol=2e-7)
            torch.testing.assert_close(
                a.distribution.covariance_matrix,
                b.distribution.covariance_matrix,
                rtol=2e-7,
                atol=2e-7,
            )
    actual.train()
    expected.train()
    mll = ExactMarginalLogLikelihood(actual.likelihood, actual)
    torch.testing.assert_close(
        mll(),
        RefMLL(expected.likelihood, expected)(expected(X), Y.squeeze(-1)),
        rtol=2e-7,
        atol=2e-7,
    )
    torch.testing.assert_close(mll(), mll(actual(*actual.train_inputs), actual.train_targets))
    a = MultiTaskEmpiricalOneDimensionalMean(H, curves, task_feature, backend="torch")
    b = RefMean(H, curves, task_feature)
    torch.testing.assert_close(a(Q), b(Q))
    a = MultiTaskEmpiricalOneDimensionalKernel(
        H, curves, task_feature, 6, correction=1, backend="torch"
    )
    b = RefKernel(H, curves, task_feature, 6, correction=1)
    torch.testing.assert_close(a(Q, X), b(Q, X).to_dense())
    torch.testing.assert_close(a(Q, diag=True), b(Q, Q, diag=True))


def oracle(model, X, Y, Q, H, curves, feature):
    def basis(query):
        labels = query[:, feature].round().long()
        values = query.new_zeros(query.shape[-2], curves[0].shape[0])
        for task, (knots, table) in enumerate(zip(H, curves, strict=True)):
            selected = labels == task
            p = query[selected, 1 - feature]
            h = knots.squeeze(-1)
            index = torch.searchsorted(h, p).clamp(1, h.numel() - 1)
            alpha = (p - h[index - 1]) / (h[index] - h[index - 1])
            W = torch.nn.functional.one_hot(index, h.numel()) * alpha.unsqueeze(-1)
            W = W + torch.nn.functional.one_hot(index - 1, h.numel()) * (1 - alpha).unsqueeze(-1)
            values[selected] = W @ table.T
        return values.mean(-1), values - values.mean(-1, keepdim=True)

    mt, U = basis(X)
    mq, V = basis(Q)
    ktt, ktq, kqq = U @ U.T / 6, U @ V.T / 6, V @ V.T / 6
    observed = ktt + model.likelihood.noise.expand(len(X)).diag_embed()
    mean = mq + ktq.T @ torch.linalg.solve(observed, Y.squeeze(-1) - mt)
    covariance = kqq - ktq.T @ torch.linalg.solve(observed, ktq)
    mll = torch.distributions.MultivariateNormal(mt, observed).log_prob(Y.squeeze(-1)) / len(X)
    return mean.square().sum() + covariance.square().sum() + mll


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("feature", [0, 1])
def test_all_continuous_data_parameter_mixed_hessians_and_zero_task_derivatives(device, feature):
    X, Y, Q, H, curves = data(device, feature, grad=True)
    model = MultiTaskEmpiricalOneDimensionalGP(X, Y, feature, H, curves, backend="torch")
    posterior = model.posterior(Q)
    actual = (
        posterior.mean.square().sum()
        + posterior.distribution.covariance_matrix.square().sum()
        + ExactMarginalLogLikelihood(model.likelihood, model)()
    )
    expected = oracle(model, X, Y, Q, H, curves, feature)
    variables = (X, Y, Q, *H, *curves, *model.parameters())
    ga = torch.autograd.grad(actual, variables, create_graph=True, retain_graph=True)
    gb = torch.autograd.grad(expected, variables, create_graph=True, retain_graph=True)
    for a, b in zip(ga, gb, strict=True):
        torch.testing.assert_close(a, b, rtol=1e-6, atol=1e-6)
    assert bool((ga[0][..., feature] == 0).all()) and bool((ga[2][..., feature] == 0).all())
    ha = torch.autograd.grad(sum(g.square().sum() for g in ga), variables)
    hb = torch.autograd.grad(sum(g.square().sum() for g in gb), variables)
    for a, b in zip(ha, hb, strict=True):
        torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("device", DEVICES)
def test_missing_task_fixed_noise_fallback_conditions_fantasies_and_task_validation(device):
    X, Y, Q, H, curves = data(device)
    X = X[[0, 2]]
    Y = Y[[0, 2]]
    known = torch.full_like(Y, 0.023)
    model = MultiTaskEmpiricalOneDimensionalGP(
        X, Y, 1, H, curves, known, output_tasks=[1], backend="torch"
    )
    query = Q[1:]
    latent = model.posterior(query)
    noisy = model.posterior(query, observation_noise=True)
    torch.testing.assert_close(
        noisy.variance - latent.variance, torch.full_like(noisy.variance, 0.023)
    )
    conditioned = model.condition_on_observations(
        query, Y.new_tensor([[0.4]]), noise=Y.new_tensor([[0.04]])
    )
    rebuilt = MultiTaskEmpiricalOneDimensionalGP(
        torch.cat((X, query)),
        torch.cat((Y, Y.new_tensor([[0.4]]))),
        1,
        H,
        curves,
        torch.cat((known, Y.new_tensor([[0.04]]))),
        output_tasks=[1],
        backend="torch",
    )
    torch.testing.assert_close(conditioned.posterior(query).mean, rebuilt.posterior(query).mean)
    fantasy = model.fantasize(query, IIDNormalSampler(torch.Size([2]), seed=19))
    assert fantasy.posterior(query).mean.shape == (2, 1, 1)
    for label in (-1.0, 2.0, 0.7, float("nan")):
        with pytest.raises(ValueError, match="Task"):
            model.posterior(query.new_tensor([[0.8, label]]))
    with pytest.raises(ValueError, match="outside"):
        model.posterior(query.new_tensor([[0.2]]))


@pytest.mark.parametrize("device", DEVICES)
def test_wide_homogeneous_native_modules_additive_kernel_and_actual_fit(device):
    from botorch_rs.kernels import RBFKernel, ScaleKernel

    _, _, _, H, curves = data(device)
    X = H[0].new_tensor([[0.6], [0.7], [0.9]])
    Y = H[0].new_tensor([[0.2, 0.6], [0.4, 0.7], [0.3, 0.9]])
    model = MultiTaskEmpiricalOneDimensionalGP.from_wide_format(X, Y, H, curves, backend="torch")
    assert model._tensor_backend == "torch"
    query = X.new_tensor([[0.65], [0.85]])
    homogeneous_curves = torch.stack([curves[0], curves[0] * 0.6 + 0.4], -1)
    homogeneous = MultiTaskEmpiricalOneDimensionalGP.from_homogeneous_data(
        model._train_X, model._train_Y, 1, H[0], homogeneous_curves, backend="torch"
    )
    assert homogeneous.posterior(query).mean.shape == (2, 2)
    mean = MultiTaskEmpiricalOneDimensionalMean(H, curves, 1, backend="native")
    kernel = MultiTaskEmpiricalOneDimensionalKernel(H, curves, 1, 6, backend="native")
    custom = MultiTaskEmpiricalOneDimensionalGP(
        model._train_X,
        model._train_Y,
        1,
        mean_module=mean,
        covar_module=kernel,
        base_covar_module=ScaleKernel(RBFKernel(ard_num_dims=2)),
        backend="torch",
    )
    before = ExactMarginalLogLikelihood(custom.likelihood, custom)().detach()
    fit_gpytorch_mll(
        ExactMarginalLogLikelihood(custom.likelihood, custom),
        optimizer_kwargs={"options": {"maxiter": 3}},
    )
    after = ExactMarginalLogLikelihood(custom.likelihood, custom)()
    assert bool(after >= before) and custom.posterior(query).mean.shape == (2, 2)


@pytest.mark.parametrize("device", DEVICES)
def test_model_query_fantasy_batch_broadcast_set_data_and_live_historical_buffers(device):
    X, Y, Q, H, curves = data(device, grad=True)
    X = X.detach().expand(2, -1, -1).clone().requires_grad_()
    Y = Y.detach().expand(2, -1, -1).clone().requires_grad_()
    model = MultiTaskEmpiricalOneDimensionalGP(X, Y, 1, H, curves, backend="torch")
    batched = Q.expand(3, 1, -1, -1)
    posterior = model.posterior(batched)
    assert posterior.mean.shape == (3, 2, 2, 1)
    variables = (X, Y, Q, *H, *curves, *model.parameters())
    gradients = torch.autograd.grad(
        posterior.mean.square().sum() + posterior.variance.sum(), variables, create_graph=True
    )
    hessian = torch.autograd.grad(sum(g.square().sum() for g in gradients), variables)
    assert all(
        bool(torch.isfinite(g).all()) and g.device.type == device for g in (*gradients, *hessian)
    )
    new_targets = (Y.detach() + 0.2).requires_grad_()
    model.set_train_data(targets=new_targets)
    assert model._train_Y is new_targets
    before = model.posterior(Q).mean.detach().clone()
    with torch.no_grad():
        curves[1].add_(0.3)
    assert not torch.equal(model.posterior(Q).mean, before)
    fantasy = model.fantasize(Q, IIDNormalSampler(torch.Size([2]), seed=71))
    assert fantasy.posterior(Q).mean.shape == (2, 2, 2, 1)


def test_automatic_selection_for_variance_only_graph_and_prebuilt_tensor_modules():
    X, Y, _, H, curves = data("cpu")
    known = torch.full_like(Y, 0.03).requires_grad_()
    model = MultiTaskEmpiricalOneDimensionalGP(X, Y, 1, H, curves, train_Yvar=known)
    assert model._tensor_backend == "torch"
    gradient = torch.autograd.grad(
        ExactMarginalLogLikelihood(model.likelihood, model)(), known, create_graph=True
    )[0]
    assert bool(torch.isfinite(torch.autograd.grad(gradient.square().sum(), known)[0]).all())
    mean = MultiTaskEmpiricalOneDimensionalMean(H, curves, 1, backend="torch")
    kernel = MultiTaskEmpiricalOneDimensionalKernel(H, curves, 1, 6, backend="torch")
    model = MultiTaskEmpiricalOneDimensionalGP(X, Y, 1, mean_module=mean, covar_module=kernel)
    assert model._tensor_backend == "torch"
    from botorch_rs.models.empirical_gps import EmpiricalOneDimensionalGP

    base = EmpiricalOneDimensionalGP(X[:, :1], Y, H[0], curves[0].unsqueeze(-1), train_Yvar=known)
    assert base._tensor_backend == "torch"

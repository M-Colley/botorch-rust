"""Explicit exact-GP likelihoods retain the actual full Gaussian noise law."""

# ruff: noqa: E402
from __future__ import annotations

import copy
import math

import pytest

torch = pytest.importorskip("torch")
gpytorch = pytest.importorskip("gpytorch")
from linear_operator.operators import DenseLinearOperator, DiagLinearOperator
from test_torch_fully_bayesian_multitask import DEVICES

from botorch_rs.mlls import ExactMarginalLogLikelihood
from botorch_rs.models import SingleTaskGP


class InputNoise(gpytorch.likelihoods.noise_models.Noise):
    def __init__(self, like, batch):
        super().__init__()
        self.batch_shape = torch.Size(batch)
        self.raw_diagonal = torch.nn.Parameter(like.new_full((*batch, 1), 0.2))
        self.amplitude = torch.nn.Parameter(like.new_full((*batch, 1), 0.3))

    @property
    def noise(self):
        return self.raw_diagonal.square() + 0.03

    def forward(self, *params, shape=None, noise=None, **kwargs):
        if noise is not None:
            return DiagLinearOperator(noise)
        X = params[0][0] if isinstance(params[0], (list, tuple)) else params[0]
        feature = (X[..., 0].sin() + 0.2) * self.amplitude
        covariance = torch.diag_embed(self.noise.expand_as(feature))
        return DenseLinearOperator(covariance + feature.unsqueeze(-1) * feature.unsqueeze(-2))


class Gaussian(gpytorch.likelihoods.gaussian_likelihood._GaussianLikelihoodBase):
    @property
    def noise(self):
        return self.noise_covar.noise


def _model(device, outputs, batch):
    X = torch.tensor(
        [[0.1, 0.3], [0.3, 0.7], [0.65, 0.2], [0.9, 0.8]], dtype=torch.double, device=device
    )
    X = X.expand(*batch, *X.shape).clone().requires_grad_()
    Y = torch.stack(
        [X[..., 0].detach().sin() + i * X[..., 1].detach().square() for i in range(1, outputs + 1)],
        -1,
    ).requires_grad_()
    augmented = (*batch, *((outputs,) if outputs > 1 else ()))
    kernel = gpytorch.kernels.ScaleKernel(
        gpytorch.kernels.RBFKernel(ard_num_dims=2, batch_shape=torch.Size(augmented)),
        batch_shape=torch.Size(augmented),
    ).to(X)
    kernel.base_kernel.lengthscale = X.new_full((*augmented, 1, 2), 0.7)
    kernel.outputscale = X.new_full(augmented or (), 0.9)
    mean = gpytorch.means.ConstantMean(batch_shape=torch.Size(augmented)).to(X)
    mean.constant = X.new_full(augmented or (), 0.1)
    likelihood = Gaussian(InputNoise(X, augmented))
    # The explicit likelihood owns its noise even if constructor Yvar exists.
    model = SingleTaskGP(
        X,
        Y,
        torch.full_like(Y, 0.8),
        likelihood=likelihood,
        covar_module=kernel,
        mean_module=mean,
        outcome_transform=None,
        backend="torch",
    )
    return model, X, Y


def _prior(model, left, right):
    delta = left.unsqueeze(-2) - right.unsqueeze(-3)
    scaled = delta / model.covar_module.base_kernel.lengthscale.unsqueeze(-2)
    return model.covar_module.outputscale[..., None, None] * (-0.5 * scaled.square().sum(-1)).exp()


def _noise(model, X):
    likelihood = model.likelihood.noise_covar
    diagonal = likelihood.raw_diagonal.square() + 0.03
    feature = (X[..., 0].sin() + 0.2) * likelihood.amplitude
    return torch.diag_embed(diagonal.expand_as(feature)) + feature.unsqueeze(
        -1
    ) * feature.unsqueeze(-2)


def _oracle(model, query, observation_noise):
    train = model._augment_X(model._train_X)
    query = model._augment_X(query)
    targets = model._augment_Y(model._train_Y)
    constant = model.mean_module.constant.unsqueeze(-1)
    observed = _prior(model, train, train) + _noise(model, train)
    cross = _prior(model, train, query)
    mean = constant + (
        cross.transpose(-1, -2) @ torch.linalg.solve(observed, (targets - constant).unsqueeze(-1))
    ).squeeze(-1)
    covariance = _prior(model, query, query) - cross.transpose(-1, -2) @ torch.linalg.solve(
        observed, cross
    )
    if isinstance(observation_noise, torch.Tensor):
        covariance = covariance + torch.diag_embed(
            model._augment_Y(observation_noise).expand_as(mean)
        )
    elif observation_noise:
        covariance = covariance + _noise(model, query)
    return mean, covariance


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("outputs,batch", [(1, ()), (2, ()), (1, (2,)), (2, (2,))])
@pytest.mark.parametrize("observation_noise", [False, True, "tensor"])
def test_exact_gaussian_actual_covariance_batches_all_graphs_and_hvp(
    device, outputs, batch, observation_noise
):
    model, X, Y = _model(device, outputs, batch)
    likelihood = model.likelihood
    query = (
        X.new_tensor([[0.23, 0.51], [0.57, 0.62]]).expand(3, *batch, 2, 2).clone().requires_grad_()
    )
    if observation_noise == "tensor":
        observation_noise = X.new_full((3, *batch, 2, outputs), 0.04, requires_grad=True)
    posterior = model.posterior(query, observation_noise=observation_noise)
    mean, covariance = _oracle(model, query, observation_noise)
    expected_mean = mean.unsqueeze(-1) if outputs == 1 else mean.transpose(-1, -2)
    torch.testing.assert_close(posterior.mean, expected_mean, atol=3e-13, rtol=3e-12)
    if outputs == 1:
        torch.testing.assert_close(
            posterior.distribution.covariance_matrix, covariance, atol=3e-13, rtol=3e-12
        )
    else:
        for i in range(outputs):
            torch.testing.assert_close(
                posterior.distribution.covariance_matrix[
                    ..., i * 2 : (i + 1) * 2, i * 2 : (i + 1) * 2
                ],
                covariance[..., i, :, :],
                atol=3e-13,
                rtol=3e-12,
            )
            for j in range(outputs):
                if i != j:
                    block = posterior.distribution.covariance_matrix[
                        ..., i * 2 : (i + 1) * 2, j * 2 : (j + 1) * 2
                    ]
                    torch.testing.assert_close(block, torch.zeros_like(block), atol=0, rtol=0)
    leaves = (
        X,
        Y,
        query,
        *((observation_noise,) if isinstance(observation_noise, torch.Tensor) else ()),
        *model.parameters(),
    )

    def derivatives(m, c):
        first = torch.autograd.grad(m.sin().sum() + c.square().sum(), leaves, create_graph=True)
        return (*first, *torch.autograd.grad(sum(g.square().sum() for g in first), leaves))

    for a, e in zip(
        derivatives(posterior.mean, posterior.distribution.covariance_matrix),
        derivatives(expected_mean, covariance),
        strict=True,
    ):
        torch.testing.assert_close(a, e, atol=3e-10, rtol=3e-9)
    assert model.likelihood is likelihood and posterior.device == X.device


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("outputs", [1, 2])
def test_exact_gaussian_mll_input_protocol_values_all_graphs_and_source_first_gradients(
    device, outputs
):
    pytest.importorskip("botorch")
    from botorch.models import SingleTaskGP as Source

    model, X, Y = _model(device, outputs, ())
    likelihood, kernel, mean = (
        copy.deepcopy(value) for value in (model.likelihood, model.covar_module, model.mean_module)
    )
    source = Source(
        X, Y, likelihood=likelihood, covar_module=kernel, mean_module=mean, outcome_transform=None
    )
    model.train()
    source.train()
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    actual = mll(model(*model.train_inputs), model.train_targets, model.train_inputs)
    source_mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, source)
    expected = source_mll(source(*source.train_inputs), source.train_targets, source.train_inputs)
    torch.testing.assert_close(actual, expected, atol=3e-13, rtol=3e-12)
    train = model._augment_X(X)
    targets = model._augment_Y(Y)
    observed = _prior(model, train, train) + _noise(model, train)
    residual = targets - model.mean_module.constant.unsqueeze(-1)
    oracle = (
        -0.5
        * (residual.unsqueeze(-2) @ torch.linalg.solve(observed, residual.unsqueeze(-1)))
        .squeeze(-1)
        .squeeze(-1)
        - 0.5 * torch.linalg.slogdet(observed).logabsdet
        - 0.5 * len(X) * math.log(2 * math.pi)
    ) / len(X)
    leaves = (X, Y, *model.parameters())
    first = torch.autograd.grad(actual.sum(), leaves, create_graph=True)
    expected_first = torch.autograd.grad(oracle.sum(), leaves, create_graph=True)
    second = torch.autograd.grad(sum(g.square().sum() for g in first), leaves)
    expected_second = torch.autograd.grad(sum(g.square().sum() for g in expected_first), leaves)
    for a, e in zip((*first, *second), (*expected_first, *expected_second), strict=True):
        torch.testing.assert_close(a, e, atol=3e-10, rtol=3e-9)
    source_named = dict(source.named_parameters())
    source_first = torch.autograd.grad(
        expected.sum(), (X, Y, *(source_named[name] for name, _ in model.named_parameters()))
    )
    for a, e in zip(first, source_first, strict=True):
        torch.testing.assert_close(a, e, atol=3e-11, rtol=3e-10)


def _fixed_oracle(model, query, noisy=False):
    train = model._augment_X(model._train_X)
    query = model._augment_X(query)
    targets = model._augment_Y(model._train_Y)
    constant = model.mean_module.constant.unsqueeze(-1)
    fixed = model.likelihood.noise_covar.noise
    second = model.likelihood.second_noise
    observed = _prior(model, train, train) + torch.diag_embed(fixed + second)
    cross = _prior(model, train, query)
    mean = constant + (
        cross.transpose(-1, -2) @ torch.linalg.solve(observed, (targets - constant).unsqueeze(-1))
    ).squeeze(-1)
    covariance = _prior(model, query, query) - cross.transpose(-1, -2) @ torch.linalg.solve(
        observed, cross
    )
    if noisy:
        # The supplied GPyTorch 1.15.2 marginal forwards an explicit noise
        # kwarg to BOTH components. Honor the actual caller-owned module;
        # this independently pins that contract without silently repairing it.
        query_noise = (fixed + second).mean(-1, keepdim=True)
        if model.likelihood.second_noise_covar is not None:
            query_noise = 2 * query_noise
        covariance = covariance + torch.diag_embed(query_noise.expand_as(mean))
    return mean, covariance


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("outputs", [1, 2])
@pytest.mark.parametrize("additional", [False, True])
def test_supplied_fixed_likelihood_conditions_actual_noise_and_preserves_all_graphs(
    device, outputs, additional
):
    model, X, Y = _model(device, outputs, ())
    raw_noise = X.new_full(((outputs,) if outputs > 1 else ()) + (4,), 0.04, requires_grad=True)
    likelihood = gpytorch.likelihoods.FixedNoiseGaussianLikelihood(
        noise=raw_noise,
        learn_additional_noise=additional,
        batch_shape=torch.Size((outputs,) if outputs > 1 else ()),
    ).to(X)
    model = SingleTaskGP(
        X,
        Y,
        likelihood=likelihood,
        covar_module=model.covar_module,
        mean_module=model.mean_module,
        outcome_transform=None,
        backend="torch",
    )
    extra_X = X.new_tensor([[0.45, 0.57]], requires_grad=True)
    extra_Y = Y.new_full((1, outputs), 0.6, requires_grad=True)
    extra_noise = Y.new_full((1, outputs), 0.03, requires_grad=True)
    conditioned = model.condition_on_observations(extra_X, extra_Y, noise=extra_noise)
    assert model.likelihood is likelihood and conditioned.likelihood is not likelihood
    torch.testing.assert_close(likelihood.noise_covar.noise, raw_noise, atol=0, rtol=0)
    torch.testing.assert_close(
        conditioned.likelihood.noise_covar.noise,
        torch.cat([raw_noise, model._augment_Y(extra_noise)], -1),
        atol=0,
        rtol=0,
    )
    if additional:
        assert (
            conditioned.likelihood.second_noise_covar.raw_noise
            is not likelihood.second_noise_covar.raw_noise
        )
    query = X.new_tensor([[0.22, 0.45], [0.56, 0.63]], requires_grad=True)
    posterior = conditioned.posterior(query, observation_noise=True)
    mean, covariance = _fixed_oracle(conditioned, query, True)
    expected_mean = mean.unsqueeze(-1) if outputs == 1 else mean.transpose(-1, -2)
    torch.testing.assert_close(posterior.mean, expected_mean, atol=3e-13, rtol=3e-12)
    if outputs == 1:
        torch.testing.assert_close(
            posterior.distribution.covariance_matrix, covariance, atol=3e-13, rtol=3e-12
        )
    else:
        for i in range(outputs):
            torch.testing.assert_close(
                posterior.distribution.covariance_matrix[
                    ..., i * 2 : (i + 1) * 2, i * 2 : (i + 1) * 2
                ],
                covariance[..., i, :, :],
                atol=3e-13,
                rtol=3e-12,
            )
    leaves = (X, Y, raw_noise, extra_X, extra_Y, extra_noise, query, *conditioned.parameters())

    def derivatives(m, c):
        first = torch.autograd.grad(m.sin().sum() + c.square().sum(), leaves, create_graph=True)
        return (
            *first,
            *torch.autograd.grad(sum(g.square().sum() for g in first), leaves, retain_graph=True),
        )

    for a, e in zip(
        derivatives(posterior.mean, posterior.distribution.covariance_matrix),
        derivatives(expected_mean, covariance),
        strict=True,
    ):
        torch.testing.assert_close(a, e, atol=3e-10, rtol=3e-9)


@pytest.mark.parametrize("device", DEVICES)
def test_supplied_gaussian_prediction_mll_and_gradients_remain_on_device(device, monkeypatch):
    model, X, Y = _model(device, 2, ())

    def reject(*args, **kwargs):
        raise AssertionError("unexpected host Tensor conversion")

    with monkeypatch.context() as patch:
        for name in ("cpu", "numpy", "tolist"):
            patch.setattr(torch.Tensor, name, reject)
        query = X.new_tensor([[0.23, 0.45], [0.57, 0.63]], requires_grad=True)
        posterior = model.posterior(query, observation_noise=True)
        value = posterior.mean.sum() + posterior.variance.sum() + model._exact_mll().sum()
        targets = (X, Y, query, *model.parameters())
        first = torch.autograd.grad(value, targets, create_graph=True)
        second = torch.autograd.grad(sum(g.square().sum() for g in first), targets)
    assert all(bool(torch.isfinite(value).all()) for value in (*first, *second))


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("outputs", [1, 2])
def test_actual_correlated_gaussian_posterior_matches_source_values_and_first_gradients(
    device, outputs
):
    pytest.importorskip("botorch")
    from botorch.models import SingleTaskGP as Source
    from botorch.settings import propagate_grads

    model, X, Y = _model(device, outputs, ())
    likelihood, kernel, mean = (
        copy.deepcopy(value) for value in (model.likelihood, model.covar_module, model.mean_module)
    )
    source = Source(
        X, Y, likelihood=likelihood, covar_module=kernel, mean_module=mean, outcome_transform=None
    )
    query = X.new_tensor(
        [[[0.23, 0.45], [0.57, 0.63]], [[0.32, 0.62], [0.71, 0.4]]], requires_grad=True
    )
    actual = model.posterior(query, observation_noise=True)
    with propagate_grads(True), gpytorch.settings.detach_test_caches(False):
        expected = source.posterior(query, observation_noise=True)
        source_parameters = dict(source.named_parameters())
        source_first = torch.autograd.grad(
            expected.mean.sin().sum() + expected.distribution.covariance_matrix.square().sum(),
            (X, Y, query, *(source_parameters[name] for name, _ in model.named_parameters())),
        )
    torch.testing.assert_close(actual.mean, expected.mean, atol=3e-13, rtol=3e-12)
    torch.testing.assert_close(
        actual.distribution.covariance_matrix,
        expected.distribution.covariance_matrix,
        atol=3e-13,
        rtol=3e-12,
    )
    first = torch.autograd.grad(
        actual.mean.sin().sum() + actual.distribution.covariance_matrix.square().sum(),
        (X, Y, query, *model.parameters()),
    )
    for a, e in zip(first, source_first, strict=True):
        torch.testing.assert_close(a, e, atol=3e-11, rtol=3e-10)


@pytest.mark.parametrize("device", DEVICES)
def test_supplied_fixed_likelihood_fantasies_append_its_actual_noise(device):
    from botorch_rs.sampling.normal import IIDNormalSampler

    original, X, Y = _model(device, 1, ())
    fixed = X.new_tensor([0.02, 0.04, 0.03, 0.05], requires_grad=True)
    likelihood = gpytorch.likelihoods.FixedNoiseGaussianLikelihood(noise=fixed)
    model = SingleTaskGP(
        X,
        Y,
        likelihood=likelihood,
        covar_module=original.covar_module,
        mean_module=original.mean_module,
        outcome_transform=None,
        backend="torch",
    )
    extra = X.new_tensor([[0.4, 0.53]], requires_grad=True)
    fantasy = model.fantasize(
        extra, IIDNormalSampler(torch.Size([2]), seed=4), propagate_grads=True
    )
    assert fantasy._train_X.shape == (2, 5, 2) and fixed.shape == (4,)
    expected_noise = torch.cat([fixed.expand(2, 4), fixed.mean().expand(2, 1)], -1)
    torch.testing.assert_close(fantasy.likelihood.noise, expected_noise, atol=0, rtol=0)
    query = X.new_tensor([[0.23, 0.45], [0.57, 0.63]], requires_grad=True)
    posterior = fantasy.posterior(query)
    first = torch.autograd.grad(
        posterior.mean.sin().sum() + posterior.variance.sum(),
        (X, Y, fixed, extra, query, *fantasy.parameters()),
        create_graph=True,
    )
    second = torch.autograd.grad(
        sum(g.square().sum() for g in first), (X, Y, fixed, extra, query, *fantasy.parameters())
    )
    assert all(bool(torch.isfinite(value).all()) for value in (*first, *second))


@pytest.mark.parametrize("device", DEVICES)
def test_unmodified_source_copies_additional_fixed_noise_parameters(device):
    source_gp = pytest.importorskip("botorch.models.gp_regression").SingleTaskGP
    model, X, Y = _model(device, 1, ())
    likelihood = gpytorch.likelihoods.FixedNoiseGaussianLikelihood(
        noise=X.new_full((4,), 0.04), learn_additional_noise=True
    ).to(X)
    source = source_gp(
        X,
        Y,
        likelihood=likelihood,
        covar_module=model.covar_module,
        mean_module=model.mean_module,
        outcome_transform=None,
    )
    extra_X = X.new_tensor([[0.45, 0.57]])
    source.posterior(extra_X)
    original = likelihood.second_noise_covar.raw_noise
    conditioned = source.condition_on_observations(
        extra_X, Y.new_full((1, 1), 0.6), noise=Y.new_full((1, 1), 0.03)
    )
    copied = conditioned.likelihood.second_noise_covar.raw_noise
    assert copied is not original and copied.data_ptr() != original.data_ptr()
    torch.testing.assert_close(copied, original)
    before = original.detach().clone()
    with torch.no_grad():
        copied.add_(0.2)
    torch.testing.assert_close(original, before)

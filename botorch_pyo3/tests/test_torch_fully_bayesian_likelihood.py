"""Caller-defined correlated Gaussian noise is evaluated as an actual marginal."""

# ruff: noqa: E402
from __future__ import annotations

import copy

import pytest

torch = pytest.importorskip("torch")
gpytorch = pytest.importorskip("gpytorch")
from linear_operator.operators import DenseLinearOperator, DiagLinearOperator
from test_torch_fully_bayesian_loader import DEVICES, _model

from botorch_rs.models.fully_bayesian import FullyBayesianSingleTaskGP
from botorch_rs.posteriors._torch_backend import TorchGPyTorchPosterior


class ForwardGP(FullyBayesianSingleTaskGP):
    def forward(self, X):
        prior = super().forward(X)
        mean = prior.mean + 0.1 * X[..., 0].square()
        covariance = prior.covariance_matrix * self.prior_scale.square()
        return TorchGPyTorchPosterior(mean.unsqueeze(-1), covariance).distribution


class InputNoise(gpytorch.likelihoods.noise_models.Noise):
    def __init__(self, like):
        super().__init__()
        self.raw_diagonal = torch.nn.Parameter(like.new_tensor([[0.1], [0.3]]))
        self.amplitude = torch.nn.Parameter(like.new_tensor([[0.2], [0.35]]))
        self.register_buffer("failure", like.new_zeros(()))

    @property
    def noise(self):
        return self.raw_diagonal.square() + 0.03

    def covariance(self, X):
        feature = (X[..., 0].sin() + 0.2) * self.amplitude
        result = torch.diag_embed(self.noise.expand(*feature.shape[:-1], feature.shape[-1]))
        return result + feature.unsqueeze(-1) * feature.unsqueeze(-2) + self.failure

    def forward(self, *params, shape=None, noise=None, **kwargs):
        if noise is not None:
            return DiagLinearOperator(noise)
        inputs = params[0][0] if isinstance(params[0], (tuple, list)) else params[0]
        return DenseLinearOperator(self.covariance(inputs))


class CorrelatedLikelihood(gpytorch.likelihoods.gaussian_likelihood._GaussianLikelihoodBase):
    @property
    def noise(self):
        # Having a public noise vector does not make the actual Gaussian
        # marginal diagonal: noise_covar also contributes a correlated term.
        return self.noise_covar.noise


def _load(device):
    model, X, Y, pyro = _model(device)
    likelihood = CorrelatedLikelihood(InputNoise(X))
    pyro.likelihood = likelihood
    model.load_mcmc_samples({"arbitrary_site": X.new_tensor([1.0, 2.0])})
    return model, X, Y, pyro


def _oracle(model, X, Y, query, observation_noise):
    pyro = model.pyro_model
    train, test = pyro.transform(X), pyro.transform(query.unsqueeze(-3))

    def prior_mean(X):
        return pyro.mean(X) + (0.1 * X[..., 0].square() if hasattr(model, "prior_scale") else 0)

    def prior_covariance(X, Y):
        return pyro.kernel(X, Y) * (
            model.prior_scale.square() if hasattr(model, "prior_scale") else 1
        )

    A = prior_covariance(train, train) + pyro.likelihood.noise_covar.covariance(train)
    cross = prior_covariance(train, test)
    mean = prior_mean(test) + (
        cross.transpose(-1, -2)
        @ torch.linalg.solve(A, (Y.squeeze(-1) - prior_mean(train)).unsqueeze(-1))
    ).squeeze(-1)
    covariance = prior_covariance(test, test) - cross.transpose(-1, -2) @ torch.linalg.solve(
        A, cross
    )
    if isinstance(observation_noise, torch.Tensor):
        covariance = covariance + torch.diag_embed(observation_noise.squeeze(-1).expand(2, 2))
    elif observation_noise:
        covariance = covariance + pyro.likelihood.noise_covar.covariance(test)
    return mean.unsqueeze(-1), covariance


@pytest.mark.parametrize("device", DEVICES)
def test_correlated_likelihood_keeps_custom_virtual_prior_and_its_parameter_graph(device):
    _, X, Y, pyro = _load(device)
    model = ForwardGP(X, Y, backend="torch")
    model.prior_scale = torch.nn.Parameter(X.new_tensor(0.7))
    model.pyro_model = pyro
    model.load_mcmc_samples({"arbitrary_site": X.new_tensor([1.0, 2.0])})
    query = X.new_tensor([[0.24], [0.63]], requires_grad=True)
    actual = model.posterior(query, observation_noise=True)
    mean, covariance = _oracle(model, X, Y, query, True)
    torch.testing.assert_close(actual.mean, mean, atol=3e-14, rtol=2e-13)
    torch.testing.assert_close(
        actual.distribution.covariance_matrix, covariance, atol=3e-14, rtol=2e-13
    )
    targets = (X, Y, query, *model.parameters())

    def derivatives(m, c):
        first = torch.autograd.grad(m.sin().sum() + c.square().sum(), targets, create_graph=True)
        second = torch.autograd.grad(sum(g.square().sum() for g in first), targets)
        return (*first, *second)

    for actual_grad, expected_grad in zip(
        derivatives(actual.mean, actual.distribution.covariance_matrix),
        derivatives(mean, covariance),
        strict=True,
    ):
        torch.testing.assert_close(actual_grad, expected_grad, atol=3e-11, rtol=3e-10)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("observation_noise", [False, True, "tensor"])
def test_correlated_likelihood_preserves_actual_covariance_all_graphs_and_mixed_hvp(
    device, observation_noise
):
    model, X, Y, pyro = _load(device)
    assert model.likelihood is pyro.likelihood
    query = X.new_tensor([[0.24], [0.63]], requires_grad=True)
    if observation_noise == "tensor":
        observation_noise = X.new_tensor([[0.08], [0.05]], requires_grad=True)
    actual = model.posterior(query, observation_noise=observation_noise)
    mean, covariance = _oracle(model, X, Y, query, observation_noise)
    torch.testing.assert_close(actual.mean, mean, atol=3e-14, rtol=2e-13)
    torch.testing.assert_close(
        actual.distribution.covariance_matrix, covariance, atol=3e-14, rtol=2e-13
    )
    train, test = pyro.transform(X), pyro.transform(query.unsqueeze(-3))
    diagonal_only = pyro.kernel(train, train) + torch.diag_embed(
        pyro.likelihood.noise.expand(2, len(X))
    )
    wrong_mean = pyro.mean(test) + (
        pyro.kernel(train, test).transpose(-1, -2)
        @ torch.linalg.solve(diagonal_only, (Y.squeeze(-1) - pyro.mean(train)).unsqueeze(-1))
    ).squeeze(-1)
    assert (wrong_mean - mean.squeeze(-1)).abs().max() > 1e-3
    targets = (X, Y, query, *model.parameters())
    if isinstance(observation_noise, torch.Tensor):
        targets += (observation_noise,)

    def derivatives(m, c):
        loss = m.sin().sum() + 0.3 * c.square().sum()
        first = torch.autograd.grad(loss, targets, create_graph=True)
        second = torch.autograd.grad(sum(g.square().sum() for g in first), targets)
        return (*first, *second)

    for actual_grad, oracle_grad in zip(
        derivatives(actual.mean, actual.distribution.covariance_matrix),
        derivatives(mean, covariance),
        strict=True,
    ):
        torch.testing.assert_close(actual_grad, oracle_grad, atol=3e-11, rtol=3e-10)
    assert actual.mean.device == X.device


@pytest.mark.parametrize("device", DEVICES)
def test_correlated_likelihood_matches_source_exact_gp_and_keeps_device_resident(
    device, monkeypatch
):
    model, X, Y, pyro = _load(device)

    class Reference(gpytorch.models.ExactGP):
        def forward(self, X):
            return gpytorch.distributions.MultivariateNormal(
                pyro.mean(X), DenseLinearOperator(pyro.kernel(X, X))
            )

    reference = Reference(pyro.transform(X), Y.squeeze(-1), pyro.likelihood).eval()
    query = X.new_tensor([[0.24], [0.63]], requires_grad=True)
    targets = (X, Y, query, *model.parameters())
    with gpytorch.settings.detach_test_caches(False):
        source = reference.likelihood(reference(pyro.transform(query)), pyro.transform(query))
        expected_mean = source.mean.unsqueeze(-1)
        expected_cov = source.covariance_matrix
        source_grad = torch.autograd.grad(
            expected_mean.sum() + expected_cov.square().sum(), targets
        )

    def reject(*args, **kwargs):
        raise AssertionError("unexpected host conversion")

    with monkeypatch.context() as patch:
        for name in ("cpu", "numpy", "tolist"):
            patch.setattr(torch.Tensor, name, reject)
        actual = model.posterior(query, observation_noise=True)
        actual_grad = torch.autograd.grad(
            actual.mean.sum() + actual.distribution.covariance_matrix.square().sum(),
            targets,
            create_graph=True,
        )
        second = torch.autograd.grad(sum(g.square().sum() for g in actual_grad), targets)
    torch.testing.assert_close(actual.mean, expected_mean, atol=3e-13, rtol=3e-12)
    torch.testing.assert_close(
        actual.distribution.covariance_matrix, expected_cov, atol=3e-13, rtol=3e-12
    )
    for actual_value, source_value in zip(actual_grad, source_grad, strict=True):
        torch.testing.assert_close(actual_value, source_value, atol=3e-11, rtol=3e-10)
    assert all(bool(torch.isfinite(value).all()) for value in second)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("mixed_scale", [False, True])
def test_invalid_correlated_noise_is_rejected_without_mutating_usable_model(device, mixed_scale):
    model, X, _, pyro = _load(device)
    expected = model.posterior(X[:2]).mean.detach().clone()
    state = copy.deepcopy(model.state_dict())
    original_likelihood = model.likelihood
    replacement = CorrelatedLikelihood(InputNoise(X))
    if mixed_scale:
        with torch.no_grad():
            replacement.noise_covar.raw_diagonal[0].fill_(1e8)
        replacement.noise_covar.failure = X.new_tensor([0.0, -10.0]).reshape(2, 1, 1)
    else:
        replacement.noise_covar.failure.fill_(-10)
    pyro.likelihood = replacement
    with pytest.raises(ValueError, match="noise covariance must be finite and PSD"):
        model.load_mcmc_samples({"arbitrary_site": X.new_tensor([1.0, 2.0])})
    assert model.likelihood is original_likelihood
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, state[name], atol=0, rtol=0)
    torch.testing.assert_close(model.posterior(X[:2]).mean, expected, atol=0, rtol=0)

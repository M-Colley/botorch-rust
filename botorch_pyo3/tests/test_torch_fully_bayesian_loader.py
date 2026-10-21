"""Actual caller-returned fully Bayesian modules and transactional loading."""

# ruff: noqa: E402
from __future__ import annotations

import copy

import pytest

torch = pytest.importorskip("torch")
from torch import nn

from botorch_rs.exceptions.errors import UnsupportedError
from botorch_rs.likelihoods import GaussianLikelihood
from botorch_rs.models._torch_backend_modules import build_likelihood
from botorch_rs.models.fully_bayesian import FullyBayesianSingleTaskGP, MaternPyroModel

DEVICES = [
    "cpu",
    pytest.param(
        "cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
    ),
]


class CustomMean(nn.Module):
    def __init__(self, like):
        super().__init__()
        self.raw_constant = nn.Parameter(like.new_tensor([0.1, -0.2]))
        self.slope = nn.Parameter(like.new_tensor([0.3, 0.4]))

    def forward(self, X):
        return self.raw_constant[..., None] + self.slope[..., None] * X[..., 0].sin()


class CustomKernel(nn.Module):
    def __init__(self, like):
        super().__init__()
        self.raw_scale = nn.Parameter(like.new_tensor([0.7, 1.1]))
        self.raw_length = nn.Parameter(like.new_tensor([0.6, 0.9]))

    def forward(self, X, Y):
        delta = X.unsqueeze(-2) - Y.unsqueeze(-3)
        distance = delta.square().sum(-1) / self.raw_length[..., None, None].square()
        return self.raw_scale[..., None, None].square() * (-0.5 * distance).exp()


class CustomInput(nn.Module):
    def __init__(self, like):
        super().__init__()
        self.scale = nn.Parameter(like.new_tensor(0.8))
        self.register_buffer("offset", like.new_tensor(0.02))

    def forward(self, X):
        return self.scale * X + self.offset

    def preprocess_transform(self, X):
        return self(X)


class ModuleLoader(MaternPyroModel):
    def __init__(self, like):
        super().__init__()
        self.mean = CustomMean(like)
        self.kernel = CustomKernel(like)
        self.likelihood = build_likelihood(GaussianLikelihood(batch_shape=(2,)), None, (2,), like)
        self.likelihood.noise = like.new_tensor([[0.04], [0.07]])
        self.transform = CustomInput(like)
        self.bad_transform = BadInput(like)

    def load_mcmc_samples(self, *, mcmc_samples):
        self.load_calls = getattr(self, "load_calls", 0) + 1
        if getattr(self, "fail", False):
            return self.mean, self.kernel, self.likelihood, self.bad_transform
        return self.mean, self.kernel, self.likelihood, self.transform


class BadInput(CustomInput):
    def forward(self, X):
        self.offset.add_(1)
        return super().forward(X) * float("nan")


def _model(device):
    X = torch.tensor(
        [[0.1], [0.35], [0.65], [0.9]], dtype=torch.double, device=device, requires_grad=True
    )
    Y = torch.tensor(
        [[0.2], [0.7], [-0.1], [-0.4]], dtype=torch.double, device=device, requires_grad=True
    )
    model = FullyBayesianSingleTaskGP(X, Y, backend="torch")
    pyro = ModuleLoader(X)
    model.pyro_model = pyro
    model.load_mcmc_samples({"arbitrary_site": X.new_tensor([1.0, 2.0])})
    return model, X, Y, pyro


def _oracle(model, X, Y, query, observation_noise=False):
    t = model.pyro_model.transform
    mean = model.pyro_model.mean
    kernel = model.pyro_model.kernel
    train, query = t(X), t(query.unsqueeze(-3))
    noise = model.likelihood.noise
    A = kernel(train, train) + torch.diag_embed(noise.expand(2, len(X)))
    cross = kernel(train, query)
    solved = torch.linalg.solve(A, (Y.squeeze(-1) - mean(train)).unsqueeze(-1))
    m = mean(query) + (cross.transpose(-1, -2) @ solved).squeeze(-1)
    covariance = kernel(query, query) - cross.transpose(-1, -2) @ torch.linalg.solve(A, cross)
    if observation_noise:
        covariance = covariance + torch.diag_embed(noise.expand(2, query.shape[-2]))
    return m.unsqueeze(-1), covariance


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("observation_noise", [False, True])
def test_returned_modules_keep_identity_values_all_graphs_and_mixed_hvp(device, observation_noise):
    model, X, Y, pyro = _model(device)
    assert model.mean_module is pyro.mean and model.covar_module is pyro.kernel
    assert model.likelihood is pyro.likelihood and model.input_transform["sample"] is pyro.transform
    assert model.pyro_model is pyro and pyro.load_calls == 1
    query = X.new_tensor([[[0.22], [0.53]], [[0.4], [0.75]]], requires_grad=True)
    posterior = model.posterior(query, observation_noise=observation_noise)
    expected_mean, expected_cov = _oracle(model, X, Y, query, observation_noise)
    torch.testing.assert_close(posterior.mean, expected_mean, atol=3e-14, rtol=2e-13)
    torch.testing.assert_close(
        posterior.distribution.covariance_matrix, expected_cov, atol=3e-14, rtol=2e-13
    )
    targets = (X, Y, query, *model.parameters())

    def derivatives(mean, covariance):
        loss = mean.sin().sum() + covariance.square().sum()
        gradient = torch.autograd.grad(loss, targets, create_graph=True)
        hvp = torch.autograd.grad(sum(g.square().sum() for g in gradient), targets)
        return (*gradient, *hvp)

    actual = derivatives(posterior.mean, posterior.distribution.covariance_matrix)
    expected = derivatives(expected_mean, expected_cov)
    for a, e in zip(actual, expected, strict=True):
        torch.testing.assert_close(a, e, atol=2e-11, rtol=2e-10)
    assert all(value.device == X.device for value in (posterior.mean, *actual))


@pytest.mark.parametrize("device", DEVICES)
def test_failed_loading_preserves_model_and_caller_module_state(device):
    model, X, _, pyro = _model(device)
    expected = model.posterior(X[:2]).mean.detach().clone()
    originals = (model.mean_module, model.covar_module, model.likelihood, model.input_transform)
    state = copy.deepcopy(model.state_dict())
    flags = [module.training for module in model.modules()]
    bad_offset = pyro.bad_transform.offset.clone()
    pyro.fail = True
    with pytest.raises((ValueError, RuntimeError), match="finite|positive-definite|cholesky"):
        model.load_mcmc_samples({"arbitrary_site": X.new_tensor([1.0, 2.0])})
    assert originals == (
        model.mean_module,
        model.covar_module,
        model.likelihood,
        model.input_transform,
    )
    assert pyro.load_calls == 1
    torch.testing.assert_close(pyro.bad_transform.offset, bad_offset, atol=0, rtol=0)
    assert flags == [module.training for module in model.modules()]
    for name, tensor in model.state_dict().items():
        torch.testing.assert_close(tensor, state[name], atol=0, rtol=0)
    torch.testing.assert_close(model.posterior(X[:2]).mean, expected, atol=0, rtol=0)


def test_arbitrary_noise_attribute_does_not_silently_become_gaussian():
    model, X, _, pyro = _model("cpu")
    pyro.likelihood = CustomInput(X)
    with pytest.raises(UnsupportedError, match="Gaussian marginal"):
        model.load_mcmc_samples({"arbitrary_site": X.new_tensor([1.0, 2.0])})


@pytest.mark.parametrize("device", DEVICES)
def test_direct_custom_module_loading_and_prediction_do_not_copy_tensor_data_to_host(
    device, monkeypatch
):
    model, X, _, _ = _model(device)

    def reject(*args, **kwargs):
        raise AssertionError("unexpected host data conversion")

    with monkeypatch.context() as patch:
        for name in ("cpu", "numpy", "tolist"):
            patch.setattr(torch.Tensor, name, reject)
        model.load_mcmc_samples({"arbitrary_site": X.new_tensor([1.0, 2.0])})
        query = X.new_tensor([[0.25], [0.5]], requires_grad=True)
        posterior = model.posterior(query)
        (first,) = torch.autograd.grad(posterior.mean.sum(), query, create_graph=True)
        (second,) = torch.autograd.grad(first.sum(), query)
    assert bool(torch.isfinite(second).all())


@pytest.mark.parametrize("device", DEVICES)
def test_real_custom_numpyro_fit_adopts_actual_modules_and_training_graphs(device):
    numpyro = pytest.importorskip("numpyro")
    dist = pytest.importorskip("numpyro.distributions")
    jnp = pytest.importorskip("jax.numpy")
    from fully_bayesian_helpers import jax_x64

    from botorch_rs.fit import fit_fully_bayesian_model_nuts

    class Program(ModuleLoader):
        def sample_mean(self):
            return numpyro.sample("mean", dist.Normal(jnp.array(0.2), jnp.array(0.4)))

    model, X, Y, _ = _model(device)
    program = Program(X)
    model.pyro_model = program
    with jax_x64():
        fit_fully_bayesian_model_nuts(
            model,
            warmup_steps=1,
            num_samples=2,
            thinning=1,
            max_tree_depth=1,
            seed=14,
            disable_progbar=True,
        )
    assert model.pyro_model is program and program.load_calls == 1
    assert model.mean_module is program.mean and model.covar_module is program.kernel
    assert (
        model.likelihood is program.likelihood
        and model.input_transform["sample"] is program.transform
    )
    query = X.new_tensor([[0.25], [0.5]], requires_grad=True)
    posterior = model.posterior(query)
    targets = (X, Y, query, *model.parameters())
    first = torch.autograd.grad(
        posterior.mean.sin().sum() + posterior.variance.sum(), targets, create_graph=True
    )
    second = torch.autograd.grad(sum(g.square().sum() for g in first), targets)
    assert all(bool(torch.isfinite(g).all()) for g in (*first, *second))
    assert posterior.mean.device == X.device
    assert model._numpyro_result["backend"] == "numpyro"


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("fixed", [False, True])
@pytest.mark.parametrize("q", [2, 4])
def test_explicit_gpytorch_gaussian_likelihood_owns_noise_and_keeps_graphs(device, fixed, q):
    gpytorch = pytest.importorskip("gpytorch")
    model, X, Y, pyro = _model(device)
    observation_noise = X.new_tensor(
        [[0.04, 0.06, 0.05, 0.03], [0.07, 0.03, 0.06, 0.08]], requires_grad=True
    )
    if fixed:
        likelihood = gpytorch.likelihoods.FixedNoiseGaussianLikelihood(
            noise=observation_noise, batch_shape=torch.Size([2])
        )
    else:
        likelihood = gpytorch.likelihoods.GaussianLikelihood(batch_shape=torch.Size([2])).to(X)
        likelihood.noise = observation_noise[:, :1].detach()
    # A returned likelihood replaces constructor-supplied fixed noise rather
    # than being accepted as decoration around the old observation law.
    model._train_Yvar = torch.full_like(Y, 0.8)
    pyro.likelihood = likelihood
    model.load_mcmc_samples({"arbitrary_site": X.new_tensor([1.0, 2.0])})
    assert model.likelihood is likelihood
    query = (
        torch.linspace(0.2, 0.7, q, dtype=X.dtype, device=X.device).unsqueeze(-1).requires_grad_()
    )
    actual = model.posterior(query, observation_noise=True)
    train, test = pyro.transform(X), pyro.transform(query.unsqueeze(-3))
    diagonal = likelihood.noise.expand(2, len(X))
    A = pyro.kernel(train, train) + torch.diag_embed(diagonal)
    K = pyro.kernel(train, test)
    expected_mean = pyro.mean(test) + (
        K.transpose(-1, -2)
        @ torch.linalg.solve(A, (Y.squeeze(-1) - pyro.mean(train)).unsqueeze(-1))
    ).squeeze(-1)
    expected_cov = pyro.kernel(test, test) - K.transpose(-1, -2) @ torch.linalg.solve(A, K)
    expected_cov = expected_cov + torch.diag_embed(diagonal.mean(-1, keepdim=True).expand(2, q))
    torch.testing.assert_close(actual.mean.squeeze(-1), expected_mean, atol=3e-14, rtol=2e-13)
    torch.testing.assert_close(
        actual.distribution.covariance_matrix, expected_cov, atol=3e-14, rtol=2e-13
    )
    targets = (X, Y, query, *((observation_noise,) if fixed else (likelihood.raw_noise,)))
    first = torch.autograd.grad(
        actual.mean.sum() + actual.variance.sum(), targets, create_graph=True
    )
    second = torch.autograd.grad(sum(g.square().sum() for g in first), targets)
    assert all(bool(torch.isfinite(g).all()) for g in (*first, *second))


@pytest.mark.parametrize("device", DEVICES)
def test_public_module_build_hooks_receive_source_keywords_and_state_load_uses_dummy_hook(device):
    class Hook(MaternPyroModel):
        def _build_mean_module(self, mcmc_samples, batch_shape, *, dtype, device):
            assert dtype == self.train_X.dtype and device == self.train_X.device
            return super()._build_mean_module(mcmc_samples, batch_shape, dtype=dtype, device=device)

        def _get_covar_module(self, use_scale_kernel, batch_shape, *, dtype, device):
            assert dtype == self.train_X.dtype and device == self.train_X.device
            return super()._get_covar_module(use_scale_kernel, batch_shape, dtype, device)

        def get_dummy_mcmc_samples(self, num_mcmc_samples, **kwargs):
            self.dummy_calls = getattr(self, "dummy_calls", 0) + 1
            return super().get_dummy_mcmc_samples(num_mcmc_samples, **kwargs)

    X = torch.tensor([[0.1], [0.4], [0.8]], dtype=torch.double, device=device)
    model = FullyBayesianSingleTaskGP(X, X.sin(), backend="torch")
    model.pyro_model = Hook()
    model.load_mcmc_samples(
        {
            "mean": X.new_tensor([0.1, -0.2]),
            "lengthscale": X.new_tensor([[0.4], [0.8]]),
            "noise": X.new_tensor([0.03, 0.06]),
        }
    )
    expected = model.posterior(X[:2]).mean.detach().clone()
    state = copy.deepcopy(model.state_dict())
    model.load_state_dict(state)
    assert model.pyro_model.dummy_calls == 1
    torch.testing.assert_close(model.posterior(X[:2]).mean, expected, atol=0, rtol=0)

"""Actual device-resident prior draws, model setters and shared aliases."""

# ruff: noqa: E402
from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")
gpytorch = pytest.importorskip("gpytorch")

from botorch_rs.constraints import GreaterThan
from botorch_rs.exceptions.warnings import BotorchWarning
from botorch_rs.kernels import RBFKernel, ScaleKernel
from botorch_rs.likelihoods import GaussianLikelihood
from botorch_rs.means import ConstantMean
from botorch_rs.models._torch_backend_modules import TorchPrior
from botorch_rs.models.approximate_gp import SingleTaskVariationalGP
from botorch_rs.models.gp_regression import SingleTaskGP
from botorch_rs.models.model_list_gp_regression import ModelListGP
from botorch_rs.models.utils.priors import BetaPrior
from botorch_rs.optim.utils.model_utils import sample_all_priors
from botorch_rs.priors import (
    GammaPrior,
    HalfCauchyPrior,
    HalfNormalPrior,
    LogNormalPrior,
    NormalPrior,
    SmoothedBoxPrior,
    UniformPrior,
)

DEVICES = [
    "cpu",
    pytest.param(
        "cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
    ),
]
KINDS = [
    "Normal",
    "LogNormal",
    "Gamma",
    "Uniform",
    "HalfCauchy",
    "HalfNormal",
    "Beta",
    "SmoothedBox",
]


def _source(kind):
    return {
        "Normal": lambda: NormalPrior(0.2, 0.7),
        "LogNormal": lambda: LogNormalPrior(-0.3, 0.4),
        "Gamma": lambda: GammaPrior(2.0, 3.0),
        "Uniform": lambda: UniformPrior(-0.2, 0.8),
        "HalfCauchy": lambda: HalfCauchyPrior(0.7),
        "HalfNormal": lambda: HalfNormalPrior(0.8),
        "Beta": lambda: BetaPrior(2.0, 3.0),
        "SmoothedBox": lambda: SmoothedBoxPrior(-0.4, 0.6, sigma=0.15),
    }[kind]()


def _seed(seed, device):
    torch.random.default_generator.manual_seed(seed)
    if device == "cuda":
        torch.cuda.manual_seed(seed)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("kind", KINDS)
def test_prior_sampling_matches_independent_torch_distribution_on_device(device, dtype, kind):
    like = torch.empty(0, dtype=dtype, device=device)
    prior = TorchPrior(_source(kind), like)
    p = prior.hyperparameters.unbind()
    factories = {
        "Normal": lambda: torch.distributions.Normal(p[0], p[1]),
        "LogNormal": lambda: torch.distributions.LogNormal(p[0], p[1]),
        "Gamma": lambda: torch.distributions.Gamma(p[0], p[1]),
        "Uniform": lambda: torch.distributions.Uniform(p[0], p[1]),
        "HalfCauchy": lambda: torch.distributions.HalfCauchy(p[0]),
        "HalfNormal": lambda: torch.distributions.HalfNormal(p[0]),
        "Beta": lambda: torch.distributions.Beta(p[0], p[1]),
        "SmoothedBox": lambda: gpytorch.priors.SmoothedBoxPrior(p[0], p[1], sigma=p[2]),
    }
    _seed(28, device)
    actual = prior.sample(torch.Size([2, 4]))
    _seed(28, device)
    expected = factories[kind]().sample(torch.Size([2, 4])).reshape(2, 4)
    torch.testing.assert_close(actual, expected)
    assert actual.shape == (2, 4)
    assert actual.device == like.device and actual.dtype == dtype
    assert not actual.requires_grad
    assert prior.sample((0, 2)).shape == (0, 2)


@pytest.mark.parametrize("device", DEVICES)
def test_smoothed_box_sampling_has_correct_middle_and_gaussian_tail_law(device):
    prior = TorchPrior(_source("SmoothedBox"), torch.empty(0, dtype=torch.double, device=device))
    _seed(91, device)
    sample = prior.sample((100000,))
    a, b, sigma = prior.hyperparameters
    width = b - a
    tail_weight = math.sqrt(2 * math.pi) * sigma / (width + math.sqrt(2 * math.pi) * sigma)
    observed = ((sample < a) | (sample > b)).double().mean()
    assert (
        abs(observed - tail_weight) < 6 * (tail_weight * (1 - tail_weight) / sample.numel()).sqrt()
    )
    center = (a + b) / 2
    variance = (1 - tail_weight) * width.square() / 12 + tail_weight * (
        width.square() / 4 + width * sigma * math.sqrt(2 / math.pi) + sigma.square()
    )
    assert abs(sample.mean() - center) < 6 * (variance / sample.numel()).sqrt()
    torch.testing.assert_close(sample.var(), variance, atol=0.006, rtol=0.025)


def _model(device, variational=False, kernel=None):
    x = torch.linspace(0.1, 0.9, 12, dtype=torch.double, device=device).reshape(6, 2)
    y = x.sin().sum(-1, keepdim=True)
    source = kernel or ScaleKernel(
        RBFKernel(ard_num_dims=2, lengthscale_prior=GammaPrior(2.0, 3.0)),
        outputscale_prior=GammaPrior(2.0, 1.0),
    )
    kwargs = dict(
        covar_module=source,
        mean_module=ConstantMean(constant_prior=NormalPrior(0.0, 1.0)),
        likelihood=GaussianLikelihood(noise_prior=GammaPrior(2.0, 5.0)),
        backend="torch",
    )
    if variational:
        return SingleTaskVariationalGP(x, y, inducing_points=x[:3], **kwargs)
    return SingleTaskGP(x, y, outcome_transform=None, **kwargs)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("variational", [False, True])
def test_sample_all_priors_preserves_leaf_identity_flags_graphs_and_seeded_rng(device, variational):
    model = _model(device, variational)
    held = dict(model.named_parameters())
    frozen = next(iter(held.values()))
    frozen.requires_grad_(False)
    old = model.posterior(model._train_X[:2])
    saved = old.mean.clone()
    original = {n: p.detach().clone() for n, p in held.items()}
    cpu_rng = torch.get_rng_state().clone()
    cuda_rng = torch.cuda.get_rng_state().clone() if device == "cuda" else None
    sample_all_priors(model, seed=17)
    sampled = {n: p.detach().clone() for n, p in held.items()}
    assert any(not torch.equal(sampled[n], original[n]) for n in held)
    assert all(p is dict(model.named_parameters())[name] for name, p in held.items())
    assert not frozen.requires_grad
    assert torch.equal(cpu_rng, torch.get_rng_state())
    if cuda_rng is not None:
        assert torch.equal(cuda_rng, torch.cuda.get_rng_state())
    torch.testing.assert_close(old.mean, saved, atol=0, rtol=0)
    sample_all_priors(model, seed=17)
    for name, p in held.items():
        torch.testing.assert_close(p, sampled[name], atol=0, rtol=0)
        if "variational_distribution" in name or "inducing_points" in name:
            torch.testing.assert_close(p, original[name], atol=0, rtol=0)
    value = model.posterior(model._train_X[:2]).mean.sum()
    active = [p for p in model.parameters() if p.requires_grad]
    derivatives = torch.autograd.grad(value, active, allow_unused=True)
    assert all(g is None or torch.isfinite(g).all() for g in derivatives)


@pytest.mark.parametrize("device", DEVICES)
def test_shared_kernel_and_repeated_model_aliases_sample_each_prior_once(device, monkeypatch):
    model = _model(device)
    second = _model(device, kernel=model.covar_module)
    models = ModelListGP(model, second, model)
    prior = model.covar_module.base_kernel.lengthscale_prior
    calls = []
    sample = prior.sample

    def count(shape):
        calls.append(tuple(shape))
        return sample(shape)

    monkeypatch.setattr(prior, "sample", count)
    sample_all_priors(models, seed=8)
    assert calls == [(1, 2)]
    assert model.covar_module is second.covar_module


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("batch", [False, True])
def test_registered_gpytorch_prior_batches_and_setting_closures(device, batch):
    like = torch.empty(0, device=device, dtype=torch.double)
    concentration = like.new_tensor([2.0, 3.0] if batch else [2.0])
    kernel = gpytorch.kernels.RBFKernel(
        ard_num_dims=2,
        lengthscale_prior=gpytorch.priors.GammaPrior(concentration, concentration + 1),
    ).to(like)
    model = _model(device, kernel=kernel)
    held = kernel.raw_lengthscale
    sample_all_priors(model, seed=18)
    assert kernel.raw_lengthscale is held
    assert kernel.lengthscale.shape == (1, 2)
    assert torch.isfinite(kernel.lengthscale).all()
    assert kernel.lengthscale[0, 0] != kernel.lengthscale[0, 1]


@pytest.mark.parametrize("device", DEVICES)
def test_foreign_transformed_prior_samples_are_adapted_to_actual_parameter_device_and_dtype(device):
    like = torch.empty(0, dtype=torch.double, device=device)
    # GPyTorch's Module.to moves buffered loc/scale while the transformed
    # distribution's cached base_dist can still sample on CPU/float32.
    prior = gpytorch.priors.LogNormalPrior(0.0, 0.5)
    kernel = gpytorch.kernels.RBFKernel(ard_num_dims=2, lengthscale_prior=prior).to(like)
    model = _model(device, kernel=kernel)
    sample_all_priors(model, seed=32)
    assert kernel.lengthscale.device == like.device
    assert kernel.lengthscale.dtype == like.dtype
    assert torch.isfinite(kernel.lengthscale).all()


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("identity", [False, True])
def test_prior_constraint_retry_restores_failed_values_and_honors_attempt_limit(
    device, identity, monkeypatch
):
    kwargs = {"transform": None} if identity else {}
    source = RBFKernel(
        ard_num_dims=2,
        lengthscale_prior=NormalPrior(0.0, 1.0),
        lengthscale_constraint=GreaterThan(0.5, **kwargs),
    )
    model = _model(device, kernel=source)
    kernel = model.covar_module
    initial = kernel.raw_lengthscale.detach().clone()
    prior = kernel.lengthscale_prior
    calls = []

    def draw(shape):
        calls.append(1)
        return initial.new_full(shape, 0.1 if len(calls) == 1 else 1.0)

    monkeypatch.setattr(prior, "sample", draw)
    with pytest.raises(RuntimeError, match="after 1 attempts"):
        sample_all_priors(model, max_retries=1, seed=10)
    torch.testing.assert_close(kernel.raw_lengthscale, initial, atol=0, rtol=0)
    calls.clear()
    sample_all_priors(model, max_retries=2, seed=10)
    assert len(calls) == 2
    torch.testing.assert_close(kernel.lengthscale, torch.ones_like(kernel.lengthscale))
    with pytest.raises(ValueError, match="positive integer"):
        sample_all_priors(model, max_retries=0)


def test_foreign_missing_inverse_unsupported_sampling_and_unrelated_errors(monkeypatch):
    kernel = gpytorch.kernels.RBFKernel(ard_num_dims=2)
    prior = gpytorch.priors.NormalPrior(0.0, 1.0)
    kernel.register_prior("custom", prior, lambda module: module.lengthscale)
    model = _model("cpu", kernel=kernel)
    with pytest.raises(RuntimeError, match="inverse transform"):
        sample_all_priors(model)
    kernel._priors["custom"] = (
        prior,
        lambda module: module.lengthscale,
        lambda module, value: setattr(module, "lengthscale", value),
    )

    def unsupported(shape):
        raise NotImplementedError

    monkeypatch.setattr(prior, "sample", unsupported)
    with pytest.warns(BotorchWarning, match="Sampling not implemented"):
        sample_all_priors(model)

    def failed(shape):
        raise RuntimeError("sampler backend failure")

    monkeypatch.setattr(prior, "sample", failed)
    with pytest.raises(RuntimeError, match="sampler backend failure"):
        sample_all_priors(model)

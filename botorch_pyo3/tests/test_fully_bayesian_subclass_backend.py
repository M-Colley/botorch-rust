"""Explicit fully Bayesian subclasses preserve extension hooks on device."""

# ruff: noqa: E402
from __future__ import annotations

import copy
import pickle

import pytest

torch = pytest.importorskip("torch")

from botorch_rs.fit import fit_fully_bayesian_model_nuts
from botorch_rs.models.fully_bayesian import (
    AbstractFullyBayesianSingleTaskGP,
    FullyBayesianLinearSingleTaskGP,
    FullyBayesianSingleTaskGP,
    LinearPyroModel,
    MaternPyroModel,
    SaasFullyBayesianSingleTaskGP,
    SaasPyroModel,
)

DEVICES = [
    "cpu",
    pytest.param(
        "cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
    ),
]


class CustomProgram(SaasPyroModel):
    def sample_mean(self):
        import numpyro
        import numpyro.distributions as dist

        return numpyro.sample("mean", dist.Normal(0.7, 0.4))


class CustomSaasGP(SaasFullyBayesianSingleTaskGP):
    _pyro_model_class = CustomProgram

    def __init__(self, *args, label="retained", **kwargs):
        super().__init__(*args, **kwargs)
        self.custom_label = label

    def posterior(self, X, **kwargs):
        self.custom_posterior_called = True
        return super().posterior(X, **kwargs)


class CustomMaternGP(FullyBayesianSingleTaskGP):
    _pyro_model_class = MaternPyroModel


class CustomLinearGP(FullyBayesianLinearSingleTaskGP):
    _pyro_model_class = LinearPyroModel


class DirectAbstractLinearGP(AbstractFullyBayesianSingleTaskGP):
    _pyro_model_class = LinearPyroModel


def _data(device):
    x = torch.tensor(
        [[0.1, 0.3], [0.3, 0.7], [0.7, 0.4], [0.9, 0.8]], dtype=torch.double, device=device
    )
    y = torch.sin(x[:, :1] * 3)
    return x, y, torch.full_like(y, 0.03)


def _samples(like, linear):
    samples = {"mean": like.new_tensor([-0.1, 0.3, 0.5])}
    samples["weight_variance" if linear else "lengthscale"] = like.new_tensor(
        [[0.4, 0.7], [0.6, 0.8], [0.9, 0.5]]
    )
    if not linear:
        samples["outputscale"] = like.new_tensor([0.8, 1.0, 1.3])
    return samples


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize(
    "cls", [CustomSaasGP, CustomMaternGP, CustomLinearGP, DirectAbstractLinearGP]
)
def test_subclass_explicit_tensor_backend_preserves_class_hooks_state_and_graphs(device, cls):
    x, y, noise = (value.requires_grad_() for value in _data(device))
    model = cls(x, y, noise, backend="torch")
    assert isinstance(model, cls) and isinstance(model, torch.nn.Module)
    assert model._tensor_backend == "torch"
    assert type(model.pyro_model) is cls._pyro_model_class
    samples = _samples(x, model.pyro_model._rs_kernel == "linear")
    model.load_mcmc_samples(samples)
    query = x.new_tensor([[0.2, 0.6], [0.6, 0.4]], requires_grad=True)
    posterior = model.posterior(query)
    if cls is CustomSaasGP:
        assert model.custom_label == "retained" and model.custom_posterior_called
    for other in (copy.deepcopy(model), pickle.loads(pickle.dumps(model))):
        assert isinstance(other, cls) and type(other) is type(model)
        torch.testing.assert_close(other.posterior(query).mean, posterior.mean, atol=0, rtol=0)
    reloaded = cls(x.clone(), y.clone(), noise.clone(), backend="torch")
    reloaded.load_state_dict(model.state_dict())
    torch.testing.assert_close(reloaded.posterior(query).mean, posterior.mean, atol=0, rtol=0)
    targets = (x, y, noise, query, *model.parameters())
    value = posterior.mean.square().sum() + posterior.variance.square().sum()
    first = torch.autograd.grad(value, targets, create_graph=True)
    second = torch.autograd.grad(sum(g.square().sum() for g in first), targets)
    assert all(bool(g.isfinite().all()) for g in (*first, *second))


@pytest.mark.parametrize("device", DEVICES)
def test_subclass_automatic_backend_matches_base_data_rules(device):
    x, y, noise = _data(device)
    if device == "cpu":
        native = CustomMaternGP(x, y, noise)
        assert getattr(native, "_tensor_backend", None) is None
        x.requires_grad_()
    model = CustomMaternGP(x, y, noise)
    assert model._tensor_backend == "torch" and isinstance(model, CustomMaternGP)


@pytest.mark.parametrize("device", DEVICES)
def test_custom_class_public_fitter_uses_actual_numpyro_program(device):
    pytest.importorskip("jax")
    pytest.importorskip("numpyro")
    x, y, noise = _data(device)
    model = CustomSaasGP(x, y, noise, backend="torch", label="custom fit")
    program = model.pyro_model
    fit_fully_bayesian_model_nuts(
        model,
        warmup_steps=1,
        num_samples=2,
        thinning=1,
        max_tree_depth=1,
        seed=19,
        disable_progbar=True,
    )
    assert model.pyro_model is program and model.custom_label == "custom fit"
    assert "mean" in model._numpyro_result["raw_samples"]
    assert model.num_mcmc_samples == 2
    query = x.new_tensor([[0.2, 0.6]], requires_grad=True)
    first = torch.autograd.grad(model.posterior(query).mean.sum(), query, create_graph=True)[0]
    second = torch.autograd.grad(first.sum(), query)[0]
    assert bool(second.isfinite().all())

"""Actual inference for caller-defined probabilistic model extensions."""

# ruff: noqa: E402
from __future__ import annotations

import copy

import pytest

torch = pytest.importorskip("torch")
jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")
numpyro = pytest.importorskip("numpyro")
dist = pytest.importorskip("numpyro.distributions")
from fully_bayesian_helpers import jax_x64
from numpyro.infer import MCMC, NUTS

from botorch_rs.fit import fit_fully_bayesian_model_nuts
from botorch_rs.models._numpyro_fitting import requires_numpyro_inference
from botorch_rs.models.fully_bayesian import SaasFullyBayesianSingleTaskGP, SaasPyroModel


class CustomSaas(SaasPyroModel):
    def sample_mean(self):
        return numpyro.sample("mean", dist.Normal(jnp.array(0.4), jnp.array(0.3)))

    def sample(self):
        numpyro.sample("auxiliary", dist.Normal(jnp.array(0.1), jnp.array(0.7)))
        super().sample()

    def postprocess_mcmc_samples(self, mcmc_samples):
        self.postprocess_calls = getattr(self, "postprocess_calls", 0) + 1
        if getattr(self, "fail_postprocess", False):
            raise RuntimeError("caller postprocess failed")
        return super().postprocess_mcmc_samples(mcmc_samples)


class CustomLoadedSaas(CustomSaas):
    def load_mcmc_samples(self, mcmc_samples):
        self.load_calls = getattr(self, "load_calls", 0) + 1
        result = super().load_mcmc_samples(mcmc_samples)
        if getattr(self, "fail_load", False):
            raise RuntimeError("caller loader failed")
        return result


class CustomGP(SaasFullyBayesianSingleTaskGP):
    _pyro_model_class = CustomLoadedSaas


def _data(device="cpu", gradients=False):
    x = torch.tensor([[0.1], [0.3], [0.6], [0.8]], dtype=torch.double, device=device)
    x.requires_grad_(gradients)
    y = (3 * x).sin()
    return x, y, torch.full_like(y, 0.03, requires_grad=gradients)


@pytest.mark.parametrize("backend", ["native", "torch"])
def test_custom_program_matches_direct_numpyro_seed_sites_postprocess_and_loader(backend):
    x, y, noise = _data()
    model = (
        CustomGP(x, y, noise)
        if backend == "native"
        else SaasFullyBayesianSingleTaskGP(x, y, noise, backend="torch")
    )
    if backend == "torch":
        model.pyro_model = CustomSaas()
    pyro = model.pyro_model
    assert requires_numpyro_inference(model)
    control = copy.copy(pyro)
    control.set_inputs(x, y, noise)
    with jax_x64():
        reference = MCMC(
            NUTS(control.sample, dense_mass=True, max_tree_depth=2),
            num_warmup=3,
            num_samples=4,
            progress_bar=False,
        )
        reference.run(jax.random.PRNGKey(71))
        fit_fully_bayesian_model_nuts(
            model,
            max_tree_depth=2,
            warmup_steps=3,
            num_samples=4,
            thinning=2,
            seed=71,
            disable_progbar=True,
            jit_compile=True,
        )
    assert model.pyro_model is pyro
    assert pyro.postprocess_calls == 1
    if backend == "native":
        assert pyro.load_calls == 1
        assert model.covar_module._model is model
        assert model.mean_module._model is model
    assert model.num_mcmc_samples == 2
    actual = model._numpyro_result["raw_samples"]
    assert "auxiliary" in actual
    assert actual.keys() == reference.get_samples().keys()
    for name, expected in reference.get_samples().items():
        assert bool(jnp.array_equal(actual[name], expected))
    assert model._numpyro_result["jax_enable_x64"]
    assert model._numpyro_result["jax_devices"] == sorted(
        {str(device) for value in actual.values() for device in value.devices()}
    )
    assert model.posterior(x[:2]).mean.shape == (2, 2, 1)


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable"),
        ),
    ],
)
def test_custom_fitted_tensor_law_keeps_data_query_parameter_graphs(device):
    x, y, noise = _data(device, gradients=True)
    model = SaasFullyBayesianSingleTaskGP(x, y, noise, backend="torch")
    model.pyro_model = CustomSaas()
    before = torch.random.get_rng_state()
    with jax_x64():
        fit_fully_bayesian_model_nuts(
            model,
            warmup_steps=2,
            num_samples=2,
            thinning=1,
            max_tree_depth=2,
            seed=14,
            disable_progbar=True,
        )
    assert torch.equal(before, torch.random.get_rng_state())
    query = x.new_tensor([[0.2], [0.5]], requires_grad=True)
    posterior = model.posterior(query)
    loss = posterior.mean.square().sum() + posterior.distribution.covariance_matrix.square().sum()
    targets = (x, noise, query, *model.parameters())
    gradient = torch.autograd.grad(loss, targets, create_graph=True)
    hvp = torch.autograd.grad(sum(value.square().sum() for value in gradient), targets)
    assert all(bool(torch.isfinite(value).all()) for value in (*gradient, *hvp))
    assert posterior.mean.device == x.device


@pytest.mark.parametrize("backend", ["native", "torch"])
def test_custom_failure_preserves_the_loaded_law_and_module_owners(backend):
    x, y, noise = _data()
    model = (
        CustomGP(x, y, noise)
        if backend == "native"
        else SaasFullyBayesianSingleTaskGP(x, y, noise, backend="torch")
    )
    if backend == "torch":
        model.pyro_model = CustomSaas()
    options = dict(
        warmup_steps=2, num_samples=2, thinning=1, max_tree_depth=2, seed=11, disable_progbar=True
    )
    with jax_x64():
        fit_fully_bayesian_model_nuts(model, **options)
        original = model.posterior(x[:2]).mean.clone()
        mean, kernel, pyro = model.mean_module, model.covar_module, model.pyro_model
        raw = kernel.base_kernel.raw_lengthscale.clone()
        pyro.fail_postprocess = True
        with pytest.raises(RuntimeError, match="caller postprocess failed"):
            fit_fully_bayesian_model_nuts(model, **options)
    assert model.mean_module is mean and model.covar_module is kernel and model.pyro_model is pyro
    assert pyro.postprocess_calls == 1
    torch.testing.assert_close(model.posterior(x[:2]).mean, original, atol=0, rtol=0)
    current = kernel.base_kernel.raw_lengthscale
    torch.testing.assert_close(current, raw, atol=0, rtol=0)
    if backend == "native":
        assert kernel._model is model and mean._model is model


def test_prior_mode_program_is_fitted_without_turning_observations_back_on():
    x, y, noise = _data()
    model = SaasFullyBayesianSingleTaskGP(x, y, noise)
    model.pyro_model._prior_mode = True
    with jax_x64():
        fit_fully_bayesian_model_nuts(
            model,
            warmup_steps=2,
            num_samples=3,
            thinning=2,
            max_tree_depth=2,
            seed=9,
            disable_progbar=True,
        )
    raw = model._numpyro_result["raw_samples"]
    assert raw["f"].shape == raw["Y"].shape == (3, len(x))
    assert model.pyro_model._prior_mode is True
    assert model.num_mcmc_samples == 2


def test_builtin_programs_keep_the_fast_sampler_route():
    x, y, noise = _data()
    for backend in ("native", "torch"):
        assert not requires_numpyro_inference(
            SaasFullyBayesianSingleTaskGP(x, y, noise, backend=backend)
        )


def test_failed_native_loader_preserves_live_model_owners_and_hyperparameters():
    x, y, noise = _data()
    model = CustomGP(x, y, noise)
    options = dict(
        warmup_steps=2, num_samples=2, thinning=1, max_tree_depth=2, seed=3, disable_progbar=True
    )
    with jax_x64():
        fit_fully_bayesian_model_nuts(model, **options)
        expected = model.posterior(x).mean.clone()
        native, kernel, mean = model._rs, model.covar_module, model.mean_module
        model.pyro_model.fail_load = True
        with pytest.raises(RuntimeError, match="caller loader failed"):
            fit_fully_bayesian_model_nuts(model, **options)
    assert model._rs is native and model.covar_module is kernel and model.mean_module is mean
    assert kernel._model is model and mean._model is model
    assert model.pyro_model.load_calls == 1
    torch.testing.assert_close(model.posterior(x).mean, expected, atol=0, rtol=0)


def test_tensor_transform_identity_and_failure_atomicity():
    from botorch_rs.models.transforms.input import Normalize
    from botorch_rs.models.transforms.outcome import Standardize

    x, y, noise = _data(gradients=True)
    model = SaasFullyBayesianSingleTaskGP(
        x, y, noise, input_transform=Normalize(1), outcome_transform=Standardize(1), backend="torch"
    )
    model.pyro_model = CustomSaas()
    transform, outcome = model.input_transform, model.outcome_transform
    options = dict(
        warmup_steps=2, num_samples=2, thinning=1, max_tree_depth=2, seed=2, disable_progbar=True
    )
    with jax_x64():
        fit_fully_bayesian_model_nuts(model, **options)
        assert model._fb_user_input_transform is transform
        assert model.outcome_transform is outcome
        expected = model.posterior(x).mean.clone()
        states = {
            "offset": transform.offset.clone(),
            "coefficient": transform.coefficient.clone(),
            "means": outcome.means.clone(),
            "stdvs": outcome.stdvs.clone(),
        }
        model.pyro_model.fail_postprocess = True
        with pytest.raises(RuntimeError, match="caller postprocess failed"):
            fit_fully_bayesian_model_nuts(model, **options)
    assert model._fb_user_input_transform is transform and model.outcome_transform is outcome
    for name, expected_state in states.items():
        actual = getattr(transform if name in ("offset", "coefficient") else outcome, name)
        torch.testing.assert_close(actual, expected_state, atol=0, rtol=0)
    torch.testing.assert_close(model.posterior(x).mean, expected, atol=0, rtol=0)


def test_custom_multitask_prior_is_supported_on_the_public_tensor_constructor():
    from botorch_rs.models.fully_bayesian_multitask import (
        MultitaskSaasPyroModel,
        SaasFullyBayesianMultiTaskGP,
    )

    class TaskPrior(MultitaskSaasPyroModel):
        def sample_mean(self):
            return numpyro.sample("mean", dist.Normal(jnp.full((self.num_tasks,), 0.4), 0.3))

    x, y, noise = _data()
    x = torch.cat((x, x.new_tensor([[3.0], [7.0], [3.0], [7.0]])), -1)
    pyro = TaskPrior()
    model = SaasFullyBayesianMultiTaskGP(x, y, 1, noise, pyro_model=pyro, backend="torch")
    with jax_x64():
        fit_fully_bayesian_model_nuts(
            model,
            warmup_steps=2,
            num_samples=2,
            thinning=1,
            max_tree_depth=2,
            seed=7,
            disable_progbar=True,
        )
    assert model.pyro_model is pyro and pyro.num_tasks == 2
    assert model._numpyro_result["raw_samples"]["mean"].shape == (2, 2)
    query = x.new_tensor([[0.25], [0.55]], requires_grad=True)
    posterior = model.posterior(query)
    (gradient,) = torch.autograd.grad(posterior.mean.sum(), query, create_graph=True)
    (second,) = torch.autograd.grad(gradient.sum(), query)
    assert bool(torch.isfinite(gradient).all() & torch.isfinite(second).all())


@pytest.mark.parametrize("family", ["matern", "linear"])
def test_custom_warped_family_inferred_noise_and_epsilon_match_loaded_law(family):
    from botorch_rs.models.fully_bayesian import (
        FullyBayesianLinearSingleTaskGP,
        FullyBayesianSingleTaskGP,
        LinearPyroModel,
        MaternPyroModel,
    )

    base = LinearPyroModel if family == "linear" else MaternPyroModel

    class Prior(base):
        def sample_mean(self):
            return numpyro.sample("mean", dist.Normal(jnp.array(0.2), jnp.array(0.4)))

    x, y, _ = _data()
    cls = FullyBayesianLinearSingleTaskGP if family == "linear" else FullyBayesianSingleTaskGP
    model = cls(x, y, use_input_warping=True, backend="torch")
    model.pyro_model = Prior(use_input_warping=True, eps=0.03)
    with jax_x64():
        fit_fully_bayesian_model_nuts(
            model,
            warmup_steps=2,
            num_samples=2,
            thinning=1,
            max_tree_depth=2,
            seed=18,
            disable_progbar=True,
        )
    raw = model._numpyro_result["raw_samples"]
    expected_noise = x.new_tensor(raw["noise"].tolist()) + 1e-4
    torch.testing.assert_close(
        model.likelihood.noise.squeeze(-1), expected_noise, atol=1e-15, rtol=1e-14
    )
    warp = model.input_transform["warp"]
    assert warp._eps == 0.03
    query = x.new_tensor([[0.0], [0.4], [1.0]], requires_grad=True)
    c0, c1 = (x.new_tensor(raw[name].tolist()).unsqueeze(-2) for name in ("c0", "c1"))
    scaled = query * 0.94 + 0.03
    expected = 1 - (1 - scaled.pow(c1)).pow(c0)
    torch.testing.assert_close(warp(query), expected, atol=2e-14, rtol=2e-13)
    posterior = model.posterior(query)
    (gradient,) = torch.autograd.grad(
        posterior.mean.sum() + posterior.variance.sum(), query, create_graph=True
    )
    (hvp,) = torch.autograd.grad(gradient.sum(), query)
    assert bool(torch.isfinite(gradient).all() & torch.isfinite(hvp).all())


def test_custom_tensor_loader_and_native_replay_boundaries_are_explicit():
    from botorch_rs.exceptions.errors import UnsupportedError

    x, y, noise = _data()
    model = SaasFullyBayesianSingleTaskGP(x, y, noise, backend="torch")
    model.pyro_model = CustomLoadedSaas()
    with pytest.raises(UnsupportedError, match="returned-module load_mcmc_samples"):
        fit_fully_bayesian_model_nuts(model, warmup_steps=0, num_samples=1)
    model.pyro_model = CustomSaas()
    with pytest.raises(UnsupportedError, match="recorded-draw replay"):
        fit_fully_bayesian_model_nuts(model, draws=[], init_params=[])


def test_postprocess_keyword_contract_matches_the_public_reference_fitter():
    class KeywordPrior(CustomSaas):
        def postprocess_mcmc_samples(self, *, mcmc_samples):
            return super().postprocess_mcmc_samples(mcmc_samples)

    x, y, noise = _data()
    model = SaasFullyBayesianSingleTaskGP(x, y, noise, backend="torch")
    model.pyro_model = KeywordPrior()
    with jax_x64():
        fit_fully_bayesian_model_nuts(
            model,
            warmup_steps=1,
            num_samples=1,
            thinning=1,
            max_tree_depth=1,
            seed=8,
            disable_progbar=True,
        )
    assert model.num_mcmc_samples == 1
    assert model.pyro_model.postprocess_calls == 1

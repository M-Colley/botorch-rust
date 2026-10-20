"""Optional real NumPyro named-site/effect-handler interoperability."""

from __future__ import annotations

import os
import subprocess
import sys
import types

import pytest

from helpers import importorskip_botorch

importorskip_botorch()
jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")
numpyro = pytest.importorskip("numpyro")
np = pytest.importorskip("numpy")
torch = pytest.importorskip("torch")

from botorch.models import fully_bayesian as reference  # noqa: E402
from botorch.models import fully_bayesian_multitask as reference_mt  # noqa: E402
from fully_bayesian_helpers import jax_x64  # noqa: E402
from numpyro import handlers  # noqa: E402
from numpyro.infer.util import log_density, potential_energy  # noqa: E402

from botorch_rs.exceptions import UnsupportedError  # noqa: E402
from botorch_rs.models import fully_bayesian as native  # noqa: E402
from botorch_rs.models import fully_bayesian_multitask as native_mt  # noqa: E402

KINDS = ["MaternPyroModel", "SaasPyroModel", "LinearPyroModel", "MultitaskSaasPyroModel"]


def pair(name, fixed=False, warped=False, prior=False, rows=4):
    x = torch.tensor([[0.0, 0.1], [0.4, 0.7], [0.8, 1.0], [0.2, 0.9]], dtype=torch.float64)[:rows]
    y = torch.tensor([[0.2], [0.6], [-0.1], [0.8]], dtype=torch.float64)[:rows]
    yvar = (
        torch.tensor([[0.01], [0.03], [0.02], [0.04]], dtype=torch.float64)[:rows]
        if fixed
        else None
    )
    multitask = name.startswith("Multitask")
    # Insert task identifiers in the middle to test nonterminal task columns.
    if multitask:
        x = torch.cat((x[:, :1], (torch.arange(rows) % 2).to(x).unsqueeze(-1), x[:, 1:]), -1)
    kwargs = {"use_input_warping": warped, "eps": 0.03}
    a = getattr(native_mt if multitask else native, name)(**kwargs)
    b = getattr(reference_mt if multitask else reference, name)(**kwargs)
    set_kwargs = {"task_feature": 1, "task_rank": 2} if multitask else {}
    a.set_inputs(x, y, yvar, **set_kwargs)
    b.set_inputs(x, y, yvar, **set_kwargs)
    a._prior_mode = b._prior_mode = prior
    if warped:

        def corrected_warp(self, X, c0, c1):
            indices = [0, 2] if multitask else [0, 1]
            z = jnp.clip(X[:, indices] * (1 - 2 * self._eps) + self._eps, self._eps, 1 - self._eps)
            return X.at[:, indices].set(1 - (1 - z**c1) ** c0)

        b._warp_jax = types.MethodType(corrected_warp, b)
    if prior and fixed:
        original = b.sample_observations

        def corrected_observations(self, mean, K_noiseless, noise):
            return original(mean, K_noiseless, noise.reshape(-1))

        b.sample_observations = types.MethodType(corrected_observations, b)
    return a, b


def traced(model, params=None):
    fn = model.sample if params is None else handlers.substitute(model.sample, data=params)
    return handlers.trace(handlers.seed(fn, rng_seed=75)).get_trace()


@pytest.mark.parametrize("name", KINDS)
@pytest.mark.parametrize("fixed", [False, True])
@pytest.mark.parametrize("warped", [False, True])
def test_exact_named_sites_seeded_values_substitution_and_log_density_gradients(
    name, fixed, warped
):
    with jax_x64():
        a, b = pair(name, fixed, warped)
        at, bt = traced(a), traced(b)
        assert list(at) == list(bt)
        for key in at:
            assert at[key]["type"] == bt[key]["type"]
            np.testing.assert_allclose(at[key]["value"], bt[key]["value"], rtol=2e-11, atol=2e-12)
            if at[key]["type"] == "sample":
                assert at[key]["is_observed"] == bt[key]["is_observed"]
                np.testing.assert_allclose(
                    at[key]["fn"].log_prob(at[key]["value"]),
                    bt[key]["fn"].log_prob(bt[key]["value"]),
                    rtol=2e-10,
                    atol=2e-11,
                )
        params = {
            key: site["value"]
            for key, site in bt.items()
            if site["type"] == "sample" and not site["is_observed"]
        }
        av, ag = jax.jit(jax.value_and_grad(lambda p: log_density(a.sample, (), {}, p)[0]))(params)
        bv, bg = jax.jit(jax.value_and_grad(lambda p: log_density(b.sample, (), {}, p)[0]))(params)
        np.testing.assert_allclose(av, bv, rtol=3e-10, atol=3e-10)
        for key in params:
            np.testing.assert_allclose(ag[key], bg[key], rtol=2e-8, atol=2e-8)
        # Substitute really controls named latent sites, rather than reseeding
        # an unrelated random generator.
        params["mean"] = jnp.full_like(params["mean"], 0.35)
        substituted = traced(a, params)
        np.testing.assert_array_equal(substituted["mean"]["value"], params["mean"])


@pytest.mark.parametrize("name", KINDS)
@pytest.mark.parametrize("fixed", [False, True])
def test_prior_mode_observation_shapes_and_seeded_reference(name, fixed):
    with jax_x64():
        a, b = pair(name, fixed=fixed, prior=True)
        at, bt = traced(a), traced(b)
        assert not at["Y"]["is_observed"]
        assert at["Y"]["value"].shape == (4,)
        assert at["f"]["value"].shape == (4,)
        for key in at:
            np.testing.assert_allclose(at[key]["value"], bt[key]["value"], rtol=2e-10, atol=2e-11)
        np.testing.assert_allclose(a.f_prior_sample, at["f"]["value"])
        np.testing.assert_allclose(a.Y_prior_sample, at["Y"]["value"])


@pytest.mark.parametrize("name", KINDS[:3])
def test_empty_training_targets_do_not_create_observation_site(name):
    with jax_x64():
        a, b = pair(name, rows=0)
        assert "Y" not in traced(a) and list(traced(a)) == list(traced(b))


def test_custom_trace_override_is_effectful_and_native_nuts_rejects_it():
    class Custom(native.SaasPyroModel):
        def sample_mean(self):
            return numpyro.sample("custom_mean", numpyro.distributions.Normal(0.2, 0.5))

    model = Custom()
    model.set_inputs([[0.2], [0.6]], [[0.3], [0.8]])
    trace = traced(model)
    assert "custom_mean" in trace and "mean" not in trace
    with pytest.raises(UnsupportedError, match="custom sample_mean override"):
        model._potential()


def test_multitask_trace_rejects_unmapped_identifiers_instead_of_jax_index_clamping():
    from botorch_rs.exceptions.errors import InputDataError

    model = native_mt.MultitaskSaasPyroModel()
    model.set_inputs([[0.2, 3.0], [0.7, 7.0]], [[0.2], [0.4]], task_feature=-1)
    with pytest.raises(InputDataError, match="must be mapped"):
        traced(model)


@pytest.mark.parametrize("name", KINDS)
@pytest.mark.parametrize("fixed", [False, True])
@pytest.mark.parametrize("warped", [False, True])
def test_trace_unconstrained_potential_and_gradients_match_native_nuts(name, fixed, warped):
    if name.startswith("Multitask") and warped:
        pytest.skip("Native fully Bayesian multitask model fitting rejects warping.")
    with jax_x64():
        model, _ = pair(name, fixed, warped)
        potential = model._potential()
        flat = jnp.linspace(-0.4, 0.3, potential.dim)
        parameters, offset = {}, 0
        for key, shape, _positive in potential.sites():
            count = int(np.prod(shape))
            parameters[key] = flat[offset : offset + count].reshape(shape)
            offset += count
        assert offset == potential.dim
        value, gradient = jax.value_and_grad(lambda p: potential_energy(model.sample, (), {}, p))(
            parameters
        )
        expected, derivative = potential.potential(np.asarray(flat).tolist())
        np.testing.assert_allclose(value, expected, rtol=2e-9, atol=2e-9)
        actual = np.concatenate(
            [np.asarray(gradient[key]).reshape(-1) for key, _, _ in potential.sites()]
        )
        np.testing.assert_allclose(actual, derivative, rtol=2e-8, atol=2e-8)


@pytest.mark.parametrize("helper", ["compute_dists", "matern52_kernel", "linear_kernel"])
def test_public_jax_kernel_helpers_preserve_tracers(helper):
    with jax_x64():
        x = jnp.array([[0.2, 0.4], [0.8, 0.6]])
        scale = jnp.array([0.7, 1.2])
        fn = getattr(native, helper)
        expected = getattr(reference, helper)
        av, ag = jax.jit(jax.value_and_grad(lambda z: fn(z, scale).sum()))(x)
        bv, bg = jax.jit(jax.value_and_grad(lambda z: expected(z, scale).sum()))(x)
        np.testing.assert_allclose(av, bv, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(ag, bg, rtol=1e-11, atol=1e-12)


def test_import_and_native_potential_do_not_import_optional_runtimes():
    code = """
import sys
from botorch_rs.models.fully_bayesian import SaasPyroModel
assert 'jax' not in sys.modules and 'numpyro' not in sys.modules and 'torch' not in sys.modules
m = SaasPyroModel()
m.set_inputs([[0.2], [0.7]], [[0.1], [0.4]])
assert m._potential().sites()
assert 'jax' not in sys.modules and 'numpyro' not in sys.modules and 'torch' not in sys.modules
"""
    subprocess.run([sys.executable, "-c", code], check=True, env=os.environ.copy())


def test_actual_numpyro_mcmc_samples_load_into_native_gp():
    with jax_x64():
        x = torch.tensor([[0.1], [0.4], [0.8]], dtype=torch.float64)
        model = native.SaasFullyBayesianSingleTaskGP(x, x.sin(), outcome_transform=None)
        inference = numpyro.infer.MCMC(
            numpyro.infer.NUTS(model.pyro_model.sample, max_tree_depth=2),
            num_warmup=4,
            num_samples=3,
            progress_bar=False,
        )
        inference.run(jax.random.PRNGKey(13))
        samples = model.pyro_model.postprocess_mcmc_samples(inference.get_samples())
        model.load_mcmc_samples(samples)
        posterior = model.posterior(torch.tensor([[0.2], [0.6]], dtype=torch.float64))
        assert model.num_mcmc_samples == 3
        assert posterior.mean.shape == (3, 2, 1)
        assert torch.isfinite(posterior.mean).all() and torch.isfinite(posterior.variance).all()
        noise = np.asarray(inference.get_samples()["noise"]) + native.MIN_INFERRED_NOISE_LEVEL
        np.testing.assert_allclose(np.asarray(model.likelihood.noise).reshape(-1), noise)

"""Optional NumPyro named-site interoperability; never used by native NUTS.

JAX arrays and tracers stay in JAX here, since effect handlers and compiled
NumPyro potentials cannot cross the native float64 array boundary.
"""

from __future__ import annotations

import math
from typing import Any

from botorch_rs._arrays import to_flat


def backend() -> tuple[Any, Any, Any]:
    try:
        import jax.numpy as jnp
        import numpyro
        import numpyro.distributions as distributions
    except ImportError as error:
        raise ImportError("NumPyro tracing requires optional jax, jaxlib, and numpyro.") from error
    return jnp, numpyro, distributions


def is_jax(value: Any) -> bool:
    return type(value).__module__.startswith(("jax.", "jaxlib."))


def array(value: Any) -> Any:
    jnp, _, _ = backend()
    if is_jax(value):
        return jnp.asarray(value)
    values, shape = to_flat(value)
    return jnp.asarray(values).reshape(shape)


def kernel(X: Any, scale: Any, kind: str) -> Any:
    jnp, _, _ = backend()
    x, scale = array(X), array(scale)
    if kind == "linear":
        weighted = x * jnp.sqrt(scale)
        return jnp.matmul(weighted, jnp.swapaxes(weighted, -2, -1))
    scaled = x / scale
    difference = jnp.expand_dims(scaled, -2) - jnp.expand_dims(scaled, -3)
    distance = jnp.sqrt(jnp.maximum(jnp.sum(difference**2, axis=-1), 1e-30))
    if kind == "distances":
        return distance
    root = math.sqrt(5) * distance
    return (1 + root + (5 / 3) * distance**2) * jnp.exp(-root)


def noise(model: Any) -> Any:
    jnp, numpyro, dist = backend()
    observed = model.train_Yvar_jax
    if observed is not None:
        return observed
    from botorch_rs.models.fully_bayesian import MIN_INFERRED_NOISE_LEVEL

    return MIN_INFERRED_NOISE_LEVEL + numpyro.sample(
        "noise", dist.Gamma(jnp.array(0.9), jnp.array(10.0))
    )


def mean(model: Any, tasks: int | None = None) -> Any:
    jnp, numpyro, dist = backend()
    prior = dist.Normal(jnp.array(0.0), jnp.array(1.0))
    return numpyro.sample("mean", prior if tasks is None else prior.expand((tasks,)))


def concentrations(model: Any) -> tuple[Any, Any]:
    jnp, numpyro, dist = backend()
    dim = model.ard_num_dims if model.indices is None else len(model.indices)
    prior = dist.LogNormal(jnp.zeros(dim), jnp.full(dim, math.sqrt(0.1)))
    return numpyro.sample("c0", prior), numpyro.sample("c1", prior)


def warp(model: Any, X: Any, c0: Any, c1: Any) -> Any:
    jnp, _, _ = backend()
    if model.indices is not None:
        indices = list(model.indices)
    elif hasattr(model, "task_feature"):
        # The task identifier is categorical and must not replace a real
        # feature in the default input-warping selection.
        indices = [i for i in range(X.shape[-1]) if i != model.task_feature]
    else:
        indices = list(range(model.ard_num_dims))
    sub = jnp.clip(X[..., indices] * (1 - 2 * model._eps) + model._eps, model._eps, 1 - model._eps)
    # Match the native Warp transform, correcting the historical Pyro
    # program's reversed concentrations and missing unit-interval scaling.
    return X.at[..., indices].set(1 - (1 - sub**c1) ** c0)


def observations(model: Any, mean: Any, covariance: Any, noise: Any) -> None:
    jnp, numpyro, dist = backend()
    if model.train_Y_jax.shape[-2] == 0:
        return
    n = model.train_X_jax.shape[0]
    loc = jnp.broadcast_to(jnp.reshape(mean, (-1,)), (n,))
    if model._prior_mode:
        model.f_prior_sample = numpyro.sample(
            "f",
            dist.MultivariateNormal(
                loc=loc,
                covariance_matrix=covariance + model._noiseless_eps_for_sampleability * jnp.eye(n),
            ),
        )
        # Known noise arrives as n×1. A vector loc requires n noise scales,
        # otherwise broadcasting incorrectly creates an n×n observation site.
        model.Y_prior_sample = numpyro.sample(
            "Y", dist.Normal(loc=model.f_prior_sample, scale=jnp.sqrt(jnp.reshape(noise, (-1,))))
        )
    else:
        numpyro.sample(
            "Y",
            dist.MultivariateNormal(loc=loc, covariance_matrix=covariance + noise * jnp.eye(n)),
            obs=jnp.squeeze(model.train_Y_jax, axis=-1),
        )


def matern_sample(model: Any) -> None:
    outputscale = model.sample_outputscale(
        concentration=model._outputscale_prior_concentration,
        rate=model._outputscale_prior_rate,
    )
    mean = model.sample_mean()
    noise = model.sample_noise()
    lengthscale = model.sample_lengthscale(dim=model.ard_num_dims)
    X = model._prepare_features(model._maybe_input_warp(model.train_X_jax))
    covariance = outputscale * kernel(X, lengthscale, "matern52")
    covariance, mean = model._maybe_multitask_transform(covariance, mean)
    model.sample_observations(mean=mean, K_noiseless=covariance, noise=noise)


def lengthscale(dim: int) -> Any:
    jnp, numpyro, dist = backend()
    return numpyro.sample(
        "lengthscale",
        dist.LogNormal(
            jnp.full((dim,), math.sqrt(2) + math.log(dim) / 2), jnp.full((dim,), math.sqrt(3))
        ),
    )


def outputscale(concentration: float | None, rate: float | None) -> Any:
    jnp, numpyro, dist = backend()
    if concentration is None or rate is None:
        return jnp.ones(1)
    return numpyro.sample("outputscale", dist.Gamma(jnp.array(concentration), jnp.array(rate)))


def saas_lengthscale(dim: int, alpha: float) -> Any:
    jnp, numpyro, dist = backend()
    tau = numpyro.sample("kernel_tausq", dist.HalfCauchy(jnp.array(alpha)))
    inverse = numpyro.sample("_kernel_inv_length_sq", dist.HalfCauchy(jnp.ones(dim)))
    scaled = numpyro.deterministic("kernel_inv_length_sq", tau * inverse)
    return numpyro.deterministic("lengthscale", 1 / jnp.sqrt(scaled))


def linear_sample(model: Any) -> None:
    mean = model.sample_mean()
    weight_variance = model.sample_weight_variance()
    X = model._prepare_features(model._maybe_input_warp(model.train_X_jax)) - 0.5
    covariance = kernel(X, weight_variance, "linear")
    noise = model.sample_noise()
    covariance, mean = model._maybe_multitask_transform(covariance, mean)
    model.sample_observations(mean=mean, K_noiseless=covariance, noise=noise)


def weight_variance(dim: int, alpha: float) -> Any:
    jnp, numpyro, dist = backend()
    tau = numpyro.sample("tau_sq", dist.HalfCauchy(jnp.array(alpha)))
    weight = numpyro.sample("_weight_variance_sq", dist.HalfCauchy(jnp.ones(dim)))
    return numpyro.deterministic("weight_variance", jnp.sqrt(tau * weight))


def latent_features(tasks: int, rank: int) -> Any:
    jnp, numpyro, dist = backend()
    return numpyro.sample(
        "latent_features", dist.Normal(jnp.array(0.0), jnp.array(1.0)).expand((tasks, rank))
    )


def task_lengthscale(rank: int, concentration: float, rate: float) -> Any:
    jnp, numpyro, dist = backend()
    return numpyro.sample(
        "task_lengthscale", dist.Gamma(jnp.array(concentration), jnp.array(rate)).expand((rank,))
    )

"""Shared helpers of the comparisons of the fully Bayesian GPs (phase 10a)
with BoTorch (``test_parity_10a_fully_bayesian.py``,
``test_equivalence_10a_fully_bayesian.py`` and
``scripts/fixtures_10a_fully_bayesian.py``).

BoTorch fits its fully Bayesian models with NumPyro's NUTS (JAX). The
recorder below captures every random number NumPyro's sampler draws (the
standard-normal momenta of ``momentum_generator`` and the uniform draws behind
its ``random.bernoulli`` calls), the initial unconstrained parameters and the
state after every iteration, without changing the computation: the draws are
recomputed from the same keys inside the jitted program and sent to the host
with ordered ``jax.debug.callback``s (BoTorch's samples are bitwise
unchanged). botorch_rs's sampler replays them (``draws``).

JAX computes in float32 unless ``jax_enable_x64`` is set (BoTorch does not
set it): exact replications use ``jax_x64()``; BoTorch's float32 chains are
compared statistically.

Import only after ``importorskip_botorch()`` and with JAX installed.
"""

from __future__ import annotations

import contextlib
import importlib
import math
import os
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch


def has_jax() -> bool:
    try:
        import jax  # noqa: F401
        import numpyro  # noqa: F401
    except (ImportError, AttributeError):
        return False
    return True


def require_jax() -> None:
    """Skips the calling test module unless JAX and NumPyro (BoTorch's
    optional dependencies of its fully Bayesian models) are importable; with
    ``BOTORCH_RS_REQUIRE_BOTORCH=1`` a missing JAX is an error instead."""
    import pytest

    if os.environ.get("BOTORCH_RS_REQUIRE_BOTORCH"):
        importlib.import_module("jax")
        importlib.import_module("numpyro")
    pytest.importorskip("jax")
    pytest.importorskip("numpyro")


@contextlib.contextmanager
def jax_x64(enabled: bool = True) -> Iterator[None]:
    """JAX in double (or single) precision (a global flag; restored)."""
    import jax

    old = bool(jax.config.jax_enable_x64)
    jax.config.update("jax_enable_x64", enabled)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", old)


def flatten_z(z: dict[str, Any]) -> list[float]:
    """NumPyro's ``ravel_pytree`` of a dict of sites (sorted by name)."""
    out: list[float] = []
    for k in sorted(z):
        out.extend(np.asarray(z[k], dtype=np.float64).reshape(-1).tolist())
    return out


@dataclass
class Recording:
    """The random draws, the initial point and the states of a NumPyro run."""

    draws: list[tuple[Any, ...]] = field(default_factory=list)
    z0: list[float] | None = None
    states: list[tuple[int, list[float], float]] = field(default_factory=list)
    #: BoTorch's post-processed samples (before thinning).
    postprocessed: dict[str, Any] | None = None
    #: The samples loaded into the model (after thinning).
    mcmc_samples: dict[str, Any] | None = None


@contextlib.contextmanager
def recording_numpyro(rec: Recording, fixed_warping: bool = False) -> Iterator[Recording]:
    """Records the draws of NumPyro's NUTS run inside the block.

    ``fixed_warping`` replaces BoTorch's JAX input warping
    (``PyroModel._warp_jax``: ``1 - (1 - clip(x)^c0)^c1``) with the warping
    of BoTorch's ``Warp`` transform that the fitted model uses
    (``kumaraswamy_warp``: ``1 - (1 - x'^c1)^c0`` of
    ``x' = clamp(x (1 - 2 eps) + eps, eps, 1 - eps)``), the fix of the BoTorch
    bug that botorch_rs implements (see ``docs/drafts/10a-fully-bayesian.md``).
    """
    import jax
    import jax.numpy as jnp
    from botorch.models import fully_bayesian as fb
    from botorch.models import fully_bayesian_multitask as fbmt
    from numpyro.infer import hmc as hmc_mod
    from numpyro.infer import hmc_util

    def rec_u(u: Any, p: Any) -> None:
        rec.draws.append(("u", float(u), float(p)))

    def rec_n(x: Any) -> None:
        rec.draws.append(("n", np.asarray(x, dtype=np.float64).reshape(-1).tolist()))

    def rec_state(i: Any, z: Any, step_size: Any) -> None:
        rec.states.append((int(i), flatten_z(z), float(step_size)))

    class RandomProxy:
        def __getattr__(self, name: str) -> Any:
            return getattr(jax.random, name)

        def bernoulli(self, key: Any, p: Any = 0.5, shape: Any = None) -> Any:
            # `jax.random.bernoulli`: `uniform(key, shape, dtype(p)) < p`.
            out = jax.random.bernoulli(key, p, shape)
            u = jax.random.uniform(key, np.shape(p) if shape is None else shape, jax.lax.dtype(p))
            jax.debug.callback(rec_u, u, p, ordered=True)
            return out

        def normal(self, key: Any, shape: Any = (), dtype: Any = float) -> Any:
            x = jax.random.normal(key, shape, dtype)
            jax.debug.callback(rec_n, x, ordered=True)
            return x

    orig_init, orig_sample = hmc_mod.HMC.init, hmc_mod.HMC.sample
    orig_random = hmc_mod.random, hmc_util.random
    orig_post = {
        cls: cls.postprocess_mcmc_samples
        for cls in (fb.MaternPyroModel, fb.SaasPyroModel, fb.LinearPyroModel)
    }
    orig_warp = fb.PyroModel._warp_jax
    orig_load = {
        cls: cls.__dict__["load_mcmc_samples"]
        for cls in (fb.AbstractFullyBayesianSingleTaskGP, fbmt.SaasFullyBayesianMultiTaskGP)
    }

    def make_load(cls: Any) -> Any:
        def load(self: Any, mcmc_samples: Any) -> Any:
            rec.mcmc_samples = {k: v.clone() for k, v in mcmc_samples.items()}
            return orig_load[cls](self, mcmc_samples)

        return load

    def init(self: Any, *args: Any, **kwargs: Any) -> Any:
        state = orig_init(self, *args, **kwargs)
        rec.z0 = flatten_z(state.z)
        return state

    def sample(self: Any, state: Any, *args: Any, **kwargs: Any) -> Any:
        new = orig_sample(self, state, *args, **kwargs)
        jax.debug.callback(rec_state, new.i, new.z, new.adapt_state.step_size, ordered=True)
        return new

    def make_post(cls: Any) -> Any:
        def post(self: Any, mcmc_samples: Any) -> Any:
            out = orig_post[cls](self, mcmc_samples)
            rec.postprocessed = {k: v.clone() for k, v in out.items()}
            return out

        return post

    def warp_fixed(self: Any, X: Any, c0: Any, c1: Any) -> Any:
        eps = self._eps
        indices = list(range(self.ard_num_dims)) if self.indices is None else self.indices
        X_sub = X[:, indices]
        X_sub = jnp.clip(X_sub * (1 - 2 * eps) + eps, eps, 1 - eps)
        warped = 1 - (1 - X_sub**c1) ** c0
        return X.at[:, indices].set(warped)

    hmc_mod.random = hmc_util.random = RandomProxy()
    hmc_mod.HMC.init, hmc_mod.HMC.sample = init, sample
    for cls in orig_post:
        cls.postprocess_mcmc_samples = make_post(cls)
    for cls in orig_load:
        cls.load_mcmc_samples = make_load(cls)
    if fixed_warping:
        fb.PyroModel._warp_jax = warp_fixed
    try:
        yield rec
    finally:
        hmc_mod.random, hmc_util.random = orig_random
        hmc_mod.HMC.init, hmc_mod.HMC.sample = orig_init, orig_sample
        for cls, fn in orig_post.items():
            cls.postprocess_mcmc_samples = fn
        fb.PyroModel._warp_jax = orig_warp
        for cls, fn in orig_load.items():
            cls.load_mcmc_samples = fn


def bt_fit_recorded(
    model: Any, x64: bool = True, fixed_warping: bool = False, **kwargs: Any
) -> Recording:
    """Fits a BoTorch fully Bayesian model with ``fit_fully_bayesian_model_nuts``
    (``disable_progbar=True``) and records NumPyro's run. The model must
    have been created under the same ``jax_x64`` setting (BoTorch converts
    the training data to JAX arrays at construction)."""
    from botorch.fit import fit_fully_bayesian_model_nuts

    rec = Recording()
    with jax_x64(x64), recording_numpyro(rec, fixed_warping=fixed_warping):
        fit_fully_bayesian_model_nuts(model, disable_progbar=True, **kwargs)
    return rec


def draws_for_rust(rec: Recording) -> list[tuple[Any, ...]]:
    return list(rec.draws)


def ess(x: Any) -> float:
    """The effective sample size of a chain (NumPyro's / Stan's estimator
    with Geyer's initial monotone sequence, one chain)."""
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    if n < 4:
        return float(n)
    xc = x - x.mean()
    var = xc.var()
    if var == 0:
        return float(n)
    f = np.fft.rfft(xc, n=2 * n)
    acov = np.fft.irfft(f * np.conj(f))[:n] / n
    rho = acov / acov[0]
    # Geyer: sums of pairs, truncated at the first negative pair, monotone.
    pairs = rho[: n - n % 2].reshape(-1, 2).sum(axis=1)
    out = []
    best = math.inf
    for p in pairs:
        if p <= 0:
            break
        best = min(best, p)
        out.append(best)
    tau = -1.0 + 2.0 * sum(out) if out else 1.0
    return float(n / max(tau, 1e-12))


def standardized_mean_difference(a: Any, b: Any) -> float:
    """``(mean(a) - mean(b)) / sqrt(se_a² + se_b²)`` with Monte-Carlo
    standard errors from the effective sample sizes."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    se2 = a.var(ddof=1) / ess(a) + b.var(ddof=1) / ess(b)
    if se2 == 0:
        return 0.0 if a.mean() == b.mean() else math.inf
    return float((a.mean() - b.mean()) / math.sqrt(se2))


def as_t(x: Any) -> torch.Tensor:
    return torch.as_tensor(x, dtype=torch.float64)


# ---------------------------------------------------------------------------
# Cases shared by the fixtures and the live tests
# ---------------------------------------------------------------------------

DT = torch.float64
#: BoTorch's ``MIN_INFERRED_NOISE_LEVEL``.
MIN_NOISE = 1e-4
#: Relative perturbations of the training inputs of BoTorch's twins.
TWIN_SCALES = (1e-14, 1e-13)


def fb_cases() -> list[dict[str, Any]]:
    """Fully Bayesian models (``model``: "saas", "matern", "linear"), their
    data (``n`` points in ``d`` dimensions, fixed noise) and NUTS settings
    (short chains: their replications are exact, see the module docs)."""
    return [
        {"name": "saas_d3", "model": "saas", "d": 3, "n": 10, "warmup": 24, "samples": 8},
        {
            "name": "saas_d4_fixed",
            "model": "saas",
            "d": 4,
            "n": 12,
            "fixed": True,
            "warmup": 24,
            "samples": 8,
        },
        {
            "name": "saas_warp",
            "model": "saas",
            "d": 3,
            "n": 10,
            "warp": True,
            "warmup": 20,
            "samples": 8,
        },
        {"name": "matern_d2", "model": "matern", "d": 2, "n": 9, "warmup": 24, "samples": 8},
        {
            "name": "matern_warp_fixed",
            "model": "matern",
            "d": 3,
            "n": 10,
            "warp": True,
            "fixed": True,
            "warmup": 20,
            "samples": 8,
        },
        {"name": "linear_d3", "model": "linear", "d": 3, "n": 10, "warmup": 24, "samples": 8},
        {
            "name": "linear_warp",
            "model": "linear",
            "d": 2,
            "n": 8,
            "warp": True,
            "warmup": 20,
            "samples": 8,
        },
    ]


def make_fb_data(case: dict[str, Any], seed: int) -> tuple[Any, Any, Any]:
    """Inputs in the unit cube, standardized outputs, noise variances."""
    g = torch.Generator().manual_seed(seed)
    n, d = case["n"], case["d"]
    X = torch.rand(n, d, generator=g, dtype=DT)
    f = torch.sin(5.0 * X[:, 0]) + 0.5 * (X[:, 1:] ** 2).sum(-1)
    Y = (f + 0.1 * torch.randn(n, generator=g, dtype=DT)).unsqueeze(-1)
    Y = (Y - Y.mean()) / Y.std()
    Yvar = 0.01 + 0.02 * torch.rand(n, 1, generator=g, dtype=DT)
    return X, Y, Yvar


def bt_class(model: str) -> Any:
    from botorch.models import fully_bayesian as fb

    return {
        "saas": fb.SaasFullyBayesianSingleTaskGP,
        "matern": fb.FullyBayesianSingleTaskGP,
        "linear": fb.FullyBayesianLinearSingleTaskGP,
    }[model]


def rs_class(model: str) -> Any:
    from botorch_rs.models import fully_bayesian as fb

    return {
        "saas": fb.SaasFullyBayesianSingleTaskGP,
        "matern": fb.FullyBayesianSingleTaskGP,
        "linear": fb.FullyBayesianLinearSingleTaskGP,
    }[model]


def bt_model(case: dict[str, Any], X: Any, Y: Any, Yvar: Any, x64: bool = True, **kw: Any) -> Any:
    """The BoTorch model of a case (created under ``jax_x64(x64)``)."""
    with jax_x64(x64):
        return bt_class(case["model"])(
            X,
            Y,
            Yvar if case.get("fixed") else None,
            use_input_warping=bool(case.get("warp")),
            **kw,
        )


def numpyro_potential(bt: Any, z: list[float]) -> tuple[float, list[float]]:
    """NumPyro's potential energy of the BoTorch model at the unconstrained
    parameters ``z`` (sites sorted by name) and its gradient (JAX), in
    double precision (the model must have been created under ``jax_x64``).
    The JAX warping is the fixed one (``recording_numpyro``)."""
    import jax
    import jax.numpy as jnp
    from botorch.models import fully_bayesian as fb
    from numpyro import handlers
    from numpyro.infer.util import potential_energy

    model = bt.pyro_model.sample
    with jax_x64(True):
        trace = handlers.trace(handlers.seed(model, rng_seed=0)).get_trace()
        sites = sorted(
            name
            for name, s in trace.items()
            if s["type"] == "sample" and not s.get("is_observed", False)
        )
        params = {}
        off = 0
        for name in sites:
            shape = tuple(np.shape(trace[name]["value"]))
            size = int(np.prod(shape)) if shape else 1
            params[name] = jnp.asarray(np.asarray(z[off : off + size]).reshape(shape))
            off += size
        assert off == len(z), (off, len(z))
        orig = fb.PyroModel._warp_jax

        def warp_fixed(self: Any, X: Any, c0: Any, c1: Any) -> Any:
            eps = self._eps
            indices = list(range(self.ard_num_dims)) if self.indices is None else self.indices
            X_sub = jnp.clip(X[:, indices] * (1 - 2 * eps) + eps, eps, 1 - eps)
            return X.at[:, indices].set(1 - (1 - X_sub**c1) ** c0)

        fb.PyroModel._warp_jax = warp_fixed
        try:
            value, grad = jax.value_and_grad(lambda p: potential_energy(model, (), {}, p))(params)
        finally:
            fb.PyroModel._warp_jax = orig
    return float(value), flatten_z(grad)


def perturbed(X: Any, scale: float, seed: int) -> Any:
    g = torch.Generator().manual_seed(seed)
    return X * (1 + scale * torch.randn(X.shape, generator=g, dtype=DT))


def twin_spread(
    case: dict[str, Any],
    X: Any,
    Y: Any,
    Yvar: Any,
    rec: Recording,
    num_twins: int,
    fit_kwargs: dict[str, Any],
) -> tuple[list[float], list[bool]]:
    """BoTorch's own change of its chain under perturbations of the training
    inputs at ``TWIN_SCALES``: per iteration, the largest change of the
    unconstrained state over ``num_twins`` twins, and whether the twin made
    the same random decisions as the original (draws compared up to the
    first difference)."""
    spread = [0.0] * len(rec.states)
    same: list[bool] = []
    for t in range(num_twins):
        Xt = perturbed(X, TWIN_SCALES[t % len(TWIN_SCALES)], 1000 + t)
        twin = bt_model(case, Xt, Y, Yvar)
        r = bt_fit_recorded(twin, fixed_warping=True, **fit_kwargs)
        for i, ((_, za, _), (_, zb, _)) in enumerate(zip(rec.states, r.states, strict=True)):
            spread[i] = max(spread[i], float(np.max(np.abs(np.asarray(za) - np.asarray(zb)))))
        same.append(
            len(r.draws) == len(rec.draws)
            and all(
                (a[0] == "n" and b[0] == "n")
                or (a[0] == b[0] == "u" and (a[1] < a[2]) == (b[1] < b[2]))
                for a, b in zip(rec.draws, r.draws, strict=True)
            )
        )
    return spread, same


# ---------------------------------------------------------------------------
# Models from given MCMC samples (model-level comparisons)
# ---------------------------------------------------------------------------


def synthetic_samples(
    model: str, d: int, num: int, seed: int, fixed: bool, warp: bool
) -> dict[str, Any]:
    """MCMC samples as ``load_mcmc_samples`` takes them (a mixture of
    typical hyperparameters, some noise below BoTorch's clamp at 1e-4)."""
    g = torch.Generator().manual_seed(seed)

    def u(lo: float, hi: float, *shape: int) -> Any:
        return lo + (hi - lo) * torch.rand(*shape, generator=g, dtype=DT)

    out: dict[str, Any] = {"mean": u(-0.5, 0.5, num)}
    if not fixed:
        noise = u(-4.5, -1.0, num)
        out["noise"] = 10.0**noise
    if model == "linear":
        out["weight_variance"] = u(0.05, 1.5, num, d)
    else:
        out["lengthscale"] = torch.exp(u(math.log(0.2), math.log(4.0), num, d))
        if model == "saas":
            out["outputscale"] = u(0.3, 3.0, num)
            out["kernel_inv_length_sq"] = 1.0 / out["lengthscale"] ** 2
    if warp:
        out["c0"] = torch.exp(u(-0.6, 0.6, num, d))
        out["c1"] = torch.exp(u(-0.6, 0.6, num, d))
    return out


def raw_box_data(n: int, d: int, seed: int, transforms: bool) -> tuple[Any, Any, Any]:
    """Data in a box (with transforms) or the unit cube (standardized)."""
    g = torch.Generator().manual_seed(seed)
    X = torch.rand(n, d, generator=g, dtype=DT)
    f = torch.sin(5.0 * X[:, 0]) + 0.5 * (X[:, 1:] ** 2).sum(-1)
    Y = (f + 0.1 * torch.randn(n, generator=g, dtype=DT)).unsqueeze(-1)
    Yvar = 0.01 + 0.02 * torch.rand(n, 1, generator=g, dtype=DT)
    if transforms:
        lo = torch.linspace(-2.0, 1.0, d, dtype=DT)
        X = lo + 3.0 * X
        Y = 2.0 + 3.0 * Y
        Yvar = 9.0 * Yvar
    else:
        Y = (Y - Y.mean()) / Y.std()
    return X, Y, Yvar


def model_pair(
    model: str,
    d: int = 3,
    n: int = 10,
    num: int = 4,
    seed: int = 0,
    fixed: bool = False,
    warp: bool = False,
    transforms: bool = True,
    samples: dict[str, Any] | None = None,
) -> tuple[Any, Any, Any, Any]:
    """A BoTorch model and its botorch_rs twin with the same MCMC samples
    loaded (``synthetic_samples`` by default), and the training data."""
    from botorch.models.transforms import Normalize as BtNormalize
    from botorch.models.transforms import Standardize as BtStandardize

    from botorch_rs.models.transforms import Normalize as RsNormalize
    from botorch_rs.models.transforms import Standardize as RsStandardize

    X, Y, Yvar = raw_box_data(n, d, seed, transforms)
    yvar = Yvar if fixed else None
    bkw: dict[str, Any] = {"use_input_warping": warp}
    rkw: dict[str, Any] = {"use_input_warping": warp}
    if transforms:
        bkw.update(outcome_transform=BtStandardize(m=1), input_transform=BtNormalize(d=d))
        rkw.update(outcome_transform=RsStandardize(m=1), input_transform=RsNormalize(d=d))
    bt = bt_class(model)(X, Y, yvar, **bkw)
    rs = rs_class(model)(X, Y, yvar, **rkw)
    if samples is None:
        samples = synthetic_samples(model, d, num, seed + 1, fixed, warp)
    bt.load_mcmc_samples({k: v.clone() for k, v in samples.items()})
    bt.eval()
    rs.load_mcmc_samples({k: v.clone() for k, v in samples.items()})
    rs.eval()
    return bt, rs, X, samples

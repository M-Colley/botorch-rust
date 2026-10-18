"""Parity of the fully Bayesian GPs of phase 10a with BoTorch, through the
public APIs of both libraries side by side:

* the potential energies of the NumPyro models (``SaasPyroModel``,
  ``MaternPyroModel``, ``LinearPyroModel``; inferred or fixed noise, input
  warping) against NumPyro's ``potential_energy`` and its JAX gradient (in
  double precision);
* ``fit_fully_bayesian_model_nuts`` replaying NumPyro's recorded random
  draws (``draws=``, ``init_params=``): NumPyro's random decisions (exact),
  the chain within 10x BoTorch's own change under perturbations of the
  training inputs at 1e-14 (two twins), and the loaded samples;
* the models built from given MCMC samples (``load_mcmc_samples``):
  posteriors (with and without observation noise, batched inputs), the
  mixture moments and quantiles, the MLL of every sample and its gradient,
  the acquisition functions averaged over the samples (analytic and Monte
  Carlo, values and input gradients), ``prune_inferior_points``,
  ``condition_on_observations``, ``fantasize``, ``median_lengthscale``,
  ``median_weight_variance``, ``num_mcmc_samples``, ``batch_shape`` and
  ``construct_inputs``; with ``Normalize`` / ``Standardize``, fixed noise
  and input warping.

Tolerances: values 1e-9, gradients 1e-8 (relative); the chains as above.
BoTorch's NumPyro model warps its inputs differently from its fitted model
(a BoTorch bug, see ``docs/drafts/10a-fully-bayesian.md``); the replays of
warped models compare with the fixed JAX warping (``fixed_warping=True``).
The randomized sweep and the statistical comparison of BoTorch's default
(single-precision) chains are in ``test_equivalence_10a_fully_bayesian.py``.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest

from helpers import importorskip_botorch

importorskip_botorch()
torch = pytest.importorskip("torch")

from fully_bayesian_helpers import require_jax  # noqa: E402

require_jax()

import numpy as np  # noqa: E402
from botorch.acquisition import analytic as bt_an  # noqa: E402
from botorch.acquisition import logei as bt_logei  # noqa: E402
from botorch.acquisition import monte_carlo as bt_mc  # noqa: E402
from botorch.acquisition.utils import prune_inferior_points as bt_prune  # noqa: E402
from botorch.sampling.normal import SobolQMCNormalSampler as BtSampler  # noqa: E402
from botorch.utils.datasets import SupervisedDataset as BtDataset  # noqa: E402
from fully_bayesian_helpers import (  # noqa: E402
    DT,
    MIN_NOISE,
    bt_fit_recorded,
    bt_model,
    fb_cases,
    make_fb_data,
    model_pair,
    numpyro_potential,
    perturbed,
    rs_class,
)
from models7b_helpers import GRAD_TOL, VALUE_TOL, check_grad, values_and_grads  # noqa: E402

from botorch_rs.acquisition import analytic as rs_an  # noqa: E402
from botorch_rs.acquisition import logei as rs_logei  # noqa: E402
from botorch_rs.acquisition import monte_carlo as rs_mc  # noqa: E402
from botorch_rs.acquisition.utils import prune_inferior_points as rs_prune  # noqa: E402
from botorch_rs.fit import fit_fully_bayesian_model_nuts as rs_fit  # noqa: E402
from botorch_rs.mlls import ExactMarginalLogLikelihood as RsMLL  # noqa: E402
from botorch_rs.sampling import SobolQMCNormalSampler as RsSampler  # noqa: E402
from botorch_rs.utils.datasets import SupervisedDataset as RsDataset  # noqa: E402
from parity_helpers import Checker  # noqa: E402

pytestmark = pytest.mark.filterwarnings("ignore")

CHECK = Checker("fully Bayesian (10a)")
MODELS = ("saas", "matern", "linear")
VARIANTS = [
    pytest.param(fixed, warp, transforms, id=f"fixed={fixed}-warp={warp}-tf={transforms}")
    for fixed, warp, transforms in [
        (False, False, True),
        (True, False, True),
        (False, True, True),
        (True, True, False),
        (False, False, False),
    ]
]


@pytest.fixture(autouse=True, scope="module")
def _single_threaded_torch() -> Iterator[None]:
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(threads)


def _gen(seed: int) -> Any:
    return torch.Generator().manual_seed(seed)


def _t(x: Any) -> Any:
    return torch.as_tensor(x, dtype=DT)


# ---------------------------------------------------------------------------
# Potentials and NUTS
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("fixed,warp", [(False, False), (True, False), (False, True)])
def test_potential_matches_numpyro(model: str, fixed: bool, warp: bool) -> None:
    case = {"model": model, "d": 3, "n": 9, "fixed": fixed, "warp": warp}
    X, Y, Yvar = make_fb_data(case, 11)
    bt = bt_model(case, X, Y, Yvar)
    rs = rs_class(model)(X, Y, Yvar if fixed else None, use_input_warping=warp)
    pot = rs.pyro_model._potential()
    g = _gen(12)
    for _ in range(3):
        z = (4 * torch.rand(pot.dim, generator=g, dtype=DT) - 2).tolist()
        vb, gb = numpyro_potential(bt, z)
        vr, gr = pot.potential(z)
        CHECK.check("potential energy", [vr], [vb], VALUE_TOL)
        check_grad(CHECK, "potential energy gradient", gr, gb)


def _replay_cases() -> list[Any]:
    names = ("saas_d3", "matern_warp_fixed", "linear_warp")
    return [pytest.param(c, id=c["name"]) for c in fb_cases() if c["name"] in names]


def _loaded(model: Any, kind: str) -> dict[str, Any]:
    """The hyperparameters of the samples loaded into a model."""
    out: dict[str, Any] = {"mean": _t(model.mean_module.constant).reshape(-1)}
    covar = model.covar_module
    if kind == "linear":
        out["weight_variance"] = _t(covar.variance).reshape(len(out["mean"]), -1)
    else:
        base = covar.base_kernel if hasattr(covar, "base_kernel") else covar
        out["lengthscale"] = _t(base.lengthscale).reshape(len(out["mean"]), -1)
        if hasattr(covar, "base_kernel"):
            out["outputscale"] = _t(covar.outputscale).reshape(-1)
    return out


@pytest.mark.parametrize("case", _replay_cases())
def test_nuts_replay_matches_numpyro(case: dict[str, Any]) -> None:
    """botorch_rs's NUTS on NumPyro's recorded draws retraces BoTorch's
    chain (JAX in double precision)."""
    X, Y, Yvar = make_fb_data(case, 21)
    kw = {"warmup_steps": case["warmup"], "num_samples": case["samples"], "thinning": 2}
    bt = bt_model(case, X, Y, Yvar)
    rec = bt_fit_recorded(bt, fixed_warping=True, **kw)
    spread = [0.0] * len(rec.states)
    for t in range(2):
        twin = bt_model(case, perturbed(X, 1e-14, 300 + t), Y, Yvar)
        r = bt_fit_recorded(twin, fixed_warping=True, **kw)
        for i, (a, b) in enumerate(zip(rec.states, r.states, strict=True)):
            spread[i] = max(spread[i], float(np.max(np.abs(np.subtract(a[1], b[1])))))
    rs = rs_class(case["model"])(
        X, Y, Yvar if case.get("fixed") else None, use_input_warping=bool(case.get("warp"))
    )
    rs_fit(rs, draws=rec.draws, init_params=rec.z0, **kw)
    res = rs._nuts_result
    CHECK.check_bound(
        "NUTS replay: unconsumed recorded draws", res["num_draws"] - res["consumed"], 0
    )
    CHECK.check_bound(
        "NUTS replay: decisions differing from NumPyro's", res["first_flip"] is not None, 0
    )
    worst = 0.0
    for t, ((_, zb, _), it) in enumerate(zip(rec.states, res["trace"], strict=True)):
        err = float(np.max(np.abs(np.subtract(it[0], zb))))
        worst = max(worst, err / max(10 * spread[t], 1e-12))
    CHECK.check_bound("NUTS replay: chain states / (10x BoTorch's twin spread)", worst, 1.0)
    # The loaded samples (noise: the noise variance of the likelihood,
    # 1e-4 + BoTorch's sample).
    assert rec.mcmc_samples is not None
    expected = {k: v for k, v in rec.mcmc_samples.items()}
    tol = max(10 * spread[-1], 1e-12)
    got = _loaded(rs, case["model"])
    for name, value in got.items():
        e = expected[name].reshape(value.shape)
        err = ((value - e).abs() / e.abs().clamp_min(1.0)).max().item()
        CHECK.check_bound(f"NUTS replay: loaded {name} / (10x twin spread)", err / tol, 1.0)
    if not case.get("fixed"):
        noise = _t(rs.likelihood.noise).reshape(-1)
        e = expected["noise"].reshape(-1) + MIN_NOISE
        err = ((noise - e).abs() / e).max().item()
        CHECK.check_bound(
            "NUTS replay: loaded noise (1e-4 + sample) / (10x twin spread)", err / tol, 1.0
        )


def test_fit_is_deterministic_and_samples_the_model() -> None:
    """``seed`` makes botorch_rs's fits reproducible; different seeds give
    different chains."""
    case = {"model": "saas", "d": 2, "n": 8}
    X, Y, _ = make_fb_data(case, 31)
    fits = []
    for seed in (0, 0, 1):
        rs = rs_class("saas")(X, Y)
        rs_fit(rs, warmup_steps=32, num_samples=16, thinning=4, seed=seed)
        fits.append(_t(rs.median_lengthscale))
    assert torch.equal(fits[0], fits[1])
    assert not torch.equal(fits[0], fits[2])


# ---------------------------------------------------------------------------
# Models from given samples
# ---------------------------------------------------------------------------


def _compare_posteriors(label: str, bt: Any, rs: Any, Xt: Any) -> None:
    with torch.no_grad():
        for noise in (False, True):
            pb = bt.posterior(Xt, observation_noise=noise)
            pr = rs.posterior(Xt, observation_noise=noise)
            CHECK.check(f"{label}: posterior mean", pr.mean, pb.mean, VALUE_TOL)
            CHECK.check(f"{label}: posterior variance", pr.variance, pb.variance, VALUE_TOL)
            CHECK.check(
                f"{label}: posterior covariance",
                pr.distribution.covariance_matrix,
                pb.distribution.covariance_matrix,
                VALUE_TOL,
            )
            CHECK.check(f"{label}: mixture mean", pr.mixture_mean, pb.mixture_mean, VALUE_TOL)
            CHECK.check(
                f"{label}: mixture variance", pr.mixture_variance, pb.mixture_variance, VALUE_TOL
            )
            if Xt.dim() == 2:
                mb = pb.mixture_covariance_matrix
                CHECK.check(
                    f"{label}: mixture covariance",
                    pr.mixture_covariance_matrix,
                    mb.to_dense() if hasattr(mb, "to_dense") else mb,
                    VALUE_TOL,
                )


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("fixed,warp,transforms", VARIANTS)
def test_posterior_matches_botorch(model: str, fixed: bool, warp: bool, transforms: bool) -> None:
    bt, rs, X, _ = model_pair(model, seed=40, fixed=fixed, warp=warp, transforms=transforms)
    lo, hi = X.min(0).values, X.max(0).values
    g = _gen(41)
    for shape in ((4, 3), (5, 2, 3), (2, 3, 1, 3)):
        Xt = lo + (hi - lo) * torch.rand(*shape, generator=g, dtype=DT)
        _compare_posteriors(f"{model} posterior", bt, rs, Xt)
    Xt = lo + (hi - lo) * torch.rand(3, 3, generator=g, dtype=DT)
    with torch.no_grad():
        qb = bt.posterior(Xt).quantile(torch.tensor([0.1, 0.5, 0.9], dtype=DT))
    qr = rs.posterior(Xt).quantile(torch.tensor([0.1, 0.5, 0.9], dtype=DT))
    # Bisection to BoTorch's tolerance (TOL = 1e-6 on the CDF).
    CHECK.check(f"{model} mixture quantiles", qr, qb, 1e-6)


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("fixed,warp,transforms", VARIANTS)
def test_mll_matches_botorch(model: str, fixed: bool, warp: bool, transforms: bool) -> None:
    from gpytorch.mlls import ExactMarginalLogLikelihood as BtMLL

    bt, rs, _, _ = model_pair(model, seed=50, fixed=fixed, warp=warp, transforms=transforms)
    bt.train(reset=False)
    bt.zero_grad()
    mll = BtMLL(bt.likelihood, bt)
    vb = mll(bt(*bt.train_inputs), bt.train_targets)
    vb.sum().backward()
    gb = {
        n: (p.grad if p.grad is not None else torch.zeros_like(p)) for n, p in bt.named_parameters()
    }
    bt.eval()
    vr, gr = RsMLL(rs.likelihood, rs).value_and_grad()
    CHECK.check(f"{model} MLL of every sample", vr, vb.detach(), VALUE_TOL)
    names_rs = [n for n, _ in rs.named_parameters()]
    assert sorted(names_rs) == sorted(gb), (names_rs, list(gb))
    offset = 0
    flat_r, flat_b = [], []
    gr_t = _t(gr)
    for name, p in rs.named_parameters():
        size = _t(p).numel()
        flat_r.append(gr_t[offset : offset + size])
        flat_b.append(gb[name].reshape(-1))
        offset += size
    check_grad(CHECK, f"{model} MLL gradient", torch.cat(flat_r), torch.cat(flat_b))


def _compare_acqf(label: str, a_rs: Any, a_bt: Any, X: Any, log: bool = False) -> None:
    vb, gb = values_and_grads(a_bt, X)
    vr, gr = values_and_grads(a_rs, X)
    if log:
        CHECK.check_log(f"{label} values", vr, vb, VALUE_TOL)
    else:
        CHECK.check(f"{label} values", vr, vb, VALUE_TOL)
    check_grad(CHECK, f"{label} gradients", gr, gb, GRAD_TOL)


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize(
    "name",
    [
        "ExpectedImprovement",
        "LogExpectedImprovement",
        "ProbabilityOfImprovement",
        "LogProbabilityOfImprovement",
        "UpperConfidenceBound",
        "PosteriorMean",
    ],
)
def test_analytic_acquisition_matches_botorch(model: str, name: str) -> None:
    bt, rs, X, _ = model_pair(model, seed=60, warp=model == "saas")
    kw: dict[str, Any] = {}
    if name == "UpperConfidenceBound":
        kw["beta"] = torch.tensor(0.4, dtype=DT)
    elif name != "PosteriorMean":
        kw["best_f"] = torch.tensor(3.0, dtype=DT)
    a_bt = getattr(bt_an, name)(bt, **kw)
    a_rs = getattr(rs_an, name)(rs, **kw)
    lo, hi = X.min(0).values, X.max(0).values
    Xt = lo + (hi - lo) * torch.rand(5, 1, 3, generator=_gen(61), dtype=DT)
    _compare_acqf(f"{name} on {model}", a_rs, a_bt, Xt, log=name.startswith("Log"))


def test_posterior_standard_deviation_is_the_sample_average() -> None:
    """BoTorch's ``PosteriorStandardDeviation`` fails on ensemble models
    (phase 7b, bug 2); botorch_rs averages the samples' standard deviations,
    compared with that average computed from BoTorch's posterior."""
    bt, rs, X, _ = model_pair("saas", seed=62)
    lo, hi = X.min(0).values, X.max(0).values
    Xt = lo + (hi - lo) * torch.rand(5, 1, 3, generator=_gen(63), dtype=DT)
    with pytest.raises(RuntimeError):
        bt_an.PosteriorStandardDeviation(bt)(Xt)
    vr, gr = values_and_grads(rs_an.PosteriorStandardDeviation(rs), Xt)
    Xg = Xt.clone().requires_grad_(True)
    ref = bt.posterior(Xg).variance.sqrt().squeeze(-1).squeeze(-1).mean(-1)
    (gref,) = torch.autograd.grad(ref.sum(), Xg)
    CHECK.check("PosteriorStandardDeviation (sample average)", vr, ref.detach(), VALUE_TOL)
    check_grad(CHECK, "PosteriorStandardDeviation (sample average) gradients", gr, gref)


def _samplers(num: int, k: int, seed: int, X: Any = None, bt: Any = None) -> tuple[Any, Any]:
    """Samplers with the same base samples (of BoTorch's shape for the
    posterior of ``bt`` at ``X`` if given: one per t-batch-collapsed
    posterior point)."""
    if bt is not None:
        s_bt = BtSampler(sample_shape=torch.Size([num]), seed=seed)
        with torch.no_grad():
            s_bt(bt.posterior(X))
        s_rs = RsSampler(sample_shape=torch.Size([num]), seed=0)
        s_rs.base_samples = s_bt.base_samples.reshape(num, -1).clone()
        return s_bt, s_rs
    Z = torch.randn(num, 1, 1, k, generator=_gen(seed), dtype=DT)
    s_bt = BtSampler(sample_shape=torch.Size([num]), seed=0)
    s_bt.base_samples = Z.clone()
    s_rs = RsSampler(sample_shape=torch.Size([num]), seed=0)
    s_rs.base_samples = Z.reshape(num, -1).clone()
    return s_bt, s_rs


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize(
    "name",
    [
        "qLogExpectedImprovement",
        "qExpectedImprovement",
        "qProbabilityOfImprovement",
        "qSimpleRegret",
        "qUpperConfidenceBound",
        "qLogNoisyExpectedImprovement",
        "qNoisyExpectedImprovement",
    ],
)
def test_mc_acquisition_matches_botorch(model: str, name: str) -> None:
    bt, rs, X, _ = model_pair(model, seed=70, fixed=model == "matern")
    q = 2
    noisy = "Noisy" in name
    k = X.shape[0] + q if noisy else q
    s_bt, s_rs = _samplers(32, k, 71)
    kw: dict[str, Any] = {}
    if name in ("qLogExpectedImprovement", "qExpectedImprovement", "qProbabilityOfImprovement"):
        kw["best_f"] = torch.tensor(3.0, dtype=DT)
    if name == "qProbabilityOfImprovement":
        kw["tau"] = torch.tensor(0.05, dtype=DT)
    if name == "qUpperConfidenceBound":
        kw["beta"] = torch.tensor(0.4, dtype=DT)
    module = bt_logei if name.startswith("qLog") else bt_mc
    rmodule = rs_logei if name.startswith("qLog") else rs_mc
    bkw, rkw = dict(kw), dict(kw)
    if noisy:
        bkw.update(X_baseline=X, prune_baseline=False, cache_root=False)
        rkw.update(X_baseline=X, prune_baseline=False, cache_root=False)
    a_bt = getattr(module, name)(bt, sampler=s_bt, **bkw)
    a_rs = getattr(rmodule, name)(rs, sampler=s_rs, **rkw)
    lo, hi = X.min(0).values, X.max(0).values
    Xq = lo + (hi - lo) * torch.rand(1 if noisy else 3, q, 3, generator=_gen(72), dtype=DT)
    _compare_acqf(f"{name} on {model}", a_rs, a_bt, Xq, log=name.startswith("qLog"))


@pytest.mark.parametrize("model", MODELS)
def test_prune_inferior_points_matches_botorch(model: str) -> None:
    bt, rs, X, _ = model_pair(model, seed=80)
    n = X.shape[0]
    s_bt, s_rs = _samplers(256, n, 81, X=X, bt=bt)
    pb = bt_prune(bt, X, sampler=s_bt)
    pr = rs_prune(rs, X, sampler=s_rs)
    CHECK.check("prune_inferior_points: kept points", pr, pb, 0.0)


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("fixed", [False, True])
def test_condition_on_observations_matches_botorch(model: str, fixed: bool) -> None:
    """Identical observations for every sample and one set per sample (X in
    the transformed input space, as in BoTorch). BoTorch fails to repeat the
    noise of identical observations of a fixed-noise model (see the draft);
    its reference gets the repeated noise."""
    bt, rs, X, _ = model_pair(model, seed=90, fixed=fixed)
    lo, hi = X.min(0).values, X.max(0).values
    g = _gen(91)
    num = rs.num_mcmc_samples
    Xn = lo + (hi - lo) * torch.rand(2, 3, generator=g, dtype=DT)
    Yn = 2.0 + torch.randn(2, 1, generator=g, dtype=DT)
    with torch.no_grad():
        bt.posterior(Xn)  # BoTorch needs its prediction caches
    Xn_tf = bt.transform_inputs(Xn)
    kb: dict[str, Any] = {}
    kr: dict[str, Any] = {}
    if fixed:
        noise = torch.full((2, 1), 0.02, dtype=DT)
        kb["noise"] = noise.repeat(num, 1, 1)
        kr["noise"] = noise
    cb = bt.condition_on_observations(Xn_tf, Yn, **kb)
    cr = rs.condition_on_observations(Xn_tf, Yn, **kr)
    Xt = lo + (hi - lo) * torch.rand(4, 3, generator=g, dtype=DT)
    _compare_posteriors(f"{model} conditioned", cb, cr, Xt)
    # One set of observations per sample.
    Xs = lo + (hi - lo) * torch.rand(num, 2, 3, generator=g, dtype=DT)
    Ys = 2.0 + torch.randn(num, 2, 1, generator=g, dtype=DT)
    kw2: dict[str, Any] = {}
    if fixed:
        kw2["noise"] = torch.full((num, 2, 1), 0.02, dtype=DT)
    Xs_tf = bt.transform_inputs(Xs)
    cb = bt.condition_on_observations(Xs_tf, Ys, **kw2)
    cr = rs.condition_on_observations(Xs_tf, Ys, **kw2)
    _compare_posteriors(f"{model} conditioned per sample", cb, cr, Xt)


def test_botorch_fails_to_condition_fixed_noise_models() -> None:
    """BoTorch bug: ``condition_on_observations`` of a fixed-noise fully
    Bayesian GP repeats ``X`` and ``Y`` over the samples but not ``noise``
    (GPyTorch's ``get_fantasy_likelihood`` fails)."""
    bt, _, X, _ = model_pair("saas", seed=92, fixed=True, transforms=False)
    with torch.no_grad():
        bt.posterior(X[:2])
    with pytest.raises(RuntimeError, match="expand"):
        bt.condition_on_observations(
            X[:2], torch.zeros(2, 1, dtype=DT), noise=torch.full((2, 1), 0.01, dtype=DT)
        )


@pytest.mark.parametrize("model", MODELS)
def test_no_fantasize_as_botorch(model: str) -> None:
    """BoTorch's fully Bayesian single-task GPs are not a
    ``FantasizeMixin``: no ``fantasize`` (so the knowledge gradient does
    not support them); botorch_rs's neither."""
    bt, rs, _, _ = model_pair(model, seed=100)
    assert not hasattr(bt, "fantasize")
    assert not hasattr(rs, "fantasize")


@pytest.mark.parametrize("model", MODELS)
def test_attributes_match_botorch(model: str) -> None:
    for num in (4, 5):
        bt, rs, _, _ = model_pair(model, seed=110 + num, num=num, warp=True)
        assert rs.num_mcmc_samples == bt.num_mcmc_samples == num
        assert tuple(rs.batch_shape) == tuple(bt.batch_shape)
        attr = "median_weight_variance" if model == "linear" else "median_lengthscale"
        CHECK.check(f"{model} {attr}", getattr(rs, attr), getattr(bt, attr).detach(), VALUE_TOL)
        names_b = sorted((n, tuple(p.shape)) for n, p in bt.named_parameters())
        names_r = sorted((n, tuple(_t(p).shape)) for n, p in rs.named_parameters())
        assert names_r == names_b
        assert rs._is_fully_bayesian and rs._is_ensemble
        assert not rs._supports_batched_models


def test_construct_inputs_matches_botorch() -> None:
    X, Y, Yvar = make_fb_data({"n": 6, "d": 2}, 120)
    names = ["a", "b"]
    db = BtDataset(X, Y, feature_names=names, outcome_names=["y"], Yvar=Yvar)
    dr = RsDataset(X, Y, feature_names=names, outcome_names=["y"], Yvar=Yvar)
    for model in MODELS:
        from fully_bayesian_helpers import bt_class

        kb = bt_class(model).construct_inputs(db, use_input_warping=True, indices_to_warp=[1])
        kr = rs_class(model).construct_inputs(dr, use_input_warping=True, indices_to_warp=[1])
        assert sorted(kb) == sorted(kr)
        for key in kb:
            if isinstance(kb[key], torch.Tensor):
                CHECK.check("construct_inputs", kr[key], kb[key], 0.0)
            else:
                assert kr[key] == kb[key]


def test_matern52_and_linear_kernel_helpers_match_botorch() -> None:
    import jax.numpy as jnp
    from botorch.models import fully_bayesian as bt_fb

    from botorch_rs.models import fully_bayesian as rs_fb

    g = _gen(130)
    X = torch.rand(5, 3, generator=g, dtype=DT)
    ls = 0.3 + torch.rand(3, generator=g, dtype=DT)
    from fully_bayesian_helpers import jax_x64

    with jax_x64(True):
        Xj, lj = jnp.asarray(X.numpy()), jnp.asarray(ls.numpy())
        for fn in ("compute_dists", "matern52_kernel", "linear_kernel"):
            e = torch.as_tensor(np.asarray(getattr(bt_fb, fn)(Xj, lj)))
            a = getattr(rs_fb, fn)(X, ls)
            CHECK.check(f"{fn}", a, e, VALUE_TOL)
    t = torch.zeros(2, 3, dtype=DT)
    v = torch.arange(6, dtype=DT)
    CHECK.check(
        "reshape_and_detach", rs_fb.reshape_and_detach(t, v), bt_fb.reshape_and_detach(t, v), 0.0
    )
    assert rs_fb.MCMC_DIM == bt_fb.MCMC_DIM

"""Parity of the fully Bayesian GPs of phase 10a with BoTorch, through the
public APIs of both libraries side by side:

* the potential energies of the NumPyro models (``SaasPyroModel``,
  ``MaternPyroModel``, ``LinearPyroModel``; inferred or fixed noise, input
  warping) against NumPyro's ``potential_energy`` and its JAX gradient (in
  double precision);
* ``fit_fully_bayesian_model_nuts`` replaying NumPyro's recorded random
  draws (``draws=``, ``init_params=``): NumPyro's random decisions (exact),
  the chain within 10x BoTorch's own change under perturbations of the
  training inputs at 1e-14 and 1e-13 (six twins), and the loaded samples;
* the models built from given MCMC samples (``load_mcmc_samples``):
  posteriors (with and without observation noise, batched inputs), the
  mixture moments and quantiles, the MLL of every sample and its gradient,
  the acquisition functions averaged over the samples (analytic and Monte
  Carlo, values and input gradients), ``prune_inferior_points``,
  ``condition_on_observations``, ``fantasize``, ``median_lengthscale``,
  ``median_weight_variance``, ``num_mcmc_samples``, ``batch_shape`` and
  ``construct_inputs``; with ``Normalize`` / ``Standardize``, fixed noise
  and input warping;
* ``qBayesianActiveLearningByDisagreement`` (BoTorch's base samples);
* ``SaasFullyBayesianMultiTaskGP``: potentials, NUTS replays (as long as
  BoTorch's twins follow NumPyro), posteriors, conditioning and acquisition
  functions; the mixture covariance of several outputs;
* lists of fully Bayesian GPs in qEHVI / qNEHVI / qLogEHVI / qLogNEHVI,
  ``prune_inferior_points_multi_objective`` and ``qLogNParEGO``;
* ``PathwiseThompsonSampling`` of fully Bayesian GPs and lists of them
  (BoTorch's path draws and ensemble indices), the errors of the cases
  BoTorch fails on, and qJES's ``NotImplementedError``.

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
    TWIN_SCALES,
    bt_fit_recorded,
    bt_model,
    fb_cases,
    first_difference,
    jax_x64,
    make_fb_data,
    model_pair,
    mt_data,
    mt_model_pair,
    numpyro_potential,
    perturbed,
    raw_box_data,
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


#: BoTorch's twins of a NUTS replay (training inputs perturbed alternately
#: at 1e-14 and 1e-13): fewer twins underestimate BoTorch's spread (with 3
#: twins one sweep chain measured 7.9x their spread, with 8 twins 1.3x).
NUM_TWINS = 6


@pytest.mark.parametrize("case", _replay_cases())
def test_nuts_replay_matches_numpyro(case: dict[str, Any]) -> None:
    """botorch_rs's NUTS on NumPyro's recorded draws retraces BoTorch's
    chain (JAX in double precision)."""
    X, Y, Yvar = make_fb_data(case, 21)
    kw = {"warmup_steps": case["warmup"], "num_samples": case["samples"], "thinning": 2}
    bt = bt_model(case, X, Y, Yvar)
    rec = bt_fit_recorded(bt, fixed_warping=True, **kw)
    spread = [0.0] * len(rec.states)
    for t in range(NUM_TWINS):
        twin = bt_model(case, perturbed(X, TWIN_SCALES[t % 2], 300 + t), Y, Yvar)
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
            if Xt.dim() == 2 and pb.mean.shape[-1] == 1:
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


@pytest.mark.parametrize("indices", [None, [0, 2]])
def test_pyro_model_warp_matches_botorch(indices: Any) -> None:
    """``PyroModel.warp`` (BoTorch's ``kumaraswamy_warp`` of the columns
    ``indices_to_warp``)."""
    from botorch.models import fully_bayesian as bt_fb

    from botorch_rs.models import fully_bayesian as rs_fb

    g = _gen(135)
    X = torch.rand(2, 5, 3, generator=g, dtype=DT)
    k = 3 if indices is None else len(indices)
    c0 = 0.5 + 2 * torch.rand(k, generator=g, dtype=DT)
    c1 = 0.5 + 2 * torch.rand(k, generator=g, dtype=DT)
    pb = bt_fb.SaasPyroModel(use_input_warping=True, indices_to_warp=indices, eps=1e-6)
    pr = rs_fb.SaasPyroModel(use_input_warping=True, indices_to_warp=indices, eps=1e-6)
    CHECK.check("PyroModel.warp", pr.warp(X, c0=c0, c1=c1), pb.warp(X, c0=c0, c1=c1), VALUE_TOL)


@pytest.mark.parametrize("kernel", ["compute_dists", "matern52_kernel", "linear_kernel"])
@pytest.mark.parametrize("scale_shape", [(3,), (1, 3), (4, 1, 3), (2, 4, 5, 3), ()])
def test_batched_kernel_helpers_match_botorch(kernel: str, scale_shape: tuple[int, ...]) -> None:
    """Arbitrary batch and row broadcasting use the same native kernel as
    unbatched helpers; invalid feature dimensions must not wrap around."""
    import jax.numpy as jnp
    from botorch.models import fully_bayesian as bt_fb

    from botorch_rs.models import fully_bayesian as rs_fb

    g = _gen(136)
    X = torch.rand(2, 1, 5, 3, generator=g, dtype=DT)
    scale = 0.2 + torch.rand(scale_shape, generator=g, dtype=DT)
    with jax_x64():
        expected = getattr(bt_fb, kernel)(jnp.asarray(X.numpy()), jnp.asarray(scale.numpy()))
    actual = getattr(rs_fb, kernel)(X, scale)
    CHECK.check(f"batched {kernel}", actual, np.asarray(expected), VALUE_TOL)


@pytest.mark.parametrize("batch_shape", [(), (1,), (2, 1), (4, 1, 1)])
def test_warp_forward_broadcasts_like_botorch(batch_shape: tuple[int, ...]) -> None:
    from botorch.models.transforms.input import Warp as BtWarp

    from botorch_rs.models.transforms.input import Warp as RsWarp

    bounds = torch.tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], dtype=DT)
    kwargs = {"d": 3, "indices": [1], "batch_shape": (2, 1), "bounds": bounds}
    bt, rs = BtWarp(**kwargs), RsWarp(**kwargs)
    c0 = torch.tensor([0.4, 2.0], dtype=DT).reshape(2, 1, 1, 1)
    c1 = torch.tensor([0.8, 1.5], dtype=DT).reshape(2, 1, 1, 1)
    bt.concentration0.data = c0.clone()
    bt.concentration1.data = c1.clone()
    rs.concentration0, rs.concentration1 = c0, c1
    X = torch.rand(*batch_shape, 5, 3, generator=_gen(137), dtype=DT)
    CHECK.check("Warp forward batch broadcasting", rs(X), bt(X), VALUE_TOL)


# ---------------------------------------------------------------------------
# Bayesian active learning
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("q,pending", [(1, False), (2, False), (2, True)])
def test_bald_matches_botorch(model: str, q: int, pending: bool) -> None:
    from botorch.acquisition.bayesian_active_learning import (
        qBayesianActiveLearningByDisagreement as BtBALD,
    )

    from botorch_rs.acquisition.bayesian_active_learning import (
        qBayesianActiveLearningByDisagreement as RsBALD,
    )

    bt, rs, X, _ = model_pair(model, seed=140, num=4, fixed=model == "linear")
    lo, hi = X.min(0).values, X.max(0).values
    g = _gen(141)
    Xt = lo + (hi - lo) * torch.rand(3, q, 3, generator=g, dtype=DT)
    Xp = lo + (hi - lo) * torch.rand(2, 3, generator=g, dtype=DT) if pending else None
    Xfull = Xt if Xp is None else torch.cat([Xt, Xp.expand(3, 2, 3)], dim=-2)
    s_bt, s_rs = _samplers(64, Xfull.shape[-2], 142, X=Xfull, bt=bt)
    a_bt = BtBALD(bt, sampler=s_bt, X_pending=Xp)
    a_rs = RsBALD(rs, sampler=s_rs, X_pending=Xp)
    _compare_acqf(f"qBayesianActiveLearningByDisagreement on {model}", a_rs, a_bt, Xt)


def test_bald_requires_a_fully_bayesian_model() -> None:
    from botorch_rs.acquisition.bayesian_active_learning import (
        FULLY_BAYESIAN_ERROR_MSG,
        qBayesianActiveLearningByDisagreement,
    )
    from botorch_rs.models import SingleTaskGP

    X, Y, _ = make_fb_data({"n": 6, "d": 2}, 143)
    with pytest.raises(RuntimeError, match="Fully Bayesian acquisition functions require"):
        qBayesianActiveLearningByDisagreement(SingleTaskGP(X, Y))
    assert "SaasFullyBayesianSingleTaskGP" in FULLY_BAYESIAN_ERROR_MSG


# ---------------------------------------------------------------------------
# Multi-task models
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "fixed,rank,output_tasks,transforms,tasks",
    [
        (False, 2, None, False, (0, 1, 2)),
        (True, 1, None, False, (0, 1, 2)),
        (False, 3, [1], True, (0, 1, 2)),
        (False, 2, [3], False, (1, 3, 5)),
    ],
)
def test_multitask_posterior_matches_botorch(
    fixed: bool, rank: int, output_tasks: Any, transforms: bool, tasks: Any
) -> None:
    bt, rs, X, _ = mt_model_pair(
        150, fixed=fixed, rank=rank, output_tasks=output_tasks, transforms=transforms, tasks=tasks
    )
    g = _gen(151)
    lo, hi = X[:, :-1].min(0).values, X[:, :-1].max(0).values
    for shape in ((4,), (3, 2)):
        Xt = lo + (hi - lo) * torch.rand(*shape, X.shape[-1] - 1, generator=g, dtype=DT)
        _compare_posteriors("multi-task posterior (output tasks)", bt, rs, Xt)
    Xd = lo + (hi - lo) * torch.rand(4, X.shape[-1] - 1, generator=g, dtype=DT)
    t = torch.tensor([[float(tasks[i % len(tasks)])] for i in range(4)], dtype=DT)
    _compare_posteriors("multi-task posterior (task column)", bt, rs, torch.cat([Xd, t], -1))
    assert rs.num_mcmc_samples == bt.num_mcmc_samples
    assert tuple(rs.batch_shape) == tuple(bt.batch_shape)
    CHECK.check(
        "multi-task median_lengthscale",
        rs.median_lengthscale,
        bt.median_lengthscale.detach(),
        VALUE_TOL,
    )


@pytest.mark.parametrize("name", ["qLogExpectedImprovement", "qLogNoisyExpectedImprovement"])
def test_multitask_acquisition_matches_botorch(name: str) -> None:
    bt, rs, X, _ = mt_model_pair(160, output_tasks=[0])
    g = _gen(161)
    q = 2
    Xb = X[X[:, -1] == 0][:, :-1]
    noisy = "Noisy" in name
    k = Xb.shape[0] + q if noisy else q
    s_bt, s_rs = _samplers(32, k, 162)
    kw: dict[str, Any] = {} if noisy else {"best_f": torch.tensor(0.5, dtype=DT)}
    if noisy:
        kw.update(X_baseline=Xb, prune_baseline=False, cache_root=False)
    a_bt = getattr(bt_logei, name)(bt, sampler=s_bt, **kw)
    a_rs = getattr(rs_logei, name)(rs, sampler=s_rs, **kw)
    Xq = torch.rand(1 if noisy else 3, q, X.shape[-1] - 1, generator=g, dtype=DT)
    _compare_acqf(f"{name} on the multi-task model", a_rs, a_bt, Xq, log=True)


def test_multitask_conditioning_matches_botorch() -> None:
    bt, rs, _, _ = mt_model_pair(170)
    g = _gen(171)
    tasks = torch.tensor([[0.0], [2.0]], dtype=DT)
    Xn = torch.cat([torch.rand(2, 2, generator=g, dtype=DT), tasks], -1)
    Yn = torch.randn(2, 1, generator=g, dtype=DT)
    with torch.no_grad():
        bt.posterior(Xn)
    cb = bt.condition_on_observations(bt.transform_inputs(Xn), Yn)
    cr = rs.condition_on_observations(Xn, Yn)
    Xt = torch.rand(3, 2, generator=g, dtype=DT)
    _compare_posteriors("multi-task conditioned", cb, cr, Xt)


def test_multitask_potential_matches_numpyro() -> None:
    from botorch.models.fully_bayesian_multitask import SaasFullyBayesianMultiTaskGP as Bt

    from botorch_rs.models.fully_bayesian_multitask import SaasFullyBayesianMultiTaskGP as Rs

    for fixed, tasks in ((False, (0, 1, 2)), (True, (2, 4))):
        X, Y, Yvar = mt_data(180, tasks=tasks)
        Y = (Y - Y.mean()) / Y.std()
        yvar = Yvar if fixed else None
        with jax_x64(True):
            bt = Bt(X, Y, task_feature=-1, train_Yvar=yvar, rank=2)
        rs = Rs(X, Y, task_feature=-1, train_Yvar=yvar, rank=2)
        pot = rs.pyro_model._potential()
        g = _gen(181)
        for _ in range(3):
            z = (4 * torch.rand(pot.dim, generator=g, dtype=DT) - 2).tolist()
            vb, gb = numpyro_potential(bt, z)
            vr, gr = pot.potential(z)
            CHECK.check("multi-task potential energy", [vr], [vb], VALUE_TOL)
            check_grad(CHECK, "multi-task potential energy gradient", gr, gb)


def test_multitask_nuts_replay_follows_numpyro_as_long_as_botorch_twins() -> None:
    """The chains of the multi-task model are chaotic from their first
    iterations: BoTorch's own chain makes other random decisions when its
    training inputs change by 1e-14 (twins). botorch_rs's replay must follow
    NumPyro's decisions at least as long as the twins do, with its states
    within 10x the twins' spread up to there."""
    from botorch.models.fully_bayesian_multitask import SaasFullyBayesianMultiTaskGP as Bt

    from botorch_rs.models.fully_bayesian_multitask import SaasFullyBayesianMultiTaskGP as Rs

    X, Y, _ = mt_data(190, n=14, tasks=(0, 1))
    Y = (Y - Y.mean()) / Y.std()
    kw = {"warmup_steps": 20, "num_samples": 8, "thinning": 2}

    def run(Xi: Any) -> Any:
        with jax_x64(True):
            bt = Bt(Xi, Y, task_feature=-1, rank=2)
        return bt_fit_recorded(bt, **kw)

    rec = run(X)
    twins = [run(perturbed(X, 1e-14, 400 + t)) for t in range(2)]
    agree = min(first_difference(rec.draws, tw.draws) for tw in twins)
    rs = Rs(X, Y, task_feature=-1, rank=2)
    out = rs.pyro_model._potential().sample_nuts(
        20, 8, z0=rec.z0, draws=rec.draws, stop_when_draws_fail=True
    )
    # (A draw of the wrong kind stops the replay after consuming it.)
    consumed = out["consumed"] - (out["stopped_at"] is not None)
    rs_agree = consumed if out["first_flip"] is None else min(out["first_flip"], consumed)
    CHECK.check_bound(
        "multi-task NUTS replay: draws the twins follow beyond botorch_rs", agree - rs_agree, 0
    )
    # The states of the iterations whose draws the twins share with NumPyro.
    used = 1  # the momentum of NumPyro's init_kernel
    worst = 0.0
    for t, (it, (_, zb, _)) in enumerate(zip(out["trace"], rec.states, strict=False)):
        nxt = used + 1
        while nxt < len(rec.draws) and rec.draws[nxt][0] == "u":
            nxt += 1
        if nxt > agree:
            break
        used = nxt
        spread = max(
            float(np.max(np.abs(np.subtract(rec.states[t][1], tw.states[t][1])))) for tw in twins
        )
        err = float(np.max(np.abs(np.subtract(it[0], zb))))
        worst = max(worst, err / max(10 * spread, 1e-12))
    CHECK.check_bound("multi-task NUTS replay: states / (10x twin spread)", worst, 1.0)


def test_mixture_covariance_of_several_outputs() -> None:
    """Compare the mixture covariance in output-major order. Older BoTorch
    releases fail on multi-output mixtures; compare their components too."""
    bt, rs, X, _ = mt_model_pair(200)
    Xt = torch.rand(3, 2, generator=_gen(201), dtype=DT)
    with torch.no_grad():
        pb = bt.posterior(Xt)
        mean = pb.mean  # S x q x m
        cov = pb.distribution.covariance_matrix  # S x (q m) x (q m), output-major
        diff = (mean - mean.mean(0)).transpose(-1, -2).reshape(mean.shape[0], -1)
        ref = cov.mean(0) + (diff.unsqueeze(-1) * diff.unsqueeze(-2)).mean(0)
        try:
            upstream_cov = pb.mixture_covariance_matrix
        except RuntimeError:
            # BoTorch 0.18.1 adds q x q mean outer products to q*m x q*m
            # covariance matrices; the audited local fork fixes flattening.
            upstream_cov = None
        if upstream_cov is not None:
            if hasattr(upstream_cov, "to_dense"):
                upstream_cov = upstream_cov.to_dense()
            CHECK.check(
                "BoTorch mixture covariance of several outputs", upstream_cov, ref, VALUE_TOL
            )
    CHECK.check(
        "mixture covariance of several outputs",
        rs.posterior(Xt).mixture_covariance_matrix,
        ref,
        VALUE_TOL,
    )


# ---------------------------------------------------------------------------
# Multi-objective acquisition functions (lists of fully Bayesian GPs)
# ---------------------------------------------------------------------------


def _fb_list_pair(seed: int, models: tuple[str, str] = ("saas", "matern")) -> tuple[Any, Any, Any]:
    """``ModelListGP``s of two fully Bayesian GPs (same training inputs, the
    same number of MCMC samples) in BoTorch and botorch_rs."""
    from botorch.models import ModelListGP as BtList

    from botorch_rs.models import ModelListGP as RsList

    bts, rss = [], []
    X = None
    for k, model in enumerate(models):
        bt, rs, X, _ = model_pair(model, seed=seed + 10 * k, num=4, transforms=False)
        bts.append(bt)
        rss.append(rs)
    # The same training inputs for both outputs (BoTorch's ModelListGP
    # broadcasts the models' posteriors).
    return BtList(*bts), RsList(*rss), X


@pytest.mark.parametrize("log", [False, True])
@pytest.mark.parametrize("noisy,prune", [(False, False), (True, False), (True, True)])
def test_multi_objective_acquisition_on_fully_bayesian_list(
    log: bool, noisy: bool, prune: bool
) -> None:
    """qEHVI / qNEHVI (and the log versions) on a ``ModelListGP`` of fully
    Bayesian GPs: averaged over the samples (``marginalize_dim=-3``), with
    BoTorch's base samples (``synced_draws``)."""
    from botorch.acquisition.multi_objective import logei as bt_mo_logei
    from botorch.acquisition.multi_objective import monte_carlo as bt_mo_mc
    from botorch.utils.multi_objective.box_decompositions.non_dominated import (
        NondominatedPartitioning as BtPartitioning,
    )
    from mo_acquisition_helpers import bt_sampler, rs_sampler, synced_draws

    from botorch_rs.acquisition.multi_objective import logei as rs_mo_logei
    from botorch_rs.acquisition.multi_objective import monte_carlo as rs_mo_mc
    from botorch_rs.utils.multi_objective.box_decompositions.non_dominated import (
        NondominatedPartitioning as RsPartitioning,
    )

    bt, rs, X = _fb_list_pair(210)
    with torch.no_grad():
        means = bt.posterior(X).mixture_mean
    ref = means.min(0).values - 0.1 * (means.max(0).values - means.min(0).values)
    q, n, m = 2, X.shape[0], 2
    num = 16
    name = ("qLog" if log else "q") + (
        "NoisyExpectedHypervolumeImprovement" if noisy else "ExpectedHypervolumeImprovement"
    )
    mod_bt = bt_mo_logei if log else bt_mo_mc
    mod_rs = rs_mo_logei if log else rs_mo_mc
    kw: dict[str, Any] = {"ref_point": ref}
    if noisy:
        kw.update(X_baseline=X, prune_baseline=prune, cache_root=False, marginalize_dim=-3)
    with synced_draws(211):
        bkw = dict(kw)
        if not noisy:
            bkw["partitioning"] = BtPartitioning(ref_point=ref, Y=means)
        a_bt = getattr(mod_bt, name)(bt, sampler=bt_sampler(num), **bkw)
    rkw = dict(kw)
    if not noisy:
        rkw["partitioning"] = RsPartitioning(ref_point=ref, Y=means)
    import botorch_rs.acquisition.multi_objective.utils as rs_mo_utils

    # The pruning draws BoTorch's base samples too (2048 samples).
    orig = rs_mo_utils.SobolQMCNormalSampler
    rs_mo_utils.SobolQMCNormalSampler = lambda sample_shape: rs_sampler(  # type: ignore[assignment]
        sample_shape[0], 211, n * m
    )
    try:
        a_rs = getattr(mod_rs, name)(rs, sampler=rs_sampler(num, 211, (n + q) * m), **rkw)
    finally:
        rs_mo_utils.SobolQMCNormalSampler = orig  # type: ignore[assignment]
    lo, hi = X.min(0).values, X.max(0).values
    Xt = lo + (hi - lo) * torch.rand(2, q, X.shape[-1], generator=_gen(212), dtype=DT)
    with synced_draws(211):
        vb, gb = values_and_grads(a_bt, Xt)
    vr, gr = values_and_grads(a_rs, Xt)
    if log:
        CHECK.check_log(f"{name} on a list of fully Bayesian GPs values", vr, vb, VALUE_TOL)
    else:
        CHECK.check(f"{name} on a list of fully Bayesian GPs values", vr, vb, VALUE_TOL)
    check_grad(CHECK, f"{name} on a list of fully Bayesian GPs gradients", gr, gb)
    if noisy:
        CHECK.check(f"{name}: baseline (pruned={prune})", a_rs.X_baseline, a_bt.X_baseline, 0.0)


def test_prune_inferior_points_multi_objective_on_fully_bayesian_list() -> None:
    from botorch.acquisition.multi_objective.utils import (
        prune_inferior_points_multi_objective as bt_prune_mo,
    )
    from mo_acquisition_helpers import rs_sampler, synced_draws

    from botorch_rs.acquisition.multi_objective.utils import (
        prune_inferior_points_multi_objective as rs_prune_mo,
    )

    bt, rs, X = _fb_list_pair(220)
    with torch.no_grad():
        means = bt.posterior(X).mixture_mean
    ref = means.min(0).values - 0.1 * (means.max(0).values - means.min(0).values)
    n = X.shape[0]
    with synced_draws(221):
        pb = bt_prune_mo(bt, X, ref, num_samples=64, marginalize_dim=-3)
    pr = rs_prune_mo(rs, X, ref, marginalize_dim=-3, sampler=rs_sampler(64, 221, 2 * n))
    CHECK.check("prune_inferior_points_multi_objective (fully Bayesian list)", pr, pb, 0.0)


@pytest.mark.parametrize("prune", [False, True])
def test_qlognparego_on_fully_bayesian_list(prune: bool) -> None:
    """qLogNParEGO on a ``ModelListGP`` of fully Bayesian GPs: one Chebyshev
    scalarization normalized by the ensemble's ``Y_baseline`` (the mean over
    the samples), the baseline pruned under it, values averaged over the
    samples (``logmeanexp``)."""
    from botorch.acquisition.multi_objective.parego import qLogNParEGO as BtParEGO
    from mo_acquisition_helpers import bt_sampler, rs_sampler, synced_draws

    import botorch_rs.acquisition.multi_objective.utils as rs_mo_utils
    from botorch_rs.acquisition.multi_objective.parego import qLogNParEGO as RsParEGO

    bt, rs, X = _fb_list_pair(230)
    q, n, m, num = 2, X.shape[0], 2, 16
    w = torch.tensor([0.3, 0.7], dtype=DT)
    kw: dict[str, Any] = {"scalarization_weights": w, "prune_baseline": prune}
    with synced_draws(231):
        a_bt = BtParEGO(bt, X, sampler=bt_sampler(num), cache_root=False, **kw)
    orig = rs_mo_utils.SobolQMCNormalSampler
    # The pruning draws BoTorch's base samples too (2048 samples).
    rs_mo_utils.SobolQMCNormalSampler = lambda sample_shape: rs_sampler(  # type: ignore[assignment]
        sample_shape[0], 231, n * m
    )
    try:
        a_rs = RsParEGO(rs, X, sampler=rs_sampler(num, 231, (n + q) * m), **kw)
    finally:
        rs_mo_utils.SobolQMCNormalSampler = orig  # type: ignore[assignment]
    CHECK.check(
        "qLogNParEGO (fully Bayesian list): Y_baseline", a_rs.Y_baseline, a_bt.Y_baseline, VALUE_TOL
    )
    CHECK.check(
        f"qLogNParEGO (fully Bayesian list): baseline (pruned={prune})",
        a_rs.X_baseline,
        a_bt.X_baseline,
        0.0,
    )
    lo, hi = X.min(0).values, X.max(0).values
    Xt = lo + (hi - lo) * torch.rand(3, q, X.shape[-1], generator=_gen(232), dtype=DT)
    with synced_draws(231):
        vb, gb = values_and_grads(a_bt, Xt)
    vr, gr = values_and_grads(a_rs, Xt)
    CHECK.check_log("qLogNParEGO on a list of fully Bayesian GPs values", vr, vb, VALUE_TOL)
    check_grad(CHECK, "qLogNParEGO on a list of fully Bayesian GPs gradients", gr, gb)


# ---------------------------------------------------------------------------
# Pathwise Thompson sampling on ensembles
# ---------------------------------------------------------------------------


def _ensemble_path_draws(paths: Any, noises: list[Any]) -> list[tuple[Any, Any, Any]]:
    """BoTorch's random numbers of the Matheron paths of a batched (ensemble)
    model or a list of them, per member and output (member-major):
    frequencies (``L/2 x d``), weights (``q x L``) and update noise
    (``q x n``) of member ``s`` (BoTorch's batch index)."""
    from mo_entropy_helpers import matheron_models

    per_output = []
    for i, sub in enumerate(matheron_models(paths)):
        prior = sub.paths["prior_paths"]
        freq = prior.feature_map.weight.detach()  # S x L/2 x d
        weight = prior.weight.detach()  # q x S x L
        per_output.append((freq, weight, noises[i]))  # noise: q x S x n
    num_members = per_output[0][0].shape[0]
    return [
        (freq[s], weight[:, s, :], noise[:, s, :])
        for s in range(num_members)
        for freq, weight, noise in per_output
    ]


@pytest.mark.parametrize("model,transforms", [("saas", True), ("matern", False), ("saas", False)])
@pytest.mark.parametrize("q", [1, 2, 3])
def test_pathwise_thompson_sampling_on_fully_bayesian_gp(
    model: str, transforms: bool, q: int
) -> None:
    """PathwiseThompsonSampling of a fully Bayesian GP: paths of every
    sample, path ``j`` of the sample ``ensemble_indices[j]`` (BoTorch's path
    draws and ensemble indices)."""
    from botorch.acquisition.thompson_sampling import PathwiseThompsonSampling as BtTS
    from mo_entropy_helpers import recorded_randn

    from botorch_rs.acquisition.thompson_sampling import PathwiseThompsonSampling as RsTS

    bt, rs, X, _ = model_pair(model, seed=240 + q, num=4, transforms=transforms)
    torch.manual_seed(241 + q)
    a_bt = BtTS(bt)
    noises: list[Any] = []
    with recorded_randn(noises):
        a_bt.redraw(q)
    a_bt.batch_size = q
    draws = _ensemble_path_draws(a_bt.samples._path, noises)
    a_rs = RsTS(rs, num_features=draws[0][0].shape[0] * 2)
    a_rs.redraw(q, draws=draws, ensemble_indices=a_bt.ensemble_indices)
    CHECK.check(
        "PathwiseThompsonSampling (fully Bayesian): ensemble indices",
        a_rs.ensemble_indices.double(),
        a_bt.ensemble_indices.double(),
        0.0,
    )
    lo, hi = X.min(0).values, X.max(0).values
    Xt = lo + (hi - lo) * torch.rand(4, q, X.shape[-1], generator=_gen(242 + q), dtype=DT)
    vb, gb = values_and_grads(a_bt, Xt)
    vr, gr = values_and_grads(a_rs, Xt)
    CHECK.check(f"PathwiseThompsonSampling (fully Bayesian {model}) values", vr, vb, VALUE_TOL)
    check_grad(CHECK, f"PathwiseThompsonSampling (fully Bayesian {model}) gradients", gr, gb)


def test_pathwise_thompson_sampling_on_fully_bayesian_list() -> None:
    """PathwiseThompsonSampling of a ``ModelListGP`` of fully Bayesian GPs
    with a scalarizing posterior transform: the ensemble index of every path
    and output."""
    from botorch.acquisition.objective import ScalarizedPosteriorTransform as BtSPT
    from botorch.acquisition.thompson_sampling import PathwiseThompsonSampling as BtTS
    from mo_entropy_helpers import recorded_randn

    from botorch_rs.acquisition.objective import ScalarizedPosteriorTransform as RsSPT
    from botorch_rs.acquisition.thompson_sampling import PathwiseThompsonSampling as RsTS

    bt, rs, X = _fb_list_pair(250)
    w = torch.tensor([0.4, -0.6], dtype=DT)
    q = 3
    torch.manual_seed(251)
    a_bt = BtTS(bt, posterior_transform=BtSPT(w))
    noises: list[Any] = []
    with recorded_randn(noises):
        a_bt.redraw(q)
    a_bt.batch_size = q
    draws = _ensemble_path_draws(a_bt.samples._path, noises)
    a_rs = RsTS(rs, posterior_transform=RsSPT(w), num_features=draws[0][0].shape[0] * 2)
    a_rs.redraw(q, draws=draws, ensemble_indices=a_bt.ensemble_indices)
    lo, hi = X.min(0).values, X.max(0).values
    Xt = lo + (hi - lo) * torch.rand(4, q, X.shape[-1], generator=_gen(252), dtype=DT)
    vb, gb = values_and_grads(a_bt, Xt)
    vr, gr = values_and_grads(a_rs, Xt)
    CHECK.check("PathwiseThompsonSampling (fully Bayesian list) values", vr, vb, VALUE_TOL)
    check_grad(CHECK, "PathwiseThompsonSampling (fully Bayesian list) gradients", gr, gb)


@pytest.mark.parametrize("model,warp", [("linear", False), ("saas", True)])
def test_pathwise_thompson_sampling_fails_on_linear_and_warped_fully_bayesian_gps(
    model: str, warp: bool
) -> None:
    """BoTorch has no random features of the linear kernel
    (``NotImplementedError``) and fails on the batched input warping of a
    fully Bayesian GP (``RuntimeError``, see the draft); botorch_rs's paths
    support neither linear kernels nor input transforms other than
    ``Normalize`` (``UnsupportedError``, raised for the input transforms of
    these models)."""
    from botorch.acquisition.thompson_sampling import PathwiseThompsonSampling as BtTS

    from botorch_rs.acquisition.thompson_sampling import PathwiseThompsonSampling as RsTS
    from botorch_rs.exceptions.errors import UnsupportedError

    bt, rs, X, _ = model_pair(model, seed=260, num=4, transforms=True, warp=warp)
    Xt = torch.rand(2, 2, X.shape[-1], generator=_gen(261), dtype=DT)
    with pytest.raises(NotImplementedError if model == "linear" else RuntimeError):
        BtTS(bt)(Xt)
    with pytest.raises(UnsupportedError, match="Normalize input transform only"):
        RsTS(rs)(Xt)


def test_joint_entropy_search_raises_botorchs_error_on_fully_bayesian_gps() -> None:
    from botorch.acquisition.joint_entropy_search import qJointEntropySearch as BtJES

    from botorch_rs.acquisition.joint_entropy_search import qJointEntropySearch as RsJES

    bt, rs, X, _ = model_pair("saas", seed=270, num=4, transforms=False)
    opt_x = X[:3].clone()
    opt_y = torch.zeros(3, 1, dtype=DT)
    with pytest.raises(NotImplementedError) as e_bt:
        BtJES(bt, opt_x, opt_y)
    with pytest.raises(NotImplementedError) as e_rs:
        RsJES(rs, opt_x, opt_y)
    assert str(e_rs.value) == str(e_bt.value)


def test_pathwise_thompson_sampling_on_map_saas_ensembles_is_unsupported() -> None:
    """BoTorch's paths of an ``EnsembleMapSaasSingleTaskGP`` (whose kernel
    has no batch shape) fail for ``1 < q != num_taus`` (see the draft);
    botorch_rs raises ``UnsupportedError``."""
    from botorch.acquisition.thompson_sampling import PathwiseThompsonSampling as BtTS
    from botorch.models.map_saas import EnsembleMapSaasSingleTaskGP as BtEnsemble

    from botorch_rs.acquisition.thompson_sampling import PathwiseThompsonSampling as RsTS
    from botorch_rs.exceptions.errors import UnsupportedError
    from botorch_rs.models.map_saas import EnsembleMapSaasSingleTaskGP as RsEnsemble

    g = _gen(280)
    X = torch.rand(10, 4, generator=g, dtype=DT)
    Y = X.sum(-1, keepdim=True)
    torch.manual_seed(281)
    bt = BtEnsemble(X, Y, num_taus=3)
    bt.eval()
    with pytest.raises(RuntimeError):
        BtTS(bt)(torch.rand(3, 2, 4, generator=g, dtype=DT))
    with pytest.raises(UnsupportedError, match="ensembles of fully Bayesian GPs"):
        RsTS(RsEnsemble(X, Y, num_taus=3))


# ---------------------------------------------------------------------------
# Input constructors
# ---------------------------------------------------------------------------


def _fb_datasets(X: Any, Y: Any) -> tuple[Any, Any]:
    names = {"feature_names": [f"x{i}" for i in range(X.shape[-1])], "outcome_names": ["y"]}
    return BtDataset(X, Y, **names), RsDataset(X, Y, **names)


def test_bald_input_constructor_matches_botorch() -> None:
    """``construct_inputs_BALD``: the same keyword arguments (also from the
    common constructor arguments, which it ignores) and the same values of
    the acquisition function built from them."""
    from botorch.acquisition import input_constructors as bic
    from botorch.acquisition.bayesian_active_learning import (
        qBayesianActiveLearningByDisagreement as BtBALD,
    )

    from botorch_rs.acquisition import input_constructors as ric
    from botorch_rs.acquisition.bayesian_active_learning import (
        qBayesianActiveLearningByDisagreement as RsBALD,
    )

    bt, rs, X, _ = model_pair("saas", seed=290, num=4)
    _, Y, _ = raw_box_data(10, 3, 290, True)  # model_pair's training data
    lo, hi = X.min(0).values, X.max(0).values
    g = _gen(291)
    Xt = lo + (hi - lo) * torch.rand(3, 2, 3, generator=g, dtype=DT)
    Xp = lo + (hi - lo) * torch.rand(1, 3, generator=g, dtype=DT)
    s_bt, s_rs = _samplers(64, 3, 292, X=torch.cat([Xt, Xp.expand(3, 1, 3)], dim=-2), bt=bt)
    cb = bic.get_acqf_input_constructor(BtBALD)
    cr = ric.get_acqf_input_constructor(RsBALD)
    assert cr.__closure__ is not None and cb.__closure__ is not None
    assert (
        cr.__closure__[1].cell_contents.__name__
        == cb.__closure__[1].cell_contents.__name__
        == "construct_inputs_BALD"
    )
    d_bt, d_rs = _fb_datasets(X, Y)
    kb = cb(model=bt, training_data=d_bt, X_pending=Xp, sampler=s_bt)
    kr = cr(model=rs, training_data=d_rs, X_pending=Xp, sampler=s_rs)
    assert set(kr) == set(kb) == {"model", "X_pending", "sampler", "posterior_transform"}
    assert kr["model"] is rs and kr["sampler"] is s_rs
    assert kr["posterior_transform"] is None and kb["posterior_transform"] is None
    CHECK.check("construct_inputs_BALD: X_pending", kr["X_pending"], kb["X_pending"], 0.0)
    _compare_acqf(
        "qBayesianActiveLearningByDisagreement from its input constructor",
        RsBALD(**kr),
        BtBALD(**kb),
        Xt,
    )
    # Defaults: no pending points, the default sampler.
    kb0, kr0 = cb(model=bt, training_data=d_bt), cr(model=rs, training_data=d_rs)
    assert kb0 == {"model": bt, "X_pending": None, "sampler": None, "posterior_transform": None}
    assert kr0 == {"model": rs, "X_pending": None, "sampler": None, "posterior_transform": None}


@pytest.mark.parametrize("name", ["qLogExpectedImprovement", "qLogNoisyExpectedImprovement"])
def test_constructors_on_fully_bayesian_gps_match_botorch(name: str) -> None:
    """The input constructors of acquisition functions averaged over the
    samples of a fully Bayesian GP (``best_f`` from the training data,
    ``X_baseline``) and the values built from them, with identical base
    samples."""
    from botorch.acquisition import input_constructors as bic

    from botorch_rs.acquisition import input_constructors as ric

    bt, rs, X, _ = model_pair("matern", seed=300, num=4)
    _, Y, _ = raw_box_data(10, 3, 300, True)  # model_pair's training data
    noisy = "Noisy" in name
    q = 2
    lo, hi = X.min(0).values, X.max(0).values
    Xq = lo + (hi - lo) * torch.rand(1 if noisy else 3, q, 3, generator=_gen(301), dtype=DT)
    k = X.shape[0] + q if noisy else q
    s_bt, s_rs = _samplers(32, k, 302)
    bmod = bt_logei
    cls_bt, cls_rs = getattr(bmod, name), getattr(rs_logei, name)
    extra: dict[str, Any] = {"prune_baseline": False, "cache_root": False} if noisy else {}
    d_bt, d_rs = _fb_datasets(X, Y)
    kb = bic.get_acqf_input_constructor(cls_bt)(model=bt, training_data=d_bt, sampler=s_bt, **extra)
    kr = ric.get_acqf_input_constructor(cls_rs)(model=rs, training_data=d_rs, sampler=s_rs, **extra)
    assert set(kr) == set(kb)
    for key in kb:
        if key in ("model", "sampler", "objective"):
            continue
        if key in ("best_f", "X_baseline"):
            CHECK.check(
                f"construct_inputs of {name} (fully Bayesian): {key}", kr[key], kb[key], 0.0
            )
        else:
            assert kr[key] == kb[key], key
    _compare_acqf(
        f"{name} (fully Bayesian) from its input constructor",
        cls_rs(**kr),
        cls_bt(**kb),
        Xq,
        log=True,
    )


@pytest.mark.parametrize("name", ["qLogEI", "qLogNEI", "qUCB"])
def test_get_acquisition_function_on_fully_bayesian_gps_matches_botorch(name: str) -> None:
    """``get_acquisition_function`` of a fully Bayesian GP: the same class,
    ``best_f`` (one per sample: the best posterior mean of every sample at
    the observed inputs, broadcast against the samples) / ``X_baseline``,
    and values with identical base samples (Sobol samplers: the mixture
    posterior is a GPyTorch posterior)."""
    from botorch.acquisition import factory as bt_factory
    from botorch.acquisition.objective import IdentityMCObjective as BtIdentity

    from botorch_rs.acquisition import factory as rs_factory
    from botorch_rs.acquisition.objective import IdentityMCObjective as RsIdentity

    bt, rs, X, _ = model_pair("saas", seed=310, num=4)
    kw: dict[str, Any] = {"beta": 0.3} if name == "qUCB" else {}
    if name == "qLogNEI":
        kw.update(prune_baseline=False, cache_root=False)
    a_bt = bt_factory.get_acquisition_function(
        name, bt, BtIdentity(), X, mc_samples=16, seed=0, **kw
    )
    a_rs = rs_factory.get_acquisition_function(
        name, rs, RsIdentity(), X, mc_samples=16, seed=0, **kw
    )
    assert type(a_rs).__name__ == type(a_bt).__name__
    if name == "qLogEI":
        CHECK.check(
            "get_acquisition_function(qLogEI, fully Bayesian): best_f",
            torch.as_tensor(a_rs.best_f).reshape(-1),
            a_bt.best_f.reshape(-1),
            VALUE_TOL,
        )
    if name == "qLogNEI":
        CHECK.check(
            "get_acquisition_function(qLogNEI, fully Bayesian): X_baseline",
            a_rs.X_baseline,
            a_bt.X_baseline,
            0.0,
        )
    q = 2
    k = X.shape[0] + q if name == "qLogNEI" else q
    s_bt, s_rs = _samplers(16, k, 311)
    lo, hi = X.min(0).values, X.max(0).values
    Xq = lo + (hi - lo) * torch.rand(1, q, 3, generator=_gen(312), dtype=DT)
    module_bt = bt_logei if name != "qUCB" else bt_mc
    module_rs = rs_logei if name != "qUCB" else rs_mc
    cls = type(a_bt).__name__
    args: dict[str, Any] = {}
    if name == "qLogEI":
        args["best_f"] = a_bt.best_f
    elif name == "qLogNEI":
        args.update(X_baseline=X, prune_baseline=False, cache_root=False)
    else:
        args["beta"] = 0.3
    # The factory's acquisition functions rebuilt with identical samplers.
    _compare_acqf(
        f"get_acquisition_function({name}, fully Bayesian)",
        getattr(module_rs, cls)(rs, sampler=s_rs, **args),
        getattr(module_bt, cls)(bt, sampler=s_bt, **args),
        Xq,
        log=name != "qUCB",
    )

"""Randomized equivalence sweep of the fully Bayesian GPs of phase 10a
against BoTorch (``equivalence`` marker), and the statistical comparison of
botorch_rs's NUTS with BoTorch's default (single-precision) NUTS.

* random models (SAAS / Matérn / linear, 1 to 4 inputs, 4 to 14 points,
  inferred or fixed noise, input warping, ``Normalize`` / ``Standardize``,
  1 to 6 MCMC samples) built from the same random MCMC samples: posteriors,
  mixture moments, the MLL of every sample and its gradient, ``LogEI``,
  ``qLogNEI`` and ``qBALD`` values and gradients;
* random potentials against NumPyro's (double precision);
* short NUTS replays of random models against NumPyro's recorded draws
  (NumPyro's decisions, states within 10x BoTorch's own change under input
  perturbations at 1e-14 and 1e-13, six twins);
* BoTorch's default ``fit_fully_bayesian_model_nuts`` runs NumPyro in
  single precision (JAX's default), so its chains cannot be replayed in
  double precision: the posterior means of the unconstrained
  hyperparameters of two chains of each library are compared with
  standardized differences (Monte-Carlo standard errors from effective
  sample sizes), ``|z| <= 4.5`` per parameter (a false-positive rate of
  about 7e-6 per parameter for normal estimates, 5e-5 per test).
"""

from __future__ import annotations

import contextlib
import random
from collections.abc import Iterator
from typing import Any

import pytest

from helpers import importorskip_botorch

importorskip_botorch()
torch = pytest.importorskip("torch")

from fully_bayesian_helpers import require_jax  # noqa: E402

require_jax()

import gpytorch  # noqa: E402
import numpy as np  # noqa: E402
from botorch.acquisition import logei as bt_logei  # noqa: E402
from botorch.acquisition.analytic import LogExpectedImprovement as BtLogEI  # noqa: E402
from botorch.acquisition.bayesian_active_learning import (  # noqa: E402
    qBayesianActiveLearningByDisagreement as BtBALD,
)
from botorch.sampling.normal import SobolQMCNormalSampler as BtSampler  # noqa: E402
from fully_bayesian_helpers import (  # noqa: E402
    DT,
    TWIN_SCALES,
    bt_fit_recorded,
    bt_model,
    ess,
    first_difference,
    make_fb_data,
    model_pair,
    numpyro_potential,
    perturbed,
    rs_class,
)
from gpytorch.mlls import ExactMarginalLogLikelihood as BtMLL  # noqa: E402
from models7b_helpers import (  # noqa: E402
    GRAD_TOL,
    VALUE_TOL,
    check_grad,
    grad_err,
    values_and_grads,
)

from botorch_rs.acquisition import logei as rs_logei  # noqa: E402
from botorch_rs.acquisition.analytic import LogExpectedImprovement as RsLogEI  # noqa: E402
from botorch_rs.acquisition.bayesian_active_learning import (  # noqa: E402
    qBayesianActiveLearningByDisagreement as RsBALD,
)
from botorch_rs.fit import fit_fully_bayesian_model_nuts as rs_fit  # noqa: E402
from botorch_rs.mlls import ExactMarginalLogLikelihood as RsMLL  # noqa: E402
from botorch_rs.sampling import SobolQMCNormalSampler as RsSampler  # noqa: E402
from parity_helpers import Checker  # noqa: E402

pytestmark = [pytest.mark.equivalence, pytest.mark.filterwarnings("ignore")]

CHECK = Checker("fully Bayesian (10a sweep)")
Z_TOL = 4.5


@pytest.fixture(autouse=True, scope="module")
def _single_threaded_torch() -> Iterator[None]:
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(threads)


def _gen(seed: int) -> Any:
    return torch.Generator().manual_seed(seed)


def _sampler_pair(Z: Any) -> tuple[Any, Any]:
    num = Z.shape[0]
    s_bt = BtSampler(sample_shape=torch.Size([num]), seed=0)
    s_bt.base_samples = Z.clone()
    s_rs = RsSampler(sample_shape=torch.Size([num]), seed=0)
    s_rs.base_samples = Z.reshape(num, -1).clone()
    return s_bt, s_rs


def _config(seed: int) -> dict[str, Any]:
    r = random.Random(seed)
    return {
        "model": r.choice(["saas", "matern", "linear"]),
        "d": r.randint(1, 4),
        "n": r.randint(4, 14),
        "num": r.randint(1, 6),
        "fixed": r.random() < 0.35,
        "warp": r.random() < 0.35,
        "transforms": r.random() < 0.6,
    }


@pytest.mark.parametrize("seed", range(30))
def test_random_models_match_botorch(seed: int) -> None:
    c = _config(seed)
    # GPyTorch's lazily evaluated linear kernel gives low-rank posterior
    # covariances (RootLinearOperator) that BoTorch samples with rank-d base
    # samples; with dense kernel matrices (the same distribution) its samples
    # use one base sample per point, as botorch_rs's (see the draft).
    dense = (
        gpytorch.settings.lazily_evaluate_kernels(False)
        if c["model"] == "linear"
        else contextlib.nullcontext()
    )
    with dense:
        _random_model_case(seed, c)


def _random_model_case(seed: int, c: dict[str, Any]) -> None:
    bt, rs, X, _ = model_pair(
        c["model"],
        d=c["d"],
        n=c["n"],
        num=c["num"],
        seed=1000 + seed,
        fixed=c["fixed"],
        warp=c["warp"],
        transforms=c["transforms"],
    )
    label = f"{c['model']}"
    lo, hi = X.min(0).values, X.max(0).values
    g = _gen(2000 + seed)
    Xt = lo + (hi - lo) * torch.rand(3, 2, c["d"], generator=g, dtype=DT)
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
    # MLL of every sample and its gradient.
    bt.train(reset=False)
    bt.zero_grad()
    vb = BtMLL(bt.likelihood, bt)(bt(*bt.train_inputs), bt.train_targets)
    vb.sum().backward()
    gb = {
        n: (p.grad if p.grad is not None else torch.zeros_like(p)) for n, p in bt.named_parameters()
    }
    bt.eval()
    vr, gr = RsMLL(rs.likelihood, rs).value_and_grad()
    CHECK.check(f"{label}: MLL", vr, vb.detach(), VALUE_TOL)
    gr_t = torch.as_tensor(gr, dtype=DT)
    flat_r, flat_b, offset = [], [], 0
    for name, p in rs.named_parameters():
        size = torch.as_tensor(p).numel()
        flat_r.append(gr_t[offset : offset + size])
        flat_b.append(gb[name].reshape(-1))
        offset += size
    check_grad(CHECK, f"{label}: MLL gradient", torch.cat(flat_r), torch.cat(flat_b))
    # Acquisition functions.
    X1 = lo + (hi - lo) * torch.rand(4, 1, c["d"], generator=g, dtype=DT)
    best = torch.tensor(float(rs.posterior(X).mixture_mean.max()), dtype=DT)
    vb_, gb_ = values_and_grads(BtLogEI(bt, best_f=best), X1)
    vr_, gr_ = values_and_grads(RsLogEI(rs, best_f=best), X1)
    CHECK.check_log(f"{label}: LogEI values", vr_, vb_, VALUE_TOL)
    check_grad(CHECK, f"{label}: LogEI gradients", gr_, gb_, GRAD_TOL)
    q = 2
    n = X.shape[0]
    Xq = lo + (hi - lo) * torch.rand(1, q, c["d"], generator=g, dtype=DT)
    Z = torch.randn(16, 1, 1, q, generator=g, dtype=DT)
    s_bt, s_rs = _sampler_pair(Z)
    vb_, gb_ = values_and_grads(bt_logei.qLogExpectedImprovement(bt, best_f=best, sampler=s_bt), Xq)
    vr_, gr_ = values_and_grads(rs_logei.qLogExpectedImprovement(rs, best_f=best, sampler=s_rs), Xq)
    CHECK.check_log(f"{label}: qLogEI values", vr_, vb_, VALUE_TOL)
    check_grad(CHECK, f"{label}: qLogEI gradients", gr_, gb_, GRAD_TOL)
    # qLogNEI over the training baseline (nearly singular joint posteriors):
    # within 10x BoTorch's own change under perturbations of the baseline at
    # 1e-14 .. 1e-11 (at least the nominal tolerance).
    Z = torch.randn(16, 1, 1, n + q, generator=g, dtype=DT)
    kw = {"prune_baseline": False, "cache_root": False}

    def bt_nei(Xb: Any) -> tuple[Any, Any]:
        s_bt, _ = _sampler_pair(Z)
        a = bt_logei.qLogNoisyExpectedImprovement(bt, X_baseline=Xb, sampler=s_bt, **kw)
        return values_and_grads(a, Xq)

    vb_, gb_ = bt_nei(X)
    spread_v, spread_g = 0.0, 0.0
    for k, scale in enumerate((1e-14, 1e-13, 1e-12, 1e-11)):
        vt, gt = bt_nei(perturbed(X, scale, 3000 + k))
        spread_v = max(spread_v, (vt - vb_).abs().max().item() / max(vb_.abs().max().item(), 1.0))
        spread_g = max(
            spread_g, (gt - gb_).abs().max().item() / max(gb_.abs().max().item(), 1e-300)
        )
    _, s_rs = _sampler_pair(Z)
    a_rs = rs_logei.qLogNoisyExpectedImprovement(rs, X_baseline=X, sampler=s_rs, **kw)
    vr_, gr_ = values_and_grads(a_rs, Xq)
    tol_v, tol_g = max(VALUE_TOL, 10 * spread_v), max(GRAD_TOL, 10 * spread_g)
    err_v = CHECK.check_log(
        f"{label}: qLogNEI values (tolerance from BoTorch's twins)", vr_, vb_, tol_v
    )
    check_grad(
        CHECK, f"{label}: qLogNEI gradients (tolerance from BoTorch's twins)", gr_, gb_, tol_g
    )
    # The displayed tolerance is the last case's: the ratios to every case's.
    CHECK.check_bound(f"{label}: qLogNEI values / tolerance", err_v / tol_v, 1.0)
    CHECK.check_bound(f"{label}: qLogNEI gradients / tolerance", grad_err(gr_, gb_) / tol_g, 1.0)
    # BALD (BoTorch's base samples of the candidates' mixture posterior).
    Xb = lo + (hi - lo) * torch.rand(2, q, c["d"], generator=g, dtype=DT)
    s_bt = BtSampler(sample_shape=torch.Size([32]), seed=seed)
    with torch.no_grad():
        s_bt(bt.posterior(Xb))
    s_rs = RsSampler(sample_shape=torch.Size([32]), seed=0)
    s_rs.base_samples = s_bt.base_samples.reshape(32, -1).clone()
    vb_, gb_ = values_and_grads(BtBALD(bt, sampler=s_bt), Xb)
    vr_, gr_ = values_and_grads(RsBALD(rs, sampler=s_rs), Xb)
    # BALD is a difference of entropies of order one: values agree to an
    # absolute 1e-12 near zero (one sample: exactly zero up to rounding),
    # gradients relative to max(|gradient|, 1e-12) (zero when the mixture's
    # components are separated).
    CHECK.check(f"{label}: qBALD values", vr_, vb_, VALUE_TOL, atol=1e-12)
    gb_t = torch.as_tensor(gb_, dtype=DT)
    err = (torch.as_tensor(gr_, dtype=DT) - gb_t).abs().max().item() / max(
        gb_t.abs().max().item(), 1e-12
    )
    CHECK.check_bound(f"{label}: qBALD gradients", err, GRAD_TOL)


@pytest.mark.parametrize("seed", range(12))
def test_random_potentials_match_numpyro(seed: int) -> None:
    c = _config(500 + seed)
    case = {"model": c["model"], "d": c["d"], "n": c["n"], "fixed": c["fixed"], "warp": c["warp"]}
    X, Y, Yvar = make_fb_data(case, 600 + seed)
    bt = bt_model(case, X, Y, Yvar)
    rs = rs_class(c["model"])(X, Y, Yvar if c["fixed"] else None, use_input_warping=c["warp"])
    pot = rs.pyro_model._potential()
    g = _gen(700 + seed)
    for _ in range(2):
        z = (4 * torch.rand(pot.dim, generator=g, dtype=DT) - 2).tolist()
        vb, gb = numpyro_potential(bt, z)
        vr, gr = pot.potential(z)
        CHECK.check("random potentials", [vr], [vb], VALUE_TOL)
        check_grad(CHECK, "random potential gradients", gr, gb)


@pytest.mark.parametrize("seed", range(6))
def test_random_nuts_replays_match_numpyro(seed: int) -> None:
    c = _config(800 + seed)
    case = {
        "model": c["model"],
        "d": c["d"],
        "n": max(c["n"], 6),
        "fixed": c["fixed"],
        "warp": c["warp"],
    }
    X, Y, Yvar = make_fb_data(case, 900 + seed)
    kw = {"warmup_steps": 20, "num_samples": 4, "thinning": 1}
    rec = bt_fit_recorded(bt_model(case, X, Y, Yvar), fixed_warping=True, **kw)
    twins = [
        bt_fit_recorded(
            bt_model(case, perturbed(X, scale, 950 + 10 * seed + k), Y, Yvar),
            fixed_warping=True,
            **kw,
        )
        for k, scale in enumerate(TWIN_SCALES * 3)
    ]
    rs = rs_class(c["model"])(X, Y, Yvar if c["fixed"] else None, use_input_warping=c["warp"])
    out = rs.pyro_model._potential().sample_nuts(
        20, 4, z0=rec.z0, draws=rec.draws, stop_when_draws_fail=True
    )
    # The draws on which the twins agree with NumPyro's chain (a draw of the
    # wrong kind stops botorch_rs's replay after consuming it).
    agree_twin = min(first_difference(rec.draws, twin.draws) for twin in twins)
    consumed = out["consumed"] - (out["stopped_at"] is not None)
    agree_rs = consumed if out["first_flip"] is None else min(out["first_flip"], consumed)
    CHECK.check_bound(
        "random NUTS replays: draws the twin follows beyond botorch_rs", agree_twin - agree_rs, 0
    )
    if out["first_flip"] is None and out["stopped_at"] is None:
        worst = 0.0
        for t, ((_, zb, _), it) in enumerate(zip(rec.states, out["trace"], strict=False)):
            spread = max(float(np.max(np.abs(np.subtract(zb, tw.states[t][1])))) for tw in twins)
            err = float(np.max(np.abs(np.subtract(it[0], zb))))
            worst = max(worst, err / max(10 * spread, 1e-12))
        CHECK.check_bound("random NUTS replays: states / (10x twin spread)", worst, 1.0)


def _z_scores(chains_a: list[Any], chains_b: list[Any]) -> list[float]:
    """Per unconstrained parameter, the difference of the means of the pooled
    chains of two libraries divided by its Monte-Carlo standard error (the
    variance of a chain over its effective sample size, chains pooled with
    equal weights)."""
    out = []
    a = [np.asarray(c, dtype=np.float64) for c in chains_a]
    b = [np.asarray(c, dtype=np.float64) for c in chains_b]

    def stats(chains: list[Any], k: int) -> tuple[float, float]:
        means = [c[:, k].mean() for c in chains]
        se2 = sum(c[:, k].var(ddof=1) / ess(c[:, k]) for c in chains) / len(chains) ** 2
        return float(np.mean(means)), float(se2)

    for k in range(a[0].shape[1]):
        ma, sa = stats(a, k)
        mb, sb = stats(b, k)
        out.append((ma - mb) / max(np.sqrt(sa + sb), 1e-300))
    return out


def test_nuts_matches_botorch_default_single_precision_statistically() -> None:
    """Two chains of BoTorch's default NUTS (JAX in single precision) and two
    of botorch_rs's (seeds 0 and 1 each) on a well-conditioned SAAS model:
    the posterior means of the unconstrained hyperparameters agree within
    ``Z_TOL`` Monte-Carlo standard errors."""
    case = {"model": "saas", "d": 3, "n": 16}
    X, Y, _ = make_fb_data(case, 1100)
    warmup, samples = 150, 300
    bt_chains: list[list[list[float]]] = []
    for seed in (0, 1):
        bt = bt_model(case, X, Y, None, x64=False)
        rec = bt_fit_recorded(
            bt, x64=False, warmup_steps=warmup, num_samples=samples, thinning=1, seed=seed
        )
        bt_chains.append([s[1] for s in rec.states[warmup:]])
    rs_chains: list[list[list[float]]] = []
    for seed in (0, 1):
        rs = rs_class("saas")(X, Y)
        rs_fit(rs, warmup_steps=warmup, num_samples=samples, thinning=1, seed=seed)
        rs_chains.append([it[0] for it in rs._nuts_result["trace"][warmup:]])
    worst = max(abs(z) for z in _z_scores(rs_chains, bt_chains))
    CHECK.check_bound(
        "NUTS vs BoTorch's single-precision NUTS: max |z| / Z_TOL", worst / Z_TOL, 1.0
    )

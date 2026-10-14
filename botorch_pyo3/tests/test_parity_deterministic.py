"""Parity of deterministic models and composite acquisition functions with
BoTorch, through the public APIs of both libraries side by side:

* ``PosteriorMeanModel`` of ``ModelListGP``s and multi-output
  ``SingleTaskGP``s: posteriors, and as the model of qLogEHVI (with the
  settings of openbo's MO-TAF source terms: ``NondominatedPartitioning``,
  ``StochasticSampler([1])``, ``tau_max=1e-3``, ``fat=True``), qEHVI, the
  analytic EHVI, the single-objective analytic and Monte-Carlo acquisition
  functions and qLogNEI (values and gradients);
* ``GenericDeterministicModel`` of a torch callable in the same roles;
* BoTorch's ``StochasticSampler`` (accepted by botorch_rs acquisition
  functions on deterministic models) and the botorch_rs port;
* 0-dim hyperparameters set from size-1 tensors (``outputscale``, noise,
  constant mean), as GPyTorch's ``Module.initialize`` accepts them;
* the weighted log-mean-exp blend of a target qLogNEHVI and source qLogEHVI
  terms (openbo's ``_CompositeTAFEHVI``): BoTorch evaluates it as a
  ``torch.nn.Module``, botorch_rs natively (``WeightedLogMeanExp``) and as a
  callable acquisition function (the same module over botorch_rs terms).
  Values, gradients, and ``optimize_acqf`` from identical initial conditions.

Tolerances: values 1e-9 (log values: absolute below 1, relative above),
gradients 1e-8 (relative to the largest component). The target qLogNEHVI uses
identical base samples (``mo_acquisition_helpers.synced_draws``). Skipped
unless BoTorch is importable (required in CI's parity job).
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Iterator
from typing import Any

import pytest

from helpers import importorskip_botorch

importorskip_botorch()
torch = pytest.importorskip("torch")

from botorch.acquisition import analytic as bt_analytic  # noqa: E402
from botorch.acquisition import logei as bt_logei_so  # noqa: E402
from botorch.acquisition import monte_carlo as bt_mc_so  # noqa: E402
from botorch.acquisition.multi_objective import analytic as bt_mo_analytic  # noqa: E402
from botorch.acquisition.multi_objective import logei as bt_logei  # noqa: E402
from botorch.acquisition.multi_objective import monte_carlo as bt_mc  # noqa: E402
from botorch.models import ModelListGP as BtModelListGP  # noqa: E402
from botorch.models import SingleTaskGP as BtSingleTaskGP  # noqa: E402
from botorch.models import deterministic as bt_det  # noqa: E402
from botorch.models.transforms.outcome import Standardize as BtStandardize  # noqa: E402
from botorch.optim import optimize as bt_optimize  # noqa: E402
from botorch.sampling.stochastic_samplers import StochasticSampler as BtStochastic  # noqa: E402
from botorch.utils.multi_objective import box_decompositions as bt_bd  # noqa: E402
from botorch.utils.multi_objective.pareto import is_non_dominated  # noqa: E402
from gpytorch import kernels as bt_kernels  # noqa: E402
from mo_acquisition_helpers import (  # noqa: E402
    bt_sampler,
    rs_sampler,
    synced_draws,
    weak_front,
)

from botorch_rs import kernels as rs_kernels  # noqa: E402
from botorch_rs.acquisition import CallableAcquisitionFunction, WeightedLogMeanExp  # noqa: E402
from botorch_rs.acquisition import analytic as rs_analytic  # noqa: E402
from botorch_rs.acquisition import logei as rs_logei_so  # noqa: E402
from botorch_rs.acquisition import monte_carlo as rs_mc_so  # noqa: E402
from botorch_rs.acquisition.multi_objective import analytic as rs_mo_analytic  # noqa: E402
from botorch_rs.acquisition.multi_objective import logei as rs_logei  # noqa: E402
from botorch_rs.acquisition.multi_objective import monte_carlo as rs_mc  # noqa: E402
from botorch_rs.models import ModelListGP as RsModelListGP  # noqa: E402
from botorch_rs.models import SingleTaskGP as RsSingleTaskGP  # noqa: E402
from botorch_rs.models import deterministic as rs_det  # noqa: E402
from botorch_rs.models.transforms.outcome import Standardize as RsStandardize  # noqa: E402
from botorch_rs.optim import optimize as rs_optimize  # noqa: E402
from botorch_rs.sampling import StochasticSampler as RsStochastic  # noqa: E402
from botorch_rs.utils.multi_objective import box_decompositions as rs_bd  # noqa: E402
from parity_helpers import (  # noqa: E402
    DT,
    Checker,
    ModelSpec,
    dense_model_list,
    list_pair,
    log_err,
    model_pair,
    points_in,
    rel_err,
    values_and_grads,
)

CHECK = Checker("deterministic models and composites")
VALUE_TOL = 1e-9
GRAD_TOL = 1e-8
NUM_SAMPLES = 16
# openbo's qLogEHVI settings of the MO-TAF source terms.
TAU_MAX = 1e-3

pytestmark = pytest.mark.filterwarnings("ignore")


@pytest.fixture(autouse=True, scope="module")
def _single_threaded_torch() -> Iterator[None]:
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(threads)


def bt_eval(fn: Any, *args: Any, **kwargs: Any) -> Any:
    with dense_model_list(), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fn(*args, **kwargs)


def compare(label: str, v_rs: Any, v_bt: Any, g_rs: Any, g_bt: Any, log: bool) -> None:
    err_v = log_err(v_rs, v_bt) if log else rel_err(v_rs, v_bt, atol=1e-300)
    err_g = rel_err(g_rs, g_bt, atol=1e-14)
    CHECK.record(f"{label} value", err_v, VALUE_TOL)
    CHECK.record(f"{label} gradient", err_g, GRAD_TOL)
    assert err_v <= VALUE_TOL, (label, "value", err_v)
    assert err_g <= GRAD_TOL, (label, "gradient", err_g)


MODELS = {
    "list_1+1": lambda: list_pair(
        (ModelSpec(n=12, d=3), ModelSpec(n=10, d=3, kernel="matern52", normalize=True)), 31
    ),
    "m2": lambda: model_pair(ModelSpec(n=10, d=2, m=2), 32),
    "list_2+1": lambda: list_pair(
        (ModelSpec(n=9, d=2, m=2), ModelSpec(n=7, d=2, normalize=True)), 33
    ),
}


@pytest.fixture(scope="module", params=list(MODELS), ids=list(MODELS))
def pair(request: pytest.FixtureRequest) -> tuple[str, Any, Any, Any]:
    bt, rs, X = MODELS[request.param]()
    return request.param, bt, rs, X


def mean_front(bt: Any, X: Any) -> tuple[Any, Any]:
    """A reference point and a front of posterior means below which new
    points improve (the weakened front of ``weak_front``, non-dominated)."""
    ref, front = bt_eval(weak_front, bt, X)
    return ref, front[is_non_dominated(front)]


# ---------------------------------------------------------------------------
# PosteriorMeanModel
# ---------------------------------------------------------------------------


def test_posterior_mean_model(pair: tuple) -> None:
    label, bt, rs, X = pair
    pm_bt, pm_rs = bt_det.PosteriorMeanModel(bt), rs_det.PosteriorMeanModel(rs)
    Xt = points_in(X, (2, 3, X.shape[-1]), 1)
    p_bt = bt_eval(pm_bt.posterior, Xt)
    p_rs = pm_rs.posterior(Xt)
    CHECK.check(f"PosteriorMeanModel mean [{label}]", p_rs.mean, p_bt.mean, VALUE_TOL)
    assert torch.equal(torch.as_tensor(p_rs.variance), p_bt.variance)
    samples = torch.as_tensor(p_rs.rsample(torch.Size([4])))
    CHECK.check(
        f"PosteriorMeanModel samples [{label}]", samples, p_bt.rsample(torch.Size([4])), VALUE_TOL
    )
    assert pm_rs.num_outputs == pm_bt.num_outputs
    assert tuple(pm_rs.batch_shape) == tuple(pm_bt.batch_shape)
    CHECK.check(f"PosteriorMeanModel forward [{label}]", pm_rs(Xt), bt_eval(pm_bt, Xt), VALUE_TOL)


EHVI_CASES = [
    # (log, q, fat, tau_max, sampler, pending)
    (True, 1, True, TAU_MAX, "stochastic-botorch", False),
    (True, 1, True, TAU_MAX, "stochastic", False),
    (True, 2, False, 1e-2, "sobol", True),
    (False, 1, False, 1e-2, "stochastic-botorch", False),
    (False, 3, False, 1e-2, "stochastic", True),
]


def _rs_sampler(kind: str) -> Any:
    if kind == "stochastic-botorch":
        return BtStochastic(torch.Size([1]))
    if kind == "stochastic":
        return RsStochastic(torch.Size([1]))
    return rs_sampler(4, 3, 8)


@pytest.mark.parametrize(
    "case", EHVI_CASES, ids=lambda c: f"{'log' if c[0] else 'lin'}-q{c[1]}-fat{c[2]}-{c[4]}"
)
def test_ehvi_on_posterior_mean(pair: tuple, case: tuple) -> None:
    label, bt, rs, X = pair
    log, q, fat, tau_max, sampler, pending = case
    m = rs.num_outputs
    if m > 2:
        pytest.skip("the hypervolume terms are compared on two outcomes")
    ref, front = mean_front(bt, X)
    kw: dict[str, Any] = {"fat": fat}
    if log:
        kw["tau_max"] = tau_max
    if pending:
        kw["X_pending"] = points_in(X, (1, X.shape[-1]), 3)
    name = "qLogExpectedHypervolumeImprovement" if log else "qExpectedHypervolumeImprovement"
    p_bt = bt_bd.NondominatedPartitioning(ref_point=ref, Y=front)
    p_rs = rs_bd.NondominatedPartitioning(ref_point=ref, Y=front)
    a_bt = bt_eval(
        getattr(bt_logei if log else bt_mc, name),
        bt_det.PosteriorMeanModel(bt),
        ref.tolist(),
        p_bt,
        sampler=BtStochastic(torch.Size([1])),
        **kw,
    )
    a_rs = getattr(rs_logei if log else rs_mc, name)(
        rs_det.PosteriorMeanModel(rs), ref.tolist(), p_rs, sampler=_rs_sampler(sampler), **kw
    )
    Xt = points_in(X, (5, q, X.shape[-1]), 4)
    v_bt, g_bt = bt_eval(values_and_grads, a_bt, Xt)
    v_rs, g_rs = values_and_grads(a_rs, Xt)
    compare(f"{name}(PosteriorMeanModel) [{label}]", v_rs, v_bt, g_rs, g_bt, log)


def test_analytic_ehvi_on_posterior_mean(pair: tuple) -> None:
    label, bt, rs, X = pair
    if rs.num_outputs > 2:
        pytest.skip("compared on two outcomes")
    ref, front = mean_front(bt, X)
    a_bt = bt_mo_analytic.ExpectedHypervolumeImprovement(
        bt_det.PosteriorMeanModel(bt), ref.tolist(), bt_bd.FastNondominatedPartitioning(ref, front)
    )
    a_rs = rs_mo_analytic.ExpectedHypervolumeImprovement(
        rs_det.PosteriorMeanModel(rs), ref.tolist(), rs_bd.FastNondominatedPartitioning(ref, front)
    )
    Xt = points_in(X, (5, 1, X.shape[-1]), 5)
    v_bt, g_bt = bt_eval(values_and_grads, a_bt, Xt)
    v_rs, g_rs = values_and_grads(a_rs, Xt)
    compare(
        f"ExpectedHypervolumeImprovement(PosteriorMeanModel) [{label}]",
        v_rs,
        v_bt,
        g_rs,
        g_bt,
        False,
    )


SO_SPECS = {
    "rbf": ModelSpec(n=10, d=2),
    "matern_norm": ModelSpec(n=12, d=3, kernel="matern52", normalize=True, lo=-1.0, hi=2.0),
}


@pytest.mark.parametrize("spec", list(SO_SPECS))
def test_single_objective_on_posterior_mean(spec: str) -> None:
    bt, rs, X = model_pair(SO_SPECS[spec], 41)
    pm_bt, pm_rs = bt_det.PosteriorMeanModel(bt), rs_det.PosteriorMeanModel(rs)
    with torch.no_grad():
        best_f = bt.posterior(X).mean.max().reshape(()).clone()
    d = X.shape[-1]
    X1 = points_in(X, (6, 1, d), 6)
    Xq = points_in(X, (4, 2, d), 7)
    with torch.no_grad():
        mu = bt.posterior(X1).mean.reshape(-1)
    # The standard deviation of a deterministic model is sqrt(1e-12) in
    # BoTorch's analytic acquisition functions: improvements are compared
    # within a few 1e-6 of the mean (deep in the tail, see
    # test_log_ei_tail_on_posterior_mean).
    for name in ("LogExpectedImprovement", "ExpectedImprovement", "LogProbabilityOfImprovement"):
        for i in range(X1.shape[0]):
            best_f = (mu[i] + (i - 2.5) * 1e-6).clone()
            a_bt = getattr(bt_analytic, name)(pm_bt, best_f=best_f)
            a_rs = getattr(rs_analytic, name)(pm_rs, best_f=best_f)
            v_bt, g_bt = bt_eval(values_and_grads, a_bt, X1[i : i + 1])
            v_rs, g_rs = values_and_grads(a_rs, X1[i : i + 1])
            compare(
                f"{name}(PosteriorMeanModel) [{spec}]",
                v_rs,
                v_bt,
                g_rs,
                g_bt,
                name.startswith("Log"),
            )
    # The standard deviation is the constant sqrt(1e-12): BoTorch's value
    # does not require gradients, botorch_rs's gradient is zero.
    a_bt = bt_analytic.PosteriorStandardDeviation(pm_bt)
    a_rs = rs_analytic.PosteriorStandardDeviation(pm_rs)
    v_rs, g_rs = values_and_grads(a_rs, X1)
    CHECK.check(
        f"PosteriorStandardDeviation(PosteriorMeanModel) [{spec}] value", v_rs, a_bt(X1), 0.0
    )
    assert not bool(torch.as_tensor(g_rs).any())
    analytic = [
        ("UpperConfidenceBound", {"beta": torch.tensor(0.7, dtype=DT)}),
        ("PosteriorMean", {}),
    ]
    for name, kw in analytic:
        a_bt = getattr(bt_analytic, name)(pm_bt, **kw)
        a_rs = getattr(rs_analytic, name)(pm_rs, **kw)
        v_bt, g_bt = bt_eval(values_and_grads, a_bt, X1)
        v_rs, g_rs = values_and_grads(a_rs, X1)
        compare(f"{name}(PosteriorMeanModel) [{spec}]", v_rs, v_bt, g_rs, g_bt, False)
    mc = [
        (bt_logei_so, rs_logei_so, "qLogExpectedImprovement", {"best_f": best_f}, True),
        (bt_mc_so, rs_mc_so, "qExpectedImprovement", {"best_f": best_f - 0.2}, False),
        (bt_mc_so, rs_mc_so, "qUpperConfidenceBound", {"beta": 0.7}, False),
        (bt_mc_so, rs_mc_so, "qSimpleRegret", {}, False),
        (bt_mc_so, rs_mc_so, "qProbabilityOfImprovement", {"best_f": best_f - 0.2}, False),
    ]
    for bt_mod, rs_mod, name, kw, log in mc:
        a_bt = getattr(bt_mod, name)(pm_bt, sampler=BtStochastic(torch.Size([3])), **kw)
        a_rs = getattr(rs_mod, name)(pm_rs, sampler=RsStochastic(torch.Size([3])), **kw)
        v_bt, g_bt = bt_eval(values_and_grads, a_bt, Xq)
        v_rs, g_rs = values_and_grads(a_rs, Xq)
        compare(f"{name}(PosteriorMeanModel) [{spec}]", v_rs, v_bt, g_rs, g_bt, log)
    for prune in (False, True):
        a_bt = bt_logei_so.qLogNoisyExpectedImprovement(
            pm_bt, X, sampler=BtStochastic(torch.Size([1])), prune_baseline=prune
        )
        a_rs = rs_logei_so.qLogNoisyExpectedImprovement(
            pm_rs, X, sampler=BtStochastic(torch.Size([1])), prune_baseline=prune
        )
        assert torch.equal(torch.as_tensor(a_rs.X_baseline), a_bt.X_baseline)
        v_bt, g_bt = bt_eval(values_and_grads, a_bt, Xq)
        v_rs, g_rs = values_and_grads(a_rs, Xq)
        compare(
            f"qLogNoisyExpectedImprovement(PosteriorMeanModel, prune={prune}) [{spec}]",
            v_rs,
            v_bt,
            g_rs,
            g_bt,
            True,
        )


# ---------------------------------------------------------------------------
# GenericDeterministicModel
# ---------------------------------------------------------------------------


def _quadratic(X: Any) -> Any:
    """Two smooth, conflicting outcomes of ``batch x n x 2`` inputs."""
    c0 = torch.tensor([0.3, 0.6], dtype=X.dtype)
    c1 = torch.tensor([0.8, 0.2], dtype=X.dtype)
    return torch.stack(
        [1.0 - ((X - c0) ** 2).sum(-1), torch.cos(2.0 * X[..., 0]) - ((X - c1) ** 2).sum(-1)], -1
    )


def test_generic_deterministic_model() -> None:
    g_bt = bt_det.GenericDeterministicModel(_quadratic, num_outputs=2)
    g_rs = rs_det.GenericDeterministicModel(_quadratic, num_outputs=2, dim=2)
    Xb = torch.rand(9, 2, generator=torch.Generator().manual_seed(3), dtype=DT)
    Y = _quadratic(Xb)
    ref = Y.min(0).values - 0.1
    front = ref + 0.6 * (Y - ref)
    front = front[is_non_dominated(front)]
    Xt = torch.rand(6, 2, 2, generator=torch.Generator().manual_seed(4), dtype=DT)
    for log in (True, False):
        name = "qLogExpectedHypervolumeImprovement" if log else "qExpectedHypervolumeImprovement"
        a_bt = getattr(bt_logei if log else bt_mc, name)(
            g_bt,
            ref.tolist(),
            bt_bd.NondominatedPartitioning(ref, front),
            sampler=BtStochastic(torch.Size([1])),
        )
        a_rs = getattr(rs_logei if log else rs_mc, name)(
            g_rs,
            ref.tolist(),
            rs_bd.NondominatedPartitioning(ref, front),
            sampler=RsStochastic(torch.Size([1])),
        )
        v_bt, g_b = values_and_grads(a_bt, Xt)
        v_rs, g_r = values_and_grads(a_rs, Xt)
        compare(f"{name}(GenericDeterministicModel)", v_rs, v_bt, g_r, g_b, log)
    # A single-output model in an analytic acquisition function (improvements
    # of a few standard deviations sqrt(1e-12), see the PosteriorMeanModel
    # tests).
    f1 = lambda X: _quadratic(X)[..., :1]  # noqa: E731
    X1 = Xt[:, :1]
    mu = f1(X1).reshape(-1)
    for i in range(X1.shape[0]):
        best_f = (mu[i] + (i - 2.5) * 1e-6).clone()
        a_bt = bt_analytic.LogExpectedImprovement(
            bt_det.GenericDeterministicModel(f1), best_f=best_f
        )
        a_rs = rs_analytic.LogExpectedImprovement(
            rs_det.GenericDeterministicModel(f1, dim=2), best_f=best_f
        )
        v_bt, g_b = values_and_grads(a_bt, X1[i : i + 1])
        v_rs, g_r = values_and_grads(a_rs, X1[i : i + 1])
        compare("LogExpectedImprovement(GenericDeterministicModel)", v_rs, v_bt, g_r, g_b, True)


def test_log_ei_tail_on_posterior_mean() -> None:
    """Far from the mean (``|u|`` up to about 1e6 with the standard deviation
    sqrt(1e-12) of a deterministic model), LogEI is compared with an exact
    reference (mpmath, 60 digits; the gradient is the exact derivative of
    log EI times BoTorch's gradient of the mean): BoTorch's autograd loses
    accuracy there (see the README's deliberate deviations), botorch_rs
    does not."""
    mpmath = pytest.importorskip("mpmath")
    mpmath.mp.dps = 60
    bt, rs, X = model_pair(SO_SPECS["rbf"], 41)
    pm_bt, pm_rs = bt_det.PosteriorMeanModel(bt), rs_det.PosteriorMeanModel(rs)
    X1 = points_in(X, (8, 1, X.shape[-1]), 8)
    Xg = X1.clone().requires_grad_(True)
    mu = bt.posterior(Xg).mean.reshape(-1)
    (dmu,) = torch.autograd.grad(mu.sum(), Xg)
    mu = mu.detach()
    best_f = (mu.mean() + 0.05).clone()
    sigma = mpmath.sqrt(mpmath.mpf(1e-12))
    exact_v, exact_g = [], []
    for i in range(X1.shape[0]):
        u = (mpmath.mpf(float(mu[i])) - mpmath.mpf(float(best_f))) / sigma
        ei = mpmath.npdf(u) + u * mpmath.ncdf(u)
        exact_v.append(float(mpmath.log(ei) + mpmath.log(sigma)))
        exact_g.append(float(mpmath.ncdf(u) / ei / sigma) * dmu[i])
    exact_v_t = torch.tensor(exact_v, dtype=DT)
    exact_g_t = torch.stack(exact_g)
    a_bt = bt_analytic.LogExpectedImprovement(pm_bt, best_f=best_f)
    a_rs = rs_analytic.LogExpectedImprovement(pm_rs, best_f=best_f)
    v_bt, g_bt = bt_eval(values_and_grads, a_bt, X1)
    v_rs, g_rs = values_and_grads(a_rs, X1)
    CHECK.record(
        "LogEI tail (|u| ~ 1e5): BoTorch's value error vs exact", log_err(v_bt, exact_v_t), None
    )
    CHECK.record(
        "LogEI tail (|u| ~ 1e5): BoTorch's gradient error vs exact",
        rel_err(g_bt, exact_g_t),
        None,
    )
    CHECK.check_log("LogEI tail (|u| ~ 1e5): value vs exact", v_rs, exact_v_t, 1e-13)
    CHECK.check("LogEI tail (|u| ~ 1e5): gradient vs exact", g_rs, exact_g_t, 1e-10)


# ---------------------------------------------------------------------------
# Size-1 tensors for 0-dim hyperparameters
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("shape", [(), (1,), (1, 1)])
def test_size_one_hyperparameters(shape: tuple[int, ...]) -> None:
    g = torch.Generator().manual_seed(5)
    X = torch.rand(10, 3, generator=g, dtype=DT)
    Y = torch.sin(4 * X).sum(-1, keepdim=True)
    bt = BtSingleTaskGP(
        X,
        Y,
        covar_module=bt_kernels.ScaleKernel(bt_kernels.MaternKernel(nu=2.5, ard_num_dims=3)),
        outcome_transform=BtStandardize(m=1),
    )
    rs = RsSingleTaskGP(
        X,
        Y,
        covar_module=rs_kernels.ScaleKernel(rs_kernels.MaternKernel(nu=2.5, ard_num_dims=3)),
        outcome_transform=RsStandardize(m=1),
    )
    values = {"outputscale": 1.7, "noise": 0.03, "constant": -0.2}
    for model in (bt, rs):
        with torch.no_grad():
            model.covar_module.base_kernel.lengthscale = torch.tensor([0.3, 0.5, 0.9], dtype=DT)
            model.covar_module.outputscale = torch.full(shape, values["outputscale"], dtype=DT)
            model.likelihood.noise = torch.full(shape, values["noise"], dtype=DT)
            model.mean_module.constant = torch.full(shape, values["constant"], dtype=DT)
        model.eval()
    for name, a, b in (
        ("outputscale", rs.covar_module.outputscale, bt.covar_module.outputscale),
        ("noise", rs.likelihood.noise, bt.likelihood.noise),
        ("constant", rs.mean_module.constant, bt.mean_module.constant),
    ):
        a, b = torch.as_tensor(a), b.detach()
        assert a.shape == b.shape, (name, a.shape, b.shape)
        CHECK.check(f"0-dim hyperparameter set from shape {shape}: {name}", a, b, 1e-15)
    Xt = torch.rand(5, 3, generator=g, dtype=DT)
    p_bt, p_rs = bt.posterior(Xt), rs.posterior(Xt)
    CHECK.check("posterior mean after size-1 hyperparameters", p_rs.mean, p_bt.mean, VALUE_TOL)
    CHECK.check(
        "posterior variance after size-1 hyperparameters", p_rs.variance, p_bt.variance, VALUE_TOL
    )


# ---------------------------------------------------------------------------
# Composite acquisition functions (openbo's MO-TAF)
# ---------------------------------------------------------------------------


class Blend(torch.nn.Module):
    """openbo's ``_CompositeTAFEHVI`` (``taf_mo_ehvi_acquisition``):
    ``logsumexp(log w_i + term_i(X)) - log(sum w_i)`` over the terms with a
    positive weight."""

    def __init__(self, terms: list[Any], weights: list[float]) -> None:
        super().__init__()
        self.terms = terms
        self.weights = weights
        self.X_pending = None

    def forward(self, X: Any) -> Any:
        logs, total = [], 0.0
        for term, w in zip(self.terms, self.weights, strict=True):
            if w <= 0.0:
                continue
            logs.append(math.log(w) + term(X))
            total += w
        return torch.logsumexp(torch.stack(logs, dim=-1), dim=-1) - math.log(total)


def openbo_sources(d: int, m: int, seed: int, num: int) -> list[tuple[Any, Any, Any]]:
    """``num`` source surrogates built as openbo's ``_materialize_source_mean``
    does (``ScaleKernel`` of Matern-5/2 or RBF, ``Standardize``, fixed
    hyperparameters set from 0-dim and size-1 tensors) in both libraries, with
    their (non-dominated) fronts."""
    g = torch.Generator().manual_seed(seed)
    out = []
    for s in range(num):
        n = 12 + 3 * s
        x = torch.rand(n, d, generator=g, dtype=DT)
        centers = torch.rand(m, d, generator=g, dtype=DT)
        y = torch.stack([1 - 2 * ((x - c) ** 2).sum(-1) for c in centers], -1)
        bts, rss = [], []
        for j in range(m):
            matern = (s + j) % 2 == 0
            ls = 0.2 + torch.rand(d, generator=g, dtype=DT)
            var = 0.5 + torch.rand(1, generator=g, dtype=DT)
            noise = 0.01 + 0.05 * torch.rand(1, generator=g, dtype=DT)
            const = 0.4 * torch.rand(1, generator=g, dtype=DT) - 0.2
            models = []
            for lib in ("bt", "rs"):
                K = bt_kernels if lib == "bt" else rs_kernels
                base = (
                    K.MaternKernel(nu=2.5, ard_num_dims=d)
                    if matern
                    else K.RBFKernel(ard_num_dims=d)
                )
                GP = BtSingleTaskGP if lib == "bt" else RsSingleTaskGP
                Std = BtStandardize if lib == "bt" else RsStandardize
                gp = GP(
                    x, y[:, j : j + 1], covar_module=K.ScaleKernel(base), outcome_transform=Std(m=1)
                )
                with torch.no_grad():
                    gp.covar_module.base_kernel.lengthscale = ls
                    gp.covar_module.outputscale = var.reshape(())  # 0-dim, as openbo
                    gp.likelihood.noise = noise  # size 1
                    gp.mean_module.constant = const  # size 1
                gp.eval()
                models.append(gp)
            bts.append(models[0])
            rss.append(models[1])
        front = y[is_non_dominated(y)]
        out.append((BtModelListGP(*bts), RsModelListGP(*rss), front))
    return out


def composite_pair(seed: int, num_sources: int, weights: list[float], q: int = 1) -> dict[str, Any]:
    """A target qLogNEHVI (identical base samples) and qLogEHVI source terms on
    ``PosteriorMeanModel``s, blended in both libraries."""
    bt, rs, X = model_pair(ModelSpec(n=10, d=3, m=2, kernel="matern52"), seed)
    n, m, d = X.shape[0], 2, X.shape[-1]
    ref, _ = bt_eval(weak_front, bt, X)
    ref = ref - 0.5
    with synced_draws(seed):
        t_bt = bt_eval(
            bt_logei.qLogNoisyExpectedHypervolumeImprovement,
            bt,
            ref.tolist(),
            X_baseline=X,
            sampler=bt_sampler(NUM_SAMPLES),
            prune_baseline=False,
        )
    t_rs = rs_logei.qLogNoisyExpectedHypervolumeImprovement(
        rs,
        ref.tolist(),
        X_baseline=X,
        sampler=rs_sampler(NUM_SAMPLES, seed, (n + q + 1) * m),
        prune_baseline=False,
    )
    s_bt, s_rs = [], []
    for src_bt, src_rs, front in openbo_sources(d, m, seed + 100, num_sources):
        front = front[(front > ref).all(-1)]
        s_bt.append(
            bt_logei.qLogExpectedHypervolumeImprovement(
                bt_det.PosteriorMeanModel(src_bt),
                ref.tolist(),
                bt_bd.NondominatedPartitioning(ref_point=ref, Y=front),
                sampler=BtStochastic(torch.Size([1])),
                tau_max=TAU_MAX,
                fat=True,
            )
        )
        s_rs.append(
            rs_logei.qLogExpectedHypervolumeImprovement(
                rs_det.PosteriorMeanModel(src_rs),
                ref.tolist(),
                rs_bd.NondominatedPartitioning(ref_point=ref, Y=front),
                sampler=BtStochastic(torch.Size([1])),
                tau_max=TAU_MAX,
                fat=True,
            )
        )
    terms_bt, terms_rs = [t_bt, *s_bt], [t_rs, *s_rs]
    lo, hi = X.min(0).values, X.max(0).values
    return {
        "seed": seed,
        "bt": Blend(terms_bt, weights),
        "native": WeightedLogMeanExp(terms_rs, weights),
        "callable": Blend(terms_rs, weights),
        "bounds": torch.stack([lo, hi]),
        "X": X,
    }


COMPOSITES = {
    "target+2 sources": (51, 2, [1.0, 0.45, 0.2]),
    "warmup (target weight 0)": (52, 2, [0.0, 0.7, 0.3]),
    "target+1 source, zero weight": (53, 2, [0.8, 0.0, 0.6]),
}


@pytest.mark.parametrize("case", list(COMPOSITES))
def test_composite_values_and_gradients(case: str) -> None:
    seed, num, weights = COMPOSITES[case]
    c = composite_pair(seed, num, weights)
    X = c["X"]
    Xt = points_in(X, (6, 1, X.shape[-1]), seed + 1)
    with synced_draws(seed):
        v_bt, g_bt = bt_eval(values_and_grads, c["bt"], Xt)
    v_n, g_n = values_and_grads(c["native"], Xt)
    compare(f"WeightedLogMeanExp (native) [{case}]", v_n, v_bt, g_n, g_bt, True)
    v_c, g_c = values_and_grads(CallableAcquisitionFunction(c["callable"], X.shape[-1]), Xt)
    compare(f"callable composite [{case}]", v_c, v_bt, g_c, g_bt, True)
    # The native batch evaluation (forward_many of every term) agrees with
    # the single evaluations.
    single = torch.stack([c["native"](Xt[i : i + 1]).reshape(()) for i in range(Xt.shape[0])])
    assert torch.equal(torch.as_tensor(c["native"](Xt)), single)


def _compare_restarts(label: str, x_rs: Any, v_rs: Any, x_bt: Any, v_bt: Any) -> None:
    k = v_bt.shape[0]
    agree = [
        abs(float(v_rs[i]) - float(v_bt[i])) <= 1e-6 * max(1.0, abs(float(v_bt[i])))
        for i in range(k)
    ]
    CHECK.record(
        f"optimize_acqf ({label}): restarts not reaching BoTorch's value", k - sum(agree), None
    )
    errs = [(torch.as_tensor(x_rs[i]) - x_bt[i]).abs().max().item() for i in range(k) if agree[i]]
    if errs:
        CHECK.record(f"optimize_acqf ({label}): candidates of agreeing restarts", max(errs), None)
    assert sum(agree) >= k - 1, (label, v_rs, v_bt)


# BoTorch 0.18.1 (with SciPy >= 1.16) optimizes all L-BFGS-B restarts as one joint
# problem by default, which stops them at other points than separate problems; main and
# botorch_rs optimize every restart on its own.
ONE_RESTART_PER_PROBLEM = {"batch_limit": 1}


@pytest.mark.parametrize("case", list(COMPOSITES)[:2])
def test_composite_optimize_acqf(case: str) -> None:
    seed, num, weights = COMPOSITES[case]
    c = composite_pair(seed, num, weights)
    d = c["X"].shape[-1]
    ics = points_in(c["X"], (4, 1, d), seed + 2, pad=0.0).clamp(c["bounds"][0], c["bounds"][1])
    with synced_draws(seed):
        x_bt, v_bt = bt_eval(
            bt_optimize.optimize_acqf,
            c["bt"],
            c["bounds"],
            q=1,
            num_restarts=4,
            batch_initial_conditions=ics,
            return_best_only=False,
            options=ONE_RESTART_PER_PROBLEM,
        )
    for kind in ("native", "callable"):
        x_rs, v_rs = rs_optimize.optimize_acqf(
            c[kind],
            c["bounds"],
            q=1,
            num_restarts=4,
            batch_initial_conditions=ics,
            return_best_only=False,
        )
        _compare_restarts(f"{kind} composite [{case}]", x_rs, v_rs, x_bt, v_bt)
        # Returned values are the acquisition values at the candidates.
        with synced_draws(seed):
            v_at = bt_eval(c["bt"], torch.as_tensor(x_rs).detach())
        CHECK.check_log(f"optimize_acqf ({kind} composite): returned values", v_rs, v_at, VALUE_TOL)


@pytest.mark.equivalence
@pytest.mark.parametrize("seed", range(6))
def test_composite_sweep(seed: int) -> None:
    """Randomized composites: number of sources, weights (some zero) and
    candidates; values, gradients and optimization from identical initial
    conditions."""
    g = torch.Generator().manual_seed(1000 + seed)
    num = 1 + seed % 3
    weights = [float(w) for w in torch.rand(num + 1, generator=g, dtype=DT)]
    if seed % 2:
        weights[int(torch.randint(num + 1, (1,), generator=g))] = 0.0
    if not any(w > 0 for w in weights):
        weights[0] = 1.0
    c = composite_pair(200 + seed, num, weights)
    X = c["X"]
    d = X.shape[-1]
    Xt = points_in(X, (8, 1, d), seed + 3)
    with synced_draws(200 + seed):
        v_bt, g_bt = bt_eval(values_and_grads, c["bt"], Xt)
    v_n, g_n = values_and_grads(c["native"], Xt)
    compare("WeightedLogMeanExp (native), randomized", v_n, v_bt, g_n, g_bt, True)
    ics = points_in(X, (3, 1, d), seed + 4, pad=0.0).clamp(c["bounds"][0], c["bounds"][1])
    with synced_draws(200 + seed):
        x_bt, v_bt = bt_eval(
            bt_optimize.optimize_acqf,
            c["bt"],
            c["bounds"],
            q=1,
            num_restarts=3,
            batch_initial_conditions=ics,
            return_best_only=False,
            options=ONE_RESTART_PER_PROBLEM,
        )
    x_rs, v_rs = rs_optimize.optimize_acqf(
        c["native"],
        c["bounds"],
        q=1,
        num_restarts=3,
        batch_initial_conditions=ics,
        return_best_only=False,
    )
    _compare_restarts("native composite, randomized", x_rs, v_rs, x_bt, v_bt)

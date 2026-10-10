"""The unified equivalence suite: botorch_rs against live BoTorch over a seeded,
randomized grid of every supported feature.

Scenarios (``SCENARIOS``) combine numbers of training points ``n`` (6, 10,
16), input dimensions ``d`` (1-4), outputs ``m`` (1-3), inferred or fixed
observation noise, with or without ``Normalize`` / ``Standardize``, RBF or
Matern-5/2 kernels, input ranges, batched models and ``ModelListGP`` models;
data, hyperparameters, test points, base samples and the remaining choices
(``q``, pending points, objectives, constraints, ``fat``, ...) are drawn
from seeded generators. For every scenario both libraries are built on
identical data and hyperparameters and compared on

* default initialization, MLL values and gradients, posteriors (joint and
  batched, with and without observation noise, scalarized) and samples from
  explicit base samples;
* fitting (``fit_gpytorch_mll``);
* analytic acquisition functions (EI, LogEI, PI, LogPI, UCB, posterior mean
  and standard deviation, constrained EI / LogEI) with and without posterior
  transforms;
* Monte-Carlo acquisition functions (qEI, qPI, qLogEI, qSR, qUCB, qLogNEI,
  qLogPF) with identity, posterior-transform, linear, generic (X-dependent)
  and constrained objectives, outcome constraints and pending points, with
  identical base samples; ``prune_inferior_points``;
* ``optimize_acqf`` without and with parameter constraints (linear, inter-
  point, equality, nonlinear);

plus the synthetic test functions and random polytopes (polytope sampling,
interior points, feasibility and constraint transformations).

Tolerances (the worst observed error of every quantity is printed in the
terminal summary, section "equivalence suite"):

=======================================  =====================================
deterministic values                     relative error <= 1e-9 (log-space
                                         values: absolute below 1)
gradients                                relative error <= 1e-8
Monte-Carlo values / gradients           1e-9 / 1e-8, or 10x BoTorch's own
                                         change under a 1e-14 relative
                                         perturbation of the training inputs
                                         if larger (nearly singular joint
                                         posteriors amplify rounding); the
                                         recorded error is scaled to 1e-9 /
                                         1e-8, the raw error is reported too
numerically singular joint posteriors    only finiteness and agreement to 0.1
(an output block with smallest           (whether Cholesky needs jitter then
eigenvalue < 1e-12 x largest)            depends on rounding; counted)
fits                                     final loss <= BoTorch's + 1e-6
optimizers, identical initial            >= 80% of the restarts reach
conditions                               BoTorch's value (relative 1e-6)
optimizers, end to end (statistical)     best value over 3 seeds >= the median
                                         of BoTorch's per-seed values - 1e-3
                                         max(1, |value|); results feasible;
                                         returned values BoTorch's (1e-9)
polytope samples (statistical)           moments within 0.05 / 0.02 of the
                                         range (squared)
=======================================  =====================================

BoTorch stores Python floats for ``best_f``, ``tau``, ``beta``, ``eta`` and
``infeasible_cost`` in float32; the suite passes float64 tensors (botorch_rs
uses exact float64 values). BoTorch's qLogNEI is used with ``cache_root=False``
and explicit lower-bound points, and its ModelListGP code paths with dense
covariances (see ``parity_helpers.dense_model_list``).

Run with ``pytest -m equivalence`` (it is part of the normal run as well);
skipped unless BoTorch is importable, which ``BOTORCH_RS_REQUIRE_BOTORCH=1``
turns into an error.
"""

from __future__ import annotations

import random
import zlib
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import pytest

from helpers import importorskip_botorch

importorskip_botorch()
torch = pytest.importorskip("torch")

from botorch.acquisition import analytic as bt_analytic  # noqa: E402
from botorch.acquisition import logei as bt_logei  # noqa: E402
from botorch.acquisition import monte_carlo as bt_mc  # noqa: E402
from botorch.acquisition import objective as bt_objective  # noqa: E402
from botorch.acquisition.utils import prune_inferior_points as bt_prune  # noqa: E402
from botorch.exceptions.errors import (  # noqa: E402
    CandidateGenerationError as BtCandidateGenerationError,
)
from botorch.fit import fit_gpytorch_mll as bt_fit  # noqa: E402
from botorch.models import ModelListGP as BtModelListGP  # noqa: E402
from botorch.optim import initializers as bt_initializers  # noqa: E402
from botorch.optim import optimize_acqf as bt_optimize_acqf  # noqa: E402
from botorch.optim import parameter_constraints as bt_pc  # noqa: E402
from botorch.sampling.normal import SobolQMCNormalSampler as BtSampler  # noqa: E402
from botorch.test_functions import synthetic as bt_synthetic  # noqa: E402
from botorch.utils import sampling as bt_sampling  # noqa: E402
from gpytorch.mlls import ExactMarginalLogLikelihood as BtMLL  # noqa: E402
from gpytorch.mlls import SumMarginalLogLikelihood as BtSumMLL  # noqa: E402

from botorch_rs.acquisition import analytic as rs_analytic  # noqa: E402
from botorch_rs.acquisition import logei as rs_logei  # noqa: E402
from botorch_rs.acquisition import monte_carlo as rs_mc  # noqa: E402
from botorch_rs.acquisition import objective as rs_objective  # noqa: E402
from botorch_rs.acquisition.utils import prune_inferior_points as rs_prune  # noqa: E402
from botorch_rs.fit import fit_gpytorch_mll as rs_fit  # noqa: E402
from botorch_rs.mlls import ExactMarginalLogLikelihood as RsMLL  # noqa: E402
from botorch_rs.mlls import SumMarginalLogLikelihood as RsSumMLL  # noqa: E402
from botorch_rs.models import ModelListGP as RsModelListGP  # noqa: E402
from botorch_rs.optim import initializers as rs_initializers  # noqa: E402
from botorch_rs.optim import optimize_acqf as rs_optimize_acqf  # noqa: E402
from botorch_rs.optim import parameter_constraints as rs_pc  # noqa: E402
from botorch_rs.sampling import SobolQMCNormalSampler as RsSampler  # noqa: E402
from botorch_rs.test_functions import synthetic as rs_synthetic  # noqa: E402
from botorch_rs.utils import sampling as rs_sampling  # noqa: E402
from parity_helpers import (  # noqa: E402
    DT,
    Checker,
    ModelSpec,
    bt_mll_value_and_grad,
    bt_mll_values,
    build_pair,
    dense_model_list,
    feasible_start_generator,
    feasible_starts,
    list_pair,
    log_err,
    make_data,
    model_pair,
    patched_lower_bound,
    perturbed_copy,
    points_in,
    rel_err,
    sampler_pair,
    values_and_grads,
)

pytestmark = [pytest.mark.equivalence, pytest.mark.filterwarnings("ignore")]

SEED = 20261010
VALUE_TOL = 1e-9
GRAD_TOL = 1e-8
FIT_TOL = 1e-6
OPT_TOL = 1e-3
NOISE_FACTOR = 10.0
SINGULAR = 1e-12
JITTER_TOL = 0.1
CHECK = Checker("equivalence suite")


@pytest.fixture(autouse=True, scope="module")
def _single_threaded_torch() -> Iterator[None]:
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(threads)


def _t(v: Any) -> Any:
    return torch.as_tensor(v, dtype=DT)


def rng_for(*keys: Any) -> random.Random:
    """A generator seeded by ``keys`` (stable across runs and platforms)."""
    return random.Random(zlib.crc32(repr((SEED, *keys)).encode()))


# ---------------------------------------------------------------------------
# Scenarios
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Scenario:
    label: str
    specs: tuple[ModelSpec, ...]
    is_list: bool
    seed: int

    @property
    def batched(self) -> bool:
        return any(s.batch for s in self.specs)

    @property
    def d(self) -> int:
        return self.specs[0].d


def _scenarios() -> list[Scenario]:
    r = random.Random(SEED)
    out: list[Scenario] = []
    ranges = [(0.0, 1.0), (-2.0, 3.0), (1.0, 4.0)]
    for i in range(14):
        spec = ModelSpec(
            n=[6, 10, 16][i % 3],
            d=[1, 2, 3, 4][i % 4],
            m=[1, 2, 3][(i // 2) % 3],
            fixed_noise=r.random() < 0.35,
            normalize=r.random() < 0.5,
            standardize=r.random() < 0.75,
            kernel=r.choice(["rbf", "matern52"]),
            batch=(2,) if i % 5 == 4 else (),
            lo=(rg := r.choice(ranges))[0],
            hi=rg[1],
        )
        out.append(Scenario(f"{i:02d}_{spec.label}", (spec,), False, 100 + i))
    lists = [
        (ModelSpec(10, 2), ModelSpec(8, 2, kernel="matern52", normalize=True)),
        (ModelSpec(12, 3, m=2), ModelSpec(9, 3, fixed_noise=True)),
        (ModelSpec(8, 1), ModelSpec(11, 1, standardize=False), ModelSpec(7, 1, m=2)),
        (
            ModelSpec(16, 4, normalize=True, lo=-1.0, hi=2.0),
            ModelSpec(10, 4, kernel="matern52", lo=-1.0, hi=2.0),
        ),
    ]
    for j, specs in enumerate(lists):
        label = "list_" + "+".join(s.label for s in specs)
        out.append(Scenario(f"{14 + j:02d}_{label}", specs, True, 200 + j))
    return out


SCENARIOS = _scenarios()
ACQF_SCENARIOS = [s for s in SCENARIOS if not s.batched]


def _build(sc: Scenario) -> tuple[Any, Any, Any]:
    if sc.is_list:
        return list_pair(sc.specs, sc.seed)
    return model_pair(sc.specs[0], sc.seed)


@pytest.fixture(scope="module", params=SCENARIOS, ids=[s.label for s in SCENARIOS])
def pair(request: pytest.FixtureRequest) -> tuple[Scenario, Any, Any, Any]:
    sc = request.param
    return (sc, *_build(sc))


@pytest.fixture(scope="module", params=ACQF_SCENARIOS, ids=[s.label for s in ACQF_SCENARIOS])
def acqf_pair(request: pytest.FixtureRequest) -> tuple[Scenario, Any, Any, Any, list[Any]]:
    """An unbatched model pair, its training inputs and two BoTorch twins with
    perturbed training inputs (see ``parity_helpers.perturbed_copy``)."""
    sc = request.param
    bt, rs, X = _build(sc)
    return sc, bt, rs, X, [perturbed_copy(bt, seed=k) for k in (1, 2)]


def compare(quantity: str, a: Any, e: Any, log: bool = False, tol: float = VALUE_TOL) -> None:
    if log:
        CHECK.check_log(quantity, a, e, tol)
    else:
        CHECK.check(quantity, a, e, tol, atol=1e-300)


def compare_grad(quantity: str, a: Any, e: Any, scale: float = 1.0) -> None:
    err = max(rel_err(a[i], e[i], atol=1e-14) for i in range(e.shape[0])) * scale
    CHECK.record(quantity, err, GRAD_TOL)
    assert err <= GRAD_TOL, (quantity, err)


def bt_eval(fn: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluates BoTorch code with dense ModelListGP covariances."""
    with dense_model_list():
        return fn(*args, **kwargs)


# ---------------------------------------------------------------------------
# Models: initialization, MLL, posteriors, samples, fits
# ---------------------------------------------------------------------------


def _compare_posteriors(label: str, pb: Any, pr: Any) -> None:
    assert tuple(pr.mean.shape) == tuple(pb.mean.shape), (label, pr.mean.shape, pb.mean.shape)
    compare(f"{label} mean", pr.mean, pb.mean)
    compare(f"{label} variance", pr.variance, pb.variance)
    compare(
        f"{label} covariance", pr.distribution.covariance_matrix, pb.distribution.covariance_matrix
    )


def test_models(pair: tuple) -> None:
    sc, bt, rs, X = pair
    d = sc.d
    if not sc.is_list:
        spec = sc.specs[0]
        b0, r0 = build_pair(spec, *make_data(spec, sc.seed))
        b_params = {k: v.detach() for k, v in b0.named_parameters()}
        r_params = dict(r0.named_parameters())
        assert list(b_params) == list(r_params)
        for name in b_params:
            compare("default raw parameters", r_params[name], b_params[name], tol=1e-12)
        compare("default MLL values", RsMLL(r0.likelihood, r0)(), bt_mll_values(b0))
        values, grad = bt_mll_value_and_grad(bt)
        rs_values, rs_grad = RsMLL(rs.likelihood, rs).value_and_grad()
        compare("MLL values", rs_values, values.reshape(-1).tolist() if values.dim() else values)
        CHECK.check("MLL gradient", rs_grad, grad, VALUE_TOL, atol=1e-14)
        batch = spec.batch
    else:
        bt.train()
        with torch.no_grad():
            sb = BtSumMLL(bt.likelihood, bt)(bt(*bt.train_inputs), bt.train_targets)
        bt.eval()
        compare("SumMarginalLogLikelihood", RsSumMLL(rs.likelihood, rs)(), sb)
        batch = ()
    r = rng_for("models", sc.label)
    shapes: list[tuple[int, ...]] = [(3, d)]
    shapes += [(*batch, 3, d), (2, *([1] * len(batch)), 3, d)] if batch else [(5, 1, d), (2, 3, d)]
    for i, shape in enumerate(shapes):
        Xt = points_in(X, shape, seed=sc.seed + i)
        for noise in (False, True):
            with torch.no_grad():
                pb = bt_eval(bt.posterior, Xt, observation_noise=noise)
            pr = rs.posterior(Xt, observation_noise=noise)
            _compare_posteriors("posterior (obs. noise)" if noise else "posterior", pb, pr)
    m = rs.num_outputs
    w = _t([r.uniform(-1.0, 1.0) for _ in range(m)])
    Xt = points_in(X, (*batch, 3, d), seed=sc.seed + 7)
    with torch.no_grad():
        pb = bt_eval(
            bt.posterior, Xt, posterior_transform=bt_objective.ScalarizedPosteriorTransform(w, 0.3)
        )
    pr = rs.posterior(Xt, posterior_transform=rs_objective.ScalarizedPosteriorTransform(w, 0.3))
    _compare_posteriors("scalarized posterior", pb, pr)
    # Samples from explicit base samples (BoTorch's S x [batch] x q x m layout).
    with torch.no_grad():
        pb = bt_eval(bt.posterior, Xt)
    pr = rs.posterior(Xt)
    Z = torch.randn(
        4, *pb.base_sample_shape, generator=torch.Generator().manual_seed(sc.seed), dtype=DT
    )
    with torch.no_grad():
        sb = bt_eval(pb.rsample_from_base_samples, torch.Size([4]), Z)
    compare("samples from base samples", pr.rsample_from_base_samples(torch.Size([4]), Z), sb)


def test_fit(pair: tuple) -> None:
    sc = pair[0]
    if sc.is_list:
        bts, rss = [], []
        for i, spec in enumerate(sc.specs):
            b, r = build_pair(spec, *make_data(spec, sc.seed + 50 + i))
            bts.append(b)
            rss.append(r)
        lb, lr = BtModelListGP(*bts), RsModelListGP(*rss)
        torch.manual_seed(0)
        bt_fit(BtSumMLL(lb.likelihood, lb))
        rs_fit(RsSumMLL(lr.likelihood, lr))
        pairs = list(zip(lb.models, lr.models, strict=True))
    else:
        spec = sc.specs[0]
        b, r = build_pair(spec, *make_data(spec, sc.seed + 50))
        torch.manual_seed(0)
        bt_fit(BtMLL(b.likelihood, b))
        rs_fit(RsMLL(r.likelihood, r))
        pairs = [(b, r)]
    for b, r in pairs:
        bt_losses = -bt_mll_values(b).reshape(-1)
        rs_losses = -torch.as_tensor(RsMLL(r.likelihood, r)(), dtype=DT).reshape(-1)
        excess = (rs_losses - bt_losses).max().item()
        CHECK.check_bound("fit: final loss excess over BoTorch (per element)", excess, FIT_TOL)


# ---------------------------------------------------------------------------
# Analytic acquisition functions
# ---------------------------------------------------------------------------

ANALYTIC = [
    "ExpectedImprovement",
    "LogExpectedImprovement",
    "ProbabilityOfImprovement",
    "LogProbabilityOfImprovement",
    "UpperConfidenceBound",
    "PosteriorMean",
    "PosteriorStandardDeviation",
]


def _transforms(m: int, r: random.Random) -> tuple[Any, Any]:
    """A posterior-transform pair (required for m > 1, random for m = 1)."""
    if m == 1 and r.random() < 0.5:
        return None, None
    w = [r.uniform(-1.0, 1.0) for _ in range(m)]
    off = r.uniform(-0.5, 0.5)
    return (
        bt_objective.ScalarizedPosteriorTransform(_t(w), off),
        rs_objective.ScalarizedPosteriorTransform(_t(w), off),
    )


def test_analytic(acqf_pair: tuple) -> None:
    sc, bt, rs, X, _ = acqf_pair
    d, m = sc.d, rs.num_outputs
    r = rng_for("analytic", sc.label)
    Xt = points_in(X, (6, 1, d), seed=sc.seed + 1)
    t_bt, t_rs = _transforms(m, r)
    with torch.no_grad():
        p = bt_eval(bt.posterior, Xt, posterior_transform=t_bt)
    mean, sd = p.mean.reshape(-1), p.variance.reshape(-1).clamp_min(0).sqrt()
    for name in ANALYTIC:
        maximize = r.random() < 0.7
        kw: dict[str, Any] = {"maximize": maximize}
        if name == "UpperConfidenceBound":
            kw["beta"] = _t(r.uniform(0.1, 2.0))
        elif name not in ("PosteriorMean", "PosteriorStandardDeviation"):
            shift = r.uniform(-0.5, 1.0) * sd.mean()
            kw["best_f"] = mean.max() + shift if maximize else mean.min() - shift
        a_bt = getattr(bt_analytic, name)(bt, posterior_transform=t_bt, **kw)
        a_rs = getattr(rs_analytic, name)(rs, posterior_transform=t_rs, **kw)
        v_bt, g_bt = bt_eval(values_and_grads, a_bt, Xt)
        v_rs, g_rs = values_and_grads(a_rs, Xt)
        compare(f"{name} value", v_rs, v_bt, log=a_rs._log)
        compare_grad(f"{name} grad", g_rs, g_bt)
    if m >= 2:
        with torch.no_grad():
            post = bt_eval(bt.posterior, Xt)
        means = post.mean.squeeze(-2)
        sds = post.variance.squeeze(-2).clamp_min(0).sqrt()
        obj = r.randrange(m)
        other = (obj + 1) % m
        med, s = means[:, other].median(), sds[:, other].mean()
        bounds = r.choice([(med, None), (None, med), (med - s, med + 0.5 * s)])
        cons = {other: tuple(None if b is None else _t(float(b)) for b in bounds)}
        maximize = r.random() < 0.7
        best = means[:, obj].max() if maximize else means[:, obj].min()
        for name in ("LogConstrainedExpectedImprovement", "ConstrainedExpectedImprovement"):
            kw = dict(best_f=best.clone(), objective_index=obj, constraints=cons, maximize=maximize)
            a_bt = getattr(bt_analytic, name)(bt, **kw)
            a_rs = getattr(rs_analytic, name)(rs, **kw)
            v_bt, g_bt = bt_eval(values_and_grads, a_bt, Xt)
            v_rs, g_rs = values_and_grads(a_rs, Xt)
            compare(f"{name} value", v_rs, v_bt, log=a_rs._log)
            compare_grad(f"{name} grad", g_rs, g_bt)


# ---------------------------------------------------------------------------
# Monte-Carlo acquisition functions
# ---------------------------------------------------------------------------


def generic_objective(Y: Any, X: Any = None) -> Any:
    out = Y[..., 0] + 0.3 * torch.sin(Y[..., -1])
    if X is not None:
        out = out - 0.1 * (X**2).sum(-1)
    return out


@dataclass
class Variant:
    """An objective / posterior-transform combination."""

    label: str
    o_bt: Any = None
    o_rs: Any = None
    t_bt: Any = None
    t_rs: Any = None
    m: int = 1  # outputs of the (transformed) samples


def variants(m: int, means: Any, r: random.Random) -> list[Variant]:
    out = [Variant("identity")] if m == 1 else []
    if m > 1:
        t_bt, t_rs = _transforms(m, r)
        out.append(Variant("posterior transform", t_bt=t_bt, t_rs=t_rs))
    w = _t([r.uniform(-1.0, 1.0) for _ in range(m)])
    out.append(
        Variant("linear", bt_objective.LinearMCObjective(w), rs_objective.LinearMCObjective(w), m=m)
    )
    out.append(
        Variant(
            "generic",
            bt_objective.GenericMCObjective(generic_objective),
            rs_objective.GenericMCObjective(generic_objective),
            m=m,
        )
    )
    c = means[:, -1].median()
    cons = [lambda Y: Y[..., -1] - c]
    eta = _t([r.uniform(0.01, 0.2)])
    cost = _t(r.uniform(0.0, 3.0))
    out.append(
        Variant(
            "constrained",
            bt_objective.ConstrainedMCObjective(lambda Y, X=None: Y[..., 0], cons, cost, eta),
            rs_objective.ConstrainedMCObjective(lambda Y, X=None: Y[..., 0], cons, cost, eta),
            m=m,
        )
    )
    return out


def _sample_stats(bt: Any, X: Any, t_bt: Any = None) -> tuple[Any, Any]:
    with torch.no_grad():
        p = bt_eval(bt.posterior, X, posterior_transform=t_bt)
    k = p.mean.shape[-1]
    return p.mean.reshape(-1, k), p.variance.reshape(-1, k).clamp_min(0).sqrt()


def random_constraints(means: Any, sds: Any, r: random.Random) -> tuple[str, list[Any], Any, Any]:
    """A random outcome-constraint set: (label, constraints, eta, fat list)."""
    c = means[:, -1].median() + r.uniform(-0.5, 0.5) * sds[:, -1].mean()
    mu0 = means[:, 0].mean()
    r2 = (r.uniform(0.5, 2.0) * sds[:, 0].mean()) ** 2

    def lin(Y: Any) -> Any:
        return Y[..., -1] - c

    def quad(Y: Any) -> Any:
        return (Y[..., 0] - mu0) ** 2 - r2

    def prob(Y: Any) -> Any:
        return torch.sigmoid(2.0 * (c - Y[..., -1]))

    choice = r.randrange(3)
    if choice == 0:
        return "one constraint", [lin], _t([r.uniform(1e-3, 0.1)]), None
    if choice == 1:
        return (
            "two constraints",
            [lin, quad],
            _t([r.uniform(1e-3, 0.1), r.uniform(0.01, 0.5)]),
            None,
        )
    return (
        "probability constraint",
        [lin, prob],
        _t([r.uniform(1e-3, 0.1), 1.0]),
        [r.random() < 0.5, None],
    )


MC = [
    ("qExpectedImprovement", bt_mc, rs_mc, True),
    ("qProbabilityOfImprovement", bt_mc, rs_mc, True),
    ("qLogExpectedImprovement", bt_logei, rs_logei, True),
    ("qSimpleRegret", bt_mc, rs_mc, False),
    ("qUpperConfidenceBound", bt_mc, rs_mc, False),
]


def _objective_values(v: Variant, means: Any) -> Any:
    with torch.no_grad():
        if v.o_bt is None:
            return means[:, 0]
        if v.label == "constrained":
            return v.o_bt(means.unsqueeze(0)).squeeze(0)
        return v.o_bt(means.unsqueeze(0), X=None).squeeze(0)


def singular_posterior(bt: Any, point_sets: list[Any], transform: Any = None) -> bool:
    """Whether a joint posterior covariance (one per point set) has a
    numerically singular output block (smallest eigenvalue below ``SINGULAR``
    times the largest): whether its Cholesky factorization needs jitter then
    depends on rounding, and BoTorch (LAPACK) and botorch_rs (nalgebra) may
    decide differently, which changes the samples by O(sqrt(jitter))."""
    for points in point_sets:
        with torch.no_grad():
            cov = bt_eval(bt.posterior, points, posterior_transform=transform)
            cov = cov.distribution.covariance_matrix
        n = points.shape[-2]
        for t in range(cov.shape[-1] // n):
            ev = torch.linalg.eigvalsh(cov[t * n : (t + 1) * n, t * n : (t + 1) * n])
            if ev.min().item() < SINGULAR * ev.max().item():
                return True
    return False


def compare_mc(
    label: str,
    bt: Any,
    twins: list[Any],
    make_bt: Any,
    a_rs: Any,
    Xt: Any,
    log: bool,
    point_sets: list[Any],
    transform: Any = None,
) -> None:
    """Compares a Monte-Carlo acquisition function with BoTorch's, built by
    ``make_bt(model) -> (acqf, sampler, base samples)`` on ``bt`` and on its
    perturbed twins: the tolerance is ``max(1e-9, NOISE_FACTOR x BoTorch's
    change under the perturbation)`` (``1e-8`` for gradients), and the error
    is recorded relative to it (scaled to the nominal tolerance).

    If the joint posterior over one of ``point_sets`` is numerically singular
    (see :func:`singular_posterior`), the results depend on the jitter
    decision: they are recorded separately and only checked for finiteness and
    loose agreement (``JITTER_TOL``)."""
    err_fn = log_err if log else rel_err

    def evaluate(model: Any) -> tuple[Any, Any]:
        a, s, Z = make_bt(model)
        out = bt_eval(values_and_grads, a, Xt)
        assert torch.equal(s.base_samples, Z), "BoTorch replaced the base samples"
        return out

    v_bt, g_bt = evaluate(bt)
    v_rs, g_rs = values_and_grads(a_rs, Xt)
    err_v, err_g = err_fn(v_rs, v_bt), rel_err(g_rs, g_bt, atol=1e-14)
    if singular_posterior(bt, point_sets, transform):
        CHECK.record(
            "MC values on numerically singular posteriors (jitter-dependent)", err_v, JITTER_TOL
        )
        CHECK.record("MC evaluations on numerically singular posteriors (count)", _count(), None)
        assert torch.isfinite(v_rs).all() and torch.isfinite(g_rs).all()
        assert err_v <= JITTER_TOL, (label, "value (singular posterior)", err_v)
        return
    floor_v = floor_g = 0.0
    for twin in twins:
        v_tw, g_tw = evaluate(twin)
        floor_v = max(floor_v, err_fn(v_tw, v_bt))
        floor_g = max(floor_g, rel_err(g_tw, g_bt, atol=1e-14))
    tol_v = max(VALUE_TOL, NOISE_FACTOR * floor_v)
    tol_g = max(GRAD_TOL, NOISE_FACTOR * floor_g)
    CHECK.record("Monte-Carlo values: raw error (all non-singular evaluations)", err_v, None)
    CHECK.record("Monte-Carlo gradients: raw error (all non-singular evaluations)", err_g, None)
    CHECK.record(f"{label} value", err_v * VALUE_TOL / tol_v, VALUE_TOL)
    CHECK.record(f"{label} grad", err_g * GRAD_TOL / tol_g, GRAD_TOL)
    CHECK.record("BoTorch noise floor: MC values (1e-14 input perturbation)", floor_v, None)
    CHECK.record("BoTorch noise floor: MC gradients (1e-14 input perturbation)", floor_g, None)
    assert err_v <= tol_v, (label, "value", err_v, tol_v, floor_v)
    assert err_g <= tol_g, (label, "grad", err_g, tol_g, floor_g)


_SINGULAR_COUNT = [0]


def _count() -> float:
    _SINGULAR_COUNT[0] += 1
    return float(_SINGULAR_COUNT[0])


def _bt_maker(cls: Any, S: int, q: int, m: int, seed: int, **kwargs: Any) -> Any:
    """``make_bt`` for :func:`compare_mc`: a fresh BoTorch sampler with the
    same base samples for every model."""

    def make(model: Any) -> tuple[Any, Any, Any]:
        s_bt, _, Z = sampler_pair(S, q, m, seed=seed)
        return cls(model, sampler=s_bt, **kwargs), s_bt, Z

    return make


def test_mc(acqf_pair: tuple) -> None:
    sc, bt, rs, X, twins = acqf_pair
    d, m = sc.d, rs.num_outputs
    r = rng_for("mc", sc.label)
    Xs = points_in(X, (8, d), seed=sc.seed + 2)
    means, _ = _sample_stats(bt, Xs)
    for v in variants(m, means, r):
        v_means, v_sds = _sample_stats(bt, Xs, v.t_bt)
        best = _objective_values(v, v_means).max()
        q, n_pending = r.choice([1, 2, 3]), r.choice([0, 0, 2])
        Xt = points_in(X, (2, q, d), seed=sc.seed + q)
        pending = points_in(X, (n_pending, d), seed=sc.seed + 4) if n_pending else None
        for name, bt_mod, rs_mod, supports_constraints in MC:
            if name.startswith("qLog") and v.label == "constrained":
                continue
            kw: dict[str, Any] = {}
            if name in (
                "qExpectedImprovement",
                "qProbabilityOfImprovement",
                "qLogExpectedImprovement",
            ):
                kw["best_f"] = best + r.uniform(-0.5, 0.5) * v_sds.mean()
            if name == "qProbabilityOfImprovement":
                kw["tau"] = _t(r.uniform(0.01, 0.5))
            if name == "qUpperConfidenceBound":
                kw["beta"] = r.uniform(0.1, 2.0)
            if name == "qLogExpectedImprovement":
                kw["fat"] = r.random() < 0.7
            label = f"{name} [{v.label}"
            if supports_constraints and v.label != "constrained" and r.random() < 0.6:
                c_label, cons, eta, fat = random_constraints(v_means, v_sds, r)
                if fat is not None and name != "qLogExpectedImprovement":
                    cons, eta = cons[:1], eta[:1]
                    c_label = "one constraint"
                elif fat is not None:
                    kw["fat"] = fat
                kw.update(constraints=cons, eta=eta)
                label += f", {c_label}"
            label += "]"
            S, seed = r.choice([32, 64]), r.randrange(10**6)
            common = dict(objective=v.o_bt, posterior_transform=v.t_bt, X_pending=pending, **kw)
            make_bt = _bt_maker(getattr(bt_mod, name), S, q + n_pending, v.m, seed, **common)
            _, s_rs, _ = sampler_pair(S, q + n_pending, v.m, seed=seed)
            a_rs = getattr(rs_mod, name)(
                rs,
                sampler=s_rs,
                **{**common, "objective": v.o_rs, "posterior_transform": v.t_rs},
            )
            sets = [torch.cat([Xt[b]] + ([pending] if n_pending else [])) for b in range(2)]
            compare_mc(label, bt, twins, make_bt, a_rs, Xt, a_rs._log, sets, v.t_bt)


def test_qlognei(acqf_pair: tuple) -> None:
    sc, bt, rs, X, twins = acqf_pair
    d, m, n = sc.d, rs.num_outputs, X.shape[0]
    r = rng_for("qlognei", sc.label)
    means, _ = _sample_stats(bt, X)
    lb_points = points_in(X, (32, d), seed=sc.seed + 3)
    for v in variants(m, means, r):
        if v.label == "constrained":
            continue
        v_means, v_sds = _sample_stats(bt, X, v.t_bt)
        q, n_pending = r.choice([1, 2]), r.choice([0, 2])
        incremental = r.random() < 0.6
        Xt = points_in(X, (2, q, d), seed=sc.seed + 5)
        pending = points_in(X, (n_pending, d), seed=sc.seed + 6) if n_pending else None
        kind = r.choice(["none", "tight", "infeasible"])
        kw: dict[str, Any] = {"fat": r.random() < 0.7}
        if kind == "tight":
            c = (v_means[:, -1] - 0.5 * v_sds[:, -1]).min()
            kw.update(constraints=[lambda Y, c=c: Y[..., -1] - c], eta=_t([r.uniform(1e-3, 0.05)]))
        elif kind == "infeasible":
            c = (v_means[:, -1] - 30.0 * v_sds[:, -1] - 10.0).min()
            kw.update(constraints=[lambda Y, c=c: Y[..., -1] - c], eta=_t([1e-2]))
        D, seed = n + n_pending + q, r.randrange(10**6)
        common = dict(
            X_baseline=X, X_pending=pending, prune_baseline=False, incremental=incremental, **kw
        )

        def make_bt(
            model: Any, v: Variant = v, common: dict = common, D: int = D, seed: int = seed
        ) -> Any:
            s_bt, _, Z = sampler_pair(64, D, v.m, seed=seed)
            with patched_lower_bound(lb_points):
                a = bt_logei.qLogNoisyExpectedImprovement(
                    model,
                    sampler=s_bt,
                    objective=v.o_bt,
                    posterior_transform=v.t_bt,
                    cache_root=False,
                    **common,
                )
            return _LowerBoundPatched(a, lb_points), s_bt, Z

        _, s_rs, _ = sampler_pair(64, D, v.m, seed=seed)
        a_rs = rs_logei.qLogNoisyExpectedImprovement(
            rs,
            sampler=s_rs,
            objective=v.o_rs,
            posterior_transform=v.t_rs,
            lower_bound_points=lb_points,
            **common,
        )
        label = f"qLogNoisyExpectedImprovement [{v.label}, {kind} baseline]"
        head = [X] + ([pending] if n_pending and incremental else [])
        tail = [pending] if n_pending and not incremental else []
        sets = [torch.cat(head + [Xt[b]] + tail) for b in range(2)]
        compare_mc(label, bt, twins, make_bt, a_rs, Xt, True, sets, v.t_bt)


class _LowerBoundPatched:
    """A BoTorch acquisition function evaluated with explicit lower-bound
    points (BoTorch computes the bound at evaluation time)."""

    def __init__(self, acqf: Any, points: Any) -> None:
        self.acqf, self.points = acqf, points

    def __call__(self, X: Any) -> Any:
        with patched_lower_bound(self.points):
            return self.acqf(X)


def test_qlogpf(acqf_pair: tuple) -> None:
    sc, bt, rs, X, twins = acqf_pair
    d, m = sc.d, rs.num_outputs
    r = rng_for("qlogpf", sc.label)
    t_bt, t_rs = _transforms(m, r) if m > 1 and r.random() < 0.3 else (None, None)
    m_eff = 1 if t_bt is not None else m
    means, sds = _sample_stats(bt, points_in(X, (8, d), seed=sc.seed), t_bt)
    for _ in range(2):
        c_label, cons, eta, fat_list = random_constraints(means, sds, r)
        fat = fat_list if fat_list is not None else r.random() < 0.5
        q, seed = r.choice([1, 2, 3]), r.randrange(10**6)
        Xt = points_in(X, (2, q, d), seed=sc.seed + 11 * q)
        common = dict(constraints=cons, eta=eta, fat=fat)
        make_bt = _bt_maker(
            bt_logei.qLogProbabilityOfFeasibility,
            64,
            q,
            m_eff,
            seed,
            posterior_transform=t_bt,
            **common,
        )
        _, s_rs, _ = sampler_pair(64, q, m_eff, seed=seed)
        a_rs = rs_logei.qLogProbabilityOfFeasibility(
            rs, sampler=s_rs, posterior_transform=t_rs, **common
        )
        compare_mc(
            f"qLogProbabilityOfFeasibility [{c_label}]",
            bt,
            twins,
            make_bt,
            a_rs,
            Xt,
            True,
            [Xt[b] for b in range(2)],
            t_bt,
        )


def test_prune(acqf_pair: tuple) -> None:
    """The pruned sets agree unless BoTorch's own result changes under the
    1e-14 input perturbation (near-ties of the per-sample maximizers)."""
    sc, bt, rs, X, twins = acqf_pair
    m, d = rs.num_outputs, sc.d
    r = rng_for("prune", sc.label)
    Xp = torch.cat([X, points_in(X, X.shape, seed=sc.seed + 8)])
    means, _ = _sample_stats(bt, Xp)
    for v in variants(m, means, r):
        v_means, v_sds = _sample_stats(bt, Xp, v.t_bt)
        cons = None
        if r.random() < 0.6:
            c = v_means[:, -1].median() + r.choice([0.0, -50.0]) * (v_sds[:, -1].mean() + 1.0)
            cons = [lambda Y, c=c: Y[..., -1] - c]
        S, seed = r.choice([128, 256]), r.randrange(1000)

        def bt_pruned(
            model: Any, v: Variant = v, cons: Any = cons, S: int = S, seed: int = seed
        ) -> tuple[Any, Any]:
            s_bt = BtSampler(sample_shape=torch.Size([S]), seed=seed)
            out = bt_eval(
                bt_prune,
                model,
                Xp,
                objective=v.o_bt,
                posterior_transform=v.t_bt,
                constraints=cons,
                sampler=s_bt,
            )
            return out, s_bt.base_samples

        pruned_bt, Z = bt_pruned(bt)
        s_rs = RsSampler(sample_shape=[S], seed=0)
        s_rs.base_samples = Z.clone()
        pruned_rs = rs_prune(
            rs, Xp, objective=v.o_rs, posterior_transform=v.t_rs, constraints=cons, sampler=s_rs
        )

        def same(a: Any, b: Any) -> bool:
            return a.shape == b.shape and bool(torch.equal(a, b))

        ok = same(pruned_rs, pruned_bt)
        noisy = not ok and any(not same(bt_pruned(t)[0], pruned_bt) for t in twins)
        singular = not ok and not noisy and singular_posterior(bt, [Xp], v.t_bt)
        CHECK.record(
            "prune_inferior_points: sets differing beyond BoTorch's noise",
            float(not (ok or noisy or singular)),
            0.0,
        )
        CHECK.record("prune_inferior_points: sets differing (any)", float(not ok), None)
        assert ok or noisy or singular, (sc.label, v.label, cons is not None, d)


# ---------------------------------------------------------------------------
# optimize_acqf
# ---------------------------------------------------------------------------


def _bounds_of(X: Any) -> Any:
    lo, hi = X.min(dim=0).values, X.max(dim=0).values
    span = (hi - lo).clamp_min(1e-3)
    return torch.stack([lo - 0.05 * span, hi + 0.05 * span])


def _opt_acqfs(sc: Scenario, bt: Any, rs: Any, X: Any, r: random.Random) -> tuple[Any, Any, int]:
    """An acquisition-function pair to optimize and its ``q``."""
    m = rs.num_outputs
    if m == 1 and r.random() < 0.5:
        best = _sample_stats(bt, X)[0].max()
        return (
            bt_analytic.LogExpectedImprovement(bt, best_f=best),
            rs_analytic.LogExpectedImprovement(rs, best_f=best),
            1,
        )
    w = _t([r.uniform(0.2, 1.0) for _ in range(m)])
    means, sds = _sample_stats(bt, X)
    best = (means @ w).max()
    q = r.choice([1, 2])
    s_bt, s_rs, _ = sampler_pair(64, q, m, seed=r.randrange(10**6))
    obj_bt = None if m == 1 else bt_objective.LinearMCObjective(w)
    obj_rs = None if m == 1 else rs_objective.LinearMCObjective(w)
    best = means[:, 0].max() if m == 1 else best
    return (
        bt_logei.qLogExpectedImprovement(bt, best_f=best, sampler=s_bt, objective=obj_bt),
        rs_logei.qLogExpectedImprovement(rs, best_f=best, sampler=s_rs, objective=obj_rs),
        q,
    )


def _random_parameter_constraints(
    d: int, bounds: Any, q: int, r: random.Random
) -> tuple[dict[str, Any], Any]:
    """Random feasible linear (intra-, inter-point, equality) and nonlinear
    constraints around a random interior point."""
    lo, span = bounds[0], bounds[1] - bounds[0]
    x_star = lo + span * _t([r.uniform(0.3, 0.7) for _ in range(d)])
    out: dict[str, Any] = {}
    k = r.randint(1, d)
    idx = sorted(r.sample(range(d), k))
    coef = _t([-r.uniform(0.5, 1.5) for _ in idx])
    rhs = float((coef * x_star[idx]).sum() - 0.2 * (coef.abs() * span[idx]).sum())
    out["inequality_constraints"] = [(torch.tensor(idx), coef, rhs)]
    kind = r.choice(["intra", "equality", "inter", "nonlinear"])
    if kind == "equality" and d >= 2:
        j = r.sample(range(d), 2)
        c = _t([1.0, -r.uniform(0.5, 1.5)])
        out["equality_constraints"] = [(torch.tensor(j), c, float((c * x_star[j]).sum()))]
    elif kind == "inter" and q > 1:
        f = r.randrange(d)
        out["inequality_constraints"].append(
            (torch.tensor([[0, f], [1, f]]), _t([1.0, -1.0]), float(-0.3 * span[f]))
        )
    elif kind == "nonlinear":
        radius = float(0.4 * span.min())
        center = x_star.clone()

        def ball(x: Any, center: Any = center, radius: float = radius, span: Any = span) -> Any:
            return radius**2 - (((x - center) / span * span.min()) ** 2).sum(-1)

        out["nonlinear_inequality_constraints"] = [(ball, True)]
    return out, x_star.expand(q, d)


def test_optimize(acqf_pair: tuple) -> None:
    """``optimize_acqf`` (a) from identical initial conditions (BoTorch's
    ``gen_batch_initial_conditions``): at least 80% of the restarts reach
    BoTorch's result (relative value error <= 1e-6); (b) end to end over three
    seeds, where BoTorch draws its initial conditions from torch's RNG, so the
    comparison is statistical: the best value over the seeds is at least the
    median of BoTorch's per-seed values (minus 1e-3 max(1, |value|)); results
    are within the bounds and feasible, and the returned values are BoTorch's
    values at the returned candidates. With nonlinear constraints (where
    BoTorch's SLSQP takes seconds per run) four restarts and one end-to-end
    run (validity only) are used."""
    sc, bt, rs, X, _ = acqf_pair
    d = sc.d
    r = rng_for("optimize", sc.label)
    a_bt, a_rs, q = _opt_acqfs(sc, bt, rs, X, r)
    bounds = _bounds_of(X)
    configs: list[tuple[str, dict[str, Any], Any]] = [("unconstrained", {}, None)]
    if d >= 2:
        configs.append(("constrained", *_random_parameter_constraints(d, bounds, q, r)))
    for label, cons, feasible_point in configs:
        nonlinear = "nonlinear_inequality_constraints" in cons
        linear = {k: v for k, v in cons.items() if k != "nonlinear_inequality_constraints"}
        restarts, seeds = (4, 1) if nonlinear else (6, 3)
        # (a) Identical initial conditions (BoTorch selects them with torch's RNG).
        torch.manual_seed(0)
        ics = bt_eval(
            bt_initializers.gen_batch_initial_conditions,
            a_bt,
            bounds,
            q=q,
            num_restarts=restarts,
            raw_samples=128,
            options={"seed": 0},
            **linear,
        )
        # One restart per problem (BoTorch otherwise solves all restarts in one
        # L-BFGS-B problem when its batched fast path is unavailable).
        ics = feasible_starts(ics, cons.get("nonlinear_inequality_constraints"), feasible_point)
        kw = dict(
            q=q,
            num_restarts=ics.shape[0],
            batch_initial_conditions=ics,
            options={"batch_limit": 1},
            return_best_only=False,
            retry_on_optimization_warning=False,
            **cons,
        )
        _, v_bt = bt_eval(bt_optimize_acqf, a_bt, bounds, **kw)
        _, v_rs = rs_optimize_acqf(a_rs, bounds, **kw)
        errs = (v_rs - v_bt).abs() / v_bt.abs().clamp_min(1.0)
        agree = errs <= 1e-6
        CHECK.check_bound(
            f"optimize_acqf ({label}) from identical initial conditions: share of restarts "
            "not at BoTorch's result",
            1.0 - agree.double().mean().item(),
            0.2,
        )
        for e in errs[agree]:
            CHECK.record(
                f"optimize_acqf ({label}) from identical initial conditions: relative value error",
                e.item(),
                1e-6,
            )
        # (b) End to end.
        values_bt, values_rs = [], []
        for seed in range(seeds):
            opts = {"seed": seed}
            extra_bt: dict[str, Any] = {}
            extra_rs: dict[str, Any] = {}
            if nonlinear:
                nl = cons["nonlinear_inequality_constraints"]
                extra_bt = {
                    "ic_generator": feasible_start_generator(
                        bt_initializers.gen_batch_initial_conditions, nl, feasible_point
                    )
                }
                extra_rs = {
                    "ic_generator": feasible_start_generator(
                        rs_initializers.gen_batch_initial_conditions, nl, feasible_point
                    )
                }
            torch.manual_seed(seed)
            try:
                _, v_bt = bt_eval(
                    bt_optimize_acqf,
                    a_bt,
                    bounds,
                    q=q,
                    num_restarts=restarts,
                    raw_samples=128,
                    options=opts,
                    **cons,
                    **extra_bt,
                )
            except BtCandidateGenerationError:
                # Released BoTorch versions without the projection of
                # infeasible SLSQP results give up here.
                CHECK.record(
                    "optimize_acqf: BoTorch raised CandidateGenerationError (runs)", 1.0, None
                )
                v_bt = None
            x_rs, v_rs = rs_optimize_acqf(
                a_rs,
                bounds,
                q=q,
                num_restarts=restarts,
                raw_samples=128,
                options=opts,
                **cons,
                **extra_rs,
            )
            assert x_rs.shape == (q, d)
            assert bool(((x_rs >= bounds[0] - 1e-12) & (x_rs <= bounds[1] + 1e-12)).all())
            if cons:
                feasible = bt_pc.evaluate_feasibility(x_rs.unsqueeze(0), **cons)
                CHECK.record(
                    "optimize_acqf (constrained): infeasible results",
                    float(not feasible.all()),
                    0.0,
                )
                assert bool(feasible.all()), (sc.label, cons)
            with torch.no_grad():
                expected = bt_eval(a_bt, x_rs.unsqueeze(0)).squeeze()
            CHECK.check_log(
                f"optimize_acqf ({label}): returned value vs BoTorch's value",
                v_rs,
                expected,
                VALUE_TOL,
            )
            if v_bt is not None:
                values_bt.append(v_bt.item())
            values_rs.append(v_rs.item())
        if len(values_bt) < 3:
            continue
        median_bt = sorted(values_bt)[1]
        best_rs, best_bt = max(values_rs), max(values_bt)
        CHECK.check_bound(
            f"optimize_acqf ({label}) end to end: best value below BoTorch's median (statistical)",
            (median_bt - best_rs) / max(1.0, abs(median_bt)),
            OPT_TOL,
        )
        CHECK.record(
            f"optimize_acqf ({label}) end to end: best value below BoTorch's best (statistical)",
            max(0.0, (best_bt - best_rs) / max(1.0, abs(best_bt))),
            None,
        )


# ---------------------------------------------------------------------------
# Test functions, polytopes and parameter-constraint utilities
# ---------------------------------------------------------------------------

TEST_FUNCTIONS: list[tuple[str, list[int] | None]] = [
    ("Ackley", [2, 3, 5]),
    ("Beale", None),
    ("Branin", None),
    ("Bukin", None),
    ("Cosine8", None),
    ("DixonPrice", [2, 3, 4]),
    ("DropWave", None),
    ("EggHolder", None),
    ("Griewank", [2, 4, 6]),
    ("Hartmann", [3, 4, 6]),
    ("HolderTable", None),
    ("Levy", [2, 3, 4]),
    ("Michalewicz", [2, 5, 10]),
    ("Powell", [4, 8]),
    ("Rastrigin", [2, 3, 5]),
    ("Rosenbrock", [2, 3, 4]),
    ("Shekel", None),
    ("SixHumpCamel", None),
    ("StyblinskiTang", [2, 3, 5]),
    ("ThreeHumpCamel", None),
]


@pytest.mark.parametrize("name,dims", TEST_FUNCTIONS, ids=[n for n, _ in TEST_FUNCTIONS])
def test_test_functions(name: str, dims: list[int] | None) -> None:
    r = rng_for("test function", name)
    kw: dict[str, Any] = {"negate": r.random() < 0.5}
    if dims is not None:
        kw["dim"] = r.choice(dims)
    if name == "Shekel":
        kw["m"] = r.choice([5, 7, 10])
    f_bt = getattr(bt_synthetic, name)(**kw)
    f_rs = getattr(rs_synthetic, name)(**kw)
    assert f_rs.dim == f_bt.dim
    compare("test function bounds", f_rs.bounds, f_bt.bounds, tol=0.0)
    try:
        expected = f_bt.optimal_value
    except NotImplementedError:
        with pytest.raises(NotImplementedError):
            _ = f_rs.optimal_value
    else:
        CHECK.check("test function optimal values", _t(f_rs.optimal_value), _t(expected), 1e-15)
    lo, hi = f_bt.bounds
    X = lo + (hi - lo) * torch.rand(
        32, f_bt.dim, generator=torch.Generator().manual_seed(r.randrange(10**6)), dtype=DT
    )
    compare("test function values", f_rs.evaluate_true(X), f_bt.evaluate_true(X), tol=1e-12)
    compare("test function values (forward)", f_rs(X, noise=False), f_bt(X, noise=False), tol=1e-12)


@pytest.mark.parametrize("seed", range(4))
def test_polytopes(seed: int) -> None:
    r = rng_for("polytope", seed)
    d = r.choice([2, 3, 4])
    lo = _t([r.uniform(-2.0, 1.0) for _ in range(d)])
    bounds = torch.stack([lo, lo + _t([r.uniform(0.5, 3.0) for _ in range(d)])])
    span = bounds[1] - bounds[0]
    x_star = bounds[0] + span * _t([r.uniform(0.35, 0.65) for _ in range(d)])
    ineq = []
    for _ in range(r.randint(1, 3)):
        idx = sorted(r.sample(range(d), r.randint(1, d)))
        coef = _t([r.uniform(-1.5, 1.5) for _ in idx])
        rhs = float((coef * x_star[idx]).sum() - 0.15 * (coef.abs() * span[idx]).sum())
        ineq.append((torch.tensor(idx), coef, rhs))
    eq = None
    if d >= 3 and r.random() < 0.5:
        j = sorted(r.sample(range(d), 2))
        c = _t([1.0, r.uniform(0.5, 1.5)])
        eq = [(torch.tensor(j), c, float((c * x_star[j]).sum()))]
    kw = dict(inequality_constraints=ineq, equality_constraints=eq, n_burnin=2000, n_thinning=8)
    x_bt = bt_sampling.get_polytope_samples(3000, bounds, seed=seed, **kw)
    x_rs = rs_sampling.get_polytope_samples(3000, bounds, seed=seed, **kw)
    assert x_rs.shape == x_bt.shape
    feas = bt_pc.evaluate_feasibility(
        x_rs.unsqueeze(-2), inequality_constraints=ineq, equality_constraints=eq
    )
    assert bool(feas.all()) and bool(
        ((x_rs >= bounds[0] - 1e-9) & (x_rs <= bounds[1] + 1e-9)).all()
    )
    mean_err = ((x_rs.mean(0) - x_bt.mean(0)).abs() / span).max().item()
    cov_err = ((torch.cov(x_rs.T) - torch.cov(x_bt.T)).abs() / torch.outer(span, span)).max().item()
    CHECK.check_bound("polytope samples: mean error / range (statistical)", mean_err, 0.05)
    CHECK.check_bound("polytope samples: covariance error / range^2 (statistical)", cov_err, 0.02)
    # Interior points of the dense polytope (bounds as inequalities).
    A, b = bt_sampling.sparse_to_dense_constraints(d, ineq)
    A_ub = torch.cat([-A, torch.eye(d, dtype=DT), -torch.eye(d, dtype=DT)]).numpy()
    b_ub = torch.cat([-b, bounds[1].unsqueeze(-1), -bounds[0].unsqueeze(-1)]).numpy()
    x0_bt = bt_sampling.find_interior_point(A_ub, b_ub)
    x0_rs = rs_sampling.find_interior_point(A_ub, b_ub)
    slack = lambda x: (b_ub.reshape(-1) - A_ub @ x).min()  # noqa: E731
    CHECK.check_bound(
        "find_interior_point: slack shortfall vs BoTorch",
        (slack(x0_bt) - slack(x0_rs)) / span.max().item(),
        1e-9,
    )
    # Feasibility and constraint transformations on random q-batches.
    q = 2
    Xq = bounds[0] + span * torch.rand(
        64, q, d, generator=torch.Generator().manual_seed(seed), dtype=DT
    )
    f = r.randrange(d)
    inter = [(torch.tensor([[0, f], [1, f]]), _t([1.0, -1.0]), float(-0.1 * span[f]))]
    for kwargs in (
        {"inequality_constraints": ineq + inter},
        {"inequality_constraints": ineq, "tolerance": 0.05},
    ):
        assert torch.equal(
            rs_pc.evaluate_feasibility(Xq, **kwargs), bt_pc.evaluate_feasibility(Xq, **kwargs)
        )
    t_bt = bt_initializers.transform_constraints(ineq + inter, q=q, d=d)
    t_rs = rs_initializers.transform_constraints(ineq + inter, q=q, d=d)
    assert all(
        torch.equal(a[0], b_[0]) and torch.equal(a[1], b_[1]) and a[2] == b_[2]
        for a, b_ in zip(t_rs, t_bt, strict=True)
    )
    CHECK.record("evaluate_feasibility / transform_constraints (mismatches)", 0.0, 0.0)


def test_scenario_coverage() -> None:
    """The grid covers every level of every model dimension."""
    singles = [s.specs[0] for s in SCENARIOS if not s.is_list]
    assert {s.n for s in singles} == {6, 10, 16}
    assert {s.d for s in singles} == {1, 2, 3, 4}
    assert {s.m for s in singles} == {1, 2, 3}
    assert {s.fixed_noise for s in singles} == {True, False}
    assert {s.normalize for s in singles} == {True, False}
    assert {s.standardize for s in singles} == {True, False}
    assert {s.kernel for s in singles} == {"rbf", "matern52"}
    assert any(s.batch for s in singles) and any(not s.batch for s in singles)
    assert any(s.is_list for s in SCENARIOS)
    assert log_err(_t([0.0]), _t([0.0])) == 0.0

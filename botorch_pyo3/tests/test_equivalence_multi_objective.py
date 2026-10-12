"""The randomized equivalence suite of the multi-objective acquisition
functions: botorch_rs against live BoTorch over a seeded grid of models and
options.

Scenarios (``SCENARIOS``) are multi-output ``SingleTaskGP`` models with
m = 2, 3, 4 outputs and ``ModelListGP`` models of single- and multi-output
sub-models, with numbers of training points ``n`` (6, 10, 14), input
dimensions ``d`` (1-4), inferred or fixed observation noise, with or without
``Normalize`` / ``Standardize``, RBF or Matern-5/2 kernels and several input
ranges. For every scenario both libraries are built on identical data and
hyperparameters, and seeded generators draw reference points, weakened
Pareto fronts, test points, pending points, ``q`` (1-3), objectives
(identity, ``outcomes`` subsets with negative indices, random weights, a
generic nonlinear objective, feasibility weighting), outcome constraints
with random ``eta``, ``fat``, ``tau_relu`` / ``tau_max``, exact or
approximate (``alpha``) partitionings, ``cache_root``, ``incremental_nehvi``
and ``max_iep``, and the base samples (identical in both libraries, see
``mo_acquisition_helpers.synced_draws``). Compared are

* the multi-output objectives on random samples;
* qEHVI and qLogEHVI (values, gradients, cell bounds);
* qNEHVI and qLogNEHVI (values, gradients, cached baselines, per-sample
  cell bounds, the improvement of cached pending points), followed -- with a
  BoTorch that includes meta-pytorch/botorch#3392 -- by a ``set_X_pending``
  call that caches pending points;
* the analytic ``ExpectedHypervolumeImprovement``;
* qLogNParEGO with random weights;
* ``prune_inferior_points_multi_objective`` (identical index sets);

and, independently of BoTorch,

* qEHVI equals the average over its posterior samples of the exact
  hypervolume improvement (``Hypervolume`` of the front with the sample
  minus that of the front);
* the analytic EHVI agrees with qEHVI over 16384 quasi-random samples;
* qLogEHVI tends to the logarithm of qEHVI as ``tau_relu = tau_max -> 0``.

Tolerances (the worst observed error of every quantity is printed in the
terminal summary, section "equivalence suite (multi-objective)"):

=======================================  =====================================
deterministic values (objectives, cell   relative error <= 1e-12
bounds of a decomposition)
acquisition values / gradients           1e-9 / 1e-8 (log values: absolute
                                         below 1), or 10x BoTorch's own change
                                         under a 1e-14 relative perturbation
                                         of the training inputs if larger
                                         (the joint posteriors of the noisy
                                         acquisition functions over their
                                         baseline are nearly singular); the
                                         recorded error is scaled to 1e-9 /
                                         1e-8
NEHVI state: per-sample hypervolumes     1e-9, or 10x BoTorch's noise floor;
and cell bounds                          decompositions may differ at near
                                         ties within the noise floor if their
                                         hypervolumes agree (counted)
improvement of cached pending points     1e-12 of the hypervolume (it is a
(and non-incremental qLogNEHVI after     mean of differences of hypervolumes;
caching, compared in linear space)       its logarithm amplifies rounding),
                                         or 10x BoTorch's noise floor
NEHVI on numerically singular joint      if the above fails: only finiteness
posteriors (an output block with         and agreement to 0.1 (whether
smallest eigenvalue < 1e-12 x largest)   Cholesky needs jitter then depends
                                         on rounding; counted)
discrete outputs                         identical (cached baselines, pruned
                                         sets, numbers of cells of
                                         deterministic decompositions)
qEHVI vs exact improvement               1e-9 of the front's hypervolume
analytic EHVI vs 16384-sample qEHVI      1e-3 of the largest EHVI of the
                                         test points (tail values have large
                                         relative MC errors)
qLogEHVI vs log qEHVI, tau = 1e-6        1e-3 (relative, in linear space)
=======================================  =====================================

Not covered here: NEHVI's ``prune_baseline`` (botorch_rs draws the pruning
samples from its own generator; covered by the phase-5B Rust fixtures, which
record BoTorch's pruning samples). Cases that BoTorch 0.18.1 computes
differently (it only appends pending points; see the parity tests) use
incremental NEHVI there.

Run with ``pytest -m equivalence``; skipped unless BoTorch is importable,
which ``BOTORCH_RS_REQUIRE_BOTORCH=1`` turns into an error.
"""

from __future__ import annotations

import contextlib
import random
import zlib
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import pytest

from helpers import importorskip_botorch

importorskip_botorch()
torch = pytest.importorskip("torch")

import botorch.acquisition.utils as bt_au  # noqa: E402
from botorch.acquisition.multi_objective import analytic as bt_analytic  # noqa: E402
from botorch.acquisition.multi_objective import logei as bt_logei  # noqa: E402
from botorch.acquisition.multi_objective import monte_carlo as bt_mc  # noqa: E402
from botorch.acquisition.multi_objective import objective as bt_obj  # noqa: E402
from botorch.acquisition.multi_objective import parego as bt_parego  # noqa: E402
from botorch.acquisition.multi_objective import utils as bt_utils  # noqa: E402
from mo_acquisition_helpers import (  # noqa: E402
    DT,
    botorch_has_pending_semantics,
    botorch_keeps_baseline_base_samples,
    bt_sampler,
    generic_mo,
    partitioning_pair,
    rs_sampler,
    synced_draws,
    zdraw,
)

from botorch_rs.acquisition.multi_objective import analytic as rs_analytic  # noqa: E402
from botorch_rs.acquisition.multi_objective import logei as rs_logei  # noqa: E402
from botorch_rs.acquisition.multi_objective import monte_carlo as rs_mc  # noqa: E402
from botorch_rs.acquisition.multi_objective import objective as rs_obj  # noqa: E402
from botorch_rs.acquisition.multi_objective import parego as rs_parego  # noqa: E402
from botorch_rs.acquisition.multi_objective import utils as rs_utils  # noqa: E402
from botorch_rs.sampling import SobolQMCNormalSampler as RsSampler  # noqa: E402
from botorch_rs.utils.multi_objective.box_decompositions import (  # noqa: E402
    FastNondominatedPartitioning as RsFast,
)
from botorch_rs.utils.multi_objective.hypervolume import Hypervolume as RsHypervolume  # noqa: E402
from botorch_rs.utils.multi_objective.pareto import is_non_dominated  # noqa: E402
from parity_helpers import (  # noqa: E402
    Checker,
    ModelSpec,
    dense_model_list,
    list_pair,
    log_err,
    model_pair,
    perturbed_copy,
    points_in,
    rel_err,
    values_and_grads,
)

pytestmark = [pytest.mark.equivalence, pytest.mark.filterwarnings("ignore")]

SEED = 20261012
VALUE_TOL = 1e-9
GRAD_TOL = 1e-8
EXACT_TOL = 1e-12
NOISE_FACTOR = 10.0
SINGULAR = 1e-12
JITTER_TOL = 0.1
PRUNE_SAMPLES = 128
CHECK = Checker("equivalence suite (multi-objective)")


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
    def d(self) -> int:
        return self.specs[0].d


def _scenarios() -> list[Scenario]:
    r = random.Random(SEED)
    out: list[Scenario] = []
    ranges = [(0.0, 1.0), (-2.0, 3.0), (1.0, 4.0)]
    for i in range(10):
        lo, hi = r.choice(ranges)
        spec = ModelSpec(
            n=[6, 10, 14][i % 3],
            d=[1, 2, 3, 4][i % 4],
            m=[2, 3, 4][(i // 2) % 3],
            fixed_noise=r.random() < 0.35,
            normalize=r.random() < 0.5,
            standardize=r.random() < 0.75,
            kernel=r.choice(["rbf", "matern52"]),
            lo=lo,
            hi=hi,
        )
        out.append(Scenario(f"{i:02d}_{spec.label}", (spec,), False, 300 + i))
    lists = [
        (ModelSpec(10, 2), ModelSpec(8, 2, kernel="matern52", normalize=True)),
        (ModelSpec(12, 3, m=2), ModelSpec(9, 3, fixed_noise=True)),
        (ModelSpec(8, 1), ModelSpec(11, 1, standardize=False), ModelSpec(7, 1, m=2)),
    ]
    for j, specs in enumerate(lists):
        label = "list_" + "+".join(s.label for s in specs)
        out.append(Scenario(f"{10 + j:02d}_{label}", specs, True, 400 + j))
    return out


SCENARIOS = _scenarios()


@pytest.fixture(scope="module", params=SCENARIOS, ids=[s.label for s in SCENARIOS])
def pair(request: pytest.FixtureRequest) -> tuple[Scenario, Any, Any, Any]:
    sc = request.param
    if sc.is_list:
        return (sc, *list_pair(sc.specs, sc.seed))
    return (sc, *model_pair(sc.specs[0], sc.seed))


def bt_eval(fn: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluates BoTorch code with dense ModelListGP covariances."""
    with dense_model_list():
        return fn(*args, **kwargs)


def bt_means(bt: Any, X: Any) -> Any:
    with torch.no_grad():
        return bt_eval(bt.posterior, X).mean


# ---------------------------------------------------------------------------
# Random choices
# ---------------------------------------------------------------------------


@dataclass
class Objective:
    label: str
    o_bt: Any
    o_rs: Any
    m: int  # number of objectives

    def values(self, means: Any) -> Any:
        if self.o_bt is None:
            return means
        with torch.no_grad():
            return bt_eval(self.o_bt, means.unsqueeze(0)).squeeze(0)


def random_objective(r: random.Random, bt: Any, rs: Any, X: Any) -> Objective:
    m = rs.num_outputs
    kinds = ["identity", "outcomes", "weighted", "generic"] + (["feasibility"] if m >= 3 else [])
    kind = r.choice(kinds)
    if kind == "identity":
        return Objective(kind, None, None, m)
    if kind == "outcomes":
        k = r.randint(2, m)
        outcomes = r.sample(range(m), k)
        if r.random() < 0.4:
            outcomes = [o - m for o in outcomes]
        kw = dict(outcomes=outcomes, num_outcomes=m)
        return Objective(
            f"outcomes {outcomes}",
            bt_obj.IdentityMCMultiOutputObjective(**kw),
            rs_obj.IdentityMCMultiOutputObjective(**kw),
            k,
        )
    if kind == "weighted":
        w = _t([r.choice([-1.0, 1.0]) * r.uniform(0.3, 1.5) for _ in range(m)])
        return Objective(
            kind,
            bt_obj.WeightedMCMultiOutputObjective(w),
            rs_obj.WeightedMCMultiOutputObjective(w),
            m,
        )
    if kind == "generic":
        return Objective(
            kind,
            bt_obj.GenericMCMultiOutputObjective(generic_mo),
            rs_obj.GenericMCMultiOutputObjective(generic_mo),
            2,
        )
    idx = r.randrange(m)
    return Objective(
        f"feasibility [{idx}]",
        bt_eval(
            bt_obj.FeasibilityWeightedMCMultiOutputObjective,
            model=bt,
            X_baseline=X,
            constraint_idcs=[idx],
        ),
        rs_obj.FeasibilityWeightedMCMultiOutputObjective(
            model=rs, X_baseline=X, constraint_idcs=[idx]
        ),
        m - 1,
    )


def random_constraints(r: random.Random, means: Any, loose: bool = False) -> tuple[str, dict]:
    """Random outcome constraints on the model outputs (feasible iff <= 0)
    with random ``eta``; ``loose`` ones are satisfied by the posterior means."""
    choice = r.randrange(3)
    if choice == 0:
        return "", {}
    last = means[:, -1]
    shift = (
        (0.5 + r.random()) * last.std().item()
        if loose
        else r.uniform(-0.3, 0.6) * last.std().item()
    )
    c0 = (last.max() if loose else last.mean()).item() + shift
    cons = [lambda Y, c=c0: Y[..., -1] - c]
    etas = [r.uniform(1e-3, 0.2)]
    if choice == 2:
        c1 = (means[:, 0] ** 2).max().item() * (1.5 if loose else r.uniform(0.8, 2.0))
        cons.append(lambda Y, c=c1: Y[..., 0] ** 2 - c)
        etas.append(r.uniform(0.01, 0.5))
    return f", {len(cons)} constraint(s)", {"constraints": cons, "eta": _t(etas)}


def ref_and_front(r: random.Random, Y: Any) -> tuple[Any, Any]:
    """A reference point below the outcomes ``Y`` and a weakened front (the
    outcomes pulled towards the reference point), so that new points
    improve on it."""
    lo, hi = Y.min(0).values, Y.max(0).values
    span = (hi - lo).clamp_min(1e-3)
    ref = lo - r.uniform(0.05, 0.3) * span
    return ref, ref + r.uniform(0.4, 0.9) * (Y - ref)


# ---------------------------------------------------------------------------
# Comparisons
# ---------------------------------------------------------------------------


def compare(
    label: str,
    v_rs: Any,
    v_bt: Any,
    g_rs: Any,
    g_bt: Any,
    log: bool,
    twins: Any = None,
) -> None:
    """Values and gradients; if they differ by more than the nominal
    tolerances and ``twins()`` (BoTorch's values and gradients for perturbed
    copies of the model) is given, the tolerance is ``NOISE_FACTOR`` times
    BoTorch's own change."""
    err_v = log_err(v_rs, v_bt) if log else rel_err(v_rs, v_bt, atol=1e-300)
    err_g = rel_err(g_rs, g_bt, atol=1e-14)
    tol_v, tol_g = VALUE_TOL, GRAD_TOL
    if (err_v > tol_v or err_g > tol_g) and twins is not None:
        floor_v = floor_g = 0.0
        for v_tw, g_tw in twins():
            floor_v = max(floor_v, log_err(v_tw, v_bt) if log else rel_err(v_tw, v_bt))
            floor_g = max(floor_g, rel_err(g_tw, g_bt, atol=1e-14))
        tol_v = max(tol_v, NOISE_FACTOR * floor_v)
        tol_g = max(tol_g, NOISE_FACTOR * floor_g)
        CHECK.record("BoTorch noise floor (noisy acquisition functions): values", floor_v, None)
        CHECK.record("BoTorch noise floor (noisy acquisition functions): gradients", floor_g, None)
    CHECK.record("acquisition values: raw error", err_v, None)
    CHECK.record("acquisition gradients: raw error", err_g, None)
    CHECK.record(f"{label} value", err_v * VALUE_TOL / tol_v, VALUE_TOL)
    CHECK.record(f"{label} gradient", err_g * GRAD_TOL / tol_g, GRAD_TOL)
    assert err_v <= tol_v, (label, "value", err_v, tol_v)
    assert err_g <= tol_g, (label, "gradient", err_g, tol_g)


def check_cells(label: str, a_rs: Any, a_bt: Any, tol: float) -> None:
    lo_rs, lo_bt = torch.as_tensor(a_rs.cell_lower_bounds), a_bt.cell_lower_bounds
    up_rs, up_bt = torch.as_tensor(a_rs.cell_upper_bounds), a_bt.cell_upper_bounds
    same = lo_rs.shape == lo_bt.shape
    CHECK.record(f"{label}: different numbers of cells", float(not same), 0.0)
    assert same, (label, lo_rs.shape, lo_bt.shape)
    CHECK.check(f"{label} cell bounds", torch.cat([lo_rs, up_rs]), torch.cat([lo_bt, up_bt]), tol)


# ---------------------------------------------------------------------------
# BoTorch comparisons
# ---------------------------------------------------------------------------


def test_objectives(pair: tuple) -> None:
    sc, bt, rs, X = pair
    r = rng_for("objectives", sc.label)
    m = rs.num_outputs
    Y = torch.randn(4, 2, 3, m, generator=torch.Generator().manual_seed(sc.seed), dtype=DT)
    Xq = points_in(X, (2, 3, sc.d), seed=sc.seed + 1)
    for _ in range(4):
        o = random_objective(r, bt, rs, X)
        if o.o_bt is None:
            continue
        CHECK.check(
            f"objective values [{o.label.split(' ')[0]}]",
            o.o_rs(Y, X=Xq),
            bt_eval(o.o_bt, Y, X=Xq),
            EXACT_TOL,
            atol=1e-300,
        )


EHVI = [
    ("qExpectedHypervolumeImprovement", bt_mc, rs_mc, False),
    ("qLogExpectedHypervolumeImprovement", bt_logei, rs_logei, True),
]


@pytest.mark.parametrize("acqf", EHVI, ids=["qEHVI", "qLogEHVI"])
def test_qehvi(pair: tuple, acqf: tuple) -> None:
    sc, bt, rs, X = pair
    name, bt_mod, rs_mod, log = acqf
    r = rng_for(name, sc.label)
    m, d = rs.num_outputs, sc.d
    means = bt_means(bt, X)
    for k in range(3):
        o = random_objective(r, bt, rs, X)
        ref, front = ref_and_front(r, o.values(means))
        part = r.choice(["fast", "nondominated"])
        alpha = r.choice([0.0, 1e-3, 1e-2]) if part == "nondominated" and o.m >= 3 else 0.0
        p_bt, p_rs = partitioning_pair(part, ref, front, alpha)
        c_label, kw = random_constraints(r, means)
        if r.random() < 0.5:
            kw["fat"] = r.random() < 0.5
        if log and r.random() < 0.4:
            kw["tau_relu"] = r.choice([1e-3, 1e-5])
            kw["tau_max"] = r.choice([1e-2, 1e-3])
        q, n_pending = r.choice([1, 2, 3]), r.choice([0, 0, 1, 2])
        pending = points_in(X, (n_pending, d), seed=sc.seed + 10 * k) if n_pending else None
        Xt = points_in(X, (2, q, d), seed=sc.seed + 10 * k + 1)
        S, seed = r.choice([16, 32]), r.randrange(10**6)
        with synced_draws(seed):
            a_bt = bt_eval(
                getattr(bt_mod, name),
                bt,
                ref,
                p_bt,
                sampler=bt_sampler(S),
                objective=o.o_bt,
                X_pending=pending,
                **kw,
            )
            v_bt, g_bt = bt_eval(values_and_grads, a_bt, Xt)
        a_rs = getattr(rs_mod, name)(
            rs,
            ref,
            p_rs,
            sampler=rs_sampler(S, seed, (q + n_pending) * m),
            objective=o.o_rs,
            X_pending=pending,
            **kw,
        )
        v_rs, g_rs = values_and_grads(a_rs, Xt)
        short = "qLogEHVI" if log else "qEHVI"
        check_cells(f"{short} ({part})", a_rs, a_bt, EXACT_TOL)
        compare(f"{short} [{o.label.split(' ')[0]}{c_label}]", v_rs, v_bt, g_rs, g_bt, log)


NEHVI = [
    ("qNoisyExpectedHypervolumeImprovement", bt_mc, rs_mc, False),
    ("qLogNoisyExpectedHypervolumeImprovement", bt_logei, rs_logei, True),
]


class Twins:
    """BoTorch's acquisition function on two copies of the model with
    perturbed training inputs (``parity_helpers.perturbed_copy``), built
    lazily by ``build(model)``; ``then(step)`` applies a state change (e.g.
    ``set_X_pending``) to them now or when they are built."""

    def __init__(self, bt: Any, build: Any) -> None:
        self.bt, self.build = bt, build
        self.steps: list[Any] = []
        self._acqfs: list[Any] | None = None

    def then(self, step: Any) -> None:
        self.steps.append(step)
        for a in self._acqfs or []:
            step(a)

    def acqfs(self) -> list[Any]:
        if self._acqfs is None:
            self._acqfs = []
            for j in (1, 2):
                a = self.build(perturbed_copy(self.bt, seed=j))
                for step in self.steps:
                    step(a)
                self._acqfs.append(a)
        return self._acqfs


@pytest.mark.parametrize("acqf", NEHVI, ids=["qNEHVI", "qLogNEHVI"])
def test_qnehvi(pair: tuple, acqf: tuple) -> None:
    sc, bt, rs, X = pair
    name, bt_mod, rs_mod, log = acqf
    r = rng_for(name, sc.label)
    m, d, n = rs.num_outputs, sc.d, X.shape[0]
    means = bt_means(bt, X)
    fixed = botorch_keeps_baseline_base_samples()
    short = "qLogNEHVI" if log else "qNEHVI"
    for k in range(2):
        o = random_objective(r, bt, rs, X)
        ref, _ = ref_and_front(r, o.values(means))
        c_label, kw = random_constraints(r, means)
        q = r.choice([1, 2, 3])
        n_pending, max_iep = r.choice([0, 0, 1, 2]), r.choice([0, 0, 1, 3])
        incremental = r.random() < 0.6
        if n_pending > max_iep and not incremental and not fixed:
            incremental = True  # BoTorch main redraws the cached base samples (#3392)
        if n_pending and not incremental and not botorch_has_pending_semantics():
            incremental = True  # BoTorch 0.18.1: baseline hypervolumes over pending points
        kw.update(
            ref_point=ref,
            cache_root=r.random() < 0.7,
            incremental_nehvi=incremental,
            alpha=r.choice([0.0, 1e-3, 1e-2]) if o.m >= 3 else 0.0,
            max_iep=max_iep,
            X_pending=points_in(X, (n_pending, d), seed=sc.seed + 10 * k) if n_pending else None,
        )
        if log and r.random() < 0.4:
            kw["fat"] = r.random() < 0.5
        Xt = points_in(X, (2, q, d), seed=sc.seed + 10 * k + 1)
        S, seed = r.choice([16, 32]), r.randrange(10**6)
        build = _nehvi_builder(getattr(bt_mod, name), X, S, seed, o.o_bt, kw)
        a_bt = build(bt)
        with synced_draws(seed):
            v_bt, g_bt = bt_eval(values_and_grads, a_bt, Xt)
        a_rs = getattr(rs_mod, name)(
            rs,
            X_baseline=X,
            # Up to three pending points are cached below.
            sampler=rs_sampler(S, seed, (n + max(n_pending, 3) + q) * m),
            objective=o.o_rs,
            **kw,
        )
        v_rs, g_rs = values_and_grads(a_rs, Xt)
        cfg = (
            f"{'inc' if incremental else 'joint'}{', cache_root' if kw['cache_root'] else ''}"
            f"{', alpha' if kw['alpha'] else ''}{', pending' if n_pending else ''}"
        )
        twins = Twins(bt, build)
        compare_nehvi(
            f"{short} [{o.label.split(' ')[0]}{c_label}; {cfg}]",
            a_rs,
            a_bt,
            (v_rs, v_bt, g_rs, g_bt),
            Xt,
            seed,
            log,
            linear=log and not incremental and n_pending > max_iep,
            twins=twins,
        )
        if not (fixed and botorch_has_pending_semantics()):
            continue
        # Pending points that are cached (#3392).
        P = points_in(X, (r.choice([1, 2, 3]), d), seed=sc.seed + 10 * k + 2)
        step = _pending_step(P, seed)
        step(a_bt)
        twins.then(step)
        with synced_draws(seed):
            v_bt, g_bt = bt_eval(values_and_grads, a_bt, Xt)
        a_rs.set_X_pending(P)
        v_rs, g_rs = values_and_grads(a_rs, Xt)
        compare_nehvi(
            f"{short} after set_X_pending [{'inc' if incremental else 'joint'}]",
            a_rs,
            a_bt,
            (v_rs, v_bt, g_rs, g_bt),
            Xt,
            seed,
            log,
            linear=log and not incremental and P.shape[0] > max_iep,
            twins=twins,
        )


def _nehvi_builder(cls: Any, X: Any, S: int, seed: int, objective: Any, kw: dict) -> Any:
    """``build(model)``: BoTorch's NEHVI of a case for a model."""

    def build(model: Any) -> Any:
        with synced_draws(seed):
            return bt_eval(
                cls, model, X_baseline=X, sampler=bt_sampler(S), objective=objective, **kw
            )

    return build


def _pending_step(P: Any, seed: int) -> Any:
    def step(a: Any) -> None:
        with synced_draws(seed):
            bt_eval(a.set_X_pending, P)

    return step


def _cells(a: Any) -> Any:
    return torch.cat([torch.as_tensor(a.cell_lower_bounds), torch.as_tensor(a.cell_upper_bounds)])


def _cells_err(a: Any, e: Any) -> float:
    """Relative error of per-sample cell bounds; infinite for different
    decompositions (numbers of cells or unbounded sides)."""
    ca, ce = _cells(a), _cells(e)
    if ca.shape != ce.shape or not torch.equal(torch.isinf(ca), torch.isinf(ce)):
        return float("inf")
    return rel_err(ca, ce)


def _linear(v: Any, g: Any) -> tuple[Any, Any]:
    """``exp(value)`` and its gradient."""
    v, g = torch.as_tensor(v).exp(), torch.as_tensor(g)
    return v, v.reshape((-1,) + (1,) * (g.dim() - 1)) * g


def singular_posterior(bt: Any, point_sets: list[Any]) -> bool:
    """Whether BoTorch's joint posterior over one of the point sets has a
    numerically singular output block (smallest eigenvalue below
    ``SINGULAR`` times the largest): whether its Cholesky factorization
    needs jitter then depends on rounding, and BoTorch (LAPACK) and
    botorch_rs (nalgebra) may decide differently, which changes the samples
    by O(sqrt(jitter))."""
    for points in point_sets:
        with torch.no_grad():
            mvn = bt_eval(bt.posterior, points).distribution
        cov, n = mvn.covariance_matrix, points.shape[-2]
        k = cov.shape[-1] // n
        for t in range(k):
            idx = (
                slice(t, None, k)
                if getattr(mvn, "_interleaved", False)
                else slice(t * n, (t + 1) * n)
            )
            ev = torch.linalg.eigvalsh(cov[idx, idx])
            if ev.min().item() < SINGULAR * ev.max().item():
                return True
    return False


def compare_nehvi(
    label: str,
    a_rs: Any,
    a_bt: Any,
    results: tuple,
    Xt: Any,
    seed: int,
    log: bool,
    linear: bool,
    twins: Twins,
) -> None:
    """Compares NEHVI: values and gradients (in linear space relative to the
    hypervolume for non-incremental qLogNEHVI with cached pending points:
    there the value is ``log(NEHVI + prev_nehvi)`` with ``prev_nehvi`` exact
    only to the rounding of hypervolume differences, which the logarithm
    amplifies), the cached baseline (identical), the per-sample hypervolumes
    and cell bounds of the cached points, and the improvement of cached
    pending points (relative to the hypervolume).

    Errors above the nominal tolerances are compared with ``NOISE_FACTOR``
    times BoTorch's change on the twins (the decompositions of sampled fronts
    may then differ at near ties if their hypervolumes agree). If they still
    exceed that and a joint posterior over the cached points (and a
    candidate set) is numerically singular (see :func:`singular_posterior`),
    the results depend on jitter decisions: they are recorded separately and
    only checked for finiteness and agreement to ``JITTER_TOL``."""
    v_rs, v_bt, g_rs, g_bt = results
    hv_scale = torch.as_tensor(a_bt._hypervolumes).abs().max().item()
    same = torch.equal(torch.as_tensor(a_rs.X_baseline), a_bt.X_baseline)
    CHECK.record("NEHVI: different cached baselines", float(not same), 0.0)
    assert same, label

    def value_errors(v: Any, g: Any) -> tuple[float, float]:
        if linear:
            (lv, lg), (lb, gb) = _linear(v, g), _linear(v_bt, g_bt)
            return ((lv - lb).abs().max() / hv_scale).item(), rel_err(lg, gb, atol=1e-14)
        err_v = log_err(v, v_bt) if log else rel_err(v, v_bt, atol=1e-300)
        return err_v, rel_err(g, g_bt, atol=1e-14)

    def state_errors(a: Any) -> tuple[float, float, float]:
        prev = abs(float(a._prev_nehvi) - float(a_bt._prev_nehvi)) / hv_scale
        return rel_err(a._hypervolumes, a_bt._hypervolumes), _cells_err(a, a_bt), prev

    names = [
        f"{label} value" + (" (linear, relative to the hypervolume)" if linear else ""),
        f"{label} gradient" + (" of exp(value)" if linear else ""),
        "NEHVI per-sample hypervolumes",
        "NEHVI per-sample cell bounds",
        "NEHVI improvement of cached pending points",
    ]
    nominal = [EXACT_TOL if linear else VALUE_TOL, GRAD_TOL, VALUE_TOL, VALUE_TOL, EXACT_TOL]
    errs = [*value_errors(v_rs, g_rs), *state_errors(a_rs)]
    tols = list(nominal)
    if any(e > t for e, t in zip(errs, tols, strict=True)):
        floors = [0.0] * 5
        for a in twins.acqfs():
            with synced_draws(seed):
                v_tw, g_tw = bt_eval(values_and_grads, a, Xt)
            errs_tw = [*value_errors(v_tw, g_tw), *state_errors(a)]
            floors = [max(f, e) for f, e in zip(floors, errs_tw, strict=True)]
        tols = [max(t, NOISE_FACTOR * f) for t, f in zip(tols, floors, strict=True)]
        CHECK.record("BoTorch noise floor (NEHVI): values", floors[0], None)
        CHECK.record("BoTorch noise floor (NEHVI): gradients", floors[1], None)
        CHECK.record("BoTorch noise floor (NEHVI): per-sample hypervolumes", floors[2], None)
        if errs[3] == float("inf") and tols[2] > VALUE_TOL and errs[2] <= tols[2]:
            # Different decompositions of a nearly singular sampled front
            # (near ties within BoTorch's noise floor) whose hypervolumes
            # agree.
            CHECK.record("NEHVI: decompositions differing at near ties (count)", _count(), None)
            tols[3] = float("inf")
    CHECK.record("acquisition values: raw error", errs[0], None)
    CHECK.record("acquisition gradients: raw error", errs[1], None)
    if all(e <= t for e, t in zip(errs, tols, strict=True)):
        for name, e, t, nom in zip(names, errs, tols, nominal, strict=True):
            if t < float("inf"):
                CHECK.record(name, e * nom / t, nom)
        return
    # The cached points, and with every candidate set and the uncached
    # pending points.
    cached = a_bt.X_baseline
    pending = [] if a_bt.X_pending is None else [a_bt.X_pending]
    sets = [cached] + [torch.cat([cached, Xt[b], *pending]) for b in range(Xt.shape[0])]
    assert singular_posterior(twins.bt, sets), (label, list(zip(names, errs, tols, strict=True)))
    # Jitter-dependent results.
    loose = [e for i, e in enumerate(errs) if i != 3]  # decompositions may differ
    CHECK.record(
        "NEHVI on numerically singular posteriors (jitter-dependent)", max(loose), JITTER_TOL
    )
    CHECK.record(
        "NEHVI evaluations on numerically singular posteriors (count)", _count_singular(), None
    )
    assert (
        torch.isfinite(torch.as_tensor(v_rs)).all() and torch.isfinite(torch.as_tensor(g_rs)).all()
    )
    assert max(errs[0], errs[2]) <= JITTER_TOL, (label, "singular posterior", errs)


_SINGULAR = [0]


def _count_singular() -> float:
    _SINGULAR[0] += 1
    return float(_SINGULAR[0])


_COUNT = [0]


def _count() -> float:
    _COUNT[0] += 1
    return float(_COUNT[0])


def test_analytic_ehvi(pair: tuple) -> None:
    sc, bt, rs, X = pair
    r = rng_for("analytic", sc.label)
    for k in range(2):
        ref, front = ref_and_front(r, bt_means(bt, X))
        part = r.choice(["fast", "nondominated"])
        p_bt, p_rs = partitioning_pair(part, ref, front)
        a_bt = bt_analytic.ExpectedHypervolumeImprovement(bt, ref.tolist(), p_bt)
        a_rs = rs_analytic.ExpectedHypervolumeImprovement(rs, ref.tolist(), p_rs)
        Xt = points_in(X, (6, 1, sc.d), seed=sc.seed + k)
        v_bt, g_bt = bt_eval(values_and_grads, a_bt, Xt)
        v_rs, g_rs = values_and_grads(a_rs, Xt)
        compare("ExpectedHypervolumeImprovement", v_rs, v_bt, g_rs, g_bt, False)


@contextlib.contextmanager
def _no_lower_bound() -> Iterator[None]:
    """qLogNEI estimates a lower bound of the objective from random points if
    the baseline is infeasible in a sample; the constraints of
    :func:`test_qlognparego` are loose enough to avoid that."""
    orig = bt_au._estimate_objective_lower_bound

    def fail(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("the baseline is infeasible in a sample")

    bt_au._estimate_objective_lower_bound = fail
    try:
        yield
    finally:
        bt_au._estimate_objective_lower_bound = orig


def test_qlognparego(pair: tuple) -> None:
    sc, bt, rs, X = pair
    r = rng_for("qLogNParEGO", sc.label)
    m, d, n = rs.num_outputs, sc.d, X.shape[0]
    means = bt_means(bt, X)
    for k in range(2):
        o = random_objective(r, bt, rs, X)
        w = _t([r.uniform(0.05, 1.0) for _ in range(o.m)])
        w = w / w.sum()
        c_label, kw = random_constraints(r, means, loose=True)
        q, n_pending = r.choice([1, 2, 3]), r.choice([0, 0, 1])
        kw.update(scalarization_weights=w, incremental=r.random() < 0.7)
        if r.random() < 0.3:
            kw["fat"] = False
        pending = points_in(X, (n_pending, d), seed=sc.seed + 10 * k) if n_pending else None
        Xt = points_in(X, (2, q, d), seed=sc.seed + 10 * k + 1)
        S, seed = r.choice([16, 32]), r.randrange(10**6)

        def evaluate(model: Any, kw: dict = kw, pending: Any = pending, o: Objective = o) -> Any:
            with synced_draws(seed), _no_lower_bound():  # noqa: B023
                a = bt_eval(
                    bt_parego.qLogNParEGO,
                    model,
                    X_baseline=X,
                    sampler=bt_sampler(S),  # noqa: B023
                    objective=o.o_bt,
                    X_pending=pending,
                    cache_root=False,
                    **kw,
                )
                return a, *bt_eval(values_and_grads, a, Xt)  # noqa: B023

        a_bt, v_bt, g_bt = evaluate(bt)
        a_rs = rs_parego.qLogNParEGO(
            rs,
            X_baseline=X,
            sampler=rs_sampler(S, seed, (n + n_pending + q) * m),
            objective=o.o_rs,
            X_pending=pending,
            **kw,
        )
        CHECK.check("qLogNParEGO Y_baseline", a_rs.Y_baseline, a_bt.Y_baseline, EXACT_TOL)
        v_rs, g_rs = values_and_grads(a_rs, Xt)

        def twins(evaluate: Any = evaluate) -> Iterator[Any]:
            for j in (1, 2):
                yield evaluate(perturbed_copy(bt, seed=j))[1:]

        label = f"qLogNParEGO [{o.label.split(' ')[0]}{c_label}]"
        compare(label, v_rs, v_bt, g_rs, g_bt, True, twins)


def test_prune_inferior_points_multi_objective(pair: tuple) -> None:
    sc, bt, rs, X = pair
    r = rng_for("prune", sc.label)
    m, n = rs.num_outputs, X.shape[0]
    means = bt_means(bt, X)
    for _ in range(3):
        o = random_objective(r, bt, rs, X)
        ref, _ = ref_and_front(r, o.values(means))
        _, ckw = random_constraints(r, means)
        kw: dict[str, Any] = {"max_frac": r.choice([1.0, 1.0, 0.5, 0.3])}
        if "constraints" in ckw:
            kw["constraints"] = ckw["constraints"]
        S, seed = PRUNE_SAMPLES, r.randrange(10**6)
        orig = bt_au.get_sampler
        bt_au.get_sampler = lambda posterior, sample_shape, **_: bt_sampler(PRUNE_SAMPLES)
        try:
            with synced_draws(seed):
                pruned_bt = bt_eval(
                    bt_utils.prune_inferior_points_multi_objective,
                    bt,
                    X,
                    ref,
                    objective=o.o_bt,
                    num_samples=S,
                    **kw,
                )
        finally:
            bt_au.get_sampler = orig
        pruned_rs = rs_utils.prune_inferior_points_multi_objective(
            rs, X, ref, objective=o.o_rs, sampler=rs_sampler(S, seed, n * m), **kw
        )
        same = torch.equal(torch.as_tensor(pruned_rs), pruned_bt)
        CHECK.record("prune_inferior_points_multi_objective: different sets", float(not same), 0.0)
        assert same, (sc.label, o.label, kw)


# ---------------------------------------------------------------------------
# Properties (independent of BoTorch)
# ---------------------------------------------------------------------------


def _front_and_hv(r: random.Random, rs: Any, X: Any) -> tuple[Any, Any, Any, float]:
    ref, front = ref_and_front(r, torch.as_tensor(rs.posterior(X).mean))
    front = front[torch.as_tensor(is_non_dominated(front))]
    return ref, front, RsFast(ref_point=ref, Y=front), RsHypervolume(ref).compute(front)


def test_qehvi_is_the_mean_exact_improvement(pair: tuple) -> None:
    sc, _, rs, X = pair
    r = rng_for("exact improvement", sc.label)
    m, d = rs.num_outputs, sc.d
    ref, front, part, hv0 = _front_and_hv(r, rs, X)
    hv = RsHypervolume(ref)
    for k in range(2):
        q, S, seed = r.choice([1, 2, 3]), 16, r.randrange(10**6)
        Xq = points_in(X, (q, d), seed=sc.seed + k)
        a = rs_mc.qExpectedHypervolumeImprovement(rs, ref, part, sampler=rs_sampler(S, seed, q * m))
        value = float(a(Xq.unsqueeze(0)))
        Z = zdraw(q * m, S, seed).reshape(S, q, m)
        samples = torch.as_tensor(rs.posterior(Xq).rsample_from_base_samples(torch.Size([S]), Z))
        improvements = []
        for s in range(S):
            Y = torch.cat([front, samples[s]])
            Y = Y[torch.as_tensor(is_non_dominated(Y))]
            improvements.append(hv.compute(Y) - hv0)
        expected = sum(improvements) / S
        err = abs(value - expected) / hv0
        CHECK.record(
            "qEHVI vs mean exact improvement (relative to the hypervolume)", err, VALUE_TOL
        )
        assert err <= VALUE_TOL, (sc.label, value, expected)


def test_analytic_ehvi_matches_qehvi(pair: tuple) -> None:
    sc, _, rs, X = pair
    r = rng_for("analytic vs MC", sc.label)
    ref, _, part, _ = _front_and_hv(r, rs, X)
    analytic = rs_analytic.ExpectedHypervolumeImprovement(rs, ref.tolist(), part)
    mc = rs_mc.qExpectedHypervolumeImprovement(
        rs,
        ref,
        part,
        sampler=RsSampler(sample_shape=torch.Size([16384]), seed=r.randrange(10**6)),
    )
    Xt = points_in(X, (8, 1, sc.d), seed=sc.seed)
    v_a, v_mc = torch.as_tensor(analytic(Xt)), torch.as_tensor(mc(Xt))
    err = ((v_a - v_mc).abs().max() / v_a.abs().max()).item()
    CHECK.record("analytic EHVI vs qEHVI (16384 QMC samples, relative to the largest)", err, 1e-3)
    assert err <= 1e-3, (sc.label, v_a, v_mc)


def test_qlogehvi_tends_to_log_qehvi(pair: tuple) -> None:
    sc, _, rs, X = pair
    r = rng_for("tau limit", sc.label)
    m, d = rs.num_outputs, sc.d
    ref, _, part, hv0 = _front_and_hv(r, rs, X)
    q, S, seed = r.choice([1, 2, 3]), 16, r.randrange(10**6)
    Xt = points_in(X, (8, q, d), seed=sc.seed)
    exact = torch.as_tensor(
        rs_mc.qExpectedHypervolumeImprovement(rs, ref, part, sampler=rs_sampler(S, seed, q * m))(Xt)
    )
    keep = exact > 1e-3 * hv0
    if not bool(keep.any()):
        pytest.skip("no test point improves on the front")
    errs = []
    for tau in (1e-2, 1e-4, 1e-6):
        log_v = rs_logei.qLogExpectedHypervolumeImprovement(
            rs, ref, part, sampler=rs_sampler(S, seed, q * m), tau_relu=tau, tau_max=tau
        )(Xt)
        errs.append(((torch.as_tensor(log_v).exp() - exact).abs() / exact)[keep].max().item())
    CHECK.record("qLogEHVI vs log qEHVI (tau = 1e-6, linear space)", errs[-1], 1e-3)
    assert errs[-1] <= 1e-3, (sc.label, errs)
    assert errs[-1] <= errs[0], (sc.label, errs)

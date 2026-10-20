"""Parity of the multi-objective acquisition functions with BoTorch, through
the public APIs of both libraries side by side:

* the multi-output objectives (identity with ``outcomes``, weighted,
  generic, feasibility-weighted) on posterior samples;
* qEHVI and qLogEHVI (``fat`` true / false, constraints, pending points,
  objectives, exact and approximate partitionings of botorch_rs or BoTorch),
  qNEHVI and qLogNEHVI (``cache_root`` true / false, incremental or not,
  ``alpha``, constraints, objectives, pending points at construction,
  uncached (``max_iep``) and cached, appended, replaced and cleared), the
  analytic EHVI and qLogNParEGO (explicit weights): values and gradients
  for m = 2, 3, 4, q = 1, 2, 3 and t-batches, on multi-output
  ``SingleTaskGP`` models and ``ModelListGP``s with identical base samples;
* ``prune_inferior_points_multi_objective`` (identical index sets),
  ``get_default_partitioning_alpha`` and ``compute_subset_indices``;
* ``optimize_acqf`` (L-BFGS-B, sequential, SLSQP with a linear constraint)
  and ``optimize_acqf_list`` from identical initial conditions;
* error types and messages.

Version differences. Cached pending points use the base samples of
meta-pytorch/botorch#3392, which botorch_rs implements: those cases are
compared with BoTorch only if it contains the fix (else botorch_rs is checked
against the documented deviation: BoTorch main redraws the base samples of
the baseline). The released 0.18.1 appends pending points instead of
replacing them and computes the baseline hypervolumes of non-incremental
NEHVI over cached pending points; such cases are skipped there. A
feasibility-weighted objective with two constraints is compared with BoTorch
only if it binds the constraint indices (#3392).

Tolerances: values 1e-9 (relative; log values absolute below 1), gradients
1e-8 (relative to the largest component); BoTorch is evaluated with dense
``ModelListGP`` covariances (see ``parity_helpers.dense_model_list``).
Skipped unless BoTorch is importable (required in CI's parity job).
"""

from __future__ import annotations

import warnings
from collections.abc import Iterator
from typing import Any

import pytest

from helpers import importorskip_botorch

importorskip_botorch()
torch = pytest.importorskip("torch")

from botorch.acquisition import objective as bt_objective  # noqa: E402
from botorch.acquisition.multi_objective import analytic as bt_analytic  # noqa: E402
from botorch.acquisition.multi_objective import logei as bt_logei  # noqa: E402
from botorch.acquisition.multi_objective import monte_carlo as bt_mc  # noqa: E402
from botorch.acquisition.multi_objective import objective as bt_obj  # noqa: E402
from botorch.acquisition.multi_objective import parego as bt_parego  # noqa: E402
from botorch.acquisition.multi_objective import utils as bt_utils  # noqa: E402
from botorch.optim import optimize as bt_optimize  # noqa: E402
from botorch.utils.multi_objective import hypervolume as bt_hv  # noqa: E402
from mo_acquisition_helpers import (  # noqa: E402
    DT,
    botorch_binds_constraint_indices,
    botorch_has_pending_semantics,
    botorch_keeps_baseline_base_samples,
    bt_sampler,
    objective_pair,
    outcome_constraints,
    partitioning_pair,
    rs_sampler,
    synced_draws,
    weak_front,
)

from botorch_rs.acquisition import objective as rs_objective  # noqa: E402
from botorch_rs.acquisition.multi_objective import analytic as rs_analytic  # noqa: E402
from botorch_rs.acquisition.multi_objective import logei as rs_logei  # noqa: E402
from botorch_rs.acquisition.multi_objective import monte_carlo as rs_mc  # noqa: E402
from botorch_rs.acquisition.multi_objective import objective as rs_obj  # noqa: E402
from botorch_rs.acquisition.multi_objective import parego as rs_parego  # noqa: E402
from botorch_rs.acquisition.multi_objective import utils as rs_utils  # noqa: E402
from botorch_rs.optim import optimize as rs_optimize  # noqa: E402
from botorch_rs.utils.multi_objective import hypervolume as rs_hv  # noqa: E402
from parity_helpers import (  # noqa: E402
    Checker,
    ModelSpec,
    dense_model_list,
    list_pair,
    log_err,
    model_pair,
    perturbed_copy,
    rel_err,
    values_and_grads,
)

CHECK = Checker("multi-objective acquisition")
VALUE_TOL = 1e-9
GRAD_TOL = 1e-8
NUM_SAMPLES = 16

pytestmark = pytest.mark.filterwarnings("ignore")


@pytest.fixture(autouse=True, scope="module")
def _single_threaded_torch() -> Iterator[None]:
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(threads)


MODELS = {
    "m2": lambda: model_pair(ModelSpec(n=9, d=2, m=2), 11),
    "m3_norm_matern": lambda: model_pair(
        ModelSpec(n=10, d=3, m=3, normalize=True, kernel="matern52", lo=-1.0, hi=2.0), 12
    ),
    "m4_fixed": lambda: model_pair(ModelSpec(n=8, d=2, m=4, fixed_noise=True), 13),
    "list_2+1": lambda: list_pair(
        (ModelSpec(n=9, d=2, m=2), ModelSpec(n=7, d=2, normalize=True)), 14
    ),
}


@pytest.fixture(scope="module", params=list(MODELS), ids=list(MODELS))
def pair(request: pytest.FixtureRequest) -> tuple[str, Any, Any, Any]:
    bt, rs, X = MODELS[request.param]()
    return request.param, bt, rs, X


def compare(
    label: str,
    v_rs: Any,
    v_bt: Any,
    g_rs: Any,
    g_bt: Any,
    log: bool,
    twins: Any = None,
) -> None:
    """Compares values and gradients. If they differ by more than the nominal
    tolerances and ``twins()`` is given, the tolerance is 10x BoTorch's own
    change when its training inputs are perturbed by 1e-14 (``twins()``
    returns BoTorch's values and gradients for perturbed copies of the
    model): the joint posteriors of noisy acquisition functions over their
    baseline (the training points) are nearly singular. Errors are recorded
    relative to the nominal tolerance."""
    err_fn = log_err if log else rel_err
    err_v = err_fn(v_rs, v_bt) if log else rel_err(v_rs, v_bt, atol=1e-300)
    err_g = rel_err(g_rs, g_bt, atol=1e-14)
    tol_v, tol_g = VALUE_TOL, GRAD_TOL
    if (err_v > tol_v or err_g > tol_g) and twins is not None:
        floor_v = floor_g = 0.0
        for v_tw, g_tw in twins():
            floor_v = max(floor_v, err_fn(v_tw, v_bt))
            floor_g = max(floor_g, rel_err(g_tw, g_bt, atol=1e-14))
        tol_v, tol_g = max(tol_v, 10 * floor_v), max(tol_g, 10 * floor_g)
        CHECK.record("BoTorch noise floor (NEHVI on the training points): values", floor_v, None)
        CHECK.record("BoTorch noise floor (NEHVI on the training points): gradients", floor_g, None)
    CHECK.record(f"{label} value", err_v * VALUE_TOL / tol_v, VALUE_TOL)
    CHECK.record(f"{label} gradient", err_g * GRAD_TOL / tol_g, GRAD_TOL)
    assert err_v <= tol_v, (label, "value", err_v, tol_v)
    assert err_g <= tol_g, (label, "gradient", err_g, tol_g)


def compare_with_prev(
    label: str, v_rs: Any, v_bt: Any, g_rs: Any, g_bt: Any, hv_scale: float
) -> None:
    """Non-incremental qLogNEHVI is ``log(NEHVI + prev_nehvi)``, where the
    improvement ``prev_nehvi`` of cached pending points is a mean of
    differences of hypervolumes, exact only to their rounding (about 1e-15 of
    the hypervolume; pending points that add nothing give 0 or 1e-15 in
    either library). Its logarithm amplifies that: the values are compared in
    linear space with an absolute tolerance of 1e-12 of the hypervolume, the
    gradients as those of ``exp(value)`` (which do not depend on
    ``prev_nehvi``)."""
    lin_rs, lin_bt = torch.as_tensor(v_rs).exp(), torch.as_tensor(v_bt).exp()
    err_v = ((lin_rs - lin_bt).abs().max() / hv_scale).item()
    CHECK.record(f"{label} value (linear, relative to the hypervolume)", err_v, 1e-12)
    assert err_v <= 1e-12, (label, "value", err_v)
    glin_rs = lin_rs[:, None, None] * torch.as_tensor(g_rs)
    glin_bt = lin_bt[:, None, None] * g_bt
    CHECK.check(f"{label} gradient of exp(value)", glin_rs, glin_bt, GRAD_TOL, atol=1e-14)


def bt_eval(fn: Any, *args: Any, **kwargs: Any) -> Any:
    with dense_model_list():
        return fn(*args, **kwargs)


def points(X: Any, shape: tuple[int, ...], seed: int) -> Any:
    lo, hi = X.min(0).values, X.max(0).values
    g = torch.Generator().manual_seed(seed)
    return lo + (hi - lo) * torch.rand(*shape, generator=g, dtype=DT)


# ---------------------------------------------------------------------------
# Objectives
# ---------------------------------------------------------------------------


def test_objectives(pair: tuple) -> None:
    label, bt, rs, X = pair
    m = rs.num_outputs
    Y = torch.randn(5, 2, 3, m, generator=torch.Generator().manual_seed(1), dtype=DT)
    Xq = points(X, (2, 3, X.shape[-1]), 2)
    kinds = ["outcomes", "weighted", "generic"] + (["feasibility"] if m >= 3 else [])
    for kind in kinds:
        o_bt, o_rs, _ = bt_eval(objective_pair, kind, m, bt, rs, X)
        CHECK.check(f"objective values [{kind}]", o_rs(Y, X=Xq), o_bt(Y, X=Xq), 1e-14, atol=1e-300)
    if m >= 3:
        f_bt = bt_eval(
            bt_obj.FeasibilityWeightedMCMultiOutputObjective,
            model=bt,
            X_baseline=X,
            constraint_idcs=[-1],
        )
        f_rs = rs_obj.FeasibilityWeightedMCMultiOutputObjective(
            model=rs, X_baseline=X, constraint_idcs=[-1]
        )
        # BoTorch's infeasible cost (of the objective outputs) and ours.
        from botorch.acquisition.utils import get_infeasible_cost

        cost = bt_eval(get_infeasible_cost, X=X, model=bt, objective=lambda y, X: y)
        CHECK.check(
            "feasibility weighting: infeasible cost", f_rs.infeasible_cost, cost[:-1], 1e-12
        )
        CHECK.check("feasibility weighting: values", f_rs(Y), f_bt(Y), 1e-14, atol=1e-300)
        # Two constraints: every index is used (meta-pytorch/botorch#3392).
        two_rs = rs_obj.FeasibilityWeightedMCMultiOutputObjective(
            model=rs, X_baseline=X, constraint_idcs=[1, 2]
        )
        two_bt = bt_eval(
            bt_obj.FeasibilityWeightedMCMultiOutputObjective,
            model=bt,
            X_baseline=X,
            constraint_idcs=[1, 2],
        )
        Yc = Y.clone()
        Yc[..., 1] = -1.0  # violates only constraint 1
        Yc[..., 2] = 1.0
        out_rs, out_bt = two_rs(Yc), two_bt(Yc)
        if botorch_binds_constraint_indices():
            CHECK.check("feasibility weighting, two constraints", out_rs, out_bt, 1e-14)
        else:
            # BoTorch main checks the last index for both constraints.
            assert not torch.allclose(out_rs, out_bt)
            cost = torch.tensor(two_rs._rs.infeasible_cost, dtype=DT)
            assert torch.allclose(out_rs, -cost.expand_as(out_rs), atol=1e-12)


# ---------------------------------------------------------------------------
# qEHVI and qLogEHVI
# ---------------------------------------------------------------------------

EHVI_CASES = [
    # (name, q, pending, constraints, objective, partitioning, extra kwargs)
    ("qExpectedHypervolumeImprovement", 1, 0, None, "identity", "fast", {}),
    ("qExpectedHypervolumeImprovement", 2, 1, "one", "identity", "fast", {"fat": True}),
    ("qExpectedHypervolumeImprovement", 3, 0, "two", "weighted", "nondominated", {}),
    ("qLogExpectedHypervolumeImprovement", 1, 0, None, "identity", "fast", {}),
    ("qLogExpectedHypervolumeImprovement", 2, 0, "one", "outcomes", "fast", {"fat": False}),
    ("qLogExpectedHypervolumeImprovement", 3, 1, "two", "generic", "nondominated", {}),
    ("qLogExpectedHypervolumeImprovement", 2, 2, None, "feasibility", "fast", {"tau_relu": 1e-3}),
]


@pytest.mark.parametrize("case", EHVI_CASES, ids=lambda c: f"{c[0]}-q{c[1]}-{c[4]}")
def test_ehvi_family(pair: tuple, case: tuple) -> None:
    label, bt, rs, X = pair
    name, q, n_pending, cons_kind, obj_kind, part_kind, extra = case
    m = rs.num_outputs
    if obj_kind == "feasibility" and m < 3:
        pytest.skip("the feasibility-weighted objective needs a constraint output")
    o_bt, o_rs, mo = bt_eval(objective_pair, obj_kind, m, bt, rs, X)
    with torch.no_grad():
        means = bt_eval(bt.posterior, X).mean
        Yobj = means if o_bt is None else o_bt(means.unsqueeze(0)).squeeze(0)
    ref = Yobj.min(0).values - 0.1 * (Yobj.max(0).values - Yobj.min(0).values)
    front = ref + 0.6 * (Yobj - ref)
    alpha = 1e-2 if part_kind == "nondominated" and mo >= 3 else 0.0
    p_bt, p_rs = partitioning_pair(part_kind, ref, front, alpha)
    kw: dict[str, Any] = dict(extra)
    if cons_kind is not None:
        kw["constraints"], kw["eta"] = outcome_constraints(means, cons_kind)
    pending = points(X, (n_pending, X.shape[-1]), 3) if n_pending else None
    Xt = points(X, (2, q, X.shape[-1]), 4 + q)
    log = name.startswith("qLog")
    for partitioning in (p_bt, p_rs):
        with synced_draws(5):
            a_bt = bt_eval(
                getattr(bt_logei if log else bt_mc, name),
                bt,
                ref,
                p_bt,
                sampler=bt_sampler(NUM_SAMPLES),
                objective=o_bt,
                X_pending=pending,
                **kw,
            )
            v_bt, g_bt = bt_eval(values_and_grads, a_bt, Xt)
        a_rs = getattr(rs_logei if log else rs_mc, name)(
            rs,
            ref,
            partitioning,
            sampler=rs_sampler(NUM_SAMPLES, 5, (q + n_pending) * m),
            objective=o_rs,
            X_pending=pending,
            **kw,
        )
        v_rs, g_rs = values_and_grads(a_rs, Xt)
        compare(f"{name} [{label}]", v_rs, v_bt, g_rs, g_bt, log)


# ---------------------------------------------------------------------------
# qNEHVI and qLogNEHVI
# ---------------------------------------------------------------------------

NEHVI_CASES = [
    # (log, q, cache_root, incremental, alpha, constraints, objective, pending, max_iep)
    (False, 1, True, True, 0.0, None, "identity", 0, 0),
    (False, 2, False, True, 0.0, "one", "identity", 0, 0),
    (False, 2, True, False, 0.0, None, "weighted", 0, 0),
    (True, 1, True, True, 0.0, None, "identity", 0, 0),
    (True, 1, False, True, 0.0, None, "identity", 0, 0),
    (True, 2, True, True, 1e-2, "two", "generic", 0, 0),
    (True, 3, True, False, 0.0, "one", "outcomes", 0, 0),
    (True, 2, False, False, 0.0, None, "feasibility", 0, 0),
    # Pending points: cached at construction (incremental), uncached.
    (True, 1, True, True, 0.0, None, "identity", 2, 0),
    (False, 2, True, True, 0.0, "one", "identity", 2, 0),
    (True, 2, True, False, 0.0, None, "identity", 1, 2),
    (False, 1, False, True, 0.0, None, "weighted", 2, 5),
]


def nehvi_pair(
    bt: Any,
    rs: Any,
    X: Any,
    log: bool,
    seed: int,
    max_dim: int,
    record: dict[int, Any] | None = None,
    **kw: Any,
) -> tuple[Any, Any]:
    name = (
        "qLogNoisyExpectedHypervolumeImprovement" if log else "qNoisyExpectedHypervolumeImprovement"
    )
    with synced_draws(seed, record):
        a_bt = bt_eval(
            getattr(bt_logei if log else bt_mc, name),
            bt,
            X_baseline=X,
            sampler=bt_sampler(NUM_SAMPLES),
            **{k: v for k, v in kw.items() if k != "objective_rs"},
        )
    kw_rs = dict(kw)
    kw_rs["objective"] = kw_rs.pop("objective_rs")
    a_rs = getattr(rs_logei if log else rs_mc, name)(
        rs, X_baseline=X, sampler=rs_sampler(NUM_SAMPLES, seed, max_dim), **kw_rs
    )
    return a_bt, a_rs


@pytest.mark.parametrize(
    "case", NEHVI_CASES, ids=lambda c: f"{'log' if c[0] else 'lin'}-q{c[1]}-cr{c[2]}-inc{c[3]}"
)
def test_nehvi_family(pair: tuple, case: tuple) -> None:
    label, bt, rs, X = pair
    log, q, cache_root, incremental, alpha, cons_kind, obj_kind, n_pending, max_iep = case
    m = rs.num_outputs
    if obj_kind == "feasibility" and m < 3:
        pytest.skip("the feasibility-weighted objective needs a constraint output")
    caches_pending = n_pending > max_iep
    if caches_pending and not incremental and not botorch_keeps_baseline_base_samples():
        pytest.skip("needs meta-pytorch/botorch#3392 (cached pending points)")
    if n_pending and not botorch_has_pending_semantics() and not incremental:
        pytest.skip("BoTorch 0.18.1 computes the baseline hypervolumes over pending points")
    o_bt, o_rs, mo = bt_eval(objective_pair, obj_kind, m, bt, rs, X)
    with torch.no_grad():
        means = bt_eval(bt.posterior, X).mean
        Yobj = means if o_bt is None else o_bt(means.unsqueeze(0)).squeeze(0)
    ref = Yobj.min(0).values - 0.1 * (Yobj.max(0).values - Yobj.min(0).values)
    kw: dict[str, Any] = dict(
        ref_point=ref,
        objective=o_bt,
        objective_rs=o_rs,
        cache_root=cache_root,
        incremental_nehvi=incremental,
        alpha=alpha if mo >= 3 else 0.0,
        max_iep=max_iep,
    )
    if cons_kind is not None:
        kw["constraints"], kw["eta"] = outcome_constraints(means, cons_kind)
    if n_pending:
        kw["X_pending"] = points(X, (n_pending, X.shape[-1]), 6)
    n = X.shape[0]
    a_bt, a_rs = nehvi_pair(bt, rs, X, log, 7, (n + n_pending + q) * m, **kw)
    Xt = points(X, (2, q, X.shape[-1]), 8 + q)
    with synced_draws(7):
        v_bt, g_bt = bt_eval(values_and_grads, a_bt, Xt)
    v_rs, g_rs = values_and_grads(a_rs, Xt)
    name = "qLogNEHVI" if log else "qNEHVI"

    def twins() -> Iterator[tuple[Any, Any]]:
        # Five copies: the noise floor of a single copy varies by an order of
        # magnitude, and two underestimated it on Linux CI.
        for k in range(1, 6):
            twin = perturbed_copy(bt, seed=k)
            a_tw, _ = nehvi_pair(twin, rs, X, log, 7, (n + n_pending + q) * m, **kw)
            with synced_draws(7):
                yield bt_eval(values_and_grads, a_tw, Xt)

    if log and not incremental and caches_pending:
        scale = torch.as_tensor(a_rs._hypervolumes).abs().max().item()
        compare_with_prev(f"{name} [{label}]", v_rs, v_bt, g_rs, g_bt, scale)
    else:
        compare(f"{name} [{label}]", v_rs, v_bt, g_rs, g_bt, log, twins)
    # The baseline state.
    CHECK.check("NEHVI cell bounds", a_rs.cell_lower_bounds, a_bt.cell_lower_bounds, 1e-9)
    CHECK.check("NEHVI cached baseline", a_rs.X_baseline, a_bt.X_baseline, 0.0)
    if not incremental:
        # BoTorch 0.18.1 calls them `_initial_hvs`.
        bt_hvs = getattr(a_bt, "_baseline_hvs", None)
        bt_hvs = a_bt._initial_hvs if bt_hvs is None else bt_hvs
        CHECK.check("NEHVI baseline hypervolumes", a_rs._baseline_hvs, bt_hvs, 1e-9)
    CHECK.check("NEHVI previous NEHVI", a_rs._prev_nehvi, a_bt._prev_nehvi, 1e-9, atol=1e-15)


PENDING_STEPS = [
    ("append", lambda P: P[:1]),
    ("append more", lambda P: P[:3]),
    ("replace", lambda P: P[3:5]),
    ("clear", lambda P: None),
    ("set again", lambda P: P[1:2]),
]


@pytest.mark.parametrize("log", [False, True], ids=["qNEHVI", "qLogNEHVI"])
@pytest.mark.parametrize("incremental", [True, False], ids=["incremental", "joint"])
def test_nehvi_pending_points(pair: tuple, log: bool, incremental: bool) -> None:
    label, bt, rs, X = pair
    if not botorch_has_pending_semantics():
        pytest.skip("BoTorch 0.18.1 only appends pending points")
    m = rs.num_outputs
    ref, _ = weak_front(bt, X)
    P = points(X, (5, X.shape[-1]), 21)
    n = X.shape[0]
    a_bt, a_rs = nehvi_pair(
        bt,
        rs,
        X,
        log,
        23,
        (n + 6) * m,
        ref_point=ref,
        objective=None,
        objective_rs=None,
        incremental_nehvi=incremental,
    )
    Xt = points(X, (2, 1, X.shape[-1]), 25)
    fixed = botorch_keeps_baseline_base_samples()
    name = "qLogNEHVI" if log else "qNEHVI"
    for step, select in PENDING_STEPS:
        Pk = select(P)
        with synced_draws(23):
            bt_eval(a_bt.set_X_pending, Pk)
            v_bt, g_bt = bt_eval(values_and_grads, a_bt, Xt)
        a_rs.set_X_pending(Pk)
        v_rs, g_rs = values_and_grads(a_rs, Xt)
        assert torch.equal(torch.as_tensor(a_rs.X_baseline), a_bt.X_baseline), step
        if (fixed or step in ("clear",)) and log and not incremental and Pk is not None:
            scale = torch.as_tensor(a_rs._hypervolumes).abs().max().item()
            compare_with_prev(f"{name} pending ({step}) [{label}]", v_rs, v_bt, g_rs, g_bt, scale)
        elif fixed or step in ("clear",):
            compare(f"{name} pending ({step}) [{label}]", v_rs, v_bt, g_rs, g_bt, log)
        else:
            # BoTorch main redraws the base samples of the baseline when
            # caching pending points; botorch_rs keeps them (#3392).
            CHECK.record(
                f"{name} pending: deviation from BoTorch main ({step})",
                (log_err if log else rel_err)(v_rs, v_bt),
                None,
            )


# ---------------------------------------------------------------------------
# Analytic EHVI and qLogNParEGO
# ---------------------------------------------------------------------------


def test_analytic_ehvi(pair: tuple) -> None:
    label, bt, rs, X = pair
    ref, front = weak_front(bt, X)
    for kind in ("fast", "nondominated"):
        p_bt, p_rs = partitioning_pair(kind, ref, front)
        a_bt = bt_analytic.ExpectedHypervolumeImprovement(bt, ref.tolist(), p_bt)
        a_rs = rs_analytic.ExpectedHypervolumeImprovement(rs, ref.tolist(), p_rs)
        Xt = points(X, (6, 1, X.shape[-1]), 31)
        v_bt, g_bt = bt_eval(values_and_grads, a_bt, Xt)
        v_rs, g_rs = values_and_grads(a_rs, Xt)
        compare(f"ExpectedHypervolumeImprovement [{label}]", v_rs, v_bt, g_rs, g_bt, False)


@pytest.mark.parametrize("variant", ["plain", "constraints+pending", "weighted", "generic"])
def test_qlognparego(pair: tuple, variant: str) -> None:
    label, bt, rs, X = pair
    m = rs.num_outputs
    with torch.no_grad():
        means = bt_eval(bt.posterior, X).mean
    kw: dict[str, Any] = {}
    obj = "identity"
    if variant == "constraints+pending":
        kw["constraints"], kw["eta"] = outcome_constraints(means, "one")
        kw["X_pending"] = points(X, (1, X.shape[-1]), 41)
    elif variant != "plain":
        obj = variant
    o_bt, o_rs, mo = bt_eval(objective_pair, obj, m, bt, rs, X)
    weights = torch.linspace(0.2, 1.0, mo, dtype=DT)
    weights = weights / weights.sum()
    q = 2
    D = (X.shape[0] + q + (1 if "X_pending" in kw else 0)) * m
    with synced_draws(43):
        a_bt = bt_eval(
            bt_parego.qLogNParEGO,
            bt,
            X_baseline=X,
            scalarization_weights=weights,
            sampler=bt_sampler(NUM_SAMPLES),
            objective=o_bt,
            cache_root=False,
            **kw,
        )
    a_rs = rs_parego.qLogNParEGO(
        rs,
        cache_root=False,
        X_baseline=X,
        scalarization_weights=weights,
        sampler=rs_sampler(NUM_SAMPLES, 43, D),
        objective=o_rs,
        **kw,
    )
    CHECK.check("qLogNParEGO Y_baseline", a_rs.Y_baseline, a_bt.Y_baseline, 1e-12)
    Xt = points(X, (2, q, X.shape[-1]), 45)
    with synced_draws(43):
        v_bt, g_bt = bt_eval(values_and_grads, a_bt, Xt)
    v_rs, g_rs = values_and_grads(a_rs, Xt)
    compare(f"qLogNParEGO [{label}]", v_rs, v_bt, g_rs, g_bt, True)


def test_qlognparego_random_weights(pair: tuple) -> None:
    _, bt, rs, X = pair
    a = rs_parego.qLogNParEGO(rs, X_baseline=X, sampler=rs_sampler(8, 1, 64))
    w = torch.as_tensor(a.scalarization_weights)
    assert w.shape == (rs.num_outputs,) and (w >= 0).all()
    assert abs(w.sum().item() - 1.0) < 1e-12
    # The scalarization equals BoTorch's for the same weights and baseline.
    from botorch.utils.multi_objective.scalarization import get_chebyshev_scalarization

    s_bt = get_chebyshev_scalarization(weights=w, Y=torch.as_tensor(a.Y_baseline))
    Y = torch.randn(4, 3, rs.num_outputs, dtype=DT, generator=torch.Generator().manual_seed(2))
    CHECK.check("qLogNParEGO scalarization", a.chebyshev_scalarization(Y), s_bt(Y), 1e-14)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def test_prune_inferior_points_multi_objective(pair: tuple) -> None:
    label, bt, rs, X = pair
    m = rs.num_outputs
    ref, _ = weak_front(bt, X)
    with torch.no_grad():
        means = bt_eval(bt.posterior, X).mean
    S = 256
    import botorch.acquisition.utils as bt_au

    for kw in (
        {},
        {"max_frac": 0.3},
        {"constraints": outcome_constraints(means, "one")[0]},
        {"objective": "weighted"},
    ):
        kw = dict(kw)
        o_bt = o_rs = None
        if kw.get("objective"):
            o_bt, o_rs, _ = objective_pair(kw.pop("objective"), m)
        orig = bt_au.get_sampler
        bt_au.get_sampler = lambda posterior, sample_shape, **_: bt_sampler(S)  # noqa: B023
        try:
            with synced_draws(51):
                pruned_bt = bt_eval(
                    bt_utils.prune_inferior_points_multi_objective,
                    bt,
                    X,
                    ref,
                    objective=o_bt,
                    num_samples=S,
                    **kw,
                )
        finally:
            bt_au.get_sampler = orig
        pruned_rs = rs_utils.prune_inferior_points_multi_objective(
            rs, X, ref, objective=o_rs, sampler=rs_sampler(S, 51, X.shape[0] * m), **kw
        )
        same = torch.equal(torch.as_tensor(pruned_rs), pruned_bt)
        CHECK.record("prune_inferior_points_multi_objective: different sets", float(not same), 0.0)
        assert same, (label, kw)


def test_default_partitioning_alpha() -> None:
    for k in range(1, 9):
        with warnings.catch_warnings(record=True) as w_bt:
            warnings.simplefilter("always")
            a_bt = bt_utils.get_default_partitioning_alpha(k)
        with warnings.catch_warnings(record=True) as w_rs:
            warnings.simplefilter("always")
            a_rs = rs_utils.get_default_partitioning_alpha(k)
        assert a_rs == a_bt and type(a_rs) is type(a_bt)
        assert [str(w.message) for w in w_rs] == [str(w.message) for w in w_bt]


def test_subset_indices() -> None:
    for q in range(1, 6):
        s_bt = bt_hv.compute_subset_indices(q)
        s_rs = rs_hv.compute_subset_indices(q)
        assert list(s_bt.keys()) == list(s_rs.keys())
        for k in s_bt:
            assert torch.equal(torch.as_tensor(s_rs[k]), s_bt[k]), (q, k)


# ---------------------------------------------------------------------------
# Optimization
# ---------------------------------------------------------------------------


def _bounds(X: Any) -> Any:
    return torch.stack([X.min(0).values, X.max(0).values])


def test_optimize_acqf(pair: tuple) -> None:
    label, bt, rs, X = pair
    m = rs.num_outputs
    if m > 3:
        pytest.skip("covered by the smaller models")
    ref, _ = weak_front(bt, X)
    bounds = _bounds(X)
    d = X.shape[-1]
    n = X.shape[0]
    ics = points(X, (4, 1, d), 61)
    kw = dict(ref_point=ref, objective=None, objective_rs=None)
    a_bt, a_rs = nehvi_pair(bt, rs, X, True, 63, (n + 3) * m, **kw)
    with synced_draws(63):
        x_bt, v_bt = bt_eval(
            bt_optimize.optimize_acqf,
            a_bt,
            bounds,
            q=1,
            num_restarts=4,
            batch_initial_conditions=ics,
            return_best_only=False,
        )
    x_rs, v_rs = rs_optimize.optimize_acqf(
        a_rs, bounds, q=1, num_restarts=4, batch_initial_conditions=ics, return_best_only=False
    )
    agree = [
        abs(float(v_rs[i]) - float(v_bt[i])) <= 1e-6 * max(1.0, abs(float(v_bt[i])))
        for i in range(4)
    ]
    CHECK.record(
        "optimize_acqf (qLogNEHVI): restarts not reaching BoTorch's value", 4 - sum(agree), None
    )
    assert sum(agree) >= 3, (label, v_rs, v_bt)
    # Sequential q = 2 from identical initial conditions of every step.
    calls = {"n": 0}

    def ic_generator(**_: Any) -> Any:
        calls["n"] += 1
        return points(X, (3, 1, d), 70 + calls["n"])

    results = []
    for lib_opt, acqf in ((bt_optimize, a_bt), (rs_optimize, a_rs)):
        calls["n"] = 0
        with synced_draws(63):
            results.append(
                bt_eval(
                    lib_opt.optimize_acqf,
                    acqf,
                    bounds,
                    q=2,
                    num_restarts=3,
                    raw_samples=8,
                    sequential=True,
                    ic_generator=ic_generator,
                )
            )
    (xb, vb), (xr, vr) = results
    vr = torch.as_tensor(vr)
    if botorch_keeps_baseline_base_samples():
        err = rel_err(vr, vb)
        CHECK.record("optimize_acqf sequential (qLogNEHVI): values", err, None)
        assert err < 1e-5, (label, vr, vb)
    else:
        # The second step caches the first candidate as a pending point:
        # BoTorch main redraws the baseline's base samples there (#3392).
        err = rel_err(vr[:1], vb[:1])
        CHECK.record("optimize_acqf sequential (qLogNEHVI): first step", err, None)
        assert err < 1e-5, (label, vr, vb)


def test_optimize_acqf_list(pair: tuple) -> None:
    label, bt, rs, X = pair
    m = rs.num_outputs
    if m > 3:
        pytest.skip("covered by the smaller models")
    d = X.shape[-1]
    bounds = _bounds(X)
    acqfs_bt, acqfs_rs = [], []
    for k in range(2):
        w = torch.tensor([0.3 + 0.4 * k] + [1.0] * (m - 1), dtype=DT)
        w = w / w.sum()
        D = (X.shape[0] + 2) * m
        with synced_draws(80):
            acqfs_bt.append(
                bt_eval(
                    bt_parego.qLogNParEGO,
                    bt,
                    X_baseline=X,
                    scalarization_weights=w,
                    sampler=bt_sampler(NUM_SAMPLES),
                    cache_root=False,
                )
            )
        acqfs_rs.append(
            rs_parego.qLogNParEGO(
                rs,
                cache_root=False,
                X_baseline=X,
                scalarization_weights=w,
                sampler=rs_sampler(NUM_SAMPLES, 80, D),
            )
        )
    calls = {"n": 0}

    def ic_generator(**_: Any) -> Any:
        calls["n"] += 1
        return points(X, (3, 1, d), 90 + calls["n"])

    calls["n"] = 0
    with synced_draws(80):
        x_bt, v_bt = bt_eval(
            bt_optimize.optimize_acqf_list,
            acqfs_bt,
            bounds,
            num_restarts=3,
            raw_samples=8,
            ic_generator=ic_generator,
        )
    calls["n"] = 0
    x_rs, v_rs = rs_optimize.optimize_acqf_list(
        acqfs_rs, bounds, num_restarts=3, raw_samples=8, ic_generator=ic_generator
    )
    x_rs = torch.as_tensor(x_rs)
    assert torch.equal(torch.as_tensor(acqfs_rs[1].X_pending), x_rs[:1])
    if not botorch_has_pending_semantics():
        # BoTorch 0.18.1 handles the pending point of the second step
        # differently (meta-pytorch/botorch#3379 aligned it) and, with SciPy
        # >= 1.16, optimizes all L-BFGS-B restarts as one joint problem: only
        # gross failures of the first step are checked.
        shortfall = float(v_bt[0]) - float(torch.as_tensor(v_rs)[0])
        CHECK.record("optimize_acqf_list: first step below BoTorch 0.18.1", shortfall, None)
        assert shortfall <= 0.05 * max(1.0, abs(float(v_bt[0]))), (label, v_rs, v_bt)
        return
    assert torch.allclose(x_rs, x_bt, atol=1e-4), (label, x_rs, x_bt)
    # The value of a candidate that coincides with the pending point depends
    # on how jitter makes the singular joint covariance definite (LAPACK and
    # botorch_rs decide differently); it is only checked loosely.
    v_rs = torch.as_tensor(v_rs)
    singular = torch.cdist(x_bt[1:], x_bt[:1]).min().item() < 1e-6
    err = rel_err(v_rs, v_bt)
    if singular:
        CHECK.record("optimize_acqf_list: values at a pending point (jitter-dependent)", err, 0.1)
        assert err < 0.1, (label, v_rs, v_bt)
    else:
        CHECK.record("optimize_acqf_list (qLogNParEGO): values", err, None)
        assert err < 1e-5, (label, v_rs, v_bt)


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


def _raises(fn: Any) -> BaseException:
    with pytest.raises(Exception) as e:
        fn()
    return e.value


def test_errors() -> None:
    bt, rs, X = MODELS["m2"]()
    ref, front = weak_front(bt, X)
    p_bt, p_rs = partitioning_pair("fast", ref, front)
    cases = [
        (
            lambda: bt_mc.qExpectedHypervolumeImprovement(bt, [0.0], p_bt),
            lambda: rs_mc.qExpectedHypervolumeImprovement(rs, [0.0], p_rs),
        ),
        (
            lambda: bt_logei.qLogExpectedHypervolumeImprovement(bt, [0.0, 1.0, 2.0], p_bt),
            lambda: rs_logei.qLogExpectedHypervolumeImprovement(rs, [0.0, 1.0, 2.0], p_rs),
        ),
        (
            lambda: bt_logei.qLogNoisyExpectedHypervolumeImprovement(bt, [0.0], X),
            lambda: rs_logei.qLogNoisyExpectedHypervolumeImprovement(rs, [0.0], X),
        ),
        (
            lambda: bt_mc.qNoisyExpectedHypervolumeImprovement(bt, ref, X.unsqueeze(0)),
            lambda: rs_mc.qNoisyExpectedHypervolumeImprovement(rs, ref, X.unsqueeze(0)),
        ),
        (
            lambda: bt_analytic.ExpectedHypervolumeImprovement(bt, [1e3, 1e3], p_bt),
            lambda: rs_analytic.ExpectedHypervolumeImprovement(rs, [1e3, 1e3], p_rs),
        ),
        (
            lambda: bt_mc.qExpectedHypervolumeImprovement(
                bt, ref, p_bt, objective=bt_objective.IdentityMCObjective()
            ),
            lambda: rs_mc.qExpectedHypervolumeImprovement(
                rs, ref, p_rs, objective=rs_objective.IdentityMCObjective()
            ),
        ),
        (
            lambda: bt_obj.IdentityMCMultiOutputObjective(outcomes=[0]),
            lambda: rs_obj.IdentityMCMultiOutputObjective(outcomes=[0]),
        ),
        (
            lambda: bt_obj.IdentityMCMultiOutputObjective(outcomes=[0, -1]),
            lambda: rs_obj.IdentityMCMultiOutputObjective(outcomes=[0, -1]),
        ),
        (
            lambda: bt_obj.WeightedMCMultiOutputObjective(torch.ones(3, dtype=DT), outcomes=[0, 1]),
            lambda: rs_obj.WeightedMCMultiOutputObjective(torch.ones(3, dtype=DT), outcomes=[0, 1]),
        ),
        (
            lambda: bt_obj.WeightedMCMultiOutputObjective(torch.ones(2, 1, dtype=DT)),
            lambda: rs_obj.WeightedMCMultiOutputObjective(torch.ones(2, 1, dtype=DT)),
        ),
        (
            lambda: bt_obj.FeasibilityWeightedMCMultiOutputObjective(bt, X, [0, -2]),
            lambda: rs_obj.FeasibilityWeightedMCMultiOutputObjective(rs, X, [0, -2]),
        ),
        (
            lambda: bt_optimize.optimize_acqf_list([], _bounds(X), 2, 4),
            lambda: rs_optimize.optimize_acqf_list([], _bounds(X), 2, 4),
        ),
        (
            lambda: bt_analytic.ExpectedHypervolumeImprovement(
                bt, ref.tolist(), p_bt
            ).set_X_pending(X[:1]),
            lambda: rs_analytic.ExpectedHypervolumeImprovement(
                rs, ref.tolist(), p_rs
            ).set_X_pending(X[:1]),
        ),
    ]
    for i, (f_bt, f_rs) in enumerate(cases):
        e_bt, e_rs = _raises(f_bt), _raises(f_rs)
        base = next(
            b
            for b in (
                NotImplementedError,
                ValueError,
                RuntimeError,
                IndexError,
                TypeError,
                Exception,
            )
            if isinstance(e_bt, b)
        )
        assert isinstance(e_rs, base), (i, e_bt, e_rs)
        assert str(e_rs) == str(e_bt), (i, str(e_rs), str(e_bt))
    CHECK.record("error types and messages (cases)", 0.0, None)

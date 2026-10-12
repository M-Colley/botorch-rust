"""Live parity of the multi-objective test problems with BoTorch
(``botorch.test_functions.multi_objective`` and
``multi_objective_multi_fidelity``) through the public APIs.

Every class is compared on its attributes (dimensions, bounds, reference
points, ``max_hv``, index sets, class-specific attributes), its errors and
signatures, and -- in the randomized, seeded sweep (marker ``equivalence``)
-- on ``evaluate_true``, ``forward`` (with ``negate``), ``evaluate_slack``,
``evaluate_slack_true`` and ``is_feasible`` at random points and corners of
the domain, for batched inputs, float32 and float64 inputs, and with
observation noise (scalar and per-objective ``noise_std``, constraint noise)
drawn from the same torch seed. Pareto fronts: the ZDT fronts are compared
with BoTorch's; the quasi-random DTLZ fronts (botorch_rs's own Sobol
sequence) are checked to lie on the front, to be non-dominated and feasible,
and to have the distribution of BoTorch's.

botorch_rs implements the corrected definitions of meta-pytorch/botorch#3402
and #3405 (``mo_reference.DEVIATIONS``): those problems are compared with
replicas of the fixes, or with BoTorch's own classes when BoTorch already
contains them (``test_known_deviations`` checks which), so the tests pass
against BoTorch main, the released 0.18.1 and the pull-request branches.

Tolerances: 1e-12 relative (per output) for values computed in float64, the
float32 rounding of the result (``2^-23``) for float32 inputs (botorch_rs
computes in float64), 1e-15 for ``max_hv``; bounds and reference points are
exact. Penicillin's explicit Euler simulation is ill-conditioned: perturbing
the inputs by one unit in the last place changes BoTorch's own values by up
to ~1e-3 (relative), so ulp-level differences of exp, expm1 and pow between
math libraries can be amplified as much. Its values are compared with
``max(tolerance, 10 x BoTorch's own change)`` per value (``noise_floor``) and
the errors are reported scaled to the nominal tolerance (and raw). The worst
errors are printed in the terminal summary, section "multi-objective test
functions".
"""

from __future__ import annotations

import inspect
import math
import zlib
from typing import Any

import pytest

from helpers import importorskip_botorch

importorskip_botorch()
torch = pytest.importorskip("torch")

import botorch  # noqa: E402
import botorch.test_functions as bt_tf  # noqa: E402
import mo_reference as ref  # noqa: E402
from botorch.test_functions import base as bt_base  # noqa: E402

import botorch_rs.test_functions as rs_tf  # noqa: E402
from botorch_rs.test_functions import base as rs_base  # noqa: E402
from botorch_rs.test_functions import multi_objective as rs_mo  # noqa: E402
from botorch_rs.test_functions import multi_objective_multi_fidelity as rs_momf  # noqa: E402
from parity_helpers import Checker, rel_err  # noqa: E402

pytestmark = pytest.mark.filterwarnings("ignore")

DT = torch.float64
CHECK = Checker("multi-objective test functions")
#: Values computed in float64 (rounding only; the Penicillin simulation
#: accumulates it over up to 2500 steps, observed <= 1e-14).
TOL = 1e-12
#: float32 inputs: botorch_rs rounds its float64 result to float32 once.
TOL_F32 = 2.0**-23
#: Closed-form constants (max_hv).
TOL_CONST = 1e-15
#: Multiple of BoTorch's own sensitivity to rounding allowed for Penicillin.
NOISE_FLOOR_FACTOR = 10.0

CONFIGS: list[tuple[str, dict[str, int]]] = [
    ("BraninCurrin", {}),
    ("DH1", {"dim": 2}),
    ("DH1", {"dim": 5}),
    ("DH2", {"dim": 3}),
    ("DH3", {"dim": 3}),
    ("DH3", {"dim": 6}),
    ("DH4", {"dim": 3}),
    ("DH4", {"dim": 7}),
    ("DTLZ1", {"dim": 3, "num_objectives": 2}),
    ("DTLZ1", {"dim": 7, "num_objectives": 4}),
    ("DTLZ2", {"dim": 2, "num_objectives": 1}),
    ("DTLZ2", {"dim": 4, "num_objectives": 3}),
    ("DTLZ2", {"dim": 10, "num_objectives": 6}),
    ("DTLZ3", {"dim": 5, "num_objectives": 2}),
    ("DTLZ3", {"dim": 6, "num_objectives": 3}),
    ("DTLZ4", {"dim": 4, "num_objectives": 2}),
    ("DTLZ4", {"dim": 6, "num_objectives": 3}),
    ("DTLZ5", {"dim": 3, "num_objectives": 1}),
    ("DTLZ5", {"dim": 4, "num_objectives": 2}),
    ("DTLZ5", {"dim": 7, "num_objectives": 4}),
    ("DTLZ7", {"dim": 4, "num_objectives": 2}),
    ("DTLZ7", {"dim": 8, "num_objectives": 4}),
    ("GMM", {"num_objectives": 2}),
    ("GMM", {"num_objectives": 3}),
    ("GMM", {"num_objectives": 4}),
    ("Penicillin", {}),
    ("ToyRobust", {}),
    ("VehicleSafety", {}),
    ("ZDT1", {"dim": 2}),
    ("ZDT1", {"dim": 6}),
    ("ZDT2", {"dim": 3}),
    ("ZDT3", {"dim": 2}),
    ("ZDT3", {"dim": 8}),
    ("CarSideImpact", {}),
    ("BNH", {}),
    ("CONSTR", {}),
    ("ConstrainedBraninCurrin", {}),
    ("C2DTLZ2", {"dim": 3, "num_objectives": 2}),
    ("C2DTLZ2", {"dim": 5, "num_objectives": 3}),
    ("C2DTLZ2", {"dim": 6, "num_objectives": 4}),
    ("C2DTLZ2", {"dim": 8, "num_objectives": 5}),
    ("DiscBrake", {}),
    ("MW7", {"dim": 2}),
    ("MW7", {"dim": 6}),
    ("OSY", {}),
    ("SRN", {}),
    ("WeldedBeam", {}),
    ("MOMFBraninCurrin", {}),
    ("MOMFPark", {}),
]
IDS = [f"{n}-" + "-".join(f"{k[0]}{v}" for k, v in kw.items()) for n, kw in CONFIGS]
#: Problems whose constructors take ``constraint_noise_std`` (for the others it
#: is an attribute, set after construction).
CONSTRAINT_NOISE_ARG = {"ConstrainedBraninCurrin", "MW7"}


def rs_class(name: str) -> type:
    module = rs_momf if name in ref.MULTI_FIDELITY else rs_mo
    return getattr(module, name)


def pair(name: str, kwargs: dict[str, Any]) -> tuple[Any, Any]:
    """The reference (BoTorch or the replica of its fix) and botorch_rs."""
    return ref.make(name, kwargs), rs_class(name)(**kwargs)


def rng_seed(*parts: Any) -> int:
    return zlib.crc32(repr(parts).encode())


def t(x: Any) -> Any:
    return torch.as_tensor(x, dtype=DT)


def col_err(a: Any, e: Any) -> float:
    """The worst relative error (``parity_helpers.rel_err``) over the outputs
    (last dimension), which have very different scales."""
    a, e = t(a), t(e)
    assert a.shape == e.shape, (a.shape, e.shape)
    if a.numel() == 0:
        return 0.0
    a2, e2 = a.reshape(-1, a.shape[-1]), e.reshape(-1, e.shape[-1])
    return max(rel_err(a2[:, j], e2[:, j]) for j in range(a2.shape[-1]))


def check(quantity: str, a: Any, e: Any, tol: float) -> float:
    err = col_err(a, e)
    CHECK.record(quantity, err, tol)
    assert err <= tol, (quantity, err, tol)
    return err


def noise_floor(problem: Any, X: Any, k: int = 4, seed: int = 0) -> Any:
    """BoTorch's own sensitivity to rounding: the largest change of each value
    of ``problem`` at ``X`` over ``k`` perturbations of every input by -1, 0 or
    +1 units in the last place (within the bounds)."""
    g = torch.Generator().manual_seed(seed)
    X = X.to(DT)
    lo, hi = problem.bounds[0].to(DT), problem.bounds[1].to(DT)
    y = problem.evaluate_true(X).to(DT)
    sens = torch.zeros_like(y)
    for _ in range(k):
        step = torch.randint(-1, 2, X.shape, generator=g)
        Xp = torch.where(step > 0, torch.nextafter(X, X + 1), X)
        Xp = torch.where(step < 0, torch.nextafter(X, X - 1), Xp)
        Xp = torch.minimum(torch.maximum(Xp, lo), hi)
        sens = torch.maximum(sens, (problem.evaluate_true(Xp).to(DT) - y).abs())
    return sens


def check_floor(quantity: str, a: Any, e: Any, sens: Any, tol: float) -> float:
    """Like ``check``, with the tolerance of each value raised to
    ``NOISE_FLOOR_FACTOR`` times BoTorch's own sensitivity ``sens`` (both
    relative per output); the error is reported scaled to ``tol``."""
    a, e, sens = t(a), t(e), t(sens)
    assert a.shape == e.shape == sens.shape, (a.shape, e.shape, sens.shape)
    a2, e2, s2 = (v.reshape(-1, v.shape[-1]) for v in (a, e, sens))
    scale = e2.abs().max(dim=0).values
    denom = e2.abs() + torch.where(scale > 0, scale, torch.ones_like(scale))
    err = (a2 - e2).abs() / denom
    tols = torch.clamp(NOISE_FLOOR_FACTOR * s2 / denom, min=tol)
    worst = (err / tols).max().item() * tol
    CHECK.record(f"{quantity}, scaled to BoTorch's noise floor", worst, tol)
    CHECK.record(f"{quantity}, raw (information)", err.max().item(), None)  # type: ignore[arg-type]
    assert worst <= tol, (quantity, worst, tol)
    return worst


def corners(bounds: Any, g: Any, n_max: int = 16) -> Any:
    """All corners of the box if there are at most ``n_max``, else the lower
    and upper corners and random corners."""
    lo, hi = bounds[0].to(DT), bounds[1].to(DT)
    d = lo.numel()
    if 2**d <= n_max:
        idx = torch.arange(2**d)
        pattern = ((idx.unsqueeze(-1) >> torch.arange(d)) & 1).bool()
    else:
        rand = torch.rand(n_max - 2, d, generator=g) < 0.5
        pattern = torch.cat([torch.zeros(1, d, dtype=torch.bool), torch.ones(1, d).bool(), rand])
    return torch.where(pattern, hi, lo)


def inside(X32: Any, bounds: Any) -> Any:
    """float32 points moved by one float32 ulp into the float64 bounds where
    rounding to float32 moved them outside (as BoTorch, botorch_rs rejects
    points outside the bounds)."""
    lo, hi = bounds[0].to(DT), bounds[1].to(DT)
    up = torch.nextafter(X32, torch.full_like(X32, math.inf))
    down = torch.nextafter(X32, torch.full_like(X32, -math.inf))
    X32 = torch.where(X32.double() < lo, up, X32)
    return torch.where(X32.double() > hi, down, X32)


def inputs(problem: Any, n: int, seed: int) -> Any:
    """``n`` random points in the domain followed by corners (float64)."""
    g = torch.Generator().manual_seed(seed)
    lo, hi = problem.bounds[0].to(DT), problem.bounds[1].to(DT)
    X = lo + (hi - lo) * torch.rand(n, problem.dim, generator=g, dtype=DT)
    return torch.cat([X, corners(problem.bounds, g)])


# ---------------------------------------------------------------------------
# Attributes
# ---------------------------------------------------------------------------


def test_reference_is_botorch() -> None:
    assert "botorch_rs" not in botorch.__file__


@pytest.mark.parametrize("name,kwargs", CONFIGS, ids=IDS)
def test_attributes(name: str, kwargs: dict) -> None:
    bt, rs = pair(name, kwargs)
    assert rs.dim == bt.dim and rs.num_objectives == bt.num_objectives
    assert torch.equal(rs.bounds, bt.bounds)
    assert [tuple(map(float, b)) for b in bt._bounds] == rs._bounds
    assert torch.equal(rs.ref_point, bt.ref_point)
    assert [float(v) for v in bt._ref_point] == rs._ref_point
    bt_neg, rs_neg = pair(name, {**kwargs, "negate": True})
    assert torch.equal(rs_neg.ref_point, bt_neg.ref_point)
    bt32, rs32 = pair(name, {**kwargs, "dtype": torch.float32})
    assert rs32.bounds.dtype == bt32.bounds.dtype == torch.float32
    assert torch.equal(rs32.bounds, bt32.bounds) and torch.equal(rs32.ref_point, bt32.ref_point)
    assert [tuple(map(float, b)) for b in bt32._bounds] == rs32._bounds
    CHECK.record("bounds, reference points (mismatches)", 0.0, 0.0)
    try:
        expected = bt.max_hv
    except NotImplementedError:
        with pytest.raises(NotImplementedError):
            _ = rs.max_hv
    else:
        err = abs(rs.max_hv - expected) / abs(expected)
        CHECK.record("max_hv", err, TOL_CONST)
        assert err <= TOL_CONST, (rs.max_hv, expected)
        assert rs_neg.max_hv == rs.max_hv  # not negated, as in BoTorch
    assert rs.continuous_inds == bt.continuous_inds
    assert rs.discrete_inds == bt.discrete_inds and rs.categorical_inds == bt.categorical_inds
    with pytest.raises(Exception, match="is not a valid property") as e_rs:
        _ = rs.is_minimization_problem
    with pytest.raises(Exception, match="is not a valid property") as e_bt:
        _ = bt.is_minimization_problem
    assert type(e_rs.value).__name__ == type(e_bt.value).__name__ == "UnsupportedError"
    assert rs.noise_std is None and rs.negate is False
    constrained = isinstance(bt, bt_base.ConstrainedBaseTestProblem)
    assert constrained == isinstance(rs, rs_base.ConstrainedBaseTestProblem)
    if constrained:
        assert rs.num_constraints == bt.num_constraints
        assert rs.constraint_noise_std is bt.constraint_noise_std is None
        assert rs.worst_feasible_value == bt.worst_feasible_value == 0.0
    else:
        assert not hasattr(rs, "evaluate_slack")
    for attr in ("k", "alpha", "beta", "_alpha", "_parts", "_eps", "_con_bounds"):
        if hasattr(bt, attr):
            assert getattr(rs, attr) == getattr(bt, attr), attr
    if name == "Penicillin":
        for attr in ("Y_xs", "Y_ps", "K_1", "K_2", "m_X", "alpha_1", "alpha_2", "alpha_3"):
            assert getattr(rs, attr) == getattr(bt, attr), attr
        for attr in ("mu_X", "K_X", "mu_p", "K_p", "K_I", "K", "k_g", "E_g", "k_d", "E_d"):
            assert getattr(rs, attr) == getattr(bt, attr), attr
        for attr in ("lambd", "T_v", "T_o", "R", "V_max"):
            assert getattr(rs, attr) == getattr(bt, attr), attr
    if name == "GMM":
        for attr in ("gmm_pos", "gmm_covar", "gmm_norm"):
            assert torch.equal(getattr(rs, attr), getattr(bt, attr)), attr
    if name == "ConstrainedBraninCurrin":
        assert torch.equal(rs.con_bounds, bt.con_bounds)
    # gen_pareto_front exists for the same problems.
    try:
        front = bt.gen_pareto_front(2)
    except NotImplementedError:
        with pytest.raises(NotImplementedError):
            rs.gen_pareto_front(2)
    else:
        assert rs.gen_pareto_front(2).shape == front.shape


# ---------------------------------------------------------------------------
# Values (the randomized sweep)
# ---------------------------------------------------------------------------


@pytest.mark.equivalence
@pytest.mark.parametrize("name,kwargs", CONFIGS, ids=IDS)
def test_values(name: str, kwargs: dict) -> None:
    seed = rng_seed(name, kwargs)
    bt, rs = pair(name, kwargs)
    bt_neg, rs_neg = pair(name, {**kwargs, "negate": True})
    X = inputs(bt, 48, seed)
    simulation = name == "Penicillin"

    def compare(quantity: str, a: Any, e: Any, tol: float, X_e: Any, problem: Any = bt) -> None:
        if simulation:  # ill-conditioned: relative to BoTorch's own sensitivity
            check_floor(quantity, a, e, noise_floor(problem, X_e, seed=seed), tol)
        else:
            check(quantity, a, e, tol)

    # float64, unbatched and batched.
    y = bt.evaluate_true(X)
    compare("objectives (float64)", rs.evaluate_true(X), y, TOL, X)
    compare("objectives (negate)", rs_neg(X, noise=False), bt_neg(X, noise=False), TOL, X)
    Xb = X[:48].reshape(3, 2, 8, -1)
    yb = rs.evaluate_true(Xb)
    assert yb.shape == (3, 2, 8, bt.num_objectives) and yb.dtype == DT
    compare("objectives (batched)", yb, bt.evaluate_true(Xb), TOL, Xb)
    compare("objectives (one point)", rs.evaluate_true(X[0]), bt.evaluate_true(X[0]), TOL, X[0])
    # float32 inputs: computed in float64 and rounded.
    X32 = inside(X.float(), bt.bounds)
    y32 = rs.evaluate_true(X32)
    assert y32.dtype == torch.float32
    compare("objectives (float32 inputs)", y32, bt.evaluate_true(X32.double()), TOL_F32, X32)
    CHECK.record(
        "objectives vs BoTorch's float32 arithmetic (information)",
        col_err(y32, bt.evaluate_true(X32)),
        None,  # type: ignore[arg-type]
    )
    # Problems with float32 bounds (`dtype=torch.float32`) validate inputs
    # against them, so they accept their own (rounded) corners.
    bt32, rs32 = pair(name, {**kwargs, "dtype": torch.float32})
    Xc = corners(bt32.bounds, torch.Generator().manual_seed(seed))
    yc = rs32.evaluate_true(Xc)
    yc_bt = bt32.evaluate_true(Xc.double())
    compare("objectives (float32 problem, its corners)", yc, yc_bt, TOL_F32, Xc, bt32)
    # NumPy and nested lists give the same values.
    assert torch.equal(t(rs.evaluate_true(X.numpy())), rs.evaluate_true(X))
    assert torch.equal(t(rs.evaluate_true(X[:4].tolist())), rs.evaluate_true(X[:4]))
    if isinstance(bt, bt_base.ConstrainedBaseTestProblem):
        c = bt.evaluate_slack_true(X)
        check("constraint slacks (float64)", rs.evaluate_slack_true(X), c, TOL)
        check("constraint slacks (negate)", rs_neg.evaluate_slack(X, noise=False), c, TOL)
        check("constraint slacks (batched)", rs.evaluate_slack_true(Xb), bt.evaluate_slack(Xb), TOL)
        c32 = rs.evaluate_slack_true(X32)
        assert c32.dtype == torch.float32
        check(
            "constraint slacks (float32 inputs)", c32, bt.evaluate_slack_true(X32.double()), TOL_F32
        )
        cc = rs32.evaluate_slack_true(Xc)
        check(
            "constraint slacks (float32 problem, its corners)",
            cc,
            bt32.evaluate_slack_true(Xc.double()),
            TOL_F32,
        )
        feasible_rs, feasible_bt = rs.is_feasible(X, noise=False), bt.is_feasible(X, noise=False)
        assert feasible_rs.dtype == torch.bool
        mismatch = feasible_rs != feasible_bt
        # Only points whose smallest slack is zero to rounding may disagree.
        assert bool((c[mismatch].min(dim=-1).values.abs() <= 1e-12 * c.abs().max()).all())
        CHECK.record("is_feasible (mismatches)", float(mismatch.sum()), 0.0)
        assert torch.equal(rs.is_feasible(Xb, noise=False), bt.is_feasible(Xb, noise=False))


@pytest.mark.equivalence
@pytest.mark.parametrize("name,kwargs", CONFIGS, ids=IDS)
def test_noise(name: str, kwargs: dict) -> None:
    """Noise from the same torch seed gives the same values (scalar and
    per-objective ``noise_std``, with and without ``negate``)."""
    seed = rng_seed("noise", name, kwargs)
    probe = ref.make(name, kwargs)
    m = probe.num_objectives
    X = inputs(probe, 24, seed)[:24].reshape(-1, 4, probe.dim)
    sens = noise_floor(probe, X) if name == "Penicillin" else None
    per_objective = [0.05 * (j + 1) for j in range(m)]
    for noise_std, negate in [(0.3, False), (per_objective, True)]:
        bt, rs = pair(name, {**kwargs, "noise_std": noise_std, "negate": negate})
        torch.manual_seed(seed)
        y_bt = bt(X)
        torch.manual_seed(seed)
        y_rs = rs(X)
        if sens is not None:
            check_floor("noisy objectives (same torch seed)", y_rs, y_bt, sens, TOL)
        else:
            check("noisy objectives (same torch seed)", y_rs, y_bt, TOL)
        assert not torch.equal(y_rs, rs(X, noise=False))
    if isinstance(probe, bt_base.ConstrainedBaseTestProblem):
        n_c = probe.num_constraints
        for c_std in [0.2, [0.1 * (j + 1) for j in range(n_c)]]:
            if name in CONSTRAINT_NOISE_ARG:
                bt, rs = pair(name, {**kwargs, "constraint_noise_std": c_std})
            else:
                bt, rs = pair(name, kwargs)
                bt.constraint_noise_std = c_std
                rs.constraint_noise_std = c_std
            torch.manual_seed(seed)
            c_bt = bt.evaluate_slack(X)
            torch.manual_seed(seed)
            c_rs = rs.evaluate_slack(X)
            check("noisy constraint slacks (same torch seed)", c_rs, c_bt, TOL)
            torch.manual_seed(seed + 1)
            f_bt = bt.is_feasible(X)
            torch.manual_seed(seed + 1)
            f_rs = rs.is_feasible(X)
            assert torch.equal(f_rs, f_bt)


@pytest.mark.equivalence
def test_penicillin_long_simulations() -> None:
    """Many points, including long simulations: every point stops at the same
    step as in BoTorch (the time is an objective) unless BoTorch's own step
    is sensitive to rounding, and the values agree to BoTorch's noise floor."""
    bt, rs = pair("Penicillin", {})
    X = inputs(bt, 2048, 7)
    y_bt, y_rs = bt.evaluate_true(X), rs.evaluate_true(X)
    sens = noise_floor(bt, X, seed=7)
    steps_differ = int(((y_bt[:, 2] != y_rs[:, 2]) & (sens[:, 2] == 0)).sum())
    CHECK.record("Penicillin: points stopping at a different step", float(steps_differ), 0.0)
    assert steps_differ == 0
    check_floor("Penicillin objectives (2048 points)", y_rs, y_bt, sens, TOL)
    worst = (sens / (y_bt.abs() + y_bt.abs().max(dim=0).values)).max().item()
    CHECK.record("Penicillin: BoTorch's own change under 1-ulp input changes", worst, None)  # type: ignore[arg-type]
    CHECK.record(
        "Penicillin: longest simulation (steps; information)", float(y_bt[:, 2].max()), None
    )  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Pareto fronts
# ---------------------------------------------------------------------------

ZDT_FRONTS = [("ZDT1", {"dim": 3}), ("ZDT2", {"dim": 2}), ("ZDT3", {"dim": 4})]


@pytest.mark.parametrize("name,kwargs", ZDT_FRONTS, ids=[n for n, _ in ZDT_FRONTS])
def test_zdt_fronts(name: str, kwargs: dict) -> None:
    for negate in (False, True):
        bt, rs = pair(name, {**kwargs, "negate": negate})
        for n in (1, 2, 3, 5, 7, 12, 101):
            check("Pareto fronts (ZDT)", rs.gen_pareto_front(n), bt.gen_pareto_front(n), TOL)
        F = rs.gen_pareto_front(64)
        assert not bool(ref.dominated(F, maximize=negate).any())
        # The front is attained at x = (f_0, 0, ..., 0).
        f0 = -F[:, 0] if negate else F[:, 0]
        X = torch.cat([f0.unsqueeze(-1), torch.zeros(64, rs.dim - 1, dtype=DT)], dim=-1)
        check("Pareto fronts (ZDT) attained", rs(X, noise=False), F, TOL)


DTLZ_FRONTS = [
    ("DTLZ1", {"dim": 4, "num_objectives": 2}),
    ("DTLZ1", {"dim": 6, "num_objectives": 4}),
    ("DTLZ2", {"dim": 3, "num_objectives": 2}),
    ("DTLZ2", {"dim": 5, "num_objectives": 3}),
    ("DTLZ3", {"dim": 6, "num_objectives": 3}),
    ("DTLZ4", {"dim": 7, "num_objectives": 4}),
    ("C2DTLZ2", {"dim": 3, "num_objectives": 2}),
    ("C2DTLZ2", {"dim": 5, "num_objectives": 3}),
    ("C2DTLZ2", {"dim": 6, "num_objectives": 4}),
]


@pytest.mark.parametrize("name,kwargs", DTLZ_FRONTS, ids=[f"{n}-{k}" for n, k in DTLZ_FRONTS])
def test_dtlz_fronts(name: str, kwargs: dict) -> None:
    """The quasi-random fronts lie on the Pareto front, are non-dominated and
    feasible, and have the distribution of BoTorch's (whose points come from
    torch's SobolEngine and cannot be reproduced)."""
    n = 1024
    m = kwargs["num_objectives"]
    for negate in (False, True):
        bt, rs = pair(name, {**kwargs, "negate": negate})
        F = rs.gen_pareto_front(n, seed=rng_seed(name, kwargs))
        assert F.shape == (n, m) and F.dtype == DT
        G = -F if negate else F
        assert bool((G >= 0).all())
        residual = (G.sum(-1) - 0.5) if name == "DTLZ1" else (G.norm(dim=-1) - 1.0)
        CHECK.record(
            "Pareto fronts (DTLZ): distance to the front", residual.abs().max().item(), 1e-14
        )
        assert residual.abs().max().item() <= 1e-14
        assert not bool(ref.dominated(F, maximize=negate).any())
        torch.manual_seed(rng_seed("bt front", name, kwargs))
        F_bt = bt.gen_pareto_front(n)
        G_bt = -F_bt if negate else F_bt
        if name == "C2DTLZ2":
            feasible = ref.c2dtlz2_feasible(G)
            CHECK.record(
                "Pareto fronts (C2DTLZ2): infeasible points", float((~feasible).sum()), 0.0
            )
            assert bool(feasible.all())
            # BoTorch's front (DTLZ2's) is partly infeasible; compare with its feasible part.
            feasible_bt = ref.c2dtlz2_feasible(G_bt)
            CHECK.record(
                "Pareto fronts (C2DTLZ2): BoTorch's infeasible fraction (information)",
                1.0 - feasible_bt.double().mean().item(),
                None,  # type: ignore[arg-type]
            )
            G_bt = G_bt[feasible_bt]
            # The problem's own constraint at a pre-image on the Pareto set.
            if m == 2:
                x0 = torch.atan2(G[:, 1], G[:, 0]) * 2 / math.pi
                X = torch.cat([x0.unsqueeze(-1), torch.full((n, rs.dim - 1), 0.5, dtype=DT)], -1)
                assert bool(rs.is_feasible(X, noise=False).all())
                assert bool(ref.make(name, kwargs).is_feasible(X, noise=False).all())
        err = (G.mean(0) - G_bt.mean(0)).abs().max().item()
        CHECK.record("Pareto fronts (DTLZ): mean vs BoTorch's (statistical)", err, 0.03)
        assert err <= 0.03
        err = (torch.cov(G.T) - torch.cov(G_bt.T)).abs().max().item()
        CHECK.record("Pareto fronts (DTLZ): covariance vs BoTorch's (statistical)", err, 0.02)
        assert err <= 0.02


# ---------------------------------------------------------------------------
# Deviations, errors, signatures and the namespace
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", sorted(ref.DEVIATIONS))
def test_known_deviations(name: str) -> None:
    """botorch_rs follows the fix; BoTorch differs unless it contains it, in
    which case the replica of the fix must reproduce BoTorch exactly."""
    dev = ref.DEVIATIONS[name]
    kwargs = next(kw for n, kw in CONFIGS if n == name)
    rs = rs_class(name)(**kwargs)
    upstream = ref.bt_class(name)(**kwargs)
    replica = ref.FIXED[name](**kwargs)
    assert rs.deviation is not None and dev.pr in rs.deviation
    X = inputs(upstream, 32, rng_seed("deviation", name))
    if ref.botorch_has_fix(name):
        quantity = "deviations: replica of the fix vs fixed BoTorch"
        check(quantity, replica.evaluate_true(X), upstream.evaluate_true(X), 0.0)
        if isinstance(upstream, bt_base.ConstrainedBaseTestProblem):
            check(quantity, replica.evaluate_slack_true(X), upstream.evaluate_slack_true(X), 0.0)
        assert replica._max_hv == upstream._max_hv
        CHECK.record(f"deviations: fixed in BoTorch ({name})", 0.0, None)  # type: ignore[arg-type]
        return
    differs = []
    if "objectives" in dev.quantities:
        differs.append(col_err(upstream.evaluate_true(X), rs.evaluate_true(X)) > 1e-6)
    if "dtype" in dev.quantities:
        # BoTorch computes in float32: the values agree to float32 precision.
        y_bt, y_rs = upstream.evaluate_true(X), rs.evaluate_true(X)
        differs.append(y_bt.dtype == torch.float32 and y_rs.dtype == DT)
        check("deviations: GMM vs BoTorch main's float32 values", y_rs, y_bt.double(), 1e-6)
    if "slack" in dev.quantities:
        try:
            differs.append(
                col_err(upstream.evaluate_slack_true(X), rs.evaluate_slack_true(X)) > 1e-6
            )
        except NotImplementedError:
            differs.append(True)
    if "max_hv" in dev.quantities:
        differs.append(upstream._max_hv != rs._max_hv)
    assert differs and all(differs), (name, dev)
    CHECK.record("deviations from BoTorch main confirmed (count)", float(len(differs)), None)  # type: ignore[arg-type]


def test_c2dtlz2_front_deviation() -> None:
    """BoTorch's C2DTLZ2 returns DTLZ2's front, which is partly infeasible."""
    assert "C2DTLZ2" in ref.FRONT_DEVIATIONS
    bt = ref.make("C2DTLZ2", {"dim": 3, "num_objectives": 2})
    torch.manual_seed(0)
    F = bt.gen_pareto_front(200)
    assert not bool(ref.c2dtlz2_feasible(F).all())


ERROR_CASES: list[tuple[str, dict[str, Any]]] = [
    ("DH1", {"dim": 1}),
    ("DH3", {"dim": 2}),
    ("DTLZ2", {"dim": 3, "num_objectives": 3}),
    ("C2DTLZ2", {"dim": 2}),
    ("ZDT1", {"dim": 4, "num_objectives": 3}),
    ("ZDT3", {"dim": 1}),
    ("GMM", {"num_objectives": 5}),
    ("GMM", {"num_objectives": 1}),
    ("MW7", {"dim": 1}),
    ("BraninCurrin", {"noise_std": [0.1, 0.2, 0.3]}),
    ("DTLZ1", {"dim": 4, "num_objectives": 3, "noise_std": [0.1]}),
    ("DTLZ", {"dim": 3}),
    ("DH", {"dim": 3}),
    ("ZDT", {"dim": 3}),
]


@pytest.mark.parametrize("name,kwargs", ERROR_CASES, ids=[f"{n}-{k}" for n, k in ERROR_CASES])
def test_constructor_errors(name: str, kwargs: dict) -> None:
    with pytest.raises(Exception) as e_bt:
        getattr(ref.bt_mo, name)(**kwargs)
    with pytest.raises(Exception) as e_rs:
        getattr(rs_mo, name)(**kwargs)
    bt_type = type(e_bt.value)
    if bt_type.__name__ in ("UnsupportedError", "InputDataError"):
        assert type(e_rs.value).__name__ == bt_type.__name__
    else:
        assert isinstance(e_rs.value, bt_type), (e_rs.value, e_bt.value)
    if not isinstance(e_bt.value, TypeError):
        assert str(e_rs.value) == str(e_bt.value)
    CHECK.record("errors: different types or messages", 0.0, 0.0)


def test_dtlz7_needs_two_objectives() -> None:
    """BoTorch constructs DTLZ7 with one objective but fails to evaluate it;
    botorch_rs rejects it when it is created."""
    bt = ref.bt_mo.DTLZ7(dim=3, num_objectives=1)
    with pytest.raises(RuntimeError):
        bt.evaluate_true(torch.rand(2, 3, dtype=DT))
    with pytest.raises(ValueError, match="DTLZ7 requires num_objectives >= 2"):
        rs_mo.DTLZ7(dim=3, num_objectives=1)


@pytest.mark.parametrize("name,kwargs", CONFIGS[::5], ids=IDS[::5])
def test_input_errors(name: str, kwargs: dict) -> None:
    bt, rs = pair(name, kwargs)
    X = inputs(bt, 2, 0)[:2]
    outside = X.clone()
    outside[0, 0] = bt.bounds[1, 0] + 1.0
    for f_bt, f_rs in [(bt.evaluate_true, rs.evaluate_true), (bt, rs)]:
        with pytest.raises(ValueError, match="within the bounds") as e_bt:
            f_bt(outside)
        with pytest.raises(ValueError, match="within the bounds") as e_rs:
            f_rs(outside)
        assert str(e_rs.value) == str(e_bt.value)
        prefix = r"Expected `X` to have shape `\(batch_shape\) x n x d`\."
        with pytest.raises(ValueError, match=prefix):
            f_bt(X[:, :-1])
        with pytest.raises(ValueError, match=prefix):
            f_rs(X[:, :-1])


ALL_CLASSES = ref.MULTI_OBJECTIVE + ref.MULTI_FIDELITY + ref.ABSTRACT


@pytest.mark.parametrize("name", ALL_CLASSES)
def test_signatures(name: str) -> None:
    """BoTorch's parameters with the same names, order and plain defaults;
    botorch_rs only adds keyword-only arguments (``seed``)."""
    bt_cls = getattr(ref.bt_momf if name in ref.MULTI_FIDELITY else ref.bt_mo, name)
    rs_cls = getattr(rs_momf if name in ref.MULTI_FIDELITY else rs_mo, name)
    pairs = [(bt_cls.__init__, rs_cls.__init__)]
    for method in ("forward", "evaluate_true", "evaluate_slack", "evaluate_slack_true"):
        if hasattr(bt_cls, method):
            pairs.append((getattr(bt_cls, method), getattr(rs_cls, method)))
    for method in ("is_feasible", "gen_pareto_front"):
        if hasattr(bt_cls, method):
            pairs.append((getattr(bt_cls, method), getattr(rs_cls, method)))
    for bt_fn, rs_fn in pairs:
        bt_params = [p for p in inspect.signature(bt_fn).parameters.values() if p.name != "self"]
        rs_params = [p for p in inspect.signature(rs_fn).parameters.values() if p.name != "self"]
        rs_positional = [p for p in rs_params if p.kind != p.KEYWORD_ONLY]
        assert [p.name for p in rs_positional] == [p.name for p in bt_params], bt_fn
        for p, q in zip(bt_params, rs_positional, strict=True):
            if isinstance(p.default, (type(None), bool, int, float, str)):
                assert q.default == p.default, (bt_fn, p.name)
        assert all(q.kind == q.KEYWORD_ONLY for q in rs_params[len(rs_positional) :])
    assert issubclass(rs_cls, rs_base.MultiObjectiveTestProblem)
    assert issubclass(rs_cls, rs_base.ConstrainedBaseTestProblem) == issubclass(
        bt_cls, bt_base.ConstrainedBaseTestProblem
    )


def test_namespace() -> None:
    """``botorch_rs.test_functions`` re-exports BoTorch's multi-objective
    problems."""
    modules = ("multi_objective", "multi_objective_multi_fidelity")

    def exported(package: Any) -> set[str]:
        return {
            n for n in package.__all__ if getattr(package, n).__module__.split(".")[-1] in modules
        }

    names = exported(bt_tf)  # all but DH1-DH4, as in BoTorch
    assert exported(rs_tf) == names
    assert names == {n for n in ref.MULTI_OBJECTIVE + ref.MULTI_FIDELITY if not n.startswith("DH")}
    for n in names:
        assert getattr(rs_tf, n) is rs_class(n)
    public = [n for n in dir(ref.bt_mo) if inspect.isclass(getattr(ref.bt_mo, n))]
    public = [n for n in public if getattr(ref.bt_mo, n).__module__ == ref.bt_mo.__name__]
    assert sorted(public) == sorted(ref.MULTI_OBJECTIVE + ref.ABSTRACT)
    assert all(hasattr(rs_mo, n) for n in public)

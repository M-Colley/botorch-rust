"""Parity with BoTorch for optimization and candidate generation (phase 9),
through the Python API, side by side on identical models and inputs:

* ``optimize_acqf_discrete`` (``q = 1``, sequential ``q > 1``, ``unique``,
  ``X_avoid``, constraints, too few choices), ``optimize_acqf_mixed`` (best
  and all restarts, sequential ``q = 2``, post-processing, combinations that
  fail), ``optimize_acqf_discrete_local_search`` (given starts; BoTorch's
  random lattice points replaced by the same seeded points in both
  libraries), ``optimize_acqf_mixed_alternating`` (BoTorch's
  ``generate_starting_points`` replaced by the same points; fixed features,
  continuous relaxation, post-processing; the building blocks
  ``discrete_step`` / ``continuous_step`` / ``get_nearest_neighbors`` /
  ``get_categorical_neighbors`` / ``round_discrete_dims``),
  ``optimize_acqf_cyclic`` (a deterministic ``ic_generator``),
  ``optimize_acqf_homotopy`` (a torch acquisition function whose sharpness
  is the homotopy parameter; the schedules);
* ``gen_candidates_torch`` (defaults, options, the callback),
  ``gen_candidates_scipy`` (parallel and serial modes), ``optimize_acqf``
  with ``gen_candidates_torch``, ``post_processing_func`` and
  ``timeout_sec``; the initializers ``initialize_q_batch``,
  ``initialize_q_batch_nonneg``, ``initialize_q_batch_topn`` (with torch's
  random draws under the same seed), the truncated normal perturbations
  (with BoTorch's Sobol draws replaced by the same points);
* ``MaxPosteriorSampling`` (with and without replacement, batches),
  ``BoltzmannSampling`` and ``ConstrainedMaxPosteriorSampling`` under the same
  torch seed, ``get_best_candidates``;
* ``fit_gpytorch_mll_torch`` (step limits; until the stopping criterion
  against BoTorch's outcomes on twins with perturbed training inputs),
  ``fit_gpytorch_mll_scipy`` (single and batched models, parameter subsets,
  bounds), ``ExpMAStoppingCriterion``, the closures and parameter helpers;
* ``gen_loo_cv_folds`` and ``batch_cross_validation``;
* ``optimize_acqf`` with ``qKnowledgeGradient`` (phase 8) and the options of
  this phase (``post_processing_func``, ``timeout_sec``,
  ``gen_candidates_torch``, ``return_full_tree``);
* the exceptions of invalid inputs.

All comparisons start from identical inputs (initial conditions, recorded or
replaced random draws), so values must agree to 1e-9 (relative) and
candidates of continuous optimizers to 1e-7 (see ``optim9_helpers.XTOL``).
BoTorch 0.18.1 takes the serial L-BFGS-B path with SciPy 1.18, so the
default (parallel) path is compared only against BoTorch main; the serial
path is compared against both. Cases that BoTorch gets wrong (see
``docs/drafts/9-optim.md``) are compared against corrected references.
Skipped unless BoTorch is importable.
"""

from __future__ import annotations

import copy
import math
import warnings
from typing import Any

import pytest

from helpers import importorskip_botorch

importorskip_botorch()
torch = pytest.importorskip("torch")

import botorch.optim.optimize as bto  # noqa: E402
import botorch.optim.optimize_mixed as btm  # noqa: E402
from botorch.cross_validation import batch_cross_validation as bt_bcv  # noqa: E402
from botorch.cross_validation import gen_loo_cv_folds as bt_folds  # noqa: E402
from botorch.generation import gen as bt_gen  # noqa: E402
from botorch.generation import sampling as bt_sampling  # noqa: E402
from botorch.models import ModelListGP as BtModelListGP  # noqa: E402
from botorch.models import SingleTaskGP as BtSingleTaskGP  # noqa: E402
from botorch.optim import fit as bt_fit  # noqa: E402
from botorch.optim import homotopy as bt_homotopy  # noqa: E402
from botorch.optim import initializers as bt_init  # noqa: E402
from botorch.optim import optimize_homotopy as bt_oh  # noqa: E402
from botorch.optim import stopping as bt_stopping  # noqa: E402
from gpytorch.mlls import ExactMarginalLogLikelihood as BtMLL  # noqa: E402
from optim9_helpers import (  # noqa: E402
    DT,
    JOINT,
    VALUE_TOL,
    XTOL,
    SharpPeak,
    botorch_parallel_lbfgsb,
    lattice_points,
    lbfgsb_modes,
    pairs,
    patched,
    starting_points,
)

import botorch_rs.optim.optimize as rso  # noqa: E402
import botorch_rs.optim.optimize_mixed as rsm  # noqa: E402
from botorch_rs.cross_validation import batch_cross_validation as rs_bcv  # noqa: E402
from botorch_rs.cross_validation import gen_loo_cv_folds as rs_folds  # noqa: E402
from botorch_rs.generation import gen as rs_gen  # noqa: E402
from botorch_rs.generation import sampling as rs_sampling  # noqa: E402
from botorch_rs.mlls import ExactMarginalLogLikelihood as RsMLL  # noqa: E402
from botorch_rs.models import ModelListGP as RsModelListGP  # noqa: E402
from botorch_rs.models import SingleTaskGP as RsSingleTaskGP  # noqa: E402
from botorch_rs.optim import fit as rs_fit  # noqa: E402
from botorch_rs.optim import homotopy as rs_homotopy  # noqa: E402
from botorch_rs.optim import initializers as rs_init  # noqa: E402
from botorch_rs.optim import optimize_homotopy as rs_oh  # noqa: E402
from botorch_rs.optim import stopping as rs_stopping  # noqa: E402
from parity_helpers import Checker, ModelSpec, model_pair  # noqa: E402

CHECK = Checker("optimization and candidate generation (phase 9)")


@pytest.fixture(autouse=True, scope="module")
def _single_threaded_torch() -> Any:
    n = torch.get_num_threads()
    torch.set_num_threads(1)
    warnings.simplefilter("ignore")
    yield
    torch.set_num_threads(n)


@pytest.fixture(scope="module", params=[(3, 12, 3), (4, 16, 5)], ids=["d3", "d4"])
def p(request: pytest.FixtureRequest) -> dict[str, Any]:
    d, n, seed = request.param
    return pairs(d=d, n=n, seed=seed)


def _gen(seed: int) -> Any:
    return torch.Generator().manual_seed(seed)


def check_pair(label: str, rs: tuple[Any, Any], bt: tuple[Any, Any], xtol: float = XTOL) -> None:
    """Candidates (``xtol``) and values (1e-9) of an optimization."""
    CHECK.check(f"{label}: candidates", rs[0], bt[0], xtol)
    if bt[1] is None:
        assert rs[1] is None
        return
    CHECK.check(f"{label}: values", rs[1], bt[1], VALUE_TOL)


# ---------------------------------------------------------------------------
# optimize_acqf_discrete
# ---------------------------------------------------------------------------


def test_optimize_acqf_discrete(p: dict[str, Any]) -> None:
    g = _gen(10)
    choices = torch.rand(80, p["d"], generator=g, dtype=DT)
    (a_bt, a_rs), (q_bt, q_rs) = p["logei"], p["qlogei"]
    check_pair(
        "discrete q=1",
        rso.optimize_acqf_discrete(a_rs, 1, choices),
        bto.optimize_acqf_discrete(a_bt, 1, choices),
        0.0,
    )
    cons = [(torch.tensor([0, 1]), torch.tensor([1.0, 0.5], dtype=DT), 0.5)]
    for kwargs in (
        {},
        {"unique": False},
        {"X_avoid": choices[:7]},
        {"inequality_constraints": cons},
        {"X_avoid": choices[:7], "inequality_constraints": cons, "max_batch_size": 9},
        {"return_acq_values": False},
    ):
        bt = bto.optimize_acqf_discrete(q_bt, 3, choices, **kwargs)
        rs = rso.optimize_acqf_discrete(q_rs, 3, choices, **kwargs)
        check_pair(f"discrete q=3 {sorted(kwargs)}", rs, bt, 0.0)
        assert q_rs.X_pending is None and q_bt.X_pending is None
    # Fewer unique choices than q: the same warning and reduced q.
    with pytest.warns(Warning) as w_bt:
        bt = bto.optimize_acqf_discrete(q_bt, 5, choices[:3])
    with pytest.warns(Warning) as w_rs:
        rs = rso.optimize_acqf_discrete(q_rs, 5, choices[:3])
    check_pair("discrete too few choices", rs, bt, 0.0)
    assert [type(w.message).__name__ for w in w_rs] == [type(w.message).__name__ for w in w_bt]
    assert str(w_rs[0].message) == str(w_bt[0].message)
    # Pending points are restored and appended to.
    pending = choices[-2:]
    q_bt.set_X_pending(pending)
    q_rs.set_X_pending(pending)
    try:
        check_pair(
            "discrete q=2 with pending",
            rso.optimize_acqf_discrete(q_rs, 2, choices),
            bto.optimize_acqf_discrete(q_bt, 2, choices),
            0.0,
        )
        CHECK.check("discrete pending restored", q_rs.X_pending, q_bt.X_pending, 0.0)
    finally:
        q_bt.set_X_pending(None)
        q_rs.set_X_pending(None)


# ---------------------------------------------------------------------------
# optimize_acqf_mixed
# ---------------------------------------------------------------------------


def test_optimize_acqf_mixed(p: dict[str, Any]) -> None:
    d = p["d"]
    g = _gen(11)
    ics = torch.rand(5, 1, d, generator=g, dtype=DT)
    ffl = [{0: 0.0}, {0: 0.5, d - 1: 0.25}, {-d: 1.0}]
    (a_bt, a_rs), (q_bt, q_rs) = p["logei"], p["qlogei"]
    bounds = p["bounds"]
    for options in lbfgsb_modes():
        tag = "serial" if options else "parallel"
        kw = dict(batch_initial_conditions=ics, options=options)
        check_pair(
            f"mixed q=1 [{tag}]",
            rso.optimize_acqf_mixed(a_rs, bounds, 1, 5, ffl, **kw),
            bto.optimize_acqf_mixed(a_bt, bounds, 1, 5, ffl, **kw),
        )
        check_pair(
            f"mixed q=1 all restarts [{tag}]",
            rso.optimize_acqf_mixed(a_rs, bounds, 1, 5, ffl, return_best_only=False, **kw),
            bto.optimize_acqf_mixed(a_bt, bounds, 1, 5, ffl, return_best_only=False, **kw),
        )
        check_pair(
            f"mixed q=2 [{tag}]",
            rso.optimize_acqf_mixed(q_rs, bounds, 2, 5, ffl, **kw),
            bto.optimize_acqf_mixed(q_bt, bounds, 2, 5, ffl, **kw),
        )
        pp = lambda X: (X * 8).round() / 8  # noqa: E731
        check_pair(
            f"mixed q=1 post-processing [{tag}]",
            rso.optimize_acqf_mixed(a_rs, bounds, 1, 5, ffl, post_processing_func=pp, **kw),
            bto.optimize_acqf_mixed(a_bt, bounds, 1, 5, ffl, post_processing_func=pp, **kw),
        )


def test_optimize_acqf_mixed_failing_combinations(p: dict[str, Any]) -> None:
    """A combination of fixed features that violates a linear constraint is
    skipped with BoTorch's warning (SLSQP: compared by the chosen
    combination and values to SLSQP accuracy)."""
    d = p["d"]
    (a_bt, a_rs) = p["logei"]
    cons = [(torch.tensor([0, 1]), torch.tensor([1.0, 1.0], dtype=DT), 0.9)]
    ffl = [{0: 0.0, 1: 0.1}, {0: 0.7}]
    ics = torch.rand(4, 1, d, generator=_gen(12), dtype=DT)
    ics[..., 1] = 0.5
    with pytest.warns(Warning) as w_bt:
        bt = bto.optimize_acqf_mixed(
            a_bt, p["bounds"], 1, 4, ffl, batch_initial_conditions=ics, inequality_constraints=cons
        )
    with pytest.warns(Warning) as w_rs:
        rs = rso.optimize_acqf_mixed(
            a_rs, p["bounds"], 1, 4, ffl, batch_initial_conditions=ics, inequality_constraints=cons
        )
    msg = [str(w.message) for w in w_bt if "Candidate generation failed" in str(w.message)]
    assert msg and msg == [
        str(w.message) for w in w_rs if "Candidate generation failed" in str(w.message)
    ]
    assert float(rs[0][0, 0]) == float(bt[0][0, 0]) == 0.7
    # SciPy >= 1.16 solves SLSQP's subproblems with LAPACK (see
    # test_parity_constraints.py): values agree to the solver's accuracy.
    CHECK.check("mixed constrained value (SLSQP)", rs[1], bt[1], 1e-6)
    with pytest.raises(Exception) as e_bt:
        bto.optimize_acqf_mixed(
            a_bt,
            p["bounds"],
            1,
            4,
            ffl[:1],
            batch_initial_conditions=ics,
            inequality_constraints=cons,
        )
    with pytest.raises(Exception) as e_rs:
        rso.optimize_acqf_mixed(
            a_rs,
            p["bounds"],
            1,
            4,
            ffl[:1],
            batch_initial_conditions=ics,
            inequality_constraints=cons,
        )
    assert type(e_bt.value).__name__ == type(e_rs.value).__name__ == "CandidateGenerationError"
    assert str(e_bt.value) == str(e_rs.value)


# ---------------------------------------------------------------------------
# optimize_acqf_discrete_local_search
# ---------------------------------------------------------------------------


def test_local_search(p: dict[str, Any]) -> None:
    d = p["d"]
    levels = torch.tensor([0.0, 0.2, 0.4, 0.6, 0.8, 1.0], dtype=DT)
    dc = [levels] * d
    (a_bt, a_rs), (q_bt, q_rs) = p["logei"], p["qlogei"]
    bic = levels[torch.randint(6, (5, 1, d), generator=_gen(13))]
    check_pair(
        "local search q=1 (given starts)",
        rso.optimize_acqf_discrete_local_search(
            a_rs, dc, 1, num_restarts=5, batch_initial_conditions=bic
        ),
        bto.optimize_acqf_discrete_local_search(
            a_bt, dc, 1, num_restarts=5, batch_initial_conditions=bic
        ),
        0.0,
    )
    cons = [(torch.tensor([0, 1]), torch.tensor([1.0, 1.0], dtype=DT), 0.6)]
    for kwargs in (
        {},
        {"unique": False},
        {"inequality_constraints": cons},
        {"X_avoid": levels[:3].unsqueeze(-1).expand(3, d)},
    ):
        lp = lattice_points(17)
        with (
            patched(bto, "_gen_batch_initial_conditions_local_search", lp),
            patched(rso, "_gen_batch_initial_conditions_local_search", lp),
        ):
            bt = bto.optimize_acqf_discrete_local_search(
                q_bt, dc, 3, num_restarts=4, raw_samples=64, **kwargs
            )
            rs = rso.optimize_acqf_discrete_local_search(
                q_rs, dc, 3, num_restarts=4, raw_samples=64, **kwargs
            )
        check_pair(f"local search q=3 {sorted(kwargs)}", rs, bt, 0.0)
        assert q_bt.X_pending is None and q_rs.X_pending is None


# ---------------------------------------------------------------------------
# optimize_acqf_mixed_alternating
# ---------------------------------------------------------------------------


def _alt_case(d: int, seed: int) -> tuple[dict, dict, Any]:
    discrete_dims = {0: [0.0, 0.25, 0.5, 0.75, 1.0]}
    cat_dims = {d - 1: [0.0, 0.5, 1.0]}
    g = _gen(seed)
    start = torch.rand(6, d, generator=g, dtype=DT)
    start[:, 0] = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0, 0.5], dtype=DT)
    start[:, d - 1] = torch.tensor([0.0, 0.5, 1.0, 0.0, 0.5, 1.0], dtype=DT)
    return discrete_dims, cat_dims, start


@pytest.mark.parametrize("variant", ["plain", "fixed", "relaxed", "post", "q2", "options"])
def test_mixed_alternating(p: dict[str, Any], variant: str) -> None:
    d = p["d"]
    discrete_dims, cat_dims, start = _alt_case(d, 14)
    (a_bt, a_rs), (q_bt, q_rs) = p["logei"], p["qlogei"]
    acq_bt, acq_rs, q = (q_bt, q_rs, 2) if variant == "q2" else (a_bt, a_rs, 1)
    kwargs: dict[str, Any] = dict(
        discrete_dims=discrete_dims, cat_dims=cat_dims, num_restarts=6, raw_samples=32, q=q
    )
    if variant == "fixed":
        kwargs["fixed_features"] = {1: 0.3}
        start = start.clone()
        start[:, 1] = 0.3
    if variant == "relaxed":
        kwargs["discrete_dims"] = {0: [i / 29 for i in range(30)]}
        start = start.clone()
        start[:, 0] = (start[:, 0] * 29).round() / 29
    if variant == "post":
        kwargs["post_processing_func"] = lambda X: (X * 16).round() / 16
    if variant == "options":
        kwargs["options"] = {"maxiter_discrete": 2, "maxiter_continuous": 4, "tol": 1e-6}
    with (
        patched(btm, "generate_starting_points", starting_points(start, acq_bt)),
        patched(rsm, "generate_starting_points", starting_points(start, acq_bt)),
    ):
        bt = btm.optimize_acqf_mixed_alternating(acq_bt, p["bounds"], **kwargs)
        rs = rsm.optimize_acqf_mixed_alternating(acq_rs, p["bounds"], **kwargs)
    _check_alternating(f"alternating [{variant}]", rs, bt)


def _check_alternating(label: str, rs: Any, bt: Any) -> None:
    """Exact with BoTorch's parallel L-BFGS-B path (main). BoTorch 0.18.1
    with SciPy 1.18 runs the continuous steps on SciPy's serial path, where
    BoTorch's ``tol`` (``gtol = 1e-8`` on the parallel path) is not a SciPy
    option, so it stops at SciPy's ``gtol = 1e-5``: there botorch_rs (which
    follows main) must reach BoTorch's value."""
    if botorch_parallel_lbfgsb():
        check_pair(label, rs, bt)
    else:
        shortfall = float(bt[1].reshape(-1)[0]) - float(rs[1].reshape(-1)[0])
        CHECK.check_bound(
            f"{label} (0.18.1): shortfall",
            shortfall / max(abs(float(bt[1].reshape(-1)[0])), 1.0),
            1e-9,
        )


def test_mixed_alternating_building_blocks(p: dict[str, Any]) -> None:
    d = p["d"]
    discrete_dims, cat_dims, start = _alt_case(d, 15)
    (a_bt, a_rs) = p["logei"]
    bounds = p["bounds"]
    X = torch.rand(7, d, generator=_gen(16), dtype=DT)
    dd = {0: [0.0, 0.3, 0.7, 1.0], 1: [0.0, 1.0]}
    CHECK.check(
        "round_discrete_dims", rsm.round_discrete_dims(X, dd), btm.round_discrete_dims(X, dd), 0.0
    )
    x = btm.round_discrete_dims(X, dd)[0]
    CHECK.check(
        "get_nearest_neighbors",
        rsm.get_nearest_neighbors(x, bounds, dd),
        btm.get_nearest_neighbors(x, bounds, dd),
        0.0,
    )
    cd = {d - 1: [0.0, 0.25, 0.5, 1.0]}
    CHECK.check(
        "get_categorical_neighbors",
        rsm.get_categorical_neighbors(x, cd),
        btm.get_categorical_neighbors(x, cd),
        0.0,
    )
    assert rsm.complement_indices([0, 2], 4) == btm.complement_indices([0, 2], 4)
    idx = torch.tensor([1, 3])
    assert (
        rsm.complement_indices_like(idx, 5).tolist() == btm.complement_indices_like(idx, 5).tolist()
    )
    for dims in ({0: [0, 1]}, {0: [0, 1, 2]}, {0: list(range(4)), 1: list(range(3))}, None):
        assert rsm.should_use_mixed_alternating_optimizer(
            dims, {2: [0, 1]}
        ) == btm.should_use_mixed_alternating_optimizer(dims, {2: [0, 1]})
    # discrete_step / continuous_step from the same points.
    from botorch.optim.optimize import OptimizeAcqfInputs as BtInputs

    from botorch_rs.optim.optimize import OptimizeAcqfInputs as RsInputs

    common = dict(
        bounds=bounds,
        q=1,
        num_restarts=6,
        raw_samples=None,
        options={},
        inequality_constraints=None,
        equality_constraints=None,
        nonlinear_inequality_constraints=None,
        fixed_features=None,
        post_processing_func=None,
        batch_initial_conditions=start.unsqueeze(1),
        return_best_only=False,
        sequential=False,
    )
    oi_bt = BtInputs(acq_function=a_bt, gen_candidates=bt_gen.gen_candidates_scipy, **common)
    oi_rs = RsInputs(acq_function=a_rs, gen_candidates=None, **common)
    xb, vb = btm.discrete_step(oi_bt, discrete_dims, cat_dims, start.clone())
    xr, vr = rsm.discrete_step(oi_rs, discrete_dims, cat_dims, start.clone())
    check_pair("discrete_step", (xr, vr), (xb, vb), 0.0)
    dt = torch.tensor(list(discrete_dims), dtype=torch.long)
    ct = torch.tensor(list(cat_dims), dtype=torch.long)
    xb, vb = btm.continuous_step(oi_bt, dt, ct, start.clone())
    xr, vr = rsm.continuous_step(oi_rs, dt, ct, start.clone())
    if botorch_parallel_lbfgsb():
        check_pair("continuous_step", (xr, vr), (xb, vb))
    else:  # SciPy's gtol on BoTorch 0.18.1's serial path (see _check_alternating)
        CHECK.check_bound("continuous_step (0.18.1): shortfall", float((vb - vr).max()), 1e-9)


def test_mixed_alternating_native_quality(p: dict[str, Any]) -> None:
    """With botorch_rs's own random starting points, the best value over
    seeds reaches BoTorch's (statistical: the random draws differ)."""
    d = p["d"]
    discrete_dims, cat_dims, _ = _alt_case(d, 18)
    (a_bt, a_rs) = p["logei"]
    kw = dict(discrete_dims=discrete_dims, cat_dims=cat_dims, num_restarts=6, raw_samples=64)
    torch.manual_seed(0)
    v_bt = max(
        float(btm.optimize_acqf_mixed_alternating(a_bt, p["bounds"], **kw)[1]) for _ in range(3)
    )
    v_rs = max(
        float(rsm.optimize_acqf_mixed_alternating(a_rs, p["bounds"], seed=s, **kw)[1])
        for s in range(3)
    )
    CHECK.check_bound(
        "alternating native: shortfall of the best of 3 seeds",
        (v_bt - v_rs) / max(abs(v_bt), 1.0),
        1e-6,
    )


# ---------------------------------------------------------------------------
# optimize_acqf_cyclic and optimize_acqf_homotopy
# ---------------------------------------------------------------------------


def test_cyclic(p: dict[str, Any]) -> None:
    d = p["d"]
    (q_bt, q_rs) = p["qlogei"]
    ics = torch.rand(4, 1, d, generator=_gen(19), dtype=DT)

    def ic_gen(
        acq_function: Any, bounds: Any, q: int, num_restarts: int, raw_samples: int, **kw: Any
    ) -> Any:
        return ics[:num_restarts].clone()

    orig_call = bt_stopping.ExpMAStoppingCriterion.__call__

    def cloning_call(self: Any, fvals: Any) -> bool:
        return orig_call(self, fvals.clone())

    for options in lbfgsb_modes():
        for cyclic_options in ({"maxiter": 3}, {}, {"minimize": False, "maxiter": 30}):
            tag = f"{'serial' if options else 'parallel'} {cyclic_options}"
            # BoTorch's criterion keeps a view of the values that
            # `optimize_acqf_cyclic` then changes in place, which corrupts the
            # first window entry; the reference copies the values (matters
            # with `minimize=False`, where the decision depends on it).
            with patched(bt_stopping.ExpMAStoppingCriterion, "__call__", cloning_call):
                bt = bto.optimize_acqf_cyclic(
                    q_bt,
                    p["bounds"],
                    3,
                    4,
                    raw_samples=16,
                    options=options,
                    ic_generator=ic_gen,
                    cyclic_options=cyclic_options,
                )
            rs = rso.optimize_acqf_cyclic(
                q_rs,
                p["bounds"],
                3,
                4,
                raw_samples=16,
                options=options,
                ic_generator=ic_gen,
                cyclic_options=cyclic_options,
            )
            check_pair(f"cyclic [{tag}]", rs, bt)


def test_homotopy_schedules() -> None:
    for start, end, n in ((1.0, 0.1, 5), (0.5, 2.0, 7), (3.0, 3.0, 1), (1e-3, 1.0, 13)):
        assert (
            rs_homotopy.LinearHomotopySchedule(start, end, n)._values
            == bt_homotopy.LinearHomotopySchedule(start, end, n)._values
        )
        CHECK.check(
            "LogLinearHomotopySchedule values",
            rs_homotopy.LogLinearHomotopySchedule(start, end, n)._values,
            bt_homotopy.LogLinearHomotopySchedule(start, end, n)._values,
            1e-15,
        )


@pytest.mark.parametrize("schedule", ["linear", "loglinear", "fixed"])
def test_optimize_acqf_homotopy(schedule: str) -> None:
    center = torch.tensor([0.31, 0.62, 0.47], dtype=DT)
    bounds = torch.stack([torch.zeros(3, dtype=DT), torch.ones(3, dtype=DT)])
    ics = torch.rand(6, 1, 3, generator=_gen(20), dtype=DT)

    def run(lib_h: Any, lib_oh: Any, prune: float = 1e-3) -> Any:
        acqf = SharpPeak(center)
        make = {
            "linear": lambda: lib_h.LinearHomotopySchedule(1.0, 0.05, 4),
            "loglinear": lambda: lib_h.LogLinearHomotopySchedule(1.0, 0.05, 4),
            "fixed": lambda: lib_h.FixedHomotopySchedule([0.5, 0.2, 0.1]),
        }[schedule]
        calls = []
        homotopy = lib_h.Homotopy(
            [lib_h.HomotopyParameter(parameter=acqf.s, schedule=make())],
            callbacks=[lambda: calls.append(float(acqf.s))],
        )
        out = lib_oh.optimize_acqf_homotopy(
            acqf,
            bounds,
            q=1,
            num_restarts=6,
            homotopy=homotopy,
            prune_tolerance=prune,
            batch_initial_conditions=ics,
            options=dict(JOINT),
            final_options=dict(JOINT),
        )
        return out, calls, float(acqf.s)

    (xb, vb), calls_b, s_b = run(bt_homotopy, bt_oh)
    (xr, vr), calls_r, s_r = run(rs_homotopy, rs_oh)
    check_pair(f"homotopy [{schedule}]", (xr, vr), (xb, vb))
    assert calls_r == calls_b and s_r == s_b == 1.0


def test_prune_candidates() -> None:
    c = torch.rand(30, 2, generator=_gen(21), dtype=DT)
    c[5] = c[0] + 1e-5
    v = torch.rand(30, generator=_gen(22), dtype=DT)
    for tol in (0.0, 1e-4, 0.05, 0.3):
        CHECK.check(
            "prune_candidates",
            rs_oh.prune_candidates(c, v, tol),
            bt_oh.prune_candidates(c, v, tol),
            0.0,
        )


# ---------------------------------------------------------------------------
# Candidate generators and optimize_acqf options
# ---------------------------------------------------------------------------


def test_gen_candidates_torch(p: dict[str, Any]) -> None:
    d = p["d"]
    (a_bt, a_rs), (q_bt, q_rs) = p["logei"], p["qlogei"]
    bounds = p["bounds"]
    ics = torch.rand(4, 1, d, generator=_gen(23), dtype=DT)
    for options in (
        None,
        {
            "optimizer_options": {"lr": 0.05, "betas": (0.8, 0.99)},
            "stopping_criterion_options": {"maxiter": 60, "rel_tol": 1e-6, "n_window": 5},
        },
        {
            "optimizer_options": {"lr": 0.01, "weight_decay": 0.1, "amsgrad": True},
            "stopping_criterion_options": {"maxiter": 80},
        },
    ):
        steps_b: list[Any] = []
        steps_r: list[Any] = []
        bt = bt_gen.gen_candidates_torch(
            ics,
            a_bt,
            bounds[0],
            bounds[1],
            options=options,
            callback=lambda i, loss, g, s=steps_b: s.append((i, float(loss))),
        )
        rs = rs_gen.gen_candidates_torch(
            ics,
            a_rs,
            bounds[0],
            bounds[1],
            options=options,
            callback=lambda i, loss, g, s=steps_r: s.append((i, float(loss))),
        )
        check_pair(
            f"gen_candidates_torch {options and sorted(options['optimizer_options'])}", rs, bt
        )
        assert len(steps_r) == len(steps_b)
        CHECK.check(
            "gen_candidates_torch losses",
            [s[1] for s in steps_r],
            [s[1] for s in steps_b],
            VALUE_TOL,
        )
    # q > 1 candidate sets.
    ics2 = torch.rand(3, 2, d, generator=_gen(24), dtype=DT)
    check_pair(
        "gen_candidates_torch q=2",
        rs_gen.gen_candidates_torch(ics2, q_rs, bounds[0], bounds[1]),
        bt_gen.gen_candidates_torch(ics2, q_bt, bounds[0], bounds[1]),
    )
    # Through optimize_acqf (chunks of batch_limit restarts stop together).
    for options in (None, {"batch_limit": 2}):
        check_pair(
            f"optimize_acqf gen_candidates_torch {options}",
            rso.optimize_acqf(
                a_rs,
                bounds,
                1,
                4,
                batch_initial_conditions=ics,
                gen_candidates=rs_gen.gen_candidates_torch,
                options=options,
                return_best_only=False,
            ),
            bto.optimize_acqf(
                a_bt,
                bounds,
                1,
                4,
                batch_initial_conditions=ics,
                gen_candidates=bt_gen.gen_candidates_torch,
                options=options,
                return_best_only=False,
            ),
        )


def test_gen_candidates_torch_fixed_features(p: dict[str, Any]) -> None:
    """BoTorch fails with fixed features and bounds (it clamps the free
    features with the bounds of all features); the reference is BoTorch on
    the free features with the fixed ones inserted by the acquisition
    function."""
    d = p["d"]
    (a_bt, a_rs) = p["logei"]
    bounds = p["bounds"]
    ics = torch.rand(4, 1, d, generator=_gen(25), dtype=DT)
    with pytest.raises(RuntimeError):
        bt_gen.gen_candidates_torch(ics, a_bt, bounds[0], bounds[1], fixed_features={1: 0.4})
    free = [i for i in range(d) if i != 1]

    def reduced(X: Any) -> Any:
        full = torch.cat([X[..., :1], torch.full_like(X[..., :1], 0.4), X[..., 1:]], -1)
        return a_bt(full)

    cb, vb = bt_gen.gen_candidates_torch(ics[..., free], reduced, bounds[0][free], bounds[1][free])
    cr, vr = rs_gen.gen_candidates_torch(ics, a_rs, bounds[0], bounds[1], fixed_features={1: 0.4})
    CHECK.check("gen_candidates_torch fixed features (reference)", cr[..., free], cb, XTOL)
    assert bool((cr[..., 1] == 0.4).all())
    CHECK.check("gen_candidates_torch fixed features values (reference)", vr, vb, VALUE_TOL)


def test_gen_candidates_scipy(p: dict[str, Any]) -> None:
    d = p["d"]
    (a_bt, a_rs) = p["logei"]
    bounds = p["bounds"]
    ics = torch.rand(4, 1, d, generator=_gen(26), dtype=DT)
    cases: list[dict[str, Any]] = [
        {"use_parallel_mode": False},
        {"options": {"maxfun": 300}},
        {"options": {"maxfun": 300, "max_optimization_problem_aggregation_size": 2}},
        {"use_parallel_mode": False, "fixed_features": {0: 0.25}},
    ]
    if botorch_parallel_lbfgsb():
        cases += [
            {},
            {"options": {"maxiter": 20}},
            {"fixed_features": {0: 0.25}},
            {"options": {"tol": 1e-8}},
        ]
    for kwargs in cases:
        bt = bt_gen.gen_candidates_scipy(ics, a_bt, bounds[0], bounds[1], **kwargs)
        rs = rs_gen.gen_candidates_scipy(ics, a_rs, bounds[0], bounds[1], **kwargs)
        check_pair(f"gen_candidates_scipy {sorted(kwargs)}", rs, bt)
    vals = torch.randn(5, generator=_gen(27), dtype=DT)
    cands = torch.rand(5, 2, 3, generator=_gen(28), dtype=DT)
    CHECK.check(
        "get_best_candidates",
        rs_gen.get_best_candidates(cands, vals),
        bt_gen.get_best_candidates(cands, vals),
        0.0,
    )


def test_optimize_acqf_options(p: dict[str, Any]) -> None:
    d = p["d"]
    a_bt, a_rs = p["logei"]
    bounds = p["bounds"]
    ics = torch.rand(5, 1, d, generator=_gen(29), dtype=DT)
    pp = lambda X: (X * 4).round() / 4  # noqa: E731
    for options in lbfgsb_modes():
        tag = "serial" if options else "parallel"
        kw = dict(batch_initial_conditions=ics, options=options)
        for extra in (
            {"post_processing_func": pp},
            {"post_processing_func": pp, "return_best_only": False},
            {"timeout_sec": 1e3},
        ):
            check_pair(
                f"optimize_acqf {sorted(extra)} [{tag}]",
                rso.optimize_acqf(a_rs, bounds, 1, 5, **kw, **extra),
                bto.optimize_acqf(a_bt, bounds, 1, 5, **kw, **extra),
            )
    # The serial path's aggregation option.
    opts = {"maxfun": 15000, "max_optimization_problem_aggregation_size": 2}
    check_pair(
        "optimize_acqf aggregation size 2",
        rso.optimize_acqf(
            a_rs, bounds, 1, 5, batch_initial_conditions=ics, options=opts, return_best_only=False
        ),
        bto.optimize_acqf(
            a_bt, bounds, 1, 5, batch_initial_conditions=ics, options=opts, return_best_only=False
        ),
    )


# ---------------------------------------------------------------------------
# Initializers
# ---------------------------------------------------------------------------


def test_initialize_q_batch() -> None:
    g = _gen(30)
    X = torch.rand(300, 2, 3, generator=g, dtype=DT)
    vals = torch.randn(300, generator=g, dtype=DT)
    for n, eta in ((10, 1.0), (25, 3.0), (300, 1.0)):
        torch.manual_seed(n)
        bt = bt_init.initialize_q_batch(X, vals, n, eta=eta)
        torch.manual_seed(n)
        rs = rs_init.initialize_q_batch(X, vals, n, eta=eta)
        check_pair(f"initialize_q_batch n={n}", rs, bt, 0.0)
    pos = vals.clamp_min(0.0)
    sparse = torch.where(vals > 1.5, vals, torch.zeros_like(vals))
    for name, v in (
        ("nonneg", pos),
        ("nonneg sparse", sparse),
        ("nonneg zeros", torch.zeros_like(vals)),
    ):
        torch.manual_seed(1)
        bt = bt_init.initialize_q_batch_nonneg(X, v, 20, alpha=1e-2)
        torch.manual_seed(1)
        rs = rs_init.initialize_q_batch_nonneg(X, v, 20, alpha=1e-2)
        check_pair(f"initialize_q_batch_{name}", rs, bt, 0.0)
    for largest, srt in ((True, True), (False, True), (True, False)):
        check_pair(
            f"initialize_q_batch_topn largest={largest} sorted={srt}",
            rs_init.initialize_q_batch_topn(X, vals, 15, largest=largest, sorted=srt),
            bt_init.initialize_q_batch_topn(X, vals, 15, largest=largest, sorted=srt),
            0.0,
        )
    torch.manual_seed(2)
    bt = bt_init.initialize_q_batch(X, torch.ones(300, dtype=DT), 5)
    torch.manual_seed(2)
    rs = rs_init.initialize_q_batch(X, torch.ones(300, dtype=DT), 5)
    check_pair("initialize_q_batch constant values (random)", rs, bt, 0.0)


def test_truncated_normal_perturbations() -> None:
    """The deterministic part with BoTorch's Sobol draws replaced by the same
    uniform points."""
    import botorch.utils.sampling as bus

    from botorch_rs import _botorch_rs as _rs

    g = _gen(31)
    bounds = torch.tensor([[-1.0, 0.0, 2.0], [1.0, 3.0, 2.5]], dtype=DT)
    center = bounds[0] + (bounds[1] - bounds[0]) * torch.rand(3, generator=g, dtype=DT)
    center[0] = bounds[0, 0]  # at the boundary
    u = torch.rand(8, 3, generator=g, dtype=DT)
    for sigma in (1e-3, 0.05, 2.0):
        with patched(bus, "draw_sobol_samples", lambda bounds, n, q, **kw: u.unsqueeze(1)):
            bt = bus.sample_truncated_normal_perturbations(center.unsqueeze(0), 8, sigma, bounds)
        rs = _rs.truncated_normal_perturbations_py(
            (center.repeat(8).tolist(), [8, 3]),
            (u.reshape(-1).tolist(), [8, 3]),
            sigma,
            (bounds.reshape(-1).tolist(), [2, 3]),
        )
        CHECK.check(
            "truncated normal perturbations", torch.tensor(rs[0], dtype=DT).reshape(8, 3), bt, 1e-12
        )


def test_sample_points_around_best_best_points(p: dict[str, Any]) -> None:
    """The best points found by ``sample_points_around_best``: BoTorch's
    selection of the best 5% (at least 1) of the baseline points."""
    (a_bt, a_rs) = p["logei"]
    best = rs_init._best_points(a_rs, p["d"], 25.0)
    assert best is not None
    X = p["X"]
    with torch.no_grad():
        f = p["bt"].posterior(X).mean.squeeze(-1)
    n_best = max(1, round(X.shape[0] * 25.0 / 100))
    expected = X[torch.topk(f, n_best).indices]
    CHECK.check(
        "sample_points_around_best best points",
        torch.tensor(best[0], dtype=DT).reshape(n_best, -1),
        expected,
        0.0,
    )


# ---------------------------------------------------------------------------
# Sampling strategies
# ---------------------------------------------------------------------------


def test_sampling_strategies(p: dict[str, Any]) -> None:
    bt, rs = p["bt"], p["rs"]
    d = p["d"]
    g = _gen(32)
    Xc = torch.rand(120, d, generator=g, dtype=DT)
    Xb = torch.rand(2, 40, d, generator=g, dtype=DT)
    for X, label in ((Xc, "unbatched"), (Xb, "batched")):
        for repl in (True, False):
            if X.ndim > 2 and not repl:
                continue
            torch.manual_seed(3)
            s_bt = bt_sampling.MaxPosteriorSampling(bt, replacement=repl)(X, num_samples=6)
            torch.manual_seed(3)
            s_rs = rs_sampling.MaxPosteriorSampling(rs, replacement=repl)(X, num_samples=6)
            CHECK.check(f"MaxPosteriorSampling {label} replacement={repl}", s_rs, s_bt, 0.0)
    for repl in (True, False):
        torch.manual_seed(4)
        s_bt = bt_sampling.BoltzmannSampling(p["logei"][0], eta=2.0, replacement=repl)(
            Xc, num_samples=7
        )
        torch.manual_seed(4)
        s_rs = rs_sampling.BoltzmannSampling(p["logei"][1], eta=2.0, replacement=repl)(
            Xc, num_samples=7
        )
        CHECK.check(f"BoltzmannSampling replacement={repl}", s_rs, s_bt, 0.0)
    c_bt, c_rs, _ = model_pair(ModelSpec(n=12, d=d), 41)
    c2_bt, c2_rs, _ = model_pair(ModelSpec(n=10, d=d), 43)
    for cm_bt, cm_rs in (
        (BtModelListGP(c_bt), RsModelListGP(c_rs)),
        (BtModelListGP(c_bt, c2_bt), RsModelListGP(c_rs, c2_rs)),
    ):
        torch.manual_seed(5)
        s_bt = bt_sampling.ConstrainedMaxPosteriorSampling(bt, cm_bt)(Xc, num_samples=4)
        torch.manual_seed(5)
        s_rs = rs_sampling.ConstrainedMaxPosteriorSampling(rs, cm_rs)(Xc, num_samples=4)
        CHECK.check("ConstrainedMaxPosteriorSampling", s_rs, s_bt, 0.0)
    obj = torch.randn(5, 30, 1, generator=g, dtype=DT)
    X30 = torch.rand(30, d, generator=g, dtype=DT)
    for repl in (True, False):
        CHECK.check(
            f"maximize_samples replacement={repl}",
            rs_sampling.MaxPosteriorSampling(rs, replacement=repl).maximize_samples(X30, obj, 5),
            bt_sampling.MaxPosteriorSampling(bt, replacement=repl).maximize_samples(X30, obj, 5),
            0.0,
        )


def test_boltzmann_sampling_batched_reference(p: dict[str, Any]) -> None:
    """BoTorch standardizes batched acquisition values over the batch
    dimension (uniform sampling for one batch); botorch_rs standardizes over
    the points, like the unbatched case. Reference: the unbatched sampling
    of every batch."""
    X = torch.rand(1, 50, p["d"], generator=_gen(33), dtype=DT)
    torch.manual_seed(6)
    batched = rs_sampling.BoltzmannSampling(p["logei"][1], eta=3.0)(X, num_samples=8)
    torch.manual_seed(6)
    reference = bt_sampling.BoltzmannSampling(p["logei"][0], eta=3.0)(X[0], num_samples=8)
    CHECK.check("BoltzmannSampling batched (reference)", batched[0], reference, 0.0)


# ---------------------------------------------------------------------------
# Fitting
# ---------------------------------------------------------------------------


def _fit_pair(spec: ModelSpec, seed: int) -> tuple[Any, Any, Any, Any]:
    bt, rs, _ = model_pair(spec, seed)
    bt.train()
    mb, mr = BtMLL(bt.likelihood, bt), RsMLL(rs.likelihood, rs)
    mb.train()
    mr.train()
    return bt, rs, mb, mr


def _params(model: Any) -> Any:
    return torch.cat(
        [torch.as_tensor(v, dtype=DT).reshape(-1) for _, v in model.named_parameters()]
    )


@pytest.mark.parametrize("m", [1, 2])
def test_fit_gpytorch_mll_torch(m: int) -> None:
    spec = ModelSpec(n=14, d=2, m=m)
    for kwargs in (
        {"step_limit": 150},
        {"step_limit": 120, "optimizer": lambda params: torch.optim.Adam(params, lr=0.03)},
        {"step_limit": 100, "stopping_criterion": None},
        {
            "step_limit": 100,
            "parameters": "subset",
            "bounds": {"model.mean_module.raw_constant": (-0.05, 0.05)},
        },
    ):
        bt, rs, mb, mr = _fit_pair(spec, 50 + m)
        kb, kr = dict(kwargs), dict(kwargs)
        if kwargs.get("parameters") == "subset":
            names = ["model.mean_module.raw_constant", "model.covar_module.raw_lengthscale"]
            kb["parameters"] = {k: v for k, v in mb.named_parameters() if k in names}
            from botorch_rs.optim.utils import get_parameters

            kr["parameters"] = {k: v for k, v in get_parameters(mr).items() if k in names}
        if "optimizer" in kwargs:
            import functools

            kr["optimizer"] = functools.partial(torch.optim.Adam, lr=0.03)
        rb = bt_fit.fit_gpytorch_mll_torch(mb, **kb)
        rr = rs_fit.fit_gpytorch_mll_torch(mr, **kr)
        label = f"fit_gpytorch_mll_torch m={m} {sorted(kwargs)}"
        assert rr.step == rb.step and rr.status.name == rb.status.name
        CHECK.check(f"{label}: raw parameters", _params(rs), _params(bt), VALUE_TOL)
        CHECK.check(f"{label}: loss", rr.fval, rb.fval, VALUE_TOL)


def test_fit_gpytorch_mll_torch_until_stopping() -> None:
    """Run until the stopping criterion holds (Adam amplifies round-off over
    hundreds of steps): botorch_rs's step count and loss lie within BoTorch's
    outcomes on 16 twins whose training inputs are perturbed at 1e-14 ..
    1e-11."""
    spec = ModelSpec(n=14, d=2)
    _, rs, _, mr = _fit_pair(spec, 60)
    rr = rs_fit.fit_gpytorch_mll_torch(mr, step_limit=4000)
    outcomes = []
    for k in range(16):
        bt, _, _, _ = _fit_pair(spec, 60)
        X0 = bt.train_inputs[0]
        rel = 10 ** (-14 + 3 * k / 15) if k else 0.0
        bt.train_inputs = (X0 * (1 + rel * torch.randn(X0.shape, generator=_gen(k), dtype=DT)),)
        bt.prediction_strategy = None
        mb = BtMLL(bt.likelihood, bt)
        mb.train()
        r = bt_fit.fit_gpytorch_mll_torch(mb, step_limit=4000)
        outcomes.append((r.step, r.fval))
    steps = [s for s, _ in outcomes]
    fvals = [f for _, f in outcomes]
    assert min(steps) <= rr.step <= max(steps), (rr.step, steps)
    excess = max(min(fvals) - rr.fval, rr.fval - max(fvals), 0.0) / max(abs(rr.fval), 1.0)
    CHECK.check_bound(
        "fit_gpytorch_mll_torch until stopping: loss outside twins", excess, VALUE_TOL
    )


@pytest.mark.parametrize("m", [1, 2])
def test_fit_gpytorch_mll_scipy(m: int) -> None:
    spec = ModelSpec(n=14, d=3, m=m)
    for kwargs in (
        {},
        {"options": {"maxiter": 15}},
        {"bounds": {"model.covar_module.raw_lengthscale": (0.1, 2.0)}},
    ):
        bt, rs, mb, mr = _fit_pair(spec, 70 + m)
        rb = bt_fit.fit_gpytorch_mll_scipy(mb, **kwargs)
        rr = rs_fit.fit_gpytorch_mll_scipy(mr, **kwargs)
        label = f"fit_gpytorch_mll_scipy m={m} {sorted(kwargs)}"
        expected = rb.status.name
        if m > 1 and "options" in kwargs:
            # BoTorch's batched fit looks up a "warnflag" key that its batched
            # L-BFGS-B results do not have (they have "status"), so elements
            # that reach the iteration limit give FAILURE instead of STOPPED.
            assert rb.status.name == "FAILURE"
            expected = "STOPPED"
        assert rr.step == rb.step and rr.status.name == expected, (rr, rb)
        CHECK.check(f"{label}: raw parameters", _params(rs), _params(bt), 1e-8)
        CHECK.check(f"{label}: loss", rr.fval, rb.fval, VALUE_TOL)


def test_stopping_criterion_and_closures() -> None:
    seq = (1.0 + torch.exp(-torch.arange(60, dtype=DT) / 7.0)).tolist()
    for kw in (
        {},
        {"minimize": False},
        {"maxiter": 20},
        {"n_window": 4, "rel_tol": 1e-3},
        {"eta": 0.5},
    ):
        cb, cr = bt_stopping.ExpMAStoppingCriterion(**kw), rs_stopping.ExpMAStoppingCriterion(**kw)
        assert [cr(v) for v in seq] == [bool(cb(torch.tensor(v, dtype=DT))) for v in seq], kw
        cr.reset()
        cb.reset()
        assert cr.iter == cb.iter == 0
    CHECK.check(
        "ExpMAStoppingCriterion default weights (bitwise)",
        rs_stopping.ExpMAStoppingCriterion().weights,
        bt_stopping.ExpMAStoppingCriterion().weights.double(),
        0.0,
    )
    # Closures and parameter helpers.
    from botorch.optim.closures import NdarrayOptimizationClosure as BtNd
    from botorch.optim.closures import get_loss_closure_with_grads as bt_glc
    from botorch.optim.utils import get_parameters_and_bounds as bt_pb

    from botorch_rs.optim.closures import NdarrayOptimizationClosure as RsNd
    from botorch_rs.optim.closures import get_loss_closure_with_grads as rs_glc
    from botorch_rs.optim.utils import get_parameters_and_bounds as rs_pb

    bt, rs, mb, mr = _fit_pair(ModelSpec(n=10, d=2), 80)
    pb, bb = bt_pb(mb)
    pr, br = rs_pb(mr)
    assert list(pr) == list(pb)
    assert {k: tuple(float(x) for x in v) for k, v in br.items()} == {
        k: tuple(float(x) for x in v) for k, v in bb.items()
    }
    nb = BtNd(bt_glc(mb, pb), pb)
    nr = RsNd(rs_glc(mr, pr), pr)
    CHECK.check("NdarrayOptimizationClosure state", nr.state, nb.state, 0.0)
    state = nb.state + 0.1
    vb, gb = nb(state)
    vr, gr = nr(state)
    CHECK.check("NdarrayOptimizationClosure value", float(vr), float(vb), VALUE_TOL)
    CHECK.check("NdarrayOptimizationClosure gradient", gr, gb, 1e-8)
    CHECK.check("NdarrayOptimizationClosure state after set", nr.state, nb.state, 0.0)


# ---------------------------------------------------------------------------
# Cross-validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("variant", ["plain", "yvar", "normalize", "two_outputs"])
def test_cross_validation(variant: str) -> None:
    from botorch.models.transforms.input import Normalize as BtNormalize

    from botorch_rs.models.transforms import Normalize as RsNormalize

    g = _gen(90)
    x = 2.0 + 3.0 * torch.rand(9, 2, generator=g, dtype=DT)
    y = torch.sin(x.sum(-1, keepdim=True)) + 0.05 * torch.randn(9, 1, generator=g, dtype=DT)
    yvar = None
    kw_b: dict[str, Any] = {}
    kw_r: dict[str, Any] = {}
    if variant == "yvar":
        yvar = 0.01 + 0.01 * torch.rand(9, 1, generator=g, dtype=DT)
    if variant == "normalize":
        kw_b["model_init_kwargs"] = {"input_transform": BtNormalize(d=2)}
        kw_r["model_init_kwargs"] = {"input_transform": RsNormalize(d=2)}
    if variant == "two_outputs":
        y = torch.cat([y, torch.cos(x[:, :1])], -1)
    fb, fr = bt_folds(x, y, yvar), rs_folds(x, y, yvar)
    for name in ("train_X", "test_X", "train_Y", "test_Y", "train_Yvar", "test_Yvar"):
        a, e = getattr(fr, name), getattr(fb, name)
        assert (a is None) == (e is None)
        if e is not None:
            CHECK.check(f"gen_loo_cv_folds {name}", a, e, 0.0)
    cb = bt_bcv(BtSingleTaskGP, BtMLL, fb, **kw_b)
    cr = rs_bcv(RsSingleTaskGP, RsMLL, fr, **kw_r)
    CHECK.check(
        f"batch_cross_validation [{variant}]: posterior mean",
        cr.posterior.mean,
        cb.posterior.mean,
        VALUE_TOL,
    )
    CHECK.check(
        f"batch_cross_validation [{variant}]: posterior variance",
        cr.posterior.variance,
        cb.posterior.variance,
        VALUE_TOL,
    )
    CHECK.check(
        f"batch_cross_validation [{variant}]: observed_Y", cr.observed_Y, cb.observed_Y, 0.0
    )


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


def _same_error(f_bt: Any, f_rs: Any) -> None:
    with pytest.raises(Exception) as e_bt:
        f_bt()
    with pytest.raises(Exception) as e_rs:
        f_rs()
    b, r = e_bt.value, e_rs.value
    builtin = [c for c in type(b).__mro__ if c.__module__ == "builtins" and c is not object]
    assert isinstance(r, tuple(builtin)), (type(b), type(r))
    assert type(r).__name__ == type(b).__name__ or type(b).__name__ in (
        "RuntimeError",
        "ValueError",
        "NotImplementedError",
        "TypeError",
    ), (b, r)
    assert str(b)[:40] == str(r)[:40] or type(b).__name__ == "TypeError", (str(b), str(r))


def test_errors_match_botorch(p: dict[str, Any]) -> None:
    d = p["d"]
    (a_bt, a_rs), (q_bt, q_rs) = p["logei"], p["qlogei"]
    bounds = p["bounds"]
    choices = torch.rand(5, d, generator=_gen(91), dtype=DT)
    _same_error(
        lambda: bto.optimize_acqf_discrete(q_bt, 2, choices[:0]),
        lambda: rso.optimize_acqf_discrete(q_rs, 2, choices[:0]),
    )
    _same_error(
        lambda: bto.optimize_acqf_discrete(q_bt, 2, choices, X_avoid=choices),
        lambda: rso.optimize_acqf_discrete(q_rs, 2, choices, X_avoid=choices),
    )
    _same_error(
        lambda: bto.optimize_acqf_mixed(a_bt, bounds, 1, 2, [], raw_samples=8),
        lambda: rso.optimize_acqf_mixed(a_rs, bounds, 1, 2, [], raw_samples=8),
    )
    _same_error(
        lambda: bto.optimize_acqf_mixed(
            q_bt, bounds, 2, 2, [{0: 0.0}], raw_samples=8, return_best_only=False
        ),
        lambda: rso.optimize_acqf_mixed(
            q_rs, bounds, 2, 2, [{0: 0.0}], raw_samples=8, return_best_only=False
        ),
    )
    inter = [(torch.tensor([[0, 0], [1, 0]]), torch.tensor([1.0, 1.0], dtype=DT), 0.0)]
    _same_error(
        lambda: bto.optimize_acqf_mixed(
            a_bt, bounds, 1, 2, [{0: 0.0}], raw_samples=8, inequality_constraints=inter
        ),
        lambda: rso.optimize_acqf_mixed(
            a_rs, bounds, 1, 2, [{0: 0.0}], raw_samples=8, inequality_constraints=inter
        ),
    )
    dc = [torch.tensor([0.0, 1.0], dtype=DT)] * d
    _same_error(
        lambda: bto.optimize_acqf_discrete_local_search(
            a_bt, dc, 1, batch_initial_conditions=torch.zeros(2, d, dtype=DT)
        ),
        lambda: rso.optimize_acqf_discrete_local_search(
            a_rs, dc, 1, batch_initial_conditions=torch.zeros(2, d, dtype=DT)
        ),
    )
    kw = dict(discrete_dims={0: [0.0, 0.5, 1.0]}, raw_samples=8, num_restarts=2)
    _same_error(
        lambda: btm.optimize_acqf_mixed_alternating(a_bt, bounds, discrete_dims={0: [0.1, 1.0]}),
        lambda: rsm.optimize_acqf_mixed_alternating(a_rs, bounds, discrete_dims={0: [0.1, 1.0]}),
    )
    _same_error(
        lambda: btm.optimize_acqf_mixed_alternating(a_bt, bounds, discrete_dims={0: [0.0, 0.9]}),
        lambda: rsm.optimize_acqf_mixed_alternating(a_rs, bounds, discrete_dims={0: [0.0, 0.9]}),
    )
    _same_error(
        lambda: btm.optimize_acqf_mixed_alternating(a_bt, bounds, options={"foo": 1}, **kw),
        lambda: rsm.optimize_acqf_mixed_alternating(a_rs, bounds, options={"foo": 1}, **kw),
    )
    _same_error(
        lambda: btm.optimize_acqf_mixed_alternating(
            a_bt, bounds, options={"max_optimization_problem_aggregation_size": 2}, **kw
        ),
        lambda: rsm.optimize_acqf_mixed_alternating(
            a_rs, bounds, options={"max_optimization_problem_aggregation_size": 2}, **kw
        ),
    )
    eq = [(torch.tensor([0, 1]), torch.tensor([1.0, 1.0], dtype=DT), 1.0)]
    _same_error(
        lambda: btm.optimize_acqf_mixed_alternating(a_bt, bounds, equality_constraints=eq, **kw),
        lambda: rsm.optimize_acqf_mixed_alternating(a_rs, bounds, equality_constraints=eq, **kw),
    )
    _same_error(
        lambda: btm.optimize_acqf_mixed_alternating(a_bt, bounds, sequential=False, **kw),
        lambda: rsm.optimize_acqf_mixed_alternating(a_rs, bounds, sequential=False, **kw),
    )
    _same_error(
        lambda: bt_oh.prune_candidates(choices, torch.ones(4, dtype=DT), 0.1),
        lambda: rs_oh.prune_candidates(choices, torch.ones(4, dtype=DT), 0.1),
    )
    _same_error(
        lambda: bt_oh.prune_candidates(choices, torch.ones(5, dtype=DT), -1.0),
        lambda: rs_oh.prune_candidates(choices, torch.ones(5, dtype=DT), -1.0),
    )
    _same_error(
        lambda: bt_sampling.ConstrainedMaxPosteriorSampling(
            p["bt"], BtModelListGP(p["bt"]), objective=lambda Y, X=None: Y
        ),
        lambda: rs_sampling.ConstrainedMaxPosteriorSampling(
            p["rs"], RsModelListGP(p["rs"]), objective=lambda Y, X=None: Y
        ),
    )
    X = torch.rand(2, 2, 5, d, generator=_gen(92), dtype=DT)
    _same_error(
        lambda: bt_sampling.MaxPosteriorSampling(p["bt"], replacement=False)(X, 2),
        lambda: rs_sampling.MaxPosteriorSampling(p["rs"], replacement=False)(X, 2),
    )
    _same_error(
        lambda: bt_init.initialize_q_batch(torch.rand(3, 1, d), torch.rand(3), 5),
        lambda: rs_init.initialize_q_batch(torch.rand(3, 1, d), torch.rand(3), 5),
    )
    _same_error(
        lambda: bto.optimize_acqf_cyclic(q_bt, bounds, 2, 2),
        lambda: rso.optimize_acqf_cyclic(q_rs, bounds, 2, 2),
    )
    _same_error(
        lambda: bt_fit.fit_gpytorch_mll_torch(
            BtMLL(p["bt"].likelihood, p["bt"]), stopping_criterion=None
        ),
        lambda: rs_fit.fit_gpytorch_mll_torch(
            RsMLL(p["rs"].likelihood, p["rs"]), stopping_criterion=None
        ),
    )
    _same_error(
        lambda: bt_stopping.ExpMAStoppingCriterion(foo=1),
        lambda: rs_stopping.ExpMAStoppingCriterion(foo=1),
    )  # type: ignore[call-arg]
    _same_error(
        lambda: bt_gen.gen_candidates_torch(
            torch.rand(2, 1, d, dtype=DT), a_bt, options={"stopping_criterion_options": {"foo": 1}}
        ),
        lambda: rs_gen.gen_candidates_torch(
            torch.rand(2, 1, d, dtype=DT), a_rs, options={"stopping_criterion_options": {"foo": 1}}
        ),
    )
    assert math.isfinite(1.0)
    _ = copy


# ---------------------------------------------------------------------------
# One-shot acquisition functions (phase 8) with the options of this phase
# ---------------------------------------------------------------------------


def _kg_pair(seed: int, n_f: int = 4) -> tuple[Any, Any]:
    """A ``qKnowledgeGradient`` pair with the same fantasy base samples."""
    from botorch.acquisition import qKnowledgeGradient as BtKG
    from botorch.sampling import SobolQMCNormalSampler as BtSobol

    from botorch_rs.acquisition import qKnowledgeGradient as RsKG
    from botorch_rs.sampling import SobolQMCNormalSampler as RsSobol

    bt, rs, _ = model_pair(ModelSpec(n=10, d=2), seed)
    base = torch.randn(n_f, 1, 1, dtype=DT, generator=_gen(seed + 1))
    sb, sr = BtSobol(torch.Size([n_f])), RsSobol(torch.Size([n_f]))
    sb.base_samples = base.clone()
    sr.base_samples = base.clone()
    return BtKG(bt, num_fantasies=n_f, sampler=sb), RsKG(rs, num_fantasies=n_f, sampler=sr)


@pytest.mark.parametrize("variant", ["post_processing", "timeout", "torch", "full_tree"])
def test_knowledge_gradient_with_phase9_options(variant: str) -> None:
    """``optimize_acqf`` with a ``qKnowledgeGradient`` and the options of this
    phase from identical augmented initial conditions: BoTorch post-processes
    and re-evaluates the full trees (``b x (q + num_fantasies) x d``) before
    choosing the best restart and extracting the candidates. The L-BFGS-B
    variants run on both of BoTorch's paths where it has them
    (``lbfgsb_modes``; 0.18.1 with SciPy 1.18 has the serial path only)."""
    a_bt, a_rs = _kg_pair(31)
    bounds = torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=DT)
    ics = torch.rand(3, 5, 2, dtype=DT, generator=_gen(32))
    calls: list[tuple[int, ...]] = []

    def post(X: Any) -> Any:
        calls.append(tuple(X.shape))
        return X.clamp(0.1, 0.9)

    kb: dict[str, Any] = {}
    kr: dict[str, Any] = {}
    modes: list[dict[str, Any]] = lbfgsb_modes()
    if variant in ("post_processing", "full_tree"):
        kb["post_processing_func"] = kr["post_processing_func"] = post
    if variant == "full_tree":
        kb["return_full_tree"] = kr["return_full_tree"] = True
    if variant == "timeout":
        # Large enough not to stop; selects the serial L-BFGS-B path.
        kb["timeout_sec"] = kr["timeout_sec"] = 100.0
        modes = [{}]
    if variant == "torch":
        kb["gen_candidates"], kr["gen_candidates"] = (
            bt_gen.gen_candidates_torch,
            rs_gen.gen_candidates_torch,
        )
        modes = [{}]
    for options in modes:
        tag = f"{variant} [{'serial' if options else 'default'}]"
        calls.clear()
        bt = bto.optimize_acqf(
            a_bt, bounds, 1, 3, batch_initial_conditions=ics, options=dict(options), **kb
        )
        n_bt = len(calls)
        rs = rso.optimize_acqf(
            a_rs, bounds, 1, 3, batch_initial_conditions=ics, options=dict(options), **kr
        )
        assert tuple(rs[0].shape) == tuple(bt[0].shape)
        assert calls[n_bt:] == calls[:n_bt]
        if variant in ("post_processing", "full_tree"):
            assert calls[0] == (3, 5, 2)
        check_pair(f"qKnowledgeGradient {tag}", rs, (bt[0], bt[1].detach()))

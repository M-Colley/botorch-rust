"""The randomized equivalence suite of optimization and candidate generation
(phase 9): botorch_rs against live BoTorch over a seeded grid of models and
configurations.

Every scenario builds a ``SingleTaskGP`` pair (d = 2-5 inputs, n = 8-20
training points, RBF or Matern-5/2, with or without ``Normalize`` and fixed
noise, explicit hyperparameters) and draws, from its seed,

* ``optimize_acqf_discrete`` problems (random choices, ``q = 1-3``,
  ``unique``, ``X_avoid``, a random linear constraint);
* ``optimize_acqf_mixed`` problems (random fixed-feature combinations and
  initial conditions, ``q = 1-2``, both L-BFGS-B paths where BoTorch has
  them);
* ``optimize_acqf_discrete_local_search`` (random lattices; the random lattice
  points replaced by the same seeded points in both libraries);
* ``optimize_acqf_mixed_alternating`` (random discrete and categorical
  dimensions, the starting points replaced by the same points);
* ``gen_candidates_torch`` (random learning rates and stopping settings),
  ``optimize_acqf_cyclic`` (a deterministic initial-condition generator);
* ``MaxPosteriorSampling`` under the same torch seed.

Tolerances: values 1e-9 (relative), candidates of continuous optimizers 1e-7
(see ``optim9_helpers.XTOL``), discrete candidates exact. The worst error of
every quantity is printed in the terminal summary (section "equivalence
suite (optimization and candidate generation)").

Run with ``pytest -m equivalence``; skipped unless BoTorch is importable.
"""

from __future__ import annotations

import random
from typing import Any

import pytest

from helpers import importorskip_botorch

importorskip_botorch()
torch = pytest.importorskip("torch")

import botorch.optim.optimize as bto  # noqa: E402
import botorch.optim.optimize_mixed as btm  # noqa: E402
from botorch.acquisition import LogExpectedImprovement as BtLogEI  # noqa: E402
from botorch.acquisition import qLogExpectedImprovement as BtqLogEI  # noqa: E402
from botorch.generation import gen as bt_gen  # noqa: E402
from botorch.generation import sampling as bt_sampling  # noqa: E402
from botorch.models.deterministic import PosteriorMeanModel as BtPM  # noqa: E402
from botorch.optim import stopping as bt_stopping  # noqa: E402
from botorch.sampling.stochastic_samplers import StochasticSampler as BtSt  # noqa: E402
from optim9_helpers import (  # noqa: E402
    DT,
    VALUE_TOL,
    XTOL,
    botorch_parallel_lbfgsb,
    lattice_points,
    lbfgsb_modes,
    patched,
    starting_points,
)

import botorch_rs.optim.optimize as rso  # noqa: E402
import botorch_rs.optim.optimize_mixed as rsm  # noqa: E402
from botorch_rs.acquisition import LogExpectedImprovement as RsLogEI  # noqa: E402
from botorch_rs.acquisition import qLogExpectedImprovement as RsqLogEI  # noqa: E402
from botorch_rs.generation import gen as rs_gen  # noqa: E402
from botorch_rs.generation import sampling as rs_sampling  # noqa: E402
from botorch_rs.models.deterministic import PosteriorMeanModel as RsPM  # noqa: E402
from botorch_rs.sampling import StochasticSampler as RsSt  # noqa: E402
from parity_helpers import Checker, ModelSpec, model_pair  # noqa: E402

pytestmark = [pytest.mark.equivalence, pytest.mark.filterwarnings("ignore")]

CHECK = Checker("equivalence suite (optimization and candidate generation)")
N_SCENARIOS = 24


@pytest.fixture(autouse=True, scope="module")
def _single_threaded_torch() -> Any:
    n = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(n)


def scenario(index: int) -> dict[str, Any]:
    r = random.Random(9000 + index)
    d = r.randint(2, 5)
    spec = ModelSpec(
        n=r.randint(8, 20),
        d=d,
        kernel=r.choice(["rbf", "matern52"]),
        normalize=r.random() < 0.4,
        fixed_noise=r.random() < 0.3,
    )
    bt, rs, X = model_pair(spec, 9100 + index)
    # A float64 tensor: BoTorch stores Python floats as float32 buffers.
    best_f = torch.tensor(r.uniform(0.5, 2.0), dtype=DT)
    return {
        "r": r,
        "d": d,
        "bt": bt,
        "rs": rs,
        "logei": (BtLogEI(bt, best_f=best_f), RsLogEI(rs, best_f=best_f)),
        "qlogei": (
            BtqLogEI(BtPM(bt), best_f=best_f, sampler=BtSt(torch.Size([1]))),
            RsqLogEI(RsPM(rs), best_f=best_f, sampler=RsSt(torch.Size([1]))),
        ),
        "bounds": torch.stack([torch.zeros(d, dtype=DT), torch.ones(d, dtype=DT)]),
        "g": torch.Generator().manual_seed(9200 + index),
    }


def _check(label: str, rs: Any, bt: Any, xtol: float) -> None:
    CHECK.check(f"{label}: candidates", rs[0], bt[0], xtol)
    if bt[1] is not None:
        CHECK.check(f"{label}: values", rs[1], bt[1], VALUE_TOL)


@pytest.mark.parametrize("index", range(N_SCENARIOS))
def test_discrete_and_local_search(index: int) -> None:
    s = scenario(index)
    r, d, g = s["r"], s["d"], s["g"]
    (a_bt, a_rs), (q_bt, q_rs) = s["logei"], s["qlogei"]
    choices = torch.rand(r.randint(20, 120), d, generator=g, dtype=DT)
    q = r.randint(1, 3)
    kwargs: dict[str, Any] = {"unique": r.random() < 0.7}
    if r.random() < 0.5:
        kwargs["X_avoid"] = choices[: r.randint(1, 6)]
    if r.random() < 0.5:
        idx = sorted(r.sample(range(d), 2))
        kwargs["inequality_constraints"] = [
            (
                torch.tensor(idx),
                torch.tensor([r.uniform(0.5, 1.5), r.uniform(-1, 1)], dtype=DT),
                r.uniform(-0.2, 0.6),
            )
        ]
    acq = (a_bt, a_rs) if q == 1 else (q_bt, q_rs)
    try:
        bt = bto.optimize_acqf_discrete(acq[0], q, choices, **kwargs)
    except Exception as e:
        with pytest.raises(type(e)):
            rso.optimize_acqf_discrete(acq[1], q, choices, **kwargs)
        return
    _check(f"discrete q={q}", rso.optimize_acqf_discrete(acq[1], q, choices, **kwargs), bt, 0.0)
    levels = torch.linspace(0, 1, r.randint(3, 7), dtype=DT)
    dc = [levels] * d
    q = r.randint(1, 3)
    acq = (a_bt, a_rs) if q == 1 else (q_bt, q_rs)
    lp = lattice_points(9300 + index)
    nrest = r.randint(2, 6)
    with (
        patched(bto, "_gen_batch_initial_conditions_local_search", lp),
        patched(rso, "_gen_batch_initial_conditions_local_search", lp),
    ):
        bt = bto.optimize_acqf_discrete_local_search(
            acq[0], dc, q, num_restarts=nrest, raw_samples=64
        )
        rs = rso.optimize_acqf_discrete_local_search(
            acq[1], dc, q, num_restarts=nrest, raw_samples=64
        )
    _check(f"local search q={q}", rs, bt, 0.0)


@pytest.mark.parametrize("index", range(N_SCENARIOS))
def test_mixed(index: int) -> None:
    s = scenario(index)
    r, d, g = s["r"], s["d"], s["g"]
    (a_bt, a_rs), (q_bt, q_rs) = s["logei"], s["qlogei"]
    nr = r.randint(2, 6)
    ics = torch.rand(nr, 1, d, generator=g, dtype=DT)
    k = r.randint(1, d - 1)
    dims = sorted(r.sample(range(d), k))
    ffl = [{j: r.choice([0.0, 0.5, 1.0]) for j in dims} for _ in range(r.randint(1, 4))]
    q = r.randint(1, 2)
    acq = (a_bt, a_rs) if q == 1 else (q_bt, q_rs)
    for options in lbfgsb_modes():
        tag = "serial" if options else "parallel"
        bt = bto.optimize_acqf_mixed(
            acq[0], s["bounds"], q, nr, ffl, batch_initial_conditions=ics, options=options
        )
        rs = rso.optimize_acqf_mixed(
            acq[1], s["bounds"], q, nr, ffl, batch_initial_conditions=ics, options=options
        )
        _check(f"mixed q={q} [{tag}]", rs, bt, XTOL)


@pytest.mark.parametrize("index", range(N_SCENARIOS))
def test_mixed_alternating(index: int) -> None:
    s = scenario(index)
    r, d, g = s["r"], s["d"], s["g"]
    if d < 3:
        pytest.skip("needs a continuous dimension besides the discrete and categorical ones")
    (a_bt, a_rs), (q_bt, q_rs) = s["logei"], s["qlogei"]
    levels = sorted({0.0, 1.0, *(round(r.random(), 3) for _ in range(r.randint(1, 4)))})
    cats = sorted({round(r.random(), 3) for _ in range(r.randint(2, 4))})
    discrete_dims = {0: levels}
    cat_dims = {d - 1: cats}
    b = r.randint(3, 7)
    start = torch.rand(b, d, generator=g, dtype=DT)
    start[:, 0] = torch.tensor([r.choice(levels) for _ in range(b)], dtype=DT)
    start[:, d - 1] = torch.tensor([r.choice(cats) for _ in range(b)], dtype=DT)
    q = r.randint(1, 2)
    acq = (a_bt, a_rs) if q == 1 else (q_bt, q_rs)
    bounds = s["bounds"].clone()
    bounds[:, d - 1] = torch.tensor([min(cats), max(cats)], dtype=DT)
    kw = dict(discrete_dims=discrete_dims, cat_dims=cat_dims, num_restarts=b, raw_samples=32, q=q)
    with (
        patched(btm, "generate_starting_points", starting_points(start, acq[0])),
        patched(rsm, "generate_starting_points", starting_points(start, acq[0])),
    ):
        bt = btm.optimize_acqf_mixed_alternating(acq[0], bounds, **kw)
        rs = rsm.optimize_acqf_mixed_alternating(acq[1], bounds, **kw)
    if botorch_parallel_lbfgsb():
        _check(f"alternating q={q}", rs, bt, XTOL)
    else:
        shortfall = float(bt[1].reshape(-1)[0]) - float(rs[1].reshape(-1)[0])
        CHECK.check_bound(
            f"alternating q={q} (0.18.1): shortfall",
            shortfall / max(abs(float(bt[1].reshape(-1)[0])), 1.0),
            1e-9,
        )


@pytest.mark.parametrize("index", range(N_SCENARIOS))
def test_generation(index: int) -> None:
    s = scenario(index)
    r, d, g = s["r"], s["d"], s["g"]
    (a_bt, a_rs), (q_bt, q_rs) = s["logei"], s["qlogei"]
    ics = torch.rand(r.randint(2, 5), 1, d, generator=g, dtype=DT)
    options = {
        "optimizer_options": {"lr": r.choice([0.01, 0.025, 0.05])},
        "stopping_criterion_options": {"maxiter": r.randint(20, 120), "n_window": r.randint(3, 10)},
    }
    bt = bt_gen.gen_candidates_torch(ics, a_bt, s["bounds"][0], s["bounds"][1], options=options)
    rs = rs_gen.gen_candidates_torch(ics, a_rs, s["bounds"][0], s["bounds"][1], options=options)
    _check("gen_candidates_torch", rs, bt, XTOL)
    cyc_ics = torch.rand(3, 1, d, generator=g, dtype=DT)

    def ic_gen(
        acq_function: Any, bounds: Any, q: int, num_restarts: int, raw_samples: int, **kw: Any
    ) -> Any:
        return cyc_ics[:num_restarts].clone()

    orig_call = bt_stopping.ExpMAStoppingCriterion.__call__
    cyclic_options = {"maxiter": r.randint(2, 6)}
    with patched(
        bt_stopping.ExpMAStoppingCriterion,
        "__call__",
        lambda self, fvals: orig_call(self, fvals.clone()),
    ):
        bt = bto.optimize_acqf_cyclic(
            q_bt,
            s["bounds"],
            2,
            3,
            raw_samples=8,
            ic_generator=ic_gen,
            cyclic_options=cyclic_options,
            options=dict(lbfgsb_modes()[0]),
        )
    rs = rso.optimize_acqf_cyclic(
        q_rs,
        s["bounds"],
        2,
        3,
        raw_samples=8,
        ic_generator=ic_gen,
        cyclic_options=cyclic_options,
        options=dict(lbfgsb_modes()[0]),
    )
    _check("cyclic", rs, bt, XTOL)
    Xc = torch.rand(r.randint(20, 150), d, generator=g, dtype=DT)
    repl = r.random() < 0.5
    num_samples = r.randint(1, 8)
    torch.manual_seed(index)
    sb = bt_sampling.MaxPosteriorSampling(s["bt"], replacement=repl)(Xc, num_samples=num_samples)
    torch.manual_seed(index)
    sr = rs_sampling.MaxPosteriorSampling(s["rs"], replacement=repl)(Xc, num_samples=num_samples)
    CHECK.check("MaxPosteriorSampling", sr, sb, 0.0)

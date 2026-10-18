"""The randomized equivalence suite of phase 8b: botorch_rs against live
BoTorch over a seeded grid of models and options.

Scenarios (``SCENARIOS``) are ``SingleTaskGP`` models with one and two
outputs and ``ModelListGP`` models, with n = 6, 10, 14 training points,
d = 1-3 inputs, inferred or fixed observation noise, with or without
``Normalize`` / ``Standardize``, RBF or Matern-5/2 kernels and several input
ranges, built identically in both libraries. Seeded generators draw the
options of

* ``qMultiStepLookahead``: one to three steps, one to three fantasies per
  step, batch sizes 1 or 2, stage and terminal value functions (posterior
  mean, UCB, EI and qEI with ``best_f`` from the model, qSR), posterior
  transforms and MC objectives, pending points, fantasy weights and
  ``collapse_fantasy_base_samples``; BoTorch's base samples are installed in
  botorch_rs;
* Python subclasses of ``SampleReducingMCAcquisitionFunction`` and
  ``LogImprovementMCAcquisitionFunction`` with objectives, transforms,
  constraints and pending points (identical base samples);
* the input constructors of the ``best_f`` acquisition functions and the
  noisy ones (datasets of both libraries, single or block designs,
  objectives, transforms) and ``get_acquisition_function``.

Tolerances (the worst observed error of every quantity is printed in the
terminal summary, section "equivalence suite (phase 8b)"): values 1e-9,
gradients 1e-8 (relative); ``best_f`` and baselines from the training data
must be identical. The look-ahead's gradients are compared with BoTorch
under ``propagate_grads(True)`` without ``fast_pred_var`` (BoTorch's own are
wrong), ``best_f`` from the model with BoTorch's corrected ``make_best_f``,
``collapse_fantasy_base_samples=False`` of single-output posteriors with
BoTorch's corrected batch range (see ``docs/drafts/8b-acquisition-rest.md``).

Run with ``pytest -m equivalence``; skipped unless BoTorch is importable,
which ``BOTORCH_RS_REQUIRE_BOTORCH=1`` turns into an error.
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

import botorch.acquisition as ba  # noqa: E402
import botorch.acquisition.factory as bt_factory  # noqa: E402
import botorch.acquisition.input_constructors as bic  # noqa: E402
from acquisition_rest_helpers import (  # noqa: E402
    best_f_argfac,
    bounded_subclass,
    copy_lookahead_samplers,
    corrected_lookahead,
    datasets,
    improvement_subclass,
    log_improvement_subclass,
    single_output_collapse_fix,
)
from botorch.acquisition import logei as bt_logei  # noqa: E402
from botorch.acquisition import monte_carlo as bt_mc  # noqa: E402
from botorch.acquisition import multi_step_lookahead as bt_msl  # noqa: E402
from botorch.acquisition.objective import GenericMCObjective as BtGeneric  # noqa: E402
from botorch.acquisition.objective import (  # noqa: E402
    ScalarizedPosteriorTransform as BtSPT,
)
from botorch.sampling.normal import SobolQMCNormalSampler as BtSampler  # noqa: E402

import botorch_rs.acquisition as ra  # noqa: E402
import botorch_rs.acquisition.factory as rs_factory  # noqa: E402
import botorch_rs.acquisition.input_constructors as ric  # noqa: E402
from botorch_rs.acquisition import logei as rs_logei  # noqa: E402
from botorch_rs.acquisition import monte_carlo as rs_mc  # noqa: E402
from botorch_rs.acquisition import multi_step_lookahead as rs_msl  # noqa: E402
from botorch_rs.acquisition.objective import GenericMCObjective as RsGeneric  # noqa: E402
from botorch_rs.acquisition.objective import (  # noqa: E402
    ScalarizedPosteriorTransform as RsSPT,
)
from botorch_rs.sampling import SobolQMCNormalSampler as RsSampler  # noqa: E402
from parity_helpers import (  # noqa: E402
    Checker,
    ModelSpec,
    list_pair,
    make_data,
    model_pair,
    points_in,
    sampler_pair,
    values_and_grads,
)

pytestmark = [pytest.mark.equivalence, pytest.mark.filterwarnings("ignore")]

SEED = 20261018
DT = torch.float64
VALUE_TOL = 1e-9
GRAD_TOL = 1e-8
CHECK = Checker("equivalence suite (phase 8b)")


@pytest.fixture(autouse=True, scope="module")
def _single_threaded_torch() -> Iterator[None]:
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(threads)


def rng_for(*keys: Any) -> random.Random:
    """A generator seeded by ``keys`` (stable across runs and platforms)."""
    return random.Random(zlib.crc32(repr((SEED, *keys)).encode()))


@dataclass(frozen=True)
class Scenario:
    label: str
    specs: tuple[ModelSpec, ...]
    is_list: bool
    seed: int

    @property
    def d(self) -> int:
        return self.specs[0].d

    @property
    def m(self) -> int:
        return sum(s.m for s in self.specs)


def _scenarios() -> list[Scenario]:
    r = random.Random(SEED)
    out: list[Scenario] = []
    ranges = [(0.0, 1.0), (-2.0, 3.0), (1.0, 4.0)]
    for i in range(10):
        lo, hi = r.choice(ranges)
        spec = ModelSpec(
            n=[6, 10, 14][i % 3],
            d=[1, 2, 3][(i // 3) % 3],
            m=2 if i in (3, 7) else 1,
            fixed_noise=r.random() < 0.3,
            normalize=r.random() < 0.5,
            standardize=r.random() < 0.75,
            kernel=r.choice(["rbf", "matern52"]),
            lo=lo,
            hi=hi,
        )
        out.append(Scenario(f"{i:02d}_{spec.label}", (spec,), False, 900 + i))
    lists = [
        (ModelSpec(10, 2), ModelSpec(8, 2, kernel="matern52", normalize=True)),
        (ModelSpec(9, 3, fixed_noise=True), ModelSpec(12, 3, standardize=False)),
    ]
    for j, specs in enumerate(lists):
        label = "list_" + "+".join(s.label for s in specs)
        out.append(Scenario(f"{10 + j:02d}_{label}", specs, True, 950 + j))
    return out


SCENARIOS = _scenarios()


@pytest.fixture(scope="module", params=SCENARIOS, ids=[s.label for s in SCENARIOS])
def pair(request: pytest.FixtureRequest) -> tuple[Scenario, Any, Any, Any, Any]:
    """``(scenario, bt, rs, X, Y)`` with the untransformed targets of all
    outputs (for a model list, those of every sub-model on its own inputs)."""
    sc = request.param
    if sc.is_list:
        bt, rs, X = list_pair(sc.specs, sc.seed)
        ys = [make_data(s, sc.seed + 13 * i)[1] for i, s in enumerate(sc.specs)]
        n = min(y.shape[0] for y in ys)
        Y = torch.cat([y[:n] for y in ys], dim=-1)
        return sc, bt, rs, X[:n], Y
    bt, rs, X = model_pair(sc.specs[0], sc.seed)
    return sc, bt, rs, X, make_data(sc.specs[0], sc.seed)[1]


def _weights(r: random.Random) -> Any:
    return torch.tensor([r.choice([0.5, 0.75, 1.0]), r.choice([-0.25, 0.25, -0.5])], dtype=DT)


# ---------------------------------------------------------------------------
# qMultiStepLookahead
# ---------------------------------------------------------------------------


def _lookahead_config(sc: Scenario, r: random.Random) -> dict:
    k = r.choice([1, 1, 2, 2, 3])
    nf = [r.choice([1, 2, 3]) for _ in range(k)]
    while k == 3 and nf[0] * nf[1] * nf[2] > 8:
        nf = [r.choice([1, 2]) for _ in range(k)]
    # BoTorch fails for one fantasy followed by several (see the draft;
    # test_parity_8b_acquisition_rest checks that case against a duplicated
    # fantasy).
    for j in range(k):
        if nf[j] == 1 and any(f > 1 for f in nf[j + 1 :]):
            nf[j] = 2
    single = sc.m == 1
    # BoTorch fails for a single t-batch of multi-output fantasies (as its
    # qKG, see the phase-8 draft).
    t = r.choice([1, 2]) if (single or sc.is_list) else 2
    batch_sizes = [r.choice([1, 2]) if (t == 1 and single and j < k - 1) else 1 for j in range(k)]
    terminal = r.choice(
        ["PosteriorMean", "qSimpleRegret", "UpperConfidenceBound"]
        + (["ExpectedImprovement", "qExpectedImprovement"] if single else [])
    )
    stages: list[str | None] = []
    for j in range(k):
        q_j = batch_sizes[j - 1] if j > 0 else None
        choice = r.choice(
            [None, None, "qSimpleRegret"] + (["qExpectedImprovement"] if single else [])
        )
        if choice is None and q_j == 1 and r.random() < 0.3:
            choice = "UpperConfidenceBound"
        stages.append(choice)
    stages.append(terminal)
    return {
        "batch_sizes": batch_sizes,
        "nf": nf,
        "t": t,
        "q": r.choice([1, 2]) if all(s is None or s.startswith("q") for s in stages[:1]) else 1,
        "stages": stages,
        "pending": r.random() < 0.3 and (stages[0] is None or stages[0].startswith("q")),
        "weights": r.random() < 0.3,
        "collapse": r.random() < 0.7,
        "objective": not single and r.random() < 0.5,
        "beta": r.choice([0.25, 1.0]),
    }


def test_multi_step_lookahead(pair: tuple[Scenario, Any, Any, Any, Any]) -> None:
    sc, bt, rs, X, _ = pair
    for rep in range(2):
        r = rng_for("msl", sc.label, rep)
        cfg = _lookahead_config(sc, r)
        stages = cfg["stages"]
        if cfg["objective"]:
            # An MC objective needs MC value functions.
            stages = [
                None if s is None else ("qSimpleRegret" if not s.startswith("q") else s)
                for s in stages
            ]
        kb: dict[str, Any] = {"batch_sizes": cfg["batch_sizes"]}
        kr: dict[str, Any] = {"batch_sizes": cfg["batch_sizes"]}
        torch.manual_seed(r.randrange(10**6))
        kb["samplers"] = [BtSampler(sample_shape=torch.Size([f])) for f in cfg["nf"]]
        kr["samplers"] = [RsSampler(sample_shape=(f,)) for f in cfg["nf"]]
        if cfg["weights"]:
            for sb, sr, f in zip(kb["samplers"], kr["samplers"], cfg["nf"], strict=True):
                w = torch.tensor([r.uniform(0.5, 2.0) for _ in range(f)], dtype=DT).reshape(f, 1)
                sb.base_weights = sr.base_weights = w
        kb["valfunc_cls"] = [None if s is None else getattr(ba, s) for s in stages]
        kr["valfunc_cls"] = [None if s is None else getattr(ra, s) for s in stages]
        beta = cfg["beta"]
        argb: list[Any] = []
        argr: list[Any] = []
        for s in stages:
            if s in ("ExpectedImprovement", "qExpectedImprovement"):
                argb.append(best_f_argfac(s == "ExpectedImprovement"))
                argr.append(rs_msl.make_best_f)
            elif s == "UpperConfidenceBound":
                argb.append(lambda model, X, beta=beta: {"beta": beta})
                argr.append(lambda model, X, beta=beta: {"beta": beta})
            else:
                argb.append(None)
                argr.append(None)
        kb["valfunc_argfacs"], kr["valfunc_argfacs"] = argb, argr
        kb["inner_mc_samples"] = kr["inner_mc_samples"] = [
            r.choice([4, 8]) if (s is not None and s.startswith("q")) else None for s in stages
        ]
        if cfg["objective"]:
            w = _weights(r)
            kb["objective"] = BtGeneric(lambda Y, X=None, w=w: Y @ w)
            kr["objective"] = RsGeneric(lambda Y, X=None, w=w: Y @ w)
        elif sc.m > 1:
            w = _weights(r)
            kb["posterior_transform"], kr["posterior_transform"] = BtSPT(w), RsSPT(w)
        if cfg["pending"]:
            p = points_in(X, (1, sc.d), r.randrange(10**6))
            kb["X_pending"] = kr["X_pending"] = p
        kb["collapse_fantasy_base_samples"] = cfg["collapse"]
        kr["collapse_fantasy_base_samples"] = cfg["collapse"]
        a_bt = bt_msl.qMultiStepLookahead(bt, **kb)
        a_rs = rs_msl.qMultiStepLookahead(rs, **kr)
        q_aug = a_bt.get_augmented_q_batch_size(cfg["q"])
        Xt = points_in(X, (cfg["t"], q_aug, sc.d), r.randrange(10**6))
        with single_output_collapse_fix():
            try:
                with torch.no_grad():
                    vb = a_bt(Xt)
            except Exception:
                # A configuration BoTorch rejects when evaluated (an analytic
                # value function of several points): botorch_rs too.
                with pytest.raises(Exception):  # noqa: B017
                    a_rs(Xt)
                CHECK.record("qMultiStepLookahead: configurations both reject", 0.0, 0.0)
                continue
            copy_lookahead_samplers(a_bt, a_rs)
            with corrected_lookahead():
                _, gb = values_and_grads(a_bt, Xt)
        vr, gr = values_and_grads(a_rs, Xt)
        CHECK.check("qMultiStepLookahead: values", vr, vb, VALUE_TOL)
        CHECK.check(
            "qMultiStepLookahead: gradients (corrected BoTorch)", gr, gb, GRAD_TOL, atol=1e-14
        )


# ---------------------------------------------------------------------------
# Python subclasses of the sample-reducing acquisition functions
# ---------------------------------------------------------------------------


def test_sample_reducing_subclasses(pair: tuple[Scenario, Any, Any, Any, Any]) -> None:
    sc, bt, rs, X, _ = pair
    r = rng_for("sr", sc.label)
    q = r.choice([1, 2, 3])
    n_pend = r.choice([0, 0, 1])
    kb: dict[str, Any] = {}
    kr: dict[str, Any] = {}
    m_s = sc.m
    if sc.m > 1:
        if r.random() < 0.5:
            w = _weights(r)
            kb["posterior_transform"], kr["posterior_transform"] = BtSPT(w), RsSPT(w)
            m_s = 1
        else:
            w = _weights(r)
            kb["objective"] = BtGeneric(lambda Y, X=None, w=w: Y @ w)
            kr["objective"] = RsGeneric(lambda Y, X=None, w=w: Y @ w)
    if r.random() < 0.5:
        c = r.uniform(-0.5, 1.5)
        kb["constraints"] = kr["constraints"] = [lambda Z, c=c: Z[..., 0] - c]
        kb["eta"] = kr["eta"] = 2.0 ** -r.choice([4, 6, 8])
    if n_pend:
        kb["X_pending"] = kr["X_pending"] = points_in(X, (n_pend, sc.d), r.randrange(10**6))
    s_bt, s_rs, _ = sampler_pair(r.choice([8, 16]), q + n_pend, m_s, r.randrange(10**6))
    best_f = r.uniform(-1.0, 1.0)
    Xt = points_in(X, (3, q, sc.d), r.randrange(10**6))
    pairs = [
        (
            "improvement",
            improvement_subclass(bt_mc.SampleReducingMCAcquisitionFunction)(
                bt, best_f, sampler=s_bt, **kb
            ),
            improvement_subclass(rs_mc.SampleReducingMCAcquisitionFunction)(
                rs, best_f, sampler=s_rs, **kr
            ),
        ),
        (
            "log improvement",
            log_improvement_subclass(bt_logei.LogImprovementMCAcquisitionFunction)(
                bt, best_f, sampler=s_bt, **kb
            ),
            log_improvement_subclass(rs_logei.LogImprovementMCAcquisitionFunction)(
                rs, best_f, sampler=s_rs, **kr
            ),
        ),
    ]
    if "constraints" not in kb:
        pairs.append(
            (
                "signed",
                bounded_subclass(bt_mc.SampleReducingMCAcquisitionFunction)(bt, sampler=s_bt, **kb),
                bounded_subclass(rs_mc.SampleReducingMCAcquisitionFunction)(rs, sampler=s_rs, **kr),
            )
        )
    for name, a_bt, a_rs in pairs:
        vb, gb = values_and_grads(a_bt, Xt)
        vr, gr = values_and_grads(a_rs, Xt)
        CHECK.check(f"SampleReducing subclass ({name}): values", vr, vb, VALUE_TOL)
        CHECK.check(f"SampleReducing subclass ({name}): gradients", gr, gb, GRAD_TOL, atol=1e-14)


# ---------------------------------------------------------------------------
# Input constructors and get_acquisition_function
# ---------------------------------------------------------------------------


def test_input_constructors(pair: tuple[Scenario, Any, Any, Any, Any]) -> None:
    sc, bt, rs, X, Y = pair
    r = rng_for("ic", sc.label)
    block = sc.m > 1 and r.random() < 0.5
    lib = r.choice(["botorch", "botorch_rs"])
    ds_b = datasets(X, Y, "botorch", block)
    ds_r = datasets(X, Y, lib, block)
    kb: dict[str, Any] = {}
    kr: dict[str, Any] = {}
    if sc.m > 1:
        w = _weights(r)
        kb["posterior_transform"], kr["posterior_transform"] = BtSPT(w), RsSPT(w)
    # Analytic best_f.
    for name in ("ExpectedImprovement", "LogProbabilityOfImprovement"):
        cb = {c.__name__: c for c in bic.ACQF_INPUT_CONSTRUCTOR_REGISTRY}[name]
        cr = {c.__name__: c for c in ric.ACQF_INPUT_CONSTRUCTOR_REGISTRY}[name]
        ab = bic.get_acqf_input_constructor(cb)(model=bt, training_data=ds_b, **kb)
        ar = ric.get_acqf_input_constructor(cr)(model=rs, training_data=ds_r, **kr)
        same = torch.equal(torch.as_tensor(ar["best_f"], dtype=DT), ab["best_f"].to(DT))
        CHECK.record("construct_inputs_best_f: best_f", 0.0 if same else float("inf"), 0.0)
        assert same
        Xt = points_in(X, (4, 1, sc.d), r.randrange(10**6))
        vb, gb = values_and_grads(cb(**ab), Xt)
        vr, gr = values_and_grads(cr(**ar), Xt)
        CHECK.check(f"{name} from its input constructor: values", vr, vb, VALUE_TOL)
        CHECK.check(f"{name} from its input constructor: gradients", gr, gb, GRAD_TOL, atol=1e-14)
    # MC best_f and X_baseline.
    q = r.choice([1, 2])
    s_bt, s_rs, _ = sampler_pair(16, q, 1 if sc.m == 1 or kb else sc.m, r.randrange(10**6))
    ab = bic.get_acqf_input_constructor(ba.qExpectedImprovement)(
        model=bt, training_data=ds_b, sampler=s_bt, **kb
    )
    ar = ric.get_acqf_input_constructor(ra.qExpectedImprovement)(
        model=rs, training_data=ds_r, sampler=s_rs, **kr
    )
    same = torch.equal(torch.as_tensor(ar["best_f"], dtype=DT), ab["best_f"].to(DT))
    CHECK.record("construct_inputs_qEI: best_f", 0.0 if same else float("inf"), 0.0)
    assert same
    Xt = points_in(X, (3, q, sc.d), r.randrange(10**6))
    vb, gb = values_and_grads(ba.qExpectedImprovement(**ab), Xt)
    vr, gr = values_and_grads(ra.qExpectedImprovement(**ar), Xt)
    CHECK.check("qExpectedImprovement from its input constructor: values", vr, vb, VALUE_TOL)
    CHECK.check(
        "qExpectedImprovement from its input constructor: gradients", gr, gb, GRAD_TOL, atol=1e-14
    )
    ab = bic.get_acqf_input_constructor(ba.qLogNoisyExpectedImprovement)(
        model=bt, training_data=ds_b, **kb
    )
    ar = ric.get_acqf_input_constructor(ra.qLogNoisyExpectedImprovement)(
        model=rs, training_data=ds_r, **kr
    )
    assert torch.equal(torch.as_tensor(ar["X_baseline"], dtype=DT), ab["X_baseline"])


def test_get_acquisition_function(pair: tuple[Scenario, Any, Any, Any, Any]) -> None:
    sc, bt, rs, X, _ = pair
    r = rng_for("factory", sc.label)
    name = r.choice(["qEI", "qLogEI", "qSR", "qUCB"])
    if sc.m > 1:
        w = _weights(r)
        ob = BtGeneric(lambda Y, X=None, w=w: Y @ w)
        orr = RsGeneric(lambda Y, X=None, w=w: Y @ w)
    else:
        ob, orr = ba.IdentityMCObjective(), ra.IdentityMCObjective()
    kw: dict[str, Any] = {"beta": 0.5} if name == "qUCB" else {}
    a_bt = bt_factory.get_acquisition_function(name, bt, ob, X, mc_samples=16, seed=1, **kw)
    a_rs = rs_factory.get_acquisition_function(name, rs, orr, X, mc_samples=16, seed=1, **kw)
    assert type(a_rs).__name__ == type(a_bt).__name__
    if name in ("qEI", "qLogEI"):
        CHECK.check("get_acquisition_function: best_f", a_rs.best_f, a_bt.best_f, VALUE_TOL)

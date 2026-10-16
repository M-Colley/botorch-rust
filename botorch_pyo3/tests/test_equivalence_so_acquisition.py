"""The randomized equivalence suite of the single-objective acquisition
functions of phase 8: botorch_rs against live BoTorch over a seeded grid of
models and options.

Scenarios (``SCENARIOS``) are ``SingleTaskGP`` models with one and two
outputs and ``ModelListGP`` models, with n = 6, 10, 14 training points,
d = 1-3 inputs, inferred or fixed observation noise, with or without
``Normalize`` / ``Standardize``, RBF or Matern-5/2 kernels and several input
ranges. For every scenario both libraries are built on identical data and
hyperparameters, seeded generators draw ``q``, the numbers of fantasies,
pending points, test points and options, and BoTorch's random numbers (base
samples, max values, path draws, Sobol points) are passed to botorch_rs.
Compared are the values and gradients of

* qKnowledgeGradient (posterior mean or MC objective, ``current_value``,
  pending points, posterior transforms);
* qNoisyExpectedImprovement (``cache_root`` on and off, pending points),
  qLowerConfidenceBound and qPosteriorStandardDeviation;
* qAnalyticProbabilityOfImprovement (BoTorch's gradients one candidate set
  at a time: its batched MVNXPB gradients are wrong);
* qMaxValueEntropy and qLowerBoundMaxValueEntropy (single output);
* qJointEntropySearch ("LB", and "MC" for one point) on the optima of
  ``get_optimal_samples`` (compared with BoTorch's on the same draws);
* PathwiseThompsonSampling and qNegIntegratedPosteriorVariance.

Tolerances (the worst observed error of every quantity is printed in the
terminal summary, section "equivalence suite (SO acquisition)"): values
1e-9, gradients 1e-8 (relative); where BoTorch is wrong (see
``docs/drafts/8-so-acquisition.md``) the gradients are compared with
central differences of botorch_rs's values (1e-7) or with BoTorch without
``fast_pred_var``; ``get_optimal_samples`` 1e-6 (L-BFGS-B).

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

from botorch.acquisition import active_learning as bt_al  # noqa: E402
from botorch.acquisition import analytic as bt_an  # noqa: E402
from botorch.acquisition import joint_entropy_search as bt_jes  # noqa: E402
from botorch.acquisition import knowledge_gradient as bt_kg  # noqa: E402
from botorch.acquisition import monte_carlo as bt_mc  # noqa: E402
from botorch.acquisition import thompson_sampling as bt_ts  # noqa: E402
from botorch.acquisition import utils as bt_autils  # noqa: E402
from botorch.acquisition.objective import GenericMCObjective as BtGeneric  # noqa: E402
from botorch.acquisition.objective import (  # noqa: E402
    ScalarizedPosteriorTransform as BtSPT,
)
from botorch.sampling.normal import SobolQMCNormalSampler as BtSampler  # noqa: E402
from hvkg_helpers import use_fantasy_mean_cache  # noqa: E402
from mo_entropy_helpers import bt_path_draws, recorded_randn  # noqa: E402
from so_acquisition_helpers import (  # noqa: E402
    botorch_has_kg_fix,
    check_against_twins,
    finite_differences,
    jes_has_3399,
    jes_optimal_inputs,
    mes_pair,
    no_fast_pred_var,
    qpi_nan_gradients,
    recorded_optimal_samples,
    rs_sampler,
    sampler_pair,
    seeded_draws,
    twin,
)

from botorch_rs.acquisition import active_learning as rs_al  # noqa: E402
from botorch_rs.acquisition import analytic as rs_an  # noqa: E402
from botorch_rs.acquisition import joint_entropy_search as rs_jes  # noqa: E402
from botorch_rs.acquisition import knowledge_gradient as rs_kg  # noqa: E402
from botorch_rs.acquisition import monte_carlo as rs_mc  # noqa: E402
from botorch_rs.acquisition import thompson_sampling as rs_ts  # noqa: E402
from botorch_rs.acquisition import utils as rs_autils  # noqa: E402
from botorch_rs.acquisition.objective import GenericMCObjective as RsGeneric  # noqa: E402
from botorch_rs.acquisition.objective import (  # noqa: E402
    ScalarizedPosteriorTransform as RsSPT,
)
from botorch_rs.sampling import SobolQMCNormalSampler as RsSampler  # noqa: E402
from parity_helpers import (  # noqa: E402
    Checker,
    ModelSpec,
    list_pair,
    model_pair,
    points_in,
    values_and_grads,
)

pytestmark = [pytest.mark.equivalence, pytest.mark.filterwarnings("ignore")]

SEED = 20261016
DT = torch.float64
VALUE_TOL = 1e-9
GRAD_TOL = 1e-8
FD_TOL = 1e-7
CHECK = Checker("equivalence suite (SO acquisition)")


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
            m=2 if i in (4, 8) else 1,
            fixed_noise=r.random() < 0.3,
            normalize=r.random() < 0.5,
            standardize=r.random() < 0.75,
            kernel=r.choice(["rbf", "matern52"]),
            lo=lo,
            hi=hi,
        )
        out.append(Scenario(f"{i:02d}_{spec.label}", (spec,), False, 700 + i))
    lists = [
        (ModelSpec(10, 2), ModelSpec(8, 2, kernel="matern52", normalize=True)),
        (ModelSpec(9, 3, fixed_noise=True), ModelSpec(12, 3, standardize=False)),
    ]
    for j, specs in enumerate(lists):
        label = "list_" + "+".join(s.label for s in specs)
        out.append(Scenario(f"{10 + j:02d}_{label}", specs, True, 800 + j))
    return out


SCENARIOS = _scenarios()


@pytest.fixture(scope="module", params=SCENARIOS, ids=[s.label for s in SCENARIOS])
def pair(request: pytest.FixtureRequest) -> tuple[Scenario, Any, Any, Any]:
    sc = request.param
    if sc.is_list:
        return (sc, *list_pair(sc.specs, sc.seed))
    return (sc, *model_pair(sc.specs[0], sc.seed))


def _transforms(sc: Scenario, r: random.Random) -> tuple[Any, Any]:
    if sc.m == 1:
        return None, None
    w = torch.tensor([r.choice([0.5, 0.75, 1.0]), r.choice([-0.25, 0.25, -0.5])], dtype=DT)
    return BtSPT(w), RsSPT(w)


def _compare(name: str, a_rs: Any, a_bt: Any, X: Any, grads: bool = True) -> None:
    vr, gr = values_and_grads(a_rs, X)
    vb, gb = values_and_grads(a_bt, X)
    assert tuple(vr.shape) == tuple(vb.shape), (name, vr.shape, vb.shape)
    CHECK.check(f"{name}: values", vr, vb, VALUE_TOL)
    if grads:
        CHECK.check(f"{name}: gradients", gr, gb, GRAD_TOL)
    else:
        CHECK.check(
            f"{name}: gradients vs finite differences",
            gr[:1],
            finite_differences(a_rs, X[:1]),
            FD_TOL,
        )


def _bounds(X: Any) -> Any:
    return torch.stack([X.min(0).values, X.max(0).values])


def test_knowledge_gradient(pair: tuple[Scenario, Any, Any, Any]) -> None:
    sc, bt, rs, X = pair
    r = rng_for("kg", sc.label)
    d = sc.d
    q, n_f = r.choice([1, 2]), r.choice([2, 4])
    n_pend = r.choice([0, 0, 1])
    pending = points_in(X, (n_pend, d), r.randrange(10**6)) if n_pend else None
    k = q + n_pend
    # BoTorch fantasizes every sub-model of a ModelListGP with the same
    # sampler (single-output base samples).
    m_fant = 1 if sc.is_list else sc.m
    s_bt, s_rs = sampler_pair(n_f, (1, k, m_fant) if m_fant > 1 else (1, k), r.randrange(10**6))
    kb: dict[str, Any] = {"num_fantasies": n_f, "sampler": s_bt, "X_pending": pending}
    kr: dict[str, Any] = {"num_fantasies": n_f, "sampler": s_rs, "X_pending": pending}
    use_mc = r.random() < 0.4
    if use_mc:
        if sc.m == 1:
            kb["objective"] = BtGeneric(lambda Y, X=None: Y[..., 0] ** 2)
            kr["objective"] = RsGeneric(lambda Y, X=None: Y[..., 0] ** 2)
        else:
            kb["objective"] = BtGeneric(lambda Y, X=None: Y[..., 0] - 0.5 * Y[..., 1])
            kr["objective"] = RsGeneric(lambda Y, X=None: Y[..., 0] - 0.5 * Y[..., 1])
        mi = sc.m
        i_bt, i_rs = sampler_pair(8, (1, 1, 1, mi) if mi > 1 else (1, 1, 1), r.randrange(10**6))
        kb["inner_sampler"], kr["inner_sampler"] = i_bt, i_rs
    else:
        kb["posterior_transform"], kr["posterior_transform"] = _transforms(sc, r)
    if r.random() < 0.3:
        cv = float(torch.tensor(r.uniform(-1.0, 1.0), dtype=torch.float32))
        kb["current_value"] = kr["current_value"] = torch.tensor(cv, dtype=DT)
    a_bt = bt_kg.qKnowledgeGradient(bt, **kb)
    a_rs = rs_kg.qKnowledgeGradient(rs, **kr)
    # Two t-batches (BoTorch fails for a single t-batch of multi-output
    # fantasies).
    Xt = points_in(X, (2, q + n_f, d), r.randrange(10**6))
    _compare("qKG", a_rs, a_bt, Xt, grads=not use_mc or botorch_has_kg_fix())


def test_qnei_qlcb_qpstd(pair: tuple[Scenario, Any, Any, Any]) -> None:
    sc, bt, rs, X = pair
    r = rng_for("qnei", sc.label)
    d = sc.d
    q = r.choice([1, 2, 3])
    n_pend = r.choice([0, 1, 2])
    pending = points_in(X, (n_pend, d), r.randrange(10**6)) if n_pend else None
    cache_root = r.random() < 0.6
    N, m = X.shape[0], sc.m
    kb: dict[str, Any] = {}
    kr: dict[str, Any] = {}
    if m > 1:
        kb["objective"] = BtGeneric(lambda Y, X=None: Y[..., 0] - 0.5 * Y[..., 1])
        kr["objective"] = RsGeneric(lambda Y, X=None: Y[..., 0] - 0.5 * Y[..., 1])
    if sc.is_list:
        # The sub-models have different training inputs: the baseline are the
        # first sub-model's.
        N = X.shape[0]
    Xt = points_in(X, (2, q, d), r.randrange(10**6))

    def outcome(k: int) -> tuple[Any, Any]:
        with seeded_draws():
            a_bt = bt_mc.qNoisyExpectedImprovement(
                twin(bt, k),
                X,
                sampler=BtSampler(sample_shape=torch.Size([32])),
                X_pending=pending,
                prune_baseline=False,
                cache_root=cache_root,
                **kb,
            )
            return values_and_grads(a_bt, Xt)

    with seeded_draws():
        a_rs = rs_mc.qNoisyExpectedImprovement(
            rs,
            X,
            sampler=rs_sampler(32, [N * m, (N + q + n_pend) * m]),
            X_pending=pending,
            prune_baseline=False,
            cache_root=cache_root,
            **kr,
        )
        rs_vg = values_and_grads(a_rs, Xt)
    # The joint posterior over the baseline (the training points) is nearly
    # singular: BoTorch's outcomes depend on rounding (see the helper).
    check_against_twins(CHECK, "qNEI", rs_vg, outcome)
    tf = _transforms(sc, r)
    for name in ("qLowerConfidenceBound", "qPosteriorStandardDeviation"):
        s_bt, s_rs = sampler_pair(32, (1, q), r.randrange(10**6))
        kb = {"posterior_transform": tf[0]}
        kr = {"posterior_transform": tf[1]}
        if name == "qLowerConfidenceBound":
            kb["beta"] = kr["beta"] = r.choice([0.25, 1.0, 2.0])
        a_bt = getattr(bt_mc, name)(bt, sampler=s_bt, **kb)
        a_rs = getattr(rs_mc, name)(rs, sampler=s_rs, **kr)
        _compare(name, a_rs, a_bt, points_in(X, (2, q, d), r.randrange(10**6)))


def test_max_value_entropy(pair: tuple[Scenario, Any, Any, Any]) -> None:
    sc, bt, rs, X = pair
    if sc.m > 1:
        pytest.skip("single-output models only (as in BoTorch)")
    r = rng_for("mes", sc.label)
    d = sc.d
    cs = points_in(X, (r.choice([10, 20]), d), r.randrange(10**6))
    for cls in ("qMaxValueEntropy", "qLowerBoundMaxValueEntropy"):
        kw: dict[str, Any] = {}
        if r.random() < 0.3:
            kw["maximize"] = False
        pending = None
        # BoTorch's GIBBON repulsion is wrong with an outcome or input
        # transform (see the draft).
        plain = not sc.specs[0].standardize and not sc.specs[0].normalize
        if r.random() < 0.4 and (cls == "qMaxValueEntropy" or plain):
            pending = points_in(X, (r.choice([1, 2]), d), r.randrange(10**6))
        a_bt, a_rs = mes_pair(cls, bt, rs, cs, pending, dict(kw), dict(kw))
        _compare(cls, a_rs, a_bt, points_in(X, (3, 1, d), r.randrange(10**6)))


def test_joint_entropy_search(pair: tuple[Scenario, Any, Any, Any]) -> None:
    sc, bt, rs, X = pair
    r = rng_for("jes", sc.label)
    d = sc.d
    bounds = _bounds(X)
    tf = _transforms(sc, r)
    draws: list[Any] = []
    cands: list[Any] = []
    num = r.choice([2, 3])
    torch.manual_seed(r.randrange(10**6))
    with recorded_optimal_samples(draws, cands):
        xb, fb = bt_autils.get_optimal_samples(
            bt, bounds, num, raw_samples=32, num_restarts=3, posterior_transform=tf[0]
        )
    xo, fo = rs_autils.get_optimal_samples(
        rs,
        bounds,
        num,
        raw_samples=32,
        num_restarts=3,
        posterior_transform=tf[1],
        num_features=draws[0][0].shape[0] * 2,
        draws=draws,
        candidates=bounds[0] + (bounds[1] - bounds[0]) * cands[0],
    )
    CHECK.check("get_optimal_samples: inputs", xo, xb, 1e-6)
    CHECK.check("get_optimal_samples: outputs", fo, fb, 1e-6)
    est = r.choice(["LB", "MC"])
    q = r.choice([1, 2]) if est == "LB" else 1
    if est == "MC" and sc.m > 1 and not jes_has_3399():
        est = "LB"
    try:
        torch.manual_seed(0)
        a_bt = bt_jes.qJointEntropySearch(
            bt,
            jes_optimal_inputs(bt, xo),
            fo,
            posterior_transform=tf[0],
            estimation_type=est,
            num_samples=16,
        )
    except Exception as e:  # noqa: BLE001 - the same error is expected
        with pytest.raises(type(e)):
            rs_jes.qJointEntropySearch(rs, xo, fo, posterior_transform=tf[1], estimation_type=est)
        CHECK.record("qJointEntropySearch: errors (identical)", 0.0, 0.0)
        return
    use_fantasy_mean_cache(a_bt.conditional_model)  # meta-pytorch/botorch#3396
    s = RsSampler(sample_shape=torch.Size([16]), seed=0)
    Xt = points_in(X, (3, q, d), r.randrange(10**6))
    if est == "MC":
        a_bt(Xt[:1])
        s.base_samples = a_bt.sampler.base_samples.reshape(16, -1).clone()
    a_rs = rs_jes.qJointEntropySearch(
        rs, xo, fo, posterior_transform=tf[1], estimation_type=est, sampler=s
    )
    _compare(f"qJointEntropySearch {est}", a_rs, a_bt, Xt)


def test_thompson_sampling_and_nipv(pair: tuple[Scenario, Any, Any, Any]) -> None:
    sc, bt, rs, X = pair
    r = rng_for("ts", sc.label)
    d = sc.d
    tf = _transforms(sc, r)
    # BoTorch sums over the t-batch with a posterior transform and q = 1.
    q = r.choice([1, 2, 3]) if sc.m == 1 else r.choice([2, 3])
    torch.manual_seed(r.randrange(10**6))
    a_bt = bt_ts.PathwiseThompsonSampling(bt, posterior_transform=tf[0])
    noises: list[Any] = []
    with recorded_randn(noises):
        a_bt.redraw(q)
    a_bt.batch_size = q
    draws = bt_path_draws(a_bt.samples._path, noises)
    a_rs = rs_ts.PathwiseThompsonSampling(
        rs, posterior_transform=tf[1], num_features=draws[0][0].shape[0] * 2
    )
    a_rs.redraw(q, draws=draws)
    _compare("PathwiseThompsonSampling", a_rs, a_bt, points_in(X, (3, q, d), r.randrange(10**6)))
    mc = points_in(X, (r.choice([8, 16]), d), r.randrange(10**6))
    n_pend = r.choice([0, 1])
    pending = points_in(X, (n_pend, d), r.randrange(10**6)) if n_pend else None
    a_bt = bt_al.qNegIntegratedPosteriorVariance(
        bt, mc, posterior_transform=tf[0], X_pending=pending
    )
    a_rs = rs_al.qNegIntegratedPosteriorVariance(
        rs, mc, posterior_transform=tf[1], X_pending=pending
    )
    with no_fast_pred_var():
        _compare(
            "qNegIntegratedPosteriorVariance",
            a_rs,
            a_bt,
            points_in(X, (2, r.choice([1, 2]), d), r.randrange(10**6)),
        )


def test_q_analytic_probability_of_improvement(pair: tuple[Scenario, Any, Any, Any]) -> None:
    sc, bt, rs, X = pair
    r = rng_for("qpi", sc.label)
    d = sc.d
    tf = _transforms(sc, r)
    q = r.choice([1, 2, 3, 4])
    maximize = r.random() < 0.6
    # A float32-exact incumbent near the posterior means (BoTorch keeps it as
    # a float32 tensor).
    mu = bt.posterior(X, posterior_transform=tf[0]).mean.detach()
    best = float(torch.tensor(mu.max().item(), dtype=torch.float32))
    a_bt = bt_an.qAnalyticProbabilityOfImprovement(
        bt, best, posterior_transform=tf[0], maximize=maximize
    )
    a_rs = rs_an.qAnalyticProbabilityOfImprovement(
        rs, best, posterior_transform=tf[1], maximize=maximize
    )
    Xt = points_in(X, (3, q, d), r.randrange(10**6))
    vr, gr = values_and_grads(a_rs, Xt)
    vb, _ = values_and_grads(a_bt, Xt)
    CHECK.check("qAnalyticProbabilityOfImprovement: values", vr, vb, VALUE_TOL)
    gb = torch.cat([values_and_grads(a_bt, Xt[k : k + 1])[1] for k in range(3)])
    gb = qpi_nan_gradients(CHECK, a_rs, Xt, gr, gb)
    CHECK.check("qAnalyticProbabilityOfImprovement: gradients", gr, gb, GRAD_TOL, atol=1e-14)

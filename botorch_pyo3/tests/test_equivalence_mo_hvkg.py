"""The randomized equivalence suite of fantasies and the hypervolume
knowledge gradient: botorch_rs against live BoTorch over a seeded grid of
models and options.

Scenarios (``SCENARIOS``) are ``ModelListGP`` models of two or three
single-output ``SingleTaskGP`` sub-models with n = 5-12 training points,
d = 1-4 inputs, inferred or fixed observation noise, with or without
``Normalize`` / ``Standardize``, RBF or Matern-5/2 kernels and several input
ranges, built on identical data and hyperparameters in both libraries. For
every scenario, seeded generators draw

* a multi-output ``SingleTaskGP`` (m = 1-3) conditioned on observations
  (random batch shapes, explicit noise of fixed-noise models) and
  fantasized (random numbers of fantasies and points, batched inputs);
* ``ModelListGP.fantasize`` with a random evaluation mask;
* qHVKG / qMFHVKG configurations: ``num_fantasies`` (1-4), ``num_pareto``
  (1-3), ``q`` (0-3), posterior mean or samples, ``log``, ``current_value``,
  pending points with random evaluation masks, candidate evaluation masks,
  objectives (identity, ``outcomes``, weights, a generic nonlinear
  objective), inverse-cost-weighted utilities and target-fidelity
  projections,

compared on values, gradients and posterior moments with BoTorch's base
samples, and, independently of BoTorch,

* conditioning equals refitting: a model conditioned on fixed-noise
  observations equals the model built on the augmented data with the same
  hyperparameters (no transforms, which a refit would refit);
* the gradients of qHVKG agree with central finite differences;
* t-batches are independent (a stacked batch gives the values of the single
  candidate sets).

Tolerances (the worst observed error of every quantity is printed in the
terminal summary, section "equivalence suite (hypervolume knowledge
gradient)"): posterior moments and acquisition values 1e-9 (relative; log
values absolute below 1), gradients 1e-8 (relative to the largest
component), conditioning vs refit 1e-9, finite differences 1e-5 (relative
to the largest gradient component; step 1e-6), t-batches 1e-12.

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

from botorch.sampling.list_sampler import ListSampler as BtListSampler  # noqa: E402
from botorch.sampling.normal import SobolQMCNormalSampler as BtSampler  # noqa: E402
from hvkg_helpers import (  # noqa: E402
    HvkgCase,
    points,
    posterior_moments,
    run_hvkg_case,
    use_fantasy_mean_cache,
)
from mo_acquisition_helpers import DT  # noqa: E402

from botorch_rs.sampling import ListSampler as RsListSampler  # noqa: E402
from botorch_rs.sampling import SobolQMCNormalSampler as RsSampler  # noqa: E402
from parity_helpers import (  # noqa: E402
    Checker,
    ModelSpec,
    build_pair,
    copy_hyperparameters,
    list_pair,
    make_data,
    model_pair,
    set_hyperparameters,
)

pytestmark = [pytest.mark.equivalence, pytest.mark.filterwarnings("ignore")]

SEED = 20261012
VALUE_TOL = 1e-9
GRAD_TOL = 1e-8
MOMENT_TOL = 1e-9
FD_TOL = 1e-5
FD_STEP = 1e-6
CHECK = Checker("equivalence suite (hypervolume knowledge gradient)")


@pytest.fixture(autouse=True, scope="module")
def _single_threaded_torch() -> Iterator[None]:
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(threads)


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
    seed: int

    @property
    def d(self) -> int:
        return self.specs[0].d


def random_spec(r: random.Random, d: int, lo: float, hi: float, m: int = 1) -> ModelSpec:
    return ModelSpec(
        n=r.randint(5, 12),
        d=d,
        m=m,
        fixed_noise=r.random() < 0.3,
        normalize=r.random() < 0.5,
        standardize=r.random() < 0.7,
        kernel=r.choice(["rbf", "matern52"]),
        lo=lo,
        hi=hi,
    )


def _scenarios() -> list[Scenario]:
    r = random.Random(SEED)
    ranges = [(0.0, 1.0), (-2.0, 3.0), (1.0, 4.0)]
    out = []
    for i in range(8):
        d = [1, 2, 3, 4][i % 4]
        lo, hi = r.choice(ranges)
        specs = tuple(random_spec(r, d, lo, hi) for _ in range(r.choice([2, 3])))
        label = f"{i:02d}_d{d}_" + "+".join(s.label.split("_", 3)[-1] for s in specs)
        out.append(Scenario(label, specs, 700 + 11 * i))
    return out


SCENARIOS = _scenarios()


@pytest.fixture(scope="module", params=SCENARIOS, ids=[s.label for s in SCENARIOS])
def scenario(request: pytest.FixtureRequest) -> tuple[Scenario, Any, Any, Any]:
    sc = request.param
    return (sc, *list_pair(sc.specs, sc.seed))


def check_moments(label: str, rs_model: Any, bt_model: Any, X: Any) -> None:
    for noise in (False, True):
        m_rs, v_rs, c_rs = posterior_moments(rs_model, X, noise)
        m_bt, v_bt, c_bt = posterior_moments(bt_model, X, noise)
        CHECK.check(f"{label}: posterior mean", m_rs, m_bt, MOMENT_TOL)
        CHECK.check(f"{label}: posterior variance", v_rs, v_bt, MOMENT_TOL)
        CHECK.check(f"{label}: posterior covariance", c_rs, c_bt, MOMENT_TOL)


# ---------------------------------------------------------------------------
# Conditioning and fantasies
# ---------------------------------------------------------------------------


def test_single_task_conditioning(scenario: tuple) -> None:
    sc, _, _, X0 = scenario
    r = rng_for("single", sc.label)
    lo, hi = sc.specs[0].lo, sc.specs[0].hi
    spec = random_spec(r, sc.d, lo, hi, m=r.randint(1, 3))
    bt, rs, X = model_pair(spec, sc.seed + 1)
    m, d = spec.m, sc.d
    Xt = points(X, (r.randint(1, 4), d), sc.seed + 2)
    bt.posterior(Xt)
    rs.eval()
    g = torch.Generator().manual_seed(sc.seed + 3)
    batched_x = r.random() < 0.4
    batched_y = batched_x or (not spec.fixed_noise and r.random() < 0.4)
    n_new = r.randint(1, 4)
    x_shape = ((2,) if batched_x else ()) + (n_new,)
    y_shape = ((2,) if batched_y else ()) + (n_new,)
    Xn = points(X, (*x_shape, d), sc.seed + 4)
    Yn = 1.0 + 2.0 * torch.randn(*y_shape, m, generator=g, dtype=DT)
    kw: dict[str, Any] = {}
    if spec.fixed_noise:
        kw["noise"] = 0.005 + 0.02 * torch.rand(*y_shape, m, generator=g, dtype=DT)
    cm_bt = bt.condition_on_observations(X=Xn, Y=Yn, **kw)
    cm_rs = rs.condition_on_observations(X=Xn, Y=Yn, **kw)
    check_moments("condition_on_observations", cm_rs, cm_bt, Xt)
    # Fantasies.
    n_f = r.randint(1, 5)
    fx_shape = ((2,) if r.random() < 0.3 else ()) + (r.randint(1, 3),)
    Xf = points(X, (*fx_shape, d), sc.seed + 5)
    noise = None
    if r.random() < 0.4:
        noise = torch.full((1, m), 0.01 + 0.05 * r.random(), dtype=DT)
    s_bt = BtSampler(sample_shape=torch.Size([n_f]), seed=sc.seed + 6)
    fm_bt = bt.fantasize(Xf, s_bt, observation_noise=noise)
    if noise is not None:
        use_fantasy_mean_cache(fm_bt)  # meta-pytorch/botorch#3396
    s_rs = RsSampler(sample_shape=(n_f,), seed=0)
    s_rs.base_samples = s_bt.base_samples.clone()
    fm_rs = rs.fantasize(Xf, s_rs, observation_noise=noise)
    check_moments("fantasize", fm_rs, fm_bt, Xt)


def test_model_list_fantasize(scenario: tuple) -> None:
    sc, bt, rs, X = scenario
    r = rng_for("list", sc.label)
    m, d = rs.num_outputs, sc.d
    Xt = points(X, (3, d), sc.seed + 7)
    n_new, n_f = r.randint(1, 4), r.randint(1, 4)
    Xn = points(X, (n_new, d), sc.seed + 8)
    mask = None
    if r.random() < 0.6:
        rows = [[r.random() < 0.6 for _ in range(m)] for _ in range(n_new)]
        rows[0][0] = True
        mask = torch.tensor(rows)
    bts = [BtSampler(sample_shape=torch.Size([n_f]), seed=sc.seed + 9 + k) for k in range(m)]
    fm_bt = bt.fantasize(Xn, BtListSampler(*bts), evaluation_mask=mask)
    rss = []
    for s in bts:
        q = RsSampler(sample_shape=(n_f,), seed=0)
        if s.base_samples is not None:
            q.base_samples = s.base_samples.clone()
        rss.append(q)
    fm_rs = rs.fantasize(Xn, RsListSampler(*rss), evaluation_mask=mask)
    check_moments("ModelListGP.fantasize", fm_rs, fm_bt, Xt)


def test_conditioning_equals_refit(scenario: tuple) -> None:
    """Independent of BoTorch: conditioning a fixed-noise model without
    transforms on new observations equals building it on the augmented data
    with the same hyperparameters."""
    sc, _, _, _ = scenario
    r = rng_for("refit", sc.label)
    spec = ModelSpec(
        n=r.randint(5, 10),
        d=sc.d,
        m=r.randint(1, 2),
        fixed_noise=True,
        standardize=False,
        kernel=r.choice(["rbf", "matern52"]),
        lo=sc.specs[0].lo,
        hi=sc.specs[0].hi,
    )
    X, Y, Yvar = make_data(spec, sc.seed + 20)
    bt, rs = build_pair(spec, X, Y, Yvar)
    set_hyperparameters(bt, sc.seed + 20)
    copy_hyperparameters(bt, rs)
    rs.eval()
    k = r.randint(1, 4)
    g = torch.Generator().manual_seed(sc.seed + 21)
    Xn = points(X, (k, spec.d), sc.seed + 22)
    Yn = Y.mean(0) + torch.randn(k, spec.m, generator=g, dtype=DT)
    noise = 0.01 + 0.02 * torch.rand(k, spec.m, generator=g, dtype=DT)
    cm = rs.condition_on_observations(X=Xn, Y=Yn, noise=noise)
    _, refit = build_pair(spec, torch.cat([X, Xn]), torch.cat([Y, Yn]), torch.cat([Yvar, noise]))
    copy_hyperparameters(bt, refit)
    refit.eval()
    Xt = points(X, (4, spec.d), sc.seed + 23)
    for noise_flag in (False, True):
        a = posterior_moments(cm, Xt, noise_flag)
        e = posterior_moments(refit, Xt, noise_flag)
        CHECK.check("conditioning = refit: mean", a[0], e[0], MOMENT_TOL)
        CHECK.check("conditioning = refit: covariance", a[2], e[2], MOMENT_TOL)


# ---------------------------------------------------------------------------
# The hypervolume knowledge gradient
# ---------------------------------------------------------------------------


def random_case(r: random.Random, m: int, d: int, index: int) -> HvkgCase:
    q = r.randint(0, 3)
    log = r.random() < 0.3
    mf = d >= 2 and r.random() < 0.25
    cost = None
    if q > 0 and r.random() < 0.35:
        cost = "fixed_log" if log else ("affine" if mf else "fixed")
    current_value = cost is not None or r.random() < 0.3
    pending = r.choice([0, 0, 1, 2])
    masks = [None, None, "same_rows", "first_only"] + ([] if cost else ["alternate"])
    return HvkgCase(
        name=f"random {index}",
        n_f=r.randint(1, 4),
        num_pareto=r.randint(1, 3),
        q=q,
        use_posterior_mean=r.random() < 0.6,
        log=log,
        current_value=current_value,
        pending=pending,
        pending_mask=r.choice([None, "alternate"]) if pending else None,
        mask=r.choice(masks) if q > 0 else None,
        objective=r.choice(["identity", "weighted", "generic"] + (["outcomes"] if m >= 3 else [])),
        cost=cost,
        mf=mf,
        inner=r.randint(2, 8),
    )


@pytest.mark.parametrize("index", [0, 1, 2])
def test_hvkg(scenario: tuple, index: int) -> None:
    sc, bt, rs, X = scenario
    r = rng_for("hvkg", sc.label, index)
    case = random_case(r, rs.num_outputs, sc.d, index)
    pair, Xs, ev_rs, ev_bt = run_hvkg_case(case, bt, rs, X, sc.seed + 100 * index, count=2)
    err_v, err_g = pair.errors(ev_rs, ev_bt)
    name = "qMFHVKG" if case.mf else "qHVKG"
    kind = "posterior mean" if case.use_posterior_mean else "posterior samples"
    CHECK.record(f"{name} ({kind}{', log' if case.log else ''}) value", err_v, VALUE_TOL)
    CHECK.record(f"{name} ({kind}) gradient", err_g, GRAD_TOL)
    assert err_v <= VALUE_TOL, (case, err_v)
    assert err_g <= GRAD_TOL, (case, err_g)
    acq = pair.rs_acqf
    # t-batches are independent.
    with torch.no_grad():
        stacked = torch.as_tensor(acq(torch.cat(Xs, dim=0)))
    single = torch.cat([v.reshape(1) for v in ev_rs.values])
    CHECK.check("t-batch = single t-batches (botorch_rs)", stacked, single, 1e-12, atol=1e-300)
    # Central finite differences of the gradient (a few coordinates).
    x0, g0 = Xs[0], ev_rs.grads[0]
    scale = max(g0.abs().max().item(), 1e-12)
    worst = 0.0
    flat = x0.reshape(-1)
    for j in r.sample(range(flat.numel()), min(6, flat.numel())):
        e = torch.zeros_like(flat)
        e[j] = FD_STEP
        with torch.no_grad():
            up = torch.as_tensor(acq((flat + e).reshape(x0.shape))).item()
            down = torch.as_tensor(acq((flat - e).reshape(x0.shape))).item()
        fd = (up - down) / (2 * FD_STEP)
        worst = max(worst, abs(fd - g0.reshape(-1)[j].item()) / scale)
    CHECK.record("qHVKG gradient vs central differences", worst, FD_TOL)
    assert worst <= FD_TOL, (case, worst)

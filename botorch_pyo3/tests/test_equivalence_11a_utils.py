"""The randomized equivalence suite of phase 11a: botorch_rs's utilities
against live BoTorch over seeded random inputs and options.

Seeded generators draw

* ``safe_math``: every function on random shapes, scales, reduction
  dimensions and temperatures (values and gradients);
* ``utils.probability``: ``log_prob_normal_in``,
  ``compute_log_prob_feas_from_bounds`` with random constraint sets, the
  ``bvn`` family, ``MVNXPB`` on random covariances and bounds (dimensions
  2-7, values and gradients), ``TruncatedMultivariateNormal.log_prob`` and
  ``LinearEllipticalSliceSampler`` chains on random polytopes (BoTorch's 2 pi
  wrapping constant in float64, see ``test_parity_11a_utils.py``);
* ``IIDNormalSampler`` in random Monte-Carlo acquisition functions on random
  ``SingleTaskGP`` models, and the QMC engines;
* the test functions of phase 11a at random points;
* ``standardize`` / ``normalize`` / ``unnormalize``, the feasibility
  indicators and ``apply_constraints``.

Tolerances (the worst observed error of every quantity is printed in the
terminal summary, section "equivalence suite (phase 11a)"): values 1e-9,
gradients 1e-8 (relative).

Run with ``pytest -m equivalence``; skipped unless BoTorch is importable,
which ``BOTORCH_RS_REQUIRE_BOTORCH=1`` turns into an error.
"""

from __future__ import annotations

import random
import sys
from typing import Any

import pytest

from helpers import importorskip_botorch

importorskip_botorch()
torch = pytest.importorskip("torch")

import botorch.utils.probability.utils as bt_pu  # noqa: E402
import botorch.utils.safe_math as bt_sm  # noqa: E402
import test_parity_11a_utils as p11a  # noqa: E402
from so_acquisition_helpers import check_against_twins, twin  # noqa: E402
from trajectory_reference import portable_trajectory_grid  # noqa: E402

import botorch_rs.utils.probability.utils as rs_pu  # noqa: E402
import botorch_rs.utils.safe_math as rs_sm  # noqa: E402
from parity_helpers import Checker, ModelSpec, model_pair, values_and_grads  # noqa: E402

pytestmark = [pytest.mark.equivalence, pytest.mark.filterwarnings("ignore")]

DT = torch.float64
CHECK = Checker("equivalence suite (phase 11a)")
TOL = 1e-9
GTOL = 1e-8
SEEDS = range(12)

bt_bvn = sys.modules[__import__("botorch.utils.probability.bvn", fromlist=["x"]).__name__]
rs_bvn = sys.modules[__import__("botorch_rs.utils.probability.bvn", fromlist=["x"]).__name__]


def gen(seed: int) -> Any:
    return torch.Generator().manual_seed(seed)


grad_of = p11a.grad_of


@pytest.mark.parametrize("seed", SEEDS)
def test_safe_math_sweep(seed: int) -> None:
    r = random.Random(seed)
    shape = [r.randint(1, 4) for _ in range(r.randint(1, 3))]
    scale = r.choice([0.1, 1.0, 10.0, 100.0])
    x = torch.randn(*shape, generator=gen(seed), dtype=DT) * scale
    y = torch.randn(*shape, generator=gen(seed + 100), dtype=DT) * scale
    tau = r.choice([1e-3, 0.1, 1.0, 3.0])
    for name in p11a.UNARY:
        xi = p11a._inputs(name, x)
        CHECK.check(f"safe_math.{name}", getattr(rs_sm, name)(xi), getattr(bt_sm, name)(xi), TOL)
        (ga,) = grad_of(getattr(rs_sm, name), xi)
        (ge,) = grad_of(getattr(bt_sm, name), xi)
        CHECK.check(f"safe_math.{name} grad", ga, ge, GTOL)
    for name in p11a.TAU_FUNCS:
        CHECK.check(
            f"safe_math.{name}",
            getattr(rs_sm, name)(x, tau=tau),
            getattr(bt_sm, name)(x, tau=tau),
            TOL,
        )
        (ga,) = grad_of(lambda t, n=name: getattr(rs_sm, n)(t, tau=tau), x)
        (ge,) = grad_of(lambda t, n=name: getattr(bt_sm, n)(t, tau=tau), x)
        CHECK.check(f"safe_math.{name} grad", ga, ge, GTOL)
    for name in p11a.BINARY:
        b = y.abs() + 0.1 if name == "div" else y
        CHECK.check(
            f"safe_math.{name}", getattr(rs_sm, name)(x, b), getattr(bt_sm, name)(x, b), TOL
        )
        ga = grad_of(getattr(rs_sm, name), x, b)
        ge = grad_of(getattr(bt_sm, name), x, b)
        for u, v in zip(ga, ge, strict=True):
            CHECK.check(f"safe_math.{name} grad", u, v, GTOL)
    lo, hi = torch.minimum(x, y), torch.maximum(x, y) + 0.1
    CHECK.check("safe_math.logdiffexp", rs_sm.logdiffexp(lo, hi), bt_sm.logdiffexp(lo, hi), TOL)
    dim = r.randrange(-len(shape), len(shape))
    for name in p11a.REDUCTIONS:
        kw: dict[str, Any] = {"dim": dim}
        if name in ("smooth_amax", "smooth_amin", "fatmax", "fatmin"):
            kw["tau"] = tau
        if name in ("logsumexp", "logmeanexp", "smooth_amax", "smooth_amin", "fatmax", "fatmin"):
            kw["keepdim"] = r.random() < 0.5
        CHECK.check(
            f"safe_math.{name}", getattr(rs_sm, name)(x, **kw), getattr(bt_sm, name)(x, **kw), TOL
        )
        (ga,) = grad_of(lambda t, n=name, k=kw: getattr(rs_sm, n)(t, **k), x)
        (ge,) = grad_of(lambda t, n=name, k=kw: getattr(bt_sm, n)(t, **k), x)
        CHECK.check(f"safe_math.{name} grad", ga, ge, GTOL)


@pytest.mark.parametrize("seed", SEEDS)
def test_probability_sweep(seed: int) -> None:
    r = random.Random(seed)
    n = r.randint(1, 6)
    a = torch.randn(n, generator=gen(seed), dtype=DT) * r.choice([1.0, 5.0, 20.0])
    b = a + torch.rand(n, generator=gen(seed + 1), dtype=DT) * r.choice([1e-6, 0.1, 3.0])
    CHECK.check(
        "log_prob_normal_in", rs_pu.log_prob_normal_in(a, b), bt_pu.log_prob_normal_in(a, b), TOL
    )
    ga = grad_of(rs_pu.log_prob_normal_in, a, b)
    ge = grad_of(bt_pu.log_prob_normal_in, a, b)
    for u, v in zip(ga, ge, strict=True):
        CHECK.check("log_prob_normal_in grad", u, v, GTOL)
    m = r.randint(3, 6)
    inds = list(range(m))
    r.shuffle(inds)
    nl, nu = r.randint(0, 2), r.randint(0, 2)
    lower, upper, both = inds[:nl], inds[nl : nl + nu], inds[nl + nu : nl + nu + r.randint(0, 2)]
    means = torch.randn(4, m, generator=gen(seed + 2), dtype=DT)
    sigmas = torch.rand(4, m, generator=gen(seed + 3), dtype=DT) + 0.1
    kw = {
        "con_lower_inds": torch.tensor(lower, dtype=torch.long),
        "con_upper_inds": torch.tensor(upper, dtype=torch.long),
        "con_both_inds": torch.tensor(both, dtype=torch.long),
        "con_lower": torch.randn(len(lower), generator=gen(seed + 4), dtype=DT),
        "con_upper": torch.randn(len(upper), generator=gen(seed + 5), dtype=DT),
        "con_both": torch.sort(
            torch.randn(len(both), 2, generator=gen(seed + 6), dtype=DT), -1
        ).values,
    }
    CHECK.check(
        "compute_log_prob_feas_from_bounds",
        rs_pu.compute_log_prob_feas_from_bounds(means=means, sigmas=sigmas, **kw),
        bt_pu.compute_log_prob_feas_from_bounds(means=means, sigmas=sigmas, **kw),
        TOL,
    )
    k = 5
    rr = torch.rand(k, generator=gen(seed + 7), dtype=DT) * 1.98 - 0.99
    xl = torch.randn(k, generator=gen(seed + 8), dtype=DT) * 2
    yl = torch.randn(k, generator=gen(seed + 9), dtype=DT) * 2
    xu = xl + torch.rand(k, generator=gen(seed + 10), dtype=DT) * 3
    yu = yl + torch.rand(k, generator=gen(seed + 11), dtype=DT) * 3
    CHECK.check("bvnu", rs_bvn.bvnu(rr, xl, yl), bt_bvn.bvnu(rr, xl, yl), TOL)
    CHECK.check("bvn", rs_bvn.bvn(rr, xl, yl, xu, yu), bt_bvn.bvn(rr, xl, yl, xu, yu), TOL)
    for u, v in zip(
        rs_bvn.bvnmom(rr, xl, yl, xu, yu), bt_bvn.bvnmom(rr, xl, yl, xu, yu), strict=True
    ):
        CHECK.check("bvnmom", u, v, TOL)


@pytest.mark.parametrize("seed", SEEDS)
def test_mvnxpb_and_tmvn_sweep(seed: int) -> None:
    from botorch.utils.probability.mvnxpb import MVNXPB as BtM
    from botorch.utils.probability.truncated_multivariate_normal import (
        TruncatedMultivariateNormal as BtT,
    )

    from botorch_rs.utils.probability.mvnxpb import MVNXPB as RsM
    from botorch_rs.utils.probability.truncated_multivariate_normal import (
        TruncatedMultivariateNormal as RsT,
    )

    r = random.Random(seed)
    d = r.randint(2, 7)
    cov, bounds = p11a._mvn_problem(seed, batch=(), n=d)
    a, e = RsM(cov, bounds), BtM(cov, bounds)
    CHECK.check("MVNXPB log_prob", a.solve(), e.solve(), TOL)
    ga = grad_of(lambda c, b: RsM(c, b).solve(), cov, bounds)
    ge = grad_of(lambda c, b: BtM(c, b).solve(), cov, bounds)
    for u, v in zip(ga, ge, strict=True):
        CHECK.check("MVNXPB log_prob grad", u, v, GTOL)
    loc = torch.randn(d, generator=gen(seed), dtype=DT) * 0.3
    tb = bounds.clone()
    ta, te = RsT(loc, covariance_matrix=cov, bounds=tb), BtT(loc, covariance_matrix=cov, bounds=tb)
    v = tb[:, 0] + (tb[:, 1] - tb[:, 0]).clamp(max=10) * torch.rand(
        3, d, generator=gen(seed + 1), dtype=DT
    )
    v = torch.where(torch.isfinite(v), v, torch.zeros_like(v))
    CHECK.check("TMVN log_prob", ta.log_prob(v), te.log_prob(v), TOL)


@pytest.mark.parametrize("seed", SEEDS)
def test_lin_ess_sweep(seed: int, monkeypatch: pytest.MonkeyPatch) -> None:
    lin_ess = sys.modules[__import__("botorch.utils.probability.lin_ess", fromlist=["x"]).__name__]
    monkeypatch.setattr(lin_ess, "_twopi", p11a.EXACT_2PI)
    from botorch_rs.utils.probability.lin_ess import LinearEllipticalSliceSampler as Rs

    r = random.Random(seed)
    d = r.randint(2, 5)
    m = r.randint(1, 5)
    A = torch.randn(m, d, generator=gen(seed), dtype=DT)
    x0 = torch.randn(d, 1, generator=gen(seed + 1), dtype=DT) * 0.2
    b = A @ x0 + torch.rand(m, 1, generator=gen(seed + 2), dtype=DT) + 0.05
    kw: dict[str, Any] = {
        "num_chains": r.randint(1, 3),
        "burnin": r.randint(0, 3),
        "thinning": r.randint(0, 2),
    }
    if r.random() < 0.5:
        L = torch.randn(d, d, generator=gen(seed + 3), dtype=DT)
        kw["covariance_matrix"] = L @ L.T + 0.5 * torch.eye(d, dtype=DT)
        kw["mean"] = torch.randn(d, 1, generator=gen(seed + 4), dtype=DT) * 0.1
    out = []
    for cls in (Rs, lin_ess.LinearEllipticalSliceSampler):
        torch.manual_seed(seed)
        s = cls(inequality_constraints=(A, b), interior_point=x0, **kw)
        out.append(s.draw(40))
    CHECK.check("LinESS chains", out[0], out[1], TOL)
    assert bool((A @ out[0].T <= b).all())


MC_NAMES = [
    "qExpectedImprovement",
    "qLogExpectedImprovement",
    "qProbabilityOfImprovement",
    "qUpperConfidenceBound",
    "qSimpleRegret",
    "qNoisyExpectedImprovement",
    "qLogNoisyExpectedImprovement",
]


@pytest.mark.parametrize("seed", SEEDS)
def test_iid_sampler_sweep(seed: int) -> None:
    import botorch.acquisition as bta
    from botorch.sampling import IIDNormalSampler as BtIID

    import botorch_rs.acquisition as rsa
    from botorch_rs.sampling import IIDNormalSampler as RsIID

    r = random.Random(seed)
    spec = ModelSpec(
        n=r.choice([6, 10]),
        d=r.randint(1, 3),
        fixed_noise=r.random() < 0.5,
        normalize=r.random() < 0.5,
        kernel=r.choice(["rbf", "matern52"]),
    )
    bt, rs, X = model_pair(spec, seed)
    name = MC_NAMES[seed % len(MC_NAMES)]
    kw: dict[str, Any] = {}
    if name in ("qExpectedImprovement", "qLogExpectedImprovement", "qProbabilityOfImprovement"):
        kw["best_f"] = r.choice([0.5, 1.0, 1.25])  # float32-representable
    elif name == "qUpperConfidenceBound":
        kw["beta"] = r.choice([0.25, 2.0])
    elif name != "qSimpleRegret":
        kw.update(X_baseline=X, prune_baseline=False, cache_root=False)
    S = r.choice([16, 64])
    sseed = r.randint(0, 1000)
    a = getattr(rsa, name)(rs, sampler=RsIID(torch.Size([S]), seed=sseed), **kw)
    Xc = torch.rand(r.randint(1, 4), r.randint(1, 3), spec.d, generator=gen(seed), dtype=DT)
    if "X_baseline" in kw:
        # The joint posterior over the training points and the candidates is
        # nearly singular: compared with BoTorch's twins (see
        # so_acquisition_helpers.check_against_twins).
        def outcome(k: int) -> tuple[Any, Any]:
            e = getattr(bta, name)(twin(bt, k), sampler=BtIID(torch.Size([S]), seed=sseed), **kw)
            return values_and_grads(e, Xc)

        check_against_twins(CHECK, f"IIDNormalSampler {name}", values_and_grads(a, Xc), outcome)
        return
    e = getattr(bta, name)(bt, sampler=BtIID(torch.Size([S]), seed=sseed), **kw)
    CHECK.check(f"IIDNormalSampler {name}", a(Xc), e(Xc), TOL)
    (ga,) = grad_of(a, Xc)
    (ge,) = grad_of(e, Xc)
    CHECK.check(f"IIDNormalSampler {name} grad", ga, ge, GTOL)


@pytest.mark.parametrize("seed", SEEDS)
def test_qmc_and_test_function_sweep(seed: int) -> None:
    from botorch.sampling.qmc import NormalQMCEngine as BtN

    from botorch_rs.sampling.qmc import NormalQMCEngine as RsN

    r = random.Random(seed)
    d, inv = r.randint(1, 6), r.random() < 0.5
    a, e = RsN(d, seed=seed, inv_transform=inv), BtN(d, seed=seed, inv_transform=inv)
    for n in (r.randint(1, 9), r.randint(1, 9)):
        CHECK.check("NormalQMCEngine", a.draw(n, dtype=DT), e.draw(n, dtype=DT), TOL)
    for mod, name, kw in p11a.PROBLEMS:
        pa, pe = p11a._problem_pair(mod, name, kw)
        X = p11a._problem_points(pe, 6, seed + 1000)
        trajectory = name == "TrajectoryPlanning" and kw.get("use_smooth_interp", True)
        with portable_trajectory_grid(trajectory):
            reference = pe.evaluate_true(X)
        CHECK.check(f"{name} evaluate_true", pa.evaluate_true(X), reference, TOL)
        if hasattr(pe, "evaluate_slack_true"):
            CHECK.check(
                f"{name} evaluate_slack_true",
                pa.evaluate_slack_true(X),
                pe.evaluate_slack_true(X),
                TOL,
            )


@pytest.mark.parametrize("seed", SEEDS)
def test_transforms_and_objective_sweep(seed: int) -> None:
    import botorch.utils.objective as bo
    import botorch.utils.transforms as bt

    import botorch_rs.utils.objective as ro
    import botorch_rs.utils.transforms as rt

    r = random.Random(seed)
    shape = (r.randint(1, 3), r.randint(2, 6), r.randint(1, 3))
    Y = torch.randn(*shape, generator=gen(seed), dtype=DT) * r.choice([1e-3, 1.0, 1e3])
    CHECK.check("standardize", rt.standardize(Y), bt.standardize(Y), TOL)
    d = shape[-1]
    lo = torch.randn(d, generator=gen(seed + 1), dtype=DT)
    bounds = torch.stack([lo, lo + torch.rand(d, generator=gen(seed + 2), dtype=DT) * 3])
    CHECK.check("normalize", rt.normalize(Y, bounds), bt.normalize(Y, bounds), TOL)
    CHECK.check("unnormalize", rt.unnormalize(Y, bounds), bt.unnormalize(Y, bounds), TOL)
    samples = torch.randn(5, 3, 2, generator=gen(seed + 3), dtype=DT)
    c0, c1 = r.uniform(-1, 1), r.uniform(-1, 1)
    cons = [lambda Z: Z[..., 0] - c0, lambda Z: c1 - Z[..., 1]]
    eta = r.choice([2.0**-8, 0.125, 1.0])
    for log in (False, True):
        fat = r.random() < 0.5
        CHECK.check(
            "compute_smoothed_feasibility_indicator",
            ro.compute_smoothed_feasibility_indicator(cons, samples, eta, log=log, fat=fat),
            bo.compute_smoothed_feasibility_indicator(cons, samples, eta, log=log, fat=fat),
            TOL,
        )
    cost = r.choice([0.0, 1.5])
    CHECK.check(
        "apply_constraints",
        ro.apply_constraints(samples[..., 0], cons, samples, cost, eta),
        bo.apply_constraints(samples[..., 0], cons, samples, cost, eta),
        TOL,
    )

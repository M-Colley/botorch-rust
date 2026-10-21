"""Parity with BoTorch for the utilities of phase 11a, side by side through
the public APIs on identical inputs:

* ``botorch.utils.safe_math`` (every function: values and gradients, float64
  and float32, infinities), ``botorch.utils.probability`` (the standard-normal
  helpers, ``log_prob_normal_in``, ``compute_log_prob_feas_from_bounds``,
  ``percentile_of_score``, ``case_dispatcher``, the positional indices,
  ``swap_along_dim_``, ``bvn`` / ``bvnu`` / ``bvnmom``, ``MVNXPB`` with partial
  solves, pivots, ``augment``, batching operations and gradients,
  ``PivotedCholesky``, ``augment_cholesky``, ``block_matrix_concat``,
  ``LinearEllipticalSliceSampler`` with torch's draws,
  ``TruncatedMultivariateNormal``, ``UnifiedSkewNormal``);
* the samplers: ``IIDNormalSampler`` (its draws, posterior samples and the
  Monte-Carlo acquisition functions with it), ``ForkedRNGSampler``,
  ``get_sampler``, ``NormalQMCEngine``, ``MultivariateNormalQMCEngine``;
* the test functions of ``multi_fidelity``, ``sensitivity_analysis``, the
  mixed / binary / trajectory / constrained synthetic functions,
  ``test_functions.base`` and ``round_nearest``;
* ``botorch.utils``: ``transforms``, ``objective``, ``rounding``,
  ``constraints``, ``containers``, ``datasets``, ``low_rank``, ``multitask``,
  ``constants``, ``context_managers``, ``dispatcher``, ``evaluation``,
  ``feasible_volume``, ``sampling`` (polytope samplers, normalization,
  ``optimize_posterior_samples``), ``torch.BufferDict``; ``settings``,
  ``logging`` and the multi-objective leftovers.

Values agree to 1e-9 and gradients to 1e-8 (relative) unless stated
otherwise; float32 inputs, whose arithmetic BoTorch performs in float32 while
botorch_rs computes in float64, agree to 1e-5. Markov chains (linear
elliptical slice sampling) are compared exactly for the first steps and
against the spread of BoTorch's chains on twins with perturbed interior
points afterwards. Skipped unless BoTorch is importable.
"""

from __future__ import annotations

import math
import sys
from typing import Any

import pytest

from helpers import importorskip_botorch

importorskip_botorch()
torch = pytest.importorskip("torch")

import botorch.utils.probability.utils as bt_pu  # noqa: E402
import botorch.utils.safe_math as bt_sm  # noqa: E402
from trajectory_reference import portable_trajectory_grid  # noqa: E402

import botorch_rs.utils.probability.utils as rs_pu  # noqa: E402
import botorch_rs.utils.safe_math as rs_sm  # noqa: E402
from parity_helpers import Checker, ModelSpec, model_pair  # noqa: E402

DT = torch.float64
C = Checker("11a utils")
TOL = 1e-9
GTOL = 1e-8
F32TOL = 1e-6

bt_bvn = sys.modules[__import__("botorch.utils.probability.bvn", fromlist=["x"]).__name__]
rs_bvn = sys.modules[__import__("botorch_rs.utils.probability.bvn", fromlist=["x"]).__name__]


def gen(seed: int) -> torch.Generator:
    return torch.Generator().manual_seed(seed)


def grad_of(f: Any, *xs: Any) -> tuple[Any, ...]:
    xs = tuple(x.detach().clone().requires_grad_(True) for x in xs)
    out = f(*xs)
    if isinstance(out, tuple):
        out = sum(o.sum() for o in out)
    gs = torch.autograd.grad(out.sum(), xs, allow_unused=True)
    return tuple(torch.zeros_like(x) if g is None else g for g, x in zip(gs, xs, strict=True))


# ---------------------------------------------------------------------------
# safe_math
# ---------------------------------------------------------------------------

UNARY = ["exp", "log", "log1mexp", "log1pexp", "logexpit", "cauchy"]
TAU_FUNCS = ["fatplus", "log_fatplus", "log_softplus", "fatmoid", "log_fatmoid"]
BINARY = ["add", "sub", "mul", "div", "logplusexp", "fatmaximum", "fatminimum"]
REDUCTIONS = ["logsumexp", "logmeanexp", "smooth_amax", "smooth_amin", "fatmax", "fatmin"]


def _inputs(name: str, x: Any) -> Any:
    if name == "log1mexp":
        return -(x.abs() + 1e-3)
    if name == "log":
        return x.abs() + 1e-300
    return x


@pytest.mark.parametrize("name", UNARY + TAU_FUNCS)
@pytest.mark.parametrize("seed", [0, 1])
def test_safe_math_unary(name: str, seed: int) -> None:
    x = torch.randn(7, 5, generator=gen(seed), dtype=DT) * 4
    x = _inputs(name, x)
    kw = {"tau": 0.3} if name in TAU_FUNCS else {}
    a = getattr(rs_sm, name)(x, **kw)
    e = getattr(bt_sm, name)(x, **kw)
    C.check(f"safe_math.{name}", a, e, TOL)
    (ga,) = grad_of(lambda t: getattr(rs_sm, name)(t, **kw), x)
    (ge,) = grad_of(lambda t: getattr(bt_sm, name)(t, **kw), x)
    C.check(f"safe_math.{name} grad", ga, ge, GTOL)
    x32 = x.float()
    C.check(
        f"safe_math.{name} float32",
        getattr(rs_sm, name)(x32, **kw).double(),
        getattr(bt_sm, name)(x32, **kw).double(),
        F32TOL,
    )


def test_safe_math_special_values() -> None:
    inf = math.inf
    a = torch.tensor([inf, inf, -inf, 0.0, 0.0, 3.0, -2.0, 1e308], dtype=DT)
    b = torch.tensor([-inf, inf, -inf, 0.0, inf, 0.0, -inf, 1e308], dtype=DT)
    for name in ("add", "sub", "mul", "div"):
        C.check(
            f"safe_math.{name} infinities",
            getattr(rs_sm, name)(a, b),
            getattr(bt_sm, name)(a, b),
            0.0,
        )
    big = torch.tensor([800.0, 709.0, -800.0, 0.0, 1e-320], dtype=DT)
    C.check("safe_math.exp clip", rs_sm.exp(big), bt_sm.exp(big), TOL)
    C.check("safe_math.log clip", rs_sm.log(big.abs()), bt_sm.log(big.abs()), TOL)
    x = torch.tensor([[1.0, inf, 2.0], [-inf, -inf, -inf], [3.0, -inf, 1.0]], dtype=DT)
    for name in ("logsumexp", "fatmax", "smooth_amax"):
        C.check(
            f"safe_math.{name} infinities",
            getattr(rs_sm, name)(x, dim=-1),
            getattr(bt_sm, name)(x, dim=-1),
            TOL,
        )
        (ga,) = grad_of(lambda t, n=name: getattr(rs_sm, n)(t, dim=-1), x)
        (ge,) = grad_of(lambda t, n=name: getattr(bt_sm, n)(t, dim=-1), x)
        # BoTorch's fatmax gradient is NaN for a row with a -inf element below
        # a finite maximum (botorch_rs: the limit, 0 for that element).
        ok = torch.isfinite(ge).all(-1)
        C.check(f"safe_math.{name} infinities grad", ga[ok], ge[ok], GTOL)
        assert torch.isfinite(ga).all() or name != "fatmax"
    la = torch.tensor([-inf, -3.0, -1e-20, -1.0], dtype=DT)
    lb = torch.tensor([-inf, -1.0, 0.0, -1.0 + 1e-12], dtype=DT)
    C.check("safe_math.logdiffexp edge", rs_sm.logdiffexp(la, lb), bt_sm.logdiffexp(la, lb), TOL)


@pytest.mark.parametrize("name", BINARY)
def test_safe_math_binary(name: str) -> None:
    a = torch.randn(4, 1, 3, generator=gen(3), dtype=DT) * 2
    b = torch.randn(5, 3, generator=gen(4), dtype=DT) * 2
    C.check(f"safe_math.{name}", getattr(rs_sm, name)(a, b), getattr(bt_sm, name)(a, b), TOL)
    ga = grad_of(lambda s, t: getattr(rs_sm, name)(s, t), a, b)
    ge = grad_of(lambda s, t: getattr(bt_sm, name)(s, t), a, b)
    for k in range(2):
        C.check(f"safe_math.{name} grad", ga[k], ge[k], GTOL)


def test_safe_math_logdiffexp() -> None:
    lb = torch.randn(6, generator=gen(5), dtype=DT)
    la = lb - torch.rand(6, generator=gen(6), dtype=DT) * 5 - 1e-6
    C.check("safe_math.logdiffexp", rs_sm.logdiffexp(la, lb), bt_sm.logdiffexp(la, lb), TOL)
    ga = grad_of(rs_sm.logdiffexp, la, lb)
    ge = grad_of(bt_sm.logdiffexp, la, lb)
    for k in range(2):
        C.check("safe_math.logdiffexp grad", ga[k], ge[k], GTOL)


@pytest.mark.parametrize("name", REDUCTIONS)
@pytest.mark.parametrize("dim", [-1, 0, (0, 2), (1,)])
@pytest.mark.parametrize("keepdim", [False, True])
def test_safe_math_reductions(name: str, dim: Any, keepdim: bool) -> None:
    x = torch.randn(3, 4, 5, generator=gen(7), dtype=DT) * 3
    kw: dict[str, Any] = {"dim": dim, "keepdim": keepdim}
    if name in ("smooth_amax", "smooth_amin", "fatmax", "fatmin"):
        kw["tau"] = 0.5
    a = getattr(rs_sm, name)(x, **kw)
    e = getattr(bt_sm, name)(x, **kw)
    C.check(f"safe_math.{name}", a, e, TOL)
    (ga,) = grad_of(lambda t: getattr(rs_sm, name)(t, **kw), x)
    (ge,) = grad_of(lambda t: getattr(bt_sm, name)(t, **kw), x)
    C.check(f"safe_math.{name} grad", ga, ge, GTOL)


def test_safe_math_sigmoid_and_tensor_tau() -> None:
    x = torch.randn(9, generator=gen(8), dtype=DT) * 10
    for log in (False, True):
        for fat in (False, True):
            C.check(
                "safe_math.sigmoid",
                rs_sm.sigmoid(x, log=log, fat=fat),
                bt_sm.sigmoid(x, log=log, fat=fat),
                TOL,
            )
            (ga,) = grad_of(lambda t, lg=log, f=fat: rs_sm.sigmoid(t, log=lg, fat=f), x)
            (ge,) = grad_of(lambda t, lg=log, f=fat: bt_sm.sigmoid(t, log=lg, fat=f), x)
            C.check("safe_math.sigmoid grad", ga, ge, GTOL)
    tau = torch.tensor([0.1, 0.5, 2.0], dtype=DT)
    y = torch.randn(4, 3, generator=gen(9), dtype=DT)
    # (BoTorch's log_softplus rejects a tensor tau with several elements.)
    for name in ("fatplus", "fatmoid", "log_fatplus", "log_fatmoid"):
        C.check(
            f"safe_math.{name} tensor tau",
            getattr(rs_sm, name)(y, tau=tau),
            getattr(bt_sm, name)(y, tau=tau),
            TOL,
        )
        ga = grad_of(lambda a, t, n=name: getattr(rs_sm, n)(a, tau=t), y, tau)
        ge = grad_of(lambda a, t, n=name: getattr(bt_sm, n)(a, tau=t), y, tau)
        for u, v in zip(ga, ge, strict=True):
            C.check(f"safe_math.{name} tensor tau grad", u, v, GTOL)
    import botorch.utils.rounding as br

    import botorch_rs.utils.rounding as rr

    ga = grad_of(lambda a, t: rr.approximate_round(a, tau=t), y, tau)
    ge = grad_of(lambda a, t: br.approximate_round(a, tau=t), y, tau)
    for u, v in zip(ga, ge, strict=True):
        C.check("approximate_round tensor tau grad", u, v, GTOL)
    t0 = torch.tensor(0.3, dtype=DT)
    for name in ("smooth_amax", "smooth_amin", "fatmax", "fatmin"):
        C.check(
            f"safe_math.{name} 0-dim tau",
            getattr(rs_sm, name)(y, dim=-1, tau=t0),
            getattr(bt_sm, name)(y, dim=-1, tau=t0),
            TOL,
        )
        ga = grad_of(lambda a, t, n=name: getattr(rs_sm, n)(a, dim=-1, tau=t), y, t0)
        ge = grad_of(lambda a, t, n=name: getattr(bt_sm, n)(a, dim=-1, tau=t), y, t0)
        for u, v in zip(ga, ge, strict=True):
            C.check(f"safe_math.{name} 0-dim tau grad", u, v, GTOL)
    for name in ("fatmaximum", "fatminimum"):
        z = torch.randn(4, 3, generator=gen(10), dtype=DT)
        ga = grad_of(lambda a, b, t, n=name: getattr(rs_sm, n)(a, b, tau=t), y, z, t0)
        ge = grad_of(lambda a, b, t, n=name: getattr(bt_sm, n)(a, b, tau=t), y, z, t0)
        for u, v in zip(ga, ge, strict=True):
            C.check(f"safe_math.{name} 0-dim tau grad", u, v, GTOL)
    # log_softplus: BoTorch passes beta = 1 / tau to softplus, which needs a
    # number, so a tau that requires gradients raises a TypeError in BoTorch;
    # botorch_rs's derivative is compared with central differences of
    # BoTorch's values (step 1e-6: truncation error ~1e-12, rounding ~1e-10).
    ga = grad_of(lambda a, t: rs_sm.log_softplus(a, tau=t), y, t0)
    (ge,) = grad_of(lambda a: bt_sm.log_softplus(a, tau=0.3), y)
    C.check("safe_math.log_softplus 0-dim tau grad x", ga[0], ge, GTOL)
    h = 1e-6
    fd = (bt_sm.log_softplus(y, tau=t0 + h).sum() - bt_sm.log_softplus(y, tau=t0 - h).sum()) / (
        2 * h
    )
    C.check("safe_math.log_softplus tau grad vs central differences", ga[1], fd, 1e-8)


# ---------------------------------------------------------------------------
# probability.utils
# ---------------------------------------------------------------------------

PROB_UNARY = [
    "ndtr",
    "phi",
    "log_phi",
    "log_ndtr",
    "log_erfc",
    "log_erfcx",
    "standard_normal_log_hazard",
]


@pytest.mark.parametrize("name", PROB_UNARY)
def test_probability_unary(name: str) -> None:
    x = torch.cat(
        [
            torch.randn(20, generator=gen(10), dtype=DT) * 6,
            torch.tensor([-1e3, -40.0, -8.0, 0.0, 8.0, 30.0, 1e3], dtype=DT),
        ]
    )
    a = getattr(rs_pu, name)(x)
    e = getattr(bt_pu, name)(x)
    C.check(f"probability.{name}", a, e, TOL)
    finite = torch.isfinite(e) & (x.abs() < 100)
    (ga,) = grad_of(getattr(rs_pu, name), x)
    (ge,) = grad_of(getattr(bt_pu, name), x)
    C.check(f"probability.{name} grad", ga[finite], ge[finite], GTOL)


def test_log_prob_normal_in_and_feasibility() -> None:
    a = torch.randn(30, generator=gen(11), dtype=DT) * 5
    b = a + torch.rand(30, generator=gen(12), dtype=DT) * 3 + 1e-3
    C.check(
        "probability.log_prob_normal_in",
        rs_pu.log_prob_normal_in(a, b),
        bt_pu.log_prob_normal_in(a, b),
        TOL,
    )
    ga = grad_of(rs_pu.log_prob_normal_in, a, b)
    ge = grad_of(bt_pu.log_prob_normal_in, a, b)
    for k in range(2):
        C.check("probability.log_prob_normal_in grad", ga[k], ge[k], GTOL)
    means = torch.randn(5, 4, generator=gen(13), dtype=DT)
    sigmas = torch.rand(5, 4, generator=gen(14), dtype=DT) + 0.2
    args = (
        torch.tensor([0, 2]),
        torch.tensor([1, 3]),
        torch.tensor([2]),
        torch.tensor([-0.5, 0.1], dtype=DT),
        torch.tensor([0.3, 1.0], dtype=DT),
        torch.tensor([[-1.0, 0.8]], dtype=DT),
    )
    a_ = rs_pu.compute_log_prob_feas_from_bounds(*args, means, sigmas)
    e_ = bt_pu.compute_log_prob_feas_from_bounds(*args, means, sigmas)
    C.check("probability.compute_log_prob_feas_from_bounds", a_, e_, TOL)
    ga = grad_of(lambda m, s: rs_pu.compute_log_prob_feas_from_bounds(*args, m, s), means, sigmas)
    ge = grad_of(lambda m, s: bt_pu.compute_log_prob_feas_from_bounds(*args, m, s), means, sigmas)
    for k in range(2):
        C.check("probability.compute_log_prob_feas_from_bounds grad", ga[k], ge[k], GTOL)


def test_percentile_case_dispatcher_and_indices() -> None:
    data = torch.randint(0, 5, (3, 6, 2), generator=gen(15)).to(DT)
    data[0, 2, 1] = float("nan")
    score = torch.randint(0, 5, (3, 1, 2), generator=gen(16)).to(DT)
    score[2, 0, 0] = float("nan")
    a = rs_pu.percentile_of_score(data, score, dim=-2)
    e = bt_pu.percentile_of_score(data, score, dim=-2)
    assert a.dtype == e.dtype
    undefined = data.isnan().any(dim=-2) | score.squeeze(-2).isnan()
    assert torch.equal(a.isnan(), undefined) and torch.equal(e.isnan(), undefined)
    C.record(
        "probability.percentile_of_score: NaN input propagation (domain behavior only)",
        float(undefined.sum()),
        None,
    )
    C.check("probability.percentile_of_score: defined inputs", a[~undefined], e[~undefined], 0.0)
    x = torch.randn(6, generator=gen(17), dtype=DT)
    out_a, out_e = torch.empty_like(x), torch.empty_like(x)
    for out, mod in ((out_a, rs_pu), (out_e, bt_pu)):
        mod.case_dispatcher(
            out=out,
            cases=((lambda: x < -0.5, lambda m: -1.0), (lambda: x > 0.5, lambda m: x[m] * 2)),
            default=lambda m: x[m] ** 2,
        )
    C.check("probability.case_dispatcher", out_a, out_e, 0.0)
    for shape, dim in (((2, 3, 4), 1), ((5,), 0), ((2, 2, 3), -1)):
        ga = list(rs_pu.gen_positional_indices(torch.Size(shape), dim))
        ge = list(bt_pu.gen_positional_indices(torch.Size(shape), dim))
        assert all(torch.equal(p, q) for p, q in zip(ga, ge, strict=True))
        assert torch.equal(
            rs_pu.build_positional_indices(torch.Size(shape), dim),
            bt_pu.build_positional_indices(torch.Size(shape), dim),
        )
    v = torch.randn(3, 4, 2, generator=gen(18), dtype=DT)
    i = torch.tensor([0, 1, 3])
    j = torch.tensor([2, 1, 0])
    va, ve = v.clone(), v.clone()
    rs_pu.swap_along_dim_(va, i, j, dim=1)
    bt_pu.swap_along_dim_(ve, i, j, dim=1)
    assert torch.equal(va, ve)
    va, ve = v.clone(), v.clone()
    rs_pu.swap_along_dim_(va, 0, 2, dim=-2)
    bt_pu.swap_along_dim_(ve, 0, 2, dim=-2)
    assert torch.equal(va, ve)


# ---------------------------------------------------------------------------
# bvn
# ---------------------------------------------------------------------------


def _bvn_inputs(seed: int, n: int = 40) -> tuple[Any, ...]:
    g = gen(seed)
    r = torch.rand(n, generator=g, dtype=DT) * 1.98 - 0.99
    r[::7] = torch.tensor([0.95, -0.96, 0.0, 0.5, -0.3, 0.99], dtype=DT)[: len(r[::7])]
    xl = torch.randn(n, generator=g, dtype=DT) * 2
    yl = torch.randn(n, generator=g, dtype=DT) * 2
    xu = xl + torch.rand(n, generator=g, dtype=DT) * 3
    yu = yl + torch.rand(n, generator=g, dtype=DT) * 3
    xl[::5] = -float("inf")
    yu[::6] = float("inf")
    return r, xl, yl, xu, yu


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_bvn(seed: int) -> None:
    r, xl, yl, xu, yu = _bvn_inputs(seed)
    C.check("bvn.bvn", rs_bvn.bvn(r, xl, yl, xu, yu), bt_bvn.bvn(r, xl, yl, xu, yu), TOL)
    C.check("bvn.bvnu", rs_bvn.bvnu(r, xl, yu), bt_bvn.bvnu(r, xl, yu), TOL)
    for a, e in zip(
        rs_bvn.bvnmom(r, xl, yl, xu, yu), bt_bvn.bvnmom(r, xl, yl, xu, yu), strict=True
    ):
        C.check("bvn.bvnmom", a, e, TOL)
    # Gradients where BoTorch's are finite (see the draft for its NaN cases).
    ga = grad_of(rs_bvn.bvn, r, xl, yl, xu, yu)
    ge = grad_of(bt_bvn.bvn, r, xl, yl, xu, yu)
    # The source's r==0 shortcut drops correlation sensitivity. Correct only
    # that entry with Plackett's independent bivariate density identity;
    # test_torch_bvn.py reproduces the unmodified reference defect directly.
    ge = list(ge)
    independent_derivative = (
        bt_pu.phi(xl) * bt_pu.phi(yl)
        - bt_pu.phi(xl) * bt_pu.phi(yu)
        - bt_pu.phi(xu) * bt_pu.phi(yl)
        + bt_pu.phi(xu) * bt_pu.phi(yu)
    )
    ge[0] = torch.where(r == 0, independent_derivative, ge[0])
    ok = torch.stack([torch.isfinite(g) for g in ge]).all(0)
    for k in range(5):
        C.check("bvn.bvn grad", ga[k][ok], ge[k][ok], GTOL)


# ---------------------------------------------------------------------------
# MVNXPB, PivotedCholesky, augment_cholesky
# ---------------------------------------------------------------------------


def _mvn_problem(seed: int, batch: tuple[int, ...] = (3,), n: int = 5) -> tuple[Any, Any]:
    g = gen(seed)
    L = torch.randn(*batch, n, n, generator=g, dtype=DT)
    cov = L @ L.mT + 0.5 * torch.eye(n, dtype=DT)
    lo = torch.randn(*batch, n, generator=g, dtype=DT) - 1
    up = lo + torch.rand(*batch, n, generator=g, dtype=DT) * 2 + 0.1
    bounds = torch.stack([lo, up], -1)
    bounds[..., 0, 0] = -float("inf")
    bounds[..., -1, 1] = float("inf")
    return cov, bounds


def _check_state(name: str, a: Any, e: Any, tol: float = TOL) -> None:
    assert a.step == e.step and a.piv_chol.step == e.piv_chol.step
    assert torch.equal(a.perm, e.perm), name
    assert torch.equal(a.piv_chol.perm, e.piv_chol.perm), name
    C.check(f"{name} bounds", a.bounds, e.bounds, tol)
    C.check(f"{name} tril", a.piv_chol.tril, e.piv_chol.tril, tol)
    # Entries beyond the completed solver step are explicitly initialized to
    # NaN. Verify that state contract from the step count, then compare only
    # computed entries numerically; an unexpected computed NaN must fail.
    uncomputed = torch.arange(e.plug_ins.shape[-1], device=e.plug_ins.device) >= e.step
    uncomputed = uncomputed.expand_as(e.plug_ins)
    assert torch.equal(a.plug_ins.isnan(), uncomputed), name
    assert torch.equal(e.plug_ins.isnan(), uncomputed), name
    C.record(
        f"{name} plug_ins: uncomputed sentinel count (state behavior only)",
        float(uncomputed.sum()),
        None,
    )
    if e.step:
        C.check(
            f"{name} plug_ins: computed entries",
            a.plug_ins[~uncomputed],
            e.plug_ins[~uncomputed],
            1e-8,
        )
    C.check(f"{name} log_prob", a.log_prob, e.log_prob, tol)


def test_mvnxpb_state_rejects_shared_nan_in_computed_entries() -> None:
    from types import SimpleNamespace

    from botorch_rs.utils.probability.mvnxpb import MVNXPB

    covariance, bounds = _mvn_problem(400, n=3)
    actual = MVNXPB(covariance, bounds)
    actual.solve(num_steps=2)
    state = SimpleNamespace(
        step=actual.step,
        piv_chol=actual.piv_chol,
        perm=actual.perm,
        bounds=actual.bounds,
        plug_ins=actual.plug_ins.clone(),
        log_prob=actual.log_prob,
    )
    state.plug_ins[..., 0] = float("nan")
    with pytest.raises(AssertionError, match="shared computed NaN"):
        _check_state("shared computed NaN", state, state)


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
@pytest.mark.parametrize("n", [1, 2, 5, 6])
def test_mvnxpb_solve(seed: int, n: int) -> None:
    from botorch.utils.probability.mvnxpb import MVNXPB as Bt

    from botorch_rs.utils.probability.mvnxpb import MVNXPB as Rs

    cov, bounds = _mvn_problem(seed, n=n)
    a, e = Rs(cov, bounds), Bt(cov, bounds)
    _check_state("MVNXPB init", a, e)
    C.check("MVNXPB log_prob", a.solve(), e.solve(), TOL)
    _check_state("MVNXPB solved", a, e)
    if n > 2:
        a, e = Rs(cov, bounds), Bt(cov, bounds)
        a.solve(num_steps=3)
        e.solve(num_steps=3)
        _check_state("MVNXPB partial", a, e)
        C.check("MVNXPB log_prob_extra", a.log_prob_extra, e.log_prob_extra, TOL)
        C.check("MVNXPB continued", a.solve(), e.solve(), TOL)


def test_mvnxpb_gradients_and_batch_ops() -> None:
    from botorch.utils.probability.mvnxpb import MVNXPB as Bt

    from botorch_rs.utils.probability.mvnxpb import MVNXPB as Rs

    cov, bounds = _mvn_problem(4, batch=(4,))
    # Gradients member by member (BoTorch's batched gradients are wrong when
    # members pivot differently; see docs/drafts/11a-utils.md).
    for b in range(4):
        ga = grad_of(lambda c, bd: Rs(c, bd).solve(), cov[b : b + 1], bounds[b : b + 1])
        ge = grad_of(lambda c, bd: Bt(c, bd).solve(), cov[b : b + 1], bounds[b : b + 1])
        C.check("MVNXPB grad covariance", ga[0], ge[0], GTOL)
        C.check("MVNXPB grad bounds", ga[1], ge[1], GTOL)
    a, e = Rs(cov, bounds), Bt(cov, bounds)
    a.solve(num_steps=2)
    e.solve(num_steps=2)
    _check_state("MVNXPB getitem", a[1:3], e[1:3])
    _check_state("MVNXPB concat", a.concat(a[:1], dim=0), e.concat(e[:1], dim=0))
    _check_state("MVNXPB expand", Rs(cov[0], bounds[0]).expand(3), Bt(cov[0], bounds[0]).expand(3))
    _check_state("MVNXPB clone", a.clone(), e.clone())
    _check_state("MVNXPB detach", a.detach(), e.detach())
    assert torch.equal(a.select_pivot(), e.select_pivot())
    piv = torch.tensor([4, 2, 3, 2])
    a.pivot_(piv)
    e.pivot_(piv)
    _check_state("MVNXPB pivot_", a, e)
    C.check("MVNXPB after pivot", a.solve(), e.solve(), TOL)


@pytest.mark.parametrize("disable_pivoting", [False, True])
def test_mvnxpb_augment(disable_pivoting: bool) -> None:
    from botorch.utils.probability.mvnxpb import MVNXPB as Bt

    from botorch_rs.utils.probability.mvnxpb import MVNXPB as Rs

    cov, bounds = _mvn_problem(5, batch=(3,), n=6)
    n = 3
    init_b = Bt(cov[..., :n, :n], bounds[..., :n, :])
    init_b.solve()
    init_r = Rs(cov[..., :n, :n], bounds[..., :n, :])
    init_r.solve()
    _check_state("MVNXPB augment init", init_r, init_b)
    # The cross-covariance in the order of the solved variables.
    cross = cov[..., n:, :n].gather(-1, init_b.perm.unsqueeze(-2).expand(-1, 6 - n, -1))
    ab = init_b.augment(
        cov[..., n:, n:], bounds[..., n:, :], cross, disable_pivoting=disable_pivoting
    )
    ar = init_r.augment(
        cov[..., n:, n:], bounds[..., n:, :], cross, disable_pivoting=disable_pivoting
    )
    _check_state("MVNXPB augmented", ar, ab)
    C.check("MVNXPB augmented log_prob", ar.solve(), ab.solve(), 1e-8)


def test_pivoted_cholesky_and_linalg() -> None:
    from botorch.utils.probability import linalg as bl

    from botorch_rs.utils.probability import linalg as rl

    g = gen(20)
    A = torch.randn(2, 5, 5, generator=g, dtype=DT)
    K = A @ A.mT + torch.eye(5, dtype=DT)
    perm = torch.arange(5).expand(2, 5).clone()
    diag = K.diagonal(dim1=-2, dim2=-1).sqrt().clone()
    pa = rl.PivotedCholesky(step=0, tril=K.tril().clone(), perm=perm.clone(), diag=diag.clone())
    pe = bl.PivotedCholesky(step=0, tril=K.tril().clone(), perm=perm.clone(), diag=diag.clone())
    for piv in ([3, 1], [2, 4], [2, 2], [4, 3]):
        p = torch.tensor(piv)
        pa.pivot_(p)
        pe.pivot_(p)
        pa.update_()
        pe.update_()
        C.check("PivotedCholesky tril", pa.tril, pe.tril, TOL)
        assert torch.equal(pa.perm, pe.perm)
        C.check("PivotedCholesky diag", pa.diag, pe.diag, 0.0)
    na = 3
    Laa = torch.linalg.cholesky(K[..., :na, :na])
    args = {"Laa": Laa, "Kbb": K[..., na:, na:], "Kba": K[..., na:, :na]}
    C.check("augment_cholesky", rl.augment_cholesky(**args), bl.augment_cholesky(**args), TOL)
    C.check(
        "augment_cholesky jitter",
        rl.augment_cholesky(**args, jitter=1e-3),
        bl.augment_cholesky(**args, jitter=1e-3),
        TOL,
    )
    Lba = torch.linalg.solve_triangular(Laa.mT, K[..., na:, :na], left=False, upper=True)
    C.check(
        "augment_cholesky Lba",
        rl.augment_cholesky(Laa=Laa, Kbb=K[..., na:, na:], Lba=Lba),
        bl.augment_cholesky(Laa=Laa, Kbb=K[..., na:, na:], Lba=Lba),
        TOL,
    )
    blocks = ([K[0, :2, :2], K[:, :2, 2:]], [K[:, 2:, :2], K[1, 2:, 2:]])
    C.check(
        "block_matrix_concat", rl.block_matrix_concat(blocks), bl.block_matrix_concat(blocks), 0.0
    )


# ---------------------------------------------------------------------------
# Linear elliptical slice sampling, truncated and unified skew normals
# ---------------------------------------------------------------------------

LIN_A = torch.tensor([[1.0, 1.0, 0.0], [-1.0, 0.5, 0.2], [0.0, -0.3, 1.0]], dtype=DT)
LIN_B = torch.tensor([[1.0], [0.8], [0.9]], dtype=DT)
LIN_BOUNDS = torch.tensor([[-1.0, -2.0, -1.5], [2.0, 1.0, 1.0]], dtype=DT)
LIN_X0 = torch.tensor([[0.1], [0.0], [-0.2]], dtype=DT)
LIN_COV = torch.tensor([[1.0, 0.3, 0.1], [0.3, 0.5, -0.1], [0.1, -0.1, 0.8]], dtype=DT)
LIN_MEAN = torch.tensor([[0.2], [-0.1], [0.05]], dtype=DT)

#: BoTorch wraps negative intersection angles with ``theta.lt(0) * _twopi``:
#: a bool tensor times a Python float has torch's default dtype, so 2 pi is
#: rounded to float32 (6.2831854820251465, 1.7e-7 too large) and every
#: wrapped angle (an arc end) misses the constraint boundary by 1.7e-7 rad.
#: botorch_rs wraps with 2 pi in float64; the chains are compared with
#: BoTorch's wrapping constant replaced by a float64 tensor (then they agree
#: to 1e-12 over 200 steps), and the angles with the exact boundary.
EXACT_2PI = torch.tensor(2.0 * math.pi, dtype=DT)


@pytest.fixture
def exact_twopi(monkeypatch: pytest.MonkeyPatch) -> None:
    lin_ess = sys.modules[__import__("botorch.utils.probability.lin_ess", fromlist=["x"]).__name__]
    monkeypatch.setattr(lin_ess, "_twopi", EXACT_2PI)


def _lin_draw(cls: Any, seed: int, x0: Any, n: int, **kw: Any) -> Any:
    torch.manual_seed(seed)
    s = cls(inequality_constraints=(LIN_A, LIN_B), bounds=LIN_BOUNDS, interior_point=x0, **kw)
    return s.draw(n), s


LIN_CASES: list[dict[str, Any]] = [
    {},
    {"covariance_matrix": LIN_COV, "mean": LIN_MEAN},
    {"covariance_root": torch.linalg.cholesky(LIN_COV)},
    {"num_chains": 3, "burnin": 4, "thinning": 2},
    {"fixed_indices": [1], "covariance_root": torch.diag(torch.tensor([1.0, 2.0, 0.5], dtype=DT))},
]


@pytest.mark.parametrize("kw", LIN_CASES, ids=["standard", "cov_mean", "root", "chains", "fixed"])
def test_linear_elliptical_slice_sampler(kw: dict[str, Any], exact_twopi: None) -> None:
    from botorch.utils.probability.lin_ess import LinearEllipticalSliceSampler as Bt

    from botorch_rs.utils.probability.lin_ess import LinearEllipticalSliceSampler as Rs

    n = 100
    a, sa = _lin_draw(Rs, 7, LIN_X0, n, **kw)
    e, se = _lin_draw(Bt, 7, LIN_X0, n, **kw)
    assert a.shape == e.shape
    assert sa.lifetime_samples == se.lifetime_samples
    C.check("LinESS chains", a, e, TOL)
    C.check("LinESS _Az", sa._Az, se._Az, TOL)
    C.check("LinESS _bz", sa._bz, se._bz, TOL)
    C.check("LinESS state", sa._x, se._x, TOL)
    pts = torch.randn(3, 6, generator=gen(30), dtype=DT)
    assert torch.equal(sa._is_feasible(pts), se._is_feasible(pts))
    torch.manual_seed(3)
    C.check("LinESS step", sa.step(), (torch.manual_seed(3), se.step())[1], TOL)


def test_linear_elliptical_slice_sampler_arc_ends() -> None:
    """The arc ends lie on the constraint boundaries (BoTorch's wrapped ends
    are off by the float32 rounding of 2 pi)."""
    from botorch.utils.probability.lin_ess import LinearEllipticalSliceSampler as Bt

    from botorch_rs.utils.probability.lin_ess import LinearEllipticalSliceSampler as Rs

    for cls in (Rs, Bt):
        torch.manual_seed(7)
        s = cls(inequality_constraints=(LIN_A, LIN_B), bounds=LIN_BOUNDS, interior_point=LIN_X0)
        s.draw(4)
        z, Az, bz = s._z, s._Az, s._bz
        nu = torch.randn(3, 1, dtype=DT)
        if cls is Rs:
            data, shape = s._rs.arc_ends(nu.reshape(-1).tolist())
            theta = torch.tensor(list(data), dtype=DT).reshape(shape)
        else:
            alpha, beta = s._find_intersection_angles(nu)
            theta = torch.cat([alpha, beta], -1)
        p, q = Az @ z, Az @ nu
        has = bz / (p**2 + q**2).sqrt() < 1
        res = (p * theta.cos() + q * theta.sin() - bz)[has.expand_as(theta)].abs().max().item()
        if cls is Rs:
            C.check_bound("LinESS arc ends off the boundary", res, 1e-14)
        else:
            C.record("BoTorch LinESS arc ends off the boundary", res, None)  # type: ignore[arg-type]


def test_truncated_and_unified_skew_normal(exact_twopi: None) -> None:
    from botorch.utils.probability.lin_ess import LinearEllipticalSliceSampler as BtL
    from botorch.utils.probability.truncated_multivariate_normal import (
        TruncatedMultivariateNormal as BtT,
    )
    from botorch.utils.probability.unified_skew_normal import UnifiedSkewNormal as BtU
    from torch.distributions import MultivariateNormal

    from botorch_rs.utils.probability.lin_ess import LinearEllipticalSliceSampler as RsL
    from botorch_rs.utils.probability.truncated_multivariate_normal import (
        TruncatedMultivariateNormal as RsT,
    )
    from botorch_rs.utils.probability.unified_skew_normal import UnifiedSkewNormal as RsU

    loc = torch.tensor([0.1, -0.2, 0.3], dtype=DT)
    L = torch.tensor([[1.0, 0, 0], [0.3, 0.8, 0], [-0.2, 0.1, 0.6]], dtype=DT)
    cov = L @ L.T
    bounds = torch.tensor([[-1.0, 1.0], [-0.5, 2.0], [-2.0, 0.4]], dtype=DT)
    ta = RsT(loc, covariance_matrix=cov, bounds=bounds)
    te = BtT(loc, covariance_matrix=cov, bounds=bounds)
    assert isinstance(ta, MultivariateNormal)
    v = torch.tensor([[0.0, 0.1, 0.0], [0.5, 1.0, -1.0], [3.0, 0.0, 0.0]], dtype=DT)
    C.check("TMVN log_prob", ta.log_prob(v), te.log_prob(v), TOL)
    C.check("TMVN log_partition", ta.log_partition, te.log_partition, TOL)
    (ga,) = grad_of(lambda c: RsT(loc, covariance_matrix=c, bounds=bounds).log_prob(v[:2]), cov)
    (ge,) = grad_of(lambda c: BtT(loc, covariance_matrix=c, bounds=bounds).log_prob(v[:2]), cov)
    C.check("TMVN log_prob grad covariance", ga, ge, GTOL)
    A = torch.cat([-torch.eye(3, dtype=DT), torch.eye(3, dtype=DT)])
    b = torch.cat([loc - bounds[:, 0], bounds[:, 1] - loc]).unsqueeze(-1)
    ip = torch.zeros(3, 1, dtype=DT)
    samples = []
    for T, S, t in ((RsT, RsL, ta), (BtT, BtL, te)):
        torch.manual_seed(5)
        sampler = S(inequality_constraints=(A, b), interior_point=ip, covariance_root=t.scale_tril)
        dist = T(loc, covariance_matrix=cov, bounds=bounds, sampler=sampler)
        samples.append(dist.rsample(torch.Size([4])))
    C.check("TMVN rsample", samples[0], samples[1], TOL)
    loc_b = torch.stack([loc, loc + 0.1])
    tb_a = RsT(loc_b, covariance_matrix=cov, bounds=bounds)
    tb_e = BtT(loc_b, covariance_matrix=cov, bounds=bounds)
    C.check("TMVN batched log_partition", tb_a.log_partition, tb_e.log_partition, TOL)
    C.check("TMVN expand", ta.expand((2,)).log_prob(v[:1]), te.expand((2,)).log_prob(v[:1]), TOL)
    gauss = MultivariateNormal(
        torch.tensor([0.5, -0.5], dtype=DT),
        covariance_matrix=torch.tensor([[1.0, 0.2], [0.2, 0.9]], dtype=DT),
    )
    cross = torch.tensor([[0.1, 0.05], [0.0, 0.2], [-0.1, 0.1]], dtype=DT)
    ua = RsU(RsT(loc, covariance_matrix=cov, bounds=bounds), gauss, cross)
    ue = BtU(BtT(loc, covariance_matrix=cov, bounds=bounds), gauss, cross)
    y = torch.tensor([[0.1, 0.2], [1.0, -1.0]], dtype=DT)
    C.check("USN log_prob", ua.log_prob(y), ue.log_prob(y), TOL)
    C.check("USN covariance_matrix", ua.covariance_matrix, ue.covariance_matrix, TOL)
    C.check("USN scale_tril", ua.scale_tril, ue.scale_tril, TOL)


# ---------------------------------------------------------------------------
# Samplers
# ---------------------------------------------------------------------------


def test_iid_draws_match_torch() -> None:
    from botorch_rs import _botorch_rs as _rs

    for seed in (0, 1, 1234, 2**40 + 3):
        for n in (1, 2, 15, 16, 17, 33, 1000):
            torch.manual_seed(seed)
            e = torch.randn(n, dtype=DT)
            assert torch.equal(torch.tensor(list(_rs.torch_randn(seed, n)[0]), dtype=DT), e)
            torch.manual_seed(seed)
            e = torch.rand(n, dtype=DT)
            assert torch.equal(torch.tensor(list(_rs.torch_rand(seed, n, False)[0]), dtype=DT), e)
            torch.manual_seed(seed)
            e = torch.rand(n, dtype=torch.float32)
            got = torch.tensor(list(_rs.torch_rand(seed, n, True)[0]), dtype=torch.float32)
            assert torch.equal(got, e)


#: Float arguments are float32-representable: BoTorch stores a Python-float
#: ``best_f`` / ``tau`` as a buffer of torch's default dtype (float32; see
#: docs/clamp-subgradients.md), botorch_rs keeps float64.
MC_CASES: list[tuple[str, dict[str, Any]]] = [
    ("qExpectedImprovement", {"best_f": 0.375}),
    ("qProbabilityOfImprovement", {"best_f": 0.375, "tau": 2.0**-10}),
    ("qUpperConfidenceBound", {"beta": 0.25}),
    ("qSimpleRegret", {}),
    ("qLogExpectedImprovement", {"best_f": 0.375}),
    ("qNoisyExpectedImprovement", {"prune_baseline": False, "cache_root": False}),
    ("qLogNoisyExpectedImprovement", {"prune_baseline": False, "cache_root": False}),
]


@pytest.mark.parametrize("name,kw", MC_CASES, ids=[n for n, _ in MC_CASES])
@pytest.mark.parametrize("m", [1, 2])
def test_iid_sampler_in_mc_acquisition(name: str, kw: dict[str, Any], m: int) -> None:
    import botorch.acquisition as bta
    from botorch.acquisition.objective import GenericMCObjective as BtObj
    from botorch.sampling import IIDNormalSampler as BtIID

    import botorch_rs.acquisition as rsa
    from botorch_rs.acquisition.objective import GenericMCObjective as RsObj
    from botorch_rs.sampling import IIDNormalSampler as RsIID

    bt, rs, X = model_pair(ModelSpec(n=8, d=2, m=m), 3)
    kw = dict(kw)
    if "prune_baseline" in kw:
        kw["X_baseline"] = X
    obj_b: dict[str, Any] = {}
    obj_r: dict[str, Any] = {}
    if m == 2:
        obj_b = {"objective": BtObj(lambda Y, X=None: Y[..., 0] - 0.5 * Y[..., 1])}
        obj_r = {"objective": RsObj(lambda Y, X=None: Y[..., 0] - 0.5 * Y[..., 1])}
    a = getattr(rsa, name)(rs, sampler=RsIID(torch.Size([64]), seed=11), **kw, **obj_r)
    e = getattr(bta, name)(bt, sampler=BtIID(torch.Size([64]), seed=11), **kw, **obj_b)
    Xc = torch.rand(4, 3, 2, generator=gen(31), dtype=DT)
    C.check(f"IIDNormalSampler {name} value", a(Xc), e(Xc), TOL)
    (ga,) = grad_of(a, Xc)
    (ge,) = grad_of(e, Xc)
    C.check(f"IIDNormalSampler {name} grad", ga, ge, GTOL)
    # BoTorch's own IIDNormalSampler passed to botorch_rs gives the same draws.
    a2 = getattr(rsa, name)(rs, sampler=BtIID(torch.Size([64]), seed=11), **kw, **obj_r)
    C.check(f"IIDNormalSampler {name} BoTorch sampler", a2(Xc), e(Xc), TOL)


def test_iid_posterior_samples_forked_and_get_sampler() -> None:
    from botorch.sampling import IIDNormalSampler as BtIID
    from botorch.sampling.get_sampler import get_sampler as bt_get
    from botorch.sampling.stochastic_samplers import ForkedRNGSampler as BtF

    from botorch_rs.sampling import ForkedRNGSampler as RsF
    from botorch_rs.sampling import IIDNormalSampler as RsIID
    from botorch_rs.sampling import get_sampler as rs_get

    for m in (1, 2):
        bt, rs, _ = model_pair(ModelSpec(n=8, d=2, m=m), 4)
        Xc = torch.rand(3, 2, 2, generator=gen(32), dtype=DT)
        pb, pr = bt.posterior(Xc), rs.posterior(Xc)
        C.check(
            "IIDNormalSampler posterior samples",
            RsIID(torch.Size([7]), seed=3)(pr),
            BtIID(torch.Size([7]), seed=3)(pb),
            TOL,
        )
        ra, rb = rs_get(pr, torch.Size([4]), seed=1), bt_get(pb, torch.Size([4]), seed=1)
        assert type(ra).__name__ == type(rb).__name__
        assert RsF(torch.Size([5]), seed=2)(pr).shape == BtF(torch.Size([5]), seed=2)(pb).shape


@pytest.mark.parametrize("inv_transform", [False, True])
@pytest.mark.parametrize("d", [1, 3, 4])
def test_qmc_engines(inv_transform: bool, d: int) -> None:
    from botorch.sampling.qmc import MultivariateNormalQMCEngine as BtM
    from botorch.sampling.qmc import NormalQMCEngine as BtN

    from botorch_rs.sampling.qmc import MultivariateNormalQMCEngine as RsM
    from botorch_rs.sampling.qmc import NormalQMCEngine as RsN

    a, e = RsN(d, seed=3, inv_transform=inv_transform), BtN(d, seed=3, inv_transform=inv_transform)
    C.check("NormalQMCEngine", a.draw(8, dtype=DT), e.draw(8, dtype=DT), TOL)
    C.check("NormalQMCEngine continued", a.draw(5, dtype=DT), e.draw(5, dtype=DT), TOL)
    C.check(
        "NormalQMCEngine float32",
        a.draw(4, dtype=torch.float32).double(),
        e.draw(4, dtype=torch.float32).double(),
        F32TOL,
    )
    mean = torch.linspace(-1, 1, d, dtype=DT)
    L = torch.randn(d, d, generator=gen(33), dtype=DT)
    cov = L @ L.T + torch.eye(d, dtype=DT)
    ma = RsM(mean, cov, seed=1, inv_transform=inv_transform)
    me = BtM(mean, cov, seed=1, inv_transform=inv_transform)
    C.check("MultivariateNormalQMCEngine", ma.draw(8), me.draw(8), TOL)


@pytest.mark.parametrize("inv_transform", [False, True])
@pytest.mark.parametrize("out_dtype", [torch.float32, torch.float64])
def test_multivariate_qmc_output_dtype(inv_transform: bool, out_dtype: Any) -> None:
    from botorch.sampling.qmc import MultivariateNormalQMCEngine as BtM

    from botorch_rs.sampling.qmc import MultivariateNormalQMCEngine as RsM

    mean = torch.linspace(-1, 1, 5, dtype=DT)
    L = torch.randn(5, 5, generator=gen(95), dtype=DT)
    cov = L @ L.T + torch.eye(5, dtype=DT)
    actual = RsM(mean, cov, seed=3, inv_transform=inv_transform)
    expected = BtM(mean, cov, seed=3, inv_transform=inv_transform)
    out_a = torch.empty(32, 5, dtype=out_dtype)
    out_e = torch.empty_like(out_a)
    assert actual.draw(32, out=out_a) is None
    assert expected.draw(32, out=out_e) is None
    C.check(
        f"MultivariateNormalQMCEngine output {out_dtype}",
        out_a.double(),
        out_e.double(),
        F32TOL if out_dtype == torch.float32 else TOL,
    )


# ---------------------------------------------------------------------------
# Test functions
# ---------------------------------------------------------------------------

PROBLEMS: list[tuple[str, str, dict[str, Any]]] = [
    ("multi_fidelity", "AugmentedBranin", {}),
    ("multi_fidelity", "AugmentedHartmann", {}),
    ("multi_fidelity", "AugmentedRosenbrock", {"dim": 3}),
    ("multi_fidelity", "AugmentedRosenbrock", {"dim": 5, "negate": True}),
    ("multi_fidelity", "WingWeightMultiFidelity", {}),
    ("multi_fidelity", "BoreholeMultiFidelity", {}),
    ("sensitivity_analysis", "Ishigami", {}),
    ("sensitivity_analysis", "Ishigami", {"b": 0.05}),
    ("sensitivity_analysis", "Gsobol", {"dim": 6}),
    ("sensitivity_analysis", "Gsobol", {"dim": 15}),
    ("sensitivity_analysis", "Gsobol", {"dim": 4, "a": [0.0, 1.0, 4.5, 9.0]}),
    ("sensitivity_analysis", "Morris", {}),
    ("synthetic", "AckleyMixed", {"dim": 7}),
    ("synthetic", "Labs", {"dim": 12}),
    ("synthetic", "TrajectoryPlanning", {"dim": 8}),
    ("synthetic", "TrajectoryPlanning", {"dim": 10, "use_smooth_interp": False}),
    ("synthetic", "TrajectoryPlanning", {"dim": 6, "max_step_size": 0.2}),
    ("synthetic", "ConstrainedGramacy", {}),
    ("synthetic", "ConstrainedHartmann", {"dim": 6}),
    ("synthetic", "ConstrainedHartmann", {"dim": 3, "negate": True}),
    ("synthetic", "ConstrainedHartmannSmooth", {"dim": 6}),
    ("synthetic", "PressureVessel", {}),
    ("synthetic", "WeldedBeamSO", {}),
    ("synthetic", "TensionCompressionString", {}),
    ("synthetic", "SpeedReducer", {}),
    ("synthetic", "KeaneBumpFunction", {"dim": 5}),
    ("synthetic", "KeaneBumpFunction", {"dim": 2, "bounds": [(0.5, 3.0), (1.0, 8.0)]}),
]

PROBLEM_ATTRS = (
    "dim",
    "num_objectives",
    "num_constraints",
    "continuous_inds",
    "discrete_inds",
    "categorical_inds",
    "_optimal_value",
    "_optimizers",
    "_check_grad_at_opt",
    "fidelities",
    "a",
    "b",
    "num_waypoints",
    "max_step_size",
    "use_smooth_interp",
)


def _problem_pair(mod: str, name: str, kw: dict[str, Any]) -> tuple[Any, Any]:
    bt_mod = __import__(f"botorch.test_functions.{mod}", fromlist=["x"])
    rs_mod = __import__(f"botorch_rs.test_functions.{mod}", fromlist=["x"])
    return getattr(rs_mod, name)(**kw), getattr(bt_mod, name)(**kw)


def _problem_points(p: Any, n: int, seed: int) -> Any:
    lo, hi = p.bounds
    X = lo + (hi - lo) * torch.rand(n, p.dim, generator=gen(seed), dtype=DT)
    inds = list(getattr(p, "discrete_inds", [])) + list(getattr(p, "categorical_inds", []))
    if inds:
        X[:, inds] = X[:, inds].round()
    return X


def _same_attr(a: Any, e: Any, label: str) -> None:
    if torch.is_tensor(e):
        C.check(label, a, e, TOL)
    elif isinstance(e, (list, tuple)) and e and isinstance(e[0], (list, tuple)):
        C.check(label, torch.tensor(a, dtype=DT), torch.tensor(e, dtype=DT), TOL)
    elif isinstance(e, float):
        C.check(label, torch.tensor(a, dtype=DT), torch.tensor(e, dtype=DT), TOL)
    else:
        assert list(a) == list(e) if isinstance(e, (list, tuple)) else a == e, (label, a, e)


@pytest.mark.parametrize(
    "mod,name,kw", PROBLEMS, ids=[f"{n}-{i}" for i, (_, n, _k) in enumerate(PROBLEMS)]
)
def test_test_functions(mod: str, name: str, kw: dict[str, Any]) -> None:
    a, e = _problem_pair(mod, name, kw)
    C.check(f"{name} bounds", a.bounds, e.bounds, 0.0)
    for attr in PROBLEM_ATTRS:
        try:
            ve = getattr(e, attr)
        except (AttributeError, NotImplementedError) as err:
            # Gsobol, Ishigami and Morris raise NotImplementedError.
            with pytest.raises(type(err)):
                getattr(a, attr)
            continue
        if not callable(ve):
            _same_attr(getattr(a, attr), ve, f"{name}.{attr}")
    X = _problem_points(e, 9, 40)
    X3 = X.reshape(3, 3, -1)
    # CPU compilers may fuse the reference's default float32 linspace.
    # Compare both spline implementations on the same explicit sampling grid.
    trajectory = name == "TrajectoryPlanning" and kw.get("use_smooth_interp", True)
    with portable_trajectory_grid(trajectory):
        ve = e.evaluate_true(X)
    C.check(f"{name} evaluate_true", a.evaluate_true(X), ve, TOL)
    # BoTorch's TrajectoryPlanning only accepts ``n x d`` inputs (it iterates
    # over the first dimension); the batched values are compared with the
    # values of the flattened batch.
    C.check(f"{name} evaluate_true batched", a.evaluate_true(X3), ve.reshape(3, 3), TOL)
    with portable_trajectory_grid(trajectory):
        forward_reference = e(X)
    C.check(f"{name} forward", a(X), forward_reference, TOL)
    if getattr(e, "_optimizers", None):
        opt = torch.tensor(e._optimizers, dtype=DT)
        C.check(
            f"{name} at optimizers", a.evaluate_true(opt), e.evaluate_true(opt), TOL, atol=1e-14
        )
    if hasattr(e, "evaluate_slack_true"):
        C.check(
            f"{name} evaluate_slack_true", a.evaluate_slack_true(X), e.evaluate_slack_true(X), TOL
        )
        assert torch.equal(a.is_feasible(X), e.is_feasible(X))
        C.check(f"{name} evaluate_slack", a.evaluate_slack(X), e.evaluate_slack(X), TOL)
    if hasattr(e, "cost"):
        C.check(f"{name} cost", a.cost(X), e.cost(X), TOL)
    if hasattr(e, "compute_dgsm"):
        for x, y in zip(a.compute_dgsm(X), e.compute_dgsm(X), strict=True):
            C.check(f"{name} compute_dgsm", torch.tensor(x), torch.tensor(y), TOL)
    if hasattr(e, "optimal_sobol_indicies"):
        assert a.optimal_sobol_indicies() is e.optimal_sobol_indicies() is None
    for attr in (
        "dgsm_gradient",
        "dgsm_gradient_abs",
        "dgsm_gradient_square",
        "vi",
        "V",
        "si",
        "si_t",
    ):
        if hasattr(e, attr):
            va, ve = getattr(a, attr), getattr(e, attr)
            if torch.is_tensor(ve):
                assert va.dtype == ve.dtype, (attr, va.dtype, ve.dtype)
            C.check(
                f"{name}.{attr}", torch.as_tensor(va, dtype=DT), torch.as_tensor(ve, dtype=DT), TOL
            )


def test_test_function_noise_and_dtypes() -> None:
    import botorch.test_functions.synthetic as bsy

    import botorch_rs.test_functions.synthetic as rsy

    for name, kw in (
        ("ConstrainedHartmann", {"dim": 6, "noise_std": 0.1, "constraint_noise_std": 0.2}),
        ("PressureVessel", {"noise_std": 0.5, "constraint_noise_std": [0.1, 0.2, 0.3, 0.4]}),
        ("AckleyMixed", {"dim": 5, "noise_std": 0.3}),
    ):
        a, e = getattr(rsy, name)(**kw), getattr(bsy, name)(**kw)
        X = _problem_points(e, 6, 41)
        torch.manual_seed(0)
        va = a(X)
        torch.manual_seed(0)
        ve = e(X)
        C.check(f"{name} noisy forward", va, ve, TOL)
        if hasattr(e, "evaluate_slack"):
            torch.manual_seed(1)
            va = a.evaluate_slack(X)
            torch.manual_seed(1)
            ve = e.evaluate_slack(X)
            C.check(f"{name} noisy slack", va, ve, TOL)
    for name, kw in (("Labs", {"dim": 8}), ("ConstrainedGramacy", {})):
        a = getattr(rsy, name)(dtype=torch.float32, **kw)
        e = getattr(bsy, name)(dtype=torch.float32, **kw)
        assert a.bounds.dtype == e.bounds.dtype == torch.float32
        X = _problem_points(e, 5, 42).float()
        C.check(f"{name} float32", a(X).double(), e(X).double(), F32TOL)
    # AckleyMixed with a randomized optimum draws it with torch's generator.
    torch.manual_seed(3)
    a = rsy.AckleyMixed(dim=6, randomize_optimum=True)
    torch.manual_seed(3)
    e = bsy.AckleyMixed(dim=6, randomize_optimum=True)
    C.check(
        "AckleyMixed random optimum", torch.tensor(a._optimizers), torch.tensor(e._optimizers), 0.0
    )
    X = _problem_points(e, 5, 43)
    C.check("AckleyMixed random optimum values", a(X), e(X), TOL)


def test_test_function_base_and_round_nearest() -> None:
    import botorch.test_functions.base as bb
    import botorch.test_functions.synthetic as bsy
    from botorch.test_functions.utils import round_nearest as bt_round

    import botorch_rs.test_functions.base as rb
    import botorch_rs.test_functions.synthetic as rsy
    from botorch_rs.test_functions.utils import round_nearest as rs_round

    X = torch.randn(5, 4, generator=gen(44), dtype=DT) * 3
    for inc, bounds in ((1.0, None), (0.25, None), (0.3, (-2.0, 2.5)), (2.0, (-3.0, 4.0))):
        C.check("round_nearest", rs_round(X, inc, bounds), bt_round(X, inc, bounds), 0.0)
    from functools import partial

    gens = (
        partial(rb.constant_outlier_generator, constant=7.0),
        partial(bb.constant_outlier_generator, constant=7.0),
    )
    a = rb.CorruptedTestProblem(rsy.Hartmann(dim=3), gens[0], outlier_fraction=0.4, seeds=[1, 2, 3])
    e = bb.CorruptedTestProblem(bsy.Hartmann(dim=3), gens[1], outlier_fraction=0.4, seeds=[1, 2, 3])
    X = torch.rand(8, 3, generator=gen(45), dtype=DT)
    for _ in range(3):
        C.check("CorruptedTestProblem", a(X), e(X), TOL)
        assert a.seed == e.seed
    C.check("CorruptedTestProblem noiseless", a(X, noise=False), e(X, noise=False), TOL)
    bounds = torch.tensor([[0.0, 0.0, 0.0], [1.0, 3.0, 2.0]], dtype=DT)
    cases = [
        (3, bounds, [0], [1], [2]),
        (3, bounds, [0, 1], [2], []),
        (3, bounds, [0], [1], []),
        (3, bounds, [0, 1], [1], [2]),
        (3, bounds, [0], [1, 3], [2]),
        (3, torch.tensor([[0.0, 0.5, 0.0], [1.0, 3.0, 2.0]], dtype=DT), [0], [1], [2]),
    ]
    for args in cases:
        errs = []
        for f in (rb.validate_parameter_indices, bb.validate_parameter_indices):
            try:
                f(*args)
                errs.append(None)
            except Exception as err:  # noqa: BLE001
                errs.append((type(err), str(err)))
        assert errs[0] == errs[1], (args, errs)
    Xv = [
        torch.tensor([[0.5, 1.0, 2.0]], dtype=DT),
        torch.tensor([[0.5, 1.5, 2.0]], dtype=DT),
        torch.tensor([[0.5, 1.0, 2.5]], dtype=DT),
        torch.tensor([[1.5, 1.0, 2.0]], dtype=DT),
        torch.tensor([[0.5, 1.0]], dtype=DT),
    ]
    for x in Xv:
        errs = []
        for f in (rb.validate_inputs, bb.validate_inputs):
            try:
                f(x, 3, bounds, [1], [2])
                errs.append(None)
            except Exception as err:  # noqa: BLE001
                errs.append((type(err), str(err)))
        assert errs[0] == errs[1], (x, errs)


# ---------------------------------------------------------------------------
# botorch.utils
# ---------------------------------------------------------------------------


def test_transforms() -> None:
    import botorch.utils.transforms as bt

    import botorch_rs.utils.transforms as rs

    Y = torch.randn(3, 5, 2, generator=gen(50), dtype=DT)
    Y[1, :, 1] = 2.5  # a constant column
    C.check("standardize", rs.standardize(Y), bt.standardize(Y), TOL)
    b = torch.tensor([[0.0, 1.0, 2.0], [1.0, 1.0, 5.0]], dtype=DT)
    X = torch.rand(2, 4, 3, generator=gen(51), dtype=DT) * 4
    for upd in (True, False):
        C.check("normalize", rs.normalize(X, b, upd), bt.normalize(X, b, upd), TOL)
        C.check("unnormalize", rs.unnormalize(X, b, upd), bt.unnormalize(X, b, upd), TOL)
    C.check(
        "_update_constant_bounds", rs._update_constant_bounds(b), bt._update_constant_bounds(b), 0.0
    )
    (ga,) = grad_of(lambda t: rs.normalize(t, b), X)
    (ge,) = grad_of(lambda t: bt.normalize(t, b), X)
    C.check("normalize grad", ga, ge, GTOL)
    for idx, d in (([0, 2], 3), ([-1], 3), (None, 2), ([1, -3], 4)):
        assert rs.normalize_indices(idx, d) == bt.normalize_indices(idx, d)
    for idx, d in (([3], 3), ([-4], 3)):
        with pytest.raises(ValueError):
            bt.normalize_indices(idx, d)
        with pytest.raises(ValueError):
            rs.normalize_indices(idx, d)
    for xs, ys in (((2, 1, 5, 3), (2, 6, 4, 3)), ((5, 3), (7, 4, 3)), ((1, 5, 3), (2, 4, 4, 3))):
        Xm, Ym = torch.rand(*xs, dtype=DT), torch.rand(*ys, dtype=DT)
        assert torch.equal(rs.match_batch_shape(Xm, Ym), bt.match_batch_shape(Xm, Ym))


def test_objective_and_rounding() -> None:
    import botorch.utils.objective as bo
    import botorch.utils.rounding as br

    import botorch_rs.utils.objective as ro
    import botorch_rs.utils.rounding as rr

    samples = torch.randn(8, 3, 2, 2, generator=gen(52), dtype=DT)
    cons = [lambda Y: Y[..., 0] - 0.2, lambda Y: -Y[..., 1]]
    obj = samples[..., 0]
    # eta values representable in float32: BoTorch builds the eta tensor with
    # torch.full (torch's default dtype), botorch_rs keeps float64.
    for eta, log, fat in (
        (0.125, True, [True, False]),
        (0.25, False, False),
        (2.0**-10, True, True),
    ):
        for eta_arg in (eta, torch.tensor([eta, 2 * eta], dtype=DT)):
            C.check(
                "compute_smoothed_feasibility_indicator",
                ro.compute_smoothed_feasibility_indicator(cons, samples, eta_arg, log=log, fat=fat),
                bo.compute_smoothed_feasibility_indicator(cons, samples, eta_arg, log=log, fat=fat),
                TOL,
            )
    for infeasible_cost, eta in ((2.0, 0.25), (0.0, torch.tensor([0.125, 0.5], dtype=DT))):
        C.check(
            "apply_constraints",
            ro.apply_constraints(obj, cons, samples, infeasible_cost, eta),
            bo.apply_constraints(obj, cons, samples, infeasible_cost, eta),
            TOL,
        )
    C.check(
        "apply_constraints_nonnegative_soft",
        ro.apply_constraints_nonnegative_soft(obj, cons, samples, 0.25),
        bo.apply_constraints_nonnegative_soft(obj, cons, samples, 0.25),
        TOL,
    )
    (ga,) = grad_of(lambda s: ro.apply_constraints(s[..., 0], cons, s, 2.0, 0.25), samples)
    (ge,) = grad_of(lambda s: bo.apply_constraints(s[..., 0], cons, s, 2.0, 0.25), samples)
    C.check("apply_constraints grad", ga, ge, GTOL)
    for md in (None, -2):
        assert torch.equal(
            ro.compute_feasibility_indicator(cons, samples, marginalize_dim=md),
            bo.compute_feasibility_indicator(cons, samples, marginalize_dim=md),
        )
    w = torch.tensor([0.75, -0.25], dtype=DT)
    C.check(
        "get_objective_weights_transform",
        ro.get_objective_weights_transform(w)(samples),
        bo.get_objective_weights_transform(w)(samples),
        TOL,
    )
    x = torch.randn(10, generator=gen(53), dtype=DT) * 3
    C.check("approximate_round", rr.approximate_round(x, 0.3), br.approximate_round(x, 0.3), TOL)
    (ga,) = grad_of(lambda t: rr.approximate_round(t, 0.3), x)
    (ge,) = grad_of(lambda t: br.approximate_round(t, 0.3), x)
    C.check("approximate_round grad", ga, ge, GTOL)
    (ga,) = grad_of(rr.RoundSTE.apply, x)
    (ge,) = grad_of(br.RoundSTE.apply, x)
    assert torch.equal(ga, ge) and torch.equal(rr.RoundSTE.apply(x), br.RoundSTE.apply(x))
    x2 = x.reshape(2, 5)
    assert torch.equal(rr.OneHotArgmaxSTE.apply(x2), br.OneHotArgmaxSTE.apply(x2))
    (ga,) = grad_of(rr.OneHotArgmaxSTE.apply, x2)
    (ge,) = grad_of(br.OneHotArgmaxSTE.apply, x2)
    assert torch.equal(ga, ge)


def test_constraints_containers_and_constants() -> None:
    import botorch.utils.constants as bk
    import botorch.utils.constraints as bc
    from botorch.utils.containers import DenseContainer as BDC
    from botorch.utils.containers import SliceContainer as BSC

    import botorch_rs.utils.constants as rk
    import botorch_rs.utils.constraints as rc
    from botorch_rs.utils.containers import DenseContainer as RDC
    from botorch_rs.utils.containers import SliceContainer as RSC

    samples = torch.randn(8, 3, 2, 2, generator=gen(54), dtype=DT)
    A = torch.tensor([[1.0, -1.0], [0.5, 2.0]], dtype=DT)
    b = torch.tensor([[0.3], [-0.1]], dtype=DT)
    for fa, fe in zip(
        rc.get_outcome_constraint_transforms((A, b)),
        bc.get_outcome_constraint_transforms((A, b)),
        strict=True,
    ):
        C.check("get_outcome_constraint_transforms", fa(samples), fe(samples), TOL)
    assert (
        rc.get_outcome_constraint_transforms(None)
        is bc.get_outcome_constraint_transforms(None)
        is None
    )
    for d, desc in ((4, True), (3, False), (1, False)):
        for p, q in zip(
            rc.get_monotonicity_constraints(d, descending=desc),
            bc.get_monotonicity_constraints(d, descending=desc),
            strict=True,
        ):
            assert torch.equal(p, q)
    v = torch.rand(16, 2, generator=gen(55), dtype=DT)
    idx = torch.stack([torch.randperm(16, generator=gen(56 + i))[:3] for i in range(8)])
    sa, se = (
        RSC(values=v, indices=idx, event_shape=torch.Size([6])),
        BSC(values=v, indices=idx, event_shape=torch.Size([6])),
    )
    assert torch.equal(sa(), se()) and sa.shape == se.shape
    da, de = RDC(values=v, event_shape=torch.Size([2])), BDC(values=v, event_shape=torch.Size([2]))
    assert torch.equal(da(), de()) and da.shape == de.shape and da.event_shape == de.event_shape
    assert (da == RDC(values=v.clone(), event_shape=torch.Size([2]))) and not (
        da == RDC(values=v + 1, event_shape=torch.Size([2]))
    )
    # BoTorch caches the constants (lru_cache): hashable values only.
    for vals in (0.5, (1.0, 2.0), (3.0,)):
        ref = torch.zeros(2, dtype=torch.float32)
        for a, e in (
            (rk.get_constants(vals, dtype=DT), bk.get_constants(vals, dtype=DT)),
            (rk.get_constants_like(vals, ref), bk.get_constants_like(vals, ref)),
        ):
            assert type(a) is type(e)
            for x, y in zip(a, e, strict=True) if isinstance(e, tuple) else ((a, e),):
                assert torch.equal(x, y) and x.dtype == y.dtype


def test_datasets() -> None:
    from botorch.utils.containers import DenseContainer as BDC
    from botorch.utils.containers import SliceContainer as BSC
    from botorch.utils.datasets import ContextualDataset as BCD
    from botorch.utils.datasets import MultiTaskDataset as BMT
    from botorch.utils.datasets import RankingDataset as BRD
    from botorch.utils.datasets import SupervisedDataset as BSD

    from botorch_rs.utils.containers import DenseContainer as RDC
    from botorch_rs.utils.containers import SliceContainer as RSC
    from botorch_rs.utils.datasets import ContextualDataset as RCD
    from botorch_rs.utils.datasets import MultiTaskDataset as RMT
    from botorch_rs.utils.datasets import RankingDataset as RRD
    from botorch_rs.utils.datasets import SupervisedDataset as RSD

    def pair(
        n: int, seed: int, outcome: str, task: float | None = None, yvar: bool = False
    ) -> tuple[Any, Any]:
        X = torch.rand(n, 3, generator=gen(seed), dtype=DT)
        if task is not None:
            X[:, 1] = task
        Y = torch.randn(n, 1, generator=gen(seed + 1), dtype=DT)
        V = torch.rand(n, 1, generator=gen(seed + 2), dtype=DT) if yvar else None
        kw = {"feature_names": ["a", "task", "c"], "outcome_names": [outcome]}
        return RSD(X, Y, Yvar=V, **kw), BSD(X, Y, Yvar=V, **kw)

    def same(a: Any, e: Any, label: str) -> None:
        assert a.feature_names == e.feature_names and a.outcome_names == e.outcome_names, label
        C.check(f"{label} X", a.X, e.X, 0.0)
        C.check(f"{label} Y", a.Y, e.Y, 0.0)
        assert (a.Yvar is None) == (e.Yvar is None), label
        if e.Yvar is not None:
            C.check(f"{label} Yvar", a.Yvar, e.Yvar, 0.0)

    for yvar in (False, True):
        ds = [
            pair(5, 60, "y0", 0.0, yvar),
            pair(4, 63, "y1", 1.0, yvar),
            pair(3, 66, "y2", 2.0, yvar),
        ]
        ma = RMT([d[0] for d in ds], target_outcome_name="y1", task_feature_index=1)
        me = BMT([d[1] for d in ds], target_outcome_name="y1", task_feature_index=1)
        same(ma, me, "MultiTaskDataset")
        assert ma.task_feature_index == me.task_feature_index
        same(
            ma.get_dataset_without_task_feature("y2"),
            me.get_dataset_without_task_feature("y2"),
            "MultiTaskDataset subset",
        )
        joint_a = RSD(me.X, me.Y, Yvar=me.Yvar, feature_names=me.feature_names, outcome_names=["y"])
        joint_e = BSD(me.X, me.Y, Yvar=me.Yvar, feature_names=me.feature_names, outcome_names=["y"])
        ja = RMT.from_joint_dataset(joint_a, task_feature_index=1, target_task_value=2)
        je = BMT.from_joint_dataset(joint_e, task_feature_index=1, target_task_value=2)
        same(ja, je, "MultiTaskDataset.from_joint_dataset")
        assert sorted(ja.datasets) == sorted(je.datasets)
    # Without a task feature: tasks are appended as a last column.
    ds = [pair(5, 70, "y0"), pair(4, 73, "y1")]
    ma = RMT([d[0] for d in ds], target_outcome_name="y0")
    me = BMT([d[1] for d in ds], target_outcome_name="y0")
    same(ma, me, "MultiTaskDataset appended task")
    # Contextual datasets.
    decomp = {"c1": ["a"], "c2": ["c"]}
    ds = [pair(4, 80, "y")]
    ca = RCD([d[0] for d in ds], parameter_decomposition=decomp)
    ce = BCD([d[1] for d in ds], parameter_decomposition=decomp)
    same(ca, ce, "ContextualDataset")
    ds = [pair(4, 83, "y_c1"), pair(4, 83, "y_c2")]
    ds[1][0].X.copy_(ds[0][0].X)
    ds[1][1].X.copy_(ds[0][1].X)
    mdec = {"c1": ["y_c1"], "c2": ["y_c2"]}
    ca = RCD([d[0] for d in ds], parameter_decomposition=decomp, metric_decomposition=mdec)
    ce = BCD([d[1] for d in ds], parameter_decomposition=decomp, metric_decomposition=mdec)
    same(ca, ce, "ContextualDataset LCE-M")
    # Ranking datasets.
    v = torch.rand(10, 2, generator=gen(86), dtype=DT)
    idx = torch.tensor([[0, 3], [2, 5], [7, 1], [4, 9]])
    Yr = torch.tensor([[0, 1], [1, 0], [0, 1], [1, 0]])
    ra = RRD(
        RSC(values=v, indices=idx, event_shape=torch.Size([4])),
        RDC(values=Yr, event_shape=torch.Size([2])),
        ["a", "b"],
        ["r"],
    )
    re = BRD(
        BSC(values=v, indices=idx, event_shape=torch.Size([4])),
        BDC(values=Yr, event_shape=torch.Size([2])),
        ["a", "b"],
        ["r"],
    )
    assert torch.equal(ra.X, re.X) and torch.equal(ra.Y, re.Y)
    for bad in (
        [[0, 0], [1, 0], [0, 1], [1, 0]],
        [[0, 2], [1, 0], [0, 1], [1, 0]],
        [[-1, 0], [1, 0], [0, 1], [1, 0]],
        [[1, 1], [1, 0], [0, 1], [1, 0]],
    ):
        outcomes = []
        for cls, sc, dc in ((RRD, RSC, RDC), (BRD, BSC, BDC)):
            try:
                cls(
                    sc(values=v, indices=idx, event_shape=torch.Size([4])),
                    dc(values=torch.tensor(bad), event_shape=torch.Size([2])),
                    ["a", "b"],
                    ["r"],
                )
                outcomes.append(None)
            except Exception as err:  # noqa: BLE001
                outcomes.append((type(err), str(err)))
        assert outcomes[0] == outcomes[1], (bad, outcomes)


def test_low_rank_and_multitask() -> None:
    from botorch.utils.low_rank import extract_batch_covar as bt_ebc
    from botorch.utils.low_rank import sample_cached_cholesky as bt_scc
    from linear_operator.utils.cholesky import psd_safe_cholesky

    from botorch_rs.utils.low_rank import extract_batch_covar as rs_ebc
    from botorch_rs.utils.low_rank import sample_cached_cholesky as rs_scc
    from botorch_rs.utils.multitask import separate_mtmvn

    bt, rs, _ = model_pair(ModelSpec(n=6, d=2, m=2), 1)
    Xq = torch.rand(3, 4, 2, generator=gen(90), dtype=DT)
    pb, pr = bt.posterior(Xq), rs.posterior(Xq)
    cov_e = bt_ebc(pb.distribution).to_dense()
    C.check("extract_batch_covar", rs_ebc(pr.distribution), cov_e, TOL)
    q = 2
    L = psd_safe_cholesky(cov_e[..., :-q, :-q])
    Z = torch.randn(5, 1, 4, 2, generator=gen(91), dtype=DT)
    C.check(
        "sample_cached_cholesky",
        rs_scc(pr, L, q, Z, torch.Size([5])),
        bt_scc(pb, L, q, Z, torch.Size([5])),
        TOL,
    )
    # separate_mtmvn: BoTorch's values are wrong for batched interleaved
    # multitask normals (linear_operator's BlockDiag slicing); the reference
    # is the dense covariance's diagonal blocks.
    dense = pb.distribution.covariance_matrix
    mean = pb.distribution.mean
    n = mean.shape[-2]
    for t, mvn in enumerate(separate_mtmvn(pr.distribution)):
        sl = slice(t, None, 2) if pb.distribution._interleaved else slice(t * n, (t + 1) * n)
        C.check("separate_mtmvn mean", mvn.mean, mean[..., t], TOL)
        C.check("separate_mtmvn covariance", mvn.covariance_matrix, dense[..., sl, sl], TOL)


def test_context_managers_dispatcher_settings_logging() -> None:
    import botorch.logging as blog
    from botorch.utils.dispatcher import Dispatcher as BD

    import botorch_rs.settings as rset
    from botorch_rs.logging import shape_to_str
    from botorch_rs.utils.context_managers import (
        delattr_ctx,
        module_rollback_ctx,
        parameter_rollback_ctx,
        zero_grad_ctx,
    )
    from botorch_rs.utils.dispatcher import Dispatcher as RD

    _, rs, _ = model_pair(ModelSpec(n=8, d=2), 2)
    before = {k: v.detach().clone() for k, v in rs.named_parameters()}
    with module_rollback_ctx(rs):
        rs.initialize(**{"likelihood.noise": 0.5})
    assert all(torch.equal(before[k], v) for k, v in rs.named_parameters())
    lin = torch.nn.Linear(2, 1).double()
    w = lin.weight.detach().clone()
    with parameter_rollback_ctx(dict(lin.named_parameters())):
        with torch.no_grad():
            lin.weight.add_(1.0)
    assert torch.equal(lin.weight, w)
    lin(torch.ones(1, 2, dtype=DT)).sum().backward()
    with zero_grad_ctx(dict(lin.named_parameters())):
        assert lin.weight.grad is None or bool((lin.weight.grad == 0).all())

    class Obj:
        a = 1

    o = Obj()
    o.b = 2  # type: ignore[attr-defined]
    with delattr_ctx(o, "b"):
        assert not hasattr(o, "b")
    assert o.b == 2  # type: ignore[attr-defined]

    class A:
        pass

    class B(A):
        pass

    class Cc(B):
        pass

    results = []
    for D in (RD, BD):
        d = D("d")
        d.register(A, object)(lambda x, y: "A,o")
        d.register(B, int)(lambda x, y: "B,int")
        d.register(object, int)(lambda x, y: "o,int")
        d.register(Cc, (int, float))(lambda x, y: "C,num")
        calls = [(A(), 1), (B(), 1), (Cc(), 1.0), (Cc(), 2), (B(), "s"), (1, 1)]
        out = []
        for x, y in calls:
            try:
                out.append(d(x, y))
            except NotImplementedError:
                out.append("NI")
        results.append((out, [tuple(t.__name__ for t in sig) for sig in d.ordering]))
    assert results[0] == results[1]
    assert shape_to_str(torch.Size([2, 3])) == blog.shape_to_str(torch.Size([2, 3]))
    assert shape_to_str(torch.Size([])) == blog.shape_to_str(torch.Size([]))
    with rset.validate_input_scaling(False):
        assert not rset.validate_input_scaling.on()
    assert rset.validate_input_scaling.on()


def test_evaluation_and_feasible_volume() -> None:
    from botorch.utils.evaluation import compute_in_sample_model_fit_metric as bfit
    from botorch.utils.feasible_volume import get_feasible_samples as bgf

    from botorch_rs.utils.evaluation import compute_in_sample_model_fit_metric as rfit
    from botorch_rs.utils.feasible_volume import get_feasible_samples as rgf

    for spec in (
        ModelSpec(n=8, d=2),
        ModelSpec(n=8, d=2, m=2),
        ModelSpec(n=8, d=3, normalize=True),
    ):
        bt, rs, _ = model_pair(spec, 2)
        for crit in ("MLL", "AIC", "BIC"):
            C.check(
                f"compute_in_sample_model_fit_metric {crit}",
                torch.tensor(rfit(rs, crit)),
                torch.tensor(bfit(bt, crit)),
                TOL,
            )
    S = torch.rand(50, 3, generator=gen(92), dtype=DT)
    cons = [
        (torch.tensor([0, 2]), torch.tensor([1.0, 1.0], dtype=DT), 0.8),
        (torch.tensor([1]), torch.tensor([-1.0], dtype=DT), -0.3),
    ]
    fa, pa = rgf(S, cons)
    fe, pe = bgf(S, cons)
    assert torch.equal(fa, fe) and pa == pe
    fa, pa = rgf(S, None)
    fe, pe = bgf(S, None)
    assert torch.equal(fa, fe) and pa == pe


def test_polytope_samplers_and_normalization() -> None:
    import botorch.utils.sampling as bs

    import botorch_rs.utils.sampling as rsamp

    bounds = torch.tensor([[0.0, 0.0, 0.0], [1.0, 2.0, 1.0]], dtype=DT)
    A = torch.tensor([[1.0, 1.0, 0.0]], dtype=DT)
    b = torch.tensor([[1.5]], dtype=DT)
    Ceq = torch.tensor([[0.0, 1.0, 1.0]], dtype=DT)
    deq = torch.tensor([[1.0]], dtype=DT)
    sparse = [(torch.tensor([0, 2]), torch.tensor([1.0, -2.0], dtype=DT), 0.25)]
    for f in ("normalize_sparse_linear_constraints",):
        for x, y in zip(
            getattr(rsamp, f)(bounds, sparse), getattr(bs, f)(bounds, sparse), strict=True
        ):
            assert torch.equal(x[0], y[0])
            C.check(f, x[1], y[1], TOL)
            C.check(f, torch.tensor(x[2]), torch.tensor(y[2]), TOL)
    C.check(
        "normalize_dense_linear_constraints",
        torch.cat(rsamp.normalize_dense_linear_constraints(bounds, (A, b)), -1),
        torch.cat(bs.normalize_dense_linear_constraints(bounds, (A, b)), -1),
        TOL,
    )
    for kw in (
        {"inequality_constraints": (A, b)},
        {"inequality_constraints": (A, b), "equality_constraints": (Ceq, deq)},
    ):
        # botorch_rs's hit-and-run chains use its own generator (compared
        # statistically in test_parity_constraints.py); here the normalized
        # system and the interior point.
        sa = rsamp.HitAndRunPolytopeSampler(bounds=bounds, n_burnin=10, n_thinning=3, seed=4, **kw)
        se = bs.HitAndRunPolytopeSampler(bounds=bounds, n_burnin=10, n_thinning=3, seed=4, **kw)
        pa, pe = sa._polytope(), se
        for attr in ("A", "b", "nullC", "new_A"):
            if getattr(pe, attr, None) is not None:
                C.check(
                    f"HitAndRunPolytopeSampler {attr}", getattr(pa, attr), getattr(pe, attr), TOL
                )
        xa = sa.draw(6)
        assert xa.shape == se.draw(6).shape
        assert all(
            bool(sa.feasible(((x - bounds[0]) / (bounds[1] - bounds[0])).unsqueeze(-1))) for x in xa
        )
        # The interior points maximize the smallest slack (a linear program
        # whose maximizers need not be unique): compare the slack.
        slack = []
        for p, x in ((pa, sa.find_interior_point()), (pe, se.find_interior_point())):
            assert bool(p.feasible(x))
            slack.append((p.b - p.A @ x).min())
        C.check("PolytopeSampler find_interior_point slack", slack[0], slack[1], TOL)
        da = rsamp.DelaunayPolytopeSampler(bounds=bounds, **kw)
        de = bs.DelaunayPolytopeSampler(bounds=bounds, **kw)
        C.check("DelaunayPolytopeSampler", da.draw(7, seed=3), de.draw(7, seed=3), TOL)
    with rsamp.manual_seed(5):
        a = torch.rand(3, dtype=DT)
    with bs.manual_seed(5):
        e = torch.rand(3, dtype=DT)
    assert torch.equal(a, e)


def test_optimize_posterior_samples() -> None:
    from botorch.models.deterministic import GenericDeterministicModel
    from botorch.utils.sampling import optimize_posterior_samples as bt_ops

    from botorch_rs.utils.sampling import optimize_posterior_samples as rs_ops

    centers = torch.tensor([[0.3, 0.7], [0.6, 0.2]], dtype=DT)

    def f(X: Any) -> Any:
        # Two sample paths (``2 x n x 1`` for ``n x d`` or ``2 x n x d``),
        # smooth with unique maxima 1 at the centers.
        d = (X - centers.unsqueeze(-2)).pow(2).sum(-1)
        return (-3.0 * d).exp().unsqueeze(-1)

    paths = GenericDeterministicModel(f, num_outputs=1)
    bounds = torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=DT)
    torch.manual_seed(0)
    xa, fa = rs_ops(paths, bounds, raw_samples=64, num_restarts=4)
    torch.manual_seed(0)
    xe, fe = bt_ops(paths, bounds, raw_samples=64, num_restarts=4)
    assert xa.shape == xe.shape and fa.shape == fe.shape
    C.check_bound("optimize_posterior_samples shortfall", (fe - fa).max().item(), 1e-9)
    C.check("optimize_posterior_samples optimum", fa, torch.ones_like(fa), 1e-9)
    # Matheron sample paths of a GP (BoTorch's paths, given to both).
    from botorch.sampling.pathwise import draw_matheron_paths

    bt, _, _ = model_pair(ModelSpec(n=8, d=2), 5)
    torch.manual_seed(1)
    paths = draw_matheron_paths(bt, torch.Size([3]))
    for ret in (False, True):
        torch.manual_seed(0)
        xa, fa = rs_ops(paths, bounds, raw_samples=128, num_restarts=6, return_transformed=ret)
        torch.manual_seed(0)
        xe, fe = bt_ops(paths, bounds, raw_samples=128, num_restarts=6, return_transformed=ret)
        assert xa.shape == xe.shape and fa.shape == fe.shape
        C.check_bound("optimize_posterior_samples Matheron shortfall", (fe - fa).max().item(), 1e-9)
        C.check("optimize_posterior_samples Matheron", fa, fe, 1e-9)


def test_buffer_dict_and_multi_objective_leftovers() -> None:
    from botorch.utils.multi_objective.hypervolume import MultiList as BML
    from botorch.utils.multi_objective.hypervolume import Node as BN
    from botorch.utils.multi_objective.hypervolume import sort_by_dimension as bsort
    from botorch.utils.torch import BufferDict as BBD

    from botorch_rs.utils.multi_objective.hypervolume import MultiList as RML
    from botorch_rs.utils.multi_objective.hypervolume import Node as RN
    from botorch_rs.utils.multi_objective.hypervolume import sort_by_dimension as rsort
    from botorch_rs.utils.torch import BufferDict as RBD

    da = RBD({"a": torch.ones(2), "b": torch.zeros(3)})
    de = BBD({"a": torch.ones(2), "b": torch.zeros(3)})
    assert list(da.keys()) == list(de.keys())
    assert {k: v.tolist() for k, v in da.state_dict().items()} == {
        k: v.tolist() for k, v in de.state_dict().items()
    }
    da["c"] = torch.full((1,), 2.0)
    de["c"] = torch.full((1,), 2.0)
    del da["a"]
    del de["a"]
    assert list(da.keys()) == list(de.keys()) and len(da) == len(de)
    assert isinstance(da, torch.nn.Module)

    data = torch.tensor([[1.0, 3.0], [2.0, 1.0], [0.5, 2.0], [2.0, 0.5]], dtype=DT)
    for i in (0, 1):
        na = [RN(m=2, dtype=DT, device=data.device, data=x) for x in data]
        ne = [BN(m=2, dtype=DT, device=data.device, data=x) for x in data]
        rsort(na, i)
        bsort(ne, i)
        assert [n.data.tolist() for n in na] == [n.data.tolist() for n in ne]
    la, le = RML(m=2, dtype=DT, device=data.device), BML(m=2, dtype=DT, device=data.device)
    na = [RN(m=2, dtype=DT, device=data.device, data=x) for x in data]
    ne = [BN(m=2, dtype=DT, device=data.device, data=x) for x in data]
    la.extend(na, 0)
    le.extend(ne, 0)
    la.extend(na[:2], 1)
    le.extend(ne[:2], 1)

    def walk(ml: Any) -> list[list[list[float]]]:
        out = []
        for i in range(ml.m):
            node, items = ml.sentinel.next[i], []
            while node is not ml.sentinel:
                items.append(node.data.tolist())
                node = node.next[i]
            out.append(items)
        return out

    assert walk(la) == walk(le)
    ba, be = torch.zeros(2, dtype=DT), torch.zeros(2, dtype=DT)
    la.remove(na[1], 0, ba)
    le.remove(ne[1], 0, be)
    assert walk(la) == walk(le) and torch.equal(ba, be)
    la.reinsert(na[1], 0, ba)
    le.reinsert(ne[1], 0, be)
    assert walk(la) == walk(le) and torch.equal(ba, be)


def test_pymoo_problem_and_repair() -> None:
    pytest.importorskip("pymoo")
    import numpy as np
    from botorch.acquisition.multioutput_acquisition import MultiOutputPosteriorMean as BtM
    from botorch.utils.multi_objective import optimize as bt_opt

    from botorch_rs.acquisition.multioutput_acquisition import MultiOutputPosteriorMean as RsM
    from botorch_rs.utils.multi_objective import optimize as rs_opt

    bt, rs, _ = model_pair(ModelSpec(n=8, d=2, m=2), 6)
    x = torch.rand(7, 2, generator=gen(95), dtype=DT).numpy()
    kw = {
        "n_var": 2,
        "n_obj": 2,
        "xl": np.zeros(2),
        "xu": np.ones(2),
        "dtype": DT,
        "device": torch.device("cpu"),
        "ref_point": torch.tensor([1.0, 2.0], dtype=DT),
        "constraints": [lambda Y: Y[..., 0] - 2.0],
        "inequality_constraints": [(torch.tensor([0, 1]), torch.tensor([1.0, 1.0], dtype=DT), 0.5)],
    }
    pa = rs_opt.BotorchPymooProblem(acqf=RsM(rs), **kw)
    pe = bt_opt.BotorchPymooProblem(acqf=BtM(bt), **kw)
    oa, oe = pa.evaluate(x, return_as_dictionary=True), pe.evaluate(x, return_as_dictionary=True)
    for key in ("F", "G"):
        C.check(
            f"BotorchPymooProblem {key}", torch.as_tensor(oa[key]), torch.as_tensor(oe[key]), TOL
        )
    choices = {0: [0.0, 0.25, 1.0], 1: [0.5]}
    ra = rs_opt.DiscreteParameterRepair(choices)._do(pa, x.copy())
    re = bt_opt.DiscreteParameterRepair(choices)._do(pe, x.copy())
    assert np.array_equal(ra, re)


@pytest.mark.parametrize(
    "name",
    [
        "qExpectedHypervolumeImprovement",
        "qNoisyExpectedHypervolumeImprovement",
        "qLogNoisyExpectedHypervolumeImprovement",
        "qKnowledgeGradient",
    ],
)
def test_iid_sampler_in_mo_and_kg(name: str) -> None:
    import botorch.acquisition as bta
    import botorch.acquisition.multi_objective as btm
    from botorch.sampling import IIDNormalSampler as BtIID
    from botorch.utils.multi_objective.box_decompositions.non_dominated import (
        FastNondominatedPartitioning as BtP,
    )

    import botorch_rs.acquisition as rsa
    import botorch_rs.acquisition.multi_objective as rsm
    from botorch_rs.sampling import IIDNormalSampler as RsIID
    from botorch_rs.utils.multi_objective.box_decompositions.non_dominated import (
        FastNondominatedPartitioning as RsP,
    )

    m = 1 if name == "qKnowledgeGradient" else 2
    bt, rs, X = model_pair(ModelSpec(n=8, d=2, m=m), 7)
    ref = torch.tensor([1.0, 2.0], dtype=DT)
    if name == "qKnowledgeGradient":
        a = rsa.qKnowledgeGradient(
            rs,
            num_fantasies=4,
            sampler=RsIID(torch.Size([4]), seed=2),
            inner_sampler=RsIID(torch.Size([8]), seed=3),
        )
        e = bta.qKnowledgeGradient(
            bt,
            num_fantasies=4,
            sampler=BtIID(torch.Size([4]), seed=2),
            inner_sampler=BtIID(torch.Size([8]), seed=3),
        )
        Xc = torch.rand(3, 2 + 4, 2, generator=gen(96), dtype=DT)
    elif name == "qExpectedHypervolumeImprovement":
        Y = bt.posterior(X).mean.detach()
        a = rsm.qExpectedHypervolumeImprovement(
            rs, ref, RsP(ref, Y), sampler=RsIID(torch.Size([32]), seed=4)
        )
        e = btm.qExpectedHypervolumeImprovement(
            bt, ref, BtP(ref, Y), sampler=BtIID(torch.Size([32]), seed=4)
        )
        Xc = torch.rand(3, 2, 2, generator=gen(96), dtype=DT)
    else:
        kw = {"X_baseline": X, "prune_baseline": False, "cache_root": False}
        a = getattr(rsm, name)(rs, ref, sampler=RsIID(torch.Size([32]), seed=4), **kw)
        e = getattr(btm, name)(bt, ref, sampler=BtIID(torch.Size([32]), seed=4), **kw)
        Xc = torch.rand(3, 2, 2, generator=gen(96), dtype=DT)
    C.check(f"IIDNormalSampler {name} value", a(Xc), e(Xc), TOL)
    (ga,) = grad_of(a, Xc)
    (ge,) = grad_of(e, Xc)
    C.check(f"IIDNormalSampler {name} grad", ga, ge, GTOL)

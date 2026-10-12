"""The randomized equivalence suite of pathwise sampling and the
information-theoretic multi-objective acquisition functions: botorch_rs
against live BoTorch over a seeded grid of models and options.

Scenarios (``SCENARIOS``) are ``SingleTaskGP`` models with m = 1, 2, 3
outputs and ``ModelListGP`` models, with n = 6, 10, 14 training points,
d = 1-3 inputs, inferred or fixed observation noise, with or without
``Normalize`` / ``Standardize``, RBF or Matern-5/2 kernels and several input
ranges. For every scenario both libraries are built on identical data and
hyperparameters, seeded generators draw the sample shapes, numbers of
features, test points, pending points, ``q``, the optimization direction and
the partitionings, and BoTorch's random numbers (path draws, candidate
sets, MC base samples) are passed to botorch_rs. Compared are

* Matheron paths, their prior and update paths, prior paths of
  ``draw_kernel_feature_paths`` and ``MatheronPathModel``: values and
  gradients at inputs with and without batch dimensions;
* ``sample_optimal_points`` (identical Pareto sets, or the same error) and
  ``compute_sample_box_decomposition`` of BoTorch's fronts (identical);
* MES and JES with every estimate ("0", "LB", "LB2", "MC"): values and
  gradients (JES on fixed-noise models: the same error);
* PES: values, gradients and the cached EP factors.

Tolerances (the worst observed error of every quantity is printed in the
terminal summary, section "equivalence suite (MO entropy search)"):

=====================================  =======================================
path values / gradients                1e-9 / 1e-8 (relative)
Pareto sets, box decompositions        identical (cell bounds to 1e-12)
MES / JES / PES values / gradients     1e-9 / 1e-8, or 10x BoTorch's own
                                       change under a 1e-14 relative
                                       perturbation of the training inputs
                                       (and, for PES, last-bit noise in exp
                                       and erf) if larger; recorded scaled
PES whose EP iteration count BoTorch   0.5 (relative; BoTorch's EP is decided
changes under that noise               by rounding there; counted)
=====================================  =======================================

Run with ``pytest -m equivalence``; skipped unless BoTorch is importable,
which ``BOTORCH_RS_REQUIRE_BOTORCH=1`` turns into an error.
"""

from __future__ import annotations

import contextlib
import functools
import random
import zlib
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import pytest

from helpers import ParityLog, importorskip_botorch

importorskip_botorch()
torch = pytest.importorskip("torch")

import botorch.acquisition.multi_objective.utils as bt_utils  # noqa: E402
from botorch.acquisition.multi_objective import joint_entropy_search as bt_jes  # noqa: E402
from botorch.acquisition.multi_objective import max_value_entropy_search as bt_mes  # noqa: E402
from botorch.acquisition.multi_objective import predictive_entropy_search as bt_pes  # noqa: E402
from botorch.models import deterministic as bt_det  # noqa: E402
from botorch.sampling import pathwise as bt_pw  # noqa: E402
from botorch.utils.multi_objective import box_decompositions as bt_bd  # noqa: E402
from mo_entropy_helpers import (  # noqa: E402
    bt_path_draws,
    jes_pareto_sets,
    raw_train_inputs,
    recorded_optima,
    recorded_randn,
    ulp_noise,
)

import botorch_rs.acquisition.multi_objective.utils as rs_utils  # noqa: E402
from botorch_rs.acquisition.multi_objective import joint_entropy_search as rs_jes  # noqa: E402
from botorch_rs.acquisition.multi_objective import max_value_entropy_search as rs_mes  # noqa: E402
from botorch_rs.acquisition.multi_objective import predictive_entropy_search as rs_pes  # noqa: E402
from botorch_rs.models import deterministic as rs_det  # noqa: E402
from botorch_rs.sampling import SobolQMCNormalSampler as RsSampler  # noqa: E402
from botorch_rs.sampling import pathwise as rs_pw  # noqa: E402
from parity_helpers import (  # noqa: E402
    Checker,
    ModelSpec,
    dense_model_list,
    list_pair,
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
UNSTABLE_TOL = 0.5
CHECK = Checker("equivalence suite (MO entropy search)")


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
    is_list: bool
    seed: int

    @property
    def d(self) -> int:
        return self.specs[0].d

    @property
    def multi_output_list(self) -> bool:
        return self.is_list and any(s.m > 1 for s in self.specs)

    @property
    def fixed_noise(self) -> bool:
        return any(s.fixed_noise for s in self.specs)


def _scenarios() -> list[Scenario]:
    r = random.Random(SEED)
    out: list[Scenario] = []
    ranges = [(0.0, 1.0), (-2.0, 3.0), (1.0, 4.0)]
    for i in range(9):
        lo, hi = r.choice(ranges)
        spec = ModelSpec(
            n=[6, 10, 14][i % 3],
            d=[1, 2, 3][(i // 3) % 3],
            m=[1, 2, 3][i % 3],
            fixed_noise=r.random() < 0.3,
            normalize=r.random() < 0.5,
            standardize=r.random() < 0.75,
            kernel=r.choice(["rbf", "matern52"]),
            lo=lo,
            hi=hi,
        )
        out.append(Scenario(f"{i:02d}_{spec.label}", (spec,), False, 500 + i))
    lists = [
        (ModelSpec(10, 2), ModelSpec(8, 2, kernel="matern52", normalize=True)),
        (ModelSpec(9, 3, fixed_noise=True), ModelSpec(12, 3, standardize=False)),
        (ModelSpec(8, 2), ModelSpec(7, 2, m=2, kernel="matern52")),
    ]
    for j, specs in enumerate(lists):
        label = "list_" + "+".join(s.label for s in specs)
        out.append(Scenario(f"{9 + j:02d}_{label}", specs, True, 600 + j))
    return out


SCENARIOS = _scenarios()


@pytest.fixture(scope="module", params=SCENARIOS, ids=[s.label for s in SCENARIOS])
def pair(request: pytest.FixtureRequest) -> tuple[Scenario, Any, Any, Any]:
    sc = request.param
    if sc.is_list:
        return (sc, *list_pair(sc.specs, sc.seed))
    return (sc, *model_pair(sc.specs[0], sc.seed))


def _floored(quantity: str, a: Any, e: Any, nominal: float, floor: float, atol: float = 0.0):
    err = rel_err(a, e, atol)
    tol = max(nominal, NOISE_FACTOR * floor)
    CHECK.record(quantity + " (scaled)", err * nominal / tol, nominal)
    # The unscaled error and BoTorch's own change under the perturbations.
    ParityLog.record(quantity + " (unscaled)", err, section=CHECK.section)
    ParityLog.record(quantity + " (BoTorch's noise floor)", floor, section=CHECK.section)
    assert err <= tol, (quantity, err, tol)


def _as_list(v: Any) -> list[Any]:
    return list(v) if isinstance(v, list) else [v]


def _bounds(X: Any) -> Any:
    X = X.reshape(-1, X.shape[-1])
    return torch.stack([X.min(0).values, X.max(0).values])


def _compare(name: str, b: Any, r: Any, X: Any) -> None:
    """Values and gradients of two paths (or path models) at ``X``, or the
    same error."""
    try:
        expected = b(X)
    except RuntimeError:
        with pytest.raises(RuntimeError):
            r(X)
        CHECK.record(f"{name}: broadcast errors (identical)", 0.0, 0.0)
        return
    for a, e in zip(_as_list(r(X)), _as_list(expected), strict=True):
        assert tuple(a.shape) == tuple(e.shape), (name, a.shape, e.shape)
        CHECK.check(f"{name}: values", a, e.detach(), VALUE_TOL)
    Xb = X.detach().clone().requires_grad_(True)
    (gb,) = torch.autograd.grad(sum(v.sum() for v in _as_list(b(Xb))), Xb)
    Xr = X.detach().clone().requires_grad_(True)
    (gr,) = torch.autograd.grad(sum(v.sum() for v in _as_list(r(Xr))), Xr)
    CHECK.check(f"{name}: gradients", gr, gb, GRAD_TOL)


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------


def test_paths(pair: tuple[Scenario, Any, Any, Any]) -> None:
    sc, bt, rs, X = pair
    r = rng_for("paths", sc.label)
    shape = r.choice([(), (2,), (2, 3)])
    features = r.choice([32, 64, 256])
    torch.manual_seed(r.randrange(2**31))
    noises: list[torch.Tensor] = []
    with recorded_randn(noises):
        bp = bt_pw.draw_matheron_paths(
            bt,
            torch.Size(shape),
            prior_sampler=functools.partial(bt_pw.draw_kernel_feature_paths, num_features=features),
        )
    rp = rs_pw.draw_matheron_paths(
        rs,
        torch.Size(shape),
        functools.partial(rs_pw.draw_kernel_feature_paths, num_features=features),
        draws=bt_path_draws(bp, noises),
    )
    d = sc.d
    inputs = [
        points_in(X, (r.randint(1, 6), d), r.randrange(10**6)),
        points_in(X, (r.choice([1, 2, 3]), r.randint(1, 4), d), r.randrange(10**6)),
    ]
    for Xs in inputs:
        _compare("Matheron paths", bp, rp, Xs)
        subs = list(zip(bp, rp, strict=True)) if isinstance(bp, bt_pw.PathList) else [(bp, rp)]
        for b_sub, r_sub in subs:
            for comp in ("prior_paths", "update_paths"):
                _compare(comp, b_sub[comp], r_sub[comp], Xs)
    # Prior paths with the outcome transform.
    torch.manual_seed(r.randrange(2**31))
    bpr = bt_pw.draw_kernel_feature_paths(bt, torch.Size(shape), num_features=features)
    rpr = rs_pw.draw_kernel_feature_paths(
        rs, shape, num_features=features, draws=bt_path_draws(bpr, None)
    )
    _compare("prior paths", bpr, rpr, inputs[0])
    # The path model of one sample.
    if not sc.multi_output_list:
        torch.manual_seed(r.randrange(2**31))
        noises = []
        with recorded_randn(noises):
            bm = bt_det.MatheronPathModel(bt)
        rm = rs_det.MatheronPathModel(rs, draws=bt_path_draws(bm._path, noises))
        CHECK.check(
            "MatheronPathModel values",
            rm.posterior(inputs[1]).mean,
            bm.posterior(inputs[1]).mean.detach(),
            VALUE_TOL,
        )


# ---------------------------------------------------------------------------
# Sampled optima and box decompositions
# ---------------------------------------------------------------------------


def test_optima_and_boxes(pair: tuple[Scenario, Any, Any, Any]) -> None:
    sc, bt, rs, X = pair
    if sc.multi_output_list:
        pytest.skip("BoTorch's path models do not support lists of multi-output models")
    r = rng_for("optima", sc.label)
    maximize = r.random() < 0.6
    num_points = 1 if rs.num_outputs == 1 else r.randint(2, 5)
    pop = r.choice([128, 256, 512])
    bounds = _bounds(X)
    draws: list[list[Any]] = []
    candidates: list[list[Any]] = []
    torch.manual_seed(r.randrange(2**31))
    kwargs = {"maximize": maximize, "optimizer_kwargs": {"pop_size": pop}}
    try:
        with recorded_optima(draws, candidates):
            ps_b, pf_b = bt_utils.sample_optimal_points(bt, bounds, 3, num_points, **kwargs)
    except RuntimeError as e:
        with pytest.raises(RuntimeError) as e_rs:
            rs_utils.sample_optimal_points(
                rs, bounds, len(draws), num_points, draws=draws, candidates=candidates, **kwargs
            )
        assert str(e_rs.value) == str(e)
        CHECK.record("random search errors (identical)", 0.0, 0.0)
        return
    ps_r, pf_r = rs_utils.sample_optimal_points(
        rs, bounds, 3, num_points, draws=draws, candidates=candidates, **kwargs
    )
    CHECK.check("Pareto sets (identical)", ps_r, ps_b, 0.0)
    CHECK.check("Pareto fronts", pf_r, pf_b, VALUE_TOL)
    partitioning = r.choice(
        [
            bt_bd.DominatedPartitioning,
            bt_bd.FastNondominatedPartitioning,
            bt_bd.NondominatedPartitioning,
        ]
    )
    k = r.choice([0, 0, 1, 2])
    e = bt_utils.compute_sample_box_decomposition(pf_b, partitioning, maximize, k)
    a = rs_utils.compute_sample_box_decomposition(pf_b, partitioning, maximize, k)
    assert tuple(a.shape) == tuple(e.shape)
    CHECK.check("sample box decompositions", a, e, EXACT_TOL)


def _rs_optima(rs: Any, X: Any, seed: int, maximize: bool) -> tuple[Any, Any]:
    """Pareto sets and fronts of botorch_rs (the inputs of both libraries)."""
    # (Fronts of correlated objectives may have fewer optimal points.)
    for num_points in (3, 2, 1) if rs.num_outputs > 1 else (1,):
        for attempt in range(5):
            try:
                return rs_utils.sample_optimal_points(
                    rs,
                    _bounds(X),
                    3,
                    num_points,
                    maximize=maximize,
                    optimizer_kwargs={"pop_size": 512},
                    num_features=128,
                    seed=seed + 1000 * attempt,
                )
            except RuntimeError:
                continue
    raise AssertionError("no seed gives enough optimal points")


# ---------------------------------------------------------------------------
# MES and JES
# ---------------------------------------------------------------------------


def _bt_entropy_search(kind: str, model: Any, ps: Any, pf: Any, cells: Any, est: str, Xp: Any):
    if kind == "MES":
        return bt_mes.qLowerBoundMultiObjectiveMaxValueEntropySearch(
            model, cells, X_pending=Xp, estimation_type=est, num_samples=16
        )
    return bt_jes.qLowerBoundMultiObjectiveJointEntropySearch(
        model,
        jes_pareto_sets(model, ps),
        pf,
        cells,
        X_pending=Xp,
        estimation_type=est,
        num_samples=16,
    )


def _rs_entropy_search(
    kind: str, rs: Any, ps: Any, pf: Any, cells: Any, est: str, Xp: Any, base: Any
):
    sampler = RsSampler([16])
    if base is not None:
        sampler.base_samples = base.reshape(16, -1).clone()
    if kind == "MES":
        return rs_mes.qLowerBoundMultiObjectiveMaxValueEntropySearch(
            rs, cells, X_pending=Xp, estimation_type=est, sampler=sampler
        )
    return rs_jes.qLowerBoundMultiObjectiveJointEntropySearch(
        rs, ps, pf, cells, X_pending=Xp, estimation_type=est, sampler=sampler
    )


@pytest.mark.parametrize("kind", ["MES", "JES"])
def test_entropy_search(pair: tuple[Scenario, Any, Any, Any], kind: str) -> None:
    sc, bt, rs, X = pair
    if sc.multi_output_list:
        pytest.skip("BoTorch's path models do not support lists of multi-output models")
    r = rng_for("entropy", kind, sc.label)
    maximize = r.random() < 0.7
    ps, pf = _rs_optima(rs, X, r.randrange(10**6), maximize)
    cells = bt_utils.compute_sample_box_decomposition(pf, maximize=maximize)
    if kind == "JES" and sc.fixed_noise:
        with pytest.raises(Exception) as e_bt:
            bt_jes.qLowerBoundMultiObjectiveJointEntropySearch(bt, ps, pf, cells)
        with pytest.raises(RuntimeError) as e_rs:
            rs_jes.qLowerBoundMultiObjectiveJointEntropySearch(rs, ps, pf, cells)
        assert str(e_bt.value) in str(e_rs.value)
        CHECK.record("JES fixed-noise errors (identical)", 0.0, 0.0)
        return
    for est in ("0", "LB", "LB2", "MC"):
        q = r.randint(1, 3)
        Xt = points_in(X, (r.randint(1, 3), q, sc.d), r.randrange(10**6))
        Xp = points_in(X, (r.randint(1, 2), sc.d), r.randrange(10**6)) if r.random() < 0.4 else None
        torch.manual_seed(0)
        with dense_model_list():
            b = _bt_entropy_search(kind, bt, ps, pf, cells, est, Xp)
            vb, gb = values_and_grads(b, Xt)
            floors_v, floors_g = [], []
            for twin in range(2):
                tw = _bt_entropy_search(
                    kind, perturbed_copy(bt, seed=71 + twin), ps, pf, cells, est, Xp
                )
                if est == "MC":
                    tw.sampler.base_samples = b.sampler.base_samples
                vt, gt = values_and_grads(tw, Xt)
                floors_v.append(rel_err(vt, vb))
                floors_g.append(rel_err(gt, gb))
        base = b.sampler.base_samples if est == "MC" else None
        acqf = _rs_entropy_search(kind, rs, ps, pf, cells, est, Xp, base)
        va, ga = values_and_grads(acqf, Xt)
        _floored(f"{kind}-{est} values", va, vb, VALUE_TOL, max(floors_v))
        _floored(f"{kind}-{est} gradients", ga, gb, GRAD_TOL, max(floors_g))


# ---------------------------------------------------------------------------
# PES
# ---------------------------------------------------------------------------


class _IterationCounter:
    """Counts BoTorch's EP iterations (calls of
    ``_update_damping_when_converged``)."""

    def __init__(self) -> None:
        self.count = 0
        self._orig = bt_pes._update_damping_when_converged

    def __enter__(self) -> _IterationCounter:
        def counted(*args: Any, **kwargs: Any) -> Any:
            self.count += 1
            return self._orig(*args, **kwargs)

        bt_pes._update_damping_when_converged = counted
        return self

    def __exit__(self, *exc: Any) -> None:
        bt_pes._update_damping_when_converged = self._orig


def _pes_gradients(ga: Any, gb: Any, twins: list[Any]) -> None:
    """PES gradients where BoTorch's are finite: BoTorch's autograd returns NaN
    where it differentiates a clamped logarithm of zero, botorch_rs the
    derivative of the clamped function (finite)."""
    assert bool(torch.isfinite(ga).all())
    mask = torch.isfinite(gb)
    if not bool(mask.all()):
        CHECK.record("PES gradients: NaN in BoTorch, finite in botorch_rs", 0.0, 0.0)
    floor = max(rel_err(t[2][mask], gb[mask], atol=1e-14) for t in twins)
    _floored("PES gradients", ga[mask], gb[mask], GRAD_TOL, floor, atol=1e-14)


def _bt_pes(model: Any, ps: Any, maximize: bool, Xp: Any, Xt: Any, noise_seed: int | None):
    noise = ulp_noise(noise_seed) if noise_seed is not None else contextlib.nullcontext()
    with dense_model_list(), noise, _IterationCounter() as counter, raw_train_inputs(model):
        acqf = bt_pes.qMultiObjectivePredictiveEntropySearch(
            model, ps, maximize=maximize, X_pending=Xp
        )
        v, g = values_and_grads(acqf, Xt)
    return acqf, v, g, counter.count


def test_pes(pair: tuple[Scenario, Any, Any, Any]) -> None:
    sc, bt, rs, X = pair
    if sc.multi_output_list:
        pytest.skip("BoTorch's path models do not support lists of multi-output models")
    r = rng_for("pes", sc.label)
    maximize = r.random() < 0.7
    ps, _ = _rs_optima(rs, X, r.randrange(10**6), maximize)
    q = r.randint(1, 2)
    Xt = points_in(X, (2, q, sc.d), r.randrange(10**6))
    Xp = points_in(X, (1, sc.d), r.randrange(10**6)) if r.random() < 0.4 else None
    try:
        b, vb, gb, iters = _bt_pes(bt, ps, maximize, Xp, Xt, None)
    except (RuntimeError, ValueError) as e:
        # EP failed in BoTorch (a covariance without a Cholesky factor).
        with pytest.raises((RuntimeError, ValueError)):
            rs_pes.qMultiObjectivePredictiveEntropySearch(rs, ps, maximize=maximize, X_pending=Xp)(
                Xt
            )
        CHECK.record(f"PES errors (both fail: {type(e).__name__})", 0.0, 0.0)
        return
    twins = [
        _bt_pes(perturbed_copy(bt, seed=81), ps, maximize, Xp, Xt, None),
        _bt_pes(bt, ps, maximize, Xp, Xt, 91),
    ]
    acqf = rs_pes.qMultiObjectivePredictiveEntropySearch(rs, ps, maximize=maximize, X_pending=Xp)
    va, ga = values_and_grads(acqf, Xt)
    if any(t[3] != iters for t in twins):
        CHECK.check("PES values (unstable EP, gross)", va, vb, UNSTABLE_TOL)
        return
    assert acqf.ep_iterations == iters
    N = b.augmented_X.shape[-2] - ps.shape[-2]
    e_cov = b._omega_f_nat_cov[..., N:, :, :, :].detach()
    floor_c = max(rel_err(t[0]._omega_f_nat_cov[..., N:, :, :, :], e_cov) for t in twins)
    _, covs = acqf.cached_factors
    _floored("PES cached factors", covs, e_cov, VALUE_TOL, floor_c)
    _floored("PES values", va, vb, VALUE_TOL, max(rel_err(t[1], vb) for t in twins))
    _pes_gradients(ga, gb, twins)

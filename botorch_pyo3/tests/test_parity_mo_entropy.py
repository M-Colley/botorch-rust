"""Parity of pathwise sampling and the information-theoretic multi-objective
acquisition functions with BoTorch, through the public APIs of both
libraries side by side, with BoTorch's random numbers passed to botorch_rs
(see ``mo_entropy_helpers``):

* ``draw_matheron_paths`` (sample shapes ``()``, ``(3,)``, ``(2, 2)``; inputs
  with and without batch dimensions, including BoTorch's broadcasting of the
  input batch with the sample and output dimensions), its sub-paths,
  weights and frequencies, ``draw_kernel_feature_paths``,
  ``gaussian_update`` and ``MatheronPathModel``: values and gradients on
  single- and multi-output ``SingleTaskGP`` models (RBF, Matern-5/2,
  ``ScaleKernel``, ``Normalize``, fixed noise) and ``ModelListGP`` models;
* ``sample_optimal_points`` (maximization and minimization, one and several
  objectives), ``random_search_optimizer`` (with its error) and
  ``compute_sample_box_decomposition`` (all partitionings, constraints,
  ``M = 1``): identical discrete outputs;
* qLowerBoundMultiObjectiveMaxValueEntropySearch and
  qLowerBoundMultiObjectiveJointEntropySearch with the estimates "0",
  "LB", "LB2" and "MC" (identical base samples), ``q = 1, 2``, pending
  points and t-batches, and the ``_compute_entropy_*`` estimators;
* qMultiObjectivePredictiveEntropySearch (EP iterations, cached factors,
  values, gradients) and ``log_cdf_robust``;
* ``optimize_acqf`` on JES from identical initial conditions;
* errors.

Tolerances: values 1e-9 and gradients 1e-8 (relative, mixed elementwise /
normwise), or 10 times BoTorch's own change under a 1e-14 relative
perturbation of the training inputs if larger (the entropy estimates are
differences of entropies; recorded scaled to the nominal tolerance). PES
also takes BoTorch's change under last-bit noise in ``exp`` and ``erf``
into account (EP amplifies it); when that noise changes BoTorch's number of
EP iterations, its EP is decided by rounding and only a gross agreement is
checked. Deviations from BoTorch main (see ``docs/drafts/mo-entropy.md``):
JES conditions on the raw Pareto sets (meta-pytorch/botorch#3399; BoTorch
without the fix gets untransformed sets) and PES uses the raw training
inputs (BoTorch is constructed in train mode, where they are raw).

Skipped unless BoTorch is importable (required in CI's parity job).
"""

from __future__ import annotations

import functools
import warnings
from collections.abc import Iterator
from typing import Any

import pytest

from helpers import importorskip_botorch

importorskip_botorch()
torch = pytest.importorskip("torch")

import botorch.acquisition.multi_objective.utils as bt_utils  # noqa: E402
import gpytorch  # noqa: E402
from botorch.acquisition.multi_objective import joint_entropy_search as bt_jes  # noqa: E402
from botorch.acquisition.multi_objective import max_value_entropy_search as bt_mes  # noqa: E402
from botorch.acquisition.multi_objective import predictive_entropy_search as bt_pes  # noqa: E402
from botorch.models import SingleTaskGP as BtSingleTaskGP  # noqa: E402
from botorch.models import deterministic as bt_det  # noqa: E402
from botorch.optim import optimize as bt_optimize  # noqa: E402
from botorch.sampling import pathwise as bt_pw  # noqa: E402
from botorch.utils.multi_objective import box_decompositions as bt_bd  # noqa: E402
from mo_entropy_helpers import (  # noqa: E402
    DT,
    bt_path_draws,
    jes_pareto_sets,
    raw_train_inputs,
    recorded_candidates,
    recorded_optima,
    recorded_randn,
    ulp_noise,
)

import botorch_rs.acquisition.multi_objective.utils as rs_utils  # noqa: E402
from botorch_rs import kernels as rs_kernels  # noqa: E402
from botorch_rs.acquisition.multi_objective import joint_entropy_search as rs_jes  # noqa: E402
from botorch_rs.acquisition.multi_objective import max_value_entropy_search as rs_mes  # noqa: E402
from botorch_rs.acquisition.multi_objective import predictive_entropy_search as rs_pes  # noqa: E402
from botorch_rs.exceptions.errors import UnsupportedError as RsUnsupportedError  # noqa: E402
from botorch_rs.models import SingleTaskGP as RsSingleTaskGP  # noqa: E402
from botorch_rs.models import deterministic as rs_det  # noqa: E402
from botorch_rs.optim import optimize as rs_optimize  # noqa: E402
from botorch_rs.sampling import SobolQMCNormalSampler as RsSampler  # noqa: E402
from botorch_rs.sampling import pathwise as rs_pw  # noqa: E402
from botorch_rs.utils.multi_objective import box_decompositions as rs_bd  # noqa: E402
from parity_helpers import (  # noqa: E402
    Checker,
    ModelSpec,
    copy_hyperparameters,
    dense_model_list,
    list_pair,
    make_data,
    model_pair,
    perturbed_copy,
    points_in,
    rel_err,
    set_hyperparameters,
    values_and_grads,
)

pytestmark = pytest.mark.filterwarnings("ignore")

VALUE_TOL = 1e-9
GRAD_TOL = 1e-8
EXACT_TOL = 1e-12
NOISE_FACTOR = 10.0
FEATURES = 64
CHECK = Checker("pathwise sampling and MO entropy search")


@pytest.fixture(autouse=True, scope="module")
def _single_threaded_torch() -> Iterator[None]:
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(threads)


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


def scale_kernel_pair(seed: int) -> tuple[Any, Any, Any]:
    """A SingleTaskGP with ``ScaleKernel(MaternKernel(nu=2.5))`` in both
    libraries."""
    spec = ModelSpec(n=8, d=2)
    X, Y, _ = make_data(spec, seed)
    bt = BtSingleTaskGP(
        X,
        Y,
        covar_module=gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.MaternKernel(nu=2.5, ard_num_dims=2)
        ),
    )
    rs = RsSingleTaskGP(
        X,
        Y,
        covar_module=rs_kernels.ScaleKernel(rs_kernels.MaternKernel(nu=2.5, ard_num_dims=2)),
    )
    set_hyperparameters(bt, seed)
    copy_hyperparameters(bt, rs)
    bt.eval()
    return bt, rs, X


CASES: dict[str, Any] = {
    "m1_rbf": lambda: model_pair(ModelSpec(n=9, d=2), 3),
    "m2_matern_norm": lambda: model_pair(
        ModelSpec(n=10, d=3, m=2, normalize=True, kernel="matern52", lo=-1.0, hi=2.0), 5
    ),
    "m1_fixed": lambda: model_pair(ModelSpec(n=8, d=2, fixed_noise=True), 7),
    "m1_scale": lambda: scale_kernel_pair(11),
    "m3_nostd": lambda: model_pair(ModelSpec(n=12, d=2, m=3, standardize=False), 13),
    "list": lambda: list_pair(
        [ModelSpec(n=8, d=2), ModelSpec(n=9, d=2, m=2, kernel="matern52")], 17
    ),
    "list11": lambda: list_pair([ModelSpec(n=8, d=2), ModelSpec(n=9, d=2, kernel="matern52")], 19),
}


def _floored(quantity: str, a: Any, e: Any, nominal: float, floor: float, atol: float = 0.0):
    """Checks ``a`` against ``e`` with tolerance ``max(nominal, 10 floor)``
    (recorded scaled to ``nominal``)."""
    err = rel_err(a, e, atol)
    tol = max(nominal, NOISE_FACTOR * floor)
    CHECK.record(quantity + " (scaled)", err * nominal / tol, nominal)
    assert err <= tol, (quantity, err, tol)


# ---------------------------------------------------------------------------
# Pathwise sampling
# ---------------------------------------------------------------------------


def _paths_pair(bt: Any, rs: Any, sample_shape: tuple[int, ...], seed: int) -> tuple[Any, Any]:
    torch.manual_seed(seed)
    noises: list[torch.Tensor] = []
    with recorded_randn(noises):
        bp = bt_pw.draw_matheron_paths(
            bt,
            torch.Size(sample_shape),
            prior_sampler=functools.partial(bt_pw.draw_kernel_feature_paths, num_features=FEATURES),
        )
    draws = bt_path_draws(bp, noises)
    rp = rs_pw.draw_matheron_paths(
        rs,
        torch.Size(sample_shape),
        functools.partial(rs_pw.draw_kernel_feature_paths, num_features=FEATURES),
        draws=draws,
    )
    return bp, rp


def _as_list(v: Any) -> list[Any]:
    return list(v) if isinstance(v, list) else [v]


def _compare_paths(name: str, bp: Any, rp: Any, X: Any) -> None:
    """Values and gradients of two paths at ``X`` (or the same error)."""
    try:
        expected = bp(X)
    except RuntimeError:
        with pytest.raises(RuntimeError):
            rp(X)
        return
    actual = rp(X)
    for a, e in zip(_as_list(actual), _as_list(expected), strict=True):
        assert tuple(a.shape) == tuple(e.shape), (name, a.shape, e.shape)
        CHECK.check(f"{name}: values", a, e.detach(), VALUE_TOL)
    Xb = X.detach().clone().requires_grad_(True)
    (gb,) = torch.autograd.grad(sum(v.sum() for v in _as_list(bp(Xb))), Xb)
    Xr = X.detach().clone().requires_grad_(True)
    (gr,) = torch.autograd.grad(sum(v.sum() for v in _as_list(rp(Xr))), Xr)
    CHECK.check(f"{name}: gradients", gr, gb, GRAD_TOL)


def _inputs(X: Any, seed: int) -> list[Any]:
    d = X.shape[-1]
    return [
        points_in(X, (5, d), seed),
        points_in(X, (1, 4, d), seed + 1),
        points_in(X, (2, 4, d), seed + 2),
        points_in(X, (3, 1, 4, d), seed + 3),
    ]


@pytest.mark.parametrize("sample_shape", [(), (3,), (2, 2)])
@pytest.mark.parametrize("case", list(CASES))
def test_matheron_paths_match_botorch(case: str, sample_shape: tuple[int, ...]) -> None:
    bt, rs, X = CASES[case]()
    bp, rp = _paths_pair(bt, rs, sample_shape, seed=len(case))
    for i, Xs in enumerate(_inputs(X, 100)):
        _compare_paths(f"Matheron paths ({case})", bp, rp, Xs)
        subs = list(zip(bp, rp, strict=True)) if isinstance(bp, bt_pw.PathList) else [(bp, rp)]
        for b_sub, r_sub in subs:
            assert isinstance(r_sub, rs_pw.MatheronPath)
            for comp in ("prior_paths", "update_paths"):
                if i == 0:
                    CHECK.check(
                        f"{comp} weights",
                        r_sub[comp].weight,
                        b_sub[comp].weight.detach(),
                        VALUE_TOL,
                    )
                try:
                    e = b_sub[comp](Xs)
                except RuntimeError:
                    continue
                CHECK.check(f"{comp} values", r_sub[comp](Xs), e.detach(), VALUE_TOL)
            CHECK.check(
                "Fourier frequencies",
                r_sub["prior_paths"].feature_map.weight,
                b_sub["prior_paths"].feature_map.weight.detach(),
                0.0,
            )
    # The random numbers of the paths reproduce them.
    subs = rp.paths if isinstance(rp, rs_pw.PathList) else [rp]
    redrawn = rs_pw.draw_matheron_paths(
        rs,
        torch.Size(sample_shape),
        functools.partial(rs_pw.draw_kernel_feature_paths, num_features=FEATURES),
        draws=[d for sub in subs for d in sub.draws],
    )
    Xs = points_in(X, (4, X.shape[-1]), 5)
    for a, e in zip(_as_list(redrawn(Xs)), _as_list(rp(Xs)), strict=True):
        assert torch.equal(a, e)


@pytest.mark.parametrize("sample_shape", [(), (4,)])
@pytest.mark.parametrize("case", ["m1_rbf", "m2_matern_norm", "m1_scale", "list"])
def test_kernel_feature_paths_match_botorch(case: str, sample_shape: tuple[int, ...]) -> None:
    bt, rs, X = CASES[case]()
    torch.manual_seed(1)
    bp = bt_pw.draw_kernel_feature_paths(bt, torch.Size(sample_shape), num_features=FEATURES)
    draws = bt_path_draws(bp, None)
    rp = rs_pw.draw_kernel_feature_paths(rs, sample_shape, num_features=FEATURES, draws=draws)
    for Xs in _inputs(X, 200):
        _compare_paths(f"prior paths ({case})", bp, rp, Xs)
    if not isinstance(bp, bt_pw.PathList):
        CHECK.check("prior path weights", rp.weight, bp.weight.detach(), 0.0)


@pytest.mark.parametrize("case", ["m1_rbf", "m2_matern_norm", "m1_fixed", "m3_nostd"])
def test_gaussian_update_matches_botorch(case: str) -> None:
    bt, rs, X = CASES[case]()
    m, n = rs.num_outputs, X.shape[-2]
    g = torch.Generator().manual_seed(3)
    values = torch.randn(3, *((m,) if m > 1 else ()), n, generator=g, dtype=DT)
    torch.manual_seed(4)
    noises: list[torch.Tensor] = []
    with recorded_randn(noises):
        bu = bt_pw.gaussian_update(bt, values)
    ru = rs_pw.gaussian_update(rs, values, noise=noises[0])
    CHECK.check("gaussian_update weights", ru.weight, bu.weight.detach(), VALUE_TOL)
    for Xs in _inputs(X, 300)[:2]:
        _compare_paths(f"gaussian_update ({case})", bu, ru, Xs)


@pytest.mark.parametrize("sample_shape", [None, (3,)])
@pytest.mark.parametrize("case", ["m1_rbf", "m2_matern_norm", "m3_nostd"])
def test_matheron_path_model_matches_botorch(case: str, sample_shape: Any) -> None:
    bt, rs, X = CASES[case]()
    torch.manual_seed(5)
    noises: list[torch.Tensor] = []
    shape = None if sample_shape is None else torch.Size(sample_shape)
    with recorded_randn(noises):
        bm = bt_det.MatheronPathModel(bt, sample_shape=shape)
    rm = rs_det.MatheronPathModel(rs, sample_shape=shape, draws=bt_path_draws(bm._path, noises))
    assert rm.num_outputs == bm.num_outputs
    assert tuple(rm.batch_shape) == tuple(bm.batch_shape)
    d = X.shape[-1]
    inputs = [points_in(X, (5, d), 1), points_in(X, (2, 4, d), 2)]
    if sample_shape is not None:
        inputs = [points_in(X, (5, d), 1), points_in(X, (3, 4, d), 2)]
    for Xs in inputs:
        e = bm.posterior(Xs).mean.detach()
        a = rm.posterior(Xs).mean
        assert tuple(a.shape) == tuple(e.shape)
        CHECK.check("MatheronPathModel values", a, e, VALUE_TOL)


def test_matheron_path_model_list_errors() -> None:
    """BoTorch's path models do not support model lists of multi-output
    models; neither do botorch_rs's (the Rust core does)."""
    bt, rs, X = CASES["list"]()
    with pytest.raises(Exception) as e_bt:
        bt_det.MatheronPathModel(bt)
    with pytest.raises(RsUnsupportedError) as e_rs:
        rs_det.MatheronPathModel(rs)
    assert str(e_rs.value) == str(e_bt.value)
    with pytest.raises(Exception) as e_bt:
        bt_utils.sample_optimal_points(bt, _bounds(X), 2, 2)
    with pytest.raises(RsUnsupportedError) as e_rs:
        rs_utils.sample_optimal_points(rs, _bounds(X), 2, 2)
    assert str(e_rs.value) == str(e_bt.value)


# ---------------------------------------------------------------------------
# Sampled Pareto sets and fronts
# ---------------------------------------------------------------------------


def _bounds(X: Any) -> Any:
    lo, hi = X.min(0).values, X.max(0).values
    return torch.stack([lo, hi])


@pytest.mark.parametrize("maximize", [True, False])
@pytest.mark.parametrize("case", ["m1_rbf", "m2_matern_norm", "m3_nostd", "list11"])
def test_sample_optimal_points_match_botorch(case: str, maximize: bool) -> None:
    bt, rs, X = CASES[case]()
    bounds = _bounds(X.reshape(-1, X.shape[-1]))
    num_points = 1 if rs.num_outputs == 1 else 4
    draws: list[list[Any]] = []
    candidates: list[list[Any]] = []
    torch.manual_seed(7)

    def rs_points(num_samples: int) -> tuple[Any, Any]:
        return rs_utils.sample_optimal_points(
            rs,
            bounds,
            num_samples,
            num_points,
            maximize=maximize,
            optimizer_kwargs={"pop_size": 256},
            draws=draws,
            candidates=candidates,
        )

    try:
        with recorded_optima(draws, candidates):
            ps_b, pf_b = bt_utils.sample_optimal_points(
                bt, bounds, 3, num_points, maximize=maximize, optimizer_kwargs={"pop_size": 256}
            )
    except RuntimeError as e:
        # The random search of the last sample found too few optimal points:
        # the same error for the same random numbers.
        with pytest.raises(RuntimeError) as e_rs:
            rs_points(len(draws))
        assert str(e_rs.value) == str(e)
        CHECK.record("Pareto sets: random search errors (identical)", 0.0, 0.0)
        return
    ps_r, pf_r = rs_points(3)
    CHECK.check("Pareto sets (identical)", ps_r, ps_b, 0.0)
    CHECK.check("Pareto fronts", pf_r, pf_b, VALUE_TOL)


def test_random_search_optimizer_matches_botorch() -> None:
    bt, rs, X = CASES["m2_matern_norm"]()
    bounds = _bounds(X)
    torch.manual_seed(9)
    noises: list[torch.Tensor] = []
    with recorded_randn(noises):
        bm = bt_det.MatheronPathModel(bt)
    rm = rs_det.MatheronPathModel(rs, draws=bt_path_draws(bm._path, noises))
    for maximize in (True, False):
        cands: list[Any] = []
        with recorded_candidates(cands):
            xb, yb = bt_utils.random_search_optimizer(bm, bounds, 4, maximize, pop_size=256)
        xr, yr = rs_utils.random_search_optimizer(
            rm, bounds, 4, maximize, pop_size=256, candidates=cands
        )
        CHECK.check("random search inputs (identical)", xr, xb, 0.0)
        CHECK.check("random search outputs", yr, yb.detach(), VALUE_TOL)
    # Too few optimal points: the same error.
    cands = []
    with recorded_candidates(cands), pytest.raises(RuntimeError) as e_bt:
        bt_utils.random_search_optimizer(bm, bounds, 200, True, pop_size=16, max_tries=2)
    with pytest.raises(RuntimeError) as e_rs:
        rs_utils.random_search_optimizer(
            rm, bounds, 200, True, pop_size=16, max_tries=2, candidates=cands
        )
    assert str(e_rs.value) == str(e_bt.value)


def _fronts(seed: int, P: int, n: int, M: int) -> Any:
    """Random mutually non-dominated fronts (on a sphere) with ties."""
    g = torch.Generator().manual_seed(seed)
    z = torch.rand(P, n, M, generator=g, dtype=DT) + 0.1
    fronts = z / z.norm(dim=-1, keepdim=True)
    fronts[:, 0] = fronts[:, 1]  # a duplicate point
    return fronts


@pytest.mark.parametrize("num_constraints", [0, 2])
@pytest.mark.parametrize("maximize", [True, False])
@pytest.mark.parametrize("M", [1, 2, 3])
def test_compute_sample_box_decomposition_matches_botorch(
    M: int, maximize: bool, num_constraints: int
) -> None:
    fronts = _fronts(M, 3, 5, M)
    pairs = [
        (bt_bd.DominatedPartitioning, rs_bd.DominatedPartitioning),
        (bt_bd.FastNondominatedPartitioning, rs_bd.FastNondominatedPartitioning),
        (bt_bd.NondominatedPartitioning, rs_bd.NondominatedPartitioning),
    ]
    for bt_cls, rs_cls in pairs:
        e = bt_utils.compute_sample_box_decomposition(fronts, bt_cls, maximize, num_constraints)
        for cls in (rs_cls, bt_cls):
            a = rs_utils.compute_sample_box_decomposition(fronts, cls, maximize, num_constraints)
            assert tuple(a.shape) == tuple(e.shape)
            CHECK.check("sample box decompositions", a, e, EXACT_TOL)


# ---------------------------------------------------------------------------
# MES and JES
# ---------------------------------------------------------------------------


def _optima(rs: Any, X: Any, seed: int, maximize: bool = True) -> tuple[Any, Any]:
    """Pareto sets and fronts of botorch_rs (the inputs of both libraries):
    the first seed from ``seed`` whose random search finds enough points."""
    bounds = _bounds(X.reshape(-1, X.shape[-1]))
    num_points = 1 if rs.num_outputs == 1 else 4
    for attempt in range(10):
        try:
            return rs_utils.sample_optimal_points(
                rs,
                bounds,
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


def _entropy_search(
    lib: str, kind: str, model: Any, ps: Any, pf: Any, cells: Any, est: str, Xp: Any
) -> Any:
    if lib == "bt":
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
    raise AssertionError(lib)


def _rs_entropy_search(
    kind: str, rs: Any, ps: Any, pf: Any, cells: Any, est: str, Xp: Any, base: Any
) -> Any:
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


@pytest.mark.parametrize("est", ["0", "LB", "LB2", "MC"])
@pytest.mark.parametrize("kind", ["MES", "JES"])
@pytest.mark.parametrize("case", ["m1_rbf", "m2_matern_norm", "m3_nostd", "list11"])
def test_entropy_search_matches_botorch(case: str, kind: str, est: str) -> None:
    bt, rs, X = CASES[case]()
    ps, pf = _optima(rs, X, seed=len(case))
    cells = bt_utils.compute_sample_box_decomposition(pf)
    d = X.shape[-1]
    for q, pending in ((1, False), (2, True)):
        Xt = points_in(X, (3, q, d), 10 + q)
        Xp = points_in(X, (1, d), 20) if pending else None
        torch.manual_seed(0)
        with dense_model_list():
            b = _entropy_search("bt", kind, bt, ps, pf, cells, est, Xp)
            vb, gb = values_and_grads(b, Xt)
            floors_v, floors_g = [], []
            for twin in range(2):
                tw = perturbed_copy(bt, seed=31 + twin)
                bt_twin = _entropy_search("bt", kind, tw, ps, pf, cells, est, Xp)
                if est == "MC":
                    bt_twin.sampler.base_samples = b.sampler.base_samples
                vt, gt = values_and_grads(bt_twin, Xt)
                floors_v.append(rel_err(vt, vb))
                floors_g.append(rel_err(gt, gb))
        base = b.sampler.base_samples if est == "MC" else None
        r = _rs_entropy_search(kind, rs, ps, pf, cells, est, Xp, base)
        va, ga = values_and_grads(r, Xt)
        _floored(f"{kind}-{est} values", va, vb, VALUE_TOL, max(floors_v))
        _floored(f"{kind}-{est} gradients", ga, gb, GRAD_TOL, max(floors_g))
        # q x d input without a t-batch.
        CHECK.check(f"{kind} unbatched input", r(Xt[0]), va[:1], 0.0)


def test_jes_fixed_noise_error_matches_botorch() -> None:
    bt, rs, X = CASES["m1_fixed"]()
    ps, pf = _optima(rs, X, seed=1)
    cells = bt_utils.compute_sample_box_decomposition(pf)
    with pytest.raises(Exception) as e_bt:
        bt_jes.qLowerBoundMultiObjectiveJointEntropySearch(bt, ps, pf, cells)
    with pytest.raises(RuntimeError) as e_rs:
        rs_jes.qLowerBoundMultiObjectiveJointEntropySearch(rs, ps, pf, cells)
    assert str(e_bt.value) in str(e_rs.value)


def test_entropy_estimators_match_botorch() -> None:
    g = torch.Generator().manual_seed(4)
    P, J, M, b, q, S = 3, 4, 2, 2, 3, 5
    lower = torch.rand(P, J, M, generator=g, dtype=DT) - 1.0
    upper = lower + torch.rand(P, J, M, generator=g, dtype=DT)
    lower[:, 0, :] = -1e10
    lower[0, -1, :] = 0.0
    upper[0, -1, :] = 0.0
    hb = torch.stack([lower, upper], dim=1)
    mean = torch.rand(b, P, q, 1, M, generator=g, dtype=DT) - 0.5
    var = 0.2 + torch.rand(b, P, q, 1, M, generator=g, dtype=DT)
    noise = 0.05 + 0.2 * torch.rand(b, P, q, 1, M, generator=g, dtype=DT)
    samples = 2.0 * torch.rand(S, b, P, q, 1, M, generator=g, dtype=DT) - 1.0
    lp = -1.0 - torch.rand(S, b, P, q, generator=g, dtype=DT)
    args = (hb, mean, var, noise)
    pairs = [
        (rs_jes._compute_entropy_noiseless(*args), bt_jes._compute_entropy_noiseless(*args)),
        (rs_jes._compute_entropy_upper_bound(*args), bt_jes._compute_entropy_upper_bound(*args)),
        (
            rs_jes._compute_entropy_upper_bound(*args, only_diagonal=True),
            bt_jes._compute_entropy_upper_bound(*args, only_diagonal=True),
        ),
        (
            rs_jes._compute_entropy_monte_carlo(*args, samples, lp),
            bt_jes._compute_entropy_monte_carlo(*args, samples, lp),
        ),
    ]
    for a, e in pairs:
        assert tuple(a.shape) == tuple(e.shape) == (b, q)
        CHECK.check("entropy estimators", a, e, EXACT_TOL)


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
    """BoTorch's PES (raw training inputs), its values and gradients at
    ``Xt`` and its number of EP iterations, optionally with last-bit noise in
    ``exp`` and ``erf``."""
    import contextlib

    noise = ulp_noise(noise_seed) if noise_seed is not None else contextlib.nullcontext()
    with dense_model_list(), noise, _IterationCounter() as counter, raw_train_inputs(model):
        acqf = bt_pes.qMultiObjectivePredictiveEntropySearch(
            model, ps, maximize=maximize, X_pending=Xp
        )
        v, g = values_and_grads(acqf, Xt)
    return acqf, v, g, counter.count


@pytest.mark.parametrize("case", ["m1_rbf", "m2_matern_norm", "m1_scale", "list11"])
def test_pes_matches_botorch(case: str) -> None:
    bt, rs, X = CASES[case]()
    d = X.shape[-1]
    for maximize, q, pending in ((True, 1, False), (False, 2, True)):
        ps, _ = _optima(rs, X, seed=3 + q, maximize=maximize)
        Xt = points_in(X, (3, q, d), 40 + q)
        Xp = points_in(X, (1, d), 50) if pending else None
        b, vb, gb, iters = _bt_pes(bt, ps, maximize, Xp, Xt, None)
        twins = [
            _bt_pes(perturbed_copy(bt, seed=61), ps, maximize, Xp, Xt, None),
            _bt_pes(bt, ps, maximize, Xp, Xt, 71),
        ]
        r = rs_pes.qMultiObjectivePredictiveEntropySearch(rs, ps, maximize=maximize, X_pending=Xp)
        va, ga = values_and_grads(r, Xt)
        floor_v = max(rel_err(t[1], vb) for t in twins)
        if any(t[3] != iters for t in twins):
            # BoTorch's EP is decided by rounding here.
            CHECK.check("PES values (unstable EP, gross)", va, vb, 0.5)
            continue
        assert r.ep_iterations == iters
        N = b.augmented_X.shape[-2] - ps.shape[-2]
        means, covs = r.cached_factors
        e_cov = b._omega_f_nat_cov[..., N:, :, :, :].detach()
        floor_c = max(rel_err(t[0]._omega_f_nat_cov[..., N:, :, :, :], e_cov) for t in twins)
        _floored("PES cached factors", covs, e_cov, VALUE_TOL, floor_c)
        _floored(
            "PES cached factor means",
            means,
            b._omega_f_nat_mean[..., N:, :, :].detach(),
            VALUE_TOL,
            floor_c,
        )
        _floored("PES values", va, vb, VALUE_TOL, floor_v)
        _pes_gradients(ga, gb, twins)


def test_log_cdf_robust_matches_botorch() -> None:
    x = torch.cat(
        [torch.linspace(-60.0, 60.0, 241, dtype=DT), torch.tensor([4.999, 5.0, 5.001], dtype=DT)]
    )
    CHECK.check("log_cdf_robust", rs_pes.log_cdf_robust(x), bt_pes.log_cdf_robust(x), EXACT_TOL)


# ---------------------------------------------------------------------------
# Optimization and errors
# ---------------------------------------------------------------------------


def test_optimize_acqf_with_jes_matches_botorch() -> None:
    bt, rs, X = CASES["m2_matern_norm"]()
    ps, pf = _optima(rs, X, seed=2)
    cells = bt_utils.compute_sample_box_decomposition(pf)
    b = bt_jes.qLowerBoundMultiObjectiveJointEntropySearch(bt, jes_pareto_sets(bt, ps), pf, cells)
    r = rs_jes.qLowerBoundMultiObjectiveJointEntropySearch(rs, ps, pf, cells)
    bounds = _bounds(X)
    ics = points_in(X, (4, 1, X.shape[-1]), 3).clamp(bounds[0], bounds[1])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cb, vb = bt_optimize.optimize_acqf(
            b, bounds, q=1, num_restarts=4, batch_initial_conditions=ics
        )
    cr, vr = rs_optimize.optimize_acqf(r, bounds, q=1, num_restarts=4, batch_initial_conditions=ics)
    # Same local optimum (L-BFGS-B stops on rounding-dependent criteria).
    CHECK.check("optimize_acqf JES value", vr, vb.detach(), 1e-6)
    CHECK.check("optimize_acqf JES candidate", cr, cb, 1e-4)


def test_errors_match_botorch() -> None:
    bt, rs, X = CASES["m2_matern_norm"]()
    ps, pf = _optima(rs, X, seed=1)
    cells = bt_utils.compute_sample_box_decomposition(pf)
    cases = [
        (
            lambda lib: (
                bt_mes if lib == "bt" else rs_mes
            ).qLowerBoundMultiObjectiveMaxValueEntropySearch(
                bt if lib == "bt" else rs, cells, estimation_type="XX"
            ),
            NotImplementedError,
        ),
        (
            lambda lib: (
                bt_mes if lib == "bt" else rs_mes
            ).qLowerBoundMultiObjectiveMaxValueEntropySearch(bt if lib == "bt" else rs, cells[0]),
            Exception,
        ),
        (
            lambda lib: (
                bt_jes if lib == "bt" else rs_jes
            ).qLowerBoundMultiObjectiveJointEntropySearch(
                bt if lib == "bt" else rs, ps[0], pf, cells
            ),
            Exception,
        ),
        (
            lambda lib: (bt_pes if lib == "bt" else rs_pes).qMultiObjectivePredictiveEntropySearch(
                bt if lib == "bt" else rs, ps[..., :1]
            ),
            Exception,
        ),
        (
            lambda lib: (bt_utils if lib == "bt" else rs_utils).compute_sample_box_decomposition(
                pf[0]
            ),
            Exception,
        ),
    ]
    for make, exc in cases:
        with pytest.raises(exc) as e_bt:
            make("bt")
        with pytest.raises(exc) as e_rs:
            make("rs")
        assert type(e_rs.value).__name__ == type(e_bt.value).__name__
        assert str(e_rs.value) == str(e_bt.value)
    bt1, rs1, X1 = CASES["m1_rbf"]()
    with pytest.raises(Exception) as e_bt:
        bt_utils.sample_optimal_points(bt1, _bounds(X1), 2, 2)
    with pytest.raises(RsUnsupportedError) as e_rs:
        rs_utils.sample_optimal_points(rs1, _bounds(X1), 2, 2)
    assert str(e_rs.value) == str(e_bt.value)

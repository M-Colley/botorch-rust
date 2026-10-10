"""Parity with BoTorch for MC objectives, outcome constraints, polytope
sampling and constrained acquisition optimization, through the Python API.

BoTorch and botorch_rs run side by side on identical data, hyperparameters
and base samples:

* Monte-Carlo acquisition values and gradients (qEI, qPI, qLogEI, qSR, qUCB,
  qLogNEI, qLogPF) on single-output, multi-output and ``ModelListGP`` models
  with ``LinearMCObjective``, ``GenericMCObjective`` (also depending on
  ``X``), ``ConstrainedMCObjective`` and outcome ``constraints`` (one or two
  constraints, per-constraint ``eta``, ``fat`` as a bool or a list with
  probability constraints), and qLogNEI with baselines that are infeasible in
  some or all samples (BoTorch's random lower-bound points are replaced by
  explicit points in both libraries);
* ``prune_inferior_points`` with objectives and constraints;
* the objective classes called directly;
* ``sparse_to_dense_constraints``, ``transform_constraints``,
  ``evaluate_feasibility`` and ``find_interior_point`` (exact), and the
  hit-and-run polytope samplers (feasibility and moments, statistically);
* ``optimize_acqf`` with linear (intra- and inter-point) inequality and
  equality constraints, nonlinear constraints and fixed features: from
  identical initial conditions (one restart per SLSQP problem) at least 80%
  of all restarts reach SciPy's result (relative value error ``<= 1e-6``);
  SciPy >= 1.16 solves the SLSQP subproblems with LAPACK, so trajectories
  differ at the 1e-9 level and may split on flat landscapes (in either
  direction); over seeds the best value is at least BoTorch's (statistical);
  error types and messages match. With released BoTorch versions that reject
  infeasible starts for nonlinear constraints, only feasible starts are used.

Values: relative error ``<= 1e-9`` (log-space values: absolute below 1);
gradients ``<= 1e-8``. Float tensors (float64) are passed for ``eta``,
``tau``, ``best_f`` and ``infeasible_cost`` because BoTorch stores Python
floats in float32. Skipped unless BoTorch is importable (required in CI's
parity job).
"""

from __future__ import annotations

import contextlib
import math
import warnings
from collections.abc import Iterator
from typing import Any

import pytest

from helpers import importorskip_botorch

importorskip_botorch()
torch = pytest.importorskip("torch")

from botorch.acquisition import logei as bt_logei  # noqa: E402
from botorch.acquisition import monte_carlo as bt_mc  # noqa: E402
from botorch.acquisition import objective as bt_objective  # noqa: E402
from botorch.acquisition.analytic import LogExpectedImprovement as BtLogEI  # noqa: E402
from botorch.acquisition.utils import prune_inferior_points as bt_prune  # noqa: E402
from botorch.exceptions.errors import (  # noqa: E402
    CandidateGenerationError as BtCandidateGenerationError,
)
from botorch.exceptions.warnings import BotorchWarning as BtBotorchWarning  # noqa: E402
from botorch.models import SingleTaskGP as BtSingleTaskGP  # noqa: E402
from botorch.optim import initializers as bt_initializers  # noqa: E402
from botorch.optim import optimize_acqf as bt_optimize_acqf  # noqa: E402
from botorch.optim import parameter_constraints as bt_pc  # noqa: E402
from botorch.sampling.normal import SobolQMCNormalSampler as BtSampler  # noqa: E402
from botorch.test_functions.synthetic import Hartmann  # noqa: E402
from botorch.utils import sampling as bt_sampling  # noqa: E402

from botorch_rs.acquisition import logei as rs_logei  # noqa: E402
from botorch_rs.acquisition import monte_carlo as rs_mc  # noqa: E402
from botorch_rs.acquisition import objective as rs_objective  # noqa: E402
from botorch_rs.acquisition.analytic import LogExpectedImprovement as RsLogEI  # noqa: E402
from botorch_rs.acquisition.utils import prune_inferior_points as rs_prune  # noqa: E402
from botorch_rs.exceptions.warnings import BotorchWarning as RsBotorchWarning  # noqa: E402
from botorch_rs.models import SingleTaskGP as RsSingleTaskGP  # noqa: E402
from botorch_rs.optim import initializers as rs_initializers  # noqa: E402
from botorch_rs.optim import optimize_acqf as rs_optimize_acqf  # noqa: E402
from botorch_rs.optim import parameter_constraints as rs_pc  # noqa: E402
from botorch_rs.sampling import SobolQMCNormalSampler as RsSampler  # noqa: E402
from botorch_rs.utils import sampling as rs_sampling  # noqa: E402
from parity_helpers import (  # noqa: E402
    DT,
    Checker,
    ModelSpec,
    copy_hyperparameters,
    dense_model_list,
    feasible_start_generator,
    feasible_starts,
    list_pair,
    model_pair,
    patched_lower_bound,
    points_in,
    sampler_pair,
    values_and_grads,
)

RTOL = 1e-9
GRAD_RTOL = 1e-8
CHECK = Checker("objectives, constraints, constrained optimization")

pytestmark = pytest.mark.filterwarnings("ignore")


@pytest.fixture(autouse=True, scope="module")
def _single_threaded_torch() -> Iterator[None]:
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(threads)


def _t(v: Any) -> Any:
    return torch.as_tensor(v, dtype=DT)


# ---------------------------------------------------------------------------
# Models, objectives and constraints
# ---------------------------------------------------------------------------

MODELS: dict[str, list[ModelSpec]] = {
    "m1": [ModelSpec(n=10, d=2)],
    "m2": [ModelSpec(n=12, d=2, m=2)],
    "m3_matern_norm": [
        ModelSpec(n=15, d=3, m=3, kernel="matern52", normalize=True, lo=-1.0, hi=2.0)
    ],
    "m2_fixed_nostd": [ModelSpec(n=9, d=2, m=2, fixed_noise=True, standardize=False)],
    "list_two": [ModelSpec(n=10, d=2), ModelSpec(n=8, d=2, normalize=True)],
    "list_mixed": [
        ModelSpec(n=12, d=3, m=2, kernel="matern52"),
        ModelSpec(n=9, d=3, fixed_noise=True),
    ],
}


@pytest.fixture(scope="module", params=list(MODELS))
def model(request: pytest.FixtureRequest) -> tuple[str, Any, Any, Any]:
    name = request.param
    specs = MODELS[name]
    seed = 1000 + 17 * list(MODELS).index(name)
    if name.startswith("list"):
        bt, rs, X = list_pair(specs, seed)
    else:
        bt, rs, X = model_pair(specs[0], seed)
    return name, bt, rs, X


def posterior_stats(bt: Any, X: Any) -> tuple[Any, Any]:
    """Posterior means and standard deviations (``k x m``) at ``X`` (``k x d``)."""
    with dense_model_list(), torch.no_grad():
        p = bt.posterior(X)
    m = p.mean.shape[-1]
    return p.mean.reshape(-1, m), p.variance.reshape(-1, m).clamp_min(0).sqrt()


def generic_objective(Y: Any, X: Any = None) -> Any:
    out = Y[..., 0] + 0.3 * torch.sin(Y[..., -1])
    if X is not None:
        out = out - 0.2 * (X**2).sum(-1)
    return out


def objective_pairs(m: int, means: Any) -> list[tuple[str, Any, Any]]:
    """(label, BoTorch objective, botorch_rs objective) for ``m`` outputs."""
    w = _t([0.6, -0.4, 0.25][:m])
    c = means[:, -1].median()
    cons = [lambda Y: Y[..., -1] - c]
    eta = _t([0.05])
    cost = _t(2.5)
    pairs: list[tuple[str, Any, Any]] = [
        ("linear", bt_objective.LinearMCObjective(w), rs_objective.LinearMCObjective(w)),
        (
            "generic",
            bt_objective.GenericMCObjective(generic_objective),
            rs_objective.GenericMCObjective(generic_objective),
        ),
        (
            "constrained",
            bt_objective.ConstrainedMCObjective(
                lambda Y, X=None: Y[..., 0], cons, infeasible_cost=cost, eta=eta
            ),
            rs_objective.ConstrainedMCObjective(
                lambda Y, X=None: Y[..., 0], cons, infeasible_cost=cost, eta=eta
            ),
        ),
    ]
    if m == 1:
        pairs.insert(0, ("none", None, None))
    return pairs


def constraint_sets(means: Any, sds: Any) -> list[tuple[str, list[Any], Any, Any]]:
    """(label, constraints, eta, fat override) with thresholds near the
    posterior medians, so that feasibility varies between samples."""
    c = means[:, -1].median()
    mu0 = means[:, 0].mean()
    r2 = (1.5 * sds[:, 0].mean()) ** 2

    def lin(Y: Any) -> Any:
        return Y[..., -1] - c

    def quad(Y: Any) -> Any:
        return (Y[..., 0] - mu0) ** 2 - r2

    def prob(Y: Any) -> Any:  # a probability of feasibility (fat=None)
        return torch.sigmoid(3.0 * (c - Y[..., -1]))

    return [
        ("linear", [lin], _t([1e-2]), None),
        ("linear+quadratic", [lin, quad], _t([1e-2, 0.1]), None),
        ("float eta", [lin], 0.0078125, None),  # exact in float32
        ("fat list", [lin, prob], _t([1e-2, 1.0]), [True, None]),
    ]


MC_ACQFS = [
    # name, BoTorch module, botorch_rs module, kwargs, supports constraints
    ("qExpectedImprovement", bt_mc, rs_mc, {"best_f": 0.0}, True),
    ("qProbabilityOfImprovement", bt_mc, rs_mc, {"best_f": 0.0, "tau": 0.1}, True),
    ("qLogExpectedImprovement", bt_logei, rs_logei, {"best_f": 0.0}, True),
    ("qLogExpectedImprovement", bt_logei, rs_logei, {"best_f": 0.0, "fat": False}, True),
    ("qSimpleRegret", bt_mc, rs_mc, {}, False),
    ("qUpperConfidenceBound", bt_mc, rs_mc, {"beta": 0.2}, False),
]


def _compare(label: str, a_bt: Any, a_rs: Any, X: Any, log: bool, s_bt: Any, Z: Any) -> None:
    with dense_model_list():
        v_bt, g_bt = values_and_grads(a_bt, X)
    assert torch.equal(s_bt.base_samples, Z), "BoTorch replaced the base samples"
    v_rs, g_rs = values_and_grads(a_rs, X)
    if log:
        CHECK.check_log(f"{label} value", v_rs, v_bt, RTOL)
    else:
        CHECK.check(f"{label} value", v_rs, v_bt, RTOL)
    CHECK.check(f"{label} grad", g_rs, g_bt, GRAD_RTOL, atol=1e-14)


# ---------------------------------------------------------------------------
# Monte-Carlo acquisition functions
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("q,n_pending", [(1, 0), (3, 0), (2, 2)])
def test_mc_objectives_and_constraints(model: tuple, q: int, n_pending: int) -> None:
    name, bt, rs, X = model
    m, d = rs.num_outputs, X.shape[-1]
    Xt = points_in(X, (2, q, d), seed=q + 7 * n_pending)
    pending = X[:n_pending] + 0.01 if n_pending else None
    means, sds = posterior_stats(bt, points_in(X, (8, d), seed=3))
    S, D = 64, q + n_pending
    for obj_label, o_bt, o_rs in objective_pairs(m, means):
        with torch.no_grad():
            if o_bt is None:
                obj_means = means[:, 0]
            elif obj_label == "constrained":
                obj_means = o_bt(means.unsqueeze(0)).squeeze(0)
            else:
                obj_means = o_bt(means.unsqueeze(0), X=None).squeeze(0)
        best = obj_means.max()
        for acqf_name, bt_mod, rs_mod, kwargs, supports_constraints in MC_ACQFS:
            if acqf_name.startswith("qLog") and obj_label == "constrained":
                continue  # BoTorch rejects ConstrainedMCObjective (tested below)
            kw = {
                k: (best.clone() if k == "best_f" else _t(v) if k == "tau" else v)
                for k, v in kwargs.items()
            }
            con_sets: list[Any] = [None]
            if supports_constraints and obj_label != "constrained":
                con_sets += constraint_sets(means, sds)
            for con in con_sets:
                extra: dict[str, Any] = {}
                label = f"{acqf_name}[{obj_label}"
                if con is not None:
                    c_label, cons, eta, fat = con
                    if fat is not None and (acqf_name != "qLogExpectedImprovement" or "fat" in kw):
                        continue  # fat lists exist only for the LogEI family
                    extra = {"constraints": cons, "eta": eta}
                    if fat is not None:
                        extra["fat"] = fat
                    label += f", {c_label} constraints"
                if kw.get("fat") is False:
                    label += ", fat=False"
                label += "]"
                s_bt, s_rs, Z = sampler_pair(S, D, m, seed=11 * q + n_pending)
                a_bt = getattr(bt_mod, acqf_name)(
                    bt, sampler=s_bt, objective=o_bt, X_pending=pending, **kw, **extra
                )
                a_rs = getattr(rs_mod, acqf_name)(
                    rs, sampler=s_rs, objective=o_rs, X_pending=pending, **kw, **extra
                )
                _compare(label, a_bt, a_rs, Xt, a_rs._log, s_bt, Z)


def test_botorch_objects_are_accepted(model: tuple) -> None:
    """BoTorch's own objective objects passed to botorch_rs acquisition
    functions give BoTorch's values."""
    name, bt, rs, X = model
    m, d = rs.num_outputs, X.shape[-1]
    Xt = points_in(X, (2, 2, d), seed=5)
    means, _ = posterior_stats(bt, points_in(X, (8, d), seed=3))
    for obj_label, o_bt, _ in objective_pairs(m, means):
        if o_bt is None:
            continue
        s_bt, s_rs, Z = sampler_pair(32, 2, m, seed=9)
        a_bt = bt_mc.qSimpleRegret(bt, sampler=s_bt, objective=o_bt)
        a_rs = rs_mc.qSimpleRegret(rs, sampler=s_rs, objective=o_bt)
        _compare(f"qSimpleRegret[BoTorch {obj_label} objective]", a_bt, a_rs, Xt, False, s_bt, Z)


@pytest.mark.parametrize("incremental,n_pending", [(True, 0), (True, 2), (False, 2)])
def test_qlognei_objectives_and_constraints(
    model: tuple, incremental: bool, n_pending: int
) -> None:
    name, bt, rs, X = model
    m, d, n = rs.num_outputs, X.shape[-1], X.shape[0]
    q = 2
    Xt = points_in(X, (2, q, d), seed=21 + n_pending)
    pending = X[:n_pending] + 0.01 if n_pending else None
    means, sds = posterior_stats(bt, X)
    lb_points = points_in(X, (32, d), seed=4)
    c_tight = (means[:, -1] - 0.5 * sds[:, -1]).min()
    c_impossible = (means[:, -1] - 30.0 * sds[:, -1] - 10.0).min()
    constraint_options: list[tuple[str, Any]] = [
        ("none", None),
        ("tight", [lambda Y: Y[..., -1] - c_tight]),
        ("infeasible", [lambda Y: Y[..., -1] - c_impossible]),
    ]
    for obj_label, o_bt, o_rs in objective_pairs(m, means):
        if obj_label == "constrained":
            continue
        for c_label, cons in constraint_options:
            extra = {} if cons is None else {"constraints": cons, "eta": _t([1e-2])}
            D = n + n_pending + q
            s_bt, s_rs, Z = sampler_pair(64, D, m, seed=D + 3 * n_pending)
            common = dict(
                X_baseline=X, X_pending=pending, prune_baseline=False, incremental=incremental
            )
            with warnings.catch_warnings(record=True) as w_bt:
                warnings.simplefilter("always")
                with patched_lower_bound(lb_points):
                    a_bt = bt_logei.qLogNoisyExpectedImprovement(
                        bt, sampler=s_bt, objective=o_bt, cache_root=False, **common, **extra
                    )
                    with dense_model_list():
                        v_bt, g_bt = values_and_grads(a_bt, Xt)
            assert torch.equal(s_bt.base_samples, Z)
            with warnings.catch_warnings(record=True) as w_rs:
                warnings.simplefilter("always")
                a_rs = rs_logei.qLogNoisyExpectedImprovement(
                    rs,
                    sampler=s_rs,
                    objective=o_rs,
                    lower_bound_points=lb_points,
                    **common,
                    **extra,
                )
                v_rs, g_rs = values_and_grads(a_rs, Xt)
            label = f"qLogNEI[{obj_label}, {c_label} baseline]"
            CHECK.check_log(f"{label} value", v_rs, v_bt, RTOL)
            CHECK.check(f"{label} grad", g_rs, g_bt, GRAD_RTOL, atol=1e-14)
            warned_bt = any(issubclass(w.category, BtBotorchWarning) for w in w_bt)
            warned_rs = any(issubclass(w.category, RsBotorchWarning) for w in w_rs)
            assert warned_rs == warned_bt, (label, warned_bt, warned_rs)


@pytest.mark.parametrize("q", [1, 3])
def test_qlogpf(model: tuple, q: int) -> None:
    name, bt, rs, X = model
    m, d = rs.num_outputs, X.shape[-1]
    Xt = points_in(X, (2, q, d), seed=31 + q)
    means, sds = posterior_stats(bt, points_in(X, (8, d), seed=3))
    for c_label, cons, eta, fat_list in constraint_sets(means, sds):
        for fat in [True, False] if fat_list is None else [fat_list]:
            s_bt, s_rs, Z = sampler_pair(64, q, m, seed=q)
            a_bt = bt_logei.qLogProbabilityOfFeasibility(
                bt, constraints=cons, sampler=s_bt, eta=eta, fat=fat
            )
            a_rs = rs_logei.qLogProbabilityOfFeasibility(
                rs, constraints=cons, sampler=s_rs, eta=eta, fat=fat
            )
            _compare(f"qLogPF[{c_label}, fat={fat}]", a_bt, a_rs, Xt, True, s_bt, Z)


def test_prune_with_objectives_and_constraints(model: tuple) -> None:
    name, bt, rs, X = model
    m = rs.num_outputs
    Xp = torch.cat([X, X + 0.03])
    means, sds = posterior_stats(bt, Xp)
    c_tight = means[:, -1].median()
    c_impossible = (means[:, -1] - 30.0 * sds[:, -1] - 10.0).min()
    for obj_label, o_bt, o_rs in objective_pairs(m, means):
        for c_label, cons in (
            ("none", None),
            ("mixed", [lambda Y: Y[..., -1] - c_tight]),
            ("infeasible", [lambda Y: Y[..., -1] - c_impossible]),
        ):
            # BoTorch draws the base samples of an unbatched `n x d` input (shape
            # `S x n [x m]`); botorch_rs gets a copy.
            s_bt = BtSampler(sample_shape=torch.Size([256]), seed=13)
            with dense_model_list():
                pruned_bt = bt_prune(bt, Xp, objective=o_bt, constraints=cons, sampler=s_bt)
            s_rs = RsSampler(sample_shape=[256], seed=0)
            s_rs.base_samples = s_bt.base_samples.clone()
            pruned_rs = rs_prune(rs, Xp, objective=o_rs, constraints=cons, sampler=s_rs)
            assert torch.equal(pruned_rs, pruned_bt), (obj_label, c_label)
            CHECK.record("prune_inferior_points with objectives/constraints (mismatches)", 0.0, 0.0)


def test_objective_errors_match_botorch(model: tuple) -> None:
    name, bt, rs, X = model
    m = rs.num_outputs
    means, _ = posterior_stats(bt, X)
    pairs = {label: (o_bt, o_rs) for label, o_bt, o_rs in objective_pairs(m, means)}
    o_bt, o_rs = pairs["constrained"]
    cons = [lambda Y: Y[..., -1]]
    for bt_cls, rs_cls in (
        (bt_logei.qLogExpectedImprovement, rs_logei.qLogExpectedImprovement),
        (bt_mc.qExpectedImprovement, rs_mc.qExpectedImprovement),
    ):
        for kwargs in ({}, {"constraints": cons}):
            try:
                bt_cls(bt, best_f=0.0, objective=o_bt, **kwargs)
            except Exception as e:
                with pytest.raises(Exception) as info:
                    rs_cls(rs, best_f=0.0, objective=o_rs, **kwargs)
                assert str(info.value) == str(e)
                assert isinstance(info.value, ValueError) == isinstance(e, ValueError)
            else:
                rs_cls(rs, best_f=0.0, objective=o_rs, **kwargs)
    if m > 1:
        for cls in ("qExpectedImprovement", "qSimpleRegret"):
            kw = {"best_f": 0.0} if cls == "qExpectedImprovement" else {}
            with pytest.raises(Exception) as e_bt:
                getattr(bt_mc, cls)(bt, **kw)
            with pytest.raises(Exception) as e_rs:
                getattr(rs_mc, cls)(rs, **kw)
            assert str(e_rs.value) == str(e_bt.value)


def test_mismatched_etas_and_fats() -> None:
    """BoTorch raises at the first evaluation, botorch_rs at construction,
    with the same messages."""
    bt, rs, X = model_pair(ModelSpec(n=8, d=2), seed=5)
    cons = [lambda Y: Y[..., 0], lambda Y: Y[..., 0] - 1.0]
    cases = [
        ({"eta": _t([1e-2])}, "qExpectedImprovement"),
        ({"eta": _t([1e-2, -1.0])}, "qExpectedImprovement"),
        ({"fat": [True]}, "qLogExpectedImprovement"),
    ]
    for kwargs, cls in cases:
        mod_bt, mod_rs = (bt_logei, rs_logei) if cls.startswith("qLog") else (bt_mc, rs_mc)
        a_bt = getattr(mod_bt, cls)(bt, best_f=0.0, constraints=cons, **kwargs)
        with pytest.raises(ValueError) as e_bt:
            a_bt(points_in(X, (1, 1, 2), seed=1))
        with pytest.raises(ValueError) as e_rs:
            getattr(mod_rs, cls)(rs, best_f=0.0, constraints=cons, **kwargs)
        assert str(e_rs.value) == str(e_bt.value)


def test_objectives_called_directly() -> None:
    g = torch.Generator().manual_seed(3)
    Y = torch.randn(5, 2, 3, 2, generator=g, dtype=DT)
    X = torch.rand(2, 3, 4, generator=g, dtype=DT)
    w = _t([0.7, -1.2])
    cons = [lambda Y: Y[..., 1] - 0.1, lambda Y: Y[..., 0] ** 2 - 1.0]
    pairs = [
        (bt_objective.IdentityMCObjective(), rs_objective.IdentityMCObjective(), Y[..., :1]),
        (bt_objective.LinearMCObjective(w), rs_objective.LinearMCObjective(w), Y),
        (
            bt_objective.GenericMCObjective(generic_objective),
            rs_objective.GenericMCObjective(generic_objective),
            Y,
        ),
        (
            bt_objective.ConstrainedMCObjective(
                generic_objective, cons, infeasible_cost=_t(1.5), eta=_t([0.05, 0.2])
            ),
            rs_objective.ConstrainedMCObjective(
                generic_objective, cons, infeasible_cost=_t(1.5), eta=_t([0.05, 0.2])
            ),
            Y,
        ),
    ]
    for o_bt, o_rs, samples in pairs:
        CHECK.check(
            f"{type(o_rs).__name__} called directly",
            o_rs(samples, X=X),
            o_bt(samples, X=X),
            1e-14,
        )
    with pytest.raises(RuntimeError) as e_bt:
        bt_objective.LinearMCObjective(w)(Y[..., :1])
    with pytest.raises(RuntimeError) as e_rs:
        rs_objective.LinearMCObjective(w)(Y[..., :1])
    assert str(e_rs.value) == str(e_bt.value)
    bad = bt_objective.GenericMCObjective(lambda Y, X: Y[..., 0, :])
    bad_rs = rs_objective.GenericMCObjective(lambda Y, X: Y[..., 0, :])
    with pytest.raises(RuntimeError) as e_bt:
        bad(Y, X=X)
    with pytest.raises(RuntimeError) as e_rs:
        bad_rs(Y, X=X)
    assert str(e_rs.value) == str(e_bt.value)


# ---------------------------------------------------------------------------
# Parameter constraints and polytope sampling
# ---------------------------------------------------------------------------

LINEAR_INEQ = [
    (torch.tensor([0, 1]), _t([1.0, 1.0]), 0.5),
    (torch.tensor([0, 2]), _t([-1.0, -0.5]), -3.5),
]
LINEAR_EQ = [(torch.tensor([0, 1, 2]), _t([1.0, 1.0, 1.0]), 4.0)]
POLY_BOUNDS = _t([[0.0, -1.0, 2.0], [2.0, 1.0, 5.0]])


def test_sparse_to_dense_and_transform_constraints() -> None:
    a_bt, b_bt = bt_sampling.sparse_to_dense_constraints(3, LINEAR_INEQ + LINEAR_EQ)
    a_rs, b_rs = rs_sampling.sparse_to_dense_constraints(3, LINEAR_INEQ + LINEAR_EQ)
    assert torch.equal(a_rs, a_bt) and torch.equal(b_rs, b_bt)
    inter = [(torch.tensor([[0, 0], [1, 2]]), _t([1.0, -2.0]), 0.3)]
    for cons in (LINEAR_INEQ, inter, LINEAR_INEQ + inter):
        t_bt = bt_initializers.transform_constraints(cons, q=3, d=3)
        t_rs = rs_initializers.transform_constraints(cons, q=3, d=3)
        assert len(t_rs) == len(t_bt)
        for (i_r, c_r, r_r), (i_b, c_b, r_b) in zip(t_rs, t_bt, strict=True):
            assert torch.equal(i_r, i_b) and torch.equal(c_r, c_b) and r_r == r_b
    assert rs_initializers.transform_constraints(None, q=2, d=3) is None
    for bad in (
        [(torch.tensor([0, 3]), _t([1.0, 1.0]), 0.0)],
        [(torch.tensor([[0, 3]]), _t([1.0]), 0.0)],
    ):
        with pytest.raises(ValueError) as e_bt:
            bt_initializers.transform_constraints(bad, q=2, d=3)
        with pytest.raises(ValueError) as e_rs:
            rs_initializers.transform_constraints(bad, q=2, d=3)
        assert str(e_rs.value) == str(e_bt.value)


def test_evaluate_feasibility() -> None:
    g = torch.Generator().manual_seed(0)
    X = torch.rand(64, 2, 3, generator=g, dtype=DT)
    X[:8, :, 0] = 1.0 - X[:8, :, 1]  # equality x0 + x1 = 1 satisfied exactly
    ineq = [
        (torch.tensor([0, 1]), _t([-1.0, -1.0]), -1.2),
        (torch.tensor([[0, 0], [1, 2]]), _t([1.0, -1.0]), -0.4),
    ]
    eq = [(torch.tensor([0, 1]), _t([1.0, 1.0]), 1.0)]

    def ball(x: Any) -> Any:
        return 0.6**2 - ((x - 0.5) ** 2).sum(-1)

    def separation(x: Any) -> Any:
        return ((x[0] - x[1]) ** 2).sum(-1) - 0.1

    nonlinear = [(ball, True), (separation, False)]
    for kwargs in (
        {"inequality_constraints": ineq},
        {"equality_constraints": eq},
        {"nonlinear_inequality_constraints": nonlinear},
        {"inequality_constraints": ineq, "nonlinear_inequality_constraints": nonlinear},
        {"inequality_constraints": ineq, "tolerance": 0.05},
    ):
        f_bt = bt_pc.evaluate_feasibility(X, **kwargs)
        f_rs = rs_pc.evaluate_feasibility(X, **kwargs)
        assert f_rs.dtype == torch.bool and torch.equal(f_rs, f_bt), kwargs
        assert 0 < int(f_bt.sum()) < 64 or "equality_constraints" in kwargs


def test_find_interior_point() -> None:
    np = pytest.importorskip("numpy")
    cases = [
        # The triangle x >= 0, y >= 0, x + 2y <= 2.
        (np.array([[-1.0, 0.0], [0.0, -1.0], [1.0, 2.0]]), np.array([[0.0], [0.0], [2.0]]), None),
        # The unit square with x + y = 1.2 (BoTorch's A_eq has the slack column).
        (
            np.array([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]]),
            np.array([[1.0], [0.0], [1.0], [0.0]]),
            (np.array([[1.0, 1.0, 0.0]]), np.array([[1.2]])),
        ),
    ]
    for A, b, eq in cases:
        kw = {} if eq is None else {"A_eq": eq[0], "b_eq": eq[1]}
        x_bt = bt_sampling.find_interior_point(A, b, **kw)
        x_rs = rs_sampling.find_interior_point(A, b, **kw)
        assert isinstance(x_rs, np.ndarray) and x_rs.shape == x_bt.shape
        CHECK.check("find_interior_point", x_rs, x_bt, 1e-9, atol=1e-12)
    from botorch.exceptions.errors import InfeasibilityError as BtInfeasibilityError

    from botorch_rs.exceptions import InfeasibilityError as RsInfeasibilityError

    A = np.array([[1.0], [-1.0]])
    b = np.array([[0.2], [-0.8]])  # x <= 0.2 and x >= 0.8
    with pytest.raises(BtInfeasibilityError) as e_bt:
        bt_sampling.find_interior_point(A, b)
    with pytest.raises(RsInfeasibilityError) as e_rs:
        rs_sampling.find_interior_point(A, b)
    assert str(e_rs.value) == str(e_bt.value)


def _moments(x: Any) -> tuple[Any, Any]:
    return x.mean(dim=0), torch.cov(x.T)


def _feasible(x: Any, bounds: Any, ineq: Any, eq: Any, tol: float = 1e-9) -> bool:
    ok = bool(((x >= bounds[0] - tol) & (x <= bounds[1] + tol)).all())
    for idx, coef, rhs in ineq or []:
        ok &= bool(((x[..., idx] * coef).sum(-1) >= rhs - tol).all())
    for idx, coef, rhs in eq or []:
        ok &= bool((((x[..., idx] * coef).sum(-1) - rhs).abs() <= 1e-8).all())
    return ok


@pytest.mark.parametrize("which", ["ineq", "ineq_eq", "eq"])
def test_polytope_samples(which: str) -> None:
    """All samples are feasible and their moments agree with BoTorch's
    (statistically: hit-and-run chains of 4000 samples)."""
    ineq = LINEAR_INEQ if "ineq" in which else None
    eq = LINEAR_EQ if "eq" in which.split("_") else None
    kw = dict(inequality_constraints=ineq, equality_constraints=eq, n_burnin=2000, n_thinning=10)
    x_bt = bt_sampling.get_polytope_samples(4000, POLY_BOUNDS, seed=0, **kw)
    x_rs = rs_sampling.get_polytope_samples(4000, POLY_BOUNDS, seed=0, **kw)
    assert x_rs.shape == x_bt.shape and _feasible(x_rs, POLY_BOUNDS, ineq, eq)
    (m_bt, c_bt), (m_rs, c_rs) = _moments(x_bt), _moments(x_rs)
    span = POLY_BOUNDS[1] - POLY_BOUNDS[0]
    mean_err = ((m_rs - m_bt).abs() / span).max().item()
    cov_err = ((c_rs - c_bt).abs() / torch.outer(span, span)).max().item()
    CHECK.check_bound("polytope samples: mean error / range (statistical)", mean_err, 0.03)
    CHECK.check_bound("polytope samples: covariance error / range^2 (statistical)", cov_err, 0.01)
    # Same seed, same samples; HitAndRunPolytopeSampler continues its chain.
    assert torch.equal(x_rs, rs_sampling.get_polytope_samples(4000, POLY_BOUNDS, seed=0, **kw))
    A, b = rs_sampling.sparse_to_dense_constraints(3, ineq) if ineq else (None, None)
    sampler = rs_sampling.HitAndRunPolytopeSampler(
        inequality_constraints=None if A is None else (-A, -b),
        equality_constraints=None if eq is None else rs_sampling.sparse_to_dense_constraints(3, eq),
        bounds=POLY_BOUNDS,
        n_burnin=100,
        n_thinning=5,
        seed=1,
    )
    d1, d2 = sampler.draw(50), sampler.draw(50)
    assert d1.shape == (50, 3) and not torch.equal(d1, d2)
    assert _feasible(torch.cat([d1, d2]), POLY_BOUNDS, ineq, eq)


def test_q_batches_from_polytope() -> None:
    b2 = _t([[0.0, 0.0], [1.0, 1.0]])
    inter = [(torch.tensor([[0, 0], [1, 0]]), _t([1.0, -1.0]), 0.2)]
    intra = [(torch.tensor([0, 1]), _t([-1.0, -1.0]), -1.0)]
    for cons in (inter, intra):
        kw = dict(n=3000, q=2, bounds=b2, n_burnin=2000, n_thinning=5, inequality_constraints=cons)
        x_bt = bt_initializers.sample_q_batches_from_polytope(seed=1, **kw)
        x_rs = rs_initializers.sample_q_batches_from_polytope(seed=1, **kw)
        assert x_rs.shape == x_bt.shape == (3000, 2, 2)
        assert bool(bt_pc.evaluate_feasibility(x_rs, inequality_constraints=cons).all())
        (m_bt, c_bt), (m_rs, c_rs) = (
            _moments(x_bt.reshape(3000, -1)),
            _moments(x_rs.reshape(3000, -1)),
        )
        CHECK.check_bound(
            "q-batch polytope samples: mean error (statistical)",
            (m_rs - m_bt).abs().max().item(),
            0.03,
        )
        CHECK.check_bound(
            "q-batch polytope samples: covariance error (statistical)",
            (c_rs - c_bt).abs().max().item(),
            0.01,
        )


# ---------------------------------------------------------------------------
# Constrained acquisition optimization
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def opt_models() -> dict[str, Any]:
    """A fixed-hyperparameter Hartmann-3 model with LogEI and qLogEI (identical
    base samples) in both libraries."""
    d = 3
    bounds = torch.stack([torch.zeros(d, dtype=DT), torch.ones(d, dtype=DT)])
    X = bt_sampling.draw_sobol_samples(bounds=bounds, n=15, q=1, seed=3).squeeze(-2)
    Y = -Hartmann(dim=d)(X, noise=False).unsqueeze(-1)
    bt = BtSingleTaskGP(X, Y)
    rs = RsSingleTaskGP(X, Y)
    bt.initialize(
        **{
            "likelihood.noise": _t([1e-3]),
            "mean_module.constant": _t(0.1),
            "covar_module.lengthscale": _t([[0.3, 0.45, 0.6]]),
        }
    )
    copy_hyperparameters(bt, rs)
    bt.eval()
    best_f = Y.max()
    s_bt, s_rs, Z = sampler_pair(64, 2, 1, seed=21)
    return {
        "bounds": bounds,
        "logei": (BtLogEI(bt, best_f=best_f), RsLogEI(rs, best_f=best_f)),
        "qlogei": (
            bt_logei.qLogExpectedImprovement(bt, best_f=best_f, sampler=s_bt),
            rs_logei.qLogExpectedImprovement(rs, best_f=best_f, sampler=s_rs),
        ),
    }


def ball(x: Any) -> Any:  # 0.35^2 - ||x - 0.4||^2 >= 0
    return 0.35**2 - ((x - 0.4) ** 2).sum(-1)


def separation(x: Any) -> Any:  # ||x_0 - x_1||^2 - 0.15 >= 0 (inter-point)
    return ((x[0] - x[1]) ** 2).sum(-1) - 0.15


# Feasible points of the nonlinear cases (for BoTorch versions that require
# feasible starts, see parity_helpers.feasible_starts).
FALLBACK = {
    "nonlinear_ball": _t([[0.4, 0.4, 0.4]]),
    "nonlinear_interpoint_q2": _t([[0.1, 0.1, 0.1], [0.5, 0.5, 0.1]]),
}
SUM_LE = (torch.tensor([0, 1, 2]), _t([-1.0, -1.0, -1.0]), -1.2)
EQ = (torch.tensor([0, 2]), _t([1.0, -1.0]), 0.1)
INTER = (torch.tensor([[0, 0], [1, 0]]), _t([1.0, -1.0]), 0.2)
OPT_CASES: list[dict[str, Any]] = [
    {"name": "linear_ineq", "acqf": "logei", "q": 1, "ineq": [SUM_LE]},
    {"name": "linear_eq_ineq", "acqf": "logei", "q": 1, "ineq": [SUM_LE], "eq": [EQ]},
    {"name": "inter_point_q2", "acqf": "qlogei", "q": 2, "ineq": [INTER, (*SUM_LE[:2], -1.5)]},
    {"name": "fixed_linear", "acqf": "logei", "q": 1, "ineq": [SUM_LE], "fixed": {1: 0.25}},
    {"name": "nonlinear_ball", "acqf": "logei", "q": 1, "nonlinear": [(ball, True)]},
    {
        "name": "nonlinear_interpoint_q2",
        "acqf": "qlogei",
        "q": 2,
        "nonlinear": [(separation, False)],
        "ineq": [SUM_LE],
    },
    {"name": "slsqp_box_only", "acqf": "qlogei", "q": 2, "options": {"method": "SLSQP"}},
]


def _constraint_kwargs(case: dict[str, Any]) -> dict[str, Any]:
    return {
        "inequality_constraints": case.get("ineq"),
        "equality_constraints": case.get("eq"),
        "nonlinear_inequality_constraints": case.get("nonlinear"),
        "fixed_features": case.get("fixed"),
    }


def _identical_start_runs(opt_models: dict, case: dict) -> tuple[Any, Any, Any, Any]:
    """BoTorch's and botorch_rs's per-restart values from the same initial
    conditions, one restart per SLSQP problem (``batch_limit=1``) and in
    BoTorch's default joint problem over all restarts (``None`` with
    nonlinear constraints, which require one restart per problem)."""
    a_bt, a_rs = opt_models[case["acqf"]]
    bounds, q = opt_models["bounds"], case["q"]
    torch.manual_seed(0)  # BoTorch selects initial conditions with torch's RNG
    ics = bt_initializers.gen_batch_initial_conditions(
        a_bt,
        bounds,
        q=q,
        num_restarts=8,
        raw_samples=64,
        options={"seed": 0},
        inequality_constraints=case.get("ineq"),
        equality_constraints=case.get("eq"),
        fixed_features=case.get("fixed"),
    )
    if case["name"] == "nonlinear_ball":  # one feasible and one infeasible start
        ics = _t([[[0.45, 0.5, 0.35]], [[0.9, 0.9, 0.1]]])
    ics = feasible_starts(ics, case.get("nonlinear"), FALLBACK.get(case["name"]))
    common = dict(
        q=q,
        num_restarts=ics.shape[0],
        batch_initial_conditions=ics,
        return_best_only=False,
        retry_on_optimization_warning=False,
        **_constraint_kwargs(case),
    )
    options = case.get("options") or {}
    single = {"options": {**options, "batch_limit": 1}}
    x_rs, v_rs = rs_optimize_acqf(a_rs, bounds, **single, **common)
    _, v_bt = bt_optimize_acqf(a_bt, bounds, **single, **common)
    feas = bt_pc.evaluate_feasibility(
        x_rs,
        inequality_constraints=case.get("ineq"),
        equality_constraints=case.get("eq"),
        nonlinear_inequality_constraints=case.get("nonlinear"),
    )
    assert bool(feas.any()), case["name"]
    if case.get("nonlinear"):
        return v_bt, v_rs, None, None
    _, jv_bt = bt_optimize_acqf(a_bt, bounds, options=options or None, **common)
    _, jv_rs = rs_optimize_acqf(a_rs, bounds, options=options or None, **common)
    return v_bt, v_rs, jv_bt, jv_rs


def test_slsqp_from_identical_initial_conditions(opt_models: dict) -> None:
    """SLSQP restarts from identical initial conditions reach SciPy's result
    (relative value error <= 1e-6) in at least 80% of all restarts and at
    least half of the restarts of every case; restarts starting on flat parts
    of the acquisition function (log values near -45) can end in different
    basins because SciPy >= 1.16 solves the subproblems with LAPACK, and such
    splits go either way. BoTorch's default joint problem over all restarts
    stops on the joint objective, so its restarts agree only loosely; it is
    reported, not asserted."""
    agree_total = n_total = 0
    for case in OPT_CASES:
        v_bt, v_rs, jv_bt, jv_rs = _identical_start_runs(opt_models, case)
        errs = (v_rs - v_bt).abs() / v_bt.abs().clamp_min(1.0)
        agree = errs <= 1e-6
        agree_total += int(agree.sum())
        n_total += agree.numel()
        CHECK.check_bound(
            "SLSQP from identical initial conditions: share of restarts not at SciPy's "
            "result (per case)",
            1.0 - agree.double().mean().item(),
            0.5,
        )
        for e in errs[agree]:
            CHECK.record(
                "SLSQP from identical initial conditions: relative value error (agreeing)",
                e.item(),
                1e-6,
            )
        wins = int(((v_rs - v_bt) > 1e-6 * v_bt.abs().clamp_min(1.0)).sum())
        losses = int((~agree).sum()) - wins
        CHECK.record(
            "SLSQP from identical initial conditions: diverging restarts botorch_rs better",
            float(wins),
            None,
        )
        CHECK.record(
            "SLSQP from identical initial conditions: diverging restarts SciPy better",
            float(losses),
            None,
        )
        if jv_bt is not None:
            j_err = (jv_rs - jv_bt).abs() / jv_bt.abs().clamp_min(1.0)
            CHECK.record(
                "SLSQP joint problem from identical initial conditions: share of restarts not "
                "at SciPy's result (informational)",
                1.0 - (j_err <= 1e-6).double().mean().item(),
                None,
            )
    CHECK.check_bound(
        "SLSQP from identical initial conditions: share of restarts not at SciPy's result",
        1.0 - agree_total / n_total,
        0.2,
    )


@pytest.mark.parametrize("case", OPT_CASES, ids=[c["name"] for c in OPT_CASES])
def test_constrained_optimize_acqf_quality(opt_models: dict, case: dict) -> None:
    a_bt, a_rs = opt_models[case["acqf"]]
    bounds, q = opt_models["bounds"], case["q"]
    kwargs = _constraint_kwargs(case)
    best_bt, best_rs = -math.inf, -math.inf
    for seed in range(3):
        options = {"seed": seed, **(case.get("options") or {})}
        extra_bt: dict[str, Any] = {}
        extra_rs: dict[str, Any] = {}
        if case.get("nonlinear"):
            nl, fb = case["nonlinear"], FALLBACK[case["name"]]
            extra_bt = {
                "ic_generator": feasible_start_generator(
                    bt_initializers.gen_batch_initial_conditions, nl, fb
                )
            }
            extra_rs = {
                "ic_generator": feasible_start_generator(
                    rs_initializers.gen_batch_initial_conditions, nl, fb
                )
            }
        torch.manual_seed(seed)
        try:
            _, v_bt = bt_optimize_acqf(
                a_bt,
                bounds,
                q=q,
                num_restarts=4,
                raw_samples=64,
                options=options,
                **kwargs,
                **extra_bt,
            )
        except BtCandidateGenerationError:
            v_bt = None  # released BoTorch versions without the projection give up
        x_rs, v_rs = rs_optimize_acqf(
            a_rs, bounds, q=q, num_restarts=4, raw_samples=64, options=options, **kwargs, **extra_rs
        )
        assert x_rs.shape == (q, 3)
        feas = bt_pc.evaluate_feasibility(
            x_rs.unsqueeze(0),
            inequality_constraints=case.get("ineq"),
            equality_constraints=case.get("eq"),
            nonlinear_inequality_constraints=case.get("nonlinear"),
        )
        assert bool(feas.all()), case["name"]
        for k, v in (case.get("fixed") or {}).items():
            assert bool((x_rs[:, k] == v).all())
        with torch.no_grad():
            check = a_bt(x_rs.unsqueeze(0)).squeeze()
        CHECK.check_log(
            "optimize_acqf (constrained): returned value vs BoTorch's value", v_rs, check, RTOL
        )
        if v_bt is not None:
            best_bt = max(best_bt, v_bt.item())
        best_rs = max(best_rs, v_rs.item())
    if best_bt == -math.inf:
        return
    shortfall = (best_bt - best_rs) / max(1.0, abs(best_bt))
    CHECK.check_bound(
        "optimize_acqf (constrained): best-value shortfall over seeds", shortfall, 1e-4
    )


def test_custom_ic_generator(opt_models: dict) -> None:
    """A user ``ic_generator`` is called with BoTorch's keyword arguments."""
    a_bt, a_rs = opt_models["logei"]
    bounds = opt_models["bounds"]
    calls: list[dict[str, Any]] = []
    ics = _t([[[0.3, 0.3, 0.3]], [[0.45, 0.5, 0.35]], [[0.35, 0.4, 0.5]]])  # feasible

    def generator(
        acq_function: Any, bounds: Any, q: int, num_restarts: int, raw_samples: int, **kwargs: Any
    ) -> Any:
        calls.append({"q": q, "num_restarts": num_restarts, "raw_samples": raw_samples, **kwargs})
        return ics[:num_restarts]

    common = dict(
        q=1,
        num_restarts=3,
        raw_samples=16,
        nonlinear_inequality_constraints=[(ball, True)],
        ic_generator=generator,
        return_best_only=False,
        retry_on_optimization_warning=False,
    )
    _, v_bt = bt_optimize_acqf(a_bt, bounds, **common)
    _, v_rs = rs_optimize_acqf(a_rs, bounds, **common)
    assert len(calls) == 2
    assert set(calls[0]) == set(calls[1])  # the same keyword arguments
    assert {k: v for k, v in calls[0].items() if not hasattr(v, "shape")} == {
        k: v for k, v in calls[1].items() if not hasattr(v, "shape")
    }
    CHECK.check_log("optimize_acqf with a custom ic_generator", v_rs, v_bt, 1e-6)


@contextlib.contextmanager
def _same_error() -> Iterator[list[BaseException]]:
    errors: list[BaseException] = []
    yield errors
    assert len(errors) == 2, errors
    e_bt, e_rs = errors
    assert str(e_rs) == str(e_bt), (str(e_rs), str(e_bt))
    for base in (ValueError, RuntimeError, NotImplementedError, RuntimeWarning):
        if isinstance(e_bt, base):
            assert isinstance(e_rs, base), (type(e_rs), type(e_bt))


def test_constrained_optimize_errors_match_botorch(opt_models: dict) -> None:
    a_bt, a_rs = opt_models["logei"]
    bounds = opt_models["bounds"]
    infeasible = [
        (torch.tensor([0]), _t([1.0]), 0.8),
        (torch.tensor([0]), _t([-1.0]), -0.2),
    ]
    cases: list[dict[str, Any]] = [
        {"nonlinear_inequality_constraints": [(ball, True)]},
        {"nonlinear_inequality_constraints": [[ball, True]], "ic_generator": "gen"},
        {"nonlinear_inequality_constraints": [(ball, True, 1)], "ic_generator": "gen"},
        {"inequality_constraints": [SUM_LE], "options": {"method": "L-BFGS-B"}},
        {
            "nonlinear_inequality_constraints": [(ball, True)],
            "ic_generator": "gen",
            "options": {"batch_limit": 2},
        },
        {"inequality_constraints": infeasible},
        {"raw_samples": None},
        {"inequality_constraints": [INTER], "sequential": True, "q": 2, "acqf": "qlogei"},
    ]
    for case in cases:
        case = dict(case)
        q = case.pop("q", 1)
        acqfs = opt_models[case.pop("acqf", "logei")]
        raw_samples = case.pop("raw_samples", 16)
        gen = case.pop("ic_generator", None)
        with _same_error() as errors:
            for lib, acqf, optimize, initializers in (
                ("bt", acqfs[0], bt_optimize_acqf, bt_initializers),
                ("rs", acqfs[1], rs_optimize_acqf, rs_initializers),
            ):
                kw = dict(case)
                if gen is not None:
                    kw["ic_generator"] = initializers.gen_batch_initial_conditions
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        optimize(acqf, bounds, q=q, num_restarts=2, raw_samples=raw_samples, **kw)
                except BaseException as e:  # noqa: BLE001 - compared below
                    errors.append(e)
                else:
                    raise AssertionError(f"{lib} did not raise for {case}")

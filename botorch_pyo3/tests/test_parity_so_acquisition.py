"""Parity of the single-objective acquisition functions of phase 8 with
BoTorch, through the public APIs of both libraries side by side, with
BoTorch's random numbers passed to botorch_rs (explicit base samples,
recorded Sobol draws, path draws and max values):

* qKnowledgeGradient (posterior mean, ``current_value``, pending points, MC
  objectives, posterior transforms, multi-output and ``ModelListGP``
  models), qMultiFidelityKnowledgeGradient (``project``, trace
  observations, cost-aware utilities, native value functions) and
  ``ProjectedAcquisitionFunction``; ``optimize_acqf`` of the one-shot
  qKG from identical initial conditions;
* qNoisyExpectedImprovement (``cache_root`` on and off, pending points,
  objectives, outcome constraints, multi-output and ``ModelListGP``
  models), qLowerConfidenceBound and qPosteriorStandardDeviation;
* the analytic NoisyExpectedImprovement and LogNoisyExpectedImprovement
  (BoTorch's recorded fantasy draws), LogProbabilityOfFeasibility,
  ScalarizedPosteriorMean and qAnalyticProbabilityOfImprovement;
* qMaxValueEntropy, qLowerBoundMaxValueEntropy (GIBBON),
  qMultiFidelityMaxValueEntropy and qMultiFidelityLowerBoundMaxValueEntropy
  (BoTorch's max values and samplers), ``_sample_max_value_Gumbel`` and
  ``_sample_max_value_Thompson``;
* qJointEntropySearch (``"LB"`` for ``q = 1, 2``, ``"MC"`` for ``q = 1``),
  qPredictiveEntropySearch and ``get_optimal_samples`` (BoTorch's path
  draws and Sobol points);
* PathwiseThompsonSampling (BoTorch's path draws) and
  qNegIntegratedPosteriorVariance;
* the penalties, ``PenalizedAcquisitionFunction`` and
  ``PenalizedMCObjective``, ``ProximalAcquisitionFunction``,
  ``FixedFeatureAcquisitionFunction`` and
  ``PriorGuidedAcquisitionFunction``;
* errors.

Tolerances: values 1e-9 and gradients 1e-8 (relative, mixed elementwise /
normwise). ``optimize_acqf`` from identical initial conditions: the same
local optimum (values 1e-6, candidates 1e-4; L-BFGS-B stops on
rounding-dependent criteria).

BoTorch bugs (see ``docs/drafts/8-so-acquisition.md``), with independent
references instead of BoTorch's result:

* qKG / qMFKG with an MC objective or an MC value function: BoTorch's
  gradient misses the dependence of the fantasy posteriors' covariances on
  the candidates (``fast_pred_var``; meta-pytorch/botorch#3427). Gradients
  are compared with a BoTorch that has the fix (if it is the one imported)
  and with central differences of botorch_rs's values (which match
  BoTorch's).
* qNegIntegratedPosteriorVariance: the same ``fast_pred_var`` problem;
  gradients are compared with BoTorch without ``fast_pred_var``.
* GIBBON with pending points and an outcome transform mixes the
  outcome-transformed cross-covariance with untransformed covariances:
  compared without an outcome transform.
* qJointEntropySearch "MC" for ``q > 1`` (and with pending points;
  meta-pytorch/botorch#3428) and the input transform of the optima
  (meta-pytorch/botorch#3399): compared where BoTorch is correct.
  The conditional-CDF denominator is explicitly corrected to a standard
  deviation; unmodified b0dd01be divides by variance even for ``q = 1``.
* PathwiseThompsonSampling with a posterior transform and ``q = 1`` sums
  over the t-batch: compared for ``q = 2``.
* qAnalyticProbabilityOfImprovement: BoTorch's MVNXPB gradients of a
  t-batch whose members pivot differently are wrong: compared one candidate
  set at a time.

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

import botorch.acquisition.utils as bt_autils  # noqa: E402
from botorch.acquisition import active_learning as bt_al  # noqa: E402
from botorch.acquisition import analytic as bt_an  # noqa: E402
from botorch.acquisition import cost_aware as bt_cost  # noqa: E402
from botorch.acquisition import fixed_feature as bt_ff  # noqa: E402
from botorch.acquisition import joint_entropy_search as bt_jes  # noqa: E402
from botorch.acquisition import knowledge_gradient as bt_kg  # noqa: E402
from botorch.acquisition import logei as bt_logei  # noqa: E402
from botorch.acquisition import max_value_entropy_search as bt_mes  # noqa: E402
from botorch.acquisition import monte_carlo as bt_mc  # noqa: E402
from botorch.acquisition import penalized as bt_pen  # noqa: E402
from botorch.acquisition import predictive_entropy_search as bt_pes  # noqa: E402
from botorch.acquisition import prior_guided as bt_pg  # noqa: E402
from botorch.acquisition import proximal as bt_prox  # noqa: E402
from botorch.acquisition import thompson_sampling as bt_ts  # noqa: E402
from botorch.acquisition.objective import GenericMCObjective as BtGeneric  # noqa: E402
from botorch.acquisition.objective import (  # noqa: E402
    ScalarizedPosteriorTransform as BtSPT,
)
from botorch.models.cost import AffineFidelityCostModel as BtAffineCost  # noqa: E402
from botorch.optim import optimize as bt_optimize  # noqa: E402
from botorch.sampling.normal import SobolQMCNormalSampler as BtSampler  # noqa: E402
from hvkg_helpers import use_fantasy_mean_cache  # noqa: E402
from mo_entropy_helpers import (  # noqa: E402
    bt_path_draws,
    raw_train_inputs,
    recorded_randn,
)
from so_acquisition_helpers import (  # noqa: E402
    NEI_SAMPLES,
    Prior,
    botorch_has_kg_fix,
    correct_jes_monte_carlo,
    f32,
    finite_differences,
    gumbel_reference,
    jes_has_3399,
    jes_optimal_inputs,
    mes_pair,
    no_fast_pred_var,
    qpi_nan_gradients,
    recorded_draws,
    recorded_optimal_samples,
    rs_sampler,
    sampler_pair,
    seeded_draws,
)

from botorch_rs.acquisition import active_learning as rs_al  # noqa: E402
from botorch_rs.acquisition import analytic as rs_an  # noqa: E402
from botorch_rs.acquisition import cost_aware as rs_cost  # noqa: E402
from botorch_rs.acquisition import fixed_feature as rs_ff  # noqa: E402
from botorch_rs.acquisition import joint_entropy_search as rs_jes  # noqa: E402
from botorch_rs.acquisition import knowledge_gradient as rs_kg  # noqa: E402
from botorch_rs.acquisition import logei as rs_logei  # noqa: E402
from botorch_rs.acquisition import max_value_entropy_search as rs_mes  # noqa: E402
from botorch_rs.acquisition import monte_carlo as rs_mc  # noqa: E402
from botorch_rs.acquisition import penalized as rs_pen  # noqa: E402
from botorch_rs.acquisition import predictive_entropy_search as rs_pes  # noqa: E402
from botorch_rs.acquisition import prior_guided as rs_pg  # noqa: E402
from botorch_rs.acquisition import proximal as rs_prox  # noqa: E402
from botorch_rs.acquisition import thompson_sampling as rs_ts  # noqa: E402
from botorch_rs.acquisition import utils as rs_autils  # noqa: E402
from botorch_rs.acquisition.objective import GenericMCObjective as RsGeneric  # noqa: E402
from botorch_rs.acquisition.objective import (  # noqa: E402
    ScalarizedPosteriorTransform as RsSPT,
)
from botorch_rs.models.cost import AffineFidelityCostModel as RsAffineCost  # noqa: E402
from botorch_rs.optim import optimize as rs_optimize  # noqa: E402
from botorch_rs.sampling import SobolQMCNormalSampler as RsSampler  # noqa: E402
from parity_helpers import (  # noqa: E402
    Checker,
    ModelSpec,
    list_pair,
    model_pair,
    patched_lower_bound,
    points_in,
    values_and_grads,
)

pytestmark = pytest.mark.filterwarnings("ignore")

DT = torch.float64
VALUE_TOL = 1e-9
GRAD_TOL = 1e-8
#: Central differences (step 1e-6) of botorch_rs's values against its
#: gradients: below 2e-9 relative in the smooth cases measured, up to 6.4e-8
#: where rounding of the values dominates the differences
#: (qAnalyticProbabilityOfImprovement on nearly coincident points; step 1e-5
#: agrees to 3e-9 there).
FD_TOL = 1e-7
CHECK = Checker("single-objective acquisition (phase 8)")


@pytest.fixture(autouse=True, scope="module")
def _single_threaded_torch() -> Iterator[None]:
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(threads)


# ---------------------------------------------------------------------------
# Models and helpers
# ---------------------------------------------------------------------------


@functools.cache
def _case(name: str) -> tuple[Any, Any, Any]:
    if name == "m1_rbf":
        return model_pair(ModelSpec(n=8, d=2), 3)
    if name == "m1_norm":
        return model_pair(ModelSpec(n=9, d=3, normalize=True, kernel="matern52"), 5)
    if name == "m1_fixed":
        return model_pair(ModelSpec(n=7, d=2, fixed_noise=True), 7)
    if name == "m1_nostd":
        return model_pair(ModelSpec(n=8, d=2, standardize=False), 9)
    if name == "m2":
        return model_pair(ModelSpec(n=8, d=2, m=2), 11)
    if name == "list":
        return list_pair([ModelSpec(n=8, d=2), ModelSpec(n=9, d=2, normalize=True)], 13)
    raise KeyError(name)


SINGLE = ["m1_rbf", "m1_norm", "m1_fixed"]
MULTI = ["m2", "list"]


def _transforms(m: int) -> tuple[Any, Any]:
    if m == 1:
        return None, None
    w = torch.tensor([0.75, -0.25], dtype=DT)
    return BtSPT(w), RsSPT(w)


def _compare(name: str, a_rs: Any, a_bt: Any, X: Any, grads: bool = True) -> None:
    vr, gr = values_and_grads(a_rs, X)
    vb, gb = values_and_grads(a_bt, X)
    assert tuple(vr.shape) == tuple(vb.shape), (name, vr.shape, vb.shape)
    CHECK.check(f"{name}: values", vr, vb, VALUE_TOL)
    if grads:
        CHECK.check(f"{name}: gradients", gr, gb, GRAD_TOL)


def _check_fd(name: str, a_rs: Any, X: Any) -> None:
    _, gr = values_and_grads(a_rs, X)
    CHECK.check(f"{name}: gradients vs finite differences", gr, finite_differences(a_rs, X), FD_TOL)


# ---------------------------------------------------------------------------
# Knowledge gradient
# ---------------------------------------------------------------------------


def _kg_pair(
    bt: Any,
    rs: Any,
    q: int,
    n_f: int,
    m_fant: int,
    pending: Any = None,
    inner: int = 0,
    cls: str = "qKnowledgeGradient",
    **kw: Any,
) -> tuple[Any, Any]:
    """qKG (or qMFKG) in both libraries with the same fantasy (and inner)
    base samples; ``m_fant``: outputs of the fantasized posterior, ``inner``:
    outputs of the inner posterior (0: no inner sampler)."""
    k = q + (0 if pending is None else pending.shape[0])
    s_bt, s_rs = sampler_pair(n_f, (1, k, m_fant) if m_fant > 1 else (1, k), 1)
    kw_bt, kw_rs = dict(kw), dict(kw)
    for key in ("objective", "posterior_transform", "cost_aware_utility", "project", "expand"):
        if key in kw:
            kw_bt[key], kw_rs[key] = kw[key]
    if inner:
        i_bt, i_rs = sampler_pair(16, (1, 1, 1, inner) if inner > 1 else (1, 1, 1), 2)
        kw_bt["inner_sampler"], kw_rs["inner_sampler"] = i_bt, i_rs
    a_bt = getattr(bt_kg, cls)(bt, num_fantasies=n_f, sampler=s_bt, X_pending=pending, **kw_bt)
    a_rs = getattr(rs_kg, cls)(rs, num_fantasies=n_f, sampler=s_rs, X_pending=pending, **kw_rs)
    return a_bt, a_rs


@pytest.mark.parametrize("variant", ["pm", "current_value", "pending", "mc_objective"])
@pytest.mark.parametrize("case", SINGLE)
def test_knowledge_gradient_matches_botorch(case: str, variant: str) -> None:
    bt, rs, X = _case(case)
    d = X.shape[-1]
    q, n_f = 2, 4
    Xt = points_in(X, (2, q + n_f, d), 21)
    kw: dict[str, Any] = {}
    pending, inner = None, 0
    if variant == "current_value":
        kw["current_value"] = torch.tensor(f32(0.25), dtype=DT)
    elif variant == "pending":
        pending = points_in(X, (2, d), 22)
    elif variant == "mc_objective":
        kw["objective"] = (
            BtGeneric(lambda Y, X=None: Y[..., 0] ** 2),
            RsGeneric(lambda Y, X=None: Y[..., 0] ** 2),
        )
        inner = 1
    a_bt, a_rs = _kg_pair(bt, rs, q, n_f, 1, pending, inner, **kw)
    exact = variant != "mc_objective" or botorch_has_kg_fix()
    _compare(f"qKG {variant}", a_rs, a_bt, Xt, grads=exact)
    if not exact:
        _check_fd(f"qKG {variant}", a_rs, Xt[:1])


@pytest.mark.parametrize("case", MULTI)
def test_knowledge_gradient_multi_output_matches_botorch(case: str) -> None:
    bt, rs, X = _case(case)
    d = X.shape[-1]
    q, n_f = 1, 3
    # A batch of two t-batches: BoTorch fails for a single t-batch of a
    # multi-output model (``[n_f, 1]`` fantasy batch; see the draft).
    Xt = points_in(X, (2, q + n_f, d), 23)
    tf = _transforms(2)
    # BoTorch fantasizes every sub-model of a ModelListGP with the same
    # sampler (single-output base samples).
    m_fant = 1 if case == "list" else 2
    a_bt, a_rs = _kg_pair(bt, rs, q, n_f, m_fant, posterior_transform=tf)
    _compare(f"qKG transform {case}", a_rs, a_bt, Xt)
    obj = (
        BtGeneric(lambda Y, X=None: Y[..., 0] - 0.5 * Y[..., 1]),
        RsGeneric(lambda Y, X=None: Y[..., 0] - 0.5 * Y[..., 1]),
    )
    a_bt, a_rs = _kg_pair(bt, rs, q, n_f, m_fant, inner=2, objective=obj)
    exact = botorch_has_kg_fix()
    _compare(f"qKG objective {case}", a_rs, a_bt, Xt, grads=exact)
    if not exact:
        _check_fd(f"qKG objective {case}", a_rs, Xt[:1])


def _mf_variants(d: int) -> dict[str, tuple[dict[str, Any], dict[str, Any], int, bool]]:
    """``variant -> (BoTorch kwargs, botorch_rs kwargs, fantasized points per
    candidate, inner sampler)`` of qMFKG (the last column is the fidelity)."""
    from botorch.acquisition.utils import expand_trace_observations as bt_expand
    from botorch.acquisition.utils import project_to_target_fidelity as bt_project

    from botorch_rs.acquisition.utils import expand_trace_observations as rs_expand
    from botorch_rs.acquisition.utils import project_to_target_fidelity as rs_project

    tf = {d - 1: 1.0}
    pb = functools.partial(bt_project, target_fidelities=tf, d=d)
    pr = functools.partial(rs_project, target_fidelities=tf, d=d)
    best = {"best_f": 1.625}
    return {
        "project": ({"project": pb}, {"project": pr}, 1, False),
        "project_lambda": (
            {"project": lambda X: bt_project(X, target_fidelities=tf, d=d)},
            {"project": lambda X: rs_project(X, target_fidelities=tf, d=d)},
            1,
            False,
        ),
        "expand": (
            {"expand": functools.partial(bt_expand, fidelity_dims=[d - 1], num_trace_obs=2)},
            {"expand": functools.partial(rs_expand, fidelity_dims=[d - 1], num_trace_obs=2)},
            3,
            False,
        ),
        "cost": (
            {
                "project": pb,
                "cost_aware_utility": bt_cost.InverseCostWeightedUtility(
                    BtAffineCost(fidelity_weights={d - 1: 0.75}, fixed_cost=0.25)
                ),
                "current_value": torch.tensor(0.25, dtype=DT),
            },
            {
                "project": pr,
                "cost_aware_utility": rs_cost.InverseCostWeightedUtility(
                    RsAffineCost(fidelity_weights={d - 1: 0.75}, fixed_cost=0.25)
                ),
                "current_value": torch.tensor(0.25, dtype=DT),
            },
            1,
            False,
        ),
        "valfunc_ei": (
            {"valfunc_cls": bt_an.ExpectedImprovement, "valfunc_argfac": lambda model: best},
            {"valfunc_cls": rs_an.ExpectedImprovement, "valfunc_argfac": lambda model: best},
            1,
            False,
        ),
        "valfunc_ucb_min": (
            {
                "valfunc_cls": bt_an.UpperConfidenceBound,
                "valfunc_argfac": lambda model: {"beta": 0.25, "maximize": False},
            },
            {
                "valfunc_cls": rs_an.UpperConfidenceBound,
                "valfunc_argfac": lambda model: {"beta": 0.25, "maximize": False},
            },
            1,
            False,
        ),
        "valfunc_qei": (
            {"valfunc_cls": bt_mc.qExpectedImprovement, "valfunc_argfac": lambda model: best},
            {"valfunc_cls": rs_mc.qExpectedImprovement, "valfunc_argfac": lambda model: best},
            1,
            True,
        ),
    }


@pytest.mark.parametrize(
    "variant",
    ["project", "project_lambda", "expand", "cost", "valfunc_ei", "valfunc_ucb_min", "valfunc_qei"],
)
def test_multi_fidelity_knowledge_gradient_matches_botorch(variant: str) -> None:
    bt, rs, X = _case("m1_norm")
    d = X.shape[-1]
    q, n_f = 2, 3
    Xt = points_in(X, (2, q + n_f, d), 25)
    kb, kr, k, inner = _mf_variants(d)[variant]
    kb, kr = dict(kb), dict(kr)
    s_bt, s_rs = sampler_pair(n_f, (1, q * k), 1)
    if inner:
        kb["inner_sampler"], kr["inner_sampler"] = sampler_pair(16, (1, 1, 1), 2)
    a_bt = bt_kg.qMultiFidelityKnowledgeGradient(bt, num_fantasies=n_f, sampler=s_bt, **kb)
    a_rs = rs_kg.qMultiFidelityKnowledgeGradient(rs, num_fantasies=n_f, sampler=s_rs, **kr)
    # Value functions other than the posterior mean: BoTorch's gradient
    # misses the fantasy covariances (#3427) unless it has the fix.
    exact = not variant.startswith("valfunc") or botorch_has_kg_fix()
    _compare(f"qMFKG {variant}", a_rs, a_bt, Xt, grads=exact)
    if not exact:
        _check_fd(f"qMFKG {variant}", a_rs, Xt[:1])


def test_projected_acquisition_function_matches_botorch() -> None:
    bt, rs, X = _case("m1_norm")
    d = X.shape[-1]
    kb, kr, _, _ = _mf_variants(d)["project"]
    b = bt_kg.ProjectedAcquisitionFunction(bt_an.PosteriorMean(bt), kb["project"])
    r = rs_kg.ProjectedAcquisitionFunction(rs_an.PosteriorMean(rs), kr["project"])
    _compare("ProjectedAcquisitionFunction", r, b, points_in(X, (3, 1, d), 26))


def test_optimize_acqf_with_knowledge_gradient_matches_botorch() -> None:
    bt, rs, X = _case("m1_rbf")
    d = X.shape[-1]
    q, n_f = 1, 4
    a_bt, a_rs = _kg_pair(bt, rs, q, n_f, 1)
    bounds = torch.stack([X.min(0).values, X.max(0).values])
    ics = points_in(X, (3, q + n_f, d), 27).clamp(bounds[0], bounds[1])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cb, vb = bt_optimize.optimize_acqf(
            a_bt, bounds, q=q, num_restarts=3, batch_initial_conditions=ics
        )
    cr, vr = rs_optimize.optimize_acqf(
        a_rs, bounds, q=q, num_restarts=3, batch_initial_conditions=ics
    )
    CHECK.check("optimize_acqf qKG value", vr, vb.detach(), 1e-6)
    CHECK.check("optimize_acqf qKG candidate", cr, cb, 1e-4)


# ---------------------------------------------------------------------------
# qNEI, qLCB, qPSTD
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("cache_root", [False, True])
@pytest.mark.parametrize("case", ["m1_norm", "m1_fixed", "m2", "list"])
def test_qnei_matches_botorch(case: str, cache_root: bool) -> None:
    bt, rs, X = _case(case)
    d = X.shape[-1]
    m = 2 if case in MULTI else 1
    N = X.shape[0]
    kw_bt: dict[str, Any] = {}
    kw_rs: dict[str, Any] = {}
    # Points of the objective's lower bound when a sample has no feasible
    # baseline point (BoTorch draws them at random).
    lb = points_in(X, (32, d), 30)
    if m > 1:
        kw_bt["objective"] = BtGeneric(lambda Y, X=None: Y[..., 0] - 0.5 * Y[..., 1])
        kw_rs["objective"] = RsGeneric(lambda Y, X=None: Y[..., 0] - 0.5 * Y[..., 1])
        kw_bt["constraints"] = [lambda Z: Z[..., 1] - 1.5]
        kw_rs["constraints"] = [lambda Z: Z[..., 1] - 1.5]
        # A temperature that BoTorch's float32 buffer holds exactly.
        kw_bt["eta"] = kw_rs["eta"] = 2.0**-10
        kw_rs["lower_bound_points"] = lb
    for q in (1, 3):
        for n_pend in (0, 2):
            pending = points_in(X, (n_pend, d), 31) if n_pend else None
            k = q + n_pend
            with seeded_draws(), patched_lower_bound(lb):
                s_bt = BtSampler(sample_shape=torch.Size([NEI_SAMPLES]), seed=0)
                a_bt = bt_mc.qNoisyExpectedImprovement(
                    bt,
                    X,
                    sampler=s_bt,
                    X_pending=pending,
                    prune_baseline=False,
                    cache_root=cache_root,
                    **kw_bt,
                )
                s_rs = rs_sampler(NEI_SAMPLES, [N * m, (N + k) * m])
                a_rs = rs_mc.qNoisyExpectedImprovement(
                    rs,
                    X,
                    sampler=s_rs,
                    X_pending=pending,
                    prune_baseline=False,
                    cache_root=cache_root,
                    **kw_rs,
                )
                Xt = points_in(X, (2, q, d), 32 + q)
                _compare(f"qNEI cache_root={cache_root}", a_rs, a_bt, Xt)


def test_qnei_pruning_keeps_botorch_points() -> None:
    bt, rs, X = _case("m1_rbf")
    torch.manual_seed(0)
    a_bt = bt_mc.qNoisyExpectedImprovement(bt, X, prune_baseline=True)
    a_rs = rs_mc.qNoisyExpectedImprovement(rs, X, prune_baseline=True)
    kept_bt = {tuple(x) for x in a_bt.X_baseline.tolist()}
    kept_rs = {tuple(x) for x in a_rs.X_baseline.tolist()}
    # Different random samples: the points with a clear chance of being the
    # best are kept by both.
    assert kept_rs <= {tuple(x) for x in X.tolist()}
    CHECK.check(
        "qNEI pruned baseline (symmetric difference)", float(len(kept_bt ^ kept_rs)), 0.0, 2.0
    )


@pytest.mark.parametrize("case", SINGLE + MULTI)
def test_qlcb_and_qpstd_match_botorch(case: str) -> None:
    bt, rs, X = _case(case)
    d = X.shape[-1]
    m = 2 if case in MULTI else 1
    tf = _transforms(m)
    for q in (1, 3):
        Xt = points_in(X, (2, q, d), 40 + q)
        # The scalarized posterior has one output.
        shape = (1, q)
        for name in ("qLowerConfidenceBound", "qPosteriorStandardDeviation"):
            s_bt, s_rs = sampler_pair(32, shape, 41)
            kw_bt = {"posterior_transform": tf[0]} if m > 1 else {}
            kw_rs = {"posterior_transform": tf[1]} if m > 1 else {}
            if name == "qLowerConfidenceBound":
                kw_bt["beta"] = kw_rs["beta"] = 0.25
            a_bt = getattr(bt_mc, name)(bt, sampler=s_bt, **kw_bt)
            a_rs = getattr(rs_mc, name)(rs, sampler=s_rs, **kw_rs)
            _compare(name, a_rs, a_bt, Xt)


# ---------------------------------------------------------------------------
# Analytic acquisition functions
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("log", [False, True])
@pytest.mark.parametrize("maximize", [True, False])
def test_noisy_expected_improvement_matches_botorch(log: bool, maximize: bool) -> None:
    bt, rs, X = _case("m1_fixed")
    d = X.shape[-1]
    n_f = 8
    draws: list[Any] = []
    cls = "LogNoisyExpectedImprovement" if log else "NoisyExpectedImprovement"
    with recorded_draws(draws):
        a_bt = getattr(bt_an, cls)(bt, X, num_fantasies=n_f, maximize=maximize)
    sampler = RsSampler(sample_shape=torch.Size([n_f]), seed=0)
    sampler.base_samples = draws[0].reshape(n_f, -1).clone()
    a_rs = getattr(rs_an, cls)(rs, X, num_fantasies=n_f, maximize=maximize, sampler=sampler)
    CHECK.check(f"{cls}: best_f", a_rs.best_f, a_bt.best_f.reshape(-1), VALUE_TOL)
    _compare(cls, a_rs, a_bt, points_in(X, (4, 1, d), 51))


def test_log_probability_of_feasibility_matches_botorch() -> None:
    for case in ("m1_rbf", "m2", "list"):
        bt, rs, X = _case(case)
        d = X.shape[-1]
        cons = {0: (None, 1.625)} if case == "m1_rbf" else {0: (0.5, None), 1: (-1.0, 2.5)}
        a_bt = bt_an.LogProbabilityOfFeasibility(bt, constraints=cons)
        a_rs = rs_an.LogProbabilityOfFeasibility(rs, constraints=cons)
        _compare(f"LogProbabilityOfFeasibility {case}", a_rs, a_bt, points_in(X, (4, 1, d), 52))


@pytest.mark.parametrize("case", ["m1_rbf", "m1_norm", "m2", "list"])
def test_q_analytic_probability_of_improvement_matches_botorch(case: str) -> None:
    bt, rs, X = _case(case)
    d = X.shape[-1]
    m = 2 if case in MULTI else 1
    tf = _transforms(m)
    best = 1.625 if m == 1 else 0.5
    for q in (1, 2, 3, 5):
        for maximize in (True, False):
            a_bt = bt_an.qAnalyticProbabilityOfImprovement(
                bt, best, posterior_transform=tf[0], maximize=maximize
            )
            a_rs = rs_an.qAnalyticProbabilityOfImprovement(
                rs, best, posterior_transform=tf[1], maximize=maximize
            )
            Xt = points_in(X, (3, q, d), 54 + q)
            vr, gr = values_and_grads(a_rs, Xt)
            vb, _ = values_and_grads(a_bt, Xt)
            CHECK.check("qAnalyticProbabilityOfImprovement: values", vr, vb, VALUE_TOL)
            # BoTorch's gradients of a t-batch whose members pivot differently
            # are wrong (see the draft): one candidate set at a time.
            gb = torch.cat([values_and_grads(a_bt, Xt[k : k + 1])[1] for k in range(3)])
            gb = qpi_nan_gradients(CHECK, a_rs, Xt, gr, gb)
            # Near value 1 the probability of no improvement (~1e-13) has lost
            # most of its digits to cancellation in both libraries (both are
            # 50% off their own finite differences there; the gradients are
            # ~1e-12): entries within 1e-14 of each other count as equal.
            CHECK.check(
                "qAnalyticProbabilityOfImprovement: gradients", gr, gb, GRAD_TOL, atol=1e-14
            )


def test_scalarized_posterior_mean_matches_botorch() -> None:
    bt, rs, X = _case("m1_norm")
    d = X.shape[-1]
    w = torch.tensor([0.75, -0.25, 0.5], dtype=DT)
    a_bt = bt_an.ScalarizedPosteriorMean(bt, weights=w)
    a_rs = rs_an.ScalarizedPosteriorMean(rs, weights=w)
    _compare("ScalarizedPosteriorMean", a_rs, a_bt, points_in(X, (3, 3, d), 53))


# ---------------------------------------------------------------------------
# Max-value entropy search
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("variant", ["plain", "minimize", "transform", "pending"])
@pytest.mark.parametrize("cls", ["qMaxValueEntropy", "qLowerBoundMaxValueEntropy"])
@pytest.mark.parametrize("case", ["m1_rbf", "m1_norm", "m1_fixed", "m1_nostd"])
def test_max_value_entropy_matches_botorch(case: str, cls: str, variant: str) -> None:
    bt, rs, X = _case(case)
    d = X.shape[-1]
    if cls == "qLowerBoundMaxValueEntropy" and variant == "pending" and case != "m1_nostd":
        # BoTorch's GIBBON repulsion mixes outcome-transformed and original
        # covariances (see the draft); compared without an outcome transform.
        pytest.skip("BoTorch's GIBBON repulsion is wrong with an outcome transform")
    cs = points_in(X, (20, d), 60)
    kb: dict[str, Any] = {}
    kr: dict[str, Any] = {}
    pending = None
    if variant == "minimize":
        kb["maximize"] = kr["maximize"] = False
    elif variant == "transform":
        kb["posterior_transform"] = BtSPT(torch.tensor([-0.75], dtype=DT), 0.25)
        kr["posterior_transform"] = RsSPT(torch.tensor([-0.75], dtype=DT), 0.25)
    elif variant == "pending":
        pending = points_in(X, (2, d), 61)
    a_bt, a_rs = mes_pair(cls, bt, rs, cs, pending, kb, kr)
    _compare(f"{cls} {variant}", a_rs, a_bt, points_in(X, (5, 1, d), 62))
    CHECK.check(
        f"{cls}: posterior_max_values",
        a_rs.posterior_max_values,
        a_bt.posterior_max_values.detach(),
        0.0,
    )


@pytest.mark.parametrize("variant", ["project", "trace", "pending", "gibbon", "gibbon_pending"])
def test_multi_fidelity_max_value_entropy_matches_botorch(variant: str) -> None:
    case = "m1_nostd" if variant == "gibbon_pending" else "m1_norm"
    bt, rs, X = _case(case)
    d = X.shape[-1]
    cs = points_in(X, (20, d), 63)
    from botorch.acquisition.utils import expand_trace_observations as bt_expand
    from botorch.acquisition.utils import project_to_target_fidelity as bt_project

    from botorch_rs.acquisition.utils import expand_trace_observations as rs_expand
    from botorch_rs.acquisition.utils import project_to_target_fidelity as rs_project

    tf = {d - 1: 1.0}
    kb: dict[str, Any] = {"project": functools.partial(bt_project, target_fidelities=tf, d=d)}
    kr: dict[str, Any] = {"project": functools.partial(rs_project, target_fidelities=tf, d=d)}
    pending = points_in(X, (2, d), 64) if variant.endswith("pending") else None
    cls = "qMultiFidelityMaxValueEntropy"
    if variant == "trace":
        kb["expand"] = functools.partial(bt_expand, fidelity_dims=[d - 1], num_trace_obs=2)
        kr["expand"] = functools.partial(rs_expand, fidelity_dims=[d - 1], num_trace_obs=2)
    if variant.startswith("gibbon"):
        cls = "qMultiFidelityLowerBoundMaxValueEntropy"
    if variant == "gibbon_pending":
        # BoTorch's constructor drops ``X_pending`` (see the draft): it is set
        # afterwards in both.
        a_bt, a_rs = mes_pair(cls, bt, rs, cs, None, kb, kr)
        a_bt.set_X_pending(pending)
        a_rs.set_X_pending(pending)
    else:
        a_bt, a_rs = mes_pair(cls, bt, rs, cs, pending, kb, kr)
    # Points in the bounding box of the training inputs (positive costs of
    # the default AffineFidelityCostModel).
    _compare(f"{cls} {variant}", a_rs, a_bt, points_in(X, (5, 1, d), 65, pad=0.0))


def test_max_value_samplers_match_botorch() -> None:
    bt, rs, X = _case("m1_norm")
    d = X.shape[-1]
    cs = points_in(X, (30, d), 66)
    for maximize in (True, False):
        torch.manual_seed(3)
        expected = bt_mes._sample_max_value_Gumbel(bt, cs, 6, maximize=maximize)
        torch.manual_seed(3)
        u = torch.rand(6, 1, dtype=cs.dtype)
        actual = rs_mes._sample_max_value_Gumbel(rs, cs, 6, maximize=maximize, uniforms=u)
        # BoTorch rounds the quantiles to float32 (see the draft): compared
        # with BoTorch's algorithm in float64, and with BoTorch within that
        # rounding.
        CHECK.check(
            "_sample_max_value_Gumbel", actual, gumbel_reference(bt, cs, u, maximize), VALUE_TOL
        )
        CHECK.check(
            "_sample_max_value_Gumbel vs BoTorch (float32 quantiles)", actual, expected, 1e-6
        )
        torch.manual_seed(4)
        expected = bt_mes._sample_max_value_Thompson(bt, cs, 6, maximize=maximize)
        torch.manual_seed(4)
        # GPyTorch draws the ``n x num_samples`` normal numbers of its
        # Cholesky root.
        z = torch.randn(cs.shape[0], 6, dtype=cs.dtype).T
        actual = rs_mes._sample_max_value_Thompson(rs, cs, 6, maximize=maximize, base_samples=z)
        CHECK.check("_sample_max_value_Thompson", actual, expected, VALUE_TOL)


# ---------------------------------------------------------------------------
# Joint and predictive entropy search, optimal samples
# ---------------------------------------------------------------------------


def _bounds(X: Any) -> Any:
    return torch.stack([X.min(0).values, X.max(0).values])


def _optimal_samples(case: str, num: int = 4, transform: bool = False) -> tuple[Any, ...]:
    """BoTorch's optimal samples and botorch_rs's with the same random
    numbers."""
    bt, rs, X = _case(case)
    bounds = _bounds(X)
    m = 2 if case in MULTI else 1
    tf = _transforms(m) if transform else (None, None)
    draws: list[Any] = []
    cands: list[Any] = []
    torch.manual_seed(5)
    with recorded_optimal_samples(draws, cands):
        xb, fb = bt_autils.get_optimal_samples(
            bt, bounds, num, raw_samples=64, num_restarts=4, posterior_transform=tf[0]
        )
    C = bounds[0] + (bounds[1] - bounds[0]) * cands[0]
    xr, fr = rs_autils.get_optimal_samples(
        rs,
        bounds,
        num,
        raw_samples=64,
        num_restarts=4,
        posterior_transform=tf[1],
        num_features=draws[0][0].shape[0] * 2,
        draws=draws,
        candidates=C,
    )
    return xb, fb, xr, fr


@pytest.mark.parametrize("case", SINGLE + MULTI)
def test_get_optimal_samples_matches_botorch(case: str) -> None:
    xb, fb, xr, fr = _optimal_samples(case, transform=case in MULTI)
    # The same local optima (L-BFGS-B stops on rounding-dependent criteria).
    CHECK.check("get_optimal_samples: inputs", xr, xb, 1e-6)
    CHECK.check("get_optimal_samples: outputs", fr, fb, 1e-6)


@pytest.mark.parametrize("est", ["LB", "MC"])
@pytest.mark.parametrize("case", ["m1_rbf", "m1_norm", "m2", "list"])
def test_joint_entropy_search_matches_botorch(case: str, est: str) -> None:
    bt, rs, X = _case(case)
    d = X.shape[-1]
    m = 2 if case in MULTI else 1
    if est == "MC" and m > 1 and not jes_has_3399():
        pytest.skip("BoTorch's MC estimate ignores the posterior transform of the optima (#3399)")
    tf = _transforms(m)
    xo, fo = rs_autils.get_optimal_samples(
        rs,
        _bounds(X),
        4,
        raw_samples=64,
        num_restarts=4,
        posterior_transform=tf[1],
        num_features=64,
        seed=3,
    )
    settings = [(1, None), (2, None), (1, points_in(X, (1, d), 70))]
    for q, pending in settings:
        if est == "MC" and q + (0 if pending is None else 1) > 1:
            continue  # BoTorch's MC estimate is wrong for q > 1 (#3428)
        torch.manual_seed(0)
        a_bt = bt_jes.qJointEntropySearch(
            bt,
            jes_optimal_inputs(bt, xo),
            fo,
            posterior_transform=tf[0],
            X_pending=pending,
            estimation_type=est,
            num_samples=16,
        )
        use_fantasy_mean_cache(a_bt.conditional_model)  # meta-pytorch/botorch#3396
        if est == "MC":
            correct_jes_monte_carlo(a_bt)  # conditional Gaussian standard deviation
        s = RsSampler(sample_shape=torch.Size([16]), seed=0)
        if est == "MC":
            a_bt(points_in(X, (1, q, d), 71))
            s.base_samples = a_bt.sampler.base_samples.reshape(16, -1).clone()
        a_rs = rs_jes.qJointEntropySearch(
            rs,
            xo,
            fo,
            posterior_transform=tf[1],
            X_pending=pending,
            estimation_type=est,
            sampler=s,
        )
        _compare(f"qJointEntropySearch {est}", a_rs, a_bt, points_in(X, (3, q, d), 72 + q))


@pytest.mark.parametrize("case", ["m1_rbf", "m1_norm"])
def test_predictive_entropy_search_matches_botorch(case: str) -> None:
    bt, rs, X = _case(case)
    d = X.shape[-1]
    for maximize in (True, False):
        # Maximizers for maximize=True, minimizers otherwise (with the wrong
        # optima EP does not converge and its result depends on rounding, in
        # BoTorch as well).
        tf = None if maximize else RsSPT(torch.tensor([-1.0], dtype=DT))
        xo, _ = rs_autils.get_optimal_samples(
            rs,
            _bounds(X),
            3,
            raw_samples=64,
            num_restarts=4,
            posterior_transform=tf,
            num_features=64,
            seed=4,
        )
        # BoTorch reads the training inputs in eval mode (transformed);
        # botorch_rs uses the raw ones (#3421).
        with raw_train_inputs(bt):
            a_bt = bt_pes.qPredictiveEntropySearch(bt, xo, maximize=maximize)
        a_rs = rs_pes.qPredictiveEntropySearch(rs, xo, maximize=maximize)
        assert 1 <= a_rs.ep_iterations < a_rs.max_ep_iterations
        _compare("qPredictiveEntropySearch", a_rs, a_bt, points_in(X, (3, 2, d), 75))


# ---------------------------------------------------------------------------
# Thompson sampling and active learning
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("case", SINGLE + MULTI)
def test_pathwise_thompson_sampling_matches_botorch(case: str) -> None:
    bt, rs, X = _case(case)
    d = X.shape[-1]
    m = 2 if case in MULTI else 1
    tf = _transforms(m)
    # With a posterior transform and q = 1, BoTorch sums the values of the
    # t-batch (see the draft).
    for q in (1, 2) if m == 1 else (2,):
        torch.manual_seed(2)
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
        _compare(f"PathwiseThompsonSampling q={q}", a_rs, a_bt, points_in(X, (3, q, d), 80 + q))


@pytest.mark.parametrize("case", SINGLE + MULTI)
def test_neg_integrated_posterior_variance_matches_botorch(case: str) -> None:
    bt, rs, X = _case(case)
    d = X.shape[-1]
    m = 2 if case in MULTI else 1
    tf = _transforms(m)
    mc = points_in(X, (12, d), 85)
    for q in (1, 2):
        for pending in (None, points_in(X, (1, d), 86)):
            a_bt = bt_al.qNegIntegratedPosteriorVariance(
                bt, mc, posterior_transform=tf[0], X_pending=pending
            )
            a_rs = rs_al.qNegIntegratedPosteriorVariance(
                rs, mc, posterior_transform=tf[1], X_pending=pending
            )
            Xt = points_in(X, (3, q, d), 87 + q)
            # BoTorch's gradients miss the dependence of the fantasy model's
            # covariances on X (``fast_pred_var``, see the draft).
            with no_fast_pred_var():
                _compare("qNegIntegratedPosteriorVariance", a_rs, a_bt, Xt)
            _check_fd("qNegIntegratedPosteriorVariance", a_rs, Xt[:1])


# ---------------------------------------------------------------------------
# Penalties and wrappers
# ---------------------------------------------------------------------------

INIT = [0.2, 0.5, 0.7]


def _penalties(init: Any) -> dict[str, tuple[Any, Any]]:
    return {
        "L2": (bt_pen.L2Penalty(init), rs_pen.L2Penalty(init)),
        "L1": (bt_pen.L1Penalty(init), rs_pen.L1Penalty(init)),
        "Gaussian": (bt_pen.GaussianPenalty(init, 0.5), rs_pen.GaussianPenalty(init, 0.5)),
        "GroupLasso": (
            bt_pen.GroupLassoPenalty(init, [[0, 2], [1]]),
            rs_pen.GroupLassoPenalty(init, [[0, 2], [1]]),
        ),
        "L0": (bt_pen.L0PenaltyApprox(init, 0.25), rs_pen.L0PenaltyApprox(init, 0.25)),
        # BoTorch's module, called through torch.
        "BoTorch module": (bt_pen.L2Penalty(init), bt_pen.L2Penalty(init)),
    }


@pytest.mark.parametrize("penalty", list(_penalties(torch.zeros(3, dtype=DT))))
def test_penalized_acquisition_function_matches_botorch(penalty: str) -> None:
    bt, rs, X = _case("m1_norm")
    d = X.shape[-1]
    init = torch.tensor(INIT, dtype=DT)
    p_bt, p_rs = _penalties(init)[penalty]
    best = f32(bt.posterior(X).mean.max().item())
    a_bt = bt_pen.PenalizedAcquisitionFunction(bt_an.ExpectedImprovement(bt, best), p_bt, 0.3)
    a_rs = rs_pen.PenalizedAcquisitionFunction(rs_an.ExpectedImprovement(rs, best), p_rs, 0.3)
    _compare(f"PenalizedAcquisitionFunction {penalty}", a_rs, a_bt, points_in(X, (4, 1, d), 90))
    if penalty != "GroupLasso":
        # q > 1 with an MC acquisition function.
        s_bt, s_rs = sampler_pair(32, (1, 2), 91)
        a_bt = bt_pen.PenalizedAcquisitionFunction(
            bt_mc.qExpectedImprovement(bt, best, sampler=s_bt), p_bt, 0.3
        )
        a_rs = rs_pen.PenalizedAcquisitionFunction(
            rs_mc.qExpectedImprovement(rs, best, sampler=s_rs), p_rs, 0.3
        )
        _compare(
            f"PenalizedAcquisitionFunction {penalty} q=2", a_rs, a_bt, points_in(X, (4, 2, d), 92)
        )
        # Called directly on torch tensors.
        Xt = points_in(X, (4, 2, d), 93)
        CHECK.check(f"{penalty} penalty", p_rs(Xt), p_bt(Xt), VALUE_TOL)


@pytest.mark.parametrize("penalty", ["L1", "L0"])
def test_penalized_mc_objective_matches_botorch(penalty: str) -> None:
    bt, rs, X = _case("m1_norm")
    d = X.shape[-1]
    init = torch.tensor(INIT, dtype=DT)
    if penalty == "L1":
        p_bt, p_rs = bt_pen.L1PenaltyObjective(init), rs_pen.L1PenaltyObjective(init)
    else:
        p_bt = bt_pen.L0PenaltyApproxObjective(init, 0.25)
        p_rs = rs_pen.L0PenaltyApproxObjective(init, 0.25)
    best = f32(bt.posterior(X).mean.max().item())
    o_bt = bt_pen.PenalizedMCObjective(lambda Y, X: Y[..., 0], p_bt, 0.2)
    o_rs = rs_pen.PenalizedMCObjective(lambda Y, X: Y[..., 0], p_rs, 0.2)
    for q in (1, 2):
        s_bt, s_rs = sampler_pair(32, (1, q), 94)
        a_bt = bt_mc.qExpectedImprovement(bt, best, sampler=s_bt, objective=o_bt)
        Xt = points_in(X, (4, q, d), 95)
        for name, obj in (("botorch_rs", o_rs), ("BoTorch", o_bt)):
            a_rs = rs_mc.qExpectedImprovement(rs, best, sampler=s_rs, objective=obj)
            _compare(f"qEI with {name}'s PenalizedMCObjective({penalty})", a_rs, a_bt, Xt)
        # qNEI evaluates the objective at the baseline and the candidates.
        with seeded_draws():
            n = X.shape[0]
            a_bt = bt_mc.qNoisyExpectedImprovement(
                bt,
                X,
                sampler=BtSampler(sample_shape=torch.Size([32])),
                objective=o_bt,
                prune_baseline=False,
                cache_root=False,
            )
            a_rs = rs_mc.qNoisyExpectedImprovement(
                rs,
                X,
                sampler=rs_sampler(32, [n, n + q]),
                objective=o_rs,
                prune_baseline=False,
                cache_root=False,
            )
            _compare(f"qNEI with PenalizedMCObjective({penalty})", a_rs, a_bt, Xt)


@pytest.mark.parametrize("beta", [None, 2.0])
@pytest.mark.parametrize("transformed_weighting", [True, False])
@pytest.mark.parametrize("case", ["m1_rbf", "m1_norm", "list"])
def test_proximal_acquisition_function_matches_botorch(
    case: str, transformed_weighting: bool, beta: Any
) -> None:
    bt, rs, X = _case(case)
    d = X.shape[-1]
    if case == "list":
        # Equal training inputs are required: a list of two copies of the
        # same sub-model.
        from botorch.models import ModelListGP as BtList

        from botorch_rs.models import ModelListGP as RsList

        b0, r0, X = _case("m1_norm")
        bt, rs = BtList(b0, b0), RsList(r0, r0)
        bt.eval()
        d = X.shape[-1]
        base_bt = bt_an.PosteriorMean(
            bt, posterior_transform=BtSPT(torch.tensor([0.5, 0.5], dtype=DT))
        )
        base_rs = rs_an.PosteriorMean(
            rs, posterior_transform=RsSPT(torch.tensor([0.5, 0.5], dtype=DT))
        )
    else:
        best = f32(bt.posterior(X).mean.max().item())
        base_bt, base_rs = bt_an.ExpectedImprovement(bt, best), rs_an.ExpectedImprovement(rs, best)
    w = torch.tensor([0.3, 0.5, 0.4][:d], dtype=DT)
    if beta is None and case == "list":
        pytest.skip("negative posterior means need beta")
    a_bt = bt_prox.ProximalAcquisitionFunction(
        base_bt, w, transformed_weighting=transformed_weighting, beta=beta
    )
    a_rs = rs_prox.ProximalAcquisitionFunction(
        base_rs, w, transformed_weighting=transformed_weighting, beta=beta
    )
    _compare("ProximalAcquisitionFunction", a_rs, a_bt, points_in(X, (4, 1, d), 96))


def test_fixed_feature_acquisition_function_matches_botorch() -> None:
    bt, rs, X = _case("m1_norm")
    d = X.shape[-1]
    best = f32(bt.posterior(X).mean.max().item())
    for cols, vals in (
        ([1], [0.4]),
        ([0, 2], torch.tensor([[0.1, 0.9], [0.3, 0.2]], dtype=DT)),
        ([2, 0], [torch.tensor(0.25, dtype=DT), 0.75]),
    ):
        s_bt, s_rs = sampler_pair(32, (1, 2), 97)
        a_bt = bt_ff.FixedFeatureAcquisitionFunction(
            bt_mc.qExpectedImprovement(bt, best, sampler=s_bt), d, cols, vals
        )
        a_rs = rs_ff.FixedFeatureAcquisitionFunction(
            rs_mc.qExpectedImprovement(rs, best, sampler=s_rs), d, cols, vals
        )
        Xs = points_in(X, (4, 2, d - len(cols)), 98)
        _compare(f"FixedFeatureAcquisitionFunction {cols}", a_rs, a_bt, Xs)
        CHECK.check(
            "FixedFeature _construct_X_full",
            a_rs._construct_X_full(Xs),
            a_bt._construct_X_full(Xs),
            0.0,
        )


@pytest.mark.parametrize("log", [False, True])
def test_prior_guided_acquisition_function_matches_botorch(log: bool) -> None:
    bt, rs, X = _case("m1_norm")
    d = X.shape[-1]
    best = f32(bt.posterior(X).mean.max().item())
    pending = points_in(X, (1, d), 99)
    for pend in (None, pending):
        k = 2 + (0 if pend is None else 1)
        s_bt, s_rs = sampler_pair(32, (1, k), 100)
        if log:
            b_bt = bt_logei.qLogExpectedImprovement(bt, best, sampler=s_bt)
            b_rs = rs_logei.qLogExpectedImprovement(rs, best, sampler=s_rs)
        else:
            b_bt = bt_mc.qExpectedImprovement(bt, best, sampler=s_bt)
            b_rs = rs_mc.qExpectedImprovement(rs, best, sampler=s_rs)
        a_bt = bt_pg.PriorGuidedAcquisitionFunction(
            b_bt, Prior(log), log=log, prior_exponent=0.75, X_pending=pend
        )
        a_rs = rs_pg.PriorGuidedAcquisitionFunction(
            b_rs, Prior(log), log=log, prior_exponent=0.75, X_pending=pend
        )
        _compare(f"PriorGuided MC log={log}", a_rs, a_bt, points_in(X, (4, 2, d), 101))
    if log:
        b_bt, b_rs = bt_an.LogExpectedImprovement(bt, best), rs_an.LogExpectedImprovement(rs, best)
    else:
        b_bt, b_rs = bt_an.ExpectedImprovement(bt, best), rs_an.ExpectedImprovement(rs, best)
    a_bt = bt_pg.PriorGuidedAcquisitionFunction(b_bt, Prior(log), log=log, prior_exponent=0.75)
    a_rs = rs_pg.PriorGuidedAcquisitionFunction(b_rs, Prior(log), log=log, prior_exponent=0.75)
    _compare(f"PriorGuided analytic log={log}", a_rs, a_bt, points_in(X, (4, 1, d), 102))


# ---------------------------------------------------------------------------
# optimize_acqf
# ---------------------------------------------------------------------------


def _optimize_pair(a_bt: Any, a_rs: Any, X: Any, q: int, name: str, seed: int) -> None:
    """``optimize_acqf`` from the same initial conditions: the same local
    optimum. L-BFGS-B stops on rounding-dependent criteria, so the values
    agree to 1e-6 and, near a quadratic optimum, the candidates to about the
    square root of that (1e-3; measured worst in the summary); BoTorch's value
    at botorch_rs's candidates is checked as well."""
    bounds = _bounds(X)
    d = X.shape[-1]
    ics = points_in(X, (3, q, d), seed).clamp(bounds[0], bounds[1])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cb, vb = bt_optimize.optimize_acqf(
            a_bt, bounds, q=q, num_restarts=3, batch_initial_conditions=ics
        )
    cr, vr = rs_optimize.optimize_acqf(
        a_rs, bounds, q=q, num_restarts=3, batch_initial_conditions=ics
    )
    CHECK.check(f"optimize_acqf {name} value", vr, vb.detach(), 1e-6)
    CHECK.check(
        f"optimize_acqf {name}: BoTorch's value at the candidate",
        a_bt(cr).detach().reshape(vb.shape),
        vb.detach(),
        1e-6,
    )
    CHECK.check(f"optimize_acqf {name} candidate", cr, cb, 1e-3)


def test_optimize_acqf_matches_botorch() -> None:
    bt, rs, X = _case("m1_rbf")
    d = X.shape[-1]
    with seeded_draws():
        a_bt = bt_mc.qNoisyExpectedImprovement(
            bt, X, sampler=BtSampler(sample_shape=torch.Size([NEI_SAMPLES])), prune_baseline=False
        )
        n = X.shape[0]
        a_rs = rs_mc.qNoisyExpectedImprovement(
            rs, X, sampler=rs_sampler(NEI_SAMPLES, [n, n + 1]), prune_baseline=False
        )
        # q = 1: with q > 1 the sample-average surface has kinks where
        # L-BFGS-B's path depends on rounding (measured: one of three restarts
        # ends at a different point of the same optimum, 1e-3 lower).
        _optimize_pair(a_bt, a_rs, X, 1, "qNEI", 110)
    cs = points_in(X, (20, d), 111)
    a_bt, a_rs = mes_pair("qLowerBoundMaxValueEntropy", bt, rs, cs, None, {}, {})
    _optimize_pair(a_bt, a_rs, X, 1, "GIBBON", 112)
    mc = points_in(X, (12, d), 113)
    a_bt = bt_al.qNegIntegratedPosteriorVariance(bt, mc)
    a_rs = rs_al.qNegIntegratedPosteriorVariance(rs, mc)
    with no_fast_pred_var():
        _optimize_pair(a_bt, a_rs, X, 1, "qNIPV", 114)


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


def _same_error(name: str, make: Any, exc: type = Exception) -> None:
    with pytest.raises(exc) as e_bt:
        make("bt")
    with pytest.raises(exc) as e_rs:
        make("rs")
    assert str(e_bt.value).strip() in str(e_rs.value), (name, str(e_bt.value), str(e_rs.value))


def test_errors_match_botorch() -> None:
    b1, r1, X = _case("m1_rbf")
    b2, r2, _ = _case("m2")
    d = X.shape[-1]
    pick = {"bt": b1, "rs": r1}
    pick2 = {"bt": b2, "rs": r2}
    kg = {"bt": bt_kg, "rs": rs_kg}
    mes = {"bt": bt_mes, "rs": rs_mes}
    jes = {"bt": bt_jes, "rs": rs_jes}
    ts = {"bt": bt_ts, "rs": rs_ts}
    an = {"bt": bt_an, "rs": rs_an}
    pen = {"bt": bt_pen, "rs": rs_pen}
    prox = {"bt": bt_prox, "rs": rs_prox}
    pg = {"bt": bt_pg, "rs": rs_pg}
    mc = {"bt": bt_mc, "rs": rs_mc}
    cs = points_in(X, (10, d), 120)
    _same_error(
        "qKG without num_fantasies",
        lambda lib: kg[lib].qKnowledgeGradient(pick[lib], num_fantasies=None),
        ValueError,
    )
    _same_error(
        "qKG multi-output without objective",
        lambda lib: kg[lib].qKnowledgeGradient(pick2[lib], num_fantasies=2),
    )
    _same_error(
        "qMFKG cost without current_value",
        lambda lib: kg[lib].qMultiFidelityKnowledgeGradient(
            pick[lib],
            num_fantasies=2,
            cost_aware_utility=(bt_cost if lib == "bt" else rs_cost).InverseCostWeightedUtility(
                (BtAffineCost if lib == "bt" else RsAffineCost)(fidelity_weights={d - 1: 1.0})
            ),
        ),
    )
    _same_error("qMES multi-output", lambda lib: mes[lib].qMaxValueEntropy(pick2[lib], cs))
    _same_error(
        "qMFGIBBON with trace observations",
        lambda lib: mes[lib].qMultiFidelityLowerBoundMaxValueEntropy(
            pick[lib], cs, expand=lambda X: X
        ),
    )
    xo = cs[:3]
    fo = torch.zeros(3, 1, dtype=DT)
    _same_error(
        "qJES estimation type",
        lambda lib: jes[lib].qJointEntropySearch(pick[lib], xo, fo, estimation_type="XX")(
            cs[:1].unsqueeze(0)
        ),
    )
    _same_error(
        "PathwiseThompsonSampling multi-output",
        lambda lib: ts[lib].PathwiseThompsonSampling(pick2[lib]),
    )

    def ts_batch_change(lib: str) -> None:
        a = ts[lib].PathwiseThompsonSampling(pick[lib])
        a(points_in(X, (1, 2, d), 121))
        a(points_in(X, (1, 3, d), 122))

    _same_error("PathwiseThompsonSampling batch size", ts_batch_change, ValueError)
    _same_error(
        "LogProbabilityOfFeasibility without constraints",
        lambda lib: an[lib].LogProbabilityOfFeasibility(pick[lib], constraints={}),
        ValueError,
    )
    _same_error(
        "Penalized analytic X_pending",
        lambda lib: (
            pen[lib]
            .PenalizedAcquisitionFunction(
                an[lib].ExpectedImprovement(pick[lib], 0.5), pen[lib].L2Penalty(xo[0]), 0.1
            )
            .set_X_pending(xo)
        ),
    )

    def prox_negative(lib: str) -> None:
        a = prox[lib].ProximalAcquisitionFunction(
            an[lib].PosteriorMean(pick[lib], maximize=False), torch.ones(d, dtype=DT)
        )
        a(points_in(X, (2, 1, d), 123))

    _same_error("Proximal negative values", prox_negative, RuntimeError)
    _same_error(
        "Proximal weights",
        lambda lib: prox[lib].ProximalAcquisitionFunction(
            an[lib].ExpectedImprovement(pick[lib], 0.5), torch.ones(d + 1, dtype=DT)
        ),
        ValueError,
    )

    def pg_pending(lib: str) -> None:
        base = mc[lib].qExpectedImprovement(pick[lib], 0.5)
        base.set_X_pending(xo)
        pg[lib].PriorGuidedAcquisitionFunction(base, Prior(False))

    _same_error("PriorGuided base X_pending", pg_pending)

    def pg_q2(lib: str) -> None:
        a = pg[lib].PriorGuidedAcquisitionFunction(
            an[lib].ExpectedImprovement(pick[lib], 0.5), Prior(False)
        )
        a(points_in(X, (1, 2, d), 124))

    _same_error("PriorGuided analytic q > 1", pg_q2, NotImplementedError)

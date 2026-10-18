"""Parity of phase 8b with BoTorch, through the public APIs of both libraries
side by side:

* the input constructors (``get_acqf_input_constructor`` and every
  ``construct_inputs_*`` of the acquisition functions botorch_rs has) with
  BoTorch's and botorch_rs's datasets, objectives, posterior transforms and
  constraints: the constructed keyword arguments (``best_f``, ``X_baseline``,
  reference points, partitionings, candidate sets, current values, ...) and
  the values and gradients of the acquisition functions built from them;
  ``get_best_f_analytic``, ``get_best_f_mc``, ``optimize_objective``,
  ``compute_best_feasible_objective``, ``get_infeasible_cost``;
* ``get_acquisition_function`` for every name BoTorch handles;
* ``qMultiStepLookahead`` (values against BoTorch, gradients against the
  corrected BoTorch), ``make_best_f``, ``warmstart_multistep``, the tree
  representation, ``optimize_acqf`` from identical initial conditions and the
  induced fantasy model;
* Python subclasses of ``SampleReducingMCAcquisitionFunction`` and
  ``LogImprovementMCAcquisitionFunction`` (the same code on both sides);
* t-batched fixed values of ``FixedFeatureAcquisitionFunction`` and batched
  candidate sets of ``_sample_max_value_Gumbel`` / ``_Thompson``.

Tolerances: values 1e-9 and gradients 1e-8 (relative, mixed elementwise /
normwise); constructed values that do not depend on a model (``best_f`` from
the training data, ``X_baseline``, reference points, candidate sets drawn
with torch's generator) must be identical; values computed through a
model's posterior (partitionings, ``get_acquisition_function``'s ``best_f``,
infeasible costs) 1e-9; ``optimize_objective`` and ``optimize_acqf`` from
identical initial conditions 1e-6 (values; L-BFGS-B stops on
rounding-dependent criteria) and 1e-4 (candidates).

BoTorch bugs (see ``docs/drafts/8b-acquisition-rest.md``), with corrected
references: the gradients of ``qMultiStepLookahead`` (compared with BoTorch
under ``propagate_grads(True)`` without ``fast_pred_var``), ``make_best_f``
(the untransformed best value; analytic value functions get a trailing
dimension), ``collapse_fantasy_base_samples=False`` for single-output
posteriors (BoTorch with the single-output batch range), the batched layout
of ``q_j > 1`` (one t-batch), ``construct_inputs_qSimpleRegret`` with
constraints and no objective (the identity objective).

Skipped unless BoTorch is importable (required in CI's parity job).
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest

from helpers import importorskip_botorch

importorskip_botorch()
torch = pytest.importorskip("torch")

import botorch.acquisition as ba  # noqa: E402
import botorch.acquisition.factory as bt_factory  # noqa: E402
import botorch.acquisition.input_constructors as bic  # noqa: E402
import botorch.acquisition.utils as bt_autils  # noqa: E402
from acquisition_rest_helpers import (  # noqa: E402
    best_f_argfac,
    bounded_subclass,
    case,
    copy_lookahead_samplers,
    corrected_lookahead,
    datasets,
    improvement_subclass,
    log_improvement_subclass,
    single_output_collapse_fix,
    values_and_grads,
)
from botorch.acquisition import fixed_feature as bt_ff  # noqa: E402
from botorch.acquisition import logei as bt_logei  # noqa: E402
from botorch.acquisition import max_value_entropy_search as bt_mes  # noqa: E402
from botorch.acquisition import monte_carlo as bt_mc  # noqa: E402
from botorch.acquisition import multi_step_lookahead as bt_msl  # noqa: E402
from botorch.acquisition.multi_objective import objective as bt_moo  # noqa: E402
from botorch.acquisition.objective import GenericMCObjective as BtGeneric  # noqa: E402
from botorch.acquisition.objective import LinearMCObjective as BtLinear  # noqa: E402
from botorch.acquisition.objective import (  # noqa: E402
    ScalarizedPosteriorTransform as BtSPT,
)
from botorch.optim import optimize as bt_optimize  # noqa: E402
from botorch.sampling.normal import SobolQMCNormalSampler as BtSampler  # noqa: E402
from so_acquisition_helpers import (  # noqa: E402
    NEI_SAMPLES,
    recorded_optimal_samples,
    rs_sampler,
    seeded_draws,
)

import botorch_rs.acquisition as ra  # noqa: E402
import botorch_rs.acquisition.factory as rs_factory  # noqa: E402
import botorch_rs.acquisition.input_constructors as ric  # noqa: E402
import botorch_rs.acquisition.utils as rs_autils  # noqa: E402
from botorch_rs.acquisition import fixed_feature as rs_ff  # noqa: E402
from botorch_rs.acquisition import logei as rs_logei  # noqa: E402
from botorch_rs.acquisition import max_value_entropy_search as rs_mes  # noqa: E402
from botorch_rs.acquisition import monte_carlo as rs_mc  # noqa: E402
from botorch_rs.acquisition import multi_step_lookahead as rs_msl  # noqa: E402
from botorch_rs.acquisition.multi_objective import objective as rs_moo  # noqa: E402
from botorch_rs.acquisition.objective import GenericMCObjective as RsGeneric  # noqa: E402
from botorch_rs.acquisition.objective import LinearMCObjective as RsLinear  # noqa: E402
from botorch_rs.acquisition.objective import (  # noqa: E402
    ScalarizedPosteriorTransform as RsSPT,
)
from botorch_rs.optim import optimize as rs_optimize  # noqa: E402
from botorch_rs.sampling import SobolQMCNormalSampler as RsSampler  # noqa: E402
from parity_helpers import Checker, points_in, sampler_pair  # noqa: E402

pytestmark = pytest.mark.filterwarnings("ignore")

DT = torch.float64
VALUE_TOL = 1e-9
GRAD_TOL = 1e-8
OPT_VALUE_TOL = 1e-6
OPT_X_TOL = 1e-4
CHECK = Checker("acquisition rest (phase 8b)")
W2 = [0.75, -0.25]


@pytest.fixture(autouse=True, scope="module")
def _single_threaded_torch() -> Iterator[None]:
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(threads)


def _bt_cls(name: str) -> Any:
    """BoTorch's acquisition function class of a name (from the registry)."""
    by_name = {c.__name__: c for c in bic.ACQF_INPUT_CONSTRUCTOR_REGISTRY}
    return by_name[name] if name in by_name else getattr(ba, name)


def _rs_cls(name: str) -> Any:
    by_name = {c.__name__: c for c in ric.ACQF_INPUT_CONSTRUCTOR_REGISTRY}
    return by_name[name] if name in by_name else getattr(ra, name)


def _exact(quantity: str, a: Any, e: Any) -> None:
    """Identical values (``torch.equal`` of float64 tensors)."""
    at = torch.as_tensor(a, dtype=DT)
    et = torch.as_tensor(e, dtype=DT)
    same = at.shape == et.shape and bool(((at == et) | (torch.isnan(at) & torch.isnan(et))).all())
    CHECK.record(quantity, 0.0 if same else float("inf"), 0.0)
    assert same, (quantity, a, e)


def _compare_restarts(
    label: str, a_bt: Any, xr: Any, vr: Any, xb: Any, vb: Any, bt_ctx: Any = None
) -> None:
    """``optimize_acqf`` from identical initial conditions, restart by
    restart: all but one restart reach BoTorch's value (L-BFGS-B stops on
    rounding-dependent criteria, and BoTorch 0.18.1 stops some restarts
    early), and BoTorch's value at botorch_rs's candidates is at least
    BoTorch's own (flat optima make the candidates themselves
    ill-conditioned)."""
    import contextlib

    vr = torch.as_tensor(vr, dtype=DT).reshape(-1)
    vb = torch.as_tensor(vb, dtype=DT).reshape(-1)
    scale = vb.abs().clamp_min(1.0)
    differ = int(((vr - vb).abs() > OPT_VALUE_TOL * scale).sum())
    CHECK.check_bound(
        f"{label}: restarts away from BoTorch's value (excess over one)", differ - 1, 0.0
    )
    with bt_ctx or contextlib.nullcontext(), torch.no_grad():
        at_rs = a_bt(torch.as_tensor(xr, dtype=DT))
    shortfall = ((vb - at_rs) / scale).max().item()
    CHECK.check_bound(
        f"{label}: BoTorch's value at the candidates (shortfall)", shortfall, OPT_VALUE_TOL
    )


def _compare_acqf(label: str, a_bt: Any, a_rs: Any, X: Any, grads: bool = True) -> None:
    if grads:
        vb, gb = values_and_grads(a_bt, X)
        vr, gr = values_and_grads(a_rs, X)
        CHECK.check(f"{label}: values", vr, vb, VALUE_TOL)
        CHECK.check(f"{label}: gradients", gr, gb, GRAD_TOL, atol=1e-14)
    else:
        with torch.no_grad():
            CHECK.check(f"{label}: values", a_rs(X), a_bt(X), VALUE_TOL)


# ---------------------------------------------------------------------------
# get_best_f_* and the registry
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("lib", ["botorch", "botorch_rs"])
@pytest.mark.parametrize("block", [False, True])
def test_get_best_f_matches_botorch(lib: str, block: bool) -> None:
    _, _, X, Y = case("m2")
    ds_bt = datasets(X, Y, "botorch", block)
    ds = datasets(X, Y, lib, block)
    # Analytic: a scalarizing transform (botorch_rs's or BoTorch's).
    for tf in (RsSPT(torch.tensor(W2, dtype=DT)), BtSPT(torch.tensor(W2, dtype=DT))):
        _exact(
            "get_best_f_analytic (transform)",
            ric.get_best_f_analytic(ds, posterior_transform=tf),
            bic.get_best_f_analytic(ds_bt, posterior_transform=BtSPT(torch.tensor(W2, dtype=DT))),
        )
    _, _, X1, Y1 = case("m1")
    _exact(
        "get_best_f_analytic",
        ric.get_best_f_analytic(datasets(X1, Y1, lib, block)),
        bic.get_best_f_analytic(datasets(X1, Y1, "botorch", block)),
    )
    # MC: objectives of both libraries, transforms, constraints.
    w = torch.tensor([1.0, -0.5], dtype=DT)
    for obj_rs, obj_bt in (
        (RsLinear(w), BtLinear(w)),
        (BtLinear(w), BtLinear(w)),
        (
            RsGeneric(lambda Y, X=None: Y[..., 0] * Y[..., 1]),
            BtGeneric(lambda Y, X=None: Y[..., 0] * Y[..., 1]),
        ),
    ):
        _exact(
            "get_best_f_mc (objective)",
            ric.get_best_f_mc(ds, objective=obj_rs),
            bic.get_best_f_mc(ds_bt, objective=obj_bt),
        )
    _exact(
        "get_best_f_mc (transform)",
        ric.get_best_f_mc(ds, posterior_transform=RsSPT(torch.tensor(W2, dtype=DT))),
        bic.get_best_f_mc(ds_bt, posterior_transform=BtSPT(torch.tensor(W2, dtype=DT))),
    )
    cons = [lambda Z: Z[..., 1] - Y[:, 1].median()]
    _exact(
        "get_best_f_mc (constraints)",
        ric.get_best_f_mc(ds, objective=RsLinear(w), constraints=cons),
        bic.get_best_f_mc(ds_bt, objective=BtLinear(w), constraints=cons),
    )


def test_get_best_f_mc_without_feasible_points_matches_botorch() -> None:
    bt, rs, X, Y = case("m2")
    w = torch.tensor([1.0, -0.5], dtype=DT)
    cons = [lambda Z: -Z[..., 1] + 100.0]
    # The lower bound of the objective is estimated at points drawn with
    # torch's generator, as in BoTorch.
    torch.manual_seed(7)
    e = bic.get_best_f_mc(
        datasets(X, Y, "botorch"), objective=BtLinear(w), constraints=cons, model=bt
    )
    torch.manual_seed(7)
    a = ric.get_best_f_mc(
        datasets(X, Y, "botorch_rs"), objective=RsLinear(w), constraints=cons, model=rs
    )
    CHECK.check("get_best_f_mc (no feasible point)", a, e, VALUE_TOL)


def test_infeasible_cost_and_best_feasible_objective_match_botorch() -> None:
    bt, rs, X, _ = case("m2")
    obj = lambda Y, X=None: Y[..., 0] - 2.0 * Y[..., 1]  # noqa: E731
    CHECK.check(
        "get_infeasible_cost",
        rs_autils.get_infeasible_cost(X, rs, obj),
        bt_autils.get_infeasible_cost(X, bt, obj),
        VALUE_TOL,
    )
    tf_b = BtSPT(torch.tensor(W2, dtype=DT))
    tf_r = RsSPT(torch.tensor(W2, dtype=DT))
    CHECK.check(
        "get_infeasible_cost (transform)",
        rs_autils.get_infeasible_cost(X, rs, posterior_transform=tf_r),
        bt_autils.get_infeasible_cost(X, bt, posterior_transform=tf_b),
        VALUE_TOL,
    )
    g = torch.Generator().manual_seed(3)
    samples = torch.randn(4, 3, 5, 2, generator=g, dtype=DT)
    o = samples[..., 0]
    cons = [lambda Z: Z[..., 1] - 0.5]
    _exact(
        "compute_best_feasible_objective",
        rs_autils.compute_best_feasible_objective(samples, o, cons),
        bt_autils.compute_best_feasible_objective(samples, o, cons),
    )
    _exact(
        "compute_best_feasible_objective (unconstrained)",
        rs_autils.compute_best_feasible_objective(samples, o, None),
        bt_autils.compute_best_feasible_objective(samples, o, None),
    )
    inf = torch.tensor(-3.0, dtype=DT)
    _exact(
        "compute_best_feasible_objective (infeasible_obj)",
        rs_autils.compute_best_feasible_objective(
            samples, o, [lambda Z: Z[..., 1] + 10.0], infeasible_obj=inf
        ),
        bt_autils.compute_best_feasible_objective(
            samples, o, [lambda Z: Z[..., 1] + 10.0], infeasible_obj=inf
        ),
    )


def test_registry_matches_botorch() -> None:
    names_bt = {c.__name__ for c in bic.ACQF_INPUT_CONSTRUCTOR_REGISTRY}
    names_rs = {c.__name__ for c in ric.ACQF_INPUT_CONSTRUCTOR_REGISTRY}
    # Only the constructors of the acquisition functions botorch_rs does not
    # have (yet) are missing.
    for name in names_bt - names_rs:
        assert not hasattr(ra, name), name
    assert names_rs <= names_bt
    for cls in ric.ACQF_INPUT_CONSTRUCTOR_REGISTRY:
        fb = bic.ACQF_INPUT_CONSTRUCTOR_REGISTRY[_bt_cls(cls.__name__)]
        fr = ric.ACQF_INPUT_CONSTRUCTOR_REGISTRY[cls]
        # The same constructor (by the name of the wrapped function).
        assert fb.__closure__ is not None and fr.__closure__ is not None
        name_b = fb.__closure__[1].cell_contents.__name__
        name_r = fr.__closure__[1].cell_contents.__name__
        assert name_b == name_r, (cls.__name__, name_b, name_r)


# ---------------------------------------------------------------------------
# Constructors of the analytic acquisition functions
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name",
    [
        "ExpectedImprovement",
        "LogExpectedImprovement",
        "ProbabilityOfImprovement",
        "LogProbabilityOfImprovement",
        "UpperConfidenceBound",
        "PosteriorMean",
    ],
)
@pytest.mark.parametrize("model", ["m1", "m1_norm", "m2"])
@pytest.mark.parametrize("lib", ["botorch", "botorch_rs"])
def test_analytic_constructors_match_botorch(name: str, model: str, lib: str) -> None:
    bt, rs, X, Y = case(model)
    kb: dict[str, Any] = {}
    kr: dict[str, Any] = {}
    if model == "m2":
        kb["posterior_transform"] = BtSPT(torch.tensor(W2, dtype=DT))
        kr["posterior_transform"] = RsSPT(torch.tensor(W2, dtype=DT))
    if name == "UpperConfidenceBound":
        # A value that BoTorch's float32 buffer holds exactly.
        kb["beta"] = kr["beta"] = 0.75
    if name.startswith(("Expected", "Log", "Prob")):
        kb["maximize"] = kr["maximize"] = model != "m1_norm"
    cb = bic.get_acqf_input_constructor(_bt_cls(name))
    cr = ric.get_acqf_input_constructor(_rs_cls(name))
    ab = cb(model=bt, training_data=datasets(X, Y, "botorch"), **kb)
    ar = cr(model=rs, training_data=datasets(X, Y, lib), **kr)
    assert set(ab) == set(ar)
    if "best_f" in ab:
        _exact("construct_inputs_best_f: best_f", ar["best_f"], ab["best_f"])
    for key in ("beta", "maximize"):
        if key in ab:
            assert ar[key] == ab[key]
    Xt = points_in(X, (3, 1, X.shape[-1]), 1)
    _compare_acqf(
        f"{name} from its input constructor", _bt_cls(name)(**ab), _rs_cls(name)(**ar), Xt
    )


def test_pof_and_logcei_constructors_match_botorch() -> None:
    bt, rs, X, Y = case("m2")
    A = torch.tensor([[0.0, 2.0], [-1.0, 0.0]], dtype=DT)
    b = torch.tensor([[1.5], [-0.25]], dtype=DT)
    ab = bic.get_acqf_input_constructor(_bt_cls("LogProbabilityOfFeasibility"))(
        model=bt, constraints_tuple=(A, b)
    )
    ar = ric.get_acqf_input_constructor(_rs_cls("LogProbabilityOfFeasibility"))(
        model=rs, constraints_tuple=(A, b)
    )
    assert ab["constraints"].keys() == ar["constraints"].keys()
    for k in ab["constraints"]:
        for x, y in zip(ab["constraints"][k], ar["constraints"][k], strict=True):
            assert (x is None and y is None) or float(x) == float(y)
    Xt = points_in(X, (3, 1, 2), 2)
    _compare_acqf(
        "LogProbabilityOfFeasibility from its input constructor",
        _bt_cls("LogProbabilityOfFeasibility")(**ab),
        _rs_cls("LogProbabilityOfFeasibility")(**ar),
        Xt,
    )
    A1 = torch.tensor([[0.0, 1.0]], dtype=DT)
    b1 = torch.tensor([[2.0]], dtype=DT)
    Y0 = Y[..., :1]
    ab = bic.get_acqf_input_constructor(_bt_cls("LogConstrainedExpectedImprovement"))(
        model=bt,
        training_data=datasets(X, Y0, "botorch"),
        objective_index=0,
        constraints_tuple=(A1, b1),
    )
    ar = ric.get_acqf_input_constructor(_rs_cls("LogConstrainedExpectedImprovement"))(
        model=rs,
        training_data=datasets(X, Y0, "botorch_rs"),
        objective_index=0,
        constraints_tuple=(A1, b1),
    )
    _exact("construct_inputs_logcei: best_f", ar["best_f"], ab["best_f"])
    _compare_acqf(
        "LogConstrainedExpectedImprovement from its input constructor",
        _bt_cls("LogConstrainedExpectedImprovement")(**ab),
        _rs_cls("LogConstrainedExpectedImprovement")(**ar),
        Xt,
    )


@pytest.mark.parametrize("name", ["NoisyExpectedImprovement", "LogNoisyExpectedImprovement"])
def test_noisy_ei_constructors_match_botorch(name: str) -> None:
    bt, rs, X, Y = case("m1_fixed")
    ab = bic.get_acqf_input_constructor(_bt_cls(name))(
        model=bt, training_data=datasets(X, Y, "botorch", block=True), num_fantasies=8
    )
    ar = ric.get_acqf_input_constructor(_rs_cls(name))(
        model=rs, training_data=datasets(X, Y, "botorch_rs", block=True), num_fantasies=8
    )
    assert set(ab) == set(ar)
    _exact("construct_inputs_noisy_ei: X_observed", ar["X_observed"], ab["X_observed"])
    assert ar["num_fantasies"] == ab["num_fantasies"] and ar["maximize"] == ab["maximize"]
    _rs_cls(name)(**ar)


# ---------------------------------------------------------------------------
# Constructors of the MC acquisition functions
# ---------------------------------------------------------------------------


def _mc_setup(model: str, variant: str) -> tuple[dict, dict]:
    """Keyword arguments of both libraries: objectives / transforms /
    constraints of a variant."""
    kb: dict[str, Any] = {}
    kr: dict[str, Any] = {}
    if model in ("m2", "list"):
        if variant == "transform":
            kb["posterior_transform"] = BtSPT(torch.tensor(W2, dtype=DT))
            kr["posterior_transform"] = RsSPT(torch.tensor(W2, dtype=DT))
        else:
            w = torch.tensor([1.0, -0.5], dtype=DT)
            kb["objective"] = BtLinear(w)
            kr["objective"] = RsLinear(w) if variant != "botorch_objective" else BtLinear(w)
    if variant == "constraints":
        kb["constraints"] = kr["constraints"] = [lambda Z: Z[..., -1] - 1.5]
        kb["eta"] = kr["eta"] = 2.0**-8
    return kb, kr


@pytest.mark.parametrize(
    "name", ["qExpectedImprovement", "qLogExpectedImprovement", "qProbabilityOfImprovement"]
)
@pytest.mark.parametrize(
    ("model", "variant"),
    [
        ("m1", "plain"),
        ("m1_nostd", "constraints"),
        ("m2", "objective"),
        ("m2", "botorch_objective"),
        ("m2", "transform"),
        ("m2", "constraints"),
        ("list", "objective"),
    ],
)
def test_best_f_mc_constructors_match_botorch(name: str, model: str, variant: str) -> None:
    bt, rs, X, Y = case(model)
    kb, kr = _mc_setup(model, variant)
    q = 2
    m = 1 if (model.startswith("m1") or variant == "transform") else 2
    if name == "qProbabilityOfImprovement":
        # A temperature that BoTorch's float32 buffer holds exactly.
        kb["tau"] = kr["tau"] = 2.0**-10
    s_bt, s_rs, _ = sampler_pair(32, q, m, 41)
    for lib in ("botorch", "botorch_rs"):
        # Without feasible observations, the lower bound of the objective is
        # estimated at points drawn with torch's generator.
        torch.manual_seed(1)
        ab = bic.get_acqf_input_constructor(_bt_cls(name))(
            model=bt, training_data=datasets(X, Y, "botorch"), sampler=s_bt, **kb
        )
        torch.manual_seed(1)
        ar = ric.get_acqf_input_constructor(_rs_cls(name))(
            model=rs, training_data=datasets(X, Y, lib), sampler=s_rs, **kr
        )
        assert set(ab) == set(ar)
        cons = kb.get("constraints")
        if cons and not bool((cons[0](Y) <= 0).any()):
            # No feasible observation: a lower bound of the objective from the
            # model's posterior (equal up to the rounding of the posteriors).
            CHECK.check(
                f"construct_inputs_{name}: best_f (no feasible observation)",
                ar["best_f"],
                ab["best_f"],
                VALUE_TOL,
            )
        else:
            _exact(f"construct_inputs_{name}: best_f", ar["best_f"], ab["best_f"])
    Xt = points_in(X, (3, q, X.shape[-1]), 3)
    _compare_acqf(
        f"{name} from its input constructor", _bt_cls(name)(**ab), _rs_cls(name)(**ar), Xt
    )


@pytest.mark.parametrize("name", ["qNoisyExpectedImprovement", "qLogNoisyExpectedImprovement"])
@pytest.mark.parametrize("model", ["m1", "m2"])
def test_noisy_mc_constructors_match_botorch(name: str, model: str) -> None:
    bt, rs, X, Y = case(model)
    kb, kr = _mc_setup(model, "objective")
    for lib in ("botorch", "botorch_rs"):
        ab = bic.get_acqf_input_constructor(_bt_cls(name))(
            model=bt, training_data=datasets(X, Y, "botorch"), **kb
        )
        ar = ric.get_acqf_input_constructor(_rs_cls(name))(
            model=rs, training_data=datasets(X, Y, lib), **kr
        )
        assert set(ab) == set(ar)
        _exact(f"construct_inputs_{name}: X_baseline", ar["X_baseline"], ab["X_baseline"])
        for key in ab:
            if key in ("model", "objective", "sampler", "X_baseline"):
                continue
            assert ar[key] == ab[key], key
    # Values with identical base samples, without pruning.
    N, m = X.shape[0], (1 if model == "m1" else 2)
    q = 2
    with seeded_draws():
        ab = bic.get_acqf_input_constructor(_bt_cls(name))(
            model=bt,
            training_data=datasets(X, Y, "botorch"),
            sampler=BtSampler(sample_shape=torch.Size([NEI_SAMPLES])),
            prune_baseline=False,
            cache_root=False,
            **kb,
        )
        a_bt = _bt_cls(name)(**ab)
    ar = ric.get_acqf_input_constructor(_rs_cls(name))(
        model=rs,
        training_data=datasets(X, Y, "botorch_rs"),
        sampler=rs_sampler(NEI_SAMPLES, [N * m, (N + q) * m]),
        prune_baseline=False,
        cache_root=False,
        **kr,
    )
    a_rs = _rs_cls(name)(**ar)
    with seeded_draws():
        _compare_acqf(f"{name} from its input constructor", a_bt, a_rs, points_in(X, (3, q, 2), 4))


@pytest.mark.parametrize(
    "name", ["qUpperConfidenceBound", "qLowerConfidenceBound", "qSimpleRegret"]
)
@pytest.mark.parametrize("variant", ["plain", "constraints"])
def test_ucb_and_sr_constructors_match_botorch(name: str, variant: str) -> None:
    bt, rs, X, Y = case("m2")
    w = torch.tensor([1.0, -0.5], dtype=DT)
    kb: dict[str, Any] = {"objective": BtLinear(w)}
    kr: dict[str, Any] = {"objective": RsLinear(w)}
    if variant == "constraints":
        kb["constraints"] = kr["constraints"] = [lambda Z: Z[..., -1] - 1.5]
        kb["X_baseline"] = kr["X_baseline"] = X
    if name != "qSimpleRegret":
        kb["beta"] = kr["beta"] = 0.4
    s_bt, s_rs, _ = sampler_pair(32, 2, 2, 43)
    ab = bic.get_acqf_input_constructor(_bt_cls(name))(model=bt, sampler=s_bt, **kb)
    ar = ric.get_acqf_input_constructor(_rs_cls(name))(model=rs, sampler=s_rs, **kr)
    assert set(ab) == set(ar)
    if variant == "constraints":
        # BoTorch keeps the default temperature 1e-3 as a float32 buffer.
        ab["objective"].eta = torch.full((1,), 1e-3, dtype=DT)
        # A ConstrainedMCObjective with BoTorch's infeasible cost.
        CHECK.check(
            f"construct_inputs_{name}: infeasible cost",
            torch.as_tensor(ar["objective"].infeasible_cost, dtype=DT).reshape(-1),
            ab["objective"].infeasible_cost.reshape(-1),
            VALUE_TOL,
        )
    Xt = points_in(X, (3, 2, 2), 5)
    _compare_acqf(
        f"{name} from its input constructor ({variant})",
        _bt_cls(name)(**ab),
        _rs_cls(name)(**ar),
        Xt,
    )


def test_qsr_constructor_with_constraints_and_no_objective() -> None:
    """BoTorch builds a ``ConstrainedMCObjective`` around ``objective=None``
    (which fails when evaluated); botorch_rs uses the identity, which equals
    BoTorch with an explicit ``IdentityMCObjective``."""
    bt, rs, X, Y = case("m1")
    cons = [lambda Z: Z[..., 0] - 1.2]
    s_bt, s_rs, _ = sampler_pair(32, 2, 1, 44)
    ab = bic.get_acqf_input_constructor(ba.qSimpleRegret)(
        model=bt, constraints=cons, X_baseline=X, sampler=s_bt
    )
    with pytest.raises(TypeError):
        ba.qSimpleRegret(**ab)(points_in(X, (1, 2, 2), 6))
    ab = bic.get_acqf_input_constructor(ba.qSimpleRegret)(
        model=bt,
        objective=ba.IdentityMCObjective(),
        constraints=cons,
        X_baseline=X,
        sampler=s_bt,
    )
    ar = ric.get_acqf_input_constructor(ra.qSimpleRegret)(
        model=rs, constraints=cons, X_baseline=X, sampler=s_rs
    )
    # BoTorch keeps the default temperature 1e-3 as a float32 buffer.
    ab["objective"].eta = torch.full((1,), 1e-3, dtype=DT)
    _compare_acqf(
        "qSimpleRegret from its input constructor (constraints, no objective)",
        ba.qSimpleRegret(**ab),
        ra.qSimpleRegret(**ar),
        points_in(X, (3, 2, 2), 6),
    )


def test_log_pf_constructor_matches_botorch() -> None:
    bt, rs, X, Y = case("m2")
    cons = [lambda Z: Z[..., 1] - 1.5, lambda Z: 0.5 - Z[..., 0]]
    s_bt, s_rs, _ = sampler_pair(32, 2, 2, 45)
    ab = bic.get_acqf_input_constructor(_bt_cls("qLogProbabilityOfFeasibility"))(
        model=bt, constraints=cons, sampler=s_bt, eta=2.0**-10
    )
    ar = ric.get_acqf_input_constructor(_rs_cls("qLogProbabilityOfFeasibility"))(
        model=rs, constraints=cons, sampler=s_rs, eta=2.0**-10
    )
    assert set(ab) == set(ar)
    _compare_acqf(
        "qLogProbabilityOfFeasibility from its input constructor",
        _bt_cls("qLogProbabilityOfFeasibility")(**ab),
        _rs_cls("qLogProbabilityOfFeasibility")(**ar),
        points_in(X, (3, 2, 2), 7),
    )


# ---------------------------------------------------------------------------
# Constructors of the multi-objective acquisition functions
# ---------------------------------------------------------------------------


def _hypercells(partitioning: Any) -> Any:
    return torch.as_tensor(partitioning.get_hypercell_bounds(), dtype=DT)


@pytest.mark.parametrize(
    ("name", "variant"),
    [
        ("ExpectedHypervolumeImprovement", "ref_point"),
        ("ExpectedHypervolumeImprovement", "thresholds"),
        ("qExpectedHypervolumeImprovement", "ref_point"),
        ("qExpectedHypervolumeImprovement", "constraints"),
        ("qLogExpectedHypervolumeImprovement", "objective"),
        ("qExpectedHypervolumeImprovement", "thresholds"),
    ],
)
def test_ehvi_constructors_match_botorch(name: str, variant: str) -> None:
    bt, rs, X, Y = case("m2")
    ref = torch.tensor([0.5, 1.0], dtype=DT)
    kb: dict[str, Any] = {}
    kr: dict[str, Any] = {}
    if variant == "thresholds":
        kb["objective_thresholds"] = kr["objective_thresholds"] = ref
    else:
        kb["ref_point"] = kr["ref_point"] = ref
    if variant == "constraints":
        kb["constraints"] = kr["constraints"] = [lambda Z: Z[..., 1] - 3.0]
        # A temperature that BoTorch's float32 buffers hold exactly.
        kb["eta"] = kr["eta"] = 2.0**-10
    if variant == "objective":
        kb["objective"] = bt_moo.WeightedMCMultiOutputObjective(torch.tensor([1.0, 0.5], dtype=DT))
        kr["objective"] = rs_moo.WeightedMCMultiOutputObjective(torch.tensor([1.0, 0.5], dtype=DT))
    if name != "ExpectedHypervolumeImprovement":
        s_bt, s_rs, _ = sampler_pair(32, 1, 2, 46)
        kb["sampler"], kr["sampler"] = s_bt, s_rs
    ab = bic.get_acqf_input_constructor(_bt_cls(name))(
        model=bt, training_data=datasets(X, Y, "botorch"), **kb
    )
    ar = ric.get_acqf_input_constructor(_rs_cls(name))(
        model=rs, training_data=datasets(X, Y, "botorch_rs"), **kr
    )
    assert set(ab) == set(ar)
    _exact(f"construct_inputs ({name}): ref_point", ar["ref_point"], ab["ref_point"])
    CHECK.check(
        f"construct_inputs ({name}): partitioning",
        _hypercells(ar["partitioning"]),
        _hypercells(ab["partitioning"]),
        VALUE_TOL,
    )
    Xt = points_in(X, (3, 1, 2), 8)
    _compare_acqf(
        f"{name} from its input constructor ({variant})",
        _bt_cls(name)(**ab),
        _rs_cls(name)(**ar),
        Xt,
    )


@pytest.mark.parametrize(
    "name",
    [
        "qNoisyExpectedHypervolumeImprovement",
        "qLogNoisyExpectedHypervolumeImprovement",
        "qLogNParEGO",
    ],
)
def test_noisy_mo_constructors_match_botorch(name: str) -> None:
    bt, rs, X, Y = case("m2")
    kb: dict[str, Any] = {}
    kr: dict[str, Any] = {}
    if name != "qLogNParEGO":
        kb["ref_point"] = kr["ref_point"] = torch.tensor([0.5, 1.0], dtype=DT)
    else:
        kb["scalarization_weights"] = kr["scalarization_weights"] = torch.tensor(
            [0.3, 0.7], dtype=DT
        )
    kb["constraints"] = kr["constraints"] = [lambda Z: Z[..., 1] - 3.0]
    for lib in ("botorch", "botorch_rs"):
        ab = bic.get_acqf_input_constructor(_bt_cls(name))(
            model=bt, training_data=datasets(X, Y, "botorch"), **kb
        )
        ar = ric.get_acqf_input_constructor(_rs_cls(name))(
            model=rs, training_data=datasets(X, Y, lib), **kr
        )
        assert set(ab) == set(ar)
        for key in ab:
            if key in ("model", "objective", "sampler", "constraints"):
                continue
            va, vb = ar[key], ab[key]
            if isinstance(vb, torch.Tensor):
                _exact(f"construct_inputs ({name}): {key}", va, vb)
            else:
                assert va == vb, key
        assert type(ar["objective"]).__name__ == type(ab["objective"]).__name__
        assert (ab["sampler"] is None) == (ar["sampler"] is None)


# ---------------------------------------------------------------------------
# Constructors of the look-ahead and entropy-search acquisition functions
# ---------------------------------------------------------------------------


def test_qmes_constructors_match_botorch() -> None:
    bt, rs, X, Y = case("m1")
    bounds = [(0.0, 1.0), (-0.5, 2.0)]
    for lib in ("botorch", "botorch_rs"):
        torch.manual_seed(11)
        ab = bic.get_acqf_input_constructor(_bt_cls("qMaxValueEntropy"))(
            model=bt, training_data=datasets(X, Y, "botorch"), bounds=bounds, candidate_size=50
        )
        torch.manual_seed(11)
        ar = ric.get_acqf_input_constructor(_rs_cls("qMaxValueEntropy"))(
            model=rs, training_data=datasets(X, Y, lib), bounds=bounds, candidate_size=50
        )
        assert set(ab) == set(ar)
        _exact("construct_inputs_qMES: candidate_set", ar["candidate_set"], ab["candidate_set"])
    tf = {1: 1.0}
    torch.manual_seed(12)
    ab = bic.get_acqf_input_constructor(_bt_cls("qMultiFidelityMaxValueEntropy"))(
        model=bt,
        training_data=datasets(X, Y, "botorch"),
        bounds=bounds,
        target_fidelities=tf,
        candidate_size=40,
        num_trace_observations=2,
        cost_intercept=0.5,
    )
    torch.manual_seed(12)
    ar = ric.get_acqf_input_constructor(_rs_cls("qMultiFidelityMaxValueEntropy"))(
        model=rs,
        training_data=datasets(X, Y, "botorch_rs"),
        bounds=bounds,
        target_fidelities=tf,
        candidate_size=40,
        num_trace_observations=2,
        cost_intercept=0.5,
    )
    assert set(ab) == set(ar)
    _exact("construct_inputs_qMFMES: candidate_set", ar["candidate_set"], ab["candidate_set"])
    Xp = points_in(X, (3, 2, 2), 9)
    _exact("construct_inputs_mf_base: project", ar["project"](Xp), ab["project"](Xp))
    _exact("construct_inputs_mf_base: expand", ar["expand"](Xp), ab["expand"](Xp))
    CHECK.check(
        "construct_inputs_mf_base: cost",
        ar["cost_aware_utility"].cost_model(Xp),
        ab["cost_aware_utility"].cost_model(Xp),
        VALUE_TOL,
    )


def _ics(X: Any, k: int, q: int, seed: int) -> Any:
    return points_in(X, (k, q, X.shape[-1]), seed)


@pytest.mark.parametrize("model", ["m1", "m2"])
def test_qkg_constructor_matches_botorch(model: str) -> None:
    bt, rs, X, Y = case(model)
    bounds = [(0.0, 1.0)] * 2
    kb: dict[str, Any] = {}
    kr: dict[str, Any] = {}
    if model == "m2":
        kb["posterior_transform"] = BtSPT(torch.tensor(W2, dtype=DT))
        kr["posterior_transform"] = RsSPT(torch.tensor(W2, dtype=DT))
    ics = _ics(X, 4, 1, 10)
    opt = {"num_restarts": 4, "raw_samples": 16, "maxiter": 200}
    # BoTorch's registry wrapper rejects the ``optimize_objective_kwargs``
    # (see the draft): its constructor function is called directly.
    ab = bic.construct_inputs_qKG(
        model=bt,
        training_data=datasets(X, Y, "botorch"),
        bounds=bounds,
        num_fantasies=4,
        with_current_value=True,
        batch_initial_conditions=ics,
        optimizer_options=dict(opt),
        **kb,
    )
    ar = ric.get_acqf_input_constructor(_rs_cls("qKnowledgeGradient"))(
        model=rs,
        training_data=datasets(X, Y, "botorch_rs"),
        bounds=bounds,
        num_fantasies=4,
        with_current_value=True,
        batch_initial_conditions=ics,
        optimizer_options=dict(opt),
        **kr,
    )
    assert set(ab) == set(ar)
    CHECK.check(
        "construct_inputs_qKG: current_value",
        ar["current_value"],
        ab["current_value"],
        OPT_VALUE_TOL,
    )
    assert ar["num_fantasies"] == ab["num_fantasies"]


def test_qmfkg_constructor_matches_botorch() -> None:
    bt, rs, X, Y = case("m1_norm")
    bounds = [(0.0, 1.0)] * 3
    tf = {2: 1.0}
    ics = _ics(X, 3, 1, 11)[..., :2]
    opt = {"num_restarts": 3, "raw_samples": 16}
    kw = dict(
        bounds=bounds,
        target_fidelities=tf,
        fidelity_weights={2: 0.5},
        num_fantasies=4,
        batch_initial_conditions=ics,
    )
    ab = bic.construct_inputs_qMFKG(
        model=bt, training_data=datasets(X, Y, "botorch"), optimizer_options=dict(opt), **kw
    )
    ar = ric.get_acqf_input_constructor(_rs_cls("qMultiFidelityKnowledgeGradient"))(
        model=rs, training_data=datasets(X, Y, "botorch_rs"), optimizer_options=dict(opt), **kw
    )
    assert set(ab) == set(ar)
    CHECK.check(
        "construct_inputs_qMFKG: current_value",
        ar["current_value"],
        ab["current_value"],
        OPT_VALUE_TOL,
    )
    Xp = points_in(X, (2, 3, 3), 12)
    _exact("construct_inputs_qMFKG: project", ar["project"](Xp), ab["project"](Xp))
    _exact("construct_inputs_qMFKG: expand", ar["expand"](Xp), ab["expand"](Xp))


@pytest.mark.parametrize("multi_fidelity", [False, True])
def test_qhvkg_constructors_match_botorch(multi_fidelity: bool) -> None:
    bt, rs, X, Y = case("list")
    bounds = [(0.0, 1.0)] * 2
    ref = torch.tensor([0.5, 1.0], dtype=DT)
    num_pareto = 2
    opt = {"num_restarts": 2, "raw_samples": 16}
    kw: dict[str, Any] = dict(bounds=bounds, ref_point=ref, num_fantasies=2, num_pareto=num_pareto)
    if multi_fidelity:
        kw["target_fidelities"] = {1: 1.0}
        kw["batch_initial_conditions"] = _ics(X, 2, num_pareto, 13)[..., :1]
        name = "qMultiFidelityHypervolumeKnowledgeGradient"
    else:
        kw["batch_initial_conditions"] = _ics(X, 2, num_pareto, 13)
        name = "qHypervolumeKnowledgeGradient"
    fb = bic.construct_inputs_qMFHVKG if multi_fidelity else bic.construct_inputs_qHVKG
    ab = fb(model=bt, training_data=datasets(X, Y, "botorch"), optimizer_options=dict(opt), **kw)
    ar = ric.get_acqf_input_constructor(_rs_cls(name))(
        model=rs, training_data=datasets(X, Y, "botorch_rs"), optimizer_options=dict(opt), **kw
    )
    assert set(ab) == set(ar)
    _exact(f"construct_inputs ({name}): ref_point", ar["ref_point"], ab["ref_point"])
    CHECK.check(
        f"construct_inputs ({name}): current_value",
        ar["current_value"],
        ab["current_value"],
        OPT_VALUE_TOL,
    )
    if multi_fidelity:
        Xp = points_in(X, (2, 3, 2), 14)
        _exact(f"construct_inputs ({name}): project", ar["project"](Xp), ab["project"](Xp))


def test_qjes_and_nipv_constructors_match_botorch() -> None:
    bt, rs, X, Y = case("m1")
    bounds = [(0.0, 1.0)] * 2
    draws: list[Any] = []
    cands: list[Any] = []
    torch.manual_seed(15)
    with recorded_optimal_samples(draws, cands):
        ab = bic.get_acqf_input_constructor(_bt_cls("qJointEntropySearch"))(
            model=bt, bounds=bounds, num_optima=4
        )
    orig = rs_autils.get_optimal_samples

    def replayed(*args: Any, **kwargs: Any) -> Any:
        b = kwargs["bounds"]
        return orig(
            *args,
            num_features=draws[0][0].shape[0] * 2,
            draws=draws,
            candidates=b[0] + (b[1] - b[0]) * cands[0],
            **kwargs,
        )

    ric.get_optimal_samples = replayed
    try:
        ar = ric.get_acqf_input_constructor(_rs_cls("qJointEntropySearch"))(
            model=rs, bounds=bounds, num_optima=4
        )
    finally:
        ric.get_optimal_samples = orig
    assert set(ab) == set(ar)
    CHECK.check(
        "construct_inputs_qJES: optimal_inputs", ar["optimal_inputs"], ab["optimal_inputs"], 1e-6
    )
    CHECK.check(
        "construct_inputs_qJES: optimal_outputs", ar["optimal_outputs"], ab["optimal_outputs"], 1e-6
    )
    # NIPV: BoTorch's Sobol points replayed in botorch_rs.
    recorded: list[Any] = []
    orig_bt = bic.draw_sobol_samples

    def record(*args: Any, **kwargs: Any) -> Any:
        out = orig_bt(*args, **kwargs)
        recorded.append(out)
        return out

    bic.draw_sobol_samples = record
    try:
        ab = bic.get_acqf_input_constructor(_bt_cls("qNegIntegratedPosteriorVariance"))(
            model=bt, bounds=bounds, num_mc_points=16
        )
    finally:
        bic.draw_sobol_samples = orig_bt
    from botorch_rs.utils import sampling as rs_sampling

    orig_rs = rs_sampling.draw_sobol_samples
    rs_sampling.draw_sobol_samples = lambda *a, **k: recorded[0]  # type: ignore[assignment]
    try:
        ar = ric.get_acqf_input_constructor(_rs_cls("qNegIntegratedPosteriorVariance"))(
            model=rs, bounds=bounds, num_mc_points=16
        )
    finally:
        rs_sampling.draw_sobol_samples = orig_rs
    _exact("construct_inputs_NIPV: mc_points", ar["mc_points"], ab["mc_points"])


# ---------------------------------------------------------------------------
# Values of the acquisition functions built by the constructors whose
# acquisition functions draw random numbers (with BoTorch's draws installed)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("log", [False, True])
def test_noisy_ei_values_from_constructors(log: bool) -> None:
    from so_acquisition_helpers import recorded_draws

    name = "LogNoisyExpectedImprovement" if log else "NoisyExpectedImprovement"
    bt, rs, X, Y = case("m1_fixed")
    n_f = 8
    ab = bic.get_acqf_input_constructor(_bt_cls(name))(
        model=bt, training_data=datasets(X, Y, "botorch"), num_fantasies=n_f
    )
    ar = ric.get_acqf_input_constructor(_rs_cls(name))(
        model=rs, training_data=datasets(X, Y, "botorch_rs"), num_fantasies=n_f
    )
    draws: list[Any] = []
    with recorded_draws(draws):
        a_bt = _bt_cls(name)(**ab)
    sampler = RsSampler(sample_shape=torch.Size([n_f]), seed=0)
    sampler.base_samples = draws[0].reshape(n_f, -1).clone()
    a_rs = _rs_cls(name)(**ar, sampler=sampler)
    _compare_acqf(f"{name} from its input constructor", a_bt, a_rs, points_in(X, (4, 1, 2), 60))


@pytest.mark.parametrize(
    "name",
    [
        "qNoisyExpectedHypervolumeImprovement",
        "qLogNoisyExpectedHypervolumeImprovement",
        "qLogNParEGO",
    ],
)
def test_noisy_mo_values_from_constructors(name: str) -> None:
    from mo_acquisition_helpers import bt_sampler, synced_draws
    from mo_acquisition_helpers import rs_sampler as mo_rs_sampler

    bt, rs, X, Y = case("m2")
    n, m, q, seed = X.shape[0], 2, 2, 61
    kb: dict[str, Any] = {"prune_baseline": False, "sampler": bt_sampler(32)}
    kr: dict[str, Any] = {"prune_baseline": False, "sampler": mo_rs_sampler(32, seed, (n + q) * m)}
    if name == "qLogNParEGO":
        kb["scalarization_weights"] = kr["scalarization_weights"] = torch.tensor(
            [0.3, 0.7], dtype=DT
        )
        # The base samples of BoTorch's cached-root sampler differ (as in
        # test_parity_mo_acquisition).
        kb["cache_root"] = False
    else:
        kb["ref_point"] = kr["ref_point"] = torch.tensor([0.5, 1.0], dtype=DT)
    with synced_draws(seed):
        ab = bic.get_acqf_input_constructor(_bt_cls(name))(
            model=bt, training_data=datasets(X, Y, "botorch"), **kb
        )
        a_bt = _bt_cls(name)(**ab)
    ar = ric.get_acqf_input_constructor(_rs_cls(name))(
        model=rs, training_data=datasets(X, Y, "botorch_rs"), **kr
    )
    a_rs = _rs_cls(name)(**ar)
    Xt = points_in(X, (3, q, 2), 62)
    with synced_draws(seed):
        vb, gb = values_and_grads(a_bt, Xt)
    vr, gr = values_and_grads(a_rs, Xt)
    CHECK.check(f"{name} from its input constructor: values", vr, vb, VALUE_TOL)
    CHECK.check(f"{name} from its input constructor: gradients", gr, gb, GRAD_TOL, atol=1e-14)


@pytest.mark.parametrize("name", ["qMaxValueEntropy", "qMultiFidelityMaxValueEntropy"])
def test_mes_values_from_constructors(name: str) -> None:
    bt, rs, X, Y = case("m1")
    kw: dict[str, Any] = {"bounds": [(0.0, 1.0)] * 2, "candidate_size": 30}
    if name == "qMultiFidelityMaxValueEntropy":
        kw.update(target_fidelities={1: 1.0}, num_fantasies=4)
    torch.manual_seed(63)
    ab = bic.get_acqf_input_constructor(_bt_cls(name))(
        model=bt, training_data=datasets(X, Y, "botorch"), **kw
    )
    torch.manual_seed(63)
    ar = ric.get_acqf_input_constructor(_rs_cls(name))(
        model=rs, training_data=datasets(X, Y, "botorch_rs"), **kw
    )
    ab = {k: v for k, v in ab.items() if k != "num_fantasies"}
    ar = {k: v for k, v in ar.items() if k != "num_fantasies"}
    torch.manual_seed(1)
    a_bt = _bt_cls(name)(**ab, num_fantasies=4, num_y_samples=16)
    Xt = points_in(X, (4, 1, 2), 64)
    a_bt(Xt[:1])  # draws the observation base samples
    s = RsSampler(sample_shape=torch.Size([16]), seed=0)
    s.base_samples = a_bt.sampler.base_samples.reshape(16, -1).clone()
    a_rs = _rs_cls(name)(**ar, num_fantasies=4, num_y_samples=16, sampler=s, seed=0)
    a_rs.posterior_max_values = a_bt.posterior_max_values.detach()
    _compare_acqf(f"{name} from its input constructor", a_bt, a_rs, Xt)


@pytest.mark.parametrize("name", ["qKnowledgeGradient", "qMultiFidelityKnowledgeGradient"])
def test_kg_values_from_constructors(name: str) -> None:
    bt, rs, X, Y = case("m1_norm")
    d = X.shape[-1]
    n_f, q = 3, 1
    kw: dict[str, Any] = {"bounds": [(0.0, 1.0)] * d, "num_fantasies": n_f}
    opt = {"num_restarts": 2, "raw_samples": 16}
    if name == "qMultiFidelityKnowledgeGradient":
        kw["target_fidelities"] = {d - 1: 1.0}
        kw["batch_initial_conditions"] = _ics(X, 2, 1, 65)[..., : d - 1]
        ab = bic.construct_inputs_qMFKG(
            model=bt, training_data=datasets(X, Y, "botorch"), optimizer_options=dict(opt), **kw
        )
    else:
        ab = bic.construct_inputs_qKG(model=bt, training_data=datasets(X, Y, "botorch"), **kw)
    kr = dict(kw)
    if name == "qMultiFidelityKnowledgeGradient":
        kr["optimizer_options"] = dict(opt)
    ar = ric.get_acqf_input_constructor(_rs_cls(name))(
        model=rs, training_data=datasets(X, Y, "botorch_rs"), **kr
    )
    if "current_value" in ab:
        # The same current value on both sides (compared above).
        ar["current_value"] = ab["current_value"]
    s_bt, s_rs, _ = sampler_pair(n_f, q, 1, 66)
    a_bt = _bt_cls(name)(**{**ab, "num_fantasies": None}, sampler=s_bt)
    a_rs = _rs_cls(name)(**{**ar, "num_fantasies": None}, sampler=s_rs)
    Xt = points_in(X, (2, q + n_f, d), 67)
    if name == "qMultiFidelityKnowledgeGradient":
        Xt[..., -1] = 0.3 + 0.5 * torch.rand(
            Xt.shape[:-1], generator=torch.Generator().manual_seed(68), dtype=DT
        )
    _compare_acqf(f"{name} from its input constructor", a_bt, a_rs, Xt)


@pytest.mark.parametrize("multi_fidelity", [False, True])
def test_hvkg_values_from_constructors(multi_fidelity: bool) -> None:
    from botorch.sampling.list_sampler import ListSampler as BtListSampler

    from botorch_rs.sampling import ListSampler as RsListSampler

    bt, rs, X, Y = case("list")
    name = (
        "qMultiFidelityHypervolumeKnowledgeGradient"
        if multi_fidelity
        else "qHypervolumeKnowledgeGradient"
    )
    n_f, num_pareto = 2, 2
    kw: dict[str, Any] = dict(
        bounds=[(0.0, 1.0)] * 2,
        ref_point=torch.tensor([0.5, 1.0], dtype=DT),
        num_fantasies=n_f,
        num_pareto=num_pareto,
    )
    if multi_fidelity:
        kw["target_fidelities"] = {1: 1.0}
        kw["batch_initial_conditions"] = _ics(X, 2, num_pareto, 69)[..., :1]
    else:
        kw["batch_initial_conditions"] = _ics(X, 2, num_pareto, 69)
    opt = {"num_restarts": 2, "raw_samples": 16}
    fb = bic.construct_inputs_qMFHVKG if multi_fidelity else bic.construct_inputs_qHVKG
    ab = fb(model=bt, training_data=datasets(X, Y, "botorch"), optimizer_options=dict(opt), **kw)
    ar = ric.get_acqf_input_constructor(_rs_cls(name))(
        model=rs, training_data=datasets(X, Y, "botorch_rs"), optimizer_options=dict(opt), **kw
    )
    ar["current_value"] = ab["current_value"]
    bt_samplers = [BtSampler(sample_shape=torch.Size([n_f]), seed=70 + k) for k in range(2)]
    a_bt = _bt_cls(name)(**ab, sampler=BtListSampler(*bt_samplers))
    Xt = points_in(X, (1, 1 + n_f * num_pareto, 2), 71)
    with gpytorch_detach(False):
        vb, gb = values_and_grads(a_bt, Xt)
    samplers = []
    for sb in bt_samplers:
        r = RsSampler(sample_shape=(n_f,), seed=0)
        r.base_samples = sb.base_samples.clone()
        samplers.append(r)
    a_rs = _rs_cls(name)(**ar, sampler=RsListSampler(*samplers))
    vr, gr = values_and_grads(a_rs, Xt)
    CHECK.check(f"{name} from its input constructor: values", vr, vb, VALUE_TOL)
    CHECK.check(f"{name} from its input constructor: gradients", gr, gb, GRAD_TOL, atol=1e-14)


def gpytorch_detach(state: bool) -> Any:
    import gpytorch

    return gpytorch.settings.detach_test_caches(state)


def test_jes_and_nipv_values_from_constructors() -> None:
    from hvkg_helpers import use_fantasy_mean_cache
    from so_acquisition_helpers import no_fast_pred_var

    bt, rs, X, Y = case("m1")
    bounds = [(0.0, 1.0)] * 2
    draws: list[Any] = []
    cands: list[Any] = []
    torch.manual_seed(72)
    with recorded_optimal_samples(draws, cands):
        ab = bic.get_acqf_input_constructor(_bt_cls("qJointEntropySearch"))(
            model=bt, bounds=bounds, num_optima=4
        )
    ar = {**ab, "model": rs}
    Xt = points_in(X, (3, 1, 2), 73)
    a_bt = _bt_cls("qJointEntropySearch")(**ab)
    use_fantasy_mean_cache(a_bt.conditional_model)  # meta-pytorch/botorch#3396
    _compare_acqf(
        "qJointEntropySearch from its input constructor",
        a_bt,
        _rs_cls("qJointEntropySearch")(**ar),
        Xt,
    )
    ab = bic.get_acqf_input_constructor(_bt_cls("qNegIntegratedPosteriorVariance"))(
        model=bt, bounds=bounds, num_mc_points=16
    )
    ar = {**ab, "model": rs}
    a_bt = _bt_cls("qNegIntegratedPosteriorVariance")(**ab)
    a_rs = _rs_cls("qNegIntegratedPosteriorVariance")(**ar)
    # BoTorch's gradients are wrong under fast_pred_var (see the phase-8 draft).
    with no_fast_pred_var():
        _compare_acqf("qNegIntegratedPosteriorVariance from its input constructor", a_bt, a_rs, Xt)


# ---------------------------------------------------------------------------
# optimize_objective and get_acquisition_function
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("variant", ["posterior_mean", "objective_fixed", "linear_constraints"])
def test_optimize_objective_matches_botorch(variant: str) -> None:
    bt, rs, X, _ = case("m2")
    bounds = torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=DT)
    kb: dict[str, Any] = {}
    kr: dict[str, Any] = {}
    ics = _ics(X, 3, 1, 16)
    if variant == "posterior_mean":
        kb["posterior_transform"] = BtSPT(torch.tensor(W2, dtype=DT))
        kr["posterior_transform"] = RsSPT(torch.tensor(W2, dtype=DT))
    elif variant == "objective_fixed":
        w = torch.tensor([1.0, -0.5], dtype=DT)
        kb["objective"], kr["objective"] = BtLinear(w), RsLinear(w)
        kb["fixed_features"] = kr["fixed_features"] = {0: 0.3}
        kb["seed_inner"] = kr["seed_inner"] = 0
        kb["mc_samples"] = kr["mc_samples"] = 16
        ics = ics[..., 1:]
    else:
        kb["posterior_transform"] = BtSPT(torch.tensor(W2, dtype=DT))
        kr["posterior_transform"] = RsSPT(torch.tensor(W2, dtype=DT))
        A = torch.tensor([[1.0, 1.0]], dtype=DT)
        b = torch.tensor([[1.0]], dtype=DT)
        kb["linear_constraints"] = kr["linear_constraints"] = (A, b)
        ics = ics * 0.4
    opt = {"num_restarts": 3, "raw_samples": 16}
    if variant == "objective_fixed":
        # The same base samples of the inner qSimpleRegret.
        s_bt, s_rs, Z = sampler_pair(16, 1, 2, 47)
        orig_b, orig_r = bic.SobolQMCNormalSampler, ric.SobolQMCNormalSampler
        bic.SobolQMCNormalSampler = lambda *a, **k: s_bt  # type: ignore[assignment]
        ric.SobolQMCNormalSampler = lambda *a, **k: s_rs  # type: ignore[assignment]
    try:
        xb, vb = bic.optimize_objective(
            bt, bounds, 1, batch_initial_conditions=ics, optimizer_options=dict(opt), **kb
        )
        xr, vr = ric.optimize_objective(
            rs, bounds, 1, batch_initial_conditions=ics, optimizer_options=dict(opt), **kr
        )
    finally:
        if variant == "objective_fixed":
            bic.SobolQMCNormalSampler, ric.SobolQMCNormalSampler = orig_b, orig_r
    CHECK.check(f"optimize_objective ({variant}): value", vr, vb, OPT_VALUE_TOL)
    CHECK.check(f"optimize_objective ({variant}): candidates", xr, xb, OPT_X_TOL)


@pytest.mark.parametrize(
    "name", ["qEI", "qLogEI", "qPI", "qNEI", "qLogNEI", "qSR", "qUCB", "qLogPF", "pTS"]
)
def test_get_acquisition_function_single_objective_matches_botorch(name: str) -> None:
    bt, rs, X, _ = case("m2")
    w = torch.tensor([1.0, -0.5], dtype=DT)
    obj_b, obj_r = BtLinear(w), RsLinear(w)
    cons = [lambda Z: Z[..., 1] - 3.0] if name not in ("qSR", "qUCB", "pTS") else None
    kw: dict[str, Any] = {"beta": 0.3} if name == "qUCB" else {}
    if name in ("qNEI", "qLogNEI"):
        kw["prune_baseline"] = False
        kw["cache_root"] = False
    # A temperature that BoTorch's float32 buffers hold exactly.
    kw["eta"] = 2.0**-10
    if name == "qPI":
        kw["tau"] = 2.0**-10
    a_bt = bt_factory.get_acquisition_function(
        name, bt, obj_b, X, constraints=cons, mc_samples=16, seed=0, **kw
    )
    a_rs = rs_factory.get_acquisition_function(
        name, rs, obj_r, X, constraints=cons, mc_samples=16, seed=0, **kw
    )
    assert type(a_rs).__name__ == type(a_bt).__name__
    if name in ("qEI", "qLogEI", "qPI"):
        # BoTorch's qPI keeps ``best_f`` with a trailing dimension.
        CHECK.check(
            f"get_acquisition_function({name}): best_f",
            torch.as_tensor(a_rs.best_f).reshape(-1),
            a_bt.best_f.reshape(-1),
            VALUE_TOL,
        )
    if name == "pTS":
        return
    if name in ("qNEI", "qLogNEI"):
        # The baseline and options (the values with identical base samples
        # are compared through the input constructors above).
        _exact(f"get_acquisition_function({name}): X_baseline", a_rs.X_baseline, a_bt.X_baseline)
        return
    # Identical base samples.
    q = 2
    s_bt, s_rs, _ = sampler_pair(16, q, 2, 48)
    a_bt.sampler = s_bt
    a_rs = rebuild_with_sampler(a_rs, s_rs)
    _compare_acqf(f"get_acquisition_function({name})", a_bt, a_rs, points_in(X, (3, q, 2), 17))


def rebuild_with_sampler(acqf: Any, sampler: Any) -> Any:
    """The botorch_rs acquisition function rebuilt with ``sampler``."""
    import inspect

    cls = type(acqf)
    params = inspect.signature(cls.__init__).parameters
    kwargs: dict[str, Any] = {"sampler": sampler}
    for name in params:
        if name in ("self", "sampler", "kwargs"):
            continue
        attr = {
            "objective": "objective",
            "posterior_transform": "posterior_transform",
            "best_f": "best_f",
            "beta": "beta",
            "tau": "tau",
            "constraints": "_constraints",
            "eta": "_eta",
            "X_pending": "X_pending",
        }.get(name)
        if attr is not None and hasattr(acqf, attr):
            kwargs[name] = getattr(acqf, attr)
    return cls(acqf.model, **kwargs)


@pytest.mark.parametrize("name", ["qEHVI", "qLogEHVI", "qNEHVI", "qLogNEHVI"])
def test_get_acquisition_function_multi_objective_matches_botorch(name: str) -> None:
    bt, rs, X, Y = case("m2")
    ref = [0.5, 1.0]
    cons = [lambda Z: Z[..., 1] - 3.0]
    ob, orr = bt_moo.IdentityMCMultiOutputObjective(), rs_moo.IdentityMCMultiOutputObjective()
    kw: dict[str, Any] = {"ref_point": ref, "Y": Y, "constraints": cons}
    if name in ("qNEHVI", "qLogNEHVI"):
        kw.pop("Y")
        kw["prune_baseline"] = False
    a_bt = bt_factory.get_acquisition_function(name, bt, ob, X, mc_samples=16, **kw)
    a_rs = rs_factory.get_acquisition_function(name, rs, orr, X, mc_samples=16, **kw)
    assert type(a_rs).__name__ == type(a_bt).__name__
    if hasattr(a_bt, "partitioning") and hasattr(a_rs, "partitioning"):
        CHECK.check(
            f"get_acquisition_function({name}): partitioning",
            _hypercells(a_rs.partitioning),
            _hypercells(a_bt.partitioning),
            VALUE_TOL,
        )


# ---------------------------------------------------------------------------
# qMultiStepLookahead
# ---------------------------------------------------------------------------

MSL_CONFIGS = {
    "pm_2step": dict(batch_sizes=[1, 1], num_fantasies=[3, 2], q=1, t=3),
    "pm_3step": dict(batch_sizes=[1, 1, 1], num_fantasies=[2, 2, 2], q=1, t=2),
    "q2": dict(batch_sizes=[2, 1], num_fantasies=[2, 2], q=2, t=1),
    "qei": dict(
        batch_sizes=[1, 1],
        num_fantasies=[2, 3],
        q=2,
        t=2,
        valfunc=["qExpectedImprovement"] * 3,
        argfac=["best_f"] * 3,
        inner=[4, 4, 4],
    ),
    "ei_stage": dict(
        batch_sizes=[1, 1],
        num_fantasies=[2, 2],
        q=1,
        t=2,
        valfunc=[None, "ExpectedImprovement", "PosteriorMean"],
        argfac=[None, "best_f_analytic", None],
    ),
    "pending": dict(batch_sizes=[1], num_fantasies=[3], q=1, t=2, pending=2),
    "collapse_false": dict(batch_sizes=[1, 1], num_fantasies=[2, 2], q=1, t=3, collapse=False),
    "weights": dict(batch_sizes=[1, 1], num_fantasies=[2, 3], q=1, t=2, weights=True),
}


def _msl_pair(
    model: str, cfg: dict, seed: int, botorch_classes: bool = False
) -> tuple[Any, Any, Any]:
    """BoTorch's and botorch_rs's look-ahead with the same base samples and
    candidate sets ``X``."""
    bt, rs, X, _ = case(model)
    d = X.shape[-1]
    kb: dict[str, Any] = {"batch_sizes": cfg["batch_sizes"]}
    kr: dict[str, Any] = {"batch_sizes": cfg["batch_sizes"]}
    torch.manual_seed(seed)
    kb["samplers"] = [BtSampler(sample_shape=torch.Size([f])) for f in cfg["num_fantasies"]]
    kr["samplers"] = [RsSampler(sample_shape=(f,)) for f in cfg["num_fantasies"]]
    if cfg.get("weights"):
        for sb, sr, f in zip(kb["samplers"], kr["samplers"], cfg["num_fantasies"], strict=True):
            sb.base_weights = sr.base_weights = torch.linspace(1.0, 2.0, f, dtype=DT).reshape(f, 1)
    if cfg.get("valfunc"):
        kb["valfunc_cls"] = [None if v is None else getattr(ba, v) for v in cfg["valfunc"]]
        kr["valfunc_cls"] = [
            None if v is None else getattr(ba if botorch_classes else ra, v) for v in cfg["valfunc"]
        ]
        kb["valfunc_argfacs"] = [
            None if a is None else best_f_argfac(a == "best_f_analytic") for a in cfg["argfac"]
        ]
        maker = bt_msl.make_best_f if botorch_classes else rs_msl.make_best_f
        kr["valfunc_argfacs"] = [None if a is None else maker for a in cfg["argfac"]]
    if cfg.get("inner"):
        kb["inner_mc_samples"] = kr["inner_mc_samples"] = cfg["inner"]
    if cfg.get("pending"):
        p = points_in(X, (cfg["pending"], d), seed + 1)
        kb["X_pending"] = kr["X_pending"] = p
    if model in ("m2", "list"):
        kb["posterior_transform"] = BtSPT(torch.tensor(W2, dtype=DT))
        kr["posterior_transform"] = RsSPT(torch.tensor(W2, dtype=DT))
    collapse = cfg.get("collapse", True)
    kb["collapse_fantasy_base_samples"] = kr["collapse_fantasy_base_samples"] = collapse
    a_bt = bt_msl.qMultiStepLookahead(bt, **kb)
    a_rs = rs_msl.qMultiStepLookahead(rs, **kr)
    q_aug = a_bt.get_augmented_q_batch_size(cfg["q"])
    assert a_rs.get_augmented_q_batch_size(cfg["q"]) == q_aug
    Xt = points_in(X, (cfg["t"], q_aug, d), seed + 2)
    with single_output_collapse_fix(), torch.no_grad():
        a_bt(Xt)  # draws BoTorch's base samples
    copy_lookahead_samplers(a_bt, a_rs)
    return a_bt, a_rs, Xt


@pytest.mark.parametrize("config", sorted(MSL_CONFIGS))
@pytest.mark.parametrize("model", ["m1", "m1_norm", "m1_fixed", "m1_nostd", "m2", "list"])
def test_multi_step_lookahead_matches_botorch(config: str, model: str) -> None:
    cfg = MSL_CONFIGS[config]
    if model in ("m2", "list") and cfg.get("argfac"):
        pytest.skip("best_f from the model requires a single-output model")
    if model in ("m2", "list") and config == "q2":
        # BoTorch fails for a single t-batch of multi-output fantasies (as its
        # qKG, see the phase-8 draft), and several t-batches mix the blocks of
        # q_j > 1 points.
        pytest.skip("BoTorch fails for one t-batch of multi-output fantasies")
    a_bt, a_rs, Xt = _msl_pair(model, cfg, 100 + len(config))
    # Values against BoTorch, gradients against the corrected BoTorch
    # (BoTorch's own are wrong, see the draft).
    with single_output_collapse_fix():
        with torch.no_grad():
            vb = a_bt(Xt)
        with corrected_lookahead():
            _, gb = values_and_grads(a_bt, Xt)
    vr, gr = values_and_grads(a_rs, Xt)
    label = f"qMultiStepLookahead {config}"
    CHECK.check(f"{label}: values", vr, vb, VALUE_TOL)
    CHECK.check(f"{label}: gradients (corrected BoTorch)", gr, gb, GRAD_TOL, atol=1e-14)


def test_multi_step_lookahead_with_botorch_value_functions() -> None:
    a_bt, a_rs, Xt = _msl_pair("m1_nostd", MSL_CONFIGS["qei"], 120, botorch_classes=True)
    with torch.no_grad():
        vb = a_bt(Xt)
    with corrected_lookahead():
        _, gb = values_and_grads(a_bt, Xt)
    vr, gr = values_and_grads(a_rs, Xt)
    CHECK.check("qMultiStepLookahead (BoTorch's classes): values", vr, vb, VALUE_TOL)
    CHECK.check("qMultiStepLookahead (BoTorch's classes): gradients", gr, gb, GRAD_TOL, atol=1e-14)


def test_multi_step_lookahead_botorch_gradients_are_wrong() -> None:
    """BoTorch's own gradients disagree with central differences of its
    values; botorch_rs's agree with them."""
    a_bt, a_rs, Xt = _msl_pair("m1", MSL_CONFIGS["pm_2step"], 130)
    _, g_bt = values_and_grads(a_bt, Xt)
    _, g_rs = values_and_grads(a_rs, Xt)
    h = 1e-6
    fd = torch.zeros_like(Xt)
    for k in range(Xt.numel()):
        e = torch.zeros_like(Xt).view(-1)
        e[k] = h
        e = e.view_as(Xt)
        with torch.no_grad():
            fd.view(-1)[k] = (a_bt(Xt + e).sum() - a_bt(Xt - e).sum()).item() / (2 * h)
    err_bt = (g_bt - fd).abs().max().item() / fd.abs().max().item()
    CHECK.check("qMultiStepLookahead: gradients vs finite differences of BoTorch", g_rs, fd, 1e-7)
    assert err_bt > 1e-2, err_bt


def test_lookahead_with_one_fantasy_matches_a_duplicated_fantasy() -> None:
    """One fantasy at the first step followed by two (BoTorch fails: see the
    draft): compared with BoTorch's tree with the first fantasy duplicated
    (two identical base samples and subtrees), whose value is the same and
    whose gradients of the duplicated points add up."""
    bt, rs, X, _ = case("m1")
    d = 2
    torch.manual_seed(171)
    z0 = torch.randn(1, 1, 1, dtype=DT)
    z1 = torch.randn(2, 1, 1, dtype=DT)
    s_rs = [RsSampler(sample_shape=(1,)), RsSampler(sample_shape=(2,))]
    s_rs[0]._rs.set_base_samples((z0.reshape(-1).tolist(), (1, 1)))
    s_rs[1]._rs.set_base_samples((z1.reshape(-1).tolist(), (2, 1)))
    s_bt = [BtSampler(sample_shape=torch.Size([2])), BtSampler(sample_shape=torch.Size([2]))]
    s_bt[0].base_samples = torch.cat([z0, z0])
    s_bt[1].base_samples = z1.reshape(2, 1, 1, 1).clone()
    a_rs = rs_msl.qMultiStepLookahead(rs, batch_sizes=[1, 1], samplers=s_rs)
    a_bt = bt_msl.qMultiStepLookahead(bt, batch_sizes=[1, 1], samplers=s_bt)
    g = torch.Generator().manual_seed(172)
    x0 = torch.rand(2, 1, d, generator=g, dtype=DT)
    x1 = torch.rand(2, 1, d, generator=g, dtype=DT)
    x2 = torch.rand(2, 2, d, generator=g, dtype=DT)  # node (0, i2) at row i2
    X_rs = torch.cat([x0, x1, x2], dim=-2)
    # BoTorch's rows: x1 twice, then node (i1, i2) at row i1 + 2 i2.
    X_bt = torch.cat([x0, x1, x1, x2[:, :1], x2[:, :1], x2[:, 1:], x2[:, 1:]], dim=-2)
    vr, gr = values_and_grads(a_rs, X_rs)
    with corrected_lookahead():
        vb, gb = values_and_grads(a_bt, X_bt)
    CHECK.check("qMultiStepLookahead one fantasy: values", vr, vb, VALUE_TOL)
    g_ref = torch.cat(
        [
            gb[:, :1],
            gb[:, 1:2] + gb[:, 2:3],
            gb[:, 3:4] + gb[:, 4:5],
            gb[:, 5:6] + gb[:, 6:7],
        ],
        dim=-2,
    )
    CHECK.check("qMultiStepLookahead one fantasy: gradients", gr, g_ref, GRAD_TOL, atol=1e-14)
    # BoTorch itself fails for this tree.
    a_bt1 = bt_msl.qMultiStepLookahead(bt, batch_sizes=[1, 1], num_fantasies=[1, 2])
    with pytest.raises(RuntimeError):
        a_bt1(X_rs)


def test_lookahead_tree_representation_matches_botorch() -> None:
    for config, model in (("pm_3step", "m1"), ("q2", "m1"), ("pending", "m1")):
        cfg = MSL_CONFIGS[config]
        a_bt, a_rs, Xt = _msl_pair(model, cfg, 140)
        # BoTorch's representation (one t-batch when q_j > 1).
        sb = a_bt.get_split_shapes(Xt)
        sr = a_rs.get_split_shapes(Xt)
        assert tuple(sb[0]) == tuple(sr[0])
        assert [tuple(s) for s in sb[1]] == [tuple(s) for s in sr[1]]
        assert list(sb[2]) == list(sr[2])
        for xb, xr in zip(
            a_bt.get_multi_step_tree_input_representation(Xt),
            a_rs.get_multi_step_tree_input_representation(Xt),
            strict=True,
        ):
            _exact("get_multi_step_tree_input_representation", xr, xb)
        _exact("extract_candidates", a_rs.extract_candidates(Xt), a_bt.extract_candidates(Xt))
        assert a_rs._num_auxiliary == a_bt._num_auxiliary


def test_make_best_f_matches_botorch_without_outcome_transform() -> None:
    bt, rs, X, _ = case("m1_nostd")
    _exact("make_best_f", rs_msl.make_best_f(rs, X)["best_f"], bt_msl.make_best_f(bt, X)["best_f"])
    # With an outcome transform BoTorch returns the standardized maximum.
    bt, rs, X, Y = case("m1")
    _exact("make_best_f (untransformed)", rs_msl.make_best_f(rs, X)["best_f"], Y.max())


def test_warmstart_multistep_matches_botorch() -> None:
    a_bt, a_rs, Xt = _msl_pair("m1", MSL_CONFIGS["pm_2step"], 150)
    bounds = torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=DT)
    full = Xt[:2]
    torch.manual_seed(5)
    eb = bt_msl.warmstart_multistep(
        a_bt, bounds, num_restarts=3, raw_samples=8, full_optimizer=full
    )
    torch.manual_seed(5)
    er = rs_msl.warmstart_multistep(
        a_rs, bounds, num_restarts=3, raw_samples=8, full_optimizer=full
    )
    _exact("warmstart_multistep", er, eb)


def test_optimize_acqf_of_lookahead_matches_corrected_botorch() -> None:
    a_bt, a_rs, Xt = _msl_pair("m1", MSL_CONFIGS["pm_2step"], 160)
    bounds = torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=DT)
    q_aug = Xt.shape[-2]
    ics = torch.rand(3, q_aug, 2, generator=torch.Generator().manual_seed(161), dtype=DT)
    with corrected_lookahead():
        xb, vb = bt_optimize.optimize_acqf(
            a_bt, bounds, q=q_aug, num_restarts=3, batch_initial_conditions=ics
        )
    xr, vr = rs_optimize.optimize_acqf(
        a_rs, bounds, q=q_aug, num_restarts=3, batch_initial_conditions=ics
    )
    # BoTorch's convention: the augmented size is passed, the candidates are
    # returned.
    assert xr.shape == xb.shape == (1, 2)
    # The full trees of every restart.
    with corrected_lookahead():
        fb, vbs = bt_optimize.optimize_acqf(
            a_bt,
            bounds,
            q=q_aug,
            num_restarts=3,
            batch_initial_conditions=ics,
            return_best_only=False,
            return_full_tree=True,
        )
    fr, vrs = rs_optimize.optimize_acqf(
        a_rs,
        bounds,
        q=q_aug,
        num_restarts=3,
        batch_initial_conditions=ics,
        return_best_only=False,
        return_full_tree=True,
    )
    assert fr.shape == fb.shape == (3, q_aug, 2)
    _compare_restarts("optimize_acqf qMultiStepLookahead", a_bt, fr, vrs, fb, vbs)


def test_induced_fantasy_model_matches_botorch() -> None:
    a_bt, a_rs, Xt = _msl_pair("m1", MSL_CONFIGS["pm_2step"], 170)
    fb = a_bt.get_induced_fantasy_model(Xt)
    fr = a_rs.get_induced_fantasy_model(Xt)
    Xs_b = a_bt.get_multi_step_tree_input_representation(Xt)
    pb = fb.posterior(Xs_b[-1])
    pr = fr.posterior(Xs_b[-1])
    CHECK.check("get_induced_fantasy_model: posterior mean", pr.mean, pb.mean, VALUE_TOL)
    CHECK.check(
        "get_induced_fantasy_model: posterior variance", pr.variance, pb.variance, VALUE_TOL
    )


# ---------------------------------------------------------------------------
# Python subclasses of the sample-reducing acquisition functions
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "variant",
    [
        "plain",
        "objective",
        "botorch_objective",
        "constraints",
        "pending",
        "transform",
        "reductions",
        "fat_list",
    ],
)
@pytest.mark.parametrize("model", ["m1", "m2", "list"])
def test_sample_reducing_subclass_matches_botorch(variant: str, model: str) -> None:
    bt, rs, X, _ = case(model)
    m = 1 if model == "m1" else 2
    kb: dict[str, Any] = {}
    kr: dict[str, Any] = {}
    w = torch.tensor([1.0, -0.5], dtype=DT)
    if m == 2:
        if variant == "transform":
            kb["posterior_transform"] = BtSPT(torch.tensor(W2, dtype=DT))
            kr["posterior_transform"] = RsSPT(torch.tensor(W2, dtype=DT))
        elif variant == "botorch_objective":
            kb["objective"] = kr["objective"] = BtGeneric(lambda Y, X=None: Y[..., 0] * Y[..., 1])
        else:
            kb["objective"] = BtLinear(w)
            kr["objective"] = RsLinear(w)
    elif variant == "objective":
        kb["objective"] = BtGeneric(lambda Y, X=None: Y[..., 0].sin())
        kr["objective"] = RsGeneric(lambda Y, X=None: Y[..., 0].sin())
    if variant in ("constraints", "fat_list"):
        kb["constraints"] = kr["constraints"] = [
            lambda Z: Z[..., -1] - 1.5,
            lambda Z: 0.2 - Z[..., 0],
        ]
        kb["eta"] = kr["eta"] = torch.tensor([2.0**-6, 2.0**-4], dtype=DT)
        if variant == "fat_list":
            kb["fat"] = kr["fat"] = [True, False]
    n_pend = 1 if variant == "pending" else 0
    if n_pend:
        kb["X_pending"] = kr["X_pending"] = points_in(X, (n_pend, X.shape[-1]), 181)
    if variant == "reductions":
        kb["sample_reduction"] = kr["sample_reduction"] = torch.amax
        kb["q_reduction"] = kr["q_reduction"] = torch.mean
    q = 2
    m_s = 1 if variant == "transform" else m
    s_bt, s_rs, _ = sampler_pair(16, q + n_pend, m_s, 182)
    Imp_b = improvement_subclass(bt_mc.SampleReducingMCAcquisitionFunction)
    Imp_r = improvement_subclass(rs_mc.SampleReducingMCAcquisitionFunction)
    a_bt = Imp_b(bt, 0.4, sampler=s_bt, **kb)
    a_rs = Imp_r(rs, 0.4, sampler=s_rs, **kr)
    Xt = points_in(X, (3, q, X.shape[-1]), 183)
    _compare_acqf(f"SampleReducingMCAcquisitionFunction subclass ({variant})", a_bt, a_rs, Xt)
    if variant not in ("constraints", "fat_list"):
        Bnd_b = bounded_subclass(bt_mc.SampleReducingMCAcquisitionFunction)
        Bnd_r = bounded_subclass(rs_mc.SampleReducingMCAcquisitionFunction)
        _compare_acqf(
            f"SampleReducingMCAcquisitionFunction signed subclass ({variant})",
            Bnd_b(bt, sampler=s_bt, **kb),
            Bnd_r(rs, sampler=s_rs, **kr),
            Xt,
        )
    if variant != "reductions":
        Log_b = log_improvement_subclass(bt_logei.LogImprovementMCAcquisitionFunction)
        Log_r = log_improvement_subclass(rs_logei.LogImprovementMCAcquisitionFunction)
        kbl = {k: v for k, v in kb.items() if k not in ("sample_reduction", "q_reduction")}
        krl = {k: v for k, v in kr.items() if k not in ("sample_reduction", "q_reduction")}
        _compare_acqf(
            f"LogImprovementMCAcquisitionFunction subclass ({variant})",
            Log_b(bt, 0.4, sampler=s_bt, **kbl),
            Log_r(rs, 0.4, sampler=s_rs, **krl),
            Xt,
        )


def test_sample_reducing_subclass_optimize_acqf_matches_botorch() -> None:
    bt, rs, X, _ = case("m1")
    s_bt, s_rs, _ = sampler_pair(16, 2, 1, 190)
    Imp_b = improvement_subclass(bt_mc.SampleReducingMCAcquisitionFunction)
    Imp_r = improvement_subclass(rs_mc.SampleReducingMCAcquisitionFunction)
    a_bt = Imp_b(bt, 0.0, sampler=s_bt)
    a_rs = Imp_r(rs, 0.0, sampler=s_rs)
    bounds = torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=DT)
    ics = points_in(X, (3, 2, 2), 191)
    xb, vb = bt_optimize.optimize_acqf(
        a_bt, bounds, q=2, num_restarts=3, batch_initial_conditions=ics, return_best_only=False
    )
    xr, vr = rs_optimize.optimize_acqf(
        a_rs, bounds, q=2, num_restarts=3, batch_initial_conditions=ics, return_best_only=False
    )
    _compare_restarts("optimize_acqf SampleReducing subclass", a_bt, xr, vr, xb, vb)


# ---------------------------------------------------------------------------
# t-batched fixed values and batched candidate sets of the max-value samplers
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("form", ["tensor", "q1", "list"])
def test_fixed_feature_with_t_batched_values_matches_botorch(form: str) -> None:
    bt, rs, X, _ = case("m1_norm")
    s_bt, s_rs, _ = sampler_pair(16, 2, 1, 200)
    # A best_f that BoTorch's float32 buffer holds exactly.
    base_b = bt_mc.qExpectedImprovement(bt, best_f=0.25, sampler=s_bt)
    base_r = rs_mc.qExpectedImprovement(rs, best_f=0.25, sampler=s_rs)
    g = torch.Generator().manual_seed(201)
    b = 3
    if form == "tensor":
        values: Any = torch.rand(b, 2, 2, generator=g, dtype=DT)
        cols = [0, 2]
    elif form == "q1":
        values = torch.rand(b, 1, 1, generator=g, dtype=DT)
        cols = [1]
    else:
        values = [0.3, torch.rand(b, 2, 1, generator=g, dtype=DT)]
        cols = [0, 2]
    d_free = 3 - len(cols)
    a_bt = bt_ff.FixedFeatureAcquisitionFunction(base_b, 3, cols, values)
    a_rs = rs_ff.FixedFeatureAcquisitionFunction(base_r, 3, cols, values)
    Xt = torch.rand(b, 2, d_free, generator=g, dtype=DT)
    _exact(
        f"FixedFeature t-batched values ({form}): X_full",
        a_rs._construct_X_full(Xt),
        a_bt._construct_X_full(Xt),
    )
    _compare_acqf(f"FixedFeatureAcquisitionFunction t-batched values ({form})", a_bt, a_rs, Xt)


@pytest.mark.parametrize("batch", [(3,), (2, 3)])
def test_max_value_samplers_with_batched_candidate_sets_match_botorch(batch: tuple) -> None:
    bt, rs, X, _ = case("m1")
    g = torch.Generator().manual_seed(210)
    cs = torch.rand(*batch, 6, 2, generator=g, dtype=DT)
    torch.manual_seed(3)
    try:
        eb = bt_mes._sample_max_value_Gumbel(bt, cs, 5)
    except RuntimeError:
        # BoTorch 0.18.1 fails for several batch dimensions
        # (meta-pytorch/botorch#3353).
        eb = None
    torch.manual_seed(3)
    u = torch.rand(5, 1, dtype=DT)
    er = rs_mes._sample_max_value_Gumbel(rs, cs, 5, uniforms=u)
    assert er.shape == (5, *batch)
    if eb is not None:
        assert eb.shape == er.shape
        # BoTorch rounds its quantiles to float32 (see the phase-8 draft).
        CHECK.check(
            "_sample_max_value_Gumbel (batched) vs BoTorch (float32 quantiles)", er, eb, 1e-6
        )
    # Every batch as one unbatched call.
    flat = cs.reshape(-1, 6, 2)
    loop = torch.stack(
        [rs_mes._sample_max_value_Gumbel(rs, c, 5, uniforms=u)[:, 0] for c in flat], dim=-1
    ).reshape(5, *batch)
    _exact("_sample_max_value_Gumbel (batched) vs one batch at a time", er, loop)
    torch.manual_seed(4)
    eb = bt_mes._sample_max_value_Thompson(bt, cs, 5)
    torch.manual_seed(4)
    # GPyTorch draws the ``batch x n x num_samples`` normal numbers of its
    # Cholesky roots.
    z = torch.randn(*batch, 6, 5, dtype=DT).movedim(-1, 0)
    er = rs_mes._sample_max_value_Thompson(rs, cs, 5, base_samples=z)
    assert er.shape == eb.shape == (5, *batch)
    CHECK.check("_sample_max_value_Thompson (batched)", er, eb, VALUE_TOL)


def test_unbatched_max_value_samplers_keep_their_shape() -> None:
    bt, rs, X, _ = case("m1")
    cs = points_in(X, (6, 2), 220)
    torch.manual_seed(5)
    eb = bt_mes._sample_max_value_Gumbel(bt, cs, 4)
    torch.manual_seed(5)
    er = rs_mes._sample_max_value_Gumbel(rs, cs, 4, uniforms=torch.rand(4, 1, dtype=DT))
    assert er.shape == eb.shape == (4, 1)

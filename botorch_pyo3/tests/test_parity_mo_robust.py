"""Parity of the robust and multi-fidelity multi-objective ports with BoTorch,
through the public APIs of both libraries side by side:

* ``InputPerturbation`` and ``ChainedInputTransform`` (fixed and
  heteroscedastic perturbation sets, bounds, ``indices``, multiplicative
  perturbations, ``Normalize`` before or after the perturbation): the
  transformed points; ``SingleTaskGP`` and ``ModelListGP`` models with these
  transforms: posterior means, variances and covariances over the ``q * n_w``
  points (with and without observation noise) and posterior samples from
  identical base samples;
* the risk measures (``CVaR``, ``VaR``, ``WorstCase``, ``Expectation``) and
  the multi-output risk measures (``MultiOutputExpectation``,
  ``IndependentCVaR``, ``IndependentVaR``, ``MultiOutputWorstCase``, ``MVaR``
  with all options, ``MARS``) on random samples, with and without
  preprocessing functions: values and gradients (``MVaR(expectation=True)``:
  values only, BoTorch's gradient is NaN), MVaR sets, MARS's baseline and
  normalization bounds;
* qEHVI, qLogEHVI, qNEHVI and qLogNEHVI with multi-output risk measures
  (including BoTorch's own objects), qLogNEI, qLogEI and qSimpleRegret with
  risk measures and MARS, on models with an ``InputPerturbation``: values and
  gradients with identical base samples, every candidate set evaluated on
  its own (BoTorch pads MVaR sets to the largest set of a whole t-batch);
* MOMF (default and callable costs, pending points, constraints): values
  and gradients;
* NSGA-II: pymoo's non-dominated sorting and crowding distance, and
  ``optimize_with_nsgaii`` against BoTorch + pymoo statistically (the mean
  hypervolume of the returned fronts over seeds); skipped without pymoo.

Tolerances: values 1e-9 (relative; log values absolute below 1), gradients
1e-8 (relative to the largest component). The joint posteriors of the noisy
acquisition functions over their (perturbed) baseline are nearly singular;
their tolerance is 10x BoTorch's own change when its training inputs are
perturbed by 1e-14 (``parity_helpers.perturbed_copy``) when that is larger.
The worst errors are listed in the terminal summary (section "robust
multi-objective"). Skipped unless BoTorch is importable (required in CI's
parity job); the randomized sweep is marked ``equivalence``.
"""

from __future__ import annotations

import importlib.util
import math
import random
from collections.abc import Callable, Iterator
from typing import Any

import pytest

from helpers import importorskip_botorch

importorskip_botorch()
torch = pytest.importorskip("torch")

from botorch.acquisition import logei as bt_logei_so  # noqa: E402
from botorch.acquisition import monte_carlo as bt_mc_so  # noqa: E402
from botorch.acquisition import objective as bt_obj  # noqa: E402
from botorch.acquisition import risk_measures as bt_risk  # noqa: E402
from botorch.acquisition.multi_objective import logei as bt_logei  # noqa: E402
from botorch.acquisition.multi_objective import monte_carlo as bt_mc  # noqa: E402
from botorch.acquisition.multi_objective import multi_fidelity as bt_mf  # noqa: E402
from botorch.acquisition.multi_objective import (  # noqa: E402
    multi_output_risk_measures as bt_morisk,
)
from botorch.acquisition.multi_objective import objective as bt_mo_obj  # noqa: E402
from botorch.models import ModelListGP as BtModelListGP  # noqa: E402
from botorch.models import SingleTaskGP as BtSingleTaskGP  # noqa: E402
from botorch.models.transforms import input as bt_input  # noqa: E402
from botorch.utils.multi_objective import box_decompositions as bt_bd  # noqa: E402
from mo_acquisition_helpers import DT, bt_sampler, rs_sampler, synced_draws, zdraw  # noqa: E402

from botorch_rs.acquisition import logei as rs_logei_so  # noqa: E402
from botorch_rs.acquisition import monte_carlo as rs_mc_so  # noqa: E402
from botorch_rs.acquisition import objective as rs_obj  # noqa: E402
from botorch_rs.acquisition import risk_measures as rs_risk  # noqa: E402
from botorch_rs.acquisition.multi_objective import logei as rs_logei  # noqa: E402
from botorch_rs.acquisition.multi_objective import monte_carlo as rs_mc  # noqa: E402
from botorch_rs.acquisition.multi_objective import multi_fidelity as rs_mf  # noqa: E402
from botorch_rs.acquisition.multi_objective import (  # noqa: E402
    multi_output_risk_measures as rs_morisk,
)
from botorch_rs.acquisition.multi_objective import objective as rs_mo_obj  # noqa: E402
from botorch_rs.acquisition.multioutput_acquisition import (  # noqa: E402
    MultiOutputPosteriorMean as RsMOPosteriorMean,
)
from botorch_rs.models import ModelListGP as RsModelListGP  # noqa: E402
from botorch_rs.models import SingleTaskGP as RsSingleTaskGP  # noqa: E402
from botorch_rs.models.transforms import input as rs_input  # noqa: E402
from botorch_rs.utils.multi_objective import box_decompositions as rs_bd  # noqa: E402
from botorch_rs.utils.multi_objective.hypervolume import Hypervolume  # noqa: E402
from botorch_rs.utils.multi_objective.optimize import (  # noqa: E402
    optimize_with_nsgaii as rs_optimize_with_nsgaii,
)
from parity_helpers import (  # noqa: E402
    Checker,
    ModelSpec,
    copy_hyperparameters,
    dense_model_list,
    log_err,
    make_data,
    perturbed_copy,
    rel_err,
    set_hyperparameters,
    values_and_grads,
)

CHECK = Checker("robust multi-objective")
VALUE_TOL = 1e-9
GRAD_TOL = 1e-8
NUM_SAMPLES = 16
N_W = 4
HAS_PYMOO = importlib.util.find_spec("pymoo") is not None

pytestmark = pytest.mark.filterwarnings("ignore")


@pytest.fixture(autouse=True, scope="module")
def _single_threaded_torch() -> Iterator[None]:
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(threads)


def bt_eval(fn: Any, *args: Any, **kwargs: Any) -> Any:
    with dense_model_list():
        return fn(*args, **kwargs)


# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------


def hetero(X: Any) -> Any:
    """A heteroscedastic perturbation set: ``n x d`` points to ``n x 2 x d``."""
    return torch.stack([0.05 * X**2, -0.08 * X], dim=-2)


def pset(nw: int, k: int, seed: int, scale: float = 0.25) -> Any:
    g = torch.Generator().manual_seed(seed)
    return scale * (torch.rand(nw, k, generator=g, dtype=DT) - 0.5)


def transform_pair(steps: list[dict[str, Any]], d: int) -> tuple[Any, Any]:
    """The same (chained) input transform in BoTorch and botorch_rs."""
    bts, rss = [], []
    for s in steps:
        if s["type"] == "normalize":
            b = s.get("bounds")
            bts.append(bt_input.Normalize(d=d, bounds=b))
            rss.append(rs_input.Normalize(d=d, bounds=b))
            continue
        ps = hetero if s.get("function") else s["set"]
        kw = {
            "bounds": s.get("bounds"),
            "indices": s.get("indices"),
            "multiplicative": s.get("multiplicative", False),
            "transform_on_eval": s.get("eval", True),
        }
        bts.append(bt_input.InputPerturbation(perturbation_set=ps, **kw))
        rss.append(rs_input.InputPerturbation(perturbation_set=ps, **kw))
    if len(steps) == 1:
        return bts[0], rss[0]
    return (
        bt_input.ChainedInputTransform(**{f"tf{i}": t for i, t in enumerate(bts)}),
        rs_input.ChainedInputTransform(**{f"tf{i}": t for i, t in enumerate(rss)}),
    )


UNIT = torch.tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], dtype=DT)

TRANSFORMS: dict[str, tuple[list[dict[str, Any]], int]] = {
    "fixed": ([{"type": "perturb", "set": pset(4, 3, 1)}], 3),
    "bounds": ([{"type": "perturb", "set": pset(5, 3, 2, 0.6), "bounds": UNIT}], 3),
    "multiplicative_indices": (
        [{"type": "perturb", "set": 1 + pset(3, 2, 3), "indices": [0, 2], "multiplicative": True}],
        3,
    ),
    "negative_index_bounds": (
        [{"type": "perturb", "set": pset(3, 1, 4, 2.0), "indices": [-1], "bounds": UNIT[:, :1]}],
        3,
    ),
    "hetero": ([{"type": "perturb", "function": True}], 3),
    "hetero_indices": ([{"type": "perturb", "function": True, "indices": [1]}], 3),
    "normalize_then_perturb": (
        [
            {"type": "normalize", "bounds": torch.tensor([[-1.0, 0.0, 0.5], [2.0, 1.0, 1.5]])},
            {"type": "perturb", "set": pset(4, 3, 5), "bounds": UNIT},
        ],
        3,
    ),
    "perturb_then_normalize": (
        [
            {"type": "perturb", "set": pset(4, 3, 6)},
            {"type": "normalize", "bounds": torch.tensor([[-1.0, 0.0, 0.5], [2.0, 1.0, 1.5]])},
        ],
        3,
    ),
    "eval_off": ([{"type": "perturb", "set": pset(4, 3, 7), "eval": False}], 3),
}


@pytest.mark.parametrize("name", list(TRANSFORMS))
def test_transforms(name: str) -> None:
    steps, d = TRANSFORMS[name]
    bt, rs = transform_pair(steps, d)
    X = -0.2 + 1.4 * torch.rand(2, 3, d, generator=torch.Generator().manual_seed(9), dtype=DT)
    bt.eval()
    rs.eval()
    out_bt, out_rs = bt(X), rs(X)
    CHECK.check("InputPerturbation / chain: transformed points", out_rs, out_bt, 1e-15)
    if isinstance(bt, bt_input.InputPerturbation):
        # ``transform`` applies regardless of the flags.
        CHECK.check("InputPerturbation.transform", rs.transform(X[0]), bt.transform(X[0]), 1e-15)


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


def model_steps(kind: str, d: int) -> list[dict[str, Any]]:
    unit = torch.tensor([[0.0] * d, [1.0] * d], dtype=DT)
    if kind == "bounds":
        return [{"type": "perturb", "set": pset(N_W, d, 7), "bounds": unit}]
    if kind == "normalize_then_perturb":  # learned bounds
        return [{"type": "normalize"}, {"type": "perturb", "set": pset(N_W, d, 7), "bounds": unit}]
    if kind == "perturb_then_normalize":
        return [{"type": "perturb", "set": pset(N_W, d, 8)}, {"type": "normalize"}]
    if kind == "hetero":
        return [{"type": "normalize"}, {"type": "perturb", "function": True}]
    raise ValueError(kind)


def robust_pair(members: list[tuple[int, str]], seed: int, d: int = 2) -> tuple[Any, Any, Any]:
    """BoTorch and botorch_rs models with identical hyperparameters and
    input transforms (a ModelListGP for several members) and the training
    inputs of the first member."""
    bts, rss, X0 = [], [], None
    for i, (m, kind) in enumerate(members):
        X, Y, _ = make_data(ModelSpec(n=9 + i, d=d, m=m, lo=0.05, hi=0.95), seed + 17 * i)
        tf_bt, tf_rs = transform_pair(model_steps(kind, d), d)
        bt = BtSingleTaskGP(X, Y, input_transform=tf_bt)
        rs = RsSingleTaskGP(X, Y, input_transform=tf_rs)
        set_hyperparameters(bt, seed + i)
        copy_hyperparameters(bt, rs)
        bt.eval()
        bts.append(bt)
        rss.append(rs)
        X0 = X if X0 is None else X0
    if len(members) == 1:
        return bts[0], rss[0], X0
    lb = BtModelListGP(*bts)
    lb.eval()
    return lb, RsModelListGP(*rss), X0


MODEL_CASES = {
    "m1_bounds": [(1, "bounds")],
    "m2_normalize_then_perturb": [(2, "normalize_then_perturb")],
    "m2_perturb_then_normalize": [(2, "perturb_then_normalize")],
    "m2_hetero": [(2, "hetero")],
    "list_m2_m1": [(2, "bounds"), (1, "bounds")],
}


@pytest.mark.parametrize("name", list(MODEL_CASES))
def test_posteriors(name: str) -> None:
    bt, rs, _ = robust_pair(MODEL_CASES[name], 31)
    g = torch.Generator().manual_seed(4)
    for shape in ((3, 2), (2, 2, 2)):
        Xq = 0.1 + 0.8 * torch.rand(*shape, generator=g, dtype=DT)
        for noise in (False, True):
            p_bt = bt_eval(bt.posterior, Xq, observation_noise=noise)
            p_rs = rs.posterior(Xq, observation_noise=noise)
            mean_bt = p_bt.mean.detach()
            assert tuple(p_rs.mean.shape) == tuple(mean_bt.shape)
            CHECK.check("posterior mean (q * n_w points)", p_rs.mean, mean_bt, VALUE_TOL)
            CHECK.check("posterior variance", p_rs.variance, p_bt.variance.detach(), VALUE_TOL)
            with dense_model_list():
                cov_bt = p_bt.distribution.covariance_matrix.detach()
            CHECK.check(
                "posterior covariance", p_rs.distribution.covariance_matrix, cov_bt, VALUE_TOL
            )
    if name.startswith("list"):
        return
    # Samples from identical base samples.
    Xq = 0.1 + 0.8 * torch.rand(2, 2, generator=g, dtype=DT)
    p_bt, p_rs = bt.posterior(Xq), rs.posterior(Xq)
    Z = torch.randn(5, *p_bt.base_sample_shape, generator=g, dtype=DT)
    s_bt = p_bt.rsample_from_base_samples(torch.Size([5]), Z).detach()
    s_rs = p_rs.rsample_from_base_samples(torch.Size([5]), Z)
    CHECK.check("posterior samples (identical base samples)", s_rs, s_bt, 1e-8)


def test_fitted_chain_and_n_w() -> None:
    bt, rs, _ = robust_pair([(2, "hetero")], 5)
    bt_norm = next(iter(bt.input_transform.values()))
    rs_norm = rs.input_transform["tf0"]
    CHECK.check("fitted Normalize bounds in a chain", rs_norm.bounds, bt_norm.bounds, 0.0)
    assert rs.input_transform["tf1"].n_w == 2


# ---------------------------------------------------------------------------
# Risk measures
# ---------------------------------------------------------------------------


def preprocessing_pair(kind: str | None, m: int, mo: bool) -> tuple[Any, Any]:
    """The same preprocessing function in BoTorch and botorch_rs (``None``:
    the default)."""
    if kind is None:
        return None, None
    if kind == "linear":
        w = torch.tensor([1.0, -0.5, 0.25][:m], dtype=DT)
        return bt_obj.LinearMCObjective(w), rs_obj.LinearMCObjective(w)
    if kind == "callable":
        if mo:

            def f_mo(Y: Any, X: Any = None) -> Any:
                return torch.stack([Y[..., 0] - Y[..., 1] ** 2, torch.sin(Y[..., -1])], dim=-1)

            return f_mo, f_mo

        def f(Y: Any, X: Any = None) -> Any:
            return Y[..., 0] * torch.exp(0.3 * Y[..., -1])

        return f, f
    if kind == "weighted":
        w = torch.tensor([1.0, -1.0, 0.5][:m], dtype=DT)
        return (
            bt_mo_obj.WeightedMCMultiOutputObjective(w),
            rs_mo_obj.WeightedMCMultiOutputObjective(w),
        )
    if kind == "outcomes":
        return (
            bt_mo_obj.IdentityMCMultiOutputObjective(outcomes=[0, 2]),
            rs_mo_obj.IdentityMCMultiOutputObjective(outcomes=[0, 2]),
        )
    if kind == "botorch_object":  # a BoTorch objective passed to botorch_rs
        w = torch.tensor([1.0, -1.0, 0.5][:m], dtype=DT)
        o = bt_mo_obj.WeightedMCMultiOutputObjective(w) if mo else bt_obj.LinearMCObjective(w)
        return o, o
    raise ValueError(kind)


# (class name, keyword arguments, m, preprocessing, multi-output)
RISK_CASES: list[tuple[str, dict[str, Any], int, str | None, bool]] = [
    ("CVaR", {"alpha": 0.5}, 1, None, False),
    ("CVaR", {"alpha": 0.28}, 1, None, False),
    ("VaR", {"alpha": 0.6}, 1, None, False),
    ("VaR", {"alpha": 1.0}, 2, "linear", False),
    ("WorstCase", {}, 2, "callable", False),
    ("Expectation", {}, 3, "linear", False),
    ("CVaR", {"alpha": 0.8}, 2, "botorch_object", False),
    ("MultiOutputExpectation", {}, 2, None, True),
    ("IndependentCVaR", {"alpha": 0.6}, 3, "weighted", True),
    ("IndependentVaR", {"alpha": 0.5}, 2, "callable", True),
    ("MultiOutputWorstCase", {}, 3, "outcomes", True),
    ("MVaR", {"alpha": 0.5}, 2, None, True),
    ("MVaR", {"alpha": 0.8}, 3, None, True),
    ("MVaR", {"alpha": 0.6, "filter_dominated": False}, 2, None, True),
    ("MVaR", {"alpha": 0.6, "pad_to_n_w": True}, 2, "weighted", True),
    ("MVaR", {"alpha": 0.5, "use_counting": True}, 2, None, True),
    ("MVaR", {"alpha": 0.7, "use_counting": True, "filter_dominated": False}, 3, None, True),
    ("MVaR", {"alpha": 0.5, "expectation": True}, 2, None, True),
    ("MVaR", {"alpha": 0.5}, 3, "callable", True),
    ("MVaR", {"alpha": 0.5}, 3, "botorch_object", True),
]


def risk_pair(name: str, kw: dict[str, Any], m: int, pre: str | None, mo: bool, n_w: int) -> Any:
    p_bt, p_rs = preprocessing_pair(pre, m, mo)
    bt_mod = bt_morisk if mo else bt_risk
    rs_mod = rs_morisk if mo else rs_risk
    return (
        getattr(bt_mod, name)(n_w=n_w, preprocessing_function=p_bt, **kw),
        getattr(rs_mod, name)(n_w=n_w, preprocessing_function=p_rs, **kw),
    )


@pytest.mark.parametrize("case", RISK_CASES, ids=lambda c: f"{c[0]}-{c[3]}-m{c[2]}")
def test_risk_measures(case: tuple) -> None:
    name, kw, m, pre, mo = case
    n_w = 25 if kw.get("alpha") == 0.28 else 6
    o_bt, o_rs = risk_pair(name, kw, m, pre, mo, n_w)
    g = torch.Generator().manual_seed(len(name) + m)
    Y = torch.randn(3, 2, 2 * n_w, m, generator=g, dtype=DT)
    label = f"{name}{'(expectation)' if kw.get('expectation') else ''}"
    if kw.get("expectation"):
        # BoTorch's gradient is NaN (and its shape wrong) when the samples
        # require gradients: values only.
        with torch.no_grad():
            CHECK.check(f"{label}: values", o_rs(Y), o_bt(Y), VALUE_TOL)
        return
    Yb = Y.clone().requires_grad_(True)
    Yr = Y.clone().requires_grad_(True)
    if name == "MVaR" and pre == "callable":
        # The preprocessing function changes the number of outputs, which
        # BoTorch main mishandles (meta-pytorch/botorch#3392): compare with
        # BoTorch's MVaR of the preprocessed samples.
        p_bt, _ = preprocessing_pair(pre, m, mo)
        plain = bt_morisk.MVaR(n_w=n_w, **kw)
        v_bt, v_rs = plain(p_bt(Yb)), o_rs(Yr)
    else:
        v_bt, v_rs = o_bt(Yb), o_rs(Yr)
    assert tuple(v_rs.shape) == tuple(v_bt.shape), (v_rs.shape, v_bt.shape)
    CHECK.check(f"{label}: values", v_rs, v_bt, VALUE_TOL)
    W = torch.randn(v_bt.shape, generator=g, dtype=DT)
    (g_bt,) = torch.autograd.grad((v_bt * W).sum(), Yb)
    (g_rs,) = torch.autograd.grad((v_rs * W).sum(), Yr)
    CHECK.check(f"{label}: gradients", g_rs, g_bt, GRAD_TOL, atol=1e-15)


@pytest.mark.parametrize("m", [2, 3])
def test_mvar_sets(m: int) -> None:
    Y = torch.randn(4, 7, m, generator=torch.Generator().manual_seed(m), dtype=DT)
    for alpha in (0.3, 0.5, 0.9):
        bt, rs = bt_morisk.MVaR(n_w=7, alpha=alpha), rs_morisk.MVaR(n_w=7, alpha=alpha)
        for method in ("get_mvar_set_vectorized", "get_mvar_set_via_counting"):
            for s_rs, s_bt in zip(getattr(rs, method)(Y), getattr(bt, method)(Y), strict=True):
                CHECK.check(f"MVaR.{method}", s_rs, s_bt, 0.0)


@pytest.mark.parametrize("ref", [None, "below", "inside"])
def test_mars(ref: str | None) -> None:
    bt_model, rs_model, X = robust_pair([(2, "normalize_then_perturb")], 41)
    ref_point = None
    if ref is not None:
        with torch.no_grad():
            means = bt_model.posterior(X).mean
        lo, hi = means.min(0).values, means.max(0).values
        ref_point = lo - 0.1 * (hi - lo) if ref == "below" else lo + 0.5 * (hi - lo)
    w = torch.tensor([0.3, 0.7], dtype=DT)
    bt = bt_morisk.MARS(alpha=0.6, n_w=N_W, chebyshev_weights=w, ref_point=ref_point)
    rs = rs_morisk.MARS(alpha=0.6, n_w=N_W, chebyshev_weights=w, ref_point=ref_point)
    bt.set_baseline_Y(bt_model, X)
    rs.set_baseline_Y(rs_model, X)
    CHECK.check("MARS.set_baseline_Y(model): baseline_Y", rs.baseline_Y, bt.baseline_Y, VALUE_TOL)
    bounds_bt = bt._get_Y_normalization_bounds(bt.baseline_Y, ref_point)
    CHECK.check("MARS normalization bounds", rs.normalization_bounds, bounds_bt, VALUE_TOL)
    CHECK.check(
        "MARS._get_Y_normalization_bounds",
        rs_morisk.MARS._get_Y_normalization_bounds(bt.baseline_Y, ref_point),
        bounds_bt,
        0.0,
    )
    g = torch.Generator().manual_seed(3)
    Y = torch.randn(4, 3 * N_W, 2, generator=g, dtype=DT)
    Yb, Yr = Y.clone().requires_grad_(True), Y.clone().requires_grad_(True)
    v_bt, v_rs = bt(Yb), rs(Yr)
    CHECK.check("MARS: values", v_rs, v_bt, VALUE_TOL)
    (g_bt,) = torch.autograd.grad(v_bt.sum(), Yb)
    (g_rs,) = torch.autograd.grad(v_rs.sum(), Yr)
    CHECK.check("MARS: gradients", g_rs, g_bt, GRAD_TOL, atol=1e-15)
    # The baseline from predictions, and new weights.
    Ys = torch.randn(5 * N_W, 2, generator=g, dtype=DT)
    bt.set_baseline_Y(None, None, Y_samples=Ys)
    rs.set_baseline_Y(None, None, Y_samples=Ys)
    bt.chebyshev_weights = rs.chebyshev_weights = torch.tensor([0.9, 0.1], dtype=DT)
    CHECK.check("MARS.set_baseline_Y(Y_samples): baseline_Y", rs.baseline_Y, bt.baseline_Y, 0.0)
    with torch.no_grad():
        CHECK.check("MARS: values", rs(Y), bt(Y), VALUE_TOL)


# ---------------------------------------------------------------------------
# Acquisition functions
# ---------------------------------------------------------------------------


def per_candidate(acqf: Any, X: Any, grads: bool = True) -> tuple[Any, Any]:
    """Values (and gradients) of every candidate set evaluated on its own."""
    vals, grs = [], []
    for i in range(X.shape[0]):
        if grads:
            v, g = values_and_grads(acqf, X[i : i + 1])
            grs.append(g)
        else:
            with torch.no_grad():
                v = acqf(X[i : i + 1])
        vals.append(v.detach())
    return torch.cat(vals), torch.cat(grs) if grads else None


def compare(
    label: str,
    rs_vg: tuple[Any, Any],
    bt_vg: tuple[Any, Any],
    log: bool,
    twins: Callable[[], Iterator[tuple[Any, Any]]] | None = None,
) -> None:
    """Values and gradients; above the nominal tolerances, the tolerance is
    10x BoTorch's own change under 1e-14 perturbations of its training
    inputs (``twins()``, four perturbations: the change varies by orders of
    magnitude between perturbations when the improvement is negligible and
    the log values are dominated by the smoothing)."""
    (v_rs, g_rs), (v_bt, g_bt) = rs_vg, bt_vg
    err_v = log_err(v_rs, v_bt) if log else rel_err(v_rs, v_bt, atol=1e-300)
    err_g = 0.0 if g_bt is None else rel_err(g_rs, g_bt, atol=1e-14)
    tol_v, tol_g = VALUE_TOL, GRAD_TOL
    if (err_v > tol_v or err_g > tol_g) and twins is not None:
        floor_v = floor_g = 0.0
        for v_tw, g_tw in twins():
            floor_v = max(floor_v, log_err(v_tw, v_bt) if log else rel_err(v_tw, v_bt))
            if g_bt is not None:
                floor_g = max(floor_g, rel_err(g_tw, g_bt, atol=1e-14))
        tol_v, tol_g = max(tol_v, 10 * floor_v), max(tol_g, 10 * floor_g)
        CHECK.record("BoTorch noise floor (perturbed baseline): values", floor_v, None)
        CHECK.record("BoTorch noise floor (perturbed baseline): gradients", floor_g, None)
    # Recorded relative to the (floor-adjusted) tolerance, and as observed.
    CHECK.record(f"{label}: values", err_v * VALUE_TOL / tol_v, VALUE_TOL)
    if g_bt is not None:
        CHECK.record(f"{label}: gradients", err_g * GRAD_TOL / tol_g, GRAD_TOL)
    if tol_v > VALUE_TOL or tol_g > GRAD_TOL:
        CHECK.record(f"{label}: values (observed, above the noise floor tolerance)", err_v, None)
        CHECK.record(f"{label}: gradients (observed)", err_g, None)
    assert err_v <= tol_v, (label, "values", err_v, tol_v)
    assert err_g <= tol_g, (label, "gradients", err_g, tol_g)


def mo_objective_pair(spec: dict[str, Any], n_w: int) -> tuple[Any, Any]:
    kw = {k: v for k, v in spec.items() if k not in ("kind", "botorch_object")}
    bt = getattr(bt_morisk, spec["kind"])(n_w=n_w, **kw)
    if spec.get("botorch_object"):
        return bt, bt
    return bt, getattr(rs_morisk, spec["kind"])(n_w=n_w, **kw)


# (acquisition function, objective, q, cache_root)
MO_ACQ_CASES: list[tuple[str, dict[str, Any], int, bool | None]] = [
    ("qNEHVI", {"kind": "MVaR", "alpha": 0.5}, 1, True),
    ("qLogNEHVI", {"kind": "MVaR", "alpha": 0.75}, 1, False),
    ("qNEHVI", {"kind": "IndependentCVaR", "alpha": 0.5}, 2, False),
    ("qLogNEHVI", {"kind": "MultiOutputExpectation"}, 1, True),
    ("qLogNEHVI", {"kind": "MultiOutputWorstCase"}, 2, True),
    ("qLogNEHVI", {"kind": "MVaR", "alpha": 0.5, "botorch_object": True}, 1, True),
    ("qEHVI", {"kind": "MVaR", "alpha": 0.5}, 1, None),
    ("qLogEHVI", {"kind": "MVaR", "alpha": 0.5, "use_counting": True}, 1, None),
    ("qLogEHVI", {"kind": "IndependentVaR", "alpha": 0.5}, 2, None),
    ("qEHVI", {"kind": "MVaR", "alpha": 0.75, "expectation": True}, 2, None),
]

MO_MODELS = ["m2_normalize_then_perturb", "list_m2_m1"]


def mo_acqf_pair(
    name: str, spec: dict[str, Any], cache_root: Any, bt: Any, rs: Any, X: Any, seed: int, q: int
) -> tuple[Callable[[Any], Any], Any, Any]:
    """``build(bt_model)`` for BoTorch (to be called within ``synced_draws``),
    the botorch_rs acquisition function and the reference point."""
    Xb = X[:5]
    with torch.no_grad():
        means = bt_eval(bt.posterior, Xb).mean
    n_w = means.shape[-2] // Xb.shape[0]
    o_bt, o_rs = mo_objective_pair(spec, n_w)
    ref = means.min(0).values - 0.2 * (means.max(0).values - means.min(0).values)
    m = means.shape[-1]
    max_dim = (Xb.shape[0] + q) * n_w * m
    if name in ("qNEHVI", "qLogNEHVI"):
        log = name == "qLogNEHVI"
        mod_bt, mod_rs = (bt_logei, rs_logei) if log else (bt_mc, rs_mc)
        cls = (
            "qLogNoisyExpectedHypervolumeImprovement"
            if log
            else "qNoisyExpectedHypervolumeImprovement"
        )

        def build(model: Any) -> Any:
            return getattr(mod_bt, cls)(
                model=model,
                ref_point=ref,
                X_baseline=Xb,
                sampler=bt_sampler(NUM_SAMPLES),
                objective=o_bt,
                prune_baseline=False,
                cache_root=cache_root,
            )

        a_rs = getattr(mod_rs, cls)(
            rs,
            ref_point=ref,
            X_baseline=Xb,
            sampler=rs_sampler(NUM_SAMPLES, seed, max_dim),
            objective=o_rs,
            prune_baseline=False,
            cache_root=cache_root,
        )
        return build, a_rs, ref
    with torch.no_grad():
        front = o_bt(means.unsqueeze(0)).squeeze(0)
    front = front.unsqueeze(0) if front.dim() == 1 else front
    part_bt = bt_bd.FastNondominatedPartitioning(ref_point=ref, Y=front)
    part_rs = rs_bd.FastNondominatedPartitioning(ref_point=ref, Y=front)
    log = name == "qLogEHVI"
    mod_bt, mod_rs = (bt_logei, rs_logei) if log else (bt_mc, rs_mc)
    cls = "qLogExpectedHypervolumeImprovement" if log else "qExpectedHypervolumeImprovement"

    def build_ehvi(model: Any) -> Any:
        return getattr(mod_bt, cls)(
            model=model,
            ref_point=ref,
            partitioning=part_bt,
            sampler=bt_sampler(NUM_SAMPLES),
            objective=o_bt,
        )

    a_rs = getattr(mod_rs, cls)(
        rs,
        ref_point=ref,
        partitioning=part_rs,
        sampler=rs_sampler(NUM_SAMPLES, seed, q * n_w * m),
        objective=o_rs,
    )
    return build_ehvi, a_rs, ref


def check_mo_acquisition(model_name: str, case: tuple, seed: int) -> None:
    name, spec, q, cache_root = case
    bt, rs, X = robust_pair(MODEL_CASES[model_name], seed)
    build, a_rs, _ = mo_acqf_pair(name, spec, cache_root, bt, rs, X, seed, q)
    with synced_draws(seed):
        a_bt = bt_eval(build, bt)
    Xt = 0.1 + 0.8 * torch.rand(2, q, 2, generator=torch.Generator().manual_seed(seed), dtype=DT)
    grads = not spec.get("expectation")
    with synced_draws(seed):
        vg_bt = bt_eval(per_candidate, a_bt, Xt, grads)
    vg_rs = per_candidate(a_rs, Xt, grads)
    noisy = name in ("qNEHVI", "qLogNEHVI")

    def twins() -> Iterator[tuple[Any, Any]]:
        for k in (1, 2, 3, 4):
            twin = perturbed_copy(bt, seed=k)
            with synced_draws(seed):
                a_tw = bt_eval(build, twin)
                yield bt_eval(per_candidate, a_tw, Xt, grads)

    label = f"{name}({spec['kind']}{', BoTorch object' if spec.get('botorch_object') else ''})"
    compare(label, vg_rs, vg_bt, name.startswith("qLog"), twins if noisy else None)


@pytest.mark.parametrize("model_name", MO_MODELS)
@pytest.mark.parametrize(
    "case", MO_ACQ_CASES, ids=lambda c: f"{c[0]}-{c[1]['kind']}-q{c[2]}-cr{c[3]}"
)
def test_mo_acquisition_with_risk_measures(model_name: str, case: tuple) -> None:
    check_mo_acquisition(model_name, case, 61)


# (acquisition function, objective, q)
SO_ACQ_CASES: list[tuple[str, dict[str, Any], int]] = [
    ("qLogNEI", {"kind": "CVaR", "alpha": 0.5}, 2),
    ("qLogNEI", {"kind": "VaR", "alpha": 0.75}, 1),
    ("qLogNEI", {"kind": "MARS", "alpha": 0.5}, 2),
    ("qLogNEI", {"kind": "CVaR", "alpha": 0.5, "botorch_object": True}, 1),
    ("qLogEI", {"kind": "Expectation"}, 2),
    ("qSimpleRegret", {"kind": "WorstCase"}, 2),
]


def nei_sampler(seed: int, d_baseline: int, d_full: int, max_dim: int) -> Any:
    """A botorch_rs sampler with BoTorch's effective base samples of qLogNEI
    with ``cache_root=True``: the baseline draw (``d_baseline`` columns)
    followed by the new points' columns of the draw for all points."""
    s = rs_sampler(NUM_SAMPLES, seed, max_dim)
    z = zdraw(d_full, NUM_SAMPLES, seed).clone()
    z[:, :d_baseline] = zdraw(d_baseline, NUM_SAMPLES, seed)
    s._rs.set_base_samples((z.reshape(-1).tolist(), (NUM_SAMPLES, d_full)))
    return s


def check_so_acquisition(case: tuple, seed: int) -> None:
    name, spec, q = case
    mars = spec["kind"] == "MARS"
    members = [(2, "normalize_then_perturb")] if mars else [(1, "bounds")]
    bt, rs, X = robust_pair(members, seed)
    Xb = X[:5]
    m = 2 if mars else 1
    if mars:
        w = torch.tensor([0.4, 0.6], dtype=DT)
        o_bt = bt_morisk.MARS(alpha=spec["alpha"], n_w=N_W, chebyshev_weights=w)
        o_rs = rs_morisk.MARS(alpha=spec["alpha"], n_w=N_W, chebyshev_weights=w)
        bt_eval(o_bt.set_baseline_Y, bt, Xb)
        o_rs.set_baseline_Y(rs, Xb)
    else:
        kw = {k: v for k, v in spec.items() if k not in ("kind", "botorch_object")}
        o_bt = getattr(bt_risk, spec["kind"])(n_w=N_W, **kw)
        o_rs = o_bt if spec.get("botorch_object") else getattr(rs_risk, spec["kind"])(n_w=N_W, **kw)
    max_dim = (Xb.shape[0] + q) * N_W * m
    with torch.no_grad():
        # A float64 tensor: BoTorch stores Python floats in float32.
        best_f = o_bt(bt.posterior(Xb).mean.unsqueeze(0)).max()
    if name == "qLogNEI":

        def build(model: Any) -> Any:
            return bt_logei_so.qLogNoisyExpectedImprovement(
                model=model,
                X_baseline=Xb,
                sampler=bt_sampler(NUM_SAMPLES),
                objective=o_bt,
                prune_baseline=False,
            )

        d_b = Xb.shape[0] * N_W * m
        a_rs = rs_logei_so.qLogNoisyExpectedImprovement(
            rs,
            Xb,
            sampler=nei_sampler(seed, d_b, max_dim, max_dim),
            objective=o_rs,
            prune_baseline=False,
        )
    elif name == "qLogEI":

        def build(model: Any) -> Any:
            return bt_logei_so.qLogExpectedImprovement(
                model=model, best_f=best_f, sampler=bt_sampler(NUM_SAMPLES), objective=o_bt
            )

        a_rs = rs_logei_so.qLogExpectedImprovement(
            rs, best_f=best_f, sampler=rs_sampler(NUM_SAMPLES, seed, max_dim), objective=o_rs
        )
    else:

        def build(model: Any) -> Any:
            return bt_mc_so.qSimpleRegret(
                model=model, sampler=bt_sampler(NUM_SAMPLES), objective=o_bt
            )

        a_rs = rs_mc_so.qSimpleRegret(
            rs, sampler=rs_sampler(NUM_SAMPLES, seed, max_dim), objective=o_rs
        )
    with synced_draws(seed):
        a_bt = bt_eval(build, bt)
    Xt = 0.1 + 0.8 * torch.rand(2, q, 2, generator=torch.Generator().manual_seed(seed), dtype=DT)
    with synced_draws(seed):
        vg_bt = bt_eval(per_candidate, a_bt, Xt)
    vg_rs = per_candidate(a_rs, Xt)

    def twins() -> Iterator[tuple[Any, Any]]:
        for k in (1, 2, 3, 4):
            twin = perturbed_copy(bt, seed=k)
            with synced_draws(seed):
                a_tw = bt_eval(build, twin)
                yield bt_eval(per_candidate, a_tw, Xt)

    label = f"{name}({spec['kind']}{', BoTorch object' if spec.get('botorch_object') else ''})"
    compare(label, vg_rs, vg_bt, name.startswith("qLog"), twins if name == "qLogNEI" else None)


@pytest.mark.parametrize("case", SO_ACQ_CASES, ids=lambda c: f"{c[0]}-{c[1]['kind']}-q{c[2]}")
def test_single_objective_acquisition_with_risk_measures(case: tuple) -> None:
    check_so_acquisition(case, 71)


# ---------------------------------------------------------------------------
# MOMF
# ---------------------------------------------------------------------------


def exp_cost(X: Any) -> Any:
    """The exponential fidelity cost of BoTorch's MOMF tutorial."""
    return torch.exp(4.8 * X[..., -1:]) / 121.5 + 0.02


@pytest.mark.parametrize(
    "cost, pending, constrained",
    [(None, False, False), ("exp", False, False), ("exp", True, False), (None, True, True)],
)
def test_momf(cost: str | None, pending: bool, constrained: bool) -> None:
    d, seed = 3, 81
    X, Y, _ = make_data(ModelSpec(n=10, d=d, m=2), seed)
    Y = torch.cat([Y, X[:, -1:]], dim=-1)  # the trust objective (the fidelity)
    bt = BtSingleTaskGP(X, Y)
    rs = RsSingleTaskGP(X, Y)
    set_hyperparameters(bt, seed)
    copy_hyperparameters(bt, rs)
    bt.eval()
    lo, hi = Y.min(0).values, Y.max(0).values
    ref = lo - 0.1 * (hi - lo)
    front = ref + 0.6 * (Y - ref)
    part_bt = bt_bd.NondominatedPartitioning(ref_point=ref, Y=front)
    part_rs = rs_bd.NondominatedPartitioning(ref_point=ref, Y=front)
    g = torch.Generator().manual_seed(seed)
    kw: dict[str, Any] = {}
    if cost == "exp":
        kw["cost_call"] = exp_cost
    q, n_p = 2, 0
    if pending:
        n_p = 2
        kw["X_pending"] = torch.rand(n_p, d, generator=g, dtype=DT)
    if constrained:
        threshold = Y[:, 0].mean().item()
        kw["constraints"] = [lambda Z: threshold - Z[..., 0]]
    with synced_draws(seed):
        a_bt = bt_mf.MOMF(bt, ref, part_bt, sampler=bt_sampler(NUM_SAMPLES), **kw)
    a_rs = rs_mf.MOMF(rs, ref, part_rs, sampler=rs_sampler(NUM_SAMPLES, seed, (q + n_p) * 3), **kw)
    Xt = torch.rand(3, q, d, generator=g, dtype=DT)
    with synced_draws(seed):
        vg_bt = values_and_grads(a_bt, Xt)
    compare("MOMF", values_and_grads(a_rs, Xt), vg_bt, False)


# ---------------------------------------------------------------------------
# NSGA-II
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not HAS_PYMOO, reason="pymoo is not installed")
@pytest.mark.parametrize("m", [2, 3])
def test_nsga2_building_blocks_match_pymoo(m: int) -> None:
    from pymoo.operators.survival.rank_and_crowding.metrics import calc_crowding_distance
    from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

    from botorch_rs import _botorch_rs as _rs

    g = torch.Generator().manual_seed(m)
    # Rounded values give ties and duplicate points.
    F = torch.round(torch.rand(60, m, generator=g, dtype=DT) * 8) / 8
    fronts = NonDominatedSorting().do(F.numpy())
    ours = _rs.nsga2_non_dominated_fronts((F.reshape(-1).tolist(), [60, m]))
    assert [sorted(int(i) for i in f) for f in fronts] == [sorted(f) for f in ours]
    # Crowding distances of fronts of distinct points (NSGA-II's survival
    # gives fronts of at most two points infinite distances).
    U = torch.unique(F, dim=0)
    U = U[torch.randperm(U.shape[0], generator=g)]
    for front in NonDominatedSorting().do(U.numpy()):
        Ff = U[torch.as_tensor(front)]
        cd_ours = torch.as_tensor(
            _rs.nsga2_crowding_distance((Ff.reshape(-1).tolist(), list(Ff.shape))), dtype=DT
        )
        if Ff.shape[0] <= 2:
            assert torch.isinf(cd_ours).all()
            continue
        cd_pymoo = torch.as_tensor(calc_crowding_distance(Ff.numpy()), dtype=DT)
        CHECK.check("NSGA-II crowding distance (pymoo)", cd_ours, cd_pymoo, 1e-15)


def _hv(Y: Any, ref: Any) -> float:
    return Hypervolume(ref_point=ref).compute(Y)


@pytest.mark.skipif(not HAS_PYMOO, reason="pymoo is not installed")
def test_optimize_with_nsgaii_matches_botorch_statistically() -> None:
    from botorch.acquisition.multioutput_acquisition import (
        MultiOutputPosteriorMean as BtMOPosteriorMean,
    )
    from botorch.utils.multi_objective.optimize import (
        optimize_with_nsgaii as bt_optimize_with_nsgaii,
    )

    X, Y, _ = make_data(ModelSpec(n=12, d=3, m=2), 91)
    bt = BtSingleTaskGP(X, Y)
    rs = RsSingleTaskGP(X, Y)
    set_hyperparameters(bt, 91)
    copy_hyperparameters(bt, rs)
    bt.eval()
    bounds = torch.tensor([[0.0] * 3, [1.0] * 3], dtype=DT)
    with torch.no_grad():
        means = bt.posterior(X).mean
    ref = means.min(0).values - 0.5 * (means.max(0).values - means.min(0).values)
    hv_bt, hv_rs = [], []
    for seed in range(10):
        for max_gen in (10, 40):
            _, Yb = bt_optimize_with_nsgaii(
                BtMOPosteriorMean(bt), bounds, 2, population_size=50, max_gen=max_gen, seed=seed
            )
            _, Yr = rs_optimize_with_nsgaii(
                RsMOPosteriorMean(rs), bounds, 2, population_size=50, max_gen=max_gen, seed=seed
            )
            hv_bt.append((max_gen, _hv(Yb.detach(), ref)))
            hv_rs.append((max_gen, _hv(Yr, ref)))
    for max_gen in (10, 40):
        a = torch.tensor([h for g_, h in hv_rs if g_ == max_gen], dtype=DT)
        b = torch.tensor([h for g_, h in hv_bt if g_ == max_gen], dtype=DT)
        diff = (a.mean() - b.mean()).abs().item()
        se = math.sqrt(a.var().item() / len(a) + b.var().item() / len(b))
        CHECK.record(
            f"optimize_with_nsgaii: |mean HV - BoTorch's| / HV ({max_gen} generations)",
            diff / b.mean().item(),
            None,
        )
        # Equal within 4 standard errors (or 1% of the hypervolume).
        assert diff <= max(4 * se, 1e-2 * b.mean().item()), (max_gen, a, b)


def test_optimize_with_nsgaii_finds_posterior_mean_front() -> None:
    """Without pymoo: the front of the posterior mean found by NSGA-II has
    (almost) the hypervolume of the front of a dense Sobol grid."""
    X, Y, _ = make_data(ModelSpec(n=12, d=2, m=2), 93)
    rs = RsSingleTaskGP(X, Y)
    bt = BtSingleTaskGP(X, Y)
    set_hyperparameters(bt, 93)
    copy_hyperparameters(bt, rs)
    bounds = torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=DT)
    _, Yr = rs_optimize_with_nsgaii(RsMOPosteriorMean(rs), bounds, 2, population_size=100, seed=0)
    grid = torch.quasirandom.SobolEngine(2, scramble=True, seed=0).draw(4096).to(DT)
    Yg = rs.posterior(grid).mean
    ref = Yg.min(0).values - 0.1 * (Yg.max(0).values - Yg.min(0).values)
    hv_rs, hv_grid = _hv(Yr, ref), _hv(Yg, ref)
    CHECK.record(
        "optimize_with_nsgaii: HV shortfall vs dense grid (relative)",
        max(1 - hv_rs / hv_grid, 0.0),
        None,
    )
    assert hv_rs >= 0.995 * hv_grid, (hv_rs, hv_grid)


# ---------------------------------------------------------------------------
# Randomized sweep
# ---------------------------------------------------------------------------


@pytest.mark.equivalence
@pytest.mark.parametrize("seed", range(16))
def test_equivalence_sweep(seed: int) -> None:
    """Random models, risk measures and acquisition functions."""
    r = random.Random(seed)
    if r.random() < 0.35:
        case = r.choice(SO_ACQ_CASES)
        check_so_acquisition(case, 200 + seed)
        return
    case = r.choice(MO_ACQ_CASES)
    model = r.choice(MO_MODELS + ["m2_hetero", "m2_perturb_then_normalize"])
    check_mo_acquisition(model, case, 300 + seed)


@pytest.mark.equivalence
@pytest.mark.parametrize("seed", range(24))
def test_equivalence_sweep_risk_measures(seed: int) -> None:
    """Random samples (shapes, n_w, alpha) for every risk measure."""
    r = random.Random(1000 + seed)
    name, kw, m, pre, mo = r.choice(RISK_CASES)
    kw = dict(kw)
    if "alpha" in kw:
        kw["alpha"] = r.choice([0.1, 0.25, 0.5, 0.65, 0.9, 1.0])
    n_w = r.randint(2, 9)
    o_bt, o_rs = risk_pair(name, kw, m, pre, mo, n_w)
    g = torch.Generator().manual_seed(seed)
    Y = torch.randn(r.randint(1, 4), r.randint(1, 3) * n_w, m, generator=g, dtype=DT)
    if kw.get("expectation"):
        with torch.no_grad():
            if pre == "callable":
                p_bt, _ = preprocessing_pair(pre, m, mo)
                o_bt, Y_bt = bt_morisk.MVaR(n_w=n_w, **kw), p_bt(Y)
            else:
                Y_bt = Y
            CHECK.check(f"sweep {name}: values", o_rs(Y), o_bt(Y_bt), VALUE_TOL)
        return
    Yb, Yr = Y.clone().requires_grad_(True), Y.clone().requires_grad_(True)
    if name == "MVaR" and pre == "callable":  # see test_risk_measures (#3392)
        p_bt, _ = preprocessing_pair(pre, m, mo)
        v_bt, v_rs = bt_morisk.MVaR(n_w=n_w, **kw)(p_bt(Yb)), o_rs(Yr)
    else:
        v_bt, v_rs = o_bt(Yb), o_rs(Yr)
    CHECK.check(f"sweep {name}: values", v_rs, v_bt, VALUE_TOL)
    (g_bt,) = torch.autograd.grad(v_bt.sum(), Yb)
    (g_rs,) = torch.autograd.grad(v_rs.sum(), Yr)
    CHECK.check(f"sweep {name}: gradients", g_rs, g_bt, GRAD_TOL, atol=1e-15)

"""Parity of input and outcome transforms (phase 6b) with BoTorch, through
the public APIs of both libraries side by side.

Every model is built from the same JSON-style transform specs in both
libraries (``transform_helpers.py``) and compared:

* ``SingleTaskGP`` models with every transform alone and chained: parameter
  names, shapes, default values and optimizer bounds (exact); the MLL and
  its gradient with respect to all raw parameters, including the learned
  parameters of ``Warp`` and ``LearnedFeatureImputation``, at the default
  and at random parameters; posteriors (means and variances of
  ``TransformedPosterior``s, with and without observation noise; samples
  from identical base samples and their input gradients); the values and
  input gradients of ``PosteriorMean``, ``ExpectedImprovement``,
  ``qExpectedImprovement`` and ``qSimpleRegret`` (identical base samples);
* one-to-many input transforms (``InputPerturbation``, ``AppendFeatures``)
  in Monte-Carlo acquisition functions with risk measures;
* ``ModelListGP`` with transformed sub-models (``PosteriorList``,
  ``ScalarizedPosteriorTransform`` at ``q = 1``, Monte-Carlo acquisition
  functions with a ``ListSampler``);
* conditioning on observations and fantasies (with ``Log``, chains,
  ``StratifiedStandardize``, input transforms and ``InputPerturbation``,
  which fantasies skip);
* batched models (``Warp`` with a batch shape, ``BatchBroadcastedInputTransform``);
* posteriors of models with transforms of some outputs (``outputs=``),
  which BoTorch cannot untransform (``NotImplementedError``), against
  single-output BoTorch models with the same parameters;
* fits from identical initial values (the rule of
  ``test_parity_6a_kernels.py``, with 4 twins).

Tolerances: values 1e-9, gradients 1e-8 (relative). BoTorch main's
transform bugs fixed by meta-pytorch/botorch#3389 are patched for the
comparisons (``transform_helpers.botorch_fixes``). The randomized sweep over
random transform chains is marked ``equivalence``.
"""

from __future__ import annotations

import math
import random
import statistics
from collections.abc import Iterator
from typing import Any

import pytest

from helpers import importorskip_botorch

importorskip_botorch()
torch = pytest.importorskip("torch")

from botorch.acquisition import ExpectedImprovement as BtEI  # noqa: E402
from botorch.acquisition import PosteriorMean as BtPosteriorMean  # noqa: E402
from botorch.acquisition import qExpectedImprovement as BtQEI  # noqa: E402
from botorch.acquisition import qSimpleRegret as BtQSR  # noqa: E402
from botorch.acquisition.objective import LinearMCObjective as BtLinearMCObjective  # noqa: E402
from botorch.acquisition.objective import (  # noqa: E402
    ScalarizedPosteriorTransform as BtScalarized,
)
from botorch.acquisition.risk_measures import CVaR as BtCVaR  # noqa: E402
from botorch.acquisition.risk_measures import Expectation as BtExpectation  # noqa: E402
from botorch.fit import fit_gpytorch_mll as bt_fit  # noqa: E402
from botorch.models import ModelListGP as BtModelListGP  # noqa: E402
from botorch.models import SingleTaskGP as BtSingleTaskGP  # noqa: E402
from botorch.optim.utils import get_parameters_and_bounds  # noqa: E402
from botorch.sampling import ListSampler as BtListSampler  # noqa: E402
from botorch.sampling.normal import SobolQMCNormalSampler as BtSampler  # noqa: E402
from clamp_subgradients import torch214_clamp_subgradients  # noqa: E402
from gpytorch.mlls import ExactMarginalLogLikelihood as BtMLL  # noqa: E402
from transform_helpers import (  # noqa: E402
    botorch_fixes,
    build_input,
    build_outcome,
    has_moments,
    prior,
    random_input_spec,
    random_outcome_spec,
)

from botorch_rs.acquisition import ExpectedImprovement as RsEI  # noqa: E402
from botorch_rs.acquisition import PosteriorMean as RsPosteriorMean  # noqa: E402
from botorch_rs.acquisition import qExpectedImprovement as RsQEI  # noqa: E402
from botorch_rs.acquisition import qSimpleRegret as RsQSR  # noqa: E402
from botorch_rs.acquisition.objective import LinearMCObjective as RsLinearMCObjective  # noqa: E402
from botorch_rs.acquisition.objective import (  # noqa: E402
    ScalarizedPosteriorTransform as RsScalarized,
)
from botorch_rs.acquisition.risk_measures import CVaR as RsCVaR  # noqa: E402
from botorch_rs.acquisition.risk_measures import Expectation as RsExpectation  # noqa: E402
from botorch_rs.fit import fit_gpytorch_mll as rs_fit  # noqa: E402
from botorch_rs.mlls import ExactMarginalLogLikelihood as RsMLL  # noqa: E402
from botorch_rs.models import ModelListGP as RsModelListGP  # noqa: E402
from botorch_rs.models import SingleTaskGP as RsSingleTaskGP  # noqa: E402
from botorch_rs.sampling import SobolQMCNormalSampler as RsSampler  # noqa: E402
from parity_helpers import Checker  # noqa: E402

pytestmark = pytest.mark.filterwarnings("ignore")

DT = torch.float64
CHECK = Checker("input and outcome transforms (6b)")
VALUE_TOL = 1e-9
GRAD_TOL = 1e-8
FIT_TOL = 1e-6
FIT_CLUSTER = 1e-4
CHAOTIC_RANGE = 1e-3
TWIN_SCALES = (1e-14, 1e-13, 1e-12, 1e-11)
N_TWINS = 4
CHAOTIC_TWINS = 16


@pytest.fixture(autouse=True, scope="module")
def _single_threaded_torch() -> Iterator[None]:
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(threads)


def _grad_err(a: Any, e: Any) -> float:
    """Gradient error relative to the largest component; NaNs must match
    (BoTorch's gradients of t-batches with NaN samples, e.g. of ``Power``
    transforms at negative base samples, are NaN)."""
    a, e = torch.as_tensor(a, dtype=DT).reshape(-1), torch.as_tensor(e, dtype=DT).reshape(-1)
    assert a.shape == e.shape
    nan_a, nan_e = torch.isnan(a), torch.isnan(e)
    if not torch.equal(nan_a, nan_e):
        return math.inf
    a, e = a[~nan_e], e[~nan_e]
    if e.numel() == 0:
        return 0.0
    scale = max(e.abs().max().item(), 1e-300)
    return (a - e).abs().max().item() / scale


def check_grad(quantity: str, a: Any, e: Any, tol: float = GRAD_TOL) -> None:
    err = _grad_err(a, e)
    CHECK.record(quantity, err, tol)
    assert err <= tol, (quantity, err)


# ---------------------------------------------------------------------------
# Cases
# ---------------------------------------------------------------------------

LOGN = prior("LogNormalPrior", 0.0, 0.75**0.5)

#: name: (input spec, outcome spec ("default": Standardize), data options).
CASES: dict[str, tuple[Any, Any, dict[str, Any]]] = {
    "log": (None, [{"type": "Log"}], {"fit": True}),
    "log_std": (None, [{"type": "Log"}, {"type": "Standardize"}], {}),
    "std_bilog": (None, [{"type": "Standardize"}, {"type": "Bilog"}], {}),
    "power_half": (None, [{"type": "Power", "power": 0.5}], {}),
    "power_two_std": (None, [{"type": "Power", "power": 2.0}, {"type": "Standardize"}], {}),
    "bilog_std": (None, [{"type": "Bilog"}, {"type": "Standardize"}], {"shift": -2.0}),
    "log_fixed_noise": (None, [{"type": "Log"}], {"fixed_noise": True}),
    "stratified": (
        None,
        [{"type": "StratifiedStandardize", "stratification_idx": 2, "task_values": [0, 1, 2]}],
        {"tasks": True, "d": 3},
    ),
    "normalize_indices_center": (
        [{"type": "Normalize", "d": 3, "indices": [0, 2], "center": 0.0}],
        "default",
        {"d": 3, "scale_x": 3.0},
    ),
    "normalize_bounds_reverse_flags": (
        [
            {
                "type": "Normalize",
                "d": 2,
                "bounds": [[-1.0, 0.0], [2.0, 2.0]],
                "flags": [True, True, False],
            }
        ],
        "default",
        {},
    ),
    "input_standardize": ([{"type": "InputStandardize", "d": 2}], [{"type": "Log"}], {}),
    "affine": (
        [
            {
                "type": "AffineInputTransform",
                "d": 2,
                "coefficient": [2.0, 0.5],
                "offset": [0.1, -0.2],
            }
        ],
        "default",
        {},
    ),
    "log10": ([{"type": "Log10", "indices": [1]}], "default", {"shift_x": 0.5}),
    "warp": ([{"type": "Warp", "d": 2, "indices": [0, 1]}], "default", {"fit": True}),
    "warp_priors_bounds": (
        [
            {
                "type": "Warp",
                "d": 2,
                "indices": [1],
                "c0_prior": LOGN,
                "c1_prior": LOGN,
                "bounds": [[-1.0, -1.0], [2.0, 2.0]],
            }
        ],
        [{"type": "Log"}],
        {"fit": True},
    ),
    "normalize_warp": (
        [{"type": "Normalize", "d": 2}, {"type": "Warp", "d": 2, "indices": [0, 1]}],
        "default",
        {"scale_x": 3.0, "shift_x": -1.0},
    ),
    "round_eval": (
        [{"type": "Round", "integer_indices": [0], "flags": [False, True, True]}],
        "default",
        {"int_col": 0},
    ),
    "round_approximate": (
        [{"type": "Round", "integer_indices": [0], "approximate": True, "tau": 0.1}],
        [{"type": "Log"}],
        {"int_col": 0},
    ),
    "filter": ([{"type": "FilterFeatures", "feature_indices": [0, 2]}], "default", {"d": 3}),
    "interaction": ([{"type": "InteractionFeatures"}], "default", {}),
    "one_hot_to_numeric": (
        [{"type": "OneHotToNumeric", "dim": 4, "categorical_features": {"1": 3}}],
        "default",
        {"one_hot": True, "d": 4},
    ),
    "numeric_to_categorical": (
        [{"type": "NumericToCategoricalEncoding", "dim": 3, "categorical_features": {"2": 3}}],
        "default",
        {"tasks": True, "d": 3},
    ),
    "imputation": (
        [
            {
                "type": "LearnedFeatureImputation",
                "feature_indices": {"0": [0, 1], "1": [0], "2": [1]},
                "d": 2,
            }
        ],
        "default",
        {"tasks": True, "d": 3},
    ),
    "chain_mixed": (
        [
            {"type": "Normalize", "d": 3, "indices": [0, 1]},
            {"type": "Warp", "d": 3, "indices": [0, 1]},
            {"type": "InteractionFeatures", "indices": [0, 2]},
        ],
        [{"type": "Bilog"}, {"type": "Standardize"}],
        {"d": 3, "scale_x": 2.0},
    ),
}


def make_data(opts: dict[str, Any], seed: int, n: int = 12) -> tuple[Any, Any, Any]:
    g = torch.Generator().manual_seed(seed)
    d, m = opts.get("d", 2), opts.get("m", 1)
    if opts.get("one_hot"):
        X = torch.cat(
            [
                torch.rand(n, 1, generator=g, dtype=DT),
                torch.nn.functional.one_hot(torch.arange(n) % 3, 3).to(DT),
            ],
            -1,
        )
    else:
        X = torch.rand(n, d + opts.get("append", 0), generator=g, dtype=DT)
    if opts.get("tasks"):
        X[:, -1] = (torch.arange(n, dtype=DT) % 3).to(DT)
    if "int_col" in opts:
        X[:, opts["int_col"]] = torch.randint(0, 4, (n,), generator=g).to(DT)
    X = X * opts.get("scale_x", 1.0) + opts.get("shift_x", 0.0)
    cols = []
    for t in range(m):
        f = torch.sin((3.0 + t) * X[:, 0]) + (1.0 + t) * (X[:, 1:] ** 2).sum(-1)
        cols.append(f + 0.05 * torch.randn(n, generator=g, dtype=DT) + 2.0 + opts.get("shift", 0.0))
    Y = torch.stack(cols, -1)
    Yvar = 0.01 + 0.02 * torch.rand(n, m, generator=g, dtype=DT)
    return X, Y, Yvar


def query_points(opts: dict[str, Any], q: int, seed: int) -> Any:
    g = torch.Generator().manual_seed(seed)
    d = opts.get("d", 2)
    if opts.get("one_hot"):
        X = torch.cat(
            [
                torch.rand(q, 1, generator=g, dtype=DT),
                torch.nn.functional.one_hot(torch.arange(q) % 3, 3).to(DT),
            ],
            -1,
        )
    else:
        X = torch.rand(q, d, generator=g, dtype=DT)
    if opts.get("tasks"):
        X[:, -1] = torch.tensor([float(i % 3) for i in range(q)], dtype=DT)
    if "int_col" in opts:
        X[:, opts["int_col"]] = 3.0 * torch.rand(q, generator=g, dtype=DT)
    return X * opts.get("scale_x", 1.0) + opts.get("shift_x", 0.0)


def model_pair(
    input_spec: Any, outcome_spec: Any, opts: dict[str, Any], seed: int, X: Any = None
) -> tuple[Any, Any, Any, Any]:
    """The same SingleTaskGP in both libraries (default parameters) and the
    training data."""
    X0, Y, Yvar = make_data(opts, seed)
    X = X0 if X is None else X
    m = opts.get("m", 1)
    kb: dict[str, Any] = {}
    kr: dict[str, Any] = {}
    if outcome_spec != "default":
        kb["outcome_transform"] = build_outcome(outcome_spec, "bt", m)
        kr["outcome_transform"] = build_outcome(outcome_spec, "rs", m)
    if input_spec is not None:
        kb["input_transform"] = build_input(input_spec, "bt")
        kr["input_transform"] = build_input(input_spec, "rs")
    yvar = Yvar if opts.get("fixed_noise") else None
    bt = BtSingleTaskGP(X, Y, train_Yvar=yvar, **kb).to(DT)
    rs = RsSingleTaskGP(X, Y, train_Yvar=yvar, **kr)
    return bt, rs, X, Y


def _bounds(bt: Any) -> dict[str, tuple[Any, Any]]:
    _, bounds = get_parameters_and_bounds(BtMLL(bt.likelihood, bt))
    return {k.removeprefix("model."): v for k, v in bounds.items()}


def _flat_bounds(bt: Any) -> list[tuple[float, float]]:
    bounds = _bounds(bt)
    out: list[tuple[float, float]] = []
    for name, p in bt.named_parameters():
        lo, hi = bounds.get(name, (None, None))
        for i in range(p.numel()):
            pair = []
            for b, default in ((lo, -math.inf), (hi, math.inf)):
                if b is None:
                    pair.append(default)
                elif torch.is_tensor(b) and b.numel() > 1:
                    pair.append(float(b.reshape(-1)[i]))
                else:
                    pair.append(float(b))
            out.append((pair[0], pair[1]))
    return out


def copy_params(bt: Any, rs: Any) -> None:
    rs.initialize(**{name: p.detach() for name, p in bt.named_parameters()})


def randomize(bt: Any, rs: Any, seed: int) -> None:
    """Random raw parameters within the optimizer bounds, in both models."""
    g = torch.Generator().manual_seed(seed)
    bounds = iter(_flat_bounds(bt))
    with torch.no_grad():
        for name, p in bt.named_parameters():
            vals = []
            for _ in range(p.numel()):
                lo, hi = next(bounds)
                u = torch.rand(1, generator=g, dtype=DT).item()
                if "raw_noise" in name:
                    v = 0.02 + 0.06 * u
                elif "raw_constant" in name:
                    v = 0.8 * (u - 0.5)
                elif "concentration" in name:
                    v = 0.4 + 1.6 * u
                elif "imputation" in name:
                    v = 2.0 * (u - 0.5)
                else:
                    v = 0.3 + 0.5 * u
                vals.append(min(max(v, lo), hi))
            p.copy_(torch.tensor(vals, dtype=DT).reshape(p.shape))
    copy_params(bt, rs)


def bt_mll_value_and_grad(bt: Any) -> tuple[Any, Any]:
    mll = BtMLL(bt.likelihood, bt)
    bt.train()
    bt.zero_grad()
    values = mll(bt(*bt.train_inputs), bt.train_targets)
    values.sum().backward()
    grad = torch.cat(
        [
            (p.grad if p.grad is not None else torch.zeros_like(p)).reshape(-1)
            for _, p in bt.named_parameters()
        ]
    )
    bt.zero_grad()
    bt.eval()
    return values.detach().reshape(-1), grad


def rs_mll_value_and_grad(rs: Any) -> tuple[Any, Any]:
    value, grad = RsMLL(rs.likelihood, rs).value_and_grad()
    return torch.as_tensor(value, dtype=DT).reshape(-1), torch.as_tensor(grad, dtype=DT)


def compare_structure(bt: Any, rs: Any) -> None:
    nb = [(n, tuple(p.shape)) for n, p in bt.named_parameters()]
    nr = [(n, tuple(torch.as_tensor(p).shape)) for n, p in rs.named_parameters()]
    assert nb == nr
    for (_, pb), (_, pr) in zip(bt.named_parameters(), rs.named_parameters(), strict=True):
        CHECK.check("model: default raw parameters", pr, pb.detach(), 0.0)
    assert [tuple(b) for b in rs._rs.raw_parameter_bounds()] == _flat_bounds(bt)


def compare_mll(label: str, bt: Any, rs: Any) -> None:
    vb, gb = bt_mll_value_and_grad(bt)
    vr, gr = rs_mll_value_and_grad(rs)
    CHECK.check(f"{label} MLL", vr, vb, VALUE_TOL)
    check_grad(f"{label} MLL gradient", gr, gb)


def values_and_grads(acqf: Any, X: Any) -> tuple[Any, Any]:
    Xg = X.detach().clone().requires_grad_(True)
    v = acqf(Xg)
    (g,) = torch.autograd.grad(v.sum(), Xg, allow_unused=True)
    return v.detach(), torch.zeros_like(X) if g is None else g


def _moments(post: Any) -> tuple[Any, Any] | None:
    try:
        return post.mean, post.variance
    except NotImplementedError:
        return None


def compare_mc(label: str, make_bt: Any, make_rs: Any, X: Any, num: int = 32) -> None:
    """A Monte-Carlo acquisition function with BoTorch's base samples."""
    sb = BtSampler(sample_shape=torch.Size([num]), seed=0)
    vb, gb = values_and_grads(make_bt(sb), X)
    sr = RsSampler(sample_shape=torch.Size([num]), seed=0)
    sr.base_samples = sb.base_samples.clone()
    vr, gr = values_and_grads(make_rs(sr), X)
    CHECK.check(f"{label}", vr, vb, VALUE_TOL)
    check_grad(f"{label} d / dx", gr, gb)


def compare_posteriors(bt: Any, rs: Any, opts: dict[str, Any], outcome: Any, seed: int) -> None:
    bt.eval()
    m = opts.get("m", 1)
    Xq = query_points(opts, 4, seed)
    # q = 1 posteriors (batch of 4 points), with and without noise.
    for noise in (False, True):
        pb = bt.posterior(Xq.unsqueeze(-2), observation_noise=noise)
        pr = rs.posterior(Xq.unsqueeze(-2), observation_noise=noise)
        mb, mr = _moments(pb), _moments(pr)
        assert (mb is None) == (mr is None)
        if mb is not None and mr is not None:
            tag = " (observation noise)" if noise else ""
            CHECK.check(f"posterior: q=1 mean{tag}", mr[0], mb[0].detach(), VALUE_TOL)
            CHECK.check(f"posterior: q=1 variance{tag}", mr[1], mb[1].detach(), VALUE_TOL)
    # Joint samples of q = 3 points from identical base samples (their input
    # gradients through the Monte-Carlo acquisition functions below; the
    # Rust fixture test compares them directly).
    Xj = query_points(opts, 3, seed + 1)
    with torch.no_grad():
        pb, pr = bt.posterior(Xj), rs.posterior(Xj)
        Z = torch.randn(
            8, *pb.base_sample_shape, generator=torch.Generator().manual_seed(9), dtype=DT
        )
        sb = pb.rsample_from_base_samples(torch.Size([8]), Z)
        sr = pr.rsample_from_base_samples(torch.Size([8]), Z)
    CHECK.check("posterior: samples (identical base samples)", sr, sb, VALUE_TOL)
    if m != 1 or opts.get("append") or opts.get("one_to_many"):
        return
    # Acquisition functions.
    best_f = 2.5 + opts.get("shift", 0.0)
    X1 = query_points(opts, 4, seed + 2).unsqueeze(-2)
    if has_moments(None if outcome == "default" else outcome):
        vb, gb = values_and_grads(BtPosteriorMean(bt), X1)
        vr, gr = values_and_grads(RsPosteriorMean(rs), X1)
        CHECK.check("acquisition: PosteriorMean", vr, vb, VALUE_TOL)
        check_grad("acquisition: PosteriorMean d / dx", gr, gb)
        vb, gb = values_and_grads(BtEI(bt, best_f=best_f), X1)
        vr, gr = values_and_grads(RsEI(rs, best_f=best_f), X1)
        CHECK.check("acquisition: ExpectedImprovement", vr, vb, VALUE_TOL)
        check_grad("acquisition: ExpectedImprovement d / dx", gr, gb)
    X2 = query_points(opts, 4, seed + 3).reshape(2, 2, -1)
    compare_mc(
        "acquisition: qExpectedImprovement",
        lambda s: BtQEI(bt, best_f=best_f, sampler=s),
        lambda s: RsQEI(rs, best_f=best_f, sampler=s),
        X2,
    )
    compare_mc(
        "acquisition: qSimpleRegret",
        lambda s: BtQSR(bt, sampler=s),
        lambda s: RsQSR(rs, sampler=s),
        X2,
    )


# ---------------------------------------------------------------------------
# SingleTaskGP with every transform
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", list(CASES))
def test_model_matches_botorch(name: str) -> None:
    input_spec, outcome_spec, opts = CASES[name]
    seed = 6200 + 7 * list(CASES).index(name)
    with botorch_fixes():
        bt, rs, _, _ = model_pair(input_spec, outcome_spec, opts, seed)
        compare_structure(bt, rs)
        bt.eval()
        x_tf = bt.train_inputs[0].detach()
        if x_tf.ndim == 3:  # m > 1: one copy per output
            x_tf = x_tf[0]
        CHECK.check(
            "model: transformed training inputs",
            torch.as_tensor(list(rs._rs.transformed_train_inputs()[0]), dtype=DT).reshape(
                x_tf.shape
            ),
            x_tf,
            VALUE_TOL,
        )
        compare_mll("model: default", bt, rs)
        randomize(bt, rs, seed + 1)
        compare_mll("model: random parameters,", bt, rs)
        compare_posteriors(bt, rs, opts, outcome_spec, seed + 2)


@pytest.mark.parametrize(
    "name,input_spec,risk",
    [
        (
            "perturbation",
            [
                {
                    "type": "InputPerturbation",
                    "perturbation_set": [[0.05, 0.0], [-0.05, 0.02], [0.0, -0.04]],
                }
            ],
            "cvar",
        ),
        ("append", [{"type": "AppendFeatures", "feature_set": [[0.1], [0.5], [0.9]]}], "mean"),
        (
            "perturbation_append",
            [
                {"type": "InputPerturbation", "perturbation_set": [[0.05, 0.0], [-0.05, 0.02]]},
                {"type": "AppendFeatures", "feature_set": [[0.2], [0.7]], "skip_expand": True},
            ],
            "cvar",
        ),
    ],
)
def test_one_to_many_transforms_in_acquisition_functions(
    name: str, input_spec: Any, risk: str
) -> None:
    """Risk measures over the points of one-to-many input transforms (a log
    outcome transform makes the samples nonlinear)."""
    append = int(any(s["type"] == "AppendFeatures" for s in input_spec))
    opts = {"append": append, "one_to_many": True}
    n_w = 3 if name != "perturbation_append" else 2
    with botorch_fixes():
        bt, rs, _, _ = model_pair(input_spec, [{"type": "Log"}], opts, 6300 + len(name))
        compare_structure(bt, rs)
        compare_mll("one-to-many: default", bt, rs)
        randomize(bt, rs, 6301)
        compare_mll("one-to-many: random parameters,", bt, rs)
        compare_posteriors(bt, rs, opts, [{"type": "Log"}], 6302)
        X = query_points(opts, 6, 6303).reshape(3, 2, -1)
        if risk == "cvar":
            ob, orr = BtCVaR(alpha=0.5, n_w=n_w), RsCVaR(alpha=0.5, n_w=n_w)
        else:
            ob, orr = BtExpectation(n_w=n_w), RsExpectation(n_w=n_w)
        compare_mc(
            f"one-to-many: qSimpleRegret with {type(ob).__name__}",
            lambda s: BtQSR(bt, sampler=s, objective=ob),
            lambda s: RsQSR(rs, sampler=s, objective=orr),
            X,
        )


# ---------------------------------------------------------------------------
# Models with transforms of some outputs
# ---------------------------------------------------------------------------


def test_outputs_subset_matches_single_output_models() -> None:
    """``Log(outputs=[0])`` and ``Standardize(m=2)`` on a two-output model:
    BoTorch cannot untransform the posterior (``NotImplementedError``);
    botorch_rs's equals that of single-output BoTorch models with the same
    parameters (``Chain(Log, Standardize)`` and ``Standardize``)."""
    opts = {"m": 2}
    X, Y, _ = make_data(opts, 6400)
    spec = [{"type": "Log", "outputs": [0]}, {"type": "Standardize", "m": 2}]
    bt2 = BtSingleTaskGP(X, Y, outcome_transform=build_outcome(spec, "bt", 2)).to(DT)
    with pytest.raises(NotImplementedError):
        bt2.posterior(X[:2])
    rs = RsSingleTaskGP(X, Y, outcome_transform=build_outcome(spec, "rs", 2))
    refs = [
        BtSingleTaskGP(
            X,
            Y[:, :1],
            outcome_transform=build_outcome([{"type": "Log"}, {"type": "Standardize"}], "bt", 1),
        ).to(DT),
        BtSingleTaskGP(X, Y[:, 1:]).to(DT),
    ]
    rr = [RsSingleTaskGP(X, Y[:, t : t + 1]) for t in range(2)]
    for t, (b, r) in enumerate(zip(refs, rr, strict=True)):
        randomize(b, r, 6401 + t)
    rs.initialize(
        **{
            name: torch.stack([dict(refs[t].named_parameters())[name].detach() for t in range(2)])
            for name, _ in refs[0].named_parameters()
        }
    )
    compare_mll_list = (bt_mll_value_and_grad(refs[0])[0], bt_mll_value_and_grad(refs[1])[0])
    vr, _ = rs_mll_value_and_grad(rs)
    CHECK.check("outputs subset: MLL per output", vr, torch.cat(compare_mll_list), VALUE_TOL)
    Xq = query_points(opts, 3, 6403)
    pr = rs.posterior(Xq)
    for t, b in enumerate(refs):
        b.eval()
        pb = b.posterior(Xq)
        CHECK.check("outputs subset: mean", pr.mean[..., t], pb.mean[..., 0].detach(), VALUE_TOL)
        CHECK.check(
            "outputs subset: variance", pr.variance[..., t], pb.variance[..., 0].detach(), VALUE_TOL
        )
    Z = torch.randn(8, 3, 2, generator=torch.Generator().manual_seed(6404), dtype=DT)
    sr = pr.rsample_from_base_samples(torch.Size([8]), Z)
    for t, b in enumerate(refs):
        pb = b.posterior(Xq)
        zt = Z[..., t].reshape(8, *pb.base_sample_shape)
        sb = pb.rsample_from_base_samples(torch.Size([8]), zt)
        CHECK.check("outputs subset: samples", sr[..., t], sb[..., 0].detach(), VALUE_TOL)


# ---------------------------------------------------------------------------
# ModelListGP
# ---------------------------------------------------------------------------


def _list_pair(seed: int, kind: str = "ModelListGP") -> tuple[Any, Any]:
    X, Y, _ = make_data({"m": 2}, seed)
    specs = [[{"type": "Log"}], [{"type": "Bilog"}, {"type": "Standardize"}]]
    bts, rss = [], []
    for t, spec in enumerate(specs):
        y = Y[:, t : t + 1] if t == 0 else -Y[:, t : t + 1]
        b = BtSingleTaskGP(X, y, outcome_transform=build_outcome(spec, "bt", 1)).to(DT)
        r = RsSingleTaskGP(X, y, outcome_transform=build_outcome(spec, "rs", 1))
        randomize(b, r, seed + 1 + t)
        bts.append(b)
        rss.append(r)
    if kind == "ModelList":
        from botorch.models import ModelList as BtModelList

        from botorch_rs.models import ModelList as RsModelList

        return BtModelList(*bts), RsModelList(*rss)
    return BtModelListGP(*bts), RsModelListGP(*rss)


@pytest.mark.parametrize("kind", ["ModelListGP", "ModelList"])
def test_model_list_with_transformed_outputs(kind: str) -> None:
    """``ModelListGP`` and (phase 7b's) ``ModelList`` of transformed models."""
    bl, rl = _list_pair(6500, kind)
    bl.eval()
    Xq = query_points({}, 3, 6503)
    pb, pr = bl.posterior(Xq), rl.posterior(Xq)
    assert type(pr).__name__ == type(pb).__name__ == "PosteriorList"
    # The sub-model with Bilog has no moments: neither has the list.
    with pytest.raises(NotImplementedError):
        pb.mean  # noqa: B018
    with pytest.raises(NotImplementedError):
        pr.mean  # noqa: B018
    CHECK.check(
        "ModelListGP: Log sub-posterior mean",
        pr.posteriors[0].mean,
        pb.posteriors[0].mean.detach(),
        VALUE_TOL,
    )
    # Monte-Carlo acquisition functions with a ListSampler.
    X = query_points({}, 4, 6504).reshape(2, 2, -1)
    w = torch.tensor([0.7, -0.3], dtype=DT)

    def bt_acq(_: Any) -> Any:
        s = BtListSampler(*(BtSampler(sample_shape=torch.Size([16]), seed=t) for t in range(2)))
        bt_acq.sampler = s  # type: ignore[attr-defined]
        return BtQSR(bl, sampler=s, objective=BtLinearMCObjective(w))

    vb, gb = values_and_grads(bt_acq(None), X)
    # botorch_rs's sampler of a ModelListGP: the base samples of all outputs
    # (stacked like those of BoTorch's multi-output posteriors).
    sr = RsSampler(sample_shape=torch.Size([16]), seed=0)
    sr.base_samples = torch.stack(
        [sub.base_samples for sub in bt_acq.sampler.samplers],  # type: ignore[attr-defined]
        dim=-1,
    )
    acq = RsQSR(rl, sampler=sr, objective=RsLinearMCObjective(w))
    vr, gr = values_and_grads(acq, X)
    CHECK.check("ModelListGP: qSimpleRegret (ListSampler)", vr, vb, VALUE_TOL)
    check_grad("ModelListGP: qSimpleRegret d / dx", gr, gb)


def test_model_list_scalarized_posterior_at_one_point() -> None:
    """BoTorch's ``scalarize_posterior`` of a ``PosteriorList`` with
    moments (``Log`` and ``Standardize`` sub-models) moment-matches at
    ``q = 1``."""
    X, Y, _ = make_data({"m": 2}, 6600)
    bts, rss = [], []
    for t, spec in enumerate([[{"type": "Log"}], None]):
        kb = {} if spec is None else {"outcome_transform": build_outcome(spec, "bt", 1)}
        kr = {} if spec is None else {"outcome_transform": build_outcome(spec, "rs", 1)}
        b = BtSingleTaskGP(X, Y[:, t : t + 1], **kb).to(DT)
        r = RsSingleTaskGP(X, Y[:, t : t + 1], **kr)
        randomize(b, r, 6601 + t)
        bts.append(b)
        rss.append(r)
    bl, rl = BtModelListGP(*bts), RsModelListGP(*rss)
    w = torch.tensor([0.6, 0.4], dtype=DT)
    X1 = query_points({}, 4, 6603).unsqueeze(-2)
    vb, gb = values_and_grads(BtEI(bl, best_f=2.0, posterior_transform=BtScalarized(w)), X1)
    vr, gr = values_and_grads(RsEI(rl, best_f=2.0, posterior_transform=RsScalarized(w)), X1)
    CHECK.check("ModelListGP: EI with ScalarizedPosteriorTransform", vr, vb, VALUE_TOL)
    check_grad("ModelListGP: EI with ScalarizedPosteriorTransform d / dx", gr, gb)


# ---------------------------------------------------------------------------
# Conditioning and fantasies
# ---------------------------------------------------------------------------

CONDITION_CASES: dict[str, tuple[Any, Any, dict[str, Any]]] = {
    "log": (None, [{"type": "Log"}], {}),
    "log_std": (None, [{"type": "Log"}, {"type": "Standardize"}], {}),
    "stratified": (
        None,
        [{"type": "StratifiedStandardize", "stratification_idx": 2, "task_values": [0, 1, 2]}],
        {"tasks": True, "d": 3},
    ),
    "normalize_warp": (
        [{"type": "Normalize", "d": 2}, {"type": "Warp", "d": 2, "indices": [0, 1]}],
        "default",
        {"scale_x": 3.0},
    ),
    "perturbation": (
        [{"type": "InputPerturbation", "perturbation_set": [[0.05, 0.0], [-0.05, 0.02]]}],
        [{"type": "Log"}],
        {},
    ),
}


@pytest.mark.parametrize("name", list(CONDITION_CASES))
def test_conditioning_and_fantasies_match_botorch(name: str) -> None:
    input_spec, outcome_spec, opts = CONDITION_CASES[name]
    with botorch_fixes():
        bt, rs, _, _ = model_pair(input_spec, outcome_spec, opts, 6700)
        randomize(bt, rs, 6701)
        bt.eval()
        Xq = query_points(opts, 3, 6702)
        Xn = query_points(opts, 2, 6703)
        perturbed = name == "perturbation"
        # GPyTorch's prediction caches, which conditioning needs (botorch_rs
        # raises GPyTorch's error without a prediction too).
        bt.posterior(Xq)
        rs.posterior(Xq)
        if not perturbed:
            Yn = torch.tensor([[2.9], [2.4]], dtype=DT)
            cb = bt.condition_on_observations(Xn, Yn)
            cr = rs.condition_on_observations(Xn, Yn)
            pb, pr = cb.posterior(Xq), cr.posterior(Xq)
            CHECK.check("conditioning: posterior mean", pr.mean, pb.mean.detach(), VALUE_TOL)
            CHECK.check(
                "conditioning: posterior variance", pr.variance, pb.variance.detach(), VALUE_TOL
            )
        sb = BtSampler(sample_shape=torch.Size([3]), seed=0)
        fb = bt.fantasize(Xn, sb)
        sr = RsSampler(sample_shape=torch.Size([3]), seed=0)
        sr.base_samples = sb.base_samples.clone()
        fr = rs.fantasize(Xn, sr)
        pb, pr = fb.posterior(Xq), fr.posterior(Xq)
        CHECK.check("fantasies: posterior mean", pr.mean, pb.mean.detach(), VALUE_TOL)
        CHECK.check("fantasies: posterior variance", pr.variance, pb.variance.detach(), VALUE_TOL)


def test_stratified_conditioning_uses_the_transformed_inputs() -> None:
    """BoTorch fits ``StratifiedStandardize`` on the transformed training
    inputs and untransforms posteriors at transformed inputs, but
    ``BatchedMultiOutputGPyTorchModel.condition_on_observations`` transforms
    the new observations with the raw inputs: with an input transform that
    changes the stratification column, the new observations are
    standardized with the wrong stratum (a BoTorch bug, see
    ``docs/drafts/6b-transforms.md``). botorch_rs uses the transformed inputs
    throughout; the reference is BoTorch's transform at the transformed
    inputs."""
    from botorch.models.transforms.input import AffineInputTransform as BtAffine
    from botorch.models.transforms.outcome import StratifiedStandardize as BtStratified

    from botorch_rs.models.transforms.input import AffineInputTransform as RsAffine
    from botorch_rs.models.transforms.outcome import StratifiedStandardize as RsStratified

    g = torch.Generator().manual_seed(6750)
    X = torch.rand(12, 3, generator=g, dtype=DT)
    X[:, 2] = (torch.arange(12) % 3).to(DT)
    Y = torch.randn(12, 1, generator=g, dtype=DT) + 3 * X[:, 2:]
    # The input transform doubles the task column: strata 0, 2, 4.
    coef, off = torch.tensor([1.0, 1.0, 0.5], dtype=DT), torch.zeros(3, dtype=DT)
    tasks = [0, 1, 2, 3, 4]
    bt = BtSingleTaskGP(
        X,
        Y,
        input_transform=BtAffine(d=3, coefficient=coef, offset=off),
        outcome_transform=BtStratified(stratification_idx=2, all_task_values=torch.tensor(tasks)),
    ).to(DT)
    rs = RsSingleTaskGP(
        X,
        Y,
        input_transform=RsAffine(d=3, coefficient=coef, offset=off),
        outcome_transform=RsStratified(stratification_idx=2, all_task_values=tasks),
    )
    copy_params(bt, rs)
    bt.eval()
    Xq = X[:3].clone()
    CHECK.check(
        "stratified: posterior (transformed inputs)",
        rs.posterior(Xq).mean,
        bt.posterior(Xq).mean.detach(),
        VALUE_TOL,
    )
    Xn = torch.tensor([[0.3, 0.4, 1.0]], dtype=DT)
    Yn = torch.tensor([[3.5]], dtype=DT)
    expected = bt.outcome_transform(Yn, X=bt.transform_inputs(Xn))[0].reshape(-1)
    raw = bt.outcome_transform(Yn, X=Xn)[0].reshape(-1)
    cb = bt.condition_on_observations(Xn, Yn)
    cr = rs.condition_on_observations(Xn, Yn)
    CHECK.record(
        "stratified: BoTorch's conditioning error (raw inputs)",
        (cb.train_targets[-1:] - expected).abs().max().item(),
        None,
    )
    assert torch.equal(cb.train_targets[-1:].detach(), raw.detach())
    CHECK.check(
        "stratified: conditioned targets (transformed inputs)",
        torch.as_tensor(cr.train_targets, dtype=DT)[-1:],
        expected.detach(),
        VALUE_TOL,
    )


# ---------------------------------------------------------------------------
# Transforms on the models of phases 7a and 7b
# ---------------------------------------------------------------------------


def _phase7_data(seed: int, categorical: bool = False) -> tuple[Any, Any, Any]:
    """Data with a fidelity (or categorical) last column, and query points."""
    g = torch.Generator().manual_seed(seed)
    X = torch.rand(14, 3, generator=g, dtype=DT)
    Xq = torch.rand(4, 3, generator=g, dtype=DT)
    if categorical:
        X[:, 0] = 3.0 * X[:, 0]
        X[:, 2] = (torch.arange(14) % 3).to(DT)
        Xq[:, 0] = 3.0 * Xq[:, 0]
        Xq[:, 2] = torch.tensor([0.0, 1.0, 2.0, 1.0], dtype=DT)
    else:
        X[:, 2] = torch.tensor([0.0, 0.5, 1.0] * 4 + [0.0, 1.0], dtype=DT)
        Xq[:, 2] = torch.tensor([0.0, 1.0, 0.5, 1.0], dtype=DT)
    Y = torch.sin(3 * X[:, :1]) + X[:, 1:2] ** 2 + 2.0
    Y = Y + 0.05 * torch.randn(14, 1, generator=g, dtype=DT)
    return X, Y, Xq


def _phase7_models() -> dict[str, tuple[Any, bool]]:
    """name -> (model factory taking ``(lib, X, Y)``, categorical data)."""
    from botorch.models import MixedSingleTaskGP as BtMixed
    from botorch.models import SingleTaskMultiFidelityGP as BtMF
    from botorch.models.additive_gp import OrthogonalAdditiveGP as BtOAK
    from botorch.models.map_saas import AdditiveMapSaasSingleTaskGP as BtAdditiveSaas
    from botorch.models.map_saas import get_map_saas_model as bt_map_saas

    from botorch_rs.models import MixedSingleTaskGP as RsMixed
    from botorch_rs.models import SingleTaskMultiFidelityGP as RsMF
    from botorch_rs.models.additive_gp import OrthogonalAdditiveGP as RsOAK
    from botorch_rs.models.map_saas import AdditiveMapSaasSingleTaskGP as RsAdditiveSaas
    from botorch_rs.models.map_saas import get_map_saas_model as rs_map_saas

    def tf(spec: Any, lib: str) -> Any:
        return build_input(spec, lib)

    def outcome(spec: Any, lib: str) -> Any:
        return build_outcome(spec, lib, 1)

    log = [{"type": "Log"}]
    bilog_std = [{"type": "Bilog"}, {"type": "Standardize"}]
    warp01 = [{"type": "Warp", "d": 3, "indices": [0, 1]}]
    norm01 = [{"type": "Normalize", "d": 3, "indices": [0, 1]}]
    std = [{"type": "InputStandardize", "d": 3}]
    cls = {"bt": (BtMF, BtMixed, bt_map_saas, BtAdditiveSaas, BtOAK)}
    cls["rs"] = (RsMF, RsMixed, rs_map_saas, RsAdditiveSaas, RsOAK)
    return {
        "multi_fidelity_log": (
            lambda lib, X, Y: cls[lib][0](
                X, Y, data_fidelities=[2], outcome_transform=outcome(log, lib)
            ),
            False,
        ),
        "multi_fidelity_warp_bilog": (
            lambda lib, X, Y: cls[lib][0](
                X,
                Y,
                iteration_fidelity=2,
                input_transform=tf(warp01, lib),
                outcome_transform=outcome(bilog_std, lib),
            ),
            False,
        ),
        "mixed_normalize_indices_log": (
            lambda lib, X, Y: cls[lib][1](
                X,
                Y,
                cat_dims=[2],
                input_transform=tf(norm01, lib),
                outcome_transform=outcome(log, lib),
            ),
            True,
        ),
        "map_saas_input_standardize_log": (
            lambda lib, X, Y: cls[lib][2](
                X, Y, input_transform=tf(std, lib), outcome_transform=outcome(log, lib)
            ),
            False,
        ),
        "additive_map_saas_power": (
            lambda lib, X, Y: cls[lib][3](
                X, Y, num_taus=2, outcome_transform=outcome([{"type": "Power", "power": 0.5}], lib)
            ),
            False,
        ),
        "orthogonal_additive_log": (
            lambda lib, X, Y: cls[lib][4](X, Y, outcome_transform=outcome(log, lib)),
            False,
        ),
    }


PHASE7_MODELS = [
    "multi_fidelity_log",
    "multi_fidelity_warp_bilog",
    "mixed_normalize_indices_log",
    "map_saas_input_standardize_log",
    "additive_map_saas_power",
    "orthogonal_additive_log",
]


@pytest.mark.parametrize("name", PHASE7_MODELS)
def test_transforms_on_phase7_models_match_botorch(name: str) -> None:
    """Input and outcome transforms on the multi-fidelity, mixed, MAP SAAS
    and orthogonal additive models (``SingleTaskGP`` subclasses, which
    BoTorch builds through ``SingleTaskGP.__init__`` too). BoTorch's MLLs are
    evaluated with GPyTorch's ``sq_dist`` diagonal zeroed, as in
    ``test_parity_7b_models.py``."""
    from kernel_tree_helpers import zero_diagonal_distances

    make, categorical = _phase7_models()[name]
    seed = 7500 + PHASE7_MODELS.index(name)
    X, Y, Xq = _phase7_data(seed, categorical)
    torch.manual_seed(seed)
    bt = make("bt", X, Y).to(DT)
    torch.manual_seed(seed)
    rs = make("rs", X, Y)
    compare_structure(bt, rs)
    with zero_diagonal_distances():
        compare_mll(f"phase 7 models ({name}): default", bt, rs)
        randomize(bt, rs, seed + 1)
        compare_mll(f"phase 7 models ({name}): random parameters,", bt, rs)
    bt.eval()
    pb, pr = bt.posterior(Xq.unsqueeze(-2)), rs.posterior(Xq.unsqueeze(-2))
    mb, mr = _moments(pb), _moments(pr)
    assert (mb is None) == (mr is None)
    if mb is not None and mr is not None:
        CHECK.check("phase 7 models: q=1 mean", mr[0], mb[0].detach(), VALUE_TOL)
        CHECK.check("phase 7 models: q=1 variance", mr[1], mb[1].detach(), VALUE_TOL)
    with torch.no_grad():
        pj, pjr = bt.posterior(Xq), rs.posterior(Xq)
        Z = torch.randn(
            8, *pj.base_sample_shape, generator=torch.Generator().manual_seed(seed), dtype=DT
        )
        CHECK.check(
            "phase 7 models: samples (identical base samples)",
            pjr.rsample_from_base_samples(torch.Size([8]), Z),
            pj.rsample_from_base_samples(torch.Size([8]), Z),
            VALUE_TOL,
        )
    # The query points' fidelities are 0 and 1, the bounds of the clamp of
    # the LinearTruncatedFidelityKernel: BoTorch's input gradients follow
    # torch ≥ 2.14's subgradients there, as botorch_rs's do, also under
    # older torch (see test_parity_clamp_bounds.py).
    with torch214_clamp_subgradients():
        compare_mc(
            "phase 7 models: qExpectedImprovement",
            lambda s: BtQEI(bt, best_f=2.5, sampler=s),
            lambda s: RsQEI(rs, best_f=2.5, sampler=s),
            Xq.reshape(2, 2, -1),
        )
        if mb is not None:
            vb, gb = values_and_grads(BtEI(bt, best_f=2.5), Xq.unsqueeze(-2))
            vr, gr = values_and_grads(RsEI(rs, best_f=2.5), Xq.unsqueeze(-2))
            CHECK.check("phase 7 models: ExpectedImprovement", vr, vb, VALUE_TOL)
            check_grad("phase 7 models: ExpectedImprovement d / dx", gr, gb)


def test_unsupported_transforms_on_phase7_models() -> None:
    """Transforms that the phase-7 models do not support raise
    ``UnsupportedError`` (a ``NotImplementedError``): an outcome transform
    other than ``Standardize`` on an ``EnsembleMapSaasSingleTaskGP`` (whose
    posterior BoTorch cannot build either: a ``TransformedPosterior`` has no
    ``distribution``), and transforms other than ``Normalize`` /
    ``Standardize`` on the multi-task GPs (supported by BoTorch, not ported)."""
    from botorch.models.map_saas import EnsembleMapSaasSingleTaskGP as BtEnsembleSaas

    from botorch_rs.exceptions.errors import UnsupportedError
    from botorch_rs.models import KroneckerMultiTaskGP as RsKronecker
    from botorch_rs.models import MultiTaskGP as RsMultiTask
    from botorch_rs.models.map_saas import EnsembleMapSaasSingleTaskGP as RsEnsembleSaas

    X, Y, Xq = _phase7_data(7600)
    torch.manual_seed(0)
    bt = BtEnsembleSaas(
        X, Y, num_taus=2, outcome_transform=build_outcome([{"type": "Log"}], "bt", 1)
    )
    with pytest.raises(AttributeError):
        bt.posterior(Xq)
    torch.manual_seed(0)
    rs = RsEnsembleSaas(
        X, Y, num_taus=2, outcome_transform=build_outcome([{"type": "Log"}], "rs", 1)
    )
    with pytest.raises(UnsupportedError, match="Standardize outcome transform only"):
        rs.posterior(Xq)
    Xt = X.clone()
    Xt[:, 2] = (torch.arange(14) % 2).to(DT)
    with pytest.raises(UnsupportedError, match="not supported by MultiTaskGP"):
        RsMultiTask(
            Xt, Y, task_feature=-1, outcome_transform=build_outcome([{"type": "Log"}], "rs", 1)
        )
    with pytest.raises(UnsupportedError, match="not supported by MultiTaskGP"):
        RsMultiTask(
            Xt,
            Y,
            task_feature=-1,
            input_transform=build_input([{"type": "Warp", "d": 3, "indices": [0]}], "rs"),
        )
    with pytest.raises(UnsupportedError, match="center=0.5"):
        RsMultiTask(
            Xt,
            Y,
            task_feature=-1,
            input_transform=build_input([{"type": "Normalize", "d": 3, "center": 0.0}], "rs"),
        )
    Y2 = torch.cat([Y, -Y], -1)
    with pytest.raises(UnsupportedError, match="Standardize"):
        RsKronecker(X, Y2, outcome_transform=build_outcome([{"type": "Log"}], "rs", 2))


# ---------------------------------------------------------------------------
# Batched models
# ---------------------------------------------------------------------------


def _batched_data(seed: int) -> tuple[Any, Any]:
    g = torch.Generator().manual_seed(seed)
    X = torch.rand(2, 10, 2, generator=g, dtype=DT) * 3.0 - 1.0
    Y = torch.sin(2 * X).sum(-1, keepdim=True) + 0.05 * torch.randn(2, 10, 1, generator=g, dtype=DT)
    return X, Y


@pytest.mark.parametrize("kind", ["warp", "broadcasted"])
def test_batched_models_match_botorch(kind: str) -> None:
    X, Y = _batched_data(6800)
    if kind == "warp":
        spec = [
            {
                "type": "Warp",
                "d": 2,
                "indices": [0, 1],
                "batch_shape": [2],
                "bounds": [[-1.0, -1.0], [2.0, 2.0]],
            }
        ]
    else:
        spec = [
            {
                "type": "BatchBroadcastedInputTransform",
                "transforms": [
                    [{"type": "Normalize", "d": 2}],
                    [{"type": "InputStandardize", "d": 2}],
                ],
            }
        ]
    with botorch_fixes():
        bt = BtSingleTaskGP(X, Y, input_transform=build_input(spec, "bt")).to(DT)
        rs = RsSingleTaskGP(X, Y, input_transform=build_input(spec, "rs"))
        compare_structure(bt, rs)
        compare_mll(f"batched {kind}: default", bt, rs)
        randomize(bt, rs, 6801)
        compare_mll(f"batched {kind}: random parameters,", bt, rs)
        bt.eval()
        Xq = torch.rand(2, 3, 2, generator=torch.Generator().manual_seed(6802), dtype=DT) * 3 - 1
        pb, pr = bt.posterior(Xq), rs.posterior(Xq)
        CHECK.check(f"batched {kind}: posterior mean", pr.mean, pb.mean.detach(), VALUE_TOL)
        CHECK.check(
            f"batched {kind}: posterior variance", pr.variance, pb.variance.detach(), VALUE_TOL
        )


# ---------------------------------------------------------------------------
# Fits
# ---------------------------------------------------------------------------


def _bt_fit(bt: Any) -> tuple[float, int]:
    import botorch.optim.fit as bt_optim_fit

    record: dict[str, Any] = {"attempts": 0}
    orig = bt_optim_fit.fit_gpytorch_mll_scipy

    def recorder(mll: Any, *args: Any, **kwargs: Any) -> Any:
        record["attempts"] += 1
        record["r"] = orig(mll, *args, **kwargs)
        return record["r"]

    bt_fit(BtMLL(bt.likelihood, bt), optimizer=recorder)
    return float(record["r"].fval), record["attempts"]


def _rs_fit(rs: Any) -> tuple[float, int]:
    mll = RsMLL(rs.likelihood, rs)
    rs_fit(mll)
    return float(mll._fit_result["fval"]), int(mll._fit_result["attempts"])


def _quantiles(v: list[float]) -> tuple[float, float, float]:
    q = statistics.quantiles(sorted(v), n=4, method="inclusive")
    return min(v), q[1], q[2]


def _twin(X0: Any, t: int) -> Any:
    if t == 0:
        return X0
    g = torch.Generator().manual_seed(t)
    rel = TWIN_SCALES[(t - 1) % len(TWIN_SCALES)]
    return X0 * (1 + rel * torch.randn(X0.shape, generator=g, dtype=DT))


@pytest.mark.parametrize("name", [n for n, (_, _, o) in CASES.items() if o.get("fit")])
def test_fit_from_identical_initial_values(name: str) -> None:
    input_spec, outcome_spec, opts = CASES[name]
    seed = 6900 + list(CASES).index(name)
    X0 = make_data(opts, seed)[0]
    bt_out: list[tuple[float, int]] = []
    rs_out: list[tuple[float, int]] = []

    def run(twins: range) -> None:
        for t in twins:
            bt, rs, _, _ = model_pair(input_spec, outcome_spec, opts, seed, X=_twin(X0, t))
            copy_params(bt, rs)
            bt_out.append(_bt_fit(bt))
            rs_out.append(_rs_fit(rs))

    with botorch_fixes():
        run(range(N_TWINS + 1))
        first = [loss for loss, a in bt_out if a == 1]
        if max(first) - min(first) > CHAOTIC_RANGE:
            run(range(N_TWINS + 1, CHAOTIC_TWINS + 1))
    bt_first = [loss for loss, a in bt_out if a == 1]
    rs_first = [loss for loss, a in rs_out if a == 1]
    best, worst = min(bt_first), max(bt_first)
    spread = max(v for v in bt_first if v <= best + FIT_CLUSTER) - best
    tol = max(FIT_TOL, 10 * spread)
    CHECK.record("fit: BoTorch's spread at its best optimum", spread, None)
    if worst - best > CHAOTIC_RANGE:
        bq, rq = _quantiles(bt_first), _quantiles(rs_first)
        CHECK.check_bound("fit (chaotic): best above BoTorch's median", rq[0] - bq[1], FIT_TOL)
        CHECK.check_bound(
            "fit (chaotic): median above BoTorch's upper quartile", rq[1] - bq[2], FIT_TOL
        )
    else:
        CHECK.check_bound(
            "fit: best loss above BoTorch's best (scaled)", (min(rs_first) - best) / tol, 1.0
        )
        if rs_out[0][1] == 1:
            CHECK.check_bound(
                "fit: loss above BoTorch's worst (scaled)", (rs_out[0][0] - worst) / tol, 1.0
            )


# ---------------------------------------------------------------------------
# Randomized sweep over random transform chains
# ---------------------------------------------------------------------------


@pytest.mark.equivalence
@pytest.mark.parametrize("seed", range(48))
def test_random_transform_chains_match_botorch(seed: int) -> None:
    r = random.Random(7000 + seed)
    d = r.choice([2, 3])
    input_spec, _, hints = random_input_spec(r, d)
    outcome_spec = random_outcome_spec(r)
    opts: dict[str, Any] = {"d": d}
    if hints.get("log10"):
        opts["shift_x"] = 0.5
    if any(s["type"] == "Round" for s in input_spec):
        opts["int_col"] = input_spec[0]["integer_indices"][0]
    outcome = "default" if outcome_spec is None and r.random() < 0.5 else outcome_spec
    with botorch_fixes():
        bt, rs, _, _ = model_pair(input_spec, outcome, opts, 7100 + seed)
        compare_structure(bt, rs)
        compare_mll("sweep: default", bt, rs)
        randomize(bt, rs, 7200 + seed)
        compare_mll("sweep: random parameters,", bt, rs)
        compare_posteriors(bt, rs, opts, outcome, 7300 + seed)

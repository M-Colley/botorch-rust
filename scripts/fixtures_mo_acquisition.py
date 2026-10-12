"""Phase 5B fixtures: BoTorch's multi-objective acquisition functions.

Written to ``<out>/phase5b/`` by ``generate_fixtures.py --phase 5b`` (with
BoTorch main) and ``--phase 5b-3392`` (with a BoTorch that includes
meta-pytorch/botorch#3392):

* ``<model>.json`` for four models with explicit hyperparameters (the
  phase-4 builders): multi-output ``SingleTaskGP`` models with m = 2 (RBF),
  m = 3 (Matern-5/2, ``Normalize``) and m = 4 (fixed noise), and a
  ``ModelListGP`` of a two-output and a one-output model. Each holds

  - the multi-output objectives (identity with ``outcomes``, weighted,
    generic, feasibility-weighted) on random samples;
  - values and gradients (t-batches of two) of qEHVI and qLogEHVI (q = 1, 2,
    3, pending points, outcome constraints with ``eta``, ``fat``,
    ``tau_relu``, objectives, exact and approximate (``alpha``)
    partitionings, whose cell bounds are recorded);
  - values and gradients of qNEHVI and qLogNEHVI (``cache_root`` true /
    false, incremental or not, ``alpha``, constraints, objectives, pending
    points at construction, cached or not (``max_iep``), ``prune_baseline``)
    together with the cached baseline, the number of cells, the per-sample
    hypervolumes and the improvement of cached pending points;
  - the analytic ``ExpectedHypervolumeImprovement``;
  - qLogNParEGO with explicit weights (``cache_root=False``, the plain
    computation botorch_rs implements);
  - ``prune_inferior_points_multi_objective`` (``max_frac``, constraints,
    objectives).

* ``misc.json``: ``get_default_partitioning_alpha`` and
  ``compute_subset_indices``.
* ``pending_3392.json`` (``--phase 5b-3392``): qNEHVI and qLogNEHVI after
  sequences of ``set_X_pending`` calls (append, append more, replace,
  clear, set again; incremental or not) and the feasibility-weighted
  objective with two constraint indices: the cases in which botorch_rs
  implements meta-pytorch/botorch#3392 and BoTorch main differs.

Base samples: every draw of BoTorch's samplers
(``draw_sobol_normal_samples`` of ``d`` jointly sampled values) is recorded.
All samplers of a file have the same seed and number of samples, so a draw
depends only on ``d`` (checked) and the file holds one ``draws`` list; the
Rust tests install the draws in a sampler with the same number of samples.
BoTorch's unseeded pruning sampler (``get_sampler``) is replaced by a seeded
sampler of 64 samples whose base samples are recorded as well
(``prune_base_samples``).

Noise floors: the joint posteriors of the noisy acquisition functions over
their baseline (the training points) are nearly singular. The NEHVI and
qLogNParEGO cases record BoTorch's own change of the values and gradients
when the training inputs are perturbed by 1e-14 (relative, two copies);
the Rust tests allow 10 times that if it exceeds the nominal tolerance.

BoTorch is evaluated with dense ``ModelListGP`` covariances (linear_operator
evaluates the block-diagonal slicing of multi-output sub-models
incorrectly; see ``_DenseSeparateMTMVN`` in ``generate_fixtures.py``).
Non-finite numbers are written as the strings "inf", "-inf" and "nan".
"""

from __future__ import annotations

import contextlib
import copy
import functools
import inspect
import json
import math
import os
import sys
import warnings
from collections.abc import Iterator
from typing import Any

import torch

DT = torch.float64
NUM_SAMPLES = 16
PRUNE_SAMPLES = 64
PRUNE_SEED = 77

MODEL_CASES = [
    dict(name="m2_rbf_d2_n10", kind="single_task", models=[dict(d=2, n=10, m=2)]),
    dict(
        name="m3_matern_d3_n10_normalize",
        kind="single_task",
        models=[dict(d=3, n=10, m=3, covar="matern", normalize="shared", x_range=(-1.0, 2.0))],
    ),
    dict(name="m4_fixed_noise_d2_n8", kind="single_task", models=[dict(d=2, n=8, m=4, yvar=True)]),
    dict(
        name="list_2_1_d2",
        kind="model_list",
        models=[dict(d=2, n=9, m=2), dict(d=2, n=7, m=1, normalize="shared", seed_offset=1)],
    ),
]

# (acquisition function, q, pending points, constraints, objective,
#  partitioning, extra keyword arguments)
EHVI_CASES = [
    ("qExpectedHypervolumeImprovement", 1, 0, None, "identity", "fast", {}),
    ("qExpectedHypervolumeImprovement", 2, 1, "one", "identity", "fast", {"fat": True}),
    ("qExpectedHypervolumeImprovement", 3, 0, "two", "weighted", "nondominated", {}),
    ("qLogExpectedHypervolumeImprovement", 1, 0, None, "identity", "fast", {}),
    ("qLogExpectedHypervolumeImprovement", 2, 0, "one", "outcomes", "fast", {"fat": False}),
    ("qLogExpectedHypervolumeImprovement", 3, 1, "two", "generic", "nondominated", {}),
    ("qLogExpectedHypervolumeImprovement", 2, 2, None, "feasibility", "fast", {"tau_relu": 1e-3}),
]

# (log, q, cache_root, incremental, alpha, constraints, objective, pending
#  points, max_iep, prune_baseline)
NEHVI_CASES = [
    (False, 1, True, True, 0.0, None, "identity", 0, 0, False),
    (False, 2, False, True, 0.0, "one", "identity", 0, 0, False),
    (False, 2, True, False, 0.0, None, "weighted", 0, 0, False),
    (True, 1, True, True, 0.0, None, "identity", 0, 0, False),
    (True, 1, False, True, 0.0, None, "identity", 0, 0, False),
    (True, 2, True, True, 1e-2, "two", "generic", 0, 0, False),
    (True, 3, True, False, 0.0, "one", "outcomes", 0, 0, False),
    (True, 2, False, False, 0.0, None, "feasibility", 0, 0, False),
    # Pending points at construction: cached (incremental) or not.
    (True, 1, True, True, 0.0, None, "identity", 2, 0, False),
    (False, 2, True, True, 0.0, "one", "identity", 2, 0, False),
    (True, 2, True, False, 0.0, None, "identity", 1, 2, False),
    (False, 1, False, True, 0.0, None, "weighted", 2, 5, False),
    # Pruned baselines.
    (True, 2, True, True, 0.0, None, "identity", 0, 0, True),
    (False, 1, True, False, 0.0, "one", "identity", 0, 0, True),
]

PAREGO_CASES = [
    # (variant, q, incremental)
    ("plain", 2, True),
    ("constraints+pending", 2, True),
    ("weighted", 1, False),
    ("generic", 2, True),
]

PRUNE_CASES = [
    {},
    {"max_frac": 0.3},
    {"constraints": "one"},
    {"objective": "weighted"},
]

PENDING_STEPS = [
    ("append", lambda P: P[:1]),
    ("append more", lambda P: P[:3]),
    ("replace", lambda P: P[3:5]),
    ("clear", lambda P: None),
    ("set again", lambda P: P[1:2]),
]


def _gf() -> Any:
    """The shared helpers of ``generate_fixtures.py``."""
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import generate_fixtures

    return generate_fixtures


def enc(x: Any) -> Any:
    """JSON-compatible values: tensors as lists, non-finite floats as strings."""
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().tolist()
    if isinstance(x, float):
        if math.isnan(x):
            return "nan"
        if math.isinf(x):
            return "inf" if x > 0 else "-inf"
        return x
    if isinstance(x, dict):
        return {k: enc(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [enc(v) for v in x]
    return x


# ---------------------------------------------------------------------------
# BoTorch instrumentation
# ---------------------------------------------------------------------------


@contextlib.contextmanager
def recorded_draws(record: dict[tuple[int, int], torch.Tensor]) -> Iterator[None]:
    """Records BoTorch's base-sample draws by (number of samples, ``d``); a
    repeated draw must be identical."""
    import botorch.sampling.normal as bt_normal

    orig = bt_normal.draw_sobol_normal_samples

    def draw(d: int, n: int, device: Any = None, dtype: Any = None, seed: Any = None) -> Any:
        z = orig(d=d, n=n, device=device, dtype=dtype, seed=seed)
        prev = record.get((n, d))
        if prev is None:
            record[(n, d)] = z.detach().clone()
        else:
            assert torch.equal(prev, z), f"conflicting base samples for d={d}"
        return z

    bt_normal.draw_sobol_normal_samples = draw
    try:
        yield
    finally:
        bt_normal.draw_sobol_normal_samples = orig


@contextlib.contextmanager
def seeded_pruning(calls: list[int]) -> Iterator[None]:
    """BoTorch's pruning sampler (``get_sampler``, unseeded) is replaced by a
    seeded sampler of ``PRUNE_SAMPLES`` samples."""
    import botorch.acquisition.utils as bt_au
    from botorch.sampling.normal import SobolQMCNormalSampler

    orig = bt_au.get_sampler

    def get_sampler(posterior: Any, sample_shape: Any, **kwargs: Any) -> Any:
        calls.append(1)
        return SobolQMCNormalSampler(sample_shape=torch.Size([PRUNE_SAMPLES]), seed=PRUNE_SEED)

    bt_au.get_sampler = get_sampler
    try:
        yield
    finally:
        bt_au.get_sampler = orig


@contextlib.contextmanager
def no_lower_bound() -> Iterator[None]:
    """Fails if qLogNEI estimates a lower bound of the objective (it draws
    random points)."""
    import botorch.acquisition.utils as bt_au

    orig = bt_au._estimate_objective_lower_bound

    def fail(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("the baseline is infeasible in a sample")

    bt_au._estimate_objective_lower_bound = fail
    try:
        yield
    finally:
        bt_au._estimate_objective_lower_bound = orig


@contextlib.contextmanager
def dense_model_list() -> Iterator[None]:
    """``ModelListGP`` posteriors from dense covariances (wrapped in a linear
    operator, so that torch does not factorize them eagerly without
    jitter)."""
    import botorch.models.gpytorch as gpytorch_models
    from gpytorch.distributions import MultitaskMultivariateNormal
    from linear_operator.operators import DenseLinearOperator

    orig = gpytorch_models.separate_mtmvn

    def separate(mvn: Any) -> Any:
        dense = MultitaskMultivariateNormal(
            mvn.mean,
            DenseLinearOperator(mvn.lazy_covariance_matrix.to_dense()),
            interleaved=mvn._interleaved,
        )
        return orig(dense)

    gpytorch_models.separate_mtmvn = separate
    try:
        yield
    finally:
        gpytorch_models.separate_mtmvn = orig


def keeps_baseline_base_samples() -> bool:
    """Whether BoTorch's NEHVI keeps the base samples of previously
    decomposed points when caching pending points (#3392)."""
    from botorch.utils.multi_objective import hypervolume as bt_hv

    return "_baseline_base_sampler" in inspect.getsource(bt_hv.NoisyExpectedHypervolumeMixin)


def binds_constraint_indices() -> bool:
    """Whether BoTorch's feasibility weighting uses every constraint index
    (#3392)."""
    from botorch.acquisition.multi_objective import objective as bt_obj

    return "lambda Y, i=i" in inspect.getsource(bt_obj.FeasibilityWeightedMCMultiOutputObjective)


def has_pending_semantics() -> bool:
    """Whether ``set_X_pending`` replaces the pending set (#3379)."""
    from botorch.utils.multi_objective import hypervolume as bt_hv

    return hasattr(bt_hv.NoisyExpectedHypervolumeMixin, "_is_prefix")


# ---------------------------------------------------------------------------
# Models, specifications and errors
# ---------------------------------------------------------------------------


def build_case_model(case: dict, seed: int) -> tuple[Any, list[dict], torch.Tensor]:
    """The BoTorch model of a case, its specification (phase-4B format) and
    the baseline (the first model's training inputs)."""
    from botorch.models import ModelListGP

    gf = _gf()
    subs, specs, xs = [], [], []
    for cfg in case["models"]:
        X, Y, Yvar, bounds = gf.p4_make_data(cfg, seed + 17 * cfg.get("seed_offset", 0))
        sub = gf.p4_build_model(cfg, X, Y, Yvar, bounds)
        n_el = int(torch.Size(gf.p4_aug_shape(cfg)).numel())
        raw = gf.p4_set_values(sub, n_el, cfg["d"])
        gf.set_flat_params(sub, raw)
        sub.eval()
        subs.append(sub)
        xs.append(X)
        specs.append(
            {
                "config": {
                    "d": cfg["d"],
                    "m": cfg["m"],
                    "covar": cfg.get("covar", "rbf"),
                    "normalize": cfg.get("normalize"),
                    "outcome": cfg.get("outcome", "standardize"),
                    "bounds": None if bounds is None else enc(bounds),
                },
                "train_x": [enc(X)],
                "train_y": [enc(Y)],
                "train_yvar": None if Yvar is None else [enc(Yvar)],
                "raw_params": raw,
            }
        )
    model = ModelListGP(*subs) if case["kind"] == "model_list" else subs[0]
    model.eval()
    return model, specs, xs[0]


def perturbed_copy(model: Any, seed: int, rel: float = 1e-14) -> Any:
    """A copy of a model whose training inputs are perturbed by ``rel``."""
    twin = copy.deepcopy(model)
    g = torch.Generator().manual_seed(seed)

    def noisy(X: torch.Tensor) -> torch.Tensor:
        return X * (1 + rel * torch.randn(X.shape, generator=g, dtype=X.dtype))

    for gp in getattr(twin, "models", [twin]):
        gp.train_inputs = tuple(noisy(X) for X in gp.train_inputs)
        if getattr(gp, "_original_train_inputs", None) is not None:
            gp._original_train_inputs = noisy(gp._original_train_inputs)
        gp.prediction_strategy = None
    return twin


def rel_err(a: torch.Tensor, e: torch.Tensor, atol: float = 0.0) -> float:
    """Mixed elementwise / normwise relative error (``common::rel_err``)."""
    diff = (a - e).abs()
    diff = torch.where(diff <= atol, torch.zeros_like(diff), diff)
    scale = e.abs().max().item()
    floor = scale if scale > 0 else 1.0
    return (diff / (e.abs() + floor)).max().item()


def log_err(a: torch.Tensor, e: torch.Tensor) -> float:
    """Error of log-space values: absolute below 1, relative above."""
    return ((a - e).abs() / e.abs().clamp_min(1.0)).max().item()


def points(X: torch.Tensor, shape: tuple[int, ...], seed: int) -> torch.Tensor:
    lo, hi = X.min(0).values, X.max(0).values
    g = torch.Generator().manual_seed(seed)
    return lo + (hi - lo) * torch.rand(*shape, generator=g, dtype=DT)


def values_and_grads(acqf: Any, X: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    Xg = X.detach().clone().requires_grad_(True)
    values = acqf(Xg)
    (grad,) = torch.autograd.grad(values.sum(), Xg)
    return values.detach(), grad


def quad_mo(spec: dict, Y: torch.Tensor) -> torch.Tensor:
    """A generic objective: ``c_k + sum_j a_kj y_j + sum_j b_kj y_j^2``."""
    cols = []
    for a, b, c in zip(spec["a"], spec["b"], spec["c"], strict=True):
        at, bt = torch.tensor(a, dtype=DT), torch.tensor(b, dtype=DT)
        cols.append(c + (Y * at).sum(-1) + (Y * Y * bt).sum(-1))
    return torch.stack(cols, dim=-1)


def objective_spec(kind: str, m: int) -> tuple[dict | None, int]:
    """The specification of an objective and its number of outputs."""
    if kind == "identity":
        return None, m
    if kind == "outcomes":
        return {"type": "identity", "outcomes": [0, -1], "num_outcomes": m}, 2
    if kind == "weighted":
        return {
            "type": "weighted",
            "weights": ([1.0, -0.5] + [0.75] * (m - 2)),
            "outcomes": None,
        }, m
    if kind == "weighted_outcomes":
        return {
            "type": "weighted",
            "weights": [0.5, 2.0],
            "outcomes": [-1, 0],
            "num_outcomes": m,
        }, 2
    if kind == "generic":
        # [y_0 + 0.1 y_last^2, -(y_1 - 0.5)^2]
        e = [[1.0 if j == k else 0.0 for j in range(m)] for k in range(m)]
        b0 = [0.0] * m
        b0[m - 1] += 0.1
        b1 = [-1.0 if j == 1 else 0.0 for j in range(m)]
        return {"type": "generic", "a": [e[0], e[1]], "b": [b0, b1], "c": [0.0, -0.25]}, 2
    if kind == "feasibility":
        return {"type": "feasibility", "constraint_idcs": [-1]}, m - 1
    if kind == "feasibility_two":
        return {"type": "feasibility", "constraint_idcs": [1, 2]}, m - 2
    raise ValueError(kind)


def bt_objective(spec: dict | None, model: Any, X_baseline: torch.Tensor) -> Any:
    from botorch.acquisition.multi_objective import objective as bt_obj

    if spec is None:
        return None
    t = spec["type"]
    if t == "identity":
        return bt_obj.IdentityMCMultiOutputObjective(
            outcomes=spec["outcomes"], num_outcomes=spec.get("num_outcomes")
        )
    if t == "weighted":
        return bt_obj.WeightedMCMultiOutputObjective(
            torch.tensor(spec["weights"], dtype=DT),
            outcomes=spec["outcomes"],
            num_outcomes=spec.get("num_outcomes"),
        )
    if t == "generic":
        return bt_obj.GenericMCMultiOutputObjective(lambda Y, X=None: quad_mo(spec, Y))
    if t == "feasibility":
        return bt_obj.FeasibilityWeightedMCMultiOutputObjective(
            model=model, X_baseline=X_baseline, constraint_idcs=spec["constraint_idcs"]
        )
    raise ValueError(t)


def constraint_specs(
    means: torch.Tensor, kind: str | None
) -> tuple[list[dict] | None, list[float] | None]:
    """Outcome constraints on the model outputs (feasible iff <= 0): ``one``
    (the last output below its mean), ``two`` (also a quadratic one)."""
    if kind is None:
        return None, None
    m = means.shape[-1]
    e_last = [0.0] * m
    e_last[-1] = 1.0
    specs = [{"type": "linear", "weights": e_last, "offset": -means[:, -1].mean().item()}]
    etas = [0.05]
    if kind == "two":
        b = [1.0 if j == 0 else 0.0 for j in range(m)]
        specs.append(
            {
                "type": "quadratic",
                "a": [0.0] * m,
                "b": b,
                "c": -1.5 * (means[:, 0] ** 2).mean().item(),
            }
        )
        etas.append(0.2)
    return specs, etas


def bt_constraints(specs: list[dict] | None) -> list[Any] | None:
    return None if specs is None else [_gf().p4b_constraint(s) for s in specs]


def draws_of(record: dict[tuple[int, int], torch.Tensor], n: int) -> list[dict]:
    return [
        {"d": d, "z": enc(z.reshape(-1))} for (rows, d), z in sorted(record.items()) if rows == n
    ]


def model_means(model: Any, X: torch.Tensor) -> torch.Tensor:
    with torch.no_grad(), dense_model_list():
        return model.posterior(X).mean


def ref_and_front(Yobj: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """A reference point below the objective values and a weakened front (the
    values pulled 40% of the way to the reference point)."""
    lo, hi = Yobj.min(0).values, Yobj.max(0).values
    ref = lo - 0.1 * (hi - lo).clamp_min(1e-3)
    return ref, ref + 0.6 * (Yobj - ref)


def objective_values(
    spec: dict | None, model: Any, X: torch.Tensor, means: torch.Tensor
) -> torch.Tensor:
    o = bt_objective(spec, model, X)
    if o is None:
        return means
    with torch.no_grad():
        return o(means.unsqueeze(0)).squeeze(0)


def partitioning(kind: str, ref: torch.Tensor, Y: torch.Tensor, alpha: float) -> Any:
    from botorch.utils.multi_objective.box_decompositions.non_dominated import (
        FastNondominatedPartitioning,
        NondominatedPartitioning,
    )

    if kind == "nondominated":
        return NondominatedPartitioning(ref_point=ref, Y=Y, alpha=alpha)
    return FastNondominatedPartitioning(ref_point=ref, Y=Y)


def floors(evaluate: Any, model: Any, v: torch.Tensor, g: torch.Tensor, log: bool) -> dict:
    """BoTorch's change under perturbations of the training inputs."""
    fv = fg = 0.0
    for k in (1, 2):
        v_tw, g_tw = evaluate(perturbed_copy(model, seed=k))
        fv = max(fv, log_err(v_tw, v) if log else rel_err(v_tw, v, atol=1e-300))
        fg = max(fg, rel_err(g_tw, g, atol=1e-14))
    return {"value": fv, "grad": fg}


# ---------------------------------------------------------------------------
# Cases
# ---------------------------------------------------------------------------


def objectives_fixture(model: Any, X: torch.Tensor, m: int) -> list[dict]:
    """The multi-output objectives on random samples (``S x n x m``)."""
    Y = torch.randn(5, 3, m, generator=torch.Generator().manual_seed(1), dtype=DT)
    kinds = ["outcomes", "weighted", "generic"]
    if m >= 3:
        kinds += ["weighted_outcomes", "feasibility"]
    out = []
    for kind in kinds:
        spec, _ = objective_spec(kind, m)
        o = bt_objective(spec, model, X)
        with dense_model_list():
            values = o(Y)
        entry = {"objective": spec, "y": enc(Y), "values": enc(values)}
        if kind == "feasibility":
            from botorch.acquisition.utils import get_infeasible_cost

            with dense_model_list():
                cost = get_infeasible_cost(X=X, model=model, objective=lambda y, X: y)
            entry["infeasible_cost"] = enc(cost[:-1])
        out.append(entry)
    return out


def ehvi_fixture(model: Any, X: torch.Tensor, m: int, seed: int, record: dict) -> list[dict]:
    from botorch.acquisition.multi_objective import logei as bt_logei
    from botorch.acquisition.multi_objective import monte_carlo as bt_mc
    from botorch.sampling.normal import SobolQMCNormalSampler

    means = model_means(model, X)
    out = []
    for i, (name, q, n_pending, cons_kind, obj_kind, part_kind, extra) in enumerate(EHVI_CASES):
        spec, mo = objective_spec(obj_kind, m)
        if obj_kind == "feasibility" and m < 3:
            continue
        ref, front = ref_and_front(objective_values(spec, model, X, means))
        alpha = 1e-2 if part_kind == "nondominated" and mo >= 3 else 0.0
        part = partitioning(part_kind, ref, front, alpha)
        cons, etas = constraint_specs(means, cons_kind)
        kw: dict[str, Any] = dict(extra)
        if cons is not None:
            kw["constraints"], kw["eta"] = bt_constraints(cons), torch.tensor(etas, dtype=DT)
        pending = points(X, (n_pending, X.shape[-1]), seed + 3 * i) if n_pending else None
        Xt = points(X, (2, q, X.shape[-1]), seed + 3 * i + 1)
        log = name.startswith("qLog")
        with recorded_draws(record), dense_model_list():
            acqf = getattr(bt_logei if log else bt_mc, name)(
                model,
                ref,
                part,
                sampler=SobolQMCNormalSampler(sample_shape=torch.Size([NUM_SAMPLES]), seed=0),
                objective=bt_objective(spec, model, X),
                X_pending=pending,
                **kw,
            )
            v, g = values_and_grads(acqf, Xt)
        lower, upper = acqf.cell_lower_bounds, acqf.cell_upper_bounds
        out.append(
            {
                "acqf": name,
                "q": q,
                "objective": spec,
                "constraints": cons,
                "eta": etas,
                "fat": extra.get("fat"),
                "tau_relu": extra.get("tau_relu"),
                "partitioning": part_kind,
                "alpha": alpha,
                "ref_point": enc(ref),
                "front": enc(front),
                "cell_shape": list(lower.shape),
                "cell_lower": enc(lower.reshape(-1)),
                "cell_upper": enc(upper.reshape(-1)),
                "x_pending": None if pending is None else enc(pending),
                "x": enc(Xt),
                "values": enc(v),
                "grads": enc(g),
            }
        )
    return out


def _nehvi(model: Any, X: torch.Tensor, log: bool, kw: dict, record: dict, calls: list[int]) -> Any:
    from botorch.acquisition.multi_objective import logei as bt_logei
    from botorch.acquisition.multi_objective import monte_carlo as bt_mc
    from botorch.sampling.normal import SobolQMCNormalSampler

    cls = (
        bt_logei.qLogNoisyExpectedHypervolumeImprovement
        if log
        else bt_mc.qNoisyExpectedHypervolumeImprovement
    )
    with recorded_draws(record), seeded_pruning(calls), dense_model_list():
        return cls(
            model,
            X_baseline=X,
            sampler=SobolQMCNormalSampler(sample_shape=torch.Size([NUM_SAMPLES]), seed=0),
            **kw,
        )


def _eval(acqf: Any, Xt: torch.Tensor, record: dict) -> tuple[torch.Tensor, torch.Tensor]:
    with recorded_draws(record), dense_model_list():
        return values_and_grads(acqf, Xt)


def nehvi_state(acqf: Any) -> dict:
    hvs = getattr(acqf, "_baseline_hvs", None)
    return {
        "x_baseline": enc(acqf.X_baseline),
        "num_cells": int(acqf.cell_lower_bounds.shape[-2]),
        "hypervolumes": enc(acqf._hypervolumes),
        "baseline_hvs": None if hvs is None else enc(hvs),
        "prev_nehvi": enc(acqf._prev_nehvi.item()),
    }


def nehvi_kwargs(
    mdl: Any, X: torch.Tensor, spec: dict | None, cons: Any, etas: Any, kw: dict
) -> dict:
    """The keyword arguments of a NEHVI case for a model."""
    out = dict(kw, objective=bt_objective(spec, mdl, X))
    if cons is not None:
        out["constraints"], out["eta"] = bt_constraints(cons), torch.tensor(etas, dtype=DT)
    return out


def _nehvi_twin(
    mdl: Any, X: torch.Tensor, log: bool, args: tuple, Xt: torch.Tensor, record: dict
) -> tuple[torch.Tensor, torch.Tensor]:
    a = _nehvi(mdl, X, log, nehvi_kwargs(mdl, X, *args), record, [])
    return _eval(a, Xt, record)


def nehvi_fixture(model: Any, X: torch.Tensor, m: int, seed: int, record: dict) -> list[dict]:
    means = model_means(model, X)
    out = []
    for i, case in enumerate(NEHVI_CASES):
        log, q, cache_root, incremental, alpha, cons_kind, obj_kind, n_pending, max_iep, prune = (
            case
        )
        if obj_kind == "feasibility" and m < 3:
            continue
        spec, mo = objective_spec(obj_kind, m)
        ref, _ = ref_and_front(objective_values(spec, model, X, means))
        cons, etas = constraint_specs(means, cons_kind)
        pending = points(X, (n_pending, X.shape[-1]), seed + 5 * i) if n_pending else None
        alpha = alpha if mo >= 3 else 0.0
        kw = dict(
            ref_point=ref,
            cache_root=cache_root,
            incremental_nehvi=incremental,
            alpha=alpha,
            max_iep=max_iep,
            prune_baseline=prune,
            X_pending=pending,
        )
        calls: list[int] = []
        acqf = _nehvi(model, X, log, nehvi_kwargs(model, X, spec, cons, etas, kw), record, calls)
        assert bool(calls) == prune, "the pruning sampler was not used"
        Xt = points(X, (2, q, X.shape[-1]), seed + 5 * i + 1)
        v, g = _eval(acqf, Xt, record)
        twin = functools.partial(
            _nehvi_twin, X=X, log=log, args=(spec, cons, etas, kw), Xt=Xt, record=record
        )
        entry = {
            "log": log,
            "q": q,
            "cache_root": cache_root,
            "incremental": incremental,
            "alpha": alpha,
            "objective": spec,
            "constraints": cons,
            "eta": etas,
            "max_iep": max_iep,
            "prune_baseline": prune,
            "ref_point": enc(ref),
            "x_pending": None if pending is None else enc(pending),
            "x": enc(Xt),
            "values": enc(v),
            "grads": enc(g),
            "floor": floors(twin, model, v, g, log),
        }
        entry.update(nehvi_state(acqf))
        out.append(entry)
    return out


def analytic_fixture(model: Any, X: torch.Tensor, seed: int) -> list[dict]:
    from botorch.acquisition.multi_objective.analytic import ExpectedHypervolumeImprovement

    ref, front = ref_and_front(model_means(model, X))
    out = []
    for kind in ("fast", "nondominated"):
        part = partitioning(kind, ref, front, 0.0)
        with dense_model_list():
            acqf = ExpectedHypervolumeImprovement(model, ref.tolist(), part)
            Xt = points(X, (6, 1, X.shape[-1]), seed)
            v, g = values_and_grads(acqf, Xt)
        out.append(
            {
                "partitioning": kind,
                "ref_point": enc(ref),
                "front": enc(front),
                "x": enc(Xt),
                "values": enc(v),
                "grads": enc(g),
            }
        )
    return out


def _parego(
    mdl: Any, X: torch.Tensor, args: tuple, Xt: torch.Tensor, record: dict
) -> tuple[Any, Any, Any]:
    """qLogNParEGO of a case for a model, its values and gradients."""
    from botorch.acquisition.multi_objective.parego import qLogNParEGO
    from botorch.sampling.normal import SobolQMCNormalSampler

    with recorded_draws(record), dense_model_list(), no_lower_bound():
        acqf = qLogNParEGO(
            mdl,
            X_baseline=X,
            sampler=SobolQMCNormalSampler(sample_shape=torch.Size([NUM_SAMPLES]), seed=0),
            **nehvi_kwargs(mdl, X, *args),
        )
        v, g = values_and_grads(acqf, Xt)
    return acqf, v, g


def parego_fixture(model: Any, X: torch.Tensor, m: int, seed: int, record: dict) -> list[dict]:
    means = model_means(model, X)
    out = []
    for i, (variant, q, incremental) in enumerate(PAREGO_CASES):
        obj_kind = variant if variant in ("weighted", "generic") else "identity"
        spec, mo = objective_spec(obj_kind, m)
        cons, etas = (None, None)
        pending = None
        if variant == "constraints+pending":
            cons, etas = constraint_specs(means, "one")
            pending = points(X, (1, X.shape[-1]), seed + 7 * i)
        weights = torch.linspace(0.2, 1.0, mo, dtype=DT)
        weights = weights / weights.sum()
        Xt = points(X, (2, q, X.shape[-1]), seed + 7 * i + 1)
        kw = dict(
            scalarization_weights=weights,
            X_pending=pending,
            cache_root=False,
            incremental=incremental,
        )
        evaluate = functools.partial(
            _parego, X=X, args=(spec, cons, etas, kw), Xt=Xt, record=record
        )
        acqf, v, g = evaluate(model)
        out.append(
            {
                "variant": variant,
                "q": q,
                "incremental": incremental,
                "objective": spec,
                "constraints": cons,
                "eta": etas,
                "weights": enc(weights),
                "x_pending": None if pending is None else enc(pending),
                "x": enc(Xt),
                "y_baseline": enc(acqf.Y_baseline),
                "values": enc(v),
                "grads": enc(g),
                "floor": floors(lambda mdl, f=evaluate: f(mdl)[1:], model, v, g, True),
            }
        )
    return out


def prune_fixture(model: Any, X: torch.Tensor, m: int, record: dict) -> list[dict]:
    from botorch.acquisition.multi_objective.utils import prune_inferior_points_multi_objective

    means = model_means(model, X)
    ref, _ = ref_and_front(means)
    out = []
    for case in PRUNE_CASES:
        kw: dict[str, Any] = {}
        spec = None
        if "objective" in case:
            spec, _ = objective_spec(case["objective"], m)
            kw["objective"] = bt_objective(spec, model, X)
        cons = None
        if "constraints" in case:
            cons, _ = constraint_specs(means, case["constraints"])
            kw["constraints"] = bt_constraints(cons)
        if "max_frac" in case:
            kw["max_frac"] = case["max_frac"]
        calls: list[int] = []
        with recorded_draws(record), seeded_pruning(calls), dense_model_list():
            pruned = prune_inferior_points_multi_objective(
                model, X, ref, num_samples=PRUNE_SAMPLES, **kw
            )
        assert calls
        idcs = [int((X == row).all(-1).nonzero()[0]) for row in pruned]
        out.append(
            {
                "objective": spec,
                "constraints": cons,
                "max_frac": case.get("max_frac", 1.0),
                "ref_point": enc(ref),
                "indices": idcs,
            }
        )
    return out


def misc_fixture() -> dict:
    from botorch.acquisition.multi_objective.utils import get_default_partitioning_alpha
    from botorch.utils.multi_objective.hypervolume import compute_subset_indices

    alphas = []
    for k in range(1, 10):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            a = get_default_partitioning_alpha(k)
        alphas.append({"num_objectives": k, "alpha": a, "warning": [str(x.message) for x in w]})
    subsets = []
    for q in range(1, 5):
        s = compute_subset_indices(q)
        subsets.append({"q": q, "subsets": [enc(s[f"q_choose_{i}"]) for i in range(1, q + 1)]})
    return {"alphas": alphas, "subsets": subsets}


def model_fixture(case: dict, seed: int) -> dict:
    gf = _gf()
    model, specs, X = build_case_model(case, seed)
    m = model.num_outputs
    # All acquisition functions use samplers with the same seed and number of
    # samples, hence the same base samples for the same dimension.
    record: dict[tuple[int, int], torch.Tensor] = {}
    fixture = {
        "name": case["name"],
        "kind": case["kind"],
        "versions": versions(gf),
        "models": specs,
        "baseline_x": enc(X),
        "num_samples": NUM_SAMPLES,
        "objectives": objectives_fixture(model, X, m),
        "ehvi": ehvi_fixture(model, X, m, seed + 100, record),
        "nehvi": nehvi_fixture(model, X, m, seed + 200, record),
        "analytic": analytic_fixture(model, X, seed + 300),
        "parego": parego_fixture(model, X, m, seed + 400, record),
        "prune": prune_fixture(model, X, m, record),
    }
    fixture["draws"] = draws_of(record, NUM_SAMPLES)
    fixture["prune_base_samples"] = enc(record[(PRUNE_SAMPLES, X.shape[0] * m)].reshape(-1))
    assert set(record) == {(n, d) for n, d in record if n in (NUM_SAMPLES, PRUNE_SAMPLES)}
    return fixture


def pending_fixture(seed: int) -> dict:
    """``set_X_pending`` sequences and two-index feasibility weighting
    (meta-pytorch/botorch#3392)."""
    gf = _gf()
    sequences = []
    draws: dict[str, list[dict]] = {}
    for c, case in enumerate(MODEL_CASES):
        if case["name"] not in ("m2_rbf_d2_n10", "m3_matern_d3_n10_normalize", "list_2_1_d2"):
            continue
        model, specs, X = build_case_model(case, seed + c)
        ref, _ = ref_and_front(model_means(model, X))
        P = points(X, (5, X.shape[-1]), seed + 21)
        Xt = points(X, (2, 1, X.shape[-1]), seed + 25)
        record: dict[tuple[int, int], torch.Tensor] = {}
        for log in (False, True):
            for incremental in (True, False):
                kw = dict(ref_point=ref, incremental_nehvi=incremental)
                acqf = _nehvi(model, X, log, kw, record, [])
                steps = []
                for step, select in PENDING_STEPS:
                    Pk = select(P)
                    with recorded_draws(record), dense_model_list():
                        acqf.set_X_pending(Pk)
                    v, g = _eval(acqf, Xt, record)
                    entry = {
                        "step": step,
                        "x_pending": None if Pk is None else enc(Pk),
                        "values": enc(v),
                        "grads": enc(g),
                    }
                    entry.update(nehvi_state(acqf))
                    steps.append(entry)
                sequences.append(
                    {
                        "model": case["name"],
                        "models": specs,
                        "kind": case["kind"],
                        "baseline_x": enc(X),
                        "log": log,
                        "incremental": incremental,
                        "ref_point": enc(ref),
                        "x": enc(Xt),
                        "steps": steps,
                    }
                )
        draws[case["name"]] = draws_of(record, NUM_SAMPLES)
    # Feasibility weighting with two constraint indices.
    feas = []
    for c, case in enumerate(MODEL_CASES):
        if case["name"] not in ("m3_matern_d3_n10_normalize", "m4_fixed_noise_d2_n8"):
            continue
        model, specs, X = build_case_model(case, seed + 50 + c)
        m = model.num_outputs
        spec, _ = objective_spec("feasibility_two", m)
        o = bt_objective(spec, model, X)
        Y = torch.randn(4, 3, m, generator=torch.Generator().manual_seed(seed + 60 + c), dtype=DT)
        Y[0, :, 1] = -1.0  # violates only constraint 1
        Y[0, :, 2] = 1.0
        feas.append(
            {
                "model": case["name"],
                "models": specs,
                "kind": case["kind"],
                "baseline_x": enc(X),
                "objective": spec,
                "y": enc(Y),
                "values": enc(o(Y)),
            }
        )
    return {
        "versions": versions(gf),
        "num_samples": NUM_SAMPLES,
        "draws": draws,
        "sequences": sequences,
        "feasibility": feas,
    }


def versions(gf: Any) -> dict:
    import botorch
    import gpytorch

    return {
        "botorch": botorch.__version__,
        "botorch_revision": gf.botorch_revision(),
        "gpytorch": gpytorch.__version__,
        "torch": torch.__version__,
    }


def _write(path: str, obj: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=None, separators=(",", ":"), allow_nan=False)
        f.write("\n")


def generate_phase5b(out: str) -> None:
    """Writes the phase-5B fixtures that BoTorch main reproduces."""
    assert has_pending_semantics(), "phase 5B needs a BoTorch with meta-pytorch/botorch#3379"
    torch.set_num_threads(1)
    out = os.path.join(out, "phase5b")
    os.makedirs(out, exist_ok=True)
    warnings.simplefilter("ignore")
    for i, case in enumerate(MODEL_CASES):
        fixture = model_fixture(case, seed=5000 + 31 * i)
        _write(os.path.join(out, f"{case['name']}.json"), fixture)
        worst = max(
            (max(e["floor"]["value"], e["floor"]["grad"]) for e in fixture["nehvi"]), default=0.0
        )
        print(
            f"phase 5b: wrote {case['name']}.json ({len(fixture['ehvi'])} EHVI, {len(fixture['nehvi'])} NEHVI, "
            f"{len(fixture['parego'])} ParEGO cases; largest NEHVI noise floor {worst:.1e})",
            file=sys.stderr,
        )
    _write(os.path.join(out, "misc.json"), misc_fixture())
    print("phase 5b: wrote misc.json", file=sys.stderr)


def generate_phase5b_3392(out: str) -> None:
    """Writes ``pending_3392.json`` (needs meta-pytorch/botorch#3392)."""
    if not (
        keeps_baseline_base_samples() and binds_constraint_indices() and has_pending_semantics()
    ):
        raise SystemExit("--phase 5b-3392 needs a BoTorch that includes meta-pytorch/botorch#3392")
    torch.set_num_threads(1)
    out = os.path.join(out, "phase5b")
    os.makedirs(out, exist_ok=True)
    warnings.simplefilter("ignore")
    _write(os.path.join(out, "pending_3392.json"), pending_fixture(seed=7000))
    print("phase 5b-3392: wrote pending_3392.json", file=sys.stderr)

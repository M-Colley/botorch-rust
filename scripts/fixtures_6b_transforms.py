"""Fixtures of input and outcome transforms in ``SingleTaskGP`` (phase 6b).

Written to ``<out>/6b_transforms/`` by ``generate_fixtures.py --phase
6b-transforms`` (BoTorch main). Every case is a ``SingleTaskGP`` with an
input and/or outcome transform described by a JSON spec (see
``botorch_pyo3/tests/transform_helpers.py``; the Rust test builds the same
transforms from it) and records:

* the parameter names and shapes (``named_parameters()``, including learned
  transform parameters such as ``Warp``'s concentrations), the optimizer
  bounds, the default raw parameters, the parameter-name prefixes of the
  input transform's steps and the training inputs as the model sees them in
  eval mode (BoTorch's ``preprocess_transform``);
* the marginal log likelihood and its gradient (autograd, also with respect
  to the transform parameters) at the default and at random raw parameters;
* posteriors at random points: ``q = 1`` means and variances (the moments
  of a ``TransformedPosterior`` where BoTorch defines them) and the
  gradients of their sums with respect to the points; posterior samples of
  ``q = 3`` points from given base samples (BoTorch's
  ``rsample_from_base_samples``) and the gradient of a weighted sum of them;
* for single-output models, ``ExpectedImprovement`` (where the posterior has
  moments) and ``qExpectedImprovement`` with given base samples: values and
  input gradients;
* ``fit_gpytorch_mll`` from the default parameters (single-output cases with
  ``fit``): the final loss and BoTorch's final losses on 64 twins whose
  training inputs are perturbed at 1e-14 .. 1e-11 (with their inputs).

BoTorch main's transform bugs fixed by meta-pytorch/botorch#3389 are
patched (``transform_helpers.botorch_fixes``). Non-finite numbers are
written as the strings "inf", "-inf" and "nan".
"""

from __future__ import annotations

import json
import math
import os
import sys
import warnings
from typing import Any

import torch

DT = torch.float64
TWIN_SCALES = (1e-14, 1e-13, 1e-12, 1e-11)
TESTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "botorch_pyo3", "tests")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import fixtures_6a_kernels as f6a  # noqa: E402


def enc(x: Any) -> Any:
    if isinstance(x, torch.Tensor):
        return enc(x.detach().tolist())
    if isinstance(x, float):
        if math.isnan(x):
            return "nan"
        if math.isinf(x):
            return "inf" if x > 0 else "-inf"
        return x
    if isinstance(x, dict):
        return {str(k): enc(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [enc(v) for v in x]
    return x


def _helpers() -> Any:
    sys.path.insert(0, os.path.normpath(TESTS))
    import transform_helpers

    return transform_helpers


def _p(*a: Any) -> dict[str, Any]:
    return _helpers().prior(*a)


def cases() -> list[dict[str, Any]]:
    """``(name, input spec, outcome spec, options)``; the outcome spec
    ``"default"`` is BoTorch's default ``Standardize``."""
    ln = ("LogNormalPrior", 0.0, 0.75**0.5)
    out: list[tuple[str, Any, Any, dict[str, Any]]] = [
        # Outcome transforms.
        ("log", None, [{"type": "Log"}], {"fit": True}),
        ("log_std", None, [{"type": "Log"}, {"type": "Standardize"}], {"fit": True}),
        ("std_bilog", None, [{"type": "Standardize"}, {"type": "Bilog"}], {}),
        ("power_half", None, [{"type": "Power", "power": 0.5}], {}),
        ("power_two_std", None, [{"type": "Power", "power": 2.0}, {"type": "Standardize"}], {}),
        ("bilog", None, [{"type": "Bilog"}], {"shift": -2.0, "fit": True}),
        ("bilog_std", None, [{"type": "Bilog"}, {"type": "Standardize"}], {"shift": -2.0}),
        (
            "log_fixed_noise",
            None,
            [{"type": "Log"}, {"type": "Standardize"}],
            {"fixed_noise": True},
        ),
        (
            "log_outputs_m2",
            None,
            [{"type": "Log", "outputs": [1]}, {"type": "Standardize", "m": 2}],
            {"m": 2},
        ),
        ("std_outputs_m2", None, [{"type": "Standardize", "m": 2, "outputs": [0]}], {"m": 2}),
        (
            "stratified",
            None,
            [{"type": "StratifiedStandardize", "stratification_idx": 2, "task_values": [0, 1, 2]}],
            {"tasks": True, "d": 3},
        ),
        # Input transforms.
        (
            "normalize_indices",
            [{"type": "Normalize", "d": 3, "indices": [0, 2]}],
            "default",
            {"d": 3},
        ),
        ("normalize_center", [{"type": "Normalize", "d": 2, "center": 0.0}], "default", {}),
        (
            "normalize_bounds_reverse",
            [
                {
                    "type": "Normalize",
                    "d": 2,
                    "bounds": [[0.0, -1.0], [2.0, 1.0]],
                    "reverse": True,
                }
            ],
            "default",
            {},
        ),
        (
            "input_standardize",
            [{"type": "InputStandardize", "d": 3, "indices": [0, 1]}],
            "default",
            {"d": 3},
        ),
        (
            "affine",
            [
                {
                    "type": "AffineInputTransform",
                    "d": 2,
                    "coefficient": [1.5, 0.5],
                    "offset": [0.2, -0.1],
                }
            ],
            "default",
            {},
        ),
        ("log10", [{"type": "Log10", "indices": [1]}], "default", {"shift_x": 0.5}),
        ("warp", [{"type": "Warp", "d": 2, "indices": [0, 1]}], "default", {"fit": True}),
        (
            "warp_priors_bounds",
            [
                {
                    "type": "Warp",
                    "d": 3,
                    "indices": [0, 2],
                    "bounds": [[-1.0, -1.0, -1.0], [2.0, 2.0, 2.0]],
                    "c0_prior": _p(*ln),
                    "c1_prior": _p(*ln),
                }
            ],
            "default",
            {"d": 3, "fit": True},
        ),
        (
            "normalize_warp",
            [
                {"type": "Normalize", "d": 2},
                {
                    "type": "Warp",
                    "d": 2,
                    "indices": [0, 1],
                    "c0_prior": _p(*ln),
                    "c1_prior": _p(*ln),
                },
            ],
            [{"type": "Log"}, {"type": "Standardize"}],
            {"scale_x": 3.0, "fit": True},
        ),
        (
            "round_eval",
            [{"type": "Round", "integer_indices": [1], "flags": [False, True, True]}],
            "default",
            {"int_col": 1},
        ),
        (
            "round_approximate",
            [{"type": "Round", "integer_indices": [0], "approximate": True, "tau": 0.1}],
            "default",
            {"int_col": 0},
        ),
        ("filter", [{"type": "FilterFeatures", "feature_indices": [2, 0]}], "default", {"d": 3}),
        ("interaction", [{"type": "InteractionFeatures"}], "default", {"d": 3}),
        (
            "interaction_indices",
            [{"type": "InteractionFeatures", "indices": [0, 2]}],
            "default",
            {"d": 3},
        ),
        (
            "one_hot_to_numeric",
            [{"type": "OneHotToNumeric", "dim": 4, "categorical_features": {"1": 3}}],
            "default",
            {"one_hot": True},
        ),
        (
            "numeric_to_categorical",
            [{"type": "NumericToCategoricalEncoding", "dim": 3, "categorical_features": {"2": 3}}],
            "default",
            {"tasks": True, "d": 3},
        ),
        (
            "imputation",
            [
                {
                    "type": "LearnedFeatureImputation",
                    "feature_indices": {"0": [0, 1], "1": [0], "2": [1]},
                    "d": 2,
                    "bounds": [[0.0, 0.0], [1.0, 1.0]],
                }
            ],
            "default",
            {"tasks": True, "d": 3},
        ),
        (
            "append_features",
            [{"type": "AppendFeatures", "feature_set": [[0.1], [0.5], [0.9]]}],
            "default",
            {"append": 1},
        ),
        (
            "perturbation_append",
            [
                {"type": "InputPerturbation", "perturbation_set": [[0.05, 0.0], [-0.05, 0.02]]},
                {"type": "AppendFeatures", "feature_set": [[0.2], [0.7]], "skip_expand": True},
            ],
            "default",
            {"append": 1},
        ),
        (
            # get_rounding_input_transform(bounds, integer_indices=[0]) (in
            # eval mode when the model is built).
            "rounding_chain",
            [
                {
                    "type": "Normalize",
                    "d": 3,
                    "bounds": [[0.0, 0.0, 0.0], [3.0, 1.0, 1.0]],
                    "indices": [0],
                    "reverse": True,
                    "flags": [False, True, True],
                    "training": False,
                },
                {
                    "type": "Round",
                    "integer_indices": [0],
                    "flags": [False, True, True],
                    "training": False,
                },
                {
                    "type": "Normalize",
                    "d": 3,
                    "bounds": [[0.0, 0.0, 0.0], [3.0, 1.0, 1.0]],
                    "indices": [0],
                    "flags": [False, True, True],
                    "training": False,
                },
            ],
            "default",
            {"d": 3, "int_col": 0},
        ),
        (
            "chain_mixed",
            [
                {"type": "Normalize", "d": 3, "indices": [0, 1]},
                {"type": "Warp", "d": 3, "indices": [0, 1]},
                {"type": "InteractionFeatures", "indices": [0, 2]},
            ],
            [{"type": "Bilog"}, {"type": "Standardize"}],
            {"d": 3, "scale_x": 2.0},
        ),
        (
            "warp_m2",
            [{"type": "Warp", "d": 2, "indices": [0]}],
            [{"type": "Log", "outputs": [0]}, {"type": "Standardize", "m": 2}],
            {"m": 2, "unsupported": True},
        ),
    ]
    return [{"name": n, "input": i, "outcome": o, **opts} for n, i, o, opts in out]


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def make_data(case: dict[str, Any], seed: int) -> tuple[Any, Any, Any]:
    g = torch.Generator().manual_seed(seed)
    n, d, m = case.get("n", 12), case.get("d", 2), case.get("m", 1)
    if case.get("one_hot"):
        X = torch.cat(
            [
                torch.rand(n, 1, generator=g, dtype=DT),
                torch.nn.functional.one_hot(torch.arange(n) % 3, 3).to(DT),
            ],
            -1,
        )
    else:
        # Training inputs of models with `AppendFeatures` (not applied in
        # training) include the appended columns.
        X = torch.rand(n, d + case.get("append", 0), generator=g, dtype=DT)
    if case.get("tasks"):
        X[:, -1] = (torch.arange(n, dtype=DT) % 3).to(DT)
    if "int_col" in case:
        X[:, case["int_col"]] = torch.randint(0, 4, (n,), generator=g).to(DT)
    X = X * case.get("scale_x", 1.0) + case.get("shift_x", 0.0)
    cols = []
    for t in range(m):
        f = torch.sin((3.0 + t) * X[:, 0]) + (1.0 + t) * (X[:, 1:] ** 2).sum(-1)
        cols.append(f + 0.05 * torch.randn(n, generator=g, dtype=DT) + 2.0 + case.get("shift", 0.0))
    Y = torch.stack(cols, -1)
    Yvar = 0.01 + 0.02 * torch.rand(n, m, generator=g, dtype=DT)
    return X, Y, Yvar


def test_points(case: dict[str, Any], q: int, seed: int) -> Any:
    g = torch.Generator().manual_seed(seed)
    d = case.get("d", 2)
    if case.get("one_hot"):
        X = torch.cat(
            [
                torch.rand(q, 1, generator=g, dtype=DT),
                torch.nn.functional.one_hot(torch.arange(q) % 3, 3).to(DT),
            ],
            -1,
        )
    else:
        X = torch.rand(q, d, generator=g, dtype=DT)
    if case.get("tasks"):
        X[:, -1] = torch.tensor([float(i % 3) for i in range(q)], dtype=DT)
    if "int_col" in case:
        X[:, case["int_col"]] = 3.0 * torch.rand(q, generator=g, dtype=DT)
    return X * case.get("scale_x", 1.0) + case.get("shift_x", 0.0)


def build_input(case: dict[str, Any], lib: str) -> Any:
    if case["input"] is None:
        return None
    return _helpers().build_input(case["input"], lib)


def build_model(case: dict[str, Any], X: Any, Y: Any, Yvar: Any) -> Any:
    from botorch.models import SingleTaskGP

    h = _helpers()
    kw: dict[str, Any] = {}
    if case["outcome"] != "default":
        kw["outcome_transform"] = h.build_outcome(case["outcome"], "bt", case.get("m", 1))
    tf = build_input(case, "bt")
    if tf is not None:
        kw["input_transform"] = tf
    model = SingleTaskGP(X, Y, train_Yvar=Yvar if case.get("fixed_noise") else None, **kw)
    return model.to(DT)


def flat(model: Any) -> list[float]:
    return [v for _, p in model.named_parameters() for v in p.detach().reshape(-1).tolist()]


def set_flat(model: Any, values: list[float]) -> None:
    offset = 0
    with torch.no_grad():
        for _, p in model.named_parameters():
            n = p.numel()
            p.copy_(torch.tensor(values[offset : offset + n], dtype=DT).reshape(p.shape))
            offset += n


def mll_value_and_grad(model: Any) -> tuple[list[float], list[float]]:
    from gpytorch.mlls import ExactMarginalLogLikelihood

    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    model.train()
    model.zero_grad()
    out = mll(model(*model.train_inputs), model.train_targets)
    out.sum().backward()
    grad = []
    for _, p in model.named_parameters():
        g = p.grad if p.grad is not None else torch.zeros_like(p)
        grad.extend(g.reshape(-1).tolist())
    model.zero_grad()
    model.eval()
    return out.detach().reshape(-1).tolist(), grad


def bounds_of(model: Any) -> list[list[float]]:
    from botorch.optim.utils import get_parameters_and_bounds
    from gpytorch.mlls import ExactMarginalLogLikelihood

    _, bounds = get_parameters_and_bounds(ExactMarginalLogLikelihood(model.likelihood, model))
    bounds = {k.removeprefix("model."): v for k, v in bounds.items()}
    out = []
    for name, p in model.named_parameters():
        lo, hi = bounds.get(name, (-math.inf, math.inf))
        n = p.numel()
        out.extend(zip(_bound_list(lo, n, -math.inf), _bound_list(hi, n, math.inf), strict=True))
    return [list(b) for b in out]


def _bound_list(bound: Any, n: int, default: float) -> list[float]:
    """A (scalar or per-element tensor) bound as one float per element."""
    if bound is None:
        return [default] * n
    if torch.is_tensor(bound) and bound.numel() > 1:
        return [float(b) for b in bound.reshape(-1)]
    return [float(bound)] * n


def random_raw(model: Any, seed: int) -> list[float]:
    """Random raw parameters within the optimizer bounds."""
    g = torch.Generator().manual_seed(seed)
    bounds = bounds_of(model)
    values = []
    i = 0
    for name, p in model.named_parameters():
        for _ in range(p.numel()):
            lo, hi = bounds[i]
            if "raw_noise" in name:
                v = 0.02 + 0.06 * torch.rand(1, generator=g, dtype=DT).item()
            elif "raw_constant" in name:
                v = 0.4 * torch.randn(1, generator=g, dtype=DT).item()
            elif "concentration" in name:
                v = 0.4 + 1.6 * torch.rand(1, generator=g, dtype=DT).item()
            elif "imputation" in name:
                v = torch.randn(1, generator=g, dtype=DT).item()
            else:
                v = 0.3 + 0.5 * torch.rand(1, generator=g, dtype=DT).item()
            values.append(min(max(v, lo), hi))
            i += 1
    return values


def _moments(post: Any) -> tuple[Any, Any] | None:
    try:
        return post.mean, post.variance
    except NotImplementedError:
        return None


def posterior_record(model: Any, case: dict[str, Any]) -> dict[str, Any]:
    model.eval()
    m = case.get("m", 1)
    if any(s.get("outputs") is not None for s in (case["outcome"] or []) if isinstance(s, dict)):
        # BoTorch's `untransform_posterior` raises NotImplementedError for
        # transforms of some outputs (the live tests compare these posteriors
        # with single-output models).
        return {}
    Xq = test_points(case, 4, 777).unsqueeze(-2)
    rec: dict[str, Any] = {"x": Xq.squeeze(-2)}
    Xg = Xq.clone().requires_grad_(True)
    post = model.posterior(Xg)
    mv = _moments(post)
    if mv is not None:
        mean, var = mv
        (dm,) = torch.autograd.grad(mean.sum(), Xg, retain_graph=True, allow_unused=True)
        (dv,) = torch.autograd.grad(var.sum(), Xg, allow_unused=True)
        rec.update(
            {
                "mean": mean.detach().reshape(4, -1),
                "variance": var.detach().reshape(4, -1),
                "dmean_dx": (torch.zeros_like(Xq) if dm is None else dm).squeeze(-2),
                "dvariance_dx": (torch.zeros_like(Xq) if dv is None else dv).squeeze(-2),
            }
        )
        with torch.no_grad():
            pn = model.posterior(Xq, observation_noise=True)
            rec["mean_noisy"] = pn.mean.reshape(4, -1)
            rec["variance_noisy"] = pn.variance.reshape(4, -1)
    # Joint samples of q = 3 points from given base samples.
    Xj = test_points(case, 3, 778)
    Xjg = Xj.clone().requires_grad_(True)
    pj = model.posterior(Xjg)
    S = 8
    g = torch.Generator().manual_seed(9)
    shape = pj.base_sample_shape
    Z = torch.randn(S, *shape, generator=g, dtype=DT)
    samples = pj.rsample_from_base_samples(torch.Size([S]), Z)
    W = torch.randn(samples.shape, generator=torch.Generator().manual_seed(10), dtype=DT)
    s = (W * samples).nansum()
    (ds,) = torch.autograd.grad(s, Xjg, allow_unused=True)
    rec.update(
        {
            "joint_x": Xj,
            "base_samples": Z.reshape(S, -1),
            "samples": samples.detach().reshape(S, -1),
            "sample_weights": W.reshape(S, -1),
            "dsamples_dx": torch.zeros_like(Xj) if ds is None else ds,
        }
    )
    if m == 1 and not case.get("append"):
        rec.update(acquisition_record(model, case, mv is not None))
    return rec


def acquisition_record(model: Any, case: dict[str, Any], moments: bool) -> dict[str, Any]:
    from botorch.acquisition import ExpectedImprovement, qExpectedImprovement
    from botorch.sampling import SobolQMCNormalSampler

    rec: dict[str, Any] = {}
    best_f = 2.5 + case.get("shift", 0.0)
    rec["best_f"] = best_f
    X1 = test_points(case, 4, 779).unsqueeze(-2)
    if moments:
        acq = ExpectedImprovement(model, best_f=best_f)
        Xg = X1.clone().requires_grad_(True)
        v = acq(Xg)
        (gx,) = torch.autograd.grad(v.sum(), Xg)
        rec["ei"] = {"x": X1.squeeze(-2), "value": v.detach(), "grad": gx.squeeze(-2)}
    X2 = test_points(case, 4, 780).reshape(2, 2, -1)
    k = model.posterior(X2[0]).base_sample_shape[-1]
    Z = torch.randn(32, 1, k, generator=torch.Generator().manual_seed(11), dtype=DT)
    sampler = SobolQMCNormalSampler(sample_shape=torch.Size([32]), seed=0)
    sampler.base_samples = Z.clone()
    acq = qExpectedImprovement(model, best_f=best_f, sampler=sampler)
    Xg = X2.clone().requires_grad_(True)
    v = acq(Xg)
    (gx,) = torch.autograd.grad(v.sum(), Xg)
    rec["qei"] = {
        "x": X2,
        "base_samples": Z.reshape(32, -1),
        "value": v.detach(),
        "grad": gx,
    }
    return rec


def perturbed_twin(
    case: dict[str, Any], X: Any, Y: Any, Yvar: Any, raw: list[float], seed: int, rel: float
) -> Any:
    """The model on training inputs perturbed at ``rel`` (relative)."""
    g = torch.Generator().manual_seed(seed)
    Xp = X * (1 + rel * torch.randn(X.shape, generator=g, dtype=DT))
    twin = build_model(case, Xp, Y, Yvar)
    set_flat(twin, raw)
    return twin, Xp


def fit(model: Any) -> dict[str, Any]:
    import botorch.optim.fit as of
    from botorch.fit import fit_gpytorch_mll
    from gpytorch.mlls import ExactMarginalLogLikelihood

    record: dict[str, Any] = {"attempts": 0}
    orig = of.fit_gpytorch_mll_scipy

    def recorder(mll: Any, *args: Any, **kwargs: Any) -> Any:
        record["attempts"] += 1
        record["r"] = orig(mll, *args, **kwargs)
        return record["r"]

    fit_gpytorch_mll(ExactMarginalLogLikelihood(model.likelihood, model), optimizer=recorder)
    r = record["r"]
    return {
        "loss": float(r.fval),
        "iterations": int(r.step),
        "status": r.status.name,
        "attempts": record["attempts"],
        "raw_params": flat(model),
    }


def _prefixes(model: Any) -> list[str] | None:
    tf = getattr(model, "input_transform", None)
    if tf is None:
        return None
    from botorch.models.transforms.input import ChainedInputTransform

    if isinstance(tf, ChainedInputTransform):
        return [f"input_transform.{k}." for k in tf.keys()]
    return ["input_transform."]


def model_case(case: dict[str, Any], seed: int) -> dict[str, Any]:
    h = _helpers()
    X, Y, Yvar = make_data(case, seed)
    out: dict[str, Any] = {
        "name": case["name"],
        "input": case["input"],
        "outcome": case["outcome"],
        "d": X.shape[-1],
        "m": case.get("m", 1),
        "fixed_noise": bool(case.get("fixed_noise")),
        "train_x": X,
        "train_y": Y,
        "train_yvar": Yvar if case.get("fixed_noise") else None,
    }
    if case.get("unsupported"):
        out["unsupported"] = True
        return out
    with h.botorch_fixes():
        model = build_model(case, X, Y, Yvar)
        names = [(n, list(p.shape)) for n, p in model.named_parameters()]
        default_raw = flat(model)
        model.eval()
        x_tf = model.train_inputs[0].detach().clone()
        dv, dg = mll_value_and_grad(model)
        raw = random_raw(model, seed + 1)
        set_flat(model, raw)
        v, g = mll_value_and_grad(model)
        model.eval()
        out.update(
            {
                "param_names": names,
                "prefixes": _prefixes(model),
                "bounds": bounds_of(model),
                "default_raw": default_raw,
                "transformed_train_x": x_tf,
                "default_mll": {"value": dv, "grad": dg},
                "raw": raw,
                "mll": {"value": v, "grad": g},
                "posterior": posterior_record(model, case),
            }
        )
        if case.get("fit"):
            set_flat(model, default_raw)
            fitted = fit(model)
            twins = []

            def add_twins(ks: range) -> None:
                for kk in ks:
                    scale = TWIN_SCALES[(kk - 1) % len(TWIN_SCALES)]
                    twin, Xp = perturbed_twin(case, X, Y, Yvar, default_raw, kk, scale)
                    r = fit(twin)
                    twins.append((Xp, r["loss"], r["attempts"]))

            add_twins(range(1, f6a.FIT_TWINS + 1))
            fitted["twin_train_x"] = [t for t, _, _ in twins]
            fitted["twin_losses"] = [loss for _, loss, _ in twins]
            fitted["twin_attempts"] = [a for _, _, a in twins]
            out["fit"] = fitted
    return out


def generate_6b_transforms(out: str) -> None:
    out = os.path.join(out, "6b_transforms")
    os.makedirs(out, exist_ok=True)
    warnings.simplefilter("ignore")
    torch.set_num_threads(1)
    for i, case in enumerate(cases()):
        fixture = model_case(case, seed=6100 + 13 * i)
        path = os.path.join(out, f"{case['name']}.json")
        with open(path, "w") as f:
            json.dump(enc(fixture), f)
        fit_s = ""
        if "fit" in fixture:
            fl = fixture["fit"]["twin_losses"]
            fit_s = (
                f" fit loss={fixture['fit']['loss']:+.8f} twins=[{min(fl):+.8f}, "
                f"{max(fl):+.8f}] iters={fixture['fit']['iterations']}"
            )
        print(f"phase 6b-transforms: {case['name']:28s}{fit_s}", file=sys.stderr)

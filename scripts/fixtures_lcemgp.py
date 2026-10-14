"""Fixtures of the Unity project's contextual models and of ``Interval``.

Written to ``<out>/unity_lcemgp/`` by ``generate_fixtures.py --phase
unity-lcemgp`` (BoTorch main or 0.18.1; ``LCEMGP`` is the same in both). The
references are the contextual models of the Unity project BOforUnity on
BoTorch (``botorch_pyo3/tests/lcemgp_reference.py``, copied from its
``context_support.py``): ``learned`` (BoTorch's ``LCEMGP`` with learned
embeddings and the Unity project's ``eval()``), ``provided`` (its
``_ProvidedEmbeddingLCEMGP``) and ``stock`` (learned plus fixed embeddings,
the task kernel resized to both).

* ``<variant>_K<K>_m<m>_d<d>.json``: one model per case with K = 2, 3, 5
  contexts (some without data), m = 1, 3 fixed embedding dimensions and
  d = 1, ..., 4 inputs:

  - the optimizer's parameter names, shapes and bounds and the default raw
    parameters (the embedding weights are given);
  - the marginal log likelihood and its gradient (BoTorch's autograd) at the
    default and at random raw parameters (embedding weights partly beyond
    the renormalization norm 1);
  - posteriors at random points: over the output task (with and without
    observation noise), over all tasks, and at inputs with a task column;
    the gradient of a weighted sum of the mean and covariance;
  - the constant means after ``eval()``;
  - ``fit_gpytorch_mll`` from the default parameters: the final loss,
    parameters and iterations, and BoTorch's own change of the final loss
    when the training inputs are perturbed by 1e-14 and when L-BFGS-B's
    variables are reversed (its noise floor: long L-BFGS-B runs on flat
    likelihoods amplify rounding);
  - qLogNEI (q = 1, 2, three candidate sets) and qLogNEHVI on a
    ``ModelListGP`` of this model and a second one on other targets:
    values and gradients with BoTorch's recorded base samples, and BoTorch's
    own change when the training inputs are perturbed by 1e-14.

* ``interval.json``: ``Interval`` constraints (sigmoid, softplus with the
  default inverse, no transform) on values, inverse values, derivatives,
  raw initial values and optimizer bounds; ``SingleTaskGP`` models with
  ``Interval`` constraints on the lengthscales, the outputscale and the
  noise: the MLL and its gradient at the default and random parameters and
  ``fit_gpytorch_mll``.

Non-finite numbers are written as the strings "inf", "-inf" and "nan".
"""

from __future__ import annotations

import copy
import json
import math
import os
import sys
import warnings
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import torch

DT = torch.float64
NUM_SAMPLES = 16
TESTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "botorch_pyo3", "tests")

CASES = [
    # variant, K, m, d, observations per observed context, unobserved contexts
    ("learned", 2, 1, 1, 5, [1]),
    ("learned", 3, 1, 2, 4, []),
    ("learned", 5, 1, 3, 3, [3, 4]),
    ("learned", 3, 1, 4, 5, [2]),
    ("provided", 2, 1, 2, 5, [1]),
    ("provided", 3, 3, 1, 4, []),
    ("provided", 5, 3, 4, 3, [4]),
    ("provided", 3, 1, 3, 4, [0]),
    ("stock", 3, 3, 2, 4, [2]),
    ("stock", 5, 1, 1, 3, []),
]


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


def _reference() -> Any:
    sys.path.insert(0, os.path.normpath(TESTS))
    import lcemgp_reference

    return lcemgp_reference


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@contextmanager
def recorded_draws(seed: int, record: dict[int, Any]) -> Iterator[None]:
    """BoTorch's Sobol normal draws replaced by seeded i.i.d. normal draws,
    recorded per dimension."""
    import botorch.sampling.normal as bt_normal

    orig = bt_normal.draw_sobol_normal_samples

    def draw(d: int, n: int, device: Any = None, dtype: Any = None, seed_: Any = None, **kw: Any):
        if d not in record:
            g = torch.Generator().manual_seed(seed * 7919 + d)
            record[d] = torch.randn(n, d, generator=g, dtype=DT)
        return record[d].clone().to(dtype=dtype or DT)

    bt_normal.draw_sobol_normal_samples = draw
    try:
        yield
    finally:
        bt_normal.draw_sobol_normal_samples = orig


def optimizer_parameters(model: Any) -> list[tuple[str, Any]]:
    """``(name, parameter)`` in the order of BoTorch's optimizer."""
    names = _reference().mll_parameter_names(model)
    params = dict(model.named_parameters())
    return [(n, params[n]) for n in names]


def flat_params(model: Any) -> list[float]:
    return [v for _, p in optimizer_parameters(model) for v in p.detach().reshape(-1).tolist()]


def set_flat_params(model: Any, values: list[float]) -> None:
    offset = 0
    with torch.no_grad():
        for _, p in optimizer_parameters(model):
            k = p.numel()
            p.copy_(torch.tensor(values[offset : offset + k], dtype=DT).reshape(p.shape))
            offset += k


def mll_value_and_grad(model: Any) -> tuple[float, list[float]]:
    from gpytorch.mlls import ExactMarginalLogLikelihood

    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    model.train()
    model.zero_grad()
    out = mll(model(*model.train_inputs), model.train_targets, *model.train_inputs)
    out.backward()
    grad = []
    for _, p in optimizer_parameters(model):
        g = p.grad if p.grad is not None else torch.zeros_like(p)
        grad.extend(g.reshape(-1).tolist())
    model.zero_grad()
    return out.item(), grad


def bounds_of(model: Any) -> list[list[float]]:
    from botorch.optim.utils import get_parameters_and_bounds
    from gpytorch.mlls import ExactMarginalLogLikelihood

    params, bounds = get_parameters_and_bounds(ExactMarginalLogLikelihood(model.likelihood, model))
    out = []
    for name, p in params.items():
        lo, hi = bounds.get(name, (-math.inf, math.inf))
        lo = -math.inf if lo is None else float(lo)
        hi = math.inf if hi is None else float(hi)
        out.extend([[lo, hi]] * p.numel())
    return out


def random_raw_params(model: Any, seed: int) -> list[float]:
    g = torch.Generator().manual_seed(seed)
    values = []
    for name, p in optimizer_parameters(model):
        k = p.numel()
        if "raw_noise" in name:
            v = 0.01 + 0.08 * torch.rand(k, generator=g, dtype=DT)
        elif "raw_constant" in name:
            v = 0.4 * torch.randn(k, generator=g, dtype=DT)
        elif name.startswith("covar_module"):
            v = 0.2 + 0.9 * torch.rand(k, generator=g, dtype=DT)
        elif name.startswith("task_covar_module_base"):
            v = 0.3 + 1.5 * torch.rand(k, generator=g, dtype=DT)
        elif name.startswith("emb_layers"):
            v = 1.3 * torch.randn(k, generator=g, dtype=DT)
        elif name == "raw_context_specific_weight":
            v = 0.5 * torch.rand(k, generator=g, dtype=DT)
        else:
            raise KeyError(name)
        values.extend(v.tolist())
    return values


def perturbed_twin(model: Any, seed: int, rel: float = 1e-14) -> Any:
    """A copy whose training inputs (not the task column) are perturbed by
    ``rel``: BoTorch's own sensitivity to rounding."""
    twin = copy.deepcopy(model)
    g = torch.Generator().manual_seed(seed)
    X = twin.train_inputs[0].clone()
    tf = twin._task_feature
    cols = [j for j in range(X.shape[-1]) if j != tf]
    X[:, cols] = X[:, cols] * (1 + rel * torch.randn(X[:, cols].shape, generator=g, dtype=DT))
    twin.train_inputs = (X,)
    twin.prediction_strategy = None
    return twin


def make_data(seed: int, K: int, d: int, n_per: int, unobserved: list[int], shift: float = 0.0):
    g = torch.Generator().manual_seed(seed)
    observed = [k for k in range(K) if k not in unobserved]
    rows = [k for k in observed for _ in range(n_per)]
    X = torch.rand(len(rows), d, generator=g, dtype=DT)
    t = torch.tensor(rows, dtype=DT)
    offsets = torch.rand(K, generator=g, dtype=DT)
    Y = torch.sin(3 * X[:, 0] + shift) + (X[:, 1:] ** 2).sum(-1) + offsets[t.long()]
    Y = (Y + 0.05 * torch.randn(len(rows), generator=g, dtype=DT)).unsqueeze(-1)
    return torch.cat([X, t.unsqueeze(-1)], -1), Y


def build(variant: str, X: Any, Y: Any, K: int, emb: Any, current: int, w0: Any) -> Any:
    model = _reference().build_model("bt", variant, X, Y, K, emb, current)
    if variant != "provided":
        with torch.no_grad():
            model.emb_layers[0].weight.copy_(w0)
    return model


def fit(model: Any, reverse: bool = False) -> dict[str, Any]:
    """BoTorch's ``fit_gpytorch_mll``; with ``reverse``, L-BFGS-B's variables
    are in reverse order (the same problem with other rounding)."""
    import botorch.optim.fit as of
    from botorch.fit import fit_gpytorch_mll
    from botorch.optim.utils import get_parameters_and_bounds
    from gpytorch.mlls import ExactMarginalLogLikelihood

    record: dict[str, Any] = {}
    orig = of.fit_gpytorch_mll_scipy

    def recorder(mll: Any, *args: Any, **kwargs: Any) -> Any:
        if reverse:
            params, bounds = get_parameters_and_bounds(mll)
            kwargs.update(parameters=dict(list(params.items())[::-1]), bounds=bounds)
        result = orig(mll, *args, **kwargs)
        record["result"] = result
        return result

    fit_gpytorch_mll(ExactMarginalLogLikelihood(model.likelihood, model), optimizer=recorder)
    r = record["result"]
    return {
        "loss": float(r.fval),
        "iterations": int(r.step),
        "status": r.status.name,
        "raw_params": flat_params(model),
    }


def values_and_grads(acqf: Any, X: Any) -> tuple[Any, Any]:
    Xg = X.clone().requires_grad_(True)
    v = acqf(Xg)
    (g,) = torch.autograd.grad(v.sum(), Xg)
    return v.detach(), g


def floor(pairs: list[tuple[Any, Any]], base: tuple[Any, Any]) -> tuple[float, float]:
    fv = max(((v - base[0]).abs() / base[0].abs().clamp_min(1.0)).max().item() for v, _ in pairs)
    fg = max(
        ((g - base[1]).abs().max() / base[1].abs().max().clamp_min(1e-300)).item() for _, g in pairs
    )
    return fv, fg


# ---------------------------------------------------------------------------
# Model cases
# ---------------------------------------------------------------------------


def model_case(
    index: int, variant: str, K: int, m: int, d: int, n_per: int, unobs: list[int]
) -> dict:
    from botorch.acquisition.logei import qLogNoisyExpectedImprovement
    from botorch.acquisition.multi_objective.logei import (
        qLogNoisyExpectedHypervolumeImprovement,
    )
    from botorch.models import ModelListGP
    from botorch.sampling.normal import SobolQMCNormalSampler

    seed = 100 + index
    current = unobs[0] if unobs else K - 1
    X, Y = make_data(seed, K, d, n_per, unobs)
    g = torch.Generator().manual_seed(seed + 1)
    emb = torch.rand(K, m, generator=g, dtype=DT)
    w0 = torch.randn(K, 1, generator=g, dtype=DT) * 1.2
    model = build(variant, X, Y, K, emb, current, w0)
    case: dict[str, Any] = {
        "name": f"{variant}_K{K}_d{d}" if variant == "learned" else f"{variant}_K{K}_m{m}_d{d}",
        "variant": variant,
        "num_contexts": K,
        "current": current,
        "unobserved": unobs,
        "d": d,
        "train_x": X,
        "train_y": Y,
        "embeddings": emb if variant != "learned" else None,
        "embedding_init": w0 if variant != "provided" else None,
        "param_names": [[n, p.numel()] for n, p in optimizer_parameters(model)],
        "bounds": bounds_of(model),
        "default_raw_params": flat_params(model),
    }
    v, gr = mll_value_and_grad(model)
    case["default_mll"] = {"value": v, "grad": gr}
    # The fit from the default parameters and BoTorch's noise floor.
    # (The MLL evaluation above renormalized the embedding weights in place.)
    initial = flat_params(model)
    fitted = fit(copy.deepcopy(model))
    fitted["initial_raw_params"] = initial
    losses = [fit(perturbed_twin(model, rep + 1))["loss"] for rep in range(4)]
    fitted["floor_inputs"] = max(abs(v - fitted["loss"]) for v in losses)
    fitted["floor_order"] = abs(fit(copy.deepcopy(model), reverse=True)["loss"] - fitted["loss"])
    fitted["floor"] = max(fitted["floor_inputs"], fitted["floor_order"])
    case["fit"] = fitted
    # Random hyperparameters.
    raw = random_raw_params(model, seed + 2)
    set_flat_params(model, raw)
    case["raw_params"] = raw
    v, gr = mll_value_and_grad(model)
    case["mll"] = {"value": v, "grad": gr}
    case["raw_params_after_forward"] = flat_params(model)  # renormalized embeddings
    model.eval()
    case["eval_constants"] = [c.item() for c in model.mean_module.base_means.parameters()]
    gx = torch.Generator().manual_seed(seed + 3)
    q = 3
    test_x = torch.rand(q, d, generator=gx, dtype=DT)
    with torch.no_grad():
        post = model.posterior(test_x)
        noisy = model.posterior(test_x, observation_noise=True)
        all_tasks = list(range(K))
        post_all = model.posterior(test_x, output_indices=all_tasks)
        tasks = torch.tensor([[float(k % K)] for k in range(q)], dtype=DT)
        x_full = torch.cat([test_x, tasks], -1)
        post_full = model.posterior(x_full)
    case["test_x"] = test_x
    case["posterior"] = {
        "mean": post.mean.reshape(-1),
        "covariance": post.distribution.covariance_matrix,
        "covariance_noisy": noisy.distribution.covariance_matrix,
    }
    case["posterior_all_tasks"] = {
        "tasks": all_tasks,
        "mean": post_all.mean,
        "covariance": post_all.distribution.covariance_matrix,
    }
    case["posterior_task_inputs"] = {
        "x": x_full,
        "mean": post_full.mean.reshape(-1),
        "covariance": post_full.distribution.covariance_matrix,
    }
    ma = torch.randn(q, generator=gx, dtype=DT)
    ca = torch.randn(q, q, generator=gx, dtype=DT)
    xg = test_x.clone().requires_grad_(True)
    p = model.posterior(xg)
    s = (p.mean.reshape(-1) * ma).sum() + (p.distribution.covariance_matrix * ca).sum()
    (gvjp,) = torch.autograd.grad(s, xg)
    case["vjp"] = {"mean_adjoint": ma, "covariance_adjoint": ca, "grad": gvjp}
    case["eval_raw_params"] = flat_params(model)

    # qLogNEI with recorded base samples.
    Xb = X[:, :-1]
    case["qlognei"] = []
    for qq in (1, 2):
        Xc = torch.rand(3, qq, d, generator=gx, dtype=DT)
        draws: dict[int, Any] = {}
        evals = []
        for mdl in [model] + [perturbed_twin(model, s) for s in (1, 2)]:
            with recorded_draws(seed + qq, draws):
                sampler = SobolQMCNormalSampler(torch.Size([NUM_SAMPLES]), seed=0)
                acqf = qLogNoisyExpectedImprovement(
                    mdl, X_baseline=Xb, sampler=sampler, prune_baseline=False, cache_root=False
                )
                evals.append(values_and_grads(acqf, Xc))
        fv, fg = floor(evals[1:], evals[0])
        case["qlognei"].append(
            {
                "q": qq,
                "x": Xc,
                "draws": {d_: z for d_, z in draws.items()},
                "values": evals[0][0],
                "grads": evals[0][1],
                "floor_values": fv,
                "floor_grads": fg,
            }
        )

    # qLogNEHVI on a list of this model and a second one (other targets).
    _, Y2 = make_data(seed, K, d, n_per, unobs, shift=1.3)
    model2 = build(variant, X, Y2, K, emb, current, torch.randn(K, 1, generator=gx, dtype=DT))
    raw2 = random_raw_params(model2, seed + 4)
    set_flat_params(model2, raw2)
    model2_raw_before = raw2
    lst = ModelListGP(model, model2)
    lst.eval()
    with torch.no_grad():
        means = lst.posterior(Xb).mean
    ref = (means.min(0).values - 0.1).tolist()
    case["nehvi"] = {
        "train_y2": Y2,
        "raw_params2": model2_raw_before,
        "ref_point": ref,
        "evaluations": [],
    }
    for qq in (1, 2):
        Xc = torch.rand(3, qq, d, generator=gx, dtype=DT)
        draws = {}
        evals = []
        lists = [lst] + [
            ModelListGP(perturbed_twin(model, s), perturbed_twin(model2, s)) for s in (1, 2)
        ]
        for lm in lists:
            lm.eval()
            with recorded_draws(seed + 10 + qq, draws):
                sampler = SobolQMCNormalSampler(torch.Size([NUM_SAMPLES]), seed=0)
                acqf = qLogNoisyExpectedHypervolumeImprovement(
                    lm, ref_point=ref, X_baseline=Xb, sampler=sampler, prune_baseline=False
                )
                evals.append(values_and_grads(acqf, Xc))
        fv, fg = floor(evals[1:], evals[0])
        case["nehvi"]["evaluations"].append(
            {
                "q": qq,
                "x": Xc,
                "draws": {d_: z for d_, z in draws.items()},
                "values": evals[0][0],
                "grads": evals[0][1],
                "floor_values": fv,
                "floor_grads": fg,
            }
        )
    case["nehvi"]["raw_params2_after"] = flat_params(model2)
    return case


# ---------------------------------------------------------------------------
# Interval
# ---------------------------------------------------------------------------


INTERVALS = [
    # lower, upper, transform (with GPyTorch's default inverse), initial value
    (0.0, 2.0, None, 1.0),
    (0.1, 3.0, "sigmoid", 1.0),
    (0.05, 4.0, "sigmoid", None),
    (-1.0, 1.5, "softplus", 0.2),
    (1e-4, 0.5, None, None),
    (0.025, 10.0, "sigmoid", 0.6931471805599453),
]


def interval_fixture() -> dict:
    from botorch.models import SingleTaskGP
    from gpytorch.constraints import Interval
    from gpytorch.kernels import RBFKernel, ScaleKernel
    from gpytorch.likelihoods import GaussianLikelihood

    raws = [-3.0, -0.5, 0.0, 0.3, 1.7, 4.0]
    constraints = []
    for lo, hi, tf, init in INTERVALS:
        kw: dict[str, Any] = {"initial_value": init}
        if tf is None:
            kw["transform"] = None
        elif tf == "softplus":
            kw["transform"] = torch.nn.functional.softplus
        c = Interval(lo, hi, **kw).to(DT)
        x = torch.tensor(raws, dtype=DT, requires_grad=True)
        v = c.transform(x)
        (dv,) = torch.autograd.grad(v.sum(), x)
        vals = torch.tensor(
            [lo + 0.13 * (hi - lo), lo + 0.5 * (hi - lo), lo + 0.91 * (hi - lo)], dtype=DT
        )
        lob, hib = c.inverse_transform(c.lower_bound), c.inverse_transform(c.upper_bound)
        constraints.append(
            {
                "lower": lo,
                "upper": hi,
                "transform": tf,
                "initial_value": init,
                "stored_bounds": [c.lower_bound.item(), c.upper_bound.item()],
                "raw": raws,
                "values": v.detach(),
                "derivatives": dv,
                "actual": vals,
                "inverse": c.inverse_transform(vals),
                "initial_raw": None if c.initial_value is None else c.initial_value.item(),
                "raw_bounds": [lob.item(), hib.item()],
            }
        )

    models = []
    g = torch.Generator().manual_seed(77)
    for idx, kind in enumerate(["lengthscale_box", "lengthscale_sigmoid", "scale_noise"]):
        d = 2 + idx
        X = torch.rand(14, d, generator=g, dtype=DT)
        Y = (
            torch.sin(4 * X[:, :1])
            + X[:, 1:].pow(2).sum(-1, keepdim=True)
            + 0.05 * torch.randn(14, 1, generator=g, dtype=DT)
        )
        if kind == "lengthscale_box":
            covar = RBFKernel(
                ard_num_dims=d,
                lengthscale_constraint=Interval(0.05, 2.0, transform=None, initial_value=0.7),
            )
            lik = None
        elif kind == "lengthscale_sigmoid":
            covar = RBFKernel(
                ard_num_dims=d, lengthscale_constraint=Interval(0.05, 3.0, initial_value=0.9)
            )
            lik = None
        else:
            covar = ScaleKernel(
                RBFKernel(ard_num_dims=d, lengthscale_constraint=Interval(0.02, 4.0)),
                outputscale_constraint=Interval(0.1, 5.0, transform=None, initial_value=1.0),
            )
            lik = GaussianLikelihood(
                noise_constraint=Interval(1e-4, 0.5, transform=None, initial_value=0.05)
            )
        model = SingleTaskGP(X, Y, covar_module=covar, likelihood=lik)
        entry: dict[str, Any] = {
            "kind": kind,
            "train_x": X,
            "train_y": Y,
            "param_names": [[n, p.numel()] for n, p in model.named_parameters()],
            "default_raw_params": [
                v for _, p in model.named_parameters() for v in p.detach().reshape(-1).tolist()
            ],
            "bounds": bounds_of(model),
        }
        v, gr = mll_value_and_grad_plain(model)
        entry["default_mll"] = {"value": v, "grad": gr}
        entry["fit"] = fit_plain(copy.deepcopy(model))
        losses = []
        for rep in range(4):
            twin = copy.deepcopy(model)
            gen = torch.Generator().manual_seed(rep + 1)
            X0 = twin.train_inputs[0]
            twin.train_inputs = (X0 * (1 + 1e-14 * torch.randn(X0.shape, generator=gen, dtype=DT)),)
            twin.prediction_strategy = None
            losses.append(fit_plain(twin)["loss"])
        entry["fit"]["floor"] = max(abs(v - entry["fit"]["loss"]) for v in losses)
        # Random raw parameters within the optimizer bounds.
        bounds = entry["bounds"]
        raw = []
        for lo, hi in bounds:
            u = torch.rand(1, generator=g, dtype=DT).item()
            if math.isfinite(lo) and math.isfinite(hi):
                raw.append(lo + (0.1 + 0.8 * u) * (hi - lo))
            elif math.isfinite(lo):
                raw.append(lo + 0.05 + 0.5 * u)
            else:
                raw.append(2.0 * u - 1.0)
        offset = 0
        with torch.no_grad():
            for _, p in model.named_parameters():
                k = p.numel()
                p.copy_(torch.tensor(raw[offset : offset + k], dtype=DT).reshape(p.shape))
                offset += k
        entry["raw_params"] = [
            v for _, p in model.named_parameters() for v in p.detach().reshape(-1).tolist()
        ]
        v, gr = mll_value_and_grad_plain(model)
        entry["mll"] = {"value": v, "grad": gr}
        models.append(entry)
    return {"constraints": constraints, "models": models}


def mll_value_and_grad_plain(model: Any) -> tuple[float, list[float]]:
    from gpytorch.mlls import ExactMarginalLogLikelihood

    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    model.train()
    model.zero_grad()
    out = mll(model(*model.train_inputs), model.train_targets)
    out.backward()
    grad = [v for _, p in model.named_parameters() for v in p.grad.reshape(-1).tolist()]
    model.zero_grad()
    model.eval()
    return out.item(), grad


def fit_plain(model: Any) -> dict[str, Any]:
    import botorch.optim.fit as of
    from botorch.fit import fit_gpytorch_mll
    from gpytorch.mlls import ExactMarginalLogLikelihood

    record: dict[str, Any] = {}
    orig = of.fit_gpytorch_mll_scipy

    def recorder(*args: Any, **kwargs: Any) -> Any:
        record["result"] = r = orig(*args, **kwargs)
        return r

    fit_gpytorch_mll(ExactMarginalLogLikelihood(model.likelihood, model), optimizer=recorder)
    r = record["result"]
    return {
        "loss": float(r.fval),
        "iterations": int(r.step),
        "status": r.status.name,
        "raw_params": [
            v for _, p in model.named_parameters() for v in p.detach().reshape(-1).tolist()
        ],
    }


def generate_unity_lcemgp(out: str) -> None:
    out = os.path.join(out, "unity_lcemgp")
    os.makedirs(out, exist_ok=True)
    warnings.simplefilter("ignore")
    for i, (variant, K, m, d, n_per, unobs) in enumerate(CASES):
        case = model_case(i, variant, K, m, d, n_per, unobs)
        with open(os.path.join(out, case["name"] + ".json"), "w") as f:
            json.dump(enc(case), f)
        print(
            f"unity-lcemgp: {case['name']}: mll={case['mll']['value']:+.6f} fit loss="
            f"{case['fit']['loss']:+.8f} iters={case['fit']['iterations']} floor="
            f"{case['fit']['floor']:.1e}",
            file=sys.stderr,
        )
    with open(os.path.join(out, "interval.json"), "w") as f:
        json.dump(enc(interval_fixture()), f)
    print("unity-lcemgp: wrote interval.json", file=sys.stderr)


if __name__ == "__main__":
    generate_unity_lcemgp(
        sys.argv[1]
        if len(sys.argv) > 1
        else os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "..", "botorch_core", "tests", "fixtures"
        )
    )

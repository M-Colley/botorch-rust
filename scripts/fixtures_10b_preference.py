"""Fixtures of phase 10b (preference learning): ``PairwiseGP`` with its
Laplace approximation and marginal log likelihood, the preference
acquisition functions, and MINPACK's ``hybrj`` (scipy's ``fsolve``).

Written to ``<out>/10b_preference/`` by ``generate_fixtures.py --phase
10b-preference`` (BoTorch main):

* ``hybrj_scipy.json``: scipy's ``fsolve(func, x0, fprime=...)`` on test
  systems whose values Rust computes bitwise like Python (polynomials in
  scalar float arithmetic): every evaluation point (scipy's two shape checks
  and its repeated evaluation at ``x0`` first), the solution, ``ier``,
  ``nfev`` and ``njev``.
* ``models.json``: ``PairwiseGP`` cases (probit and logit likelihoods, the
  default and custom ``ScaleKernel`` modules as JSON kernel specs of
  ``botorch_pyo3/tests/kernel_tree_helpers.py``, duplicates that are
  consolidated, ``Normalize``, ``jitter`` / ``xtol`` / ``maxfev`` /
  consolidation options, a nonzero mean constant). Every case records the
  data, the numpy draws of the initial utility (``np.random.seed`` before
  construction), the consolidated data, parameter names, shapes, bounds and
  default raw values; the Laplace state after construction (``_x0``,
  ``utility``, ``likelihood_hess``, ``covar``, ``covar_chol``,
  ``covar_inv``); a sequence of MLL evaluations (BoTorch's training forward
  pass and ``PairwiseLaplaceMarginalLogLikelihood``, warm-started from the
  previous one) with values, autograd gradients and states; posteriors at
  test points (``q = 1`` batches and a joint ``q = 3`` set: means,
  variances, covariances, and the autograd gradients of their sums or of a
  weighted sum); and ``fit_gpytorch_mll`` from the default parameters with
  BoTorch's final losses and attempts on 64 twins whose datapoints are
  perturbed at 1e-14 .. 1e-11 (the same comparisons and draws).
"""

from __future__ import annotations

import json
import math
import os
import sys
import warnings
from typing import Any

import numpy as np
import torch

DT = torch.float64
TWIN_SCALES = (1e-14, 1e-13, 1e-12, 1e-11)
FIT_TWINS = 64
TESTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "botorch_pyo3", "tests")


def _helpers() -> Any:
    sys.path.insert(0, os.path.normpath(TESTS))
    import kernel_tree_helpers

    return kernel_tree_helpers


# ---------------------------------------------------------------------------
# MINPACK hybrj against scipy
# ---------------------------------------------------------------------------

S5, S10 = math.sqrt(5.0), math.sqrt(10.0)


def _rosen(x: list[float]) -> list[float]:
    return [10.0 * (x[1] - x[0] * x[0]), 1.0 - x[0]]


def _rosen_j(x: list[float]) -> list[list[float]]:
    return [[-20.0 * x[0], 10.0], [-1.0, 0.0]]


def _powell(x: list[float]) -> list[float]:
    a = x[1] - 2.0 * x[2]
    b = x[0] - x[3]
    return [x[0] + 10.0 * x[1], S5 * (x[2] - x[3]), a * a, S10 * (b * b)]


def _powell_j(x: list[float]) -> list[list[float]]:
    a = x[1] - 2.0 * x[2]
    b = x[0] - x[3]
    return [
        [1.0, 10.0, 0.0, 0.0],
        [0.0, 0.0, S5, -S5],
        [0.0, 2.0 * a, -4.0 * a, 0.0],
        [2.0 * S10 * b, 0.0, 0.0, -2.0 * S10 * b],
    ]


def _btri(x: list[float]) -> list[float]:
    n = len(x)
    out = []
    for i in range(n):
        xm = x[i - 1] if i > 0 else 0.0
        xp = x[i + 1] if i < n - 1 else 0.0
        out.append((3.0 - 2.0 * x[i]) * x[i] - xm - 2.0 * xp + 1.0)
    return out


def _btri_j(x: list[float]) -> list[list[float]]:
    n = len(x)
    jac = [[0.0] * n for _ in range(n)]
    for i in range(n):
        jac[i][i] = 3.0 - 4.0 * x[i]
        if i > 0:
            jac[i][i - 1] = -1.0
        if i < n - 1:
            jac[i][i + 1] = -2.0
    return jac


def _bband(x: list[float]) -> list[float]:
    n = len(x)
    out = []
    for i in range(n):
        s = 0.0
        for j in range(max(0, i - 5), min(n, i + 2)):
            if j != i:
                s += x[j] * (1.0 + x[j])
        out.append(x[i] * (2.0 + 5.0 * x[i] * x[i]) + 1.0 - s)
    return out


def _bband_j(x: list[float]) -> list[list[float]]:
    n = len(x)
    jac = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(max(0, i - 5), min(n, i + 2)):
            if j != i:
                jac[i][j] = -(1.0 + 2.0 * x[j])
        jac[i][i] = 2.0 + 15.0 * x[i] * x[i]
    return jac


def _noroot(x: list[float]) -> list[float]:
    return [x[0] * x[0] + 1.0, x[1] - 2.0]


def _noroot_j(x: list[float]) -> list[list[float]]:
    return [[2.0 * x[0], 0.0], [0.0, 1.0]]


HYBRJ_PROBLEMS = {
    "rosenbrock": (_rosen, _rosen_j, [-1.2, 1.0], 1.49012e-8, 0),
    "rosenbrock_maxfev5": (_rosen, _rosen_j, [-1.2, 1.0], 1.49012e-8, 5),
    "powell_singular": (_powell, _powell_j, [3.0, -1.0, 0.0, 1.0], 1.49012e-8, 0),
    "broyden_tridiagonal": (_btri, _btri_j, [-1.0] * 10, 1e-6, 100),
    "broyden_banded": (_bband, _bband_j, [-1.0] * 10, 1e-10, 0),
    "no_root": (_noroot, _noroot_j, [0.5, 0.0], 1e-6, 100),
}


def hybrj_fixture() -> dict[str, Any]:
    from scipy import optimize

    out = {}
    for name, (f, j, x0, xtol, maxfev) in HYBRJ_PROBLEMS.items():
        calls: list[Any] = []

        def ff(x: Any, f: Any = f, calls: list[Any] = calls) -> Any:
            calls.append(["f", [float(v) for v in x]])
            return np.array(f([float(v) for v in x]))

        def jj(x: Any, j: Any = j, calls: list[Any] = calls) -> Any:
            calls.append(["j", [float(v) for v in x]])
            return np.array(j([float(v) for v in x]))

        x, info, ier, _ = optimize.fsolve(
            ff, x0, fprime=jj, xtol=xtol, maxfev=maxfev, full_output=True
        )
        out[name] = {
            "x0": x0,
            "xtol": xtol,
            "maxfev": maxfev,
            "calls": calls,
            "x": [float(v) for v in x],
            "ier": int(ier),
            "nfev": int(info["nfev"]),
            "njev": int(info["njev"]),
        }
    return out


# ---------------------------------------------------------------------------
# PairwiseGP
# ---------------------------------------------------------------------------


def _random_comparisons(gen: torch.Generator, n: int, m: int) -> list[list[int]]:
    out = []
    while len(out) < m:
        a, b = torch.randint(0, n, (2,), generator=gen).tolist()
        if a != b:
            out.append([a, b])
    return out


def model_cases() -> list[dict[str, Any]]:
    h = _helpers()
    k, sub, prior, con = h.kernel, h.sub, h.prior, h.constraint
    return [
        {"name": "probit_default_d2", "n": 8, "d": 2, "m": 12, "seed": 1},
        {"name": "logit_default_d3", "n": 10, "d": 3, "m": 16, "seed": 2, "likelihood": "logit"},
        {"name": "probit_dups_d2", "n": 9, "d": 2, "m": 14, "seed": 3, "dups": True},
        {
            "name": "logit_dups_d2",
            "n": 9,
            "d": 2,
            "m": 14,
            "seed": 4,
            "dups": True,
            "likelihood": "logit",
        },
        {"name": "probit_n25_d3", "n": 25, "d": 3, "m": 40, "seed": 5},
        {
            "name": "probit_matern_tree_d2",
            "n": 10,
            "d": 2,
            "m": 15,
            "seed": 6,
            "kernel": k(
                "ScaleKernel",
                sub(
                    k(
                        "MaternKernel",
                        nu=2.5,
                        ard_num_dims=2,
                        lengthscale_prior=prior("GammaPrior", 3.0, 6.0),
                    )
                ),
                outputscale_prior=prior("GammaPrior", 2.0, 0.15),
            ),
        },
        {
            "name": "logit_rbf_shared_tree_d2",
            "n": 9,
            "d": 2,
            "m": 12,
            "seed": 7,
            "likelihood": "logit",
            "kernel": k(
                "ScaleKernel", sub(k("RBFKernel", lengthscale_constraint=con("GreaterThan", 1e-3)))
            ),
        },
        {
            "name": "probit_sum_tree_d3",
            "n": 10,
            "d": 3,
            "m": 14,
            "seed": 8,
            "kernel": k(
                "ScaleKernel",
                sub(
                    k(
                        "AdditiveKernel",
                        sub(k("RBFKernel", ard_num_dims=2, active_dims=[0, 1])),
                        sub(k("MaternKernel", nu=1.5, active_dims=[2])),
                    )
                ),
            ),
        },
        {
            "name": "probit_normalize_d2",
            "n": 8,
            "d": 2,
            "m": 10,
            "seed": 9,
            "normalize": True,
            "scale": 5.0,
        },
        {
            "name": "probit_options_d2",
            "n": 8,
            "d": 2,
            "m": 12,
            "seed": 10,
            "options": {"jitter": 1e-5, "xtol": 1e-4, "maxfev": 8, "consolidate_atol": 0.05},
        },
        {"name": "probit_mean_d2", "n": 8, "d": 2, "m": 10, "seed": 11, "mean_constant": 0.3},
        {"name": "probit_saturated_d1", "n": 6, "d": 1, "m": 20, "seed": 12, "consistent": True},
    ]


def _data(case: dict[str, Any]) -> tuple[torch.Tensor, list[list[int]]]:
    gen = torch.Generator().manual_seed(1000 + case["seed"])
    n, d = case["n"], case["d"]
    X = torch.rand(n, d, generator=gen, dtype=DT) * case.get("scale", 1.0)
    if case.get("dups"):
        # Two near-duplicates of earlier points (merged by consolidation).
        X[n - 1] = X[0] + 2e-5
        X[n - 2] = X[3] - 3e-5
    if case.get("dups"):
        # Comparisons between a near-duplicate and its original become
        # self-comparisons after consolidation (a BoTorch bug, tested
        # separately); keep them out of the fixtures.
        pairs = _random_comparisons(gen, n, 3 * case["m"])
        bad = {(n - 1, 0), (0, n - 1), (n - 2, 3), (3, n - 2)}
        return X, [c for c in pairs if tuple(c) not in bad][: case["m"]]
    if case.get("consistent"):
        # Comparisons consistent with a utility, many of them, so that
        # probit z-scores reach the clamp at +-3.
        u = (X * 7.0).sin().sum(-1)
        comps = []
        while len(comps) < case["m"]:
            a, b = torch.randint(0, n, (2,), generator=gen).tolist()
            if a != b:
                comps.append([a, b] if u[a] > u[b] else [b, a])
        return X, comps
    return X, _random_comparisons(gen, n, case["m"])


def _build(case: dict[str, Any], X: torch.Tensor, comps: list[list[int]]) -> Any:
    from botorch.models import PairwiseGP
    from botorch.models.likelihoods.pairwise import (
        PairwiseLogitLikelihood,
        PairwiseProbitLikelihood,
    )
    from botorch.models.transforms.input import Normalize

    lik = PairwiseLogitLikelihood() if case.get("likelihood") == "logit" else None
    if lik is None and case.get("likelihood") == "probit_explicit":
        lik = PairwiseProbitLikelihood()
    covar = None
    if case.get("kernel") is not None:
        covar = _helpers().build_kernel(case["kernel"], "bt").to(DT)
    tf = Normalize(d=case["d"]) if case.get("normalize") else None
    np.random.seed(case["seed"])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = PairwiseGP(
            X,
            torch.tensor(comps, dtype=torch.long),
            likelihood=lik,
            covar_module=covar,
            input_transform=tf,
            **case.get("options", {}),
        )
        if case.get("mean_constant") is not None:
            model.mean_module.constant.data.fill_(case["mean_constant"])
            model._update(model.transform_inputs(model.datapoints))
    return model


def _json_num(v: float) -> Any:
    """A number for JSON (infinities as the strings ``"inf"`` / ``"-inf"``)."""
    if math.isinf(v):
        return "inf" if v > 0 else "-inf"
    return v


def _tolist(t: Any) -> Any:
    if isinstance(t, torch.Tensor):
        return t.detach().tolist()
    if isinstance(t, np.ndarray):
        return t.tolist()
    return t


def _state(model: Any) -> dict[str, Any]:
    return {
        "x0": _tolist(model._x0),
        "utility": _tolist(model.utility),
        "likelihood_hess": _tolist(model.likelihood_hess),
        "covar": _tolist(model.covar),
        "covar_chol": _tolist(model.covar_chol),
        "covar_inv": _tolist(model.covar_inv),
    }


def _params(model: Any) -> list[tuple[str, Any]]:
    return [(n, p) for n, p in model.named_parameters() if p.requires_grad]


def _mll_eval(model: Any) -> dict[str, Any]:
    """BoTorch's training forward pass and Laplace MLL with autograd."""
    from botorch.models.pairwise_gp import PairwiseLaplaceMarginalLogLikelihood

    model.train()
    warm = _tolist(model._x0)
    params = _params(model)
    raw = [v for _, p in params for v in p.detach().reshape(-1).tolist()]
    for _, p in params:
        p.grad = None
    mll = PairwiseLaplaceMarginalLogLikelihood(model.likelihood, model)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = model(model.unconsolidated_datapoints)
        value = mll(out, model.unconsolidated_comparisons)
    value.backward()
    grad = []
    for _, p in params:
        g = p.grad if p.grad is not None else torch.zeros_like(p)
        grad.extend(g.reshape(-1).tolist())
    rec = {"raw": raw, "warm": warm, "value": value.item(), "grad": grad}
    rec.update(_state(model))
    return rec


def _set_raw(model: Any, raw: list[float]) -> None:
    off = 0
    with torch.no_grad():
        for _, p in _params(model):
            k = p.numel()
            p.copy_(torch.tensor(raw[off : off + k], dtype=DT).reshape(p.shape))
            off += k


def _posteriors(model: Any, gen: torch.Generator, d: int) -> dict[str, Any]:
    model.eval()
    lo, hi = model.datapoints.min(0).values, model.datapoints.max(0).values
    X1 = (lo + (hi - lo) * torch.rand(5, 1, d, generator=gen, dtype=DT)).requires_grad_(True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        p1 = model.posterior(X1)
        mean1, var1 = p1.mean, p1.variance
        (mean1.sum() + var1.sum()).backward()
    X3 = lo + (hi - lo) * torch.rand(3, d, generator=gen, dtype=DT)
    # Include a training point (a posterior at observed data).
    X3[1] = model.datapoints[2]
    X3 = X3.requires_grad_(True)
    w_mean = torch.tensor([0.3, -0.7, 1.1], dtype=DT)
    w_cov = torch.tensor([[0.5, -0.2, 0.1], [-0.2, 0.8, 0.3], [0.1, 0.3, -0.4]], dtype=DT)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        p3 = model.posterior(X3)
        mean3 = p3.mean.squeeze(-1)
        cov3 = p3.distribution.covariance_matrix
        ((mean3 * w_mean).sum() + (cov3 * w_cov).sum()).backward()
    return {
        "x_q1": _tolist(X1.squeeze(-2)),
        "mean_q1": _tolist(mean1.reshape(-1)),
        "var_q1": _tolist(var1.reshape(-1)),
        "grad_q1": _tolist(X1.grad.squeeze(-2)),
        "x_q3": _tolist(X3),
        "mean_q3": _tolist(mean3),
        "cov_q3": _tolist(cov3),
        "w_mean": _tolist(w_mean),
        "w_cov": _tolist(w_cov),
        "grad_q3": _tolist(X3.grad),
        "hlcov_eye": _tolist(model.hlcov_eye),
    }


def fit(model: Any) -> dict[str, Any]:
    import botorch.optim.fit as of
    from botorch.fit import fit_gpytorch_mll
    from botorch.models.pairwise_gp import PairwiseLaplaceMarginalLogLikelihood

    record: dict[str, Any] = {"attempts": 0}
    orig = of.fit_gpytorch_mll_scipy

    def recorder(mll: Any, *args: Any, **kwargs: Any) -> Any:
        record["attempts"] += 1
        record["r"] = orig(mll, *args, **kwargs)
        return record["r"]

    mll = PairwiseLaplaceMarginalLogLikelihood(model.likelihood, model)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit_gpytorch_mll(mll, optimizer=recorder)
    r = record["r"]
    return {
        "loss": float(r.fval),
        "iterations": int(r.step),
        "attempts": record["attempts"],
        "raw": [v for _, p in _params(model) for v in p.detach().reshape(-1).tolist()],
        "utility": _tolist(model.utility),
    }


def perturbed(X: torch.Tensor, seed: int, rel: float) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return X * (1 + rel * torch.randn(X.shape, generator=g, dtype=DT))


def model_fixture(case: dict[str, Any], fits: bool = True) -> dict[str, Any]:
    from botorch.optim.utils import get_parameters_and_bounds

    X, comps = _data(case)
    model = _build(case, X, comps)
    n = model.datapoints.shape[-2]
    draws = np.random.RandomState(case["seed"]).standard_normal(n).tolist()
    params = _params(model)
    _, bounds = get_parameters_and_bounds(model)
    num_bounds = [
        (
            -math.inf if bounds.get(nm, (None, None))[0] is None else float(bounds[nm][0]),
            math.inf if bounds.get(nm, (None, None))[1] is None else float(bounds[nm][1]),
        )
        for nm, p in params
        for _ in range(p.numel())
    ]
    fixture: dict[str, Any] = {
        "name": case["name"],
        "likelihood": case.get("likelihood", "probit"),
        "kernel": case.get("kernel"),
        "normalize": bool(case.get("normalize")),
        "options": case.get("options", {}),
        "mean_constant": case.get("mean_constant", 0.0),
        "d": case["d"],
        "datapoints": _tolist(X),
        "comparisons": comps,
        "draws": draws,
        "consolidated_datapoints": _tolist(model.datapoints),
        "consolidated_comparisons": _tolist(model.comparisons),
        "consolidated_indices": _tolist(model.consolidated_indices),
        "param_names": [nm for nm, _ in params],
        "param_shapes": [list(p.shape) for _, p in params],
        "default_raw": [v for _, p in params for v in p.detach().reshape(-1).tolist()],
        "bounds": [[_json_num(lo), _json_num(hi)] for lo, hi in num_bounds],
        "state": _state(model),
    }
    # MLL evaluations: at the defaults, then at random raw parameters (each
    # warm-started from the previous evaluation's solution).
    gen = torch.Generator().manual_seed(2000 + case["seed"])
    evals = [_mll_eval(model)]
    default = fixture["default_raw"]
    for _ in range(3):
        raw = [v + 0.3 * float(torch.randn(1, generator=gen, dtype=DT)) for v in default]
        raw = [max(lo + 1e-3, min(hi, v)) for v, (lo, hi) in zip(raw, num_bounds, strict=True)]
        _set_raw(model, raw)
        evals.append(_mll_eval(model))
    fixture["mll_evals"] = evals
    fixture["posterior"] = _posteriors(model, gen, case["d"])
    if fits:
        fm = _build(case, X, comps)
        fixture["fit"] = fit(fm)
        twins = []
        for t in range(1, FIT_TWINS + 1):
            Xt = perturbed(X, 5000 + t, TWIN_SCALES[(t - 1) % len(TWIN_SCALES)])
            tm = _build(case, Xt, comps)
            r = fit(tm)
            twins.append((_tolist(Xt), r["loss"], r["attempts"]))
        fixture["fit"]["twin_datapoints"] = [t for t, _, _ in twins]
        fixture["fit"]["twin_losses"] = [loss for _, loss, _ in twins]
        fixture["fit"]["twin_attempts"] = [a for _, _, a in twins]
    return fixture


#: The model cases of the acquisition fixtures.
ACQUISITION_CASES = ("probit_default_d2", "logit_rbf_shared_tree_d2", "probit_normalize_d2")


def _vg(acqf: Any, X: torch.Tensor) -> tuple[list[Any], list[Any]]:
    X = X.clone().requires_grad_(True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        v = acqf(X)
    v.sum().backward()
    return _tolist(v), _tolist(X.grad)


def acquisition_fixture(case: dict[str, Any]) -> dict[str, Any]:
    """The preference acquisition functions on the model of ``case`` (as
    constructed): analytic EUBO (pairs; against a previous winner; through an
    ``AffineDeterministicModel`` outcome model), qEUBO with recorded base
    samples (also with pending points and through the outcome model), BALD
    with recorded draws, ``PairwiseMCPosteriorVariance`` with recorded base
    samples (``IdentityMCObjective``) and ``LearnedObjective`` with its
    sampler's base samples (values and a weighted vector-Jacobian product)."""
    from botorch.acquisition import active_learning as bt_al
    from botorch.acquisition import preference as bt_pref
    from botorch.acquisition.objective import IdentityMCObjective, LearnedObjective
    from botorch.models.deterministic import AffineDeterministicModel
    from botorch.sampling.normal import SobolQMCNormalSampler

    X, comps = _data(case)
    model = _build(case, X, comps)
    model.eval()
    d = case["d"]
    gen = torch.Generator().manual_seed(3000 + case["seed"])
    lo, hi = model.datapoints.min(0).values, model.datapoints.max(0).values

    def pts(*shape: int) -> torch.Tensor:
        return lo + (hi - lo) * torch.rand(*shape, d, generator=gen, dtype=DT)

    out: dict[str, Any] = {"name": case["name"]}
    Xp = pts(4, 2)
    v, g = _vg(bt_pref.AnalyticExpectedUtilityOfBestOption(model), Xp)
    out["eubo"] = {"x": _tolist(Xp), "values": v, "grads": g}
    winner = pts(1)
    Xw = pts(4, 1)
    v, g = _vg(bt_pref.AnalyticExpectedUtilityOfBestOption(model, previous_winner=winner), Xw)
    out["eubo_winner"] = {"winner": _tolist(winner), "x": _tolist(Xw), "values": v, "grads": g}
    dx = 3
    a = torch.rand(d, generator=gen, dtype=DT) * 0.2
    b = torch.rand(dx, d, generator=gen, dtype=DT) * 0.5
    om = AffineDeterministicModel(b=a, a=b)
    Xo = torch.rand(4, 2, dx, generator=gen, dtype=DT)
    v, g = _vg(bt_pref.AnalyticExpectedUtilityOfBestOption(model, outcome_model=om), Xo)
    out["eubo_outcome"] = {
        "a": _tolist(b),
        "b": _tolist(a),
        "x": _tolist(Xo),
        "values": v,
        "grads": g,
    }
    for key, q, pending, outcome in (
        ("qeubo", 3, None, None),
        ("qeubo_pending", 2, pts(1), None),
        ("qeubo_outcome", 2, None, om),
    ):
        sampler = SobolQMCNormalSampler(torch.Size([64]), seed=11)
        acqf = bt_pref.qExpectedUtilityOfBestOption(
            model, outcome_model=outcome, sampler=sampler, X_pending=pending
        )
        Xq = torch.rand(3, q, dx, generator=gen, dtype=DT) if outcome is not None else pts(3, q)
        v, g = _vg(acqf, Xq)
        out[key] = {
            "x": _tolist(Xq),
            "pending": _tolist(pending),
            "base_samples": _tolist(sampler.base_samples.reshape(64, -1)),
            "values": v,
            "grads": g,
        }
    bald = bt_pref.PairwiseBayesianActiveLearningByDisagreement(
        model, num_samples=64, std_noise=0.05
    )
    Xb = pts(3, 2)
    torch.manual_seed(77)
    eps = torch.empty(64, 3, dtype=DT).normal_()
    torch.manual_seed(77)
    v, g = _vg(bald, Xb)
    out["bald"] = {
        "x": _tolist(Xb),
        "eps": _tolist(eps),
        "std_noise": 0.05,
        "values": v,
        "grads": g,
    }
    sampler = SobolQMCNormalSampler(torch.Size([64]), seed=13)
    acqf = bt_al.PairwiseMCPosteriorVariance(
        model, objective=IdentityMCObjective(), sampler=sampler
    )
    Xm = pts(3, 4)
    v, g = _vg(acqf, Xm)
    out["mc_variance"] = {
        "x": _tolist(Xm),
        "base_samples": _tolist(sampler.base_samples.reshape(64, -1)),
        "values": v,
        "grads": g,
    }
    lobj = LearnedObjective(model, sample_shape=torch.Size([4]), seed=17)
    Y = pts(3, 2, 2).requires_grad_(True)
    vals = lobj(Y)
    w = torch.rand(vals.shape, generator=gen, dtype=DT)
    (vals * w).sum().backward()
    out["learned"] = {
        "y": _tolist(Y),
        "base_samples": _tolist(lobj.sampler.base_samples),
        "values": _tolist(vals),
        "weights": _tolist(w),
        "vjp": _tolist(Y.grad),
    }
    return out


def generate_10b_preference(out: str) -> None:
    torch.set_default_dtype(torch.float32)
    d = os.path.join(out, "10b_preference")
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "hybrj_scipy.json"), "w") as f:
        json.dump(hybrj_fixture(), f)
    print("phase 10b-preference: wrote hybrj_scipy.json", file=sys.stderr)
    models = []
    for case in model_cases():
        fx = model_fixture(case)
        fl = fx["fit"]["twin_losses"]
        print(
            f"phase 10b-preference: {case['name']}: fit loss={fx['fit']['loss']:+.8f} "
            f"twins=[{min(fl):+.8f}, {max(fl):+.8f}] attempts={fx['fit']['attempts']}",
            file=sys.stderr,
        )
        models.append(fx)
    with open(os.path.join(d, "models.json"), "w") as f:
        json.dump(models, f)
    print("phase 10b-preference: wrote models.json", file=sys.stderr)
    acq = [acquisition_fixture(c) for c in model_cases() if c["name"] in ACQUISITION_CASES]
    with open(os.path.join(d, "acquisition.json"), "w") as f:
        json.dump(acq, f)
    print("phase 10b-preference: wrote acquisition.json", file=sys.stderr)


if __name__ == "__main__":
    generate_10b_preference(sys.argv[1] if len(sys.argv) > 1 else ".")

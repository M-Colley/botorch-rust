"""Phase 9 fixtures: optimization and candidate generation (``9-optim``).

Written to ``<out>/phase9_optim/`` by ``generate_fixtures.py --phase 9-optim``
(BoTorch main). Every case holds a ``SingleTaskGP`` (training data, raw
hyperparameters in BoTorch's order; explicit values, RBF kernel) and BoTorch's
results from identical inputs, so that no random number of BoTorch's is
needed:

* ``optimize_acqf_discrete``: analytic LogEI for ``q = 1``; qLogEI on
  ``PosteriorMeanModel`` (deterministic, so the values do not depend on the
  sampler) for ``q = 3`` with ``X_avoid`` and a linear constraint;
* ``optimize_acqf_mixed`` from given initial conditions (``q = 1``: the best
  and all restarts of the best fixed features; ``q = 2``);
* ``optimize_acqf_discrete_local_search`` from given initial conditions
  (``q = 1``) and with BoTorch's random lattice points replaced by recorded
  points (``q = 3``);
* ``optimize_acqf_mixed_alternating`` from recorded starting points
  (BoTorch's ``generate_starting_points`` replaced), ``q = 1`` and ``q = 2``;
* ``gen_candidates_torch`` (Adam, default and custom settings),
  ``gen_candidates_scipy`` in the serial (joint) mode, ``optimize_acqf`` with
  a post-processing function, ``optimize_acqf_cyclic`` with a deterministic
  ``ic_generator``;
* ``ExpMAStoppingCriterion`` weights and decisions, ``prune_candidates``,
  ``MaxPosteriorSampling.maximize_samples`` (with and without replacement)
  and the constrained scores, the truncated normal perturbations of
  ``sample_truncated_normal_perturbations`` (from recorded uniform draws);
* ``fit_gpytorch_mll_torch`` (step limit, and the default stopping criterion)
  and ``fit_gpytorch_mll_scipy``: final raw hyperparameters, losses, steps;
* ``gen_loo_cv_folds`` and ``batch_cross_validation``: posterior means and
  variances at the held-out points.
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

CASES = [
    dict(name="optim_d3", d=3, n=12, seed=9100),
    dict(name="optim_d4", d=4, n=16, seed=9200),
]


def enc(x: Any) -> Any:
    """JSON-compatible values (tensors as lists)."""
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().tolist()
    if isinstance(x, float) and not math.isfinite(x):
        return "nan" if math.isnan(x) else ("inf" if x > 0 else "-inf")
    if isinstance(x, dict):
        return {str(k): enc(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [enc(v) for v in x]
    return x


def _model(d: int, n: int, seed: int) -> tuple[Any, dict]:
    """A SingleTaskGP with explicit hyperparameters and its specification."""
    from botorch.models import SingleTaskGP

    g = torch.Generator().manual_seed(seed)
    x = torch.rand(n, d, generator=g, dtype=DT)
    y = (
        torch.sin(4 * x[:, :1]) + (x[:, 1:] - 0.4).pow(2).sum(-1, keepdim=True)
    ) + 0.05 * torch.randn(n, 1, generator=g, dtype=DT)
    model = SingleTaskGP(x, y)
    with torch.no_grad():
        model.covar_module.lengthscale = 0.3 + 0.5 * torch.rand(1, d, generator=g, dtype=DT)
        model.likelihood.noise = torch.tensor([0.02], dtype=DT)
        model.mean_module.constant = torch.tensor(0.1, dtype=DT)
    model.eval()
    named = list(model.named_parameters())
    spec = {
        "train_x": enc(x),
        "train_y": enc(y),
        "param_names": [k for k, _ in named],
        "raw_params": [v for _, p in named for v in p.detach().reshape(-1).tolist()],
    }
    return model, spec


def _acqfs(model: Any) -> tuple[Any, Any]:
    """Analytic LogEI and qLogEI on the posterior mean (deterministic)."""
    from botorch.acquisition import LogExpectedImprovement, qLogExpectedImprovement
    from botorch.models.deterministic import PosteriorMeanModel
    from botorch.sampling.stochastic_samplers import StochasticSampler

    best_f = 1.0
    logei = LogExpectedImprovement(model, best_f=best_f)
    qlogei = qLogExpectedImprovement(
        PosteriorMeanModel(model), best_f=best_f, sampler=StochasticSampler(torch.Size([1]))
    )
    return logei, qlogei


def _discrete(logei: Any, qlogei: Any, d: int, g: Any) -> dict:
    from botorch.optim.optimize import optimize_acqf_discrete

    choices = torch.rand(60, d, generator=g, dtype=DT)
    x1, v1 = optimize_acqf_discrete(logei, 1, choices)
    avoid = choices[:6]
    cons = [(torch.tensor([0, 1]), torch.tensor([1.0, 1.0], dtype=DT), 0.6)]
    x3, v3 = optimize_acqf_discrete(qlogei, 3, choices, X_avoid=avoid, inequality_constraints=cons)
    x3n, v3n = optimize_acqf_discrete(qlogei, 3, choices, unique=False)
    return {
        "choices": enc(choices),
        "x_avoid": enc(avoid),
        "constraint": {"indices": [0, 1], "coefficients": [1.0, 1.0], "rhs": 0.6},
        "q1": {"candidates": enc(x1), "value": float(v1)},
        "q3": {"candidates": enc(x3), "values": enc(v3)},
        "q3_not_unique": {"candidates": enc(x3n), "values": enc(v3n)},
    }


def _mixed(logei: Any, qlogei: Any, d: int, g: Any) -> dict:
    from botorch.optim.optimize import optimize_acqf_mixed

    bounds = torch.stack([torch.zeros(d, dtype=DT), torch.ones(d, dtype=DT)])
    ics = torch.rand(5, 1, d, generator=g, dtype=DT)
    ffl = [{0: 0.0}, {0: 0.5, d - 1: 0.25}, {0: 1.0}]
    x1, v1 = optimize_acqf_mixed(logei, bounds, 1, 5, ffl, batch_initial_conditions=ics)
    xa, va = optimize_acqf_mixed(
        logei, bounds, 1, 5, ffl, batch_initial_conditions=ics, return_best_only=False
    )
    x2, v2 = optimize_acqf_mixed(qlogei, bounds, 2, 5, ffl, batch_initial_conditions=ics)
    return {
        "initial_conditions": enc(ics),
        "fixed_features_list": [{str(k): v for k, v in ff.items()} for ff in ffl],
        "q1": {"candidates": enc(x1), "value": float(v1)},
        "q1_all": {"candidates": enc(xa), "values": enc(va)},
        "q2": {"candidates": enc(x2), "value": float(v2.reshape(-1)[0])},
    }


def _local_search(logei: Any, qlogei: Any, d: int, g: Any) -> dict:
    import botorch.optim.optimize as bto

    levels = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0], dtype=DT)
    dc = [levels] * d
    bic = levels[torch.randint(5, (4, 1, d), generator=g)]
    x1, v1 = bto.optimize_acqf_discrete_local_search(
        logei, dc, 1, num_restarts=4, batch_initial_conditions=bic
    )
    recorded: list[Any] = []
    orig = bto._gen_batch_initial_conditions_local_search

    def lattice(
        discrete_choices, raw_samples, X_avoid, inequality_constraints, min_points, max_tries=100
    ):
        out = orig(
            discrete_choices, raw_samples, X_avoid, inequality_constraints, min_points, max_tries
        )
        recorded.append(enc(out))
        return out

    bto._gen_batch_initial_conditions_local_search = lattice
    try:
        torch.manual_seed(91)
        x3, v3 = bto.optimize_acqf_discrete_local_search(
            qlogei, dc, 3, num_restarts=4, raw_samples=48
        )
    finally:
        bto._gen_batch_initial_conditions_local_search = orig
    return {
        "levels": enc(levels),
        "batch_initial_conditions": enc(bic),
        "q1": {"candidates": enc(x1), "value": float(v1.reshape(-1)[0])},
        "lattice_points": recorded,
        "q3": {"candidates": enc(x3), "value": float(v3.reshape(-1)[0])},
    }


def _alternating(logei: Any, qlogei: Any, d: int, g: Any) -> dict:
    import botorch.optim.optimize_mixed as btm

    bounds = torch.stack([torch.zeros(d, dtype=DT), torch.ones(d, dtype=DT)])
    discrete_dims = {0: [0.0, 0.25, 0.5, 0.75, 1.0]}
    cat_dims = {d - 1: [0.0, 0.5, 1.0]}
    start = torch.rand(6, d, generator=g, dtype=DT)
    start[:, 0] = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0, 0.5], dtype=DT)
    start[:, d - 1] = torch.tensor([0.0, 0.5, 1.0, 0.0, 0.5, 1.0], dtype=DT)
    orig = btm.generate_starting_points
    out = {
        "discrete_dims": {str(k): v for k, v in discrete_dims.items()},
        "cat_dims": {str(k): v for k, v in cat_dims.items()},
        "starting_points": enc(start),
    }
    try:
        for label, acqf, q in (("q1", logei, 1), ("q2", qlogei, 2)):

            def gsp(opt_inputs, discrete_dims, cat_dims, cont_dims, _acqf=acqf):
                with torch.no_grad():
                    return start.clone(), _acqf(start.unsqueeze(-2))

            btm.generate_starting_points = gsp
            x, v = btm.optimize_acqf_mixed_alternating(
                acqf,
                bounds,
                discrete_dims=discrete_dims,
                cat_dims=cat_dims,
                q=q,
                num_restarts=6,
                raw_samples=32,
            )
            out[label] = {"candidates": enc(x), "value": float(v.reshape(-1)[0])}
    finally:
        btm.generate_starting_points = orig
    return out


def _generation(logei: Any, qlogei: Any, d: int, g: Any) -> dict:
    from botorch.generation.gen import gen_candidates_scipy, gen_candidates_torch
    from botorch.optim.optimize import optimize_acqf, optimize_acqf_cyclic

    bounds = torch.stack([torch.zeros(d, dtype=DT), torch.ones(d, dtype=DT)])
    ics = torch.rand(4, 1, d, generator=g, dtype=DT)
    steps: list[int] = []
    ct, vt = gen_candidates_torch(
        ics, logei, bounds[0], bounds[1], callback=lambda i, loss, grad: steps.append(i)
    )
    custom = {
        "optimizer_options": {"lr": 0.05, "betas": (0.8, 0.99)},
        "stopping_criterion_options": {"maxiter": 60, "rel_tol": 1e-6, "n_window": 5},
    }
    steps2: list[int] = []
    ct2, vt2 = gen_candidates_torch(
        ics,
        logei,
        bounds[0],
        bounds[1],
        options=custom,
        callback=lambda i, loss, grad: steps2.append(i),
    )
    cj, vj = gen_candidates_scipy(ics, logei, bounds[0], bounds[1], options={"maxfun": 400})
    pp_x, pp_v = optimize_acqf(
        logei,
        bounds,
        1,
        4,
        batch_initial_conditions=ics,
        post_processing_func=lambda X: (X * 8).round() / 8,
    )
    cyc_ics = torch.rand(4, 1, d, generator=g, dtype=DT)

    def ic_gen(acq_function, bounds, q, num_restarts, raw_samples, **kw):
        return cyc_ics[:num_restarts].clone()

    cx, cv = optimize_acqf_cyclic(
        qlogei, bounds, 3, 4, raw_samples=16, ic_generator=ic_gen, cyclic_options={"maxiter": 4}
    )
    return {
        "initial_conditions": enc(ics),
        "torch": {"candidates": enc(ct), "values": enc(vt), "steps": len(steps)},
        "torch_custom": {"candidates": enc(ct2), "values": enc(vt2), "steps": len(steps2)},
        "scipy_joint": {"maxfun": 400, "candidates": enc(cj), "values": enc(vj)},
        "post_processing": {"step": 0.125, "candidates": enc(pp_x), "value": float(pp_v)},
        "cyclic": {
            "initial_conditions": enc(cyc_ics),
            "maxiter": 4,
            "candidates": enc(cx),
            "values": enc(cv),
        },
    }


def _utilities(g: Any) -> dict:
    import botorch.utils.sampling as bus
    from botorch.generation.sampling import ConstrainedMaxPosteriorSampling, MaxPosteriorSampling
    from botorch.optim.optimize_homotopy import prune_candidates
    from botorch.optim.stopping import ExpMAStoppingCriterion

    weights = []
    for eta, n in ((1.0, 10), (0.5, 7), (2.0, 3), (1.0, 25)):
        weights.append(
            {
                "eta": eta,
                "n_window": n,
                "weights": enc(ExpMAStoppingCriterion(eta=eta, n_window=n).weights.double()),
            }
        )
    seq = (1.0 + torch.exp(-torch.arange(40, dtype=DT) / 6.0)).tolist()
    decisions = []
    for kw in ({}, {"minimize": False}, {"maxiter": 15}, {"n_window": 4, "rel_tol": 1e-3}):
        crit = ExpMAStoppingCriterion(**kw)
        decisions.append(
            {"kwargs": kw, "stops": [bool(crit(torch.tensor(v, dtype=DT))) for v in seq]}
        )
    cands = torch.rand(12, 2, generator=g, dtype=DT)
    cands[3] = cands[0] + 1e-6
    vals = torch.rand(12, generator=g, dtype=DT)
    pruned = prune_candidates(cands, vals, 1e-3)
    obj = torch.randn(4, 2, 9, 1, generator=g, dtype=DT)
    Xs = torch.rand(2, 9, 2, generator=g, dtype=DT)
    mps = {}
    for repl in (True, False):
        mps[str(repl)] = enc(
            MaxPosteriorSampling(None, replacement=repl).maximize_samples(Xs, obj, 4)
        )
    y = torch.randn(4, 1, 9, 1, generator=g, dtype=DT)
    c = torch.randn(4, 1, 9, 2, generator=g, dtype=DT) + 0.8
    c[1] = c[1].abs() + 0.1  # a realization without feasible points
    scores = ConstrainedMaxPosteriorSampling.__new__(
        ConstrainedMaxPosteriorSampling
    )._convert_samples_to_scores(y, c)
    # Truncated normal perturbations from recorded uniform draws.
    centers = torch.rand(6, 3, generator=g, dtype=DT) * 2 - 0.5
    bounds = torch.tensor([[-0.5, -0.5, -0.5], [1.5, 1.5, 1.5]], dtype=DT)
    u = torch.rand(6, 3, generator=g, dtype=DT)
    orig = bus.draw_sobol_samples
    bus.draw_sobol_samples = lambda bounds, n, q, **kw: u.unsqueeze(1)
    try:
        pert = bus.sample_truncated_normal_perturbations(centers[:1], 6, 0.05, bounds)
    finally:
        bus.draw_sobol_samples = orig
    return {
        "stopping": {"weights": weights, "sequence": seq, "decisions": decisions},
        "prune": {
            "candidates": enc(cands),
            "values": enc(vals),
            "tolerance": 1e-3,
            "pruned": enc(pruned),
        },
        "maximize_samples": {"obj": enc(obj), "X": enc(Xs), "selected": mps},
        "constrained_scores": {"y": enc(y), "c": enc(c), "scores": enc(scores)},
        "truncated_normal": {
            "center": enc(centers[0]),
            "bounds": enc(bounds),
            "u": enc(u),
            "sigma": 0.05,
            "perturbed": enc(pert),
        },
    }


def _fits(seed: int) -> dict:
    from botorch.optim.fit import fit_gpytorch_mll_scipy, fit_gpytorch_mll_torch
    from gpytorch.mlls import ExactMarginalLogLikelihood

    out = {}
    for label, run in (
        ("torch_200", lambda mll: fit_gpytorch_mll_torch(mll, step_limit=200)),
        ("torch_default", lambda mll: fit_gpytorch_mll_torch(mll, step_limit=3000)),
        (
            "torch_lr",
            lambda mll: fit_gpytorch_mll_torch(
                mll, step_limit=150, optimizer=lambda params: torch.optim.Adam(params, lr=0.02)
            ),
        ),
        ("scipy", lambda mll: fit_gpytorch_mll_scipy(mll)),
    ):
        model, spec = _model(2, 14, seed)
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        mll.train()
        result = run(mll)
        out[label] = {
            "model": spec,
            "raw_params": [
                v for _, p in model.named_parameters() for v in p.detach().reshape(-1).tolist()
            ],
            "fval": float(result.fval),
            "step": int(result.step),
            "status": result.status.name,
        }
        if label == "torch_default":
            out[label]["twins"] = _twin_fits(seed, run)
    return out


def _twin_fits(seed: int, run: Any) -> list[dict]:
    """BoTorch's outcomes on 16 twins whose training inputs are perturbed at
    relative scales 1e-14 .. 1e-11 (long Adam runs amplify round-off: the
    number of steps until the stopping criterion holds varies)."""
    from botorch.models import SingleTaskGP
    from gpytorch.mlls import ExactMarginalLogLikelihood

    model, spec = _model(2, 14, seed)
    X = torch.tensor(spec["train_x"], dtype=DT)
    Y = torch.tensor(spec["train_y"], dtype=DT)
    out = []
    for k in range(16):
        rel = 10 ** (-14 + 3 * k / 15)
        g = torch.Generator().manual_seed(k)
        Xk = X * (1 + rel * torch.randn(X.shape, generator=g, dtype=DT))
        twin = SingleTaskGP(Xk, Y)
        twin.load_state_dict(model.state_dict())
        mll = ExactMarginalLogLikelihood(twin.likelihood, twin)
        mll.train()
        r = run(mll)
        out.append({"step": int(r.step), "fval": float(r.fval)})
    return out


def _cross_validation(seed: int) -> dict:
    from botorch.cross_validation import batch_cross_validation, gen_loo_cv_folds
    from botorch.models import SingleTaskGP
    from gpytorch.mlls import ExactMarginalLogLikelihood

    g = torch.Generator().manual_seed(seed)
    x = torch.rand(9, 2, generator=g, dtype=DT)
    y = torch.sin(5 * x.sum(-1, keepdim=True)) + 0.05 * torch.randn(9, 1, generator=g, dtype=DT)
    folds = gen_loo_cv_folds(x, y)
    res = batch_cross_validation(SingleTaskGP, ExactMarginalLogLikelihood, folds)
    return {
        "train_x": enc(x),
        "train_y": enc(y),
        "folds_train_x": enc(folds.train_X),
        "folds_test_y": enc(folds.test_Y),
        "mean": enc(res.posterior.mean),
        "variance": enc(res.posterior.variance),
        "raw_params": [
            v for _, p in res.model.named_parameters() for v in p.detach().reshape(-1).tolist()
        ],
    }


def generate_phase9_optim(out: str) -> None:
    out = os.path.join(out, "phase9_optim")
    os.makedirs(out, exist_ok=True)
    warnings.simplefilter("ignore")
    torch.set_num_threads(1)
    for case in CASES:
        d, n, seed = case["d"], case["n"], case["seed"]
        model, spec = _model(d, n, seed)
        logei, qlogei = _acqfs(model)
        g = torch.Generator().manual_seed(seed + 1)
        fixture = {
            "name": case["name"],
            "d": d,
            "model": spec,
            "best_f": 1.0,
            "discrete": _discrete(logei, qlogei, d, g),
            "mixed": _mixed(logei, qlogei, d, g),
            "local_search": _local_search(logei, qlogei, d, g),
            "alternating": _alternating(logei, qlogei, d, g),
            "generation": _generation(logei, qlogei, d, g),
            "utilities": _utilities(g),
            "fits": _fits(seed + 7),
            "cross_validation": _cross_validation(seed + 11),
        }
        path = os.path.join(out, f"{case['name']}.json")
        with open(path, "w") as f:
            json.dump(fixture, f)
        print(f"phase 9-optim: wrote {case['name']}.json", file=sys.stderr)

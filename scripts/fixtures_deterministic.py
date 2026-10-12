"""Phase 5D fixtures: deterministic models and composite acquisition functions.

Written to ``<out>/phase5d_deterministic/`` by ``generate_fixtures.py --phase
5d-deterministic`` (BoTorch main). The cases follow openbo's multi-objective
transfer acquisition function (MO-TAF): a target qLogNEHVI on a two-output
``SingleTaskGP`` (the phase-4B model format of ``fixtures_mo_acquisition``),
and source terms qLogEHVI(``PosteriorMeanModel(ModelListGP)``) with
``NondominatedPartitioning``, ``StochasticSampler([1])``, ``tau_max=1e-3`` and
``fat=True``, whose sub-models are built like openbo's
``_materialize_source_mean`` (``ScaleKernel`` of a Matern-5/2 or RBF kernel,
``Standardize``, fixed hyperparameters). ``<case>.json`` holds

* the source models (training data, raw hyperparameters in BoTorch's order
  and names), their posterior means at test points;
* values and gradients of every source term (qLogEHVI as above, and qEHVI)
  for q = 1 and q = 2 t-batches;
* values and gradients of the target term and of the blend
  ``logsumexp(log w_i + term_i(X)) - log(sum w_i)`` over the terms with a
  positive weight (openbo's ``taf_mo_ehvi_acquisition``), for several weight
  vectors (including zero weights);
* ``optimize_acqf`` of a blend from given initial conditions
  (``return_best_only=False``): BoTorch's candidates and values per restart.

The target's base samples are BoTorch's recorded draws (one per dimension).
Non-finite numbers are written as the strings "inf", "-inf" and "nan".
"""

from __future__ import annotations

import math
import os
import sys
import warnings
from typing import Any

import torch

DT = torch.float64
NUM_SAMPLES = 16
TAU_MAX = 1e-3

CASES = [
    dict(name="mo_taf_d2_two_sources", d=2, sources=2, seed=8100),
    dict(name="mo_taf_d3_three_sources", d=3, sources=3, seed=8200),
]

WEIGHTS = {
    2: [[1.0, 0.45, 0.2], [0.0, 0.7, 0.3], [0.8, 0.0, 0.6]],
    3: [[1.0, 0.3, 0.2, 0.5], [0.0, 0.0, 1.0, 0.25], [0.25, 0.25, 0.25, 0.25]],
}


def _mo() -> Any:
    """The helpers of ``fixtures_mo_acquisition.py``."""
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import fixtures_mo_acquisition

    return fixtures_mo_acquisition


def openbo_sources(d: int, seed: int, num: int) -> list[dict]:
    """``num`` two-objective sources as openbo's ``_materialize_source_mean``
    builds them, with their specifications and fronts."""
    from botorch.models import ModelListGP, SingleTaskGP
    from botorch.models.transforms.outcome import Standardize
    from botorch.utils.multi_objective.pareto import is_non_dominated
    from gpytorch.kernels import MaternKernel, RBFKernel, ScaleKernel

    enc = _mo().enc
    g = torch.Generator().manual_seed(seed)
    out = []
    for s in range(num):
        n = 10 + 3 * s
        x = torch.rand(n, d, generator=g, dtype=DT)
        centers = torch.rand(2, d, generator=g, dtype=DT)
        y = torch.stack([1 - 2 * ((x - c) ** 2).sum(-1) for c in centers], -1)
        gps, specs = [], []
        for j in range(2):
            kernel = "matern" if (s + j) % 2 == 0 else "rbf"
            base = (
                MaternKernel(nu=2.5, ard_num_dims=d)
                if kernel == "matern"
                else RBFKernel(ard_num_dims=d)
            )
            gp = SingleTaskGP(
                x,
                y[:, j : j + 1],
                covar_module=ScaleKernel(base),
                outcome_transform=Standardize(m=1),
            )
            with torch.no_grad():
                gp.covar_module.base_kernel.lengthscale = 0.2 + torch.rand(d, generator=g, dtype=DT)
                gp.covar_module.outputscale = (0.5 + torch.rand(1, generator=g, dtype=DT)).reshape(
                    ()
                )
                gp.likelihood.noise = 0.01 + 0.05 * torch.rand(1, generator=g, dtype=DT)
                gp.mean_module.constant = (
                    0.4 * torch.rand(1, generator=g, dtype=DT) - 0.2
                ).reshape(())
            gp.eval()
            gps.append(gp)
            named = list(gp.named_parameters())
            specs.append(
                {
                    "kernel": kernel,
                    "train_x": enc(x),
                    "train_y": enc(y[:, j : j + 1]),
                    "param_names": [name for name, _ in named],
                    "raw_params": [v for _, p in named for v in p.detach().reshape(-1).tolist()],
                }
            )
        model = ModelListGP(*gps)
        model.eval()
        out.append({"model": model, "specs": specs, "front": y[is_non_dominated(y)]})
    return out


class Blend(torch.nn.Module):
    """openbo's ``_CompositeTAFEHVI`` (``taf_mo_ehvi_acquisition``)."""

    def __init__(self, terms: list[Any], weights: list[float]) -> None:
        super().__init__()
        self.terms = terms
        self.weights = weights
        self.X_pending = None

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        logs, total = [], 0.0
        for term, w in zip(self.terms, self.weights, strict=True):
            if w <= 0.0:
                continue
            logs.append(math.log(w) + term(X))
            total += w
        return torch.logsumexp(torch.stack(logs, dim=-1), dim=-1) - math.log(total)


def case_fixture(case: dict) -> dict:
    from botorch.acquisition.multi_objective.logei import (
        qLogExpectedHypervolumeImprovement,
        qLogNoisyExpectedHypervolumeImprovement,
    )
    from botorch.acquisition.multi_objective.monte_carlo import qExpectedHypervolumeImprovement
    from botorch.models.deterministic import PosteriorMeanModel
    from botorch.optim import optimize_acqf
    from botorch.sampling.normal import SobolQMCNormalSampler
    from botorch.sampling.stochastic_samplers import StochasticSampler
    from botorch.utils.multi_objective.box_decompositions.non_dominated import (
        NondominatedPartitioning,
    )

    mo = _mo()
    enc = mo.enc
    seed, d = case["seed"], case["d"]
    target_case = dict(kind="single_task", models=[dict(d=d, n=10, m=2, covar="matern")])
    target, target_specs, X = mo.build_case_model(target_case, seed)
    ref = mo.ref_and_front(mo.model_means(target, X))[0] - 0.5
    record: dict[tuple[int, int], torch.Tensor] = {}
    with mo.recorded_draws(record):
        t_term = qLogNoisyExpectedHypervolumeImprovement(
            target,
            ref.tolist(),
            X_baseline=X,
            sampler=SobolQMCNormalSampler(sample_shape=torch.Size([NUM_SAMPLES]), seed=0),
            prune_baseline=False,
        )
    lo, hi = X.min(0).values, X.max(0).values
    g = torch.Generator().manual_seed(seed + 1)
    X1 = lo + (hi - lo) * torch.rand(5, 1, d, generator=g, dtype=DT)
    X2 = lo + (hi - lo) * torch.rand(3, 2, d, generator=g, dtype=DT)
    sources = []
    terms = [t_term]
    for src in openbo_sources(d, seed + 100, case["sources"]):
        front = src["front"][(src["front"] > ref).all(-1)]
        part = NondominatedPartitioning(ref_point=ref, Y=front)
        pm = PosteriorMeanModel(src["model"])
        log_term = qLogExpectedHypervolumeImprovement(
            pm,
            ref.tolist(),
            part,
            sampler=StochasticSampler(sample_shape=torch.Size([1])),
            tau_max=TAU_MAX,
            fat=True,
        )
        lin_term = qExpectedHypervolumeImprovement(
            pm, ref.tolist(), part, sampler=StochasticSampler(sample_shape=torch.Size([1]))
        )
        terms.append(log_term)
        entry = {
            "models": src["specs"],
            "cell_bounds": enc(part.get_hypercell_bounds()),
            "posterior_mean": enc(pm.posterior(X2).mean),
        }
        for name, term in (("qlogehvi", log_term), ("qehvi", lin_term)):
            for q, Xq in ((1, X1), (2, X2)):
                v, gr = mo.values_and_grads(term, Xq)
                entry[f"{name}_q{q}"] = {"values": enc(v), "grads": enc(gr)}
        sources.append(entry)
    with mo.recorded_draws(record):
        v, gr = mo.values_and_grads(t_term, X1)
    composites = []
    for w in WEIGHTS[case["sources"]]:
        blend = Blend(terms, w)
        with mo.recorded_draws(record):
            cv, cg = mo.values_and_grads(blend, X1)
        composites.append({"weights": w, "values": enc(cv), "grads": enc(cg)})
    bounds = torch.stack([lo, hi])
    ics = lo + (hi - lo) * torch.rand(4, 1, d, generator=g, dtype=DT)
    w = WEIGHTS[case["sources"]][0]
    with mo.recorded_draws(record), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cand, vals = optimize_acqf(
            Blend(terms, w),
            bounds,
            q=1,
            num_restarts=4,
            batch_initial_conditions=ics,
            return_best_only=False,
        )
    return {
        "name": case["name"],
        "versions": mo.versions(mo._gf()),
        "d": d,
        "ref_point": enc(ref),
        "tau_max": TAU_MAX,
        "target": {
            "models": target_specs,
            "baseline_x": enc(X),
            "num_samples": NUM_SAMPLES,
            "values": enc(v),
            "grads": enc(gr),
        },
        "sources": sources,
        "x_q1": enc(X1),
        "x_q2": enc(X2),
        "composites": composites,
        "optimize": {
            "weights": w,
            "bounds": enc(bounds),
            "initial_conditions": enc(ics),
            "candidates": enc(cand),
            "values": enc(vals),
        },
        "draws": mo.draws_of(record, NUM_SAMPLES),
    }


def generate_phase5d_deterministic(out: str) -> None:
    """Writes the phase-5D fixtures (BoTorch main)."""
    torch.set_num_threads(1)
    out = os.path.join(out, "phase5d_deterministic")
    os.makedirs(out, exist_ok=True)
    warnings.simplefilter("ignore")
    mo = _mo()
    for case in CASES:
        fixture = case_fixture(case)
        mo._write(os.path.join(out, f"{case['name']}.json"), fixture)
        print(
            f"phase 5d-deterministic: wrote {case['name']}.json "
            f"({len(fixture['sources'])} sources, {len(fixture['composites'])} blends)",
            file=sys.stderr,
        )

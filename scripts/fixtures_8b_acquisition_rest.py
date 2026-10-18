"""Phase 8b fixtures: the multi-step look-ahead acquisition function.

Written to ``<out>/8b-acquisition-rest/`` by ``generate_fixtures.py --phase
8b-acquisition-rest`` (BoTorch main). The models are the phase-8 models
(``fixtures_8_so_acquisition.MODEL_CASES``: single-output RBF, Matern-5/2
with ``Normalize``, fixed noise, no outcome transform, two-output
``SingleTaskGP`` and ``ModelListGP``). Every ``<case>.json`` holds a list
``msl`` of ``qMultiStepLookahead`` configurations with BoTorch's recorded
base samples (the fantasy samplers of every step and the inner samplers of
the MC value functions), candidate sets, BoTorch's values and the gradients
of a corrected BoTorch, and ``optimize`` (for ``m1_rbf_d2``): ``optimize_acqf``
from given initial conditions.

Corrections of BoTorch (see ``docs/drafts/8b-acquisition-rest.md``):

* gradients: BoTorch evaluates the value functions of the fantasy models
  with detached test caches and under ``fast_pred_var``, so its gradients
  are wrong; they are recorded under ``settings.propagate_grads(True)``
  without ``fast_pred_var`` (which agree with central differences);
* ``make_best_f``: BoTorch's best value is the maximum of the
  outcome-transformed training targets, and analytic value functions
  broadcast it against the outputs; the argument factory used here is the
  maximum of the untransformed targets (with a trailing dimension for the
  analytic value functions), the semantics of botorch_rs's
  ``best_f_from_model``;
* ``collapse_fantasy_base_samples=False`` is recorded for multi-output
  posteriors only (for single-output posteriors BoTorch varies the base
  samples over the t-batches instead of the fantasies).

Non-finite numbers are written as the strings "inf", "-inf" and "nan".
"""

from __future__ import annotations

import contextlib
import os
import sys
import warnings
from collections.abc import Iterator
from typing import Any

import torch

DT = torch.float64
TRANSFORM = [0.75, -0.25]
LINEAR = [1.0, -0.5]


def _so() -> Any:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import fixtures_8_so_acquisition

    return fixtures_8_so_acquisition


def _mo() -> Any:
    return _so()._mo()


def enc(x: Any) -> Any:
    return _mo().enc(x)


@contextlib.contextmanager
def corrected_botorch() -> Iterator[None]:
    """BoTorch with gradients through the fantasy models (no detached test
    caches, no ``fast_pred_var``)."""
    import gpytorch
    from botorch import settings

    with settings.propagate_grads(True), gpytorch.settings.fast_pred_var(False):
        yield


def best_f_argfac(analytic: bool) -> Any:
    """``make_best_f`` with the best untransformed target (and a trailing
    dimension for analytic value functions)."""

    def argfac(model: Any, X: Any) -> dict:
        t = model.train_targets
        if hasattr(model, "outcome_transform"):
            t = model.outcome_transform.untransform(t.unsqueeze(-1))[0].squeeze(-1)
        b = t.max(dim=-1).values
        return {"best_f": b.unsqueeze(-1) if analytic else b}

    return argfac


def constant_argfac(**kwargs: Any) -> Any:
    def argfac(model: Any, X: Any) -> dict:
        return dict(kwargs)

    return argfac


def flat_base(sampler: Any) -> Any:
    z = sampler.base_samples
    return None if z is None else enc(z.reshape(z.shape[0], -1))


def msl_configs(m: int, kind: str) -> list[dict]:
    """The configurations of a model with ``m`` outputs (``collapse=False``
    only for multi-output ``SingleTaskGP``s: BoTorch fantasizes the
    sub-models of a ``ModelListGP`` from single-output posteriors, whose base
    samples it varies over the t-batches instead of the fantasies)."""
    from botorch.acquisition import analytic as an
    from botorch.acquisition import monte_carlo as mc

    if m == 1:
        return [
            dict(name="pm_2step", batch_sizes=[1, 1], nf=[3, 2], q=1, t=2),
            dict(name="pm_3step", batch_sizes=[1, 1, 1], nf=[2, 2, 2], q=1, t=2),
            dict(name="q2", batch_sizes=[2, 1], nf=[2, 2], q=2, t=1),
            dict(
                name="qei_best_f",
                batch_sizes=[1, 1],
                nf=[2, 3],
                q=2,
                t=2,
                valfunc=[mc.qExpectedImprovement] * 3,
                argfac=["best_f"] * 3,
                inner=[4, 4, 4],
            ),
            dict(
                name="ei_stage",
                batch_sizes=[1, 1],
                nf=[2, 2],
                q=1,
                t=2,
                valfunc=[None, an.ExpectedImprovement, an.PosteriorMean],
                argfac=[None, "best_f_analytic", None],
            ),
            dict(
                name="ucb_stage",
                batch_sizes=[1],
                nf=[3],
                q=1,
                t=2,
                valfunc=[an.UpperConfidenceBound, an.PosteriorMean],
                argfac=[{"beta": 0.5}, None],
            ),
            dict(name="pending", batch_sizes=[1], nf=[3], q=1, t=2, pending=2),
            dict(name="weights", batch_sizes=[1, 1], nf=[2, 3], q=1, t=2, weights=True),
        ]
    configs = [
        dict(name="pm_transform", batch_sizes=[1, 1], nf=[2, 2], q=1, t=2, transform=True),
        dict(
            name="objective",
            batch_sizes=[1, 1],
            nf=[2, 2],
            q=1,
            t=2,
            objective=True,
            valfunc=[None, None, mc.qSimpleRegret],
            inner=[None, None, 8],
        ),
    ]
    if kind == "single_task":
        configs.append(
            dict(
                name="collapse_false",
                batch_sizes=[1, 1],
                nf=[2, 2],
                q=1,
                t=2,
                transform=True,
                collapse=False,
            )
        )
    return configs


def msl_fixture(case: dict, model: Any, X: torch.Tensor, seed: int) -> list[dict]:
    from botorch.acquisition.multi_step_lookahead import qMultiStepLookahead
    from botorch.acquisition.objective import LinearMCObjective, ScalarizedPosteriorTransform
    from botorch.sampling.normal import SobolQMCNormalSampler

    so = _so()
    m = so.num_outputs(case)
    d = X.shape[-1]
    out = []
    for i, cfg in enumerate(msl_configs(m, case["kind"])):
        torch.manual_seed(seed + i)
        samplers = [SobolQMCNormalSampler(sample_shape=torch.Size([f])) for f in cfg["nf"]]
        weights = None
        if cfg.get("weights"):
            weights = [torch.linspace(1.0, 2.0, f, dtype=DT).reshape(f, 1) for f in cfg["nf"]]
            for s, w in zip(samplers, weights, strict=True):
                s.base_weights = w
        valfunc = cfg.get("valfunc")
        argfacs = None
        if cfg.get("argfac"):
            argfacs = []
            for a in cfg["argfac"]:
                if a is None:
                    argfacs.append(None)
                elif a == "best_f":
                    argfacs.append(best_f_argfac(False))
                elif a == "best_f_analytic":
                    argfacs.append(best_f_argfac(True))
                else:
                    argfacs.append(constant_argfac(**a))
        kw: dict[str, Any] = {
            "batch_sizes": cfg["batch_sizes"],
            "samplers": samplers,
            "valfunc_cls": valfunc,
            "valfunc_argfacs": argfacs,
            "inner_mc_samples": cfg.get("inner"),
            "collapse_fantasy_base_samples": cfg.get("collapse", True),
        }
        if cfg.get("transform"):
            kw["posterior_transform"] = ScalarizedPosteriorTransform(
                torch.tensor(TRANSFORM, dtype=DT)
            )
        if cfg.get("objective"):
            kw["objective"] = LinearMCObjective(torch.tensor(LINEAR, dtype=DT))
        pending = None
        if cfg.get("pending"):
            pending = so.points(X, (cfg["pending"], d), seed + 50 + i)
            kw["X_pending"] = pending
        acqf = qMultiStepLookahead(model, **kw)
        q_aug = acqf.get_augmented_q_batch_size(cfg["q"])
        Xt = so.points(X, (cfg["t"], q_aug, d), seed + 100 + i)
        values = acqf(Xt).detach()
        with corrected_botorch():
            Xg = Xt.clone().requires_grad_(True)
            v2 = acqf(Xg)
            (g,) = torch.autograd.grad(v2.sum(), Xg)
        assert torch.allclose(values, v2.detach(), rtol=1e-12, atol=1e-14)
        stages = []
        for j, vfc in enumerate(valfunc or [None] * len(cfg["nf"]) + ["PosteriorMean"]):
            if vfc is None:
                stages.append(None)
                continue
            name = vfc if isinstance(vfc, str) else vfc.__name__
            a = (cfg.get("argfac") or [None] * (len(cfg["nf"]) + 1))[j]
            inner = acqf.inner_samplers[j]
            stages.append(
                {
                    "name": name,
                    "kwargs": a if isinstance(a, dict) else {},
                    "best_f_from_model": isinstance(a, str),
                    "inner_base_samples": None if inner is None else flat_base(inner),
                }
            )
        out.append(
            {
                "name": cfg["name"],
                "batch_sizes": cfg["batch_sizes"],
                "num_fantasies": cfg["nf"],
                "q": cfg["q"],
                "base_samples": [flat_base(s) for s in samplers],
                "weights": None if weights is None else [enc(w.reshape(-1)) for w in weights],
                "stages": stages,
                "transform": TRANSFORM if cfg.get("transform") else None,
                "objective": LINEAR if cfg.get("objective") else None,
                "pending": None if pending is None else enc(pending),
                "collapse": cfg.get("collapse", True),
                "eval": {"x": enc(Xt), "values": enc(values.reshape(-1)), "grads": enc(g)},
            }
        )
    return out


def optimize_fixture(model: Any, X: torch.Tensor, seed: int) -> dict:
    """``optimize_acqf`` of a two-step look-ahead from given initial
    conditions (the corrected BoTorch)."""
    from botorch.acquisition.multi_step_lookahead import qMultiStepLookahead
    from botorch.optim import optimize_acqf
    from botorch.sampling.normal import SobolQMCNormalSampler

    so = _so()
    d = X.shape[-1]
    torch.manual_seed(seed)
    samplers = [SobolQMCNormalSampler(sample_shape=torch.Size([f])) for f in (2, 2)]
    acqf = qMultiStepLookahead(model, batch_sizes=[1, 1], samplers=samplers)
    q_aug = acqf.get_augmented_q_batch_size(1)
    lo, hi = X.min(0).values, X.max(0).values
    bounds = torch.stack([lo, hi])
    ics = so.points(X, (3, q_aug, d), seed + 1)
    acqf(ics)  # draw the base samples
    with corrected_botorch():
        cands, value = optimize_acqf(
            acqf,
            bounds,
            q=q_aug,
            num_restarts=3,
            batch_initial_conditions=ics,
            return_best_only=False,
            return_full_tree=True,
            options={"maxiter": 200},
        )
    return {
        "batch_sizes": [1, 1],
        "num_fantasies": [2, 2],
        "base_samples": [flat_base(s) for s in samplers],
        "bounds": enc(bounds),
        "initial_conditions": enc(ics),
        "candidates": enc(cands.detach()),
        "values": enc(value.detach().reshape(-1)),
    }


def model_fixture(case: dict, seed: int) -> dict:
    so = _so()
    mo = _mo()
    model, specs, X = mo.build_case_model(case, seed)
    with mo.dense_model_list():
        fixture: dict[str, Any] = {
            "name": case["name"],
            "kind": case["kind"],
            "versions": mo.versions(mo._gf()),
            "models": specs,
            "msl": msl_fixture(case, model, X, seed + 100),
        }
        if case["name"] == "m1_rbf_d2":
            fixture["optimize"] = optimize_fixture(model, X, seed + 200)
    del so
    return fixture


def generate_8b_acquisition_rest(out: str) -> None:
    """Writes the phase-8b fixtures (BoTorch main)."""
    torch.set_num_threads(1)
    out = os.path.join(out, "8b-acquisition-rest")
    os.makedirs(out, exist_ok=True)
    warnings.simplefilter("ignore")
    mo = _mo()
    for i, case in enumerate(_so().MODEL_CASES):
        fixture = model_fixture(case, seed=8900 + 41 * i)
        mo._write(os.path.join(out, f"{case['name']}.json"), fixture)
        print(
            f"phase 8b-acquisition-rest: wrote {case['name']}.json "
            f"({len(fixture['msl'])} look-ahead cases)",
            file=sys.stderr,
        )

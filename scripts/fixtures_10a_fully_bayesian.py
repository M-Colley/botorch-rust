"""Fixtures of the fully Bayesian GPs of phase 10a: BoTorch's
``SaasFullyBayesianSingleTaskGP``, ``FullyBayesianSingleTaskGP`` and
``FullyBayesianLinearSingleTaskGP`` with NumPyro's NUTS
(``fit_fully_bayesian_model_nuts``).

Written to ``<out>/10a_fully_bayesian/`` by ``generate_fixtures.py --phase
10a-fully-bayesian`` (BoTorch main, JAX in double precision: ``jax_x64``).
The cases are those of ``botorch_pyo3/tests/fully_bayesian_helpers.py``
(``fb_cases``). Recorded per case:

* the training data of the NumPyro model, its latent sites (NumPyro's order
  of the unconstrained parameters) and its potential energy with the
  gradient (JAX) at random unconstrained parameters;
* a NUTS run: the initial parameters, every random draw of NumPyro's
  sampler (momenta and the uniform draws of its Bernoulli trials, recorded
  with ``recording_numpyro``), the state and step size after every
  iteration, BoTorch's post-processed samples, and BoTorch's own change of
  the states under perturbations of the training inputs at 1e-14 and 1e-13
  (``twin_spread``, 4 twins) — the scale of the rounding sensitivity of the
  chain;
* the model built from the (thinned) samples: its kernel spec and raw
  parameters, the MLL of every sample and its gradient, and posteriors of
  every sample (as ``fixtures_7b_models.py``), for the cases without input
  warping.

BoTorch's NumPyro model warps its inputs with ``1 - (1 - clamp(x)^c0)^c1``
but its fitted model with ``Warp`` (``1 - (1 - x'^c1)^c0``); the fixtures use
the fixed JAX warping (``fixed_warping=True``), which botorch_rs implements.
"""

from __future__ import annotations

import json
import os
import sys
import warnings
from typing import Any

import torch

DT = torch.float64
TESTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "botorch_pyo3", "tests")
sys.path.insert(0, os.path.normpath(TESTS))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import fixtures_6a_kernels as f6a  # noqa: E402
import fixtures_7b_models as f7b  # noqa: E402
import fully_bayesian_helpers as fbh  # noqa: E402

NUM_TWINS = 4


def _spec(case: dict[str, Any], num: int) -> dict[str, Any]:
    """The kernel spec of the fitted model (batch of ``num`` samples)."""
    k = f6a._helpers().kernel
    batch = {"batch_shape": [num]}
    if case["model"] == "linear":
        return k("LinearKernel", ard_num_dims=case["d"], **batch)
    matern = k("MaternKernel", nu=2.5, ard_num_dims=case["d"], **batch)
    if case["model"] == "matern":
        return matern
    return k("ScaleKernel", {"__kernel__": matern}, **batch)


def _mll(model: Any) -> tuple[list[float], list[float]]:
    """The MLL of every sample and its gradient (raw parameters)."""
    from gpytorch.mlls import ExactMarginalLogLikelihood

    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    model.train(reset=False)
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


def _posteriors(model: Any, case: dict[str, Any]) -> dict[str, Any]:
    """Posteriors of every sample (BoTorch's batched GP without the mixture),
    as ``fixtures_7b_models.posterior_record``. The inputs are recorded in
    the original space (the linear model's ``Normalize(center=0)`` shifts
    them by -0.5 before its kernel)."""
    return f7b.posterior_record(model, {"d": case["d"], "kind": "fb"}, batched=True)


def _case(case: dict[str, Any], seed: int) -> dict[str, Any]:
    X, Y, Yvar = fbh.make_fb_data(case, seed)
    bt = fbh.bt_model(case, X, Y, Yvar)
    pm = bt.pyro_model
    fit_kwargs = {
        "warmup_steps": case["warmup"],
        "num_samples": case["samples"],
        "thinning": 2,
    }
    rec = fbh.bt_fit_recorded(bt, fixed_warping=True, **fit_kwargs)
    dim = len(rec.z0 or [])
    g = torch.Generator().manual_seed(seed + 1)
    points = [(4 * torch.rand(dim, generator=g, dtype=DT) - 2).tolist() for _ in range(3)]
    points.append(rec.states[-1][1])
    potentials = []
    for z in points:
        v, gr = fbh.numpyro_potential(bt, z)
        potentials.append({"z": z, "value": v, "grad": gr})
    spread, step_spread, agree = fbh.twin_spread(case, X, Y, Yvar, rec, NUM_TWINS, fit_kwargs)
    num = next(iter(rec.mcmc_samples.values())).shape[0]
    out: dict[str, Any] = {
        "name": case["name"],
        "model": case["model"],
        "d": case["d"],
        "train_x": pm.train_X,
        "train_y": pm.train_Y.reshape(-1),
        "train_yvar": None if pm.train_Yvar is None else pm.train_Yvar.reshape(-1),
        "outputscale_prior": (
            None
            if getattr(pm, "_outputscale_prior_concentration", None) is None
            else [pm._outputscale_prior_concentration, pm._outputscale_prior_rate]
        ),
        "warp_indices": list(range(case["d"])) if case.get("warp") else None,
        "warp_eps": pm._eps,
        "potentials": potentials,
        "nuts": {
            "num_warmup": case["warmup"],
            "num_samples": case["samples"],
            "max_tree_depth": 6,
            "z0": rec.z0,
            "draws": [list(dr) for dr in rec.draws],
            "states": [s[1] for s in rec.states],
            "step_sizes": [s[2] for s in rec.states],
            "twin_spread": spread,
            "twin_step_spread": step_spread,
            "twins_agree": agree,
            "postprocessed": {k: v for k, v in (rec.postprocessed or {}).items()},
        },
        "num_mcmc_samples": num,
    }
    if not case.get("warp"):
        bt.eval()
        names = [(n, list(p.shape)) for n, p in bt.named_parameters()]
        raw = f6a.flat(bt)
        value, grad = _mll(bt)
        post = _posteriors(bt, case)
        out["model_level"] = {
            "spec": _spec(case, num),
            "param_names": names,
            "raw": raw,
            "mll": {"value": value, "grad": grad},
            "posterior": post,
            "bald": _bald(bt, case, seed),
        }
    return out


def _bald(model: Any, case: dict[str, Any], seed: int) -> list[dict[str, Any]]:
    """BALD values and input gradients at random candidate sets (q = 1, 2,
    3) with BoTorch's base samples (shared by the samples, ``num x q``)."""
    from botorch.acquisition.bayesian_active_learning import (
        qBayesianActiveLearningByDisagreement,
    )
    from botorch.sampling.normal import SobolQMCNormalSampler

    g = torch.Generator().manual_seed(seed + 2)
    out = []
    for q in (1, 2, 3):
        X = torch.rand(2, q, case["d"], generator=g, dtype=DT)
        sampler = SobolQMCNormalSampler(sample_shape=torch.Size([32]), seed=seed + q)
        with torch.no_grad():
            sampler(model.posterior(X))
        acqf = qBayesianActiveLearningByDisagreement(model, sampler=sampler)
        Xg = X.clone().requires_grad_(True)
        values = acqf(Xg)
        (grad,) = torch.autograd.grad(values.sum(), Xg)
        out.append(
            {
                "x": X,
                "base_samples": sampler.base_samples.reshape(32, q),
                "values": values.detach(),
                "grad": grad,
            }
        )
    return out


def _mt_case(name: str, seed: int, fixed: bool) -> dict[str, Any]:
    """A multi-task SAAS model: potentials and a NUTS run with its twins
    (the chains of these models are chaotic from their first iterations,
    see ``test_parity_10a_fully_bayesian.py``)."""
    from botorch.models.fully_bayesian_multitask import SaasFullyBayesianMultiTaskGP

    X, Y, Yvar = fbh.mt_data(seed, n=14, tasks=(0, 1))
    Y = (Y - Y.mean()) / Y.std()
    yvar = Yvar if fixed else None

    def make(Xi: Any) -> Any:
        with fbh.jax_x64(True):
            return SaasFullyBayesianMultiTaskGP(Xi, Y, task_feature=-1, train_Yvar=yvar, rank=2)

    bt = make(X)
    pm = bt.pyro_model
    fit_kwargs = {"warmup_steps": 20, "num_samples": 8, "thinning": 2}
    rec = fbh.bt_fit_recorded(bt, **fit_kwargs)
    dim = len(rec.z0 or [])
    g = torch.Generator().manual_seed(seed + 1)
    potentials = []
    for _ in range(3):
        z = (4 * torch.rand(dim, generator=g, dtype=DT) - 2).tolist()
        v, gr = fbh.numpyro_potential(bt, z)
        potentials.append({"z": z, "value": v, "grad": gr})
    spread, step_spread, agree = fbh.twin_spread(
        {}, X, Y, Yvar, rec, NUM_TWINS, fit_kwargs, make_model=make
    )
    return {
        "name": name,
        "model": "saas",
        "d": X.shape[-1] - 1,
        "train_x": pm.train_X,
        "train_y": pm.train_Y.reshape(-1),
        "train_yvar": None if pm.train_Yvar is None else pm.train_Yvar.reshape(-1),
        "outputscale_prior": [pm._outputscale_prior_concentration, pm._outputscale_prior_rate],
        "warp_indices": None,
        "warp_eps": pm._eps,
        "task": [pm.task_feature % pm.train_X.shape[-1], pm.num_tasks, pm.task_rank],
        "potentials": potentials,
        "nuts": {
            "num_warmup": 20,
            "num_samples": 8,
            "max_tree_depth": 6,
            "z0": rec.z0,
            "draws": [list(dr) for dr in rec.draws],
            "states": [s[1] for s in rec.states],
            "step_sizes": [s[2] for s in rec.states],
            "twin_spread": spread,
            "twin_step_spread": step_spread,
            "twins_agree": agree,
            "postprocessed": {k: v for k, v in (rec.postprocessed or {}).items()},
        },
        "num_mcmc_samples": 4,
    }


def generate_10a_fully_bayesian(out: str) -> None:
    import botorch

    print(f"phase 10a-fully-bayesian: BoTorch {botorch.__file__}", file=sys.stderr)
    out = os.path.join(out, "10a_fully_bayesian")
    os.makedirs(out, exist_ok=True)
    warnings.simplefilter("ignore")
    torch.set_num_threads(1)
    cases: list[tuple[str, Any]] = [
        (c["name"], lambda c=c, i=i: _case(c, seed=10_100 + 31 * i))
        for i, c in enumerate(fbh.fb_cases())
    ]
    cases.append(("mt_saas", lambda: _mt_case("mt_saas", 10_700, False)))
    cases.append(("mt_saas_fixed", lambda: _mt_case("mt_saas_fixed", 10_731, True)))
    for name, make in cases:
        fixture = make()
        path = os.path.join(out, f"{name}.json")
        with open(path, "w") as f:
            json.dump(f6a.enc(fixture), f)
        nuts = fixture["nuts"]
        print(
            f"phase 10a-fully-bayesian: {name:20s} draws={len(nuts['draws'])} "
            f"iterations={len(nuts['states'])} twin spread (last)="
            f"{nuts['twin_spread'][-1]:.1e} draws the twins share="
            f"{nuts['twins_agree']}",
            file=sys.stderr,
        )

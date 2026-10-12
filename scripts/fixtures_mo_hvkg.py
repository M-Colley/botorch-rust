"""Phase 5C-HVKG fixtures: fantasy models, conditioning on observations, cost
models and utilities, and BoTorch's hypervolume knowledge gradient.

Written to ``<out>/phase5c_hvkg/`` by ``generate_fixtures.py --phase 5c-hvkg``
(with BoTorch main) and ``--phase 5c-hvkg-prs`` (with a BoTorch that includes
meta-pytorch/botorch#3396 or #3399):

* ``conditioning.json``: ``condition_on_observations`` of single- and
  multi-output ``SingleTaskGP``s (inferred and fixed noise, ``Normalize``,
  ``Standardize``, batched observations), ``fantasize`` (explicit base
  samples, batched inputs, default and explicit observation noise) and
  ``ModelListGP.fantasize`` (shared and per-output samplers, evaluation
  masks), and ``ModelListGP.condition_on_observations`` of sub-models
  without transforms: joint posteriors of the resulting (batched) models,
  with and without observation noise.
* ``conditioning_3396.json`` / ``conditioning_3399.json`` (``--phase
  5c-hvkg-prs``): the cases that BoTorch main computes incorrectly, generated
  with the corresponding pull request: explicit noise of new observations of
  homoskedastic models (#3396, the posterior mean uses it) and
  ``ModelListGP.condition_on_observations`` with input and outcome
  transforms and noise in the original space (#3399; the explicit noise of
  its homoskedastic sub-models needs #3396 too, which is applied to the
  conditioned models if the BoTorch at hand lacks it).
* ``hvkg.json``: ``qHypervolumeKnowledgeGradient`` and
  ``qMultiFidelityHypervolumeKnowledgeGradient`` on lists of single-output
  models: values and gradients of single t-batches (BoTorch's semantics for
  one t-batch; see the Rust docs for larger batches) for posterior means and
  samples, ``log``, ``current_value``, pending points, evaluation masks,
  objectives, cost-aware utilities (fixed and affine cost models, log
  utilities) and projections onto target fidelities, with the base samples
  of the fantasy and inner samplers. With posterior samples, the reference
  gradients are computed with ``detach_test_caches(False)`` (BoTorch's
  default detaches the fantasy models' covariance caches and misses part of
  the gradient with respect to the candidates; its gradient is recorded as
  ``grad_detached``). Also ``_get_hv_value_function`` and
  ``_split_hvkg_fantasy_points``.
* ``optimize.json``: ``optimize_acqf`` of qHVKG from explicit initial
  conditions (one restart per problem, ``batch_limit=1``), and
  ``gen_one_shot_hvkg_initial_conditions`` with botorch_rs's Sobol raw
  samples (``draw_sobol_samples`` patched; requires the botorch_rs
  extension): all inner raw samples are optimized (no random selection of
  the inner starts), the inner optimization results are recorded, and a
  huge Boltzmann temperature (``eta``) makes the selection of the
  candidates deterministic given the fantasy designs. The fantasy designs
  are still drawn at random among inner optima of (nearly) equal value.
* ``cost.json``: ``AffineFidelityCostModel``, ``FixedCostModel``,
  ``InverseCostWeightedUtility`` (with evaluation masks, log),
  ``project_to_target_fidelity`` and ``expand_trace_observations``.

Non-finite numbers are written as the strings "inf", "-inf" and "nan".
"""

from __future__ import annotations

import contextlib
import functools
import inspect
import math
import os
import sys
import warnings
from collections.abc import Iterator
from typing import Any

import torch

DT = torch.float64


def _mo() -> Any:
    """The helpers of the phase-5B generator."""
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import fixtures_mo_acquisition

    return fixtures_mo_acquisition


def enc(x: Any) -> Any:
    return _mo().enc(x)


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

# Lists of single-output models (the phase-4 builders with explicit
# hyperparameters).
LIST_CASES = [
    dict(
        name="list3_d2",
        models=[
            dict(d=2, n=9, m=1, normalize="shared"),
            dict(d=2, n=7, m=1, seed_offset=1),
            dict(d=2, n=8, m=1, yvar=True, normalize="shared", seed_offset=2),
        ],
    ),
    dict(
        name="list2_d3",
        models=[
            dict(d=3, n=10, m=1, covar="matern", normalize="shared", x_range=(-1.0, 2.0)),
            dict(d=3, n=9, m=1, outcome="none", seed_offset=3, x_range=(-1.0, 2.0)),
        ],
    ),
]

# Single models for the conditioning cases.
SINGLE_CASES = [
    dict(name="m1_normalize", models=[dict(d=2, n=8, m=1, normalize="shared")]),
    dict(name="m1_fixed_noise", models=[dict(d=2, n=7, m=1, yvar=True, normalize="shared")]),
    dict(name="m2_matern", models=[dict(d=3, n=9, m=2, covar="matern", normalize="shared")]),
    dict(name="m2_plain", models=[dict(d=2, n=8, m=2, outcome="none")]),
]


def build(case: dict, seed: int, kind: str) -> tuple[Any, list[dict], torch.Tensor]:
    return _mo().build_case_model(dict(case, kind=kind), seed)


def points(X: torch.Tensor, shape: tuple[int, ...], seed: int) -> torch.Tensor:
    return _mo().points(X, shape, seed)


def joint(model: Any, X: torch.Tensor, noise: bool) -> dict:
    """The joint posterior (mean ``batch x q x m``, covariance ``batch x qm x
    qm`` output-major) of ``X``."""
    with torch.no_grad():
        post = model.posterior(X, observation_noise=noise)
        return {
            "mean": enc(post.mean.reshape(-1)),
            "mean_shape": list(post.mean.shape),
            "covariance": enc(post.distribution.covariance_matrix.reshape(-1)),
        }


def has_fantasy_mean_cache_fix() -> bool:
    """Whether BoTorch includes meta-pytorch/botorch#3396."""
    import botorch.models.gpytorch as bg

    return hasattr(bg, "_use_fantasy_mean_cache")


def use_fantasy_mean_cache(model: Any) -> None:
    """meta-pytorch/botorch#3396 applied to a conditioned model (for BoTorch
    versions without it): the posterior mean uses the updated mean cache of
    GPyTorch's fantasy strategy, i.e. the noise of the new observations."""
    from gpytorch.utils.errors import CachingError
    from gpytorch.utils.memoize import add_to_cache, get_from_cache

    for gp in getattr(model, "models", [model]):
        strategy = gp.prediction_strategy
        try:
            get_from_cache(strategy, "mean_cache", "ignore")
            continue
        except CachingError:
            pass
        mean_cache = get_from_cache(strategy, "mean_cache")
        add_to_cache(strategy, "mean_cache", mean_cache.detach(), "ignore")


def has_model_list_conditioning_fix() -> bool:
    """Whether BoTorch includes meta-pytorch/botorch#3399."""
    from botorch.models import model_list_gp_regression as ml

    return "conditioned_models" in inspect.getsource(ml.ModelListGP.condition_on_observations)


# ---------------------------------------------------------------------------
# Conditioning and fantasies
# ---------------------------------------------------------------------------


def sampler_draws(sampler: Any, n_f: int) -> list[float]:
    """The base samples a sampler used (``n_f x 1 x n x m``), row-major as
    ``n_f x (n m)``."""
    return enc(sampler.base_samples.reshape(n_f, -1))


def single_conditioning(case: dict, seed: int, fixes: set[str]) -> dict:
    """Conditioning and fantasy cases of a single (possibly multi-output)
    ``SingleTaskGP``."""
    from botorch.sampling.normal import SobolQMCNormalSampler

    model, specs, X = build(case, seed, "single_task")
    m, d = model.num_outputs, X.shape[-1]
    fixed = model.likelihood.__class__.__name__.startswith("FixedNoise")
    X_test = points(X, (4, d), seed + 1)
    out: dict = {"name": case["name"], "models": specs, "x_test": enc(X_test), "cases": []}
    g = torch.Generator().manual_seed(seed + 2)

    def obs(shape: tuple[int, ...]) -> torch.Tensor:
        return 2.0 + torch.randn(*shape, m, generator=g, dtype=DT)

    def record(name: str, fm: Any, **extra: Any) -> None:
        out["cases"].append(
            dict(
                name=name,
                posterior=joint(fm, X_test, False),
                posterior_noise=joint(fm, X_test, True),
                **extra,
            )
        )

    if "main" in fixes:
        model.posterior(X_test)  # GPyTorch needs the prediction caches
        # Conditioning on observations (unbatched and batched).
        for name, x_shape, y_shape in [
            ("condition", (3,), (3,)),
            ("condition_batched_y", (3,), (2, 3)),
            ("condition_batched_xy", (2, 3), (2, 3)),
        ]:
            if fixed and len(y_shape) > len(x_shape):
                # GPyTorch fails to concatenate batched noise of unbatched
                # inputs ("All tensors must have the same number of
                # dimensions").
                continue
            Xn = points(X, (*x_shape, d), seed + 3 + len(out["cases"]))
            Yn = obs(y_shape)
            noise = None
            kwargs = {}
            if fixed:
                noise = (0.01 + 0.05 * torch.rand(*y_shape, m, generator=g, dtype=DT)) / 4
                kwargs["noise"] = noise
            cm = model.condition_on_observations(X=Xn, Y=Yn, **kwargs)
            record(
                name,
                cm,
                x=enc(Xn),
                x_shape=list(Xn.shape),
                y=enc(Yn),
                y_shape=list(Yn.shape),
                noise=None if noise is None else enc(noise),
            )
        # Fantasies with explicit base samples (the sampler's draws).
        for name, x_shape, n_f, explicit in [
            ("fantasize", (2,), 3, False),
            ("fantasize_batched", (2, 2), 4, False),
            ("fantasize_one_point", (1,), 2, False),
        ] + ([("fantasize_explicit_noise", (2,), 3, True)] if fixed else []):
            Xn = points(X, (*x_shape, d), seed + 11 + len(out["cases"]))
            sampler = SobolQMCNormalSampler(sample_shape=torch.Size([n_f]), seed=seed + 13)
            noise = None
            if explicit:
                noise = torch.full((1, m), 0.02, dtype=DT)
            fm = model.fantasize(Xn, sampler, observation_noise=noise)
            record(
                name,
                fm,
                x=enc(Xn),
                x_shape=list(Xn.shape),
                num_fantasies=n_f,
                base_samples=sampler_draws(sampler, n_f),
                observation_noise=None if noise is None else enc(noise),
            )
    if "3396" in fixes and not fixed:
        # Explicit noise of new observations of a homoskedastic model.
        model.posterior(X_test)
        Xn = points(X, (3, d), seed + 21)
        Yn = obs((3,))
        noise = torch.tensor([1e-6, 0.3, 0.05], dtype=DT).unsqueeze(-1).expand(3, m)
        cm = model.condition_on_observations(X=Xn, Y=Yn, noise=noise)
        record(
            "condition_explicit_noise",
            cm,
            x=enc(Xn),
            x_shape=list(Xn.shape),
            y=enc(Yn),
            y_shape=list(Yn.shape),
            noise=enc(noise),
        )
        sampler = SobolQMCNormalSampler(sample_shape=torch.Size([3]), seed=seed + 23)
        Xn = points(X, (2, d), seed + 22)
        noise = torch.full((1, m), 0.07, dtype=DT)
        fm = model.fantasize(Xn, sampler, observation_noise=noise)
        record(
            "fantasize_explicit_noise",
            fm,
            x=enc(Xn),
            x_shape=list(Xn.shape),
            num_fantasies=3,
            base_samples=sampler_draws(sampler, 3),
            observation_noise=enc(noise),
        )
    return out


def list_conditioning(case: dict, seed: int, fixes: set[str]) -> dict:
    """``ModelListGP.fantasize`` and ``condition_on_observations``."""
    from botorch.models import ModelListGP
    from botorch.sampling.list_sampler import ListSampler
    from botorch.sampling.normal import SobolQMCNormalSampler

    model, specs, X = build(case, seed, "model_list")
    m, d = model.num_outputs, X.shape[-1]
    X_test = points(X, (4, d), seed + 1)
    out: dict = {"name": case["name"], "models": specs, "x_test": enc(X_test), "cases": []}

    def record(name: str, fm: Any, **extra: Any) -> None:
        out["cases"].append(
            dict(
                name=name,
                posterior=joint(fm, X_test, False),
                posterior_noise=joint(fm, X_test, True),
                **extra,
            )
        )

    if "main" in fixes:
        n_f = 3
        Xn = points(X, (3, d), seed + 2)
        masks = [None, [[True] * m, [True] + [False] * (m - 1), [False] * (m - 1) + [True]]]
        for mask in masks:
            samplers = [
                SobolQMCNormalSampler(sample_shape=torch.Size([n_f]), seed=seed + 5 + k)
                for k in range(m)
            ]
            em = None if mask is None else torch.tensor(mask)
            fm = model.fantasize(Xn, ListSampler(*samplers), evaluation_mask=em)
            record(
                "fantasize_list" if mask is None else "fantasize_mask",
                fm,
                x=enc(Xn),
                x_shape=list(Xn.shape),
                num_fantasies=n_f,
                base_samples=[
                    None if s.base_samples is None else sampler_draws(s, n_f) for s in samplers
                ],
                evaluation_mask=mask,
            )
        shared = SobolQMCNormalSampler(sample_shape=torch.Size([n_f]), seed=seed + 9)
        fm = model.fantasize(Xn, shared)
        record(
            "fantasize_shared",
            fm,
            x=enc(Xn),
            x_shape=list(Xn.shape),
            num_fantasies=n_f,
            base_samples=[sampler_draws(shared, n_f)],
            evaluation_mask=None,
        )
    plain = all(
        s["config"]["normalize"] is None and s["config"]["outcome"] == "none" for s in specs
    )
    if ("main" in fixes and plain) or ("3399" in fixes and not plain):
        model.posterior(X_test)
        g = torch.Generator().manual_seed(seed + 3)
        Xs = [points(X, (2, d), seed + 30 + k) for k in range(m)]
        Yn = 2.0 + torch.randn(2, m, generator=g, dtype=DT)
        noise = 0.01 + 0.05 * torch.rand(2, m, generator=g, dtype=DT)
        cm = ModelListGP.condition_on_observations(model, X=Xs, Y=Yn, noise=noise)
        if not has_fantasy_mean_cache_fix():
            # The explicit noise of homoskedastic sub-models (#3396).
            use_fantasy_mean_cache(cm)
        record(
            "condition_list",
            cm,
            xs=[enc(x) for x in Xs],
            y=enc(Yn),
            noise=enc(noise),
        )
    return out


# ---------------------------------------------------------------------------
# The hypervolume knowledge gradient
# ---------------------------------------------------------------------------

# (name, n_f, num_pareto, q, use_posterior_mean, log, current value, pending
#  points (with mask), candidates' evaluation mask, objective, cost utility,
#  multi-fidelity, inner samples)
HVKG_CASES = [
    dict(name="default", n_f=4, np=3, q=1),
    dict(name="samples", n_f=3, np=2, q=2, upm=False, inner=8),
    dict(name="log", n_f=3, np=2, q=1, log=True, cv="half"),
    dict(name="current_value", n_f=3, np=3, q=2, cv="half"),
    dict(
        name="decoupled_pending",
        n_f=3,
        np=2,
        q=2,
        upm=False,
        inner=8,
        pending=2,
        pending_mask="alternate",
        mask="alternate",
    ),
    dict(name="unobserved_output", n_f=2, np=2, q=2, mask="first_only"),
    dict(name="pending_all_outputs", n_f=2, np=2, q=1, pending=1),
    dict(name="objective_outcomes", n_f=3, np=2, q=1, objective="outcomes"),
    dict(name="objective_weighted", n_f=2, np=2, q=1, objective="weighted", upm=False, inner=8),
    dict(name="objective_generic", n_f=2, np=2, q=1, objective="generic"),
    dict(name="cost_fixed", n_f=3, np=2, q=1, cv="half", cost="fixed", mask="same_rows"),
    dict(name="cost_log", n_f=2, np=2, q=2, cv="half", log=True, cost="fixed_log"),
    dict(name="multi_fidelity", n_f=3, np=2, q=1, cv="half", cost="affine", mf=True),
    dict(name="multi_fidelity_samples", n_f=2, np=2, q=1, mf=True, upm=False, inner=8),
    dict(name="q0", n_f=3, np=2, q=0),
]


def mask_rows(kind: str | None, rows: int, m: int) -> list[list[bool]] | None:
    if kind is None:
        return None
    if kind == "alternate":
        return [[(r + k) % 2 == 0 for k in range(m)] for r in range(rows)]
    if kind == "first_only":
        return [[k == 0 for k in range(m)] for _ in range(rows)]
    if kind == "same_rows":
        return [[k != 1 for k in range(m)] for _ in range(rows)]
    raise ValueError(kind)


def ref_point_of(model: Any, spec: dict | None, X: torch.Tensor, seed: int) -> torch.Tensor:
    """A reference point below the objective values of the posterior mean at
    random points."""
    mo = _mo()
    Xr = points(X, (64, X.shape[-1]), seed)
    means = mo.model_means(model, Xr)
    Yobj = mo.objective_values(spec, model, Xr, means)
    lo, hi = Yobj.min(0).values, Yobj.max(0).values
    return lo - 0.05 * (hi - lo)


def cost_spec(kind: str | None, m: int, d: int) -> dict | None:
    if kind is None:
        return None
    if kind in ("fixed", "fixed_log"):
        return {
            "type": "fixed",
            "fixed_cost": [1.0 + 0.5 * k for k in range(m)],
            "log": kind == "fixed_log",
        }
    if kind == "affine":
        return {"type": "affine", "weights": [[d - 1, 1.5]], "fixed_cost": 0.25, "log": False}
    raise ValueError(kind)


def bt_cost(spec: dict | None) -> Any:
    from botorch.acquisition.cost_aware import InverseCostWeightedUtility
    from botorch.models.cost import AffineFidelityCostModel, FixedCostModel

    if spec is None:
        return None
    if spec["type"] == "fixed":
        cm: Any = FixedCostModel(torch.tensor(spec["fixed_cost"], dtype=DT))
    else:
        cm = AffineFidelityCostModel(
            fidelity_weights={int(k): float(w) for k, w in spec["weights"]},
            fixed_cost=spec["fixed_cost"],
        )
        cm.weights = cm.weights.to(DT)
    return InverseCostWeightedUtility(cost_model=cm, log=spec["log"])


def hvkg_case(model: Any, X: torch.Tensor, c: dict, seed: int) -> dict:
    """Values and gradients of one qHVKG configuration at two candidate sets
    (each evaluated as its own t-batch)."""
    import gpytorch
    from botorch.acquisition.multi_objective.hypervolume_knowledge_gradient import (
        qHypervolumeKnowledgeGradient,
        qMultiFidelityHypervolumeKnowledgeGradient,
    )
    from botorch.acquisition.utils import project_to_target_fidelity
    from botorch.sampling.list_sampler import ListSampler
    from botorch.sampling.normal import SobolQMCNormalSampler

    mo = _mo()
    m, d = model.num_outputs, X.shape[-1]
    n_f, np_, q = c["n_f"], c["np"], c["q"]
    upm = c.get("upm", True)
    log = c.get("log", False)
    spec, _ = mo.objective_spec(c.get("objective", "identity"), m)
    ref = ref_point_of(model, spec, X, seed)
    objective = mo.bt_objective(spec, model, X)
    g = torch.Generator().manual_seed(seed)
    pending = None
    if c.get("pending"):
        pending = points(X, (c["pending"], d), seed + 1)
    pmask = mask_rows(c.get("pending_mask"), c.get("pending", 0), m)
    xmask = mask_rows(c.get("mask"), q, m)
    cost = cost_spec(c.get("cost"), m, d)
    mf = c.get("mf", False)
    target = {d - 1: 1.0}
    samplers = [
        SobolQMCNormalSampler(sample_shape=torch.Size([n_f]), seed=seed + 7 + k) for k in range(m)
    ]
    inner = SobolQMCNormalSampler(sample_shape=torch.Size([c.get("inner", 4)]), seed=seed + 3)
    Xs = [points(X, (1, q + n_f * np_, d), seed + 11 + k) for k in range(2)]
    if mf:
        for x in Xs:
            # Fidelities spread over [0.2, 1].
            x[..., -1] = 0.2 + 0.8 * torch.rand(x.shape[:-1], generator=g, dtype=DT)
    kwargs: dict = dict(
        model=model,
        ref_point=ref,
        num_fantasies=n_f,
        num_pareto=np_,
        sampler=ListSampler(*samplers),
        objective=objective,
        inner_sampler=inner,
        X_pending=pending,
        X_evaluation_mask=None if xmask is None else torch.tensor(xmask),
        X_pending_evaluation_mask=None if pmask is None else torch.tensor(pmask),
        use_posterior_mean=upm,
        log=log,
    )
    if mf:
        kwargs["target_fidelities"] = target
        kwargs["project"] = functools.partial(
            project_to_target_fidelity, target_fidelities=target, d=d
        )
        cls: Any = qMultiFidelityHypervolumeKnowledgeGradient
    else:
        cls = qHypervolumeKnowledgeGradient
    current_value = None
    if c.get("cv"):
        probe = cls(**kwargs)
        with torch.no_grad():
            v0, v1 = (probe(x).item() for x in Xs)
        # Linear: half of the first value (some improvements are negative);
        # log: far below both values (every fantasy must exceed it).
        current_value = min(v0, v1) - math.log(1000.0) if log else 0.5 * v0
        kwargs["current_value"] = torch.tensor(current_value, dtype=DT)
        kwargs["cost_aware_utility"] = bt_cost(cost)
    acqf = cls(**kwargs)
    values, grads, grads_detached = [], [], []
    for x in Xs:
        if upm:
            v, gr = mo.values_and_grads(acqf, x)
            gd = gr
        else:
            with gpytorch.settings.detach_test_caches(False):
                v, gr = mo.values_and_grads(acqf, x)
            _, gd = mo.values_and_grads(acqf, x)
        values.append(enc(v.item()))
        grads.append(enc(gr.reshape(-1)))
        grads_detached.append(enc(gd.reshape(-1)))
    return dict(
        name=c["name"],
        num_fantasies=n_f,
        num_pareto=np_,
        q=q,
        use_posterior_mean=upm,
        log=log,
        ref_point=enc(ref),
        objective=spec,
        x_pending=None if pending is None else enc(pending.reshape(-1)),
        x_pending_evaluation_mask=pmask,
        x_evaluation_mask=xmask,
        current_value=current_value,
        cost=cost,
        target_fidelities=[[d - 1, 1.0]] if mf else None,
        x=[enc(x.reshape(-1)) for x in Xs],
        values=values,
        grads=grads,
        grads_detached=grads_detached,
        base_samples=[None if s.base_samples is None else sampler_draws(s, n_f) for s in samplers],
        inner_base_samples=None
        if inner.base_samples is None
        else enc(inner.base_samples.reshape(inner.sample_shape[0], -1)),
        num_inner_samples=inner.sample_shape[0],
    )


def value_function_cases(model: Any, X: torch.Tensor, seed: int) -> list[dict]:
    from botorch.acquisition.multi_objective.hypervolume_knowledge_gradient import (
        _get_hv_value_function,
    )
    from botorch.acquisition.utils import project_to_target_fidelity
    from botorch.sampling.normal import SobolQMCNormalSampler

    mo = _mo()
    m, d = model.num_outputs, X.shape[-1]
    out = []
    for k, (upm, log, proj, obj) in enumerate(
        [
            (True, False, False, "identity"),
            (False, False, False, "identity"),
            (True, True, False, "weighted"),
            (False, True, True, "identity"),
        ]
    ):
        spec, _ = mo.objective_spec(obj, m)
        ref = ref_point_of(model, spec, X, seed + k)
        sampler = SobolQMCNormalSampler(sample_shape=torch.Size([8]), seed=seed + 20 + k)
        target = {d - 1: 1.0}
        vf = _get_hv_value_function(
            model=model,
            ref_point=ref,
            objective=mo.bt_objective(spec, model, X),
            sampler=sampler,
            project=functools.partial(project_to_target_fidelity, target_fidelities=target, d=d)
            if proj
            else None,
            use_posterior_mean=upm,
            log=log,
        )
        x = points(X, (1, 3, d), seed + 30 + k)
        v, gr = mo.values_and_grads(vf, x)
        base = vf.base_value_function if proj else vf
        used = None
        if not upm and getattr(base.sampler, "base_samples", None) is not None:
            used = enc(base.sampler.base_samples.reshape(8, -1))
        out.append(
            dict(
                use_posterior_mean=upm,
                log=log,
                project=proj,
                objective=spec,
                ref_point=enc(ref),
                x=enc(x.reshape(-1)),
                value=v.item(),
                grad=enc(gr.reshape(-1)),
                base_samples=used,
            )
        )
    return out


def hvkg_fixture(case: dict, seed: int) -> dict:
    model, specs, X = build(case, seed, "model_list")
    cases = []
    for i, c in enumerate(HVKG_CASES):
        if c.get("objective") == "generic" and model.num_outputs < 2:
            continue
        cases.append(hvkg_case(model, X, c, seed + 100 * (i + 1)))
    return {
        "name": case["name"],
        "models": specs,
        "cases": cases,
        "value_functions": value_function_cases(model, X, seed + 50),
    }


def split_cases() -> list[dict]:
    from botorch.acquisition.multi_objective.hypervolume_knowledge_gradient import (
        _split_hvkg_fantasy_points,
    )

    out = []
    for n_f, np_, q in [(1, 1, 1), (2, 3, 2), (3, 2, 0)]:
        X = torch.arange((q + n_f * np_) * 2, dtype=DT).reshape(1, -1, 2)
        Xa, Xf = _split_hvkg_fantasy_points(X=X, n_f=n_f, num_pareto=np_)
        out.append(
            dict(
                n_f=n_f,
                num_pareto=np_,
                x=enc(X.reshape(-1)),
                rows=X.shape[-2],
                actual=enc(Xa.reshape(-1)),
                fantasies=enc(Xf.reshape(-1)),
            )
        )
    return out


# ---------------------------------------------------------------------------
# Optimization and initial conditions
# ---------------------------------------------------------------------------


@contextlib.contextmanager
def rs_raw_samples() -> Iterator[bool]:
    """BoTorch's raw samples of ``gen_batch_initial_conditions`` from
    botorch_rs's Sobol sequence (``False`` if botorch_rs is not importable)."""
    try:
        from botorch_rs.utils.sampling import draw_sobol_samples as rs_draw
    except ImportError:
        yield False
        return
    import botorch.optim.initializers as bi

    orig = bi.draw_sobol_samples

    def draw(bounds: torch.Tensor, n: int, q: int, seed: Any = None, **kwargs: Any) -> Any:
        return torch.as_tensor(rs_draw(bounds, n=n, q=q, seed=seed), dtype=bounds.dtype)

    bi.draw_sobol_samples = draw
    try:
        yield True
    finally:
        bi.draw_sobol_samples = orig


def optimize_fixture(case: dict, seed: int) -> dict:
    """``optimize_acqf`` from explicit initial conditions and the one-shot
    initial conditions of qHVKG and its multi-fidelity variant."""
    import gpytorch
    from botorch.acquisition.multi_objective.hypervolume_knowledge_gradient import (
        qHypervolumeKnowledgeGradient,
        qMultiFidelityHypervolumeKnowledgeGradient,
    )
    from botorch.optim import optimize_acqf
    from botorch.optim.initializers import gen_one_shot_hvkg_initial_conditions
    from botorch.sampling.list_sampler import ListSampler
    from botorch.sampling.normal import SobolQMCNormalSampler

    model, specs, X = build(case, seed, "model_list")
    m, d = model.num_outputs, X.shape[-1]
    lo, hi = X.min(0).values, X.max(0).values
    bounds = torch.stack([lo, hi])
    out: dict = {
        "name": case["name"],
        "models": specs,
        "bounds": enc(bounds),
        "runs": [],
        "ics": [],
    }
    for k, (n_f, np_, q, upm) in enumerate([(2, 2, 1, True), (2, 2, 2, False)]):
        samplers = [
            SobolQMCNormalSampler(sample_shape=torch.Size([n_f]), seed=seed + 7 + j)
            for j in range(m)
        ]
        inner = SobolQMCNormalSampler(sample_shape=torch.Size([4]), seed=seed + 3)
        ref = ref_point_of(model, None, X, seed + k)
        acqf = qHypervolumeKnowledgeGradient(
            model=model,
            ref_point=ref,
            num_fantasies=n_f,
            num_pareto=np_,
            sampler=ListSampler(*samplers),
            inner_sampler=inner,
            use_posterior_mean=upm,
        )
        ics = points(X, (3, q + n_f * np_, d), seed + 20 + k)
        # With posterior samples, BoTorch's default gradients are incomplete
        # (the fantasy models' covariance caches are detached); optimize with
        # the exact gradients, and record the default run's values too.
        default_values = None
        if not upm:
            _, dv = optimize_acqf(
                acqf,
                bounds=bounds,
                q=q,
                num_restarts=3,
                batch_initial_conditions=ics,
                options={"batch_limit": 1},
                return_best_only=False,
                return_full_tree=True,
            )
            default_values = enc(dv.reshape(-1))
        with gpytorch.settings.detach_test_caches(upm):
            cands, values = optimize_acqf(
                acqf,
                bounds=bounds,
                q=q,
                num_restarts=3,
                batch_initial_conditions=ics,
                options={"batch_limit": 1},
                return_best_only=False,
                return_full_tree=True,
            )
        out["runs"].append(
            dict(
                num_fantasies=n_f,
                num_pareto=np_,
                q=q,
                use_posterior_mean=upm,
                ref_point=enc(ref),
                initial_conditions=enc(ics.reshape(-1)),
                candidates=enc(cands.reshape(-1)),
                values=enc(values.reshape(-1)),
                default_values=default_values,
                base_samples=[sampler_draws(s, n_f) for s in samplers],
                inner_base_samples=None
                if inner.base_samples is None
                else enc(inner.base_samples.reshape(4, -1)),
            )
        )
    with rs_raw_samples() as available:
        if not available:
            print(
                "phase 5c-hvkg: botorch_rs is not importable; skipping the initial conditions",
                file=sys.stderr,
            )
            return out
        for k, (n_f, np_, q, mf) in enumerate(
            [(2, 2, 1, False), (3, 2, 2, False), (2, 2, 1, True)]
        ):
            samplers = [
                SobolQMCNormalSampler(sample_shape=torch.Size([n_f]), seed=seed + 40 + j)
                for j in range(m)
            ]
            ref = ref_point_of(model, None, X, seed + 30 + k)
            kwargs: dict = dict(
                model=model,
                ref_point=ref,
                num_fantasies=n_f,
                num_pareto=np_,
                sampler=ListSampler(*samplers),
            )
            if mf:
                kwargs["target_fidelities"] = {d - 1: float(hi[-1])}
                acqf = qMultiFidelityHypervolumeKnowledgeGradient(**kwargs)
            else:
                acqf = qHypervolumeKnowledgeGradient(**kwargs)
            num_restarts, raw_samples, inner = 3, 16, 6
            options = {"seed": seed + 50 + k, "eta": 1e6}
            # All inner raw samples are optimized (num_inner_restarts ==
            # raw_inner_samples). These options are set through a wrapper of
            # the inner optimize_acqf call: BoTorch main forwards them to
            # SciPy (meta-pytorch/botorch#3395).
            import botorch.optim.optimize as bo

            orig_opt = bo.optimize_acqf

            captured: list[Any] = []

            def inner_opt(
                *args: Any,
                _np: int = np_,
                _n: int = inner,
                _opt: Any = orig_opt,
                _out: list[Any] = captured,
                **kw: Any,
            ) -> Any:
                if kw.get("q") == _np and kw.get("return_best_only") is False:
                    kw["num_restarts"] = _n
                    kw["raw_samples"] = _n
                    res = _opt(*args, **kw)
                    _out.append(res)
                    return res
                return _opt(*args, **kw)

            bo.optimize_acqf = inner_opt
            try:
                ics = gen_one_shot_hvkg_initial_conditions(
                    acq_function=acqf,
                    bounds=bounds,
                    q=q,
                    num_restarts=num_restarts,
                    raw_samples=raw_samples,
                    options=options,
                )
            finally:
                bo.optimize_acqf = orig_opt
            out["ics"].append(
                dict(
                    num_fantasies=n_f,
                    num_pareto=np_,
                    q=q,
                    multi_fidelity=mf,
                    target_fidelities=[[d - 1, float(hi[-1])]] if mf else None,
                    ref_point=enc(ref),
                    num_restarts=num_restarts,
                    raw_samples=raw_samples,
                    inner_restarts=inner,
                    # The inner optimization (in the reduced space for the
                    # multi-fidelity function).
                    inner_candidates=enc(captured[0][0].reshape(-1)),
                    inner_values=enc(captured[0][1].reshape(-1)),
                    seed=seed + 50 + k,
                    eta=1e6,
                    initial_conditions=enc(ics.reshape(-1)),
                    base_samples=[
                        None if s.base_samples is None else sampler_draws(s, n_f) for s in samplers
                    ],
                )
            )
    return out


# ---------------------------------------------------------------------------
# Cost models, utilities and fidelity helpers
# ---------------------------------------------------------------------------


def cost_fixture(seed: int) -> dict:
    from botorch.acquisition.cost_aware import InverseCostWeightedUtility
    from botorch.acquisition.utils import expand_trace_observations, project_to_target_fidelity
    from botorch.models.cost import AffineFidelityCostModel, FixedCostModel

    g = torch.Generator().manual_seed(seed)
    X = 0.1 + 0.8 * torch.rand(3, 4, generator=g, dtype=DT)
    out: dict = {
        "x": enc(X.reshape(-1)),
        "affine": [],
        "fixed": [],
        "utility": [],
        "project": [],
        "expand": [],
    }
    for weights, fixed in [(None, 0.01), ({1: 0.5, 3: 2.0}, 1.0), ({-1: 1.5, 0: 0.25}, 0.2)]:
        cm = AffineFidelityCostModel(fidelity_weights=weights, fixed_cost=fixed)
        cm.weights = cm.weights.to(DT)
        out["affine"].append(
            dict(
                weights=None if weights is None else [[k, v] for k, v in weights.items()],
                fixed_cost=fixed,
                cost=enc(cm(X).reshape(-1)),
            )
        )
    fc = FixedCostModel(torch.tensor([1.0, 2.5, 0.75], dtype=DT))
    out["fixed"].append(dict(fixed_cost=[1.0, 2.5, 0.75], cost=enc(fc(X).reshape(-1))))
    deltas = torch.tensor([[0.3], [-0.2], [0.0], [1.1]], dtype=DT)
    for model_kind, mask, log in [
        ("affine", None, False),
        ("affine", None, True),
        ("fixed", None, False),
        ("fixed", [[True, False, True]] * 3, False),
        ("fixed", [[False, True, False]] * 3, True),
    ]:
        if model_kind == "affine":
            cm = AffineFidelityCostModel(fidelity_weights={3: 2.0}, fixed_cost=0.5)
            cm.weights = cm.weights.to(DT)
        else:
            cm = fc
        u = InverseCostWeightedUtility(cost_model=cm, log=log)
        d_in = deltas.abs() + 0.1 if log else deltas
        Xb = X.unsqueeze(0)
        values = u(
            X=Xb,
            deltas=d_in,
            X_evaluation_mask=None if mask is None else torch.tensor(mask),
        )
        out["utility"].append(
            dict(
                model=model_kind,
                mask=mask,
                log=log,
                deltas=enc(d_in.reshape(-1)),
                values=enc(values.reshape(-1)),
            )
        )
    for tf, cols in [({3: 1.0}, 4), ({-1: 0.9, 0: 0.5}, 4), ({1: 0.7}, 3), ({0: 0.2, 2: 0.4}, 2)]:
        d = 4
        Xp = X[:, :cols]
        out["project"].append(
            dict(
                target_fidelities=[[k, v] for k, v in tf.items()],
                d=d,
                x=enc(Xp.reshape(-1)),
                cols=cols,
                projected=enc(
                    project_to_target_fidelity(Xp, target_fidelities=tf, d=d).reshape(-1)
                ),
            )
        )
    for dims, k in [(None, 0), (None, 1), (None, 3), ([0, 2], 2)]:
        e = expand_trace_observations(X, fidelity_dims=dims, num_trace_obs=k)
        out["expand"].append(
            dict(fidelity_dims=dims, num_trace_obs=k, expanded=enc(e.reshape(-1)), rows=e.shape[0])
        )
    return out


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------


def _write(path: str, obj: dict) -> None:
    _mo()._write(path, obj)


def generate_phase5c_hvkg(out: str) -> None:
    """Writes the fixtures that BoTorch main reproduces."""
    torch.set_num_threads(1)
    out = os.path.join(out, "phase5c_hvkg")
    os.makedirs(out, exist_ok=True)
    warnings.simplefilter("ignore")
    fixes = {"main"}
    cond = [single_conditioning(c, 8000 + 37 * i, fixes) for i, c in enumerate(SINGLE_CASES)]
    cond += [list_conditioning(c, 8500 + 41 * i, fixes) for i, c in enumerate(LIST_CASES)]
    _write(os.path.join(out, "conditioning.json"), {"models": cond})
    print(
        f"phase 5c-hvkg: wrote conditioning.json ({sum(len(c['cases']) for c in cond)} cases)",
        file=sys.stderr,
    )
    hv = [hvkg_fixture(c, 9000 + 53 * i) for i, c in enumerate(LIST_CASES)]
    _write(os.path.join(out, "hvkg.json"), {"models": hv, "split": split_cases()})
    print(
        f"phase 5c-hvkg: wrote hvkg.json ({sum(len(h['cases']) for h in hv)} cases)",
        file=sys.stderr,
    )
    _write(os.path.join(out, "optimize.json"), {"models": [optimize_fixture(LIST_CASES[0], 9700)]})
    print("phase 5c-hvkg: wrote optimize.json", file=sys.stderr)
    _write(os.path.join(out, "cost.json"), cost_fixture(9900))
    print("phase 5c-hvkg: wrote cost.json", file=sys.stderr)


def generate_phase5c_hvkg_prs(out: str) -> None:
    """Writes the cases that need meta-pytorch/botorch#3396 or #3399 (with a
    BoTorch that includes the respective pull request)."""
    torch.set_num_threads(1)
    out = os.path.join(out, "phase5c_hvkg")
    os.makedirs(out, exist_ok=True)
    warnings.simplefilter("ignore")
    wrote = False
    if has_fantasy_mean_cache_fix():
        cond = [single_conditioning(c, 8000 + 37 * i, {"3396"}) for i, c in enumerate(SINGLE_CASES)]
        _write(os.path.join(out, "conditioning_3396.json"), {"models": cond})
        print("phase 5c-hvkg-prs: wrote conditioning_3396.json", file=sys.stderr)
        wrote = True
    if has_model_list_conditioning_fix():
        cond = [list_conditioning(c, 8500 + 41 * i, {"3399"}) for i, c in enumerate(LIST_CASES)]
        _write(os.path.join(out, "conditioning_3399.json"), {"models": cond})
        print("phase 5c-hvkg-prs: wrote conditioning_3399.json", file=sys.stderr)
        wrote = True
    if not wrote:
        raise SystemExit(
            "--phase 5c-hvkg-prs needs a BoTorch with meta-pytorch/botorch#3396 or #3399"
        )

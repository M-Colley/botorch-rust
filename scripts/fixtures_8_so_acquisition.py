"""Phase 8 fixtures: the single-objective acquisition functions.

Written to ``<out>/phase8_so_acquisition/`` by ``generate_fixtures.py --phase
8-so-acquisition`` (BoTorch main). The models are built like the phase-4B
models of ``fixtures_mo_acquisition`` (training data, raw hyperparameters in
BoTorch's order; single-output, two-output and ``ModelListGP`` models).
Every ``<case>.json`` holds, with BoTorch's random numbers recorded:

* ``kg``: qKnowledgeGradient (posterior mean, ``current_value``, pending
  points, posterior transforms, linear MC objectives with an inner sampler)
  and ``mfkg``: qMultiFidelityKnowledgeGradient (target-fidelity projection,
  trace observations, ``InverseCostWeightedUtility`` of an affine cost
  model): fantasy (and inner) base samples, values and gradients;
* ``nei``: qNoisyExpectedImprovement (``cache_root`` on and off, pending
  points, linear objectives) with the base samples of every dimension BoTorch
  draws; ``mc``: qLowerConfidenceBound and qPosteriorStandardDeviation;
* ``analytic``: (Log)NoisyExpectedImprovement (the recorded fantasy draws and
  ``best_f``), LogProbabilityOfFeasibility, ScalarizedPosteriorMean,
  qAnalyticProbabilityOfImprovement (gradients of one candidate set at a
  time: BoTorch's batched MVNXPB gradients are wrong, see the draft);
* ``mes``: the max-value entropy search family with BoTorch's max values and
  sampler base samples;
* ``jes``: qJointEntropySearch ("LB" for q = 1, 2 and pending points, "MC"
  for q = 1) on BoTorch's optimal samples (the raw optimal inputs, passed to
  BoTorch so that it conditions on them (#3399); the conditioned models use
  the explicit noise in their mean (#3396)); ``pes``: qPredictiveEntropySearch
  (training inputs in train mode, i.e. raw);
* ``optimal_samples``: ``get_optimal_samples`` with the path draws and Sobol
  points; ``ts``: PathwiseThompsonSampling with the path draws;
* ``nipv``: qNegIntegratedPosteriorVariance (gradients without
  ``fast_pred_var``, which makes BoTorch's wrong; see the draft);
* ``wrappers``: the penalties of ``PenalizedAcquisitionFunction``,
  ``PenalizedMCObjective(L1PenaltyObjective)``, ``ProximalAcquisitionFunction``,
  ``FixedFeatureAcquisitionFunction`` and ``PriorGuidedAcquisitionFunction``
  (prior ``exp(-0.5 ||(x - 0.5) / 0.3||^2)``, log or not);
* ``optimize``: ``optimize_acqf`` of qKG and GIBBON from given initial
  conditions (candidates and values per restart).

Gradients of BoTorch's qKG with MC objectives are not recorded (BoTorch's are
wrong, meta-pytorch/botorch#3427); the Rust tests compare them with finite
differences. Non-finite numbers are written as the strings "inf", "-inf" and
"nan".
"""

from __future__ import annotations

import contextlib
import functools
import os
import sys
import warnings
from collections.abc import Iterator
from typing import Any

import torch

DT = torch.float64

MODEL_CASES = [
    dict(name="m1_rbf_d2", kind="single_task", models=[dict(d=2, n=9, m=1)]),
    dict(
        name="m1_matern_norm_d3",
        kind="single_task",
        models=[dict(d=3, n=10, m=1, covar="matern", normalize="shared")],
    ),
    dict(name="m1_fixed_noise_d2", kind="single_task", models=[dict(d=2, n=8, m=1, yvar=True)]),
    dict(name="m1_nostd_d2", kind="single_task", models=[dict(d=2, n=8, m=1, outcome="none")]),
    dict(name="m2_rbf_d2", kind="single_task", models=[dict(d=2, n=9, m=2)]),
    dict(
        name="list_1_1_d2",
        kind="model_list",
        models=[dict(d=2, n=9, m=1), dict(d=2, n=7, m=1, normalize="shared", seed_offset=1)],
    ),
]

TRANSFORM = [0.75, -0.25]
LINEAR = [1.0, -0.5]
NEI_SAMPLES = 32


def _mo() -> Any:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import fixtures_mo_acquisition

    return fixtures_mo_acquisition


def _entropy() -> Any:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import fixtures_mo_entropy

    return fixtures_mo_entropy


def _hvkg() -> Any:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import fixtures_mo_hvkg

    return fixtures_mo_hvkg


def enc(x: Any) -> Any:
    return _mo().enc(x)


def vg(acqf: Any, X: torch.Tensor, grads: bool = True) -> dict:
    """Values (and gradients of their sum) of ``acqf`` at ``X``."""
    Xg = X.detach().clone().requires_grad_(grads)
    v = acqf(Xg)
    out = {"x": enc(X), "values": enc(v.detach().reshape(-1))}
    if grads:
        (g,) = torch.autograd.grad(v.sum(), Xg)
        out["grads"] = enc(g)
    return out


def points(X: torch.Tensor, shape: tuple[int, ...], seed: int) -> torch.Tensor:
    """Uniform points in the bounding box of the training inputs."""
    lo, hi = X.min(0).values, X.max(0).values
    g = torch.Generator().manual_seed(seed)
    return lo + (hi - lo) * torch.rand(*shape, generator=g, dtype=DT)


def sampler(n: int, Z: torch.Tensor) -> Any:
    from botorch.sampling.normal import SobolQMCNormalSampler

    s = SobolQMCNormalSampler(sample_shape=torch.Size([n]), seed=0)
    s.base_samples = Z.clone()
    return s


def normal(shape: tuple[int, ...], seed: int) -> torch.Tensor:
    return torch.randn(*shape, generator=torch.Generator().manual_seed(seed), dtype=DT)


def seeded_draw(d: int, n: int, device: Any = None, dtype: Any = None, seed: Any = None) -> Any:
    return normal((n, d), 1000 + d)


@contextlib.contextmanager
def seeded_draws() -> Iterator[None]:
    import botorch.sampling.normal as bt_normal

    orig = bt_normal.draw_sobol_normal_samples
    bt_normal.draw_sobol_normal_samples = seeded_draw
    try:
        yield
    finally:
        bt_normal.draw_sobol_normal_samples = orig


@contextlib.contextmanager
def recorded_draws(record: list[torch.Tensor]) -> Iterator[None]:
    import botorch.sampling.normal as bt_normal

    orig = bt_normal.draw_sobol_normal_samples

    def draw(d: int, n: int, device: Any = None, dtype: Any = None, seed: Any = None) -> Any:
        z = orig(d=d, n=n, device=device, dtype=dtype, seed=seed)
        record.append(z.detach().clone())
        return z

    bt_normal.draw_sobol_normal_samples = draw
    try:
        yield
    finally:
        bt_normal.draw_sobol_normal_samples = orig


@contextlib.contextmanager
def no_fast_pred_var() -> Iterator[None]:
    import gpytorch.settings as gs
    from botorch.models import gpytorch as bg

    orig = bg.gpt_posterior_settings

    @contextlib.contextmanager
    def settings() -> Iterator[None]:
        with orig(), gs.fast_pred_var(False):
            yield

    bg.gpt_posterior_settings = settings
    try:
        yield
    finally:
        bg.gpt_posterior_settings = orig


def transform(m: int) -> Any:
    from botorch.acquisition.objective import ScalarizedPosteriorTransform

    return None if m == 1 else ScalarizedPosteriorTransform(torch.tensor(TRANSFORM, dtype=DT))


def num_outputs(case: dict) -> int:
    return sum(c["m"] for c in case["models"])


# ---------------------------------------------------------------------------
# Knowledge gradient
# ---------------------------------------------------------------------------


def kg_fixture(case: dict, model: Any, X: torch.Tensor, seed: int) -> list[dict]:
    from botorch.acquisition import knowledge_gradient as bt_kg
    from botorch.acquisition.objective import LinearMCObjective

    m = num_outputs(case)
    is_list = case["kind"] == "model_list"
    d = X.shape[-1]
    # BoTorch fantasizes every sub-model of a ModelListGP with the same
    # sampler (single-output base samples).
    m_fant = 1 if is_list else m
    out = []
    variants = ["pm", "current_value", "pending", "objective"] if m == 1 else ["pm", "objective"]
    for i, variant in enumerate(variants):
        q, n_f = 2 if m == 1 else 1, 3
        pending = points(X, (1, d), seed + i) if variant == "pending" else None
        k = q + (0 if pending is None else 1)
        Z = normal((n_f, k * m_fant), seed + 10 + i)
        bt_shape = (n_f, 1, k, m_fant) if m_fant > 1 else (n_f, 1, k)
        kw: dict[str, Any] = {"sampler": sampler(n_f, Z.reshape(bt_shape)), "X_pending": pending}
        entry: dict[str, Any] = {
            "variant": variant,
            "q": q,
            "num_fantasies": n_f,
            "base_samples": enc(Z),
            "pending": None if pending is None else enc(pending),
            "transform": TRANSFORM if (m > 1 and variant == "pm") else None,
            "objective": None,
            "current_value": None,
            "inner_base_samples": None,
        }
        if variant == "pm" and m > 1:
            kw["posterior_transform"] = transform(m)
        if variant == "current_value":
            kw["current_value"] = torch.tensor(0.25, dtype=DT)
            entry["current_value"] = 0.25
        if variant == "objective":
            w = [1.0] if m == 1 else LINEAR
            kw["objective"] = LinearMCObjective(torch.tensor(w, dtype=DT))
            Zi = normal((16, m), seed + 20 + i)
            # BoTorch's base-sample shape of the inner posteriors of the
            # fantasy models (other shapes are redrawn at random).
            kw["inner_sampler"] = sampler(
                16, Zi.reshape((16, 1, 1, 1, m) if m > 1 else (16, 1, 1, 1))
            )
            entry["objective"] = w
            entry["inner_base_samples"] = enc(Zi)
        acqf = bt_kg.qKnowledgeGradient(model, num_fantasies=n_f, **kw)
        # Two t-batches: BoTorch fails for one t-batch of multi-output
        # fantasies.
        Xt = points(X, (2, q + n_f, d), seed + 30 + i)
        entry["eval"] = vg(acqf, Xt, grads=variant != "objective")
        out.append(entry)
    return out


def mfkg_fixture(model: Any, X: torch.Tensor, seed: int) -> list[dict]:
    from botorch.acquisition import knowledge_gradient as bt_kg
    from botorch.acquisition.cost_aware import InverseCostWeightedUtility
    from botorch.acquisition.utils import expand_trace_observations, project_to_target_fidelity
    from botorch.models.cost import AffineFidelityCostModel

    d = X.shape[-1]
    tf = {d - 1: 1.0}
    project = functools.partial(project_to_target_fidelity, target_fidelities=tf, d=d)
    out = []
    for i, variant in enumerate(["project", "trace", "cost"]):
        q, n_f = 2, 3
        kw: dict[str, Any] = {"project": project}
        entry: dict[str, Any] = {
            "variant": variant,
            "q": q,
            "num_fantasies": n_f,
            "target_fidelities": [[d - 1, 1.0]],
            "expand": None,
            "cost": None,
            "current_value": None,
        }
        k = q
        if variant == "trace":
            kw["expand"] = functools.partial(
                expand_trace_observations, fidelity_dims=[d - 1], num_trace_obs=2
            )
            entry["expand"] = {"fidelity_dims": [d - 1], "num_trace_obs": 2}
            k = 3 * q
        if variant == "cost":
            cost = AffineFidelityCostModel(fidelity_weights={d - 1: 0.75}, fixed_cost=0.25)
            kw["cost_aware_utility"] = InverseCostWeightedUtility(cost_model=cost)
            kw["current_value"] = torch.tensor(0.25, dtype=DT)
            entry["cost"] = {"fidelity_weights": [[d - 1, 0.75]], "fixed_cost": 0.25}
            entry["current_value"] = 0.25
        Z = normal((n_f, k), seed + 10 + i)
        kw["sampler"] = sampler(n_f, Z.reshape(n_f, 1, k))
        entry["base_samples"] = enc(Z)
        acqf = bt_kg.qMultiFidelityKnowledgeGradient(model, num_fantasies=n_f, **kw)
        entry["eval"] = vg(acqf, points(X, (2, q + n_f, d), seed + 30 + i))
        out.append(entry)
    return out


# ---------------------------------------------------------------------------
# qNEI, qLCB, qPSTD
# ---------------------------------------------------------------------------


def nei_fixture(case: dict, model: Any, X: torch.Tensor, seed: int) -> list[dict]:
    from botorch.acquisition import monte_carlo as bt_mc
    from botorch.acquisition.objective import LinearMCObjective
    from botorch.sampling.normal import SobolQMCNormalSampler

    m = num_outputs(case)
    d = X.shape[-1]
    N = X.shape[0]
    out = []
    for i, (q, n_pend, cache_root) in enumerate(
        [(1, 0, True), (2, 0, False), (2, 1, True), (1, 2, False)]
    ):
        pending = points(X, (n_pend, d), seed + i) if n_pend else None
        kw: dict[str, Any] = {}
        if m > 1:
            kw["objective"] = LinearMCObjective(torch.tensor(LINEAR, dtype=DT))
        k = q + n_pend
        dims = [N * m, (N + k) * m]
        with seeded_draws():
            acqf = bt_mc.qNoisyExpectedImprovement(
                model,
                X,
                sampler=SobolQMCNormalSampler(sample_shape=torch.Size([NEI_SAMPLES])),
                X_pending=pending,
                prune_baseline=False,
                cache_root=cache_root,
                **kw,
            )
            ev = vg(acqf, points(X, (3, q, d), seed + 10 + i))
        out.append(
            {
                "q": q,
                "pending": None if pending is None else enc(pending),
                "cache_root": cache_root,
                "objective": LINEAR if m > 1 else None,
                "num_samples": NEI_SAMPLES,
                "draws": [enc(seeded_draw(dim, NEI_SAMPLES)) for dim in dims],
                "eval": ev,
            }
        )
    return out


def mc_fixture(case: dict, model: Any, X: torch.Tensor, seed: int) -> list[dict]:
    from botorch.acquisition import monte_carlo as bt_mc

    m = num_outputs(case)
    d = X.shape[-1]
    out = []
    for i, (name, q) in enumerate(
        [
            ("qLowerConfidenceBound", 1),
            ("qLowerConfidenceBound", 3),
            ("qPosteriorStandardDeviation", 2),
        ]
    ):
        # The scalarized posterior has one output.
        Z = normal((16, q), seed + i)
        kw: dict[str, Any] = {"sampler": sampler(16, Z.reshape(16, 1, q))}
        if m > 1:
            kw["posterior_transform"] = transform(m)
        if name == "qLowerConfidenceBound":
            kw["beta"] = 0.25
        acqf = getattr(bt_mc, name)(model, **kw)
        out.append(
            {
                "name": name,
                "beta": kw.get("beta"),
                "transform": TRANSFORM if m > 1 else None,
                "base_samples": enc(Z),
                "eval": vg(acqf, points(X, (3, q, d), seed + 10 + i)),
            }
        )
    return out


# ---------------------------------------------------------------------------
# Analytic
# ---------------------------------------------------------------------------


def analytic_fixture(case: dict, model: Any, X: torch.Tensor, seed: int) -> dict:
    from botorch.acquisition import analytic as bt_an

    m = num_outputs(case)
    d = X.shape[-1]
    out: dict[str, Any] = {"nei": [], "logpf": [], "spm": None, "qpi": []}
    if case["name"] == "m1_fixed_noise_d2":
        for i, (log, maximize) in enumerate([(False, True), (True, True), (True, False)]):
            draws: list[torch.Tensor] = []
            cls = "LogNoisyExpectedImprovement" if log else "NoisyExpectedImprovement"
            with recorded_draws(draws):
                acqf = getattr(bt_an, cls)(model, X, num_fantasies=6, maximize=maximize)
            out["nei"].append(
                {
                    "log": log,
                    "maximize": maximize,
                    "num_fantasies": 6,
                    "base_samples": enc(draws[0].reshape(6, -1)),
                    "best_f": enc(acqf.best_f.reshape(-1)),
                    "eval": vg(acqf, points(X, (3, 1, d), seed + i)),
                }
            )
    # qAnalyticProbabilityOfImprovement: BoTorch's gradients of a t-batch
    # are wrong when its members pivot differently (see the draft), so they
    # are recorded one candidate set at a time.
    best = 1.625 if m == 1 else 0.5
    for i, (q, maximize) in enumerate([(1, True), (2, False), (3, True), (4, False)]):
        acqf = bt_an.qAnalyticProbabilityOfImprovement(
            model, best, posterior_transform=transform(m), maximize=maximize
        )
        Xt = points(X, (3, q, d), seed + 20 + i)
        ev = vg(acqf, Xt, grads=False)
        ev["grads"] = [vg(acqf, Xt[k : k + 1])["grads"][0] for k in range(3)]
        out["qpi"].append(
            {
                "best_f": best,
                "maximize": maximize,
                "transform": TRANSFORM if m > 1 else None,
                "eval": ev,
            }
        )
    cons = [[0, None, 1.625]] if m == 1 else [[0, 0.5, None], [1, -1.0, 2.5]]
    acqf = bt_an.LogProbabilityOfFeasibility(model, constraints={c[0]: (c[1], c[2]) for c in cons})
    out["logpf"].append({"constraints": cons, "eval": vg(acqf, points(X, (3, 1, d), seed + 10))})
    if m == 1:
        w = [0.75, -0.25, 0.5]
        acqf = bt_an.ScalarizedPosteriorMean(model, weights=torch.tensor(w, dtype=DT))
        out["spm"] = {"weights": w, "eval": vg(acqf, points(X, (3, 3, d), seed + 11))}
    return out


# ---------------------------------------------------------------------------
# Max-value entropy search
# ---------------------------------------------------------------------------


def mes_fixture(case: dict, model: Any, X: torch.Tensor, seed: int) -> list[dict]:
    from botorch.acquisition import max_value_entropy_search as bt_mes
    from botorch.acquisition.objective import ScalarizedPosteriorTransform
    from botorch.acquisition.utils import expand_trace_observations, project_to_target_fidelity

    d = X.shape[-1]
    cs = points(X, (16, d), seed)
    plain = case["name"] == "m1_nostd_d2"
    variants = [
        ("qMaxValueEntropy", "plain"),
        ("qMaxValueEntropy", "minimize"),
        ("qMaxValueEntropy", "transform"),
        ("qMaxValueEntropy", "pending"),
        ("qLowerBoundMaxValueEntropy", "plain"),
        ("qLowerBoundMaxValueEntropy", "transform"),
    ]
    if plain:
        # BoTorch's GIBBON repulsion is wrong with outcome and input
        # transforms (see the draft).
        variants.append(("qLowerBoundMaxValueEntropy", "pending"))
    if case["name"] == "m1_matern_norm_d3":
        variants += [
            ("qMultiFidelityMaxValueEntropy", "project"),
            ("qMultiFidelityMaxValueEntropy", "trace"),
            ("qMultiFidelityLowerBoundMaxValueEntropy", "project"),
        ]
    out = []
    for i, (cls, variant) in enumerate(variants):
        mc = cls in ("qMaxValueEntropy", "qMultiFidelityMaxValueEntropy")
        kw: dict[str, Any] = {}
        entry: dict[str, Any] = {
            "cls": cls,
            "variant": variant,
            "candidate_set": enc(cs),
            "maximize": variant != "minimize",
            "transform": None,
            "pending": None,
            "target_fidelities": None,
            "expand": None,
        }
        if variant == "minimize":
            kw["maximize"] = False
        if variant == "transform":
            kw["posterior_transform"] = ScalarizedPosteriorTransform(
                torch.tensor([-0.75], dtype=DT), 0.25
            )
            entry["transform"] = [[-0.75], 0.25]
        pending = points(X, (2, d), seed + 50 + i) if variant == "pending" else None
        if pending is not None:
            entry["pending"] = enc(pending)
        if variant in ("project", "trace"):
            kw["project"] = functools.partial(
                project_to_target_fidelity, target_fidelities={d - 1: 1.0}, d=d
            )
            entry["target_fidelities"] = [[d - 1, 1.0]]
        if variant == "trace":
            kw["expand"] = functools.partial(
                expand_trace_observations, fidelity_dims=[d - 1], num_trace_obs=2
            )
            entry["expand"] = {"fidelity_dims": [d - 1], "num_trace_obs": 2}
        if mc:
            kw.update(num_fantasies=4, num_y_samples=16)
        torch.manual_seed(seed + 100 + i)
        acqf = getattr(bt_mes, cls)(model, cs, X_pending=pending, **kw)
        Xt = points(X, (4, 1, d), seed + 200 + i)
        if mc:
            acqf(Xt[:1])  # BoTorch draws the observation base samples here
            entry["y_base_samples"] = enc(acqf.sampler.base_samples.reshape(16, -1))
            entry["fantasy_base_samples"] = (
                enc(acqf.fantasies_sampler.base_samples.reshape(4, -1))
                if pending is not None
                else None
            )
        entry["max_values"] = enc(acqf.posterior_max_values.reshape(-1))
        entry["eval"] = vg(acqf, Xt)
        out.append(entry)
    return out


# ---------------------------------------------------------------------------
# Entropy search with optimal samples, Thompson sampling, active learning
# ---------------------------------------------------------------------------


@contextlib.contextmanager
def recorded_optimal_samples(draws: list[Any], candidates: list[Any]) -> Iterator[None]:
    import botorch.acquisition.utils as bt_autils
    import botorch.utils.sampling as bt_usampling

    ent = _entropy()
    noises: list[torch.Tensor] = []
    orig_get = bt_autils.get_matheron_path_model
    orig_sobol = bt_usampling.SobolEngine

    def get_path_model(*args: Any, **kwargs: Any) -> Any:
        start = len(noises)
        pm = orig_get(*args, **kwargs)
        draws.extend(ent.path_draws(pm._path, noises[start:]))
        return pm

    class Sobol(orig_sobol):  # type: ignore[misc, valid-type]
        def draw(self, *args: Any, **kwargs: Any) -> Any:
            x = super().draw(*args, **kwargs)
            candidates.append(x.clone())
            return x

    bt_autils.get_matheron_path_model = get_path_model
    bt_usampling.SobolEngine = Sobol
    try:
        with ent.recorded_randn(noises):
            yield
    finally:
        bt_autils.get_matheron_path_model = orig_get
        bt_usampling.SobolEngine = orig_sobol


def bounds_of(X: torch.Tensor) -> torch.Tensor:
    return torch.stack([X.min(0).values, X.max(0).values])


def optimal_samples_fixture(case: dict, model: Any, X: torch.Tensor, seed: int) -> dict:
    from botorch.acquisition.utils import get_optimal_samples

    m = num_outputs(case)
    bounds = bounds_of(X)
    draws: list[Any] = []
    cands: list[Any] = []
    torch.manual_seed(seed)
    with recorded_optimal_samples(draws, cands):
        xo, fo = get_optimal_samples(
            model, bounds, 4, raw_samples=64, num_restarts=4, posterior_transform=transform(m)
        )
    return {
        "bounds": enc(bounds),
        "num_optima": 4,
        "raw_samples": 64,
        "num_restarts": 4,
        "transform": TRANSFORM if m > 1 else None,
        "draws": draws,
        "candidates": enc(bounds[0] + (bounds[1] - bounds[0]) * cands[0]),
        "optimal_inputs": enc(xo),
        "optimal_outputs": enc(fo),
    }


def condition_noise(case: dict, model: Any) -> list[float] | None:
    """The noise variance of the noiseless conditioning of every output in
    its outcome-transformed space (``ModelListGP``: 1e-4 / std^2)."""
    if case["kind"] != "model_list":
        return None
    out = []
    for sub in model.models:
        tf = getattr(sub, "outcome_transform", None)
        stdvs = [1.0] * sub.num_outputs if tf is None else tf.stdvs.reshape(-1).tolist()
        out.extend(1e-4 / s**2 for s in stdvs)
    return out


def jes_fixture(case: dict, model: Any, X: torch.Tensor, opt: dict, seed: int) -> list[dict]:
    from botorch.acquisition import joint_entropy_search as bt_jes

    hv = _hvkg()
    m = num_outputs(case)
    d = X.shape[-1]
    xo = torch.tensor(opt["optimal_inputs"], dtype=DT)
    fo = torch.tensor(opt["optimal_outputs"], dtype=DT)
    # BoTorch main transforms the optimal inputs twice (#3399): pass the
    # untransformed raw inputs, which it maps back.
    tf = getattr(getattr(model, "models", [model])[0], "input_transform", None)
    xo_bt = xo if tf is None else tf.untransform(xo)
    out = []
    settings = [("LB", 1, 0), ("LB", 2, 0), ("LB", 1, 1)]
    if m == 1:
        settings.append(("MC", 1, 0))
    for i, (est, q, n_pend) in enumerate(settings):
        pending = points(X, (n_pend, d), seed + i) if n_pend else None
        torch.manual_seed(0)
        acqf = bt_jes.qJointEntropySearch(
            model,
            xo_bt,
            fo,
            posterior_transform=transform(m),
            X_pending=pending,
            estimation_type=est,
            num_samples=16,
        )
        if not hv.has_fantasy_mean_cache_fix():
            hv.use_fantasy_mean_cache(acqf.conditional_model)  # #3396
        Xt = points(X, (3, q, d), seed + 10 + i)
        entry: dict[str, Any] = {
            "estimation_type": est,
            "q": q,
            "pending": None if pending is None else enc(pending),
            "transform": TRANSFORM if m > 1 else None,
            "condition_noise": condition_noise(case, model),
            "base_samples": None,
        }
        if est == "MC":
            acqf(Xt[:1])
            entry["base_samples"] = enc(acqf.sampler.base_samples.reshape(16, -1))
        entry["eval"] = vg(acqf, Xt)
        out.append(entry)
    return out


def pes_fixture(model: Any, X: torch.Tensor, opt: dict, seed: int) -> list[dict]:
    from botorch.acquisition import predictive_entropy_search as bt_pes
    from botorch.acquisition.objective import ScalarizedPosteriorTransform
    from botorch.acquisition.utils import get_optimal_samples

    d = X.shape[-1]
    torch.manual_seed(seed)
    # Minimizers for maximize=False (with maximizers EP does not converge and
    # its result depends on rounding, in BoTorch as well).
    minimizers, _ = get_optimal_samples(
        model,
        bounds_of(X),
        3,
        raw_samples=64,
        num_restarts=4,
        posterior_transform=ScalarizedPosteriorTransform(torch.tensor([-1.0], dtype=DT)),
    )
    out = []
    for i, maximize in enumerate([True, False]):
        xo = torch.tensor(opt["optimal_inputs"], dtype=DT)[:3] if maximize else minimizers
        model.train()  # raw training inputs (BoTorch reads them at construction)
        try:
            acqf = bt_pes.qPredictiveEntropySearch(model, xo, maximize=maximize)
        finally:
            model.eval()
        out.append(
            {
                "maximize": maximize,
                "optimal_inputs": enc(xo),
                "eval": vg(acqf, points(X, (3, 2, d), seed + i)),
            }
        )
    return out


def ts_fixture(case: dict, model: Any, X: torch.Tensor, seed: int) -> list[dict]:
    from botorch.acquisition import thompson_sampling as bt_ts

    ent = _entropy()
    m = num_outputs(case)
    d = X.shape[-1]
    out = []
    # With a posterior transform and q = 1, BoTorch sums over the t-batch
    # (see the draft).
    for i, q in enumerate((1, 2) if m == 1 else (2,)):
        torch.manual_seed(seed + i)
        acqf = bt_ts.PathwiseThompsonSampling(model, posterior_transform=transform(m))
        noises: list[torch.Tensor] = []
        with ent.recorded_randn(noises):
            acqf.redraw(q)
        acqf.batch_size = q
        out.append(
            {
                "q": q,
                "transform": TRANSFORM if m > 1 else None,
                "draws": ent.path_draws(acqf.samples._path, noises),
                "eval": vg(acqf, points(X, (3, q, d), seed + 10 + i)),
            }
        )
    return out


def nipv_fixture(case: dict, model: Any, X: torch.Tensor, seed: int) -> list[dict]:
    from botorch.acquisition import active_learning as bt_al

    m = num_outputs(case)
    d = X.shape[-1]
    mc = points(X, (10, d), seed)
    out = []
    for i, (q, n_pend) in enumerate([(1, 0), (2, 1)]):
        pending = points(X, (n_pend, d), seed + 1 + i) if n_pend else None
        acqf = bt_al.qNegIntegratedPosteriorVariance(
            model, mc, posterior_transform=transform(m), X_pending=pending
        )
        with no_fast_pred_var():
            ev = vg(acqf, points(X, (3, q, d), seed + 10 + i))
        out.append(
            {
                "mc_points": enc(mc),
                "pending": None if pending is None else enc(pending),
                "transform": TRANSFORM if m > 1 else None,
                "eval": ev,
            }
        )
    return out


# ---------------------------------------------------------------------------
# Wrappers
# ---------------------------------------------------------------------------


class Prior(torch.nn.Module):
    def __init__(self, log: bool) -> None:
        super().__init__()
        self.log = log

    def forward(self, X: Any) -> Any:
        lp = -0.5 * ((X - 0.5) / 0.3).pow(2).sum(-1)
        return lp if self.log else lp.exp()


def wrappers_fixture(model: Any, X: torch.Tensor, seed: int) -> dict:
    from botorch.acquisition import analytic as bt_an
    from botorch.acquisition import fixed_feature as bt_ff
    from botorch.acquisition import logei as bt_logei
    from botorch.acquisition import monte_carlo as bt_mc
    from botorch.acquisition import penalized as bt_pen
    from botorch.acquisition import prior_guided as bt_pg
    from botorch.acquisition import proximal as bt_prox

    d = X.shape[-1]
    init = torch.tensor([0.2, 0.5, 0.7][:d], dtype=DT)
    best = 1.625
    ei = bt_an.ExpectedImprovement(model, best)
    X1 = points(X, (3, 1, d), seed)
    X2 = points(X, (3, 2, d), seed + 1)
    out: dict[str, Any] = {"best_f": best, "init_point": enc(init), "penalized": []}
    for name, pen in [
        ("L2", bt_pen.L2Penalty(init)),
        ("L1", bt_pen.L1Penalty(init)),
        ("Gaussian", bt_pen.GaussianPenalty(init, 0.5)),
        ("GroupLasso", bt_pen.GroupLassoPenalty(init, [[0, 2], [1]])),
        ("L0", bt_pen.L0PenaltyApprox(init, 0.25)),
    ]:
        acqf = bt_pen.PenalizedAcquisitionFunction(ei, pen, 0.3)
        out["penalized"].append({"penalty": name, "eval": vg(acqf, X1)})
    Z = normal((16, 2), seed + 2)
    obj = bt_pen.PenalizedMCObjective(lambda Y, X: Y[..., 0], bt_pen.L1PenaltyObjective(init), 0.2)
    acqf = bt_mc.qExpectedImprovement(
        model, best, sampler=sampler(16, Z.reshape(16, 1, 2)), objective=obj
    )
    out["penalized_objective"] = {"base_samples": enc(Z), "eval": vg(acqf, X2)}
    out["proximal"] = []
    for tw in (True, False):
        for beta in (None, 2.0):
            acqf = bt_prox.ProximalAcquisitionFunction(
                ei, torch.tensor([0.3, 0.5, 0.4][:d], dtype=DT), transformed_weighting=tw, beta=beta
            )
            out["proximal"].append(
                {
                    "transformed_weighting": tw,
                    "beta": beta,
                    "weights": [0.3, 0.5, 0.4][:d],
                    "eval": vg(acqf, X1),
                }
            )
    Zf = normal((16, 2), seed + 3)
    acqf = bt_ff.FixedFeatureAcquisitionFunction(
        bt_mc.qExpectedImprovement(model, best, sampler=sampler(16, Zf.reshape(16, 1, 2))),
        d,
        [0, 2],
        torch.tensor([[0.1, 0.9], [0.3, 0.2]], dtype=DT),
    )
    out["fixed_feature"] = {
        "columns": [0, 2],
        "values": [[0.1, 0.9], [0.3, 0.2]],
        "base_samples": enc(Zf),
        "eval": vg(acqf, X2[..., 1:2].contiguous()),
    }
    out["prior_guided"] = []
    pending = points(X, (1, d), seed + 4)
    for log in (False, True):
        for pend in (None, pending):
            k = 2 + (0 if pend is None else 1)
            Zp = normal((16, k), seed + 5 + k)
            s = sampler(16, Zp.reshape(16, 1, k))
            base = (
                bt_logei.qLogExpectedImprovement(model, best, sampler=s)
                if log
                else bt_mc.qExpectedImprovement(model, best, sampler=s)
            )
            acqf = bt_pg.PriorGuidedAcquisitionFunction(
                base, Prior(log), log=log, prior_exponent=0.75, X_pending=pend
            )
            out["prior_guided"].append(
                {
                    "log": log,
                    "base": "qLogExpectedImprovement" if log else "qExpectedImprovement",
                    "pending": None if pend is None else enc(pend),
                    "base_samples": enc(Zp),
                    "eval": vg(acqf, X2),
                }
            )
        base = bt_an.LogExpectedImprovement(model, best) if log else ei
        acqf = bt_pg.PriorGuidedAcquisitionFunction(base, Prior(log), log=log, prior_exponent=0.75)
        out["prior_guided"].append(
            {
                "log": log,
                "base": "LogExpectedImprovement" if log else "ExpectedImprovement",
                "pending": None,
                "base_samples": None,
                "eval": vg(acqf, X1),
            }
        )
    return out


# ---------------------------------------------------------------------------
# optimize_acqf
# ---------------------------------------------------------------------------


def optimize_fixture(model: Any, X: torch.Tensor, seed: int) -> list[dict]:
    from botorch.acquisition import knowledge_gradient as bt_kg
    from botorch.acquisition import max_value_entropy_search as bt_mes
    from botorch.optim import optimize_acqf

    d = X.shape[-1]
    bounds = bounds_of(X)
    out = []
    n_f = 3
    Z = normal((n_f, 1), seed)
    kg = bt_kg.qKnowledgeGradient(
        model, num_fantasies=n_f, sampler=sampler(n_f, Z.reshape(n_f, 1, 1))
    )
    ics = points(X, (3, 1 + n_f, d), seed + 1)
    cands, values = optimize_acqf(
        kg, bounds, q=1, num_restarts=3, batch_initial_conditions=ics, return_best_only=False
    )
    out.append(
        {
            "acqf": "qKnowledgeGradient",
            "num_fantasies": n_f,
            "base_samples": enc(Z),
            "bounds": enc(bounds),
            "ics": enc(ics),
            "candidates": enc(cands),
            "values": enc(values.reshape(-1)),
        }
    )
    cs = points(X, (16, d), seed + 2)
    torch.manual_seed(seed + 3)
    gibbon = bt_mes.qLowerBoundMaxValueEntropy(model, cs)
    ics = points(X, (3, 1, d), seed + 4)
    cands, values = optimize_acqf(
        gibbon, bounds, q=1, num_restarts=3, batch_initial_conditions=ics, return_best_only=False
    )
    out.append(
        {
            "acqf": "qLowerBoundMaxValueEntropy",
            "candidate_set": enc(cs),
            "max_values": enc(gibbon.posterior_max_values.reshape(-1)),
            "bounds": enc(bounds),
            "ics": enc(ics),
            "candidates": enc(cands),
            "values": enc(values.reshape(-1)),
        }
    )
    return out


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def model_fixture(case: dict, seed: int) -> dict:
    mo = _mo()
    model, specs, X = mo.build_case_model(case, seed)
    m = num_outputs(case)
    with mo.dense_model_list():
        opt = optimal_samples_fixture(case, model, X, seed + 1)
        fixture: dict[str, Any] = {
            "name": case["name"],
            "kind": case["kind"],
            "versions": mo.versions(mo._gf()),
            "models": specs,
            "baseline": enc(X),
            "kg": kg_fixture(case, model, X, seed + 100),
            "nei": nei_fixture(case, model, X, seed + 200),
            "mc": mc_fixture(case, model, X, seed + 300),
            "analytic": analytic_fixture(case, model, X, seed + 400),
            "optimal_samples": opt,
            "jes": jes_fixture(case, model, X, opt, seed + 500),
            "ts": ts_fixture(case, model, X, seed + 600),
            "nipv": nipv_fixture(case, model, X, seed + 700),
        }
        if m == 1:
            fixture["mes"] = mes_fixture(case, model, X, seed + 800)
        if case["name"] == "m1_rbf_d2":
            fixture["pes"] = pes_fixture(model, X, opt, seed + 900)
            fixture["optimize"] = optimize_fixture(model, X, seed + 1000)
        if case["name"] == "m1_matern_norm_d3":
            fixture["mfkg"] = mfkg_fixture(model, X, seed + 1100)
            fixture["wrappers"] = wrappers_fixture(model, X, seed + 1200)
    return fixture


def generate_phase8_so_acquisition(out: str) -> None:
    """Writes the phase-8 fixtures (BoTorch main)."""
    torch.set_num_threads(1)
    out = os.path.join(out, "phase8_so_acquisition")
    os.makedirs(out, exist_ok=True)
    warnings.simplefilter("ignore")
    mo = _mo()
    for i, case in enumerate(MODEL_CASES):
        fixture = model_fixture(case, seed=8800 + 37 * i)
        mo._write(os.path.join(out, f"{case['name']}.json"), fixture)
        sections = [
            k
            for k, v in fixture.items()
            if isinstance(v, (list, dict)) and k not in ("models", "versions", "baseline")
        ]
        print(
            f"phase 8-so-acquisition: wrote {case['name']}.json ({', '.join(sections)})",
            file=sys.stderr,
        )

"""Phase 5C-entropy fixtures: pathwise sampling and BoTorch's multi-objective
information-theoretic acquisition functions.

Written to ``<out>/phase5c_entropy/`` by ``generate_fixtures.py --phase
5c-entropy`` (with BoTorch main or a release):

* ``<model>.json`` for five models with explicit hyperparameters (the
  phase-4 builders): multi-output ``SingleTaskGP`` models with m = 2 (RBF)
  and m = 3 (Matern-5/2, ``Normalize``), a ``ModelListGP`` of two
  single-output models (different training data, one with ``Normalize``), a
  single-output model with a ``ScaleKernel`` (M = 1) and a fixed-noise model
  with m = 2. Each holds

  - ``paths``: ``draw_matheron_paths`` (64 Fourier features, 3 samples) with
    BoTorch's random numbers (the Fourier frequencies and prior weights read
    from the path objects, the noise of ``gaussian_update`` recorded from
    ``torch.randn_like``), the path values and their autograd derivatives at
    test points, the update weights and the prior paths of
    ``draw_kernel_feature_paths`` (with the outcome transform);
  - ``optima``: ``sample_optimal_points`` (3 samples, 4 points, the default
    1024 features, ``pop_size=256``) with every path draw and every candidate
    set of ``random_search_optimizer``, and the sampled Pareto sets and
    fronts;
  - ``boxes``: ``compute_sample_box_decomposition`` of the fronts
    (maximization, minimization of the negated fronts, one constraint);
  - ``acqf``: values and gradients (t-batches of three, q = 1 and 2, pending
    points) of ``qLowerBoundMultiObjectiveMaxValueEntropySearch`` and
    ``qLowerBoundMultiObjectiveJointEntropySearch`` with every
    ``estimation_type`` (the MC base samples are recorded), on the sampled
    Pareto sets and fronts;
  - ``pes``: ``qMultiObjectivePredictiveEntropySearch`` (maximization and
    minimization): the number of EP iterations, the damping factors, the
    cached EP factors, values and gradients, and BoTorch's own changes of
    them (the noise floor) when the training inputs are perturbed by 1e-14
    (relative, two copies) and when every result of ``torch.exp`` and
    ``torch.erf`` is perturbed by one unit in the last place (two copies,
    ``ulp_noise``): EP evaluates ``log1p(-exp(S))`` for ``S`` near zero and
    normal CDFs far in the tail with cancellation, so its factors depend on
    the last bits of the math library.

  - ``ep_steps`` (``m2_rbf_d2_n10``; maximization and minimization): the
    inputs and outputs of every step of the first EP iteration and of the
    first one that tests convergence (the fourth), and the change of every
    new factor of ``_update_omega`` under last-bit noise in exp and erf on the
    same inputs.

* ``estimators.json``: BoTorch's ``_compute_entropy_noiseless``,
  ``_compute_entropy_upper_bound`` (both variants) and
  ``_compute_entropy_monte_carlo`` with their autograd derivatives on random
  inputs (padded cells, cells reaching -1e10, M = 1, 2, 3), and
  ``log_cdf_robust`` on a grid.

Corrected behaviour (as botorch_rs implements it):

* JES conditions on the Pareto sets transformed once
  (meta-pytorch/botorch#3399). Without that fix (BoTorch main, 0.18.1), the
  fixtures pass ``model.input_transform.untransform(pareto_sets)`` to JES for
  models with an input transform, which BoTorch then transforms twice: the
  same conditioning as #3399.
* PES evaluates its EP at the raw training inputs: BoTorch reads
  ``model.train_inputs``, which are already transformed in eval mode, and
  transforms them again in ``model.posterior``. The fixtures construct PES
  with the model in train mode, where ``train_inputs`` are the raw inputs.

``ModelListGP`` posteriors are evaluated with dense covariances (see
``dense_model_list`` in ``fixtures_mo_acquisition.py``). Non-finite numbers
are written as the strings "inf", "-inf" and "nan".
"""

from __future__ import annotations

import contextlib
import functools
import inspect
import json
import os
import sys
import warnings
from collections.abc import Iterator
from typing import Any

import torch

DT = torch.float64
PATH_SAMPLES = 3
PATH_FEATURES = 64
NUM_PARETO_SAMPLES = 3
NUM_PARETO_POINTS = 4
POP_SIZE = 256
MC_SAMPLES = 16
ESTIMATION_TYPES = ["0", "LB", "LB2", "MC"]
# The EP iterations recorded step by step (``ep_steps``): the first and the
# first one that tests convergence.
EP_STEPS = 4
EP_RECORD = (0, 3)
EP_STEP_CASES = ("m2_rbf_d2_n10",)

MODEL_CASES = [
    dict(name="m2_rbf_d2_n10", kind="single_task", models=[dict(d=2, n=10, m=2)]),
    dict(
        name="m3_matern_d3_n10_normalize",
        kind="single_task",
        models=[dict(d=3, n=10, m=3, covar="matern", normalize="shared", x_range=(-1.0, 2.0))],
    ),
    dict(
        name="list_1_1_d2",
        kind="model_list",
        models=[
            dict(d=2, n=9, m=1),
            dict(d=2, n=7, m=1, covar="matern", normalize="shared", seed_offset=1),
        ],
    ),
    dict(name="m1_scale_d2_n8", kind="single_task", models=[dict(d=2, n=8, m=1, covar="matern_gamma")]),
    dict(name="m2_fixed_noise_d2_n8", kind="single_task", models=[dict(d=2, n=8, m=2, yvar=True)]),
]


def _mo() -> Any:
    """The shared helpers of ``fixtures_mo_acquisition.py``."""
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import fixtures_mo_acquisition

    return fixtures_mo_acquisition


def enc(x: Any) -> Any:
    return _mo().enc(x)


# ---------------------------------------------------------------------------
# BoTorch instrumentation
# ---------------------------------------------------------------------------


@contextlib.contextmanager
def recorded_randn(record: list[torch.Tensor]) -> Iterator[None]:
    """Records the draws of ``torch.randn_like`` (the noise of
    ``gaussian_update``)."""
    orig = torch.randn_like

    def randn_like(t: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        z = orig(t, *args, **kwargs)
        record.append(z.detach().clone())
        return z

    torch.randn_like = randn_like  # type: ignore[assignment]
    try:
        yield
    finally:
        torch.randn_like = orig  # type: ignore[assignment]


@contextlib.contextmanager
def ulp_noise(seed: int) -> Iterator[None]:
    """Perturbs every result of ``torch.exp`` and ``torch.erf`` by about one
    unit in the last place (a factor ``1 ± 2^-52``, random signs from
    ``seed``): BoTorch with a math library whose last bits differ. PES
    evaluates ``log1p(-exp(S))`` for ``S`` near zero and the normal CDF far in
    the tail as ``0.5 (1 + erf(x / sqrt(2)))``, where such last-bit
    differences change the EP factors by up to a few percent."""
    g = torch.Generator().manual_seed(seed)
    orig_exp, orig_erf = torch.exp, torch.erf

    def factors(y: torch.Tensor, keep: torch.Tensor) -> torch.Tensor:
        sign = torch.randint(0, 2, y.shape, generator=g).to(y.dtype) * 2 - 1
        return torch.where(keep, 1 + sign * 2.0**-52, torch.ones_like(y))

    def exp(x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        y = orig_exp(x, *args, **kwargs)
        return y * factors(y, torch.ones_like(y, dtype=torch.bool))

    def erf(x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        y = orig_erf(x, *args, **kwargs)
        return y * factors(y, y.abs() <= 1 - 2.0**-50)

    torch.exp = exp  # type: ignore[assignment]
    torch.erf = erf  # type: ignore[assignment]
    try:
        yield
    finally:
        torch.exp = orig_exp  # type: ignore[assignment]
        torch.erf = orig_erf  # type: ignore[assignment]


def has_3399() -> bool:
    """Whether BoTorch's JES conditions on the raw Pareto sets (#3399)."""
    from botorch.acquisition.multi_objective import joint_entropy_search as jes

    src = inspect.getsource(jes.qLowerBoundMultiObjectiveJointEntropySearch.__init__)
    return "transform_inputs(self.pareto_sets)" not in src


def jes_pareto_sets(model: Any, pareto_sets: torch.Tensor) -> torch.Tensor:
    """The Pareto sets to pass to BoTorch's JES for #3399's conditioning."""
    tf = getattr(model, "input_transform", None)
    if has_3399() or tf is None:
        return pareto_sets
    return tf.untransform(pareto_sets)


def matheron_models(paths: Any) -> list[Any]:
    """The ``MatheronPath`` of every sub-model."""
    from botorch.sampling.pathwise import PathList

    return list(paths) if isinstance(paths, PathList) else [paths]


def path_draws(paths: Any, noises: list[torch.Tensor]) -> list[dict]:
    """BoTorch's random numbers of Matheron paths, one entry per output:
    frequencies (``L/2 x d``), weights (``S x L``) and standard normal noise
    (``S x n``)."""
    out = []
    subs = matheron_models(paths)
    assert len(subs) == len(noises), (len(subs), len(noises))
    for sub, noise in zip(subs, noises, strict=True):
        prior = sub.paths["prior_paths"]
        freq = prior.feature_map.weight.detach()
        weight = prior.weight.detach()
        if freq.ndim == 2:  # single-output: no batch dimension
            freq, weight, noise = freq.unsqueeze(0), weight.unsqueeze(-2), noise.unsqueeze(-2)
        for t in range(freq.shape[0]):
            out.append(
                {
                    "frequencies": enc(freq[t]),
                    "weights": enc(weight[..., t, :].reshape(-1, weight.shape[-1])),
                    "noise": enc(noise[..., t, :].reshape(-1, noise.shape[-1])),
                }
            )
    return out


def per_output(values: Any, m_per_model: list[int]) -> list[torch.Tensor]:
    """Path values as one ``S x n`` tensor per output."""
    vals = values if isinstance(values, list) else [values]
    out = []
    for v, m in zip(vals, m_per_model, strict=True):
        if m == 1:
            out.append(v)
        else:
            out.extend(v[..., t, :] for t in range(m))
    return out


# ---------------------------------------------------------------------------
# Fixture parts
# ---------------------------------------------------------------------------


def paths_fixture(model: Any, ms: list[int], X: torch.Tensor, seed: int) -> dict:
    from botorch.sampling.pathwise import draw_kernel_feature_paths, draw_matheron_paths

    torch.manual_seed(seed)
    noises: list[torch.Tensor] = []
    sampler = functools.partial(draw_kernel_feature_paths, num_features=PATH_FEATURES)
    with recorded_randn(noises):
        paths = draw_matheron_paths(model, torch.Size([PATH_SAMPLES]), prior_sampler=sampler)
    Xg = X.detach().clone().requires_grad_(True)
    vals = per_output(paths(Xg), ms)
    grads = []
    for v in vals:
        per_sample = []
        for s in range(PATH_SAMPLES):
            (g,) = torch.autograd.grad(v[s].sum(), Xg, retain_graph=True)
            per_sample.append(enc(g))
        grads.append(per_sample)
    update_weights = []
    for sub, m in zip(matheron_models(paths), ms, strict=True):
        w = sub.paths["update_paths"].weight.detach()
        update_weights.extend([w] if m == 1 else [w[..., t, :] for t in range(m)])
    # The prior paths with the outcome transform (draw_kernel_feature_paths).
    torch.manual_seed(seed + 1)
    prior = draw_kernel_feature_paths(model, sample_shape=torch.Size([PATH_SAMPLES]), num_features=PATH_FEATURES)
    prior_draws = []
    for sub in matheron_models(prior):
        freq, weight = sub.feature_map.weight.detach(), sub.weight.detach()
        if freq.ndim == 2:
            freq, weight = freq.unsqueeze(0), weight.unsqueeze(-2)
        prior_draws.extend(
            {"frequencies": enc(freq[t]), "weights": enc(weight[..., t, :])} for t in range(freq.shape[0])
        )
    with torch.no_grad():
        prior_vals = per_output(prior(X), ms)
    return {
        "num_samples": PATH_SAMPLES,
        "num_features": PATH_FEATURES,
        "draws": path_draws(paths, noises),
        "x": enc(X),
        "values": [enc(v.detach()) for v in vals],
        "grads": grads,
        "update_weights": [enc(w) for w in update_weights],
        "prior_draws": prior_draws,
        "prior_values": [enc(v) for v in prior_vals],
    }


def optima_fixture(model: Any, bounds: torch.Tensor, seed: int, maximize: bool = True) -> dict:
    import botorch.acquisition.multi_objective.utils as bt_utils

    m = model.num_outputs
    num_points = NUM_PARETO_POINTS if m > 1 else 1
    draws: list[list[dict]] = []
    candidates: list[list[Any]] = []
    noises: list[torch.Tensor] = []
    orig_get = bt_utils.get_matheron_path_model
    orig_draw = bt_utils.draw_sobol_samples

    def get_path_model(model: Any, *args: Any, **kwargs: Any) -> Any:
        start = len(noises)
        pm = orig_get(model, *args, **kwargs)
        draws.append(path_draws(pm._path, noises[start:]))
        candidates.append([])
        return pm

    def draw_sobol(*args: Any, **kwargs: Any) -> torch.Tensor:
        X = orig_draw(*args, **kwargs)
        candidates[-1].append(enc(X.squeeze(-2)))
        return X

    torch.manual_seed(seed)
    bt_utils.get_matheron_path_model = get_path_model
    bt_utils.draw_sobol_samples = draw_sobol
    try:
        with recorded_randn(noises):
            ps, pf = bt_utils.sample_optimal_points(
                model,
                bounds,
                num_samples=NUM_PARETO_SAMPLES,
                num_points=num_points,
                maximize=maximize,
                optimizer_kwargs={"pop_size": POP_SIZE},
            )
    finally:
        bt_utils.get_matheron_path_model = orig_get
        bt_utils.draw_sobol_samples = orig_draw
    return {
        "num_samples": NUM_PARETO_SAMPLES,
        "num_points": num_points,
        "pop_size": POP_SIZE,
        "max_tries": 10,
        "maximize": maximize,
        "draws": draws,
        "candidates": candidates,
        "pareto_sets": enc(ps),
        "pareto_fronts": enc(pf),
    }


def boxes_fixture(pf: torch.Tensor) -> list[dict]:
    from botorch.acquisition.multi_objective.utils import compute_sample_box_decomposition

    out = []
    for maximize, k in ((True, 0), (False, 0), (True, 1)):
        fronts = pf if maximize else -pf
        hb = compute_sample_box_decomposition(fronts, maximize=maximize, num_constraints=k)
        out.append(
            {
                "maximize": maximize,
                "num_constraints": k,
                "fronts": enc(fronts),
                "shape": list(hb.shape),
                "bounds": enc(hb.reshape(-1)),
            }
        )
    return out


def acqf_fixture(model: Any, X: torch.Tensor, ps: torch.Tensor, pf: torch.Tensor, seed: int) -> list[dict]:
    from botorch.acquisition.multi_objective.joint_entropy_search import (
        qLowerBoundMultiObjectiveJointEntropySearch,
    )
    from botorch.acquisition.multi_objective.max_value_entropy_search import (
        qLowerBoundMultiObjectiveMaxValueEntropySearch,
    )
    from botorch.acquisition.multi_objective.utils import compute_sample_box_decomposition

    mo = _mo()
    hb = compute_sample_box_decomposition(pf)
    d = X.shape[-1]
    lo, hi = X.min(0).values, X.max(0).values
    g = torch.Generator().manual_seed(seed)
    out = []
    for acqf_name in ("MES", "JES"):
        for est in ESTIMATION_TYPES:
            for q, pending in ((1, False), (2, False), (1, True)):
                if pending and est not in ("LB", "MC"):
                    continue
                Xt = lo + (hi - lo) * torch.rand(3, q, d, generator=g, dtype=DT)
                Xp = lo + (hi - lo) * torch.rand(1, d, generator=g, dtype=DT) if pending else None
                entry: dict[str, Any] = {
                    "acqf": acqf_name,
                    "estimation_type": est,
                    "x": enc(Xt),
                    "x_pending": None if Xp is None else enc(Xp),
                }

                def evaluate(
                    m: Any, name: str = acqf_name, est: str = est, Xp: Any = Xp, Xt: Any = Xt, q: int = q
                ) -> Any:
                    torch.manual_seed(seed + q)
                    with mo.dense_model_list():
                        if name == "MES":
                            acqf = qLowerBoundMultiObjectiveMaxValueEntropySearch(
                                m, hb, X_pending=Xp, estimation_type=est, num_samples=MC_SAMPLES
                            )
                        else:
                            acqf = qLowerBoundMultiObjectiveJointEntropySearch(
                                m,
                                jes_pareto_sets(m, ps),
                                pf,
                                hb,
                                X_pending=Xp,
                                estimation_type=est,
                                num_samples=MC_SAMPLES,
                            )
                        v, gr = mo.values_and_grads(acqf, Xt)
                    return v, gr, acqf

                try:
                    v, gr, acqf = evaluate(model)
                except RuntimeError as e:
                    entry["error"] = str(e)
                    out.append(entry)
                    continue
                entry["floor"] = mo.floors(lambda m: evaluate(m)[:2], model, v, gr, log=False)
                entry["values"] = enc(v)
                entry["grads"] = enc(gr)
                entry["base_samples"] = (
                    enc(acqf.sampler.base_samples.reshape(MC_SAMPLES, -1)) if est == "MC" else None
                )
                out.append(entry)
    return out


def finite_rel_err(a: torch.Tensor, e: torch.Tensor, atol: float = 0.0) -> float:
    """``rel_err`` over the entries that are finite in both (BoTorch's PES
    gradients are NaN where autograd differentiates through clamped
    logarithms)."""
    mask = torch.isfinite(a) & torch.isfinite(e)
    if not mask.any():
        return 0.0
    return _mo().rel_err(a[mask], e[mask], atol=atol)


def pes_fixture(model: Any, X: torch.Tensor, ps: torch.Tensor, seed: int) -> list[dict]:
    from botorch.acquisition.multi_objective import predictive_entropy_search as bt_pes

    mo = _mo()
    d = X.shape[-1]
    lo, hi = X.min(0).values, X.max(0).values
    g = torch.Generator().manual_seed(seed)
    out = []
    for maximize, q, pending in ((True, 1, False), (True, 2, False), (False, 1, True)):
        Xt = lo + (hi - lo) * torch.rand(3, q, d, generator=g, dtype=DT)
        Xp = lo + (hi - lo) * torch.rand(1, d, generator=g, dtype=DT) if pending else None
        entry: dict[str, Any] = {
            "maximize": maximize,
            "x": enc(Xt),
            "x_pending": None if Xp is None else enc(Xp),
        }
        results = []
        # The model, two copies with perturbed training inputs and two with
        # last-bit noise in exp and erf.
        for twin in range(5):
            m = model if twin in (0, 3, 4) else mo.perturbed_copy(model, seed=seed + 7 * twin)
            noise = ulp_noise(seed + 11 * twin) if twin >= 3 else contextlib.nullcontext()
            log: list[torch.Tensor] = []
            orig = bt_pes._update_damping_when_converged

            def converged(*args: Any, **kwargs: Any) -> Any:
                res = orig(*args, **kwargs)
                log.append(res[0].detach().clone())
                return res

            bt_pes._update_damping_when_converged = converged
            try:
                with mo.dense_model_list(), noise:
                    # The raw training inputs (see the module documentation).
                    m.train()
                    acqf = bt_pes.qMultiObjectivePredictiveEntropySearch(
                        m, ps, maximize=maximize, X_pending=Xp
                    )
                    v, gr = mo.values_and_grads(acqf, Xt)
            except (RuntimeError, ValueError) as e:
                entry["error"] = f"{type(e).__name__}: {e}"
                results = []
                break
            finally:
                bt_pes._update_damping_when_converged = orig
                m.eval()
            results.append((v, gr, acqf, log))
        if not results:
            out.append(entry)
            continue
        v, gr, acqf, log = results[0]
        N = acqf.augmented_X.shape[-2] - ps.shape[-2]
        entry.update(
            {
                "values": enc(v),
                "grads": enc(gr),
                "iterations": len(log),
                "damping": enc(log[-1]),
                "omega_nat_mean": enc(acqf._omega_f_nat_mean[..., N:, :, :]),
                "omega_nat_cov": enc(acqf._omega_f_nat_cov[..., N:, :, :, :]),
                "floor": {
                    "value": max(finite_rel_err(r[0], v) for r in results[1:]),
                    "grad": max(finite_rel_err(r[1], gr, atol=1e-14) for r in results[1:]),
                    "omega": max(
                        finite_rel_err(
                            r[2]._omega_f_nat_cov[..., N:, :, :, :], acqf._omega_f_nat_cov[..., N:, :, :, :]
                        )
                        for r in results[1:]
                    ),
                    "iterations": [len(r[3]) for r in results],
                },
            }
        )
        out.append(entry)
    return out


def ep_steps_fixture(model: Any, ps: torch.Tensor, maximize: bool, seed: int) -> dict:
    """BoTorch's EP iterates: the inputs and outputs of every step of the
    iterations ``EP_RECORD`` (``_update_omega``, ``_update_marginals``,
    ``_update_damping``, ``_update_damping_when_converged``), and the change
    of every new factor of ``_update_omega`` under last-bit noise in exp and
    erf (``ulp_noise``, two copies) on the same inputs."""
    from botorch.acquisition.multi_objective import predictive_entropy_search as bt_pes

    steps: list[dict] = []
    names = ("_update_omega", "_update_marginals", "_update_damping", "_update_damping_when_converged")
    origs = {name: getattr(bt_pes, name) for name in names}

    def record(name: str) -> Any:
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            if len(steps) > EP_STEPS:
                return origs[name](*args, **kwargs)
            if name == "_update_omega":
                steps.append({})
            inputs = {k: (v.detach().clone() if torch.is_tensor(v) else v) for k, v in kwargs.items()}
            out = origs[name](*args, **kwargs)
            if len(steps) <= EP_STEPS:
                steps[-1][name] = {"inputs": inputs, "outputs": out}
            return out

        return wrapper

    for name in names:
        setattr(bt_pes, name, record(name))
    try:
        model.train()
        bt_pes.qMultiObjectivePredictiveEntropySearch(model, ps, maximize=maximize)
    finally:
        for name in names:
            setattr(bt_pes, name, origs[name])
        model.eval()
    out = []
    for k in EP_RECORD:
        step = steps[k]
        om = step["_update_omega"]
        kw = om["inputs"]
        new_mean, new_cov = om["outputs"]
        floor_mean = torch.zeros_like(new_mean)
        floor_cov = torch.zeros_like(new_cov)
        for twin in (1, 2):
            with ulp_noise(seed + 13 * twin):
                tm, tc = origs["_update_omega"](**{k: (v.clone() if torch.is_tensor(v) else v) for k, v in kw.items()})
            floor_mean = torch.maximum(floor_mean, (tm - new_mean).abs())
            floor_cov = torch.maximum(floor_cov, (tc - new_cov).abs())
        mg = step["_update_marginals"]
        dp = step["_update_damping"]
        cv = step["_update_damping_when_converged"]
        out.append(
            {
                "omega": {
                    "mean_f": enc(kw["mean_f"]),
                    "cov_f": enc(kw["cov_f"]),
                    "omega_mean": enc(kw["omega_f_nat_mean"]),
                    "omega_cov": enc(kw["omega_f_nat_cov"]),
                    "N": kw["N"],
                    "P": kw["P"],
                    "jitter": kw["jitter"],
                    "new_mean": enc(new_mean),
                    "new_cov": enc(new_cov),
                    "floor_mean": enc(floor_mean),
                    "floor_cov": enc(floor_cov),
                },
                "marginals": {
                    "pred_nat_mean": enc(mg["inputs"]["pred_f_nat_mean"]),
                    "pred_nat_cov": enc(mg["inputs"]["pred_f_nat_cov"]),
                    "omega_mean": enc(mg["inputs"]["omega_f_nat_mean"]),
                    "omega_cov": enc(mg["inputs"]["omega_f_nat_cov"]),
                    "nat_mean": enc(mg["outputs"][0]),
                    "nat_cov": enc(mg["outputs"][1]),
                },
                "damping": {
                    "nat_cov": enc(dp["inputs"]["nat_cov"]),
                    "nat_cov_new": enc(dp["inputs"]["nat_cov_new"]),
                    "damping": enc(dp["inputs"]["damping_factor"]),
                    "jitter": dp["inputs"]["jitter"],
                    "new_damping": enc(dp["outputs"][0]),
                    "cholesky": enc(dp["outputs"][1]),
                },
                "iteration": k,
                "converged": {
                    "mean_old": enc(cv["inputs"]["mean_old"]),
                    "mean_new": enc(cv["inputs"]["mean_new"]),
                    "cov_old": enc(cv["inputs"]["cov_old"]),
                    "cov_new": enc(cv["inputs"]["cov_new"]),
                    "damping": enc(cv["inputs"]["damping_factor"]),
                    "iteration": cv["inputs"]["iteration"],
                    "threshold": cv["inputs"]["threshold"],
                    "new_damping": enc(cv["outputs"][0]),
                },
            }
        )
    return {"maximize": maximize, "steps": out}


def model_fixture(case: dict, seed: int) -> dict:
    mo = _mo()
    model, specs, Xtrain = mo.build_case_model(case, seed)
    cfg = case["models"][0]
    lo, hi = cfg.get("x_range", (0.0, 1.0))
    d = cfg["d"]
    bounds = torch.stack([torch.full((d,), lo, dtype=DT), torch.full((d,), hi, dtype=DT)])
    ms = [c["m"] for c in case["models"]]
    g = torch.Generator().manual_seed(seed + 3)
    X = lo + (hi - lo) * torch.rand(5, d, generator=g, dtype=DT)
    with mo.dense_model_list():
        fixture = {
            "name": case["name"],
            "kind": case["kind"],
            "versions": mo.versions(mo._gf()),
            "has_3399": has_3399(),
            "models": specs,
            "bounds": enc(bounds),
            "paths": paths_fixture(model, ms, X, seed + 10),
            "optima": optima_fixture(model, bounds, seed + 20),
        }
    ps = torch.tensor(fixture["optima"]["pareto_sets"], dtype=DT)
    pf = torch.tensor(fixture["optima"]["pareto_fronts"], dtype=DT)
    fixture["boxes"] = boxes_fixture(pf)
    fixture["acqf"] = acqf_fixture(model, Xtrain, ps, pf, seed + 30)
    fixture["pes"] = pes_fixture(model, Xtrain, ps, seed + 40)
    if case["name"] in EP_STEP_CASES:
        fixture["ep_steps"] = [ep_steps_fixture(model, ps, maximize, seed + 50) for maximize in (True, False)]
    return fixture


def estimators_fixture(seed: int) -> dict:
    from botorch.acquisition.multi_objective import joint_entropy_search as jes
    from botorch.acquisition.multi_objective.predictive_entropy_search import log_cdf_robust

    g = torch.Generator().manual_seed(seed)
    cases = []
    for m in (1, 2, 3):
        b, P, q, J, S = 2, 3, 2, 5, 7
        lower = -1.5 + 2.0 * torch.rand(P, J, m, generator=g, dtype=DT)
        upper = lower + torch.rand(P, J, m, generator=g, dtype=DT)
        lower[:, 0, :] = -1e10  # a cell reaching NEG_INF
        lower[0, -1, :] = 0.0  # a padded (empty) cell
        upper[0, -1, :] = 0.0
        hb = torch.stack([lower, upper], dim=1)
        mean = (torch.rand(b, P, q, 1, m, generator=g, dtype=DT) - 0.5).requires_grad_(True)
        var = (0.2 + torch.rand(b, P, q, 1, m, generator=g, dtype=DT)).requires_grad_(True)
        noise = (0.05 + 0.2 * torch.rand(b, P, q, 1, m, generator=g, dtype=DT)).requires_grad_(True)
        samples = (2.0 * torch.rand(S, b, P, q, 1, m, generator=g, dtype=DT) - 1.0).requires_grad_(True)
        lp = (-1.0 - torch.rand(S, b, P, q, generator=g, dtype=DT)).requires_grad_(True)
        results = {}
        for name, fn in (
            ("0", lambda: jes._compute_entropy_noiseless(hb, mean, var, noise)),
            ("LB", lambda: jes._compute_entropy_upper_bound(hb, mean, var, noise, only_diagonal=False)),
            ("LB2", lambda: jes._compute_entropy_upper_bound(hb, mean, var, noise, only_diagonal=True)),
            ("MC", lambda: jes._compute_entropy_monte_carlo(hb, mean, var, noise, samples, lp)),
        ):
            h = fn()
            inputs = [mean, var, noise] + ([samples, lp] if name == "MC" else [])
            grads = torch.autograd.grad(h.sum(), inputs)
            results[name] = {
                "values": enc(h),
                "grad_mean": enc(grads[0]),
                "grad_variance": enc(grads[1]),
                "grad_noise": enc(grads[2]),
                "grad_samples": enc(grads[3]) if name == "MC" else None,
                "grad_log_prob": enc(grads[4]) if name == "MC" else None,
            }
        cases.append(
            {
                "m": m,
                "shape": [b, P, q, J, S],
                "hypercell_bounds": enc(hb),
                "mean": enc(mean),
                "variance": enc(var),
                "noise": enc(noise),
                "samples": enc(samples),
                "log_prob": enc(lp),
                "results": results,
            }
        )
    x = torch.cat(
        [
            torch.linspace(-40.0, 40.0, 161, dtype=DT),
            torch.tensor([-8.3, -8.2, 4.999, 5.0, 5.001, 8.2, 8.3, 0.0], dtype=DT),
        ]
    )
    return {
        "versions": _mo().versions(_mo()._gf()),
        "cases": cases,
        "log_cdf_robust": {"x": enc(x), "values": enc(log_cdf_robust(x))},
    }


def generate_phase5c_entropy(out: str) -> None:
    """Writes the phase-5C entropy-search fixtures."""
    torch.set_num_threads(1)
    out = os.path.join(out, "phase5c_entropy")
    os.makedirs(out, exist_ok=True)
    warnings.simplefilter("ignore")
    mo = _mo()
    for i, case in enumerate(MODEL_CASES):
        fixture = model_fixture(case, seed=7000 + 31 * i)
        mo._write(os.path.join(out, f"{case['name']}.json"), fixture)
        floors = [e["floor"] for e in fixture["pes"] if "floor" in e]
        print(
            f"phase 5c-entropy: wrote {case['name']}.json ({len(fixture['acqf'])} MES/JES and "
            f"{len(fixture['pes'])} PES cases; PES iterations "
            f"{[f['iterations'] for f in floors]}, largest PES noise floor "
            f"{max((max(f['value'], f['grad']) for f in floors), default=0.0):.1e})",
            file=sys.stderr,
        )
    mo._write(os.path.join(out, "estimators.json"), estimators_fixture(seed=7777))
    print("phase 5c-entropy: wrote estimators.json", file=sys.stderr)

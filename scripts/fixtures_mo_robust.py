"""Phase 5C fixtures: robust multi-objective optimization, MOMF and NSGA-II.

Written to ``<out>/phase5c_robust/`` by ``generate_fixtures.py --phase
5c-robust`` (with BoTorch main) and ``--phase 5c-robust-3392`` (with a BoTorch
that includes meta-pytorch/botorch#3392):

* ``perturbation.json``: ``InputPerturbation`` and ``ChainedInputTransform``
  (fixed and heteroscedastic perturbation sets, bounds, ``indices``,
  multiplicative perturbations, ``Normalize`` before or after the
  perturbation) on points: the transformed points and the gradient of a
  weighted sum of them; ``SingleTaskGP`` models (one and two outputs) and a
  ``ModelListGP`` with these transforms: the joint posterior of a q-batch
  (``q·n_w`` points; with and without observation noise) and the gradient of
  a weighted sum of its mean and covariance with respect to the inputs.
* ``risk_measures.json``: ``CVaR``, ``VaR``, ``WorstCase``, ``Expectation``,
  ``MultiOutputExpectation``, ``IndependentCVaR``, ``IndependentVaR``,
  ``MultiOutputWorstCase``, ``MVaR`` (``expectation``, ``pad_to_n_w``,
  ``filter_dominated``, ``use_counting``) and ``MARS`` (with ``set_baseline_Y``
  from samples, with and without a reference point) on random samples, with
  and without preprocessing functions: values and the gradient of a weighted
  sum of the values (except ``MVaR(expectation=True)``, whose gradient is NaN
  in BoTorch).
* ``acquisition.json``: qEHVI, qLogEHVI, qNEHVI and qLogNEHVI (``cache_root``
  true / false) with risk-measure objectives (``MVaR``, ``IndependentCVaR``,
  ``MultiOutputExpectation``, ``MultiOutputWorstCase``) on models with an
  ``InputPerturbation``, ``prune_inferior_points_multi_objective`` with MVaR,
  ``MARS.set_baseline_Y(model, X_baseline)`` and qLogNEI with ``MARS``:
  values and gradients (t-batches of two) with BoTorch's recorded base
  samples.
* ``momf.json``: ``MOMF`` on models of MOMFBraninCurrin and MOMFPark (the
  objectives and a trust objective): the default affine cost and the
  exponential cost of BoTorch's tutorial, pending points and an outcome
  constraint; values and gradients.
* ``nsga2.json``: the deterministic parts of NSGA-II: Pareto fronts (from
  BoTorch's ``is_non_dominated``), pymoo's crowding distance, SBX and
  polynomial mutation given the uniform random numbers (pymoo's
  ``calc_crowding_distance``, ``cross_sbx`` and ``mut_pm``; a line-by-line
  transcription of pymoo 0.6.2 when pymoo is not installed, see
  ``nsga2_reference``).
* ``risk_measures_3392.json`` (``--phase 5c-robust-3392``): ``MVaR`` with ties
  in the samples (both implementations) and with a preprocessing function
  that changes the number of outputs: the two MVaR fixes of #3392.

Base samples: every draw of BoTorch's samplers is recorded by its dimension
(all samplers of a file have the same seed and number of samples), as in
``fixtures_mo_acquisition.py``. Non-finite numbers are written as the strings
"inf", "-inf" and "nan".
"""

from __future__ import annotations

import json
import math
import os
import sys
from typing import Any

import torch

DT = torch.float64
NUM_SAMPLES = 16
N_W = 4


def _mo() -> Any:
    """The helpers of ``fixtures_mo_acquisition.py``."""
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import fixtures_mo_acquisition

    return fixtures_mo_acquisition


def enc(x: Any) -> Any:
    return _mo().enc(x)


def _write(out: str, name: str, data: dict) -> None:
    path = os.path.join(out, name)
    with open(path, "w") as f:
        json.dump(enc(data), f)
    print(f"phase 5c-robust: wrote {path}", file=sys.stderr)


def grad_of(f: Any, X: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """``f(X)`` and the gradient of its sum."""
    Xg = X.detach().clone().requires_grad_(True)
    v = f(Xg)
    (g,) = torch.autograd.grad(v.sum(), Xg, allow_unused=True)
    return v.detach(), torch.zeros_like(X) if g is None else g


# ---------------------------------------------------------------------------
# Input transforms
# ---------------------------------------------------------------------------


def hetero(X: torch.Tensor) -> torch.Tensor:
    """The heteroscedastic perturbation set of the fixtures:
    ``[0.05 x², -0.08 x]`` per point (``... x n x 2 x k``)."""
    return torch.stack([0.05 * X**2, -0.08 * X], dim=-2)


def perturbation_set(nw: int, k: int, seed: int, scale: float = 0.1) -> list[list[float]]:
    g = torch.Generator().manual_seed(seed)
    return (scale * (torch.rand(nw, k, generator=g, dtype=DT) - 0.5)).tolist()


def bt_transform(spec: list[dict], d: int) -> Any:
    """BoTorch's transform of a chain specification."""
    from botorch.models.transforms.input import (
        ChainedInputTransform,
        InputPerturbation,
        Normalize,
    )

    steps = []
    for s in spec:
        if s["type"] == "normalize":
            b = s.get("bounds")
            steps.append(Normalize(d=d, bounds=None if b is None else torch.tensor(b, dtype=DT)))
        else:
            ps = hetero if s.get("function") == "hetero" else torch.tensor(s["set"], dtype=DT)
            b = s.get("bounds")
            steps.append(
                InputPerturbation(
                    perturbation_set=ps,
                    bounds=None if b is None else torch.tensor(b, dtype=DT),
                    indices=s.get("indices"),
                    multiplicative=s.get("multiplicative", False),
                    transform_on_eval=s.get("eval", True),
                )
            )
    if len(steps) == 1:
        return steps[0]
    return ChainedInputTransform(**{f"tf{i}": t for i, t in enumerate(steps)})


def transform_cases() -> list[tuple[str, list[dict], int]]:
    """(name, chain specification, d)."""
    return [
        ("additive", [{"type": "perturbation", "set": perturbation_set(4, 3, 1)}], 3),
        (
            "bounds",
            [
                {
                    "type": "perturbation",
                    "set": perturbation_set(3, 2, 2, 0.6),
                    "bounds": [[0.0, 0.1], [1.0, 0.8]],
                }
            ],
            2,
        ),
        (
            "multiplicative_indices",
            [
                {
                    "type": "perturbation",
                    "set": [[0.9], [1.1], [1.3]],
                    "indices": [2],
                    "multiplicative": True,
                }
            ],
            3,
        ),
        (
            "hetero_indices_bounds",
            [
                {
                    "type": "perturbation",
                    "function": "hetero",
                    "indices": [1, 0],
                    "bounds": [[0.0, 0.0], [0.62, 0.9]],
                }
            ],
            3,
        ),
        (
            "normalize_then_perturb",
            [
                {"type": "normalize", "bounds": [[-1.0, 0.0], [2.0, 3.0]]},
                {
                    "type": "perturbation",
                    "set": perturbation_set(4, 2, 3, 0.3),
                    "bounds": [[0.0, 0.0], [1.0, 1.0]],
                },
            ],
            2,
        ),
        (
            "perturb_then_normalize",
            [
                {"type": "perturbation", "set": perturbation_set(3, 2, 4, 0.3)},
                {"type": "normalize", "bounds": [[-1.0, 0.0], [2.0, 3.0]]},
            ],
            2,
        ),
        (
            "two_perturbations",
            [
                {"type": "perturbation", "set": perturbation_set(2, 2, 5, 0.2)},
                {"type": "perturbation", "set": [[1.0, 0.95], [1.05, 1.0]], "multiplicative": True},
            ],
            2,
        ),
        (
            "not_in_eval",
            [{"type": "perturbation", "set": perturbation_set(3, 2, 6), "eval": False}],
            2,
        ),
    ]


def gen_transforms() -> list[dict]:
    cases = []
    for name, spec, d in transform_cases():
        tf = bt_transform(spec, d)
        tf.eval()
        g = torch.Generator().manual_seed(len(cases) + 10)
        X = 0.1 + 0.8 * torch.rand(3, d, generator=g, dtype=DT)
        # A point at the upper bound (perturbations beyond it are clamped).
        X[0, 0] = 1.0 if name == "bounds" else X[0, 0]
        out = tf(X)
        bar = torch.randn(out.shape, generator=g, dtype=DT)
        _, grad = grad_of(lambda Z, tf=tf, bar=bar: (tf(Z) * bar).sum(), X)
        cases.append({"name": name, "spec": spec, "x": X, "out": out, "bar": bar, "grad": grad})
    return cases


MODEL_CASES = [
    # (name, kind, member configurations (m, chain), d)
    (
        "single_bounds",
        "single_task",
        [(1, "bounds")],
        2,
    ),
    ("multi_normalize_then_perturb", "single_task", [(2, "normalize_then_perturb")], 2),
    ("multi_perturb_then_normalize", "single_task", [(2, "perturb_then_normalize")], 2),
    ("multi_hetero", "single_task", [(2, "hetero2")], 2),
    ("list_bounds", "model_list", [(2, "bounds"), (1, "bounds")], 2),
]


def model_chain(kind: str, d: int) -> list[dict]:
    """The chains of the models (learned Normalize bounds)."""
    pert = {
        "type": "perturbation",
        "set": perturbation_set(N_W, d, 7, 0.25),
        "bounds": [[0.0] * d, [1.0] * d],
    }
    if kind == "bounds":
        return [pert]
    if kind == "normalize_then_perturb":
        return [{"type": "normalize"}, pert]
    if kind == "perturb_then_normalize":
        return [
            {"type": "perturbation", "set": perturbation_set(N_W, d, 8, 0.25)},
            {"type": "normalize"},
        ]
    if kind == "hetero2":
        return [{"type": "normalize"}, {"type": "perturbation", "function": "hetero"}]
    raise ValueError(kind)


def build_robust_model(case: tuple, seed: int) -> tuple[Any, list[dict], torch.Tensor]:
    """A BoTorch model with InputPerturbation transforms, its specification
    and the first member's training inputs."""
    from botorch.models import ModelListGP, SingleTaskGP

    gf = _mo()._gf()
    name, kind, members, d = case
    subs, specs, xs = [], [], []
    for i, (m, chain_kind) in enumerate(members):
        cfg = {"d": d, "n": 9 + i, "m": m, "x_range": (0.05, 0.95)}
        X, Y, _, _ = gf.p4_make_data(cfg, seed + 17 * i)
        chain = model_chain(chain_kind, d)
        sub = SingleTaskGP(X, Y, input_transform=bt_transform(chain, d))
        n_el = int(torch.Size(gf.p4_aug_shape(cfg)).numel())
        raw = gf.p4_set_values(sub, n_el, d)
        gf.set_flat_params(sub, raw)
        sub.eval()
        subs.append(sub)
        xs.append(X)
        specs.append({"m": m, "chain": chain, "train_x": X, "train_y": Y, "raw_params": raw})
    model = ModelListGP(*subs) if kind == "model_list" else subs[0]
    model.eval()
    return model, specs, xs[0]


def gen_models() -> list[dict]:
    mo = _mo()
    out = []
    for i, case in enumerate(MODEL_CASES):
        model, specs, X0 = build_robust_model(case, seed=100 + 10 * i)
        d = case[3]
        g = torch.Generator().manual_seed(200 + i)
        Xq = 0.1 + 0.8 * torch.rand(2, d, generator=g, dtype=DT)
        with mo.dense_model_list():
            post = model.posterior(Xq)
            mean, cov = post.mean.detach(), post.distribution.covariance_matrix.detach()
            post_n = model.posterior(Xq, observation_noise=True)
            cov_n = post_n.distribution.covariance_matrix.detach()
            bar_mean = torch.randn(mean.shape, generator=g, dtype=DT)
            bar_cov = torch.randn(cov.shape, generator=g, dtype=DT)

            def functional(
                Z: torch.Tensor,
                model: Any = model,
                bar_mean: Any = bar_mean,
                bar_cov: Any = bar_cov,
            ) -> torch.Tensor:
                p = model.posterior(Z)
                return (p.mean * bar_mean).sum() + (
                    p.distribution.covariance_matrix * bar_cov
                ).sum()

            _, grad = grad_of(functional, Xq)
        out.append(
            {
                "name": case[0],
                "kind": case[1],
                "members": specs,
                "x": Xq,
                "mean": mean,
                "covariance": cov,
                "covariance_noisy": cov_n,
                "bar_mean": bar_mean,
                "bar_cov": bar_cov,
                "grad": grad,
            }
        )
    return out


def generate_perturbation(out: str) -> None:
    _write(out, "perturbation.json", {"transforms": gen_transforms(), "models": gen_models()})


# ---------------------------------------------------------------------------
# Risk measures
# ---------------------------------------------------------------------------


def preprocessing(spec: dict | None) -> Any:
    from botorch.acquisition.multi_objective.objective import WeightedMCMultiOutputObjective
    from botorch.acquisition.objective import LinearMCObjective

    if spec is None:
        return None
    if spec["type"] == "linear":
        return LinearMCObjective(torch.tensor(spec["weights"], dtype=DT))
    return WeightedMCMultiOutputObjective(
        torch.tensor(spec["weights"], dtype=DT), outcomes=spec.get("outcomes")
    )


def risk_measure(spec: dict) -> Any:
    from botorch.acquisition import risk_measures as rm
    from botorch.acquisition.multi_objective import multi_output_risk_measures as mrm

    kind = spec["kind"]
    pre = preprocessing(spec.get("preprocessing"))
    n_w = spec["n_w"]
    if kind in ("CVaR", "VaR"):
        return getattr(rm, kind)(alpha=spec["alpha"], n_w=n_w, preprocessing_function=pre)
    if kind in ("WorstCase", "Expectation"):
        return getattr(rm, kind)(n_w=n_w, preprocessing_function=pre)
    if kind in ("IndependentCVaR", "IndependentVaR"):
        return getattr(mrm, kind)(alpha=spec["alpha"], n_w=n_w, preprocessing_function=pre)
    if kind in ("MultiOutputExpectation", "MultiOutputWorstCase"):
        return getattr(mrm, kind)(n_w=n_w, preprocessing_function=pre)
    if kind == "MVaR":
        return mrm.MVaR(
            n_w=n_w,
            alpha=spec["alpha"],
            expectation=spec.get("expectation", False),
            preprocessing_function=pre,
            pad_to_n_w=spec.get("pad_to_n_w", False),
            filter_dominated=spec.get("filter_dominated", True),
            use_counting=spec.get("use_counting", False),
        )
    if kind == "MARS":
        ref = spec.get("ref_point")
        mars = mrm.MARS(
            alpha=spec["alpha"],
            n_w=n_w,
            chebyshev_weights=torch.tensor(spec["weights"], dtype=DT),
            ref_point=None if ref is None else torch.tensor(ref, dtype=DT),
            preprocessing_function=pre,
        )
        mars.set_baseline_Y(
            model=None, X_baseline=None, Y_samples=torch.tensor(spec["y_samples"], dtype=DT)
        )
        return mars
    raise ValueError(kind)


def risk_specs() -> list[tuple[dict, int]]:
    """(specification, number of outputs of the samples)."""
    g = torch.Generator().manual_seed(31)
    specs: list[tuple[dict, int]] = []
    for kind in ("CVaR", "VaR", "WorstCase", "Expectation"):
        for alpha in (0.5, 0.8):
            if kind in ("WorstCase", "Expectation") and alpha != 0.5:
                continue
            specs.append(({"kind": kind, "alpha": alpha, "n_w": 8}, 1))
            specs.append(
                (
                    {
                        "kind": kind,
                        "alpha": alpha,
                        "n_w": 8,
                        "preprocessing": {"type": "linear", "weights": [0.7, -1.2]},
                    },
                    2,
                )
            )
    # 25 * 0.28 = 7.000000000000001: alpha_idx = 7 in floating point.
    specs.append(({"kind": "VaR", "alpha": 0.28, "n_w": 25}, 1))
    specs.append(({"kind": "CVaR", "alpha": 0.28, "n_w": 25}, 1))
    for kind in (
        "MultiOutputExpectation",
        "MultiOutputWorstCase",
        "IndependentCVaR",
        "IndependentVaR",
    ):
        specs.append(({"kind": kind, "alpha": 0.6, "n_w": 8}, 2))
        specs.append(
            (
                {
                    "kind": kind,
                    "alpha": 0.7,
                    "n_w": 8,
                    "preprocessing": {
                        "type": "weighted",
                        "weights": [1.0, -0.5],
                        "outcomes": [2, 0],
                    },
                },
                3,
            )
        )
    for m in (2, 3):
        for alpha in (0.5, 0.75):
            specs.append(({"kind": "MVaR", "alpha": alpha, "n_w": 8}, m))
        specs.append(({"kind": "MVaR", "alpha": 0.6, "n_w": 8, "use_counting": True}, m))
        specs.append(({"kind": "MVaR", "alpha": 0.6, "n_w": 8, "pad_to_n_w": True}, m))
        specs.append(({"kind": "MVaR", "alpha": 0.6, "n_w": 8, "filter_dominated": False}, m))
        specs.append(({"kind": "MVaR", "alpha": 0.6, "n_w": 8, "expectation": True}, m))
    specs.append(
        (
            {
                "kind": "MVaR",
                "alpha": 0.5,
                "n_w": 8,
                "preprocessing": {"type": "weighted", "weights": [1.0, -0.5, 2.0]},
            },
            3,
        )
    )
    for use_counting in (False, True):
        specs.append(({"kind": "MVaR", "alpha": 0.5, "n_w": 4, "use_counting": use_counting}, 3))
    for ref in (None, [0.1, -0.4]):
        y_samples = torch.randn(5 * 8, 2, generator=g, dtype=DT)
        specs.append(
            (
                {
                    "kind": "MARS",
                    "alpha": 0.6,
                    "n_w": 8,
                    "weights": [0.3, 0.7],
                    "ref_point": ref,
                    "y_samples": y_samples.tolist(),
                },
                2,
            )
        )
    y_samples = torch.randn(4 * 8, 3, generator=g, dtype=DT)
    specs.append(
        (
            {
                "kind": "MARS",
                "alpha": 0.5,
                "n_w": 8,
                "weights": [0.6, 0.4],
                "ref_point": None,
                "preprocessing": {"type": "weighted", "weights": [1.0, -1.0], "outcomes": [0, 2]},
                "y_samples": y_samples.tolist(),
            },
            3,
        )
    )
    return specs


def gen_risk_case(spec: dict, m: int, seed: int) -> dict:
    g = torch.Generator().manual_seed(seed)
    S, q = 3, 2
    samples = torch.randn(S, q * spec["n_w"], m, generator=g, dtype=DT)
    obj = risk_measure(spec)
    values = obj(samples).detach()
    case: dict = {"spec": spec, "m": m, "samples": samples, "values": values}
    if spec["kind"] == "MARS":
        case["baseline_y"] = obj.baseline_Y
        case["bounds"] = obj._get_Y_normalization_bounds(obj.baseline_Y, obj.ref_point)
    if not (spec["kind"] == "MVaR" and spec.get("expectation")):
        bar = torch.randn(values.shape, generator=g, dtype=DT)
        Sg = samples.clone().requires_grad_(True)
        v = obj(Sg)
        (grad,) = torch.autograd.grad((v * bar).sum(), Sg)
        case["bar"] = bar
        case["values_grad"] = v.detach()
        case["grad"] = grad
    return case


def generate_risk_measures(out: str) -> None:
    cases = [gen_risk_case(spec, m, 1000 + i) for i, (spec, m) in enumerate(risk_specs())]
    _write(out, "risk_measures.json", {"cases": cases})


def generate_risk_measures_3392(out: str) -> None:
    """MVaR with ties and with a preprocessing function that changes the
    number of outputs (the fixes of #3392)."""
    from botorch.acquisition.multi_objective import multi_output_risk_measures as mrm

    if not hasattr(mrm, "_is_maximal_alpha_level_point"):
        raise RuntimeError("phase 5c-robust-3392 needs a BoTorch with meta-pytorch/botorch#3392")
    cases = []
    g = torch.Generator().manual_seed(77)
    for i in range(12):
        m = 2 + i % 2
        n_w = 8
        # Integer-valued samples: ties in every output.
        samples = torch.randint(0, 4, (2, 2 * n_w, m), generator=g).to(DT)
        for use_counting in (False, True):
            alpha = (0.25, 0.5, 0.75)[i % 3]
            spec = {"kind": "MVaR", "alpha": alpha, "n_w": n_w, "use_counting": use_counting}
            obj = risk_measure(spec)
            cases.append({"spec": spec, "m": m, "samples": samples, "values": obj(samples)})
    # A preprocessing function with fewer outputs (m = 3 -> m' = 2).
    samples = torch.randn(3, 2 * 8, 3, generator=g, dtype=DT)
    for expectation in (False, True):
        spec = {
            "kind": "MVaR",
            "alpha": 0.5,
            "n_w": 8,
            "expectation": expectation,
            "preprocessing": {"type": "weighted", "weights": [1.0, -1.0], "outcomes": [0, 2]},
        }
        cases.append(
            {"spec": spec, "m": 3, "samples": samples, "values": risk_measure(spec)(samples)}
        )
    _write(out, "risk_measures_3392.json", {"cases": cases})


# ---------------------------------------------------------------------------
# Acquisition functions with risk measures
# ---------------------------------------------------------------------------


def per_candidate(acqf: Any, X: torch.Tensor, grads: bool = True) -> tuple[Any, Any]:
    """Values (and gradients) of every candidate set of the t-batch ``X``
    evaluated on its own. MVaR pads the MVaR sets to the largest set of the
    whole batch, so that BoTorch's (log) values of a candidate set depend on
    the other candidate sets of a t-batch; botorch_rs evaluates one candidate
    set at a time."""
    vals, grs = [], []
    for i in range(X.shape[0]):
        if grads:
            v, g = _mo().values_and_grads(acqf, X[i : i + 1])
            grs.append(g)
        else:
            with torch.no_grad():
                v = acqf(X[i : i + 1])
        vals.append(v)
    return torch.cat(vals), (torch.cat(grs) if grads else None)


# MVaR returns a set of k' points per candidate, so qEHVI's inclusion-exclusion
# runs over 2^(q k') subsets (as in BoTorch): its cases use q = 1, or q = 2 with
# alpha = 0.75 (at most two points per set for n_w = 4 and two objectives).
ACQ_CASES = [
    # (acqf, objective specification, q, cache_root)
    ("qNEHVI", {"kind": "MVaR", "alpha": 0.5}, 1, True),
    ("qLogNEHVI", {"kind": "MVaR", "alpha": 0.75}, 1, True),
    ("qNEHVI", {"kind": "IndependentCVaR", "alpha": 0.5}, 2, False),
    ("qLogNEHVI", {"kind": "MultiOutputExpectation"}, 1, False),
    ("qLogNEHVI", {"kind": "MultiOutputWorstCase"}, 2, True),
    ("qEHVI", {"kind": "MVaR", "alpha": 0.5}, 1, None),
    ("qLogEHVI", {"kind": "MVaR", "alpha": 0.5, "use_counting": True}, 1, None),
    ("qLogEHVI", {"kind": "IndependentVaR", "alpha": 0.5}, 2, None),
    ("qEHVI", {"kind": "MVaR", "alpha": 0.75, "expectation": True}, 2, None),
]


def gen_acquisition_for(model_case: tuple, seed: int) -> dict:
    from botorch.acquisition.logei import qLogNoisyExpectedImprovement
    from botorch.acquisition.multi_objective import logei as bt_logei
    from botorch.acquisition.multi_objective import monte_carlo as bt_mc
    from botorch.acquisition.multi_objective.multi_output_risk_measures import MARS
    from botorch.acquisition.multi_objective.utils import prune_inferior_points_multi_objective
    from botorch.sampling.normal import SobolQMCNormalSampler
    from botorch.utils.multi_objective.box_decompositions.non_dominated import (
        FastNondominatedPartitioning,
    )

    mo = _mo()
    model, specs, X0 = build_robust_model(model_case, seed)
    d = model_case[3]
    record: dict[tuple[int, int], torch.Tensor] = {}
    g = torch.Generator().manual_seed(seed + 1)
    Xb = X0[:5]
    cases = []
    with mo.recorded_draws(record), mo.dense_model_list():
        means = model.posterior(Xb).mean.detach()  # (5·n_w) x m
        ref = means.min(0).values - 0.2 * (means.max(0).values - means.min(0).values)
        cases_here = list(ACQ_CASES)
        if means.shape[-1] == 2:
            cases_here.append(("qLogNEHVI", {"kind": "MVaR", "alpha": 0.75}, 2, False))
        for name, ospec, q, cache_root in cases_here:
            print(f"  {model_case[0]}: {name} {ospec} q={q}", file=sys.stderr, flush=True)
            spec = dict(ospec, n_w=N_W)
            obj = risk_measure(spec)
            sampler = SobolQMCNormalSampler(sample_shape=torch.Size([NUM_SAMPLES]), seed=0)
            x = 0.1 + 0.8 * torch.rand(2, q, d, generator=g, dtype=DT)
            entry: dict = {"acqf": name, "objective": spec, "q": q, "ref_point": ref, "x": x}
            floor_of = None
            if name in ("qNEHVI", "qLogNEHVI"):
                cls = (
                    bt_mc.qNoisyExpectedHypervolumeImprovement
                    if name == "qNEHVI"
                    else bt_logei.qLogNoisyExpectedHypervolumeImprovement
                )

                def build(
                    mdl: Any, cls: Any = cls, obj: Any = obj, cache_root: Any = cache_root
                ) -> Any:
                    return cls(
                        model=mdl,
                        ref_point=ref,
                        X_baseline=Xb,
                        sampler=SobolQMCNormalSampler(
                            sample_shape=torch.Size([NUM_SAMPLES]), seed=0
                        ),
                        objective=obj,
                        prune_baseline=False,
                        cache_root=cache_root,
                    )

                acqf = build(model)
                floor_of = build
                entry["cache_root"] = cache_root
                entry["x_baseline"] = Xb
                entry["num_cells"] = int(acqf.cell_lower_bounds.shape[-2])
            else:
                # The front: the objective of the baseline means.
                with torch.no_grad():
                    front = obj(means.unsqueeze(0)).squeeze(0)
                if front.dim() == 1:
                    front = front.unsqueeze(0)
                part = FastNondominatedPartitioning(ref_point=ref, Y=front)
                cls = (
                    bt_mc.qExpectedHypervolumeImprovement
                    if name == "qEHVI"
                    else bt_logei.qLogExpectedHypervolumeImprovement
                )
                acqf = cls(
                    model=model, ref_point=ref, partitioning=part, sampler=sampler, objective=obj
                )
                entry["front"] = front
            if spec.get("expectation"):
                # BoTorch's MVaR(expectation=True) returns NaN values of the
                # wrong shape when the samples require gradients (and qEHVI
                # then runs over 2^(q * batch) subsets): values only.
                entry["values"], _ = per_candidate(acqf, x, grads=False)
            else:
                v, gr = per_candidate(acqf, x)
                entry["values"] = v
                entry["grads"] = gr
                if floor_of is not None:
                    # BoTorch's own change under 1e-14 perturbations of the
                    # training inputs (the joint posteriors of the perturbed
                    # baseline are nearly singular).
                    entry["floor"] = mo.floors(
                        lambda mdl, fo=floor_of, x=x: per_candidate(fo(mdl), x),
                        model,
                        v,
                        gr,
                        name.startswith("qLog"),
                    )
            cases.append(entry)
        # Pruning with MVaR (a seeded sampler of 64 samples).
        calls: list[int] = []
        prune_record: dict[tuple[int, int], torch.Tensor] = {}
        with mo.recorded_draws(prune_record), mo.seeded_pruning(calls):
            obj = risk_measure({"kind": "MVaR", "alpha": 0.5, "n_w": N_W})
            kept = prune_inferior_points_multi_objective(
                model=model, X=X0, ref_point=ref, objective=obj
            )
        idx = [int((X0 == k).all(-1).nonzero()[0]) for k in kept]
        prune = {
            "x": X0,
            "kept": idx,
            "base_samples": [
                {"d": dd, "z": enc(z.reshape(-1))} for (_, dd), z in sorted(prune_record.items())
            ],
        }
        # MARS: baseline from the model, and qLogNEI with MARS.
        weights = [0.4, 0.6] if means.shape[-1] == 2 else [0.3, 0.3, 0.4]
        mars = MARS(alpha=0.5, n_w=N_W, chebyshev_weights=torch.tensor(weights, dtype=DT))
        mars.set_baseline_Y(model=model, X_baseline=Xb)
        mars_case: dict = {
            "weights": weights,
            "baseline_y": mars.baseline_Y,
            "bounds": mars._get_Y_normalization_bounds(mars.baseline_Y),
        }
        nei_record: dict[tuple[int, int], torch.Tensor] = {}
        with mo.recorded_draws(nei_record), mo.no_lower_bound():
            sampler = SobolQMCNormalSampler(sample_shape=torch.Size([NUM_SAMPLES]), seed=0)

            def build_nei(mdl: Any) -> Any:
                return qLogNoisyExpectedImprovement(
                    model=mdl,
                    X_baseline=Xb,
                    sampler=SobolQMCNormalSampler(sample_shape=torch.Size([NUM_SAMPLES]), seed=0),
                    objective=mars,
                    prune_baseline=False,
                )

            acqf = build_nei(model)
            x = 0.1 + 0.8 * torch.rand(2, 2, d, generator=g, dtype=DT)
            v, gr = per_candidate(acqf, x)
            mars_case["floor"] = mo.floors(
                lambda mdl: per_candidate(build_nei(mdl), x), model, v, gr, True
            )
        mars_case.update(
            {
                "x": x,
                "values": v,
                "grads": gr,
                "draws": [
                    {"d": dd, "z": enc(z.reshape(-1))} for (_, dd), z in sorted(nei_record.items())
                ],
            }
        )
    return {
        "name": model_case[0],
        "kind": model_case[1],
        "members": specs,
        "num_samples": NUM_SAMPLES,
        "draws": mo.draws_of(record, NUM_SAMPLES),
        "cases": cases,
        "prune": prune,
        "mars": mars_case,
    }


def generate_acquisition(out: str) -> None:
    models = [MODEL_CASES[1], MODEL_CASES[4]]
    data = {"models": [gen_acquisition_for(c, 300 + 7 * i) for i, c in enumerate(models)]}
    _write(out, "acquisition.json", data)


# ---------------------------------------------------------------------------
# MOMF
# ---------------------------------------------------------------------------


def exp_cost(X: torch.Tensor) -> torch.Tensor:
    """The cost of BoTorch's MOMF tutorial: ``exp(4.8 s)``."""
    return torch.exp(torch.tensor(4.8, dtype=X.dtype) * X[..., -1:])


def gen_momf_for(problem_name: str, seed: int) -> dict:
    from botorch.acquisition.multi_objective.multi_fidelity import MOMF
    from botorch.models import SingleTaskGP
    from botorch.sampling.normal import SobolQMCNormalSampler
    from botorch.test_functions import multi_objective_multi_fidelity as momf_tf
    from botorch.utils.multi_objective.box_decompositions.non_dominated import (
        NondominatedPartitioning,
    )

    mo = _mo()
    gf = mo._gf()
    problem = getattr(momf_tf, problem_name)(negate=True)
    d = problem.dim
    g = torch.Generator().manual_seed(seed)
    X = torch.rand(10, d, generator=g, dtype=DT)
    Y = torch.cat([problem(X), X[..., -1:]], dim=-1)  # objectives + trust objective
    model = SingleTaskGP(X, Y)
    raw = gf.p4_set_values(model, 3, d)
    gf.set_flat_params(model, raw)
    model.eval()
    ref = (
        torch.zeros(3, dtype=DT) + torch.tensor([Y[:, 0].min(), Y[:, 1].min(), 0.0], dtype=DT) - 0.1
    )
    record: dict[tuple[int, int], torch.Tensor] = {}
    cases = []
    with mo.recorded_draws(record):
        part = NondominatedPartitioning(ref_point=ref, Y=Y)
        cells = part.get_hypercell_bounds()
        for variant in ("default", "exp_cost", "pending", "constraint"):
            sampler = SobolQMCNormalSampler(sample_shape=torch.Size([NUM_SAMPLES]), seed=0)
            kwargs: dict = {}
            if variant == "exp_cost":
                kwargs["cost_call"] = exp_cost
            if variant == "pending":
                kwargs["X_pending"] = torch.rand(1, d, generator=g, dtype=DT)
            if variant == "constraint":
                kwargs["constraints"] = [lambda Z: Z[..., 0] - Y[:, 0].mean()]
                kwargs["eta"] = torch.tensor([0.1], dtype=DT)
            acqf = MOMF(model=model, ref_point=ref, partitioning=part, sampler=sampler, **kwargs)
            x = torch.rand(2, 2, d, generator=g, dtype=DT)
            v, gr = mo.values_and_grads(acqf, x)
            cases.append(
                {
                    "variant": variant,
                    "x": x,
                    "x_pending": kwargs.get("X_pending"),
                    "constraint_offset": Y[:, 0].mean().item(),
                    "values": v,
                    "grads": gr,
                }
            )
    return {
        "problem": problem_name,
        "train_x": X,
        "train_y": Y,
        "raw_params": raw,
        "ref_point": ref,
        "cell_lower": cells[0],
        "cell_upper": cells[1],
        "num_samples": NUM_SAMPLES,
        "draws": mo.draws_of(record, NUM_SAMPLES),
        "cases": cases,
    }


def generate_momf(out: str) -> None:
    data = {
        "problems": [
            gen_momf_for(p, 500 + i) for i, p in enumerate(("MOMFBraninCurrin", "MOMFPark"))
        ]
    }
    _write(out, "momf.json", data)


# ---------------------------------------------------------------------------
# NSGA-II
# ---------------------------------------------------------------------------


def nsga2_reference() -> dict[str, Any]:
    """pymoo's crowding distance, SBX and polynomial mutation: pymoo itself
    if installed, otherwise a line-by-line transcription of pymoo 0.6.2
    (``calc_crowding_distance``, ``cross_sbx``, ``mut_pm``)."""
    import numpy as np

    try:
        from pymoo.operators.crossover.sbx import cross_sbx
        from pymoo.operators.mutation.pm import mut_pm
        from pymoo.operators.survival.rank_and_crowding.metrics import calc_crowding_distance

        return {"source": "pymoo", "cd": calc_crowding_distance, "sbx": cross_sbx, "pm": mut_pm}
    except ImportError:
        pass

    def calc_crowding_distance(F: Any, **kwargs: Any) -> Any:
        n_points, n_obj = F.shape
        I = np.argsort(F, axis=0, kind="mergesort")  # noqa: E741
        F = F[I, np.arange(n_obj)]
        dist = np.vstack([F, np.full(n_obj, np.inf)]) - np.vstack([np.full(n_obj, -np.inf), F])
        norm = np.max(F, axis=0) - np.min(F, axis=0)
        norm[norm == 0] = np.nan
        dist_to_last, dist_to_next = dist, np.copy(dist)
        dist_to_last, dist_to_next = dist_to_last[:-1] / norm, dist_to_next[1:] / norm
        dist_to_last[np.isnan(dist_to_last)] = 0.0
        dist_to_next[np.isnan(dist_to_next)] = 0.0
        J = np.argsort(I, axis=0)
        return (
            np.sum(dist_to_last[J, np.arange(n_obj)] + dist_to_next[J, np.arange(n_obj)], axis=1)
            / n_obj
        )

    def cross_sbx(
        X: Any,
        xl: Any,
        xu: Any,
        eta: Any,
        prob_var: Any,
        prob_bin: Any,
        eps: float = 1.0e-14,
        random_state: Any = None,
    ) -> Any:
        n_parents, n_matings, n_var = X.shape
        cross = random_state.random((n_matings, n_var)) < prob_var
        too_close = np.abs(X[0] - X[1]) <= eps
        cross[too_close] = False
        cross[:, xl == xu] = False
        y1 = np.min(X, axis=0)[cross]
        y2 = np.max(X, axis=0)[cross]
        _xl = np.repeat(xl[None, :], n_matings, axis=0)[cross]
        _xu = np.repeat(xu[None, :], n_matings, axis=0)[cross]
        eta = eta.repeat(n_var, axis=1)[cross]
        prob_bin = prob_bin.repeat(n_var, axis=1)[cross]
        rand = random_state.random(len(eta))

        def calc_betaq(beta: Any) -> Any:
            alpha = 2.0 - np.power(beta, -(eta + 1.0))
            mask, mask_not = (rand <= (1.0 / alpha)), (rand > (1.0 / alpha))
            betaq = np.zeros(mask.shape)
            betaq[mask] = np.power((rand * alpha), (1.0 / (eta + 1.0)))[mask]
            betaq[mask_not] = np.power((1.0 / (2.0 - rand * alpha)), (1.0 / (eta + 1.0)))[mask_not]
            return betaq

        delta = y2 - y1
        beta = 1.0 + (2.0 * (y1 - _xl) / delta)
        betaq = calc_betaq(beta)
        c1 = 0.5 * ((y1 + y2) - betaq * delta)
        beta = 1.0 + (2.0 * (_xu - y2) / delta)
        betaq = calc_betaq(beta)
        c2 = 0.5 * ((y1 + y2) + betaq * delta)
        b = (random_state.random(len(prob_bin)) < prob_bin) ^ (X[0, cross] > X[1, cross])
        tmp = np.copy(c1[b])
        c1[b] = c2[b]
        c2[b] = tmp
        Q = np.copy(X)
        Q[0, cross] = c1
        Q[1, cross] = c2
        for k in (0, 1):
            Q[k] = np.where(Q[k] < xl, xl, Q[k])
            Q[k] = np.where(Q[k] > xu, xu, Q[k])
        return Q

    def mut_pm(
        X: Any,
        xl: Any,
        xu: Any,
        eta: Any,
        prob: Any,
        at_least_once: bool = False,
        random_state: Any = None,
    ) -> Any:
        n, n_var = X.shape
        Xp = np.full(X.shape, np.inf)
        mut = random_state.random((n, n_var)) < (np.ones(n) * prob)[:, None]
        mut[:, xl == xu] = False
        Xp[:, :] = X
        _xl = np.repeat(xl[None, :], X.shape[0], axis=0)[mut]
        _xu = np.repeat(xu[None, :], X.shape[0], axis=0)[mut]
        X = X[mut]
        eta = np.tile(eta[:, None], (1, n_var))[mut]
        delta1 = (X - _xl) / (_xu - _xl)
        delta2 = (_xu - X) / (_xu - _xl)
        mut_pow = 1.0 / (eta + 1.0)
        rand = random_state.random(X.shape)
        mask = rand <= 0.5
        mask_not = np.logical_not(mask)
        deltaq = np.zeros(X.shape)
        xy = 1.0 - delta1
        val = 2.0 * rand + (1.0 - 2.0 * rand) * (np.power(xy, (eta + 1.0)))
        d = np.power(val, mut_pow) - 1.0
        deltaq[mask] = d[mask]
        xy = 1.0 - delta2
        val = 2.0 * (1.0 - rand) + 2.0 * (rand - 0.5) * (np.power(xy, (eta + 1.0)))
        d = 1.0 - (np.power(val, mut_pow))
        deltaq[mask_not] = d[mask_not]
        _Y = X + deltaq * (_xu - _xl)
        _Y[_Y < _xl] = _xl[_Y < _xl]
        _Y[_Y > _xu] = _xu[_Y > _xu]
        Xp[mut] = _Y
        Xp = np.where(Xp < xl, xl, Xp)
        Xp = np.where(Xp > xu, xu, Xp)
        return Xp

    return {
        "source": "transcription of pymoo 0.6.2",
        "cd": calc_crowding_distance,
        "sbx": cross_sbx,
        "pm": mut_pm,
    }


class RecordingRandom:
    """A random state that returns recorded uniform numbers (``random(size)``)
    in the order pymoo draws them."""

    def __init__(self, seed: int) -> None:
        import numpy as np

        self.rng = np.random.default_rng(seed)
        self.draws: list[Any] = []

    def random(self, size: Any = None) -> Any:
        u = self.rng.random(size)
        self.draws.append(u)
        return u


def gen_nsga2() -> dict:
    import numpy as np
    from botorch.utils.multi_objective.pareto import is_non_dominated

    ref = nsga2_reference()
    g = torch.Generator().manual_seed(9)
    fronts_cases = []
    for n, m in ((30, 2), (40, 3), (12, 2)):
        F = torch.randint(0, 6, (n, m), generator=g).to(DT) + 0.1 * torch.rand(
            n, m, generator=g, dtype=DT
        )
        if n == 12:
            F = torch.randint(0, 3, (n, m), generator=g).to(DT)  # ties and duplicates
        remaining = list(range(n))
        fronts = []
        while remaining:
            sub = F[remaining]
            mask = is_non_dominated(-sub, deduplicate=False)
            front = [remaining[i] for i in range(len(remaining)) if mask[i]]
            fronts.append(front)
            remaining = [i for i in remaining if i not in front]
        cds = [
            ref["cd"](F[f].numpy()).tolist() if len(f) > 2 else [math.inf] * len(f) for f in fronts
        ]
        fronts_cases.append({"f": F, "fronts": fronts, "crowding": cds})
    sbx_cases = []
    for i, (n_matings, n_var) in enumerate(((5, 3), (8, 6))):
        rs = RecordingRandom(100 + i)
        rng = np.random.default_rng(200 + i)
        xl, xu = np.zeros(n_var), np.ones(n_var)
        xu[-1] = 2.0
        if n_var > 3:
            xl[1] = xu[1] = 0.5  # a fixed variable
        X = xl + (xu - xl) * rng.random((2, n_matings, n_var))
        X[1, 0, 0] = X[0, 0, 0]  # parents too close
        eta = np.full((n_matings, 1), 15.0)
        prob_var = np.full((n_matings, 1), 0.5)
        prob_bin = np.full((n_matings, 1), 0.5)
        Q = ref["sbx"](X, xl, xu, eta, prob_var, prob_bin, random_state=rs)
        sbx_cases.append(
            {
                "a": X[0].tolist(),
                "b": X[1].tolist(),
                "xl": xl.tolist(),
                "xu": xu.tolist(),
                "u_var": rs.draws[0].tolist(),
                "u_beta": rs.draws[1].tolist(),
                "u_exchange": rs.draws[2].tolist(),
                "c1": Q[0].tolist(),
                "c2": Q[1].tolist(),
            }
        )
    pm_cases = []
    for i, (n, n_var) in enumerate(((6, 3), (10, 5))):
        rs = RecordingRandom(300 + i)
        rng = np.random.default_rng(400 + i)
        xl, xu = np.zeros(n_var), np.ones(n_var)
        xl[0] = -1.0
        X = xl + (xu - xl) * rng.random((n, n_var))
        X[0, 1] = xu[1]  # on the bound
        prob_var = 0.6
        Xp = ref["pm"](
            X, xl, xu, np.full(n, 20.0), np.full(n, prob_var), at_least_once=False, random_state=rs
        )
        pm_cases.append(
            {
                "x": X.tolist(),
                "xl": xl.tolist(),
                "xu": xu.tolist(),
                "prob_var": prob_var,
                "u_mask": rs.draws[0].tolist(),
                "u": rs.draws[1].tolist(),
                "out": Xp.tolist(),
            }
        )
    return {"source": ref["source"], "fronts": fronts_cases, "sbx": sbx_cases, "pm": pm_cases}


def generate_nsga2(out: str) -> None:
    _write(out, "nsga2.json", gen_nsga2())


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------


def _out_dir(out: str) -> str:
    d = os.path.join(out, "phase5c_robust")
    os.makedirs(d, exist_ok=True)
    return d


def generate_phase5c_robust(out: str) -> None:
    torch.manual_seed(0)
    d = _out_dir(out)
    generate_perturbation(d)
    generate_risk_measures(d)
    generate_acquisition(d)
    generate_momf(d)
    generate_nsga2(d)


def generate_phase5c_robust_3392(out: str) -> None:
    generate_risk_measures_3392(_out_dir(out))

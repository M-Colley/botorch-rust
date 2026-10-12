"""Phase 5A-TF fixtures: BoTorch's multi-objective test problems.

Written to ``<out>/phase5a_tf/`` by ``generate_fixtures.py --phase 5a-tf``:

* ``mo_test_functions.json``: for every problem configuration (all classes of
  ``botorch.test_functions.multi_objective`` and
  ``multi_objective_multi_fidelity``, several dimensions and numbers of
  objectives), the bounds, reference points, ``max_hv``, and the values of
  ``evaluate_true``, of ``forward(X, noise=False)`` with ``negate=True`` and of
  ``evaluate_slack_true`` at random points and at points on the boundary of
  the domain (corners); for Penicillin also at points whose simulation runs
  long. The deterministic Pareto fronts (ZDT) are included as well.
* ``sampling.json``: BoTorch's ``sample_simplex`` and ``sample_hypersphere``
  (``qmc=True``) together with the Sobol and normal draws they transform, which
  are the generators of the DTLZ Pareto fronts.

botorch_rs implements the corrected definitions of meta-pytorch/botorch#3402
and #3405 (``botorch_pyo3/tests/mo_reference.py``). For those problems the
fixture holds the corrected values (``values`` etc., from the replicas of the
fixes unless the imported BoTorch contains them) and BoTorch's own values
(``botorch``), so that the Rust tests check both the agreement with the
corrected definition and that the deviation is real.
"""

from __future__ import annotations

import importlib.util
import itertools
import json
import os
import subprocess
import sys
from typing import Any

import torch

DT = torch.float64
_HERE = os.path.dirname(os.path.abspath(__file__))
_REFERENCE = os.path.join(_HERE, "..", "botorch_pyo3", "tests", "mo_reference.py")


def _load_reference() -> Any:
    spec = importlib.util.spec_from_file_location("mo_reference", os.path.normpath(_REFERENCE))
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["mo_reference"] = module
    spec.loader.exec_module(module)
    return module


# (class name, constructor arguments)
CASES: list[tuple[str, dict[str, int]]] = [
    ("BraninCurrin", {}),
    ("DH1", {"dim": 2}),
    ("DH1", {"dim": 5}),
    ("DH2", {"dim": 3}),
    ("DH3", {"dim": 3}),
    ("DH3", {"dim": 6}),
    ("DH4", {"dim": 3}),
    ("DH4", {"dim": 5}),
    ("DTLZ1", {"dim": 3, "num_objectives": 2}),
    ("DTLZ1", {"dim": 6, "num_objectives": 3}),
    ("DTLZ1", {"dim": 7, "num_objectives": 4}),
    ("DTLZ2", {"dim": 2, "num_objectives": 1}),
    ("DTLZ2", {"dim": 3, "num_objectives": 2}),
    ("DTLZ2", {"dim": 6, "num_objectives": 3}),
    ("DTLZ2", {"dim": 9, "num_objectives": 5}),
    ("DTLZ3", {"dim": 5, "num_objectives": 2}),
    ("DTLZ3", {"dim": 6, "num_objectives": 3}),
    ("DTLZ4", {"dim": 4, "num_objectives": 2}),
    ("DTLZ4", {"dim": 6, "num_objectives": 3}),
    ("DTLZ5", {"dim": 4, "num_objectives": 2}),
    ("DTLZ5", {"dim": 6, "num_objectives": 3}),
    ("DTLZ5", {"dim": 7, "num_objectives": 4}),
    ("DTLZ7", {"dim": 4, "num_objectives": 2}),
    ("DTLZ7", {"dim": 6, "num_objectives": 3}),
    ("DTLZ7", {"dim": 8, "num_objectives": 4}),
    ("GMM", {"num_objectives": 2}),
    ("GMM", {"num_objectives": 3}),
    ("GMM", {"num_objectives": 4}),
    ("Penicillin", {}),
    ("ToyRobust", {}),
    ("VehicleSafety", {}),
    ("ZDT1", {"dim": 2}),
    ("ZDT1", {"dim": 5}),
    ("ZDT2", {"dim": 3}),
    ("ZDT3", {"dim": 2}),
    ("ZDT3", {"dim": 6}),
    ("CarSideImpact", {}),
    ("BNH", {}),
    ("CONSTR", {}),
    ("ConstrainedBraninCurrin", {}),
    ("C2DTLZ2", {"dim": 3, "num_objectives": 2}),
    ("C2DTLZ2", {"dim": 6, "num_objectives": 3}),
    ("C2DTLZ2", {"dim": 7, "num_objectives": 4}),
    ("DiscBrake", {}),
    ("MW7", {"dim": 2}),
    ("MW7", {"dim": 5}),
    ("OSY", {}),
    ("SRN", {}),
    ("WeldedBeam", {}),
    ("MOMFBraninCurrin", {}),
    ("MOMFPark", {}),
]

#: Numbers of points of the deterministic Pareto fronts.
FRONT_SIZES = [1, 2, 4, 7, 11, 23]


def tolist(t: torch.Tensor) -> Any:
    return t.detach().to(DT).cpu().tolist()


def boundary_points(bounds: torch.Tensor, g: torch.Generator, n_max: int = 16) -> torch.Tensor:
    """All corners of the box if there are at most ``n_max``, else the lower
    and upper corners and random corners."""
    lo, hi = bounds[0].to(DT), bounds[1].to(DT)
    d = lo.numel()
    if 2**d <= n_max:
        pattern = torch.tensor(list(itertools.product([0, 1], repeat=d)), dtype=torch.bool)
    else:
        rand = torch.rand(n_max - 2, d, generator=g) < 0.5
        pattern = torch.cat(
            [torch.zeros(1, d, dtype=torch.bool), torch.ones(1, d, dtype=torch.bool), rand]
        )
    return torch.where(pattern, hi, lo)


def long_penicillin_runs(problem: Any, g: torch.Generator, n: int = 8) -> torch.Tensor:
    """Points whose Penicillin simulation runs longest (among 4096 random ones),
    where rounding errors accumulate most."""
    lo, hi = problem.bounds[0].to(DT), problem.bounds[1].to(DT)
    X = lo + (hi - lo) * torch.rand(4096, problem.dim, generator=g, dtype=DT)
    t = problem.evaluate_true(X)[:, 2]
    return X[torch.argsort(t, descending=True)[:n]]


def _finite(name: str, what: str, t: torch.Tensor) -> None:
    if not bool(torch.isfinite(t).all()):
        raise RuntimeError(f"{name}: non-finite {what} (JSON cannot represent them)")


def _max_hv(problem: Any) -> float | None:
    try:
        return float(problem.max_hv)
    except NotImplementedError:
        return None


def _front(problem: Any, n: int) -> Any:
    try:
        return tolist(problem.gen_pareto_front(n))
    except NotImplementedError:
        return None


def case_fixture(ref: Any, name: str, kwargs: dict[str, int], seed: int) -> dict:
    g = torch.Generator().manual_seed(seed)
    corrected = ref.make(name, kwargs)
    botorch = ref.make(name, kwargs, corrected=False)
    negated = ref.make(name, {**kwargs, "negate": True})
    lo, hi = corrected.bounds[0].to(DT), corrected.bounds[1].to(DT)
    X = torch.cat(
        [
            lo + (hi - lo) * torch.rand(16, corrected.dim, generator=g, dtype=DT),
            boundary_points(corrected.bounds, g),
        ]
    )
    if name == "Penicillin":
        X = torch.cat([X, long_penicillin_runs(corrected, g)])
    values = corrected.evaluate_true(X)
    _finite(name, "values", values)
    entry: dict[str, Any] = {
        "name": name,
        "kwargs": kwargs,
        "reference": ref.reference_source(name),
        "deviation": None,
        "dim": corrected.dim,
        "num_objectives": corrected.num_objectives,
        "num_constraints": getattr(corrected, "num_constraints", 0),
        "bounds": tolist(corrected.bounds),
        "raw_ref_point": [float(v) for v in corrected._ref_point],
        "ref_point": tolist(corrected.ref_point),
        "ref_point_negated": tolist(negated.ref_point),
        "max_hv": _max_hv(corrected),
        "x": tolist(X),
        "values": tolist(values),
        "values_negated": tolist(negated(X, noise=False)),
        "slack": None,
        "pareto_fronts": None,
    }
    if hasattr(corrected, "evaluate_slack_true"):
        slack = corrected.evaluate_slack_true(X)
        _finite(name, "slack", slack)
        entry["slack"] = tolist(slack)
    fronts = {str(n): _front(corrected, n) for n in FRONT_SIZES}
    if name.startswith("ZDT") and all(v is not None for v in fronts.values()):
        entry["pareto_fronts"] = fronts
        entry["pareto_fronts_negated"] = {str(n): _front(negated, n) for n in FRONT_SIZES}
    if name in ref.DEVIATIONS:
        dev = ref.DEVIATIONS[name]
        entry["deviation"] = f"{dev.pr}: {dev.reason}"
        bt: dict[str, Any] = {"max_hv": _max_hv(botorch)}
        bt["values"] = tolist(botorch.evaluate_true(X))
        if entry["slack"] is not None:
            try:
                bt["slack"] = tolist(botorch.evaluate_slack_true(X))
            except NotImplementedError:  # e.g. BoTorch's C2DTLZ2 with batched inputs
                bt["slack"] = None
        entry["botorch"] = bt
    return entry


def sampling_fixture() -> dict:
    """``sample_simplex`` / ``sample_hypersphere`` with ``qmc=True`` and the
    quasi-random draws they transform (same seeds)."""
    from botorch.utils.sampling import draw_sobol_normal_samples, sample_hypersphere, sample_simplex
    from torch.quasirandom import SobolEngine

    simplex, sphere = [], []
    for d, n, seed in [(2, 8, 0), (3, 16, 1), (4, 9, 2), (6, 5, 3)]:
        u = SobolEngine(d - 1, scramble=True, seed=seed).draw(n, dtype=DT)
        s = sample_simplex(d=d, n=n, qmc=True, seed=seed, dtype=DT)
        simplex.append({"d": d, "uniform": tolist(u), "samples": tolist(s)})
        z = draw_sobol_normal_samples(d=d, n=n, seed=seed, dtype=DT)
        h = sample_hypersphere(d=d, n=n, qmc=True, seed=seed, dtype=DT)
        sphere.append({"d": d, "normal": tolist(z), "samples": tolist(h)})
    return {"sample_simplex": simplex, "sample_hypersphere": sphere}


def botorch_revision() -> str | None:
    """Git revision of the imported BoTorch checkout (None for installed wheels)."""
    import botorch

    try:
        root = os.path.dirname(os.path.dirname(os.path.abspath(botorch.__file__)))
        cmd = ["git", "-C", root, "rev-parse", "HEAD"]
        return subprocess.run(cmd, capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def generate_phase5a_tf(out: str) -> None:
    """Writes the phase 5A-TF fixtures to ``<out>/phase5a_tf``."""
    import botorch

    ref = _load_reference()
    out = os.path.join(out, "phase5a_tf")
    os.makedirs(out, exist_ok=True)
    assert len({name for name, _ in CASES}) == len(ref.MULTI_OBJECTIVE) + len(ref.MULTI_FIDELITY)
    entries = []
    for i, (name, kwargs) in enumerate(CASES):
        entry = case_fixture(ref, name, kwargs, seed=5000 + i)
        entries.append(entry)
        print(
            f"{name:24s} {str(kwargs):38s} reference={entry['reference']}"
            + (f"  [deviation {entry['deviation'][:24]}...]" if entry["deviation"] else ""),
            file=sys.stderr,
        )
    fixture = {
        "versions": {
            "botorch": botorch.__version__,
            "botorch_revision": botorch_revision(),
            "torch": torch.__version__,
        },
        "problems": entries,
    }
    for path, obj in [
        (os.path.join(out, "mo_test_functions.json"), fixture),
        (os.path.join(out, "sampling.json"), sampling_fixture()),
    ]:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=1, allow_nan=False)
            f.write("\n")

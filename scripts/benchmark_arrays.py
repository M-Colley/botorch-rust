#!/usr/bin/env python3
"""Wall time of array exchange between Python and the extension.

Cheap functions evaluated on many points are dominated by the conversion of
their inputs and results, so these tasks measure the bindings rather than the
Rust core:

* ``model.posterior(X)`` with ``mean`` and ``variance`` at 10,000 x 6 inputs
  (a ``SingleTaskGP`` on 50 points);
* ``LogExpectedImprovement`` at 10,000 x 1 x 6 candidate sets (raw samples);
* ``DTLZ2(dim=6)`` (two objectives) at 10,000 points;
* the same three calls at a single point (``q = 1``), where the fixed cost of
  a call matters.

Every task runs with torch tensors, NumPy arrays and nested lists (each when
installed), and with BoTorch for reference (``--botorch``). Every timing is
the median over ``--repeats`` runs after a warm-up run, where a run repeats a
call until it takes ``--min-time`` seconds. ``--threads n`` sets the number of
threads of torch and botorch_rs (default: each library's default).

Usage (``scripts/develop.py`` builds the extension)::

    PYTHONPATH=botorch_pyo3/python python scripts/benchmark_arrays.py [--botorch]
"""

from __future__ import annotations

import argparse
import importlib.util
import statistics
import time
from collections.abc import Callable
from typing import Any

import botorch_rs
from botorch_rs.acquisition import LogExpectedImprovement
from botorch_rs.models import SingleTaskGP
from botorch_rs.test_functions.multi_objective import DTLZ2

KINDS = ["list"] + [lib for lib in ("numpy", "torch") if importlib.util.find_spec(lib)]


def time_run(f: Callable[[], Any], min_time: float) -> float:
    """Seconds per call of ``f``, repeated until ``min_time`` has elapsed."""
    calls, start = 0, time.perf_counter()
    while True:
        f()
        calls += 1
        elapsed = time.perf_counter() - start
        if elapsed >= min_time:
            return elapsed / calls


def median_time(f: Callable[[], Any], repeats: int, min_time: float) -> float:
    f()  # warm-up
    return statistics.median(time_run(f, min_time) for _ in range(repeats))


def fmt(seconds: float) -> str:
    if seconds >= 1e-3:
        return f"{seconds * 1e3:.2f} ms"
    return f"{seconds * 1e6:.1f} µs"


def points(n: int, d: int, seed: int) -> list[list[float]]:
    """``n`` deterministic points in [0, 1]^d (a Weyl sequence)."""
    alphas = [(k + 1) * 0.6180339887498949 % 1.0 + 0.1 for k in range(d)]
    return [[(seed + i * a) % 1.0 for a in alphas] for i in range(n)]


def as_kind(x: list[Any], kind: str) -> Any:
    if kind == "numpy":
        import numpy as np

        return np.asarray(x, dtype=np.float64)
    if kind == "torch":
        import torch

        return torch.tensor(x, dtype=torch.float64)
    return x


def tasks(lib: Any, kind: str) -> list[tuple[str, Callable[[], Any]]]:
    """The timed calls of ``lib`` (``botorch_rs`` or ``botorch``) for arrays of ``kind``."""
    if lib is botorch_rs:
        gp, logei, dtlz2 = SingleTaskGP, LogExpectedImprovement, DTLZ2
    else:
        from botorch.acquisition import LogExpectedImprovement as logei
        from botorch.models import SingleTaskGP as gp
        from botorch.test_functions.multi_objective import DTLZ2 as dtlz2

    train_X = points(50, 6, seed=0)
    train_Y = [[sum(x) - x[0] * x[1]] for x in train_X]
    model = gp(as_kind(train_X, kind), as_kind(train_Y, kind))
    if lib is not botorch_rs:
        model.eval()
    acqf = logei(model, best_f=max(y[0] for y in train_Y))
    problem = dtlz2(dim=6, num_objectives=2)
    X = as_kind(points(10_000, 6, seed=1), kind)
    X3 = as_kind([[x] for x in points(10_000, 6, seed=2)], kind)
    x1 = as_kind(points(1, 6, seed=3), kind)

    def posterior(x: Any) -> Any:
        p = model.posterior(x)
        return p.mean, p.variance

    return [
        ("posterior mean + variance, 10,000 x 6", lambda: posterior(X)),
        ("LogEI, 10,000 x 1 x 6", lambda: acqf(X3)),
        ("DTLZ2, 10,000 x 6", lambda: problem(X)),
        ("posterior mean + variance, 1 x 6", lambda: posterior(x1)),
        ("LogEI, 1 x 6", lambda: acqf(x1)),
        ("DTLZ2, 1 x 6", lambda: problem(x1)),
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--min-time", type=float, default=0.2)
    parser.add_argument("--threads", type=int, default=0)
    parser.add_argument("--botorch", action="store_true", help="also time BoTorch (torch)")
    args = parser.parse_args()
    if args.threads:
        botorch_rs.set_num_threads(args.threads)
    runs: list[tuple[str, str, Any]] = [("botorch_rs", kind, botorch_rs) for kind in KINDS]
    if "torch" in KINDS:
        import torch

        if args.threads:
            torch.set_num_threads(args.threads)
        if args.botorch:
            import botorch

            runs.append(("BoTorch", "torch", botorch))
    results: dict[str, dict[str, float]] = {}
    for name, kind, lib in runs:
        column = f"{name} ({kind})"
        for task, f in tasks(lib, kind):
            if kind == "list" and "10,000" in task and lib is botorch_rs and args.repeats > 3:
                t = median_time(f, 3, args.min_time)  # nested lists are slow
            else:
                t = median_time(f, args.repeats, args.min_time)
            results.setdefault(task, {})[column] = t
    columns = [f"{name} ({kind})" for name, kind, _ in runs]
    print("| Task | " + " | ".join(columns) + " |")
    print("|---|" + "---|" * len(columns))
    for task, row in results.items():
        print(f"| {task} | " + " | ".join(fmt(row[c]) for c in columns) + " |")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Wall-time comparison of the multi-objective utilities of botorch_rs and
BoTorch through the Python API.

Tasks, for m = 2, 3, 4 objectives and n = 10, 50, 100, 200 points near the
positive unit sphere (so that most points are Pareto optimal):

* ``Hypervolume(ref_point).compute(Y)`` (dimension sweep);
* ``NondominatedPartitioning``, ``FastNondominatedPartitioning`` and
  ``DominatedPartitioning``: construction, ``get_hypercell_bounds()`` and
  ``compute_hypervolume()``;
* the incremental ``update`` of a ``FastNondominatedPartitioning`` with 5 new
  points (m > 2);
* qNEHVI-style decompositions of 128 Monte-Carlo samples of 20 points: one
  batched ``FastNondominatedPartitioning`` (m = 2) and a
  ``BoxDecompositionList`` of 128 decompositions (m = 3);
* ``is_non_dominated`` of 128 x 200 points (batched).

Both libraries run single-threaded (``torch.set_num_threads(1)``,
``botorch_rs.set_num_threads(1)``); the batched tasks are also timed with
botorch_rs's default number of threads. Every timing is the median over
``--repeats`` runs after a warm-up run, where a run repeats a task until it
takes ``--min-time`` seconds. The hypervolumes of both libraries are checked
to agree. Tasks whose BoTorch run exceeds ``--max-time`` seconds are not
repeated for larger n.

Usage (``scripts/develop.py`` builds the extension)::

    PYTHONPATH=/path/to/botorch:botorch_pyo3/python python scripts/benchmark_mo_utils.py
"""

from __future__ import annotations

import argparse
import statistics
import time
from collections.abc import Callable
from typing import Any

import botorch
import torch
from botorch.utils.multi_objective import box_decompositions as bt_bd
from botorch.utils.multi_objective import hypervolume as bt_hv
from botorch.utils.multi_objective import pareto as bt_pareto

import botorch_rs
from botorch_rs.utils.multi_objective import box_decompositions as rs_bd
from botorch_rs.utils.multi_objective import hypervolume as rs_hv
from botorch_rs.utils.multi_objective import pareto as rs_pareto

DT = torch.float64
CLASSES = ("NondominatedPartitioning", "FastNondominatedPartitioning", "DominatedPartitioning")


def front(n: int, m: int, seed: int, batch: tuple[int, ...] = ()) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    u = torch.rand(*batch, n, m, generator=g, dtype=DT) + 1e-3
    u = u / u.norm(dim=-1, keepdim=True)
    return u * (1.0 - 0.05 * torch.rand(*batch, n, 1, generator=g, dtype=DT))


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
    if seconds >= 1.0:
        return f"{seconds:.2f} s"
    if seconds >= 1e-3:
        return f"{seconds * 1e3:.2f} ms"
    return f"{seconds * 1e6:.0f} µs"


def decomposition(lib: Any, name: str, ref: torch.Tensor, Y: torch.Tensor) -> float:
    p = getattr(lib, name)(ref_point=ref, Y=Y)
    p.get_hypercell_bounds()
    return float(p.compute_hypervolume())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--min-time", type=float, default=0.1)
    parser.add_argument("--max-time", type=float, default=10.0)
    args = parser.parse_args()
    torch.set_num_threads(1)
    default_threads = botorch_rs.get_num_threads()
    print(f"botorch {botorch.__version__} ({botorch.__file__}), torch {torch.__version__}")
    print("| Task | m | n | botorch_rs | BoTorch | speedup |")
    print("|---|---|---|---|---|---|")
    too_slow: set[tuple[str, int]] = set()

    def report(task: str, m: int, n: Any, rs: Callable[[], Any], bt: Callable[[], Any]) -> None:
        botorch_rs.set_num_threads(1)
        t_rs = median_time(rs, args.repeats, args.min_time)
        if (task, m) in too_slow:
            print(f"| {task} | {m} | {n} | {fmt(t_rs)} | (skipped) | |")
            return
        a, e = float(torch.as_tensor(rs()).sum()), float(torch.as_tensor(bt()).sum())
        assert abs(a - e) <= 1e-10 * max(1.0, abs(e)), (task, m, n, a, e)
        t_bt = median_time(bt, args.repeats, args.min_time)
        if t_bt > args.max_time:
            too_slow.add((task, m))
        print(
            f"| {task} | {m} | {n} | {fmt(t_rs)} | {fmt(t_bt)} | {t_bt / t_rs:.0f}× |", flush=True
        )

    for m in (2, 3, 4):
        ref = torch.zeros(m, dtype=DT)
        for n in (10, 50, 100, 200):
            Y = front(n, m, seed=n + m)
            P = Y[bt_pareto.is_non_dominated(Y)]
            report(
                "Hypervolume.compute",
                m,
                n,
                lambda ref=ref, P=P: rs_hv.Hypervolume(ref).compute(P),
                lambda ref=ref, P=P: bt_hv.Hypervolume(ref).compute(P),
            )
            for name in CLASSES:
                report(
                    name,
                    m,
                    n,
                    lambda name=name, ref=ref, Y=Y: decomposition(rs_bd, name, ref, Y),
                    lambda name=name, ref=ref, Y=Y: decomposition(bt_bd, name, ref, Y),
                )
            if m > 2:
                new = front(5, m, seed=7) * 1.04

                def update(
                    lib: Any, ref: torch.Tensor = ref, Y: torch.Tensor = Y, new: torch.Tensor = new
                ) -> Any:
                    p = lib.FastNondominatedPartitioning(ref_point=ref, Y=Y)
                    p.update(new)
                    return p.compute_hypervolume()

                report(
                    "FastNondominatedPartitioning + update(5 points)",
                    m,
                    n,
                    lambda update=update: update(rs_bd),
                    lambda update=update: update(bt_bd),
                )

    # qNEHVI-style decompositions of 128 MC samples.
    samples2 = front(20, 2, seed=1, batch=(128,))
    samples3 = front(20, 3, seed=2, batch=(128,))
    ref2, ref3 = torch.zeros(2, dtype=DT), torch.zeros(3, dtype=DT)

    def batched(lib: Any) -> Any:
        return lib.FastNondominatedPartitioning(ref_point=ref2, Y=samples2).compute_hypervolume()

    def listed(lib: Any) -> Any:
        bdl = lib.BoxDecompositionList(
            *(lib.FastNondominatedPartitioning(ref_point=ref3, Y=y) for y in samples3)
        )
        bdl.get_hypercell_bounds()
        return bdl.compute_hypervolume()

    big = front(200, 3, seed=3, batch=(128,))
    for threads in (1, default_threads):
        for task, m, n, rs, bt in (
            ("batched FastNondominatedPartitioning, 128 samples", 2, 20, batched, batched),
            ("BoxDecompositionList of 128", 3, 20, listed, listed),
            (
                "is_non_dominated, batch 128",
                3,
                200,
                rs_pareto.is_non_dominated,
                bt_pareto.is_non_dominated,
            ),
        ):
            arg_rs = big if task.startswith("is_") else rs_bd
            arg_bt = big if task.startswith("is_") else bt_bd
            botorch_rs.set_num_threads(threads)
            t_rs = median_time(lambda rs=rs, a=arg_rs: rs(a), args.repeats, args.min_time)
            botorch_rs.set_num_threads(1)
            t_bt = median_time(lambda bt=bt, a=arg_bt: bt(a), args.repeats, args.min_time)
            label = f"{task} ({threads} thread{'s' if threads > 1 else ''})"
            print(f"| {label} | {m} | {n} | {fmt(t_rs)} | {fmt(t_bt)} | {t_bt / t_rs:.0f}× |")
    botorch_rs.set_num_threads(default_threads)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Wall-time comparison of pathwise sampling and the information-theoretic
multi-objective acquisition functions of botorch_rs and BoTorch through the
Python API.

Tasks, on a two-output ``SingleTaskGP`` with ``Normalize`` and
``Standardize`` (d = 3, n = 20 and 50 training points):

* ``draw_matheron_paths`` of 16 paths with 1024 Fourier features, and the
  evaluation of the paths (values and gradients) at 1024 points;
* ``sample_optimal_points`` of 10 Pareto sets of 10 points (random search,
  1024 points per try) and ``compute_sample_box_decomposition`` of the fronts;
* qLowerBoundMultiObjectiveMaxValueEntropySearch and
  qLowerBoundMultiObjectiveJointEntropySearch ("LB" and "MC"): construction,
  and values with gradients at 256 candidate sets of q = 1 and q = 2;
* qMultiObjectivePredictiveEntropySearch: construction (EP) and values with
  gradients at 256 candidate sets of q = 1.

Both libraries run single-threaded (``torch.set_num_threads(1)``,
``botorch_rs.set_num_threads(1)``); botorch_rs is also timed with its
default number of threads (``RAYON_NUM_THREADS`` if set). Every timing is the
median over ``--repeats`` runs after a warm-up run, where a run repeats a
task until it takes ``--min-time`` seconds. The random numbers of the two
libraries differ (each draws its own), so the sampled optima differ; the
acquisition functions are evaluated on the same Pareto sets and fronts.

Usage (``scripts/develop.py`` builds the extension)::

    PYTHONPATH=/path/to/botorch:botorch_pyo3/python python scripts/benchmark_mo_entropy.py
"""

from __future__ import annotations

import argparse
import functools
import statistics
import time
from collections.abc import Callable
from typing import Any

import botorch
import torch
from botorch.acquisition.multi_objective import joint_entropy_search as bt_jes
from botorch.acquisition.multi_objective import max_value_entropy_search as bt_mes
from botorch.acquisition.multi_objective import predictive_entropy_search as bt_pes
from botorch.acquisition.multi_objective import utils as bt_utils
from botorch.models import SingleTaskGP as BtSingleTaskGP
from botorch.models.transforms.input import Normalize as BtNormalize
from botorch.sampling import pathwise as bt_pw

import botorch_rs
from botorch_rs.acquisition.multi_objective import joint_entropy_search as rs_jes
from botorch_rs.acquisition.multi_objective import max_value_entropy_search as rs_mes
from botorch_rs.acquisition.multi_objective import predictive_entropy_search as rs_pes
from botorch_rs.acquisition.multi_objective import utils as rs_utils
from botorch_rs.models import SingleTaskGP as RsSingleTaskGP
from botorch_rs.models.transforms import Normalize as RsNormalize
from botorch_rs.sampling import pathwise as rs_pw

DT = torch.float64


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
        return f"{seconds * 1e3:.1f} ms"
    return f"{seconds * 1e6:.0f} us"


def models(n: int, seed: int) -> tuple[Any, Any, Any]:
    """The same two-output model in both libraries (hyperparameters copied)."""
    g = torch.Generator().manual_seed(seed)
    X = torch.rand(n, 3, generator=g, dtype=DT)
    Y = torch.stack(
        [torch.sin(3 * X[:, 0]) + X[:, 1], torch.cos(3 * X[:, 0]) - X[:, 2] ** 2], dim=-1
    )
    bt = BtSingleTaskGP(X, Y, input_transform=BtNormalize(d=3))
    bt.initialize(
        **{
            "covar_module.lengthscale": torch.tensor([[[0.4, 0.6, 0.8]], [[0.5, 0.7, 0.3]]]),
            "likelihood.noise": torch.tensor([[1e-3], [2e-3]]),
        }
    )
    bt.eval()
    rs = RsSingleTaskGP(X, Y, input_transform=RsNormalize(d=3))
    rs.initialize(**{k: v.detach() for k, v in bt.named_parameters()})
    return bt, rs, X


def grad_eval(acqf: Any, X: torch.Tensor) -> None:
    Xg = X.clone().requires_grad_(True)
    torch.autograd.grad(acqf(Xg).sum(), Xg)


Task = tuple[str, Callable[[], Any], Callable[[], Any]]


def tasks_for(n: int) -> list[Task]:
    """The tasks on models with ``n`` training points: (name, BoTorch,
    botorch_rs)."""
    bt, rs, _ = models(n, n)
    bounds = torch.tensor([[0.0] * 3, [1.0] * 3], dtype=DT)
    g = torch.Generator().manual_seed(1)
    Xe = torch.rand(1024, 3, generator=g, dtype=DT)
    X1 = torch.rand(256, 1, 3, generator=g, dtype=DT)
    X2 = torch.rand(256, 2, 3, generator=g, dtype=DT)
    prior = functools.partial(bt_pw.draw_kernel_feature_paths, num_features=1024)
    rs_prior = functools.partial(rs_pw.draw_kernel_feature_paths, num_features=1024)
    bp = bt_pw.draw_matheron_paths(bt, torch.Size([16]), prior_sampler=prior)
    rp = rs_pw.draw_matheron_paths(rs, torch.Size([16]), rs_prior, seed=0)
    ps, pf = rs_utils.sample_optimal_points(rs, bounds, 10, 10, seed=0)
    cells = bt_utils.compute_sample_box_decomposition(pf)
    tasks: list[Task] = [
        (
            "draw_matheron_paths (16 paths, 1024 features)",
            lambda: bt_pw.draw_matheron_paths(bt, torch.Size([16]), prior_sampler=prior),
            lambda: rs_pw.draw_matheron_paths(rs, torch.Size([16]), rs_prior, seed=0),
        ),
        (
            "paths: values + gradients at 1024 points",
            lambda: grad_eval(bp, Xe),
            lambda: grad_eval(rp, Xe),
        ),
        (
            "sample_optimal_points (10 sets of 10 points)",
            lambda: bt_utils.sample_optimal_points(bt, bounds, 10, 10),
            lambda: rs_utils.sample_optimal_points(rs, bounds, 10, 10, seed=0),
        ),
        (
            "compute_sample_box_decomposition (10 fronts)",
            lambda: bt_utils.compute_sample_box_decomposition(pf),
            lambda: rs_utils.compute_sample_box_decomposition(pf),
        ),
    ]

    def jes(lib: Any, model: Any, est: str) -> Any:
        return lib.qLowerBoundMultiObjectiveJointEntropySearch(
            model, ps, pf, cells, estimation_type=est
        )

    for est in ("LB", "MC"):
        bm = bt_mes.qLowerBoundMultiObjectiveMaxValueEntropySearch(bt, cells, estimation_type=est)
        rm = rs_mes.qLowerBoundMultiObjectiveMaxValueEntropySearch(rs, cells, estimation_type=est)
        bj, rj = jes(bt_jes, bt, est), jes(rs_jes, rs, est)
        tasks += [
            (
                f"JES-{est} construction",
                functools.partial(jes, bt_jes, bt, est),
                functools.partial(jes, rs_jes, rs, est),
            ),
            (
                f"MES-{est}: 256 x q=1 values + gradients",
                functools.partial(grad_eval, bm, X1),
                functools.partial(grad_eval, rm, X1),
            ),
            (
                f"JES-{est}: 256 x q=1 values + gradients",
                functools.partial(grad_eval, bj, X1),
                functools.partial(grad_eval, rj, X1),
            ),
            (
                f"JES-{est}: 256 x q=2 values + gradients",
                functools.partial(grad_eval, bj, X2),
                functools.partial(grad_eval, rj, X2),
            ),
        ]
    ps_pes = ps[:, :4]

    def bt_pes_new() -> Any:
        bt.train()  # the raw training inputs (see the PES documentation)
        try:
            return bt_pes.qMultiObjectivePredictiveEntropySearch(bt, ps_pes)
        finally:
            bt.eval()

    bpes = bt_pes_new()
    rpes = rs_pes.qMultiObjectivePredictiveEntropySearch(rs, ps_pes)
    tasks += [
        (
            f"PES construction (EP, {rpes.ep_iterations} iterations)",
            bt_pes_new,
            lambda: rs_pes.qMultiObjectivePredictiveEntropySearch(rs, ps_pes),
        ),
        (
            "PES: 256 x q=1 values + gradients",
            lambda: grad_eval(bpes, X1),
            lambda: grad_eval(rpes, X1),
        ),
    ]
    return tasks


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--min-time", type=float, default=0.2)
    parser.add_argument("--sizes", type=int, nargs="*", default=[20, 50])
    args = parser.parse_args()
    torch.set_num_threads(1)
    print(f"BoTorch {botorch.__version__}, botorch_rs {botorch_rs.__version__}")
    print(f"botorch_rs default threads: {botorch_rs.get_num_threads()}")
    header = f"{'task':<52} {'BoTorch':>10} {'rs (1 thr)':>11} {'rs (default)':>13} {'speedup':>8}"
    for n in args.sizes:
        print(f"\nn = {n}\n{header}")
        for name, f_bt, f_rs in tasks_for(n):
            t_bt = median_time(f_bt, args.repeats, args.min_time)
            botorch_rs.set_num_threads(1)
            try:
                t_rs1 = median_time(f_rs, args.repeats, args.min_time)
            finally:
                botorch_rs.set_num_threads(None)
            t_rs = median_time(f_rs, args.repeats, args.min_time)
            print(
                f"{name:<52} {fmt(t_bt):>10} {fmt(t_rs1):>11} {fmt(t_rs):>13} "
                f"{t_bt / t_rs1:>7.1f}x",
                flush=True,
            )


if __name__ == "__main__":
    main()

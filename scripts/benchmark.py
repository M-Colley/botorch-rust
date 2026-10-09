#!/usr/bin/env python3
"""Wall-time comparison of botorch_rs and BoTorch through the Python API.

Tasks (all on the 6-dimensional Hartmann function, negated for maximization):

* ``fit``: ``SingleTaskGP(X, Y)`` + ``fit_gpytorch_mll`` for n = 20, 100, 300
  training points (a Weyl sequence in [0, 1]^6, noise-free targets), from
  BoTorch's default initialization;
* ``optimize_acqf`` with 10 restarts and 512 raw samples on a 50-point model
  with fixed hyperparameters (noise 1e-3, constant 0, lengthscales 0.5):
  ``LogExpectedImprovement`` (q = 1), ``qLogExpectedImprovement`` (q = 3, the
  same 512 explicit base samples in both libraries) and
  ``qLogNoisyExpectedImprovement`` (q = 1, defaults: 512 Sobol samples, pruned
  baseline; the libraries' Sobol sequences differ);
* ``BO loop``: 30 iterations of fitting a default ``SingleTaskGP`` and
  maximizing LogEI from the same 14-point Sobol design (the loop of
  ``botorch_core/examples/bo_loop.rs``).

Every task is timed with one thread (``torch.set_num_threads(1)``,
``botorch_rs.set_num_threads(1)``) and with each library's default number of
threads; the median over repeated runs is reported, after one warm-up run.

Usage (``scripts/develop.py`` builds the extension)::

    PYTHONPATH=/path/to/botorch:botorch_pyo3/python python scripts/benchmark.py [--repeats 5]

Use ``--no-botorch`` to time botorch_rs only.
"""

from __future__ import annotations

import argparse
import math
import os
import platform
import statistics
import sys
import time
import warnings
from collections.abc import Callable
from typing import Any

import torch

import botorch_rs

DT = torch.float64
D = 6
FIT_SIZES = (20, 100, 300)
LOOP_ITERATIONS = 30


def weyl(n: int, d: int = D) -> torch.Tensor:
    primes = [2, 3, 5, 7, 11, 13, 17, 19]
    alpha = torch.tensor([math.sqrt(p) % 1.0 for p in primes[:d]], dtype=DT)
    i = torch.arange(1, n + 1, dtype=DT).unsqueeze(-1)
    return torch.remainder(0.5 + i * alpha, 1.0)


class Library:
    """The pieces of one implementation (BoTorch or botorch_rs)."""

    def __init__(self, name: str) -> None:
        self.name = name
        if name == "botorch_rs":
            from botorch_rs import acquisition, fit, mlls, models, optim, sampling, test_functions
            from botorch_rs.utils.sampling import draw_sobol_samples

            self._set_threads = botorch_rs.set_num_threads
            self.default_threads = botorch_rs.get_num_threads()
        else:
            import botorch
            from botorch import acquisition, fit, models, optim, sampling, test_functions
            from botorch.utils.sampling import draw_sobol_samples
            from gpytorch import mlls

            self._set_threads = torch.set_num_threads
            self.default_threads = torch.get_num_threads()
            self.file = botorch.__file__
        self.SingleTaskGP = models.SingleTaskGP
        self.MLL = mlls.ExactMarginalLogLikelihood
        self.fit = fit.fit_gpytorch_mll
        self.LogEI = acquisition.LogExpectedImprovement
        self.qLogEI = acquisition.qLogExpectedImprovement
        self.qLogNEI = acquisition.qLogNoisyExpectedImprovement
        self.Sampler = sampling.SobolQMCNormalSampler
        self.optimize_acqf = optim.optimize_acqf
        self.Hartmann = test_functions.Hartmann
        self.draw_sobol_samples = draw_sobol_samples

    def use_threads(self, single: bool) -> None:
        """One thread, or the library's default number of threads."""
        if single:
            self._set_threads(1)
        elif self.name == "botorch_rs":
            self._set_threads(None)  # rayon's global pool
        else:
            self._set_threads(self.default_threads)

    def mll_loss(self, model: Any) -> float:
        mll = self.MLL(model.likelihood, model)
        if self.name == "botorch_rs":
            return -float(mll())
        model.train()
        with torch.no_grad():
            loss = -mll(model(*model.train_inputs), model.train_targets).item()
        model.eval()
        return loss


def fixed_model(lib: Library) -> tuple[Any, torch.Tensor, torch.Tensor]:
    X = weyl(50)
    Y = lib.Hartmann(dim=D, negate=True, dtype=DT)(X, noise=False).unsqueeze(-1)
    model = lib.SingleTaskGP(X, Y)
    model.initialize(
        **{
            "likelihood.noise_covar.raw_noise": torch.tensor([1e-3], dtype=DT),
            "mean_module.raw_constant": torch.tensor(0.0, dtype=DT),
            "covar_module.raw_lengthscale": torch.full((1, D), 0.5, dtype=DT),
        }
    )
    model.eval()
    return model, X, Y


def tasks(lib: Library) -> dict[str, Callable[[int], float]]:
    """Task name -> function(seed) returning a result value (for sanity checks)."""
    bounds = torch.stack([torch.zeros(D, dtype=DT), torch.ones(D, dtype=DT)])
    problem = lib.Hartmann(dim=D, negate=True, dtype=DT)
    out: dict[str, Callable[[int], float]] = {}

    for n in FIT_SIZES:
        X = weyl(n)
        Y = problem(X, noise=False).unsqueeze(-1)

        def fit_task(seed: int, X: torch.Tensor = X, Y: torch.Tensor = Y) -> float:
            model = lib.SingleTaskGP(X, Y)
            lib.fit(lib.MLL(model.likelihood, model))
            return lib.mll_loss(model)

        out[f"fit, n = {n}"] = fit_task

    model, X50, Y50 = fixed_model(lib)

    def logei(seed: int) -> float:
        acqf = lib.LogEI(model, best_f=Y50.max())
        _, v = lib.optimize_acqf(
            acqf, bounds, q=1, num_restarts=10, raw_samples=512, options={"seed": seed}
        )
        return float(v)

    # Identical base samples in both libraries, so that both optimize the same
    # sample-average approximation (their Sobol sequences differ).
    Z = torch.randn(512, 3, dtype=DT, generator=torch.Generator().manual_seed(0))

    def qlogei(seed: int) -> float:
        sampler = lib.Sampler(sample_shape=torch.Size([512]), seed=0)
        sampler.base_samples = Z.view(512, 1, 3).clone()
        acqf = lib.qLogEI(model, best_f=Y50.max(), sampler=sampler)
        _, v = lib.optimize_acqf(
            acqf, bounds, q=3, num_restarts=10, raw_samples=512, options={"seed": seed}
        )
        return float(v)

    def qlognei(seed: int) -> float:
        sampler = lib.Sampler(sample_shape=torch.Size([512]), seed=0)
        acqf = lib.qLogNEI(model, X_baseline=X50, sampler=sampler)
        _, v = lib.optimize_acqf(
            acqf, bounds, q=1, num_restarts=10, raw_samples=512, options={"seed": seed}
        )
        return float(v)

    out["optimize_acqf, LogEI, q = 1"] = logei
    out["optimize_acqf, qLogEI, q = 3"] = qlogei
    out["optimize_acqf, qLogNEI, q = 1"] = qlognei

    init = botorch_rs.utils.sampling.draw_sobol_samples(bounds, n=2 * (D + 1), q=1, seed=0)
    X0 = init.squeeze(-2)

    def bo_loop(seed: int) -> float:
        X = X0.clone()
        Y = problem(X, noise=False).unsqueeze(-1)
        for it in range(LOOP_ITERATIONS):
            gp = lib.SingleTaskGP(X, Y)
            lib.fit(lib.MLL(gp.likelihood, gp))
            acqf = lib.LogEI(gp, best_f=Y.max())
            cand, _ = lib.optimize_acqf(
                acqf,
                bounds,
                q=1,
                num_restarts=10,
                raw_samples=512,
                options={"seed": 1000 * seed + it},
            )
            X = torch.cat([X, cand])
            Y = torch.cat([Y, problem(cand, noise=False).unsqueeze(-1)])
        return -Y.max().item()

    out[f"BO loop, {LOOP_ITERATIONS} iterations"] = bo_loop
    return out


def time_task(fn: Callable[[int], float], repeats: int) -> tuple[float, float]:
    """Median wall time over ``repeats`` runs (after a warm-up run) and the
    median result value."""
    fn(0)
    times, values = [], []
    for r in range(repeats):
        t = time.perf_counter()
        values.append(fn(r))
        times.append(time.perf_counter() - t)
    return statistics.median(times), statistics.median(values)


def fmt_speedup(ratio: float) -> str:
    return f"{ratio:.1f}x" if ratio < 10 else f"{ratio:.0f}x"


def fmt_time(seconds: float) -> str:
    if seconds < 1e-3:
        return f"{seconds * 1e6:.0f} µs"
    if seconds < 1.0:
        return f"{seconds * 1e3:.1f} ms"
    return f"{seconds:.2f} s"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--repeats", type=int, default=5, help="timed runs per task")
    parser.add_argument("--loop-repeats", type=int, default=3, help="timed BO loops")
    parser.add_argument("--no-botorch", action="store_true", help="time botorch_rs only")
    args = parser.parse_args()
    warnings.simplefilter("ignore")

    libs = [Library("botorch_rs")]
    if not args.no_botorch:
        libs.append(Library("botorch"))
    default_threads = {lib.name: lib.default_threads for lib in libs}

    print(
        f"Python {platform.python_version()}, torch {torch.__version__}, "
        f"botorch_rs {botorch_rs.__version__}"
    )
    if not args.no_botorch:
        print(f"BoTorch from {libs[1].file}")
    print(
        f"{platform.platform()}, {platform.processor() or platform.machine()}, "
        f"{os.cpu_count()} logical CPUs; default threads: "
        + ", ".join(f"{k} {v}" for k, v in default_threads.items())
    )

    results: dict[tuple[str, str, str], tuple[float, float]] = {}
    for setting in ("1", "default"):
        for lib in libs:
            lib.use_threads(setting == "1")
            for name, fn in tasks(lib).items():
                repeats = args.loop_repeats if name.startswith("BO loop") else args.repeats
                results[(name, lib.name, setting)] = time_task(fn, repeats)
                print(
                    f"  {lib.name:10s} threads={setting:7s} {name:32s} "
                    f"{fmt_time(results[(name, lib.name, setting)][0])}",
                    file=sys.stderr,
                )
        for lib in libs:
            lib.use_threads(False)

    names = list(tasks(libs[0]))
    rs_n, bt_n = default_threads["botorch_rs"], default_threads.get("botorch", 0)
    header = ["Task", "botorch_rs, 1 thread"]
    if not args.no_botorch:
        header += ["BoTorch, 1 thread", "speedup"]
    header += [f"botorch_rs, {rs_n} threads"]
    if not args.no_botorch:
        header += [f"BoTorch, {bt_n} threads", "speedup"]
    print()
    print("| " + " | ".join(header) + " |")
    print("|" + "---|" * len(header))
    for name in names:
        row = [name, fmt_time(results[(name, "botorch_rs", "1")][0])]
        if not args.no_botorch:
            rs, bt = results[(name, "botorch_rs", "1")][0], results[(name, "botorch", "1")][0]
            row += [fmt_time(bt), fmt_speedup(bt / rs)]
        row += [fmt_time(results[(name, "botorch_rs", "default")][0])]
        if not args.no_botorch:
            rs = results[(name, "botorch_rs", "default")][0]
            bt = results[(name, "botorch", "default")][0]
            row += [fmt_time(bt), fmt_speedup(bt / rs)]
        print("| " + " | ".join(row) + " |")

    if not args.no_botorch:
        print()
        print(
            "Results (median over runs, single thread): final loss for fits, "
            "acquisition value for optimize_acqf, best Hartmann value for the loop."
        )
        print()
        print("| Task | botorch_rs | BoTorch |")
        print("|---|---|---|")
        for name in names:
            rs, bt = results[(name, "botorch_rs", "1")][1], results[(name, "botorch", "1")][1]
            print(f"| {name} | {rs:.6f} | {bt:.6f} |")


if __name__ == "__main__":
    main()

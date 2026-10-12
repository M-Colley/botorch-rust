#!/usr/bin/env python3
"""Wall-time comparison of botorch_rs and BoTorch through the Python API.

Tasks (all on the 6-dimensional Hartmann function, negated for maximization):

* ``BO loop``: 30 iterations of fitting a default ``SingleTaskGP`` and
  maximizing LogEI from the same 14-point Sobol design (the loop of
  ``botorch_core/examples/bo_loop.rs``);
* ``optimize_acqf`` with 10 restarts and 512 raw samples on a 50-point model
  with fixed hyperparameters (noise 1e-3, constant 0, lengthscales 0.5):
  ``LogExpectedImprovement`` (q = 1), the same with the linear constraint
  ``sum(x) <= 2`` (initial conditions from the hit-and-run polytope sampler,
  then SLSQP), ``qLogExpectedImprovement`` (q = 3, the same 512 explicit base
  samples in both libraries) and ``qLogNoisyExpectedImprovement`` (q = 1,
  defaults: 512 Sobol samples, pruned baseline; the libraries' Sobol
  sequences differ);
* ``fit``: ``SingleTaskGP(X, Y)`` + ``fit_gpytorch_mll`` for n = 20, 100, 300
  training points (a Weyl sequence in [0, 1]^6, noise-free targets), from
  BoTorch's default initialization.

Multi-objective tasks (objectives negated for maximization):

* ``MOBO loop``: 20 iterations of BoTorch's qLogNEHVI tutorial on
  BraninCurrin (the loop of ``botorch_core/examples/mobo_loop.rs``): fitting a
  two-output ``SingleTaskGP`` with ``Normalize`` from the same 6-point Sobol
  design and maximizing qLogNEHVI (128 Sobol samples, pruned baseline) for
  q = 2 candidates with ``sequential=True``, 10 restarts and 512 raw samples;
  the result is the final hypervolume;
* ``optimize_acqf`` with 10 restarts and 512 raw samples on 30-point
  DTLZ2 models (d = 6, a Weyl sequence, fixed hyperparameters as above):
  ``qLogNoisyExpectedHypervolumeImprovement`` (m = 2, q = 2,
  ``sequential=True``, 128 Sobol samples; the libraries' Sobol sequences
  differ) and ``qLogExpectedHypervolumeImprovement`` (m = 3, q = 2, the same
  128 explicit base samples in both libraries, the partitioning of the
  posterior mean at the training points).

Every task is timed with one thread (``torch.set_num_threads(1)``,
``botorch_rs.set_num_threads(1)``) and with each library's default number of
threads; the median over runs with different seeds is reported, after one
warm-up run (fast tasks are repeated within a run until it takes 0.2 s).
``--json`` also writes all runs, the result values and the environment, the
input of ``scripts/plot_benchmarks.py``.

Usage (``scripts/develop.py`` builds the extension)::

    PYTHONPATH=/path/to/botorch:botorch_pyo3/python python scripts/benchmark.py [--repeats 5]
    PYTHONPATH=... python scripts/benchmark.py --json docs/assets/benchmark.json

Use ``--no-botorch`` to time botorch_rs only and ``-k`` to select tasks.
"""

from __future__ import annotations

import argparse
import datetime
import json
import math
import os
import platform
import re
import statistics
import subprocess
import sys
import time
import warnings
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

import botorch_rs

DT = torch.float64
D = 6
FIT_SIZES = (20, 100, 300)
LOOP_ITERATIONS = 30
MOBO_ITERATIONS = 20


@dataclass(frozen=True)
class Task:
    name: str  # row of the markdown tables
    label: str  # row label of the chart
    detail: str  # setting shown under the chart label
    run: Callable[[int], float]  # seed -> result value (for sanity checks)


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
            from botorch_rs.acquisition import multi_objective as mo_acquisition
            from botorch_rs.models.transforms.input import Normalize
            from botorch_rs.test_functions import multi_objective as mo_problems
            from botorch_rs.utils.multi_objective import box_decompositions
            from botorch_rs.utils.sampling import draw_sobol_samples

            self._set_threads = botorch_rs.set_num_threads
            self.default_threads = botorch_rs.get_num_threads()
            self.version = botorch_rs.__version__
        else:
            import botorch
            from botorch import acquisition, fit, models, optim, sampling, test_functions
            from botorch.acquisition import multi_objective as mo_acquisition
            from botorch.models.transforms.input import Normalize
            from botorch.test_functions import multi_objective as mo_problems
            from botorch.utils.multi_objective import box_decompositions
            from botorch.utils.sampling import draw_sobol_samples
            from gpytorch import mlls

            self._set_threads = torch.set_num_threads
            self.default_threads = torch.get_num_threads()
            self.file = botorch.__file__
            self.version = botorch.__version__
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
        self.qLogNEHVI = mo_acquisition.qLogNoisyExpectedHypervolumeImprovement
        self.qLogEHVI = mo_acquisition.qLogExpectedHypervolumeImprovement
        self.BraninCurrin = mo_problems.BraninCurrin
        self.DTLZ2 = mo_problems.DTLZ2
        self.FastNondominatedPartitioning = box_decompositions.FastNondominatedPartitioning
        self.DominatedPartitioning = box_decompositions.DominatedPartitioning
        self.Normalize = Normalize

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


def tasks(lib: Library) -> list[Task]:
    """The benchmark tasks, in the order of the tables and the chart."""
    bounds = torch.stack([torch.zeros(D, dtype=DT), torch.ones(D, dtype=DT)])
    problem = lib.Hartmann(dim=D, negate=True, dtype=DT)
    out: list[Task] = []

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

    out.append(
        Task(
            f"BO loop, {LOOP_ITERATIONS} iterations",
            "Bayesian optimization",
            f"Hartmann-6, {LOOP_ITERATIONS} iterations",
            bo_loop,
        )
    )

    model, X50, Y50 = fixed_model(lib)

    def logei(seed: int) -> float:
        acqf = lib.LogEI(model, best_f=Y50.max())
        _, v = lib.optimize_acqf(
            acqf, bounds, q=1, num_restarts=10, raw_samples=512, options={"seed": seed}
        )
        return float(v)

    # sum(x) <= 2, as BoTorch's (indices, coefficients, rhs) with sum >= rhs.
    budget = [(torch.arange(D), torch.full((D,), -1.0, dtype=DT), -2.0)]

    def logei_constrained(seed: int) -> float:
        acqf = lib.LogEI(model, best_f=Y50.max())
        _, v = lib.optimize_acqf(
            acqf,
            bounds,
            q=1,
            num_restarts=10,
            raw_samples=512,
            inequality_constraints=budget,
            options={"seed": seed},
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

    out += [
        Task("optimize_acqf, LogEI, q = 1", "Optimize LogEI", "q = 1, 10 restarts", logei),
        Task(
            "optimize_acqf, LogEI, q = 1, sum(x) <= 2",
            "Optimize LogEI",
            "linear constraint (SLSQP)",
            logei_constrained,
        ),
        Task("optimize_acqf, qLogEI, q = 3", "Optimize qLogEI", "q = 3", qlogei),
        Task("optimize_acqf, qLogNEI, q = 1", "Optimize qLogNEI", "q = 1", qlognei),
    ]

    for n in FIT_SIZES:
        X = weyl(n)
        Y = problem(X, noise=False).unsqueeze(-1)

        def fit_task(seed: int, X: torch.Tensor = X, Y: torch.Tensor = Y) -> float:
            model = lib.SingleTaskGP(X, Y)
            lib.fit(lib.MLL(model.likelihood, model))
            return lib.mll_loss(model)

        out.append(Task(f"fit, n = {n}", "Fit GP", f"n = {n}", fit_task))
    return out + multi_objective_tasks(lib)


def mo_model(lib: Library, m: int, n: int) -> tuple[Any, torch.Tensor, torch.Tensor]:
    """An m-output SingleTaskGP on DTLZ2 (d = 6) with fixed hyperparameters
    (the values of :func:`fixed_model` for every output) and the reference
    point."""
    problem = lib.DTLZ2(dim=D, num_objectives=m, negate=True)
    X = weyl(n)
    model = lib.SingleTaskGP(X, problem(X, noise=False))
    model.initialize(
        **{
            "likelihood.noise_covar.raw_noise": torch.full((m, 1), 1e-3, dtype=DT),
            "mean_module.raw_constant": torch.zeros(m, dtype=DT),
            "covar_module.raw_lengthscale": torch.full((m, 1, D), 0.5, dtype=DT),
        }
    )
    model.eval()
    return model, X, torch.as_tensor(problem.ref_point, dtype=DT)


def multi_objective_tasks(lib: Library) -> list[Task]:
    """The multi-objective tasks (see the module documentation)."""
    bounds = torch.stack([torch.zeros(D, dtype=DT), torch.ones(D, dtype=DT)])
    bounds2 = bounds[:, :2]
    branin_currin = lib.BraninCurrin(negate=True)
    ref_bc = torch.as_tensor(branin_currin.ref_point, dtype=DT)
    X0 = botorch_rs.utils.sampling.draw_sobol_samples(bounds2, n=6, q=1, seed=0).squeeze(-2)

    def hypervolume(Y: torch.Tensor) -> float:
        hv = lib.DominatedPartitioning(ref_point=ref_bc, Y=Y).compute_hypervolume()
        return float(torch.as_tensor(hv).reshape(-1)[0])

    def mobo_loop(seed: int) -> float:
        X = X0.clone()
        Y = torch.as_tensor(branin_currin(X, noise=False), dtype=DT)
        for it in range(MOBO_ITERATIONS):
            gp = lib.SingleTaskGP(X, Y, input_transform=lib.Normalize(d=2, bounds=bounds2))
            lib.fit(lib.MLL(gp.likelihood, gp))
            acqf = lib.qLogNEHVI(
                gp,
                ref_point=ref_bc,
                X_baseline=X,
                prune_baseline=True,
                sampler=lib.Sampler(sample_shape=torch.Size([128]), seed=1000 * seed + it),
            )
            cand, _ = lib.optimize_acqf(
                acqf,
                bounds2,
                q=2,
                num_restarts=10,
                raw_samples=512,
                sequential=True,
                options={"seed": 1000 * seed + it},
            )
            X = torch.cat([X, cand])
            Y = torch.cat([Y, torch.as_tensor(branin_currin(cand, noise=False), dtype=DT)])
        return hypervolume(Y)

    model2, X2, ref2 = mo_model(lib, 2, 30)

    def qlognehvi(seed: int) -> float:
        sampler = lib.Sampler(sample_shape=torch.Size([128]), seed=0)
        acqf = lib.qLogNEHVI(model2, ref_point=ref2, X_baseline=X2, sampler=sampler)
        _, v = lib.optimize_acqf(
            acqf,
            bounds,
            q=2,
            num_restarts=10,
            raw_samples=512,
            sequential=True,
            options={"seed": seed},
        )
        return float(torch.as_tensor(v).sum())

    model3, X3, ref3 = mo_model(lib, 3, 30)
    with torch.no_grad():
        mean3 = torch.as_tensor(model3.posterior(X3).mean, dtype=DT)
    partitioning3 = lib.FastNondominatedPartitioning(ref_point=ref3, Y=mean3)
    # Identical base samples (q x m = 6 values per sample) in both libraries.
    Z3 = torch.randn(128, 6, dtype=DT, generator=torch.Generator().manual_seed(0))

    def qlogehvi(seed: int) -> float:
        sampler = lib.Sampler(sample_shape=torch.Size([128]), seed=0)
        sampler.base_samples = Z3.view(128, 1, 2, 3).clone()
        acqf = lib.qLogEHVI(model3, ref_point=ref3, partitioning=partitioning3, sampler=sampler)
        _, v = lib.optimize_acqf(
            acqf, bounds, q=2, num_restarts=10, raw_samples=512, options={"seed": seed}
        )
        return float(v)

    return [
        Task(
            f"MOBO loop, qLogNEHVI, {MOBO_ITERATIONS} iterations",
            "Multi-objective BO",
            f"BraninCurrin, qLogNEHVI, {MOBO_ITERATIONS} iterations",
            mobo_loop,
        ),
        Task(
            "optimize_acqf, qLogNEHVI, m = 2, q = 2",
            "Optimize qLogNEHVI",
            "m = 2, q = 2 sequential",
            qlognehvi,
        ),
        Task(
            "optimize_acqf, qLogEHVI, m = 3, q = 2", "Optimize qLogEHVI", "m = 3, q = 2", qlogehvi
        ),
    ]


def time_task(
    fn: Callable[[int], float], repeats: int, min_time: float = 0.2
) -> tuple[list[float], list[float]]:
    """Wall times per call and result values of ``repeats`` runs (seeds 0, 1,
    ...), after a warm-up run. Like ``timeit``, a run of a fast task calls it
    repeatedly with the same seed until the run takes at least ``min_time``
    seconds, and reports the mean time per call."""
    t = time.perf_counter()
    fn(0)
    calls = max(1, math.ceil(min_time / (time.perf_counter() - t)))
    times, values = [], []
    for r in range(repeats):
        t = time.perf_counter()
        for _ in range(calls):
            value = fn(r)
        times.append((time.perf_counter() - t) / calls)
        values.append(value)
    return times, values


def fmt_speedup(ratio: float) -> str:
    return f"{ratio:.1f}x" if ratio < 10 else f"{ratio:.0f}x"


def fmt_time(seconds: float) -> str:
    if seconds < 1e-3:
        return f"{seconds * 1e6:.0f} µs"
    if seconds < 1.0:
        return f"{seconds * 1e3:.1f} ms"
    return f"{seconds:.2f} s"


def cpu_name() -> str:
    """The processor's name: from py-cpuinfo if it is installed, else the OS."""
    name = ""
    try:
        import cpuinfo  # py-cpuinfo

        name = cpuinfo.get_cpu_info().get("brand_raw", "")
    except Exception:
        try:
            if sys.platform == "win32":
                import winreg

                key = r"HARDWARE\DESCRIPTION\System\CentralProcessor\0"
                with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, key) as k:
                    name = winreg.QueryValueEx(k, "ProcessorNameString")[0]
            elif sys.platform == "darwin":
                name = subprocess.run(
                    ["sysctl", "-n", "machdep.cpu.brand_string"],
                    capture_output=True,
                    text=True,
                    check=True,
                ).stdout
            else:
                with open("/proc/cpuinfo", encoding="utf-8") as f:
                    name = next(
                        (ln.split(":", 1)[1] for ln in f if ln.startswith("model name")), ""
                    )
        except Exception:
            pass
    name = re.sub(r"\((R|TM)\)", "", name)
    return " ".join(name.split()) or platform.processor() or platform.machine()


def git_revision(path: str) -> str | None:
    """The short commit of the git checkout containing ``path``, if any."""
    try:
        out = subprocess.run(
            ["git", "-C", os.path.dirname(path), "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return out.stdout.strip() or None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--repeats", type=int, default=5, help="timed runs per task")
    parser.add_argument("--loop-repeats", type=int, default=3, help="timed BO loops")
    parser.add_argument("--no-botorch", action="store_true", help="time botorch_rs only")
    parser.add_argument("-k", default="", help="only tasks whose name contains this string")
    parser.add_argument(
        "--json", type=Path, help="also write all results to this JSON file (for the chart)"
    )
    args = parser.parse_args()
    warnings.simplefilter("ignore")

    libs = [Library("botorch_rs")]
    if not args.no_botorch:
        libs.append(Library("botorch"))
    default_threads = {lib.name: lib.default_threads for lib in libs}
    cpu = cpu_name()

    print(
        f"Python {platform.python_version()}, torch {torch.__version__}, "
        f"botorch_rs {botorch_rs.__version__}"
    )
    if not args.no_botorch:
        rev = git_revision(libs[1].file)
        print(f"BoTorch {libs[1].version} from {libs[1].file}" + (f" ({rev})" if rev else ""))
    print(
        f"{platform.platform()}, {cpu}, {os.cpu_count()} logical CPUs; default threads: "
        + ", ".join(f"{k} {v}" for k, v in default_threads.items())
    )

    def selected(lib: Library) -> list[Task]:
        return [task for task in tasks(lib) if args.k in task.name]

    def repeats_of(name: str) -> int:
        return args.loop_repeats if name.startswith(("BO loop", "MOBO loop")) else args.repeats

    runs: dict[tuple[str, str, str], tuple[list[float], list[float]]] = {}
    for setting in ("1", "default"):
        for lib in libs:
            lib.use_threads(setting == "1")
            for task in selected(lib):
                runs[(task.name, lib.name, setting)] = time_task(task.run, repeats_of(task.name))
                times = runs[(task.name, lib.name, setting)][0]
                print(
                    f"  {lib.name:10s} threads={setting:7s} {task.name:40s} "
                    f"{fmt_time(statistics.median(times))} "
                    f"(max/min {max(times) / min(times):.2f})",
                    file=sys.stderr,
                )
        for lib in libs:
            lib.use_threads(False)

    def median_time(name: str, lib: str, setting: str) -> float:
        return statistics.median(runs[(name, lib, setting)][0])

    def median_value(name: str, lib: str) -> float:
        return statistics.median(runs[(name, lib, "1")][1])

    task_list = selected(libs[0])
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
    for task in task_list:
        name = task.name
        row = [name, fmt_time(median_time(name, "botorch_rs", "1"))]
        if not args.no_botorch:
            rs, bt = median_time(name, "botorch_rs", "1"), median_time(name, "botorch", "1")
            row += [fmt_time(bt), fmt_speedup(bt / rs)]
        row += [fmt_time(median_time(name, "botorch_rs", "default"))]
        if not args.no_botorch:
            rs = median_time(name, "botorch_rs", "default")
            bt = median_time(name, "botorch", "default")
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
        for task in task_list:
            rs, bt = median_value(task.name, "botorch_rs"), median_value(task.name, "botorch")
            print(f"| {task.name} | {rs:.6f} | {bt:.6f} |")

    if args.json is not None:
        if args.no_botorch:
            parser.error("--json needs BoTorch for the comparison")
        has_loop = any(task.name.startswith(("BO loop", "MOBO loop")) for task in task_list)
        loops = f" ({args.loop_repeats} for the BO loops)" if has_loop else ""
        n_runs = f"{args.repeats} run" + ("s" if args.repeats != 1 else "")
        meta = {
            "subtitle": f"Hartmann-6, one thread each, median of {n_runs}{loops}. {cpu}.",
            "date": datetime.date.today().isoformat(),
            "repeats": args.repeats,
            "loop_repeats": args.loop_repeats,
            "python": platform.python_version(),
            "torch": torch.__version__,
            "botorch_rs": botorch_rs.__version__,
            "botorch": libs[1].version,
            "botorch_revision": git_revision(libs[1].file),
            "platform": platform.platform(),
            "cpu": cpu,
            "logical_cpus": os.cpu_count(),
            "default_threads": default_threads,
        }
        rows = []
        for task in task_list:
            name = task.name
            rows.append(
                {
                    "task": name,
                    "label": task.label,
                    "detail": task.detail,
                    # Single-thread medians (the chart).
                    "botorch_rs_s": median_time(name, "botorch_rs", "1"),
                    "botorch_s": median_time(name, "botorch", "1"),
                    "default_threads": {
                        "botorch_rs_s": median_time(name, "botorch_rs", "default"),
                        "botorch_s": median_time(name, "botorch", "default"),
                    },
                    "values": {
                        "botorch_rs": median_value(name, "botorch_rs"),
                        "botorch": median_value(name, "botorch"),
                    },
                    "runs_s": {
                        setting: {lib.name: runs[(name, lib.name, setting)][0] for lib in libs}
                        for setting in ("1", "default")
                    },
                }
            )
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps({"meta": meta, "rows": rows}, indent=2) + "\n")
        print(f"\nWrote {args.json}", file=sys.stderr)


if __name__ == "__main__":
    main()

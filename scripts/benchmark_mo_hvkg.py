#!/usr/bin/env python3
"""Wall-time comparison of the hypervolume knowledge gradient of botorch_rs
and BoTorch through the Python API.

The model is a ``ModelListGP`` of two ``SingleTaskGP`` models of the negated
DTLZ2 objectives (d = 6) on 30 points of a Weyl sequence, with fixed
hyperparameters (noise 1e-3, constant 0, lengthscales 0.5). Tasks:

* ``evaluate``: values and gradients (``torch.autograd``) of
  ``qHypervolumeKnowledgeGradient`` at 8 augmented candidate sets with
  BoTorch's defaults (``num_fantasies=8``, ``num_pareto=10``, q = 1: 81
  points per set), with the hypervolume of the posterior mean
  (``use_posterior_mean=True``) and with 32 posterior samples; both
  libraries use identical base samples (BoTorch's draws), and their values
  are checked to agree;
* ``initial conditions``: ``gen_one_shot_hvkg_initial_conditions`` (q = 1,
  ``num_fantasies=4``, ``num_pareto=5``, 4 restarts, 128 raw samples, 5
  inner restarts from 256 raw inner samples);
* ``optimize_acqf``: the same problem optimized end to end from each
  library's own one-shot initial conditions (L-BFGS-B, at most 200
  iterations); the best values (evaluated with botorch_rs on the same base
  samples) are reported.

BoTorch runs with one thread (``torch.set_num_threads(1)``); botorch_rs with
one thread and with its default number of threads (``RAYON_NUM_THREADS``
caps it). Every timing is the median over ``--repeats`` runs (different
seeds for the optimization) after a warm-up run.

Usage (``scripts/develop.py`` builds the extension)::

    PYTHONPATH=/path/to/botorch:botorch_pyo3/python python scripts/benchmark_mo_hvkg.py
"""

from __future__ import annotations

import argparse
import math
import statistics
import time
from collections.abc import Callable
from typing import Any

import torch
from botorch.acquisition.multi_objective import hypervolume_knowledge_gradient as bt_hvkg
from botorch.models import ModelListGP as BtModelListGP
from botorch.models import SingleTaskGP as BtSingleTaskGP
from botorch.optim import initializers as bt_init
from botorch.optim import optimize as bt_optimize
from botorch.sampling.list_sampler import ListSampler as BtListSampler
from botorch.sampling.normal import SobolQMCNormalSampler as BtSampler
from botorch.test_functions.multi_objective import DTLZ2

import botorch_rs
from botorch_rs.acquisition.multi_objective import hypervolume_knowledge_gradient as rs_hvkg
from botorch_rs.models import ModelListGP as RsModelListGP
from botorch_rs.models import SingleTaskGP as RsSingleTaskGP
from botorch_rs.optim import initializers as rs_init
from botorch_rs.optim import optimize as rs_optimize
from botorch_rs.sampling import ListSampler as RsListSampler
from botorch_rs.sampling import SobolQMCNormalSampler as RsSampler

DT = torch.float64
D = 6
M = 2
N = 30


def weyl(n: int, d: int = D) -> torch.Tensor:
    primes = [2, 3, 5, 7, 11, 13, 17, 19]
    alpha = torch.tensor([math.sqrt(p) % 1.0 for p in primes[:d]], dtype=DT)
    i = torch.arange(1, n + 1, dtype=DT).unsqueeze(-1)
    return torch.remainder(0.5 + i * alpha, 1.0)


def models() -> tuple[Any, Any, torch.Tensor]:
    problem = DTLZ2(dim=D, num_objectives=M, negate=True)
    X = weyl(N)
    Y = problem(X, noise=False)
    params = {
        "likelihood.noise_covar.raw_noise": torch.tensor([1e-3], dtype=DT),
        "mean_module.raw_constant": torch.tensor(0.0, dtype=DT),
        "covar_module.raw_lengthscale": torch.full((1, D), 0.5, dtype=DT),
    }
    bts, rss = [], []
    for k in range(M):
        bt, rs = BtSingleTaskGP(X, Y[:, k : k + 1]), RsSingleTaskGP(X, Y[:, k : k + 1])
        bt.initialize(**params)
        rs.initialize(**params)
        bts.append(bt)
        rss.append(rs)
    bt_list = BtModelListGP(*bts)
    bt_list.eval()
    return bt_list, RsModelListGP(*rss), torch.as_tensor(problem.ref_point, dtype=DT)


def acqf_pair(
    bt_model: Any,
    rs_model: Any,
    ref: torch.Tensor,
    n_f: int,
    n_p: int,
    use_posterior_mean: bool,
    probe: torch.Tensor,
) -> tuple[Any, Any]:
    """qHVKG in both libraries with identical base samples (BoTorch's draws
    at ``probe``)."""
    samplers = [BtSampler(sample_shape=torch.Size([n_f]), seed=k) for k in range(M)]
    inner = BtSampler(sample_shape=torch.Size([32]), seed=7)
    bt = bt_hvkg.qHypervolumeKnowledgeGradient(
        bt_model,
        ref_point=ref,
        num_fantasies=n_f,
        num_pareto=n_p,
        sampler=BtListSampler(*samplers),
        inner_sampler=inner,
        use_posterior_mean=use_posterior_mean,
    )
    with torch.no_grad():
        bt(probe)
    rs_samplers = []
    for s in samplers:
        r = RsSampler(sample_shape=(n_f,), seed=0)
        r.base_samples = s.base_samples.clone()
        rs_samplers.append(r)
    rs_inner = RsSampler(sample_shape=(32,), seed=0)
    if inner.base_samples is not None:
        rs_inner.base_samples = inner.base_samples.clone()
    rs = rs_hvkg.qHypervolumeKnowledgeGradient(
        rs_model,
        ref_point=ref,
        num_fantasies=n_f,
        num_pareto=n_p,
        sampler=RsListSampler(*rs_samplers),
        inner_sampler=rs_inner,
        use_posterior_mean=use_posterior_mean,
    )
    return bt, rs


def value_and_grad(acqf: Any, X: torch.Tensor) -> torch.Tensor:
    Xg = X.clone().requires_grad_(True)
    values = acqf(Xg)
    values.sum().backward()
    return values.detach()


def median_time(fn: Callable[[int], Any], repeats: int, min_time: float) -> float:
    """The median wall time of ``fn(seed)`` over ``repeats`` runs after a
    warm-up run; fast calls are repeated within a run until it takes
    ``min_time`` seconds."""
    t = time.perf_counter()
    fn(0)
    calls = max(1, math.ceil(min_time / max(time.perf_counter() - t, 1e-9)))
    times = []
    for r in range(repeats):
        t = time.perf_counter()
        for _ in range(calls):
            fn(r)
        times.append((time.perf_counter() - t) / calls)
    return statistics.median(times)


def fmt(seconds: float) -> str:
    if seconds < 1e-3:
        return f"{seconds * 1e6:.0f} us"
    if seconds < 1.0:
        return f"{seconds * 1e3:.1f} ms"
    return f"{seconds:.2f} s"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--min-time", type=float, default=0.2)
    args = parser.parse_args()
    torch.set_num_threads(1)
    bt_model, rs_model, ref = models()
    bounds = torch.stack([torch.zeros(D, dtype=DT), torch.ones(D, dtype=DT)])
    rows: list[tuple[str, float, float, float]] = []

    def time_rs(fn: Callable[[int], Any]) -> tuple[float, float]:
        botorch_rs.set_num_threads(1)
        single = median_time(fn, args.repeats, args.min_time)
        botorch_rs.set_num_threads(None)
        default = median_time(fn, args.repeats, args.min_time)
        return single, default

    # Values and gradients at 8 augmented candidate sets (BoTorch's defaults).
    X = torch.rand(8, 1 + 8 * 10, D, generator=torch.Generator().manual_seed(0), dtype=DT)
    for upm, label in [(True, "posterior mean"), (False, "32 posterior samples")]:
        bt, rs = acqf_pair(bt_model, rs_model, ref, 8, 10, upm, X)
        v_rs = value_and_grad(rs, X)
        # BoTorch couples the t-batches (see docs/drafts/mo-hvkg.md): compare
        # with its values of the single candidate sets.
        with torch.no_grad():
            single = torch.cat([bt(X[b : b + 1]) for b in range(X.shape[0])])
        err = ((v_rs - single).abs() / single.abs().clamp_min(1e-12)).max().item()
        t_bt = median_time(lambda _, f=bt: value_and_grad(f, X), args.repeats, args.min_time)
        t_rs1, t_rs = time_rs(lambda _, f=rs: value_and_grad(f, X))
        rows.append(
            (f"evaluate qHVKG + gradients, {label} (max rel. diff {err:.1e})", t_bt, t_rs1, t_rs)
        )

    # One-shot initial conditions and optimize_acqf (smaller problem).
    probe = torch.rand(1, 1 + 4 * 5, D, generator=torch.Generator().manual_seed(1), dtype=DT)
    bt, rs = acqf_pair(bt_model, rs_model, ref, 4, 5, True, probe)
    opts = {"num_inner_restarts": 5, "raw_inner_samples": 256}

    def ics_bt(seed: int) -> Any:
        torch.manual_seed(seed)
        return bt_init.gen_one_shot_hvkg_initial_conditions(
            bt, bounds, q=1, num_restarts=4, raw_samples=128, options=dict(opts)
        )

    def ics_rs(seed: int) -> Any:
        return rs_init.gen_one_shot_hvkg_initial_conditions(
            rs, bounds, q=1, num_restarts=4, raw_samples=128, options=dict(opts, seed=seed)
        )

    t_bt = median_time(ics_bt, args.repeats, args.min_time)
    t_rs1, t_rs = time_rs(ics_rs)
    rows.append(("gen_one_shot_hvkg_initial_conditions (4 x 21 x 6)", t_bt, t_rs1, t_rs))

    best: dict[str, list[float]] = {}

    def opt_bt(seed: int, batch_limit: int | None = None) -> Any:
        torch.manual_seed(seed)
        options: dict[str, Any] = dict(opts)
        if batch_limit is not None:
            options["batch_limit"] = batch_limit
        full, _ = bt_optimize.optimize_acqf(
            bt,
            bounds,
            q=1,
            num_restarts=4,
            raw_samples=128,
            options=options,
            return_full_tree=True,
        )
        with torch.no_grad():
            key = f"BoTorch (batch_limit={batch_limit})"
            best.setdefault(key, []).append(float(rs(full.unsqueeze(0))))

    def opt_rs(seed: int) -> Any:
        full, _ = rs_optimize.optimize_acqf(
            rs,
            bounds,
            q=1,
            num_restarts=4,
            raw_samples=128,
            options=dict(opts, seed=seed),
            return_full_tree=True,
        )
        with torch.no_grad():
            best.setdefault("botorch_rs", []).append(float(rs(torch.as_tensor(full).unsqueeze(0))))

    t_bt = median_time(opt_bt, args.repeats, 0.0)
    t_rs1, t_rs = time_rs(opt_rs)
    rows.append(("optimize_acqf qHVKG (q = 1, 4 fantasies x 5 designs)", t_bt, t_rs1, t_rs))
    # BoTorch optimizes the restarts jointly by default, which mixes their
    # fantasy designs; one restart per batch optimizes the true objective.
    t_bt = median_time(lambda seed: opt_bt(seed, 1), args.repeats, 0.0)
    rows.append(("optimize_acqf qHVKG, BoTorch with batch_limit=1", t_bt, t_rs1, t_rs))

    print(f"BoTorch {__import__('botorch').__version__}, botorch_rs {botorch_rs.__version__}")
    print(f"botorch_rs default threads: {botorch_rs.get_num_threads()}\n")
    print("| task | BoTorch (1 thread) | botorch_rs (1 thread) | botorch_rs (default) | speedup |")
    print("|---|---:|---:|---:|---:|")
    for name, t_bt, t_rs1, t_rs in rows:
        print(
            f"| {name} | {fmt(t_bt)} | {fmt(t_rs1)} | {fmt(t_rs)} | {t_bt / t_rs1:.1f}x / {t_bt / t_rs:.1f}x |"
        )
    for lib, values in best.items():
        print(
            f"\nbest qHVKG value found by {lib} (per candidate set): median {statistics.median(values):.6g} "
            f"(min {min(values):.6g}, max {max(values):.6g}, {len(values)} runs)"
        )


if __name__ == "__main__":
    main()

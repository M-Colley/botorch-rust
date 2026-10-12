"""A complete multi-objective Bayesian optimization loop on BraninCurrin: the
Python version of ``botorch_core/examples/mobo_loop.rs`` (6 Sobol points,
then iterations of fitting a multi-output SingleTaskGP with ``Normalize`` and
``Standardize`` and maximizing qLogNEHVI for ``q = 2`` candidates with
``sequential=True``), compared statistically with the same loop in BoTorch.

The loops are short so that the tests fit into CI (6 iterations, 4
restarts and 128 raw samples per optimization); the example runs the
tutorial's settings (20 iterations, 10 restarts, 512 raw samples).
"""

from __future__ import annotations

import statistics
import warnings
from typing import Any

import pytest

from botorch_rs.acquisition.multi_objective import qLogNoisyExpectedHypervolumeImprovement
from botorch_rs.fit import fit_gpytorch_mll
from botorch_rs.mlls import ExactMarginalLogLikelihood
from botorch_rs.models import SingleTaskGP
from botorch_rs.models.transforms.input import Normalize
from botorch_rs.optim import optimize_acqf
from botorch_rs.sampling import SobolQMCNormalSampler
from botorch_rs.test_functions.multi_objective import BraninCurrin
from botorch_rs.utils.multi_objective.box_decompositions import DominatedPartitioning
from botorch_rs.utils.sampling import draw_sobol_samples
from helpers import KINDS, ParityLog, as_kind, flatten, importorskip_botorch, to_list

D, M, Q = 2, 2, 2
N_INIT = 2 * (D + 1)
ITERATIONS = 6
SEEDS = range(4)
BOUNDS = [[0.0] * D, [1.0] * D]
SETTINGS = {"num_restarts": 4, "raw_samples": 128, "sequential": True}
SECTION = "multi-objective BO loop (BraninCurrin, hypervolume)"
MAX_HV = BraninCurrin(negate=True).max_hv


def initial_design(seed: int) -> list[list[float]]:
    return [x[0] for x in to_list(draw_sobol_samples(BOUNDS, n=N_INIT, q=1, seed=seed))]


def hypervolume(ref_point: Any, Y: Any) -> float:
    return float(flatten(DominatedPartitioning(ref_point=ref_point, Y=Y).compute_hypervolume())[0])


def rows(x: Any) -> list[list[float]]:
    return [[float(v) for v in row] for row in to_list(x)]


def mobo_loop(seed: int, kind: str = "list", iterations: int = ITERATIONS) -> list[float]:
    """botorch_rs: the hypervolume of the observations after every iteration
    (index 0: the initial design)."""
    problem = BraninCurrin(negate=True)
    bounds = as_kind(BOUNDS, kind)
    ref = problem.ref_point
    X = as_kind(initial_design(seed), kind)
    Y = problem(X, noise=False)
    trace = [hypervolume(ref, Y)]
    for it in range(iterations):
        model = SingleTaskGP(X, Y, input_transform=Normalize(d=D, bounds=bounds))
        fit_gpytorch_mll(ExactMarginalLogLikelihood(model.likelihood, model))
        acqf = qLogNoisyExpectedHypervolumeImprovement(
            model,
            ref_point=ref,
            X_baseline=X,
            prune_baseline=True,
            sampler=SobolQMCNormalSampler(sample_shape=[128], seed=seed * 7919 + it),
        )
        candidates, _ = optimize_acqf(
            acqf, bounds, q=Q, options={"seed": seed * 1000 + it}, **SETTINGS
        )
        X = as_kind(rows(X) + rows(candidates), kind)
        Y = as_kind(rows(Y) + rows(problem(candidates, noise=False)), kind)
        trace.append(hypervolume(ref, Y))
    return trace


def bt_mobo_loop(seed: int, iterations: int = ITERATIONS) -> list[float]:
    """The same loop in BoTorch (same initial design and settings)."""
    import torch
    from botorch.acquisition.multi_objective.logei import (
        qLogNoisyExpectedHypervolumeImprovement as BtQLogNEHVI,
    )
    from botorch.fit import fit_gpytorch_mll as bt_fit
    from botorch.models import SingleTaskGP as BtSingleTaskGP
    from botorch.models.transforms.input import Normalize as BtNormalize
    from botorch.optim import optimize_acqf as bt_optimize_acqf
    from botorch.sampling.normal import SobolQMCNormalSampler as BtSampler
    from botorch.test_functions.multi_objective import BraninCurrin as BtBraninCurrin
    from botorch.utils.multi_objective.box_decompositions import (
        DominatedPartitioning as BtDominatedPartitioning,
    )
    from gpytorch.mlls import ExactMarginalLogLikelihood as BtMLL

    dt = torch.float64
    problem = BtBraninCurrin(negate=True).to(dt)
    bounds = torch.tensor(BOUNDS, dtype=dt)
    X = torch.tensor(initial_design(seed), dtype=dt)
    Y = problem(X)

    def hv(Y: Any) -> float:
        return (
            BtDominatedPartitioning(ref_point=problem.ref_point, Y=Y).compute_hypervolume().item()
        )

    trace = [hv(Y)]
    for it in range(iterations):
        torch.manual_seed(seed * 1000 + it)
        model = BtSingleTaskGP(X, Y, input_transform=BtNormalize(d=D, bounds=bounds))
        bt_fit(BtMLL(model.likelihood, model))
        acqf = BtQLogNEHVI(
            model,
            ref_point=problem.ref_point,
            X_baseline=X,
            prune_baseline=True,
            sampler=BtSampler(sample_shape=torch.Size([128]), seed=seed * 7919 + it),
        )
        candidates, _ = bt_optimize_acqf(
            acqf, bounds, q=Q, options={"seed": seed * 1000 + it}, **SETTINGS
        )
        X = torch.cat([X, candidates])
        Y = torch.cat([Y, problem(candidates)])
        trace.append(hv(Y))
    return trace


def test_mobo_loop_improves_the_hypervolume() -> None:
    """The hypervolume never decreases, every run improves on its initial
    design, and after 6 iterations the median final hypervolume exceeds 80%
    of the maximum (59.4066; the example reaches 97% after 20 iterations)."""
    finals = []
    for seed in SEEDS:
        trace = mobo_loop(seed)
        assert all(b >= a for a, b in zip(trace, trace[1:], strict=False)), trace
        assert trace[-1] > trace[0], trace
        finals.append(trace[-1])
    assert statistics.median(finals) > 0.8 * MAX_HV, finals


def test_mobo_loop_is_reproducible_and_independent_of_the_array_type() -> None:
    traces = [mobo_loop(0, kind, iterations=2) for kind in KINDS]
    for trace in traces[1:]:
        assert trace == traces[0]


def test_mobo_loop_matches_botorch_statistically() -> None:
    """Over the same initial designs, both libraries improve on every design
    and botorch_rs' median shortfall against BoTorch's final hypervolume is
    at most 5% of the maximum hypervolume (a guard against gross failures:
    the comparison of four short loops has little statistical power).

    Individual runs differ: the acquisition values agree to rounding (see
    the parity tests), but the optimizers' random initial conditions differ,
    and 6-iteration loops have heavy tails. Over 24 designs (BoTorch main,
    Windows) the final hypervolumes average 50.54 (botorch_rs) and 49.90
    (BoTorch; medians 50.85 and 50.69); the shortfall of botorch_rs is
    -1.1% +- 1.2% (mean +- standard error) with a median of 0.02%, single
    runs differ by -21% (a BoTorch run that stalled at 35.7) to +6.4%, and
    the per-run shortfalls have a standard deviation of 5.7%."""
    importorskip_botorch()
    pytest.importorskip("torch")
    shortfalls = []
    for seed in SEEDS:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            bt = bt_mobo_loop(seed)
        rs = mobo_loop(seed)
        assert rs[0] == pytest.approx(bt[0], rel=1e-12)
        assert bt[-1] > bt[0] and rs[-1] > rs[0], (rs, bt)
        shortfall = (bt[-1] - rs[-1]) / MAX_HV
        ParityLog.record(
            "final hypervolume shortfall vs BoTorch (per run, / max HV)",
            shortfall,
            section=SECTION,
        )
        shortfalls.append(shortfall)
    median = statistics.median(shortfalls)
    ParityLog.record(
        "final hypervolume shortfall vs BoTorch (median, / max HV)",
        median,
        section=SECTION,
        tol=0.05,
    )
    assert median <= 0.05, shortfalls

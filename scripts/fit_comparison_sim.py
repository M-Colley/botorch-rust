#!/usr/bin/env python3
"""Monte Carlo of the two-stage comparison of fits with BoTorch
(``botorch_pyo3/tests/parity_helpers.py::compare_twin_fits``,
``botorch_core/tests/common/mod.rs::compare_fits``; ``docs/fitting.md``,
"Comparing fits"): how often it fails by chance when botorch_rs's outcomes
are another sample of BoTorch's (false positives), and how often it fails
when botorch_rs's outcomes are worse in distribution (power).

Stage 1 compares ``n1`` outcomes per library (5: the live tests with 4
twins; 17: the Rust fixture tests and ``test_parity.py`` with 16 twins):
botorch_rs's best must reach BoTorch's best, and none of its outcomes may
end above BoTorch's worst (``every``; the former one-sample bound checked
only its outcome on the model itself, ``own``), within ``max(1e-6, 10 x``
BoTorch's spread at its best optimum``)``. Stage 2 compares 65 outcomes per
library: a one-sided Mann-Whitney test at level 1e-3 (ties within 1e-6) and
botorch_rs's best against BoTorch's 10-th best. The procedure fails if both
stages fail. Outcomes are drawn independently (the positive dependence of a
twin pair, fitted on the same perturbed inputs, makes both stages more
conservative).

Losses are relative to the best optimum. Usage (NumPy only)::

    python scripts/fit_comparison_sim.py [replications_null] [replications_regressions]
"""

from __future__ import annotations

import math
import sys
from collections.abc import Callable

import numpy as np

FIT_TOL = 1e-6
FIT_CLUSTER = 1e-4
FIT_Z = 3.090232306167813
N2 = 65

Sampler = Callable[[np.random.Generator, int, int], np.ndarray]


def rank_bound_k(n_rs: int, n_bt: int, alpha: float = 1e-3) -> int:
    p = 1.0
    for k in range(1, n_bt + 1):
        p *= (n_bt - k + 1) / (n_rs + n_bt - k + 1)
        if p <= alpha:
            return k
    return n_bt + 1


K = rank_bound_k(N2, N2)


def stage1_fails(B: np.ndarray, R: np.ndarray, every: bool) -> np.ndarray:
    """Stage 1 on ``reps x n1`` outcomes (column 0: the model itself)."""
    best, worst = B.min(1), B.max(1)
    spread = np.where(B <= best[:, None] + FIT_CLUSTER, B, -np.inf).max(1) - best
    tol = np.maximum(FIT_TOL, 10 * spread)
    top = R.max(1) if every else R[:, 0]
    return (R.min(1) - best > tol) | (top - worst > tol)


def stage2_fails(B: np.ndarray, R: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Stage 2 on ``reps x 65`` outcomes: failure, Mann-Whitney, rank bound."""
    d = R[:, :, None] - B[:, None, :]
    u = (d > FIT_TOL).sum((1, 2)) + 0.5 * (np.abs(d) <= FIT_TOL).sum((1, 2))
    n1, n2 = R.shape[1], B.shape[1]
    z = (u - n1 * n2 / 2 - 0.5) / math.sqrt(n1 * n2 * (n1 + n2 + 1) / 12)
    mw = z > FIT_Z
    rank = R.min(1) - np.sort(B, 1)[:, K - 1] > FIT_TOL
    return mw | rank, mw, rank


def tail(p: float) -> Sampler:
    """Most fits at the optimum (spread ~1e-8), a fraction ``p`` stopped
    early on a flat region 2.1e-4 .. 2.3e-4 above it (6b
    ``warp_priors_bounds`` on Linux: BoTorch 6 of 65, botorch_rs 4 of 65)."""

    def draw(rng: np.random.Generator, reps: int, n: int) -> np.ndarray:
        at_opt = np.abs(rng.normal(0, 3e-8, (reps, n)))
        early = 2.1e-4 + rng.uniform(0, 2e-5, (reps, n))
        return np.where(rng.random((reps, n)) < p, early, at_opt)

    return draw


def tight(offset: float = 0.0) -> Sampler:
    """One optimum, BoTorch's spread ~1e-9 (most fit cases)."""

    def draw(rng: np.random.Generator, reps: int, n: int) -> np.ndarray:
        return offset + np.abs(rng.normal(0, 1e-9, (reps, n)))

    return draw


def chaotic(scale: float) -> Sampler:
    """Outcomes spread continuously over ~``scale`` (e.g. 7a ``default_t3_d2``)."""

    def draw(rng: np.random.Generator, reps: int, n: int) -> np.ndarray:
        return rng.exponential(scale, (reps, n))

    return draw


def modes(probs: list[float]) -> Sampler:
    """Four optima (0, 2e-4, 5e-4, 2e-3) with probabilities ``probs``, plus
    noise of 3e-7 (ties at FIT_TOL)."""
    levels = np.array([0.0, 2e-4, 5e-4, 2e-3])

    def draw(rng: np.random.Generator, reps: int, n: int) -> np.ndarray:
        idx = rng.choice(4, size=(reps, n), p=probs)
        return levels[idx] + np.abs(rng.normal(0, 3e-7, (reps, n)))

    return draw


def degraded(rng: np.random.Generator, reps: int, n: int) -> np.ndarray:
    """Every fit stops far above the optimum (``maxiter = 5``: 8e-3 .. 0.12
    above for ``test_parity.py``'s ``default`` and ``legacy_gamma``)."""
    return 0.01 + rng.exponential(0.03, (reps, n))


def run(name: str, f_bt: Sampler, f_rs: Sampler, reps: int, n1: int, seed: int) -> None:
    rng = np.random.default_rng(seed)
    counts = np.zeros(7)
    done = 0
    while done < reps:
        m = min(2000, reps - done)
        B, R = f_bt(rng, m, N2), f_rs(rng, m, N2)
        own = stage1_fails(B[:, :n1], R[:, :n1], every=False)
        every = stage1_fails(B[:, :n1], R[:, :n1], every=True)
        s2, mw, rank = stage2_fails(B, R)
        for i, v in enumerate([own, every, s2, mw, rank, own & s2, every & s2]):
            counts[i] += v.sum()
        done += m
    r = counts / reps
    print(
        f"{name:42s} n1={n1:2d} | stage 1 fails: own {r[0]:7.2%}, every {r[1]:7.2%} | "
        f"stage 2 alone {r[2]:7.3%} (MW {r[3]:7.3%}, rank {r[4]:7.3%}) | "
        f"two-stage fails: own {r[5]:7.3%}, every {r[6]:7.3%}"
    )


def main() -> None:
    reps_null = int(sys.argv[1]) if len(sys.argv) > 1 else 200_000
    reps_alt = int(sys.argv[2]) if len(sys.argv) > 2 else 20_000
    print(f"k = {K}; {reps_null} replications under the null, {reps_alt} for regressions")
    print("== null hypothesis: botorch_rs's outcomes another sample of BoTorch's")
    nulls: list[tuple[str, Sampler]] = [
        ("one optimum (spread 1e-9)", tight()),
        ("tail of early stops 8% (Linux, 6b warp)", tail(6 / 65)),
        ("tail of early stops 25%", tail(0.25)),
        ("chaotic (exponential, scale 1e-3)", chaotic(1e-3)),
        ("four optima 30/30/20/20%", modes([0.3, 0.3, 0.2, 0.2])),
        ("four optima, best rare 5/45/30/20%", modes([0.05, 0.45, 0.3, 0.2])),
    ]
    for i, (name, f) in enumerate(nulls):
        for n1 in (5, 17):
            run(name, f, f, reps_null, n1, 100 + 10 * i + n1)
    print("== regressions: botorch_rs's outcomes worse in distribution")
    alts: list[tuple[str, Sampler, Sampler]] = [
        ("tail 8% -> 20%", tail(6 / 65), tail(0.2)),
        ("tail 8% -> 40%", tail(6 / 65), tail(0.4)),
        ("tail 8% -> 70%", tail(6 / 65), tail(0.7)),
        ("tail 8% -> 100% (optimum never reached)", tail(6 / 65), tail(1.0)),
        ("optimum 2e-6 higher", tight(), tight(2e-6)),
        (
            "four optima 30/30/20/20% -> 10/30/30/30%",
            modes([0.3, 0.3, 0.2, 0.2]),
            modes([0.1, 0.3, 0.3, 0.3]),
        ),
        (
            "four optima 30/30/20/20% -> 0/40/30/30%",
            modes([0.3, 0.3, 0.2, 0.2]),
            modes([0.0, 0.4, 0.3, 0.3]),
        ),
        ("chaotic, scale 1e-3 -> 1.5e-3", chaotic(1e-3), chaotic(1.5e-3)),
        ("chaotic, scale 1e-3 -> 2e-3", chaotic(1e-3), chaotic(2e-3)),
        ("degraded fits (maxiter = 5)", tight(), degraded),
    ]
    for i, (name, fb, fr) in enumerate(alts):
        for n1 in (5, 17):
            run(name, fb, fr, reps_alt, n1, 1000 + 10 * i + n1)


if __name__ == "__main__":
    main()

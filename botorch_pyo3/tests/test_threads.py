"""Thread control and release of the GIL during Rust work."""

from __future__ import annotations

import sys
import threading

import botorch_rs
from botorch_rs.acquisition import qLogExpectedImprovement
from botorch_rs.fit import fit_gpytorch_mll
from botorch_rs.mlls import ExactMarginalLogLikelihood
from botorch_rs.models import SingleTaskGP
from botorch_rs.optim import optimize_acqf
from botorch_rs.sampling import SobolQMCNormalSampler
from helpers import make_data


def _problem() -> tuple[qLogExpectedImprovement, list]:
    X, Y = make_data(n=20, d=4, seed=3)
    model = SingleTaskGP(X, Y)
    fit_gpytorch_mll(ExactMarginalLogLikelihood(model.likelihood, model))
    sampler = SobolQMCNormalSampler(sample_shape=[128], seed=0)
    acqf = qLogExpectedImprovement(model, best_f=max(y[0] for y in Y), sampler=sampler)
    return acqf, [[0.0] * 4, [1.0] * 4]


def test_num_threads_do_not_change_results() -> None:
    acqf, bounds = _problem()
    default = botorch_rs.get_num_threads()
    assert default >= 1
    try:
        botorch_rs.set_num_threads(1)
        assert botorch_rs.get_num_threads() == 1
        one = optimize_acqf(acqf, bounds, q=2, num_restarts=6, raw_samples=64, options={"seed": 0})
        botorch_rs.set_num_threads(3)
        assert botorch_rs.get_num_threads() == 3
        three = optimize_acqf(
            acqf, bounds, q=2, num_restarts=6, raw_samples=64, options={"seed": 0}
        )
    finally:
        botorch_rs.set_num_threads(None)
    assert botorch_rs.get_num_threads() == default
    assert one == three


def test_optimize_releases_the_gil() -> None:
    """The worker runs Rust computations back to back; the main thread can set
    a flag the worker observes only if the GIL is released inside them (the
    switch interval is raised so that it cannot force a switch)."""
    acqf, bounds = _problem()
    old_interval = sys.getswitchinterval()
    sys.setswitchinterval(5.0)
    try:
        started = threading.Event()
        progress: dict[str, object] = {"flag": False, "seen_at": None}

        def worker() -> None:
            for i in range(200):
                optimize_acqf(
                    acqf, bounds, q=2, num_restarts=2, raw_samples=16, options={"seed": i}
                )
                started.set()
                if progress["flag"] and progress["seen_at"] is None:
                    progress["seen_at"] = i

        thread = threading.Thread(target=worker, daemon=True)
        thread.start()
        assert started.wait(timeout=60)
        progress["flag"] = True
        thread.join(timeout=120)
        assert not thread.is_alive()
        assert progress["seen_at"] is not None
    finally:
        sys.setswitchinterval(old_interval)

#!/usr/bin/env python3
"""Time BoTorch's ``optimize_acqf`` on the benchmark problem of
``botorch_core/examples/benchmark_optimize_acqf.rs``.

The model is identical in both implementations (no random numbers are
involved): 50 training inputs from a Weyl sequence in [0, 1]^6, targets
``-Hartmann6(x)``, a default ``SingleTaskGP`` (Standardize, RBF kernel) with
fixed hyperparameters (noise 1e-3, constant 0, lengthscales 0.5). Timed:

* ``optimize_acqf(LogExpectedImprovement, q=1, num_restarts=10, raw_samples=512)``
* ``optimize_acqf(qLogExpectedImprovement (512 Sobol samples), q=3, num_restarts=10,
  raw_samples=512)``

with ``torch.set_num_threads(1)`` and with torch's default number of threads.

Usage::

    PYTHONPATH=/path/to/botorch python -P scripts/benchmark_optimize_acqf.py [repeats]
"""

from __future__ import annotations

import math
import sys
import time
import warnings

import torch

DT = torch.float64


def weyl(n: int, d: int) -> torch.Tensor:
    primes = [2, 3, 5, 7, 11, 13, 17, 19]
    alpha = torch.tensor([math.sqrt(p) % 1.0 for p in primes[:d]], dtype=DT)
    i = torch.arange(1, n + 1, dtype=DT).unsqueeze(-1)
    return torch.remainder(0.5 + i * alpha, 1.0)


def main() -> None:
    import botorch
    from botorch.acquisition import LogExpectedImprovement
    from botorch.acquisition.logei import qLogExpectedImprovement
    from botorch.models import SingleTaskGP
    from botorch.optim import optimize_acqf
    from botorch.sampling.normal import SobolQMCNormalSampler
    from botorch.test_functions import Hartmann

    warnings.simplefilter("ignore")
    repeats = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    print(f"botorch from {botorch.__file__}; torch {torch.__version__}")
    X = weyl(50, 6)
    Y = Hartmann(dim=6, negate=True)(X, noise=False).unsqueeze(-1)
    model = SingleTaskGP(X, Y)
    with torch.no_grad():
        model.likelihood.noise_covar.raw_noise.fill_(1e-3)
        model.mean_module.raw_constant.fill_(0.0)
        model.covar_module.raw_lengthscale.fill_(0.5)
    model.eval()
    bounds = torch.stack([torch.zeros(6, dtype=DT), torch.ones(6, dtype=DT)])
    best_f = Y.max()

    def bench(name, make_acqf, q, threads):
        torch.set_num_threads(threads)
        times, values = [], []
        for s in range(repeats):
            acqf = make_acqf()
            t = time.perf_counter()
            _, v = optimize_acqf(acqf, bounds, q=q, num_restarts=10, raw_samples=512, options={"seed": s})
            times.append(time.perf_counter() - t)
            values.append(v.item())
        times.sort()
        print(
            f"{name:28s} threads={threads:2d}: median {times[len(times) // 2]:8.3f} s "
            f"(min {times[0]:.3f}, max {times[-1]:.3f}); best value {max(values):.6f}"
        )

    default_threads = torch.get_num_threads()
    for threads in (1, default_threads):
        bench("LogEI q=1", lambda: LogExpectedImprovement(model, best_f=best_f), 1, threads)
        bench(
            "qLogEI q=3 (512 samples)",
            lambda: qLogExpectedImprovement(
                model, best_f=best_f, sampler=SobolQMCNormalSampler(torch.Size([512]), seed=0)
            ),
            3,
            threads,
        )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Compare cached LOO diagnostic evaluation against official BoTorch.

Build the extension in release mode first with ``python scripts/develop.py``.
This benchmark uses identical float64 models, one torch/Rust worker and no
hyperparameter autograd. It verifies the means and variances before timing.
It measures evaluation of existing fitted models, without fitting or imports.
"""

from __future__ import annotations

import argparse
import statistics
import time
import warnings

import torch
from botorch.cross_validation import efficient_loo_cv as bt_loo
from botorch.models import SingleTaskGP as BtGP

import botorch_rs
from botorch_rs.cross_validation import efficient_loo_cv as rs_loo
from botorch_rs.models import SingleTaskGP as RsGP


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--sizes", nargs="+", type=int, default=[16, 64, 128, 256])
    parser.add_argument("--repeats", type=int, default=15)
    args = parser.parse_args()
    if args.repeats < 1 or any(n < 1 for n in args.sizes):
        parser.error("sizes and repeats must be positive")
    torch.set_num_threads(1)
    botorch_rs.set_num_threads(1)
    generator = torch.Generator().manual_seed(17)
    warnings.simplefilter("ignore")
    print("Single-thread cached LOO, float64, no hyperparameter autograd")
    print("n       BoTorch ms     Rust ms     ratio")
    with torch.no_grad():
        for n in args.sizes:
            x = torch.rand(n, 3, generator=generator, dtype=torch.float64)
            y = (6 * x[:, :1]).sin() + x[:, 1:2]
            models = [cls(x, y, outcome_transform=None).eval() for cls in (BtGP, RsGP)]
            for model in models:
                model.covar_module.lengthscale = torch.tensor(0.4, dtype=torch.float64)
                model.likelihood.noise = torch.tensor(0.02, dtype=torch.float64)
            outputs = [fun(model) for fun, model in zip((bt_loo, rs_loo), models, strict=True)]
            for name in ("mean", "variance"):
                torch.testing.assert_close(
                    getattr(outputs[0].posterior, name),
                    getattr(outputs[1].posterior, name),
                    rtol=1e-9,
                    atol=1e-10,
                )
            timings = [[], []]
            # Alternate order to reduce drift and warmed-cache ordering bias.
            for repeat in range(args.repeats):
                for index in (0, 1) if repeat % 2 == 0 else (1, 0):
                    fun, model = (bt_loo, models[0]) if index == 0 else (rs_loo, models[1])
                    start = time.perf_counter()
                    result = fun(model)
                    _ = result.posterior.mean, result.posterior.variance
                    timings[index].append(time.perf_counter() - start)
            bt, rs = [statistics.median(values) * 1000 for values in timings]
            print(f"{n:<6d}  {bt:10.4f}  {rs:10.4f}  {bt / rs:7.2f}x")


if __name__ == "__main__":
    main()

"""Reproduce end-to-end cached / joint qLogNEI forward timings.

Example (put the port and official checkout on PYTHONPATH first)::

    python scripts/benchmark_noisy_cache.py --rust-profile release --output cache.json

The default matches the investigation workload: float64 CPU, one worker,
32 training points, 180 baseline points, 8 candidate sets with q=2, and
128 MC samples. Model construction, baseline factorization, and one warmup
call are excluded. Seven complete public-API forwards are timed under
torch.no_grad; Rust array conversion and reference Python overhead remain
included. This measures forwards, not fitting, gradients, or GPU work.
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import subprocess
import time
import warnings
from pathlib import Path

import botorch
import botorch_rs
import torch
from botorch.acquisition.logei import qLogNoisyExpectedImprovement as ReferenceNEI
from botorch.models import SingleTaskGP as ReferenceGP
from botorch.sampling.normal import SobolQMCNormalSampler as ReferenceSampler
from botorch_rs.acquisition.logei import qLogNoisyExpectedImprovement as RustNEI
from botorch_rs.models import SingleTaskGP as RustGP
from botorch_rs.sampling.normal import SobolQMCNormalSampler as RustSampler


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name, default in (
        ("train", 32),
        ("baseline", 180),
        ("batches", 8),
        ("q", 2),
        ("samples", 128),
        ("threads", 1),
        ("warmup", 1),
        ("repeats", 7),
    ):
        parser.add_argument(f"--{name}", type=int, default=default)
    parser.add_argument("--rust-profile", default="unspecified")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if any(
        getattr(args, name) < 1
        for name in (
            "train",
            "baseline",
            "batches",
            "q",
            "samples",
            "threads",
            "warmup",
            "repeats",
        )
    ):
        parser.error(
            "all workload sizes, thread counts, warmups and repeats must be positive"
        )
    torch.set_num_threads(args.threads)
    botorch_rs.set_num_threads(args.threads)
    warnings.simplefilter("ignore")
    dtype = torch.float64
    train_x = torch.linspace(0, 1, args.train, dtype=dtype).reshape(-1, 1)
    train_y = torch.sin(6 * train_x)
    baseline = torch.linspace(0, 1, args.baseline, dtype=dtype).reshape(-1, 1)
    candidates = torch.linspace(0.1, 0.9, args.batches * args.q, dtype=dtype).reshape(
        args.batches, args.q, 1
    )
    results = []
    for name, model_class, acq_class, sampler_class in (
        ("rust", RustGP, RustNEI, RustSampler),
        ("reference", ReferenceGP, ReferenceNEI, ReferenceSampler),
    ):
        model = model_class(
            train_x, train_y, torch.full_like(train_y, 0.02), outcome_transform=None
        ).eval()
        for cache_root in (False, True):
            acquisition = acq_class(
                model,
                baseline,
                sampler=sampler_class(torch.Size([args.samples]), seed=7),
                prune_baseline=False,
                cache_root=cache_root,
            )
            with torch.no_grad():
                for _ in range(args.warmup):
                    acquisition(candidates)
                elapsed = []
                for _ in range(args.repeats):
                    start = time.perf_counter()
                    value = acquisition(candidates)
                    elapsed.append((time.perf_counter() - start) * 1000)
            assert torch.isfinite(value).all()
            results.append(
                {
                    "backend": name,
                    "cache_root": cache_root,
                    "median_ms": statistics.median(elapsed),
                    "times_ms": elapsed,
                }
            )
    revision = subprocess.run(
        [
            "git",
            "-C",
            str(Path(botorch.__file__).resolve().parents[1]),
            "rev-parse",
            "HEAD",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    report = {
        "parameters": {
            key: value for key, value in vars(args).items() if key != "output"
        },
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "botorch": botorch.__version__,
        "botorch_path": botorch.__file__,
        "reference_revision": revision.stdout.strip()
        if revision.returncode == 0
        else None,
        "botorch_rs": botorch_rs.__version__,
        "results": results,
    }
    output = json.dumps(report, indent=2)
    print(output)
    if args.output is not None:
        args.output.write_text(output + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

"""Measure warmed exact-GP prediction with synchronized CUDA timings.

Run after builds and tests finish, with PYTHONPATH pointing at the installed
wheel being measured. Construction and the first training factor are excluded.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import random
import statistics
import time
from datetime import UTC, datetime
from pathlib import Path

os.environ.setdefault("RAYON_NUM_THREADS", "1")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sizes", type=int, nargs="+", default=[32, 256, 1024])
    parser.add_argument("--dtype", choices=["float32", "float64"], default="float64")
    parser.add_argument("--repeats", type=int, default=31)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--rust-artifact", type=Path, required=True)
    args = parser.parse_args()

    import torch
    import botorch

    from botorch.models import SingleTaskGP as ReferenceGP
    from botorch_rs import _botorch_rs
    from botorch_rs.models import SingleTaskGP

    torch.set_num_threads(1)
    actual_hash = hashlib.sha256(Path(_botorch_rs.__file__).read_bytes()).hexdigest()
    artifact_hash = hashlib.sha256(args.rust_artifact.read_bytes()).hexdigest()
    if actual_hash != artifact_hash:
        raise RuntimeError(
            "Loaded extension differs from --rust-artifact; choose the installed release wheel."
        )
    if args.repeats < 3 or args.warmup < 1 or any(size < 2 for size in args.sizes):
        raise ValueError(
            "Use at least three repetitions, one warmup, and two training points."
        )
    dtype = getattr(torch, args.dtype)
    devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
    report = {
        "created_utc": datetime.now(UTC).isoformat(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "botorch": botorch.__version__,
        "botorch_source": botorch.__file__,
        "native_extension": _botorch_rs.__file__,
        "native_sha256": actual_hash,
        "cuda_device": torch.cuda.get_device_name(0) if "cuda" in devices else None,
        "dtype": args.dtype,
        "torch_threads": torch.get_num_threads(),
        "rayon_threads": os.environ["RAYON_NUM_THREADS"],
        "method": "interleaved shuffled warmed no_grad posterior mean+variance; CUDA synchronized before and after each timed call; excludes construction, fitting, imports, first training factor",
        "warmup": args.warmup,
        "repeats": args.repeats,
        "cases": [],
    }

    for size in args.sizes:
        generator = torch.Generator().manual_seed(20261020 + size)
        X = torch.rand(size, 4, dtype=dtype, generator=generator)
        Y = (X[:, :1] * 6).sin() + X[:, 1:2].square()
        query = torch.rand(16, 2, 4, dtype=dtype, generator=generator)
        entries = []
        for device in devices:
            x, y, q = X.to(device), Y.to(device), query.to(device)
            entries.append(
                (f"rust_torch_{device}", SingleTaskGP(x, y, backend="torch"), q)
            )
            entries.append((f"botorch_{device}", ReferenceGP(x, y), q))
        if dtype == torch.float64:
            entries.append(
                ("rust_native_cpu_float64", SingleTaskGP(X, Y, backend="native"), query)
            )

        def evaluate(entry):
            _, model, points = entry
            posterior = model.posterior(points)
            return posterior.mean, posterior.variance

        def synchronize(entry):
            if entry[2].device.type == "cuda":
                torch.cuda.synchronize(entry[2].device)

        values = {}
        durations = {name: [] for name, _, _ in entries}
        rng = random.Random(20261020)
        with torch.no_grad():
            for entry in entries:
                for _ in range(args.warmup):
                    values[entry[0]] = tuple(
                        tensor.detach() for tensor in evaluate(entry)
                    )
                synchronize(entry)
            for _ in range(args.repeats):
                order = list(entries)
                rng.shuffle(order)
                for entry in order:
                    synchronize(entry)
                    start = time.perf_counter_ns()
                    result = evaluate(entry)
                    synchronize(entry)
                    durations[entry[0]].append((time.perf_counter_ns() - start) / 1e6)
                    del result
        reference = values["botorch_cpu"]
        errors = {}
        for name, result in values.items():
            errors[name] = {
                label: (value.cpu() - expected).abs().max().item()
                for label, value, expected in zip(
                    ("mean", "variance"), result, reference, strict=True
                )
            }
            tolerance = 3e-5 if dtype == torch.float32 else 1e-7
            for value, expected in zip(result, reference, strict=True):
                torch.testing.assert_close(
                    value.cpu(), expected, atol=tolerance, rtol=tolerance
                )
        timings = {
            name: {
                "median_ms": statistics.median(samples),
                "min_ms": min(samples),
                "max_ms": max(samples),
                "samples_ms": samples,
            }
            for name, samples in durations.items()
        }
        case = {
            "train_n": size,
            "dim": 4,
            "query_batches": 16,
            "q": 2,
            "timings": timings,
            "max_abs_error": errors,
        }
        report["cases"].append(case)
        print(
            json.dumps(
                {
                    "train_n": size,
                    "median_ms": {
                        name: entry["median_ms"] for name, entry in timings.items()
                    },
                }
            ),
            flush=True,
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

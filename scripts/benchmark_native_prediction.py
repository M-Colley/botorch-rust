"""Compare native public snapshots and direct warm-cache marginal prediction."""

from __future__ import annotations

import argparse
import hashlib
import importlib.machinery
import importlib.util
import json
import os
import platform
import random
import statistics
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path


def timed(function, count):
    durations = []
    for _ in range(count):
        start = time.perf_counter_ns()
        function()
        durations.append((time.perf_counter_ns() - start) / 1e6)
    return {"median_ms": statistics.median(durations), "samples_ms": durations}


def revision(path):
    return subprocess.check_output(
        ["git", "-C", str(path), "rev-parse", "HEAD"], text=True
    ).strip()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-extension", type=Path, required=True)
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sizes", type=int, nargs="+", default=[32, 256, 1024])
    parser.add_argument("--repeats", type=int, default=21)
    parser.add_argument("--warmup", type=int, default=5)
    args = parser.parse_args()
    os.environ.update(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", RAYON_NUM_THREADS="1")
    root = Path(__file__).resolve().parent.parent
    sys.path[:0] = [str(root / "botorch_pyo3" / "python"), str(args.reference_root)]
    name = "botorch_rs._botorch_rs"
    spec = importlib.util.spec_from_file_location(
        name,
        args.native_extension,
        loader=importlib.machinery.ExtensionFileLoader(
            name, str(args.native_extension)
        ),
    )
    native = importlib.util.module_from_spec(spec)
    sys.modules[name] = native
    spec.loader.exec_module(native)
    import torch
    import botorch
    import botorch_rs
    from botorch.models import SingleTaskGP as Reference
    from botorch_rs.models import SingleTaskGP

    torch.set_num_threads(1)
    report = {
        "created_utc": datetime.now(UTC).isoformat(),
        "source_revision": revision(root),
        "reference_revision": revision(args.reference_root),
        "torch": torch.__version__,
        "platform": platform.platform(),
        "native_extension": str(args.native_extension.resolve()),
        "native_sha256": hashlib.sha256(args.native_extension.read_bytes()).hexdigest(),
        "python_package": botorch_rs.__file__,
        "reference_package": botorch.__file__,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "threads": {
            "torch": torch.get_num_threads(),
            "rayon": os.environ["RAYON_NUM_THREADS"],
        },
        "warmup": args.warmup,
        "repeats": args.repeats,
        "method": "CPU float64, shared exact hyperparameters, no_grad; shuffled interleaved public calls before native parent direct-cache warm, then component probes",
        "cases": [],
    }
    for n in args.sizes:
        generator = torch.Generator().manual_seed(20261020 + n)
        x = torch.rand(n, 4, dtype=torch.double, generator=generator)
        y = (x[:, :1] * 6).sin() + x[:, 1:2].square()
        query = torch.rand(16, 2, 4, dtype=torch.double, generator=generator)
        own = SingleTaskGP(x, y, outcome_transform=None, backend="native")
        reference = Reference(x, y, outcome_transform=None)
        for model in (own, reference):
            model.covar_module.lengthscale = torch.full(
                (1, 4), 0.31, dtype=torch.double
            )
            model.likelihood.noise = torch.tensor(0.02, dtype=torch.double)
            model.mean_module.constant = torch.tensor(0.1, dtype=torch.double)
            model.requires_grad_(False)
            model.eval()
        flat = (query.reshape(-1).tolist(), list(query.shape))

        def public(model):
            posterior = model.posterior(query)
            return posterior.mean, posterior.variance

        with torch.no_grad():
            observed, expected = public(own), public(reference)
            errors = {}
            for label, actual, target in zip(
                ("mean", "variance"), observed, expected, strict=True
            ):
                torch.testing.assert_close(actual, target, atol=1e-9, rtol=1e-9)
                errors[label] = float((actual - target).abs().max())
            for _ in range(args.warmup):
                public(own)
                public(reference)
            durations = {"native_public_unwarmed_parent": [], "botorch_public": []}
            entries = [
                ("native_public_unwarmed_parent", own),
                ("botorch_public", reference),
            ]
            rng = random.Random(731)
            for _ in range(args.repeats):
                rng.shuffle(entries)
                for label, model in entries:
                    start = time.perf_counter_ns()
                    public(model)
                    durations[label].append((time.perf_counter_ns() - start) / 1e6)
            results = {
                key: {"median_ms": statistics.median(values), "samples_ms": values}
                for key, values in durations.items()
            }
            results["native_copy_before_direct_parent_warm"] = timed(
                own._rs.copy, args.repeats
            )
            results["native_public_construction_before_parent_warm"] = timed(
                lambda: own.posterior(query), args.repeats
            )
            results["native_first_direct_parent_marginals"] = timed(
                lambda: own._rs.posterior_marginals(flat), 1
            )
            direct_mean, direct_variance = own._rs.posterior_marginals(flat)
            for actual, target in zip(
                (direct_mean, direct_variance), observed, strict=True
            ):
                torch.testing.assert_close(
                    torch.tensor(actual[0], dtype=torch.double).reshape(target.shape),
                    target,
                )
            results["native_direct_warm_marginals"] = timed(
                lambda: own._rs.posterior_marginals(flat), args.repeats
            )
            results["native_copy_after_direct_parent_warm"] = timed(
                own._rs.copy, args.repeats
            )
            results["native_public_after_direct_parent_warm"] = timed(
                lambda: public(own), args.repeats
            )
        case = {
            "n": n,
            "d": 4,
            "query_batches": 16,
            "q": 2,
            "raw_parameters": own._rs.raw_parameters(),
            "max_abs_error": errors,
            "timings": results,
        }
        report["cases"].append(case)
        print(
            json.dumps(
                {"n": n, **{key: value["median_ms"] for key, value in results.items()}}
            ),
            flush=True,
        )
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

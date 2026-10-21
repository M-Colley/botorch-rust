"""Reproduce end-to-end cached / joint qLogNEI forward timings.

Example (put the port and official checkout on PYTHONPATH first)::

    python scripts/benchmark_noisy_cache.py --rust-profile release --output cache.json

The default matches the investigation workload: float64 CPU, one worker,
32 training points, 180 baseline points, 8 candidate sets with q=2, and
128 MC samples. Model construction, baseline factorization, a shared-draw
value-parity check, and one additional warmup call are excluded. Seven
complete public-API forwards are timed under
torch.no_grad; Rust array conversion and reference Python overhead remain
included. This measures forwards, not fitting, gradients, or GPU work.
The release label requires the loaded extension's SHA256 to match the local
Cargo release artifact (or the explicit ``--rust-artifact`` path).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import random
import statistics
import subprocess
import sys
import time
import warnings
from pathlib import Path
from unittest.mock import patch

import botorch
import torch
from botorch.acquisition.logei import qLogNoisyExpectedImprovement as ReferenceNEI
from botorch.models import SingleTaskGP as ReferenceGP
from botorch.sampling.normal import SobolQMCNormalSampler as ReferenceSampler

import botorch_rs
from botorch_rs import _botorch_rs as extension
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
    parser.add_argument(
        "--rust-artifact",
        type=Path,
        help="Cargo release library to match when labeling a run --rust-profile release",
    )
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
        parser.error("all workload sizes, thread counts, warmups and repeats must be positive")
    source_root = Path(__file__).resolve().parents[1]
    extension_path = Path(extension.__file__).resolve()
    extension_sha256 = hashlib.sha256(extension_path.read_bytes()).hexdigest()
    artifact = args.rust_artifact
    if args.rust_profile == "release" and artifact is None:
        target = Path(os.environ.get("CARGO_TARGET_DIR", source_root / "target"))
        if not target.is_absolute():
            target = source_root / target
        filename = {
            "win32": "_botorch_rs.dll",
            "darwin": "lib_botorch_rs.dylib",
        }.get(sys.platform, "lib_botorch_rs.so")
        artifact = target / "release" / filename
    artifact_validation = None
    if artifact is not None:
        artifact = artifact.resolve()
        if not artifact.is_file():
            parser.error(f"Cargo artifact is missing: {artifact}")
        artifact_sha256 = hashlib.sha256(artifact.read_bytes()).hexdigest()
        if artifact_sha256 != extension_sha256:
            parser.error(
                "the loaded extension does not match the declared Cargo artifact; "
                "install the intended build before timing it"
            )
        artifact_validation = {"path": str(artifact), "sha256": artifact_sha256}
    torch.set_default_dtype(torch.float32)
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

    # Dimension-specific shared standard normals replace both native and
    # reference Sobol draws, including independently drawn cached baselines.
    # The reference's normal sampler is patched only inside this benchmark.
    def shared_draw(d, n, device=None, dtype=None, seed=None):
        return torch.randn(
            n, d, generator=torch.Generator().manual_seed(1000 + d), dtype=torch.float64
        ).to(device=device, dtype=dtype)

    acquisitions = {}
    with patch("botorch.sampling.normal.draw_sobol_normal_samples", shared_draw):
        for name, model_class, acq_class, sampler_class in (
            ("rust", RustGP, RustNEI, RustSampler),
            ("reference", ReferenceGP, ReferenceNEI, ReferenceSampler),
        ):
            model = model_class(
                train_x, train_y, torch.full_like(train_y, 0.02), outcome_transform=None
            ).eval()
            for cache_root in (False, True):
                sampler = sampler_class(torch.Size([args.samples]), seed=7)
                if name == "rust":
                    for dimension in (args.baseline, args.baseline + args.q):
                        sampler.base_samples = shared_draw(dimension, args.samples)
                acquisitions[name, cache_root] = acq_class(
                    model,
                    baseline,
                    sampler=sampler,
                    prune_baseline=False,
                    cache_root=cache_root,
                )
        checks = []
        with torch.no_grad():
            for cache_root in (False, True):
                actual = acquisitions["rust", cache_root](candidates)
                expected = acquisitions["reference", cache_root](candidates)
                torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)
                checks.append(
                    {
                        "cache_root": cache_root,
                        "max_abs_error": (actual - expected).abs().max().item(),
                        "atol": 1e-6,
                        "rtol": 1e-6,
                    }
                )
            timings = {key: [] for key in acquisitions}
            for acquisition in acquisitions.values():
                for _ in range(args.warmup):
                    acquisition(candidates)
            order_rng = random.Random(20261020)
            for _ in range(args.repeats):
                order = list(acquisitions)
                order_rng.shuffle(order)
                for key in order:
                    start = time.perf_counter()
                    value = acquisitions[key](candidates)
                    timings[key].append((time.perf_counter() - start) * 1000)
                    assert torch.isfinite(value).all()
            results = []
            for (name, cache_root), elapsed in timings.items():
                results.append(
                    {
                        "backend": name,
                        "cache_root": cache_root,
                        "median_ms": statistics.median(elapsed),
                        "times_ms": elapsed,
                    }
                )

    def git_info(path):
        result = {}
        for name, arguments in (
            ("revision", ["rev-parse", "HEAD"]),
            ("status", ["status", "--porcelain"]),
        ):
            command = subprocess.run(
                ["git", "-C", str(path), *arguments],
                capture_output=True,
                text=True,
                check=False,
            )
            result[name] = command.stdout.strip() if command.returncode == 0 else None
        return result

    report = {
        "parameters": {
            key: value
            for key, value in vars(args).items()
            if key not in ("output", "rust_artifact")
        },
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "botorch": botorch.__version__,
        "botorch_path": botorch.__file__,
        "reference_source": git_info(Path(botorch.__file__).resolve().parents[1]),
        "rust_source": git_info(source_root),
        "extension_path": str(extension_path),
        "extension_sha256": extension_sha256,
        "matching_cargo_artifact": artifact_validation,
        "draws": "torch.randn float64 with per-dimension seed=1000+dimension, shared across implementations",
        "timing_protocol": "warm all variants, then shuffle all four variants in each measured round (seed=20261020)",
        "value_checks": checks,
        "botorch_rs": botorch_rs.__version__,
        "results": results,
    }
    output = json.dumps(report, indent=2)
    print(output)
    if args.output is not None:
        args.output.write_text(output + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

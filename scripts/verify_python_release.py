"""Run a canonical Python suite against an installed wheel and record provenance."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def revision(path: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(path), "rev-parse", "HEAD"], text=True
    ).strip()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-root", type=Path, required=True)
    parser.add_argument("--native-sha256", required=True)
    parser.add_argument("--reference", type=Path, default=Path(".reference/botorch"))
    parser.add_argument("--phase", choices=["broad", "equivalence"], required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent.parent
    package_root = args.package_root.resolve()
    reference = args.reference.resolve()
    package = package_root / "botorch_rs"
    if not package.is_dir():
        raise ValueError(
            "--package-root must contain the installed botorch_rs package."
        )
    os.environ["PYTHONPATH"] = os.pathsep.join((str(package_root), str(reference)))
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "RAYON_NUM_THREADS"):
        os.environ[name] = "1"
    os.environ["BOTORCH_RS_REQUIRE_BOTORCH"] = "1"
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    sys.dont_write_bytecode = True
    sys.path[:0] = [str(package_root), str(reference)]
    import botorch
    import botorch_rs
    import pytest
    import torch

    from botorch_rs import _botorch_rs

    torch.set_num_threads(1)
    native = Path(_botorch_rs.__file__).resolve()
    if not native.is_relative_to(package):
        raise RuntimeError(
            "Tests would load an extension outside the installed package."
        )
    native_hash = hashlib.sha256(native.read_bytes()).hexdigest()
    if native_hash.lower() != args.native_sha256.lower():
        raise RuntimeError("Installed extension SHA256 differs from --native-sha256.")
    if not Path(botorch.__file__).resolve().is_relative_to(reference):
        raise RuntimeError("The independent reference did not load from --reference.")
    source_hash = hashlib.sha256()
    files = sorted(
        p
        for p in package.rglob("*")
        if p.is_file() and (p.suffix in {".py", ".pyi"} or p.name == "py.typed")
    )
    for path in files:
        source_hash.update(path.relative_to(package).as_posix().encode() + b"\0")
        source_hash.update(path.read_bytes() + b"\0")

    class Ledger:
        selected: list[str] = []
        deselected: list[str] = []
        collection_skips: list[str] = []
        results: dict[str, dict[str, Any]] = {}

        def pytest_deselected(self, items: Any) -> None:
            self.deselected.extend(item.nodeid for item in items)

        def pytest_collection_finish(self, session: Any) -> None:
            self.selected = [item.nodeid for item in session.items]
            self.results = {
                nodeid: {"outcome": "not_run", "duration_s": 0.0}
                for nodeid in self.selected
            }

        def pytest_collectreport(self, report: Any) -> None:
            if report.skipped:
                self.collection_skips.append(report.nodeid)

        def pytest_runtest_logreport(self, report: Any) -> None:
            result = self.results[report.nodeid]
            result["duration_s"] += report.duration
            if report.failed:
                result["outcome"] = "failed" if report.when == "call" else "error"
                result["failure_phase"] = report.when
            elif report.skipped and result["outcome"] not in {"failed", "error"}:
                result["outcome"] = "skipped"
            elif (
                report.passed
                and report.when == "call"
                and result["outcome"] == "not_run"
            ):
                result["outcome"] = "passed"

    ledger = Ledger()
    args.report.parent.mkdir(parents=True, exist_ok=True)
    marker = "not equivalence" if args.phase == "broad" else "equivalence"
    command = [
        str(root / "botorch_pyo3/tests"),
        "-q",
        "-rs",
        "-m",
        marker,
        f"--junitxml={args.report.with_suffix('.xml')}",
    ]
    started = datetime.now(UTC)
    result = pytest.main(command, plugins=[ledger])
    duplicates = [
        name for name, count in Counter(ledger.selected).items() if count != 1
    ]
    report = {
        "started_utc": started.isoformat(),
        "finished_utc": datetime.now(UTC).isoformat(),
        "exit_code": int(result),
        "phase": args.phase,
        "source_revision": revision(root),
        "reference_revision": revision(reference),
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_device": torch.cuda.get_device_name(0)
        if torch.cuda.is_available()
        else None,
        "package_path": botorch_rs.__file__,
        "native_path": str(native),
        "native_sha256": native_hash,
        "python_sources_sha256": source_hash.hexdigest(),
        "python_source_files": len(files),
        "helper_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "pytest_arguments": command,
        "environment": {
            name: os.environ[name]
            for name in (
                "PYTHONPATH",
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
                "RAYON_NUM_THREADS",
                "BOTORCH_RS_REQUIRE_BOTORCH",
                "PYTHONDONTWRITEBYTECODE",
            )
        },
        "selected": ledger.selected,
        "deselected": ledger.deselected,
        "collection_skips": ledger.collection_skips,
        "duplicate_selected_ids": duplicates,
        "totals": dict(Counter(value["outcome"] for value in ledger.results.values())),
        "cases": ledger.results,
    }
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "report": str(args.report),
                "totals": report["totals"],
                "exit_code": int(result),
            }
        )
    )
    return int(result) or int(bool(duplicates))


if __name__ == "__main__":
    raise SystemExit(main())

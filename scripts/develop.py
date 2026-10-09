#!/usr/bin/env python3
"""Build the ``botorch_rs`` extension with cargo and place it in the source tree.

This is a lightweight alternative to ``maturin develop`` for local development:
it runs ``cargo build --release -p botorch_pyo3`` (with ``PYO3_PYTHON`` set to
the interpreter running this script) and copies the resulting library to
``botorch_pyo3/python/botorch_rs/_botorch_rs.pyd`` (Windows) or
``_botorch_rs.so`` (Linux, macOS). The package can then be used from the
source tree::

    python scripts/develop.py
    PYTHONPATH=botorch_pyo3/python python -m pytest botorch_pyo3/tests -q

Options: ``--debug`` builds without optimizations, ``--python`` selects the
interpreter that configures the build (any CPython >= 3.11 can import the
abi3 library).
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PACKAGE = ROOT / "botorch_pyo3" / "python" / "botorch_rs"


def library_paths(target: Path) -> tuple[Path, Path]:
    """The cargo artifact and its destination in the package."""
    if sys.platform == "win32":
        return target / "_botorch_rs.dll", PACKAGE / "_botorch_rs.pyd"
    if sys.platform == "darwin":
        return target / "lib_botorch_rs.dylib", PACKAGE / "_botorch_rs.so"
    return target / "lib_botorch_rs.so", PACKAGE / "_botorch_rs.so"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--debug", action="store_true", help="build without --release")
    parser.add_argument("--python", default=sys.executable, help="interpreter for PyO3")
    parser.add_argument("--cargo", default=os.environ.get("CARGO", "cargo"))
    args = parser.parse_args()

    profile = "debug" if args.debug else "release"
    cmd = [args.cargo, "build", "-p", "botorch_pyo3"]
    if not args.debug:
        cmd.append("--release")
    env = dict(os.environ, PYO3_PYTHON=args.python)
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=ROOT, env=env, check=True)

    target = Path(os.environ.get("CARGO_TARGET_DIR", ROOT / "target")) / profile
    source, destination = library_paths(target)
    if not source.exists():
        print(f"error: {source} was not produced by cargo", file=sys.stderr)
        return 1
    for stale in [*PACKAGE.glob("_botorch_rs*.pyd"), *PACKAGE.glob("_botorch_rs*.so")]:
        if stale != destination:
            stale.unlink()
    try:
        shutil.copy2(source, destination)
    except PermissionError:
        print(
            f"error: cannot overwrite {destination}; close Python processes that use it",
            file=sys.stderr,
        )
        return 1
    print(f"installed {destination.relative_to(ROOT)} ({profile})")
    return 0


if __name__ == "__main__":
    sys.exit(main())

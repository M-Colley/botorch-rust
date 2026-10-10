"""pytest configuration of the botorch_rs tests.

The API tests run for every array type that is available: nested lists
always, NumPy arrays and torch tensors when those libraries are installed.
The parity tests record the worst error of every compared quantity, which is
printed at the end of the session (per section, with the tolerances).
"""

from __future__ import annotations

from typing import Any

import pytest

from helpers import KINDS, ParityLog


@pytest.fixture(params=KINDS)
def kind(request: pytest.FixtureRequest) -> str:
    """The array type under test: "list", "numpy" or "torch"."""
    return request.param


def pytest_terminal_summary(terminalreporter: Any) -> None:
    if ParityLog.worst:
        terminalreporter.section("worst errors vs BoTorch (multi-output parity)")
        for quantity, err in sorted(ParityLog.worst.items()):
            terminalreporter.write_line(f"{quantity:60s} {err:9.2e}")
    for section, entries in sorted(ParityLog.sections.items()):
        terminalreporter.section(f"worst errors vs BoTorch ({section})")
        terminalreporter.write_line(f"{'quantity':60s} {'worst':>9s} {'tolerance':>9s}")
        for quantity, (err, tol) in sorted(entries.items()):
            tol_s = "" if tol is None else f"{tol:9.1e}"
            terminalreporter.write_line(f"{quantity:60s} {err:9.2e} {tol_s:>9s}")

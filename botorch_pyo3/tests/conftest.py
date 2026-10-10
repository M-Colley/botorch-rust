"""pytest configuration of the botorch_rs tests.

The API tests run for every array type that is available: nested lists
always, NumPy arrays and torch tensors when those libraries are installed.
The parity tests record the worst error of every compared quantity, which is
printed at the end of the session.
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
    if not ParityLog.worst:
        return
    terminalreporter.section("worst errors vs BoTorch (multi-output parity)")
    for quantity, err in sorted(ParityLog.worst.items()):
        terminalreporter.write_line(f"{quantity:60s} {err:9.2e}")

"""pytest configuration of the botorch_rs tests.

The API tests run for every array type that is available: nested lists
always, NumPy arrays and torch tensors when those libraries are installed.
"""

from __future__ import annotations

import pytest

from helpers import KINDS


@pytest.fixture(params=KINDS)
def kind(request: pytest.FixtureRequest) -> str:
    """The array type under test: "list", "numpy" or "torch"."""
    return request.param

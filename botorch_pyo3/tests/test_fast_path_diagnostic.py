"""Compatibility diagnostics of the pinned official SciPy candidate driver."""

from __future__ import annotations

import pytest

from helpers import importorskip_botorch

importorskip_botorch()

from botorch.generation import gen as bt_gen  # noqa: E402

from botorch_rs.generation import gen as rs_gen  # noqa: E402


@pytest.mark.parametrize("method", ["L-BFGS-B", "SLSQP"])
@pytest.mark.parametrize("with_grad", [False, True])
@pytest.mark.parametrize("options", [{}, {"maxiter": 4, "ftol": 1e-6}, {"maxfun": 5}])
@pytest.mark.parametrize("timeout", [None, 0.0])
def test_fast_path_reasons(method, with_grad, options, timeout, monkeypatch) -> None:
    if not hasattr(bt_gen, "get_reasons_against_fast_path"):
        pytest.skip("requires BoTorch's SciPy fast-path diagnostic")
    # Releases use a different tested SciPy version range. Check the pinned
    # official API's range while preserving each implementation's diagnostic.
    monkeypatch.setattr(bt_gen, "SCIPY_MIN_VERSION", rs_gen.SCIPY_MIN_VERSION)
    monkeypatch.setattr(bt_gen, "SCIPY_UNTESTED_VERSION", rs_gen.SCIPY_UNTESTED_VERSION)
    assert rs_gen.get_reasons_against_fast_path(
        method, with_grad, options, timeout
    ) == bt_gen.get_reasons_against_fast_path(method, with_grad, options, timeout)

"""Public scalar-temperature and tensor-metadata helper parity."""

from __future__ import annotations

import math

import pytest

from botorch_rs.acquisition.logei import check_tau


@pytest.mark.parametrize("tau", [0.0, -1.0, float("nan"), [], [1.0, 2.0]])
def test_check_tau_rejects_invalid(tau) -> None:
    with pytest.raises(ValueError, match="temperature"):
        check_tau(tau, "temperature")


@pytest.mark.parametrize("tau", [0.1, float("inf"), [1.0]])
def test_check_tau_preserves_value(tau) -> None:
    assert check_tau(tau, "tau") is tau


def test_helper_parity_with_botorch() -> None:
    from helpers import importorskip_botorch

    importorskip_botorch()
    import torch
    from botorch.acquisition.fixed_feature import get_device_of_sequence as bt_device
    from botorch.acquisition.fixed_feature import get_dtype_of_sequence as bt_dtype
    from botorch.acquisition.logei import check_tau as bt_check

    from botorch_rs.acquisition.fixed_feature import get_device_of_sequence, get_dtype_of_sequence

    for dtype in (torch.float16, torch.float32, torch.float64):
        tensor = torch.tensor([0.5], dtype=dtype)
        for values in ([], [tensor], [tensor, tensor], [tensor, 1.0], [1.0]):
            assert get_dtype_of_sequence(values) == bt_dtype(values)
            assert get_device_of_sequence(values) == bt_device(values)
    for tensor in (torch.tensor(0.5), torch.tensor([0.5]), torch.tensor([[0.5]])):
        assert bt_check(tensor, "tau") is check_tau(tensor, "tau")
    for value in (0.0, -1.0, math.nan, torch.empty(0), torch.ones(2)):
        for check in (bt_check, check_tau):
            with pytest.raises(ValueError):
                check(value, "tau")

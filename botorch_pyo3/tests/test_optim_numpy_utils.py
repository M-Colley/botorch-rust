"""Storage semantics and optimizer bound conversions against live BoTorch."""

from __future__ import annotations

import pytest

from botorch_rs.exceptions.errors import UnsupportedError
from botorch_rs.optim.utils import numpy_utils as rs
from helpers import KINDS, as_kind

np = pytest.importorskip("numpy")


@pytest.mark.parametrize("kind", KINDS)
def test_as_ndarray_values_and_copy_semantics(kind: str) -> None:
    values = as_kind([[1.0, 2.0], [3.0, 4.0]], kind)
    actual = rs.as_ndarray(values)
    assert actual.dtype == np.float64
    np.testing.assert_array_equal(actual, [[1, 2], [3, 4]])
    same_dtype = rs.as_ndarray(values, dtype=np.float64, inplace=False)
    changed_dtype = rs.as_ndarray(values, dtype=np.float32, inplace=False)
    assert same_dtype.dtype == np.float64 and changed_dtype.dtype == np.float32
    if kind != "list":
        original = values.detach().numpy() if kind == "torch" else values
        assert np.shares_memory(actual, original)
        assert not np.shares_memory(same_dtype, original)
        assert not np.shares_memory(changed_dtype, original)
    same_dtype[0, 0] = -1
    np.testing.assert_array_equal(rs.as_ndarray(values), [[1, 2], [3, 4]])


def test_as_ndarray_torch_dtypes_and_gradients() -> None:
    torch = pytest.importorskip("torch")
    from botorch_rs.optim.utils.numpy_utils import torch_to_numpy_dtype_dict

    for torch_dtype, numpy_dtype in torch_to_numpy_dtype_dict.items():
        values = torch.zeros(2, dtype=torch_dtype)
        converted = rs.as_ndarray(values, dtype=numpy_dtype)
        assert converted.dtype == numpy_dtype
        assert np.shares_memory(converted, values.numpy())
    values = torch.ones(2, dtype=torch.float32, requires_grad=True)
    assert rs.as_ndarray(values).dtype == np.float64
    result = rs.as_ndarray(values, dtype=np.float32)
    assert np.shares_memory(result, values.detach().numpy())
    # This combination raises KeyError in BoTorch's reversed dtype lookup.
    result = rs.as_ndarray(values, dtype=np.float32, inplace=False)
    assert not np.shares_memory(result, values.detach().numpy())


@pytest.mark.parametrize("kind", KINDS)
def test_optimizer_bounds(kind: str) -> None:
    parameters = {"a": as_kind([1], kind), "b": as_kind([2, 3], kind)}
    bounds = {"a": (None, 1), "b": (as_kind([0, -1], kind), 2)}
    actual = rs.get_bounds_as_ndarray(parameters, bounds)
    np.testing.assert_array_equal(actual, [[-np.inf, 1], [0, 2], [-1, 2]])
    assert actual.dtype == np.float64
    assert rs.get_bounds_as_ndarray(parameters, {}) is None
    assert rs.get_bounds_as_ndarray({}, {}) is None


@pytest.mark.parametrize("kind", KINDS)
def test_per_element_bounds(kind: str) -> None:
    parameters = {"a": as_kind([[1, 2, 3]] * 2, kind), "b": as_kind([[0]] * 2, kind)}
    bounds = {"a": (0, as_kind([1, 2, 3], kind)), "b": (-1, 2)}
    actual = rs.get_per_element_bounds(parameters, bounds, batch_shape=(2,))
    np.testing.assert_array_equal(actual, [[0, 1], [0, 2], [0, 3], [-1, 2]])
    assert rs.get_per_element_bounds(parameters, {}, batch_shape=(2,)) is None
    bounds["a"] = (as_kind([[0, 0, 0]] * 2, kind), as_kind([[1, 2, 3]] * 2, kind))
    np.testing.assert_array_equal(
        rs.get_per_element_bounds(parameters, bounds, batch_shape=(2,)), actual
    )
    bounds["a"] = (as_kind([[0, 0, 0], [1, 0, 0]], kind), 2)
    with pytest.raises(UnsupportedError, match="different bounds"):
        rs.get_per_element_bounds(parameters, bounds, batch_shape=(2,))
    with pytest.raises(ValueError, match="batch shape"):
        rs.get_per_element_bounds(parameters, {}, batch_shape=(3,))


def test_live_botorch_numpy_helpers() -> None:
    torch = pytest.importorskip("torch")
    bt = pytest.importorskip("botorch.optim.utils.numpy_utils")
    for dtype in (torch.float32, torch.float64):
        parameters = {"a": torch.zeros(2, 3, dtype=dtype), "b": torch.ones(2, 1, dtype=dtype)}
        bounds = {"a": (-1, 2), "b": (0, None)}
        np.testing.assert_array_equal(
            rs.get_bounds_as_ndarray(parameters, bounds),
            bt.get_bounds_as_ndarray(parameters, bounds),
        )
        np.testing.assert_array_equal(
            rs.get_per_element_bounds(parameters, bounds, (2,)),
            bt.get_per_element_bounds(parameters, bounds, torch.Size([2])),
        )
        values = torch.arange(4, dtype=dtype).requires_grad_()
        for target_dtype in (None, np.float32, np.float64):
            np.testing.assert_array_equal(
                rs.as_ndarray(values, dtype=target_dtype),
                bt.as_ndarray(values, dtype=target_dtype),
            )

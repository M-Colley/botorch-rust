"""NumPy conversion and flattened optimizer bounds.

NumPy is loaded when these explicitly NumPy-facing functions are called.
Parameters and bounds may be torch tensors, NumPy arrays or nested lists.
"""

from __future__ import annotations

import importlib
import math
from typing import Any

from botorch_rs._arrays import is_torch_tensor
from botorch_rs._ops import shape_of
from botorch_rs.exceptions.errors import UnsupportedError

torch_to_numpy_dtype_dict: dict[Any, Any]


def as_ndarray(values: Any, dtype: Any = None, inplace: bool = True) -> Any:
    """Convert to NumPy, sharing storage when possible only if ``inplace``.

    Tensor inputs are detached and transferred to CPU. The default dtype is
    float64, matching BoTorch's ``ndarray.astype(None)``. A requested dtype
    conversion allocates only when necessary; ``inplace=False`` always
    protects the original storage, including when the dtype stays the same.
    """
    np = importlib.import_module("numpy")
    target_dtype = np.dtype(dtype)
    if is_torch_tensor(values):
        tensor = values.detach().cpu().resolve_conj().resolve_neg()
        out = tensor.numpy()
        return out.astype(
            target_dtype,
            copy=not inplace and tensor.device == values.device,
        )
    out = np.asarray(values, dtype=target_dtype)
    if not inplace and isinstance(values, np.ndarray) and np.shares_memory(out, values):
        return out.copy()
    return out


def _bound_values(bound: Any, shape: tuple[int, ...], size: int) -> Any:
    np = importlib.import_module("numpy")
    value = as_ndarray(bound)
    if value.size == size:
        return value.reshape(-1)
    return np.broadcast_to(value, shape).reshape(-1)


def get_bounds_as_ndarray(parameters: dict[str, Any], bounds: dict[str, tuple[Any, Any]]) -> Any:
    """Flatten parameter bounds in dictionary order, returning ``None`` if unbounded."""
    np = importlib.import_module("numpy")
    shapes = {name: shape_of(parameter) for name, parameter in parameters.items()}
    out = np.full((sum(math.prod(shape) for shape in shapes.values()), 2), (-np.inf, np.inf))
    start = 0
    for name, shape in shapes.items():
        size = math.prod(shape)
        if name in bounds:
            lower, upper = bounds[name]
            if lower is not None:
                out[start : start + size, 0] = _bound_values(lower, shape, size)
            if upper is not None:
                out[start : start + size, 1] = _bound_values(upper, shape, size)
        start += size
    return None if np.isinf(out).all() else out


def get_per_element_bounds(
    parameters: dict[str, Any],
    bounds: dict[str, tuple[Any, Any]],
    batch_shape: Any,
) -> Any:
    """Extract the shared bounds of one element of batched model parameters.

    Full batched bounds are accepted when every batch element has identical
    bounds. Different bounds per batch cannot be represented by this return
    value and raise ``UnsupportedError``.
    """
    np = importlib.import_module("numpy")
    batch_shape = tuple(batch_shape)
    if any(size <= 0 for size in batch_shape):
        raise ValueError("batch_shape dimensions must be positive.")
    batch_size = math.prod(batch_shape)
    trailing: dict[str, tuple[int, ...]] = {}
    for name, parameter in parameters.items():
        shape = shape_of(parameter)
        if shape[: len(batch_shape)] != batch_shape:
            raise ValueError(f"Parameter {name!r} does not have batch shape {batch_shape}.")
        trailing[name] = shape[len(batch_shape) :]
    out = np.full((sum(math.prod(shape) for shape in trailing.values()), 2), (-np.inf, np.inf))
    start = 0
    for name, shape in trailing.items():
        size = math.prod(shape)
        if name in bounds:
            for column, bound in enumerate(bounds[name]):
                if bound is None:
                    continue
                value = as_ndarray(bound)
                full_shape = shape_of(parameters[name])
                if batch_shape and value.shape == full_shape:
                    rows = value.reshape(batch_size, size)
                    if not np.array_equal(
                        rows, np.broadcast_to(rows[0], rows.shape), equal_nan=True
                    ):
                        raise UnsupportedError(
                            f"Parameter {name!r} has different bounds across batch elements."
                        )
                    value = rows[0]
                out[start : start + size, column] = _bound_values(value, shape, size)
        start += size
    return None if np.isinf(out).all() else out


def __getattr__(name: str) -> Any:
    if name != "torch_to_numpy_dtype_dict":
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    torch = importlib.import_module("torch")
    np = importlib.import_module("numpy")
    mapping = {
        torch.bool: bool,
        **{
            getattr(torch, dtype): getattr(np, dtype)
            for dtype in (
                "uint8",
                "int8",
                "int16",
                "int32",
                "int64",
                "float16",
                "float32",
                "float64",
                "complex64",
                "complex128",
            )
        },
    }
    globals()[name] = mapping
    return mapping


__all__ = [
    "as_ndarray",
    "get_bounds_as_ndarray",
    "get_per_element_bounds",
    "torch_to_numpy_dtype_dict",
]

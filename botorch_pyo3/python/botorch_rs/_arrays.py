"""Array interoperability without runtime dependencies.

Every array argument may be a torch tensor, a NumPy array or a (nested) list of
numbers. Arrays are passed to the extension as a flat row-major list of floats
plus a shape, and results are returned with the type of the array they were
computed from: torch tensors (same floating dtype -- float64 unless a different
floating dtype was passed -- and device), NumPy arrays, or nested lists of
floats. torch and NumPy are never imported by this module; they are only used
when the caller's arrays already come from them.
"""

from __future__ import annotations

import math
import sys
from collections.abc import Sequence
from typing import Any, NamedTuple

#: ``(data, shape)`` as produced by :func:`to_flat`.
Flat = tuple[list[float], tuple[int, ...]]
#: ``(data, shape)`` as returned by the extension module.
Pair = tuple[Sequence[float], Sequence[int]]


class ArrayKind(NamedTuple):
    """The array type that results are returned as."""

    lib: str  # "list", "numpy" or "torch"
    dtype: Any = None  # floating dtype to restore (None: float64)
    device: Any = None  # torch device (None: CPU)


LIST = ArrayKind("list")


def _torch() -> Any:
    return sys.modules.get("torch")


def _numpy() -> Any:
    return sys.modules.get("numpy")


def is_torch_tensor(x: Any) -> bool:
    torch = _torch()
    return torch is not None and isinstance(x, torch.Tensor)


def _is_numpy(x: Any) -> bool:
    np = _numpy()
    return np is not None and isinstance(x, (np.ndarray, np.generic))


def kind_of(x: Any) -> ArrayKind:
    """The array kind of ``x`` (lists for anything that is not an array)."""
    if is_torch_tensor(x):
        return ArrayKind("torch", x.dtype if x.is_floating_point() else None, x.device)
    if _is_numpy(x):
        np = _numpy()
        dtype = x.dtype if np.issubdtype(x.dtype, np.floating) else None
        return ArrayKind("numpy", dtype)
    return LIST


def first_kind(*xs: Any) -> ArrayKind:
    """The kind of the first argument that is a torch tensor or NumPy array."""
    for x in xs:
        kind = kind_of(x)
        if kind.lib != "list":
            return kind
    return LIST


def kind_from_dtype(dtype: Any = None, device: Any = None) -> ArrayKind:
    """The kind of results that are not computed from an array argument (for
    example the bounds of a test function).

    An explicit torch dtype selects torch tensors and a NumPy dtype NumPy
    arrays. Without ``dtype``, results are float64 torch tensors if torch has
    been imported (BoTorch's default ``dtype=torch.double``), else lists.
    """
    torch = _torch()
    if dtype is None:
        if torch is not None:
            return ArrayKind("torch", torch.float64, device)
        return LIST
    if torch is not None and isinstance(dtype, torch.dtype):
        return ArrayKind("torch", dtype if dtype.is_floating_point else None, device)
    np = _numpy()
    if np is not None:
        try:
            npdtype = np.dtype(dtype)
        except TypeError:
            npdtype = None
        if npdtype is not None:
            return ArrayKind("numpy", npdtype if npdtype.kind == "f" else None)
    raise TypeError(f"Unsupported dtype {dtype!r}: pass a torch dtype, a NumPy dtype, or None.")


def _seq(v: Any) -> Sequence[Any] | None:
    """``v`` as a sequence if it is a list, tuple or a non-scalar array."""
    if isinstance(v, (list, tuple)):
        return v
    if is_torch_tensor(v) or _is_numpy(v):
        if v.ndim == 0:
            return None
        return v.tolist()
    return None


def _flatten_nested(x: Sequence[Any], name: str) -> Flat:
    shape: list[int] = []
    level: list[Any] = [x]
    while level:
        seqs = [_seq(v) for v in level]
        if all(s is None for s in seqs):
            break
        if any(s is None for s in seqs):
            raise ValueError(f"`{name}` is a ragged nested sequence.")
        n = len(seqs[0])  # type: ignore[arg-type]
        if any(len(s) != n for s in seqs):  # type: ignore[arg-type]
            raise ValueError(f"`{name}` is a ragged nested sequence.")
        shape.append(n)
        level = [v for s in seqs for v in s]  # type: ignore[union-attr]
    for v in level:
        if isinstance(v, (str, bytes, complex)) or isinstance(v, (list, tuple)):
            raise TypeError(f"`{name}` must contain real numbers, got {type(v).__name__}.")
    return level, tuple(shape)


def to_flat(x: Any, name: str = "X") -> Flat:
    """``(data, shape)``: the values of ``x`` in row-major order and its shape."""
    if is_torch_tensor(x):
        if x.is_complex():
            raise TypeError(f"`{name}` must be real-valued.")
        t = x.detach()
        return t.reshape(-1).cpu().tolist(), tuple(t.shape)
    if _is_numpy(x):
        np = _numpy()
        a = np.asarray(x)
        if a.dtype.kind not in "biuf":
            raise TypeError(f"`{name}` must be a real-valued array, got dtype {a.dtype}.")
        return a.reshape(-1).tolist(), tuple(a.shape)
    if isinstance(x, (bool, int, float)):
        return [float(x)], ()
    if isinstance(x, (list, tuple)):
        return _flatten_nested(x, name)
    if hasattr(x, "tolist") and hasattr(x, "shape"):  # other array libraries
        return to_flat(x.tolist(), name)
    raise TypeError(
        f"`{name}` must be a torch tensor, a NumPy array or a nested list of numbers, "
        f"got {type(x).__name__}."
    )


def _unflatten(data: Sequence[float], shape: tuple[int, ...]) -> Any:
    if not shape:
        return float(data[0])

    def build(offset: int, dims: tuple[int, ...]) -> list[Any]:
        if len(dims) == 1:
            return [float(v) for v in data[offset : offset + dims[0]]]
        step = math.prod(dims[1:])
        return [build(offset + i * step, dims[1:]) for i in range(dims[0])]

    return build(0, shape)


def from_flat(data: Sequence[float], shape: Sequence[int], kind: ArrayKind) -> Any:
    """An array of the given kind and shape holding ``data`` (row-major).

    A shape of ``()`` gives a 0-dim tensor, a NumPy scalar or a float.
    """
    shape = tuple(shape)
    if kind.lib == "torch":
        torch = _torch()
        if torch is None:  # pragma: no cover - torch was imported for the input
            import torch
        values = data if isinstance(data, list) else list(data)
        t = torch.tensor(values, dtype=kind.dtype or torch.float64).reshape(shape)
        if kind.device is not None and kind.device.type != "cpu":
            t = t.to(kind.device)
        return t
    if kind.lib == "numpy":
        np = _numpy()
        if np is None:  # pragma: no cover - numpy was imported for the input
            import numpy as np
        dtype = kind.dtype or np.float64
        if not shape:
            return np.dtype(dtype).type(data[0])
        return np.asarray(data, dtype=dtype).reshape(shape)
    return _unflatten(data, shape)


def bool_array(data: Sequence[bool], shape: Sequence[int], kind: ArrayKind) -> Any:
    """A boolean array of the given kind and shape (a bool tensor, a NumPy
    bool array, or nested lists of bools)."""
    shape = tuple(shape)
    values = [bool(v) for v in data]
    if kind.lib == "torch":
        import torch

        t = torch.tensor(values, dtype=torch.bool).reshape(shape)
        if kind.device is not None and kind.device.type != "cpu":
            t = t.to(kind.device)
        return t
    if kind.lib == "numpy":
        import numpy as np

        return np.asarray(values, dtype=bool).reshape(shape)
    if not shape:
        return values[0]

    def build(offset: int, dims: tuple[int, ...]) -> list[Any]:
        if len(dims) == 1:
            return values[offset : offset + dims[0]]
        step = math.prod(dims[1:])
        return [build(offset + i * step, dims[1:]) for i in range(dims[0])]

    return build(0, shape)


def from_pair(pair: Pair, kind: ArrayKind) -> Any:
    """Converts a ``(data, shape)`` result of the extension."""
    return from_flat(pair[0], pair[1], kind)


def as_float(x: Any, name: str) -> float:
    """A Python float from a number or a one-element array."""
    data, _ = to_flat(x, name)
    if len(data) != 1:
        raise ValueError(f"`{name}` must be a scalar, got {len(data)} values.")
    return float(data[0])


def shape_of(x: Any) -> tuple[int, ...]:
    """The shape of an array or nested list."""
    return to_flat(x)[1]


def broadcast_shapes(*shapes: Sequence[int]) -> tuple[int, ...] | None:
    """The numpy / torch broadcast of the shapes, or ``None`` if they are
    incompatible."""
    n = max((len(s) for s in shapes), default=0)
    out = [1] * n
    for s in shapes:
        s = tuple(s)
        for i, v in enumerate(s):
            j = n - len(s) + i
            if out[j] == 1:
                out[j] = v
            elif v not in (1, out[j]):
                return None
    return tuple(out)


def broadcast_to(data: Sequence[float], shape: Sequence[int], target: Sequence[int]) -> list[float]:
    """``data`` (row-major, of shape ``shape``) broadcast to ``target``."""
    shape, target = tuple(shape), tuple(target)
    if broadcast_shapes(shape, target) != target:
        raise ValueError(f"A value of shape {shape} cannot be broadcast to the shape {target}.")
    offset = len(target) - len(shape)
    strides = [0] * len(shape)
    s = 1
    for i in range(len(shape) - 1, -1, -1):
        strides[i] = 0 if shape[i] == 1 else s
        s *= shape[i]
    out = []
    total = math.prod(target)
    idx = [0] * len(target)
    for _ in range(total):
        flat = sum(idx[offset + i] * st for i, st in enumerate(strides))
        out.append(float(data[flat]))
        for k in range(len(target) - 1, -1, -1):
            idx[k] += 1
            if idx[k] < target[k]:
                break
            idx[k] = 0
    return out

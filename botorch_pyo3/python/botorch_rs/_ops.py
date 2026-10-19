"""Element-wise functions and reductions of the extension on arrays of every
kind (phase 11a).

:func:`apply` evaluates an element-wise function of the extension
(``_botorch_rs.elementwise``: the special functions of
``botorch.utils.safe_math``, ``botorch.utils.probability``, ...) on
broadcast arrays, and :func:`reduce` a reduction over one or more
dimensions (``_botorch_rs.reduce``). Inputs may be torch tensors, NumPy
arrays, nested lists or numbers; results have the type of the first array
argument. For torch tensors that require gradients (with gradients enabled)
the values and their exact first derivatives are computed together in Rust
and connected to the autograd graph by a ``torch.autograd.Function``
(first-order: the derivatives themselves are not differentiable).
"""

from __future__ import annotations

import importlib
import math
import sys
from collections.abc import Sequence
from typing import Any

from botorch_rs import _botorch_rs as _rs
from botorch_rs._arrays import (
    LIST,
    ArrayKind,
    broadcast_shapes,
    broadcast_to,
    first_kind,
    from_flat,
    is_torch_tensor,
    kind_of,
    to_flat,
)


def _torch() -> Any:
    return sys.modules.get("torch")


def is_single_precision(kind: ArrayKind) -> bool:
    """Whether arrays of ``kind`` are float32 (BoTorch then uses float32
    constants in its dtype-dependent functions)."""
    if kind.dtype is None:
        return False
    if kind.lib == "torch":
        return bool(kind.dtype == _torch().float32)
    if kind.lib == "numpy":
        return str(kind.dtype) == "float32"
    return False


def dtype_name(x: Any) -> str | None:
    """The floating dtype of a tensor or NumPy array as ``"float32"`` /
    ``"float64"``, another name for other dtypes, or ``None`` for lists and
    numbers (which are float64)."""
    if is_torch_tensor(x):
        return str(x.dtype).replace("torch.", "")
    np = sys.modules.get("numpy")
    if np is not None and isinstance(x, (np.ndarray, np.generic)):
        return str(np.asarray(x).dtype)
    return None


def _requires_grad(xs: Sequence[Any]) -> bool:
    torch = _torch()
    if torch is None or not torch.is_grad_enabled():
        return False
    return any(is_torch_tensor(x) and x.requires_grad for x in xs)


def _as_array(x: Any, kind: ArrayKind) -> Any:
    """``x`` as an array of ``kind`` (numbers and lists are converted)."""
    if kind.lib == "torch":
        torch = _torch()
        if is_torch_tensor(x):
            return x
        dtype = kind.dtype or torch.float64
        return torch.as_tensor(x, dtype=dtype, device=kind.device)
    if kind.lib == "numpy":
        np = importlib.import_module("numpy")
        return np.asarray(x, dtype=kind.dtype or np.float64)
    return x


def _broadcast(xs: Sequence[Any], kind: ArrayKind) -> tuple[list[Any], tuple[int, ...]]:
    """The arrays broadcast to their common shape (torch: tracked by
    autograd), and that shape."""
    if kind.lib == "torch":
        torch = _torch()
        arrays = [_as_array(x, kind) for x in xs]
        out = list(torch.broadcast_tensors(*arrays))
        return out, tuple(out[0].shape)
    if kind.lib == "numpy":
        np = importlib.import_module("numpy")
        out = list(np.broadcast_arrays(*[_as_array(x, kind) for x in xs]))
        return out, tuple(out[0].shape)
    flats = [to_flat(x) for x in xs]
    shape = broadcast_shapes(*(f[1] for f in flats))
    if shape is None:
        raise RuntimeError(
            "The shapes "
            + ", ".join(str(tuple(f[1])) for f in flats)
            + " cannot be broadcast to a common shape."
        )
    out = [broadcast_to(f[0], f[1], shape) for f in flats]
    return out, shape


def _flat(x: Any) -> Any:
    if isinstance(x, list) and (not x or not isinstance(x[0], list)):
        return x
    return to_flat(x)[0]


_FUNCTIONS: dict[str, Any] = {}


def _elementwise_function() -> Any:
    fn = _FUNCTIONS.get("elementwise")
    if fn is not None:
        return fn
    torch = importlib.import_module("torch")

    class _Elementwise(torch.autograd.Function):
        @staticmethod
        def forward(  # type: ignore[override]
            ctx: Any, name: str, params: list[float], single: bool, *inputs: Any
        ) -> Any:
            shape = tuple(inputs[0].shape)
            kind = ArrayKind("torch", inputs[0].dtype, inputs[0].device)
            datas = [to_flat(t)[0] for t in inputs]
            values, grads = _rs.elementwise(name, datas, params, single, True)
            ctx.grads = [[from_flat(g, shape, kind) for g in per_out] for per_out in grads]
            outs = tuple(from_flat(v, shape, kind) for v in values)
            return outs if len(outs) > 1 else outs[0]

        @staticmethod
        def backward(ctx: Any, *grad_outputs: Any) -> Any:  # type: ignore[override]
            n_in = len(ctx.grads[0])
            result: list[Any] = [None] * n_in
            for o, g_out in enumerate(grad_outputs):
                if g_out is None:
                    continue
                for k in range(n_in):
                    term = g_out * ctx.grads[o][k]
                    result[k] = term if result[k] is None else result[k] + term
            return (None, None, None, *result)

    _FUNCTIONS["elementwise"] = _Elementwise
    return _Elementwise


def _reduce_function() -> Any:
    fn = _FUNCTIONS.get("reduce")
    if fn is not None:
        return fn
    torch = importlib.import_module("torch")

    class _Reduce(torch.autograd.Function):
        @staticmethod
        def forward(ctx: Any, name: str, params: list[float], x2d: Any) -> Any:  # type: ignore[override]
            rows, n = x2d.shape
            kind = ArrayKind("torch", x2d.dtype, x2d.device)
            values, grads = _rs.reduce(name, to_flat(x2d)[0], n, params, True)
            ctx.grads = from_flat(grads, (rows, n), kind)
            return from_flat(values, (rows,), kind)

        @staticmethod
        def backward(ctx: Any, grad_output: Any) -> Any:  # type: ignore[override]
            return None, None, grad_output.unsqueeze(-1) * ctx.grads

    _FUNCTIONS["reduce"] = _Reduce
    return _Reduce


def apply(name: str, *xs: Any, params: Sequence[float] = (), kind: ArrayKind | None = None) -> Any:
    """The element-wise function ``name`` of the broadcast arrays ``xs``
    (``params``: the scalar parameters of ``fatmaximum`` / ``fatminimum``).
    Returns an array (a tuple for functions with several outputs) of the kind
    of the first array argument."""
    kind = kind or first_kind(*xs)
    single = is_single_precision(kind)
    arrays, shape = _broadcast(xs, kind)
    if kind.lib == "torch" and _requires_grad(arrays):
        return _elementwise_function().apply(name, [float(p) for p in params], single, *arrays)
    values, _ = _rs.elementwise(
        name, [_flat(a) for a in arrays], [float(p) for p in params], single, False
    )
    outs = tuple(from_flat(v, shape, kind) for v in values)
    return outs if len(outs) > 1 else outs[0]


def normalize_dims(dim: Any, ndim: int) -> tuple[int, ...]:
    """``dim`` (an int or a tuple of ints, negative from the back) as a
    sorted tuple of non-negative dimensions; ``IndexError`` like torch for
    dimensions out of range."""
    dims = (dim,) if isinstance(dim, int) else tuple(dim)
    out = []
    for d in dims:
        if not isinstance(d, int):
            raise TypeError(f"`dim` must be an int or a tuple of ints, got {dim!r}.")
        nd = max(ndim, 1)
        if d < -nd or d >= nd:
            raise IndexError(
                f"Dimension out of range (expected to be in range of [{-nd}, {nd - 1}], but got {d})"
            )
        out.append(d % nd)
    if len(set(out)) != len(out):
        raise RuntimeError(f"dim {out} appears multiple times in the list of dims")
    return tuple(sorted(out))


def _permute_flat(
    data: Sequence[float], shape: tuple[int, ...], order: tuple[int, ...]
) -> list[float]:
    """Row-major ``data`` of ``shape`` with its dimensions permuted."""
    new_shape = tuple(shape[o] for o in order)
    strides = [math.prod(shape[i + 1 :]) for i in range(len(shape))]
    out: list[float] = []
    total = math.prod(new_shape)
    idx = [0] * len(new_shape)
    for _ in range(total):
        out.append(float(data[sum(idx[k] * strides[order[k]] for k in range(len(order)))]))
        for k in range(len(new_shape) - 1, -1, -1):
            idx[k] += 1
            if idx[k] < new_shape[k]:
                break
            idx[k] = 0
    return out


def reduce(
    name: str,
    x: Any,
    dim: Any,
    keepdim: bool = False,
    params: Sequence[float] = (),
    empty: float = -math.inf,
) -> Any:
    """The reduction ``name`` of ``x`` over the dimensions ``dim`` (an int
    or a tuple), keeping them with size 1 if ``keepdim``. Over empty
    dimensions the result is ``empty`` (BoTorch: ``x.sum(dim) - inf``)."""
    kind = kind_of(x)
    params = [float(p) for p in params]
    if kind.lib == "torch":
        shape = tuple(x.shape)
    else:
        shape = to_flat(x)[1]
    dims = normalize_dims(dim, len(shape))
    if not shape:
        shape_eff: tuple[int, ...] = (1,)
    else:
        shape_eff = shape
    keep = tuple(i for i in range(len(shape_eff)) if i not in dims)
    n = math.prod(shape_eff[i] for i in dims)
    out_shape = tuple(1 if i in dims else shape_eff[i] for i in range(len(shape_eff)))
    if not keepdim:
        out_shape = tuple(shape_eff[i] for i in keep)
    if not shape:
        out_shape = ()
    if n == 0:
        if kind.lib == "torch":
            return x.sum(dim=dims, keepdim=keepdim) + empty
        if kind.lib == "numpy":
            np = importlib.import_module("numpy")
            return np.sum(x, axis=dims, keepdims=keepdim) + empty
        return from_flat([empty] * math.prod(out_shape), out_shape, LIST)
    order = (*keep, *dims)
    if kind.lib == "torch":
        xp = x.reshape(shape_eff).permute(*order).reshape(-1, n)
        if _requires_grad([x]):
            values = _reduce_function().apply(name, params, xp)
            return values.reshape(out_shape)
        vals, _ = _rs.reduce(name, to_flat(xp)[0], n, params, False)
        return from_flat(vals, out_shape, kind)
    if kind.lib == "numpy":
        np = importlib.import_module("numpy")
        xp = np.transpose(np.reshape(np.asarray(x), shape_eff), order).reshape(-1, n)
        vals, _ = _rs.reduce(name, to_flat(xp)[0], n, params, False)
        return from_flat(vals, out_shape, kind)
    data = to_flat(x)[0]
    permuted = _permute_flat(data, shape_eff, order)
    vals, _ = _rs.reduce(name, permuted, n, params, False)
    return from_flat(vals, out_shape, kind)


def check_float_dtype(x: Any, name: str, exc: type[Exception] = TypeError) -> None:
    """Raises ``exc`` with BoTorch's message unless ``x`` has a float32 or
    float64 dtype (lists and numbers are float64)."""
    dt = dtype_name(x)
    if dt is not None and dt not in ("float32", "float64"):
        if exc is TypeError:
            raise TypeError(
                f"{name} only supports torch.float32 and torch.float64 dtypes, but "
                f"received x.dtype=torch.{dt}."
            )
        raise exc(f"Only dtypes float32 and float64 are supported, but received torch.{dt}.")


__all__ = ["apply", "check_float_dtype", "is_single_precision", "normalize_dims", "reduce"]


# -- Array manipulation for every kind of array ------------------------------


def shape_of(x: Any) -> tuple[int, ...]:
    """The shape of an array, nested list or number."""
    if is_torch_tensor(x):
        return tuple(x.shape)
    np = sys.modules.get("numpy")
    if np is not None and isinstance(x, (np.ndarray, np.generic)):
        return tuple(np.shape(x))
    return to_flat(x)[1]


def _basic_index(shape: tuple[int, ...], key: Any) -> tuple[list[list[int] | None], tuple[int, ...]]:
    """For basic indexing (ints, slices, Ellipsis, None) of an array of
    ``shape``: the selected indices per dimension (``None`` for an int, which
    drops the dimension, wrapped as ``[i]``) and the resulting shape."""
    if not isinstance(key, tuple):
        key = (key,)
    n_ell = sum(1 for k in key if k is Ellipsis)
    if n_ell > 1:
        raise IndexError("an index can only have a single ellipsis ('...')")
    n_real = sum(1 for k in key if k is not None and k is not Ellipsis)
    if n_real > len(shape):
        raise IndexError(f"too many indices for an array of dimension {len(shape)}")
    expanded: list[Any] = []
    for k in key:
        if k is Ellipsis:
            expanded.extend([slice(None)] * (len(shape) - n_real))
        else:
            expanded.append(k)
    if n_ell == 0:
        expanded.extend([slice(None)] * (len(shape) - n_real))
    sel: list[Any] = []
    out_shape: list[int] = []
    d = 0
    for k in expanded:
        if k is None:
            out_shape.append(1)
            sel.append(("new",))
            continue
        size = shape[d]
        if isinstance(k, bool) or not isinstance(k, (int, slice)):
            if hasattr(k, "__index__"):
                k = int(k.__index__())
            else:
                raise IndexError(
                    "botorch_rs supports ints, slices, Ellipsis and None to index lists."
                )
        if isinstance(k, int):
            i = k + size if k < 0 else k
            if not 0 <= i < size:
                raise IndexError(f"index {k} is out of bounds for dimension {d} with size {size}")
            sel.append(("int", i))
        else:
            idx = list(range(size))[k]
            sel.append(("slice", idx))
            out_shape.append(len(idx))
        d += 1
    return sel, tuple(out_shape)


def getitem(x: Any, key: Any) -> Any:
    """``x[key]`` for tensors, NumPy arrays and nested lists (basic indexing
    for lists)."""
    kind = kind_of(x)
    if kind.lib != "list":
        return x[key]
    data, shape = to_flat(x)
    sel, out_shape = _basic_index(shape, key)
    strides = [math.prod(shape[i + 1 :]) for i in range(len(shape))]
    # Offsets of the selected elements, dimension by dimension.
    offsets = [0]
    d = 0
    for s in sel:
        if s[0] == "new":
            continue
        if s[0] == "int":
            offsets = [o + s[1] * strides[d] for o in offsets]
        else:
            offsets = [o + i * strides[d] for o in offsets for i in s[1]]
        d += 1
    return from_flat([data[o] for o in offsets], out_shape, kind)


def concat(xs: Sequence[Any], dim: int) -> Any:
    """``torch.cat`` / ``np.concatenate`` of arrays of one kind (lists too)."""
    kind = first_kind(*xs)
    if kind.lib == "torch":
        return _torch().cat(list(xs), dim=dim)
    if kind.lib == "numpy":
        return importlib.import_module("numpy").concatenate(list(xs), axis=dim)
    flats = [to_flat(x) for x in xs]
    nd = len(flats[0][1])
    d = dim % nd
    for f in flats:
        if len(f[1]) != nd or any(f[1][i] != flats[0][1][i] for i in range(nd) if i != d):
            raise RuntimeError("Sizes of tensors must match except in the concatenated dimension.")
    outer = math.prod(flats[0][1][:d])
    out: list[float] = []
    for o in range(outer):
        for data, shape in flats:
            inner = math.prod(shape[d:])
            out.extend(float(v) for v in data[o * inner : (o + 1) * inner])
    out_shape = list(flats[0][1])
    out_shape[d] = sum(f[1][d] for f in flats)
    return from_flat(out, tuple(out_shape), kind)


def expand(x: Any, shape: Sequence[int]) -> Any:
    """``x`` broadcast to ``shape`` (``-1`` keeps a dimension)."""
    shape = tuple(int(s) for s in shape)
    kind = kind_of(x)
    if kind.lib == "torch":
        return x.expand(*shape)
    xs = shape_of(x)
    target = tuple(
        xs[i - (len(shape) - len(xs))] if s == -1 else s for i, s in enumerate(shape)
    )
    if kind.lib == "numpy":
        return importlib.import_module("numpy").broadcast_to(x, target)
    data, src = to_flat(x)
    return from_flat(broadcast_to(data, src, target), target, kind)


def clone(x: Any) -> Any:
    """A copy of an array (``detach().clone()`` is not implied)."""
    kind = kind_of(x)
    if kind.lib == "torch":
        return x.clone()
    if kind.lib == "numpy":
        return importlib.import_module("numpy").array(x, copy=True)
    data, shape = to_flat(x)
    return from_flat(list(data), shape, kind)


def detach(x: Any) -> Any:
    """``x.detach()`` for tensors, ``x`` otherwise."""
    return x.detach() if is_torch_tensor(x) else x


def int_array(values: Sequence[int], shape: Sequence[int], like: ArrayKind) -> Any:
    """An integer array (``torch.long`` tensor, ``int64`` NumPy array or
    nested lists of ints) of the given shape."""
    shape = tuple(shape)
    vals = [int(v) for v in values]
    if like.lib == "torch":
        torch = _torch()
        t = torch.tensor(vals, dtype=torch.long).reshape(shape)
        if like.device is not None and like.device.type != "cpu":
            t = t.to(like.device)
        return t
    if like.lib == "numpy":
        np = importlib.import_module("numpy")
        return np.asarray(vals, dtype=np.int64).reshape(shape)
    if not shape:
        return vals[0]

    def build(offset: int, dims: tuple[int, ...]) -> list[Any]:
        if len(dims) == 1:
            return vals[offset : offset + dims[0]]
        step = math.prod(dims[1:])
        return [build(offset + i * step, dims[1:]) for i in range(dims[0])]

    return build(0, shape)


def ints(x: Any) -> list[int]:
    """The values of an integer array as Python ints (row-major)."""
    if is_torch_tensor(x):
        return [int(v) for v in x.reshape(-1).tolist()]
    np = sys.modules.get("numpy")
    if np is not None and isinstance(x, (np.ndarray, np.generic)):
        return [int(v) for v in np.asarray(x).reshape(-1).tolist()]
    if isinstance(x, (int,)):
        return [int(x)]
    data, _ = to_flat(x)
    return [int(v) for v in data]


def matmul(a: Any, b: Any) -> Any:
    """``a @ b`` (for nested lists: 2-dimensional matrices)."""
    kind = first_kind(a, b)
    if kind.lib != "list":
        return _as_array(a, kind) @ _as_array(b, kind)
    da, sa = to_flat(a)
    db, sb = to_flat(b)
    if len(sa) != 2 or len(sb) != 2 or sa[1] != sb[0]:
        raise RuntimeError(f"mat1 and mat2 shapes cannot be multiplied ({sa} and {sb})")
    r, k, c = sa[0], sa[1], sb[1]
    out = [
        math.fsum(float(da[i * k + t]) * float(db[t * c + j]) for t in range(k))
        for i in range(r)
        for j in range(c)
    ]
    return from_flat(out, (r, c), kind)


def transpose(a: Any) -> Any:
    """``a`` with its last two dimensions swapped."""
    kind = kind_of(a)
    if kind.lib == "torch":
        return a.transpose(-1, -2)
    if kind.lib == "numpy":
        return importlib.import_module("numpy").swapaxes(a, -1, -2)
    data, shape = to_flat(a)
    order = (*range(len(shape) - 2), len(shape) - 1, len(shape) - 2)
    new_shape = tuple(shape[o] for o in order)
    return from_flat(_permute_flat(data, shape, order), new_shape, kind)


def full(shape: Sequence[int], value: float, kind: ArrayKind) -> Any:
    """An array of ``shape`` filled with ``value``."""
    shape = tuple(shape)
    return from_flat([float(value)] * math.prod(shape), shape, kind)


def eye(n: int, kind: ArrayKind) -> Any:
    """The ``n x n`` identity."""
    return from_flat([1.0 if i == j else 0.0 for i in range(n) for j in range(n)], (n, n), kind)


def _div(x: float, y: float) -> float:
    if y != 0:
        return x / y
    if x == 0 or math.isnan(x):
        return math.nan
    return math.copysign(math.inf, x) * math.copysign(1.0, y)


_ARITH = {
    "add": lambda x, y: x + y,
    "sub": lambda x, y: x - y,
    "mul": lambda x, y: x * y,
    "div": _div,
}


def arith(op: str, a: Any, b: Any) -> Any:
    """Broadcast ``a <op> b`` (``op``: add, sub, mul, div) for every kind of
    array (plain arithmetic, not the safe functions)."""
    kind = first_kind(a, b)
    if kind.lib != "list":
        x, y = _as_array(a, kind), _as_array(b, kind)
        if op == "div":
            return x / y
        return _ARITH[op](x, y)
    flats = [to_flat(a), to_flat(b)]
    shape = broadcast_shapes(flats[0][1], flats[1][1])
    if shape is None:
        raise RuntimeError(f"The shapes {flats[0][1]} and {flats[1][1]} cannot be broadcast.")
    xa = broadcast_to(flats[0][0], flats[0][1], shape)
    xb = broadcast_to(flats[1][0], flats[1][1], shape)
    f = _ARITH[op]
    return from_flat([f(float(u), float(v)) for u, v in zip(xa, xb, strict=True)], shape, kind)

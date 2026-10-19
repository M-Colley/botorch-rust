"""Standard-normal utilities (``botorch.utils.probability.utils``): ``ndtr``,
``phi``, ``log_phi``, ``log_ndtr``, ``log_erfc``, ``log_erfcx``,
``standard_normal_log_hazard``, ``log_prob_normal_in``,
``compute_log_prob_feas_from_bounds`` and ``percentile_of_score`` evaluated
in Rust (BoTorch's formulas, with exact first derivatives for torch tensors
that require gradients), and the tensor helpers ``case_dispatcher``,
``gen_positional_indices``, ``build_positional_indices`` and
``swap_along_dim_`` for torch tensors, NumPy arrays and nested lists.
"""

from __future__ import annotations

import importlib
import math
import sys
from collections.abc import Callable, Iterable, Iterator
from functools import lru_cache
from math import pi
from typing import Any

from botorch_rs import _botorch_rs as _rs
from botorch_rs import _ops
from botorch_rs._arrays import (
    LIST,
    ArrayKind,
    first_kind,
    from_flat,
    is_torch_tensor,
    kind_from_dtype,
    kind_of,
    to_flat,
)

CaseNd = tuple[Callable[[], Any], Callable[[Any], Any]]

_log_2 = math.log(2)
_inv_sqrt_2pi = 1 / math.sqrt(2 * pi)
_inv_sqrt_2 = 1 / math.sqrt(2)
_neg_inv_sqrt_2 = -_inv_sqrt_2
_log_sqrt_2pi = math.log(2 * pi) / 2
STANDARDIZED_RANGE: tuple[float, float] = (-1e6, 1e6)
_log_two_inv_sqrt_2pi = _log_2 - _log_sqrt_2pi  # = log(2 / sqrt(2 * pi))


def _is_numpy(x: Any) -> bool:
    np = sys.modules.get("numpy")
    return np is not None and isinstance(x, (np.ndarray, np.generic))


def _any(mask: Any) -> bool:
    if is_torch_tensor(mask) or _is_numpy(mask):
        return bool(mask.any())
    if isinstance(mask, bool):
        return mask
    return any(bool(v) for v in to_flat(mask)[0])


def _all(mask: Any) -> bool:
    if is_torch_tensor(mask) or _is_numpy(mask):
        return bool(mask.all())
    if isinstance(mask, bool):
        return mask
    return all(bool(v) for v in to_flat(mask)[0])


def _list_mask(mask: Any, shape: tuple[int, ...]) -> list[bool]:
    data, mshape = to_flat(mask)
    from botorch_rs._arrays import broadcast_to

    return [bool(v) for v in broadcast_to([float(v) for v in data], mshape, shape)]


def case_dispatcher(
    out: Any,
    cases: Iterable[CaseNd] = (),
    default: Callable[[Any], Any] | None = None,
) -> Any:
    r"""Basic implementation of a tensorized switching case statement.

    Args:
        out: Tensor to which case outcomes are written.
        cases: Iterable of function pairs (pred, func), where ``mask=pred()``
            specifies whether ``func`` is applicable for each entry in ``out``. Note
            that cases are resolved first-come, first-serve.
        default: Optional ``func`` to which all unclaimed entries of ``out`` are
            dispatched.

    Torch tensors and NumPy arrays are updated in place with boolean-mask
    indexing, as in BoTorch; nested lists are updated in place too (a mask is
    a nested list of booleans and ``func(mask)`` returns the values of the
    masked entries in row-major order, or a scalar).
    """
    if is_torch_tensor(out) or _is_numpy(out):
        active = None
        for closure, func in cases:
            pred = closure()
            if not _any(pred):
                continue
            mask = pred if (active is None) else pred & active
            if not _any(mask):
                continue
            if _all(mask):  # where possible, use Ellipsis to avoid indexing
                out[...] = func(...)
                return out
            out[mask] = func(mask)
            if active is None:
                active = ~mask
            else:
                active[mask] = False
            if not _any(active):
                break
        if default is not None:
            if active is None:
                out[...] = default(...)
            elif _any(active):
                out[active] = default(active)
        return out
    return _case_dispatcher_list(out, cases, default)


def _assign_list(out: list[Any], shape: tuple[int, ...], mask: list[bool], values: Any) -> None:
    flat, _ = to_flat(out)
    flat = list(flat)
    n_sel = sum(mask)
    vals, vshape = to_flat(values)
    vals = list(vals)
    if len(vals) == 1 and n_sel != 1:
        vals = vals * n_sel
    if len(vals) == len(mask) and n_sel != len(mask):
        vals = [v for v, m in zip(vals, mask, strict=True) if m]
    if len(vals) != n_sel:
        raise RuntimeError(
            f"shape mismatch: value of shape {tuple(vshape)} cannot be assigned to {n_sel} entries"
        )
    it = iter(vals)
    for i, m in enumerate(mask):
        if m:
            flat[i] = float(next(it))
    new = from_flat(flat, shape, LIST)
    if isinstance(new, list):
        out[:] = new


def _case_dispatcher_list(
    out: Any, cases: Iterable[CaseNd], default: Callable[[Any], Any] | None
) -> Any:
    _, shape = to_flat(out)
    size = math.prod(shape)
    active: list[bool] | None = None
    for closure, func in cases:
        pred = _list_mask(closure(), shape)
        if not any(pred):
            continue
        mask = pred if active is None else [p and a for p, a in zip(pred, active, strict=True)]
        if not any(mask):
            continue
        if all(mask):
            _assign_list(out, shape, [True] * size, func(...))
            return out
        _assign_list(out, shape, mask, func(from_flat(mask, shape, LIST)))
        if active is None:
            active = [not m for m in mask]
        else:
            active = [a and not m for a, m in zip(active, mask, strict=True)]
        if not any(active):
            break
    if default is not None:
        if active is None:
            _assign_list(out, shape, [True] * size, default(...))
        elif any(active):
            _assign_list(out, shape, active, default(from_flat(active, shape, LIST)))
    return out


def gen_positional_indices(
    shape: Any,
    dim: int,
    device: Any = None,
) -> Iterator[Any]:
    """The positional indices of the dimensions ``0..dim`` of an array of
    ``shape`` (BoTorch's ``gen_positional_indices``): one integer array per
    dimension, last dimension first, shaped to broadcast against each other
    (torch ``long`` tensors if torch is imported or ``device`` is given, else
    nested lists of ints)."""
    shape = tuple(int(s) for s in shape)
    ndim = len(shape)
    _dim = ndim + dim if dim < 0 else dim
    if _dim >= ndim or _dim < 0:
        raise ValueError(f"dim={dim} invalid for shape {shape}.")
    kind = kind_from_dtype(None, device)
    if device is not None and kind.lib != "torch":
        kind = kind_from_dtype(importlib.import_module("torch").float64, device)
    cumsize = math.prod(shape[_dim + 1 :])
    for i, s in enumerate(reversed(shape[: _dim + 1])):
        values = list(range(0, s * cumsize, cumsize))
        yield _ops.int_array(values, (s, *(1,) * i), kind)
        cumsize *= s


def build_positional_indices(
    shape: Any,
    dim: int,
    device: Any = None,
) -> Any:
    """The linear indices of the slices of the dimensions ``0..dim`` (the
    sum of :func:`gen_positional_indices`)."""
    gens = list(gen_positional_indices(shape=shape, dim=dim, device=device))
    if is_torch_tensor(gens[0]):
        return sum(gens)
    from botorch_rs._arrays import broadcast_shapes, broadcast_to

    shapes = [_ops.shape_of(g) for g in gens]
    out_shape = broadcast_shapes(*shapes) or ()
    total = [0] * math.prod(out_shape)
    for g, s in zip(gens, shapes, strict=True):
        data = broadcast_to([float(v) for v in _ops.ints(g)], s, out_shape)
        total = [t + int(v) for t, v in zip(total, data, strict=True)]
    return _ops.int_array(total, out_shape, LIST)


@lru_cache(maxsize=None)
def _leggauss(deg: int) -> tuple[tuple[float, ...], tuple[float, ...]]:
    x, w = _rs.leggauss(deg)
    return tuple(x), tuple(w)


def leggauss(deg: int, **tkwargs: Any) -> tuple[Any, Any]:
    """The Gauss–Legendre nodes and weights of degree ``deg`` (NumPy's
    ``leggauss``), as arrays of ``tkwargs``' dtype and device."""
    x, w = _leggauss(int(deg))
    kind = kind_from_dtype(tkwargs.get("dtype"), tkwargs.get("device"))
    return from_flat(list(x), (deg,), kind), from_flat(list(w), (deg,), kind)


def ndtr(x: Any) -> Any:
    r"""Standard normal CDF."""
    return _ops.apply("ndtr", x)


def phi(x: Any) -> Any:
    r"""Standard normal PDF."""
    return _ops.apply("phi", x)


def log_phi(x: Any) -> Any:
    r"""Logarithm of standard normal pdf"""
    return _ops.apply("log_phi", x)


def log_ndtr(x: Any) -> Any:
    """Implementation of log_ndtr that remedies problems of torch.special's version
    for large negative x, where the torch implementation yields Inf or NaN gradients.

    Args:
        x: An input tensor with dtype torch.float32 or torch.float64.

    Returns:
        A tensor of values of the same type and shape as x containing log(ndtr(x)).
    """
    _ops.check_float_dtype(x, "log_Phi")
    return _ops.apply("log_ndtr", x)


def log_erfc(x: Any) -> Any:
    """Computes the logarithm of the complementary error function in a numerically
    stable manner. The GitHub issue https://github.com/pytorch/pytorch/issues/31945
    tracks progress toward moving this feature into PyTorch in C++.

    Args:
        x: An input tensor with dtype torch.float32 or torch.float64.

    Returns:
        A tensor of values of the same type and shape as x containing log(erfc(x)).
    """
    _ops.check_float_dtype(x, "log_erfc")
    return _ops.apply("log_erfc", x)


def log_erfcx(x: Any) -> Any:
    """Computes the logarithm of the complementary scaled error function in a
    numerically stable manner. The GitHub issue tracks progress toward moving this
    feature into PyTorch in C++: https://github.com/pytorch/pytorch/issues/31945.

    Args:
        x: An input tensor with dtype torch.float32 or torch.float64.

    Returns:
        A tensor of values of the same type and shape as x containing log(erfcx(x)).
    """
    return _ops.apply("log_erfcx", x)


def standard_normal_log_hazard(x: Any) -> Any:
    """Computes the logarithm of the hazard function of the standard normal
    distribution, i.e. ``log(phi(x) / Phi(-x))``.

    Args:
        x: A tensor of any shape, with either float32 or float64 dtypes.

    Returns:
        A Tensor of the same shape ``x``, containing the values of the logarithm of the
        hazard function evaluated at ``x``.
    """
    return _ops.apply("standard_normal_log_hazard", x)


def log_prob_normal_in(a: Any, b: Any) -> Any:
    r"""Computes the probability that a standard normal random variable takes a value
    in \[a, b\], i.e. log(Phi(b) - Phi(a)), where Phi is the standard normal CDF.
    Returns accurate values and permits numerically stable backward passes for inputs
    in [-1e100, 1e100] for double precision and [-1e20, 1e20] for single precision.
    In contrast, a naive approach is not numerically accurate beyond [-10, 10].

    Args:
        a: Tensor of lower integration bounds of the Gaussian probability measure.
        b: Tensor of upper integration bounds of the Gaussian probability measure.

    Returns:
        Tensor of the log probabilities.
    """
    try:
        return _ops.apply("log_prob_normal_in", a, b)
    except ValueError as e:
        raise ValueError(str(e)) from None


def swap_along_dim_(
    values: Any,
    i: Any,
    j: Any,
    dim: int,
    buffer: Any = None,
) -> Any:
    r"""Swaps Tensor slices in-place along dimension ``dim``.

    When passed as Tensors, ``i`` (and ``j``) should be ``dim``-dimensional tensors
    with the same shape as ``values.shape[:dim]``. The exception to this rule occurs
    when ``dim=0``, in which case ``i`` (and ``j``) should be (at most) one-dimensional
    when passed as a Tensor.

    Args:
        values: Tensor whose values are to be swapped.
        i: Indices for slices along dimension ``dim``.
        j: Indices for slices along dimension ``dim``.
        dim: The dimension of ``values`` along which to swap slices.
        buffer: Optional buffer used internally to store copied values.

    Returns:
        The original ``values`` tensor.

    Torch tensors and NumPy arrays are swapped with their indexing (as in
    BoTorch); nested lists are swapped in place element by element.
    """
    shape = _ops.shape_of(values)
    dim = len(shape) + dim if dim < 0 else dim

    def _ndim(v: Any) -> int:
        return 0 if isinstance(v, int) else len(_ops.shape_of(v))

    batched = dim and (_ndim(i) or _ndim(j))
    if batched:
        if _ndim(i) and _ops.shape_of(i) != tuple(shape[:dim]):
            raise ValueError("Batch shapes of `i` and `values` do not match.")
        if _ndim(j) and _ops.shape_of(j) != tuple(shape[:dim]):
            raise ValueError("Batch shapes of `j` and `values` do not match.")
    else:
        if not isinstance(i, int) and _ndim(i) > 1:
            raise ValueError("Tensor `i` must be at most 1-dimensional when `dim=0`.")
        if not isinstance(j, int) and _ndim(j) > 1:
            raise ValueError("Tensor `j` must be at most 1-dimensional when `dim=0`.")
    if is_torch_tensor(values) or _is_numpy(values):
        return _swap_native(values, i, j, dim, buffer, bool(batched), shape)
    return _swap_list(values, i, j, dim, bool(batched), shape)


def _swap_native(
    values: Any, i: Any, j: Any, dim: int, buffer: Any, batched: bool, shape: tuple[int, ...]
) -> Any:
    if batched:
        nb = math.prod(shape[:dim])
        size = shape[dim]
        ii = [int(i)] * nb if isinstance(i, int) else _ops.ints(i)
        jj = [int(j)] * nb if isinstance(j, int) else _ops.ints(j)
        kind = kind_of(values)
        idx_i = _ops.int_array([b * size + v for b, v in enumerate(ii)], (nb,), kind)
        idx_j = _ops.int_array([b * size + v for b, v in enumerate(jj)], (nb,), kind)
        view = values.reshape(-1, *shape[dim + 1 :])
        _swap_native(view, idx_i, idx_j, 0, buffer, False, tuple(view.shape))
        return values
    ctx = (slice(None),) * dim
    ki = ctx + (i,)
    kj = ctx + (j,)
    if buffer is None:
        buffer = values[ki].copy() if _is_numpy(values) else values[ki].clone()
    else:
        buffer[...] = values[ki]
    values[ki] = values[kj]
    values[kj] = buffer
    return values


def _swap_list(
    values: list[Any], i: Any, j: Any, dim: int, batched: bool, shape: tuple[int, ...]
) -> list[Any]:
    data, _ = to_flat(values)
    flat = list(data)
    inner = math.prod(shape[dim + 1 :])
    size = shape[dim]
    nb = math.prod(shape[:dim])

    def rows(b: int, idx: list[int]) -> list[list[float]]:
        return [flat[(b * size + a) * inner : (b * size + a + 1) * inner] for a in idx]

    def write(b: int, idx: list[int], vals: list[list[float]]) -> None:
        for a, v in zip(idx, vals, strict=True):
            flat[(b * size + a) * inner : (b * size + a + 1) * inner] = v

    for b in range(nb):
        if batched:
            ii = [int(i) if isinstance(i, int) else _ops.ints(i)[b]]
            jj = [int(j) if isinstance(j, int) else _ops.ints(j)[b]]
        else:
            ii = [int(i)] if isinstance(i, int) else _ops.ints(i)
            jj = [int(j)] if isinstance(j, int) else _ops.ints(j)
            if len(ii) == 1 and len(jj) > 1:
                ii = ii * len(jj)
            if len(jj) == 1 and len(ii) > 1:
                jj = jj * len(ii)
        buf = rows(b, ii)
        write(b, ii, rows(b, jj))
        write(b, jj, buf)
    new = from_flat(flat, shape, LIST)
    values[:] = new
    return values


def compute_log_prob_feas_from_bounds(
    con_lower_inds: Any,
    con_upper_inds: Any,
    con_both_inds: Any,
    con_lower: Any,
    con_upper: Any,
    con_both: Any,
    means: Any,
    sigmas: Any,
) -> Any:
    r"""Compute logarithm of the feasibility probability for each batch of mean/sigma.

    Args:
        means: A ``(b) x m``-dim Tensor of means.
        sigmas: A ``(b) x m``-dim Tensor of standard deviations.
        con_lower_inds: 1d Tensor of indices con_lower applies to
            in the second dimension of means and sigmas.
        con_upper_inds: 1d Tensor of indices con_upper applies to
            in the second dimension of means and sigmas.
        con_both_inds: 1d Tensor of indices con_both applies to
            in the second dimension of means and sigmas.
        con_lower: 1d Tensor of lower bounds on the constraints
            equal in dimension to con_lower_inds.
        con_upper: 1d Tensor of upper bounds on the constraints
            equal in dimension to con_upper_inds.
        con_both: 2d Tensor of "both" bounds on the constraints
            equal in length to con_both_inds.

    Returns:
        A ``(b)``-dim tensor of log feasibility probabilities

    Gradients flow to ``means`` and ``sigmas`` (not to the bounds).
    """
    kind = first_kind(means, sigmas)
    lo = [float(v) for v in to_flat(con_lower, "con_lower")[0]]
    up = [float(v) for v in to_flat(con_upper, "con_upper")[0]]
    both = [float(v) for v in to_flat(con_both, "con_both")[0]]
    li, ui, bi = _ops.ints(con_lower_inds), _ops.ints(con_upper_inds), _ops.ints(con_both_inds)
    if len(li) != len(lo) or len(ui) != len(up) or 2 * len(bi) != len(both):
        raise ValueError("The constraint indices and bounds do not match.")
    lower = list(zip(li, lo, strict=True))
    upper = list(zip(ui, up, strict=True))
    two = [(k, both[2 * n], both[2 * n + 1]) for n, k in enumerate(bi)]
    arrays, shape = _ops._broadcast([means, sigmas], kind)
    m = shape[-1] if shape else 1
    out_shape = shape[:-1]
    if kind.lib == "torch" and _ops._requires_grad(arrays):
        fn = _log_prob_feas_function()
        return fn.apply(lower, upper, two, m, arrays[0], arrays[1])
    try:
        values, _ = _rs.log_prob_feas_from_bounds(
            lower, upper, two, _ops._flat(arrays[0]), _ops._flat(arrays[1]), m, False
        )
    except ValueError as e:
        raise ValueError(str(e)) from None
    return from_flat(values, out_shape, kind)


def _log_prob_feas_function() -> Any:
    fn = _ops._FUNCTIONS.get("log_prob_feas")
    if fn is not None:
        return fn
    torch = importlib.import_module("torch")

    class _LogProbFeas(torch.autograd.Function):
        @staticmethod
        def forward(  # type: ignore[override]
            ctx: Any, lower: Any, upper: Any, both: Any, m: int, means: Any, sigmas: Any
        ) -> Any:
            kind = ArrayKind("torch", means.dtype, means.device)
            shape = tuple(means.shape)
            values, grads = _rs.log_prob_feas_from_bounds(
                lower, upper, both, to_flat(means)[0], to_flat(sigmas)[0], m, True
            )
            ctx.gm = from_flat(grads[0], shape, kind)
            ctx.gs = from_flat(grads[1], shape, kind)
            return from_flat(values, shape[:-1], kind)

        @staticmethod
        def backward(ctx: Any, grad_output: Any) -> Any:  # type: ignore[override]
            g = grad_output.unsqueeze(-1)
            return None, None, None, None, g * ctx.gm, g * ctx.gs

    _ops._FUNCTIONS["log_prob_feas"] = _LogProbFeas
    return _LogProbFeas


def percentile_of_score(data: Any, score: Any, dim: int = -1) -> Any:
    """Compute the percentile rank of ``score`` relative to ``data``.
    For example, if this function returns 70 then 70% of the
    values in ``data`` are below ``score``.

    This implementation is based on ``scipy.stats.percentileofscore``,
    with ``kind='rank'`` and ``nan_policy='propagate'``, which is the default.

    Args:
        data: A ``... x n x output_shape``-dim Tensor of data.
        score: A ``... x 1 x output_shape``-dim Tensor of scores.

    Returns:
        A ``... x output_shape``-dim Tensor of percentile ranks.

    For torch tensors the result has torch's default dtype and is computed
    in that precision, as in BoTorch (an integer count times a Python float).
    """
    kind = first_kind(data, score)
    dshape = _ops.shape_of(data)
    d = dim % len(dshape)
    n = dshape[d]
    # Move `dim` to the end and broadcast the scores along it.
    arrays, shape = _ops._broadcast([data, score], kind)
    order = (*[k for k in range(len(shape)) if k != d], d)
    flats = []
    for a in arrays:
        # Nested lists are broadcast to flat (row-major) lists.
        f, s = to_flat(a) if kind.lib != "list" else (list(a), shape)
        flats.append(_ops._permute_flat(f, s, order) if order != tuple(range(len(s))) else list(f))
    out_shape = tuple(shape[k] for k in order[:-1])
    rows = math.prod(out_shape)
    scores = [flats[1][r * n] if n else float("nan") for r in range(rows)]
    if n == 0:
        scores = [float(v) for v in to_flat(score)[0]][:rows] or [float("nan")] * rows
    single = False
    if kind.lib == "torch":
        torch = sys.modules["torch"]
        single = torch.get_default_dtype() == torch.float32
        kind = ArrayKind("torch", torch.get_default_dtype(), kind.device)
    values = _rs.percentile_of_score(flats[0], scores, n, single)
    return from_flat(values, out_shape, kind)


__all__ = [
    "STANDARDIZED_RANGE",
    "build_positional_indices",
    "case_dispatcher",
    "compute_log_prob_feas_from_bounds",
    "gen_positional_indices",
    "leggauss",
    "log_erfc",
    "log_erfcx",
    "log_ndtr",
    "log_phi",
    "log_prob_normal_in",
    "ndtr",
    "percentile_of_score",
    "phi",
    "standard_normal_log_hazard",
    "swap_along_dim_",
]

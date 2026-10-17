"""PyTorch ≥ 2.14's subgradients of ``clamp`` at its bounds, for the parity
tests under older torch versions.

PyTorch 2.14 (pytorch#191142) changed the gradients of ``clamp``,
``clamp_min``, ``clamp_max`` and ``clip`` exactly at their bounds:

* a scalar bound is a fixed parameter, so the input's gradient at equality
  is the minimum-norm subgradient 0 (it was 1);
* a tensor bound is part of the differentiable input space, so the gradient
  is split evenly between the input and the bound at a tie (the input got
  all of it), also when the bound does not require a gradient.

``fmin`` / ``fmax`` split ties evenly too; ``torch.minimum`` /
``torch.maximum``, ``max`` / ``min`` / ``amax`` / ``amin`` reductions,
``relu`` and ``where`` did not change. botorch_rs follows the ≥ 2.14
convention, so its gradients differ from BoTorch's under torch ≤ 2.13 exactly
at clamp bounds.

:func:`torch214_clamp_subgradients` makes BoTorch (and GPyTorch and
linear_operator) compute the ≥ 2.14 gradients under torch ≤ 2.13: it
replaces the clamp family with a ``torch.autograd.Function`` that returns the
native forward value (bitwise) and applies the ≥ 2.14 rule in the backward
pass. Under torch ≥ 2.14 it does nothing, so the tests compare directly
against BoTorch. NaN inputs or bounds are not emulated (their gradients are
0 here; torch 2.14 changed some of those cases too), and neither is
forward-mode AD.
"""

from __future__ import annotations

import contextlib
import numbers
from collections.abc import Iterator
from typing import Any

import torch

#: Whether the installed torch has the ≥ 2.14 clamp subgradients natively.
TORCH_GE_214: bool = tuple(int(p) for p in torch.__version__.split("+")[0].split(".")[:2]) >= (
    2,
    14,
)

_NATIVE = {
    "clamp": torch.clamp,
    "clip": torch.clip,
    "clamp_min": torch.clamp_min,
    "clamp_max": torch.clamp_max,
}
_NATIVE_METHODS = {
    name: getattr(torch.Tensor, name)
    for name in (
        "clamp",
        "clip",
        "clamp_min",
        "clamp_max",
        "clamp_",
        "clip_",
        "clamp_min_",
        "clamp_max_",
    )
}


def _is_number(b: Any) -> bool:
    return b is None or (isinstance(b, numbers.Number) and not isinstance(b, bool))


class _Clamp214(torch.autograd.Function):
    """``torch.clamp(x, lo, hi)`` with torch ≥ 2.14's backward."""

    @staticmethod
    def forward(ctx: Any, x: Any, lo: Any, hi: Any) -> Any:  # type: ignore[override]
        tensor_bounds = isinstance(lo, torch.Tensor) or isinstance(hi, torch.Tensor)
        ctx.tensor_bounds = tensor_bounds
        ctx.lo_is_t = isinstance(lo, torch.Tensor)
        ctx.hi_is_t = isinstance(hi, torch.Tensor)
        ctx.scalars = (None if ctx.lo_is_t else lo, None if ctx.hi_is_t else hi)
        saved = [x]
        saved += [lo] if ctx.lo_is_t else []
        saved += [hi] if ctx.hi_is_t else []
        ctx.save_for_backward(*saved)
        with torch.no_grad():
            return _NATIVE["clamp"](x, lo, hi)

    @staticmethod
    def backward(ctx: Any, g: Any) -> Any:  # type: ignore[override]
        saved = list(ctx.saved_tensors)
        x = saved.pop(0)
        lo = saved.pop(0) if ctx.lo_is_t else ctx.scalars[0]
        hi = saved.pop(0) if ctx.hi_is_t else ctx.scalars[1]
        zero = torch.zeros((), dtype=g.dtype, device=g.device)
        if not ctx.tensor_bounds:
            inside = torch.ones_like(x, dtype=torch.bool)
            if lo is not None:
                inside = inside & (x > lo)
            if hi is not None:
                inside = inside & (x < hi)
            return torch.where(inside, g, zero), None, None
        x_b = x
        lo_b = (
            None
            if lo is None
            else torch.broadcast_to(lo, torch.broadcast_shapes(x.shape, lo.shape))
        )
        hi_b = (
            None
            if hi is None
            else torch.broadcast_to(hi, torch.broadcast_shapes(x.shape, hi.shape))
        )
        shape = g.shape
        one = torch.ones(shape, dtype=g.dtype, device=g.device)
        half = 0.5 * one
        wx = torch.zeros(shape, dtype=g.dtype, device=g.device)
        wlo = torch.zeros_like(wx)
        whi = torch.zeros_like(wx)
        if lo_b is not None and hi_b is not None:
            inverted = lo_b > hi_b
            degenerate = lo_b == hi_b
            regular = lo_b < hi_b
            whi = torch.where(inverted, one, whi)
            # lo == hi: the input keeps a tie with both bounds.
            wx = torch.where(degenerate & (x_b == lo_b), one, wx)
            wlo = torch.where(degenerate & (x_b < lo_b), one, wlo)
            whi = torch.where(degenerate & (x_b > hi_b), one, whi)
            wlo = torch.where(regular & (x_b < lo_b), one, wlo)
            wx = torch.where(regular & (x_b == lo_b), half, wx)
            wlo = torch.where(regular & (x_b == lo_b), half, wlo)
            wx = torch.where(regular & (x_b > lo_b) & (x_b < hi_b), one, wx)
            wx = torch.where(regular & (x_b == hi_b), half, wx)
            whi = torch.where(regular & (x_b == hi_b), half, whi)
            whi = torch.where(regular & (x_b > hi_b), one, whi)
        elif lo_b is not None:
            wlo = torch.where(x_b < lo_b, one, wlo)
            wx = torch.where(x_b == lo_b, half, wx)
            wlo = torch.where(x_b == lo_b, half, wlo)
            wx = torch.where(x_b > lo_b, one, wx)
        else:
            assert hi_b is not None
            whi = torch.where(x_b > hi_b, one, whi)
            wx = torch.where(x_b == hi_b, half, wx)
            whi = torch.where(x_b == hi_b, half, whi)
            wx = torch.where(x_b < hi_b, one, wx)

        def reduce(t: Any, like: Any) -> Any:
            return (g * t).sum_to_size(like.shape)

        gx = reduce(wx, x)
        glo = reduce(wlo, lo) if ctx.lo_is_t and ctx.needs_input_grad[1] else None
        ghi = reduce(whi, hi) if ctx.hi_is_t and ctx.needs_input_grad[2] else None
        return gx, glo, ghi


def _needs_emulation(x: Any, lo: Any, hi: Any) -> bool:
    if not torch.is_grad_enabled() or not isinstance(x, torch.Tensor):
        return False
    if not x.is_floating_point():
        return False
    if not (_is_number(lo) or isinstance(lo, torch.Tensor)):
        return False
    if not (_is_number(hi) or isinstance(hi, torch.Tensor)):
        return False
    if lo is None and hi is None:
        return False
    return any(isinstance(t, torch.Tensor) and t.requires_grad for t in (x, lo, hi))


def clamp214(x: Any, lo: Any = None, hi: Any = None) -> Any:
    """``torch.clamp(x, lo, hi)`` with torch ≥ 2.14's subgradients."""
    if not _needs_emulation(x, lo, hi):
        return _NATIVE["clamp"](x, lo, hi)
    return _Clamp214.apply(x, lo, hi)


def _split(args: tuple, kwargs: dict, names: tuple[str, ...]) -> tuple[Any, ...]:
    out = list(args) + [None] * (len(names) - len(args))
    for i, n in enumerate(names):
        if n in kwargs:
            out[i] = kwargs.pop(n)
    return tuple(out)


def _function(name: str) -> Any:
    native = _NATIVE[name]

    def f(*args: Any, **kwargs: Any) -> Any:
        if "out" in kwargs:
            return native(*args, **kwargs)
        kw = dict(kwargs)
        if name == "clamp_min":
            x, lo = _split(args, kw, ("input", "min"))
            hi = None
        elif name == "clamp_max":
            x, hi = _split(args, kw, ("input", "max"))
            lo = None
        else:
            x, lo, hi = _split(args, kw, ("input", "min", "max"))
        if kw or not _needs_emulation(x, lo, hi):
            return native(*args, **kwargs)
        return _Clamp214.apply(x, lo, hi)

    return f


def _method(name: str) -> Any:
    native = _NATIVE_METHODS[name]
    inplace = name.endswith("_")
    base = name.rstrip("_")

    def m(self: Any, *args: Any, **kwargs: Any) -> Any:
        kw = dict(kwargs)
        if base == "clamp_min":
            (lo,) = _split(args, kw, ("min",))
            hi = None
        elif base == "clamp_max":
            (hi,) = _split(args, kw, ("max",))
            lo = None
        else:
            lo, hi = _split(args, kw, ("min", "max"))
        if kw or not _needs_emulation(self, lo, hi):
            return native(self, *args, **kwargs)
        if not inplace:
            return _Clamp214.apply(self, lo, hi)
        # The clone keeps the input's pre-modification value for the backward.
        return self.copy_(_Clamp214.apply(self.clone(), lo, hi))

    return m


@contextlib.contextmanager
def torch214_clamp_subgradients() -> Iterator[None]:
    """Within the context, BoTorch, GPyTorch and linear_operator compute
    torch ≥ 2.14's clamp subgradients (a no-op under torch ≥ 2.14)."""
    if TORCH_GE_214:
        yield
        return
    for name in _NATIVE:
        setattr(torch, name, _function(name))
    for name in _NATIVE_METHODS:
        setattr(torch.Tensor, name, _method(name))
    try:
        yield
    finally:
        for name, f in _NATIVE.items():
            setattr(torch, name, f)
        for name, f in _NATIVE_METHODS.items():
            setattr(torch.Tensor, name, f)

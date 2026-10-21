"""First-order exact native MLL adjoints for persistent Torch GP leaves."""

from __future__ import annotations

from typing import Any

from botorch_rs._arrays import from_flat, kind_of, to_flat
from botorch_rs.exceptions.errors import UnsupportedError

_function: Any = None


def mll_output(model: Any, values: Any, shape: tuple[int, ...]) -> Any:
    global _function
    result = from_flat(values, shape, model._kind)
    sync = getattr(model, "_sync_torch_parameters", None)
    state = None if sync is None else sync()
    if state is None:
        return result
    import torch

    leaves = tuple(p for _, p in state.entries)
    if not torch.is_grad_enabled() or not any(p.requires_grad for p in leaves):
        return result
    if _function is None:
        from torch.autograd.function import once_differentiable

        class _NativeMLL(torch.autograd.Function):
            @staticmethod
            def forward(ctx: Any, native: Any, output: Any, *parameters: Any) -> Any:  # type: ignore[override]
                ctx.native, ctx.error = native
                ctx.layout = [(tuple(p.shape), p.numel(), kind_of(p)) for p in parameters]
                return output

            @staticmethod
            @once_differentiable
            def backward(ctx: Any, adjoint: Any) -> tuple[Any, ...]:  # type: ignore[override]
                if ctx.error is not None:
                    raise UnsupportedError(ctx.error)
                data, _ = ctx.native.mll_parameter_vjp(to_flat(adjoint))
                offset = 0
                gradients: list[Any] = []
                for shape_, size, kind in ctx.layout:
                    gradients.append(from_flat(data[offset : offset + size], shape_, kind))
                    offset += size
                return None, None, *gradients

        _function = _NativeMLL
    return _function.apply((model._rs.copy(), state.gradient_error), result, *leaves)

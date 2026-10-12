"""Autograd support for sample paths evaluated at torch inputs that require
gradients (imported only then).

A path value depends only on its own input point, so the extension returns
the values (``m x S x N``) with their derivatives with respect to the points
(``m x S x N x d``), and the backward pass contracts the adjoint of the values
with them.
"""

from __future__ import annotations

from typing import Any

import torch

from botorch_rs._arrays import ArrayKind, from_flat, to_flat


class _PathValues(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, X: torch.Tensor, handle: Any, component: str | None) -> torch.Tensor:  # type: ignore[override]
        kind = ArrayKind("torch", X.dtype, X.device)
        (values, vshape), (grads, gshape) = handle.forward_with_grad(to_flat(X, "X"), component)
        ctx.save_for_backward(from_flat(grads, tuple(gshape), kind))
        ctx.x_shape = tuple(X.shape)
        return from_flat(values, tuple(vshape), kind)

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> tuple[torch.Tensor, None, None]:  # type: ignore[override]
        (grad,) = ctx.saved_tensors
        grad_x = (grad_output.unsqueeze(-1) * grad).sum(dim=(0, 1))
        return grad_x.reshape(ctx.x_shape), None, None


def path_values(handle: Any, X: torch.Tensor, component: str | None) -> torch.Tensor:
    """The ``m x S x N`` values of every output of ``handle`` at the points of
    ``X``, connected to the autograd graph of ``X``."""
    return _PathValues.apply(X, handle, component)

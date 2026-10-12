"""Autograd support for torch inputs (imported only when torch tensors that
require gradients are passed to an acquisition function).

The value and its exact gradient are computed together in Rust; the gradient
is stored for the backward pass, so that ``acqf(X).sum().backward()`` and
``torch.autograd.grad`` work as with BoTorch.
"""

from __future__ import annotations

from typing import Any

import torch

from botorch_rs._arrays import ArrayKind, from_flat, to_flat


class _AcquisitionValue(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, X: torch.Tensor, acqf_rs: Any) -> torch.Tensor:  # type: ignore[override]
        kind = ArrayKind("torch", X.dtype, X.device)
        (values, vshape), (grads, _) = acqf_rs.forward_with_grad(to_flat(X))
        ctx.save_for_backward(from_flat(grads, tuple(X.shape), kind))
        return from_flat(values, tuple(vshape), kind)

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> tuple[torch.Tensor, None]:  # type: ignore[override]
        (grad,) = ctx.saved_tensors
        return grad_output[..., None, None] * grad, None


def requires_grad(X: Any) -> bool:
    return bool(X.requires_grad) and torch.is_grad_enabled()


def evaluate(acqf_rs: Any, X: torch.Tensor) -> torch.Tensor:
    """Acquisition values at ``X`` (``batch x q x d``) connected to the
    autograd graph of ``X``."""
    return _AcquisitionValue.apply(X, acqf_rs)

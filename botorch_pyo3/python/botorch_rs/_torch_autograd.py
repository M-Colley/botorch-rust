"""Autograd support for torch inputs (imported only when torch tensors that
require gradients are passed to an acquisition function).

The value and its exact gradient are computed together in Rust; the gradient
is stored for the backward pass, so that ``acqf(X).sum().backward()`` and
``torch.autograd.grad`` work as with BoTorch.
"""

from __future__ import annotations

from typing import Any

import torch


class _AcquisitionValue(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, X: torch.Tensor, acqf_rs: Any) -> torch.Tensor:  # type: ignore[override]
        flat = X.detach().reshape(-1).cpu().tolist()
        (values, vshape), (grads, _) = acqf_rs.forward_with_grad((flat, tuple(X.shape)))
        grad = torch.tensor(grads, dtype=X.dtype).reshape(X.shape).to(X.device)
        ctx.save_for_backward(grad)
        return torch.tensor(values, dtype=X.dtype).reshape(tuple(vshape)).to(X.device)

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

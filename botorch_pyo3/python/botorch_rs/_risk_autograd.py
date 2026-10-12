"""Autograd support for the MC objectives computed in Rust (imported only
when a risk measure is called on torch tensors that require gradients): the
values are computed in Rust and the backward pass is Rust's exact
vector-Jacobian product."""

from __future__ import annotations

from typing import Any

import torch

from botorch_rs._arrays import ArrayKind, from_pair, to_flat


class NativeObjective(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, samples: torch.Tensor, rs: Any) -> torch.Tensor:  # type: ignore[override]
        ctx.rs = rs
        ctx.save_for_backward(samples)
        kind = ArrayKind("torch", samples.dtype, samples.device)
        return from_pair(rs.forward(to_flat(samples.detach())), kind)

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> tuple[torch.Tensor, None]:  # type: ignore[override]
        (samples,) = ctx.saved_tensors
        kind = ArrayKind("torch", samples.dtype, samples.device)
        adjoint = to_flat(grad_output.detach().contiguous())
        grad = from_pair(ctx.rs.vjp(to_flat(samples.detach()), adjoint), kind)
        return grad, None


def evaluate(rs: Any, samples: torch.Tensor) -> torch.Tensor:
    """The values of ``rs`` at ``samples``, connected to the autograd graph."""
    return NativeObjective.apply(samples, rs)

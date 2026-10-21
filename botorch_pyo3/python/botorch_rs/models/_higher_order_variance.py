"""Native variance evaluation of caller-supplied Kronecker factors."""

from __future__ import annotations

from typing import Any

from botorch_rs import _botorch_rs as _rs
from botorch_rs._arrays import first_kind, from_flat, is_torch_tensor, to_flat
from botorch_rs._ops import getitem, shape_of
from botorch_rs.exceptions.errors import UnsupportedError


def posterior_variances(joint: Any, num_train: int, noise: Any) -> Any:
    factors = getattr(joint, "linear_ops", None)
    if factors is None:
        if not isinstance(joint, (list, tuple)):
            raise UnsupportedError(
                "Supply a Kronecker operator or a sequence of its matrix factors."
            )
        factors = joint
    arrays = []
    for factor in factors:
        if hasattr(factor, "evaluate_kernel"):
            factor = factor.evaluate_kernel()
        if hasattr(factor, "to_dense"):
            factor = factor.to_dense()
        arrays.append(factor)
    if not arrays:
        raise ValueError("At least a data covariance factor is required.")
    kind = first_kind(*arrays, noise)
    noise_shape = shape_of(noise)
    if noise_shape and noise_shape[-1] == 1:
        noise = getitem(noise, (..., 0))
    arguments = [*arrays, noise]
    if kind.lib == "torch":
        import torch

        if torch.is_grad_enabled() and any(
            is_torch_tensor(a) and a.requires_grad for a in arguments
        ):
            return _with_autograd(arrays, num_train, noise, kind)
    data, shape = _rs.higher_order_variances(
        [to_flat(a, "factor") for a in arrays], num_train, to_flat(noise, "noise")
    )
    return from_flat(data, tuple(shape), kind)


def _with_autograd(factors: list[Any], train: int, noise: Any, kind: Any) -> Any:
    import torch
    from torch.autograd.function import once_differentiable

    tensors = [a if is_torch_tensor(a) else from_flat(*to_flat(a), kind) for a in [*factors, noise]]

    class NativeVariance(torch.autograd.Function):
        @staticmethod
        def forward(ctx: Any, *args: Any) -> Any:
            ctx.save_for_backward(*[a.detach().clone() for a in args])
            data, shape = _rs.higher_order_variances(
                [to_flat(a) for a in args[:-1]], train, to_flat(args[-1])
            )
            return from_flat(data, tuple(shape), kind)

        @staticmethod
        @once_differentiable
        def backward(ctx: Any, gradient: Any) -> tuple[Any, ...]:
            args = ctx.saved_tensors
            factors_gradient, noise_gradient = _rs.higher_order_variances_vjp(
                [to_flat(a) for a in args[:-1]], train, to_flat(args[-1]), to_flat(gradient)
            )
            return tuple(
                from_flat(data, tuple(shape), first_kind(arg)) if required else None
                for (data, shape), arg, required in zip(
                    [*factors_gradient, noise_gradient], args, ctx.needs_input_grad, strict=True
                )
            )

    return NativeVariance.apply(*tensors)

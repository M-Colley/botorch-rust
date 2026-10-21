"""Lazy first-order torch adapter for native posterior query adjoints."""

from __future__ import annotations

import sys
from typing import Any

from botorch_rs import _botorch_rs as _rs
from botorch_rs._arrays import from_flat, kind_of, to_flat

_function: Any = None


def requires_query_grad(value: Any) -> bool:
    """Whether torch is tracking first-order query derivatives for ``value``."""
    torch = sys.modules.get("torch")
    return bool(
        torch is not None and torch.is_grad_enabled() and getattr(value, "requires_grad", False)
    )


def query_output(
    posterior: Any, mode: str, data: Any, shape: tuple[int, ...], base: Any = None
) -> Any:
    """Connect already-computed native values to the snapshotted query graph."""
    global _function
    import torch

    if _function is None:
        from torch.autograd.function import once_differentiable

        class _NativePosteriorQuery(torch.autograd.Function):
            @staticmethod
            def forward(
                ctx: Any,
                query: Any,
                post: Any,
                operation: str,
                values: Any,
                output_shape: Any,
                normals: Any,
            ) -> Any:  # type: ignore[override]
                ctx.save_for_backward(query)
                ctx.posterior, ctx.operation = post, operation
                ctx.normals = None if normals is None else (list(normals[0]), tuple(normals[1]))
                return from_flat(values, output_shape, post._kind)

            @staticmethod
            @once_differentiable
            def backward(ctx: Any, output_adjoint: Any) -> tuple[Any, ...]:  # type: ignore[override]
                (query,) = ctx.saved_tensors
                post = ctx.posterior
                kwargs = post._kwargs()
                if ctx.operation == "samples":
                    result = _rs.posterior_query_samples_vjp(
                        post._rs,
                        post._x,
                        ctx.normals,
                        to_flat(output_adjoint),
                        observation_noise=post._observation_noise,
                        **kwargs,
                    )
                else:
                    result = _rs.posterior_query_vjp(
                        post._rs,
                        post._x,
                        to_flat(output_adjoint),
                        ctx.operation,
                        observation_noise=post._observation_noise,
                        **kwargs,
                    )
                gradient = from_flat(result[0], tuple(result[1]), kind_of(query))
                return gradient, None, None, None, None, None

        _function = _NativePosteriorQuery
    return _function.apply(posterior._query_inputs, posterior, mode, data, shape, base)

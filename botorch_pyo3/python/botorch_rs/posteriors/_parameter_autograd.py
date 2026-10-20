"""First-order parameter adjoints attached to immutable native posterior laws."""

from __future__ import annotations

from typing import Any

from botorch_rs._arrays import from_flat, kind_of, to_flat
from botorch_rs.exceptions.errors import UnsupportedError

_function: Any = None


def parameter_output(
    posterior: Any, mode: str, result: Any, shape: tuple[int, ...], normals: Any = None
) -> Any:
    """Attach a parameter-only graph, independently of the query-input graph.

    Keeping the graphs separate lets query-only differentiation work even when
    a parameter operation is explicitly unsupported. Leaves are graph inputs,
    never saved tensor values: backward uses the posterior's native snapshot.
    """
    global _function
    import torch

    parameters = posterior._parameter_inputs
    if not torch.is_grad_enabled() or not any(p.requires_grad for p in parameters):
        return result
    if _function is None:
        from torch.autograd.function import once_differentiable

        class _NativePosteriorParameters(torch.autograd.Function):
            @staticmethod
            def forward(
                ctx: Any, post: Any, operation: str, output_shape: Any, base: Any, *leaves: Any
            ) -> Any:  # type: ignore[override]
                ctx.posterior, ctx.operation = post, operation
                ctx.layout = [(tuple(p.shape), p.numel(), kind_of(p)) for p in leaves]
                ctx.normals = None if base is None else (list(base[0]), tuple(base[1]))
                return torch.zeros((), dtype=post._kind.dtype, device=post._kind.device).expand(
                    output_shape
                )

            @staticmethod
            @once_differentiable
            def backward(ctx: Any, output_adjoint: Any) -> tuple[Any, ...]:  # type: ignore[override]
                post = ctx.posterior
                if post._parameter_gradient_error is not None:
                    raise UnsupportedError(post._parameter_gradient_error)
                if ctx.operation == "samples":
                    data, _ = post._rs.posterior_samples_parameter_vjp(
                        post._x,
                        ctx.normals,
                        to_flat(output_adjoint),
                        observation_noise=post._observation_noise,
                        **post._kwargs(),
                    )
                else:
                    data, _ = post._rs.posterior_moment_parameter_vjp(
                        post._x,
                        to_flat(output_adjoint),
                        ctx.operation,
                        observation_noise=post._observation_noise,
                        **post._kwargs(),
                    )
                gradients: list[Any] = []
                offset = 0
                for shape_, size, kind in ctx.layout:
                    gradients.append(from_flat(data[offset : offset + size], shape_, kind))
                    offset += size
                return None, None, None, None, *gradients

        _function = _NativePosteriorParameters
    return result + _function.apply(posterior, mode, shape, normals, *parameters)

"""Lazy first-order Torch adapter for native structured posterior adjoints."""

from __future__ import annotations

import math
import sys
from typing import Any

from botorch_rs._arrays import from_flat, kind_of, to_flat

_function: Any = None


def connected(posterior: Any) -> bool:
    torch = sys.modules.get("torch")
    return bool(
        torch is not None
        and torch.is_grad_enabled()
        and any(
            getattr(getattr(posterior, name, None), "requires_grad", False)
            for name in ("_query_inputs", "_query_times")
        )
    )


def output(posterior: Any, mode: str, values: Any, shape: Any, normals: Any = None) -> Any:
    if not connected(posterior):
        return from_flat(values, shape, posterior._kind)
    global _function
    import torch

    if _function is None:
        from torch.autograd.function import once_differentiable

        class _StructuredQuery(torch.autograd.Function):
            @staticmethod
            def forward(
                ctx: Any,
                x: Any,
                t: Any,
                post: Any,
                operation: str,
                data: Any,
                output_shape: Any,
                base: Any,
            ) -> Any:  # type: ignore[override]
                ctx.save_for_backward(x, t)
                ctx.posterior, ctx.operation = post, operation
                ctx.base = None if base is None else (list(base[0]), tuple(base[1]))
                return from_flat(data, output_shape, post._kind)

            @staticmethod
            @once_differentiable
            def backward(ctx: Any, adjoint: Any) -> tuple[Any, ...]:  # type: ignore[override]
                x, t = ctx.saved_tensors
                gx, gt = _vjp(ctx.posterior, ctx.operation, to_flat(adjoint), ctx.base)
                return (
                    None if x is None else from_flat(gx[0], gx[1], kind_of(x)),
                    None if t is None else from_flat(gt[0], gt[1], kind_of(t)),
                    None,
                    None,
                    None,
                    None,
                    None,
                )

        _function = _StructuredQuery
    return _function.apply(
        getattr(posterior, "_query_inputs", None),
        getattr(posterior, "_query_times", None),
        posterior,
        mode,
        values,
        tuple(shape),
        normals,
    )


def _normal_batch(normals: Any, batch: int, samples: int, total: int) -> Any:
    data, shape = normals
    nb = shape[1] if len(shape) == 3 else 1
    return (
        [
            data[(s * nb + (batch if nb > 1 else 0)) * shape[-1] + j]
            for s in range(samples)
            for j in range(total)
        ],
        (samples, total),
    )


def _vjp(p: Any, mode: str, adjoint: Any, normals: Any) -> tuple[Any, Any]:
    data, _ = adjoint
    batches = math.prod(p._batch)
    latent = hasattr(p, "_t")
    higher = hasattr(p, "_target_shape")
    width = p._q * (p._tasks if latent else p._S if higher else p._m)
    samples = normals[1][0] if mode == "samples" else 1
    xvalues = [0.0] * len(p._x[0])
    tvalues = [0.0] * len(p._t[0]) if latent else []
    matrices: Any = p._batches() if latent or higher else None
    indices: Any = None
    if latent:
        from botorch_rs.kernels import _indices

        indices = list(_indices(p._batch))
    for b in range(batches):
        if latent:
            xb, tb = matrices[b]
        elif higher:
            xb = matrices[b]
        else:
            size = p._q * p._x[1][-1]
            xb = (list(p._x[0][b * size : (b + 1) * size]), (p._q, p._x[1][-1]))
        if mode == "samples":
            weights = (
                [data[(s * batches + b) * width + j] for s in range(samples) for j in range(width)],
                (samples, width),
            )
            total = p.base_sample_shape[-1] - (p._q * p._S if higher and p._noise else 0)
            base = _normal_batch(normals, b, samples, total)
            if latent:
                gx, gt = p._rs.samples_vjp(xb, tb, base, weights)
            elif higher:
                gx, gt = p._rs.samples_vjp(xb, base, weights, p._original), None
            else:
                gx, gt = (
                    p._model_rs.matheron_samples_vjp(xb, base, weights, p._observation_noise),
                    None,
                )
        else:
            stride = width * width if mode in ("covariance", "covariance_diagonal") else width
            values = data[b * stride : (b + 1) * stride]
            mean, variance = [0.0] * width, [0.0] * width
            covariance = [0.0] * (width * width)
            if mode == "mean":
                mean = values
            elif mode == "variance":
                variance = values
            elif mode == "covariance_diagonal":
                variance = [values[i * width + i] for i in range(width)]
            else:
                covariance = values
            if latent:
                if mode == "variance":
                    raw = p._moments()[1]
                    for i in range(width):
                        if raw[b * width * width + i * width + i] > 1e-10:
                            covariance[i * width + i] = variance[i]
                gx, gt = p._rs.posterior_grid_vjp(
                    xb, tb, (mean, (width,)), (covariance, (width, width))
                )
            else:
                outputs = p._S if higher else p._m
                ma, va = (mean, (p._q, outputs)), (variance, (p._q, outputs))
                gx, gt = (
                    (
                        p._rs.moment_vjp(xb, ma, va, p._original)
                        if higher
                        else p._model_rs.marginal_vjp(xb, ma, va, p._observation_noise)
                    ),
                    None,
                )
        if latent:
            index = indices[b]
            for target, value, original in ((xvalues, gx, p._x), (tvalues, gt, p._t)):
                original_batch = original[1][:-2]
                bid = 0
                for dim, at in zip(
                    original_batch, index[len(index) - len(original_batch) :], strict=True
                ):
                    bid = bid * dim + (0 if dim == 1 else at)
                start = bid * math.prod(original[1][-2:])
                for i, v in enumerate(value[0]):
                    target[start + i] += v
        else:
            start = b * len(gx[0])
            for i, v in enumerate(gx[0]):
                xvalues[start + i] += v
    return (xvalues, tuple(p._x[1])), (tvalues, tuple(p._t[1]) if latent else ())

"""Native first-order variational ELBO adjoints for persistent Torch leaves."""

from __future__ import annotations

from typing import Any

from botorch_rs._arrays import from_flat, kind_of, to_flat

_function: Any = None


def elbo_output(
    model: Any,
    x: Any,
    y: Any,
    num_data: float,
    beta: float,
    terms: Any,
    combine: bool,
    native: Any = None,
    leaves: Any = None,
) -> Any:
    global _function
    values = (terms[0] - terms[1] + terms[2],) if combine else terms
    outputs = tuple(from_flat([value], (), model._kind) for value in values)
    if leaves is None:
        state = model._sync_torch_parameters()
        leaves = () if state is None else tuple(parameter for _, parameter in state.entries)
    if not leaves:
        return outputs[0] if combine else outputs
    import torch

    if not torch.is_grad_enabled() or not any(parameter.requires_grad for parameter in leaves):
        return outputs[0] if combine else outputs
    if _function is None:
        from torch.autograd.function import once_differentiable

        class _NativeELBO(torch.autograd.Function):
            @staticmethod
            def forward(ctx: Any, native: Any, outputs_: Any, *parameters: Any) -> Any:  # type: ignore[override]
                ctx.native = native
                ctx.layout = [(tuple(p.shape), p.numel(), kind_of(p)) for p in parameters]
                ctx.set_materialize_grads(False)
                return outputs_[0] if len(outputs_) == 1 else outputs_

            @staticmethod
            @once_differentiable
            def backward(ctx: Any, *adjoints: Any) -> tuple[Any, ...]:  # type: ignore[override]
                native, x_, y_, n_, beta_, combined = ctx.native
                weights = [
                    0.0 if adjoint is None else to_flat(adjoint)[0][0] for adjoint in adjoints
                ]
                if combined:
                    weights = [weights[0], -weights[0], weights[0]]
                data = native.elbo_terms_parameter_vjp(x_, y_, n_, beta_, weights)
                offset = 0
                gradients = []
                for shape, size, kind in ctx.layout:
                    gradients.append(from_flat(data[offset : offset + size], shape, kind))
                    offset += size
                return None, None, *gradients

        _function = _NativeELBO
    source = model._rs if native is None else native
    native = (source.copy(), x, y, num_data, beta, combine)
    return _function.apply(native, outputs, *leaves)

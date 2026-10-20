"""Optimization of the actual device-resident Tensor-backend parameters."""

from __future__ import annotations

import functools
from typing import Any


def parameter_bounds(module: Any) -> dict[str, tuple[float | None, float | None]]:
    model = getattr(module, "model", module)
    roots = list(getattr(model, "models", (model,)))
    by_identity = {}
    for root in roots:
        for _, child in root.named_modules():
            method = getattr(child, "raw_parameter_bounds", None)
            if method is None:
                continue
            parameters = dict(child.named_parameters())
            for name, limits in method().items():
                if name in parameters:
                    by_identity[id(parameters[name])] = limits
    return {
        name: by_identity[id(value)]
        for name, value in module.named_parameters()
        if id(value) in by_identity
    }


def fit_torch(
    mll: Any,
    parameters: Any,
    bounds: Any,
    closure: Any,
    closure_kwargs: Any,
    step_limit: Any,
    stopping_criterion: Any,
    optimizer: Any,
    scheduler: Any,
    callback: Any,
    timeout_sec: Any,
) -> Any:
    import torch

    from botorch_rs.optim.core import OptimizationResult, OptimizationStatus, torch_minimize

    parameters = (
        dict((name, value) for name, value in mll.named_parameters() if value.requires_grad)
        if parameters is None
        else dict(parameters)
    )
    if not parameters:
        return OptimizationResult(
            0,
            float(-mll().sum().detach()),
            OptimizationStatus.SUCCESS,
            message="No trainable parameters.",
        )
    if closure is None:

        def closure() -> Any:
            loss = -mll().sum()
            grads = torch.autograd.grad(loss, tuple(parameters.values()), allow_unused=True)
            for parameter, gradient in zip(parameters.values(), grads, strict=True):
                parameter.grad = torch.zeros_like(parameter) if gradient is None else gradient
            return loss, grads

    elif closure_kwargs:
        closure = functools.partial(closure, **closure_kwargs)
    limits = parameter_bounds(mll)
    if bounds is not None:
        limits.update(bounds)
    return torch_minimize(
        closure,
        parameters,
        bounds=limits,
        optimizer=optimizer,
        scheduler=scheduler,
        step_limit=step_limit,
        timeout_sec=timeout_sec,
        stopping_criterion=stopping_criterion,
        callback=callback,
    )

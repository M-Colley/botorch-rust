"""Optimization of the actual device-resident Tensor-backend parameters."""

from __future__ import annotations

import functools
from typing import Any


def parameter_bounds(module: Any) -> dict[str, tuple[Any, Any]]:
    model = getattr(module, "model", module)
    roots = list(getattr(model, "models", (model,)))
    likelihood = getattr(module, "likelihood", None)
    if likelihood is not None:
        roots.append(likelihood)
    by_identity = {}
    for root in roots:
        for _, child in root.named_modules():
            constraint_for_name = getattr(child, "constraint_for_parameter_name", None)
            if constraint_for_name is not None:
                for name, value in child.named_parameters(recurse=False):
                    constraint = constraint_for_name(name)
                    if constraint is not None and not constraint.enforced:
                        # Untransformed constraints bound the actual raw leaf.
                        # Keep elementwise bounds on their original device;
                        # transformed constraints instead act in constrained space.
                        by_identity[id(value)] = (
                            constraint.lower_bound,
                            constraint.upper_bound,
                        )
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
        loss = -mll().sum() if closure is None else closure(**(closure_kwargs or {}))[0]
        return OptimizationResult(
            0,
            float(loss.detach()),
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

    else:
        supplied_closure = (
            functools.partial(closure, **closure_kwargs) if closure_kwargs else closure
        )

        def closure() -> Any:
            loss, gradients = supplied_closure()
            for parameter, gradient in zip(parameters.values(), gradients, strict=True):
                parameter.grad = gradient
            return loss, gradients

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

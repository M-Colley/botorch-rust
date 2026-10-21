"""Lazy backend selection for native-compatible public Torch descriptors."""

from __future__ import annotations

import sys
from typing import Any


def native_port_descriptor_type(value: Any) -> type | None:
    """Identify an unchanged built-in descriptor beneath its Torch protocol."""
    descriptor = getattr(type(value), "_native_port_descriptor", None)
    if not isinstance(descriptor, type):
        return None
    if descriptor.__module__ not in (
        "botorch_rs.models.transforms.input",
        "botorch_rs.models.transforms.outcome",
        "botorch_rs.models.higher_order_gp",
    ) or not type(value).__module__.startswith("botorch_rs.models.transforms."):
        return None
    for name in (
        "forward",
        "transform",
        "untransform",
        "preprocess_transform",
        "untransform_posterior",
    ):
        original = getattr(descriptor, name, None)
        if original is not None and (
            name in value.__dict__ or getattr(type(value), name, None) is not original
        ):
            return None
    return descriptor


def requires_tensor_backend(value: Any) -> bool:
    """Keep native auto selection for unchanged CPU port descriptors only.

    Caller modules, overrides, live Tensor buffers and supplied Parameters
    require Tensor inference. Built-in default transform Parameters can remain
    eligible for native optimization; their actual objects are recorded by the
    descriptor factory, preserving identity through pickle and deepcopy.
    """
    torch = sys.modules.get("torch")
    if torch is None or not isinstance(value, torch.nn.Module):
        return False
    seen: set[int] = set()

    def native(module: Any) -> bool:
        if id(module) in seen:
            return True
        seen.add(id(module))
        container = type(module) in (torch.nn.ModuleDict, torch.nn.ModuleList)
        internal_state = type(
            module
        ).__module__ == "botorch_rs.models._torch_backend_modules" and type(module).__name__ in (
            "TorchConstraint",
            "TorchPrior",
        )
        if not (container or internal_state or native_port_descriptor_type(module) is not None):
            return False
        defaults = module.__dict__.get("_native_default_parameters", {})
        for name, parameter in module._parameters.items():
            if parameter is not None and (
                parameter is not defaults.get(name) or parameter.device.type != "cpu"
            ):
                return False
        for tensor in (*module._buffers.values(), *module.__dict__.values()):
            if isinstance(tensor, torch.Tensor) and (
                tensor.requires_grad or tensor.device.type != "cpu"
            ):
                return False
        return all(child is None or native(child) for child in module._modules.values())

    return not native(value)

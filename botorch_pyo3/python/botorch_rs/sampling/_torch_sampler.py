"""Lazy registration and serialization for the public sampler modules."""

from __future__ import annotations

from typing import Any, cast

import torch
from torch import nn

_TYPES: dict[type, type] = {}


def _setattr(self: Any, name: str, value: Any) -> None:
    if name in ("base_samples", "_base_samples"):
        if name == "base_samples":
            from botorch_rs.sampling.normal import SobolQMCNormalSampler

            if isinstance(self, SobolQMCNormalSampler):
                cast(Any, SobolQMCNormalSampler.base_samples).fset(self, value)
                return
        self.__dict__.pop("_native_base_samples", None)
        if value is None or isinstance(value, torch.Tensor):
            nn.Module.__setattr__(self, "base_samples", cast(Any, value))
        else:
            # Lists and arrays remain usable by the native CPU sampler. The
            # public Torch cache is a real buffer whenever it holds a Tensor.
            self._buffers["base_samples"] = None
            self.__dict__["_native_base_samples"] = value
        return
    if name == "samplers" and isinstance(value, (tuple, list)):
        value = nn.ModuleList(value)
    nn.Module.__setattr__(self, name, value)


def sampler_type(public_class: type) -> type:
    if public_class.__dict__.get("_is_tensor_sampler_type", False):
        return public_class
    if public_class not in _TYPES:
        bases = (
            (public_class,) if issubclass(public_class, nn.Module) else (public_class, nn.Module)
        )
        _TYPES[public_class] = type(
            f"Torch{public_class.__name__}",
            bases,
            {
                "__module__": __name__,
                "_sampler_public_type": public_class,
                "_is_tensor_sampler_type": True,
                "__call__": _caller_call(public_class),
            },
        )
    return _TYPES[public_class]


def _caller_call(public_class: type) -> Any:
    for parent in public_class.__mro__:
        if parent.__module__.startswith("botorch_rs.sampling."):
            break
        if "__call__" in parent.__dict__:
            return parent.__dict__["__call__"]
    return nn.Module.__call__


def allocate_sampler(public_class: type) -> Any:
    result: Any = object.__new__(sampler_type(public_class))
    nn.Module.__init__(result)
    # Sobol's native descriptor exposes a property of this name. Initializing
    # the canonical slot directly avoids replacing that public setter.
    if not any(
        base.__module__ == "botorch_rs.sampling.list_sampler" and base.__name__ == "ListSampler"
        for base in public_class.__mro__
    ):
        result._buffers["base_samples"] = None
    return result


def restore_sampler(public_class: type) -> Any:
    return allocate_sampler(public_class)

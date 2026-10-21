"""Torch module allocation for caller-defined public model classes."""

from __future__ import annotations

from typing import Any

from torch import nn

from botorch_rs.models.model import Model

_TYPES: dict[tuple[type, type], type] = {}


def _rebuild(public: type, tensor: type, state: Any) -> Any:
    model: Any = object.__new__(model_type(public, tensor))
    model.__setstate__(state)
    return model


def _reduce(self: Any, protocol: int) -> Any:
    return _rebuild, (
        self._tensor_public_model_class,
        self._tensor_concrete_model_class,
        self.__getstate__(),
    )


class _TorchCustomModel(nn.Module, Model):
    _tensor_backend = "torch"


def model_type(public: type, tensor: type) -> type:
    """Retain the caller's cooperative constructors and method overrides."""
    if public is tensor or issubclass(public, tensor):
        return public
    key = (public, tensor)
    if key not in _TYPES:
        attributes: dict[str, Any] = {
            "__module__": __name__,
            "_tensor_public_model_class": public,
            "_tensor_concrete_model_class": tensor,
        }
        if not any(
            name in base.__dict__
            for base in public.__mro__[: public.__mro__.index(Model)]
            for name in ("__reduce__", "__reduce_ex__")
        ):
            attributes["__reduce_ex__"] = _reduce
        _TYPES[key] = type(f"Torch{public.__name__}", (public, tensor), attributes)
    return _TYPES[key]


def custom_model(public: type) -> Any:
    model: Any = object.__new__(
        public if issubclass(public, nn.Module) else model_type(public, _TorchCustomModel)
    )
    # Preserve older port custom models whose constructor omitted super().
    # A source-style cooperative super().__init__ remains fully supported.
    nn.Module.__init__(model)
    model._tensor_backend = "torch"
    return model

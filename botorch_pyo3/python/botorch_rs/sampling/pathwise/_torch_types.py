"""Cooperative, picklable Torch implementations of caller-owned path classes."""

from __future__ import annotations

from typing import Any

_TYPES: dict[tuple[type, type], type] = {}


def _rebuild(public: type, tensor: type, base: type, state: Any) -> Any:
    value: Any = object.__new__(module_type(public, tensor, base))
    value.__setstate__(state)
    return value


def _reduce(self: Any, protocol: int) -> Any:
    return _rebuild, (
        self._tensor_public_type,
        self._tensor_implementation_type,
        self._tensor_base_type,
        self.__getstate__(),
    )


def module_type(public: type, tensor: type, base: type) -> type:
    if public is base:
        return tensor
    if issubclass(public, tensor):
        return public
    key = public, tensor
    if key not in _TYPES:
        attributes: dict[str, Any] = {
            "__module__": __name__,
            "_tensor_public_type": public,
            "_tensor_implementation_type": tensor,
            "_tensor_base_type": base,
        }
        if not any(
            name in ancestor.__dict__
            for ancestor in public.__mro__[: public.__mro__.index(base)]
            for name in ("__reduce__", "__reduce_ex__")
        ):
            attributes["__reduce_ex__"] = _reduce
        _TYPES[key] = type(f"Torch{public.__name__}", (public, tensor), attributes)
    return _TYPES[key]

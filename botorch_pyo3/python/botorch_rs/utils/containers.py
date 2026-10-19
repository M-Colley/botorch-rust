r"""Representations for different kinds of data (``botorch.utils.containers``).

The containers hold torch tensors, NumPy arrays or nested lists; ``shape``,
``device`` and ``dtype`` follow the array type (``device`` and ``dtype`` are
``None`` for nested lists).
"""

from __future__ import annotations

import dataclasses
import math
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass, fields
from typing import Any

from botorch_rs import _ops
from botorch_rs._arrays import from_flat, is_torch_tensor, kind_of, to_flat


def _equal(a: Any, b: Any) -> bool:
    if is_torch_tensor(a) and is_torch_tensor(b):
        return bool(a.equal(b))
    fa, fb = to_flat(a), to_flat(b)
    return tuple(fa[1]) == tuple(fb[1]) and list(fa[0]) == list(fb[0])


def _device(x: Any) -> Any:
    return x.device if is_torch_tensor(x) else None


def _dtype(x: Any) -> Any:
    if is_torch_tensor(x):
        return x.dtype
    np = sys.modules.get("numpy")
    if np is not None and isinstance(x, np.ndarray):
        return x.dtype
    return None


def _size(shape: Any) -> Any:
    torch = sys.modules.get("torch")
    return torch.Size(shape) if torch is not None else tuple(shape)


class BotorchContainer(ABC):
    r"""Abstract base class for BoTorch's data containers.

    A BotorchContainer represents a tensor, which should be the sole object
    returned by its ``__call__`` method. Said tensor is expected to consist of
    one or more "events" (e.g. data points or feature vectors), whose shape is
    given by the required ``event_shape`` field.
    """

    event_shape: Any

    def __post_init__(self, validate_init: bool = True) -> None:
        if validate_init:
            self._validate()

    @abstractmethod
    def __call__(self) -> Any:
        raise NotImplementedError

    @abstractmethod
    def __eq__(self, other: Any) -> bool:
        raise NotImplementedError

    @property
    @abstractmethod
    def shape(self) -> Any:
        raise NotImplementedError

    @property
    @abstractmethod
    def device(self) -> Any:
        raise NotImplementedError

    @property
    @abstractmethod
    def dtype(self) -> Any:
        raise NotImplementedError

    def _validate(self) -> None:
        for field in fields(self):  # type: ignore[arg-type]
            if field.name == "event_shape":
                return
        raise AttributeError("Missing required field `event_shape`.")


@dataclass(eq=False, slots=True, kw_only=True)
class DenseContainer(BotorchContainer):
    r"""Basic representation of data stored as a dense Tensor."""

    values: Any
    event_shape: Any

    def __call__(self) -> Any:
        """Returns a dense tensor representation of the container's contents."""
        return self.values

    def __eq__(self, other: Any) -> bool:
        return (
            type(other) is type(self)
            and tuple(self.shape) == tuple(other.shape)
            and _equal(self.values, other.values)
        )

    __hash__ = None  # type: ignore[assignment]

    @property
    def shape(self) -> Any:
        return _size(_ops.shape_of(self.values))

    @property
    def device(self) -> Any:
        return _device(self.values)

    @property
    def dtype(self) -> Any:
        return _dtype(self.values)

    def _validate(self) -> None:
        BotorchContainer._validate(self)
        vshape = _ops.shape_of(self.values)
        for a, b in zip(reversed(tuple(self.event_shape)), reversed(vshape), strict=False):
            if a != b:
                raise ValueError(
                    f"Shape of `values` {_size(vshape)} incompatible with "
                    f"`event shape` {_size(tuple(self.event_shape))}."
                )

    def clone(self) -> DenseContainer:
        return dataclasses.replace(self)


@dataclass(eq=False, slots=True, kw_only=True)
class SliceContainer(BotorchContainer):
    r"""Represent data points formed by concatenating (n-1)-dimensional slices
    taken from the leading dimension of an n-dimensional source tensor."""

    values: Any
    indices: Any
    event_shape: Any

    def __call__(self) -> Any:
        ishape = _ops.shape_of(self.indices)
        vshape = _ops.shape_of(self.values)
        idx = _ops.ints(self.indices)
        if is_torch_tensor(self.values):
            flat = self.values.index_select(dim=0, index=self.indices.reshape(-1))
            return flat.view(*ishape[:-1], -1, *vshape[2:])
        data, _ = to_flat(self.values, "values")
        row = math.prod(vshape[1:])
        out: list[float] = []
        for i in idx:
            out.extend(float(v) for v in data[i * row : (i + 1) * row])
        shape = (*ishape[:-1], ishape[-1] * vshape[1], *vshape[2:])
        return from_flat(out, shape, kind_of(self.values))

    def __eq__(self, other: Any) -> bool:
        return (
            type(other) is type(self)
            and _equal(self.values, other.values)
            and _equal(self.indices, other.indices)
        )

    __hash__ = None  # type: ignore[assignment]

    @property
    def shape(self) -> Any:
        return _size((*_ops.shape_of(self.indices)[:-1], *tuple(self.event_shape)))

    @property
    def device(self) -> Any:
        return _device(self.values)

    @property
    def dtype(self) -> Any:
        return _dtype(self.values)

    def _validate(self) -> None:
        BotorchContainer._validate(self)
        values = self.values
        vshape = _ops.shape_of(values)
        ishape = _ops.shape_of(self.indices)
        idx = _ops.ints(self.indices)
        assert len(ishape) > 1
        assert (-1 < min(idx)) & (max(idx) < vshape[0])

        event_shape = tuple(self.event_shape)
        _event_shape = (ishape[-1] * vshape[1],) + tuple(vshape[2:])
        if event_shape != _event_shape:
            raise ValueError(
                f"Shapes of `values` {_size(vshape)} and `indices` "
                f"{_size(ishape)} incompatible with `event_shape` {_size(event_shape)}."
            )

    def clone(self) -> SliceContainer:
        return type(self)(
            values=_ops.clone(self.values),
            indices=_ops.clone(self.indices) if is_torch_tensor(self.indices) else self.indices,
            event_shape=_size(tuple(self.event_shape)),
        )


__all__ = ["BotorchContainer", "DenseContainer", "SliceContainer"]

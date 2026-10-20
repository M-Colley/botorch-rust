"""BoTorch's priors (``botorch.models.utils.priors``): ``BetaPrior``."""

from __future__ import annotations

import math
import struct
import sys
from operator import index as integer_index
from typing import Any

from botorch_rs import _botorch_rs as _rs
from botorch_rs import _ops
from botorch_rs._arrays import as_float, broadcast_shapes, first_kind, from_flat, kind_of, to_flat
from botorch_rs.priors import Prior, _check_transform, _float32


class BetaPrior(Prior):
    """Beta prior with density ``x**(alpha - 1) (1 - x)**(beta - 1) / B(alpha, beta)``
    on ``[0, 1]``, parameterized by ``concentration1`` (``alpha``) and
    ``concentration0`` (``beta``); a prior on task correlations (e.g. of a
    ``PositiveIndexKernel``)."""

    def __init__(
        self,
        concentration1: Any,
        concentration0: Any,
        validate_args: bool = False,
        transform: Any = None,
    ) -> None:
        _check_transform(transform)
        a = as_float(concentration1, "concentration1")
        b = as_float(concentration0, "concentration0")
        self._kind = first_kind(concentration1, concentration0)
        torch = sys.modules.get("torch")
        if torch is not None and all(
            isinstance(value, torch.Tensor) for value in (concentration1, concentration0)
        ):
            self._kind = self._kind._replace(
                dtype=torch.promote_types(concentration1.dtype, concentration0.dtype)
            )
        elif self._kind.lib == "numpy":
            np = sys.modules["numpy"]
            self._kind = self._kind._replace(dtype=np.result_type(concentration1, concentration0))
        shape = broadcast_shapes(to_flat(concentration1)[1], to_flat(concentration0)[1])
        assert shape is not None
        self._batch_shape = shape
        self._validate_args = bool(validate_args)
        self._transform = transform
        single = _float32(concentration1, concentration0)
        self._rs = _rs.Prior.beta(a, b, single)
        self._a, self._b = (
            (
                float(struct.unpack("f", struct.pack("f", a))[0]),
                float(struct.unpack("f", struct.pack("f", b))[0]),
            )
            if single
            else (a, b)
        )

    @property
    def concentration1(self) -> Any:
        return self._parameter(self._a)

    @property
    def concentration0(self) -> Any:
        return self._parameter(self._b)

    @property
    def batch_shape(self) -> tuple[int, ...]:
        return self._batch_shape

    @property
    def event_shape(self) -> tuple[int, ...]:
        return ()

    def _parameter(self, value: float) -> Any:
        if not self.batch_shape and self._kind.lib == "list":
            return value
        return from_flat([value] * math.prod(self.batch_shape), self.batch_shape, self._kind)

    @property
    def mode(self) -> Any:
        return self._parameter(self._rs.mode)

    def expand(self, batch_shape: Any) -> BetaPrior:
        """Broadcast the prior to a batch shape without changing its density.

        Expanded concentration values remain identical, and every density is
        evaluated by the original native scalar prior. Expansion follows
        Tensor.expand's existing-dimension and singleton rules.
        """
        shape = tuple(integer_index(d) for d in batch_shape)
        if len(shape) < len(self.batch_shape):
            raise RuntimeError("expand: the number of sizes must be at least the prior batch rank")
        previous = (1,) * (len(shape) - len(self.batch_shape)) + self.batch_shape
        leading = len(shape) - len(self.batch_shape)
        if any(d < -1 or (d == -1 and i < leading) for i, d in enumerate(shape)):
            raise RuntimeError("expand: -1 is only allowed for an existing dimension")
        shape = tuple(old if new == -1 else new for old, new in zip(previous, shape, strict=True))
        if any(old != new and old != 1 for old, new in zip(previous, shape, strict=True)):
            raise RuntimeError("expand: target size must match each non-singleton dimension")
        expanded = type(self).__new__(type(self))
        expanded.__dict__ = self.__dict__.copy()
        expanded._batch_shape = shape
        return expanded

    def log_prob(self, x: Any) -> Any:
        data, shape = to_flat(x, "x")
        if self._validate_args and any(not 0.0 <= v <= 1.0 for v in data):
            raise ValueError("Expected value argument to be within the support Interval(0, 1).")
        target = broadcast_shapes(shape, self.batch_shape)
        if target is None:
            raise ValueError(
                f"Value shape {shape} is not broadcastable with batch_shape {self.batch_shape}."
            )
        kind = kind_of(x)
        kind = self._kind if kind.lib == "list" else kind
        if kind != kind_of(x):
            x = from_flat(data, shape, kind)
        return _ops.apply(
            "beta_log_prob", _ops.expand(x, target), params=[self._a, self._b], kind=kind
        )

    def __repr__(self) -> str:
        return (
            f"BetaPrior(concentration1={self.concentration1}, concentration0={self.concentration0})"
        )


__all__ = ["BetaPrior"]

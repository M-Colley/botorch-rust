"""Outcome transforms (``botorch.models.transforms.outcome``): ``Standardize``."""

from __future__ import annotations

import math
from typing import Any

from botorch_rs import _botorch_rs as _rs
from botorch_rs._arrays import LIST, ArrayKind, first_kind, from_flat, from_pair, to_flat
from botorch_rs.exceptions.errors import BotorchTensorDimensionError, UnsupportedError


class OutcomeTransform:
    """Base class of the outcome transforms."""

    training: bool = True

    def train(self, mode: bool = True) -> OutcomeTransform:
        self.training = mode
        return self

    def eval(self) -> OutcomeTransform:
        return self.train(False)


class Standardize(OutcomeTransform):
    r"""Standardizes outcomes to zero mean and unit variance (per output and
    batch element).

    Fitted in train mode (and by the model that uses it) with the column means
    and unbiased standard deviations; standard deviations below ``min_stdv``
    are replaced by 1.
    """

    def __init__(
        self,
        m: int,
        outputs: Any = None,
        batch_shape: Any = (),
        min_stdv: float = 1e-8,
    ) -> None:
        self._m = int(m)
        if outputs is not None:
            idx = sorted(int(o) % self._m for o in outputs)
            if idx != list(range(self._m)):
                raise UnsupportedError("Standardizing a subset of the outputs is not supported.")
        self._batch_shape = tuple(int(b) for b in (batch_shape or ()))
        self._rs = _rs.Standardize(min_stdv, list(self._batch_shape))
        self._kind: ArrayKind = LIST
        self.training = True

    def _set_fitted(self, fitted: _rs.Standardize, kind: ArrayKind) -> None:
        self._rs = fitted
        self._kind = kind

    @property
    def _is_trained(self) -> bool:
        return self._rs.is_trained

    @property
    def min_stdv(self) -> float:
        return self._rs.min_stdv

    @property
    def batch_shape(self) -> tuple[int, ...]:
        return self._batch_shape

    def _stats(self, pair: Any, fill: float) -> Any:
        if not self._rs.is_trained:
            shape = (*self._batch_shape, 1, self._m)
            return from_flat([fill] * math.prod(shape), shape, self._kind)
        return from_pair(pair, self._kind)

    @property
    def means(self) -> Any:
        """The output means, shape ``batch_shape x 1 x m`` (zeros until fitted)."""
        return self._stats(self._rs.means, 0.0)

    @property
    def stdvs(self) -> Any:
        """The output standard deviations, shape ``batch_shape x 1 x m``."""
        return self._stats(self._rs.stdvs, 1.0)

    @property
    def _stdvs_sq(self) -> Any:
        if not self._rs.is_trained:
            return self._stats(None, 1.0)
        data, shape = self._rs.stdvs
        return from_flat([s * s for s in data], tuple(shape), self._kind)

    def _check(self, shape: tuple[int, ...], train: bool) -> None:
        if len(shape) < 2:
            raise BotorchTensorDimensionError(
                f"Expected Y with shape `batch_shape x n x m`, got shape {shape}."
            )
        if train and tuple(shape[:-2]) != self._batch_shape:
            raise RuntimeError(
                f"Expected Y.shape[:-2] to be {self._batch_shape}, matching the "
                f"`batch_shape` argument to `Standardize`, but got "
                f"Y.shape[:-2]={tuple(shape[:-2])}."
            )
        if shape[-1] != self._m:
            raise RuntimeError(
                f"Wrong output dimension. Y.size(-1) is {shape[-1]}; expected {self._m}."
            )

    def forward(self, Y: Any, Yvar: Any = None, X: Any = None) -> tuple[Any, Any]:
        """Standardizes ``Y`` (``batch_shape x n x m``) and ``Yvar``; in train
        mode the statistics are first fitted to ``Y``."""
        kind = first_kind(Y)
        y = to_flat(Y, "Y")
        self._check(y[1], train=self.training)
        v = to_flat(Yvar, "Yvar") if Yvar is not None else None
        if self.training:
            fitted, y_tf, v_tf = self._rs.fit_transform(y, v)
            self._set_fitted(fitted, kind)
        else:
            y_tf, v_tf = self._rs.transform(y, v)
        return from_pair(y_tf, kind), (from_pair(v_tf, kind) if v_tf is not None else None)

    __call__ = forward

    def untransform(self, Y: Any, Yvar: Any = None, X: Any = None) -> tuple[Any, Any]:
        """Un-standardizes ``Y`` (``... x n x m``) and ``Yvar``."""
        kind = first_kind(Y)
        y = to_flat(Y, "Y")
        self._check(y[1], train=False)
        v = to_flat(Yvar, "Yvar") if Yvar is not None else None
        y_out, v_out = self._rs.untransform(y, v)
        return from_pair(y_out, kind), (from_pair(v_out, kind) if v_out is not None else None)

    def __repr__(self) -> str:
        return f"Standardize(m={self._m})"


__all__ = ["OutcomeTransform", "Standardize"]

"""Input transforms (``botorch.models.transforms.input``): ``Normalize``."""

from __future__ import annotations

import warnings
from typing import Any

from botorch_rs import _botorch_rs as _rs
from botorch_rs._arrays import ArrayKind, first_kind, from_flat, from_pair, to_flat
from botorch_rs.exceptions.errors import UnsupportedError
from botorch_rs.exceptions.warnings import UserInputWarning


class InputTransform:
    """Base class of the input transforms."""

    training: bool = True

    def train(self, mode: bool = True) -> InputTransform:
        self.training = mode
        return self

    def eval(self) -> InputTransform:
        return self.train(False)


class Normalize(InputTransform):
    r"""Normalizes the inputs to the unit cube, ``(X - offset) / coefficient``.

    With ``bounds=None`` (or ``learn_bounds=True``) the bounds are learned from
    the training inputs when the transform is used by a model (or called in
    train mode). Only the arguments of BoTorch's ``Normalize`` that affect all
    input dimensions with ``center=0.5`` are supported.
    """

    def __init__(
        self,
        d: int,
        indices: Any = None,
        bounds: Any = None,
        batch_shape: Any = None,
        transform_on_train: bool = True,
        transform_on_eval: bool = True,
        transform_on_fantasize: bool = True,
        reverse: bool = False,
        min_range: float = 1e-8,
        learn_bounds: bool | None = None,
        almost_zero: float = 1e-12,
        center: float = 0.5,
    ) -> None:
        if indices is not None:
            idx = [int(i) for i in to_flat(indices, "indices")[0]]
            if sorted(i % d for i in idx) != list(range(d)):
                raise UnsupportedError(
                    "Normalize over a subset of the inputs (`indices`) is not supported."
                )
        if batch_shape is not None and len(tuple(batch_shape)) > 0:
            raise UnsupportedError("Batched transforms (batch_shape) are not supported.")
        if not (transform_on_train and transform_on_eval and transform_on_fantasize):
            raise UnsupportedError("transform_on_train/eval/fantasize=False is not supported.")
        if reverse:
            raise UnsupportedError("reverse=True is not supported.")
        if center != 0.5:
            raise UnsupportedError("Only center=0.5 is supported.")
        learn = (bounds is None) if learn_bounds is None else bool(learn_bounds)
        if not learn and bounds is None:
            warnings.warn(
                "learn_bounds is False and no bounds were provided. The bounds "
                "will not be updated and the transform will be a no-op.",
                UserInputWarning,
                stacklevel=2,
            )
        self._kind: ArrayKind = first_kind(bounds)
        self._rs = _rs.Normalize(
            d, to_flat(bounds, "bounds") if bounds is not None else None, learn, min_range
        )
        self.almost_zero = almost_zero
        self.training = True

    # -- state -------------------------------------------------------------

    def _set_fitted(self, fitted: _rs.Normalize, kind: ArrayKind) -> None:
        self._rs = fitted
        if kind.lib != "list" or self._kind.lib == "list":
            self._kind = kind

    @property
    def learn_bounds(self) -> bool:
        return self._rs.learn_bounds

    @property
    def min_range(self) -> float:
        return self._rs.min_range

    @property
    def center(self) -> float:
        return 0.5

    @property
    def bounds(self) -> Any:
        """The ``2 x d`` normalization bounds."""
        return from_pair(self._rs.bounds, self._kind)

    @property
    def offset(self) -> Any:
        """The offsets (lower bounds), shape ``1 x d``."""
        values = self._rs.offset
        return from_flat(values, (1, len(values)), self._kind)

    @property
    def coefficient(self) -> Any:
        """The coefficients (ranges), shape ``1 x d``."""
        values = self._rs.coefficient
        return from_flat(values, (1, len(values)), self._kind)

    mins = offset
    ranges = coefficient

    # -- application -------------------------------------------------------

    def _apply(self, X: Any, inverse: bool) -> Any:
        kind = first_kind(X)
        flat = to_flat(X, "X")
        if not inverse and self.training and self.learn_bounds:
            self._set_fitted(self._rs.fit(flat), kind if kind.lib != "list" else self._kind)
        out = self._rs.untransform(flat) if inverse else self._rs.transform(flat)
        return from_pair(out, kind)

    def transform(self, X: Any) -> Any:
        """Normalizes ``X`` (``... x d``); in train mode, learnable bounds are
        first fitted to ``X``."""
        return self._apply(X, inverse=False)

    def forward(self, X: Any) -> Any:
        return self.transform(X)

    __call__ = forward

    def untransform(self, X: Any) -> Any:
        """Maps normalized inputs back to the original space."""
        return self._apply(X, inverse=True)

    def __repr__(self) -> str:
        return f"Normalize(d={self._rs.d}, learn_bounds={self.learn_bounds})"


__all__ = ["InputTransform", "Normalize"]

"""Outcome transforms (``botorch.models.transforms.outcome``): ``Standardize``."""

from __future__ import annotations

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
    r"""Standardizes outcomes to zero mean and unit variance (per output).

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
        if batch_shape is not None and len(tuple(batch_shape)) > 0:
            raise UnsupportedError("Batched transforms (batch_shape) are not supported.")
        self._rs = _rs.Standardize(min_stdv)
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
    def means(self) -> Any:
        """The output means, shape ``1 x m`` (empty until fitted)."""
        values = self._rs.means
        return from_flat(values, (1, len(values)), self._kind)

    @property
    def stdvs(self) -> Any:
        """The output standard deviations, shape ``1 x m``."""
        values = self._rs.stdvs
        return from_flat(values, (1, len(values)), self._kind)

    @property
    def _stdvs_sq(self) -> Any:
        values = [s * s for s in self._rs.stdvs]
        return from_flat(values, (1, len(values)), self._kind)

    def _check(self, shape: tuple[int, ...]) -> None:
        if len(shape) != 2:
            raise BotorchTensorDimensionError(
                f"Expected Y with shape `n x m`, got shape {shape} (batches are not supported)."
            )
        if shape[-1] != self._m:
            raise RuntimeError(
                f"Wrong output dimension. Y.size(-1) is {shape[-1]}; expected {self._m}."
            )

    def forward(self, Y: Any, Yvar: Any = None, X: Any = None) -> tuple[Any, Any]:
        """Standardizes ``Y`` (``n x m``) and ``Yvar``; in train mode the
        statistics are first fitted to ``Y``."""
        kind = first_kind(Y)
        y = to_flat(Y, "Y")
        self._check(y[1])
        v = to_flat(Yvar, "Yvar") if Yvar is not None else None
        if self.training:
            fitted, y_tf, v_tf = self._rs.fit_transform(y, v)
            self._set_fitted(fitted, kind)
        else:
            y_tf, v_tf = self._rs.transform(y, v)
        return from_pair(y_tf, kind), (from_pair(v_tf, kind) if v_tf is not None else None)

    __call__ = forward

    def untransform(self, Y: Any, Yvar: Any = None, X: Any = None) -> tuple[Any, Any]:
        """Un-standardizes ``Y`` (``n x m``) and ``Yvar``."""
        kind = first_kind(Y)
        y = to_flat(Y, "Y")
        self._check(y[1])
        v = to_flat(Yvar, "Yvar") if Yvar is not None else None
        y_out, v_out = self._rs.untransform(y, v)
        return from_pair(y_out, kind), (from_pair(v_out, kind) if v_out is not None else None)

    def __repr__(self) -> str:
        return f"Standardize(m={self._m})"


__all__ = ["OutcomeTransform", "Standardize"]

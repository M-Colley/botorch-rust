"""Base class of the GP module objects (kernels, likelihoods, means).

Like GPyTorch modules, these objects describe a component of a model. Once a
model is built with them they are *bound* to it: their hyperparameter
attributes (``lengthscale``, ``noise``, ...) then read and write the model's
current values, returned with the array type of the model's training data.
For a model with several outputs or a batch shape, every hyperparameter has
the model's ``_aug_batch_shape`` as leading dimensions (BoTorch's modules
with ``batch_shape = _aug_batch_shape``).
"""

from __future__ import annotations

import math
from typing import Any

from botorch_rs._arrays import LIST, from_flat
from botorch_rs.exceptions.errors import UnsupportedError


class GPModule:
    """A GPyTorch-style module whose parameters live in a bound model."""

    _model: Any = None
    #: The batch shape of the module (GPyTorch's ``batch_shape``).
    batch_shape: tuple[int, ...] = ()

    def _bind(self, model: Any) -> None:
        if self._model is not None and self._model is not model:
            raise UnsupportedError(
                f"This {type(self).__name__} is already part of another model; "
                "create a new one for each model."
            )
        self._model = model

    @property
    def _rs_model(self) -> Any:
        if self._model is None:
            raise AttributeError(
                f"{type(self).__name__} is not part of a model yet; its parameters are "
                "available once a model has been built with it."
            )
        return self._model._rs

    @property
    def _aug(self) -> tuple[int, ...]:
        """The ``_aug_batch_shape`` of the bound model."""
        return tuple(self._rs_model.aug_batch_shape)

    def _out(self, values: list[float], shape: tuple[int, ...]) -> Any:
        kind = self._model._kind if self._model is not None else LIST
        return from_flat(values, shape, kind)

    def _param_out(self, values: list[float], trailing: tuple[int, ...]) -> Any:
        """``values`` of all elements with shape ``_aug_batch_shape x trailing``
        (``trailing`` may contain one ``-1``, inferred from the number of values)."""
        aug = self._aug
        if -1 in trailing:
            known = math.prod(v for v in trailing if v != -1) * math.prod(aug)
            trailing = tuple(len(values) // max(known, 1) if v == -1 else v for v in trailing)
        return self._out(values, (*aug, *trailing))

    def train(self, mode: bool = True) -> GPModule:
        return self

    def eval(self) -> GPModule:
        return self

"""Base protocol shared by native Gaussian-process models.

The public name mirrors BoTorch's GPyTorchModel. Numerical inference is
implemented by the Rust-backed subclasses; this base does not wrap an
arbitrary Python GPyTorch ExactGP.
"""

from __future__ import annotations

import warnings
from typing import Any

from botorch_rs._arrays import to_flat
from botorch_rs._ops import dtype_name
from botorch_rs.exceptions.errors import BotorchTensorDimensionError, InputDataError
from botorch_rs.exceptions.warnings import BotorchTensorDimensionWarning, InputDataWarning
from botorch_rs.models.model import Model


class GPyTorchModel(Model):
    """Base protocol for native GP posterior, batch, and output interfaces."""

    @staticmethod
    def _validate_tensor_args(X: Any, Y: Any, Yvar: Any = None, strict: bool = True) -> None:
        xs, ys = to_flat(X, "X")[1], to_flat(Y, "Y")[1]
        if len(xs) != len(ys):
            if len(xs) - len(ys) == 1 and xs[:-1] == ys:
                message = (
                    "An explicit output dimension is required for targets."
                    f" Expected Y with dimension {len(xs)} (got Y.dim()={len(ys)})."
                )
            else:
                message = (
                    "Expected X and Y to have the same number of dimensions"
                    f" (got X with dimension {len(xs)} and Y with dimension {len(ys)})."
                )
            if strict:
                raise BotorchTensorDimensionError(message)
            warnings.warn(
                "Non-strict enforcement of botorch tensor conventions. The following "
                "error would have been raised with strict enforcement: " + message,
                BotorchTensorDimensionWarning,
                stacklevel=2,
            )
        if Yvar is not None:
            vs = to_flat(Yvar, "Yvar")[1]
            if ys[-len(vs) :] != vs:
                raise BotorchTensorDimensionError(
                    "An explicit output dimension is required for observation noise."
                    f" Expected Yvar with shape: {ys[-len(vs) :]} (got {vs})."
                )
        xd, yd = dtype_name(X) or "float64", dtype_name(Y) or "float64"
        vd = (dtype_name(Yvar) or "float64") if Yvar is not None else None
        if xd != yd or (vd is not None and yd != vd):
            raise InputDataError(
                f"Expected all inputs to share the same dtype. Got {xd} for X, "
                f"{yd} for Y, and {vd} for Yvar."
            )
        if xd != "float64":
            warnings.warn(
                f"The model inputs are of type {xd}. It is strongly recommended to use "
                "double precision in BoTorch, as this improves both precision and stability "
                "and can help avoid numerical errors.",
                InputDataWarning,
                stacklevel=3,
            )

    @property
    def batch_shape(self) -> tuple[int, ...]:
        return tuple(to_flat(self.train_inputs[0], "train_inputs")[1][:-2])  # type: ignore[attr-defined]

    @property
    def num_outputs(self) -> int:
        return int(self._num_outputs)  # type: ignore[attr-defined]


__all__ = ["GPyTorchModel"]

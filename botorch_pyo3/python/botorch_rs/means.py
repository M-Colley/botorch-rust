"""Mean modules (``gpytorch.means``): ``ConstantMean``."""

from __future__ import annotations

from typing import Any

from botorch_rs._arrays import to_flat
from botorch_rs._module import GPModule
from botorch_rs.exceptions.errors import UnsupportedError
from botorch_rs.kernels import _check_batch_shape, _check_kwargs


class Mean(GPModule):
    """Base class of the mean modules."""


class ConstantMean(Mean):
    """A constant mean (in the model's transformed outcome space)."""

    def __init__(
        self,
        constant_prior: Any = None,
        constant_constraint: Any = None,
        batch_shape: Any = None,
        **kwargs: Any,
    ) -> None:
        if constant_prior is not None or constant_constraint is not None:
            raise UnsupportedError(
                "Priors and constraints on the constant mean are not supported by botorch_rs."
            )
        _check_batch_shape(batch_shape)
        _check_kwargs(kwargs, "ConstantMean")

    @property
    def constant(self) -> Any:
        """The constant, shape ``()``."""
        return self._out([self._rs_model.constant], ())

    @constant.setter
    def constant(self, value: Any) -> None:
        data = to_flat(value, "constant")[0]
        if len(data) != 1:
            raise ValueError("The constant mean is a single value.")
        self._rs_model.set_constant(data[0])

    @property
    def raw_constant(self) -> Any:
        return self.constant

    @raw_constant.setter
    def raw_constant(self, value: Any) -> None:
        self.constant = value

    def __repr__(self) -> str:
        return "ConstantMean()"


__all__ = ["ConstantMean", "Mean"]

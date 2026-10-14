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
        batch_shape: Any = (),
        **kwargs: Any,
    ) -> None:
        if constant_prior is not None or constant_constraint is not None:
            raise UnsupportedError(
                "Priors and constraints on the constant mean are not supported by botorch_rs."
            )
        self.batch_shape = _check_batch_shape(batch_shape)
        _check_kwargs(kwargs, "ConstantMean")

    @property
    def constant(self) -> Any:
        """The constant, shape ``batch_shape``."""
        return self._param_out(self._rs_model.constant, ())

    @constant.setter
    def constant(self, value: Any) -> None:
        self._rs_model.set_constant(to_flat(value, "constant"))

    @property
    def raw_constant(self) -> Any:
        return self.constant

    @raw_constant.setter
    def raw_constant(self, value: Any) -> None:
        self.constant = value

    def __repr__(self) -> str:
        return "ConstantMean()"


class _TaskConstantMean(Mean):
    """The ``ConstantMean`` of one task of a ``MultitaskMean``."""

    def __init__(self, model: Any, task: int) -> None:
        self._task = task
        self._bind(model)

    @property
    def _name(self) -> str:
        return f"mean_module.base_means.{self._task}.raw_constant"

    @property
    def constant(self) -> Any:
        """The constant of the task (a scalar)."""
        return self._out(self._model._raw_param(self._name), ())

    @constant.setter
    def constant(self, value: Any) -> None:
        self._model._set_raw_param(self._name, value)

    @property
    def raw_constant(self) -> Any:
        return self.constant

    @raw_constant.setter
    def raw_constant(self, value: Any) -> None:
        self.constant = value

    def __repr__(self) -> str:
        return "ConstantMean()"


class MultitaskMean(Mean):
    """One ``ConstantMean`` per task (``gpytorch.means.MultitaskMean``), the
    mean of an ``LCEMGP``: ``base_means[t].constant`` (transformed outcome
    space)."""

    num_tasks: int
    base_means: list[_TaskConstantMean]

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        raise UnsupportedError(
            "MultitaskMean is created by LCEMGP; custom mean modules are not supported."
        )

    @classmethod
    def _create(cls, model: Any, num_tasks: int) -> MultitaskMean:
        obj = cls.__new__(cls)
        obj.num_tasks = num_tasks
        obj.base_means = [_TaskConstantMean(model, t) for t in range(num_tasks)]
        obj._bind(model)
        return obj

    def __repr__(self) -> str:
        return f"MultitaskMean(num_tasks={self.num_tasks})"


__all__ = ["ConstantMean", "Mean", "MultitaskMean"]

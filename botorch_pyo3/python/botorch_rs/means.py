"""Mean modules (``gpytorch.means``): ``ConstantMean``, ``ZeroMean``."""

from __future__ import annotations

import math
from typing import Any

from botorch_rs._arrays import LIST, broadcast_to, from_flat, kind_of, to_flat
from botorch_rs._module import GPModule
from botorch_rs.exceptions.errors import UnsupportedError
from botorch_rs.kernels import _check_batch_shape, _check_kwargs


class Mean(GPModule):
    """Base class of the mean modules."""


class ConstantMean(Mean):
    """A constant mean (in the model's transformed outcome space).

    ``constant_prior`` (any univariate prior) and ``constant_constraint`` (a
    constraint that is not enforced through a transform, e.g.
    ``Interval(-10, 10, transform=None)`` of BoTorch's
    ``get_mean_module_with_normal_prior``: its bounds are imposed by the
    optimizer) are supported.
    """

    def __init__(
        self,
        constant_prior: Any = None,
        constant_constraint: Any = None,
        batch_shape: Any = (),
        **kwargs: Any,
    ) -> None:
        from botorch_rs.constraints import Interval
        from botorch_rs.priors import Prior

        if constant_prior is not None and not isinstance(constant_prior, Prior):
            raise TypeError(
                f"Expected gpytorch.priors.Prior but got {type(constant_prior).__name__}"
            )
        if constant_constraint is not None:
            if not isinstance(constant_constraint, Interval):
                raise UnsupportedError(
                    "`constant_constraint` must be a botorch_rs.constraints constraint."
                )
            if constant_constraint.enforced:
                raise UnsupportedError(
                    "botorch_rs supports constraints on the constant mean that are not "
                    "enforced through a transform (e.g. Interval(lower, upper, transform=None))."
                )
        self.batch_shape = _check_batch_shape(batch_shape)
        _check_kwargs(kwargs, "ConstantMean")
        self.mean_prior = constant_prior
        self.raw_constant_constraint = constant_constraint
        initial = None if constant_constraint is None else constant_constraint.initial_value
        self._local_raw = [0.0 if initial is None else initial] * math.prod(self.batch_shape)
        self._local_kind = LIST
        self._has_local_constant = initial is not None

    def _bind(self, model: Any) -> None:
        super()._bind(model)
        # Only transfer explicitly initialized values. Rebinding a copied model
        # must preserve the parameters already stored in its native snapshot.
        if self._has_local_constant:
            self.constant = from_flat(self._local_raw, self.batch_shape, self._local_kind)
            self._has_local_constant = False

    @property
    def constant(self) -> Any:
        """The constant, shape ``batch_shape``."""
        if self._model is None:
            return from_flat(self._local_raw, self.batch_shape, self._local_kind)
        getter = getattr(self._model, "_raw_param_array", None)
        if getter is not None:
            return getter("mean_module.raw_constant")
        return self._param_out(self._rs_model.constant, ())

    @constant.setter
    def constant(self, value: Any) -> None:
        data, shape = to_flat(value, "constant")
        if any(not math.isfinite(v) for v in data):
            raise ValueError("constant must contain finite values.")
        if self._model is None:
            self._local_raw = broadcast_to(data, shape, self.batch_shape)
            self._local_kind = kind_of(value)
            self._has_local_constant = True
        else:
            self._rs_model.set_constant((data, shape))

    def initialize(self, **kwargs: Any) -> ConstantMean:
        for name, value in kwargs.items():
            if name not in ("constant", "raw_constant"):
                raise AttributeError(f"ConstantMean has no parameter {name!r}.")
            setattr(self, name, value)
        return self

    @property
    def raw_constant(self) -> Any:
        return self.constant

    @raw_constant.setter
    def raw_constant(self, value: Any) -> None:
        self.constant = value

    def __repr__(self) -> str:
        return "ConstantMean()"


class ZeroMean(Mean):
    """A zero mean (``gpytorch.means.ZeroMean``; no parameters). Supported by
    ``SingleTaskVariationalGP``."""

    def __init__(self, batch_shape: Any = (), **kwargs: Any) -> None:
        self.batch_shape = _check_batch_shape(batch_shape)
        _check_kwargs(kwargs, "ZeroMean")

    def __repr__(self) -> str:
        return "ZeroMean()"


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


__all__ = ["ConstantMean", "Mean", "MultitaskMean", "ZeroMean"]

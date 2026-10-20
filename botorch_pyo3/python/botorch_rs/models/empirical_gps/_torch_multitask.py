"""Exact task-indexed empirical covariance over paired historical curves."""

from __future__ import annotations

from operator import index as integer_index
from typing import Any, cast

import torch
from torch import nn

from botorch_rs.exceptions.errors import UnsupportedError
from botorch_rs.likelihoods import GaussianLikelihood
from botorch_rs.models._torch_backend import TorchSingleTaskGP
from botorch_rs.models._torch_backend_modules import _TorchModule, module_log_prior
from botorch_rs.models._torch_backend_multitask import TorchMultiTaskGP
from botorch_rs.models.empirical_gps._torch_backend import (
    TensorBaseAugmentedKernel,
    TensorLinearInterpolation,
    validate_history,
)
from botorch_rs.models.empirical_gps.multitask_empirical_1d_gp import (
    MultiTaskEmpiricalOneDimensionalGP,
    MultiTaskEmpiricalOneDimensionalKernel,
    MultiTaskEmpiricalOneDimensionalMean,
)


def validate_task_history(Xs: Any, Ys: Any) -> int:
    if (
        not isinstance(Xs, (list, tuple))
        or not isinstance(Ys, (list, tuple))
        or not Xs
        or len(Xs) != len(Ys)
    ):
        raise ValueError("Historical inputs/curves must be equally sized nonempty lists.")
    curves = None
    for X, Y in zip(Xs, Ys, strict=True):
        if not isinstance(Y, torch.Tensor) or Y.ndim != 2:
            raise ValueError("Historical task curves must have shape curves x progression.")
        validate_history(X, Y.unsqueeze(-1))
        if curves is not None and Y.shape[0] != curves:
            raise ValueError("Historical tasks must have paired curve counts.")
        if X.dtype != Xs[0].dtype or X.device != Xs[0].device:
            raise ValueError("Historical tasks must share dtype and device.")
        curves = Y.shape[0]
    return int(Ys[0].shape[0])


def task_inputs(X: Any, feature: int, count: int) -> Any:
    if not isinstance(X, torch.Tensor) or X.ndim < 2 or X.shape[-1] != 2:
        raise ValueError("Empirical task inputs must have two feature columns.")
    if not bool(torch.isfinite(X[..., 1 - feature]).all()):
        raise ValueError("Progression inputs must be finite.")
    values = X[..., feature]
    rounded = values.round()
    if not bool(
        (
            torch.isfinite(values)
            & ((values - rounded).abs() <= 1e-6)
            & (rounded >= 0)
            & (rounded < count)
        ).all()
    ):
        raise ValueError("Task labels must be integers in the historical task range.")
    return X[..., 1 - feature], rounded.long()


class _TaskHistory(nn.Module):
    X: Any
    Y: Any

    def __init__(self, X: Any, Y: Any) -> None:
        super().__init__()
        self.register_buffer("X", X)
        self.register_buffer("Y", Y)

    def interpolant(self, centered: bool) -> Any:
        mean = self.Y.mean(0)
        table = self.Y - mean if centered else mean
        return TensorLinearInterpolation(self.X.squeeze(-1), table)


class TensorTaskEmpiricalMean(_TorchModule, MultiTaskEmpiricalOneDimensionalMean):
    _histories: Any
    _interpolants_override: Any

    def __init__(
        self, historical_Xs: Any, historical_Ys: Any, task_feature: int, *, backend: str = "torch"
    ) -> None:
        validate_task_history(historical_Xs, historical_Ys)
        if task_feature not in (-2, -1, 0, 1):
            raise ValueError("Invalid task_feature.")
        _TorchModule.__init__(self, historical_Xs[0])
        self.task_feature, self.num_tasks = task_feature % 2, len(historical_Xs)
        self._histories = nn.ModuleList(
            [_TaskHistory(X, Y) for X, Y in zip(historical_Xs, historical_Ys, strict=True)]
        )

    @property
    def interpolants(self) -> Any:
        return (
            self._interpolants_override
            if hasattr(self, "_interpolants_override")
            else nn.ModuleList([history.interpolant(False) for history in self._histories])
        )

    @interpolants.setter
    def interpolants(self, value: Any) -> None:
        self._interpolants_override = value

    def forward(self, X: Any) -> Any:
        progression, tasks = task_inputs(X, self.task_feature, self.num_tasks)
        result = torch.zeros_like(progression)
        for task, interpolant in enumerate(self.interpolants):
            mask = tasks == task
            if bool(mask.any()):
                result[mask] = interpolant(progression[mask])
        return result

    __call__ = nn.Module.__call__


class TensorTaskEmpiricalKernel(_TorchModule, MultiTaskEmpiricalOneDimensionalKernel):
    _histories: Any
    _interpolants_override: Any

    def __init__(
        self,
        historical_Xs: Any,
        historical_Ys: Any,
        task_feature: int,
        num_curves: int,
        correction: int = 0,
        *,
        backend: str = "torch",
    ) -> None:
        count = validate_task_history(historical_Xs, historical_Ys)
        if (
            num_curves != count
            or isinstance(correction, bool)
            or not isinstance(correction, int)
            or correction >= count
        ):
            raise ValueError("num_curves must match histories and correction must be smaller.")
        if task_feature not in (-2, -1, 0, 1):
            raise ValueError("Invalid task_feature.")
        _TorchModule.__init__(self, historical_Xs[0])
        self.task_feature, self.num_tasks = task_feature % 2, len(historical_Xs)
        self.num_outputs = self.num_tasks
        self.num_curves, self.correction = count, correction
        self.ard, self._use_svd = False, False
        self._histories = nn.ModuleList(
            [_TaskHistory(X, Y) for X, Y in zip(historical_Xs, historical_Ys, strict=True)]
        )

    @property
    def interpolants(self) -> Any:
        return (
            self._interpolants_override
            if hasattr(self, "_interpolants_override")
            else nn.ModuleList([history.interpolant(True) for history in self._histories])
        )

    @interpolants.setter
    def interpolants(self, value: Any) -> None:
        self._interpolants_override = value

    def _basis_tensor(self, X: Any) -> Any:
        progression, tasks = task_inputs(X, self.task_feature, self.num_tasks)
        values = progression.new_zeros(*progression.shape, self.num_curves)
        for task, interpolant in enumerate(self.interpolants):
            mask = tasks == task
            if bool(mask.any()):
                values[mask] = interpolant(progression[mask]).transpose(-1, -2)
        return values.transpose(-1, -2)

    def forward(
        self,
        x1: Any,
        x2: Any = None,
        diag: bool = False,
        last_dim_is_batch: bool = False,
        **params: Any,
    ) -> Any:
        if params or last_dim_is_batch:
            raise UnsupportedError(
                "Empirical task kernels do not support feature batching or extra arguments."
            )
        U = self._basis_tensor(x1)
        V = U if x2 is None or x2 is x1 else self._basis_tensor(x2)
        value = (U * V).sum(-2) if diag else U.transpose(-1, -2) @ V
        return value / (self.num_curves - self.correction)

    def initialize(self, **kwargs: Any) -> Any:
        return _TorchModule.initialize(self, **kwargs)

    __call__ = nn.Module.__call__


def convert_task_module(source: Any, like: Any) -> Any:
    if isinstance(source, nn.Module):
        return source
    if not isinstance(
        source, (MultiTaskEmpiricalOneDimensionalMean, MultiTaskEmpiricalOneDimensionalKernel)
    ):
        raise UnsupportedError("Expected empirical task modules or actual Torch modules.")
    kernel = isinstance(source, MultiTaskEmpiricalOneDimensionalKernel)
    cls = TensorTaskEmpiricalKernel if kernel else TensorTaskEmpiricalMean
    result: Any = object.__new__(cls)
    _TorchModule.__init__(result, like)
    result.task_feature, result.num_tasks = source.task_feature, source.num_tasks
    result._interpolants_override = nn.ModuleList(
        [
            TensorLinearInterpolation(
                torch.as_tensor(f._x, dtype=like.dtype, device=like.device),
                torch.as_tensor(f._y, dtype=like.dtype, device=like.device),
            )
            for f in source.interpolants
        ]
    )
    if isinstance(source, MultiTaskEmpiricalOneDimensionalKernel):
        result.num_outputs, result.num_curves = source.num_tasks, source.num_curves
        result.correction = source.correction
        result.ard, result._use_svd = False, False
    return result


class TensorTaskEmpiricalGP(TorchMultiTaskGP, MultiTaskEmpiricalOneDimensionalGP):
    _supports_batched_models = True

    def _training_values(self, clamp_noise: bool = True) -> Any:
        return TorchSingleTaskGP._training_values(cast(Any, self), clamp_noise=clamp_noise)

    def __init__(
        self,
        train_X: Any,
        train_Y: Any,
        task_feature: int,
        historical_Xs: Any = None,
        historical_Ys: Any = None,
        train_Yvar: Any = None,
        output_tasks: Any = None,
        correction: int = 0,
        mean_module: Any = None,
        covar_module: Any = None,
        base_covar_module: Any = None,
        *,
        backend: str = "torch",
    ) -> None:
        if task_feature not in (-2, -1, 0, 1):
            raise ValueError("Invalid task_feature.")
        if (mean_module is None) != (covar_module is None):
            raise ValueError("mean_module and covar_module must be supplied together.")
        self._task_feature = task_feature % 2
        if mean_module is None:
            self.num_curves = validate_task_history(historical_Xs, historical_Ys)
            if historical_Xs[0].dtype != train_X.dtype or historical_Xs[0].device != train_X.device:
                raise ValueError("Historical and training data must share dtype and device.")
            mean_module = TensorTaskEmpiricalMean(historical_Xs, historical_Ys, task_feature)
            covar_module = TensorTaskEmpiricalKernel(
                historical_Xs, historical_Ys, task_feature, self.num_curves, correction
            )
        elif correction != 0:
            raise ValueError("Set correction when constructing the supplied covariance module.")
        mean_module = convert_task_module(mean_module, train_X)
        covar_module = convert_task_module(covar_module, train_X)
        self.num_tasks, self.num_curves = covar_module.num_tasks, covar_module.num_curves
        for module in (mean_module, covar_module):
            if (
                getattr(module, "num_tasks", self.num_tasks) != self.num_tasks
                or getattr(module, "task_feature", self._task_feature) % 2 != self._task_feature
            ):
                raise ValueError("Mean, kernel and task_feature metadata must match.")
        output_tasks = list(range(self.num_tasks)) if output_tasks is None else list(output_tasks)
        try:
            outputs = [integer_index(task) for task in output_tasks]
        except TypeError:
            raise ValueError("Output tasks must be integer historical task labels.") from None
        if not outputs or any(task < 0 or task >= self.num_tasks for task in outputs):
            raise ValueError("Output tasks must be valid historical task labels.")
        self._num_outputs, self._output_tasks = len(outputs), outputs
        self.num_non_task_features = 1
        self._validate_task_values = True
        task_inputs(train_X, self._task_feature, self.num_tasks)
        if train_Y.shape[-1] != 1:
            raise ValueError("Long-format targets must have one output column.")
        if base_covar_module is not None:
            covar_module = TensorBaseAugmentedKernel(covar_module, base_covar_module)
        TorchSingleTaskGP.__init__(
            cast(Any, self),
            train_X,
            train_Y,
            train_Yvar,
            None if train_Yvar is not None else GaussianLikelihood(),
            covar_module,
            mean_module,
            outcome_transform=None,
            input_transform=None,
            backend=backend,
        )
        self.register_buffer("_task_labels", train_X.new_tensor(range(self.num_tasks)))
        self.register_buffer("_task_mapper", None)
        self.register_buffer(
            "_base_idxr",
            torch.tensor([1 - self._task_feature], device=train_X.device, dtype=torch.long),
        )
        labels = self._map_tasks(train_X[..., self._task_feature]).long()
        observed = (
            labels.reshape(-1, 1) == torch.arange(self.num_tasks, device=train_X.device)
        ).any(0)
        self.register_buffer("_observed_task_indices", observed.nonzero().squeeze(-1))
        self.register_buffer("_unobserved_task_indices", (~observed).nonzero().squeeze(-1))

    @property
    def _aug_batch_shape(self) -> Any:
        return self.batch_shape

    @_aug_batch_shape.setter
    def _aug_batch_shape(self, value: Any) -> None:
        if torch.Size(value) != self.batch_shape:
            raise UnsupportedError("Empirical task batches derive from training data.")

    def _map_tasks(self, values: Any) -> Any:
        rounded = values.round()
        if not bool(
            (
                torch.isfinite(values)
                & ((values - rounded).abs() <= 1e-6)
                & (rounded >= 0)
                & (rounded < self.num_tasks)
            ).all()
        ):
            raise ValueError("Task labels must be integers in the historical task range.")
        return rounded

    def _split_inputs(self, X: Any) -> Any:
        return (
            X[..., : self._task_feature],
            self._map_tasks(X[..., self._task_feature]).long().unsqueeze(-1),
            X[..., self._task_feature + 1 :],
        )

    def _query_inputs(self, X: Any, output_indices: Any) -> Any:
        if X.shape[-1] == 2:
            task_inputs(X, self._task_feature, self.num_tasks)
        return TorchMultiTaskGP._query_inputs(self, X, output_indices)

    def eval(self) -> Any:
        return nn.Module.train(self, False)

    def _log_prior(self) -> Any:
        return sum(
            module_log_prior(module, self.batch_shape)
            for module in (self.likelihood, self.mean_module, self.covar_module)
        )

    def set_train_data(self, inputs: Any = None, targets: Any = None, strict: bool = True) -> None:
        X = self._train_X if inputs is None else inputs[0] if isinstance(inputs, tuple) else inputs
        Y = self._train_Y if targets is None else targets
        if Y.ndim == X.ndim - 1:
            Y = Y.unsqueeze(-1)
        self._validate_data(X, Y, self._train_Yvar)
        if Y.shape[-1] != 1 or X.shape[-1] != 2:
            raise ValueError("Empirical task data must remain in long format.")
        task_inputs(X, self._task_feature, self.num_tasks)
        if strict and (X.shape != self._train_X.shape or Y.shape != self._train_Y.shape):
            raise RuntimeError("Cannot modify training data shape with strict=True.")
        labels = self._map_tasks(X[..., self._task_feature]).long()
        observed = (labels.reshape(-1, 1) == torch.arange(self.num_tasks, device=X.device)).any(0)
        self._train_X, self._train_Y = X, Y
        self._transform_fit_X, self._transform_fit_Y = X, Y
        self._observed_task_indices, self._unobserved_task_indices = (
            observed.nonzero().squeeze(-1),
            (~observed).nonzero().squeeze(-1),
        )

    @classmethod
    def from_wide_format(
        cls,
        train_X: Any,
        train_Y: Any,
        historical_Xs: Any,
        historical_Ys: Any,
        train_Yvar: Any = None,
        output_tasks: Any = None,
        correction: int = 0,
        *,
        backend: str = "torch",
    ) -> Any:
        count = len(historical_Ys)
        if (
            train_X.shape[-1] != 1
            or train_Y.shape[-1] != count
            or train_X.shape[:-1] != train_Y.shape[:-1]
        ):
            raise ValueError("Wide training inputs and targets must match historical tasks.")
        batch, points = train_X.shape[:-2], train_X.shape[-2]
        progression = train_X.unsqueeze(-2).expand(*batch, points, count, 1)
        tasks = (
            torch.arange(count, dtype=train_X.dtype, device=train_X.device)
            .reshape(*(1 for _ in batch), 1, count, 1)
            .expand_as(progression)
        )
        long_X = torch.cat((progression, tasks), -1).reshape(*batch, points * count, 2)
        return cls(
            long_X,
            train_Y.reshape(*batch, points * count, 1),
            1,
            historical_Xs,
            historical_Ys,
            None
            if train_Yvar is None
            else train_Yvar.expand_as(train_Y).reshape(*batch, points * count, 1),
            output_tasks,
            correction,
            backend=backend,
        )

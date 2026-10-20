"""Device covariance and exact transfer across heterogeneous search spaces."""

from __future__ import annotations

import inspect
from operator import index as integer_index
from typing import Any, cast

import torch
from torch import Tensor, nn

from botorch_rs.exceptions.errors import BotorchTensorDimensionError, UnsupportedError
from botorch_rs.models._torch_backend_modules import TorchKernel, TorchScaleKernel, build_kernel
from botorch_rs.models._torch_backend_multitask import TorchMultiTaskGP
from botorch_rs.models.heterogeneous_mtgp import HeterogeneousMTGP
from botorch_rs.models.kernels.heterogeneous_multitask import (
    CombinatorialCovarModule,
    DeltaKernel,
    MultiTaskConditionalKernel,
    _features,
)


class TorchDeltaKernel(TorchKernel, DeltaKernel):  # type: ignore[misc]
    @property
    def batch_shape(self) -> Any:
        return self._tensor_batch_shape

    @batch_shape.setter
    def batch_shape(self, value: Any) -> None:
        self._tensor_batch_shape = torch.Size(value)

    def __init__(self, *args: Any, _like: Tensor | None = None, **kwargs: Any) -> None:
        kwargs.pop("backend", None)
        source = DeltaKernel(*args, **kwargs)
        like = torch.empty((), dtype=torch.get_default_dtype()) if _like is None else _like
        TorchKernel.__init__(self, like, source.active_dims)
        self.batch_shape = source.batch_shape

    def forward(
        self,
        x1: Tensor,
        x2: Tensor | None = None,
        diag: bool = False,
        last_dim_is_batch: bool = False,
        **params: Any,
    ) -> Tensor:
        x1, x2 = self._inputs(x1, x2)
        return self._forward_selected(x1, x2, diag, last_dim_is_batch)

    def _forward_selected(
        self, x1: Tensor, x2: Tensor, diag: bool, last_dim_is_batch: bool
    ) -> Tensor:
        if x1.shape[-1] != 1:
            raise ValueError("DeltaKernel expects 1D inputs.")
        left, right = x1[..., 0] == 1, x2[..., 0] == 1
        result = ((left & right) if diag else (left.unsqueeze(-1) & right.unsqueeze(-2))).to(x1)
        return result.unsqueeze(-2 if diag else -3) if last_dim_is_batch else result


class TorchCombinatorialCovarModule(TorchScaleKernel, CombinatorialCovarModule):  # type: ignore[misc]
    @property
    def batch_shape(self) -> Any:
        return self._tensor_batch_shape

    @batch_shape.setter
    def batch_shape(self, value: Any) -> None:
        self._tensor_batch_shape = torch.Size(value)

    def __init__(
        self,
        ard_num_dims: int | None = None,
        *,
        backend: str = "torch",
        _like: Tensor | None = None,
        _source: CombinatorialCovarModule | None = None,
    ) -> None:
        source = CombinatorialCovarModule(ard_num_dims) if _source is None else _source
        like = torch.empty((), dtype=torch.get_default_dtype()) if _like is None else _like
        TorchScaleKernel.__init__(
            self,
            source,
            cast(TorchKernel, build_kernel(source.base_kernel, ard_num_dims or 1, (), like)),
            like,
        )


class TorchMultiTaskConditionalKernel(TorchKernel, MultiTaskConditionalKernel):  # type: ignore[misc]
    _task_membership: Tensor

    @property
    def batch_shape(self) -> Any:
        return self._tensor_batch_shape

    @batch_shape.setter
    def batch_shape(self, value: Any) -> None:
        self._tensor_batch_shape = torch.Size(value)

    def __init__(
        self,
        *args: Any,
        _like: Tensor | None = None,
        _source: MultiTaskConditionalKernel | None = None,
        **kwargs: Any,
    ) -> None:
        kwargs.pop("backend", None)
        source = MultiTaskConditionalKernel(*args, **kwargs) if _source is None else _source
        like = torch.empty((), dtype=torch.get_default_dtype()) if _like is None else _like
        TorchKernel.__init__(self, like, source.active_dims)
        self.batch_shape = torch.Size()
        self.feature_indices = source.feature_indices
        self.task_feature_index = source.task_feature_index
        self.use_saas_prior = source.use_saas_prior
        self.use_combinatorial_kernel = source.use_combinatorial_kernel
        self.active_index_map, self.binary_map = source.active_index_map, source.binary_map
        self.register_buffer("_task_membership", like.new_tensor(self.binary_map))
        self.kernels = nn.ModuleList(  # type: ignore[assignment]
            [build_kernel(kernel, 0, (), like) for kernel in source.kernels]
        )
        self.combinatorial_kernel: Any = TorchCombinatorialCovarModule(
            len(self.kernels), _like=like, _source=source.combinatorial_kernel
        )

    def map_task_to_binary(self, x_task: Tensor) -> Tensor:
        if not bool(
            ((x_task >= 0) & (x_task < len(self.binary_map)) & (x_task == x_task.long())).all()
        ):
            raise ValueError("Task indices must be valid integer feature-map indices.")
        return self._task_membership[x_task.long()]

    def forward(
        self,
        x1: Tensor,
        x2: Tensor | None = None,
        diag: bool = False,
        last_dim_is_batch: bool = False,
        **params: Any,
    ) -> Tensor:
        x1, x2 = self._inputs(x1, x2)
        return self._forward_selected(x1, x2, diag, last_dim_is_batch)

    def _forward_selected(
        self, x1: Tensor, x2: Tensor, diag: bool, last_dim_is_batch: bool
    ) -> Tensor:
        if last_dim_is_batch:
            raise UnsupportedError("Heterogeneous task gates do not define feature batches.")
        left = self.map_task_to_binary(x1[..., self.task_feature_index])
        right = self.map_task_to_binary(x2[..., self.task_feature_index])
        shape = torch.broadcast_shapes(x1.shape[:-2], x2.shape[:-2])
        tail = (x1.shape[-2],) if diag else (x1.shape[-2], x2.shape[-2])
        result = x1.new_zeros(*shape, *tail)
        if self.use_combinatorial_kernel:
            result = self.combinatorial_kernel(left, right, diag=diag)
        for index, kernel in enumerate(self.kernels):
            a, b = left[..., index], right[..., index]
            gate = a * b if diag else a.unsqueeze(-1) * b.unsqueeze(-2)
            result = result + kernel(x1, x2, diag=diag) * gate
        return result


class TorchHeterogeneousMTGP(TorchMultiTaskGP, HeterogeneousMTGP):
    """Joint device inference with original task-feature observations as buffers."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        bound = inspect.signature(HeterogeneousMTGP.__init__).bind(self, *args, **kwargs)
        bound.apply_defaults()
        values = dict(bound.arguments)
        Xs, Ys, variances = values["train_Xs"], values["train_Ys"], values["train_Yvars"]
        features = values["feature_indices"]
        _features(features)
        if not Xs or len(Xs) != len(features) or len(Ys) != len(features):
            raise BotorchTensorDimensionError(
                "Each heterogeneous task needs inputs, targets and a feature map."
            )
        if variances is not None and len(variances) != len(features):
            raise BotorchTensorDimensionError(
                "Observation-noise lists must have one entry per task."
            )
        self.full_feature_dim = integer_index(values["full_feature_dim"])
        if self.full_feature_dim <= 0 or any(
            j >= self.full_feature_dim for row in features for j in row
        ):
            raise ValueError("Feature indices must lie within the positive full_feature_dim.")
        self.feature_indices = [list(row) for row in features]
        for index, (X, Y) in enumerate(zip(Xs, Ys, strict=True)):
            if not isinstance(X, Tensor) or not isinstance(Y, Tensor):
                raise TypeError("Heterogeneous Tensor models require Tensor data for every task.")
            if (
                X.ndim != 2
                or Y.ndim != 2
                or X.shape[-1] != len(features[index])
                or Y.shape != (X.shape[0], 1)
            ):
                raise BotorchTensorDimensionError(
                    "Each task needs unbatched inputs and one target column."
                )
            if X.shape[0]:
                self._validate_data(X, Y, None if variances is None else variances[index])
            elif (
                X.dtype not in (torch.float32, torch.float64)
                or Y.dtype != X.dtype
                or Y.device != X.device
            ):
                raise ValueError(
                    "Empty task inputs and targets must share a floating dtype/device."
                )
            if variances is not None and (
                not isinstance(variances[index], Tensor) or variances[index].shape != Y.shape
            ):
                raise BotorchTensorDimensionError(
                    "Every fixed-noise task needs noise with its target shape."
                )
            if X.dtype != Xs[0].dtype or X.device != Xs[0].device:
                raise ValueError("All tasks must share a dtype and device.")
        tasks = list(range(len(features)))
        if values["all_tasks"] is not None and list(values["all_tasks"]) != tasks:
            raise UnsupportedError(
                "Heterogeneous task labels must follow feature-map order 0..T-1."
            )
        imputation = self._compute_imputation_values(Xs, features, self.full_feature_dim)
        full_X = torch.cat([self.map_to_full_tensor(X, i, imputation) for i, X in enumerate(Xs)])
        full_Y = torch.cat(Ys)
        kernel = TorchMultiTaskConditionalKernel(
            features,
            use_saas_prior=values["use_saas_prior"],
            use_combinatorial_kernel=values["use_combinatorial_kernel"],
            _like=full_X,
        )
        kernel.active_dims = torch.arange(self.full_feature_dim + 1, device=full_X.device)
        self._hetero_ready = False
        self._hetero_task_count, self._hetero_condition_count = len(Xs), 0
        TorchMultiTaskGP.__init__(
            self,
            full_X,
            full_Y,
            task_feature=-1,
            train_Yvar=None if variances is None else torch.cat(variances),
            covar_module=kernel,
            output_tasks=[0],
            rank=values["rank"],
            all_tasks=tasks,
            input_transform=values["input_transform"],
            outcome_transform=values["outcome_transform"],
            validate_task_values=values["validate_task_values"],
        )
        self.register_buffer("feature_imputation_values", imputation)
        for index, (X, Y) in enumerate(zip(Xs, Ys, strict=True)):
            self.register_buffer(f"_hetero_X_{index}", X)
            self.register_buffer(f"_hetero_Y_{index}", Y)
            self.register_buffer(
                f"_hetero_var_{index}", None if variances is None else variances[index]
            )
        self._hetero_ready = True

    @staticmethod
    def _compute_imputation_values(
        train_Xs: list[Tensor], feature_indices: list[list[int]], full_feature_dim: int
    ) -> Tensor:
        columns = []
        for feature in range(full_feature_dim):
            observed = [
                X[:, indices.index(feature)]
                for X, indices in zip(train_Xs, feature_indices, strict=True)
                if feature in indices and X.shape[-2]
            ]
            columns.append(torch.cat(observed).mean() if observed else train_Xs[0].new_zeros(()))
        return torch.stack(columns)

    def map_to_full_tensor(
        self, X: Tensor, task_index: int, imputation_values: Tensor | None = None
    ) -> Tensor:
        if not 0 <= task_index < len(self.feature_indices):
            raise ValueError("Task has no feature mapping.")
        indices = self.feature_indices[task_index]
        if X.ndim < 2 or X.shape[-1] != len(indices):
            raise BotorchTensorDimensionError("Input width must match the task feature map.")
        imputation = (
            self.feature_imputation_values if imputation_values is None else imputation_values
        )
        if imputation.shape != (self.full_feature_dim,):
            raise BotorchTensorDimensionError("Imputation must contain full_feature_dim values.")
        columns = [
            X[..., indices.index(j) : indices.index(j) + 1]
            if j in indices
            else imputation[j].expand(*X.shape[:-1], 1)
            for j in range(self.full_feature_dim)
        ]
        return torch.cat([*columns, X.new_full((*X.shape[:-1], 1), task_index)], -1)

    def _target_full_inputs(self, X: Tensor) -> Tensor:
        if (
            not isinstance(X, Tensor)
            or X.ndim < 2
            or X.shape[-1] != len(self.feature_indices[0]) + 1
        ):
            raise ValueError(
                "Target inputs need their task-specific features followed by a task column."
            )
        if not bool((X[..., -1] == 0).all()):
            raise UnsupportedError("Heterogeneous predictions support target task 0.")
        return self.map_to_full_tensor(X[..., :-1], 0)

    def _refresh_training_graph(self) -> None:
        if not self._hetero_ready:
            return
        Xs = [getattr(self, f"_hetero_X_{i}") for i in range(self._hetero_task_count)]
        Ys = [getattr(self, f"_hetero_Y_{i}") for i in range(self._hetero_task_count)]
        extra_Xs = [
            getattr(self, f"_hetero_condition_X_{i}") for i in range(self._hetero_condition_count)
        ]
        expected = sum(X.shape[-2] for X in (*Xs, *extra_Xs))
        # Base conditioning builds its copy before registering the new public
        # observation. Do not overwrite that intermediate copy's appended rows.
        if self._train_X.shape[-2] != expected:
            return
        self.feature_imputation_values = self._compute_imputation_values(
            Xs, self.feature_indices, self.full_feature_dim
        )
        original_X = torch.cat([self.map_to_full_tensor(X, i) for i, X in enumerate(Xs)])
        original_Y = torch.cat(Ys)
        full_Xs = [original_X, *(self._target_full_inputs(X) for X in extra_Xs)]
        full_Ys = [
            original_Y,
            *(
                getattr(self, f"_hetero_condition_Y_{i}")
                for i in range(self._hetero_condition_count)
            ),
        ]
        batch = torch.broadcast_shapes(*(X.shape[:-2] for X in (*full_Xs, *full_Ys)))
        self._train_X = torch.cat([X.expand(*batch, *X.shape[-2:]) for X in full_Xs], -2)
        self._train_Y = torch.cat([Y.expand(*batch, *Y.shape[-2:]) for Y in full_Ys], -2)
        self._transform_fit_X, self._transform_fit_Y = original_X, original_Y

    def _fit_transform_graphs(self) -> None:
        self._refresh_training_graph()
        TorchMultiTaskGP._fit_transform_graphs(self)
        if self._hetero_ready and self._train_Yvar is not None:
            noise = torch.cat(
                [getattr(self, f"_hetero_var_{i}") for i in range(self._hetero_task_count)]
            )
            pieces = [noise]
            for index in range(self._hetero_condition_count):
                extra = getattr(self, f"_hetero_condition_noise_{index}")
                if self.outcome_transform is not None:
                    extra = self.outcome_transform.untransform(torch.zeros_like(extra), extra)[1]
                pieces.append(extra)
            if sum(part.shape[-2] for part in pieces) == self._train_Y.shape[-2]:
                self._train_Yvar = torch.cat(
                    [part.expand(*self.batch_shape, *part.shape[-2:]) for part in pieces], -2
                )

    def posterior(
        self,
        X: Tensor,
        output_indices: Any = None,
        observation_noise: Any = False,
        posterior_transform: Any = None,
        **kwargs: Any,
    ) -> Any:
        if output_indices is not None and list(output_indices) != [0]:
            raise UnsupportedError("Heterogeneous predictions support target task 0.")
        self._refresh_training_graph()
        return TorchMultiTaskGP.posterior(
            self,
            self._target_full_inputs(X),
            observation_noise=observation_noise,
            posterior_transform=posterior_transform,
            **kwargs,
        )

    def condition_on_observations(
        self, X: Tensor, Y: Tensor, noise: Any = None, **kwargs: Any
    ) -> Any:
        self._refresh_training_graph()
        result = TorchMultiTaskGP.condition_on_observations(
            self, self._target_full_inputs(X), Y, noise, **kwargs
        )
        index = self._hetero_condition_count
        result.register_buffer(f"_hetero_condition_X_{index}", X)
        result.register_buffer(
            f"_hetero_condition_Y_{index}",
            Y if Y.ndim >= 2 and Y.shape[-1] == 1 else Y.unsqueeze(-1),
        )
        result.register_buffer(f"_hetero_condition_noise_{index}", noise)
        result._hetero_condition_count = index + 1
        return result

    def set_train_data(self, inputs: Any = None, targets: Any = None, strict: bool = True) -> None:
        # Explicit full-space replacement supersedes the constructor's
        # heterogeneous source lists, following ExactGP's public data API.
        previous = self._hetero_ready
        self._hetero_ready = False
        try:
            TorchMultiTaskGP.set_train_data(self, inputs, targets, strict)
        except BaseException:
            self._hetero_ready = previous
            raise

    def fantasize(
        self, X: Tensor, sampler: Any, observation_noise: Any = None, **kwargs: Any
    ) -> Any:
        propagate = kwargs.pop("propagate_grads", False)
        if kwargs or observation_noise is not None:
            raise UnsupportedError(
                "Heterogeneous fantasies use inferred task noise and optional propagate_grads."
            )
        targets = sampler(self.posterior(X, observation_noise=True))
        if not propagate:
            targets = targets.detach()
        noise = None
        if self._train_Yvar is not None:
            _, _, fixed = self._training_values()
            full = self._target_full_inputs(X)
            noise = self._fixed_noise_at(
                self._mapped_inputs(self.transform_inputs(full)), fixed
            ).unsqueeze(-1)
        return self.condition_on_observations(X, targets, noise=noise)

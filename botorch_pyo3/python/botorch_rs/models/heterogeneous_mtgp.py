"""Native exact multitask GPs over heterogeneous feature spaces."""

from __future__ import annotations

from operator import index as integer_index
from typing import Any, cast

from botorch_rs import _botorch_rs as _rs
from botorch_rs import _ops
from botorch_rs._arrays import kind_of, to_flat
from botorch_rs.exceptions.errors import BotorchTensorDimensionError, UnsupportedError
from botorch_rs.models.kernels.heterogeneous_multitask import MultiTaskConditionalKernel, _features
from botorch_rs.models.multitask import MultiTaskGP


class HeterogeneousMTGP(MultiTaskGP):
    """Transfer across tasks with partially shared features, predicting task 0.

    Spatial and task-membership covariances, exact GP inference, priors,
    derivatives, and fitting use the Rust kernels and multitask model.
    """

    def __init__(
        self,
        train_Xs: list[Any],
        train_Ys: list[Any],
        train_Yvars: list[Any] | None,
        feature_indices: list[list[int]],
        full_feature_dim: int,
        rank: int | None = None,
        use_saas_prior: bool = True,
        use_combinatorial_kernel: bool = True,
        all_tasks: list[int] | None = None,
        input_transform: Any = None,
        outcome_transform: Any = None,
        validate_task_values: bool = True,
    ) -> None:
        features = _features(feature_indices)
        count = len(features)
        if len(train_Xs) != count or len(train_Ys) != count:
            raise BotorchTensorDimensionError(
                "Training inputs, targets, and feature maps must have one entry per task."
            )
        if train_Yvars is not None and len(train_Yvars) != count:
            raise BotorchTensorDimensionError(
                "Observation-noise lists must have one entry per task."
            )
        self.full_feature_dim = integer_index(full_feature_dim)
        if self.full_feature_dim <= 0 or any(
            index >= self.full_feature_dim for task in features for index in task
        ):
            raise ValueError("Feature indices must lie within the positive full_feature_dim.")
        self.feature_indices = [list(indices) for indices in feature_indices]
        if input_transform is not None:
            transformed = getattr(input_transform, "indices", None)
            if transformed is None or self.full_feature_dim in transformed:
                raise UnsupportedError(
                    "Heterogeneous input transforms must leave the task column unchanged; "
                    "use Normalize(full_feature_dim + 1, indices=range(full_feature_dim))."
                )
        for i, (x, y) in enumerate(zip(train_Xs, train_Ys, strict=True)):
            xs, ys = to_flat(x, f"train_Xs[{i}]")[1], to_flat(y, f"train_Ys[{i}]")[1]
            if len(xs) != 2 or xs[-1] != len(self.feature_indices[i]):
                raise BotorchTensorDimensionError(
                    f"Task {i} requires unbatched n x {len(self.feature_indices[i])} inputs, got {xs}."
                )
            if tuple(ys) != (xs[0], 1):
                raise BotorchTensorDimensionError(
                    f"Task {i} requires targets with shape {(xs[0], 1)}."
                )
            if train_Yvars is not None:
                if train_Yvars[i] is None or to_flat(train_Yvars[i])[1] != ys:
                    raise BotorchTensorDimensionError(
                        f"Task {i} requires observation noise with target shape {ys}."
                    )
        task_values = (
            list(range(count)) if all_tasks is None else [integer_index(t) for t in all_tasks]
        )
        if task_values != list(range(count)):
            raise UnsupportedError(
                "Heterogeneous task values must be 0..len(feature_indices)-1 in feature-map order."
            )
        imputation = self._compute_imputation_values(
            train_Xs, feature_indices, self.full_feature_dim
        )
        full_x = _ops.concat(
            [self.map_to_full_tensor(x, i, imputation) for i, x in enumerate(train_Xs)], dim=0
        )
        full_y = _ops.concat(train_Ys, dim=0)
        full_yvar = None if train_Yvars is None else _ops.concat(train_Yvars, dim=0)
        kernel = MultiTaskConditionalKernel(
            feature_indices,
            use_saas_prior=use_saas_prior,
            use_combinatorial_kernel=use_combinatorial_kernel,
        )
        kernel.active_dims = tuple(range(self.full_feature_dim + 1))
        super().__init__(
            full_x,
            full_y,
            task_feature=-1,
            train_Yvar=full_yvar,
            covar_module=kernel,
            output_tasks=[0],
            rank=rank,
            all_tasks=task_values,
            input_transform=input_transform,
            outcome_transform=outcome_transform,
            validate_task_values=validate_task_values,
        )
        self.feature_imputation_values = imputation

    @classmethod
    def get_all_tasks(
        cls, train_X: Any, task_feature: int, output_tasks: list[int] | None = None
    ) -> tuple[list[int], int, int]:
        tasks, feature, dim = super().get_all_tasks(train_X, task_feature, output_tasks)
        return (tasks if 0 in tasks else [0, *tasks]), feature, dim

    @staticmethod
    def _compute_imputation_values(
        train_Xs: list[Any], feature_indices: list[list[int]], full_feature_dim: int
    ) -> Any:
        """Empirical means of observed feature columns, reduced in Rust."""
        kind = kind_of(train_Xs[0])
        columns = []
        for feature in range(full_feature_dim):
            observed = [
                _ops.getitem(x, (Ellipsis, indices.index(feature)))
                for x, indices in zip(train_Xs, feature_indices, strict=True)
                if feature in indices and to_flat(x)[1][0] > 0
            ]
            if observed:
                mean = _ops.reduce("mean", _ops.concat(observed, dim=0), dim=0)
                columns.append(_ops.getitem(mean, None))
            else:
                columns.append(_ops.full((1,), 0.0, kind))
        return _ops.concat(columns, dim=0)

    def map_to_full_tensor(self, X: Any, task_index: int, imputation_values: Any = None) -> Any:
        """Embed task features and append their task indicator.

        Missing features receive the empirical mean across their observed
        tasks. Structural array operations preserve query-input autograd.
        """
        task = integer_index(task_index)
        if not 0 <= task < len(self.feature_indices):
            raise ValueError(f"Task {task} has no feature mapping.")
        indices = self.feature_indices[task]
        shape = _ops.shape_of(X)
        if len(shape) < 2 or shape[-1] != len(indices):
            raise BotorchTensorDimensionError(
                f"Task {task} expects inputs with {len(indices)} columns, got {shape}."
            )
        kind = kind_of(X)
        imputation = (
            self.feature_imputation_values if imputation_values is None else imputation_values
        )
        values, value_shape = to_flat(imputation, "imputation_values")
        if tuple(value_shape) != (self.full_feature_dim,):
            raise BotorchTensorDimensionError(
                "Imputation values must have full_feature_dim entries."
            )
        columns = [
            _ops.getitem(X, (Ellipsis, slice(indices.index(j), indices.index(j) + 1)))
            if j in indices
            else _ops.full((*shape[:-1], 1), values[j], kind)
            for j in range(self.full_feature_dim)
        ]
        columns.append(_ops.full((*shape[:-1], 1), float(task), kind))
        return _ops.concat(columns, dim=-1)

    def posterior(
        self,
        X: Any,
        output_indices: list[int] | None = None,
        observation_noise: Any = False,
        posterior_transform: Any = None,
        **kwargs: Any,
    ) -> Any:
        if output_indices is not None and output_indices != [0]:
            raise UnsupportedError("Heterogeneous MTGP does not support `output_indices`.")
        full = self._target_full_inputs(X)
        return super().posterior(
            full,
            observation_noise=observation_noise,
            posterior_transform=posterior_transform,
            **kwargs,
        )

    def _target_full_inputs(self, X: Any) -> Any:
        data, shape = to_flat(X, "X")
        columns = len(self.feature_indices[0]) + 1
        if len(shape) < 2 or shape[-1] != columns:
            raise ValueError(
                f"Expected X with {columns} columns ({columns - 1} target features + 1 task column), got {shape}."
            )
        if any(value != 0.0 for value in data[columns - 1 :: columns]):
            raise UnsupportedError("Posterior can only be called for the target task.")
        return self.map_to_full_tensor(_ops.getitem(X, (Ellipsis, slice(None, -1))), 0)

    def _acquisition_model(self, posterior_transform: Any = None) -> _rs.Model:
        # The native multitask source accepts full features without the task
        # column. The adapter applies the public heterogeneous input contract
        # and projects all input derivatives to the target feature ordering.
        return _rs.target_task_model(
            super()._acquisition_model(posterior_transform),
            self.feature_indices[0],
            list(to_flat(self.feature_imputation_values)[0]),
        )

    def condition_on_observations(
        self, X: Any, Y: Any, noise: Any = None, **kwargs: Any
    ) -> HeterogeneousMTGP:
        return cast(
            HeterogeneousMTGP,
            super().condition_on_observations(self._target_full_inputs(X), Y, noise, **kwargs),
        )

    def fantasize(
        self, X: Any, sampler: Any, observation_noise: Any = None, **kwargs: Any
    ) -> HeterogeneousMTGP:
        return cast(
            HeterogeneousMTGP,
            super().fantasize(self._target_full_inputs(X), sampler, observation_noise, **kwargs),
        )

    def _split_inputs(self, x: Any) -> tuple[Any, Any, Any]:
        """Return full features, integer task indices, and an empty trailing block."""
        task = _ops.getitem(x, (Ellipsis, slice(self._task_feature, self._task_feature + 1)))
        data, shape = to_flat(task)
        return (
            x,
            _ops.int_array([int(value) for value in data], shape, kind_of(x)),
            _ops.getitem(x, (Ellipsis, slice(0, 0))),
        )

    @classmethod
    def construct_inputs(  # type: ignore[override]
        cls,
        training_data: Any,
        task_feature: int = -1,
        output_tasks: list[int] | None = None,
        rank: int | None = None,
        use_saas_prior: bool = True,
        use_combinatorial_kernel: bool = True,
        map_heterogeneous_to_full: bool = False,
    ) -> dict[str, Any]:
        if training_data.task_feature_index != -1:
            raise NotImplementedError("Heterogeneous MTGP requires `task_feature_index` to be -1.")
        if task_feature != -1:
            raise NotImplementedError("Heterogeneous MTGP requires `task_feature=-1`.")
        if output_tasks is not None and output_tasks != [0]:
            raise NotImplementedError(
                "Heterogeneous MTGP currently only supports output_tasks=[0]."
            )
        datasets, features, dim = training_data.get_heterogeneous_feature_mapping()
        return {
            "train_Xs": [_ops.getitem(ds.X, (Ellipsis, slice(None, -1))) for ds in datasets],
            "train_Ys": [ds.Y for ds in datasets],
            "train_Yvars": None if datasets[0].Yvar is None else [ds.Yvar for ds in datasets],
            "feature_indices": features,
            "full_feature_dim": dim,
            "rank": rank,
            "use_saas_prior": use_saas_prior,
            "use_combinatorial_kernel": use_combinatorial_kernel,
            "all_tasks": list(range(len(datasets))),
        }


__all__ = ["HeterogeneousMTGP"]

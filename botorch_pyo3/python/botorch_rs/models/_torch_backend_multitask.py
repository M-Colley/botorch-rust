"""Device-resident long-format exact multitask GPs with complete Torch graphs."""

from __future__ import annotations

import copy
from typing import Any, cast

import torch
from torch import nn

from botorch_rs import _botorch_rs as _rs
from botorch_rs._arrays import kind_of
from botorch_rs.constraints import GreaterThan, Positive
from botorch_rs.exceptions.errors import UnsupportedError
from botorch_rs.models._torch_backend import TorchSingleTaskGP
from botorch_rs.models._torch_backend_modules import (
    TorchCompositeKernel,
    TorchConstraint,
    TorchFixedNoiseLikelihood,
    TorchGaussianLikelihood,
    TorchKernel,
    _TorchModule,
    build_input_transform,
    build_kernel,
    build_mean,
    build_outcome_transform,
    module_log_prior,
)
from botorch_rs.models.multitask import MultiTaskGP, _is_default
from botorch_rs.posteriors._torch_backend import TorchGPyTorchPosterior
from botorch_rs.utils.types import DEFAULT


class TorchPositiveIndexKernel(TorchKernel):
    """Positive low-rank task covariance, normalized by task zero's variance."""

    def __init__(self, num_tasks: int, rank: int, prior: Any, task_feature: int, like: Any):
        super().__init__(like, [task_feature])
        if prior is not None and "LKJ" in type(prior).__name__:
            raise UnsupportedError(
                "Torch task covariance currently supports univariate correlation priors; LKJ covariance priors require a separate matrix-prior implementation."
            )
        if isinstance(prior, nn.Module) and not callable(getattr(prior, "log_prob", None)):
            raise TypeError("A task covariance prior must expose log_prob.")
        self.num_tasks, self.rank = num_tasks, rank
        self.batch_shape = torch.Size()
        self.target_task_index, self.unit_scale_for_target = 0, True
        self.normalize_covar_matrix = False
        self.raw_var_constraint = TorchConstraint(Positive(), like)
        self.raw_covar_factor_constraint = TorchConstraint(GreaterThan(0.0), like)
        # Match upstream's construction dtype/generator before Module.to(X).
        self.raw_var = nn.Parameter(torch.randn(num_tasks).to(like))
        self.raw_covar_factor = nn.Parameter(torch.rand(num_tasks, rank).to(like))
        if isinstance(prior, nn.Module):
            self.add_module("IndexKernelPrior", prior.to(like))
            self._prior_parameters["IndexKernelPrior"] = ("_lower_triangle_corr", 1)
        else:
            self._register_prior("IndexKernelPrior", prior, "_lower_triangle_corr", 1)

    @property
    def var(self) -> Any:
        return self.raw_var_constraint.transform(self.raw_var)

    @var.setter
    def var(self, value: Any) -> None:
        self._assign("raw_var", value, self.raw_var_constraint)

    @property
    def covar_factor(self) -> Any:
        return self.raw_covar_factor_constraint.transform(self.raw_covar_factor)

    @covar_factor.setter
    def covar_factor(self, value: Any) -> None:
        self._set_covar_factor(value)

    def _set_covar_factor(self, value: Any) -> None:
        self._assign("raw_covar_factor", value, self.raw_covar_factor_constraint)

    @property
    def covar_matrix(self) -> Any:
        factor = self.covar_factor
        covariance = factor @ factor.transpose(-1, -2) + torch.diag_embed(self.var)
        return covariance / covariance[self.target_task_index, self.target_task_index]

    @property
    def _lower_triangle_corr(self) -> Any:
        covariance = self.covar_matrix
        std = covariance.diagonal().sqrt()
        correlation = covariance / (std.unsqueeze(-1) * std.unsqueeze(-2))
        indices = torch.tril_indices(self.num_tasks, self.num_tasks, -1, device=std.device)
        return correlation[indices[0], indices[1]]

    def forward(
        self, x1: Any, x2: Any = None, diag: bool = False, last_dim_is_batch: bool = False
    ) -> Any:
        if last_dim_is_batch:
            raise UnsupportedError("Task covariance does not support last_dim_is_batch.")
        x1, x2 = self._inputs(x1, x2)
        left, right = x1[..., 0].long(), x2[..., 0].long()
        covariance = self.covar_matrix
        return (
            covariance[left, right] if diag else covariance[left.unsqueeze(-1), right.unsqueeze(-2)]
        )


class TorchTaskMean(_TorchModule):
    def __init__(self, num_tasks: int, like: Any):
        super().__init__(like)
        self.num_tasks = num_tasks
        self.base_means = nn.ModuleList([build_mean(None, (), like) for _ in range(num_tasks)])

    def forward(self, X: Any) -> Any:
        return torch.stack([mean(X) for mean in self.base_means], dim=-1)


class _TorchTaskNoise(_TorchModule):
    def __init__(self, source: Any, like: Any):
        super().__init__(like)
        self.raw_noise_constraint = TorchConstraint(source.noise_constraint, like)
        initial = self.raw_noise_constraint.initial_value
        self.raw_noise = nn.Parameter(
            like.new_full((source.num_tasks,), 0 if initial is None else initial)
        )
        if getattr(source, "_model", None) is not None:
            self._assign("raw_noise", source.raw_noise)
        self._register_prior("noise_prior", source.noise_prior, "noise", 1)

    @property
    def noise_constraint(self) -> Any:
        return self.raw_noise_constraint

    @property
    def noise(self) -> Any:
        return self.raw_noise_constraint.transform(self.raw_noise)

    @noise.setter
    def noise(self, value: Any) -> None:
        self._assign("raw_noise", value, self.raw_noise_constraint)


class TorchHadamardLikelihood(TorchGaussianLikelihood):
    noise_covar: Any

    def __init__(self, source: Any, task_feature: int, like: Any):
        _TorchModule.__init__(self, like)
        self.batch_shape = torch.Size()
        self.num_tasks, self.task_feature_index = source.num_tasks, task_feature
        self.noise_covar = _TorchTaskNoise(source, like)

    def forward(self, distribution: Any, *inputs: Any, noise: Any = None, **kwargs: Any) -> Any:
        if noise is None:
            if len(inputs) != 1:
                raise ValueError(
                    "Hadamard likelihood requires mapped task inputs or explicit noise."
                )
            noise = self.noise[inputs[0][..., self.task_feature_index].long()]
        return super().forward(distribution, noise=noise, **kwargs)


class _ContinuousKernel(_TorchModule):
    """Give a custom module without active-dimension dispatch only continuous inputs."""

    def __init__(self, kernel: Any, indices: Any, like: Any):
        super().__init__(like)
        self.base_kernel = kernel
        self.batch_shape = torch.Size(getattr(kernel, "batch_shape", ()))
        self.register_buffer("indices", indices)

    def forward(self, x1: Any, x2: Any, diag: bool = False, last_dim_is_batch: bool = False) -> Any:
        if last_dim_is_batch:
            raise UnsupportedError("Custom continuous kernels do not support last_dim_is_batch.")
        value = self.base_kernel(
            x1.index_select(-1, self.indices), x2.index_select(-1, self.indices)
        )
        value = value if isinstance(value, torch.Tensor) else value.to_dense()
        return value.diagonal(dim1=-2, dim2=-1) if diag else value

    def log_prior(self) -> Any:
        return module_log_prior(self.base_kernel, self.batch_shape)


class TorchMultiTaskGP(nn.Module, MultiTaskGP):
    """An ICM exact GP: native tensor solves and port-owned Torch task modules.

    Training data are unbatched, as in the pinned upstream MultiTaskGP. Query
    batches broadcast freely. Task labels are categorical: their gradients are
    zero, while continuous inputs, targets, noise and raw parameters retain
    higher-order autograd graphs. Conditioning and fantasies append long-format
    observations to the correlated law. Tensor-valued posterior noise remains
    explicitly unsupported, as in the pinned reference.
    """

    _tensor_backend = "torch"
    _supports_batched_models = False
    _supports_cache_root = False
    _train_X: Any
    _conditioned_train_X: Any
    _train_Y: Any
    _train_Yvar: Any
    _noise_override: Any
    _transform_fit_X: Any
    _transform_fit_Y: Any
    _base_idxr: Any
    _task_mapper: Any
    _task_labels: Any
    _observed_task_indices: Any
    _unobserved_task_indices: Any
    input_transform: Any
    outcome_transform: Any
    likelihood: Any
    mean_module: Any
    covar_module: Any
    _validate_data = staticmethod(TorchSingleTaskGP._validate_data)
    _fit_transform_graphs: Any = TorchSingleTaskGP._fit_transform_graphs
    train: Any = TorchSingleTaskGP.train
    transform_inputs: Any = TorchSingleTaskGP.transform_inputs
    _covariance: Any = TorchSingleTaskGP._covariance
    _sync_torch_parameters: Any = TorchSingleTaskGP._sync_torch_parameters
    initialize: Any = TorchSingleTaskGP.initialize
    __deepcopy__: Any = TorchSingleTaskGP.__deepcopy__

    def __init__(
        self,
        train_X: Any,
        train_Y: Any,
        task_feature: int,
        train_Yvar: Any = None,
        mean_module: Any = None,
        covar_module: Any = None,
        likelihood: Any = None,
        task_covar_prior: Any = DEFAULT,
        output_tasks: Any = None,
        rank: int | None = None,
        all_tasks: Any = None,
        outcome_transform: Any = DEFAULT,
        input_transform: Any = None,
        validate_task_values: bool = True,
        *,
        backend: str = "torch",
    ) -> None:
        from botorch_rs.likelihoods import GaussianLikelihood, HadamardGaussianLikelihood
        from botorch_rs.models.utils.gpytorch_modules import get_covar_module_with_dim_scaled_prior
        from botorch_rs.models.utils.priors import BetaPrior
        from botorch_rs.priors import LogNormalPrior

        nn.Module.__init__(self)
        self._validate_data(train_X, train_Y, train_Yvar)
        if train_X.ndim != 2 or train_Y.shape[-1] != 1:
            raise ValueError("MultiTaskGP requires unbatched train_X (n,d+1) and train_Y (n,1).")
        self.input_transform = build_input_transform(input_transform, train_X.shape[-1], train_X)
        with torch.no_grad():
            outcome_inputs = (
                train_X
                if self.input_transform is None
                else getattr(self.input_transform, "preprocess_transform", self.input_transform)(
                    train_X
                )
            )
        if outcome_inputs.shape[-2] != train_Y.shape[-2]:
            raise ValueError(
                "The training input transform must preserve the number of target rows."
            )
        d = outcome_inputs.shape[-1] - 1
        if d < 1 or not (-(d + 1) <= task_feature <= d):
            raise ValueError(f"task_feature must be between {-d - 1} and {d}.")
        self._task_feature = task_feature % (d + 1)
        self.num_non_task_features = d
        raw_tasks = outcome_inputs[..., self._task_feature]
        if not bool(((raw_tasks >= 0) & (raw_tasks == raw_tasks.long())).all()):
            raise ValueError("Task labels must be nonnegative integers.")
        observed = raw_tasks.long().unique(sorted=True).tolist()
        tasks = observed if all_tasks is None else list(all_tasks)
        if (
            not tasks
            or len(set(tasks)) != len(tasks)
            or any(not isinstance(t, int) or t < 0 for t in tasks)
        ):
            raise ValueError("all_tasks must contain distinct nonnegative integer labels.")
        if not set(observed).issubset(tasks):
            raise UnsupportedError(
                "all_tasks must contain every task present in the training data."
            )
        self.num_tasks = len(tasks)
        self.all_tasks = tasks
        self._output_tasks = tasks if output_tasks is None else list(output_tasks)
        if not self._output_tasks or not set(self._output_tasks).issubset(tasks):
            raise ValueError("output_tasks must be nonempty and present in all_tasks.")
        self._num_outputs = len(self._output_tasks)
        self._validate_task_values = validate_task_values
        self._expected_task_values = set(tasks)
        self._rank = self.num_tasks if rank is None else rank
        if not isinstance(self._rank, int) or not (0 <= self._rank <= self.num_tasks):
            raise ValueError("rank must be between zero and the number of tasks.")
        self._kind = kind_of(train_X)
        self._rs = None
        self._prediction_cache: Any = None
        self._transform_fit_versions: dict[int, Any] = {}
        self.register_buffer("_train_X", train_X)
        self.register_buffer("_train_Y", train_Y)
        self.register_buffer("_train_Yvar", train_Yvar)
        self.register_buffer("_transform_fit_X", train_X)
        self.register_buffer("_transform_fit_Y", train_Y)
        self.register_buffer("_noise_override", None)
        self.register_buffer("_conditioned_train_X", None)
        sorted_tasks = sorted(tasks)
        labels = torch.tensor(sorted_tasks, dtype=torch.long, device=train_X.device)
        self.register_buffer("_task_labels", labels)
        self.register_buffer(
            "_base_idxr",
            torch.tensor(
                [i for i in range(d + 1) if i != self._task_feature],
                dtype=torch.long,
                device=train_X.device,
            ),
        )
        mapper = None
        if sorted_tasks != list(range(self.num_tasks)):
            mapper = train_X.new_full((sorted_tasks[-1] + 1,), torch.nan)
            mapper[labels] = torch.arange(
                self.num_tasks, dtype=train_X.dtype, device=train_X.device
            )
        self.register_buffer("_task_mapper", mapper)
        self.register_buffer(
            "_observed_task_indices",
            torch.tensor(
                [i for i, t in enumerate(sorted_tasks) if t in observed],
                dtype=torch.long,
                device=train_X.device,
            ),
        )
        self.register_buffer(
            "_unobserved_task_indices",
            torch.tensor(
                [i for i, t in enumerate(sorted_tasks) if t not in observed],
                dtype=torch.long,
                device=train_X.device,
            ),
        )
        self.outcome_transform = build_outcome_transform(
            outcome_transform, 1, train_Y, train_X=outcome_inputs
        )
        self._is_custom_likelihood = likelihood is not None
        _, _, fixed_noise = self._training_values()
        if covar_module is None:
            covar_module = get_covar_module_with_dim_scaled_prior(
                d, active_dims=self._base_idxr.tolist()
            )
        data_kernel: Any = build_kernel(covar_module, d, (), train_X)
        if hasattr(data_kernel, "active_dims"):
            if data_kernel.active_dims is None:
                data_kernel.active_dims = self._base_idxr
        else:
            data_kernel = _ContinuousKernel(data_kernel, self._base_idxr, train_X)
        if _is_default(task_covar_prior):
            task_covar_prior = BetaPrior(2.5, 1.5)
        task_kernel = TorchPositiveIndexKernel(
            self.num_tasks, self._rank, task_covar_prior, self._task_feature, train_X
        )
        self.covar_module = TorchCompositeKernel([data_kernel, task_kernel], "product", train_X)
        self.mean_module = (
            TorchTaskMean(self.num_tasks, train_X)
            if mean_module is None
            else build_mean(mean_module, (), train_X)
        )
        if fixed_noise is not None and likelihood is None:
            self.likelihood = TorchFixedNoiseLikelihood(fixed_noise, (), train_X)
        else:
            if likelihood is None:
                noise_prior = LogNormalPrior(-4.0, 1.0)
                likelihood = HadamardGaussianLikelihood(
                    self.num_tasks,
                    noise_prior=noise_prior,
                    noise_constraint=GreaterThan(
                        1e-4, transform=None, initial_value=noise_prior.mode
                    ),
                    task_feature_index=self._task_feature,
                )
            if type(likelihood) is HadamardGaussianLikelihood:
                if likelihood.num_tasks != self.num_tasks:
                    raise ValueError("Likelihood task count must match the model.")
                self.likelihood = TorchHadamardLikelihood(likelihood, self._task_feature, train_X)
            elif type(likelihood) is GaussianLikelihood:
                self.likelihood = TorchGaussianLikelihood(likelihood, train_X)
            elif isinstance(likelihood, nn.Module):
                gaussian = isinstance(
                    likelihood, (TorchGaussianLikelihood, TorchFixedNoiseLikelihood)
                ) or any(
                    cls.__name__
                    in (
                        "_GaussianLikelihoodBase",
                        "GaussianLikelihood",
                        "HadamardGaussianLikelihood",
                    )
                    for cls in type(likelihood).__mro__
                )
                if not gaussian:
                    raise UnsupportedError(
                        "A custom likelihood must be a Torch Gaussian likelihood; a noise attribute alone does not specify a Gaussian distribution."
                    )
                self.likelihood = cast(Any, likelihood).to(train_X)
            else:
                raise UnsupportedError("Torch MultiTaskGP requires a Gaussian likelihood marginal.")

    @property
    def num_outputs(self) -> int:
        return self._num_outputs

    @property
    def batch_shape(self) -> Any:
        return self._train_X.shape[:-2]

    @property
    def _aug_batch_shape(self) -> Any:
        return self.batch_shape

    @property
    def _input_batch_shape(self) -> Any:
        return self.batch_shape

    def _augment_X(self, X: Any) -> Any:
        return X

    def _augment_Y(self, Y: Any) -> Any:
        return Y.squeeze(-1)

    def _training_values(self) -> Any:
        X, Y, noise = TorchSingleTaskGP._training_values(cast(Any, self))
        if getattr(self, "_conditioned_train_X", None) is None:
            self._check_task_transform(self._train_X, X)
        return X, Y, noise

    def _check_task_transform(self, original: Any, transformed: Any) -> None:
        if original.shape[-1] != transformed.shape[-1]:
            self._map_tasks(transformed[..., self._task_feature])
            return
        count = original.shape[-2]
        if count == 0:
            return
        copies, remainder = divmod(transformed.shape[-2], count)
        expected = original[..., self._task_feature].repeat_interleave(copies, dim=-1)
        if remainder or not bool((transformed[..., self._task_feature] == expected).all()):
            raise UnsupportedError(
                "Input transforms must preserve categorical task labels; exclude task_feature."
            )

    @property
    def train_inputs(self) -> Any:
        conditioned = getattr(self, "_conditioned_train_X", None)
        return (self._train_X if conditioned is None or self.training else conditioned,)

    @property
    def train_targets(self) -> Any:
        return self._training_values()[1]

    def _map_tasks(self, task_values: Any) -> Any:
        if not self._validate_task_values:
            return (
                task_values if self._task_mapper is None else self._task_mapper[task_values.long()]
            )
        matches = task_values.unsqueeze(-1) == self._task_labels
        if not bool(matches.any(-1).all()):
            raise ValueError("Received invalid task labels; expected labels in all_tasks.")
        indices = matches.long().argmax(-1)
        return indices.to(dtype=self._train_X.dtype)

    def _mapped_inputs(self, X: Any) -> Any:
        mapped = self._map_tasks(X[..., self._task_feature])
        if not self._validate_task_values and not bool(
            ((mapped >= 0) & (mapped < self.num_tasks)).all()
        ):
            # Prevent an unchecked CUDA gather from triggering a device-side
            # assertion and poisoning all later operations in this process.
            raise IndexError("Task remapping produced an invalid covariance index.")
        return torch.cat(
            (
                X[..., : self._task_feature],
                mapped.unsqueeze(-1),
                X[..., self._task_feature + 1 :],
            ),
            dim=-1,
        )

    def _mean(self, X: Any) -> Any:
        if isinstance(self.mean_module, TorchTaskMean) or hasattr(self.mean_module, "base_means"):
            means = self.mean_module(X.index_select(-1, self._base_idxr))
            return means.gather(-1, X[..., self._task_feature].long().unsqueeze(-1)).squeeze(-1)
        return self.mean_module(X)

    def _training_noise(self, Y: Any, fixed_noise: Any) -> Any:
        inputs = getattr(self, "_conditioned_train_X", None)
        if inputs is None:
            inputs = self._train_X
            if inputs.shape[-1] != self.num_non_task_features + 1:
                inputs = self._training_values()[0]
        result = (
            fixed_noise if fixed_noise is not None else self._noise_at(self._mapped_inputs(inputs))
        )
        if self._noise_override is not None:
            result = torch.where(torch.isnan(self._noise_override), result, self._noise_override)
        return result

    def _fixed_noise_at(self, mapped_X: Any, fixed: Any) -> Any:
        inputs = getattr(self, "_conditioned_train_X", None)
        if inputs is None:
            inputs = self._train_X
            if inputs.shape[-1] != self.num_non_task_features + 1:
                inputs = self._training_values()[0]
        train_tasks = self._mapped_inputs(inputs)[..., self._task_feature].long()
        mask = train_tasks.unsqueeze(-1) == torch.arange(self.num_tasks, device=mapped_X.device)
        counts = mask.sum(-2)
        noise_by_task = (fixed.unsqueeze(-1) * mask).sum(-2) / counts.clamp_min(1)
        # An unobserved task has no task-specific noise estimate. Use the
        # observed global mean, matching the repaired native/empirical law.
        noise_by_task = torch.where(counts > 0, noise_by_task, fixed.mean(-1, keepdim=True))
        query_tasks = mapped_X[..., self._task_feature].long()
        batch = torch.broadcast_shapes(noise_by_task.shape[:-1], query_tasks.shape[:-1])
        tasks = query_tasks.expand(*batch, query_tasks.shape[-1])
        return noise_by_task.expand(*batch, self.num_tasks).gather(-1, tasks)

    def _noise_at(self, mapped_X: Any) -> Any:
        noise = self.likelihood.noise
        if noise.numel() == self.num_tasks and (
            isinstance(self.likelihood, TorchHadamardLikelihood) or noise.numel() > 1
        ):
            return noise.reshape(-1)[mapped_X[..., self._task_feature].long()]
        return noise.expand(mapped_X.shape[:-1])

    def _log_prior(self) -> Any:
        result = self._train_X.new_zeros(())
        seen: set[int] = set()
        for module in (
            self.likelihood,
            self.mean_module,
            *self.covar_module.kernels,
            self.input_transform,
        ):
            if module is not None and id(module) not in seen:
                seen.add(id(module))
                result = result + module_log_prior(module, self.batch_shape)
        return result

    def eval(self) -> Any:
        if self._unobserved_task_indices.numel():
            kernel = self.covar_module.kernels[-1]
            with torch.no_grad():
                factor = kernel.covar_factor.index_select(0, self._observed_task_indices).mean(0)
                raw = kernel.raw_covar_factor_constraint.inverse_transform(factor)
                current = kernel.raw_covar_factor.index_select(0, self._unobserved_task_indices)
                wanted = raw.expand_as(current)
                if not torch.equal(current, wanted):
                    kernel.raw_covar_factor.index_copy_(0, self._unobserved_task_indices, wanted)
        return self.train(False)

    def _query_inputs(self, X: Any, output_indices: Any) -> tuple[Any, int, int]:
        if X.shape[-1] == self.num_non_task_features + 1:
            if output_indices is not None:
                raise ValueError("output_indices must be None when X includes task features.")
            return X, 1, X.shape[-2]
        if X.shape[-1] != self.num_non_task_features:
            raise ValueError(
                "X must contain the continuous features, optionally including task_feature."
            )
        tasks = self._output_tasks if output_indices is None else list(output_indices)
        if not tasks:
            raise ValueError("output_indices must be nonempty.")
        labels = torch.as_tensor(tasks, dtype=X.dtype, device=X.device)
        self._map_tasks(labels)
        q, m = X.shape[-2], len(tasks)
        continuous = X.unsqueeze(-3).expand(*X.shape[:-2], m, q, X.shape[-1])
        task_column = labels.reshape(*(1 for _ in X.shape[:-2]), m, 1, 1).expand(
            *X.shape[:-2], m, q, 1
        )
        full = torch.cat(
            (
                continuous[..., : self._task_feature],
                task_column,
                continuous[..., self._task_feature :],
            ),
            dim=-1,
        )
        return full.reshape(*X.shape[:-2], m * q, X.shape[-1] + 1), m, q

    def _task_prior_blocks(self, train: Any, query: Any) -> tuple[Any, ...]:
        """Latent prior blocks on transformed physical inputs, before task mapping."""
        if type(self).forward is TorchMultiTaskGP.forward:
            train, query = self._mapped_inputs(train), self._mapped_inputs(query)
            return (
                self._mean(train),
                self._covariance(train, train),
                self._covariance(train, query),
                self._mean(query),
                self._covariance(query, query),
            )
        training_prior = nn.Module.__call__(self, train)
        batch = torch.broadcast_shapes(train.shape[:-2], query.shape[:-2])
        joined = torch.cat(
            (train.expand(*batch, *train.shape[-2:]), query.expand(*batch, *query.shape[-2:])), -2
        )
        joint = nn.Module.__call__(self, joined)
        if any(
            not isinstance(getattr(prior, attribute, None), torch.Tensor)
            for prior in (training_prior, joint)
            for attribute in ("mean", "covariance_matrix")
        ):
            raise TypeError(
                "forward must return a Gaussian with Tensor mean and covariance_matrix."
            )
        n = train.shape[-2]
        covariance = joint.covariance_matrix
        return (
            training_prior.mean,
            training_prior.covariance_matrix,
            covariance[..., :n, n:],
            joint.mean[..., n:],
            covariance[..., n:, n:],
        )

    def _uses_likelihood_marginal(self) -> bool:
        from botorch_rs.models._torch_gaussian_likelihood import is_fixed_noise, uses_marginal

        return type(self.likelihood) is not TorchHadamardLikelihood and (
            uses_marginal(self.likelihood)
            or (getattr(self, "_is_custom_likelihood", False) and is_fixed_noise(self.likelihood))
        )

    def _likelihood_marginal(
        self, mean: Any, covariance: Any, X: Any, *, training: bool = False, **kwargs: Any
    ) -> Any:
        from botorch_rs.models._torch_gaussian_likelihood import gaussian_marginal

        mapped = self._mapped_inputs(X)
        if isinstance(self.likelihood, TorchHadamardLikelihood) and kwargs.get("params") is None:
            kwargs["params"] = (mapped,)
        return gaussian_marginal(
            self.likelihood, mean, covariance, mapped, training=training, **kwargs
        )

    def _conditioned_tensors(
        self, X: Any, observation_noise: bool = False, *, inputs_transformed: bool = False
    ) -> Any:
        """Condition in fitted target units, without output selection or untransform."""
        self.eval()
        train, targets, fixed = self._training_values()
        query = X if inputs_transformed else self.transform_inputs(X)
        self._check_task_transform(X, query)
        batch = torch.broadcast_shapes(train.shape[:-2], query.shape[:-2])
        query = query.expand(*batch, *query.shape[-2:])
        train_mean, train_cov, cross, query_mean, query_cov = self._task_prior_blocks(train, query)
        if self._uses_likelihood_marginal():
            observed_mean, observed_cov = self._likelihood_marginal(
                train_mean, train_cov, train, training=True
            )
            if self._noise_override is not None:
                diagonal = (observed_cov - train_cov).diagonal(dim1=-2, dim2=-1)
                noise = torch.where(
                    torch.isnan(self._noise_override), diagonal, self._noise_override
                )
                observed_mean, observed_cov = self._likelihood_marginal(
                    train_mean, train_cov, train, training=True, noise=noise
                )
            train_mean, train_cov = observed_mean, observed_cov
            train_noise = torch.zeros_like(train_mean)
        else:
            train_noise = self._training_noise(targets, fixed)
        chol, alpha = _rs.tensor_exact_gp_factor(targets, train_cov, train_mean, train_noise)
        from botorch_rs.models._torch_fantasy import prediction_factor

        chol, alpha = prediction_factor(chol, alpha)
        mean, covariance = _rs.tensor_exact_gp_predict(chol, alpha, cross, query_cov, query_mean)
        if observation_noise:
            if self._uses_likelihood_marginal():
                from botorch_rs.models._torch_gaussian_likelihood import is_fixed_noise

                query_noise = (
                    self._fixed_noise_at(self._mapped_inputs(query), self.likelihood.noise)
                    if is_fixed_noise(self.likelihood)
                    else None
                )
                mean, covariance = self._likelihood_marginal(
                    mean, covariance, query, noise=query_noise
                )
            else:
                mapped = self._mapped_inputs(query)
                noise = (
                    self._noise_at(mapped) if fixed is None else self._fixed_noise_at(mapped, fixed)
                )
                covariance = covariance + torch.diag_embed(noise)
        return mean, covariance

    def __call__(self, X: Any, *args: Any, **kwargs: Any) -> Any:
        if self.training:
            return nn.Module.__call__(self, X, *args, **kwargs)
        if args or kwargs:
            raise UnsupportedError(
                "Evaluation calls accept one input Tensor; use posterior for options."
            )
        if (
            not isinstance(X, torch.Tensor)
            or X.ndim < 2
            or X.shape[-1] != self._train_X.shape[-1]
            or X.dtype != self._train_X.dtype
            or X.device != self._train_X.device
        ):
            raise ValueError(
                "Evaluation inputs must include tasks and match the model's dtype/device."
            )
        mean, covariance = self._conditioned_tensors(X, inputs_transformed=True)
        return TorchGPyTorchPosterior(mean.unsqueeze(-1), covariance).distribution

    def posterior(
        self,
        X: Any,
        output_indices: Any = None,
        observation_noise: Any = False,
        posterior_transform: Any = None,
        **kwargs: Any,
    ) -> Any:
        if kwargs:
            raise UnsupportedError(f"Unsupported posterior arguments: {sorted(kwargs)}.")
        if (
            not isinstance(X, torch.Tensor)
            or X.ndim < 2
            or X.dtype != self._train_X.dtype
            or X.device != self._train_X.device
        ):
            raise ValueError(
                "X must be a Tensor with the model's dtype/device and shape (...,q,d)."
            )
        if isinstance(observation_noise, torch.Tensor):
            raise NotImplementedError("Tensor observation_noise is not supported by MultiTaskGP.")
        if not isinstance(observation_noise, bool):
            raise TypeError("observation_noise must be bool.")
        self.eval()
        self._fit_transform_graphs()
        full, m, _ = self._query_inputs(X, output_indices)
        transformed = self.transform_inputs(full)
        self._check_task_transform(full, transformed)
        mean, covariance = self._conditioned_tensors(
            transformed, observation_noise, inputs_transformed=True
        )
        q = transformed.shape[-2] // m
        mean = mean.reshape(*mean.shape[:-1], m, q).transpose(-1, -2)
        posterior = TorchGPyTorchPosterior(mean, covariance)
        if self.outcome_transform is not None:
            outcome = self.outcome_transform
            if getattr(outcome, "_requires_X", False):
                from botorch_rs.models.transforms._torch_standardize import (
                    untransform_multitask_outcome,
                )

                # Long-format inputs preserve physical task labels through the
                # input transform; task-specific outcome statistics need these
                # labels, including for a taskless public query expanded above.
                posterior = untransform_multitask_outcome(outcome, posterior, transformed)
                return (
                    posterior
                    if posterior_transform is None
                    else posterior_transform(posterior, X=X)
                )
            if m > 1:
                subset = getattr(outcome, "subset_output", None)
                if subset is None:
                    raise UnsupportedError(
                        "A multi-output outcome transform must support subset_output to broadcast its long-format statistics across tasks."
                    )
                # Repeat the one fitted long-format outcome rather than treating
                # the correlated tasks as independently standardized outputs.
                outcome = subset([0] * m)
            posterior = outcome.untransform_posterior(posterior, X=X)
        return posterior if posterior_transform is None else posterior_transform(posterior, X=X)

    def forward(self, X: Any) -> Any:
        """The latent prior, including in eval mode; __call__ conditions in eval."""
        if self.training:
            self._fit_transform_graphs()
        transform = self.input_transform if self.training else None
        if transform is not None:
            X_transformed = getattr(transform, "preprocess_transform", transform)(X)
            self._check_task_transform(X, X_transformed)
            X = X_transformed
        X = self._mapped_inputs(X)
        return TorchGPyTorchPosterior(
            self._mean(X).unsqueeze(-1), self._covariance(X, X)
        ).distribution

    def _exact_mll(
        self, output: Any = None, target: Any = None, *params: Any, **kwargs: Any
    ) -> Any:
        X, Y, fixed = self._training_values()
        if output is None:
            if type(self).forward is TorchMultiTaskGP.forward:
                mapped = self._mapped_inputs(X)
                mean, covariance = self._mean(mapped), self._covariance(mapped, mapped)
            else:
                prior = nn.Module.__call__(self, self.train_inputs[0] if self.training else X)
                mean, covariance = prior.mean, prior.covariance_matrix
        else:
            Y = Y if target is None else target
            mean, covariance = output.mean, output.covariance_matrix
        if self._uses_likelihood_marginal():
            prior_mean, prior_covariance = mean, covariance
            mean, covariance = self._likelihood_marginal(
                mean, covariance, X, training=True, params=params or None, kwargs=kwargs
            )
            if self._noise_override is not None and "noise" not in kwargs:
                diagonal = (covariance - prior_covariance).diagonal(dim1=-2, dim2=-1)
                override = torch.where(
                    torch.isnan(self._noise_override), diagonal, self._noise_override
                )
                mean, covariance = self._likelihood_marginal(
                    prior_mean,
                    prior_covariance,
                    X,
                    training=True,
                    params=params or None,
                    kwargs={**kwargs, "noise": override},
                )
            noise = torch.zeros_like(mean)
        else:
            if params or kwargs:
                raise TypeError(
                    "Additional likelihood arguments require a supplied Gaussian marginal."
                )
            noise = self._training_noise(Y, fixed)
        value = _rs.tensor_exact_mll(Y, covariance, mean, noise, normalize=False)
        return (value + self._log_prior()) / Y.shape[-1]

    def set_train_data(self, inputs: Any = None, targets: Any = None, strict: bool = True) -> None:
        X = self._train_X if inputs is None else inputs[0] if isinstance(inputs, tuple) else inputs
        Y = self._train_Y if targets is None else targets
        if Y.ndim == X.ndim - 1:
            Y = Y.unsqueeze(-1)
        self._validate_data(X, Y, self._train_Yvar)
        if X.ndim != 2 or Y.shape[-1] != 1 or X.shape[-1] != self._train_X.shape[-1]:
            raise ValueError(
                "New data must retain long-format unbatched inputs and one target column."
            )
        self._map_tasks(X[..., self._task_feature])
        if strict and (X.shape != self._train_X.shape or Y.shape != self._train_Y.shape):
            raise RuntimeError("Cannot modify training data shape with strict=True.")
        self._train_X, self._train_Y = X, Y
        self._transform_fit_X, self._transform_fit_Y = X, Y
        self._conditioned_train_X = None

    def condition_on_observations(self, X: Any, Y: Any, noise: Any = None, **kwargs: Any) -> Any:
        from botorch_rs.models._torch_gaussian_likelihood import (
            is_fixed_noise,
            is_gpytorch_gaussian,
        )

        if getattr(self, "_is_custom_likelihood", False) and is_fixed_noise(self.likelihood):
            from botorch_rs.models._torch_fully_bayesian_loader import _copy_model

            original_fixed = (
                self.likelihood.noise_covar.noise
                if is_gpytorch_gaussian(self.likelihood)
                else self.likelihood.noise
            )
            fixed = original_fixed.unsqueeze(-1)
            self._fit_transform_graphs()
            if self.outcome_transform is not None:
                fixed = self.outcome_transform.untransform(torch.zeros_like(fixed), fixed)[1]
            stage = _copy_model(self)
            stage._train_Yvar, stage._is_custom_likelihood = fixed, False
            conditioned = TorchMultiTaskGP.condition_on_observations(
                stage, X, Y, noise=noise, **kwargs
            )
            # The supplied likelihood's fixed noise and new conditioning noise
            # already use fitted target units. Preserve their actual graph
            # directly instead of retaining an unnecessary transform round trip.
            extra = noise.squeeze(-1) if noise.ndim > 1 and noise.shape[-1] == 1 else noise
            batch, count = conditioned.batch_shape, Y.shape[-2]
            conditioned.likelihood.noise = torch.cat(
                (
                    original_fixed.expand(*batch, original_fixed.shape[-1]),
                    extra.expand(*batch, count),
                ),
                dim=-1,
            )
            conditioned._is_custom_likelihood = True
            return conditioned
        if kwargs:
            raise UnsupportedError(f"Unsupported conditioning arguments: {sorted(kwargs)}.")
        if not isinstance(X, torch.Tensor) or X.ndim < 2 or X.shape[-1] != self._train_X.shape[-1]:
            raise ValueError("Multitask conditioning requires inputs including the task feature.")
        if not isinstance(Y, torch.Tensor):
            raise TypeError("Conditioning targets must be a Tensor.")
        if Y.ndim == X.ndim - 1:
            Y = Y.unsqueeze(-1)
        self._fit_transform_graphs()
        transformed = self.transform_inputs(X)
        self._check_task_transform(X, transformed)
        if Y.ndim < 2 or Y.shape[-1] != 1 or Y.shape[-2] != transformed.shape[-2]:
            raise ValueError("Multitask targets must have shape (...,q,1).")
        if any(t.dtype != self._train_X.dtype or t.device != self._train_X.device for t in (X, Y)):
            raise ValueError("Conditioning data must have the model dtype and device.")
        if not bool(torch.isfinite(X).all() & torch.isfinite(Y).all()):
            raise ValueError("Conditioning data must be finite.")
        self._map_tasks(transformed[..., self._task_feature])
        if noise is not None:
            if (
                not isinstance(noise, torch.Tensor)
                or noise.dtype != Y.dtype
                or noise.device != Y.device
            ):
                raise ValueError("Conditioning noise must be a Tensor with the model dtype/device.")
            if noise.shape == Y.shape[:-1]:
                noise = noise.unsqueeze(-1)
            if not bool(torch.isfinite(noise).all() & (noise >= 0).all()):
                raise ValueError("Conditioning noise must be finite and nonnegative.")
        elif self._train_Yvar is not None and not self._is_custom_likelihood:
            raise ValueError("Fixed-noise multitask conditioning requires noise.")
        batch = torch.broadcast_shapes(
            self.batch_shape, X.shape[:-2], Y.shape[:-2], () if noise is None else noise.shape[:-2]
        )
        memo = {} if self._prediction_cache is None else {id(self._prediction_cache): None}
        model = copy.deepcopy(self, memo)
        model._prediction_cache, model._transform_fit_versions = None, {}
        previous_inputs = self._training_values()[0]
        model._conditioned_train_X = torch.cat(
            (
                previous_inputs.expand(*batch, *previous_inputs.shape[-2:]),
                transformed.expand(*batch, *transformed.shape[-2:]),
            ),
            dim=-2,
        )
        model._train_X = torch.cat(
            (
                self._train_X.expand(*batch, *self._train_X.shape[-2:]),
                X.expand(*batch, *X.shape[-2:]),
            ),
            dim=-2,
        )
        model._train_Y = torch.cat(
            (
                self._train_Y.expand(*batch, *self._train_Y.shape[-2:]),
                Y.expand(*batch, *Y.shape[-2:]),
            ),
            dim=-2,
        )
        if self._train_Yvar is not None and not self._is_custom_likelihood:
            # As in BoTorch, supplied conditioning noise is already transformed.
            self._fit_transform_graphs()
            raw_noise = noise
            if self.outcome_transform is not None:
                raw_noise = self.outcome_transform.untransform(
                    torch.zeros_like(Y), noise, X=self.transform_inputs(X)
                )[1]
            model._train_Yvar = torch.cat(
                (
                    self._train_Yvar.expand(*batch, *self._train_Y.shape[-2:]),
                    raw_noise.expand(*batch, *Y.shape[-2:]),
                ),
                dim=-2,
            )
        elif noise is not None or self._noise_override is not None:
            previous = self._noise_override
            if previous is None:
                previous = torch.full_like(self._train_Y.squeeze(-1), torch.nan)
            extra = (
                torch.full_like(Y.squeeze(-1), torch.nan) if noise is None else noise.squeeze(-1)
            )
            model._noise_override = torch.cat(
                (previous.expand(*batch, previous.shape[-1]), extra.expand(*batch, Y.shape[-2])),
                dim=-1,
            )
        tasks = model._map_tasks(model._conditioned_train_X[..., self._task_feature]).long()
        observed = (tasks.reshape(-1, 1) == torch.arange(self.num_tasks, device=X.device)).any(0)
        model._observed_task_indices = observed.nonzero().squeeze(-1)
        model._unobserved_task_indices = (~observed).nonzero().squeeze(-1)
        if self._train_Yvar is not None and not self._is_custom_likelihood:
            model.likelihood.noise = model._training_values()[2].clone()
        return model.eval()

    def fantasize(self, X: Any, sampler: Any, observation_noise: Any = None, **kwargs: Any) -> Any:
        from botorch_rs.models._torch_fantasy import prediction_cache_gradients
        from botorch_rs.models.utils.assorted import fantasize

        propagate = kwargs.pop("propagate_grads", False)
        if kwargs:
            raise UnsupportedError(f"Unsupported fantasy arguments: {sorted(kwargs)}.")
        if isinstance(observation_noise, bool):
            raise TypeError("fantasize observation_noise must be None or a Tensor.")
        if observation_noise is not None:
            raise NotImplementedError("Tensor fantasy noise is not supported by MultiTaskGP.")
        if (
            not isinstance(X, torch.Tensor)
            or X.ndim < 2
            or X.shape[-1] != self.num_non_task_features + 1
        ):
            raise ValueError("Multitask fantasies require inputs including the task feature.")
        with fantasize():
            with prediction_cache_gradients(propagate):
                posterior = self.posterior(X, observation_noise=True)
            targets = sampler(posterior)
            noise = None
            if self._train_Yvar is not None and not self._is_custom_likelihood:
                _, _, fixed = self._training_values()
                noise = self._fixed_noise_at(
                    self._mapped_inputs(self.transform_inputs(X)), fixed
                ).unsqueeze(-1)
            elif getattr(self, "_is_custom_likelihood", False):
                from botorch_rs.models._torch_gaussian_likelihood import is_fixed_noise

                if is_fixed_noise(self.likelihood):
                    noise = self._fixed_noise_at(
                        self._mapped_inputs(self.transform_inputs(X)), self.likelihood.noise
                    ).unsqueeze(-1)
            return self.condition_on_observations(X, targets, noise=noise)

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
        d = train_X.shape[-1] - 1
        if d < 1 or not (-(d + 1) <= task_feature <= d):
            raise ValueError(f"task_feature must be between {-d - 1} and {d}.")
        self._task_feature = task_feature % (d + 1)
        self.num_non_task_features = d
        raw_tasks = train_X[..., self._task_feature]
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
        self.input_transform = build_input_transform(input_transform, d + 1, train_X)
        self.outcome_transform = build_outcome_transform(outcome_transform, 1, train_Y)
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
        if fixed_noise is not None:
            if likelihood is not None:
                raise UnsupportedError("Specify train_Yvar or a likelihood, not both.")
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
            elif isinstance(likelihood, nn.Module) and isinstance(
                getattr(likelihood, "noise", None), torch.Tensor
            ):
                gaussian = isinstance(likelihood, TorchGaussianLikelihood) or any(
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
                if cast(torch.Tensor, likelihood.noise).numel() not in (1, self.num_tasks):
                    raise UnsupportedError(
                        "A custom Gaussian likelihood must expose scalar or per-task Tensor noise."
                    )
                self.likelihood = cast(Any, likelihood).to(train_X)
            else:
                raise UnsupportedError(
                    "Torch MultiTaskGP requires Gaussian, Hadamard, fixed noise, or a Torch Gaussian likelihood exposing Tensor noise."
                )

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
        if not torch.equal(X[..., self._task_feature], self._train_X[..., self._task_feature]):
            raise UnsupportedError(
                "Input transforms must preserve categorical task labels; exclude task_feature."
            )
        return X, Y, noise

    @property
    def train_inputs(self) -> Any:
        return (self._train_X,)

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
        result = (
            fixed_noise
            if fixed_noise is not None
            else self._noise_at(self._mapped_inputs(self._train_X))
        )
        if self._noise_override is not None:
            result = torch.where(torch.isnan(self._noise_override), result, self._noise_override)
        return result

    def _fixed_noise_at(self, mapped_X: Any, fixed: Any) -> Any:
        train_tasks = self._mapped_inputs(self._train_X)[..., self._task_feature].long()
        mask = train_tasks.unsqueeze(-1) == torch.arange(self.num_tasks, device=mapped_X.device)
        counts = mask.sum(-2)
        noise_by_task = (fixed.unsqueeze(-1) * mask).sum(-2) / counts.clamp_min(1)
        query_tasks = mapped_X[..., self._task_feature].long()
        batch = torch.broadcast_shapes(noise_by_task.shape[:-1], query_tasks.shape[:-1])
        tasks = query_tasks.expand(*batch, query_tasks.shape[-1])
        available = counts.expand(*batch, self.num_tasks).gather(-1, tasks)
        if bool((available == 0).any()):
            raise UnsupportedError("Cannot infer fixed observation noise for an unobserved task.")
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
        full, m, q = self._query_inputs(X, output_indices)
        self.eval()
        raw_train, Y, fixed = self._training_values()
        transformed = self.transform_inputs(full)
        if not torch.equal(transformed[..., self._task_feature], full[..., self._task_feature]):
            raise UnsupportedError(
                "Input transforms must preserve categorical task labels; exclude task_feature."
            )
        train_X, query_X = self._mapped_inputs(raw_train), self._mapped_inputs(transformed)
        chol, alpha = _rs.tensor_exact_gp_factor(
            Y,
            self._covariance(train_X, train_X),
            self._mean(train_X),
            self._training_noise(Y, fixed),
        )
        query_noise = None
        if observation_noise:
            if fixed is None:
                query_noise = self._noise_at(query_X)
            else:
                query_noise = self._fixed_noise_at(query_X, fixed)
        mean, covariance = _rs.tensor_exact_gp_predict(
            chol,
            alpha,
            self._covariance(train_X, query_X),
            self._covariance(query_X, query_X),
            self._mean(query_X),
            observation_noise=query_noise,
        )
        mean = mean.reshape(*mean.shape[:-1], m, q).transpose(-1, -2)
        posterior = TorchGPyTorchPosterior(mean, covariance)
        if self.outcome_transform is not None:
            outcome = self.outcome_transform
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
        if not self.training:
            return self.posterior(X).distribution
        self._fit_transform_graphs()
        transform = self.input_transform
        if transform is not None:
            X_transformed = getattr(transform, "preprocess_transform", transform)(X)
            if not torch.equal(X_transformed[..., self._task_feature], X[..., self._task_feature]):
                raise UnsupportedError(
                    "Input transforms must preserve categorical task labels; exclude task_feature."
                )
            X = X_transformed
        X = self._mapped_inputs(X)
        return TorchGPyTorchPosterior(
            self._mean(X).unsqueeze(-1), self._covariance(X, X)
        ).distribution

    def _exact_mll(self, output: Any = None, target: Any = None) -> Any:
        X, Y, fixed = self._training_values()
        if output is None:
            X = self._mapped_inputs(X)
            mean, covariance = self._mean(X), self._covariance(X, X)
        else:
            Y = Y if target is None else target
            mean, covariance = output.mean, output.covariance_matrix
        value = _rs.tensor_exact_mll(Y, covariance, mean, self._training_noise(Y, fixed))
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

    def condition_on_observations(self, X: Any, Y: Any, noise: Any = None, **kwargs: Any) -> Any:
        if kwargs:
            raise UnsupportedError(f"Unsupported conditioning arguments: {sorted(kwargs)}.")
        if (
            not isinstance(X, torch.Tensor)
            or X.ndim < 2
            or X.shape[-1] != self.num_non_task_features + 1
        ):
            raise ValueError("Multitask conditioning requires inputs including the task feature.")
        if not isinstance(Y, torch.Tensor):
            raise TypeError("Conditioning targets must be a Tensor.")
        if Y.ndim == X.ndim - 1:
            Y = Y.unsqueeze(-1)
        if Y.ndim < 2 or Y.shape[-1] != 1 or Y.shape[-2] != X.shape[-2]:
            raise ValueError("Multitask targets must have shape (...,q,1).")
        if any(t.dtype != self._train_X.dtype or t.device != self._train_X.device for t in (X, Y)):
            raise ValueError("Conditioning data must have the model dtype and device.")
        if not bool(torch.isfinite(X).all() & torch.isfinite(Y).all()):
            raise ValueError("Conditioning data must be finite.")
        self._map_tasks(X[..., self._task_feature])
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
        elif self._train_Yvar is not None:
            raise ValueError("Fixed-noise multitask conditioning requires noise.")
        batch = torch.broadcast_shapes(
            self.batch_shape, X.shape[:-2], Y.shape[:-2], () if noise is None else noise.shape[:-2]
        )
        model = copy.copy(self)
        model._buffers = self._buffers.copy()
        model._modules = self._modules.copy()
        model._transform_fit_versions = self._transform_fit_versions.copy()
        model._prediction_cache = None
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
        if self._train_Yvar is not None:
            # As in BoTorch, supplied conditioning noise is already transformed.
            self._fit_transform_graphs()
            raw_noise = noise
            if self.outcome_transform is not None:
                raw_noise = self.outcome_transform.untransform(torch.zeros_like(Y), noise)[1]
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
        tasks = model._map_tasks(model._train_X[..., self._task_feature]).long()
        observed = (tasks.reshape(-1, 1) == torch.arange(self.num_tasks, device=X.device)).any(0)
        model._observed_task_indices = observed.nonzero().squeeze(-1)
        model._unobserved_task_indices = (~observed).nonzero().squeeze(-1)
        if self._train_Yvar is not None:
            model.likelihood = copy.copy(self.likelihood)
            model.likelihood._buffers = self.likelihood._buffers.copy()
            model.likelihood.noise = model._training_values()[2].detach().clone()
        return model.eval()

    def fantasize(self, X: Any, sampler: Any, observation_noise: Any = None, **kwargs: Any) -> Any:
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
        posterior = self.posterior(X, observation_noise=True)
        targets = sampler(posterior)
        if not propagate:
            targets = targets.detach()
        noise = None
        if self._train_Yvar is not None:
            _, _, fixed = self._training_values()
            noise = self._fixed_noise_at(
                self._mapped_inputs(self.transform_inputs(X)), fixed
            ).unsqueeze(-1)
        return self.condition_on_observations(X, targets, noise=noise)

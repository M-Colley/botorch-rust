"""Exact correlated input-by-task Gaussian laws on the Tensor device."""

from __future__ import annotations

import copy
import math
from typing import Any

import torch
from torch import Tensor, nn

from botorch_rs import _botorch_rs as _rs
from botorch_rs.constraints import Positive
from botorch_rs.exceptions.errors import UnsupportedError
from botorch_rs.models._torch_backend import TorchSingleTaskGP
from botorch_rs.models._torch_backend_modules import (
    TorchConstantMean,
    TorchConstraint,
    TorchPrior,
    _TorchModule,
    build_input_transform,
    build_kernel,
    build_outcome_transform,
    module_log_prior,
)
from botorch_rs.models.multitask import KroneckerMultiTaskGP
from botorch_rs.posteriors._torch_backend import TorchGPyTorchPosterior
from botorch_rs.posteriors.multitask import MultitaskGPPosterior


def tensor_kron(a: Tensor, b: Tensor) -> Tensor:
    value = a.unsqueeze(-1).unsqueeze(-3) * b.unsqueeze(-2).unsqueeze(-4)
    return value.reshape(*value.shape[:-4], a.shape[-2] * b.shape[-2], a.shape[-1] * b.shape[-1])


class TorchLKJCovariancePrior(nn.Module):
    """One LKJ correlation density plus the marginal-scale joint density."""

    eta: Tensor
    sd_prior: Any

    def __init__(self, n: int, eta: Any, sd_prior: Any, like: Tensor) -> None:
        super().__init__()
        self.n = n
        self.register_buffer("eta", torch.as_tensor(eta, dtype=like.dtype, device=like.device))
        self.sd_prior = (
            sd_prior.to(like) if isinstance(sd_prior, nn.Module) else TorchPrior(sd_prior, like)
        )

    def log_prob(self, value: Tensor) -> Tensor:
        standard = value.diagonal(dim1=-2, dim2=-1).sqrt()
        correlation = value / standard.unsqueeze(-1) / standard.unsqueeze(-2)
        factor = torch.linalg.cholesky(correlation)
        probability = torch.distributions.LKJCholesky(
            self.n, self.eta, validate_args=False
        ).log_prob(factor)
        scales = self.sd_prior.log_prob(standard)
        if scales.ndim == probability.ndim + 1:
            scales = scales.sum(-1)
        return probability + scales


class TorchIndexKernel(nn.Module):
    _anchor: Tensor

    def __init__(self, tasks: int, rank: int, batch: Any, prior: Any, like: Tensor) -> None:
        super().__init__()
        self.register_buffer("_anchor", like.new_empty(0), persistent=False)
        self.batch_shape = torch.Size(batch)
        self.num_tasks, self.rank = tasks, rank
        self.covar_factor = nn.Parameter(
            torch.randn(*batch, tasks, rank, device=like.device, dtype=like.dtype)
        )
        self.raw_var = nn.Parameter(
            torch.randn(*batch, tasks, device=like.device, dtype=like.dtype)
        )
        self.raw_var_constraint = TorchConstraint(Positive(), like)
        self.IndexKernelPrior = prior

    @property
    def var(self) -> Tensor:
        return self.raw_var_constraint.transform(self.raw_var)

    @var.setter
    def var(self, value: Any) -> None:
        value = torch.as_tensor(value, dtype=self.raw_var.dtype, device=self.raw_var.device)
        with torch.no_grad():
            self.raw_var.copy_(self.raw_var_constraint.inverse_transform(value))

    @property
    def covar_matrix(self) -> Tensor:
        return self.covar_factor @ self.covar_factor.transpose(-1, -2) + torch.diag_embed(self.var)

    def log_prior(self) -> Tensor:
        if self.IndexKernelPrior is None:
            return self._anchor.new_zeros(self.batch_shape)
        value = self.IndexKernelPrior.log_prob(self.covar_matrix)
        event_dims = value.ndim - len(self.batch_shape)
        return value.sum(tuple(range(-event_dims, 0))) if event_dims else value

    def named_priors(self) -> Any:
        if self.IndexKernelPrior is not None:
            # Match IndexKernel's matrix prior protocol: no inverse setter is
            # defined for an arbitrary low-rank-plus-diagonal covariance.
            yield (
                "IndexKernelPrior",
                self,
                self.IndexKernelPrior,
                lambda module: module.covar_matrix,
                None,
            )


class TorchMultitaskKernel(_TorchModule):
    def __init__(self, data: Any, tasks: TorchIndexKernel, like: Tensor) -> None:
        super().__init__(like)
        self.data_covar_module, self.task_covar_module = data, tasks
        self.batch_shape = tasks.batch_shape

    def forward(
        self, x1: Tensor, x2: Tensor | None = None, diag: bool = False, **kwargs: Any
    ) -> Tensor:
        if kwargs:
            raise UnsupportedError(f"Unsupported multitask-kernel options: {sorted(kwargs)}.")
        data = self.data_covar_module(x1, x2, diag=diag)
        data = data if isinstance(data, Tensor) else data.to_dense()
        task = self.task_covar_module.covar_matrix
        if diag:
            return (data.unsqueeze(-1) * task.diagonal(dim1=-2, dim2=-1).unsqueeze(-2)).flatten(-2)
        return tensor_kron(data, task)

    def log_prior(self) -> Tensor:
        return (
            module_log_prior(self.data_covar_module, self.batch_shape)
            + self.task_covar_module.log_prior()
        )


class TorchMultitaskMean(nn.Module):
    def __init__(self, tasks: int, batch: Any, like: Tensor) -> None:
        from botorch_rs.means import ConstantMean

        super().__init__()
        self.batch_shape = torch.Size(batch)
        self.base_means = nn.ModuleList(
            [TorchConstantMean(ConstantMean(batch_shape=batch), like) for _ in range(tasks)]
        )

    def forward(self, X: Tensor) -> Tensor:
        return torch.stack([mean(X) for mean in self.base_means], dim=-1)


class TorchMultitaskGaussianLikelihood(_TorchModule):
    def __init__(self, tasks: int, rank: int, batch: Any, like: Tensor, source: Any = None) -> None:
        from botorch_rs.constraints import GreaterThan
        from botorch_rs.priors import LogNormalPrior

        super().__init__(like)
        self.num_tasks, self.rank = tasks, rank
        if source is not None:
            batch = getattr(source, "batch_shape", ())
        self.batch_shape = torch.Size(batch)
        self.has_global_noise = getattr(source, "has_global_noise", True)
        self.has_task_noise = getattr(source, "has_task_noise", True)
        if not 0 <= rank <= tasks or not (self.has_global_noise or self.has_task_noise):
            raise ValueError("Invalid multitask likelihood rank/noise configuration.")
        prior = LogNormalPrior(-4.0, 1.0) if source is None else source.noise_prior
        constraint = (
            GreaterThan(1e-4, transform=None, initial_value=math.exp(-5))
            if source is None
            else source.noise_constraint
        )
        self.raw_noise_constraint = TorchConstraint(constraint, like)
        initial = self.raw_noise_constraint.initial_value
        if self.has_global_noise:
            self.raw_noise = nn.Parameter(
                like.new_full((*batch, 1), 0.0 if initial is None else initial)
            )
            self._register_prior("noise_prior", prior, "noise", 1)
        if self.has_task_noise and rank == 0:
            self.raw_task_noises = nn.Parameter(
                like.new_full((*batch, tasks), 0.0 if initial is None else initial)
            )
            self.raw_task_noises_constraint = TorchConstraint(constraint, like)
            self._register_prior("raw_task_noises_prior", prior, "task_noises", 1)
        elif self.has_task_noise:
            original = getattr(source, "_local_factor", None)
            factor = (
                torch.randn(*batch, tasks, rank, device=like.device, dtype=like.dtype)
                if original is None
                else torch.as_tensor(original, dtype=like.dtype, device=like.device).reshape(
                    *batch, tasks, rank
                )
            )
            self.task_noise_covar_factor = nn.Parameter(factor)
        if source is not None and getattr(source, "_model", None) is not None:
            with torch.no_grad():
                if self.has_global_noise:
                    self.raw_noise.copy_(
                        torch.as_tensor(source.raw_noise, dtype=like.dtype, device=like.device)
                    )
                if self.has_task_noise and self.rank == 0:
                    self.raw_task_noises.copy_(
                        torch.as_tensor(
                            source.raw_task_noises, dtype=like.dtype, device=like.device
                        )
                    )
                elif self.has_task_noise:
                    self.task_noise_covar_factor.copy_(
                        torch.as_tensor(
                            source.task_noise_covar_factor, dtype=like.dtype, device=like.device
                        )
                    )

    @property
    def noise(self) -> Tensor:
        return self.raw_noise_constraint.transform(self.raw_noise)

    @noise.setter
    def noise(self, value: Any) -> None:
        self._assign("raw_noise", value, self.raw_noise_constraint)

    @property
    def task_noises(self) -> Tensor:
        return self.raw_task_noises_constraint.transform(self.raw_task_noises)

    @task_noises.setter
    def task_noises(self, value: Any) -> None:
        self._assign("raw_task_noises", value, self.raw_task_noises_constraint)

    @property
    def task_noise_covar(self) -> Tensor:
        if self.rank == 0:
            return torch.diag_embed(self.task_noises)
        return self.task_noise_covar_factor @ self.task_noise_covar_factor.transpose(-1, -2)

    def noise_matrix(self) -> Tensor:
        value = self._anchor.new_zeros(*self.batch_shape, self.num_tasks, self.num_tasks)
        if self.has_task_noise:
            value = value + self.task_noise_covar
        if self.has_global_noise:
            value = value + self.noise.unsqueeze(-1) * torch.eye(
                self.num_tasks, device=value.device, dtype=value.dtype
            )
        return value

    def forward(self, distribution: Any, *inputs: Any, **kwargs: Any) -> Any:
        if kwargs:
            raise UnsupportedError(f"Unsupported multitask likelihood options: {sorted(kwargs)}.")
        posterior = distribution._p if hasattr(distribution, "_p") else None
        if posterior is None:
            raise TypeError("Use the Tensor Gaussian distribution returned by this model.")
        q = posterior.mean.shape[-2]
        # Gaussian posterior covariance uses output-major ordering.
        noise = tensor_kron(
            self.noise_matrix(), torch.eye(q, device=posterior.device, dtype=posterior.dtype)
        )
        return TorchGPyTorchPosterior(
            posterior.mean, posterior._covariance_tensor + noise
        ).distribution


class TorchKroneckerPosterior(TorchGPyTorchPosterior, MultitaskGPPosterior):
    """Correlated Matheron samples with the public structured normal layout."""

    _uses_matheron = True

    def __init__(
        self,
        mean: Tensor,
        covariance: Tensor,
        joint: Tensor,
        train: Tensor,
        cross: Tensor,
        residual: Tensor,
        query_mean: Tensor,
        noise: Tensor,
        test_noise: Tensor | None,
        num_train: int,
        tasks: int,
        location: Any = 0.0,
        scale: Any = 1.0,
    ) -> None:
        super().__init__(mean, covariance)
        from botorch_rs.models._torch_fantasy import prediction_factor

        previous_train = train
        train, residual = prediction_factor(train, residual)
        noise, _ = prediction_factor(noise, noise)
        if train is not previous_train:
            count = num_train * tasks
            joint = torch.cat(
                (
                    torch.cat(
                        (joint[..., :count, :count].detach(), joint[..., :count, count:]), -1
                    ),
                    joint[..., count:, :],
                ),
                -2,
            )
        self.joint_covariance_matrix, self.train_train_covar = joint, train
        self.test_train_covar, self.train_diff = cross, residual
        self.test_mean = query_mean
        self.train_noise, self.test_noise = noise, test_noise
        self.num_train, self.num_tasks = num_train, tasks
        self.observation_noise = test_noise is not None
        self._location, self._scale = location, scale

    @property
    def base_sample_shape(self) -> Any:
        size = self.joint_covariance_matrix.shape[-1] + self.train_noise.shape[-1]
        if self.test_noise is not None:
            size += self.test_noise.shape[-1]
        return self._batch + torch.Size([size])

    @property
    def batch_range(self) -> tuple[int, int]:
        return (0, -1)

    def rsample_from_base_samples(
        self, sample_shape: Any, base_samples: Any, train_diff: Any = None
    ) -> Tensor:
        shape = torch.Size(sample_shape)
        if base_samples.shape[: len(shape)] != shape:
            raise RuntimeError("sample_shape disagrees with base_samples.shape.")
        base = base_samples.expand(shape + self.base_sample_shape)
        joint_size = self.joint_covariance_matrix.shape[-1]
        count = self.num_train * self.num_tasks
        joint_factor = _rs.tensor_psd_safe_cholesky(self.joint_covariance_matrix)
        noise_factor = _rs.tensor_psd_safe_cholesky(self.train_noise)
        prior = (joint_factor @ base[..., :joint_size].unsqueeze(-1)).squeeze(-1)
        noise = (noise_factor @ base[..., joint_size : joint_size + count].unsqueeze(-1)).squeeze(
            -1
        )
        residual = self.train_diff if train_diff is None else train_diff
        residual = residual.reshape(*residual.shape[:-2], -1) - prior[..., :count] - noise
        factor = torch.linalg.cholesky(self.train_train_covar + self.train_noise)
        update = self.test_train_covar @ torch.cholesky_solve(residual.unsqueeze(-1), factor)
        samples = prior[..., count:] + update.squeeze(-1) + self.test_mean.flatten(-2)
        if self.test_noise is not None:
            factor = _rs.tensor_psd_safe_cholesky(self.test_noise)
            samples = samples + (
                factor @ base[..., -self.test_noise.shape[-1] :].unsqueeze(-1)
            ).squeeze(-1)
        return samples.reshape(*samples.shape[:-1], self._q, self._m) * self._scale + self._location


class TorchKroneckerMultiTaskGP(nn.Module, KroneckerMultiTaskGP):
    _tensor_backend = "torch"
    _supports_cache_root = False
    _train_X: Any
    _train_Y: Any
    _train_Yvar: Any
    _transform_fit_X: Any
    _transform_fit_Y: Any
    likelihood: Any
    mean_module: Any
    covar_module: Any
    input_transform: Any
    outcome_transform: Any
    _validate_data = staticmethod(TorchSingleTaskGP._validate_data)
    _fit_transform_graphs: Any = TorchSingleTaskGP._fit_transform_graphs
    transform_inputs: Any = TorchSingleTaskGP.transform_inputs
    initialize: Any = TorchSingleTaskGP.initialize
    __deepcopy__: Any = TorchSingleTaskGP.__deepcopy__
    _sync_torch_parameters: Any = TorchSingleTaskGP._sync_torch_parameters

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        likelihood: Any = None,
        data_covar_module: Any = None,
        task_covar_prior: Any = None,
        rank: int | None = None,
        outcome_transform: Any = None,
        input_transform: Any = None,
        *,
        backend: str = "torch",
        **kwargs: Any,
    ) -> None:
        from botorch_rs.priors import SmoothedBoxPrior

        nn.Module.__init__(self)
        self._validate_data(train_X, train_Y)
        if set(kwargs) - {"eta", "sd_prior", "likelihood_rank"}:
            raise UnsupportedError(f"Unsupported Kronecker options: {sorted(kwargs)}.")
        tasks = train_Y.shape[-1]
        if tasks < 2:
            raise ValueError("KroneckerMultiTaskGP requires at least two tasks.")
        rank = tasks if rank is None else rank
        if not isinstance(rank, int) or not 0 <= rank <= tasks:
            raise ValueError("rank must lie between zero and num_tasks.")
        self._num_outputs = tasks
        self._prediction_cache = None
        self._transform_fit_versions: dict[int, Any] = {}
        for name, value in (
            ("_train_X", train_X),
            ("_train_Y", train_Y),
            ("_train_Yvar", None),
            ("_transform_fit_X", train_X),
            ("_transform_fit_Y", train_Y),
        ):
            self.register_buffer(name, value)
        self.input_transform = build_input_transform(input_transform, train_X.shape[-1], train_X)
        with torch.no_grad():
            outcome_inputs = self.transform_inputs(train_X)
        self.outcome_transform = build_outcome_transform(
            outcome_transform, tasks, train_Y, train_X=outcome_inputs
        )
        batch = self.batch_shape
        if isinstance(likelihood, nn.Module):
            self.likelihood = likelihood.to(train_X)
            if not hasattr(likelihood, "noise_matrix") and (
                not hasattr(likelihood, "_shaped_noise_covar")
                or getattr(likelihood, "num_tasks", None) != tasks
            ):
                raise UnsupportedError(
                    "Custom structured likelihoods must expose their Gaussian task-noise covariance."
                )
        else:
            from botorch_rs.likelihoods import MultitaskGaussianLikelihood

            if likelihood is not None and (
                not isinstance(likelihood, MultitaskGaussianLikelihood)
                or likelihood.num_tasks != tasks
            ):
                raise UnsupportedError(
                    "A structured model requires a matching MultitaskGaussianLikelihood."
                )
            self.likelihood = TorchMultitaskGaussianLikelihood(
                tasks,
                getattr(likelihood, "rank", kwargs.get("likelihood_rank", 0)),
                batch,
                train_X,
                likelihood,
            )
        self.mean_module = TorchMultitaskMean(tasks, batch, train_X)
        prior: Any
        if task_covar_prior is None:
            prior = TorchLKJCovariancePrior(
                tasks,
                kwargs.get("eta", 1.5),
                kwargs.get("sd_prior", SmoothedBoxPrior(math.exp(-6), math.exp(1.25), 0.05)),
                train_X,
            )
        elif type(task_covar_prior).__name__ == "LKJCovariancePrior":
            prior = TorchLKJCovariancePrior(
                tasks, task_covar_prior.eta, task_covar_prior.sd_prior, train_X
            )
        elif isinstance(task_covar_prior, nn.Module):
            prior = task_covar_prior.to(train_X)
        else:
            prior = TorchPrior(task_covar_prior, train_X)
        self.covar_module = TorchMultitaskKernel(
            build_kernel(data_covar_module, train_X.shape[-1], batch, train_X),
            TorchIndexKernel(tasks, rank, batch, prior, train_X),
            train_X,
        )

    @property
    def batch_shape(self) -> Any:
        return self._train_X.shape[:-2]

    @property
    def _aug_batch_shape(self) -> Any:
        return self.batch_shape

    @property
    def num_outputs(self) -> int:
        return self._num_outputs

    @property
    def train_inputs(self) -> Any:
        return (self._train_X,)

    @property
    def train_targets(self) -> Any:
        return self._training_values()[1]

    def _training_values(self) -> Any:
        self._fit_transform_graphs()
        X = (
            self._train_X
            if self.input_transform is None
            else getattr(self.input_transform, "preprocess_transform", self.input_transform)(
                self._train_X
            )
        )
        Y = (
            self._train_Y
            if self.outcome_transform is None
            else self.outcome_transform(self._train_Y, X=X)[0]
        )
        return X, Y

    def _noise_matrix(self) -> Tensor:
        method = getattr(self.likelihood, "noise_matrix", None)
        if callable(method):
            return method()
        # GPyTorch's lazy noise operator can use a first-order custom backward.
        # Construct the same covariance from its actual live Tensor parameters
        # so the likelihood factor and raw-noise Hessians remain connected.
        like = self._train_X
        value = like.new_zeros(self.num_outputs, self.num_outputs)
        if self.likelihood.has_task_noise:
            if self.likelihood.rank == 0:
                value = value + torch.diag_embed(self.likelihood.task_noises)
            else:
                factor = self.likelihood.task_noise_covar_factor
                value = value + factor @ factor.transpose(-1, -2)
        if self.likelihood.has_global_noise:
            value = value + self.likelihood.noise.unsqueeze(-1) * torch.eye(
                self.num_outputs, dtype=like.dtype, device=like.device
            )
        return value

    def _log_prior(self) -> Tensor:
        result = self._train_X.new_zeros(self.batch_shape)
        for module in (self.likelihood, self.covar_module, self.mean_module):
            result = result + module_log_prior(module, self.batch_shape)
        return result

    def _exact_mll(
        self, output: Any = None, target: Any = None, *params: Any, **kwargs: Any
    ) -> Tensor:
        from botorch_rs.models._torch_structured_likelihood import (
            conditioned_observed_law,
            multitask_observed_law,
        )

        X, Y = self._training_values()
        mean = self.mean_module(X)
        covariance = self.covar_module(X, X)
        if output is None and type(self).forward is not TorchKroneckerMultiTaskGP.forward:
            output = nn.Module.__call__(self, self._train_X if self.training else X)
        if output is not None:
            mean, covariance = self._point_major_law(output)
            Y = Y if target is None else target
        latent_covariance = covariance
        mean, covariance = multitask_observed_law(
            self, mean, covariance, X, params=params, kwargs=kwargs
        )
        covariance = conditioned_observed_law(self, mean, latent_covariance, covariance)
        value = _rs.tensor_exact_mll(
            Y.flatten(-2), covariance, mean.flatten(-2), torch.zeros_like(Y.flatten(-2))
        )
        return (value + self._log_prior()) / (Y.shape[-2] * Y.shape[-1])

    def _point_major_law(self, distribution: Any) -> tuple[Tensor, Tensor]:
        mean, covariance = distribution.mean, distribution.covariance_matrix
        if not isinstance(mean, Tensor) or not isinstance(covariance, Tensor):
            raise TypeError(
                "forward must return a Gaussian with Tensor mean and covariance_matrix."
            )
        if self.num_outputs == 1 and len(getattr(distribution, "event_shape", ())) == 1:
            mean = mean.unsqueeze(-1)
        if mean.ndim < 2 or mean.shape[-1] != self.num_outputs:
            raise ValueError("The Gaussian prior must retain the point and task event axes.")
        if not getattr(distribution, "_interleaved", False):
            q, m = mean.shape[-2:]
            index = torch.arange(q * m, device=mean.device).reshape(m, q).T.flatten()
            covariance = covariance.index_select(-2, index).index_select(-1, index)
        return mean, covariance

    def posterior(
        self,
        X: Tensor,
        output_indices: Any = None,
        observation_noise: Any = False,
        posterior_transform: Any = None,
        **kwargs: Any,
    ) -> Any:
        return self._posterior(X, output_indices, observation_noise, posterior_transform, **kwargs)

    def _posterior(
        self,
        X: Tensor,
        output_indices: Any = None,
        observation_noise: Any = False,
        posterior_transform: Any = None,
        *,
        inputs_transformed: bool = False,
        untransform: bool = True,
        **kwargs: Any,
    ) -> Any:
        from botorch_rs.models._torch_structured_likelihood import (
            conditioned_observed_law,
            multitask_observed_law,
        )

        if kwargs or not isinstance(observation_noise, bool):
            raise UnsupportedError(
                "Structured Tensor prediction requires bool observation_noise and no extra options."
            )
        if (
            not isinstance(X, Tensor)
            or X.ndim < 2
            or X.shape[-1] != self._train_X.shape[-1]
            or X.device != self._train_X.device
            or X.dtype != self._train_X.dtype
        ):
            raise ValueError("Query tensors must match the model feature width, dtype and device.")
        self.eval()
        train, Y = self._training_values()
        query = X if inputs_transformed else self.transform_inputs(X)
        batch = torch.broadcast_shapes(train.shape[:-2], query.shape[:-2])
        n, q, m = train.shape[-2], query.shape[-2], self.num_outputs
        train, query = (
            train.expand(*batch, n, train.shape[-1]),
            query.expand(*batch, q, query.shape[-1]),
        )
        full = torch.cat((train, query), -2)
        if type(self).forward is TorchKroneckerMultiTaskGP.forward:
            ktt, ktq, kqq = (
                self.covar_module(train, train),
                self.covar_module(train, query),
                self.covar_module(query, query),
            )
            train_mean, query_mean = self.mean_module(train), self.mean_module(query)
            joint = self.covar_module(full, full)
        else:
            train_mean, ktt = self._point_major_law(nn.Module.__call__(self, train))
            full_mean, joint = self._point_major_law(nn.Module.__call__(self, full))
            count = n * m
            ktq, kqq = joint[..., :count, count:], joint[..., count:, count:]
            query_mean = full_mean[..., n:, :]
        observed_mean, observed_covariance = multitask_observed_law(
            self, train_mean, ktt, train, training=True
        )
        observed_covariance = conditioned_observed_law(
            self, observed_mean, ktt, observed_covariance
        )
        noise = observed_covariance - ktt
        train_mean = observed_mean
        test_noise = None
        if observation_noise:
            query_mean, observed_covariance = multitask_observed_law(self, query_mean, kqq, query)
            test_noise = observed_covariance - kqq
        factor, alpha = _rs.tensor_exact_gp_factor(
            Y.flatten(-2), ktt + noise, train_mean.flatten(-2), torch.zeros_like(Y.flatten(-2))
        )
        from botorch_rs.models._torch_fantasy import prediction_factor

        factor, alpha = prediction_factor(factor, alpha)
        mean, covariance = _rs.tensor_exact_gp_predict(
            factor,
            alpha,
            ktq,
            kqq + (0 if test_noise is None else test_noise),
            query_mean.flatten(-2),
        )
        mean = mean.reshape(*batch, q, m)
        permutation = torch.arange(q * m, device=X.device).reshape(q, m).transpose(-1, -2).flatten()
        covariance = covariance.index_select(-2, permutation).index_select(-1, permutation)
        location, scale = 0.0, 1.0
        from botorch_rs.models._torch_structured_outcomes import is_scalar_standardize

        standardize = is_scalar_standardize(self.outcome_transform)
        if untransform and standardize:
            location, scale = (
                self.outcome_transform.means.clone(),
                self.outcome_transform.stdvs.clone(),
            )
            mean = mean * scale + location
            expanded = scale.movedim(-1, -2).expand(*scale.shape[:-2], m, q).flatten(-2)
            covariance = covariance * expanded.unsqueeze(-1) * expanded.unsqueeze(-2)
        posterior: Any = TorchKroneckerPosterior(
            mean,
            covariance,
            joint,
            ktt,
            ktq.transpose(-1, -2),
            Y - train_mean,
            query_mean.clone(),
            noise,
            test_noise,
            n,
            m,
            location,
            scale,
        )
        if untransform and self.outcome_transform is not None and not standardize:
            from botorch_rs.models._torch_structured_outcomes import TorchStructuredOutcomePosterior

            transform = self.outcome_transform
            if output_indices is not None:
                posterior = posterior.subset_output(output_indices)
                transform = transform.subset_output(output_indices)
                output_indices = None
            posterior = TorchStructuredOutcomePosterior(
                posterior, transform, query, scalar_events=False
            )
        if output_indices is not None:
            posterior = posterior.subset_output(output_indices)
        return posterior if posterior_transform is None else posterior_transform(posterior, X=X)

    def __call__(self, X: Tensor, *args: Any, **kwargs: Any) -> Any:
        if not self.training:
            if args or kwargs:
                raise TypeError("KroneckerMultiTaskGP evaluation accepts one input Tensor.")
            return self._posterior(X, inputs_transformed=True, untransform=False).distribution
        return nn.Module.__call__(self, X, *args, **kwargs)

    def forward(self, X: Tensor) -> Any:
        if self.training:
            self._fit_transform_graphs()
            X = self.transform_inputs(X)
        mean = self.mean_module(X)
        covariance = self.covar_module(X, X)
        q, m = X.shape[-2], self.num_outputs
        indices = torch.arange(q * m, device=X.device).reshape(q, m).transpose(-1, -2).flatten()
        return TorchGPyTorchPosterior(
            mean, covariance.index_select(-1, indices).index_select(-2, indices)
        ).distribution

    def condition_on_observations(
        self, X: Tensor, Y: Tensor, noise: Any = None, **kwargs: Any
    ) -> Any:
        if kwargs:
            raise UnsupportedError(
                "Structured conditioning uses the model's configured Gaussian task noise."
            )
        if (
            X.shape[-1] != self._train_X.shape[-1]
            or Y.shape[-1] != self.num_outputs
            or X.device != self._train_X.device
            or X.dtype != self._train_X.dtype
        ):
            raise ValueError(
                "Conditioning data must match the model inputs, outputs, dtype and device."
            )
        batch = torch.broadcast_shapes(self.batch_shape, X.shape[:-2], Y.shape[:-2])
        self._validate_data(X.expand(*batch, *X.shape[-2:]), Y.expand(*batch, *Y.shape[-2:]))
        self._fit_transform_graphs()
        model = copy.deepcopy(self)
        model._transform_fit_versions = {}
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
        from botorch_rs.models._torch_structured_likelihood import condition_likelihood

        self.eval()
        condition_likelihood(self, model, noise, Y, X=X)
        return model.eval()

    def fantasize(
        self, X: Tensor, sampler: Any, observation_noise: Any = None, **kwargs: Any
    ) -> Any:
        propagate = kwargs.pop("propagate_grads", False)
        if kwargs or observation_noise is not None:
            raise UnsupportedError("Structured fantasies currently use the configured task noise.")
        from botorch_rs.models._torch_fantasy import prediction_cache_gradients
        from botorch_rs.models.utils.assorted import fantasize

        with fantasize():
            with prediction_cache_gradients(propagate):
                posterior = self.posterior(X, observation_noise=True)
            Y = sampler(posterior)
            return self.condition_on_observations(X, Y)

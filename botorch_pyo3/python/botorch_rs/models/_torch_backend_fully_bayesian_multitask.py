"""Fully Bayesian latent-feature multitask GPs on the training Tensor device."""

from __future__ import annotations

from typing import Any, cast

import torch
from torch import Tensor, nn

from botorch_rs import _botorch_rs as _rs
from botorch_rs.constraints import GreaterThan
from botorch_rs.exceptions.errors import UnsupportedError
from botorch_rs.kernels import LinearKernel, MaternKernel, ScaleKernel
from botorch_rs.likelihoods import GaussianLikelihood
from botorch_rs.means import ConstantMean
from botorch_rs.models._torch_backend import TorchSingleTaskGP
from botorch_rs.models._torch_backend_fully_bayesian import TorchGaussianMixturePosterior
from botorch_rs.models._torch_backend_kronecker import TorchIndexKernel
from botorch_rs.models._torch_backend_modules import (
    TorchCompositeKernel,
    TorchInputChain,
    build_input_transform,
    build_kernel,
    build_likelihood,
    build_mean,
)
from botorch_rs.models._torch_backend_multitask import TorchMultiTaskGP, TorchTaskMean
from botorch_rs.models.fully_bayesian_multitask import (
    LatentFeatureMultiTaskPyroMixin,
    MultitaskSaasPyroModel,
    SaasFullyBayesianMultiTaskGP,
)
from botorch_rs.models.transforms.input import Warp
from botorch_rs.posteriors._torch_backend import TorchGPyTorchPosterior


class _SampleIndexKernel(TorchIndexKernel):
    active_dims: Tensor

    def __init__(self, tasks: int, batch: Any, feature: int, like: Tensor) -> None:
        with torch.random.fork_rng(devices=[] if like.device.type == "cpu" else [like.device]):
            super().__init__(tasks, tasks, batch, None, like)
        self.register_buffer("active_dims", torch.tensor([feature], device=like.device))

    def forward(
        self,
        x1: Tensor,
        x2: Tensor | None = None,
        diag: bool = False,
        last_dim_is_batch: bool = False,
    ) -> Tensor:
        if last_dim_is_batch:
            raise UnsupportedError("Task kernels do not support last_dim_is_batch.")
        x2 = x1 if x2 is None else x2
        left, right = (x.index_select(-1, self.active_dims).squeeze(-1).long() for x in (x1, x2))
        matrix = self.covar_matrix
        batch = torch.broadcast_shapes(matrix.shape[:-2], left.shape[:-1], right.shape[:-1])
        matrix = matrix.expand(*batch, self.num_tasks, self.num_tasks)
        left, right = left.expand(*batch, left.shape[-1]), right.expand(*batch, right.shape[-1])
        rows = matrix.gather(-2, left.unsqueeze(-1).expand(*batch, left.shape[-1], self.num_tasks))
        if diag:
            if left.shape[-1] != right.shape[-1]:
                raise ValueError("Paired task diagonals require equally many points.")
            return rows.gather(-1, right.unsqueeze(-1)).squeeze(-1)
        return rows.gather(-1, right.unsqueeze(-2).expand(*batch, left.shape[-1], right.shape[-1]))


class _CenterFeatures(nn.Module):
    indices: Tensor

    def __init__(self, indices: Tensor):
        super().__init__()
        self.register_buffer("indices", indices)

    def forward(self, X: Tensor) -> Tensor:
        return X.index_copy(-1, self.indices, X.index_select(-1, self.indices) - 0.5)


class TorchSaasFullyBayesianMultiTaskGP(TorchMultiTaskGP, SaasFullyBayesianMultiTaskGP):
    """A device-resident batch of correlated latent-feature GP components."""

    _mcmc_count: int | None
    _fb_user_input_transform: Any
    pyro_model: Any

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        task_feature: int,
        train_Yvar: Tensor | None = None,
        output_tasks: Any = None,
        rank: int | None = None,
        all_tasks: Any = None,
        outcome_transform: Any = None,
        input_transform: Any = None,
        pyro_model: Any = None,
        validate_task_values: bool = True,
        *,
        backend: str = "torch",
    ) -> None:
        self._mcmc_count, self._initializing = None, True
        TorchMultiTaskGP.__init__(
            self,
            train_X,
            train_Y,
            task_feature,
            train_Yvar,
            output_tasks=output_tasks,
            rank=rank or None,
            all_tasks=all_tasks,
            outcome_transform=outcome_transform,
            input_transform=input_transform,
            task_covar_prior=None,
            validate_task_values=validate_task_values,
        )
        if self.outcome_transform is not None and not getattr(
            self.outcome_transform, "_is_linear", False
        ):
            raise UnsupportedError(
                "A fully Bayesian Gaussian mixture requires a linear outcome transform."
            )
        object.__setattr__(self, "_fb_user_input_transform", self.input_transform)
        pyro_model = MultitaskSaasPyroModel() if pyro_model is None else pyro_model
        if not isinstance(pyro_model, LatentFeatureMultiTaskPyroMixin):
            raise UnsupportedError(
                "Torch multitask inference requires a latent-feature Pyro mixin."
            )
        pyro_model = cast(Any, pyro_model)
        self.pyro_model = pyro_model
        pyro_model.train_X, pyro_model.train_Y, pyro_model.train_Yvar = train_X, train_Y, train_Yvar
        pyro_model._prior_mode = False
        pyro_model.ard_num_dims = self.num_non_task_features
        pyro_model.task_feature, pyro_model.num_tasks, pyro_model.task_rank = (
            self._task_feature,
            self.num_tasks,
            self._rank,
        )
        if pyro_model.use_input_warping:
            pyro_model._warp_indices()
        self.mean_module, self.covar_module, self.likelihood = None, None, None
        self._initializing = False
        from botorch_rs.models._numpyro_fitting import requires_numpyro_inference
        from botorch_rs.models._torch_backend_mcmc import TorchMCMCPotential

        if not requires_numpyro_inference(self):
            TorchMCMCPotential.from_model(self)  # Validate built-in probabilistic hooks eagerly.

    def _check_if_fitted(self) -> None:
        if self.covar_module is None:
            raise RuntimeError(
                "Model has not been fitted. Call load_mcmc_samples or fit_fully_bayesian_model_nuts."
            )

    @property
    def num_mcmc_samples(self) -> int:
        self._check_if_fitted()
        assert self._mcmc_count is not None
        return self._mcmc_count

    @property
    def batch_shape(self) -> torch.Size:
        return torch.Size() if self._initializing else torch.Size([self.num_mcmc_samples])

    @property
    def _input_batch_shape(self) -> torch.Size:
        return torch.Size()

    def train(self, mode: bool = True, reset: bool = True) -> Any:
        nn.Module.train(self, mode)
        if mode and reset:
            self.mean_module, self.covar_module, self.likelihood = None, None, None
            self._mcmc_count = None
            self.input_transform = self._fb_user_input_transform
        return self

    def eval(self) -> Any:
        self._check_if_fitted()
        return self.train(False)

    def _training_values(self, clamp_noise: bool = True) -> tuple[Any, Any, Any]:
        X, Y, noise = TorchSingleTaskGP._training_values(cast(Any, self), clamp_noise=False)
        labels = self._train_X[..., self._task_feature].expand(X.shape[:-1])
        if not torch.equal(X[..., self._task_feature], labels):
            raise UnsupportedError("Input transforms must preserve categorical task labels.")
        if self._mcmc_count is not None:
            batch = torch.broadcast_shapes(X.shape[:-2], torch.Size([self._mcmc_count]))
            X = X.expand(*batch, *X.shape[-2:])
        if noise is not None and clamp_noise:
            noise = noise.clamp_min(1e-4)
        return X, Y, noise

    def _noise_at(self, X: Tensor) -> Tensor:
        noise = self.likelihood.noise
        batch = torch.broadcast_shapes(noise.shape[:-1], X.shape[:-2])
        return noise.expand(*batch, X.shape[-2])

    def forward(self, X: Tensor) -> Any:
        self._check_if_fitted()
        if not isinstance(X, Tensor) or X.ndim < 2 or X.shape[-1] != self._train_X.shape[-1]:
            raise ValueError("Prior inputs must include the task feature column.")
        batch = torch.broadcast_shapes(X.shape[:-2], self.batch_shape)
        X = X.expand(*batch, *X.shape[-2:])
        if self.training:
            self._fit_transform_graphs()
            transform = self.input_transform
            if transform is not None:
                transformed = getattr(transform, "preprocess_transform", transform)(X)
                if not torch.equal(
                    transformed[..., self._task_feature], X[..., self._task_feature]
                ):
                    raise UnsupportedError(
                        "Input transforms must preserve categorical task labels."
                    )
                X = transformed
        mapped = self._mapped_inputs(X)
        return TorchGPyTorchPosterior(
            self._mean(mapped).unsqueeze(-1), self._covariance(mapped, mapped)
        ).distribution

    def posterior(
        self,
        X: Tensor,
        output_indices: Any = None,
        observation_noise: Any = False,
        posterior_transform: Any = None,
        **kwargs: Any,
    ) -> TorchGaussianMixturePosterior:
        self._check_if_fitted()
        if not isinstance(X, Tensor) or X.ndim < 2:
            raise ValueError("X must be a Tensor with shape (..., q, d).")
        expanded = X.unsqueeze(-3).expand(*X.shape[:-2], self.num_mcmc_samples, *X.shape[-2:])
        posterior = TorchMultiTaskGP.posterior(
            self,
            expanded,
            output_indices=output_indices,
            observation_noise=observation_noise,
            posterior_transform=posterior_transform,
            **kwargs,
        )
        if not isinstance(posterior, TorchGPyTorchPosterior):
            raise UnsupportedError(
                "A fully Bayesian mixture requires Gaussian component posteriors."
            )
        return TorchGaussianMixturePosterior(
            posterior.mean, posterior.distribution.covariance_matrix
        )

    def _exact_mll(self, output: Any = None, target: Any = None) -> Any:
        raise UnsupportedError("Fully Bayesian models use NUTS, not MAP fitting.")

    def load_mcmc_samples(self, samples: dict[str, Tensor]) -> None:
        like, d, tasks = self._train_X, self.num_non_task_features, self.num_tasks
        means = samples.get("mean")
        if (
            not isinstance(means, Tensor)
            or means.ndim != 2
            or means.shape[-1] != tasks
            or not means.shape[0]
        ):
            raise ValueError("MCMC mean must have shape (nonempty samples, num_tasks).")
        count, batch = means.shape[0], torch.Size([means.shape[0]])

        def sample(name: str, shape: tuple[int, ...], positive: bool = False) -> Tensor:
            value = samples.get(name)
            if (
                not isinstance(value, Tensor)
                or value.shape != (count, *shape)
                or value.dtype != like.dtype
                or value.device != like.device
                or not bool(torch.isfinite(value).all())
                or (positive and not bool((value > 0).all()))
            ):
                raise ValueError(
                    f"MCMC {name} must have shape {(count, *shape)}, finite"
                    + (" positive" if positive else "")
                    + " values and the training dtype/device."
                )
            return value.detach()

        means = sample("mean", (tasks,))
        linear = self.pyro_model._rs_kernel == "linear"
        parameters = sample("weight_variance" if linear else "lengthscale", (d,), True)
        scale = sample("outputscale", (), True) if "outputscale" in samples and not linear else None
        noise = sample("noise", ()) if self._train_Yvar is None else None
        rank = self.pyro_model.task_rank
        latent = sample("latent_features", (tasks, rank))
        task_ls = sample("task_lengthscale", (rank,), True)
        indices = self.pyro_model._warp_indices()
        concentrations = (
            {f"c{i}": sample(f"c{i}", (len(indices),), True) for i in (0, 1)}
            if indices is not None
            else {}
        )
        mean = TorchTaskMean(tasks, like)
        mean.base_means = nn.ModuleList(
            [build_mean(ConstantMean(batch_shape=batch), batch, like) for _ in range(tasks)]
        )
        for i, component in enumerate(mean.base_means):
            component.constant = means[:, i]
        source: Any = (
            LinearKernel(ard_num_dims=d, batch_shape=batch, active_dims=self._base_idxr)
            if linear
            else MaternKernel(ard_num_dims=d, batch_shape=batch, active_dims=self._base_idxr)
        )
        if scale is not None:
            source = ScaleKernel(source, batch_shape=batch)
        data: Any = build_kernel(source, d, batch, like)
        base = data.base_kernel if scale is not None else data
        if linear:
            base.variance = parameters.reshape(count, 1, d)
        else:
            base.lengthscale = parameters.reshape(count, 1, d)
        if scale is not None:
            data.outputscale = scale
        task_kernel = _SampleIndexKernel(tasks, batch, self._task_feature, like)
        covariance = _rs.tensor_kernel("matern", latent, latent, task_ls.unsqueeze(-2), nu=2.5)
        factor = _rs.tensor_psd_safe_cholesky(covariance)
        with torch.no_grad():
            task_kernel.covar_factor.copy_(factor)
            task_kernel.raw_var.fill_(-torch.inf)
        transforms: dict[str, nn.Module] = {}
        original = self._fb_user_input_transform
        if original is not None:
            transforms["user"] = original
        if concentrations:
            assert indices is not None
            columns = like.shape[-1]
            warp: Any = build_input_transform(
                Warp(
                    d=columns,
                    indices=indices,
                    batch_shape=batch,
                    bounds=[[0.0] * columns, [1.0] * columns],
                    eps=self.pyro_model._eps,
                ),
                columns,
                like,
            )
            with torch.no_grad():
                for i in (0, 1):
                    getattr(warp, f"concentration{i}").copy_(
                        concentrations[f"c{i}"].reshape(count, 1, len(indices))
                    )
            transforms["warp"] = warp
        if linear:
            transforms["center"] = _CenterFeatures(self._base_idxr)
        input_transform = TorchInputChain(transforms, like) if transforms else None
        old_transform = self.input_transform
        self.input_transform, self._mcmc_count = input_transform, count
        try:
            _, _, fixed = self._training_values()
            fixed = None if fixed is None else fixed.expand(count, fixed.shape[-1])
            likelihood: Any = build_likelihood(
                GaussianLikelihood(batch_shape=batch, noise_constraint=GreaterThan(1e-4))
                if fixed is None
                else None,
                fixed,
                batch,
                like,
            )
            if noise is not None:
                with torch.no_grad():
                    likelihood.raw_noise.copy_(
                        likelihood.noise_constraint.inverse_transform(
                            noise.reshape(count, 1).clamp_min(1e-4)
                        )
                    )
        except Exception:
            self.input_transform = old_transform
            self._mcmc_count = (
                None if self.covar_module is None else self.covar_module.batch_shape[0]
            )
            raise
        self.mean_module, self.covar_module, self.likelihood = (
            mean,
            TorchCompositeKernel([data, cast(Any, task_kernel)], "product", like),
            likelihood,
        )
        self._prediction_cache, self._transform_fit_versions = None, {}

    @property
    def median_lengthscale(self) -> Tensor:
        self._check_if_fitted()
        kernel = self.covar_module.kernels[0]
        return getattr(kernel, "base_kernel", kernel).lengthscale.median(0).values.squeeze(0)

    def load_state_dict(self, state_dict: Any, strict: bool = True, assign: bool = False) -> Any:
        mean = state_dict["mean_module.base_means.0.raw_constant"]
        count, tasks, rank, d = (
            mean.numel(),
            self.num_tasks,
            self.pyro_model.task_rank,
            self.num_non_task_features,
        )
        samples = {
            "mean": mean.new_zeros(count, tasks),
            "noise": mean.new_ones(count),
            "lengthscale": mean.new_ones(count, d),
            "weight_variance": mean.new_ones(count, d),
            "task_lengthscale": mean.new_ones(count, rank),
            "latent_features": torch.arange(tasks, dtype=mean.dtype, device=mean.device)
            .reshape(1, tasks, 1)
            .expand(count, tasks, rank)
            .clone(),
        }
        if "covar_module.kernels.0.raw_outputscale" in state_dict:
            samples["outputscale"] = mean.new_ones(count)
        indices = self.pyro_model._warp_indices()
        if indices is not None:
            samples.update({f"c{i}": mean.new_ones(count, len(indices)) for i in (0, 1)})
        self.load_mcmc_samples(samples)
        return nn.Module.load_state_dict(self, state_dict, strict=strict, assign=assign)

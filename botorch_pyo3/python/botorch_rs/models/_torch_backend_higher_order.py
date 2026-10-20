"""Correlated tensor-grid GPs with learned latent coordinates and full graphs."""

from __future__ import annotations

import copy
import math
import warnings
from typing import Any

import torch
from torch import Tensor, nn

from botorch_rs import _botorch_rs as _rs
from botorch_rs.exceptions.errors import UnsupportedError
from botorch_rs.models._torch_backend import TorchSingleTaskGP
from botorch_rs.models._torch_backend_kronecker import TorchKroneckerPosterior, tensor_kron
from botorch_rs.models._torch_backend_modules import (
    TorchStandardize,
    build_input_transform,
    build_kernel,
    build_likelihood,
    module_log_prior,
)
from botorch_rs.models.higher_order_gp import FlattenedStandardize, HigherOrderGP
from botorch_rs.models.transforms.outcome import Standardize
from botorch_rs.posteriors._torch_backend import TorchGPyTorchPosterior
from botorch_rs.posteriors.higher_order import HigherOrderGPPosterior
from botorch_rs.utils.types import DEFAULT


class TorchFlattenedStandardize(TorchStandardize):
    def __init__(self, output_shape: Any, batch: Any, like: Tensor, min_stdv: float = 1e-8) -> None:
        super().__init__(1, batch, like, min_stdv=min_stdv)
        self.output_shape = torch.Size(output_shape)

    def _flat(self, Y: Tensor) -> Tensor:
        return Y.reshape(*Y.shape[: -len(self.output_shape) - 1], -1, 1)

    def fit(self, Y: Tensor) -> TorchFlattenedStandardize:
        super().fit(self._flat(Y))
        return self

    def forward(self, Y: Tensor, Yvar: Tensor | None = None, X: Any = None) -> Any:
        if self.training:
            self.fit(Y)
        value = (self._flat(Y) - self.means) / self.stdvs
        variance = None if Yvar is None else self._flat(Yvar) / self._stdvs_sq
        return value.reshape(Y.shape), None if variance is None else variance.reshape(Y.shape)

    def untransform(self, Y: Tensor, Yvar: Tensor | None = None, X: Any = None) -> Any:
        value, variance = super().untransform(
            self._flat(Y), None if Yvar is None else self._flat(Yvar)
        )
        return value.reshape(Y.shape), None if variance is None else variance.reshape(Y.shape)

    def untransform_posterior(self, posterior: Any, X: Any = None) -> Any:
        from botorch_rs.posteriors.transformed import TransformedPosterior

        shape = self._batch_shape + torch.Size([1] * (len(self.output_shape) + 1))
        means, stdvs = self.means.reshape(shape).clone(), self.stdvs.reshape(shape).clone()
        return TransformedPosterior(
            posterior,
            sample_transform=lambda value: value * stdvs + means,
            mean_transform=lambda mean, variance: mean * stdvs + means,
            variance_transform=lambda mean, variance: variance * stdvs.square(),
        )


class TorchHigherOrderPosterior(HigherOrderGPPosterior):
    _tensor_backend = "torch"
    _uses_matheron = True

    def __init__(
        self, base: TorchKroneckerPosterior, target_shape: Any, num_outputs: int = 1
    ) -> None:
        self._flat = base
        self.target_shape = torch.Size(target_shape)
        self._output_shape = base.mean.shape[:-1] + self.target_shape
        self._batch = base.mean.shape[:-2]
        self._q = base.mean.shape[-2]
        self._m = math.prod(target_shape)
        self.num_outputs = num_outputs
        self.joint_covariance_matrix = base.joint_covariance_matrix
        self.train_train_covar = base.train_train_covar + base.train_noise
        self.test_train_covar = base.test_train_covar

    @property
    def output_shape(self) -> Any:
        return self._output_shape

    @property
    def mean(self) -> Tensor:
        return self._flat.mean.reshape(self.output_shape)

    @property
    def variance(self) -> Tensor:
        return self._flat.variance.reshape(self.output_shape)

    @property
    def device(self) -> Any:
        return self.mean.device

    @property
    def dtype(self) -> Any:
        return self.mean.dtype

    @property
    def distribution(self) -> Any:
        index = (
            torch.arange(self._q * self._m, device=self.device)
            .reshape(self._m, self._q)
            .T.flatten()
        )
        covariance = self._flat._covariance_tensor.index_select(-1, index).index_select(-2, index)
        return TorchGPyTorchPosterior(
            self._flat.mean.flatten(-2).unsqueeze(-1), covariance
        ).distribution

    @property
    def mvn(self) -> Any:
        return self.distribution

    @property
    def base_sample_shape(self) -> Any:
        return self._flat.base_sample_shape

    @property
    def batch_range(self) -> tuple[int, int]:
        return (0, -1)

    def _extended_shape(self, sample_shape: Any = ()) -> Any:
        return torch.Size(sample_shape) + self.output_shape

    def rsample(self, sample_shape: Any = None) -> Tensor:
        shape = torch.Size([1] if sample_shape is None else sample_shape)
        return self._flat.rsample(shape).reshape(shape + self.output_shape)

    def rsample_from_base_samples(self, sample_shape: Any, base_samples: Tensor) -> Tensor:
        shape = torch.Size(sample_shape)
        return self._flat.rsample_from_base_samples(shape, base_samples).reshape(
            shape + self.output_shape
        )


class TorchHigherOrderGP(nn.Module, HigherOrderGP):
    _tensor_backend = "torch"
    _supports_cache_root = False
    _train_X: Any
    _train_Y: Any
    _transform_fit_X: Any
    _transform_fit_Y: Any
    input_transform: Any
    outcome_transform: Any
    likelihood: Any
    covar_modules: Any
    _fit_transform_graphs: Any = TorchSingleTaskGP._fit_transform_graphs
    transform_inputs: Any = TorchSingleTaskGP.transform_inputs
    initialize: Any = TorchSingleTaskGP.initialize
    __deepcopy__: Any = TorchSingleTaskGP.__deepcopy__
    _sync_torch_parameters: Any = TorchSingleTaskGP._sync_torch_parameters

    @property
    def latent_parameters(self) -> Any:
        return self._modules["latent_parameters"]

    @latent_parameters.setter
    def latent_parameters(self, value: Any) -> None:
        self._modules["latent_parameters"] = value

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        likelihood: Any = None,
        covar_modules: Any = None,
        num_latent_dims: Any = None,
        learn_latent_pars: bool = True,
        latent_init: str = "default",
        outcome_transform: Any = DEFAULT,
        input_transform: Any = None,
        *,
        backend: str = "torch",
    ) -> None:
        from botorch_rs.models.utils.gpytorch_modules import get_covar_module_with_dim_scaled_prior

        super().__init__()
        if not isinstance(train_X, Tensor) or not isinstance(train_Y, Tensor) or train_X.ndim < 2:
            raise ValueError(
                "HigherOrderGP requires Tensor training features and tensor-grid targets."
            )
        if (
            train_Y.dtype != train_X.dtype
            or train_Y.device != train_X.device
            or train_X.dtype not in (torch.float32, torch.double)
        ):
            raise ValueError(
                "Training inputs and targets must share float32/float64 dtype and device."
            )
        batch = train_X.shape[:-2]
        if train_Y.shape[: len(batch) + 1] != train_X.shape[:-1] or train_Y.ndim < train_X.ndim:
            raise ValueError("Expected training targets with shape batch x n x output_shape.")
        if not bool(torch.isfinite(train_X).all() & torch.isfinite(train_Y).all()):
            raise ValueError("Training data must be finite.")
        self.target_shape = torch.Size(train_Y.shape[len(batch) + 1 :])
        if not all(self.target_shape) or not train_X.shape[-2]:
            raise ValueError("Training and target dimensions must be nonempty.")
        self._grid_size = math.prod(self.target_shape)
        self._num_dimensions = len(self.target_shape) + 1
        self._num_outputs = batch[0] if batch else 1
        self._transform_fit_versions: dict[int, Any] = {}
        for name, value in (
            ("_train_X", train_X),
            ("_train_Y", train_Y),
            ("_transform_fit_X", train_X),
            ("_transform_fit_Y", train_Y),
        ):
            self.register_buffer(name, value)
        self.input_transform = build_input_transform(input_transform, train_X.shape[-1], train_X)
        if (
            outcome_transform is DEFAULT
            or type(outcome_transform).__name__ == "_DefaultType"
            or isinstance(outcome_transform, Standardize)
        ):
            if isinstance(outcome_transform, Standardize) and not isinstance(
                outcome_transform, FlattenedStandardize
            ):
                warnings.warn(
                    "HigherOrderGP uses FlattenedStandardize instead of Standardize.",
                    RuntimeWarning,
                    stacklevel=2,
                )
            self.outcome_transform = TorchFlattenedStandardize(
                self.target_shape,
                batch,
                train_Y,
                min_stdv=getattr(outcome_transform, "_min_stdv", 1e-8),
            )
        elif outcome_transform is None:
            self.outcome_transform = None
        elif isinstance(outcome_transform, TorchFlattenedStandardize):
            self.outcome_transform = outcome_transform.to(train_X)
        else:
            raise UnsupportedError(
                "Tensor HigherOrderGP outcomes support FlattenedStandardize or None."
            )
        if isinstance(likelihood, nn.Module):
            external_noise: Any = getattr(likelihood, "noise", None)
            if (
                not isinstance(external_noise, Tensor)
                or external_noise.shape[-1:] != (1,)
                or not any(
                    cls.__name__ in {"GaussianLikelihood", "TorchGaussianLikelihood"}
                    for cls in type(likelihood).__mro__
                )
            ):
                raise UnsupportedError(
                    "Tensor HigherOrderGP requires Gaussian scalar observation noise."
                )
            self.likelihood = likelihood.to(train_X)
        else:
            self.likelihood = build_likelihood(likelihood, None, batch, train_X)
        dimensions = (
            [1] * len(self.target_shape) if num_latent_dims is None else list(num_latent_dims)
        )
        if len(dimensions) != len(self.target_shape) or any(
            not isinstance(d, int) or d < 1 for d in dimensions
        ):
            raise ValueError(
                "num_latent_dims must contain one positive integer per output dimension."
            )
        if covar_modules is None:
            covar_modules = [
                get_covar_module_with_dim_scaled_prior(d, batch_shape=batch)
                for d in [train_X.shape[-1], *dimensions]
            ]
        if len(covar_modules) != self._num_dimensions:
            raise ValueError("Expected one input covariance and one covariance per output axis.")
        self.covar_modules = nn.ModuleList(
            [
                build_kernel(source, dimension, batch, train_X)
                for source, dimension in zip(
                    covar_modules, [train_X.shape[-1], *dimensions], strict=True
                )
            ]
        )
        self.latent_parameters = nn.ParameterList()
        self._latent_prior_covariances = []
        if latent_init not in {"default", "gp"}:
            raise ValueError("latent_init must be 'default' or 'gp'.")
        for i, (size, dimension) in enumerate(zip(self.target_shape, dimensions, strict=True)):
            if latent_init == "default":
                latent = torch.rand(
                    *batch, size, dimension, dtype=train_X.dtype, device=train_X.device
                )
            else:
                grid = torch.linspace(
                    0, 1, size, dtype=train_X.dtype, device=train_X.device
                ).unsqueeze(-1)
                if dimension != 1:
                    grid = grid.expand(size, dimension)
                with torch.no_grad():
                    covariance = (
                        self._kernel(self.covar_modules[i + 1], grid, grid)
                        + torch.eye(size, dtype=grid.dtype, device=grid.device) * 1e-4
                    )
                    normal = torch.randn(
                        dimension, *batch, size, dtype=train_X.dtype, device=train_X.device
                    )
                    latent = (
                        (torch.linalg.cholesky(covariance) @ normal.unsqueeze(-1))
                        .squeeze(-1)
                        .movedim(0, -1)
                    )
                self.register_buffer(f"_latent_prior_{i}", covariance)
                self._latent_prior_covariances.append(i)
            self.latent_parameters.append(nn.Parameter(latent, requires_grad=learn_latent_pars))

    @staticmethod
    def _kernel(module: Any, X: Tensor, Q: Tensor) -> Tensor:
        result = module(X, Q)
        return result if isinstance(result, Tensor) else result.to_dense()

    @property
    def batch_shape(self) -> Any:
        return self._train_X.shape[:-2]

    @property
    def _aug_batch_shape(self) -> Any:  # type: ignore[override]
        return self.batch_shape

    @property
    def _input_batch_shape(self) -> Any:  # type: ignore[override]
        return self.batch_shape

    @property
    def num_outputs(self) -> int:
        return self._num_outputs

    @property
    def train_inputs(self) -> Any:
        return (self._train_X,)

    @property
    def train_targets(self) -> Any:
        return self._training_values()[1].flatten(start_dim=len(self.batch_shape))

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
            else self.outcome_transform(self._train_Y)[0]
        )
        return X, Y.reshape(*Y.shape[: len(self.batch_shape) + 1], self._grid_size)

    def _latent_covariance(self) -> Tensor:
        values = [
            self._kernel(module, latent, latent)
            for module, latent in zip(self.covar_modules[1:], self.latent_parameters, strict=True)
        ]
        result = values[0]
        for value in values[1:]:
            result = tensor_kron(result, value)
        return result

    def _log_prior(self) -> Tensor:
        value = self._train_X.new_zeros(self.batch_shape)
        for module in (self.likelihood, *self.covar_modules):
            value = value + module_log_prior(module, self.batch_shape)
        for i in self._latent_prior_covariances:
            latent = self.latent_parameters[i]
            covariance = getattr(self, f"_latent_prior_{i}")
            prior = torch.distributions.MultivariateNormal(
                torch.zeros_like(latent[..., 0]), covariance_matrix=covariance
            )
            value = value + prior.log_prob(latent.transpose(-1, -2).movedim(-2, 0)).sum(0)
        return value

    def _exact_mll(self, output: Any = None, target: Any = None) -> Tensor:
        X, Y = self._training_values()
        covariance = (
            tensor_kron(self._kernel(self.covar_modules[0], X, X), self._latent_covariance())
            if output is None
            else output.covariance_matrix
        )
        targets = Y.flatten(-2) if target is None else target
        value = _rs.tensor_exact_mll(
            targets, covariance, torch.zeros_like(targets), self.likelihood.noise
        )
        return (value + self._log_prior()) / targets.shape[-1]

    def posterior(
        self,
        X: Tensor,
        output_indices: Any = None,
        observation_noise: Any = False,
        posterior_transform: Any = None,
        **kwargs: Any,
    ) -> Any:
        if (
            output_indices is not None
            or posterior_transform is not None
            or kwargs
            or not isinstance(observation_noise, bool)
        ):
            raise UnsupportedError(
                "Higher-order Tensor prediction supports bool noise and no output/posterior transform overrides."
            )
        if (
            not isinstance(X, Tensor)
            or X.ndim < 2
            or X.device != self._train_X.device
            or X.dtype != self._train_X.dtype
            or X.shape[-1] != self._train_X.shape[-1]
        ):
            raise ValueError("Query dtype/device/features must match the model.")
        self.eval()
        train, Y = self._training_values()
        query = self.transform_inputs(X)
        batch = torch.broadcast_shapes(train.shape[:-2], query.shape[:-2])
        n, q, m = train.shape[-2], query.shape[-2], self._grid_size
        train, query = (
            train.expand(*batch, *train.shape[-2:]),
            query.expand(*batch, *query.shape[-2:]),
        )
        latent = self._latent_covariance()
        ktt = tensor_kron(self._kernel(self.covar_modules[0], train, train), latent)
        ktq = tensor_kron(self._kernel(self.covar_modules[0], train, query), latent)
        kqq = tensor_kron(self._kernel(self.covar_modules[0], query, query), latent)
        variance = self.likelihood.noise
        noise = torch.diag_embed(variance.expand(*batch, n * m))
        test_noise = torch.diag_embed(variance.expand(*batch, q * m)) if observation_noise else None
        targets = Y.flatten(-2)
        factor, alpha = _rs.tensor_exact_gp_factor(
            targets, ktt, torch.zeros_like(targets), variance
        )
        mean, covariance = _rs.tensor_exact_gp_predict(
            factor,
            alpha,
            ktq,
            kqq + (0 if test_noise is None else test_noise),
            query.new_zeros(*batch, q * m),
        )
        mean = mean.reshape(*batch, q, m)
        index = torch.arange(q * m, device=X.device).reshape(q, m).T.flatten()
        covariance = covariance.index_select(-1, index).index_select(-2, index)
        location, scale = 0.0, 1.0
        if self.outcome_transform is not None:
            location, scale = (
                self.outcome_transform.means.clone(),
                self.outcome_transform.stdvs.clone(),
            )
            mean = mean * scale + location
            covariance = covariance * scale.square()
        joint_X = torch.cat((train, query), -2)
        joint = tensor_kron(self._kernel(self.covar_modules[0], joint_X, joint_X), latent)
        base = TorchKroneckerPosterior(
            mean,
            covariance,
            joint,
            ktt,
            ktq.transpose(-1, -2),
            Y,
            torch.zeros_like(mean),
            noise,
            test_noise,
            n,
            m,
            location,
            scale,
        )
        return TorchHigherOrderPosterior(base, self.target_shape, self.num_outputs)

    def make_posterior_variances(self, joint_covariance_matrix: Any) -> Tensor:
        joint = joint_covariance_matrix
        if not isinstance(joint, Tensor):
            joint = joint.to_dense()
        count = self._train_X.shape[-2] * self._grid_size
        train = joint[..., :count, :count]
        cross = joint[..., :count, count:]
        noise = torch.diag_embed(self.likelihood.noise.expand(*train.shape[:-2], count))
        solve = torch.linalg.solve(train + noise, cross)
        return joint[..., count:, count:].diagonal(dim1=-2, dim2=-1) - (cross * solve).sum(-2)

    def forward(self, X: Tensor) -> Any:
        if not self.training:
            return self.posterior(X).distribution
        self._fit_transform_graphs()
        if self.input_transform is not None:
            X = getattr(self.input_transform, "preprocess_transform", self.input_transform)(X)
        covariance = tensor_kron(
            self._kernel(self.covar_modules[0], X, X), self._latent_covariance()
        )
        mean = X.new_zeros(*covariance.shape[:-1], 1)
        return TorchGPyTorchPosterior(mean, covariance).distribution

    def condition_on_observations(
        self, X: Tensor, Y: Tensor, noise: Any = None, **kwargs: Any
    ) -> Any:
        if noise is not None or kwargs:
            raise UnsupportedError(
                "Higher-order conditioning currently uses the configured Gaussian noise."
            )
        if (
            not isinstance(X, Tensor)
            or not isinstance(Y, Tensor)
            or X.ndim < 2
            or X.shape[-1] != self._train_X.shape[-1]
            or any(
                t.dtype != self._train_X.dtype or t.device != self._train_X.device for t in (X, Y)
            )
        ):
            raise ValueError("Conditioning data must match the model dtype, device and features.")
        if not bool(torch.isfinite(X).all() & torch.isfinite(Y).all()):
            raise ValueError("Conditioning data must be finite.")
        dims = len(self.target_shape)
        if tuple(Y.shape[-dims:]) != tuple(self.target_shape) or Y.shape[-dims - 1] != X.shape[-2]:
            raise ValueError("Conditioning targets must match q and target_shape.")
        batch = torch.broadcast_shapes(self.batch_shape, X.shape[:-2], Y.shape[: -dims - 1])
        model = copy.copy(self)
        model._buffers = self._buffers.copy()
        model._modules = self._modules.copy()
        model._transform_fit_versions = self._transform_fit_versions.copy()
        model._train_X = torch.cat(
            (
                self._train_X.expand(*batch, *self._train_X.shape[-2:]),
                X.expand(*batch, *X.shape[-2:]),
            ),
            dim=-2,
        )
        model._train_Y = torch.cat(
            (
                self._train_Y.expand(*batch, *self._train_Y.shape[-dims - 1 :]),
                Y.expand(*batch, *Y.shape[-dims - 1 :]),
            ),
            dim=-dims - 1,
        )
        return model.eval()

    def fantasize(
        self, X: Tensor, sampler: Any, observation_noise: Any = None, **kwargs: Any
    ) -> Any:
        propagate = kwargs.pop("propagate_grads", False)
        if kwargs or observation_noise is not None:
            raise UnsupportedError(
                "Higher-order fantasies currently use the configured Gaussian noise."
            )
        Y = sampler(self.posterior(X, observation_noise=True))
        return self.condition_on_observations(X, Y if propagate else Y.detach())

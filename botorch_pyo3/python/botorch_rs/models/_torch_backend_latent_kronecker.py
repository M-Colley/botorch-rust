"""Masked correlated feature/time GPs on CPU/CUDA with complete Tensor graphs."""

from __future__ import annotations

import copy
from typing import Any

import torch
from torch import Tensor, nn

from botorch_rs import _botorch_rs as _rs
from botorch_rs.exceptions.errors import UnsupportedError
from botorch_rs.kernels import MaternKernel, ScaleKernel
from botorch_rs.likelihoods import GaussianLikelihood
from botorch_rs.means import ZeroMean
from botorch_rs.models._torch_backend import TorchSingleTaskGP
from botorch_rs.models._torch_backend_kronecker import tensor_kron
from botorch_rs.models._torch_backend_modules import (
    TorchStandardize,
    build_input_transform,
    build_kernel,
    build_likelihood,
    build_mean,
    build_outcome_transform,
    module_log_prior,
)
from botorch_rs.models.latent_kronecker_gp import LatentKroneckerGP
from botorch_rs.posteriors._torch_backend import TorchGPyTorchPosterior
from botorch_rs.posteriors.latent_kronecker import LatentKroneckerGPPosterior
from botorch_rs.utils.types import DEFAULT


class TorchLatentKroneckerPosterior(TorchGPyTorchPosterior, LatentKroneckerGPPosterior):  # type: ignore[override]
    """A full Gaussian conditional with observed-mask Matheron samples."""

    _uses_matheron = True

    def __init__(
        self,
        mean: Tensor,
        covariance: Tensor,
        joint: Tensor,
        cross: Tensor,
        train_covariance: Tensor,
        targets: Tensor,
        train_mean: Tensor,
        test_mean: Tensor,
        noise: Tensor,
        mask: Tensor,
        location: Any = 0.0,
        scale: Any = 1.0,
    ) -> None:
        super().__init__(mean, covariance)
        self._joint_tensor, self._cross, self._train_covariance = joint, cross, train_covariance
        self._targets, self._train_mean, self._test_mean = targets, train_mean, test_mean
        self._noise, self._mask, self._location, self._scale = noise, mask, location, scale
        self._nfull = mask.numel()
        self._nobserved = int(mask.sum())

    @property
    def base_sample_shape(self) -> Any:
        return self._batch + torch.Size([self._nfull + self._nobserved + self._q * self._m])

    @property
    def batch_range(self) -> tuple[int, int]:
        return (0, -1)

    @property
    def distribution(self) -> Any:
        index = (
            torch.arange(self._q * self._m, device=self.device)
            .reshape(self._m, self._q)
            .T.flatten()
        )
        covariance = self._covariance_tensor.index_select(-1, index).index_select(-2, index)
        return TorchGPyTorchPosterior(self.mean.flatten(-2).unsqueeze(-1), covariance).distribution

    def rsample_from_base_samples(self, sample_shape: Any, base_samples: Tensor) -> Tensor:
        shape = torch.Size(sample_shape)
        if base_samples.shape[: len(shape)] != shape:
            raise RuntimeError("sample_shape disagrees with base_samples.shape.")
        tail = base_samples.shape[len(shape) :]
        missing = len(self._batch) + 1 - len(tail)
        if missing > 0:
            base_samples = base_samples.reshape(*shape, *([1] * missing), *tail)
        base = base_samples.expand(shape + self.base_sample_shape)
        # Public layout: full-grid prior, observed training noise, test prior.
        normal = torch.cat(
            (base[..., : self._nfull], base[..., self._nfull + self._nobserved :]), -1
        )
        prior = (_rs.tensor_psd_safe_cholesky(self._joint_tensor) @ normal.unsqueeze(-1)).squeeze(
            -1
        )
        epsilon = base[..., self._nfull : self._nfull + self._nobserved] * self._noise.sqrt()
        residual = (
            self._targets - self._train_mean - prior[..., : self._nfull][..., self._mask] - epsilon
        )
        factor = torch.linalg.cholesky(
            self._train_covariance + torch.diag_embed(self._noise.expand_as(self._targets))
        )
        update = self._cross.transpose(-1, -2) @ torch.cholesky_solve(
            residual.unsqueeze(-1), factor
        )
        value = self._test_mean + prior[..., self._nfull :] + update.squeeze(-1)
        return value.reshape(*shape, *self._batch, self._q, self._m) * self._scale + self._location


class TorchLatentKroneckerGP(nn.Module, LatentKroneckerGP):
    _tensor_backend = "torch"
    _supports_cache_root = False
    _train_X: Any
    _train_T: Any
    _train_Y: Any
    _transform_fit_X: Any
    _transform_fit_Y: Any
    _mask_valid: Any
    _freeze_outcome_statistics: bool = False
    input_transform: Any
    outcome_transform: Any
    likelihood: Any
    mean_module_X: Any
    mean_module_T: Any
    covar_module_X: Any
    covar_module_T: Any
    _fit_transform_graphs: Any = TorchSingleTaskGP._fit_transform_graphs
    initialize: Any = TorchSingleTaskGP.initialize
    __deepcopy__: Any = TorchSingleTaskGP.__deepcopy__
    _sync_torch_parameters: Any = TorchSingleTaskGP._sync_torch_parameters

    def __init__(
        self,
        train_X: Tensor,
        train_T: Tensor,
        train_Y: Tensor,
        likelihood: Any = None,
        mean_module_X: Any = None,
        mean_module_T: Any = None,
        covar_module_X: Any = None,
        covar_module_T: Any = None,
        input_transform: Any = None,
        outcome_transform: Any = DEFAULT,
        *,
        backend: str = "torch",
    ) -> None:
        nn.Module.__init__(self)
        if (
            not all(isinstance(v, Tensor) for v in (train_X, train_T, train_Y))
            or train_X.ndim < 2
            or train_T.ndim < 2
        ):
            raise ValueError("Expected Tensor train_X, train_T and train_Y on a feature/time grid.")
        if train_X.dtype not in (torch.float32, torch.double) or any(
            v.dtype != train_X.dtype or v.device != train_X.device for v in (train_T, train_Y)
        ):
            raise ValueError("Training tensors must share float32/float64 dtype and device.")
        batch = train_X.shape[:-2]
        if train_T.shape[-1] != 1 or train_Y.shape != (
            *batch,
            train_X.shape[-2],
            train_T.shape[-2],
        ):
            raise ValueError(
                "Expected train_X batch x n x d, train_T batch x t x 1 and train_Y batch x n x t."
            )
        try:
            train_T = train_T.expand(*batch, *train_T.shape[-2:])
        except RuntimeError as error:
            raise ValueError(
                "train_T batch shape must broadcast to train_X's batch shape."
            ) from error
        if (
            not train_X.shape[-2]
            or not train_T.shape[-2]
            or not bool(torch.isfinite(train_X).all() & torch.isfinite(train_T).all())
        ):
            raise ValueError("Feature/time inputs must be finite and nonempty.")
        mask = self._validate_mask(train_Y)
        self._transform_fit_versions: dict[int, Any] = {}
        for name, value in (
            ("_train_X", train_X),
            ("_train_T", train_T),
            ("_train_Y", train_Y),
            ("_transform_fit_X", train_X),
            ("_transform_fit_Y", train_Y.flatten(-2)[..., mask].unsqueeze(-1)),
            ("_mask_valid", mask),
        ):
            self.register_buffer(name, value)
        self.input_transform = build_input_transform(input_transform, train_X.shape[-1], train_X)
        self.outcome_transform = build_outcome_transform(
            outcome_transform, 1, self._transform_fit_Y
        )
        if self.outcome_transform is not None and not isinstance(
            self.outcome_transform, TorchStandardize
        ):
            raise UnsupportedError(
                "Tensor LatentKronecker outcomes currently support Standardize(m=1) or None."
            )
        if isinstance(likelihood, nn.Module):
            from botorch_rs.models._torch_structured_likelihood import validate_scalar_likelihood

            validate_scalar_likelihood(likelihood)
            self.likelihood = likelihood.to(train_X)
        else:
            self.likelihood = build_likelihood(
                GaussianLikelihood(batch_shape=batch) if likelihood is None else likelihood,
                None,
                batch,
                train_X,
            )
        self.mean_module_X = build_mean(
            ZeroMean(batch_shape=batch) if mean_module_X is None else mean_module_X, batch, train_X
        )
        self.mean_module_T = build_mean(
            ZeroMean(batch_shape=batch) if mean_module_T is None else mean_module_T, batch, train_X
        )
        self.covar_module_X = build_kernel(
            MaternKernel(nu=2.5, ard_num_dims=train_X.shape[-1], batch_shape=batch)
            if covar_module_X is None
            else covar_module_X,
            train_X.shape[-1],
            batch,
            train_X,
        )
        self.covar_module_T = build_kernel(
            ScaleKernel(MaternKernel(nu=2.5, ard_num_dims=1, batch_shape=batch), batch_shape=batch)
            if covar_module_T is None
            else covar_module_T,
            1,
            batch,
            train_X,
        )

    @staticmethod
    def _validate_mask(Y: Tensor) -> Tensor:
        masks = torch.isfinite(Y).flatten(-2).reshape(-1, Y.shape[-2] * Y.shape[-1])
        if not bool((masks == masks[0]).all()):
            raise ValueError("Pattern of missing values must agree across model batches.")
        if not bool(masks[0].any()):
            raise ValueError("At least one finite observation is required.")
        return masks[0].clone()

    @property
    def num_outputs(self) -> int:
        return self._train_T.shape[-2]

    @property
    def batch_shape(self) -> Any:
        return self._train_X.shape[:-2]

    @property
    def _aug_batch_shape(self) -> Any:  # type: ignore[override]
        return self.batch_shape

    @property
    def _input_batch_shape(self) -> Any:
        return self.batch_shape

    @property
    def train_inputs(self) -> Any:
        return self._train_X, self._train_T

    @property
    def train_T(self) -> Tensor:
        return self._train_T

    @property
    def mask_valid(self) -> Tensor:
        return self._mask_valid

    @property
    def train_targets(self) -> Tensor:
        return self._training_values()[2]

    def transform_inputs(self, X: Tensor, input_transform: Any = None) -> Tensor:
        if X is self._train_T:
            return X
        transform = self.input_transform if input_transform is None else input_transform
        return X if transform is None else transform(X)

    def _training_values(self) -> Any:
        current = self._validate_mask(self._train_Y)
        if not torch.equal(current, self.mask_valid):
            raise ValueError("The training missing-value pattern changed; reconstruct the model.")
        observed = self._train_Y.flatten(-2)[..., self.mask_valid].unsqueeze(-1)
        if not getattr(self, "_freeze_outcome_statistics", False):
            self._transform_fit_Y = observed
        self._fit_transform_graphs()
        X = (
            self._train_X
            if self.input_transform is None
            else getattr(self.input_transform, "preprocess_transform", self.input_transform)(
                self._train_X
            )
        )
        Y = observed if self.outcome_transform is None else self.outcome_transform(observed)[0]
        return X, self._train_T, Y.squeeze(-1)

    @staticmethod
    def _kernel(module: Any, X: Tensor, Q: Tensor) -> Tensor:
        value = module(X, Q)
        return value if isinstance(value, Tensor) else value.to_dense()

    def _covariance(self, X: Tensor, T: Tensor, Q: Tensor, U: Tensor) -> Tensor:
        return tensor_kron(
            self._kernel(self.covar_module_X, X, Q), self._kernel(self.covar_module_T, T, U)
        )

    def _get_mean(self, X: Tensor, T: Tensor, mask: Any = None) -> Tensor:
        value = (self.mean_module_X(X).unsqueeze(-1) * self.mean_module_T(T).unsqueeze(-2)).flatten(
            -2
        )
        return value if mask is None else value[..., mask]

    def _log_prior(self) -> Tensor:
        value = self._train_X.new_zeros(self.batch_shape)
        for module in (
            self.likelihood,
            self.mean_module_X,
            self.mean_module_T,
            self.covar_module_X,
            self.covar_module_T,
        ):
            value = value + module_log_prior(module, self.batch_shape)
        return value

    def _exact_mll(
        self, output: Any = None, target: Any = None, *params: Any, **kwargs: Any
    ) -> Tensor:
        from botorch_rs.models._torch_structured_likelihood import scalar_observed_law

        X, T, Y = self._training_values()
        if output is None and type(self).forward is not TorchLatentKroneckerGP.forward:
            output = nn.Module.__call__(self, self._train_X if self.training else X, T)
        covariance = (
            self._covariance(X, T, X, T)[..., self.mask_valid, :][..., :, self.mask_valid]
            if output is None
            else output.covariance_matrix
        )
        mean = self._get_mean(X, T, self.mask_valid) if output is None else output.mean
        target = Y if target is None else target
        mean, covariance = scalar_observed_law(
            self.likelihood, mean, covariance, (X, T), params=params, kwargs=kwargs
        )
        return (
            _rs.tensor_exact_mll(target, covariance, mean, torch.zeros_like(target))
            + self._log_prior()
        ) / target.shape[-1]

    def posterior(
        self,
        X: Tensor,
        T: Any = None,
        observation_noise: Any = False,
        posterior_transform: Any = None,
        **kwargs: Any,
    ) -> Any:
        return self._posterior(X, T, observation_noise, posterior_transform, **kwargs)

    def _posterior(
        self,
        X: Tensor,
        T: Any = None,
        observation_noise: Any = False,
        posterior_transform: Any = None,
        *,
        inputs_transformed: bool = False,
        untransform: bool = True,
        **kwargs: Any,
    ) -> Any:
        if kwargs or observation_noise is not False or posterior_transform is not None:
            raise UnsupportedError(
                "LatentKronecker prediction supports latent outputs and no posterior transform overrides."
            )
        T = self.train_T if T is None else T
        if (
            any(
                not isinstance(v, Tensor)
                or v.ndim < 2
                or v.dtype != self._train_X.dtype
                or v.device != self._train_X.device
                for v in (X, T)
            )
            or X.shape[-1] != self._train_X.shape[-1]
            or T.shape[-1] != 1
        ):
            raise ValueError("Query feature/time shapes, dtype and device must match the model.")
        self.eval()
        train, time, targets = self._training_values()
        query = X if inputs_transformed else self.transform_inputs(X)
        batch = torch.broadcast_shapes(self.batch_shape, query.shape[:-2], T.shape[:-2])
        train, time, query, T = [v.expand(*batch, *v.shape[-2:]) for v in (train, time, query, T)]
        full_train = self._covariance(train, time, train, time)
        full_cross = self._covariance(train, time, query, T)
        test = self._covariance(query, T, query, T)
        mask = self.mask_valid
        covariance = full_train[..., mask, :][..., :, mask]
        cross = full_cross[..., mask, :]
        mean_train = self._get_mean(train, time, mask)
        mean_test = self._get_mean(query, T)
        factor, alpha = _rs.tensor_exact_gp_factor(
            targets, covariance, mean_train, self.likelihood.noise
        )
        mean, cov = _rs.tensor_exact_gp_predict(factor, alpha, cross, test, mean_test)
        q, t = query.shape[-2], T.shape[-2]
        mean = mean.reshape(*batch, q, t)
        location, scale = 0.0, 1.0
        if untransform and self.outcome_transform is not None:
            location, scale = (
                self.outcome_transform.means.clone(),
                self.outcome_transform.stdvs.clone(),
            )
            mean = mean * scale + location
            cov = cov * scale.square()
        index = torch.arange(q * t, device=X.device).reshape(q, t).T.flatten()
        cov = cov.index_select(-1, index).index_select(-2, index)
        joint = torch.cat(
            (
                torch.cat((full_train, full_cross), -1),
                torch.cat((full_cross.transpose(-1, -2), test), -1),
            ),
            -2,
        )
        return TorchLatentKroneckerPosterior(
            mean,
            cov,
            joint,
            cross,
            covariance,
            targets,
            mean_train,
            mean_test,
            self.likelihood.noise,
            mask,
            location,
            scale,
        )

    def __call__(self, X: Tensor, T: Any = None, **kwargs: Any) -> Any:
        if not self.training:
            if kwargs:
                raise TypeError("LatentKroneckerGP evaluation accepts feature and time Tensors.")
            return self._posterior(X, T, inputs_transformed=True, untransform=False).distribution
        return nn.Module.__call__(self, X, T, **kwargs)

    def forward(self, X: Tensor, T: Any = None) -> Any:
        T = self.train_T if T is None else T
        if self.training:
            self._training_values()
            X = self.transform_inputs(X)
            mask = self.mask_valid
        else:
            mask = torch.ones(X.shape[-2] * T.shape[-2], dtype=torch.bool, device=X.device)
            mask[: self.mask_valid.numel()] = self.mask_valid
        covariance = self._covariance(X, T, X, T)[..., mask, :][..., :, mask]
        return TorchGPyTorchPosterior(
            self._get_mean(X, T, mask).unsqueeze(-1), covariance
        ).distribution

    def _rsample_from_base_samples(
        self, X: Tensor, T: Tensor, base_samples: Tensor, observation_noise: Any = False
    ) -> Tensor:
        posterior = self.posterior(X, T, observation_noise=observation_noise)
        count = base_samples.ndim - len(posterior._batch) - 1
        return posterior.rsample_from_base_samples(
            torch.Size(base_samples.shape[:count]), base_samples
        )

    def use_iterative_methods(self, *args: Any, **kwargs: Any) -> Any:
        raise UnsupportedError(
            "Tensor LatentKronecker currently uses exact dense solves; iterative settings are unsupported."
        )

    def condition_on_observations(
        self, X: Tensor, Y: Tensor, noise: Any = None, **kwargs: Any
    ) -> Any:
        if noise is not None or kwargs:
            raise UnsupportedError(
                "LatentKronecker conditioning uses the existing time grid and configured Gaussian noise."
            )
        if (
            not isinstance(X, Tensor)
            or not isinstance(Y, Tensor)
            or X.ndim < 2
            or Y.ndim < 2
            or any(
                v.device != self._train_X.device or v.dtype != self._train_X.dtype for v in (X, Y)
            )
            or X.shape[-1] != self._train_X.shape[-1]
            or Y.shape[-2:] != (X.shape[-2], self.num_outputs)
        ):
            raise ValueError("Conditioning data must match feature/time grid, dtype and device.")
        if not bool(torch.isfinite(X).all()):
            raise ValueError("Conditioning feature inputs must be finite.")
        batch = torch.broadcast_shapes(self.batch_shape, X.shape[:-2], Y.shape[:-2])
        model = copy.copy(self)
        model._buffers = self._buffers.copy()
        model._modules = self._modules.copy()
        model._transform_fit_versions = self._transform_fit_versions.copy()
        model._transform_fit_X = self._transform_fit_X.clone()
        model._train_X = torch.cat(
            (
                self._train_X.expand(*batch, *self._train_X.shape[-2:]),
                X.expand(*batch, *X.shape[-2:]),
            ),
            -2,
        )
        model._train_Y = torch.cat(
            (
                self._train_Y.expand(*batch, *self._train_Y.shape[-2:]),
                Y.expand(*batch, *Y.shape[-2:]),
            ),
            -2,
        )
        model._train_T = self._train_T.expand(*batch, *self._train_T.shape[-2:])
        model._mask_valid = self._validate_mask(model._train_Y)
        model._freeze_outcome_statistics = True
        return model.eval()

    def fantasize(
        self, X: Tensor, sampler: Any, observation_noise: Any = None, **kwargs: Any
    ) -> Any:
        propagate = kwargs.pop("propagate_grads", False)
        if observation_noise is not None or kwargs:
            raise UnsupportedError("LatentKronecker fantasies use the configured Gaussian noise.")
        posterior = self.posterior(X)
        # Sample the full observed law through the supplied sampler. No extra
        # ambient RNG draws are introduced outside its base-sample protocol.
        variance = self.likelihood.noise
        if self.outcome_transform is not None:
            variance = variance * self.outcome_transform.stdvs.squeeze(-1).square()
        diagonal = variance.expand(*posterior._batch, posterior._q * posterior._m)
        observed = TorchGPyTorchPosterior(
            posterior.mean, posterior._covariance_tensor + torch.diag_embed(diagonal)
        )
        Y = sampler(observed)
        return self.condition_on_observations(X, Y if propagate else Y.detach())

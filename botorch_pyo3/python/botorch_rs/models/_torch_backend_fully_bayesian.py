"""Device-resident exact inference from loaded MCMC hyperparameter samples."""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor, nn

from botorch_rs.constraints import GreaterThan
from botorch_rs.exceptions.errors import UnsupportedError
from botorch_rs.kernels import LinearKernel, MaternKernel, ScaleKernel
from botorch_rs.likelihoods import GaussianLikelihood
from botorch_rs.means import ConstantMean
from botorch_rs.models._torch_backend import TorchSingleTaskGP
from botorch_rs.models._torch_backend_modules import (
    TorchInputChain,
    build_input_transform,
    build_kernel,
    build_likelihood,
    build_mean,
)
from botorch_rs.models.fully_bayesian import (
    MIN_INFERRED_NOISE_LEVEL,
    AbstractFullyBayesianSingleTaskGP,
    FullyBayesianLinearSingleTaskGP,
    FullyBayesianSingleTaskGP,
    SaasFullyBayesianSingleTaskGP,
)
from botorch_rs.models.transforms.input import Normalize, Warp
from botorch_rs.posteriors._torch_backend import TorchGPyTorchPosterior
from botorch_rs.posteriors.fully_bayesian import GaussianMixturePosterior


class TorchGaussianMixturePosterior(TorchGPyTorchPosterior, GaussianMixturePosterior):
    """Component Gaussian laws and differentiable equal-weight mixture moments."""

    _is_gaussian_mixture = True

    def scalarize(self, weights: Any, offset: Any = 0.0) -> TorchGaussianMixturePosterior:
        posterior = super().scalarize(weights, offset)
        return type(self)(posterior.mean, posterior.distribution.covariance_matrix)

    def subset_output(self, indices: Any) -> TorchGaussianMixturePosterior:
        posterior = super().subset_output(indices)
        return type(self)(posterior.mean, posterior.distribution.covariance_matrix)

    @property
    def mixture_mean(self) -> Tensor:
        return self.mean.mean(-3)

    @property
    def mixture_variance(self) -> Tensor:
        return self.variance.mean(-3) + self.mean.var(-3, unbiased=False)

    @property
    def mixture_covariance_matrix(self) -> Tensor:
        mean = self.mean.transpose(-1, -2).flatten(-2)
        centered = mean - mean.mean(-2, keepdim=True)
        return (
            self._covariance_tensor.mean(-3)
            + (centered.transpose(-1, -2) @ centered) / mean.shape[-2]
        )

    @property
    def batch_range(self) -> tuple[int, int]:
        return (0, -2) if self._is_mt else (0, -1)

    def quantile(self, value: Any) -> Tensor:
        value = torch.as_tensor(value, device=self.device, dtype=self.dtype)
        if value.numel() != 1:
            return torch.stack([self.quantile(v) for v in value.reshape(-1)])
        if not bool(((value > 0) & (value < 1)).all()):
            raise ValueError("value is expected to be in the range (0, 1).")
        distribution = torch.distributions.Normal(self.mean, self.variance.sqrt())
        component_quantiles = distribution.icdf(value)
        if self.mean.shape[-3] == 1:
            return component_quantiles.squeeze(-3)
        # Locate the root without recording a discontinuous bisection graph.
        # Differentiable Newton refinements at that root retain the implicit
        # quantile derivatives, including higher-order parameter derivatives.
        with torch.no_grad():
            lower = component_quantiles.amin(-3)
            upper = component_quantiles.amax(-3)
            for _ in range(64):
                center = (lower + upper) * 0.5
                left = distribution.cdf(center.unsqueeze(-3)).mean(-3) > value
                lower, upper = torch.where(left, lower, center), torch.where(left, center, upper)
            root = (lower + upper) * 0.5
        for _ in range(8):
            point = root.unsqueeze(-3)
            cdf = distribution.cdf(point).mean(-3)
            density = distribution.log_prob(point).exp().mean(-3)
            root = root - (cdf - value) / density
        return root


class _TorchFullyBayesianBase(TorchSingleTaskGP):
    """A batch of exact GP components sharing original training data."""

    _mcmc_count: int | None
    _pyro_model_class: type[Any]
    _fb_user_input_transform: Any

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        train_Yvar: Tensor | None = None,
        outcome_transform: Any = None,
        input_transform: Any = None,
        use_input_warping: bool = False,
        indices_to_warp: list[int] | None = None,
        *,
        backend: str = "torch",
    ) -> None:
        if train_X.ndim != 2 or train_Y.ndim != 2 or train_Y.shape[-1] != 1:
            raise ValueError("Expected train_X to have shape n x d and train_Y to have shape n x 1")
        if train_Yvar is not None and train_Yvar.shape != train_Y.shape:
            raise ValueError("Expected train_Yvar to have the same shape as train_Y")
        self._mcmc_count, self._initializing = None, True
        TorchSingleTaskGP.__init__(
            self,
            train_X,
            train_Y,
            train_Yvar,
            outcome_transform=outcome_transform,
            input_transform=input_transform,
        )
        if self.outcome_transform is not None and not getattr(
            self.outcome_transform, "_is_linear", False
        ):
            raise UnsupportedError(
                "A Gaussian mixture posterior requires a linear outcome transform."
            )
        # Store the original module without registering a second alias. It
        # remains a child of input_transform, directly or within its chain.
        object.__setattr__(self, "_fb_user_input_transform", self.input_transform)
        self.pyro_model = self._pyro_model_class(
            use_input_warping=use_input_warping, indices_to_warp=indices_to_warp
        )
        self.pyro_model.train_X = train_X
        self.pyro_model.train_Y = train_Y
        self.pyro_model.train_Yvar = train_Yvar
        self.pyro_model.ard_num_dims = train_X.shape[-1]
        self.mean_module, self.covar_module, self.likelihood = None, None, None
        self._initializing = False

    @property
    def num_mcmc_samples(self) -> int:
        self._check_if_fitted()
        assert self._mcmc_count is not None
        return self._mcmc_count

    def _check_if_fitted(self) -> None:
        if self.covar_module is None:
            raise RuntimeError("Model has not been fitted. Call load_mcmc_samples first.")

    @property
    def batch_shape(self) -> torch.Size:
        return torch.Size() if self._initializing else torch.Size([self.num_mcmc_samples])

    @property
    def _input_batch_shape(self) -> torch.Size:
        return torch.Size()

    def train(self, mode: bool = True, reset: bool = True) -> Any:
        TorchSingleTaskGP.train(self, mode)
        if mode and reset:
            self.mean_module, self.covar_module, self.likelihood = None, None, None
            self._mcmc_count = None
            self.input_transform = self._fb_user_input_transform
        return self

    def _training_values(self, clamp_noise: bool = True) -> tuple[Any, Any, Any]:
        X, Y, noise = TorchSingleTaskGP._training_values(self, clamp_noise=False)
        if getattr(self, "_fb_custom_likelihood", False):
            # An explicitly returned Gaussian likelihood owns its observation
            # law, including when the original constructor supplied Yvar.
            return (
                X,
                Y,
                (self.likelihood.noise if getattr(self, "_fb_custom_fixed_noise", False) else None),
            )
        if noise is not None and clamp_noise:
            noise = noise.clamp_min(MIN_INFERRED_NOISE_LEVEL)
        return X, Y, noise

    def load_mcmc_samples(self, mcmc_samples: dict[str, Tensor]) -> None:
        """Load detached hyperparameter draws as actual registered Parameters."""
        from botorch_rs.models._torch_fully_bayesian_loader import (
            has_custom_loader,
            load_custom_modules,
        )

        if has_custom_loader(self.pyro_model):
            load_custom_modules(self, mcmc_samples)
            return
        like, d = self._train_X, self._train_X.shape[-1]
        mean = mcmc_samples.get("mean")
        if not isinstance(mean, Tensor) or mean.ndim != 1 or not mean.numel():
            raise ValueError("MCMC mean must be a nonempty one-dimensional Tensor.")
        count, batch = mean.numel(), torch.Size([mean.numel()])

        def sample(name: str, size: int, positive: bool = False) -> Tensor:
            result = mcmc_samples.get(name)
            if (
                not isinstance(result, Tensor)
                or result.device != like.device
                or result.dtype != like.dtype
                or result.ndim == 0
                or result.shape[0] != count
                or result.numel() != count * size
                or not bool(torch.isfinite(result).all())
                or (positive and not bool((result > 0).all()))
            ):
                raise ValueError(
                    f"MCMC {name} must contain {count} x {size} finite"
                    + (" positive" if positive else "")
                    + " values with the training dtype and device."
                )
            return result.detach()

        mean = sample("mean", 1).reshape(count)
        linear = self.pyro_model._rs_kernel == "linear"
        values = sample("weight_variance" if linear else "lengthscale", d, positive=True)
        scale = (
            sample("outputscale", 1, positive=True)
            if "outputscale" in mcmc_samples and not linear
            else None
        )
        noise = None if self._train_Yvar is not None else sample("noise", 1)
        indices = self.pyro_model.indices
        indices = list(range(d)) if indices is None else indices
        c0 = c1 = None
        if self.pyro_model.use_input_warping:
            c0, c1 = (sample(f"c{i}", len(indices), positive=True) for i in (0, 1))
        mean_module: Any = build_mean(ConstantMean(batch_shape=batch), batch, like)
        source: Any = (
            LinearKernel(ard_num_dims=d, batch_shape=batch)
            if linear
            else MaternKernel(ard_num_dims=d, batch_shape=batch)
        )
        if scale is not None:
            source = ScaleKernel(source, batch_shape=batch)
        kernel: Any = build_kernel(source, d, batch, like)
        mean_module.constant = mean
        if scale is not None:
            kernel.outputscale = scale.reshape(count)
        base = kernel.base_kernel if scale is not None else kernel
        if linear:
            base.variance = values.reshape(count, 1, d)
        else:
            base.lengthscale = values.reshape(count, 1, d)
        # Never append sampled warps to a previously loaded sampled chain.
        transforms: dict[str, nn.Module] = {}
        original = self._fb_user_input_transform
        if original is not None:
            transforms["user"] = original
        if c0 is not None:
            warp: Any = build_input_transform(
                Warp(
                    d=d,
                    indices=indices,
                    batch_shape=batch,
                    bounds=[[0.0] * d, [1.0] * d],
                    eps=self.pyro_model._eps,
                ),
                d,
                like,
            )
            assert warp is not None and c1 is not None
            with torch.no_grad():
                warp.concentration0.copy_(c0.reshape(count, 1, len(indices)))
                warp.concentration1.copy_(c1.reshape(count, 1, len(indices)))
            transforms["warp"] = warp
        if linear:
            normalize = build_input_transform(
                Normalize(d=d, bounds=[[0.0] * d, [1.0] * d], center=0.0), d, like
            )
            assert normalize is not None
            transforms["normalize"] = normalize
        transform = TorchInputChain(transforms, like) if transforms else None
        # Training transforms can add the MCMC axis (sampled warps), so install
        # the batch before recomputing live transformed fixed noise.
        self._mcmc_count = count
        self.input_transform = transform
        self.mean_module, self.covar_module = mean_module, kernel
        _, _, fixed_noise = self._training_values()
        if fixed_noise is not None:
            fixed_noise = fixed_noise.expand(count, fixed_noise.shape[-1])
        likelihood: Any = build_likelihood(
            GaussianLikelihood(
                batch_shape=batch, noise_constraint=GreaterThan(MIN_INFERRED_NOISE_LEVEL)
            )
            if fixed_noise is None
            else None,
            fixed_noise,
            batch,
            like,
        )
        if noise is not None:
            # The reference allows the exact floor, represented by raw -inf.
            # Its constrained value and all subsequent GP arithmetic are finite.
            with torch.no_grad():
                likelihood.raw_noise.copy_(
                    likelihood.noise_constraint.inverse_transform(
                        noise.reshape(count, 1).clamp_min(MIN_INFERRED_NOISE_LEVEL)
                    )
                )
        self.likelihood = likelihood
        self._fb_custom_likelihood = self._fb_custom_fixed_noise = False
        self._prediction_cache = None
        self._transform_fit_versions = {}

    def forward(self, X: Tensor) -> Any:
        self._check_if_fitted()
        self._fit_transform_graphs()
        if self.training:
            X = self.transform_inputs(X)
        return TorchGPyTorchPosterior(
            self.mean_module(X).unsqueeze(-1), self._covariance(X, X)
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
        if isinstance(observation_noise, Tensor):
            observation_noise = observation_noise.unsqueeze(-3)
        posterior = TorchSingleTaskGP.posterior(
            self,
            X.unsqueeze(-3),
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
        raise UnsupportedError(
            "Fully Bayesian Torch models use loaded MCMC samples, not MAP fitting."
        )

    @property
    def fantasize(self) -> Any:
        raise AttributeError(f"'{type(self).__name__}' object has no attribute 'fantasize'")

    def load_state_dict(self, state_dict: Any, strict: bool = True, assign: bool = False) -> Any:
        from botorch_rs.models._torch_fully_bayesian_loader import has_custom_loader

        mean = state_dict["mean_module.raw_constant"]
        if has_custom_loader(self.pyro_model):
            self.load_mcmc_samples(
                self.pyro_model.get_dummy_mcmc_samples(
                    mean.shape[0], dtype=mean.dtype, device=mean.device
                )
            )
            return nn.Module.load_state_dict(self, state_dict, strict=strict, assign=assign)
        count, d = mean.numel(), self._train_X.shape[-1]
        samples = {"mean": torch.zeros_like(mean), "noise": mean.new_ones(count)}
        samples["weight_variance" if self.pyro_model._rs_kernel == "linear" else "lengthscale"] = (
            mean.new_ones(count, d)
        )
        if "covar_module.raw_outputscale" in state_dict:
            samples["outputscale"] = mean.new_ones(count)
        if self.pyro_model.use_input_warping:
            width = d if self.pyro_model.indices is None else len(self.pyro_model.indices)
            samples.update({f"c{i}": mean.new_ones(count, width) for i in (0, 1)})
        self.load_mcmc_samples(samples)
        return nn.Module.load_state_dict(self, state_dict, strict=strict, assign=assign)


class TorchFullyBayesianSingleTaskGP(_TorchFullyBayesianBase, FullyBayesianSingleTaskGP):
    @property
    def median_lengthscale(self) -> Tensor:
        self._check_if_fitted()
        kernel = self.covar_module
        base = getattr(kernel, "base_kernel", kernel)
        return base.lengthscale.median(0).values.squeeze(0)


class TorchSaasFullyBayesianSingleTaskGP(
    TorchFullyBayesianSingleTaskGP, SaasFullyBayesianSingleTaskGP
):
    pass


class TorchFullyBayesianLinearSingleTaskGP(
    _TorchFullyBayesianBase, FullyBayesianLinearSingleTaskGP
):
    @property
    def median_weight_variance(self) -> Tensor:
        self._check_if_fitted()
        return self.covar_module.variance.median(0).values.squeeze(0)


_CUSTOM_TENSOR_CLASSES: dict[type, type] = {}


class _TorchAbstractFullyBayesianSingleTaskGP(
    _TorchFullyBayesianBase, AbstractFullyBayesianSingleTaskGP
):
    """Common device implementation for direct public abstract-base subclasses."""


def _restore_tensor_fully_bayesian(public_class: type) -> Any:
    from botorch_rs.models.fully_bayesian import AbstractFullyBayesianSingleTaskGP

    return AbstractFullyBayesianSingleTaskGP.__new__(public_class, backend="torch")


def _reduce_tensor_fully_bayesian(self: Any, protocol: int) -> Any:
    return (_restore_tensor_fully_bayesian, (self._public_model_class,), self.__getstate__())


def tensor_fully_bayesian_subclass(public_class: type, tensor_family: type | None) -> type:
    """Preserve explicit model subclasses and their cooperative overrides."""
    if tensor_family is not None and public_class in tensor_family.__bases__:
        return tensor_family
    tensor_class = _CUSTOM_TENSOR_CLASSES.get(public_class)
    if tensor_class is None:
        attributes: dict[str, Any] = {"__module__": __name__, "_public_model_class": public_class}
        if public_class.__reduce_ex__ is object.__reduce_ex__:
            # Serialize the public class and module state, rather than a
            # process-local generated class name. Explicit user reducers win.
            attributes["__reduce_ex__"] = _reduce_tensor_fully_bayesian
        tensor_class = type(
            f"Torch{public_class.__name__}",
            (
                public_class,
                _TorchAbstractFullyBayesianSingleTaskGP if tensor_family is None else tensor_family,
            ),
            attributes,
        )
        _CUSTOM_TENSOR_CLASSES[public_class] = tensor_class
    return tensor_class

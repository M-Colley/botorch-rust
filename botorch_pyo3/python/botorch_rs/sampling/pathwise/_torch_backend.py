"""Device Fourier paths and exact/whitened variational Matheron corrections.

Stationary covariances and PSD factorization use the port's native Tensor
bindings. All feature, solve, transform and path operations preserve graphs.
"""

from __future__ import annotations

import math
import sys
from typing import Any

import torch
from torch import Tensor, nn

from botorch_rs import _botorch_rs as _rs
from botorch_rs.exceptions.errors import UnsupportedError
from botorch_rs.models._torch_backend import TorchSingleTaskGP
from botorch_rs.models._torch_backend_modules import TorchScaleKernel, TorchStationaryKernel
from botorch_rs.models._torch_backend_variational import TorchSingleTaskVariationalGP
from botorch_rs.models.deterministic import MatheronPathModel
from botorch_rs.sampling.pathwise.features.maps import KernelEvaluationMap, KernelFeatureMap
from botorch_rs.sampling.pathwise.paths import GeneralizedLinearPath, PathList
from botorch_rs.sampling.pathwise.posterior_samplers import MatheronPath


def _dense(value: Any) -> Tensor:
    return value if isinstance(value, Tensor) else value.to_dense()


def _seed(seed: int | None) -> int:
    value = int(torch.randint(0, 1_000_000, (1,)).item()) if seed is None else int(seed)
    if value < 0:
        raise ValueError("The path seed must be nonnegative.")
    return value


def _normals(shape: Any, like: Tensor, seed: int) -> Tensor:
    # Only newly generated Sobol uniforms originate on CPU. Inference data,
    # model parameters and their normal transforms remain on the model device.
    dimension = shape[-1]
    uniforms = (
        torch.quasirandom.SobolEngine(dimension, scramble=True, seed=seed)
        .draw(math.prod(shape[:-1]), dtype=like.dtype)
        .to(like.device)
    )
    eps = torch.finfo(like.dtype).eps
    return uniforms.clamp(eps, 1 - eps).mul(2).sub(1).erfinv().mul(math.sqrt(2)).reshape(shape)


def _external_kernel(kernel: Any, name: str) -> bool:
    gpytorch = sys.modules.get("gpytorch")
    return gpytorch is not None and isinstance(kernel, getattr(gpytorch.kernels, name))


def _is_scale(kernel: Any) -> bool:
    return isinstance(kernel, TorchScaleKernel) or _external_kernel(kernel, "ScaleKernel")


def _family(kernel: Any) -> str:
    if isinstance(kernel, TorchStationaryKernel):
        return str(kernel.kind)
    if _external_kernel(kernel, "RBFKernel"):
        return "rbf"
    if _external_kernel(kernel, "MaternKernel"):
        return "matern"
    raise UnsupportedError("Device Fourier paths require RBF/Matérn kernels, optionally scaled.")


def _leaf(kernel: Any, num_inputs: int) -> tuple[Any, int]:
    while _is_scale(kernel):
        if kernel.active_dims is not None:
            num_inputs = kernel.active_dims.numel()
        kernel = kernel.base_kernel
    _family(kernel)
    return kernel, num_inputs if kernel.active_dims is None else kernel.active_dims.numel()


class TorchKernelFeatureMap(KernelFeatureMap):
    def __init__(self, kernel: Any, weight: Tensor) -> None:
        self.kernel, self.weight = kernel, weight
        self.bias, self.input_transform, self.output_transform = None, None, None
        self._num_features = 2 * weight.shape[-2]

    # The public FeatureMap bases use x while TensorTransform uses values.
    def forward(self, x: Any, **kwargs: Any) -> Tensor:  # type: ignore[override]
        if kwargs:
            raise UnsupportedError(f"Unsupported Fourier feature options: {sorted(kwargs)}.")
        kernel = self.kernel
        scale: Any = 1.0
        while _is_scale(kernel):
            # ScaleKernel inherits its base's selection. GPyTorch shares the
            # exact index object; the port registers separate device buffers.
            # In both cases apply an inherited selection only at the base.
            inherited = (
                kernel.base_kernel.active_dims is not None
                if isinstance(kernel, TorchScaleKernel)
                else kernel.active_dims is kernel.base_kernel.active_dims
            )
            if kernel.active_dims is not None and not inherited:
                x = x.index_select(-1, kernel.active_dims)
            scale = scale * kernel.outputscale.unsqueeze(-1).unsqueeze(-1)
            kernel = kernel.base_kernel
        if kernel.active_dims is not None:
            x = x.index_select(-1, kernel.active_dims)
        phase = (x / kernel.lengthscale) @ self.weight.transpose(-1, -2)
        features = torch.cat((phase.sin(), phase.cos()), -1) * math.sqrt(2 / self.num_outputs)
        return features * torch.as_tensor(scale, dtype=x.dtype, device=x.device).sqrt()


class TorchKernelEvaluationMap(KernelEvaluationMap):
    def forward(self, x: Any, **kwargs: Any) -> Tensor:  # type: ignore[override]
        if kwargs:
            raise UnsupportedError(f"Unsupported kernel evaluation options: {sorted(kwargs)}.")
        result = self.kernel(x, self.points)
        return result if isinstance(result, Tensor) else result.to_dense()


class TorchGeneralizedLinearPath(GeneralizedLinearPath):
    _tensor_backend = "torch"

    def forward(self, x: Tensor, **kwargs: Any) -> Tensor:
        if self.is_ensemble and not self.ensemble_as_batch:
            x = x.unsqueeze(-3)
        features = self.feature_map(x, **kwargs)
        out = (features @ self.weight.unsqueeze(-1)).squeeze(-1)
        return out if self.bias_module is None else out + self.bias_module(x)


class TorchMatheronPath(MatheronPath):
    _tensor_backend = "torch"
    _draw_record: list[tuple[Tensor, Tensor, Tensor]]

    @property
    def draws(self) -> Any:
        if not hasattr(self, "_draw_record"):
            raise UnsupportedError("Explicitly composed paths do not have a joint draw record.")
        return self._draw_record


class TorchMatheronPathModel(nn.Module, MatheronPathModel):
    _tensor_backend = "torch"

    def __init__(
        self,
        model: Any,
        sample_shape: Any = None,
        ensemble_as_batch: bool = False,
        seed: Any = None,
        *,
        num_features: int = 1024,
        draws: Any = None,
    ) -> None:
        from botorch_rs.sampling.pathwise.utils import sample_shape_arg

        nn.Module.__init__(self)
        self.model = model
        if hasattr(model, "models") and len(model.models) != model.num_outputs:
            raise UnsupportedError("A model-list of multi-output models is not supported.")
        self.sample_shape = sample_shape_arg(sample_shape)
        self.ensemble_as_batch = bool(ensemble_as_batch)
        self._features, self._seed = num_features, seed
        self._path = tensor_matheron_paths(model, self.sample_shape, num_features, seed, draws)
        self._path.set_ensemble_as_batch(ensemble_as_batch)
        from botorch_rs.utils.transforms import is_ensemble

        self._is_ensemble = is_ensemble(model) or bool(self.sample_shape)

    def forward(self, X: Tensor) -> Tensor:
        if hasattr(self.model, "models"):
            return torch.stack(self._path(X), -1)
        if self.model.num_outputs == 1:
            return self._path(X).unsqueeze(-1)
        return self._path(X.unsqueeze(-3)).transpose(-1, -2)

    def posterior(
        self,
        X: Tensor,
        output_indices: Any = None,
        posterior_transform: Any = None,
        observation_noise: Any = False,
        **kwargs: Any,
    ) -> Any:
        from botorch_rs.models._torch_backend_cost import TorchDeterministicCostPosterior

        if kwargs or observation_noise is not False:
            raise UnsupportedError(
                "A deterministic path posterior does not support observation noise or extra options."
            )
        values = self(X)
        if output_indices is not None:
            values = values[..., output_indices]
        posterior = TorchDeterministicCostPosterior(values)
        return posterior if posterior_transform is None else posterior_transform(posterior, X=X)

    def _apply(self, fn: Any, recurse: bool = True) -> Any:
        # Rebuild the exact correction after explicit device/dtype conversion,
        # retaining the same random path and correct transformed train data.
        paths = self._path.paths if isinstance(self._path, PathList) else [self._path]
        draws: list[tuple[Tensor, ...]] = []
        for path in paths:
            if not isinstance(path, TorchMatheronPath):
                raise UnsupportedError("Only device Matheron paths support model conversion.")
            draws.extend(tuple(fn(value) for value in draw) for draw in path.draws)
        nn.Module._apply(self, fn, recurse)
        self._path = tensor_matheron_paths(
            self.model, self.sample_shape, self._features, self._seed, draws
        )
        self._path.set_ensemble_as_batch(self.ensemble_as_batch)
        return self


def tensor_kernel_features(kernel: Any, num_inputs: int, num_outputs: int, seed: Any = None) -> Any:
    if num_outputs < 1 or num_outputs % 2 or num_inputs < 1:
        raise ValueError(
            "Fourier features require positive inputs and an even positive feature count."
        )
    leaf, dimensions = _leaf(kernel, num_inputs)
    like = leaf.lengthscale
    value = _seed(seed)
    weight = _normals((*kernel.batch_shape, num_outputs // 2, dimensions), like, value)
    if _family(leaf) == "matern":
        concentration = torch.full(
            (*kernel.batch_shape, num_outputs // 2, 1),
            leaf.nu,
            dtype=like.dtype,
            device=like.device,
        )
        generator = torch.Generator(device=like.device).manual_seed((value + 1) % (2**63 - 1))
        gamma = torch._standard_gamma(concentration, generator=generator) / leaf.nu
        weight = weight * gamma.rsqrt()
    return TorchKernelFeatureMap(kernel, weight)


def _require_model(model: Any) -> None:
    if not isinstance(model, (TorchSingleTaskGP, TorchSingleTaskVariationalGP)):
        raise UnsupportedError(
            "Device Gaussian paths require SingleTaskGP or SingleTaskVariationalGP models, or ModelLists thereof."
        )


def _modules(model: Any) -> Any:
    return model.model if isinstance(model, TorchSingleTaskVariationalGP) else model


def _fit_transforms(model: Any) -> None:
    if isinstance(model, TorchSingleTaskVariationalGP):
        model._fit_transforms()
    else:
        model._fit_transform_graphs()


def _outcome(model: Any) -> Any:
    if getattr(model, "outcome_transform", None) is None:
        return None

    def untransform(values: Tensor) -> Tensor:
        _fit_transforms(model)
        outcomes = values.unsqueeze(-1) if model.num_outputs == 1 else values.transpose(-1, -2)
        result = model.outcome_transform.untransform(outcomes)[0]
        return result.squeeze(-1) if model.num_outputs == 1 else result.transpose(-1, -2)

    return untransform


def _draw_components(
    model: Any, shape: tuple[int, ...], features: int, seed: int, draws: Any, original: bool
) -> Any:
    _require_model(model)
    model.eval()
    _fit_transforms(model)
    modules = _modules(model)
    batch = tuple(modules.covar_module.batch_shape)
    if draws is None:
        feature_map = tensor_kernel_features(
            modules.covar_module, model._train_X.shape[-1], features, seed
        )
        weight = _normals((*shape, *batch, features), model._train_X, (seed + 2) % (2**31 - 1))
    else:
        if len(draws) != model.num_outputs or any(len(d) != 2 for d in draws):
            raise ValueError("Provide one (frequencies, weights) tuple per output.")
        frequencies = [
            torch.as_tensor(d[0], dtype=model._train_X.dtype, device=model._train_X.device)
            for d in draws
        ]
        weights = [
            torch.as_tensor(d[1], dtype=model._train_X.dtype, device=model._train_X.device)
            for d in draws
        ]
        _, dimensions = _leaf(modules.covar_module, model._train_X.shape[-1])
        feature_map = TorchKernelFeatureMap(
            modules.covar_module,
            frequencies[0] if model.num_outputs == 1 else torch.stack(frequencies, -3),
        )
        weight = weights[0] if model.num_outputs == 1 else torch.stack(weights, -2)
        if weight.numel() == math.prod((*shape, *batch, features)):
            weight = weight.reshape(*shape, *batch, features)
        if (
            weight.shape != (*shape, *batch, features)
            or feature_map.weight.shape[:-2] != batch
            or feature_map.weight.shape[-2] != features // 2
            or feature_map.weight.shape[-1] != dimensions
            or features < 1
            or features % 2
        ):
            raise ValueError(
                "Explicit frequencies/weights must match sample, model and feature shapes."
            )
    return TorchGeneralizedLinearPath(
        feature_map,
        weight,
        bias_module=modules.mean_module,
        input_transform=model.input_transform,
        output_transform=_outcome(model) if original else None,
        is_ensemble=bool(getattr(model, "_is_ensemble", False)),
    )


def tensor_prior_paths(
    model: Any, shape: tuple[int, ...], features: int, seed: Any, draws: Any
) -> Any:
    value = _seed(seed)
    if hasattr(model, "models"):
        if draws is not None and len(draws) != model.num_outputs:
            raise ValueError("Provide one draw tuple per output.")
        start, paths = 0, []
        for i, child in enumerate(model.models):
            stop = start + child.num_outputs
            paths.append(
                tensor_prior_paths(
                    child, shape, features, value + i, None if draws is None else draws[start:stop]
                )
            )
            start = stop
        return PathList(paths)
    return _draw_components(model, shape, features, value, draws, True)


def tensor_gaussian_update(
    model: Any,
    sample_values: Tensor,
    likelihood: Any,
    target_values: Any,
    noise: Any,
    seed: Any,
    noise_covariance: Any = None,
    *,
    points: Any = None,
    scale_tril: Any = None,
) -> Any:
    from botorch_rs.sampling.pathwise.update_strategies import DEFAULT

    _require_model(model)
    if isinstance(model, TorchSingleTaskVariationalGP):
        if points is not None or scale_tril is not None:
            raise TypeError("points and scale_tril override exact-GP updates only.")
        return _variational_update(
            model, sample_values, target_values, noise, seed, noise_covariance
        )
    training_points, targets, fixed = model._training_values()
    points = training_points if points is None else points
    if (
        not isinstance(points, Tensor)
        or points.device != training_points.device
        or points.dtype != training_points.dtype
    ):
        raise ValueError("Update points must share the model dtype and device.")
    if not isinstance(sample_values, Tensor) or sample_values.shape[-1] != points.shape[-2]:
        raise ValueError(
            "Prior sample values must be device Tensors ending in the training-point dimension."
        )
    target = (
        targets
        if target_values is None
        else torch.as_tensor(target_values, dtype=targets.dtype, device=targets.device)
    )
    if likelihood is DEFAULT:
        likelihood = model.likelihood
    if noise_covariance is None:
        if likelihood is None:
            covariance_noise = None
        elif likelihood is model.likelihood:
            variance = (
                likelihood.noise_for(points.shape[-2])
                if fixed is None
                else model._training_noise(targets, fixed)
            )
            # FixedGaussianNoise has no noise entries for an unmatched set of
            # update points, exactly as its public noise_covar protocol.
            covariance_noise = (
                None
                if fixed is not None and variance.shape[-1] != points.shape[-2]
                else torch.diag_embed(variance)
            )
        elif callable(getattr(likelihood, "noise_covar", None)):
            covariance_noise = likelihood.noise_covar(shape=points.shape[:-1])
            covariance_noise = (
                None
                if type(covariance_noise).__name__ == "ZeroLinearOperator"
                else _dense(covariance_noise)
            )
        elif hasattr(likelihood, "noise_for"):
            covariance_noise = torch.diag_embed(likelihood.noise_for(points.shape[-2]))
        else:
            raise TypeError("Custom likelihoods must expose Gaussian noise_covar or noise_for.")
    elif type(noise_covariance).__name__ == "ZeroLinearOperator":
        covariance_noise = None
    else:
        covariance_noise = _dense(noise_covariance)
    errors = target - sample_values
    if covariance_noise is not None:
        if noise is None:
            generator = torch.Generator(device=targets.device).manual_seed(_seed(seed))
            noise = torch.randn(
                sample_values.shape, dtype=targets.dtype, device=targets.device, generator=generator
            )
        else:
            noise = torch.as_tensor(noise, dtype=targets.dtype, device=targets.device)
            if noise.shape != sample_values.shape:
                raise ValueError("Standard normal update noise must match sample_values.shape.")
        noise_factor = _rs.tensor_psd_safe_cholesky(covariance_noise)
        errors = errors - (noise_factor @ noise.unsqueeze(-1)).squeeze(-1)
    factor = (
        _rs.tensor_psd_safe_cholesky(
            model._covariance(points, points)
            + (0 if covariance_noise is None else covariance_noise)
        )
        if scale_tril is None
        else _dense(scale_tril)
    )
    weight = torch.cholesky_solve(errors.unsqueeze(-1), factor).squeeze(-1)
    features = TorchKernelEvaluationMap(
        model.covar_module, points, input_transform=model.input_transform
    )
    return TorchGeneralizedLinearPath(
        features, weight, is_ensemble=bool(getattr(model, "_is_ensemble", False))
    )


def _variational_update(
    model: Any,
    sample_values: Tensor,
    target_values: Any,
    noise: Any,
    seed: Any,
    noise_covariance: Any = None,
) -> Any:
    if model.num_outputs != 1:
        raise NotImplementedError(
            "Matheron updates require the scalar whitened VariationalStrategy; the reference does not register IndependentMultitaskVariationalStrategy updates."
        )
    if noise_covariance is not None and type(noise_covariance).__name__ != "ZeroLinearOperator":
        raise NotImplementedError(
            "Variational path updates do not support observation noise covariance."
        )
    model.eval()
    _fit_transforms(model)
    strategy, modules = model._strategy, model.model
    if getattr(strategy, "decoupled", False):
        raise UnsupportedError(
            "Decoupled mean/covariance inducing sets require a custom pathwise update."
        )
    if not strategy.whitened:
        raise NotImplementedError(
            "The reference Matheron dispatcher does not register UnwhitenedVariationalStrategy updates."
        )
    points = strategy.inducing_points  # Already in transformed input coordinates.
    batch = tuple(modules.covar_module.batch_shape)
    if (
        not isinstance(sample_values, Tensor)
        or sample_values.dtype != points.dtype
        or sample_values.device != points.device
        or sample_values.ndim < len(batch) + 1
        or sample_values.shape[-1] != points.shape[-2]
        or tuple(sample_values.shape[-len(batch) - 1 : -1]) != batch
    ):
        raise ValueError(
            "Variational prior values must have shape sample_shape x model_batch x num_inducing with the model dtype/device."
        )
    covariance = modules.covar_module(points, points)
    # The official update factors K_ZZ itself, not K_ZZ + strategy.jitter_val I.
    # Like its _cholesky_factor, factor in float64 and cast to the path dtype.
    factor = _rs.tensor_psd_safe_cholesky(covariance.double()).to(sample_values.dtype)
    if target_values is None:
        if noise is None:
            generator = torch.Generator(device=points.device).manual_seed(_seed(seed))
            noise = torch.randn(
                sample_values.shape, dtype=points.dtype, device=points.device, generator=generator
            )
        else:
            noise = torch.as_tensor(noise, dtype=points.dtype, device=points.device)
            if noise.shape != sample_values.shape:
                raise ValueError(
                    "Whitened variational base normals must match sample_values.shape."
                )
        distribution = strategy._variational_distribution
        from botorch_rs.models._torch_backend_variational import sample_variational_with_base

        whitened = sample_variational_with_base(distribution, noise)
        target = modules.mean_module(points) + (factor @ whitened.unsqueeze(-1)).squeeze(-1)
    else:
        if noise is not None:
            raise ValueError("Whitened base normals are only used when target_values is absent.")
        target = torch.as_tensor(target_values, dtype=points.dtype, device=points.device)
    errors = target - sample_values
    weight = torch.cholesky_solve(errors.unsqueeze(-1), factor).squeeze(-1)
    features = TorchKernelEvaluationMap(
        modules.covar_module, points, input_transform=model.input_transform
    )
    return TorchGeneralizedLinearPath(features, weight)


def tensor_matheron_paths(
    model: Any, shape: tuple[int, ...], features: int, seed: Any, draws: Any
) -> Any:
    from botorch_rs.sampling.pathwise.update_strategies import DEFAULT

    value = _seed(seed)
    if hasattr(model, "models"):
        if draws is not None and len(draws) != model.num_outputs:
            raise ValueError("Provide one draw tuple per output.")
        start, paths = 0, []
        for i, child in enumerate(model.models):
            stop = start + child.num_outputs
            paths.append(
                tensor_matheron_paths(
                    child, shape, features, value + i, None if draws is None else draws[start:stop]
                )
            )
            start = stop
        return PathList(paths)
    _require_model(model)
    if isinstance(model, TorchSingleTaskVariationalGP) and getattr(
        model._strategy, "decoupled", False
    ):
        raise UnsupportedError(
            "Decoupled mean/covariance inducing sets require a custom pathwise update."
        )
    if draws is not None and (len(draws) != model.num_outputs or any(len(d) != 3 for d in draws)):
        raise ValueError("Provide one (frequencies, weights, noise) tuple per output.")
    prior = _draw_components(
        model, shape, features, value, None if draws is None else [d[:2] for d in draws], False
    )
    if isinstance(model, TorchSingleTaskVariationalGP) and model.num_outputs != 1:
        raise NotImplementedError(
            "The reference Matheron sampler does not support IndependentMultitaskVariationalStrategy."
        )
    points = (
        model._strategy.inducing_points
        if isinstance(model, TorchSingleTaskVariationalGP)
        else model._training_values()[0]
    )
    sample_values = prior.forward(points)
    if draws is None:
        generator = torch.Generator(device=points.device).manual_seed((value + 3) % (2**63 - 1))
        noise = torch.randn(
            sample_values.shape, dtype=points.dtype, device=points.device, generator=generator
        )
    else:
        noises = [torch.as_tensor(d[2], dtype=points.dtype, device=points.device) for d in draws]
        noise = noises[0] if model.num_outputs == 1 else torch.stack(noises, -2)
        if noise.numel() == sample_values.numel():
            noise = noise.reshape(sample_values.shape)
    update = tensor_gaussian_update(model, sample_values, DEFAULT, None, noise, value)
    path = TorchMatheronPath(prior, update, output_transform=_outcome(model))
    path.join = lambda values: sum(values[1:], values[0])
    frequencies, weights = prior.feature_map.weight, prior.weight
    path._draw_record = (
        [(frequencies, weights, noise)]
        if model.num_outputs == 1
        else [
            (frequencies[..., i, :, :], weights[..., i, :], noise[..., i, :])
            for i in range(model.num_outputs)
        ]
    )
    return path

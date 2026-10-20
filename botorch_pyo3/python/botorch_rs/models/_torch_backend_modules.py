"""Port-owned modules for the optional device-preserving Torch backend.

These modules are independent of the CPU/f64 parameter bridge. Raw parameters
are ordinary ``nn.Parameter`` objects; constraints, priors and transforms use
Torch tensor operations, and stationary covariance evaluation dispatches to
Rust's ``tensor_kernel``. Import this module only when selecting that backend.
"""

from __future__ import annotations

import math
import warnings
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from botorch_rs import _botorch_rs as _rs
from botorch_rs.exceptions.errors import UnsupportedError
from botorch_rs.exceptions.warnings import NumericsWarning


def _tensor(value: Any, like: Tensor) -> Tensor:
    return torch.as_tensor(value, dtype=like.dtype, device=like.device)


class TorchConstraint(nn.Module):
    """Differentiable constraint with the original port constraint's bounds."""

    lower_bound: Tensor
    upper_bound: Tensor

    def __init__(self, source: Any, like: Tensor) -> None:
        super().__init__()
        self.kind = "Identity" if source is None else source._rs.kind
        self.transform_name = None if source is None else source._rs.transform_name
        self.enforced = source is not None and source.enforced
        self.initial_value = None if source is None else source.initial_value
        self.raw_bounds = (-math.inf, math.inf) if source is None else source._rs.raw_bounds
        self.register_buffer(
            "lower_bound", _tensor(-math.inf if source is None else source.lower_bound, like)
        )
        self.register_buffer(
            "upper_bound", _tensor(math.inf if source is None else source.upper_bound, like)
        )
        if self.kind == "LogTransformedInterval":
            self.transform_name = "exp"
        if self.transform_name not in (None, "softplus", "sigmoid", "exp"):
            raise UnsupportedError(f"Torch backend constraint transform {self.transform_name!r}.")

    def transform(self, raw: Tensor) -> Tensor:
        if self.transform_name == "exp":
            return raw.exp()
        if not self.enforced:
            return raw
        if self.transform_name == "sigmoid":
            return self.lower_bound + (self.upper_bound - self.lower_bound) * raw.sigmoid()
        if self.kind == "LessThan":
            return self.upper_bound - F.softplus(-raw)
        return F.softplus(raw) + self.lower_bound

    def inverse_transform(self, value: Tensor) -> Tensor:
        if self.transform_name == "exp":
            return value.log()
        if not self.enforced:
            return value
        if self.transform_name == "sigmoid":
            p = (value - self.lower_bound) / (self.upper_bound - self.lower_bound)
            return torch.logit(p)
        if self.kind == "LessThan":
            x = self.upper_bound - value
            return -(x + (-x).expm1().neg().log())
        x = value - self.lower_bound
        return x + (-x).expm1().neg().log()


class TorchPrior(nn.Module):
    """Univariate prior evaluated in the parameter's device and dtype."""

    hyperparameters: Tensor
    _normal_variance: Tensor
    _normal_log_scale: Tensor

    def __init__(self, source: Any, like: Tensor) -> None:
        super().__init__()
        self.kind = source._rs.kind
        self.register_buffer("hyperparameters", _tensor(source._rs.parameters, like))
        if self.kind == "LogNormal":
            # GPyTorch's transformed LogNormal keeps a Normal base_dist with
            # its construction dtype even after Module.to(). Its variance
            # and log(scale) are therefore computed before promotion to the
            # sampled parameter's dtype. Preserve those constants explicitly,
            # including the default float32 prior on float64 GP parameters.
            dtypes = source.__dict__.get("_buffer_argument_dtypes", {}).values()
            dtype = (
                "float64"
                if "float64" in dtypes
                else source.__dict__.get("_buffer_factory_dtype", "float32")
            )
            factory = torch.tensor(source._rs.parameters, dtype=getattr(torch, dtype))
            self.register_buffer("_normal_variance", factory[1].square().to(like))
            self.register_buffer("_normal_log_scale", factory[1].log().to(like))
        if self.kind not in (
            "LogNormal",
            "Gamma",
            "Normal",
            "Uniform",
            "HalfCauchy",
            "HalfNormal",
            "Beta",
            "SmoothedBox",
        ):
            raise UnsupportedError(f"Torch backend does not support prior {self.kind}.")

    def log_prob(self, value: Tensor) -> Tensor:
        p = self.hyperparameters.unbind()
        distributions = torch.distributions
        if self.kind == "Normal":
            return distributions.Normal(p[0], p[1]).log_prob(value)
        if self.kind == "LogNormal":
            if not bool((value > 0).all()):
                raise ValueError("LogNormal prior values must be positive.")
            log_value = value.log()
            return (
                -(log_value - p[0]).square() / (2 * self._normal_variance)
                - self._normal_log_scale
                - 0.5 * math.log(2 * math.pi)
                - log_value
            )
        if self.kind == "Gamma":
            return distributions.Gamma(p[0], p[1], validate_args=False).log_prob(value)
        if self.kind == "Uniform":
            return distributions.Uniform(p[0], p[1]).log_prob(value)
        if self.kind == "HalfCauchy":
            return distributions.HalfCauchy(p[0]).log_prob(value)
        if self.kind == "HalfNormal":
            return distributions.HalfNormal(p[0]).log_prob(value)
        if self.kind == "Beta":
            return distributions.Beta(p[0], p[1]).log_prob(value)
        a, b, sigma = p
        distance = (value - (a + b) / 2).abs() - (b - a) / 2
        return (
            -distance.clamp_min(0).square() / (2 * sigma.square())
            - (b - a + math.sqrt(2 * math.pi) * sigma).log()
        )


class _TorchModule(nn.Module):
    _anchor: Tensor

    def __init__(self, like: Tensor) -> None:
        super().__init__()
        self.register_buffer("_anchor", like.new_empty(0), persistent=False)
        self._prior_parameters: dict[str, tuple[str, int]] = {}

    def _register_prior(self, name: str, source: Any, parameter: str, event_ndim: int = 0) -> None:
        if source is not None:
            self.add_module(name, TorchPrior(source, self._anchor))
            self._prior_parameters[name] = parameter, event_ndim

    def log_prior(self) -> Tensor:
        total = self._anchor.new_zeros(getattr(self, "batch_shape", ()))
        # modules() removes shared aliases, so a shared parameter/prior pair is
        # counted once even when a kernel occurs in multiple tree positions.
        for module in self.modules():
            if isinstance(module, _TorchModule):
                for name, (parameter, event_ndim) in module._prior_parameters.items():
                    value = getattr(module, name).log_prob(getattr(module, parameter))
                    if event_ndim:
                        value = value.sum(tuple(range(-event_ndim, 0)))
                    total = total + value
        return total

    def named_parameters_and_constraints(self) -> Any:
        """GPyTorch's parameter/constraint traversal, preserving shared leaves."""
        modules = dict(self.named_modules())
        for name, parameter in self.named_parameters():
            prefix, _, local = name.rpartition(".")
            module = modules[prefix]
            constraint = getattr(module, f"{local}_constraint", None)
            if local == "raw_noise":
                constraint = getattr(module, "noise_constraint", constraint)
            yield name, parameter, constraint

    def raw_parameter_bounds(self) -> dict[str, tuple[float, float]]:
        return {
            name: constraint.raw_bounds
            for name, _, constraint in self.named_parameters_and_constraints()
            if constraint is not None
        }

    def _assign(self, name: str, value: Any, constraint: TorchConstraint | None = None) -> None:
        parameter = getattr(self, name)
        tensor = _tensor(value, parameter)
        if constraint is not None:
            tensor = constraint.inverse_transform(tensor)
        tensor = torch.broadcast_to(tensor, parameter.shape)
        if not torch.isfinite(tensor).all():
            raise ValueError(f"{name} must have finite raw values within the constraint's support.")
        with torch.no_grad():
            parameter.copy_(tensor)

    def initialize(self, **kwargs: Any) -> _TorchModule:
        for name, value in kwargs.items():
            current = getattr(self, name)
            if isinstance(current, nn.Parameter):
                self._assign(name, value)
            else:
                setattr(self, name, value)
        return self


class TorchKernel(_TorchModule):
    """Dense covariance module; native tensor operations retain autograd."""

    active_dims: Tensor | None
    batch_shape: torch.Size

    def __init__(self, like: Tensor, active_dims: Any = None) -> None:
        super().__init__(like)
        self.register_buffer(
            "active_dims",
            None
            if active_dims is None
            else torch.as_tensor(active_dims, dtype=torch.long, device=like.device),
        )

    def _inputs(self, x1: Tensor, x2: Tensor | None) -> tuple[Tensor, Tensor]:
        x2 = x1 if x2 is None else x2
        for x in (x1, x2):
            if not isinstance(x, Tensor) or not x.is_floating_point() or x.ndim < 2:
                raise ValueError(
                    "Kernel inputs must be floating Torch tensors with shape ... x n x d."
                )
            if x.device != self._anchor.device or x.dtype != self._anchor.dtype:
                raise ValueError(
                    "Kernel inputs and parameters must have the same device and dtype."
                )
        if x1.shape[-1] != x2.shape[-1]:
            raise ValueError("Kernel input feature dimensions must agree.")
        if self.active_dims is not None:
            x1 = x1.index_select(-1, self.active_dims)
            x2 = x2.index_select(-1, self.active_dims)
        return x1, x2

    def __add__(self, other: TorchKernel) -> TorchCompositeKernel:
        return TorchCompositeKernel([self, other], "add", self._anchor)

    def __mul__(self, other: TorchKernel) -> TorchCompositeKernel:
        return TorchCompositeKernel([self, other], "product", self._anchor)

    def _forward_selected(
        self, x1: Tensor, x2: Tensor, diag: bool, last_dim_is_batch: bool
    ) -> Tensor:
        raise NotImplementedError


class TorchStationaryKernel(TorchKernel):
    def __init__(self, source: Any, like: Tensor) -> None:
        super().__init__(like, source.active_dims)
        self.ard_num_dims = source.ard_num_dims
        self.batch_shape = torch.Size(source._batch_shape)
        self.kind = "rbf" if type(source).__name__ == "RBFKernel" else "matern"
        self.nu = float(getattr(source, "nu", 2.5))
        self.raw_lengthscale_constraint = TorchConstraint(source.raw_lengthscale_constraint, like)
        original = source.raw_lengthscale
        raw = _tensor(original, like).reshape(source._param_shape("raw_lengthscale"))
        self.raw_lengthscale = nn.Parameter(
            raw.clone(),
            requires_grad=not isinstance(original, nn.Parameter) or original.requires_grad,
        )
        self._register_prior("lengthscale_prior", source.lengthscale_prior, "lengthscale", 2)

    @property
    def lengthscale(self) -> Tensor:
        return self.raw_lengthscale_constraint.transform(self.raw_lengthscale)

    @lengthscale.setter
    def lengthscale(self, value: Any) -> None:
        self._assign("raw_lengthscale", value, self.raw_lengthscale_constraint)

    def forward(
        self,
        x1: Tensor,
        x2: Tensor | None = None,
        diag: bool = False,
        last_dim_is_batch: bool = False,
    ) -> Tensor:
        x1, x2 = self._inputs(x1, x2)
        return self._forward_selected(x1, x2, diag, last_dim_is_batch)

    def _forward_selected(
        self, x1: Tensor, x2: Tensor, diag: bool, last_dim_is_batch: bool
    ) -> Tensor:
        if self.ard_num_dims is not None and x1.shape[-1] != self.ard_num_dims:
            raise ValueError(
                f"Expected {self.ard_num_dims} selected input dimensions, got {x1.shape[-1]}."
            )
        return _rs.tensor_kernel(
            self.kind, x1, x2, self.lengthscale, self.nu, diag, last_dim_is_batch
        )


class TorchScaleKernel(TorchKernel):
    def __init__(self, source: Any, base_kernel: TorchKernel, like: Tensor) -> None:
        super().__init__(like, source.active_dims)
        self.base_kernel = base_kernel
        self.batch_shape = torch.Size(source.batch_shape)
        self.raw_outputscale_constraint = TorchConstraint(source.raw_outputscale_constraint, like)
        original = source.raw_outputscale
        raw = _tensor(original, like).reshape(source._param_shape("raw_outputscale"))
        self.raw_outputscale = nn.Parameter(
            raw.clone(),
            requires_grad=not isinstance(original, nn.Parameter) or original.requires_grad,
        )
        self._register_prior("outputscale_prior", source.outputscale_prior, "outputscale")

    @property
    def outputscale(self) -> Tensor:
        return self.raw_outputscale_constraint.transform(self.raw_outputscale)

    @outputscale.setter
    def outputscale(self, value: Any) -> None:
        self._assign("raw_outputscale", value, self.raw_outputscale_constraint)

    def forward(
        self,
        x1: Tensor,
        x2: Tensor | None = None,
        diag: bool = False,
        last_dim_is_batch: bool = False,
    ) -> Tensor:
        x1, x2 = self._inputs(x1, x2)
        return self._forward_selected(x1, x2, diag, last_dim_is_batch)

    def _forward_selected(
        self, x1: Tensor, x2: Tensor, diag: bool, last_dim_is_batch: bool
    ) -> Tensor:
        # GPyTorch ScaleKernel calls base.forward directly: its inherited
        # active_dims select once at this level, bypassing base.__call__.
        value = self.base_kernel._forward_selected(x1, x2, diag, last_dim_is_batch)
        scale = self.outputscale
        for _ in range((1 if diag else 2) + int(last_dim_is_batch)):
            scale = scale.unsqueeze(-1)
        return value * scale


class TorchCompositeKernel(TorchKernel):
    def __init__(self, kernels: list[TorchKernel], kind: str, like: Tensor) -> None:
        super().__init__(like)
        if not kernels:
            raise ValueError("A composite kernel needs at least one child.")
        self.kernels = nn.ModuleList(kernels)
        self.kind = kind
        self.batch_shape = torch.broadcast_shapes(*(kernel.batch_shape for kernel in kernels))

    def forward(
        self,
        x1: Tensor,
        x2: Tensor | None = None,
        diag: bool = False,
        last_dim_is_batch: bool = False,
    ) -> Tensor:
        x1, x2 = self._inputs(x1, x2)
        return self._forward_selected(x1, x2, diag, last_dim_is_batch)

    def _forward_selected(
        self, x1: Tensor, x2: Tensor, diag: bool, last_dim_is_batch: bool
    ) -> Tensor:
        values = [
            kernel(x1, x2, diag=diag, last_dim_is_batch=last_dim_is_batch)
            for kernel in self.kernels
        ]
        result = values[0]
        for value in values[1:]:
            result = result + value if self.kind == "add" else result * value
        return result


class TorchAlgebraKernel(TorchKernel):
    """Parameter plumbing for the remaining elementary native Torch kernels."""

    def __init__(self, source: Any, like: Tensor) -> None:
        super().__init__(like, source.active_dims)
        self.batch_shape = torch.Size(source.batch_shape)
        self.ard_num_dims = source.ard_num_dims
        self.kind = type(source).__name__.removesuffix("Kernel").lower()
        self.power = float(getattr(source, "power", 2))
        self._logical_parameters: dict[str, str] = {}
        event_shapes = source._element_shapes()
        for raw_name, constraint, prior in source._own_params():
            logical = raw_name.removeprefix("raw_")
            original = getattr(source, raw_name)
            raw = _tensor(original, like).reshape(source._param_shape(raw_name)).clone()
            self.register_parameter(
                raw_name,
                nn.Parameter(
                    raw,
                    requires_grad=not isinstance(original, nn.Parameter) or original.requires_grad,
                ),
            )
            self.add_module(f"{raw_name}_constraint", TorchConstraint(constraint, like))
            self._logical_parameters[logical] = raw_name
            self._register_prior(f"{logical}_prior", prior, logical, len(event_shapes[raw_name]))

    def __getattr__(self, name: str) -> Any:
        logical = self.__dict__.get("_logical_parameters", {})
        if name in logical:
            raw_name = logical[name]
            return getattr(self, f"{raw_name}_constraint").transform(getattr(self, raw_name))
        return super().__getattr__(name)

    def __setattr__(self, name: str, value: Any) -> None:
        logical = self.__dict__.get("_logical_parameters", {})
        if name in logical:
            raw_name = logical[name]
            self._assign(raw_name, value, getattr(self, f"{raw_name}_constraint"))
        else:
            super().__setattr__(name, value)

    def forward(
        self,
        x1: Tensor,
        x2: Tensor | None = None,
        diag: bool = False,
        last_dim_is_batch: bool = False,
    ) -> Tensor:
        x1, x2 = self._inputs(x1, x2)
        return self._forward_selected(x1, x2, diag, last_dim_is_batch)

    def _forward_selected(
        self, x1: Tensor, x2: Tensor, diag: bool, last_dim_is_batch: bool
    ) -> Tensor:
        if self.ard_num_dims is not None and x1.shape[-1] != self.ard_num_dims:
            raise ValueError(
                f"Expected {self.ard_num_dims} selected input dimensions, got {x1.shape[-1]}."
            )
        parameters = {
            name: getattr(self, name) for name in self._logical_parameters if name != "lengthscale"
        }
        if self.kind == "polynomial":
            parameters["power"] = self.power
        return _rs.tensor_kernel(
            self.kind,
            x1,
            x2,
            getattr(self, "lengthscale", None),
            2.5,
            diag,
            last_dim_is_batch,
            parameters=parameters,
        )


class TorchConstantMean(_TorchModule):
    def __init__(self, source: Any, like: Tensor) -> None:
        super().__init__(like)
        self.batch_shape = torch.Size(source.batch_shape)
        self.raw_constant_constraint = TorchConstraint(source.raw_constant_constraint, like)
        raw = _tensor(source.raw_constant, like).expand(self.batch_shape).clone()
        self.raw_constant = nn.Parameter(
            raw,
            requires_grad=not isinstance(source.raw_constant, nn.Parameter)
            or source.raw_constant.requires_grad,
        )
        self._register_prior("mean_prior", source.mean_prior, "constant")

    @property
    def constant(self) -> Tensor:
        return self.raw_constant_constraint.transform(self.raw_constant)

    @constant.setter
    def constant(self, value: Any) -> None:
        self._assign("raw_constant", value, self.raw_constant_constraint)

    def forward(self, x: Tensor) -> Tensor:
        shape = torch.broadcast_shapes(self.batch_shape, x.shape[:-2])
        return self.constant.unsqueeze(-1).expand(*shape, x.shape[-2])


class TorchZeroMean(_TorchModule):
    def __init__(self, batch_shape: Any, like: Tensor) -> None:
        super().__init__(like)
        self.batch_shape = torch.Size(batch_shape)

    def forward(self, x: Tensor) -> Tensor:
        shape = torch.broadcast_shapes(self.batch_shape, x.shape[:-2])
        return x.new_zeros(*shape, x.shape[-2])


class _TorchGaussianNoise(_TorchModule):
    def __init__(self, source: Any, like: Tensor) -> None:
        super().__init__(like)
        self.batch_shape = torch.Size(source.batch_shape)
        self.raw_noise_constraint = TorchConstraint(source.noise_constraint, like)
        initial = self.noise_constraint.initial_value
        self.raw_noise = nn.Parameter(
            like.new_full((*self.batch_shape, 1), 0 if initial is None else initial)
        )
        # Native likelihoods have no independent unbound parameter storage.
        if getattr(source, "_model", None) is not None:
            self._assign("raw_noise", source.raw_noise)
            if isinstance(source.raw_noise, nn.Parameter):
                self.raw_noise.requires_grad_(source.raw_noise.requires_grad)
        self._register_prior("noise_prior", source.noise_prior, "noise", 1)

    @property
    def noise_constraint(self) -> TorchConstraint:
        return self.raw_noise_constraint

    @property
    def noise(self) -> Tensor:
        return self.noise_constraint.transform(self.raw_noise)

    @noise.setter
    def noise(self, value: Any) -> None:
        self._assign("raw_noise", value, self.noise_constraint)


class TorchGaussianLikelihood(_TorchModule):
    def __init__(self, source: Any, like: Tensor) -> None:
        super().__init__(like)
        self.batch_shape = torch.Size(source.batch_shape)
        self.noise_covar = _TorchGaussianNoise(source, like)

    @property
    def raw_noise(self) -> nn.Parameter:
        return self.noise_covar.raw_noise

    @raw_noise.setter
    def raw_noise(self, value: Any) -> None:
        self.noise_covar._assign("raw_noise", value)

    @property
    def noise_constraint(self) -> TorchConstraint:
        return self.noise_covar.noise_constraint

    @property
    def raw_noise_constraint(self) -> TorchConstraint:
        return self.noise_covar.noise_constraint

    @property
    def noise(self) -> Tensor:
        return self.noise_covar.noise

    @noise.setter
    def noise(self, value: Any) -> None:
        self.noise_covar.noise = value

    def noise_for(self, n: int, noise: Tensor | None = None) -> Tensor:
        return self.noise.expand(*self.batch_shape, n) if noise is None else noise

    def forward(self, distribution: Any, noise: Tensor | None = None, **kwargs: Any) -> Any:
        if kwargs:
            raise UnsupportedError(f"Unsupported Gaussian likelihood arguments: {sorted(kwargs)}.")
        if isinstance(distribution, Tensor):
            return torch.distributions.Normal(
                distribution, self.noise_for(distribution.shape[-1], noise).sqrt()
            )
        from botorch_rs.posteriors._torch_backend import TorchGPyTorchPosterior

        mean, covariance = distribution.mean, distribution.covariance_matrix
        return TorchGPyTorchPosterior(
            mean.unsqueeze(-1), covariance + torch.diag_embed(self.noise_for(mean.shape[-1], noise))
        ).distribution


class TorchFixedNoiseLikelihood(_TorchModule):
    noise: Tensor

    def __init__(self, noise: Tensor, batch_shape: Any, like: Tensor) -> None:
        super().__init__(like)
        self.batch_shape = torch.Size(batch_shape)
        # The model recomputes transformed training noise from its original
        # Tensor on each evaluation. This inspection buffer is an initial
        # snapshot, avoiding a consumed construction-time autograd graph.
        minimum = 1e-4 if noise.dtype == torch.float32 else 1e-6
        if bool((noise < minimum).any()):
            warnings.warn(
                f"Very small noise values detected. Rounding small noise values up to {minimum}.",
                NumericsWarning,
                stacklevel=3,
            )
        self.register_buffer("noise", noise.detach().clamp_min(minimum).clone())

    @property
    def noise_covar(self) -> TorchFixedNoiseLikelihood:
        return self

    def noise_for(self, n: int, noise: Tensor | None = None) -> Tensor:
        if noise is not None:
            return noise
        if self.noise.shape[-1] != n:
            raise ValueError(
                "Fixed-noise predictions at a different number of points require explicit observation noise."
            )
        return self.noise

    def forward(self, distribution: Any, noise: Tensor | None = None, **kwargs: Any) -> Any:
        if kwargs:
            raise UnsupportedError(
                f"Unsupported fixed-noise likelihood arguments: {sorted(kwargs)}."
            )
        if isinstance(distribution, Tensor):
            return torch.distributions.Normal(
                distribution, self.noise_for(distribution.shape[-1], noise).sqrt()
            )
        from botorch_rs.posteriors._torch_backend import TorchGPyTorchPosterior

        mean, covariance = distribution.mean, distribution.covariance_matrix
        return TorchGPyTorchPosterior(
            mean.unsqueeze(-1), covariance + torch.diag_embed(self.noise_for(mean.shape[-1], noise))
        ).distribution


def build_kernel(source: Any, d: int, aug_batch_shape: Any, train_X: Tensor) -> nn.Module:
    """Convert supported port kernels, retaining parameter aliases in trees."""
    from botorch_rs.kernels import (
        AdditiveKernel,
        CosineKernel,
        LinearKernel,
        MaternKernel,
        PeriodicKernel,
        PolynomialKernel,
        ProductKernel,
        RBFKernel,
        RQKernel,
        ScaleKernel,
    )
    from botorch_rs.models.utils.gpytorch_modules import get_covar_module_with_dim_scaled_prior

    if source is None:
        source = get_covar_module_with_dim_scaled_prior(d, batch_shape=aug_batch_shape)
    if isinstance(source, nn.Module):
        return source.to(train_X)
    memo: dict[int, TorchKernel] = {}

    def convert(kernel: Any) -> TorchKernel:
        if id(kernel) in memo:
            return memo[id(kernel)]
        if type(kernel) in (RBFKernel, MaternKernel):
            result: TorchKernel = TorchStationaryKernel(kernel, train_X)
        elif type(kernel) in (
            RQKernel,
            PeriodicKernel,
            CosineKernel,
            LinearKernel,
            PolynomialKernel,
        ):
            result = TorchAlgebraKernel(kernel, train_X)
        elif type(kernel) is ScaleKernel:
            result = TorchScaleKernel(kernel, convert(kernel.base_kernel), train_X)
        elif type(kernel) in (AdditiveKernel, ProductKernel):
            result = TorchCompositeKernel(
                [convert(k) for k in kernel.kernels],
                "add" if type(kernel) is AdditiveKernel else "product",
                train_X,
            )
        else:
            raise UnsupportedError(
                "Torch backend kernels currently support RBF, Matérn, RQ, Periodic, Cosine, Linear, Polynomial, Scale, Additive and Product; a custom torch.nn.Module kernel is also accepted."
            )
        memo[id(kernel)] = result
        return result

    return convert(source)


def build_mean(source: Any, aug_batch_shape: Any, train_X: Tensor) -> nn.Module:
    from botorch_rs.means import ConstantMean, ZeroMean

    if source is None:
        source = ConstantMean(batch_shape=aug_batch_shape)
    if isinstance(source, nn.Module):
        return source.to(train_X)
    if type(source) is ConstantMean:
        return TorchConstantMean(source, train_X)
    if type(source) is ZeroMean:
        return TorchZeroMean(source.batch_shape, train_X)
    raise UnsupportedError(
        "Torch backend means support ConstantMean, ZeroMean or a torch.nn.Module."
    )


def build_likelihood(
    source: Any, train_Yvar_transformed: Tensor | None, aug_batch_shape: Any, train_X: Tensor
) -> nn.Module:
    from botorch_rs.likelihoods import GaussianLikelihood
    from botorch_rs.models.utils.gpytorch_modules import (
        get_gaussian_likelihood_with_lognormal_prior,
    )

    if source is not None and train_Yvar_transformed is not None:
        raise UnsupportedError(
            "Specify either a learned likelihood or train_Yvar for the Torch backend."
        )
    if train_Yvar_transformed is not None:
        # The model already moves the output axis into augmented batch shape:
        # (..., n) for one output; (..., m, n) for multiple outputs.
        noise = train_Yvar_transformed
        return TorchFixedNoiseLikelihood(
            noise.expand(*aug_batch_shape, noise.shape[-1]), aug_batch_shape, train_X
        )
    if source is None:
        source = get_gaussian_likelihood_with_lognormal_prior(batch_shape=aug_batch_shape)
    if type(source) is GaussianLikelihood:
        return TorchGaussianLikelihood(source, train_X)
    raise UnsupportedError(
        "Torch backend likelihoods currently support GaussianLikelihood or train_Yvar."
    )


class TorchNormalize(nn.Module):
    _tensor_backend = "torch"
    indices: Tensor | None
    offset: Tensor
    coefficient: Tensor

    def __init__(self, source: Any, d: int, train_X: Tensor) -> None:
        super().__init__()
        if source._d != d:
            raise ValueError("Normalize's d must equal the training input dimension.")
        self.d = d
        self.batch_shape = torch.Size(source.batch_shape)
        self.learn_bounds = source.learn_bounds
        self.min_range, self.center = source.min_range, source.center
        self.reverse = source.reverse
        self.transform_on_train = source.transform_on_train
        self.transform_on_eval = source.transform_on_eval
        self.transform_on_fantasize = source.transform_on_fantasize
        self.register_buffer(
            "indices",
            None
            if source.indices is None
            else torch.tensor(source.indices, dtype=torch.long, device=train_X.device),
        )
        self.register_buffer("offset", _tensor(source.offset, train_X).clone())
        self.register_buffer("coefficient", _tensor(source.coefficient, train_X).clone())

    def _fit(self, x: Tensor) -> None:
        selected = x if self.indices is None else x.index_select(-1, self.indices)
        batch_ndim = min(len(self.batch_shape), selected.ndim - 2)
        reduce_dims = (*range(selected.ndim - batch_ndim - 2), selected.ndim - 2)
        low = selected.amin(dim=reduce_dims).unsqueeze(-2)
        spread = selected.amax(dim=reduce_dims).unsqueeze(-2) - low
        small = spread < self.min_range
        self.coefficient = torch.where(small, 1.0, spread)
        self.offset = torch.where(small, 0.0, low) + (0.5 - self.center) * self.coefficient

    def fit(self, x: Tensor) -> TorchNormalize:
        if self.learn_bounds and self.transform_on_train and not self.reverse:
            self._fit(x)
        return self

    def _map(self, x: Tensor, inverse: bool) -> Tensor:
        if x.shape[-1] != self.d:
            raise ValueError(f"Normalize expected {self.d} features.")
        selected = x if self.indices is None else x.index_select(-1, self.indices)
        values = (
            selected * self.coefficient + self.offset
            if inverse
            else (selected - self.offset) / self.coefficient
        )
        return values if self.indices is None else x.index_copy(-1, self.indices, values)

    def forward(self, x: Tensor) -> Tensor:
        if not (self.transform_on_train if self.training else self.transform_on_eval):
            return x
        if not self.training and not self.transform_on_fantasize:
            from botorch_rs.models.utils.assorted import fantasize

            if fantasize.on():
                return x
        if self.training and self.learn_bounds and not self.reverse:
            self._fit(x)
        return self._map(x, self.reverse)

    def untransform(self, x: Tensor) -> Tensor:
        return self._map(x, not self.reverse)

    def transform(self, x: Tensor) -> Tensor:
        return self._map(x, self.reverse)

    def preprocess_transform(self, x: Tensor) -> Tensor:
        return self.transform(x) if self.transform_on_train else x


class TorchWarp(_TorchModule):
    """Learned Kumaraswamy CDF, including full input and parameter graphs."""

    _tensor_backend = "torch"
    indices: Tensor
    concentration0: nn.Parameter
    concentration1: nn.Parameter

    def __init__(self, source: Any, d: int, train_X: Tensor) -> None:
        from botorch_rs.constraints import GreaterThan

        super().__init__(train_X)
        if source._d != d:
            raise ValueError("Warp's d must equal the training input dimension.")
        self.d = d
        self.batch_shape = torch.Size(source.batch_shape)
        self._eps = source._eps
        self.reverse = source.reverse
        self.transform_on_train = source.transform_on_train
        self.transform_on_eval = source.transform_on_eval
        self.transform_on_fantasize = source.transform_on_fantasize
        self.register_buffer(
            "indices", torch.tensor(source.indices, dtype=torch.long, device=train_X.device)
        )
        self._normalize = TorchNormalize(source._normalize, d, train_X)
        for i in (0, 1):
            name = f"concentration{i}"
            original = getattr(source, name)
            self.register_parameter(
                name,
                nn.Parameter(
                    _tensor(original, train_X).clone(),
                    requires_grad=not isinstance(original, nn.Parameter) or original.requires_grad,
                ),
            )
            self.add_module(
                f"{name}_constraint",
                TorchConstraint(
                    GreaterThan(source._min_concentration_level, transform=None), train_X
                ),
            )
            self._register_prior(
                f"{name}_prior",
                getattr(source, f"{name}_prior"),
                name,
                2 if self.batch_shape else 1,
            )

    def fit(self, x: Tensor) -> TorchWarp:
        if self.transform_on_train and not self.reverse:
            self._normalize.fit(x)
        return self

    def _warp(self, x: Tensor, inverse: bool) -> Tensor:
        if x.shape[-1] != self.d:
            raise ValueError(f"Warp expected {self.d} features.")
        batch = torch.broadcast_shapes(x.shape[:-2], self.batch_shape)
        x = x.expand(*batch, *x.shape[-2:])
        selected = x.index_select(-1, self.indices)
        c0, c1, eps = self.concentration0, self.concentration1, self._eps
        if inverse:
            values = (((1 - (1 - selected).pow(1 / c0)).pow(1 / c1) - eps) / (1 - 2 * eps)).clamp(
                0, 1
            )
        else:
            selected = (selected * (1 - 2 * eps) + eps).clamp(eps, 1 - eps)
            values = 1 - (1 - selected.pow(c1)).pow(c0)
        return x.index_copy(-1, self.indices, values)

    def transform(self, x: Tensor) -> Tensor:
        if self.reverse:
            return self._normalize.untransform(self._warp(x, True))
        return self._warp(self._normalize.transform(x), False)

    def untransform(self, x: Tensor) -> Tensor:
        if self.reverse:
            return self._warp(self._normalize.transform(x), False)
        return self._normalize.untransform(self._warp(x, True))

    def forward(self, x: Tensor) -> Tensor:
        if not (self.transform_on_train if self.training else self.transform_on_eval):
            return x
        if not self.training and not self.transform_on_fantasize:
            from botorch_rs.models.utils.assorted import fantasize

            if fantasize.on():
                return x
        if self.training and not self.reverse:
            self.fit(x)
        return self.transform(x)

    def preprocess_transform(self, x: Tensor) -> Tensor:
        return self.transform(x) if self.transform_on_train else x

    def _set_concentration(self, i: int, value: Any) -> None:
        self._assign(f"concentration{i}", value)


class TorchInputChain(_TorchModule):
    _tensor_backend = "torch"

    def __init__(self, transforms: dict[str, nn.Module], like: Tensor) -> None:
        super().__init__(like)
        self._transform_names = tuple(transforms)
        for name, transform in transforms.items():
            self.add_module(name, transform)

    def __getitem__(self, name: str) -> nn.Module:
        if name not in self._transform_names:
            raise KeyError(name)
        return getattr(self, name)

    def forward(self, x: Tensor) -> Tensor:
        for name in self._transform_names:
            x = self[name](x)
        return x

    def fit(self, x: Tensor) -> TorchInputChain:
        states = [(self[name], self[name].training) for name in self._transform_names]
        try:
            for module, _ in states:
                module.train()
                x = module(x)
        finally:
            for module, state in states:
                module.train(state)
        return self

    def preprocess_transform(self, x: Tensor) -> Tensor:
        for name in self._transform_names:
            module = self[name]
            method = getattr(module, "preprocess_transform", module)
            x = method(x)
        return x

    def untransform(self, x: Tensor) -> Tensor:
        for name in reversed(self._transform_names):
            module: Any = self[name]
            x = module.untransform(x)
        return x


class TorchStandardize(nn.Module):
    _tensor_backend = "torch"

    def __init__(
        self, m: int, batch_shape: Any, like: Tensor, outputs: Any = None, min_stdv: float = 1e-8
    ) -> None:
        super().__init__()
        self._m = m
        self._batch_shape = torch.Size(batch_shape)
        self._outputs = None if outputs is None else tuple(i % m for i in outputs)
        self._min_stdv = min_stdv
        self.register_buffer("means", like.new_zeros(*batch_shape, 1, m))
        self.register_buffer("stdvs", like.new_ones(*batch_shape, 1, m))
        self._is_trained = False

    @property
    def _stdvs_sq(self) -> Tensor:
        return self.stdvs.square()

    def forward(
        self, Y: Tensor, Yvar: Tensor | None = None, X: Tensor | None = None
    ) -> tuple[Tensor, Tensor | None]:
        if Y.ndim < 2 or Y.shape[-1] != self._m:
            raise ValueError(f"Standardize expected ... x n x {self._m} targets.")
        if self.training:
            self.fit(Y)
        return (Y - self.means) / self.stdvs, None if Yvar is None else Yvar / self._stdvs_sq

    def fit(self, Y: Tensor) -> TorchStandardize:
        if Y.shape[:-2] != self._batch_shape or Y.shape[-2] == 0 or Y.shape[-1] != self._m:
            raise ValueError(
                "Standardize training targets must have its declared shape and at least one observation."
            )
        self.means = Y.nanmean(dim=-2, keepdim=True)
        if Y.shape[-2] == 1:
            std = torch.ones_like(self.means)
        else:
            count = (~Y.isnan()).sum(dim=-2, keepdim=True)
            std = ((Y - self.means).square().nansum(dim=-2, keepdim=True) / (count - 1)).sqrt()
        self.stdvs = torch.where(std >= self._min_stdv, std, 1.0)
        if self._outputs is not None:
            used = torch.zeros(self._m, dtype=torch.bool, device=Y.device)
            used[list(self._outputs)] = True
            self.means = torch.where(used, self.means, 0.0)
            self.stdvs = torch.where(used, self.stdvs, 1.0)
        self._is_trained = True
        return self

    def untransform(
        self, Y: Tensor, Yvar: Tensor | None = None, X: Tensor | None = None
    ) -> tuple[Tensor, Tensor | None]:
        if not self._is_trained:
            raise RuntimeError("Standardize must be fitted before untransform.")
        return Y * self.stdvs + self.means, None if Yvar is None else Yvar * self._stdvs_sq

    def untransform_moments(self, mean: Tensor, covariance: Tensor) -> tuple[Tensor, Tensor]:
        """Undo scaling for mean (...,q,m), covariance (...,m,q,q)."""
        value, _ = self.untransform(mean)
        scale = self.stdvs.movedim(-1, -2).unsqueeze(-1)
        return value, covariance * scale.square()


def build_input_transform(source: Any, d: int, train_X: Tensor) -> nn.Module | None:
    from botorch_rs.models.transforms.input import ChainedInputTransform, Normalize, Warp

    if source is None:
        return None
    if isinstance(source, nn.Module):
        return source.to(train_X)
    if type(source) is Normalize:
        with torch.no_grad():
            return TorchNormalize(source, d, train_X).fit(train_X).eval()
    if type(source) is Warp:
        with torch.no_grad():
            return TorchWarp(source, d, train_X).fit(train_X).eval()
    if type(source) is ChainedInputTransform:
        children: dict[str, nn.Module] = {}
        for name, child in source.items():
            module = build_input_transform(child, d, train_X)
            assert module is not None
            children[name] = module
        with torch.no_grad():
            return TorchInputChain(children, train_X).fit(train_X).eval()
    raise UnsupportedError(
        "Torch backend input transforms currently support Normalize, Warp, ChainedInputTransform or a torch.nn.Module."
    )


def build_outcome_transform(source: Any, m: int, train_Y: Tensor) -> nn.Module | None:
    from botorch_rs.models.transforms.outcome import Standardize
    from botorch_rs.utils.types import DEFAULT

    if source is None:
        return None
    if isinstance(source, nn.Module):
        return source.to(train_Y)
    if source is DEFAULT or type(source).__name__ == "_DefaultType":
        result = TorchStandardize(m, train_Y.shape[:-2], train_Y)
    elif type(source) is Standardize:
        if source._m != m:
            raise ValueError("Standardize's m must equal the model's number of outputs.")
        result = TorchStandardize(
            m, source._batch_shape, train_Y, source._outputs, source._min_stdv
        )
    else:
        raise UnsupportedError(
            "Torch backend outcome transforms currently support Standardize or a torch.nn.Module."
        )
    with torch.no_grad():
        result(train_Y)
    return result.eval()

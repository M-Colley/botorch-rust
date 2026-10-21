"""Port-owned modules for the optional device-preserving Torch backend.

These modules are independent of the CPU/f64 parameter bridge. Raw parameters
are ordinary ``nn.Parameter`` objects; constraints, priors and transforms use
Torch tensor operations, and stationary covariance evaluation dispatches to
Rust's ``tensor_kernel``. Import this module only when selecting that backend.
"""

from __future__ import annotations

import copy
import math
import warnings
from typing import Any, cast

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from botorch_rs import _botorch_rs as _rs
from botorch_rs.exceptions.errors import UnsupportedError
from botorch_rs.exceptions.warnings import NumericsWarning


def _tensor(value: Any, like: Tensor) -> Tensor:
    return torch.as_tensor(value, dtype=like.dtype, device=like.device)


def _subset_indices(indices: Any, m: int, like: Tensor | None = None) -> Tensor:
    selected = torch.as_tensor(
        indices, dtype=torch.long, device=None if like is None else like.device
    )
    if selected.ndim != 1 or not selected.numel():
        raise ValueError("Output indices must be a nonempty one-dimensional sequence.")
    if bool(((selected < -m) | (selected >= m)).any()):
        raise IndexError("Output index out of range.")
    return selected.remainder(m)


def module_log_prior(module: nn.Module, batch_shape: Any) -> Any:
    """Prior terms of a port or registered-prior Torch module.

    Return a Python zero for modules without priors: an empty custom module
    has no parameter or buffer from which to infer its device. Tensor prior
    terms always retain their device and gradient graph.
    """
    method = getattr(module, "log_prior", None)
    if callable(method):
        return method()
    named = getattr(module, "named_priors", None)
    if not callable(named):
        return 0.0
    result: Tensor | None = None
    seen: set[tuple[int, str, int]] = set()
    for name, owner, prior, closure, _ in named():
        key = (id(owner), name.rsplit(".", 1)[-1], id(prior))
        if key in seen:
            continue
        seen.add(key)
        value = prior.log_prob(closure(owner))
        if not isinstance(value, Tensor):
            raise TypeError("A registered Torch prior must return a Tensor log density.")
        owner_shape = getattr(owner, "batch_shape", batch_shape)
        event_ndim = max(0, value.ndim - len(owner_shape))
        if event_ndim:
            value = value.sum(tuple(range(-event_ndim, 0)))
        if result is None:
            result = value.new_zeros(batch_shape)
        result = result + value
    return 0.0 if result is None else result


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
    has_rsample = True

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

    def _extended_shape(self, sample_shape: Any = ()) -> torch.Size:
        return torch.Size(sample_shape)

    def rsample(self, sample_shape: Any = ()) -> Tensor:
        """Draw on the prior device; SmoothedBox retains its true tails."""
        shape, p = torch.Size(sample_shape), self.hyperparameters.unbind()
        distributions = torch.distributions
        factories = {
            "Normal": lambda: distributions.Normal(p[0], p[1]),
            "LogNormal": lambda: distributions.LogNormal(p[0], p[1]),
            "Gamma": lambda: distributions.Gamma(p[0], p[1]),
            "Uniform": lambda: distributions.Uniform(p[0], p[1]),
            "HalfCauchy": lambda: distributions.HalfCauchy(p[0]),
            "HalfNormal": lambda: distributions.HalfNormal(p[0]),
            "Beta": lambda: distributions.Beta(p[0], p[1]),
        }
        if self.kind in factories:
            return factories[self.kind]().rsample(shape)
        a, b, sigma = p
        tail_weight = math.sqrt(2 * math.pi) * sigma / (b - a + math.sqrt(2 * math.pi) * sigma)
        # As in the reference, this discrete mixture selection is not a
        # pathwise derivative of the mixture weight. Hyperparameters are
        # fixed prior buffers; sample_all_priors runs under no_grad.
        tails = torch.bernoulli(tail_weight.expand(shape))
        uniform = a + torch.rand(shape, dtype=a.dtype, device=a.device) * (b - a)
        gaussian = torch.randn(shape, dtype=a.dtype, device=a.device) * sigma
        return torch.where(tails > 0, gaussian + torch.where(gaussian < 0, a, b), uniform)

    def sample(self, sample_shape: Any = ()) -> Tensor:
        with torch.no_grad():
            return self.rsample(sample_shape)


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
            path, _, attribute = name.rpartition(".")
            module = self.get_submodule(path) if path else self
            current = getattr(module, attribute)
            if isinstance(current, nn.Parameter):
                if isinstance(module, _TorchModule):
                    module._assign(attribute, value)
                else:
                    with torch.no_grad():
                        current.copy_(_tensor(value, current))
            else:
                setattr(module, attribute, value)
        return self

    def named_priors(self) -> Any:
        yield from tensor_prior_records(self)


def tensor_prior_records(module: nn.Module) -> Any:
    """Registered owned and GPyTorch priors, deduplicated across module aliases."""
    seen: set[tuple[int, str, int]] = set()
    for prefix, child in module.named_modules():
        if isinstance(child, _TorchModule):
            for name, (attribute, _) in child._prior_parameters.items():
                prior = getattr(child, name)

                def closure(owner: Any, attribute: str = attribute) -> Tensor:
                    return getattr(owner, attribute)

                def setting(owner: Any, value: Tensor, attribute: str = attribute) -> None:
                    current = getattr(owner, attribute)
                    if isinstance(current, nn.Parameter):
                        constraint = getattr(
                            owner,
                            f"{attribute}_constraint",
                            getattr(owner, f"raw_{attribute}_constraint", None),
                        )
                        if constraint is not None and not bool(
                            (
                                (value >= constraint.lower_bound)
                                & (value <= constraint.upper_bound)
                            ).all()
                        ):
                            raise ValueError("Sample is out of bounds of its current constraints.")
                        owner._assign(attribute, value)
                    else:
                        setattr(owner, attribute, value)

                key = (id(child), name, id(prior))
                if key not in seen:
                    seen.add(key)
                    yield f"{prefix}.{name}" if prefix else name, child, prior, closure, setting
        else:
            named = getattr(child, "named_priors", None)
            if not callable(named):
                continue
            for name, owner, prior, get, set_value in named():
                key = (id(owner), name.rsplit(".", 1)[-1], id(prior))
                if key not in seen:
                    seen.add(key)
                    yield f"{prefix}.{name}" if prefix else name, owner, prior, get, set_value


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
        from botorch_rs.kernels import AdditiveKernel

        return cast(
            TorchCompositeKernel, AdditiveKernel(cast(Any, self), cast(Any, other), backend="torch")
        )

    def __mul__(self, other: TorchKernel) -> TorchCompositeKernel:
        from botorch_rs.kernels import ProductKernel

        return cast(
            TorchCompositeKernel, ProductKernel(cast(Any, self), cast(Any, other), backend="torch")
        )

    def _forward_selected(
        self, x1: Tensor, x2: Tensor, diag: bool, last_dim_is_batch: bool
    ) -> Tensor:
        raise NotImplementedError


class TorchStationaryKernel(TorchKernel):
    _tensor_backend = "torch"
    has_lengthscale = True
    _saas_fixed_tau: Tensor | None

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
        if getattr(source, "_saas", None) is not None:
            self.configure_saas_prior(
                source._saas_helper._tau,
                log_scale=type(source.raw_lengthscale_constraint).__name__
                == "LogTransformedInterval",
            )
            self._assign("raw_lengthscale", raw)
            if source._saas_helper._tau is None:
                self._assign("raw_tau", source.raw_tau)

    def configure_saas_prior(
        self, tau: Any = None, log_scale: bool = True
    ) -> TorchStationaryKernel:
        """Install the coupled SAAS prior, retaining its device and parameter graph."""
        from botorch_rs.constraints import Interval
        from botorch_rs.models.map_saas import _double_half_cauchy
        from botorch_rs.utils.constraints import LogTransformedInterval

        if hasattr(self, "lengthscale_prior") or hasattr(self, "inv_lengthscale_prior"):
            raise UnsupportedError("base_kernel must not specify a lengthscale or SAAS prior")
        fixed = None
        if tau is not None:
            fixed = torch.as_tensor(tau).to(self._anchor)
            if not isinstance(tau, (int, float)) and fixed.shape != self.batch_shape:
                raise ValueError(f"Expected tau shape {self.batch_shape}, got {fixed.shape}.")
            if not bool(torch.isfinite(fixed).all() & (fixed > 0).all()):
                raise ValueError("SAAS tau must be finite and positive.")
        interval = LogTransformedInterval if log_scale else Interval

        def constraint(source: Any) -> TorchConstraint:
            result = TorchConstraint(source, self._anchor)
            # GPyTorch creates these bounds in the Torch default dtype,
            # even when add_saas_prior receives an already-double kernel.
            # Preserve that factory precision before device/dtype promotion.
            result.lower_bound = torch.tensor(source.lower_bound).to(self._anchor)
            result.upper_bound = torch.tensor(source.upper_bound).to(self._anchor)
            if result.transform_name == "exp":
                result.raw_bounds = (
                    float(result.lower_bound.log()),
                    float(result.upper_bound.log()),
                )
            return result

        self.raw_lengthscale_constraint = constraint(interval(0.01, 1e4, initial_value=1))
        self._assign(
            "raw_lengthscale",
            self.raw_lengthscale.new_zeros(self.raw_lengthscale.shape)
            if log_scale
            else self.raw_lengthscale_constraint.inverse_transform(
                torch.ones_like(self.raw_lengthscale)
            ),
        )
        if tau is None:
            self.raw_tau_constraint = constraint(interval(1e-3, 10, initial_value=0.1))
            initial = self._anchor.new_full(self.batch_shape, 0.1)
            self.raw_tau = nn.Parameter(self.raw_tau_constraint.inverse_transform(initial))
            self._register_prior("tau_prior", _double_half_cauchy(0.1), "tau")
            self.register_buffer("_saas_fixed_tau", None)
        else:
            self.register_buffer("_saas_fixed_tau", fixed)
        self._register_prior(
            "inv_lengthscale_prior", _double_half_cauchy(1.0), "inv_lengthscale", 2
        )
        return self

    @property
    def tau(self) -> Tensor:
        fixed = self._saas_fixed_tau
        return self.raw_tau_constraint.transform(self.raw_tau) if fixed is None else fixed

    @tau.setter
    def tau(self, value: Any) -> None:
        if self._saas_fixed_tau is not None:
            raise UnsupportedError(
                "A fixed SAAS tau cannot be set through its learned-prior closure."
            )
        constraint = self.raw_tau_constraint
        bounded = _tensor(value, self.raw_tau).clamp(
            constraint.lower_bound + 1e-8, constraint.upper_bound - 1e-8
        )
        self._assign("raw_tau", bounded, constraint)

    @property
    def inv_lengthscale(self) -> Tensor:
        return self.tau[..., None, None] / self.lengthscale.square()

    @inv_lengthscale.setter
    def inv_lengthscale(self, value: Any) -> None:
        constraint = self.raw_lengthscale_constraint
        self.lengthscale = (
            (self.tau[..., None, None] / _tensor(value, self.raw_lengthscale))
            .sqrt()
            .clamp(constraint.lower_bound + 1e-8, constraint.upper_bound - 1e-8)
        )

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
        IndexKernel,
        LinearKernel,
        MaternKernel,
        MultitaskKernel,
        PeriodicKernel,
        PolynomialKernel,
        ProductKernel,
        RBFKernel,
        RQKernel,
        ScaleKernel,
    )
    from botorch_rs.models.hierarchical.conditional_kernel_gp import HierarchicalConditionalKernel
    from botorch_rs.models.kernels.categorical import CategoricalKernel
    from botorch_rs.models.kernels.contextual_lcea import LCEAKernel
    from botorch_rs.models.kernels.contextual_sac import SACKernel
    from botorch_rs.models.kernels.downsampling import DownsamplingKernel
    from botorch_rs.models.kernels.exponential_decay import ExponentialDecayKernel
    from botorch_rs.models.kernels.heterogeneous_multitask import MultiTaskConditionalKernel
    from botorch_rs.models.kernels.infinite_width_bnn import InfiniteWidthBNNKernel
    from botorch_rs.models.kernels.integrated_white_noise import IntegratedWhiteNoiseKernel
    from botorch_rs.models.kernels.linear_truncated_fidelity import LinearTruncatedFidelityKernel
    from botorch_rs.models.kernels.orthogonal_additive_kernel import OrthogonalAdditiveKernel
    from botorch_rs.models.kernels.positive_index import PositiveIndexKernel
    from botorch_rs.models.utils.gpytorch_modules import get_covar_module_with_dim_scaled_prior

    if source is None:
        source = get_covar_module_with_dim_scaled_prior(d, batch_shape=aug_batch_shape)
    if isinstance(source, nn.Module):
        return source.to(train_X)
    memo: dict[int, TorchKernel] = {}

    def convert(kernel: Any) -> TorchKernel:
        if id(kernel) in memo:
            return memo[id(kernel)]
        if type(kernel) in (IndexKernel, MultitaskKernel):
            from botorch_rs.models._torch_backend_core_kernels import tensor_kernel_class

            options = {"_source": kernel, "_like": train_X}
            if type(kernel) is MultitaskKernel:
                options["_convert"] = convert
            result = tensor_kernel_class(type(kernel))(**options)
        elif type(kernel) in (RBFKernel, MaternKernel):
            result = TorchStationaryKernel(kernel, train_X)
        elif type(kernel) is IntegratedWhiteNoiseKernel:
            from botorch_rs.models._torch_backend_wiener import TorchIntegratedWhiteNoiseKernel

            result = TorchIntegratedWhiteNoiseKernel(_source=kernel, _like=train_X)
        elif type(kernel) in (
            CategoricalKernel,
            DownsamplingKernel,
            ExponentialDecayKernel,
            LinearTruncatedFidelityKernel,
            PositiveIndexKernel,
        ):
            from botorch_rs.models import _torch_backend_kernel_catalogue as catalogue

            factory = getattr(catalogue, f"Torch{type(kernel).__name__}")
            options = {"_source": kernel, "_like": train_X}
            if type(kernel) in (
                DownsamplingKernel,
                ExponentialDecayKernel,
                LinearTruncatedFidelityKernel,
            ):
                options["_convert"] = convert
            result = factory(**options)
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
        elif type(kernel) in (SACKernel, LCEAKernel):
            from botorch_rs.models._torch_backend_contextual import TorchLCEAKernel, TorchSACKernel

            factory = TorchSACKernel if type(kernel) is SACKernel else TorchLCEAKernel
            result = factory(_source=kernel, _like=train_X)
        elif type(kernel) is OrthogonalAdditiveKernel:
            from botorch_rs.models._torch_backend_additive import TorchOrthogonalAdditiveKernel

            result = TorchOrthogonalAdditiveKernel(_source=kernel, _like=train_X)
        elif type(kernel) is HierarchicalConditionalKernel:
            from botorch_rs.models._torch_backend_hierarchical import (
                TorchHierarchicalConditionalKernel,
            )

            result = TorchHierarchicalConditionalKernel(_source=kernel, _like=train_X)
        elif type(kernel) is MultiTaskConditionalKernel:
            from botorch_rs.models._torch_backend_heterogeneous import (
                TorchMultiTaskConditionalKernel,
            )

            result = TorchMultiTaskConditionalKernel(_source=kernel, _like=train_X)
        elif type(kernel) is InfiniteWidthBNNKernel:
            from botorch_rs.models._torch_backend_neural import TorchInfiniteWidthBNNKernel

            result = TorchInfiniteWidthBNNKernel(_source=kernel, _like=train_X)
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

    if isinstance(source, nn.Module):
        from botorch_rs.models._torch_gaussian_likelihood import is_gpytorch_gaussian

        if isinstance(
            source, (TorchGaussianLikelihood, TorchFixedNoiseLikelihood)
        ) or is_gpytorch_gaussian(source):
            return source
        raise UnsupportedError("Exact inference requires a Gaussian likelihood marginal.")
    if source is not None:
        if type(source) is GaussianLikelihood:
            return TorchGaussianLikelihood(source, train_X)
        raise UnsupportedError("Torch backend likelihoods require a supported Gaussian marginal.")
    if train_Yvar_transformed is not None:
        # The model already moves the output axis into augmented batch shape:
        # (..., n) for one output; (..., m, n) for multiple outputs.
        noise = train_Yvar_transformed
        return TorchFixedNoiseLikelihood(
            noise.expand(*aug_batch_shape, noise.shape[-1]), aug_batch_shape, train_X
        )
    if source is None:
        source = get_gaussian_likelihood_with_lognormal_prior(batch_shape=aug_batch_shape)
    return TorchGaussianLikelihood(source, train_X)


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
        if source._model is None:
            from botorch_rs.models.transforms._torch_input import warp_parameters

            warp_parameters(source, train_X)
        for i in (0, 1):
            name = f"concentration{i}"
            original = getattr(source, name)
            self.register_parameter(
                name,
                original
                if isinstance(original, nn.Parameter)
                and original.dtype == train_X.dtype
                and original.device == train_X.device
                else nn.Parameter(
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
    _is_linear = True

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
        from botorch_rs.models.transforms._torch_standardize import apply_standardize

        if Y.ndim < 2 or Y.shape[-1] != self._m:
            raise ValueError(f"Standardize expected ... x n x {self._m} targets.")
        if self.training:
            self.fit(Y)
        return apply_standardize(Y, Yvar, self.means, self.stdvs, self._stdvs_sq)

    def fit(self, Y: Tensor) -> TorchStandardize:
        from botorch_rs.models.transforms._torch_standardize import standardize_statistics

        if Y.shape[:-2] != self._batch_shape or Y.shape[-2] == 0 or Y.shape[-1] != self._m:
            raise ValueError(
                "Standardize training targets must have its declared shape and at least one observation."
            )
        self.means, self.stdvs = standardize_statistics(Y, self._min_stdv)
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
        from botorch_rs.models.transforms._torch_standardize import apply_standardize

        if not self._is_trained:
            raise RuntimeError("Standardize must be fitted before untransform.")
        return apply_standardize(Y, Yvar, self.means, self.stdvs, self._stdvs_sq, inverse=True)

    def untransform_moments(self, mean: Tensor, covariance: Tensor) -> tuple[Tensor, Tensor]:
        """Undo scaling for mean (...,q,m), covariance (...,m,q,q)."""
        value, _ = self.untransform(mean)
        scale = self.stdvs.movedim(-1, -2).unsqueeze(-1)
        return value, covariance * scale.square()

    def untransform_posterior(self, posterior: Any, X: Any = None) -> Any:
        from botorch_rs.posteriors._torch_backend import TorchGPyTorchPosterior
        from botorch_rs.posteriors.transformed import TransformedPosterior

        if not self._is_trained:
            raise RuntimeError("Standardize must be fitted before untransforming a posterior.")
        means, stdvs = self.means.clone(), self.stdvs.clone()
        if isinstance(posterior, TorchGPyTorchPosterior):
            q = posterior.mean.shape[-2]
            scale = stdvs.movedim(-1, -2).expand(*stdvs.shape[:-2], self._m, q).flatten(-2)
            covariance = posterior._covariance_tensor * scale.unsqueeze(-1) * scale.unsqueeze(-2)
            return TorchGPyTorchPosterior(posterior.mean * stdvs + means, covariance)
        return TransformedPosterior(
            posterior,
            sample_transform=lambda samples: samples * stdvs + means,
            mean_transform=lambda mean, variance: mean * stdvs + means,
            variance_transform=lambda mean, variance: variance * stdvs.square(),
        )

    def subset_output(self, idcs: Any) -> TorchStandardize:
        indices = _subset_indices(idcs, self._m, self.means)
        outputs = (
            None
            if self._outputs is None
            else [j for j, i in enumerate(indices.tolist()) if i in self._outputs]
        )
        result = TorchStandardize(
            indices.numel(), self._batch_shape, self.means, outputs, self._min_stdv
        )
        result.means = self.means.index_select(-1, indices).clone()
        result.stdvs = self.stdvs.index_select(-1, indices).clone()
        result._is_trained = self._is_trained
        return result.train(self.training)


def _elementwise_outcome(value: Tensor, kind: str, power: Any, inverse: bool) -> Tensor:
    if kind == "Log":
        return value.exp() if inverse else value.log()
    if kind == "Power":
        return value.pow(1 / power if inverse else power)
    # Bilog is C1 with slope one at zero. sign(x)*log1p(abs(x)) has the
    # incorrect Torch derivative zero there. Mask each inactive branch to
    # keep all its operations in their domain, including repeated backward.
    # Scalar clamp_min changed its equality derivative from one to zero in
    # Torch 2.14. Select the inactive branch explicitly, retaining the true
    # derivative one at the origin on every supported Torch version.
    positive = value.masked_fill(value < 0, 0)
    negative = (-value).masked_fill(value > 0, 0)
    if inverse:
        return torch.where(value >= 0, positive.expm1(), -negative.expm1())
    return torch.where(value >= 0, positive.log1p(), -negative.log1p())


class TorchElementwiseOutcome(nn.Module):
    _tensor_backend = "torch"
    _is_linear = False
    _m: int

    def __init__(self, source: Any, m: int) -> None:
        super().__init__()
        self.kind = getattr(source, "_name", type(source).__name__)
        self._m = m
        power = getattr(source, "power", 1)
        self.power = power if isinstance(power, Tensor) else float(power)
        if self.kind == "Power" and (
            not bool(torch.isfinite(torch.as_tensor(self.power)).all())
            or bool((torch.as_tensor(self.power) == 0).any())
        ):
            raise ValueError(
                "Power must be finite and nonzero for an invertible outcome transform."
            )
        indices = getattr(source, "_outputs", None)
        if indices is not None and any(i < -m or i >= m for i in indices):
            raise IndexError("Outcome transform output index out of range.")
        self._outputs = None if indices is None else tuple(i % m for i in indices)

    def _select(self, original: Tensor, transformed: Tensor) -> Tensor:
        if self._outputs is None:
            return transformed
        used = torch.zeros(self._m, dtype=torch.bool, device=original.device)
        used[list(self._outputs)] = True
        return torch.where(used, transformed, original)

    def _map(self, Y: Tensor, Yvar: Tensor | None, inverse: bool) -> tuple[Tensor, Tensor | None]:
        if Y.ndim < 2 or Y.shape[-1] != self._m:
            raise ValueError(f"Outcome transform expected ... x n x {self._m} values.")
        if Yvar is not None and self.kind != "Log":
            raise NotImplementedError(
                f"{self.kind} does not support transforming observation noise."
            )
        if self.kind == "Log" and not inverse:
            selected = Y if self._outputs is None else Y[..., list(self._outputs)]
            if not bool((selected > 0).all()):
                raise ValueError("Log-transformed outcomes must be positive.")
        safe_values = self._select(torch.ones_like(Y), Y)
        transformed = _elementwise_outcome(safe_values, self.kind, self.power, inverse)
        output = self._select(Y, transformed)
        variance = None
        if Yvar is not None:
            transformed_variance = (
                Yvar * (2 * Y).exp() if inverse else Yvar / Y.clamp_min(1e-8).square()
            )
            variance = self._select(Yvar, transformed_variance)
        return output, variance

    def forward(
        self, Y: Tensor, Yvar: Tensor | None = None, X: Tensor | None = None
    ) -> tuple[Tensor, Tensor | None]:
        return self._map(Y, Yvar, False)

    def fit(self, Y: Tensor) -> TorchElementwiseOutcome:
        return self

    def untransform(
        self, Y: Tensor, Yvar: Tensor | None = None, X: Tensor | None = None
    ) -> tuple[Tensor, Tensor | None]:
        return self._map(Y, Yvar, True)

    def untransform_posterior(self, posterior: Any, X: Any = None) -> Any:
        from botorch_rs.posteriors.transformed import TransformedPosterior

        kind, power, outputs, m = self.kind, self.power, self._outputs, self._m

        def select(original: Tensor, value: Tensor) -> Tensor:
            if outputs is None:
                return value
            used = torch.zeros(m, dtype=torch.bool, device=original.device)
            used[list(outputs)] = True
            return torch.where(used, value, original)

        def samples(value: Tensor) -> Tensor:
            safe_values = select(torch.ones_like(value), value)
            return select(value, _elementwise_outcome(safe_values, kind, power, True))

        def mean(value: Tensor, variance: Tensor) -> Tensor:
            return select(value, (value + 0.5 * variance).exp())

        def variance(value: Tensor, var: Tensor) -> Tensor:
            return select(var, var.expm1() * (2 * value + var).exp())

        return TransformedPosterior(
            posterior, samples, mean if kind == "Log" else None, variance if kind == "Log" else None
        )

    def subset_output(self, idcs: Any) -> TorchElementwiseOutcome:
        indices = _subset_indices(idcs, self._m)
        result = copy.deepcopy(self)
        result._m = indices.numel()
        result._outputs = (
            None
            if self._outputs is None
            else tuple(j for j, i in enumerate(indices.tolist()) if i in self._outputs)
        )
        return result


class TorchOutcomeChain(nn.Module):
    _tensor_backend = "torch"

    def __init__(self, transforms: dict[str, nn.Module]) -> None:
        super().__init__()
        self._transform_names = tuple(transforms)
        for name, transform in transforms.items():
            self.add_module(name, transform)

    def __getitem__(self, name: str) -> Any:
        if name not in self._transform_names:
            raise KeyError(name)
        return getattr(self, name)

    @property
    def _is_linear(self) -> bool:
        return all(getattr(self[name], "_is_linear", False) for name in self._transform_names)

    @property
    def _requires_X(self) -> bool:
        return any(getattr(self[name], "_requires_X", False) for name in self._transform_names)

    def forward(
        self, Y: Tensor, Yvar: Tensor | None = None, X: Tensor | None = None
    ) -> tuple[Tensor, Tensor | None]:
        for name in self._transform_names:
            Y, Yvar = self[name](Y, Yvar, X=X)
        return Y, Yvar

    def fit(self, Y: Tensor, X: Tensor | None = None) -> TorchOutcomeChain:
        states = [(self[name], self[name].training) for name in self._transform_names]
        try:
            for module, _ in states:
                module.train()
                Y, _ = module(Y, X=X)
        finally:
            for module, state in states:
                module.train(state)
        return self

    def untransform(
        self, Y: Tensor, Yvar: Tensor | None = None, X: Tensor | None = None
    ) -> tuple[Tensor, Tensor | None]:
        for name in reversed(self._transform_names):
            Y, Yvar = self[name].untransform(Y, Yvar, X=X)
        return Y, Yvar

    def untransform_posterior(self, posterior: Any, X: Any = None) -> Any:
        for name in reversed(self._transform_names):
            posterior = self[name].untransform_posterior(posterior, X=X)
        return posterior

    def subset_output(self, idcs: Any) -> TorchOutcomeChain:
        result = TorchOutcomeChain(
            {name: self[name].subset_output(idcs) for name in self._transform_names}
        )
        return result.train(self.training)


def build_input_transform(
    source: Any, d: int | None, train_X: Tensor, *, fit: bool = True
) -> nn.Module | None:
    """Register the actual transform; ``fit=False`` attaches without training data.

    In that mode ``train_X`` supplies only dtype/device and may be a scalar
    parameter from an external approximate model.
    """
    from botorch_rs.models.transforms.input import InputTransform

    if source is None:
        return None
    if isinstance(source, InputTransform):
        from botorch_rs.models.transforms._torch_input_module import TorchPublicInputTransform

        result = TorchPublicInputTransform(source, train_X).to(train_X)
        if not fit:
            return result
        with torch.no_grad():
            return result.fit(train_X).eval()
    if isinstance(source, nn.Module):
        return source.to(train_X)
    raise UnsupportedError(
        "Torch backend input transforms must be port InputTransform objects or torch.nn.Module instances."
    )


def _canonical_outcome_module(source: Any, name: str) -> bool:
    return (
        isinstance(source, nn.Module)
        and type(source).__call__ is nn.Module.__call__
        and all(
            getattr(getattr(type(source), method, None), "__module__", None)
            == "botorch.models.transforms.outcome"
            and getattr(getattr(type(source), method, None), "__qualname__", None)
            == f"{name}.{method}"
            for method in ("forward", "untransform", "untransform_posterior")
        )
    )


def build_outcome_transform(
    source: Any, m: int, train_Y: Tensor | None, train_X: Tensor | None = None
) -> nn.Module | None:
    from botorch_rs.models.transforms.outcome import (
        Bilog,
        ChainedOutcomeTransform,
        Log,
        Power,
        Standardize,
        StratifiedStandardize,
    )
    from botorch_rs.utils.types import DEFAULT

    if source is None:
        return None
    like = train_Y if train_Y is not None else train_X
    if isinstance(source, nn.Module) and getattr(type(source), "_torch_outcome_factory", False):
        source.to(like)
        if train_Y is not None:
            source.train()
            with torch.no_grad():
                source(train_Y, X=train_X)
        return source.eval()
    if source is DEFAULT or type(source).__name__ == "_DefaultType":
        if like is None:
            raise ValueError("An outcome transform requires train_Y or train_X for dtype/device.")
        result = TorchStandardize(m, like.shape[:-2], like)
    elif (
        isinstance(source, StratifiedStandardize)
        and type(source).forward is StratifiedStandardize.forward
        and type(source).untransform is StratifiedStandardize.untransform
    ) or (
        isinstance(source, nn.Module)
        and hasattr(source, "strata_mapping")
        and type(source).__call__ is nn.Module.__call__
        and all(
            getattr(getattr(type(source), name, None), "__module__", None)
            == "botorch.models.transforms.outcome"
            and getattr(getattr(type(source), name, None), "__qualname__", None)
            == f"StratifiedStandardize.{name}"
            for name in ("forward", "untransform", "untransform_posterior")
        )
    ):
        from types import SimpleNamespace

        from botorch_rs.models.transforms._torch_standardize import TensorStandardizeState

        if m != 1:
            raise ValueError("StratifiedStandardize requires a single observed output.")
        if train_X is None:
            raise ValueError(
                "StratifiedStandardize requires train_X when building a model transform."
            )
        specification: Any = source
        if not isinstance(source, StratifiedStandardize):
            canonical: Any = source
            mapping = canonical.strata_mapping
            tasks = torch.nonzero(~mapping.isnan(), as_tuple=True)[0].tolist()
            specification = SimpleNamespace(
                _m=1,
                _outputs=None,
                _batch_shape=tuple(canonical._batch_shape),
                _min_stdv=canonical._min_stdv,
                _stratification_idx=canonical._stratification_idx,
                _task_values=tasks,
                _task_values_dtype=None,
                _strata_dtype=mapping.dtype if mapping.is_floating_point() else None,
                training=True,
            )
        assert like is not None
        stratified = TensorStandardizeState(specification, like)
        # GP constructors explicitly train their supplied outcome transform,
        # even if the descriptor was already fitted and evaluated.
        if train_Y is not None:
            with torch.no_grad():
                stratified.fit(train_Y, X=train_X)
        elif bool(source._is_trained):
            for name in ("means", "stdvs", "_stdvs_sq"):
                setattr(
                    stratified,
                    name,
                    torch.as_tensor(
                        getattr(source, name), device=like.device, dtype=like.dtype
                    ).clone(),
                )
            stratified._is_trained.fill_(True)
        return stratified.eval()
    elif (
        isinstance(source, Standardize)
        and type(source).forward is Standardize.forward
        and type(source).untransform is Standardize.untransform
        and type(source).__call__ is Standardize.__call__
    ) or (
        isinstance(source, nn.Module)
        and all(
            hasattr(source, attribute)
            for attribute in ("_m", "_batch_shape", "_min_stdv", "_outputs", "means", "stdvs")
        )
        and not hasattr(source, "_stratification_idx")
        and type(source).__call__ is nn.Module.__call__
        and all(
            getattr(getattr(type(source), name, None), "__module__", None)
            == "botorch.models.transforms.outcome"
            and getattr(getattr(type(source), name, None), "__qualname__", None)
            == f"Standardize.{name}"
            for name in ("forward", "untransform", "untransform_posterior")
        )
    ):
        if source._m != m:
            raise ValueError("Standardize's m must equal the model's number of outputs.")
        minimum_stdv: Any = source._min_stdv
        if like is None:
            raise ValueError("An outcome transform requires train_Y or train_X for dtype/device.")
        result = TorchStandardize(
            m, source._batch_shape, like, source._outputs, float(minimum_stdv)
        )
        if train_Y is None and bool(source._is_trained):
            result.means = torch.as_tensor(
                source.means, device=like.device, dtype=like.dtype
            ).clone()
            result.stdvs = torch.as_tensor(
                source.stdvs, device=like.device, dtype=like.dtype
            ).clone()
            result._is_trained = True
    elif type(source) in (Log, Bilog, Power) or any(
        _canonical_outcome_module(source, name) for name in ("Log", "Bilog", "Power")
    ):
        return TorchElementwiseOutcome(source, m).eval()
    elif type(source) is ChainedOutcomeTransform or _canonical_outcome_module(
        source, "ChainedOutcomeTransform"
    ):
        children: dict[str, nn.Module] = {}
        values = train_Y
        for name, child in source.items():
            transform = build_outcome_transform(child, m, values, train_X=train_X)
            assert transform is not None
            children[name] = transform
            if values is not None:
                with torch.no_grad():
                    values, _ = transform(values, X=train_X)
        return TorchOutcomeChain(children).eval()
    elif isinstance(source, nn.Module):
        if all(
            hasattr(source, name) for name in ("_m", "_batch_shape", "_min_stdv", "means", "stdvs")
        ) and not (
            callable(getattr(source, "fit", None))
            and getattr(source, "_tensor_backend", None) == "torch"
        ):
            raise UnsupportedError(
                "Custom Standardize-like Torch modules must provide fit(Y) and "
                "_tensor_backend='torch' for their training-statistics graph; "
                "overridden behavior is not canonicalized."
            )
        return source.to(like)
    else:
        raise UnsupportedError(
            "Torch backend outcome transforms support Standardize, Log, Bilog, Power, ChainedOutcomeTransform or a torch.nn.Module."
        )
    if train_Y is not None:
        with torch.no_grad():
            result(train_Y)
    return result.eval()

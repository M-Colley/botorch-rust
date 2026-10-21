"""Cooperative Torch protocols on the actual public input-transform object.

The public numerical methods remain in charge. This module is imported only
after Torch has been imported by the application; array-only programs retain
the native descriptors and never acquire a Torch dependency.
"""

from __future__ import annotations

from typing import Any, SupportsIndex

import torch
from torch import Tensor, nn

from botorch_rs._arrays import to_flat
from botorch_rs._buffers import _logical_dtype
from botorch_rs.models.transforms import _torch_input

_types: dict[type, type] = {}


def _rebuild(public: Any, state: dict[str, Any]) -> Any:
    result: Any = object.__new__(module_type(public))
    public.__setstate__(result, state)
    return result


def module_type(public: type) -> type:
    if issubclass(public, TorchInputModule):
        return public
    if public in _types:
        return _types[public]

    def initialize(self: Any, *args: Any, **kwargs: Any) -> None:
        public.__init__(self, *args, **kwargs)
        self._finish_module()

    initialize._buffer_aware = True  # type: ignore[attr-defined]
    attributes: dict[str, Any] = {
        "__module__": __name__,
        "__init__": initialize,
        "_native_port_descriptor": public,
    }
    # Public classes own methods with the same names as Module protocols.
    # In particular Affine/Warp._apply is a numerical map, so dispatch it
    # separately from Module._apply used by dtype/device migration.
    for name in (
        "__call__",
        "__getattr__",
        "__setattr__",
        "__reduce_ex__",
        "train",
        "eval",
        "parameters",
        "named_parameters",
        "buffers",
        "named_buffers",
        "_apply",
        "register_parameter",
        "_save_to_state_dict",
        "_load_from_state_dict",
    ):
        attributes[name] = getattr(TorchInputModule, name)
    result = type(public.__name__, (public, TorchInputModule), attributes)
    _types[public] = result
    return result


def allocate(public: type) -> Any:
    result: Any = object.__new__(module_type(public))
    nn.Module.__init__(result)
    return result


def promote(value: Any) -> Any:
    """Register an earlier native child in place, preserving caller aliases."""
    if isinstance(value, nn.Module):
        return value
    state = value.__dict__.copy()
    public = type(value)
    value.__class__ = module_type(public)
    nn.Module.__init__(value)
    value.__dict__.update(state)
    value._finish_module()
    return value


class TorchInputModule(nn.Module):
    _native_port_descriptor: Any

    def __getattr__(self, name: str) -> Any:
        try:
            return nn.Module.__getattr__(self, name)
        except AttributeError:
            public = type(self)._native_port_descriptor
            getter = getattr(public, "__getattr__", None)
            if getter is not None:
                return getter(self, name)
            raise

    def __setattr__(self, name: str, value: Any) -> None:
        descriptor = getattr(type(self)._native_port_descriptor, name, None)
        if isinstance(descriptor, property) and descriptor.fset is not None:
            descriptor.fset(self, value)
            parameter = self.__dict__.get("_direct_parameters", {}).get(name)
            if parameter is not None and "_parameters" in self.__dict__:
                self._parameters[name] = parameter
            return
        nn.Module.__setattr__(self, name, value)

    def __reduce_ex__(self, protocol: SupportsIndex) -> Any:
        return _rebuild, (type(self)._native_port_descriptor, self.__getstate__())

    def register_parameter(self, name: str, parameter: nn.Parameter | None) -> None:
        nn.Module.register_parameter(self, name, parameter)
        direct = self.__dict__.get("_direct_parameters", {})
        if name in direct:
            direct[name] = parameter

    def _put_buffer(self, name: str, value: Any, *, integer: bool = False) -> None:
        if value is not None and not isinstance(value, Tensor):
            dtype = torch.long if integer else getattr(torch, _logical_dtype(self, name))
            value = torch.as_tensor(value, dtype=dtype)
        self.__dict__.pop(name, None)
        self._buffers[name] = value

    def _refresh_affine_buffers(self, dtype: Any = None) -> None:
        if not self.__dict__.get("_native_affine_dormant", False):
            return
        # Native array state remains authoritative until a Tensor call or an
        # explicit migration. Introspection must not round a float64 native
        # coefficient to a default float32 snapshot and export it back.
        for name, value in (("_coefficient", self.coefficient), ("_offset", self.offset)):
            data, shape = to_flat(value)
            previous = self._buffers.get(name)
            actual_dtype = dtype or (
                previous.dtype if previous is not None else getattr(torch, _logical_dtype(self))
            )
            self._buffers[name] = torch.tensor(list(data), dtype=actual_dtype).reshape(shape)

    def _finish_module(self) -> None:
        from botorch_rs.constraints import GreaterThan
        from botorch_rs.models._torch_backend_modules import TorchConstraint, TorchPrior
        from botorch_rs.models.transforms.input import (
            AffineInputTransform,
            BatchBroadcastedInputTransform,
            ChainedInputTransform,
            InputPerturbation,
            LearnedFeatureImputation,
            NumericToCategoricalEncoding,
            Warp,
        )

        # Constructor Tensor arguments remain the actual registered objects.
        for name, value in tuple(self.__dict__.items()):
            if isinstance(value, Tensor) and not isinstance(value, nn.Parameter):
                self._put_buffer(name, value)
        if isinstance(self, AffineInputTransform):
            self.__dict__["_native_affine_dormant"] = not isinstance(
                self._buffers.get("_coefficient"), Tensor
            )
            self._refresh_affine_buffers()
        for name in ("indices", "integer_indices", "feature_indices"):
            value = getattr(self, name, None)
            if value is not None and not isinstance(value, dict):
                self._put_buffer(name, value, integer=True)
        for name in ("feature_set", "perturbation_set", "bounds"):
            # Normalize.bounds is a derived property, never a separate buffer.
            if name in self.__dict__ and not callable(self.__dict__[name]):
                self._put_buffer(name, self.__dict__[name])
        if isinstance(self, InputPerturbation):
            self.__dict__.pop("_perturbations", None)
            self._buffers.setdefault("_perturbations", None)
        if isinstance(self, ChainedInputTransform):
            for name, child in self._transforms.items():
                self._modules[name] = promote(child)
        if isinstance(self, BatchBroadcastedInputTransform):
            for index, child in enumerate(self.transforms):
                self.add_module(str(index), promote(child))
        if isinstance(self, Warp) and not isinstance(self._normalize, nn.Module):
            self._normalize = promote(self._normalize)
        if isinstance(self, NumericToCategoricalEncoding):
            modules = {str(i): f for i, f in self.encoders.items() if isinstance(f, nn.Module)}
            if modules:
                self.add_module("_encoder_modules", nn.ModuleDict(modules))
        if isinstance(self, (Warp, LearnedFeatureImputation)):
            existing = self.__dict__.get("_direct_parameters", {}).copy()
            like = next(
                (
                    x
                    for x in self._buffers.values()
                    if isinstance(x, Tensor) and x.is_floating_point()
                ),
                None,
            )
            if not isinstance(like, Tensor):
                like = torch.empty((), dtype=getattr(torch, _logical_dtype(self)))
            if isinstance(self, Warp):
                parameters = _torch_input.warp_parameters(self, like)
            else:
                _torch_input.imputation_parameter(self, like)
                parameters = self.__dict__["_direct_parameters"]
            for name, parameter in parameters.items():
                self._parameters[name] = parameter
            self.__dict__["_native_default_parameters"] = {
                name: parameter for name, parameter in parameters.items() if name not in existing
            }
        if isinstance(self, LearnedFeatureImputation):
            parameter = self.raw_imputation_values
            tasks = sorted(self.feature_indices)
            self._buffers["_task_values"] = torch.tensor(
                tasks, dtype=torch.long, device=parameter.device
            )
            missing = torch.ones(
                self.num_tasks, self.d + 1, dtype=torch.bool, device=parameter.device
            )
            missing[:, -1] = False
            for row, task in enumerate(tasks):
                missing[row, self.feature_indices[task]] = False
            self._buffers["missing_mask"] = missing
            if self.bounds is not None:
                bounds = self.bounds.to(parameter)
                constraint = TorchConstraint(None, parameter)
                constraint.kind, constraint.transform_name, constraint.enforced = (
                    "Interval",
                    "sigmoid",
                    True,
                )
                constraint.lower_bound = torch.cat((bounds[0], parameter.new_zeros(1))).repeat(
                    self.num_tasks
                )
                constraint.upper_bound = torch.cat((bounds[1], parameter.new_ones(1))).repeat(
                    self.num_tasks
                )
                self.add_module("raw_imputation_values_constraint", constraint)
        if isinstance(self, Warp):
            like = self.concentration0
            for i in (0, 1):
                name = f"concentration{i}"
                self.add_module(
                    f"{name}_constraint",
                    TorchConstraint(
                        GreaterThan(self._min_concentration_level, transform=None), like
                    ),
                )
                prior = getattr(self, f"{name}_prior")
                if prior is not None:
                    self.add_module(f"_registered_{name}_prior", TorchPrior(prior, like))

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        if args and isinstance(args[0], Tensor):
            # A standalone Tensor call exposes live parameter leaves just as
            # the earlier lazy public parameter creation did.
            self.__dict__.get("_native_default_parameters", {}).clear()
        result = nn.Module.__call__(self, *args, **kwargs)
        self._refresh_affine_buffers()
        return result

    def train(self, mode: bool = True) -> Any:
        return nn.Module.train(self, mode)

    def eval(self) -> Any:
        return self.train(False)

    parameters = nn.Module.parameters
    named_parameters = nn.Module.named_parameters
    buffers = nn.Module.buffers

    def named_buffers(self, *args: Any, **kwargs: Any) -> Any:
        for module in self.modules():
            if isinstance(module, TorchInputModule):
                module._refresh_affine_buffers()
        return nn.Module.named_buffers(self, *args, **kwargs)

    def _apply(self, fn: Any, *args: Any, **kwargs: Any) -> Any:
        if not callable(fn):
            return type(self)._native_port_descriptor._apply(self, fn, *args, **kwargs)
        dormant = self.__dict__.get("_native_affine_dormant", False)
        if dormant:
            self._refresh_affine_buffers(dtype=torch.double)
        defaults = self.__dict__.get("_native_default_parameters", {}).copy()
        result = nn.Module._apply(self, fn, *args, **kwargs)
        if dormant:
            self.__dict__["_native_affine_dormant"] = False
        direct = self.__dict__.get("_direct_parameters", {})
        for name, old in tuple(direct.items()):
            direct[name] = self._parameters[name]
            if defaults.get(name) is old:
                self._native_default_parameters[name] = self._parameters[name]
        return result

    def _save_to_state_dict(self, destination: Any, prefix: str, keep_vars: bool) -> None:
        self._refresh_affine_buffers()
        nn.Module._save_to_state_dict(self, destination, prefix, keep_vars)

    def _load_from_state_dict(self, *args: Any, **kwargs: Any) -> None:
        nn.Module._load_from_state_dict(self, *args, **kwargs)
        if "_native_affine_dormant" in self.__dict__:
            self.__dict__["_native_affine_dormant"] = False
        for name in self.__dict__.get("_direct_parameters", {}):
            self._direct_parameters[name] = self._parameters[name]

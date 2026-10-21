"""Actual Torch module protocols for public port outcome transforms."""

from __future__ import annotations

from typing import Any

from torch import nn

_TYPES: dict[type[Any], type[Any]] = {}


def _ensure_state(module: Any) -> None:
    from botorch_rs.models.transforms.outcome import Standardize

    if isinstance(module, Standardize):
        module._tensor_state()


class TorchOutcomeModule(nn.Module):
    _tensor_backend = "torch"

    @property
    def _requires_X(self) -> bool:
        from botorch_rs.models.transforms.outcome import (
            ChainedOutcomeTransform,
            StratifiedStandardize,
        )

        return isinstance(self, StratifiedStandardize) or (
            isinstance(self, ChainedOutcomeTransform)
            and any(getattr(child, "_requires_X", False) for child in self.values())
        )

    def fit(self, Y: Any, X: Any = None) -> Any:
        from botorch_rs.models.higher_order_gp import FlattenedStandardize
        from botorch_rs.models.transforms.outcome import ChainedOutcomeTransform, Standardize

        if isinstance(self, ChainedOutcomeTransform):
            for child in self.values():
                fit = getattr(child, "fit", None)
                if fit is not None:
                    fit(Y, X=X) if getattr(child, "_requires_X", False) else fit(Y)
                    Y, _ = child(Y, X=X)
                else:
                    mode = child.training
                    try:
                        child.train()
                        Y, _ = child(Y, X=X)
                    finally:
                        child.train(mode)
        elif isinstance(self, Standardize):
            values = (
                self._squeeze_to_single_output(Y) if isinstance(self, FlattenedStandardize) else Y
            )
            self._tensor_state(Y).fit(values, X=X)
        return self


def _initialize(self: Any, *args: Any, **kwargs: Any) -> None:
    nn.Module.__init__(self)
    self._native_port_descriptor.__init__(self, *args, **kwargs)
    from botorch_rs.models.transforms.outcome import ChainedOutcomeTransform

    if isinstance(self, ChainedOutcomeTransform):
        owner: Any = self
        for name, child in self.items():
            owner.add_module(name, child)
    _ensure_state(self)


def _apply(self: Any, fn: Any, recurse: bool = True) -> Any:
    _ensure_state(self)
    return nn.Module._apply(self, fn, recurse)


def _state_dict(self: Any, *args: Any, **kwargs: Any) -> Any:
    _ensure_state(self)
    return nn.Module.state_dict(self, *args, **kwargs)


def _load_state_dict(self: Any, *args: Any, **kwargs: Any) -> Any:
    _ensure_state(self)
    return nn.Module.load_state_dict(self, *args, **kwargs)


def _named_modules(self: Any, *args: Any, **kwargs: Any) -> Any:
    _ensure_state(self)
    yield from nn.Module.named_modules(self, *args, **kwargs)


def _trained(self: Any) -> Any:
    return self._tensor_state()._is_trained


def _reduce(self: Any, protocol: int) -> Any:
    return rebuild, (self._native_port_descriptor,), self.__getstate__()


def _setattr(self: Any, name: str, value: Any) -> None:
    from botorch_rs.models.transforms.outcome import ChainedOutcomeTransform

    if name == "_transforms" and isinstance(self, ChainedOutcomeTransform):
        owner: Any = self
        children = list(value.items())
        owner._modules.clear()
        for key, child in children:
            owner.add_module(key, child)
        object.__setattr__(self, name, owner._modules)
    else:
        nn.Module.__setattr__(self, name, value)


def rebuild(public_type: type[Any]) -> Any:
    return object.__new__(outcome_module_type(public_type))


def outcome_module_type(public_type: type[Any]) -> type[Any]:
    if public_type not in _TYPES:
        from botorch_rs.models.transforms.outcome import ChainedOutcomeTransform, Standardize

        methods: dict[str, Any] = {
            "__module__": __name__,
            "_native_port_descriptor": public_type,
            "_torch_outcome_factory": True,
            "__init__": _initialize,
            "__call__": nn.Module.__call__,
            "__getattr__": nn.Module.__getattr__,
            "__setattr__": _setattr,
            "__delattr__": nn.Module.__delattr__,
            "__reduce_ex__": _reduce,
            "_apply": _apply,
            "train": nn.Module.train,
            "eval": nn.Module.eval,
            "to": nn.Module.to,
            "cpu": nn.Module.cpu,
            "cuda": nn.Module.cuda,
            "float": nn.Module.float,
            "double": nn.Module.double,
            "parameters": nn.Module.parameters,
            "named_parameters": nn.Module.named_parameters,
            "buffers": nn.Module.buffers,
            "named_buffers": nn.Module.named_buffers,
            "named_modules": _named_modules,
            "state_dict": _state_dict,
            "load_state_dict": _load_state_dict,
        }
        if issubclass(public_type, Standardize):
            methods["_is_trained"] = property(_trained)
        bases: tuple[type[Any], ...] = (public_type, TorchOutcomeModule)
        if issubclass(public_type, ChainedOutcomeTransform):
            bases += (nn.ModuleDict,)
        _TYPES[public_type] = type(f"Torch{public_type.__name__}", bases, methods)
    return _TYPES[public_type]


def expose_statistics(module: Any, state: nn.Module) -> None:
    if isinstance(module, nn.Module):
        module._buffers.update(state._buffers)
        state._buffers = module._buffers
        module._non_persistent_buffers_set.update(state._non_persistent_buffers_set)
        state._non_persistent_buffers_set = module._non_persistent_buffers_set


def discard_statistics(module: Any) -> None:
    state = module.__dict__.pop("_tensor_standardize", None)
    if isinstance(module, nn.Module) and state is not None:
        module.__dict__["_standardize_tensor_spec"] = (state.means.dtype, state.means.device)
        for name in ("means", "stdvs", "_stdvs_sq", "_is_trained", "strata_mapping"):
            module._buffers.pop(name, None)
            module._non_persistent_buffers_set.discard(name)

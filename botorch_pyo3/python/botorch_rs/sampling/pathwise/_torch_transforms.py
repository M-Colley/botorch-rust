"""Module protocols for path transforms when Torch is already in use."""

from __future__ import annotations

import operator
from typing import Any

import torch
from torch import nn

from botorch_rs.sampling.pathwise import utils


class _TorchPathTransform(nn.Module):
    _port_type: Any
    _tensor_backend = "torch"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        nn.Module.__init__(self)
        self._port_type.__init__(self, *args, **kwargs)


class TorchChainedTransform(_TorchPathTransform, utils.ChainedTransform):
    transforms: Any
    forward = utils.ChainedTransform.forward

    def __init__(self, *transforms: Any) -> None:
        nn.Module.__init__(self)
        self.transforms = (
            nn.ModuleList(transforms)
            if all(isinstance(transform, nn.Module) for transform in transforms)
            else list(transforms)
        )


class TorchSineCosineTransform(_TorchPathTransform, utils.SineCosineTransform):
    _port_type = utils.SineCosineTransform
    forward = utils.SineCosineTransform.forward


class TorchInverseLengthscaleTransform(_TorchPathTransform, utils.InverseLengthscaleTransform):
    _port_type = utils.InverseLengthscaleTransform
    forward = utils.InverseLengthscaleTransform.forward


class TorchOutputscaleTransform(_TorchPathTransform, utils.OutputscaleTransform):
    _port_type = utils.OutputscaleTransform
    forward = utils.OutputscaleTransform.forward


class TorchFeatureSelector(_TorchPathTransform, utils.FeatureSelector):
    dim: Any
    indices: Any

    def __init__(self, indices: Any, dim: Any = -1) -> None:
        nn.Module.__init__(self)
        self.register_buffer(
            "dim", dim if isinstance(dim, torch.Tensor) else torch.tensor(operator.index(dim))
        )
        self.register_buffer(
            "indices",
            indices
            if isinstance(indices, torch.Tensor)
            else torch.tensor(tuple(operator.index(i) for i in indices), dtype=torch.long),
        )

    def forward(self, values: Any, **kwargs: Any) -> Any:
        if isinstance(values, torch.Tensor):
            dim = operator.index(self.dim)
            if values.ndim == 0 or not -values.ndim <= dim < values.ndim:
                raise IndexError(f"Dimension out of range: {dim} for {values.ndim} dimensions.")
            if bool(((self.indices < 0) | (self.indices >= values.shape[dim])).any()):
                raise IndexError(
                    f"Feature index out of range for axis of size {values.shape[dim]}."
                )
            return values.index_select(self.dim, self.indices.to(device=values.device))
        # Integer selector metadata has no differentiable inference state.
        dim = operator.index(self.dim)
        indices = tuple(operator.index(i) for i in self.indices)
        selector = object.__new__(utils.FeatureSelector)
        selector.__dict__.update(dim=dim, indices=indices)
        return utils.FeatureSelector.forward(selector, values)


class TorchOutcomeUntransformer(_TorchPathTransform, utils.OutcomeUntransformer):
    forward = utils.OutcomeUntransformer.forward

    def __init__(self, transform: Any, num_outputs: Any) -> None:
        nn.Module.__init__(self)
        self.transform = transform
        self.register_buffer(
            "num_outputs",
            num_outputs if isinstance(num_outputs, torch.Tensor) else torch.tensor(num_outputs),
        )


_TYPES: dict[type[Any], type[Any]] = {
    utils.ChainedTransform: TorchChainedTransform,
    utils.SineCosineTransform: TorchSineCosineTransform,
    utils.InverseLengthscaleTransform: TorchInverseLengthscaleTransform,
    utils.OutputscaleTransform: TorchOutputscaleTransform,
    utils.FeatureSelector: TorchFeatureSelector,
    utils.OutcomeUntransformer: TorchOutcomeUntransformer,
}


def tensor_transform_type(public_type: type[Any]) -> type[Any] | None:
    if getattr(public_type, "_tensor_backend", None) == "torch":
        return None
    from botorch_rs.sampling.pathwise._torch_types import module_type

    for base in public_type.__mro__:
        if base in _TYPES:
            return module_type(public_type, _TYPES[base], base)
    return module_type(public_type, TorchTensorTransformModule, utils.TensorTransform)


class TorchTensorTransformModule(nn.Module, utils.TensorTransform):
    _tensor_backend = "torch"

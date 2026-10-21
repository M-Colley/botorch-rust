"""Actual Tensor buffers and device-resident deterministic cost laws."""

from __future__ import annotations

from typing import Any, cast

import torch
from torch import nn

from botorch_rs import _botorch_rs as _rs
from botorch_rs._arrays import kind_of, to_flat
from botorch_rs.exceptions.errors import BotorchTensorDimensionError, UnsupportedError
from botorch_rs.models.cost import AffineFidelityCostModel, DeterministicCostModel, FixedCostModel


class TorchDeterministicCostPosterior:
    """A point mass: samples are exact copies, with no artificial jitter/noise."""

    _tensor_backend = "torch"

    def __init__(self, values: Any):
        self.mean = values
        self._kind = kind_of(values)

    @property
    def variance(self) -> Any:
        return torch.zeros_like(self.mean)

    @property
    def device(self) -> Any:
        return self.mean.device

    @property
    def dtype(self) -> Any:
        return self.mean.dtype

    @property
    def batch_shape(self) -> Any:
        return self.mean.shape[:-2]

    @property
    def base_sample_shape(self) -> Any:
        return self.mean.shape

    @property
    def batch_range(self) -> tuple[int, int]:
        return (0, -2)

    def _extended_shape(self, sample_shape: Any = ()) -> Any:
        return torch.Size(sample_shape) + self.mean.shape

    def rsample(self, sample_shape: Any = None) -> Any:
        shape = torch.Size([1] if sample_shape is None else sample_shape)
        return self.mean.expand(shape + self.mean.shape)

    def rsample_from_base_samples(self, sample_shape: Any, base_samples: Any) -> Any:
        return self.rsample(sample_shape)

    def scalarize(self, weights: Any, offset: Any = 0.0) -> Any:
        weights = torch.as_tensor(weights, dtype=self.dtype, device=self.device)
        if weights.shape != self.mean.shape[-1:]:
            raise ValueError("Cost scalarization weights must match its output count.")
        return TorchDeterministicCostPosterior((self.mean @ weights + offset).unsqueeze(-1))


def tensor_cost_forward(model: Any, X: Any) -> Any:
    if X.ndim < 2:
        raise BotorchTensorDimensionError("Expected X to have shape batch_shape x q x d.")
    if isinstance(model, AffineFidelityCostModel):
        weights = torch.as_tensor(model.weights, dtype=X.dtype, device=X.device)
        cost = torch.einsum("...f,f", X[..., model.fidelity_dims], weights).unsqueeze(-1)
        fixed: Any = model.fixed_cost
        if isinstance(fixed, torch.Tensor):
            fixed = fixed.to(X)
        return fixed + cost
    if isinstance(model, FixedCostModel):
        fixed = model.fixed_cost
        if not isinstance(fixed, torch.Tensor):
            fixed = torch.as_tensor(fixed, dtype=X.dtype, device=X.device)
        return fixed.view(*(1 for _ in X.shape[:-1]), model.num_outputs).expand(
            *X.shape[:-1], model.num_outputs
        )
    raise UnsupportedError("The deterministic Tensor cost path supports affine and fixed costs.")


def tensor_cost_posterior(model: Any, X: Any, output_indices: Any, kwargs: dict[str, Any]) -> Any:
    transform = kwargs.pop("posterior_transform", None)
    noise = kwargs.pop("observation_noise", False)
    if noise is not False or kwargs:
        raise UnsupportedError(
            "Deterministic costs support no observation noise or extra posterior arguments."
        )
    values = model.forward(X)
    if output_indices is not None:
        indices = torch.as_tensor(output_indices, dtype=torch.long, device=values.device)
        if indices.ndim != 1 or bool(
            ((indices < -model.num_outputs) | (indices >= model.num_outputs)).any()
        ):
            raise IndexError("Cost output indices must be a valid one-dimensional vector.")
        values = values.index_select(-1, indices.remainder(model.num_outputs))
    posterior = TorchDeterministicCostPosterior(values)
    return posterior if transform is None else transform(posterior, X=X)


class _TensorCostMixin:
    _tensor_backend = "torch"
    _parameters: Any
    _buffers: Any
    _num_outputs: int

    @property
    def batch_shape(self) -> Any:
        return torch.Size()

    @property
    def num_outputs(self) -> int:
        return self._num_outputs

    def forward(self, X: Any) -> Any:
        return DeterministicCostModel.forward(cast(Any, self), X)

    def posterior(self, X: Any, output_indices: Any = None, **kwargs: Any) -> Any:
        return DeterministicCostModel.posterior(cast(Any, self), X, output_indices, **kwargs)

    def _buffer_value(self, name: str) -> Any:
        if name in self._parameters:
            return self._parameters[name]
        if name in self._buffers:
            return self._buffers[name]
        raise AttributeError(name)


class TorchAffineFidelityCostModel(_TensorCostMixin, nn.Module, AffineFidelityCostModel):
    def __init__(self, fidelity_weights: Any = None, fixed_cost: Any = 0.01):
        nn.Module.__init__(self)
        fidelity_weights = {-1: 1.0} if fidelity_weights is None else fidelity_weights
        self.fidelity_dims = sorted(int(key) for key in fidelity_weights)
        values = [fidelity_weights[key] for key in self.fidelity_dims]
        tensor = next((value for value in values if isinstance(value, torch.Tensor)), None)
        weights = (
            torch.tensor(values)
            if tensor is None
            else torch.stack(
                [
                    torch.as_tensor(value, dtype=tensor.dtype, device=tensor.device)
                    for value in values
                ]
            )
        )
        self.register_buffer("weights", weights)
        self.fixed_cost = fixed_cost
        self._num_outputs = 1

    @property
    def weights(self) -> Any:
        return self._buffer_value("weights")

    @property
    def _rs_cost(self) -> Any:
        values, _ = to_flat(self.weights, "weights")
        fixed, shape = to_flat(self.fixed_cost, "fixed_cost")
        if shape:
            raise ValueError("Native affine fixed_cost must be scalar.")
        return _rs.CostModel.affine(list(zip(self.fidelity_dims, values, strict=True)), fixed[0])


class TorchFixedCostModel(_TensorCostMixin, nn.Module, FixedCostModel):
    def __init__(self, fixed_cost: Any):
        nn.Module.__init__(self)
        if fixed_cost.ndim != 1 or not fixed_cost.numel():
            raise BotorchTensorDimensionError(
                "fixed_cost must be a nonempty one-dimensional Tensor."
            )
        self.register_buffer("fixed_cost", fixed_cost)
        self._num_outputs = fixed_cost.shape[-1]
        self._kind = kind_of(fixed_cost)

    @property
    def fixed_cost(self) -> Any:
        return self._buffer_value("fixed_cost")

    @property
    def _rs_cost(self) -> Any:
        return _rs.CostModel.fixed(to_flat(self.fixed_cost, "fixed_cost")[0])

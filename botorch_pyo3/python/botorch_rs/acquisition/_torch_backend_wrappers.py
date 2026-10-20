"""Compositional feature wrappers that retain complete acquisition graphs."""

from __future__ import annotations

from collections.abc import Sequence
from numbers import Number
from typing import Any

import torch
from torch import Tensor, nn


class TensorFixedFeature(nn.Module):
    values: Tensor
    acq_func: Any
    model: Any

    def __init__(self, acq_function: Any, d: int, columns: list[int], values: Any) -> None:
        nn.Module.__init__(self)
        self._tensor_backend = "torch"
        self.acq_func = acq_function
        self.model = acq_function.model
        self.d = int(d)
        if len(set(columns)) != len(columns) or any(i < 0 or i >= d for i in columns):
            raise ValueError("Fixed columns must be unique indices within d.")
        self.columns = list(columns)
        self._log = getattr(acq_function, "_log", False)
        if not isinstance(values, Tensor) and isinstance(values, Sequence):
            if all(isinstance(value, Number) for value in values):
                values = torch.as_tensor(values, dtype=torch.float64)
            else:
                tensors = [value for value in values if isinstance(value, Tensor)]
                device = next((t.device for t in tensors if t.is_cuda), torch.device("cpu"))
                dtype = (
                    torch.float32
                    if len(tensors) == len(values)
                    and all(t.dtype == torch.float32 for t in tensors)
                    else torch.float64
                )
                parts = [
                    torch.as_tensor(value, dtype=dtype, device=device).reshape(1)
                    if isinstance(value, Number) or (isinstance(value, Tensor) and value.ndim == 0)
                    else torch.as_tensor(value, dtype=dtype, device=device)
                    for value in values
                ]
                values = torch.cat(torch.broadcast_tensors(*parts), dim=-1)
        values = torch.as_tensor(values)
        if values.ndim == 0 or values.shape[-1] != len(columns) or len(columns) >= d:
            raise ValueError(
                "Fixed values must end in len(columns), with fewer fixed columns than d."
            )
        self.register_buffer("values", values)

    def _construct_X_full(self, X: Tensor) -> Tensor:
        if not isinstance(X, Tensor) or X.ndim < 2 or X.shape[-1] != self.d - len(self.columns):
            raise ValueError("Input feature dimension must be d - len(columns).")
        values = self.values.to(X).expand(*X.shape[:-1], len(self.columns))
        joined = torch.cat((X, values), dim=-1)
        fixed = {
            index: self.d - len(self.columns) + position
            for position, index in enumerate(self.columns)
        }
        selector, free = [], 0
        for index in range(self.d):
            if index in fixed:
                selector.append(fixed[index])
            else:
                selector.append(free)
                free += 1
        return joined[..., selector]

    def forward(self, X: Tensor) -> Tensor:
        return self.acq_func(self._construct_X_full(X))

    @property
    def X_pending(self) -> Any:
        return self.acq_func.X_pending

    @X_pending.setter
    def X_pending(self, value: Any) -> None:
        self.set_X_pending(value)

    def set_X_pending(self, X_pending: Any = None) -> None:
        self.acq_func.set_X_pending(
            None if X_pending is None else self._construct_X_full(X_pending)
        )


class TensorProjected(nn.Module):
    def __init__(self, base_value_function: Any, project: Any) -> None:
        nn.Module.__init__(self)
        self._tensor_backend = "torch"
        self.base_value_function = base_value_function
        self.model = base_value_function.model
        self.project = project
        self._log = getattr(base_value_function, "_log", False)
        self.objective = getattr(base_value_function, "objective", None)
        self.posterior_transform = getattr(base_value_function, "posterior_transform", None)
        self.sampler = getattr(base_value_function, "sampler", None)

    def forward(self, X: Tensor) -> Tensor:
        return self.base_value_function(self.project(X))

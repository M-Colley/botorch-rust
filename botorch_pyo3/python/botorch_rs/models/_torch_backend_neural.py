"""Device arc-cosine recursion with differentiable coincident covariances."""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

from botorch_rs.exceptions.errors import UnsupportedError
from botorch_rs.models._torch_backend_modules import TorchAlgebraKernel
from botorch_rs.models.kernels.infinite_width_bnn import InfiniteWidthBNNKernel


class TorchInfiniteWidthBNNKernel(TorchAlgebraKernel, InfiniteWidthBNNKernel):  # type: ignore[misc]
    raw_weight_var_constraint: Any
    raw_bias_var_constraint: Any
    @property
    def batch_shape(self) -> Any:
        return self._tensor_batch_shape

    @batch_shape.setter
    def batch_shape(self, value: Any) -> None:
        self._tensor_batch_shape = torch.Size(value)

    @property
    def weight_var(self) -> Tensor:
        return self.raw_weight_var_constraint.transform(self.raw_weight_var)

    @weight_var.setter
    def weight_var(self, value: Any) -> None:
        self._assign("raw_weight_var", value, self.raw_weight_var_constraint)

    @property
    def bias_var(self) -> Tensor:
        return self.raw_bias_var_constraint.transform(self.raw_bias_var)

    @bias_var.setter
    def bias_var(self, value: Any) -> None:
        self._assign("raw_bias_var", value, self.raw_bias_var_constraint)

    def __init__(
        self, *args: Any, _source: Any = None, _like: Tensor | None = None, **kwargs: Any
    ) -> None:
        kwargs.pop("backend", None)
        source = InfiniteWidthBNNKernel(*args, **kwargs) if _source is None else _source
        source._base_node()  # Validate the same depth and epsilon domain as Rust.
        like = torch.empty((), device=kwargs.get("device")) if _like is None else _like
        TorchAlgebraKernel.__init__(self, source, like)
        self.depth, self.acos_eps = source.depth, source.acos_eps

    def _initialize_var(self, x: Tensor) -> Tensor:
        return self.weight_var * x.square().sum(-1, keepdim=True) / x.shape[-1] + self.bias_var

    def _update_var(self, value: Tensor, x: Tensor | None = None) -> Tensor:
        return self.weight_var * value / 2 + self.bias_var

    def _forward_selected(
        self, x1: Tensor, x2: Tensor, diag: bool, last_dim_is_batch: bool
    ) -> Tensor:
        if last_dim_is_batch:
            raise UnsupportedError("InfiniteWidthBNNKernel does not define feature batches.")
        if diag and x1.shape[-2] != x2.shape[-2]:
            raise ValueError("Paired diagonals need equally many inputs.")
        dot = (x1 * x2).sum(-1, keepdim=True) if diag else x1 @ x2.transpose(-2, -1)
        cross = self.weight_var * dot / x1.shape[-1] + self.bias_var
        coincident = (
            (x1 == x2).all(-1, keepdim=True)
            if diag
            else (x1.unsqueeze(-2) == x2.unsqueeze(-3)).all(-1)
        )
        left, right = self._initialize_var(x1), self._initialize_var(x2)
        for layer in range(self.depth):
            if layer:
                left, right = self._update_var(left), self._update_var(right)
            scale = left.sqrt() * (right.sqrt() if diag else right.sqrt().transpose(-2, -1))
            fraction = (cross / scale).clamp(-1 + self.acos_eps, 1 - self.acos_eps)
            fraction = torch.where(coincident, torch.zeros_like(fraction), fraction)
            angle = fraction.acos()
            regular = (
                self.weight_var
                / (2 * torch.pi)
                * scale
                * (angle.sin() + (torch.pi - angle) * fraction)
                + self.bias_var
            )
            cross = torch.where(coincident, self._update_var(cross), regular)
        return cross.squeeze(-1) if diag else cross

"""Integrated white-noise covariance with device-resident derivative graphs."""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import Tensor

from botorch_rs.models._torch_backend_modules import TorchKernel
from botorch_rs.models.kernels.integrated_white_noise import IntegratedWhiteNoiseKernel


class TorchIntegratedWhiteNoiseKernel(TorchKernel, IntegratedWhiteNoiseKernel):  # type: ignore[misc]
    """The scalar integrated Wiener law, with correct smooth diagonal limits."""

    _tensor_backend = "torch"
    has_lengthscale = False
    is_stationary = False

    @property
    def batch_shape(self) -> Any:
        return self._tensor_batch_shape

    @batch_shape.setter
    def batch_shape(self, shape: Any) -> None:
        self._tensor_batch_shape = torch.Size(shape)

    def __init__(
        self, *args: Any, _source: Any = None, _like: Tensor | None = None, **kwargs: Any
    ) -> None:
        kwargs.pop("backend", None)
        dtype, device = kwargs.pop("dtype", None), kwargs.pop("device", None)
        source = (
            IntegratedWhiteNoiseKernel(*args, backend="native", **kwargs)
            if _source is None
            else _source
        )
        like = torch.empty((), dtype=dtype, device=device) if _like is None else _like
        TorchKernel.__init__(self, like, source.active_dims)
        self.order, self.batch_shape = source.order, source.batch_shape
        denominator = math.factorial(self.order - 1) ** 2
        self._terms = [
            (
                math.comb(self.order - 1, j) / denominator / (self.order + j),
                self.order - 1 - j,
                self.order + j,
            )
            for j in range(self.order)
        ]

    def forward(
        self,
        x1: Tensor,
        x2: Tensor | None = None,
        diag: bool = False,
        last_dim_is_batch: bool = False,
        **params: Any,
    ) -> Tensor:
        x1, x2 = self._inputs(x1, x2)
        return self._forward_selected(x1, x2, diag, last_dim_is_batch)

    def _forward_selected(
        self, x1: Tensor, x2: Tensor, diag: bool, last_dim_is_batch: bool
    ) -> Tensor:
        if x1.shape[-1] != 1 or x2.shape[-1] != 1:
            raise ValueError("IntegratedWhiteNoiseKernel requires scalar time inputs.")
        if not bool(
            torch.isfinite(x1).all() & torch.isfinite(x2).all() & (x1 >= 0).all() & (x2 >= 0).all()
        ):
            raise ValueError("IntegratedWhiteNoiseKernel requires finite nonnegative time inputs.")
        left, right = x1[..., 0], x2[..., 0]
        if diag:
            if left.shape[-1] != right.shape[-1]:
                raise ValueError("diag=True requires equal numbers of input points.")
        else:
            left, right = left.unsqueeze(-1), right.unsqueeze(-2)
        if self.order == 1:
            # Brownian covariance is genuinely nonsmooth at equal independent
            # times. Keep Torch's symmetric first-derivative convention.
            result = torch.minimum(left, right)
        else:
            lower_branch = left <= right
            lower = torch.where(lower_branch, left, right)
            upper = torch.where(lower_branch, right, left)
            difference = upper - lower
            # A consistent region selects the correct coincident derivatives
            # through smoothness order 2p-2. Independently splitting the min
            # and max derivatives at ties gives an incorrect Hessian.
            result = torch.zeros_like(lower)
            for coefficient, a, b in self._terms:
                result = result + coefficient * difference.pow(a) * lower.pow(b)
        trailing = 1 if diag else 2
        batch = torch.broadcast_shapes(self.batch_shape, result.shape[:-trailing])
        result = result.expand(*batch, *result.shape[-trailing:])
        return result.unsqueeze(-trailing - 1) if last_dim_is_batch else result

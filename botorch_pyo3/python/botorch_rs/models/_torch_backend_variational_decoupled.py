"""Decoupled mean/covariance inducing variables with port-owned Tensor algebra."""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import Tensor

from botorch_rs import _botorch_rs as _rs
from botorch_rs.models._torch_backend_variational import (
    _dense,
    _distribution_moments,
    _TorchVariationalOutput,
    _TorchVariationalStrategy,
)


class TorchBatchDecoupledStrategy(_TorchVariationalStrategy):
    decoupled = True

    def __init__(
        self,
        owner: Any,
        points: Tensor,
        distribution: Any,
        learn: bool,
        mean_var_batch_dim: int | None = None,
    ) -> None:
        if mean_var_batch_dim is not None and mean_var_batch_dim >= 0:
            raise ValueError(
                f"mean_var_batch_dim should be negative indexed, got {mean_var_batch_dim}"
            )
        self.mean_var_batch_dim = mean_var_batch_dim
        dimension = -3 if mean_var_batch_dim is None else mean_var_batch_dim - 2
        super().__init__(owner, torch.stack((points, points), dim=dimension), distribution, learn)

    def kl_divergence(self, moments: tuple[Tensor, Tensor, bool] | None = None) -> Tensor:
        if moments is None:
            moments = _distribution_moments(self._variational_distribution)
        if moments[2]:
            raise NotImplementedError(
                "BatchDecoupledVariationalStrategy does not work with DeltaVariationalDistribution"
            )
        # Source: KL(Delta(q.mean), N(0,I)) + KL(N(0,q.covariance), N(0,I)).
        # The Delta mean term retains the prior-density normalizing constant.
        return super().kl_divergence(moments) + 0.5 * moments[0].shape[-1] * math.log(2 * math.pi)

    def variational_output(self, X: Tensor) -> Any:
        owner = self._owner_ref()
        X = X.clone()
        points = self.inducing_points
        axis = self.mean_var_batch_dim or -1
        query = X.unsqueeze(axis - 2)
        modules = owner.model
        m, n = points.shape[-2], X.shape[-2]
        batch = torch.broadcast_shapes(
            points.shape[:-2], query.shape[:-2], modules.covar_module.batch_shape
        )
        points = points.expand(*batch, *points.shape[-2:])
        query = query.expand(*batch, *query.shape[-2:])
        eye = torch.eye(m, dtype=X.dtype, device=X.device)
        kzz = _dense(modules.covar_module(points, points)) + self.jitter_val * eye
        # The source factors in double precision even for float32 kernels.
        root = _rs.tensor_psd_safe_cholesky(kzz.double()).to(X.dtype)
        interpolation = torch.linalg.solve_triangular(
            root, _dense(modules.covar_module(points, query)), upper=False
        )
        mean_interpolation = interpolation.select(axis - 2, 0)
        covariance_interpolation = interpolation.select(axis - 2, 1)
        moments = _distribution_moments(self._variational_distribution)
        if moments[2]:
            raise NotImplementedError(
                "BatchDecoupledVariationalStrategy does not work with DeltaVariationalDistribution"
            )
        mean_q, covariance_q, _ = moments
        mean = modules.mean_module(query).select(axis - 1, 0) + (
            mean_interpolation.mT @ mean_q.unsqueeze(-1)
        ).squeeze(-1)
        prior_covariance = _dense(modules.covar_module(query, query))
        prior_covariance = (
            prior_covariance + self.jitter_val * torch.eye(n, dtype=X.dtype, device=X.device)
        ).select(axis - 2, 1)
        covariance = prior_covariance + covariance_interpolation.mT @ (
            (covariance_q - eye) @ covariance_interpolation
        )
        return _TorchVariationalOutput(
            owner,
            mean,
            covariance.diagonal(dim1=-2, dim2=-1),
            lambda: covariance,
            variational_moments=moments,
        )

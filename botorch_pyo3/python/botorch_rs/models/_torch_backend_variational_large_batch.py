"""Grouped float64 variational prediction with ordinary Torch autograd."""

from __future__ import annotations

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


class TorchLargeBatchVariationalStrategy(_TorchVariationalStrategy):
    grouped_middle = True

    def variational_output(self, X: Tensor, diag: bool = True) -> Any:
        owner = self._owner_ref()
        X = X.clone()
        query = owner._augment(X)
        points = self.inducing_points
        inducing = points if points.ndim == owner._train_X.ndim + 1 else owner._augment(points)
        kernel = owner.model.covar_module
        m, n = points.shape[-2], X.shape[-2]
        jitter = self.jitter_val
        eye = torch.eye(m, dtype=X.dtype, device=X.device)
        covariance_zz = _dense(kernel(inducing, inducing)) + jitter * eye
        root = _rs.tensor_psd_safe_cholesky(covariance_zz.double())
        cross = _dense(kernel(inducing, query))
        cross_double = cross.double()
        moments = _distribution_moments(self._variational_distribution)
        mean_q, covariance_q, _ = moments

        # Group the inducing-sized systems before multiplying the potentially
        # very wide cross covariance. No triangular solve has n RHS columns.
        mean_weight = torch.linalg.solve_triangular(
            root.mT, mean_q.double().unsqueeze(-1), upper=True
        )
        middle = torch.linalg.solve_triangular(
            root, (covariance_q - eye).double(), upper=False, left=False
        )
        middle = torch.linalg.solve_triangular(root.mT, middle, upper=True)
        mean = owner.model.mean_module(query) + (cross_double.mT @ mean_weight).squeeze(-1).to(
            cross.dtype
        )
        update_double = middle @ cross_double
        diagonal_only = diag and owner.model.training and owner.num_outputs == 1
        variance_update = (
            (cross_double * update_double).sum(-2).to(cross.dtype)
            if diagonal_only
            else (cross * update_double.to(cross.dtype)).sum(-2)
        )
        variance = _dense(kernel(query, query, diag=True)) + jitter + variance_update
        parameters = {name: value.clone() for name, value in kernel.named_parameters()}
        buffers = {name: value.clone() for name, value in kernel.named_buffers()}

        def covariance() -> Tensor:
            prior = _dense(
                torch.func.functional_call(kernel, (parameters, buffers), (query, query))
            )
            correction = (
                torch.diag_embed(variance_update)
                if diagonal_only
                else cross.mT @ update_double.to(cross.dtype)
            )
            return prior + correction + jitter * torch.eye(n, dtype=X.dtype, device=X.device)

        return _TorchVariationalOutput(
            owner, mean, variance, covariance, variational_moments=moments
        )

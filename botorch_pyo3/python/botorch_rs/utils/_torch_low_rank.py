"""Cached independent Gaussian sampling using device-resident Torch algebra."""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import Tensor

from botorch_rs import _botorch_rs as _rs
from botorch_rs.exceptions.errors import BotorchError, NanError


def covariance_blocks(distribution: Any, q: int, m: int) -> Tensor:
    """Extract independent output blocks, retaining their complete graph."""
    covariance = distribution.covariance_matrix
    if not isinstance(covariance, Tensor):
        covariance = covariance.to_dense()
    batch = covariance.shape[:-2]
    if m == 1:
        return covariance.unsqueeze(-3)
    if getattr(distribution, "_interleaved", False):
        covariance = covariance.reshape(*batch, q, m, q, m).transpose(-4, -3).transpose(-2, -1)
    else:
        covariance = covariance.reshape(*batch, m, q, m, q)
    same_output = torch.eye(m, dtype=torch.bool, device=covariance.device).reshape(m, 1, m, 1)
    if bool(covariance.masked_fill(same_output, 0).ne(0).any()):
        raise BotorchError(
            "Expected BlockDiagLinearOperator, but got a dense covariance "
            "matrix (the outputs are correlated)."
        )
    return covariance.diagonal(dim1=-4, dim2=-2).movedim(-1, -3)


def sample_cached_cholesky(
    posterior: Any,
    baseline_L: Any,
    q: int,
    base_samples: Any,
    sample_shape: Any,
    max_tries: int,
) -> Tensor:
    mean = posterior.mean
    batch, total, m = mean.shape[:-2], mean.shape[-2], mean.shape[-1]
    if not isinstance(q, int) or q < 1 or q > total:
        raise ValueError("q must be a positive integer no greater than the posterior point count.")
    sample_shape = tuple(int(value) for value in sample_shape)
    count = math.prod(sample_shape)
    n = total - q
    covariance = covariance_blocks(posterior.distribution, total, m)
    if not isinstance(baseline_L, Tensor):
        baseline_L = torch.as_tensor(baseline_L, dtype=mean.dtype, device=mean.device)
    target = (*batch, m, n, n) if posterior._is_mt else (*batch, n, n)
    try:
        factor = baseline_L.expand(target)
    except RuntimeError as error:
        raise BotorchError(
            f"baseline_L of shape {tuple(baseline_L.shape)} does not match the posterior "
            f"(batch {tuple(batch)}, {m} outputs, {n} baseline points)."
        ) from error
    if not posterior._is_mt:
        factor = factor.unsqueeze(-3)
    bottom = covariance[..., -q:, :]
    cross, new = bottom.split((n, q), dim=-1)
    cross_root = torch.linalg.solve_triangular(factor, cross.transpose(-1, -2), upper=False)
    cross_root = cross_root.transpose(-1, -2)
    conditional_root = _rs.tensor_psd_safe_cholesky(
        new - cross_root @ cross_root.transpose(-1, -2), max_tries=max_tries
    )
    root = torch.cat((cross_root, conditional_root), dim=-1)
    if not isinstance(base_samples, Tensor):
        base_samples = torch.as_tensor(base_samples, dtype=mean.dtype, device=mean.device)
    if base_samples.numel() != count * total * m:
        raise RuntimeError(
            f"shape {tuple(base_samples.shape)} of the base samples is invalid for the sample shape "
            f"{sample_shape} and {total} x {m} points and outputs."
        )
    # Normal draws are shared across every model/t-batch and retain the
    # point-major public q x m layout before changing to output-major blocks.
    normals = base_samples.reshape(count, total, m).permute(2, 1, 0)
    centered = root @ normals
    result = centered + mean[..., -q:, :].transpose(-1, -2).unsqueeze(-1)
    result = result.permute(-1, *range(len(batch)), -2, -3).reshape(*sample_shape, *batch, q, m)
    contains_nans = bool(result.isnan().any())
    contains_infs = bool(result.isinf().any())
    if contains_nans or contains_infs:
        suffix = " and ".join(
            label
            for label, present in (("nans", contains_nans), ("infs", contains_infs))
            if present
        )
        raise NanError(f"Samples contain {suffix}.")
    return result.contiguous()

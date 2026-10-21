"""Device-resident, differentiable state transitions for MVNXPB."""

from __future__ import annotations

import warnings
from typing import Any

import torch
from torch import Tensor

from botorch_rs import _botorch_rs as _rs
from botorch_rs._arrays import kind_of
from botorch_rs.exceptions.errors import NotPSDError
from botorch_rs.utils.probability._torch_bvn import _cdf, _density, bvn, bvnmom
from botorch_rs.utils.probability.linalg import (
    PivotedCholesky,
    augment_cholesky,
    block_matrix_concat,
)
from botorch_rs.utils.probability.utils import STANDARDIZED_RANGE
from botorch_rs.utils.safe_math import log as safe_log


def _replace(vector: Tensor, index: int, value: Tensor) -> Tensor:
    return torch.cat((vector[..., :index], value.unsqueeze(-1), vector[..., index + 1 :]), -1)


def _nonnegative_root(value: Tensor) -> Tensor:
    positive = value > 0
    return value.masked_fill(~positive, 1).sqrt().masked_fill(~positive, 0)


def _order(pivot: Tensor, step: int, size: int) -> Tensor:
    if bool(((pivot < step) | (pivot >= size)).any()):
        raise ValueError("Pivot indices must lie between the current step and matrix size.")
    indices = torch.arange(size, dtype=torch.long, device=pivot.device).expand(*pivot.shape, size)
    return torch.where(
        indices == step,
        pivot.unsqueeze(-1),
        torch.where(indices == pivot.unsqueeze(-1), step, indices),
    )


def cholesky_pivot(state: Any, pivot: Tensor) -> None:
    matrix = state.tril
    n = matrix.shape[-1]
    pivot = torch.as_tensor(pivot, dtype=torch.long, device=matrix.device)
    order = _order(pivot, state.step, n)
    symmetric = matrix + matrix.transpose(-1, -2) - matrix.diagonal(dim1=-2, dim2=-1).diag_embed()
    rows = order.unsqueeze(-1).expand(*order.shape, n)
    columns = order.unsqueeze(-2).expand(*order.shape[:-1], n, n)
    shuffled = symmetric.gather(-2, rows).gather(-1, columns).tril()
    head = matrix.gather(-2, rows)[..., : state.step]
    state.tril = torch.cat((head, shuffled[..., state.step :]), -1)
    state.perm = state.perm.gather(-1, order)
    if state.diag is not None:
        state.diag = state.diag.gather(-1, order)


def cholesky_update(state: Any, eps: float) -> None:
    matrix, i = state.tril, state.step
    diagonal = matrix[..., i, i].clamp_min(0)
    valid = diagonal > (i * eps) ** 2
    std = diagonal.masked_fill(~valid, 1).sqrt()
    lower_column = (matrix[..., i + 1 :, i] / std.unsqueeze(-1)).masked_fill(
        ~valid.unsqueeze(-1), 0
    )
    diagonal = std.masked_fill(~valid, 0)
    row = torch.cat(
        (matrix[..., i, :i], diagonal.unsqueeze(-1), torch.zeros_like(matrix[..., i, i + 1 :])), -1
    )
    rank_one = (lower_column.unsqueeze(-1) * lower_column.unsqueeze(-2)).tril()
    tail = matrix[..., i + 1 :, i + 1 :] - rank_one
    bottom = torch.cat((matrix[..., i + 1 :, :i], lower_column.unsqueeze(-1), tail), -1)
    state.tril = torch.cat((matrix[..., :i, :], row.unsqueeze(-2), bottom), -2)
    state.step += 1


def initialize(state: Any, covariance: Tensor, bounds: Tensor) -> None:
    if covariance.ndim < 2 or covariance.shape[-2] != covariance.shape[-1]:
        raise ValueError("covariance_matrix must contain square matrices.")
    n = covariance.shape[-1]
    if bounds.ndim < 2 or bounds.shape[-2:] != (n, 2):
        raise ValueError("bounds must have shape batch_shape x n x 2.")
    batch = torch.broadcast_shapes(covariance.shape[:-2], bounds.shape[:-2])
    covariance = covariance.expand(*batch, n, n)
    bounds = bounds.to(covariance).expand(*batch, n, 2)
    variance = covariance.diagonal(dim1=-2, dim2=-1).unsqueeze(-1)
    if bool((variance <= 0).any()):
        raise ValueError("Covariance diagonal entries must be positive.")
    std, inverse_std = variance.sqrt(), variance.rsqrt()
    matrix = inverse_std * covariance * inverse_std.transpose(-1, -2)
    lower, upper = STANDARDIZED_RANGE
    bounds = inverse_std * bounds.clamp(min=lower * std, max=upper * std)
    perm = torch.arange(n, device=covariance.device, dtype=torch.long).expand(*batch, n).clone()
    state.step = 0
    state.perm, state.bounds = perm, bounds
    state.piv_chol = PivotedCholesky(0, matrix.tril(), perm.clone(), std.squeeze(-1).clone())
    state.plug_ins = covariance.new_full((*batch, n), torch.nan)
    state.log_prob = covariance.new_zeros(batch)
    state.log_prob_extra = None
    state._kind = kind_of(covariance)
    state._source = None


def select_pivot(state: Any) -> Tensor:
    i, matrix = state.piv_chol.step, state.piv_chol.tril
    bounds = state.bounds[..., i:, :]
    if i:
        bounds = bounds - matrix[..., i:, :i] @ state.plug_ins[..., :i, None]
    variance = matrix.diagonal(dim1=-2, dim2=-1)[..., i:].clamp_min(0)
    std = _nonnegative_root(variance)
    # At a deterministic conditional variable, evaluate the limiting event
    # explicitly rather than multiplying infinite inverse scales by zero.
    safe_std = std.masked_fill(std == 0, 1).unsqueeze(-1)
    probabilities = _cdf(bounds / safe_std).diff(dim=-1).squeeze(-1)
    deterministic = (bounds[..., 0] <= 0) & (bounds[..., 1] >= 0)
    probabilities = torch.where(std == 0, deterministic.to(probabilities), probabilities)
    return i + probabilities.argmin(dim=-1)


def pivot(state: Any, index: Tensor) -> None:
    if state.step > state.piv_chol.step:
        raise ValueError("Invalid state: solver ran ahead of matrix decomposition.")
    index = torch.as_tensor(index, device=state.bounds.device, dtype=torch.long)
    if index.shape != state.bounds.shape[:-2]:
        raise ValueError("Argument `pivot` does to match with batch shape`.")
    order = _order(index, state.step, state.bounds.shape[-2])
    if state.step == state.piv_chol.step:
        state.piv_chol.pivot_(index)
    state.perm = state.perm.gather(-1, order)
    state.bounds = state.bounds.gather(-2, order.unsqueeze(-1).expand(*order.shape, 2))


def solve(state: Any, num_steps: int | None, eps: float) -> Tensor:
    if state.step > state.piv_chol.step:
        raise ValueError("Invalid state: solver ran ahead of matrix decomposition.")
    start, n = state.step, state.plug_ins.shape[-1]
    steps = num_steps or n - start
    if steps < 0 or start + steps > n:
        raise ValueError("num_steps must not exceed the remaining number of variables.")
    if not steps:
        return state.log_prob
    if n > start and start % 2:
        state.log_prob = state.log_prob - state.log_prob_extra
        state.log_prob_extra = None
    for i in range(start, start + steps):
        update = state.step == state.piv_chol.step
        if update:
            index = state.select_pivot()
        else:
            matching = state.perm[..., i:] == state.piv_chol.perm[..., i : i + 1]
            if not bool(matching.any(dim=-1).all()):
                raise ValueError("The precomputed Cholesky permutation is inconsistent.")
            index = i + matching.to(torch.long).argmax(dim=-1)
        if bool((index > i).any()):
            state.pivot_(index)
        matrix = state.piv_chol.tril
        diagonal = _nonnegative_root(matrix[..., i, i]) if update else matrix[..., i, i]
        valid = diagonal > i * eps
        inverse = diagonal.masked_fill(~valid, 1).reciprocal().masked_fill(~valid, 0)
        bounds = state.bounds[..., i, :]
        if i:
            bounds = bounds - (matrix[..., i, :i] * state.plug_ins[..., :i]).sum(-1, keepdim=True)
        lb, ub = (inverse.unsqueeze(-1) * bounds).unbind(-1)
        probability = _cdf(ub) - _cdf(lb)
        deterministic = (bounds[..., 0] < 0) & (bounds[..., 1] > 0)
        probability = torch.where(valid, probability, deterministic.to(probability))
        small = probability <= i * eps
        plugin = (_density(lb) - _density(ub)) / probability.masked_fill(small, 1)
        fallback = torch.where(lb < -9, ub, torch.where(lb > 9, lb, 0.5 * (lb + ub)))
        state.plug_ins = _replace(state.plug_ins, i, torch.where(small, fallback, plugin))
        if i % 2:
            h = i - 1
            previous_std, cross = matrix[..., h, h], matrix[..., i, h]
            current_std = _nonnegative_root(diagonal.square() + cross.square())
            marginal_std = torch.stack((previous_std, current_std), -1)
            block = state.bounds[..., h : i + 1, :]
            if h:
                block = block - matrix[..., h : i + 1, :h] @ state.plug_ins[..., :h, None]
            zero_std = marginal_std == 0
            deterministic = (block[..., 0] < 0) & (block[..., 1] > 0)
            block = block / marginal_std.masked_fill(zero_std, 1).unsqueeze(-1)
            lower, upper = block.unbind(-1)
            lower = torch.where(zero_std, torch.where(deterministic, -torch.inf, torch.inf), lower)
            upper = upper.masked_fill(zero_std, torch.inf)
            correlation = (
                previous_std * cross / marginal_std.prod(-1).masked_fill(zero_std.any(-1), 1)
            )
            xl, yl = lower.unbind(-1)
            xu, yu = upper.unbind(-1)
            block_probability = bvn(correlation, xl, yl, xu, yu)
            good = block_probability > 0
            zh, zi = bvnmom(
                correlation.masked_fill(~good, 0),
                xl.masked_fill(~good, -1),
                yl.masked_fill(~good, -1),
                xu.masked_fill(~good, 1),
                yu.masked_fill(~good, 1),
                block_probability.masked_fill(~good, 1),
            )
            zh, zi = zh.masked_fill(~good, 0), zi.masked_fill(~good, 0)
            state.plug_ins = _replace(state.plug_ins, h, zh)
            state.plug_ins = _replace(state.plug_ins, i, inverse * (current_std * zi - cross * zh))
            state.log_prob = state.log_prob + safe_log(block_probability)
        state.step += 1
        if update:
            state.piv_chol.update_(eps)
    if state.step % 2:
        state.log_prob_extra = safe_log(probability)
        state.log_prob = state.log_prob + state.log_prob_extra
    return state.log_prob


def augment(
    state: Any,
    covariance: Tensor,
    bounds: Tensor,
    cross_covariance: Tensor,
    disable_pivoting: bool,
    jitter: float | None,
    max_tries: int | None,
) -> Any:
    n, m = state.perm.shape[-1], covariance.shape[-1]
    if n != state.piv_chol.step:
        raise NotImplementedError("Augmentation of incomplete solutions not implemented yet.")
    batch = torch.broadcast_shapes(
        state.bounds.shape[:-2],
        covariance.shape[:-2],
        bounds.shape[:-2],
        cross_covariance.shape[:-2],
    )
    state = state.expand(*batch)
    covariance = covariance.expand(*batch, m, m)
    bounds = bounds.expand(*batch, m, 2)
    cross_covariance = cross_covariance.expand(*batch, m, n)
    variance = covariance.diagonal(dim1=-2, dim2=-1).unsqueeze(-1)
    std, inverse_std = variance.sqrt(), variance.rsqrt()
    cross = inverse_std * cross_covariance
    old_diag = state.piv_chol.diag
    if old_diag is None:
        diagonal = torch.cat((std.new_ones((*batch, n)), std.squeeze(-1)), -1)
    else:
        cross = cross / old_diag.unsqueeze(-2)
        diagonal = torch.cat((old_diag, std.squeeze(-1)), -1)
    new_covariance = inverse_std * covariance * inverse_std.transpose(-1, -2)
    old = state.piv_chol.tril
    try:
        factor = augment_cholesky(old, new_covariance, Kba=cross, jitter=jitter)
    except NotPSDError:
        warnings.warn(
            "Joint covariance matrix not positive definite, attempting recovery.", stacklevel=3
        )
        joint = block_matrix_concat(
            ((old @ old.transpose(-1, -2), cross.transpose(-1, -2)), (cross, new_covariance))
        )
        factor = _rs.tensor_psd_safe_cholesky(joint, jitter, 3 if max_tries is None else max_tries)
    if not disable_pivoting:
        block = factor[..., n:, n:]
        lower = torch.cat((factor[..., n:, :n], (block @ block.transpose(-1, -2)).tril()), -1)
        factor = torch.cat((factor[..., :n, :], lower), -2)
    lo, hi = STANDARDIZED_RANGE
    bounds = inverse_std * bounds.clamp(min=lo * std, max=hi * std)
    new_perm = torch.arange(n, n + m, dtype=state.perm.dtype, device=state.perm.device).expand(
        *batch, m
    )
    cholesky = PivotedCholesky(
        n + m if disable_pivoting else n,
        factor,
        torch.cat((state.piv_chol.perm, new_perm), -1),
        diagonal,
    )
    return state.build(
        step=state.step,
        perm=torch.cat((state.perm, new_perm), -1),
        bounds=torch.cat((state.bounds, bounds), -2),
        piv_chol=cholesky,
        plug_ins=torch.cat((state.plug_ins, state.plug_ins.new_full((*batch, m), torch.nan)), -1),
        log_prob=state.log_prob,
        log_prob_extra=state.log_prob_extra,
    )

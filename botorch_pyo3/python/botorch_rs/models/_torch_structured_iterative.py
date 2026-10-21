"""Differentiable iterative Gaussian solves and stochastic log determinants."""

from __future__ import annotations

import math
import warnings

import torch
from torch import Tensor

from botorch_rs import _botorch_rs as _rs
from botorch_rs.exceptions.warnings import NumericsWarning
from botorch_rs.models._structured_iterative import IterativeSettings, current


def enabled(matrix: Tensor, flag: str, config: IterativeSettings | None) -> bool:
    return (
        config is not None
        and bool(getattr(config, flag))
        and matrix.shape[-1] > config.max_cholesky_size
    )


def conjugate_gradient(matrix: Tensor, rhs: Tensor, config: IterativeSettings) -> Tensor:
    """Jacobi-preconditioned multiple-RHS CG using ordinary Tensor operations."""
    batch = torch.broadcast_shapes(matrix.shape[:-2], rhs.shape[:-2])
    rhs = rhs.expand(*batch, *rhs.shape[-2:])
    matrix = matrix.expand(*batch, *matrix.shape[-2:])
    diagonal = matrix.diagonal(dim1=-2, dim2=-1).unsqueeze(-1)
    result = torch.zeros_like(rhs)
    residual = rhs
    norm = rhs.norm(dim=-2, keepdim=True)
    norm = torch.where(norm > 0, norm, torch.ones_like(norm))
    preconditioned = residual / diagonal
    direction = preconditioned
    product = (residual * preconditioned).sum(-2, keepdim=True)
    relative = residual.norm(dim=-2, keepdim=True) / norm
    for _ in range(min(config.max_iter, matrix.shape[-1])):
        active = relative > config.tol
        if not bool(active.any()):
            break
        applied = matrix @ direction
        denominator = (direction * applied).sum(-2, keepdim=True)
        safe = torch.where(active, denominator, torch.ones_like(denominator))
        step = torch.where(active, product / safe, torch.zeros_like(product))
        result = result + direction * step
        residual = residual - applied * step
        preconditioned = residual / diagonal
        updated = (residual * preconditioned).sum(-2, keepdim=True)
        safe = torch.where(active, product, torch.ones_like(product))
        ratio = torch.where(active, updated / safe, torch.zeros_like(updated))
        direction = preconditioned + direction * ratio
        product = updated
        relative = residual.norm(dim=-2, keepdim=True) / norm
    if bool((relative > config.tol).any()):
        warnings.warn(
            "Conjugate gradients reached the iteration limit before the requested tolerance.",
            NumericsWarning,
            stacklevel=2,
        )
    return result


def solve(matrix: Tensor, rhs: Tensor, config: IterativeSettings | None = None) -> Tensor:
    config = current() if config is None else config
    if enabled(matrix, "solves", config):
        assert config is not None
        return conjugate_gradient(matrix, rhs, config)
    return torch.cholesky_solve(rhs, _rs.tensor_psd_safe_cholesky(matrix))


def _lanczos_log_quadrature(matrix: Tensor, probes: Tensor, iterations: int) -> Tensor:
    norm = probes.norm(dim=-2, keepdim=True)
    vector = probes / norm
    vectors: list[Tensor] = []
    diagonals: list[Tensor] = []
    off_diagonals: list[Tensor] = []
    for index in range(min(iterations, matrix.shape[-1])):
        vectors.append(vector)
        residual = matrix @ vector
        alpha = (vector * residual).sum(-2)
        diagonals.append(alpha)
        residual = residual - vector * alpha.unsqueeze(-2)
        # Full reorthogonalization retains accuracy for covariance spectra.
        for previous in vectors:
            residual = residual - previous * (previous * residual).sum(-2, keepdim=True)
        beta = residual.norm(dim=-2)
        if index + 1 == min(iterations, matrix.shape[-1]) or bool(
            (beta < torch.finfo(matrix.dtype).eps * 10).all()
        ):
            break
        off_diagonals.append(beta)
        safe = beta.clamp_min(torch.finfo(matrix.dtype).tiny).unsqueeze(-2)
        vector = residual / safe
    diagonal = torch.stack(diagonals, dim=-1)
    tridiagonal = torch.diag_embed(diagonal)
    if off_diagonals:
        off = torch.stack(off_diagonals, dim=-1)
        tridiagonal = (
            tridiagonal + torch.diag_embed(off, offset=1) + torch.diag_embed(off, offset=-1)
        )
    values, eigenvectors = torch.linalg.eigh(tridiagonal)
    weights = eigenvectors[..., 0, :].square()
    quadrature = (weights * values.clamp_min(torch.finfo(matrix.dtype).tiny).log()).sum(-1)
    return quadrature * norm.squeeze(-2).square()


def logdet(matrix: Tensor, config: IterativeSettings | None = None) -> Tensor:
    config = current() if config is None else config
    if not enabled(matrix, "log_prob", config):
        factor = _rs.tensor_psd_safe_cholesky(matrix)
        return 2 * factor.diagonal(dim1=-2, dim2=-1).log().sum(-1)
    assert config is not None
    samples = config.num_trace_samples
    count = matrix.shape[-1]
    if samples == 0:
        probes = torch.eye(count, dtype=matrix.dtype, device=matrix.device)
    else:
        probes = matrix.new_empty(*matrix.shape[:-2], count, samples).bernoulli_(0.5) * 2 - 1
    estimates = _lanczos_log_quadrature(
        matrix, probes, min(config.max_iter, config.max_lanczos_quadrature_iterations)
    )
    return estimates.sum(-1) if samples == 0 else estimates.mean(-1)


def log_prob(target: Tensor, mean: Tensor, covariance: Tensor) -> Tensor:
    residual = target - mean
    inv = solve(covariance, residual.unsqueeze(-1)).squeeze(-1)
    return -0.5 * (
        (residual * inv).sum(-1) + logdet(covariance) + target.shape[-1] * math.log(2 * math.pi)
    )

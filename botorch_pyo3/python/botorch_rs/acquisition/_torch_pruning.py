"""Device-resident posterior winner selection for noisy baseline pruning."""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import Tensor

from botorch_rs.exceptions.errors import UnsupportedError
from botorch_rs.utils.multi_objective.pareto import is_non_dominated
from botorch_rs.utils.objective import (
    compute_feasibility_indicator,
    compute_smoothed_feasibility_indicator,
)
from botorch_rs.utils.transforms import is_ensemble


def prune_indices(
    model: Any,
    X: Tensor,
    objective: Any = None,
    posterior_transform: Any = None,
    constraints: Any = None,
    num_samples: int = 2048,
    max_frac: float = 1.0,
    sampler: Any = None,
    marginalize_dim: int | None = None,
) -> Tensor:
    if X.ndim != 2:
        raise UnsupportedError("Batched inputs X are unsupported by prune_inferior_points.")
    if not X.shape[-2]:
        raise ValueError("X must have at least one point.")
    if not 0 < max_frac <= 1:
        raise ValueError("max_frac must take values in (0, 1].")
    if marginalize_dim is None and is_ensemble(model):
        marginalize_dim = -3
    with torch.no_grad():
        posterior = model.posterior(X=X, posterior_transform=posterior_transform)
        if sampler is None:
            from botorch_rs.sampling.get_sampler import get_sampler

            sampler = get_sampler(posterior, sample_shape=torch.Size([num_samples]))
        samples = sampler(posterior)
        values = samples.squeeze(-1) if objective is None else objective(samples, X=X)
        if values.ndim > 2:
            if values.ndim != 3 or marginalize_dim is None:
                raise UnsupportedError(
                    "Pruning requires at most one marginalized model batch dimension."
                )
            dim = marginalize_dim
            if dim < 0:
                if not -values.ndim <= dim < 0:
                    raise IndexError("marginalize_dim is out of range.")
                dim = 1 + dim % values.ndim
            values = values.mean(dim=dim)
        infeasible = ~compute_feasibility_indicator(constraints, samples, marginalize_dim)
        if bool(infeasible.all()):
            values = compute_smoothed_feasibility_indicator(
                constraints or [], samples, eta=1e-3, log=True
            )
        elif bool(infeasible.any()):
            values = values.masked_fill(infeasible, values.amin() - 1)
        indices, counts = values.argmax(dim=-1).unique(return_counts=True)
        limit = math.ceil(max_frac * len(X))
        if len(indices) > limit:
            # Preserve the actual point labels; sorting frequency positions
            # alone is the known reference indexing defect.
            indices = indices[counts.argsort(stable=True, descending=True)[:limit]]
        return indices


def prune_indices_multi_objective(
    model: Any,
    X: Tensor,
    ref_point: Any,
    objective: Any = None,
    constraints: Any = None,
    num_samples: int = 2048,
    max_frac: float = 1.0,
    marginalize_dim: int | None = None,
    sampler: Any = None,
) -> Tensor:
    if X.ndim != 2:
        raise UnsupportedError("Batched inputs X are unsupported by multi-objective pruning.")
    if not X.shape[-2]:
        raise ValueError("X must have at least one point.")
    if not 0 < max_frac <= 1:
        raise ValueError("max_frac must take values in (0, 1].")
    if marginalize_dim is None and is_ensemble(model):
        marginalize_dim = -3
    with torch.no_grad():
        posterior = model.posterior(X=X)
        if sampler is None:
            from botorch_rs.sampling.get_sampler import get_sampler

            sampler = get_sampler(posterior, sample_shape=torch.Size([num_samples]))
        samples = sampler(posterior)
        values = samples if objective is None else objective(samples, X=X)
        if values.ndim > 3:
            if values.ndim != 4 or marginalize_dim is None:
                raise UnsupportedError(
                    "Pruning requires at most one marginalized model batch dimension."
                )
            values = values.mean(dim=marginalize_dim)
        infeasible = ~compute_feasibility_indicator(constraints, samples, marginalize_dim)
        ref = torch.as_tensor(ref_point, dtype=values.dtype, device=values.device)
        values = torch.where(infeasible.unsqueeze(-1), ref, values)
        mask = is_non_dominated(values, deduplicate=False) & (values > ref).all(dim=-1)
        probabilities = mask.to(X).mean(dim=0)
        indices = probabilities.nonzero().flatten()
        limit = math.ceil(max_frac * len(X))
        if len(indices) > limit:
            indices = probabilities.argsort(stable=True, descending=True)[:limit]
        multiplicity, remainder = divmod(values.shape[-2], X.shape[-2])
        if remainder or multiplicity < 1:
            raise ValueError("The objective point count must be a multiple of the input count.")
        return (indices // multiplicity).unique()

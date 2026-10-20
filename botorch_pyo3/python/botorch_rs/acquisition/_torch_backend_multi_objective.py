"""Tensor inclusion-exclusion hypervolume acquisitions, ported from BoTorch."""

from __future__ import annotations

import inspect
import itertools
import math
from typing import Any

import torch
from torch import Tensor, nn

from botorch_rs.acquisition._torch_backend_acquisition import _log_softplus, _TensorAcquisition
from botorch_rs.acquisition._torch_reductions import fatmax, logsumexp, smooth_amax
from botorch_rs.acquisition.multi_objective.objective import IdentityMCMultiOutputObjective
from botorch_rs.exceptions.errors import UnsupportedError
from botorch_rs.exceptions.warnings import legacy_ei_numerics_warning
from botorch_rs.sampling.normal import SobolQMCNormalSampler


def nondominated_mask(Y: Tensor, ref: Tensor, feasible: Tensor | None = None) -> Tensor:
    """Maximization Pareto membership; comparisons select differentiable values."""
    valid = (Y > ref).all(dim=-1)
    if feasible is not None:
        valid = valid & feasible
    a, b = Y.unsqueeze(-3), Y.unsqueeze(-2)
    dominates = (a >= b).all(-1) & (a > b).any(-1)
    dominates = dominates & valid.unsqueeze(-2)
    duplicate = (a == b).all(-1)
    indices = torch.arange(Y.shape[-2], device=Y.device)
    duplicate = duplicate & (indices.unsqueeze(-2) < indices.unsqueeze(-1))
    return valid & ~(dominates | duplicate).any(dim=-1)


def partition_2d(
    Y: Tensor, ref: Tensor, feasible: Tensor | None = None
) -> tuple[Tensor, Tensor, Tensor]:
    """Vectorized exact 2-D staircase cells with masked padding for ragged fronts."""
    n, batch = Y.shape[-2], Y.shape[:-2]
    reference = ref.expand(*batch, 2)
    if n == 0:
        return (
            reference.unsqueeze(-2),
            torch.full((*batch, 1, 2), torch.inf, dtype=Y.dtype, device=Y.device),
            torch.ones((*batch, 1), dtype=torch.bool, device=Y.device),
        )
    mask = nondominated_mask(Y, ref, feasible)
    order = Y[..., 0].masked_fill(~mask, torch.inf).argsort(dim=-1, stable=True)
    points = Y.gather(-2, order.unsqueeze(-1).expand(*order.shape, 2))
    count = mask.sum(dim=-1, keepdim=True)
    positions = torch.arange(n + 1, device=Y.device)
    lower_x = torch.cat((reference[..., :1], points[..., 0]), dim=-1)
    upper_x = torch.cat((points[..., 0], torch.full_like(reference[..., :1], torch.inf)), dim=-1)
    lower_y = torch.cat((points[..., 1], reference[..., 1:]), dim=-1)
    lower_y = torch.where(positions < count, lower_y, reference[..., 1:])
    upper_x = torch.where(positions < count, upper_x, torch.inf)
    lower = torch.stack((lower_x, lower_y), dim=-1)
    upper = torch.stack((upper_x, torch.full_like(upper_x, torch.inf)), dim=-1)
    active = positions <= count
    lower = torch.where(active.unsqueeze(-1), lower, reference.unsqueeze(-2))
    upper = torch.where(active.unsqueeze(-1), upper, reference.unsqueeze(-2) + 1)
    return lower, upper, active


def _log_difference(even: Tensor, odd: Tensor) -> Tensor:
    difference = even - odd.masked_fill(odd == -torch.inf, 0)
    near = difference.masked_fill(difference < -math.log(2), -math.log(2))
    far = difference.masked_fill(difference >= -math.log(2), -math.log(2))
    return odd + torch.where(difference > -math.log(2), (-near.expm1()).log(), (-far.exp()).log1p())


class TensorMultiObjectiveAcquisition(_TensorAcquisition):
    """Device-resident qEHVI and actual smoothed log qEHVI, without upstream calls."""

    ref_point: Tensor
    _cell_lower_tensor: Tensor
    _cell_upper_tensor: Tensor
    _observed_source: Tensor | None
    _reference_cells: Tensor

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        nn.Module.__init__(self)
        bound = inspect.signature(inspect.getattr_static(self._public_class, "__init__")).bind(
            self, *args, **kwargs
        )
        bound.apply_defaults()
        values = dict(bound.arguments)
        self._name = self._public_class.__name__
        self._tensor_backend = "torch"
        self._is_analytic = False
        self._log = self._name.startswith("qLog")
        self.model = values["model"]
        self.posterior_transform = None
        self.objective = values["objective"] or IdentityMCMultiOutputObjective()
        self.sampler = values["sampler"] or SobolQMCNormalSampler((128,))
        self._constraints = self.constraints = values["constraints"]
        self.register_buffer("_eta", torch.as_tensor(values["eta"]))
        self.fat = self._fat = values["fat"]
        self.register_buffer("_X_pending", None)
        self.set_X_pending(values["X_pending"])
        like = next(self.model.parameters())
        ref = torch.as_tensor(values["ref_point"], dtype=like.dtype, device=like.device)
        if ref.ndim != 1 or ref.numel() < 2:
            raise ValueError(
                "Hypervolume improvement requires a one-dimensional reference with m>=2."
            )
        self.register_buffer("ref_point", ref)
        if self._log:
            for field in ("tau_relu", "tau_max"):
                value = torch.as_tensor(values[field], dtype=torch.float64)
                if value.numel() != 1 or not (value > 0).all():
                    raise ValueError(f"{field} must be a positive scalar.")
                self.register_buffer(field, value)
        else:
            legacy_ei_numerics_warning(legacy_name=self._name)
        partitioning = values["partitioning"]
        if partitioning.num_outcomes != ref.numel():
            raise ValueError("The reference point must match the number of partitioning outcomes.")
        bounds = torch.as_tensor(
            partitioning.get_hypercell_bounds(), dtype=like.dtype, device=like.device
        )
        self.register_buffer("_cell_lower_tensor", bounds[0])
        self.register_buffer("_cell_upper_tensor", bounds[1])
        source = getattr(partitioning, "_tensor_source_Y", None)
        if source is None:
            source = getattr(partitioning, "Y", None)
        if (
            isinstance(source, Tensor)
            and (source.requires_grad or ref.requires_grad)
            and ref.numel() == 2
        ):
            if getattr(partitioning, "alpha", 0) != 0:
                raise UnsupportedError(
                    "Differentiable observed fronts require exact partitioning (alpha=0)."
                )
            self.register_buffer("_observed_source", source)
        else:
            if isinstance(source, Tensor) and source.requires_grad:
                raise UnsupportedError(
                    "Differentiable observed-front rebuilding currently requires two objectives."
                )
            self.register_buffer("_observed_source", None)
        old_ref = torch.as_tensor(partitioning.ref_point, dtype=like.dtype, device=like.device)
        self.register_buffer("_reference_cells", self.cell_lower_bounds == old_ref)
        self._subset_cache: dict[tuple[int, torch.device], dict[str, Tensor]] = {}

    @property
    def cell_lower_bounds(self) -> Tensor:
        return self._cell_lower_tensor

    @property
    def cell_upper_bounds(self) -> Tensor:
        return self._cell_upper_tensor

    def compute_q_subset_indices(self, q_out: int, device: Any) -> dict[str, Tensor]:
        self.q_out = q_out
        key = (q_out, torch.device(device))
        if key not in self._subset_cache:
            self._subset_cache[key] = {
                f"q_choose_{size}": torch.tensor(
                    list(itertools.combinations(range(q_out), size)),
                    dtype=torch.long,
                    device=device,
                )
                for size in range(1, q_out + 1)
            }
        return self._subset_cache[key]

    def _cells(self, obj: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        if self._observed_source is not None:
            return partition_2d(self._observed_source.to(obj), self.ref_point.to(obj))
        lower = self.cell_lower_bounds.to(obj)
        # The mathematical reference vertex stays differentiable even when a
        # fixed native decomposition supplied its other boundary coordinates.
        lower = torch.where(self._reference_cells, self.ref_point.to(obj), lower)
        upper = self.cell_upper_bounds.to(obj)
        return lower, upper, torch.ones(lower.shape[:-1], dtype=torch.bool, device=obj.device)

    def _draw_mo(self, X: Tensor) -> tuple[Tensor, Tensor]:
        samples = self.sampler(self.model.posterior(X))
        objective = self.objective(samples, X=X)
        if (
            not isinstance(objective, Tensor)
            or objective.shape[-2] != samples.shape[-2]
            or objective.shape[-1] != self.ref_point.numel()
        ):
            raise RuntimeError(
                "The objective must return sample_shape x batch_shape x q x m matching the reference point."
            )
        return samples, objective

    def _smooth_min(self, X: Tensor, dim: int) -> Tensor:
        tau = self.tau_max.to(X)
        return -(fatmax if self.fat else smooth_amax)(-X, dim=dim, tau=tau)

    def _hypervolume(self, samples: Tensor, obj: Tensor) -> Tensor:
        lower, upper, mask = self._cells(obj)
        batch = obj.shape[:-2]
        leading = (1,) * (len(batch) - len(lower.shape[:-2]))
        lower = lower.reshape(*leading, *lower.shape[:-2], lower.shape[-2], 1, obj.shape[-1])
        upper = upper.reshape(*leading, *upper.shape[:-2], upper.shape[-2], 1, obj.shape[-1])
        mask = mask.reshape(*leading, *mask.shape)
        subsets = self.compute_q_subset_indices(obj.shape[-2], obj.device)
        feasibility = (
            self._constraint_indicator(samples, self._log)
            if self._constraints is not None
            else None
        )
        terms: list[list[Tensor]] = [[], []]
        for size in range(1, obj.shape[-2] + 1):
            indices = subsets[f"q_choose_{size}"]
            selected = obj.index_select(-2, indices.flatten()).reshape(
                *batch, *indices.shape, obj.shape[-1]
            )
            if self._log:
                improvements = _log_softplus(
                    selected.unsqueeze(-4) - lower.unsqueeze(-3),
                    self.tau_relu.to(obj),
                    bool(self.fat),
                )
                overlap = self._smooth_min(improvements, -2)
                width = upper.clamp_max(1e10 if obj.dtype == torch.double else 1e8) - lower
                log_width = torch.where(mask.unsqueeze(-1).unsqueeze(-1), width, 1).log()
                paired = torch.stack(torch.broadcast_tensors(overlap, log_width), dim=-1)
                areas = self._smooth_min(paired, -1).sum(dim=-1)
            else:
                overlap = selected.min(dim=-2).values.unsqueeze(-3)
                areas = (torch.minimum(overlap, upper) - lower).clamp_min(0).prod(dim=-1)
            if feasibility is not None:
                weights = feasibility.index_select(-1, indices.flatten()).reshape(
                    *feasibility.shape[:-1], *indices.shape
                )
                weights = weights.unsqueeze(-3)
                areas = areas + weights.sum(-1) if self._log else areas * weights.prod(-1)
            areas = logsumexp(areas, -1) if self._log else areas.sum(-1)
            areas = areas.masked_fill(~mask, -torch.inf if self._log else 0)
            terms[size % 2].append(areas)
        if self._log:
            odd = logsumexp(torch.stack(terms[1]), 0)
            per_cell = (
                _log_difference(logsumexp(torch.stack(terms[0]), 0), odd) if terms[0] else odd
            )
            per_sample = logsumexp(per_cell, -1)
            dimensions = tuple(range(len(self.sample_shape)))
            return logsumexp(per_sample, dimensions) - math.log(math.prod(self.sample_shape))
        per_sample = (
            torch.stack(terms[1]).sum(0) - (torch.stack(terms[0]).sum(0) if terms[0] else 0)
        ).sum(-1)
        return per_sample.mean(dim=tuple(range(len(self.sample_shape))))

    def _compute_qehvi(self, samples: Tensor, X: Tensor | None = None) -> Tensor:
        return self._hypervolume(samples, self.objective(samples, X=X))

    def _compute_log_qehvi(self, samples: Tensor, X: Tensor | None = None) -> Tensor:
        return self._compute_qehvi(samples, X)

    def forward(self, X: Any) -> Tensor:
        X = self._candidates_tensor(X)
        samples, objective = self._draw_mo(X)
        return self._hypervolume(samples, objective)

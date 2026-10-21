"""Tensor inclusion-exclusion hypervolume acquisitions, ported from BoTorch."""

from __future__ import annotations

import copy
import inspect
import itertools
import math
import warnings
from typing import Any

import torch
from torch import Tensor, nn

from botorch_rs.acquisition._torch_backend_acquisition import (
    _average_ensemble,
    _log_softplus,
    _TensorAcquisition,
)
from botorch_rs.acquisition._torch_box_decomposition import (
    nondominated_mask,
    partition_binary,
    partition_nd,
)
from botorch_rs.acquisition._torch_reductions import fatmax, logsumexp, smooth_amax
from botorch_rs.acquisition.multi_objective.objective import IdentityMCMultiOutputObjective
from botorch_rs.exceptions.errors import UnsupportedError
from botorch_rs.exceptions.warnings import legacy_ei_numerics_warning
from botorch_rs.sampling.normal import NormalMCSampler, SobolQMCNormalSampler


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


def _partition(
    Y: Tensor, ref: Tensor, feasible: Tensor | None = None, alpha: float | None = None
) -> tuple[Tensor, Tensor, Tensor]:
    if Y.shape[-1] == 2:
        return partition_2d(Y, ref, feasible)
    return (
        partition_nd(Y, ref, feasible)
        if alpha is None
        else partition_binary(Y, ref, feasible, alpha)
    )


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
    _observed_alpha: float | None
    _reference_cells: Tensor

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        nn.Module.__init__(self)
        bound = inspect.signature(inspect.getattr_static(self._public_class, "__init__")).bind(
            self, *args, **kwargs
        )
        bound.apply_defaults()
        values = dict(bound.arguments)
        self._init_common(values)
        self.set_X_pending(values["X_pending"])
        self._init_partition(values["partitioning"])

    def _init_common(self, values: dict[str, Any]) -> None:
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
        self._subset_cache: dict[tuple[int, torch.device], dict[str, Tensor]] = {}

    def _init_partition(self, partitioning: Any) -> None:
        ref = self.ref_point
        like = ref
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
        self._observed_alpha = None
        if isinstance(source, Tensor) and (source.requires_grad or ref.requires_grad):
            fast = getattr(partitioning, "_rs_kind", None) == "fast_nondominated" or any(
                cls.__name__ == "FastNondominatedPartitioning" for cls in type(partitioning).__mro__
            )
            self._observed_alpha = None if fast else float(getattr(partitioning, "alpha", 0))
            self.register_buffer("_observed_source", source)
        else:
            self.register_buffer("_observed_source", None)
        old_ref = torch.as_tensor(partitioning.ref_point, dtype=like.dtype, device=like.device)
        self.register_buffer("_reference_cells", self.cell_lower_bounds == old_ref)

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
            return _partition(
                self._observed_source.to(obj), self.ref_point.to(obj), alpha=self._observed_alpha
            )
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

    def _hypervolume(
        self, samples: Tensor, obj: Tensor, cells: tuple[Tensor, Tensor, Tensor] | None = None
    ) -> Tensor:
        lower, upper, mask = self._cells(obj) if cells is None else cells
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
        return _average_ensemble(self._hypervolume(samples, objective), self.model, self._log)


def _dominated_volume(Y: Tensor, ref: Tensor, feasible: Tensor | None = None) -> Tensor:
    lower, upper, active = _partition(Y, ref, feasible)
    if Y.shape[-1] > 2:
        valid = (Y > ref).all(-1)
        if feasible is not None:
            valid = valid & feasible
        ideal = (
            torch.where(valid.unsqueeze(-1), Y, ref).amax(-2)
            if Y.shape[-2]
            else ref.expand(*Y.shape[:-2], Y.shape[-1])
        )
        side = (torch.minimum(upper, ideal.unsqueeze(-2)) - lower).clamp_min(0)
        nondominated = side.prod(-1).masked_fill(~active, 0).sum(-1)
        return (ideal - ref).clamp_min(0).prod(-1) - nondominated
    finite = torch.isfinite(upper[..., 0]) & active
    widths = torch.where(finite, upper[..., 0], lower[..., 0]) - lower[..., 0]
    return (widths * (lower[..., 1] - ref[..., 1])).sum(-1)


class TensorNoisyMultiObjectiveAcquisition(TensorMultiObjectiveAcquisition):
    """Exact joint noisy hypervolume sampling, with device-only Pareto cells."""

    _X_baseline: Tensor
    base_sampler: Any
    _observed_sampler: Any
    _last_baseline_objective: Tensor | None
    _last_cell_mask: Tensor | None

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        nn.Module.__init__(self)
        kwargs = dict(kwargs)
        self.detach_baseline = bool(kwargs.pop("detach_baseline", True))
        bound = inspect.signature(inspect.getattr_static(self._public_class, "__init__")).bind(
            self, *args, **kwargs
        )
        bound.apply_defaults()
        values = dict(bound.arguments)
        self._init_common(values)
        if values["objective"] is None and self.ref_point.numel() != self.model.num_outputs:
            raise ValueError(
                "The reference point must match the model outcomes unless an objective selects or expands outputs."
            )
        self.marginalize_dim = values["marginalize_dim"]
        self.alpha = float(values["alpha"])
        baseline = values["X_baseline"]
        if not isinstance(baseline, Tensor) or baseline.ndim != 2:
            raise ValueError("X_baseline must be an n x d Torch tensor.")
        if not callable(getattr(self.sampler, "_update_base_samples", None)):
            raise UnsupportedError(
                "Noisy hypervolume requires a normal sampler supporting baseline base-sample updates."
            )
        self.register_buffer("_X_baseline", baseline)
        self.cache_root = self._cache_root = False
        if values["cache_root"]:
            warnings.warn(
                "backend='torch' uses full-joint noisy hypervolume sampling; setting cache_root=False.",
                RuntimeWarning,
                stacklevel=3,
            )
        self.cache_pending = bool(values["cache_pending"])
        self._max_iep = int(values["max_iep"])
        if self._max_iep < 0:
            raise ValueError("max_iep must be non-negative.")
        self.incremental_nehvi = bool(values["incremental_nehvi"])
        self._cached_pending_count = 0
        self._prefix_can_extend = False
        self._last_prefix_count = -1
        pending = values["X_pending"]
        self._initial_pending_seed_full = bool(
            self.cache_pending and pending is not None and pending.shape[-2] > self._max_iep
        )
        self.base_sampler = None
        self._observed_sampler = self._new_sampler()
        self.prune_baseline = bool(values["prune_baseline"])
        if self.prune_baseline:
            self._prune_front()
        self.register_buffer("_cell_lower_tensor", None)
        self.register_buffer("_cell_upper_tensor", None)
        self._last_baseline_objective = None
        self._last_cell_mask = None
        self.set_X_pending(values["X_pending"])

    def _ensure_baseline_geometry(self) -> None:
        if self._last_baseline_objective is not None:
            return
        baseline = self.X_baseline
        if baseline.shape[-2]:
            posterior = self.model.posterior(baseline)
            samples = self._joint_samples(posterior, baseline.shape[-2])
            if self.detach_baseline:
                samples = samples.detach()
        else:
            samples = self.ref_point.new_empty(
                (*self.sample_shape, *self.model.batch_shape, 0, self.model.num_outputs)
            )
        objective = self.objective(samples, X=baseline)
        cells = _partition(
            objective,
            self.ref_point,
            self._feasible(samples),
            alpha=self.alpha if self.alpha > 0 else None,
        )
        self._cell_lower_tensor, self._cell_upper_tensor, self._last_cell_mask = cells
        self._last_baseline_objective = objective

    @property
    def cell_lower_bounds(self) -> Tensor:
        self._ensure_baseline_geometry()
        return self._cell_lower_tensor

    @property
    def cell_upper_bounds(self) -> Tensor:
        self._ensure_baseline_geometry()
        return self._cell_upper_tensor

    def _cells(self, obj: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        self._ensure_baseline_geometry()
        assert self._last_cell_mask is not None
        return self._cell_lower_tensor, self._cell_upper_tensor, self._last_cell_mask

    def _new_sampler(self) -> Any:
        if isinstance(self.sampler, NormalMCSampler):
            sampler_type: Any = type(self.sampler)
            return sampler_type(sample_shape=self.sample_shape, seed=self.sampler.seed)
        return copy.deepcopy(self.sampler)

    @property
    def X_baseline(self) -> Tensor:
        if self._cached_pending_count and self._X_pending is not None:
            return torch.cat(
                (self._X_baseline, self._X_pending[: self._cached_pending_count]), dim=-2
            )
        return self._X_baseline

    def set_X_pending(self, X_pending: Any = None) -> None:
        previous = self._X_pending
        old_count = self._cached_pending_count
        if X_pending is not None and (not isinstance(X_pending, Tensor) or X_pending.ndim != 2):
            raise ValueError("Noisy hypervolume pending points must be an n x d tensor.")
        _TensorAcquisition.set_X_pending(self, X_pending)
        current = self._X_pending
        append = (
            previous is not None
            and current is not None
            and current.shape[-2] >= previous.shape[-2]
            and torch.equal(current[: previous.shape[-2]], previous)
        )
        if not self.cache_pending or current is None:
            new_count = 0
        elif append and current.shape[-2] - old_count <= self._max_iep:
            new_count = old_count
        else:
            new_count = current.shape[-2] if current.shape[-2] > self._max_iep else 0
        self._cached_pending_count = new_count
        self._prefix_can_extend = append and new_count >= old_count
        if not append or new_count != old_count:
            self._last_prefix_count = -1
        self._last_baseline_objective = None
        self._last_cell_mask = None

    def _prune_front(self) -> None:
        if not self._X_baseline.shape[-2]:
            return
        from botorch_rs.acquisition._torch_pruning import prune_indices_multi_objective

        indices = prune_indices_multi_objective(
            self.model,
            self._X_baseline,
            self.ref_point,
            objective=self.objective,
            constraints=self._constraints,
            marginalize_dim=self.marginalize_dim,
            sampler=SobolQMCNormalSampler((2048,), seed=self.sampler.seed),
        )
        self._X_baseline = self._X_baseline.index_select(-2, indices)

    @staticmethod
    def _prefix_posterior(posterior: Any, count: int) -> Any:
        from botorch_rs.posteriors._torch_backend import TorchGPyTorchPosterior

        underlying = posterior
        while hasattr(underlying, "_posterior"):
            underlying = underlying._posterior
        mean = underlying.mean[..., :count, :]
        q, m = underlying.mean.shape[-2:]
        indices = (
            torch.arange(m, device=mean.device).unsqueeze(-1) * q
            + torch.arange(count, device=mean.device)
        ).flatten()
        covariance = underlying.distribution.covariance_matrix.index_select(
            -2, indices
        ).index_select(-1, indices)
        return TorchGPyTorchPosterior(mean, covariance)

    def _joint_samples(self, posterior: Any, baseline_count: int) -> Tensor:
        original = self._X_baseline.shape[-2]
        if self._initial_pending_seed_full and self.base_sampler is None:
            # Reference incremental NEHVI initializes directly on B+P, so
            # Sobol's dimension is that complete prefix on the first draw.
            prefix = self._prefix_posterior(posterior, baseline_count)
            self.base_sampler = self._new_sampler()
            self.base_sampler._construct_base_samples(prefix)
            if original:
                self._observed_sampler._construct_base_samples(
                    self._prefix_posterior(posterior, original)
                )
                if self.incremental_nehvi:
                    normals = self.base_sampler.base_samples
                    point_axis = -2 if prefix._is_mt else -1
                    normals = normals.narrow(point_axis, 0, original)
                    if isinstance(self._observed_sampler, SobolQMCNormalSampler):
                        self._observed_sampler._base_samples = normals
                    else:
                        self._observed_sampler.base_samples = normals
                    self._observed_sampler._base_shape = tuple(normals.shape)
            self._last_prefix_count = baseline_count
            self._initial_pending_seed_full = False
        if original and self._observed_sampler.base_samples is None:
            self._observed_sampler._construct_base_samples(
                self._prefix_posterior(posterior, original)
            )
        if baseline_count and self._last_prefix_count != baseline_count:
            source = (
                self.base_sampler
                if self._prefix_can_extend and self.base_sampler is not None
                else self._observed_sampler
            )
            sampler = self._new_sampler()
            sampler._update_base_samples(self._prefix_posterior(posterior, baseline_count), source)
            self.base_sampler = sampler
            self._last_prefix_count = baseline_count
        if baseline_count:
            self.sampler._update_base_samples(posterior, self.base_sampler)
        return self.sampler(posterior)

    def forward(self, X: Any) -> Tensor:
        if not isinstance(X, Tensor) or X.ndim < 2:
            raise ValueError("X must be a Torch tensor with shape batch_shape x q x d.")
        X = X.unsqueeze(0) if X.ndim == 2 else X
        if self._X_pending is not None:
            pending = self._X_pending[self._cached_pending_count :].to(X)
            if pending.shape[-2]:
                X = torch.cat((X, pending.expand(*X.shape[:-2], *pending.shape[-2:])), dim=-2)
        baseline = self.X_baseline.to(X)
        joint = torch.cat((baseline.expand(*X.shape[:-2], *baseline.shape[-2:]), X), dim=-2)
        count = baseline.shape[-2]
        posterior = self.model.posterior(joint)
        samples = self._joint_samples(posterior, count)
        if count:
            # Source partition geometry uses the standalone prefix draw.
            # A repeated query can require jitter in the full joint factor;
            # that stabilization must not perturb the fixed baseline frontier.
            baseline_samples = self.base_sampler(self.model.posterior(joint[..., :count, :]))
        else:
            baseline_samples = samples[..., :0, :]
        if self.detach_baseline:
            baseline_samples = baseline_samples.detach()
        processed = torch.cat((baseline_samples, samples[..., count:, :]), dim=-2)
        objective = self.objective(processed, X=joint)
        baseline_obj, candidate_obj = objective[..., :count, :], objective[..., count:, :]
        self.baseline_samples, self.baseline_obj = baseline_samples, baseline_obj
        self._last_baseline_objective = baseline_obj
        feasible = self._feasible(baseline_samples)
        cells = _partition(
            baseline_obj,
            self.ref_point.to(objective),
            feasible,
            alpha=self.alpha if self.alpha > 0 else None,
        )
        self._cell_lower_tensor, self._cell_upper_tensor, self._last_cell_mask = cells
        value = self._hypervolume(samples[..., count:, :], candidate_obj, cells)
        if not self.incremental_nehvi and self._cached_pending_count:
            n = self._X_baseline.shape[-2]
            pending_hv = _dominated_volume(baseline_obj, self.ref_point, feasible)
            if n:
                observed_samples = self._observed_sampler(self._prefix_posterior(posterior, n))
                if self.detach_baseline:
                    observed_samples = observed_samples.detach()
                observed_obj = self.objective(observed_samples, X=self._X_baseline)
                observed_hv = _dominated_volume(
                    observed_obj, self.ref_point, self._feasible(observed_samples)
                )
                if self.detach_baseline:
                    observed_hv = observed_hv.detach()
            else:
                observed_hv = pending_hv * 0
            previous = (pending_hv - observed_hv).clamp_min(0).mean()
            value = torch.logaddexp(value, previous.log()) if self._log else value + previous
        return _average_ensemble(value, self.model, self._log)

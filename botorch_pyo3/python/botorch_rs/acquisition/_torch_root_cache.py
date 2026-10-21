"""Graph-safe independent Gaussian baseline roots and Schur sampling."""

from __future__ import annotations

import copy
import warnings
from typing import Any

import torch
from torch import Tensor

from botorch_rs import _botorch_rs as _rs
from botorch_rs.acquisition.monte_carlo import supports_cache_root
from botorch_rs.exceptions.errors import BotorchError, NanError, NotPSDError
from botorch_rs.exceptions.warnings import BotorchWarning
from botorch_rs.posteriors._base_samples import expand_base_samples
from botorch_rs.posteriors._torch_backend import TorchGPyTorchPosterior
from botorch_rs.sampling.normal import NormalMCSampler
from botorch_rs.utils._torch_low_rank import covariance_blocks, sample_cached_cholesky


def configure(owner: Any, requested: bool | None) -> None:
    supported = supports_cache_root(owner.model) and isinstance(owner.sampler, NormalMCSampler)
    owner.cache_root = owner._cache_root = (
        supported if requested is None else bool(requested and supported)
    )
    if requested and not supported:
        warnings.warn(
            "`cache_root` requires an independent Gaussian model and a normal sampler; "
            "setting cache_root=False.",
            RuntimeWarning,
            stacklevel=3,
        )
    owner._tensor_root_cache = TensorRootCache()
    owner.register_buffer("_baseline_L", None)


def new_sampler(sampler: Any) -> Any:
    # Preserve custom batch-range/normal-sampler configuration without
    # deep-copying user supplied non-leaf normal draws.
    result = copy.copy(sampler)
    if isinstance(result, torch.nn.Module):
        # nn.Module's shallow copy otherwise shares the registered dictionaries.
        for name in ("_buffers", "_parameters", "_modules"):
            setattr(result, name, getattr(result, name).copy())
        result._non_persistent_buffers_set = result._non_persistent_buffers_set.copy()
    result.base_samples = None
    result._base_samples_explicit = False
    result._base_shape = None
    result._tensor_sampler_key = None
    return result


def prefix_sampler(sampler: Any, posterior: Any) -> Any:
    """Resize a prefix while retaining the common normal coordinates."""
    previous = sampler.base_samples
    if previous is None:
        sampler._construct_base_samples(posterior)
        return sampler
    point_axis = -2 if posterior._is_mt else -1
    if previous.shape[point_axis] == posterior._q:
        # Do not regenerate a prefix populated by _update_base_samples: its
        # old coordinates intentionally differ from a fresh draw of this size.
        # Samplers are not nn.Modules, so migrate their retained normals here.
        current = previous.to(dtype=posterior.dtype, device=posterior.device)
        if hasattr(sampler, "_base_samples"):
            sampler._base_samples = current
        else:
            sampler.base_samples = current
        sampler._base_shape = tuple(current.shape)
        return sampler
    result = new_sampler(sampler)
    result._construct_base_samples(posterior)
    target = result.base_samples
    count = min(previous.shape[point_axis], posterior._q)
    prefix = previous.narrow(point_axis, 0, count).to(target)
    sample_dims = len(result.sample_shape)
    # A joint query can add collapsed singleton t-batch dimensions. They do
    # not introduce new random coordinates when returning to a shorter prefix.
    event_dims = 2 if posterior._is_mt else 1
    tail = prefix.shape[sample_dims:-event_dims]
    if prefix.ndim > target.ndim and all(size == 1 for size in tail):
        prefix = prefix.reshape(
            *result.sample_shape,
            *([1] * (target.ndim - sample_dims - event_dims)),
            *prefix.shape[-event_dims:],
        )
    prefix = prefix.reshape(
        *result.sample_shape,
        *([1] * (target.ndim - prefix.ndim)),
        *prefix.shape[sample_dims:],
    )
    shape = list(target.shape)
    shape[point_axis] = count
    updated = torch.cat(
        (prefix.expand(shape), target.narrow(point_axis, count, posterior._q - count)),
        dim=point_axis,
    )
    if hasattr(result, "_base_samples"):
        result._base_samples = updated
    else:
        result.base_samples = updated
    result._base_shape = tuple(updated.shape)
    return result


def _model_key(model: Any) -> Any:
    method = getattr(model, "_prediction_key", None)
    if method is not None:
        return method()
    children = getattr(model, "models", None)
    if children is not None:
        keys = tuple((id(child), _model_key(child)) for child in children)
        return keys if all(key is not None for _, key in keys) else None
    return None


def _transform_key(transform: Any) -> Any:
    if transform is None:
        return ()
    from botorch_rs.acquisition.objective import ScalarizedPosteriorTransform

    if type(transform) is not ScalarizedPosteriorTransform:
        return None
    result = []
    for value in (transform.weights, transform.offset):
        if isinstance(value, Tensor):
            if torch.is_grad_enabled() and value.requires_grad:
                return None
            try:
                result.append(
                    (id(value), value.data_ptr(), value._version, value.dtype, value.device)
                )
            except RuntimeError:
                return None
        elif isinstance(value, (list, tuple)):
            result.append(tuple(value))
        else:
            result.append(value)
    return tuple(result)


class TensorRootCache:
    """Keep at most the original and current frozen prefix; live graphs rebuild."""

    def __init__(self) -> None:
        self.key: Any = None
        self.entries: dict[int, tuple[Tensor, Tensor, Tensor]] = {}

    def factor(self, model: Any, points: Tensor, posterior: Any, transform: Any = None) -> Tensor:
        covariance = covariance_blocks(posterior.distribution, posterior._q, posterior._m)
        model_key, transform_key = _model_key(model), _transform_key(transform)
        key = (
            None
            if model_key is None
            or transform_key is None
            or (torch.is_grad_enabled() and points.requires_grad)
            else (model_key, transform_key)
        )
        if key != self.key:
            self.entries.clear()
            self.key = key
        n = posterior._q
        if key is not None:
            entry = self.entries.get(n)
            if (
                entry is not None
                and torch.equal(points, entry[0])
                and torch.equal(covariance, entry[1])
            ):
                return entry[2]
            # Append-only pending prefixes can extend their leading root.
            for count in sorted(self.entries, reverse=True):
                old_points, old_covariance, old_root = self.entries[count]
                if (
                    count < n
                    and torch.equal(points[..., :count, :], old_points)
                    and torch.equal(covariance[..., :count, :count], old_covariance)
                ):
                    cross = covariance[..., count:, :count]
                    lower = torch.linalg.solve_triangular(old_root, cross.mT, upper=False).mT
                    corner = covariance[..., count:, count:] - lower @ lower.mT
                    corner_root = _rs.tensor_psd_safe_cholesky(corner)
                    top = torch.cat((old_root, torch.zeros_like(lower.mT)), -1)
                    root = torch.cat((top, torch.cat((lower, corner_root), -1)), -2)
                    break
            else:
                root = _rs.tensor_psd_safe_cholesky(covariance)
            self.entries[n] = (
                points.detach().clone(),
                covariance.detach().clone(),
                root.detach().clone(),
            )
            if len(self.entries) > 2:
                first = min(self.entries)
                self.entries = {
                    count: entry for count, entry in self.entries.items() if count in (first, n)
                }
            return self.entries[n][2]
        # Never reuse an autograd graph across forward/backward pairs. The
        # Schur candidate solve still avoids factoring the full joint matrix.
        return _rs.tensor_psd_safe_cholesky(covariance)


def root_samples(posterior: Any, root: Tensor, normals: Tensor, sample_shape: Any) -> Tensor:
    # Samplers may retain redundant collapsed t-batch axes from a joint law.
    event_ndim = 2 if posterior._is_mt else 1
    sample_shape = tuple(sample_shape)
    event = normals.shape[-event_ndim:]
    tail = normals.shape[len(sample_shape) : -event_ndim]
    if len(tail) > len(posterior.batch_shape) and all(size == 1 for size in tail):
        normals = normals.reshape(*sample_shape, *([1] * len(posterior.batch_shape)), *event)
    normals = expand_base_samples(
        normals,
        sample_shape,
        posterior.base_sample_shape,
        None if posterior._is_mt else posterior._q,
    )
    if not posterior._is_mt:
        normals = normals.unsqueeze(-1)
    return posterior.mean + (root @ normals.transpose(-1, -2).unsqueeze(-1)).squeeze(-1).transpose(
        -1, -2
    )


def joint_samples(owner: Any, posterior: Any, baseline: Tensor, base_sampler: Any) -> Tensor | None:
    """Complete common-normal draw, retaining joint objective correlations."""
    owner._last_root_cache_used = False
    if not owner._cache_root:
        return None
    if not isinstance(posterior, TorchGPyTorchPosterior):
        owner.cache_root = owner._cache_root = False
        warnings.warn(
            "Root caching does not support non-Gaussian posterior sampling; setting cache_root=False.",
            RuntimeWarning,
            stacklevel=3,
        )
        return None
    try:
        prefix = owner._posterior(baseline)
        if not isinstance(prefix, TorchGPyTorchPosterior) or prefix._q != baseline.shape[-2]:
            owner.cache_root = owner._cache_root = False
            warnings.warn(
                "Root caching requires matching Gaussian point layouts; setting cache_root=False.",
                RuntimeWarning,
                stacklevel=3,
            )
            return None
        factor = owner._tensor_root_cache.factor(
            owner.model, baseline, prefix, owner.posterior_transform
        )
        owner._baseline_L = factor if prefix._is_mt else factor.squeeze(-3)
        base_sampler = prefix_sampler(base_sampler, prefix)
        owner.base_sampler = base_sampler
        owner.sampler._update_base_samples(posterior, base_sampler)
        q = posterior._q - prefix._q
        if q < 1:
            # Geometry-only prefix draws need no conditional factorization.
            result = root_samples(prefix, factor, owner.sampler.base_samples, owner.sample_shape)
            owner._last_root_cache_used = True
            return result
        candidates = sample_cached_cholesky(
            posterior, owner._baseline_L, q, owner.sampler.base_samples, owner.sample_shape, 6
        )
        initial = root_samples(prefix, factor, base_sampler.base_samples, owner.sample_shape)
        sample_dims = len(owner.sample_shape)
        missing = candidates.ndim - initial.ndim
        if missing:
            initial = initial.reshape(
                *initial.shape[:sample_dims], *([1] * missing), *initial.shape[sample_dims:]
            )
        initial = initial.expand(*candidates.shape[:-2], prefix._q, prefix._m)
        owner._last_root_cache_used = True
        return torch.cat((initial, candidates), -2)
    except BotorchError as error:
        if isinstance(error, (NanError, NotPSDError)):
            warnings.warn(
                "Low-rank Cholesky update failed; falling back to full-joint sampling.",
                BotorchWarning,
                stacklevel=3,
            )
            return None
        if "outputs are correlated" not in str(error):
            raise
        owner.cache_root = owner._cache_root = False
        warnings.warn(
            "Root caching requires independent Gaussian outputs; setting cache_root=False.",
            RuntimeWarning,
            stacklevel=3,
        )
        return None

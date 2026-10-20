"""Native cached-Cholesky sampling for Gaussian acquisition functions."""

from __future__ import annotations

import math
import warnings
from typing import Any

from botorch_rs import _botorch_rs as _rs
from botorch_rs import _ops
from botorch_rs._arrays import FloatData, from_flat, to_flat
from botorch_rs.acquisition.acquisition import MCSamplerMixin
from botorch_rs.acquisition.monte_carlo import supports_cache_root
from botorch_rs.exceptions.errors import NanError, NotPSDError, UnsupportedError
from botorch_rs.exceptions.warnings import BotorchWarning
from botorch_rs.posteriors.gpytorch import GPyTorchPosterior
from botorch_rs.utils.low_rank import _blocks, sample_cached_cholesky


class CachedCholeskyMCSamplerMixin(MCSamplerMixin):
    """Reuse native Gaussian baseline roots and normals for candidate samples."""

    q_in: int
    base_sampler: Any
    model: Any

    def __init__(self, model: Any, cache_root: bool | None = None, sampler: Any = None) -> None:
        MCSamplerMixin.__init__(self, sampler)
        supported = supports_cache_root(model)
        if cache_root is None:
            cache_root = supported
        elif cache_root and not supported:
            warnings.warn(
                "`cache_root` is only supported for GPyTorchModels that are not MultiTask "
                "models and don't produce a TransformedPosterior. "
                f"Got a model of type {type(model)}. Setting `cache_root = False`.",
                RuntimeWarning,
                stacklevel=3,
            )
            cache_root = False
        self._cache_root = cache_root

    def _compute_root_decomposition(self, posterior: Any) -> Any:
        if not isinstance(posterior, GPyTorchPosterior):
            raise UnsupportedError("Cached roots require a botorch_rs Gaussian posterior.")
        covariance: FloatData
        if posterior._is_mt:
            covariance, shape = _blocks(posterior)
        else:
            _, (covariance, shape) = posterior._get_joint()
        n = shape[-1]
        root, _ = _rs.psd_safe_cholesky(list(covariance), math.prod(shape[:-2]), n)
        return from_flat(root, tuple(shape), posterior._kind)

    def _get_f_X_samples(self, posterior: Any, q_in: int) -> Any:
        if self._cache_root and hasattr(self, "_baseline_L"):
            try:
                return sample_cached_cholesky(
                    posterior,
                    self._baseline_L,
                    q_in,
                    self.sampler.base_samples,
                    self.sampler.sample_shape,
                )
            except (NanError, NotPSDError):
                warnings.warn(
                    "Low-rank cholesky updates failed due NaNs or due to an "
                    "ill-conditioned covariance matrix. Falling back to standard sampling.",
                    BotorchWarning,
                    stacklevel=3,
                )
        samples = self.get_posterior_samples(posterior)
        from botorch_rs.models.higher_order_gp import HigherOrderGP

        axis = -self.model._num_dimensions if isinstance(self.model, HigherOrderGP) else -2
        shape = to_flat(samples)[1]
        index = [slice(None)] * len(shape)
        index[axis] = slice(shape[axis] - q_in, shape[axis])
        return _ops.getitem(samples, tuple(index))

    def _set_sampler(self, q_in: int, posterior: Any) -> None:
        if self.q_in != q_in and self.base_sampler is not None:
            self.sampler._update_base_samples(posterior=posterior, base_sampler=self.base_sampler)
            self.q_in = q_in


__all__ = ["CachedCholeskyMCSamplerMixin", "supports_cache_root"]

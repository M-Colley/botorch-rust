"""Device-resident statistics for the public standardization descriptors."""

from __future__ import annotations

import warnings
from typing import Any

import torch
from torch import Tensor, nn


def standardize_statistics(Y: Tensor, minimum: float) -> tuple[Tensor, Tensor]:
    """Unbiased, NaN-aware statistics with finite constant-data derivatives."""
    means = Y.nanmean(dim=-2, keepdim=True)
    if Y.shape[-2] == 1:
        return means, torch.ones_like(means)
    valid = ~Y.isnan()
    count = valid.sum(dim=-2, keepdim=True)
    # Avoid NaN arithmetic even in the inactive branches of the reduction.
    differences = torch.where(valid, Y, 0) - torch.where(valid, means, 0)
    variance = differences.square().sum(dim=-2, keepdim=True) / (count - 1).clamp_min(1)
    stdvs = variance.clamp_min(minimum**2).sqrt()
    stdvs = torch.where((count > 1) & (variance >= minimum**2), stdvs, 1.0)
    return means, stdvs


def apply_standardize(
    Y: Tensor,
    Yvar: Tensor | None,
    means: Tensor,
    stdvs: Tensor,
    squared: Tensor,
    inverse: bool = False,
) -> tuple[Tensor, Tensor | None]:
    """Preserve missing observations without NaN arithmetic in their VJPs."""
    valid = ~Y.isnan()
    values = torch.where(valid, Y, 0)
    transformed = values * stdvs + means if inverse else (values - means) / stdvs
    result = torch.where(valid, transformed, Y)
    variance = None
    if Yvar is not None:
        observed = ~Yvar.isnan()
        values = torch.where(observed, Yvar, 0)
        transformed = values * squared if inverse else values / squared
        variance = torch.where(observed, transformed, Yvar)
    return result, variance


class TensorStandardizeState(nn.Module):
    """Actual Torch buffers; public descriptors retain optional-dependency support."""

    _tensor_backend = "torch"
    _is_linear = True
    means: Tensor
    stdvs: Tensor
    _stdvs_sq: Tensor
    _is_trained: Tensor
    strata_mapping: Tensor

    def __init__(self, source: Any, like: Tensor) -> None:
        super().__init__()
        self._m = source._m
        self._batch_shape = source._batch_shape
        self._min_stdv = source._min_stdv
        self._outputs = source._outputs
        self._stratification_idx = getattr(source, "_stratification_idx", None)
        self._task_values = tuple(getattr(source, "_task_values", ()))
        if self._stratification_idx is None:
            shape = (*self._batch_shape, 1, self._m)
        else:
            if not self._task_values:
                raise ValueError("StratifiedStandardize requires at least one task value.")
            tasks = torch.tensor(self._task_values, dtype=torch.long, device=like.device)
            contiguous = self._task_values == tuple(range(len(self._task_values)))
            mapping = (
                tasks.to(dtype=getattr(source, "_task_values_dtype", None) or torch.long)
                if contiguous
                else torch.full(
                    (max(self._task_values) + 1,),
                    float("nan"),
                    device=like.device,
                    dtype=getattr(source, "_strata_dtype", None) or torch.double,
                )
            )
            if not contiguous:
                mapping[tasks] = torch.arange(len(tasks), device=like.device, dtype=mapping.dtype)
            self.register_buffer("strata_mapping", mapping, persistent=False)
            shape = (*self._batch_shape, mapping.numel(), 1)
        self.register_buffer("means", like.new_zeros(shape))
        self.register_buffer("stdvs", like.new_ones(shape))
        self.register_buffer("_stdvs_sq", like.new_ones(shape))
        self.register_buffer("_is_trained", torch.tensor(False, device=like.device))
        self.train(source.training)

    @property
    def _requires_X(self) -> bool:
        return self._stratification_idx is not None

    def _validate(self, Y: Tensor, Yvar: Tensor | None, X: Tensor | None) -> None:
        if Y.ndim < 2 or Y.shape[-1] != self._m:
            raise RuntimeError(f"Expected outcomes with shape ... x n x {self._m}.")
        if Yvar is not None:
            try:
                torch.broadcast_shapes(Yvar.shape, Y.shape)
            except RuntimeError as error:
                raise RuntimeError("Yvar and Y must have broadcast-compatible shapes.") from error
        if self._stratification_idx is not None:
            if X is None:
                raise ValueError("X is required for StratifiedStandardize.")
            if X.shape[-2] != Y.shape[-2]:
                raise RuntimeError("X and Y must have matching observation shapes.")
            torch.broadcast_shapes(X.shape[:-2], Y.shape[:-2])

    def fit(self, Y: Tensor, X: Tensor | None = None) -> TensorStandardizeState:
        self._validate(Y, None, X)
        if Y.shape[:-2] != self._batch_shape or Y.shape[-2] == 0:
            raise ValueError("Standardize training targets must match batch_shape and be nonempty.")
        if self._stratification_idx is None:
            means, stdvs = standardize_statistics(Y, self._min_stdv)
            if self._outputs is not None:
                used = torch.zeros(self._m, dtype=torch.bool, device=Y.device)
                used[self._outputs] = True
                means = torch.where(used, means, 0.0)
                stdvs = torch.where(used, stdvs, 1.0)
        else:
            assert X is not None
            self.means = self.means.to(Y)
            self.stdvs = self.stdvs.to(Y)
            self.strata_mapping = self.strata_mapping.to(device=Y.device)
            strata = X[..., self._stratification_idx].long()
            means_per_task, stdvs_per_task = [], []
            for index, task in enumerate(self._task_values):
                observed = (strata == task).unsqueeze(-1) & ~Y.isnan()
                count = observed.sum(dim=-2)
                values = torch.where(observed, Y, 0)
                mean = values.sum(-2) / count.clamp_min(1)
                difference = values - torch.where(observed, mean.unsqueeze(-2), 0)
                variance = difference.square().sum(-2) / (count - 1).clamp_min(1)
                std = variance.clamp_min(self._min_stdv**2).sqrt()
                std = torch.where((count > 1) & (variance >= self._min_stdv**2), std, 1.0)
                means_per_task.append(torch.where(count > 0, mean, self.means[..., index, :]))
                stdvs_per_task.append(torch.where(count > 0, std, self.stdvs[..., index, :]))
            # Non-contiguous task mappings can contain unused table rows.
            for index in range(len(self._task_values), self.means.shape[-2]):
                means_per_task.append(self.means[..., index, :])
                stdvs_per_task.append(self.stdvs[..., index, :])
            means, stdvs = torch.stack(means_per_task, -2), torch.stack(stdvs_per_task, -2)
        self.means, self.stdvs, self._stdvs_sq = means, stdvs, stdvs.square()
        self._is_trained = torch.tensor(True, device=Y.device)
        return self

    def statistics(self, X: Tensor | None = None) -> tuple[Tensor, Tensor, Tensor]:
        if self._stratification_idx is None:
            return self.means, self.stdvs, self._stdvs_sq
        if X is None:
            raise ValueError("X is required for StratifiedStandardize.")
        strata = X[..., self._stratification_idx].long()
        # Equality with declared tasks handles out-of-range and negative values
        # without indexing an unrelated table row.
        tasks = torch.tensor(self._task_values, device=X.device, dtype=torch.long)
        matches = strata.unsqueeze(-1) == tasks
        known = matches.any(-1, keepdim=True)
        indices = matches.long().argmax(-1, keepdim=True)
        if not bool(known.all()):
            warnings.warn(
                "Predictions are being made for tasks that were not observed during training. "
                "These tasks will use an identity transform (mean=0, stdv=1).",
                stacklevel=3,
            )
        batch = torch.broadcast_shapes(self.means.shape[:-2], indices.shape[:-2])
        index = indices.expand(*batch, *indices.shape[-2:])
        shape = (*batch, *self.means.shape[-2:])
        means = self.means.expand(shape).gather(-2, index)
        stdvs = self.stdvs.expand(shape).gather(-2, index)
        squared = self._stdvs_sq.expand(shape).gather(-2, index)
        return (
            torch.where(known, means, 0.0),
            torch.where(known, stdvs, 1.0),
            torch.where(known, squared, 1.0),
        )

    def forward(
        self, Y: Tensor, Yvar: Tensor | None = None, X: Tensor | None = None
    ) -> tuple[Tensor, Tensor | None]:
        self._validate(Y, Yvar, X)
        if self.training:
            self.fit(Y, X)
        means, stdvs, squared = self.statistics(X)
        return apply_standardize(Y, Yvar, means, stdvs, squared)

    def untransform(
        self, Y: Tensor, Yvar: Tensor | None = None, X: Tensor | None = None
    ) -> tuple[Tensor, Tensor | None]:
        self._validate(Y, Yvar, X)
        if not bool(self._is_trained):
            raise RuntimeError("Standardize must be fitted before untransform.")
        means, stdvs, squared = self.statistics(X)
        return apply_standardize(Y, Yvar, means, stdvs, squared, inverse=True)

    def untransform_posterior(self, posterior: Any, X: Tensor | None = None) -> Any:
        from botorch_rs.posteriors._torch_backend import TorchGPyTorchPosterior
        from botorch_rs.posteriors.transformed import TransformedPosterior

        if not bool(self._is_trained):
            raise RuntimeError("Standardize must be fitted before untransform_posterior.")
        if posterior.mean.shape[-1] != self._m and not self._requires_X:
            raise RuntimeError("Incompatible output dimensions encountered.")
        means, stdvs, squared = (value.clone() for value in self.statistics(X))
        q, m = posterior.mean.shape[-2:]
        if self._requires_X and means.shape[-2] != q:
            if means.shape[-2] != q * m:
                raise RuntimeError("Stratified input context does not match the posterior event.")
            means, stdvs, squared = (
                value.reshape(*value.shape[:-2], m, q).transpose(-1, -2)
                for value in (means, stdvs, squared)
            )
        if isinstance(posterior, TorchGPyTorchPosterior):
            mean = posterior.mean * stdvs + means
            scale = stdvs.expand_as(mean).transpose(-1, -2).flatten(-2)
            covariance = posterior._covariance_tensor * scale.unsqueeze(-1) * scale.unsqueeze(-2)
            cls = TorchGPyTorchPosterior
            if getattr(posterior, "_is_gaussian_mixture", False):
                from botorch_rs.models._torch_backend_fully_bayesian import (
                    TorchGaussianMixturePosterior,
                )

                cls = TorchGaussianMixturePosterior
            return cls(mean, covariance)
        return TransformedPosterior(
            posterior,
            sample_transform=lambda samples: samples * stdvs + means,
            mean_transform=lambda mean, variance: mean * stdvs + means,
            variance_transform=lambda mean, variance: variance * squared,
        )


def untransform_multitask_outcome(transform: Any, posterior: Any, X: Tensor) -> Any:
    """Apply long-format statistics to a task-expanded, output-major law."""
    from botorch_rs.models._torch_backend_modules import TorchOutcomeChain

    if isinstance(transform, TorchOutcomeChain):
        for name in reversed(transform._transform_names):
            posterior = untransform_multitask_outcome(transform[name], posterior, X)
        return posterior
    if not getattr(transform, "_requires_X", False) and posterior.mean.shape[-1] > 1:
        transform = transform.subset_output([0] * posterior.mean.shape[-1])
    return transform.untransform_posterior(posterior, X=X)

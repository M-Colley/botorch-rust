"""Tensor numerical paths for the public empirical GP utilities."""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn

from botorch_rs.exceptions.errors import UnsupportedError
from botorch_rs.models.empirical_gps.utils import LinearInterpolation1D, UniqueInputs


def orthogonal_basis(A: Any, method: str) -> Any:
    if A.ndim < 2 or not A.shape[-2] or not A.shape[-1]:
        raise ValueError("An empirical basis requires nonempty matrix dimensions.")
    if method == "svd":
        _, singular_values, right = torch.linalg.svd(A, full_matrices=False)
        return singular_values.unsqueeze(-1) * right
    if method == "eigh":
        if A.shape[-2] < A.shape[-1]:
            # The row Gram avoids a repeated artificial zero eigenspace of
            # the larger column Gram. U.T @ A has the same scaled right
            # basis and preserves A.T @ A, including its complete graph.
            _, left = torch.linalg.eigh(A @ A.mT)
            return left.flip(-1).mT @ A
        eigenvalues, vectors = torch.linalg.eigh(A.mT @ A)
        eigenvalues, vectors = eigenvalues.flip(-1).clamp_min(0), vectors.flip(-1)
        return (eigenvalues.sqrt().unsqueeze(-1) * vectors.mT)[..., : min(A.shape[-2:]), :]
    raise ValueError(f"Unknown method: {method!r}. Must be 'svd' or 'eigh'.")


def covariance(U1: Any, U2: Any, num_curves: int, diag: bool, correction: int) -> Any:
    U2 = U1 if U2 is None else U2
    if U1.ndim < 2 or U2.ndim < 2 or U1.shape[-2] != U2.shape[-2]:
        raise ValueError("Basis matrices need the same curve dimension.")
    if (
        not isinstance(num_curves, int)
        or not isinstance(correction, int)
        or num_curves <= correction
    ):
        raise ValueError("num_curves must be greater than correction.")
    if U1.dtype != U2.dtype or U1.device != U2.device:
        raise ValueError("Basis matrices must share dtype and device.")
    return ((U1 * U2).sum(-2) if diag else U1.mT @ U2) / (num_curves - correction)


def interpolate(x: Any, y: Any, query: Any, bounds_error: Any, fill_value: Any) -> Any:
    if not all(isinstance(value, torch.Tensor) for value in (x, y, query)):
        raise TypeError("Tensor interpolation requires Tensor knots, values and queries.")
    if x.ndim != 1 or x.numel() < 2 or not y.ndim or y.shape[-1] != x.numel():
        raise ValueError(
            "Interpolation requires at least two knots and a matching final value dimension."
        )
    if (
        x.dtype != y.dtype
        or x.dtype != query.dtype
        or x.device != y.device
        or x.device != query.device
    ):
        raise ValueError("Interpolation tensors must share dtype and device.")
    if not bool(torch.isfinite(x).all()) or not bool((x.diff() > 0).all()):
        raise ValueError("Interpolation knots must be finite and strictly increasing.")
    outside = (query < x[0]) | (query > x[-1])
    if bounds_error is not False and bool(outside.any()):
        raise ValueError("A query is outside the interpolation range.")
    indices = (torch.searchsorted(x.contiguous(), query.contiguous()) - 1).clamp(0, x.numel() - 2)
    # Outer query batching, not pairwise broadcasting with value batches.
    weights = (query - x[indices]) / (x[indices + 1] - x[indices])
    result = y[..., indices] + (y[..., indices + 1] - y[..., indices]) * weights
    fill = torch.as_tensor(fill_value, device=result.device, dtype=result.dtype)
    return torch.where(outside, fill, result)


class TensorInterpolation(nn.Module, LinearInterpolation1D):
    """Registered interpolation state with fresh data graphs on every call."""

    _x: Any
    _y: Any
    _source_x: Any
    _source_y: Any
    _bounds_error: Any
    _fill_value: Any
    _public_versions: Any

    def __init__(
        self,
        x: Any,
        y: Any,
        bounds_error: bool = True,
        fill_value: float = math.nan,
        assume_sorted: bool = False,
    ) -> None:
        nn.Module.__init__(self)
        if x.ndim != 1:
            raise UnsupportedError(f"Expected x to be 1-dim, but got {tuple(x.shape)}.")
        if not isinstance(y, torch.Tensor) or not y.ndim or y.shape[-1] != x.numel():
            raise ValueError("Expected y with a final dimension matching the number of knots.")
        self._assume_sorted = bool(assume_sorted)
        self.register_buffer("_source_x", x, persistent=False)
        self.register_buffer("_source_y", y, persistent=False)
        knots, values = self._sorted_sources()
        interpolate(knots, values, x.new_empty(0), False, fill_value)
        self.register_buffer("_x", knots.detach().clone())
        self.register_buffer("_y", values.detach().clone())
        # Capture the source's default logical fill dtype (which may differ
        # from the data dtype), without transferring numerical data to CPU.
        self.register_buffer("_bounds_error", torch.tensor(bounds_error, dtype=torch.bool))
        self.register_buffer("_fill_value", torch.tensor(fill_value))
        self._record_versions()

    def _sorted_sources(self) -> tuple[Any, Any]:
        if self._assume_sorted:
            return self._source_x, self._source_y
        order = self._source_x.argsort()
        return self._source_x[order], self._source_y[..., order]

    def _record_versions(self) -> None:
        try:
            self._public_versions = tuple(
                (id(value), value._version) for value in (self._x, self._y)
            )
        except RuntimeError:
            self._public_versions = None

    def _adopt_public_changes(self) -> None:
        try:
            versions = tuple((id(value), value._version) for value in (self._x, self._y))
        except RuntimeError:
            versions = None
        if versions != self._public_versions:
            # Explicit public-buffer mutation or state loading replaces the
            # underlying interpolation data, as an ordinary Module would.
            self._source_x, self._source_y = self._x, self._y
            self._assume_sorted = True

    def forward(self, x_new: Any) -> Any:
        self._adopt_public_changes()
        knots, values = self._sorted_sources()
        result = interpolate(knots, values, x_new, bool(self._bounds_error), self._fill_value)
        self._x, self._y = knots.detach().clone(), values.detach().clone()
        self._record_versions()
        return result

    __call__ = nn.Module.__call__

    def _apply(self, fn: Any, recurse: bool = True) -> Any:
        self._adopt_public_changes()
        result = nn.Module._apply(self, fn, recurse)
        self._record_versions()
        return result

    def _save_to_state_dict(self, *args: Any, **kwargs: Any) -> None:
        self._adopt_public_changes()
        knots, values = self._sorted_sources()
        self._x, self._y = knots.detach().clone(), values.detach().clone()
        self._record_versions()
        nn.Module._save_to_state_dict(self, *args, **kwargs)

    def _load_from_state_dict(self, *args: Any, **kwargs: Any) -> None:
        nn.Module._load_from_state_dict(self, *args, **kwargs)
        self._source_x, self._source_y = self._x, self._y
        self._assume_sorted = True
        self._record_versions()

    def state_dict(self, *args: Any, **kwargs: Any) -> Any:
        return nn.Module.state_dict(self, *args, **kwargs)

    def to(self, *args: Any, **kwargs: Any) -> Any:
        return nn.Module.to(self, *args, **kwargs)


def unique_inputs(datasets: Any, X_forward: Any) -> UniqueInputs:
    arrays = [data.X for data in datasets] + ([] if X_forward is None else [X_forward])
    if not arrays:
        raise ValueError("Cannot build unique inputs from empty datasets and no forward inputs.")
    like = arrays[0]
    if any(
        not isinstance(value, torch.Tensor)
        or value.ndim != 2
        or value.shape[-1] != like.shape[-1]
        or value.device != like.device
        or value.dtype != like.dtype
        for value in arrays
    ):
        raise ValueError(
            "Unique inputs require matrices with a common feature dimension, dtype and device."
        )
    grid, inverse = torch.unique(
        torch.cat([value.detach() for value in arrays]), dim=0, sorted=True, return_inverse=True
    )
    labels = list(inverse.split([value.shape[0] for value in arrays]))
    return UniqueInputs(
        grid,
        labels[: len(datasets)],
        labels[-1]
        if X_forward is not None
        else torch.empty(0, device=like.device, dtype=torch.long),
    )


def shrinkage(cov: Any, target: Any, alpha: Any) -> Any:
    if cov.ndim != 2 or target.shape != cov.shape or cov.shape[-1] != cov.shape[-2]:
        raise ValueError("Shrinkage requires matching square covariance and target matrices.")
    if cov.dtype != target.dtype or cov.device != target.device:
        raise ValueError("Shrinkage matrices must share dtype and device.")
    value = torch.as_tensor(alpha, device=cov.device, dtype=cov.dtype)
    if (
        value.numel() != 1
        or not bool(torch.isfinite(value).all())
        or bool(((value < 0) | (value > 1)).any())
    ):
        raise ValueError("Shrinkage alpha must be a finite scalar in [0, 1].")
    return (1 - value) * cov + value * (cov.trace() / target.trace().clamp_min(1e-12)) * target


def basis_curves(
    mu: Any, Sigma: Any, num_modes: Any, correction: int, eigenvalue_tol: float
) -> Any:
    if mu.ndim != 1 or Sigma.ndim != 2 or Sigma.shape != (mu.numel(), mu.numel()):
        raise ValueError("Expected mu of shape (n,) and Sigma of shape (n, n).")
    if num_modes is not None and (not isinstance(num_modes, int) or num_modes < 1):
        raise ValueError("num_modes must be >= 1 if specified.")
    eigenvalues, vectors = torch.linalg.eigh((Sigma + Sigma.mT) * 0.5)
    eigenvalues, vectors = eigenvalues.flip(-1), vectors.flip(-1)
    if num_modes is not None:
        eigenvalues, vectors = eigenvalues[:num_modes], vectors[:, :num_modes]
    keep = eigenvalues > eigenvalue_tol
    eigenvalues, vectors = eigenvalues[keep], vectors[:, keep]
    rank = eigenvalues.numel()
    if rank == 0:
        raise ValueError("The prior is degenerate and cannot be encoded as empirical basis curves.")
    if correction >= rank + 1:
        raise ValueError("correction must be less than the number of synthesized curves.")
    ones = Sigma.new_ones(rank + 1, 1)
    simplex = torch.linalg.qr(ones, mode="complete").Q[:, 1:].mT
    deviations = (vectors * eigenvalues.clamp_min(0).sqrt()) @ (
        math.sqrt(rank + 1 - correction) * simplex
    )
    return (mu.to(Sigma).unsqueeze(-1) + deviations).mT.unsqueeze(-1)

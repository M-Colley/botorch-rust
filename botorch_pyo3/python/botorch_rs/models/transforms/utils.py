"""Helpers of the transforms (``botorch.models.transforms.utils``).

Tensor arguments retain their device and autograd graph. NumPy arrays and
nested lists use the native Rust calculations without importing Torch.
"""

from __future__ import annotations

import sys
from collections.abc import Callable
from functools import wraps
from typing import Any

from botorch_rs import _botorch_rs as _rs
from botorch_rs._arrays import (
    broadcast_shapes,
    broadcast_to,
    first_kind,
    from_flat,
    from_pair,
    is_torch_tensor,
    to_flat,
)


def _tensor_inputs(*values: Any) -> tuple[Any, ...] | None:
    like = next((value for value in values if is_torch_tensor(value)), None)
    if like is None:
        return None
    torch = sys.modules["torch"]
    return tuple(
        value
        if is_torch_tensor(value)
        else torch.as_tensor(value, dtype=like.dtype, device=like.device)
        for value in values
    )


def _same_shape(a: Any, b: Any, name: str) -> tuple[Any, Any]:
    """``a`` and ``b`` broadcast to a common shape, as ``(data, shape)``."""
    da, sa = to_flat(a, "mu")
    db, sb = to_flat(b, name)
    shape = broadcast_shapes(tuple(sa), tuple(sb))
    if shape is None:
        raise RuntimeError(f"The size of tensor a ({sa}) must match the size of tensor b ({sb}).")
    return (broadcast_to(list(da), tuple(sa), shape), shape), (
        broadcast_to(list(db), tuple(sb), shape),
        shape,
    )


def norm_to_lognorm_mean(mu: Any, var: Any) -> Any:
    """The mean of a log-MVN from its MVN marginals: ``exp(mu + var / 2)``."""
    tensors = _tensor_inputs(mu, var)
    if tensors is not None:
        mu, var = tensors
        return (mu + 0.5 * var).exp()
    a, b = _same_shape(mu, var, "var")
    return from_pair(_rs.norm_to_lognorm_mean_py(a, b), first_kind(mu, var))


def norm_to_lognorm_variance(mu: Any, var: Any) -> Any:
    """The variance of a log-MVN from its MVN marginals:
    ``expm1(var) * exp(2 * (mu + var / 2))``."""
    tensors = _tensor_inputs(mu, var)
    if tensors is not None:
        mu, var = tensors
        return var.expm1() * (2 * (mu + 0.5 * var)).exp()
    a, b = _same_shape(mu, var, "var")
    return from_pair(_rs.norm_to_lognorm_variance_py(a, b), first_kind(mu, var))


def norm_to_lognorm(mu: Any, Cov: Any) -> tuple[Any, Any]:
    """The mean (``batch_shape x n``) and covariance (``batch_shape x n x n``)
    of a log-MVN from those of its MVN."""
    tensors = _tensor_inputs(mu, Cov)
    if tensors is not None:
        mu, Cov = tensors
        centered = mu + 0.5 * Cov.diagonal(dim1=-2, dim2=-1)
        return centered.exp(), Cov.expm1() * (centered.unsqueeze(-1) + centered.unsqueeze(-2)).exp()
    kind = first_kind(mu, Cov)
    m, c = _rs.lognormal_moments_py(to_flat(mu, "mu"), to_flat(Cov, "Cov"), False)
    return from_pair(m, kind), from_pair(c, kind)


def lognorm_to_norm(mu: Any, Cov: Any) -> tuple[Any, Any]:
    """The mean (``batch_shape x n``) and covariance (``batch_shape x n x n``)
    of the MVN of a log-MVN with mean ``mu`` and covariance ``Cov``."""
    tensors = _tensor_inputs(mu, Cov)
    if tensors is not None:
        mu, Cov = tensors
        covariance = (Cov / (mu.unsqueeze(-1) * mu.unsqueeze(-2))).log1p()
        return mu.log() - 0.5 * covariance.diagonal(dim1=-2, dim2=-1), covariance
    kind = first_kind(mu, Cov)
    m, c = _rs.lognormal_moments_py(to_flat(mu, "mu"), to_flat(Cov, "Cov"), True)
    return from_pair(m, kind), from_pair(c, kind)


def expand_and_copy_tensor(X: Any, batch_shape: Any) -> Any:
    r"""Expand and copy ``X`` (``input_batch_shape x n x d``) to the batch
    shape broadcast from ``input_batch_shape`` and ``batch_shape``."""
    if is_torch_tensor(X):
        torch = sys.modules["torch"]
        try:
            out = torch.broadcast_shapes(X.shape[:-2], batch_shape)
        except RuntimeError:
            raise RuntimeError(
                f"Provided batch shape ({batch_shape}) and input batch shape ({X.shape[:-2]}) "
                "are not broadcastable."
            ) from None
        return X.expand(out + X.shape[-2:]).clone()
    data, shape = to_flat(X, "X")
    bshape = tuple(int(b) for b in batch_shape)
    out = broadcast_shapes(tuple(shape[:-2]), bshape)
    if out is None:
        raise RuntimeError(
            f"Provided batch shape ({bshape}) and input batch shape ({tuple(shape[:-2])}) are "
            "not broadcastable."
        )
    target = (*out, *shape[-2:])
    return from_flat(broadcast_to(list(data), tuple(shape), target), target, first_kind(X))


def subset_transform(transform: Callable[..., Any]) -> Callable[..., Any]:
    r"""Decorator of an input transform function to separate out indexing
    logic: the function is applied to the columns ``self.indices`` only (all
    columns if the transform has no ``indices``)."""

    @wraps(transform)
    def f(self: Any, X: Any, **kwargs: Any) -> Any:
        indices = getattr(self, "indices", None)
        if indices is None:
            return transform(self, X, **kwargs)
        if is_torch_tensor(X):
            Y = (
                expand_and_copy_tensor(X, self.batch_shape)
                if hasattr(self, "batch_shape")
                else X.clone()
            )
            # Advanced indexing retains source ordering, negative columns,
            # boolean masks and the transformation's broadcast semantics.
            Y[..., indices] = transform(self, X[..., indices], **kwargs)
            return Y
        idx = [
            int(i)
            for i in (to_flat(indices, "indices")[0] if not isinstance(indices, list) else indices)
        ]
        kind = first_kind(X)
        has_shape = hasattr(self, "batch_shape")
        Y = expand_and_copy_tensor(X, self.batch_shape) if has_shape else X
        ydata, yshape = to_flat(Y, "X")
        xdata, xshape = to_flat(X, "X")
        d = xshape[-1]
        idx = [i + d if i < 0 else i for i in idx]
        rows = len(xdata) // d if d else 0
        sub = [xdata[r * d + i] for r in range(rows) for i in idx]
        res = transform(self, from_flat(sub, (*xshape[:-1], len(idx)), kind), **kwargs)
        rdata, rshape = to_flat(res, "result")
        values = list(ydata)
        k = len(idx)
        yrows = len(values) // d if d else 0
        rrows = len(rdata) // k if k else 0
        for r in range(yrows):
            src = r % rrows if rrows else 0
            for c, i in enumerate(idx):
                values[r * d + i] = rdata[src * k + c]
        return from_flat(values, yshape, kind)

    return f


def interaction_features(X: Any) -> Any:
    """The first-order interaction features ``x_i * x_j`` (``i < j``) of the
    inputs ``X`` (``batch_shape x q x d``): ``batch_shape x q x 1 x d(d-1)/2``."""
    if is_torch_tensor(X):
        torch = sys.modules["torch"]
        row, column = torch.triu_indices(X.shape[-1], X.shape[-1], offset=1, device=X.device)
        return (X[..., row] * X[..., column]).unsqueeze(-2)
    return from_pair(_rs.interaction_features_py(to_flat(X, "X")), first_kind(X))


def nanstd(X: Any, dim: int, keepdim: bool = False) -> Any:
    """The standard deviation of ``X`` along ``dim``, ignoring NaNs (with
    Bessel's correction)."""
    if is_torch_tensor(X):
        torch = sys.modules["torch"]
        valid = ~torch.isnan(X)
        safe = torch.where(valid, X, 0)
        count = valid.sum(dim=dim, keepdim=True)
        mean = safe.sum(dim=dim, keepdim=True) / count.clamp_min(1)
        centered = torch.where(valid, safe - mean, 0)
        variance = centered.square().sum(dim=dim, keepdim=True) / (count - 1).clamp_min(1)
        # The constant slice selects the zero subgradient of the standard
        # deviation norm. Do not differentiate sqrt at zero or arithmetic NaNs.
        positive = variance > 0
        std = torch.where(positive, torch.where(positive, variance, 1).sqrt(), 0)
        result = torch.where(count > 1, std, std.new_full((), float("nan")))
        return result if keepdim else result.squeeze(dim)
    data, shape = to_flat(X, "X")
    nd = len(shape)
    if not -nd <= dim < nd:
        raise IndexError(
            f"Dimension out of range (expected to be in range of [{-nd}, {nd - 1}], but got {dim})"
        )
    return from_pair(_rs.nanstd_py((data, shape), dim % nd, bool(keepdim)), first_kind(X))


def _concentrations(c: Any, name: str) -> list[float]:
    return [float(v) for v in to_flat(c, name)[0]]


def kumaraswamy_warp(X: Any, c0: Any, c1: Any, eps: float = 1e-8) -> Any:
    """Warp inputs through a Kumaraswamy CDF: ``X`` (``batch_shape x n x d``,
    in the unit cube) is mapped to ``[eps, 1 - eps]`` and through
    ``1 - (1 - x^c1)^c0`` (``c0``, ``c1``: one concentration per dimension)."""
    tensors = _tensor_inputs(X, c0, c1)
    if tensors is not None:
        X, c0, c1 = tensors
        X = (X * (1 - 2 * eps) + eps).clamp(eps, 1 - eps)
        return 1 - (1 - X.pow(c1)).pow(c0)
    return from_pair(
        _rs.kumaraswamy_warp_py(
            to_flat(X, "X"), _concentrations(c0, "c0"), _concentrations(c1, "c1"), float(eps), False
        ),
        first_kind(X),
    )


def inv_kumaraswamy_warp(X: Any, c0: Any, c1: Any, eps: float = 1e-8) -> Any:
    """Map warped inputs through the inverse Kumaraswamy CDF and back from
    ``[eps, 1 - eps]`` to the unit cube (clamped to ``[0, 1]``)."""
    tensors = _tensor_inputs(X, c0, c1)
    if tensors is not None:
        X, c0, c1 = tensors
        untransformed = (1 - (1 - X).pow(1 / c0)).pow(1 / c1)
        return ((untransformed - eps) / (1 - 2 * eps)).clamp(0, 1)
    return from_pair(
        _rs.kumaraswamy_warp_py(
            to_flat(X, "X"), _concentrations(c0, "c0"), _concentrations(c1, "c1"), float(eps), True
        ),
        first_kind(X),
    )


__all__ = [
    "expand_and_copy_tensor",
    "interaction_features",
    "inv_kumaraswamy_warp",
    "kumaraswamy_warp",
    "lognorm_to_norm",
    "nanstd",
    "norm_to_lognorm",
    "norm_to_lognorm_mean",
    "norm_to_lognorm_variance",
    "subset_transform",
]

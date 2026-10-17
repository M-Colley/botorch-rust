"""Evaluation of the Python callables of the input transforms (imported by
the extension): the feature callables ``f`` of ``AppendFeatures`` and the
``encoders`` of ``NumericToCategoricalEncoding``.

As in :mod:`botorch_rs._robust_bridge`, the inputs arrive as flat float64
buffers (row-major), the callables are evaluated on torch tensors and
gradients come from ``torch.autograd``. A feature callable maps ``n x k``
points to ``n x n_f x d_f`` features (it must act pointwise, as BoTorch's
``batch_shape x q x d`` to ``batch_shape x q x n_f x d_f`` callables do); an
encoder maps ``n`` integer (``torch.long``) category values to ``n x c``
encodings.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from botorch_rs._arrays import ArrayKind, FloatData, from_flat, to_flat
from botorch_rs._torch_bridge import _torch


def _float64() -> ArrayKind:
    return ArrayKind("torch", _torch().float64)


def _check_features(out: Any, n: int) -> None:
    if out.dim() != 3 or out.shape[0] != n:
        raise ValueError(
            "The feature callable `f` must return a `batch_shape x q x n_f x d_f`-dim tensor; "
            f"got shape {tuple(out.shape)} for {n} points."
        )


def features(
    fn: Any, x: Sequence[float], n: int, k: int, fkwargs: Any
) -> tuple[FloatData, int, int]:
    """The features ``fn(X, **fkwargs)`` of the points ``X`` (``n x k``),
    flattened from ``n x n_f x d_f``, with ``n_f`` and ``d_f``."""
    torch = _torch()
    X = from_flat(x, (n, k), _float64())
    with torch.no_grad():
        out = torch.as_tensor(fn(X, **(fkwargs or {})), dtype=torch.float64)
    _check_features(out, n)
    return to_flat(out, "features")[0], int(out.shape[1]), int(out.shape[2])


def features_vjp(
    fn: Any,
    x: Sequence[float],
    n: int,
    k: int,
    adjoint: Sequence[float],
    fkwargs: Any,
) -> FloatData:
    """The vector-Jacobian product of ``fn`` at ``X`` (``n x k``) with the
    adjoint of the features (``n x n_f x d_f``, flattened)."""
    torch = _torch()
    X = from_flat(x, (n, k), _float64()).requires_grad_(True)
    with torch.enable_grad():
        out = torch.as_tensor(fn(X, **(fkwargs or {})), dtype=torch.float64)
    _check_features(out, n)
    if not out.requires_grad:
        return to_flat(torch.zeros_like(X))[0]
    g = from_flat(adjoint, tuple(out.shape), _float64())
    (grad,) = torch.autograd.grad(out, X, grad_outputs=g, allow_unused=True)
    return to_flat(torch.zeros_like(X) if grad is None else grad)[0]


def encode(fn: Any, values: Sequence[int]) -> tuple[FloatData, int, int]:
    """The encodings ``fn(values)`` of integer category values (a
    ``torch.long`` tensor), flattened, with their shape."""
    torch = _torch()
    v = torch.tensor(list(values), dtype=torch.long)
    out = torch.as_tensor(fn(v)).to(torch.float64)
    if out.dim() == 1:
        out = out.unsqueeze(-1)
    if out.dim() != 2 or out.shape[0] != len(values):
        raise ValueError(
            f"An encoder must return one encoding per value; got shape {tuple(out.shape)} for "
            f"{len(values)} values."
        )
    return to_flat(out, "encodings")[0], int(out.shape[0]), int(out.shape[1])


__all__ = ["encode", "features", "features_vjp"]

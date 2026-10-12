"""Evaluation of the Python callables of the robust and multi-fidelity
multi-objective ports (imported by the extension): perturbation functions of
``InputPerturbation`` (BoTorch's callable ``perturbation_set``) and the
``cost_call`` of ``MOMF``.

As in :mod:`botorch_rs._torch_bridge`, the inputs arrive as flat float64
buffers (row-major), the callables are evaluated on float64 torch tensors and
gradients come from ``torch.autograd``. The callables act pointwise: a
perturbation function maps ``n x k`` points to ``n x n_w x k``
perturbations, a cost callable maps ``1 x n x d`` points to ``n`` costs
(``1 x n x 1``, as BoTorch's ``GenericDeterministicModel(cost_call)``).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from botorch_rs._arrays import ArrayKind, FloatData, from_flat, to_flat
from botorch_rs._torch_bridge import _torch


def _float64() -> ArrayKind:
    return ArrayKind("torch", _torch().float64)


def perturbations(fn: Any, x: Sequence[float], n: int, k: int) -> FloatData:
    """The perturbations ``fn(X)`` of the points ``X`` (``n x k``), flattened
    from ``n x n_w x k``."""
    torch = _torch()
    X = from_flat(x, (n, k), _float64())
    with torch.no_grad():
        out = torch.as_tensor(fn(X), dtype=torch.float64)
    _check_perturbations(out, n, k)
    return to_flat(out, "perturbations")[0]


def _check_perturbations(out: Any, n: int, k: int) -> None:
    if out.dim() != 3 or out.shape[0] != n or out.shape[-1] != k:
        raise ValueError(
            f"The perturbation function must return a `batch x n_p x d`-dim tensor of "
            f"perturbations for inputs of shape `batch x d`; got shape {tuple(out.shape)} "
            f"for inputs of shape {(n, k)}."
        )


def perturbation_vjp(
    fn: Any, x: Sequence[float], n: int, k: int, adjoint: Sequence[float]
) -> FloatData:
    """The vector-Jacobian product of ``fn`` at ``X`` (``n x k``) with the
    adjoint of the perturbations (``n x n_w x k``, flattened)."""
    torch = _torch()
    X = from_flat(x, (n, k), _float64()).requires_grad_(True)
    with torch.enable_grad():
        out = torch.as_tensor(fn(X), dtype=torch.float64)
    _check_perturbations(out, n, k)
    g = from_flat(adjoint, tuple(out.shape), _float64())
    if not out.requires_grad:
        return to_flat(torch.zeros_like(X))[0]
    (grad,) = torch.autograd.grad(out, X, grad_outputs=g, allow_unused=True)
    return to_flat(torch.zeros_like(X) if grad is None else grad)[0]


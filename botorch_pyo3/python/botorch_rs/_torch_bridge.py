"""Evaluation of Python callables for the Rust core (imported by the extension).

BoTorch's generic MC objectives, outcome constraints and nonlinear parameter
constraints are callables on torch tensors. The Rust core calls them through
this module: the samples (or inputs) arrive as flat lists, are turned into
float64 tensors that require gradients, the callable is evaluated, and the
autograd graph is kept in an opaque handle. For gradients, the core passes the
adjoint of the outputs back and the vector-Jacobian product is computed with
``torch.autograd.grad``.

Layouts (row-major): posterior samples ``S x 1 x n x m`` (BoTorch's
``sample_shape x batch_shape x q x m`` for one candidate set), inputs
``1 x n x d``, objective and constraint values ``S x 1 x n``.
"""

from __future__ import annotations

from typing import Any

from botorch_rs.exceptions.errors import UnsupportedError


def _torch() -> Any:
    try:
        import torch
    except ImportError as e:  # pragma: no cover - torch is installed with callables
        raise UnsupportedError(
            "Callables (GenericMCObjective, ConstrainedMCObjective, outcome constraints and "
            "nonlinear inequality constraints) operate on torch tensors and require torch."
        ) from e
    return torch


def require_torch() -> None:
    """Raises ``UnsupportedError`` if torch is not installed."""
    _torch()


def forward(
    fn: Any,
    kind: str,
    samples: list[float],
    shape: tuple[int, int, int],
    x: tuple[list[float], int] | None,
    with_grad: bool,
) -> tuple[list[float], Any]:
    """Evaluates an objective (``fn(samples, X=X)``) or a constraint
    (``fn(samples)``) on posterior samples; returns the values (``S x n``,
    row-major) and a handle for :func:`backward`."""
    torch = _torch()
    s, n, m = shape
    Y = torch.tensor(samples, dtype=torch.float64).reshape(s, 1, n, m)
    X = None
    if x is not None:
        data, d = x
        X = torch.tensor(data, dtype=torch.float64).reshape(1, n, d)
    if with_grad:
        Y.requires_grad_(True)
        if X is not None:
            X.requires_grad_(True)
    with torch.enable_grad() if with_grad else torch.no_grad():
        out = fn(Y, X=X) if kind == "objective" else fn(Y)
        out = torch.as_tensor(out, dtype=torch.float64)
    if out.numel() != s * n:
        raise RuntimeError(
            f"The {kind} returned values of shape {tuple(out.shape)}, expected "
            f"{(s, 1, n)} (`sample_shape x batch_shape x q`)."
        )
    values = out.detach().reshape(-1).tolist()
    handle = (out, Y, X) if with_grad else None
    return values, handle


def backward(handle: Any, adjoint: list[float]) -> tuple[list[float], list[float] | None]:
    """The vector-Jacobian product of :func:`forward`'s values with the
    adjoint (``S x n``): the adjoints of the samples (``S x 1 x n x m``) and,
    if the objective received ``X``, of the inputs (``1 x n x d``)."""
    torch = _torch()
    out, Y, X = handle
    g = torch.tensor(adjoint, dtype=torch.float64).reshape(out.shape)
    inputs = [Y] if X is None else [Y, X]
    if not out.requires_grad:
        grads: list[Any] = [None] * len(inputs)
    else:
        grads = list(torch.autograd.grad(out, inputs, grad_outputs=g, allow_unused=True))
    gy = grads[0] if grads[0] is not None else torch.zeros_like(Y)
    gx = None
    if X is not None:
        gx = (grads[1] if grads[1] is not None else torch.zeros_like(X)).reshape(-1).tolist()
    return gy.reshape(-1).tolist(), gx


def nonlinear_constraint(fn: Any, x: list[float], shape: list[int]) -> tuple[float, list[float]]:
    """Value and gradient of a nonlinear inequality constraint ``fn(x) >= 0``
    at a point (``shape = [d]``) or a q-batch (``shape = [q, d]``)."""
    torch = _torch()
    X = torch.tensor(x, dtype=torch.float64).reshape(shape).requires_grad_(True)
    with torch.enable_grad():
        out = torch.as_tensor(fn(X), dtype=torch.float64)
        value = out.sum()
    if out.numel() != 1:
        raise ValueError(
            f"A nonlinear constraint must return a scalar, got a tensor of shape "
            f"{tuple(out.shape)}."
        )
    if value.requires_grad:
        (g,) = torch.autograd.grad(value, X, allow_unused=True)
    else:
        g = None
    grad = torch.zeros_like(X) if g is None else g
    return float(value.detach()), grad.reshape(-1).tolist()

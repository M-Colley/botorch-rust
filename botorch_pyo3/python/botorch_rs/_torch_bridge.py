"""Evaluation of Python callables for the Rust core (imported by the extension).

BoTorch's generic MC objectives, outcome constraints and nonlinear parameter
constraints are callables on torch tensors. The Rust core calls them through
this module: the samples (or inputs) arrive as flat float64 buffers
(``array.array('d')``; lists for a few values), are turned into float64 tensors
that require gradients, the callable is evaluated, and the autograd graph is
kept in an opaque handle. For gradients, the core passes the adjoint of the
outputs back and the vector-Jacobian product is computed with
``torch.autograd.grad``. Values and gradients go back the same way (see
``botorch_rs._arrays.to_flat``).

Layouts (row-major): posterior samples ``S x 1 x n x m`` (BoTorch's
``sample_shape x batch_shape x q x m`` for one candidate set), inputs
``1 x q x d``, objective and constraint values ``S x 1 x n``. With a
one-to-many input transform (``InputPerturbation``) the samples have
``n = q * n_w`` points and an objective that is a risk measure returns
``S x 1 x q`` values.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from botorch_rs._arrays import ArrayKind, FloatData, from_flat, to_flat
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


def _float64() -> ArrayKind:
    return ArrayKind("torch", _torch().float64)


def forward(
    fn: Any,
    kind: str,
    samples: Sequence[float],
    shape: tuple[int, int, int],
    x: tuple[Sequence[float], int] | None,
    with_grad: bool,
) -> tuple[FloatData, int, Any]:
    """Evaluates an objective (``fn(samples, X=X)``) or a constraint
    (``fn(samples)``) on posterior samples; returns the values (``S x n``,
    row-major) and a handle for :func:`backward`."""
    torch = _torch()
    s, n, m = shape
    Y = from_flat(samples, (s, 1, n, m), _float64())
    X = None
    if x is not None:
        data, d = x
        X = from_flat(data, (1, len(data) // d, d), _float64())
    if with_grad:
        Y.requires_grad_(True)
        if X is not None:
            X.requires_grad_(True)
    with torch.enable_grad() if with_grad else torch.no_grad():
        out = fn(Y, X=X) if kind == "objective" else fn(Y)
        out = torch.as_tensor(out, dtype=torch.float64)
    points = int(out.shape[-1]) if out.ndim else 0
    count = out.numel() // points if points else 0
    if not points or points > n or not count or count % s or (kind != "objective" and count != s):
        raise RuntimeError(
            f"The {kind} returned values of shape {tuple(out.shape)}, expected "
            f"{(s, 1, n)} (`sample_shape x batch_shape x q`)."
        )
    values = to_flat(out, kind)[0]
    handle = (out, Y, X) if with_grad else None
    return values, count, handle


def forward_mo(
    fn: Any,
    samples: Sequence[float],
    shape: tuple[int, int, int],
    x: tuple[Sequence[float], int] | None,
    with_grad: bool,
    pass_x: bool = True,
) -> tuple[FloatData, int, int, Any]:
    """Evaluates a multi-output objective ``fn(samples, X=X)`` (``fn(samples)``
    without ``pass_x``, e.g. the preprocessing function of a risk measure) on
    posterior samples; returns the values (``S x n' x m'``, row-major),
    ``n'``, ``m'`` and a handle for :func:`backward`. ``n'`` is the number of
    sample points ``n``, or fewer for a multi-output risk measure."""
    torch = _torch()
    s, n, m = shape
    Y = from_flat(samples, (s, 1, n, m), _float64())
    X = None
    if x is not None:
        data, d = x
        X = from_flat(data, (1, len(data) // d, d), _float64())
    if with_grad:
        Y.requires_grad_(True)
        if X is not None:
            X.requires_grad_(True)
    with torch.enable_grad() if with_grad else torch.no_grad():
        out = torch.as_tensor(fn(Y, X=X) if pass_x else fn(Y), dtype=torch.float64)
    if out.dim() < 2 or out.numel() == 0 or out.numel() % (s * out.shape[-2]) != 0:
        raise RuntimeError(
            f"The multi-output objective returned values of shape {tuple(out.shape)}, "
            f"expected {(s, 1, n)} x m' (`sample_shape x batch_shape x q x m'`)."
        )
    nv = int(out.shape[-2])
    values = to_flat(out, "objective")[0]
    handle = (out, Y, X) if with_grad else None
    return values, nv, out.numel() // (s * nv), handle


def backward(handle: Any, adjoint: Sequence[float]) -> tuple[FloatData, FloatData | None]:
    """The vector-Jacobian product of :func:`forward`'s values with the
    adjoint (``S x n``): the adjoints of the samples (``S x 1 x n x m``) and,
    if the objective received ``X``, of the inputs (``1 x n x d``)."""
    torch = _torch()
    out, Y, X = handle
    g = from_flat(adjoint, tuple(out.shape), _float64())
    inputs = [Y] if X is None else [Y, X]
    if not out.requires_grad:
        grads: list[Any] = [None] * len(inputs)
    else:
        grads = list(torch.autograd.grad(out, inputs, grad_outputs=g, allow_unused=True))
    gy = grads[0] if grads[0] is not None else torch.zeros_like(Y)
    gx = None
    if X is not None:
        gx = to_flat(grads[1] if grads[1] is not None else torch.zeros_like(X))[0]
    return to_flat(gy)[0], gx


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


def call_tensor_function(
    fn: Any, args: list[tuple[Sequence[float], Sequence[int]]], with_grad: bool
) -> tuple[FloatData, list[int], Any]:
    """Evaluates ``fn(*args)`` on float64 tensors built from ``(data, shape)``
    pairs; returns the values (row-major), their shape and a handle for
    :func:`tensor_function_vjp` (with gradients)."""
    torch = _torch()
    inputs = [from_flat(data, tuple(shape), _float64()) for data, shape in args]
    if with_grad:
        for t in inputs:
            t.requires_grad_(True)
    with torch.enable_grad() if with_grad else torch.no_grad():
        out = torch.as_tensor(fn(*inputs), dtype=torch.float64)
    values = to_flat(out, "values")[0]
    handle = (out, inputs) if with_grad else None
    return values, list(out.shape), handle


def tensor_function_vjp(handle: Any, adjoint: Sequence[float]) -> list[FloatData]:
    """The adjoints of the arguments of :func:`call_tensor_function` for the
    adjoint of its values."""
    torch = _torch()
    out, inputs = handle
    g = from_flat(adjoint, tuple(out.shape), _float64())
    if out.requires_grad:
        grads = torch.autograd.grad(out, inputs, grad_outputs=g, allow_unused=True)
    else:
        grads = (None,) * len(inputs)
    return [
        to_flat(gr if gr is not None else torch.zeros_like(t))[0]
        for gr, t in zip(grads, inputs, strict=True)
    ]


def _deterministic_values(fn: Any, X: Any) -> Any:
    """``fn(X)`` of a ``GenericDeterministicModel`` with gradients; an output
    that does not depend on ``X`` through autograd is an error (as in BoTorch,
    whose optimizers differentiate it)."""
    torch = _torch()
    with torch.enable_grad():
        out = torch.as_tensor(fn(X), dtype=torch.float64)
    if not out.requires_grad:
        raise RuntimeError(
            "GenericDeterministicModel: f(X) does not require gradients, so its gradient "
            "with respect to X cannot be computed with torch.autograd (botorch_rs models do "
            "not support autograd through their posteriors; use PosteriorMeanModel for the "
            "posterior mean of a botorch_rs model)."
        )
    return out


def deterministic_model_vjp(
    fn: Any, data: Sequence[float], shape: Sequence[int], adjoint: Sequence[float]
) -> FloatData:
    """The vector-Jacobian product of ``fn`` (a ``GenericDeterministicModel``)
    at ``X`` (``shape``) with the adjoint of its values (row-major)."""
    torch = _torch()
    X = from_flat(data, tuple(shape), _float64()).requires_grad_(True)
    out = _deterministic_values(fn, X)
    g = from_flat(adjoint, tuple(out.shape), _float64())
    (grad,) = torch.autograd.grad(out, X, grad_outputs=g, allow_unused=True)
    return to_flat(torch.zeros_like(X) if grad is None else grad)[0]


def deterministic_model_jacobian(
    fn: Any, data: Sequence[float], shape: Sequence[int]
) -> tuple[FloatData, list[int], FloatData]:
    """``fn(X)`` of a ``GenericDeterministicModel`` with its Jacobian: the
    values (row-major), their shape and the Jacobian (row-major,
    ``values.numel() x X.numel()``)."""
    torch = _torch()
    X = from_flat(data, tuple(shape), _float64()).requires_grad_(True)
    out = _deterministic_values(fn, X)
    rows = []
    # Callers may run in no-grad mode (e.g. torch.autograd.Function.forward).
    with torch.enable_grad():
        flat = out.reshape(-1)
        for k in range(flat.numel()):
            (g,) = torch.autograd.grad(flat[k], X, retain_graph=True, allow_unused=True)
            rows.append(torch.zeros(X.numel(), dtype=torch.float64) if g is None else g.reshape(-1))
    jac = torch.stack(rows) if rows else torch.zeros(0, X.numel(), dtype=torch.float64)
    return to_flat(out.detach())[0], list(out.shape), to_flat(jac)[0]


def acquisition_forward(
    fn: Any, data: Sequence[float], shape: Sequence[int], with_grad: bool
) -> tuple[FloatData, FloatData | None]:
    """Values of an acquisition function given as a callable on torch tensors
    (``fn(X)`` with ``X`` of shape ``b x q x d``, returning ``b`` values) and,
    with ``with_grad``, the gradient of their sum with respect to ``X``
    (row-major, ``b x q x d``)."""
    torch = _torch()
    X = from_flat(data, tuple(shape), _float64())
    if with_grad:
        X.requires_grad_(True)
    b = int(shape[0])
    grads = None
    # Callers may run in no-grad mode (e.g. torch.autograd.Function.forward),
    # so the graph is built and differentiated in one enable_grad block.
    with torch.enable_grad() if with_grad else torch.no_grad():
        out = torch.as_tensor(fn(X), dtype=torch.float64)
        if out.numel() != b:
            raise RuntimeError(
                f"The acquisition function returned values of shape {tuple(out.shape)} for "
                f"candidate sets of shape {tuple(shape)}, expected {b} values (`batch_shape`)."
            )
        if with_grad:
            g = None
            if out.requires_grad:
                (g,) = torch.autograd.grad(out.sum(), X, allow_unused=True)
            grads = to_flat(torch.zeros_like(X) if g is None else g)[0]
    return to_flat(out.detach().reshape(b))[0], grads


def acquisition_x_pending(fn: Any) -> tuple[FloatData, list[int]] | None:
    """The ``X_pending`` attribute (``m x d``) of a callable acquisition
    function, if it has one."""
    pending = getattr(fn, "X_pending", None)
    if pending is None:
        return None
    flat, shape = to_flat(pending, "X_pending")
    return flat, list(shape)


def acquisition_set_x_pending(fn: Any, x: tuple[Sequence[float], Sequence[int]] | None) -> None:
    """Calls ``fn.set_X_pending`` with a float64 tensor (or ``None``)."""
    setter = getattr(fn, "set_X_pending", None)
    if setter is None:
        raise UnsupportedError(
            f"{type(fn).__name__} has no `set_X_pending`; pending points (sequential "
            "optimization, optimize_acqf_list) are not supported."
        )
    setter(None if x is None else from_flat(x[0], tuple(x[1]), _float64()))

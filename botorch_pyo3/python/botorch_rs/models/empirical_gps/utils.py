"""Empirical GP matrix and interpolation helpers implemented in Rust.

Python handles array layout and module metadata. Interpolation, covariance,
PSD projection and shrinkage have native first-order vector-Jacobian products.
Decomposition of historical basis data is presently a fixed-data operation.
"""

from __future__ import annotations

import importlib
import math
import sys
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from botorch_rs import _botorch_rs as _rs
from botorch_rs import _ops
from botorch_rs._arrays import (
    LIST,
    ArrayKind,
    first_kind,
    from_flat,
    is_torch_tensor,
    kind_of,
    to_flat,
)
from botorch_rs.exceptions.errors import UnsupportedError
from botorch_rs.utils.constraints import NonTransformedInterval


def _reshape(x: Any, shape: tuple[int, ...]) -> Any:
    if kind_of(x).lib != "list":
        return x.reshape(shape)
    return from_flat(to_flat(x)[0], shape, LIST)


def _permute(x: Any, order: tuple[int, ...]) -> Any:
    kind = kind_of(x)
    if kind.lib == "torch":
        return x.permute(order)
    if kind.lib == "numpy":
        return x.transpose(order)
    data, shape = to_flat(x)
    return from_flat(_ops._permute_flat(data, shape, order), tuple(shape[i] for i in order), kind)


def _moveaxis(x: Any, source: int, target: int) -> Any:
    n = len(_ops.shape_of(x))
    order = list(range(n))
    axis = order.pop(source % n)
    order.insert(target % n, axis)
    return _permute(x, tuple(order))


def _pair_output(pair: Any, kind: ArrayKind) -> Any:
    return from_flat(pair[0], tuple(pair[1]), kind)


def _reject_decomposition_grad(*values: Any) -> None:
    if _ops._requires_grad(values):
        raise UnsupportedError(
            "Empirical basis decompositions currently require fixed historical covariance data; "
            "decomposition gradients are not yet supported."
        )


_AUTOGRAD: Any = None


def _native_function() -> Any:
    global _AUTOGRAD
    if _AUTOGRAD is not None:
        return _AUTOGRAD
    torch: Any = importlib.import_module("torch")

    class NativeEmpirical(torch.autograd.Function):
        @staticmethod
        def forward(ctx: Any, operation: str, options: tuple[Any, ...], *inputs: Any) -> Any:
            ctx.operation, ctx.options = operation, options
            ctx.save_for_backward(*inputs)
            return _evaluate(operation, options, inputs)

        @staticmethod
        def backward(ctx: Any, gradient: Any) -> Any:
            inputs, options = ctx.saved_tensors, ctx.options
            pairs: list[Any] = [to_flat(v) for v in inputs]
            outputs: Any
            if ctx.operation == "covariance":
                outputs = _rs.empirical_covariance_vjp(
                    pairs[0], pairs[1], to_flat(gradient), *options
                )
            elif ctx.operation == "psd":
                outputs = (_rs.empirical_project_psd_vjp(pairs[0], to_flat(gradient), *options),)
            elif ctx.operation == "shrinkage":
                gc, gt, ga = _rs.empirical_shrinkage_vjp(
                    pairs[0], pairs[1], pairs[2][0][0], to_flat(gradient)
                )
                outputs = gc, gt, ([ga], pairs[2][1])
            else:
                outputs = _rs.empirical_interp_vjp(
                    pairs[0], pairs[1], pairs[2], to_flat(gradient), options[0]
                )
            return (
                None,
                None,
                *(_pair_output(g, kind_of(v)) for g, v in zip(outputs, inputs, strict=True)),
            )

    _AUTOGRAD = NativeEmpirical
    return _AUTOGRAD


def _evaluate(operation: str, options: tuple[Any, ...], inputs: Any) -> Any:
    pairs: list[Any] = [to_flat(v) for v in inputs]
    if operation == "covariance":
        pair = _rs.empirical_covariance(pairs[0], pairs[1], *options)
    elif operation == "psd":
        pair = _rs.empirical_project_psd(pairs[0], *options)
    elif operation == "shrinkage":
        pair = _rs.empirical_shrinkage(pairs[0], pairs[1], pairs[2][0][0])
    else:
        pair = _rs.empirical_interp(pairs[0], pairs[1], pairs[2], *options)
    return _pair_output(pair, first_kind(*inputs))


def _native(operation: str, options: tuple[Any, ...], *inputs: Any) -> Any:
    if _ops._requires_grad(inputs):
        kind = next(kind_of(v) for v in inputs if is_torch_tensor(v))
        return _native_function().apply(
            operation, options, *(_ops._as_array(v, kind) for v in inputs)
        )
    return _evaluate(operation, options, inputs)


def compute_orthogonal_basis(A: Any, method: str = "svd") -> Any:
    """Economy basis B satisfying B.T @ B = A.T @ A, for arbitrary batches."""
    _reject_decomposition_grad(A)
    return _pair_output(_rs.empirical_orthogonal_basis(to_flat(A), method), kind_of(A))


def center_curves(Y: Any, curve_dim: int = -2) -> tuple[Any, Any]:
    shape = _ops.shape_of(Y)
    if not shape or not -len(shape) <= curve_dim < len(shape):
        raise IndexError("curve_dim is outside the historical curve dimensions.")
    axis = curve_dim % len(shape)
    mean = _ops.reduce("mean", Y, dim=axis, keepdim=True)
    return _reshape(mean, shape[:axis] + shape[axis + 1 :]), _ops.apply("sub", Y, mean)


def compute_sample_covariance(
    U1: Any, U2: Any | None, num_curves: int, diag: bool = False, correction: int = 0
) -> Any:
    U2 = U1 if U2 is None else U2
    left, right = _ops.shape_of(U1), _ops.shape_of(U2)
    if len(left) < 2 or len(right) < 2 or left[-2] != right[-2]:
        raise ValueError("Basis matrices need the same curve dimension.")
    from botorch_rs._arrays import broadcast_shapes

    batch = broadcast_shapes(left[:-2], right[:-2])
    if batch is None:
        raise ValueError("Basis batches do not broadcast.")
    U1 = _ops.expand(U1, (*batch, *left[-2:]))
    U2 = _ops.expand(U2, (*batch, *right[-2:]))
    return _native("covariance", (num_curves, correction, diag), U1, U2)


def extract_slice_for_interp(x: Any, num_outputs: int) -> Any:
    shape = _ops.shape_of(x)
    if num_outputs > 1 and len(shape) >= 3 and shape[-3] == num_outputs:
        x = _ops.getitem(x, (..., 0, slice(None), slice(None)))
        shape = _ops.shape_of(x)
    return _reshape(x, shape[:-1]) if shape and shape[-1] == 1 else x


def compute_basis_matrix(
    f: Callable[[Any], Any], x: Any, num_outputs: int, curve_weights: Any = None
) -> Any:
    basis = _moveaxis(f(x), 1, -2)
    if curve_weights is not None:
        weights = _reshape(curve_weights, (*_ops.shape_of(curve_weights), 1))
        basis = _ops.apply("mul", basis, weights)
    return basis


def instantiate_ard(
    obj: Any,
    num_curves: int,
    curve_weights: Any = None,
    dtype: Any = None,
    device: Any = None,
) -> None:
    """Register native module metadata for nonnegative curve weights.

    Objects with register_parameter/register_constraint can implement their
    own registration. Plain native descriptor objects receive attributes.
    Torch Parameter is only used as a registration container when supplied.
    """
    torch: Any = sys.modules.get("torch")
    if num_curves < 1:
        raise ValueError("num_curves must be positive.")
    generated = curve_weights is None
    if generated:
        kind = ArrayKind("torch", dtype or torch.float32, device) if torch is not None else LIST
        curve_weights = from_flat([1.0] * num_curves, (num_curves,), kind)
    if _ops.shape_of(curve_weights) != (num_curves,):
        raise ValueError("curve_weights must have shape (num_curves,).")
    if is_torch_tensor(curve_weights) and not isinstance(curve_weights, torch.nn.Parameter):
        curve_weights = torch.nn.Parameter(
            curve_weights, requires_grad=generated or curve_weights.requires_grad
        )
    constraint = NonTransformedInterval(0.0, math.inf)
    if callable(getattr(obj, "register_parameter", None)):
        obj.register_parameter("curve_weights", curve_weights)
    else:
        obj.curve_weights = curve_weights
    if callable(getattr(obj, "register_constraint", None)):
        obj.register_constraint("curve_weights", constraint)
    else:
        obj.curve_weights_constraint = constraint
    obj.ard = True


def _gather_final(y: Any, indices: list[int]) -> Any:
    kind = kind_of(y)
    if kind.lib == "torch":
        torch = importlib.import_module("torch")
        return y.index_select(-1, torch.tensor(indices, dtype=torch.long, device=y.device))
    if kind.lib == "numpy":
        return y[..., indices]
    values, shape = to_flat(y)
    n = shape[-1]
    data = [values[i * n + j] for i in range(math.prod(shape[:-1])) for j in indices]
    return from_flat(data, (*shape[:-1], len(indices)), kind)


class LinearInterpolation1D:
    """Native piecewise-linear interpolation with outer query batching."""

    def __init__(
        self,
        x: Any,
        y: Any,
        bounds_error: bool = True,
        fill_value: float = math.nan,
        assume_sorted: bool = False,
    ) -> None:
        values, shape = to_flat(x)
        if len(shape) != 1:
            raise UnsupportedError(f"Expected x to be 1-dim, but got {shape}.")
        if not assume_sorted:
            order = sorted(range(len(values)), key=values.__getitem__)
            x, y = _gather_final(x, order), _gather_final(y, order)
        self._x, self._y = x, y
        self._bounds_error, self._fill_value = bool(bounds_error), float(fill_value)
        # Validation is native and does not evaluate any query locations.
        _rs.empirical_interp(to_flat(x), to_flat(y), ([], (0,)), False, math.nan)

    def forward(self, x_new: Any) -> Any:
        return _interp1d_torch(self._x, self._y, x_new, self._bounds_error, self._fill_value)

    __call__ = forward

    def state_dict(self) -> dict[str, Any]:
        return {
            "_x": self._x,
            "_y": self._y,
            "_bounds_error": self._bounds_error,
            "_fill_value": self._fill_value,
        }

    def to(self, *args: Any, **kwargs: Any) -> LinearInterpolation1D:
        if is_torch_tensor(self._x):
            self._x, self._y = self._x.to(*args, **kwargs), self._y.to(*args, **kwargs)
        return self


def _interp1d_torch(
    x: Any, y: Any, x_new: Any, bounds_error: bool | None = None, fill_value: float = math.nan
) -> Any:
    return _native("interp", (bounds_error is not False, fill_value), x, y, x_new)


def _interp1d_raise_out_of_bounds_error(x_new: Any, x: Any) -> None:
    queries, knots = to_flat(x_new)[0], to_flat(x)[0]
    if not queries or not knots:
        return
    if min(queries) < min(knots):
        raise ValueError(
            f"A value ({min(queries)}) in x_new is below the interpolation range's minimum value ({min(knots)})."
        )
    if max(queries) > max(knots):
        raise ValueError(
            f"A value ({max(queries)}) in x_new is above the interpolation range's maximum value ({max(knots)})."
        )


def validate_historical_curves_3d(Y: Any, name: str = "Y_full") -> None:
    ndim = len(_ops.shape_of(Y))
    if ndim != 3:
        raise ValueError(
            f"Expected {name} to be 3-dim (num_curves x num_progression x m), got {ndim}-dim."
        )


def validate_no_transforms(input_transform: Any, outcome_transform: Any, model_name: str) -> None:
    for name, transform in (
        ("input_transform", input_transform),
        ("outcome_transform", outcome_transform),
    ):
        if transform is not None:
            raise UnsupportedError(f"{name} is not yet supported for {model_name}.")


def build_mean_interpolant(X_full: Any, Y_full: Any) -> tuple[LinearInterpolation1D, int, Any]:
    validate_historical_curves_3d(Y_full)
    mean = _ops.reduce("mean", Y_full, dim=0)
    mean = _permute(mean, (1, 0))
    n, m = _ops.shape_of(Y_full)[-2:]
    return LinearInterpolation1D(_reshape(X_full, (n,)), mean), m, mean


def build_basis_interpolant(
    X_full: Any,
    Y_full: Any,
    *,
    ard: bool,
    use_svd: bool | None,
    vectorize_outputs: bool,
    method: str = "svd",
) -> tuple[LinearInterpolation1D, int, bool]:
    validate_historical_curves_3d(Y_full)
    curves, progression, outputs = _ops.shape_of(Y_full)
    _, centered = center_curves(Y_full, curve_dim=-3)
    threshold = progression * outputs if vectorize_outputs else progression
    use_svd = not ard and curves > threshold if use_svd is None else use_svd
    if use_svd and vectorize_outputs:
        basis = compute_orthogonal_basis(
            _reshape(centered, (curves, progression * outputs)), method
        )
        effective = _ops.shape_of(basis)[0]
        basis = _moveaxis(_reshape(basis, (effective, progression, outputs)), -1, 0)
    elif use_svd:
        basis = compute_orthogonal_basis(_moveaxis(centered, -1, 0), method)
        effective = _ops.shape_of(basis)[-2]
    else:
        basis, effective = _moveaxis(centered, -1, 0), curves
    return LinearInterpolation1D(_reshape(X_full, (progression,)), basis), effective, use_svd


@dataclass
class ExperimentDataset:
    X: Any
    Y: Any
    Yvar: Any = None


@dataclass
class UniqueInputs:
    X_all: Any
    experiment_indices: list[Any]
    forward_indices: Any


def build_unique_inputs(datasets: list[ExperimentDataset], X_forward: Any = None) -> UniqueInputs:
    arrays = [d.X for d in datasets] + ([] if X_forward is None else [X_forward])
    if not arrays:
        raise ValueError("Cannot build unique inputs: datasets is empty and X_forward is None.")
    flats = [to_flat(x) for x in arrays]
    width = flats[0][1][-1]
    if any(len(s) != 2 or s[-1] != width for _, s in flats):
        raise ValueError("Unique inputs require unbatched n x d arrays with common d.")
    rows = [[tuple(v[i * width : (i + 1) * width]) for i in range(s[0])] for v, s in flats]
    unique = sorted(set(row for group in rows for row in group))
    index = {row: i for i, row in enumerate(unique)}
    kind = first_kind(*arrays)
    maps = [_ops.int_array([index[row] for row in group], (len(group),), kind) for group in rows]
    forward = maps[-1] if X_forward is not None else _ops.int_array([], (0,), kind)
    return UniqueInputs(
        from_flat([v for row in unique for v in row], (len(unique), width), kind),
        maps[: len(datasets)],
        forward,
    )


def project_psd(A: Any, min_eigval: float = 0.0) -> Any:
    return _native("psd", (float(min_eigval),), A)


def trace_matched_shrinkage(cov: Any, target: Any, alpha: Any) -> Any:
    if len(to_flat(alpha)[0]) != 1:
        raise ValueError("Shrinkage intensity alpha must be scalar.")
    return _native("shrinkage", (), cov, target, alpha)


def em_prior_to_basis_curves(
    mu: Any,
    Sigma: Any,
    num_modes: int | None = None,
    correction: int = 0,
    eigenvalue_tol: float = 0.0,
) -> Any:
    _reject_decomposition_grad(Sigma)
    if num_modes is not None and num_modes < 1:
        raise ValueError(f"num_modes must be >= 1 if specified; got {num_modes}.")
    values, shape = to_flat(mu)
    if len(shape) != 1:
        raise ValueError("Expected mu of shape (n,) and Sigma of shape (n, n).")
    pair = _rs.empirical_em_basis(
        ([0.0] * len(values), shape), to_flat(Sigma), num_modes, correction, eigenvalue_tol
    )
    basis = _pair_output(pair, kind_of(Sigma))
    return _ops.apply("add", basis, _reshape(mu, (1, len(values), 1)))


__all__ = [
    "ExperimentDataset",
    "UniqueInputs",
    "LinearInterpolation1D",
    "build_basis_interpolant",
    "build_mean_interpolant",
    "build_unique_inputs",
    "center_curves",
    "compute_basis_matrix",
    "compute_orthogonal_basis",
    "compute_sample_covariance",
    "em_prior_to_basis_curves",
    "extract_slice_for_interp",
    "instantiate_ard",
    "project_psd",
    "trace_matched_shrinkage",
    "validate_historical_curves_3d",
    "validate_no_transforms",
]

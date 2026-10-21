"""Utilities for maximizing acquisition functions
(``botorch.optim.utils.acquisition_utils``)."""

from __future__ import annotations

import warnings
from collections.abc import Mapping
from typing import Any

from botorch_rs._arrays import from_flat, is_torch_tensor, kind_of, to_flat
from botorch_rs.exceptions.errors import BotorchError, BotorchTensorDimensionError
from botorch_rs.exceptions.warnings import BotorchWarning


def _bound_values(bound: Any, n: int, d: int) -> list[float] | None:
    """A bound (scalar or ``d`` values, broadcast over the rows) as ``n * d``
    values in row-major order."""
    if bound is None:
        return None
    data, shape = to_flat(bound, "bound")
    if len(data) == 1:
        return [float(data[0])] * (n * d)
    if len(data) == d:
        return [float(v) for v in data] * n
    if len(data) == n * d:
        return [float(v) for v in data]
    raise BotorchTensorDimensionError(
        f"The bounds of shape {tuple(shape)} cannot be broadcast to the columns of X."
    )


def columnwise_clamp(
    X: Any,
    lower: Any = None,
    upper: Any = None,
    raise_on_violation: bool = False,
) -> Any:
    r"""Clamp values of an array in column-wise fashion (with support for
    t-batches), like BoTorch's ``columnwise_clamp``.

    Args:
        X: The ``b x n x d`` input array. If 2-dimensional, ``b`` is assumed to
            be 1.
        lower: The column-wise lower bounds. If scalar, apply bound to all
            columns.
        upper: The column-wise upper bounds. If scalar, apply bound to all
            columns.
        raise_on_violation: If ``True``, raise an exception when the elements
            in ``X`` are out of the specified bounds (up to numerical accuracy,
            ``torch.allclose``'s tolerances).

    Returns:
        The clamped array (of the type of ``X``).
    """
    if is_torch_tensor(X):
        from botorch_rs.optim._torch_optim_utils import columnwise_clamp as tensor_clamp

        return tensor_clamp(X, lower, upper, raise_on_violation)
    if lower is None and upper is None:
        return X
    kind = kind_of(X)
    data, shape = to_flat(X, "X")
    d = shape[-1] if shape else 1
    n = len(data) // d if d else 0
    lo = _bound_values(lower, n, d)
    hi = _bound_values(upper, n, d)
    if (
        lo is not None
        and hi is not None
        and any(a > b for a, b in zip(lo[:d], hi[:d], strict=True))
    ):
        raise ValueError("Lower bounds cannot exceed upper bounds.")
    out = []
    for i, v in enumerate(data):
        x = float(v)
        if x == x:  # NaN stays NaN
            if lo is not None:
                x = max(x, lo[i])
            if hi is not None:
                x = min(x, hi[i])
        out.append(x)
    if raise_on_violation:
        close = all(
            abs(float(a) - b) <= 1e-8 + 1e-5 * abs(b) for a, b in zip(data, out, strict=True)
        )
        if not close:
            raise BotorchError(
                f"Original value(s) are out of bounds: out={out}, X={list(data)}, "
                f"lower={lower}, upper={upper}."
            )
    return from_flat(out, shape, kind)


def fix_features(
    X: Any,
    fixed_features: Mapping[int, Any] | None = None,
    replace_current_value: bool = True,
) -> Any:
    r"""Fix feature values in an array (BoTorch's ``fix_features``).

    Args:
        X: input array with shape ``b x q x (reduced_p | p)``, where
            ``reduced_p`` is the number of features not fixed to a constant
            value and ``p`` the full number; use ``p`` if
            ``replace_current_value`` is True, and ``reduced_p`` otherwise.
        fixed_features: A mapping with keys as column indices and values
            equal to what the feature should be set to in ``X``. A value may
            be an array of shape ``b x q`` or ``b`` (the same value across the
            q dimension).
        replace_current_value: If True, replace the specified indexes,
            otherwise the indices are inserted.

    Returns:
        The array ``X`` with fixed features.
    """
    if is_torch_tensor(X):
        from botorch_rs.optim._torch_acq_optim import fix_features as tensor_fix_features

        d = X.shape[-1] + (len(fixed_features or {}) if not replace_current_value else 0)
        if any(not isinstance(index, int) or not -d <= index < d for index in fixed_features or {}):
            raise ValueError("Fixed feature indices must lie within the input dimension.")
        if len({index % d for index in fixed_features or {}}) != len(fixed_features or {}):
            raise ValueError("Fixed feature indices must be unique.")
        return tensor_fix_features(X, fixed_features, insert=not replace_current_value)
    if fixed_features is None:
        return X
    kind = kind_of(X)
    data, shape = to_flat(X, "X")
    p_in = shape[-1]
    rows = len(data) // p_in if p_in else 0
    p = p_in if replace_current_value else p_in + len(fixed_features)
    values: dict[int, list[float]] = {}
    for k, v in fixed_features.items():
        vdata, vshape = to_flat(v, f"fixed_features[{k}]")
        if len(vshape) == 0:
            values[int(k)] = [float(vdata[0])] * rows
            continue
        if len(shape) != 3:
            raise BotorchTensorDimensionError(
                "X must be a 3-dimensional tensor, as value is a tensor."
                f"X.shape = {tuple(shape)}, value.shape = {tuple(vshape)}."
            )
        b, q = shape[0], shape[1]
        if len(vshape) == 1:
            values[int(k)] = [float(vdata[i]) for i in range(b) for _ in range(q)]
        else:
            values[int(k)] = [float(x) for x in vdata]
    out: list[float] = []
    for r in range(rows):
        row = data[r * p_in : (r + 1) * p_in]
        j = 0
        for i in range(p):
            if i in values:
                out.append(values[i][r])
                if replace_current_value:
                    j += 1
            else:
                out.append(float(row[j]))
                j += 1
    return from_flat(out, (*shape[:-1], p), kind)


def get_X_baseline(acq_function: Any) -> Any:
    r"""Extract ``X_baseline`` from an acquisition function (BoTorch's
    ``get_X_baseline``): its ``X_baseline`` attribute if it has non-empty
    baseline points, else the original training inputs of its model (of the
    first sub-model of a ``ModelListGP``), reduced to the first batch.

    Returns:
        An optional ``n x d`` array of baseline points; ``None`` (with a
        ``BotorchWarning``) if no baseline points are found.
    """
    X = None
    try:
        X = acq_function.X_baseline
        if X is None or (X.shape[0] if is_torch_tensor(X) else to_flat(X, "X_baseline")[1][0]) == 0:
            raise BotorchError
    except (BotorchError, AttributeError, IndexError):
        try:
            model = acq_function.model
        except AttributeError:
            warnings.warn("Failed to extract X_baseline.", BotorchWarning, stacklevel=2)
            return None
        try:
            m = model.models[0] if hasattr(model, "models") else model
            X = m.train_inputs[0]
        except (BotorchError, AttributeError, IndexError, TypeError):
            warnings.warn("Failed to extract X_baseline.", BotorchWarning, stacklevel=2)
            return None
    if is_torch_tensor(X):
        while X.ndim > 2:
            X = X[0]
        return X
    kind = kind_of(X)
    data, shape = to_flat(X, "X_baseline")
    while len(shape) > 2:
        size = 1
        for s in shape[1:]:
            size *= s
        data, shape = data[:size], tuple(shape[1:])
    return from_flat(data, shape, kind)


__all__ = ["columnwise_clamp", "fix_features", "get_X_baseline"]

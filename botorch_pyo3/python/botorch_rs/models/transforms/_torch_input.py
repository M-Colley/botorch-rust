"""Lazy device operations for public input-transform objects.

Array calls retain the native implementation. Tensor state is exported to
native descriptors only at an explicit array/model boundary, never during a
Tensor numerical call.
"""

from __future__ import annotations

import sys
from typing import Any


def is_tensor(value: Any) -> bool:
    torch = sys.modules.get("torch")
    return torch is not None and isinstance(value, torch.Tensor)


def _tensor(value: Any, like: Any) -> Any:
    import torch

    return (
        value.to(like)
        if isinstance(value, torch.Tensor)
        else torch.as_tensor(value, dtype=like.dtype, device=like.device)
    )


def affine_stat(transform: Any, which: int) -> Any:
    value = getattr(transform, "_coefficient" if which else "_offset", None)
    if value is None or not is_tensor(value):
        return None
    return value if transform.learn_coefficients and transform.training else value.detach()


def _select(X: Any, indices: Any) -> Any:
    return X if indices is None else X[..., indices]


def _replace(X: Any, indices: Any, values: Any) -> Any:
    import torch

    if indices is None:
        return values
    batch = torch.broadcast_shapes(X.shape[:-2], values.shape[:-2])
    result = X.expand(*batch, *X.shape[-2:]).clone()
    result[..., indices] = values
    return result


def affine_apply(transform: Any, X: Any, inverse: bool) -> Any:
    import torch

    from botorch_rs._arrays import kind_of
    from botorch_rs.exceptions.errors import BotorchTensorDimensionError

    if X.ndim < 2:
        raise BotorchTensorDimensionError(f"`X` must have at least 2 dimensions, but has {X.ndim}.")
    transform._check_shape(tuple(X.shape))
    selected = _select(X, transform.indices)
    actual_inverse = inverse != transform.reverse
    learn = not actual_inverse and transform.learn_coefficients and transform.training
    if learn:
        batch_ndim = min(len(transform.batch_shape), selected.ndim - 2)
        dimensions = (*range(selected.ndim - batch_ndim - 2), selected.ndim - 2)
        if transform._kind_name == "normalize":
            offset = selected.amin(dim=dimensions).unsqueeze(-2)
            coefficient = selected.amax(dim=dimensions).unsqueeze(-2) - offset
            small = coefficient < transform.min_range
            transform._coefficient = torch.where(small, 1.0, coefficient)
            transform._offset = (
                torch.where(small, 0.0, offset) + (0.5 - transform.center) * transform._coefficient
            )
        elif transform._kind_name == "standardize":
            count = 1
            for dimension in dimensions:
                count *= selected.shape[dimension]
            if count <= 1:
                offset = selected.mean(dim=dimensions).unsqueeze(-2)
                transform._coefficient = torch.ones_like(offset)
                transform._offset = torch.zeros_like(offset)
            else:
                variance, offset = torch.var_mean(selected, dim=dimensions, correction=1)
                small = variance < transform.min_std**2
                # The selected constant-slice branch is a no-op, without an
                # inactive sqrt(0) contaminating its first or higher backward.
                transform._coefficient = torch.where(
                    small, 1.0, torch.where(small, 1.0, variance).sqrt()
                ).unsqueeze(-2)
                transform._offset = torch.where(small, 0.0, offset).unsqueeze(-2)
        else:
            transform._update_coefficients(selected)
    if not is_tensor(getattr(transform, "_coefficient", None)):
        transform._coefficient = _tensor(transform.coefficient, X)
        transform._offset = _tensor(transform.offset, X)
    coefficient = affine_stat(transform, 1).to(X)
    offset = affine_stat(transform, 0).to(X)
    transform._coefficient, transform._offset = coefficient, offset
    transform._kind = kind_of(X)
    values = (
        selected * coefficient + offset if actual_inverse else (selected - offset) / coefficient
    )
    return _replace(X, transform.indices, values)


def sync_affine_native(transform: Any) -> None:
    """Snapshot direct Tensor state when a native descriptor is requested."""
    if not is_tensor(getattr(transform, "_coefficient", None)):
        return
    import math

    from botorch_rs import _botorch_rs as rs
    from botorch_rs._arrays import to_flat

    if not transform.training or not transform.learn_coefficients:
        transform._coefficient = transform._coefficient.detach()
        transform._offset = transform._offset.detach()
    k, nb = len(transform._coefs[0]), math.prod(transform.batch_shape)
    for attribute, name in (("_coefs", "_coefficient"), ("_offs", "_offset")):
        data, shape = to_flat(getattr(transform, name), name)
        setattr(transform, attribute, transform._expand(list(data), tuple(shape), k, nb, name))
    transform._steps = [transform._make_step(b) for b in range(nb)]
    if getattr(transform, "_plain", False):
        bounds = []
        for off, coef in zip(transform._offs, transform._coefs, strict=True):
            bounds.extend(off)
            bounds.extend(a + b for a, b in zip(off, coef, strict=True))
        transform._rs = rs.Normalize(
            k,
            (bounds, (*transform.batch_shape, 2, k)),
            transform.learn_bounds,
            transform.min_range,
            list(transform.batch_shape),
        )


def clear_affine_tensor_state(transform: Any) -> None:
    transform.__dict__.pop("_coefficient", None)
    transform.__dict__.pop("_offset", None)


def warp_parameters(transform: Any, like: Any) -> dict[str, Any]:
    import torch

    parameters = transform.__dict__.setdefault("_direct_parameters", {})
    for i in (0, 1):
        name = f"concentration{i}"
        if name not in parameters:
            values = transform._get(i)
            parameters[name] = torch.nn.Parameter(
                _tensor(values, like).reshape(transform._param_shape()).clone()
            )
        parameter = parameters[name]
        if parameter.device != like.device or parameter.dtype != like.dtype:
            with torch.no_grad():
                parameter.data = parameter.to(like)
    return parameters


def warp_apply(transform: Any, X: Any, inverse: bool) -> Any:
    import torch

    from botorch_rs.exceptions.errors import BotorchTensorDimensionError
    from botorch_rs.models.transforms.utils import inv_kumaraswamy_warp, kumaraswamy_warp

    if X.ndim < 2 or X.shape[-1] != transform._d:
        raise BotorchTensorDimensionError(f"Warp inputs must have shape (..., n, {transform._d}).")
    if (
        inverse
        and transform.batch_shape
        and tuple(X.shape[-2 - len(transform.batch_shape) : -2]) != transform.batch_shape
    ):
        raise BotorchTensorDimensionError(
            f"The right most batch dims of X must match self.batch_shape: ({transform.batch_shape})."
        )
    parameters = (
        warp_parameters(transform, X)
        if transform._model is None
        else {f"concentration{i}": transform._get(i) for i in (0, 1)}
    )
    if not is_tensor(getattr(transform._normalize, "_coefficient", None)):
        offset, coefficient = transform._steps[0].affine_state
        transform._normalize._coefficient = _tensor(coefficient, X).unsqueeze(-2)
        transform._normalize._offset = _tensor(offset, X).unsqueeze(-2)
    transform._normalize.train(transform.training)
    if not inverse:
        X = transform._normalize.transform(X)
    batch = torch.broadcast_shapes(X.shape[:-2], transform.batch_shape)
    X = X.expand(*batch, *X.shape[-2:])
    selected = X[..., transform.indices]
    values = (inv_kumaraswamy_warp if inverse else kumaraswamy_warp)(
        selected, parameters["concentration0"], parameters["concentration1"], transform._eps
    )
    result = _replace(X, transform.indices, values)
    return transform._normalize.untransform(result) if inverse else result


def round_apply(transform: Any, X: Any) -> Any:
    import torch

    result = X.clone()
    selected = X[..., transform.integer_indices]
    if transform.approximate:
        base = selected.floor()
        values = base + (1 + ((selected - base - 0.5) / transform.tau).tanh()) / 2
    else:
        values = selected + (selected.round() - selected).detach()
    result[..., transform.integer_indices] = values
    for start, cardinality in transform.categorical_features.items():
        selected = X[..., start : start + cardinality]
        one_hot = torch.nn.functional.one_hot(selected.argmax(-1), cardinality).to(X)
        result[..., start : start + cardinality] = selected + (one_hot - selected).detach()
    return result


def log10_apply(transform: Any, X: Any, inverse: bool) -> Any:
    selected = X[..., transform.indices]
    return _replace(X, transform.indices, 10.0**selected if inverse else selected.log10())

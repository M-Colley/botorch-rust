"""Lazy device operations for public input-transform objects.

Array calls retain the native implementation. Tensor state is exported to
native descriptors only at an explicit array/model boundary, never during a
Tensor numerical call.
"""

from __future__ import annotations

import sys
from contextvars import ContextVar
from typing import Any

# Model adapters rebuild fitted-statistic graphs from training data. Direct
# public calls retain the reference's detached eval/preprocessing getters.
_model_affine_graphs: ContextVar[frozenset[int]] = ContextVar(
    "botorch_input_model_affine_graphs", default=frozenset()
)


def is_tensor(value: Any) -> bool:
    torch = sys.modules.get("torch")
    return torch is not None and isinstance(value, torch.Tensor)


def has_tensor_state(transform: Any) -> bool:
    """Whether an actual leaf/module or device config requires the Tensor model.

    Inspect only explicit transform/container state, without exporting any
    native parameter arrays or importing Torch for an array-only program.
    """
    torch = sys.modules.get("torch")
    if torch is None:
        return False
    from botorch_rs.models.transforms.input import InputTransform

    seen: set[int] = set()

    def visit(value: Any) -> bool:
        if id(value) in seen:
            return False
        seen.add(id(value))
        if isinstance(value, torch.nn.Module):
            from botorch_rs.models._backend_selection import requires_tensor_backend

            return requires_tensor_backend(value)
        if isinstance(value, torch.nn.Parameter):
            return True
        if isinstance(value, torch.Tensor):
            return value.requires_grad or value.device.type != "cpu"
        if isinstance(value, InputTransform):
            return any(visit(child) for child in value.__dict__.values())
        if isinstance(value, dict):
            return any(visit(child) for child in value.values())
        if isinstance(value, (tuple, list)):
            return any(visit(child) for child in value)
        return False

    return visit(transform)


def _tensor(value: Any, like: Any) -> Any:
    import torch

    return (
        value.to(like)
        if isinstance(value, torch.Tensor)
        else torch.as_tensor(value, dtype=like.dtype, device=like.device)
    )


def native_buffer_value(transform: Any, name: str) -> Any:
    """Keep untouched array constructor values exact at a native boundary."""
    value = getattr(transform, name)
    torch = sys.modules.get("torch")
    if torch is None:
        return value
    record = transform.__dict__.get("_native_buffer_values", {}).get(name)
    if record is not None:
        original, snapshot = record
        if value is snapshot and torch.equal(
            value, torch.as_tensor(original, dtype=value.dtype, device=value.device)
        ):
            return original
    return value


def affine_stat(transform: Any, which: int) -> Any:
    if transform.__dict__.get("_native_affine_dormant", False):
        return None
    value = getattr(transform, "_coefficient" if which else "_offset", None)
    if value is None or not is_tensor(value):
        return None
    live = transform.learn_coefficients and transform.training
    return value if live or id(transform) in _model_affine_graphs.get() else value.detach()


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

    if transform.__dict__.get("_native_affine_dormant", False):
        transform._buffers["_coefficient"] = None
        transform._buffers["_offset"] = None
        transform.__dict__["_native_affine_dormant"] = False
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
        # Reading either native property can export a complete Tensor state.
        # Capture both old values before publishing either new attribute.
        coefficient, offset = transform.coefficient, transform.offset
        transform._coefficient, transform._offset = _tensor(coefficient, X), _tensor(offset, X)
    # A detached getter must not overwrite the registered fitted buffer: its
    # graph may still be needed by a model using the same caller transform.
    transform._coefficient = transform._coefficient.to(X)
    transform._offset = transform._offset.to(X)
    coefficient, offset = affine_stat(transform, 1), affine_stat(transform, 0)
    transform._kind = kind_of(X)
    values = (
        selected * coefficient + offset if actual_inverse else (selected - offset) / coefficient
    )
    return _replace(X, transform.indices, values)


def sync_affine_native(transform: Any) -> None:
    """Snapshot direct Tensor state when a native descriptor is requested."""
    if transform.__dict__.get("_native_affine_dormant", False):
        return
    if not is_tensor(getattr(transform, "_coefficient", None)) or not is_tensor(
        getattr(transform, "_offset", None)
    ):
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
    if getattr(transform, "_native", False):
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
    if "_buffers" in transform.__dict__:
        transform._buffers["_coefficient"] = None
        transform._buffers["_offset"] = None
        transform.__dict__["_native_affine_dormant"] = True


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

    transform.__dict__.get("_native_default_parameters", {}).clear()

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


def append_apply(transform: Any, X: Any) -> Any:
    import torch

    if transform._f is not None:
        selected = X if transform.indices is None else X[..., transform.indices]
        features = transform._f(selected, **transform.fkwargs)
        if not isinstance(features, torch.Tensor):
            raise TypeError("An append callable must return a Tensor for Tensor inputs.")
        count = features.shape[-2]
    else:
        features = _tensor(transform.feature_set, X)
        count = features.shape[-2]
    if transform.skip_expand:
        expanded = X.reshape(*X.shape[:-2], -1, count, X.shape[-1])
    else:
        expanded = X.unsqueeze(-2).expand(*X.shape[:-1], count, X.shape[-1])
    if transform._f is None:
        features = features.expand(*expanded.shape[:-1], features.shape[-1])
    result = torch.cat((expanded, features), -1)
    transform._n_f = count
    return result.reshape(*result.shape[:-3], -1, result.shape[-1])


def one_hot_apply(transform: Any, X: Any, inverse: bool) -> Any:
    import torch

    if not transform.categorical_features:
        return X
    groups = [
        list(range(start, start + cardinality))
        for start, cardinality in transform.categorical_features.items()
    ]
    if inverse:
        result = X.new_zeros(*X.shape[:-1], transform.dim)
        result[..., transform.numerical_idx] = X[..., transform.new_numerical_idx]
        for position, group in zip(transform.ordinal_idx, groups, strict=True):
            result[..., group] = torch.nn.functional.one_hot(
                X[..., position].long(), len(group)
            ).to(X)
    else:
        result = X[..., : transform.numeric_dim].clone()
        result[..., transform.new_numerical_idx] = X[..., transform.numerical_idx]
        for position, group in zip(transform.ordinal_idx, groups, strict=True):
            result[..., position] = X[..., group].argmax(-1)
    return result


def categorical_apply(transform: Any, X: Any) -> Any:
    result = X.new_zeros(
        *X.shape[:-1], len(transform.numerical_idx) + sum(transform.categorical_features.values())
    )
    result[..., transform.new_numerical_idx] = X[..., transform.numerical_idx]
    for index, columns in zip(transform.ordinal_idx, transform.encoded_idx, strict=True):
        encoded = transform.encoders[index](X[..., index].long())
        if not is_tensor(encoded):
            raise TypeError("A categorical encoder must return a Tensor for Tensor inputs.")
        result[..., columns] = encoded.to(result)
    return result


def imputation_parameter(transform: Any, like: Any) -> Any:
    import torch

    parameters = transform.__dict__.setdefault("_direct_parameters", {})
    if "raw_imputation_values" not in parameters:
        parameters["raw_imputation_values"] = torch.nn.Parameter(
            _tensor(transform._step.params, like)
        )
    return parameters["raw_imputation_values"]


def imputation_values(transform: Any, like: Any) -> Any:
    import torch

    raw = (
        imputation_parameter(transform, like)
        if transform._model is None
        else _tensor(transform.raw_imputation_values, like)
    )
    values = raw.reshape(transform.num_tasks, transform.d + 1)
    if transform.bounds is not None:
        constraint = getattr(transform, "raw_imputation_values_constraint", None)
        if constraint is not None:
            return constraint.transform(raw).reshape(transform.num_tasks, transform.d + 1)
        bounds = _tensor(transform.bounds, values)
        lower = torch.cat((bounds[0], values.new_zeros(1)))
        upper = torch.cat((bounds[1], values.new_ones(1)))
        values = lower + (upper - lower) * values.sigmoid()
    return values


def imputation_apply(transform: Any, X: Any) -> Any:
    import torch

    transform.__dict__.get("_native_default_parameters", {}).clear()

    if X.shape[-1] == transform.d:
        if transform.target_task is None:
            raise ValueError("Inputs without a task column require target_task.")
        X = torch.cat((X, X.new_full((*X.shape[:-1], 1), transform.target_task)), -1)
    elif X.shape[-1] != transform.d + 1:
        raise ValueError(f"Expected {transform.d} or {transform.d + 1} input columns.")
    values = imputation_values(transform, X).to(X)
    result = X.clone()
    tasks = X[..., -1].long()
    task_values = getattr(transform, "_task_values", None)
    missing_mask = getattr(transform, "missing_mask", None)
    if task_values is not None and missing_mask is not None:
        for position, task in enumerate(task_values):
            mask = (tasks == task).unsqueeze(-1) & missing_mask[position]
            result = torch.where(mask, values[position], result)
        return result
    for position, (task, observed) in enumerate(sorted(transform.feature_indices.items())):
        missing = [column for column in range(transform.d) if column not in observed]
        result[..., missing] = torch.where(
            (tasks == task).unsqueeze(-1), values[position, missing], result[..., missing]
        )
    return result


def perturbation_apply(transform: Any, X: Any) -> Any:
    import torch

    selected = X if transform.indices is None else X[..., transform.indices]
    if transform._function is None:
        perturbations = _tensor(transform.perturbation_set, X)
        perturbations = perturbations.expand(X.shape[-2], *perturbations.shape)
    else:
        perturbations = transform._function(selected)
        if not isinstance(perturbations, torch.Tensor):
            raise TypeError("A perturbation callable must return a Tensor for Tensor inputs.")
    if (
        perturbations.ndim < 3
        or perturbations.shape[-3] != X.shape[-2]
        or perturbations.shape[-1] != selected.shape[-1]
    ):
        raise ValueError("Perturbations must have shape (..., q, n_w, selected_features).")
    transform._n_w = perturbations.shape[-2]
    # The expanded fixed set has overlapping storage. Cached buffers must be
    # writable so Module.load_state_dict can copy their saved values back.
    transform._perturbations = perturbations.transpose(-3, -2).clone()
    transform._last_batch_shape = tuple(transform._perturbations.shape[:-2])
    batch = torch.broadcast_shapes(X.shape[:-2], perturbations.shape[:-3])
    expanded = selected.unsqueeze(-2)
    perturbed = expanded * perturbations if transform.multiplicative else expanded + perturbations
    if transform.bounds is not None:
        bounds = _tensor(transform.bounds, X)
        perturbed = torch.maximum(torch.minimum(perturbed, bounds[1]), bounds[0])
    if transform.indices is None:
        result = perturbed
    else:
        result = X.unsqueeze(-2).expand(*batch, X.shape[-2], transform._n_w, X.shape[-1]).clone()
        result[..., transform.indices] = perturbed
    return result.reshape(*batch, X.shape[-2] * transform._n_w, X.shape[-1])

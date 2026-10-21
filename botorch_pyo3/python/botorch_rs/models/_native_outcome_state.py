"""Keep native inverse outcome maps aligned with registered public buffers.

Training targets and training noise already live in transformed units. Updating
an inverse map must leave those fitted observations and their factorization
unchanged, just as mutating an outcome module after fitting a Torch model does.
"""

from __future__ import annotations

import copy
import math
import sys
from typing import Any

from botorch_rs import _botorch_rs as _rs
from botorch_rs._arrays import broadcast_to, to_flat
from botorch_rs.models.transforms.outcome import (
    ChainedOutcomeTransform,
    Standardize,
    StratifiedStandardize,
    _normalize_outputs,
    is_plain_standardize,
)


def _statistics(transform: Any) -> tuple[Any, tuple[Any, ...]]:
    if not transform._is_trained:
        raise RuntimeError("Standardize must be fitted before computing a posterior.")
    fitted = transform._rs
    means, shape = fitted.means
    stdvs, _ = fitted.stdvs
    signature = (tuple(means), tuple(stdvs), tuple(shape), transform.min_stdv)
    return fitted, signature


def _steps(
    transform: Any, outputs: int, batch: tuple[int, ...], dimension: int
) -> tuple[list[list[Any]], tuple[Any, ...]]:
    count = math.prod(batch) * outputs
    if isinstance(transform, ChainedOutcomeTransform):
        steps: list[list[Any]] = [[] for _ in range(count)]
        signatures = []
        for child in transform.values():
            values, signature = _steps(child, outputs, batch, dimension)
            for target, value in zip(steps, values, strict=True):
                target.extend(value)
            signatures.append(signature)
        return steps, ("chain", *signatures)
    selected = _normalize_outputs(getattr(transform, "_outputs", None), outputs)
    if isinstance(transform, StratifiedStandardize):
        if not transform._is_trained:
            raise RuntimeError("StratifiedStandardize must be fitted before prediction.")
        transform._sync_native_statistics()
        step = transform._resolved_step(dimension)
        means, stdvs = step.stats
        state = transform.__dict__.get("_tensor_standardize")
        signature = (
            "stratified",
            transform._stratification_idx,
            tuple(transform._task_values),
            tuple(means),
            tuple(stdvs),
            transform.min_stdv,
            None if state is None else tuple(to_flat(state.strata_mapping)[0]),
        )
        return [[step] for _ in range(count)], signature
    if isinstance(transform, Standardize):
        fitted, signature = _statistics(transform)
        means, shape = fitted.means
        stdvs, _ = fitted.stdvs
        target_shape = (*batch, 1, outputs)
        means = broadcast_to(means, shape, target_shape)
        stdvs = broadcast_to(stdvs, shape, target_shape)
        return [
            []
            if selected is not None and i % outputs not in selected
            else [_rs.OutcomeStep.standardize_fitted(mean, stdv, transform.min_stdv)]
            for i, (mean, stdv) in enumerate(zip(means, stdvs, strict=True))
        ], ("standardize", signature, None if selected is None else tuple(selected))
    step = transform._step()
    return [
        [] if selected is not None and i % outputs not in selected else [step] for i in range(count)
    ], (
        transform._name,
        getattr(transform, "power", None),
        None if selected is None else tuple(selected),
    )


def _scalar_affine(transform: Any) -> tuple[float, float, tuple[Any, ...]]:
    """Fold native scalar Standardize compositions in their inverse order."""
    if isinstance(transform, ChainedOutcomeTransform):
        offset, scale, signatures = 0.0, 1.0, []
        for child in reversed(list(transform.values())):
            child_offset, child_scale, signature = _scalar_affine(child)
            offset, scale = child_offset + child_scale * offset, child_scale * scale
            signatures.append(signature)
        return offset, scale, ("chain", *signatures)
    fitted, signature = _statistics(transform)
    means, _ = fitted.means
    stdvs, _ = fitted.stdvs
    if len(means) != 1:
        raise RuntimeError("This native outcome composition requires scalar statistics.")
    return float(means[0]), float(stdvs[0]), signature


def sync_native_outcome(model: Any) -> None:
    """Refresh only the inverse map before a prediction or native snapshot."""
    if "torch" in sys.modules and sys.modules["torch"] is None:
        return
    native = getattr(model, "_rs", None)
    setter = getattr(native, "_set_fitted_outcome_state", None)
    if setter is None:
        return
    try:
        transform = model.outcome_transform
    except AttributeError:
        # Native module binding can request a parameter snapshot before the
        # constructor has attached its fitted outcome descriptor. Absence at
        # that point must not erase the already-fitted inverse map in Rust.
        return
    samples = model.__dict__.get("_samples")
    signature: tuple[Any, ...]
    key: tuple[Any, ...]
    if "_fb_outcome_transform" in model.__dict__ and samples is not None:
        signature = ("identity",) if transform is None else _scalar_affine(transform)[2]
        key = (tuple(id(member) for member in samples), signature)
        if model.__dict__.get("_native_outcome_state") != key:
            for member in samples:
                member.outcome_transform = copy.deepcopy(transform)
                sync_native_outcome(member)
            model.__dict__["_native_outcome_state"] = key
        return
    standardize, chains = None, None
    if transform is None:
        signature = ("identity",)
    elif isinstance(native, _rs.SingleTaskGP):
        if is_plain_standardize(transform):
            standardize, signature = _statistics(transform)
        else:
            chains, signature = _steps(
                transform,
                model.num_outputs,
                tuple(native.batch_shape),
                getattr(native, "transformed_dim", native.dim),
            )
    elif isinstance(transform, Standardize) and not isinstance(transform, StratifiedStandardize):
        standardize, signature = _statistics(transform)
    else:
        offset, scale, signature = _scalar_affine(transform)
        standardize = _rs.Standardize.fitted([[offset]], [[scale]], 1e-8, [])
    members = model.__dict__.get("_fantasy_batch")
    key = (id(native), None if members is None else tuple(map(id, members)), signature)
    if model.__dict__.get("_native_outcome_state") == key:
        return
    setter(standardize, chains)
    if members is not None:
        for member in members:
            if member is not native:
                member._set_fitted_outcome_state(standardize, chains)
    model.__dict__["_native_outcome_state"] = key

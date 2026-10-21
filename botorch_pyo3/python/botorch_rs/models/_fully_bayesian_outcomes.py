"""Native Gaussian outcome compositions used by fully Bayesian loaders."""

from __future__ import annotations

import copy
import math
from typing import Any

from botorch_rs import _botorch_rs as _rs
from botorch_rs._arrays import from_flat, to_flat
from botorch_rs.exceptions.errors import UnsupportedError
from botorch_rs.models.transforms.outcome import ChainedOutcomeTransform, Standardize


def validate_native_outcome(transform: Any) -> None:
    """Allow scalar Standardize compositions with their actual native law."""
    if transform is None:
        return
    transform_type: Any = getattr(type(transform), "_native_port_descriptor", type(transform))
    if isinstance(transform, ChainedOutcomeTransform) and all(
        getattr(transform_type, name) is getattr(ChainedOutcomeTransform, name)
        for name in ("forward", "__call__", "untransform", "_output_steps")
    ):
        for child in transform.values():
            validate_native_outcome(child)
        return
    if (
        isinstance(transform, Standardize)
        and transform_type.forward is Standardize.forward
        and transform_type.untransform is Standardize.untransform
        and transform_type.__call__ is Standardize.__call__
        and transform_type._output_steps is Standardize._output_steps
        and transform._m == 1
    ):
        return
    raise UnsupportedError(
        "Native fully Bayesian Gaussian outcomes support scalar Standardize "
        "and its behavior-preserving subclasses and chains. Use the Tensor "
        "backend for other supported differentiable linear outcome modules."
    )


def batched_native_outcome(transform: Any, count: int) -> Any:
    """Clone the transform graph for identical MCMC-batched training tables."""
    if transform is None:
        return None
    result = copy.deepcopy(transform)
    if isinstance(result, ChainedOutcomeTransform):
        result._transforms = {
            name: batched_native_outcome(child, count) for name, child in result.items()
        }
    else:
        result._batch_shape = (count,)
        result._rs = _rs.Standardize(result.min_stdv, [count])
    return result.train()


def native_outcome_affine(transform: Any, kind: Any) -> tuple[float, float]:
    """The fitted `raw = offset + scale * transformed` scalar map."""
    zero, one = (from_flat([v], (1, 1), kind) for v in (0.0, 1.0))
    offset, variance = transform.untransform(zero, one)
    location = float(to_flat(offset, "outcome offset")[0][0])
    scale = math.sqrt(float(to_flat(variance, "outcome variance scale")[0][0]))
    return location, scale

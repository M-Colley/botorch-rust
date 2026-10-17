"""Input transform factories (``botorch.models.transforms.factory``)."""

from __future__ import annotations

from collections import OrderedDict
from typing import Any

from botorch_rs._arrays import first_kind, from_flat, to_flat
from botorch_rs.models.transforms.input import (
    ChainedInputTransform,
    InputTransform,
    Normalize,
    OneHotToNumeric,
    Round,
)


def get_rounding_input_transform(
    one_hot_bounds: Any,
    integer_indices: list[int] | None = None,
    categorical_features: dict[int, int] | None = None,
    initialization: bool = False,
    return_numeric: bool = False,
    approximate: bool = False,
) -> ChainedInputTransform:
    """Get a rounding input transform: inputs from the unit cube are
    un-normalized to the raw integer ranges, rounded (integers; one-hot
    encoded categoricals to the largest entry) and normalized back.

    Args:
        one_hot_bounds: The raw search space bounds (``2 x d``) where
            categoricals are one-hot encoded and the integer parameters are
            not normalized.
        integer_indices: The indices of the integer parameters.
        categorical_features: A dictionary mapping indices to cardinalities
            of the categorical features.
        initialization: Whether the rounding is for initialization (the
            bounds of the integers are widened by 0.4999 so that the end
            points of a range are as likely as interior points).
        return_numeric: Whether to return numeric rather than one-hot encoded
            categoricals.
        approximate: Whether to use approximate rounding.

    Returns:
        The rounding ``ChainedInputTransform`` (in eval mode).
    """
    has_integers = integer_indices is not None and len(integer_indices) > 0
    has_categoricals = categorical_features is not None and len(categorical_features) > 0
    if not (has_integers or has_categoricals):
        raise ValueError(
            "A rounding function is a no-op if there are no integer or categorical parammeters."
        )
    data, shape = to_flat(one_hot_bounds, "one_hot_bounds")
    d = shape[-1]
    kind = first_kind(one_hot_bounds)
    if initialization and has_integers:
        values = list(data)
        for i in integer_indices or []:
            values[i] -= 0.4999
            values[d + i] += 0.4999
        init_bounds = from_flat(values, shape, kind)
    else:
        init_bounds = one_hot_bounds
    tfs: dict[str, InputTransform] = OrderedDict()
    if has_integers:
        tfs["unnormalize_tf"] = Normalize(
            d=d,
            bounds=init_bounds,
            indices=integer_indices,
            transform_on_train=False,
            transform_on_eval=True,
            transform_on_fantasize=True,
            reverse=True,
        )
    tfs["round"] = Round(
        approximate=approximate,
        transform_on_train=False,
        transform_on_fantasize=True,
        integer_indices=integer_indices,
        categorical_features=categorical_features,
    )
    if has_integers:
        tfs["normalize_tf"] = Normalize(
            d=d,
            bounds=one_hot_bounds,
            indices=integer_indices,
            transform_on_train=False,
            transform_on_eval=True,
            transform_on_fantasize=True,
            reverse=False,
        )
    if return_numeric and has_categoricals:
        tfs["one_hot_to_numeric"] = OneHotToNumeric(
            dim=d,
            categorical_features=categorical_features,
            transform_on_train=True,
            transform_on_eval=True,
            transform_on_fantasize=True,
        )
    tf = ChainedInputTransform(**tfs)
    tf.eval()
    return tf


__all__ = ["get_rounding_input_transform"]

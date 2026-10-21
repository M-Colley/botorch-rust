"""Serialize public input state without pickling disposable native caches."""

from __future__ import annotations

from typing import Any


def get_state(transform: Any) -> dict[str, Any]:
    from botorch_rs.models.transforms.input import (
        AffineInputTransform,
        InputPerturbation,
        LearnedFeatureImputation,
        Normalize,
        NumericToCategoricalEncoding,
        OneHotToNumeric,
        Warp,
    )

    state = transform.__dict__.copy()
    native: dict[str, Any] = {}
    if isinstance(transform, AffineInputTransform):
        steps = state.pop("_steps")
        # Native array calls may have fitted these independently of the
        # original constructor's coefficient and offset lists.
        native["affine"] = [step.affine_state for step in steps]
    if isinstance(transform, Normalize):
        normalize = state.pop("_native_rs", None)
        if normalize is not None:
            native["normalize"] = (
                normalize.d,
                normalize.bounds,
                normalize.learn_bounds,
                normalize.min_range,
                normalize.batch_shape,
            )
    if isinstance(transform, Warp):
        native["warp"] = [(step.params, step.affine_state) for step in state.pop("_steps")]
    if isinstance(transform, (OneHotToNumeric, NumericToCategoricalEncoding)):
        state.pop("_rs_step")
    if isinstance(transform, LearnedFeatureImputation):
        native["imputation"] = state.pop("_step").params
    if isinstance(transform, InputPerturbation):
        state.pop("_rs_cache")
    state["_pickle_native_state"] = native
    return state


def set_state(transform: Any, state: dict[str, Any]) -> None:
    from botorch_rs import _botorch_rs as rs
    from botorch_rs.models.transforms.input import (
        AffineInputTransform,
        InputPerturbation,
        LearnedFeatureImputation,
        Normalize,
        NumericToCategoricalEncoding,
        OneHotToNumeric,
        Warp,
        _categorical_pairs,
        _one_hot_encoder_classes,
    )

    state = state.copy()
    native = state.pop("_pickle_native_state", {})
    transform.__dict__.update(state)
    if isinstance(transform, AffineInputTransform):
        saved = native["affine"]
        transform._offs = [list(offset) for offset, _ in saved]
        transform._coefs = [list(coefficient) for _, coefficient in saved]
        transform._steps = [transform._make_step(i) for i in range(len(saved))]
    if isinstance(transform, Normalize):
        saved_normalize = native.get("normalize")
        transform._native_rs = None if saved_normalize is None else rs.Normalize(*saved_normalize)
    if isinstance(transform, Warp):
        transform._steps = []
        for parameters, affine in native["warp"]:
            step = transform._new_step()
            if affine is not None and transform._bounds is None:
                # Reconstruct the cached learned bounds, retaining the step's
                # learning policy. Tensor model calls use the actual shared
                # Normalize buffers; this cache serves later native exports.
                offset, coefficient = affine
                low, high = [0.0] * transform._d, [1.0] * transform._d
                for i, column in enumerate(transform.indices):
                    low[column] = offset[i]
                    high[column] = offset[i] + coefficient[i]
                step = step.fit((low + high, (2, transform._d)))
            transform._steps.append(step.with_params(parameters))
    if isinstance(transform, OneHotToNumeric):
        transform._rs_step = rs.InputStep.one_hot_to_numeric(
            transform.dim, _categorical_pairs(transform.categorical_features), transform._flags()
        )
    if isinstance(transform, NumericToCategoricalEncoding):
        pairs = sorted(transform.categorical_features.items())
        encoders = [
            None
            if _one_hot_encoder_classes(transform.encoders[index]) == cardinality
            else transform.encoders[index]
            for index, cardinality in pairs
        ]
        transform._rs_step = rs.InputStep.numeric_to_categorical(
            transform.dim, pairs, encoders, transform._flags()
        )
    if isinstance(transform, LearnedFeatureImputation):
        transform._step = rs.InputStep.imputation(
            sorted(transform.feature_indices.items()),
            transform.d,
            transform.target_task,
            transform._bounds,
            transform._flags(),
        ).with_params(native["imputation"])
    if isinstance(transform, InputPerturbation):
        transform._rs_cache = {}

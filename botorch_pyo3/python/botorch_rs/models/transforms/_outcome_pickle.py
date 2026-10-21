"""Logical native outcome state behind optional Torch module protocols."""

from __future__ import annotations

from typing import Any


def get_state(transform: Any) -> dict[str, Any]:
    state = transform.__dict__.copy()
    native = state.pop("_native_standardize", None)
    if native is not None:
        state["_serialized_standardize"] = (
            native.is_trained,
            native.means if native.is_trained else None,
            native.stdvs if native.is_trained else None,
        )
    step = state.pop("_step", None)
    if step is not None:
        state["_serialized_strata"] = step.stats if step.is_trained else None
    state.pop("_compiled_call_impl", None)
    return state


def set_state(transform: Any, state: dict[str, Any]) -> None:
    from botorch_rs import _botorch_rs as _rs

    state = state.copy()
    standardize = state.pop("_serialized_standardize", None)
    strata = state.pop("_serialized_strata", None)
    transform.__dict__.update(state)
    if standardize is not None:
        trained, means, stdvs = standardize
        transform._native_standardize = (
            _rs.Standardize.fitted(
                [means[0][i : i + transform._m] for i in range(0, len(means[0]), transform._m)],
                [stdvs[0][i : i + transform._m] for i in range(0, len(stdvs[0]), transform._m)],
                transform._min_stdv,
                list(transform._batch_shape),
            )
            if trained
            else _rs.Standardize(transform._min_stdv, list(transform._batch_shape))
        )
    if hasattr(transform, "_stratification_idx"):
        transform._step = _rs.OutcomeStep.stratified(
            max(transform._stratification_idx, 0),
            list(transform._task_values),
            transform._min_stdv,
            strata,
        )

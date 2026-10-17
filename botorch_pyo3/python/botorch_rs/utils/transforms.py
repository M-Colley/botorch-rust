"""Helpers for tensor transformations (``botorch.utils.transforms``):
``normalize_indices``, ``is_ensemble`` and ``is_fully_bayesian``."""

from __future__ import annotations

from typing import Any


def normalize_indices(indices: list[int] | None, d: int) -> list[int] | None:
    r"""Normalize a list of indices to ensure that they are positive.

    Args:
        indices: A list of indices (may contain negative indices for indexing
            "from the back").
        d: The dimension of the tensor to index.

    Returns:
        A normalized list of indices such that each index is between ``0`` and
        ``d-1``, or None if indices is None.
    """
    if indices is None:
        return indices
    normalized_indices = []
    for i in indices:
        if i < 0:
            i = i + d
        if i < 0 or i > d - 1:
            raise ValueError(f"Index {i} out of bounds for tensor or length {d}.")
        normalized_indices.append(i)
    return normalized_indices


def is_ensemble(model: Any) -> bool:
    r"""Check if at least one model is an ensemble model.

    Args:
        model: A BoTorch model (may be a ``ModelList`` or ``ModelListGP``)

    Returns:
       True if at least one model is an ensemble model.
    """
    models = getattr(model, "models", None)
    if isinstance(models, (list, tuple)):
        return any(is_ensemble(m) for m in models)
    return bool(getattr(model, "_is_ensemble", False))


def is_fully_bayesian(model: Any) -> bool:
    r"""Check if at least one model is a fully Bayesian model.

    Args:
        model: A BoTorch model (may be a ``ModelList`` or ``ModelListGP``)

    Returns:
        True if at least one model is a fully Bayesian model.
    """
    models = getattr(model, "models", None)
    if isinstance(models, (list, tuple)):
        return any(is_fully_bayesian(m) for m in models)
    return bool(getattr(model, "_is_fully_bayesian", False))


__all__ = ["is_ensemble", "is_fully_bayesian", "normalize_indices"]

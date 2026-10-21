"""Utilities (``botorch.utils``)."""

from __future__ import annotations

import importlib
from typing import Any

from botorch_rs.utils import multi_objective
from botorch_rs.utils.sampling import (
    HitAndRunPolytopeSampler,
    draw_sobol_normal_samples,
    draw_sobol_samples,
    find_interior_point,
    get_polytope_samples,
    manual_seed,
    sample_polytope,
    sparse_to_dense_constraints,
)
from botorch_rs.utils.types import DEFAULT

#: Names imported on first access (as in BoTorch).
_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    "get_outcome_constraint_transforms": (".constraints", "get_outcome_constraint_transforms"),
    "estimate_feasible_volume": (".feasible_volume", "estimate_feasible_volume"),
    "apply_constraints": (".objective", "apply_constraints"),
    "get_objective_weights_transform": (".objective", "get_objective_weights_transform"),
    "approximate_round": (".rounding", "approximate_round"),
    "batched_multinomial": (".sampling", "batched_multinomial"),
    "average_over_ensemble_models": (".transforms", "average_over_ensemble_models"),
    "standardize": (".transforms", "standardize"),
    "t_batch_mode_transform": (".transforms", "t_batch_mode_transform"),
}

__all__ = [
    "DEFAULT",
    "HitAndRunPolytopeSampler",
    "draw_sobol_normal_samples",
    "draw_sobol_samples",
    "find_interior_point",
    "get_polytope_samples",
    "manual_seed",
    "sample_polytope",
    "sparse_to_dense_constraints",
    "multi_objective",
    *_LAZY_IMPORTS,
]


def __getattr__(name: str) -> Any:
    if name in _LAZY_IMPORTS:
        rel_module, attr = _LAZY_IMPORTS[name]
        module = importlib.import_module(rel_module, __name__)
        value = getattr(module, attr)
        globals()[name] = value
        return value
    try:
        return importlib.import_module(f".{name}", __name__)
    except ImportError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None

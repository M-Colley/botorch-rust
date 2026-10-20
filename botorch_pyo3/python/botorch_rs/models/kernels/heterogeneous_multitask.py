"""Feature partitions and exact binary activation kernels for heterogeneous tasks."""

from __future__ import annotations

from operator import index as integer_index
from typing import Any

from botorch_rs import _botorch_rs as _rs
from botorch_rs._arrays import to_flat
from botorch_rs.kernels import Kernel


def _features(feature_indices: list[list[int]]) -> list[set[int]]:
    if not feature_indices:
        raise ValueError("At least one task feature list is required.")
    output = []
    for features in feature_indices:
        indices = [integer_index(value) for value in features]
        if any(value < 0 for value in indices):
            raise ValueError("Feature indices must be nonnegative integers.")
        if len(set(indices)) != len(indices):
            raise ValueError("Each task feature list must contain unique indices.")
        output.append(set(indices))
    return output


def find_subsets(feature_indices: list[list[int]]) -> list[set[int]]:
    """Return maximal disjoint feature groups with the same active tasks."""
    features = _features(feature_indices)
    subsets = [features[0]]
    for task in feature_indices[1:]:
        remaining_task = set(task)
        refined = []
        for subset in subsets:
            common = remaining_task & subset
            remaining = subset - common
            if common:
                refined.append(common)
                remaining_task = remaining_task.difference(common)
            if remaining:
                refined.append(remaining)
        if remaining_task:
            refined.append(remaining_task)
        subsets = refined
    return subsets


def map_subsets(
    subsets: list[set[int]], feature_indices: list[list[int]]
) -> tuple[dict[tuple[int, ...], list[int]], list[list[int]]]:
    """Map a disjoint feature partition to task memberships and binary rows."""
    features = _features(feature_indices)
    keys = [tuple(integer_index(value) for value in subset) for subset in subsets]
    normalized = [set(key) for key in keys]
    all_features = set().union(*features)
    seen: set[int] = set()
    for subset in normalized:
        if any(value < 0 for value in subset):
            raise ValueError("Feature indices must be nonnegative integers.")
        if seen & subset:
            raise ValueError("Feature subsets must be disjoint; found duplicate indices.")
        if not subset <= all_features:
            raise ValueError("A subset contains a feature absent from every task.")
        if not any(subset <= task for task in features):
            raise ValueError("Each feature subset must be fully active in at least one task.")
        seen |= subset
    if seen != all_features:
        raise ValueError("Feature subsets must cover every task feature.")
    if len(normalized) != len({tuple(subset) for subset in normalized}):
        raise ValueError("Feature subsets must be unique.")
    mapping = {
        key: [i for i, task in enumerate(features) if subset <= task]
        for key, subset in zip(keys, normalized, strict=True)
    }
    binary = [[int(subset <= task) for subset in normalized] for task in features]
    return mapping, binary


class DeltaKernel(Kernel):
    """Exact zero/one covariance ``1[x1 == 1] * 1[x2 == 1]`` for one feature."""

    def _base_node(self) -> _rs.KernelNode:
        node_type: Any = _rs.KernelNode
        return node_type.activation([(0, 1.0)])

    def _standard_spec(self, d: int) -> _rs.KernelSpec | None:
        assert (len(self.active_dims) if self.active_dims is not None else d) == 1, (
            "DeltaKernel expects 1D inputs!"
        )
        return None

    def __call__(
        self,
        x1: Any,
        x2: Any = None,
        diag: bool = False,
        last_dim_is_batch: bool = False,
        **params: Any,
    ) -> Any:
        for value in (x1, x2):
            if value is not None:
                shape = to_flat(value)[1]
                dimensions = (
                    len(self.active_dims)
                    if self.active_dims is not None
                    else shape[-1]
                    if len(shape) > 1
                    else 1
                )
                assert dimensions == 1, "DeltaKernel expects 1D inputs!"
        return super().__call__(x1, x2, diag, last_dim_is_batch, **params)

    forward = __call__


__all__ = ["DeltaKernel", "find_subsets", "map_subsets"]

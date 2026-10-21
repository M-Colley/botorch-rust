"""Hierarchical feature parsing against official BoTorch, plus graph validation."""

from __future__ import annotations

import random

import pytest

from botorch_rs.models.hierarchical import utils as rs


@pytest.mark.parametrize("keep", [False, True])
@pytest.mark.parametrize("separate", [False, True])
def test_hierarchical_paths_and_blocks(keep: bool, separate: bool) -> None:
    bt = pytest.importorskip("botorch.models.hierarchical.utils")
    rng = random.Random(51)
    forests = [
        (7, {2: {0: [3], 1: [4]}, 4: {0: [5], 1: [6]}}),
        (3, {}),
        (4, {0: {0: [], 1: [1, 2]}}),
    ]
    for _ in range(10):
        dependencies = {}
        for child in range(1, 12):
            if rng.random() < 0.75:
                parent = rng.randrange(child)
                value = rng.randrange(3)
                dependencies.setdefault(parent, {}).setdefault(value, []).append(child)
        forests.append((12, dependencies))
    for dim, dependencies in forests:
        assert rs.get_orphan_feature_indices(dim, dependencies) == bt.get_orphan_feature_indices(
            dim, dependencies
        )
        assert rs.get_index_to_path(dim, dependencies) == bt.get_index_to_path(dim, dependencies)
        assert rs.get_blocks_with_paths(
            dim, dependencies, keep, separate
        ) == bt.get_blocks_with_paths(dim, dependencies, keep, separate)


@pytest.mark.parametrize(
    "function", [rs.get_orphan_feature_indices, rs.get_index_to_path, rs.get_blocks_with_paths]
)
@pytest.mark.parametrize(
    "dependencies, message",
    [
        ({0: {0: [1]}, 1: {0: [0]}}, "cycle"),
        ({0: {0: [0]}}, "cycle"),
        ({0: {0: [2], 1: [2]}}, "duplicate"),
        ({4: {0: [1]}}, "outside"),
        ({0: {0: [-1]}}, "outside"),
    ],
)
def test_invalid_dependency_graphs(function, dependencies, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        function(3, dependencies)


def test_deep_forest_and_empty_space() -> None:
    dependencies = {parent: {1: [parent + 1]} for parent in range(1199)}
    paths = rs.get_index_to_path(1200, dependencies)
    assert paths[1199] == [(parent, 1) for parent in range(1199)]
    assert rs.get_orphan_feature_indices(0, {}) == []
    assert rs.get_index_to_path(0, {}) == {}
    assert rs.get_blocks_with_paths(0, {}, True, False) == ([], [])

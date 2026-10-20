"""Parse hierarchical feature forests and reject invalid dependency graphs."""

from __future__ import annotations

from operator import index as integer_index

Dependencies = dict[int, dict[int, list[int]]]


def _parents(dim: int, dependencies: Dependencies) -> dict[int, tuple[int, int]]:
    dim = integer_index(dim)
    if dim < 0:
        raise ValueError("dim must be nonnegative.")
    parents: dict[int, tuple[int, int]] = {}
    for parent, branches in dependencies.items():
        if not 0 <= integer_index(parent) < dim:
            raise ValueError(f"Parent feature index {parent} is outside [0, {dim}).")
        for value, children in branches.items():
            for child in children:
                if not 0 <= integer_index(child) < dim:
                    raise ValueError(f"Child feature index {child} is outside [0, {dim}).")
                if child in parents:
                    raise ValueError("The hierarchical dependencies contain duplicate indices.")
                parents[child] = (parent, value)
    # Check every component, including a closed cycle with no orphan root.
    visited: set[int] = set()
    for start in range(dim):
        path: set[int] = set()
        current = start
        while current not in visited:
            if current in path:
                raise ValueError("The hierarchical dependencies contain a cycle.")
            path.add(current)
            if current not in parents:
                break
            current = parents[current][0]
        visited.update(path)
    return parents


def get_orphan_feature_indices(dim: int, hierarchical_dependencies: Dependencies) -> list[int]:
    """Return sorted feature indices with no parent dependency."""
    parents = _parents(dim, hierarchical_dependencies)
    return [feature for feature in range(dim) if feature not in parents]


def get_index_to_path(
    dim: int, hierarchical_dependencies: Dependencies
) -> dict[int, list[tuple[int, int]]]:
    """Map each feature to the ancestor values that activate it."""
    parents = _parents(dim, hierarchical_dependencies)
    paths: dict[int, list[tuple[int, int]]] = {}
    roots = [feature for feature in range(dim) if feature not in parents]
    pending: list[tuple[int, list[tuple[int, int]]]] = [(root, []) for root in reversed(roots)]
    while pending:
        feature, path = pending.pop()
        paths[feature] = path
        descendants = [
            (child, path + [(feature, value)])
            for value, children in hierarchical_dependencies.get(feature, {}).items()
            for child in children
        ]
        pending.extend(reversed(descendants))
    return paths


def get_blocks_with_paths(
    dim: int,
    hierarchical_dependencies: Dependencies,
    keep_hierarchical_features: bool = True,
    separate_hierarchical_features: bool = True,
) -> tuple[list[list[int]], list[list[tuple[int, int]]]]:
    """Group features activated together and return each block's activation path."""
    orphans = get_orphan_feature_indices(dim, hierarchical_dependencies)
    partition = [
        list(children)
        for branches in hierarchical_dependencies.values()
        for children in branches.values()
        if children
    ] + [orphans]
    if not keep_hierarchical_features or separate_hierarchical_features:
        refined = [
            [feature for feature in block if not hierarchical_dependencies.get(feature)]
            for block in partition
        ]
        if keep_hierarchical_features:
            refined += [
                [feature for feature in block if hierarchical_dependencies.get(feature)]
                for block in partition
            ]
        partition = refined
    blocks = sorted(sorted(block) for block in partition if block)
    paths = get_index_to_path(dim, hierarchical_dependencies)
    return blocks, [paths[block[0]] for block in blocks]


__all__ = ["get_blocks_with_paths", "get_index_to_path", "get_orphan_feature_indices"]

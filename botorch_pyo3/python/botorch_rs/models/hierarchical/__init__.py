"""Structural utilities for hierarchical feature dependencies."""

from botorch_rs.models.hierarchical.utils import (
    get_blocks_with_paths,
    get_index_to_path,
    get_orphan_feature_indices,
)

__all__ = ["get_blocks_with_paths", "get_index_to_path", "get_orphan_feature_indices"]

"""The partitioning of the dominated space into hyperrectangles
(``botorch.utils.multi_objective.box_decompositions.dominated``)."""

from __future__ import annotations

from botorch_rs.utils.multi_objective.box_decompositions.box_decomposition import (
    FastPartitioning,
)


class DominatedPartitioning(FastPartitioning):
    r"""Partition dominated space into axis-aligned hyperrectangles.

    This uses the Algorithm 1 from Lacour et al. (2017).

    Example:
        >>> bd = DominatedPartitioning(ref_point, Y)
    """

    _rs_kind = "dominated"


__all__ = ["DominatedPartitioning"]

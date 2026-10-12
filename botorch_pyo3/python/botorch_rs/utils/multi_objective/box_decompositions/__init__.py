"""Box decompositions of the dominated and non-dominated space
(``botorch.utils.multi_objective.box_decompositions``), computed in Rust."""

from botorch_rs.utils.multi_objective.box_decompositions.box_decomposition_list import (
    BoxDecompositionList,
)
from botorch_rs.utils.multi_objective.box_decompositions.dominated import (
    DominatedPartitioning,
)
from botorch_rs.utils.multi_objective.box_decompositions.non_dominated import (
    FastNondominatedPartitioning,
    NondominatedPartitioning,
)
from botorch_rs.utils.multi_objective.box_decompositions.utils import (
    compute_dominated_hypercell_bounds_2d,
    compute_non_dominated_hypercell_bounds_2d,
)

__all__ = [
    "compute_dominated_hypercell_bounds_2d",
    "compute_non_dominated_hypercell_bounds_2d",
    "BoxDecompositionList",
    "DominatedPartitioning",
    "FastNondominatedPartitioning",
    "NondominatedPartitioning",
]

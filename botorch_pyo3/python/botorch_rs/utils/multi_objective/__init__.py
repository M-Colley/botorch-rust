"""Multi-objective utilities (``botorch.utils.multi_objective``): Pareto
dominance, hypervolumes, box decompositions and scalarizations, computed in
Rust. ``botorch_rs.utils.multi_objective.optimize.optimize_with_nsgaii``
runs NSGA-II in Rust (BoTorch uses pymoo, which is not needed here)."""

from botorch_rs.utils.multi_objective import box_decompositions, hypervolume, pareto, scalarization
from botorch_rs.utils.multi_objective.hypervolume import Hypervolume, infer_reference_point
from botorch_rs.utils.multi_objective.pareto import is_non_dominated
from botorch_rs.utils.multi_objective.scalarization import get_chebyshev_scalarization

__all__ = [
    "Hypervolume",
    "box_decompositions",
    "get_chebyshev_scalarization",
    "hypervolume",
    "infer_reference_point",
    "is_non_dominated",
    "pareto",
    "scalarization",
]

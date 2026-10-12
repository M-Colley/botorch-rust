"""Partitionings of the non-dominated space
(``botorch.utils.multi_objective.box_decompositions.non_dominated``):
``NondominatedPartitioning`` and ``FastNondominatedPartitioning``.

References:
    I. Couckuyt, D. Deschrijver and T. Dhaene, "Towards Efficient
    Multiobjective Optimization: Multiobjective statistical criterions," IEEE
    CEC 2012.
    S. Watanabe. "Approximation of Box Decomposition Algorithm for Fast
    Hypervolume-Based Multi-Objective Optimization," arXiv:2512.05825, 2025.
"""

from __future__ import annotations

from typing import Any

from botorch_rs._arrays import from_pair
from botorch_rs.utils.multi_objective.box_decompositions.box_decomposition import (
    BoxDecomposition,
    FastPartitioning,
)


class NondominatedPartitioning(BoxDecomposition):
    r"""A class for partitioning the non-dominated space into hyper-cells
    (the binary partitioning of Couckuyt et al., 2012; maximization).

    Note: it is only feasible to use this algorithm to compute an exact
    decomposition of the non-dominated space for ``m<5`` objectives
    (``alpha=0.0``).

    The alpha parameter can be increased to obtain an approximate
    partitioning faster. The ``alpha`` is a fraction of the total hypervolume
    encapsuling the entire Pareto set. When a hypercell's volume divided by
    the total hypervolume is less than ``alpha``, we discard the hypercell.

    Example:
        >>> bd = NondominatedPartitioning(ref_point, Y=Y1)
    """

    _rs_kind = "nondominated"

    def __init__(self, ref_point: Any, Y: Any = None, alpha: float = 0.0) -> None:
        """Initialize NondominatedPartitioning.

        Args:
            ref_point: A ``m``-dim array containing the reference point.
            Y: A ``(batch_shape) x n x m``-dim array.
            alpha: A thresold fraction of total volume used in an approximate
                decomposition.
        """
        self.alpha = alpha
        self._setup(ref_point, Y, sort=True, alpha=float(alpha))

    def get_hypercell_bounds(self) -> Any:
        r"""Get the bounds of each hypercell in the decomposition.

        Returns:
            A ``2 x (batch_shape) x num_cells x m``-dim array containing the
            lower and upper vertices bounding each hypercell.
        """
        pair = self._rs.get_hypercell_bounds()
        if pair is None:
            raise AttributeError(f"'{type(self).__name__}' object has no attribute 'batch_shape'")
        return from_pair(pair, self._kind)


class FastNondominatedPartitioning(FastPartitioning):
    r"""A class for partitioning the non-dominated space into hyper-cells.

    Note: this assumes maximization. Internally, it multiplies by -1 and
    performs the decomposition under minimization.

    This class is far more efficient than NondominatedPartitioning for exact
    box partitionings.

    This class uses the two-step approach similar to that in Yang et al.
    (2019), where:
        a) first, Alg 1 from Lacour et al. (2017) is used to find the local
           lower bounds for the maximization problem
        b) second, the local lower bounds are used as the Pareto frontier for
           the minimization problem, and Lacour et al. (2017) is applied again
           to partition the space dominated by that Pareto frontier.

    Example:
        >>> bd = FastNondominatedPartitioning(ref_point, Y=Y1)
    """

    _rs_kind = "fast_nondominated"


__all__ = ["FastNondominatedPartitioning", "NondominatedPartitioning"]

"""The base classes of the box decompositions
(``botorch.utils.multi_objective.box_decompositions.box_decomposition``):
``BoxDecomposition`` and ``FastPartitioning``.

The decompositions are stateful objects backed by Rust: the outcomes, the
Pareto frontier, the local upper bounds and the cells live in the Rust
object, and every method computes in Rust (in float64). Results have the
array type of the outcomes (torch tensors with their floating dtype and
device, NumPy arrays or nested lists).

Reference:
    R. Lacour, K. Klamroth, C. Fonseca. A box decomposition algorithm to
    compute the hypervolume indicator. Computers & Operations Research,
    Volume 79, 2017.
"""

from __future__ import annotations

from typing import Any, ClassVar

from botorch_rs import _botorch_rs as _rs
from botorch_rs._arrays import LIST, ArrayKind, first_kind, from_pair, kind_of, to_flat
from botorch_rs.exceptions.errors import BotorchError
from botorch_rs.utils.multi_objective._common import scalar_zero, size


class BoxDecomposition:
    r"""An abstract class for box decompositions (maximization).

    Note: Internally, the Rust implementation stores the negative reference
    point and outcomes (minimization), as BoTorch does.
    """

    #: The kind of Rust decomposition of the concrete class.
    _rs_kind: ClassVar[str] = ""

    def __init__(self, ref_point: Any, sort: bool, Y: Any = None) -> None:
        """Initialize BoxDecomposition.

        Args:
            ref_point: A ``m``-dim array containing the reference point.
            sort: A boolean indicating whether to sort the Pareto frontier.
            Y: A ``(batch_shape) x n x m``-dim array of outcomes.
        """
        self._setup(ref_point, Y, sort)

    def _setup(self, ref_point: Any, Y: Any, sort: bool, alpha: float = 0.0) -> None:
        if not self._rs_kind:
            raise TypeError(f"Can't instantiate abstract class {type(self).__name__}")
        self.sort = bool(sort)
        self._ref_kind = kind_of(ref_point)
        self._kind: ArrayKind = first_kind(Y, ref_point) if Y is not None else LIST
        y = None if Y is None else to_flat(Y, "Y")
        self._rs = _rs.BoxDecomposition(self._rs_kind, to_flat(ref_point, "ref_point"), y, alpha)

    @property
    def pareto_Y(self) -> Any:
        r"""This returns the non-dominated set.

        Returns:
            A ``(batch_shape) x n_pareto x m``-dim array of outcomes.
        """
        pair = self._rs.pareto_y()
        if pair is None:
            raise BotorchError("pareto_Y has not been initialized")
        return from_pair(pair, self._kind)

    @property
    def ref_point(self) -> Any:
        r"""The reference point (``m`` or ``batch_shape x m``)."""
        return from_pair(self._rs.ref_point, self._ref_kind)

    @property
    def Y(self) -> Any:
        r"""The raw outcomes (``(batch_shape) x n x m``)."""
        pair = self._rs.y()
        if pair is None:
            raise BotorchError("Y data has not been initialized")
        return from_pair(pair, self._kind)

    @property
    def num_outcomes(self) -> int:
        """The number of outcomes."""
        return self._rs.num_outcomes

    @property
    def batch_shape(self) -> Any:
        """The batch shape of the outcomes (``torch.Size`` for torch
        outcomes, else a tuple)."""
        shape = self._rs.batch_shape
        if shape is None:
            raise AttributeError(f"'{type(self).__name__}' object has no attribute 'batch_shape'")
        return size(shape, self._kind)

    def _ensure_kind(self, Y: Any) -> None:
        # Results take the array type of the first outcomes (else of the
        # reference point).
        if not self._rs.has_outcomes:
            kind = kind_of(Y)
            self._kind = kind if kind.lib != "list" else self._ref_kind

    def update(self, Y: Any) -> None:
        r"""Update non-dominated front and decomposition.

        By default, the partitioning is recomputed (``FastPartitioning``
        updates it incrementally for more than two outcomes).

        Args:
            Y: A ``(batch_shape) x n x m``-dim array of new, incremental
                outcomes.
        """
        self._ensure_kind(Y)
        self._rs.update(to_flat(Y, "Y"))

    def reset(self) -> None:
        r"""Reset non-dominated front and decomposition."""
        if not self._rs.has_outcomes:
            raise BotorchError("Y data has not been initialized")
        self._rs.reset()

    def partition_space(self) -> None:
        r"""Compute box decomposition."""
        if not self._rs.has_outcomes:
            raise BotorchError("pareto_Y has not been initialized")
        self._rs.partition_space()

    def get_hypercell_bounds(self) -> Any:  # pragma: no cover - abstract
        r"""Get the bounds of each hypercell in the decomposition.

        Returns:
            A ``2 x (batch_shape) x num_cells x num_outcomes``-dim array
            containing the lower and upper vertices bounding each hypercell.
        """
        raise NotImplementedError

    def compute_hypervolume(self) -> Any:
        r"""Compute hypervolume that is dominated by the Pareto Frontier.

        Returns:
            A ``(batch_shape)``-dim array containing the hypervolume
            dominated by each Pareto frontier (BoTorch's 0-dim
            ``torch.tensor(0.0)`` before outcomes are given).
        """
        hv = self._rs.compute_hypervolume()
        if hv is None:
            return scalar_zero(self._kind if self._kind.lib != "list" else self._ref_kind)
        return from_pair(hv, self._kind)


class FastPartitioning(BoxDecomposition):
    r"""A class for partitioning the (non-)dominated space into hyper-cells
    with Algorithm 1 of Lacour et al. (2017) (abstract).

    Note: this assumes maximization. Internally, it multiplies outcomes by -1
    and performs the decomposition under minimization.

    With more than two outcomes, ``update`` inserts the new Pareto points
    into the local upper bounds incrementally.
    """

    def __init__(self, ref_point: Any, Y: Any = None) -> None:
        """
        Args:
            ref_point: A ``m``-dim array containing the reference point.
            Y: A ``(batch_shape) x n x m``-dim array of outcomes.
        """
        m = to_flat(ref_point, "ref_point")[1][-1:]
        self._setup(ref_point, Y, sort=tuple(m) == (2,))

    def get_hypercell_bounds(self) -> Any:
        r"""Get the bounds of each hypercell in the decomposition.

        Returns:
            A ``2 x (batch_shape) x num_cells x m``-dim array containing the
            lower and upper vertices bounding each hypercell (``None`` before
            outcomes are given).
        """
        pair = self._rs.get_hypercell_bounds()
        return None if pair is None else from_pair(pair, self._kind)

    @property
    def hypercell_bounds(self) -> Any:
        """The cell bounds (BoTorch's buffer ``hypercell_bounds``)."""
        return self.get_hypercell_bounds()

    def _local_upper_bounds(self, name: str) -> tuple[Any, Any]:
        lub = self._rs.local_upper_bounds()
        if lub is None:
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
        return lub

    @property
    def _U(self) -> Any:
        """The local upper bounds of the minimization problem (``n x m``;
        more than two outcomes)."""
        return from_pair(self._local_upper_bounds("_U")[0], self._kind)

    @property
    def _Z(self) -> Any:
        """The defining points of the local upper bounds (``n x m x m``)."""
        return from_pair(self._local_upper_bounds("_Z")[1], self._kind)


__all__ = ["BoxDecomposition", "FastPartitioning"]

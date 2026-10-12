"""A list of box decompositions
(``botorch.utils.multi_objective.box_decompositions.box_decomposition_list``)."""

from __future__ import annotations

from typing import Any

from botorch_rs._arrays import LIST, from_flat, is_torch_tensor, kind_of, to_flat
from botorch_rs.exceptions.errors import BotorchTensorDimensionError
from botorch_rs.utils.multi_objective.box_decompositions.box_decomposition import (
    BoxDecomposition,
)


class BoxDecompositionList:
    r"""A list of box decompositions."""

    def __init__(self, *box_decompositions: BoxDecomposition) -> None:
        r"""Initialize the box decomposition list.

        Args:
            *box_decompositions: An variable number of box decompositions

        Example:
            >>> bd1 = FastNondominatedPartitioning(ref_point, Y=Y1)
            >>> bd2 = FastNondominatedPartitioning(ref_point, Y=Y2)
            >>> bd = BoxDecompositionList(bd1, bd2)
        """
        self.box_decompositions = list(box_decompositions)

    @property
    def pareto_Y(self) -> list[Any]:
        r"""The non-dominated set of each box decomposition (``n_pareto_i x m``)."""
        return [p.pareto_Y for p in self.box_decompositions]

    @property
    def ref_point(self) -> Any:
        r"""The reference points, stacked (``n_box_decompositions x m``)."""
        refs = [to_flat(p.ref_point, "ref_point") for p in self.box_decompositions]
        if not refs:
            raise RuntimeError("stack expects a non-empty TensorList")
        shape = refs[0][1]
        if any(r[1] != shape for r in refs):
            raise RuntimeError("stack expects each tensor to be equal size")
        data = [v for r in refs for v in r[0]]
        return from_flat(data, (len(refs), *shape), kind_of(self.box_decompositions[0].ref_point))

    def get_hypercell_bounds(self) -> Any:
        r"""Get the bounds of each hypercell in the decomposition.

        Returns:
            A ``2 x n_box_decompositions x num_cells x num_outcomes``-dim
            array containing the lower and upper vertices bounding each
            hypercell (decompositions with fewer cells are padded with empty
            cells of zeros).
        """
        if not self.box_decompositions:
            raise RuntimeError("stack expects a non-empty TensorList")
        flats = []
        for p in self.box_decompositions:
            bounds = p.get_hypercell_bounds()
            if bounds is None:
                raise AttributeError("'NoneType' object has no attribute 'shape'")
            data, shape = to_flat(bounds, "bounds")
            if len(shape) != 3:
                raise BotorchTensorDimensionError(
                    "BoxDecompositionList supports unbatched box decompositions only, got "
                    f"cell bounds of shape {tuple(shape)}."
                )
            flats.append((data, shape))
        m = flats[0][1][-1]
        if any(shape[-1] != m for _, shape in flats):
            raise RuntimeError("the box decompositions have different numbers of outcomes")
        max_cells = max(shape[1] for _, shape in flats)
        lower: list[float] = []
        upper: list[float] = []
        for data, shape in flats:
            half = shape[1] * m
            pad = [0.0] * ((max_cells - shape[1]) * m)
            lower.extend(data[:half])
            lower.extend(pad)
            upper.extend(data[half:])
            upper.extend(pad)
        kind = self.box_decompositions[0]._kind
        if kind is LIST:
            kind = self.box_decompositions[0]._ref_kind
        return from_flat(lower + upper, (2, len(flats), max_cells, m), kind)

    def update(self, Y: Any) -> None:
        r"""Update the partitioning.

        Args:
            Y: A ``n_box_decompositions x n x num_outcomes``-dim array or a
                list where the ith element contains the new points for
                box_decomposition ``i``.
        """
        n = len(self.box_decompositions)
        is_array = is_torch_tensor(Y) or kind_of(Y).lib == "numpy"
        if (is_array and Y.ndim != 3 and Y.shape[0] != n) or (isinstance(Y, list) and len(Y) != n):
            raise BotorchTensorDimensionError(
                "BoxDecompositionList.update requires either a batched tensor Y, "
                "with one batch per box decomposition or a list of tensors with "
                "one element per box decomposition."
            )
        for i, p in enumerate(self.box_decompositions):
            p.update(Y[i])

    def compute_hypervolume(self) -> Any:
        r"""Compute hypervolume that is dominated by the Pareto Froniter.

        Returns:
            A ``n_box_decompositions x (batch_shape)``-dim array containing
            the hypervolume dominated by each Pareto frontier.
        """
        if not self.box_decompositions:
            raise RuntimeError("stack expects a non-empty TensorList")
        hvs = [p.compute_hypervolume() for p in self.box_decompositions]
        flats = [to_flat(h, "hypervolume") for h in hvs]
        shape = flats[0][1]
        if any(f[1] != shape for f in flats):
            raise RuntimeError("stack expects each tensor to be equal size")
        data = [v for f in flats for v in f[0]]
        return from_flat(data, (len(flats), *shape), kind_of(hvs[0]))


__all__ = ["BoxDecompositionList"]

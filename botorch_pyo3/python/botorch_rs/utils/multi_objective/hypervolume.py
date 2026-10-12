"""Hypervolume utilities (``botorch.utils.multi_objective.hypervolume``):
``Hypervolume``, ``infer_reference_point`` and
``get_hypervolume_maximizing_subset``.

``Hypervolume`` is BoTorch's dimension-sweep algorithm (Fonseca et al.,
2006) ported line by line to Rust, so float64 results are BoTorch's.
Inputs of other floating dtypes (e.g. float32) are computed in float64. The
NEHVI mixins of BoTorch's module belong to the acquisition functions.

References:
    C. M. Fonseca, L. Paquete, and M. Lopez-Ibanez. An improved
    dimension-sweep algorithm for the hypervolume indicator. IEEE CEC 2006.
    H. Ishibuchi, N. Akedo, and Y. Nojima. A many-objective test problem for
    visually examining diversity maintenance behavior in a decision space.
    GECCO 2011.
"""

from __future__ import annotations

import math
import random
from typing import Any

from botorch_rs import _botorch_rs as _rs
from botorch_rs._arrays import first_kind, from_flat, from_pair, kind_of, to_flat
from botorch_rs.exceptions.errors import BotorchError, BotorchTensorDimensionError
from botorch_rs.utils.multi_objective._common import long_array

#: The smallest objective range used by ``infer_reference_point``.
MIN_Y_RANGE = 1e-7


def infer_reference_point(
    pareto_Y: Any,
    max_ref_point: Any = None,
    scale: float = 0.1,
    scale_max_ref_point: bool = False,
) -> Any:
    r"""Get reference point for hypervolume computations.

    This sets the reference point to be ``ref_point = nadir - scale * range``
    when there is no ``pareto_Y`` that is better than ``max_ref_point``. If
    there's ``pareto_Y`` better than ``max_ref_point``, the reference point
    will be set to ``max_ref_point - scale * range`` if
    ``scale_max_ref_point`` is true and to ``max_ref_point`` otherwise.

    Note: this assumes maximization of all objectives.

    Args:
        pareto_Y: A ``n x m``-dim array of Pareto-optimal points.
        max_ref_point: A ``m`` dim array indicating the maximum reference
            point. Some elements can be NaN, except when ``pareto_Y`` is
            empty.
        scale: A multiplier used to scale back the reference point based on
            the range of each objective.
        scale_max_ref_point: A boolean indicating whether to apply scaling to
            the max_ref_point based on the range of each objective.

    Returns:
        A ``m``-dim array containing the reference point.
    """
    data, shape = to_flat(pareto_Y, "pareto_Y")
    if len(shape) != 2:
        raise BotorchTensorDimensionError(
            f"Expected pareto_Y to be a `n x m` tensor, got shape {tuple(shape)}."
        )
    max_ref = None if max_ref_point is None else to_flat(max_ref_point, "max_ref_point")[0]
    if shape[0] == 0:
        if max_ref is None:
            raise BotorchError("Empty pareto set and no max ref point provided")
        if any(math.isnan(v) for v in max_ref):
            raise BotorchError("Empty pareto set and max ref point includes NaN.")
        kind = first_kind(max_ref_point, pareto_Y)
    else:
        kind = first_kind(pareto_Y, max_ref_point)
    ref = _rs.infer_reference_point((data, shape), max_ref, float(scale), bool(scale_max_ref_point))
    return from_flat(ref, (len(ref),), kind)


class Hypervolume:
    r"""Hypervolume computation dimension sweep algorithm from Fonseca et al.
    (2006), in the variant that BoTorch adapted from Simon Wessing's
    implementation (variant 3, version 1.2).

    Maximization is assumed.
    """

    def __init__(self, ref_point: Any) -> None:
        r"""Initialize hypervolume object.

        Args:
            ref_point: ``m``-dim array containing the reference point.
        """
        self.ref_point = ref_point

    @property
    def ref_point(self) -> Any:
        r"""The reference point (for maximization), an ``m``-dim array."""
        return from_flat(self._ref, (len(self._ref),), self._kind)

    @ref_point.setter
    def ref_point(self, ref_point: Any) -> None:
        data, shape = to_flat(ref_point, "ref_point")
        if len(shape) != 1:
            raise BotorchTensorDimensionError(
                f"Expected ref_point to be a `m`-dim tensor, got shape {tuple(shape)}."
            )
        self._ref = data
        self._kind = kind_of(ref_point)

    def compute(self, pareto_Y: Any) -> float:
        r"""Compute hypervolume.

        Args:
            pareto_Y: A ``n x m``-dim array of Pareto optimal outcomes (points
                that do not dominate the reference point are ignored).

        Returns:
            The hypervolume.
        """
        data, shape = to_flat(pareto_Y, "pareto_Y")
        m = len(self._ref)
        if not shape or shape[-1] != m:
            got = shape[-1] if shape else None
            raise BotorchTensorDimensionError(
                "pareto_Y must have the same number of objectives as ref_point. "
                f"Got {got}, expected {m}."
            )
        elif len(shape) != 2:
            raise BotorchTensorDimensionError(
                f"pareto_Y must have exactly two dimensions, got {len(shape)}."
            )
        return _rs.hypervolume(self._ref, (data, shape))


def _random_choice(remaining: list[int]) -> int:
    # BoTorch draws from Python's global random generator:
    # ``random.choice(list(remaining_idcs))`` with the indices in increasing
    # order (the iteration order of a set of small integers).
    return random.choice(list(remaining))


def get_hypervolume_maximizing_subset(n: int, Y: Any, ref_point: Any) -> tuple[Any, Any]:
    r"""Find an approximately hypervolume-maximizing subset of size ``n``.

    This greedily selects points from Y to maximize the hypervolume of the
    subset sequentially. This has bounded error since hypervolume is
    submodular. When no remaining point increases the hypervolume, one is
    drawn with Python's ``random.choice`` as in BoTorch (so ``random.seed``
    makes the selection reproducible, identically to BoTorch).

    Args:
        n: The size of the subset to return.
        Y: A ``n' x m``-dim array of outcomes.
        ref_point: A ``m``-dim array containing the reference point.

    Returns:
        A two-element tuple containing
            - A ``n x m``-dim array of outcomes.
            - A ``n``-dim array of indices of the outcomes in the original set
              (int64; a list of ints for nested lists).
    """
    data, shape = to_flat(Y, "Y")
    if len(shape) != 2:
        raise NotImplementedError("Only two dimensions are supported (no additional) batch dims.")
    elif shape[0] < n:
        raise ValueError(f"Y has fewer points ({shape[0]}) than the requested subset size ({n}).")
    ref = to_flat(ref_point, "ref_point")[0]
    subset, indices = _rs.get_hypervolume_maximizing_subset(
        int(n), (data, shape), ref, _random_choice
    )
    kind = kind_of(Y)
    return from_pair(subset, kind), long_array(indices, kind)


__all__ = [
    "MIN_Y_RANGE",
    "Hypervolume",
    "get_hypervolume_maximizing_subset",
    "infer_reference_point",
]

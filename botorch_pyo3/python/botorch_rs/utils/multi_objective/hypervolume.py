"""Hypervolume utilities (``botorch.utils.multi_objective.hypervolume``):
``Hypervolume``, ``infer_reference_point``,
``get_hypervolume_maximizing_subset`` and the mixins of the noisy
hypervolume acquisition functions (``SubsetIndexCachingMixin``,
``compute_subset_indices``, ``NoisyExpectedHypervolumeMixin``).

``Hypervolume`` is BoTorch's dimension-sweep algorithm (Fonseca et al.,
2006) ported line by line to Rust, so float64 results are BoTorch's.
Inputs of other floating dtypes (e.g. float32) are computed in float64. The
NEHVI computations of the mixins run in Rust inside the acquisition
functions; the mixins expose their state.

References:
    C. M. Fonseca, L. Paquete, and M. Lopez-Ibanez. An improved
    dimension-sweep algorithm for the hypervolume indicator. IEEE CEC 2006.
    H. Ishibuchi, N. Akedo, and Y. Nojima. A many-objective test problem for
    visually examining diversity maintenance behavior in a decision space.
    GECCO 2011.
"""

from __future__ import annotations

import importlib
import math
import random
import sys
from itertools import combinations
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


def compute_subset_indices(q: int, device: Any = None) -> dict[str, Any]:
    r"""Compute all ``2^q - 1`` distinct non-empty subsets of ``{0, ..., q - 1}``.

    Args:
        q: An integer defining the set ``{0, ..., q - 1}`` whose subsets to
            compute.
        device: The torch device of the result (a torch tensor is returned if
            a device is given or torch has been imported).

    Returns:
        A dict that maps ``"q_choose_i"`` to all size-``i`` subsets (a
        ``(q choose i) x i`` int64 array, or nested lists without torch), in
        the order of ``itertools.combinations``.
    """
    torch = sys.modules.get("torch")
    if torch is None and device is not None:
        torch = importlib.import_module("torch")
    out: dict[str, Any] = {}
    for i in range(1, q + 1):
        subsets = [list(c) for c in combinations(range(q), i)]
        if torch is not None:
            out[f"q_choose_{i}"] = torch.tensor(subsets, dtype=torch.long, device=device)
        else:
            out[f"q_choose_{i}"] = subsets
    return out


class SubsetIndexCachingMixin:
    """A mixin that adds q-subset index computations and caching."""

    def __init__(self) -> None:
        """Initializes the class with ``q_out = -1`` and no subset indices."""
        self.q_out: int = -1
        self.q_subset_indices: dict[str, Any] = {}

    def compute_q_subset_indices(self, q_out: int, device: Any = None) -> dict[str, Any]:
        r"""Returns and caches a dict of the subsets of ``{0, ..., q_out - 1}``
        (recomputed only when ``q_out`` changes).

        Args:
            q_out: The batch size of the objectives.
            device: The torch device of the indices.

        Returns:
            A dict that maps ``"q_choose_i"`` to all size-``i`` subsets.
        """
        if q_out != self.q_out:
            self.q_subset_indices = compute_subset_indices(q_out, device=device)
            self.q_out = q_out
        return self.q_subset_indices


class NoisyExpectedHypervolumeMixin:
    r"""The state of the noisy hypervolume acquisition functions (qNEHVI and
    qLogNEHVI): the box decompositions of the Pareto fronts of the baseline
    under every MC sample, and cached pending points.

    The decompositions, base samples and hypervolumes live in the Rust
    acquisition function; this mixin exposes them with BoTorch's attribute
    names (read-only).
    """

    _rs_nehvi: Any
    _baseline_kind: Any

    @property
    def X_baseline(self) -> Any:
        r"""``X_baseline`` augmented with the pending points cached in the box
        decompositions."""
        return from_pair(self._rs_nehvi.x_baseline_and_pending(), self._baseline_kind)

    @property
    def _X_baseline(self) -> Any:
        """The observed (pruned) baseline."""
        return from_pair(self._rs_nehvi.x_baseline(), self._baseline_kind)

    def _nehvi_cells(self, which: int) -> Any:
        data, shape = self._rs_nehvi.cell_bounds()
        half = len(data) // 2
        return from_flat(data[which * half : (which + 1) * half], shape[1:], self._baseline_kind)

    @property
    def cell_lower_bounds(self) -> Any:
        """The lower vertices of the hypercells under every MC sample
        (``sample_shape x num_cells x m``, padded with empty cells)."""
        return self._nehvi_cells(0)

    @property
    def cell_upper_bounds(self) -> Any:
        """The upper vertices of the hypercells under every MC sample."""
        return self._nehvi_cells(1)

    @property
    def _hypervolumes(self) -> Any:
        """The hypervolume of the cached points under every MC sample."""
        return from_pair(self._rs_nehvi.hypervolumes(), self._baseline_kind)

    @property
    def _baseline_hvs(self) -> Any:
        """The hypervolume of the observed baseline under every MC sample
        (non-incremental NEHVI only)."""
        hvs = self._rs_nehvi.baseline_hvs()
        if hvs is None:
            raise AttributeError("_baseline_hvs is only computed for non-incremental NEHVI.")
        return from_pair(hvs, self._baseline_kind)

    @property
    def _prev_nehvi(self) -> Any:
        """The expected hypervolume improvement of the cached pending points
        (non-incremental NEHVI; zero otherwise)."""
        return from_flat([self._rs_nehvi.prev_nehvi()], (), self._baseline_kind)


__all__ = [
    "MIN_Y_RANGE",
    "Hypervolume",
    "NoisyExpectedHypervolumeMixin",
    "SubsetIndexCachingMixin",
    "compute_subset_indices",
    "get_hypervolume_maximizing_subset",
    "infer_reference_point",
]

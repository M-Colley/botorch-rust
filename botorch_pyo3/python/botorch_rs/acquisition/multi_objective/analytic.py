"""Analytic Expected Hypervolume Improvement
(``botorch.acquisition.multi_objective.analytic``; Yang et al., J. Glob.
Optim. 2019, with gradients as in Daulton et al., 2020)."""

from __future__ import annotations

from typing import Any

from botorch_rs import _botorch_rs as _rs
from botorch_rs import _ops
from botorch_rs._arrays import first_kind, from_flat, to_flat
from botorch_rs.acquisition.multi_objective.base import (
    MultiObjectiveAnalyticAcquisitionFunction,
    partitioning_cells,
    ref_point_arg,
)


class ExpectedHypervolumeImprovement(MultiObjectiveAnalyticAcquisitionFunction):
    r"""Expected Hypervolume Improvement supporting ``m >= 2`` outcomes, for a
    single point (``q = 1``) and independent outcomes.

    Computed per hypercell of the partitioning as the product over the
    outcomes of ``psi(l, l) - psi(l, u) + nu(l, u)``, which equals BoTorch
    main's sum over the ``2^m`` products of the factors by the distributive
    law (meta-pytorch/botorch#3392); posterior transforms are not supported.

    Example:
        >>> model = SingleTaskGP(train_X, train_Y)
        >>> ref_point = [0.0, 0.0]
        >>> EHVI = ExpectedHypervolumeImprovement(model, ref_point, partitioning)
        >>> ehvi = EHVI(test_X)
    """

    def __init__(
        self,
        model: Any,
        ref_point: list[float],
        partitioning: Any,
        posterior_transform: Any = None,
    ) -> None:
        ref = ref_point_arg(ref_point)
        num_outcomes = int(partitioning.num_outcomes)
        if len(ref) != num_outcomes:
            raise ValueError(
                "The length of the reference point must match the number of outcomes. "
                f"Got ref_point with {len(ref)} elements, but expected {num_outcomes}."
            )
        pareto = partitioning.pareto_Y
        p_data, p_shape = to_flat(pareto, "pareto_Y")
        m = num_outcomes
        better = any(all(p_data[i * m + j] > ref[j] for j in range(m)) for i in range(p_shape[0]))
        if not better and p_shape[0] > 0:
            raise ValueError("At least one pareto point must be better than the reference point.")
        super().__init__(model, posterior_transform=posterior_transform)
        self.partitioning = partitioning
        self._kind = first_kind(pareto, ref_point)
        self.ref_point = from_flat(ref, (len(ref),), self._kind)
        self._cells = partitioning_cells(partitioning)
        self._rs = _rs.expected_hypervolume_improvement(
            self._model_handle(None), ref, self._cells, (p_data, tuple(p_shape))
        )

    def _cell_bounds(self, which: int) -> Any:
        data, shape = self._cells
        half = len(data) // 2
        return from_flat(data[which * half : (which + 1) * half], shape[1:], self._kind)

    @property
    def cell_lower_bounds(self) -> Any:
        """The lower vertices of the hypercells (``num_cells x m``)."""
        return self._cell_bounds(0)

    @property
    def cell_upper_bounds(self) -> Any:
        """The upper vertices of the hypercells (``num_cells x m``)."""
        return self._cell_bounds(1)

    def psi(self, lower: Any, upper: Any, mu: Any, sigma: Any) -> Any:
        """Gaussian tail integral for broadcast cell bounds and marginal moments."""
        standardized = _ops.apply("div", _ops.apply("sub", upper, mu), sigma)
        tail = _ops.apply("sub", 1.0, _ops.apply("ndtr", standardized))
        return _ops.apply(
            "add",
            _ops.apply("mul", sigma, _ops.apply("phi", standardized)),
            _ops.apply("mul", _ops.apply("sub", mu, lower), tail),
        )

    def nu(self, lower: Any, upper: Any, mu: Any, sigma: Any) -> Any:
        """Gaussian tail probability times the broadcast hypercell width."""
        standardized = _ops.apply("div", _ops.apply("sub", upper, mu), sigma)
        tail = _ops.apply("sub", 1.0, _ops.apply("ndtr", standardized))
        return _ops.apply("mul", _ops.apply("sub", upper, lower), tail)


__all__ = ["ExpectedHypervolumeImprovement"]

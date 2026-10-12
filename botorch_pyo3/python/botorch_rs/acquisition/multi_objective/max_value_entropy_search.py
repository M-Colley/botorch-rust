r"""Lower bound multi-objective max-value entropy search
(``botorch.acquisition.multi_objective.max_value_entropy_search``), computed
in Rust with exact gradients.

MES estimates the mutual information between the observations at the
candidates and the sampled Pareto fronts: the entropy of the Gaussian
posterior minus the expected entropy of the posterior truncated to the
region each sampled front dominates (given by its box decomposition, e.g.
``compute_sample_box_decomposition``).

References:
    B. Tu, A. Gandy, N. Kantas and B. Shafei. Joint Entropy Search for
    Multi-Objective Bayesian Optimization. NeurIPS 2022.
"""

from __future__ import annotations

from typing import Any

from botorch_rs import _botorch_rs as _rs
from botorch_rs.acquisition.multi_objective.joint_entropy_search import (
    LowerBoundMultiObjectiveEntropySearch,
)
from botorch_rs.sampling.normal import SobolQMCNormalSampler


class qLowerBoundMultiObjectiveMaxValueEntropySearch(LowerBoundMultiObjectiveEntropySearch):
    r"""The acquisition function for the multi-objective max-value entropy
    search, where the batches ``q > 1`` are supported through the lower bound
    formulation.

    This acquisition function computes the mutual information between the
    observation at a candidate point ``X`` and the Pareto optimal outputs.
    The estimated value can be negative, and the lower bound is not monotone
    in ``q``.

    Example:
        >>> _, pf = sample_optimal_points(model, bounds, num_samples=8, num_points=10)
        >>> cells = compute_sample_box_decomposition(pf)
        >>> acq = qLowerBoundMultiObjectiveMaxValueEntropySearch(model, cells)
        >>> value = acq(test_X)
    """

    def __init__(
        self,
        model: Any,
        hypercell_bounds: Any,
        X_pending: Any = None,
        estimation_type: str = "LB",
        num_samples: int = 64,
        *,
        sampler: SobolQMCNormalSampler | None = None,
    ) -> None:
        super().__init__(
            model,
            None,
            None,
            hypercell_bounds,
            X_pending=X_pending,
            estimation_type=estimation_type,
            num_samples=num_samples,
            sampler=sampler,
        )
        self._rs = _rs.q_lower_bound_mo_max_value_entropy_search(
            self._model_handle(),
            self._cells,
            self._pending,
            estimation_type,
            self.sampler._rs,
        )


__all__ = ["qLowerBoundMultiObjectiveMaxValueEntropySearch"]

r"""Predictive entropy search (``botorch.acquisition.predictive_entropy_search``),
computed in Rust.

``qPredictiveEntropySearch`` is the multi-objective PES
(``qMultiObjectivePredictiveEntropySearch``) with one optimal input per
sample: expectation propagation runs at construction over the training
inputs and the optimal inputs, and every evaluation performs one EP update at
the candidates with the cached factors of the optimal inputs. As in BoTorch
(meta-pytorch/botorch#3429), the factors of the training inputs are not used
at the candidates.

References:
    J. Hernández-Lobato, M. Hoffman and Z. Ghahramani. Predictive Entropy
    Search for Efficient Global Optimization of Black-box Functions. NeurIPS
    2014.
"""

from __future__ import annotations

from typing import Any

from botorch_rs._arrays import from_flat, kind_of, to_flat
from botorch_rs.acquisition.multi_objective.predictive_entropy_search import (
    qMultiObjectivePredictiveEntropySearch,
)
from botorch_rs.exceptions.errors import UnsupportedError


class qPredictiveEntropySearch(qMultiObjectivePredictiveEntropySearch):
    r"""The acquisition function for Predictive Entropy Search.

    This acquisition function approximates the mutual information between
    the observation at a candidate point ``X`` and the optimal set of inputs
    using expectation propagation (EP). The batch case ``q > 1`` is supported
    (the joint predictive covariance of the batch).

    Example:
        >>> optimal_inputs, _ = get_optimal_samples(model, bounds, num_optima=12)
        >>> qPES = qPredictiveEntropySearch(model, optimal_inputs)
        >>> qpes = qPES(test_X)
    """

    def __init__(
        self,
        model: Any,
        optimal_inputs: Any,
        maximize: bool = True,
        X_pending: Any = None,
        max_ep_iterations: int = 250,
        ep_jitter: float = 1e-4,
        test_jitter: float = 1e-4,
        threshold: float = 1e-2,
    ) -> None:
        if getattr(model, "num_outputs", 1) != 1:
            raise UnsupportedError("qPredictiveEntropySearch requires a single-output model.")
        data, shape = to_flat(optimal_inputs, "optimal_inputs")
        if len(shape) != 2:
            raise UnsupportedError(
                "The optimal inputs should have a shape of `num_samples x input_dim`."
            )
        pareto_sets = from_flat(data, (shape[0], 1, shape[1]), kind_of(optimal_inputs))
        super().__init__(
            model,
            pareto_sets,
            maximize=maximize,
            X_pending=X_pending,
            max_ep_iterations=max_ep_iterations,
            ep_jitter=ep_jitter,
            test_jitter=test_jitter,
            threshold=threshold,
        )
        self.optimal_inputs = optimal_inputs


__all__ = ["qPredictiveEntropySearch"]

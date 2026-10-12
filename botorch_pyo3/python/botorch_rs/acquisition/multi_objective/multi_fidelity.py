r"""Multi-objective multi-fidelity optimization
(``botorch.acquisition.multi_objective.multi_fidelity``): ``MOMF`` (Irshad,
Karsch and Döpp, *Expected hypervolume improvement for simultaneous
multi-objective and multi-fidelity optimization*, 2021).

MOMF is qEHVI over the objectives and a trust objective (the fidelity,
appended as the last output of the training data), weighted by the inverse
cost of the candidates and the pending points: ``HVI / C`` for a positive
improvement and ``HVI * C`` otherwise. The cost is BoTorch's default
``C(s) = 1 + s`` for the fidelity ``s`` in the last input, or ``cost_call``
(any callable on torch tensors mapping ``batch_shape x q x d`` points to
``batch_shape x q x 1`` costs, differentiated with ``torch.autograd``).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from botorch_rs import _botorch_rs as _rs
from botorch_rs.acquisition.multi_objective.monte_carlo import qExpectedHypervolumeImprovement
from botorch_rs.sampling.normal import SobolQMCNormalSampler


class MOMF(qExpectedHypervolumeImprovement):
    r"""MOMF acquisition function supporting ``m >= 2`` outcomes plus the
    trust objective.

    The model's ``train_X`` has the fidelity parameter ``s`` as its last
    input and ``train_Y`` the trust objective as its last output; the
    reference point and the partitioning include the trust objective.

    Example:
        >>> model = SingleTaskGP(train_X, train_Y)
        >>> ref_point = [0.0, 0.0, 0.0]
        >>> cost_func = lambda X: 5 + X[..., -1:]
        >>> momf = MOMF(model, ref_point, partitioning, cost_call=cost_func)
        >>> momf_val = momf(test_X)
    """

    def __init__(
        self,
        model: Any,
        ref_point: Any,
        partitioning: Any,
        sampler: SobolQMCNormalSampler | None = None,
        objective: Any = None,
        constraints: Any = None,
        eta: Any = 1e-3,
        X_pending: Any = None,
        cost_call: Callable[[Any], Any] | None = None,
    ) -> None:
        r"""MOMF acquisition function.

        Args:
            model: A fitted botorch_rs model (fidelity as last input, trust
                objective as last output).
            ref_point: The ``m + 1`` reference point (including the trust
                objective) of the objective values.
            partitioning: A ``NondominatedPartitioning`` of the current
                front (including the trust objective).
            sampler: The sampler of the base samples (default: 128 Sobol
                samples).
            objective: The ``MCMultiOutputObjective`` of the samples.
            constraints: Outcome constraints (callables, feasible iff
                negative).
            eta: The temperature of the constraint sigmoids.
            X_pending: Pending points (``m x d``), concatenated to every
                candidate set (their cost counts too).
            cost_call: The cost of points (``batch_shape x q x d`` to
                ``batch_shape x q x 1``, torch); default ``1 + s``.
        """
        ref = self._setup_ehvi(
            model,
            ref_point,
            partitioning,
            sampler,
            objective,
            constraints,
            X_pending,
            eta,
            False,
            "The length of the reference point must match the number of outcomes. "
            "Got ref_point with {} elements, but expected {}.",
        )
        if cost_call is not None:
            from botorch_rs._torch_bridge import require_torch

            require_torch()
        self.cost_call = cost_call
        self._rs = _rs.momf(
            self._rs_model,
            ref,
            self._cells,
            self.sampler._rs,
            self._pending,
            objective=self._mo_objective,
            cost_call=cost_call,
            **self._constraint_args,
        )


__all__ = ["MOMF"]

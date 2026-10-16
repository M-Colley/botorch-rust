"""``PositiveIndexKernel`` (``botorch.models.kernels.positive_index``)."""

from __future__ import annotations

from typing import Any

from botorch_rs import _botorch_rs as _rs
from botorch_rs.constraints import GreaterThan, Interval, Positive
from botorch_rs.kernels import (
    IndexKernel,
    Kernel,
    _check_constraint,
    _check_prior,
    _properties,
    _randn,
    _task_covariance,
)
from botorch_rs.priors import LKJCovariancePrior, Prior


class PositiveIndexKernel(IndexKernel):
    r"""A kernel for discrete indices with strictly positive correlations:
    ``k(i, j) = (L L^T + diag(var))_{ij} / (L L^T + diag(var))_{tt}`` with a
    positive factor ``L`` (``covar_factor``) and the target task ``t``
    (normalized to unit variance if ``unit_scale_for_target``).

    ``task_prior`` is a univariate prior on the lower-triangle task
    correlations (e.g. ``BetaPrior(2.5, 1.5)``, BoTorch's ``MultiTaskGP``
    default).
    """

    _params = (
        ("raw_var", "raw_var_constraint", None),
        ("raw_covar_factor", "raw_covar_factor_constraint", None),
    )

    def __init__(
        self,
        num_tasks: int,
        rank: int = 1,
        task_prior: Prior | None = None,
        normalize_covar_matrix: bool = False,
        var_constraint: Interval | None = None,
        target_task_index: int = 0,
        unit_scale_for_target: bool = True,
        **kwargs: Any,
    ) -> None:
        if rank > num_tasks:
            raise RuntimeError(
                "Cannot create a task covariance matrix larger than the number of tasks"
            )
        if not (0 <= target_task_index < num_tasks):
            raise ValueError(
                f"target_task_index must be between 0 and {num_tasks - 1}, "
                f"got {target_task_index}"
            )
        Kernel.__init__(self, **kwargs)
        _check_prior(task_prior, "task_prior")
        if isinstance(task_prior, LKJCovariancePrior):
            raise TypeError(
                "The prior of a PositiveIndexKernel acts on the lower-triangle correlations; "
                "use a univariate prior such as BetaPrior."
            )
        _check_constraint(var_constraint, "var_constraint")
        self.num_tasks = int(num_tasks)
        self.rank = int(rank)
        self.normalize_covar_matrix = normalize_covar_matrix
        self.target_task_index = int(target_task_index)
        self.unit_scale_for_target = bool(unit_scale_for_target)
        self.raw_var_constraint = var_constraint if var_constraint is not None else Positive()
        self.raw_covar_factor_constraint = GreaterThan(0.0)
        self.IndexKernelPrior = task_prior
        batch = self._batch_shape
        raw_var = _randn((*batch, self.num_tasks))
        if self.raw_var_constraint.initial_value is None:
            self._local["raw_var"] = raw_var
        self._local["raw_covar_factor"] = _randn((*batch, self.num_tasks, self.rank), uniform=True)

    def _base_node(self) -> _rs.KernelNode:
        return _rs.KernelNode.positive_index(
            self.num_tasks, self.rank, self.target_task_index, self.unit_scale_for_target
        )

    def _node(self) -> _rs.KernelNode:
        node = Kernel._node(self)
        if self.IndexKernelPrior is not None:
            node = node.with_task_prior(self.IndexKernelPrior._rs)
        return node

    raw_covar_factor = _properties("raw_covar_factor")[0]

    @property
    def covar_factor(self) -> Any:  # type: ignore[override]
        """The positive factor (``batch_shape x num_tasks x rank``)."""
        return self._get_value("raw_covar_factor")

    @covar_factor.setter
    def covar_factor(self, value: Any) -> None:
        self._set_value("raw_covar_factor", value)

    def _eval_covar_matrix(self) -> Any:
        return _task_covariance(
            self, normalize=self.target_task_index if self.unit_scale_for_target else None
        )

    @property
    def covar_matrix(self) -> Any:
        return self._eval_covar_matrix()


__all__ = ["PositiveIndexKernel"]

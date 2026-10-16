r"""Regularization of acquisition functions (``botorch.acquisition.penalized``).

The penalties ``L2Penalty``, ``L1Penalty``, ``GaussianPenalty``,
``GroupLassoPenalty`` and ``L0PenaltyApprox`` of candidate sets and the
penalty objectives ``L1PenaltyObjective`` and ``L0PenaltyApproxObjective`` of
points are evaluated in Rust by ``PenalizedAcquisitionFunction`` and
``PenalizedMCObjective`` (with exact gradients); other penalties (callables of
torch tensors, e.g. BoTorch modules) are called through torch, with gradients
from ``torch.autograd``. Called directly, the penalties evaluate torch tensors
with torch operations, like BoTorch.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

from botorch_rs import _botorch_rs as _rs
from botorch_rs._arrays import to_flat
from botorch_rs.acquisition.acquisition import AcquisitionFunction
from botorch_rs.acquisition.analytic import AnalyticAcquisitionFunction
from botorch_rs.acquisition.objective import (
    GenericMCObjective,
    _botorch_class_names,
)
from botorch_rs.exceptions.errors import UnsupportedError


def _point(value: Any, name: str) -> list[float]:
    data, _ = to_flat(value, name)
    return [float(v) for v in data]


def _torch() -> Any:
    from botorch_rs._torch_bridge import require_torch

    require_torch()
    import torch

    return torch


class L2Penalty:
    r"""L2 penalty class to be added to any arbitrary acquisition function to
    construct a ``PenalizedAcquisitionFunction``: ``max_j ||x_j - p||^2``."""

    def __init__(self, init_point: Any) -> None:
        self.init_point = init_point

    def _spec(self) -> tuple[Any, ...]:
        return ("L2", _point(self.init_point, "init_point"))

    def __call__(self, X: Any) -> Any:
        torch = _torch()
        return torch.linalg.norm((X - self.init_point), ord=2, dim=-1).max(dim=-1).values ** 2


class L1Penalty:
    r"""L1 penalty class to be added to any arbitrary acquisition function to
    construct a ``PenalizedAcquisitionFunction``: ``max_j ||x_j - p||_1``."""

    def __init__(self, init_point: Any) -> None:
        self.init_point = init_point

    def _spec(self) -> tuple[Any, ...]:
        return ("L1", _point(self.init_point, "init_point"))

    def __call__(self, X: Any) -> Any:
        torch = _torch()
        return torch.linalg.norm((X - self.init_point), ord=1, dim=-1).max(dim=-1).values


class GaussianPenalty:
    r"""Gaussian penalty class to be added to any arbitrary acquisition
    function to construct a ``PenalizedAcquisitionFunction``: ``max_j
    exp(||x_j - p||^2 / (2 sigma^2))``."""

    def __init__(self, init_point: Any, sigma: float) -> None:
        self.init_point = init_point
        self.sigma = sigma

    def _spec(self) -> tuple[Any, ...]:
        return ("Gaussian", _point(self.init_point, "init_point"), float(self.sigma))

    def __call__(self, X: Any) -> Any:
        torch = _torch()
        sq_diff = torch.linalg.norm((X - self.init_point), ord=2, dim=-1) ** 2
        return torch.exp(sq_diff / 2 / self.sigma**2).max(dim=-1).values


class GroupLassoPenalty:
    r"""Group lasso penalty class to be added to any arbitrary acquisition
    function to construct a ``PenalizedAcquisitionFunction`` (``q = 1``)."""

    def __init__(self, init_point: Any, groups: list[list[int]]) -> None:
        self.init_point = init_point
        self.groups = groups

    def _spec(self) -> tuple[Any, ...]:
        return (
            "GroupLasso",
            _point(self.init_point, "init_point"),
            [[int(k) for k in g] for g in self.groups],
        )

    def __call__(self, X: Any) -> Any:
        if X.shape[-2] != 1:
            raise NotImplementedError("group-lasso has not been implemented for q>1 yet.")
        return group_lasso_regularizer(X=X.squeeze(-2) - self.init_point, groups=self.groups)


def group_lasso_regularizer(X: Any, groups: list[list[int]]) -> Any:
    r"""Computes the group lasso regularization function for the given point:
    ``sum_g sqrt(|g|) ||X[..., g]||_2``."""
    torch = _torch()
    return torch.sum(
        torch.stack(
            [math.sqrt(len(g)) * torch.linalg.norm(X[..., g], ord=2, dim=-1) for g in groups],
            dim=-1,
        ),
        dim=-1,
    )


def narrow_gaussian(X: Any, a: Any) -> Any:
    """``exp(-0.5 (X / a)^2)``."""
    torch = _torch()
    return torch.exp(-0.5 * (X / a) ** 2)


def nnz_approx(X: Any, target_point: Any, a: Any) -> Any:
    r"""Differentiable relaxation of ``||X - target_point||_0``:
    ``d - sum_k narrow_gaussian(X_k - target_point_k, a)`` (``n x 1``)."""
    d = X.shape[-1]
    if d != target_point.shape[-1]:
        raise ValueError("X and target_point have different shapes.")
    return d - narrow_gaussian(X - target_point, a).sum(dim=-1, keepdim=True)


class L0Approximation:
    r"""Differentiable relaxation of the L0 norm using a Gaussian basis
    function."""

    def __init__(self, target_point: Any, a: float = 1.0, **tkwargs: Any) -> None:
        self.target_point = target_point
        self.a = float(a)

    def __call__(self, X: Any) -> Any:
        torch = _torch()
        return nnz_approx(X=X, target_point=self.target_point, a=torch.tensor(self.a))


class L0PenaltyApprox(L0Approximation):
    r"""Differentiable relaxation of the L0 norm to be added to any arbitrary
    acquisition function to construct a ``PenalizedAcquisitionFunction``: the
    minimum over the points of the relaxation."""

    def _spec(self) -> tuple[Any, ...]:
        return ("L0", _point(self.target_point, "target_point"), self.a)

    def __call__(self, X: Any) -> Any:
        return super().__call__(X).squeeze(dim=-1).min(dim=-1).values


class L1PenaltyObjective:
    r"""L1 penalty objective class. An instance of this class can be added to
    any arbitrary objective to construct a ``PenalizedMCObjective``:
    ``||x_j - p||_1`` of every point (``1 x batch_shape x q``)."""

    def __init__(self, init_point: Any) -> None:
        self.init_point = init_point

    def _spec(self) -> tuple[Any, ...]:
        return ("L1", _point(self.init_point, "init_point"))

    def __call__(self, X: Any) -> Any:
        torch = _torch()
        return torch.linalg.norm((X - self.init_point), ord=1, dim=-1).unsqueeze(dim=0)


class L0PenaltyApproxObjective(L0Approximation):
    r"""Differentiable relaxation of the L0 norm penalty objective class. An
    instance of this class can be added to any arbitrary objective to
    construct a ``PenalizedMCObjective`` (``1 x batch_shape x q``)."""

    def _spec(self) -> tuple[Any, ...]:
        return ("L0", _point(self.target_point, "target_point"), self.a)

    def __call__(self, X: Any) -> Any:
        return super().__call__(X).squeeze(dim=-1).unsqueeze(dim=0)


def _penalty_spec(penalty: Any, objective: bool) -> tuple[Any, ...]:
    """The specification of a penalty (``objective``: of points)."""
    native = (
        (L1PenaltyObjective, L0PenaltyApproxObjective)
        if objective
        else (L2Penalty, L1Penalty, GaussianPenalty, GroupLassoPenalty, L0PenaltyApprox)
    )
    if isinstance(penalty, native):
        return penalty._spec()
    names = _botorch_class_names(penalty)
    name = type(penalty).__name__ if type(penalty).__name__ in names else None
    if objective:
        if name == "L1PenaltyObjective":
            return ("L1", _point(penalty.init_point, "init_point"))
        if name == "L0PenaltyApproxObjective":
            return ("L0", _point(penalty.target_point, "target_point"), float(penalty.a))
    else:
        if name in ("L2Penalty", "L1Penalty"):
            return (name[:2], _point(penalty.init_point, "init_point"))
        if name == "GaussianPenalty":
            return ("Gaussian", _point(penalty.init_point, "init_point"), float(penalty.sigma))
        if name == "GroupLassoPenalty":
            return (
                "GroupLasso",
                _point(penalty.init_point, "init_point"),
                [[int(k) for k in g] for g in penalty.groups],
            )
        if name == "L0PenaltyApprox":
            return ("L0", _point(penalty.target_point, "target_point"), float(penalty.a))
    if not callable(penalty):
        raise TypeError(f"The penalty must be callable, got {type(penalty).__name__}.")
    _torch()
    return ("callable", penalty)


class PenalizedAcquisitionFunction(AcquisitionFunction):
    r"""Single-outcome acquisition function regularized by the given penalty:
    ``raw_acqf(X) - regularization_parameter * penalty_func(X)``.

    The usage is similar to:
        raw_acqf = NoisyExpectedImprovement(...)
        penalty = GroupLassoPenalty(...)
        acqf = PenalizedAcquisitionFunction(raw_acqf, penalty)
    """

    def __init__(
        self,
        raw_acqf: AcquisitionFunction,
        penalty_func: Callable[[Any], Any],
        regularization_parameter: float,
    ) -> None:
        if not isinstance(raw_acqf, AcquisitionFunction):
            raise UnsupportedError("`raw_acqf` must be a botorch_rs acquisition function.")
        super().__init__(raw_acqf.model)
        self.raw_acqf = raw_acqf
        self.penalty_func = penalty_func
        self.regularization_parameter = regularization_parameter
        self._log = raw_acqf._log
        self._rs = _rs.penalized_acquisition_function(
            raw_acqf._rs,
            _penalty_spec(penalty_func, objective=False),
            float(regularization_parameter),
        )

    @property
    def X_pending(self) -> Any:
        return self.raw_acqf.X_pending

    def set_X_pending(self, X_pending: Any = None) -> None:
        if isinstance(self.raw_acqf, AnalyticAcquisitionFunction):
            raise UnsupportedError(
                "The raw acquisition function is Analytic and does not account for X_pending yet."
            )
        self.raw_acqf.set_X_pending(X_pending)

    def _after_evaluation(self) -> None:
        self.raw_acqf._after_evaluation()


class PenalizedMCObjective(GenericMCObjective):
    r"""Penalized MC objective: ``objective(samples, X) - regularization_parameter
    * penalty_objective(X)``, the penalty of every point subtracted from all
    of its samples. Evaluated in Rust by the botorch_rs acquisition functions
    (the objective callable through torch).

    Example:
        >>> regularization_parameter = 0.01
        >>> init_point = torch.zeros(3) # assume data dim is 3
        >>> objective = lambda Y, X: torch.sqrt(Y).sum(dim=-1)
        >>> l1_penalty_objective = L1PenaltyObjective(init_point=init_point)
        >>> l1_penalized_objective = PenalizedMCObjective(
        ...     objective, l1_penalty_objective, regularization_parameter
        ... )
    """

    def __init__(
        self,
        objective: Callable[..., Any],
        penalty_objective: Any,
        regularization_parameter: float,
        expand_dim: int | None = None,
    ) -> None:
        super().__init__(objective=objective)
        self.penalty_objective = penalty_objective
        self.regularization_parameter = regularization_parameter
        self.expand_dim = expand_dim

    def forward(self, samples: Any, X: Any = None) -> Any:
        obj = super().forward(samples=samples, X=X)
        penalty_obj = self.penalty_objective(X)
        if self.expand_dim is not None:
            penalty_obj = penalty_obj.unsqueeze(self.expand_dim)
        if obj.ndim == 1:
            penalty_obj = penalty_obj.squeeze(dim=0)
        return obj - self.regularization_parameter * penalty_obj

    def _spec(self) -> tuple[Any, ...]:
        if self.expand_dim is not None:
            return ("callable", self)
        return penalized_objective_spec(
            self.objective, self.penalty_objective, self.regularization_parameter
        )


def penalized_objective_spec(
    objective: Any, penalty: Any, regularization_parameter: Any
) -> tuple[Any, ...]:
    """The specification of a penalized MC objective (of botorch_rs or
    BoTorch) for the extension."""
    from botorch_rs.acquisition.objective import MCAcquisitionObjective

    inner = (
        objective._spec()
        if isinstance(objective, MCAcquisitionObjective)
        else ("callable", objective)
    )
    _torch()
    return (
        "penalized",
        inner,
        _penalty_spec(penalty, objective=True),
        float(regularization_parameter),
    )


__all__ = [
    "GaussianPenalty",
    "GroupLassoPenalty",
    "L0Approximation",
    "L0PenaltyApprox",
    "L0PenaltyApproxObjective",
    "L1Penalty",
    "L1PenaltyObjective",
    "L2Penalty",
    "PenalizedAcquisitionFunction",
    "PenalizedMCObjective",
    "group_lasso_regularizer",
    "narrow_gaussian",
    "nnz_approx",
]

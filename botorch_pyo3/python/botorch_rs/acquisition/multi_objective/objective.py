"""Multi-output MC objectives (``botorch.acquisition.multi_objective.objective``).

A multi-output objective maps posterior samples (``sample_shape x
batch_shape x q x m``) to ``m'`` objective values per point
(``sample_shape x batch_shape x q x m'``), all to be maximized.
``IdentityMCMultiOutputObjective``, ``WeightedMCMultiOutputObjective`` and
``FeasibilityWeightedMCMultiOutputObjective`` are evaluated in Rust by the
acquisition functions; ``GenericMCMultiOutputObjective`` and other
``MCMultiOutputObjective`` subclasses are callables on torch tensors, called
from the Rust computation with gradients from ``torch.autograd`` (as for
``GenericMCObjective``). BoTorch's own objective objects are accepted too.
"""

from __future__ import annotations

from typing import Any

from botorch_rs import _botorch_rs as _rs
from botorch_rs._arrays import from_flat, is_torch_tensor, kind_of, to_flat
from botorch_rs.acquisition.objective import (
    GenericMCObjective,
    MCAcquisitionObjective,
    _botorch_class_names,
)
from botorch_rs.exceptions.errors import (
    BotorchError,
    BotorchTensorDimensionError,
    UnsupportedError,
)

#: The temperature of ``apply_constraints`` in
#: ``FeasibilityWeightedMCMultiOutputObjective``: BoTorch's ``eta=1e-3``,
#: which ``compute_smoothed_feasibility_indicator`` stores in float32.
FEASIBILITY_WEIGHTING_ETA = 0.0010000000474974513


def _normalize_indices(indices: list[int], d: int) -> list[int]:
    """BoTorch's ``normalize_indices``."""
    out = []
    for i in indices:
        if i < 0:
            i = i + d
        if i < 0 or i > d - 1:
            raise ValueError(f"Index {i} out of bounds for tensor or length {d}.")
        out.append(i)
    return out


def _take_last(samples: Any, idx: list[int]) -> Any:
    """``samples[..., idx]`` for torch tensors, NumPy arrays and nested lists."""
    if is_torch_tensor(samples):
        import torch

        return samples.index_select(-1, torch.tensor(idx, dtype=torch.long, device=samples.device))
    kind = kind_of(samples)
    data, shape = to_flat(samples, "samples")
    m = shape[-1]
    out = [data[i * m + j] for i in range(len(data) // m) for j in idx]
    return from_flat(out, (*shape[:-1], len(idx)), kind)


def _scale_last(samples: Any, weights: list[float]) -> Any:
    """``samples * weights`` (over the last dimension)."""
    if is_torch_tensor(samples):
        import torch

        return samples * torch.as_tensor(weights, dtype=samples.dtype, device=samples.device)
    kind = kind_of(samples)
    data, shape = to_flat(samples, "samples")
    m = shape[-1]
    if m != len(weights):
        raise RuntimeError(
            f"The size of tensor a ({m}) must match the size of tensor b ({len(weights)}) at "
            f"non-singleton dimension {len(shape) - 1}"
        )
    out = [v * weights[i % m] for i, v in enumerate(data)]
    return from_flat(out, shape, kind)


class MCMultiOutputObjective(MCAcquisitionObjective):
    r"""Abstract base class for MC multi-output objectives.

    Subclasses implement ``forward(samples, X=None)`` on torch tensors
    (``sample_shape x batch_shape x q x m`` samples to ``sample_shape x
    batch_shape x q x m'`` objective values).
    """

    _is_mo: bool = True

    def forward(self, samples: Any, X: Any = None) -> Any:  # pragma: no cover - abstract
        raise NotImplementedError

    def _mo_spec(self) -> tuple[Any, ...]:
        """The objective as passed to the extension: evaluated by calling it."""
        return ("callable", self)


class GenericMCMultiOutputObjective(GenericMCObjective, MCMultiOutputObjective):
    r"""Multi-output objective generated from a generic callable
    ``f(samples, X)`` on torch tensors (differentiable with torch.autograd for
    gradient-based optimization).

    Example:
        >>> objective = GenericMCMultiOutputObjective(
        ...     lambda Y, X: torch.stack([Y[..., 0], -Y[..., 1]], dim=-1)
        ... )
    """


class IdentityMCMultiOutputObjective(MCMultiOutputObjective):
    r"""Trivial objective that returns the unaltered samples, or the
    outputs ``outcomes``.

    Example:
        >>> identity_objective = IdentityMCMultiOutputObjective()
        >>> samples = sampler(posterior)
        >>> objective = identity_objective(samples)
    """

    def __init__(self, outcomes: list[int] | None = None, num_outcomes: int | None = None) -> None:
        r"""Initialize Objective.

        Args:
            outcomes: A list of the ``m'`` indices that the weights should be
                applied to.
            num_outcomes: The total number of outcomes ``m``.
        """
        self._outcomes: list[int] | None = None
        if outcomes is not None:
            outcomes = [int(i) for i in outcomes]
            if len(outcomes) < 2:
                raise BotorchTensorDimensionError("Must specify at least two outcomes for MOO.")
            if any(i < 0 for i in outcomes):
                if num_outcomes is None:
                    raise BotorchError("num_outcomes is required if any outcomes are less than 0.")
                outcomes = _normalize_indices(outcomes, num_outcomes)
            self._outcomes = outcomes

    @property
    def outcomes(self) -> list[int]:
        """The selected outcomes (only defined if ``outcomes`` were given)."""
        if self._outcomes is None:
            raise AttributeError(
                "'IdentityMCMultiOutputObjective' object has no attribute 'outcomes'"
            )
        return list(self._outcomes)

    def forward(self, samples: Any, X: Any = None) -> Any:
        if self._outcomes is not None:
            return _take_last(samples, self._outcomes)
        return samples

    def _mo_spec(self) -> tuple[Any, ...]:
        return ("identity", self._outcomes)

    def __repr__(self) -> str:
        return f"{type(self).__name__}(outcomes={self._outcomes})"


class WeightedMCMultiOutputObjective(IdentityMCMultiOutputObjective):
    r"""Objective that reweights the (selected) samples by a weight vector,
    e.g. ``-1`` to minimize an output.

    Example:
        >>> weights = torch.tensor([1.0, -1.0], dtype=torch.float64)
        >>> weighted_objective = WeightedMCMultiOutputObjective(weights)
        >>> samples = sampler(posterior)
        >>> objective = weighted_objective(samples)
    """

    def __init__(
        self, weights: Any, outcomes: list[int] | None = None, num_outcomes: int | None = None
    ) -> None:
        r"""Initialize Objective.

        Args:
            weights: ``m'``-dim array of outcome weights.
            outcomes: A list of the ``m'`` indices that the weights should be
                applied to.
            num_outcomes: the total number of outcomes ``m``.
        """
        super().__init__(outcomes=outcomes, num_outcomes=num_outcomes)
        data, shape = to_flat(weights, "weights")
        if len(shape) != 1:
            shape_s = tuple(weights.shape) if hasattr(weights, "shape") else tuple(shape)
            raise BotorchTensorDimensionError(f"weights must be an 1-D tensor, but got {shape_s}.")
        if outcomes is not None and shape[0] != len(outcomes):
            raise BotorchTensorDimensionError(
                "weights must contain the same number of elements as outcomes, "
                f"but got {len(data)} weights and {len(outcomes)} outcomes."
            )
        self.weights = weights
        self._weights = [float(w) for w in data]

    def forward(self, samples: Any, X: Any = None) -> Any:
        return _scale_last(super().forward(samples), self._weights)

    def _mo_spec(self) -> tuple[Any, ...]:
        return ("weighted", list(self._weights), self._outcomes)

    def __repr__(self) -> str:
        return f"{type(self).__name__}(weights={self._weights}, outcomes={self._outcomes})"


class FeasibilityWeightedMCMultiOutputObjective(MCMultiOutputObjective):
    r"""Feasibility-weighted objective: outputs ``constraint_idcs`` are
    constraints (feasible iff non-negative); the other outputs are weighted
    with BoTorch's ``apply_constraints`` (infeasible cost ``M`` of each
    output at ``X_baseline``, sigmoid temperature ``1e-3``) before the
    optional ``objective`` is applied. Without constraints, the samples are
    passed to the objective unchanged.

    Every constraint uses its own output (BoTorch main evaluates the last
    constraint index for every constraint, a late-binding bug fixed in
    meta-pytorch/botorch#3392).
    """

    def __init__(
        self,
        model: Any,
        X_baseline: Any,
        constraint_idcs: list[int],
        objective: MCMultiOutputObjective | None = None,
    ) -> None:
        r"""Construct a feasibility-weighted objective.

        Args:
            model: A fitted botorch_rs model.
            X_baseline: An ``n x d``-dim array of points already observed.
            constraint_idcs: The outcome indices of the constraints. A
                positive constraint outcome implies feasibility.
            objective: An optional objective to apply after
                feasibility-weighting the samples.
        """
        num_outputs = model.num_outputs
        idcs = [num_outputs + int(i) if int(i) < 0 else int(i) for i in constraint_idcs]
        if len(idcs) != len(set(idcs)):
            raise ValueError("Received duplicate entries for `constraint_idcs`.")
        self.constraint_idcs = idcs
        self.objective_idcs = [i for i in range(num_outputs) if i not in idcs]
        spec = None if objective is None else mo_objective_arg(objective)
        self._rs = _rs.MultiOutputObjective.feasibility_weighted(
            model._acquisition_model(None), to_flat(X_baseline, "X_baseline"), idcs, spec
        )
        self.objective = objective
        if objective is not None:
            self._verify_output_shape = objective._verify_output_shape
        self._kind = kind_of(X_baseline)

    @property
    def infeasible_cost(self) -> Any:
        """The infeasible cost ``M`` of every objective output (empty without
        constraints)."""
        cost = self._rs.infeasible_cost
        return from_flat(cost, (len(cost),), self._kind)

    def _apply_feasibility_weights(self, Y: Any) -> Any:
        if not self.constraint_idcs:
            return Y
        if not is_torch_tensor(Y):
            raise UnsupportedError(
                "FeasibilityWeightedMCMultiOutputObjective can be called on torch tensors only."
            )
        import torch

        cost = torch.as_tensor(self._rs.infeasible_cost, dtype=Y.dtype, device=Y.device)
        obj = Y[..., self.objective_idcs]
        log_w = torch.zeros_like(Y[..., 0])
        for i in self.constraint_idcs:
            log_w = log_w + torch.nn.functional.logsigmoid(Y[..., i] / FEASIBILITY_WEIGHTING_ETA)
        return obj.add(cost).clamp_min(0).mul(log_w.exp().unsqueeze(-1)).add(-cost)

    def forward(self, samples: Any, X: Any = None) -> Any:
        Y = self._apply_feasibility_weights(samples)
        if self.objective is None:
            return Y
        return self.objective(Y, X=X)

    def _mo_spec(self) -> tuple[Any, ...]:
        return ("native", self._rs)


def mo_objective_arg(objective: Any) -> tuple[Any, ...] | None:
    """The specification of a multi-output objective for the extension:
    ``None`` (the identity), ``("identity", outcomes)``, ``("weighted",
    weights, outcomes)``, ``("callable", f)`` or ``("native", obj)``.
    BoTorch's own objectives are converted (identity and weighted objectives
    are evaluated in Rust, others are called)."""
    if objective is None:
        return None
    spec: tuple[Any, ...]
    if isinstance(objective, MCMultiOutputObjective):
        spec = objective._mo_spec()
    else:
        names = _botorch_class_names(objective)
        if "MCMultiOutputObjective" not in names:
            raise UnsupportedError(
                "Only objectives of type MCMultiOutputObjective are supported for "
                "Multi-Objective MC acquisition functions."
            )
        name = type(objective).__name__
        outcomes = getattr(objective, "outcomes", None)
        outcomes = None if outcomes is None else [int(i) for i in outcomes.tolist()]
        if name == "IdentityMCMultiOutputObjective" and name in names:
            spec = ("identity", outcomes)
        elif name == "WeightedMCMultiOutputObjective" and name in names:
            spec = (
                "weighted",
                [float(w) for w in objective.weights.reshape(-1).tolist()],
                outcomes,
            )
        else:
            spec = ("callable", objective)
    if spec[0] == "callable":
        from botorch_rs._torch_bridge import require_torch

        require_torch()
    return spec


def is_mo_objective(objective: Any) -> bool:
    """Whether ``objective`` is a multi-output objective (of botorch_rs or
    BoTorch)."""
    if isinstance(objective, MCMultiOutputObjective):
        return True
    return "MCMultiOutputObjective" in _botorch_class_names(objective)


__all__ = [
    "FeasibilityWeightedMCMultiOutputObjective",
    "GenericMCMultiOutputObjective",
    "IdentityMCMultiOutputObjective",
    "MCMultiOutputObjective",
    "WeightedMCMultiOutputObjective",
]

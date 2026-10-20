"""Cost models for multi-fidelity and cost-aware optimization
(``botorch.models.cost``), evaluated in Rust."""

from __future__ import annotations

import sys
from typing import Any

from botorch_rs import _botorch_rs as _rs
from botorch_rs._arrays import (
    LIST,
    ArrayKind,
    from_flat,
    from_pair,
    is_torch_tensor,
    kind_of,
    to_flat,
)
from botorch_rs.exceptions.errors import BotorchTensorDimensionError
from botorch_rs.models.model import Model


class _CostModelValue:
    """Autograd of a cost model's values for torch inputs that require
    gradients (the gradient is computed in Rust)."""

    @staticmethod
    def apply(rs: Any, X: Any) -> Any:
        import torch

        class _Fn(torch.autograd.Function):
            @staticmethod
            def forward(ctx: Any, Xt: Any) -> Any:  # type: ignore[override]
                kind = kind_of(Xt)
                ctx.save_for_backward(Xt)
                return from_pair(rs.forward(to_flat(Xt)), kind)

            @staticmethod
            def backward(ctx: Any, grad_output: Any) -> Any:  # type: ignore[override]
                (Xt,) = ctx.saved_tensors
                kind = kind_of(Xt)
                g = rs.backward(to_flat(Xt), to_flat(grad_output.contiguous()))
                return from_pair(g, kind)

        return _Fn.apply(X)


class DeterministicCostModel(Model):
    """Base class of the deterministic cost models: ``model(X)`` maps a
    ``batch_shape x q x d`` candidate set to its ``batch_shape x q x m``
    costs (connected to torch's autograd graph for inputs that require
    gradients)."""

    _rs_cost: Any

    def forward(self, X: Any) -> Any:
        flat = to_flat(X, "X")
        if len(flat[1]) < 2:
            raise BotorchTensorDimensionError(
                f"Expected X to be `batch_shape x q x d`, got shape {flat[1]}."
            )
        if is_torch_tensor(X):
            import torch

            if X.requires_grad and torch.is_grad_enabled():
                return _CostModelValue.apply(self._rs_cost, X)
        return from_pair(self._rs_cost.forward(flat), kind_of(X))

    __call__ = forward

    @property
    def num_outputs(self) -> int:
        return int(self._rs_cost.num_outputs)

    def posterior(self, X: Any, output_indices: list[int] | None = None, **kwargs: Any) -> Any:
        """A deterministic posterior: ``mean`` holds the costs (of the outputs
        ``output_indices``), ``variance`` zeros."""
        flat = to_flat(X, "X")
        data, shape = self._rs_cost.forward(flat)
        m = shape[-1]
        idx = list(range(m)) if output_indices is None else [int(i) % m for i in output_indices]
        rows = len(data) // max(m, 1)
        values = [data[r * m + i] for r in range(rows) for i in idx]
        out_shape = (*shape[:-1], len(idx))
        kind = kind_of(X)
        return _DeterministicPosterior(
            from_flat(values, out_shape, kind), from_flat([0.0] * len(values), out_shape, kind)
        )


class _DeterministicPosterior:
    """The posterior of a deterministic model (``mean`` and zero
    ``variance``)."""

    def __init__(self, mean: Any, variance: Any) -> None:
        self.mean = mean
        self.variance = variance


class AffineFidelityCostModel(DeterministicCostModel):
    r"""Deterministic, affine cost model operating on fidelity parameters:
    the cost of every point of a candidate set is
    ``fixed_cost + sum_j weights[j] * X[..., fidelity_dims[j]]``.

    The fidelity dimensions are the sorted keys of ``fidelity_weights``
    (default ``{-1: 1.0}``: the last column with weight 1). BoTorch stores the
    weights in torch's default dtype (float32 unless changed); botorch_rs uses
    them as exact float64 values.

    Example:
        >>> cost_model = AffineFidelityCostModel(fidelity_weights={6: 1.0}, fixed_cost=5.0)
        >>> cost_aware_utility = InverseCostWeightedUtility(cost_model=cost_model)
    """

    def __init__(
        self,
        fidelity_weights: dict[int, float] | None = None,
        fixed_cost: float = 0.01,
    ) -> None:
        if fidelity_weights is None:
            fidelity_weights = {-1: 1.0}
        self.fidelity_dims = sorted(int(k) for k in fidelity_weights)
        self.fixed_cost = fixed_cost
        self._weights = [float(fidelity_weights[k]) for k in self.fidelity_dims]
        self._rs_cost = _rs.CostModel.affine(
            [(k, w) for k, w in zip(self.fidelity_dims, self._weights, strict=True)],
            float(fixed_cost),
        )

    @property
    def weights(self) -> Any:
        """Weights in the constructor's default floating dtype."""
        torch = sys.modules.get("torch")
        kind = (
            LIST
            if torch is None
            else ArrayKind("torch", getattr(torch, self._buffer_factory_dtype))
        )
        return from_flat(self._weights, (len(self._weights),), kind)

    def __repr__(self) -> str:
        return (
            f"AffineFidelityCostModel(fidelity_dims={self.fidelity_dims}, "
            f"fixed_cost={self.fixed_cost})"
        )


class FixedCostModel(DeterministicCostModel):
    r"""Deterministic, fixed cost model: every point of a candidate set
    costs ``fixed_cost`` (one cost per output, e.g. per objective of a
    decoupled multi-objective problem)."""

    def __init__(self, fixed_cost: Any) -> None:
        data, shape = to_flat(fixed_cost, "fixed_cost")
        if len(shape) != 1:
            raise BotorchTensorDimensionError(
                f"Expected a 1-dimensional `fixed_cost`, got shape {shape}."
            )
        self._kind = kind_of(fixed_cost)
        self._fixed = [float(v) for v in data]
        self._rs_cost = _rs.CostModel.fixed(self._fixed)

    @property
    def fixed_cost(self) -> Any:
        return from_flat(self._fixed, (len(self._fixed),), self._kind)

    def __repr__(self) -> str:
        return f"FixedCostModel(fixed_cost={self._fixed})"


def cost_model_handle(cost_model: Any) -> Any:
    """The Rust handle of a cost model: a botorch_rs cost model, the
    posterior mean of a botorch_rs GP, or any other model called on torch
    tensors (e.g. BoTorch's ``GenericDeterministicModel``)."""
    from botorch_rs.models.gp_regression import SingleTaskGP
    from botorch_rs.models.model_list_gp_regression import ModelListGP

    if isinstance(cost_model, DeterministicCostModel):
        return cost_model._rs_cost
    if isinstance(cost_model, (SingleTaskGP, ModelListGP)):
        return _rs.CostModel.posterior_mean(cost_model._acquisition_model(None))
    num_outputs = int(getattr(cost_model, "num_outputs", 1))
    return _rs.CostModel.callable(cost_model, num_outputs)


__all__ = ["AffineFidelityCostModel", "DeterministicCostModel", "FixedCostModel"]

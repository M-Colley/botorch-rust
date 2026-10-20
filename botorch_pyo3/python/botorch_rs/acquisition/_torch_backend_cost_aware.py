"""Device-resident cost weighting, including stochastic GP cost graphs."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from botorch_rs.acquisition.cost_aware import InverseCostWeightedUtility
from botorch_rs.exceptions.errors import BotorchTensorDimensionError, UnsupportedError


def tensor_inverse_cost(utility: Any, X: Any, deltas: Any, sampler: Any, mask: Any) -> Any:
    if not isinstance(X, torch.Tensor):
        X = torch.as_tensor(X, device=deltas.device, dtype=deltas.dtype)
    if not isinstance(deltas, torch.Tensor):
        deltas = torch.as_tensor(deltas, device=X.device, dtype=X.dtype)
    if X.ndim < 2:
        raise BotorchTensorDimensionError("Expected X with shape batch_shape x q x d.")
    if X.device != deltas.device:
        raise ValueError("X and deltas must share a device.")
    model = utility.cost_model
    if X.device.type != "cpu" and getattr(model, "_tensor_backend", None) != "torch":
        from botorch_rs.models.cost import DeterministicCostModel
        from botorch_rs.models.deterministic import GenericDeterministicModel

        if (
            not isinstance(model, (DeterministicCostModel, GenericDeterministicModel, nn.Module))
            and hasattr(model, "posterior")
            and getattr(getattr(model, "model", None), "_tensor_backend", None) != "torch"
        ):
            raise UnsupportedError(
                "GPU cost weighting requires a device-resident cost model; construct a GP with backend='torch'."
            )
    output_indices = None
    num_outputs = int(getattr(model, "num_outputs", 1))
    if mask is not None:
        mask = torch.as_tensor(mask, device=X.device)
        if mask.dtype != torch.bool or mask.shape != (X.shape[-2], num_outputs):
            raise BotorchTensorDimensionError(
                "X_evaluation_mask must be a q x num_outputs boolean Tensor."
            )
        if not torch.equal(mask, mask[:1].expand_as(mask)):
            raise NotImplementedError(
                "Currently, all candidates must be evaluated on the same outputs."
            )
        output_indices = mask[0].nonzero(as_tuple=True)[0]
        if not output_indices.numel():
            raise ValueError(
                "Costs must be strictly positive; the evaluation mask selects no outputs."
            )
    method = getattr(model, "posterior", None)
    if method is None:
        from botorch_rs.models._torch_backend_cost import TorchDeterministicCostPosterior

        if not callable(model):
            raise TypeError("A cost model must expose posterior or be callable.")
        values = model(X)
        if not isinstance(values, torch.Tensor) or values.shape != X.shape[:-1] + (num_outputs,):
            raise BotorchTensorDimensionError(
                "A callable cost must return batch_shape x q x num_outputs Tensor values."
            )
        if output_indices is not None:
            values = values.index_select(-1, output_indices)
        posterior = TorchDeterministicCostPosterior(values)
    else:
        posterior = method(X, output_indices=output_indices)
        from botorch_rs.models.deterministic import DeterministicPosterior

        if isinstance(posterior, DeterministicPosterior):
            from botorch_rs.models._torch_backend_cost import TorchDeterministicCostPosterior

            posterior = TorchDeterministicCostPosterior(posterior.mean)
    cost = posterior.mean if utility._use_mean else sampler(posterior)
    if not isinstance(cost, torch.Tensor):
        raise UnsupportedError("A Tensor cost model posterior/sampler must return Tensor costs.")
    if cost.device != X.device:
        raise ValueError("Cost predictions must share X's device; move the cost model with .to(X).")
    cost = utility.cost_objective(cost)
    if not isinstance(cost, torch.Tensor) or cost.ndim < 1 or cost.shape[-1] != X.shape[-2]:
        raise BotorchTensorDimensionError(
            "cost_objective must return one cost per candidate point."
        )
    if not bool((cost > 0).all()):
        raise ValueError("Costs must be strictly positive. Consider clamping cost_objective.")
    cost = cost.sum(dim=-1)
    return (
        deltas - cost.log()
        if utility._log
        else torch.where(deltas > 0, deltas / cost, deltas * cost)
    )


class TorchInverseCostWeightedUtility(nn.Module, InverseCostWeightedUtility):
    _tensor_backend = "torch"

    def __init__(
        self, cost_model: Any, use_mean: bool = True, cost_objective: Any = None, log: bool = False
    ):
        nn.Module.__init__(self)
        InverseCostWeightedUtility.__init__(self, cost_model, use_mean, cost_objective, log)

    def forward(
        self, X: Any, deltas: Any, sampler: Any = None, X_evaluation_mask: Any = None
    ) -> Any:
        return InverseCostWeightedUtility.forward(self, X, deltas, sampler, X_evaluation_mask)

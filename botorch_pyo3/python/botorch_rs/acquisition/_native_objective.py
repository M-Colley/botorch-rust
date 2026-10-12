"""Direct evaluation of the MC objectives that are computed in Rust (the risk
measures): values for any array type, connected to torch's autograd graph
(with the exact vector-Jacobian product from Rust) for torch tensors that
require gradients."""

from __future__ import annotations

import importlib
from typing import Any

from botorch_rs._arrays import from_pair, is_torch_tensor, kind_of, to_flat


def evaluate(rs: Any, samples: Any) -> Any:
    """``rs.forward(samples)`` in the array type of ``samples`` (differentiable
    for torch tensors that require gradients)."""
    if is_torch_tensor(samples) and samples.requires_grad:
        torch = importlib.import_module("torch")
        if torch.is_grad_enabled():
            return importlib.import_module("botorch_rs._risk_autograd").evaluate(rs, samples)
    return from_pair(rs.forward(to_flat(samples, "samples")), kind_of(samples))

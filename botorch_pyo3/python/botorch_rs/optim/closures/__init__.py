"""Closures for optimization (``botorch.optim.closures``)."""

from botorch_rs.optim.closures.core import ForwardBackwardClosure, NdarrayOptimizationClosure
from botorch_rs.optim.closures.model_closures import (
    get_loss_closure,
    get_loss_closure_with_grads,
)

__all__ = [
    "ForwardBackwardClosure",
    "NdarrayOptimizationClosure",
    "get_loss_closure",
    "get_loss_closure_with_grads",
]

"""``InfiniteWidthBNNKernel`` (``botorch.models.kernels.infinite_width_bnn``)."""

from __future__ import annotations

from typing import Any

from botorch_rs import _botorch_rs as _rs
from botorch_rs.constraints import Positive
from botorch_rs.kernels import Kernel, _properties


class InfiniteWidthBNNKernel(Kernel):
    r"""Infinite-width BNN kernel: the covariance of a fully-connected ReLU
    network of depth ``depth`` with i.i.d. priors in the infinite-width limit
    (the arc-cosine recursion of Cho and Saul, 2009), with weight and bias
    variances ``weight_var`` and ``bias_var``.

    Like BoTorch, the arc-cosine argument is clamped to
    ``[-1 + acos_eps, 1 - acos_eps]`` in the kernel matrix, while
    ``diag=True`` (posterior variances) uses the exact recursion of the
    variances.
    """

    _params = (
        ("raw_weight_var", "raw_weight_var_constraint", None),
        ("raw_bias_var", "raw_bias_var_constraint", None),
    )

    def __init__(
        self,
        depth: int = 3,
        batch_shape: Any = None,
        active_dims: Any = None,
        acos_eps: float = 1e-7,
        device: Any = None,
    ) -> None:
        super().__init__(batch_shape=batch_shape, active_dims=active_dims)
        self.depth = int(depth)
        self.acos_eps = float(acos_eps)
        self.raw_weight_var_constraint = Positive()
        self.raw_bias_var_constraint = Positive()

    def _base_node(self) -> _rs.KernelNode:
        return _rs.KernelNode.infinite_width_bnn(self.depth, self.acos_eps)

    raw_weight_var, weight_var = _properties("raw_weight_var")
    raw_bias_var, bias_var = _properties("raw_bias_var")


__all__ = ["InfiniteWidthBNNKernel"]

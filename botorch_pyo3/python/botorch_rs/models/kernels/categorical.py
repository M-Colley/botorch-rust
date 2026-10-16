"""``CategoricalKernel`` (``botorch.models.kernels.categorical``)."""

from __future__ import annotations

from botorch_rs import _botorch_rs as _rs
from botorch_rs.kernels import Kernel


class CategoricalKernel(Kernel):
    r"""A kernel for categorical features: ``exp(-mean_j(d_j / l_j))`` where
    ``d_j`` is 0 if ``x1_j == x2_j`` and 1 otherwise.

    Note: This kernel is NOT differentiable w.r.t. the inputs (its gradient
    with respect to the inputs is zero).
    """

    has_lengthscale = True
    _params = (("raw_lengthscale", "raw_lengthscale_constraint", "lengthscale_prior"),)

    def _base_node(self) -> _rs.KernelNode:
        return _rs.KernelNode.categorical(self.ard_num_dims)


__all__ = ["CategoricalKernel"]

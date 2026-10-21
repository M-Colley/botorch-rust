"""``CategoricalKernel`` (``botorch.models.kernels.categorical``)."""

from __future__ import annotations

from typing import Any

from botorch_rs import _botorch_rs as _rs
from botorch_rs.kernels import Kernel


class CategoricalKernel(Kernel):
    r"""A kernel for categorical features: ``exp(-mean_j(d_j / l_j))`` where
    ``d_j`` is 0 if ``x1_j == x2_j`` and 1 otherwise.

    Note: This kernel is NOT differentiable w.r.t. the inputs (its gradient
    with respect to the inputs is zero).
    """

    def __new__(cls, *args: Any, backend: str = "auto", **kwargs: Any) -> Any:
        from botorch_rs.models.map_saas import _use_torch_backend

        device = kwargs.get("device")
        if cls is CategoricalKernel and (
            _use_torch_backend(backend, *args, *kwargs.values())
            or (backend == "auto" and device is not None and str(device) != "cpu")
        ):
            from botorch_rs.models._torch_backend_kernel_catalogue import TorchCategoricalKernel

            return object.__new__(TorchCategoricalKernel)
        return object.__new__(cls)

    def __init__(self, *args: Any, backend: str = "auto", **kwargs: Any) -> None:
        Kernel.__init__(self, *args, **kwargs)

    has_lengthscale = True
    _params = (("raw_lengthscale", "raw_lengthscale_constraint", "lengthscale_prior"),)

    def _base_node(self) -> _rs.KernelNode:
        return _rs.KernelNode.categorical(self.ard_num_dims)


__all__ = ["CategoricalKernel"]

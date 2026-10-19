"""Covariance of repeatedly integrated white noise on scalar time inputs."""

from __future__ import annotations

from typing import Any

from botorch_rs import _botorch_rs as _rs
from botorch_rs.kernels import Kernel


class IntegratedWhiteNoiseKernel(Kernel):
    r"""White noise integrated ``order`` times (``order=1`` is Brownian motion).

    Inputs are scalar, nonnegative times; ``active_dims`` selects a time
    column from higher-dimensional data. The kernel has no parameters.
    Wrap it in ``ScaleKernel`` to learn a covariance scale.

    ``diag=True`` returns the diagonal of the actual cross-covariance when
    two different input sets are supplied. Official BoTorch revision
    b0dd01be6 instead ignores the second input set in this case.
    """

    def __init__(self, order: int = 1, **kwargs: Any) -> None:
        if isinstance(order, bool) or not isinstance(order, int) or order < 1:
            raise ValueError(
                "IntegratedWhiteNoiseKernel requires an integer order >= 1, "
                f"but got order={order!r}."
            )
        super().__init__(**kwargs)
        self.order = order

    def _base_node(self) -> _rs.KernelNode:
        return _rs.KernelNode.integrated_white_noise(self.order)


__all__ = ["IntegratedWhiteNoiseKernel"]

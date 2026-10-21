"""Empirical GP with the full historical cross-output covariance."""

from __future__ import annotations

from typing import Any

from botorch_rs import _ops
from botorch_rs.models.empirical_gps.empirical_1d_gp import (
    EmpiricalOneDimensionalGP,
    EmpiricalOneDimensionalKernel,
    EmpiricalOneDimensionalMean,
)
from botorch_rs.models.empirical_gps.utils import _moveaxis, _reshape


class MultiOutputEmpiricalOneDimensionalMean(EmpiricalOneDimensionalMean):
    """Historical mean flattened in point-major interleaved order."""

    def forward(self, x: Any) -> Any:
        result = super().forward(x)
        if self.num_outputs > 1:
            result = _moveaxis(result, -2, -1)
            shape = _ops.shape_of(result)
            return _reshape(result, (*shape[:-2], shape[-2] * shape[-1]))
        return result

    __call__ = forward


class MultiOutputEmpiricalOneDimensionalKernel(EmpiricalOneDimensionalKernel):
    """Covariance over every point/output pair, including cross-output blocks."""

    _vectorize_outputs = True


class MultiOutputEmpiricalOneDimensionalGP(EmpiricalOneDimensionalGP):
    """Exact GP over interleaved output observations with a shared likelihood."""

    _mean_class = MultiOutputEmpiricalOneDimensionalMean
    _kernel_class = MultiOutputEmpiricalOneDimensionalKernel
    _correlated = True


__all__ = [
    "MultiOutputEmpiricalOneDimensionalMean",
    "MultiOutputEmpiricalOneDimensionalKernel",
    "MultiOutputEmpiricalOneDimensionalGP",
]

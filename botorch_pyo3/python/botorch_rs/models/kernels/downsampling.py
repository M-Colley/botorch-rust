"""``DownsamplingKernel`` (``botorch.models.kernels.downsampling``)."""

from __future__ import annotations

from typing import Any

from botorch_rs import _botorch_rs as _rs
from botorch_rs.constraints import Interval, Positive
from botorch_rs.kernels import Kernel, _check_constraint, _check_prior, _properties
from botorch_rs.priors import Prior


class DownsamplingKernel(Kernel):
    r"""Downsampling kernel ``c + (1 - x_1)^(1 + delta) (1 - x_2)^(1 + delta)``
    (for ``d = 1``; for several inputs the terms are summed), with offset ``c``
    and power ``delta``.

    For ``d > 1`` BoTorch's ``diag=True`` mode computes
    ``c + (sum_j (1 - x_j)^2)^(1 + delta)``, which is not the diagonal of the
    kernel matrix; botorch_rs uses the diagonal of the kernel matrix.
    """

    _params = (
        ("raw_power", "raw_power_constraint", "power_prior"),
        ("raw_offset", "raw_offset_constraint", "offset_prior"),
    )

    def __init__(
        self,
        power_prior: Prior | None = None,
        offset_prior: Prior | None = None,
        power_constraint: Interval | None = None,
        offset_constraint: Interval | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        for prior, name in ((power_prior, "power_prior"), (offset_prior, "offset_prior")):
            _check_prior(prior, name)
        _check_constraint(power_constraint, "power_constraint")
        _check_constraint(offset_constraint, "offset_constraint")
        self.power_prior = power_prior
        self.offset_prior = offset_prior
        self.raw_power_constraint = power_constraint if power_constraint is not None else Positive()
        self.raw_offset_constraint = (
            offset_constraint if offset_constraint is not None else Positive()
        )

    def _base_node(self) -> _rs.KernelNode:
        return _rs.KernelNode.downsampling()

    raw_power, power = _properties("raw_power")
    raw_offset, offset = _properties("raw_offset")


__all__ = ["DownsamplingKernel"]

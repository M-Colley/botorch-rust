"""``ExponentialDecayKernel`` (``botorch.models.kernels.exponential_decay``)."""

from __future__ import annotations

from typing import Any

from botorch_rs import _botorch_rs as _rs
from botorch_rs.constraints import Interval, Positive
from botorch_rs.kernels import Kernel, _check_constraint, _check_prior, _properties
from botorch_rs.priors import Prior


class ExponentialDecayKernel(Kernel):
    r"""Exponential decay kernel ``w + beta^alpha / (x_1 + x_2 + beta)^alpha``,
    computed as ``w + (1 + ||(x_1 + x_2) / l||)^(-alpha)`` with offset ``w``,
    lengthscale ``l = beta`` and power ``alpha``.

    BoTorch registers ``offset_constraint`` for the power (``power_constraint``
    is ignored); botorch_rs constrains the power with ``power_constraint``.
    """

    has_lengthscale = True
    _params = (
        ("raw_lengthscale", "raw_lengthscale_constraint", "lengthscale_prior"),
        ("raw_power", "raw_power_constraint", "power_prior"),
        ("raw_offset", "raw_offset_constraint", "offset_prior"),
    )

    def __new__(cls, *args: Any, backend: str = "auto", **kwargs: Any) -> Any:
        from botorch_rs.models.map_saas import _use_torch_backend

        device = kwargs.get("device")
        if cls is ExponentialDecayKernel and (
            _use_torch_backend(backend, *args, *kwargs.values())
            or (backend == "auto" and device is not None and str(device) != "cpu")
        ):
            from botorch_rs.models._torch_backend_kernel_catalogue import (
                TorchExponentialDecayKernel,
            )

            return object.__new__(TorchExponentialDecayKernel)
        return object.__new__(cls)

    def __init__(
        self,
        power_prior: Prior | None = None,
        offset_prior: Prior | None = None,
        power_constraint: Interval | None = None,
        offset_constraint: Interval | None = None,
        *,
        backend: str = "auto",
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
        return _rs.KernelNode.exponential_decay(self.ard_num_dims)

    raw_power, power = _properties("raw_power")
    raw_offset, offset = _properties("raw_offset")


__all__ = ["ExponentialDecayKernel"]

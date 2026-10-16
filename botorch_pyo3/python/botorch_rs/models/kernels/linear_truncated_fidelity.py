"""``LinearTruncatedFidelityKernel``
(``botorch.models.kernels.linear_truncated_fidelity``)."""

from __future__ import annotations

from typing import Any

from botorch_rs import _botorch_rs as _rs
from botorch_rs.constraints import Interval, Positive
from botorch_rs.exceptions.errors import UnsupportedError
from botorch_rs.kernels import Kernel, MaternKernel, _check_constraint, _check_prior, _properties
from botorch_rs.priors import GammaPrior, Prior


class LinearTruncatedFidelityKernel(Kernel):
    r"""Linear truncated fidelity kernel for one or two fidelity parameters:

    ``K(x_1, x_2) = k_0 + c_1(x_1, x_2) k_1 + c_2(x_1, x_2) k_2 + c_3(x_1, x_2) k_3``

    where ``k_0`` (``covar_module_unbiased``) and ``k_i`` (``i > 0``,
    ``covar_module_biased``) are kernels on the non-fidelity inputs (Matern
    kernels with ``GammaPrior(3, 6)`` and ``GammaPrior(6, 2)`` lengthscale
    priors by default) and the ``c_i`` the polynomial bias terms of the
    fidelities (clamped to ``[0, 1]``) with power ``power``.
    """

    _params = (("raw_power", "raw_power_constraint", "power_prior"),)

    def __init__(
        self,
        fidelity_dims: list[int],
        dimension: int | None = None,
        power_prior: Prior | None = None,
        power_constraint: Interval | None = None,
        nu: float = 2.5,
        lengthscale_prior_unbiased: Prior | None = None,
        lengthscale_prior_biased: Prior | None = None,
        lengthscale_constraint_unbiased: Interval | None = None,
        lengthscale_constraint_biased: Interval | None = None,
        covar_module_unbiased: Kernel | None = None,
        covar_module_biased: Kernel | None = None,
        **kwargs: Any,
    ) -> None:
        if dimension is None and kwargs.get("active_dims") is None:
            raise UnsupportedError("Must specify dimension when not specifying active_dims.")
        n_fidelity = len(fidelity_dims)
        if len(set(fidelity_dims)) != n_fidelity:
            raise ValueError("fidelity_dims must not have repeated elements")
        if n_fidelity not in {1, 2}:
            raise UnsupportedError(
                "LinearTruncatedFidelityKernel accepts either one or twofidelity parameters."
            )
        if nu not in {0.5, 1.5, 2.5}:
            raise ValueError("nu must be one of 0.5, 1.5, or 2.5")
        super().__init__(**kwargs)
        self.fidelity_dims = [int(f) for f in fidelity_dims]
        _check_prior(power_prior, "power_prior")
        _check_constraint(power_constraint, "power_constraint")
        self.power_prior = power_prior
        self.raw_power_constraint = power_constraint if power_constraint is not None else Positive()
        if lengthscale_prior_unbiased is None:
            lengthscale_prior_unbiased = GammaPrior(3, 6)
        if lengthscale_prior_biased is None:
            lengthscale_prior_biased = GammaPrior(6, 2)
        if self.active_dims is not None:
            dimension = len(self.active_dims)
        assert dimension is not None
        if covar_module_unbiased is None:
            covar_module_unbiased = MaternKernel(
                nu=nu,
                batch_shape=self._batch_shape,
                lengthscale_prior=lengthscale_prior_unbiased,
                ard_num_dims=dimension - n_fidelity,
                lengthscale_constraint=lengthscale_constraint_unbiased,
            )
        if covar_module_biased is None:
            covar_module_biased = MaternKernel(
                nu=nu,
                batch_shape=self._batch_shape,
                lengthscale_prior=lengthscale_prior_biased,
                ard_num_dims=dimension - n_fidelity,
                lengthscale_constraint=lengthscale_constraint_biased,
            )
        self.covar_module_unbiased = covar_module_unbiased
        self.covar_module_biased = covar_module_biased

    def _children(self) -> list[tuple[str, Kernel]]:
        return [
            ("covar_module_unbiased", self.covar_module_unbiased),
            ("covar_module_biased", self.covar_module_biased),
        ]

    def _base_node(self) -> _rs.KernelNode:
        return _rs.KernelNode.linear_truncated_fidelity(
            self.fidelity_dims,
            self.covar_module_unbiased._node(),
            self.covar_module_biased._node(),
        )

    raw_power, power = _properties("raw_power")


__all__ = ["LinearTruncatedFidelityKernel"]

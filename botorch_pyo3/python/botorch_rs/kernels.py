"""Covariance modules (``gpytorch.kernels``): ARD ``RBFKernel`` and
``MaternKernel(nu=2.5)``, optionally wrapped in a ``ScaleKernel``.

A kernel passed to a model as ``covar_module`` is bound to it: ``lengthscale``
and ``outputscale`` then read and set the model's hyperparameters (shapes
``batch_shape x 1 x d`` and ``batch_shape`` as in GPyTorch, where
``batch_shape`` is the model's ``_aug_batch_shape``).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from botorch_rs import _botorch_rs as _rs
from botorch_rs._arrays import to_flat
from botorch_rs._module import GPModule
from botorch_rs.constraints import Interval, Positive
from botorch_rs.constraints import _from_rs as _constraint_from_rs
from botorch_rs.exceptions.errors import BotorchTensorDimensionError, UnsupportedError
from botorch_rs.priors import Prior
from botorch_rs.priors import _from_rs as _prior_from_rs


def _check_batch_shape(batch_shape: Any) -> tuple[int, ...]:
    """A module's ``batch_shape`` as a tuple of ints."""
    if batch_shape is None:
        return ()
    try:
        return tuple(int(b) for b in batch_shape)
    except TypeError as e:
        raise UnsupportedError(f"Invalid batch_shape {batch_shape!r}.") from e


def _check_prior(prior: Any, name: str) -> None:
    if prior is not None and not isinstance(prior, Prior):
        raise UnsupportedError(
            f"`{name}` must be a botorch_rs.priors prior (LogNormalPrior or GammaPrior)."
        )


def _check_constraint(constraint: Any, name: str) -> None:
    if constraint is not None and not isinstance(constraint, Interval):
        raise UnsupportedError(
            f"`{name}` must be a botorch_rs.constraints.GreaterThan, Positive or Interval."
        )


def _check_kwargs(kwargs: dict[str, Any], cls: str) -> None:
    if kwargs:
        raise UnsupportedError(f"Unsupported arguments of {cls}: {sorted(kwargs)}.")


class Kernel(GPModule):
    """Base class of the covariance modules."""

    _preset: Callable[[int], _rs.KernelSpec] | None = None

    def _spec(self, d: int) -> _rs.KernelSpec:  # pragma: no cover - abstract
        raise NotImplementedError


class _StationaryKernel(Kernel):
    _kernel_kind: str = ""

    def __init__(
        self,
        ard_num_dims: int | None = None,
        batch_shape: Any = None,
        active_dims: Any = None,
        lengthscale_prior: Prior | None = None,
        lengthscale_constraint: Interval | None = None,
        eps: float = 1e-6,
        **kwargs: Any,
    ) -> None:
        self.batch_shape = _check_batch_shape(batch_shape)
        if active_dims is not None:
            raise UnsupportedError("`active_dims` is not supported by botorch_rs.")
        _check_kwargs(kwargs, type(self).__name__)
        _check_prior(lengthscale_prior, "lengthscale_prior")
        _check_constraint(lengthscale_constraint, "lengthscale_constraint")
        self.ard_num_dims = ard_num_dims
        self.lengthscale_prior = lengthscale_prior
        self.lengthscale_constraint = (
            lengthscale_constraint if lengthscale_constraint is not None else Positive()
        )
        self._preset = None

    def _spec(self, d: int) -> _rs.KernelSpec:
        ard = self.ard_num_dims
        if ard is None and d != 1:
            raise UnsupportedError(
                "Kernels with a single lengthscale shared by all dimensions are not "
                f"supported; pass ard_num_dims={d}."
            )
        if ard is not None and ard != d:
            raise BotorchTensorDimensionError(
                f"The kernel has ard_num_dims={ard} but the inputs have d={d}."
            )
        if self._preset is not None:
            return self._preset(d)
        prior = self.lengthscale_prior._rs if self.lengthscale_prior is not None else None
        return _rs.KernelSpec(self._kernel_kind, d, prior, self.lengthscale_constraint._rs)

    @property
    def lengthscale(self) -> Any:
        """The lengthscales, shape ``batch_shape x 1 x d``."""
        return self._param_out(self._rs_model.lengthscale, (1, -1))

    @lengthscale.setter
    def lengthscale(self, value: Any) -> None:
        self._rs_model.set_lengthscale(to_flat(value, "lengthscale"))

    @property
    def raw_lengthscale(self) -> Any:
        """The raw lengthscales (BoTorch's parameterization), shape
        ``batch_shape x 1 x d``."""
        return self._param_out(self._model._raw_param("raw_lengthscale"), (1, -1))

    @raw_lengthscale.setter
    def raw_lengthscale(self, value: Any) -> None:
        self._model._set_raw_param("raw_lengthscale", value)

    @property
    def raw_lengthscale_constraint(self) -> Interval:
        return self.lengthscale_constraint

    def __repr__(self) -> str:
        return f"{type(self).__name__}(ard_num_dims={self.ard_num_dims})"


class RBFKernel(_StationaryKernel):
    """Squared-exponential kernel ``exp(-r**2 / 2)`` with ARD lengthscales."""

    _kernel_kind = "rbf"


class MaternKernel(_StationaryKernel):
    """Matern kernel; only ``nu=2.5`` is supported."""

    _kernel_kind = "matern52"

    def __init__(
        self,
        nu: float = 2.5,
        ard_num_dims: int | None = None,
        batch_shape: Any = None,
        active_dims: Any = None,
        lengthscale_prior: Prior | None = None,
        lengthscale_constraint: Interval | None = None,
        eps: float = 1e-6,
        **kwargs: Any,
    ) -> None:
        if nu != 2.5:
            raise UnsupportedError(f"Only nu=2.5 is supported by botorch_rs, got nu={nu}.")
        self.nu = nu
        super().__init__(
            ard_num_dims=ard_num_dims,
            batch_shape=batch_shape,
            active_dims=active_dims,
            lengthscale_prior=lengthscale_prior,
            lengthscale_constraint=lengthscale_constraint,
            eps=eps,
            **kwargs,
        )


class ScaleKernel(Kernel):
    """``outputscale * base_kernel``."""

    def __init__(
        self,
        base_kernel: Kernel,
        outputscale_prior: Prior | None = None,
        outputscale_constraint: Interval | None = None,
        **kwargs: Any,
    ) -> None:
        if not isinstance(base_kernel, _StationaryKernel):
            raise UnsupportedError("ScaleKernel supports an RBFKernel or MaternKernel base kernel.")
        self.batch_shape = _check_batch_shape(kwargs.pop("batch_shape", None))
        _check_kwargs(kwargs, "ScaleKernel")
        _check_prior(outputscale_prior, "outputscale_prior")
        _check_constraint(outputscale_constraint, "outputscale_constraint")
        self.base_kernel = base_kernel
        self.outputscale_prior = outputscale_prior
        self.outputscale_constraint = (
            outputscale_constraint if outputscale_constraint is not None else Positive()
        )
        self._preset = None

    def _spec(self, d: int) -> _rs.KernelSpec:
        if self._preset is not None:
            self.base_kernel._spec(d)  # validates ard_num_dims
            return self._preset(d)
        prior = self.outputscale_prior._rs if self.outputscale_prior is not None else None
        return self.base_kernel._spec(d).with_scale_kernel(prior, self.outputscale_constraint._rs)

    def _bind(self, model: Any) -> None:
        super()._bind(model)
        self.base_kernel._bind(model)

    @property
    def outputscale(self) -> Any:
        """The outputscale, shape ``batch_shape``."""
        return self._param_out(self._rs_model.outputscale, ())

    @outputscale.setter
    def outputscale(self, value: Any) -> None:
        self._rs_model.set_outputscale(to_flat(value, "outputscale"))

    @property
    def raw_outputscale(self) -> Any:
        return self._param_out(self._model._raw_param("raw_outputscale"), ())

    @raw_outputscale.setter
    def raw_outputscale(self, value: Any) -> None:
        self._model._set_raw_param("raw_outputscale", value)

    @property
    def raw_outputscale_constraint(self) -> Interval:
        return self.outputscale_constraint

    def __repr__(self) -> str:
        return f"ScaleKernel({self.base_kernel!r})"


def _kernel_from_spec(
    spec: _rs.KernelSpec,
    preset: Callable[[int], _rs.KernelSpec],
    batch_shape: tuple[int, ...] = (),
) -> Kernel:
    """A kernel object describing ``spec`` that builds itself with ``preset``."""
    base: _StationaryKernel
    cls = RBFKernel if spec.kind == "rbf" else MaternKernel
    base = cls(
        ard_num_dims=spec.ard_num_dims,
        batch_shape=batch_shape,
        lengthscale_prior=_prior_from_rs(spec.lengthscale_prior),
        lengthscale_constraint=_constraint_from_rs(spec.lengthscale_constraint),
    )
    if not spec.has_scale_kernel:
        base._preset = preset
        return base
    outputscale_constraint = spec.outputscale_constraint
    kernel = ScaleKernel(
        base,
        outputscale_prior=_prior_from_rs(spec.outputscale_prior),
        outputscale_constraint=(
            _constraint_from_rs(outputscale_constraint)
            if outputscale_constraint is not None
            else None
        ),
        batch_shape=batch_shape,
    )
    kernel._preset = preset
    return kernel


__all__ = ["Kernel", "MaternKernel", "RBFKernel", "ScaleKernel"]

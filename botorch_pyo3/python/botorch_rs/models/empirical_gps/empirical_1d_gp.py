"""Empirical learning-curve means, covariances and exact native GPs."""

from __future__ import annotations

import copy
import math
import sys
from typing import Any

from botorch_rs import _botorch_rs as _rs
from botorch_rs import _ops
from botorch_rs._arrays import (
    broadcast_to,
    first_kind,
    from_flat,
    is_torch_tensor,
    kind_of,
    to_flat,
)
from botorch_rs.constraints import GreaterThan
from botorch_rs.exceptions.errors import UnsupportedError
from botorch_rs.kernels import AdditiveKernel, Kernel
from botorch_rs.likelihoods import FixedNoiseGaussianLikelihood, GaussianLikelihood
from botorch_rs.means import Mean
from botorch_rs.models.empirical_gps.utils import (
    _moveaxis,
    _reshape,
    build_basis_interpolant,
    build_mean_interpolant,
    compute_basis_matrix,
    compute_sample_covariance,
    extract_slice_for_interp,
    validate_no_transforms,
)
from botorch_rs.models.gpytorch import BatchedMultiOutputGPyTorchModel, GPyTorchModel
from botorch_rs.models.model import FantasizeMixin
from botorch_rs.posteriors.gpytorch import GPyTorchPosterior


def _use_tensor_backend(values: Any, backend: str) -> bool:
    if backend not in ("auto", "native", "torch"):
        raise ValueError("backend must be 'auto', 'native', or 'torch'.")
    return backend == "torch" or (
        backend == "auto"
        and any(
            getattr(value, "requires_grad", False)
            or getattr(getattr(value, "device", None), "type", "cpu") != "cpu"
            for value in values
        )
    )


class _SharedKernelBinding:
    """Kernel parameters are shared; likelihood output batches are separate."""

    def __init__(self, model: Any) -> None:
        self._owner = model

    @property
    def _rs(self) -> Any:
        return self

    @property
    def aug_batch_shape(self) -> tuple[()]:
        return ()

    def __getattr__(self, name: str) -> Any:
        # deepcopy may inspect this adapter before its fields have been copied.
        if name.startswith("__"):
            raise AttributeError(name)
        owner = self.__dict__.get("_owner")
        if owner is None:
            raise AttributeError(name)
        return getattr(owner, name)


def _stack(values: list[Any], axis: int = 0) -> Any:
    kind = first_kind(*values)
    if kind.lib == "torch":
        import torch

        return torch.stack(values, dim=axis)
    if kind.lib == "numpy":
        import numpy as np

        return np.stack(values, axis=axis)
    shape = _ops.shape_of(values[0])
    result = from_flat([v for a in values for v in to_flat(a)[0]], (len(values), *shape), kind)
    return _moveaxis(result, 0, axis)


def _validate_history(X: Any, Y: Any) -> tuple[int, int, int]:
    xs, ys = _ops.shape_of(X), _ops.shape_of(Y)
    if len(ys) != 3 or ys[0] == 0 or ys[1] < 2 or ys[2] == 0 or xs != (ys[1], 1):
        raise ValueError(
            "Historical X must be p x 1 and Y must be curves x p x outputs, with p >= 2."
        )
    if any(not math.isfinite(v) for v in to_flat(Y)[0]):
        raise ValueError("Historical curves must be finite.")
    if _ops._requires_grad((X, Y)):
        raise UnsupportedError(
            "Historical curve data are fixed; use query and curve-weight derivatives."
        )
    return ys


class EmpiricalOneDimensionalMean(Mean):
    """Piecewise-linear interpolation of the historical sample mean."""

    def __new__(
        cls,
        X_full: Any = None,
        Y_full: Any = None,
        *args: Any,
        backend: str = "auto",
        **kwargs: Any,
    ) -> Any:
        if _use_tensor_backend((X_full, Y_full), backend):
            from botorch_rs.models.empirical_gps._torch_backend import (
                TensorCorrelatedMean,
                TensorEmpiricalMean,
            )
            from botorch_rs.models.empirical_gps.multioutput_empirical_1d_gp import (
                MultiOutputEmpiricalOneDimensionalMean,
            )

            if cls is MultiOutputEmpiricalOneDimensionalMean:
                return object.__new__(TensorCorrelatedMean)
            if cls is EmpiricalOneDimensionalMean:
                return object.__new__(TensorEmpiricalMean)
        return object.__new__(cls)

    def __init__(self, X_full: Any, Y_full: Any, *, backend: str = "auto") -> None:
        _validate_history(X_full, Y_full)
        self.f, self.num_outputs, self.mean_full = build_mean_interpolant(X_full, Y_full)

    def forward(self, x: Any) -> Any:
        x = extract_slice_for_interp(x, self.num_outputs)
        result = _moveaxis(self.f(x), 0, -2)
        return _ops.getitem(result, (..., 0, slice(None))) if self.num_outputs == 1 else result

    __call__ = forward

    def _mean_node(self) -> Any:
        knots = list(to_flat(self.f._x)[0])
        tables = [
            to_flat(_reshape(_ops.getitem(self.f._y, i), (1, len(knots))))
            for i in range(self.num_outputs)
        ]
        return _rs.KernelNode.empirical_basis(
            [knots] * self.num_outputs,
            tables,
            [1.0],
            False,
            1.0,
            None if self.num_outputs == 1 else 1,
            False,
        )


class EmpiricalOneDimensionalKernel(Kernel):
    """Native covariance of interpolated centered historical curves.

    Outputs are independent. ``curve_weights`` are shared across outputs and
    are direct nonnegative parameters when ``ard=True``.
    """

    _vectorize_outputs = False

    def __new__(
        cls,
        X_full: Any = None,
        Y_full: Any = None,
        *args: Any,
        backend: str = "auto",
        **kwargs: Any,
    ) -> Any:
        if _use_tensor_backend((X_full, Y_full), backend):
            from botorch_rs.models.empirical_gps._torch_backend import (
                TensorCorrelatedKernel,
                TensorEmpiricalKernel,
            )
            from botorch_rs.models.empirical_gps.multioutput_empirical_1d_gp import (
                MultiOutputEmpiricalOneDimensionalKernel,
            )

            if cls is MultiOutputEmpiricalOneDimensionalKernel:
                return object.__new__(TensorCorrelatedKernel)
            if cls is EmpiricalOneDimensionalKernel:
                return object.__new__(TensorEmpiricalKernel)
        return object.__new__(cls)

    def __init__(
        self,
        X_full: Any,
        Y_full: Any,
        ard: bool = False,
        curve_weights: Any = None,
        use_svd: bool | None = None,
        correction: int = 0,
        *,
        backend: str = "auto",
    ) -> None:
        self.num_curves, _, self.num_outputs = _validate_history(X_full, Y_full)
        if not isinstance(correction, int) or correction >= self.num_curves:
            raise ValueError("correction must be an integer smaller than num_curves.")
        super().__init__()
        self.ard, self.correction = bool(ard), correction
        self.f, self._effective_num_curves, self._use_svd = build_basis_interpolant(
            X_full,
            Y_full,
            ard=ard,
            use_svd=use_svd,
            vectorize_outputs=self._vectorize_outputs,
            method="svd" if self._vectorize_outputs else "eigh",
        )
        self._weight_array = None
        self._extra_params = (("curve_weights", "curve_weights_constraint", None),) if ard else ()
        if ard:
            self.curve_weights_constraint = GreaterThan(0.0, transform=None)
        self.curve_weights = (
            from_flat(
                [1.0] * self._effective_num_curves, (self._effective_num_curves,), kind_of(Y_full)
            )
            if ard and curve_weights is None
            else curve_weights
        )

    @property
    def use_svd(self) -> bool:
        return self._use_svd

    @property
    def curve_weights(self) -> Any:
        if self.ard and self._model is not None:
            return self._get_raw("curve_weights")
        return self._weight_array

    @curve_weights.setter
    def curve_weights(self, value: Any) -> None:
        if not self.ard and self._model is not None:
            raise UnsupportedError("Fixed curve weights cannot be changed after binding a model.")
        if value is not None:
            data, shape = to_flat(value)
            if shape != (self._effective_num_curves,) or any(not math.isfinite(v) for v in data):
                raise ValueError(
                    "curve_weights must be a finite vector with one entry per effective curve."
                )
            if self.ard and any(v < 0.0 for v in data):
                raise ValueError("ARD curve weights must be nonnegative.")
        if self.ard and self._model is not None:
            self._set_raw("curve_weights", value)
        else:
            self._weight_array = value

    def _raw_values(self, name: str) -> list[float]:
        if name == "curve_weights" and self._model is None:
            return list(to_flat(self._weight_array)[0])
        return super()._raw_values(name)

    def _base_node(self) -> Any:
        knots = list(to_flat(self.f._x)[0])
        tables = [to_flat(_ops.getitem(self.f._y, i)) for i in range(self.num_outputs)]
        weights = (
            [1.0] * self._effective_num_curves
            if self._weight_array is None
            else to_flat(self._weight_array)[0]
        )
        return _rs.KernelNode.empirical_basis(
            [knots] * self.num_outputs,
            tables,
            weights,
            self.ard,
            float(self.num_curves - self.correction),
            None if self.num_outputs == 1 else 1,
            not self._vectorize_outputs,
        )

    def _basis(self, x: Any) -> Any:
        query = extract_slice_for_interp(x, self.num_outputs)
        return compute_basis_matrix(self.f, query, self.num_outputs, self.curve_weights)

    def __call__(
        self,
        x1: Any,
        x2: Any = None,
        diag: bool = False,
        last_dim_is_batch: bool = False,
        **params: Any,
    ) -> Any:
        if last_dim_is_batch or params:
            raise UnsupportedError(
                "Empirical kernels do not support last_dim_is_batch or extra arguments."
            )
        a = self._basis(x1)
        b = a if x2 is None or x2 is x1 else self._basis(x2)
        if self._vectorize_outputs:

            def flatten(value: Any) -> Any:
                value = _moveaxis(value, 0, -1)
                shape = _ops.shape_of(value)
                return _reshape(value, (*shape[:-2], shape[-2] * shape[-1]))

            return compute_sample_covariance(
                flatten(a), flatten(b), self.num_curves, diag=diag, correction=self.correction
            )
        result = compute_sample_covariance(
            a, b, self.num_curves, diag=diag, correction=self.correction
        )
        result = _moveaxis(result, 0, -2 if diag else -3)
        if self.num_outputs == 1:
            return _ops.getitem(
                result, (..., 0, slice(None)) if diag else (..., 0, slice(None), slice(None))
            )
        return result

    forward = __call__


class BaseAugmentedEmpiricalKernel(AdditiveKernel):
    """Empirical covariance plus an ordinary native covariance on progression."""

    def __new__(cls, empirical_kernel: Any = None, base_kernel: Any = None) -> Any:
        torch = sys.modules.get("torch")
        if (
            cls is BaseAugmentedEmpiricalKernel
            and torch is not None
            and isinstance(empirical_kernel, torch.nn.Module)
        ):
            from botorch_rs.models.empirical_gps._torch_backend import TensorBaseAugmentedKernel

            return object.__new__(TensorBaseAugmentedKernel)
        return object.__new__(cls)

    def __init__(
        self, empirical_kernel: EmpiricalOneDimensionalKernel, base_kernel: Kernel
    ) -> None:
        if not isinstance(empirical_kernel, EmpiricalOneDimensionalKernel) or not isinstance(
            base_kernel, Kernel
        ):
            raise TypeError("Expected native empirical and base kernels.")
        self.empirical_kernel, self.base_kernel = empirical_kernel, base_kernel
        super().__init__(empirical_kernel, base_kernel)

    def _children(self) -> list[tuple[str, Kernel]]:
        return [("empirical_kernel", self.empirical_kernel), ("base_kernel", self.base_kernel)]

    def _base_node(self) -> Any:
        if getattr(self.empirical_kernel, "_is_task_kernel", False):
            return _rs.KernelNode.additive(
                [self.empirical_kernel._node(), self.base_kernel._node()]
            )
        return _rs.KernelNode.additive(
            [self.empirical_kernel._node(), self.base_kernel._node().with_parent_active_dims([0])]
        )

    def __call__(
        self,
        x1: Any,
        x2: Any = None,
        diag: bool = False,
        last_dim_is_batch: bool = False,
        **kwargs: Any,
    ) -> Any:
        empirical = self.empirical_kernel(x1, x2, diag, last_dim_is_batch, **kwargs)
        if self.empirical_kernel._vectorize_outputs:

            def expand(value: Any) -> Any:
                shape = _ops.shape_of(value)
                m = self.empirical_kernel.num_outputs
                return _reshape(
                    _ops.expand(_reshape(value, (*shape[:-1], 1, 1)), (*shape[:-1], m, 1)),
                    (*shape[:-2], shape[-2] * m, 1),
                )

            x1 = expand(x1)
            x2 = None if x2 is None else expand(x2)
        return _ops.apply(
            "add", empirical, self.base_kernel(x1, x2, diag, last_dim_is_batch, **kwargs)
        )

    forward = __call__


class EmpiricalOneDimensionalGP(GPyTorchModel, FantasizeMixin):
    """Exact native empirical GP with independent outputs and shared curve weights."""

    _mean_class = EmpiricalOneDimensionalMean
    _kernel_class = EmpiricalOneDimensionalKernel
    _correlated = False

    get_batch_dimensions = staticmethod(BatchedMultiOutputGPyTorchModel.get_batch_dimensions)

    def __new__(
        cls,
        train_X: Any = None,
        train_Y: Any = None,
        historical_X: Any = None,
        historical_Y: Any = None,
        *args: Any,
        backend: str = "auto",
        **kwargs: Any,
    ) -> Any:
        torch = sys.modules.get("torch")
        modules = [
            *args[1:],
            *(
                kwargs.get(name)
                for name in ("mean_module", "covar_module", "likelihood", "base_covar_module")
            ),
        ]
        use_tensor = _use_tensor_backend(
            (train_X, train_Y, historical_X, historical_Y, kwargs.get("train_Yvar"), *args[:1]),
            backend,
        )
        if backend == "auto" and torch is not None:
            use_tensor = use_tensor or any(
                isinstance(module, torch.nn.Module) for module in modules
            )
        if use_tensor:
            from botorch_rs.models.empirical_gps._torch_backend import (
                TensorCorrelatedGP,
                TensorEmpiricalGP,
            )
            from botorch_rs.models.empirical_gps.multioutput_empirical_1d_gp import (
                MultiOutputEmpiricalOneDimensionalGP,
            )

            if cls is MultiOutputEmpiricalOneDimensionalGP:
                return object.__new__(TensorCorrelatedGP)
            if cls is EmpiricalOneDimensionalGP:
                return object.__new__(TensorEmpiricalGP)
        return object.__new__(cls)

    def __init__(
        self,
        train_X: Any,
        train_Y: Any,
        historical_X: Any,
        historical_Y: Any,
        train_Yvar: Any = None,
        likelihood: Any = None,
        input_transform: Any = None,
        outcome_transform: Any = None,
        mean_module: Any = None,
        covar_module: Any = None,
        base_covar_module: Any = None,
        ard: bool = False,
        *,
        backend: str = "auto",
    ) -> None:
        validate_no_transforms(input_transform, outcome_transform, type(self).__name__)
        _, _, m = _validate_history(historical_X, historical_Y)
        self._validate_tensor_args(train_X, train_Y, train_Yvar)
        x, xs = to_flat(train_X)
        y, ys = to_flat(train_Y)
        if len(xs) != 2 or xs[-1] != 1 or ys != (xs[0], m) or xs[0] == 0:
            raise UnsupportedError(
                "Empirical GP training requires unbatched n x 1 inputs and n x m targets."
            )
        if train_Yvar is not None and _ops.shape_of(train_Yvar) != ys:
            raise ValueError("train_Yvar must match train_Y.")
        mean_module = (
            self._mean_class(historical_X, historical_Y) if mean_module is None else mean_module
        )
        covar_module = (
            self._kernel_class(historical_X, historical_Y, ard=ard)
            if covar_module is None
            else covar_module
        )
        if not isinstance(mean_module, self._mean_class) or not isinstance(
            covar_module, self._kernel_class
        ):
            raise UnsupportedError(
                "This model requires its corresponding native empirical mean and kernel."
            )
        if covar_module.ard != ard or mean_module.num_outputs != m or covar_module.num_outputs != m:
            raise ValueError("Empirical mean, kernel, outputs and ard configuration must match.")
        if base_covar_module is not None:
            if tuple(base_covar_module.batch_shape):
                raise UnsupportedError("Batched additive base kernels are not yet supported.")
            covar_module = BaseAugmentedEmpiricalKernel(covar_module, base_covar_module)
        self._has_base_kernel = base_covar_module is not None
        self._kind, self._num_outputs = kind_of(train_X), m
        self._aug_batch_shape = (m,) if m > 1 and not self._correlated else ()
        if likelihood is None:
            spec = (
                _rs.GaussianLikelihoodSpec()
                if self._correlated
                else _rs.GaussianLikelihoodSpec.lognormal()
            )
            likelihood = GaussianLikelihood._from_spec(spec, self._aug_batch_shape)
        if not isinstance(likelihood, GaussianLikelihood) or likelihood._model is not None:
            raise UnsupportedError("Expected an unbound native GaussianLikelihood.")
        for module in (mean_module, covar_module):
            if module._model is not None:
                raise UnsupportedError(
                    "Empirical modules must be unbound when constructing a model."
                )
        from botorch_rs.models.higher_order_gp import _initial_raw_noise

        self._train_X, self._train_Y, self._train_Yvar = train_X, train_Y, train_Yvar
        variance = None if train_Yvar is None else to_flat(train_Yvar)[0]
        if self._correlated:
            xx = [
                ([value for p in x for g in range(m) for value in (p, float(g))], (len(x) * m, 2))
            ]
            yy = [(y, (len(x) * m, 1))]
            vv = None if variance is None else [(variance, (len(x) * m, 1))]
        else:
            xx = [([value for p in x for value in (p, float(g))], (len(x), 2)) for g in range(m)]
            yy = [([y[p * m + g] for p in range(len(x))], (len(x), 1)) for g in range(m)]
            vv = (
                None
                if variance is None
                else [([variance[p * m + g] for p in range(len(x))], (len(x), 1)) for g in range(m)]
            )
        self._rs = _rs.EmpiricalGP(
            xx,
            yy,
            vv,
            mean_module._mean_node(),
            covar_module._node(),
            likelihood._spec,
            covar_module._raw_flat(),
            [_initial_raw_noise(likelihood)] * (1 if self._correlated else m),
            self._correlated,
        )
        self._param_shapes = {
            self._public_name(name): tuple(shape)
            for name, shape in covar_module._node().param_shapes("covar_module.")
        }
        if variance is None:
            self._param_shapes["likelihood.noise_covar.raw_noise"] = (*self._aug_batch_shape, 1)
        else:
            likelihood = FixedNoiseGaussianLikelihood._create(self._aug_batch_shape)
        self.likelihood: Any = likelihood
        self.mean_module: Any = mean_module
        self.covar_module: Any = covar_module
        likelihood._bind(self)
        mean_module._bind(self)
        self._shared_kernel_binding = _SharedKernelBinding(self)
        covar_module._bind(self._shared_kernel_binding, "covar_module.")
        self.training = True

    def _offsets(self) -> dict[str, tuple[int, int]]:
        result, start = {}, 0
        for name, count in self._rs.parameter_names():
            result[self._public_name(name)] = start, count
            start += count
        return result

    def _public_name(self, name: str) -> str:
        if self._has_base_kernel:
            return name.replace(
                "covar_module.kernels.0.", "covar_module.empirical_kernel."
            ).replace("covar_module.kernels.1.", "covar_module.base_kernel.")
        return name

    def _param_index(self, suffix: str) -> tuple[str, int, int]:
        for name, (offset, count) in self._offsets().items():
            if name == suffix or name.endswith(suffix):
                return name, offset, count
        raise AttributeError(suffix)

    def _raw_param(self, suffix: str) -> list[float]:
        _, offset, count = self._param_index(suffix)
        return list(self._rs.raw_parameters()[offset : offset + count])

    def _raw_param_shape(self, suffix: str) -> tuple[int, ...]:
        return self._param_shapes[self._param_index(suffix)[0]]

    def _set_raw_param(self, suffix: str, value: Any) -> None:
        name, offset, count = self._param_index(suffix)
        data, shape = to_flat(value)
        values = broadcast_to(data, tuple(shape), self._param_shapes[name])
        raw = self._rs.raw_parameters()
        raw[offset : offset + count] = values
        self._rs.set_raw_parameters(raw)

    def named_parameters(self) -> Any:
        for name, (offset, count) in self._offsets().items():
            yield (
                name,
                from_flat(
                    self._rs.raw_parameters()[offset : offset + count],
                    self._param_shapes[name],
                    self._kind,
                ),
            )

    def parameters(self) -> Any:
        return (value for _, value in self.named_parameters())

    def _param_entries(self) -> list[Any]:
        return [(name, value, *self._offsets()[name]) for name, value in self.named_parameters()]

    @property
    def batch_shape(self) -> tuple[int, ...]:
        return tuple(getattr(self, "_condition_batch_shape", ()))

    @property
    def train_inputs(self) -> tuple[Any]:
        if self._correlated:
            data, shape = to_flat(self._train_X)
            return (
                from_flat(
                    [value for value in data for _ in range(self.num_outputs)],
                    (*shape[:-2], shape[-2] * self.num_outputs, 1),
                    self._kind,
                ),
            )
        if self.num_outputs > 1 and not self._correlated:
            n = _ops.shape_of(self._train_X)[-2]
            return (
                _ops.expand(
                    _ops.getitem(self._train_X, (Ellipsis, None, slice(None), slice(None))),
                    (*self.batch_shape, self.num_outputs, n, 1),
                ),
            )
        return (self._train_X,)

    @property
    def train_targets(self) -> Any:
        n = _ops.shape_of(self._train_Y)[-2]
        return (
            _reshape(_moveaxis(self._train_Y, -1, -2), (*self.batch_shape, self.num_outputs, n))
            if self.num_outputs > 1 and not self._correlated
            else _reshape(self._train_Y, (*self.batch_shape, n * self.num_outputs))
        )

    def condition_on_observations(self, X: Any, Y: Any, noise: Any = None, **kwargs: Any) -> Any:
        from botorch_rs.models.empirical_gps._conditioning import condition

        return condition(self, X, Y, noise=noise, **kwargs)

    def fantasize(self, X: Any, sampler: Any, observation_noise: Any = None, **kwargs: Any) -> Any:
        from botorch_rs.models.empirical_gps._conditioning import fantasize

        return fantasize(self, X, sampler, observation_noise=observation_noise, **kwargs)

    def posterior(
        self,
        X: Any,
        output_indices: Any = None,
        observation_noise: Any = False,
        posterior_transform: Any = None,
        **kwargs: Any,
    ) -> Any:
        from botorch_rs.acquisition.objective import transform_arg

        if kwargs or not isinstance(observation_noise, bool):
            raise UnsupportedError(
                "Empirical posteriors currently accept boolean observation_noise and no extra arguments."
            )
        if self._correlated and output_indices is not None:
            raise UnsupportedError(
                "MultiOutputEmpiricalOneDimensionalGP does not support output_indices."
            )
        self.eval()
        return GPyTorchPosterior(
            self._rs.model_handle(),
            to_flat(X),
            kind_of(X),
            observation_noise,
            num_outputs=self.num_outputs,
            model_batch_shape=self.batch_shape,
            output_indices=output_indices,
            transform=transform_arg(posterior_transform),
            interleaved=self._correlated,
            query_inputs=X,
        )

    def _acquisition_model(self, posterior_transform: Any = None) -> Any:
        from botorch_rs.acquisition.objective import transform_arg

        model = self._rs.model_handle()
        transform = transform_arg(posterior_transform)
        return model if transform is None else _rs.Model.from_models([model], transform)

    def train(self, mode: bool = True) -> EmpiricalOneDimensionalGP:
        self.training = mode
        return self

    def to(self, *args: Any, **kwargs: Any) -> EmpiricalOneDimensionalGP:
        if is_torch_tensor(self._train_X):
            self._train_X = self._train_X.to(*args, **kwargs)
            self._train_Y = self._train_Y.to(*args, **kwargs)
            self._kind = kind_of(self._train_X)
        return self

    def __deepcopy__(self, memo: dict[int, Any]) -> EmpiricalOneDimensionalGP:
        new = self.__class__.__new__(self.__class__)
        memo[id(self)] = new
        for name, value in self.__dict__.items():
            new.__dict__[name] = value.copy() if name == "_rs" else copy.deepcopy(value, memo)
        return new


__all__ = [
    "BaseAugmentedEmpiricalKernel",
    "EmpiricalOneDimensionalGP",
    "EmpiricalOneDimensionalMean",
    "EmpiricalOneDimensionalKernel",
]

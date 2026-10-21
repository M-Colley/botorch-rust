"""Differentiable empirical learning-curve means, kernels, and exact GPs."""

from __future__ import annotations

import copy
from typing import Any

import torch
from torch import nn

from botorch_rs import _botorch_rs as _rs
from botorch_rs.exceptions.errors import UnsupportedError
from botorch_rs.models._torch_backend import TorchSingleTaskGP
from botorch_rs.models._torch_backend_modules import (
    TorchConstraint,
    _TorchModule,
    build_kernel,
    module_log_prior,
)
from botorch_rs.models.empirical_gps.empirical_1d_gp import (
    BaseAugmentedEmpiricalKernel,
    EmpiricalOneDimensionalGP,
    EmpiricalOneDimensionalKernel,
    EmpiricalOneDimensionalMean,
)
from botorch_rs.models.empirical_gps.multioutput_empirical_1d_gp import (
    MultiOutputEmpiricalOneDimensionalGP,
    MultiOutputEmpiricalOneDimensionalKernel,
    MultiOutputEmpiricalOneDimensionalMean,
)
from botorch_rs.posteriors._torch_backend import TorchGPyTorchPosterior
from botorch_rs.utils.constraints import NonTransformedInterval


def validate_history(X: Any, Y: Any) -> None:
    if not isinstance(X, torch.Tensor) or not isinstance(Y, torch.Tensor):
        raise TypeError("The Tensor empirical backend requires Tensor historical data.")
    if (
        Y.ndim != 3
        or not Y.shape[0]
        or Y.shape[1] < 2
        or not Y.shape[2]
        or X.shape != (Y.shape[1], 1)
    ):
        raise ValueError("Historical X must be p x 1 and Y curves x p x outputs, with p >= 2.")
    if X.dtype not in (torch.float32, torch.float64) or Y.dtype != X.dtype or Y.device != X.device:
        raise TypeError("Historical tensors must share float32/float64 dtype and device.")
    if not bool(torch.isfinite(X).all()) or not bool(torch.isfinite(Y).all()):
        raise ValueError("Historical data must be finite.")
    if not bool((X.squeeze(-1).sort().values.diff() > 0).all()):
        raise ValueError("Historical progression knots must be distinct.")


class TensorLinearInterpolation(nn.Module):
    """Linear interpolation; interval labels are discrete, all weights are tensors."""

    _x: Any
    _y: Any

    def __init__(self, x: Any, y: Any) -> None:
        super().__init__()
        order = x.argsort()
        self.register_buffer("_x", x[order])
        self.register_buffer("_y", y.index_select(-1, order))

    def forward(self, query: Any) -> Any:
        x, y = self._x, self._y
        if not bool(torch.isfinite(query).all()):
            raise ValueError("Interpolation queries must be finite.")
        if bool(((query < x[0]) | (query > x[-1])).any()):
            raise ValueError("Query is outside the historical interpolation range.")
        indices = torch.searchsorted(x, query.contiguous()).clamp(1, x.numel() - 1)
        left, right = x[indices - 1], x[indices]
        weight = (query - left) / (right - left)
        return y[..., indices - 1] * (1 - weight) + y[..., indices] * weight


def _query_slice(X: Any, outputs: int) -> Any:
    if X.shape[-1] != 1:
        raise ValueError("Empirical progression inputs must have one feature.")
    if outputs > 1 and X.ndim >= 3 and X.shape[-3] == outputs:
        X = X.select(-3, 0)
    return X.squeeze(-1)


class TensorEmpiricalMean(_TorchModule, EmpiricalOneDimensionalMean):
    _correlated = False
    _historical_X: Any
    _historical_Y: Any

    def __init__(self, X_full: Any, Y_full: Any, *, backend: str = "torch") -> None:
        validate_history(X_full, Y_full)
        _TorchModule.__init__(self, X_full)
        self.register_buffer("_historical_X", X_full)
        self.register_buffer("_historical_Y", Y_full)
        self.num_outputs = Y_full.shape[-1]

    @property
    def mean_full(self) -> Any:
        if hasattr(self, "_mean_override"):
            return self._mean_override
        return self._historical_Y.mean(0).transpose(-1, -2)

    @property
    def f(self) -> Any:
        if hasattr(self, "_f_override"):
            return self._f_override
        return TensorLinearInterpolation(self._historical_X.squeeze(-1), self.mean_full)

    @f.setter
    def f(self, value: Any) -> None:
        self._f_override = value

    def forward(self, X: Any) -> Any:
        query = X.squeeze(-1) if self._correlated else _query_slice(X, self.num_outputs)
        result = self.f(query).movedim(0, -2)
        if self._correlated:
            return result.transpose(-1, -2).flatten(-2)
        return result.squeeze(-2) if self.num_outputs == 1 else result

    __call__ = nn.Module.__call__


class TensorCorrelatedMean(TensorEmpiricalMean, MultiOutputEmpiricalOneDimensionalMean):
    _correlated = True


class TensorEmpiricalKernel(_TorchModule, EmpiricalOneDimensionalKernel):
    _vectorize_outputs = False
    _historical_X: Any
    _historical_Y: Any
    curve_weights_constraint: Any

    def __init__(
        self,
        X_full: Any,
        Y_full: Any,
        ard: bool = False,
        curve_weights: Any = None,
        use_svd: Any = None,
        correction: int = 0,
        *,
        backend: str = "torch",
    ) -> None:
        validate_history(X_full, Y_full)
        _TorchModule.__init__(self, X_full)
        self.register_buffer("_historical_X", X_full)
        self.register_buffer("_historical_Y", Y_full)
        self.num_curves, progression, self.num_outputs = Y_full.shape
        if (
            isinstance(correction, bool)
            or not isinstance(correction, int)
            or correction >= self.num_curves
        ):
            raise ValueError("correction must be an integer smaller than num_curves.")
        self.correction, self.ard = correction, bool(ard)
        threshold = progression * self.num_outputs if self._vectorize_outputs else progression
        self._use_svd = (
            not ard and self.num_curves > threshold if use_svd is None else bool(use_svd)
        )
        self._effective_num_curves = (
            min(self.num_curves, threshold) if self.use_svd else self.num_curves
        )
        if curve_weights is None and ard:
            curve_weights = X_full.new_ones(self._effective_num_curves)
            requires_grad = True
        else:
            requires_grad = getattr(curve_weights, "requires_grad", False)
        if curve_weights is not None:
            curve_weights = torch.as_tensor(curve_weights, dtype=X_full.dtype, device=X_full.device)
            if curve_weights.shape != (self._effective_num_curves,) or not bool(
                torch.isfinite(curve_weights).all()
            ):
                raise ValueError("curve_weights must have one finite value per effective curve.")
            if ard and bool((curve_weights < 0).any()):
                raise ValueError("ARD curve weights must be nonnegative.")
        if ard:
            self.curve_weights_constraint = TorchConstraint(
                NonTransformedInterval(0.0, float("inf")), X_full
            )
            self.register_parameter(
                "curve_weights",
                curve_weights
                if isinstance(curve_weights, nn.Parameter)
                else nn.Parameter(curve_weights, requires_grad=requires_grad),
            )
        else:
            self.register_buffer("curve_weights", curve_weights)

    @property
    def curve_weights(self) -> Any:
        if "curve_weights" in self._parameters:
            return self._parameters["curve_weights"]
        try:
            return self._buffers["curve_weights"]
        except KeyError:
            raise AttributeError("curve_weights") from None

    @curve_weights.setter
    def curve_weights(self, value: Any) -> None:
        # nn.Module assigns registered Parameters directly; this setter is for
        # the native descriptor protocol and type-compatible property override.
        if self.ard:
            self._parameters["curve_weights"] = value
        else:
            self._buffers["curve_weights"] = value

    @property
    def f(self) -> Any:
        if hasattr(self, "_f_override"):
            return self._f_override
        Y = self._historical_Y
        centered = Y - Y.mean(0, keepdim=True)
        table = centered.movedim(-1, 0)
        if self.use_svd:
            if self._vectorize_outputs:
                _, values, vectors = torch.linalg.svd(centered.flatten(-2), full_matrices=False)
                table = (
                    (values.unsqueeze(-1) * vectors)
                    .reshape(self._effective_num_curves, Y.shape[-2], self.num_outputs)
                    .movedim(-1, 0)
                )
            else:
                values, vectors = torch.linalg.eigh(table.transpose(-1, -2) @ table)
                values, vectors = values.flip(-1).clamp_min(0), vectors.flip(-1)
                table = (values.sqrt().unsqueeze(-1) * vectors.transpose(-1, -2))[
                    ..., : self._effective_num_curves, :
                ]
        return TensorLinearInterpolation(self._historical_X.squeeze(-1), table)

    @f.setter
    def f(self, value: Any) -> None:
        self._f_override = value

    def initialize(self, **kwargs: Any) -> Any:
        return _TorchModule.initialize(self, **kwargs)

    def _basis_tensor(self, X: Any) -> Any:
        query = X.squeeze(-1) if self._vectorize_outputs else _query_slice(X, self.num_outputs)
        values = self.f(query).movedim(1, -2)
        if self.curve_weights is not None:
            values = values * self.curve_weights.unsqueeze(-1)
        if self._vectorize_outputs:
            values = values.movedim(0, -1).flatten(-2)
        return values

    def forward(
        self,
        x1: Any,
        x2: Any = None,
        diag: bool = False,
        last_dim_is_batch: bool = False,
        **kwargs: Any,
    ) -> Any:
        if last_dim_is_batch or kwargs:
            raise UnsupportedError(
                "Empirical kernels do not support feature batching or extra arguments."
            )
        a = self._basis_tensor(x1)
        b = a if x2 is None or x2 is x1 else self._basis_tensor(x2)
        value = (a * b).sum(-2) if diag else a.transpose(-1, -2) @ b
        value = value / (self.num_curves - self.correction)
        if not self._vectorize_outputs:
            value = value.movedim(0, -2 if diag else -3)
            if self.num_outputs == 1:
                value = value.squeeze(-2 if diag else -3)
        return value

    __call__ = nn.Module.__call__


class TensorCorrelatedKernel(TensorEmpiricalKernel, MultiOutputEmpiricalOneDimensionalKernel):
    _vectorize_outputs = True


class TensorBaseAugmentedKernel(_TorchModule, BaseAugmentedEmpiricalKernel):
    empirical_kernel: Any
    base_kernel: Any

    def __init__(self, empirical_kernel: Any, base_kernel: Any) -> None:
        _TorchModule.__init__(self, empirical_kernel._anchor)
        self.empirical_kernel = empirical_kernel
        self.base_kernel = build_kernel(base_kernel, 1, (), self._anchor)

    def forward(
        self,
        x1: Any,
        x2: Any = None,
        diag: bool = False,
        last_dim_is_batch: bool = False,
        **kwargs: Any,
    ) -> Any:
        a = self.empirical_kernel(x1, x2, diag=diag, last_dim_is_batch=last_dim_is_batch, **kwargs)
        b = self.base_kernel(x1, x2, diag=diag, last_dim_is_batch=last_dim_is_batch, **kwargs)
        return a + (b if isinstance(b, torch.Tensor) else b.to_dense())

    def log_prior(self) -> Any:
        total = module_log_prior(self.empirical_kernel, ())
        return (
            total
            if self.empirical_kernel is self.base_kernel
            else total + module_log_prior(self.base_kernel, ())
        )

    def initialize(self, **kwargs: Any) -> Any:
        return _TorchModule.initialize(self, **kwargs)

    def raw_parameter_bounds(self) -> Any:
        result = {
            "empirical_kernel." + name: bounds
            for name, bounds in module_bounds(self.empirical_kernel).items()
        }
        result.update(
            {
                "base_kernel." + name: bounds
                for name, bounds in module_bounds(self.base_kernel).items()
            }
        )
        return result

    __call__ = nn.Module.__call__


def module_bounds(module: Any) -> Any:
    method = getattr(module, "raw_parameter_bounds", None)
    if method is not None:
        return method()
    result = {}
    traversal = getattr(module, "named_parameters_and_constraints", None)
    if traversal is not None:
        for name, _, constraint in traversal():
            if constraint is not None and not constraint.enforced:
                result[name] = (float(constraint.lower_bound), float(constraint.upper_bound))
    return result


def convert_curve_module(source: Any, like: Any) -> Any:
    """Convert actual native interpolation buffers, never fabricate old history."""
    if isinstance(source, nn.Module):
        return source
    if isinstance(source, BaseAugmentedEmpiricalKernel):
        return TensorBaseAugmentedKernel(
            convert_curve_module(source.empirical_kernel, like), source.base_kernel
        )
    if isinstance(source, EmpiricalOneDimensionalMean):
        cls = (
            TensorCorrelatedMean
            if isinstance(source, MultiOutputEmpiricalOneDimensionalMean)
            else TensorEmpiricalMean
        )
        result: Any = object.__new__(cls)
        _TorchModule.__init__(result, like)
        knots = torch.as_tensor(source.f._x, dtype=like.dtype, device=like.device)
        table = torch.as_tensor(source.f._y, dtype=like.dtype, device=like.device)
        result.num_outputs = source.num_outputs
        result.register_buffer("_historical_X", knots.unsqueeze(-1))
        result.register_buffer(
            "_mean_override",
            torch.as_tensor(source.mean_full, dtype=like.dtype, device=like.device),
        )
        result._f_override = TensorLinearInterpolation(knots, table)
        return result
    if isinstance(source, EmpiricalOneDimensionalKernel):
        cls_kernel = TensorCorrelatedKernel if source._vectorize_outputs else TensorEmpiricalKernel
        result = object.__new__(cls_kernel)
        _TorchModule.__init__(result, like)
        for name in (
            "ard",
            "correction",
            "num_curves",
            "num_outputs",
            "_effective_num_curves",
            "_use_svd",
        ):
            setattr(result, name, getattr(source, name))
        knots = torch.as_tensor(source.f._x, dtype=like.dtype, device=like.device)
        table = torch.as_tensor(source.f._y, dtype=like.dtype, device=like.device)
        result.register_buffer("_historical_X", knots.unsqueeze(-1))
        result._f_override = TensorLinearInterpolation(knots, table)
        weights = source.curve_weights
        if weights is not None:
            weights = torch.as_tensor(weights, dtype=like.dtype, device=like.device)
        if source.ard:
            result.register_parameter(
                "curve_weights",
                weights
                if isinstance(weights, nn.Parameter)
                else nn.Parameter(
                    weights, requires_grad=getattr(source.curve_weights, "requires_grad", True)
                ),
            )
            result.curve_weights_constraint = TorchConstraint(source.curve_weights_constraint, like)
        else:
            result.register_buffer("curve_weights", weights)
        return result
    return source


class TensorVectorMean(_TorchModule):
    def __init__(self, base: Any, outputs: int, like: Any) -> None:
        super().__init__(like)
        self.base_mean, self.num_outputs = base, outputs

    def forward(self, X: Any) -> Any:
        if X.shape[-2] % self.num_outputs:
            raise ValueError("Expanded empirical inputs must contain one row per output.")
        return self.base_mean(X[..., :: self.num_outputs, :])

    def log_prior(self) -> Any:
        return module_log_prior(self.base_mean, ())

    def raw_parameter_bounds(self) -> Any:
        return {
            "base_mean." + name: bounds for name, bounds in module_bounds(self.base_mean).items()
        }


class TensorVectorKernel(_TorchModule):
    def __init__(self, base: Any, outputs: int, like: Any) -> None:
        super().__init__(like)
        self.base_kernel, self.num_outputs = base, outputs

    def forward(self, X: Any, Z: Any = None, **kwargs: Any) -> Any:
        if X.shape[-2] % self.num_outputs or (Z is not None and Z.shape[-2] % self.num_outputs):
            raise ValueError("Expanded empirical inputs must contain one row per output.")
        return self.base_kernel.forward(
            X[..., :: self.num_outputs, :],
            None if Z is None else Z[..., :: self.num_outputs, :],
            **kwargs,
        )

    def log_prior(self) -> Any:
        return module_log_prior(self.base_kernel, ())

    def raw_parameter_bounds(self) -> Any:
        return {
            "base_kernel." + name: bounds
            for name, bounds in module_bounds(self.base_kernel).items()
        }


class TensorEmpiricalGP(TorchSingleTaskGP, EmpiricalOneDimensionalGP):
    @property
    def _num_outputs(self) -> Any:
        return self.num_outputs

    @_num_outputs.setter
    def _num_outputs(self, value: Any) -> None:
        if value != self.num_outputs:
            raise UnsupportedError("Output count derives from the training target layout.")

    @property
    def _aug_batch_shape(self) -> Any:
        return self.batch_shape + ((self.num_outputs,) if self.num_outputs > 1 else ())

    @_aug_batch_shape.setter
    def _aug_batch_shape(self, value: Any) -> None:
        if torch.Size(value) != self._aug_batch_shape:
            raise UnsupportedError("Augmented batch shape derives from the training targets.")

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
        backend: str = "torch",
    ) -> None:
        from botorch_rs.models.empirical_gps.utils import validate_no_transforms

        validate_no_transforms(input_transform, outcome_transform, type(self).__name__)
        validate_history(historical_X, historical_Y)
        if train_X.shape[-1] != 1 or train_Y.shape[-1] != historical_Y.shape[-1]:
            raise ValueError("Training progression/output dimensions must match historical curves.")
        if historical_X.dtype != train_X.dtype or historical_X.device != train_X.device:
            raise ValueError("Historical and training data must share dtype and device.")
        correlated = isinstance(self, TensorCorrelatedGP)
        mean_type = TensorCorrelatedMean if correlated else TensorEmpiricalMean
        kernel_type = TensorCorrelatedKernel if correlated else TensorEmpiricalKernel
        mean = mean_type(historical_X, historical_Y) if mean_module is None else mean_module
        kernel = (
            kernel_type(historical_X, historical_Y, ard=ard)
            if covar_module is None
            else covar_module
        )
        mean = convert_curve_module(mean, train_X)
        kernel = convert_curve_module(kernel, train_X)
        if hasattr(kernel, "ard") and kernel.ard != ard:
            raise ValueError("ard must equal covar_module.ard.")
        if base_covar_module is not None and not correlated:
            kernel = TensorBaseAugmentedKernel(kernel, base_covar_module)
        external = isinstance(likelihood, nn.Module)
        if external and train_Yvar is not None:
            raise UnsupportedError("Specify a likelihood or train_Yvar, not both.")
        if external and not any(
            cls.__name__ in ("GaussianLikelihood", "TorchGaussianLikelihood")
            for cls in type(likelihood).__mro__
        ):
            raise UnsupportedError(
                "Tensor empirical models accept GaussianLikelihood modules or train_Yvar for fixed noise."
            )
        if correlated and likelihood is None and train_Yvar is None:
            from botorch_rs.likelihoods import GaussianLikelihood

            likelihood = GaussianLikelihood()
        TorchSingleTaskGP.__init__(
            self,
            train_X,
            train_Y,
            train_Yvar,
            None if external else likelihood,
            kernel,
            mean,
            outcome_transform=None,
            input_transform=None,
            backend=backend,
        )
        if external:
            self.likelihood = likelihood
        if correlated:
            self.mean_module = TensorVectorMean(mean, self.num_outputs, train_X)
            vector_kernel = TensorVectorKernel(kernel, self.num_outputs, train_X)
            self.covar_module = (
                vector_kernel
                if base_covar_module is None
                else TensorBaseAugmentedKernel(vector_kernel, base_covar_module)
            )
            self._base_mean = mean
            self._base_kernel = kernel

    def subset_output(self, idcs: Any) -> Any:
        result = TorchSingleTaskGP.subset_output(self, idcs)
        indices = torch.as_tensor(idcs, dtype=torch.long, device=self._train_X.device).remainder(
            self.num_outputs
        )
        for module in result.modules():
            if isinstance(module, (TensorEmpiricalMean, TensorEmpiricalKernel)):
                if hasattr(module, "_historical_Y"):
                    module._historical_Y = module._historical_Y.index_select(-1, indices)
                else:
                    module._f_override._y = module._f_override._y.index_select(0, indices)
                    if hasattr(module, "_mean_override"):
                        mean_values: Any = module._mean_override
                        module._mean_override = mean_values.index_select(0, indices)
                module.num_outputs = indices.numel()
        return result


class TensorCorrelatedGP(TensorEmpiricalGP, MultiOutputEmpiricalOneDimensionalGP):
    @property
    def _aug_batch_shape(self) -> Any:
        return self.batch_shape

    @_aug_batch_shape.setter
    def _aug_batch_shape(self, value: Any) -> None:
        if torch.Size(value) != self.batch_shape:
            raise UnsupportedError("Correlated empirical batches derive from training data.")

    def _augment_X(self, X: Any) -> Any:
        return X

    def _augment_Y(self, Y: Any) -> Any:
        return Y.flatten(-2)

    @property
    def train_inputs(self) -> Any:
        return (self._train_X.repeat_interleave(self.num_outputs, dim=-2),)

    def _prior_mean(self, X: Any) -> Any:
        return self.mean_module(X.repeat_interleave(self.num_outputs, dim=-2))

    def _covariance(self, X: Any, Z: Any) -> Any:
        value = self.covar_module(
            X.repeat_interleave(self.num_outputs, dim=-2),
            Z.repeat_interleave(self.num_outputs, dim=-2),
        )
        return value if isinstance(value, torch.Tensor) else value.to_dense()

    def forward(self, X: Any) -> Any:
        if self.training:
            mean, covariance = self.mean_module(X), self.covar_module(X, X)
            if not isinstance(covariance, torch.Tensor):
                covariance = covariance.to_dense()
        else:
            mean, covariance = self._posterior_tensors(X[..., :: self.num_outputs, :])
        return TorchGPyTorchPosterior(mean.unsqueeze(-1), covariance).distribution

    def _exact_mll(self, output: Any = None, target: Any = None) -> Any:
        X, Y, fixed = self._training_values()
        if output is None:
            mean, covariance = self._prior_mean(X), self._covariance(X, X)
        else:
            mean, covariance = output.mean, output.covariance_matrix
            Y = Y if target is None else target
        return (
            _rs.tensor_exact_mll(
                Y, covariance, mean, self._training_noise(Y, fixed), normalize=False
            )
            + self._log_prior()
        ) / Y.shape[-1]

    def _posterior_tensors(
        self, X: Any, observation_noise: Any = False, *, inputs_transformed: bool = False
    ) -> Any:
        self.eval()
        if X.shape[-1] != 1 or X.dtype != self._train_X.dtype or X.device != self._train_X.device:
            raise ValueError(
                "Queries must have one progression feature and the model dtype/device."
            )
        train, Y, fixed = self._training_values()
        chol, alpha = _rs.tensor_exact_gp_factor(
            Y,
            self._covariance(train, train),
            self._prior_mean(train),
            self._training_noise(Y, fixed),
        )
        noise = None
        if isinstance(observation_noise, torch.Tensor):
            observation_noise = observation_noise.to(X)
            batch = torch.broadcast_shapes(
                self.batch_shape,
                X.shape[:-2],
                observation_noise.shape[:-2] if observation_noise.ndim >= 2 else (),
            )
            noise = observation_noise.expand(*batch, X.shape[-2], self.num_outputs).flatten(-2)
        elif observation_noise is True:
            noise = (
                fixed.mean()
                if fixed is not None
                else self._noise_for(X.shape[-2] * self.num_outputs)
            )
        elif observation_noise is not False:
            raise TypeError("observation_noise must be bool or a Tensor.")
        return _rs.tensor_exact_gp_predict(
            chol,
            alpha,
            self._covariance(train, X),
            self._covariance(X, X),
            self._prior_mean(X),
            observation_noise=noise,
        )

    def posterior(
        self,
        X: Any,
        output_indices: Any = None,
        observation_noise: Any = False,
        posterior_transform: Any = None,
        **kwargs: Any,
    ) -> Any:
        if output_indices is not None or kwargs:
            raise UnsupportedError(
                "Correlated empirical posteriors do not support output_indices or extra arguments."
            )
        mean, covariance = self._posterior_tensors(X, observation_noise)
        q, m = X.shape[-2], self.num_outputs
        mean = mean.reshape(*mean.shape[:-1], q, m)
        permutation = torch.arange(q * m, device=X.device).reshape(q, m).transpose(-1, -2).flatten()
        covariance = covariance.index_select(-2, permutation).index_select(-1, permutation)
        posterior = TorchGPyTorchPosterior(mean, covariance)
        return posterior if posterior_transform is None else posterior_transform(posterior, X=X)

    def subset_output(self, idcs: Any) -> Any:
        raise UnsupportedError("Correlated empirical models do not implement subset_output.")

    def condition_on_observations(self, X: Any, Y: Any, noise: Any = None, **kwargs: Any) -> Any:
        if kwargs or X.shape[-2] != Y.shape[-2] or Y.shape[-1] != self.num_outputs:
            raise ValueError("Conditioning inputs and output grid dimensions must match the model.")
        batch = torch.broadcast_shapes(self.batch_shape, X.shape[:-2], Y.shape[:-2])
        model = copy.copy(self)
        model._buffers, model._modules = self._buffers.copy(), self._modules.copy()
        model._train_X = torch.cat(
            (
                self._train_X.expand(*batch, *self._train_X.shape[-2:]),
                X.expand(*batch, *X.shape[-2:]),
            ),
            -2,
        )
        model._train_Y = torch.cat(
            (
                self._train_Y.expand(*batch, *self._train_Y.shape[-2:]),
                Y.expand(*batch, *Y.shape[-2:]),
            ),
            -2,
        )
        if self._train_Yvar is not None:
            added = self._train_Yvar.mean() if noise is None else noise
            added = torch.as_tensor(added, dtype=Y.dtype, device=Y.device).expand(
                *batch, *Y.shape[-2:]
            )
            model._train_Yvar = torch.cat(
                (self._train_Yvar.expand(*batch, *self._train_Y.shape[-2:]), added), -2
            )
        elif noise is not None or self._noise_override is not None:
            previous = (
                torch.full_like(self._augment_Y(self._train_Y), torch.nan)
                if self._noise_override is None
                else self._noise_override
            )
            extra = (
                torch.full_like(self._augment_Y(Y), torch.nan)
                if noise is None
                else self._augment_Y(noise.expand_as(Y))
            )
            model._noise_override = torch.cat(
                (
                    previous.expand(*batch, previous.shape[-1]),
                    extra.expand(*batch, extra.shape[-1]),
                ),
                -1,
            )
        model._prediction_cache = None
        return model.eval()

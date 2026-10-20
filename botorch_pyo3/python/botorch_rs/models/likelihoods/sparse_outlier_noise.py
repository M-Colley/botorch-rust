"""Sparse Gaussian outlier noise, including native exact-GP fitting."""

from __future__ import annotations

import math
import warnings
from typing import Any

from botorch_rs import _ops
from botorch_rs._arrays import LIST, from_flat, kind_of, to_flat
from botorch_rs._module import GPModule
from botorch_rs.exceptions.errors import UnsupportedError
from botorch_rs.exceptions.warnings import InputDataWarning
from botorch_rs.likelihoods import FixedNoiseGaussianLikelihood, GaussianLikelihood, Likelihood
from botorch_rs.models.relevance_pursuit import RelevancePursuitMixin
from botorch_rs.priors import Prior
from botorch_rs.utils.constraints import NonTransformedInterval


class _DiagonalNoise:
    def __init__(self, values: Any) -> None:
        self._values = values
        self.shape = (*_ops.shape_of(values), _ops.shape_of(values)[-1])

    def diagonal(self, dim1: int = -2, dim2: int = -1) -> Any:
        if (dim1, dim2) != (-2, -1):
            raise UnsupportedError("Only the matrix diagonal is supported.")
        return self._values

    def to_dense(self) -> Any:
        if kind_of(self._values).lib == "torch":
            return self._values.diag_embed()
        data, shape = to_flat(self._values)
        n = shape[-1]
        values = [
            data[b * n + i] if i == j else 0.0
            for b in range(math.prod(shape[:-1]))
            for i in range(n)
            for j in range(n)
        ]
        return from_flat(values, (*shape, n), kind_of(self._values))


class SparseOutlierNoise(Likelihood, RelevancePursuitMixin):
    """One nonnegative outlier variance per active training observation.

    Exact native fitting currently supports an unbatched, single-output GP.
    Batched sparse parameters are rejected at construction rather than being
    indexed along the wrong dimension.
    """

    def __init__(
        self,
        base_noise: Any,
        dim: int,
        outlier_indices: list[int] | None = None,
        rho_prior: Prior | None = None,
        rho_constraint: Any = None,
        batch_shape: Any = None,
        convex_parameterization: bool = True,
        loo: bool = True,
    ) -> None:
        RelevancePursuitMixin.__init__(self, dim, outlier_indices)
        batch = (
            tuple(getattr(base_noise, "batch_shape", ()))
            if batch_shape is None
            else tuple(batch_shape)
        )
        if batch:
            raise UnsupportedError("Sparse outlier noise currently requires batch_shape=().")
        if rho_prior is not None and not isinstance(rho_prior, Prior):
            raise UnsupportedError("rho_prior must be a native botorch_rs Prior.")
        constraint = rho_constraint or NonTransformedInterval(
            0.0, 0.999 if convex_parameterization else math.inf, initial_value=0.0
        )
        if (
            not isinstance(constraint, NonTransformedInterval)
            or constraint.lower_bound < 0
            or (convex_parameterization and constraint.upper_bound >= 1)
        ):
            raise ValueError(
                "rho_constraint must be a nonnegative NonTransformedInterval, with upper bound < 1 for convex parameterization."
            )
        if constraint.lower_bound != 0.0:
            raise UnsupportedError(
                "Sparse relevance pursuit requires rho_constraint.lower_bound=0 because inactive variance factors are zero."
            )
        self.base_noise = base_noise
        self.rho_prior = rho_prior
        self.raw_rho_constraint = constraint
        self.batch_shape = batch
        self._convex_parameterization = bool(convex_parameterization)
        self.loo = bool(loo)
        initial = (
            constraint.initial_value
            if constraint.initial_value is not None
            else constraint.lower_bound
        )
        self._raw = [initial] * len(self.support)
        self._kind = LIST
        self._cached_train_inputs: Any = None
        self.training = True

    @staticmethod
    def _from_model(model: Any) -> SparseOutlierNoise:
        noise = getattr(getattr(model, "likelihood", None), "noise_covar", None)
        if not isinstance(noise, SparseOutlierNoise):
            raise ValueError("Expected model.likelihood.noise_covar to be SparseOutlierNoise.")
        return noise

    def _bind(self, model: Any) -> None:
        GPModule._bind(self, model)
        self._kind = model._kind
        self.base_noise._bind(model)
        self._push(self._raw)

    def _push(self, values: list[float]) -> None:
        if self._model is not None:
            self._model._sync_torch_parameters()
            c = self.raw_rho_constraint
            indices = self.support if self.is_sparse else list(range(self.dim))
            self._model._rs.set_sparse_outlier_noise(
                indices,
                values,
                self.convex_parameterization,
                c.lower_bound,
                c.upper_bound,
                None if self.rho_prior is None else self.rho_prior._rs,
            )
        self._raw = list(values)

    @property
    def convex_parameterization(self) -> bool:
        return self._convex_parameterization

    @property
    def raw_rho(self) -> Any:
        if self._model is not None:
            return self._model._raw_param_array("raw_rho")
        values = self._raw
        return from_flat(values, (len(values),), self._kind)

    @raw_rho.setter
    def raw_rho(self, value: Any) -> None:
        self.set_sparse_parameter(value)

    @property
    def sparse_parameter(self) -> Any:
        return self.raw_rho

    def set_sparse_parameter(self, value: Any) -> None:
        data, shape = to_flat(value, "raw_rho")
        expected = len(self.support) if self.is_sparse else self.dim
        if shape != (expected,):
            raise ValueError(f"raw_rho must have shape ({expected},)")
        values = list(data)
        c = self.raw_rho_constraint
        if any(not math.isfinite(v) or not c.lower_bound <= v <= c.upper_bound for v in values):
            raise ValueError("raw_rho is outside its non-transformed constraint.")
        if self._model is None:
            self._kind = kind_of(value)
        self._push(values)

    @property
    def rho(self) -> Any:
        raw = from_flat(self._dense(), (self.dim,), self._kind)
        return (
            _ops.apply("div_raw", raw, _ops.apply("sub", 1.0, raw))
            if self.convex_parameterization
            else raw
        )

    def _convex_rho(self) -> Any:
        raw = self.raw_rho
        return _ops.apply("div_raw", raw, _ops.apply("sub", 1.0, raw))

    def train(self, mode: bool = True) -> SparseOutlierNoise:
        self.training = bool(mode)
        return self

    def eval(self) -> SparseOutlierNoise:
        return self.train(False)

    def expansion_objective(self, mll: Any) -> Any:
        if not self.loo:
            return self._sparse_parameter_gradient(mll)
        return self._optimal_rhos(mll)

    def _optimal_rhos(self, mll: Any) -> Any:
        mll.train()
        values = mll.model._rs.sparse_outlier_expansion(True)
        inactive = [i for i in range(self.dim) if i not in self.support]
        return from_flat([values[i] for i in inactive], (len(inactive),), self._kind)

    def forward(
        self, X: Any = None, shape: Any = None, diag_K: Any = None, **kwargs: Any
    ) -> _DiagonalNoise:
        if kwargs:
            raise UnsupportedError(f"Unsupported noise arguments: {sorted(kwargs)}")
        if isinstance(X, (tuple, list)) and len(X) == 1 and len(_ops.shape_of(X[0])) >= 2:
            X = X[0]
        if self._model is None and X is not None:
            self._kind = kind_of(X)
        n = (
            int(tuple(shape)[-1])
            if shape is not None
            else (_ops.shape_of(X)[-2] if X is not None else self.dim)
        )
        if isinstance(self.base_noise, GaussianLikelihood) and self.base_noise._model is None:
            constraint = self.base_noise.noise_constraint
            raw = constraint.initial_value if constraint.initial_value is not None else 0.0
            noise = from_flat([constraint.transform(raw)], (1,), self._kind)
        elif hasattr(self.base_noise, "noise"):
            noise = self.base_noise.noise
        else:
            result = self.base_noise(X, shape=shape)
            noise = result.diagonal()
        data, nsh = to_flat(noise)
        if isinstance(self.base_noise, FixedNoiseGaussianLikelihood) and len(data) != n:
            noise = from_flat([0.0] * n, (n,), kind_of(noise))
        elif len(data) == 1:
            noise = _ops.expand(noise, (n,))
        elif nsh != (n,):
            # Fixed noise outside its training shape is omitted by GPyTorch.
            noise = from_flat([0.0] * n, (n,), kind_of(noise))
        apply = X is not None and n == self.dim
        if apply and (self.training or self._cached_train_inputs is None):
            flat = to_flat(X)
            self._cached_train_inputs = (list(flat[0]), flat[1])
        elif apply:
            flat = to_flat(X)
            apply = (list(flat[0]), flat[1]) == self._cached_train_inputs
        if apply:
            rho = self.rho
            if diag_K is not None:
                rho = _ops.apply("mul_raw", _ops.apply("add", diag_K, noise), rho)
            noise = _ops.apply("add", noise, rho)
        else:
            warnings.warn(
                "SparseOutlierNoise: Robust rho not applied because inputs do not match the training data.",
                InputDataWarning,
                stacklevel=2,
            )
        return _DiagonalNoise(noise)

    __call__ = forward


class SparseOutlierGaussianLikelihood(Likelihood):
    def __init__(
        self,
        base_noise: Any,
        dim: int,
        outlier_indices: list[int] | None = None,
        rho_prior: Prior | None = None,
        rho_constraint: Any = None,
        batch_shape: Any = None,
        convex_parameterization: bool = True,
        loo: bool = True,
    ) -> None:
        self._noise_covar = SparseOutlierNoise(
            base_noise,
            dim,
            outlier_indices,
            rho_prior,
            rho_constraint,
            batch_shape,
            convex_parameterization,
            loo,
        )
        self.batch_shape = self._noise_covar.batch_shape
        if isinstance(base_noise, GaussianLikelihood):
            self._spec = base_noise._spec

    @property
    def noise_covar(self) -> SparseOutlierNoise:
        return self._noise_covar

    def _bind(self, model: Any) -> None:
        GPModule._bind(self, model)
        if self.noise_covar.dim != model._rs.num_train:
            raise ValueError("Sparse outlier dim must equal the number of training observations.")
        self.noise_covar._bind(model)

    def train(self, mode: bool = True) -> SparseOutlierGaussianLikelihood:
        self.noise_covar.train(mode)
        return self

    def eval(self) -> SparseOutlierGaussianLikelihood:
        return self.train(False)

    def marginal(self, function_dist: Any, X: Any = None, **kwargs: Any) -> Any:
        from botorch_rs.posteriors.gpytorch import _MomentPosterior

        if not hasattr(function_dist, "covariance_matrix"):
            raise UnsupportedError(
                "Sparse likelihood marginal requires a Gaussian distribution; samplewise likelihood evaluation is not supported."
            )
        mean = function_dist.mean
        covariance = function_dist.covariance_matrix
        cov, shape = to_flat(covariance)
        if len(shape) != 2 or shape[0] != shape[1]:
            raise UnsupportedError(
                "Sparse likelihood marginal currently supports an unbatched Gaussian."
            )
        n = shape[0]
        diagonal = (
            covariance.diagonal()
            if kind_of(covariance).lib == "torch"
            else from_flat([cov[i * n + i] for i in range(n)], (n,), kind_of(covariance))
        )
        noise = self.noise_covar(
            X=X,
            shape=(n,),
            diag_K=diagonal if self.noise_covar.convex_parameterization else None,
            **kwargs,
        ).to_dense()
        covariance = _ops.apply("add", covariance, noise)
        values, _ = to_flat(mean)
        return _MomentPosterior((values, (n,)), to_flat(covariance), kind_of(mean)).distribution

    __call__ = marginal

    def expected_log_prob(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError(
            "SparseOutlierGaussianLikelihood does not support variational inference."
        )


__all__ = ["SparseOutlierGaussianLikelihood", "SparseOutlierNoise"]

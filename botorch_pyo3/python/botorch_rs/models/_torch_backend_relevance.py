"""Device-resident sparse observation variances and exact relevance pursuit."""

from __future__ import annotations

import copy
import math
import warnings
from typing import Any

import torch
from torch import nn

from botorch_rs import _botorch_rs as _rs
from botorch_rs.exceptions.errors import UnsupportedError
from botorch_rs.exceptions.warnings import InputDataWarning
from botorch_rs.models._torch_backend import TorchSingleTaskGP
from botorch_rs.models._torch_backend_modules import (
    TorchConstraint,
    TorchFixedNoiseLikelihood,
    TorchPrior,
    _TorchModule,
    module_log_prior,
)
from botorch_rs.models.likelihoods.sparse_outlier_noise import (
    SparseOutlierGaussianLikelihood,
    SparseOutlierNoise,
    _DiagonalNoise,
)
from botorch_rs.models.relevance_pursuit import RelevancePursuitMixin, _indices
from botorch_rs.models.robust_relevance_pursuit_model import (
    FRACTIONS_OF_OUTLIERS,
    RobustRelevancePursuitSingleTaskGP,
)
from botorch_rs.utils.constraints import NonTransformedInterval
from botorch_rs.utils.types import DEFAULT


def tensor_support_posterior(rp_class: Any, trace: Any, prior: Any, prior_mean: Any) -> Any:
    from botorch_rs.mlls import ExactMarginalLogLikelihood

    anchor = trace[0]._train_X
    sizes = anchor.new_tensor([len(rp_class._from_model(model).support) for model in trace])
    values = []
    for model in trace:
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        mll.train()
        value = mll()
        if value.numel() != 1:
            raise UnsupportedError("Support comparison requires unbatched single-output models.")
        values.append(value.reshape(()))
    if prior is None:
        log_prior = -sizes / prior_mean - math.log(prior_mean)
    else:
        terms = [prior(size) for size in sizes.unbind()]
        if any(not isinstance(term, torch.Tensor) or term.shape for term in terms):
            raise ValueError("log_support_prior must return a scalar Tensor log density.")
        log_prior = torch.stack(terms)
    logits = torch.stack(values) + log_prior
    return sizes, (logits - logits.logsumexp(-1)).exp()


def tensor_initialize_dense(model: Any) -> Any:
    values = {}
    with torch.no_grad():
        for name, value in model.named_parameters():
            prefix, _, local = name.rpartition(".")
            owner = model.get_submodule(prefix)
            constraint = getattr(owner, local + "_constraint", None)
            if constraint is None:
                initial = value.new_zeros(())
            elif constraint.initial_value is not None:
                initial = torch.as_tensor(
                    constraint.initial_value, dtype=value.dtype, device=value.device
                )
            else:
                lower, upper = constraint.lower_bound, constraint.upper_bound
                finite_lower, finite_upper = torch.isfinite(lower), torch.isfinite(upper)
                chosen = torch.where(
                    finite_upper,
                    torch.where(
                        finite_lower,
                        lower + torch.minimum(torch.ones_like(lower), (upper - lower) / 2),
                        upper - 1,
                    ),
                    torch.where(finite_lower, lower + 1, torch.zeros_like(lower)),
                )
                initial = constraint.inverse_transform(chosen)
            value.copy_(initial.expand_as(value))
            values[name] = value.clone()
    return model, values


class TorchSparseOutlierNoise(_TorchModule, SparseOutlierNoise):
    """Shared support, independent batch variances, indexed on the final axis."""

    base_noise: Any
    raw_rho_constraint: Any
    rho_prior: Any
    _cached_train_inputs: Any

    def __init__(
        self,
        base_noise: Any,
        dim: int,
        outlier_indices: Any = None,
        rho_prior: Any = None,
        rho_constraint: Any = None,
        batch_shape: Any = None,
        convex_parameterization: bool = True,
        loo: bool = True,
    ) -> None:
        like = base_noise.noise
        _TorchModule.__init__(self, like)
        RelevancePursuitMixin.__init__(self, dim, outlier_indices)
        self.base_noise = base_noise
        self.batch_shape = torch.Size(
            getattr(base_noise, "batch_shape", like.shape[:-1])
            if batch_shape is None
            else batch_shape
        )
        source: Any = rho_constraint or NonTransformedInterval(
            0.0, 0.999 if convex_parameterization else math.inf, initial_value=0.0
        )
        lower: Any = source.lower_bound
        upper: Any = source.upper_bound
        if (
            not (
                isinstance(source, NonTransformedInterval)
                or (isinstance(source, nn.Module) and not source.enforced)
            )
            or lower != 0
            or (convex_parameterization and upper >= 1)
        ):
            raise ValueError(
                "rho_constraint must be untransformed with lower bound 0 and convex upper bound < 1."
            )
        self.raw_rho_constraint = (
            source.to(like) if isinstance(source, nn.Module) else TorchConstraint(source, like)
        )
        self._rho_raw_bounds = (float(lower), float(upper))
        self._convex_parameterization = bool(convex_parameterization)
        self.loo = bool(loo)
        initial = 0.0 if source.initial_value is None else source.initial_value
        self.register_parameter(
            "raw_rho",
            nn.Parameter(
                like.new_zeros(*self.batch_shape, len(self.support))
                + torch.as_tensor(initial, dtype=like.dtype, device=like.device)
            ),
        )
        if rho_prior is not None:
            self.rho_prior = (
                rho_prior if isinstance(rho_prior, nn.Module) else TorchPrior(rho_prior, like)
            )
            self._prior_parameters["rho_prior"] = ("rho", 1)
        self.register_buffer("_cached_train_inputs", None, persistent=False)

    def raw_parameter_bounds(self) -> Any:
        result = {"raw_rho": self._rho_raw_bounds}
        method = getattr(self.base_noise, "raw_parameter_bounds", None)
        if method is not None:
            result.update({"base_noise." + name: bound for name, bound in method().items()})
        else:
            traversal = getattr(self.base_noise, "named_parameters_and_constraints", None)
            if traversal is not None:
                for name, _, constraint in traversal():
                    if constraint is not None and not constraint.enforced:
                        result["base_noise." + name] = (
                            float(constraint.lower_bound),
                            float(constraint.upper_bound),
                        )
        return result

    def log_prior(self) -> Any:
        total = self._anchor.new_zeros(self.batch_shape) + module_log_prior(
            self.base_noise, self.batch_shape
        )
        if hasattr(self, "rho_prior"):
            total = total + self.rho_prior.log_prob(self.rho).sum(-1)
        return total

    @property
    def raw_rho(self) -> Any:
        try:
            return self._parameters["raw_rho"]
        except KeyError:
            raise AttributeError("raw_rho") from None

    @raw_rho.setter
    def raw_rho(self, value: Any) -> None:
        self.set_sparse_parameter(value)

    def set_sparse_parameter(self, value: Any) -> None:
        old = self.raw_rho
        value = torch.as_tensor(value, dtype=old.dtype, device=old.device)
        count = len(self.support) if self.is_sparse else self.dim
        if value.shape != (*self.batch_shape, count):
            raise ValueError(f"raw_rho must have shape {(*self.batch_shape, count)}.")
        lower, upper = self.raw_rho_constraint.lower_bound, self.raw_rho_constraint.upper_bound
        if not bool((torch.isfinite(value) & (value >= lower) & (value <= upper)).all()):
            raise ValueError("raw_rho is outside its non-transformed constraint.")
        self._parameters["raw_rho"] = (
            value
            if isinstance(value, nn.Parameter)
            else nn.Parameter(value, requires_grad=old.requires_grad)
        )

    def _dense_tensor(self) -> Any:
        raw = self.raw_rho
        if not self.is_sparse:
            return raw
        indices = torch.tensor(self.support, dtype=torch.long, device=raw.device)
        return raw.new_zeros(*raw.shape[:-1], self.dim).index_copy(-1, indices, raw)

    @property
    def rho(self) -> Any:
        raw = self._dense_tensor()
        return raw / (1 - raw) if self.convex_parameterization else raw

    @property
    def _convex_rho(self) -> Any:
        return self.raw_rho / (1 - self.raw_rho)

    def _replace_tensor(self, support: list[int], dense: Any, sparse: bool) -> None:
        if sparse:
            dense = dense.index_select(
                -1, torch.tensor(support, dtype=torch.long, device=dense.device)
            )
        previous = self._support, self._is_sparse
        self._support, self._is_sparse = support, sparse
        try:
            self.set_sparse_parameter(nn.Parameter(dense, requires_grad=self.raw_rho.requires_grad))
        except Exception:
            self._support, self._is_sparse = previous
            raise

    def to_sparse(self) -> Any:
        if not self.is_sparse:
            self._replace_tensor(self.support.copy(), self._dense_tensor(), True)
        return self

    def to_dense(self) -> Any:
        if self.is_sparse:
            self._replace_tensor(self.support.copy(), self._dense_tensor(), False)
        return self

    def expand_support(self, indices: Any) -> Any:
        indices = _indices(indices, self.dim)
        if any(i in self.support for i in indices):
            raise ValueError("Cannot expand support with already active indices.")
        self._replace_tensor(self.support + indices, self._dense_tensor(), self.is_sparse)
        return self

    def contract_support(self, indices: Any) -> Any:
        indices = _indices(indices, self.dim)
        if any(i not in self.support for i in indices):
            raise ValueError("Cannot contract support with inactive indices.")
        dense = self._dense_tensor().clone()
        dense[..., indices] = 0
        self._replace_tensor([i for i in self.support if i not in indices], dense, self.is_sparse)
        return self

    def remove_support(self) -> Any:
        self._replace_tensor(
            [], self.raw_rho.new_zeros(*self.batch_shape, self.dim), self.is_sparse
        )
        return self

    def _require_unbatched_ranking(self) -> None:
        if self.batch_shape or getattr(self, "_ranking_batch_shape", ()):
            raise UnsupportedError(
                "Discrete relevance-pursuit ranking requires an unbatched single-output model; fixed-support batched likelihoods are supported."
            )

    def base_variance(self, count: int, noise: Any = None, X: Any = None, shape: Any = None) -> Any:
        if noise is not None:
            return noise
        if not isinstance(self.base_noise, _TorchModule):
            result = self.base_noise(X, shape=shape if shape is not None else torch.Size([count]))
            return result.diagonal(dim1=-2, dim2=-1)
        base: Any = self.base_noise.noise
        if base.shape[-1] not in (1, count):
            return base.new_zeros(*base.shape[:-1], count)
        return base.expand(*base.shape[:-1], count)

    def forward(self, X: Any = None, shape: Any = None, diag_K: Any = None, **kwargs: Any) -> Any:
        noise = kwargs.pop("noise", None)
        if kwargs:
            raise UnsupportedError(f"Unsupported sparse noise arguments: {sorted(kwargs)}.")
        if isinstance(X, (list, tuple)):
            if len(X) != 1:
                raise UnsupportedError("Sparse outlier noise requires one input tensor.")
            X = X[0]
        count = shape[-1] if shape is not None else self.dim if X is None else X.shape[-2]
        base = self.base_variance(count, noise, X, shape)
        apply = X is not None and count == self.dim
        if apply and (self.training or self._cached_train_inputs is None):
            self._cached_train_inputs = X.detach().clone()
        elif apply:
            apply = torch.equal(X, self._cached_train_inputs)
        if apply:
            robust = self.rho
            if diag_K is not None:
                robust = (diag_K + base) * robust
            base = base + robust
        else:
            warnings.warn(
                "SparseOutlierNoise: Robust rho not applied because inputs do not match the training data.",
                InputDataWarning,
                stacklevel=2,
            )
        return _DiagonalNoise(base)

    __call__ = nn.Module.__call__

    def _optimal_rhos(self, mll: Any) -> Any:
        self._require_unbatched_ranking()
        mll.train()
        X, Y, fixed = mll.model._training_values()
        mean, covariance = mll.model.mean_module(X), mll.model._covariance(X, X)
        noise = self(
            X,
            shape=Y.shape,
            diag_K=covariance.diagonal(dim1=-2, dim2=-1) if self.convex_parameterization else None,
            noise=mll.model._base_training_noise(X, Y, fixed),
        ).diagonal()
        chol, alpha = _rs.tensor_exact_gp_factor(Y, covariance, mean, noise)
        inverse_diag = torch.cholesky_inverse(chol).diagonal(dim1=-2, dim2=-1)
        error = alpha / inverse_diag
        scores = (error.square() - inverse_diag.reciprocal() - self.rho).clamp_min(0)
        return scores[..., ~self.is_active]

    def _sparse_parameter_gradient(self, mll: Any) -> Any:
        self._require_unbatched_ranking()
        original, sparse = self.raw_rho, self.is_sparse
        requires_grad = original.requires_grad
        try:
            self.to_dense()
            self.raw_rho.requires_grad_(True)
            mll.train()
            gradient = torch.autograd.grad(mll().sum(), self.raw_rho)[0]
            return gradient[..., ~self.is_active]
        finally:
            self._is_sparse = sparse
            self._parameters["raw_rho"] = original
            original.requires_grad_(requires_grad)

    def support_expansion(self, mll: Any, n: int = 1, modifier: Any = None) -> bool:
        self._require_unbatched_ranking()
        if n <= 0 or len(self.support) == self.dim:
            return False
        scores = self.expansion_objective(mll)
        modifier = self._expansion_modifier if modifier is None else modifier
        scores = scores if modifier is None else modifier(scores)
        if scores.shape != (self.dim - len(self.support),):
            raise ValueError("Expansion scores must have one value per inactive index.")
        selected = scores.argsort(descending=True)[:n]
        selected = self.inactive_indices[selected[scores[selected] > 0]]
        # These are discrete support labels, never covariance/gradient values.
        indices = selected.tolist()
        if not indices:
            return False
        self.expand_support(indices)
        return True

    def support_contraction(self, mll: Any, n: int = 1, modifier: Any = None) -> bool:
        self._require_unbatched_ranking()
        if n <= 0 or not self.support:
            return False
        values = self._dense_tensor()[..., self.support]
        modifier = self._contraction_modifier if modifier is None else modifier
        values = values if modifier is None else modifier(values)
        if values.shape != (len(self.support),):
            raise ValueError("Contraction scores must have one value per active index.")
        chosen = values.argsort()[:n].tolist()
        self.contract_support([self.support[i] for i in chosen])
        return True

    def optimize_mll(
        self,
        mll: Any,
        model_trace: Any = None,
        reset_parameters: bool = True,
        reset_dense_parameters: bool = False,
        closure: Any = None,
        optimizer: Any = None,
        closure_kwargs: Any = None,
        optimizer_kwargs: Any = None,
    ) -> Any:
        from botorch_rs.fit import fit_gpytorch_mll
        from botorch_rs.models.relevance_pursuit import initialize_dense_parameters

        self._require_unbatched_ranking()
        if reset_parameters:
            self.set_sparse_parameter(torch.zeros_like(self.raw_rho))
        if reset_dense_parameters:
            initialize_dense_parameters(mll.model)
        self.to_sparse()
        fit_gpytorch_mll(
            mll,
            closure=closure,
            optimizer=optimizer,
            closure_kwargs=closure_kwargs,
            optimizer_kwargs=optimizer_kwargs,
        )
        if model_trace is not None:
            model_trace.append(copy.deepcopy(mll.model))
        return mll


class TorchSparseOutlierGaussianLikelihood(_TorchModule, SparseOutlierGaussianLikelihood):
    def __init__(
        self,
        base_noise: Any,
        dim: int,
        outlier_indices: Any = None,
        rho_prior: Any = None,
        rho_constraint: Any = None,
        batch_shape: Any = None,
        convex_parameterization: bool = True,
        loo: bool = True,
    ) -> None:
        _TorchModule.__init__(self, base_noise.noise)
        self.noise_covar = TorchSparseOutlierNoise(
            base_noise,
            dim,
            outlier_indices,
            rho_prior,
            rho_constraint,
            batch_shape,
            convex_parameterization,
            loo,
        )
        self.batch_shape = self.noise_covar.batch_shape

    @property
    def noise_covar(self) -> Any:
        try:
            return self._modules["noise_covar"]
        except KeyError:
            raise AttributeError("noise_covar") from None

    @noise_covar.setter
    def noise_covar(self, value: Any) -> None:
        self._modules["noise_covar"] = value

    @property
    def noise(self) -> Any:
        return self.noise_covar.base_noise.noise

    def log_prior(self) -> Any:
        return self.noise_covar.log_prior()

    def raw_parameter_bounds(self) -> Any:
        return {
            "noise_covar." + name: bounds
            for name, bounds in self.noise_covar.raw_parameter_bounds().items()
        }

    def marginal(self, function_dist: Any, X: Any = None, **kwargs: Any) -> Any:
        from botorch_rs.posteriors._torch_backend import TorchGPyTorchPosterior

        if not hasattr(function_dist, "covariance_matrix"):
            raise UnsupportedError(
                "Sparse likelihood evaluation requires a Gaussian distribution; samplewise evaluation is unsupported."
            )
        mean, covariance = function_dist.mean, function_dist.covariance_matrix
        noise = self.noise_covar(
            X,
            shape=mean.shape,
            diag_K=covariance.diagonal(dim1=-2, dim2=-1)
            if self.noise_covar.convex_parameterization
            else None,
            **kwargs,
        ).diagonal()
        return TorchGPyTorchPosterior(
            mean.unsqueeze(-1), covariance + noise.diag_embed()
        ).distribution

    forward = marginal
    __call__ = nn.Module.__call__


class TorchSparseSingleTaskGP(TorchSingleTaskGP):
    def __init__(
        self,
        train_X: Any,
        train_Y: Any,
        train_Yvar: Any = None,
        likelihood: Any = None,
        covar_module: Any = None,
        mean_module: Any = None,
        outcome_transform: Any = DEFAULT,
        input_transform: Any = None,
        *,
        backend: str = "torch",
    ) -> None:
        if likelihood is None:
            raise ValueError("A sparse likelihood is required.")
        source = likelihood.noise_covar
        if source.dim != train_X.shape[-2]:
            raise ValueError("Sparse noise dim must equal the number of training observations.")
        if isinstance(likelihood, nn.Module):
            base = None
        else:
            base = source.base_noise
        TorchSingleTaskGP.__init__(
            self,
            train_X,
            train_Y,
            train_Yvar,
            base,
            covar_module,
            mean_module,
            outcome_transform,
            input_transform,
            backend=backend,
        )
        if isinstance(likelihood, nn.Module):
            self.likelihood = likelihood
        else:
            self.likelihood = TorchSparseOutlierGaussianLikelihood(
                self.likelihood.noise_covar,
                source.dim,
                source.support.copy(),
                source.rho_prior,
                source.raw_rho_constraint,
                self._aug_batch_shape,
                source.convex_parameterization,
                source.loo,
            )
            if not source.is_sparse:
                self.likelihood.noise_covar.to_dense()
            self.likelihood.noise_covar.set_sparse_parameter(
                train_X.new_tensor(source._raw)
                .expand(*self._aug_batch_shape, len(source._raw))
                .clone()
            )
        if self.likelihood.batch_shape != self._aug_batch_shape:
            raise ValueError(
                "Sparse likelihood batch_shape must equal the model's augmented batch shape."
            )

    def _noise_for(self, count: int) -> Any:
        return self.likelihood.noise_covar.base_variance(count)

    def _base_training_noise(self, X: Any, Y: Any, fixed: Any) -> Any:
        base = self.likelihood.noise_covar.base_variance(Y.shape[-1], fixed, X, Y.shape)
        return (
            base
            if self._noise_override is None
            else torch.where(self._noise_override.isnan(), base, self._noise_override)
        )

    def _sparse_noise(self, X: Any, Y: Any, covariance: Any, fixed: Any) -> Any:
        return self.likelihood.noise_covar(
            X,
            shape=Y.shape,
            diag_K=covariance.diagonal(dim1=-2, dim2=-1)
            if self.likelihood.noise_covar.convex_parameterization
            else None,
            noise=fixed,
        ).diagonal()

    def _exact_mll(self, output: Any = None, target: Any = None) -> Any:
        X, Y, fixed = self._training_values()
        if output is None:
            mean, covariance = self.mean_module(X), self._covariance(X, X)
        else:
            mean, covariance = output.mean, output.covariance_matrix
            Y = Y if target is None else target
        noise = self._sparse_noise(X, Y, covariance, self._base_training_noise(X, Y, fixed))
        return (
            _rs.tensor_exact_mll(Y, covariance, mean, noise, normalize=False) + self._log_prior()
        ) / Y.shape[-1]

    def _posterior_tensors(
        self, X: Any, observation_noise: Any = False, *, inputs_transformed: bool = False
    ) -> Any:
        if not isinstance(X, torch.Tensor) or X.ndim < 2 or X.shape[-1] != self._train_X.shape[-1]:
            raise ValueError("X must be a Tensor with shape (..., q, d).")
        if X.dtype != self._train_X.dtype or X.device != self._train_X.device:
            raise ValueError("X must share the model's dtype and device.")
        self.eval()
        train_X, Y, fixed = self._training_values()
        query = X if inputs_transformed else self._augment_X(self.transform_inputs(X))
        covariance = self._covariance(train_X, train_X)
        # Training inputs always establish the match cache before query noise.
        self.likelihood.noise_covar._cached_train_inputs = train_X.detach().clone()
        noise = self._sparse_noise(
            train_X, Y, covariance, self._base_training_noise(train_X, Y, fixed)
        )
        factor, alpha = _rs.tensor_exact_gp_factor(Y, covariance, self.mean_module(train_X), noise)
        kqq = self._covariance(query, query)
        mean, posterior_covariance = _rs.tensor_exact_gp_predict(
            factor,
            alpha,
            self._covariance(train_X, query),
            kqq,
            self.mean_module(query),
        )
        query_noise = None
        if isinstance(observation_noise, torch.Tensor):
            query_noise = self._augment_Y(observation_noise)
        elif observation_noise is True:
            query_noise = self._sparse_noise(
                query,
                mean,
                posterior_covariance,
                fixed if fixed is not None and query.shape[-2] == fixed.shape[-1] else None,
            )
        elif observation_noise is not False:
            raise TypeError("observation_noise must be bool or a Tensor.")
        if query_noise is not None:
            posterior_covariance = posterior_covariance + query_noise.diag_embed()
        return mean, posterior_covariance

    def condition_on_observations(self, X: Any, Y: Any, noise: Any = None, **kwargs: Any) -> Any:
        if kwargs:
            raise UnsupportedError(f"Unsupported conditioning arguments: {sorted(kwargs)}.")
        if (
            not isinstance(X, torch.Tensor)
            or not isinstance(Y, torch.Tensor)
            or X.dtype != self._train_X.dtype
            or Y.dtype != self._train_Y.dtype
            or X.device != self._train_X.device
            or Y.device != self._train_Y.device
        ):
            raise ValueError("Conditioning data must share the model's dtype and device.")
        if (
            X.ndim < 2
            or Y.ndim < 2
            or X.shape[-1] != self._train_X.shape[-1]
            or X.shape[-2] != Y.shape[-2]
            or Y.shape[-1] != self.num_outputs
        ):
            raise ValueError("Conditioning input and target dimensions must match the model.")
        if not bool(torch.isfinite(X).all()) or not bool(torch.isfinite(Y).all()):
            raise ValueError("Conditioning inputs and targets must be finite.")
        batch = torch.broadcast_shapes(
            self.batch_shape,
            X.shape[:-2],
            Y.shape[:-2],
            noise.shape[:-2] if isinstance(noise, torch.Tensor) and noise.ndim >= 2 else (),
        )
        X = X.expand(*batch, *X.shape[-2:])
        Y = Y.expand(*batch, *Y.shape[-2:])
        if noise is not None:
            if (
                not isinstance(noise, torch.Tensor)
                or noise.dtype != Y.dtype
                or noise.device != Y.device
            ):
                raise ValueError("Conditioning noise must share the target dtype and device.")
            if not bool((torch.isfinite(noise) & (noise >= 0)).all()):
                raise ValueError("Conditioning noise must be finite and nonnegative.")
            noise = noise.expand_as(Y)
        model = TorchSingleTaskGP.condition_on_observations(self, X, Y, noise=noise)
        old = self.likelihood.noise_covar
        # A dense old parameter represents all old observations, including any
        # currently inactive entries. Represent it as an explicit old-row support
        # so its exact persistent leaf can remain shared with the original model.
        support = old.support.copy() if old.is_sparse else list(range(old.dim))
        base = old.base_noise
        if model._train_Yvar is not None:
            fixed = model._training_values()[2]
            base = TorchFixedNoiseLikelihood(fixed, model._aug_batch_shape, model._train_X)
        likelihood = TorchSparseOutlierGaussianLikelihood(
            base,
            model._train_X.shape[-2],
            support,
            getattr(old, "rho_prior", None),
            old.raw_rho_constraint,
            old.batch_shape,
            old.convex_parameterization,
            old.loo,
        )
        likelihood.noise_covar._parameters["raw_rho"] = old.raw_rho
        likelihood.noise_covar._ranking_batch_shape = model._aug_batch_shape
        model.likelihood = likelihood
        model._prediction_cache = None
        for name in ("bmc_support_sizes", "bmc_probabilities", "model_trace"):
            if hasattr(model, name):
                setattr(model, name, None)
        return model.eval()

    def fantasize(self, X: Any, sampler: Any, observation_noise: Any = None, **kwargs: Any) -> Any:
        propagate = kwargs.pop("propagate_grads", False)
        if kwargs:
            raise UnsupportedError(f"Unsupported fantasy arguments: {sorted(kwargs)}.")
        if isinstance(observation_noise, bool):
            raise TypeError("fantasize observation_noise must be None or a Tensor.")
        if observation_noise is None:
            _, _, fixed = self._training_values()
            query = self._augment_X(self.transform_inputs(X))
            base = (
                fixed.mean(-1, keepdim=True)
                if fixed is not None
                else self.likelihood.noise_covar.base_variance(
                    query.shape[-2], X=query, shape=query.shape[:-1]
                )
            )
            batch = torch.broadcast_shapes(self._aug_batch_shape, query.shape[:-2], base.shape[:-1])
            base = base.expand(*batch, query.shape[-2])
            # New observation labels have zero outlier variance, even if their
            # coordinates happen to equal the existing training inputs.
            observation_noise = (
                base.transpose(-1, -2) if self.num_outputs > 1 else base.unsqueeze(-1)
            )
        posterior = self.posterior(X, observation_noise=observation_noise)
        Y = sampler(posterior)
        if not propagate:
            Y = Y.detach()
        return self.condition_on_observations(X, Y, noise=observation_noise)

    def set_train_data(self, inputs: Any = None, targets: Any = None, strict: bool = True) -> None:
        X = self._train_X if inputs is None else inputs[0] if isinstance(inputs, tuple) else inputs
        if X.shape[-2] != self.likelihood.noise_covar.dim:
            raise UnsupportedError(
                "Changing the training count requires reconstructing the sparse support."
            )
        super().set_train_data(inputs, targets, strict)
        self.likelihood.noise_covar._ranking_batch_shape = self._aug_batch_shape


class TorchRobustRelevancePursuitGP(TorchSparseSingleTaskGP, RobustRelevancePursuitSingleTaskGP):
    def __init__(
        self,
        train_X: Any,
        train_Y: Any,
        train_Yvar: Any = None,
        likelihood: Any = None,
        covar_module: Any = None,
        mean_module: Any = None,
        outcome_transform: Any = DEFAULT,
        input_transform: Any = None,
        convex_parameterization: bool = True,
        prior_mean_of_support: Any = None,
        cache_model_trace: bool = False,
        *,
        backend: str = "torch",
    ) -> None:
        external = isinstance(likelihood, nn.Module)
        if external and (train_Yvar is not None or not hasattr(likelihood, "noise_covar")):
            raise UnsupportedError("Supply a Gaussian noise likelihood or train_Yvar, not both.")
        TorchSingleTaskGP.__init__(
            self,
            train_X,
            train_Y,
            train_Yvar,
            None if external else likelihood,
            covar_module,
            mean_module,
            outcome_transform,
            input_transform,
            backend=backend,
        )
        mean = (
            float(max(1, int(0.2 * train_X.shape[-2])))
            if prior_mean_of_support is None
            else float(prior_mean_of_support)
        )
        if not math.isfinite(mean) or mean <= 0:
            raise ValueError("prior_mean_of_support must be finite and positive.")
        self.likelihood = TorchSparseOutlierGaussianLikelihood(
            likelihood.noise_covar if external else self.likelihood.noise_covar,
            train_X.shape[-2],
            batch_shape=self._aug_batch_shape,
            convex_parameterization=convex_parameterization,
        )
        self.prior_mean_of_support = mean
        self.cache_model_trace = bool(cache_model_trace)
        self.bmc_support_sizes = self.bmc_probabilities = self.model_trace = None

    def to_standard_model(self) -> Any:
        model: Any = copy.deepcopy(self)
        model.__class__ = TorchSparseSingleTaskGP
        return model

    def load_standard_model(self, standard_model: Any) -> Any:
        metadata = {
            key: getattr(self, key)
            for key in (
                "prior_mean_of_support",
                "cache_model_trace",
                "bmc_support_sizes",
                "bmc_probabilities",
                "model_trace",
            )
        }
        state = copy.deepcopy(standard_model).__dict__
        self.__dict__.clear()
        self.__dict__.update(state)
        self.__dict__.update(metadata)
        return self

    def custom_fit(
        self,
        mll: Any,
        *,
        numbers_of_outliers: Any = None,
        fractions_of_outliers: Any = None,
        timeout_sec: Any = None,
        relevance_pursuit_optimizer: Any = None,
        reset_parameters: bool = True,
        reset_dense_parameters: bool = False,
        closure: Any = None,
        optimizer: Any = None,
        closure_kwargs: Any = None,
        optimizer_kwargs: Any = None,
    ) -> Any:
        from botorch_rs.mlls import ExactMarginalLogLikelihood
        from botorch_rs.models.relevance_pursuit import (
            backward_relevance_pursuit,
            get_posterior_over_support,
        )

        if not isinstance(mll, ExactMarginalLogLikelihood):
            raise UnsupportedError("Robust relevance pursuit requires an exact Gaussian MLL.")
        self.likelihood.noise_covar._require_unbatched_ranking()
        if numbers_of_outliers is None:
            fractions = (
                FRACTIONS_OF_OUTLIERS
                if fractions_of_outliers is None
                else list(fractions_of_outliers)
            )
            if any(not math.isfinite(float(p)) or not 0 <= p <= 1 for p in fractions):
                raise ValueError("fractions_of_outliers must lie in [0, 1].")
            numbers_of_outliers = [int(float(p) * self._train_X.shape[-2]) for p in fractions]
        numbers_of_outliers = list(numbers_of_outliers)
        if not numbers_of_outliers:
            raise ValueError("numbers_of_outliers must not be empty.")
        options = dict(optimizer_kwargs or {})
        if timeout_sec is not None:
            if not math.isfinite(float(timeout_sec)) or timeout_sec < 0:
                raise ValueError("timeout_sec must be finite and nonnegative.")
            # Match the official contract: apportioned timeout per numerical fit,
            # not a background worker or a strict wall-clock interruption of EM/
            # support scoring/model comparison outside those numerical fits.
            options["timeout_sec"] = float(timeout_sec) / len(numbers_of_outliers)
        original_model, original_likelihood = mll.model, mll.likelihood
        try:
            mll.model = self.to_standard_model()
            mll.likelihood = mll.model.likelihood
            _, trace = (
                backward_relevance_pursuit
                if relevance_pursuit_optimizer is None
                else relevance_pursuit_optimizer
            )(
                sparse_module=mll.model.likelihood.noise_covar,
                mll=mll,
                sparsity_levels=numbers_of_outliers,
                record_model_trace=True,
                reset_parameters=reset_parameters,
                reset_dense_parameters=reset_dense_parameters,
                closure=closure,
                optimizer=optimizer,
                closure_kwargs=closure_kwargs,
                optimizer_kwargs=options,
            )
            sizes, probabilities = get_posterior_over_support(
                SparseOutlierNoise, trace, prior_mean_of_support=self.prior_mean_of_support
            )
            selected = int(probabilities.argmax())
            self.load_standard_model(trace[selected])
            self.bmc_support_sizes, self.bmc_probabilities = sizes.detach(), probabilities.detach()
            self.model_trace = trace if self.cache_model_trace else None
        finally:
            mll.model = original_model
            mll.likelihood = self.likelihood if original_model is self else original_likelihood
        return mll.eval()

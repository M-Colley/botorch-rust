"""Device-resident preference GPs with an implicitly differentiated Laplace mode."""

from __future__ import annotations

import copy
import math
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from botorch_rs import _botorch_rs as _rs
from botorch_rs.exceptions.errors import UnsupportedError
from botorch_rs.means import ConstantMean
from botorch_rs.models._torch_backend_modules import (
    build_input_transform,
    build_kernel,
    build_mean,
    module_log_prior,
)
from botorch_rs.models.likelihoods.pairwise import (
    PairwiseLikelihood,
    PairwiseLogitLikelihood,
    PairwiseProbitLikelihood,
)
from botorch_rs.models.pairwise_gp import (
    PairwiseGP,
    PairwiseLaplaceMarginalLogLikelihood,
    _default_covar_module,
)
from botorch_rs.posteriors._torch_backend import TorchGPyTorchPosterior


def _likelihood_terms(
    utility: Tensor, D: Tensor, kind: str, limit: float, *, root_jacobian: bool = False
) -> tuple[Tensor, Tensor, Tensor]:
    difference = (D @ utility.unsqueeze(-1)).squeeze(-1)
    if kind == "probit":
        raw_z = difference / math.sqrt(2)
        z = raw_z.clamp(-limit, limit)
        log_p = torch.special.log_ndtr(z)
        hazard = (-0.5 * z.square() - 0.5 * math.log(2 * math.pi) - log_p).exp()
        gradient = -(D.transpose(-1, -2) @ (hazard / math.sqrt(2)).unsqueeze(-1)).squeeze(-1)
        curvature = hazard * (hazard + z) / 2
        if root_jacobian:
            # The source gradient clamps z but its supplied Hessian omits the
            # clamp derivative. The implicit solve needs the actual Jacobian
            # of the solved gradient equation, including saturated regions.
            curvature = curvature * ((raw_z >= -limit) & (raw_z <= limit)).to(curvature)
    else:
        log_p = F.logsigmoid(difference.clamp(-limit, limit))
        probability = torch.sigmoid(-difference)
        gradient = -(D.transpose(-1, -2) @ probability.unsqueeze(-1)).squeeze(-1)
        curvature = probability * torch.sigmoid(difference)
    hessian = (D.transpose(-1, -2) * curvature.unsqueeze(-2)) @ D
    return log_p, gradient, hessian


class _ImplicitMode(torch.autograd.Function):
    """Root of K^{-1}(f-m)+g(f)=0, with differentiable implicit adjoints.

    Saving the returned root (rather than a detached intermediate) makes its
    dependence available when differentiating the adjoint itself. This is
    required for second and higher derivatives through the converged mode.
    """

    @staticmethod
    def forward(
        ctx: Any,
        covariance: Tensor,
        mean: Tensor,
        D: Tensor,
        kind: str,
        limit: float,
        tolerance: float,
        max_iterations: int,
    ) -> Tensor:
        n = covariance.shape[-1]
        identity = torch.eye(n, device=covariance.device, dtype=covariance.dtype)
        inverse = torch.linalg.solve(covariance, identity)
        utility = mean.clone()
        threshold = max(tolerance, 20 * torch.finfo(covariance.dtype).eps)

        def equation(value: Tensor) -> tuple[Tensor, Tensor]:
            _, gradient, hessian = _likelihood_terms(value, D, kind, limit, root_jacobian=True)
            residual = (inverse @ (value - mean).unsqueeze(-1)).squeeze(-1) + gradient
            return residual, inverse + hessian

        for _ in range(max_iterations):
            residual, jacobian = equation(utility)
            if bool((residual.abs().amax(-1) <= threshold).all()):
                break
            direction = torch.linalg.solve(jacobian, residual.unsqueeze(-1)).squeeze(-1)
            old_norm = residual.square().sum(-1)
            step = torch.ones_like(old_norm)
            accepted = old_norm <= threshold**2
            candidate = utility
            for _ in range(24):
                trial = utility - step.unsqueeze(-1) * direction
                trial_residual, _ = equation(trial)
                improved = trial_residual.square().sum(-1) <= old_norm * (1 - 1e-4 * step)
                improved = improved | (trial_residual.abs().amax(-1) <= threshold)
                select = (~accepted) & improved
                candidate = torch.where(select.unsqueeze(-1), trial, candidate)
                accepted = accepted | improved
                if bool(accepted.all()):
                    break
                step = step / 2
            utility = candidate
        residual, _ = equation(utility)
        if not bool(torch.isfinite(utility).all()) or not bool(
            (residual.abs().amax(-1) <= threshold * 10).all()
        ):
            raise RuntimeError("Device Laplace utility solve did not converge; increase maxfev.")
        ctx.kind, ctx.limit = kind, limit
        ctx.save_for_backward(covariance, mean, D, utility)
        return utility

    @staticmethod
    def backward(ctx: Any, gradient: Tensor) -> tuple[Any, ...]:
        covariance, mean, D, utility = ctx.saved_tensors
        identity = torch.eye(covariance.shape[-1], dtype=covariance.dtype, device=covariance.device)
        inverse = torch.linalg.solve(covariance, identity)
        _, _, curvature = _likelihood_terms(utility, D, ctx.kind, ctx.limit, root_jacobian=True)
        adjoint = torch.linalg.solve(
            (inverse + curvature).transpose(-1, -2), gradient.unsqueeze(-1)
        )
        projected = torch.linalg.solve(covariance.transpose(-1, -2), adjoint)
        alpha = torch.linalg.solve(covariance, (utility - mean).unsqueeze(-1))
        covariance_gradient = projected @ alpha.transpose(-1, -2)
        return covariance_gradient, projected.squeeze(-1), None, None, None, None, None


class TorchPairwiseLikelihood(nn.Module):
    _tensor_backend = "torch"

    def __init__(self, source: PairwiseLikelihood | None = None) -> None:
        super().__init__()
        if source is None or type(source) is PairwiseProbitLikelihood:
            self.kind, self.limit = "probit", float(getattr(source, "_zlim", 3))
        elif type(source) is PairwiseLogitLikelihood:
            self.kind, self.limit = "logit", float(source._logit_lim)
        elif isinstance(source, TorchPairwiseLikelihood):
            self.kind, self.limit = source.kind, source.limit
        else:
            raise UnsupportedError("Torch PairwiseGP supports the built-in probit and logit laws.")

    def log_p(self, utility: Tensor, D: Tensor) -> Tensor:
        return _likelihood_terms(utility, D.to(utility), self.kind, self.limit)[0]

    def p(self, utility: Tensor, D: Tensor) -> Tensor:
        return self.log_p(utility, D).exp()

    def negative_log_gradient_sum(self, utility: Tensor, D: Tensor) -> Tensor:
        return _likelihood_terms(utility, D.to(utility), self.kind, self.limit)[1]

    def negative_log_hessian_sum(self, utility: Tensor, D: Tensor) -> Tensor:
        return _likelihood_terms(utility, D.to(utility), self.kind, self.limit)[2]

    def forward(self, utility: Tensor, D: Tensor) -> Any:
        return torch.distributions.Bernoulli(probs=self.p(utility, D))


class TorchPairwiseProbitLikelihood(TorchPairwiseLikelihood, PairwiseProbitLikelihood):
    pass


class TorchPairwiseLogitLikelihood(TorchPairwiseLikelihood, PairwiseLogitLikelihood):
    pass


def _consolidate(X: Tensor, comparisons: Tensor, rtol: float, atol: float) -> Any:
    """Reference duplicate selection, with all index arrays kept on device."""
    if X.ndim != 2 or comparisons.ndim != 2:
        return X, comparisons, None
    n = X.shape[-2]
    distance = (X.unsqueeze(-2) - X.unsqueeze(-3)).abs().amax(-1)
    maximum = X.abs().amax(-1)
    tolerance = atol + rtol * torch.maximum(maximum.unsqueeze(-1), maximum.unsqueeze(-2))
    adjusted = (distance - tolerance).masked_fill(
        ~torch.ones(n, n, device=X.device, dtype=torch.bool).triu(1), torch.inf
    )
    minimum, original = adjusted.min(-2)
    duplicate = minimum < 0
    used = torch.zeros(n, device=X.device, dtype=torch.long)
    used.scatter_add_(0, original, duplicate.long())
    common = duplicate & (used > 0)
    duplicate = duplicate & ~common & ~common[original]
    if not bool(duplicate.any()):
        return X, comparisons, torch.arange(n, dtype=comparisons.dtype, device=X.device)
    keep = (~duplicate).nonzero().squeeze(-1)
    new_indices = (~duplicate).long().cumsum(0) - 1
    new_indices = torch.where(duplicate, new_indices[original], new_indices)
    return X.index_select(-2, keep), new_indices[comparisons], new_indices


class TorchPairwiseGP(nn.Module, PairwiseGP):
    """Preference GP with on-device Laplace inference and full mode autograd."""

    _tensor_backend = "torch"
    _num_outputs = 1
    dim: int | None = None
    n: int = 0
    m: int = 0
    input_transform: Any
    covar_module: Any
    mean_module: Any
    likelihood: Any
    consolidated_datapoints: Any
    consolidated_comparisons: Any
    unconsolidated_datapoints: Any
    unconsolidated_comparisons: Any
    consolidated_indices: Any
    D: Any
    DT: Any
    utility: Any
    covar: Any
    covar_chol: Any
    likelihood_hess: Any
    covar_inv: Any
    hlcov_eye: Any

    def __init__(
        self,
        datapoints: Tensor | None,
        comparisons: Tensor | None,
        likelihood: Any = None,
        covar_module: Any = None,
        input_transform: Any = None,
        *,
        jitter: float = 1e-6,
        xtol: float | None = None,
        consolidate_rtol: float = 0.0,
        consolidate_atol: float = 1e-4,
        maxfev: int | None = None,
        backend: str = "torch",
    ) -> None:
        nn.Module.__init__(self)
        if (
            not math.isfinite(jitter)
            or jitter <= 0
            or any(not math.isfinite(v) or v < 0 for v in (consolidate_rtol, consolidate_atol))
            or (xtol is not None and (not math.isfinite(xtol) or xtol <= 0))
            or (maxfev is not None and (type(maxfev) is not int or maxfev < 1))
        ):
            raise ValueError("Invalid jitter, solver tolerance, duplicate tolerance or maxfev.")
        self._jitter, self._xtol, self._maxfev = jitter, xtol, maxfev
        self._consolidate_rtol, self._consolidate_atol = consolidate_rtol, consolidate_atol
        if datapoints is not None and not isinstance(datapoints, Tensor):
            raise TypeError("Torch PairwiseGP requires Tensor datapoints.")
        like = datapoints if isinstance(datapoints, Tensor) else torch.zeros(1, 1)
        if datapoints is None and isinstance(covar_module, nn.Module):
            like = next(iter(covar_module.parameters()), like)
        self.dim = None if datapoints is None else datapoints.shape[-1]
        for name in PairwiseGP._buffer_names:
            self._buffers[name] = None
        batch = torch.Size() if datapoints is None else datapoints.shape[:-2]
        source = (
            _default_covar_module(self.dim, tuple(batch)) if covar_module is None else covar_module
        )
        self.covar_module = build_kernel(source, self.dim or 1, batch, like)
        if not hasattr(self.covar_module, "outputscale"):
            raise UnsupportedError("PairwiseGP must be used with a ScaleKernel.")
        parameter_batch = getattr(self.covar_module, "batch_shape", ())
        if datapoints is not None and torch.broadcast_shapes(batch, parameter_batch) != batch:
            raise UnsupportedError(
                "Pairwise kernel parameter batches must broadcast to the training batch."
            )
        self.mean_module = build_mean(ConstantMean(), (), like)
        self.mean_module.requires_grad_(False)
        likelihood_type = (
            TorchPairwiseLogitLikelihood
            if isinstance(likelihood, PairwiseLogitLikelihood)
            else TorchPairwiseProbitLikelihood
        )
        self.likelihood = likelihood_type(likelihood)
        self.input_transform = build_input_transform(input_transform, self.dim or 1, like)
        self.pred_cov_fac_need_update = True
        self._train_inputs: tuple[Any, ...] = ()
        self._train_targets: Any = None
        self.set_train_data(
            datapoints, comparisons, update_model=datapoints is not None and comparisons is not None
        )

    @property
    def train_inputs(self) -> Any:
        return self._train_inputs

    @property
    def train_targets(self) -> Any:
        return self._train_targets

    @property
    def num_outputs(self) -> int:
        return 1

    @property
    def batch_shape(self) -> torch.Size:
        return torch.Size() if self.datapoints is None else self.datapoints.shape[:-2]

    @property
    def datapoints(self) -> Any:
        return self.consolidated_datapoints

    @property
    def comparisons(self) -> Any:
        return self.consolidated_comparisons

    @property
    def unconsolidated_utility(self) -> Any:
        return (
            self.utility
            if self.consolidated_indices is None
            else self.utility[self.consolidated_indices]
        )

    def _has_no_data(self) -> bool:
        return self.datapoints is None or self.comparisons is None

    def transform_inputs(self, X: Tensor, input_transform: Any = None) -> Tensor:
        transform = self.input_transform if input_transform is None else input_transform
        return X if transform is None else transform(X)

    def _fit_transform(self) -> None:
        transform = self.input_transform
        if transform is None or self.datapoints is None:
            return
        if getattr(transform, "_tensor_backend", None) == "torch":
            if getattr(transform, "transform_on_train", True) and hasattr(transform, "fit"):
                transform.fit(self.datapoints)
        transform.eval()

    def _calc_covar(self, X1: Tensor, X2: Tensor) -> Tensor:
        result = self.covar_module(X1, X2)
        result = result if isinstance(result, Tensor) else result.to_dense()
        if X1 is X2:
            scale = self.covar_module.outputscale.detach().unsqueeze(-1).unsqueeze(-1)
            factor = _rs.tensor_psd_safe_cholesky(result / scale, jitter=self._jitter)
            result = (factor @ factor.transpose(-1, -2)) * scale
        return result

    def _update(self, datapoints: Tensor | None = None, **kwargs: Any) -> None:
        if kwargs:
            raise UnsupportedError(f"Unsupported mode solver arguments: {sorted(kwargs)}.")
        if self._has_no_data():
            return
        if datapoints is None:
            # Rebuild selected data from their original source every time;
            # cached index_select graphs cannot survive independent backwards.
            self.set_train_data(
                self.unconsolidated_datapoints,
                self.unconsolidated_comparisons,
                update_model=False,
            )
            self._fit_transform()
            datapoints = self.transform_inputs(self.datapoints)
        self.covar = self._calc_covar(datapoints, datapoints)
        self.covar_chol = torch.linalg.cholesky(self.covar)
        mean = self.mean_module(datapoints)
        batch = torch.broadcast_shapes(self.covar.shape[:-2], mean.shape[:-1])
        covariance = self.covar.expand(*batch, self.n, self.n)
        mean = mean.expand(*batch, self.n)
        D = self.D.expand(*batch, self.m, self.n)
        self.utility = _ImplicitMode.apply(
            covariance,
            mean,
            D,
            self.likelihood.kind,
            self.likelihood.limit,
            min(self._xtol or 1e-6, 1e-11),
            self._maxfev or 100,
        )
        self.likelihood_hess = self.likelihood.negative_log_hessian_sum(self.utility, D)
        self.covar_inv = torch.cholesky_inverse(self.covar_chol)
        eye = torch.eye(self.n, dtype=covariance.dtype, device=covariance.device)
        self.hlcov_eye = self.likelihood_hess @ covariance + eye
        self.pred_cov_fac_need_update = False

    def set_train_data(
        self,
        datapoints: Any = None,
        comparisons: Any = None,
        strict: bool = False,
        update_model: bool = True,
    ) -> None:
        if datapoints is None or comparisons is None:
            self.unconsolidated_datapoints, self.unconsolidated_comparisons = (
                datapoints,
                comparisons,
            )
            self.consolidated_datapoints, self.consolidated_comparisons = datapoints, comparisons
            return
        if not isinstance(datapoints, Tensor) or not isinstance(comparisons, Tensor):
            raise TypeError("Torch PairwiseGP requires Tensor datapoints and comparisons.")
        if (
            datapoints.ndim < 2
            or not datapoints.shape[-2]
            or not datapoints.shape[-1]
            or datapoints.dtype not in (torch.float32, torch.float64)
            or not bool(torch.isfinite(datapoints).all())
        ):
            raise ValueError("datapoints must be finite float32/float64 (...,n,d).")
        if (
            comparisons.ndim != datapoints.ndim
            or comparisons.shape[-1] != 2
            or comparisons.shape[:-2] != datapoints.shape[:-2]
            or comparisons.device != datapoints.device
            or not bool(torch.isfinite(comparisons).all())
            or bool((comparisons != comparisons.long()).any())
            or bool(((comparisons < 0) | (comparisons >= datapoints.shape[-2])).any())
        ):
            raise ValueError("comparisons must be in-range integer (...,m,2) on the data device.")
        if strict and self.unconsolidated_datapoints is not None:
            for new, old in (
                (datapoints, self.unconsolidated_datapoints),
                (comparisons, self.unconsolidated_comparisons),
            ):
                if (new.shape, new.dtype, new.device) != (old.shape, old.dtype, old.device):
                    raise RuntimeError("Cannot modify shape, dtype or device under strict=True.")
        comparisons = comparisons.long()
        self.unconsolidated_datapoints, self.unconsolidated_comparisons = datapoints, comparisons
        self._train_inputs, self._train_targets = (datapoints,), comparisons
        self.consolidated_datapoints, self.consolidated_comparisons, self.consolidated_indices = (
            _consolidate(datapoints, comparisons, self._consolidate_rtol, self._consolidate_atol)
        )
        self.dim, self.n, self.m = (
            datapoints.shape[-1],
            self.datapoints.shape[-2],
            self.comparisons.shape[-2],
        )
        self.D = datapoints.new_zeros(*self.batch_shape, self.m, self.n)
        signs = torch.tensor(
            [1.0, -1.0], device=datapoints.device, dtype=datapoints.dtype
        ).expand_as(self.comparisons)
        self.D.scatter_add_(-1, self.comparisons, signs)
        self.DT = self.D.transpose(-1, -2)
        if update_model:
            self._update()

    def forward(self, datapoints: Tensor) -> Any:
        if self.training:
            if self._has_no_data():
                raise RuntimeError("Training requires datapoints and comparisons.")
            if datapoints is not self.unconsolidated_datapoints:
                raise RuntimeError("Must train on training data")
            self._update()
            covariance = self.covar - self.covar @ torch.linalg.solve(
                self.hlcov_eye, self.likelihood_hess @ self.covar
            )
            mean = self.utility
        else:
            if not self._has_no_data():
                self._update()
            else:
                self._fit_transform()
            query = self.transform_inputs(datapoints)
            mean = self.mean_module(query)
            covariance = self._calc_covar(query, query)
            if not self._has_no_data():
                train = self.transform_inputs(self.datapoints)
                cross = self._calc_covar(query, train)
                residual = self.utility - self.mean_module(train)
                mean = mean + (
                    cross @ torch.cholesky_solve(residual.unsqueeze(-1), self.covar_chol)
                ).squeeze(-1)
                covariance = covariance - cross @ torch.linalg.solve(
                    self.hlcov_eye, self.likelihood_hess @ cross.transpose(-1, -2)
                )
        covariance = (covariance + covariance.transpose(-1, -2)) / 2
        scale = self.covar_module.outputscale.detach().unsqueeze(-1).unsqueeze(-1)
        factor = _rs.tensor_psd_safe_cholesky(covariance / scale, jitter=self._jitter)
        covariance = (factor @ factor.transpose(-1, -2)) * scale
        output = TorchGPyTorchPosterior(mean.unsqueeze(-1).clone(), covariance).distribution
        if self.training:
            prior: Any = 0.0
            for module in (self.covar_module, self.mean_module, self.input_transform):
                if module is not None:
                    part = module_log_prior(module, self.batch_shape)
                    prior = prior + (part.sum() if isinstance(part, Tensor) else part)
            output._pairwise_state = {
                "owner": self,
                "comparisons": self.unconsolidated_comparisons,
                "D": self.D,
                "cholesky": self.covar_chol,
                "prior_mean": self.mean_module(self.transform_inputs(self.datapoints)).clone(),
                "hlcov_eye": self.hlcov_eye,
                "log_prior": prior,
                "kind": self.likelihood.kind,
                "limit": self.likelihood.limit,
            }
        return output

    def posterior(
        self,
        X: Tensor,
        output_indices: Any = None,
        observation_noise: Any = False,
        posterior_transform: Any = None,
        **kwargs: Any,
    ) -> Any:
        if kwargs:
            raise UnsupportedError(f"Unsupported posterior arguments: {sorted(kwargs)}.")
        if output_indices is not None:
            raise RuntimeError("output_indices must be None for the single-output PairwiseGP.")
        if (
            not isinstance(X, Tensor)
            or X.ndim < 2
            or (self.dim is not None and X.shape[-1] != self.dim)
        ):
            raise ValueError("X must have shape (...,q,d) matching the model.")
        like = next(self.parameters())
        if X.dtype != like.dtype or X.device != like.device or not bool(torch.isfinite(X).all()):
            raise ValueError("X must be finite and have the model dtype/device.")
        self.eval()
        distribution = self(X)
        posterior = TorchGPyTorchPosterior(
            distribution.mean.unsqueeze(-1), distribution.covariance_matrix
        )
        return (
            posterior
            if posterior_transform is None
            else posterior_transform(posterior=posterior, X=X)
        )

    def condition_on_observations(self, X: Tensor, Y: Tensor, **kwargs: Any) -> Any:
        if kwargs:
            raise UnsupportedError(f"Unsupported conditioning arguments: {sorted(kwargs)}.")
        if not isinstance(Y, Tensor) or Y.is_floating_point() or Y.shape[-1] != 2:
            raise RuntimeError("Conditioning requires integer pairwise comparison observations.")
        model = copy.deepcopy(self)
        if self._has_no_data():
            model.set_train_data(X, Y)
        else:
            batch = torch.broadcast_shapes(self.batch_shape, X.shape[:-2], Y.shape[:-2])
            data = torch.cat(
                (self.datapoints.expand(*batch, self.n, self.dim), X.expand(*batch, *X.shape[-2:])),
                -2,
            )
            comparisons = torch.cat(
                (
                    self.comparisons.expand(*batch, self.m, 2),
                    (Y + self.n).expand(*batch, *Y.shape[-2:]),
                ),
                -2,
            )
            model.set_train_data(data, comparisons)
        return model.eval()

    def __deepcopy__(self, memo: Any) -> Any:
        # nn.Module's ordinary deepcopy cannot copy nonleaf derived buffers.
        # Rebuild the numerical state from copied leaves/data instead.
        for module in (self.covar_module, self.input_transform, self.mean_module):
            if module is not None:
                for buffer in module.buffers():
                    if not buffer.is_leaf:
                        memo.setdefault(id(buffer), buffer.detach().clone())
        result = type(self)(
            self.unconsolidated_datapoints.clone()
            if self.unconsolidated_datapoints is not None
            else None,
            self.unconsolidated_comparisons.clone()
            if self.unconsolidated_comparisons is not None
            else None,
            copy.deepcopy(self.likelihood, memo),
            copy.deepcopy(self.covar_module, memo),
            copy.deepcopy(self.input_transform, memo),
            jitter=self._jitter,
            xtol=self._xtol,
            consolidate_rtol=self._consolidate_rtol,
            consolidate_atol=self._consolidate_atol,
            maxfev=self._maxfev,
        )
        memo[id(self)] = result
        result.mean_module = copy.deepcopy(self.mean_module, memo)
        return result.train(self.training)

    def __repr__(self) -> str:
        return f"TorchPairwiseGP(n={0 if self._has_no_data() else self.n}, batch_shape={tuple(self.batch_shape)})"


def _buffer_property(name: str) -> property:
    return property(lambda self: self._buffers.get(name))


for _buffer_name in PairwiseGP._buffer_names:
    setattr(TorchPairwiseGP, _buffer_name, _buffer_property(_buffer_name))


class TorchPairwiseLaplaceMarginalLogLikelihood(nn.Module, PairwiseLaplaceMarginalLogLikelihood):
    _tensor_backend = "torch"
    likelihood: Any
    model: Any

    def __init__(self, likelihood: Any, model: TorchPairwiseGP) -> None:
        nn.Module.__init__(self)
        if likelihood is not model.likelihood:
            raise UnsupportedError("The Laplace MLL requires this PairwiseGP's likelihood.")
        self.likelihood, self.model = likelihood, model

    def forward(self, post: Any = None, comp: Any = None, **kwargs: Any) -> Tensor:
        if kwargs:
            raise UnsupportedError(f"Unsupported Laplace MLL arguments: {sorted(kwargs)}.")
        model = self.model
        if post is None and comp is None:
            if model._has_no_data():
                raise RuntimeError("Training requires datapoints and comparisons.")
            model.train()
            post = model(model.unconsolidated_datapoints)
            comp = model.unconsolidated_comparisons
        state = getattr(post, "_pairwise_state", None)
        if state is None or state["owner"] is not model:
            raise UnsupportedError("Laplace MLL requires a training output from this PairwiseGP.")
        if comp is not state["comparisons"]:
            raise RuntimeError("Must train on training data")
        utility = post.mean
        log_likelihood = _likelihood_terms(utility, state["D"], state["kind"], state["limit"])[
            0
        ].sum(-1)
        residual = utility - state["prior_mean"]
        quadratic = 0.5 * (
            residual.unsqueeze(-2) @ torch.cholesky_solve(residual.unsqueeze(-1), state["cholesky"])
        ).squeeze(-1).squeeze(-1)
        value = (
            -(quadratic - log_likelihood).clamp_min(0)
            - 0.5 * torch.linalg.slogdet(state["hlcov_eye"]).logabsdet
        ).sum()
        return value + state["log_prior"]

    def value_and_grad(self, raw_parameters: Any = None) -> Any:
        if raw_parameters is not None:
            raise UnsupportedError("Set actual Torch Parameters before evaluating value_and_grad.")
        parameters = tuple(self.parameters())
        trainable = tuple(p for p in parameters if p.requires_grad)
        value = self()
        gradients = torch.autograd.grad(value, trainable, allow_unused=True) if trainable else ()
        by_identity = dict(zip(map(id, trainable), gradients, strict=True))
        parts = [
            (torch.zeros_like(p) if by_identity.get(id(p)) is None else by_identity[id(p)]).reshape(
                -1
            )
            for p in parameters
        ]
        return float(value.detach()), torch.cat(parts).detach().cpu().tolist() if parts else []

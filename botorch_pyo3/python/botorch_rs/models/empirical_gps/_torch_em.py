"""Device-local EM empirical priors and exact Gaussian conditioning.

The EM loop is expressed using Torch tensors; Gaussian factorizations are
dispatched by Rust. No historical numerical work delegates to BoTorch.
"""

from __future__ import annotations

import copy
import math
from typing import Any, cast

import torch
from torch import nn

from botorch_rs import _botorch_rs as _rs
from botorch_rs.exceptions.errors import UnsupportedError
from botorch_rs.models._torch_backend import TorchSingleTaskGP
from botorch_rs.models._torch_backend_modules import build_kernel, build_mean, module_log_prior
from botorch_rs.models.empirical_gps.em_empirical_gp import (
    EMEmpiricalGaussianProcess,
    EMPriorContainer,
)
from botorch_rs.models.empirical_gps.utils import ExperimentDataset, UniqueInputs
from botorch_rs.posteriors._torch_backend import TorchGPyTorchPosterior


def dense(module: Any, X: Any, Z: Any) -> Any:
    result = module(X, Z)
    return result if isinstance(result, torch.Tensor) else result.to_dense()


def gaussian_factor(Y: Any, covariance: Any, mean: Any, noise: Any) -> tuple[Any, Any]:
    factor = _rs.tensor_psd_safe_cholesky(
        covariance + torch.diag_embed(noise.expand(*covariance.shape[:-1]))
    )
    alpha = torch.cholesky_solve((Y - mean).unsqueeze(-1), factor).squeeze(-1)
    return factor, alpha


def gaussian_log_prob(Y: Any, covariance: Any, mean: Any, noise: Any) -> Any:
    factor, alpha = gaussian_factor(Y, covariance, mean, noise)
    return -0.5 * (
        ((Y - mean) * alpha).sum(-1)
        + 2 * factor.diagonal(dim1=-2, dim2=-1).log().sum(-1)
        + Y.shape[-1] * math.log(2 * math.pi)
    )


def mean_at(module: Any, X: Any) -> Any:
    value = module(X)
    return value.squeeze(-1) if value.shape == (*X.shape[:-1], 1) else value


def validate_datasets(datasets: Any, like: Any = None) -> Any:
    if not isinstance(datasets, (tuple, list)) or not datasets:
        raise ValueError("At least one historical dataset is required.")
    reference = datasets[0].X if like is None else like
    for data in datasets:
        TorchSingleTaskGP._validate_data(data.X, data.Y)
        if data.X.ndim != 2 or data.Y.shape[-1] != 1:
            raise ValueError("EM histories require unbatched n x d inputs and n x 1 targets.")
        if data.X.shape[-1] != reference.shape[-1]:
            raise ValueError("Historical feature dimensions must agree.")
        if data.X.device != reference.device or data.X.dtype != reference.dtype:
            raise ValueError("Historical data must share the model's dtype and device.")
    return reference


def unique_inputs(datasets: Any) -> UniqueInputs:
    # Set membership and ordering define discrete observation labels. The
    # non-explicit estimation grid is fixed, as in the reference Parameter.
    values = torch.cat([data.X.detach() for data in datasets])
    grid, inverse = torch.unique(values, dim=0, sorted=True, return_inverse=True)
    return UniqueInputs(
        grid,
        list(inverse.split([data.X.shape[0] for data in datasets])),
        torch.empty(0, device=grid.device, dtype=torch.long),
    )


def _settings(iterations: Any, tolerance: Any, shrinkage: Any, init_mode: str) -> None:
    if not isinstance(iterations, int) or isinstance(iterations, bool) or iterations < 0:
        raise ValueError("num_em_iterations must be a nonnegative integer.")
    if tolerance is not None and (not math.isfinite(tolerance) or tolerance < 0):
        raise ValueError("em_convergence_tol must be finite and nonnegative or None.")
    if (
        not isinstance(shrinkage, (int, float))
        or not math.isfinite(shrinkage)
        or not 0 <= shrinkage <= 1
    ):
        raise ValueError("covariance_shrinkage must be between zero and one.")
    if init_mode not in ("kernel", "naive"):
        raise ValueError("init_mode must be 'kernel' or 'naive'.")


def _degrees(value: Any, size: int, enabled: bool) -> Any:
    if not enabled:
        return value
    value = float(size + 2) if value is None else float(value)
    if not math.isfinite(value) or value <= size - 1:
        raise ValueError("iw_nu must be finite and greater than M - 1.")
    return value


def _noise(value: Any, like: Any) -> Any:
    result = torch.as_tensor(
        1e-10 if value is None else value, device=like.device, dtype=like.dtype
    )
    if result.numel() != 1 or not bool(torch.isfinite(result).all()) or bool((result < 0).any()):
        raise ValueError("EM noise must be a finite nonnegative scalar variance.")
    return result.reshape(())


def psd_projection(value: Any, floor: float = 1e-10) -> Any:
    symmetric = (value + value.mT) * 0.5
    # Away from the floor the map is exactly the identity. Returning that
    # identity also avoids spurious eigenvector singularities at repeated
    # positive eigenvalues (the function itself is smooth there).
    eigenvalues = torch.linalg.eigvalsh(symmetric)
    if bool((eigenvalues >= floor).all()):
        return symmetric
    if bool((eigenvalues < floor).all()):
        return symmetric * 0 + floor * torch.eye(
            value.shape[-1], device=value.device, dtype=value.dtype
        )
    tolerance = (
        torch.finfo(value.dtype).eps * eigenvalues.abs().max().clamp_min(1) * value.shape[-1]
    )
    if (
        torch.is_grad_enabled()
        and symmetric.requires_grad
        and bool((eigenvalues.diff().abs() <= tolerance).any())
    ):
        raise UnsupportedError(
            "Differentiating a mixed PSD-floor projection with repeated eigenvalues requires a spectral Fréchet rule; this Tensor configuration is not yet supported."
        )
    eigenvalues, vectors = torch.linalg.eigh(symmetric)
    return (vectors * eigenvalues.clamp_min(floor).unsqueeze(-2)) @ vectors.mT


def interpolate(
    X: Any, mean: Any, kernel: Any, Z: Any, mu: Any, covariance: Any, factor: Any = None
) -> tuple[Any, Any, Any]:
    factor = _rs.tensor_psd_safe_cholesky(dense(kernel, Z, Z)) if factor is None else factor
    Kxz = dense(kernel, X, Z)
    V = torch.linalg.solve_triangular(factor, Kxz.mT, upper=False)
    weights = torch.linalg.solve_triangular(factor.mT, V, upper=True).mT
    shifted = mean_at(mean, X) + (weights @ (mu - mean_at(mean, Z)).unsqueeze(-1)).squeeze(-1)
    residual = dense(kernel, X, X) - V.mT @ V
    return shifted, residual + weights @ covariance @ weights.mT, covariance @ weights.mT


def run_em(
    datasets: Any,
    Z: Any,
    mean: Any,
    kernel: Any,
    noise: Any,
    indices: Any,
    iterations: int,
    mean_prior: bool,
    covariance_prior: bool,
    degrees: Any,
    tolerance: Any,
    shrinkage: float,
    init_mode: str = "kernel",
    warm: Any = None,
    history: Any = None,
) -> tuple[Any, Any]:
    size = Z.shape[-2]
    gram = dense(kernel, Z, Z)
    initial_mean = mean_at(mean, Z)
    if gram.shape != (size, size) or initial_mean.shape != (size,):
        raise UnsupportedError("EM requires unbatched scalar mean and covariance modules.")
    if warm is not None:
        mu, covariance = (value.detach().clone() for value in warm)
    elif init_mode == "naive":
        mu, covariance = Z.new_zeros(size), torch.eye(size, device=Z.device, dtype=Z.dtype)
    else:
        mu, covariance = initial_mean, gram
    # Initial pretraining uses its selected initialization for the prior
    # matrices. Subsequent cold/warm reruns use the parametric kernel gram.
    prior_covariance = covariance if warm is None and init_mode == "naive" else gram
    psi = (degrees + size + 1) * prior_covariance if covariance_prior else None
    prior_factor = _rs.tensor_psd_safe_cholesky(prior_covariance) if mean_prior else None
    interpolation_factor = _rs.tensor_psd_safe_cholesky(gram) if indices is None else None
    count = len(datasets)
    for _ in range(iterations):
        old_mu, old_covariance = mu, covariance
        conditional_means, conditional_covariances = [], []
        for position, dataset in enumerate(datasets):
            if indices is None:
                observed_mean, observed_covariance, cross = interpolate(
                    dataset.X, mean, kernel, Z, mu, covariance, interpolation_factor
                )
            else:
                index = indices[position]
                observed_mean = mu[index]
                observed_covariance = covariance[index][:, index]
                cross = covariance[:, index]
            factor, alpha = gaussian_factor(
                dataset.Y.squeeze(-1), observed_covariance, observed_mean, noise
            )
            cond_mu, cond_cov = _rs.tensor_exact_gp_predict(factor, alpha, cross.mT, covariance, mu)
            conditional_means.append(cond_mu)
            conditional_covariances.append(cond_cov)
        means = torch.stack(conditional_means)
        if mean_prior:
            assert prior_factor is not None
            inverse = torch.cholesky_inverse(_rs.tensor_psd_safe_cholesky(covariance))
            prior_inverse = torch.cholesky_inverse(prior_factor)
            precision = count * inverse + prior_inverse
            rhs = inverse @ means.sum(0) + prior_inverse @ initial_mean
            mu = torch.cholesky_solve(
                rhs.unsqueeze(-1), _rs.tensor_psd_safe_cholesky(precision)
            ).squeeze(-1)
        else:
            mu = means.mean(0)
        centered = means - mu
        scatter = torch.stack(conditional_covariances).sum(0) + centered.mT @ centered
        covariance = (
            scatter / count if psi is None else (scatter + psi) / (count + degrees + size + 1)
        )
        if shrinkage > 0:
            scale = covariance.diagonal().sum() / gram.diagonal().sum().clamp_min(1e-12)
            covariance = (1 - shrinkage) * covariance + shrinkage * scale * gram
        covariance = psd_projection(covariance)
        if history is not None:
            history.append((mu.clone(), covariance.clone()))
        if (
            tolerance is not None
            and bool((mu - old_mu).abs().max() < tolerance)
            and bool((covariance - old_covariance).abs().max() < tolerance)
        ):
            break
    return mu, covariance


def pretrain(
    datasets: Any,
    mean_module: Any,
    covar_module: Any,
    likelihood_noise: Any = None,
    inducing_points: Any = None,
    num_em_iterations: int = 16,
    use_mean_prior: bool = False,
    use_covar_prior: bool = False,
    iw_nu: Any = None,
    em_convergence_tol: Any = 1e-6,
    enable_interpolation: bool = True,
    init_mode: str = "kernel",
    iteration_history: Any = None,
    covariance_shrinkage: float = 0.0,
) -> EMPriorContainer:
    like = validate_datasets(datasets)
    _settings(num_em_iterations, em_convergence_tol, covariance_shrinkage, init_mode)
    unique = unique_inputs(datasets)
    Z = unique.X_all if inducing_points is None else inducing_points
    if (
        not isinstance(Z, torch.Tensor)
        or Z.ndim != 2
        or not Z.shape[0]
        or Z.shape[-1] != like.shape[-1]
    ):
        raise ValueError("inducing_points must be a nonempty M x d Tensor.")
    if Z.device != like.device or Z.dtype != like.dtype or not bool(torch.isfinite(Z).all()):
        raise ValueError(
            "Inducing points must be finite and share the historical dtype and device."
        )
    mean = build_mean(mean_module, (), like)
    kernel = build_kernel(covar_module, like.shape[-1], (), like)
    noise = _noise(likelihood_noise, Z)
    degrees = _degrees(iw_nu, Z.shape[0], use_covar_prior)
    mu, covariance = run_em(
        datasets,
        Z,
        mean,
        kernel,
        noise,
        unique.experiment_indices if inducing_points is None else None,
        num_em_iterations,
        use_mean_prior,
        use_covar_prior,
        degrees,
        em_convergence_tol,
        covariance_shrinkage,
        init_mode,
        history=iteration_history,
    )
    factor = _rs.tensor_psd_safe_cholesky(dense(kernel, Z, Z))
    # The container is a reusable snapshot, including source modules and all
    # historical datasets. Future fitting or mutation cannot change its law.
    return EMPriorContainer(
        [
            ExperimentDataset(
                data.X.detach().clone(),
                data.Y.detach().clone(),
                None if data.Yvar is None else data.Yvar.detach().clone(),
            )
            for data in datasets
        ],
        cast(Any, copy.deepcopy(mean)),
        cast(Any, copy.deepcopy(kernel)),
        noise.detach().clone(),
        num_em_iterations,
        use_mean_prior,
        use_covar_prior,
        degrees,
        em_convergence_tol,
        enable_interpolation,
        inducing_points is not None,
        Z.detach().clone(),
        mu.detach().clone(),
        covariance.detach().clone(),
        factor.detach().clone(),
        (mu - mean_at(mean, Z)).detach().clone(),
        UniqueInputs(
            unique.X_all.detach().clone(),
            [v.clone() for v in unique.experiment_indices],
            unique.forward_indices.clone(),
        ),
    )


class _HistoricalData(nn.Module):
    X: Any
    Y: Any
    Yvar: Any

    def __init__(self, value: ExperimentDataset) -> None:
        super().__init__()
        self.register_buffer("X", value.X)
        self.register_buffer("Y", value.Y)
        self.register_buffer("Yvar", value.Yvar)


class TensorEMGaussianProcess(TorchSingleTaskGP, EMEmpiricalGaussianProcess):
    """Exact empirical GP with a differentiable, device-local EM training law."""

    _additive_base: Any

    def __init__(
        self,
        train_X: Any,
        train_Y: Any,
        em_prior: Any = None,
        datasets: Any = None,
        mean_module: Any = None,
        covar_module: Any = None,
        likelihood: Any = None,
        num_em_iterations: int = 16,
        use_mean_prior: bool = False,
        use_covar_prior: bool = False,
        iw_nu: Any = None,
        inducing_points: Any = None,
        enable_interpolation: bool = True,
        warm_start_em: bool = False,
        em_convergence_tol: Any = 1e-6,
        init_mode: str = "kernel",
        learnable_inducing_points: bool = False,
        covariance_shrinkage: float = 0.0,
        *,
        backend: str = "torch",
    ) -> None:
        from botorch_rs.likelihoods import GaussianLikelihood

        self._validate_data(train_X, train_Y)
        if train_X.ndim != 2 or train_Y.shape[-1] != 1:
            raise ValueError("EM conditioning requires unbatched n x d inputs and n x 1 targets.")
        if em_prior is not None and covariance_shrinkage > 0:
            raise ValueError("covariance_shrinkage only applies to EM reruns from datasets.")
        using_pretrained = em_prior is not None
        if using_pretrained:
            if not isinstance(em_prior, EMPriorContainer):
                raise TypeError("em_prior must be an EMPriorContainer.")
            datasets = [
                ExperimentDataset(
                    torch.as_tensor(data.X, device=train_X.device, dtype=train_X.dtype)
                    .detach()
                    .clone(),
                    torch.as_tensor(data.Y, device=train_X.device, dtype=train_X.dtype)
                    .detach()
                    .clone(),
                    None
                    if data.Yvar is None
                    else torch.as_tensor(data.Yvar, device=train_X.device, dtype=train_X.dtype)
                    .detach()
                    .clone(),
                )
                for data in em_prior.datasets
            ]
            mean_module, covar_module = copy.deepcopy((em_prior.mean_module, em_prior.covar_module))
            num_em_iterations = em_prior.num_em_iterations
            use_mean_prior, use_covar_prior = em_prior.use_mean_prior, em_prior.use_covar_prior
            iw_nu, em_convergence_tol = em_prior.iw_nu, em_prior.em_convergence_tol
            enable_interpolation = em_prior.enable_interpolation
            inducing_points = torch.as_tensor(
                em_prior.X_inducing, device=train_X.device, dtype=train_X.dtype
            )
        elif datasets is None or mean_module is None or covar_module is None:
            raise ValueError("Provide an em_prior or datasets with mean_module and covar_module.")
        validate_datasets(datasets, train_X)
        _settings(num_em_iterations, em_convergence_tol, covariance_shrinkage, init_mode)
        external = isinstance(likelihood, nn.Module)
        if external and not any(
            cls.__name__ in ("GaussianLikelihood", "TorchGaussianLikelihood")
            for cls in type(likelihood).__mro__
        ):
            raise ValueError("EM requires an unbatched GaussianLikelihood.")
        super().__init__(
            train_X,
            train_Y,
            likelihood=None if external else (likelihood or GaussianLikelihood()),
            covar_module=covar_module,
            mean_module=mean_module,
            outcome_transform=None,
            backend=backend,
        )
        if external:
            self.likelihood = likelihood.to(train_X)
        _noise(self.likelihood.noise, train_X)
        self.initial_mean_module, self.initial_covar_module = self.mean_module, self.covar_module
        self._history = nn.ModuleList([_HistoricalData(data) for data in datasets])
        self._unique_inputs_obs = unique_inputs(datasets)
        explicit = em_prior.use_inducing_points if using_pretrained else inducing_points is not None
        Z = self._unique_inputs_obs.X_all if inducing_points is None else inducing_points
        if (
            not isinstance(Z, torch.Tensor)
            or Z.ndim != 2
            or not Z.shape[0]
            or Z.shape[-1] != train_X.shape[-1]
        ):
            raise ValueError("inducing_points must have nonempty M x d shape.")
        if (
            Z.dtype != train_X.dtype
            or Z.device != train_X.device
            or not bool(torch.isfinite(Z).all())
        ):
            raise ValueError(
                "Inducing points must be finite and match the model's dtype and device."
            )
        if learnable_inducing_points and not explicit:
            raise UnsupportedError("Learnable EM locations require explicit inducing points.")
        self.register_parameter(
            "_X_inducing", nn.Parameter(Z.detach().clone(), requires_grad=learnable_inducing_points)
        )
        self._N_inducing = Z.shape[0]
        self._use_inducing_points, self._learnable_inducing_points = (
            explicit,
            learnable_inducing_points,
        )
        self._using_pretrained_prior = using_pretrained
        self._freeze_pretrained = False
        self._additive_base = None
        self.enable_interpolation, self.warm_start_em = enable_interpolation, warm_start_em
        self.num_em_iterations = num_em_iterations
        self.use_mean_prior, self.use_covar_prior = use_mean_prior, use_covar_prior
        self.iw_nu = _degrees(iw_nu, Z.shape[0], use_covar_prior)
        self.em_convergence_tol, self._covariance_shrinkage = (
            em_convergence_tol,
            covariance_shrinkage,
        )
        if using_pretrained:
            mu, covariance = (
                torch.as_tensor(em_prior.mu_inducing, device=train_X.device, dtype=train_X.dtype)
                .detach()
                .clone(),
                torch.as_tensor(em_prior.Sigma_inducing, device=train_X.device, dtype=train_X.dtype)
                .detach()
                .clone(),
            )
            if mu.shape != (Z.shape[0],) or covariance.shape != (Z.shape[0], Z.shape[0]):
                raise ValueError("Pretrained inducing moments must have M and M x M shapes.")
            if not bool(torch.isfinite(mu).all()) or not bool(torch.isfinite(covariance).all()):
                raise ValueError("Pretrained inducing moments must be finite.")
            tolerance = (
                torch.finfo(covariance.dtype).eps * Z.shape[0] * covariance.abs().max().clamp_min(1)
            )
            if not bool(((covariance - covariance.mT).abs() <= tolerance).all()):
                raise ValueError("Pretrained inducing covariance must be symmetric.")
            covariance = (covariance + covariance.mT) * 0.5
            if bool((torch.linalg.eigvalsh(covariance) < -tolerance).any()):
                raise ValueError("Pretrained inducing covariance must be positive semidefinite.")
        else:
            with torch.no_grad():
                mu, covariance = self._run_em(initial_mode=init_mode)
        self.register_buffer("_mu_inducing", mu.detach())
        self.register_buffer("_Sigma_inducing", covariance.detach())
        self._em_refreshes = 0

    @property
    def _X_inducing(self) -> Any:
        try:
            return self._parameters["_X_inducing"]
        except (AttributeError, KeyError):
            raise AttributeError("_X_inducing") from None

    @_X_inducing.setter
    def _X_inducing(self, value: Any) -> None:
        if not self._learnable_inducing_points:
            raise UnsupportedError("Inducing locations were configured as fixed.")
        if value.shape != self._X_inducing.shape:
            raise ValueError("Inducing location shape cannot change.")
        with torch.no_grad():
            self._X_inducing.copy_(value)

    @property
    def _num_outputs(self) -> int:
        return self.num_outputs

    @_num_outputs.setter
    def _num_outputs(self, value: int) -> None:
        if value != self.num_outputs:
            raise ValueError("An EM Gaussian process has one output.")

    @property
    def datasets(self) -> list[ExperimentDataset]:
        return [ExperimentDataset(data.X, data.Y, data.Yvar) for data in self._history]

    @datasets.setter
    def datasets(self, value: Any) -> None:
        validate_datasets(value, self._train_X)
        unique = unique_inputs(value)
        if not self._use_inducing_points and not torch.equal(unique.X_all, self._X_inducing):
            raise ValueError("Replacing indexed histories cannot change the fixed estimation grid.")
        self._history = nn.ModuleList([_HistoricalData(data) for data in value])
        self._unique_inputs_obs = unique
        self._prediction_cache = None

    def _indices(self) -> Any:
        return (
            None
            if self._use_inducing_points
            else [self._find_indices_in_historical(data.X) for data in self.datasets]
        )

    def _run_em(self, initial_mode: str = "kernel") -> tuple[Any, Any]:
        warm = (
            (self._mu_inducing, self._Sigma_inducing)
            if self.warm_start_em and hasattr(self, "_mu_inducing")
            else None
        )
        return run_em(
            self.datasets,
            self._X_inducing,
            self.initial_mean_module,
            self.initial_covar_module,
            _noise(self.likelihood.noise, self._train_X),
            self._indices(),
            self.num_em_iterations,
            self.use_mean_prior,
            self.use_covar_prior,
            self.iw_nu,
            self.em_convergence_tol,
            self._covariance_shrinkage,
            initial_mode,
            warm,
        )

    def _refresh_em(self) -> tuple[Any, Any]:
        if not self._using_pretrained_prior:
            moments = self._run_em()
            # Persist only the learned law. A returned forward owns its own
            # update graph; a later prediction must not reuse a consumed graph.
            self._mu_inducing, self._Sigma_inducing = (value.detach().clone() for value in moments)
            self._em_refreshes += 1
        else:
            moments = self._mu_inducing, self._Sigma_inducing
        self._prediction_cache = None
        return moments

    def _prior(self, X: Any, moments: Any = None) -> tuple[Any, Any]:
        mu, learned_covariance = (
            (self._mu_inducing, self._Sigma_inducing) if moments is None else moments
        )
        if self.enable_interpolation or self._use_inducing_points:
            mean, covariance, _ = interpolate(
                X,
                self.initial_mean_module,
                self.initial_covar_module,
                self._X_inducing,
                mu,
                learned_covariance,
            )
        else:
            indices = self._find_indices_in_historical(X)
            mean = mu[indices]
            covariance = learned_covariance[indices.unsqueeze(-1), indices.unsqueeze(-2)]
        if self._additive_base is not None:
            covariance = covariance + dense(self._additive_base, X, X)
        return mean, covariance

    def _find_indices_in_historical(self, X: Any) -> Any:
        squared = ((X.unsqueeze(-2) - self._X_inducing) ** 2).sum(-1)
        distances, indices = squared.min(-1)
        if bool((distances > 1e-12).any()):
            raise ValueError("Interpolation is disabled and X is not a historical subset.")
        return indices

    @property
    def _cached_L_kernel_inducing(self) -> Any:
        return _rs.tensor_psd_safe_cholesky(
            dense(self.initial_covar_module, self._X_inducing, self._X_inducing)
        )

    @property
    def _cached_delta_mu(self) -> Any:
        return self._mu_inducing - mean_at(self.initial_mean_module, self._X_inducing)

    def _exact_mll(self, output: Any = None, target: Any = None) -> Any:
        if output is None:
            moments = self._refresh_em() if self.training else None
            mean, covariance = self._prior(self._train_X, moments)
        else:
            mean, covariance = output.mean, output.covariance_matrix
        targets = self.train_targets if target is None else target
        return (
            gaussian_log_prob(targets, covariance, mean, self._training_noise(targets, None))
            + self._log_prior()
        ) / targets.shape[-1]

    def _posterior_tensors(self, X: Any, observation_noise: Any = False) -> Any:
        if not isinstance(X, torch.Tensor) or X.ndim < 2 or X.shape[-1] != self._train_X.shape[-1]:
            raise ValueError("X must have shape (..., q, d).")
        if X.device != self._train_X.device or X.dtype != self._train_X.dtype:
            raise ValueError("X must share the model's dtype and device.")
        self.eval()
        batch = torch.broadcast_shapes(X.shape[:-2], self.batch_shape)
        train = self._train_X.expand(*batch, *self._train_X.shape[-2:])
        query = X.expand(*batch, *X.shape[-2:])
        joint = torch.cat((train, query), -2)
        mean, covariance = self._prior(joint)
        n = train.shape[-2]
        factor, alpha = gaussian_factor(
            self._train_Y.squeeze(-1),
            covariance[..., :n, :n],
            mean[..., :n],
            self._training_noise(self.train_targets, None),
        )
        noise = None
        if isinstance(observation_noise, torch.Tensor):
            noise = observation_noise.squeeze(-1).to(X)
            if not bool(torch.isfinite(noise).all()) or bool((noise < 0).any()):
                raise ValueError("Predictive noise must be finite and nonnegative.")
        elif observation_noise is True:
            noise = self.likelihood.noise
        elif observation_noise is not False:
            raise TypeError("observation_noise must be bool or Tensor.")
        return _rs.tensor_exact_gp_predict(
            factor,
            alpha,
            covariance[..., :n, n:],
            covariance[..., n:, n:],
            mean[..., n:],
            observation_noise=noise,
        )

    def forward(self, X: Any) -> Any:
        if not self.training:
            mean, covariance = self._posterior_tensors(X)
        else:
            moments = self._refresh_em()
            mean, covariance = self._prior(X, moments)
        return TorchGPyTorchPosterior(mean.unsqueeze(-1), covariance).distribution

    __call__ = nn.Module.__call__

    def _log_prior(self) -> Any:
        result = self._train_X.new_zeros(())
        seen: set[int] = set()
        for module in (
            self.likelihood,
            self.initial_mean_module,
            self.initial_covar_module,
            self._additive_base,
        ):
            if module is not None and id(module) not in seen:
                seen.add(id(module))
                result = result + module_log_prior(module, ())
        return result

    def _historical_mll(self) -> Any:
        mu, learned_covariance = self._refresh_em()
        result = self._train_X.new_zeros(())
        indices = self._indices()
        for position, dataset in enumerate(self.datasets):
            if indices is None:
                mean, covariance = self._prior(dataset.X, (mu, learned_covariance))
            else:
                index = indices[position]
                mean, covariance = mu[index], learned_covariance[index][:, index]
                if self._additive_base is not None:
                    covariance = covariance + dense(self._additive_base, dataset.X, dataset.X)
            result = result + gaussian_log_prob(
                dataset.Y.squeeze(-1), covariance, mean, self.likelihood.noise
            )
        return (result + self._log_prior()) / sum(data.X.shape[0] for data in self.datasets)

    def freeze_pretrained_parameters(self) -> None:
        for module in (self.initial_mean_module, self.initial_covar_module):
            module.requires_grad_(False)
        self._mu_inducing, self._Sigma_inducing = (
            self._mu_inducing.detach(),
            self._Sigma_inducing.detach(),
        )
        self._freeze_pretrained = True

    @classmethod
    def from_pretrained(
        cls,
        em_prior: Any,
        train_X: Any,
        train_Y: Any,
        likelihood: Any = None,
        freeze_pretrained: bool = True,
        learnable_inducing_points: bool = False,
        base_covar_module: Any = None,
        *,
        backend: str = "torch",
    ) -> Any:
        model = cls(
            train_X,
            train_Y,
            em_prior=em_prior,
            likelihood=likelihood,
            learnable_inducing_points=learnable_inducing_points,
            backend=backend,
        )
        if freeze_pretrained:
            model.freeze_pretrained_parameters()
        if base_covar_module is not None:
            model._additive_base = build_kernel(base_covar_module, train_X.shape[-1], (), train_X)
        return model

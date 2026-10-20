# Copyright (c) Meta Platforms, Inc. and affiliates.
# Adapted from BoTorch b0dd01be under the MIT license (see LICENSE).
"""Port-owned device-resident predictive entropy search and EP algebra.

The initial and one-step EP updates follow the reference's deliberately noisy
candidate covariance convention. Damping is piecewise constant; derivatives
are defined within a fixed damping/fallback branch, not at branch boundaries.
"""

from __future__ import annotations

import inspect
from typing import Any

import torch
from torch import Tensor, nn
from torch.distributions import Normal

from botorch_rs.acquisition._torch_backend_acquisition import _TensorAcquisition
from botorch_rs.exceptions.errors import InputDataError, UnsupportedError
from botorch_rs.posteriors._torch_backend import TorchGPyTorchPosterior


def check_no_nans(value: Tensor) -> None:
    if bool(torch.isnan(value).any()):
        raise InputDataError("PES expectation propagation produced NaNs.")


class TensorPredictiveEntropy(_TensorAcquisition):
    pareto_sets: Tensor
    augmented_X: Tensor
    _raw_train_X: Tensor
    _omega_f_nat_mean: Tensor
    _omega_f_nat_cov: Tensor
    _ep_damping: Tensor

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        nn.Module.__init__(self)
        bound = inspect.signature(inspect.getattr_static(self._public_class, "__init__")).bind(
            self, *args, **kwargs
        )
        bound.apply_defaults()
        values = bound.arguments
        self.model = values["model"]
        if self.model.batch_shape:
            raise UnsupportedError("PES requires an initially unbatched model, as in BoTorch.")
        self._is_analytic, self._name = False, self._public_class.__name__
        self.maximize = bool(values["maximize"])
        self.register_buffer("_X_pending", None)
        self.set_X_pending(values["X_pending"])
        first = next(iter(self.model.models)) if hasattr(self.model, "models") else self.model
        train = getattr(first, "_train_X", None)
        if train is None:
            raise UnsupportedError("PES requires raw training locations exposed by the model.")
        if train.ndim != 2:
            raise UnsupportedError("PES requires unbatched raw training locations.")
        if self._name == "qPredictiveEntropySearch":
            optimal = values["optimal_inputs"]
            if self.model.num_outputs != 1 or not isinstance(optimal, Tensor) or optimal.ndim != 2:
                raise ValueError("Scalar PES requires a scalar model and n x d optimal inputs.")
            self.register_buffer("optimal_inputs", optimal)
            pareto = optimal.unsqueeze(-2)
        else:
            pareto = values["pareto_sets"]
        if (
            not isinstance(pareto, Tensor)
            or pareto.ndim != 3
            or not pareto.shape[0]
            or not pareto.shape[1]
            or pareto.shape[-1] != train.shape[-1]
            or pareto.dtype != train.dtype
            or pareto.device != train.device
        ):
            raise ValueError(
                "Pareto sets must be nonempty samples x points x d tensors on the model device."
            )
        self.num_pareto_samples = pareto.shape[0]
        self.register_buffer("pareto_sets", pareto)
        self.register_buffer("_raw_train_X", train)
        self.register_buffer("augmented_X", None)
        self.register_buffer("_omega_f_nat_mean", None)
        self.register_buffer("_omega_f_nat_cov", None)
        self.register_buffer("_ep_damping", None)
        self.max_ep_iterations = int(values["max_ep_iterations"])
        self.ep_jitter, self.test_jitter, self.threshold = (
            float(values[key]) for key in ("ep_jitter", "test_jitter", "threshold")
        )
        if self.max_ep_iterations < 0 or min(self.ep_jitter, self.test_jitter, self.threshold) <= 0:
            raise ValueError("PES requires positive jitters/threshold and nonnegative iterations.")
        with torch.no_grad():
            self._refresh_ep()

    def _refresh_ep(self) -> None:
        self.augmented_X = torch.cat(
            (
                self._raw_train_X.expand(self.num_pareto_samples, *self._raw_train_X.shape),
                self.pareto_sets,
            ),
            dim=-2,
        )
        self._expectation_propagation()

    @property
    def ep_iterations(self) -> int:
        return self._ep_iterations

    @property
    def ep_damping(self) -> Tensor:
        return self._ep_damping

    @property
    def cached_factors(self) -> tuple[Tensor, Tensor]:
        n = self._raw_train_X.shape[-2]
        return self._omega_f_nat_mean[..., n:, :, :], self._omega_f_nat_cov[..., n:, :, :, :]

    def forward(self, X: Tensor) -> Tensor:
        X = self._candidates_tensor(X)
        # EP factors depend on training data and hyperparameters. Refresh the
        # tensor graph on each differentiable evaluation rather than retaining
        # a consumed autograd tape or silently detaching those dependencies.
        if torch.is_grad_enabled():
            self._refresh_ep()
        return self._compute_information_gain(X)

    def _expectation_propagation(self) -> None:
        M = self.model.num_outputs
        train_X = self._raw_train_X
        tkwargs: dict[str, Any] = {"dtype": train_X.dtype, "device": train_X.device}
        N = len(train_X)
        num_pareto_samples = self.num_pareto_samples
        P = self.pareto_sets.shape[-2]
        pred_nat_mean, pred_nat_cov, pred_mean, pred_cov = _initialize_predictive_matrices(
            X=self.augmented_X,
            model=self.model,
            observation_noise=False,
            jitter=self.ep_jitter,
            natural=True,
        )
        pred_f_mean = pred_mean[..., 0:M, :]
        pred_f_nat_mean = pred_nat_mean[..., 0:M, :]
        pred_f_cov = pred_cov[..., 0:M, :, :]
        pred_f_nat_cov = pred_nat_cov[..., 0:M, :, :]
        mean_f = pred_f_mean.clone()
        nat_mean_f = pred_f_nat_mean.clone()
        cov_f = pred_f_cov.clone()
        nat_cov_f = pred_f_nat_cov.clone()
        omega_f_nat_mean = torch.zeros((num_pareto_samples, M, N + P, P, 2), **tkwargs)
        omega_f_nat_cov = torch.zeros((num_pareto_samples, M, N + P, P, 2, 2), **tkwargs)
        damping = torch.ones(num_pareto_samples, M, **tkwargs)
        iteration = 0
        while torch.sum(damping) > 0 and iteration < self.max_ep_iterations:
            omega_f_nat_mean_new, omega_f_nat_cov_new = _safe_update_omega(
                mean_f=mean_f,
                cov_f=cov_f,
                omega_f_nat_mean=omega_f_nat_mean,
                omega_f_nat_cov=omega_f_nat_cov,
                N=N,
                P=P,
                M=M,
                maximize=self.maximize,
                jitter=self.ep_jitter,
            )
            nat_mean_f_new, nat_cov_f_new = _update_marginals(
                pred_f_nat_mean=pred_f_nat_mean,
                pred_f_nat_cov=pred_f_nat_cov,
                omega_f_nat_mean=omega_f_nat_mean_new,
                omega_f_nat_cov=omega_f_nat_cov_new,
                N=N,
                P=P,
            )
            damping, cholesky_nat_cov_f = _update_damping(
                nat_cov=nat_cov_f,
                nat_cov_new=nat_cov_f_new,
                damping_factor=damping,
                jitter=self.ep_jitter,
            )
            check_no_nans(cholesky_nat_cov_f)
            omega_f_nat_mean = _damped_update(
                old_factor=omega_f_nat_mean, new_factor=omega_f_nat_mean_new, damping_factor=damping
            )
            omega_f_nat_cov = _damped_update(
                old_factor=omega_f_nat_cov, new_factor=omega_f_nat_cov_new, damping_factor=damping
            )
            nat_mean_f = _damped_update(
                old_factor=nat_mean_f, new_factor=nat_mean_f_new, damping_factor=damping
            )
            nat_cov_f = _damped_update(
                old_factor=nat_cov_f, new_factor=nat_cov_f_new, damping_factor=damping
            )
            cov_f_new = torch.cholesky_inverse(cholesky_nat_cov_f)
            mean_f_new = torch.einsum("...ij,...j->...i", cov_f_new, nat_mean_f)
            check_no_nans(cov_f_new)
            damping, delta_mean_f, delta_cov_f = _update_damping_when_converged(
                mean_old=mean_f,
                mean_new=mean_f_new,
                cov_old=cov_f,
                cov_new=cov_f_new,
                damping_factor=damping,
                threshold=self.threshold,
                iteration=iteration,
            )
            cov_f = cov_f_new
            mean_f = mean_f_new
            iteration = iteration + 1
        check_no_nans(omega_f_nat_mean)
        check_no_nans(omega_f_nat_cov)
        self._omega_f_nat_mean = omega_f_nat_mean
        self._omega_f_nat_cov = omega_f_nat_cov
        self._ep_iterations = iteration
        self._ep_damping = damping

    def _compute_information_gain(self, X: Tensor) -> Tensor:
        tkwargs: dict[str, Any] = {"dtype": X.dtype, "device": X.device}
        batch_shape = X.shape[0:-2]
        q = X.shape[-2]
        M = self.model.num_outputs
        N = self._raw_train_X.shape[-2]
        P = self.pareto_sets.shape[-2]
        num_pareto_samples = self.num_pareto_samples
        new_shape = batch_shape + torch.Size([num_pareto_samples]) + X.shape[-2:]
        expanded_X = X.unsqueeze(-3).expand(new_shape)
        expanded_ps = self.pareto_sets.expand(X.shape[0:-2] + self.pareto_sets.shape)
        aug_X = torch.cat([expanded_X, expanded_ps], dim=-2)
        pred_nat_mean, pred_nat_cov, pred_mean, pred_cov = _initialize_predictive_matrices(
            X=aug_X, model=self.model, observation_noise=True, jitter=self.test_jitter, natural=True
        )
        pred_f_mean = pred_mean[..., 0:M, :]
        pred_f_nat_mean = pred_nat_mean[..., 0:M, :]
        pred_f_cov = pred_cov[..., 0:M, :, :]
        pred_f_nat_cov = pred_nat_cov[..., 0:M, :, :]
        _, _, _, pred_cov_noise = _initialize_predictive_matrices(
            X=aug_X,
            model=self.model,
            observation_noise=True,
            jitter=self.test_jitter,
            natural=False,
        )
        pred_f_cov_noise = pred_cov_noise[..., 0:M, :, :]
        observation_noise = pred_f_cov_noise - pred_f_cov
        omega_f_nat_mean = torch.zeros(
            batch_shape + torch.Size([num_pareto_samples, M, q + P, P, 2]), **tkwargs
        )
        omega_f_nat_cov = torch.zeros(
            batch_shape + torch.Size([num_pareto_samples, M, q + P, P, 2, 2]), **tkwargs
        )
        omega_f_nat_mean, omega_f_nat_cov = _safe_update_omega(
            mean_f=pred_f_mean,
            cov_f=pred_f_cov,
            omega_f_nat_mean=omega_f_nat_mean,
            omega_f_nat_cov=omega_f_nat_cov,
            N=q,
            P=P,
            M=M,
            maximize=self.maximize,
            jitter=self.test_jitter,
        )
        omega_f_nat_mean, omega_f_nat_cov = _augment_factors_with_cached_factors(
            q=q,
            N=N,
            omega_f_nat_mean=omega_f_nat_mean,
            cached_omega_f_nat_mean=self._omega_f_nat_mean,
            omega_f_nat_cov=omega_f_nat_cov,
            cached_omega_f_nat_cov=self._omega_f_nat_cov,
        )
        nat_mean_f, nat_cov_f = _update_marginals(
            pred_f_nat_mean=pred_f_nat_mean,
            pred_f_nat_cov=pred_f_nat_cov,
            omega_f_nat_mean=omega_f_nat_mean,
            omega_f_nat_cov=omega_f_nat_cov,
            N=q,
            P=P,
        )
        damping = torch.ones(batch_shape + torch.Size([num_pareto_samples, M]), **tkwargs)
        damping, cholesky_nat_cov_f_new = _update_damping(
            nat_cov=pred_f_nat_cov,
            nat_cov_new=nat_cov_f,
            damping_factor=damping,
            jitter=self.test_jitter,
        )
        cov_f_new = torch.cholesky_inverse(cholesky_nat_cov_f_new)
        check_no_nans(cov_f_new)
        log_det_pred_f_cov_noise = _compute_log_determinant(cov=pred_f_cov_noise, q=q)
        log_det_cov_f = _compute_log_determinant(cov=cov_f_new + observation_noise, q=q)
        q_pes_f = log_det_pred_f_cov_noise - log_det_cov_f
        check_no_nans(q_pes_f)
        return 0.5 * q_pes_f


def log_cdf_robust(x: Tensor) -> Tensor:
    eps = torch.finfo(x.dtype).eps
    # Mask before evaluating each branch: log(0) in an inactive branch
    # otherwise contaminates first and higher derivatives through 0 * inf.
    cdf = torch.special.ndtr(x.masked_fill(x >= 5, 5))
    positive = cdf > 0
    lower = torch.where(positive, torch.where(positive, cdf, 1).log(), torch.finfo(x.dtype).min)
    upper = -torch.special.ndtr(-x.masked_fill(x < 5, 5))
    return torch.where(x < 5, lower, upper).clamp(torch.finfo(x.dtype).min, -eps)


def _initialize_predictive_matrices(
    X: Tensor,
    model: Any,
    observation_noise: bool = True,
    jitter: float = 0.0001,
    natural: bool = True,
) -> Any:
    tkwargs: dict[str, Any] = {"dtype": X.dtype, "device": X.device}
    posterior = model.posterior(X, observation_noise=observation_noise)
    if not isinstance(posterior, TorchGPyTorchPosterior):
        raise UnsupportedError("PES requires Gaussian device posteriors.")
    init_pred_cov = posterior.mvn.covariance_matrix
    num_outputs = model.num_outputs
    R = int(init_pred_cov.shape[-1] / num_outputs)
    blocks = [
        init_pred_cov[..., m * R : (m + 1) * R, m * R : (m + 1) * R].unsqueeze(-1)
        for m in range(num_outputs)
    ]
    pred_cov = torch.cat(blocks, dim=-1).swapaxes(-2, -1).swapaxes(-3, -2)
    identity = torch.diag_embed(torch.ones(pred_cov.shape[:-1], **tkwargs))
    pred_cov = pred_cov + jitter * identity
    pred_mean = posterior.mean.swapaxes(-2, -1)
    if natural:
        cholesky_pred_cov, _ = torch.linalg.cholesky_ex(pred_cov)
        pred_nat_cov = torch.cholesky_inverse(cholesky_pred_cov)
        pred_nat_mean = torch.einsum("...ij,...j->...i", pred_nat_cov, pred_mean)
        return (pred_nat_mean, pred_nat_cov, pred_mean, pred_cov)
    else:
        return (None, None, pred_mean, pred_cov)


def _get_omega_f_contribution(
    mean: Tensor, cov: Tensor, N: int, P: int, M: int
) -> tuple[Tensor, Tensor]:
    left = mean.unsqueeze(-1).expand(*mean.shape, P)
    right = mean[..., N:].unsqueeze(-2).expand_as(left)
    means = torch.stack((left, right), -1)
    diagonal = cov.diagonal(dim1=-2, dim2=-1)
    first = diagonal.unsqueeze(-1).expand_as(left)
    second = diagonal[..., N:].unsqueeze(-2).expand_as(left)
    cross = cov[..., :, N:]
    covariance = torch.stack(
        (torch.stack((first, cross), -1), torch.stack((cross, second), -1)), -2
    )
    return means, covariance


def _replace_pareto_diagonal(A: Tensor) -> Tensor:
    P, R = A.shape[-3], A.shape[-4]
    rows = torch.arange(R, device=A.device)
    columns = torch.arange(P, device=A.device) + R - P
    mask = (rows[:, None] == columns[None, :])[..., None, None]
    return torch.where(mask, torch.eye(2, dtype=A.dtype, device=A.device), A)


def _update_omega(
    mean_f: Tensor,
    cov_f: Tensor,
    omega_f_nat_mean: Tensor,
    omega_f_nat_cov: Tensor,
    N: int,
    P: int,
    M: int,
    maximize: bool = True,
    jitter: float = 1e-06,
) -> tuple[Tensor, Tensor]:
    tkwargs: dict[str, Any] = {"dtype": mean_f.dtype, "device": mean_f.device}
    CLAMP_LB = torch.finfo(tkwargs["dtype"]).eps
    NEG_INF = torch.finfo(tkwargs["dtype"]).min
    weight = 1.0 if maximize else -1.0
    mean_fX_fS, cov_fX_fS = _get_omega_f_contribution(mean_f, cov_f, N, P, M)
    identity = torch.diag_embed(torch.ones(cov_fX_fS.shape[:-1], **tkwargs))
    cov_fX_fS = _replace_pareto_diagonal(cov_fX_fS + jitter * identity)
    nat_cov_fX_fS = torch.linalg.inv(cov_fX_fS)
    nat_mean_fX_fS = torch.einsum("...ij,...j->...i", nat_cov_fX_fS, mean_fX_fS)
    cav_nat_mean_f = nat_mean_fX_fS - omega_f_nat_mean
    cav_nat_cov_f = nat_cov_fX_fS - omega_f_nat_cov
    cav_nat_cov_f = _replace_pareto_diagonal(cav_nat_cov_f)
    identity = torch.diag_embed(torch.ones(cav_nat_cov_f.shape[:-1], **tkwargs))
    cav_cov_f = torch.linalg.inv(cav_nat_cov_f + jitter * identity)
    cav_mean_f = torch.einsum("...ij,...j->...i", cav_cov_f, cav_nat_mean_f)
    cav_var_fX_minus_fS = (
        cav_cov_f[..., 0, 0] + cav_cov_f[..., 1, 1] - 2 * cav_cov_f[..., 0, 1]
    ).clamp_min(CLAMP_LB)
    cav_std_fX_minus_fS = torch.sqrt(cav_var_fX_minus_fS).clamp_min(CLAMP_LB)
    cav_mean_fX_minus_fS = weight * (cav_mean_f[..., 0] - cav_mean_f[..., 1])
    cav_alpha = cav_mean_fX_minus_fS / cav_std_fX_minus_fS
    normal_alpha = Normal(torch.zeros_like(cav_alpha), torch.ones_like(cav_alpha))
    cav_alpha_log_cdf = log_cdf_robust(cav_alpha)
    cav_alpha_log_pdf = normal_alpha.log_prob(cav_alpha).clamp_min(NEG_INF)
    cav_sum_alpha_log_cdf = torch.sum(cav_alpha_log_cdf, dim=-3).clamp_min(NEG_INF)
    cav_log_zeta = torch.log1p(-torch.exp(cav_sum_alpha_log_cdf)).clamp_min(NEG_INF)
    cav_logZ = cav_log_zeta
    cav_log_rho = -cav_logZ + cav_sum_alpha_log_cdf
    cav_log_rho = cav_log_rho.unsqueeze(-3) - cav_alpha_log_cdf + cav_alpha_log_pdf
    cav_rho = -torch.exp(cav_log_rho).clamp(NEG_INF, -NEG_INF)
    ones_mean = torch.ones(cav_mean_f.shape, **tkwargs)
    ones_mean[..., 1] = -ones_mean[..., 1]
    ones_cov = torch.ones(cav_cov_f.shape, **tkwargs)
    ones_cov[..., 0, 1] = -ones_cov[..., 0, 1]
    ones_cov[..., 1, 0] = -ones_cov[..., 1, 0]
    cav_dlogZ_dm = cav_rho / cav_std_fX_minus_fS
    cav_dlogZ_dm = weight * cav_dlogZ_dm.unsqueeze(-1) * ones_mean
    cav_d2logZ_dm2 = -cav_rho * (cav_rho + cav_alpha) / cav_var_fX_minus_fS
    cav_d2logZ_dm2 = cav_d2logZ_dm2.unsqueeze(-1).unsqueeze(-1) * ones_cov
    cav_updated_mean_f = cav_mean_f + torch.einsum("...ij,...j->...i", cav_cov_f, cav_dlogZ_dm)
    cav_updated_cov_f = cav_cov_f + torch.einsum(
        "...ij,...jk,...kl->...il", cav_cov_f, cav_d2logZ_dm2, cav_cov_f
    )
    cav_updated_cov_f = _replace_pareto_diagonal(cav_updated_cov_f)
    identity = torch.diag_embed(torch.ones(cav_updated_cov_f.shape[:-1], **tkwargs))
    cav_updated_nat_cov_f = torch.linalg.inv(cav_updated_cov_f + jitter * identity)
    cav_updated_nat_mean_f = torch.einsum(
        "...ij,...j->...i", cav_updated_nat_cov_f, cav_updated_mean_f
    )
    omega_f_nat_mean_new = cav_updated_nat_mean_f - cav_nat_mean_f
    omega_f_nat_cov_new = cav_updated_nat_cov_f - cav_nat_cov_f
    return (omega_f_nat_mean_new, omega_f_nat_cov_new)


def _safe_update_omega(
    mean_f: Tensor,
    cov_f: Tensor,
    omega_f_nat_mean: Tensor,
    omega_f_nat_cov: Tensor,
    N: int,
    P: int,
    M: int,
    maximize: bool = True,
    jitter: float = 1e-6,
) -> tuple[Tensor, Tensor]:
    # An invalid Pareto/query member must not discard unrelated members.
    batch = mean_f.shape[:-2]
    means = mean_f.reshape(-1, M, N + P)
    covariances = cov_f.reshape(-1, M, N + P, N + P)
    old_means = omega_f_nat_mean.reshape(-1, M, N + P, P, 2)
    old_covariances = omega_f_nat_cov.reshape(-1, M, N + P, P, 2, 2)
    result_means, result_covariances = [], []
    for i in range(means.shape[0]):
        try:
            updated_mean, updated_covariance = _update_omega(
                means[i],
                covariances[i],
                old_means[i],
                old_covariances[i],
                N,
                P,
                M,
                maximize,
                jitter,
            )
            check_no_nans(updated_mean)
            check_no_nans(updated_covariance)
        except (RuntimeError, InputDataError):
            updated_mean, updated_covariance = old_means[i], old_covariances[i]
        result_means.append(updated_mean)
        result_covariances.append(updated_covariance)
    return (
        torch.stack(result_means).reshape(*batch, M, N + P, P, 2),
        torch.stack(result_covariances).reshape(*batch, M, N + P, P, 2, 2),
    )


def _update_marginals(
    pred_f_nat_mean: Tensor,
    pred_f_nat_cov: Tensor,
    omega_f_nat_mean: Tensor,
    omega_f_nat_cov: Tensor,
    N: int,
    P: int,
) -> tuple[Tensor, Tensor]:
    R = N + P
    rows = torch.arange(R, device=pred_f_nat_mean.device)
    columns = torch.arange(P, device=pred_f_nat_mean.device) + N
    mask = rows[:, None] != columns[None, :]
    means = omega_f_nat_mean * mask[..., None]
    covariances = omega_f_nat_cov * mask[..., None, None]
    pad = torch.nn.functional.pad
    natural_mean = pred_f_nat_mean + means[..., 0].sum(-1) + pad(means[..., 1].sum(-2), (N, 0))
    diagonal = covariances[..., 0, 0].sum(-1) + pad(covariances[..., 1, 1].sum(-2), (N, 0))
    cross_left = pad(covariances[..., 0, 1], (N, 0))
    cross_right = pad(covariances[..., 1, 0], (N, 0)).transpose(-1, -2)
    natural_covariance = pred_f_nat_cov + torch.diag_embed(diagonal) + cross_left + cross_right
    return natural_mean, natural_covariance


def _damped_update(old_factor: Tensor, new_factor: Tensor, damping_factor: Tensor) -> Tensor:
    bs = damping_factor.shape
    fs = old_factor.shape
    df = damping_factor
    for _ in range(len(fs[len(bs) :])):
        df = df.unsqueeze(-1)
    return df * new_factor + (1 - df) * old_factor


def _update_damping(
    nat_cov: Tensor, nat_cov_new: Tensor, damping_factor: Tensor, jitter: float
) -> tuple[Tensor, Tensor]:
    eye = torch.eye(nat_cov.shape[-1], dtype=nat_cov.dtype, device=nat_cov.device)
    # Discrete damping decisions carry no local derivatives. Factor only the
    # final positive matrix in the graph: differentiating failed Choleskys
    # and masking them afterwards still creates NaNs in backward.
    with torch.no_grad():
        _, initial_info = torch.linalg.cholesky_ex(nat_cov + jitter * eye)
        if bool((initial_info != 0).any()):
            raise ValueError("The previous PES precision is not positive definite with jitter.")
        df = damping_factor.clone()
        proposal = _damped_update(nat_cov, nat_cov_new, df)
        _, info = torch.linalg.cholesky_ex(proposal)
        retried = info != 0
        for _ in range(12):
            if not bool((info != 0).any()):
                break
            alternative = df * 0.5
            alternative = torch.where(alternative > 1e-3, alternative, 0)
            df = torch.where(info == 0, df, alternative)
            proposal = (
                _damped_update(nat_cov, nat_cov_new, df) + jitter * eye * retried[..., None, None]
            )
            _, info = torch.linalg.cholesky_ex(proposal)
        if bool((info != 0).any()):
            raise ValueError("PES damping could not recover a positive definite precision.")
    selected = _damped_update(nat_cov, nat_cov_new, df) + jitter * eye * retried[..., None, None]
    return df, torch.linalg.cholesky(selected)


def _update_damping_when_converged(
    mean_old: Tensor,
    mean_new: Tensor,
    cov_old: Tensor,
    cov_new: Tensor,
    damping_factor: Tensor,
    iteration: int,
    threshold: float = 0.001,
) -> tuple[Tensor, Tensor, Tensor]:
    df = damping_factor.clone()
    delta_mean = mean_new - mean_old
    delta_cov = cov_new - cov_old
    am = torch.amax(abs(mean_old), dim=-1)
    ac = torch.amax(abs(cov_old), dim=(-2, -1))
    if iteration > 2:
        mask_mean = torch.amax(abs(delta_mean), dim=-1) < threshold * am
        mask_cov = torch.amax(abs(delta_cov), dim=(-2, -1)) < threshold * ac
        mask = torch.logical_and(mask_mean, mask_cov)
        df[mask] = 0
    return (df, delta_mean, delta_cov)


def _augment_factors_with_cached_factors(
    q: int,
    N: int,
    omega_f_nat_mean: Tensor,
    cached_omega_f_nat_mean: Tensor,
    omega_f_nat_cov: Tensor,
    cached_omega_f_nat_cov: Tensor,
) -> tuple[Tensor, Tensor]:
    means = cached_omega_f_nat_mean[..., N:, :, :].expand(omega_f_nat_mean[..., q:, :, :].shape)
    covariances = cached_omega_f_nat_cov[..., N:, :, :, :].expand(
        omega_f_nat_cov[..., q:, :, :, :].shape
    )
    return (
        torch.cat((omega_f_nat_mean[..., :q, :, :], means), -3),
        torch.cat((omega_f_nat_cov[..., :q, :, :, :], covariances), -4),
    )


def _compute_log_determinant(cov: Tensor, q: int) -> Tensor:
    log_det_cov = torch.logdet(cov[..., 0:q, 0:q])
    check_no_nans(log_det_cov)
    return log_det_cov.sum(dim=-1).mean(dim=-1)

"""Device-resident empirical-prior KL, coordinate fitting and shared GPs."""

from __future__ import annotations

import copy
import math
from dataclasses import replace
from typing import Any, cast

import torch
from torch import nn

from botorch_rs import _botorch_rs as _rs
from botorch_rs.fit import fit_gpytorch_mll
from botorch_rs.mlls import SumMarginalLogLikelihood
from botorch_rs.models._torch_backend import TorchSingleTaskGP
from botorch_rs.models._torch_backend_modules import (
    build_kernel,
    build_mean,
)
from botorch_rs.models.empirical_gps._torch_em import validate_datasets
from botorch_rs.models.empirical_gps.em_empirical_gp import (
    EMEmpiricalGaussianProcess,
    EMEmpiricalMarginalLogLikelihood,
    pretrain_em_prior,
)
from botorch_rs.models.model_list_gp_regression import ModelListGP
from botorch_rs.optim.fit import fit_gpytorch_mll_scipy, fit_gpytorch_mll_torch


def gaussian_kl(mu_p: Any, covariance_p: Any, mu_q: Any, covariance_q: Any) -> Any:
    values = mu_p, covariance_p, mu_q, covariance_q
    if any(not isinstance(value, torch.Tensor) for value in values):
        raise TypeError("Tensor Gaussian KL requires four Tensor inputs.")
    if any(value.dtype != mu_p.dtype or value.device != mu_p.device for value in values):
        raise ValueError("Gaussian KL tensors must share dtype and device.")
    if mu_p.ndim < 1:
        raise ValueError("Gaussian means must have at least one dimension.")
    n = mu_p.shape[-1]
    if (
        mu_p.ndim < 1
        or mu_q.shape[-1:] != (n,)
        or covariance_p.shape[-2:] != (n, n)
        or covariance_q.shape[-2:] != (n, n)
    ):
        raise ValueError("Gaussian means and covariance dimensions must agree.")
    if not all(bool(torch.isfinite(value).all()) for value in values):
        raise ValueError("Gaussian KL inputs must be finite.")
    factor_p = _rs.tensor_psd_safe_cholesky(covariance_p)
    factor_q = _rs.tensor_psd_safe_cholesky(covariance_q)
    log_determinant = 2 * (
        factor_q.diagonal(dim1=-2, dim2=-1).log().sum(-1)
        - factor_p.diagonal(dim1=-2, dim2=-1).log().sum(-1)
    )
    trace = torch.cholesky_solve(covariance_p, factor_q).diagonal(dim1=-2, dim2=-1).sum(-1)
    difference = mu_q - mu_p
    alpha = torch.cholesky_solve(difference.unsqueeze(-1), factor_q).squeeze(-1)
    return 0.5 * (log_determinant + trace + (difference * alpha).sum(-1) - n)


def install_module(target: Any, fitted: Any) -> None:
    """Update caller-owned modules; Tensor modules retain exact leaf identities."""
    from botorch_rs.means import ConstantMean, ZeroMean

    if target is fitted:
        return
    source_parameters = dict(fitted.named_parameters())
    if isinstance(target, ConstantMean):
        target.raw_constant = source_parameters["raw_constant"].detach()
        return
    if isinstance(target, ZeroMean):
        return
    if isinstance(target, nn.Module):
        parameters = dict(target.named_parameters())
        if parameters.keys() != source_parameters.keys():
            raise ValueError("Fitted and original module parameter layouts differ.")
        with torch.no_grad():
            for name, value in parameters.items():
                value.copy_(source_parameters[name])
        return
    for name in dict(target.named_parameters()):
        if name not in source_parameters:
            raise ValueError(f"Missing fitted parameter {name!r}.")
        module = target
        *parents, attribute = name.split(".")
        for parent in parents:
            module = getattr(module, parent)
        # This explicit descriptor adapter exports fitted hyperparameter
        # state. Optimization and its gradients have already run on device.
        setattr(module, attribute, source_parameters[name].detach())


def fit_kernel(mean_module: Any, covar_module: Any, X_ref: Any, mu_em: Any, Sigma_em: Any) -> None:
    from botorch_rs.models.empirical_gps.em_coordinate_ascent import KLPriorFitMLL

    if X_ref.ndim != 2 or not X_ref.shape[0]:
        raise ValueError("X_ref must be a nonempty reference matrix.")
    mean = build_mean(mean_module, (), X_ref)
    kernel = build_kernel(covar_module, X_ref.shape[-1], (), X_ref)
    model = TorchSingleTaskGP(
        X_ref[:1],
        X_ref.new_zeros(1, 1),
        mean_module=mean,
        covar_module=kernel,
        outcome_transform=None,
    )
    mll = KLPriorFitMLL(model.likelihood, model, X_ref, mu_em, Sigma_em)
    model.likelihood.requires_grad_(False)
    fit_gpytorch_mll(mll)
    install_module(mean_module, model.mean_module)
    install_module(covar_module, model.covar_module)


def shared_models(
    datasets: Any, mean_module: Any, covar_module: Any, observation_noise: Any
) -> Any:
    like = validate_datasets(datasets)
    mean = build_mean(mean_module, (), like)
    kernel = build_kernel(covar_module, like.shape[-1], (), like)
    if observation_noise is not None:
        noise = torch.as_tensor(observation_noise, device=like.device, dtype=like.dtype)
        if (
            noise.numel() != 1
            or not bool(torch.isfinite(noise).all())
            or bool((noise < 1e-4).any())
        ):
            raise ValueError("observation_noise must be scalar and satisfy the lower bound 1e-4.")
    models = []
    for dataset in datasets:
        model = TorchSingleTaskGP(
            dataset.X,
            dataset.Y,
            mean_module=mean,
            covar_module=kernel,
            outcome_transform=None,
        )
        if observation_noise is not None:
            model.likelihood.noise = noise
            model.likelihood.raw_noise.requires_grad_(False)
        models.append(model)
    result = ModelListGP(*models)
    return result, TensorSharedMLL(result.likelihood, result, mean_module, covar_module)


class TensorSharedMLL(SumMarginalLogLikelihood):
    """A joint Tensor objective; shared mean/kernel leaves occur once."""

    def __init__(self, likelihood: Any, model: Any, mean_source: Any, kernel_source: Any) -> None:
        super().__init__(likelihood, model)
        self._mean_source, self._kernel_source = mean_source, kernel_source

    def eval(self) -> Any:
        # Converted native descriptors are boundary metadata. Keep them in
        # sync after a numerical fit, without rebinding independent models.
        install_module(self._mean_source, self.model.models[0].mean_module)
        install_module(self._kernel_source, self.model.models[0].covar_module)
        return super().eval()


def coordinate_ascent(
    datasets: Any,
    mean_module: Any,
    covar_module: Any,
    likelihood_noise: Any = None,
    num_em_iterations: int = 16,
    n_coord_ascent_iterations: int = 1,
    inducing_points: Any = None,
    optimize_inducing_points: bool = False,
    use_mean_prior: bool = False,
    use_covar_prior: bool = False,
    iw_nu: Any = None,
    em_convergence_tol: Any = 1e-6,
    enable_interpolation: bool = True,
    init_mode: str = "kernel",
    mll_max_iter: int = 50,
    mll_dataset_subsample: Any = None,
    mll_use_adam: Any = None,
    mll_adam_lr: float = 0.01,
    mll_adam_steps: int = 200,
    container_history: Any = None,
) -> Any:
    from botorch_rs.constraints import GreaterThan
    from botorch_rs.likelihoods import GaussianLikelihood

    like = validate_datasets(datasets)
    if inducing_points is None and optimize_inducing_points:
        raise ValueError("Optimizing inducing locations requires explicit inducing_points.")
    if (
        not isinstance(n_coord_ascent_iterations, int)
        or isinstance(n_coord_ascent_iterations, bool)
        or n_coord_ascent_iterations < 0
    ):
        raise ValueError("n_coord_ascent_iterations must be a nonnegative integer.")
    if mll_dataset_subsample is not None and (
        not isinstance(mll_dataset_subsample, int)
        or isinstance(mll_dataset_subsample, bool)
        or mll_dataset_subsample < 1
    ):
        raise ValueError("mll_dataset_subsample must be a positive integer.")
    if (
        not isinstance(mll_max_iter, int)
        or mll_max_iter < 0
        or not isinstance(mll_adam_steps, int)
        or mll_adam_steps < 1
        or not math.isfinite(mll_adam_lr)
        or mll_adam_lr <= 0
    ):
        raise ValueError("Coordinate fit iteration counts and learning rate are invalid.")
    if mll_use_adam is not None and not isinstance(mll_use_adam, bool):
        raise TypeError("mll_use_adam must be bool or None.")
    adam = optimize_inducing_points if mll_use_adam is None else mll_use_adam
    mean = build_mean(mean_module, (), like)
    kernel = build_kernel(covar_module, like.shape[-1], (), like)
    options = dict(
        likelihood_noise=likelihood_noise,
        inducing_points=inducing_points,
        num_em_iterations=num_em_iterations,
        use_mean_prior=use_mean_prior,
        use_covar_prior=use_covar_prior,
        iw_nu=iw_nu,
        em_convergence_tol=em_convergence_tol,
        enable_interpolation=enable_interpolation,
        init_mode=init_mode,
        backend="torch",
    )
    prior = pretrain_em_prior(datasets, cast(Any, mean), cast(Any, kernel), **options)
    if container_history is not None:
        container_history.append(copy.deepcopy(prior))
    for _ in range(n_coord_ascent_iterations):
        # Keep observation variance fixed during the kernel coordinate step.
        # None uses the same GaussianLikelihood raw initialization as upstream.
        likelihood = GaussianLikelihood(noise_constraint=GreaterThan(1e-16, transform=None))
        selected = datasets
        if mll_dataset_subsample is not None and mll_dataset_subsample < len(datasets):
            labels = torch.randperm(len(datasets), device=like.device)[:mll_dataset_subsample]
            # Python owns the heterogeneous dataset container; only its small
            # integer labels cross the boundary, never historical numerics.
            selected = [datasets[int(index)] for index in labels]
        fitting_prior = replace(prior, datasets=selected)
        model = EMEmpiricalGaussianProcess.from_pretrained(
            fitting_prior,
            prior.X_inducing[:1],
            like.new_zeros(1, 1),
            likelihood=likelihood,
            freeze_pretrained=False,
            learnable_inducing_points=optimize_inducing_points,
            backend="torch",
        )
        if likelihood_noise is not None:
            model.likelihood.noise = torch.as_tensor(
                likelihood_noise, device=like.device, dtype=like.dtype
            ).detach()
        model.likelihood.raw_noise.requires_grad_(False)
        mll = EMEmpiricalMarginalLogLikelihood(model.likelihood, model)
        # Indexed fixed moments may make the coordinate objective constant.
        # An exact no-op is valid; registered parameter priors can still give
        # this objective a genuine gradient, so inspect the actual loss.
        differentiable = mll().requires_grad
        if differentiable and adam:
            fit_gpytorch_mll_torch(
                mll,
                step_limit=mll_adam_steps,
                stopping_criterion=None,
                optimizer=lambda parameters: torch.optim.Adam(
                    parameters, lr=mll_adam_lr, foreach=True
                ),
            )
        elif differentiable:
            fit_gpytorch_mll_scipy(mll, options={"maxiter": mll_max_iter})
        install_module(mean, model.initial_mean_module)
        install_module(kernel, model.initial_covar_module)
        if optimize_inducing_points:
            options["inducing_points"] = model._X_inducing.detach().clone()
        prior = pretrain_em_prior(datasets, cast(Any, mean), cast(Any, kernel), **options)
        if container_history is not None:
            container_history.append(copy.deepcopy(prior))
    install_module(mean_module, mean)
    install_module(covar_module, kernel)
    return prior

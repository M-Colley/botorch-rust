# Copyright (c) Meta Platforms, Inc. and affiliates.
# Adapted from BoTorch b0dd01be under the MIT license (see LICENSE).
"""Device-resident multiobjective MES/JES and truncated Gaussian entropies."""

from __future__ import annotations

import inspect
import math
from math import pi
from typing import Any

import torch
from torch import Tensor, nn

from botorch_rs import _botorch_rs as _rs
from botorch_rs.acquisition._torch_backend_acquisition import _TensorAcquisition
from botorch_rs.exceptions.errors import UnsupportedError
from botorch_rs.posteriors._torch_backend import TorchGPyTorchPosterior
from botorch_rs.sampling.normal import SobolQMCNormalSampler


def _cdf_limits(value: Tensor, bounds: Tensor) -> Tensor:
    return torch.where(
        torch.isposinf(bounds), 1, torch.where(torch.isneginf(bounds), 0, torch.special.ndtr(value))
    )


def _bound_statistics(
    bounds: Tensor, mean: Tensor, variance: Tensor
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    bounds = bounds.unsqueeze(-4)
    finite = torch.isfinite(bounds)
    # Replace infinities BEFORE division; masking their standardized value
    # afterwards leaves 0*inf in variance derivatives.
    safe = torch.where(finite, bounds, 0)
    g = (safe - mean.unsqueeze(-2)) / variance.unsqueeze(-2).sqrt()
    gcdf = _cdf_limits(g, bounds)
    gpdf = torch.where(finite, (-0.5 * (g.square() + math.log(2 * pi))).exp(), 0)
    return g, gcdf, gpdf, g * gpdf


class TensorMultiObjectiveEntropy(_TensorAcquisition):
    """The four reference lower-bound estimators with full device graphs."""

    hypercell_bounds: Tensor
    pareto_sets: Tensor | None
    pareto_fronts: Tensor | None

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        nn.Module.__init__(self)
        bound = inspect.signature(inspect.getattr_static(self._public_class, "__init__")).bind(
            self, *args, **kwargs
        )
        bound.apply_defaults()
        values = bound.arguments
        self.initial_model = self.model = values["model"]
        if self.model.batch_shape:
            raise UnsupportedError("Multiobjective entropy requires an initially unbatched model.")
        self._name, self._is_analytic = self._public_class.__name__, False
        self._joint = "JointEntropySearch" in self._name
        cells = values["hypercell_bounds"]
        if (
            not isinstance(cells, Tensor)
            or cells.ndim != 4
            or cells.shape[1] != 2
            or not cells.shape[0]
            or not cells.shape[2]
            or cells.shape[-1] != self.model.num_outputs
            or bool(torch.isnan(cells).any())
            or bool((cells[:, 0] > cells[:, 1]).any())
        ):
            raise ValueError(
                "Hypercell bounds must have shape samples x 2 x boxes x outputs and ordered sides."
            )
        like = next(self.model.parameters(), cells)
        if cells.device != like.device or cells.dtype != like.dtype:
            raise ValueError("Hypercell bounds must share the model device and dtype.")
        self.num_pareto_samples = cells.shape[0]
        self.register_buffer("hypercell_bounds", cells)
        self.register_buffer("pareto_sets", values.get("pareto_sets"))
        self.register_buffer("pareto_fronts", values.get("pareto_fronts"))
        self.register_buffer("_X_pending", None)
        self.set_X_pending(values["X_pending"])
        self.estimation_type = values["estimation_type"]
        if self.estimation_type not in ("0", "LB", "LB2", "MC"):
            raise UnsupportedError("Entropy estimation_type must be '0', 'LB', 'LB2' or 'MC'.")
        self.sampler = values["sampler"]
        if self.sampler is None:
            seed = int(torch.randint(0, 1_000_000, (1,)).item())
            self.sampler = SobolQMCNormalSampler((values["num_samples"],), seed=seed)
        if len(self.sampler.sample_shape) != 1 or self.sampler.sample_shape[0] < 1:
            raise UnsupportedError("Entropy sampling requires one positive sample dimension.")
        if self._joint:
            sets, fronts = self.pareto_sets, self.pareto_fronts
            if (
                sets is None
                or fronts is None
                or sets.ndim != 3
                or fronts.ndim != 3
                or sets.shape[:2] != fronts.shape[:2]
                or sets.shape[0] != cells.shape[0]
                or fronts.shape[-1] != self.model.num_outputs
                or not sets.shape[1]
                or sets.device != cells.device
                or fronts.device != cells.device
                or sets.dtype != cells.dtype
                or fronts.dtype != cells.dtype
            ):
                raise ValueError(
                    "Pareto sets/fronts must have matching samples x points x d/m shapes."
                )
            children = getattr(self.model, "models", None)
            components = [self.model] if children is None else children
            if any(getattr(model, "_train_Yvar", None) is not None for model in components):
                raise UnsupportedError(
                    "JES conditioning on fixed-noise models needs explicit Pareto observation noise, as in BoTorch."
                )
            self.conditional_model = self.initial_model.condition_on_observations(
                sets if children is None else [sets] * len(children), fronts
            )

    def _model_posterior(self, model: Any, X: Tensor, noise: bool) -> TorchGPyTorchPosterior:
        posterior = model.posterior(X, observation_noise=noise)
        if not isinstance(posterior, TorchGPyTorchPosterior):
            raise UnsupportedError("Multiobjective entropy requires Gaussian device posteriors.")
        return posterior

    def _compute_posterior_statistics(self, X: Tensor) -> dict[str, Any]:
        eps = torch.finfo(X.dtype).eps
        initial = self._model_posterior(self.initial_model, X, True)
        # Preserve the existing reference/native multiobjective convention:
        # the Gaussian constant is M/2*(1+log(2pi)), including for q>1.
        initial_entropy = 0.5 * self.model.num_outputs * (1 + math.log(2 * pi))
        initial_entropy = (
            initial_entropy
            + 0.5 * torch.linalg.slogdet(initial.distribution.covariance_matrix).logabsdet
        )
        points = X.unsqueeze(-2)
        model = self.conditional_model if self._joint else self.initial_model
        if self._joint:
            points = points.unsqueeze(-3)
        noisy = self._model_posterior(model, points, True)
        latent = self._model_posterior(model, points, False)
        mean, variance, noisy_variance = (
            noisy.mean,
            latent.variance.clamp_min(eps),
            noisy.variance.clamp_min(eps),
        )
        if self._joint:
            mean, variance, noisy_variance = (
                value.swapaxes(-4, -3) for value in (mean, variance, noisy_variance)
            )
        else:
            shape = mean.shape[:-3] + (self.num_pareto_samples,) + mean.shape[-3:]
            mean, variance, noisy_variance = (
                value.unsqueeze(-4).expand(shape) for value in (mean, variance, noisy_variance)
            )
        observation_noise = noisy_variance - variance
        if self._joint:
            observation_noise = observation_noise.clamp_min(eps)
        return dict(
            initial_entropy=initial_entropy,
            posterior_mean=mean,
            posterior_variance=variance,
            observation_noise=observation_noise,
            posterior_with_noise=noisy,
        )

    def _compute_monte_carlo_variables(self, posterior: Any) -> tuple[Tensor, Tensor]:
        samples = self.sampler(posterior)
        covariance = posterior.distribution.covariance_matrix
        root = _rs.tensor_psd_safe_cholesky(covariance)
        difference = (samples - posterior.mean).squeeze(-2)
        whitened = torch.linalg.solve_triangular(
            root, difference.unsqueeze(-1), upper=False
        ).squeeze(-1)
        log_probability = -0.5 * (
            self.model.num_outputs * math.log(2 * pi)
            + 2 * root.diagonal(dim1=-2, dim2=-1).log().sum(-1)
            + whitened.square().sum(-1)
        )
        if self._joint:
            return samples.swapaxes(-4, -3), log_probability.swapaxes(-2, -1)
        sample_shape = samples.shape[:-3] + (self.num_pareto_samples,) + samples.shape[-3:]
        log_shape = (
            log_probability.shape[:-1] + (self.num_pareto_samples,) + log_probability.shape[-1:]
        )
        return samples.unsqueeze(-4).expand(sample_shape), log_probability.unsqueeze(-2).expand(
            log_shape
        )

    def _compute_lower_bound_information_gain(self, X: Tensor) -> Tensor:
        statistics = self._compute_posterior_statistics(X)
        arguments = dict(
            hypercell_bounds=self.hypercell_bounds,
            mean=statistics["posterior_mean"],
            variance=statistics["posterior_variance"],
            observation_noise=statistics["observation_noise"],
        )
        if self.estimation_type == "0":
            entropy = _compute_entropy_noiseless(**arguments)
        elif self.estimation_type in ("LB", "LB2"):
            entropy = _compute_entropy_upper_bound(
                **arguments, only_diagonal=self.estimation_type == "LB2"
            )
        else:
            samples, log_probability = self._compute_monte_carlo_variables(
                statistics["posterior_with_noise"]
            )
            entropy = _compute_entropy_monte_carlo(
                **arguments, samples=samples, samples_log_prob=log_probability
            )
        return statistics["initial_entropy"] - entropy.sum(-1)

    def forward(self, X: Tensor) -> Tensor:
        return self._compute_lower_bound_information_gain(self._candidates_tensor(X))


def _compute_entropy_noiseless(
    hypercell_bounds: Tensor, mean: Tensor, variance: Tensor, observation_noise: Tensor
) -> Tensor:
    tkwargs: dict[str, Any] = {"dtype": hypercell_bounds.dtype, "device": hypercell_bounds.device}
    CLAMP_LB = torch.finfo(tkwargs["dtype"]).eps
    variance_plus_noise = variance + observation_noise
    g, gcdf, gpdf, g_times_gpdf = _bound_statistics(hypercell_bounds, mean, variance)
    Wjm = (gcdf[..., 1, :, :] - gcdf[..., 0, :, :]).clamp_min(CLAMP_LB)
    Vjm = g_times_gpdf[..., 1, :, :] - g_times_gpdf[..., 0, :, :]
    Wj = torch.exp(torch.sum(torch.log(Wjm), dim=-1, keepdim=True))
    W = torch.sum(Wj, dim=-2, keepdim=True).clamp_max(1.0)
    ratios = 0.5 * (Wj * (Vjm / Wjm)) / W
    ratio_term = torch.sum(ratios, dim=(-2, -1), keepdim=True)
    log_term = 0.5 * torch.log(variance_plus_noise).sum(-1, keepdim=True)
    log_term = log_term + torch.log(W)
    M_plus_K = mean.shape[-1]
    add_term = 0.5 * M_plus_K * (1 + torch.log(torch.ones(1, **tkwargs) * 2 * pi))
    entropy = add_term + (log_term - ratio_term).squeeze(-1).squeeze(-1)
    return entropy.mean(-2)


def _compute_entropy_upper_bound(
    hypercell_bounds: Tensor,
    mean: Tensor,
    variance: Tensor,
    observation_noise: Tensor,
    only_diagonal: bool = False,
) -> Tensor:
    tkwargs: dict[str, Any] = {"dtype": hypercell_bounds.dtype, "device": hypercell_bounds.device}
    CLAMP_LB = torch.finfo(tkwargs["dtype"]).eps
    variance_plus_noise = variance + observation_noise
    g, gcdf, gpdf, g_times_gpdf = _bound_statistics(hypercell_bounds, mean, variance)
    Wjm = (gcdf[..., 1, :, :] - gcdf[..., 0, :, :]).clamp_min(CLAMP_LB)
    Vjm = g_times_gpdf[..., 1, :, :] - g_times_gpdf[..., 0, :, :]
    Gjm = gpdf[..., 1, :, :] - gpdf[..., 0, :, :]
    Wj = torch.exp(torch.sum(torch.log(Wjm), dim=-1, keepdim=True))
    W = torch.sum(Wj, dim=-2, keepdim=True).clamp_max(1.0)
    Cjm = Gjm / Wjm
    Rjm = Cjm * Wj / W
    mom1 = mean - torch.sqrt(variance) * Rjm.sum(-2, keepdim=True)
    diag_weighted_sum = (Wj * variance * Vjm / Wjm / W).sum(-2, keepdim=True)
    if only_diagonal:
        mean_squared = mean.pow(2)
        cross_sum = -2 * (mean * torch.sqrt(variance) * Rjm).sum(-2, keepdim=True)
        mom2 = variance_plus_noise - diag_weighted_sum + cross_sum + mean_squared
        var = (mom2 - mom1.pow(2)).clamp_min(CLAMP_LB)
        log_det_term = 0.5 * torch.log(var).sum(dim=-1).squeeze(-1)
    else:
        cross_mom1 = torch.einsum("...i,...j->...ij", mom1, mom1)
        mom2_cross1 = -torch.einsum("...i,...j->...ij", mean, torch.sqrt(variance) * Cjm)
        mom2_cross2 = -torch.einsum("...i,...j->...ji", mean, torch.sqrt(variance) * Cjm)
        mom2_mean_squared = torch.einsum("...i,...j->...ij", mean, mean)
        mom2_weighted_sum = ((mom2_cross1 + mom2_cross2) * Wj.unsqueeze(-1) / W.unsqueeze(-1)).sum(
            -3, keepdim=True
        )
        mom2_weighted_sum = mom2_weighted_sum + mom2_mean_squared
        mom2_off_diag = torch.einsum(
            "...i,...j->...ij", torch.sqrt(variance) * Cjm, torch.sqrt(variance) * Cjm
        )
        mom2_off_diag_sum = (mom2_off_diag * Wj.unsqueeze(-1) / W.unsqueeze(-1)).sum(
            -3, keepdim=True
        )
        init_diag = torch.diagonal(mom2_off_diag_sum, dim1=-2, dim2=-1)
        diag_weighted_sum = torch.diag_embed(variance_plus_noise - diag_weighted_sum - init_diag)
        mom2 = mom2_weighted_sum + mom2_off_diag_sum + diag_weighted_sum
        var = (mom2 - cross_mom1).squeeze(-3)
        jitter_diag = 1e-06 * torch.diag_embed(torch.ones(var.shape[:-1], **tkwargs))
        log_det_term = 0.5 * torch.logdet(var + jitter_diag)
    M_plus_K = mean.shape[-1]
    add_term = 0.5 * M_plus_K * (1 + torch.log(torch.ones(1, **tkwargs) * 2 * pi))
    entropy = add_term + log_det_term
    return entropy.mean(-2)


def _compute_entropy_monte_carlo(
    hypercell_bounds: Tensor,
    mean: Tensor,
    variance: Tensor,
    observation_noise: Tensor,
    samples: Tensor,
    samples_log_prob: Tensor,
) -> Tensor:
    tkwargs: dict[str, Any] = {"dtype": hypercell_bounds.dtype, "device": hypercell_bounds.device}
    CLAMP_LB = torch.finfo(tkwargs["dtype"]).eps
    variance_plus_noise = variance + observation_noise
    g, gcdf, _, _ = _bound_statistics(hypercell_bounds, mean, variance)
    rho = torch.sqrt(variance / variance_plus_noise)
    Wjm = (gcdf[..., 1, :, :] - gcdf[..., 0, :, :]).clamp_min(CLAMP_LB)
    Wj = torch.exp(torch.sum(torch.log(Wjm), dim=-1, keepdim=True))
    W = torch.sum(Wj, dim=-2, keepdim=True).clamp_max(1.0)
    g = g.unsqueeze(0)
    rho = rho.unsqueeze(0).unsqueeze(-2)
    z = ((samples - mean) / torch.sqrt(variance_plus_noise)).unsqueeze(-2)
    g_new = (g - rho * z) / torch.sqrt((1 - rho * rho).clamp_min(CLAMP_LB))
    gcdf_new = _cdf_limits(g_new, hypercell_bounds.unsqueeze(-4).unsqueeze(0))
    Wjm_new = (gcdf_new[..., 1, :, :] - gcdf_new[..., 0, :, :]).clamp_min(CLAMP_LB)
    Wj_new = torch.exp(torch.sum(torch.log(Wjm_new), dim=-1, keepdim=True))
    W_new = torch.sum(Wj_new, dim=-2, keepdim=True).clamp_max(1.0)
    W_ratio = torch.exp(torch.log(W_new) - torch.log(W).unsqueeze(0))
    samples_log_prob = samples_log_prob.unsqueeze(-1).unsqueeze(-1)
    log_term = torch.log(W_new) + samples_log_prob
    mc_estimate = -(W_ratio * log_term).mean(0)
    entropy = (mc_estimate + torch.log(W)).squeeze(-1).squeeze(-1)
    return entropy.mean(-2)

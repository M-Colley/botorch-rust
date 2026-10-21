"""Fantasy-based single-point noisy improvement with actual Tensor GP models."""

from __future__ import annotations

import copy
from contextlib import nullcontext
from typing import Any, cast

import torch
from torch import Tensor, nn

from botorch_rs.acquisition._torch_backend_acquisition import (
    _log_ei,
    _logmeanexp,
    _TensorAcquisition,
)
from botorch_rs.exceptions.errors import UnsupportedError
from botorch_rs.exceptions.warnings import legacy_ei_numerics_warning
from botorch_rs.models.gp_regression import SingleTaskGP


def _clone_module(module: nn.Module | None) -> Any:
    if module is None:
        return None
    # Fitted transform statistics may be non-leaf graph tensors. Fantasy
    # hyperparameters/statistics are copies, as in the reference factory.
    memo = {id(buffer): buffer.detach().clone() for buffer in module.buffers()}
    return copy.deepcopy(module, memo)


def _noiseless_fantasies(model: Any, X: Tensor, Y: Tensor) -> Any:
    fantasy = copy.copy(model)
    fantasy._buffers = model._buffers.copy()
    fantasy._parameters = model._parameters.copy()
    fantasy._modules = {name: _clone_module(module) for name, module in model._modules.items()}
    fantasy._train_X = X
    fantasy._train_Y = Y.unsqueeze(-1)
    fantasy._train_Yvar = torch.full_like(fantasy._train_Y, 1e-7)
    fantasy._transform_fit_X = model._transform_fit_X.detach().clone()
    fantasy._transform_fit_Y = model._transform_fit_Y.detach().clone()
    fantasy._transform_fit_versions = {}
    fantasy._prediction_cache = None
    # The source replaces fixed noise *after* construction, so its 1e-7 value
    # bypasses the constructor's usual minimum and is in transformed units.
    fantasy._noise_override = torch.full_like(Y, 1e-7)
    fantasy.likelihood.noise = fantasy._noise_override
    return fantasy.eval()


class TensorNoisyAnalytic(_TensorAcquisition):
    def __init__(
        self,
        model: Any,
        X_observed: Tensor,
        num_fantasies: int = 20,
        maximize: bool = True,
        posterior_transform: Any = None,
        *,
        sampler: Any = None,
    ) -> None:
        nn.Module.__init__(self)
        from botorch_rs.models._torch_backend_modules import TorchFixedNoiseLikelihood
        from botorch_rs.sampling.normal import SobolQMCNormalSampler

        if not isinstance(model, SingleTaskGP) or model.num_outputs != 1:
            raise UnsupportedError("Fantasy-based NEI requires a single-output SingleTaskGP.")
        if getattr(model, "_train_Yvar", None) is None or not isinstance(
            model.likelihood, TorchFixedNoiseLikelihood
        ):
            raise UnsupportedError("Fantasy-based NEI requires known observation noise.")
        if X_observed.ndim < 2 or X_observed.shape[-1] != cast(Any, model)._train_X.shape[-1]:
            raise ValueError("X_observed must have shape (..., n, d).")
        sampler = sampler or SobolQMCNormalSampler((int(num_fantasies),))
        if tuple(sampler.sample_shape) != (int(num_fantasies),):
            raise ValueError("The sampler must draw num_fantasies samples.")
        with nullcontext() if X_observed.requires_grad else torch.no_grad():
            posterior = model.posterior(X_observed)
        targets = sampler(posterior).squeeze(-1)
        inputs = X_observed.expand(int(num_fantasies), *X_observed.shape)
        self.model = _noiseless_fantasies(model, inputs, targets)
        self.sampler = sampler
        self.posterior_transform = posterior_transform
        self.maximize = maximize
        self.num_fantasies = int(num_fantasies)
        self._tensor_backend = "torch"
        self._is_analytic = True
        self._name = self._public_class.__name__
        self._log = self._name.startswith("Log")
        self.register_buffer("_X_pending", None)
        best = targets.max(-1).values if maximize else targets.min(-1).values
        self.register_buffer("best_f", best)
        if not self._log:
            legacy_ei_numerics_warning("NoisyExpectedImprovement")

    def forward(self, X: Tensor) -> Tensor:
        X = self._candidates_tensor(X)
        posterior = self.model.posterior(
            X.unsqueeze(-3), posterior_transform=self.posterior_transform
        )
        mean = posterior.mean.squeeze(-1).squeeze(-1)
        sigma = posterior.variance.clamp_min(1e-12).sqrt().squeeze(-1).squeeze(-1)
        improvement = (mean - self.best_f.to(mean)) * (1 if self.maximize else -1)
        u = improvement / sigma
        if self._log:
            return _logmeanexp(_log_ei(u) + sigma.log(), (-1,))
        density = (-0.5 * u.square()).exp() / (2 * torch.pi) ** 0.5
        return (sigma * (density + u * torch.special.ndtr(u))).mean(-1)

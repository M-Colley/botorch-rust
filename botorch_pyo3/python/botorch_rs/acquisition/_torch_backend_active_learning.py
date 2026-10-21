"""Device-resident feasibility and posterior-uncertainty acquisitions."""

from __future__ import annotations

import inspect
from typing import Any

import torch
from torch import Tensor, nn

from botorch_rs.acquisition._torch_backend_acquisition import (
    _average_ensemble,
    _TensorAcquisition,
)
from botorch_rs.exceptions.errors import UnsupportedError


class TensorActiveLearning(_TensorAcquisition):
    """Preserve posterior, fantasy, link-objective and constraint graphs."""

    mc_points: Tensor

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        from botorch_rs.acquisition.objective import IdentityMCObjective
        from botorch_rs.sampling.normal import SobolQMCNormalSampler

        bound = inspect.signature(inspect.getattr_static(self._public_class, "__init__")).bind(
            self, *args, **kwargs
        )
        bound.apply_defaults()
        values = dict(bound.arguments)
        values.pop("self")
        name = self._public_class.__name__
        if name == "qLogProbabilityOfFeasibility":
            if values["constraints"] is None:
                raise TypeError("qLogProbabilityOfFeasibility requires `constraints`.")
            # The public objective argument is deliberately unused by this
            # family, including for multioutput models, as in the source.
            values["objective"] = IdentityMCObjective()
            super().__init__(**values)
            if values["sampler"] is None:
                self.sampler = None
            return
        nn.Module.__init__(self)
        self._name = name
        self._tensor_backend = "torch"
        self._is_analytic = self._log = False
        self.model = values["model"]
        self.posterior_transform = values.get("posterior_transform")
        self.objective = values.get("objective")
        sampler = values.get("sampler")
        self.sampler = (
            SobolQMCNormalSampler(sample_shape=torch.Size([1]))
            if sampler is None and name.startswith("qNeg")
            else sampler
        )
        self.register_buffer("_X_pending", None)
        self.set_X_pending(values.get("X_pending"))
        if name == "qNegIntegratedPosteriorVariance":
            if self.model.num_outputs != 1 and self.posterior_transform is None:
                raise UnsupportedError(
                    "Must specify a posterior transform when using a multi-output model."
                )
            mc_points = values["mc_points"]
            if not isinstance(mc_points, Tensor) or mc_points.ndim < 2:
                raise ValueError("mc_points must be a Tensor with shape (..., N, d).")
            if mc_points.shape[-2] == 0:
                raise ValueError("mc_points must contain at least one integration point.")
            self.register_buffer("mc_points", mc_points)

    @property
    def sample_shape(self) -> Any:
        return torch.Size([512]) if self.sampler is None else self.sampler.sample_shape

    def get_posterior_samples(self, posterior: Any) -> Tensor:
        if self.sampler is None:
            from botorch_rs.sampling.get_sampler import get_sampler

            self.sampler = get_sampler(posterior, sample_shape=self.sample_shape)
        return self.sampler(posterior)

    def _non_reduced_forward(self, X: Tensor) -> Tensor:
        if self._name == "qLogProbabilityOfFeasibility":
            return self._constraint_indicator(
                self.get_posterior_samples(self._posterior(X)), log=True
            )
        return super()._non_reduced_forward(X)

    def forward(self, X: Tensor) -> Tensor:
        if self._name == "qLogProbabilityOfFeasibility":
            return super().forward(X)
        X = self._candidates_tensor(X)
        if self._name == "PairwiseMCPosteriorVariance":
            if not X.shape[-2] or X.shape[-2] % 2:
                raise RuntimeError("q must be a multiple of 2 for PairwiseMCPosteriorVariance")
            samples = self.get_posterior_samples(self.model.posterior(X))
            differences = samples[..., ::2, :] - samples[..., 1::2, :]
            value = self.objective(differences).var(dim=0).mean(dim=-1)
            return _average_ensemble(value, self.model)
        # Use the actual model fantasy protocol. In an exact Gaussian model
        # the variance is independent of fantasy targets, but this also
        # preserves the semantics of supplied samplers and custom models.
        fantasy = self.model.fantasize(X=X, sampler=self.sampler)
        batch_singletons = (1,) * len(X.shape[:-2])
        points = self.mc_points.to(X)
        if points.shape[-1] != X.shape[-1]:
            raise ValueError("mc_points and candidates must have the same input dimension.")
        if self.model.num_outputs > 1:
            points = points.reshape(-1, *batch_singletons, 1, X.shape[-1])
        else:
            points = points.reshape(*batch_singletons, -1, X.shape[-1])
        variance = fantasy.posterior(points, posterior_transform=self.posterior_transform).variance
        value = (
            -variance.mean(dim=0).squeeze(-1).squeeze(-1)
            if self.model.num_outputs > 1
            else -variance.mean(dim=-2).squeeze(-1).squeeze(0)
        )
        return _average_ensemble(value, self.model)

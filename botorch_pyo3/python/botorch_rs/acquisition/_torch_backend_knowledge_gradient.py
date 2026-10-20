"""Device-native one-shot knowledge gradient with functional fantasy graphs."""

from __future__ import annotations

import copy
import inspect
import warnings
from typing import Any

import torch
from torch import Tensor, nn

from botorch_rs.exceptions.errors import UnsupportedError
from botorch_rs.exceptions.warnings import BotorchWarning


class TensorKnowledgeGradient(nn.Module):
    """Mixin implementing the Torch variants of the public KG classes.

    Fantasy observations remain in the autograd graph. This includes the
    covariance dependence of the inner MC value function on the outer points.
    """

    _public_class: type

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        nn.Module.__init__(self)
        bound = inspect.signature(inspect.getattr_static(self._public_class, "__init__")).bind(
            self, *args, **kwargs
        )
        bound.apply_defaults()
        values = dict(bound.arguments)
        from botorch_rs.acquisition.knowledge_gradient import _is_mc_objective
        from botorch_rs.sampling.normal import SobolQMCNormalSampler

        self._tensor_backend = "torch"
        self.model = values["model"]
        self.sampler = values["sampler"]
        count = values["num_fantasies"]
        if self.sampler is None:
            if count is None:
                raise ValueError("Must specify num_fantasies if no sampler is provided.")
            self.sampler = SobolQMCNormalSampler(sample_shape=(count,))
        shape = tuple(self.sampler.sample_shape)
        if len(shape) != 1 or shape[0] <= 0:
            raise ValueError("The fantasy sampler must have one positive sample dimension.")
        if count is not None and shape != (count,):
            raise ValueError(f"The sampler shape must match num_fantasies={count}.")
        self.num_fantasies = shape[0]
        self.objective = values["objective"]
        self.posterior_transform = values["posterior_transform"]
        self.inner_sampler = values["inner_sampler"]
        if self.objective is not None:
            if not _is_mc_objective(self.objective):
                raise UnsupportedError("Knowledge gradient requires an MCAcquisitionObjective.")
            if self.inner_sampler is None:
                self.inner_sampler = SobolQMCNormalSampler(sample_shape=(128,))
        elif self.model.num_outputs != 1 and not getattr(
            self.posterior_transform, "scalarize", False
        ):
            raise UnsupportedError("A multi-output KG model requires a scalarizing transform.")
        value = values["current_value"]
        self.register_buffer(
            "current_value",
            None
            if value is None
            else torch.as_tensor(value, dtype=None if isinstance(value, Tensor) else torch.float64),
        )
        self.register_buffer("_X_pending", None)
        self.set_X_pending(values["X_pending"])
        self.project = values.get("project")
        self.expand = values.get("expand")
        self.cost_aware_utility = values.get("cost_aware_utility")
        self.valfunc_cls = values.get("valfunc_cls")
        self.valfunc_argfac = values.get("valfunc_argfac")
        self._cost_sampler: Any = None
        if self.current_value is None and self.cost_aware_utility is not None:
            raise UnsupportedError("Cost-aware KG requires current_value to be specified.")

    @property
    def X_pending(self) -> Any:
        return self._X_pending

    def set_X_pending(self, X_pending: Any = None) -> None:
        if X_pending is not None:
            if not isinstance(X_pending, Tensor) or X_pending.ndim < 2:
                raise ValueError("X_pending must be a tensor with at least two dimensions.")
            if X_pending.requires_grad:
                warnings.warn(
                    "Pending points are copied without gradients.", BotorchWarning, stacklevel=2
                )
            X_pending = X_pending.detach().clone()
        self._X_pending = X_pending

    @property
    def cost_sampler(self) -> Any:
        if self._cost_sampler is None:
            self._cost_sampler = copy.deepcopy(self.sampler)
        return self._cost_sampler

    def get_augmented_q_batch_size(self, q: int) -> int:
        return q + self.num_fantasies

    def extract_candidates(self, X_full: Tensor) -> Tensor:
        return X_full[..., : -self.num_fantasies, :]

    def _value_function(self, model: Any) -> Any:
        from botorch_rs.acquisition.analytic import PosteriorMean
        from botorch_rs.acquisition.monte_carlo import MCAcquisitionFunction, qSimpleRegret

        if self.valfunc_cls is not None:
            arguments = {"model": model, "posterior_transform": self.posterior_transform}
            if issubclass(self.valfunc_cls, MCAcquisitionFunction):
                arguments.update(sampler=self.inner_sampler, objective=self.objective)
            if self.valfunc_argfac is not None:
                arguments.update(self.valfunc_argfac(model=model))
            return self.valfunc_cls(**arguments)
        if self.objective is None:
            return PosteriorMean(model=model, posterior_transform=self.posterior_transform)
        return qSimpleRegret(
            model=model,
            sampler=self.inner_sampler,
            objective=self.objective,
            posterior_transform=self.posterior_transform,
        )

    def forward(self, X: Tensor) -> Tensor:
        if not isinstance(X, Tensor) or X.ndim < 2:
            raise ValueError("X must have shape batch_shape x (q + num_fantasies) x d.")
        if X.shape[-2] <= self.num_fantasies:
            raise ValueError("Knowledge gradient requires at least one actual candidate.")
        X = X.unsqueeze(0) if X.ndim == 2 else X
        actual = self.extract_candidates(X)
        fantasy_points = X[..., -self.num_fantasies :, :].movedim(-2, 0).unsqueeze(-2)
        evaluated = actual
        if self.X_pending is not None:
            pending = self.X_pending.to(X)
            batch = torch.broadcast_shapes(actual.shape[:-2], pending.shape[:-2])
            evaluated = torch.cat(
                (
                    actual.expand(*batch, *actual.shape[-2:]),
                    pending.expand(*batch, *pending.shape[-2:]),
                ),
                dim=-2,
            )
        if self.expand is not None:
            evaluated = self.expand(evaluated)
        model = self.model.fantasize(X=evaluated, sampler=self.sampler, propagate_grads=True)
        if self.project is not None:
            fantasy_points = self.project(fantasy_points)
        values = self._value_function(model)(fantasy_points)
        if self.current_value is not None:
            values = values - self.current_value.to(values)
        if self.cost_aware_utility is not None:
            values = self.cost_aware_utility(X=actual, deltas=values, sampler=self.cost_sampler)
        return values.mean(dim=0)

    def evaluate(self, X: Any, bounds: Any, **kwargs: Any) -> Any:
        raise UnsupportedError(
            "Torch KG currently supports the augmented one-shot forward method. "
            "The separately optimized evaluate method is not yet implemented."
        )

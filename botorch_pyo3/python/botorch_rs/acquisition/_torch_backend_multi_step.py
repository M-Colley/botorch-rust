"""Functional device fantasy trees for one-shot multi-step lookahead.

The recursion follows BoTorch's MIT-licensed multi-step algorithm. Every
stage calls a port acquisition/model; inference never calls upstream BoTorch.
"""

from __future__ import annotations

import inspect
from typing import Any

import torch
from torch import Tensor, nn

from botorch_rs.acquisition._torch_backend_acquisition import _TensorAcquisition
from botorch_rs.acquisition.monte_carlo import MCAcquisitionFunction
from botorch_rs.exceptions.errors import UnsupportedError
from botorch_rs.sampling.normal import SobolQMCNormalSampler


def _normal_sampler(count: int) -> SobolQMCNormalSampler:
    return SobolQMCNormalSampler((count,), seed=int(torch.randint(0, 1_000_000, (1,)).item()))


def _stage_value(
    model: Any,
    X: Tensor,
    cls: Any,
    factory: Any,
    sampler: Any,
    objective: Any,
    posterior_transform: Any,
) -> Tensor | None:
    if cls is None:
        return None
    arguments = dict(model=model, posterior_transform=posterior_transform)
    if issubclass(cls, MCAcquisitionFunction):
        arguments.update(sampler=sampler, objective=objective)
    extra = {} if factory is None else factory(model=model, X=X)
    return cls(**arguments, **extra)(X)


def _sample_weights(previous: Tensor, sampler: Any) -> Tensor:
    count = int(sampler.sample_shape[0])
    values = getattr(sampler, "base_weights", None)
    weights = (
        torch.ones(count, dtype=previous.dtype, device=previous.device)
        if values is None
        else torch.as_tensor(values, dtype=previous.dtype, device=previous.device)
    )
    if (
        weights.numel() != count
        or not bool(torch.isfinite(weights).all())
        or not bool(weights.sum() != 0)
    ):
        raise ValueError("Fantasy base_weights must be a finite vector with nonzero sum.")
    return previous * (weights / weights.sum()).reshape(count, *((1,) * previous.ndim))


class TensorMultiStepLookahead(_TensorAcquisition):
    """Recompute conditional stages without detaching any fantasy dependency."""

    _num_auxiliary: int
    get_multi_step_tree_input_representation: Any

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        from botorch_rs.acquisition.analytic import PosteriorMean
        from botorch_rs.acquisition.knowledge_gradient import _is_mc_objective
        from botorch_rs.acquisition.multi_step_lookahead import _construct_inner_samplers

        nn.Module.__init__(self)
        bound = inspect.signature(inspect.getattr_static(self._public_class, "__init__")).bind(
            self, *args, **kwargs
        )
        bound.apply_defaults()
        values = bound.arguments
        self.model = values["model"]
        self._name, self._is_analytic = "qMultiStepLookahead", False
        self.batch_sizes = list(values["batch_sizes"])
        self.objective, self.posterior_transform = (
            values["objective"],
            values["posterior_transform"],
        )
        if self.objective is not None and not _is_mc_objective(self.objective):
            raise UnsupportedError(
                "Multi-step lookahead requires an MC objective or a posterior transform."
            )
        counts, samplers = values["num_fantasies"], values["samplers"]
        if (counts is None) == (samplers is None):
            raise UnsupportedError("Specify exactly one of num_fantasies or samplers.")
        if samplers is None:
            if any(int(n) != n or n < 1 for n in counts):
                raise ValueError("num_fantasies must contain positive integers.")
            samplers = [_normal_sampler(n) for n in counts]
        else:
            samplers = list(samplers)
        if any(len(s.sample_shape) != 1 or s.sample_shape[0] < 1 for s in samplers):
            raise ValueError("Fantasy samplers must have one positive sample dimension.")
        self.samplers = samplers
        self.num_fantasies = [int(s.sample_shape[0]) for s in samplers]
        steps = len(self.batch_sizes)
        if len(samplers) != steps or any(q < 1 or int(q) != q for q in self.batch_sizes):
            raise ValueError("Positive batch_sizes must match the number of fantasy samplers.")
        stages = values["valfunc_cls"]
        self._valfunc_cls = [None] * steps + [PosteriorMean] if stages is None else list(stages)
        factories = values["valfunc_argfacs"]
        self._valfunc_argfacs = [None] * (steps + 1) if factories is None else list(factories)
        inner_counts = values["inner_mc_samples"]
        inner_counts = [None] * (steps + 1) if inner_counts is None else list(inner_counts)
        if any(
            len(v) != steps + 1 for v in (self._valfunc_cls, self._valfunc_argfacs, inner_counts)
        ):
            raise ValueError(
                "Stage classes, factories and inner sample counts must have k+1 entries."
            )
        if all(cls is None for cls in self._valfunc_cls):
            raise ValueError("At least one stage must have a value function.")
        if any(
            cls is not None and cls.__module__.startswith("botorch.") for cls in self._valfunc_cls
        ):
            raise UnsupportedError("Use port acquisition classes for device stage value functions.")
        checked = _construct_inner_samplers(
            self.batch_sizes, self._valfunc_cls, inner_counts, self.objective
        )
        self.inner_samplers = [
            None if s is None else _normal_sampler(int(s.sample_shape[0])) for s in checked
        ]
        self._collapse_fantasy_base_samples = bool(values["collapse_fantasy_base_samples"])
        self.register_buffer("_X_pending", None)
        self.set_X_pending(values["X_pending"])

    def _set_samplers_batch_range(self, batch_shape: Any) -> None:
        # The final base-sample event axis is q for scalar posteriors and q,m
        # for multitask posteriors. Determine the appropriate trailing boundary
        # at the posterior, rather than retaining an MCMC/fantasy axis by accident.
        self._query_batch_ndim = len(batch_shape)

    def _set_sampler_range(self, sampler: Any, model: Any, X: Tensor) -> None:
        if self._collapse_fantasy_base_samples:
            return
        posterior = model.posterior(X)
        event = 2 if getattr(posterior, "_is_mt", False) else 1
        sampler.batch_range_override = (-event - self._query_batch_ndim, -event)

    def _tree(self, X: Tensor) -> list[Tensor]:
        if not isinstance(X, Tensor) or X.ndim < 2 or X.shape[-2] <= self._num_auxiliary:
            raise ValueError("An augmented tree requires real candidates and all fantasy designs.")
        X = X.unsqueeze(0) if X.ndim == 2 else X
        self._set_samplers_batch_range(X.shape[:-2])
        return self.get_multi_step_tree_input_representation(X)

    def forward(self, X: Tensor) -> Tensor:
        tree = self._tree(X)
        model = self.model
        weights = torch.ones(tree[0].shape[:-2], dtype=X.dtype, device=X.device)
        running: Tensor | None = None
        for i, inputs in enumerate(tree):
            value = _stage_value(
                model,
                inputs,
                self._valfunc_cls[i],
                self._valfunc_argfacs[i],
                self.inner_samplers[i],
                self.objective,
                self.posterior_transform,
            )
            if value is not None:
                running = value if running is None else running + value
            if i < len(self.samplers):
                sampler = self.samplers[i]
                self._set_sampler_range(sampler, model, inputs)
                model = model.fantasize(inputs, sampler, propagate_grads=True)
                weights = _sample_weights(weights, sampler)
        assert running is not None
        weighted = running * weights
        return weighted.sum(tuple(range(len(self.samplers)))) if self.samplers else weighted

    def get_induced_fantasy_model(self, X: Tensor) -> Any:
        model = self.model
        for inputs, sampler in zip(self._tree(X), self.samplers, strict=False):
            self._set_sampler_range(sampler, model, inputs)
            model = model.fantasize(inputs, sampler, propagate_grads=True)
        return model

    def extract_candidates(self, X_full: Tensor) -> Tensor:
        return X_full if self._num_auxiliary == 0 else X_full[..., : -self._num_auxiliary, :]

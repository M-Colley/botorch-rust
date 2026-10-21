"""Device hypervolume knowledge gradient with functional fantasy models."""

from __future__ import annotations

import inspect
import math
from typing import Any

import torch
from torch import Tensor, nn

from botorch_rs.acquisition._torch_backend_acquisition import _TensorAcquisition
from botorch_rs.acquisition._torch_backend_entropy import _clone_sampler
from botorch_rs.exceptions.errors import BotorchTensorDimensionError, UnsupportedError
from botorch_rs.models._torch_backend_cost import TorchDeterministicCostPosterior
from botorch_rs.posteriors._torch_backend import TorchGPyTorchPosterior
from botorch_rs.sampling.list_sampler import ListSampler
from botorch_rs.sampling.normal import SobolQMCNormalSampler


class _MeanSampler(nn.Module):
    sample_shape = torch.Size([1])

    def forward(self, posterior: Any) -> Tensor:
        return posterior.mean.unsqueeze(0)


class _EmptyPartition:
    def __init__(self, ref_point: Tensor) -> None:
        self.ref_point, self.num_outcomes = ref_point, ref_point.numel()

    def get_hypercell_bounds(self) -> Tensor:
        return torch.stack(
            (self.ref_point.unsqueeze(0), torch.full_like(self.ref_point.unsqueeze(0), torch.inf))
        )


class _CostListSampler(ListSampler):
    """Clone the fantasy streams, retaining a Gaussian cost's joint law."""

    def forward(self, posterior: Any) -> Tensor:
        if isinstance(posterior, TorchDeterministicCostPosterior):
            return posterior.rsample(self.sample_shape)
        if isinstance(posterior, TorchGPyTorchPosterior):
            if posterior._m == 1:
                return self.samplers[0](posterior)
            if len(self.samplers) < posterior._m:
                raise UnsupportedError("Provide at least one fantasy stream per cost output.")
            normals = []
            for i in range(posterior._m):
                component = posterior.subset_output([i])
                sampler = self.samplers[i]
                sampler(component)
                base = getattr(sampler, "base_samples", None)
                if not isinstance(base, Tensor):
                    raise UnsupportedError(
                        "Joint Gaussian cost sampling requires Tensor normal base samples."
                    )
                shape = torch.Size(self.sample_shape)
                tail = base.shape[len(shape) :]
                if len(tail) >= 2 and tail[-2:] == (component._q, 1):
                    base, tail = base.squeeze(-1), tail[:-1]
                missing = len(component.base_sample_shape) - len(tail)
                base = base.reshape(*shape, *((1,) * max(missing, 0)), *tail)
                normals.append(base.expand(shape + component.base_sample_shape))
            return posterior.rsample_from_base_samples(self.sample_shape, torch.stack(normals, -1))
        if hasattr(posterior, "posteriors"):
            return super().forward(posterior)
        return self.samplers[0](posterior)

    __call__ = forward


def tensor_hv_value_function(
    model: Any,
    ref_point: Tensor,
    objective: Any = None,
    sampler: Any = None,
    project: Any = None,
    valfunc_cls: Any = None,
    valfunc_argfac: Any = None,
    use_posterior_mean: bool = False,
    log: bool = False,
) -> Any:
    """The reference's empty-front qEHVI value, using the port's Tensor algebra."""
    from botorch_rs.acquisition.knowledge_gradient import ProjectedAcquisitionFunction
    from botorch_rs.acquisition.multi_objective.logei import qLogExpectedHypervolumeImprovement
    from botorch_rs.acquisition.multi_objective.monte_carlo import qExpectedHypervolumeImprovement

    like = next(model.parameters())
    ref_point = torch.as_tensor(ref_point, dtype=like.dtype, device=like.device)
    cls = qLogExpectedHypervolumeImprovement if log else qExpectedHypervolumeImprovement
    value_sampler: Any = _MeanSampler() if use_posterior_mean else sampler
    value = cls(
        model,
        ref_point,
        _EmptyPartition(ref_point),
        sampler=value_sampler,
        objective=objective,
    )
    # These reserved arguments have no effect in the official value factory.
    return value if project is None else ProjectedAcquisitionFunction(value, project)


class TensorHypervolumeKnowledgeGradient(_TensorAcquisition):
    ref_point: Tensor
    current_value: Tensor | None
    _X_evaluation_mask: Tensor | None
    _X_pending_evaluation_mask: Tensor | None

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        nn.Module.__init__(self)
        bound = inspect.signature(inspect.getattr_static(self._public_class, "__init__")).bind(
            self, *args, **kwargs
        )
        bound.apply_defaults()
        values = bound.arguments
        self.model = values["model"]
        children = getattr(self.model, "models", None)
        if children is None or any(model.num_outputs != 1 for model in children):
            raise UnsupportedError("HVKG requires a ModelListGP of single-output component models.")
        self.num_outputs = self.model.num_outputs
        self._name, self._is_analytic, self._log = (
            self._public_class.__name__,
            False,
            bool(values["log"]),
        )
        self.num_fantasies, self.num_pareto = (
            int(values["num_fantasies"]),
            int(values["num_pareto"]),
        )
        if min(self.num_fantasies, self.num_pareto) < 1:
            raise ValueError("num_fantasies and num_pareto must be positive.")
        self.num_pseudo_points = self.num_fantasies * self.num_pareto
        self.sampler = values["sampler"]
        if self.sampler is None:
            self.sampler = ListSampler(
                *[
                    SobolQMCNormalSampler(
                        (self.num_fantasies,), seed=int(torch.randint(0, 1_000_000, (1,)).item())
                    )
                    for _ in children
                ]
            )
        if not isinstance(self.sampler, ListSampler) or len(self.sampler.samplers) != len(children):
            raise ValueError("HVKG requires one fantasy sampler per component model.")
        if any(tuple(s.sample_shape) != (self.num_fantasies,) for s in self.sampler.samplers):
            raise ValueError("Fantasy sampler shapes must match num_fantasies.")
        self.inner_sampler = values["inner_sampler"]
        if self.inner_sampler is None:
            self.inner_sampler = SobolQMCNormalSampler(
                (32,), seed=int(torch.randint(0, 1_000_000, (1,)).item())
            )
        self.objective, self.use_posterior_mean = (
            values["objective"],
            bool(values["use_posterior_mean"]),
        )
        self.cost_aware_utility = values["cost_aware_utility"]
        if self.cost_aware_utility is not None and values["current_value"] is None:
            raise UnsupportedError("Cost-aware HVKG requires current_value.")
        like = next(self.model.parameters())
        ref = torch.as_tensor(values["ref_point"], dtype=like.dtype, device=like.device)
        if ref.ndim != 1 or ref.numel() < 2:
            raise ValueError("HVKG reference point must have at least two objectives.")
        self.register_buffer("ref_point", ref)
        self.register_buffer(
            "current_value",
            None
            if values["current_value"] is None
            else torch.as_tensor(values["current_value"], dtype=ref.dtype, device=ref.device),
        )
        self.register_buffer("_X_pending", None)
        self.register_buffer("_X_evaluation_mask", None)
        self.register_buffer("_X_pending_evaluation_mask", None)
        self.X_evaluation_mask = values["X_evaluation_mask"]
        self.set_X_pending(values["X_pending"], values["X_pending_evaluation_mask"])
        self._cost_sampler: Any = None
        self.project = values.get("project")
        self.target_fidelities = values.get("target_fidelities")
        self.valfunc_cls, self.valfunc_argfac = (
            values.get("valfunc_cls"),
            values.get("valfunc_argfac"),
        )
        extra = values.get("kwargs", {})
        if extra.get("expand") is not None:
            raise UnsupportedError("HVKG trace observations are not supported, as in BoTorch.")
        if set(extra) - {"expand"}:
            raise UnsupportedError(f"Unsupported HVKG options: {sorted(extra)}.")

    def _mask(self, value: Any, rows: int | None = None) -> Tensor | None:
        if value is None:
            return None
        mask = torch.as_tensor(value, device=self.ref_point.device)
        if (
            mask.dtype != torch.bool
            or mask.ndim != 2
            or mask.shape[-1] != self.num_outputs
            or (rows is not None and mask.shape[0] != rows)
        ):
            raise BotorchTensorDimensionError(
                "Evaluation masks must be boolean q x num_outputs tensors."
            )
        return mask

    @property
    def X_evaluation_mask(self) -> Tensor | None:
        return self._X_evaluation_mask

    @X_evaluation_mask.setter
    def X_evaluation_mask(self, value: Any) -> None:
        self._X_evaluation_mask = self._mask(value)

    @property
    def X_pending_evaluation_mask(self) -> Tensor | None:
        return self._X_pending_evaluation_mask

    def set_X_pending(self, X_pending: Any = None, X_pending_evaluation_mask: Any = None) -> None:
        if X_pending is not None and (not isinstance(X_pending, Tensor) or X_pending.ndim != 2):
            raise ValueError("HVKG pending points must be an n x d Tensor.")
        self._X_pending = None if X_pending is None else X_pending.detach().clone()
        self._X_pending_evaluation_mask = (
            None
            if X_pending is None
            else self._mask(X_pending_evaluation_mask, X_pending.shape[-2])
        )

    def construct_evaluation_mask(self, X: Tensor) -> Tensor | None:
        current = self._mask(self.X_evaluation_mask, X.shape[-2])
        pending = self.X_pending_evaluation_mask
        if current is None and pending is None:
            return None
        if current is None:
            current = torch.ones((X.shape[-2], self.num_outputs), dtype=torch.bool, device=X.device)
        if self.X_pending is None:
            return current
        if pending is None:
            pending = torch.ones(
                (self.X_pending.shape[-2], self.num_outputs), dtype=torch.bool, device=X.device
            )
        return torch.cat((current, pending), -2)

    @property
    def cost_sampler(self) -> Any:
        if self._cost_sampler is None:
            self._cost_sampler = _CostListSampler(
                *[_clone_sampler(s) for s in self.sampler.samplers]
            )
        return self._cost_sampler

    def get_augmented_q_batch_size(self, q: int) -> int:
        return q + self.num_pseudo_points

    def extract_candidates(self, X_full: Tensor) -> Tensor:
        return X_full[..., : -self.num_pseudo_points, :]

    def forward(self, X: Tensor) -> Tensor:
        from botorch_rs.acquisition.multi_objective.hypervolume_knowledge_gradient import (
            _check_log_utilities,
            _split_hvkg_fantasy_points,
        )

        if not isinstance(X, Tensor) or X.ndim < 2 or X.shape[-2] <= self.num_pseudo_points:
            raise ValueError("HVKG requires real candidates followed by its fantasy designs.")
        X = X.unsqueeze(0) if X.ndim == 2 else X
        actual, fantasies = _split_hvkg_fantasy_points(X, self.num_fantasies, self.num_pareto)
        evaluated = actual
        mask = self.construct_evaluation_mask(actual)
        if self.X_pending is not None:
            pending = self.X_pending.expand(*actual.shape[:-2], *self.X_pending.shape)
            evaluated = torch.cat((actual, pending), -2)
        model = self.model.fantasize(
            evaluated, self.sampler, evaluation_mask=mask, propagate_grads=True
        )
        value_function = tensor_hv_value_function(
            model,
            self.ref_point,
            self.objective,
            self.inner_sampler,
            self.project,
            self.valfunc_cls,
            self.valfunc_argfac,
            self.use_posterior_mean,
            self._log,
        )
        values = value_function(fantasies)
        if self.current_value is not None:
            if self._log:
                difference = self.current_value - values
                near = difference.masked_fill(difference <= -math.log(2), -math.log(2))
                far = difference.masked_fill(difference > -math.log(2), -math.log(2))
                values = values + torch.where(
                    difference > -math.log(2), (-near.expm1()).log(), (-far.exp()).log1p()
                )
            else:
                values = values - self.current_value
        if self.cost_aware_utility is not None:
            if self._log:
                _check_log_utilities(self.cost_aware_utility)
            values = self.cost_aware_utility(
                X=actual,
                deltas=values,
                sampler=self.cost_sampler,
                X_evaluation_mask=self.X_evaluation_mask,
            )
        return (
            torch.logsumexp(values, 0) - math.log(self.num_fantasies)
            if self._log
            else values.mean(0)
        )

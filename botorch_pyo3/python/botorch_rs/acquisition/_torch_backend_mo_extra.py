"""Analytic hypervolume, cost-aware multi-fidelity HV and ParEGO on device."""

from __future__ import annotations

import inspect
import math
from typing import Any

import torch
from torch import Tensor, nn

from botorch_rs.acquisition._torch_backend_acquisition import _average_ensemble, _TensorAcquisition
from botorch_rs.acquisition._torch_backend_multi_objective import TensorMultiObjectiveAcquisition
from botorch_rs.utils.transforms import is_ensemble


class TensorAnalyticHypervolume(TensorMultiObjectiveAcquisition):
    def __init__(
        self, model: Any, ref_point: Any, partitioning: Any, posterior_transform: Any = None
    ) -> None:
        nn.Module.__init__(self)
        self._tensor_backend = "torch"
        self.model = model
        self.posterior_transform = posterior_transform
        self.partitioning = partitioning
        self._is_analytic = True
        self._log = False
        self._name = "ExpectedHypervolumeImprovement"
        self.register_buffer("_X_pending", None)
        like = partitioning.pareto_Y
        ref = torch.as_tensor(ref_point, dtype=like.dtype, device=like.device)
        if ref.ndim != 1 or ref.numel() != partitioning.num_outcomes:
            raise ValueError("The reference point must match the number of partitioning outcomes.")
        if like.shape[-2] and not bool((like > ref).all(-1).any()):
            raise ValueError("At least one pareto point must be better than the reference point.")
        self.register_buffer("ref_point", ref)
        self._init_partition(partitioning)
        self.normal = torch.distributions.Normal(0, 1)

    def psi(self, lower: Tensor, upper: Tensor, mu: Tensor, sigma: Tensor) -> Tensor:
        z = (upper - mu) / sigma
        density = (-0.5 * z.square()).exp() / math.sqrt(2 * math.pi)
        return sigma * density + (mu - lower) * (1 - self.normal.cdf(z))

    def nu(self, lower: Tensor, upper: Tensor, mu: Tensor, sigma: Tensor) -> Tensor:
        return (upper - lower) * (1 - self.normal.cdf((upper - mu) / sigma))

    def forward(self, X: Tensor) -> Tensor:
        X = self._candidates_tensor(X)
        posterior = self.model.posterior(X, posterior_transform=self.posterior_transform)
        mean, sigma = posterior.mean, posterior.variance.clamp_min(1e-9).sqrt()
        lower, upper, active = self._cells(mean)
        upper = upper.clamp_max(1e10 if X.dtype == torch.double else 1e8)
        factors = (
            self.psi(lower, lower, mean, sigma)
            - self.psi(lower, upper, mean, sigma)
            + self.nu(lower, upper, mean, sigma)
        )
        value = factors.prod(-1).masked_fill(~active, 0).sum(-1)
        return _average_ensemble(value, self.model)


class TensorMOMF(TensorMultiObjectiveAcquisition):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        nn.Module.__init__(self)
        bound = inspect.signature(inspect.getattr_static(self._public_class, "__init__")).bind(
            self, *args, **kwargs
        )
        bound.apply_defaults()
        values = dict(bound.arguments)
        values["fat"] = False
        self._init_common(values)
        self.set_X_pending(values["X_pending"])
        self._init_partition(values["partitioning"])
        from botorch_rs.acquisition.cost_aware import InverseCostWeightedUtility
        from botorch_rs.models.cost import AffineFidelityCostModel
        from botorch_rs.models.deterministic import GenericDeterministicModel

        self.cost_call = values["cost_call"]
        cost_model = (
            AffineFidelityCostModel(fidelity_weights={-1: 1.0}, fixed_cost=1.0)
            if self.cost_call is None
            else GenericDeterministicModel(self.cost_call)
        )
        self.cost_aware_utility = InverseCostWeightedUtility(cost_model)

    def forward(self, X: Tensor) -> Tensor:
        X = self._candidates_tensor(X)
        samples, objective = self._draw_mo(X)
        gain = self._hypervolume(samples, objective)
        value = self.cost_aware_utility(X, deltas=gain)
        return _average_ensemble(value, self.model)


class TensorParego(_TensorAcquisition):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        bound = inspect.signature(inspect.getattr_static(self._public_class, "__init__")).bind(
            self, *args, **kwargs
        )
        bound.apply_defaults()
        values = dict(bound.arguments)
        values.pop("self")
        model, baseline = values["model"], values["X_baseline"]
        from botorch_rs.acquisition.logei import qLogNoisyExpectedImprovement
        from botorch_rs.acquisition.multi_objective.objective import IdentityMCMultiOutputObjective
        from botorch_rs.acquisition.objective import GenericMCObjective
        from botorch_rs.utils.multi_objective.scalarization import get_chebyshev_scalarization

        objective = values.get("objective") or IdentityMCMultiOutputObjective()
        observed = values.pop("_y_baseline", None)
        if observed is None:
            with torch.no_grad():
                observed = objective(model.posterior(baseline).mean)
            if is_ensemble(model):
                observed = observed.mean(-3)
        weights = values.pop("scalarization_weights")
        if weights is None:
            # Independent exponentials normalized by their sum give Dirichlet(1),
            # i.e. the uniform simplex law used by the reference.
            weights = torch.empty(
                observed.shape[-1], dtype=baseline.dtype, device=baseline.device
            ).exponential_()
            weights = weights / weights.sum()
        scalarization = get_chebyshev_scalarization(weights, observed)
        values["objective"] = GenericMCObjective(
            lambda samples, X=None: scalarization(objective(samples, X=X), X=X)
        )
        # Reuse the actual Tensor noisy-improvement implementation and its
        # constructor protocol; preserve the public ParEGO type in the MRO.
        public_class = self._public_class
        object.__setattr__(self, "_public_class", qLogNoisyExpectedImprovement)
        try:
            _TensorAcquisition.__init__(self, **values)
        finally:
            object.__setattr__(self, "_public_class", public_class)
        self._org_objective = objective
        self.chebyshev_scalarization = scalarization
        self.scalarization_weights = weights
        self.Y_baseline = observed

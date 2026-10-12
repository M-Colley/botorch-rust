"""Shared helpers of the parity tests of fantasies and the hypervolume
knowledge gradient (``test_parity_mo_hvkg.py``,
``test_equivalence_mo_hvkg.py``).

Import only after ``importorskip_botorch()``: this module requires torch and
BoTorch.

Base samples. BoTorch's fantasy samplers (one per output of a
``ListSampler``) and its inner sampler draw their base samples on the first
evaluation; :class:`HvkgPair` evaluates BoTorch first and installs the drawn
base samples in botorch_rs's samplers, so both libraries use identical
fantasies and posterior samples.
"""

from __future__ import annotations

import functools
import inspect
from dataclasses import dataclass, field
from typing import Any

import gpytorch
import torch
from botorch.acquisition import cost_aware as bt_cost_aware
from botorch.acquisition import utils as bt_acq_utils
from botorch.acquisition.multi_objective import hypervolume_knowledge_gradient as bt_hvkg
from botorch.models import cost as bt_cost
from botorch.sampling.list_sampler import ListSampler as BtListSampler
from botorch.sampling.normal import SobolQMCNormalSampler as BtSampler
from mo_acquisition_helpers import DT, objective_pair

from botorch_rs.acquisition import cost_aware as rs_cost_aware
from botorch_rs.acquisition import utils as rs_acq_utils
from botorch_rs.acquisition.multi_objective import hypervolume_knowledge_gradient as rs_hvkg
from botorch_rs.models import cost as rs_cost
from botorch_rs.sampling import ListSampler as RsListSampler
from botorch_rs.sampling import SobolQMCNormalSampler as RsSampler
from parity_helpers import log_err, rel_err

# ---------------------------------------------------------------------------
# BoTorch versions
# ---------------------------------------------------------------------------


def botorch_has_fantasy_mean_cache_fix() -> bool:
    """Whether BoTorch includes meta-pytorch/botorch#3396 (the posterior
    mean of a model conditioned with explicit noise uses that noise)."""
    import botorch.models.gpytorch as bg

    return hasattr(bg, "_use_fantasy_mean_cache")


def botorch_has_model_list_conditioning_fix() -> bool:
    """Whether BoTorch includes meta-pytorch/botorch#3399
    (``ModelListGP.condition_on_observations`` applies the sub-models'
    transforms)."""
    from botorch.models import model_list_gp_regression as ml

    return "conditioned_models" in inspect.getsource(ml.ModelListGP.condition_on_observations)


def use_fantasy_mean_cache(model: Any) -> None:
    """meta-pytorch/botorch#3396 applied to a conditioned BoTorch model (for
    versions without it): the posterior mean uses the updated mean cache of
    GPyTorch's fantasy strategy, i.e. the noise of the new observations."""
    from gpytorch.utils.errors import CachingError
    from gpytorch.utils.memoize import add_to_cache, get_from_cache

    for gp in getattr(model, "models", [model]):
        strategy = gp.prediction_strategy
        try:
            get_from_cache(strategy, "mean_cache", "ignore")
            continue
        except CachingError:
            pass
        mean_cache = get_from_cache(strategy, "mean_cache")
        add_to_cache(strategy, "mean_cache", mean_cache.detach(), "ignore")


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------


def points(X: torch.Tensor, shape: tuple[int, ...], seed: int) -> torch.Tensor:
    """Uniform points in the bounding box of the training inputs ``X``."""
    lo, hi = X.min(0).values, X.max(0).values
    g = torch.Generator().manual_seed(seed)
    return lo + (hi - lo) * torch.rand(*shape, generator=g, dtype=DT)


def mask_rows(kind: str | None, rows: int, m: int) -> torch.Tensor | None:
    """An evaluation mask (``rows x m``): ``alternate`` outputs,
    ``first_only`` or ``same_rows`` (all outputs but the second)."""
    if kind is None:
        return None
    if kind == "alternate":
        data = [[(r + k) % 2 == 0 for k in range(m)] for r in range(rows)]
    elif kind == "first_only":
        data = [[k == 0 for k in range(m)] for _ in range(rows)]
    elif kind == "same_rows":
        data = [[k != 1 for k in range(m)] for _ in range(rows)]
    else:
        raise ValueError(kind)
    return torch.tensor(data, dtype=torch.bool).reshape(rows, m)


def ref_point(bt_model: Any, bt_objective: Any, X: torch.Tensor, seed: int) -> torch.Tensor:
    """A reference point below the objective values of the posterior mean at
    random points."""
    Xr = points(X, (64, X.shape[-1]), seed)
    with torch.no_grad():
        Y = bt_model.posterior(Xr).mean
        if bt_objective is not None:
            Y = bt_objective(Y)
    lo, hi = Y.min(0).values, Y.max(0).values
    return lo - 0.05 * (hi - lo)


def posterior_moments(model: Any, X: Any, noise: bool = False) -> tuple[Any, Any, Any]:
    """The posterior mean, marginal variances and joint covariance
    (output-major) of a model of either library."""
    with torch.no_grad():
        post = model.posterior(X, observation_noise=noise)
        return (
            torch.as_tensor(post.mean),
            torch.as_tensor(post.variance),
            torch.as_tensor(post.distribution.covariance_matrix),
        )


# ---------------------------------------------------------------------------
# Cost-aware utilities
# ---------------------------------------------------------------------------


def cost_pair(kind: str | None, m: int, d: int) -> tuple[Any, Any]:
    """The same ``InverseCostWeightedUtility`` in both libraries: ``fixed``
    (one cost per output), ``fixed_log`` (log utilities) or ``affine`` (in
    the last column; BoTorch's float32 weights are exact here)."""
    if kind is None:
        return None, None
    log = kind == "fixed_log"
    if kind in ("fixed", "fixed_log"):
        costs = torch.tensor([1.0 + 0.5 * k for k in range(m)], dtype=DT)
        cm_bt: Any = bt_cost.FixedCostModel(costs)
        cm_rs: Any = rs_cost.FixedCostModel(costs)
    elif kind == "affine":
        cm_bt = bt_cost.AffineFidelityCostModel(fidelity_weights={d - 1: 1.5}, fixed_cost=0.25)
        cm_rs = rs_cost.AffineFidelityCostModel(fidelity_weights={d - 1: 1.5}, fixed_cost=0.25)
    else:
        raise ValueError(kind)
    return (
        bt_cost_aware.InverseCostWeightedUtility(cost_model=cm_bt, log=log),
        rs_cost_aware.InverseCostWeightedUtility(cost_model=cm_rs, log=log),
    )


# ---------------------------------------------------------------------------
# The hypervolume knowledge gradient in both libraries
# ---------------------------------------------------------------------------


@dataclass
class HvkgCase:
    """A qHVKG / qMFHVKG configuration."""

    name: str
    n_f: int
    num_pareto: int
    q: int
    use_posterior_mean: bool = True
    log: bool = False
    current_value: bool = False
    pending: int = 0
    pending_mask: str | None = None
    mask: str | None = None
    objective: str = "identity"
    cost: str | None = None
    mf: bool = False
    inner: int = 4
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class Evaluation:
    """Values and gradients of one library at a list of candidate sets."""

    values: list[torch.Tensor]
    grads: list[torch.Tensor]


def values_and_grads(acqf: Any, X: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    Xg = X.detach().clone().requires_grad_(True)
    values = acqf(Xg)
    (grad,) = torch.autograd.grad(values.sum(), Xg)
    return values.detach(), grad


class HvkgPair:
    """A qHVKG configuration on a ``ModelListGP`` pair: BoTorch is evaluated
    first (its gradients with ``detach_test_caches(False)``, see below), and
    botorch_rs is built with the base samples BoTorch drew.

    BoTorch detaches the test caches of the fantasy models by default
    (GPyTorch's ``detach_test_caches``): with posterior samples
    (``use_posterior_mean=False``) its gradient misses the dependence of the
    fantasy models' covariances on the candidates; the exact reference
    gradient is BoTorch's with the caches attached."""

    def __init__(
        self, case: HvkgCase, bt_model: Any, rs_model: Any, X: torch.Tensor, seed: int
    ) -> None:
        self.case = case
        self.bt_model, self.rs_model = bt_model, rs_model
        m, d = bt_model.num_outputs, X.shape[-1]
        self.m, self.d = m, d
        c = case
        self.obj_bt, self.obj_rs, _ = objective_pair(c.objective, m, bt_model, rs_model, X)
        self.ref = ref_point(bt_model, self.obj_bt, X, seed)
        self.pending = points(X, (c.pending, d), seed + 1) if c.pending else None
        self.pmask = mask_rows(c.pending_mask, c.pending, m)
        self.xmask = mask_rows(c.mask, c.q, m)
        self.target = {d - 1: 1.0}
        self.bt_samplers = [
            BtSampler(sample_shape=torch.Size([c.n_f]), seed=seed + 7 + k) for k in range(m)
        ]
        self.bt_inner = BtSampler(sample_shape=torch.Size([c.inner]), seed=seed + 3)
        self.cost_bt, self.cost_rs = cost_pair(c.cost, m, d)
        self.current_value: float | None = None
        self.bt_acqf = self._build_bt()
        self.rs_acqf: Any = None

    def candidate_sets(self, X: torch.Tensor, count: int, seed: int) -> list[torch.Tensor]:
        """``count`` augmented candidate sets (``1 x (q + n_f num_pareto) x d``;
        fidelities in [0.2, 1] for multi-fidelity cases)."""
        c = self.case
        g = torch.Generator().manual_seed(seed)
        Xs = [
            points(X, (1, c.q + c.n_f * c.num_pareto, self.d), seed + 11 + k) for k in range(count)
        ]
        if c.mf:
            for x in Xs:
                x[..., -1] = 0.2 + 0.8 * torch.rand(x.shape[:-1], generator=g, dtype=DT)
        return Xs

    def _kwargs(self, rs: bool) -> dict[str, Any]:
        c = self.case
        kw: dict[str, Any] = dict(
            ref_point=self.ref,
            num_fantasies=c.n_f,
            num_pareto=c.num_pareto,
            objective=self.obj_rs if rs else self.obj_bt,
            X_pending=self.pending,
            X_evaluation_mask=self.xmask,
            X_pending_evaluation_mask=self.pmask,
            use_posterior_mean=c.use_posterior_mean,
            log=c.log,
            **c.extra,
        )
        if c.mf:
            project = (
                rs_acq_utils.project_to_target_fidelity
                if rs
                else bt_acq_utils.project_to_target_fidelity
            )
            kw["target_fidelities"] = self.target
            kw["project"] = functools.partial(project, target_fidelities=self.target, d=self.d)
        if self.current_value is not None:
            kw["current_value"] = torch.tensor(self.current_value, dtype=DT)
            kw["cost_aware_utility"] = self.cost_rs if rs else self.cost_bt
        return kw

    def _build_bt(self) -> Any:
        cls = (
            bt_hvkg.qMultiFidelityHypervolumeKnowledgeGradient
            if self.case.mf
            else bt_hvkg.qHypervolumeKnowledgeGradient
        )
        return cls(
            model=self.bt_model,
            sampler=BtListSampler(*self.bt_samplers),
            inner_sampler=self.bt_inner,
            **self._kwargs(rs=False),
        )

    def set_current_value(self, Xs: list[torch.Tensor]) -> None:
        """BoTorch's ``current_value`` (and the cost-aware utility): linear,
        half of the first value (some improvements are negative); log, far
        below every value (every fantasy must exceed it)."""
        with torch.no_grad():
            v = [self.bt_acqf(x).item() for x in Xs]
        self.current_value = min(v) - 6.907755278982137 if self.case.log else 0.5 * v[0]
        self.bt_acqf = self._build_bt()

    def evaluate_bt(self, Xs: list[torch.Tensor]) -> Evaluation:
        values, grads = [], []
        for x in Xs:
            with gpytorch.settings.detach_test_caches(False):
                v, g = values_and_grads(self.bt_acqf, x)
            values.append(v)
            grads.append(g)
        return Evaluation(values, grads)

    def build_rs(self) -> Any:
        """botorch_rs's acquisition function with BoTorch's base samples."""
        c = self.case
        samplers = []
        for s in self.bt_samplers:
            r = RsSampler(sample_shape=(c.n_f,), seed=0)
            if s.base_samples is not None:
                r.base_samples = s.base_samples.clone()
            samplers.append(r)
        inner = RsSampler(sample_shape=(c.inner,), seed=0)
        if self.bt_inner.base_samples is not None:
            inner.base_samples = self.bt_inner.base_samples.clone()
        cls = (
            rs_hvkg.qMultiFidelityHypervolumeKnowledgeGradient
            if c.mf
            else rs_hvkg.qHypervolumeKnowledgeGradient
        )
        self.rs_acqf = cls(
            model=self.rs_model,
            sampler=RsListSampler(*samplers),
            inner_sampler=inner,
            **self._kwargs(rs=True),
        )
        return self.rs_acqf

    def evaluate_rs(self, Xs: list[torch.Tensor]) -> Evaluation:
        values, grads = [], []
        for x in Xs:
            v, g = values_and_grads(self.rs_acqf, x)
            values.append(v)
            grads.append(g)
        return Evaluation(values, grads)

    def errors(self, rs: Evaluation, bt: Evaluation) -> tuple[float, float]:
        """The worst value error (log values: absolute below 1) and gradient
        error over the candidate sets."""
        err_v = max(
            (log_err(a, e) if self.case.log else rel_err(a, e, atol=1e-300))
            for a, e in zip(rs.values, bt.values, strict=True)
        )
        err_g = max(rel_err(a, e, atol=1e-14) for a, e in zip(rs.grads, bt.grads, strict=True))
        return err_v, err_g


def run_hvkg_case(
    case: HvkgCase, bt_model: Any, rs_model: Any, X: torch.Tensor, seed: int, count: int = 2
) -> tuple[HvkgPair, list[torch.Tensor], Evaluation, Evaluation]:
    """Builds both acquisition functions and evaluates them at ``count``
    candidate sets (BoTorch first)."""
    pair = HvkgPair(case, bt_model, rs_model, X, seed)
    Xs = pair.candidate_sets(X, count, seed)
    if case.current_value:
        pair.set_current_value(Xs)
    bt = pair.evaluate_bt(Xs)
    pair.build_rs()
    rs = pair.evaluate_rs(Xs)
    return pair, Xs, rs, bt

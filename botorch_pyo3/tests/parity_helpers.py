"""Shared helpers of the live comparisons with BoTorch
(``test_parity_constraints.py`` and the unified ``test_equivalence.py``).

Import only after ``importorskip_botorch()``: this module requires torch,
GPyTorch and BoTorch.

Error measures: values and gradients are compared with a mixed elementwise /
normwise relative error (``|a - e| / (|e| + max|e|)``), log-space values with
an error that is absolute below 1 and relative above.
"""

from __future__ import annotations

import contextlib
import random
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import Any

import gpytorch
import torch
from botorch.models import ModelListGP as BtModelListGP
from botorch.models import SingleTaskGP as BtSingleTaskGP
from botorch.models.transforms.input import Normalize as BtNormalize
from botorch.models.utils import gpytorch_modules as bt_modules
from botorch.sampling.normal import SobolQMCNormalSampler as BtSampler
from gpytorch.mlls import ExactMarginalLogLikelihood as BtMLL

from botorch_rs.models import ModelListGP as RsModelListGP
from botorch_rs.models import SingleTaskGP as RsSingleTaskGP
from botorch_rs.models.transforms import Normalize as RsNormalize
from botorch_rs.models.utils import gpytorch_modules as rs_modules
from botorch_rs.sampling import SobolQMCNormalSampler as RsSampler
from helpers import ParityLog

DT = torch.float64


# ---------------------------------------------------------------------------
# Error measures
# ---------------------------------------------------------------------------


def as_t(x: Any) -> Any:
    return torch.as_tensor(x, dtype=DT).detach()


def rel_err(a: Any, e: Any, atol: float = 0.0) -> float:
    """Mixed elementwise / normwise relative error; entries that agree to
    ``atol`` count as exact."""
    a, e = as_t(a), as_t(e)
    assert a.shape == e.shape, (a.shape, e.shape)
    if e.numel() == 0:
        return 0.0
    if not torch.isfinite(e).all() or not torch.isfinite(a).all():
        same = (a == e) | (torch.isnan(a) & torch.isnan(e))
        finite = torch.isfinite(e) & torch.isfinite(a)
        if not bool((same | finite).all()):
            return float("inf")
        a, e = torch.where(finite, a, 0.0), torch.where(finite, e, 0.0)
    diff = (a - e).abs()
    diff = torch.where(diff <= atol, torch.zeros_like(diff), diff)
    scale = e.abs().max().item()
    floor = scale if scale > 0 else 1.0
    return (diff / (e.abs() + floor)).max().item()


def log_err(a: Any, e: Any) -> float:
    """Error of log-space values: absolute below 1, relative above
    (``-inf`` must match exactly)."""
    a, e = as_t(a), as_t(e)
    assert a.shape == e.shape, (a.shape, e.shape)
    inf = torch.isinf(e) | torch.isinf(a)
    if bool(inf.any()):
        if not bool((a[inf] == e[inf]).all()):
            return float("inf")
        a, e = a[~inf], e[~inf]
        if e.numel() == 0:
            return 0.0
    return ((a - e).abs() / e.abs().clamp_min(1.0)).max().item()


class Checker:
    """Compares quantities and records the worst error of each under a
    section of the terminal summary."""

    def __init__(self, section: str) -> None:
        self.section = section

    def record(self, quantity: str, err: float, tol: float) -> None:
        ParityLog.record(quantity, err, section=self.section, tol=tol)

    def check(self, quantity: str, a: Any, e: Any, tol: float, atol: float = 0.0) -> float:
        err = rel_err(a, e, atol)
        self.record(quantity, err, tol)
        assert err <= tol, (quantity, err, tol)
        return err

    def check_log(self, quantity: str, a: Any, e: Any, tol: float) -> float:
        err = log_err(a, e)
        self.record(quantity, err, tol)
        assert err <= tol, (quantity, err, tol)
        return err

    def check_bound(self, quantity: str, excess: float, tol: float) -> None:
        """Records a one-sided quantity (e.g. how far botorch_rs falls short
        of BoTorch's optimum)."""
        excess = max(float(excess), 0.0)
        self.record(quantity, excess, tol)
        assert excess <= tol, (quantity, excess, tol)


def values_and_grads(acqf: Any, X: Any) -> tuple[Any, Any]:
    """Acquisition values at ``X`` and the gradient of their sum."""
    Xg = X.detach().clone().requires_grad_(True)
    values = acqf(Xg)
    (grad,) = torch.autograd.grad(values.sum(), Xg)
    return values.detach(), grad


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ModelSpec:
    """A SingleTaskGP configuration."""

    n: int
    d: int
    m: int = 1
    fixed_noise: bool = False
    normalize: bool = False
    standardize: bool = True
    kernel: str = "rbf"  # "rbf" or "matern52"
    batch: tuple[int, ...] = ()
    lo: float = 0.0
    hi: float = 1.0

    @property
    def label(self) -> str:
        parts = [f"n{self.n}", f"d{self.d}", f"m{self.m}", self.kernel]
        if self.batch:
            parts.append("b" + "x".join(map(str, self.batch)))
        if self.fixed_noise:
            parts.append("fixed")
        if self.normalize:
            parts.append("norm")
        if not self.standardize:
            parts.append("nostd")
        return "_".join(parts)


def make_data(spec: ModelSpec, seed: int) -> tuple[Any, Any, Any]:
    """Training inputs in ``[lo, hi]^d``, smooth outputs plus noise, and
    observation noise variances."""
    g = torch.Generator().manual_seed(seed)
    X = spec.lo + (spec.hi - spec.lo) * torch.rand(
        *spec.batch, spec.n, spec.d, generator=g, dtype=DT
    )
    U = (X - spec.lo) / (spec.hi - spec.lo)
    cols = []
    for t in range(spec.m):
        f = torch.sin((3.0 + t) * U[..., 0]) + (1.0 + 0.5 * t) * U[..., 1:].sum(-1) ** 2
        f = f + 0.05 * torch.randn(*spec.batch, spec.n, generator=g, dtype=DT)
        cols.append((1.0 + t) + (0.5 + t) * f)
    Y = torch.stack(cols, dim=-1)
    Yvar = 0.01 + 0.02 * torch.rand(*spec.batch, spec.n, spec.m, generator=g, dtype=DT)
    return X, Y, Yvar


def build_pair(spec: ModelSpec, X: Any, Y: Any, Yvar: Any) -> tuple[Any, Any]:
    """The same SingleTaskGP in BoTorch and botorch_rs (default
    hyperparameters)."""
    aug = torch.Size(spec.batch + ((spec.m,) if spec.m > 1 else ()))
    bt_kw: dict[str, Any] = {}
    rs_kw: dict[str, Any] = {}
    if spec.normalize:
        bt_kw["input_transform"] = BtNormalize(d=spec.d)
        rs_kw["input_transform"] = RsNormalize(d=spec.d)
    if not spec.standardize:
        bt_kw["outcome_transform"] = None
        rs_kw["outcome_transform"] = None
    if spec.kernel == "matern52":
        bt_kw["covar_module"] = bt_modules.get_covar_module_with_dim_scaled_prior(
            ard_num_dims=spec.d, use_rbf_kernel=False, batch_shape=aug
        )
        rs_kw["covar_module"] = rs_modules.get_covar_module_with_dim_scaled_prior(
            ard_num_dims=spec.d, use_rbf_kernel=False, batch_shape=aug
        )
    yvar = Yvar if spec.fixed_noise else None
    bt = BtSingleTaskGP(X, Y, train_Yvar=yvar, **bt_kw)
    rs = RsSingleTaskGP(X, Y, train_Yvar=yvar, **rs_kw)
    return bt, rs


def set_hyperparameters(bt: Any, seed: int) -> None:
    """Random explicit hyperparameters (different for every element of the
    augmented batch shape)."""
    r = random.Random(seed)
    aug = bt._aug_batch_shape
    k = max(aug.numel(), 1)
    d = bt.train_inputs[0].shape[-1]
    scale = 1.0
    if not hasattr(bt, "input_transform"):
        X = bt.train_inputs[0]
        scale = max((X.max() - X.min()).item(), 1e-3)
    ls = torch.tensor(
        [[r.uniform(0.25, 1.2) * scale for _ in range(d)] for _ in range(k)], dtype=DT
    ).reshape(*aug, 1, d)
    const = torch.tensor([r.uniform(-0.4, 0.4) for _ in range(k)], dtype=DT).reshape(aug)
    if not hasattr(bt, "outcome_transform"):
        const = const + bt.train_targets.mean().item()
    kwargs: dict[str, Any] = {"mean_module.constant": const}
    if isinstance(bt.covar_module, gpytorch.kernels.ScaleKernel):
        kwargs["covar_module.outputscale"] = torch.tensor(
            [r.uniform(0.5, 2.0) for _ in range(k)], dtype=DT
        ).reshape(aug)
        kwargs["covar_module.base_kernel.lengthscale"] = ls
    else:
        kwargs["covar_module.lengthscale"] = ls
    if not isinstance(bt.likelihood, gpytorch.likelihoods.FixedNoiseGaussianLikelihood):
        kwargs["likelihood.noise"] = torch.tensor(
            [r.uniform(0.01, 0.15) for _ in range(k)], dtype=DT
        ).reshape(*aug, 1)
    bt.initialize(**kwargs)


def copy_hyperparameters(bt: Any, rs: Any) -> None:
    rs.initialize(**{name: value.detach() for name, value in bt.named_parameters()})


def model_pair(spec: ModelSpec, seed: int) -> tuple[Any, Any, Any]:
    """A SingleTaskGP pair with identical explicit hyperparameters (eval
    mode) and its training inputs."""
    X, Y, Yvar = make_data(spec, seed)
    bt, rs = build_pair(spec, X, Y, Yvar)
    set_hyperparameters(bt, seed)
    copy_hyperparameters(bt, rs)
    bt.eval()
    return bt, rs, X


def list_pair(specs: Sequence[ModelSpec], seed: int) -> tuple[Any, Any, Any]:
    """A ModelListGP pair of the given sub-models (same input dimension) and
    the training inputs of the first sub-model."""
    bts, rss, X0 = [], [], None
    for i, spec in enumerate(specs):
        bt, rs, X = model_pair(spec, seed + 13 * i)
        bts.append(bt)
        rss.append(rs)
        X0 = X if X0 is None else X0
    lb = BtModelListGP(*bts)
    lb.eval()
    return lb, RsModelListGP(*rss), X0


def perturbed_copy(model: Any, seed: int, rel: float = 1e-14) -> Any:
    """A copy of a BoTorch model whose (transformed) training inputs are
    perturbed by ``rel`` (relative): BoTorch's own sensitivity to rounding.
    Monte-Carlo quantities on nearly singular joint posteriors (e.g. qLogNEI
    over the training points) change by much more than ``rel`` under such a
    perturbation; this "noise floor" bounds the agreement any two
    implementations can reach."""
    import copy

    from gpytorch.models import ExactGP

    # Caches that BoTorch builds while differentiating hold tensors with
    # autograd history (prediction strategies; e.g. the `_perturbations` buffer
    # of an `InputPerturbation` with a callable perturbation set), which torch
    # cannot deep-copy. The twin recomputes them: copy prediction strategies as
    # None and such tensors detached, without touching the original model.
    memo: dict[int, Any] = {}
    for m in model.modules():
        if isinstance(m, ExactGP) and m.prediction_strategy is not None:
            memo[id(m.prediction_strategy)] = None
        for v in list(vars(m).get("_buffers", {}).values()) + list(vars(m).values()):
            if torch.is_tensor(v) and not v.is_leaf:
                memo[id(v)] = v.detach().clone()
    twin = copy.deepcopy(model, memo)
    g = torch.Generator().manual_seed(seed)

    def perturb(gp: Any) -> None:
        def noisy(X: Any) -> Any:
            return X * (1 + rel * torch.randn(X.shape, generator=g, dtype=X.dtype))

        gp.train_inputs = tuple(noisy(X) for X in gp.train_inputs)
        if getattr(gp, "_original_train_inputs", None) is not None:
            gp._original_train_inputs = noisy(gp._original_train_inputs)
        gp.prediction_strategy = None

    for gp in getattr(twin, "models", [twin]):
        perturb(gp)
    return twin


def bt_mll_values(model: Any) -> Any:
    mll = BtMLL(model.likelihood, model)
    model.train()
    with torch.no_grad():
        values = mll(model(*model.train_inputs), model.train_targets)
    model.eval()
    return values


def bt_mll_value_and_grad(model: Any) -> tuple[Any, list[float]]:
    mll = BtMLL(model.likelihood, model)
    model.train()
    model.zero_grad()
    values = mll(model(*model.train_inputs), model.train_targets)
    values.sum().backward()
    grad = [g for _, p in model.named_parameters() for g in p.grad.reshape(-1).tolist()]
    model.zero_grad()
    model.eval()
    return values.detach(), grad


@contextlib.contextmanager
def dense_model_list() -> Iterator[None]:
    """BoTorch's ModelListGP posterior and scalarization with dense
    covariances: two code paths fail with linear_operator 0.6.1 for
    ``ModelListGP`` models (``BlockDiagLinearOperator`` slicing when a
    sub-model has several outputs, and block indexing in
    ``scalarize_posterior`` for ``q > 1``)."""
    import botorch.acquisition.objective as objective
    import botorch.models.gpytorch as gpytorch_models
    from botorch.posteriors.gpytorch import GPyTorchPosterior
    from gpytorch.distributions import MultitaskMultivariateNormal
    from linear_operator.operators import DenseLinearOperator

    orig_sep = gpytorch_models.separate_mtmvn
    orig_scal = objective.scalarize_posterior

    # Dense covariances wrapped in a linear operator: a plain tensor would make
    # torch's MultivariateNormal factorize it eagerly (without jitter).
    def separate(mvn: Any) -> Any:
        dense = MultitaskMultivariateNormal(
            mvn.mean,
            DenseLinearOperator(mvn.lazy_covariance_matrix.to_dense()),
            interleaved=mvn._interleaved,
        )
        return orig_sep(dense)

    def scalarize(posterior: Any, weights: Any, offset: float = 0.0) -> Any:
        mvn = posterior.distribution
        if isinstance(mvn, MultitaskMultivariateNormal) and not mvn._interleaved:
            posterior = GPyTorchPosterior(
                MultitaskMultivariateNormal(
                    mvn.mean, DenseLinearOperator(mvn.covariance_matrix), interleaved=False
                )
            )
        return orig_scal(posterior=posterior, weights=weights, offset=offset)

    gpytorch_models.separate_mtmvn = separate
    objective.scalarize_posterior = scalarize
    try:
        yield
    finally:
        gpytorch_models.separate_mtmvn = orig_sep
        objective.scalarize_posterior = orig_scal


# ---------------------------------------------------------------------------
# Monte-Carlo inputs
# ---------------------------------------------------------------------------


def sampler_pair(S: int, q: int, m: int, seed: int) -> tuple[Any, Any, Any]:
    """BoTorch and botorch_rs samplers with the same explicit base samples
    ``Z`` for a posterior over ``q`` points and ``m`` outputs (BoTorch's
    shapes: ``S x 1 x q`` for a single output, ``S x 1 x q x m`` otherwise)."""
    event = (q,) if m == 1 else (q, m)
    Z = torch.randn(S, 1, *event, generator=torch.Generator().manual_seed(seed), dtype=DT)
    s_bt = BtSampler(sample_shape=torch.Size([S]), seed=0)
    s_bt.base_samples = Z.clone()
    s_rs = RsSampler(sample_shape=torch.Size([S]), seed=0)
    s_rs.base_samples = Z.clone()
    return s_bt, s_rs, Z


@contextlib.contextmanager
def patched_lower_bound(points: Any) -> Iterator[None]:
    """Makes BoTorch's ``_estimate_objective_lower_bound`` (used by qLogNEI
    when some sample has no feasible baseline point) evaluate the objective's
    lower bound at ``points`` instead of ``torch.rand`` draws; botorch_rs gets
    the same points through ``lower_bound_points``."""
    import botorch.acquisition.utils as au

    orig = au._estimate_objective_lower_bound

    def patched(model: Any, objective: Any, posterior_transform: Any, X: Any) -> Any:
        return -au.get_infeasible_cost(
            X=points, model=model, objective=objective, posterior_transform=posterior_transform
        )

    au._estimate_objective_lower_bound = patched
    try:
        yield
    finally:
        au._estimate_objective_lower_bound = orig


def points_in(X: Any, shape: Sequence[int], seed: int, pad: float = 0.1) -> Any:
    """Random points of the given shape in the (padded) bounding box of ``X``."""
    lo, hi = X.min().item(), X.max().item()
    span = max(hi - lo, 1e-3)
    g = torch.Generator().manual_seed(seed)
    return lo - pad * span + (1 + 2 * pad) * span * torch.rand(*shape, generator=g, dtype=DT)


# ---------------------------------------------------------------------------
# BoTorch version differences
# ---------------------------------------------------------------------------


def botorch_allows_infeasible_starts() -> bool:
    """Whether BoTorch's SLSQP accepts initial conditions that violate the
    nonlinear constraints (BoTorch's main branch; the released 0.18.1 raises
    "`batch_initial_conditions` must satisfy the non-linear inequality
    constraints.", while botorch_rs follows main)."""
    import inspect

    from botorch.optim import parameter_constraints as pc

    params = inspect.signature(pc.make_scipy_nonlinear_inequality_constraints).parameters
    return "validate_feasibility" in params


def feasible_starts(ics: Any, nonlinear: Any, fallback: Any) -> Any:
    """``ics`` (``b x q x d``) for both libraries: unchanged if BoTorch
    accepts infeasible starts, else only the starts satisfying the nonlinear
    constraints (``fallback``, a feasible ``q x d`` point, if none does)."""
    if not nonlinear or botorch_allows_infeasible_starts():
        return ics
    from botorch.optim import parameter_constraints as pc

    keep = pc.evaluate_feasibility(ics, nonlinear_inequality_constraints=nonlinear)
    kept = ics[keep]
    return kept if kept.shape[0] > 0 else fallback.unsqueeze(0)


def feasible_start_generator(base: Any, nonlinear: Any, fallback: Any) -> Any:
    """An ``ic_generator`` wrapping ``base`` (a library's
    ``gen_batch_initial_conditions``) that, for BoTorch versions requiring
    feasible starts, keeps the starts that satisfy the nonlinear constraints
    and pads with ``fallback``. Used for both libraries alike."""
    if botorch_allows_infeasible_starts():
        return base

    def generate(
        acq_function: Any, bounds: Any, q: int, num_restarts: int, raw_samples: int, **kw: Any
    ) -> Any:
        ics = base(acq_function, bounds, q, 4 * num_restarts, raw_samples, **kw)
        ics = feasible_starts(ics, nonlinear, fallback)[:num_restarts]
        if ics.shape[0] < num_restarts:
            pad = fallback.unsqueeze(0).expand(num_restarts - ics.shape[0], *fallback.shape)
            ics = torch.cat([ics, pad.to(ics)])
        return ics

    return generate

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

    orig_sep = gpytorch_models.separate_mtmvn
    orig_scal = objective.scalarize_posterior

    def separate(mvn: Any) -> Any:
        dense = MultitaskMultivariateNormal(
            mvn.mean, mvn.lazy_covariance_matrix.to_dense(), interleaved=mvn._interleaved
        )
        return orig_sep(dense)

    def scalarize(posterior: Any, weights: Any, offset: float = 0.0) -> Any:
        mvn = posterior.distribution
        if isinstance(mvn, MultitaskMultivariateNormal) and not mvn._interleaved:
            posterior = GPyTorchPosterior(
                MultitaskMultivariateNormal(mvn.mean, mvn.covariance_matrix, interleaved=False)
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

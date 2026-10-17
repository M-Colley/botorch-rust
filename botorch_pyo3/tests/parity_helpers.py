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
import math
import random
from collections.abc import Callable, Iterator, Sequence
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


# ---------------------------------------------------------------------------
# Comparison of fits on twins (two stages)
# ---------------------------------------------------------------------------

#: Fit losses within this distance count as equal: the tolerance of every
#: fit comparison and the ties of the statistical checks.
FIT_TOL = 1e-6
#: BoTorch's outcomes within this distance of its best loss belong to its
#: best optimum; their spread scales the stage-1 tolerance.
FIT_CLUSTER = 1e-4
#: Twins (besides the model itself) fitted in stage 2.
FIT_TWINS = 64
#: One-sided significance level of each statistical check of stage 2.
FIT_ALPHA = 1e-3
#: The standard normal quantile ``Φ⁻¹(1 - FIT_ALPHA)``.
FIT_Z = 3.090232306167813

#: Relative perturbations of the training inputs of twins 1 to 4 (repeated
#: from twin 5 on): BoTorch's own sensitivity to rounding.
TWIN_SCALES = (1e-14, 1e-13, 1e-12, 1e-11)

#: The outcome of a fit: the final loss (or the final losses of the model's
#: elements, e.g. outputs or ensemble members) and the number of attempts.
FitOutcome = tuple[float | Sequence[float], int]


def mann_whitney_z(a: Sequence[float], b: Sequence[float], tie: float) -> float:
    """The z statistic of the one-sided Mann–Whitney U test that the values
    ``a`` are stochastically larger than ``b``: ``U`` counts the pairs with
    ``a_i > b_j + tie`` (pairs within ``tie`` count one half), standardized
    with the no-ties variance and a continuity correction. Ties and positive
    dependence within twin pairs only reduce the variance of ``U``, so the
    test is conservative."""
    n1, n2 = len(a), len(b)
    u = sum(1.0 if x > y + tie else (0.5 if abs(x - y) <= tie else 0.0) for x in a for y in b)
    sd = math.sqrt(n1 * n2 * (n1 + n2 + 1) / 12.0)
    return (u - n1 * n2 / 2.0 - 0.5) / sd


def rank_bound_k(n_rs: int, n_bt: int, alpha: float = FIT_ALPHA) -> int:
    """The smallest ``k`` such that all ``k`` smallest of ``n_rs + n_bt``
    exchangeable values come from the ``n_bt`` with probability at most
    ``alpha``: ``C(n_bt, k) / C(n_rs + n_bt, k) <= alpha``. Then botorch_rs's
    best of ``n_rs`` outcomes exceeds BoTorch's ``k``-th smallest with
    probability at most ``alpha`` (``n_bt + 1`` if there is no such ``k``)."""
    p = 1.0
    for k in range(1, n_bt + 1):
        p *= (n_bt - k + 1) / (n_rs + n_bt - k + 1)
        if p <= alpha:
            return k
    return n_bt + 1


@dataclass
class FitComparison:
    """The two-stage comparison of the fits of one model element (see
    :func:`two_stage_fit_comparisons`)."""

    label: str
    #: BoTorch's and botorch_rs's first-attempt losses of the deciding stage.
    bt: list[float]
    rs: list[float]
    #: botorch_rs's final loss on the model itself minus BoTorch's.
    gap: float
    #: The numbers of fits (of all twins fitted) that needed retries, in
    #: BoTorch and in botorch_rs.
    retries: tuple[int, int]
    #: Stage 1: BoTorch's spread at its best optimum, the tolerance
    #: ``max(FIT_TOL, 10 x spread)``, and the excesses (in units of the
    #: tolerance) of botorch_rs's best loss over BoTorch's best and of its
    #: own fit (on the model itself, if its first attempt succeeded) over
    #: BoTorch's worst.
    spread: float
    tol: float
    best_excess: float
    worst_excess: float
    #: Stage 2 (``None`` if stage 1 decided): the Mann–Whitney z of
    #: botorch_rs's losses, the rank ``k`` and BoTorch's ``k``-th best loss.
    z: float | None = None
    k: int | None = None
    kth: float | None = None

    @property
    def stage(self) -> int:
        return 1 if self.z is None else 2

    @property
    def passed(self) -> bool:
        if self.z is None or self.kth is None:
            return self.best_excess <= 1.0 and self.worst_excess <= 1.0
        return self.z <= FIT_Z and min(self.rs, default=math.inf) - self.kth <= FIT_TOL

    def diagnostics(self) -> str:
        """The deciding values, and for stage 2 the sorted outcomes."""
        lines = [
            f"{self.label}: stage 1: best above BoTorch's best {self.best_excess:.3g} x tol, "
            f"own fit above BoTorch's worst {self.worst_excess:.3g} x tol (tol {self.tol:.1e}, "
            f"BoTorch's spread at its best {self.spread:.1e})"
        ]
        if self.z is not None and self.kth is not None:
            best = min(self.rs, default=math.inf)
            lines += [
                f"stage 2 on {len(self.bt)} BoTorch and {len(self.rs)} botorch_rs first-attempt "
                f"outcomes: Mann-Whitney z = {self.z:.3f} (limit {FIT_Z:.3f}), k = {self.k}, "
                f"BoTorch's k-th best {self.kth:+.10f}, botorch_rs's best {best:+.10f} "
                f"(tolerance {FIT_TOL:g}): {'pass' if self.passed else 'FAIL'}",
                "BoTorch (sorted): " + " ".join(f"{v:+.10f}" for v in sorted(self.bt)),
                "botorch_rs (sorted): " + " ".join(f"{v:+.10f}" for v in sorted(self.rs)),
            ]
        return "\n".join(lines)


def twin_inputs(X: Any, t: int) -> Any:
    """The training inputs ``X`` of twin ``t`` (``t = 0``: ``X`` itself),
    perturbed at ``TWIN_SCALES[(t - 1) % 4]`` (relative), with the seed
    ``t``."""
    if t == 0:
        return X
    g = torch.Generator().manual_seed(t)
    rel = TWIN_SCALES[(t - 1) % len(TWIN_SCALES)]
    return X * (1 + rel * torch.randn(X.shape, generator=g, dtype=X.dtype))


def _losses(outcome: FitOutcome) -> tuple[list[float], int]:
    loss, attempts = outcome
    losses = [float(loss)] if isinstance(loss, (int, float)) else [float(v) for v in loss]
    return losses, int(attempts)


def two_stage_fit_comparisons(
    label: str,
    fit_twin: Callable[[int], tuple[FitOutcome, FitOutcome]],
    twins: int,
    max_twins: int = FIT_TWINS,
    bt_extra: Sequence[float] = (),
) -> list[FitComparison]:
    """Fits from identical initial values in both libraries, compared in two
    stages per model element. ``fit_twin(t)`` returns BoTorch's and
    botorch_rs's outcomes on twin ``t``: the model itself for ``t = 0``, else
    a copy whose training inputs are perturbed at rounding level (BoTorch's
    own outcome depends on rounding). Only first attempts are compared
    (retries start from prior samples, which the libraries draw from
    different random number generators); ``bt_extra`` holds further
    first-attempt outcomes of BoTorch on the model (single-element models,
    e.g. with L-BFGS-B's variables reversed), used in both stages.

    1. On the model and twins ``1..twins``: botorch_rs's best loss must reach
       BoTorch's best, and its own fit (on the model itself, if its first
       attempt succeeded) must not end above BoTorch's worst, both within
       ``max(FIT_TOL, 10 x`` BoTorch's spread at its best optimum``)``. If
       so, the element passes. (The Rust fixture tests, where stage 2 costs
       only botorch_rs's fits, check every stage-1 fit against BoTorch's
       worst; here stage 2 needs up to 64 more BoTorch fits.)
    2. Otherwise (if ``max_twins > twins``) twins up to ``max_twins`` are
       fitted as well, and on all outcomes botorch_rs's losses must pass a
       one-sided Mann–Whitney U test at level ``FIT_ALPHA`` (not
       stochastically larger than BoTorch's; ties within ``FIT_TOL``), and
       its best loss must reach BoTorch's ``k``-th best (:func:`rank_bound_k`)
       within ``FIT_TOL``.

    The procedure fails only if both stages fail; stage 2 alone fails by
    chance (botorch_rs's outcomes another sample of BoTorch's) with
    probability at most ``2 x FIT_ALPHA``, so the procedure does as well (see
    ``docs/fitting.md``, "Comparing fits"). Elements whose BoTorch fits all
    needed retries are not compared."""
    bt_out: list[tuple[list[float], int]] = []
    rs_out: list[tuple[list[float], int]] = []

    def run(ts: range) -> None:
        for t in ts:
            b, r = fit_twin(t)
            bt_out.append(_losses(b))
            rs_out.append(_losses(r))

    def first(out: list[tuple[list[float], int]], e: int) -> list[float]:
        return [losses[e] for losses, a in out if a == 1]

    run(range(twins + 1))
    n_el = len(bt_out[0][0])
    assert not bt_extra or n_el == 1, "bt_extra needs a single-element model"
    extra = [float(v) for v in bt_extra]
    comparisons: dict[int, FitComparison] = {}
    for e in range(n_el):
        bt = first(bt_out, e) + extra
        if not bt:
            continue
        rs = first(rs_out, e)
        best, worst = min(bt), max(bt)
        spread = max(v for v in bt if v <= best + FIT_CLUSTER) - best
        tol = max(FIT_TOL, 10 * spread)
        own = rs_out[0][0][e] if rs_out[0][1] == 1 else -math.inf
        comparisons[e] = FitComparison(
            label=label,
            bt=bt,
            rs=rs,
            gap=rs_out[0][0][e] - bt_out[0][0][e],
            retries=(0, 0),
            spread=spread,
            tol=tol,
            best_excess=max(min(rs, default=math.inf) - best, 0.0) / tol,
            worst_excess=max(own - worst, 0.0) / tol,
        )
    failed = [e for e, c in comparisons.items() if not c.passed]
    if failed and max_twins > twins:
        run(range(twins + 1, max_twins + 1))
        for e in failed:
            c = comparisons[e]
            c.bt, c.rs = first(bt_out, e) + extra, first(rs_out, e)
            c.z = mann_whitney_z(c.rs, c.bt, FIT_TOL) if c.rs else math.inf
            c.k = rank_bound_k(len(c.rs), len(c.bt))
            c.kth = sorted(c.bt)[c.k - 1] if c.k <= len(c.bt) else math.inf
    retries = (sum(a > 1 for _, a in bt_out), sum(a > 1 for _, a in rs_out))
    for c in comparisons.values():
        c.retries = retries
    return list(comparisons.values())


def compare_twin_fits(
    check: Checker,
    label: str,
    fit_twin: Callable[[int], tuple[FitOutcome, FitOutcome]],
    twins: int,
    max_twins: int = FIT_TWINS,
    bt_extra: Sequence[float] = (),
) -> list[FitComparison]:
    """:func:`two_stage_fit_comparisons`, with the deciding quantities
    recorded in ``check`` and asserted. A comparison decided by stage 2
    prints its diagnostics (stage 1's excesses, z, k, BoTorch's ``k``-th
    best and the sorted outcomes of both libraries), which the assertion
    message repeats."""
    comparisons = two_stage_fit_comparisons(label, fit_twin, twins, max_twins, bt_extra)
    for c in comparisons:
        check.record(
            f"{label}: first attempts failed (botorch_rs - BoTorch)",
            max(c.retries[1] - c.retries[0], 0),
            None,
        )
        check.record(f"{label}: BoTorch's spread at its best optimum", c.spread, None)
        if c.z is None or c.kth is None:
            check.record(f"{label}: best loss above BoTorch's best (scaled)", c.best_excess, 1.0)
            check.record(f"{label}: loss above BoTorch's worst (scaled)", c.worst_excess, 1.0)
        else:
            print(c.diagnostics())
            check.record(
                f"{label} (stage 2): stage-1 excess (scaled)",
                max(c.best_excess, c.worst_excess),
                None,
            )
            check.record(f"{label} (stage 2): Mann-Whitney z", max(c.z, 0.0), FIT_Z)
            check.record(
                f"{label} (stage 2): best above BoTorch's k-th best",
                max(min(c.rs, default=math.inf) - c.kth, 0.0),
                FIT_TOL,
            )
        assert c.passed, c.diagnostics()
    return comparisons


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

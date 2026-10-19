"""Shared helpers of the phase-10b tests (preference learning): pairs of
BoTorch / botorch_rs ``PairwiseGP`` models built from the same data and the
same numpy draws, and their comparisons.

Import only after ``importorskip_botorch()`` (requires torch, numpy and
BoTorch).
"""

from __future__ import annotations

import contextlib
import math
import random
import warnings
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
from botorch.models import PairwiseGP as BtPGP
from botorch.models.likelihoods.pairwise import PairwiseLogitLikelihood as BtLogit
from botorch.models.likelihoods.pairwise import PairwiseProbitLikelihood as BtProbit
from botorch.models.pairwise_gp import PairwiseLaplaceMarginalLogLikelihood as BtMLL
from botorch.models.transforms.input import Normalize as BtNormalize
from kernel_tree_helpers import build_kernel

from botorch_rs.models import PairwiseGP as RsPGP
from botorch_rs.models import PairwiseLaplaceMarginalLogLikelihood as RsMLL
from botorch_rs.models.likelihoods.pairwise import PairwiseLogitLikelihood as RsLogit
from botorch_rs.models.likelihoods.pairwise import PairwiseProbitLikelihood as RsProbit
from botorch_rs.models.transforms import Normalize as RsNormalize
from parity_helpers import Checker, rel_err

DT = torch.float64
#: Values (relative) and gradients (relative to the largest component).
VALUE_TOL = 1e-9
GRAD_TOL = 1e-8


@dataclass
class Case:
    """A ``PairwiseGP`` configuration."""

    n: int
    d: int
    m: int
    likelihood: str = "probit"
    kernel: dict[str, Any] | None = None
    normalize: bool = False
    options: dict[str, Any] = field(default_factory=dict)
    scale: float = 1.0
    dups: bool = False
    batch: tuple[int, ...] = ()

    def label(self) -> str:
        parts = [self.likelihood, f"n{self.n}", f"d{self.d}", f"m{self.m}"]
        if self.kernel is not None:
            parts.append(self.kernel["type"])
        if self.normalize:
            parts.append("normalize")
        if self.dups:
            parts.append("dups")
        if self.batch:
            parts.append(f"batch{self.batch}")
        parts.extend(f"{k}={v}" for k, v in self.options.items())
        return "-".join(parts)


def make_data(case: Case, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Random datapoints and comparisons (no comparison of an item with
    itself or, with ``dups``, of a near-duplicate with its original)."""
    g = torch.Generator().manual_seed(seed)
    X = torch.rand(*case.batch, case.n, case.d, generator=g, dtype=DT) * case.scale
    bad: set[tuple[int, int]] = set()
    if case.dups:
        X[..., case.n - 1, :] = X[..., 0, :] + 2e-5
        X[..., case.n - 2, :] = X[..., 3, :] - 3e-5
        bad = {(case.n - 1, 0), (0, case.n - 1), (case.n - 2, 3), (3, case.n - 2)}
    nb = math.prod(case.batch)
    comps = []
    for _ in range(nb):
        cb: list[list[int]] = []
        while len(cb) < case.m:
            a, b = torch.randint(0, case.n, (2,), generator=g).tolist()
            if a != b and (a, b) not in bad:
                cb.append([a, b])
        comps.append(cb)
    C = torch.tensor(comps, dtype=torch.long).reshape(*case.batch, case.m, 2)
    return X, C


def build_pair(case: Case, X: Any, C: Any, seed: int = 0) -> tuple[Any, Any]:
    """BoTorch's and botorch_rs's models of ``case`` on ``(X, C)``, both built
    after ``np.random.seed(seed)`` (the same initial utility)."""
    kw_bt, kw_rs = model_kwargs(case)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        np.random.seed(seed)
        bt = BtPGP(X, C, **kw_bt)
        np.random.seed(seed)
        rs = RsPGP(X, C, **kw_rs)
    return bt, rs


def build_rs(case: Case, X: Any, C: Any, seed: int = 0) -> Any:
    """botorch_rs's model of ``case`` (see :func:`build_pair`)."""
    _, kw_rs = model_kwargs(case)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        np.random.seed(seed)
        return RsPGP(X, C, **kw_rs)


def model_kwargs(case: Case) -> tuple[dict[str, Any], dict[str, Any]]:
    """The keyword arguments of BoTorch's and botorch_rs's models of ``case``."""
    kw_bt: dict[str, Any] = dict(case.options)
    kw_rs: dict[str, Any] = dict(case.options)
    if case.likelihood == "logit":
        kw_bt["likelihood"], kw_rs["likelihood"] = BtLogit(), RsLogit()
    elif case.likelihood == "probit_explicit":
        kw_bt["likelihood"], kw_rs["likelihood"] = BtProbit(), RsProbit()
    if case.kernel is not None:
        kw_bt["covar_module"] = build_kernel(case.kernel, "bt").to(DT)
        kw_rs["covar_module"] = build_kernel(case.kernel, "rs")
    if case.normalize:
        kw_bt["input_transform"] = BtNormalize(d=case.d)
        kw_rs["input_transform"] = RsNormalize(d=case.d)
    return kw_bt, kw_rs


def cond_eps(K: Any) -> float:
    """``cond(K) · eps``, the accuracy of an inverse of the SPD matrix ``K``."""
    e = torch.linalg.eigvalsh(torch.as_tensor(K, dtype=DT))
    return float((e.abs().max() / e.min()).item() * torch.finfo(DT).eps)


def inverse_err(a: Any, e: Any, K: Any) -> float:
    """The normwise error of an inverse relative to ``cond(K) · eps``."""
    a, e = torch.as_tensor(a, dtype=DT), torch.as_tensor(e, dtype=DT)
    err = ((a - e).abs().max() / e.abs().max()).item()
    return err / max(cond_eps(K), VALUE_TOL)


def xtol_of(case: Case) -> float:
    return float(case.options.get("xtol") or 1e-6)


def compare_state(check: Checker, bt: Any, rs: Any, xtol: float, what: str = "") -> None:
    """The consolidated data (exact) and the Laplace state: the fsolve
    solution to the solver's tolerance (``xtol``), the utility, Hessian,
    covariance and its Cholesky factor to 1e-9, the inverse covariance to
    ``cond(K) · eps``."""
    pre = f"{what}: " if what else ""
    assert torch.equal(torch.as_tensor(rs.datapoints, dtype=DT), bt.datapoints)
    assert torch.equal(torch.as_tensor(rs.comparisons), bt.comparisons.long())
    if bt.consolidated_indices is None:
        assert rs.consolidated_indices is None
    else:
        assert torch.equal(torch.as_tensor(rs.consolidated_indices), bt.consolidated_indices)
    check.check(f"{pre}fsolve solution (tol xtol)", rs._x0, bt._x0, xtol)
    check.check(f"{pre}utility", rs.utility, bt.utility.detach(), VALUE_TOL)
    check.check(f"{pre}likelihood_hess", rs.likelihood_hess, bt.likelihood_hess.detach(), VALUE_TOL)
    check.check(f"{pre}covar", rs.covar, bt.covar.detach(), VALUE_TOL)
    check.check(f"{pre}covar_chol", rs.covar_chol, bt.covar_chol.detach(), VALUE_TOL)
    K = bt.covar.detach()
    if K.dim() == 2:
        err = inverse_err(rs.covar_inv, bt.covar_inv.detach(), K)
    else:
        err = max(
            inverse_err(a, e, k)
            for a, e, k in zip(
                torch.as_tensor(rs.covar_inv, dtype=DT).reshape(-1, *K.shape[-2:]),
                bt.covar_inv.detach().reshape(-1, *K.shape[-2:]),
                K.reshape(-1, *K.shape[-2:]),
                strict=True,
            )
        )
    check.record(f"{pre}covar_inv / (cond(K) eps)", err, 1.0)
    assert err <= 1.0, err


def raw_params(model: Any) -> list[float]:
    """The raw hyperparameters that are fitted (``requires_grad``)."""
    return [
        v for _, p in model.named_parameters() if p.requires_grad for v in p.reshape(-1).tolist()
    ]


def set_raw(bt: Any, rs: Any, raw: list[float]) -> None:
    off = 0
    with torch.no_grad():
        for _, p in bt.named_parameters():
            if not p.requires_grad:
                continue
            k = p.numel()
            p.copy_(torch.tensor(raw[off : off + k], dtype=DT).reshape(p.shape))
            off += k
    rs._rs.set_raw_parameters([float(v) for v in raw])


def bt_mll_grad(bt: Any) -> tuple[float, list[float]]:
    """BoTorch's training forward pass, Laplace MLL and autograd gradient."""
    bt.train()
    params = [p for _, p in bt.named_parameters() if p.requires_grad]
    for p in params:
        p.grad = None
    mll = BtMLL(bt.likelihood, bt)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        v = mll(bt(bt.unconsolidated_datapoints), bt.unconsolidated_comparisons)
    v.backward()
    grads = [
        g
        for p in params
        for g in (p.grad if p.grad is not None else torch.zeros_like(p)).reshape(-1).tolist()
    ]
    return v.item(), grads


def rs_mll_grad(rs: Any) -> tuple[float, list[float]]:
    rs.train()
    return RsMLL(rs.likelihood, rs).value_and_grad()


def grad_err(a: Any, e: Any) -> float:
    a, e = torch.as_tensor(a, dtype=DT), torch.as_tensor(e, dtype=DT)
    scale = e.abs().max().clamp_min(1e-300)
    return float(((a - e).abs() / scale).max().item())


def bt_fit(bt: Any) -> tuple[float, int]:
    """BoTorch's ``fit_gpytorch_mll`` (final loss, attempts)."""
    import botorch.optim.fit as of
    from botorch.fit import fit_gpytorch_mll

    record: dict[str, Any] = {"attempts": 0}
    orig = of.fit_gpytorch_mll_scipy

    def recorder(mll: Any, *args: Any, **kwargs: Any) -> Any:
        record["attempts"] += 1
        record["r"] = orig(mll, *args, **kwargs)
        return record["r"]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit_gpytorch_mll(BtMLL(bt.likelihood, bt), optimizer=recorder)
    return float(record["r"].fval), record["attempts"]


def rs_fit(rs: Any) -> tuple[float, int]:
    from botorch_rs.fit import fit_gpytorch_mll

    mll = RsMLL(rs.likelihood, rs)
    fit_gpytorch_mll(mll)
    return float(mll._fit_result["fval"]), int(mll._fit_result["attempts"])


@contextlib.contextmanager
def fixed_pairwise_sampler() -> Iterator[None]:
    """BoTorch's pairwise samplers with the orientation of the comparisons
    fixed (the preferred item first; BoTorch lists the item with the smaller
    sampled utility first, a bug)."""
    import botorch.sampling.pairwise_samplers as bps

    orig = bps.PairwiseMCSampler.forward

    def forward(self: Any, posterior: Any) -> Any:
        comps = orig(self, posterior)
        return comps.flip(-1)

    bps.PairwiseMCSampler.forward = forward
    try:
        yield
    finally:
        bps.PairwiseMCSampler.forward = orig


def random_points(model: Any, shape: tuple[int, ...], seed: int, pad: float = 0.1) -> Any:
    """Random points in the box of the model's datapoints (padded)."""
    dp = model.datapoints.reshape(-1, model.datapoints.shape[-1])
    lo, hi = dp.min(0).values, dp.max(0).values
    width = (hi - lo).clamp_min(1e-3)
    g = torch.Generator().manual_seed(seed)
    u = torch.rand(*shape, dp.shape[-1], generator=g, dtype=DT)
    return lo - pad * width + (1 + 2 * pad) * width * u


def rel(a: Any, e: Any) -> float:
    return rel_err(a, e)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


__all__ = [
    "DT",
    "GRAD_TOL",
    "VALUE_TOL",
    "Case",
    "bt_fit",
    "bt_mll_grad",
    "build_pair",
    "build_rs",
    "compare_state",
    "fixed_pairwise_sampler",
    "grad_err",
    "make_data",
    "random_points",
    "raw_params",
    "rs_fit",
    "rs_mll_grad",
    "set_raw",
    "xtol_of",
]

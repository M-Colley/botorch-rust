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


def bt_newton_utility(bt: Any, x0: Any) -> Any:
    """BoTorch's utility from the fsolve solution ``x0``: the two Newton
    steps of ``PairwiseGP._update`` with the model's current state (its
    ``likelihood_hess`` is restored). From BoTorch's own ``_x0`` this is
    BoTorch's utility; from botorch_rs's fsolve solution it gives both
    libraries the same input to everything downstream of the MAP solve."""
    hess = bt.likelihood_hess
    try:
        with torch.no_grad():
            dp = bt.transform_inputs(bt.datapoints)
            x = torch.as_tensor(x0, dtype=DT).reshape(bt.utility.shape)
            return bt._util_newton_updates(dp=dp, x0=x, max_iter=2).detach()
    finally:
        bt.likelihood_hess = hess


#: Unit roundoff of float64.
UNIT_ROUNDOFF = 2.0**-53


def cholesky_ambiguous(A: Any) -> bool:
    """Whether the outcome of a float64 Cholesky factorization of the
    symmetric matrix ``A`` (``n x n``) is decided by rounding: its smallest
    eigenvalue is within ``20 n^{3/2} u ||A||_2`` of zero, the bound of
    Demmel's theorem (Higham, *Accuracy and Stability of Numerical
    Algorithms*, 2nd ed., Thm. 10.7) below which the factorization may fail
    or succeed depending on the order of the operations (and above which it
    succeeds in every implementation). Two libraries can then decide
    differently whether to add jitter; both decisions follow BoTorch's
    algorithm."""
    A = torch.as_tensor(A, dtype=DT).detach()
    n = A.shape[-1]
    ev = torch.linalg.eigvalsh(A)
    bound = 20 * n**1.5 * UNIT_ROUNDOFF * float(ev.abs().max())
    return bool(float(ev[0].abs()) <= bound)


#: The relative accuracy of the Cholesky factor of a positive semidefinite
#: ``2 x 2`` matrix that is singular within rounding: a perturbation of size
#: ``20 n^{3/2} u ||A||`` (the ambiguity bound of :func:`cholesky_ambiguous`)
#: changes the factor by up to its square root (the factor of a singular
#: matrix is Hoelder continuous with exponent 1/2), i.e. by 7.9e-8 relative.
ROOT_TOL = math.sqrt(20 * 2**1.5 * UNIT_ROUNDOFF)


class BtJitterDecisions:
    """BoTorch's two jitter decisions of ``PairwiseGP`` posteriors, recorded
    and optionally flipped where they are ambiguous (:func:`cholesky_ambiguous`):
    ``_ensure_psd_with_jitter`` of the prior covariance of the new points and
    ``psd_safe_cholesky`` of the posterior covariance (in
    ``_scaled_psd_safe_cholesky``). Use on one t-batch at a time.

    Each Cholesky attempt with an ambiguous outcome gets an index (in call
    order; ``ambiguous`` counts them); the attempts in ``flip`` take the other
    outcome. A flipped failure is a success on the matrix itself (factorized
    as ``A + 2 |lambda_min| I``, which differs from ``A`` by rounding only), a
    flipped success adds the next jitter as BoTorch does after a failure."""

    def __init__(self, flip: frozenset[int] = frozenset()) -> None:
        self.flip = flip
        self.ambiguous = 0
        #: Whether a posterior covariance that was factorized is singular
        #: within rounding (its root is then determined only to
        #: :data:`ROOT_TOL`).
        self.singular_root = False

    def _decide(self, A: Any) -> bool | None:
        """The flipped outcome of an ambiguous attempt in ``flip`` (else None)."""
        if A.reshape(-1, *A.shape[-2:]).shape[0] != 1 or not cholesky_ambiguous(
            A.reshape(A.shape[-2:])
        ):
            return None
        k = self.ambiguous
        self.ambiguous += 1
        if k not in self.flip:
            return None
        _, info = torch.linalg.cholesky_ex(A)
        return bool((info != 0).any())  # the opposite of the actual success

    @staticmethod
    def _factor(A: Any) -> Any:
        L, info = torch.linalg.cholesky_ex(A)
        if bool((info == 0).all()):
            return L
        lam = torch.linalg.eigvalsh(A)[..., :1].abs().unsqueeze(-1)
        eye = torch.eye(A.shape[-1], dtype=A.dtype, device=A.device)
        return torch.linalg.cholesky(A + 2 * lam * eye)

    def ensure_psd_with_jitter(
        self, matrix: Any, scale: Any = 1.0, jitter: float = 1e-8, max_tries: int = 3
    ) -> Any:
        """``botorch.models.pairwise_gp._ensure_psd_with_jitter``."""
        from linear_operator.utils.errors import NotPSDError

        scaled = matrix / scale
        new_jitter: float = 0.0
        psd = False
        for i in range(max_tries):
            scaled = scaled + new_jitter * torch.diag_embed(
                torch.ones(scaled.shape[:-1], device=scaled.device, dtype=scaled.dtype)
            )
            _, info = torch.linalg.cholesky_ex(scaled)
            psd = bool((info == 0).all())
            forced = self._decide(scaled.detach())
            if forced is not None:
                psd = forced
            if psd:
                break
            new_jitter = jitter * (10**i) - new_jitter
        if not psd:
            raise NotPSDError(
                "Matrix not positive definite after repeatedly adding jitter "
                f"up to {jitter * (10**i):.1e}."
            )
        return scaled * scale

    def scaled_psd_safe_cholesky(self, matrix: Any, scale: Any, jitter: float | None = None) -> Any:
        """``botorch.models.pairwise_gp._scaled_psd_safe_cholesky`` (with
        linear_operator's ``psd_safe_cholesky`` for one matrix)."""
        from linear_operator import settings as lo_settings
        from linear_operator.utils.errors import NotPSDError

        A = matrix / scale
        if jitter is None:
            jitter = lo_settings.cholesky_jitter.value(A.dtype)
        max_tries = lo_settings.cholesky_max_tries.value()
        Aprime = A
        jitter_prev = 0.0
        for i in range(max_tries + 1):
            if i > 0:
                jitter_new = jitter * (10 ** (i - 1))
                eye = torch.eye(A.shape[-1], dtype=A.dtype, device=A.device)
                Aprime = Aprime + (jitter_new - jitter_prev) * eye
                jitter_prev = jitter_new
            _, info = torch.linalg.cholesky_ex(Aprime)
            ok = bool((info == 0).all())
            forced = self._decide(Aprime.detach())
            if forced is not None:
                ok = forced
            if ok:
                if cholesky_ambiguous(Aprime.reshape(Aprime.shape[-2:])):
                    self.singular_root = True
                return self._factor(Aprime) * scale.sqrt()
        raise NotPSDError("Matrix not positive definite after repeatedly adding jitter.")

    @contextlib.contextmanager
    def patched(self) -> Iterator[BtJitterDecisions]:
        import botorch.models.pairwise_gp as bt_pgp

        orig = (bt_pgp._ensure_psd_with_jitter, bt_pgp._scaled_psd_safe_cholesky)
        bt_pgp._ensure_psd_with_jitter = self.ensure_psd_with_jitter  # type: ignore[assignment]
        bt_pgp._scaled_psd_safe_cholesky = self.scaled_psd_safe_cholesky  # type: ignore[assignment]
        try:
            yield self
        finally:
            bt_pgp._ensure_psd_with_jitter, bt_pgp._scaled_psd_safe_cholesky = orig


def xtol_of(case: Case) -> float:
    return float(case.options.get("xtol") or 1e-6)


def compare_state(check: Checker, bt: Any, rs: Any, xtol: float, what: str = "") -> None:
    """The consolidated data (exact) and the Laplace state: the utility,
    Hessian, covariance and its Cholesky factor to 1e-9, the inverse
    covariance to ``cond(K) · eps``. The fsolve solutions are not compared
    (rounding moves fsolve's last iterate within its tolerance ``xtol``; the
    difference relative to ``xtol`` is recorded): BoTorch's Newton steps from
    botorch_rs's fsolve solution must give botorch_rs's utility to 1e-9
    (:func:`bt_newton_utility`), the same input to everything downstream."""
    pre = f"{what}: " if what else ""
    assert torch.equal(torch.as_tensor(rs.datapoints, dtype=DT), bt.datapoints)
    assert torch.equal(torch.as_tensor(rs.comparisons), bt.comparisons.long())
    if bt.consolidated_indices is None:
        assert rs.consolidated_indices is None
    else:
        assert torch.equal(torch.as_tensor(rs.consolidated_indices), bt.consolidated_indices)
    check.record(
        f"{pre}fsolve solution / xtol (not compared)", rel_err(rs._x0, bt._x0) / xtol, None
    )  # type: ignore[arg-type]
    check.check(
        f"{pre}utility vs BoTorch's Newton steps from botorch_rs's fsolve solution",
        rs.utility,
        bt_newton_utility(bt, rs._x0),
        VALUE_TOL,
    )
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

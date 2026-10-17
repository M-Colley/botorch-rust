"""Shared helpers of the live comparisons of the models of phase 7b
(``test_parity_7b_models.py``, ``test_equivalence_7b_models.py``):
model structure, optimizer bounds, the MLL and its gradient, posteriors and
fits of a BoTorch model and its botorch_rs counterpart.

Import only after ``importorskip_botorch()``.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

import torch
from botorch.acquisition import PosteriorMean as BtPosteriorMean
from botorch.acquisition import PosteriorStandardDeviation as BtPSD
from botorch.fit import fit_gpytorch_mll as bt_fit
from botorch.optim.utils import get_parameters_and_bounds
from gpytorch.mlls import ExactMarginalLogLikelihood as BtMLL
from kernel_tree_helpers import zero_diagonal_distances

from botorch_rs.acquisition import PosteriorMean as RsPosteriorMean
from botorch_rs.acquisition import PosteriorStandardDeviation as RsPSD
from botorch_rs.fit import fit_gpytorch_mll as rs_fit
from botorch_rs.mlls import ExactMarginalLogLikelihood as RsMLL
from parity_helpers import CHAOTIC_TWINS, Checker, compare_chaotic_fits, rel_err

DT = torch.float64
VALUE_TOL = 1e-9
GRAD_TOL = 1e-8
FIT_TOL = 1e-6
FIT_CLUSTER = 1e-4
CHAOTIC_RANGE = 1e-3
TWIN_SCALES = (1e-14, 1e-13, 1e-12, 1e-11)


def grad_err(a: Any, e: Any) -> float:
    """Gradient error relative to the largest component."""
    a = torch.as_tensor(a, dtype=DT).reshape(-1)
    e = torch.as_tensor(e, dtype=DT).reshape(-1)
    assert a.shape == e.shape, (a.shape, e.shape)
    scale = max(e.abs().max().item(), 1e-300)
    return (a - e).abs().max().item() / scale


def check_grad(check: Checker, quantity: str, a: Any, e: Any, tol: float = GRAD_TOL) -> None:
    err = grad_err(a, e)
    check.record(quantity, err, tol)
    assert err <= tol, (quantity, err)


def bounds_of(bt: Any) -> dict[str, tuple[float, float]]:
    """BoTorch's optimizer bounds by parameter name."""
    _, bounds = get_parameters_and_bounds(BtMLL(bt.likelihood, bt))
    out = {}
    for name, (lo, hi) in bounds.items():
        out[name.removeprefix("model.")] = (
            -math.inf if lo is None else float(lo),
            math.inf if hi is None else float(hi),
        )
    return out


def compare_structure(check: Checker, bt: Any, rs: Any, exact_defaults: bool = True) -> None:
    """Parameter names and shapes (equal), default raw values (exact) and
    optimizer bounds (exact)."""
    nb = [(n, tuple(p.shape)) for n, p in bt.named_parameters()]
    nr = [(n, tuple(torch.as_tensor(p).shape)) for n, p in rs.named_parameters()]
    assert nb == nr, (nb, nr)
    if exact_defaults:
        for (_, pb), (_, pr) in zip(bt.named_parameters(), rs.named_parameters(), strict=True):
            check.check("default raw parameters (exact)", pr, pb.detach(), 0.0)
    bounds = bounds_of(bt)
    expected = []
    for name, p in bt.named_parameters():
        expected.extend([bounds.get(name, (-math.inf, math.inf))] * p.numel())
    actual = [tuple(b) for b in rs._rs.raw_parameter_bounds()]
    assert actual == expected, (actual, expected)


def randomize(bt: Any, rs: Any, seed: int, scale: float = 0.5) -> None:
    """Random raw parameters (within the optimizer bounds) in both models."""
    g = torch.Generator().manual_seed(seed)
    bounds = bounds_of(bt)
    with torch.no_grad():
        for name, p in bt.named_parameters():
            lo, hi = bounds.get(name, (-math.inf, math.inf))
            if "raw_noise" in name and not math.isfinite(lo):
                v = torch.full(p.shape, 0.03, dtype=DT).expm1().log()
            elif "raw_noise" in name and not math.isfinite(hi):
                v = torch.full(p.shape, 0.03, dtype=DT).clamp_min(lo)
            else:
                v = p.detach() + scale * torch.randn(p.shape, generator=g, dtype=DT)
                if math.isfinite(lo) and math.isfinite(hi):
                    v = v.clamp(lo + 0.05 * (hi - lo) / 10, hi - 0.05 * (hi - lo) / 10)
                elif math.isfinite(lo):
                    v = lo + 0.1 + (v - lo).abs()
                elif math.isfinite(hi):
                    v = v.clamp(max=hi - 0.05)
            p.copy_(v)
    rs.initialize(**{name: p.detach() for name, p in bt.named_parameters()})


def bt_mll_value_and_grad(bt: Any) -> tuple[Any, Any]:
    mll = BtMLL(bt.likelihood, bt)
    bt.train()
    bt.zero_grad()
    values = mll(bt(*bt.train_inputs), bt.train_targets)
    values.sum().backward()
    grad = torch.cat(
        [
            (p.grad if p.grad is not None else torch.zeros_like(p)).reshape(-1)
            for _, p in bt.named_parameters()
        ]
    )
    bt.zero_grad()
    bt.eval()
    return values.detach().reshape(-1), grad


def rs_mll_value_and_grad(rs: Any) -> tuple[Any, Any]:
    value, grad = RsMLL(rs.likelihood, rs).value_and_grad()
    return torch.as_tensor(value, dtype=DT).reshape(-1), torch.as_tensor(grad, dtype=DT)


def compare_mll(check: Checker, label: str, bt: Any, rs: Any) -> None:
    """The MLL and its gradient, with BoTorch's evaluated with GPyTorch's
    ``sq_dist`` zeroing the distances of the training inputs to themselves
    also when they require gradients (as without gradients; Matérn-1/2
    kernels, e.g. of ``SingleTaskMultiFidelityGP(nu=0.5)``, see the BoTorch
    bugs of phase 6a). The change this makes is recorded as BoTorch's
    noise."""
    vb_raw, gb_raw = bt_mll_value_and_grad(bt)
    with zero_diagonal_distances():
        vb, gb = bt_mll_value_and_grad(bt)
    vr, gr = rs_mll_value_and_grad(rs)
    check.check(f"{label}: MLL", vr, vb, VALUE_TOL)
    check_grad(check, f"{label}: MLL gradient", gr, gb)
    check.record(f"{label}: BoTorch's sq_dist diagonal noise (MLL)", rel_err(vb_raw, vb), None)
    check.record(
        f"{label}: BoTorch's sq_dist diagonal noise (MLL gradient)", grad_err(gb_raw, gb), None
    )


def values_and_grads(acqf: Any, X: Any) -> tuple[Any, Any]:
    Xg = X.detach().clone().requires_grad_(True)
    v = acqf(Xg)
    if not v.requires_grad:
        # Constant in X (e.g. the zero standard deviation of a one-member
        # EnsembleModel in BoTorch).
        return v.detach(), torch.zeros_like(X)
    (g,) = torch.autograd.grad(v.sum(), Xg, allow_unused=True)
    return v.detach(), torch.zeros_like(X) if g is None else g


def compare_posteriors(
    check: Checker, label: str, bt: Any, rs: Any, Xt: Any, analytic: bool = True
) -> None:
    """Posterior means, variances and covariances (with and without
    observation noise), and the values and input gradients of
    ``PosteriorMean`` / ``PosteriorStandardDeviation`` (single output)."""
    bt.eval()
    with torch.no_grad():
        for noise in (False, True):
            pb = bt.posterior(Xt, observation_noise=noise)
            pr = rs.posterior(Xt, observation_noise=noise)
            check.check(f"{label}: posterior mean", pr.mean, pb.mean, VALUE_TOL)
            check.check(f"{label}: posterior variance", pr.variance, pb.variance, VALUE_TOL)
            check.check(
                f"{label}: posterior covariance",
                pr.distribution.covariance_matrix,
                pb.distribution.covariance_matrix,
                VALUE_TOL,
            )
    if analytic and bt.num_outputs == 1:
        X1 = Xt.reshape(-1, 1, Xt.shape[-1])
        for b, r, name in (
            (BtPosteriorMean(bt), RsPosteriorMean(rs), "PosteriorMean"),
            (BtPSD(bt), RsPSD(rs), "PosteriorStandardDeviation"),
        ):
            vb, gb = values_and_grads(b, X1)
            vr, gr = values_and_grads(r, X1)
            check.check(f"{label}: {name} values", vr, vb, VALUE_TOL)
            check_grad(check, f"{label}: {name} gradients", gr, gb)


# ---------------------------------------------------------------------------
# Fits
# ---------------------------------------------------------------------------


def bt_fit_losses(bt: Any) -> tuple[list[float], int]:
    """BoTorch's ``fit_gpytorch_mll``: the final negative MLL of every
    element of the model and the number of attempts."""
    import botorch.optim.fit as bt_optim_fit

    record: dict[str, Any] = {"attempts": 0}
    orig = bt_optim_fit.fit_gpytorch_mll_scipy

    def recorder(mll: Any, *args: Any, **kwargs: Any) -> Any:
        record["attempts"] += 1
        return orig(mll, *args, **kwargs)

    bt_fit(BtMLL(bt.likelihood, bt), optimizer=recorder)
    vb, _ = bt_mll_value_and_grad(bt)
    return [-float(v) for v in vb], record["attempts"]


def rs_fit_losses(rs: Any) -> tuple[list[float], int]:
    mll = RsMLL(rs.likelihood, rs)
    rs_fit(mll)
    vr, _ = rs_mll_value_and_grad(rs)
    return [-float(v) for v in vr], int(mll._fit_result["attempts"])


def twin_inputs(X0: Any, t: int, cols: list[int] | None = None) -> Any:
    """Twin ``t`` of the training inputs ``X0`` (``t = 0``: ``X0``): the
    columns ``cols`` (all by default) perturbed at a relative 1e-14 .. 1e-11."""
    if t == 0:
        return X0
    g = torch.Generator().manual_seed(t)
    rel = TWIN_SCALES[(t - 1) % len(TWIN_SCALES)]
    X = X0.clone()
    cols = list(range(X.shape[-1])) if cols is None else cols
    X[..., cols] = X[..., cols] * (1 + rel * torch.randn(X[..., cols].shape, generator=g, dtype=DT))
    return X


def compare_fits(
    check: Checker,
    label: str,
    make_pair: Callable[[Any], tuple[Any, Any]],
    X0: Any,
    cols: list[int] | None = None,
    twins: int = 4,
    chaotic_twins: int = CHAOTIC_TWINS,
) -> None:
    """Fits of the model and its twins from identical initial values, per
    model element (outputs, ensemble members): botorch_rs's best
    first-attempt loss must reach BoTorch's best and its own loss must not
    end above BoTorch's worst, within ``max(1e-6, 10 x`` BoTorch's spread at
    its best optimum``)``; on chaotic likelihoods (BoTorch's outcomes spread
    over more than 1e-3, or fewer than a quarter of them within 1e-4 of the
    best) statistically on ``chaotic_twins`` twins
    (``parity_helpers.compare_chaotic_fits``)."""
    bt_out: list[tuple[list[float], int]] = []
    rs_out: list[tuple[list[float], int]] = []

    def run(ts: range) -> None:
        for t in ts:
            bt, rs = make_pair(twin_inputs(X0, t, cols))
            bt_out.append(bt_fit_losses(bt))
            rs_out.append(rs_fit_losses(rs))

    run(range(twins + 1))
    n_el = len(bt_out[0][0])

    def first(out: list[tuple[list[float], int]], e: int) -> list[float]:
        return [losses[e] for losses, a in out if a == 1]

    if any(max(first(bt_out, e)) - min(first(bt_out, e)) > CHAOTIC_RANGE for e in range(n_el)):
        run(range(twins + 1, chaotic_twins + 1))
    check.record(
        f"{label}: first attempts failed (botorch_rs - BoTorch)",
        max(sum(a > 1 for _, a in rs_out) - sum(a > 1 for _, a in bt_out), 0),
        None,
    )
    for e in range(n_el):
        bt_first, rs_first = first(bt_out, e), first(rs_out, e)
        if not bt_first or not rs_first:
            continue
        best, worst = min(bt_first), max(bt_first)
        spread = max(v for v in bt_first if v <= best + FIT_CLUSTER) - best
        tol = max(FIT_TOL, 10 * spread)
        # Multimodal: BoTorch itself rarely reaches its best outcome (fewer
        # than a quarter of its fits within FIT_CLUSTER of it), compared
        # statistically like chaotic likelihoods.
        at_best = sum(v <= best + FIT_CLUSTER for v in bt_first)
        if worst - best > CHAOTIC_RANGE or 4 * at_best < len(bt_first):
            compare_chaotic_fits(check, label, bt_first, rs_first, FIT_TOL)
        else:
            check.check_bound(
                f"{label}: best loss above BoTorch's best (scaled)",
                (min(rs_first) - best) / tol,
                1.0,
            )
            if rs_out[0][1] == 1:
                check.check_bound(
                    f"{label}: loss above BoTorch's worst (scaled)",
                    (rs_out[0][0][e] - worst) / tol,
                    1.0,
                )

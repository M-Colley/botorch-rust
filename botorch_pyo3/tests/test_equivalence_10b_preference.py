"""Randomized equivalence sweep of phase 10b (preference learning) against
live BoTorch (``-m equivalence``): random comparisons of random datapoints
(``n``, ``d``, ``m``, near-duplicates, scales), probit and logit
likelihoods, BoTorch's default kernel or a random ``ScaleKernel`` around a
random kernel tree (``kernel_tree_helpers.random_spec``), ``Normalize`` and
solver options. Every case compares the Laplace state after construction
(the same numpy draws), the MLL and its gradient at the default and at
random parameters, posteriors, analytic EUBO and qEUBO (identical base
samples) values and input gradients, BALD (the same torch seed); every
fourth case also the fit (on twins, ``compare_twin_fits``).

Tolerances: values 1e-9, gradients 1e-8. Where rounding decides a
discrete step of BoTorch's algorithm, both libraries get the same input
instead of a wider tolerance (``docs/preference.md``, "Equivalence tests"):
the fsolve solutions are not compared, BoTorch's Newton steps from
botorch_rs's fsolve solution must give botorch_rs's utility; every t-batch
is evaluated on its own (BoTorch couples them, bug 6), and a t-batch that
differs must match BoTorch's result for the other outcome of a jitter
decision on a matrix that is singular within rounding (Demmel's bound,
``preference_helpers.cholesky_ambiguous``). Sample-based quantities of a
t-batch whose posterior covariance is singular within rounding are compared
to ``ROOT_TOL`` (the accuracy of its Cholesky factor). BoTorch's numerical
quirks of kernels (``HalfCauchyPrior`` in float32, the diagonal of
``sq_dist``) are fixed for the comparison as in the 6a tests.
"""

from __future__ import annotations

import random
import warnings
from typing import Any

import pytest

from helpers import importorskip_botorch

importorskip_botorch()
torch = pytest.importorskip("torch")
np = pytest.importorskip("numpy")

from botorch.acquisition import preference as bt_pref  # noqa: E402
from botorch.sampling.normal import SobolQMCNormalSampler as BtSampler  # noqa: E402
from kernel_tree_helpers import (  # noqa: E402
    double_precision_half_cauchy,
    kernel,
    random_spec,
    sub,
    zero_diagonal_distances,
)
from preference_helpers import (  # noqa: E402
    DT,
    GRAD_TOL,
    ROOT_TOL,
    VALUE_TOL,
    BtJitterDecisions,
    Case,
    bt_fit,
    bt_mll_grad,
    bt_newton_utility,
    build_pair,
    build_rs,
    grad_err,
    inverse_err,
    make_data,
    random_points,
    raw_params,
    rs_fit,
    rs_mll_grad,
    xtol_of,
)

from botorch_rs.acquisition import preference as rs_pref  # noqa: E402
from botorch_rs.sampling import SobolQMCNormalSampler as RsSampler  # noqa: E402
from parity_helpers import Checker, compare_twin_fits, rel_err, twin_inputs  # noqa: E402

pytestmark = [pytest.mark.equivalence, pytest.mark.filterwarnings("ignore")]

check = Checker("10b preference sweep")

SEEDS = list(range(40))


def random_case(seed: int) -> Case:
    r = random.Random(seed)
    d = r.randint(1, 4)
    n = r.randint(4, 18)
    m = r.randint(2, 3 * n)
    kernel_spec = None
    if r.random() < 0.5:
        node = random_spec(r, d)
        kernel_spec = node if node["type"] == "ScaleKernel" else kernel("ScaleKernel", sub(node))
    options: dict[str, object] = {}
    if r.random() < 0.3:
        options["jitter"] = r.choice([1e-5, 1e-4])
    if r.random() < 0.3:
        options["xtol"] = r.choice([1e-8, 1e-5])
    return Case(
        n,
        d,
        m,
        likelihood=r.choice(["probit", "logit"]),
        kernel=kernel_spec,
        normalize=r.random() < 0.25,
        options=options,
        scale=r.choice([1.0, 1.0, 3.0]),
        dups=n >= 5 and r.random() < 0.2,
    )


class _Guarded:
    """BoTorch's kernel quirks fixed for the comparison."""

    def __enter__(self) -> None:
        self._a = double_precision_half_cauchy()
        self._b = zero_diagonal_distances()
        self._a.__enter__()
        self._b.__enter__()

    def __exit__(self, *exc: object) -> None:
        self._b.__exit__(*exc)
        self._a.__exit__(*exc)


def _vg(acqf: Any, X: Any) -> tuple[Any, Any]:
    X = X.clone().requires_grad_(True)
    v = acqf(X)
    v.sum().backward()
    return v.detach(), X.grad


def _set_bt_raw(bt: Any, raw: list[float]) -> None:
    off = 0
    with torch.no_grad():
        for _, p in bt.named_parameters():
            if p.requires_grad:
                k = p.numel()
                p.copy_(torch.tensor(raw[off : off + k], dtype=DT).reshape(p.shape))
                off += k


def _state(bt: Any, rs: Any, raw: list[float]) -> dict[str, Any]:
    """The Laplace state after construction, then the MLL and its gradient at
    the default and at the random parameters ``raw`` (the model keeps the
    state of the last evaluation), of a BoTorch (``rs is None``) or botorch_rs
    model (``rs``; ``bt`` is then the BoTorch model of the same data)."""
    m = bt if rs is None else rs
    out: dict[str, Any] = {
        "fsolve solution": torch.as_tensor(m._x0, dtype=DT),
        "utility": torch.as_tensor(m.utility, dtype=DT).detach(),
        "covar_chol": torch.as_tensor(m.covar_chol, dtype=DT).detach(),
        "covar_inv": torch.as_tensor(m.covar_inv, dtype=DT).detach(),
    }
    default = raw_params(bt)
    for k, r in enumerate((default, raw)):
        if rs is None:
            _set_bt_raw(bt, r)
            v, g = bt_mll_grad(bt)
        else:
            rs._rs.set_raw_parameters([float(x) for x in r])
            v, g = rs_mll_grad(rs)
        what = "default" if k == 0 else "random"
        out[f"MLL ({what})"] = torch.tensor([v], dtype=DT)
        out[f"MLL gradient ({what})"] = torch.tensor(g, dtype=DT)
    m.eval()
    return out


def _tbatch(m: Any, lib: Any, seed: int, kind: str, X: Any, i: int, base: Any) -> dict[str, Any]:
    """The compared quantities of t-batch ``i`` of the points ``X``, each
    t-batch evaluated on its own (BoTorch's jitter of the prior covariance
    couples the t-batches of one call, BoTorch bug 6 of
    ``docs/preference.md``): posteriors (``kind`` "q=1" or
    "joint") or the acquisition functions ("acq": EUBO, qEUBO with the base
    samples ``base``, BALD with torch's draws after ``torch.manual_seed``)."""
    Xi = X[i : i + 1] if X.dim() > 2 else X
    if kind in ("q=1", "joint"):
        p = m.posterior(Xi)
        return {
            f"posterior mean ({kind})": torch.as_tensor(p.mean, dtype=DT).detach(),
            f"posterior covariance ({kind})": torch.as_tensor(
                p.distribution.covariance_matrix, dtype=DT
            ).detach(),
        }
    out: dict[str, Any] = {}
    out["EUBO value"], out["EUBO gradient"] = _vg(lib.AnalyticExpectedUtilityOfBestOption(m), Xi)
    if lib is bt_pref:
        sb = BtSampler(torch.Size([32]), seed=seed)
        sb.base_samples = base
        acqf = bt_pref.qExpectedUtilityOfBestOption(m, sampler=sb)
    else:
        sr = RsSampler(32)
        sr.base_samples = base
        acqf = rs_pref.qExpectedUtilityOfBestOption(m, sampler=sr)
    out["qEUBO value"], out["qEUBO gradient"] = _vg(acqf, Xi)
    bald = lib.PairwiseBayesianActiveLearningByDisagreement(m, num_samples=64)
    torch.manual_seed(1000 * seed + i)
    out["BALD value"] = bald(Xi).detach()
    return out


#: The quantities computed from the posterior samples ``mean + L z`` (with
#: the Cholesky factor ``L`` of the posterior covariance of the t-batch).
ROOT_KEYS = ("qEUBO value", "qEUBO gradient", "BALD value")


def _err(key: str, a: Any, e: Any, singular_root: bool = False) -> tuple[float, float]:
    """The error of ``a`` and its tolerance: 1e-9 (values) / 1e-8
    (gradients), or :data:`ROOT_TOL` for the sample-based quantities of a
    t-batch whose posterior covariance is singular within rounding."""
    grad = "gradient" in key
    tol = GRAD_TOL if grad else VALUE_TOL
    if singular_root and key in ROOT_KEYS:
        tol = max(tol, ROOT_TOL)
    return (grad_err(a, e) if grad else rel_err(a, e)), tol


def _compare_state(
    out_rs: dict[str, Any], out_bt: dict[str, Any], K: Any, xtol: float, newton: Any
) -> None:
    """The Laplace state and the MLLs. The fsolve solutions themselves are
    not compared: fsolve stops after a step of relative size ``xtol``, and
    where ``cond(K)`` reaches 1e10 .. 1e15 rounding moves its iterates within
    that (the solutions differ by up to 3.4 ``xtol``, recorded). Instead,
    BoTorch's two Newton steps from botorch_rs's fsolve solution (``newton``,
    :func:`bt_newton_utility`) must give botorch_rs's utility to 1e-9: both
    libraries get the same input to everything downstream of the MAP solve.
    ``covar_chol`` and ``covar_inv`` are compared normwise to
    ``max(cond(K) eps, 1e-9)`` (the accuracy of a Cholesky factor and of an
    inverse of ``K``; Higham, Thm. 10.8 and Sec. 14.1), everything else to
    1e-9 (values) / 1e-8 (gradients)."""
    err = rel_err(out_rs["utility"], newton)
    check.record(
        "utility vs BoTorch's Newton steps from botorch_rs's fsolve solution", err, VALUE_TOL
    )
    assert err <= VALUE_TOL, ("utility from botorch_rs's fsolve solution", err)
    for key, e in out_bt.items():
        a = out_rs[key]
        if key in ("covar_chol", "covar_inv"):
            err = inverse_err(a, e, K)
            check.record(f"{key} / max(cond(K) eps, 1e-9)", err, 1.0)
            assert err <= 1.0, (key, err)
            continue
        err, tol = _err(key, a, e)
        if key == "fsolve solution":
            # Relative to the size of the solution (not defined for the zero
            # solution of seed 36, where BoTorch stops at denormals).
            if float(torch.as_tensor(e).abs().max()) >= torch.finfo(DT).tiny:
                check.record("fsolve solution / xtol (not compared)", err / xtol, None)
            continue
        check.record(key, err, tol)
        assert err <= tol, (key, err)


def _compare_tbatches(bt: Any, rs: Any, seed: int, kind: str, X: Any, base: Any = None) -> None:
    """Every t-batch to 1e-9 (values) / 1e-8 (gradients). A t-batch that
    differs must contain a jitter decision of BoTorch on a matrix that is
    singular within rounding (:func:`cholesky_ambiguous`); it is then compared
    with BoTorch's results for the other outcomes of its ambiguous decisions,
    at the same tolerances."""
    n = X.shape[0] if X.dim() > 2 else 1
    for i in range(n):
        out_rs = _tbatch(rs, rs_pref, seed, kind, X, i, base)
        dec = BtJitterDecisions()
        with dec.patched():
            out_bt = _tbatch(bt, bt_pref, seed, kind, X, i, base)
        errs = {k: _err(k, out_rs[k], e, dec.singular_root) for k, e in out_bt.items()}
        if all(err <= tol for err, tol in errs.values()):
            for key, (err, tol) in errs.items():
                check.record(f"{key} (singular root)" if tol == ROOT_TOL else key, err, tol)
            continue
        assert dec.ambiguous > 0, ("no ambiguous jitter decision", kind, i, errs)
        # Search the outcomes of the ambiguous decisions (flipping one can make
        # a later decision ambiguous, e.g. the posterior covariance of a pair
        # whose prior covariance got no jitter).
        matched = None
        todo = [frozenset([j]) for j in range(dec.ambiguous)]
        while todo and matched is None:
            flip = todo.pop(0)
            alt_dec = BtJitterDecisions(flip)
            with alt_dec.patched():
                alt = _tbatch(bt, bt_pref, seed, kind, X, i, base)
            alt_errs = {k: _err(k, out_rs[k], e, alt_dec.singular_root) for k, e in alt.items()}
            if all(err <= tol for err, tol in alt_errs.values()):
                matched = alt_errs
            todo += [flip | {j} for j in range(max(flip) + 1, alt_dec.ambiguous)]
        assert matched is not None, ("no outcome of BoTorch's decisions matches", kind, i, errs)
        for key, (err, tol) in matched.items():
            root = " (singular root)" if tol == ROOT_TOL else ""
            check.record(f"{key} (other outcome of an ambiguous jitter decision){root}", err, tol)
        check.record("t-batches with the other outcome of an ambiguous jitter decision", 1.0, None)


@pytest.mark.parametrize("seed", SEEDS)
def test_sweep(seed: int) -> None:
    case = random_case(seed)
    X, C = make_data(case, 1000 + seed)
    with _Guarded():
        try:
            bt, rs = build_pair(case, X, C, seed=seed)
        except Exception as e:  # BoTorch fails: botorch_rs must fail alike
            with pytest.raises(Exception) as info:
                build_rs(case, X, C, seed=seed)
            assert type(info.value).__name__ == type(e).__name__, (info.value, e)
            check.record("construction errors (same class)", 0.0, 0.0)
            return
        default = raw_params(bt)
        g = torch.Generator().manual_seed(seed)
        bounds = rs._rs.raw_parameter_bounds()
        raw = [v + 0.2 * float(torch.randn(1, generator=g, dtype=DT)) for v in default]
        raw = [max(lo + 1e-3, min(hi, v)) for v, (lo, hi) in zip(raw, bounds, strict=True)]
        K = bt.covar.detach()
        newton = bt_newton_utility(bt, rs._x0)
        _compare_state(_state(bt, rs, raw), _state(bt, None, raw), K, xtol_of(case), newton)
        X1 = random_points(bt, (4, 1), seed)
        X3 = random_points(bt, (3,), seed + 1)
        Xp = random_points(bt, (3, 2), seed + 2)
        _compare_tbatches(bt, rs, seed, "q=1", X1)
        _compare_tbatches(bt, rs, seed, "joint", X3)
        sb = BtSampler(torch.Size([32]), seed=seed)
        sb(bt.posterior(Xp))
        _compare_tbatches(bt, rs, seed, "acq", Xp, sb.base_samples)
    if seed % 4 == 0:

        def fit_twin(t: int) -> tuple[tuple[float, int], tuple[float, int]]:
            with _Guarded(), warnings.catch_warnings():
                warnings.simplefilter("ignore")
                bt_t, rs_t = build_pair(case, twin_inputs(X, t), C, seed=seed)
                return bt_fit(bt_t), rs_fit(rs_t)

        compare_twin_fits(check, f"sweep fit {seed}", fit_twin, twins=4)

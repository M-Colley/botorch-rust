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

Tolerances as in ``test_parity_10b_preference.py``: values 1e-9, gradients
1e-8. Some quantities depend on rounding in BoTorch itself: fsolve's
iterates (its solution ``_x0`` when ``cond(K)`` reaches 1e11 .. 1e15), and
the jitter that ``psd_safe_cholesky`` adds to nearly singular posterior
covariances (a linear kernel on one input, a rank-deficient joint posterior
of a pair). A quantity that misses the nominal tolerance is compared with
BoTorch's outcomes on 16 twins whose datapoints are perturbed at 1e-14 ..
1e-11: it must lie within ``10 x`` BoTorch's spread (as stage 1 of the fit
comparison). BoTorch's numerical quirks of kernels (``HalfCauchyPrior`` in
float32, the diagonal of ``sq_dist``) are fixed for the comparison as in the
6a tests.
"""

from __future__ import annotations

import random
import warnings
from collections.abc import Callable
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
    VALUE_TOL,
    Case,
    bt_fit,
    bt_mll_grad,
    build_pair,
    build_rs,
    grad_err,
    inverse_err,
    make_data,
    random_points,
    raw_params,
    rs_fit,
    rs_mll_grad,
)

from botorch_rs.acquisition import preference as rs_pref  # noqa: E402
from botorch_rs.sampling import SobolQMCNormalSampler as RsSampler  # noqa: E402
from parity_helpers import Checker, compare_twin_fits, rel_err, twin_inputs  # noqa: E402

pytestmark = [pytest.mark.equivalence, pytest.mark.filterwarnings("ignore")]

check = Checker("10b preference sweep")

SEEDS = list(range(40))
#: Twins of the rounding-dependent comparisons.
TWINS = 16


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


def _outcomes(
    bt: Any, rs: Any, seed: int, raw: list[float], points: tuple[Any, Any, Any]
) -> dict[str, Any]:
    """The compared quantities of a BoTorch (``rs is None``) or botorch_rs
    model (``rs``; ``bt`` is then the BoTorch model of the same data) at the
    inputs ``points``."""
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
    X1, X3, Xp = points
    for name, Xt in (("q=1", X1), ("joint", X3)):
        p = m.posterior(Xt)
        out[f"posterior mean ({name})"] = torch.as_tensor(p.mean, dtype=DT).detach()
        out[f"posterior covariance ({name})"] = torch.as_tensor(
            p.distribution.covariance_matrix, dtype=DT
        ).detach()
    lib = bt_pref if rs is None else rs_pref
    v, g = _vg(lib.AnalyticExpectedUtilityOfBestOption(m), Xp)
    out["EUBO value"], out["EUBO gradient"] = v, g
    sb = BtSampler(torch.Size([32]), seed=seed)
    sb(bt.posterior(Xp))
    if rs is None:
        acqf = bt_pref.qExpectedUtilityOfBestOption(m, sampler=sb)
    else:
        sr = RsSampler(32)
        sr.base_samples = sb.base_samples
        acqf = rs_pref.qExpectedUtilityOfBestOption(m, sampler=sr)
    v, g = _vg(acqf, Xp)
    out["qEUBO value"], out["qEUBO gradient"] = v, g
    bald = lib.PairwiseBayesianActiveLearningByDisagreement(m, num_samples=64)
    torch.manual_seed(seed)
    out["BALD value"] = bald(Xp).detach()
    return out


def _compare(
    out_rs: dict[str, Any],
    out_bt: dict[str, Any],
    twins: Callable[[], list[dict[str, Any]]],
    K: Any,
) -> None:
    """Every quantity to its nominal tolerance, else within 10 x BoTorch's
    spread on the twins."""
    cache: list[list[dict[str, Any]]] = []
    for key, e in out_bt.items():
        a = out_rs[key]
        grad = "gradient" in key
        tol = GRAD_TOL if grad else VALUE_TOL
        if key == "covar_inv":
            err = inverse_err(a, e, K)
            check.record("covar_inv / (cond(K) eps)", err, 1.0)
            assert err <= 1.0, err
            continue
        err = grad_err(a, e) if grad else rel_err(a, e)
        if err <= tol:
            check.record(key, err, tol)
            continue
        if not cache:
            cache.append(twins())
        spread = max((grad_err if grad else rel_err)(t[key], e) for t in cache[0])
        bound = max(tol, 10 * spread)
        check.record(
            f"{key} (rounding-dependent, error / max(tol, 10 x twin spread))", err / bound, 1.0
        )
        assert err <= bound, (key, err, spread)


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
        points = (
            random_points(bt, (4, 1), seed),
            random_points(bt, (3,), seed + 1),
            random_points(bt, (3, 2), seed + 2),
        )
        out_rs = _outcomes(bt, rs, seed, raw, points)
        out_bt = _outcomes(bt, None, seed, raw, points)

        def twins() -> list[dict[str, Any]]:
            res = []
            for t in range(1, TWINS + 1):
                bt_t, _ = build_pair(case, twin_inputs(X, t), C, seed=seed)
                res.append(_outcomes(bt_t, None, seed, raw, points))
            return res

        _compare(out_rs, out_bt, twins, K)
    if seed % 4 == 0:

        def fit_twin(t: int) -> tuple[tuple[float, int], tuple[float, int]]:
            with _Guarded(), warnings.catch_warnings():
                warnings.simplefilter("ignore")
                bt_t, rs_t = build_pair(case, twin_inputs(X, t), C, seed=seed)
                return bt_fit(bt_t), rs_fit(rs_t)

        compare_twin_fits(check, f"sweep fit {seed}", fit_twin, twins=4)

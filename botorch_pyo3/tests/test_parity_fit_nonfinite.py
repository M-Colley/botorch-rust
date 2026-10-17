"""Fits through extreme hyperparameters, compared with live BoTorch.

L-BFGS-B's line search can try extreme raw hyperparameters, e.g. a softplus
lengthscale of ``1e-293`` or exactly zero, or an outputscale of zero. There
GPyTorch's ``sq_dist`` overflows (NaN covariance, linear_operator's
``NanError``, which BoTorch's closure maps to a NaN loss) or a prior density
vanishes (loss ``+inf``); botorch_rs must hand L-BFGS-B the same values
(``botorch_core::kernels::nonfinite``), or the line search ends elsewhere.

* ``test_mll_at_extreme_hyperparameters`` (``equivalence``): BoTorch's fit
  closure and botorch_rs's MLL at random extreme raw hyperparameters of RBF
  and Matérn kernels (1 to 3 inputs, standard modules and kernel trees, also
  behind a learned ``Warp`` input transform with concentrations up to 1e12):
  the same class of loss (finite, NaN, ``+inf``), the same NaN gradients, and
  finite values within 1e-9 / 1e-8 wherever the scaled inputs are moderate
  (below that GPyTorch's values are BLAS rounding noise). Rational
  quadratic, periodic and cosine kernels: the same class of loss.
* ``test_regression_seeds``: the data seeds whose fits diverged (botorch_rs
  ended in another optimum, or raised ``lengthscales must be positive``)
  reach BoTorch's loss (twin criterion of ``test_parity``).
* ``test_fit_sweep`` (``equivalence``): 200 random data seeds over
  ``test_parity.CONFIGS``; botorch_rs's loss must not exceed BoTorch's by
  more than 1e-6, else the twin criterion decides.
* ``test_batched_elements_are_independent``: BoTorch's batched closure gives
  every element a NaN loss with zero gradients when any element's covariance
  is NaN (a BoTorch bug, see ``docs/drafts/fit-nonfinite.md``); botorch_rs
  fits the elements independently and matches BoTorch's unbatched fits.
"""

from __future__ import annotations

import math
from collections.abc import Iterator
from typing import Any

import pytest

from helpers import ParityLog, importorskip_botorch

importorskip_botorch()
torch = pytest.importorskip("torch")
np = pytest.importorskip("numpy")

import gpytorch  # noqa: E402
from botorch.models import SingleTaskGP as BtSingleTaskGP  # noqa: E402
from botorch.models.utils import gpytorch_modules as bt_modules  # noqa: E402
from botorch.optim.closures import get_loss_closure_with_grads  # noqa: E402
from botorch.optim.utils import get_parameters_and_bounds  # noqa: E402
from gpytorch.mlls import ExactMarginalLogLikelihood as BtMLL  # noqa: E402
from kernel_tree_helpers import zero_diagonal_distances  # noqa: E402
from test_parity import CONFIGS, FIT_TOL, build, compare_fits, fit_losses, make_data  # noqa: E402

from botorch_rs import kernels as rs_kernels  # noqa: E402
from botorch_rs import priors as rs_priors  # noqa: E402
from botorch_rs.models import SingleTaskGP as RsSingleTaskGP  # noqa: E402
from botorch_rs.models.utils import gpytorch_modules as rs_modules  # noqa: E402
from parity_helpers import Checker  # noqa: E402

DT = torch.float64
CHECK = Checker("fits through extreme hyperparameters")
VALUE_TOL = 1e-9
GRAD_TOL = 1e-8

pytestmark = pytest.mark.filterwarnings("ignore")


@pytest.fixture(autouse=True, scope="module")
def _single_threaded_torch() -> Iterator[None]:
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(threads)


# ---------------------------------------------------------------------------
# The MLL at extreme hyperparameters
# ---------------------------------------------------------------------------


def _kernel_pair(name: str, d: int) -> tuple[Any, Any]:
    G, R = gpytorch.kernels, rs_kernels
    gamma = (gpytorch.priors.GammaPrior(3.0, 6.0), rs_priors.GammaPrior(3.0, 6.0))
    if name in ("rbf", "matern12", "matern32", "matern52"):
        nu = {"matern12": 0.5, "matern32": 1.5, "matern52": 2.5}.get(name)
        if nu is None:
            b = G.RBFKernel(ard_num_dims=d, lengthscale_prior=gamma[0])
            r = R.RBFKernel(ard_num_dims=d, lengthscale_prior=gamma[1])
        else:
            b = G.MaternKernel(nu=nu, ard_num_dims=d, lengthscale_prior=gamma[0])
            r = R.MaternKernel(nu=nu, ard_num_dims=d, lengthscale_prior=gamma[1])
        return (
            G.ScaleKernel(b, outputscale_prior=gpytorch.priors.GammaPrior(2.0, 0.15)),
            R.ScaleKernel(r, outputscale_prior=rs_priors.GammaPrior(2.0, 0.15)),
        )
    if name == "tree_shared":  # one lengthscale: RBFCovariance / MaternCovariance
        return (
            G.ScaleKernel(G.AdditiveKernel(G.RBFKernel(), G.MaternKernel(nu=2.5))),
            R.ScaleKernel(R.AdditiveKernel(R.RBFKernel(), R.MaternKernel(nu=2.5))),
        )
    if name == "tree_ard":
        return (
            G.ScaleKernel(
                G.ProductKernel(G.RBFKernel(ard_num_dims=d), G.MaternKernel(nu=1.5, ard_num_dims=d))
            ),
            R.ScaleKernel(
                R.ProductKernel(R.RBFKernel(ard_num_dims=d), R.MaternKernel(nu=1.5, ard_num_dims=d))
            ),
        )
    if name == "rq":
        return G.ScaleKernel(G.RQKernel(ard_num_dims=d)), R.ScaleKernel(R.RQKernel(ard_num_dims=d))
    if name == "periodic":
        return (
            G.ScaleKernel(G.PeriodicKernel(ard_num_dims=d)),
            R.ScaleKernel(R.PeriodicKernel(ard_num_dims=d)),
        )
    if name == "cosine":
        return G.ScaleKernel(G.CosineKernel()), R.ScaleKernel(R.CosineKernel())
    raise ValueError(name)


def _warp_pair(name: str, X: Any, Y: Any) -> tuple[Any, Any]:
    """`SingleTaskGP`s with `Warp(concentration priors LogNormal(0, 0.75^½))`
    on all inputs, the legacy Matérn-5/2 module or a `ScaleKernel(RBF)`
    tree (both with softplus constraints)."""
    from botorch.models.transforms.input import Warp as BtWarp

    from botorch_rs.models.transforms.input import Warp as RsWarp

    d = X.shape[-1]
    sd = 0.75**0.5
    bt_w = BtWarp(
        d=d,
        indices=list(range(d)),
        concentration0_prior=gpytorch.priors.LogNormalPrior(0.0, sd),
        concentration1_prior=gpytorch.priors.LogNormalPrior(0.0, sd),
    )
    rs_w = RsWarp(
        d=d,
        indices=list(range(d)),
        concentration0_prior=rs_priors.LogNormalPrior(0.0, sd),
        concentration1_prior=rs_priors.LogNormalPrior(0.0, sd),
    )
    if name == "warp_matern52":
        bk = bt_modules.get_matern_kernel_with_gamma_prior(ard_num_dims=d)
        rk = rs_modules.get_matern_kernel_with_gamma_prior(ard_num_dims=d)
    else:
        bk, rk = _kernel_pair("rbf", d)
    bt = BtSingleTaskGP(X, Y, input_transform=bt_w, covar_module=bk)
    rs = RsSingleTaskGP(X, Y, input_transform=rs_w, covar_module=rk)
    return bt, rs


EXTREME_MODELS = [(m, d) for m in ("rbf", "matern12", "matern32", "matern52") for d in (1, 2, 3)]
EXTREME_MODELS += [("tree_shared", 1), ("tree_shared", 2), ("tree_ard", 2)]
#: Models with a learned `Warp` input transform (concentrations fitted with
#: the GP): the legacy Matérn module and an RBF kernel tree.
EXTREME_MODELS += [("warp_matern52", 1), ("warp_matern52", 2), ("warp_rbf", 2)]
#: Kernels whose gradients at extreme points are not compared (GPyTorch's
#: autograd overflows in other places, see the draft).
VALUE_CLASS_ONLY = [("rq", 2), ("periodic", 2), ("cosine", 2)]


def _kind(v: float) -> str:
    return "nan" if math.isnan(v) else ("+inf" if v == math.inf else "finite")


def _bt_closure(bt: Any) -> tuple[Any, dict[str, Any]]:
    mll = BtMLL(bt.likelihood, bt)
    mll.train()
    params = {k: p for k, p in get_parameters_and_bounds(mll)[0].items() if p.requires_grad}
    return get_loss_closure_with_grads(mll, parameters=params), params


def _bt_eval(closure: Any, params: dict[str, Any], raw: list[float]) -> tuple[Any, Any]:
    i = 0
    with torch.no_grad():
        for p in params.values():
            p.copy_(torch.tensor(raw[i : i + p.numel()], dtype=DT).reshape(p.shape))
            i += p.numel()
    try:
        loss, grads = closure()
    except Exception as e:  # BoTorch's _handle_numerical_errors
        if type(e).__name__ != "NanError":
            return f"raises {type(e).__name__}", None
        return math.nan, np.full(len(raw), np.nan)
    return float(loss), torch.cat([g.reshape(-1) for g in grads]).numpy()


def _rs_eval(rs: Any, raw: list[float]) -> tuple[Any, Any]:
    try:
        (values, _), grad = rs._rs.mll_value_and_grad(raw)
    except Exception as e:
        if type(e).__name__ != "NanError":
            return f"raises {type(e).__name__}", None
        return math.nan, np.full(len(raw), np.nan)
    return -sum(values), -np.asarray(grad)


def _random_raw(
    params: dict[str, Any], g: Any, ls_floor: float = -900.0
) -> tuple[list[float], bool]:
    """Raw values: noise and constant moderate, softplus parameters either
    moderate or extreme (-900 .. -30, lengthscales from `ls_floor`), `Warp`
    concentrations up to 1e12; whether every softplus value is moderate
    (lengthscales and periods at least ~1e-3)."""
    raw, moderate = [], True
    for name, p in params.items():
        for _ in range(p.numel()):
            u = torch.rand(2, generator=g, dtype=DT).tolist()
            if "raw_noise" in name:
                raw.append(1e-4 + 0.5 * u[0])
            elif "raw_constant" in name:
                raw.append(-2 + 4 * u[0])
            elif "concentration" in name:
                # Bounded below by 1e-4 (no transform); up to 1e12.
                if u[1] < 0.5:
                    raw.append(10 ** (-1 + 2 * u[0]))
                else:
                    raw.append(10 ** (-4 + 16 * u[0]))
                    moderate = False
            elif u[1] < 0.5:
                raw.append(-3 + 6 * u[0])
                moderate &= raw[-1] > -6.9 or "outputscale" in name
            else:
                lo = ls_floor if "lengthscale" in name else -900.0
                raw.append(lo + (-30 - lo) * u[0])
                moderate &= "outputscale" in name
    return raw, moderate


@pytest.mark.equivalence
@pytest.mark.parametrize(
    "name,d",
    EXTREME_MODELS + VALUE_CLASS_ONLY,
    ids=[f"{m}-{d}" for m, d in EXTREME_MODELS + VALUE_CLASS_ONLY],
)
def test_mll_at_extreme_hyperparameters(name: str, d: int) -> None:
    X, Y, _ = make_data(10, d, seed=5)
    if name.startswith("warp_"):
        bt, rs = _warp_pair(name, X, Y)
    else:
        bk, rk = _kernel_pair(name, d)
        bt = BtSingleTaskGP(X, Y, covar_module=bk)
        rs = RsSingleTaskGP(X, Y, covar_module=rk)
    closure, params = _bt_closure(bt)
    assert [n.removeprefix("model.") for n in params] == [n for n, _ in rs.named_parameters()]
    g = torch.Generator().manual_seed(1000 + 17 * d + len(name))
    # `Warp` models: loss classes and finite values; their gradient NaNs at
    # extreme lengthscales follow last-bit differences of the warped inputs.
    warp = name.startswith("warp_")
    grads_too = (name, d) not in VALUE_CLASS_ONLY and not warp
    counts: dict[str, int] = {}
    worst_v = worst_g = 0.0
    for _ in range(60):
        # Inputs from a learned transform differ from BoTorch's in the last
        # bits (a saturated warp gives a constant column in one library, not
        # quite in the other); divided by lengthscales below ~1e-170 such
        # differences alone overflow, so the lengthscales of `Warp` models
        # stay above 1e-169, where the squares of non-constant scaled inputs
        # still overflow (deviation 2 of the draft).
        floor = -388.0 if warp else -900.0
        raw, moderate = _random_raw(params, g, floor)
        bv, bg = _bt_eval(closure, params, raw)
        rv, rg = _rs_eval(rs, raw)
        if isinstance(bv, str) or isinstance(rv, str):
            # Not-PSD covariances (BoTorch raises NotPSDError): which jitter
            # level suffices depends on rounding; both may raise.
            assert "NotPSD" in str(bv) or "NotPSD" in str(rv), (bv, rv, raw)
            continue
        kind = _kind(bv)
        counts[kind] = counts.get(kind, 0) + 1
        assert _kind(rv) == kind, (bv, rv, raw)
        if grads_too:
            assert (np.isnan(bg) == np.isnan(rg)).all(), (bg, rg, raw)
        if kind == "finite" and moderate and (grads_too or warp):
            if name == "matern12" and d > 1:
                # GPyTorch's training diagonal of Matérn-1/2 kernels with
                # several lengthscales is sqrt of rounding noise (a quirk the
                # 6a tests fix the same way); compare without it.
                with zero_diagonal_distances():
                    bv, bg = _bt_eval(closure, params, raw)
            worst_v = max(worst_v, abs(rv - bv) / max(1.0, abs(bv)))
            fin = np.isfinite(bg)
            scale = max(1.0, float(np.abs(bg[fin]).max())) if fin.any() else 1.0
            worst_g = max(worst_g, float(np.abs(rg[fin] - bg[fin]).max(initial=0.0)) / scale)
    ParityLog.record(
        f"extreme MLL: points by loss class ({name}, d={d})",
        float(sum(counts.values())),
        section=CHECK.section,
    )
    CHECK.check_bound("extreme MLL: finite values at moderate points", worst_v, VALUE_TOL)
    CHECK.check_bound("extreme MLL: finite gradients at moderate points", worst_g, GRAD_TOL)
    assert counts.get("nan", 0) > 0 or warp, counts


def test_user_specified_zero_hyperparameters_raise() -> None:
    """A zero lengthscale or outputscale set by the user is rejected by the
    RBF / Matérn modules as before (kernel trees accept any raw values, like
    GPyTorch); only the MLL (and so the fit) evaluates them, as GPyTorch
    does: a zero lengthscale gives NaN (``NanError``), a zero outputscale a
    loss of ``+inf`` under its ``GammaPrior``."""
    X, Y, Yvar = make_data(12, 2, seed=104)
    bt, rs = build("legacy_gamma", X, Y, Yvar)
    with pytest.raises(ValueError, match="positive"):
        rs.initialize(**{"covar_module.base_kernel.raw_lengthscale": torch.tensor([[-800.0, 0.0]])})
    with pytest.raises(ValueError, match="positive"):
        rs.initialize(**{"covar_module.raw_outputscale": torch.tensor(-800.0)})
    tree_bt, tree_rs = _kernel_pair("rbf", 2)
    pairs = [
        (bt, rs),
        (BtSingleTaskGP(X, Y, covar_module=tree_bt), RsSingleTaskGP(X, Y, covar_module=tree_rs)),
    ]
    for bt_model, rs_model in pairs:
        closure, params = _bt_closure(bt_model)
        raw = [v for p in params.values() for v in p.detach().reshape(-1).tolist()]
        # noise, constant, raw outputscale, raw lengthscales
        for index, kind in ((4, "nan"), (2, "+inf")):
            bad = list(raw)
            bad[index] = -800.0
            bv, bg = _bt_eval(closure, params, bad)
            rv, rg = _rs_eval(rs_model, bad)
            assert _kind(bv) == _kind(rv) == kind, (bv, rv)
            assert (np.isnan(bg) == np.isnan(rg)).all(), (bg, rg)


# ---------------------------------------------------------------------------
# Fits
# ---------------------------------------------------------------------------

#: Seeds (of test_parity's data) whose fits diverged before: an overflowing
#: or zero lengthscale where GPyTorch's covariance is NaN, and an
#: outputscale of zero (loss +inf under its GammaPrior).
REGRESSION = [
    ("custom_scale_rbf", 514),
    ("legacy_gamma", 104),
    ("legacy_gamma", 203),
    ("legacy_gamma", 417),
    ("legacy_gamma", 665),
    ("legacy_gamma", 667),
    ("legacy_gamma", 753),
]


@pytest.mark.parametrize("config,seed", REGRESSION, ids=[f"{c}-{s}" for c, s in REGRESSION])
def test_regression_seeds(config: str, seed: int) -> None:
    _, d, n, xrange = next(c for c in CONFIGS if c[0] == config)
    X, Y, Yvar = make_data(n, d, seed=seed, low=xrange[0], high=xrange[1])
    gap, _ = compare_fits(f"regression {config}", config, X, Y, Yvar, twins=4)
    CHECK.record("regression seeds: botorch_rs's loss minus BoTorch's", abs(gap), FIT_TOL)


N_SWEEP = 200


@pytest.mark.equivalence
def test_fit_sweep() -> None:
    """Random data seeds over all configurations: botorch_rs's fit must not
    end above BoTorch's by more than 1e-6, unless BoTorch's own outcome is
    ambiguous (then the twin criterion of ``test_parity`` decides)."""
    worst, ambiguous = 0.0, 0
    for i in range(N_SWEEP):
        config, d, n, xrange = CONFIGS[i % len(CONFIGS)]
        X, Y, Yvar = make_data(n, d, seed=20_000 + i, low=xrange[0], high=xrange[1])
        (bt_loss, bt_a), (rs_loss, rs_a) = fit_losses(config, X, Y, Yvar)
        if bt_a == rs_a == 1 and rs_loss <= bt_loss + FIT_TOL:
            worst = max(worst, rs_loss - bt_loss)
            continue
        ambiguous += 1
        compare_fits(f"sweep {config}", config, X, Y, Yvar)
    CHECK.record("sweep: botorch_rs's loss above BoTorch's (direct)", max(worst, 0.0), FIT_TOL)
    ParityLog.record("sweep: seeds decided by twins", float(ambiguous), section=CHECK.section)


# ---------------------------------------------------------------------------
# Batched fits
# ---------------------------------------------------------------------------


def _batched_pair(X: Any, Y: Any) -> tuple[Any, Any]:
    kw = {"batch_shape": torch.Size([Y.shape[-1]])} if Y.shape[-1] > 1 else {}
    bt = BtSingleTaskGP(
        X,
        Y,
        covar_module=bt_modules.get_matern_kernel_with_gamma_prior(ard_num_dims=X.shape[-1], **kw),
        likelihood=bt_modules.get_gaussian_likelihood_with_gamma_prior(**kw),
    )
    rs = RsSingleTaskGP(
        X,
        Y,
        covar_module=rs_modules.get_matern_kernel_with_gamma_prior(ard_num_dims=X.shape[-1], **kw),
        likelihood=rs_modules.get_gaussian_likelihood_with_gamma_prior(**kw),
    )
    return bt, rs


def _bt_losses(bt: Any) -> list[float]:
    from botorch.fit import fit_gpytorch_mll

    fit_gpytorch_mll(BtMLL(bt.likelihood, bt))
    bt.train()
    with torch.no_grad():
        out = -BtMLL(bt.likelihood, bt)(bt(*bt.train_inputs), bt.train_targets)
    bt.eval()
    return out.reshape(-1).tolist()


def test_batched_elements_are_independent() -> None:
    """Output 0 (legacy_gamma, seed 104) visits a NaN covariance; in
    BoTorch's batched fit, the joint NaN evaluation makes output 1 stop
    "converged" with a NaN loss at a poor point (4.18 instead of 1.46).
    botorch_rs's elements match BoTorch's separate (unbatched) fits."""
    X, Y1, _ = make_data(12, 2, seed=104)
    g = torch.Generator().manual_seed(2)
    Y2 = torch.sin(3 * X[:, 1:]) * (1 + torch.rand(1, generator=g, dtype=DT))
    Y2 = Y2 + 0.1 * torch.randn(12, 1, generator=g, dtype=DT)
    Y = torch.cat([Y1, Y2], -1)
    bt, rs = _batched_pair(X, Y)
    batched = _bt_losses(bt)
    single = [_bt_losses(_batched_pair(X, Y[:, j : j + 1])[0])[0] for j in range(2)]
    from botorch_rs.fit import fit_gpytorch_mll as rs_fit
    from botorch_rs.mlls import ExactMarginalLogLikelihood as RsMLL

    rs_fit(RsMLL(rs.likelihood, rs))
    (values, _), _ = rs._rs.mll_value_and_grad(None)
    rs_losses = [-v for v in values]
    ParityLog.record(
        "batched: BoTorch's batched minus unbatched loss (bug)",
        max(abs(a - b) for a, b in zip(batched, single, strict=True)),
        section=CHECK.section,
    )
    for j in range(2):
        CHECK.check_bound(
            "batched: element loss above BoTorch's unbatched fit", rs_losses[j] - single[j], FIT_TOL
        )

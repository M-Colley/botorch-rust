"""Parity with BoTorch for multi-output and batched ``SingleTaskGP`` models,
``ModelListGP``, ``ScalarizedPosteriorTransform`` and the analytic constrained
expected improvement, through the Python API.

BoTorch and botorch_rs are run side by side on identical data and
hyperparameters over a seeded grid of configurations (numbers of outputs,
batch shapes, input dimensions, training sizes, transforms, noise models and
kernels): default initializations, per-element MLL values and gradients,
posteriors (``q = 1`` batches, joint posteriors with and without observation
noise, batched and broadcast inputs, posterior transforms, samples from
explicit base samples), fits, and acquisition values and gradients (analytic,
MC with identical base samples, qLogNEI, ``prune_inferior_points``, LogCEI and
CEI). The worst error of every quantity is printed at the end of the session.

Two BoTorch code paths fail with linear_operator 0.6.1 for ``ModelListGP``
models (``BlockDiagLinearOperator`` slicing when a sub-model has several
outputs, and block indexing in ``scalarize_posterior`` for ``q > 1``); the
references for those cases use the same BoTorch code with dense covariances
(see ``dense_model_list``), and ``test_model_list_reference`` documents the
difference.

Skipped unless BoTorch is importable (required in CI's parity job).
"""

from __future__ import annotations

import contextlib
import math
from collections.abc import Iterator
from typing import Any

import pytest

from helpers import ParityLog, importorskip_botorch

importorskip_botorch()
torch = pytest.importorskip("torch")

import gpytorch  # noqa: E402
from botorch.acquisition import analytic as bt_analytic  # noqa: E402
from botorch.acquisition import logei as bt_logei  # noqa: E402
from botorch.acquisition import monte_carlo as bt_mc  # noqa: E402
from botorch.acquisition.objective import (  # noqa: E402
    ScalarizedPosteriorTransform as BtTransform,
)
from botorch.acquisition.utils import prune_inferior_points as bt_prune  # noqa: E402
from botorch.fit import fit_gpytorch_mll as bt_fit  # noqa: E402
from botorch.models import ModelListGP as BtModelListGP  # noqa: E402
from botorch.models import SingleTaskGP as BtSingleTaskGP  # noqa: E402
from botorch.models.transforms.input import Normalize as BtNormalize  # noqa: E402
from botorch.models.utils import gpytorch_modules as bt_modules  # noqa: E402
from botorch.sampling.normal import SobolQMCNormalSampler as BtSampler  # noqa: E402
from gpytorch.mlls import ExactMarginalLogLikelihood as BtMLL  # noqa: E402
from gpytorch.mlls import SumMarginalLogLikelihood as BtSumMLL  # noqa: E402

from botorch_rs import kernels as rs_kernels  # noqa: E402
from botorch_rs import priors as rs_priors  # noqa: E402
from botorch_rs.acquisition import analytic as rs_analytic  # noqa: E402
from botorch_rs.acquisition import logei as rs_logei  # noqa: E402
from botorch_rs.acquisition import monte_carlo as rs_mc  # noqa: E402
from botorch_rs.acquisition.objective import (  # noqa: E402
    ScalarizedPosteriorTransform as RsTransform,
)
from botorch_rs.acquisition.utils import prune_inferior_points as rs_prune  # noqa: E402
from botorch_rs.fit import fit_gpytorch_mll as rs_fit  # noqa: E402
from botorch_rs.mlls import ExactMarginalLogLikelihood as RsMLL  # noqa: E402
from botorch_rs.mlls import SumMarginalLogLikelihood as RsSumMLL  # noqa: E402
from botorch_rs.models import ModelListGP as RsModelListGP  # noqa: E402
from botorch_rs.models import SingleTaskGP as RsSingleTaskGP  # noqa: E402
from botorch_rs.models.transforms import Normalize as RsNormalize  # noqa: E402
from botorch_rs.models.utils import gpytorch_modules as rs_modules  # noqa: E402
from botorch_rs.sampling import SobolQMCNormalSampler as RsSampler  # noqa: E402

DT = torch.float64
RTOL = 1e-9
GRAD_RTOL = 1e-8

pytestmark = pytest.mark.filterwarnings("ignore")


@pytest.fixture(autouse=True, scope="module")
def _single_threaded_torch() -> Iterator[None]:
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(threads)


# ---------------------------------------------------------------------------
# Error measures
# ---------------------------------------------------------------------------


def _t(x: Any) -> Any:
    return torch.as_tensor(x, dtype=DT).detach()


def rel_err(a: Any, e: Any, atol: float = 0.0) -> float:
    """Mixed elementwise / normwise relative error (``|a - e| / (|e| + max|e|)``);
    entries that agree to ``atol`` count as exact."""
    a, e = _t(a), _t(e)
    assert a.shape == e.shape, (a.shape, e.shape)
    if e.numel() == 0:
        return 0.0
    diff = (a - e).abs()
    diff = torch.where(diff <= atol, torch.zeros_like(diff), diff)
    scale = e.abs().max().item()
    floor = scale if scale > 0 else 1.0
    return (diff / (e.abs() + floor)).max().item()


def log_err(a: Any, e: Any) -> float:
    """Error of log-space values: absolute below 1, relative above."""
    a, e = _t(a), _t(e)
    assert a.shape == e.shape, (a.shape, e.shape)
    return ((a - e).abs() / e.abs().clamp_min(1.0)).max().item()


def check(quantity: str, a: Any, e: Any, tol: float = RTOL, atol: float = 0.0) -> None:
    err = rel_err(a, e, atol)
    ParityLog.record(quantity, err)
    assert err <= tol, (quantity, err)


def check_log(quantity: str, a: Any, e: Any, tol: float = RTOL) -> None:
    err = log_err(a, e)
    ParityLog.record(quantity, err)
    assert err <= tol, (quantity, err)


# ---------------------------------------------------------------------------
# Configurations
# ---------------------------------------------------------------------------
# name, d, n, m, batch_shape, options (normalize: "shared" | "per_batch" |
# "explicit"; outcome: "none"; yvar; covar: "matern" | "legacy" | "custom_scale")
CONFIGS: list[tuple[str, int, int, int, tuple[int, ...], dict[str, Any]]] = [
    ("m2", 2, 12, 2, (), {}),
    ("m3_normalize", 3, 15, 3, (), {"normalize": "shared", "range": (-2.0, 3.0)}),
    ("m2_fixed_noise", 2, 10, 2, (), {"yvar": True}),
    (
        "m2_no_outcome",
        4,
        9,
        2,
        (),
        {"outcome": "none", "normalize": "explicit", "range": (-1.0, 1.0)},
    ),
    ("m3_legacy", 2, 11, 3, (), {"covar": "legacy"}),
    ("m2_matern", 3, 10, 2, (), {"covar": "matern"}),
    ("b3_m1", 2, 10, 1, (3,), {}),
    ("b2_m2_normalize", 2, 8, 2, (2,), {"normalize": "shared", "range": (1.0, 4.0)}),
    ("b2x2_m1_per_batch", 3, 9, 1, (2, 2), {"normalize": "per_batch", "covar": "matern"}),
    ("b2_m2_fixed_noise", 2, 7, 2, (2,), {"yvar": True}),
    ("b2_m3_custom_scale", 2, 9, 3, (2,), {"covar": "custom_scale"}),
]
UNBATCHED = [c for c in CONFIGS if not c[4]]
IDS = [c[0] for c in CONFIGS]


def make_data(cfg: tuple, seed: int) -> tuple[Any, Any, Any]:
    name, d, n, m, batch, opts = cfg
    lo, hi = opts.get("range", (0.0, 1.0))
    g = torch.Generator().manual_seed(seed)
    X = lo + (hi - lo) * torch.rand(*batch, n, d, generator=g, dtype=DT)
    U = (X - lo) / (hi - lo)
    cols = []
    for t in range(m):
        f = torch.sin((3.0 + t) * U[..., 0]) + (1.0 + 0.5 * t) * U[..., 1:].sum(-1) ** 2
        f = f + 0.05 * torch.randn(*batch, n, generator=g, dtype=DT)
        cols.append((1.0 + t) + (0.5 + t) * f)
    Y = torch.stack(cols, dim=-1)
    Yvar = 0.01 + 0.02 * torch.rand(*batch, n, m, generator=g, dtype=DT)
    return X, Y, Yvar


def build(cfg: tuple, X: Any, Y: Any, Yvar: Any) -> tuple[Any, Any]:
    """The same model in BoTorch and botorch_rs."""
    name, d, n, m, batch, opts = cfg
    aug = torch.Size(batch + ((m,) if m > 1 else ()))
    bt_kw: dict[str, Any] = {}
    rs_kw: dict[str, Any] = {}
    lo, hi = opts.get("range", (0.0, 1.0))
    normalize = opts.get("normalize")
    if normalize == "shared":
        bt_kw["input_transform"] = BtNormalize(d=d)
        rs_kw["input_transform"] = RsNormalize(d=d)
    elif normalize == "per_batch":
        bt_kw["input_transform"] = BtNormalize(d=d, batch_shape=torch.Size(batch))
        rs_kw["input_transform"] = RsNormalize(d=d, batch_shape=batch)
    elif normalize == "explicit":
        bounds = torch.tensor([[lo - 0.5] * d, [hi + 1.0] * d], dtype=DT)
        bt_kw["input_transform"] = BtNormalize(d=d, bounds=bounds)
        rs_kw["input_transform"] = RsNormalize(d=d, bounds=bounds)
    if opts.get("outcome") == "none":
        bt_kw["outcome_transform"] = None
        rs_kw["outcome_transform"] = None
    covar = opts.get("covar")
    if covar == "matern":
        bt_kw["covar_module"] = bt_modules.get_covar_module_with_dim_scaled_prior(
            ard_num_dims=d, use_rbf_kernel=False, batch_shape=aug
        )
        rs_kw["covar_module"] = rs_modules.get_covar_module_with_dim_scaled_prior(
            ard_num_dims=d, use_rbf_kernel=False, batch_shape=aug
        )
    elif covar == "legacy":
        bt_kw["covar_module"] = bt_modules.get_matern_kernel_with_gamma_prior(d, batch_shape=aug)
        bt_kw["likelihood"] = bt_modules.get_gaussian_likelihood_with_gamma_prior(batch_shape=aug)
        rs_kw["covar_module"] = rs_modules.get_matern_kernel_with_gamma_prior(d, batch_shape=aug)
        rs_kw["likelihood"] = rs_modules.get_gaussian_likelihood_with_gamma_prior(batch_shape=aug)
    elif covar == "custom_scale":
        bt_kw["covar_module"] = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(
                ard_num_dims=d,
                batch_shape=aug,
                lengthscale_prior=gpytorch.priors.GammaPrior(3.0, 6.0),
            ),
            batch_shape=aug,
            outputscale_prior=gpytorch.priors.GammaPrior(2.0, 0.15),
        )
        rs_kw["covar_module"] = rs_kernels.ScaleKernel(
            rs_kernels.RBFKernel(
                ard_num_dims=d,
                batch_shape=aug,
                lengthscale_prior=rs_priors.GammaPrior(3.0, 6.0),
            ),
            batch_shape=aug,
            outputscale_prior=rs_priors.GammaPrior(2.0, 0.15),
        )
    yvar = Yvar if opts.get("yvar") else None
    bt = BtSingleTaskGP(X, Y, train_Yvar=yvar, **bt_kw)
    rs = RsSingleTaskGP(X, Y, train_Yvar=yvar, **rs_kw)
    return bt, rs


def set_hyperparameters(bt: Any) -> None:
    """Explicit hyperparameters that differ between the elements."""
    aug = bt._aug_batch_shape
    e = torch.arange(max(aug.numel(), 1), dtype=DT).reshape(aug)
    d = bt.train_inputs[0].shape[-1]
    base = torch.linspace(0.3, 0.9, d, dtype=DT)
    ls = (base * (1 + 0.2 * e.unsqueeze(-1))).unsqueeze(-2)
    kwargs: dict[str, Any] = {"mean_module.constant": 0.15 - 0.1 * e}
    if isinstance(bt.covar_module, gpytorch.kernels.ScaleKernel):
        kwargs["covar_module.outputscale"] = 1.7 + 0.25 * e
        kwargs["covar_module.base_kernel.lengthscale"] = ls
    else:
        kwargs["covar_module.lengthscale"] = ls
    if not isinstance(bt.likelihood, gpytorch.likelihoods.FixedNoiseGaussianLikelihood):
        kwargs["likelihood.noise"] = (0.05 * (1 + 0.5 * e)).unsqueeze(-1)
    bt.initialize(**kwargs)


def copy_hyperparameters(bt: Any, rs: Any) -> None:
    rs.initialize(**{name: value.detach() for name, value in bt.named_parameters()})


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


@pytest.fixture(scope="module", params=CONFIGS, ids=IDS)
def pair(request: pytest.FixtureRequest) -> tuple[tuple, Any, Any]:
    """A BoTorch / botorch_rs model pair with explicit hyperparameters."""
    cfg = request.param
    X, Y, Yvar = make_data(cfg, seed=101 + IDS.index(cfg[0]))
    bt, rs = build(cfg, X, Y, Yvar)
    set_hyperparameters(bt)
    copy_hyperparameters(bt, rs)
    bt.eval()
    return cfg, bt, rs


def points_for(cfg: tuple, shape: tuple[int, ...], seed: int) -> Any:
    """Random points of the given shape in the input range of a configuration."""
    lo, hi = cfg[5].get("range", (0.0, 1.0))
    g = torch.Generator().manual_seed(seed)
    return lo + (hi - lo) * torch.rand(*shape, generator=g, dtype=DT)


# ---------------------------------------------------------------------------
# SingleTaskGP: initialization, MLL, posteriors, fitting
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("cfg", CONFIGS, ids=IDS)
def test_default_initialization_and_mll(cfg: tuple) -> None:
    X, Y, Yvar = make_data(cfg, seed=7 + IDS.index(cfg[0]))
    bt, rs = build(cfg, X, Y, Yvar)
    bt_params = {k: v.detach() for k, v in bt.named_parameters()}
    rs_params = dict(rs.named_parameters())
    assert list(bt_params) == list(rs_params)
    for name in bt_params:
        assert tuple(rs_params[name].shape) == tuple(bt_params[name].shape), name
        check("default raw parameters", rs_params[name], bt_params[name], tol=1e-12)
    assert rs.num_outputs == bt.num_outputs and rs.batch_shape == tuple(bt.batch_shape)
    assert rs._aug_batch_shape == tuple(bt._aug_batch_shape)
    check("train_targets", rs.train_targets, bt.train_targets, tol=1e-12)
    assert tuple(rs.train_inputs[0].shape) == tuple(bt.train_inputs[0].shape)
    check("default MLL values", RsMLL(rs.likelihood, rs)(), bt_mll_values(bt))
    if hasattr(bt, "outcome_transform"):
        check("Standardize means", rs.outcome_transform.means, bt.outcome_transform.means)
        check("Standardize stdvs", rs.outcome_transform.stdvs, bt.outcome_transform.stdvs)
    if hasattr(bt, "input_transform"):
        check("Normalize bounds", rs.input_transform.bounds, bt.input_transform.bounds)


def test_mll_values_and_gradients(pair: tuple) -> None:
    _, bt, rs = pair
    values, grad = bt_mll_value_and_grad(bt)
    rs_values, rs_grad = RsMLL(rs.likelihood, rs).value_and_grad()
    check("MLL values", rs_values, values.reshape(-1).tolist() if values.dim() else values)
    check("MLL gradient", rs_grad, grad, tol=1e-9, atol=1e-14)
    check("MLL values (call)", RsMLL(rs.likelihood, rs)(), values)


def _compare_posteriors(label: str, pb: Any, pr: Any) -> None:
    assert tuple(pr.mean.shape) == tuple(pb.mean.shape), (label, pr.mean.shape, pb.mean.shape)
    check(f"{label} mean", pr.mean, pb.mean)
    check(f"{label} variance", pr.variance, pb.variance)
    cb = pb.distribution.covariance_matrix
    cr = pr.distribution.covariance_matrix
    check(f"{label} covariance", cr, cb)
    assert type(pr.distribution).__name__ == type(pb.distribution).__name__


def test_posteriors(pair: tuple) -> None:
    cfg, bt, rs = pair
    d, batch = cfg[1], cfg[4]
    shapes: list[tuple[int, ...]] = [(3, d)]
    if batch:
        shapes += [(*batch, 3, d), (2, *([1] * len(batch)), 3, d)]
    else:
        shapes += [(6, 1, d), (2, 3, d)]
    for i, shape in enumerate(shapes):
        X = points_for(cfg, shape, seed=11 + i)
        for noise in (False, True):
            with torch.no_grad():
                pb = bt.posterior(X, observation_noise=noise)
            pr = rs.posterior(X, observation_noise=noise)
            label = "posterior (obs noise)" if noise else "posterior"
            _compare_posteriors(label, pb, pr)


def test_posterior_output_indices(pair: tuple) -> None:
    """``output_indices`` select outputs (BoTorch's multi-output SingleTaskGP
    fails for every proper subset, so the reference is BoTorch's full
    posterior restricted to the outputs)."""
    cfg, bt, rs = pair
    m = cfg[3]
    if m == 1:
        return
    X = points_for(cfg, (2, cfg[1]), seed=3)
    with torch.no_grad():
        full = bt.posterior(X)
    q = 2
    for idx in ([m - 1], [m - 1, 0]):
        pr = rs.posterior(X, output_indices=idx)
        check("output_indices mean", pr.mean, full.mean[..., idx])
        rows = [t * q + p for t in idx for p in range(q)]
        cov = full.distribution.covariance_matrix[..., rows, :][..., rows]
        check("output_indices covariance", pr.distribution.covariance_matrix, cov)


def weights_for(m: int) -> Any:
    """Scalarization weights for ``m`` outputs."""
    if m == 1:
        return torch.tensor([-1.5], dtype=DT)
    return torch.tensor([0.6, -0.4, 0.25, 0.8][:m] + [0.1] * max(m - 4, 0), dtype=DT)


def test_posterior_transform(pair: tuple) -> None:
    cfg, bt, rs = pair
    d, batch, m = cfg[1], cfg[4], cfg[3]
    w = weights_for(m)
    shapes: list[tuple[int, ...]] = [(1, d), (3, d)]
    if batch:
        shapes.append((*batch, 2, d))
    for i, shape in enumerate(shapes):
        X = points_for(cfg, shape, seed=21 + i)
        for noise in (False, True):
            with torch.no_grad():
                pb = bt.posterior(
                    X, observation_noise=noise, posterior_transform=BtTransform(w, 0.3)
                )
            pr = rs.posterior(X, observation_noise=noise, posterior_transform=RsTransform(w, 0.3))
            _compare_posteriors("scalarized posterior", pb, pr)
    # The transform applied to an existing posterior.
    X = points_for(cfg, (3, d), seed=29)
    with torch.no_grad():
        pb = BtTransform(w, -0.2)(bt.posterior(X))
    pr = RsTransform(w, -0.2)(rs.posterior(X))
    _compare_posteriors("scalarized posterior", pb, pr)


def test_samples_from_base_samples(pair: tuple) -> None:
    """Explicit base samples give BoTorch's samples (BoTorch's mapping of
    ``S x q x m`` base samples to the outputs of a multi-output posterior)."""
    cfg, bt, rs = pair
    X = points_for(cfg, (3, cfg[1]), seed=31)
    with torch.no_grad():
        pb = bt.posterior(X)
    pr = rs.posterior(X)
    g = torch.Generator().manual_seed(5)
    Z = torch.randn(4, *pb.base_sample_shape, generator=g, dtype=DT)
    with torch.no_grad():
        sb = pb.rsample_from_base_samples(torch.Size([4]), Z)
    sr = pr.rsample_from_base_samples(torch.Size([4]), Z)
    check("samples from base samples", sr, sb)
    assert tuple(pr.base_sample_shape) == tuple(pb.base_sample_shape)
    # The sampler (BoTorch's collapsed `S x 1 x q x m` base samples).
    s_bt = BtSampler(sample_shape=torch.Size([8]), seed=0)
    with torch.no_grad():
        yb = s_bt(pb)
    s_rs = RsSampler(sample_shape=[8], seed=0)
    s_rs.base_samples = s_bt.base_samples.clone()
    check("sampler samples", s_rs(pr), yb)


def test_fit(pair: tuple) -> None:
    cfg = pair[0]
    X, Y, Yvar = make_data(cfg, seed=301 + IDS.index(cfg[0]))
    bt, rs = build(cfg, X, Y, Yvar)
    torch.manual_seed(0)
    bt_fit(BtMLL(bt.likelihood, bt))
    rs_mll = RsMLL(rs.likelihood, rs)
    rs_fit(rs_mll)
    bt_losses = -bt_mll_values(bt).reshape(-1)
    rs_losses = -_t(rs_mll()).reshape(-1)
    excess = (rs_losses - bt_losses).clamp_min(0.0).max().item()
    ParityLog.record("fit loss excess (per element)", excess)
    assert excess <= 1e-6, (rs_losses, bt_losses)
    info = rs_mll._fit_result
    assert info["status"] in ("SUCCESS", "STOPPED")
    assert len(info["elements"]) == max(math.prod(rs._aug_batch_shape), 1)
    check("fit fval", info["fval"], rs_losses.sum(), tol=1e-12)


# ---------------------------------------------------------------------------
# ModelListGP
# ---------------------------------------------------------------------------

LIST_CONFIGS: list[tuple[str, list[tuple]]] = [
    ("two_single", [("a", 2, 10, 1, (), {}), ("b", 2, 7, 1, (), {"normalize": "shared"})]),
    (
        "mixed",
        [("a", 3, 12, 2, (), {"covar": "matern"}), ("b", 3, 9, 1, (), {"yvar": True})],
    ),
    (
        "three_kernels",
        [
            ("a", 2, 9, 1, (), {}),
            ("b", 2, 11, 1, (), {"covar": "matern"}),
            ("c", 2, 8, 1, (), {"covar": "legacy"}),
        ],
    ),
    (
        "two_multi_output",
        [("a", 2, 8, 2, (), {"normalize": "shared"}), ("b", 2, 10, 2, (), {"outcome": "none"})],
    ),
]
LIST_IDS = [c[0] for c in LIST_CONFIGS]


@contextlib.contextmanager
def dense_model_list() -> Iterator[None]:
    """BoTorch's ModelListGP posterior and scalarization with dense
    covariances (see the module docstring)."""
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


def build_list(cfgs: list[tuple], seed: int, explicit: bool = True) -> tuple[Any, Any]:
    bts, rss = [], []
    for i, cfg in enumerate(cfgs):
        X, Y, Yvar = make_data(cfg, seed=seed + 13 * i)
        bt, rs = build(cfg, X, Y, Yvar)
        if explicit:
            set_hyperparameters(bt)
            copy_hyperparameters(bt, rs)
        bts.append(bt)
        rss.append(rs)
    lb, lr = BtModelListGP(*bts), RsModelListGP(*rss)
    lb.eval()
    return lb, lr


@pytest.fixture(scope="module", params=LIST_CONFIGS, ids=LIST_IDS)
def list_pair(request: pytest.FixtureRequest) -> tuple[list[tuple], Any, Any]:
    cfgs = request.param[1]
    lb, lr = build_list(cfgs, seed=501 + LIST_IDS.index(request.param[0]))
    return cfgs, lb, lr


def test_model_list_posteriors(list_pair: tuple) -> None:
    cfgs, lb, lr = list_pair
    d = cfgs[0][1]
    assert lr.num_outputs == lb.num_outputs
    for i, shape in enumerate([(3, d), (5, 1, d), (2, 2, d)]):
        X = points_for(cfgs[0], shape, seed=41 + i)
        for noise in (False, True):
            with dense_model_list(), torch.no_grad():
                pb = lb.posterior(X, observation_noise=noise)
            pr = lr.posterior(X, observation_noise=noise)
            _compare_posteriors("ModelListGP posterior", pb, pr)
    w = weights_for(lr.num_outputs)
    X = points_for(cfgs[0], (3, d), seed=45)
    with dense_model_list(), torch.no_grad():
        pb = lb.posterior(X, posterior_transform=BtTransform(w, 0.1))
    pr = lr.posterior(X, posterior_transform=RsTransform(w, 0.1))
    _compare_posteriors("ModelListGP scalarized posterior", pb, pr)
    # output_indices (BoTorch supports subsets that keep sub-models whole).
    if len(cfgs) > 1 and cfgs[-1][3] == 1:
        idx = [lr.num_outputs - 1]
        with torch.no_grad():
            pb = lb.posterior(X, output_indices=idx)
        _compare_posteriors("ModelListGP output_indices", pb, lr.posterior(X, output_indices=idx))


def test_model_list_reference(list_pair: tuple) -> None:
    """The ModelListGP posterior is the block-diagonal concatenation of the
    sub-model posteriors. BoTorch's own result agrees with it for lists of
    single-output models; with multi-output sub-models it depends on the type
    of the sub-model's lazy covariance (``BlockDiagLinearOperator`` slicing is
    wrong in linear_operator 0.6.1) and is not compared."""
    cfgs, lb, lr = list_pair
    X = points_for(cfgs[0], (3, cfgs[0][1]), seed=47)
    with torch.no_grad():
        subs = [m.posterior(X) for m in lb.models]
        unpatched = lb.posterior(X).distribution.covariance_matrix
    means = torch.cat([p.mean for p in subs], dim=-1)
    covs = torch.block_diag(*[p.distribution.covariance_matrix for p in subs])
    pr = lr.posterior(X)
    check("ModelListGP vs sub-model posteriors mean", pr.mean, means)
    check("ModelListGP vs sub-model posteriors covariance", pr.distribution.covariance_matrix, covs)
    if all(c[3] == 1 for c in cfgs):
        check("ModelListGP vs BoTorch (unpatched)", pr.distribution.covariance_matrix, unpatched)


def test_model_list_mll_and_fit(list_pair: tuple) -> None:
    cfgs = list_pair[0]
    lb, lr = build_list(cfgs, seed=601, explicit=False)
    lb.train()
    with torch.no_grad():
        sb = BtSumMLL(lb.likelihood, lb)(lb(*lb.train_inputs), lb.train_targets)
    check("SumMarginalLogLikelihood (default)", RsSumMLL(lr.likelihood, lr)(), sb)
    torch.manual_seed(0)
    bt_fit(BtSumMLL(lb.likelihood, lb))
    rs_mll = RsSumMLL(lr.likelihood, lr)
    rs_fit(rs_mll)
    assert len(rs_mll._fit_result) == len(cfgs) and not rs_mll.training
    for mb, mr in zip(lb.models, lr.models, strict=True):
        bt_losses = -bt_mll_values(mb).reshape(-1)
        rs_losses = -_t(RsMLL(mr.likelihood, mr)()).reshape(-1)
        excess = (rs_losses - bt_losses).clamp_min(0.0).max().item()
        ParityLog.record("ModelListGP fit loss excess", excess)
        assert excess <= 1e-6
    lb.train()
    with torch.no_grad():
        sb = BtSumMLL(lb.likelihood, lb)(lb(*lb.train_inputs), lb.train_targets)
    check("SumMarginalLogLikelihood (fitted)", rs_mll(), sb, tol=1e-6)


# ---------------------------------------------------------------------------
# Acquisition functions with posterior transforms, constrained EI
# ---------------------------------------------------------------------------

ACQF_MODELS = [c[0] for c in UNBATCHED] + [f"list_{c[0]}" for c in LIST_CONFIGS]


@pytest.fixture(scope="module", params=ACQF_MODELS)
def acqf_pair(request: pytest.FixtureRequest) -> tuple[str, Any, Any, Any]:
    """An unbatched model pair (single-task or model list) with explicit
    hyperparameters, and its training inputs."""
    name = request.param
    if name.startswith("list_"):
        cfgs = dict(LIST_CONFIGS)[name[5:]]
        seed = 701 + ACQF_MODELS.index(name)
        lb, lr = build_list(cfgs, seed=seed)
        X = make_data(cfgs[0], seed=seed)[0]  # the training inputs of the first sub-model
        return name, lb, lr, X
    cfg = next(c for c in UNBATCHED if c[0] == name)
    X, Y, Yvar = make_data(cfg, seed=801 + ACQF_MODELS.index(name))
    bt, rs = build(cfg, X, Y, Yvar)
    set_hyperparameters(bt)
    copy_hyperparameters(bt, rs)
    bt.eval()
    return name, bt, rs, X


def values_and_grads(acqf: Any, X: Any) -> tuple[Any, Any]:
    Xg = X.clone().requires_grad_(True)
    values = acqf(Xg)
    (grad,) = torch.autograd.grad(values.sum(), Xg)
    return values.detach(), grad


def _in_range(X: Any, n: int, q: int, seed: int) -> Any:
    lo, hi = X.min().item(), X.max().item()
    g = torch.Generator().manual_seed(seed)
    span = hi - lo
    return lo - 0.1 * span + 1.2 * span * torch.rand(n, q, X.shape[-1], generator=g, dtype=DT)


def _scalarized_stats(bt: Any, w: Any, Xt: Any) -> tuple[float, float]:
    with dense_model_list(), torch.no_grad():
        p = bt.posterior(Xt, posterior_transform=BtTransform(w, 0.2))
    return p.mean.max().item(), p.variance.sqrt().mean().item()


ANALYTIC = [
    ("LogExpectedImprovement", 0.0, True),
    ("LogExpectedImprovement", 3.0, True),
    ("LogExpectedImprovement", -1.0, False),
    ("ExpectedImprovement", 0.0, True),
    ("ProbabilityOfImprovement", 0.5, True),
    ("LogProbabilityOfImprovement", 2.0, True),
    ("UpperConfidenceBound", None, True),
    ("UpperConfidenceBound", None, False),
    ("PosteriorMean", None, False),
    ("PosteriorStandardDeviation", None, True),
]


def _log_ei_grad_scale(bt: Any, Xt: Any, w: Any, best_f: float) -> Any:
    """BoTorch's LogEI gradient has a relative error of about ``eps u^2``
    deep in the tail (see test_parity.py); errors are reported relative to
    that bound."""
    with dense_model_list(), torch.no_grad():
        p = bt.posterior(Xt, posterior_transform=BtTransform(w, 0.2))
    u = (p.mean.reshape(-1) - best_f) / p.variance.reshape(-1).clamp_min(1e-12).sqrt()
    return (RTOL / (RTOL + 4 * 2.2e-16 * u**2)).min().item()


def test_scalarized_analytic_acquisition(acqf_pair: tuple) -> None:
    name, bt, rs, X = acqf_pair
    w = weights_for(rs.num_outputs)
    Xt = _in_range(X, 8, 1, seed=3)
    best, spread = _scalarized_stats(bt, w, Xt)
    for acqf_name, shift, maximize in ANALYTIC:
        kw: dict[str, Any] = {"maximize": maximize}
        if acqf_name == "UpperConfidenceBound":
            kw["beta"] = torch.tensor(0.4 if maximize else 1.5, dtype=DT)
        elif shift is not None:
            kw["best_f"] = torch.tensor(best + shift * spread, dtype=DT)
        a_bt = getattr(bt_analytic, acqf_name)(bt, posterior_transform=BtTransform(w, 0.2), **kw)
        a_rs = getattr(rs_analytic, acqf_name)(rs, posterior_transform=RsTransform(w, 0.2), **kw)
        with dense_model_list():
            v_bt, g_bt = values_and_grads(a_bt, Xt)
        v_rs, g_rs = values_and_grads(a_rs, Xt)
        if a_rs._log:
            check_log(f"scalarized {acqf_name} value", v_rs, v_bt)
        else:
            check(f"scalarized {acqf_name} value", v_rs, v_bt)
        err = max(rel_err(g_rs[i], g_bt[i], atol=1e-14) for i in range(Xt.shape[0]))
        if acqf_name == "LogExpectedImprovement":
            err *= _log_ei_grad_scale(bt, Xt, w, kw["best_f"].item())
        ParityLog.record(f"scalarized {acqf_name} grad", err)
        assert err <= GRAD_RTOL, (name, acqf_name, err)


MC = [
    ("qExpectedImprovement", bt_mc, rs_mc, {"best_f": -0.5}),
    ("qLogExpectedImprovement", bt_logei, rs_logei, {"best_f": 0.0}),
    ("qLogExpectedImprovement", bt_logei, rs_logei, {"best_f": 0.0, "fat": False}),
    ("qProbabilityOfImprovement", bt_mc, rs_mc, {"best_f": 0.0, "tau": 0.1}),
    ("qUpperConfidenceBound", bt_mc, rs_mc, {"beta": 0.2}),
    ("qSimpleRegret", bt_mc, rs_mc, {}),
]


@pytest.mark.parametrize("q,n_pending", [(1, 0), (2, 0), (3, 2)])
def test_scalarized_mc_acquisition(acqf_pair: tuple, q: int, n_pending: int) -> None:
    name, bt, rs, X = acqf_pair
    w = weights_for(rs.num_outputs)
    Xt = _in_range(X, 1, q, seed=5 + q)
    best, spread = _scalarized_stats(bt, w, _in_range(X, 6, 1, seed=8))
    S, D = 64, q + n_pending
    Z = torch.randn(S, D, generator=torch.Generator().manual_seed(q + 10 * n_pending), dtype=DT)
    pending = X[:n_pending] + 0.01 if n_pending else None
    for acqf_name, bt_mod, rs_mod, kwargs in MC:
        # Float64 tensors: BoTorch stores Python floats for best_f / tau in float32.
        kw = {
            k: (
                torch.tensor(best + v * spread, dtype=DT)
                if k == "best_f"
                else torch.tensor(v, dtype=DT)
                if k == "tau"
                else v
            )
            for k, v in kwargs.items()
        }
        s_bt = BtSampler(sample_shape=torch.Size([S]), seed=0)
        s_bt.base_samples = Z.view(S, 1, D).clone()
        s_rs = RsSampler(sample_shape=torch.Size([S]), seed=0)
        s_rs.base_samples = Z.view(S, 1, D).clone()
        common = {"X_pending": pending, "posterior_transform": None}
        a_bt = getattr(bt_mod, acqf_name)(
            bt, sampler=s_bt, **{**common, "posterior_transform": BtTransform(w, 0.2)}, **kw
        )
        a_rs = getattr(rs_mod, acqf_name)(
            rs, sampler=s_rs, **{**common, "posterior_transform": RsTransform(w, 0.2)}, **kw
        )
        with dense_model_list():
            v_bt, g_bt = values_and_grads(a_bt, Xt)
        assert torch.equal(s_bt.base_samples.view(S, D), Z)
        v_rs, g_rs = values_and_grads(a_rs, Xt)
        label = f"scalarized {acqf_name}"
        if a_rs._log:
            check_log(f"{label} value", v_rs, v_bt)
        else:
            check(f"{label} value", v_rs, v_bt)
        check(f"{label} grad", g_rs, g_bt, tol=GRAD_RTOL, atol=1e-14)


@pytest.mark.parametrize("q,n_pending,incremental", [(1, 0, True), (2, 2, True), (2, 2, False)])
def test_scalarized_qlognei(acqf_pair: tuple, q: int, n_pending: int, incremental: bool) -> None:
    name, bt, rs, X = acqf_pair
    w = weights_for(rs.num_outputs)
    S = 64
    D = X.shape[0] + n_pending + q
    Z = torch.randn(S, D, generator=torch.Generator().manual_seed(D), dtype=DT)
    s_bt = BtSampler(sample_shape=torch.Size([S]), seed=0)
    s_bt.base_samples = Z.view(S, 1, D).clone()
    s_rs = RsSampler(sample_shape=[S], seed=0)
    s_rs.base_samples = Z
    pending = X[:n_pending] + 0.01 if n_pending else None
    common = dict(X_baseline=X, X_pending=pending, prune_baseline=False, incremental=incremental)
    with dense_model_list():
        a_bt = bt_logei.qLogNoisyExpectedImprovement(
            bt, sampler=s_bt, cache_root=False, posterior_transform=BtTransform(w, 0.2), **common
        )
        Xt = _in_range(X, 1, q, seed=9)
        v_bt, g_bt = values_and_grads(a_bt, Xt)
    a_rs = rs_logei.qLogNoisyExpectedImprovement(
        rs, sampler=s_rs, posterior_transform=RsTransform(w, 0.2), **common
    )
    v_rs, g_rs = values_and_grads(a_rs, Xt)
    assert torch.equal(s_bt.base_samples.view(S, D), Z)
    check_log("scalarized qLogNEI value", v_rs, v_bt)
    check("scalarized qLogNEI grad", g_rs, g_bt, tol=GRAD_RTOL, atol=1e-14)


def test_scalarized_prune_inferior_points(acqf_pair: tuple) -> None:
    name, bt, rs, X = acqf_pair
    w = weights_for(rs.num_outputs)
    Xp = torch.cat([X, X + 0.03])
    s_bt = BtSampler(sample_shape=torch.Size([256]), seed=7)
    with dense_model_list():
        pruned_bt = bt_prune(bt, Xp, posterior_transform=BtTransform(w, 0.2), sampler=s_bt)
    s_rs = RsSampler(sample_shape=[256], seed=0)
    s_rs.base_samples = s_bt.base_samples.view(256, -1).clone()
    pruned_rs = rs_prune(rs, Xp, posterior_transform=RsTransform(w, 0.2), sampler=s_rs)
    assert torch.equal(pruned_rs, pruned_bt)
    ParityLog.record("scalarized prune_inferior_points (mismatches)", 0.0)


def _constraint_configs(m: int, means: Any, sds: Any) -> list[tuple[int, dict[int, Any], bool]]:
    def b(v: Any) -> Any:
        return torch.tensor(float(v), dtype=DT)

    med, sd = means.median(dim=0).values, sds.mean(dim=0)
    configs = [
        (0, {1: (b(med[1]), None)}, True),
        (1, {0: (None, b(med[0]))}, True),
        (0, {1: (b(med[1] - sd[1]), b(med[1] + 0.5 * sd[1]))}, False),
        (1, {0: (b(med[0] + 8 * sd[0]), None)}, True),  # deep in the tail
    ]
    if m >= 3:
        configs += [
            (1, {0: (b(med[0] - sd[0]), None), 2: (b(med[2] - sd[2]), b(med[2] + sd[2]))}, True),
            (2, {1: (None, b(med[1])), 0: (b(med[0] - 2 * sd[0]), b(med[0] + sd[0]))}, False),
        ]
    return configs


def test_constrained_expected_improvement(acqf_pair: tuple) -> None:
    name, bt, rs, X = acqf_pair
    m = rs.num_outputs
    if m < 2:
        return
    Xt = _in_range(X, 8, 1, seed=13)
    with dense_model_list(), torch.no_grad():
        post = bt.posterior(Xt)
    means, sds = post.mean.squeeze(-2), post.variance.squeeze(-2).sqrt()
    for obj, cons, maximize in _constraint_configs(m, means, sds):
        best = means[:, obj].max() if maximize else means[:, obj].min()
        for acqf_name in ("LogConstrainedExpectedImprovement", "ConstrainedExpectedImprovement"):
            kw = dict(best_f=best.clone(), objective_index=obj, constraints=cons, maximize=maximize)
            a_bt = getattr(bt_analytic, acqf_name)(bt, **kw)
            a_rs = getattr(rs_analytic, acqf_name)(rs, **kw)
            with dense_model_list():
                v_bt, g_bt = values_and_grads(a_bt, Xt)
            v_rs, g_rs = values_and_grads(a_rs, Xt)
            if a_rs._log:
                check_log(f"{acqf_name} value", v_rs, v_bt)
            else:
                check(f"{acqf_name} value", v_rs, v_bt, atol=1e-300)
            err = max(rel_err(g_rs[i], g_bt[i], atol=1e-14) for i in range(Xt.shape[0]))
            ParityLog.record(f"{acqf_name} grad", err)
            assert err <= GRAD_RTOL, (name, acqf_name, cons, err)


def test_single_output_scalarization() -> None:
    """A posterior transform on a single-output model (``m = 1`` weights)."""
    cfg = ("so", 2, 10, 1, (), {})
    X, Y, Yvar = make_data(cfg, seed=901)
    bt, rs = build(cfg, X, Y, Yvar)
    set_hyperparameters(bt)
    copy_hyperparameters(bt, rs)
    bt.eval()
    w = torch.tensor([-1.5], dtype=DT)
    Xt = _in_range(X, 6, 1, seed=2)
    for acqf_name, kw in [
        ("LogExpectedImprovement", {"best_f": torch.tensor(-1.0, dtype=DT)}),
        ("UpperConfidenceBound", {"beta": torch.tensor(0.5, dtype=DT)}),
    ]:
        a_bt = getattr(bt_analytic, acqf_name)(bt, posterior_transform=BtTransform(w, 0.4), **kw)
        a_rs = getattr(rs_analytic, acqf_name)(rs, posterior_transform=RsTransform(w, 0.4), **kw)
        v_bt, g_bt = values_and_grads(a_bt, Xt)
        v_rs, g_rs = values_and_grads(a_rs, Xt)
        check_log(f"m=1 scalarized {acqf_name} value", v_rs, v_bt)
        check(f"m=1 scalarized {acqf_name} grad", g_rs, g_bt, tol=GRAD_RTOL, atol=1e-14)
    with torch.no_grad():
        pb = bt.posterior(Xt.squeeze(1), posterior_transform=BtTransform(w, 0.4))
    pr = rs.posterior(Xt.squeeze(1), posterior_transform=RsTransform(w, 0.4))
    _compare_posteriors("m=1 scalarized posterior", pb, pr)

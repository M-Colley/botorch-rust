"""Parity with BoTorch through the Python API.

Identical training data and hyperparameters must give identical (to ~1e-9)
default initializations, marginal log likelihoods, posteriors and acquisition
values and gradients (Monte-Carlo acquisition functions with identical base
samples); fitting must reach BoTorch's final loss and acquisition
optimization BoTorch's best value.

Fits are compared on the model and twins whose training inputs are
perturbed at 1e-14 .. 1e-11 (BoTorch's own outcome depends on rounding), in
two stages (``parity_helpers.compare_twin_fits``): on 16 twins, botorch_rs's
best first-attempt loss must reach BoTorch's best, and its own fit must not
end above BoTorch's worst, within ``max(1e-6, 10 x`` the spread of BoTorch's
outcomes at its best optimum``)``; if not, on 64 twins, botorch_rs's losses
must not be stochastically larger than BoTorch's (one-sided Mann–Whitney
test at 1e-3) and its best must reach BoTorch's ``k``-th best. Data seeds
are deterministic (``zlib.crc32`` of the configuration name; Python's
``hash`` is salted per process).

Skipped unless BoTorch is importable; to test against a source checkout use
``PYTHONPATH=/path/to/botorch``.
"""

from __future__ import annotations

import math
import warnings
import zlib
from collections.abc import Iterator
from typing import Any

import pytest

from helpers import importorskip_botorch

importorskip_botorch()
torch = pytest.importorskip("torch")

import botorch  # noqa: E402
import botorch.optim.initializers  # noqa: E402
import botorch.optim.parameter_constraints  # noqa: E402
import gpytorch  # noqa: E402
from botorch.acquisition import analytic as bt_analytic  # noqa: E402
from botorch.acquisition import logei as bt_logei  # noqa: E402
from botorch.acquisition import monte_carlo as bt_mc  # noqa: E402
from botorch.acquisition.utils import prune_inferior_points as bt_prune  # noqa: E402
from botorch.fit import fit_gpytorch_mll as bt_fit  # noqa: E402
from botorch.models import SingleTaskGP as BtSingleTaskGP  # noqa: E402
from botorch.models.transforms.input import Normalize as BtNormalize  # noqa: E402
from botorch.models.utils import gpytorch_modules as bt_modules  # noqa: E402
from botorch.optim import optimize_acqf as bt_optimize_acqf  # noqa: E402
from botorch.sampling.normal import SobolQMCNormalSampler as BtSampler  # noqa: E402
from botorch.test_functions import synthetic as bt_synthetic  # noqa: E402
from gpytorch.mlls import ExactMarginalLogLikelihood as BtMLL  # noqa: E402

import botorch_rs  # noqa: E402
import botorch_rs.optim.initializers  # noqa: E402
import botorch_rs.optim.parameter_constraints  # noqa: E402
from botorch_rs import kernels as rs_kernels  # noqa: E402
from botorch_rs import priors as rs_priors  # noqa: E402
from botorch_rs.acquisition import analytic as rs_analytic  # noqa: E402
from botorch_rs.acquisition import logei as rs_logei  # noqa: E402
from botorch_rs.acquisition import monte_carlo as rs_mc  # noqa: E402
from botorch_rs.acquisition.utils import prune_inferior_points as rs_prune  # noqa: E402
from botorch_rs.exceptions import InputDataWarning as RsInputDataWarning  # noqa: E402
from botorch_rs.fit import fit_gpytorch_mll as rs_fit  # noqa: E402
from botorch_rs.mlls import ExactMarginalLogLikelihood as RsMLL  # noqa: E402
from botorch_rs.models import SingleTaskGP as RsSingleTaskGP  # noqa: E402
from botorch_rs.models.transforms import Normalize as RsNormalize  # noqa: E402
from botorch_rs.models.utils import gpytorch_modules as rs_modules  # noqa: E402
from botorch_rs.optim import optimize_acqf as rs_optimize_acqf  # noqa: E402
from botorch_rs.sampling import SobolQMCNormalSampler as RsSampler  # noqa: E402
from botorch_rs.test_functions import synthetic as rs_synthetic  # noqa: E402
from parity_helpers import Checker, FitOutcome, compare_twin_fits  # noqa: E402

DT = torch.float64
RTOL = 1e-9
TWIN_SCALES = (1e-14, 1e-13, 1e-12, 1e-11)
N_TWINS = 16
CHECK = Checker("SingleTaskGP fits (test_parity)")


pytestmark = pytest.mark.filterwarnings("ignore")


@pytest.fixture(autouse=True, scope="module")
def _single_threaded_torch() -> Iterator[None]:
    """Single-threaded torch, for deterministic BoTorch results."""
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(threads)


def test_reference_is_botorch() -> None:
    assert "botorch_rs" not in botorch.__file__


def close(a: Any, b: Any, rtol: float = RTOL, atol: float = 1e-12) -> bool:
    a = torch.as_tensor(a, dtype=DT).detach()
    b = torch.as_tensor(b, dtype=DT).detach()
    assert a.shape == b.shape, (a.shape, b.shape)
    return bool(torch.allclose(a, b, rtol=rtol, atol=atol))


def config_seed(config: str) -> int:
    """A deterministic data seed of a configuration (``hash`` is salted)."""
    return zlib.crc32(config.encode()) % 1000


def make_data(n: int, d: int, seed: int, low: float = 0.0, high: float = 1.0) -> tuple:
    g = torch.Generator().manual_seed(seed)
    X = low + (high - low) * torch.rand(n, d, generator=g, dtype=DT)
    Xn = (X - low) / (high - low)
    Y = torch.sin(6 * Xn[:, :1]) + Xn[:, 1:].sum(-1, keepdim=True) ** 2
    Y = Y + 0.05 * torch.randn(n, 1, generator=g, dtype=DT)
    Yvar = 0.01 + 0.02 * torch.rand(n, 1, generator=g, dtype=DT)
    return X, Y, Yvar


# Model configurations: (name, d, n, input range).
CONFIGS = [
    ("default", 2, 12, (0.0, 1.0)),
    ("normalize", 3, 15, (-2.0, 5.0)),
    ("explicit_bounds", 2, 10, (-1.0, 1.0)),
    ("fixed_noise", 2, 12, (0.0, 1.0)),
    ("no_outcome_transform", 2, 12, (0.0, 1.0)),
    ("matern", 4, 16, (0.0, 1.0)),
    ("legacy_gamma", 2, 12, (0.0, 1.0)),
    ("custom_scale_rbf", 3, 12, (0.0, 1.0)),
]


def build(config: str, X: Any, Y: Any, Yvar: Any) -> tuple[Any, Any]:
    """The same model in BoTorch and botorch_rs."""
    d = X.shape[-1]
    bt_kw: dict[str, Any] = {}
    rs_kw: dict[str, Any] = {}
    if config == "normalize":
        bt_kw["input_transform"] = BtNormalize(d=d)
        rs_kw["input_transform"] = RsNormalize(d=d)
    elif config == "explicit_bounds":
        bounds = torch.tensor([[-1.0] * d, [1.0] * d], dtype=DT)
        bt_kw["input_transform"] = BtNormalize(d=d, bounds=bounds)
        rs_kw["input_transform"] = RsNormalize(d=d, bounds=bounds)
    elif config == "fixed_noise":
        bt_kw["train_Yvar"] = Yvar
        rs_kw["train_Yvar"] = Yvar
    elif config == "no_outcome_transform":
        bt_kw["outcome_transform"] = None
        rs_kw["outcome_transform"] = None
    elif config == "matern":
        bt_kw["covar_module"] = bt_modules.get_covar_module_with_dim_scaled_prior(
            ard_num_dims=d, use_rbf_kernel=False
        )
        rs_kw["covar_module"] = rs_modules.get_covar_module_with_dim_scaled_prior(
            ard_num_dims=d, use_rbf_kernel=False
        )
    elif config == "legacy_gamma":
        bt_kw["covar_module"] = bt_modules.get_matern_kernel_with_gamma_prior(ard_num_dims=d)
        bt_kw["likelihood"] = bt_modules.get_gaussian_likelihood_with_gamma_prior()
        rs_kw["covar_module"] = rs_modules.get_matern_kernel_with_gamma_prior(ard_num_dims=d)
        rs_kw["likelihood"] = rs_modules.get_gaussian_likelihood_with_gamma_prior()
    elif config == "custom_scale_rbf":
        bt_kw["covar_module"] = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(
                ard_num_dims=d, lengthscale_prior=gpytorch.priors.GammaPrior(3.0, 6.0)
            ),
            outputscale_prior=gpytorch.priors.GammaPrior(2.0, 0.15),
        )
        rs_kw["covar_module"] = rs_kernels.ScaleKernel(
            rs_kernels.RBFKernel(ard_num_dims=d, lengthscale_prior=rs_priors.GammaPrior(3.0, 6.0)),
            outputscale_prior=rs_priors.GammaPrior(2.0, 0.15),
        )
    bt = BtSingleTaskGP(
        X,
        Y,
        **{k: v for k, v in bt_kw.items() if k != "train_Yvar"},
        train_Yvar=bt_kw.get("train_Yvar"),
    )
    rs = RsSingleTaskGP(
        X,
        Y,
        **{k: v for k, v in rs_kw.items() if k != "train_Yvar"},
        train_Yvar=rs_kw.get("train_Yvar"),
    )
    return bt, rs


def bt_mll_value(model: Any) -> float:
    mll = BtMLL(model.likelihood, model)
    model.train()
    with torch.no_grad():
        value = mll(model(*model.train_inputs), model.train_targets).item()
    model.eval()
    return value


def set_hyperparameters(bt: Any) -> None:
    """Explicit (non-default) hyperparameter values in the BoTorch model."""
    d = bt.train_inputs[0].shape[-1]
    values = torch.linspace(0.3, 0.9, d, dtype=DT)
    kwargs: dict[str, Any] = {"mean_module.constant": 0.15}
    if isinstance(bt.covar_module, gpytorch.kernels.ScaleKernel):
        kwargs["covar_module.outputscale"] = 1.7
        kwargs["covar_module.base_kernel.lengthscale"] = values
    else:
        kwargs["covar_module.lengthscale"] = values
    if not isinstance(bt.likelihood, gpytorch.likelihoods.FixedNoiseGaussianLikelihood):
        kwargs["likelihood.noise"] = 0.05
    bt.initialize(**kwargs)


def copy_hyperparameters(bt: Any, rs: Any) -> None:
    rs.initialize(**{name: value.detach() for name, value in bt.named_parameters()})


@pytest.mark.parametrize("config,d,n,xrange", CONFIGS, ids=[c[0] for c in CONFIGS])
def test_model_parity(config: str, d: int, n: int, xrange: tuple) -> None:
    X, Y, Yvar = make_data(n, d, seed=config_seed(config), low=xrange[0], high=xrange[1])
    bt, rs = build(config, X, Y, Yvar)
    # Default initialization (including GPyTorch's float32 rounding of constants).
    bt_params = {k: v.detach() for k, v in bt.named_parameters()}
    rs_params = dict(rs.named_parameters())
    assert list(bt_params) == list(rs_params)
    for name in bt_params:
        assert close(rs_params[name], bt_params[name], rtol=1e-12), name
    assert close(rs.train_targets, bt.train_targets, rtol=1e-12)
    assert close(RsMLL(rs.likelihood, rs)(), bt_mll_value(bt))
    # Explicit hyperparameters: MLL and posteriors.
    set_hyperparameters(bt)
    copy_hyperparameters(bt, rs)
    assert close(RsMLL(rs.likelihood, rs)(), bt_mll_value(bt))
    g = torch.Generator().manual_seed(1)
    Xt = xrange[0] + (xrange[1] - xrange[0]) * torch.rand(6, d, generator=g, dtype=DT)
    with torch.no_grad():
        for noise in (False, True):
            pb = bt.posterior(Xt.unsqueeze(1), observation_noise=noise)
            pr = rs.posterior(Xt.unsqueeze(1), observation_noise=noise)
            assert close(pr.mean, pb.mean) and close(pr.variance, pb.variance)
            jb = bt.posterior(Xt[:3], observation_noise=noise)
            jr = rs.posterior(Xt[:3], observation_noise=noise)
            assert close(jr.mean, jb.mean)
            assert close(jr.distribution.covariance_matrix, jb.distribution.covariance_matrix)
    assert close(
        rs.covar_module.lengthscale
        if not isinstance(rs.covar_module, rs_kernels.ScaleKernel)
        else rs.covar_module.base_kernel.lengthscale,
        bt.covar_module.lengthscale
        if not isinstance(bt.covar_module, gpytorch.kernels.ScaleKernel)
        else bt.covar_module.base_kernel.lengthscale,
    )
    assert close(rs.likelihood.noise, bt.likelihood.noise)


def twin_inputs(X: Any, t: int) -> Any:
    """Twin ``t`` of the training inputs ``X`` (``t = 0``: ``X``), perturbed
    at 1e-14 .. 1e-11 relative."""
    if t == 0:
        return X
    g = torch.Generator().manual_seed(t)
    rel = TWIN_SCALES[(t - 1) % len(TWIN_SCALES)]
    return X * (1 + rel * torch.randn(X.shape, generator=g, dtype=DT))


def fit_losses(config: str, X: Any, Y: Any, Yvar: Any) -> tuple[tuple, tuple]:
    """The final losses and attempt counts of BoTorch's and botorch_rs's fits
    of the same model from the default initialization."""
    import botorch.optim.fit as bt_optim_fit

    bt, rs = build(config, X, Y, Yvar)
    record: dict[str, Any] = {"attempts": 0}
    orig = bt_optim_fit.fit_gpytorch_mll_scipy

    def recorder(mll: Any, *args: Any, **kwargs: Any) -> Any:
        record["attempts"] += 1
        record["r"] = orig(mll, *args, **kwargs)
        return record["r"]

    bt_fit(BtMLL(bt.likelihood, bt), optimizer=recorder)
    rs_mll = RsMLL(rs.likelihood, rs)
    rs_fit(rs_mll)
    rs_out = (float(rs_mll._fit_result["fval"]), int(rs_mll._fit_result["attempts"]))
    return (float(record["r"].fval), record["attempts"]), rs_out


def compare_fits(
    label: str, config: str, X: Any, Y: Any, Yvar: Any, twins: int = N_TWINS
) -> tuple[float, float]:
    """Fits of the model and its twins in both libraries, compared in two
    stages (see the module docstring), ``twins`` twins in stage 1. Returns
    the gap of botorch_rs's own fit above BoTorch's and BoTorch's spread at
    its best optimum."""

    def fit_twin(t: int) -> tuple[FitOutcome, FitOutcome]:
        return fit_losses(config, twin_inputs(X, t), Y, Yvar)

    (c,) = compare_twin_fits(CHECK, label, fit_twin, twins)
    return c.gap, c.spread


@pytest.mark.parametrize("config,d,n,xrange", CONFIGS, ids=[c[0] for c in CONFIGS])
def test_fit_parity(config: str, d: int, n: int, xrange: tuple) -> None:
    """From the default initialization, fitting reaches BoTorch's final loss
    (compared on twins in two stages)."""
    X, Y, Yvar = make_data(n, d, seed=7 + config_seed(config), low=xrange[0], high=xrange[1])
    compare_fits("fit", config, X, Y, Yvar)


def test_fit_iterates_match_botorch() -> None:
    """A few L-BFGS-B iterations from the same start give the same loss
    (the L-BFGS-B port reproduces scipy's iterates)."""
    X, Y, _ = make_data(15, 3, seed=3, low=-2.0, high=5.0)
    for maxiter in (1, 3, 8):
        bt, rs = build("normalize", X, Y, None)
        options = {"options": {"maxiter": maxiter}}
        bt_fit(BtMLL(bt.likelihood, bt), optimizer_kwargs=options)
        rs_mll = RsMLL(rs.likelihood, rs)
        rs_fit(rs_mll, optimizer_kwargs=options)
        assert close(-float(rs_mll()), -bt_mll_value(bt), rtol=1e-8)


@pytest.fixture(scope="module")
def models() -> tuple[Any, Any, Any, Any]:
    """A BoTorch / botorch_rs model pair with identical explicit hyperparameters."""
    X, Y, _ = make_data(14, 3, seed=11, low=-1.0, high=2.0)
    bt, rs = build("normalize", X, Y, None)
    set_hyperparameters(bt)
    copy_hyperparameters(bt, rs)
    bt.eval()
    return bt, rs, X, Y


def values_and_grads(acqf: Any, X: Any) -> tuple[Any, Any]:
    Xg = X.clone().requires_grad_(True)
    values = acqf(Xg)
    (grad,) = torch.autograd.grad(values.sum(), Xg)
    return values.detach(), grad


ANALYTIC = [
    ("ExpectedImprovement", {"best_f": 0.0}),
    ("ExpectedImprovement", {"best_f": 1.5}),
    ("LogExpectedImprovement", {"best_f": 0.0}),
    ("LogExpectedImprovement", {"best_f": 3.0}),
    ("ProbabilityOfImprovement", {"best_f": 0.5}),
    ("LogProbabilityOfImprovement", {"best_f": 2.0}),
    ("UpperConfidenceBound", {"beta": 0.2}),
    ("PosteriorMean", {}),
    ("PosteriorStandardDeviation", {}),
]


@pytest.mark.parametrize("maximize", [True, False])
@pytest.mark.parametrize(
    "name,kwargs", ANALYTIC, ids=[f"{n}-{i}" for i, (n, _) in enumerate(ANALYTIC)]
)
def test_analytic_acquisition_parity(
    models: tuple, name: str, kwargs: dict, maximize: bool
) -> None:
    bt, rs, X, _ = models
    kw = {k: torch.tensor(v, dtype=DT) for k, v in kwargs.items()}
    a_bt = getattr(bt_analytic, name)(bt, maximize=maximize, **kw)
    a_rs = getattr(rs_analytic, name)(rs, maximize=maximize, **kw)
    g = torch.Generator().manual_seed(2)
    Xt = -1.0 + 3.0 * torch.rand(8, 1, 3, generator=g, dtype=DT)
    v_bt, g_bt = values_and_grads(a_bt, Xt)
    v_rs, g_rs = values_and_grads(a_rs, Xt)
    assert close(v_rs, v_bt)
    assert close(g_rs, g_bt, rtol=1e-8, atol=1e-10)


MC = [
    ("qExpectedImprovement", bt_mc, rs_mc, {"best_f": 1.0}),
    ("qLogExpectedImprovement", bt_logei, rs_logei, {"best_f": 1.0}),
    ("qLogExpectedImprovement", bt_logei, rs_logei, {"best_f": 1.0, "fat": False}),
    ("qProbabilityOfImprovement", bt_mc, rs_mc, {"best_f": 1.0, "tau": 0.1}),
    ("qUpperConfidenceBound", bt_mc, rs_mc, {"beta": 0.2}),
    ("qSimpleRegret", bt_mc, rs_mc, {}),
]


@pytest.mark.parametrize("q,n_pending", [(1, 0), (2, 0), (3, 2)])
@pytest.mark.parametrize(
    "name,bt_mod,rs_mod,kwargs", MC, ids=[f"{m[0]}-{i}" for i, m in enumerate(MC)]
)
def test_mc_acquisition_parity(
    models: tuple, name: str, bt_mod: Any, rs_mod: Any, kwargs: dict, q: int, n_pending: int
) -> None:
    bt, rs, X, _ = models
    S, D = 128, q + n_pending
    Z = torch.randn(S, D, generator=torch.Generator().manual_seed(q + 10 * n_pending), dtype=DT)
    s_bt = BtSampler(sample_shape=torch.Size([S]), seed=0)
    s_bt.base_samples = Z.view(S, 1, D).clone()
    s_rs = RsSampler(sample_shape=torch.Size([S]), seed=0)
    s_rs.base_samples = Z.view(S, 1, D).clone()
    pending = X[:n_pending] + 0.1 if n_pending else None
    kw = {
        k: (torch.tensor(v, dtype=DT) if k in ("best_f", "tau") else v) for k, v in kwargs.items()
    }
    a_bt = getattr(bt_mod, name)(bt, sampler=s_bt, X_pending=pending, **kw)
    a_rs = getattr(rs_mod, name)(rs, sampler=s_rs, X_pending=pending, **kw)
    g = torch.Generator().manual_seed(5)
    Xt = -1.0 + 3.0 * torch.rand(1, q, 3, generator=g, dtype=DT)
    v_bt, g_bt = values_and_grads(a_bt, Xt)
    assert torch.equal(s_bt.base_samples.view(S, D), Z), "BoTorch replaced the base samples"
    v_rs, g_rs = values_and_grads(a_rs, Xt)
    assert close(v_rs, v_bt)
    assert close(g_rs, g_bt, rtol=1e-8, atol=1e-10)


@pytest.mark.parametrize("q,n_pending,incremental", [(1, 0, True), (2, 2, True), (2, 2, False)])
def test_qlognei_parity(models: tuple, q: int, n_pending: int, incremental: bool) -> None:
    bt, rs, X, _ = models
    S = 64
    D = X.shape[0] + n_pending + q
    Z = torch.randn(S, D, generator=torch.Generator().manual_seed(D), dtype=DT)
    s_bt = BtSampler(sample_shape=torch.Size([S]), seed=0)
    s_bt.base_samples = Z.view(S, 1, D).clone()
    s_rs = RsSampler(sample_shape=[S], seed=0)
    s_rs.base_samples = Z
    pending = X[:n_pending] + 0.1 if n_pending else None
    common = dict(X_baseline=X, X_pending=pending, prune_baseline=False, incremental=incremental)
    a_bt = bt_logei.qLogNoisyExpectedImprovement(bt, sampler=s_bt, cache_root=False, **common)
    a_rs = rs_logei.qLogNoisyExpectedImprovement(rs, sampler=s_rs, **common)
    Xt = -1.0 + 3.0 * torch.rand(1, q, 3, generator=torch.Generator().manual_seed(9), dtype=DT)
    v_bt, g_bt = values_and_grads(a_bt, Xt)
    assert torch.equal(s_bt.base_samples.view(S, D), Z)
    v_rs, g_rs = values_and_grads(a_rs, Xt)
    assert close(v_rs, v_bt)
    assert close(g_rs, g_bt, rtol=1e-8, atol=1e-10)
    assert close(a_rs.X_baseline, a_bt.X_baseline, rtol=0, atol=0)


def test_prune_inferior_points_parity(models: tuple) -> None:
    bt, rs, X, _ = models
    Xp = torch.cat([X, X + 0.05])
    s_bt = BtSampler(sample_shape=torch.Size([256]), seed=7)
    pruned_bt = bt_prune(bt, Xp, sampler=s_bt)
    s_rs = RsSampler(sample_shape=[256], seed=0)
    s_rs.base_samples = s_bt.base_samples.view(256, -1).clone()
    pruned_rs = rs_prune(rs, Xp, sampler=s_rs)
    assert torch.equal(pruned_rs, pruned_bt)


def weyl(n: int, d: int) -> Any:
    primes = [2, 3, 5, 7, 11, 13, 17, 19]
    alpha = torch.tensor([math.sqrt(p) % 1.0 for p in primes[:d]], dtype=DT)
    i = torch.arange(1, n + 1, dtype=DT).unsqueeze(-1)
    return torch.remainder(0.5 + i * alpha, 1.0)


@pytest.fixture(scope="module")
def hartmann_models() -> tuple[Any, Any, Any]:
    """The fixed-hyperparameter Hartmann-6 model of scripts/benchmark.py."""
    X = weyl(50, 6)
    Y = bt_synthetic.Hartmann(dim=6, negate=True)(X, noise=False).unsqueeze(-1)
    bt = BtSingleTaskGP(X, Y)
    rs = RsSingleTaskGP(X, Y)
    with torch.no_grad():
        bt.likelihood.noise_covar.raw_noise.fill_(1e-3)
        bt.mean_module.raw_constant.fill_(0.0)
        bt.covar_module.raw_lengthscale.fill_(0.5)
    copy_hyperparameters(bt, rs)
    bt.eval()
    return bt, rs, Y


def test_optimize_acqf_logei_parity(hartmann_models: tuple) -> None:
    bt, rs, Y = hartmann_models
    bounds = torch.stack([torch.zeros(6, dtype=DT), torch.ones(6, dtype=DT)])
    best_bt, best_rs = -math.inf, -math.inf
    for seed in range(3):
        _, v_bt = bt_optimize_acqf(
            bt_analytic.LogExpectedImprovement(bt, best_f=Y.max()),
            bounds,
            q=1,
            num_restarts=10,
            raw_samples=512,
            options={"seed": seed},
        )
        x_rs, v_rs = rs_optimize_acqf(
            rs_analytic.LogExpectedImprovement(rs, best_f=Y.max()),
            bounds,
            q=1,
            num_restarts=10,
            raw_samples=512,
            options={"seed": seed},
        )
        assert x_rs.shape == (1, 6) and v_rs.shape == ()
        best_bt, best_rs = max(best_bt, v_bt.item()), max(best_rs, v_rs.item())
        # The returned value is BoTorch's value of the returned candidate.
        with torch.no_grad():
            check = bt_analytic.LogExpectedImprovement(bt, best_f=Y.max())(x_rs.unsqueeze(0))
        assert close(v_rs, check.squeeze())
    assert best_rs >= best_bt - 1e-6 * abs(best_bt), (best_rs, best_bt)


def test_optimize_acqf_qlogei_parity(hartmann_models: tuple) -> None:
    """Joint q = 2 optimization of qLogEI with identical base samples."""
    bt, rs, Y = hartmann_models
    bounds = torch.stack([torch.zeros(6, dtype=DT), torch.ones(6, dtype=DT)])
    Z = torch.randn(128, 2, generator=torch.Generator().manual_seed(0), dtype=DT)
    s_bt = BtSampler(sample_shape=torch.Size([128]), seed=0)
    s_bt.base_samples = Z.view(128, 1, 2).clone()
    s_rs = RsSampler(sample_shape=[128], seed=0)
    s_rs.base_samples = Z
    a_bt = bt_logei.qLogExpectedImprovement(bt, best_f=Y.max(), sampler=s_bt)
    a_rs = rs_logei.qLogExpectedImprovement(rs, best_f=Y.max(), sampler=s_rs)
    best_bt, best_rs = -math.inf, -math.inf
    for seed in range(3):
        _, v_bt = bt_optimize_acqf(
            a_bt, bounds, q=2, num_restarts=10, raw_samples=512, options={"seed": seed}
        )
        x_rs, v_rs = rs_optimize_acqf(
            a_rs, bounds, q=2, num_restarts=10, raw_samples=512, options={"seed": seed}
        )
        best_bt, best_rs = max(best_bt, v_bt.item()), max(best_rs, v_rs.item())
        assert close(v_rs, a_bt(x_rs.unsqueeze(0)).detach().squeeze())
    assert torch.equal(s_bt.base_samples.view(128, 2), Z)
    assert best_rs >= best_bt - 1e-4 * abs(best_bt), (best_rs, best_bt)


def test_optimize_acqf_qlognei_runs(hartmann_models: tuple) -> None:
    bt, rs, Y = hartmann_models
    bounds = torch.stack([torch.zeros(6, dtype=DT), torch.ones(6, dtype=DT)])
    a_rs = rs_logei.qLogNoisyExpectedImprovement(rs, X_baseline=rs.train_inputs[0])
    x, v = rs_optimize_acqf(
        a_rs, bounds, q=1, num_restarts=10, raw_samples=512, options={"seed": 0}
    )
    assert x.shape == (1, 6) and torch.isfinite(v)


TEST_FUNCTIONS = [
    ("Ackley", {"dim": 3}),
    ("Beale", {}),
    ("Branin", {}),
    ("Bukin", {}),
    ("Cosine8", {}),
    ("DixonPrice", {"dim": 3}),
    ("DropWave", {}),
    ("EggHolder", {}),
    ("Griewank", {"dim": 4}),
    ("Hartmann", {"dim": 3}),
    ("Hartmann", {"dim": 4}),
    ("Hartmann", {"dim": 6}),
    ("HolderTable", {}),
    ("Levy", {"dim": 4}),
    ("Michalewicz", {"dim": 5}),
    ("Powell", {"dim": 8}),
    ("Rastrigin", {"dim": 3}),
    ("Rosenbrock", {"dim": 4}),
    ("Shekel", {"m": 7}),
    ("SixHumpCamel", {}),
    ("StyblinskiTang", {"dim": 3}),
    ("ThreeHumpCamel", {}),
]


@pytest.mark.parametrize(
    "name,kwargs", TEST_FUNCTIONS, ids=[f"{n}-{i}" for i, (n, _) in enumerate(TEST_FUNCTIONS)]
)
def test_test_function_parity(name: str, kwargs: dict) -> None:
    f_bt = getattr(bt_synthetic, name)(negate=True, **kwargs)
    f_rs = getattr(rs_synthetic, name)(negate=True, **kwargs)
    assert f_rs.dim == f_bt.dim
    assert close(f_rs.bounds, f_bt.bounds, rtol=0, atol=0)
    try:
        expected = f_bt.optimal_value
    except NotImplementedError:
        with pytest.raises(NotImplementedError):
            _ = f_rs.optimal_value
    else:
        assert f_rs.optimal_value == pytest.approx(expected, rel=1e-15)
    try:
        optimizers = f_bt.optimizers
    except NotImplementedError:
        with pytest.raises(NotImplementedError):
            _ = f_rs.optimizers
    else:
        assert close(f_rs.optimizers, optimizers, rtol=0, atol=0)
    lo, hi = f_bt.bounds
    X = lo + (hi - lo) * torch.rand(
        16, f_bt.dim, generator=torch.Generator().manual_seed(0), dtype=DT
    )
    assert close(f_rs(X, noise=False), f_bt(X, noise=False), rtol=1e-12, atol=1e-12)
    assert close(f_rs.evaluate_true(X), f_bt.evaluate_true(X), rtol=1e-12, atol=1e-12)


def test_input_scaling_warnings_match() -> None:
    cases = [
        make_data(10, 2, seed=1)[:2],
        (make_data(10, 2, seed=1)[0] * 5, make_data(10, 2, seed=1)[1]),
    ]
    for X, Y in cases:
        for outcome_transform in ("default", None):
            kw = {} if outcome_transform == "default" else {"outcome_transform": None}
            with warnings.catch_warnings(record=True) as w_bt:
                warnings.simplefilter("always")
                BtSingleTaskGP(X, Y, **kw)
            with warnings.catch_warnings(record=True) as w_rs:
                warnings.simplefilter("always")
                RsSingleTaskGP(X, Y, **kw)
            bt_msgs = sorted(
                str(w.message)[:40]
                for w in w_bt
                if "InputDataWarning" in w.category.__name__ and "precision" not in str(w.message)
            )
            rs_msgs = sorted(
                str(w.message)[:40] for w in w_rs if issubclass(w.category, RsInputDataWarning)
            )
            assert bt_msgs == rs_msgs


SIGNATURES = [
    (BtSingleTaskGP, RsSingleTaskGP),
    (BtNormalize, RsNormalize),
    (botorch.models.transforms.outcome.Standardize, botorch_rs.models.transforms.Standardize),
    (BtMLL, RsMLL),
    (bt_fit, rs_fit),
    (bt_optimize_acqf, rs_optimize_acqf),
    (BtSampler, RsSampler),
    (bt_prune, rs_prune),
    (botorch.utils.sampling.draw_sobol_samples, botorch_rs.utils.sampling.draw_sobol_samples),
    (
        botorch.utils.sampling.draw_sobol_normal_samples,
        botorch_rs.utils.sampling.draw_sobol_normal_samples,
    ),
    *[
        (getattr(bt_modules, name), getattr(rs_modules, name))
        for name in (
            "get_covar_module_with_dim_scaled_prior",
            "get_matern_kernel_with_gamma_prior",
            "get_gaussian_likelihood_with_lognormal_prior",
            "get_gaussian_likelihood_with_gamma_prior",
        )
    ],
    (gpytorch.kernels.ScaleKernel, rs_kernels.ScaleKernel),
    (gpytorch.priors.GammaPrior, rs_priors.GammaPrior),
    (gpytorch.priors.LogNormalPrior, rs_priors.LogNormalPrior),
    (gpytorch.likelihoods.GaussianLikelihood, botorch_rs.likelihoods.GaussianLikelihood),
    (gpytorch.means.ConstantMean, botorch_rs.means.ConstantMean),
    *[(getattr(bt_analytic, n), getattr(rs_analytic, n)) for n, _ in ANALYTIC],
    *[(getattr(b, n), getattr(r, n)) for n, b, r, _ in MC],
    (bt_logei.qLogNoisyExpectedImprovement, rs_logei.qLogNoisyExpectedImprovement),
    *[(getattr(bt_synthetic, n), getattr(rs_synthetic, n)) for n, _ in TEST_FUNCTIONS],
    # Multi-output models, posterior transforms and constrained EI (phase 4).
    (botorch.models.ModelListGP, botorch_rs.models.ModelListGP),
    (gpytorch.mlls.SumMarginalLogLikelihood, botorch_rs.mlls.SumMarginalLogLikelihood),
    (gpytorch.likelihoods.LikelihoodList, botorch_rs.likelihoods.LikelihoodList),
    (
        botorch.acquisition.objective.ScalarizedPosteriorTransform,
        botorch_rs.acquisition.objective.ScalarizedPosteriorTransform,
    ),
    (bt_analytic.LogConstrainedExpectedImprovement, rs_analytic.LogConstrainedExpectedImprovement),
    (bt_analytic.ConstrainedExpectedImprovement, rs_analytic.ConstrainedExpectedImprovement),
    (bt_analytic.AnalyticAcquisitionFunction, rs_analytic.AnalyticAcquisitionFunction),
    (bt_mc.MCAcquisitionFunction, rs_mc.MCAcquisitionFunction),
    (botorch.models.SingleTaskGP.posterior, botorch_rs.models.SingleTaskGP.posterior),
    (botorch.models.ModelListGP.posterior, botorch_rs.models.ModelListGP.posterior),
    # MC objectives, outcome constraints and constrained optimization (phase 4B).
    *[
        (getattr(botorch.acquisition.objective, n), getattr(botorch_rs.acquisition.objective, n))
        for n in ("LinearMCObjective", "GenericMCObjective", "ConstrainedMCObjective")
    ],
    (
        botorch.acquisition.objective.MCAcquisitionObjective.forward,
        botorch_rs.acquisition.objective.MCAcquisitionObjective.forward,
    ),
    (bt_logei.qLogProbabilityOfFeasibility, rs_logei.qLogProbabilityOfFeasibility),
    *[
        (getattr(botorch.optim.initializers, n), getattr(botorch_rs.optim.initializers, n))
        for n in (
            "gen_batch_initial_conditions",
            "sample_q_batches_from_polytope",
            "transform_constraints",
        )
    ],
    (
        botorch.optim.parameter_constraints.evaluate_feasibility,
        botorch_rs.optim.parameter_constraints.evaluate_feasibility,
    ),
    *[
        (getattr(botorch.utils.sampling, n), getattr(botorch_rs.utils.sampling, n))
        for n in (
            "get_polytope_samples",
            "HitAndRunPolytopeSampler",
            "find_interior_point",
            "sample_polytope",
            "sparse_to_dense_constraints",
        )
    ],
    (
        botorch.utils.sampling.HitAndRunPolytopeSampler.draw,
        botorch_rs.utils.sampling.HitAndRunPolytopeSampler.draw,
    ),
]


@pytest.mark.parametrize(
    "bt_obj,rs_obj", SIGNATURES, ids=[f"{r.__name__}-{i}" for i, (_, r) in enumerate(SIGNATURES)]
)
def test_signatures_match_botorch(bt_obj: Any, rs_obj: Any) -> None:
    """BoTorch's parameters appear with the same names, order and (plain)
    defaults; botorch_rs only adds keyword-only extensions."""
    import inspect

    def params(obj: Any) -> list[inspect.Parameter]:
        sig = inspect.signature(obj.__init__ if inspect.isclass(obj) else obj)
        return [p for p in sig.parameters.values() if p.name != "self"]

    bt_params = [p for p in params(bt_obj) if p.kind not in (p.VAR_KEYWORD, p.VAR_POSITIONAL)]
    rs_params = params(rs_obj)
    rs_names = [p.name for p in rs_params]
    positional = [p for p in bt_params if p.kind != p.KEYWORD_ONLY]
    assert rs_names[: len(positional)] == [p.name for p in positional]
    plain = (type(None), bool, int, float, str)
    for p in bt_params:
        assert p.name in rs_names, p.name
        q = rs_params[rs_names.index(p.name)]
        assert q.kind == p.kind or p.kind == p.POSITIONAL_OR_KEYWORD, p.name
        if isinstance(p.default, plain) and p.default is not p.empty:
            assert q.default == p.default, (p.name, p.default, q.default)
    bt_all = params(bt_obj)
    for p in bt_all:
        if p.kind == p.VAR_POSITIONAL:  # e.g. ModelListGP(*gp_models)
            assert any(q.kind == q.VAR_POSITIONAL and q.name == p.name for q in rs_params), p.name
    for q in rs_params:
        if q.name not in [p.name for p in bt_all]:
            assert q.kind in (q.KEYWORD_ONLY, q.VAR_KEYWORD) or "kwargs" in [
                p.name for p in bt_all
            ], q.name


def test_botorch_rs_namespace() -> None:
    assert botorch_rs.SingleTaskGP is RsSingleTaskGP
    assert botorch_rs.optimize_acqf is rs_optimize_acqf

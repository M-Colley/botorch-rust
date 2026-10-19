"""Live parity of phase 10c (approximate GPs) with BoTorch:
``SingleTaskVariationalGP`` / ``ApproximateGPyTorchModel``, GPyTorch's
``VariationalELBO`` and the inducing point allocators of
``botorch.models.utils.inducing_point_allocators``.

Every case of ``approximate_helpers.CASES`` compares the construction (the
inducing points chosen by the greedy pivoted-Cholesky allocation, parameter
names, shapes, bounds and initial values), the ELBO terms and the ELBO's
gradient at random parameters, the latent ``q(f)`` of ``model.model(X)``,
posteriors (with and without observation noise, batched, posterior
transforms, samples from given base samples) and acquisition functions
(analytic and Monte Carlo with identical base samples, values and input
gradients). Fits are compared on twins (``parity_helpers.compare_twin_fits``);
Adam fits (``fit_gpytorch_mll_torch``) step by step.

Tolerances: values 1e-9, gradients 1e-8 (relative). The randomized sweep is
in ``test_equivalence_10c_approximate.py``.
"""

from __future__ import annotations

import warnings

import pytest

from helpers import importorskip_botorch

importorskip_botorch()
torch = pytest.importorskip("torch")

from approximate_helpers import (  # noqa: E402
    CASES,
    DT,
    GRAD_TOL,
    VALUE_TOL,
    Case,
    bt_elbo_and_grad,
    build_pair,
    copy_parameters,
    elbo_pair,
    make_data,
    set_random_parameters,
    strategy,
)
from botorch.acquisition import analytic as bt_an  # noqa: E402
from botorch.acquisition import logei as bt_logei  # noqa: E402
from botorch.acquisition import monte_carlo as bt_mc  # noqa: E402
from botorch.acquisition.objective import (  # noqa: E402
    ScalarizedPosteriorTransform as BtScalarized,
)
from botorch.models import SingleTaskGP as BtSingleTaskGP  # noqa: E402
from botorch.models import SingleTaskVariationalGP as BtSVGP  # noqa: E402
from botorch.models.approximate_gp import (  # noqa: E402
    ApproximateGPyTorchModel as BtApproximate,
)
from botorch.models.utils import inducing_point_allocators as bt_ipa  # noqa: E402
from botorch.optim.utils import get_parameters_and_bounds as bt_params_bounds  # noqa: E402
from parity_helpers import (  # noqa: E402
    Checker,
    compare_twin_fits,
    rel_err,
    sampler_pair,
    twin_inputs,
    values_and_grads,
)

import botorch_rs.acquisition.analytic as rs_an  # noqa: E402
import botorch_rs.acquisition.logei as rs_logei  # noqa: E402
import botorch_rs.acquisition.monte_carlo as rs_mc  # noqa: E402
from botorch_rs.acquisition.objective import (  # noqa: E402
    ScalarizedPosteriorTransform as RsScalarized,
)
from botorch_rs.models import SingleTaskGP as RsSingleTaskGP  # noqa: E402
from botorch_rs.models import SingleTaskVariationalGP as RsSVGP  # noqa: E402
from botorch_rs.models.approximate_gp import (  # noqa: E402
    ApproximateGPyTorchModel as RsApproximate,
)
from botorch_rs.models.utils import inducing_point_allocators as rs_ipa  # noqa: E402
from botorch_rs.optim.utils import get_parameters_and_bounds as rs_params_bounds  # noqa: E402

pytestmark = pytest.mark.filterwarnings("ignore")

check = Checker("10c approximate")
IDS = [c.name for c in CASES]


def _mll_names(mll: object) -> list[str]:
    return [n for n, _ in mll.named_parameters()]  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Inducing point allocators
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", range(4))
def test_pivoted_cholesky_init(seed: int) -> None:
    g = torch.Generator().manual_seed(seed)
    n = 12 + 3 * seed
    X = torch.rand(n, 2, generator=g, dtype=DT)
    if seed == 3:
        X = torch.cat([X, X[:5]])  # duplicates: the selection stops early
    from gpytorch.kernels import MaternKernel

    from botorch_rs.kernels import MaternKernel as RsMatern

    kb, kr = MaternKernel(nu=1.5).to(DT), RsMatern(nu=1.5)
    raw = torch.tensor([[-0.9]], dtype=DT)
    kb.raw_lengthscale.data = raw.clone()
    kr.raw_lengthscale = raw.clone()
    with torch.no_grad():
        K = kb(X).to_dense()
    check.check("kernel matrix of the allocation", kr(X), K, VALUE_TOL)
    quality = torch.rand(X.shape[0], generator=g, dtype=DT) + 0.1
    for max_length in (1, 5, X.shape[0]):
        for q in (torch.ones(X.shape[0], dtype=DT), quality):
            zb = bt_ipa._pivoted_cholesky_init(X, K, max_length, q)
            zr = rs_ipa._pivoted_cholesky_init(X, K, max_length, q)
            assert tuple(torch.as_tensor(zr).shape) == tuple(zb.shape)
            check.check("_pivoted_cholesky_init: selected points", zr, zb, 0.0)


@pytest.mark.parametrize("maximize", [True, False])
def test_greedy_improvement_reduction(maximize: bool) -> None:
    g = torch.Generator().manual_seed(11)
    X = torch.rand(14, 2, generator=g, dtype=DT)
    Y = torch.sin(5 * X[:, :1]) + X[:, 1:]
    prev_b, prev_r = BtSingleTaskGP(X, Y), RsSingleTaskGP(X, Y)
    Xn = torch.rand(20, 2, generator=g, dtype=DT)
    qb = bt_ipa.ExpectedImprovementQualityFunction(prev_b, maximize)(Xn)
    qr = rs_ipa.ExpectedImprovementQualityFunction(prev_r, maximize)(Xn)
    check.check("ExpectedImprovementQualityFunction", qr, qb, VALUE_TOL)
    check.check(
        "UnitQualityFunction", rs_ipa.UnitQualityFunction()(Xn), bt_ipa.UnitQualityFunction()(Xn), 0
    )
    Yn = torch.cos(3 * Xn[:, :1])
    zb = BtSVGP(Xn, Yn, inducing_points=6, inducing_point_allocator=bt_ipa.GreedyImprovementReduction(prev_b, maximize))
    zr = RsSVGP(Xn, Yn, inducing_points=6, inducing_point_allocator=rs_ipa.GreedyImprovementReduction(prev_r, maximize))
    check.check(
        "GreedyImprovementReduction: inducing points",
        strategy(zr).inducing_points,
        strategy(zb).inducing_points.detach(),
        0.0,
    )
    # init_inducing_points with the fitted kernel.
    X3 = torch.rand(20, 2, generator=g, dtype=DT)
    check.check(
        "init_inducing_points (GreedyImprovementReduction)",
        zr.init_inducing_points(X3),
        zb.init_inducing_points(X3),
        0.0,
    )


def test_allocator_cases() -> None:
    """The multi-task case (a kernel with a batch shape) and batched inputs."""
    from gpytorch.kernels import RBFKernel

    from botorch_rs.kernels import RBFKernel as RsRBF

    g = torch.Generator().manual_seed(3)
    X = torch.rand(16, 2, generator=g, dtype=DT)
    kb = RBFKernel(batch_shape=torch.Size([2]), ard_num_dims=2).to(DT)
    kr = RsRBF(batch_shape=(2,), ard_num_dims=2)
    lb = torch.tensor([[[0.2, 0.5]], [[0.7, 0.1]]], dtype=DT)
    kb.raw_lengthscale.data = lb.clone()
    kr.raw_lengthscale = lb.clone()
    alloc_b, alloc_r = bt_ipa.GreedyVarianceReduction(), rs_ipa.GreedyVarianceReduction()
    zb = alloc_b.allocate_inducing_points(X, kb, 5, torch.Size([]))
    zr = alloc_r.allocate_inducing_points(X, kr, 5, ())
    check.check("allocation, multi-task case (first kernel)", zr, zb, 0.0)
    Xb = torch.rand(2, 16, 2, generator=g, dtype=DT)
    kb1, kr1 = RBFKernel().to(DT), RsRBF()
    zb = alloc_b.allocate_inducing_points(Xb, kb1, 4, torch.Size([2]))
    zr = alloc_r.allocate_inducing_points(Xb, kr1, 4, (2,))
    check.check("allocation, batched inputs", zr, zb, 0.0)
    zb = alloc_b.allocate_inducing_points(X, kb1, 4, torch.Size([3]))
    zr = alloc_r.allocate_inducing_points(X, kr1, 4, (3,))
    check.check("allocation, unbatched inputs expanded to the batch", zr, zb, 0.0)


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("case", CASES, ids=IDS)
def test_construction(case: Case) -> None:
    bt, rs, X, Y = build_pair(case)
    check.check("inducing points (allocation)", strategy(rs).inducing_points, strategy(bt).inducing_points.detach(), 0.0)
    mb, mr = elbo_pair(bt, rs, case.n)
    assert _mll_names(mr) == _mll_names(mb)
    assert [n for n, _ in rs.named_parameters()] == [n for n, _ in bt.named_parameters()]
    rs_named = dict(mr.named_parameters())
    for name, p in mb.named_parameters():
        assert tuple(torch.as_tensor(rs_named[name]).shape) == tuple(p.shape), name
        if "variational" not in name:
            check.check("initial raw parameters", rs_named[name], p.detach(), 0.0)
    _, bounds_b = bt_params_bounds(mb)
    _, bounds_r = rs_params_bounds(mr)
    for name in _mll_names(mb):
        def norm(b: tuple[object, object]) -> tuple[float, float]:
            lo, hi = b
            return (
                -float("inf") if lo is None else float(lo),  # type: ignore[arg-type]
                float("inf") if hi is None else float(hi),  # type: ignore[arg-type]
            )

        assert norm(bounds_r.get(name, (None, None))) == norm(bounds_b.get(name, (None, None))), name
    assert rs.num_outputs == bt.num_outputs
    assert rs.batch_shape == tuple(bt.batch_shape)
    check.check("train_targets", rs.model.train_targets, bt.model.train_targets, 0.0)
    check.check("train_inputs (transformed)", rs.model.train_inputs[0], bt.model.train_inputs[0], 0.0)


@pytest.mark.parametrize("seed", [0, 7])
def test_variational_initialization_draws(seed: int) -> None:
    """GPyTorch draws the variational mean at the first evaluation from
    torch's global generator; botorch_rs draws the same values."""
    case = CASES[5]
    bt, rs, X, _ = build_pair(case)
    torch.manual_seed(seed)
    out_b = bt.model(X)
    torch.manual_seed(seed)
    out_r = rs.model(X)
    vb, vr = strategy(bt)._variational_distribution, strategy(rs)._variational_distribution
    check.check("variational mean after initialization", vr.variational_mean, vb.variational_mean.detach(), 0.0)
    check.check("chol_variational_covar after initialization", vr.chol_variational_covar, vb.chol_variational_covar.detach(), 0.0)
    check.check("q(f) mean at initialization", out_r.mean, out_b.mean.detach(), VALUE_TOL)
    assert int(strategy(rs).variational_params_initialized.item()) == 1


# ---------------------------------------------------------------------------
# ELBO
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("case", CASES, ids=IDS)
@pytest.mark.parametrize("beta", [1.0, 0.3])
def test_elbo(case: Case, beta: float) -> None:
    bt, rs, X, Y = build_pair(case)
    for seed in range(2):
        set_random_parameters(bt, rs, seed)
        num_data = case.n + 7 * seed
        mb, mr = elbo_pair(bt, rs, num_data, beta=beta, combine=False)
        bt.train()
        tb = mb(bt.model(*bt.model.train_inputs), bt.model.train_targets)
        tr = mr()
        for label, a, e in zip(("expected log likelihood", "KL divergence", "log prior"), tr, tb):
            check.check(f"ELBO term: {label}", a, e.detach(), VALUE_TOL)
        mb, mr = elbo_pair(bt, rs, num_data, beta=beta)
        vb, gb = bt_elbo_and_grad(mb)
        vr, gr = mr.value_and_grad()
        check.check("ELBO", vr, vb, VALUE_TOL)
        check.check("ELBO gradient", gr, gb, GRAD_TOL)
        check.check("ELBO via mll()", mr(), vb, VALUE_TOL)
        # Other data: mll(model.model(X2), y2).
        g = torch.Generator().manual_seed(100 + seed)
        X2 = torch.rand(5, case.d, generator=g, dtype=DT)
        y2 = torch.randn(5, case.m, generator=g, dtype=DT).squeeze(-1)
        bt.train()
        vb2 = mb(bt.model(X2), y2)
        vr2 = mr(rs.model(X2), y2)
        check.check("ELBO on other data", vr2, vb2.detach(), VALUE_TOL)


@pytest.mark.parametrize("case", CASES, ids=IDS)
def test_latent_and_kl(case: Case) -> None:
    bt, rs, X, Y = build_pair(case)
    set_random_parameters(bt, rs, 3)
    g = torch.Generator().manual_seed(4)
    Xt = torch.rand(6, case.d, generator=g, dtype=DT)
    for mode in ("train", "eval"):
        getattr(bt.model, mode)()
        getattr(rs.model, mode)()
        ob, orr = bt.model(Xt), rs.model(Xt)
        check.check("q(f) mean", orr.mean, ob.mean.detach(), VALUE_TOL)
        check.check("q(f) variance", orr.variance, ob.variance.detach(), VALUE_TOL)
        check.check("q(f) covariance", orr.covariance_matrix, ob.covariance_matrix.detach(), VALUE_TOL)
    kb = bt.model.variational_strategy.kl_divergence()
    kr = rs.model.variational_strategy.kl_divergence()
    check.check("kl_divergence()", kr, kb.detach(), VALUE_TOL)


# ---------------------------------------------------------------------------
# Posteriors and acquisition functions
# ---------------------------------------------------------------------------


def _test_points(case: Case, shape: tuple[int, ...], seed: int) -> object:
    g = torch.Generator().manual_seed(seed)
    X = case.scale * torch.rand(*shape, case.d, generator=g, dtype=DT)
    return X + 1.0 if case.input_tf else X


@pytest.mark.parametrize("case", CASES, ids=IDS)
def test_posterior(case: Case) -> None:
    bt, rs, X, Y = build_pair(case)
    set_random_parameters(bt, rs, 5)
    for shape in ((4,), (3, 2), (2, 1)):
        Xt = _test_points(case, shape, 6)
        for noise in (False, True):
            pb = bt.posterior(Xt, observation_noise=noise)
            pr = rs.posterior(Xt, observation_noise=noise)
            check.check("posterior mean", pr.mean, pb.mean.detach(), VALUE_TOL)
            check.check("posterior variance", pr.variance, pb.variance.detach(), VALUE_TOL)
            check.check(
                "posterior covariance",
                pr.distribution.covariance_matrix,
                pb.distribution.covariance_matrix.detach(),
                VALUE_TOL,
            )
        pb, pr = bt.posterior(Xt), rs.posterior(Xt)
        Z = torch.randn(3, *pb.base_sample_shape, generator=torch.Generator().manual_seed(1), dtype=DT)
        check.check(
            "samples from base samples",
            pr.rsample_from_base_samples(torch.Size([3]), Z),
            pb.rsample_from_base_samples(torch.Size([3]), Z).detach(),
            VALUE_TOL,
        )
    if case.m > 1:
        w = torch.linspace(0.5, 1.5, case.m, dtype=DT)
        Xt = _test_points(case, (5,), 8)
        pb = bt.posterior(Xt, posterior_transform=BtScalarized(w))
        pr = rs.posterior(Xt, posterior_transform=RsScalarized(w))
        check.check("posterior (ScalarizedPosteriorTransform) mean", pr.mean, pb.mean.detach(), VALUE_TOL)
        check.check("posterior (ScalarizedPosteriorTransform) variance", pr.variance, pb.variance.detach(), VALUE_TOL)


@pytest.mark.parametrize("case", CASES, ids=IDS)
def test_acquisition(case: Case) -> None:
    bt, rs, X, Y = build_pair(case)
    set_random_parameters(bt, rs, 9)
    bt.eval()
    Xq = _test_points(case, (4, 1), 10)
    w = torch.linspace(0.5, 1.5, case.m, dtype=DT)
    tb = BtScalarized(w) if case.m > 1 else None
    tr = RsScalarized(w) if case.m > 1 else None
    best = torch.tensor(float(Y.mean()), dtype=DT) if case.m == 1 else torch.tensor(float((Y * w).sum(-1).mean()), dtype=DT)
    pairs = [
        ("LogExpectedImprovement", bt_an.LogExpectedImprovement(bt, best_f=best, posterior_transform=tb), rs_an.LogExpectedImprovement(rs, best_f=best, posterior_transform=tr)),
        ("PosteriorMean", bt_an.PosteriorMean(bt, posterior_transform=tb), rs_an.PosteriorMean(rs, posterior_transform=tr)),
        ("UpperConfidenceBound", bt_an.UpperConfidenceBound(bt, beta=torch.tensor(0.7, dtype=DT), posterior_transform=tb), rs_an.UpperConfidenceBound(rs, beta=torch.tensor(0.7, dtype=DT), posterior_transform=tr)),
    ]
    for name, ab, ar in pairs:
        vb, gb = values_and_grads(ab, Xq)
        vr, gr = ar.value_and_grad(Xq)
        check.check(f"{name} values", vr, vb, VALUE_TOL)
        check.check(f"{name} input gradients", gr, gb, GRAD_TOL)
    Xq = _test_points(case, (3, 3), 11)
    m_eff = 1  # the scalarized posterior of several outputs has one output
    sb, sr, _ = sampler_pair(32, 3, m_eff, 12)
    sb2, sr2, _ = sampler_pair(32, 3 + case.n, m_eff, 13)
    pairs = [
        ("qExpectedImprovement", bt_mc.qExpectedImprovement(bt, best_f=best, sampler=sb, posterior_transform=tb), rs_mc.qExpectedImprovement(rs, best_f=best, sampler=sr, posterior_transform=tr)),
        ("qUpperConfidenceBound", bt_mc.qUpperConfidenceBound(bt, beta=0.4, sampler=sb, posterior_transform=tb), rs_mc.qUpperConfidenceBound(rs, beta=0.4, sampler=sr, posterior_transform=tr)),
        ("qLogNoisyExpectedImprovement", bt_logei.qLogNoisyExpectedImprovement(bt, X_baseline=X, sampler=sb2, posterior_transform=tb, prune_baseline=False, cache_root=False), rs_logei.qLogNoisyExpectedImprovement(rs, X_baseline=X, sampler=sr2, posterior_transform=tr, prune_baseline=False, cache_root=False)),
    ]
    for name, ab, ar in pairs:
        vb, gb = values_and_grads(ab, Xq)
        vr, gr = ar.value_and_grad(Xq)
        check.check(f"{name} values", vr, vb, VALUE_TOL)
        check.check(f"{name} input gradients", gr, gb, GRAD_TOL)


# ---------------------------------------------------------------------------
# Fits
# ---------------------------------------------------------------------------


def _bt_fit(bt: object, n: int) -> tuple[float, int]:
    import botorch.optim.fit as of
    from botorch.fit import fit_gpytorch_mll
    from gpytorch.mlls import VariationalELBO

    record: dict[str, object] = {"attempts": 0}
    orig = of.fit_gpytorch_mll_scipy

    def recorder(mll: object, *args: object, **kwargs: object) -> object:
        record["attempts"] = int(record["attempts"]) + 1  # type: ignore[call-overload]
        record["r"] = orig(mll, *args, **kwargs)
        return record["r"]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit_gpytorch_mll(VariationalELBO(bt.likelihood, bt.model, num_data=n), optimizer=recorder)  # type: ignore[attr-defined]
    return float(record["r"].fval), int(record["attempts"])  # type: ignore[attr-defined]


def _rs_fit(rs: object, n: int) -> tuple[float, int]:
    from botorch_rs.fit import fit_gpytorch_mll
    from botorch_rs.mlls import VariationalELBO

    mll = VariationalELBO(rs.likelihood, rs.model, num_data=n)  # type: ignore[attr-defined]
    fit_gpytorch_mll(mll)
    return float(mll._fit_result["fval"]), int(mll._fit_result["attempts"])


FIT_CASES = [
    Case("fit_fixed_inducing", n=16, inducing=4, learn=False, seed=21),
    Case("fit_default", n=12, inducing=3, seed=22),
    Case("fit_two_outputs", n=12, m=2, inducing=3, learn=False, seed=23),
]


@pytest.mark.parametrize("case", FIT_CASES, ids=[c.name for c in FIT_CASES])
def test_fit(case: Case) -> None:
    X, Y = make_data(case)

    def fit_twin(t: int) -> tuple[tuple[float, int], tuple[float, int]]:
        Xt = twin_inputs(X, t)
        torch.manual_seed(t)
        bt, rs, _, _ = build_pair(case, Xt, Y)
        torch.manual_seed(1000 + t)
        b = _bt_fit(bt, case.n)
        torch.manual_seed(1000 + t)
        r = _rs_fit(rs, case.n)
        return b, r

    compare_twin_fits(check, f"fit {case.name}", fit_twin, twins=4)


@pytest.mark.parametrize("case", [CASES[0], CASES[5]], ids=["one_output", "two_outputs"])
def test_fit_torch(case: Case) -> None:
    """Adam (``fit_gpytorch_mll_torch``) step by step, directly and as
    ``fit_gpytorch_mll``'s path for more than ``full_batch_limit`` points."""
    from botorch.fit import fit_gpytorch_mll as bt_fit_mll
    from botorch.optim.fit import fit_gpytorch_mll_torch as bt_torch

    from botorch_rs.fit import fit_gpytorch_mll as rs_fit_mll
    from botorch_rs.optim.fit import fit_gpytorch_mll_torch as rs_torch

    bt, rs, X, Y = build_pair(case)
    set_random_parameters(bt, rs, 1)
    mb, mr = elbo_pair(bt, rs, case.n)
    rb = bt_torch(mb, step_limit=40)
    rr = rs_torch(mr, step_limit=40)
    assert rb.step == rr.step
    check.check("fit_gpytorch_mll_torch: final loss", rr.fval, float(rb.fval), VALUE_TOL)
    rs_named = dict(mr.named_parameters())
    for name, p in mb.named_parameters():
        check.check("fit_gpytorch_mll_torch: parameters", rs_named[name], p.detach(), 1e-8)
    bt, rs, X, Y = build_pair(case)
    mb, mr = elbo_pair(bt, rs, case.n)
    torch.manual_seed(2)
    bt_fit_mll(mb, full_batch_limit=5, optimizer_kwargs={"step_limit": 30})
    torch.manual_seed(2)
    rs_fit_mll(mr, full_batch_limit=5, optimizer_kwargs={"step_limit": 30})
    rs_named = dict(mr.named_parameters())
    for name, p in mb.named_parameters():
        check.check("fit_gpytorch_mll (Adam path): parameters", rs_named[name], p.detach(), 1e-8)


# ---------------------------------------------------------------------------
# Further API
# ---------------------------------------------------------------------------


def test_init_inducing_points_and_reinitialization() -> None:
    case = CASES[0]
    bt, rs, X, Y = build_pair(case)
    set_random_parameters(bt, rs, 2)
    X2 = torch.rand(15, 2, generator=torch.Generator().manual_seed(3), dtype=DT)
    check.check("init_inducing_points", rs.init_inducing_points(X2), bt.init_inducing_points(X2), 0.0)
    assert int(strategy(rs).variational_params_initialized.item()) == 0
    torch.manual_seed(4)
    pb = bt.posterior(X2[:3])
    torch.manual_seed(4)
    pr = rs.posterior(X2[:3])
    check.check("posterior after re-initialization", pr.mean, pb.mean.detach(), VALUE_TOL)


@pytest.mark.parametrize("num_outputs", [1, 2])
def test_approximate_gpytorch_model(num_outputs: int) -> None:
    X = torch.rand(10, 2, generator=torch.Generator().manual_seed(5), dtype=DT)
    torch.manual_seed(1)
    bt = BtApproximate(train_X=X, num_outputs=num_outputs, inducing_points=4).to(X)
    rs = RsApproximate(train_X=X, num_outputs=num_outputs, inducing_points=4)
    assert [n for n, _ in rs.named_parameters()] == [n for n, _ in bt.named_parameters()]
    for name, p in bt.named_parameters():
        if "variational" not in name:
            check.check("ApproximateGPyTorchModel: initial parameters", dict(rs.named_parameters())[name], p.detach(), 0.0)
    torch.manual_seed(2)
    pb = bt.posterior(X[:3])
    torch.manual_seed(2)
    pr = rs.posterior(X[:3])
    check.check("ApproximateGPyTorchModel: posterior mean", pr.mean, pb.mean.detach(), VALUE_TOL)
    check.check("ApproximateGPyTorchModel: posterior variance", pr.variance, pb.variance.detach(), VALUE_TOL)


def test_errors_match_botorch() -> None:
    from botorch.sampling import SobolQMCNormalSampler

    case = CASES[0]
    bt, rs, X, Y = build_pair(case)
    for model in (bt, rs):
        with pytest.raises(NotImplementedError, match="does not support output indices"):
            model.posterior(X[:2], output_indices=[0])
        with pytest.raises(AttributeError, match="get_fantasy_model"):
            model.condition_on_observations(X[:2], Y[:2])
        with pytest.raises(AttributeError, match="fantasize"):
            model.fantasize(X[:2], SobolQMCNormalSampler(torch.Size([2])))
    for cls in (BtSVGP, RsSVGP):
        with pytest.warns(UserWarning, match="set `learn_inducing_points` to False"):
            alloc = bt_ipa.GreedyVarianceReduction() if cls is BtSVGP else rs_ipa.GreedyVarianceReduction()
            cls(X, Y, inducing_point_allocator=alloc)
    with pytest.raises(ValueError, match="requires a quality score"):
        bt_ipa._pivoted_cholesky_init(X, torch.eye(12, dtype=DT), 3, torch.ones(5, dtype=DT))
    with pytest.raises(ValueError, match="requires a quality score"):
        rs_ipa._pivoted_cholesky_init(X, torch.eye(12, dtype=DT), 3, torch.ones(5, dtype=DT))
    with pytest.raises(NotImplementedError, match="Multi-output models"):
        bt_ipa.ExpectedImprovementQualityFunction(BtSingleTaskGP(X, torch.cat([Y, Y], -1)), True)
    with pytest.raises(NotImplementedError, match="Multi-output models"):
        rs_ipa.ExpectedImprovementQualityFunction(RsSingleTaskGP(X, torch.cat([Y, Y], -1)), True)
    err = rel_err(torch.tensor([1.0]), torch.tensor([1.0]))
    assert err == 0.0

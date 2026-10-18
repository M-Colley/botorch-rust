"""Parity of the preference-learning components of phase 10b with BoTorch,
through the public APIs of both libraries side by side:

* ``PairwiseGP`` (probit and logit likelihoods, BoTorch's default and custom
  ``ScaleKernel`` modules, duplicates that are consolidated, ``Normalize``,
  ``jitter`` / ``xtol`` / ``maxfev`` / consolidation options, batched data),
  built after the same ``np.random.seed`` (the same initial utility): the
  consolidated data, parameter names, shapes and values, the Laplace state
  (``_x0``, ``utility``, ``likelihood_hess``, ``covar``, ``covar_chol``,
  ``covar_inv``), the Laplace MLL and its gradient at the default and at
  random parameters (each warm-started from the previous solution, as in
  BoTorch's training forward pass), posteriors (``q = 1`` batches, joint,
  batched inputs, a posterior transform), the input gradients of
  ``PosteriorMean`` / ``PosteriorStandardDeviation`` and of ``qSimpleRegret``
  with identical base samples, fits, ``condition_on_observations`` and
  ``fantasize`` with the pairwise samplers (BoTorch's orientation bug fixed,
  see ``preference_helpers.fixed_pairwise_sampler``), ``load_state_dict``,
  the likelihood functions and ``PairwiseGP`` in a ``ModelListGP``;
* the acquisition functions: analytic EUBO (pairs, a previous winner, a
  ``PosteriorMeanModel`` / ``AffineDeterministicModel`` outcome model),
  qEUBO (identical base samples, pending points, an outcome model), BALD
  (identical draws: the same torch seed), ``PairwiseMCPosteriorVariance``
  (a probit link), ``LearnedObjective`` (its ``IIDNormalSampler`` draws the
  same base samples from torch; values and autograd), and the samplers.

Tolerances: values 1e-9 (relative), gradients 1e-8 (relative to the largest
component). The fsolve solution ``_x0`` is compared to the solver's
tolerance ``xtol`` (BoTorch's own solutions move by up to 3.5e-9 between
twins whose datapoints differ at 1e-14 when ``cond(K)`` is 1e10), the
inverse covariance to ``cond(K) · eps``; fits on twins as in
``parity_helpers.compare_twin_fits``. The randomized sweep is in
``test_equivalence_10b_preference.py``.
"""

from __future__ import annotations

import math
import warnings

import pytest

from helpers import importorskip_botorch

importorskip_botorch()
torch = pytest.importorskip("torch")
np = pytest.importorskip("numpy")

from botorch.acquisition import active_learning as bt_al  # noqa: E402
from botorch.acquisition import analytic as bt_an  # noqa: E402
from botorch.acquisition import monte_carlo as bt_mc  # noqa: E402
from botorch.acquisition import preference as bt_pref  # noqa: E402
from botorch.acquisition.input_constructors import (  # noqa: E402
    get_acqf_input_constructor as bt_get_ctor,
)
from botorch.acquisition.objective import GenericMCObjective as BtGeneric  # noqa: E402
from botorch.acquisition.objective import LearnedObjective as BtLearned  # noqa: E402
from botorch.acquisition.objective import (  # noqa: E402
    ScalarizedPosteriorTransform as BtScalarized,
)
from botorch.models import ModelListGP as BtModelListGP  # noqa: E402
from botorch.models import SingleTaskGP as BtSTGP  # noqa: E402
from botorch.models.deterministic import AffineDeterministicModel as BtAffine  # noqa: E402
from botorch.models.deterministic import FixedSingleSampleModel as BtFSSM  # noqa: E402
from botorch.models.deterministic import PosteriorMeanModel as BtPMM  # noqa: E402
from botorch.models.likelihoods.pairwise import PairwiseLogitLikelihood as BtLogit  # noqa: E402
from botorch.models.likelihoods.pairwise import PairwiseProbitLikelihood as BtProbit  # noqa: E402
from botorch.sampling.normal import IIDNormalSampler as BtIID  # noqa: E402
from botorch.sampling.normal import SobolQMCNormalSampler as BtSampler  # noqa: E402
from botorch.sampling.pairwise_samplers import (  # noqa: E402
    PairwiseIIDNormalSampler as BtPIID,
)
from botorch.sampling.pairwise_samplers import (  # noqa: E402
    PairwiseSobolQMCNormalSampler as BtPSobol,
)
from kernel_tree_helpers import constraint, kernel, prior, sub  # noqa: E402
from preference_helpers import (  # noqa: E402
    DT,
    GRAD_TOL,
    VALUE_TOL,
    Case,
    bt_fit,
    bt_mll_grad,
    build_pair,
    compare_state,
    fixed_pairwise_sampler,
    grad_err,
    make_data,
    random_points,
    raw_params,
    rs_fit,
    rs_mll_grad,
    set_raw,
    xtol_of,
)

from botorch_rs.acquisition import active_learning as rs_al  # noqa: E402
from botorch_rs.acquisition import analytic as rs_an  # noqa: E402
from botorch_rs.acquisition import monte_carlo as rs_mc  # noqa: E402
from botorch_rs.acquisition import preference as rs_pref  # noqa: E402
from botorch_rs.acquisition.input_constructors import (  # noqa: E402
    get_acqf_input_constructor as rs_get_ctor,
)
from botorch_rs.acquisition.objective import GenericMCObjective as RsGeneric  # noqa: E402
from botorch_rs.acquisition.objective import LearnedObjective as RsLearned  # noqa: E402
from botorch_rs.acquisition.objective import (  # noqa: E402
    ScalarizedPosteriorTransform as RsScalarized,
)
from botorch_rs.models import ModelListGP as RsModelListGP  # noqa: E402
from botorch_rs.models import SingleTaskGP as RsSTGP  # noqa: E402
from botorch_rs.models.deterministic import AffineDeterministicModel as RsAffine  # noqa: E402
from botorch_rs.models.deterministic import FixedSingleSampleModel as RsFSSM  # noqa: E402
from botorch_rs.models.deterministic import PosteriorMeanModel as RsPMM  # noqa: E402
from botorch_rs.models.likelihoods.pairwise import PairwiseLogitLikelihood as RsLogit  # noqa: E402
from botorch_rs.models.likelihoods.pairwise import (  # noqa: E402
    PairwiseProbitLikelihood as RsProbit,
)
from botorch_rs.sampling import IIDNormalSampler as RsIID  # noqa: E402
from botorch_rs.sampling import PairwiseIIDNormalSampler as RsPIID  # noqa: E402
from botorch_rs.sampling import PairwiseSobolQMCNormalSampler as RsPSobol  # noqa: E402
from botorch_rs.sampling import SobolQMCNormalSampler as RsSampler  # noqa: E402
from parity_helpers import Checker, compare_twin_fits, twin_inputs  # noqa: E402

pytestmark = pytest.mark.filterwarnings("ignore")

check = Checker("10b preference")

MATERN = kernel(
    "ScaleKernel",
    sub(
        kernel(
            "MaternKernel", nu=2.5, ard_num_dims=2, lengthscale_prior=prior("GammaPrior", 3.0, 6.0)
        )
    ),
    outputscale_prior=prior("GammaPrior", 2.0, 0.15),
)
RBF_SHARED = kernel(
    "ScaleKernel", sub(kernel("RBFKernel", lengthscale_constraint=constraint("GreaterThan", 1e-3)))
)
SUM3 = kernel(
    "ScaleKernel",
    sub(
        kernel(
            "AdditiveKernel",
            sub(kernel("RBFKernel", ard_num_dims=2, active_dims=[0, 1])),
            sub(kernel("MaternKernel", nu=1.5, active_dims=[2])),
        )
    ),
)

CASES = [
    Case(8, 2, 12),
    Case(10, 3, 16, likelihood="logit"),
    Case(9, 2, 14, dups=True),
    Case(9, 2, 14, likelihood="logit", dups=True),
    Case(25, 3, 40),
    Case(10, 2, 15, kernel=MATERN),
    Case(9, 2, 12, likelihood="logit", kernel=RBF_SHARED),
    Case(10, 3, 14, kernel=SUM3),
    Case(8, 2, 10, normalize=True, scale=5.0),
    Case(8, 2, 12, options={"jitter": 1e-5, "xtol": 1e-4, "maxfev": 8, "consolidate_atol": 0.05}),
    Case(8, 2, 12, likelihood="probit_explicit", options={"consolidate_rtol": 0.01}),
]


def _ids(cases: list[Case]) -> list[str]:
    return [c.label() for c in cases]


def _pair(case: Case, seed: int = 0) -> tuple[Case, object, object, object, object]:
    X, C = make_data(case, 100 + seed)
    bt, rs = build_pair(case, X, C, seed=seed)
    return case, X, C, bt, rs


# ---------------------------------------------------------------------------
# PairwiseGP
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("case", CASES, ids=_ids(CASES))
def test_construction_and_parameters(case: Case) -> None:
    _, _, _, bt, rs = _pair(case)
    compare_state(check, bt, rs, xtol_of(case), "construction")
    names_bt = [(n, tuple(p.shape)) for n, p in bt.named_parameters()]
    names_rs = [(n, tuple(p.shape)) for n, p in rs.named_parameters()]
    assert names_bt == names_rs
    for (_, a), (_, b) in zip(rs.named_parameters(), bt.named_parameters(), strict=True):
        assert torch.equal(torch.as_tensor(a, dtype=DT), b.detach().to(DT))
    check.record("parameter names, shapes and default values", 0.0, 0.0)
    from botorch.optim.utils import get_parameters_and_bounds

    _, bounds = get_parameters_and_bounds(bt)
    expected = []
    for n, p in bt.named_parameters():
        if p.requires_grad:
            lo, hi = bounds.get(n, (None, None))
            expected.extend(
                [
                    (
                        -float("inf") if lo is None else float(lo),
                        float("inf") if hi is None else float(hi),
                    )
                ]
                * p.numel()
            )
    assert [tuple(b) for b in rs._rs.raw_parameter_bounds()] == expected
    check.record("optimizer bounds", 0.0, 0.0)


@pytest.mark.parametrize("case", CASES, ids=_ids(CASES))
def test_mll_and_gradient(case: Case) -> None:
    _, _, _, bt, rs = _pair(case)
    default = raw_params(bt)
    g = torch.Generator().manual_seed(7)
    for k in range(3):
        if k > 0:
            raw = [v + 0.3 * float(torch.randn(1, generator=g, dtype=DT)) for v in default]
            # Keep non-enforced bounds (lengthscales >= 1e-4 / 1e-3).
            bounds = rs._rs.raw_parameter_bounds()
            raw = [max(lo + 1e-3, min(hi, v)) for v, (lo, hi) in zip(raw, bounds, strict=True)]
            set_raw(bt, rs, raw)
        vb, gb = bt_mll_grad(bt)
        vr, gr = rs_mll_grad(rs)
        what = "default" if k == 0 else "random"
        check.check(f"MLL ({what})", vr, vb, VALUE_TOL)
        err = grad_err(gr, gb)
        check.record(f"MLL gradient ({what})", err, GRAD_TOL)
        assert err <= GRAD_TOL, (err, gr, gb)
        compare_state(check, bt, rs, xtol_of(case), f"MLL ({what})")


@pytest.mark.parametrize("case", CASES, ids=_ids(CASES))
def test_posterior(case: Case) -> None:
    _, _, _, bt, rs = _pair(case)
    X1 = random_points(bt, (6, 1), 1)
    X3 = random_points(bt, (2, 3), 2)
    X3[0, 1] = bt.datapoints[1]  # a training point
    for X in (X1, X3, X3[0]):
        pb, pr = bt.posterior(X), rs.posterior(X)
        check.check("posterior mean", pr.mean, pb.mean.detach(), VALUE_TOL)
        check.check("posterior variance", pr.variance, pb.variance.detach(), VALUE_TOL)
        check.check(
            "posterior covariance",
            pr.distribution.covariance_matrix,
            pb.distribution.covariance_matrix.detach(),
            VALUE_TOL,
        )
    w = torch.tensor([-1.7], dtype=DT)
    pb = bt.posterior(X3, posterior_transform=BtScalarized(w, offset=0.3))
    pr = rs.posterior(X3, posterior_transform=RsScalarized(w, offset=0.3))
    check.check("posterior (transform) mean", pr.mean, pb.mean.detach(), VALUE_TOL)
    check.check("posterior (transform) variance", pr.variance, pb.variance.detach(), VALUE_TOL)
    # Input gradients through the analytic and MC acquisition functions.
    for bcls, rcls in (
        (bt_an.PosteriorMean, rs_an.PosteriorMean),
        (bt_an.PosteriorStandardDeviation, rs_an.PosteriorStandardDeviation),
    ):
        Xb = X1.clone().requires_grad_(True)
        Xr = X1.clone().requires_grad_(True)
        vb, vr = bcls(bt)(Xb), rcls(rs)(Xr)
        vb.sum().backward()
        vr.sum().backward()
        check.check(f"{bcls.__name__} value", vr.detach(), vb.detach(), VALUE_TOL)
        check.check(f"{bcls.__name__} input gradient", Xr.grad, Xb.grad, GRAD_TOL)
    sb = BtSampler(torch.Size([32]), seed=3)
    ab = bt_mc.qSimpleRegret(bt, sampler=sb)
    ab(X3)
    sr = RsSampler(32)
    sr.base_samples = sb.base_samples
    ar = rs_mc.qSimpleRegret(rs, sampler=sr)
    Xb = X3.clone().requires_grad_(True)
    Xr = X3.clone().requires_grad_(True)
    vb, vr = ab(Xb), ar(Xr)
    vb.sum().backward()
    vr.sum().backward()
    check.check("qSimpleRegret value", vr.detach(), vb.detach(), VALUE_TOL)
    check.check("qSimpleRegret input gradient", Xr.grad, Xb.grad, GRAD_TOL)


FIT_CASES = [CASES[0], CASES[1], CASES[2], CASES[5], CASES[6], CASES[8], CASES[9]]


@pytest.mark.parametrize("case", FIT_CASES, ids=_ids(FIT_CASES))
def test_fit(case: Case) -> None:
    X, C = make_data(case, 100)

    def fit_twin(t: int) -> tuple[tuple[float, int], tuple[float, int]]:
        bt, rs = build_pair(case, twin_inputs(X, t), C, seed=0)
        b = bt_fit(bt)
        r = rs_fit(rs)
        if t == 0:
            # The fitted models agree (posterior at the training points).
            pb, pr = bt.posterior(bt.datapoints), rs.posterior(bt.datapoints)
            check.record(
                "fitted posterior mean (twin 0)",
                float((torch.as_tensor(pr.mean) - pb.mean).abs().max()),
                None,
            )
        return b, r

    compare_twin_fits(check, f"fit {case.label()}", fit_twin, twins=4)


@pytest.mark.parametrize(
    "case", [CASES[0], CASES[1], CASES[8]], ids=_ids([CASES[0], CASES[1], CASES[8]])
)
def test_condition_on_observations(case: Case) -> None:
    _, _, _, bt, rs = _pair(case)
    g = torch.Generator().manual_seed(5)
    Xn = random_points(bt, (3,), 9)
    Yn = torch.tensor([[0, 2], [1, 0]], dtype=torch.long)
    for model_mode in ("eval", "train"):
        getattr(bt, model_mode)()
        getattr(rs, model_mode)()
        np.random.seed(21)
        cb = bt.condition_on_observations(Xn, Yn)
        np.random.seed(21)
        cr = rs.condition_on_observations(Xn, Yn)
        compare_state(check, cb, cr, xtol_of(case), f"conditioned ({model_mode})")
        Xt = random_points(bt, (4, 2), 11)
        check.check(
            "conditioned posterior mean",
            cr.posterior(Xt).mean,
            cb.posterior(Xt).mean.detach(),
            VALUE_TOL,
        )
    # Batched comparisons (a batch of conditioned models, not consolidated).
    Yb = torch.stack([Yn, Yn.flip(-1)])
    np.random.seed(22)
    cb = bt.condition_on_observations(Xn, Yb)
    np.random.seed(22)
    cr = rs.condition_on_observations(Xn, Yb)
    assert cr.batch_shape == tuple(cb.batch_shape)
    check.check("batched conditioning: utility", cr.utility, cb.utility.detach(), VALUE_TOL)
    Xt = random_points(bt, (2, 3), 12)
    check.check(
        "batched conditioning: posterior mean",
        cr.posterior(Xt).mean,
        cb.posterior(Xt).mean.detach(),
        VALUE_TOL,
    )
    del g


@pytest.mark.parametrize("sampler_kind", ["sobol", "iid"])
def test_fantasize_and_pairwise_samplers(sampler_kind: str) -> None:
    case = CASES[0]
    _, _, _, bt, rs = _pair(case)
    Xf = random_points(bt, (4,), 13)
    if sampler_kind == "sobol":
        sb = BtPSobol(torch.Size([3]), seed=4, max_num_comparisons=4)
        sr = RsPSobol(3, seed=4, max_num_comparisons=4)
        base = BtSampler(torch.Size([3]), seed=4)
        base(bt.posterior(Xf))
        sr.base_samples = base.base_samples
    else:
        sb = BtPIID(torch.Size([3]), seed=4, max_num_comparisons=4)
        sr = RsPIID(3, seed=4, max_num_comparisons=4)
    # BoTorch's orientation (the item with the smaller sample first) is a bug.
    comps_bug = sb(bt.posterior(Xf))
    with fixed_pairwise_sampler():
        comps_bt = sb(bt.posterior(Xf))
        state_bt = np.random.get_state()[1].copy()
        comps_rs = sr(rs.posterior(Xf))
        state_rs = np.random.get_state()[1].copy()
        assert torch.equal(torch.as_tensor(comps_rs), comps_bt)
        assert torch.equal(comps_bug.flip(-1), comps_bt)
        # Like BoTorch, the sampler leaves numpy's generator seeded.
        assert (state_bt == state_rs).all()
        check.record("pairwise sampler comparisons (exact)", 0.0, 0.0)
        fb = bt.fantasize(Xf, sb)
        fr = rs.fantasize(Xf, sr)
    assert fr.batch_shape == tuple(fb.batch_shape)
    check.check("fantasy utility", fr.utility, fb.utility.detach(), VALUE_TOL)
    Xt = random_points(bt, (2,), 14)
    pb, pr = fb.posterior(Xt), fr.posterior(Xt)
    check.check("fantasy posterior mean", pr.mean, pb.mean.detach(), VALUE_TOL)
    check.check(
        "fantasy posterior covariance",
        pr.distribution.covariance_matrix,
        pb.distribution.covariance_matrix.detach(),
        VALUE_TOL,
    )


def test_iid_normal_sampler() -> None:
    """``IIDNormalSampler`` draws BoTorch's base samples (torch's generator
    seeded with ``seed``), also with a ``batch_range_override``."""
    case = CASES[0]
    _, _, _, bt, rs = _pair(case)
    X = random_points(bt, (2, 3, 2), 15)
    for override in (None, (1, -1)):
        sb, sr = BtIID(torch.Size([5]), seed=9), RsIID(5, seed=9)
        if override is not None:
            sb.batch_range_override = override
            sr.batch_range_override = override
        yb, yr = sb(bt.posterior(X)), sr(rs.posterior(X))
        assert torch.equal(torch.as_tensor(sr.base_samples), sb.base_samples)
        check.check("IIDNormalSampler samples", yr, yb.detach(), VALUE_TOL)


def test_load_state_dict_and_mean_constant() -> None:
    case = CASES[0]
    _, X, C, bt, rs = _pair(case)
    sd = {
        "covar_module.raw_outputscale": torch.tensor(0.4, dtype=DT),
        "covar_module.base_kernel.raw_lengthscale": torch.tensor([[0.3, 0.9]], dtype=DT),
    }
    bt.load_state_dict({**bt.state_dict(), **sd})
    rs.load_state_dict(sd)
    compare_state(check, bt, rs, xtol_of(case), "load_state_dict")
    # A nonzero (not fitted) mean constant: the Newton steps use f - mu, the
    # MLL uses f (BoTorch's expressions).
    with torch.no_grad():
        bt.mean_module.constant.fill_(0.3)
    rs.mean_module.constant = 0.3
    bt._update(bt.transform_inputs(bt.datapoints))
    rs._update()
    compare_state(check, bt, rs, xtol_of(case), "mean constant")
    check.check("MLL (mean constant)", rs_mll_grad(rs)[0], bt_mll_grad(bt)[0], VALUE_TOL)


@pytest.mark.parametrize("lik", ["probit", "logit"])
def test_likelihood_functions(lik: str) -> None:
    lb, lr = (BtProbit(), RsProbit()) if lik == "probit" else (BtLogit(), RsLogit())
    g = torch.Generator().manual_seed(3)
    u = torch.randn(2, 6, generator=g, dtype=DT) * 3.0
    D = torch.zeros(2, 5, 6, dtype=DT)
    for b in range(2):
        for k in range(5):
            i, j = torch.randperm(6, generator=g)[:2].tolist()
            D[b, k, i], D[b, k, j] = 1.0, -1.0
    for name in ("p", "log_p", "negative_log_gradient_sum", "negative_log_hessian_sum"):
        vb = getattr(lb, name)(utility=u, D=D)
        vr = getattr(lr, name)(utility=u, D=D)
        check.check(f"likelihood {name}", vr, vb, VALUE_TOL)


def test_model_list_gp() -> None:
    """``ModelListGP`` of a ``PairwiseGP`` and a ``SingleTaskGP``: posteriors
    as in BoTorch; fitting with a ``SumMarginalLogLikelihood`` and fantasies
    raise BoTorch's errors."""
    from gpytorch.mlls import SumMarginalLogLikelihood as BtSum

    from botorch_rs.mlls import SumMarginalLogLikelihood as RsSum

    case = CASES[0]
    _, X, C, bt, rs = _pair(case)
    Y = X.sum(-1, keepdim=True)
    sb, sr = BtSTGP(X, Y), RsSTGP(X, Y)
    lb, lr = BtModelListGP(bt, sb), RsModelListGP(rs, sr)
    Xt = random_points(bt, (3, 2), 16)
    pb, pr = lb.posterior(Xt), lr.posterior(Xt)
    check.check("ModelListGP posterior mean", pr.mean, pb.mean.detach(), VALUE_TOL)
    check.check("ModelListGP posterior variance", pr.variance, pb.variance.detach(), VALUE_TOL)
    for build in (lambda: BtSum(lb.likelihood, lb), lambda: RsSum(lr.likelihood, lr)):
        with pytest.raises(RuntimeError, match="Likelihood must be Gaussian for exact inference"):
            build()
    sampler_b, sampler_r = BtSampler(torch.Size([2]), seed=1), RsSampler(2, seed=1)
    for model, sampler in ((lb, sampler_b), (lr, sampler_r)):
        with pytest.raises(RuntimeError, match="Conditioning on non-pairwise comparison"):
            model.fantasize(Xt, sampler)


# ---------------------------------------------------------------------------
# Acquisition functions
# ---------------------------------------------------------------------------


def _vg(acqf: object, X: object) -> tuple[object, object]:
    X = X.clone().requires_grad_(True)  # type: ignore[attr-defined]
    v = acqf(X)  # type: ignore[operator]
    v.sum().backward()
    return v.detach(), X.grad


def _check_vg(label: str, ab: object, ar: object, X: object) -> None:
    vb, gb = _vg(ab, X)
    vr, gr = _vg(ar, X)
    check.check(f"{label} value", vr, vb, VALUE_TOL)
    err = grad_err(gr, gb)
    check.record(f"{label} input gradient", err, GRAD_TOL)
    assert err <= GRAD_TOL, (label, err)


def _outcome_models(case: Case, seed: int) -> list[tuple[str, object, object, int]]:
    g = torch.Generator().manual_seed(seed)
    dx = 3
    a = torch.rand(dx, case.d, generator=g, dtype=DT) * 0.5
    b = torch.rand(case.d, generator=g, dtype=DT) * 0.2
    Xo = torch.rand(10, dx, generator=g, dtype=DT)
    Yo = torch.stack([(Xo * (k + 1)).sin().sum(-1) for k in range(case.d)], -1) * 0.3 + 0.4
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gb, gr = BtSTGP(Xo, Yo), RsSTGP(Xo, Yo)
    return [
        ("affine", BtAffine(a, b), RsAffine(a, b), dx),
        ("posterior mean", BtPMM(gb), RsPMM(gr), dx),
    ]


ACQ_CASES = [CASES[0], CASES[1], CASES[6], CASES[8]]


@pytest.mark.parametrize("case", ACQ_CASES, ids=_ids(ACQ_CASES))
def test_analytic_eubo(case: Case) -> None:
    _, _, _, bt, rs = _pair(case)
    X = random_points(bt, (5, 2), 20)
    _check_vg(
        "EUBO",
        bt_pref.AnalyticExpectedUtilityOfBestOption(bt),
        rs_pref.AnalyticExpectedUtilityOfBestOption(rs),
        X,
    )
    winner = random_points(bt, (1,), 21)
    _check_vg(
        "EUBO (previous winner)",
        bt_pref.AnalyticExpectedUtilityOfBestOption(bt, previous_winner=winner),
        rs_pref.AnalyticExpectedUtilityOfBestOption(rs, previous_winner=winner),
        X[:, :1],
    )
    for label, ob, orr, dx in _outcome_models(case, 22):
        Xo = torch.rand(4, 2, dx, generator=torch.Generator().manual_seed(23), dtype=DT)
        _check_vg(
            f"EUBO (outcome model: {label})",
            bt_pref.AnalyticExpectedUtilityOfBestOption(bt, outcome_model=ob),
            rs_pref.AnalyticExpectedUtilityOfBestOption(rs, outcome_model=orr),
            Xo,
        )


@pytest.mark.parametrize("case", ACQ_CASES, ids=_ids(ACQ_CASES))
def test_qeubo(case: Case) -> None:
    _, _, _, bt, rs = _pair(case)

    def pair(q: int, X: object, **kw: object) -> tuple[object, object]:
        sb = BtSampler(torch.Size([64]), seed=31)
        kw_b = dict(kw)
        kw_r = dict(kw)
        if "outcome_model" in kw:
            kw_b["outcome_model"], kw_r["outcome_model"] = kw["outcome_model"]  # type: ignore[misc]
        ab = bt_pref.qExpectedUtilityOfBestOption(bt, sampler=sb, **kw_b)
        ab(X)
        sr = RsSampler(64)
        sr.base_samples = sb.base_samples
        return ab, rs_pref.qExpectedUtilityOfBestOption(rs, sampler=sr, **kw_r)

    for q in (1, 2, 4):
        X = random_points(bt, (3, q), 30 + q)
        _check_vg(f"qEUBO (q={q})", *pair(q, X), X)
    X = random_points(bt, (3, 2), 35)
    pending = random_points(bt, (2,), 36)
    _check_vg("qEUBO (pending)", *pair(2, X, X_pending=pending), X)
    for label, ob, orr, dx in _outcome_models(case, 37):
        Xo = torch.rand(3, 2, dx, generator=torch.Generator().manual_seed(38), dtype=DT)
        _check_vg(f"qEUBO (outcome model: {label})", *pair(2, Xo, outcome_model=(ob, orr)), Xo)


def _outcome_gps(case: Case, seed: int) -> tuple[object, object, int]:
    """BoTorch's and botorch_rs's outcome ``SingleTaskGP`` (``case.d``
    outcomes of 3 inputs) with the same data and hyperparameters."""
    g = torch.Generator().manual_seed(seed)
    dx = 3
    Xo = torch.rand(10, dx, generator=g, dtype=DT)
    Yo = torch.stack([(Xo * (k + 1)).sin().sum(-1) for k in range(case.d)], -1) * 0.3 + 0.4
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return BtSTGP(Xo, Yo), RsSTGP(Xo, Yo), dx


def _check_constructed(kb: dict, kr: dict, models: dict) -> None:
    """The kwargs of BoTorch's and botorch_rs's constructors: the same keys,
    the same objects passed through (``models`` maps BoTorch's objects to
    botorch_rs's), and the one-sample outcome model with BoTorch's ``w``."""
    assert sorted(kb) == sorted(kr)
    for key, vb in kb.items():
        vr = kr[key]
        if isinstance(vb, BtFSSM):
            assert isinstance(vr, RsFSSM) and vr.model is models[vb.model]
            assert torch.equal(torch.as_tensor(vr.w), vb.w), (vr.w, vb.w)
            assert vr.jitter == vb.jitter
        elif vb is None or isinstance(vb, (torch.Tensor, float)):
            assert vr is vb, key
        else:
            assert vr is models[vb], key
    check.record("constructed kwargs (objects and w, exact)", 0.0, 0.0)


@pytest.mark.parametrize("case", ACQ_CASES[:2], ids=_ids(ACQ_CASES[:2]))
def test_input_constructors(case: Case) -> None:
    """``construct_inputs_analytic_eubo`` and ``construct_inputs_qeubo``
    through the registries: the constructed kwargs (PBO: the preference model
    itself; BOPE: ``FixedSingleSampleModel(model, w)`` with ``w`` from
    ``torch.randn`` after the same seed), and the values and input gradients
    of the acquisition functions built from them."""
    _, _, _, bt, rs = _pair(case)
    gb, gr, dx = _outcome_gps(case, 50)
    models = {bt: rs, gb: gr}
    eubo_b = bt_get_ctor(bt_pref.AnalyticExpectedUtilityOfBestOption)
    eubo_r = rs_get_ctor(rs_pref.AnalyticExpectedUtilityOfBestOption)
    qeubo_b = bt_get_ctor(bt_pref.qExpectedUtilityOfBestOption)
    qeubo_r = rs_get_ctor(rs_pref.qExpectedUtilityOfBestOption)

    # PBO: the preference model is `model`; `training_data`, `objective` and
    # `posterior_transform` are accepted and ignored (analytic EUBO).
    winner = random_points(bt, (1,), 51)
    extra = {"training_data": None, "objective": None, "posterior_transform": None}
    for prev in (None, winner):
        kb = eubo_b(model=bt, previous_winner=prev, **extra)
        kr = eubo_r(model=rs, previous_winner=prev, **extra)
        _check_constructed(kb, kr, models)
        X = random_points(bt, (4, 2 if prev is None else 1), 52)
        _check_vg(
            "constructed EUBO (PBO)",
            bt_pref.AnalyticExpectedUtilityOfBestOption(**kb),
            rs_pref.AnalyticExpectedUtilityOfBestOption(**kr),
            X,
        )
    sb = BtSampler(torch.Size([64]), seed=53)
    pending = random_points(bt, (2,), 54)
    kb = qeubo_b(model=bt, sampler=sb, X_pending=pending)
    ab = bt_pref.qExpectedUtilityOfBestOption(**kb)
    X = random_points(bt, (3, 2), 55)
    ab(X)  # BoTorch's sampler draws its base samples at the first call
    sr = RsSampler(64)
    sr.base_samples = sb.base_samples
    models[sb] = sr
    kr = qeubo_r(model=rs, sampler=sr, X_pending=pending)
    _check_constructed(kb, kr, models)
    _check_vg("constructed qEUBO (PBO)", ab, rs_pref.qExpectedUtilityOfBestOption(**kr), X)

    # BOPE: EUBO-zeta on the outcome model, w = torch.randn(pref_model.dim) *
    # sample_multiplier from torch's global generator.
    Xo = torch.rand(4, 2, dx, generator=torch.Generator().manual_seed(56), dtype=DT)
    for mult in (1.0, 0.3):
        torch.manual_seed(57)
        kb = eubo_b(model=gb, pref_model=bt, sample_multiplier=mult, **extra)
        torch.manual_seed(57)
        kr = eubo_r(model=gr, pref_model=rs, sample_multiplier=mult, **extra)
        _check_constructed(kb, kr, models)
        _check_vg(
            "constructed EUBO (BOPE)",
            bt_pref.AnalyticExpectedUtilityOfBestOption(**kb),
            rs_pref.AnalyticExpectedUtilityOfBestOption(**kr),
            Xo,
        )
        sb = BtSampler(torch.Size([64]), seed=58)
        torch.manual_seed(59)
        kb = qeubo_b(model=gb, pref_model=bt, sample_multiplier=mult, sampler=sb)
        ab = bt_pref.qExpectedUtilityOfBestOption(**kb)
        ab(Xo)
        sr = RsSampler(64)
        sr.base_samples = sb.base_samples
        models[sb] = sr
        torch.manual_seed(59)
        kr = qeubo_r(model=gr, pref_model=rs, sample_multiplier=mult, sampler=sr)
        _check_constructed(kb, kr, models)
        _check_vg("constructed qEUBO (BOPE)", ab, rs_pref.qExpectedUtilityOfBestOption(**kr), Xo)

    # BoTorch's errors: unknown keyword arguments and `sample_multiplier=None`
    # with a preference model.
    for ctor_b, ctor_r, m_b, m_r in ((eubo_b, eubo_r, gb, gr), (qeubo_b, qeubo_r, gb, gr)):
        with pytest.raises(TypeError) as eb:
            ctor_b(model=bt, foo=1)
        with pytest.raises(TypeError) as er:
            ctor_r(model=rs, foo=1)
        assert str(er.value) == str(eb.value)
        with pytest.raises(TypeError):
            ctor_b(model=m_b, pref_model=bt, sample_multiplier=None)
        with pytest.raises(TypeError):
            ctor_r(model=m_r, pref_model=rs, sample_multiplier=None)
    check.record("constructor errors (same class and message)", 0.0, 0.0)


def test_phase8b_classes_on_pairwise_gp() -> None:
    """Phase 8b's classes with a ``PairwiseGP``: a Python subclass of
    ``SampleReducingMCAcquisitionFunction`` (BoTorch's base samples), and
    ``qMultiStepLookahead``, which fails as BoTorch's does (BoTorch when it
    fantasizes, botorch_rs at construction)."""
    from botorch.acquisition.monte_carlo import (
        SampleReducingMCAcquisitionFunction as BtSampleReducing,
    )
    from botorch.acquisition.multi_step_lookahead import qMultiStepLookahead as BtMSL

    from botorch_rs.acquisition.monte_carlo import (
        SampleReducingMCAcquisitionFunction as RsSampleReducing,
    )
    from botorch_rs.acquisition.multi_step_lookahead import qMultiStepLookahead as RsMSL

    case = ACQ_CASES[0]
    _, _, _, bt, rs = _pair(case)

    class BtSquared(BtSampleReducing):
        def _sample_forward(self, obj: torch.Tensor) -> torch.Tensor:
            return obj.square()

    class RsSquared(RsSampleReducing):
        def _sample_forward(self, obj: torch.Tensor) -> torch.Tensor:
            return obj.square()

    sb = BtSampler(torch.Size([32]), seed=60)
    ab = BtSquared(bt, sampler=sb)
    X = random_points(bt, (3, 2), 61)
    ab(X)
    sr = RsSampler(32)
    sr.base_samples = sb.base_samples
    _check_vg("SampleReducingMCAcquisitionFunction subclass", ab, RsSquared(rs, sampler=sr), X)

    Xq = random_points(bt, (2, 3), 62)
    for mb, mr in ((bt, rs), (BtModelListGP(bt), RsModelListGP(rs))):
        with pytest.raises(RuntimeError) as eb:
            BtMSL(mb, batch_sizes=[1], num_fantasies=[2])(Xq)
        with pytest.raises(RuntimeError) as er:
            RsMSL(mr, batch_sizes=[1], num_fantasies=[2])
        assert str(er.value) == str(eb.value)
    check.record("qMultiStepLookahead on PairwiseGP: error (same class and message)", 0.0, 0.0)


@pytest.mark.parametrize("case", ACQ_CASES, ids=_ids(ACQ_CASES))
def test_bald(case: Case) -> None:
    _, _, _, bt, rs = _pair(case)
    X = random_points(bt, (4, 2), 40)
    for std_noise, n in ((0.0, 256), (0.1, 64)):
        ab = bt_pref.PairwiseBayesianActiveLearningByDisagreement(
            bt, num_samples=n, std_noise=std_noise
        )
        ar = rs_pref.PairwiseBayesianActiveLearningByDisagreement(
            rs, num_samples=n, std_noise=std_noise
        )
        # The same torch seed: the same draws (BoTorch draws them in forward).
        torch.manual_seed(41)
        vb, gb = _vg(ab, X)
        torch.manual_seed(41)
        vr, gr = _vg(ar, X)
        check.check("BALD value", vr, vb, VALUE_TOL)
        check.record("BALD input gradient", grad_err(gr, gb), GRAD_TOL)
        assert grad_err(gr, gb) <= GRAD_TOL
    for label, ob, orr, dx in _outcome_models(case, 42):
        Xo = torch.rand(3, 2, dx, generator=torch.Generator().manual_seed(43), dtype=DT)
        ab = bt_pref.PairwiseBayesianActiveLearningByDisagreement(bt, outcome_model=ob)
        ar = rs_pref.PairwiseBayesianActiveLearningByDisagreement(rs, outcome_model=orr)
        torch.manual_seed(44)
        vb, gb = _vg(ab, Xo)
        torch.manual_seed(44)
        vr, gr = _vg(ar, Xo)
        check.check(f"BALD (outcome model: {label}) value", vr, vb, VALUE_TOL)
        assert grad_err(gr, gb) <= GRAD_TOL


@pytest.mark.parametrize("case", ACQ_CASES, ids=_ids(ACQ_CASES))
def test_pairwise_mc_posterior_variance(case: Case) -> None:
    _, _, _, bt, rs = _pair(case)

    def link(Y: object, X: object = None) -> object:
        return torch.distributions.Normal(0.0, 1.0).cdf(Y.squeeze(-1))  # type: ignore[attr-defined]

    for q in (2, 4):
        X = random_points(bt, (3, q), 50 + q)
        sb = BtSampler(torch.Size([128]), seed=51)
        ab = bt_al.PairwiseMCPosteriorVariance(bt, objective=BtGeneric(link), sampler=sb)
        ab(X)
        sr = RsSampler(128)
        sr.base_samples = sb.base_samples
        ar = rs_al.PairwiseMCPosteriorVariance(rs, objective=RsGeneric(link), sampler=sr)
        _check_vg(f"PairwiseMCPosteriorVariance (q={q})", ab, ar, X)


@pytest.mark.parametrize("case", ACQ_CASES, ids=_ids(ACQ_CASES))
def test_learned_objective(case: Case) -> None:
    _, _, _, bt, rs = _pair(case)
    for sample_shape, shape in (((4,), (3, 2, 2)), ((2, 3), (2, 3))):
        lb = BtLearned(bt, sample_shape=torch.Size(sample_shape), seed=61)
        lr = RsLearned(rs, sample_shape=sample_shape, seed=61)
        Y = random_points(bt, shape, 62)
        Yb = Y.clone().requires_grad_(True)
        Yr = Y.clone().requires_grad_(True)
        ob, orr = lb(Yb), lr(Yr)
        if len(sample_shape) == 1:
            assert tuple(orr.shape) == tuple(ob.shape)
        else:
            # BoTorch merges only the first sample dimension with the
            # outcome samples' (its documented shape is
            # `(sample_size * num_samples) x batch_shape x q`); the order of
            # the values is the same.
            assert ob.shape[0] == math.prod(sample_shape)
        ob = ob.reshape(orr.shape)
        check.check("LearnedObjective samples", orr.detach(), ob.detach(), VALUE_TOL)
        w = torch.rand(ob.shape, generator=torch.Generator().manual_seed(63), dtype=DT)
        (ob * w).sum().backward()
        (orr * w).sum().backward()
        check.check("LearnedObjective gradient", Yr.grad, Yb.grad, GRAD_TOL)
        assert torch.equal(torch.as_tensor(lr.sampler.base_samples), lb.sampler.base_samples)
    # A deterministic preference model: the posterior mean.
    g = torch.Generator().manual_seed(64)
    a = torch.rand(case.d, 1, generator=g, dtype=DT)
    lb, lr = BtLearned(BtAffine(a)), RsLearned(RsAffine(a))
    Y = random_points(bt, (2, 3, 2), 65)
    check.check("LearnedObjective (deterministic)", lr(Y), lb(Y), VALUE_TOL)

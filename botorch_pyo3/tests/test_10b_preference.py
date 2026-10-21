"""API and error tests of the preference-learning components of phase 10b
(``PairwiseGP``, its likelihoods and MLL, the preference acquisition
functions, ``LearnedObjective`` and the samplers): array types, prior-only
models, copies, fits, optimization, and BoTorch's exceptions for invalid
input. Comparisons with BoTorch are in ``test_parity_10b_preference.py``.
"""

from __future__ import annotations

import copy
import math
import warnings

import pytest

from botorch_rs.acquisition.objective import IdentityMCObjective, LearnedObjective
from botorch_rs.acquisition.preference import (
    AnalyticExpectedUtilityOfBestOption,
    PairwiseBayesianActiveLearningByDisagreement,
    qExpectedUtilityOfBestOption,
)
from botorch_rs.exceptions.errors import UnsupportedError
from botorch_rs.fit import fit_gpytorch_mll
from botorch_rs.kernels import MaternKernel, RBFKernel, ScaleKernel
from botorch_rs.models import PairwiseGP, PairwiseLaplaceMarginalLogLikelihood, SingleTaskGP
from botorch_rs.models.likelihoods import PairwiseLogitLikelihood, PairwiseProbitLikelihood
from botorch_rs.sampling import (
    IIDNormalSampler,
    PairwiseIIDNormalSampler,
    PairwiseSobolQMCNormalSampler,
    SobolQMCNormalSampler,
)
from helpers import KINDS, as_kind, shape, to_list

X_LIST = [[0.1, 0.2], [0.5, 0.9], [0.3, 0.4], [0.8, 0.1], [0.65, 0.55], [0.2, 0.75]]
C_LIST = [[0, 1], [2, 3], [1, 4], [5, 2], [4, 0]]


def _comps(kind: str) -> object:
    if kind == "torch":
        import torch

        return torch.tensor(C_LIST, dtype=torch.long)
    if kind == "numpy":
        import numpy as np

        return np.asarray(C_LIST, dtype=np.int64)
    return C_LIST


@pytest.mark.parametrize("kind", KINDS)
def test_array_kinds_and_state(kind: str) -> None:
    m = PairwiseGP(as_kind(X_LIST, kind), _comps(kind))
    assert shape(m.utility) == (6,)
    assert shape(m.covar) == (6, 6)
    assert shape(m.D) == (5, 6)
    assert shape(m.DT) == (6, 5)
    assert m.n == 6 and m.m == 5 and m.dim == 2 and m.num_outputs == 1
    assert m.batch_shape == ()
    p = m.posterior(as_kind([[0.3, 0.3], [0.6, 0.6]], kind))
    assert shape(p.mean) == (2, 1)
    assert shape(p.distribution.covariance_matrix) == (2, 2)
    if kind == "torch":
        import torch

        assert m.comparisons.dtype == torch.long
        assert m.consolidated_indices.dtype == torch.long
    # The win-count heuristic and two Newton steps solve the MAP problem.
    g = m.likelihood.negative_log_gradient_sum(m.utility, m.D)
    assert max(abs(v) for v in to_list(g)) < 10


def test_names_parameters_and_modules() -> None:
    m = PairwiseGP(X_LIST, C_LIST)
    names = [n for n, _ in m.named_parameters()]
    assert names == [
        "mean_module.raw_constant",
        "covar_module.raw_outputscale",
        "covar_module.base_kernel.raw_lengthscale",
    ]
    assert isinstance(m.likelihood, PairwiseProbitLikelihood)
    assert isinstance(m.covar_module, ScaleKernel)
    # BoTorch's defaults: outputscale 1, lengthscale (2.4 - 1) / 2.7.
    assert abs(float(to_list(m.covar_module.outputscale)) - 1.0) < 1e-6
    assert all(
        abs(v - 1.4 / 2.7) < 1e-6 for v in to_list(m.covar_module.base_kernel.lengthscale)[0]
    )
    m.covar_module.base_kernel.lengthscale = [[0.3, 0.6]]
    assert to_list(m.covar_module.base_kernel.lengthscale)[0] == pytest.approx([0.3, 0.6])
    assert m.mean_constant == 0.0
    sd = m.state_dict()
    m.load_state_dict({**sd, "utility": [0.0] * 6})  # buffers are ignored


def test_prior_only_model() -> None:
    m = PairwiseGP(None, None)
    assert m._has_no_data() and m.datapoints is None and m.utility is None
    p = m.posterior([[0.1, 0.2], [0.4, 0.4]])
    assert to_list(p.mean) == [[0.0], [0.0]]
    cov = to_list(p.distribution.covariance_matrix)
    assert cov[0][0] == pytest.approx(1.0) and 0.0 < cov[0][1] < 1.0
    with pytest.raises(RuntimeError, match="datapoints and comparisons cannot be None"):
        m.train()([[0.1, 0.2]])
    c = m.condition_on_observations(X_LIST, C_LIST)
    assert c.n == 6 and c.utility is not None
    with pytest.raises(AttributeError, match="num_outputs"):
        AnalyticExpectedUtilityOfBestOption(m)


def test_errors_for_invalid_comparisons() -> None:
    with pytest.raises(RuntimeError, match="index 6 is out of bounds for dimension 1 with size 6"):
        PairwiseGP(X_LIST, [[0, 6]])
    with pytest.raises(RuntimeError, match="index -1 is out of bounds"):
        PairwiseGP(X_LIST, [[0, -1]])
    with pytest.raises(RuntimeError, match=r"shape '\[-1, 1, 2\]' is invalid for input of size 3"):
        PairwiseGP(X_LIST, [[0, 1, 2]])
    with pytest.raises(IndexError):
        PairwiseGP(X_LIST, [0, 1])
    with pytest.raises(UnsupportedError, match="PairwiseGP must be used with a ScaleKernel"):
        PairwiseGP(X_LIST, C_LIST, covar_module=RBFKernel())
    m = PairwiseGP(X_LIST, C_LIST)
    with pytest.raises(RuntimeError, match="should not be amulti-output model"):
        m.posterior([[0.1, 0.1]], output_indices=[0])
    for Y in ([[0.0, 1.5]], [[0, 1, 1]]):
        with pytest.raises(RuntimeError, match="Conditioning on non-pairwise comparison"):
            m.condition_on_observations([[0.1, 0.1], [0.2, 0.2]], Y)
    with pytest.raises(UnsupportedError, match="Passing strict=True is not supported"):
        m.load_state_dict({}, strict=True)
    m.train()
    with pytest.raises(RuntimeError, match="Must train on training data"):
        m([[0.1, 0.2]])
    mll = PairwiseLaplaceMarginalLogLikelihood(m.likelihood, m)
    out = m(m.unconsolidated_datapoints)
    with pytest.raises(RuntimeError, match="Must train on training data"):
        mll(out, [[0, 1]])
    assert math.isfinite(float(mll(out, m.unconsolidated_comparisons)))
    with pytest.raises(UnsupportedError):
        PairwiseLaplaceMarginalLogLikelihood(m.likelihood, SingleTaskGP(X_LIST, [[0.0]] * 6))


def test_strict_set_train_data() -> None:
    torch = pytest.importorskip("torch")
    X = torch.tensor(X_LIST, dtype=torch.float64)
    C = torch.tensor(C_LIST)
    m = PairwiseGP(X, C)
    with pytest.raises(RuntimeError, match="Cannot modify shape of inputs"):
        m.set_train_data(X.unsqueeze(0), C.unsqueeze(0), strict=True)
    with pytest.raises(RuntimeError, match="Cannot modify shape of targets"):
        m.set_train_data(X, C[:2], strict=True)
    m.set_train_data(X[:5], torch.tensor([[0, 1], [2, 3]]))
    assert m.n == 5 and m.m == 2
    with pytest.warns(Warning, match="double precision"):
        PairwiseGP(X.float(), C)


def test_duplicates_are_consolidated() -> None:
    X = [*X_LIST, [0.10001, 0.2]]
    C = [*C_LIST, [6, 3]]
    m = PairwiseGP(X, C)
    assert m.n == 6 and to_list(m.consolidated_indices) == [0, 1, 2, 3, 4, 5, 0]
    assert to_list(m.comparisons)[-1] == [0, 3]
    assert len(to_list(m.unconsolidated_utility)) == 7
    # A comparison of two near-duplicates is uninformative (BoTorch keeps a
    # -1 in D for the item, a bug): the model equals the one without it.
    a = PairwiseGP(X, [*C, [6, 0]])
    b = PairwiseGP(X, C)
    assert to_list(a.utility) == pytest.approx(to_list(b.utility), rel=1e-9, abs=1e-12)
    la = PairwiseLaplaceMarginalLogLikelihood(a.likelihood, a).value_and_grad()[0]
    lb = PairwiseLaplaceMarginalLogLikelihood(b.likelihood, b).value_and_grad()[0]
    assert la == pytest.approx(lb + math.log(0.5), rel=1e-12)
    PairwiseGP(X, [*C, [6, 0]], likelihood=PairwiseLogitLikelihood())  # no crash


def test_copy_condition_fantasize() -> None:
    m = PairwiseGP(X_LIST, C_LIST)
    c = copy.deepcopy(m)
    c.covar_module.outputscale = 2.0
    assert float(to_list(m.covar_module.outputscale)) == pytest.approx(1.0)
    m.eval()
    cond = m.condition_on_observations([[0.4, 0.4], [0.9, 0.9]], [[1, 0]])
    assert cond.n == 8 and cond.m == 6
    sampler = PairwiseSobolQMCNormalSampler(4, seed=0, max_num_comparisons=2)
    f = m.fantasize([[0.4, 0.4], [0.9, 0.9], [0.5, 0.1]], sampler)
    assert f.batch_shape == (4,)
    assert shape(f.comparisons) == (4, 7, 2)
    assert shape(f.posterior([[0.3, 0.3]]).mean) == (4, 1, 1)
    with pytest.raises(RuntimeError, match="Conditioning on non-pairwise comparison"):
        m.fantasize([[0.4, 0.4], [0.9, 0.9]], SobolQMCNormalSampler(2, seed=0))


def test_samplers() -> None:
    m = PairwiseGP(X_LIST, C_LIST)
    post = m.posterior([[0.4, 0.4], [0.9, 0.9], [0.5, 0.1]])
    for sampler in (
        PairwiseSobolQMCNormalSampler(3, seed=2),
        PairwiseIIDNormalSampler(3, seed=2),
    ):
        comps = to_list(sampler(post))
        assert len(comps) == 3 and all(len(c) == 3 for c in comps)
        # The preferred item first: the samples order every pair.
        samples = to_list(sampler._base_forward(post))
        for s, cs in zip(samples, comps, strict=True):
            for w, lo in cs:
                assert s[w][0] > s[lo][0]
    with pytest.raises(RuntimeError, match="Number of samples < 2"):
        PairwiseSobolQMCNormalSampler(2, seed=0)(m.posterior([[0.4, 0.4]]))
    iid = IIDNormalSampler(4, seed=3)
    a = to_list(iid(post))
    b = to_list(IIDNormalSampler(4, seed=3)(post))
    assert a == b and shape(a) == (4, 3, 1)


def test_fit_and_mll() -> None:
    m = PairwiseGP(X_LIST, C_LIST)
    mll = PairwiseLaplaceMarginalLogLikelihood(m.likelihood, m)
    v0, g0 = mll.value_and_grad()
    assert len(g0) == 3
    out = fit_gpytorch_mll(mll)
    assert out is mll and not m.training
    assert mll._fit_result["status"] in ("SUCCESS", "STOPPED")
    assert -mll._fit_result["fval"] >= v0 - 1e-9
    # The installed state is that of the optimum.
    v1, _ = mll.value_and_grad()
    assert v1 == pytest.approx(-mll._fit_result["fval"], rel=1e-9)


def test_logit_and_custom_kernel() -> None:
    k = ScaleKernel(MaternKernel(nu=1.5, ard_num_dims=2))
    m = PairwiseGP(X_LIST, C_LIST, likelihood=PairwiseLogitLikelihood(), covar_module=k)
    assert m.likelihood._logit_lim == 8
    fit_gpytorch_mll(PairwiseLaplaceMarginalLogLikelihood(m.likelihood, m))
    assert shape(m.posterior([[0.1, 0.1]]).mean) == (1, 1)


def test_batched_model() -> None:
    torch = pytest.importorskip("torch")
    X = torch.tensor([X_LIST, [[1 - v for v in r] for r in X_LIST]], dtype=torch.float64)
    C = torch.tensor([C_LIST, C_LIST])
    m = PairwiseGP(X, C)
    assert m.batch_shape == (2,)
    assert tuple(m.covar_module.base_kernel.raw_lengthscale.shape) == (2, 1, 2)
    assert tuple(m.covar_module.raw_outputscale.shape) == (2,)
    assert tuple(m.posterior(torch.rand(3, 2, dtype=torch.float64)).mean.shape) == (2, 3, 1)
    fit_gpytorch_mll(PairwiseLaplaceMarginalLogLikelihood(m.likelihood, m))


def test_acquisition_functions_and_optimization() -> None:
    from botorch_rs.acquisition.active_learning import PairwiseMCPosteriorVariance
    from botorch_rs.optim import optimize_acqf

    m = PairwiseGP(X_LIST, C_LIST)
    bounds = [[0.0, 0.0], [1.0, 1.0]]
    eubo = AnalyticExpectedUtilityOfBestOption(m)
    cand, val = optimize_acqf(eubo, bounds, q=2, num_restarts=2, raw_samples=16)
    assert shape(cand) == (2, 2)
    with pytest.raises(UnsupportedError, match="only supports q=2"):
        eubo([[[0.1, 0.1], [0.2, 0.2], [0.3, 0.3]]])
    w = AnalyticExpectedUtilityOfBestOption(m, previous_winner=[[0.5, 0.5]])
    assert shape(w([[[0.1, 0.1]]])) == (1,)
    qeubo = qExpectedUtilityOfBestOption(m, sampler=SobolQMCNormalSampler(32, seed=0))
    v, g = qeubo.value_and_grad([[[0.1, 0.1], [0.6, 0.6], [0.3, 0.9]]])
    assert shape(g) == (1, 3, 2)
    bald = PairwiseBayesianActiveLearningByDisagreement(m, num_samples=64)
    with pytest.raises(AssertionError, match="q=2"):
        bald([[[0.1, 0.1], [0.2, 0.2], [0.3, 0.3]]])
    vb = to_list(bald([[[0.1, 0.1], [0.6, 0.6]]]))
    assert 0.0 <= vb[0] <= math.log(2)
    cand, _ = optimize_acqf(bald, bounds, q=2, num_restarts=2, raw_samples=16)
    assert shape(cand) == (2, 2)
    mcv = PairwiseMCPosteriorVariance(m, objective=IdentityMCObjective())
    with pytest.raises(RuntimeError, match="q must be a multiple of 2"):
        mcv([[[0.1, 0.1], [0.2, 0.2], [0.3, 0.3]]])
    with pytest.raises(UnsupportedError, match="deterministic"):
        AnalyticExpectedUtilityOfBestOption(m, outcome_model=SingleTaskGP(X_LIST, [[0.1]] * 6))


def test_learned_objective_acquisition() -> None:
    pytest.importorskip("torch")
    from botorch_rs.acquisition.monte_carlo import qSimpleRegret

    m = PairwiseGP(X_LIST, C_LIST)
    lo = LearnedObjective(m, sample_shape=(3,), seed=1)
    outcome = SingleTaskGP(X_LIST, [[0.1, 0.3]] * 6)
    assert math.isfinite(float(to_list(qSimpleRegret(outcome, objective=lo)([[[0.2, 0.4]]]))[0]))


@pytest.mark.parametrize("kind", KINDS)
def test_input_constructors(kind: str) -> None:
    """The registered EUBO constructors: preferential BO (the preference
    model is ``model``) and BOPE (EUBO-zeta on a ``FixedSingleSampleModel``
    of the outcome model with ``pref_model.dim`` weights)."""
    from botorch_rs.acquisition import get_acqf_input_constructor
    from botorch_rs.acquisition.input_constructors import (
        construct_inputs_analytic_eubo,
        construct_inputs_qeubo,
    )
    from botorch_rs.models.deterministic import FixedSingleSampleModel

    pref = PairwiseGP(as_kind(X_LIST, kind), _comps(kind))
    Xo = [[0.1 * i, 0.3, 1.0 - 0.1 * i] for i in range(6)]
    Yo = [[0.5 * a + b, a - c] for a, b, c in Xo]
    outcome = SingleTaskGP(as_kind(Xo, kind), as_kind(Yo, kind))
    eubo_ctor = get_acqf_input_constructor(AnalyticExpectedUtilityOfBestOption)
    qeubo_ctor = get_acqf_input_constructor(qExpectedUtilityOfBestOption)
    kw = eubo_ctor(model=pref, training_data=None, objective=None)
    assert kw == {"pref_model": pref, "outcome_model": None, "previous_winner": None}
    v = AnalyticExpectedUtilityOfBestOption(**kw)(as_kind([[[0.1, 0.1], [0.6, 0.6]]], kind))
    assert shape(v) == (1,)
    kw = construct_inputs_analytic_eubo(model=outcome, pref_model=pref, sample_multiplier=0.5)
    om = kw["outcome_model"]
    assert kw["pref_model"] is pref and isinstance(om, FixedSingleSampleModel)
    assert om.model is outcome and shape(om.w) == (2,)
    v = AnalyticExpectedUtilityOfBestOption(**kw)(as_kind([[Xo[0], Xo[3]], [Xo[1], Xo[5]]], kind))
    assert shape(v) == (2,)
    sampler = SobolQMCNormalSampler(32, seed=0)
    kw = qeubo_ctor(model=outcome, pref_model=pref, sampler=sampler, X_pending=None)
    assert kw["sampler"] is sampler and kw["objective"] is None
    assert kw["posterior_transform"] is None and kw["X_pending"] is None
    v = qExpectedUtilityOfBestOption(**kw)(as_kind([[Xo[0], Xo[2], Xo[4]]], kind))
    assert shape(v) == (1,)
    kw = construct_inputs_qeubo(model=pref)
    assert kw["pref_model"] is pref and kw["outcome_model"] is None
    with pytest.raises(TypeError, match="Unexpected keyword argument `bounds_x`"):
        qeubo_ctor(model=pref, bounds_x=None)


def test_learned_objective_api() -> None:
    from botorch_rs.models import AffineDeterministicModel

    m = PairwiseGP(X_LIST, C_LIST)
    lo = LearnedObjective(m)
    assert lo.sampler.sample_shape == (16,)
    samples = [[[[0.1, 0.2], [0.5, 0.5]]] * 3] * 2  # 2 x 3 x 2 x 2
    out = lo(samples)
    assert shape(out) == (32, 3, 2)
    with pytest.raises(ValueError, match="at least 3 dimensions"):
        lo([[0.1, 0.2]])
    det = AffineDeterministicModel([[1.0], [2.0]])
    with pytest.raises(ValueError, match="sample_shape must be None"):
        LearnedObjective(det, sample_shape=(2,))
    assert shape(LearnedObjective(det)(samples)) == (2, 3, 2)


def test_likelihood_functions() -> None:
    D = [[1.0, -1.0, 0.0], [0.0, 1.0, -1.0]]
    u = [0.2, -0.4, 1.0]
    for lik in (PairwiseProbitLikelihood(), PairwiseLogitLikelihood()):
        p = to_list(lik.p(u, D))
        assert 0 < p[0] < 1 and len(p) == 2
        assert to_list(lik.log_p(u, D)) == pytest.approx([math.log(v) for v in p], rel=1e-12)
        assert shape(lik.negative_log_hessian_sum(u, D)) == (3, 3)
    # BoTorch's logit gradient `e^{u_l} / (e^{u_w} + e^{u_l})` is NaN above
    # utilities of ~709 (a bug); botorch_rs evaluates it stably there.
    g = to_list(PairwiseLogitLikelihood().negative_log_gradient_sum([800.0, 800.0], [[1.0, -1.0]]))
    assert g == [-0.5, 0.5]
    torch = pytest.importorskip("torch")
    b = PairwiseProbitLikelihood()(torch.tensor(u), torch.tensor(D))
    assert isinstance(b, torch.distributions.Bernoulli)


def test_model_list() -> None:
    from botorch_rs.models import ModelListGP

    m = PairwiseGP(X_LIST, C_LIST)
    st = SingleTaskGP(X_LIST, [[float(i)] for i in range(6)])
    ml = ModelListGP(m, st)
    assert shape(ml.posterior([[0.1, 0.1]]).mean) == (1, 2)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(RuntimeError, match="Conditioning on non-pairwise"):
            ml.fantasize([[0.1, 0.1]], SobolQMCNormalSampler(2, seed=0))

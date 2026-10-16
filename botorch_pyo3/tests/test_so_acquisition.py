"""The single-objective acquisition functions of phase 8 through the public
API, independently of BoTorch: result types and shapes for every array type,
gradients (finite differences of ``value_and_grad``), properties that follow
from the definitions (KG's inner maximization, qLCB <= qUCB, the information
gain of nearly independent points, qNIPV's monotonicity, the closed forms of
the wrappers), reproducibility from seeds and draws, ``optimize_acqf`` and
errors. The comparisons with BoTorch are in ``test_parity_so_acquisition.py``
and ``test_equivalence_so_acquisition.py``.
"""

from __future__ import annotations

import functools
import math
import random
from typing import Any

import pytest

from botorch_rs.acquisition import (
    ExpectedImprovement,
    FixedFeatureAcquisitionFunction,
    GaussianPenalty,
    GroupLassoPenalty,
    L0PenaltyApprox,
    L1Penalty,
    L2Penalty,
    LogNoisyExpectedImprovement,
    NoisyExpectedImprovement,
    PathwiseThompsonSampling,
    PenalizedAcquisitionFunction,
    PosteriorMean,
    PriorGuidedAcquisitionFunction,
    ProbabilityOfImprovement,
    ProximalAcquisitionFunction,
    get_optimal_samples,
    qAnalyticProbabilityOfImprovement,
    qExpectedImprovement,
    qJointEntropySearch,
    qKnowledgeGradient,
    qLowerBoundMaxValueEntropy,
    qLowerConfidenceBound,
    qMaxValueEntropy,
    qMultiFidelityKnowledgeGradient,
    qMultiFidelityLowerBoundMaxValueEntropy,
    qNegIntegratedPosteriorVariance,
    qNoisyExpectedImprovement,
    qPosteriorStandardDeviation,
    qPredictiveEntropySearch,
    qUpperConfidenceBound,
)
from botorch_rs.acquisition.cost_aware import InverseCostWeightedUtility
from botorch_rs.acquisition.decoupled import DecoupledAcquisitionFunction
from botorch_rs.acquisition.max_value_entropy_search import (
    _sample_max_value_Gumbel,
    _sample_max_value_Thompson,
)
from botorch_rs.acquisition.objective import ScalarizedPosteriorTransform
from botorch_rs.acquisition.utils import project_to_target_fidelity
from botorch_rs.exceptions import UnsupportedError
from botorch_rs.exceptions.errors import BotorchTensorDimensionError
from botorch_rs.models import ModelListGP, SingleTaskGP
from botorch_rs.models.cost import AffineFidelityCostModel
from botorch_rs.optim import optimize_acqf
from botorch_rs.sampling import SobolQMCNormalSampler
from helpers import HAS_TORCH, KINDS, as_kind, assert_close, check_kind, flatten, shape

# The legacy-EI numerics warnings and the input scaling warnings of the
# one-dimensional model are BoTorch's behavior, not under test here.
pytestmark = [
    pytest.mark.filterwarnings("ignore::botorch_rs.exceptions.warnings.NumericsWarning"),
    pytest.mark.filterwarnings("ignore::botorch_rs.exceptions.warnings.InputDataWarning"),
]

BOUNDS = [[0.0, 0.0], [1.0, 1.0]]


def _data(n: int, seed: int, d: int = 2, m: int = 1) -> tuple[list, list]:
    rng = random.Random(seed)
    X = [[rng.random() for _ in range(d)] for _ in range(n)]
    Y = [
        [
            math.sin(3.0 * x[0] + t) + (0.5 + t) * sum(x[1:]) + 0.02 * rng.gauss(0.0, 1.0)
            for t in range(m)
        ]
        for x in X
    ]
    return X, Y


def _model(m: int = 1, d: int = 2, fixed_noise: bool = False, n: int = 9) -> SingleTaskGP:
    X, Y = _data(n, seed=5, d=d, m=m)
    kw: dict[str, Any] = {}
    if fixed_noise:
        kw["train_Yvar"] = [[0.01] * m for _ in X]
    return SingleTaskGP(X, Y, **kw)


@functools.cache
def model() -> SingleTaskGP:
    return _model()


def _points(n: int, q: int, seed: int, d: int = 2) -> list:
    rng = random.Random(seed)
    return [[[rng.random() for _ in range(d)] for _ in range(q)] for _ in range(n)]


def _fd_check(acqf: Any, X: list, h: float = 1e-6, tol: float = 1e-6) -> None:
    """``value_and_grad`` against central differences of the values."""
    _, grads = acqf.value_and_grad(X)
    g = flatten(grads)
    flat = flatten(X)
    sh = shape(X)
    for k in range(len(flat)):
        up, dn = list(flat), list(flat)
        up[k] += h
        dn[k] -= h
        fd = (sum(flatten(acqf(_reshape(up, sh)))) - sum(flatten(acqf(_reshape(dn, sh))))) / (2 * h)
        assert abs(fd - g[k]) <= tol * max(1.0, abs(fd)), (k, fd, g[k])


def _reshape(flat: list[float], sh: tuple[int, ...]) -> list:
    if len(sh) == 1:
        return list(flat)
    step = len(flat) // sh[0]
    return [_reshape(flat[i * step : (i + 1) * step], sh[1:]) for i in range(sh[0])]


def _sampler(n: int, seed: int = 0) -> SobolQMCNormalSampler:
    return SobolQMCNormalSampler(sample_shape=(n,), seed=seed)


# ---------------------------------------------------------------------------
# Knowledge gradient
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("kind", KINDS)
def test_knowledge_gradient_types_and_gradients(kind: str) -> None:
    acqf = qKnowledgeGradient(model(), num_fantasies=3, sampler=_sampler(3))
    assert acqf.get_augmented_q_batch_size(2) == 5
    X = _points(2, 5, seed=1)
    values = acqf(as_kind(X, kind))
    check_kind(values, kind)
    assert shape(values) == (2,)
    cands = acqf.extract_candidates(as_kind(X, kind))
    assert shape(cands) == (2, 2, 2)
    assert_close(flatten(cands), flatten([x[:2] for x in X]))
    _fd_check(acqf, X)


def test_knowledge_gradient_inner_maximization() -> None:
    """``evaluate`` solves the inner problems: its value is at least the
    one-shot value at any fantasy points."""
    acqf = qKnowledgeGradient(model(), num_fantasies=4, sampler=_sampler(4, seed=2))
    X = _points(3, 5, seed=3)
    one_shot = flatten(acqf(X))
    full = flatten(
        acqf.evaluate(
            [x[:1] for x in X], bounds=BOUNDS, num_restarts=4, raw_samples=64, options={"seed": 0}
        )
    )
    for f, o in zip(full, one_shot, strict=True):
        assert f >= o - 1e-9


def test_knowledge_gradient_current_value_and_pending() -> None:
    m = model()
    X = _points(2, 4, seed=4)
    base = flatten(qKnowledgeGradient(m, num_fantasies=3, sampler=_sampler(3))(X))
    shifted = flatten(
        qKnowledgeGradient(m, num_fantasies=3, sampler=_sampler(3), current_value=0.5)(X)
    )
    assert_close(shifted, [b - 0.5 for b in base])
    acqf = qKnowledgeGradient(m, num_fantasies=3, sampler=_sampler(3))
    acqf.set_X_pending([[0.3, 0.7]])
    assert shape(acqf.X_pending) == (1, 2)
    acqf.set_X_pending(None)
    assert acqf.X_pending is None


def test_optimize_acqf_with_knowledge_gradient() -> None:
    acqf = qKnowledgeGradient(model(), num_fantasies=4, sampler=_sampler(4))
    cands, value = optimize_acqf(
        acqf, BOUNDS, q=2, num_restarts=2, raw_samples=32, options={"seed": 1}
    )
    assert shape(cands) == (2, 2)
    assert all(0.0 <= v <= 1.0 for v in flatten(cands))
    assert math.isfinite(flatten(value)[0])


def test_knowledge_gradient_errors() -> None:
    m = model()
    with pytest.raises(ValueError, match="num_fantasies"):
        qKnowledgeGradient(m, num_fantasies=None)
    with pytest.raises(ValueError, match="sampler shape"):
        qKnowledgeGradient(m, num_fantasies=3, sampler=_sampler(4))
    with pytest.raises(UnsupportedError, match="objective or a posterior transform"):
        qKnowledgeGradient(_model(m=2), num_fantasies=3)
    cost = InverseCostWeightedUtility(AffineFidelityCostModel(fidelity_weights={1: 1.0}))
    with pytest.raises(UnsupportedError, match="current_value"):
        qMultiFidelityKnowledgeGradient(m, num_fantasies=2, cost_aware_utility=cost)
    acqf = qKnowledgeGradient(m, num_fantasies=3)
    with pytest.raises(ValueError, match="q-batch dimension"):
        acqf(_points(1, 2, seed=5))


def test_multi_fidelity_knowledge_gradient_projects() -> None:
    project = functools.partial(project_to_target_fidelity, target_fidelities={1: 1.0}, d=2)
    acqf = qMultiFidelityKnowledgeGradient(
        model(), num_fantasies=3, sampler=_sampler(3), project=project
    )
    X = _points(2, 4, seed=6)
    assert shape(acqf(X)) == (2,)
    _fd_check(acqf, X)


# ---------------------------------------------------------------------------
# qNEI, qLCB, qPSTD, analytic
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("kind", KINDS)
def test_qnei(kind: str) -> None:
    m = model()
    Xb, _ = _data(9, seed=5)
    acqf = qNoisyExpectedImprovement(m, as_kind(Xb, kind), sampler=_sampler(64))
    assert set(map(tuple, flatten_rows(acqf.X_baseline))) <= set(map(tuple, Xb))
    X = _points(3, 2, seed=7)
    values = acqf(as_kind(X, kind))
    check_kind(values, kind)
    assert all(v >= 0.0 for v in flatten(values))
    plain = qNoisyExpectedImprovement(
        m, Xb, sampler=_sampler(64), cache_root=False, prune_baseline=False
    )
    cached = qNoisyExpectedImprovement(
        m, Xb, sampler=_sampler(64), cache_root=True, prune_baseline=False
    )
    assert cached._state.cache_root and not plain._state.cache_root
    _fd_check(cached, X)


def flatten_rows(x: Any) -> list[list[float]]:
    data = flatten(x)
    d = shape(x)[-1]
    return [data[i : i + d] for i in range(0, len(data), d)]


def test_confidence_bounds_and_standard_deviation() -> None:
    m = model()
    X = _points(4, 3, seed=8)
    ucb = flatten(qUpperConfidenceBound(m, beta=0.5, sampler=_sampler(32))(X))
    lcb = flatten(qLowerConfidenceBound(m, beta=0.5, sampler=_sampler(32))(X))
    std = flatten(qPosteriorStandardDeviation(m, sampler=_sampler(32))(X))
    assert all(lo <= up for lo, up in zip(lcb, ucb, strict=True))
    assert all(s >= 0.0 for s in std)
    _fd_check(qLowerConfidenceBound(m, beta=0.5, sampler=_sampler(32)), X)


def test_noisy_expected_improvement() -> None:
    m = _model(fixed_noise=True)
    Xb, _ = _data(9, seed=5)
    nei = NoisyExpectedImprovement(m, Xb, num_fantasies=8, sampler=_sampler(8))
    lnei = LogNoisyExpectedImprovement(m, Xb, num_fantasies=8, sampler=_sampler(8))
    X = _points(3, 1, seed=9)
    assert shape(nei.best_f) == (8,)
    # log of the mean over fantasies of EI.
    for a, b in zip(flatten(lnei(X)), flatten(nei(X)), strict=True):
        assert abs(a - math.log(b)) < 1e-9 * max(1.0, abs(a))
    with pytest.raises(UnsupportedError, match="FixedNoiseGaussianLikelihood"):
        NoisyExpectedImprovement(model(), Xb)


def test_q_analytic_probability_of_improvement() -> None:
    m = model()
    X = _points(3, 3, seed=26)
    acqf = qAnalyticProbabilityOfImprovement(m, best_f=1.0)
    values = flatten(acqf(X))
    assert all(0.0 <= v <= 1.0 for v in values)
    # One point: the analytic probability of improvement.
    first = [[x[0]] for x in X]
    single = flatten(acqf(first))
    pi = flatten(ProbabilityOfImprovement(m, best_f=1.0)(first))
    assert_close(single, pi, rtol=1e-12)
    # More points never lower the probability of improvement.
    assert all(v >= s - 1e-12 for v, s in zip(values, single, strict=True))
    _fd_check(acqf, X)
    _fd_check(qAnalyticProbabilityOfImprovement(m, best_f=1.0, maximize=False), X)


# ---------------------------------------------------------------------------
# Max-value entropy search
# ---------------------------------------------------------------------------


def test_max_value_entropy() -> None:
    m = model()
    cs = [x[0] for x in _points(20, 1, seed=10)]
    mes = qMaxValueEntropy(m, cs, num_mv_samples=6, seed=3)
    assert shape(mes.posterior_max_values) == (6, 1)
    again = qMaxValueEntropy(m, cs, num_mv_samples=6, seed=3)
    assert flatten(again.posterior_max_values) == flatten(mes.posterior_max_values)
    mes.posterior_max_values = [[2.0], [2.5]]
    assert flatten(mes.posterior_max_values) == [2.0, 2.5]
    X = _points(3, 1, seed=11)
    # (A Monte-Carlo estimate: it can be slightly negative.)
    assert all(math.isfinite(v) for v in flatten(mes(X)))
    _fd_check(mes, X)
    gibbon = qLowerBoundMaxValueEntropy(m, cs, seed=3)
    gibbon.set_X_pending([[0.2, 0.3]])
    _fd_check(gibbon, X)
    with pytest.raises(UnsupportedError, match="Multi-output models"):
        qMaxValueEntropy(_model(m=2), cs)
    with pytest.raises(UnsupportedError, match="trace observations"):
        qMultiFidelityLowerBoundMaxValueEntropy(m, cs, expand=lambda X: X)
    with pytest.raises(ValueError):
        mes(_points(1, 2, seed=12))


def test_max_value_samplers() -> None:
    m = model()
    cs = [x[0] for x in _points(30, 1, seed=13)]
    g = _sample_max_value_Gumbel(m, cs, 5, uniforms=[0.1, 0.3, 0.5, 0.7, 0.9])
    assert shape(g) == (5, 1)
    # Increasing uniform numbers give increasing Gumbel samples.
    assert flatten(g) == sorted(flatten(g))
    t = _sample_max_value_Thompson(m, cs, 4, seed=1)
    assert shape(t) == (4, 1)
    assert flatten(t) == flatten(_sample_max_value_Thompson(m, cs, 4, seed=1))


# ---------------------------------------------------------------------------
# Entropy search with optimal samples
# ---------------------------------------------------------------------------


def test_get_optimal_samples() -> None:
    m = model()
    xo, fo = get_optimal_samples(
        m, BOUNDS, 3, raw_samples=64, num_restarts=4, num_features=128, seed=1
    )
    assert shape(xo) == (3, 2) and shape(fo) == (3, 1)
    assert all(0.0 <= v <= 1.0 for v in flatten(xo))
    xo2, fo2 = get_optimal_samples(
        m, BOUNDS, 3, raw_samples=64, num_restarts=4, num_features=128, seed=1
    )
    assert flatten(xo2) == flatten(xo) and flatten(fo2) == flatten(fo)
    m2 = _model(m=2)
    tf = ScalarizedPosteriorTransform([0.5, 0.5])
    _, f2 = get_optimal_samples(
        m2, BOUNDS, 2, raw_samples=32, num_restarts=2, posterior_transform=tf, seed=2
    )
    assert shape(f2) == (2, 2)
    _, f2t = get_optimal_samples(
        m2,
        BOUNDS,
        2,
        raw_samples=32,
        num_restarts=2,
        posterior_transform=tf,
        return_transformed=True,
        seed=2,
    )
    assert shape(f2t) == (2, 1)
    with pytest.raises(ValueError, match="Only one of"):
        get_optimal_samples(m2, BOUNDS, 2, posterior_transform=tf, objective=object())


def test_joint_entropy_search() -> None:
    m = model()
    xo, fo = get_optimal_samples(
        m, BOUNDS, 4, raw_samples=64, num_restarts=4, num_features=128, seed=3
    )
    X = _points(3, 2, seed=14)
    for est in ("LB", "MC"):
        acqf = qJointEntropySearch(m, xo, fo, estimation_type=est, sampler=_sampler(64))
        assert shape(acqf(X)) == (3,)
        _fd_check(acqf, X)
    with pytest.raises(ValueError, match="Estimation type XX is not valid"):
        qJointEntropySearch(m, xo, fo, estimation_type="XX")


def test_joint_entropy_search_mc_is_additive_over_independent_points() -> None:
    """The "MC" estimate of two points far apart (nearly independent
    observations) is the sum of their estimates (meta-pytorch/botorch#3428:
    BoTorch's is not)."""
    X, Y = _data(9, seed=5, d=1)
    X = [[3.0 * x[0]] for x in X]
    m = SingleTaskGP(X, Y)
    m.covar_module.lengthscale = [[0.05]]
    xo, fo = get_optimal_samples(
        m, [[0.0], [3.0]], 4, raw_samples=64, num_restarts=4, num_features=128, seed=4
    )
    # The same normal numbers for every point in the joint and the single
    # evaluations.
    rng = random.Random(7)
    z = [[rng.gauss(0.0, 1.0), rng.gauss(0.0, 1.0)] for _ in range(256)]

    def acqf(cols: list[int]) -> Any:
        s = _sampler(256)
        s.base_samples = [[row[c] for c in cols] for row in z]
        return qJointEntropySearch(m, xo, fo, estimation_type="MC", sampler=s)

    a, b = [-1.0], [5.0]
    joint = flatten(acqf([0, 1])([[a, b]]))[0]
    single = flatten(acqf([0])([[a]]))[0] + flatten(acqf([1])([[b]]))[0]
    assert abs(joint - single) <= 1e-12 * max(1.0, abs(single))


def test_predictive_entropy_search() -> None:
    m = model()
    xo, _ = get_optimal_samples(
        m, BOUNDS, 3, raw_samples=64, num_restarts=4, num_features=128, seed=5
    )
    acqf = qPredictiveEntropySearch(m, xo)
    assert acqf.ep_iterations >= 1
    assert shape(acqf(_points(2, 2, seed=15))) == (2,)
    with pytest.raises(UnsupportedError, match="single-output"):
        qPredictiveEntropySearch(_model(m=2), xo)


# ---------------------------------------------------------------------------
# Thompson sampling and active learning
# ---------------------------------------------------------------------------


def test_pathwise_thompson_sampling() -> None:
    m = model()
    ts = PathwiseThompsonSampling(m, num_features=256, seed=6)
    assert ts.batch_size is None
    X = _points(3, 2, seed=16)
    v = flatten(ts(X))
    assert ts.batch_size == 2
    assert flatten(ts(X)) == v
    # The same seed draws the same paths; every redraw uses a new stream.
    ts1 = PathwiseThompsonSampling(m, num_features=256, seed=6)
    ts1.redraw(2)
    assert flatten(ts1(X)) == v
    ts1.redraw(2)
    assert flatten(ts1(X)) != v
    _fd_check(ts, X)
    with pytest.raises(ValueError, match="should not change during a forward pass"):
        ts(_points(1, 3, seed=17))
    with pytest.raises(UnsupportedError, match="objective or a posterior transform"):
        PathwiseThompsonSampling(_model(m=2))


def test_neg_integrated_posterior_variance() -> None:
    m = model()
    mc = [x[0] for x in _points(16, 1, seed=18)]
    acqf = qNegIntegratedPosteriorVariance(m, mc)
    X = _points(3, 1, seed=19)
    v1 = flatten(acqf(X))
    assert all(v <= 0.0 for v in v1)
    # More observations never increase the posterior variance.
    X2 = [x + [[0.5, 0.5]] for x in X]
    v2 = flatten(acqf(X2))
    assert all(b >= a - 1e-12 for a, b in zip(v1, v2, strict=True))
    _fd_check(acqf, X2)
    lst = ModelListGP(_model(), _model(n=7))
    tf = ScalarizedPosteriorTransform([0.5, 0.5])
    _fd_check(qNegIntegratedPosteriorVariance(lst, mc, posterior_transform=tf), X)


# ---------------------------------------------------------------------------
# Penalties and wrappers
# ---------------------------------------------------------------------------


def test_penalized_acquisition_function_closed_forms() -> None:
    m = model()
    ei = ExpectedImprovement(m, best_f=0.5)
    init = [0.3, 0.6]
    X = _points(3, 1, seed=20)
    base = flatten(ei(X))
    for penalty, f in (
        (L2Penalty(init), lambda x: sum((a - b) ** 2 for a, b in zip(x, init, strict=True))),
        (L1Penalty(init), lambda x: sum(abs(a - b) for a, b in zip(x, init, strict=True))),
        (
            GaussianPenalty(init, 0.5),
            lambda x: math.exp(sum((a - b) ** 2 for a, b in zip(x, init, strict=True)) / 0.5),
        ),
        (GroupLassoPenalty(init, [[0, 1]]), lambda x: math.sqrt(2) * math.dist(x, init)),
        (
            L0PenaltyApprox(init, 0.5),
            lambda x: (
                2 - sum(math.exp(-0.5 * ((a - b) / 0.5) ** 2) for a, b in zip(x, init, strict=True))
            ),
        ),
    ):
        acqf = PenalizedAcquisitionFunction(ei, penalty, 0.3)
        expected = [b - 0.3 * f(x[0]) for b, x in zip(base, X, strict=True)]
        assert_close(flatten(acqf(X)), expected, rtol=1e-12)
        _fd_check(acqf, X)
    with pytest.raises(UnsupportedError, match="Analytic"):
        PenalizedAcquisitionFunction(ei, L2Penalty(init), 0.3).set_X_pending([[0.1, 0.1]])


def test_proximal_acquisition_function_closed_form() -> None:
    m = model()
    pm = PosteriorMean(m)
    ei = ExpectedImprovement(m, best_f=0.5)
    w = [0.3, 0.5]
    X = _points(3, 1, seed=21)
    last = _data(9, seed=5)[0][-1]
    acqf = ProximalAcquisitionFunction(ei, w)
    weights = [
        math.exp(-0.5 * sum(((a - b) / c) ** 2 for a, b, c in zip(x[0], last, w, strict=True)))
        for x in X
    ]
    expected = [e * k for e, k in zip(flatten(ei(X)), weights, strict=True)]
    assert_close(flatten(acqf(X)), expected, rtol=1e-12)
    _fd_check(acqf, X)
    with pytest.raises(ValueError, match="one dimensional tensor"):
        ProximalAcquisitionFunction(ei, [1.0, 1.0, 1.0])
    with pytest.raises(RuntimeError, match="negative acquisition function values"):
        ProximalAcquisitionFunction(PosteriorMean(m, maximize=False), w)(X)
    _fd_check(ProximalAcquisitionFunction(pm, w, beta=2.0), X)


def test_fixed_feature_acquisition_function() -> None:
    m = model()
    qei = qExpectedImprovement(m, best_f=0.5, sampler=_sampler(32))
    acqf = FixedFeatureAcquisitionFunction(qei, 2, [1], [0.4])
    X = [[[0.2], [0.7]], [[0.5], [0.1]]]
    full = [[[x[0], 0.4] for x in xs] for xs in X]
    assert_close(flatten(acqf(X)), flatten(qei(full)), rtol=1e-14)
    _fd_check(acqf, X)
    acqf.X_pending = [[0.3]]
    assert flatten(qei.X_pending) == [0.3, 0.4]
    with pytest.raises(ValueError, match="Feature dimension"):
        acqf(_points(1, 2, seed=22))


def test_prior_guided_acquisition_function() -> None:
    if not HAS_TORCH:
        pytest.skip("the prior is a torch module")
    import torch

    def prior(X: Any) -> Any:
        return torch.exp(-0.5 * ((X - 0.5) / 0.3).pow(2).sum(-1))

    m = model()
    ei = ExpectedImprovement(m, best_f=0.5)
    acqf = PriorGuidedAcquisitionFunction(ei, prior, prior_exponent=0.5)
    X = torch.tensor(_points(3, 1, seed=23), dtype=torch.float64)
    expected = ei(X) * prior(X).squeeze(-1) ** 0.5
    assert_close(flatten(acqf(X)), flatten(expected), rtol=1e-12)
    _fd_check(acqf, X.tolist())
    qei = qExpectedImprovement(m, best_f=0.5, sampler=_sampler(32))
    guided = PriorGuidedAcquisitionFunction(qei, prior, X_pending=[[0.1, 0.9]])
    _fd_check(guided, _points(2, 2, seed=24))
    with pytest.raises(NotImplementedError, match="q-batches with q>1"):
        acqf(torch.tensor(_points(1, 2, seed=25), dtype=torch.float64))


def test_decoupled_acquisition_function_masks() -> None:
    class Decoupled(DecoupledAcquisitionFunction):
        pass

    lst = ModelListGP(_model(), _model(n=7))
    with pytest.raises(ValueError, match="requires using a ModelList"):
        Decoupled(model())
    acqf = Decoupled(lst, X_evaluation_mask=[[True, False]])
    with pytest.raises(BotorchTensorDimensionError):
        acqf.X_evaluation_mask = [[True]]
    with pytest.raises(ValueError, match="X_pending_evaluation_mask"):
        acqf.set_X_pending([[0.1, 0.2]])
    acqf.set_X_pending([[0.1, 0.2]], [[False, True]])
    if HAS_TORCH:
        mask = acqf.construct_evaluation_mask([[0.3, 0.4]])
        assert mask.tolist() == [[True, False], [False, True]]

"""Acquisition functions: shapes, array types, pending points, gradients."""

from __future__ import annotations

import math
import warnings
from typing import Any

import pytest

from botorch_rs.acquisition import (
    AcquisitionFunction,
    ExpectedImprovement,
    LogExpectedImprovement,
    LogProbabilityOfImprovement,
    PosteriorMean,
    PosteriorStandardDeviation,
    ProbabilityOfImprovement,
    UpperConfidenceBound,
    prune_inferior_points,
    qExpectedImprovement,
    qLogExpectedImprovement,
    qLogNoisyExpectedImprovement,
    qProbabilityOfImprovement,
    qSimpleRegret,
    qUpperConfidenceBound,
)
from botorch_rs.exceptions import BotorchTensorDimensionError, NumericsWarning, UnsupportedError
from botorch_rs.fit import fit_gpytorch_mll
from botorch_rs.mlls import ExactMarginalLogLikelihood
from botorch_rs.models import SingleTaskGP
from botorch_rs.sampling import SobolQMCNormalSampler
from helpers import HAS_TORCH, as_kind, assert_close, check_kind, flatten, make_data, shape

X_TRAIN, Y_TRAIN = make_data(n=14, d=2, seed=7)
BEST_F = max(y[0] for y in Y_TRAIN)

ANALYTIC: list[tuple[type, dict[str, Any]]] = [
    (ExpectedImprovement, {"best_f": BEST_F}),
    (LogExpectedImprovement, {"best_f": BEST_F}),
    (ProbabilityOfImprovement, {"best_f": BEST_F}),
    (LogProbabilityOfImprovement, {"best_f": BEST_F}),
    (UpperConfidenceBound, {"beta": 0.2}),
    (PosteriorMean, {}),
    (PosteriorStandardDeviation, {}),
]
# An incumbent below the best observation, so that improvements are likely.
MC_BEST_F = BEST_F - 0.5
MONTE_CARLO: list[tuple[type, dict[str, Any]]] = [
    (qExpectedImprovement, {"best_f": MC_BEST_F}),
    (qLogExpectedImprovement, {"best_f": MC_BEST_F}),
    (qProbabilityOfImprovement, {"best_f": MC_BEST_F, "tau": 0.05}),
    (qUpperConfidenceBound, {"beta": 0.2}),
    (qSimpleRegret, {}),
    (qLogNoisyExpectedImprovement, {"X_baseline": X_TRAIN}),
]
POINTS = [[[0.1, 0.9]], [[0.5, 0.5]], [[0.95, 0.05]]]
SETS = [[[0.1, 0.9], [0.4, 0.2]], [[0.5, 0.5], [0.8, 0.7]]]


@pytest.fixture(scope="module")
def model() -> SingleTaskGP:
    m = SingleTaskGP(X_TRAIN, Y_TRAIN)
    fit_gpytorch_mll(ExactMarginalLogLikelihood(m.likelihood, m))
    return m


def make(cls: type, model: SingleTaskGP, **kwargs: Any) -> AcquisitionFunction:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", NumericsWarning)
        if cls in [c for c, _ in MONTE_CARLO]:
            kwargs.setdefault("sampler", SobolQMCNormalSampler(sample_shape=[256], seed=3))
        return cls(model, **kwargs)


@pytest.mark.parametrize("cls,kwargs", ANALYTIC, ids=[c.__name__ for c, _ in ANALYTIC])
def test_analytic_shapes(cls: type, kwargs: dict, kind: str, model: SingleTaskGP) -> None:
    acqf = make(cls, model, **kwargs)
    values = acqf(as_kind(POINTS, kind))
    check_kind(values, kind)
    assert shape(values) == (3,)
    assert shape(acqf(as_kind(POINTS[0], kind))) == (1,)  # a single `q x d` set
    assert shape(acqf(as_kind([POINTS] * 2, kind))) == (2, 3)
    assert all(math.isfinite(v) for v in flatten(values))
    with pytest.raises(BotorchTensorDimensionError):
        acqf(as_kind(SETS, kind))  # q = 2
    with pytest.raises(ValueError):
        acqf(as_kind([0.5, 0.5], kind))
    with pytest.raises(UnsupportedError):
        acqf.set_X_pending(as_kind([[0.5, 0.5]], kind))


def test_analytic_relations(model: SingleTaskGP) -> None:
    ei = flatten(make(ExpectedImprovement, model, best_f=BEST_F)(POINTS))
    log_ei = flatten(make(LogExpectedImprovement, model, best_f=BEST_F)(POINTS))
    assert_close(ei, [math.exp(v) for v in log_ei], rtol=1e-10)
    pi = flatten(make(ProbabilityOfImprovement, model, best_f=BEST_F)(POINTS))
    log_pi = flatten(make(LogProbabilityOfImprovement, model, best_f=BEST_F)(POINTS))
    assert_close(pi, [math.exp(v) for v in log_pi], rtol=1e-10)
    post = model.posterior(POINTS)
    mean, var = flatten(post.mean), flatten(post.variance)
    assert_close(flatten(make(PosteriorMean, model)(POINTS)), mean)
    assert_close(flatten(make(PosteriorMean, model, maximize=False)(POINTS)), [-m for m in mean])
    assert_close(
        flatten(make(PosteriorStandardDeviation, model)(POINTS)), [math.sqrt(v) for v in var]
    )
    ucb = flatten(make(UpperConfidenceBound, model, beta=0.2)(POINTS))
    assert_close(ucb, [m + math.sqrt(0.2 * v) for m, v in zip(mean, var, strict=True)])
    # Minimization: EI of -f at -best_f.
    ei_min = flatten(make(ExpectedImprovement, model, best_f=BEST_F, maximize=False)(POINTS))
    assert all(v >= 0 for v in ei_min)


@pytest.mark.parametrize("cls,kwargs", MONTE_CARLO, ids=[c.__name__ for c, _ in MONTE_CARLO])
def test_monte_carlo_shapes_and_pending(
    cls: type, kwargs: dict, kind: str, model: SingleTaskGP
) -> None:
    acqf = make(cls, model, **kwargs)
    values = acqf(as_kind(SETS, kind))
    check_kind(values, kind)
    assert shape(values) == (2,)
    assert shape(acqf(as_kind(SETS[0], kind))) == (1,)
    assert shape(acqf(as_kind([SETS] * 3, kind))) == (3, 2)
    assert acqf.X_pending is None
    acqf.set_X_pending(as_kind([[0.3, 0.3], [0.7, 0.1]], kind))
    pending = acqf.X_pending
    check_kind(pending, kind)
    assert shape(pending) == (2, 2)
    with_pending = flatten(acqf(as_kind(SETS, kind)))
    acqf.set_X_pending(None)
    assert acqf.X_pending is None
    assert flatten(acqf(as_kind(SETS, kind))) == flatten(values)
    assert with_pending != flatten(values)
    with pytest.raises(BotorchTensorDimensionError):
        acqf.set_X_pending(as_kind([[0.3, 0.3, 0.3]], kind))
    constructed = make(cls, model, X_pending=as_kind([[0.3, 0.3], [0.7, 0.1]], kind), **kwargs)
    assert_close(flatten(constructed(as_kind(SETS, kind))), with_pending, rtol=1e-12)


def test_monte_carlo_agrees_with_analytic(model: SingleTaskGP) -> None:
    sampler = SobolQMCNormalSampler(sample_shape=[4096], seed=0)
    for mc_cls, an_cls, kwargs in [
        (qExpectedImprovement, ExpectedImprovement, {"best_f": MC_BEST_F}),
        (qSimpleRegret, PosteriorMean, {}),
    ]:
        mc = flatten(make(mc_cls, model, sampler=sampler, **kwargs)(POINTS))
        an = flatten(make(an_cls, model, **kwargs)(POINTS))
        assert_close(mc, an, rtol=2e-3, atol=2e-3)
    # Where improvement is likely, the smoothed log qEI is close to log EI.
    log_qei = make(qLogExpectedImprovement, model, sampler=sampler, best_f=MC_BEST_F)(POINTS)
    log_ei = make(LogExpectedImprovement, model, best_f=MC_BEST_F)(POINTS)
    ei = flatten(make(ExpectedImprovement, model, best_f=MC_BEST_F)(POINTS))
    for a, b, e in zip(flatten(log_qei), flatten(log_ei), ei, strict=True):
        if e > 1e-3:
            assert abs(a - b) < 0.02, (a, b)


def test_sampler_controls_the_base_samples(model: SingleTaskGP) -> None:
    def value(seed: int) -> float:
        s = SobolQMCNormalSampler(sample_shape=[64], seed=seed)
        return flatten(qLogExpectedImprovement(model, best_f=BEST_F, sampler=s)(SETS))[0]

    assert value(1) == value(1)
    assert value(1) != value(2)
    default = qLogExpectedImprovement(model, best_f=BEST_F)
    assert default.sampler.sample_shape == (512,)


ALL = ANALYTIC + MONTE_CARLO


@pytest.mark.parametrize("cls,kwargs", ALL, ids=[c.__name__ for c, _ in ALL])
def test_gradients_match_finite_differences(cls: type, kwargs: dict, model: SingleTaskGP) -> None:
    acqf = make(cls, model, **kwargs)
    X = [[[0.3, 0.6]]] if cls in [c for c, _ in ANALYTIC] else [[[0.3, 0.6], [0.7, 0.4]]]
    values, grads = acqf.value_and_grad(X)
    assert shape(grads) == shape(X)
    assert_close(values, acqf(X), rtol=1e-14)
    h = 1e-6
    q = len(X[0])
    for i in range(q):
        for j in range(2):
            up = [[list(p) for p in X[0]]]
            down = [[list(p) for p in X[0]]]
            up[0][i][j] += h
            down[0][i][j] -= h
            fd = (flatten(acqf(up))[0] - flatten(acqf(down))[0]) / (2 * h)
            assert grads[0][i][j] == pytest.approx(fd, rel=1e-4, abs=1e-6)


@pytest.mark.skipif(not HAS_TORCH, reason="torch is not installed")
def test_torch_autograd(model: SingleTaskGP) -> None:
    import torch

    for cls, kwargs in [
        (LogExpectedImprovement, {"best_f": BEST_F}),
        (qLogExpectedImprovement, {"best_f": BEST_F}),
    ]:
        acqf = make(cls, model, **kwargs)
        q = 1 if cls is LogExpectedImprovement else 2
        X = torch.rand(4, q, 2, dtype=torch.float64, requires_grad=True)
        values = acqf(X)
        assert values.requires_grad and values.shape == (4,)
        (grad,) = torch.autograd.grad(values.sum(), X)
        _, expected = acqf.value_and_grad(X.detach())
        assert torch.allclose(grad, expected, rtol=1e-12, atol=0)
        # 2-d input and chained operations.
        X2 = X[0].detach().clone().requires_grad_(True)
        (2.0 * acqf(X2)).sum().backward()
        assert torch.allclose(X2.grad, 2.0 * expected[0], rtol=1e-12, atol=0)
        with torch.no_grad():
            assert not acqf(X).requires_grad


def test_nei_baseline_and_pruning(model: SingleTaskGP) -> None:
    sampler = SobolQMCNormalSampler(sample_shape=[128], seed=4)
    full = qLogNoisyExpectedImprovement(
        model, X_baseline=X_TRAIN, sampler=sampler, prune_baseline=False
    )
    assert_close(full.X_baseline, X_TRAIN, rtol=0, atol=0)
    pruned = qLogNoisyExpectedImprovement(model, X_baseline=X_TRAIN, sampler=sampler)
    kept = [list(x) for x in pruned.X_baseline]
    assert 1 <= len(kept) <= len(X_TRAIN) and all(x in X_TRAIN for x in kept)
    pruned.set_X_pending([[0.2, 0.2]])
    assert len(pruned.X_baseline) == len(pruned._X_baseline) + 1  # incremental
    points = prune_inferior_points(model, X_TRAIN, sampler=sampler)
    assert all(list(x) in X_TRAIN for x in points)
    assert len(prune_inferior_points(model, X_TRAIN, max_frac=0.1)) <= 2
    # Reproducible for a given sampler seed (pruning is seeded from it).
    again = qLogNoisyExpectedImprovement(
        model, X_baseline=X_TRAIN, sampler=SobolQMCNormalSampler([128], seed=4)
    )
    assert flatten(again(SETS)) == flatten(
        qLogNoisyExpectedImprovement(model, X_baseline=X_TRAIN, sampler=sampler)(SETS)
    )


def test_unsupported_arguments(model: SingleTaskGP) -> None:
    for kwargs in (
        {"objective": object()},
        {"posterior_transform": object()},
    ):
        with pytest.raises(UnsupportedError):
            qLogExpectedImprovement(model, best_f=0.0, **kwargs)
    with pytest.raises(TypeError, match="callables"):
        qLogExpectedImprovement(model, best_f=0.0, constraints=[object()])
    with pytest.raises(UnsupportedError):
        LogExpectedImprovement(model, best_f=0.0, posterior_transform=object())
    with pytest.raises(UnsupportedError):
        qLogNoisyExpectedImprovement(model, X_baseline=X_TRAIN, marginalize_dim=0)
    with pytest.raises(UnsupportedError):
        qLogExpectedImprovement(model, best_f=0.0, sampler=object())
    with pytest.raises(UnsupportedError):
        LogExpectedImprovement(object(), best_f=0.0)
    with pytest.raises(UnsupportedError):
        LogExpectedImprovement(model, best_f=[0.0, 1.0])  # batched best_f
    # Accepted BoTorch arguments without effect.
    qLogExpectedImprovement(model, best_f=0.0, constraints=None, eta=1e-2)
    qLogNoisyExpectedImprovement(model, X_baseline=X_TRAIN, cache_root=True)


def test_legacy_ei_warns(model: SingleTaskGP) -> None:
    with pytest.warns(NumericsWarning, match="LogExpectedImprovement"):
        ExpectedImprovement(model, best_f=0.0)
    with pytest.warns(NumericsWarning, match="qLogExpectedImprovement"):
        qExpectedImprovement(model, best_f=0.0)

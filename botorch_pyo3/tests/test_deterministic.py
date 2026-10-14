"""Deterministic models, ``StochasticSampler`` and the composite and callable
acquisition functions through the public API, for every array type: values
and gradients (finite differences, torch.autograd), independence of the
sampler, pending points, ``optimize_acqf``, thread-count independence and
errors. The comparisons with BoTorch are in ``test_parity_deterministic.py``.
"""

from __future__ import annotations

import math
import os
import random
import subprocess
import sys
import textwrap
from typing import Any

import pytest

import botorch_rs
from botorch_rs.acquisition import (
    CallableAcquisitionFunction,
    LogExpectedImprovement,
    PosteriorMean,
    WeightedLogMeanExp,
    qLogNoisyExpectedImprovement,
    qSimpleRegret,
)
from botorch_rs.acquisition.multi_objective import (
    qLogExpectedHypervolumeImprovement,
    qLogNoisyExpectedHypervolumeImprovement,
)
from botorch_rs.acquisition.objective import ScalarizedPosteriorTransform
from botorch_rs.exceptions import UnsupportedError
from botorch_rs.models import (
    GenericDeterministicModel,
    ModelListGP,
    PosteriorMeanModel,
    SingleTaskGP,
)
from botorch_rs.optim import optimize_acqf
from botorch_rs.sampling import SobolQMCNormalSampler, StochasticSampler
from botorch_rs.utils.multi_objective.box_decompositions import NondominatedPartitioning
from helpers import HAS_TORCH, as_kind, assert_close, check_kind, flatten, shape


def _data(n: int, seed: int) -> tuple[list, list]:
    rng = random.Random(seed)
    X = [[rng.random(), rng.random()] for _ in range(n)]
    Y = [
        [
            1.0 - (x[0] - 0.3) ** 2 - (x[1] - 0.6) ** 2 + 0.01 * rng.gauss(0.0, 1.0),
            1.0 - (x[0] - 0.7) ** 2 - (x[1] - 0.2) ** 2 + 0.01 * rng.gauss(0.0, 1.0),
        ]
        for x in X
    ]
    return X, Y


def _model_list(kind: str, seed: int = 0) -> ModelListGP:
    X, Y = _data(9, seed)
    Xk = as_kind(X, kind)
    models = []
    for j in range(2):
        gp = SingleTaskGP(Xk, as_kind([[y[j]] for y in Y], kind))
        gp.covar_module.lengthscale = as_kind([[0.3 + 0.1 * j, 0.5]], kind)
        gp.likelihood.noise = 1e-3
        models.append(gp)
    return ModelListGP(*models)


def _front(kind: str) -> tuple[list[float], Any]:
    ref = [0.0, 0.0]
    front = [[0.9, 0.3], [0.6, 0.6], [0.3, 0.85]]
    return ref, NondominatedPartitioning(ref_point=as_kind(ref, kind), Y=as_kind(front, kind))


def _points(n: int, q: int, seed: int) -> list:
    rng = random.Random(seed)
    return [[[rng.random(), rng.random()] for _ in range(q)] for _ in range(n)]


# ---------------------------------------------------------------------------
# PosteriorMeanModel
# ---------------------------------------------------------------------------


def test_posterior_mean_model(kind: str) -> None:
    ml = _model_list(kind)
    pm = PosteriorMeanModel(ml)
    assert pm.num_outputs == 2 and tuple(pm.batch_shape) == ()
    X = as_kind(_points(3, 2, 1), kind)
    post = pm.posterior(X)
    check_kind(post.mean, kind)
    assert_close(post.mean, ml.posterior(X).mean, rtol=0.0, atol=0.0)
    assert shape(post.variance) == (3, 2, 2) and max(map(abs, flatten(post.variance))) == 0.0
    samples = post.rsample([4])
    assert shape(samples) == (4, 3, 2, 2)
    assert_close(flatten(samples)[:12], flatten(post.mean), rtol=0.0, atol=0.0)
    assert_close(pm(X), post.mean, rtol=0.0, atol=0.0)
    with pytest.raises(UnsupportedError):
        pm.posterior(X, posterior_transform=ScalarizedPosteriorTransform(as_kind([1.0, 1.0], kind)))


def test_ehvi_on_posterior_mean_does_not_depend_on_the_sampler(kind: str) -> None:
    pm = PosteriorMeanModel(_model_list(kind))
    ref, part = _front(kind)
    X = as_kind(_points(5, 1, 2), kind)
    values = []
    for sampler in (
        StochasticSampler([1]),
        StochasticSampler([7], seed=3),
        SobolQMCNormalSampler([16], seed=1),
        None,
    ):
        acqf = qLogExpectedHypervolumeImprovement(
            pm, ref, part, sampler=sampler, tau_max=1e-3, fat=True
        )
        v = acqf(X)
        check_kind(v, kind)
        values.append(flatten(v))
    for v in values[1:]:
        assert_close(v, values[0], rtol=1e-14, atol=1e-14)


def test_gradients_through_the_posterior_mean(kind: str) -> None:
    pm = PosteriorMeanModel(_model_list(kind))
    ref, part = _front(kind)
    acqf = qLogExpectedHypervolumeImprovement(
        pm, ref, part, sampler=StochasticSampler([1]), tau_max=1e-3, fat=True
    )
    X = _points(3, 2, 3)
    values, grads = acqf.value_and_grad(as_kind(X, kind))
    assert shape(grads) == (3, 2, 2)
    h = 1e-6
    g = flatten(grads)
    for b in range(3):
        for i in range(2):
            for k in range(2):
                xp = [[list(p) for p in s] for s in X]
                xm = [[list(p) for p in s] for s in X]
                xp[b][i][k] += h
                xm[b][i][k] -= h
                fd = (flatten(acqf(xp))[b] - flatten(acqf(xm))[b]) / (2 * h)
                assert abs(fd - g[b * 4 + i * 2 + k]) <= 1e-5 * max(1.0, abs(fd))


def test_noisy_acquisition_functions_on_posterior_mean(kind: str) -> None:
    ml = _model_list(kind)
    X, _ = _data(9, 0)
    pm = PosteriorMeanModel(ml)
    acqf = qLogNoisyExpectedHypervolumeImprovement(
        pm, [0.0, 0.0], as_kind(X, kind), sampler=StochasticSampler([1]), prune_baseline=True
    )
    v = flatten(acqf(as_kind(_points(4, 1, 5), kind)))
    assert all(math.isfinite(x) for x in v)
    single = PosteriorMeanModel(ml.models[0])
    nei = qLogNoisyExpectedImprovement(single, as_kind(X, kind), sampler=StochasticSampler([1]))
    assert all(math.isfinite(x) for x in flatten(nei(as_kind(_points(4, 2, 6), kind))))
    # The samples are the mean: qSimpleRegret is the largest mean of the q points.
    sr = qSimpleRegret(single, sampler=StochasticSampler([3]))
    Xq = as_kind(_points(4, 2, 7), kind)
    means = flatten(single.posterior(Xq).mean)
    expected = [max(means[2 * b], means[2 * b + 1]) for b in range(4)]
    assert_close(sr(Xq), expected, rtol=1e-14)


def test_analytic_on_posterior_mean_uses_zero_variance(kind: str) -> None:
    single = PosteriorMeanModel(_model_list(kind).models[0])
    X = as_kind(_points(3, 1, 8), kind)
    mean = flatten(single(X))
    assert_close(PosteriorMean(single)(X), mean, rtol=1e-15)
    # sigma = sqrt(1e-12) as in BoTorch's _mean_and_sigma.
    sigma = math.sqrt(1e-12)
    best_f = mean[0] - 0.5e-6
    u = (mean[0] - best_f) / sigma
    ei = math.exp(-0.5 * u * u) / math.sqrt(2 * math.pi) + u * 0.5 * math.erfc(-u / math.sqrt(2))
    v = flatten(LogExpectedImprovement(single, best_f=best_f)(X))
    assert abs(v[0] - (math.log(ei) + math.log(sigma))) <= 1e-9


def test_stochastic_sampler_errors(kind: str) -> None:
    ml = _model_list(kind)
    ref, part = _front(kind)
    with pytest.raises(UnsupportedError, match="deterministic"):
        qLogExpectedHypervolumeImprovement(ml, ref, part, sampler=StochasticSampler([1]))
    s = StochasticSampler([2, 3], seed=5)
    assert s.sample_shape == (2, 3) and s.num_samples == 6 and s.seed == 5
    post = ml.posterior(as_kind(_points(1, 2, 9)[0], kind))
    assert shape(s(post)) == (2, 3, 2, 2)


# ---------------------------------------------------------------------------
# WeightedLogMeanExp
# ---------------------------------------------------------------------------


def _terms(kind: str) -> list[Any]:
    ref, part = _front(kind)
    out = []
    for seed in (0, 1):
        out.append(
            qLogExpectedHypervolumeImprovement(
                PosteriorMeanModel(_model_list(kind, seed)),
                ref,
                part,
                sampler=StochasticSampler([1]),
                tau_max=1e-3,
                fat=True,
            )
        )
    X, _ = _data(9, 0)
    out.append(
        qLogNoisyExpectedHypervolumeImprovement(
            _model_list(kind, 2),
            ref,
            as_kind(X, kind),
            sampler=SobolQMCNormalSampler([16], seed=0),
        )
    )
    return out


def test_weighted_log_mean_exp_values(kind: str) -> None:
    terms = _terms(kind)
    w = [0.3, 0.0, 1.2]
    acqf = WeightedLogMeanExp(terms, as_kind(w, kind))
    X = as_kind(_points(6, 1, 10), kind)
    v = acqf(X)
    check_kind(v, kind)
    a = [flatten(t(X)) for t in terms]
    for b in range(6):
        expected = math.log((0.3 * math.exp(a[0][b]) + 1.2 * math.exp(a[2][b])) / 1.5)
        assert abs(flatten(v)[b] - expected) <= 1e-12 * max(1.0, abs(expected))
    values, grads = acqf.value_and_grad(X)
    assert_close(values, v, rtol=0.0, atol=0.0)
    h = 1e-6
    Xl = _points(6, 1, 10)
    g = flatten(grads)
    for b in range(2):
        for k in range(2):
            xp = [[list(p) for p in s] for s in Xl]
            xm = [[list(p) for p in s] for s in Xl]
            xp[b][0][k] += h
            xm[b][0][k] -= h
            fd = (flatten(acqf(xp))[b] - flatten(acqf(xm))[b]) / (2 * h)
            assert abs(fd - g[b * 2 + k]) <= 1e-5 * max(1.0, abs(fd))


def test_weighted_log_mean_exp_pending_points(kind: str) -> None:
    terms = _terms(kind)
    acqf = WeightedLogMeanExp(terms, [1.0, 1.0, 1.0])
    assert acqf.X_pending is None
    P = as_kind([[0.5, 0.5]], kind)
    acqf.set_X_pending(P)
    for t in terms:
        assert_close(t.X_pending, [[0.5, 0.5]], rtol=0.0)
    assert_close(acqf.X_pending, [[0.5, 0.5]], rtol=0.0)
    acqf.set_X_pending(None)
    assert all(t.X_pending is None for t in terms)


def test_weighted_log_mean_exp_optimize_and_threads(kind: str) -> None:
    acqf = WeightedLogMeanExp(_terms(kind), [0.6, 0.3, 0.1])
    bounds = as_kind([[0.0, 0.0], [1.0, 1.0]], kind)
    results = []
    for threads in (1, 3):
        botorch_rs.set_num_threads(threads)
        try:
            results.append(
                optimize_acqf(
                    acqf, bounds, q=1, num_restarts=3, raw_samples=32, options={"seed": 3}
                )
            )
        finally:
            botorch_rs.set_num_threads(None)
    (x1, v1), (x2, v2) = results
    check_kind(x1, kind)
    assert flatten(x1) == flatten(x2) and flatten(v1) == flatten(v2)
    assert_close(acqf(as_kind([flatten(x1)], kind)), v1, rtol=1e-14)


def test_weighted_log_mean_exp_errors(kind: str) -> None:
    terms = _terms(kind)
    with pytest.raises(ValueError):
        WeightedLogMeanExp([], [])
    with pytest.raises(ValueError):
        WeightedLogMeanExp(terms, [1.0, 1.0])
    with pytest.raises(ValueError):
        WeightedLogMeanExp(terms, [1.0, -0.5, 1.0])
    with pytest.raises(ValueError):
        WeightedLogMeanExp(terms, [0.0, 0.0, 0.0])
    with pytest.raises(UnsupportedError):
        WeightedLogMeanExp([lambda X: X.sum((-1, -2))], [1.0])


# ---------------------------------------------------------------------------
# Callable acquisition functions and GenericDeterministicModel (torch)
# ---------------------------------------------------------------------------


needs_torch = pytest.mark.skipif(not HAS_TORCH, reason="callables operate on torch tensors")


@needs_torch
def test_optimize_acqf_with_a_torch_module() -> None:
    import torch

    terms = _terms("torch")
    w = [0.6, 0.3, 0.1]

    class Blend(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.X_pending = None

        def forward(self, X: Any) -> Any:
            z = torch.stack([math.log(wi) + t(X) for t, wi in zip(terms, w, strict=True)], -1)
            return torch.logsumexp(z, -1) - math.log(sum(w))

    native = WeightedLogMeanExp(terms, w)
    X = torch.tensor(_points(7, 1, 11), dtype=torch.float64)
    blend = CallableAcquisitionFunction(Blend(), dim=2)
    assert_close(blend(X), native(X), rtol=1e-13)
    v, g = blend.value_and_grad(X)
    vn, gn = native.value_and_grad(X)
    assert_close(v, vn, rtol=1e-13)
    assert_close(g, gn, rtol=1e-10, atol=1e-12)
    bounds = torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.float64)
    ics = torch.tensor(_points(3, 1, 12), dtype=torch.float64)
    x_c, v_c = optimize_acqf(
        Blend(), bounds, q=1, num_restarts=3, batch_initial_conditions=ics, return_best_only=False
    )
    x_n, v_n = optimize_acqf(
        native, bounds, q=1, num_restarts=3, batch_initial_conditions=ics, return_best_only=False
    )
    assert isinstance(x_c, torch.Tensor) and x_c.shape == (3, 1, 2)
    assert_close(v_c, v_n, rtol=1e-9)
    x_best, v_best = optimize_acqf(Blend(), bounds, q=1, num_restarts=3, raw_samples=16)
    assert x_best.shape == (1, 2) and math.isfinite(float(v_best))


@needs_torch
def test_callable_acquisition_pending_points_and_errors() -> None:
    import torch

    term = _terms("torch")[0]

    class WithPending(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.X_pending = None
            self.calls: list[Any] = []

        def set_X_pending(self, X_pending: Any = None) -> None:
            self.calls.append(None if X_pending is None else X_pending.clone())
            self.X_pending = X_pending

        def forward(self, X: Any) -> Any:
            return term(X)

    f = WithPending()
    bounds = torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.float64)
    X, _ = optimize_acqf(f, bounds, q=2, num_restarts=2, raw_samples=8, sequential=True)
    assert X.shape == (2, 2)
    # The first candidate was pending in the second step; the original
    # (empty) pending points are restored at the end.
    assert len(f.calls) == 2 and f.calls[-1] is None
    assert torch.allclose(f.calls[0], X[:1])

    class Boom(torch.nn.Module):
        def forward(self, X: Any) -> Any:
            raise KeyError("boom")

    with pytest.raises(KeyError, match="boom"):
        optimize_acqf(Boom(), bounds, q=1, num_restarts=2, raw_samples=8)

    def wrong_shape(X: Any) -> Any:
        return X.sum(-1)

    with pytest.raises(RuntimeError, match="expected"):
        optimize_acqf(wrong_shape, bounds, q=2, num_restarts=2, raw_samples=8)
    with pytest.raises(UnsupportedError, match="set_X_pending"):
        optimize_acqf(
            lambda X: term(X), bounds, q=2, num_restarts=2, raw_samples=8, sequential=True
        )


@needs_torch
def test_generic_deterministic_model() -> None:
    import torch

    def f(X: Any) -> Any:
        return torch.stack([1.0 - ((X - 0.3) ** 2).sum(-1), torch.sin(3.0 * X).sum(-1)], -1)

    model = GenericDeterministicModel(f, num_outputs=2, dim=2)
    X = torch.tensor(_points(4, 3, 13), dtype=torch.float64)
    assert torch.equal(model(X), f(X))
    assert torch.equal(model.posterior(X).mean, f(X))
    ref, part = _front("torch")
    acqf = qLogExpectedHypervolumeImprovement(model, ref, part, sampler=StochasticSampler([1]))
    Xg = X[:, :1].clone().requires_grad_(True)
    v = acqf(Xg)
    (g,) = torch.autograd.grad(v.sum(), Xg)
    h = 1e-6
    for b in range(4):
        for k in range(2):
            xp, xm = X[:, :1].clone(), X[:, :1].clone()
            xp[b, 0, k] += h
            xm[b, 0, k] -= h
            fd = (acqf(xp)[b] - acqf(xm)[b]).item() / (2 * h)
            assert abs(fd - g[b, 0, k].item()) <= 1e-5 * max(1.0, abs(fd))
    sub = model.subset_output([1])
    assert sub.num_outputs == 1 and torch.equal(sub(X), f(X)[..., 1:])
    with pytest.raises(UnsupportedError, match="dim"):
        qLogExpectedHypervolumeImprovement(GenericDeterministicModel(f, num_outputs=2), ref, part)
    pm = PosteriorMeanModel(_model_list("torch"))
    # botorch_rs models are not differentiable with torch.autograd.
    not_differentiable = GenericDeterministicModel(lambda X: pm(X), num_outputs=2, dim=2)
    acqf = qLogExpectedHypervolumeImprovement(not_differentiable, ref, part)
    assert torch.isfinite(acqf(X[:, :1])).all()
    with pytest.raises(RuntimeError, match="does not require gradients"):
        acqf(X[:, :1].clone().requires_grad_(True))


_NESTED_SCRIPT = """
import torch

import botorch_rs
from botorch_rs.acquisition.multi_objective import qLogExpectedHypervolumeImprovement
from botorch_rs.models import GenericDeterministicModel, PosteriorMeanModel
from test_deterministic import _front, _model_list


class Boom(Exception):
    pass


pm = PosteriorMeanModel(_model_list("torch"))


def f(X):
    if bool((X[..., 0] > 0.8).any()):
        raise Boom()
    return pm(X) + 0.0 * X.sum(-1, keepdim=True)


ref, part = _front("torch")
acqf = qLogExpectedHypervolumeImprovement(GenericDeterministicModel(f, num_outputs=2, dim=2), ref, part)
g = torch.Generator().manual_seed(0)
botorch_rs.set_num_threads(4)
for i in range(60):
    X = torch.rand(48, 1, 2, generator=g, dtype=torch.float64).requires_grad_(True)
    try:
        acqf(X)
    except Boom:
        continue
    raise AssertionError(f"iteration {i}: the callable's exception was not re-raised")
"""


@needs_torch
def test_nested_computations_keep_callable_exceptions() -> None:
    # Callables on worker threads that start botorch_rs computations themselves
    # (a GenericDeterministicModel evaluating a botorch_rs model), while a
    # callable on another worker raises: the exception comes back unchanged,
    # not cleared by a nested computation, and nothing deadlocks (the GIL and
    # the pending-exception lock were once taken in opposite orders). In a
    # subprocess, so that a deadlock fails the test instead of hanging it.
    tests_dir = os.path.dirname(os.path.abspath(__file__))
    package_root = os.path.dirname(os.path.dirname(botorch_rs.__file__))
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        p for p in (tests_dir, package_root, env.get("PYTHONPATH")) if p
    )
    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(_NESTED_SCRIPT)],
        env=env,
        capture_output=True,
        text=True,
        timeout=240,
    )
    assert result.returncode == 0, result.stderr[-3000:]

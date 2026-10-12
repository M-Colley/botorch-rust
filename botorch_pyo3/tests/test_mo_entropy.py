"""Pathwise sampling and the information-theoretic multi-objective acquisition
functions through the public API, for every array type, independently of
BoTorch: result types and shapes, reproducibility from seeds and draws,
Matheron's rule (prior paths plus Gaussian updates), the posterior moments
of sample paths, Pareto optimality of sampled optima, gradients (finite
differences, torch.autograd), pending points, ``optimize_acqf``,
thread-count independence and errors. The comparisons with BoTorch are in
``test_parity_mo_entropy.py`` and ``test_equivalence_mo_entropy.py``.
"""

from __future__ import annotations

import functools
import math
import random
from typing import Any

import pytest

import botorch_rs
from botorch_rs.acquisition.multi_objective.joint_entropy_search import (
    qLowerBoundMultiObjectiveJointEntropySearch,
)
from botorch_rs.acquisition.multi_objective.max_value_entropy_search import (
    qLowerBoundMultiObjectiveMaxValueEntropySearch,
)
from botorch_rs.acquisition.multi_objective.predictive_entropy_search import (
    log_cdf_robust,
    qMultiObjectivePredictiveEntropySearch,
)
from botorch_rs.acquisition.multi_objective.utils import (
    compute_sample_box_decomposition,
    random_search_optimizer,
    sample_optimal_points,
)
from botorch_rs.exceptions import UnsupportedError
from botorch_rs.models import ModelListGP, SingleTaskGP
from botorch_rs.models.deterministic import MatheronPathModel
from botorch_rs.optim import optimize_acqf
from botorch_rs.sampling import SobolQMCNormalSampler
from botorch_rs.sampling.pathwise import (
    MatheronPath,
    PathList,
    draw_kernel_feature_paths,
    draw_matheron_paths,
    gaussian_update,
    get_matheron_path_model,
)
from botorch_rs.utils.multi_objective.box_decompositions import NondominatedPartitioning
from botorch_rs.utils.multi_objective.pareto import is_non_dominated
from helpers import HAS_TORCH, as_kind, assert_close, check_kind, flatten, shape

BOUNDS = [[0.0, 0.0], [1.0, 1.0]]


def _data(n: int, seed: int, m: int = 2) -> tuple[list, list]:
    """Conflicting smooth objectives on [0, 1]^2 plus noise."""
    rng = random.Random(seed)
    X = [[rng.random(), rng.random()] for _ in range(n)]
    Y = [
        [math.sin(3.0 * x[0] + t) + (0.5 - t) * x[1] + 0.02 * rng.gauss(0.0, 1.0) for t in range(m)]
        for x in X
    ]
    return X, Y


X_TRAIN, Y_TRAIN = _data(10, seed=5)


def _model(m: int = 2, **kwargs: Any) -> SingleTaskGP:
    X, Y = _data(10, seed=5, m=m)
    return SingleTaskGP(X, Y, **kwargs)


@pytest.fixture(scope="module")
def model() -> SingleTaskGP:
    return _model()


@pytest.fixture(scope="module")
def model_list() -> ModelListGP:
    X, Y = _data(10, seed=6)
    return ModelListGP(SingleTaskGP(X, [[y[0]] for y in Y]), SingleTaskGP(X, [[y[1]] for y in Y]))


def _optima(model: Any, seed: int = 1, maximize: bool = True) -> tuple[Any, Any]:
    return sample_optimal_points(
        model,
        BOUNDS,
        3,
        4,
        maximize=maximize,
        optimizer_kwargs={"pop_size": 256},
        num_features=128,
        seed=seed,
    )


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------


def test_paths_shapes_and_reproducibility(kind: str, model: SingleTaskGP) -> None:
    sampler = functools.partial(draw_kernel_feature_paths, num_features=32)
    paths = draw_matheron_paths(model, [2, 3], sampler, seed=3)
    assert isinstance(paths, MatheronPath)
    X = as_kind([[0.1, 0.2], [0.3, 0.9], [0.5, 0.5], [0.7, 0.1]], kind)
    values = paths(X)
    check_kind(values, kind)
    assert shape(values) == (2, 3, 2, 4)
    assert shape(paths(as_kind([[[0.1, 0.2]]] * 2, kind))) == (2, 3, 2, 1)
    assert shape(paths["prior_paths"](X)) == (2, 3, 2, 4)
    assert shape(paths["prior_paths"].weight) == (2, 3, 2, 32)
    assert shape(paths["prior_paths"].feature_map.weight) == (2, 16, 2)
    assert shape(paths["update_paths"].weight) == (2, 3, 2, 10)
    # The same seed gives the same paths, another seed other paths.
    again = draw_matheron_paths(model, [2, 3], sampler, seed=3)
    assert flatten(again(X)) == flatten(values)
    other = draw_matheron_paths(model, [2, 3], sampler, seed=4)
    assert flatten(other(X)) != flatten(values)
    # The draws reproduce the paths.
    redrawn = draw_matheron_paths(model, [2, 3], sampler, draws=paths.draws)
    assert flatten(redrawn(X)) == flatten(values)


def test_model_list_paths(model_list: ModelListGP) -> None:
    paths = draw_matheron_paths(model_list, [5], seed=1)
    assert isinstance(paths, PathList) and len(paths) == 2
    values = paths([[0.2, 0.3], [0.6, 0.1]])
    assert [shape(v) for v in values] == [(5, 2), (5, 2)]


def test_matheron_rule(kind: str) -> None:
    """Prior paths plus Gaussian updates with the noise of the same seed are
    the Matheron paths (without an outcome transform, all in one space)."""
    m = _model(outcome_transform=None)
    sampler = functools.partial(draw_kernel_feature_paths, num_features=64)
    paths = draw_matheron_paths(m, [4], sampler, seed=9)
    prior = draw_kernel_feature_paths(m, [4], num_features=64, seed=9)
    update = gaussian_update(m, prior(as_kind(X_TRAIN, kind)), seed=9)
    X = as_kind([[0.1, 0.2], [0.8, 0.4]], kind)
    assert_close(prior(X), paths["prior_paths"](X), rtol=1e-12, atol=1e-12)
    assert_close(update(X), paths["update_paths"](X), rtol=1e-12, atol=1e-12)
    joint = [a + b for a, b in zip(flatten(prior(X)), flatten(update(X)), strict=True)]
    assert_close(paths(X), joint, rtol=1e-12, atol=1e-12)


def test_paths_sample_the_posterior(model: SingleTaskGP) -> None:
    """The mean and variance of many paths are the posterior's."""
    S = 2048
    paths = draw_matheron_paths(
        model, [S], functools.partial(draw_kernel_feature_paths, num_features=2048), seed=2
    )
    X = [[0.15, 0.25], [0.55, 0.75], [0.9, 0.1]]
    values = paths(X)  # S x 2 x 3
    post = model.posterior(X)
    mean, var = post.mean, post.variance  # 3 x 2
    for t in range(2):
        for i in range(3):
            v = [values[s][t][i] for s in range(S)]
            mu = sum(v) / S
            sd = math.sqrt(sum((x - mu) ** 2 for x in v) / (S - 1))
            scale = math.sqrt(var[i][t])
            assert abs(mu - mean[i][t]) < 0.1 * scale + 1e-9, (t, i)
            assert abs(sd - scale) < 0.1 * scale + 1e-9, (t, i)


@pytest.mark.skipif(not HAS_TORCH, reason="requires torch")
def test_path_gradients_match_finite_differences(model: SingleTaskGP) -> None:
    import torch

    paths = draw_matheron_paths(model, [2], seed=4)
    X = torch.tensor([[0.3, 0.4], [0.6, 0.2]], dtype=torch.float64, requires_grad=True)
    (grad,) = torch.autograd.grad(paths(X).sum(), X)
    h = 1e-6
    for i in range(2):
        for j in range(2):
            E = torch.zeros_like(X)
            E[i, j] = h
            fd = (paths(X.detach() + E).sum() - paths(X.detach() - E).sum()) / (2 * h)
            assert abs(fd.item() - grad[i, j].item()) < 1e-6 * max(1.0, abs(fd.item()))


def test_matheron_path_model(kind: str, model: SingleTaskGP, model_list: ModelListGP) -> None:
    pm = get_matheron_path_model(model, seed=1)
    X = as_kind([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]], kind)
    mean = pm.posterior(X).mean
    check_kind(mean, kind)
    assert shape(mean) == (3, 2)
    assert flatten(pm.posterior(X).variance) == [0.0] * 6
    assert_close(pm(X), mean, rtol=0.0, atol=0.0)
    assert shape(pm.posterior(X, output_indices=[1]).mean) == (3, 1)
    # The path of the same seed (outputs as the last batch dimension).
    pv = flatten(draw_matheron_paths(model, [], seed=1)(X))
    assert flatten(pm(X)) == [pv[t * 3 + i] for i in range(3) for t in range(2)]
    many = MatheronPathModel(model, sample_shape=[4], seed=2)
    assert shape(many(X)) == (4, 3, 2)
    assert many.batch_shape == (4,)
    assert shape(get_matheron_path_model(model_list, seed=1)(X)) == (3, 2)


# ---------------------------------------------------------------------------
# Sampled optima and box decompositions
# ---------------------------------------------------------------------------


def test_sample_optimal_points(kind: str, model: SingleTaskGP) -> None:
    bounds = as_kind(BOUNDS, kind)
    ps, pf = sample_optimal_points(
        model, bounds, 3, 4, optimizer_kwargs={"pop_size": 256}, num_features=128, seed=1
    )
    check_kind(ps, kind)
    assert shape(ps) == (3, 4, 2) and shape(pf) == (3, 4, 2)
    for s in range(3):
        front = [[flatten(pf)[(s * 4 + i) * 2 + t] for t in range(2)] for i in range(4)]
        assert all(flatten(is_non_dominated(front)))
    assert all(0.0 <= v <= 1.0 for v in flatten(ps))
    again = sample_optimal_points(
        model, bounds, 3, 4, optimizer_kwargs={"pop_size": 256}, num_features=128, seed=1
    )
    assert flatten(again[0]) == flatten(ps)
    # Minimization: non-dominated for the negated objectives.
    _, pf_min = _optima(model, maximize=False)
    for s in range(3):
        neg = [[-pf_min[s][i][t] for t in range(2)] for i in range(4)]
        assert all(flatten(is_non_dominated(neg)))


def test_custom_optimizer_gets_path_models(model: SingleTaskGP) -> None:
    seen: list[Any] = []
    points = [[0.2, 0.3], [0.4, 0.5], [0.6, 0.7]]

    def optimizer(model: Any, bounds: Any, num_points: int, maximize: bool, **kw: Any) -> Any:
        assert isinstance(model, MatheronPathModel)
        assert kw == {"extra": 1}
        seen.append(model)
        return points, model.posterior(points).mean

    ps, pf = sample_optimal_points(
        model, BOUNDS, 2, 3, optimizer=optimizer, optimizer_kwargs={"extra": 1}, seed=3
    )
    assert len(seen) == 2 and shape(ps) == (2, 3, 2)
    for s in range(2):
        assert_close(pf[s], seen[s](points), rtol=0.0, atol=0.0)
    assert flatten(pf[0]) != flatten(pf[1])


def test_random_search_optimizer(model: SingleTaskGP) -> None:
    pm = MatheronPathModel(model, seed=4)
    x, y = random_search_optimizer(pm, BOUNDS, 3, True, pop_size=128, seed=1)
    assert shape(x) == (3, 2) and shape(y) == (3, 2)
    assert_close(y, pm(x), rtol=1e-12)

    class Wrapped:
        def posterior(self, X: Any) -> Any:
            return pm.posterior(X)

    x2, y2 = random_search_optimizer(Wrapped(), BOUNDS, 3, True, pop_size=128, seed=1)
    assert flatten(x2) == flatten(x) and flatten(y2) == flatten(y)
    with pytest.raises(RuntimeError, match="optimal points instead of 500"):
        random_search_optimizer(pm, BOUNDS, 500, True, pop_size=8, max_tries=2, seed=1)

    class Failing:
        def posterior(self, X: Any) -> Any:
            raise KeyError("boom")

    with pytest.raises(KeyError, match="boom"):
        random_search_optimizer(Failing(), BOUNDS, 3, True, pop_size=8, seed=1)


def test_compute_sample_box_decomposition(kind: str) -> None:
    fronts = as_kind([[[1.0, 0.0], [0.0, 1.0]], [[0.5, 0.5], [0.5, 0.5]]], kind)
    cells = compute_sample_box_decomposition(fronts)
    check_kind(cells, kind)
    assert shape(cells) == (2, 2, 2, 2)
    # The second (duplicated) front has one box and one empty padding box.
    c = flatten(cells)
    lower2, upper2 = c[8:12], c[12:16]
    assert lower2[:2] == [-1e10, -1e10] and upper2[:2] == [0.5, 0.5]
    assert lower2[2:] == upper2[2:]
    constrained = compute_sample_box_decomposition(fronts, num_constraints=1)
    assert shape(constrained) == (2, 2, 3, 3)
    single = compute_sample_box_decomposition(as_kind([[[1.0], [2.0]]], kind), maximize=False)
    assert flatten(single) == [1.0, 1e10]
    approx = compute_sample_box_decomposition(fronts, NondominatedPartitioning)
    assert shape(approx)[0] == 2


# ---------------------------------------------------------------------------
# Acquisition functions
# ---------------------------------------------------------------------------


def _acqfs(model: Any, est: str = "LB") -> list[Any]:
    ps, pf = _optima(model)
    cells = compute_sample_box_decomposition(pf)
    return [
        qLowerBoundMultiObjectiveMaxValueEntropySearch(model, cells, estimation_type=est),
        qLowerBoundMultiObjectiveJointEntropySearch(model, ps, pf, cells, estimation_type=est),
        qMultiObjectivePredictiveEntropySearch(model, ps),
    ]


SETS = [[[0.1, 0.9], [0.4, 0.2]], [[0.5, 0.5], [0.8, 0.7]]]


def test_acquisition_shapes_and_pending(kind: str, model: SingleTaskGP) -> None:
    for acqf in _acqfs(model):
        values = acqf(as_kind(SETS, kind))
        check_kind(values, kind)
        assert shape(values) == (2,)
        assert all(math.isfinite(v) for v in flatten(values))
        assert shape(acqf(as_kind(SETS[0], kind))) == (1,)
        pending = as_kind([[0.3, 0.3]], kind)
        acqf.set_X_pending(pending)
        assert flatten(acqf(as_kind(SETS, kind))) != flatten(values)
        acqf.set_X_pending(None)
        assert_close(acqf(as_kind(SETS, kind)), values, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("est", ["0", "LB", "LB2", "MC"])
def test_entropy_search_gradients(est: str, model_list: ModelListGP) -> None:
    for acqf in _acqfs(model_list, est)[: 2 if est != "LB" else 3]:
        X = [[[0.3, 0.6], [0.7, 0.2]]]
        value, grad = acqf.value_and_grad(X)
        h = 1e-6
        for i in range(2):
            for j in range(2):
                Xp = [[list(r) for r in X[0]]]
                Xm = [[list(r) for r in X[0]]]
                Xp[0][i][j] += h
                Xm[0][i][j] -= h
                fd = (acqf(Xp)[0] - acqf(Xm)[0]) / (2 * h)
                assert abs(fd - grad[0][i][j]) < 1e-5 * max(1.0, abs(fd)), (est, fd, grad)


def test_mc_sampler_and_attributes(model: SingleTaskGP) -> None:
    ps, pf = _optima(model)
    cells = compute_sample_box_decomposition(pf)
    sampler = SobolQMCNormalSampler(sample_shape=[32], seed=1)
    a = qLowerBoundMultiObjectiveJointEntropySearch(
        model, ps, pf, cells, estimation_type="MC", sampler=sampler
    )
    assert a.sampler is sampler and a.num_pareto_samples == 3 and a.estimation_type == "MC"
    b = qLowerBoundMultiObjectiveJointEntropySearch(
        model, ps, pf, cells, estimation_type="MC", sampler=SobolQMCNormalSampler([32], seed=1)
    )
    assert flatten(a(SETS)) == flatten(b(SETS))
    default = qLowerBoundMultiObjectiveJointEntropySearch(
        model, ps, pf, cells, estimation_type="MC"
    )
    assert default.sampler.num_samples == 64
    pes = qMultiObjectivePredictiveEntropySearch(model, ps)
    assert pes.ep_iterations >= 1
    assert shape(pes.ep_damping) == (3, 2)
    means, covs = pes.cached_factors
    assert shape(means) == (3, 2, 4, 4, 2) and shape(covs) == (3, 2, 4, 4, 2, 2)


def test_single_objective_mes_is_non_negative() -> None:
    m = _model(m=1)
    ps, pf = sample_optimal_points(m, BOUNDS, 4, 1, num_features=128, seed=2)
    cells = compute_sample_box_decomposition(pf)
    X = [[[i / 10, j / 10]] for i in range(11) for j in range(0, 11, 5)]
    # The lower bounds add a jitter of 1e-6 to the conditional variances (as
    # BoTorch does), which lowers the gain by up to 1e-6 / (2 Var[y]).
    var = flatten(m.posterior([x[0] for x in X], observation_noise=True).variance)
    for est in ("0", "LB", "LB2"):
        acqf = qLowerBoundMultiObjectiveMaxValueEntropySearch(m, cells, estimation_type=est)
        for v, vy in zip(flatten(acqf(X)), var, strict=True):
            assert v >= -1.1e-6 / (2.0 * vy) - 1e-12, (est, v, vy)


def test_optimize_acqf(model: SingleTaskGP) -> None:
    ps, pf = _optima(model)
    cells = compute_sample_box_decomposition(pf)
    acqf = qLowerBoundMultiObjectiveJointEntropySearch(model, ps, pf, cells)
    cand, value = optimize_acqf(acqf, BOUNDS, q=2, num_restarts=4, raw_samples=32)
    assert shape(cand) == (2, 2)
    assert all(0.0 <= v <= 1.0 for v in flatten(cand))
    assert_close(value, acqf([cand]), rtol=1e-9)


def test_results_do_not_depend_on_threads(model: SingleTaskGP) -> None:
    default = _optima(model, seed=8)
    botorch_rs.set_num_threads(1)
    try:
        single = _optima(model, seed=8)
    finally:
        botorch_rs.set_num_threads(None)
    assert flatten(single[0]) == flatten(default[0])
    assert flatten(single[1]) == flatten(default[1])


def test_log_cdf_robust(kind: str) -> None:
    x = as_kind([-30.0, -1.0, 0.0, 4.0, 6.0, 40.0], kind)
    values = log_cdf_robust(x)
    check_kind(values, kind)
    v = flatten(values)
    assert_close(v[1], math.log(0.5 * math.erfc(1.0 / math.sqrt(2.0))), rtol=1e-14)
    assert_close(v[2], math.log(0.5), rtol=1e-15)
    assert v[-1] < 0.0 and v[0] < -400.0


def test_errors(model: SingleTaskGP, model_list: ModelListGP) -> None:
    ps, pf = _optima(model)
    cells = compute_sample_box_decomposition(pf)
    with pytest.raises(NotImplementedError, match="estimation type"):
        qLowerBoundMultiObjectiveMaxValueEntropySearch(model, cells, estimation_type="XX")
    with pytest.raises(UnsupportedError):
        compute_sample_box_decomposition(pf, partitioning=dict)
    with pytest.raises(UnsupportedError):
        draw_matheron_paths(model, [2], prior_sampler=lambda **kw: None)
    with pytest.raises(UnsupportedError):
        draw_matheron_paths(model, [2], update_strategy=lambda **kw: None)
    with pytest.raises(UnsupportedError):
        draw_kernel_feature_paths(model, [2], weight_generator=None)
    with pytest.raises(UnsupportedError, match="even number"):
        draw_kernel_feature_paths(model, [2], num_features=33)
    with pytest.raises(UnsupportedError, match="num_points"):
        sample_optimal_points(_model(m=1), BOUNDS, 2, 3)
    with pytest.raises(TypeError):
        sample_optimal_points(model, BOUNDS, 2, 3, optimizer_kwargs={"bad": 1})
    with pytest.raises(ValueError):
        draw_matheron_paths(model, [2], seed=1)([0.1, 0.2])
    fixed = _model(train_Yvar=[[0.01, 0.01]] * 10)
    with pytest.raises(RuntimeError, match="noise"):
        qLowerBoundMultiObjectiveJointEntropySearch(fixed, ps, pf, cells)

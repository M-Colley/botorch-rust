"""API and property tests of the robust and multi-fidelity multi-objective
ports (no BoTorch needed; nested lists, NumPy arrays and torch tensors):

* ``InputPerturbation`` and ``ChainedInputTransform``: the layout of the
  perturbed points, bounds, ``indices``, multiplicative perturbations, the
  train/eval flags, models whose posterior is the plain posterior at the
  perturbed points, fitted ``Normalize`` steps, rejected configurations;
* the risk measures on hand-computable examples (``CVaR``, ``VaR``,
  ``WorstCase``, ``Expectation``, the multi-output ones and ``MVaR`` against a
  brute-force definition, with ties), ``MARS`` and their errors;
* acquisition functions with risk measures (values, gradients, errors);
* ``MOMF`` against ``qEHVI`` divided by the cost;
* ``optimize_with_nsgaii``: options, reproducibility, the native and the
  Python evaluation paths, and convergence on ZDT1 and DTLZ2 (the
  hypervolume of the returned front against the known optimum).
"""

from __future__ import annotations

import itertools
import math
import random
from typing import Any

import pytest

from botorch_rs.acquisition import PosteriorMean, qSimpleRegret
from botorch_rs.acquisition.logei import qLogNoisyExpectedImprovement
from botorch_rs.acquisition.multi_objective import (
    MOMF,
    qExpectedHypervolumeImprovement,
    qLogNoisyExpectedHypervolumeImprovement,
)
from botorch_rs.acquisition.multi_objective.multi_output_risk_measures import (
    MARS,
    IndependentCVaR,
    IndependentVaR,
    MultiOutputExpectation,
    MultiOutputWorstCase,
    MVaR,
)
from botorch_rs.acquisition.multioutput_acquisition import (
    MultiOutputAcquisitionFunction,
    MultiOutputAcquisitionFunctionWrapper,
    MultiOutputPosteriorMean,
)
from botorch_rs.acquisition.objective import LinearMCObjective, ScalarizedPosteriorTransform
from botorch_rs.acquisition.risk_measures import CVaR, Expectation, VaR, WorstCase
from botorch_rs.exceptions.errors import UnsupportedError
from botorch_rs.models import ModelListGP, SingleTaskGP
from botorch_rs.models.transforms import ChainedInputTransform, InputPerturbation, Normalize
from botorch_rs.models.transforms.outcome import Standardize
from botorch_rs.sampling import SobolQMCNormalSampler
from botorch_rs.test_functions.multi_objective import DTLZ2, ZDT1
from botorch_rs.utils.multi_objective.box_decompositions import NondominatedPartitioning
from botorch_rs.utils.multi_objective.hypervolume import Hypervolume
from botorch_rs.utils.multi_objective.optimize import optimize_with_nsgaii
from helpers import HAS_TORCH, as_kind, check_kind, flatten, shape, to_list

PSET = [[0.1, -0.05], [-0.02, 0.2], [0.0, 0.0]]


def data(n: int = 8, d: int = 2, m: int = 2, seed: int = 0) -> tuple[list, list]:
    rng = random.Random(seed)
    X = [[rng.random() for _ in range(d)] for _ in range(n)]
    Y = [
        [math.sin((3 + t) * x[0]) + (1 + t) * x[-1] ** 2 + 0.05 * rng.gauss(0, 1) for t in range(m)]
        for x in X
    ]
    return X, Y


def close(a: Any, b: Any, tol: float = 1e-10) -> None:
    fa, fb = flatten(a), flatten(b)
    assert len(fa) == len(fb), (shape(a), shape(b))
    for u, v in zip(fa, fb, strict=True):
        assert abs(u - v) <= tol * (1 + abs(v)), (u, v)


# ---------------------------------------------------------------------------
# InputPerturbation and ChainedInputTransform
# ---------------------------------------------------------------------------


def test_input_perturbation_layout(kind: str) -> None:
    X = as_kind([[[0.5, 0.5], [0.95, 0.0]]], kind)
    tf = InputPerturbation(as_kind(PSET, kind))
    out = tf.transform(X)
    check_kind(out, kind)
    assert shape(out) == (1, 6, 2)
    rows = to_list(out)[0]
    for i, x in enumerate(to_list(X)[0]):
        for w, p in enumerate(PSET):
            close(rows[i * 3 + w], [x[0] + p[0], x[1] + p[1]])
    # Bounds clamp the perturbed inputs; multiplicative perturbations.
    tf = InputPerturbation(as_kind(PSET, kind), bounds=as_kind([[0.0, 0.0], [1.0, 1.0]], kind))
    rows = to_list(tf.transform(X))[0]
    assert rows[3] == pytest.approx([1.0, 0.0]) and rows[4] == pytest.approx([0.93, 0.2])
    tf = InputPerturbation(as_kind([[2.0], [0.5]], kind), indices=[-1], multiplicative=True)
    rows = to_list(tf.transform(X))[0]
    close(rows, [[0.5, 1.0], [0.5, 0.25], [0.95, 0.0], [0.95, 0.0]])
    assert tf.n_w == 2 and tf.is_one_to_many


def test_input_perturbation_modes_and_errors(kind: str) -> None:
    X = as_kind([[0.2, 0.3]], kind)
    tf = InputPerturbation(as_kind(PSET, kind))
    assert shape(tf(X)) == (1, 2)  # train mode: not applied
    tf.eval()
    assert shape(tf(X)) == (3, 2)
    with pytest.raises(ValueError, match="n_p x d"):
        InputPerturbation(as_kind([0.1, 0.2], kind))
    with pytest.raises(ValueError, match="same number of columns"):
        InputPerturbation(as_kind(PSET, kind), bounds=as_kind([[0.0], [1.0]], kind))
    Xt, Yt = data()
    with pytest.raises(UnsupportedError, match="transform_on_train"):
        SingleTaskGP(
            as_kind(Xt, kind),
            as_kind(Yt, kind),
            input_transform=InputPerturbation(as_kind(PSET, kind), transform_on_train=True),
        )
    with pytest.raises(UnsupportedError, match="chains its own input transforms"):
        ChainedInputTransform(a=Standardize(m=1))  # type: ignore[arg-type]


def _same_hyperparameters(src: SingleTaskGP, dst: SingleTaskGP) -> None:
    dst.initialize(**dict(src.named_parameters()))


def test_model_posterior_is_plain_posterior_at_perturbed_points(kind: str) -> None:
    Xt, Yt = data(m=2)
    bounds = [[0.0, 0.0], [1.0, 1.0]]
    tf = InputPerturbation(as_kind(PSET, kind), bounds=as_kind(bounds, kind))
    robust = SingleTaskGP(as_kind(Xt, kind), as_kind(Yt, kind), input_transform=tf)
    plain = SingleTaskGP(as_kind(Xt, kind), as_kind(Yt, kind))
    _same_hyperparameters(plain, robust)
    Xq = as_kind([[0.3, 0.9], [0.99, 0.02]], kind)
    post = robust.posterior(Xq)
    assert shape(post.mean) == (6, 2)
    tf.eval()
    ref = plain.posterior(tf.transform(Xq))
    close(post.mean, ref.mean)
    close(post.variance, ref.variance)
    close(post.distribution.covariance_matrix, ref.distribution.covariance_matrix)
    # Model lists of perturbed models.
    lst = ModelListGP(robust, robust)
    assert shape(lst.posterior(Xq).mean) == (6, 4)


def test_conditioning_is_rejected_and_fantasies_skip_perturbations(kind: str) -> None:
    """BoTorch fantasizes in a "fantasize mode" that skips the perturbations
    (``transform_on_fantasize=False``); conditioning in eval mode would
    perturb the inputs, which then do not match the observations."""
    Xt, Yt = data(m=1)
    tf = InputPerturbation(as_kind(PSET, kind))
    model = SingleTaskGP(as_kind(Xt, kind), as_kind(Yt, kind), input_transform=tf)
    model.posterior(as_kind([[0.5, 0.5]], kind))  # eval mode
    X = as_kind([[0.2, 0.3]], kind)
    with pytest.raises(UnsupportedError, match="one-to-many"):
        model.condition_on_observations(X, as_kind([[0.1]], kind))
    fm = model.fantasize(X, SobolQMCNormalSampler(sample_shape=[4], seed=0))
    assert fm.batch_shape == (4,)
    # The fantasy points enter the training data unperturbed.
    tail = [b[-1] for b in to_list(fm.train_inputs[0])]
    close(tail, [[0.2, 0.3]] * 4)


def test_chained_transform_fits_normalize(kind: str) -> None:
    Xt, Yt = data(m=1)
    Xs = [[3 * a - 1, 2 * b] for a, b in Xt]
    norm = Normalize(d=2)
    chain = ChainedInputTransform(normalize=norm, perturb=InputPerturbation(as_kind(PSET, kind)))
    model = SingleTaskGP(as_kind(Xs, kind), as_kind(Yt, kind), input_transform=chain)
    lo = [min(x[j] for x in Xs) for j in range(2)]
    hi = [max(x[j] for x in Xs) for j in range(2)]
    close(chain["normalize"].bounds, [lo, hi])
    assert list(chain) == ["normalize", "perturb"] and len(chain) == 2
    assert chain.is_one_to_many
    assert shape(model.posterior(as_kind([[0.0, 1.0]], kind)).mean) == (3, 1)


@pytest.mark.skipif(not HAS_TORCH, reason="perturbation functions are callables on torch")
def test_heteroscedastic_perturbation_function() -> None:
    import torch

    def hetero(X: Any) -> Any:
        return torch.stack([0.05 * X**2, -0.08 * X], dim=-2)

    Xt, Yt = data(m=1)
    tf = InputPerturbation(hetero)
    assert tf.n_w is None
    model = SingleTaskGP(
        torch.tensor(Xt, dtype=torch.float64),
        torch.tensor(Yt, dtype=torch.float64),
        input_transform=tf,
    )
    assert tf.n_w == 2
    Xq = torch.tensor([[0.5, 0.25]], dtype=torch.float64)
    assert tuple(model.posterior(Xq).mean.shape) == (2, 1)
    tf.eval()
    expected = torch.cat([Xq + 0.05 * Xq**2, Xq - 0.08 * Xq])
    assert torch.allclose(tf.transform(Xq), expected)
    # Gradients flow through the perturbation function.
    acqf = qSimpleRegret(model, objective=Expectation(n_w=2))
    Xg = Xq.unsqueeze(0).clone().requires_grad_(True)
    acqf(Xg).sum().backward()
    assert Xg.grad is not None and torch.isfinite(Xg.grad).all()


# ---------------------------------------------------------------------------
# Risk measures
# ---------------------------------------------------------------------------


def test_single_output_risk_measures(kind: str) -> None:
    # Two candidates with n_w = 4 values each: (S=1) x 8 x 1.
    vals = [3.0, 1.0, 4.0, 2.0, -1.0, 0.5, 7.0, 0.0]
    Y = as_kind([[[v] for v in vals]], kind)
    out = CVaR(alpha=0.5, n_w=4)(Y)  # alpha_idx 1: mean of the 3 smallest
    check_kind(out, kind)
    close(out, [[2.0, -0.5 / 3]])
    # VaR: the largest of the n_w - alpha_idx smallest values.
    close(VaR(alpha=0.5, n_w=4)(Y), [[3.0, 0.5]])
    close(WorstCase(n_w=4)(Y), [[1.0, -1.0]])
    close(Expectation(n_w=4)(Y), [[2.5, 6.5 / 4]])
    # A linear preprocessing function scalarizes two outputs.
    Y2 = as_kind([[[v, 1.0] for v in vals]], kind)
    pre = LinearMCObjective(as_kind([1.0, -1.0], kind))
    close(WorstCase(n_w=4, preprocessing_function=pre)(Y2), [[0.0, -2.0]])
    with pytest.raises(RuntimeError, match="scalarized"):
        CVaR(alpha=0.5, n_w=4)(Y2)
    with pytest.raises(ValueError, match="alpha"):
        CVaR(alpha=0.0, n_w=4)


def test_multi_output_risk_measures(kind: str) -> None:
    rows = [[1.0, 4.0], [2.0, 3.0], [3.0, 2.0], [4.0, 1.0]]
    Y = as_kind([rows], kind)
    close(MultiOutputExpectation(n_w=4)(Y), [[[2.5, 2.5]]])
    close(MultiOutputWorstCase(n_w=4)(Y), [[[1.0, 1.0]]])
    close(IndependentCVaR(alpha=0.5, n_w=4)(Y), [[[2.0, 2.0]]])
    close(IndependentVaR(alpha=0.5, n_w=4)(Y), [[[3.0, 3.0]]])
    # MVaR at alpha = 0.5: the points dominated by at least two samples.
    mvar = MVaR(n_w=4, alpha=0.5)
    got = sorted(map(tuple, to_list(mvar(Y))[0]))
    assert got == [(1.0, 3.0), (2.0, 2.0), (3.0, 1.0)]
    close(MVaR(n_w=4, alpha=0.5, expectation=True)(Y), [[[2.0, 2.0]]])
    assert shape(MVaR(n_w=4, alpha=0.5, pad_to_n_w=True)(Y)) == (1, 4, 2)
    sets = mvar.get_mvar_set_vectorized(Y)
    assert len(sets) == 1 and shape(sets[0]) == (3, 2)
    assert sorted(map(tuple, to_list(mvar.get_mvar_set_via_counting(Y)[0]))) == got


def brute_force_mvar(Y: list[list[float]], alpha: float) -> set[tuple[float, ...]]:
    """The non-dominated points ``z`` of the grid of sample values with
    ``#{y >= z} >= ceil(alpha n_w)``."""
    n_w, m = len(Y), len(Y[0])
    k = math.ceil(alpha * n_w)
    grid = [sorted({y[t] for y in Y}) for t in range(m)]
    cands = [
        z
        for z in itertools.product(*grid)
        if sum(all(y[t] >= z[t] for t in range(m)) for y in Y) >= k
    ]
    return {
        z for z in cands if not any(all(w[t] >= z[t] for t in range(m)) and w != z for w in cands)
    }


@pytest.mark.parametrize("seed", range(6))
@pytest.mark.parametrize("alpha", [0.25, 0.5, 0.8, 1.0])
def test_mvar_matches_brute_force(seed: int, alpha: float) -> None:
    rng = random.Random(seed)
    n_w, m = 7, 2 + seed % 2
    # Rounded values give ties.
    Y = [[round(rng.gauss(0, 1), 1) for _ in range(m)] for _ in range(n_w)]
    expected = brute_force_mvar(Y, alpha)
    for counting in (False, True):
        mvar = MVaR(n_w=n_w, alpha=alpha, use_counting=counting)
        got = {tuple(p) for p in to_list(mvar([Y]))[0]}
        assert got == expected, (counting, got, expected)


def test_mars(kind: str) -> None:
    Ys = [
        [1.0, 4.0],
        [2.0, 3.0],
        [3.0, 2.0],
        [4.0, 1.0],
        [0.0, 5.0],
        [5.0, 0.5],
        [2.5, 2.5],
        [1.5, 1.5],
    ]
    mars = MARS(alpha=0.5, n_w=4, chebyshev_weights=as_kind([0.5, 0.5], kind))
    with pytest.raises(RuntimeError, match="baseline_Y"):
        mars(as_kind([Ys], kind))
    mars.set_baseline_Y(None, None, Y_samples=as_kind(Ys, kind))
    base = to_list(mars.baseline_Y)
    assert shape(base)[1] == 2
    bounds = to_list(MARS._get_Y_normalization_bounds(as_kind(base, kind)))
    close(mars.normalization_bounds, bounds)
    out = mars(as_kind([Ys], kind))
    assert shape(out) == (1, 2)
    # The VaR of the scalarized values, computed by hand.
    lo, hi = bounds

    def scal(y: list[float]) -> float:
        return min(0.5 * (y[t] - lo[t]) / (hi[t] - lo[t]) for t in range(2))

    for c in range(2):
        vals = sorted(scal(y) for y in Ys[4 * c : 4 * c + 4])
        # VaR (alpha_idx = 1): the third smallest of the four values.
        assert flatten(out)[c] == pytest.approx(vals[2], abs=1e-12)
    mars.chebyshev_weights = as_kind([0.9, 0.1], kind)
    close(mars.chebyshev_weights, [0.9, 0.1])
    with pytest.raises(UnsupportedError, match="Negative"):
        MARS(alpha=0.5, n_w=4, chebyshev_weights=[1.0, -1.0])


def test_acquisition_with_risk_measures(kind: str) -> None:
    Xt, Yt = data(m=1)
    model = SingleTaskGP(
        as_kind(Xt, kind), as_kind(Yt, kind), input_transform=InputPerturbation(as_kind(PSET, kind))
    )
    sampler = SobolQMCNormalSampler(sample_shape=[32], seed=1)
    acqf = qLogNoisyExpectedImprovement(
        model,
        as_kind(Xt, kind),
        sampler=sampler,
        objective=CVaR(alpha=0.6, n_w=3),
        prune_baseline=False,
    )
    X = as_kind([[[0.2, 0.4]], [[0.7, 0.1]]], kind)
    values, grads = acqf.value_and_grad(X)
    assert shape(values) == (2,) and shape(grads) == (2, 1, 2)
    assert all(math.isfinite(v) for v in flatten(values) + flatten(grads))
    # Without the risk measure the objective values do not match X.
    with pytest.raises(RuntimeError, match="q-batch shape"):
        qSimpleRegret(model)(X)
    # Outcome constraints (callables on torch tensors) are not supported
    # with one-to-many transforms.
    if not HAS_TORCH:
        return
    with pytest.raises(UnsupportedError, match="input perturbations"):
        qLogNoisyExpectedImprovement(
            model,
            as_kind(Xt, kind),
            objective=CVaR(alpha=0.6, n_w=3),
            constraints=[lambda Y: Y[..., 0]],
            prune_baseline=False,
        )


def test_mo_acquisition_with_mvar(kind: str) -> None:
    Xt, Yt = data(m=2)
    model = SingleTaskGP(
        as_kind(Xt, kind), as_kind(Yt, kind), input_transform=InputPerturbation(as_kind(PSET, kind))
    )
    acqf = qLogNoisyExpectedHypervolumeImprovement(
        model,
        as_kind([-2.0, -2.0], kind),
        as_kind(Xt, kind),
        objective=MVaR(n_w=3, alpha=0.6),
        prune_baseline=False,
    )
    values, grads = acqf.value_and_grad(as_kind([[[0.2, 0.4]], [[0.7, 0.1]]], kind))
    assert all(math.isfinite(v) for v in flatten(values) + flatten(grads))
    with pytest.raises(UnsupportedError, match="single-output"):
        qLogNoisyExpectedHypervolumeImprovement(
            model,
            [-2.0, -2.0],
            as_kind(Xt, kind),
            objective=MARS(alpha=0.5, n_w=3, chebyshev_weights=[0.5, 0.5]),
        )


# ---------------------------------------------------------------------------
# MOMF
# ---------------------------------------------------------------------------


def test_momf_is_cost_weighted_qehvi(kind: str) -> None:
    Xt, Yt = data(n=10, d=3, m=2, seed=3)
    Yt = [y + [x[-1]] for x, y in zip(Xt, Yt, strict=True)]  # the trust objective
    model = SingleTaskGP(as_kind(Xt, kind), as_kind(Yt, kind))
    ref = as_kind([-3.0, -3.0, 0.0], kind)
    part = NondominatedPartitioning(ref_point=ref, Y=as_kind(Yt, kind))
    momf = MOMF(model, ref, part, sampler=SobolQMCNormalSampler(sample_shape=[32], seed=2))
    with pytest.warns(Warning):
        ehvi = qExpectedHypervolumeImprovement(
            model, ref, part, sampler=SobolQMCNormalSampler(sample_shape=[32], seed=2)
        )
    X = [[[0.2, 0.4, 0.9]], [[0.7, 0.1, 0.3]], [[0.5, 0.5, 0.0]]]
    v_momf, v_ehvi = flatten(momf(as_kind(X, kind))), flatten(ehvi(as_kind(X, kind)))
    for x, a, b in zip(X, v_momf, v_ehvi, strict=True):
        cost = 1.0 + x[0][-1]
        assert a == pytest.approx(b / cost if b > 0 else b * cost, rel=1e-12, abs=1e-15)
    if HAS_TORCH:
        cost_momf = MOMF(model, ref, part, cost_call=lambda Z: 2.0 + Z[..., -1:] ** 2)
        assert all(math.isfinite(v) for v in flatten(cost_momf(as_kind(X, kind))))


# ---------------------------------------------------------------------------
# NSGA-II
# ---------------------------------------------------------------------------


def _gp(kind: str) -> SingleTaskGP:
    Xt, Yt = data(n=10, m=2, seed=4)
    return SingleTaskGP(as_kind(Xt, kind), as_kind(Yt, kind))


def test_nsgaii_options(kind: str) -> None:
    acqf = MultiOutputPosteriorMean(_gp(kind))
    bounds = as_kind([[0.0, 0.0], [1.0, 1.0]], kind)
    X, Y = optimize_with_nsgaii(acqf, bounds, 2, q=5, population_size=40, max_gen=10, seed=3)
    check_kind(X, kind)
    assert shape(X) == (5, 2) and shape(Y) == (5, 2)
    X2, Y2 = optimize_with_nsgaii(acqf, bounds, 2, q=5, population_size=40, max_gen=10, seed=3)
    assert to_list(X) == to_list(X2) and to_list(Y) == to_list(Y2)
    close(Y, acqf(as_kind([[x] for x in to_list(X)], kind)))
    X, _ = optimize_with_nsgaii(
        acqf,
        bounds,
        2,
        population_size=30,
        max_gen=5,
        seed=1,
        fixed_features={0: 0.25},
        discrete_choices={1: [0.0, 0.5, 1.0]},
    )
    assert all(x[0] == 0.25 and x[1] in (0.0, 0.5, 1.0) for x in to_list(X))
    X, _ = optimize_with_nsgaii(
        acqf,
        bounds,
        2,
        population_size=30,
        max_gen=10,
        seed=1,
        inequality_constraints=[([0, 1], [1.0, 1.0], 1.0)],
    )
    assert all(x[0] + x[1] >= 1.0 - 1e-12 for x in to_list(X))
    with pytest.raises(RuntimeError, match="feasible point"):
        optimize_with_nsgaii(
            acqf,
            bounds,
            2,
            population_size=10,
            max_gen=2,
            seed=0,
            ref_point=[1e9, 1e9],
        )


def test_nsgaii_wrapper_matches_posterior_mean(kind: str) -> None:
    model = _gp(kind)
    mean = MultiOutputPosteriorMean(model, weights=as_kind([1.0, -2.0], kind))
    wrapper = MultiOutputAcquisitionFunctionWrapper(
        [
            PosteriorMean(model, posterior_transform=ScalarizedPosteriorTransform(as_kind(w, kind)))
            for w in ([1.0, 0.0], [0.0, -2.0])
        ]
    )
    X = as_kind([[[0.2, 0.3]], [[0.9, 0.1]]], kind)
    assert shape(mean(X)) == (2, 2)
    close(wrapper(X), mean(X), 1e-12)


@pytest.mark.skipif(not HAS_TORCH, reason="objectives operate on torch tensors")
def test_nsgaii_python_path_matches_native() -> None:
    model = _gp("torch")
    acqf = MultiOutputPosteriorMean(model)
    import torch

    bounds = torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.float64)
    native = optimize_with_nsgaii(acqf, bounds, 2, population_size=30, max_gen=15, seed=7)
    python = optimize_with_nsgaii(
        acqf, bounds, 2, population_size=30, max_gen=15, seed=7, objective=lambda y: y
    )
    assert torch.equal(native[0], python[0]) and torch.equal(native[1], python[1])


class _Negated(MultiOutputAcquisitionFunction):
    """The negated objectives of a test problem (NSGA-II maximizes)."""

    def __init__(self, problem: Any) -> None:
        self.problem = problem
        self.model = None

    def forward(self, X: Any) -> Any:
        data, (n, _, d) = list(flatten(X)), shape(X)
        Xf = [data[i * d : (i + 1) * d] for i in range(n)]
        return [[-v for v in row] for row in to_list(self.problem.evaluate_true(Xf))]


def _das_dennis_sphere(h: int) -> list[list[float]]:
    """Uniformly spread points of the DTLZ2 front (Das and Dennis, 1998)."""
    pts = []
    for i in range(h + 1):
        for j in range(h + 1 - i):
            v = [i / h, j / h, (h - i - j) / h]
            norm = math.sqrt(sum(x * x for x in v))
            pts.append([x / norm for x in v])
    return pts


def test_nsgaii_converges_on_zdt1() -> None:
    problem = ZDT1(dim=6)
    _, Y = optimize_with_nsgaii(
        _Negated(problem), [[0.0] * 6, [1.0] * 6], 2, population_size=100, max_gen=250, seed=0
    )
    # The front f1 = 1 - sqrt(f0) (Y holds the negated objectives).
    assert max(abs(-y[1] - (1 - math.sqrt(-y[0]))) for y in to_list(Y)) < 0.02
    # HV of [0, 11]^2 minus the area above the front: 121 - 1/3.
    optimum = 121.0 - 1.0 / 3.0
    hv = Hypervolume(ref_point=[-11.0, -11.0]).compute(Y)
    assert 0.995 * optimum <= hv <= optimum * (1 + 1e-12), (hv, optimum)


def test_nsgaii_converges_on_dtlz2() -> None:
    problem = DTLZ2(dim=5, num_objectives=3)
    _, Y = optimize_with_nsgaii(
        _Negated(problem), [[0.0] * 5, [1.0] * 5], 3, population_size=100, max_gen=250, seed=0
    )
    # The front is the positive octant of the unit sphere.
    gaps = [abs(sum(v * v for v in y) - 1.0) for y in to_list(Y)]
    assert max(gaps) < 0.1 and sum(gaps) / len(gaps) < 0.02, (max(gaps), sum(gaps) / len(gaps))
    # NSGA-II's crowding distance spreads three-objective fronts unevenly
    # (Deb and Jain, 2014): compare with a uniform set of about the same size.
    hv = Hypervolume(ref_point=[-1.1] * 3).compute(Y)
    uniform = _das_dennis_sphere(13)  # 105 points
    hv_uniform = Hypervolume(ref_point=[1.1] * 3).compute([[-v for v in p] for p in uniform])
    optimum = 1.1**3 - math.pi / 6.0  # [0, 1.1]^3 minus the ball's octant
    assert 0.9 * hv_uniform <= hv <= optimum, (hv, hv_uniform, optimum)

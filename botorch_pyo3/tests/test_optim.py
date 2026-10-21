"""optimize_acqf: return shapes, options, fixed features, warnings."""

from __future__ import annotations

import warnings

import pytest

from botorch_rs.acquisition import (
    LogExpectedImprovement,
    PosteriorMean,
    qExpectedImprovement,
    qLogExpectedImprovement,
)
from botorch_rs.exceptions import (
    BadInitialCandidatesWarning,
    BotorchRsError,
    BotorchTensorDimensionError,
    NumericsWarning,
    UnsupportedError,
)
from botorch_rs.fit import fit_gpytorch_mll
from botorch_rs.mlls import ExactMarginalLogLikelihood
from botorch_rs.models import SingleTaskGP
from botorch_rs.optim import optimize_acqf
from botorch_rs.sampling import SobolQMCNormalSampler
from helpers import HAS_TORCH, as_kind, check_kind, flatten, make_data, shape

X_TRAIN, Y_TRAIN = make_data(n=12, d=3, seed=11)
BOUNDS = [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]


@pytest.fixture(scope="module")
def model() -> SingleTaskGP:
    m = SingleTaskGP(X_TRAIN, Y_TRAIN)
    fit_gpytorch_mll(ExactMarginalLogLikelihood(m.likelihood, m))
    return m


def qlogei(model: SingleTaskGP) -> qLogExpectedImprovement:
    sampler = SobolQMCNormalSampler(sample_shape=[128], seed=0)
    return qLogExpectedImprovement(model, best_f=max(y[0] for y in Y_TRAIN), sampler=sampler)


def test_joint_optimization(kind: str, model: SingleTaskGP) -> None:
    acqf = LogExpectedImprovement(model, best_f=max(y[0] for y in Y_TRAIN))
    bounds = as_kind(BOUNDS, kind)
    X, value = optimize_acqf(
        acqf, bounds=bounds, q=1, num_restarts=5, raw_samples=64, options={"seed": 0}
    )
    check_kind(X, kind)
    assert shape(X) == (1, 3) and shape(value) == ()
    assert all(0.0 <= v <= 1.0 for v in flatten(X))
    assert float(value) == pytest.approx(flatten(acqf(X))[0], rel=1e-12)
    # Reproducible with a seed.
    X2, value2 = optimize_acqf(
        acqf, bounds=bounds, q=1, num_restarts=5, raw_samples=64, options={"seed": 0}
    )
    assert flatten(X2) == flatten(X) and float(value2) == float(value)
    Xq, vq = optimize_acqf(
        qlogei(model), bounds=bounds, q=2, num_restarts=3, raw_samples=32, options={"seed": 1}
    )
    assert shape(Xq) == (2, 3) and shape(vq) == ()


def test_return_variants(kind: str, model: SingleTaskGP) -> None:
    acqf = qlogei(model)
    bounds = as_kind(BOUNDS, kind)
    X, values = optimize_acqf(
        acqf,
        bounds,
        q=2,
        num_restarts=4,
        raw_samples=32,
        return_best_only=False,
        options={"seed": 2},
    )
    check_kind(X, kind)
    assert shape(X) == (4, 2, 3) and shape(values) == (4,)
    best, best_value = optimize_acqf(
        acqf, bounds, q=2, num_restarts=4, raw_samples=32, options={"seed": 2}
    )
    assert float(best_value) == pytest.approx(max(flatten(values)), rel=1e-12)
    X, values = optimize_acqf(
        acqf, bounds, q=3, num_restarts=3, raw_samples=32, sequential=True, options={"seed": 3}
    )
    assert shape(X) == (3, 3) and shape(values) == (3,)
    assert acqf.X_pending is None  # restored after sequential optimization
    X, values = optimize_acqf(
        acqf, bounds, q=1, num_restarts=2, raw_samples=16, return_acq_values=False
    )
    assert values is None and shape(X) == (1, 3)
    with pytest.raises(UnsupportedError):
        optimize_acqf(
            acqf,
            bounds,
            q=2,
            num_restarts=2,
            raw_samples=8,
            sequential=True,
            return_best_only=False,
        )


def test_fixed_features_and_initial_conditions(model: SingleTaskGP) -> None:
    acqf = qlogei(model)
    X, _ = optimize_acqf(
        acqf, BOUNDS, q=2, num_restarts=3, raw_samples=32, fixed_features={0: 0.25, -1: 0.75}
    )
    assert all(p[0] == 0.25 and p[2] == 0.75 for p in X)
    ics = [[[0.2, 0.3, 0.4], [0.6, 0.5, 0.4]], [[0.9, 0.1, 0.5], [0.1, 0.9, 0.5]]]
    X, values = optimize_acqf(
        acqf, BOUNDS, q=2, num_restarts=2, batch_initial_conditions=ics, return_best_only=False
    )
    assert shape(X) == (2, 2, 3)
    start = flatten(acqf(ics))
    assert all(v >= s - 1e-12 for v, s in zip(flatten(values), start, strict=True))
    with pytest.raises(BotorchRsError, match="raw_samples"):
        optimize_acqf(acqf, BOUNDS, q=2, num_restarts=2)
    with pytest.raises(BotorchTensorDimensionError):
        optimize_acqf(acqf, [[0.0, 0.0], [1.0, 1.0]], q=1, num_restarts=2, raw_samples=8)
    with pytest.raises(BotorchTensorDimensionError):
        optimize_acqf(acqf, [0.0, 1.0], q=1, num_restarts=2, raw_samples=8)


def test_options(model: SingleTaskGP) -> None:
    acqf = PosteriorMean(model)
    options = {
        "seed": 5,
        "maxiter": 50,
        "batch_limit": 2,
        "init_batch_limit": 16,
        "eta": 2.0,
        "ftol": 1e-12,
        "gtol": 1e-8,
        "maxcor": 5,
        "maxls": 10,
        "maxfun": 100,
        "method": "L-BFGS-B",
        "disp": False,
    }
    X, value = optimize_acqf(acqf, BOUNDS, q=1, num_restarts=4, raw_samples=32, options=options)
    assert shape(X) == (1, 3) and isinstance(value, float)
    X2, value2 = optimize_acqf(
        acqf, BOUNDS, q=1, num_restarts=4, raw_samples=32, options={**options, "method": "SLSQP"}
    )
    assert shape(X2) == (1, 3) and abs(float(value2) - float(value)) < 1e-4
    if HAS_TORCH:
        for extra in ({"method": "trust-constr"}, {"with_grad": False}):
            initial = [[[0.2, 0.3, 0.4]], [[0.8, 0.7, 0.6]]]
            points, scores = optimize_acqf(
                acqf,
                BOUNDS,
                q=1,
                num_restarts=2,
                batch_initial_conditions=initial,
                return_best_only=False,
                options={**extra, "maxiter": 100},
            )
            assert shape(points) == (2, 1, 3)
            assert all(0 <= x <= 1 for x in flatten(points))
            assert all(
                v >= s - 1e-7 for v, s in zip(flatten(scores), flatten(acqf(initial)), strict=True)
            )


def test_unsupported_arguments(model: SingleTaskGP) -> None:
    acqf = PosteriorMean(model)
    with pytest.raises(UnsupportedError):
        optimize_acqf(acqf, BOUNDS, q=1, num_restarts=2, raw_samples=8, unknown_ic_kwarg=1)
    with pytest.raises(ValueError, match="requires sequential"):
        optimize_acqf(
            acqf, BOUNDS, q=1, num_restarts=2, raw_samples=8, acq_function_sequence=[acqf]
        )
    with pytest.raises(UnsupportedError):
        optimize_acqf(object(), BOUNDS, q=1, num_restarts=2, raw_samples=8)  # type: ignore[arg-type]
    # Callables are evaluated on torch tensors (see test_deterministic.py); their
    # values must have the batch shape.
    with pytest.raises(UnsupportedError if not HAS_TORCH else RuntimeError):
        optimize_acqf(lambda X: X, BOUNDS, q=1, num_restarts=2, raw_samples=8)
    # As in BoTorch, return_full_tree only applies to one-shot acquisition
    # functions (see test_mo_hvkg.py).
    X, _ = optimize_acqf(acqf, BOUNDS, q=1, num_restarts=2, raw_samples=8, return_full_tree=True)
    assert shape(X) == (1, 3)
    with pytest.raises(BotorchRsError):  # analytic acquisition functions need q = 1
        optimize_acqf(acqf, BOUNDS, q=2, num_restarts=2, raw_samples=8)


def test_bad_initial_candidates_warning(model: SingleTaskGP) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", NumericsWarning)
        acqf = qExpectedImprovement(
            model, best_f=1e6, sampler=SobolQMCNormalSampler(sample_shape=[16], seed=0)
        )
    with pytest.warns(BadInitialCandidatesWarning):
        optimize_acqf(acqf, BOUNDS, q=1, num_restarts=2, raw_samples=16, options={"seed": 0})

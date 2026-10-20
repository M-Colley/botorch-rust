"""Optimization test budgets exercise real native and SciPy optimizers."""

from __future__ import annotations

import importlib

import pytest

from botorch_rs import _botorch_rs as rs
from botorch_rs.acquisition.analytic import PosteriorMean
from botorch_rs.models import SingleTaskGP
from botorch_rs.optim.initializers import gen_batch_initial_conditions
from botorch_rs.optim.optimize import optimize_acqf
from botorch_rs.test_utils.mock import mock_optimize, mock_optimize_context_manager


def _acquisition():
    model = SingleTaskGP([[0.0], [0.3], [0.8], [1.0]], [[0.0], [0.7], [0.3], [-0.2]])
    return PosteriorMean(model)


def test_mock_context_limits_real_native_optimization_and_restores() -> None:
    acquisition = _acquisition()
    original = rs.optimize_acqf
    kwargs = dict(
        acq_function=acquisition,
        bounds=[[0.0], [1.0]],
        q=1,
        num_restarts=12,
        raw_samples=128,
        options={"seed": 51, "maxiter": 100},
        return_best_only=False,
    )
    with mock_optimize_context_manager():
        candidates, values = optimize_acqf(**kwargs)
    assert len(candidates) == len(values) == 2
    assert rs.optimize_acqf is original
    expected, expected_values = optimize_acqf(
        **{**kwargs, "num_restarts": 2, "raw_samples": 4, "options": {"seed": 51, "maxiter": 2}}
    )
    assert candidates == expected and values == expected_values
    assert kwargs["options"]["maxiter"] == 100


def test_mock_decorator_initialization_and_nested_context() -> None:
    acquisition = _acquisition()

    @mock_optimize
    def generate(seed):
        """Small test initialization."""
        return gen_batch_initial_conditions(
            acquisition, [[0.0], [1.0]], q=1, num_restarts=8, raw_samples=64, options={"seed": seed}
        )

    assert generate.__name__ == "generate"
    assert generate.__doc__ == "Small test initialization."
    assert len(generate(seed=17)) == 2
    with mock_optimize_context_manager(), mock_optimize_context_manager():
        assert len(generate(seed=17)) == 2


def test_mock_unused_force_and_exception_cleanup() -> None:
    original = rs.optimize_acqf
    mixed = importlib.import_module("botorch_rs.optim.optimize_mixed")
    original_iterations = mixed.MAX_ITER_ALTER
    with pytest.raises(AssertionError, match="No mocks were called"):
        with mock_optimize_context_manager():
            assert mixed.MAX_ITER_ALTER == 1
    assert mixed.MAX_ITER_ALTER == original_iterations and rs.optimize_acqf is original
    with mock_optimize_context_manager(force=True):
        pass
    with pytest.raises(RuntimeError, match="body failed"):
        with mock_optimize_context_manager():
            raise RuntimeError("body failed")
    assert rs.optimize_acqf is original


def test_mock_limits_native_fit() -> None:
    from botorch_rs.mlls import ExactMarginalLogLikelihood
    from botorch_rs.optim.fit import fit_gpytorch_mll_scipy

    acquisition = _acquisition()
    model = acquisition.model
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    with mock_optimize_context_manager():
        result = fit_gpytorch_mll_scipy(mll, options={"maxiter": 100})
    assert 0 < result.step <= 2


def test_mock_reaches_preimported_generic_optimizer() -> None:
    torch = pytest.importorskip("torch")
    pytest.importorskip("scipy")
    from botorch_rs.optim.core import scipy_minimize

    parameter = torch.tensor([3.0], dtype=torch.double, requires_grad=True)

    def closure():
        value = parameter.pow(4).sum()
        return value, (torch.autograd.grad(value, parameter)[0],)

    options = {"maxiter": 100}
    with mock_optimize_context_manager():
        result = scipy_minimize(closure, {"x": parameter}, options=options)
    assert result.step == 2 and options == {"maxiter": 100}


def test_mock_mixed_optimizer_defaults_are_used() -> None:
    mixed = importlib.import_module("botorch_rs.optim.optimize_mixed")
    with mock_optimize_context_manager():
        candidate, value = mixed.optimize_acqf_mixed_alternating(
            _acquisition(),
            [[0.0], [1.0]],
            discrete_dims={0: [0.0, 1.0]},
            num_restarts=2,
            raw_samples=4,
            seed=17,
        )
    assert len(candidate) == 1 and candidate[0][0] in (0.0, 1.0)
    assert value is not None

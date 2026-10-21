# ruff: noqa: E402
"""Batched scheduling, resumable state, curvature, and official solver parity."""

from __future__ import annotations

import pytest

pytest.importorskip("numpy")


import numpy as np

from botorch_rs.optim.batched_lbfgs_b import OptimState, fmin_l_bfgs_b_batched
from helpers import importorskip_botorch


def quadratic(x, batch_indices=None):
    return (x * x).sum(axis=-1), 2 * x


def rosenbrock(x, batch_indices=None):
    a = x[:, 1] - x[:, 0] ** 2
    return 100 * a**2 + (1 - x[:, 0]) ** 2, np.stack(
        (-400 * x[:, 0] * a + 2 * (x[:, 0] - 1), 200 * a), axis=-1
    )


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("bounds", [None, [(-2, 2), (None, 3)], [(0.4, 0.4), (0.2, None)]])
def test_quadratic_official_parity(dtype, bounds):
    importorskip_botorch()
    pytest.importorskip("botorch")
    from botorch.optim.batched_lbfgs_b import fmin_l_bfgs_b_batched as official

    x = np.asarray([[1.0, 2.0], [-1.0, 0.0], [0.0, 0.0]], dtype=dtype)
    preserved = x.copy()
    actual = fmin_l_bfgs_b_batched(quadratic, x, bounds=bounds)
    expected = official(quadratic, x, bounds=bounds)
    np.testing.assert_array_equal(x, preserved)
    np.testing.assert_allclose(actual[0], expected[0], atol=1e-12)
    np.testing.assert_allclose(actual[1], expected[1], atol=1e-12)
    for a, e in zip(actual[2], expected[2], strict=True):
        assert a.success == e.success
        assert a.status == e.status
        assert a.nit == e.nit
        assert a.nfev == e.nfev
        np.testing.assert_allclose(a.jac, e.jac, atol=1e-12)
        np.testing.assert_allclose(a.hess_inv.todense(), e.hess_inv.todense(), atol=1e-10)


@pytest.mark.parametrize("bounds", [None, [(-2, 2), (-1, 3)], [(0, 0.8), (0, 0.7)]])
@pytest.mark.parametrize(
    "options", [{}, {"maxiter": 3}, {"maxls": 1}, {"ftol": 0.0, "factr": None, "tol": 1e-8}]
)
def test_rosenbrock_official_values_and_termination(bounds, options):
    importorskip_botorch()
    pytest.importorskip("botorch")
    from botorch.optim.batched_lbfgs_b import fmin_l_bfgs_b_batched as official

    x = np.array([[-1.2, 1.0], [2.0, 2.0]])
    actual = fmin_l_bfgs_b_batched(rosenbrock, x, bounds=bounds, **options)
    expected = official(rosenbrock, x, bounds=bounds, **options)
    np.testing.assert_allclose(actual[0], expected[0], rtol=2e-6, atol=2e-6)
    if options.get("maxls") == 1:
        # SciPy's C wrapper restores x/g on failure but discards the restored
        # scalar f. BoTorch inherits that stale-value result. Check f(x).
        np.testing.assert_allclose(actual[1], rosenbrock(actual[0])[0], atol=1e-12)
    else:
        np.testing.assert_allclose(actual[1], expected[1], rtol=2e-6, atol=2e-10)
    for a, e in zip(actual[2], expected[2], strict=True):
        assert a.success == e.success
        assert a.status == e.status
        np.testing.assert_allclose(a.jac, e.jac, rtol=1e-4, atol=1e-4)


def test_active_rows_preserve_original_indices_and_float32_gradients():
    pytest.importorskip("scipy")
    seen = []
    centers = np.array([[0.0, 0.0], [1.0, -2.0], [-3.0, 1.0]])
    scales = np.array([[1.0, 1.0], [2.0, 9.0], [0.3, 5.0]])

    def function(x, batch_indices):
        seen.append(list(batch_indices))
        delta = x - centers[batch_indices]
        return (scales[batch_indices] * delta**2).sum(-1), (
            2 * scales[batch_indices] * delta
        ).astype(np.float32)

    x, f, results = fmin_l_bfgs_b_batched(
        function, np.zeros((3, 2)), pass_batch_indices=True, pgtol=1e-7
    )
    assert seen[0] == [0, 1, 2]
    assert all(0 not in indices for indices in seen[1:])
    assert any(len(indices) < 2 for indices in seen[1:])
    np.testing.assert_allclose(x, centers, atol=2e-6)
    np.testing.assert_allclose(f, 0, atol=1e-11)
    assert all(result.jac.dtype == np.float64 for result in results)


@pytest.mark.parametrize("intermediate", [False, True])
def test_callback_stops_each_independent_state(intermediate):
    pytest.importorskip("scipy")
    points = []

    def simple_callback(x):
        points.append(x.copy())
        raise StopIteration

    def result_callback(intermediate_result):
        points.append(intermediate_result.x.copy())
        assert intermediate_result.fun.shape == ()
        raise StopIteration

    x, f, results = fmin_l_bfgs_b_batched(
        quadratic, [[10.0], [20.0]], callback=result_callback if intermediate else simple_callback
    )
    assert len(points) == 2
    assert all(result.nit == 1 and result.status == 2 and not result.success for result in results)
    assert (x > 0).all() and (f > 0).all()


def test_inverse_hessian_native_operator_matches_scipy_curvature_operator():
    pytest.importorskip("scipy")
    from scipy.optimize._lbfgsb_py import LbfgsInvHessProduct

    _, _, results = fmin_l_bfgs_b_batched(rosenbrock, [[-1.2, 1.0]], maxiter=7, maxcor=3)
    op = results[0].hess_inv
    expected = LbfgsInvHessProduct(op.sk, op.yk)
    matrix = np.array([[1.0, -2.0, 0.2], [0.3, 0.8, 1.2]])
    np.testing.assert_allclose(op @ matrix, expected @ matrix, rtol=2e-14, atol=2e-14)
    np.testing.assert_allclose(op.todense(), expected.todense(), rtol=2e-14, atol=2e-14)
    np.testing.assert_allclose(op.rmatvec(matrix[:, 0]), op.T @ matrix[:, 0], atol=1e-15)
    assert op.matmat(np.empty((2, 0))).shape == (2, 0)
    with pytest.raises(ValueError):
        op.matvec([1.0])


def test_suspended_state_can_fork_line_search_without_aliasing():
    state = OptimState(np.array([[-2.0, -1.0], [2.0, 3.0]]), 20, [-1.2, 1.0], 2, 4)
    while state.n_iterations < 4 or state.task[0] != 3:
        if state.task[0] == 3:
            f, g = rosenbrock(state.x[None, :])
            state.provide_evaluation(f[0], g[0])
        else:
            state.resume()
    fork = state.copy()
    preserved = fork.x.copy()
    f, g = rosenbrock(state.x[None, :])
    state.provide_evaluation(f[0], g[0])
    np.testing.assert_array_equal(fork.x, preserved)

    def finish(candidate):
        while candidate.task[0] in (1, 3):
            if candidate.task[0] == 3:
                f, g = rosenbrock(candidate.x[None, :])
                candidate.provide_evaluation(f[0], g[0])
            else:
                candidate.resume()
        return candidate.x, candidate.f, candidate.n_iterations, candidate.fun_calls

    a, b = finish(state), finish(fork)
    for av, bv in zip(a, b, strict=True):
        np.testing.assert_array_equal(av, bv)
    np.testing.assert_array_equal(state.nbd, [2, 2])
    assert state.task[0] == 4


@pytest.mark.parametrize(
    "options",
    [
        {"maxls": 0},
        {"maxcor": 0},
        {"pgtol": -1},
        {"factr": -1},
        {"bounds": [(1, 0)]},
        {"bounds": [(0, 1), (0, 1)]},
        {"ftol": 1e-5},
    ],
)
def test_invalid_options(options):
    pytest.importorskip("scipy")
    with pytest.raises((ValueError, AssertionError)):
        fmin_l_bfgs_b_batched(quadratic, [[1.0]], **options)


@pytest.mark.parametrize(
    "func",
    [
        lambda x: (np.zeros(len(x)), np.zeros((len(x), 3))),
        lambda x: (np.zeros((len(x), 1)), np.zeros_like(x)),
        lambda x: (np.full(len(x), np.nan), np.zeros_like(x)),
    ],
)
def test_invalid_objective_results_are_rejected(func):
    pytest.importorskip("scipy")
    with pytest.raises(ValueError):
        fmin_l_bfgs_b_batched(func, [[1.0, 2.0]])


def test_objective_exception_propagates():
    pytest.importorskip("scipy")

    def fail(x):
        raise RuntimeError("objective failed")

    with pytest.raises(RuntimeError, match="objective failed"):
        fmin_l_bfgs_b_batched(fail, [[1.0]])

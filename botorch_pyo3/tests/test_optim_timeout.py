"""SciPy timeout parity, scalar objectives, callbacks and cleanup."""

from __future__ import annotations

import warnings

import pytest

from botorch_rs.optim.utils.common import check_scipy_version_at_least
from botorch_rs.optim.utils.timeout import minimize_with_timeout

np = pytest.importorskip("numpy")
scipy = pytest.importorskip("scipy")


def objective(x, target=0.0):
    return ((x - target) ** 2).sum(), 2 * (x - target)


def test_version_comparison() -> None:
    major, minor = map(int, scipy.__version__.split(".")[:2])
    assert check_scipy_version_at_least(minor, major)
    assert not check_scipy_version_at_least(minor + 1, major)
    assert check_scipy_version_at_least(999, major - 1)


@pytest.mark.parametrize("timeout", [None, 1.0])
def test_nonbinding_timeout_parity(timeout) -> None:
    bt = pytest.importorskip("botorch.optim.utils.timeout")
    kwargs = dict(
        fun=objective,
        x0=np.array([1.0, -1.0]),
        args=(0.2,),
        jac=True,
        method="L-BFGS-B",
        bounds=[(-2, 2)] * 2,
    )
    original_filters = warnings.filters[:]
    actual = minimize_with_timeout(**kwargs, timeout_sec=timeout)
    assert warnings.filters == original_filters
    expected = bt.minimize_with_timeout(**kwargs, timeout_sec=timeout)
    assert actual.success and expected.success
    np.testing.assert_array_equal(actual.x, expected.x)
    assert actual.nit == expected.nit


@pytest.mark.parametrize("with_gradient", [False, True])
def test_binding_timeout_supports_scalar_objective(with_gradient: bool) -> None:
    result = minimize_with_timeout(
        objective if with_gradient else lambda x: objective(x)[0],
        np.array([1.0, -1.0]),
        jac=with_gradient,
        method="L-BFGS-B",
        timeout_sec=0.0,
    )
    assert not result.success and result.status == 1 and result.nit == 1
    assert "Optimization timed out" in result.message
    np.testing.assert_allclose(result.fun, objective(result.x)[0])


def test_callbacks_and_custom_method() -> None:
    calls = []
    result = minimize_with_timeout(
        objective,
        np.array([1.0]),
        jac=True,
        method="L-BFGS-B",
        callback=lambda x: calls.append(x.copy()),
        timeout_sec=1.0,
    )
    assert result.success and calls
    calls.clear()
    result = minimize_with_timeout(
        objective,
        np.array([1.0]),
        jac=True,
        method="trust-constr",
        callback=lambda x, state: calls.append(x.copy()),
        timeout_sec=0.0,
    )
    assert not result.success and calls
    with pytest.raises(NotImplementedError, match="Custom callable"):
        minimize_with_timeout(
            objective,
            np.array([1.0]),
            jac=True,
            method=lambda *args, **kwargs: None,
            callback=lambda x: None,
            timeout_sec=0.0,
        )

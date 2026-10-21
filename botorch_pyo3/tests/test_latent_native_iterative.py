"""Native scoped solvers: array protocols, snapshots and independent source laws."""

from __future__ import annotations

import copy
import math
from unittest.mock import patch

import pytest
from test_latent_kronecker import data

from botorch_rs.exceptions.warnings import NumericsWarning
from botorch_rs.mlls import ExactMarginalLogLikelihood
from botorch_rs.models import LatentKroneckerGP
from botorch_rs.models._structured_iterative import current
from helpers import KINDS, as_kind, assert_close, to_list


def exact_trace(name, default):
    return {
        "max_cholesky_size": 0,
        "num_trace_samples": 0,
        "max_lanczos_quadrature_iterations": 40,
    }.get(name, default)


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("missing", [False, True])
@pytest.mark.parametrize("log_prob", [False, True])
def test_native_iterative_array_mll_moments_samples_and_snapshots(kind, missing, log_prob):
    model = LatentKroneckerGP(*(as_kind(v, kind) for v in data(missing)), backend="native")
    query = as_kind([[0.4, 0.6], [0.7, 0.2]], kind)
    times = as_kind([[0.1], [0.8]], kind)
    expected = model.posterior(query, times)
    expected_value, expected_grad = ExactMarginalLogLikelihood(
        model.likelihood, model
    ).value_and_grad()
    width = math.prod(expected.base_sample_shape)
    base = as_kind(
        [[math.sin((i * 3 + j * 7) * 0.17) for j in range(width)] for i in range(3)], kind
    )
    expected_samples = expected.rsample_from_base_samples((3,), base)
    original = model._rs
    with patch("botorch_rs.models._structured_iterative.setting", side_effect=exact_trace):
        with model.use_iterative_methods(tol=1e-11, max_iter=100, log_prob=log_prob):
            assert model._rs is not original
            actual = model.posterior(query, times)
            value, gradient = ExactMarginalLogLikelihood(model.likelihood, model).value_and_grad()
            assert_close(value, expected_value, atol=1e-9, rtol=1e-9)
            assert_close(gradient, expected_grad, atol=1e-8, rtol=1e-8)
            assert_close(actual.mean, expected.mean, atol=1e-9, rtol=1e-9)
            assert_close(
                actual.distribution.covariance_matrix,
                expected.distribution.covariance_matrix,
                atol=1e-9,
                rtol=1e-9,
            )
            assert_close(
                actual.rsample_from_base_samples((3,), base), expected_samples, atol=1e-9, rtol=1e-9
            )
            snapshot = copy.deepcopy(model)
    assert model._rs is original
    assert current() is None
    model.likelihood.noise = 0.8
    # Both posterior and deep-copied model retain the settings and parameters.
    assert_close(actual.mean, expected.mean, atol=1e-9, rtol=1e-9)
    assert_close(snapshot.posterior(query, times).mean, expected.mean, atol=1e-9, rtol=1e-9)


def test_nested_native_scopes_restore_on_exception_and_keep_parameter_updates():
    model = LatentKroneckerGP(*data(True), backend="native")
    initial = model._rs
    with patch("botorch_rs.models._structured_iterative.setting", side_effect=exact_trace):
        with model.use_iterative_methods(tol=1e-9):
            outer = model._rs
            with pytest.raises(RuntimeError, match="abort"):
                with model.use_iterative_methods(tol=1e-5, solves=False):
                    assert model._rs is not outer
                    model.likelihood.noise = 0.3
                    raise RuntimeError("abort")
            assert model._rs is outer
            assert to_list(model.likelihood.noise)[0] == pytest.approx(0.3)
            assert current().tol == 1e-9
    assert model._rs is initial
    assert current() is None
    assert to_list(model.likelihood.noise)[0] == pytest.approx(0.3)


@pytest.mark.parametrize("tol", [0.0, -1.0, math.nan, math.inf])
def test_invalid_native_tolerances_preserve_original_model(tol):
    model = LatentKroneckerGP(*data(), backend="native")
    original = model._rs
    with pytest.raises(ValueError, match="positive"):
        with model.use_iterative_methods(tol=tol):
            pytest.fail("Invalid settings entered the scope")
    assert model._rs is original
    assert current() is None


def test_native_iteration_limit_warns_and_snapshot_retains_actual_solver():
    model = LatentKroneckerGP(*data(True), backend="native", outcome_transform=None)
    query = [[0.4, 0.6], [0.7, 0.2]]
    exact = model.posterior(query).mean
    with patch("botorch_rs.models._structured_iterative.setting", side_effect=exact_trace):
        with model.use_iterative_methods(tol=1e-13, max_iter=1):
            retained = model.posterior(query)
    with pytest.warns(NumericsWarning, match="iteration limit"):
        approximate = retained.mean
    assert abs(to_list(approximate)[0][0] - to_list(exact)[0][0]) > 1e-4
    assert_close(model.posterior(query).mean, exact)


@pytest.mark.parametrize("missing", [False, True])
def test_unmodified_source_forced_cg_law_mll_and_feature_time_query_gradients(missing):
    torch = pytest.importorskip("torch")
    gpytorch = pytest.importorskip("gpytorch")
    pytest.importorskip("botorch")
    from botorch.models.latent_kronecker_gp import LatentKroneckerGP as Reference
    from gpytorch.mlls import ExactMarginalLogLikelihood as ReferenceMLL

    X = torch.tensor([[0.1], [0.45], [0.9]], dtype=torch.double)
    T = X.new_tensor([[0.0], [0.6]])
    Y = (4 * X + T.squeeze(-1)).sin()
    if missing:
        Y[1, 0] = float("nan")
    model = LatentKroneckerGP(X, T, Y, outcome_transform=None, backend="native")
    # Shorter training correlations keep this source CG control accurate at
    # its requested tolerance; its division guard limits tighter tolerances.
    model.covar_module_X.lengthscale = 0.06
    model.covar_module_T.base_kernel.lengthscale = 0.09
    model.likelihood.noise = 0.12
    reference = Reference(X, T, Y, outcome_transform=None).to(X)
    names = reference.state_dict()
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            names[name].copy_(parameter)
    query = X.new_tensor([[0.2], [0.7]]).requires_grad_()
    times = T.new_tensor([[0.1], [0.8]]).requires_grad_()
    reference.train()
    with (
        gpytorch.settings.max_cholesky_size(0),
        reference.use_iterative_methods(tol=1e-7, max_iter=100, log_prob=False),
    ):
        expected_mll = ReferenceMLL(reference.likelihood, reference)(
            reference(X, T), reference.train_targets
        )
    reference.eval()
    with (
        gpytorch.settings.max_cholesky_size(0),
        gpytorch.settings.detach_test_caches(False),
        reference.use_iterative_methods(tol=1e-7, max_iter=100, log_prob=False),
    ):
        distribution = reference(query, times)
        expected_mean, expected_cov = distribution.mean, distribution.covariance_matrix
        expected_gradient = torch.autograd.grad(
            expected_mean.square().sum() + expected_cov.square().sum(), (query, times)
        )
    with (
        gpytorch.settings.max_cholesky_size(0),
        model.use_iterative_methods(tol=1e-7, max_iter=100, log_prob=False),
    ):
        actual_mll = ExactMarginalLogLikelihood(model.likelihood, model)()
        posterior = model.posterior(query, times)
        actual_mean = posterior.mean.flatten()
        actual_cov = posterior.distribution.covariance_matrix
        torch.testing.assert_close(actual_mll, expected_mll, atol=2e-8, rtol=2e-8)
        torch.testing.assert_close(actual_mean, expected_mean, atol=2e-8, rtol=2e-8)
        torch.testing.assert_close(actual_cov, expected_cov, atol=2e-8, rtol=2e-8)
        actual_gradient = torch.autograd.grad(
            actual_mean.square().sum() + actual_cov.square().sum(), (query, times)
        )
        for a, b in zip(actual_gradient, expected_gradient, strict=True):
            torch.testing.assert_close(a, b, atol=2e-7, rtol=2e-7)

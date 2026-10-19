"""Gaussian posterior sampling, marginal evaluations and affine scalarization."""

from __future__ import annotations

import math
from statistics import NormalDist
from typing import Any

import pytest

from botorch_rs._arrays import kind_of, to_flat
from botorch_rs.acquisition.objective import ScalarizedPosteriorTransform
from botorch_rs.models import SingleTaskGP
from botorch_rs.posteriors import (
    GPyTorchPosterior,
    Posterior,
    scalarize_posterior,
    scalarize_posterior_gpytorch,
)
from botorch_rs.posteriors.posterior_list import PosteriorList
from helpers import as_kind, assert_close, check_kind, flatten, shape


def _model(kind: str) -> Any:
    return SingleTaskGP(
        as_kind([[0.0], [0.3], [0.8], [1.0]], kind),
        as_kind([[0.0, 0.4], [1.0, 0.2], [-0.2, 1.0], [0.1, 0.3]], kind),
    )


def test_gaussian_quantiles_and_density(kind: str) -> None:
    posterior = _model(kind).posterior(as_kind([[0.2], [0.6]], kind))
    mean, variance = flatten(posterior.mean), flatten(posterior.variance)
    probabilities = [0.1, 0.5, 0.9]
    quantile = posterior.quantile(as_kind(probabilities, kind))
    check_kind(quantile, kind)
    assert shape(quantile) == (3, 2, 2)
    expected = [
        m + math.sqrt(v) * NormalDist().inv_cdf(p)
        for p in probabilities
        for m, v in zip(mean, variance, strict=True)
    ]
    assert flatten(quantile) == pytest.approx(expected, abs=1e-13)
    values = [-0.5, 0.0, 1.0]
    density = posterior.density(as_kind(values, kind))
    check_kind(density, kind)
    assert shape(density) == (3, 2, 2)
    expected = [
        math.exp(-0.5 * (x - m) ** 2 / v) / math.sqrt(2 * math.pi * v)
        for x in values
        for m, v in zip(mean, variance, strict=True)
    ]
    assert flatten(density) == pytest.approx(expected, abs=1e-13)
    # A one-element probability tensor follows BoTorch's scalar behavior.
    assert shape(posterior.quantile(as_kind([0.5], kind))) == (2, 2)
    assert_close(posterior.quantile(0.5), posterior.mean, atol=0.0, rtol=0.0)
    assert all(math.isinf(v) and v < 0 for v in flatten(posterior.quantile(0.0)))
    assert all(math.isinf(v) and v > 0 for v in flatten(posterior.quantile(1.0)))
    assert all(math.isnan(v) for v in flatten(posterior.quantile(-0.1)))


def test_posterior_sample_has_no_gradients() -> None:
    torch = pytest.importorskip("torch")

    class DifferentiablePosterior(Posterior):
        def __init__(self) -> None:
            self.parameter = torch.tensor(2.0, requires_grad=True)

        def rsample(self, sample_shape: Any = None) -> Any:
            return self.parameter * torch.ones(sample_shape or (1,))

    posterior = DifferentiablePosterior()
    assert posterior.rsample((3,)).requires_grad
    assert not posterior.sample((3,)).requires_grad
    assert not posterior.sample().requires_grad
    with pytest.raises(NotImplementedError, match="quantile"):
        posterior.quantile(0.5)
    with pytest.raises(NotImplementedError, match="density"):
        posterior.density(0.0)


def test_gaussian_marginal_value_gradients() -> None:
    torch = pytest.importorskip("torch")
    posterior = _model("torch").posterior(torch.tensor([[0.2]], dtype=torch.float64))
    probability = torch.tensor(0.3, dtype=torch.float64, requires_grad=True)
    quantile = posterior.quantile(probability)
    actual = torch.autograd.grad(quantile.sum(), probability)[0]
    normal = torch.distributions.Normal(posterior.mean, posterior.variance.sqrt())
    expected = torch.autograd.grad(normal.icdf(probability).sum(), probability)[0]
    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
    value = torch.tensor(0.2, dtype=torch.float64, requires_grad=True)
    actual = torch.autograd.grad(posterior.density(value).sum(), value)[0]
    expected = torch.autograd.grad(normal.log_prob(value).exp().sum(), value)[0]
    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("batched", [False, True])
def test_source_scalarization_and_composition(kind: str, batched: bool) -> None:
    model = _model(kind)
    X = [[[0.2], [0.6]], [[0.1], [0.9]]] if batched else [[0.2], [0.6]]
    weights = as_kind([0.7, -0.2], kind)
    posterior = model.posterior(as_kind(X, kind))
    scalar = scalarize_posterior(posterior, weights, offset=0.3)
    assert isinstance(scalar, GPyTorchPosterior)
    assert scalar._joint is None  # Preserve lazy native source evaluation.
    direct = model.posterior(
        as_kind(X, kind), posterior_transform=ScalarizedPosteriorTransform(weights, 0.3)
    )
    assert_close(scalar.mean, direct.mean)
    assert_close(scalar.distribution.covariance_matrix, direct.distribution.covariance_matrix)
    mu, cov = scalarize_posterior_gpytorch(posterior, weights, offset=0.3)
    assert shape(mu) == shape(scalar.mean)[:-1]
    assert_close(flatten(mu), flatten(scalar.mean))
    assert_close(cov, scalar.distribution.covariance_matrix)
    composed = scalarize_posterior(scalar, as_kind([-2.0], kind), offset=0.1)
    expected = [0.1 - 2.0 * v for v in flatten(scalar.mean)]
    assert flatten(composed.mean) == pytest.approx(expected)
    assert flatten(composed.distribution.covariance_matrix) == pytest.approx(
        [4.0 * v for v in flatten(cov)]
    )
    assert shape(composed.sample((3,))) == (3, *shape(composed.mean))


def test_explicit_covariance_scalarization_preserves_cross_output_terms(kind: str) -> None:
    # A positive-definite covariance in output-major order with correlated
    # outputs. Each of two batches has distinct means and covariances.
    covariance = [
        [1.0, 0.2, 0.3, 0.1],
        [0.2, 2.0, 0.4, 0.5],
        [0.3, 0.4, 1.5, 0.6],
        [0.1, 0.5, 0.6, 2.5],
    ]
    mean = [[[0.1, 0.2], [0.3, 0.4]], [[1.1, 1.2], [1.3, 1.4]]]
    covariances = [covariance, [[2 * v for v in row] for row in covariance]]
    posterior = GPyTorchPosterior(None, ([], (2, 2, 0)), kind_of(as_kind(mean, kind)), False, 2)
    posterior._joint = (to_flat(as_kind(mean, kind)), to_flat(as_kind(covariances, kind)))
    scalar = scalarize_posterior(posterior, as_kind([0.7, -0.2], kind), offset=0.3)
    expected_mean = [0.3 + 0.7 * row[0] - 0.2 * row[1] for batch in mean for row in batch]
    expected_covariance = [
        sum(
            wt * c[t * 2 + i][u * 2 + j] * wu
            for t, wt in enumerate((0.7, -0.2))
            for u, wu in enumerate((0.7, -0.2))
        )
        for c in covariances
        for i in range(2)
        for j in range(2)
    ]
    check_kind(scalar.mean, kind)
    assert flatten(scalar.mean) == pytest.approx(expected_mean)
    assert flatten(scalar.distribution.covariance_matrix) == pytest.approx(expected_covariance)
    base = as_kind([[0.0, 0.0], [1.0, 1.0]], kind)
    samples = scalar.rsample_from_base_samples((2,), base)
    assert shape(samples) == (2, 2, 2, 1)
    assert_close(flatten(samples)[0:4], expected_mean)


def test_scalarize_posterior_list_and_validation(kind: str) -> None:
    model = _model(kind)
    X = as_kind([[0.2]], kind)
    posteriors = [model.posterior(X, output_indices=[i]) for i in range(2)]
    posterior = PosteriorList(*posteriors)
    scalar = scalarize_posterior(posterior, as_kind([0.7, -0.2], kind), offset=0.3)
    expected_mean = (
        0.3 + 0.7 * flatten(posteriors[0].mean)[0] - 0.2 * flatten(posteriors[1].mean)[0]
    )
    expected_var = (
        0.49 * flatten(posteriors[0].variance)[0] + 0.04 * flatten(posteriors[1].variance)[0]
    )
    assert flatten(scalar.mean) == pytest.approx([expected_mean])
    assert flatten(scalar.variance) == pytest.approx([expected_var])
    assert shape(scalar.sample((3,))) == (3, 1, 1)
    with pytest.raises(RuntimeError, match="Output shape"):
        scalarize_posterior(posterior, [1.0])
    with pytest.raises(ValueError, match="one-dimensional"):
        scalarize_posterior(posterior, [[1.0, 1.0]])
    with pytest.raises(NotImplementedError, match="q=1"):
        scalarize_posterior(
            PosteriorList(model.posterior(as_kind([[0.2], [0.6]], kind))), [1.0, 1.0]
        )

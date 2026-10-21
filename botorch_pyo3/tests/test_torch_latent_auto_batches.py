"""Automatic CPU construction must reach the implemented batched grid backend."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from test_torch_latent_kronecker import data, law  # noqa: E402

from botorch_rs.exceptions.errors import UnsupportedError  # noqa: E402
from botorch_rs.models import LatentKroneckerGP  # noqa: E402


@pytest.mark.parametrize("missing", [False, True])
@pytest.mark.parametrize("standardize", [False, True])
def test_cpu_batched_auto_construction_matches_actual_source_gaussian_laws(missing, standardize):
    pytest.importorskip("botorch")
    from botorch.models.latent_kronecker_gp import LatentKroneckerGP as Reference

    X, T, Y, Q, U = data("cpu", batch=True, missing=missing)
    options = {} if standardize else {"outcome_transform": None}
    model = LatentKroneckerGP(X, T, Y, **options)
    reference = Reference(X, T, Y, **options).double()
    assert model._tensor_backend == "torch"
    assert isinstance(model, torch.nn.Module)
    assert set(dict(model.named_parameters())) == set(dict(reference.named_parameters()))
    with torch.no_grad():
        for name, value in model.named_parameters():
            value.copy_(dict(reference.named_parameters())[name])
    query = Q.clone().requires_grad_()
    actual = model.posterior(query, U)
    mean, covariance, train, train_mean = law(reference, X, T, query, U)
    torch.testing.assert_close(actual.mean.flatten(-2), mean, atol=2e-8, rtol=2e-8)
    torch.testing.assert_close(
        actual.distribution.covariance_matrix, covariance, atol=2e-8, rtol=2e-8
    )
    expected_mll = (
        torch.distributions.MultivariateNormal(train_mean, covariance_matrix=train).log_prob(
            reference.train_targets
        )
        / reference.train_targets.shape[-1]
    )
    torch.testing.assert_close(model._exact_mll(), expected_mll, atol=2e-8, rtol=2e-8)
    value = actual.mean.square().sum() + actual.variance.square().sum()
    first = torch.autograd.grad(value, query, create_graph=True)[0]
    expected_value = mean.square().sum() + covariance.diagonal(dim1=-2, dim2=-1).square().sum()
    other = torch.autograd.grad(expected_value, query)[0]
    torch.testing.assert_close(first, other, atol=2e-7, rtol=2e-7)
    assert torch.isfinite(torch.autograd.grad(first.square().sum(), query)[0]).all()


def test_unbatched_cpu_auto_retains_native_backend_and_explicit_native_batch_error():
    X, T, Y, _, _ = data("cpu")
    model = LatentKroneckerGP(X, T, Y, outcome_transform=None)
    assert getattr(model, "_tensor_backend", None) != "torch"
    assert model.posterior(X).mean.shape == Y.shape
    X, T, Y, _, _ = data("cpu", batch=True)
    with pytest.raises(UnsupportedError, match="unbatched n x d"):
        LatentKroneckerGP(X, T, Y, outcome_transform=None, backend="native")


@pytest.mark.parametrize("standardize", [False, True])
def test_batched_auto_has_complete_gaussian_prediction_without_reference_dependencies(standardize):
    X, T, Y, Q, U = data("cpu", batch=True, missing=True)
    options = {} if standardize else {"outcome_transform": None}
    model = LatentKroneckerGP(X, T, Y, **options)
    query = Q.clone().requires_grad_()
    actual = model.posterior(query, U)
    mean, covariance, _, _ = law(model, X, T, query, U)
    torch.testing.assert_close(actual.mean.flatten(-2), mean)
    torch.testing.assert_close(actual.distribution.covariance_matrix, covariance)
    first = torch.autograd.grad(actual.mean.sin().sum(), query, create_graph=True)[0]
    assert torch.isfinite(torch.autograd.grad(first.square().sum(), query)[0]).all()

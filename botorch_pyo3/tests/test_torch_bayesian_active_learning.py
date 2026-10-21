"""Mixture information gain keeps all Gaussian component and device graphs."""

# ruff: noqa: E402
from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("botorch")
pytest.importorskip("jax")
pytest.importorskip("numpyro")

from botorch import settings
from botorch.acquisition.bayesian_active_learning import (
    qBayesianActiveLearningByDisagreement as ReferenceBALD,
)
from botorch.acquisition.objective import ScalarizedPosteriorTransform as ReferenceScalarize
from botorch.models import ModelListGP as ReferenceModelList
from botorch.sampling.normal import IIDNormalSampler as ReferenceSampler
from test_torch_fully_bayesian import DEVICES, FAMILIES, _pair

from botorch_rs.acquisition.bayesian_active_learning import qBayesianActiveLearningByDisagreement
from botorch_rs.acquisition.objective import ScalarizedPosteriorTransform
from botorch_rs.models import ModelListGP
from botorch_rs.sampling.normal import IIDNormalSampler


def _samplers(like, q, outputs=1, sample_shape=(12,)):
    shape = (*sample_shape, 1, 1, q, outputs) if outputs > 1 else (*sample_shape, 1, 1, q)
    normal = torch.linspace(
        -1.7, 1.9, math.prod(shape), dtype=like.dtype, device=like.device
    ).reshape(shape)
    samplers = []
    for cls in (IIDNormalSampler, ReferenceSampler):
        sampler = cls(torch.Size(sample_shape), seed=29)
        sampler.base_samples = normal.clone()
        samplers.append(sampler)
    return samplers


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("pending", [False, True])
def test_scalar_bald_matches_official_values_query_gradient_and_hvp(device, family, pending):
    (model, reference), data, _ = _pair(device, family, fixed=True)
    query = data[0][0].new_tensor(
        [[[0.24, 0.33, 0.61], [0.62, 0.54, 0.36]], [[0.31, 0.43, 0.58], [0.66, 0.49, 0.41]]]
    )
    X_pending = query.new_tensor([[0.47, 0.52, 0.48]]) if pending else None
    samplers = _samplers(query, 3 if pending else 2)
    results, acqs = [], []
    for cls, obj, sampler in zip(
        (qBayesianActiveLearningByDisagreement, ReferenceBALD),
        (model, reference),
        samplers,
        strict=True,
    ):
        acq = cls(obj, sampler=sampler, X_pending=X_pending)
        acqs.append(acq)
        X = query.clone().requires_grad_()
        value = acq(X)
        (gradient,) = torch.autograd.grad(
            value.sum(), X, create_graph=cls is qBayesianActiveLearningByDisagreement
        )
        results.append((value, gradient, X))
    for actual, expected in zip(results[0][:2], results[1][:2], strict=True):
        torch.testing.assert_close(actual, expected, atol=3e-10, rtol=3e-8)
    direction = torch.full_like(query, 0.17)
    (hvp,) = torch.autograd.grad((results[0][1] * direction).sum(), results[0][2])
    derivatives = []
    for sign in (-1, 1):
        X = (query + sign * 2e-5 * direction).requires_grad_()
        (derivative,) = torch.autograd.grad(acqs[1](X).sum(), X)
        derivatives.append(derivative)
    torch.testing.assert_close(hvp, (derivatives[1] - derivatives[0]) / 4e-5, atol=2e-7, rtol=3e-6)
    actual, gradient = acqs[0].value_and_grad(query)
    torch.testing.assert_close(actual, results[0][0], atol=0, rtol=0)
    torch.testing.assert_close(gradient, results[0][1], atol=0, rtol=0)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("scalarized", [False, True])
def test_fully_bayesian_model_list_joint_density_layout_and_graphs(device, scalarized):
    pairs = [_pair(device, FAMILIES[0], fixed=True, gradients=True) for _ in range(2)]
    model = ModelListGP(*(pair[0][0] for pair in pairs))
    reference = ReferenceModelList(*(pair[0][1] for pair in pairs))
    weights = pairs[0][1][0][0].new_tensor([0.3, 0.7], requires_grad=True)
    transform = ScalarizedPosteriorTransform(weights) if scalarized else None
    ref_transform = ReferenceScalarize(weights) if scalarized else None
    query = weights.new_tensor([[[0.24, 0.33, 0.61], [0.62, 0.54, 0.36]]], requires_grad=True)
    samplers = _samplers(query, 2, 1 if scalarized else 2, sample_shape=(2, 3))
    acq = qBayesianActiveLearningByDisagreement(
        model, sampler=samplers[0], posterior_transform=transform
    )
    actual = acq(query)
    posterior = model.posterior(query, observation_noise=True, posterior_transform=transform)
    mean = posterior.mean.transpose(-1, -2).flatten(-2)
    samples = samplers[0](posterior).reshape(6, *posterior.mean.shape).transpose(-1, -2).flatten(-2)
    covariance = posterior.distribution.covariance_matrix
    logdensities, own_logdensities = [], []
    # An independent component-by-component log density calculation checks
    # the joint output-major covariance without relying on the vectorized
    # component-axis broadcast of the acquisition implementation.
    for origin in range(3):
        columns = []
        for target in range(3):
            chol = torch.linalg.cholesky(covariance[..., target, :, :])
            residual = samples[..., origin, :] - mean[..., target, :]
            solve = torch.linalg.solve_triangular(
                chol, residual.unsqueeze(-1), upper=False
            ).squeeze(-1)
            columns.append(
                -0.5 * (mean.shape[-1] * math.log(2 * math.pi) + solve.square().sum(-1))
                - chol.diagonal(dim1=-2, dim2=-1).log().sum(-1)
            )
        logdensities.append(torch.logsumexp(torch.stack(columns, -1), -1) - math.log(3))
        own_logdensities.append(columns[origin])
    expected = -torch.stack(logdensities).mean((0, 1)) + torch.stack(own_logdensities).mean((0, 1))
    torch.testing.assert_close(actual, expected, atol=3e-13, rtol=3e-12)
    actual_gradient = torch.autograd.grad(actual.sum(), query, create_graph=True)[0]
    expected_gradient = torch.autograd.grad(expected.sum(), query, create_graph=True)[0]
    torch.testing.assert_close(actual_gradient, expected_gradient, atol=3e-11, rtol=3e-10)
    targets = [query, *model.parameters()]
    for pair in pairs:
        targets.extend(value for value in pair[1][0] if value is not None)
    if scalarized:
        targets.append(weights)
        # Official BALD assumes one MC sample axis, so compare its standard
        # shape here; multidimensional axes above have an independent oracle.
        port_sampler, ref_sampler = _samplers(query, 2)
        port_acq = qBayesianActiveLearningByDisagreement(model, port_sampler, transform)
        reference_acq = ReferenceBALD(reference, ref_sampler, ref_transform)
        with settings.propagate_grads(True):
            torch.testing.assert_close(port_acq(query), reference_acq(query), atol=3e-10, rtol=3e-8)
    first = torch.autograd.grad(acq(query).sum(), targets, create_graph=True)
    second = torch.autograd.grad(sum(v.square().sum() for v in first), targets)
    assert all(bool(v.isfinite().all()) for v in (*first, *second))


@pytest.mark.parametrize("device", DEVICES)
def test_identical_component_bald_is_zero_and_default_sampler_is_device_native(device):
    (model, _), data, samples = _pair(device, FAMILIES[0], fixed=True)
    samples = {name: value[:1] for name, value in samples.items()}
    model.load_mcmc_samples(samples)
    query = data[0][0].new_tensor([[0.24, 0.33, 0.61]], requires_grad=True)
    acq = qBayesianActiveLearningByDisagreement(model)
    value = acq(query)
    gradient = torch.autograd.grad(value.sum(), query, create_graph=True)[0]
    hvp = torch.autograd.grad(gradient.sum(), query)[0]
    torch.testing.assert_close(value, torch.zeros_like(value), atol=2e-15, rtol=0)
    torch.testing.assert_close(gradient, torch.zeros_like(gradient), atol=2e-14, rtol=0)
    assert bool(hvp.isfinite().all())
    assert acq.sampler.base_samples.device.type == device

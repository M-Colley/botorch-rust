"""Public supplied-distribution constructors preserve the caller's Torch law."""

from __future__ import annotations

import copy
import pickle
import subprocess
import sys

import pytest

from helpers import importorskip_botorch

importorskip_botorch()
torch = pytest.importorskip("torch")
from botorch.posteriors.gpytorch import GPyTorchPosterior as BtPosterior  # noqa: E402
from gpytorch.distributions import MultitaskMultivariateNormal, MultivariateNormal  # noqa: E402

from botorch_rs.posteriors import GPyTorchPosterior  # noqa: E402
from botorch_rs.posteriors.gpytorch import scalarize_posterior  # noqa: E402
from botorch_rs.sampling import IIDNormalSampler, SobolQMCNormalSampler, get_sampler  # noqa: E402


def distribution(multitask=False, interleaved=False, batch=(2,), torch_only=False):
    generator = torch.Generator().manual_seed(63)
    q, m = 3, 2 if multitask else 1
    mean = torch.rand(*batch, q, m, generator=generator, dtype=torch.float64)
    if not multitask:
        mean = mean.squeeze(-1)
    mean.requires_grad_()
    factor = torch.rand(*batch, q * m, q * m, generator=generator, dtype=torch.float64)
    factor = (factor * 0.2).requires_grad_()
    cov = factor @ factor.transpose(-1, -2) + torch.eye(q * m, dtype=mean.dtype)
    if multitask:
        return MultitaskMultivariateNormal(mean, cov, interleaved=interleaved), mean, factor
    cls = torch.distributions.MultivariateNormal if torch_only else MultivariateNormal
    return cls(mean, cov), mean, factor


@pytest.mark.parametrize("multitask,interleaved", [(False, False), (True, False), (True, True)])
@pytest.mark.parametrize("sample_shape", [(), (2,), (2, 3)])
def test_distribution_constructor_moments_samples_gradients(multitask, interleaved, sample_shape):
    dist, mean, factor = distribution(multitask, interleaved)
    actual, expected = GPyTorchPosterior(distribution=dist), BtPosterior(dist)
    assert actual.distribution is dist and actual.mvn is dist
    assert actual.dtype == mean.dtype and actual.device == mean.device
    assert actual.base_sample_shape == expected.base_sample_shape
    assert actual.batch_range == expected.batch_range
    assert actual._extended_shape(sample_shape) == expected._extended_shape(
        torch.Size(sample_shape)
    )
    torch.testing.assert_close(actual.mean, expected.mean)
    torch.testing.assert_close(actual.variance, expected.variance)
    z = torch.linspace(
        -0.8, 0.9, torch.Size((*sample_shape, *actual.base_sample_shape)).numel(), dtype=mean.dtype
    )
    z = z.reshape(*sample_shape, *actual.base_sample_shape)
    av = actual.rsample_from_base_samples(sample_shape, z)
    ev = expected.rsample_from_base_samples(torch.Size(sample_shape), z)
    torch.testing.assert_close(av, ev)
    ag = torch.autograd.grad(av.square().sum(), (mean, factor), retain_graph=True)
    eg = torch.autograd.grad(ev.square().sum(), (mean, factor), retain_graph=True)
    for a, e in zip(ag, eg, strict=True):
        torch.testing.assert_close(a, e)
    assert actual.rsample().shape == expected._extended_shape(torch.Size([1]))
    assert not actual.sample((2,)).requires_grad
    with torch.no_grad():
        assert not actual.rsample((2,)).requires_grad


@pytest.mark.parametrize("interleaved", [False, True])
def test_distribution_scalarization_preserves_parameter_and_weight_graphs(interleaved):
    from botorch.posteriors.gpytorch import scalarize_posterior as bt_scalarize

    dist, mean, factor = distribution(True, interleaved)
    weights = torch.tensor([-0.3, 0.8], dtype=mean.dtype, requires_grad=True)
    a, e = (
        scalarize_posterior(GPyTorchPosterior(dist), weights, 0.2),
        bt_scalarize(BtPosterior(dist), weights, 0.2),
    )
    av = a.mean.square().sum() + a.distribution.covariance_matrix.square().sum()
    ev = e.mean.square().sum() + e.distribution.covariance_matrix.square().sum()
    torch.testing.assert_close(av, ev)
    ag = torch.autograd.grad(av, (mean, factor, weights), retain_graph=True)
    eg = torch.autograd.grad(ev, (mean, factor, weights))
    for actual, expected in zip(ag, eg, strict=True):
        torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("multitask", [False, True])
def test_distribution_marginals_sampler_and_serialization(multitask):
    dist, _, _ = distribution(multitask, False)
    actual, expected = GPyTorchPosterior(dist), BtPosterior(dist)
    for value in [torch.tensor(0.3), torch.tensor([0.1, 0.7])]:
        torch.testing.assert_close(actual.quantile(value), expected.quantile(value))
        torch.testing.assert_close(actual.density(value), expected.density(value))
    assert isinstance(get_sampler(actual, (3,), seed=72), SobolQMCNormalSampler)
    for sampler_type in (IIDNormalSampler, SobolQMCNormalSampler):
        draws = sampler_type((3,), seed=72)(actual)
        assert draws.shape == actual._extended_shape((3,)) and draws.requires_grad
    # Copying nonleaf Torch graph nodes is disallowed by Torch itself. A
    # detached caller distribution has ordinary pickle/deepcopy semantics.
    detached = GPyTorchPosterior(MultivariateNormal(torch.zeros(3), torch.eye(3)))
    for restored in (copy.deepcopy(detached), pickle.loads(pickle.dumps(detached))):
        torch.testing.assert_close(restored.mean, detached.mean)
        torch.testing.assert_close(restored.variance, detached.variance)


def test_torch_multivariate_normal_without_gpytorch_sampling_protocol():
    dist, mean, factor = distribution(torch_only=True)
    posterior = GPyTorchPosterior(distribution=dist)
    z = torch.randn(2, *posterior.base_sample_shape, dtype=mean.dtype)
    actual = posterior.rsample_from_base_samples((2,), z)
    expected = (dist.loc + (dist.scale_tril @ z.unsqueeze(-1)).squeeze(-1)).unsqueeze(-1)
    torch.testing.assert_close(actual, expected)
    assert all(torch.isfinite(g).all() for g in torch.autograd.grad(actual.sum(), (mean, factor)))
    with pytest.raises(RuntimeError, match="sample_shape"):
        posterior.rsample_from_base_samples((3,), z)
    with pytest.raises(TypeError, match="multivariate normal"):
        GPyTorchPosterior(distribution=torch.distributions.Normal(0.0, 1.0))


def test_package_import_remains_torch_free():
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import botorch_rs; assert 'torch' not in sys.modules; assert 'gpytorch' not in sys.modules",
        ],
        check=True,
    )


@pytest.mark.parametrize("multitask,interleaved", [(False, False), (True, False), (True, True)])
def test_mixture_distribution_constructor_moments_quantiles_and_parameter_gradients(
    multitask, interleaved
):
    from botorch.posteriors.fully_bayesian import GaussianMixturePosterior as BtMixture

    from botorch_rs.posteriors.fully_bayesian import GaussianMixturePosterior

    dist, mean, factor = distribution(multitask, interleaved, batch=(2, 3))
    actual, expected = GaussianMixturePosterior(distribution=dist), BtMixture(dist)
    torch.testing.assert_close(actual.mixture_mean, expected.mixture_mean)
    torch.testing.assert_close(actual.mixture_variance, expected.mixture_variance)
    # Official b0dd's multi-output between-model term has incompatible
    # covariance layout. Compare with the correct total-covariance identity.
    loc = (
        mean.flatten(-2) if multitask and interleaved else actual.mean.transpose(-1, -2).flatten(-2)
    )
    centered = loc - loc.mean(-2, keepdim=True)
    covariance = dist.covariance_matrix.mean(-3) + (
        centered.unsqueeze(-1) * centered.unsqueeze(-2)
    ).mean(-3)
    torch.testing.assert_close(actual.mixture_covariance_matrix, covariance)
    if not multitask:
        torch.testing.assert_close(
            actual.mixture_covariance_matrix, expected.mixture_covariance_matrix.to_dense()
        )
    for value in (torch.tensor(0.3), torch.tensor([0.1, 0.7])):
        torch.testing.assert_close(actual.quantile(value), expected.quantile(value))
    loss = (
        actual.mixture_mean.square().sum()
        + actual.mixture_variance.sum()
        + actual.mixture_covariance_matrix.square().sum()
    )
    assert all(torch.isfinite(g).all() for g in torch.autograd.grad(loss, (mean, factor)))

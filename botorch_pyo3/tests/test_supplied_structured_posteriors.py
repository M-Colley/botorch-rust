"""Structured constructor and Matheron parity for explicit operator interop."""

from __future__ import annotations

import pytest

from helpers import importorskip_botorch

importorskip_botorch()
torch = pytest.importorskip("torch")

from botorch_rs.posteriors import (  # noqa: E402
    HigherOrderGPPosterior,
    LatentKroneckerGPPosterior,
    MultitaskGPPosterior,
)
from botorch_rs.sampling import IIDNormalSampler, SobolQMCNormalSampler, get_sampler  # noqa: E402


@pytest.mark.parametrize("batch", [(), (2,)])
def test_supplied_hogp_operators_match_official_matheron_and_query_graph(batch):
    from botorch.models.higher_order_gp import HigherOrderGP

    generator = torch.Generator().manual_seed(74)
    x = torch.rand(*batch, 4, 1, generator=generator, dtype=torch.float64)
    y = torch.rand(*batch, 4, 2, generator=generator, dtype=torch.float64)
    model = HigherOrderGP(x, y, outcome_transform=None)
    query = torch.rand(*batch, 2, 1, generator=generator, dtype=torch.float64, requires_grad=True)
    expected = model.posterior(query)
    actual = HigherOrderGPPosterior(
        distribution=expected.distribution,
        joint_covariance_matrix=expected.joint_covariance_matrix,
        train_train_covar=expected.train_train_covar,
        test_train_covar=expected.test_train_covar,
        train_targets=expected.train_targets,
        output_shape=expected.output_shape,
        num_outputs=expected.num_outputs,
    )
    assert actual.base_sample_shape == expected.base_sample_shape
    torch.testing.assert_close(actual.mean, expected.mean)
    torch.testing.assert_close(actual.variance, expected.variance)
    value = torch.tensor([0.2, 0.7], dtype=torch.float64)
    torch.testing.assert_close(actual.quantile(value), expected.quantile(value))
    torch.testing.assert_close(actual.density(value), expected.density(value))
    base = torch.rand(
        3, *actual.base_sample_shape, generator=generator, dtype=torch.float64, requires_grad=True
    )
    a = actual.rsample_from_base_samples((3,), base)
    e = expected.rsample_from_base_samples(torch.Size([3]), base)
    torch.testing.assert_close(a, e, rtol=2e-7, atol=2e-8)
    ag = torch.autograd.grad(a.square().sum(), (query, base), retain_graph=True)
    eg = torch.autograd.grad(e.square().sum(), (query, base), retain_graph=True)
    for av, ev in zip(ag, eg, strict=True):
        torch.testing.assert_close(av, ev, rtol=3e-6, atol=3e-7)
    assert isinstance(get_sampler(actual, (3,), seed=73), SobolQMCNormalSampler)
    assert IIDNormalSampler((3,), seed=73)(actual).shape == a.shape
    assert not actual.sample((2,)).requires_grad
    with pytest.raises(RuntimeError, match="sample_shape"):
        actual.rsample_from_base_samples((4,), base)


@pytest.mark.parametrize("noisy", [False, True])
@pytest.mark.parametrize("batch", [(), (2,)])
@pytest.mark.parametrize("noise_rank", [0, 1])
def test_supplied_multitask_operators_match_official_matheron(noisy, batch, noise_rank):
    from botorch.models import KroneckerMultiTaskGP
    from botorch.posteriors.multitask import MultitaskGPPosterior as BtPosterior
    from gpytorch.distributions import MultitaskMultivariateNormal
    from gpytorch.likelihoods import MultitaskGaussianLikelihood

    generator = torch.Generator().manual_seed(76)
    x = torch.rand(4, 1, generator=generator, dtype=torch.float64)
    y = torch.rand(4, 2, generator=generator, dtype=torch.float64)
    model = KroneckerMultiTaskGP(
        x,
        y,
        outcome_transform=None,
        likelihood=MultitaskGaussianLikelihood(num_tasks=2, rank=noise_rank),
    )
    query = torch.rand(2, 1, generator=generator, dtype=torch.float64, requires_grad=True)
    expected = model.posterior(query, observation_noise=noisy)
    if batch:
        # The official model's default batched kernel double-repeats its task
        # batch with the installed GPyTorch. Audit the posterior constructor
        # itself using broadcast copies of a valid unbatched operator system.
        def expand(operator):
            return operator.expand(*batch, *operator.shape[-2:])

        distribution = MultitaskMultivariateNormal(
            expected.mean.expand(*batch, *expected.mean.shape),
            expand(expected.distribution.lazy_covariance_matrix),
        )
        expected = BtPosterior(
            distribution,
            expand(expected.joint_covariance_matrix),
            expand(expected.test_train_covar),
            expand(expected.train_diff),
            expand(expected.test_mean),
            expand(expected.train_train_covar),
            expand(expected.train_noise),
            None if expected.test_noise is None else expand(expected.test_noise),
        )
    actual = MultitaskGPPosterior(
        expected.distribution,
        expected.joint_covariance_matrix,
        expected.test_train_covar,
        expected.train_diff,
        expected.test_mean,
        expected.train_train_covar,
        expected.train_noise,
        expected.test_noise,
    )
    assert actual.base_sample_shape == expected.base_sample_shape
    assert actual.batch_range == expected.batch_range
    torch.testing.assert_close(actual.mean, expected.mean)
    torch.testing.assert_close(actual.variance, expected.variance)
    base = torch.rand(
        2, *actual.base_sample_shape, generator=generator, dtype=torch.float64, requires_grad=True
    )
    a = actual.rsample_from_base_samples((2,), base)
    e = expected.rsample_from_base_samples(torch.Size([2]), base)
    torch.testing.assert_close(a, e, rtol=2e-7, atol=2e-8)
    ag = torch.autograd.grad(a.square().sum(), (query, base), retain_graph=True)
    eg = torch.autograd.grad(e.square().sum(), (query, base), retain_graph=True)
    for av, ev in zip(ag, eg, strict=True):
        torch.testing.assert_close(av, ev, rtol=3e-6, atol=3e-7)
    multi = torch.rand(2, 3, *actual.base_sample_shape, generator=generator, dtype=torch.float64)
    torch.testing.assert_close(
        actual.rsample_from_base_samples((2, 3), multi),
        actual.rsample_from_base_samples((6,), multi.reshape(6, *actual.base_sample_shape)).reshape(
            2, 3, *actual.mean.shape
        ),
    )
    for sampler in (IIDNormalSampler((3,), seed=76), SobolQMCNormalSampler((3,), seed=76)):
        assert sampler(actual).shape == actual._extended_shape((3,))
    replacement = actual.train_diff + 0.4
    torch.testing.assert_close(
        actual.rsample_from_base_samples((2,), base, train_diff=replacement),
        expected.rsample_from_base_samples(torch.Size([2]), base, train_diff=replacement),
        rtol=2e-7,
        atol=2e-8,
    )


class ExplicitLatentModel:
    batch_shape = torch.Size()
    train_inputs = (torch.zeros(3, 1),)
    train_T = torch.zeros(2, 1)
    train_targets = torch.zeros(5)

    def _rsample_from_base_samples(self, X, T, base_samples):
        self.last_base = base_samples
        return (
            X[..., :, :1]
            + T.transpose(-1, -2)
            + base_samples[..., -X.shape[-2] * T.shape[-2] :].reshape(
                *base_samples.shape[:-1], X.shape[-2], T.shape[-2]
            )
        )


def test_supplied_latent_model_protocol_preserves_query_time_and_base_graphs():
    from gpytorch.distributions import MultivariateNormal

    x = torch.tensor([[0.2], [0.7]], dtype=torch.float64, requires_grad=True)
    t = torch.tensor([[0.1], [0.8], [1.0]], dtype=torch.float64, requires_grad=True)
    distribution = MultivariateNormal(x + t.T, torch.eye(3, dtype=x.dtype).expand(2, 3, 3))
    model = ExplicitLatentModel()
    p = LatentKroneckerGPPosterior(model, distribution, x, t)
    assert p.distribution is distribution and p.mvn is distribution
    assert p.base_sample_shape == (17,)
    base = torch.randn(2, 17, dtype=x.dtype, requires_grad=True)
    samples = p.rsample_from_base_samples((2,), base)
    assert samples.shape == p._extended_shape((2,))
    assert all(
        torch.isfinite(g).all() for g in torch.autograd.grad(samples.square().sum(), (x, t, base))
    )
    assert IIDNormalSampler((3,), seed=73)(p).shape == (3, 2, 3)
    assert not p.sample((2,)).requires_grad
    with pytest.raises(TypeError, match="implement"):
        LatentKroneckerGPPosterior(object(), distribution, x, t)
    with pytest.raises(RuntimeError, match="sample_shape"):
        p.rsample_from_base_samples((3,), base)


def test_native_latent_model_accepts_supplied_moment_distribution_and_keeps_rust_sampling():
    from gpytorch.distributions import MultivariateNormal

    from botorch_rs.models import LatentKroneckerGP

    x, t = (
        torch.tensor([[0.1], [0.6]], dtype=torch.float64),
        torch.tensor([[0.2], [0.8]], dtype=torch.float64),
    )
    y = torch.tensor([[0.2, 0.8], [0.5, 0.7]], dtype=torch.float64)
    model = LatentKroneckerGP(x, t, y)
    query = torch.tensor([[0.4]], dtype=torch.float64)
    native = model.posterior(query)
    distribution = MultivariateNormal(native.mean, torch.diag_embed(native.variance))
    p = LatentKroneckerGPPosterior(model, distribution, query, t)
    legacy = LatentKroneckerGPPosterior(model, query, t)
    base = torch.randn(2, *native.base_sample_shape, dtype=x.dtype)
    torch.testing.assert_close(p.mean, native.mean)
    torch.testing.assert_close(legacy.mean, native.mean)
    torch.testing.assert_close(
        p.rsample_from_base_samples((2,), base), native.rsample_from_base_samples((2,), base)
    )


@pytest.mark.parametrize("family", ["higher_order", "latent", "kronecker"])
def test_native_structured_default_sampler_uses_actual_base_layout(family):
    from botorch_rs.models import HigherOrderGP, KroneckerMultiTaskGP, LatentKroneckerGP

    x = torch.tensor([[0.1], [0.5], [0.9]], dtype=torch.float64)
    y = torch.tensor([[0.2, 0.8], [0.4, 0.7], [0.9, 0.1]], dtype=torch.float64)
    if family == "higher_order":
        model = HigherOrderGP(x, y)
    elif family == "latent":
        model = LatentKroneckerGP(x, torch.tensor([[0.2], [0.8]], dtype=x.dtype), y)
    else:
        model = KroneckerMultiTaskGP(x, y)
    posterior = model.posterior(x[:2])
    for sampler in (get_sampler(posterior, (3,), seed=17), IIDNormalSampler((3,), seed=17)):
        draws = sampler(posterior)
        assert draws.shape == posterior._extended_shape((3,))
        assert sampler.base_samples.shape[-1] == posterior.base_sample_shape[-1]
        assert torch.isfinite(draws).all()

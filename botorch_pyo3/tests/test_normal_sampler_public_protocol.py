"""Normal samplers depend on public posterior metadata and sampling methods."""

# ruff: noqa: E402
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from botorch_rs.models.gp_regression import SingleTaskGP
from botorch_rs.sampling.normal import IIDNormalSampler, SobolQMCNormalSampler

DEVICES = [
    "cpu",
    pytest.param(
        "cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
    ),
]


class PublicPosterior:
    """Deliberately lacks the port's private posterior/array-kind markers."""

    def __init__(self, mean, scale):
        self.mean, self.scale = mean, scale
        self.received = None

    @property
    def device(self):
        return self.mean.device

    @property
    def dtype(self):
        return self.mean.dtype

    @property
    def base_sample_shape(self):
        return self.mean.shape[:-1]

    @property
    def batch_range(self):
        return (0, -1)

    def rsample_from_base_samples(self, *, sample_shape, base_samples):
        assert tuple(base_samples.shape) == (*sample_shape, *self.base_sample_shape)
        self.received = base_samples
        return self.mean + self.scale * base_samples.unsqueeze(-1)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("sampler_class", [IIDNormalSampler, SobolQMCNormalSampler])
@pytest.mark.parametrize("sample_shape", [(), (4,), (2, 3)])
def test_custom_public_posterior_device_base_layout_and_higher_graph(
    device, sampler_class, sample_shape
):
    mean = (
        torch.linspace(0.1, 0.8, 24, dtype=torch.double, device=device)
        .reshape(2, 4, 3, 1)
        .requires_grad_()
    )
    scale = mean.new_tensor(0.7, requires_grad=True)
    posterior = PublicPosterior(mean, scale)
    sampler = sampler_class(sample_shape, seed=29)
    state = torch.random.get_rng_state().clone()
    value = sampler(posterior)
    assert value.shape == (*sample_shape, *mean.shape)
    assert value.device == mean.device and value.dtype == mean.dtype
    assert sampler.base_samples.shape == (*sample_shape, 1, 1, 3)
    torch.testing.assert_close(state, torch.random.get_rng_state(), atol=0, rtol=0)
    torch.testing.assert_close(
        posterior.received[..., 0, 0, :], posterior.received[..., 1, 3, :], atol=0, rtol=0
    )
    torch.testing.assert_close(
        value, mean + scale * posterior.received.unsqueeze(-1), atol=0, rtol=0
    )
    first = torch.autograd.grad(value.square().sum(), (mean, scale), create_graph=True)
    second = torch.autograd.grad(sum(g.square().sum() for g in first), (mean, scale))
    assert all(bool(g.isfinite().all()) for g in (*first, *second))
    stored = sampler.base_samples
    torch.testing.assert_close(sampler(posterior), value, atol=0, rtol=0)
    assert sampler.base_samples is stored


@pytest.mark.parametrize("sampler_class", [IIDNormalSampler, SobolQMCNormalSampler])
@pytest.mark.parametrize("sample_shape", [(), (4,), (2, 3)])
def test_custom_public_posterior_matches_official_seeded_sampler(sampler_class, sample_shape):
    pytest.importorskip("botorch")
    from botorch.sampling import normal as reference

    mean = torch.linspace(0.1, 0.8, 24, dtype=torch.double).reshape(2, 4, 3, 1)
    posterior = PublicPosterior(mean, mean.new_tensor(0.7))
    actual_sampler = sampler_class(sample_shape, seed=29)
    expected_sampler = getattr(reference, sampler_class.__name__)(torch.Size(sample_shape), seed=29)
    actual, expected = actual_sampler(posterior), expected_sampler(posterior)
    torch.testing.assert_close(
        actual_sampler.base_samples, expected_sampler.base_samples, atol=2e-15, rtol=2e-14
    )
    torch.testing.assert_close(actual, expected, atol=2e-15, rtol=2e-14)


@pytest.mark.parametrize("sampler_class", [IIDNormalSampler, SobolQMCNormalSampler])
@pytest.mark.parametrize("backend", ["native", "torch"])
def test_empty_sample_shape_on_real_gaussian_and_explicit_normals(sampler_class, backend):
    x = torch.tensor([[0.1], [0.4], [0.8]], dtype=torch.double)
    model = SingleTaskGP(
        x,
        torch.sin(x * 3),
        train_Yvar=torch.full_like(x, 0.03),
        outcome_transform=None,
        backend=backend,
    )
    posterior = model.posterior(x[:2])
    sampler = sampler_class((), seed=3)
    actual = sampler(posterior)
    assert actual.shape == (2, 1)
    explicit = x.new_tensor([0.2, -0.4])
    sampler.base_samples = explicit
    expected = posterior.rsample_from_base_samples(sample_shape=(), base_samples=explicit)
    torch.testing.assert_close(sampler(posterior), expected, atol=1e-14, rtol=1e-13)
    if sampler_class is SobolQMCNormalSampler:
        sampler.base_samples = None
        assert sampler.base_samples is None
        torch.testing.assert_close(sampler(posterior), actual, atol=0, rtol=0)


@pytest.mark.parametrize("sampler_class", [IIDNormalSampler, SobolQMCNormalSampler])
def test_explicit_multidimensional_sample_shape_is_preserved(sampler_class):
    posterior = PublicPosterior(
        torch.zeros(2, 4, 3, 1, dtype=torch.double), torch.tensor(0.7, dtype=torch.double)
    )
    sampler = sampler_class((2, 3), seed=29)
    explicit = torch.linspace(-0.5, 0.5, 18, dtype=torch.double).reshape(2, 3, 1, 1, 3)
    sampler.base_samples = explicit
    torch.testing.assert_close(
        sampler(posterior),
        posterior.mean + posterior.scale * explicit.unsqueeze(-1),
        atol=0,
        rtol=0,
    )

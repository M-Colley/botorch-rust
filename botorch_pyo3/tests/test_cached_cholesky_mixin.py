"""Cached Gaussian sampling and preservation of baseline standard normals."""

from __future__ import annotations

import copy

import pytest

from botorch_rs.acquisition.cached_cholesky import CachedCholeskyMCSamplerMixin
from botorch_rs.models import SingleTaskGP
from botorch_rs.sampling.normal import IIDNormalSampler, SobolQMCNormalSampler
from helpers import KINDS, as_kind, to_list


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("outputs", [1, 2])
@pytest.mark.parametrize("sampler_cls", [IIDNormalSampler, SobolQMCNormalSampler])
def test_cached_samples_and_base_normals(kind: str, outputs: int, sampler_cls) -> None:
    torch = pytest.importorskip("torch")
    X = as_kind([[0.0], [0.3], [0.8], [1.0]], kind)
    Y = as_kind(
        [[0.0, 0.2][:outputs], [0.7, 0.8][:outputs], [0.3, 0.4][:outputs], [-0.2, 0.1][:outputs]],
        kind,
    )
    model = SingleTaskGP(X, Y)
    baseline = model.posterior(as_kind([[0.15], [0.65]], kind))
    joint = model.posterior(as_kind([[[0.15], [0.65], [0.45]], [[0.15], [0.65], [0.9]]], kind))
    sampler = sampler_cls((8,), seed=17)
    sampler(baseline)
    saved = torch.tensor(to_list(sampler.base_samples), dtype=torch.double)
    mixin = CachedCholeskyMCSamplerMixin(model, sampler=copy.deepcopy(sampler))
    mixin.model = model
    mixin._baseline_L = mixin._compute_root_decomposition(baseline)
    mixin.q_in = -1
    mixin.base_sampler = sampler
    mixin._set_sampler(1, joint)
    updated = torch.tensor(to_list(mixin.sampler.base_samples), dtype=torch.double)
    if outputs == 1:
        torch.testing.assert_close(updated[..., :2], saved.reshape(8, 1, 2), rtol=0, atol=0)
    else:
        torch.testing.assert_close(updated[..., :2, :], saved.reshape(8, 1, 2, 2), rtol=0, atol=0)
    torch.testing.assert_close(
        torch.tensor(to_list(sampler.base_samples), dtype=torch.double), saved, rtol=0, atol=0
    )
    cached = torch.tensor(to_list(mixin._get_f_X_samples(joint, 1)), dtype=torch.double)
    full = torch.tensor(to_list(mixin.sampler(joint)), dtype=torch.double)[..., -1:, :]
    torch.testing.assert_close(cached, full, atol=1e-10, rtol=1e-10)
    mixin._cache_root = False
    fallback = torch.tensor(to_list(mixin._get_f_X_samples(joint, 1)), dtype=torch.double)
    torch.testing.assert_close(fallback, full, atol=1e-10, rtol=1e-10)
    assert mixin.q_in == 1


@pytest.mark.parametrize("outputs", [1, 2])
def test_iid_base_update_matches_official(outputs: int) -> None:
    torch = pytest.importorskip("torch")
    bt = pytest.importorskip("botorch.sampling.normal")
    bt_posterior = pytest.importorskip("botorch.posteriors.gpytorch")
    distributions = pytest.importorskip("gpytorch.distributions")
    model = SingleTaskGP(
        torch.tensor([[0.0], [0.5], [1.0]], dtype=torch.double),
        torch.tensor(
            [[0.1, 0.3][:outputs], [0.6, 0.2][:outputs], [0.2, 0.1][:outputs]], dtype=torch.double
        ),
    )
    baseline = model.posterior(torch.tensor([[0.2], [0.6]], dtype=torch.double))
    joint = model.posterior(torch.tensor([[[0.2], [0.6], [0.7]]], dtype=torch.double))

    def reference(posterior):
        (mean, shape), (covariance, cshape) = posterior._get_joint()
        mean = torch.tensor(list(mean), dtype=torch.double).reshape(shape)
        covariance = torch.tensor(list(covariance), dtype=torch.double).reshape(cshape)
        distribution = (
            distributions.MultivariateNormal(mean.squeeze(-1), covariance)
            if outputs == 1
            else distributions.MultitaskMultivariateNormal(mean, covariance, interleaved=False)
        )
        return bt_posterior.GPyTorchPosterior(distribution)

    actual_base, expected_base = (
        IIDNormalSampler((8,), seed=17),
        bt.IIDNormalSampler(torch.Size([8]), seed=17),
    )
    actual, expected = (
        IIDNormalSampler((8,), seed=21),
        bt.IIDNormalSampler(torch.Size([8]), seed=21),
    )
    actual_base(baseline)
    expected_base(reference(baseline))
    actual._update_base_samples(joint, actual_base)
    expected._update_base_samples(reference(joint), expected_base)
    torch.testing.assert_close(actual.base_samples, expected.base_samples, rtol=0, atol=0)
    torch.testing.assert_close(actual(joint), expected(reference(joint)), atol=1e-10, rtol=1e-10)


def test_cache_fallback_and_unsupported_model_warning(monkeypatch) -> None:
    from botorch_rs.exceptions.errors import NanError

    with pytest.warns(RuntimeWarning, match="cache_root"):
        unsupported = CachedCholeskyMCSamplerMixin(object(), cache_root=True)
    assert unsupported._cache_root is False
    model = SingleTaskGP([[0.0], [1.0]], [[0.1], [0.5]])
    mixin = CachedCholeskyMCSamplerMixin(model, sampler=IIDNormalSampler((4,), seed=8))
    mixin.model = model
    posterior = model.posterior([[0.2], [0.8]])
    mixin.sampler(posterior)
    mixin._baseline_L = [[1.0]]
    monkeypatch.setattr(
        "botorch_rs.acquisition.cached_cholesky.sample_cached_cholesky",
        lambda *args, **kwargs: (_ for _ in ()).throw(NanError("bad covariance")),
    )
    with pytest.warns(Warning, match="Falling back"):
        samples = mixin._get_f_X_samples(posterior, 1)
    assert len(samples) == 4 and len(samples[0]) == 1

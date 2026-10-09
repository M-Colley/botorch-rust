"""SobolQMCNormalSampler, draw_sobol_samples and draw_sobol_normal_samples."""

from __future__ import annotations

import math

import pytest

from botorch_rs.exceptions import BotorchTensorDimensionError
from botorch_rs.models import SingleTaskGP
from botorch_rs.sampling import SobolQMCNormalSampler
from botorch_rs.utils.sampling import draw_sobol_normal_samples, draw_sobol_samples
from helpers import HAS_TORCH, as_kind, check_kind, flatten, make_data, shape


def test_sampler_construction() -> None:
    assert SobolQMCNormalSampler(64).sample_shape == (64,)
    assert SobolQMCNormalSampler([64]).num_samples == 64
    s = SobolQMCNormalSampler((4, 16), seed=3)
    assert s.sample_shape == (4, 16) and s.num_samples == 64 and s.seed == 3
    assert SobolQMCNormalSampler(8).seed != SobolQMCNormalSampler(8).seed  # random seeds
    if HAS_TORCH:
        import torch

        assert SobolQMCNormalSampler(torch.Size([32])).num_samples == 32
    with pytest.raises(TypeError):
        SobolQMCNormalSampler("many")
    with pytest.raises(TypeError):  # like BoTorch's signature
        SobolQMCNormalSampler(8, unknown=1)


def test_sampler_draws_posterior_samples(kind: str) -> None:
    X, Y = make_data(n=8, d=2, seed=1)
    model = SingleTaskGP(as_kind(X, kind), as_kind(Y, kind))
    post = model.posterior(as_kind([[[0.2, 0.3], [0.7, 0.8], [0.4, 0.1]]] * 2, kind))
    sampler = SobolQMCNormalSampler(sample_shape=[256], seed=0)
    samples = sampler(post)
    check_kind(samples, kind)
    assert shape(samples) == (256, 2, 3, 1)
    again = SobolQMCNormalSampler(sample_shape=[256], seed=0)(post)
    assert flatten(again) == flatten(samples)
    # Both batches use the same base samples (identical posteriors -> identical samples).
    s = [flatten(x) for x in samples[0]] if kind == "list" else [flatten(x) for x in samples[0]]
    assert s[0] == s[1]
    # QMC estimate of the mean.
    values = flatten(samples)
    means = [sum(values[i::6]) / 256 for i in range(3)]
    for m_est, m in zip(means, flatten(post.mean)[:3], strict=True):
        assert abs(m_est - m) < 0.05 * (1 + abs(m))


def test_explicit_base_samples() -> None:
    X, Y = make_data(n=8, d=2, seed=1)
    model = SingleTaskGP(X, Y)
    post = model.posterior([[0.2, 0.3], [0.7, 0.8]])
    sampler = SobolQMCNormalSampler(sample_shape=[2], seed=0)
    sampler.base_samples = [[[0.0, 0.0]], [[1.0, -1.0]]]  # BoTorch's `S x 1 x q` shape
    samples = sampler(post)
    assert flatten(samples)[:2] == pytest.approx(flatten(post.mean), rel=1e-14)
    explicit = post.rsample_from_base_samples([2], [[0.0, 0.0], [1.0, -1.0]])
    assert flatten(explicit) == flatten(samples)
    assert sampler.base_samples == [[[0.0, 0.0]], [[1.0, -1.0]]]
    with pytest.raises(BotorchTensorDimensionError):
        sampler.base_samples = [[0.0, 0.0]] * 3
    with pytest.raises(BotorchTensorDimensionError):
        sampler.base_samples = [[[0.0, 0.0], [0.0, 0.0]]] * 2


def test_draw_sobol_samples(kind: str) -> None:
    bounds = as_kind([[0.0, -2.0, 10.0], [1.0, 2.0, 20.0]], kind)
    X = draw_sobol_samples(bounds, n=64, q=2, seed=0)
    check_kind(X, kind)
    assert shape(X) == (64, 2, 3)
    values = flatten(X)
    for k, (lo, hi) in enumerate([(0.0, 1.0), (-2.0, 2.0), (10.0, 20.0)]):
        col = values[k::3]
        assert all(lo < v < hi for v in col)
        assert sum(col) / len(col) == pytest.approx((lo + hi) / 2, rel=0.05, abs=0.05)
    assert flatten(draw_sobol_samples(bounds, n=64, q=2, seed=0)) == values
    assert flatten(draw_sobol_samples(bounds, n=64, q=2, seed=1)) != values
    batched = draw_sobol_samples(bounds, n=4, q=2, batch_shape=[3], seed=0)
    assert shape(batched) == (4, 3, 2, 3)


def test_draw_sobol_normal_samples() -> None:
    Z = draw_sobol_normal_samples(d=2, n=1024, seed=0)
    assert shape(Z) == (1024, 2)
    values = flatten(Z)
    for k in range(2):
        col = values[k::2]
        mean = sum(col) / len(col)
        var = sum((v - mean) ** 2 for v in col) / (len(col) - 1)
        assert abs(mean) < 0.01 and abs(var - 1) < 0.02
    if HAS_TORCH:
        import torch

        Zt = draw_sobol_normal_samples(d=2, n=8, dtype=torch.float32, seed=0)
        assert Zt.dtype == torch.float32 and Zt.shape == (8, 2)
    assert math.isfinite(max(values))

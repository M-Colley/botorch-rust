"""Native structured posterior moment and Matheron query-adjoint checks."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from botorch_rs.models import HigherOrderGP, KroneckerMultiTaskGP, LatentKroneckerGP  # noqa: E402
from botorch_rs.models.transforms.input import Normalize  # noqa: E402
from botorch_rs.sampling import IIDNormalSampler, SobolQMCNormalSampler  # noqa: E402


def model_of(family, *, normalize=False, missing=False, dtype=torch.float64, standardize=True):
    x = torch.tensor([[0.1], [0.4], [0.8], [0.95]], dtype=dtype)
    y = torch.tensor([[0.2, 0.7], [0.5, 0.1], [0.8, 0.4], [0.3, 0.9]], dtype=dtype)
    kwargs = {"input_transform": Normalize(1)} if normalize else {}
    if not standardize:
        kwargs["outcome_transform"] = None
    if family == "higher":
        torch.manual_seed(42)
        return HigherOrderGP(x, y, **kwargs)
    if family == "kronecker":
        torch.manual_seed(42)
        return KroneckerMultiTaskGP(x, y, **kwargs)
    if missing:
        y[1, 0] = torch.nan
    return LatentKroneckerGP(x, torch.tensor([[0.2], [0.8]], dtype=dtype), y, **kwargs)


def value(model, query, mode, noisy=False, T=None):
    kwargs = {"T": T} if isinstance(model, LatentKroneckerGP) and T is not None else {}
    posterior = model.posterior(query, observation_noise=noisy, **kwargs)
    if mode == "samples":
        total = posterior.base_sample_shape[-1]
        base = torch.linspace(-0.7, 0.8, 2 * total, dtype=query.dtype).reshape(
            2, *(1 for _ in posterior.base_sample_shape[:-1]), total
        )
        values = posterior.rsample_from_base_samples((2,), base)
    elif mode == "covariance":
        values = posterior.distribution.covariance_matrix
    else:
        values = getattr(posterior, mode)
    weights = torch.linspace(0.2, 0.7, values.numel(), dtype=query.dtype).reshape(values.shape)
    return (values.square() * weights).sum()


@pytest.mark.parametrize("family", ["higher", "kronecker", "latent"])
@pytest.mark.parametrize("mode", ["mean", "variance", "covariance", "samples"])
@pytest.mark.parametrize("batch", [False, True])
def test_structured_query_gradients_match_native_finite_differences(family, mode, batch):
    model = model_of(family, standardize=not (family == "higher" and mode == "covariance"))
    query = torch.tensor([[0.27], [0.66]], dtype=torch.float64)
    if batch:
        query = torch.stack([query, query + 0.025])
    query.requires_grad_()
    gradient = torch.autograd.grad(value(model, query, mode), query)[0]
    for index in range(query.numel()):
        plus, minus = query.detach().clone(), query.detach().clone()
        plus.reshape(-1)[index] += 1e-6
        minus.reshape(-1)[index] -= 1e-6
        finite = (value(model, plus, mode) - value(model, minus, mode)) / 2e-6
        torch.testing.assert_close(gradient.reshape(-1)[index], finite, rtol=3e-5, atol=3e-6)


@pytest.mark.parametrize("family", ["higher", "kronecker"])
@pytest.mark.parametrize("mode", ["mean", "variance", "samples"])
def test_structured_noisy_input_transform_query_chain(family, mode):
    model = model_of(family, normalize=True)
    query = torch.tensor([[0.27], [0.66]], dtype=torch.float64, requires_grad=True)
    gradient = torch.autograd.grad(value(model, query, mode, noisy=True), query)[0]
    for i in range(2):
        plus, minus = query.detach().clone(), query.detach().clone()
        plus[i, 0] += 1e-6
        minus[i, 0] -= 1e-6
        finite = (
            value(model, plus, mode, noisy=True) - value(model, minus, mode, noisy=True)
        ) / 2e-6
        torch.testing.assert_close(gradient[i, 0], finite, rtol=3e-5, atol=3e-6)


@pytest.mark.parametrize("missing", [False, True])
@pytest.mark.parametrize("mode", ["mean", "variance", "covariance", "samples"])
def test_latent_feature_and_time_broadcast_gradients(missing, mode):
    model = model_of("latent", normalize=True, missing=missing)
    query = torch.tensor(
        [[[0.27], [0.66]], [[0.31], [0.69]]], dtype=torch.float64, requires_grad=True
    )
    times = torch.tensor([[[0.32], [0.63], [0.91]]], dtype=torch.float64, requires_grad=True)
    gx, gt = torch.autograd.grad(value(model, query, mode, T=times), (query, times))
    for temporal, source, gradient in ((False, query, gx), (True, times, gt)):
        for i in range(source.numel()):
            plus, minus = source.detach().clone(), source.detach().clone()
            plus.reshape(-1)[i] += 1e-6
            minus.reshape(-1)[i] -= 1e-6
            if temporal:
                finite = (
                    value(model, query.detach(), mode, T=plus)
                    - value(model, query.detach(), mode, T=minus)
                ) / 2e-6
            else:
                finite = (
                    value(model, plus, mode, T=times.detach())
                    - value(model, minus, mode, T=times.detach())
                ) / 2e-6
            torch.testing.assert_close(gradient.reshape(-1)[i], finite, rtol=3e-5, atol=3e-6)


@pytest.mark.parametrize("family", ["higher", "kronecker", "latent"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_structured_sampler_no_grad_dtype_and_posterior_snapshot(family, dtype):
    model = model_of(family, dtype=dtype)
    query = torch.tensor([[0.27], [0.66]], dtype=dtype, requires_grad=True)
    p = model.posterior(query)
    base = torch.zeros(2, *p.base_sample_shape, dtype=dtype)
    before = p.rsample_from_base_samples((2,), base)
    expected = torch.autograd.grad(before.square().sum() + p.mean.square().sum(), query)[0]
    model.likelihood.noise = 100.0
    after = p.rsample_from_base_samples((2,), base)
    torch.testing.assert_close(after, before)
    actual = torch.autograd.grad(after.square().sum() + p.mean.square().sum(), query)[0]
    torch.testing.assert_close(actual, expected)
    assert actual.dtype == dtype
    for sampler in (IIDNormalSampler((3,), seed=12), SobolQMCNormalSampler((3,), seed=12)):
        assert torch.isfinite(torch.autograd.grad(sampler(p).sum(), query)[0]).all()
    assert not p.sample((2,)).requires_grad
    with torch.no_grad():
        assert not p.mean.requires_grad and not p.rsample((2,)).requires_grad


@pytest.mark.parametrize("standardize", [False, True])
@pytest.mark.parametrize("mode", ["mean", "variance", "samples"])
def test_hogp_query_gradients_and_fixed_base_samples_match_official(standardize, mode):
    from helpers import importorskip_botorch

    importorskip_botorch()
    pytest.importorskip("botorch")
    from botorch.models.higher_order_gp import HigherOrderGP as BtHigher

    pytest.importorskip("gpytorch")
    from gpytorch.settings import max_cholesky_size

    pytest.importorskip("linear_operator")
    from linear_operator.settings import _fast_solves

    x = torch.tensor([[0.1], [0.4], [0.8]], dtype=torch.float64)
    y = torch.tensor(
        [[[0.2, 0.7], [0.4, 0.9]], [[0.5, 0.1], [0.2, 0.8]], [[0.8, 0.4], [0.5, 0.6]]],
        dtype=x.dtype,
    )
    kwargs = {} if standardize else {"outcome_transform": None}
    torch.manual_seed(24)
    actual = HigherOrderGP(x, y, **kwargs)
    torch.manual_seed(24)
    reference = BtHigher(x, y, **kwargs)
    query = torch.tensor([[[0.27], [0.66]], [[0.31], [0.69]]], dtype=x.dtype, requires_grad=True)
    other = query.detach().clone().requires_grad_()
    with _fast_solves(False), max_cholesky_size(10000):
        a, b = actual.posterior(query), reference.posterior(other)
        if mode == "samples":
            base = torch.linspace(
                -0.7, 0.8, 3 * torch.Size(a.base_sample_shape).numel(), dtype=x.dtype
            ).reshape(3, *a.base_sample_shape)
            av, bv = (
                a.rsample_from_base_samples((3,), base),
                b.rsample_from_base_samples(torch.Size([3]), base),
            )
        else:
            av, bv = getattr(a, mode), getattr(b, mode)
        torch.testing.assert_close(av, bv, rtol=2e-7, atol=2e-8)
        ag, bg = (
            torch.autograd.grad(av.square().sum(), query)[0],
            torch.autograd.grad(bv.square().sum(), other)[0],
        )
        torch.testing.assert_close(ag, bg, rtol=3e-6, atol=3e-7)

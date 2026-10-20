"""Device integrated Wiener law and correct coincident higher derivatives."""

# ruff: noqa: E402
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("botorch")

from botorch.models import SingleTaskGP as ReferenceGP
from botorch.models.kernels.integrated_white_noise import (
    IntegratedWhiteNoiseKernel as ReferenceKernel,
)
from gpytorch.kernels import ScaleKernel as ReferenceScale

from botorch_rs.kernels import ScaleKernel
from botorch_rs.mlls import ExactMarginalLogLikelihood
from botorch_rs.models import SingleTaskGP
from botorch_rs.models.kernels.integrated_white_noise import IntegratedWhiteNoiseKernel

DEVICES = [
    "cpu",
    pytest.param(
        "cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
    ),
]


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("order", [1, 2, 3, 4, 6])
def test_full_cross_diagonal_and_query_graphs(device, dtype, order):
    kernel = IntegratedWhiteNoiseKernel(
        order, backend="torch", dtype=dtype, device=device, batch_shape=(2,)
    )
    reference = ReferenceKernel(order, batch_shape=torch.Size([2]))
    x = torch.tensor([[[0.2], [0.8], [1.3]]], dtype=dtype, device=device, requires_grad=True)
    z = torch.tensor(
        [[[0.3], [0.7], [1.1], [1.5]], [[0.4], [0.6], [1.2], [1.6]]],
        dtype=dtype,
        device=device,
        requires_grad=True,
    )
    actual = kernel(x, z)
    expected = reference(x, z).to_dense()
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(
        kernel(x, z[..., :3, :], diag=True),
        reference(x, z[..., :3, :]).to_dense().diagonal(dim1=-2, dim2=-1),
    )
    assert kernel(x, z, last_dim_is_batch=True).shape == (2, 1, 3, 4)
    first = torch.autograd.grad(actual.square().sum(), (x, z), create_graph=True, retain_graph=True)
    expected_first = torch.autograd.grad(
        expected.square().sum(), (x, z), create_graph=True, retain_graph=True
    )
    for a, b in zip(first, expected_first, strict=True):
        torch.testing.assert_close(a, b)
    second = torch.autograd.grad(sum(g.sum() for g in first), (x, z))
    expected_second = torch.autograd.grad(sum(g.sum() for g in expected_first), (x, z))
    for a, b in zip(second, expected_second, strict=True):
        torch.testing.assert_close(a, b)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("order", [2, 3, 4, 6])
def test_smooth_coincident_hessian_and_gradgradcheck(device, order):
    kernel = IntegratedWhiteNoiseKernel(order, backend="torch").double().to(device)
    s = torch.tensor([[1.0]], dtype=torch.double, device=device, requires_grad=True)
    t = s.detach().clone().requires_grad_()
    assert torch.autograd.gradcheck(kernel, (s, t))
    assert torch.autograd.gradgradcheck(kernel, (s, t))
    if order == 2:
        ds = torch.autograd.grad(kernel(s, t).sum(), s, create_graph=True)[0]
        dss, dst = torch.autograd.grad(ds.sum(), (s, t))
        torch.testing.assert_close(dss, torch.zeros_like(s), atol=1e-15, rtol=0)
        torch.testing.assert_close(dst, torch.ones_like(s), atol=1e-15, rtol=0)
    # Differentiating (p-1) times in each time recovers Brownian covariance.
    # This is the highest smooth mixed derivative, of total order 2p-2.
    value = kernel(s, t).sum()
    for _ in range(order - 1):
        value = torch.autograd.grad(value, s, create_graph=True)[0].sum()
    for _ in range(order - 1):
        value = torch.autograd.grad(value, t, create_graph=True)[0].sum()
    torch.testing.assert_close(value, s.squeeze(), atol=2e-11, rtol=3e-10)


@pytest.mark.parametrize("device", DEVICES)
def test_scaled_active_dimension_gp_reference_and_full_graphs(device):
    torch.manual_seed(41)
    x = torch.rand(6, 2, dtype=torch.double, device=device, requires_grad=True)
    y = torch.randn(6, 1, dtype=x.dtype, device=device, requires_grad=True)
    kernel = ScaleKernel(IntegratedWhiteNoiseKernel(2, active_dims=[1]))
    model = SingleTaskGP(x, y, covar_module=kernel, backend="torch")
    reference = ReferenceGP(
        x.detach(), y.detach(), covar_module=ReferenceScale(ReferenceKernel(2, active_dims=[1]))
    ).to(x)
    with torch.no_grad():
        port = dict(model.named_parameters())
        for name, parameter in reference.named_parameters():
            parameter.copy_(port[name])
    query = torch.rand(3, 2, dtype=x.dtype, device=device, requires_grad=True)
    actual, expected = model.posterior(query), reference.posterior(query)
    torch.testing.assert_close(actual.mean, expected.mean, atol=2e-11, rtol=2e-9)
    torch.testing.assert_close(
        actual.distribution.covariance_matrix,
        expected.distribution.covariance_matrix,
        atol=2e-11,
        rtol=2e-9,
    )
    torch.testing.assert_close(
        torch.autograd.grad(actual.mean.sum(), query, retain_graph=True)[0],
        torch.autograd.grad(expected.mean.sum(), query)[0],
        atol=2e-10,
        rtol=2e-8,
    )
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    leaves = (x, y, query, *model.parameters())
    for _ in range(2):
        p = model.posterior(query)
        value = (
            p.mean.square().sum()
            + p.variance.sum()
            + mll(model(*model.train_inputs), model.train_targets).sum()
        )
        first = torch.autograd.grad(value, leaves, create_graph=True)
        second = torch.autograd.grad(sum(g.square().sum() for g in first), leaves)
        assert all(torch.isfinite(g).all() for g in (*first, *second))
        assert torch.equal(first[0][..., 0], torch.zeros_like(x[..., 0]))
    optimizer = torch.optim.Adam(mll.parameters(), lr=0.003)
    before = model.covar_module.raw_outputscale.detach().clone()
    optimizer.zero_grad()
    (-mll(model(*model.train_inputs), model.train_targets)).backward()
    optimizer.step()
    assert not torch.equal(before, model.covar_module.raw_outputscale)


def test_validation_and_brownian_tie_convention():
    kernel = IntegratedWhiteNoiseKernel(1, backend="torch").double()
    s = torch.tensor([[1.0]], dtype=torch.double, requires_grad=True)
    t = s.detach().clone().requires_grad_()
    gradient = torch.autograd.grad(kernel(s, t).sum(), (s, t))
    for value in gradient:
        torch.testing.assert_close(value, torch.full_like(value, 0.5))
    with pytest.raises(ValueError, match="scalar"):
        kernel(torch.ones(2, 2, dtype=torch.double))
    with pytest.raises(ValueError, match="nonnegative"):
        kernel(-s)
    with pytest.raises(ValueError, match="finite"):
        kernel(torch.full_like(s, float("nan")))
    with pytest.raises(ValueError, match="equal numbers"):
        kernel(s, t.expand(2, 1), diag=True)

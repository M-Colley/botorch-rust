"""Rank-zero generic and positive task covariance: native and device paths."""

# ruff: noqa: E402
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
gpytorch = pytest.importorskip("gpytorch")
from botorch.models.kernels.positive_index import PositiveIndexKernel as RefPositive

from botorch_rs.kernels import IndexKernel
from botorch_rs.mlls import ExactMarginalLogLikelihood
from botorch_rs.models import SingleTaskGP
from botorch_rs.models.kernels import PositiveIndexKernel

CONFIGS = [
    ("native", "cpu"),
    ("torch", "cpu"),
    pytest.param(
        "torch",
        "cuda",
        marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable"),
    ),
]


@pytest.mark.parametrize("backend,device", CONFIGS)
@pytest.mark.parametrize("positive", [False, True])
@pytest.mark.parametrize("batch", [(), (2,)])
def test_zero_rank_reference_diagonal_and_parameter_layout(backend, device, positive, batch):
    cls = PositiveIndexKernel if positive else IndexKernel
    ref_cls = RefPositive if positive else gpytorch.kernels.IndexKernel
    kernel = cls(3, rank=0, batch_shape=batch, active_dims=[1], backend=backend)
    if backend == "torch":
        kernel = kernel.double().to(device)
    raw = torch.tensor([-0.4, 0.2, 0.9], dtype=torch.double, device=device).expand(*batch, 3)
    kernel.initialize(raw_var=raw)
    reference = (
        ref_cls(3, rank=0, batch_shape=torch.Size(batch), active_dims=[1]).double().to(device)
    )
    reference.initialize(raw_var=raw)
    labels = torch.tensor(
        [[0.1, 0.0], [0.2, 2.0], [0.3, 1.0], [0.4, 0.0]], dtype=raw.dtype, device=device
    )
    query = labels[[1, 2, 0]]
    actual, expected = kernel(labels, query), reference(labels, query).to_dense()
    torch.testing.assert_close(actual, expected, atol=2e-13, rtol=2e-11)
    torch.testing.assert_close(
        kernel(labels[:3], query, diag=True), actual[..., :3, :].diagonal(dim1=-2, dim2=-1)
    )
    assert tuple(torch.as_tensor(kernel.covar_factor).shape) == (*batch, 3, 0)
    torch.testing.assert_close(torch.as_tensor(kernel.var), torch.nn.functional.softplus(raw))
    if backend == "torch":
        first = torch.autograd.grad(
            actual.square().sum(), tuple(kernel.parameters()), create_graph=True
        )
        oracle = torch.autograd.grad(
            expected.square().sum(), tuple(reference.parameters()), create_graph=True
        )
        for a, b in zip(first, oracle, strict=True):
            torch.testing.assert_close(a, b)
        second = torch.autograd.grad(
            sum(g.square().sum() for g in first), tuple(kernel.parameters())
        )
        refsecond = torch.autograd.grad(
            sum(g.square().sum() for g in oracle), tuple(reference.parameters())
        )
        for a, b in zip(second, refsecond, strict=True):
            torch.testing.assert_close(a, b)


@pytest.mark.parametrize("backend,device", CONFIGS)
@pytest.mark.parametrize("positive", [False, True])
def test_zero_rank_exact_gp_mll_parameter_graph(backend, device, positive):
    cls = PositiveIndexKernel if positive else IndexKernel
    kernel = cls(3, rank=0, backend=backend)
    if backend == "torch":
        kernel = kernel.double().to(device)
    x = torch.tensor([[0.0], [1.0], [2.0], [0.0], [1.0]], dtype=torch.double, device=device)
    y = torch.tensor([[0.3], [0.7], [-0.2], [0.4], [0.6]], dtype=x.dtype, device=device)
    noise = torch.full_like(y, 0.03)
    model = SingleTaskGP(x, y, noise, covar_module=kernel, outcome_transform=None, backend=backend)
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    loss = -mll(model(*model.train_inputs), model.train_targets).sum()
    assert torch.isfinite(loss)
    loss.backward()
    parameters = list(model.parameters())
    assert all(
        p.grad is not None and torch.isfinite(p.grad).all() for p in parameters if p.requires_grad
    )

# ruff: noqa: E402
"""Arc-cosine device parity and independent coincident/overflow laws."""

import pytest

pytest.importorskip("torch")


import torch

from botorch_rs.kernels import ScaleKernel
from botorch_rs.mlls import ExactMarginalLogLikelihood
from botorch_rs.models.gp_regression import SingleTaskGP
from botorch_rs.models.kernels.infinite_width_bnn import InfiniteWidthBNNKernel

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("depth", [0, 1, 3])
def test_nondegenerate_reference_values_and_full_graph(device, depth):
    pytest.importorskip("botorch")
    from botorch.models.kernels.infinite_width_bnn import InfiniteWidthBNNKernel as Reference

    actual = (
        InfiniteWidthBNNKernel(depth, batch_shape=[2], active_dims=[2, 0], backend="torch")
        .double()
        .to(device)
    )
    expected = Reference(depth, batch_shape=torch.Size([2]), active_dims=[2, 0]).double().to(device)
    X = torch.tensor(
        [[0.2, 7.0, 0.7], [0.8, 8.0, 0.4]], dtype=torch.double, device=device, requires_grad=True
    )
    Y = X.new_tensor([[0.6, 3.0, 0.1], [0.35, 4.0, 0.9], [0.4, 2.0, 0.25]], requires_grad=True)
    value, reference = actual(X, Y), expected(X, Y).to_dense()
    torch.testing.assert_close(value, reference)
    paired = actual(X, Y[:2], diag=True)
    torch.testing.assert_close(paired, value[..., :2].diagonal(dim1=-2, dim2=-1))
    inputs = (X, Y, *actual.parameters())
    first = torch.autograd.grad(value.sum(), inputs, create_graph=True)
    expected_first = torch.autograd.grad(reference.sum(), (X, Y, *expected.parameters()))
    for a, b in zip(first, expected_first, strict=True):
        torch.testing.assert_close(a, b)
    second = torch.autograd.grad(sum(g.square().sum() for g in first), inputs, allow_unused=True)
    assert all(g is None or torch.isfinite(g).all() for g in second)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("eps", [0.0, 1e-7, 1e-2])
def test_coincident_closed_recurrence_and_all_derivatives(device, eps):
    kernel = InfiniteWidthBNNKernel(3, acos_eps=eps, backend="torch").double().to(device)
    X = torch.tensor(
        [[0.2, 0.7], [0.4, 0.5], [0.2, 0.7]], dtype=torch.double, device=device, requires_grad=True
    )
    full = kernel(X)
    expected = kernel.weight_var * X.square().mean(-1) + kernel.bias_var.squeeze(-1)
    for _ in range(3):
        expected = kernel.weight_var.squeeze(-1) * expected / 2 + kernel.bias_var.squeeze(-1)
    torch.testing.assert_close(full.diagonal(), expected.flatten())
    torch.testing.assert_close(kernel(X, diag=True), expected.flatten())
    torch.testing.assert_close(full[0, 2], expected.flatten()[0])
    inputs = (X, *kernel.parameters())
    first = torch.autograd.grad(full.diagonal().sum(), inputs, create_graph=True)
    reference = torch.autograd.grad(expected.sum(), inputs, create_graph=True)
    second = torch.autograd.grad(sum(g.square().sum() for g in first), inputs)
    reference_second = torch.autograd.grad(sum(g.square().sum() for g in reference), inputs)
    for a, b in zip((*first, *second), (*reference, *reference_second), strict=True):
        torch.testing.assert_close(a, b)


@pytest.mark.parametrize("device", DEVICES)
def test_distinct_equal_inputs_preserve_mixed_partials_and_large_variance(device):
    kernel = InfiniteWidthBNNKernel(2, acos_eps=0, backend="torch").double().to(device)
    X = torch.tensor([[0.2, 0.7]], dtype=torch.double, device=device, requires_grad=True)
    Y = X.detach().clone().requires_grad_()
    value = kernel(X, Y).sum()
    expected = kernel.weight_var * (X * Y).mean() + kernel.bias_var
    for _ in range(2):
        expected = kernel.weight_var * expected / 2 + kernel.bias_var
    first = torch.autograd.grad(value, X, create_graph=True)[0]
    reference = torch.autograd.grad(expected.sum(), X, create_graph=True)[0]
    torch.testing.assert_close(first, reference)
    torch.testing.assert_close(
        torch.autograd.grad(first.sum(), Y)[0], torch.autograd.grad(reference.sum(), Y)[0]
    )
    large = X.new_tensor([[1e100]], requires_grad=True)
    kernel.weight_var, kernel.bias_var = 1.0, 1.0
    result = kernel(large).sum()
    torch.testing.assert_close(result, large.square().sum() / 4 + 1.75)
    torch.testing.assert_close(torch.autograd.grad(result, large)[0], large / 2)


@pytest.mark.parametrize("device", DEVICES)
def test_exact_gp_conversion_gradgradcheck_and_no_host_boundary(device, monkeypatch):
    X = torch.tensor(
        [[0.2, 0.7], [0.6, 0.3], [0.8, 0.9]], dtype=torch.double, device=device, requires_grad=True
    )
    Y = torch.tensor([[0.3], [0.5], [-0.2]], dtype=torch.double, device=device, requires_grad=True)
    query = X.new_tensor([[0.35, 0.45]], requires_grad=True)

    def moments(data, targets, test):
        source = InfiniteWidthBNNKernel(2, acos_eps=0)
        model = SingleTaskGP(
            data,
            targets,
            torch.full_like(targets, 0.04),
            covar_module=ScaleKernel(source),
            outcome_transform=None,
            backend="torch",
        )
        p = model.posterior(test)
        return p.mean + p.variance

    assert torch.autograd.gradcheck(moments, (X, Y, query), fast_mode=True)
    assert torch.autograd.gradgradcheck(moments, (X, Y, query), fast_mode=True)
    model = SingleTaskGP(
        X, Y, covar_module=InfiniteWidthBNNKernel(2), outcome_transform=None, backend="torch"
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("Neural covariance crossed a host-array boundary")

    with monkeypatch.context() as patch:
        for name in ("cpu", "numpy", "tolist"):
            patch.setattr(torch.Tensor, name, forbidden)
        p = model.posterior(query)
        loss = (
            p.mean.sum() + p.variance.sum() + ExactMarginalLogLikelihood(model.likelihood, model)()
        )
        first = torch.autograd.grad(loss, (X, Y, query, *model.parameters()), create_graph=True)
        second = torch.autograd.grad(
            sum(g.square().sum() for g in first),
            (X, Y, query, *model.parameters()),
            allow_unused=True,
        )
        assert all(g is None or torch.isfinite(g).all() for g in (*first, *second))

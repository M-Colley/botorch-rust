"""Dense tensor kernel primitives against GPyTorch, including derivative graphs."""

from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")
gpytorch = pytest.importorskip("gpytorch")

from botorch_rs import _botorch_rs as native  # noqa: E402

DEVICES = ["cpu", *(["cuda"] if torch.cuda.is_available() else [])]
KINDS = ["rq", "periodic", "cosine", "linear", "polynomial"]


def differentiable_distance(
    x1, x2, diag=False, last_dim_is_batch=False, square_dist=False, **kwargs
):
    """Equivalent reference distance without cdist's missing second backward."""
    if last_dim_is_batch:
        x1, x2 = x1.transpose(-1, -2).unsqueeze(-1), x2.transpose(-1, -2).unsqueeze(-1)
    difference = x1 - x2 if diag else x1.unsqueeze(-2) - x2.unsqueeze(-3)
    squared = difference.square().sum(-1)
    return squared if square_dist else squared.clamp_min(1e-30).sqrt()


def reference_kernel(kind, batch, device):
    kwargs = dict(batch_shape=torch.Size([2]) if batch else torch.Size())
    if kind in ("rq", "periodic", "linear"):
        kwargs["ard_num_dims"] = 3
    classes = dict(
        rq=gpytorch.kernels.RQKernel,
        periodic=gpytorch.kernels.PeriodicKernel,
        cosine=gpytorch.kernels.CosineKernel,
        linear=gpytorch.kernels.LinearKernel,
        polynomial=gpytorch.kernels.PolynomialKernel,
    )
    if kind == "polynomial":
        kwargs["power"] = 1.5
    return classes[kind](**kwargs).to(device=device, dtype=torch.double)


def actual(kernel, kind, x, z, diag=False, last=False):
    names = dict(
        rq=["alpha"],
        periodic=["period_length"],
        cosine=["period_length"],
        linear=["variance"],
        polynomial=["offset", "power"],
    )[kind]
    parameters = {name: getattr(kernel, name) for name in names}
    return native.tensor_kernel(
        kind,
        x,
        z,
        getattr(kernel, "lengthscale", None),
        diag=diag,
        last_dim_is_batch=last,
        parameters=parameters,
    )


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("batch", [False, True])
@pytest.mark.parametrize("diag,last", [(False, False), (True, False), (False, True), (True, True)])
def test_tensor_kernels_values_parameters_query_and_hvp_against_gpytorch(
    device, kind, batch, diag, last
):
    reference = reference_kernel(kind, batch, device)
    size = (2, 3, 3) if batch else (3, 3)
    x = torch.linspace(0.1, 0.8, math.prod(size), dtype=torch.double, device=device).reshape(size)
    x = x.transpose(-1, -2).requires_grad_()
    z = (
        torch.linspace(0.2, 0.9, 9, dtype=torch.double, device=device)
        .reshape(3, 3)
        .requires_grad_()
    )
    output = actual(reference, kind, x, z, diag, last)
    # GPyTorch's polynomial offset misses the extra feature batch in both
    # full and diagonal paths. Evaluate its actual formula per feature.
    if kind == "polynomial" and batch and last:
        expected = torch.stack(
            [
                reference.forward(x[..., j : j + 1], z[..., j : j + 1], diag=diag)
                for j in range(x.shape[-1])
            ],
            dim=-2 if diag else -3,
        )
    else:
        expected = reference.forward(x, z, diag=diag, last_dim_is_batch=last)
        if not torch.is_tensor(expected):
            expected = expected.to_dense()
    assert output.device.type == device and output.grad_fn is not None
    torch.testing.assert_close(output, expected, rtol=2e-12, atol=2e-12)
    args = (x, z, *reference.parameters())
    weights = torch.linspace(-0.3, 0.7, output.numel(), dtype=output.dtype, device=device).reshape(
        output.shape
    )
    gradients = torch.autograd.grad((output * weights).sum(), args, create_graph=True)
    expected_gradients = torch.autograd.grad((expected * weights).sum(), args, create_graph=True)
    for value, want in zip(gradients, expected_gradients, strict=True):
        torch.testing.assert_close(value, want, rtol=2e-9, atol=2e-10)
    if kind in ("periodic", "cosine"):
        # The installed reference uses torch.cdist, whose second backward is
        # unimplemented. Keep its kernel formula and replace only its distance
        # primitive, after checking actual reference values/first derivatives.
        reference.covar_dist = differentiable_distance
        expected = reference.forward(x, z, diag=diag, last_dim_is_batch=last)
        expected_gradients = torch.autograd.grad(
            (expected * weights).sum(), args, create_graph=True
        )
    hvp = torch.autograd.grad(sum(g.square().sum() for g in gradients), args)
    expected_hvp = torch.autograd.grad(sum(g.square().sum() for g in expected_gradients), args)
    for value, want in zip(hvp, expected_hvp, strict=True):
        torch.testing.assert_close(value, want, rtol=2e-8, atol=2e-9)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_cosine_large_phase_accuracy_and_higher_derivatives(device, dtype):
    x = torch.tensor(
        [[4.0, 1.0], [2.0, 2.0], [8.0, 0.0]],
        device=device,
        dtype=dtype,
        requires_grad=True,
    )
    z = torch.tensor(
        [[0.0, 0.0], [2.0, 1.0], [1.0, 0.0]],
        device=device,
        dtype=dtype,
        requires_grad=True,
    )
    period = torch.ones(1, 1, device=device, dtype=dtype, requires_grad=True)
    result = native.tensor_kernel("cosine", x, z, parameters={"period_length": period})
    distance = ((x.unsqueeze(-2) - z.unsqueeze(-3)) / period).norm(dim=-1)
    expected = (math.pi * distance).cos()
    tolerance = 3e-6 if dtype == torch.float32 else 3e-12
    torch.testing.assert_close(result, expected, rtol=tolerance, atol=tolerance)
    variables = (x, z, period)
    actual_grad = torch.autograd.grad(result.sum(), variables, create_graph=True)
    expected_grad = torch.autograd.grad(expected.sum(), variables, create_graph=True)
    actual_hvp = torch.autograd.grad(sum(g.square().sum() for g in actual_grad), variables)
    expected_hvp = torch.autograd.grad(sum(g.square().sum() for g in expected_grad), variables)
    for actual_value, expected_value in zip(actual_hvp, expected_hvp, strict=True):
        torch.testing.assert_close(
            actual_value,
            expected_value,
            rtol=4e-5 if dtype == torch.float32 else 2e-10,
            atol=2e-3 if dtype == torch.float32 else 2e-9,
        )


@pytest.mark.parametrize("device", DEVICES)
def test_cosine_all_derivative_orders_at_coincident_points(device):
    point = torch.zeros(1, 1, device=device, dtype=torch.double, requires_grad=True)
    period = torch.ones(1, 1, device=device, dtype=torch.double)
    value = native.tensor_kernel(
        "cosine", point, torch.zeros_like(point), parameters={"period_length": period}
    ).sum()
    for order in range(1, 7):
        (value,) = torch.autograd.grad(value.sum(), point, create_graph=True)
        expected = 0.0 if order % 2 else (-1.0) ** (order // 2) * math.pi**order
        torch.testing.assert_close(value, torch.full_like(point, expected), rtol=2e-10, atol=2e-10)


@pytest.mark.parametrize("device", DEVICES)
def test_linear_zero_variance_has_finite_exact_parameter_gradients(device):
    variance = torch.zeros(1, 3, dtype=torch.double, device=device, requires_grad=True)
    x = torch.tensor([[0.2, 0.4, 0.6]], dtype=torch.double, device=device, requires_grad=True)
    z = torch.tensor([[0.3, 0.5, 0.7]], dtype=torch.double, device=device, requires_grad=True)
    output = native.tensor_kernel("linear", x, z, parameters={"variance": variance})
    (derivative,) = torch.autograd.grad(output.sum(), variance, create_graph=True)
    torch.testing.assert_close(derivative, x * z)
    assert all(torch.isfinite(g).all() for g in torch.autograd.grad(derivative.sum(), (x, z)))


def test_kernel_configuration_errors_do_not_silently_drop_arguments():
    x = torch.ones(2, 3, dtype=torch.double)
    with pytest.raises(ValueError, match="requires parameter"):
        native.tensor_kernel("rq", x, x)
    with pytest.raises(ValueError, match="Unknown"):
        native.tensor_kernel(
            "linear", x, x, parameters={"variance": torch.ones(1, 1), "alpha": 1.0}
        )
    with pytest.raises(ValueError, match="do not use lengthscale"):
        native.tensor_kernel(
            "linear", x, x, torch.ones(1, 3), parameters={"variance": torch.ones(1, 1)}
        )
    with pytest.raises(ValueError, match="fixed Python number"):
        native.tensor_kernel(
            "polynomial", x, x, parameters={"offset": torch.ones(1), "power": torch.tensor(2.0)}
        )

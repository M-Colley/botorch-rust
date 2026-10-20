"""Torch-dispatched exact GP algebra retains device and higher-order graphs."""

from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")

from botorch_rs import _botorch_rs as native  # noqa: E402

DEVICES = ["cpu", *(["cuda"] if torch.cuda.is_available() else [])]


def kernel(x, z, lengthscale, scale, kind="rbf", nu=2.5):
    return scale * native.tensor_kernel(kind, x, z, lengthscale, nu=nu)


def inputs(device="cpu", dtype=torch.double):
    values = [
        [[0.1, 0.2], [0.4, 0.7], [0.9, 0.6]],
        [0.3, -0.2, 0.5],
        [[0.25, 0.35], [0.65, 0.45]],
        [0.05, 0.07, 0.04],
        [[0.6, 0.8]],
        1.2,
        0.15,
        [0.02, 0.03],
    ]
    return tuple(torch.tensor(v, dtype=dtype, device=device, requires_grad=True) for v in values)


def native_outputs(*args, kind="rbf", nu=2.5):
    x, y, query, noise, lengthscale, scale, mean, observation_noise = args
    ktt = kernel(x, x, lengthscale, scale, kind, nu)
    ktq = kernel(x, query, lengthscale, scale, kind, nu)
    kqq = kernel(query, query, lengthscale, scale, kind, nu)
    posterior = native.tensor_exact_gp_posterior(
        y,
        ktt,
        ktq,
        kqq,
        mean.expand_as(y),
        mean.expand(query.shape[:-1]),
        noise,
        observation_noise=observation_noise,
    )
    mll = native.tensor_exact_mll(y, ktt, mean.expand_as(y), noise, normalize=True)
    return *posterior, mll


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_exact_conditioning_and_mll_match_independent_torch_linalg(device, dtype):
    args = inputs(device, dtype)
    x, y, query, noise, lengthscale, scale, mean, observation_noise = args

    def expected_kernel(a, b):
        delta = (a.unsqueeze(-2) - b.unsqueeze(-3)) / lengthscale
        return scale * torch.exp(-0.5 * delta.square().sum(-1))

    ktt, ktq, kqq = expected_kernel(x, x), expected_kernel(x, query), expected_kernel(query, query)
    covariance = ktt + torch.diag_embed(noise)
    residual = y - mean
    expected_mean = mean + (ktq.T @ torch.linalg.solve(covariance, residual.unsqueeze(-1))).squeeze(
        -1
    )
    expected_covariance = kqq - ktq.T @ torch.linalg.solve(covariance, ktq)
    expected_covariance = expected_covariance + torch.diag_embed(observation_noise)
    expected_mll = (
        torch.distributions.MultivariateNormal(
            mean.expand_as(y), covariance_matrix=covariance
        ).log_prob(y)
        / y.shape[-1]
    )
    actual = native_outputs(*args)
    expected = (expected_mean, expected_covariance, expected_mll)
    for value, want in zip(actual, expected, strict=True):
        assert value.device.type == device and value.dtype == dtype and value.grad_fn is not None
        torch.testing.assert_close(
            value,
            want,
            rtol=2e-5 if dtype == torch.float32 else 2e-12,
            atol=2e-6 if dtype == torch.float32 else 2e-12,
        )
    loss = actual[0].square().sum() + actual[1].square().sum() + actual[2]
    expected_loss = expected[0].square().sum() + expected[1].square().sum() + expected[2]
    gradients = torch.autograd.grad(loss, args, create_graph=True)
    expected_gradients = torch.autograd.grad(expected_loss, args, create_graph=True)
    for value, want in zip(gradients, expected_gradients, strict=True):
        torch.testing.assert_close(
            value,
            want,
            rtol=5e-5 if dtype == torch.float32 else 2e-11,
            atol=5e-6 if dtype == torch.float32 else 2e-11,
        )
    hvp = torch.autograd.grad(sum(g.square().sum() for g in gradients), args)
    expected_hvp = torch.autograd.grad(sum(g.square().sum() for g in expected_gradients), args)
    for value, want in zip(hvp, expected_hvp, strict=True):
        torch.testing.assert_close(
            value,
            want,
            rtol=2e-4 if dtype == torch.float32 else 2e-10,
            atol=2e-5 if dtype == torch.float32 else 2e-10,
        )


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize(("kind", "nu"), [("rbf", 2.5), ("matern", 1.5), ("matern", 2.5)])
def test_exact_gp_all_tensor_inputs_pass_gradcheck_and_gradgradcheck(device, kind, nu):
    args = inputs(device)

    def function(*values):
        return native_outputs(*values, kind=kind, nu=nu)

    assert torch.autograd.gradcheck(function, args, fast_mode=True, atol=2e-5, rtol=2e-4)
    assert torch.autograd.gradgradcheck(function, args, fast_mode=True, atol=3e-5, rtol=3e-4)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize(
    ("kind", "nu", "curvature"),
    [("rbf", 2.5, 1.0), ("matern", 1.5, 3.0), ("matern", 2.5, 5.0 / 3.0)],
)
def test_coincident_kernel_has_correct_hessian(device, kind, nu, curvature):
    point = torch.tensor([[0.3, 0.6]], dtype=torch.double, device=device, requires_grad=True)
    other = point.detach().clone()
    lengthscale = torch.tensor([[0.7, 0.9]], dtype=torch.double, device=device, requires_grad=True)
    value = native.tensor_kernel(kind, point, other, lengthscale, nu=nu).sum()
    (gradient,) = torch.autograd.grad(value, point, create_graph=True)
    assert torch.equal(gradient, torch.zeros_like(gradient))
    rows = [torch.autograd.grad(gradient[0, j], point, retain_graph=True)[0][0] for j in range(2)]
    hessian = torch.stack(rows)
    torch.testing.assert_close(hessian, -torch.diag(curvature / lengthscale[0].square()))


@pytest.mark.parametrize("device", DEVICES)
def test_arbitrary_broadcast_batches_and_independent_outputs(device):
    args = inputs(device)
    x, y, query, noise, lengthscale, scale, mean, observation_noise = args
    x = x.expand(2, 1, 3, 2)
    y = torch.stack([y, y + 0.1]).unsqueeze(1)
    query = torch.stack([query, query + 0.05, query + 0.1]).unsqueeze(0)
    noise = noise.expand(2, 1, 3)
    values = native_outputs(x, y, query, noise, lengthscale, scale, mean, observation_noise)
    assert values[0].shape == (2, 3, 2)
    assert values[1].shape == (2, 3, 2, 2)
    assert values[2].shape == (2, 1)
    for output in range(2):
        for candidate in range(3):
            direct = native_outputs(
                x[output, 0],
                y[output, 0],
                query[0, candidate],
                noise[output, 0],
                lengthscale,
                scale,
                mean,
                observation_noise,
            )
            torch.testing.assert_close(values[0][output, candidate], direct[0])
            torch.testing.assert_close(values[1][output, candidate], direct[1])
    gradients = torch.autograd.grad(sum(v.sum() for v in values), args, create_graph=True)
    assert all(torch.isfinite(g).all() and g.device.type == device for g in gradients)


@pytest.mark.parametrize("device", DEVICES)
def test_kernel_diag_last_dimension_and_noncontiguous_inputs(device):
    x = torch.arange(12, dtype=torch.double, device=device).reshape(3, 4).T / 12
    z = x + 0.1
    lengthscale = torch.tensor([[0.6, 0.8, 0.9]], dtype=torch.double, device=device)
    full = native.tensor_kernel("rbf", x, z, lengthscale)
    diag = native.tensor_kernel("rbf", x, z, lengthscale, diag=True)
    torch.testing.assert_close(diag, full.diagonal())
    by_dimension = native.tensor_kernel("rbf", x, z, lengthscale, last_dim_is_batch=True)
    expected = torch.exp(
        -0.5 * (((x / lengthscale).T.unsqueeze(-1) - (z / lengthscale).T.unsqueeze(-2)) ** 2)
    )
    torch.testing.assert_close(by_dimension, expected)


def test_jitter_is_explicit_and_shape_errors_are_clear():
    y = torch.tensor([0.1, 0.2], dtype=torch.double)
    bad_covariance = torch.tensor([[1.0, 2.0], [2.0, 1.0]], dtype=torch.double)
    mean, noise = torch.zeros_like(y), torch.zeros_like(y)
    with pytest.raises(torch.linalg.LinAlgError):
        native.tensor_exact_mll(y, bad_covariance, mean, noise)
    value = native.tensor_exact_mll(y, bad_covariance, mean, noise, jitter=2.0)
    assert torch.isfinite(value)
    with pytest.raises(ValueError, match="jitter"):
        native.tensor_exact_mll(y, bad_covariance, mean, noise, jitter=math.nan)
    with pytest.raises(ValueError, match="k_train"):
        native.tensor_exact_mll(y, torch.eye(3), mean, noise)
    with pytest.raises(ValueError, match="Torch tensor"):
        native.tensor_kernel("rbf", [[0.1]], [[0.2]])
    with pytest.raises(NotImplementedError, match="nu"):
        native.tensor_kernel("matern", y.unsqueeze(-1), y.unsqueeze(-1), nu=0.8)

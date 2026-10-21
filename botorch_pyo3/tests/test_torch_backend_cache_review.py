"""Cache context transitions retain live, repeatable training and query graphs."""

from __future__ import annotations

from unittest.mock import patch

import pytest

torch = pytest.importorskip("torch")

from botorch_rs import _botorch_rs as native  # noqa: E402
from botorch_rs.mlls import ExactMarginalLogLikelihood  # noqa: E402
from botorch_rs.models import SingleTaskGP  # noqa: E402

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


def training_data(device="cpu", dtype=torch.double):
    x = torch.tensor([[0.1], [0.45], [0.9]], dtype=dtype, device=device)
    y = torch.tensor([[0.2], [0.7], [-0.3]], dtype=dtype, device=device)
    return x, y


@pytest.mark.parametrize("device", DEVICES)
def test_inference_tensor_construction_and_mutation(device):
    with torch.inference_mode():
        x, y = training_data(device)
        model = SingleTaskGP(x, y, backend="torch")
        query = x + 0.03
        first = model.posterior(query).mean.clone()
        y.add_(0.25)
        changed = model.posterior(query).mean
        assert torch.isfinite(changed).all()
        assert not torch.equal(first, changed)
        fresh = SingleTaskGP(x, y, backend="torch").posterior(query)
        torch.testing.assert_close(changed, fresh.mean)


@pytest.mark.parametrize("device", DEVICES)
def test_inference_cache_cannot_escape_into_query_autograd(device):
    x, y = training_data(device)
    model = SingleTaskGP(x, y, backend="torch").requires_grad_(False)
    query = (x + 0.03).requires_grad_()
    with patch.object(
        native, "tensor_exact_gp_factor", wraps=native.tensor_exact_gp_factor
    ) as factor:
        with torch.inference_mode():
            model.posterior(query)
            assert factor.call_count == 1
        for _ in range(2):
            posterior = model.posterior(query)
            loss = posterior.mean.square().sum() + posterior.variance.sum()
            first = torch.autograd.grad(loss, query, create_graph=True)[0]
            second = torch.autograd.grad(first.sum(), query)[0]
            assert torch.isfinite(first).all() and torch.isfinite(second).all()
        # Rebuild once when leaving inference mode, then reuse only normal
        # constant factors while constructing a fresh query graph each time.
        assert factor.call_count == 2


def weighted_loss(model, query):
    posterior = model.posterior(query, observation_noise=True)
    return (
        (posterior.mean * query.square()).sum()
        + posterior.variance.square().sum()
        + ExactMarginalLogLikelihood(model.likelihood, model)().sum()
    )


def derivatives(loss, arguments):
    first = torch.autograd.grad(loss, arguments, create_graph=True)
    direction = sum(
        (
            gradient
            * torch.linspace(0.2, 0.8, gradient.numel(), device=gradient.device).reshape(
                gradient.shape
            )
        ).sum()
        for gradient in first
    )
    second = torch.autograd.grad(direction, arguments)
    return first, second


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("fixed_noise", [False, True])
def test_no_grad_warmup_and_data_mutation_preserve_training_first_and_second_gradients(
    device, fixed_noise
):
    x, y = (value.requires_grad_() for value in training_data(device))
    query = torch.tensor([[0.3], [0.65]], dtype=x.dtype, device=device, requires_grad=True)
    noise = torch.full_like(y, 0.03, requires_grad=True) if fixed_noise else None
    model = SingleTaskGP(x, y, noise, backend="torch")
    with torch.no_grad():
        model.posterior(query)
        x.add_(0.02)
        y.add_(torch.tensor([[0.01], [0.03], [-0.02]], device=device))
        if noise is not None:
            noise.mul_(1.1)
        warmed = model.posterior(query).mean
    fresh = SingleTaskGP(x, y, noise, backend="torch")
    shared = (x, y, query) + (() if noise is None else (noise,))
    actual = derivatives(weighted_loss(model, query), shared + tuple(model.parameters()))
    expected = derivatives(weighted_loss(fresh, query), shared + tuple(fresh.parameters()))
    for actual_order, expected_order in zip(actual, expected, strict=True):
        for actual_gradient, expected_gradient in zip(actual_order, expected_order, strict=True):
            assert torch.isfinite(actual_gradient).all()
            torch.testing.assert_close(actual_gradient, expected_gradient, atol=1e-9, rtol=1e-9)
    with torch.no_grad():
        torch.testing.assert_close(warmed, fresh.posterior(query).mean)


@pytest.mark.parametrize("device", DEVICES)
def test_dtype_and_device_move_invalidate_factor_and_preserve_query_graph(device):
    x, y = training_data()
    model = SingleTaskGP(x, y, backend="torch").requires_grad_(False)
    with patch.object(
        native, "tensor_exact_gp_factor", wraps=native.tensor_exact_gp_factor
    ) as factor:
        with torch.no_grad():
            model.posterior(x + 0.03)
            assert factor.call_count == 1
        model.to(device=device, dtype=torch.float32)
        query = (x + 0.03).to(device=device, dtype=torch.float32).requires_grad_()
        posterior = model.posterior(query)
        first = torch.autograd.grad(posterior.mean.sum(), query, create_graph=True)[0]
        assert torch.isfinite(torch.autograd.grad(first.sum(), query)[0]).all()
        assert posterior.dtype == torch.float32 and posterior.device == query.device
        assert factor.call_count == 2
        with torch.no_grad():
            cached = model.posterior(query).mean
            assert factor.call_count == 2
            model._prediction_cache = None
            torch.testing.assert_close(cached, model.posterior(query).mean)
            assert factor.call_count == 3

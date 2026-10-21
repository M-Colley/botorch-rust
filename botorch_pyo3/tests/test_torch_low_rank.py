"""Cached Cholesky utility values, layouts and full graphs on CPU/CUDA."""

from __future__ import annotations

from unittest.mock import patch

import pytest

torch = pytest.importorskip("torch")

from botorch_rs.exceptions.errors import BotorchError, NanError  # noqa: E402
from botorch_rs.posteriors._torch_backend import (  # noqa: E402
    TorchGPyTorchPosterior,
    independent_covariance,
)
from botorch_rs.posteriors.gpytorch import GPyTorchPosterior  # noqa: E402
from botorch_rs.utils.low_rank import extract_batch_covar, sample_cached_cholesky  # noqa: E402

DEVICES = ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]


def _data(device, dtype, batch, m, sample_shape=(3,)):
    generator = torch.Generator(device=device).manual_seed(17)
    mean = torch.randn(*batch, 4, m, generator=generator, device=device, dtype=dtype)
    a = torch.randn(*batch, m, 4, 4, generator=generator, device=device, dtype=dtype) * 0.2
    blocks = a @ a.transpose(-1, -2) + torch.eye(4, device=device, dtype=dtype)
    posterior = TorchGPyTorchPosterior(mean, independent_covariance(blocks))
    factor = torch.linalg.cholesky(blocks[..., :2, :2])
    if m == 1:
        factor = factor.squeeze(-3)
    normals = torch.randn(
        *sample_shape,
        *((1,) * len(batch)),
        4,
        m,
        generator=generator,
        device=device,
        dtype=dtype,
    )
    return posterior, factor, normals, blocks


def _reference(posterior, blocks):
    pytest.importorskip("gpytorch")
    pytest.importorskip("botorch")
    from botorch.posteriors.gpytorch import GPyTorchPosterior as ReferencePosterior
    from gpytorch.distributions import MultitaskMultivariateNormal, MultivariateNormal
    from linear_operator.operators import BlockDiagLinearOperator, DenseLinearOperator

    mean = posterior.mean
    distribution = (
        MultivariateNormal(mean.squeeze(-1), blocks.squeeze(-3))
        if mean.shape[-1] == 1
        else MultitaskMultivariateNormal(
            mean,
            BlockDiagLinearOperator(DenseLinearOperator(blocks), block_dim=-3),
            interleaved=False,
        )
    )
    return ReferencePosterior(distribution)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.double])
@pytest.mark.parametrize("batch", [(), (2,), (2, 3)])
@pytest.mark.parametrize("m", [1, 2])
def test_cached_draws_and_covariance_blocks_match_reference(device, dtype, batch, m):
    pytest.importorskip("botorch")
    from botorch.utils.low_rank import sample_cached_cholesky as reference_sample

    posterior, factor, normals, blocks = _data(device, dtype, batch, m)
    expected = reference_sample(_reference(posterior, blocks), factor, 2, normals, torch.Size([3]))
    actual = sample_cached_cholesky(posterior, factor, 2, normals, (3,))
    torch.testing.assert_close(actual, expected)
    assert actual.device == posterior.device and actual.dtype == dtype
    assert actual.shape == (3, *batch, 2, m) and actual.is_contiguous()
    if m > 1:
        torch.testing.assert_close(extract_batch_covar(posterior.distribution), blocks)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("m", [1, 2])
@pytest.mark.parametrize("supplied", [False, True])
def test_cached_draws_full_hessian_includes_factor_covariance_mean_and_normals(device, m, supplied):
    pytest.importorskip("botorch")
    from botorch.utils.low_rank import sample_cached_cholesky as reference_sample

    original, factor, normals, blocks = _data(device, torch.double, (2,), m)
    mean = original.mean.detach().requires_grad_()
    covariance_root = torch.linalg.cholesky(blocks).detach().requires_grad_()
    covariance = covariance_root @ covariance_root.transpose(-1, -2)
    factor = factor.detach().requires_grad_()
    normals = normals.detach().requires_grad_()
    posterior = TorchGPyTorchPosterior(mean, independent_covariance(covariance))
    reference = _reference(posterior, covariance)
    if supplied:
        posterior = GPyTorchPosterior(reference.distribution)
    actual = sample_cached_cholesky(posterior, factor, 2, normals, (3,))
    expected = reference_sample(reference, factor, 2, normals, torch.Size([3]))
    weights = torch.linspace(0.3, 1.1, actual.numel(), device=device).reshape_as(actual)
    inputs = (mean, covariance_root, factor, normals)
    gradients = [
        torch.autograd.grad((value.square() * weights).sum(), inputs, create_graph=True)
        for value in (actual, expected)
    ]
    directions = [torch.linspace(-0.4, 0.7, x.numel(), device=device).reshape_as(x) for x in inputs]
    hessians = [
        torch.autograd.grad(
            sum((g * direction).sum() for g, direction in zip(gs, directions, strict=True)),
            inputs,
            retain_graph=True,
        )
        for gs in gradients
    ]
    torch.testing.assert_close(actual, expected, atol=1e-11, rtol=1e-10)
    for actual_terms, expected_terms in ((gradients[0], gradients[1]), (hessians[0], hessians[1])):
        for a, e in zip(actual_terms, expected_terms, strict=True):
            torch.testing.assert_close(a, e, atol=2e-10, rtol=2e-9)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("sample_shape", [(), (2, 3)])
@pytest.mark.parametrize("m", [1, 2])
def test_general_sample_shapes_match_full_joint_draws_without_host_transfers(
    device, sample_shape, m
):
    posterior, factor, normals, _ = _data(device, torch.double, (2,), m, sample_shape)
    expected = posterior.rsample_from_base_samples(torch.Size(sample_shape), normals)[..., -2:, :]
    error = AssertionError("Tensor sampling must not transfer numerical values to host")
    with (
        patch.object(torch.Tensor, "cpu", side_effect=error),
        patch.object(torch.Tensor, "numpy", side_effect=error),
        patch.object(torch.Tensor, "tolist", side_effect=error),
        patch("botorch_rs.utils.low_rank.to_flat", side_effect=error),
    ):
        actual = sample_cached_cholesky(posterior, factor, 2, normals, sample_shape)
    torch.testing.assert_close(actual, expected, atol=1e-11, rtol=1e-10)
    assert actual.shape == (*sample_shape, 2, 2, m)


@pytest.mark.parametrize("device", DEVICES)
def test_correlated_outputs_invalid_inputs_and_nonfinite_samples_rejected(device):
    posterior, factor, normals, _ = _data(device, torch.double, (), 2)
    correlated = posterior.distribution.covariance_matrix + 0.01
    bad = TorchGPyTorchPosterior(posterior.mean, correlated)
    with pytest.raises(BotorchError, match="correlated"):
        sample_cached_cholesky(bad, factor, 2, normals, (3,))
    with pytest.raises(BotorchError, match="baseline_L"):
        sample_cached_cholesky(posterior, torch.eye(3, device=device), 2, normals, (3,))
    for q in (0, -1, 5):
        with pytest.raises(ValueError, match="q must"):
            sample_cached_cholesky(posterior, factor, q, normals, (3,))
    with pytest.raises(RuntimeError, match="base samples is invalid"):
        sample_cached_cholesky(posterior, factor, 2, normals[:2], (3,))
    with pytest.raises(NanError, match="nans|infs"):
        sample_cached_cholesky(posterior, factor, 2, torch.full_like(normals, float("inf")), (3,))


@pytest.mark.parametrize("device", DEVICES)
def test_zero_baseline_and_singular_conditional_use_adaptive_jitter(device):
    mean = torch.zeros(3, 1, dtype=torch.double, device=device, requires_grad=True)
    covariance = torch.ones(3, 3, dtype=torch.double, device=device, requires_grad=True)
    posterior = TorchGPyTorchPosterior(mean, covariance)
    normals = torch.tensor([[0.2, 0.4, -0.3]], dtype=mean.dtype, device=device)
    with pytest.warns(Warning, match="jitter"):
        result = sample_cached_cholesky(
            posterior,
            torch.empty(0, 0, dtype=mean.dtype, device=device),
            3,
            normals,
            (1,),
            max_tries=2,
        )
    expected = mean + (
        torch.linalg.cholesky(covariance + 1e-8 * torch.eye(3, device=device, dtype=mean.dtype))
        @ normals.T
    )
    torch.testing.assert_close(result[0], expected, atol=1e-12, rtol=1e-10)
    gradients = torch.autograd.grad(result.square().sum(), (mean, covariance), create_graph=True)
    assert all(torch.isfinite(gradient).all() for gradient in gradients)


@pytest.mark.parametrize("device", DEVICES)
def test_interleaved_supplied_independent_distribution_keeps_point_major_normals(device):
    pytest.importorskip("gpytorch")
    from gpytorch.distributions import MultitaskMultivariateNormal

    canonical, factor, normals, _ = _data(device, torch.double, (2,), 2)
    covariance = canonical.distribution.covariance_matrix.reshape(2, 2, 4, 2, 4)
    interleaved = covariance.transpose(-4, -3).transpose(-2, -1).reshape(2, 8, 8)
    posterior = GPyTorchPosterior(
        MultitaskMultivariateNormal(canonical.mean, interleaved, interleaved=True)
    )
    expected = canonical.rsample_from_base_samples((3,), normals)[..., -2:, :]
    actual = sample_cached_cholesky(posterior, factor, 2, normals, (3,))
    torch.testing.assert_close(actual, expected, atol=1e-11, rtol=1e-10)


@pytest.mark.parametrize("device", DEVICES)
def test_exact_gp_sampling_training_query_and_parameter_first_second_gradients(device):
    from botorch_rs.models import SingleTaskGP

    x = (
        torch.linspace(0.05, 0.9, 5, dtype=torch.double, device=device)
        .unsqueeze(-1)
        .requires_grad_()
    )
    y = (x.detach() * 4).sin().requires_grad_()
    query = torch.tensor(
        [[0.12], [0.32], [0.51], [0.81]], dtype=x.dtype, device=device, requires_grad=True
    )
    model = SingleTaskGP(x, y, backend="torch", outcome_transform=None)
    posterior = model.posterior(query)
    covariance = posterior.distribution.covariance_matrix
    baseline = torch.linalg.cholesky(covariance[..., :2, :2])
    normals = torch.tensor(
        [[0.2, -0.3, 0.5, 0.7], [-0.4, 0.6, 0.1, -0.2]], dtype=x.dtype, device=device
    )
    actual = sample_cached_cholesky(posterior, baseline, 2, normals, (2,))
    expected = posterior.rsample_from_base_samples((2,), normals)[..., -2:, :]
    inputs = (x, y, query, *model.parameters())
    gradients = [
        torch.autograd.grad(value.square().sum(), inputs, create_graph=True)
        for value in (actual, expected)
    ]
    hessians = [
        torch.autograd.grad(sum(g.square().sum() * 0.3 for g in gs), inputs, retain_graph=True)
        for gs in gradients
    ]
    torch.testing.assert_close(actual, expected, atol=1e-10, rtol=1e-9)
    for a_terms, e_terms in ((gradients[0], gradients[1]), (hessians[0], hessians[1])):
        for a, e in zip(a_terms, e_terms, strict=True):
            torch.testing.assert_close(a, e, atol=2e-7, rtol=2e-6)

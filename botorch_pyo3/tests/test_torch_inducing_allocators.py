"""Device pivot allocation and batched variational construction."""

# ruff: noqa: E402
from __future__ import annotations

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
gpytorch = pytest.importorskip("gpytorch")
pytest.importorskip("botorch")

from botorch.models.approximate_gp import SingleTaskVariationalGP as ReferenceGP
from botorch.models.utils import inducing_point_allocators as reference

from botorch_rs import _botorch_rs as _rs
from botorch_rs.kernels import RBFKernel
from botorch_rs.mlls import VariationalELBO
from botorch_rs.models.approximate_gp import SingleTaskVariationalGP
from botorch_rs.models.utils.inducing_point_allocators import (
    ExpectedImprovementQualityFunction,
    GreedyImprovementReduction,
    GreedyVarianceReduction,
    InducingPointAllocator,
    _pivoted_cholesky_init,
)

DEVICES = [
    "cpu",
    pytest.param(
        "cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
    ),
]


def _inputs(device, dtype):
    return torch.tensor(
        [[0.1, 0.6], [0.3, 0.2], [0.8, 0.4], [0.2, 0.9], [0.9, 0.8], [0.5, 0.3]],
        device=device,
        dtype=dtype,
    )


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("weighted", [False, True])
def test_device_pivots_match_reference_and_keep_selected_input_graph(
    device, dtype, weighted, monkeypatch
):
    x = _inputs(device, dtype).requires_grad_()
    kernel = gpytorch.kernels.RBFKernel(ard_num_dims=2).to(x)
    with torch.no_grad():
        covariance = kernel(x).to_dense()
    quality = torch.linspace(0.4, 1.7, len(x), device=device, dtype=dtype)
    if not weighted:
        quality.fill_(1)
    expected = reference._pivoted_cholesky_init(x, covariance, 4, quality)

    def forbidden(*args, **kwargs):
        raise AssertionError("Tensor allocation must not flatten to native CPU arrays.")

    monkeypatch.setattr(_rs, "pivoted_cholesky_init_py", forbidden)
    actual = _pivoted_cholesky_init(x, covariance, 4, quality)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    assert actual.device == x.device
    first = torch.autograd.grad(actual.square().sum(), x, create_graph=True)[0]
    second = torch.autograd.grad(first.sum(), x)[0]
    chosen = torch.isin(torch.arange(len(x), device=device), torch.cdist(x, actual).argmin(0))
    torch.testing.assert_close(first, 2 * x * chosen[:, None], atol=0, rtol=0)
    torch.testing.assert_close(second, 2 * chosen[:, None].expand_as(x).to(x), atol=0, rtol=0)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("maximize", [False, True])
def test_expected_improvement_quality_matches_reference_on_device(device, maximize):
    x = _inputs(device, torch.double)
    model = SimpleNamespace(
        num_outputs=1,
        posterior=lambda q: SimpleNamespace(
            mean=(q.sum(-1).sin()).unsqueeze(-1),
            variance=(0.1 + q.square().sum(-1)).unsqueeze(-1),
        ),
    )
    actual = ExpectedImprovementQualityFunction(model, maximize)(x)
    expected = reference.ExpectedImprovementQualityFunction(model, maximize)(x)
    torch.testing.assert_close(actual, expected, atol=2e-15, rtol=2e-14)
    assert actual.device == x.device and not actual.requires_grad
    kernel = gpytorch.kernels.RBFKernel(ard_num_dims=2).to(x)
    points = GreedyImprovementReduction(model, maximize).allocate_inducing_points(x, kernel, 3, ())
    expected_points = reference.GreedyImprovementReduction(
        model, maximize
    ).allocate_inducing_points(x, kernel, 3, ())
    torch.testing.assert_close(points, expected_points, atol=0, rtol=0)


class _WeightedAllocator(InducingPointAllocator):
    def _get_quality_function(self):
        return lambda x: 0.2 + x[:, 0]


class _ReferenceWeightedAllocator(reference.InducingPointAllocator):
    def _get_quality_function(self):
        return lambda x: 0.2 + x[:, 0]


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("custom", [False, True])
def test_batched_default_and_custom_allocation_svgp_elbo_and_reinitialization(device, custom):
    x = _inputs(device, torch.double)
    x = torch.stack((x, x.flip(0) * 0.8)).requires_grad_()
    y = x.sum(-1).sin().unsqueeze(-1)
    own_allocator = _WeightedAllocator() if custom else GreedyVarianceReduction()
    ref_allocator = _ReferenceWeightedAllocator() if custom else reference.GreedyVarianceReduction()
    actual = SingleTaskVariationalGP(
        x,
        y,
        covar_module=RBFKernel(ard_num_dims=2, batch_shape=(2,)),
        inducing_points=3,
        inducing_point_allocator=own_allocator,
        backend="torch",
    )
    expected = ReferenceGP(
        x,
        y,
        covar_module=gpytorch.kernels.RBFKernel(ard_num_dims=2, batch_shape=(2,)).to(x),
        inducing_points=3,
        inducing_point_allocator=ref_allocator,
    ).to(x)
    strategies = [model.model.variational_strategy for model in (actual, expected)]
    torch.testing.assert_close(
        strategies[0].inducing_points, strategies[1].inducing_points, atol=0, rtol=0
    )
    for model, strategy in zip((actual, expected), strategies, strict=True):
        model.likelihood.noise = 0.2
        with torch.no_grad():
            strategy.variational_params_initialized.fill_(1)
            strategy._variational_distribution.variational_mean.fill_(0.2)
            strategy._variational_distribution.chol_variational_covar.mul_(0.7)
    losses = [
        VariationalELBO(actual.likelihood, actual.model, num_data=6)(
            actual.model(x), y.squeeze(-1)
        ),
        gpytorch.mlls.VariationalELBO(expected.likelihood, expected.model, num_data=6)(
            expected.model(x), y.squeeze(-1)
        ),
    ]
    torch.testing.assert_close(losses[0], losses[1], atol=3e-8, rtol=3e-9)
    gradients = [torch.autograd.grad(loss.sum(), x, retain_graph=True)[0] for loss in losses]
    torch.testing.assert_close(*gradients, atol=3e-9, rtol=3e-8)
    with torch.no_grad():
        points = [model.init_inducing_points(x.flip(-2)) for model in (actual, expected)]
    torch.testing.assert_close(*points, atol=0, rtol=0)
    assert actual._inducing_point_allocator is own_allocator
    assert all(not bool(strategy.variational_params_initialized) for strategy in strategies)


@pytest.mark.parametrize("device", DEVICES)
def test_batched_allocation_uses_each_model_kernel(device):
    x = _inputs(device, torch.double).expand(2, -1, -1)
    kernel = gpytorch.kernels.RBFKernel(ard_num_dims=2, batch_shape=(2,)).to(x)
    kernel.lengthscale = x.new_tensor([[[0.1, 2.0]], [[2.0, 0.1]]])
    actual = GreedyVarianceReduction().allocate_inducing_points(x, kernel, 3, (2,))
    expected = []
    with torch.no_grad():
        covariances = kernel(x).to_dense()
        for row, covariance in zip(x, covariances, strict=True):
            expected.append(reference._pivoted_cholesky_init(row, covariance, 3, row.new_ones(6)))
    torch.testing.assert_close(actual, torch.stack(expected), atol=0, rtol=0)
    assert not torch.equal(actual[0], actual[1])
    # Pinned b0dd indexes kernel_element[0] for every model batch, so its
    # public allocator uses the first model's anisotropy for both rows.
    source = reference.GreedyVarianceReduction().allocate_inducing_points(x, kernel, 3, (2,))
    assert torch.equal(source[0], source[1])
    assert not torch.equal(actual[1], source[1])

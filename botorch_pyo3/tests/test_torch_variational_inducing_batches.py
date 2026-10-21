"""Supplied inducing batches follow the strategy's actual coordinate layout."""

# ruff: noqa: E402
from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")
gpytorch = pytest.importorskip("gpytorch")
pytest.importorskip("botorch")

from botorch.models import SingleTaskVariationalGP as Source
from test_torch_variational_shared_distribution import DenseKernel, DenseMean

from botorch_rs.models import SingleTaskVariationalGP

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
CONFIGURATIONS = [
    (1, (), (2,)),
    (1, (2,), (3, 1)),
    (2, (), (3, 1)),
    (2, (2,), (2,)),
    (2, (2,), (3, 1, 1)),
]


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("outputs,batch,point_batch", CONFIGURATIONS)
@pytest.mark.parametrize(
    "strategy_name",
    [
        "VariationalStrategy",
        "UnwhitenedVariationalStrategy",
        "BatchDecoupledVariationalStrategy",
        "CiqVariationalStrategy",
        "LargeBatchVariationalStrategy",
    ],
)
def test_inducing_batches_match_source_values_and_all_first_derivatives(
    device, outputs, batch, point_batch, strategy_name
):
    records, gradients = [], []
    for cls in (Source, SingleTaskVariationalGP):
        source = cls is Source
        x = torch.tensor(
            [[0.1, 0.2], [0.3, 0.7], [0.6, 0.4], [0.9, 0.8]],
            device=device,
            dtype=torch.double,
        ).expand(*batch, 4, 2)
        points = x.new_tensor([[0.12, 0.2], [0.44, 0.51], [0.84, 0.92]])
        offsets = (
            torch.arange(math.prod(point_batch), device=device, dtype=x.dtype).reshape(
                *point_batch, 1, 1
            )
            / 50
        )
        points = points.expand(*point_batch, 3, 2) + offsets
        augmented_batch = batch + ((outputs,) if outputs > 1 else ())
        model = cls(
            x,
            num_outputs=outputs,
            inducing_points=points,
            variational_strategy=getattr(gpytorch.variational, strategy_name),
            covar_module=DenseKernel(augmented_batch, x),
            mean_module=DenseMean(augmented_batch, x),
            **({} if source else {"backend": "torch"}),
        )
        strategy = model.model.variational_strategy
        if outputs > 1:
            strategy = strategy.base_variational_strategy
        expected_points = (
            torch.stack((points, points), dim=-3)
            if strategy_name == "BatchDecoupledVariationalStrategy"
            else points
        )
        torch.testing.assert_close(strategy.inducing_points, expected_points)
        with torch.no_grad():
            strategy.variational_params_initialized.fill_(1)
            strategy._variational_distribution.variational_mean.copy_(
                x.new_tensor([-0.2, 0.1, 0.3])
            )
            strategy._variational_distribution.chol_variational_covar.copy_(
                torch.eye(3, device=device) * 0.8
            )
        query = x.new_tensor([[0.31, 0.48], [0.68, 0.28]], requires_grad=True)
        posterior = model.posterior(query, observation_noise=True)
        covariance = posterior.distribution.covariance_matrix
        if outputs > 1 and getattr(posterior.distribution, "_interleaved", False):
            q, m = posterior.mean.shape[-2:]
            covariance = torch.einsum(
                "...itju->...tiuj", covariance.reshape(*covariance.shape[:-2], q, m, q, m)
            ).reshape(*covariance.shape[:-2], q * m, q * m)
        records.append((posterior.mean, posterior.variance, covariance))
        parameters = dict(model.named_parameters())
        leaves = (query, *(parameters[name] for name in sorted(parameters)))
        check_hvp = not source and strategy_name in (
            "VariationalStrategy",
            "LargeBatchVariationalStrategy",
        )
        loss = posterior.mean.sin().sum() + 0.03 * posterior.variance.square().sum()
        first = torch.autograd.grad(loss, leaves, create_graph=check_hvp)
        gradients.append(first)
        if check_hvp:
            test = query.unsqueeze(-3) if outputs > 1 else query
            kernel = model.model.covar_module
            inducing = strategy.inducing_points
            eye = torch.eye(3, dtype=query.dtype, device=query.device)
            root = torch.linalg.cholesky(kernel(inducing, inducing) + strategy.jitter_val * eye)
            interpolation = torch.linalg.solve_triangular(root, kernel(inducing, test), upper=False)
            distribution = strategy._variational_distribution
            factor = distribution.chol_variational_covar.tril()
            q_covariance = factor @ factor.mT
            mean = model.model.mean_module(test) + (
                interpolation.mT @ distribution.variational_mean.unsqueeze(-1)
            ).squeeze(-1)
            covariance = (
                kernel(test, test)
                + interpolation.mT @ (q_covariance - eye) @ interpolation
                + strategy.jitter_val * torch.eye(2, dtype=query.dtype, device=query.device)
            )
            variance = covariance.diagonal(dim1=-2, dim2=-1) + model.likelihood.noise_for(2)
            if outputs > 1:
                mean, variance = mean.transpose(-1, -2), variance.transpose(-1, -2)
            manual_loss = mean.sin().sum() + 0.03 * variance.square().sum()
            manual_first = torch.autograd.grad(manual_loss, leaves, create_graph=True)
            for actual, expected in zip(first, manual_first, strict=True):
                torch.testing.assert_close(actual, expected, atol=3e-8, rtol=3e-8)
            second = [
                torch.autograd.grad(sum(g.square().sum() / 2 for g in derivative), leaves)
                for derivative in (first, manual_first)
            ]
            for actual, expected in zip(*second, strict=True):
                torch.testing.assert_close(actual, expected, atol=5e-7, rtol=5e-7)
    tolerance = 3e-5 if strategy_name == "CiqVariationalStrategy" else 3e-8
    for a, b in zip(*records, strict=True):
        torch.testing.assert_close(a, b, atol=tolerance, rtol=tolerance)
    for a, b in zip(*gradients, strict=True):
        torch.testing.assert_close(a, b, atol=tolerance, rtol=tolerance)

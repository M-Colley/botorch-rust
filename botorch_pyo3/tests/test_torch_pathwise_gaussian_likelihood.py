"""Pathwise updates use the actual supplied full Gaussian observation law."""

# ruff: noqa: E402
from __future__ import annotations

import math
import warnings

import pytest

torch = pytest.importorskip("torch")
gpytorch = pytest.importorskip("gpytorch")

from test_torch_exact_gaussian_likelihood import Gaussian, _model, _noise, _prior

from botorch_rs.sampling.pathwise import gaussian_update

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


def _grad(value, leaves):
    gradients = torch.autograd.grad(value, leaves, create_graph=True, allow_unused=True)
    return tuple(
        leaf * 0 if gradient is None else gradient
        for leaf, gradient in zip(leaves, gradients, strict=True)
    )


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("outputs,batch", [(1, ()), (2, ()), (1, (2,)), (2, (2,))])
@pytest.mark.parametrize("shifted", [False, True])
def test_actual_full_noise_and_mean_shift_update_matches_dense_law_all_graphs(
    device, outputs, batch, shifted
):
    model, X, Y = _model(device, outputs, batch)
    if shifted:

        class ShiftedGaussian(Gaussian):
            def __init__(self, noise_covar):
                super().__init__(noise_covar)
                self.mean_shift = torch.nn.Parameter(X.new_tensor(0.07))

            def marginal(self, distribution, *params, **kwargs):
                observed = super().marginal(distribution, *params, **kwargs)
                points = params[0][0] if isinstance(params[0], (tuple, list)) else params[0]
                return gpytorch.distributions.MultivariateNormal(
                    observed.mean + self.mean_shift * (points[..., 0] + 0.1),
                    observed.lazy_covariance_matrix,
                )

        model.likelihood = ShiftedGaussian(model.likelihood.noise_covar)
    points, target, _ = model._training_values()
    shape = (2, *target.shape)
    prior_values = (
        torch.linspace(-0.3, 0.7, math.prod(shape), device=device, dtype=X.dtype)
        .reshape(shape)
        .requires_grad_()
    )
    normals = (
        torch.linspace(-0.6, 0.4, math.prod(shape), device=device, dtype=X.dtype)
        .reshape(shape)
        .requires_grad_()
    )
    query = X.new_tensor([[0.21, 0.51], [0.56, 0.73]], requires_grad=True)
    leaves = (query, X, Y, prior_values, normals, *model.parameters())
    for _ in range(2):
        actual = gaussian_update(model, prior_values, noise=normals)(query)
        points, target, _ = model._training_values()
        covariance = _noise(model, points)
        mean_shift = model.likelihood.mean_shift * (points[..., 0] + 0.1) if shifted else 0
        residual = (
            target
            - mean_shift
            - prior_values
            - (torch.linalg.cholesky(covariance) @ normals.unsqueeze(-1)).squeeze(-1)
        )
        weights = torch.linalg.solve(
            _prior(model, points, points) + covariance, residual.unsqueeze(-1)
        )
        expected = (_prior(model, query, points) @ weights).squeeze(-1)
        torch.testing.assert_close(actual, expected, rtol=3e-10, atol=3e-11)
        first, reference_first = (
            _grad(actual.square().sum(), leaves),
            _grad(expected.square().sum(), leaves),
        )
        second = _grad(sum(g.square().sum() for g in first), leaves)
        reference_second = _grad(sum(g.square().sum() for g in reference_first), leaves)
        for a, b in zip((*first, *second), (*reference_first, *reference_second), strict=True):
            assert a.device == X.device and bool(torch.isfinite(a).all())
            torch.testing.assert_close(a, b, rtol=3e-6, atol=3e-7)


@pytest.mark.parametrize("device", DEVICES)
def test_supplied_fixed_noise_unmatched_points_keep_exact_zero_noise_law(device):
    from botorch_rs.models import SingleTaskGP

    X = torch.linspace(0.1, 0.9, 8, dtype=torch.double, device=device).reshape(4, 2)
    Y = X[:, :1].sin()
    likelihood = gpytorch.likelihoods.FixedNoiseGaussianLikelihood(noise=X.new_full((4,), 0.04)).to(
        X
    )
    model = SingleTaskGP(X, Y, likelihood=likelihood, outcome_transform=None, backend="torch")
    points, target = X[[0, 2, 3]], Y[[0, 2, 3], 0]
    prior = torch.zeros(2, 3, dtype=X.dtype, device=device)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        actual = gaussian_update(model, prior, points=points, target_values=target)(X[:2])
    kernel = model.covar_module(points, points)
    weights = torch.linalg.solve(kernel, (target - prior).unsqueeze(-1))
    expected = (model.covar_module(X[:2], points) @ weights).squeeze(-1)
    torch.testing.assert_close(actual, expected, rtol=3e-10, atol=3e-11)

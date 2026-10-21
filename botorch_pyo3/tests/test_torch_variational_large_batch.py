"""Grouped inducing systems retain the source law and every Torch derivative."""

# ruff: noqa: E402
from __future__ import annotations

import copy
import functools
import math
import pickle
from unittest.mock import patch

import pytest

torch = pytest.importorskip("torch")

from test_torch_variational_shared_distribution import DenseKernel, DenseMean, loss_terms

from botorch_rs.mlls import VariationalELBO
from botorch_rs.models import SingleTaskVariationalGP
from botorch_rs.variational import CholeskyVariationalDistribution, LargeBatchVariationalStrategy

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
CONFIGURATIONS = [(1, (), 2), (1, (2,), 5), (2, (), 2), (2, (2,), 5)]


def make_model(device, outputs, batch, *, source=False, dtype=None, external=False):
    dtype = torch.double if dtype is None else dtype
    x = torch.tensor(
        [[0.1, 0.2], [0.3, 0.7], [0.6, 0.4], [0.9, 0.8]], device=device, dtype=dtype
    ).expand(*batch, 4, 2)
    aug = batch + ((outputs,) if outputs > 1 else ())
    cls, strategy_cls, distribution_cls = (
        SingleTaskVariationalGP,
        LargeBatchVariationalStrategy,
        CholeskyVariationalDistribution,
    )
    if source or external:
        gpytorch = pytest.importorskip("gpytorch")
        strategy_cls = gpytorch.variational.LargeBatchVariationalStrategy
        distribution_cls = gpytorch.variational.CholeskyVariationalDistribution
    if source:
        pytest.importorskip("botorch")
        from botorch.models import SingleTaskVariationalGP as cls
    model = cls(
        x,
        num_outputs=outputs,
        inducing_points=x.new_tensor([[0.12, 0.2], [0.44, 0.51], [0.84, 0.92]]),
        variational_strategy=functools.partial(strategy_cls, jitter_val=2e-4),
        variational_distribution=distribution_cls(3, batch_shape=torch.Size(aug), mean_init_std=0),
        covar_module=DenseKernel(aug, x),
        mean_module=DenseMean(aug, x),
        **({} if source else {"backend": "torch"}),
    ).to(x)
    strategy = model.model.variational_strategy
    if outputs > 1:
        strategy = strategy.base_variational_strategy
    with torch.no_grad():
        strategy.variational_params_initialized.fill_(1)
        strategy._variational_distribution.variational_mean.copy_(x.new_tensor([-0.2, 0.15, 0.3]))
        strategy._variational_distribution.chol_variational_covar.copy_(
            x.new_tensor([[0.8, 0, 0], [0.1, 0.9, 0], [-0.05, 0.12, 0.7]])
        )
    return model, strategy


def data(model, n):
    like = next(model.parameters())
    query = torch.stack(
        (
            torch.linspace(0.17, 0.81, n, device=like.device, dtype=like.dtype),
            torch.linspace(0.67, 0.25, n, device=like.device, dtype=like.dtype),
        ),
        -1,
    ).requires_grad_()
    target = query.new_full((*model.batch_shape, n, model.num_outputs), 0.4).requires_grad_()
    return query, target


def dense_law(model, strategy, query, target, training):
    multi = model.num_outputs > 1
    test = query.unsqueeze(-3) if multi else query
    points = strategy.inducing_points.unsqueeze(-3) if multi else strategy.inducing_points
    kernel = model.model.covar_module
    eye = torch.eye(3, device=query.device, dtype=query.dtype)
    root = torch.linalg.cholesky(kernel(points, points) + strategy.jitter_val * eye)
    interpolation = torch.linalg.solve_triangular(root, kernel(points, test), upper=False)
    distribution = strategy._variational_distribution
    mean_q = distribution.variational_mean
    factor = distribution.chol_variational_covar.tril()
    covariance_q = factor @ factor.mT
    mean = model.model.mean_module(test) + (interpolation.mT @ mean_q.unsqueeze(-1)).squeeze(-1)
    correction = interpolation.mT @ (covariance_q - eye) @ interpolation
    if training and not multi:
        correction = torch.diag_embed(correction.diagonal(dim1=-2, dim2=-1))
    covariance = (
        kernel(test, test)
        + correction
        + strategy.jitter_val * torch.eye(query.shape[-2], dtype=query.dtype, device=query.device)
    )
    target = target.transpose(-1, -2) if multi else target.squeeze(-1)
    noise = model.likelihood.noise_for(query.shape[-2])
    expected_log_prob = -0.5 * (
        ((target - mean).square() + covariance.diagonal(dim1=-2, dim2=-1)) / noise
        + noise.log()
        + math.log(2 * math.pi)
    ).mean(-1)
    kl = (
        mean_q.square().sum(-1)
        + covariance_q.diagonal(dim1=-2, dim2=-1).sum(-1)
        - 3
        - 2 * factor.diagonal(dim1=-2, dim2=-1).log().sum(-1)
    ) / 2
    if multi:
        expected_log_prob, kl = expected_log_prob.sum(-1), kl.sum(-1)
    elbo = expected_log_prob - 0.7 * kl / 11 + model._log_prior() / 11
    return mean, covariance, elbo


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("outputs,batch,n", CONFIGURATIONS)
@pytest.mark.parametrize("training", [False, True])
@pytest.mark.parametrize("external", [False, True])
def test_grouped_large_batch_matches_independent_dense_all_leaf_hvps(
    device, outputs, batch, n, training, external
):
    model, strategy = make_model(device, outputs, batch, external=external)
    assert not model._uses_caller_strategy and strategy.grouped_middle
    model.train(training)
    query, target = data(model, n)
    output = model.model(query)
    elbo = VariationalELBO(model.likelihood, model.model, 11, beta=0.7)(
        output, target.squeeze(-1) if outputs == 1 else target
    )
    actual = (output._mean, output._blocks(), elbo)
    expected = dense_law(model, strategy, query, target, training)
    for a, b in zip(actual, expected, strict=True):
        torch.testing.assert_close(a, b, atol=2e-9, rtol=2e-9)
    leaves = (*model.parameters(), query, target)
    gradients = [
        torch.autograd.grad(loss_terms(*terms), leaves, create_graph=True)
        for terms in (actual, expected)
    ]
    for a, b in zip(*gradients, strict=True):
        torch.testing.assert_close(a, b, atol=3e-8, rtol=3e-8)
    hvps = [torch.autograd.grad(sum(g.square().sum() / 2 for g in gs), leaves) for gs in gradients]
    for a, b in zip(*hvps, strict=True):
        torch.testing.assert_close(a, b, atol=2e-7, rtol=2e-7)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("outputs,batch,n", CONFIGURATIONS)
@pytest.mark.parametrize("training", [False, True])
@pytest.mark.parametrize("dtype", [torch.double, torch.float32])
def test_actual_gpytorch_large_batch_values_and_first_derivatives(
    device, outputs, batch, n, training, dtype
):
    pytest.importorskip("botorch")
    from gpytorch.mlls import VariationalELBO as SourceELBO

    values, gradients = [], []
    for source in (False, True):
        model, _ = make_model(device, outputs, batch, source=source, dtype=dtype, external=True)
        model.train(training)
        query, target = data(model, n)
        inputs = query.unsqueeze(-3) if source and outputs > 1 and query.ndim > 2 else query
        output = model.model(inputs)
        elbo = (SourceELBO if source else VariationalELBO)(
            model.likelihood, model.model, 11, beta=0.7
        )(output, target.squeeze(-1) if outputs == 1 else target)
        values.append((output.mean, output.variance, output.covariance_matrix, elbo))
        parameters = dict(model.named_parameters())
        gradients.append(
            torch.autograd.grad(
                output.mean.sin().sum()
                + 0.03 * output.covariance_matrix.square().sum()
                + elbo.sum(),
                (query, target, *(parameters[name] for name in sorted(parameters))),
            )
        )
    tolerance = 3e-4 if dtype == torch.float32 else 4e-8
    for a, b in zip(*values, strict=True):
        torch.testing.assert_close(a, b, atol=tolerance, rtol=tolerance)
    for a, b in zip(*gradients, strict=True):
        torch.testing.assert_close(a, b, atol=tolerance, rtol=tolerance)


@pytest.mark.parametrize("device", DEVICES)
def test_large_batch_groups_only_inducing_sized_triangular_systems(device):
    model, _ = make_model(device, 1, ())
    query, _ = data(model, 13)
    systems = []
    solve = torch.linalg.solve_triangular

    def recording_solve(matrix, rhs, **kwargs):
        systems.append((matrix.shape[-2:], rhs.shape[-2:], matrix.dtype))
        return solve(matrix, rhs, **kwargs)

    with patch.object(torch.linalg, "solve_triangular", side_effect=recording_solve):
        posterior = model.posterior(query)
        posterior.mean.sum().backward()
    assert len(systems) == 3
    assert all(matrix == (3, 3) and rhs in ((3, 1), (3, 3)) for matrix, rhs, _ in systems)
    assert all(dtype == torch.double for _, _, dtype in systems)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("kind", ["deepcopy", "pickle", "state"])
def test_grouped_large_batch_snapshots_and_serialized_owner_links(device, kind):
    model, strategy = make_model(device, 2, ())
    query, _ = data(model, 5)
    output = model.model(query)
    expected_mean, expected_cov = output.mean.clone(), output.covariance_matrix.clone()
    if kind == "deepcopy":
        clone = copy.deepcopy(model)
    elif kind == "pickle":
        clone = pickle.loads(pickle.dumps(model))
    else:
        clone, _ = make_model(device, 2, ())
        clone.load_state_dict(model.state_dict())
    assert clone._strategy._owner_ref() is clone and clone.model._outer_ref() is clone
    torch.testing.assert_close(clone.model(query).mean, expected_mean)
    torch.testing.assert_close(clone.model(query).covariance_matrix, expected_cov)
    with torch.no_grad():
        strategy._variational_distribution.variational_mean.add_(0.2)
        model.model.covar_module.raw_length.add_(0.1)
    assert not torch.allclose(model.model(query).mean, expected_mean)
    torch.testing.assert_close(output.mean, expected_mean)
    torch.testing.assert_close(output.covariance_matrix, expected_cov)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("n", [2, 5])
def test_explicit_full_covariance_training_matches_source(device, n):
    records = []
    for source in (False, True):
        model, _ = make_model(device, 1, (), source=source)
        query, _ = data(model, n)
        model.train()
        diagonal, full = model.model(query), model.model(query, diag=False)
        torch.testing.assert_close(diagonal.mean, full.mean)
        torch.testing.assert_close(diagonal.variance, full.variance)
        assert not torch.allclose(diagonal.covariance_matrix, full.covariance_matrix)
        records.append(full.covariance_matrix)
    torch.testing.assert_close(*records, atol=2e-9, rtol=2e-9)


@pytest.mark.parametrize("device", DEVICES)
def test_grouped_prediction_and_higher_backward_stay_on_device(device):
    model, _ = make_model(device, 1, ())
    query, _ = data(model, 5)
    with (
        patch.object(torch.Tensor, "cpu", side_effect=AssertionError("Tensor.cpu")),
        patch.object(torch.Tensor, "numpy", side_effect=AssertionError("Tensor.numpy")),
        patch.object(torch.Tensor, "tolist", side_effect=AssertionError("Tensor.tolist")),
        patch("botorch_rs._arrays.to_flat", side_effect=AssertionError("to_flat")),
    ):
        posterior = model.posterior(query, observation_noise=True)
        loss = posterior.mean.sin().sum() + posterior.variance.square().sum()
        leaves = (query, *model.parameters())
        first = torch.autograd.grad(loss, leaves, create_graph=True)
        second = torch.autograd.grad(sum(g.square().sum() / 2 for g in first), leaves)
    assert all(
        value.device.type == device and torch.isfinite(value).all() for value in (*first, *second)
    )

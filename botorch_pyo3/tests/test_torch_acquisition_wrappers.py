"""Feature wrappers retain fixed-value, projection, and model derivative graphs."""

from unittest.mock import patch

import pytest
import torch

from botorch_rs.acquisition.analytic import PosteriorMean
from botorch_rs.acquisition.fixed_feature import FixedFeatureAcquisitionFunction
from botorch_rs.acquisition.knowledge_gradient import ProjectedAcquisitionFunction
from botorch_rs.acquisition.monte_carlo import qSimpleRegret
from botorch_rs.models import SingleTaskGP
from botorch_rs.sampling.normal import SobolQMCNormalSampler

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("shape", [(1,), (2, 1), (3, 2, 1)])
def test_fixed_features_broadcast_values_and_full_graphs(device, shape):
    X = torch.tensor(
        [[0.1, 0.2], [0.3, 0.5], [0.7, 0.9], [0.9, 0.1]], dtype=torch.double, device=device
    )
    model = SingleTaskGP(X, (4 * X[:, :1]).sin() + X[:, 1:].square(), backend="torch")
    base = qSimpleRegret(model, sampler=SobolQMCNormalSampler((5,), seed=17))
    values = torch.full(shape, 0.4, dtype=X.dtype, device=X.device, requires_grad=True)
    query = (
        torch.linspace(0.2, 0.8, 6, dtype=X.dtype, device=X.device)
        .reshape(3, 2, 1)
        .requires_grad_()
    )
    with (
        patch.object(torch.Tensor, "cpu", side_effect=AssertionError("host transfer")),
        patch.object(torch.Tensor, "tolist", side_effect=AssertionError("host transfer")),
        patch.object(torch.Tensor, "numpy", side_effect=AssertionError("host transfer")),
    ):
        wrapped = FixedFeatureAcquisitionFunction(
            acq_function=base, d=2, columns=[1], values=values
        )
        actual = wrapped(query)
        full = torch.cat((query, values.expand_as(query)), dim=-1)
        torch.testing.assert_close(actual, base(full))
        leaves = (query, values, *model.parameters())
        gradient = torch.autograd.grad(actual.square().sum(), leaves, create_graph=True)
        second = torch.autograd.grad(sum(g.square().sum() for g in gradient), leaves)
    assert all(torch.isfinite(g).all() for g in second)
    assert isinstance(wrapped, FixedFeatureAcquisitionFunction)
    assert isinstance(wrapped, torch.nn.Module)


@pytest.mark.parametrize("device", DEVICES)
def test_fixed_feature_sequences_and_pending_points(device):
    X = torch.tensor(
        [[0.1, 0.2, 0.4], [0.3, 0.5, 0.7], [0.7, 0.9, 0.8], [0.9, 0.1, 0.2]],
        dtype=torch.double,
        device=device,
    )
    model = SingleTaskGP(X, X.sum(-1, keepdim=True).sin(), backend="torch")
    base = qSimpleRegret(model, sampler=SobolQMCNormalSampler((4,), seed=11))
    value = X.new_tensor(0.4, requires_grad=True)
    wrapped = FixedFeatureAcquisitionFunction(base, 3, [2, 0], [value, 0.6])
    query = X.new_tensor([[0.21], [0.64]], requires_grad=True)
    full = wrapped._construct_X_full(query)
    torch.testing.assert_close(full[:, 0], value.expand(2))
    torch.testing.assert_close(full[:, 2], full.new_full((2,), 0.6))
    wrapped.set_X_pending(query[:1])
    torch.testing.assert_close(base.X_pending, full[:1])
    assert not base.X_pending.requires_grad
    wrapped.set_X_pending(None)
    assert base.X_pending is None
    loss = wrapped(query).square().sum()
    assert torch.isfinite(torch.autograd.grad(loss, value)[0])


@pytest.mark.parametrize("device", DEVICES)
def test_projected_acquisition_keeps_project_parameter_and_model_graphs(device):
    X = torch.tensor([[0.1], [0.3], [0.7], [0.9]], dtype=torch.double, device=device)
    model = SingleTaskGP(X, (4 * X).sin(), backend="torch")
    base = PosteriorMean(model)
    scale = X.new_tensor(0.8, requires_grad=True)
    wrapped = ProjectedAcquisitionFunction(base_value_function=base, project=lambda Z: scale * Z)
    query = X.new_tensor([[0.24]], requires_grad=True)
    actual = wrapped(query)
    torch.testing.assert_close(actual, base(scale * query))
    leaves = (query, scale, *model.parameters())
    gradient = torch.autograd.grad(actual.square().sum(), leaves, create_graph=True)
    second = torch.autograd.grad(sum(g.square().sum() for g in gradient), leaves)
    assert all(torch.isfinite(g).all() for g in second)
    value, derivative = wrapped.value_and_grad(query)
    assert derivative.requires_grad
    torch.testing.assert_close(value, wrapped(query))

"""Live objective weights must remain differentiable after construction."""

import pytest
import torch

from botorch_rs.acquisition.objective import LinearMCObjective, ScalarizedPosteriorTransform
from botorch_rs.models import SingleTaskGP

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


@pytest.mark.parametrize("device", DEVICES)
def test_linear_objective_uses_live_weight_values_and_higher_derivatives(device):
    samples = torch.randn(3, 2, 2, dtype=torch.double, device=device, requires_grad=True)
    weights = torch.tensor([0.4, -0.7], dtype=torch.double, device=device, requires_grad=True)
    objective = LinearMCObjective(weights)
    with torch.no_grad():
        weights.add_(0.2)
    actual = objective(samples)
    torch.testing.assert_close(actual, samples @ weights)
    gradients = torch.autograd.grad(actual.square().sum(), (samples, weights), create_graph=True)
    second = torch.autograd.grad(sum(g.square().sum() for g in gradients), (samples, weights))
    assert all(torch.isfinite(g).all() for g in second)
    assert torch.autograd.gradcheck(lambda a, b: LinearMCObjective(b)(a), (samples, weights))
    assert torch.autograd.gradgradcheck(lambda a, b: LinearMCObjective(b)(a), (samples, weights))


@pytest.mark.parametrize("device", DEVICES)
def test_posterior_scalarization_weights_and_offset_keep_their_graphs(device):
    X = torch.tensor([[0.05], [0.31], [0.63], [0.91]], dtype=torch.double, device=device)
    Y = torch.cat(((5 * X).sin(), (3 * X).cos()), dim=-1)
    model = SingleTaskGP(X, Y, backend="torch")
    query = X.new_tensor([[0.21], [0.76]], requires_grad=True)
    weights = X.new_tensor([0.6, -0.3], requires_grad=True)
    offset = X.new_tensor(0.1, requires_grad=True)
    transform = ScalarizedPosteriorTransform(weights, offset)
    posterior = model.posterior(query, posterior_transform=transform)
    raw = model.posterior(query)
    torch.testing.assert_close(posterior.mean.squeeze(-1), raw.mean @ weights + offset)
    loss = posterior.mean.square().sum() + posterior.variance.sum()
    leaves = (weights, offset, query, *model.parameters())
    gradient = torch.autograd.grad(loss, leaves, create_graph=True)
    second = torch.autograd.grad(sum(g.square().sum() for g in gradient), leaves)
    assert all(torch.isfinite(g).all() for g in second)

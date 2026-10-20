# ruff: noqa: E402
"""Hierarchical Tensor gates, exact inference and higher-order graphs."""

import pytest

pytest.importorskip("torch")


import torch

from botorch_rs.mlls import ExactMarginalLogLikelihood
from botorch_rs.models.gp_regression import SingleTaskGP
from botorch_rs.models.hierarchical.conditional_kernel_gp import (
    HierarchicalConditionalKernel,
    HierarchicalConditionalKernelGP,
    HierarchicalConditionalKernelMultiTaskGP,
)
from botorch_rs.models.transforms.input import Normalize
from botorch_rs.sampling.normal import IIDNormalSampler

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


@pytest.mark.parametrize("device", DEVICES)
def test_public_selection_kernel_conversion_and_no_host_algebra(device, monkeypatch):
    X = torch.tensor([[0.0, 0.2], [1.0, 0.7], [0.0, 0.5]], dtype=torch.double, device=device)
    Y = X[:, 1:].sin().requires_grad_()
    source = HierarchicalConditionalKernel(2, {0: {0.0: [1]}}, use_saas_prior=False)
    source.kernels[0].outputscale = 3.5
    ordinary = SingleTaskGP(X, Y, covar_module=source, outcome_transform=None, backend="torch")
    model = HierarchicalConditionalKernelGP(
        X, Y, {0: {0.0: [1]}}, use_saas_prior=True, outcome_transform=None
    )
    assert model._tensor_backend == "torch"
    torch.testing.assert_close(
        ordinary.covar_module.kernels[0].outputscale,
        X.new_tensor(3.5),
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("Tensor inference crossed a host-array boundary")

    with monkeypatch.context() as patch:
        patch.setattr(torch.Tensor, "cpu", forbidden)
        patch.setattr(torch.Tensor, "numpy", forbidden)
        patch.setattr(torch.Tensor, "tolist", forbidden)
        query = X.new_tensor([[0.0, 0.35], [1.0, 0.6]], requires_grad=True)
        posterior = model.posterior(query)
        loss = (
            posterior.mean.sum()
            + posterior.variance.sum()
            + ExactMarginalLogLikelihood(model.likelihood, model)()
        )
        inputs = (Y, query, *model.parameters())
        first = torch.autograd.grad(loss, inputs, create_graph=True)
        second = torch.autograd.grad(
            sum(g.square().sum() for g in first), inputs, allow_unused=True
        )
        assert all(g is None or torch.isfinite(g).all() for g in (*first, *second))


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("saas", [False, True])
def test_standalone_kernel_values_parameters_and_full_graph(device, saas):
    from botorch.models.hierarchical.conditional_kernel_gp import (
        HierarchicalConditionalKernel as Reference,
    )

    dependencies = {0: {0.0: [1], 1.0: [2]}}
    torch.manual_seed(17)
    actual = (
        HierarchicalConditionalKernel(3, dependencies, use_saas_prior=saas, backend="torch")
        .double()
        .to(device)
    )
    torch.manual_seed(17)
    expected = Reference(3, dependencies, use_saas_prior=saas).double().to(device)
    X = torch.tensor(
        [[0.0, 0.2, 0.8], [1.0, 0.5, 0.6], [0.0, 0.7, 0.1]],
        dtype=torch.double,
        device=device,
        requires_grad=True,
    )
    Y = X.detach().new_tensor([[1.0, 0.3, 0.9], [0.0, 0.8, 0.4]])
    value, reference = actual(X, Y), expected(X, Y).to_dense()
    torch.testing.assert_close(value, reference)
    gradient = torch.autograd.grad(value.sum(), X, create_graph=True)[0]
    torch.testing.assert_close(gradient, torch.autograd.grad(reference.sum(), X)[0])
    assert torch.isfinite(torch.autograd.grad(gradient.square().sum(), X)[0]).all()
    for a, b in zip(actual.parameters(), expected.parameters(), strict=True):
        torch.testing.assert_close(a, b)


@pytest.mark.parametrize("device", DEVICES)
def test_normalized_exact_model_data_and_parameter_graphs(device):
    X = torch.tensor(
        [[2.0, 0.2], [4.0, 0.8], [2.0, 0.5], [4.0, 0.4]], dtype=torch.double, device=device
    )
    Y = torch.tensor(
        [[0.3], [0.8], [-0.2], [0.1]], dtype=X.dtype, device=device, requires_grad=True
    )
    model = HierarchicalConditionalKernelGP(
        X,
        Y,
        {0: {2.0: [1]}},
        input_transform=Normalize(d=2),
        use_saas_prior=False,
        outcome_transform=None,
        backend="torch",
    )
    query = X.new_tensor([[2.0, 0.35], [4.0, 0.6]], requires_grad=True)
    posterior = model.posterior(query)
    loss = (
        posterior.mean.sum()
        + posterior.variance.sum()
        + ExactMarginalLogLikelihood(model.likelihood, model)()
    )
    parameters = (Y, query, *model.parameters())
    first = torch.autograd.grad(loss, parameters, create_graph=True)
    second = torch.autograd.grad(
        sum(g.square().sum() for g in first), parameters, allow_unused=True
    )
    assert all(g is None or torch.isfinite(g).all() for g in (*first, *second))


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("outputs", [1, 2])
def test_batched_transformed_gates_and_independent_outputs(device, outputs):
    X = torch.tensor(
        [[[2.0, 0.2], [4.0, 0.8], [2.0, 0.5]], [[5.0, 0.3], [9.0, 0.9], [5.0, 0.6]]],
        dtype=torch.double,
        device=device,
    )
    # The raw category has different fitted coordinates in the two batches;
    # the second batch contains no active child rows.
    dependencies = {0: {2.0: [1]}}
    Y = X[..., 1:].sin().expand(2, 3, outputs).clone()
    model = HierarchicalConditionalKernelGP(
        X,
        Y,
        dependencies,
        train_Yvar=torch.full_like(Y, 0.03),
        input_transform=Normalize(d=2, batch_shape=torch.Size([2])),
        use_saas_prior=False,
        outcome_transform=None,
        backend="torch",
    )
    query = X[:, :2].clone().requires_grad_()
    result = model.posterior(query)
    assert result.mean.shape == (2, 2, outputs)
    for batch in range(2):
        separate = HierarchicalConditionalKernelGP(
            X[batch],
            Y[batch],
            dependencies,
            train_Yvar=torch.full_like(Y[batch], 0.03),
            input_transform=Normalize(d=2),
            use_saas_prior=False,
            outcome_transform=None,
            backend="torch",
        )
        torch.testing.assert_close(result.mean[batch], separate.posterior(query[batch]).mean)
        torch.testing.assert_close(
            result.variance[batch], separate.posterior(query[batch]).variance
        )
    first = torch.autograd.grad(
        result.mean.sum() + result.variance.sum(), query, create_graph=True
    )[0]
    assert torch.isfinite(torch.autograd.grad(first.square().sum(), query)[0]).all()


@pytest.mark.parametrize("device", DEVICES)
def test_continuous_data_and_query_gradgradcheck(device):
    coordinates = torch.tensor(
        [0.2, 0.55, 0.8], dtype=torch.double, device=device, requires_grad=True
    )
    Y = coordinates.detach().sin().unsqueeze(-1).requires_grad_()
    query = coordinates.detach().new_tensor([0.35, 0.7], requires_grad=True)

    def moments(values, targets, test):
        X = torch.stack((torch.zeros_like(values), values), -1)
        model = HierarchicalConditionalKernelGP(
            X,
            targets,
            {0: {0.0: [1]}},
            train_Yvar=torch.full_like(targets, 0.04),
            use_saas_prior=False,
            outcome_transform=None,
            backend="torch",
        )
        posterior = model.posterior(torch.stack((torch.zeros_like(test), test), -1))
        return torch.cat(
            (posterior.mean.flatten(), posterior.distribution.covariance_matrix.flatten())
        )

    assert torch.autograd.gradcheck(moments, (coordinates, Y, query), fast_mode=True)
    assert torch.autograd.gradgradcheck(moments, (coordinates, Y, query), fast_mode=True)


@pytest.mark.parametrize("device", DEVICES)
def test_conditioning_fantasies_and_module_migration(device):
    X = torch.tensor([[2.0, 0.2], [4.0, 0.8], [2.0, 0.55]], dtype=torch.double, device=device)
    Y = X[:, 1:].sin()
    model = HierarchicalConditionalKernelGP(
        X,
        Y,
        {0: {2.0: [1]}},
        use_saas_prior=False,
        input_transform=Normalize(d=2),
        outcome_transform=None,
        backend="torch",
    )
    observed = X.new_tensor([[2.0, 0.35]])
    query = X.new_tensor([[2.0, 0.7], [4.0, 0.5]])
    joint = model.posterior(torch.cat((observed, query)))
    mean, covariance = joint.mean.squeeze(-1), joint.distribution.covariance_matrix
    noise = model.likelihood.noise.flatten()[0]
    target = Y.new_tensor([[0.25]], requires_grad=True)
    gain = covariance[1:, :1] / (covariance[:1, :1] + noise)
    expected_mean = mean[1:] + (gain @ (target.flatten() - mean[:1]))
    expected_covariance = covariance[1:, 1:] - gain @ covariance[:1, 1:]
    conditioned = model.condition_on_observations(observed, target)
    actual = conditioned.posterior(query)
    torch.testing.assert_close(actual.mean.squeeze(-1), expected_mean)
    torch.testing.assert_close(actual.distribution.covariance_matrix, expected_covariance)
    assert conditioned.covar_module.gate_input_transform is conditioned.input_transform
    fantasy = model.fantasize(
        observed, IIDNormalSampler(torch.Size([2, 3]), seed=29), propagate_grads=True
    )
    assert fantasy.posterior(query).mean.shape == (2, 3, 2, 1)
    assert torch.isfinite(torch.autograd.grad(actual.mean.sum(), target)[0]).all()
    moved = model.float().cpu()
    assert moved.covar_module.gate_input_transform is moved.input_transform
    assert moved.posterior(query.float().cpu()).mean.dtype == torch.float32
    for targets in model.covar_module.ancestor_values:
        if targets.numel():
            torch.testing.assert_close(targets, targets.new_zeros(targets.shape))


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("task_feature", [0, 1, 2])
def test_multitask_hierarchy_maps_task_columns(device, task_feature):
    basic = torch.tensor(
        [[0.1, 2.0], [0.5, 4.0], [0.4, 2.0], [0.8, 4.0]], dtype=torch.double, device=device
    )
    tasks = basic.new_tensor([[0.0], [1.0], [0.0], [1.0]])
    X = torch.cat((basic[:, :task_feature], tasks, basic[:, task_feature:]), -1)
    Y = basic[:, :1].sin()
    model = HierarchicalConditionalKernelMultiTaskGP(
        X,
        Y,
        task_feature,
        {1: {2.0: [0]}},
        use_saas_prior=False,
        input_transform=Normalize(d=3, indices=[i for i in range(3) if i != task_feature]),
        outcome_transform=None,
        backend="torch",
    )
    posterior = model.posterior(basic[:2].clone().requires_grad_())
    assert posterior.mean.shape == (2, 2)
    first = torch.autograd.grad(
        posterior.mean.sum() + posterior.variance.sum(),
        tuple(model.parameters()),
        create_graph=True,
    )
    second = torch.autograd.grad(
        sum(g.square().sum() for g in first), tuple(model.parameters()), allow_unused=True
    )
    assert all(g is None or torch.isfinite(g).all() for g in (*first, *second))

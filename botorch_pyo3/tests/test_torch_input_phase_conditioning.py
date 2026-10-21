"""Conditioning stores the coordinates selected by its actual transform phase."""

# ruff: noqa: E402
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from test_torch_input_model_conditioning import MeanSampler
from test_torch_input_models import DEVICES, _data, _derivatives

from botorch_rs.kernels import RBFKernel
from botorch_rs.models import MultiTaskGP, SingleTaskGP
from botorch_rs.models.transforms.input import InputPerturbation, Warp


def _kernel(model, left, right):
    delta = (left.unsqueeze(-2) - right.unsqueeze(-3)) / model.covar_module.lengthscale.unsqueeze(
        -2
    )
    return (-0.5 * delta.square().sum(-1)).exp()


def _sample_mean(model, train, targets, query, propagate):
    matrix = _kernel(model, train, train) + torch.diag_embed(model._train_Yvar.squeeze(-1))
    constant = model.mean_module.constant
    alpha = torch.linalg.solve(matrix, targets - constant)
    if not propagate:
        alpha = alpha.detach()
    return constant + _kernel(model, train, query).transpose(-1, -2) @ alpha


def _conditional(model, train, targets, query):
    matrix = _kernel(model, train, train) + torch.diag_embed(model._train_Yvar.squeeze(-1))
    cross = _kernel(model, train, query)
    mean = model.mean_module.constant
    predicted = mean + cross.transpose(-1, -2) @ torch.linalg.solve(matrix, targets - mean)
    covariance = _kernel(model, query, query) - cross.transpose(-1, -2) @ torch.linalg.solve(
        matrix, cross
    )
    return predicted, covariance


def _warp(value, transform):
    selected = value[..., [0, 2]] * (1 - 2 * transform._eps) + transform._eps
    result = value.clone()
    result[..., [0, 2]] = 1 - (1 - selected.pow(transform.concentration1)).pow(
        transform.concentration0
    )
    return result


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("on_fantasy", [False, True])
@pytest.mark.parametrize("propagate", [False, True])
def test_warp_fantasy_phase_source_coordinates_and_independent_full_hessians(
    device, on_fantasy, propagate
):
    source_input = pytest.importorskip("botorch.models.transforms.input")
    source_gp = pytest.importorskip("botorch.models.gp_regression").SingleTaskGP
    source_kernel = pytest.importorskip("gpytorch.kernels").RBFKernel
    settings = pytest.importorskip("botorch.settings")
    X, Y, extra = _data(device)
    query = (extra.detach() + 0.025).requires_grad_()
    options = dict(
        d=3,
        indices=[0, 2],
        transform_on_fantasize=on_fantasy,
        bounds=X.new_tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]),
    )
    transform, source_transform = Warp(**options).to(X), source_input.Warp(**options).to(X)
    for current in (transform, source_transform):
        current.initialize(concentration0=1.4, concentration1=0.8)
    noise = torch.full_like(Y, 0.03)
    actual = SingleTaskGP(
        X,
        Y,
        noise,
        input_transform=transform,
        covar_module=RBFKernel(ard_num_dims=3),
        outcome_transform=None,
        backend="torch",
    )
    source = source_gp(
        X,
        Y,
        noise,
        input_transform=source_transform,
        covar_module=source_kernel(ard_num_dims=3),
        outcome_transform=None,
    ).to(X)
    source.covar_module.lengthscale = actual.covar_module.lengthscale.detach()
    source.mean_module.constant = actual.mean_module.constant.detach()
    left, right = MeanSampler(), MeanSampler()
    fantasy = actual.fantasize(extra, left, propagate_grads=propagate)
    reference = source.fantasize(extra, right, propagate_grads=propagate)
    reference.prediction_strategy = None
    with settings.propagate_grads(True):
        result, expected = fantasy.posterior(query), reference.posterior(query)
    torch.testing.assert_close(fantasy._training_values()[0], reference.train_inputs[0])
    torch.testing.assert_close(result.mean, expected.mean, atol=4e-10, rtol=4e-9)
    torch.testing.assert_close(
        result.distribution.covariance_matrix,
        expected.distribution.covariance_matrix,
        atol=4e-10,
        rtol=4e-9,
    )
    original_train = _warp(X, transform)
    appended = _warp(extra, transform) if on_fantasy else extra
    mean = _sample_mean(actual, original_train, Y, appended, propagate)
    samples = torch.stack((mean - 0.05, mean + 0.05))
    independent = _conditional(
        fantasy,
        torch.cat((original_train, appended), -2),
        torch.cat((Y.expand(2, *Y.shape), samples), -2),
        _warp(query, fantasy.input_transform),
    )
    independent = (independent[0], independent[1].expand_as(result.distribution.covariance_matrix))
    leaves = (
        X,
        Y,
        extra,
        query,
        actual.mean_module.raw_constant,
        actual.covar_module.raw_lengthscale,
        *transform.parameters(),
        *fantasy.parameters(),
    )
    for values, oracle in zip(
        _derivatives(result.mean, result.distribution.covariance_matrix, leaves),
        _derivatives(*independent, leaves),
        strict=True,
    ):
        for value, reference_value in zip(values, oracle, strict=True):
            torch.testing.assert_close(value, reference_value, atol=3e-7, rtol=3e-6)
    # The source caches original training preprocessing under no_grad. Its
    # future transform leaves nevertheless have the same query-only ownership.
    actual_parts = torch.autograd.grad(
        result.mean.sin().sum(),
        (query, fantasy.input_transform.concentration0, fantasy.input_transform.concentration1),
        retain_graph=True,
    )
    source_parts = torch.autograd.grad(
        expected.mean.sin().sum(),
        (query, reference.input_transform.concentration0, reference.input_transform.concentration1),
        retain_graph=True,
    )
    for value, reference_value in zip(actual_parts, source_parts, strict=True):
        torch.testing.assert_close(value, reference_value, atol=3e-8, rtol=3e-7)
    fantasy.train()
    reference.train()
    fantasy.eval()
    reference.eval()
    query_after_training = query.expand(2, *query.shape)
    result, expected = (
        fantasy.posterior(query_after_training),
        reference.posterior(query_after_training),
    )
    torch.testing.assert_close(result.mean, expected.mean, atol=4e-10, rtol=4e-9)
    torch.testing.assert_close(
        result.distribution.covariance_matrix,
        expected.distribution.covariance_matrix,
        atol=4e-10,
        rtol=4e-9,
    )


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("on_fantasy", [False, True])
def test_one_to_many_scalar_fantasies_match_supported_source_phase(device, on_fantasy):
    source_input = pytest.importorskip("botorch.models.transforms.input")
    source_gp = pytest.importorskip("botorch.models.gp_regression").SingleTaskGP
    source_kernel = pytest.importorskip("gpytorch.kernels").RBFKernel
    X, Y, extra = _data(device)
    perturbations = X.new_tensor([[0.02, -0.03, 0.01], [-0.01, 0.04, -0.02]])
    options = dict(perturbation_set=perturbations, transform_on_fantasize=on_fantasy)
    noise = torch.full_like(Y, 0.03)
    actual = SingleTaskGP(
        X,
        Y,
        noise,
        input_transform=InputPerturbation(**options),
        covar_module=RBFKernel(ard_num_dims=3),
        outcome_transform=None,
        backend="torch",
    )
    source = source_gp(
        X,
        Y,
        noise,
        input_transform=source_input.InputPerturbation(**options),
        covar_module=source_kernel(ard_num_dims=3),
        outcome_transform=None,
    ).to(X)
    source.covar_module.lengthscale = actual.covar_module.lengthscale.detach()
    source.mean_module.constant = actual.mean_module.constant.detach()
    left, right = MeanSampler(), MeanSampler()
    fantasy, reference = actual.fantasize(extra, left), source.fantasize(extra, right)
    assert left.shape == right.shape
    torch.testing.assert_close(fantasy._training_values()[0], reference.train_inputs[0])
    result, expected = fantasy.posterior(extra), reference.posterior(extra)
    torch.testing.assert_close(result.mean, expected.mean, atol=4e-10, rtol=4e-9)
    torch.testing.assert_close(
        result.distribution.covariance_matrix,
        expected.distribution.covariance_matrix,
        atol=4e-10,
        rtol=4e-9,
    )


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("with_tasks", [False, True])
@pytest.mark.parametrize("observed", [False, True])
def test_multitask_one_to_many_source_layout_values_and_gradients(device, with_tasks, observed):
    from botorch.models import MultiTaskGP as Source
    from botorch.models.transforms.input import InputPerturbation as SourceTransform
    from botorch.settings import propagate_grads
    from test_torch_backend_multitask import data

    X, Y, Q = data(device)
    X, Y, Q = X.requires_grad_(), Y.requires_grad_(), Q[0].requires_grad_()
    if with_tasks:
        Q = torch.cat((Q, Q.new_tensor([[0.0], [1.0]])), -1).detach().requires_grad_()
    delta = X.new_tensor([[0.02], [-0.03]], requires_grad=True)
    options = dict(perturbation_set=delta, indices=[0])
    noise = torch.full_like(Y, 0.03)
    torch.manual_seed(23)
    actual = MultiTaskGP(
        X,
        Y,
        -1,
        noise,
        input_transform=InputPerturbation(**options),
        outcome_transform=None,
        task_covar_prior=None,
        backend="torch",
    )
    torch.manual_seed(23)
    source = Source(
        X,
        Y,
        -1,
        noise,
        input_transform=SourceTransform(**options),
        outcome_transform=None,
        task_covar_prior=None,
    ).to(X)
    with torch.no_grad():
        expected_parameters = dict(source.named_parameters())
        for name, parameter in actual.named_parameters():
            expected_parameters[name].copy_(parameter)
    result = actual.posterior(Q, observation_noise=observed)
    with propagate_grads(True):
        expected = source.posterior(Q, observation_noise=observed)
    assert result.mean.shape == expected.mean.shape
    torch.testing.assert_close(result.mean, expected.mean, atol=3e-9, rtol=3e-8)
    torch.testing.assert_close(
        result.distribution.covariance_matrix,
        expected.distribution.covariance_matrix,
        atol=3e-9,
        rtol=3e-8,
    )
    leaves = (X, Y, Q, delta, *actual.parameters())
    source_leaves = (
        X,
        Y,
        Q,
        delta,
        *(dict(source.named_parameters())[name] for name, _ in actual.named_parameters()),
    )
    first = torch.autograd.grad(
        result.mean.sin().sum() + 0.1 * result.variance.square().sum(),
        leaves,
        create_graph=True,
        retain_graph=True,
    )
    reference = torch.autograd.grad(
        expected.mean.sin().sum() + 0.1 * expected.variance.square().sum(),
        source_leaves,
        retain_graph=True,
    )
    for value, reference_value in zip(first, reference, strict=True):
        torch.testing.assert_close(value, reference_value, atol=3e-7, rtol=3e-6)
    second = torch.autograd.grad(sum(p.square().sum() for p in first), leaves)
    # Independently form the task-major, point/perturbation-major query and
    # condition with dense solves, without the model's posterior dispatcher.
    full = (
        Q
        if with_tasks
        else torch.cat(
            (
                torch.cat((Q, Q.new_zeros(*Q.shape[:-1], 1)), -1),
                torch.cat((Q, Q.new_ones(*Q.shape[:-1], 1)), -1),
            ),
            -2,
        )
    )
    expanded = full.unsqueeze(-2).expand(*full.shape[:-1], 2, 2).clone()
    expanded[..., 0] = expanded[..., 0] + delta.squeeze(-1)
    expanded = expanded.flatten(-3, -2)
    train_mean, query_mean = actual._mean(X), actual._mean(expanded)
    matrix = actual._covariance(X, X) + torch.diag_embed(noise.squeeze(-1))
    cross = actual._covariance(X, expanded)
    manual_mean = query_mean + (
        cross.transpose(-1, -2) @ torch.linalg.solve(matrix, Y - train_mean.unsqueeze(-1))
    ).squeeze(-1)
    manual_cov = actual._covariance(expanded, expanded) - cross.transpose(
        -1, -2
    ) @ torch.linalg.solve(matrix, cross)
    if observed:
        manual_cov = (
            manual_cov + torch.eye(expanded.shape[-2], device=X.device, dtype=X.dtype) * 0.03
        )
    outputs = 1 if with_tasks else 2
    manual_mean = manual_mean.reshape(outputs, -1).transpose(-1, -2)
    manual_variance = manual_cov.diagonal().reshape(outputs, -1).transpose(-1, -2)
    manual_first = torch.autograd.grad(
        manual_mean.sin().sum() + 0.1 * manual_variance.square().sum(),
        leaves,
        create_graph=True,
        retain_graph=True,
    )
    manual_second = torch.autograd.grad(sum(p.square().sum() for p in manual_first), leaves)
    for values, controls in ((first, manual_first), (second, manual_second)):
        for value, reference_value in zip(values, controls, strict=True):
            torch.testing.assert_close(value, reference_value, atol=3e-7, rtol=3e-6)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("on_fantasy", [False, True])
def test_multitask_fantasy_expansion_phase_matches_unmodified_source(device, on_fantasy):
    from botorch.models import MultiTaskGP as Source
    from botorch.models.transforms.input import InputPerturbation as SourceTransform
    from test_torch_backend_multitask import data

    X, Y, Q = data(device)
    extra = torch.cat((Q[0], Q.new_tensor([[0.0], [1.0]])), -1)
    delta = X.new_tensor([[0.02], [-0.03]])
    options = dict(perturbation_set=delta, indices=[0], transform_on_fantasize=on_fantasy)
    torch.manual_seed(23)
    actual = MultiTaskGP(
        X,
        Y,
        -1,
        input_transform=InputPerturbation(**options),
        outcome_transform=None,
        task_covar_prior=None,
        backend="torch",
    )
    torch.manual_seed(23)
    source = Source(
        X,
        Y,
        -1,
        input_transform=SourceTransform(**options),
        outcome_transform=None,
        task_covar_prior=None,
    ).to(X)
    with torch.no_grad():
        parameters = dict(source.named_parameters())
        for name, parameter in actual.named_parameters():
            parameters[name].copy_(parameter)
    left, right = MeanSampler(), MeanSampler()
    fantasy, reference = actual.fantasize(extra, left), source.fantasize(extra, right)
    assert left.shape == right.shape
    torch.testing.assert_close(fantasy.train_inputs[0], reference.train_inputs[0])
    result, expected = fantasy.posterior(Q[0]), reference.posterior(Q[0])
    torch.testing.assert_close(result.mean, expected.mean, atol=3e-8, rtol=3e-7)
    torch.testing.assert_close(
        result.distribution.covariance_matrix,
        expected.distribution.covariance_matrix,
        atol=3e-8,
        rtol=3e-7,
    )

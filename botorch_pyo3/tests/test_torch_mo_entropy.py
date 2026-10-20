# ruff: noqa: E402
"""Device multiobjective entropy, synchronized reference samples and full graphs."""

import pytest

pytest.importorskip("torch")


import torch

from botorch_rs.acquisition.multi_objective.joint_entropy_search import (
    qLowerBoundMultiObjectiveJointEntropySearch,
)
from botorch_rs.acquisition.multi_objective.max_value_entropy_search import (
    qLowerBoundMultiObjectiveMaxValueEntropySearch,
)
from botorch_rs.models import ModelListGP, SingleTaskGP
from botorch_rs.sampling.normal import SobolQMCNormalSampler

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
ESTIMATORS = ["0", "LB", "LB2", "MC"]


def data(device, outputs=2, dtype=torch.double):
    X = torch.tensor([[0.05], [0.3], [0.65], [0.9]], dtype=dtype, device=device)
    Y = torch.cat((1 - 4 * (X - 0.45).square(), 1 - 4 * (X - 0.65).square()), -1)[..., :outputs]
    sets = X.new_tensor([[[0.39], [0.59]], [[0.43], [0.63]]])
    fronts = X.new_tensor([[[1.0, 0.8], [0.8, 1.0]], [[1.05, 0.75], [0.75, 1.05]]])[..., :outputs]
    if outputs == 2:
        cells = X.new_tensor(
            [
                [[[-4.0, -4.0], [0.8, -4.0]], [[0.8, 1.0], [1.0, 0.8]]],
                [[[-4.0, -4.0], [0.75, -4.0]], [[0.75, 1.05], [1.05, 0.75]]],
            ]
        )
    else:
        cells = X.new_tensor([[[[-4.0]], [[1.0]]], [[[-4.0]], [[1.05]]]])
    return X, Y, sets, fronts, cells


def make_pair(
    device,
    joint,
    estimator,
    outputs=2,
    pending=False,
    model_list=False,
    normalized=False,
    dtype=torch.double,
):
    from botorch.acquisition.multi_objective.joint_entropy_search import (
        qLowerBoundMultiObjectiveJointEntropySearch as ReferenceJES,
    )
    from botorch.acquisition.multi_objective.max_value_entropy_search import (
        qLowerBoundMultiObjectiveMaxValueEntropySearch as ReferenceMES,
    )
    from botorch.models import ModelListGP as ReferenceList
    from botorch.models import SingleTaskGP as ReferenceGP
    from botorch.models.transforms.input import Normalize as ReferenceNormalize
    from botorch.sampling.normal import SobolQMCNormalSampler as ReferenceSampler
    from mo_entropy_helpers import jes_pareto_sets

    from botorch_rs.models.transforms.input import Normalize

    X, Y, sets, fronts, cells = data(device, outputs, dtype)
    query = X.new_tensor([[[0.17], [0.77]], [[0.23], [0.83]]])
    pending_X = X.new_tensor([[0.92]]) if pending else None
    if normalized:
        X, sets, query = X * 3 + 2, sets * 3 + 2, query * 3 + 2
        if pending_X is not None:
            pending_X = pending_X * 3 + 2

    def build(kind):
        cls = SingleTaskGP if kind == 0 else ReferenceGP
        tf = None if not normalized else (Normalize if kind == 0 else ReferenceNormalize)(1)
        kwargs = dict(outcome_transform=None, input_transform=tf)
        if kind == 0:
            kwargs["backend"] = "torch"
        if model_list:
            list_class = ModelListGP if kind == 0 else ReferenceList
            return list_class(*[cls(X, Y[..., i : i + 1], **kwargs) for i in range(outputs)])
        return cls(X, Y, **kwargs)

    actual_model, reference_model = build(0), build(1)
    options = dict(X_pending=pending_X, estimation_type=estimator, num_samples=16)
    if joint:
        actual = qLowerBoundMultiObjectiveJointEntropySearch(
            actual_model,
            sets,
            fronts,
            cells,
            **options,
            sampler=SobolQMCNormalSampler((16,), seed=53),
        )
        expected = ReferenceJES(
            reference_model,
            sets if model_list else jes_pareto_sets(reference_model, sets),
            fronts,
            cells,
            **options,
        )
    else:
        actual = qLowerBoundMultiObjectiveMaxValueEntropySearch(
            actual_model, cells, **options, sampler=SobolQMCNormalSampler((16,), seed=53)
        )
        expected = ReferenceMES(reference_model, cells, **options)
    expected.sampler = ReferenceSampler(torch.Size((16,)), seed=53)
    return actual, expected, query


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("joint", [False, True])
@pytest.mark.parametrize("estimator", ESTIMATORS)
@pytest.mark.parametrize("outputs", [1, 2])
def test_values_query_gradients_and_hvps(device, joint, estimator, outputs):
    actual, reference, query = make_pair(device, joint, estimator, outputs)
    query.requires_grad_()
    value, expected = actual(query), reference(query)
    torch.testing.assert_close(value, expected, rtol=2e-7, atol=2e-8)
    gradient = torch.autograd.grad(value.sum(), query, create_graph=True)[0]
    expected_gradient = torch.autograd.grad(expected.sum(), query)[0]
    torch.testing.assert_close(gradient, expected_gradient, rtol=3e-5, atol=3e-6)
    assert torch.isfinite(torch.autograd.grad(gradient.square().sum(), query)[0]).all()
    assert value.device == query.device


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("joint", [False, True])
@pytest.mark.parametrize("estimator", ["LB", "MC"])
def test_normalize_model_list_pending_and_multiple_t_batches(device, joint, estimator):
    actual, reference, query = make_pair(
        device, joint, estimator, pending=True, model_list=True, normalized=True
    )
    query = query[:, None].expand(2, 3, 2, 1).clone().requires_grad_()
    value, expected = actual(query), reference(query)
    torch.testing.assert_close(value, expected, rtol=3e-7, atol=3e-8)
    gradient = torch.autograd.grad(value.sum(), query, create_graph=True)[0]
    expected_gradient = torch.autograd.grad(expected.sum(), query)[0]
    torch.testing.assert_close(gradient, expected_gradient, rtol=4e-5, atol=4e-6)
    assert torch.isfinite(torch.autograd.grad(gradient.square().sum(), query)[0]).all()


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("joint", [False, True])
@pytest.mark.parametrize("estimator", ESTIMATORS)
def test_training_cells_and_optima_complete_gradgradcheck(device, joint, estimator):
    X, Y, sets, fronts, cells = data(device)
    X, Y = X[:3].requires_grad_(), Y[:3].requires_grad_()
    sets, fronts = sets[:1, :1].requires_grad_(), fronts[:1, :1].requires_grad_()
    # One box tests the smooth interior. Perturbing duplicate shared faces
    # independently can make boxes overlap and cross the reference W<=1
    # mass clamp, where a classical Hessian need not exist.
    cells = torch.stack((torch.full_like(fronts, -0.2), fronts.detach()), 1).requires_grad_()
    query = X.new_tensor([[0.21]], requires_grad=True)

    def evaluate(train_X, train_Y, optimal_X, optimal_Y, bounds, points):
        model = SingleTaskGP(train_X, train_Y, backend="torch", outcome_transform=None)
        kwargs = dict(estimation_type=estimator, sampler=SobolQMCNormalSampler((8,), seed=59))
        if joint:
            acquisition = qLowerBoundMultiObjectiveJointEntropySearch(
                model, optimal_X, optimal_Y, bounds, **kwargs
            )
        else:
            acquisition = qLowerBoundMultiObjectiveMaxValueEntropySearch(model, bounds, **kwargs)
        return acquisition(points)

    arguments = X, Y, sets, fronts, cells, query
    assert torch.autograd.gradcheck(evaluate, arguments, fast_mode=True, rtol=3e-4, atol=3e-5)
    assert torch.autograd.gradgradcheck(evaluate, arguments, fast_mode=True, rtol=4e-4, atol=4e-5)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("joint", [False, True])
@pytest.mark.parametrize("estimator", ["LB", "MC"])
def test_all_parameter_graphs_without_inference_host_data(device, joint, estimator, monkeypatch):
    actual, _, query = make_pair(device, joint, estimator)
    query.requires_grad_()

    def forbidden(*args, **kwargs):
        raise AssertionError("Device entropy transferred inference data to host.")

    monkeypatch.setattr(torch.Tensor, "cpu", forbidden)
    monkeypatch.setattr(torch.Tensor, "numpy", forbidden)
    monkeypatch.setattr(torch.Tensor, "tolist", forbidden)
    value = actual(query)
    parameters = (query, *actual.model.parameters())
    first = torch.autograd.grad(value.sum(), parameters, create_graph=True)
    second = torch.autograd.grad(sum(g.square().sum() for g in first), parameters)
    assert all(torch.isfinite(g).all() for g in (*first, *second))


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("estimator", ["0", "LB", "LB2"])
def test_infinite_box_limits_have_gaussian_values_and_finite_hessians(device, estimator):
    import math

    from botorch.acquisition.multi_objective import joint_entropy_search as reference

    from botorch_rs.acquisition.multi_objective.joint_entropy_search import _entropy

    mean = torch.tensor([[[[[0.2, 0.3]]]]], dtype=torch.double, device=device, requires_grad=True)
    variance = mean.new_tensor([[[[[0.7, 0.8]]]]], requires_grad=True)
    noise = torch.full_like(variance, 0.1, requires_grad=True)
    bounds = mean.new_tensor([[[[-torch.inf, -torch.inf]], [[torch.inf, torch.inf]]]])
    value = _entropy(estimator, bounds, mean, variance, noise)
    jitter = 1e-6 if estimator == "LB" else 0.0
    expected = 0.5 * 2 * (1 + math.log(2 * math.pi)) + 0.5 * (variance + noise + jitter).log().sum(
        -1
    ).squeeze(-1).mean(-2)
    torch.testing.assert_close(value, expected)
    if estimator == "0":
        original = reference._compute_entropy_noiseless(bounds, mean, variance, noise)
    else:
        original = reference._compute_entropy_upper_bound(
            bounds, mean, variance, noise, only_diagonal=estimator == "LB2"
        )
    assert not torch.isfinite(original).any()
    first = torch.autograd.grad(value.sum(), (mean, variance, noise), create_graph=True)
    second = torch.autograd.grad(
        sum(g.square().sum() for g in first), (mean, variance, noise), allow_unused=True
    )
    assert all(g is None or torch.isfinite(g).all() for g in (*first, *second))


@pytest.mark.parametrize("device", DEVICES)
def test_monte_carlo_infinite_box_preserves_gaussian_entropy_gradients(device):
    import math

    from botorch.acquisition.multi_objective import joint_entropy_search as reference

    from botorch_rs.acquisition.multi_objective.joint_entropy_search import _entropy

    mean = torch.tensor([[[[[0.2, 0.3]]]]], dtype=torch.double, device=device, requires_grad=True)
    variance = mean.new_tensor([[[[[0.7, 0.8]]]]], requires_grad=True)
    noise = torch.full_like(variance, 0.1, requires_grad=True)
    bounds = mean.new_tensor([[[[-torch.inf, -torch.inf]], [[torch.inf, torch.inf]]]])
    normals = torch.randn(
        (8, *mean.shape),
        dtype=mean.dtype,
        device=device,
        generator=torch.Generator(device=device).manual_seed(61),
    )
    samples = mean + (variance + noise).sqrt() * normals
    log_probability = -0.5 * (
        normals.square() + (variance + noise).log() + math.log(2 * math.pi)
    ).sum(-1).squeeze(-1)
    value = _entropy("MC", bounds, mean, variance, noise, samples, log_probability)
    torch.testing.assert_close(value, -log_probability.mean(0).mean(-2))
    original = reference._compute_entropy_monte_carlo(
        bounds, mean, variance, noise, samples, log_probability
    )
    original_gradient = torch.autograd.grad(original.sum(), variance, retain_graph=True)[0]
    assert not torch.isfinite(original_gradient).all()
    first = torch.autograd.grad(value.sum(), (mean, variance, noise), create_graph=True)
    second = torch.autograd.grad(
        sum(g.square().sum() for g in first), (mean, variance, noise), allow_unused=True
    )
    assert all(g is None or torch.isfinite(g).all() for g in (*first, *second))


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("joint", [False, True])
@pytest.mark.parametrize("estimator", ["LB", "MC"])
def test_float32_values_and_finite_query_hvp(device, joint, estimator):
    actual, reference, query = make_pair(device, joint, estimator, dtype=torch.float32)
    query.requires_grad_()
    value, expected = actual(query), reference(query)
    torch.testing.assert_close(value, expected, rtol=3e-4, atol=3e-5)
    gradient = torch.autograd.grad(value.sum(), query, create_graph=True)[0]
    assert torch.isfinite(torch.autograd.grad(gradient.square().sum(), query)[0]).all()
    assert value.dtype == torch.float32

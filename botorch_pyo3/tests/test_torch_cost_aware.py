"""Cost models and utilities preserve their Tensor buffers and full graphs."""

from __future__ import annotations

from unittest.mock import patch

import pytest

torch = pytest.importorskip("torch")

from botorch_rs.acquisition.cost_aware import InverseCostWeightedUtility  # noqa: E402
from botorch_rs.acquisition.objective import GenericMCObjective  # noqa: E402
from botorch_rs.models import SingleTaskGP  # noqa: E402
from botorch_rs.models.cost import AffineFidelityCostModel, FixedCostModel  # noqa: E402
from botorch_rs.sampling.normal import IIDNormalSampler  # noqa: E402

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


def inputs(device, dtype=torch.double):
    X = torch.tensor(
        [[[0.1, 0.2], [0.4, 0.7]], [[0.6, 0.8], [0.2, 0.5]]],
        device=device,
        dtype=dtype,
        requires_grad=True,
    )
    deltas = torch.tensor(
        [[0.2, -0.4], [-0.3, 0.8], [0.4, 0.7]], device=device, dtype=dtype, requires_grad=True
    )
    return X, deltas


def gradients_and_hvp(loss, arguments):
    first = torch.autograd.grad(loss, arguments, create_graph=True)
    second = torch.autograd.grad(sum(g.square().sum() for g in first), arguments)
    return first, second


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.double])
def test_affine_persistent_weights_fixed_cost_and_full_graph_reference(device, dtype):
    pytest.importorskip("botorch")
    from botorch.models.cost import AffineFidelityCostModel as ReferenceCost

    X, _ = inputs(device, dtype)
    fixed = X.new_tensor(0.15, requires_grad=True)
    ref_fixed = fixed.detach().clone().requires_grad_()
    actual = AffineFidelityCostModel({0: 0.123456789, -1: 0.345678912}, fixed).to(X)
    expected = ReferenceCost({0: 0.123456789, -1: 0.345678912}, ref_fixed).to(X)
    actual.weights.requires_grad_()
    expected.weights.requires_grad_()
    assert dict(actual.named_buffers())["weights"] is actual.weights
    a, e = actual(X), expected(X)
    torch.testing.assert_close(a, e)
    ga = gradients_and_hvp(a.square().sum(), (X, actual.weights, fixed))
    ge = gradients_and_hvp(e.square().sum(), (X, expected.weights, ref_fixed))
    for actual_order, expected_order in zip(ga, ge, strict=True):
        for value, reference in zip(actual_order, expected_order, strict=True):
            torch.testing.assert_close(value, reference)
    posterior = actual.posterior(X)
    torch.testing.assert_close(posterior.mean, a)
    assert posterior.dtype == dtype and posterior.device == X.device
    assert torch.equal(posterior.variance, torch.zeros_like(a))


@pytest.mark.parametrize("device", DEVICES)
def test_fixed_cost_buffer_identity_sampling_and_second_derivatives(device):
    pytest.importorskip("botorch")
    from botorch.models.cost import FixedCostModel as ReferenceCost

    X, deltas = inputs(device)
    cost = X.new_tensor([0.4, 0.9], requires_grad=True)
    ref_cost = cost.detach().clone().requires_grad_()
    actual, expected = FixedCostModel(cost), ReferenceCost(ref_cost)
    assert actual.fixed_cost is cost and dict(actual.named_buffers())["fixed_cost"] is cost
    torch.testing.assert_close(actual(X), expected(X))
    sampled = IIDNormalSampler(torch.Size([3]), seed=5)(actual.posterior(X))
    torch.testing.assert_close(sampled, actual(X).expand(3, *actual(X).shape))
    utility = InverseCostWeightedUtility(actual)
    from botorch.acquisition.cost_aware import InverseCostWeightedUtility as ReferenceUtility

    reference = ReferenceUtility(expected)
    a, e = utility(X, deltas), reference(X, deltas)
    torch.testing.assert_close(a, e)
    ga, ge = (
        gradients_and_hvp(a.square().sum(), (cost, deltas)),
        gradients_and_hvp(e.square().sum(), (ref_cost, deltas)),
    )
    for av, ev in zip(ga, ge, strict=True):
        for value, expected_value in zip(av, ev, strict=True):
            torch.testing.assert_close(value, expected_value)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("log", [False, True])
@pytest.mark.parametrize("use_mean", [False, True])
@pytest.mark.parametrize("mask_outputs", [False, True])
def test_inverse_cost_defaults_masks_signed_log_and_stochastic_reference(
    device, log, use_mean, mask_outputs
):
    pytest.importorskip("botorch")
    from botorch.acquisition.cost_aware import InverseCostWeightedUtility as ReferenceUtility
    from botorch.models.cost import FixedCostModel as ReferenceCost

    X, deltas = inputs(device)
    fixed = X.new_tensor([0.6, 1.2], requires_grad=True)
    actual = InverseCostWeightedUtility(FixedCostModel(fixed), use_mean=use_mean, log=log)
    expected = ReferenceUtility(ReferenceCost(fixed), use_mean=use_mean, log=log)
    mask = torch.tensor([[False, True], [False, True]], device=device) if mask_outputs else None

    def sampler(posterior):
        return posterior.rsample(torch.Size([3]))

    a = actual(X, deltas, sampler=sampler, X_evaluation_mask=mask)
    e = expected(X, deltas, sampler=sampler, X_evaluation_mask=mask)
    torch.testing.assert_close(a, e)
    gradients = gradients_and_hvp(a.square().sum(), (fixed, deltas))
    assert all(torch.isfinite(g).all() for order in gradients for g in order)


@pytest.mark.parametrize("device", DEVICES)
def test_affine_weights_targets_and_deltas_gradcheck_gradgradcheck(device):
    X, deltas = inputs(device)
    weights = X.new_tensor([0.4, 0.8], requires_grad=True)
    fixed = X.new_tensor(0.2, requires_grad=True)

    def evaluate(x, d, w, c):
        model = AffineFidelityCostModel({0: w[0], 1: w[1]}, c)
        return InverseCostWeightedUtility(model)(x, d)

    assert torch.autograd.gradcheck(evaluate, (X, deltas, weights, fixed), fast_mode=True)
    assert torch.autograd.gradgradcheck(evaluate, (X, deltas, weights, fixed), fast_mode=True)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("use_mean", [False, True])
def test_gp_cost_parameters_training_data_and_queries_keep_full_graph(device, use_mean):
    train = torch.tensor(
        [[0.1, 0.2], [0.5, 0.7], [0.8, 0.4]], dtype=torch.double, device=device, requires_grad=True
    )
    targets = torch.tensor(
        [[2.1], [2.5], [2.7]], dtype=torch.double, device=device, requires_grad=True
    )
    cost_model = SingleTaskGP(train, targets, backend="torch")
    with patch.object(
        cost_model, "_acquisition_model", side_effect=AssertionError("Native model handle")
    ):
        utility = InverseCostWeightedUtility(cost_model, use_mean=use_mean)
    assert utility.cost_model is cost_model
    assert {id(p) for p in utility.parameters()} == {id(p) for p in cost_model.parameters()}
    X, deltas = inputs(device)
    for _ in range(2):
        sampler = IIDNormalSampler(torch.Size([3]), seed=4)
        result = utility(X, deltas, sampler=sampler)
        arguments = (X, deltas, train, targets, *tuple(cost_model.parameters()))
        first = torch.autograd.grad(result.square().sum(), arguments, create_graph=True)
        second = torch.autograd.grad(sum(g.square().sum() for g in first), arguments)
        assert all(torch.isfinite(g).all() for g in (*first, *second))
        assert utility._native_utility is None


@pytest.mark.parametrize("device", DEVICES)
def test_tensor_forward_does_not_flatten_copy_or_detach_costs_or_masks(device):
    X, deltas = inputs(device)
    model = FixedCostModel(X.new_tensor([0.4, 0.8], requires_grad=True))
    utility = InverseCostWeightedUtility(model)
    mask = torch.tensor([[True, False], [True, False]], device=device)
    with (
        patch.object(torch.Tensor, "tolist", side_effect=AssertionError("Host list")),
        patch.object(torch.Tensor, "cpu", side_effect=AssertionError("CPU copy")),
        patch.object(torch.Tensor, "numpy", side_effect=AssertionError("NumPy copy")),
        patch.object(torch.Tensor, "detach", side_effect=AssertionError("Detached graph")),
    ):
        result = utility(X, deltas, X_evaluation_mask=mask)
        assert torch.isfinite(
            torch.autograd.grad(result.sum(), model.fixed_cost, create_graph=True)[0]
        ).all()


@pytest.mark.parametrize("device", DEVICES)
def test_custom_objective_positive_cost_validation_and_mask_errors(device):
    X, deltas = inputs(device)
    objective = GenericMCObjective(lambda samples, X=None: samples.exp().sum(-1))
    model = AffineFidelityCostModel({0: -0.5}, fixed_cost=-0.1)
    utility = InverseCostWeightedUtility(model, cost_objective=objective)
    assert torch.isfinite(utility(X, deltas)).all()
    with pytest.raises(ValueError, match="strictly positive"):
        InverseCostWeightedUtility(model)(X, deltas)
    fixed = FixedCostModel(X.new_tensor([1.0, 2.0]))
    utility = InverseCostWeightedUtility(fixed)
    with pytest.raises(NotImplementedError, match="same outputs"):
        utility(
            X, deltas, X_evaluation_mask=torch.tensor([[True, False], [False, True]], device=device)
        )
    with pytest.raises(ValueError, match="strictly positive"):
        utility(X, deltas, X_evaluation_mask=torch.zeros(2, 2, dtype=torch.bool, device=device))
    with pytest.raises(RuntimeError, match="sampler"):
        InverseCostWeightedUtility(fixed, use_mean=False)(X, deltas)


def test_native_list_and_numpy_behavior_remains_available_with_torch_loaded():
    import numpy as np

    model = AffineFidelityCostModel({-1: 0.75}, fixed_cost=0.25)
    utility = InverseCostWeightedUtility(model)
    X = [[[0.2, 0.4], [0.3, 0.8]]]
    deltas = [[0.3], [-0.2]]
    expected = [[0.3 / 1.4], [-0.2 * 1.4]]
    np.testing.assert_allclose(utility(X, deltas), expected, atol=1e-15, rtol=1e-15)
    np.testing.assert_allclose(utility(np.array(X), np.array(deltas)), expected)
    model.weights.add_(0.25)
    assert model([[0.2, 0.4]]) == [[0.65]]
    np.testing.assert_allclose(utility(X, deltas), [[0.3 / 1.7], [-0.2 * 1.7]])


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("stochastic", [False, True])
def test_multifidelity_kg_inverse_cost_end_to_end_parameters_and_double_backward(
    device, stochastic
):
    from botorch_rs.acquisition.knowledge_gradient import qMultiFidelityKnowledgeGradient
    from botorch_rs.sampling.normal import SobolQMCNormalSampler

    X = torch.tensor([[0.1], [0.45], [0.9]], dtype=torch.double, device=device)
    outcome = SingleTaskGP(X, X.sin(), backend="torch", outcome_transform=None)
    if stochastic:
        targets = (2.0 + X.square()).requires_grad_()
        cost_model = SingleTaskGP(X, targets, backend="torch")
        utility = InverseCostWeightedUtility(
            cost_model,
            use_mean=False,
            cost_objective=GenericMCObjective(lambda Y, X=None: Y.squeeze(-1).exp()),
        )
        cost_leaves = (targets, *tuple(cost_model.parameters()))
    else:
        cost_model = AffineFidelityCostModel({-1: 0.5}, fixed_cost=0.2).to(X)
        cost_model.weights.requires_grad_()
        utility = InverseCostWeightedUtility(cost_model)
        cost_leaves = (cost_model.weights,)
    acq = qMultiFidelityKnowledgeGradient(
        outcome,
        num_fantasies=2,
        sampler=SobolQMCNormalSampler((2,), seed=7),
        current_value=0.3,
        cost_aware_utility=utility,
    )
    query = X.new_tensor([[0.23], [0.41], [0.86]], requires_grad=True)
    value = acq(query)
    arguments = (query, *cost_leaves, *tuple(outcome.parameters()))
    first = torch.autograd.grad(value.sum(), arguments, create_graph=True)
    second = torch.autograd.grad(sum(g.square().sum() for g in first), arguments)
    assert all(torch.isfinite(g).all() for g in (*first, *second))


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("use_mean", [False, True])
def test_gaussian_cost_mean_or_shared_normal_samples_and_parameter_gradients_reference(
    device, use_mean
):
    pytest.importorskip("botorch")
    from botorch.acquisition.cost_aware import InverseCostWeightedUtility as ReferenceUtility
    from botorch.models import SingleTaskGP as ReferenceGP
    from botorch.settings import propagate_grads

    pytest.importorskip("gpytorch")
    from gpytorch.settings import detach_test_caches

    train = torch.tensor([[0.1, 0.2], [0.5, 0.7], [0.8, 0.4]], dtype=torch.double, device=device)
    targets = torch.tensor([[2.1], [2.5], [2.7]], dtype=torch.double, device=device)
    model, reference = SingleTaskGP(train, targets, backend="torch"), ReferenceGP(train, targets)
    X, deltas = inputs(device)
    normals = X.new_tensor([[[0.1, -0.4]], [[0.3, 0.7]], [[-0.2, 0.6]]])

    def sampler(posterior):
        base = normals.expand(torch.Size([3]) + posterior.base_sample_shape)
        return posterior.rsample_from_base_samples(torch.Size([3]), base)

    with propagate_grads(True), detach_test_caches(False):
        actual = InverseCostWeightedUtility(model, use_mean=use_mean)(X, deltas, sampler=sampler)
        expected = ReferenceUtility(reference, use_mean=use_mean)(X, deltas, sampler=sampler)
        torch.testing.assert_close(actual, expected, atol=1e-9, rtol=1e-9)
        named = dict(reference.named_parameters())
        ga = torch.autograd.grad(actual.square().sum(), (X, deltas, *tuple(model.parameters())))
        ge = torch.autograd.grad(
            expected.square().sum(),
            (X, deltas, *(named[name] for name, _ in model.named_parameters())),
        )
        for value, reference_value in zip(ga, ge, strict=True):
            torch.testing.assert_close(value, reference_value, atol=2e-8, rtol=2e-8)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("generic_model", [False, True])
def test_callable_and_generic_deterministic_costs_keep_legacy_tensor_support(device, generic_model):
    from botorch_rs.models.deterministic import GenericDeterministicModel

    X, deltas = inputs(device)
    weight = X.new_tensor(0.7, requires_grad=True)

    def cost(x):
        return 1 + weight * x.square().sum(-1, keepdim=True)

    model = GenericDeterministicModel(cost) if generic_model else cost
    utility = InverseCostWeightedUtility(model, use_mean=False)
    actual = utility(X, deltas, sampler=IIDNormalSampler((3,), seed=7))
    total = cost(X).squeeze(-1).sum(-1)
    expected = torch.where(deltas > 0, deltas / total, deltas * total)
    torch.testing.assert_close(actual, expected)
    first = torch.autograd.grad(actual.sum(), (X, deltas, weight), create_graph=True)
    second = torch.autograd.grad(sum(g.square().sum() for g in first), (X, deltas, weight))
    assert all(torch.isfinite(g).all() for g in (*first, *second))

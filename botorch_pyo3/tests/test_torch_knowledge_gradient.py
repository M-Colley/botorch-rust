# ruff: noqa: E402
"""One-shot lookahead values and full functional derivatives on CPU/CUDA."""

import pytest

pytest.importorskip("torch")


from unittest.mock import patch

import torch

from botorch_rs.acquisition.knowledge_gradient import (
    qKnowledgeGradient,
    qMultiFidelityKnowledgeGradient,
)
from botorch_rs.acquisition.objective import GenericMCObjective, LinearMCObjective
from botorch_rs.models import SingleTaskGP
from botorch_rs.sampling.normal import SobolQMCNormalSampler

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


def model_data(device, outputs=1):
    X = torch.tensor([[0.04], [0.27], [0.56], [0.81]], dtype=torch.double, device=device)
    Y = torch.cat([(X * 5.3).sin(), (X * 3.1).cos()], dim=-1)[..., :outputs]
    return X, Y


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("objective_kind", ["mean", "nonlinear", "linear"])
@pytest.mark.parametrize("pending", [False, True])
def test_kg_values_match_pinned_reference(device, objective_kind, pending):
    pytest.importorskip("botorch")
    from botorch.acquisition.knowledge_gradient import qKnowledgeGradient as ReferenceKG
    from botorch.acquisition.objective import GenericMCObjective as ReferenceObjective
    from botorch.models import SingleTaskGP as ReferenceGP
    from botorch.sampling.normal import SobolQMCNormalSampler as ReferenceSampler

    X, Y = model_data(device, outputs=2 if objective_kind == "linear" else 1)
    model = SingleTaskGP(X, Y, backend="torch", outcome_transform=None)
    reference = ReferenceGP(X, Y, outcome_transform=None)
    objective = reference_objective = None
    if objective_kind == "nonlinear":

        def function(Z, X=None):
            return Z.squeeze(-1).square()

        objective = GenericMCObjective(function)
        reference_objective = ReferenceObjective(function)
    elif objective_kind == "linear":
        weights = torch.tensor([0.7, -0.2], device=device, dtype=torch.double)
        objective = LinearMCObjective(weights)
        reference_objective = ReferenceObjective(lambda Z, X=None: Z @ weights)
    arguments = {"num_fantasies": 3, "current_value": 0.2}
    if pending:
        arguments["X_pending"] = X.new_tensor([[0.95]])
    acq = qKnowledgeGradient(
        model,
        sampler=SobolQMCNormalSampler((3,), seed=41),
        inner_sampler=SobolQMCNormalSampler((8,), seed=23),
        objective=objective,
        **arguments,
    )
    ref = ReferenceKG(
        reference,
        sampler=ReferenceSampler(torch.Size([3]), seed=41),
        inner_sampler=ReferenceSampler(torch.Size([8]), seed=23),
        objective=reference_objective,
        **arguments,
    )
    query = X.new_tensor(
        [[[0.19], [0.65], [0.15], [0.49], [0.89]], [[0.31], [0.74], [0.21], [0.53], [0.93]]]
    )
    actual, expected = acq(query), ref(query)
    torch.testing.assert_close(actual, expected, atol=2e-8, rtol=2e-8)
    assert isinstance(acq, qKnowledgeGradient)
    assert isinstance(acq, torch.nn.Module)
    assert actual.device == X.device
    assert acq.get_augmented_q_batch_size(2) == 5
    torch.testing.assert_close(acq.extract_candidates(query), query[..., :2, :])


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("nonlinear", [False, True])
def test_kg_training_candidate_and_parameter_second_derivatives(device, nonlinear):
    X, Y = model_data(device)
    X.requires_grad_()
    Y.requires_grad_()
    query = X.new_tensor([[0.17], [0.44], [0.87]], requires_grad=True)
    current_value = X.new_tensor(0.1, requires_grad=True)

    def make_acq(train_X, train_Y):
        model = SingleTaskGP(train_X, train_Y, backend="torch", outcome_transform=None)
        objective = (
            GenericMCObjective(lambda Z, X=None: Z.squeeze(-1).square()) if nonlinear else None
        )
        return qKnowledgeGradient(
            model,
            num_fantasies=2,
            sampler=SobolQMCNormalSampler((2,), seed=19),
            inner_sampler=SobolQMCNormalSampler((4,), seed=11),
            objective=objective,
            current_value=current_value,
        )

    acq = make_acq(X, Y)
    with (
        patch.object(torch.Tensor, "cpu", side_effect=AssertionError("host transfer")),
        patch.object(torch.Tensor, "numpy", side_effect=AssertionError("host transfer")),
        patch.object(torch.Tensor, "tolist", side_effect=AssertionError("host transfer")),
    ):
        leaves = (X, Y, query, current_value, *acq.model.parameters())
        gradients = torch.autograd.grad(acq(query).sum(), leaves, create_graph=True)
        # current_value is affine, hence its gradient has no graph of its own.
        second = torch.autograd.grad(
            sum(g.square().sum() for g in gradients if g.requires_grad), leaves, allow_unused=True
        )
    assert all(torch.isfinite(g).all() for g in gradients)
    assert all(g is None or torch.isfinite(g).all() for g in second)

    def function(a, b, c):
        return make_acq(a, b)(c)

    assert torch.autograd.gradcheck(function, (X, Y, query), atol=2e-5, rtol=2e-4)
    assert torch.autograd.gradgradcheck(function, (X, Y, query), atol=3e-5, rtol=3e-4)


@pytest.mark.parametrize("device", DEVICES)
def test_multifidelity_kg_project_expand_and_generic_cost_graphs(device):
    from botorch_rs.acquisition.cost_aware import GenericCostAwareUtility

    X, Y = model_data(device)
    model = SingleTaskGP(X, Y, backend="torch", outcome_transform=None)
    current = X.new_tensor(0.3, requires_grad=True)

    def project(Z):
        return 0.1 + 0.8 * Z

    def expand(Z):
        return torch.cat((Z, 0.1 + 0.8 * Z), dim=-2)

    utility = GenericCostAwareUtility(lambda X, deltas: deltas / (1 + X.square().sum(dim=(-2, -1))))
    acq = qMultiFidelityKnowledgeGradient(
        model,
        num_fantasies=2,
        sampler=SobolQMCNormalSampler((2,), seed=7),
        project=project,
        expand=expand,
        current_value=current,
        cost_aware_utility=utility,
    )
    query = X.new_tensor([[0.23], [0.41], [0.86]], requires_grad=True)
    value = acq(query)
    gradients = torch.autograd.grad(
        value.sum(), (query, current, *model.parameters()), create_graph=True
    )
    second = torch.autograd.grad(
        sum(g.square().sum() for g in gradients), (query, *model.parameters())
    )
    assert all(torch.isfinite(g).all() for g in second)


@pytest.mark.parametrize("device", DEVICES)
def test_kg_evaluate_inner_optimum_and_outer_envelope_gradients(device):
    pytest.importorskip("scipy")
    X, Y = model_data(device)
    model = SingleTaskGP(X, Y, backend="torch", outcome_transform=None)
    acq = qKnowledgeGradient(
        model, num_fantasies=2, sampler=SobolQMCNormalSampler((2,), seed=7), current_value=0.2
    )
    query = X.new_tensor([[0.21]], requires_grad=True)
    bounds = X.new_tensor([[0.0], [1.0]])
    value = acq.evaluate(
        query,
        bounds,
        num_restarts=4,
        raw_samples=32,
        options={"seed": 17},
        scipy_options={"maxiter": 35, "ftol": 1e-12},
    )
    fantasy = model.fantasize(query.unsqueeze(0), acq.sampler, propagate_grads=True)
    grid = (
        torch.linspace(0, 1, 1001, dtype=X.dtype, device=X.device)
        .reshape(1001, 1, 1, 1, 1)
        .expand(1001, 2, 1, 1, 1)
    )
    with torch.no_grad():
        expected = (
            fantasy.posterior(grid).mean.squeeze(-1).squeeze(-1).amax(dim=0).mean(dim=0) - 0.2
        )
    torch.testing.assert_close(value, expected, atol=2e-5, rtol=2e-5)
    gradient = torch.autograd.grad(value.sum(), (query, *model.parameters()))
    assert all(torch.isfinite(g).all() for g in gradient)
    epsilon = 1e-5

    def evaluate(points):
        return acq.evaluate(
            points,
            bounds,
            num_restarts=4,
            raw_samples=32,
            options={"seed": 17},
            scipy_options={"maxiter": 35, "ftol": 1e-12},
        )

    finite_difference = (
        evaluate(query.detach() + epsilon) - evaluate(query.detach() - epsilon)
    ) / (2 * epsilon)
    torch.testing.assert_close(
        gradient[0].squeeze(), finite_difference.squeeze(), atol=3e-4, rtol=3e-4
    )


@pytest.mark.parametrize("device", DEVICES)
def test_kg_evaluate_torch_optimizer_has_no_host_value_transfer(device):
    from botorch_rs.generation.gen import gen_candidates_torch

    X, Y = model_data(device)
    model = SingleTaskGP(X, Y, backend="torch", outcome_transform=None)
    acq = qKnowledgeGradient(model, num_fantasies=2, sampler=SobolQMCNormalSampler((2,), seed=7))
    query = X.new_tensor([[0.21]], requires_grad=True)
    with (
        patch.object(torch.Tensor, "cpu", side_effect=AssertionError("host transfer")),
        patch.object(torch.Tensor, "tolist", side_effect=AssertionError("host transfer")),
        patch.object(torch.Tensor, "numpy", side_effect=AssertionError("host transfer")),
    ):
        value = acq.evaluate(
            query,
            X.new_tensor([[0.0], [1.0]]),
            num_restarts=2,
            raw_samples=8,
            options={"seed": 19, "stopping_criterion_options": {"maxiter": 3}},
            gen_candidates=gen_candidates_torch,
        )
        gradient = torch.autograd.grad(value.sum(), query)[0]
    assert torch.isfinite(value).all()
    assert torch.isfinite(gradient).all()

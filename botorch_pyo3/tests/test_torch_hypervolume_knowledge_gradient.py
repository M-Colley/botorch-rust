"""Device HVKG fantasies, reference values and complete functional graphs."""

from unittest.mock import patch

import pytest
import torch

from botorch_rs.acquisition.multi_objective.hypervolume_knowledge_gradient import (
    _get_hv_value_function,
    _split_hvkg_fantasy_points,
    qHypervolumeKnowledgeGradient,
    qMultiFidelityHypervolumeKnowledgeGradient,
)
from botorch_rs.models import ModelListGP, SingleTaskGP
from botorch_rs.sampling.list_sampler import ListSampler
from botorch_rs.sampling.normal import SobolQMCNormalSampler

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


def data(device):
    X = torch.tensor([[0.04], [0.27], [0.56], [0.81]], dtype=torch.double, device=device)
    Y = torch.cat(((X * 5.3).sin(), (X * 3.1).cos()), -1)
    return X, Y


def build(X, Y, reference=False):
    if reference:
        from botorch.models import ModelListGP as ModelList
        from botorch.models import SingleTaskGP as GP
    else:
        ModelList, GP = ModelListGP, SingleTaskGP
    return ModelList(
        *[
            GP(
                X,
                Y[..., i : i + 1],
                outcome_transform=None,
                **({} if reference else {"backend": "torch"}),
            )
            for i in range(Y.shape[-1])
        ]
    )


def options(reference=False, fantasies=2):
    if reference:
        from botorch.sampling.list_sampler import ListSampler as List
        from botorch.sampling.normal import SobolQMCNormalSampler as Sampler
    else:
        List, Sampler = ListSampler, SobolQMCNormalSampler
    return dict(
        num_fantasies=fantasies,
        num_pareto=2,
        sampler=List(*[Sampler(torch.Size([fantasies]), seed=seed) for seed in (31, 41)]),
        inner_sampler=Sampler(torch.Size([8]), seed=53),
    )


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("mean", [False, True])
@pytest.mark.parametrize("log", [False, True])
@pytest.mark.parametrize("pending", [False, True])
def test_hvkg_reference_values(device, mean, log, pending):
    from botorch.acquisition.multi_objective.hypervolume_knowledge_gradient import (
        qHypervolumeKnowledgeGradient as Reference,
    )

    X, Y = data(device)
    args = dict(
        ref_point=X.new_tensor([-1.4, -1.4]),
        use_posterior_mean=mean,
        log=log,
        current_value=X.new_tensor(-2.0 if log else 0.1),
    )
    if pending:
        args["X_pending"] = X.new_tensor([[0.95]])
        args["X_pending_evaluation_mask"] = torch.tensor([[True, False]], device=device)
        args["X_evaluation_mask"] = torch.tensor([[False, True]], device=device)
    acq = qHypervolumeKnowledgeGradient(build(X, Y), **args, **options())
    ref = Reference(build(X, Y, True), **args, **options(True))
    query = X.new_tensor([[0.19], [0.15], [0.63], [0.33], [0.75]])
    actual, expected = acq(query), ref(query)
    torch.testing.assert_close(actual, expected, atol=2e-8, rtol=2e-8)
    assert isinstance(acq, qHypervolumeKnowledgeGradient)
    assert actual.device == query.device
    assert acq.get_augmented_q_batch_size(1) == 5
    torch.testing.assert_close(acq.extract_candidates(query), query[:1])


@pytest.mark.parametrize("device", DEVICES)
def test_decoupled_unobserved_output_retains_model(device):
    X, Y = data(device)
    model = build(X, Y)
    mask = torch.tensor([[True, False]], device=device)
    query = X.new_tensor([[0.2]])
    fantasy = model.fantasize(
        query, options()["sampler"], evaluation_mask=mask, propagate_grads=True
    )
    assert fantasy.models[1] is model.models[1]
    assert fantasy.models[0].train_inputs[0].shape[-2] == X.shape[-2] + 1
    expected = model.models[1].posterior(query)
    actual = fantasy.models[1].posterior(query)
    torch.testing.assert_close(actual.mean, expected.mean)
    torch.testing.assert_close(actual.variance, expected.variance)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("mean", [False, True])
def test_hvkg_full_graph_and_no_host_transfer(device, mean):
    X, Y = data(device)
    X.requires_grad_()
    Y.requires_grad_()
    query = X.new_tensor([[0.19], [0.15], [0.63], [0.33], [0.75]], requires_grad=True)
    reference = X.new_tensor([-1.4, -1.4], requires_grad=True)
    current = X.new_tensor(0.1, requires_grad=True)

    def make(a, b):
        return qHypervolumeKnowledgeGradient(
            build(a, b),
            ref_point=reference,
            current_value=current,
            use_posterior_mean=mean,
            **options(),
        )

    acq = make(X, Y)
    with (
        patch.object(torch.Tensor, "cpu", side_effect=AssertionError("host transfer")),
        patch.object(torch.Tensor, "numpy", side_effect=AssertionError("host transfer")),
        patch.object(torch.Tensor, "tolist", side_effect=AssertionError("host transfer")),
    ):
        leaves = (X, Y, query, reference, current, *acq.model.parameters())
        gradients = torch.autograd.grad(acq(query).sum(), leaves, create_graph=True)
        second = torch.autograd.grad(
            sum(g.square().sum() for g in gradients if g.requires_grad), leaves, allow_unused=True
        )
    assert all(torch.isfinite(g).all() for g in gradients)
    assert all(g is None or torch.isfinite(g).all() for g in second)
    assert torch.autograd.gradcheck(
        lambda a, b, c: make(a, b)(c), (X, Y, query), atol=2e-5, rtol=2e-4
    )
    assert torch.autograd.gradgradcheck(
        lambda a, b, c: make(a, b)(c), (X, Y, query), atol=3e-5, rtol=3e-4
    )


@pytest.mark.parametrize("device", DEVICES)
def test_hvkg_batch_split_preserves_each_fantasy_and_query_graph(device):
    X, Y = data(device)
    query = X.new_tensor(
        [[[0.19], [0.15], [0.63], [0.33], [0.75]], [[0.23], [0.21], [0.69], [0.39], [0.81]]]
    ).requires_grad_()
    actual, fantasies = _split_hvkg_fantasy_points(query, 2, 2)
    for i in range(2):
        torch.testing.assert_close(fantasies[i], query[:, 1 + 2 * i : 3 + 2 * i])
    torch.testing.assert_close(actual, query[:, :1])
    acq = qHypervolumeKnowledgeGradient(
        build(X, Y), ref_point=X.new_tensor([-1.4, -1.4]), **options()
    )
    together = acq(query)
    independent = torch.cat([acq(q) for q in query])
    torch.testing.assert_close(together, independent)
    assert torch.isfinite(torch.autograd.grad(together.sum(), query)[0]).all()


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("stochastic", [False, True])
def test_mf_hvkg_cost_projection_and_separate_sampler(device, stochastic):
    from botorch_rs.acquisition.cost_aware import InverseCostWeightedUtility
    from botorch_rs.models.cost import AffineFidelityCostModel

    X, Y = data(device)
    costs = (
        SingleTaskGP(X, 2 + Y[:, :1].square(), backend="torch", outcome_transform=None)
        if stochastic
        else AffineFidelityCostModel({0: 0.5}, fixed_cost=2)
    )
    utility = InverseCostWeightedUtility(costs, use_mean=not stochastic)

    def project(z):
        return 0.1 + 0.8 * z

    acq = qMultiFidelityHypervolumeKnowledgeGradient(
        build(X, Y),
        ref_point=X.new_tensor([-1.4, -1.4]),
        current_value=X.new_tensor(0.1),
        target_fidelities={0: 1.0},
        project=project,
        cost_aware_utility=utility,
        **options(),
    )
    query = X.new_tensor([[0.19], [0.15], [0.63], [0.33], [0.75]], requires_grad=True)
    value = acq(query)
    streams = [sampler.base_samples.clone() for sampler in acq.sampler.samplers]
    acq.cost_sampler(costs.posterior(query[:1]))
    for saved, sampler in zip(streams, acq.sampler.samplers, strict=True):
        torch.testing.assert_close(saved, sampler.base_samples)
    assert torch.isfinite(torch.autograd.grad(value.sum(), query, create_graph=True)[0]).all()
    base = _get_hv_value_function(acq.model, acq.ref_point, use_posterior_mean=True)
    torch.testing.assert_close(
        base(project(query[:2])),
        _get_hv_value_function(acq.model, acq.ref_point, project=project, use_posterior_mean=True)(
            query[:2]
        ),
    )


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("mean", [False, True])
def test_three_objectives_and_live_objective_weights(device, mean):
    from botorch.acquisition.multi_objective.hypervolume_knowledge_gradient import (
        qHypervolumeKnowledgeGradient as Reference,
    )
    from botorch.acquisition.multi_objective.objective import (
        WeightedMCMultiOutputObjective as ReferenceObjective,
    )
    from botorch.sampling.list_sampler import ListSampler as ReferenceListSampler
    from botorch.sampling.normal import SobolQMCNormalSampler as ReferenceSampler

    from botorch_rs.acquisition.multi_objective.objective import WeightedMCMultiOutputObjective

    X, Y = data(device)
    Y = torch.cat((Y, 0.5 + X.square()), -1)
    weights = X.new_tensor([0.7, 0.8, 0.9], requires_grad=True)
    kwargs = dict(ref_point=X.new_tensor([-1.4, -1.4, -0.5]), use_posterior_mean=mean)
    actual_options, reference_options = options(), options(True)
    actual_options["sampler"] = ListSampler(
        *[SobolQMCNormalSampler((2,), seed=s) for s in (31, 41, 61)]
    )
    reference_options["sampler"] = ReferenceListSampler(
        *[ReferenceSampler(torch.Size([2]), seed=s) for s in (31, 41, 61)]
    )
    acq = qHypervolumeKnowledgeGradient(
        build(X, Y), objective=WeightedMCMultiOutputObjective(weights), **kwargs, **actual_options
    )
    reference = Reference(
        build(X, Y, True), objective=ReferenceObjective(weights), **kwargs, **reference_options
    )
    query = X.new_tensor([[0.19], [0.15], [0.63], [0.33], [0.75]])
    value = acq(query)
    torch.testing.assert_close(value, reference(query), atol=2e-8, rtol=2e-8)
    gradient = torch.autograd.grad(value.sum(), weights, create_graph=True)[0]
    assert torch.isfinite(gradient).all()
    assert torch.isfinite(torch.autograd.grad(gradient.sum(), weights)[0]).all()


@pytest.mark.parametrize("device", DEVICES)
def test_joint_multioutput_cost_sampler_keeps_correlations(device):
    from botorch_rs.acquisition._torch_backend_hvkg import _CostListSampler

    X, Y = data(device)
    model = SingleTaskGP(X, Y + 3, backend="torch", outcome_transform=None)
    query = X.new_tensor([[0.2], [0.7]], requires_grad=True)
    posterior = model.posterior(query)
    sampler = _CostListSampler(*options()["sampler"].samplers)
    value = sampler(posterior)
    bases = torch.stack(
        [
            s.base_samples.squeeze(-1) if s.base_samples.ndim > 2 else s.base_samples
            for s in sampler.samplers
        ],
        -1,
    )
    expected = posterior.rsample_from_base_samples(sampler.sample_shape, bases)
    torch.testing.assert_close(value, expected)
    gradient = torch.autograd.grad(value.square().sum(), query, create_graph=True)[0]
    assert torch.isfinite(torch.autograd.grad(gradient.sum(), query)[0]).all()

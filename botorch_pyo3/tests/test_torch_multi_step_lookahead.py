"""Recursive device lookahead, synchronized trees and complete derivatives."""

from unittest.mock import patch

import pytest
import torch

from botorch_rs.acquisition.analytic import ExpectedImprovement, PosteriorMean
from botorch_rs.acquisition.monte_carlo import qExpectedImprovement
from botorch_rs.acquisition.multi_step_lookahead import make_best_f, qMultiStepLookahead
from botorch_rs.acquisition.objective import GenericMCObjective
from botorch_rs.models import SingleTaskGP
from botorch_rs.sampling.normal import SobolQMCNormalSampler

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


def data(device):
    X = torch.tensor([[0.04], [0.27], [0.56], [0.81]], dtype=torch.double, device=device)
    return X, (X * 5.3).sin()


def model(X, Y, reference=False):
    if reference:
        from botorch.models import SingleTaskGP as ReferenceGP

        return ReferenceGP(X, Y, outcome_transform=None)
    return SingleTaskGP(X, Y, outcome_transform=None, backend="torch")


def arguments(depth, kind, reference=False):
    if reference:
        from botorch.acquisition.analytic import ExpectedImprovement as EI
        from botorch.acquisition.analytic import PosteriorMean as Mean
        from botorch.acquisition.monte_carlo import qExpectedImprovement as qEI
        from botorch.acquisition.multi_step_lookahead import make_best_f as best
        from botorch.sampling.normal import SobolQMCNormalSampler as Sampler
    else:
        EI, Mean, qEI, best, Sampler = (
            ExpectedImprovement,
            PosteriorMean,
            qExpectedImprovement,
            make_best_f,
            SobolQMCNormalSampler,
        )
    stages = (
        [None] * depth + [Mean] if kind == "mean" else [EI if kind == "ei" else qEI] * (depth + 1)
    )
    return dict(
        batch_sizes=[1] * depth,
        samplers=[Sampler(torch.Size([2]), seed=31 + i * 10) for i in range(depth)],
        valfunc_cls=stages,
        valfunc_argfacs=(
            [None] * (depth + 1)
            if kind == "mean"
            else [lambda model, X: {"best_f": 0.1}] * (depth + 1)
            if kind == "ei"
            else [best] * (depth + 1)
        ),
        inner_mc_samples=[8] * (depth + 1),
    )


def synchronize_inner(acq, reference):
    from botorch.sampling.normal import SobolQMCNormalSampler as ReferenceSampler

    for i, sampler in enumerate(acq.inner_samplers):
        if sampler is not None:
            acq.inner_samplers[i] = SobolQMCNormalSampler((8,), seed=73 + i)
            reference.inner_samplers[i] = ReferenceSampler(torch.Size([8]), seed=73 + i)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("depth", [1, 2])
@pytest.mark.parametrize("kind", ["mean", "ei", "mc"])
@pytest.mark.parametrize("pending", [False, True])
def test_multi_step_values_match_reference(device, depth, kind, pending):
    from botorch.acquisition.multi_step_lookahead import qMultiStepLookahead as Reference

    X, Y = data(device)
    extra = {} if not pending else {"X_pending": X.new_tensor([[0.97]])}
    actual_args, reference_args = arguments(depth, kind), arguments(depth, kind, True)
    if pending and kind == "ei":
        # Pending points increase the initial q, which analytic EI rejects in
        # both implementations; later analytic stages still each use q=1.
        actual_args["valfunc_cls"][0] = reference_args["valfunc_cls"][0] = None
    acq = qMultiStepLookahead(model(X, Y), **actual_args, **extra)
    reference = Reference(model(X, Y, True), **reference_args, **extra)
    synchronize_inner(acq, reference)
    size = acq.get_augmented_q_batch_size(1)
    query = torch.linspace(0.13, 0.89, size, dtype=X.dtype, device=X.device).reshape(1, size, 1)
    query = torch.cat((query, query * 0.9 + 0.02), 0)
    actual, expected = acq(query), reference(query)
    torch.testing.assert_close(actual, expected, atol=3e-8, rtol=3e-7)
    assert isinstance(acq, qMultiStepLookahead)
    assert actual.shape == (2,) and actual.device == X.device
    torch.testing.assert_close(acq.extract_candidates(query), query[:, :1])


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("kind", ["mean", "mc"])
def test_multi_step_full_graph_without_host_transfers(device, kind):
    X, Y = data(device)
    X.requires_grad_()
    Y.requires_grad_()
    query = X.new_tensor([[0.19], [0.34], [0.78]], requires_grad=True)

    def make(a, b):
        acq = qMultiStepLookahead(model(a, b), **arguments(1, kind))
        for i, s in enumerate(acq.inner_samplers):
            if s is not None:
                acq.inner_samplers[i] = SobolQMCNormalSampler((8,), seed=73 + i)
        return acq

    acq = make(X, Y)
    with (
        patch.object(torch.Tensor, "cpu", side_effect=AssertionError("host transfer")),
        patch.object(torch.Tensor, "numpy", side_effect=AssertionError("host transfer")),
        patch.object(torch.Tensor, "tolist", side_effect=AssertionError("host transfer")),
    ):
        leaves = (X, Y, query, *acq.model.parameters())
        gradients = torch.autograd.grad(acq(query).sum(), leaves, create_graph=True)
        second = torch.autograd.grad(
            sum(g.square().sum() for g in gradients), leaves, allow_unused=True
        )
    assert all(torch.isfinite(g).all() for g in gradients)
    assert all(g is None or torch.isfinite(g).all() for g in second)
    assert torch.autograd.gradcheck(
        lambda a, b, c: make(a, b)(c), (X, Y, query), atol=3e-5, rtol=3e-4
    )
    assert torch.autograd.gradgradcheck(
        lambda a, b, c: make(a, b)(c), (X, Y, query), atol=4e-5, rtol=4e-4
    )


@pytest.mark.parametrize("device", DEVICES)
def test_tree_keeps_fantasies_separate_and_returns_induced_model(device):
    X, Y = data(device)
    acq = qMultiStepLookahead(
        model(X, Y),
        batch_sizes=[2, 1],
        samplers=[SobolQMCNormalSampler((2,), seed=31), SobolQMCNormalSampler((3,), seed=41)],
        valfunc_cls=[None, None, PosteriorMean],
        collapse_fantasy_base_samples=False,
    )
    query = (
        torch.linspace(0.1, 0.9, 11, dtype=X.dtype, device=X.device)
        .reshape(1, 11, 1)
        .expand(2, 11, 1)
        .clone()
        .requires_grad_()
    )
    tree = acq.get_multi_step_tree_input_representation(query)
    assert [tuple(t.shape) for t in tree] == [(2, 1, 1), (2, 2, 2, 1), (3, 2, 2, 1, 1)]
    for b in range(2):
        torch.testing.assert_close(tree[1][:, b].reshape(4, 1), query[b, 1:5])
        torch.testing.assert_close(tree[2][:, :, b].reshape(6, 1), query[b, 5:])
    fantasy = acq.get_induced_fantasy_model(query)
    assert fantasy.batch_shape == torch.Size([3, 2, 2])
    assert fantasy.train_inputs[0].shape[-2] == X.shape[-2] + 3
    assert tuple(acq.samplers[0].base_samples.shape) == (2, 1, 1)
    assert tuple(acq.samplers[1].base_samples.shape) == (3, 2, 1, 2)
    value = acq(query)
    assert torch.isfinite(torch.autograd.grad(value.sum(), query)[0]).all()


@pytest.mark.parametrize("device", DEVICES)
def test_live_sample_weights_and_fantasy_argument_factories(device):
    X, Y = data(device)
    weights = X.new_tensor([0.3, 0.7], requires_grad=True)
    sampler = SobolQMCNormalSampler((2,), seed=31)
    sampler.base_weights = weights
    seen = []

    def factory(model, X):
        seen.append((model.batch_shape, X.shape))
        return {"best_f": model._train_Y.amax(dim=-2).squeeze(-1) - 0.2}

    acq = qMultiStepLookahead(
        model(X, Y),
        batch_sizes=[1],
        samplers=[sampler],
        valfunc_cls=[None, ExpectedImprovement],
        valfunc_argfacs=[None, factory],
    )
    query = X.new_tensor([[0.19], [0.34], [0.78]], requires_grad=True)
    value = acq(query)
    assert seen == [(torch.Size([2, 1]), torch.Size([2, 1, 1, 1]))]
    assert torch.isfinite(torch.autograd.grad(value.sum(), weights, create_graph=True)[0]).all()
    assert torch.isfinite(torch.autograd.grad(value.sum(), query, retain_graph=True)[0]).all()


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("scalarize", [False, True])
def test_multioutput_transform_and_nonlinear_objective_reference(device, scalarize):
    from botorch.acquisition.analytic import PosteriorMean as ReferenceMean
    from botorch.acquisition.monte_carlo import qSimpleRegret as ReferenceRegret
    from botorch.acquisition.multi_step_lookahead import qMultiStepLookahead as Reference
    from botorch.acquisition.objective import GenericMCObjective as ReferenceObjective
    from botorch.acquisition.objective import ScalarizedPosteriorTransform as ReferenceTransform
    from botorch.models import ModelListGP as ReferenceList
    from botorch.sampling.normal import SobolQMCNormalSampler as ReferenceSampler

    from botorch_rs.acquisition.monte_carlo import qSimpleRegret
    from botorch_rs.acquisition.objective import ScalarizedPosteriorTransform
    from botorch_rs.models import ModelListGP

    X, Y = data(device)
    Y = torch.cat((Y, (X * 3.1).cos()), -1)
    weights = X.new_tensor([0.7, -0.2])

    def function(samples, X=None):
        return (samples @ weights).square()

    actual_kwargs = (
        dict(posterior_transform=ScalarizedPosteriorTransform(weights))
        if scalarize
        else dict(objective=GenericMCObjective(function))
    )
    reference_kwargs = (
        dict(posterior_transform=ReferenceTransform(weights))
        if scalarize
        else dict(objective=ReferenceObjective(function))
    )
    actual_model = ModelListGP(*[model(X, Y[:, i : i + 1]) for i in range(2)])
    reference_model = ReferenceList(*[model(X, Y[:, i : i + 1], True) for i in range(2)])
    acq = qMultiStepLookahead(
        actual_model,
        batch_sizes=[1],
        samplers=[SobolQMCNormalSampler((2,), seed=31)],
        valfunc_cls=[None, PosteriorMean if scalarize else qSimpleRegret],
        **actual_kwargs,
    )
    reference = Reference(
        reference_model,
        batch_sizes=[1],
        samplers=[ReferenceSampler(torch.Size([2]), seed=31)],
        valfunc_cls=[None, ReferenceMean if scalarize else ReferenceRegret],
        **reference_kwargs,
    )
    synchronize_inner(acq, reference)
    query = X.new_tensor([[0.19], [0.34], [0.78]], requires_grad=True)
    value = acq(query)
    torch.testing.assert_close(value, reference(query), atol=3e-8, rtol=3e-7)
    grad = torch.autograd.grad(value.sum(), query, create_graph=True)[0]
    assert torch.isfinite(torch.autograd.grad(grad.square().sum(), query)[0]).all()


@pytest.mark.parametrize("device", DEVICES)
def test_make_best_f_retains_raw_outcome_scale_and_fantasy_batch(device):
    from botorch_rs.models.transforms.outcome import Standardize

    X, Y = data(device)
    Y = Y * 12 + 7
    gp = SingleTaskGP(X, Y, backend="torch", outcome_transform=Standardize(1))
    fantasy = gp.fantasize(
        X.new_tensor([[[0.19]], [[0.37]]]),
        SobolQMCNormalSampler((3,), seed=31),
        propagate_grads=True,
    )
    best = make_best_f(fantasy, X)["best_f"]
    torch.testing.assert_close(best, fantasy._train_Y.amax(dim=-2).squeeze(-1))
    assert best.shape == (3, 2)
    value = ExpectedImprovement(fantasy, best)(
        X.new_tensor([[[[0.51]], [[0.73]]]]).expand(3, 2, 1, 1)
    )
    assert value.shape == (3, 2) and torch.isfinite(value).all()

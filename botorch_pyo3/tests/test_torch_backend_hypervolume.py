"""Device-resident hypervolume inclusion/exclusion and front derivatives."""

import importlib
import warnings

import pytest
import torch
from test_torch_backend_acquisition import FixedSampler, native_data

from botorch_rs.models import SingleTaskGP
from botorch_rs.utils.multi_objective.box_decompositions.non_dominated import (
    FastNondominatedPartitioning,
)

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


def acquisition(log, reference=False):
    prefix = "botorch" if reference else "botorch_rs"
    module = "logei" if log else "monte_carlo"
    name = "qLogExpectedHypervolumeImprovement" if log else "qExpectedHypervolumeImprovement"
    return getattr(importlib.import_module(f"{prefix}.acquisition.multi_objective.{module}"), name)


def data(device, outcomes):
    X, y = native_data(device)
    Y = torch.cat([0.2 * j + (y * (j + 1)).sin() for j in range(outcomes)], dim=-1)
    reference = torch.full((outcomes,), -0.3, device=device, dtype=torch.double)
    observed = torch.stack(
        [
            torch.linspace(
                0.1 + j * 0.1, 0.4 - j * 0.05, outcomes, device=device, dtype=torch.double
            )
            for j in range(3)
        ]
    )
    return X.requires_grad_(), Y.requires_grad_(), reference, observed


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("outcomes", [2, 3])
@pytest.mark.parametrize("q", [1, 2])
@pytest.mark.parametrize("log", [False, True])
def test_hypervolume_reference_device_and_full_graph(device, outcomes, q, log):
    from botorch.utils.multi_objective.box_decompositions.non_dominated import (
        FastNondominatedPartitioning as ReferencePartition,
    )

    X, Y, ref, observed = data(device, outcomes)
    model = SingleTaskGP(X, Y, backend="torch")
    partition = FastNondominatedPartitioning(ref, Y=observed)
    reference_partition = ReferencePartition(ref, Y=observed)
    query = torch.tensor([[0.21], [0.56]][:q], device=device, dtype=torch.double).requires_grad_()
    sampler = FixedSampler(q * outcomes, device)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        actual = acquisition(log)(model, ref, partition, sampler=sampler)
        expected = acquisition(log, True)(model, ref, reference_partition, sampler=sampler)
    value, reference_value = actual(query), expected(query)
    assert value.device == query.device
    torch.testing.assert_close(value, reference_value, rtol=2e-10, atol=2e-11)
    leaves = (query, X, Y, *model.parameters())
    first = torch.autograd.grad(value.sum(), leaves, create_graph=True, retain_graph=True)
    reference_first = torch.autograd.grad(
        reference_value.sum(), leaves, create_graph=True, retain_graph=True
    )
    for gradient, reference_gradient in zip(first, reference_first, strict=True):
        torch.testing.assert_close(gradient, reference_gradient, rtol=2e-7, atol=2e-8)
    second = torch.autograd.grad(sum(g.square().sum() for g in first), leaves, retain_graph=True)
    reference_second = torch.autograd.grad(sum(g.square().sum() for g in reference_first), leaves)
    for gradient, reference_gradient in zip(second, reference_second, strict=True):
        assert gradient.device == query.device and torch.isfinite(gradient).all()
        torch.testing.assert_close(gradient, reference_gradient, rtol=2e-6, atol=2e-7)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("log", [False, True])
def test_observed_front_and_reference_gradcheck(device, log):
    X, Y, ref, _ = data(device, 2)
    model = SingleTaskGP(X.detach(), Y.detach(), backend="torch")
    ref.requires_grad_()
    observed = torch.tensor(
        [[0.1, 0.55], [0.3, 0.3], [0.6, 0.05]], device=device, dtype=torch.double
    ).requires_grad_()
    query = torch.tensor([[0.21], [0.56]], device=device, dtype=torch.double).requires_grad_()

    def evaluate(front, reference, candidate):
        partition = FastNondominatedPartitioning(reference, Y=front)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return acquisition(log)(model, reference, partition, sampler=FixedSampler(4, device))(
                candidate
            )

    assert torch.autograd.gradcheck(evaluate, (observed, ref, query), fast_mode=True)
    assert torch.autograd.gradgradcheck(evaluate, (observed, ref, query), fast_mode=True)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("log", [False, True])
@pytest.mark.parametrize("fat", [False, True])
def test_constrained_objective_pending_hypervolume(device, log, fat):
    from botorch.acquisition.multi_objective.objective import (
        GenericMCMultiOutputObjective as ReferenceObjective,
    )
    from botorch.utils.multi_objective.box_decompositions.non_dominated import (
        FastNondominatedPartitioning as ReferencePartition,
    )

    from botorch_rs.acquisition.multi_objective.objective import GenericMCMultiOutputObjective

    X, Y, _, _ = data(device, 3)
    model = SingleTaskGP(X, Y, backend="torch")
    ref = torch.tensor([-0.3, -0.3], device=device, dtype=torch.double)
    observed = torch.tensor(
        [[0.1, 0.55], [0.3, 0.3], [0.6, 0.05]], device=device, dtype=torch.double
    )
    partition, reference_partition = (
        FastNondominatedPartitioning(ref, Y=observed),
        ReferencePartition(ref, Y=observed),
    )
    query = torch.tensor([[0.21], [0.56]], device=device, dtype=torch.double).requires_grad_()
    pending = torch.tensor([[0.81]], device=device, dtype=torch.double)
    weight = torch.tensor(0.05, device=device, dtype=torch.double, requires_grad=True)
    eta = torch.tensor([0.2], device=device, dtype=torch.double, requires_grad=True)

    def objective(samples, X=None):
        return torch.stack(
            (
                samples[..., 0] + weight * samples[..., 2],
                samples[..., 1].square() + 0.01 * X[..., 0],
            ),
            -1,
        )

    arguments = dict(
        sampler=FixedSampler(9, device),
        constraints=[lambda y: y[..., 2] - 0.5],
        eta=eta,
        fat=fat,
        X_pending=pending,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        actual = acquisition(log)(
            model, ref, partition, objective=GenericMCMultiOutputObjective(objective), **arguments
        )
        expected = acquisition(log, True)(
            model, ref, reference_partition, objective=ReferenceObjective(objective), **arguments
        )
    value, reference_value = actual(query), expected(query)
    torch.testing.assert_close(value, reference_value, rtol=2e-10, atol=2e-11)
    leaves = (query, X, Y, weight, eta, *model.parameters())
    first = torch.autograd.grad(value.sum(), leaves, create_graph=True, retain_graph=True)
    reference_first = torch.autograd.grad(
        reference_value.sum(), leaves, create_graph=True, retain_graph=True
    )
    for gradient, reference_gradient in zip(first, reference_first, strict=True):
        torch.testing.assert_close(gradient, reference_gradient, rtol=2e-7, atol=2e-8)
    second = torch.autograd.grad(sum(g.square().sum() for g in first), leaves, retain_graph=True)
    reference_second = torch.autograd.grad(sum(g.square().sum() for g in reference_first), leaves)
    for gradient, reference_gradient in zip(second, reference_second, strict=True):
        torch.testing.assert_close(gradient, reference_gradient, rtol=2e-6, atol=2e-7)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("log", [False, True])
def test_hypervolume_default_sampler_no_host_conversion(device, log, monkeypatch):
    import botorch_rs._arrays as arrays

    X, Y, ref, observed = data(device, 2)
    model = SingleTaskGP(X, Y, backend="torch")
    partition = FastNondominatedPartitioning(ref, Y=observed)
    query = torch.tensor([[0.21], [0.56]], device=device, dtype=torch.double).requires_grad_()

    def forbidden(*args, **kwargs):
        raise AssertionError("Hypervolume acquisition copied inference data to host")

    for name in ("cpu", "tolist", "numpy"):
        monkeypatch.setattr(torch.Tensor, name, forbidden)
    monkeypatch.setattr(arrays, "to_flat", forbidden)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        actual = acquisition(log)(model, ref, partition)
    value = actual(query)
    leaves = (query, X, Y, *model.parameters())
    first = torch.autograd.grad(value.sum(), leaves, create_graph=True)
    second = torch.autograd.grad(sum(g.square().sum() for g in first), leaves)
    assert all(g.device == query.device and torch.isfinite(g).all() for g in (*first, *second))

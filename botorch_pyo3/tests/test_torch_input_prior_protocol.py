"""Meaningful GPyTorch prior/constraint calls retain actual public leaves."""

# ruff: noqa: E402
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
gpytorch = pytest.importorskip("gpytorch")

from test_torch_input_models import DEVICES, _data

from botorch_rs.models import SingleTaskGP
from botorch_rs.models.transforms.input import (
    ChainedInputTransform,
    LearnedFeatureImputation,
    Normalize,
    Warp,
)
from botorch_rs.optim.utils import get_parameters_and_bounds, sample_all_priors
from botorch_rs.priors import LogNormalPrior


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("family", ["warp", "impute"])
def test_public_input_parameter_constraints_and_bounds_match_source(device, family):
    source = pytest.importorskip("botorch.models.transforms.input")
    X, _, _ = _data(device)
    if family == "warp":
        actual = Warp(3, [0, 2]).to(X)
        expected = source.Warp(3, [0, 2]).to(X)
    else:
        options = dict(
            feature_indices={0: [0, 2], 1: [1, 2]},
            d=3,
            bounds=X.new_tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]),
            dtype=X.dtype,
            device=X.device,
        )
        actual, expected = (
            LearnedFeatureImputation(**options),
            source.LearnedFeatureImputation(**options),
        )
    entries, references = (
        list(actual.named_parameters_and_constraints()),
        list(expected.named_parameters_and_constraints()),
    )
    assert [name for name, _, _ in entries] == [name for name, _, _ in references]
    for (name, value, constraint), (_, reference, source_constraint) in zip(
        entries, references, strict=True
    ):
        assert value is dict(actual.named_parameters())[name]
        assert constraint is actual.constraint_for_parameter_name(name)
        torch.testing.assert_close(value, reference)
        torch.testing.assert_close(constraint.lower_bound, source_constraint.lower_bound)
        torch.testing.assert_close(constraint.upper_bound, source_constraint.upper_bound)
    assert dict(actual.named_constraints()).keys() == dict(expected.named_constraints()).keys()
    assert len(list(actual.constraints())) == len(list(expected.constraints()))
    if family == "impute":
        for current in (actual, expected):
            current.register_constraint(
                "raw_imputation_values", current.raw_imputation_values_constraint, replace=False
            )
    values = {name: X.new_tensor(0.7) for name, _ in actual.named_parameters()}
    actual.initialize(**values)
    expected.initialize(**values)
    for name, value in actual.named_parameters():
        torch.testing.assert_close(value, dict(expected.named_parameters())[name])
    actual_params, actual_bounds = get_parameters_and_bounds(actual)
    source_params, source_bounds = pytest.importorskip(
        "botorch.optim.utils"
    ).get_parameters_and_bounds(expected)
    assert actual_params.keys() == source_params.keys()
    assert actual_bounds.keys() == source_bounds.keys()
    for name in actual_bounds:
        for value, reference in zip(actual_bounds[name], source_bounds[name], strict=True):
            torch.testing.assert_close(
                torch.as_tensor(value).to(X), torch.as_tensor(reference).to(X)
            )


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("prior_kind", ["port", "gpytorch"])
def test_warp_registered_prior_identity_density_sampling_and_model_graphs(device, prior_kind):
    X, Y, Q = (value.detach() for value in _data(device))
    prior = (
        LogNormalPrior(0, 0.3) if prior_kind == "port" else gpytorch.priors.LogNormalPrior(0, 0.3)
    )
    transform = Warp(3, [0, 2], concentration0_prior=prior).to(X)
    assert transform.concentration0_prior is prior
    named = list(transform.named_priors())
    assert len(named) == 1 and named[0][0] == "concentration0_prior"
    name, owner, density, closure, setter = named[0]
    assert owner is transform
    assert closure(owner) is transform.concentration0
    if prior_kind == "gpytorch":
        assert density is prior
    value = density.log_prob(closure(owner)).sum()
    gradient = torch.autograd.grad(value, transform.concentration0, create_graph=True)[0]
    second = torch.autograd.grad(gradient.sum(), transform.concentration0)[0]
    assert torch.isfinite(gradient).all() and torch.isfinite(second).all()
    leaf = transform.concentration0
    setter(owner, X.new_full(leaf.shape, 1.3))
    assert transform.concentration0 is leaf
    transform.sample_from_prior(name)
    assert transform.concentration0 is leaf
    sample_all_priors(transform, seed=17)
    assert transform.concentration0 is leaf and bool((leaf > 0).all())
    shared = ChainedInputTransform(left=transform, right=transform)
    assert len(list(shared.named_priors())) == 1
    # Use one transform per numerical pass; the chain above only checks alias
    # traversal. The model retains an actual supplied GPyTorch prior instance.
    model = SingleTaskGP(
        X,
        Y,
        input_transform=transform,
        outcome_transform=None,
        backend="torch" if prior_kind == "port" else "auto",
    )
    assert isinstance(model, torch.nn.Module)
    if prior_kind == "gpytorch":
        assert model.input_transform.concentration0_prior is prior
    posterior = model.posterior(Q)
    loss = posterior.mean.sin().sum() + model._log_prior()
    first = torch.autograd.grad(loss, tuple(model.parameters()), create_graph=True)
    second = torch.autograd.grad(sum(p.square().sum() for p in first), tuple(model.parameters()))
    assert all(torch.isfinite(p).all() for p in (*first, *second))


@pytest.mark.parametrize("device", DEVICES)
def test_warp_actual_source_prior_and_constraint_registration(device):
    source = pytest.importorskip("botorch.models.transforms.input")
    X, _, _ = _data(device)
    left = gpytorch.priors.LogNormalPrior(X.new_tensor(0.0), X.new_tensor(0.3))
    right = gpytorch.priors.LogNormalPrior(X.new_tensor(0.0), X.new_tensor(0.3))
    actual = Warp(3, [0, 2], concentration0_prior=left).to(X)
    expected = source.Warp(3, [0, 2], concentration0_prior=right).to(X)
    assert set(actual.state_dict()) == set(expected.state_dict())
    for name, value in actual.state_dict().items():
        torch.testing.assert_close(value, expected.state_dict()[name])
    actual.initialize(concentration0=1.3, concentration1=0.8)
    expected.initialize(concentration0=1.3, concentration1=0.8)
    actual_prior = next(actual.named_priors())
    source_prior = next(expected.named_priors())
    av = actual_prior[2].log_prob(actual_prior[3](actual)).sum()
    ev = source_prior[2].log_prob(source_prior[3](expected)).sum()
    torch.testing.assert_close(av, ev)
    ag = torch.autograd.grad(av, actual.concentration0, create_graph=True)[0]
    eg = torch.autograd.grad(ev, expected.concentration0, create_graph=True)[0]
    torch.testing.assert_close(ag, eg)
    torch.testing.assert_close(
        torch.autograd.grad(ag.square().sum(), actual.concentration0)[0],
        torch.autograd.grad(eg.square().sum(), expected.concentration0)[0],
    )
    replacement = gpytorch.constraints.GreaterThan(1e-3, transform=None).to(X)
    actual.register_constraint("concentration0", replacement)
    assert actual.constraint_for_parameter_name("concentration0") is replacement
    prior = gpytorch.priors.NormalPrior(X.new_tensor(0.0), X.new_tensor(1.0))
    actual.register_prior(
        "log_concentration_prior", prior, lambda owner: owner.concentration0.log()
    )
    assert (
        dict((name, value) for name, _, value, _, _ in actual.named_priors())[
            "log_concentration_prior"
        ]
        is prior
    )


def test_public_warp_initialize_updates_bound_native_owner_without_replacing_leaf():
    X, Y, _ = (value.detach() for value in _data("cpu"))
    transform = Warp(3, [0, 2])
    leaf = transform.concentration0
    model = SingleTaskGP(X, Y, input_transform=transform, outcome_transform=None, backend="native")
    transform.initialize(concentration0=1.3)
    assert transform.concentration0 is leaf
    torch.testing.assert_close(leaf, torch.full_like(leaf, 1.3))
    torch.testing.assert_close(model.input_transform.concentration0, leaf)


@pytest.mark.parametrize("device", DEVICES)
def test_custom_input_call_preserves_source_super_call_and_hooks(device):
    source = pytest.importorskip("botorch.models.transforms.input")

    class Custom(Normalize):
        def __call__(self, X):
            return super().__call__(X) + 0.07

    class Source(source.Normalize):
        def __call__(self, X):
            return super().__call__(X) + 0.07

    X, _, _ = _data(device)
    bounds = X.new_tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
    actual, expected = Custom(3, bounds=bounds), Source(3, bounds=bounds)
    actual_events, source_events = [], []
    actual.register_forward_hook(lambda owner, args, result: actual_events.append(result))
    expected.register_forward_hook(lambda owner, args, result: source_events.append(result))
    result, reference = actual(X), expected(X)
    torch.testing.assert_close(result, reference)
    assert len(actual_events) == len(source_events) == 1
    torch.testing.assert_close(actual_events[0], source_events[0])
    for value in (result, reference):
        first = torch.autograd.grad(value.sin().sum(), X, create_graph=True, retain_graph=True)[0]
        assert torch.isfinite(
            torch.autograd.grad(first.square().sum(), X, retain_graph=True)[0]
        ).all()

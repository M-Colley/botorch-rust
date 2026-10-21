"""Persistent SVGP leaves and genuine native full-data / minibatch ELBO graphs."""

from __future__ import annotations

import copy

import pytest

from helpers import importorskip_botorch

importorskip_botorch()
torch = pytest.importorskip("torch")

from approximate_helpers import (  # noqa: E402
    CASES,
    build_pair,
    elbo_pair,
    set_random_parameters,
    strategy,
)

from botorch_rs.fit import fit_gpytorch_mll  # noqa: E402
from botorch_rs.models import SingleTaskVariationalGP  # noqa: E402
from botorch_rs.optim.fit import (  # noqa: E402
    fit_gpytorch_mll_scipy,
    fit_gpytorch_mll_torch,
)

pytestmark = pytest.mark.filterwarnings("ignore")
CONFIGURATIONS = [CASES[index] for index in [0, 2, 3, 4, 5, 6, 8]]


def data(model, minibatch):
    x, target = model.model.train_inputs[0], model.model.train_targets
    return (x[::3], target[::3]) if minibatch else (x, target)


@pytest.mark.parametrize("case", CONFIGURATIONS, ids=lambda case: case.name)
@pytest.mark.parametrize("minibatch", [False, True])
@pytest.mark.parametrize("combine", [False, True])
def test_native_elbo_graph_matches_official_parameters(case, minibatch, combine):
    reference, model, _, _ = build_pair(case)
    set_random_parameters(reference, model, 43)
    expected_mll, mll = elbo_pair(reference, model, 31, beta=0.7, combine=combine)
    x, y = data(model, minibatch)
    expected_x, expected_y = data(reference, minibatch)
    actual = mll(model.model(x), y)
    expected = expected_mll(reference.model(expected_x), expected_y)
    if combine:
        actual, expected = (actual,), (expected,)
        weights = [-0.6]
    else:
        weights = [0.3, -0.7, 1.4]
    for value, want in zip(actual, expected, strict=True):
        torch.testing.assert_close(value, want, rtol=3e-8, atol=3e-9)
    objective = sum(weight * value for weight, value in zip(weights, actual, strict=True))
    expected_objective = sum(
        weight * value for weight, value in zip(weights, expected, strict=True)
    )
    leaves, expected_leaves = tuple(mll.parameters()), tuple(expected_mll.parameters())
    gradients = torch.autograd.grad(objective, leaves)
    expected_gradients = torch.autograd.grad(
        expected_objective, expected_leaves, allow_unused=True, materialize_grads=True
    )
    assert list(dict(mll.named_parameters())) == list(dict(expected_mll.named_parameters()))
    for gradient, want in zip(gradients, expected_gradients, strict=True):
        torch.testing.assert_close(gradient, want, rtol=3e-7, atol=3e-8)


@pytest.mark.parametrize("outputs", [1, 2])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_variational_parameter_views_are_persistent_leaves(outputs, dtype):
    x = torch.tensor([[0.1], [0.4], [0.7], [0.9]], dtype=dtype)
    y = torch.sin(x * 4).expand(4, outputs)
    model = SingleTaskVariationalGP(x, y, inducing_points=x[:2])
    entries = dict(model.named_parameters())
    again = dict(model.named_parameters())
    assert entries.keys() == again.keys()
    for name, leaf in entries.items():
        assert leaf is again[name]
        assert isinstance(leaf, torch.nn.Parameter) and leaf.is_leaf and leaf.requires_grad
        assert leaf.dtype == dtype and leaf.device.type == "cpu"
    vs = strategy(model)
    prefix = "model.variational_strategy."
    if outputs > 1:
        prefix += "base_variational_strategy."
        assert model.likelihood.raw_task_noises is entries["likelihood.raw_task_noises"]
        assert model.likelihood.raw_noise is entries["likelihood.raw_noise"]
    else:
        assert model.likelihood.raw_noise is entries["likelihood.noise_covar.raw_noise"]
    assert vs.inducing_points is entries[prefix + "inducing_points"]
    assert (
        vs._variational_distribution.variational_mean
        is entries[prefix + "_variational_distribution.variational_mean"]
    )
    assert (
        vs._variational_distribution.chol_variational_covar
        is entries[prefix + "_variational_distribution.chol_variational_covar"]
    )
    model.model(x)
    assert int(vs.variational_params_initialized) == 1
    assert all(leaf is dict(model.named_parameters())[name] for name, leaf in entries.items())
    raw = torch.tensor(model._rs.raw_parameters(), dtype=dtype)
    ordered = torch.cat([leaf.detach().reshape(-1) for _, leaf in model._named_raw()])
    torch.testing.assert_close(raw, ordered, rtol=0, atol=0)


@pytest.mark.parametrize("term", [0, 1, 2])
def test_individual_elbo_terms_have_independent_native_gradients(term):
    reference, model, _, _ = build_pair(CASES[3])
    set_random_parameters(reference, model, 21)
    expected_mll, mll = elbo_pair(reference, model, 17, beta=0.4, combine=False)
    actual = mll()[term]
    x, y = data(reference, False)
    expected = expected_mll(reference.model(x), y)[term]
    gradients = torch.autograd.grad(actual, tuple(mll.parameters()))
    want = torch.autograd.grad(
        expected, tuple(expected_mll.parameters()), allow_unused=True, materialize_grads=True
    )
    for gradient, expected_gradient in zip(gradients, want, strict=True):
        torch.testing.assert_close(gradient, expected_gradient, rtol=3e-7, atol=3e-8)


def test_saved_elbo_graph_snapshots_parameters_and_minibatch_data():
    reference, model, _, _ = build_pair(CASES[3])
    set_random_parameters(reference, model, 5)
    _, mll = elbo_pair(reference, model, 19)
    x, y = (value.clone() for value in data(model, True))
    value = mll(model.model(x), y)
    leaves = tuple(mll.parameters())
    expected = torch.autograd.grad(value, leaves, retain_graph=True)
    x.add_(0.2)
    y.add_(2.0)
    model.model.mean_module.constant = 0.4
    tuple(model.parameters())
    with torch.no_grad():
        strategy(model)._variational_distribution.variational_mean.add_(0.2)
    assert not torch.equal(mll(), value)
    actual = torch.autograd.grad(value, leaves)
    for gradient, want in zip(actual, expected, strict=True):
        torch.testing.assert_close(gradient, want, rtol=0, atol=0)


@pytest.mark.parametrize("minibatch", [False, True])
def test_external_torch_adam_trains_native_elbo_like_official(minibatch):
    reference, model, _, _ = build_pair(CASES[2])
    set_random_parameters(reference, model, 23)
    expected_mll, mll = elbo_pair(reference, model, 20, beta=0.6)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
    expected_optimizer = torch.optim.Adam(reference.parameters(), lr=0.003)
    losses = []
    for _ in range(5):
        optimizer.zero_grad()
        expected_optimizer.zero_grad()
        x, y = data(model, minibatch)
        rx, ry = data(reference, minibatch)
        loss, want = -mll(model.model(x), y), -expected_mll(reference.model(rx), ry)
        torch.testing.assert_close(loss, want, rtol=3e-8, atol=3e-9)
        losses.append(loss.item())
        loss.backward()
        want.backward()
        optimizer.step()
        expected_optimizer.step()
    assert losses[-1] < losses[0]
    for leaf, want in zip(model.parameters(), reference.parameters(), strict=True):
        torch.testing.assert_close(leaf, want, rtol=3e-7, atol=3e-8)


@pytest.mark.parametrize("optimizer", ["scipy", "adam", "default"])
def test_native_fitting_keeps_frozen_leaves_and_pulls_updates(optimizer):
    reference, model, _, _ = build_pair(CASES[2])
    set_random_parameters(reference, model, 13)
    _, mll = elbo_pair(reference, model, 12)
    leaves = dict(mll.named_parameters())
    frozen = next(value for name, value in leaves.items() if name.endswith("raw_lengthscale"))
    frozen.requires_grad_(False)
    before = {name: value.detach().clone() for name, value in leaves.items()}
    if optimizer == "scipy":
        fit_gpytorch_mll_scipy(mll, options={"maxiter": 3})
    elif optimizer == "adam":
        fit_gpytorch_mll_torch(mll, step_limit=3)
    else:
        fit_gpytorch_mll(mll, optimizer_kwargs={"options": {"maxiter": 3}}, max_attempts=1)
    # Check the held leaves directly, before requesting a fresh parameter view.
    assert any(not torch.equal(value, before[name]) for name, value in leaves.items())
    for name, value in leaves.items():
        if value is frozen:
            torch.testing.assert_close(value, before[name], rtol=0, atol=0)
    assert all(value is dict(mll.named_parameters())[name] for name, value in leaves.items())
    assert not frozen.requires_grad


def test_variational_scipy_parameter_selection_and_explicit_bounds():
    reference, model, _, _ = build_pair(CASES[2])
    set_random_parameters(reference, model, 13)
    _, mll = elbo_pair(reference, model, 12)
    leaves = dict(mll.named_parameters())
    name = next(name for name in leaves if name.endswith("variational_mean"))
    before = {key: value.detach().clone() for key, value in leaves.items()}
    fit_gpytorch_mll_scipy(
        mll,
        parameters={name: leaves[name]},
        bounds={name: (-0.05, 0.05)},
        options={"maxiter": 3},
    )
    assert (leaves[name] >= -0.05).all() and (leaves[name] <= 0.05).all()
    assert not torch.equal(leaves[name], before[name])
    for key, leaf in leaves.items():
        if key != name:
            torch.testing.assert_close(leaf, before[key], rtol=0, atol=0)


def test_variational_no_grad_frozen_and_deepcopy_protocol():
    reference, model, _, _ = build_pair(CASES[0])
    set_random_parameters(reference, model, 7)
    _, mll = elbo_pair(reference, model, 12)
    leaves = dict(mll.named_parameters())
    first = next(iter(leaves.values()))
    first.requires_grad_(False)
    copied = copy.deepcopy(model)
    assert dict(copied._named_raw()).keys() == leaves.keys()
    for name, leaf in copied._named_raw():
        assert leaf is not leaves[name]
        assert leaf.requires_grad == leaves[name].requires_grad
        torch.testing.assert_close(leaf, leaves[name])
    with torch.no_grad():
        assert not mll().requires_grad
    for leaf in leaves.values():
        leaf.requires_grad_(False)
    assert not mll().requires_grad


def test_conflicting_native_and_torch_variational_edits_fail_explicitly():
    reference, model, _, _ = build_pair(CASES[0])
    set_random_parameters(reference, model, 7)
    leaf = model.model.mean_module.raw_constant
    with torch.no_grad():
        leaf.add_(0.2)
    raw = model._rs.raw_parameters()
    raw[-1] += 0.1
    model._rs.set_raw_parameters(raw)
    with pytest.raises(RuntimeError, match="both modified"):
        tuple(model.parameters())


@pytest.mark.parametrize("case", [CASES[0], CASES[2]], ids=lambda case: case.name)
def test_variational_prior_sampling_synchronizes_existing_parameter_leaves(case):
    from botorch_rs.optim.utils.model_utils import sample_all_priors

    _, model, _, _ = build_pair(case)
    expected = copy.deepcopy(model)
    leaves = dict(model._named_raw())
    with torch.no_grad():
        model.model.mean_module.raw_constant.fill_(0.45)
    expected.model.mean_module.constant = 0.45
    sample_all_priors(model, seed=37)
    sample_all_priors(expected, seed=37)
    raw = model._rs.raw_parameters()
    offset = 0
    for name, count in model._rs.parameter_names():
        torch.testing.assert_close(
            leaves[name].detach().reshape(-1),
            torch.tensor(raw[offset : offset + count], dtype=leaves[name].dtype),
            rtol=0,
            atol=0,
        )
        offset += count
    assert raw == expected._rs.raw_parameters()

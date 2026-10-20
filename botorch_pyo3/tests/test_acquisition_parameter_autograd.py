"""One-pass native acquisition tapes and persistent model parameter adjoints."""

from __future__ import annotations

import pytest

from helpers import importorskip_botorch

importorskip_botorch()
torch = pytest.importorskip("torch")

import botorch.acquisition.analytic as bt_analytic  # noqa: E402
import botorch.acquisition.monte_carlo as bt_mc  # noqa: E402
from botorch.acquisition.logei import qLogExpectedImprovement as BtLogEI  # noqa: E402
from botorch.acquisition.objective import ScalarizedPosteriorTransform as BtScalarize  # noqa: E402
from botorch.settings import propagate_grads  # noqa: E402
from gpytorch.settings import fast_pred_var  # noqa: E402
from so_acquisition_helpers import sampler_pair  # noqa: E402
from test_posterior_parameter_gradients import model_pair  # noqa: E402

import botorch_rs.acquisition.analytic as analytic  # noqa: E402
import botorch_rs.acquisition.monte_carlo as mc  # noqa: E402
from botorch_rs.acquisition.logei import qLogExpectedImprovement  # noqa: E402
from botorch_rs.acquisition.objective import ScalarizedPosteriorTransform  # noqa: E402
from botorch_rs.exceptions.errors import UnsupportedError  # noqa: E402

ANALYTIC = [
    "ExpectedImprovement",
    "LogExpectedImprovement",
    "ProbabilityOfImprovement",
    "LogProbabilityOfImprovement",
    "UpperConfidenceBound",
    "PosteriorMean",
    "PosteriorStandardDeviation",
]
MONTE_CARLO = [
    "qExpectedImprovement",
    "qLogExpectedImprovement",
    "qProbabilityOfImprovement",
    "qSimpleRegret",
    "qUpperConfidenceBound",
    "qLowerConfidenceBound",
    "qPosteriorStandardDeviation",
]


@pytest.mark.parametrize("depth", [1, 3])
@pytest.mark.parametrize("outputs", [1, 2])
def test_exact_gp_bnn_tape_matches_official_marginal_and_parameter_gradients(depth, outputs):
    from botorch.models import SingleTaskGP as BtGP
    from botorch.models.kernels.infinite_width_bnn import InfiniteWidthBNNKernel as BtBNN

    from botorch_rs.models import SingleTaskGP
    from botorch_rs.models.kernels.infinite_width_bnn import InfiniteWidthBNNKernel

    train = torch.tensor([[0.1, 0.2], [0.3, 0.8], [0.6, 0.4], [0.9, 0.7]], dtype=torch.double)
    targets = (train[:, :1] * 4).sin()
    if outputs == 2:
        targets = torch.cat([targets, train[:, 1:].cos()], dim=-1)
    batch = torch.Size([outputs]) if outputs > 1 else torch.Size()
    noise = torch.full_like(targets, 0.03)
    model = SingleTaskGP(
        train,
        targets,
        noise,
        outcome_transform=None,
        covar_module=InfiniteWidthBNNKernel(depth=depth, batch_shape=batch),
    )
    reference = BtGP(
        train,
        targets,
        noise,
        outcome_transform=None,
        covar_module=BtBNN(depth=depth, batch_shape=batch).double(),
    )
    kwargs, expected_kwargs = {}, {}
    if outputs == 2:
        weights = torch.tensor([0.3, -0.2], dtype=torch.double)
        kwargs["posterior_transform"] = ScalarizedPosteriorTransform(weights)
        expected_kwargs["posterior_transform"] = BtScalarize(weights)
    best = torch.tensor(0.4, dtype=torch.double)
    acquisition = analytic.LogExpectedImprovement(model, best_f=best, **kwargs)
    expected_acquisition = bt_analytic.LogExpectedImprovement(
        reference, best_f=best, **expected_kwargs
    )
    query = torch.tensor([[[0.2, 0.6]], [[0.7, 0.3]]], dtype=torch.double, requires_grad=True)
    expected_query = query.detach().clone().requires_grad_()
    actual = acquisition(query)
    with propagate_grads(True), fast_pred_var(False):
        expected = expected_acquisition(expected_query)
    torch.testing.assert_close(actual, expected, rtol=1e-11, atol=1e-11)
    upstream = torch.tensor([-0.3, 0.7], dtype=torch.double)
    gradients = torch.autograd.grad((actual * upstream).sum(), (*model.parameters(), query))
    expected_gradients = torch.autograd.grad(
        (expected * upstream).sum(), (*reference.parameters(), expected_query)
    )
    for gradient, want in zip(gradients, expected_gradients, strict=True):
        torch.testing.assert_close(gradient, want, rtol=2e-10, atol=2e-10)


@pytest.mark.parametrize("name", ANALYTIC + MONTE_CARLO)
@pytest.mark.parametrize("scalar", [False, True])
def test_common_acquisition_parameter_and_query_gradients(name, scalar):
    model, reference = model_pair(outputs=2 if scalar else 1, tree=not scalar)
    is_mc = name.startswith("q")
    q = 2 if is_mc else 1
    query = (
        torch.linspace(0.15, 0.85, 3 * q * 2, dtype=torch.double).reshape(3, q, 2).requires_grad_()
    )
    expected_query = query.detach().clone().requires_grad_()
    kwargs, expected_kwargs = {}, {}
    if "Improvement" in name:
        kwargs["best_f"] = expected_kwargs["best_f"] = torch.tensor(0.4, dtype=torch.double)
    if "ConfidenceBound" in name:
        kwargs["beta"] = expected_kwargs["beta"] = 0.2
    if scalar:
        w = torch.tensor([0.3, -0.6], dtype=torch.double)
        kwargs["posterior_transform"] = ScalarizedPosteriorTransform(w, offset=0.2)
        expected_kwargs["posterior_transform"] = BtScalarize(w, offset=0.2)
    if is_mc:
        expected_sampler, sampler = sampler_pair(
            16, (1, q) if scalar or model.num_outputs == 1 else (1, q, 2), 812
        )
        kwargs["sampler"], expected_kwargs["sampler"] = sampler, expected_sampler
    actual_cls = (
        qLogExpectedImprovement
        if name == "qLogExpectedImprovement"
        else getattr(mc if is_mc else analytic, name)
    )
    expected_cls = (
        BtLogEI
        if name == "qLogExpectedImprovement"
        else getattr(bt_mc if is_mc else bt_analytic, name)
    )
    acquisition, expected_acquisition = (
        actual_cls(model, **kwargs),
        expected_cls(reference, **expected_kwargs),
    )
    with propagate_grads(True), fast_pred_var(False):
        expected = expected_acquisition(expected_query)
    actual = acquisition(query)
    torch.testing.assert_close(actual, expected, rtol=2e-6, atol=2e-7)
    upstream = torch.tensor([-0.3, 0.0, 0.7], dtype=torch.double)
    actual_gradients = torch.autograd.grad((actual * upstream).sum(), (*model.parameters(), query))
    expected_gradients = torch.autograd.grad(
        (expected * upstream).sum(),
        (*reference.parameters(), expected_query),
        allow_unused=True,
        materialize_grads=True,
    )
    for gradient, expected_gradient in zip(actual_gradients, expected_gradients, strict=True):
        torch.testing.assert_close(gradient, expected_gradient, rtol=3e-6, atol=3e-7)


def test_acquisition_only_parameter_inputs_receive_graph_without_query_grad():
    model, _ = model_pair()
    acquisition = analytic.PosteriorMean(model)
    query = torch.tensor([[[0.2, 0.6]], [[0.7, 0.3]]], dtype=torch.double)
    acquisition(query).sum().backward()
    assert all(p.grad is not None for p in model.parameters())


@pytest.mark.parametrize("edit", ["leaf", "native", "target"])
def test_changed_model_requires_reconstruction_but_old_backward_snapshot_works(edit):
    model, _ = model_pair()
    acquisition = analytic.LogExpectedImprovement(model, best_f=0.4)
    query = torch.tensor([[[0.2, 0.6]]], dtype=torch.double)
    original = acquisition(query)
    expected = torch.autograd.grad(original.sum(), tuple(model.parameters()), retain_graph=True)
    if edit == "leaf":
        with torch.no_grad():
            model.mean_module.raw_constant.add_(0.1)
    elif edit == "native":
        model.mean_module.constant = 0.2
    else:
        model.set_train_data(targets=model.train_targets + 0.1)
    with pytest.raises(RuntimeError, match="Reconstruct the acquisition"):
        acquisition(query)
    actual = torch.autograd.grad(original.sum(), tuple(model.parameters()))
    for result, want in zip(actual, expected, strict=True):
        torch.testing.assert_close(result, want)
    assert torch.isfinite(analytic.LogExpectedImprovement(model, best_f=0.4)(query)).all()


def test_unsupported_joint_analytic_parameter_backward_preserves_query_only():
    model, _ = model_pair()
    acquisition = analytic.qAnalyticProbabilityOfImprovement(model, best_f=0.4)
    query = torch.tensor([[[0.2, 0.6]]], dtype=torch.double, requires_grad=True)
    value = acquisition(query)
    assert torch.isfinite(torch.autograd.grad(value.sum(), query, retain_graph=True)[0]).all()
    with pytest.raises(UnsupportedError, match="does not support automatic model-parameter"):
        value.sum().backward()


def test_query_only_backward_does_not_contract_parameter_tape(monkeypatch):
    from botorch_rs.acquisition import _parameter_autograd as bridge

    model, _ = model_pair()
    acquisition = analytic.PosteriorMean(model)
    query = torch.tensor([[[0.2, 0.6]]], dtype=torch.double, requires_grad=True)
    original = bridge.evaluate

    def poisoned(acq, points, snapshot):
        # An error on the parameter-only branch must be pruned by grad(..., X).
        snapshot.error = "parameter branch was unexpectedly evaluated"
        return original(acq, points, snapshot)

    monkeypatch.setattr(bridge, "evaluate", poisoned)
    value = acquisition(query)
    assert torch.isfinite(torch.autograd.grad(value.sum(), query)[0]).all()


@pytest.mark.parametrize("alias", [False, True])
def test_model_list_scalarization_and_alias_gradients(alias):
    from botorch.models import ModelListGP as BtList

    from botorch_rs.models import ModelListGP

    first, ref_first = model_pair()
    second, ref_second = (first, ref_first) if alias else model_pair(fixed=True)
    model, reference = ModelListGP(first, second), BtList(ref_first, ref_second)
    weights = torch.tensor([0.3, -0.6], dtype=torch.double)
    expected_sampler, sampler = sampler_pair(16, (1, 2), 812)
    acquisition = qLogExpectedImprovement(
        model,
        best_f=0.4,
        sampler=sampler,
        posterior_transform=ScalarizedPosteriorTransform(weights),
    )
    expected_acquisition = BtLogEI(
        reference, best_f=0.4, sampler=expected_sampler, posterior_transform=BtScalarize(weights)
    )
    query = torch.tensor([[[0.2, 0.6], [0.7, 0.3]], [[0.4, 0.2], [0.6, 0.5]]], dtype=torch.double)
    actual = acquisition(query)
    with propagate_grads(True), fast_pred_var(False):
        expected = expected_acquisition(query)
    upstream = torch.tensor([-0.3, 0.7], dtype=torch.double)
    gradients = torch.autograd.grad((actual * upstream).sum(), tuple(model.parameters()))
    expected_gradients = torch.autograd.grad(
        (expected * upstream).sum(), tuple(reference.parameters())
    )
    for result, want in zip(gradients, expected_gradients, strict=True):
        torch.testing.assert_close(result, want, rtol=3e-6, atol=3e-7)


def test_mc_objective_direct_input_and_constraint_bars_are_preserved():
    from botorch.acquisition.objective import GenericMCObjective as BtObjective

    from botorch_rs.acquisition.objective import GenericMCObjective

    model, reference = model_pair()

    def objective(samples, X=None):
        return samples[..., 0] + 0.2 * X[..., 0]

    def constraint(samples):
        return samples[..., 0] - 0.6

    expected_sampler, sampler = sampler_pair(16, (1, 2), 812)
    actual_acq = qLogExpectedImprovement(
        model,
        best_f=0.4,
        sampler=sampler,
        objective=GenericMCObjective(objective),
        constraints=[constraint],
    )
    expected_acq = BtLogEI(
        reference,
        best_f=0.4,
        sampler=expected_sampler,
        objective=BtObjective(objective),
        constraints=[constraint],
    )
    query = torch.tensor([[[0.2, 0.6], [0.7, 0.3]]], dtype=torch.double, requires_grad=True)
    expected_query = query.detach().clone().requires_grad_()
    actual = actual_acq(query)
    with propagate_grads(True), fast_pred_var(False):
        expected = expected_acq(expected_query)
    torch.testing.assert_close(actual, expected, rtol=3e-6, atol=3e-7)
    gradients = torch.autograd.grad(actual.sum(), (*model.parameters(), query))
    expected_gradients = torch.autograd.grad(
        expected.sum(), (*reference.parameters(), expected_query)
    )
    for result, want in zip(gradients, expected_gradients, strict=True):
        torch.testing.assert_close(result, want, rtol=3e-6, atol=3e-7)


@pytest.mark.parametrize("family", ["analytic", "mc"])
@pytest.mark.parametrize("transform", ["warp", "log"])
def test_transformed_training_and_outcome_parameter_graphs(family, transform):
    if transform == "warp":
        model, reference = model_pair(tree=True, warp=True)
    else:
        from botorch.models import SingleTaskGP as BtGP
        from botorch.models.transforms.outcome import Log as BtLog

        from botorch_rs.models import SingleTaskGP
        from botorch_rs.models.transforms.outcome import Log

        source, _ = model_pair()
        x = source.train_inputs[0]
        y = torch.exp(source.train_targets).unsqueeze(-1)
        model, reference = (
            SingleTaskGP(x, y, outcome_transform=Log()),
            BtGP(x, y, outcome_transform=BtLog()),
        )
    kwargs, expected_kwargs = {}, {}
    if family == "mc":
        q = 2
        expected_sampler, sampler = sampler_pair(16, (1, q), 812)
        kwargs.update(best_f=0.4, sampler=sampler)
        expected_kwargs.update(best_f=0.4, sampler=expected_sampler)
        actual_cls, expected_cls = qLogExpectedImprovement, BtLogEI
    else:
        q = 1
        actual_cls, expected_cls = analytic.PosteriorMean, bt_analytic.PosteriorMean
    acquisition, expected_acquisition = (
        actual_cls(model, **kwargs),
        expected_cls(reference, **expected_kwargs),
    )
    query = torch.tensor([[[0.2, 0.6], [0.7, 0.3]][:q]], dtype=torch.double, requires_grad=True)
    expected_query = query.detach().clone().requires_grad_()
    actual = acquisition(query)
    with propagate_grads(True), fast_pred_var(False):
        if transform == "warp":
            # GPyTorch's eval transition normally caches transformed inputs
            # under no_grad; rebuild them for the full training-input chain.
            reference.eval()
            reference.set_train_data(
                inputs=reference.input_transform(reference._original_train_inputs), strict=False
            )
        expected = expected_acquisition(expected_query)
    torch.testing.assert_close(actual, expected, rtol=3e-6, atol=3e-7)
    actual_gradients = torch.autograd.grad(actual.sum(), (*model.parameters(), query))
    expected_gradients = torch.autograd.grad(
        expected.sum(), (*reference.parameters(), expected_query)
    )
    for result, want in zip(actual_gradients, expected_gradients, strict=True):
        torch.testing.assert_close(result, want, rtol=3e-6, atol=3e-7)


def test_mc_pending_points_contribute_parameters_without_query_gradients():
    model, reference = model_pair()
    pending = torch.tensor([[0.45, 0.15], [0.8, 0.75]], dtype=torch.double)
    expected_sampler, sampler = sampler_pair(16, (1, 3), 812)
    acquisition = qLogExpectedImprovement(model, best_f=0.4, sampler=sampler, X_pending=pending)
    expected_acquisition = BtLogEI(
        reference, best_f=0.4, sampler=expected_sampler, X_pending=pending
    )
    query = torch.tensor([[[0.2, 0.6]]], dtype=torch.double, requires_grad=True)
    expected_query = query.detach().clone().requires_grad_()
    actual = acquisition(query)
    with propagate_grads(True), fast_pred_var(False):
        expected = expected_acquisition(expected_query)
    gradients = torch.autograd.grad(actual.sum(), (*model.parameters(), query))
    expected_gradients = torch.autograd.grad(
        expected.sum(), (*reference.parameters(), expected_query)
    )
    for result, want in zip(gradients, expected_gradients, strict=True):
        torch.testing.assert_close(result, want, rtol=3e-6, atol=3e-7)


def test_frozen_leaves_no_grad_and_parameter_backward_do_not_repeat_objective():
    from botorch_rs.acquisition.objective import GenericMCObjective

    model, _ = model_pair()
    leaves = tuple(model.parameters())
    leaves[0].requires_grad_(False)
    calls = []

    def objective(samples, X=None):
        calls.append(tuple(samples.shape))
        return samples[..., 0]

    acquisition = qLogExpectedImprovement(
        model, best_f=0.4, objective=GenericMCObjective(objective)
    )
    query = torch.tensor([[[0.2, 0.6], [0.7, 0.3]]], dtype=torch.double)
    with torch.no_grad():
        assert not acquisition(query).requires_grad
    value = acquisition(query)
    calls_before_backward = len(calls)
    value.sum().backward()
    assert len(calls) == calls_before_backward
    assert leaves[0].grad is None
    assert all(leaf.grad is not None for leaf in leaves[1:])


@pytest.mark.parametrize("name", ANALYTIC)
def test_minimizing_analytic_parameter_bars(name):
    model, reference = model_pair()
    kwargs = {"maximize": False}
    if "Improvement" in name:
        kwargs["best_f"] = torch.tensor(0.4, dtype=torch.double)
    if "ConfidenceBound" in name:
        kwargs["beta"] = 0.2
    acquisition, expected_acquisition = (
        getattr(analytic, name)(model, **kwargs),
        getattr(bt_analytic, name)(reference, **kwargs),
    )
    query = torch.tensor([[[0.2, 0.6]], [[0.7, 0.3]]], dtype=torch.double)
    actual = acquisition(query)
    with propagate_grads(True), fast_pred_var(False):
        expected = expected_acquisition(query)
    weights = torch.tensor([-0.2, 0.6], dtype=torch.double)
    gradients = torch.autograd.grad((actual * weights).sum(), tuple(model.parameters()))
    expected_gradients = torch.autograd.grad(
        (expected * weights).sum(),
        tuple(reference.parameters()),
        allow_unused=True,
        materialize_grads=True,
    )
    for result, want in zip(gradients, expected_gradients, strict=True):
        torch.testing.assert_close(result, want, rtol=3e-6, atol=3e-7)


def test_mixed_multioutput_model_list_raw_parameter_layout(monkeypatch):
    from test_model_list_parameter_autograd import model_pair as list_pair
    from test_model_list_parameter_autograd import reference_posterior

    model, reference = list_pair()
    # The corrected conditional is separately checked against official child
    # posteriors; mixed-list lazy task slicing is an upstream defect on these deps.
    monkeypatch.setattr(
        reference, "posterior", lambda X, **kwargs: reference_posterior(reference, X, **kwargs)
    )
    weights = torch.tensor([0.3, -0.6, 0.4], dtype=torch.double)
    expected_sampler, sampler = sampler_pair(16, (1, 2), 812)
    acquisition = qLogExpectedImprovement(
        model,
        best_f=0.4,
        sampler=sampler,
        posterior_transform=ScalarizedPosteriorTransform(weights),
    )
    expected_acquisition = BtLogEI(
        reference,
        best_f=0.4,
        sampler=expected_sampler,
        posterior_transform=BtScalarize(weights),
    )
    query = torch.tensor([[[0.2, 0.6], [0.7, 0.3]]], dtype=torch.double, requires_grad=True)
    expected_query = query.detach().clone().requires_grad_()
    actual, expected = acquisition(query), expected_acquisition(expected_query)
    torch.testing.assert_close(actual, expected, rtol=3e-6, atol=3e-7)
    gradients = torch.autograd.grad(actual.sum(), (*model.parameters(), query))
    expected_gradients = torch.autograd.grad(
        expected.sum(), (*reference.parameters(), expected_query)
    )
    for result, want in zip(gradients, expected_gradients, strict=True):
        torch.testing.assert_close(result, want, rtol=3e-6, atol=3e-7)


def test_native_wrapper_parameter_boundary_and_original_source_detection():
    from botorch_rs.acquisition.fixed_feature import FixedFeatureAcquisitionFunction

    model, _ = model_pair()
    base = analytic.LogExpectedImprovement(model, best_f=0.4)
    acquisition = FixedFeatureAcquisitionFunction(base, d=2, columns=[1], values=[0.6])
    query = torch.tensor([[[0.2]]], dtype=torch.double, requires_grad=True)
    value = acquisition(query)
    assert torch.isfinite(torch.autograd.grad(value.sum(), query, retain_graph=True)[0]).all()
    with pytest.raises(UnsupportedError, match="does not support automatic model-parameter"):
        value.sum().backward()
    with torch.no_grad():
        model.mean_module.raw_constant.add_(0.1)
    newly_wrapped_old_base = FixedFeatureAcquisitionFunction(base, d=2, columns=[1], values=[0.6])
    with pytest.raises(RuntimeError, match="Reconstruct the acquisition"):
        newly_wrapped_old_base(query)


def test_changed_shared_binding_metadata_requires_new_acquisition_snapshot():
    from botorch_rs.means import ConstantMean
    from botorch_rs.models.empirical_gps.shared_gp import _SharedBinding

    model, _ = model_pair()
    acquisition = analytic.PosteriorMean(model)
    query = torch.tensor([[[0.2, 0.6]]], dtype=torch.double)
    original = acquisition(query)
    parameters = tuple(model.parameters())
    expected = torch.autograd.grad(original.sum(), parameters, retain_graph=True)
    generation = model._rs.parameter_state_version
    shared_mean = ConstantMean()
    shared_mean._bind(_SharedBinding([model]))
    model.mean_module = shared_mean
    assert model._rs.parameter_state_version == generation
    with pytest.raises(RuntimeError, match="Reconstruct the acquisition"):
        acquisition(query)
    actual = torch.autograd.grad(original.sum(), parameters)
    for result, want in zip(actual, expected, strict=True):
        torch.testing.assert_close(result, want)
    rebuilt = analytic.PosteriorMean(model)
    query.requires_grad_()
    value = rebuilt(query)
    assert torch.isfinite(torch.autograd.grad(value.sum(), query, retain_graph=True)[0]).all()
    with pytest.raises(UnsupportedError, match="shared mean/kernel bindings"):
        value.sum().backward()


@pytest.mark.parametrize("container", [False, True])
def test_equal_valued_leaf_rebinding_requires_reconstruction_and_preserves_old_backward(container):
    from botorch_rs.models import ModelListGP

    first, _ = model_pair()
    second, _ = model_pair(fixed=True)
    model = ModelListGP(first, second) if container else second
    shared, discarded = first.mean_module.raw_constant, second.mean_module.raw_constant
    transform = (
        ScalarizedPosteriorTransform(torch.tensor([0.3, -0.6], dtype=torch.double))
        if container
        else None
    )
    acquisition = analytic.PosteriorMean(model, posterior_transform=transform)
    query = torch.tensor([[[0.2, 0.6]]], dtype=torch.double)
    original = acquisition(query)
    original_parameters = tuple(model.parameters())
    expected = torch.autograd.grad(original.sum(), original_parameters, retain_graph=True)
    expected_shared = sum(
        (
            gradient
            for parameter, gradient in zip(original_parameters, expected, strict=True)
            if parameter is shared or parameter is discarded
        ),
        torch.zeros_like(shared),
    )
    assert torch.count_nonzero(expected_shared)
    generation = second._rs.parameter_state_version
    state = second._sync_torch_parameters()
    state.entries = [
        (name, shared if name == "mean_module.raw_constant" else parameter)
        for name, parameter in state.entries
    ]
    assert second._rs.parameter_state_version == generation
    assert second.mean_module.raw_constant is shared
    with pytest.raises(RuntimeError, match="Reconstruct the acquisition"):
        acquisition(query)
    with pytest.raises(RuntimeError, match="Reconstruct the acquisition"):
        acquisition.value_and_grad(query)
    after = torch.autograd.grad(original.sum(), original_parameters)
    for result, want in zip(after, expected, strict=True):
        torch.testing.assert_close(result, want, rtol=0, atol=0)
    rebuilt = analytic.PosteriorMean(model, posterior_transform=transform)(query)
    torch.testing.assert_close(rebuilt, original)
    shared_gradient, old_gradient = torch.autograd.grad(
        rebuilt.sum(), (shared, discarded), allow_unused=True
    )
    torch.testing.assert_close(shared_gradient, expected_shared)
    assert old_gradient is None

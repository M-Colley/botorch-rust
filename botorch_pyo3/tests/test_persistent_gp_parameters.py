"""Persistent Torch leaves and native posterior moment parameter autograd."""

from __future__ import annotations

import copy

import pytest

from helpers import importorskip_botorch

importorskip_botorch()
torch = pytest.importorskip("torch")

from botorch.acquisition.objective import ScalarizedPosteriorTransform as BtScalarize  # noqa: E402
from botorch.settings import propagate_grads  # noqa: E402
from gpytorch.settings import fast_pred_var  # noqa: E402
from test_posterior_parameter_gradients import model_pair  # noqa: E402

from botorch_rs.acquisition.objective import ScalarizedPosteriorTransform  # noqa: E402
from botorch_rs.models import SingleTaskGP  # noqa: E402
from botorch_rs.optim.utils.model_utils import (  # noqa: E402
    get_parameters,
    get_parameters_and_bounds,
)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_parameters_are_persistent_leaves_and_bound_raw_attributes(dtype):
    x = torch.tensor([[0.1], [0.5], [0.9]], dtype=dtype)
    model = SingleTaskGP(x, torch.sin(x), outcome_transform=None)
    first, second = dict(model.named_parameters()), dict(model.named_parameters())
    assert first.keys() == second.keys()
    for name, value in first.items():
        assert isinstance(value, torch.nn.Parameter)
        assert value.is_leaf and value.requires_grad
        assert value is second[name]
        assert value.dtype == dtype
    assert model.likelihood.raw_noise is first["likelihood.noise_covar.raw_noise"]
    assert model.covar_module.raw_lengthscale is first["covar_module.raw_lengthscale"]
    assert model.mean_module.raw_constant is first["mean_module.raw_constant"]
    assert model.mean_module.constant is first["mean_module.raw_constant"]
    torch.testing.assert_close(
        torch.tensor(model._rs.raw_parameters(), dtype=dtype),
        torch.cat([p.detach().reshape(-1) for p in first.values()]),
        rtol=0,
        atol=0,
    )


def test_external_optimizer_push_and_native_setter_pull_preserve_identity():
    model, _ = model_pair()
    query = torch.tensor([[0.25, 0.65], [0.75, 0.35]], dtype=torch.double)
    entries = dict(model.named_parameters())
    old_mean = model.posterior(query).mean.detach().clone()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    loss = model.posterior(query).mean.square().sum()
    loss.backward()
    optimizer.step()
    new_mean = model.posterior(query).mean
    assert not torch.equal(old_mean, new_mean)
    assert all(value is dict(model.named_parameters())[name] for name, value in entries.items())
    model.covar_module.lengthscale = 0.37
    torch.testing.assert_close(
        model.covar_module.lengthscale, torch.full((1, 2), 0.37, dtype=torch.double)
    )
    assert model.covar_module.raw_lengthscale is entries["covar_module.raw_lengthscale"]
    raw = model._rs.raw_parameters()
    raw[1] += 0.2
    model._rs.set_raw_parameters(raw)
    assert model.mean_module.raw_constant is entries["mean_module.raw_constant"]
    assert model.mean_module.raw_constant.item() == pytest.approx(raw[1])


def test_conflicting_native_and_leaf_updates_fail_without_silent_overwrite():
    model, _ = model_pair()
    leaf = model.mean_module.raw_constant
    with torch.no_grad():
        leaf.add_(0.2)
    raw = model._rs.raw_parameters()
    raw[1] += 0.3
    model._rs.set_raw_parameters(raw)
    with pytest.raises(RuntimeError, match="both modified"):
        list(model.named_parameters())
    assert model._rs.raw_parameters() == raw
    assert leaf.item() == pytest.approx(0.2)


@pytest.mark.parametrize("mode", ["mean", "variance", "covariance"])
@pytest.mark.parametrize(
    "outputs,batch,tree,warp,fixed,selection",
    [
        (1, (), False, False, False, None),
        (2, (2,), False, False, False, [1]),
        (2, (2,), True, False, True, "scalar"),
        (1, (), True, True, False, None),
    ],
)
def test_automatic_moment_gradients_match_full_official_conditional(
    mode, outputs, batch, tree, warp, fixed, selection
):
    model, reference = model_pair(outputs, batch, tree, warp, fixed)
    query = torch.tensor([[[0.2, 0.6], [0.7, 0.3]]], dtype=torch.double).requires_grad_()
    expected_query = query.detach().clone().requires_grad_()
    kwargs, expected_kwargs = {}, {}
    if selection == "scalar":
        weights = torch.tensor([0.3, -0.6], dtype=torch.double)
        kwargs["posterior_transform"] = ScalarizedPosteriorTransform(weights, offset=0.2)
        expected_kwargs["posterior_transform"] = BtScalarize(weights, offset=0.2)
    elif selection is not None:
        kwargs["output_indices"] = selection
    with propagate_grads(True), fast_pred_var(False):
        if warp:
            reference.eval()
            reference.set_train_data(
                inputs=reference.input_transform(reference._original_train_inputs), strict=False
            )
        expected = reference.posterior(expected_query, observation_noise=True, **expected_kwargs)
    actual = model.posterior(query, observation_noise=True, **kwargs)
    value = actual.distribution.covariance_matrix if mode == "covariance" else getattr(actual, mode)
    expected_value = (
        expected.distribution.covariance_matrix if mode == "covariance" else getattr(expected, mode)
    )
    if isinstance(selection, list):
        # Official b0dd rejects single-output selection in from_independent_mvns.
        # Slice its full Gaussian to compare the same conditional distribution.
        if mode == "covariance":
            idcs = torch.tensor([t * 2 + p for t in selection for p in range(2)])
            expected_value = expected_value.index_select(-2, idcs).index_select(-1, idcs)
        else:
            expected_value = expected_value[..., selection]
    torch.testing.assert_close(value, expected_value, atol=3e-8, rtol=3e-7)
    weights = torch.linspace(-0.3, 0.7, value.numel(), dtype=torch.double).reshape(value.shape)
    gradients = torch.autograd.grad((value * weights).sum(), (*model.parameters(), query))
    expected_gradients = torch.autograd.grad(
        (expected_value * weights).sum(),
        (*reference.parameters(), expected_query),
        allow_unused=True,
        materialize_grads=True,
    )
    for gradient, expected_gradient in zip(gradients, expected_gradients, strict=True):
        torch.testing.assert_close(gradient, expected_gradient, rtol=5e-7, atol=5e-8)


def test_snapshot_backward_after_native_and_leaf_updates():
    model, _ = model_pair()
    query = torch.tensor([[0.2, 0.6], [0.7, 0.3]], dtype=torch.double)
    posterior = model.posterior(query)
    values = posterior.mean
    expected = torch.autograd.grad(values.sum(), tuple(model.parameters()), retain_graph=True)
    model.mean_module.constant = 0.7
    list(model.parameters())
    with torch.no_grad():
        model.covar_module.raw_lengthscale.add_(0.05)
    model.posterior(query)
    actual = torch.autograd.grad(values.sum(), tuple(model.parameters()))
    torch.testing.assert_close(posterior.mean, values)
    for result, want in zip(actual, expected, strict=True):
        torch.testing.assert_close(result, want)


def test_freezing_copying_and_no_grad_preserve_protocol():
    model, _ = model_pair()
    leaf = model.mean_module.raw_constant
    leaf.requires_grad_(False)
    assert "mean_module.raw_constant" in get_parameters(model, requires_grad=False)
    assert "mean_module.raw_constant" not in get_parameters(model, requires_grad=True)
    assert (
        get_parameters_and_bounds(model, requires_grad=False)[0]["mean_module.raw_constant"] is leaf
    )
    copied = copy.deepcopy(model)
    assert copied.mean_module.raw_constant is not leaf
    assert not copied.mean_module.raw_constant.requires_grad
    with torch.no_grad():
        post = model.posterior(torch.tensor([[0.2, 0.6]], dtype=torch.double))
        assert not post.mean.requires_grad
        assert not post.variance.requires_grad
    assert isinstance(model.posterior([[0.2, 0.6]]).mean, list)


def test_native_fitting_updates_existing_leaves_and_raw_data_edits_push():
    from botorch_rs.fit import fit_gpytorch_mll
    from botorch_rs.mlls import ExactMarginalLogLikelihood

    model, _ = model_pair()
    leaves = dict(model.named_parameters())
    before = model._rs.raw_parameters()
    fit_gpytorch_mll(
        ExactMarginalLogLikelihood(model.likelihood, model),
        optimizer_kwargs={"options": {"maxiter": 3}},
        max_attempts=1,
    )
    assert before != model._rs.raw_parameters()
    after = dict(model.named_parameters())
    assert all(after[name] is leaf for name, leaf in leaves.items())
    assert model._rs.raw_parameters() == [
        v for p in after.values() for v in p.detach().reshape(-1).tolist()
    ]
    # Optimizer-style .data writes bypass Tensor._version; synchronization
    # compares actual leaf values so these updates are not silently lost.
    leaves["mean_module.raw_constant"].data.add_(0.25)
    model.posterior(torch.tensor([[0.2, 0.6]], dtype=torch.double))
    assert model._rs.raw_parameters()[1] == leaves["mean_module.raw_constant"].item()


@pytest.mark.parametrize("mode", ["mean", "variance"])
def test_log_outcome_moment_parameter_gradients(mode):
    from botorch.models import SingleTaskGP as BtGP
    from botorch.models.transforms.outcome import Log as BtLog

    from botorch_rs.models.transforms.outcome import Log

    x = torch.tensor([[0.1], [0.4], [0.7], [0.9]], dtype=torch.double)
    y = torch.exp(x * 0.3)
    model, reference = (
        SingleTaskGP(x, y, outcome_transform=Log()),
        BtGP(x, y, outcome_transform=BtLog()),
    )
    query = torch.tensor([[0.2], [0.8]], dtype=torch.double)
    with propagate_grads(True), fast_pred_var(False):
        expected = getattr(reference.posterior(query), mode)
    actual = getattr(model.posterior(query), mode)
    torch.testing.assert_close(actual, expected)
    gradients = torch.autograd.grad(actual.sum(), tuple(model.parameters()))
    expected_gradients = torch.autograd.grad(expected.sum(), tuple(reference.parameters()))
    for result, want in zip(gradients, expected_gradients, strict=True):
        torch.testing.assert_close(result, want, rtol=5e-7, atol=5e-8)


def test_constrained_kernel_attribute_uses_native_transform_derivative():
    model, reference = model_pair(tree=True)
    raw = model.covar_module.raw_outputscale
    ref_raw = reference.covar_module.raw_outputscale
    actual = torch.autograd.grad(model.covar_module.outputscale.sum(), raw)[0]
    expected = torch.autograd.grad(reference.covar_module.outputscale.sum(), ref_raw)[0]
    torch.testing.assert_close(actual, expected)


def test_native_low_level_fit_respects_frozen_leaves():
    from botorch_rs.mlls import ExactMarginalLogLikelihood
    from botorch_rs.optim.fit import fit_gpytorch_mll_scipy

    model, _ = model_pair()
    parameter = model.covar_module.raw_lengthscale
    parameter.requires_grad_(False)
    initial = parameter.detach().clone()
    fit_gpytorch_mll_scipy(
        ExactMarginalLogLikelihood(model.likelihood, model), options={"maxiter": 3}
    )
    assert model.covar_module.raw_lengthscale is parameter
    torch.testing.assert_close(parameter, initial, rtol=0, atol=0)


@pytest.mark.parametrize("operation", ["condition", "subset", "member"])
def test_derived_models_push_pending_torch_parameter_edits(operation):
    model, _ = model_pair(outputs=2, batch=(2,))
    model.eval()
    with torch.no_grad():
        model.mean_module.raw_constant.fill_(0.45)
    if operation == "condition":
        result = model.condition_on_observations(
            torch.tensor([[0.2, 0.6]], dtype=torch.double),
            torch.tensor([[[0.2, 0.3]], [[0.4, 0.5]]], dtype=torch.double),
        )
    elif operation == "subset":
        result = model.subset_output([1])
    else:
        result = model._member(1)
    torch.testing.assert_close(
        result.mean_module.raw_constant,
        torch.full_like(result.mean_module.raw_constant, 0.45),
    )


def test_sample_query_and_parameter_gradients_are_both_available():
    model, _ = model_pair()
    query = torch.tensor([[0.2, 0.6], [0.7, 0.3]], dtype=torch.double, requires_grad=True)
    posterior = model.posterior(query)
    samples = posterior.rsample_from_base_samples(
        torch.Size([2]), torch.tensor([[-0.3, 0.5], [0.4, -0.8]], dtype=torch.double)
    )
    assert torch.isfinite(torch.autograd.grad(samples.sum(), query, retain_graph=True)[0]).all()
    for gradient in torch.autograd.grad(samples.sum(), tuple(model.parameters())):
        assert torch.isfinite(gradient).all()

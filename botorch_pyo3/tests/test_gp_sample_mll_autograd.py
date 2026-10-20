"""Native reparameterized sample and weighted exact-MLL parameter adjoints."""

from __future__ import annotations

import pytest

from helpers import importorskip_botorch

importorskip_botorch()
torch = pytest.importorskip("torch")

from botorch.acquisition.objective import ScalarizedPosteriorTransform as BtScalarize  # noqa: E402
from botorch.settings import propagate_grads  # noqa: E402
from gpytorch.mlls import ExactMarginalLogLikelihood as BtMLL  # noqa: E402
from gpytorch.settings import fast_pred_var  # noqa: E402
from test_posterior_parameter_gradients import model_pair  # noqa: E402

from botorch_rs.acquisition.objective import ScalarizedPosteriorTransform  # noqa: E402
from botorch_rs.mlls import ExactMarginalLogLikelihood  # noqa: E402


@pytest.mark.parametrize("selection", [None, "scalar"])
@pytest.mark.parametrize(
    "outputs,batch,tree,warp,fixed",
    [
        (1, (), False, False, False),
        (2, (2,), True, False, True),
        (1, (), True, True, False),
        (1, (2,), False, True, False),
    ],
)
def test_posterior_sample_parameter_and_query_gradients(
    selection, outputs, batch, tree, warp, fixed
):
    if outputs == 1 and selection is not None:
        selection = None
    model, reference = model_pair(outputs, batch, tree, warp, fixed)
    query = torch.tensor([[[0.23, 0.61], [0.71, 0.32]]], dtype=torch.double).requires_grad_()
    expected_query = query.detach().clone().requires_grad_()
    kwargs, expected_kwargs = {}, {}
    if selection == "scalar":
        weights = torch.tensor([0.3, -0.6], dtype=torch.double)
        kwargs["posterior_transform"] = ScalarizedPosteriorTransform(weights, offset=0.2)
        expected_kwargs["posterior_transform"] = BtScalarize(weights, offset=0.2)
    with propagate_grads(True), fast_pred_var(False):
        if warp:
            reference.eval()
            reference.set_train_data(
                inputs=reference.input_transform(reference._original_train_inputs), strict=False
            )
        expected = reference.posterior(expected_query, observation_noise=True, **expected_kwargs)
    actual = model.posterior(query, observation_noise=True, **kwargs)
    shape = (2, 3, *expected.base_sample_shape)
    normals = torch.linspace(-0.8, 1.1, torch.Size(shape).numel(), dtype=torch.double).reshape(
        shape
    )
    actual_samples = actual.rsample_from_base_samples(torch.Size([2, 3]), normals)
    expected_samples = expected.rsample_from_base_samples(torch.Size([2, 3]), normals)
    torch.testing.assert_close(actual_samples, expected_samples, atol=2e-8, rtol=2e-7)
    weights = torch.linspace(-0.4, 0.7, actual_samples.numel(), dtype=torch.double).reshape(
        actual_samples.shape
    )
    actual_grads = torch.autograd.grad(
        (actual_samples * weights).sum(), (*model.parameters(), query)
    )
    expected_grads = torch.autograd.grad(
        (expected_samples * weights).sum(), (*reference.parameters(), expected_query)
    )
    for actual_grad, expected_grad in zip(actual_grads, expected_grads, strict=True):
        torch.testing.assert_close(actual_grad, expected_grad, atol=6e-8, rtol=6e-7)


@pytest.mark.parametrize(
    "outputs,batch,tree,warp,fixed",
    [
        (1, (), False, False, False),
        (2, (2,), True, False, True),
        (1, (), True, True, False),
        (1, (2,), False, True, False),
    ],
)
def test_weighted_exact_mll_parameter_backward(outputs, batch, tree, warp, fixed):
    model, reference = model_pair(outputs, batch, tree, warp, fixed)
    native_mll, expected_mll = (
        ExactMarginalLogLikelihood(model.likelihood, model),
        BtMLL(reference.likelihood, reference),
    )
    reference.train()
    actual = native_mll()
    expected = expected_mll(reference(*reference.train_inputs), reference.train_targets)
    torch.testing.assert_close(actual, expected, rtol=2e-7, atol=2e-8)
    weights = torch.linspace(0.2, 0.9, actual.numel(), dtype=torch.double).reshape(actual.shape)
    actual_grads = torch.autograd.grad((actual * weights).sum(), tuple(model.parameters()))
    expected_grads = torch.autograd.grad((expected * weights).sum(), tuple(reference.parameters()))
    for actual_grad, expected_grad in zip(actual_grads, expected_grads, strict=True):
        torch.testing.assert_close(actual_grad, expected_grad, atol=6e-8, rtol=6e-7)


def test_torch_adam_uses_native_mll_and_snapshot_backward():
    model, _ = model_pair()
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    leaves = tuple(model.parameters())
    original = mll()
    expected = torch.autograd.grad(original, leaves, retain_graph=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    losses = []
    for _ in range(4):
        optimizer.zero_grad()
        loss = -mll()
        losses.append(loss.item())
        loss.backward()
        optimizer.step()
    assert losses[-1] < losses[0]
    actual = torch.autograd.grad(original, leaves)
    for result, want in zip(actual, expected, strict=True):
        torch.testing.assert_close(result, want)
    with torch.no_grad():
        assert not mll().requires_grad


def test_sample_normals_are_snapshotted_for_parameter_backward():
    model, _ = model_pair()
    query = torch.tensor([[0.2, 0.6], [0.7, 0.3]], dtype=torch.double)
    posterior = model.posterior(query)
    normals = torch.tensor([[-0.3, 0.5], [0.4, -0.8]], dtype=torch.double)
    samples = posterior.rsample_from_base_samples(torch.Size([2]), normals)
    expected = torch.autograd.grad(samples.sum(), tuple(model.parameters()), retain_graph=True)
    normals.add_(10.0)
    model.mean_module.constant = 0.8
    tuple(model.parameters())
    actual = torch.autograd.grad(samples.sum(), tuple(model.parameters()))
    for result, want in zip(actual, expected, strict=True):
        torch.testing.assert_close(result, want)


@pytest.mark.parametrize("batch,outputs", [((), 2), ((2,), 1)])
def test_shared_warp_parameter_configuration_explicitly_rejected(batch, outputs):
    from botorch_rs.exceptions.errors import UnsupportedError
    from botorch_rs.models import SingleTaskGP
    from botorch_rs.models.transforms.input import Warp

    x = torch.tensor([[0.1], [0.4], [0.7], [0.9]], dtype=torch.double).expand(*batch, 4, 1)
    y = torch.sin(x).expand(*batch, 4, outputs)
    with pytest.raises(UnsupportedError, match="shared by several outputs or batch elements"):
        SingleTaskGP(x, y, input_transform=Warp(1, [0]))


def test_incompatible_copied_transform_descriptor_rejects_parameter_backward():
    from botorch_rs.exceptions.errors import UnsupportedError

    model, _ = model_pair(batch=(2,), warp=True)
    # Simulate a changed/copied Python transform descriptor without altering
    # the valid, independent native model state.
    model.input_transform.batch_shape = ()
    model.__dict__.pop("_torch_parameter_state")
    query = torch.tensor([[0.2, 0.6]], dtype=torch.double, requires_grad=True)
    posterior = model.posterior(query)
    assert torch.isfinite(torch.autograd.grad(posterior.mean.sum(), query)[0]).all()
    with pytest.raises(UnsupportedError, match="Warp shares them"):
        posterior.mean.sum().backward()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.posterior(query).mean.sum().backward()
    assert torch.isfinite(query.grad).all()


def test_selected_output_sample_parameter_gradients():
    model, reference = model_pair(outputs=2, batch=(2,), fixed=True)
    query = torch.tensor([[0.2, 0.6], [0.7, 0.3]], dtype=torch.double)
    with propagate_grads(True), fast_pred_var(False):
        expected = reference.posterior(query)
    actual = model.posterior(query, output_indices=[1])
    normals = torch.linspace(-0.5, 0.8, 12, dtype=torch.double).reshape(3, 2, 2)
    full_normals = torch.zeros(3, *expected.base_sample_shape, dtype=torch.double)
    full_normals[..., 1] = normals
    samples = actual.rsample_from_base_samples(torch.Size([3]), normals)
    expected_samples = expected.rsample_from_base_samples(torch.Size([3]), full_normals)[..., 1:2]
    torch.testing.assert_close(samples, expected_samples)
    gradients = torch.autograd.grad(samples.square().sum(), tuple(model.parameters()))
    expected_gradients = torch.autograd.grad(
        expected_samples.square().sum(), tuple(reference.parameters())
    )
    for result, want in zip(gradients, expected_gradients, strict=True):
        torch.testing.assert_close(result, want, rtol=6e-7, atol=6e-8)


def test_log_outcome_sample_parameter_gradients():
    from botorch.models import SingleTaskGP as BtGP
    from botorch.models.transforms.outcome import Log as BtLog

    from botorch_rs.models import SingleTaskGP
    from botorch_rs.models.transforms.outcome import Log

    x = torch.tensor([[0.1], [0.4], [0.7], [0.9]], dtype=torch.double)
    y = torch.exp(x * 0.3)
    model = SingleTaskGP(x, y, outcome_transform=Log())
    reference = BtGP(x, y, outcome_transform=BtLog())
    query = torch.tensor([[0.2], [0.8]], dtype=torch.double)
    normals = torch.tensor([[-0.3, 0.5], [0.4, -0.8]], dtype=torch.double)
    with propagate_grads(True), fast_pred_var(False):
        expected = reference.posterior(query).rsample_from_base_samples(torch.Size([2]), normals)
    actual = model.posterior(query).rsample_from_base_samples(torch.Size([2]), normals)
    torch.testing.assert_close(actual, expected)
    gradients = torch.autograd.grad(actual.sum(), tuple(model.parameters()))
    expected_gradients = torch.autograd.grad(expected.sum(), tuple(reference.parameters()))
    for result, want in zip(gradients, expected_gradients, strict=True):
        torch.testing.assert_close(result, want, rtol=6e-7, atol=6e-8)

# ruff: noqa: E402
"""Native handle replacement must not resurrect stale Tensor statistics."""

import pytest

torch = pytest.importorskip("torch")

from botorch_rs import _botorch_rs as _rs
from botorch_rs.models._fully_bayesian_outcomes import batched_native_outcome
from botorch_rs.models.transforms.outcome import ChainedOutcomeTransform, Standardize

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


@pytest.mark.parametrize("device", DEVICES)
def test_explicit_native_replacement_and_tensor_export_have_distinct_authority(device):
    Y = torch.tensor([[0.1, 0.4], [0.7, 0.8], [1.1, 1.4]], dtype=torch.double, device=device)
    Y.requires_grad_()
    transform = Standardize(2)
    values, _ = transform(Y)
    tensor_state = transform._tensor_state()
    _ = transform._rs
    _ = transform._rs
    assert transform._tensor_state() is tensor_state
    gradient = torch.autograd.grad(values.square().sum(), Y, create_graph=True)[0]
    assert torch.isfinite(torch.autograd.grad(gradient.sum(), Y)[0]).all()
    transform._batch_shape = (2,)
    transform._rs = _rs.Standardize.fitted(
        [[3.0, 4.0], [5.0, 6.0]], [[2.0, 3.0], [4.0, 5.0]], 1e-8, [2]
    )
    assert "_tensor_standardize" not in transform.__dict__
    transform.eval()
    query = Y.new_tensor([[[0.2, 0.8]], [[1.2, 1.5]]], requires_grad=True)
    actual, _ = transform.untransform(query)
    means = query.new_tensor([[[3.0, 4.0]], [[5.0, 6.0]]])
    stdvs = query.new_tensor([[[2.0, 3.0]], [[4.0, 5.0]]])
    torch.testing.assert_close(actual, query * stdvs + means)
    torch.testing.assert_close(torch.autograd.grad(actual.sum(), query)[0], stdvs)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("count", [3, 4])
@pytest.mark.parametrize("chained", [False, True])
def test_fitted_tensor_outcome_can_be_retrained_for_native_mcmc_batch(device, count, chained):
    transform = (
        ChainedOutcomeTransform(first=Standardize(1), second=Standardize(1))
        if chained
        else Standardize(1)
    )
    Y = torch.tensor([[0.2], [0.7], [1.5]], dtype=torch.double, device=device)
    expected, _ = transform(Y)
    children = list(transform.values()) if chained else [transform]
    states = [child._tensor_state() for child in children]
    batched = batched_native_outcome(transform, count)
    batched_children = list(batched.values()) if chained else [batched]
    for child in batched_children:
        assert child.batch_shape == (count,)
        assert "_tensor_standardize" not in child.__dict__
        assert not child._rs.is_trained
    native_values, _ = batched(Y.cpu().unsqueeze(0).expand(count, -1, -1).tolist())
    torch.testing.assert_close(
        torch.tensor(native_values, dtype=Y.dtype, device=device), expected.expand(count, -1, -1)
    )
    assert all(
        child._tensor_state() is state for child, state in zip(children, states, strict=True)
    )

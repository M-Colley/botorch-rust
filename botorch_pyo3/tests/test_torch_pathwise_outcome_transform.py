# ruff: noqa: E402
"""Path outcome transforms retain calibration data and sample graphs on device."""

from unittest.mock import patch

import pytest

torch = pytest.importorskip("torch")

from botorch_rs.models.transforms.outcome import Standardize
from botorch_rs.sampling.pathwise.utils import OutcomeUntransformer

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("outputs", [1, 2])
@pytest.mark.parametrize("batch", [(), (2,)])
def test_path_standardize_retains_all_calibration_sample_gradients_and_higher_graphs(
    device, outputs, batch
):
    y = torch.linspace(0.2, 1.7, 5 * outputs, device=device, dtype=torch.double).reshape(5, outputs)
    y = y.expand(*batch, 5, outputs).clone().requires_grad_()
    values = torch.linspace(-0.6, 0.9, 3 * 4 * outputs, dtype=y.dtype, device=device).reshape(
        3, outputs, 4
    )
    if batch:
        values = values.unsqueeze(1).expand(3, *batch, outputs, 4).clone()
    if outputs == 1:
        values = values.squeeze(-2)
    values.requires_grad_()
    transform = Standardize(outputs, batch_shape=batch)
    transform(y)
    transform.eval()
    with (
        patch("botorch_rs._arrays.to_flat", side_effect=AssertionError("host copy")),
        patch(
            "botorch_rs.models.transforms.outcome.to_flat", side_effect=AssertionError("host copy")
        ),
    ):
        actual = OutcomeUntransformer(transform, outputs)(values)
    mean, scale = y.mean(-2, keepdim=True), y.std(-2, keepdim=True)
    if outputs == 1:
        expected = (mean + scale * values.unsqueeze(-1)).squeeze(-1)
    else:
        expected = (mean + scale * values.transpose(-1, -2)).transpose(-1, -2)
    torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)
    score, reference = actual.sin().sum(), expected.sin().sum()
    first = torch.autograd.grad(score, (y, values), create_graph=True)
    other = torch.autograd.grad(reference, (y, values), create_graph=True)
    second = torch.autograd.grad(sum(g.square().sum() for g in first), (y, values))
    other_second = torch.autograd.grad(sum(g.square().sum() for g in other), (y, values))
    for a, b in zip((*first, *second), (*other, *other_second), strict=True):
        assert bool(torch.isfinite(a).all())
        torch.testing.assert_close(a, b, atol=3e-9, rtol=3e-8)
    assert actual.device == values.device


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("outputs", [1, 2])
def test_path_standardize_direct_values_and_gradients_match_unmodified_source(device, outputs):
    pytest.importorskip("botorch")
    from botorch.models.transforms.outcome import Standardize as ReferenceStandardize
    from botorch.sampling.pathwise.utils import OutcomeUntransformer as ReferenceUntransformer

    y = torch.tensor([[0.2, 0.5], [0.7, 1.3], [1.4, 2.0]], dtype=torch.double, device=device)
    y = y[:, :outputs].clone().requires_grad_()
    values = torch.linspace(-0.5, 1.1, 8 * outputs, dtype=y.dtype, device=device).reshape(
        2, outputs, 4
    )
    if outputs == 1:
        values = values.squeeze(-2)
    values.requires_grad_()
    actual_transform, source_transform = Standardize(outputs), ReferenceStandardize(outputs).to(y)
    actual_transform(y)
    source_transform(y)
    actual = OutcomeUntransformer(actual_transform, outputs)(values)
    source = ReferenceUntransformer(source_transform, outputs)(values)
    torch.testing.assert_close(actual, source, atol=1e-12, rtol=1e-12)
    a = torch.autograd.grad(actual.sin().sum(), (y, values))
    b = torch.autograd.grad(source.sin().sum(), (y, values))
    for left, right in zip(a, b, strict=True):
        torch.testing.assert_close(left, right, atol=1e-12, rtol=1e-12)

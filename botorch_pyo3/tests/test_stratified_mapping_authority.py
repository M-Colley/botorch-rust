"""Actual registered stratum remapping controls fitting and inverse maps."""

# ruff: noqa: E402
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from botorch_rs.models.gp_regression import SingleTaskGP
from botorch_rs.models.transforms.outcome import StratifiedStandardize

DEVICES = [
    "cpu",
    pytest.param(
        "cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
    ),
]


def _data(device, sparse):
    tasks = (1, 3) if sparse else (0, 1)
    x = torch.tensor(
        [[0.1, tasks[0]], [0.3, tasks[0]], [0.6, tasks[1]], [0.9, tasks[1]]],
        dtype=torch.double,
        device=device,
    )
    y = x.new_tensor([[1.0], [3.0], [11.0], [13.0]])
    return x, y, tasks


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("sparse", [False, True])
@pytest.mark.parametrize("mode", ["inverse", "refit", "colliding_refit", "array_refit", "unknown"])
def test_registered_mapping_matches_unmodified_source(device, sparse, mode):
    source = pytest.importorskip("botorch.models.transforms.outcome")
    x, y, tasks = _data(device, sparse)
    actual = StratifiedStandardize(1, torch.tensor(tasks, device=device)).to(y)
    expected = source.StratifiedStandardize(1, torch.tensor(tasks, device=device)).to(y)
    for transform in (actual, expected):
        transform(y, X=x)
        transform.strata_mapping[tasks[0]] = 1
        transform.strata_mapping[tasks[1]] = 1 if mode == "colliding_refit" else 0
    if mode == "unknown":
        if not sparse:
            # A floating mapping is required to represent the source NaN sentinel.
            for transform in (actual, expected):
                transform.strata_mapping = transform.strata_mapping.double()
        for transform in (actual, expected):
            transform.strata_mapping[tasks[0]] = float("nan")
    if "refit" in mode:
        values = y * 1.3 + 0.7
        if mode == "array_refit":
            actual(values.cpu().tolist(), X=x.cpu().tolist())
        else:
            actual(values, X=x)
        expected(values, X=x)
    for transform in (actual, expected):
        transform.eval()
    z, variance = torch.zeros_like(y), torch.full_like(y, 0.4)
    if mode == "unknown":
        with pytest.warns(UserWarning, match="not observed"):
            actual_values = actual.untransform(z, variance, X=x)
        with pytest.warns(UserWarning, match="not observed"):
            expected_values = expected.untransform(z, variance, X=x)
    else:
        actual_values, expected_values = (
            actual.untransform(z, variance, X=x),
            expected.untransform(z, variance, X=x),
        )
    for left, right in zip(actual_values, expected_values, strict=True):
        torch.testing.assert_close(left, right)
    # Registered mapping is nonpersistent and survives fitting, without a shadow copy.
    assert dict(actual.named_buffers())["strata_mapping"] is actual.strata_mapping
    assert "strata_mapping" not in actual.state_dict()


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("invalid", [-1, 10])
def test_invalid_mapping_indices_preserve_source_errors(device, invalid):
    source = pytest.importorskip("botorch.models.transforms.outcome")
    x, y, tasks = _data(device, False)
    for cls in (StratifiedStandardize, source.StratifiedStandardize):
        transform = cls(1, torch.tensor(tasks, device=device)).to(y)
        transform(y, X=x)
        transform.eval()
        transform.strata_mapping[0] = invalid
        if device == "cuda":
            # Validate on CPU to avoid poisoning the process with a device assertion.
            transform.cpu()
            transform.strata_mapping = transform.strata_mapping.cpu()
        with pytest.raises(RuntimeError):
            transform.untransform(torch.zeros_like(y.cpu()), X=x.cpu())


def test_native_posterior_uses_live_stratification_mapping():
    x, y, tasks = _data("cpu", True)
    outcome = StratifiedStandardize(1, torch.tensor(tasks))
    model = SingleTaskGP(x, y, outcome_transform=outcome, backend="native")
    query = x[[0, 2]]
    before = model.posterior(query)
    standardized_mean = (before.mean - y.new_tensor([[2.0], [12.0]])) / (2.0**0.5)
    outcome.strata_mapping[tasks[0]], outcome.strata_mapping[tasks[1]] = 1, 0
    expected = standardized_mean * (2.0**0.5) + y.new_tensor([[12.0], [2.0]])
    torch.testing.assert_close(model.posterior(query).mean, expected, atol=3e-10, rtol=3e-10)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("sparse", [False, True])
def test_remapped_statistics_match_source_observed_stratum_derivatives(device, sparse):
    source = pytest.importorskip("botorch.models.transforms.outcome")
    x, initial, tasks = _data(device, sparse)
    results = []
    for direct in (True, False):
        y = initial.clone().requires_grad_()
        if direct:
            transform = StratifiedStandardize(1, torch.tensor(tasks, device=device)).to(y)
            transform(y, X=x)
            transform.strata_mapping[tasks[0]], transform.strata_mapping[tasks[1]] = 1, 0
            transform.eval()
            value, variance = transform.untransform(
                torch.full_like(y, 0.3), torch.full_like(y, 0.4), X=x
            )
        else:
            # Source StratifiedStandardize inserts NaNs before nanstd and has
            # undefined backward (BoTorch #3520). Its unmodified Standardize
            # on the independently observed strata supplies a finite oracle.
            transform = source.Standardize(1, batch_shape=torch.Size([2])).to(y)
            grouped = y.reshape(2, 2, 1)
            transform(grouped)
            transform.eval()
            value, variance = transform.untransform(
                torch.full_like(grouped, 0.3), torch.full_like(grouped, 0.4)
            )
            value, variance = value.flip(0).reshape_as(y), variance.flip(0).reshape_as(y)
        loss = value.square().sum() + variance.sum() * 0.2
        gradient = torch.autograd.grad(loss, y, create_graph=True)[0]
        direction = torch.linspace(0.2, 0.8, y.numel(), device=device, dtype=y.dtype).reshape_as(y)
        hessian = torch.autograd.grad((gradient * direction).sum(), y)[0]
        results.append((value, variance, gradient, hessian))
    for actual, expected in zip(*results, strict=True):
        assert torch.isfinite(expected).all()
        torch.testing.assert_close(actual, expected, atol=2e-11, rtol=2e-11)

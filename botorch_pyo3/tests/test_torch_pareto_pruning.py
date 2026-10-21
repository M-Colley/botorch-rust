"""Pareto selection and multi-output pruning keep their device calculations."""

# ruff: noqa: E402
from __future__ import annotations

from unittest.mock import patch

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("botorch")

from botorch.acquisition.multi_objective.utils import (
    prune_inferior_points_multi_objective as reference_prune,
)
from botorch.utils.multi_objective.pareto import is_non_dominated as reference_pareto
from test_torch_backend_acquisition import FixedSampler

from botorch_rs.acquisition.multi_objective.utils import prune_inferior_points_multi_objective
from botorch_rs.models import SingleTaskGP
from botorch_rs.utils.multi_objective import pareto

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("maximize", [False, True])
@pytest.mark.parametrize("deduplicate", [False, True])
@pytest.mark.parametrize("loop", [False, True])
def test_pareto_duplicate_batch_masks_match_source_without_host_transfers(
    monkeypatch, device, maximize, deduplicate, loop
):
    y = torch.tensor(
        [
            [[0.1, 0.8], [0.7, 0.2], [0.1, 0.8], [0.2, 0.3], [0.4, 0.6]],
            [[-0.2, 0.6], [0.8, 0.2], [-0.2, 0.6], [0.1, 0.3], [0.5, 0.5]],
        ],
        device=device,
        dtype=torch.double,
    )
    expected = reference_pareto(y, maximize=maximize, deduplicate=deduplicate)
    if loop:
        monkeypatch.setattr(pareto, "MAX_BYTES", 0)
    with (
        patch.object(torch.Tensor, "cpu", side_effect=AssertionError("host transfer")),
        patch.object(torch.Tensor, "tolist", side_effect=AssertionError("host transfer")),
    ):
        actual = pareto.is_non_dominated(y, maximize=maximize, deduplicate=deduplicate)
    torch.testing.assert_close(actual, expected)
    assert actual.device == y.device


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("loop", [False, True])
def test_pareto_nan_rows_are_excluded_and_empty_fronts_keep_shape(monkeypatch, device, loop):
    if loop:
        monkeypatch.setattr(pareto, "MAX_BYTES", 0)
    y = torch.tensor([[0.2, 0.9], [float("nan"), 0.7], [0.8, 0.3], [0.1, 0.2]], device=device)
    torch.testing.assert_close(
        pareto.is_non_dominated(y), torch.tensor([True, False, True, False], device=device)
    )
    assert pareto.is_non_dominated(y.new_empty(2, 0, 2)).shape == (2, 0)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("batch", [False, True])
@pytest.mark.parametrize("max_frac", [0.3, 1.0])
@pytest.mark.parametrize("constrained", [False, True])
def test_multioutput_pruning_matches_source_with_marginalization_and_probability_caps(
    device, batch, max_frac, constrained
):
    x = torch.tensor([[0.03], [0.29], [0.58], [0.87]], device=device, dtype=torch.double)
    y = torch.cat(((5 * x).sin(), (3 * x).cos()), dim=-1)
    if batch:
        model = SingleTaskGP(x.expand(2, 4, 1), torch.stack((y, y + 0.1)), backend="torch")
    else:
        model = SingleTaskGP(x, y, backend="torch")
    x = x.clone().requires_grad_()
    ref = x.new_tensor([-0.5, -0.8])
    sampler = FixedSampler(8, device, (41,))
    constraints = [lambda samples: samples[..., 0] - 0.4] if constrained else None
    kwargs = dict(max_frac=max_frac, constraints=constraints, marginalize_dim=-3 if batch else None)
    with patch("botorch.acquisition.utils.get_sampler", return_value=sampler):
        expected = reference_prune(model, x, ref, **kwargs)
    with (
        patch.object(torch.Tensor, "cpu", side_effect=AssertionError("host transfer")),
        patch.object(torch.Tensor, "tolist", side_effect=AssertionError("host transfer")),
    ):
        actual = prune_inferior_points_multi_objective(model, x, ref, sampler=sampler, **kwargs)
        gradient = torch.autograd.grad(actual.sum(), x)[0]
    torch.testing.assert_close(actual, expected)
    assert gradient.sum() == actual.numel()

"""Public cell-integral values, broadcast shapes, and native first derivatives."""

from __future__ import annotations

import pytest

from botorch_rs.acquisition.multi_objective.analytic import ExpectedHypervolumeImprovement
from helpers import KINDS, as_kind, assert_close, check_kind, importorskip_botorch, shape


def inputs():
    return (
        [[0.1, -0.2], [0.3, 0.4]],
        [[0.7, 1.2], [1.1, 1.8]],
        [[[0.4, 0.7]], [[0.8, 0.3]]],
        [[[0.3, 0.8]], [[0.4, 0.6]]],
    )


def official_integral(name, values):
    # These mathematical helpers use only the standard normal distribution;
    # isolate them from GP construction and hypercell partition generation.
    from types import SimpleNamespace

    import torch
    from botorch.acquisition.multi_objective.analytic import (
        ExpectedHypervolumeImprovement as Official,
    )

    context = SimpleNamespace(normal=torch.distributions.Normal(0.0, 1.0))
    return getattr(Official, name)(context, *values)


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("name", ["psi", "nu"])
def test_official_values_with_batch_cell_and_objective_broadcast(kind, name):
    importorskip_botorch()
    import torch

    values = inputs()
    actual = getattr(ExpectedHypervolumeImprovement, name)(
        None, *(as_kind(value, kind) for value in values)
    )
    expected = official_integral(
        name, [torch.tensor(value, dtype=torch.double) for value in values]
    )
    assert shape(actual) == (2, 2, 2)
    check_kind(actual, kind)
    assert_close(actual, expected, atol=2e-14)


@pytest.mark.parametrize("name", ["psi", "nu"])
def test_all_broadcast_input_adjoints_match_official(name):
    importorskip_botorch()
    import torch

    actual_inputs = [
        torch.tensor(value, dtype=torch.double, requires_grad=True) for value in inputs()
    ]
    expected_inputs = [value.detach().clone().requires_grad_() for value in actual_inputs]
    weights = torch.tensor(
        [[[0.1, 0.7], [-0.3, 0.2]], [[0.8, -0.2], [0.4, 0.3]]], dtype=torch.double
    )
    actual = getattr(ExpectedHypervolumeImprovement, name)(None, *actual_inputs)
    expected = official_integral(name, expected_inputs)
    actual_grad = torch.autograd.grad((actual * weights).sum(), actual_inputs)
    expected_grad = torch.autograd.grad((expected * weights).sum(), expected_inputs)
    for a, e in zip(actual_grad, expected_grad, strict=True):
        torch.testing.assert_close(a, e, rtol=2e-13, atol=2e-14)

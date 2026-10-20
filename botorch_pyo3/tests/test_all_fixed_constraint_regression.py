# ruff: noqa: E402
"""Pinned upstream all-fixed optimization must not bypass feasibility checks."""

import pytest

pytest.importorskip("torch")


import torch

from botorch_rs.exceptions.errors import CandidateGenerationError
from botorch_rs.optim import optimize_acqf

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


class Quadratic(torch.nn.Module):
    _tensor_backend = "torch"

    def forward(self, X):
        return -X.square().sum(dim=(-2, -1))


@pytest.mark.parametrize("device", DEVICES)
def test_all_fixed_optimizer_rejects_constraint_ignored_by_pinned_reference(device):
    from botorch.optim import optimize_acqf as reference

    bounds = torch.tensor([[0.0], [1.0]], dtype=torch.double, device=device)
    indices = torch.tensor([0], device=device)
    coefficients = bounds.new_tensor([1.0])
    options = {
        "bounds": bounds,
        "q": 1,
        "num_restarts": 1,
        "batch_initial_conditions": bounds.new_tensor([[[0.5]]]),
        "fixed_features": {0: 0.0},
        "inequality_constraints": [(indices, coefficients, 0.5)],
    }

    # Official b0dd01be's all-fixed shortcut never receives the constraints.
    candidate, value = reference(Quadratic(), **options)
    torch.testing.assert_close(candidate, bounds.new_zeros(1, 1))
    torch.testing.assert_close(value, bounds.new_zeros(()))
    residual = (candidate[..., indices] * coefficients).sum(-1) - 0.5
    assert bool((residual == -0.5).all())

    # The port must reject this impossible combination, including on CUDA.
    with pytest.raises(CandidateGenerationError, match="feasible"):
        optimize_acqf(Quadratic(), **options)

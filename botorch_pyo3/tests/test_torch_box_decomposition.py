# ruff: noqa: E402
"""Exact general-dimensional Tensor local-upper-bound geometry."""

import pytest

pytest.importorskip("torch")


import torch

from botorch_rs.acquisition._torch_box_decomposition import partition_nd

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


def sorted_rows(values):
    for dimension in reversed(range(values.shape[-1])):
        values = values[values[:, dimension].argsort(stable=True)]
    return values


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("outcomes", [3, 4])
def test_general_tensor_cells_match_fast_partitioning(device, outcomes):
    pytest.importorskip("botorch")
    from botorch.utils.multi_objective.box_decompositions.non_dominated import (
        FastNondominatedPartitioning,
    )

    generator = torch.Generator(device=device).manual_seed(13)
    Y = (
        torch.rand((2, 3, 5, outcomes), device=device, dtype=torch.double, generator=generator)
        * 1.2
        - 0.1
    )
    ref = torch.zeros(outcomes, device=device, dtype=torch.double)
    feasible = torch.rand((2, 3, 5), device=device, generator=generator) > 0.2
    feasible[0, 0] = False
    lower, upper, mask = partition_nd(Y, ref, feasible)
    assert lower.device == Y.device
    actual = torch.cat((lower, upper), -1).reshape(6, -1, 2 * outcomes)
    for i, front in enumerate(Y.reshape(6, 5, outcomes)):
        partition = FastNondominatedPartitioning(ref, Y=front[feasible.reshape(6, 5)[i]])
        bounds = partition.get_hypercell_bounds()
        expected = torch.cat((bounds[0], bounds[1]), -1)
        torch.testing.assert_close(
            sorted_rows(actual[i, mask.reshape(6, -1)[i]]),
            sorted_rows(expected),
            rtol=1e-12,
            atol=1e-12,
        )


@pytest.mark.parametrize("device", DEVICES)
def test_general_tensor_geometry_retains_boundary_gradients_without_host_copy(device, monkeypatch):
    Y = torch.tensor(
        [[0.1, 0.6, 0.3], [0.3, 0.3, 0.5], [0.7, 0.1, 0.2]], device=device, dtype=torch.double
    ).requires_grad_()
    ref = torch.zeros(3, device=device, dtype=torch.double, requires_grad=True)

    def forbidden(*args, **kwargs):
        raise AssertionError("Box geometry copied floating point data to the host")

    for name in ("cpu", "tolist", "numpy"):
        monkeypatch.setattr(torch.Tensor, name, forbidden)

    def evaluate(front, reference):
        lower, upper, mask = partition_nd(front, reference)
        finite = upper.clamp_max(2)
        volumes = (finite - lower).prod(-1)
        return volumes.masked_fill(~mask, 0).sum()

    assert torch.autograd.gradcheck(evaluate, (Y, ref), fast_mode=True)
    assert torch.autograd.gradgradcheck(evaluate, (Y, ref), fast_mode=True)

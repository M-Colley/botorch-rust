"""Graph and device parity for analytic HV, MOMF, ParEGO and scalarization."""

# ruff: noqa: E402
from __future__ import annotations

import warnings
from unittest.mock import patch

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("botorch")

from botorch.acquisition.multi_objective.analytic import (
    ExpectedHypervolumeImprovement as ReferenceEHVI,
)
from botorch.acquisition.multi_objective.multi_fidelity import MOMF as ReferenceMOMF
from botorch.acquisition.multi_objective.parego import qLogNParEGO as ReferenceParego
from botorch.utils.multi_objective.box_decompositions.non_dominated import (
    FastNondominatedPartitioning as ReferencePartition,
)
from botorch.utils.multi_objective.scalarization import (
    get_chebyshev_scalarization as reference_scalarization,
)
from test_torch_backend_acquisition import FixedSampler
from test_torch_backend_hypervolume import data
from test_torch_regularized_acquisitions import _compare_graphs

from botorch_rs.acquisition.multi_objective.analytic import ExpectedHypervolumeImprovement
from botorch_rs.acquisition.multi_objective.multi_fidelity import MOMF
from botorch_rs.acquisition.multi_objective.parego import qLogNParEGO
from botorch_rs.models import SingleTaskGP
from botorch_rs.utils.multi_objective.box_decompositions.non_dominated import (
    FastNondominatedPartitioning,
)
from botorch_rs.utils.multi_objective.scalarization import get_chebyshev_scalarization

DEVICES = [
    "cpu",
    pytest.param(
        "cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
    ),
]


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("count", [0, 1, 4])
@pytest.mark.parametrize("negative", [False, True])
def test_scalarization_full_weight_observation_and_query_graphs(device, count, negative):
    if count == 0 and negative:
        pytest.skip("The reference rejects negative weights without observations.")
    observed = torch.tensor(
        [[0.2, 0.7], [0.8, 0.4], [0.5, 0.9], [0.6, 0.2]], dtype=torch.double, device=device
    )[:count].requires_grad_()
    weights = observed.new_tensor([0.3, -0.7 if negative else 0.7], requires_grad=True)
    query = observed.new_tensor(
        [[[0.15, 0.33], [0.67, 0.81]], [[0.5, 0.23], [0.73, 0.15]]], requires_grad=True
    )
    with (
        patch.object(torch.Tensor, "cpu", side_effect=AssertionError("host transfer")),
        patch.object(torch.Tensor, "numpy", side_effect=AssertionError("host transfer")),
        patch.object(torch.Tensor, "tolist", side_effect=AssertionError("host transfer")),
    ):
        own = get_chebyshev_scalarization(weights, observed)
        reference = reference_scalarization(weights, observed)
        _compare_graphs(own(query), reference(query), (query, weights, observed))


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("outcomes", [2, 3])
def test_analytic_hypervolume_complete_graph_matches_reference(device, outcomes):
    x, y, ref, observed = data(device, outcomes)
    model = SingleTaskGP(x, y, backend="torch")
    query = x.new_tensor([[[0.21]], [[0.56]]], requires_grad=True)
    own = ExpectedHypervolumeImprovement(model, ref, FastNondominatedPartitioning(ref, Y=observed))
    reference = ReferenceEHVI(model, ref, ReferencePartition(ref, Y=observed))
    _compare_graphs(own(query), reference(query), (query, x, y, *model.parameters()))


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("custom_cost", [False, True])
def test_momf_cost_and_model_graphs_match_reference(device, custom_cost):
    x, y, ref, observed = data(device, 3)
    model = SingleTaskGP(x, y, backend="torch")
    query = x.new_tensor([[[0.21], [0.56]], [[0.43], [0.85]]], requires_grad=True)
    scale = x.new_tensor(1.3, requires_grad=True)

    def cost(X):
        return 0.4 + scale * X[..., -1:].exp()

    kwargs = {"sampler": FixedSampler(6, device), "cost_call": cost if custom_cost else None}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        own = MOMF(model, ref, FastNondominatedPartitioning(ref, Y=observed), **kwargs)
        reference = ReferenceMOMF(model, ref, ReferencePartition(ref, Y=observed), **kwargs)
    leaves = (query, x, y, *model.parameters(), *((scale,) if custom_cost else ()))
    _compare_graphs(own(query), reference(query), leaves)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("negative", [False, True])
def test_parego_scalarization_and_noisy_improvement_match_reference(device, negative):
    x, y, _, _ = data(device, 2)
    model = SingleTaskGP(x, y, backend="torch")
    baseline = x.detach()[::2]
    query = x.new_tensor([[[0.21], [0.56]], [[0.43], [0.85]]], requires_grad=True)
    weights = x.new_tensor([0.3, -0.7 if negative else 0.7], requires_grad=True)
    kwargs = {
        "sampler": FixedSampler((len(baseline) + 2) * 2, device),
        "scalarization_weights": weights,
        "cache_root": False,
        "prune_baseline": False,
    }
    own = qLogNParEGO(model, baseline, **kwargs)
    reference = ReferenceParego(model, baseline, **kwargs)
    _compare_graphs(own(query), reference(query), (query, x, y, weights, *model.parameters()))
    torch.testing.assert_close(own.Y_baseline, reference.Y_baseline)
    torch.testing.assert_close(own.scalarization_weights, reference.scalarization_weights)

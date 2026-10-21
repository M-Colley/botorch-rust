"""Fixed native path/output ensemble selection and tensor pullbacks."""

from __future__ import annotations

import math

import pytest

from helpers import KINDS, as_kind, assert_close, importorskip_botorch

importorskip_botorch()
torch = pytest.importorskip("torch")

from botorch.acquisition.objective import ScalarizedPosteriorTransform as BtTransform  # noqa: E402
from botorch.acquisition.thompson_sampling import PathwiseThompsonSampling as BtTS  # noqa: E402
from botorch.models import ModelListGP as BtList  # noqa: E402
from fully_bayesian_helpers import model_pair  # noqa: E402

from botorch_rs.acquisition.objective import ScalarizedPosteriorTransform  # noqa: E402
from botorch_rs.acquisition.thompson_sampling import PathwiseThompsonSampling  # noqa: E402
from botorch_rs.models import ModelListGP, SingleTaskGP  # noqa: E402


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("outputs", [1, 2])
def test_selection_matches_official_for_batched_paths(kind, outputs):
    pairs = [
        model_pair("saas", d=2, n=5, num=3, seed=741 + i, transforms=False) for i in range(outputs)
    ]
    if outputs == 1:
        reference = BtTS(pairs[0][0])
        actual = PathwiseThompsonSampling(pairs[0][1], num_features=16, seed=7)
    else:
        weights = torch.ones(outputs, dtype=torch.double) / outputs
        reference = BtTS(BtList(*(p[0] for p in pairs)), posterior_transform=BtTransform(weights))
        actual = PathwiseThompsonSampling(
            ModelListGP(*(p[1] for p in pairs)),
            posterior_transform=ScalarizedPosteriorTransform(weights),
            num_features=16,
            seed=7,
        )
    indices = (torch.arange(2 * outputs) % 3).reshape(2, 1, outputs)
    reference.ensemble_indices = indices
    actual.redraw(2, ensemble_indices=indices)
    shape = (2, 4, 2, 3, outputs)
    data = torch.linspace(-0.8, 0.9, math.prod(shape), dtype=torch.double).reshape(shape)
    values = as_kind(data.tolist(), kind)
    if kind == "torch":
        values.requires_grad_()
    expected = reference.select_from_ensemble_models(data)
    selected = actual.select_from_ensemble_models(values)
    assert_close(selected, expected)
    # Selection is deterministic and uses the actual cached draw indices.
    assert_close(actual.select_from_ensemble_models(values), selected)
    if kind == "torch":
        (gradient,) = torch.autograd.grad(selected.sum(), values)
        expected_values = data.clone().requires_grad_()
        (expected_gradient,) = torch.autograd.grad(
            reference.select_from_ensemble_models(expected_values).sum(), expected_values
        )
        torch.testing.assert_close(gradient, expected_gradient, rtol=0, atol=0)


def test_single_model_selection_preserves_input_and_graph():
    x = torch.tensor([[0.0], [0.5], [1.0]], dtype=torch.double)
    model = SingleTaskGP(x, x.square())
    acquisition = PathwiseThompsonSampling(model, num_features=16)
    values = torch.ones(2, 1, dtype=torch.double, requires_grad=True)
    assert acquisition.select_from_ensemble_models(values) is values

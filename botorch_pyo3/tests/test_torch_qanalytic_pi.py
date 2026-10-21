"""Batch analytic PI retains device Gaussian and threshold derivative graphs."""

# ruff: noqa: E402
from __future__ import annotations

from unittest.mock import patch

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("botorch")

from botorch.acquisition.analytic import qAnalyticProbabilityOfImprovement as ReferencePI
from test_torch_backend_acquisition import TensorModel

from botorch_rs.acquisition.analytic import qAnalyticProbabilityOfImprovement
from botorch_rs.exceptions.errors import UnsupportedError

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


def _first(value, leaves, create_graph=False):
    return torch.autograd.grad(value.sum(), leaves, create_graph=create_graph, retain_graph=True)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("q", [1, 2, 3, 4])
@pytest.mark.parametrize("maximize", [False, True])
def test_batch_probability_values_complete_gradients_and_directional_hessian(device, q, maximize):
    model = TensorModel(device)
    x = model.train_X.new_tensor([[[0.18], [0.48], [0.82], [0.94]]])[:, :q].requires_grad_()
    threshold = x.new_tensor([0.23], requires_grad=True)
    actual = qAnalyticProbabilityOfImprovement(model, threshold, maximize=maximize)
    reference = ReferencePI(model, threshold, maximize=maximize)
    leaves = (x, threshold, *model.parameters())
    value, expected = actual(x), reference(x)
    assert value.device == x.device
    torch.testing.assert_close(value, expected, atol=2e-10, rtol=2e-9)
    gradients = _first(value, leaves, create_graph=True)
    expected_gradients = _first(expected, leaves)
    for a, b in zip(gradients, expected_gradients, strict=True):
        torch.testing.assert_close(a, b, atol=3e-8, rtol=3e-7)
    directions = [
        torch.linspace(0.4, 0.8, leaf.numel(), device=device, dtype=leaf.dtype).reshape_as(leaf)
        for leaf in leaves
    ]
    hvp = _first(sum((g * d).sum() for g, d in zip(gradients, directions, strict=True)), leaves)
    # The pinned source's in-place solver loses parts of its Hessian. Its
    # memberwise first derivatives provide an independent central-difference
    # directional oracle without modifying or substituting the source solver.
    epsilon = 2e-5
    displaced = []
    originals = [leaf.detach().clone() for leaf in leaves]
    try:
        for sign in (-1, 1):
            with torch.no_grad():
                for leaf, base, direction in zip(leaves, originals, directions, strict=True):
                    leaf.copy_(base + sign * epsilon * direction)
            displaced.append(_first(reference(x), leaves))
    finally:
        with torch.no_grad():
            for leaf, base in zip(leaves, originals, strict=True):
                leaf.copy_(base)
    for actual_hvp, minus, plus in zip(hvp, *displaced, strict=True):
        torch.testing.assert_close(actual_hvp, (plus - minus) / (2 * epsilon), atol=3e-5, rtol=3e-4)


@pytest.mark.parametrize("device", DEVICES)
def test_probability_batched_thresholds_and_query_graphs_do_not_cross_host(device):
    model = TensorModel(device)
    x = model.train_X.new_tensor([[[0.18], [0.48]], [[0.26], [0.78]]], requires_grad=True)
    threshold = x.new_tensor([0.13, 0.29], requires_grad=True)
    actual = qAnalyticProbabilityOfImprovement(model, threshold)
    reference = ReferencePI(model, threshold)
    expected = reference(x)
    with (
        patch.object(torch.Tensor, "cpu", side_effect=AssertionError("host transfer")),
        patch.object(torch.Tensor, "numpy", side_effect=AssertionError("host transfer")),
        patch.object(torch.Tensor, "tolist", side_effect=AssertionError("host transfer")),
    ):
        value = actual(x)
        gradients = _first(value, (x, threshold, *model.parameters()), create_graph=True)
        second = _first(
            sum(g.square().sum() for g in gradients), (x, threshold, *model.parameters())
        )
    torch.testing.assert_close(value, expected, atol=2e-10, rtol=2e-9)
    assert all(g.device == x.device and g.isfinite().all() for g in (*gradients, *second))
    with pytest.raises(UnsupportedError, match="pending"):
        actual.set_X_pending(x)


@pytest.mark.parametrize("device", DEVICES)
def test_probability_mixed_query_and_threshold_gradgradcheck(device):
    model = TensorModel(device)
    x = model.train_X.new_tensor([[0.21], [0.53], [0.87]], requires_grad=True)
    threshold = x.new_tensor(0.19, requires_grad=True)

    def probability(query, best):
        return qAnalyticProbabilityOfImprovement(model, best)(query)

    assert torch.autograd.gradcheck(probability, (x, threshold), atol=3e-5, rtol=3e-4)
    assert torch.autograd.gradgradcheck(probability, (x, threshold), atol=3e-5, rtol=3e-4)


@pytest.mark.parametrize("device", DEVICES)
def test_probability_scalarized_multioutput_model_and_live_weights(device):
    from botorch.acquisition.objective import ScalarizedPosteriorTransform as ReferenceScalarize
    from botorch.models import SingleTaskGP as ReferenceGP

    from botorch_rs.acquisition.objective import ScalarizedPosteriorTransform
    from botorch_rs.models import SingleTaskGP

    x = torch.tensor([[0.03], [0.29], [0.58], [0.87]], device=device, dtype=torch.double)
    y = torch.cat(((5 * x).sin(), (3 * x).cos()), dim=-1)
    model = SingleTaskGP(x, y, backend="torch")
    ref_model = ReferenceGP(x, y)
    weights = x.new_tensor([0.7, -0.3], requires_grad=True)
    own = qAnalyticProbabilityOfImprovement(model, 0.2, ScalarizedPosteriorTransform(weights))
    reference = ReferencePI(ref_model, 0.2, ReferenceScalarize(weights))
    query = x.new_tensor([[[0.19], [0.71]]], requires_grad=True)
    actual, expected = own(query), reference(query)
    torch.testing.assert_close(actual, expected, atol=3e-8, rtol=3e-8)
    gradients = _first(actual, (query, weights), create_graph=True)
    expected_gradients = _first(expected, (query, weights))
    for a, b in zip(gradients, expected_gradients, strict=True):
        torch.testing.assert_close(a, b, atol=3e-6, rtol=3e-6)
    second = _first(sum(g.square().sum() for g in gradients), (query, weights))
    assert all(g.isfinite().all() for g in second)
    with pytest.raises(UnsupportedError, match="posterior transform"):
        qAnalyticProbabilityOfImprovement(model, 0.2)


@pytest.mark.parametrize("device", DEVICES)
def test_probability_loaded_ensemble_reduces_component_probabilities(device):
    from test_torch_fully_bayesian import _pair

    (model, reference_model), data, _ = _pair(device, "SaasFullyBayesianSingleTaskGP")
    query = data[0][0].new_tensor([[[0.21, 0.4, 0.6], [0.7, 0.3, 0.2]]], requires_grad=True)
    actual = qAnalyticProbabilityOfImprovement(model, 0.35)(query)
    expected = ReferencePI(reference_model, 0.35)(query)
    torch.testing.assert_close(actual, expected, atol=3e-8, rtol=3e-8)
    assert actual.shape == (1,)
    gradient = _first(actual, (query,), create_graph=True)[0]
    expected_gradient = _first(expected, (query,))[0]
    torch.testing.assert_close(gradient, expected_gradient, atol=3e-6, rtol=3e-6)
    assert _first(gradient.square().sum(), (query,))[0].isfinite().all()

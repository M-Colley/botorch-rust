# ruff: noqa: E402
"""Public risk-measure and expanded-prior methods against official BoTorch."""

import pytest

pytest.importorskip("numpy")
pytest.importorskip("torch")
pytest.importorskip("botorch")


import numpy as np
import torch
from botorch.acquisition.multi_objective.multi_output_risk_measures import MARS as RefMARS
from botorch.acquisition.multi_objective.multi_output_risk_measures import MVaR as RefMVaR
from botorch.models.utils.priors import BetaPrior as RefBetaPrior

from botorch_rs._arrays import to_flat
from botorch_rs.acquisition.multi_objective.multi_output_risk_measures import MARS, MVaR
from botorch_rs.exceptions.errors import BotorchTensorDimensionError
from botorch_rs.models.utils.priors import BetaPrior

DT = torch.float64


def reference_beta(alpha, beta, **kwargs):
    try:
        return RefBetaPrior(alpha, beta, **kwargs)
    except AttributeError as error:
        # Released BoTorch 0.18.1's prior constructor deletes a read-only
        # concentration property with current PyTorch. The current official
        # source fixes this. Its underlying Beta distribution is the exact
        # density/expand reference when checking that older release.
        assert "concentration" in str(error) and "deleter" in str(error)
        return torch.distributions.Beta(alpha, beta, **kwargs)


@pytest.mark.parametrize("shape", [(), (2, 3), (0, 3), (4, 1, 3)])
def test_beta_expand_density_shapes_concentrations_and_input_gradients(shape):
    alpha, beta = torch.tensor(2.3, dtype=DT), torch.tensor(1.7, dtype=DT)
    original = BetaPrior(alpha, beta, validate_args=True)
    actual = original.expand(shape)
    try:
        reference = reference_beta(alpha, beta, validate_args=True).expand(shape)
    except RuntimeError as error:
        # Official BetaPrior.expand reconstructs and revalidates an empty
        # concentration tensor, hitting PyTorch's ambiguous reshape(-1).
        # The distribution's own expansion correctly preserves validation.
        assert 0 in shape and "ambiguous" in str(error)
        reference = torch.distributions.Beta(alpha, beta, validate_args=True).expand(shape)
    assert original.batch_shape == ()
    assert actual.batch_shape == tuple(reference.batch_shape)
    torch.testing.assert_close(actual.concentration1, reference.concentration1)
    torch.testing.assert_close(actual.concentration0, reference.concentration0)
    torch.testing.assert_close(actual.mode, reference.mode)
    x = torch.tensor(0.37, dtype=DT, requires_grad=True)
    y = x.detach().clone().requires_grad_(True)
    value = actual.log_prob(x)
    expected = reference.log_prob(y)
    torch.testing.assert_close(value, expected, atol=2e-12, rtol=2e-12)
    value.sum().backward()
    expected.sum().backward()
    torch.testing.assert_close(x.grad, y.grad, atol=2e-11, rtol=2e-11)


@pytest.mark.parametrize("kind", ["list", "numpy", "torch"])
def test_beta_expand_broadcast_validation_and_array_types(kind):
    prior = BetaPrior(2.0, 3.0, validate_args=True).expand((2, 1)).expand((4, 2, 3))
    reference = (
        reference_beta(torch.tensor(2.0, dtype=DT), torch.tensor(3.0, dtype=DT), validate_args=True)
        .expand((2, 1))
        .expand((4, 2, 3))
    )
    values = [0.2, 0.5, 0.8]
    argument = (
        values
        if kind == "list"
        else np.asarray(values)
        if kind == "numpy"
        else torch.tensor(values, dtype=DT)
    )
    result = prior.log_prob(argument)
    np.testing.assert_allclose(
        result, reference.log_prob(torch.tensor(values, dtype=DT)), atol=2e-12, rtol=2e-12
    )
    assert isinstance(result, {"list": list, "numpy": np.ndarray, "torch": torch.Tensor}[kind])
    assert prior.expand((-1, 2, -1)).batch_shape == (4, 2, 3)
    for shape in ((2,), (5, 2, 3), (-1, 4, 2, 3)):
        with pytest.raises(RuntimeError):
            prior.expand(shape)
    with pytest.raises(ValueError, match="support"):
        prior.log_prob(-0.1)
    with pytest.raises(ValueError, match="broadcast"):
        prior.log_prob(torch.zeros(5, dtype=DT))


@pytest.mark.parametrize("kind", ["list", "numpy", "torch"])
def test_mvar_make_differentiable_matches_ties_and_unattained_components(kind):
    prepared = torch.tensor(
        [
            [[1.0, 2.0], [1.0, 4.0], [3.0, 2.0], [5.0, 6.0]],
            [[3.0, 1.0], [3.0, 4.0], [2.0, 1.0], [8.0, 6.0]],
        ],
        dtype=DT,
    )
    points = torch.tensor(
        [[[1.0, 2.0], [5.0, 4.0], [7.0, 5.0]], [[3.0, 1.0], [2.0, 6.0], [7.0, 5.0]]], dtype=DT
    )
    expected = RefMVaR(n_w=4, alpha=0.5).make_differentiable(prepared, points)

    def convert(value):
        return value.tolist() if kind == "list" else value.numpy() if kind == "numpy" else value

    actual = MVaR(n_w=4, alpha=0.5).make_differentiable(convert(prepared), convert(points))
    np.testing.assert_allclose(actual, expected, equal_nan=True, atol=0.0, rtol=0.0)


def test_mvar_make_differentiable_exact_native_vjp_and_frozen_match_state():
    prepared = torch.tensor(
        [[[1.0, 2.0], [1.0, 4.0], [3.0, 2.0], [5.0, 6.0]]], dtype=DT, requires_grad=True
    )
    expected_prepared = prepared.detach().clone().requires_grad_(True)
    points = torch.tensor([[[1.0, 2.0], [5.0, 4.0], [7.0, 5.0]]], dtype=DT, requires_grad=True)
    reference_points = points.detach().clone().requires_grad_(True)
    actual = MVaR(n_w=4, alpha=0.5).make_differentiable(prepared, points)
    expected = RefMVaR(n_w=4, alpha=0.5).make_differentiable(expected_prepared, reference_points)
    weights = torch.tensor([[[2.0, 6.0], [7.0, 5.0], [9.0, 8.0]]], dtype=DT)
    (actual * weights).sum().backward()
    (expected * weights).sum().backward()
    torch.testing.assert_close(prepared.grad, expected_prepared.grad, atol=0.0, rtol=0.0)
    assert points.grad is None and reference_points.grad is None
    with pytest.raises(BotorchTensorDimensionError):
        MVaR(n_w=4, alpha=0.5).make_differentiable(prepared[0], points)


@pytest.mark.parametrize("with_reference", [False, True])
def test_mars_chebyshev_objective_values_gradients_and_cache_invalidation(with_reference):
    baseline = torch.tensor([[1.0, 4.0], [2.0, 3.0], [4.0, 1.0]], dtype=DT)
    reference_point = torch.tensor([0.5, 0.5], dtype=DT) if with_reference else None
    weights = torch.tensor([0.3, 0.7], dtype=DT)
    objective = MARS(
        alpha=0.5, n_w=2, chebyshev_weights=weights, baseline_Y=baseline, ref_point=reference_point
    )
    reference = RefMARS(
        alpha=0.5, n_w=2, chebyshev_weights=weights, baseline_Y=baseline, ref_point=reference_point
    )
    samples = torch.tensor(
        [[[1.2, 3.1], [2.7, 1.8], [4.1, 0.9], [2.0, 3.3]]], dtype=DT, requires_grad=True
    )
    ref_samples = samples.detach().clone().requires_grad_(True)
    old = objective.chebyshev_objective
    old_reference = reference.chebyshev_objective
    assert old is objective.chebyshev_objective
    actual = old(samples)
    expected = old_reference(ref_samples)
    torch.testing.assert_close(actual, expected, atol=2e-12, rtol=2e-12)
    actual.sum().backward()
    expected.sum().backward()
    torch.testing.assert_close(samples.grad, ref_samples.grad, atol=2e-12, rtol=2e-12)
    updated = torch.tensor([0.8, 0.2], dtype=DT)
    objective.chebyshev_weights = updated
    reference.chebyshev_weights = updated
    assert objective.chebyshev_objective is not old
    # Old callables retain their bounds but read the current weights, upstream.
    torch.testing.assert_close(old(samples), old_reference(samples), atol=2e-12, rtol=2e-12)
    objective.baseline_Y = baseline * 1.5
    reference.baseline_Y = baseline * 1.5
    torch.testing.assert_close(
        objective.chebyshev_objective(samples),
        reference.chebyshev_objective(samples),
        atol=2e-12,
        rtol=2e-12,
    )
    objective.baseline_Y = None
    with pytest.raises(RuntimeError, match="baseline_Y"):
        _ = objective.chebyshev_objective


@pytest.mark.parametrize("kind", ["list", "numpy", "torch"])
def test_mars_chebyshev_objective_nonfinite_values_and_tied_gradients(kind):
    baseline = [[0.0, 2.0], [2.0, 0.0]]
    objective = MARS(alpha=0.5, n_w=2, chebyshev_weights=[0.5, 0.5], baseline_Y=baseline)
    values = [[1.0, 1.0], [float("inf"), 1.0], [float("-inf"), 1.0]]
    samples = (
        values
        if kind == "list"
        else np.asarray(values)
        if kind == "numpy"
        else torch.tensor(values, dtype=DT, requires_grad=True)
    )
    result = objective.chebyshev_objective(samples)
    np.testing.assert_allclose(to_flat(result)[0], [0.25, 0.25, -np.inf], atol=0.0, rtol=0.0)
    if kind == "torch":
        result.sum().backward()
        torch.testing.assert_close(
            samples.grad, torch.tensor([[0.25, 0.0], [0.0, 0.25], [0.25, 0.0]], dtype=DT)
        )

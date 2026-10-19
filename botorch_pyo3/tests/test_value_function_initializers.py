"""Batched restart selection and KG value-function initialization."""

from __future__ import annotations

import importlib
import math
from types import SimpleNamespace

import pytest

from botorch_rs.optim import initializers as rs
from helpers import KINDS, as_kind, to_list


@pytest.fixture
def corrected_boltzmann_reference(monkeypatch):
    """Correct official standardize(dim=-2) to normalize its category axis."""
    sampling = pytest.importorskip("botorch.utils.sampling")
    original = sampling.standardize

    def standardize_categories(values):
        return original(values) if values.ndim == 1 else original(values.unsqueeze(-1)).squeeze(-1)

    monkeypatch.setattr(sampling, "standardize", standardize_categories)


@pytest.mark.parametrize("batch", [(), (2,), (2, 3)])
@pytest.mark.parametrize("replacement", [False, True])
def test_boltzmann_categories_match_corrected_reference(
    batch: tuple[int, ...], replacement: bool, corrected_boltzmann_reference
) -> None:
    torch = pytest.importorskip("torch")
    bt = pytest.importorskip("botorch.utils.sampling")
    from botorch_rs.utils.sampling import boltzmann_sample

    values = torch.randn(9, *batch, dtype=torch.double, generator=torch.Generator().manual_seed(31))
    values = values.permute(*range(1, len(batch) + 1), 0)
    for eta in (1.5, 10000.0):
        torch.manual_seed(18)
        actual = boltzmann_sample(values, 4, eta, replacement)
        torch.manual_seed(18)
        expected = bt.boltzmann_sample(values, 4, eta, replacement)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("kind", KINDS)
def test_boltzmann_batched_weights_follow_values(kind: str) -> None:
    from botorch_rs.utils.sampling import boltzmann_sample

    values = as_kind([[-5.0, 0.0, 8.0], [8.0, 0.0, -5.0]], kind)
    result = to_list(boltzmann_sample(values, 3000, 2.0, replacement=True))
    assert sum(index == 2 for index in result[0]) / 3000 > 0.8
    assert sum(index == 0 for index in result[1]) / 3000 > 0.8
    with pytest.raises(ValueError, match="eta must be finite"):
        boltzmann_sample(values, 1, math.inf)
    with pytest.raises(ValueError, match="temp_decrease"):
        boltzmann_sample(values, 1, 1.0, temp_decrease=1.0)


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("batch", [(), (2,), (2, 3)])
def test_restart_selection_batches(
    kind: str, batch: tuple[int, ...], corrected_boltzmann_reference
) -> None:
    torch = pytest.importorskip("torch")
    bt = pytest.importorskip("botorch.optim.initializers")
    generator = torch.Generator().manual_seed(71)
    points = torch.rand(9, *batch, 1, 2, dtype=torch.double, generator=generator)
    values = torch.randn(9, *batch, dtype=torch.double, generator=generator)
    X, Y = as_kind(points.tolist(), kind), as_kind(values.tolist(), kind)
    torch.manual_seed(19)
    actual, chosen = rs.initialize_q_batch(X, Y, n=3, eta=1.5)
    assert len(to_list(actual)) == 3
    selected = torch.tensor(to_list(chosen), dtype=torch.double).reshape(3, -1)
    maxima = values.reshape(9, -1).max(dim=0).values
    assert (selected == maxima).any(dim=0).all()
    if kind == "torch":
        torch.manual_seed(19)
        expected, expected_chosen = bt.initialize_q_batch(points, values, n=3, eta=1.5)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        torch.testing.assert_close(chosen, expected_chosen, rtol=0, atol=0)


@pytest.mark.parametrize("batch", [(), (2,), (2, 3)])
def test_restart_selection_retains_gradients(
    batch: tuple[int, ...], corrected_boltzmann_reference
) -> None:
    torch = pytest.importorskip("torch")
    bt = pytest.importorskip("botorch.optim.initializers")
    points = (
        torch.arange(8 * math.prod(batch) * 2, dtype=torch.double)
        .reshape(8, *batch, 1, 2)
        .requires_grad_()
    )
    values = (
        torch.arange(8 * math.prod(batch), dtype=torch.double).reshape(8, *batch).requires_grad_()
    )
    torch.manual_seed(8)
    actual = rs.initialize_q_batch(points, values, n=3)
    actual_grad = torch.autograd.grad(actual[0].sum() + actual[1].sum(), (points, values))
    torch.manual_seed(8)
    expected = bt.initialize_q_batch(points, values, n=3)
    expected_grad = torch.autograd.grad(expected[0].sum() + expected[1].sum(), (points, values))
    for a, e in zip(actual_grad, expected_grad, strict=True):
        torch.testing.assert_close(a, e, rtol=0, atol=0)


def test_one_constant_batch_uses_shared_random_subset() -> None:
    torch = pytest.importorskip("torch")
    bt = pytest.importorskip("botorch.optim.initializers")
    points = torch.arange(8 * 2 * 2, dtype=torch.double).reshape(8, 2, 1, 2)
    values = torch.stack([torch.zeros(8), torch.arange(8)], dim=-1).double()
    torch.manual_seed(8)
    with pytest.warns(Warning, match="same or not finite"):
        actual = rs.initialize_q_batch(points, values, n=3)
    torch.manual_seed(8)
    with pytest.warns(Warning, match="same or not finite"):
        expected = bt.initialize_q_batch(points, values, n=3)
    for a, e in zip(actual, expected, strict=True):
        torch.testing.assert_close(a, e, rtol=0, atol=0)


@pytest.mark.parametrize("batch", [(), (3,), (2, 3)])
@pytest.mark.parametrize("frac_random", [0.1, 0.99])
def test_value_initializer_parity_after_current_model_optimization(
    monkeypatch, batch, frac_random, corrected_boltzmann_reference
) -> None:
    torch = pytest.importorskip("torch")
    bt = pytest.importorskip("botorch.optim.initializers")
    bt_opt = importlib.import_module("botorch.optim.optimize")
    rs_opt = importlib.import_module("botorch_rs.optim.optimize")
    rs_kg = importlib.import_module("botorch_rs.acquisition.knowledge_gradient")
    calls = []
    candidates = torch.tensor(
        [[[0.1, 0.2]], [[0.3, 0.4]], [[0.7, 0.6]], [[0.9, 0.8]]], dtype=torch.double
    )
    values = torch.tensor([-0.8, 0.2, 0.5, -0.1], dtype=torch.double)

    def optimizer(**kwargs):
        calls.append(kwargs)
        return candidates, values

    monkeypatch.setattr(bt_opt, "optimize_acqf", optimizer)
    monkeypatch.setattr(rs_opt, "optimize_acqf", optimizer)
    monkeypatch.setattr(bt, "_get_value_function", lambda **kwargs: "current value")
    monkeypatch.setattr(rs_kg, "_get_value_function", lambda **kwargs: "current value")

    class ValueFunction:
        model = SimpleNamespace(batch_shape=torch.Size(batch))
        posterior_transform = None

        def __call__(self, X):
            return -((X - 0.4) ** 2).sum(dim=(-1, -2))

    bounds = torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.double)
    kwargs = dict(
        acq_function=ValueFunction(),
        bounds=bounds,
        num_restarts=3,
        raw_samples=12,
        current_model=object(),
        fixed_features={1: 0.25},
        options={
            "frac_random": frac_random,
            "num_inner_restarts": 4,
            "raw_inner_samples": 8,
            "eta": 1.5,
            "seed": 13,
            "maxiter": 9,
        },
    )
    torch.manual_seed(99)
    actual = rs.gen_value_function_initial_conditions(**kwargs)
    torch.manual_seed(99)
    expected = bt.gen_value_function_initial_conditions(**kwargs)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert actual.shape == (3, *batch, 1, 2)
    assert torch.equal(actual[..., 1], torch.full_like(actual[..., 1], 0.25))
    for call in calls:
        assert call["options"] == {"seed": 13, "maxiter": 9}
        assert call["fixed_features"] == {1: 0.25}


def test_higher_current_fantasy_values_increase_resampling_frequency(monkeypatch) -> None:
    torch = pytest.importorskip("torch")
    rs_opt = importlib.import_module("botorch_rs.optim.optimize")
    rs_kg = importlib.import_module("botorch_rs.acquisition.knowledge_gradient")
    candidates = torch.tensor([[[0.1]], [[0.2]], [[0.9]]], dtype=torch.double)
    values = torch.tensor([-5.0, 0.0, 8.0], dtype=torch.double)
    monkeypatch.setattr(rs_opt, "optimize_acqf", lambda **kwargs: (candidates, values))
    monkeypatch.setattr(rs_kg, "_get_value_function", lambda **kwargs: None)

    class ValueFunction:
        model = SimpleNamespace(batch_shape=(4,))
        posterior_transform = None

        def __call__(self, X):
            return torch.zeros(X.shape[:-2], dtype=X.dtype)

    torch.manual_seed(19)
    result = rs.gen_value_function_initial_conditions(
        ValueFunction(),
        torch.tensor([[0.0], [1.0]], dtype=torch.double),
        num_restarts=1000,
        raw_samples=1000,
        current_model=object(),
        options={"frac_random": 0.01, "eta": 2.0, "seed": 9},
    )
    # Correct weighting gives the best optimizer about 90% of the mass.
    # Standardizing expanded rows across batches instead would give 1/3.
    frequency = (result[:990, :, 0, 0] == 0.9).double().mean(dim=0)
    assert (frequency > 0.8).all(), frequency


@pytest.mark.parametrize("kind", KINDS)
def test_value_initializer_real_native_optimizer(kind: str) -> None:
    from botorch_rs.acquisition.analytic import PosteriorMean
    from botorch_rs.models import SingleTaskGP

    X = as_kind([[0.0], [0.3], [0.8], [1.0]], kind)
    Y = as_kind([[0.0], [0.7], [0.3], [-0.2]], kind)
    model = SingleTaskGP(X, Y)
    value = PosteriorMean(model)
    bounds = as_kind([[0.0], [1.0]], kind)
    actual = rs.gen_value_function_initial_conditions(
        value,
        bounds,
        3,
        12,
        model,
        options={"num_inner_restarts": 3, "raw_inner_samples": 8, "seed": 15, "maxiter": 20},
    )
    points = to_list(actual)
    assert len(points) == 3
    assert all(0 <= point[0][0] <= 1 for point in points)
    assert max(to_list(value(actual))) > 0.5

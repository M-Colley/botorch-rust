"""Batched closure state and gradients against official BoTorch."""

from __future__ import annotations

import pytest

from botorch_rs.optim.batched_lbfgs_b import translate_bounds_for_lbfgsb
from botorch_rs.optim.closures import BatchedNDarrayOptimizationClosure
from helpers import KINDS, as_kind


@pytest.mark.parametrize("batch", [(), (3,), (2, 3)])
@pytest.mark.parametrize("dtype_name", ["float32", "float64"])
def test_batched_closure_state_values_and_gradients(
    batch: tuple[int, ...], dtype_name: str
) -> None:
    torch = pytest.importorskip("torch")
    np = pytest.importorskip("numpy")
    bt = pytest.importorskip("botorch.optim.closures.core")
    seed = (
        torch.arange(
            max(1, torch.Size(batch).numel()) * 2, dtype=getattr(torch, dtype_name)
        ).reshape(*batch, 2)
        / 10
    )
    wrappers = []
    for implementation in (BatchedNDarrayOptimizationClosure, bt.BatchedNDarrayOptimizationClosure):
        weight = seed.clone().requires_grad_()
        bias = torch.ones(batch, dtype=weight.dtype, requires_grad=True)
        unused = torch.ones(*batch, 1, dtype=weight.dtype, requires_grad=True)
        params = {"weight": weight, "bias": bias, "unused": unused}

        def forward(scale=1.0, weight=weight, bias=bias):
            return scale * weight.square().sum(dim=-1) + bias.square()

        wrappers.append(implementation(forward, params, torch.Size(batch)))
    actual, expected = wrappers
    np.testing.assert_array_equal(actual.state, expected.state)
    state = actual.state + 0.13
    for a, e in zip(actual(state, scale=1.7), expected(state, scale=1.7), strict=True):
        np.testing.assert_array_equal(a, e)
        assert a.dtype == np.float64
    np.testing.assert_array_equal(actual.state, expected.state)
    indices = np.array([0]) if actual.batch_size == 1 else np.array([0, actual.batch_size - 1])
    partial = actual.state[indices] + 0.21
    for a, e in zip(
        actual(partial, batch_indices=indices),
        expected(partial, batch_indices=indices),
        strict=True,
    ):
        np.testing.assert_array_equal(a, e)
    np.testing.assert_array_equal(actual.state, expected.state)
    # Existing gradients must be reset on each evaluation.
    for a, e in zip(actual(), expected(), strict=True):
        np.testing.assert_array_equal(a, e)


@pytest.mark.parametrize("kind", KINDS)
def test_native_mll_batched_closure_gradients(kind: str) -> None:
    np = pytest.importorskip("numpy")
    from botorch_rs.mlls import ExactMarginalLogLikelihood
    from botorch_rs.models import SingleTaskGP
    from botorch_rs.optim.closures import get_loss_closure
    from botorch_rs.optim.utils.model_utils import get_parameters

    model = SingleTaskGP(
        as_kind([[0.0], [0.3], [0.8], [1.0]], kind),
        as_kind([[0.0, 0.2], [0.7, 0.8], [0.3, 0.4], [-0.2, 0.1]], kind),
    )
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    closure = BatchedNDarrayOptimizationClosure(get_loss_closure(mll), get_parameters(mll), (2,))
    initial = closure.state
    values, gradients = closure(initial)
    assert values.shape == (2,)
    assert gradients.shape == initial.shape
    assert np.isfinite(values).all() and np.isfinite(gradients).all()
    for batch in range(2):
        for column in range(closure._per_element_size):
            step = np.zeros_like(initial)
            step[batch, column] = 1e-6
            upper, _ = closure(initial + step)
            lower, _ = closure(initial - step)
            assert gradients[batch, column] == pytest.approx(
                (upper[batch] - lower[batch]) / 2e-6, abs=2e-5, rel=2e-5
            )
    closure.state = initial
    np.testing.assert_array_equal(closure.state, initial)
    subset_values, subset_gradients = closure(initial[[1]], batch_indices=np.array([1]))
    np.testing.assert_array_equal(subset_values, values[[1]])
    np.testing.assert_array_equal(subset_gradients, gradients[[1]])


def test_batched_closure_runtime_error_parity() -> None:
    torch = pytest.importorskip("torch")
    np = pytest.importorskip("numpy")
    bt = pytest.importorskip("botorch.optim.closures.core")
    from botorch_rs.exceptions.errors import NotPSDError

    def failing():
        raise RuntimeError("singular")

    for implementation in (BatchedNDarrayOptimizationClosure, bt.BatchedNDarrayOptimizationClosure):
        wrapper = implementation(
            failing, {"x": torch.ones(3, 2, requires_grad=True)}, torch.Size([3])
        )
        values, gradients = wrapper(batch_indices=np.array([0, 2]))
        assert np.isnan(values).all() and values.shape == (2,)
        np.testing.assert_array_equal(gradients, np.zeros((2, 2)))
    wrapper = BatchedNDarrayOptimizationClosure(
        failing, {"x": torch.ones(3, 2, requires_grad=True)}, (3,)
    )
    wrapper.forward = lambda: (_ for _ in ()).throw(NotPSDError("singular"))
    with pytest.raises(NotPSDError):
        wrapper()
    wrapper.forward = lambda: (_ for _ in ()).throw(RuntimeError("unrelated error"))
    with pytest.raises(RuntimeError, match="unrelated error"):
        wrapper()


@pytest.mark.parametrize(
    "bounds", [(None, None), (0.0, 1.0), ([0.0, None], [0.5, 1.0]), ([0.0] * 6, [1.0] * 6)]
)
def test_translate_bounds(bounds) -> None:
    bt = pytest.importorskip("botorch.optim.batched_lbfgs_b")
    assert translate_bounds_for_lbfgsb(
        *bounds, num_features=2, q=3
    ) == bt.translate_bounds_for_lbfgsb(*bounds, num_features=2, q=3)
    with pytest.raises(AssertionError, match="Instead got"):
        translate_bounds_for_lbfgsb([0.0] * 4, None, num_features=2, q=3)

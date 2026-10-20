"""Real subclass hooks and native optimization for the MES base protocol."""

from __future__ import annotations

import pytest

from helpers import importorskip_botorch

importorskip_botorch()
torch = pytest.importorskip("torch")

from botorch.acquisition.max_value_entropy_search import MaxValueBase as BtBase  # noqa: E402
from botorch.models import SingleTaskGP as BtGP  # noqa: E402

from botorch_rs.acquisition.max_value_entropy_search import MaxValueBase  # noqa: E402
from botorch_rs.models import SingleTaskGP  # noqa: E402
from botorch_rs.optim import optimize_acqf  # noqa: E402

DT = torch.float64


class CustomMES(MaxValueBase):
    def _compute_information_gain(self, X, mean_M, variance_M, covar_mM):
        self.last_query = X
        return (
            mean_M.square()
            + variance_M
            + covar_mM.squeeze(-1)
            + self.posterior_max_values.mean() * mean_M
        ).movedim(-1, 0)


class ReferenceMES(BtBase):
    def _compute_information_gain(self, X, mean_M, variance_M, covar_mM):
        return (
            mean_M.square()
            + variance_M
            + covar_mM.squeeze(-1)
            + self.posterior_max_values.mean() * mean_M
        ).movedim(-1, 0)


def models():
    x = torch.tensor([[0.0], [0.21], [0.48], [0.76], [1.0]], dtype=DT)
    y = torch.tensor([[0.3], [0.7], [1.1], [0.4], [0.2]], dtype=DT)
    return SingleTaskGP(x, y), BtGP(x, y)


@pytest.mark.parametrize("gumbel", [False, True])
@pytest.mark.parametrize("maximize", [False, True])
def test_custom_mes_values_gradients_and_constructor(gumbel, maximize):
    model, other = models()
    candidates = torch.linspace(0, 1, 7, dtype=DT)[:, None]
    pending = torch.tensor([[0.62]], dtype=DT)
    actual = CustomMES(
        model,
        candidates,
        num_mv_samples=4,
        use_gumbel=gumbel,
        maximize=maximize,
        X_pending=pending,
        seed=82,
    )
    reference = ReferenceMES(
        other, candidates, num_mv_samples=4, use_gumbel=gumbel, maximize=maximize, X_pending=pending
    )
    torch.testing.assert_close(actual.candidate_set, reference.candidate_set)
    assert actual.posterior_max_values.shape == (4, 1)
    assert torch.isfinite(actual.posterior_max_values).all()
    actual.posterior_max_values = reference.posterior_max_values = torch.tensor(
        [[0.8], [1.0], [1.2], [1.4]], dtype=DT
    )
    query = torch.tensor([[[0.27]], [[0.65]]], dtype=DT, requires_grad=True)
    other_query = query.detach().clone().requires_grad_()
    value, expected = actual(query), reference(other_query)
    torch.testing.assert_close(value, expected, rtol=2e-9, atol=2e-10)
    gradient = torch.autograd.grad(value.sum(), query)[0]
    expected_gradient = torch.autograd.grad(expected.sum(), other_query)[0]
    torch.testing.assert_close(gradient, expected_gradient, rtol=2e-8, atol=2e-9)
    native_value, native_gradient = actual.value_and_grad(query.detach())
    torch.testing.assert_close(native_value, value.detach(), rtol=2e-9, atol=2e-10)
    torch.testing.assert_close(native_gradient, gradient, rtol=2e-8, atol=2e-9)
    assert actual.last_query.shape[-2] == 1
    actual.set_X_pending(None)
    assert actual.X_pending is None


def test_custom_mes_runs_through_native_optimizer_hook():
    model, _ = models()
    acquisition = CustomMES(model, torch.linspace(0, 1, 8, dtype=DT)[:, None], seed=7)
    candidates, values = optimize_acqf(
        acquisition,
        torch.tensor([[0.0], [1.0]], dtype=DT),
        q=1,
        num_restarts=3,
        raw_samples=24,
        options={"maxiter": 30, "seed": 9},
    )
    assert candidates.shape == (1, 1)
    assert 0.0 <= candidates.item() <= 1.0
    torch.testing.assert_close(values, acquisition(candidates).squeeze(), rtol=2e-9, atol=2e-10)
    grid = torch.linspace(0, 1, 151, dtype=DT).reshape(-1, 1, 1)
    assert values.item() >= acquisition(grid).max().item() - 2e-4


def test_max_value_base_requires_real_hook():
    model, _ = models()
    with pytest.raises(TypeError, match="_compute_information_gain"):
        MaxValueBase(model, torch.tensor([[0.5]], dtype=DT))

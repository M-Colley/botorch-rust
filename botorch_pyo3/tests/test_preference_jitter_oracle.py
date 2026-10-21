"""Roundoff-only alternate roots used by the preference equivalence oracle."""

from __future__ import annotations

import pytest

from helpers import importorskip_botorch

importorskip_botorch()
torch = pytest.importorskip("torch")

from preference_helpers import BtJitterDecisions  # noqa: E402

pytestmark = pytest.mark.equivalence


def test_unmodified_source_accepts_rank_one_covariance_with_normal_jitter():
    from botorch.models.pairwise_gp import _scaled_psd_safe_cholesky

    matrix = torch.ones(1, 2, 2, dtype=torch.double)
    factor = _scaled_psd_safe_cholesky(
        matrix, scale=torch.tensor(1.0, dtype=torch.double), jitter=1e-8
    )
    assert bool(torch.isfinite(factor).all())
    assert bool((factor.diagonal(dim1=-2, dim2=-1) > 0).all())


@pytest.mark.parametrize("scale", [1e-309, 1e-200, 1.0, 1e200])
@pytest.mark.parametrize("batch_shape", [(), (1,)])
def test_rank_one_alternate_factor_stays_within_roundoff(scale, batch_shape):
    # The exact minimum eigenvalue is zero, so 2 * abs(lambda_min) supplies
    # no positive margin. The independent target covariance is scale * 11^T.
    matrix = torch.full((*batch_shape, 2, 2), scale, dtype=torch.double)
    factor = BtJitterDecisions._factor(matrix)
    reconstructed = factor @ factor.transpose(-1, -2)
    assert bool(torch.isfinite(factor).all())
    assert bool((factor.diagonal(dim1=-2, dim2=-1) > 0).all())
    # The existing Demmel ambiguity budget, expressed relative to scale
    # to avoid overflow/underflow in this scale-invariance regression.
    unit_roundoff = torch.finfo(torch.double).eps / 2
    relative_budget = 20 * 2**1.5 * unit_roundoff * 2
    assert float(((reconstructed - matrix) / scale).abs().max()) <= relative_budget


def test_alternate_factor_preserves_positive_definite_values_and_gradients():
    matrix = torch.tensor([[1.0, 0.2], [0.2, 0.7]], dtype=torch.double, requires_grad=True)
    factor = BtJitterDecisions._factor(matrix)
    expected = torch.linalg.cholesky(matrix)
    torch.testing.assert_close(factor, expected, atol=0, rtol=0)
    actual_gradient = torch.autograd.grad(factor.sum(), matrix)[0]
    expected_gradient = torch.autograd.grad(expected.sum(), matrix)[0]
    torch.testing.assert_close(actual_gradient, expected_gradient, atol=0, rtol=0)


@pytest.mark.parametrize("scale", [1e-200, 1.0, 1e200])
def test_alternate_factor_rejects_genuinely_indefinite_covariance(scale):
    matrix = torch.diag(torch.tensor([scale, -scale], dtype=torch.double))
    with pytest.raises(torch.linalg.LinAlgError):
        BtJitterDecisions._factor(matrix)


def test_exact_zero_has_no_roundoff_dependent_cholesky_outcome():
    decisions = BtJitterDecisions(frozenset([0]))
    matrix = torch.zeros(1, 2, 2, dtype=torch.double)
    assert decisions._decide(matrix) is None
    assert decisions.ambiguous == 0
    with pytest.raises(torch.linalg.LinAlgError):
        decisions._factor(matrix)


def test_unrepresentable_roundoff_budget_cannot_force_an_alternate_outcome():
    minimum = torch.nextafter(torch.tensor(0.0, dtype=torch.double), torch.tensor(1.0))
    matrix = minimum.expand(1, 2, 2).clone()
    decisions = BtJitterDecisions(frozenset([0]))
    assert decisions._decide(matrix) is None
    assert decisions.ambiguous == 0
    with pytest.raises(torch.linalg.LinAlgError):
        decisions._factor(matrix)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_nonfinite_covariance_cannot_force_an_alternate_outcome(value):
    decisions = BtJitterDecisions(frozenset([0]))
    matrix = torch.diag(torch.tensor([1.0, value], dtype=torch.double))
    assert decisions._decide(matrix) is None
    assert decisions.ambiguous == 0
    with pytest.raises(ValueError, match="finite matrix"):
        decisions._factor(matrix)

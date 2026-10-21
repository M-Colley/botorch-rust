"""Native preference member broadcasting and folded learned-sample VJPs."""

# ruff: noqa: E402

from __future__ import annotations

import math

import pytest

from helpers import KINDS, as_kind, assert_close, importorskip_botorch

importorskip_botorch()
torch = pytest.importorskip("torch")

from botorch.acquisition.objective import LearnedObjective as BtLearned
from preference_helpers import Case, build_pair, make_data

from botorch_rs import _botorch_rs as native
from botorch_rs.acquisition.objective import LearnedObjective as RsLearned


def models(batch):
    case = Case(n=6, d=2, m=6, batch=batch)
    x, comps = make_data(case, 192)
    return build_pair(case, x, comps, seed=193)


def reference_samples(objective, x):
    # Official PairwiseGP expands training data only to the query's existing
    # batch shape. Explicitly expand the query to the standard joint broadcast
    # shape where its singleton dimensions need expansion too. Autograd folds
    # that broadcast back into x for the reference adjoint.
    batch = torch.broadcast_shapes(x.shape[:-2], objective.pref_model.batch_shape)
    return objective(x.expand(*batch, *x.shape[-2:]))


CASES = [
    ((2,), (4, 1)),
    ((2,), (4, 2)),
    ((2, 1), (4, 1, 3)),
    ((2, 3), (4, 1, 3)),
    ((3,), (3,)),
]


@pytest.mark.parametrize("batch,input_batch", CASES)
@pytest.mark.parametrize("sample_shape", [(3,), (2, 3)])
@pytest.mark.parametrize("kind", KINDS)
def test_batched_preference_samples_match_official(batch, input_batch, sample_shape, kind):
    bp, rp = models(batch)
    bo = BtLearned(bp, sample_shape=torch.Size(sample_shape), seed=194)
    ro = RsLearned(rp, sample_shape=sample_shape, seed=194)
    x = torch.rand(
        *input_batch, 3, 2, generator=torch.Generator().manual_seed(195), dtype=torch.float64
    )
    expected = reference_samples(bo, x)
    actual = ro(as_kind(x.tolist(), kind))
    output_batch = torch.broadcast_shapes(torch.Size(input_batch), torch.Size(batch))
    output_shape = (math.prod(sample_shape) * output_batch[0], *output_batch[1:], 3)
    assert tuple(torch.as_tensor(actual).shape) == output_shape
    assert_close(actual, expected.reshape(output_shape), rtol=2e-7, atol=2e-8)
    torch.testing.assert_close(
        torch.as_tensor(ro.sampler.base_samples, dtype=torch.float64), bo.sampler.base_samples
    )


@pytest.mark.parametrize("batch,input_batch", CASES)
@pytest.mark.parametrize("sample_shape", [(3,), (2, 3)])
def test_batched_preference_query_adjoint_folds_broadcast_repetitions(
    batch, input_batch, sample_shape
):
    bp, rp = models(batch)
    bo = BtLearned(bp, sample_shape=torch.Size(sample_shape), seed=194)
    ro = RsLearned(rp, sample_shape=sample_shape, seed=194)
    x = torch.rand(
        *input_batch,
        3,
        2,
        generator=torch.Generator().manual_seed(195),
        dtype=torch.float64,
        requires_grad=True,
    )
    actual = ro(x)
    expected = reference_samples(bo, x).reshape(actual.shape)
    weights = torch.rand(
        actual.shape, generator=torch.Generator().manual_seed(196), dtype=torch.float64
    )
    ga = torch.autograd.grad((actual * weights).sum(), x)[0]
    ge = torch.autograd.grad((expected * weights).sum(), x)[0]
    torch.testing.assert_close(actual, expected, rtol=2e-7, atol=2e-8)
    torch.testing.assert_close(ga, ge, rtol=2e-6, atol=2e-7)


def test_batched_preference_vjp_matches_finite_differences():
    _, preference = models((2, 1))
    objective = RsLearned(preference, sample_shape=(2, 3), seed=194)
    x = torch.tensor([[[[[0.2, 0.4], [0.7, 0.5]]]]], dtype=torch.float64, requires_grad=True)
    value = objective(x)
    weights = torch.rand(
        value.shape, generator=torch.Generator().manual_seed(197), dtype=torch.float64
    )
    gradient = torch.autograd.grad((value * weights).sum(), x)[0]
    for index in range(x.numel()):
        xp = x.detach().clone()
        xm = xp.clone()
        xp.reshape(-1)[index] += 1e-6
        xm.reshape(-1)[index] -= 1e-6
        finite = ((objective(xp) - objective(xm)) * weights).sum() / 2e-6
        torch.testing.assert_close(gradient.reshape(-1)[index], finite, rtol=2e-5, atol=2e-6)


def test_native_batch_validation_checks_normals_and_adjoints_before_dispatch():
    _, preference = models((2,))
    handle = preference._acquisition_model(None)
    y = ([0.2, 0.4, 0.7, 0.5], (1, 1, 2, 2))
    with pytest.raises(ValueError, match="base samples"):
        native.learned_objective_samples(handle, y, ([0.0], (1, 1)))
    with pytest.raises(ValueError, match="adjoint"):
        native.learned_objective_samples(handle, y, ([0.0] * 4, (1, 1, 2, 2)), ([0.0], (1,)))

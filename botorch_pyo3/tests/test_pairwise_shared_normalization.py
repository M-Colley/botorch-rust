"""Shared PairwiseGP transform coefficients reduce across training batches."""

# ruff: noqa: E402

from __future__ import annotations

import pytest

pytest.importorskip("numpy")
pytest.importorskip("botorch")


import math

import numpy as np

from helpers import KINDS, as_kind, assert_close, importorskip_botorch

importorskip_botorch()
torch = pytest.importorskip("torch")

from botorch.models import PairwiseGP as BtGP
from botorch.models.pairwise_gp import PairwiseLaplaceMarginalLogLikelihood as BtMLL
from botorch.models.transforms.input import ChainedInputTransform as BtChain
from botorch.models.transforms.input import Normalize as BtNormalize

from botorch_rs.models import PairwiseGP as RsGP
from botorch_rs.models import PairwiseLaplaceMarginalLogLikelihood as RsMLL
from botorch_rs.models.transforms.input import ChainedInputTransform as RsChain
from botorch_rs.models.transforms.input import Normalize as RsNormalize


def pair(batch, kind="torch", chained=False):
    base = torch.tensor(
        [[0.1, 0.2], [0.5, 0.9], [0.3, 0.4], [0.8, 0.1], [0.65, 0.55], [0.2, 0.75]],
        dtype=torch.float64,
    )
    offset = (2 * torch.arange(math.prod(batch), dtype=torch.float64)).reshape(*batch, 1, 1)
    x = base + offset
    comparisons = torch.tensor([[0, 1], [2, 3], [1, 4], [5, 2], [4, 0]]).expand(*batch, 5, 2)
    bn, rn = BtNormalize(2), RsNormalize(2)
    if chained:
        bt = BtChain(first=bn, second=BtNormalize(2, indices=[0], center=0.3))
        rt = RsChain(first=rn, second=RsNormalize(2, indices=[0], center=0.3))
    else:
        bt, rt = bn, rn
    np.random.seed(224)
    bm = BtGP(x, comparisons, input_transform=bt)
    np.random.seed(224)
    rm = RsGP(as_kind(x.tolist(), kind), comparisons, input_transform=rt)
    return bm, rm, bn, rn


@pytest.mark.parametrize("batch", [(2,), (2, 2)])
@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("chained", [False, True])
def test_pooled_bounds_laplace_state_and_mll(batch, kind, chained):
    bm, rm, bn, rn = pair(batch, kind, chained)
    assert_close(rn.bounds, bn.bounds, rtol=1e-12, atol=1e-12)
    assert rn.learn_bounds
    native_shapes = {
        name: tuple(torch.as_tensor(value).shape) for name, value in rm.named_parameters()
    }
    reference_shapes = {name: tuple(value.shape) for name, value in bm.named_parameters()}
    assert native_shapes == reference_shapes
    for name in ("covar", "covar_chol", "utility", "likelihood_hess"):
        assert_close(getattr(rm, name), getattr(bm, name), rtol=2e-7, atol=2e-8)
    bm.train()
    value = BtMLL(bm.likelihood, bm)(
        bm(bm.unconsolidated_datapoints), bm.unconsolidated_comparisons
    )
    params = [p for p in bm.parameters() if p.requires_grad]
    expected = torch.autograd.grad(value, params)
    actual, gradient = RsMLL(rm.likelihood, rm).value_and_grad()
    assert_close(actual, value, rtol=2e-7, atol=2e-8)
    assert_close(
        gradient,
        torch.cat([g.reshape(-1) for g in expected]),
        rtol=2e-6,
        atol=2e-7,
    )


@pytest.mark.parametrize("training", [False, True])
def test_conditioning_preserves_shared_mode_and_refits_only_in_train_mode(training):
    bm, rm, bn, rn = pair((2,))
    bm.train(training)
    rm.train(training)
    x = torch.tensor([[[4.0, 3.0], [5.0, 4.0]]] * 2, dtype=torch.float64)
    comps = torch.tensor([[[0, 1]]] * 2)
    np.random.seed(225)
    bc = bm.condition_on_observations(x, comps)
    np.random.seed(225)
    rc = rm.condition_on_observations(x, comps)
    assert_close(rc.input_transform.bounds, bc.input_transform.bounds, rtol=1e-12, atol=1e-12)
    assert_close(rc.covar, bc.covar, rtol=2e-7, atol=2e-8)
    # The source model and its coefficient-learning flag remain unchanged.
    assert_close(rn.bounds, bn.bounds, rtol=1e-12, atol=1e-12)
    assert rn.learn_bounds and rc.input_transform.learn_bounds


def test_pooled_normalization_query_adjoint_matches_official():
    bm, rm, _, _ = pair((2,))
    query = torch.tensor(
        [[[0.3, 0.7], [0.8, 0.2]], [[2.4, 2.6], [2.7, 2.3]]],
        dtype=torch.float64,
        requires_grad=True,
    )
    expected = bm.posterior(query)
    actual = rm.posterior(query)
    loss = expected.mean.square().sum() + 0.3 * expected.variance.sum()
    native_loss = actual.mean.square().sum() + 0.3 * actual.variance.sum()
    eg = torch.autograd.grad(loss, query)[0]
    ag = torch.autograd.grad(native_loss, query)[0]
    torch.testing.assert_close(actual.mean, expected.mean, rtol=2e-7, atol=2e-8)
    torch.testing.assert_close(actual.variance, expected.variance, rtol=2e-7, atol=2e-8)
    torch.testing.assert_close(ag, eg, rtol=2e-6, atol=2e-7)

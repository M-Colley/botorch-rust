"""Shared and per-member Warp ownership in native batched preference GPs."""

# ruff: noqa: E402

from __future__ import annotations

import pytest

pytest.importorskip("numpy")
pytest.importorskip("botorch")
pytest.importorskip("gpytorch")


import copy
import math

import numpy as np

from helpers import KINDS, as_kind, assert_close, importorskip_botorch

importorskip_botorch()
torch = pytest.importorskip("torch")

from botorch.models import PairwiseGP as BtGP
from botorch.models.likelihoods.pairwise import PairwiseLogitLikelihood as BtLogit
from botorch.models.transforms.input import ChainedInputTransform as BtChain
from botorch.models.transforms.input import Normalize as BtNormalize
from botorch.models.transforms.input import Warp as BtWarp
from gpytorch.kernels import MaternKernel as BtMatern
from gpytorch.kernels import RBFKernel as BtRBF
from gpytorch.kernels import ScaleKernel as BtScale
from gpytorch.priors import LogNormalPrior as BtPrior
from preference_helpers import bt_fit, rs_fit
from test_pairwise_warp import C, X, reference_mll_gradient

from botorch_rs.kernels import MaternKernel as RsMatern
from botorch_rs.kernels import RBFKernel as RsRBF
from botorch_rs.kernels import ScaleKernel as RsScale
from botorch_rs.models import PairwiseGP as RsGP
from botorch_rs.models import PairwiseLaplaceMarginalLogLikelihood as RsMLL
from botorch_rs.models.likelihoods.pairwise import PairwiseLogitLikelihood as RsLogit
from botorch_rs.models.transforms.input import ChainedInputTransform as RsChain
from botorch_rs.models.transforms.input import Normalize as RsNormalize
from botorch_rs.models.transforms.input import Warp as RsWarp
from botorch_rs.priors import LogNormalPrior as RsPrior


def models(batch=(2,), kind="torch", per_member=False, chained=False, tree=False, logit=False):
    nb = math.prod(batch)
    offset = (2 * torch.arange(nb, dtype=torch.float64)).reshape(*batch, 1, 1)
    x = torch.tensor(X, dtype=torch.float64) + offset
    comparisons = torch.tensor(C).expand(*batch, len(C), 2)
    warp_batch = batch if per_member else ()
    bw = BtWarp(
        2,
        [1, 0],
        batch_shape=torch.Size(warp_batch),
        eps=2e-5,
        concentration0_prior=BtPrior(0.0, 0.3),
        concentration1_prior=BtPrior(0.0, 0.3),
    ).to(x)
    rw = RsWarp(
        2,
        [1, 0],
        batch_shape=warp_batch,
        eps=2e-5,
        concentration0_prior=RsPrior(0.0, 0.3),
        concentration1_prior=RsPrior(0.0, 0.3),
    )
    c0 = torch.tensor([0.8, 1.2], dtype=torch.float64)
    c1 = torch.tensor([1.4, 0.9], dtype=torch.float64)
    if per_member:
        delta = (0.1 * torch.arange(nb, dtype=torch.float64)).reshape(*batch, 1, 1)
        c0, c1 = c0 + delta, c1 - delta
    bw._set_concentration(0, c0)
    bw._set_concentration(1, c1)
    rw.initialize(concentration0=c0, concentration1=c1)
    bt, rt = (
        (BtChain(norm=BtNormalize(2), warp=bw), RsChain(norm=RsNormalize(2), warp=rw))
        if chained
        else (bw, rw)
    )
    bk, rk = {}, {}
    if tree:
        bk["covar_module"] = BtScale(BtRBF(ard_num_dims=2) + BtMatern(nu=2.5, ard_num_dims=2)).to(x)
        rk["covar_module"] = RsScale(RsRBF(ard_num_dims=2) + RsMatern(nu=2.5, ard_num_dims=2))
    if logit:
        bk["likelihood"], rk["likelihood"] = BtLogit(), RsLogit()
    np.random.seed(251)
    bm = BtGP(x, comparisons, input_transform=bt, **bk)
    np.random.seed(251)
    rm = RsGP(as_kind(x.tolist(), kind), comparisons, input_transform=rt, **rk)
    return bm, rm, bw, rw


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("per_member", [False, True])
@pytest.mark.parametrize("chained", [False, True])
@pytest.mark.parametrize("tree", [False, True])
@pytest.mark.parametrize("logit", [False, True])
def test_native_batched_warp_state_mll_and_global_gradients(kind, per_member, chained, tree, logit):
    bm, rm, bw, rw = models(
        kind=kind, per_member=per_member, chained=chained, tree=tree, logit=logit
    )
    expected_parameters = dict(bm.named_parameters())
    actual_parameters = dict(rm.named_parameters())
    assert list(actual_parameters) == list(expected_parameters)
    for name, value in expected_parameters.items():
        assert_close(actual_parameters[name], value)
        assert tuple(torch.as_tensor(actual_parameters[name]).shape) == tuple(value.shape)
    for name in ("covar", "covar_chol", "utility", "likelihood_hess"):
        assert_close(getattr(rm, name), getattr(bm, name), rtol=4e-7, atol=4e-8)
    value, gradients = reference_mll_gradient(bm)
    actual, gradient = RsMLL(rm.likelihood, rm).value_and_grad()
    assert_close(actual, value, rtol=4e-7, atol=4e-8)
    assert_close(
        gradient,
        torch.cat([gradients[n].reshape(-1) for n, _ in rm._param_names()]),
        rtol=4e-6,
        atol=4e-7,
    )
    assert_close(rw.concentration0, bw.concentration0)
    assert_close(rw.concentration1, bw.concentration1)
    assert_close(
        rm.transform_inputs(bm.datapoints), bm.transform_inputs(bm.datapoints), rtol=2e-8, atol=2e-9
    )


@pytest.mark.parametrize("batch", [(2,), (2, 2)])
@pytest.mark.parametrize("per_member", [False, True])
def test_batched_warp_query_adjoints_and_changed_trial_concentrations(batch, per_member):
    bm, rm, bw, rw = models(batch=batch, per_member=per_member, chained=True)
    c0, c1 = bw.concentration0.detach() + 0.15, bw.concentration1.detach() - 0.1
    bw._set_concentration(0, c0)
    bw._set_concentration(1, c1)
    raw = list(rm._rs.raw_parameters())
    count = c0.numel()
    raw[-2 * count :] = c0.reshape(-1).tolist() + c1.reshape(-1).tolist()
    expected_value, gradients = reference_mll_gradient(bm)
    value, gradient = RsMLL(rm.likelihood, rm).value_and_grad(raw)
    assert_close(value, expected_value, rtol=4e-7, atol=4e-8)
    assert_close(
        gradient,
        torch.cat([gradients[n].reshape(-1) for n, _ in rm._param_names()]),
        rtol=4e-6,
        atol=4e-7,
    )
    assert_close(rw.concentration0, c0)
    assert_close(rw.concentration1, c1)
    offset = (2 * torch.arange(math.prod(batch), dtype=torch.float64)).reshape(*batch, 1, 1)
    query = (
        torch.tensor([[0.22, 0.31], [0.61, 0.44]], dtype=torch.float64) + offset
    ).requires_grad_()
    bp, rp = bm.posterior(query), rm.posterior(query)
    eg = torch.autograd.grad(bp.mean.square().sum() + 0.3 * bp.variance.sum(), query)[0]
    ag = torch.autograd.grad(rp.mean.square().sum() + 0.3 * rp.variance.sum(), query)[0]
    torch.testing.assert_close(rp.mean, bp.mean, rtol=4e-7, atol=4e-8)
    torch.testing.assert_close(rp.variance, bp.variance, rtol=4e-7, atol=4e-8)
    torch.testing.assert_close(ag, eg, rtol=4e-6, atol=4e-7)


@pytest.mark.parametrize("training", [False, True])
@pytest.mark.parametrize("per_member", [False, True])
def test_conditioned_batched_warp_preserves_parameter_ownership(training, per_member):
    bm, rm, _, _ = models(per_member=per_member, chained=True)
    # Distinct kernel values make collapsing the parent raw map observable.
    bm.covar_module.outputscale = torch.tensor([0.9, 1.5], dtype=torch.float64)
    rm.covar_module.outputscale = torch.tensor([0.9, 1.5], dtype=torch.float64)
    bm.covar_module.base_kernel.lengthscale = torch.tensor(
        [[[0.8, 1.1]], [[1.3, 0.7]]], dtype=torch.float64
    )
    rm.covar_module.base_kernel.lengthscale = torch.tensor(
        [[[0.8, 1.1]], [[1.3, 0.7]]], dtype=torch.float64
    )
    bm.train(training)
    rm.train(training)
    points = torch.tensor([[[1.1, 0.4], [0.4, 1.1]], [[3.3, 2.5], [2.4, 3.2]]], dtype=torch.float64)
    comparisons = torch.tensor([[[0, 1]], [[1, 0]]])
    np.random.seed(252)
    bc = bm.condition_on_observations(points, comparisons)
    np.random.seed(252)
    rc = rm.condition_on_observations(points, comparisons)
    assert_close(rc.covar, bc.covar, rtol=4e-7, atol=4e-8)
    assert_close(rc.utility, bc.utility, rtol=4e-7, atol=4e-8)
    assert rc._rs.raw_parameters() == rm._rs.raw_parameters()
    for name, value in rc.named_parameters():
        assert_close(value, dict(bc.named_parameters())[name])
    value, gradients = reference_mll_gradient(bc)
    actual, gradient = RsMLL(rc.likelihood, rc).value_and_grad()
    assert_close(actual, value, rtol=4e-7, atol=4e-8)
    assert_close(
        gradient,
        torch.cat([gradients[n].reshape(-1) for n, _ in rc._param_names()]),
        rtol=4e-6,
        atol=4e-7,
    )
    cloned = copy.deepcopy(rc)
    cloned.load_state_dict(rc.state_dict())
    assert_close(cloned.covar, rc.covar, rtol=4e-7, atol=4e-8)


@pytest.mark.parametrize("per_member", [False, True])
def test_batched_native_warp_fit_matches_official(per_member):
    bm, rm, _, rw = models(per_member=per_member, chained=True)
    initial = torch.as_tensor(rw.concentration0).clone()
    expected, _ = bt_fit(bm)
    actual, _ = rs_fit(rm)
    assert_close(actual, expected, rtol=3e-5, atol=3e-6)
    assert (torch.as_tensor(rw.concentration0) - initial).abs().max() > 1e-3
    query = torch.tensor([[[0.3, 0.6], [0.7, 0.2]], [[2.3, 2.6], [2.7, 2.2]]], dtype=torch.float64)
    assert_close(rm.posterior(query).mean, bm.posterior(query).mean, rtol=8e-4, atol=8e-5)


@pytest.mark.parametrize("per_member", [False, True])
def test_fantasy_axis_reuses_original_batched_parameter_ownership(per_member):
    bm, rm, _, _ = models(per_member=per_member, chained=True)
    points = torch.tensor(
        [[[0.23, 0.4], [0.7, 0.3]], [[2.23, 2.4], [2.7, 2.3]]], dtype=torch.float64
    )
    comparisons = torch.tensor([[[[0, 1]], [[1, 0]]], [[[1, 0]], [[0, 1]]], [[[0, 1]], [[0, 1]]]])
    bm.eval()
    rm.eval()
    np.random.seed(253)
    bc = bm.condition_on_observations(points, comparisons)
    np.random.seed(253)
    rc = rm.condition_on_observations(points, comparisons)
    assert rc.batch_shape == (3, 2)
    assert_close(rc.covar, bc.covar, rtol=4e-7, atol=4e-8)
    assert_close(rc.utility, bc.utility, rtol=4e-7, atol=4e-8)
    assert rc._rs.raw_parameters() == rm._rs.raw_parameters()
    assert list(rc._param_names()) == list(rm._param_names())
    value, gradients = reference_mll_gradient(bc)
    actual, gradient = RsMLL(rc.likelihood, rc).value_and_grad()
    assert_close(actual, value, rtol=4e-7, atol=4e-8)
    assert_close(
        gradient,
        torch.cat([gradients[n].reshape(-1) for n, _ in rc._param_names()]),
        rtol=4e-6,
        atol=4e-7,
    )


@pytest.mark.parametrize("per_member", [False, True])
def test_replacing_batched_training_data_preserves_kernel_and_warp_parameters(per_member):
    bm, rm, _, _ = models(per_member=per_member, chained=True)
    bm.covar_module.outputscale = torch.tensor([0.9, 1.5], dtype=torch.float64)
    rm.covar_module.outputscale = torch.tensor([0.9, 1.5], dtype=torch.float64)
    replacement = bm.datapoints + torch.tensor([0.1, -0.2], dtype=torch.float64)
    np.random.seed(254)
    bm.set_train_data(replacement, bm.comparisons)
    np.random.seed(254)
    rm.set_train_data(replacement, rm.comparisons)
    for name, value in rm.named_parameters():
        assert_close(value, dict(bm.named_parameters())[name])
    assert_close(rm.covar, bm.covar, rtol=4e-7, atol=4e-8)
    value, gradients = reference_mll_gradient(bm)
    actual, gradient = RsMLL(rm.likelihood, rm).value_and_grad()
    assert_close(actual, value, rtol=4e-7, atol=4e-8)
    assert_close(
        gradient,
        torch.cat([gradients[n].reshape(-1) for n, _ in rm._param_names()]),
        rtol=4e-6,
        atol=4e-7,
    )


def test_chain_mixes_shared_and_member_owned_concentrations():
    bounds = torch.tensor([[0.0, 0.0], [3.0, 3.0]], dtype=torch.float64)
    bt = BtChain(
        first=BtWarp(2, [1], concentration0_prior=BtPrior(0.0, 0.3)).to(bounds),
        second=BtWarp(
            2,
            [0],
            batch_shape=torch.Size([2]),
            bounds=bounds,
            concentration1_prior=BtPrior(0.0, 0.3),
        ).to(bounds),
    )
    rt = RsChain(
        first=RsWarp(2, [1], concentration0_prior=RsPrior(0.0, 0.3)),
        second=RsWarp(
            2, [0], batch_shape=(2,), bounds=bounds, concentration1_prior=RsPrior(0.0, 0.3)
        ),
    )
    bt.first._set_concentration(0, 0.8)
    rt.first.initialize(concentration0=0.8)
    c1 = torch.tensor([[[1.2]], [[0.7]]], dtype=torch.float64)
    bt.second._set_concentration(1, c1)
    rt.second.initialize(concentration1=c1)
    x = torch.stack(
        [torch.tensor(X, dtype=torch.float64), torch.tensor(X, dtype=torch.float64) + 2]
    )
    comparisons = torch.tensor([C, C])
    np.random.seed(255)
    bm = BtGP(x, comparisons, input_transform=bt)
    np.random.seed(255)
    rm = RsGP(x, comparisons, input_transform=rt)
    value, gradients = reference_mll_gradient(bm)
    actual, gradient = RsMLL(rm.likelihood, rm).value_and_grad()
    assert_close(actual, value, rtol=4e-7, atol=4e-8)
    assert_close(
        gradient,
        torch.cat([gradients[n].reshape(-1) for n, _ in rm._param_names()]),
        rtol=4e-6,
        atol=4e-7,
    )

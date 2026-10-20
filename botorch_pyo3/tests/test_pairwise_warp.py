"""Native learned PairwiseGP input warping against official BoTorch."""

# ruff: noqa: E402

from __future__ import annotations

import copy

import numpy as np
import pytest

from helpers import KINDS, as_kind, assert_close, importorskip_botorch

importorskip_botorch()
torch = pytest.importorskip("torch")

from botorch.models import PairwiseGP as BtGP
from botorch.models.likelihoods.pairwise import PairwiseLogitLikelihood as BtLogit
from botorch.models.pairwise_gp import PairwiseLaplaceMarginalLogLikelihood as BtMLL
from botorch.models.transforms.input import ChainedInputTransform as BtChain
from botorch.models.transforms.input import Normalize as BtNormalize
from botorch.models.transforms.input import Warp as BtWarp
from botorch.sampling.pairwise_samplers import PairwiseIIDNormalSampler as BtPairSampler
from gpytorch.kernels import MaternKernel as BtMatern
from gpytorch.kernels import RBFKernel as BtRBF
from gpytorch.kernels import ScaleKernel as BtScale
from gpytorch.priors import LogNormalPrior as BtPrior
from preference_helpers import bt_fit, fixed_pairwise_sampler, rs_fit

from botorch_rs.exceptions.errors import UnsupportedError
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
from botorch_rs.sampling.pairwise_samplers import PairwiseIIDNormalSampler as RsPairSampler

X = [[0.1, 0.2], [0.5, 0.9], [0.3, 0.4], [0.8, 0.1], [0.65, 0.55], [0.2, 0.75]]
C = [[0, 1], [2, 3], [1, 4], [5, 2], [4, 0]]


def models(kind="torch", chained=False, tree=False, logit=False, priors=True):
    x = torch.tensor(X, dtype=torch.float64)
    comps = torch.tensor(C)
    indices = [1] if chained else [1, 0]
    bp, rp = (BtPrior(0.0, 0.3), RsPrior(0.0, 0.3)) if priors else (None, None)
    bw = BtWarp(2, indices, eps=2e-5, concentration0_prior=bp, concentration1_prior=bp).to(x)
    rw = RsWarp(2, indices, eps=2e-5, concentration0_prior=rp, concentration1_prior=rp)
    k = len(indices)
    c0 = torch.tensor([0.8, 1.2][:k], dtype=torch.float64)
    c1 = torch.tensor([1.4, 0.9][:k], dtype=torch.float64)
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
    np.random.seed(231)
    bm = BtGP(x, comps, input_transform=bt, **bk)
    np.random.seed(231)
    rm = RsGP(as_kind(X, kind), comps, input_transform=rt, **rk)
    return bm, rm, bw, rw


def reference_mll_gradient(model):
    model.train()
    value = BtMLL(model.likelihood, model)(
        model(model.unconsolidated_datapoints), model.unconsolidated_comparisons
    )
    parameters = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    gradients = torch.autograd.grad(value, [p for _, p in parameters])
    return value, dict(zip([n for n, _ in parameters], gradients, strict=True))


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("chained", [False, True])
@pytest.mark.parametrize("tree", [False, True])
@pytest.mark.parametrize("logit", [False, True])
def test_warp_parameters_laplace_state_and_native_mll_gradient(kind, chained, tree, logit):
    bm, rm, _, rw = models(kind, chained, tree, logit)
    source = dict(bm.named_parameters())
    actual = dict(rm.named_parameters())
    assert list(actual) == list(source)
    assert {n: tuple(torch.as_tensor(v).shape) for n, v in actual.items()} == {
        n: tuple(v.shape) for n, v in source.items()
    }
    for name in ("covar", "covar_chol", "utility", "likelihood_hess"):
        assert_close(getattr(rm, name), getattr(bm, name), rtol=3e-7, atol=3e-8)
    value, gradients = reference_mll_gradient(bm)
    actual_value, actual_grad = RsMLL(rm.likelihood, rm).value_and_grad()
    expected_grad = torch.cat([gradients[n].reshape(-1) for n, _ in rm._param_names()])
    assert_close(actual_value, value, rtol=3e-7, atol=3e-8)
    assert_close(actual_grad, expected_grad, rtol=3e-6, atol=3e-7)
    assert rw._model is rm


@pytest.mark.parametrize("chained", [False, True])
@pytest.mark.parametrize("tree", [False, True])
@pytest.mark.parametrize("priors", [False, True])
def test_warped_posterior_and_query_adjoint(chained, tree, priors):
    bm, rm, _, _ = models(chained=chained, tree=tree, priors=priors)
    query = torch.tensor([[0.22, 0.31], [0.61, 0.44]], dtype=torch.float64, requires_grad=True)
    bp, rp = bm.posterior(query), rm.posterior(query)
    expected = bp.mean.square().sum() + 0.3 * bp.distribution.covariance_matrix.square().sum()
    actual = rp.mean.square().sum() + 0.3 * rp.distribution.covariance_matrix.square().sum()
    eg = torch.autograd.grad(expected, query)[0]
    ag = torch.autograd.grad(actual, query)[0]
    torch.testing.assert_close(rp.mean, bp.mean, rtol=3e-7, atol=3e-8)
    torch.testing.assert_close(rp.variance, bp.variance, rtol=3e-7, atol=3e-8)
    torch.testing.assert_close(ag, eg, rtol=3e-6, atol=3e-7)
    assert_close(rm.transform_inputs(query), bm.transform_inputs(query), rtol=2e-8, atol=2e-9)


def test_trial_concentrations_change_training_covariance_and_native_gradient():
    bm, rm, bw, rw = models()
    previous = torch.as_tensor(rm.covar).clone()
    c0 = torch.tensor([1.1, 0.7], dtype=torch.float64)
    c1 = torch.tensor([0.9, 1.3], dtype=torch.float64)
    bw._set_concentration(0, c0)
    bw._set_concentration(1, c1)
    raw = list(rm._rs.raw_parameters())
    raw[-4:] = c0.tolist() + c1.tolist()
    actual, gradient = RsMLL(rm.likelihood, rm).value_and_grad(raw)
    expected, source = reference_mll_gradient(bm)
    assert (torch.as_tensor(rm.covar) - previous).abs().max() > 0.01
    assert_close(actual, expected, rtol=3e-7, atol=3e-8)
    assert_close(
        gradient,
        torch.cat([source[n].reshape(-1) for n, _ in rm._param_names()]),
        rtol=3e-6,
        atol=3e-7,
    )
    assert_close(rw.concentration0, c0)
    assert_close(rw.concentration1, c1)


@pytest.mark.parametrize("training", [False, True])
def test_warp_conditioning_and_state_roundtrip(training):
    bm, rm, _, _ = models(chained=True)
    clone = copy.deepcopy(rm)
    clone.load_state_dict(rm.state_dict())
    assert_close(clone.covar, rm.covar, rtol=3e-7, atol=3e-8)
    bm.train(training)
    rm.train(training)
    points = torch.tensor([[1.2, 0.3], [0.4, 1.1]], dtype=torch.float64)
    comps = torch.tensor([[0, 1]])
    np.random.seed(232)
    bc = bm.condition_on_observations(points, comps)
    np.random.seed(232)
    rc = rm.condition_on_observations(points, comps)
    assert_close(rc.covar, bc.covar, rtol=3e-7, atol=3e-8)
    assert_close(rc.utility, bc.utility, rtol=3e-7, atol=3e-8)
    query = torch.tensor([[0.35, 0.6], [0.7, 0.2]], dtype=torch.float64)
    assert_close(rc.posterior(query).mean, bc.posterior(query).mean, rtol=3e-7, atol=3e-8)


def test_warp_fantasies_use_native_conditioned_members():
    bm, rm, _, _ = models()
    points = torch.tensor([[0.23, 0.4], [0.7, 0.3], [0.6, 0.8]], dtype=torch.float64)
    bs = BtPairSampler(torch.Size([2]), seed=233)
    rs = RsPairSampler(2, seed=233)
    # Existing audited official pairwise-sampler orientation defect is
    # corrected only in this comparison context; production stays native.
    with fixed_pairwise_sampler():
        bc = bm.fantasize(points, bs)
        rc = rm.fantasize(points, rs)
    assert rc.batch_shape == tuple(bc.batch_shape)
    assert_close(rc.covar, bc.covar, rtol=3e-7, atol=3e-8)
    query = torch.tensor([[0.35, 0.6], [0.7, 0.2]], dtype=torch.float64)
    assert_close(rc.posterior(query).mean, bc.posterior(query).mean, rtol=3e-7, atol=3e-8)


@pytest.mark.parametrize("chained", [False, True])
def test_native_warp_fitting_matches_official(chained):
    bm, rm, _, rw = models(chained=chained)
    initial = torch.as_tensor(rw.concentration0).clone()
    expected, _ = bt_fit(bm)
    actual, _ = rs_fit(rm)
    assert_close(actual, expected, rtol=2e-6, atol=2e-7)
    assert (torch.as_tensor(rw.concentration0) - initial).abs().max() > 1e-3
    query = torch.tensor([[0.35, 0.6], [0.7, 0.2]], dtype=torch.float64)
    assert_close(rm.posterior(query).mean, bm.posterior(query).mean, rtol=3e-4, atol=3e-5)


def test_warp_boundaries_are_explicit():
    with pytest.raises(UnsupportedError, match="per-member coefficient chains"):
        RsGP([X, X], [C, C], input_transform=RsNormalize(2, batch_shape=(2,)))
    with pytest.raises(UnsupportedError, match="learns its coefficients after"):
        RsGP(X, C, input_transform=RsChain(warp=RsWarp(2, [0]), norm=RsNormalize(2)))
    with pytest.raises(UnsupportedError, match="reverse=True"):
        RsGP(X, C, input_transform=RsWarp(2, [0], reverse=True))

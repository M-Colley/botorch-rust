"""Higher-order GP API checks and live BoTorch numerical parity.

The dense Rust references are in models::higher_order_gp::tests. These
checks cover the public Python wiring, Kronecker likelihood gradients,
structured posteriors, identical-base-sample Matheron draws and fitting.
"""

from __future__ import annotations

import copy
import math

import pytest

from botorch_rs.exceptions.errors import BotorchTensorDimensionError, UnsupportedError
from botorch_rs.fit import fit_gpytorch_mll
from botorch_rs.mlls import ExactMarginalLogLikelihood
from botorch_rs.models.higher_order_gp import FlattenedStandardize, HigherOrderGP
from botorch_rs.models.transforms.input import Normalize, Warp
from botorch_rs.optim.fit import fit_gpytorch_mll_scipy, fit_gpytorch_mll_torch
from botorch_rs.optim.utils import get_parameters_and_bounds
from botorch_rs.sampling.normal import IIDNormalSampler, SobolQMCNormalSampler
from helpers import KINDS, as_kind, assert_close, check_kind, importorskip_botorch, shape, to_list


def data() -> tuple[list, list]:
    X = [[0.1, 0.2], [0.3, 0.8], [0.55, 0.4], [0.8, 0.1], [0.9, 0.7]]
    Y = [[[math.sin(3 * x[0] + j) + x[1] * (i + 1) for j in range(3)] for i in range(2)] for x in X]
    return X, Y


@pytest.mark.parametrize("kind", KINDS)
def test_flattened_standardize_round_trip(kind: str) -> None:
    X, Y = data()
    tf = FlattenedStandardize((2, 3))
    Y = as_kind(Y, kind)
    Yvar = as_kind([[[0.2] * 3 for _ in range(2)] for _ in X], kind)
    standardized, transformed_var = tf(Y, Yvar)
    restored, restored_var = tf.untransform(standardized, transformed_var)
    assert_close(restored, Y)
    assert_close(restored_var, Yvar)
    check_kind(restored, kind)
    assert shape(standardized) == (5, 2, 3)


@pytest.mark.parametrize("kind", KINDS)
def test_api_fit_condition_and_copy(kind: str) -> None:
    X, Y = data()
    model = HigherOrderGP(as_kind(X, kind), as_kind(Y, kind), learn_latent_pars=False)
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    before = float(to_list(mll()))
    latents = [to_list(p) for p in model.latent_parameters]
    params, _ = get_parameters_and_bounds(mll)
    assert "likelihood.noise_covar.raw_noise" in params
    assert not any("latent_parameters" in n for n in params)
    fit_gpytorch_mll(mll, optimizer_kwargs={"options": {"maxiter": 10}})
    assert float(to_list(mll())) > before
    assert [to_list(p) for p in model.latent_parameters] == latents
    raw_before_adam = model._rs.raw_parameters()
    fit_gpytorch_mll_torch(mll, step_limit=3, stopping_criterion=None)
    assert math.isfinite(float(to_list(mll())))
    assert model._rs.raw_parameters() != raw_before_adam
    assert [to_list(p) for p in model.latent_parameters] == latents
    result = fit_gpytorch_mll_scipy(mll, options={"maxiter": 2})
    assert math.isfinite(result.fval)
    query = as_kind([[0.4, 0.6], [0.7, 0.2]], kind)
    post = model.posterior(query)
    assert shape(post.mean) == (2, 2, 3)
    check_kind(post.mean, kind)
    assert shape(post.rsample((3,))) == (3, 2, 2, 3)
    noisy = model.posterior(query, observation_noise=True)
    mu, sd = model.outcome_transform._scalar_stats()
    del mu
    expected_noise = to_list(model.likelihood.noise)[0] * sd * sd
    for v, vn in zip(to_list(post.variance)[0][0], to_list(noisy.variance)[0][0], strict=True):
        assert vn - v == pytest.approx(expected_noise)
    clone = copy.deepcopy(model)
    clone.likelihood.noise = 0.2
    assert to_list(model.likelihood.noise)[0] != pytest.approx(0.2)
    conditioned = model.condition_on_observations(query, post.mean)
    assert shape(conditioned.train_inputs[0]) == (7, 2)
    assert shape(conditioned.posterior(query).mean) == (2, 2, 3)
    for sampler_cls in (SobolQMCNormalSampler, IIDNormalSampler):
        sampler = sampler_cls((2,), seed=12)
        assert shape(sampler(post)) == (2, 2, 2, 3)
        fantasy = model.fantasize(query, sampler)
        assert fantasy.batch_shape == (2,)
        fp = fantasy.posterior(query)
        assert shape(fp.mean) == (2, 2, 2, 3)
        assert shape(sampler_cls((3,), seed=4)(fp)) == (3, 2, 2, 2, 3)


def test_reject_invalid_shapes_and_unsupported_options() -> None:
    X, Y = data()
    with pytest.raises(BotorchTensorDimensionError):
        HigherOrderGP([0.1, 0.2], Y)
    with pytest.raises(ValueError, match="positive"):
        HigherOrderGP(X, Y, num_latent_dims=[0, 1])
    with pytest.raises(UnsupportedError, match="learned input transforms"):
        HigherOrderGP(X, Y, input_transform=Warp(d=2, indices=[0, 1]))
    model = HigherOrderGP(X, Y)
    with pytest.raises(UnsupportedError, match="output_indices"):
        model.posterior([[0.4, 0.2]], output_indices=[0])
    with pytest.raises(UnsupportedError, match="Unsupported arguments"):
        model.posterior([[0.4, 0.2]], unknown=True)
    with pytest.raises(UnsupportedError, match="unbatched"):
        HigherOrderGP([X, X], [Y, Y])


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("standardize", [False, True])
def test_posterior_samples_snapshot_parameters(kind: str, standardize: bool) -> None:
    X, Y = data()
    options = {} if standardize else {"outcome_transform": None}
    model = HigherOrderGP(as_kind(X, kind), as_kind(Y, kind), learn_latent_pars=False, **options)
    query = as_kind([[0.4, 0.6], [0.7, 0.2]], kind)
    posterior = model.posterior(query)
    lazy_posterior = model.posterior(query)
    if standardize:
        from botorch_rs.posteriors.transformed import TransformedPosterior

        def nested(post):
            return TransformedPosterior(
                post,
                sample_transform=lambda samples: samples,
                mean_transform=lambda mean, variance: mean,
                variance_transform=lambda mean, variance: variance,
            )

        posterior, lazy_posterior = nested(posterior), nested(lazy_posterior)
    original = to_list(posterior.mean)
    width = posterior.base_sample_shape[-1]
    base = as_kind([[0.0] * width], kind)
    assert_close(to_list(posterior.rsample_from_base_samples((1,), base))[0], original)
    model.likelihood.noise = 100.0
    assert_close(posterior.mean, original)
    assert_close(lazy_posterior.mean, original)
    assert_close(to_list(posterior.rsample_from_base_samples((1,), base))[0], original)
    assert to_list(model.posterior(query).mean) != original


@pytest.mark.parametrize("seed", range(8))
@pytest.mark.parametrize("latent_init", ["default", "gp"])
def test_live_botorch_parity(seed: int, latent_init: str) -> None:
    importorskip_botorch()
    import torch
    from botorch.models.higher_order_gp import HigherOrderGP as BtHigherOrderGP
    from gpytorch.mlls import ExactMarginalLogLikelihood as BtMLL
    from gpytorch.settings import max_cholesky_size
    from linear_operator.settings import _fast_solves

    torch.set_num_threads(1)
    generator = torch.Generator().manual_seed(400 + seed)
    n = 4 + seed % 3
    d = 1 + seed % 2
    target_shape = (2, 2 + seed % 2)
    X = torch.rand(n, d, dtype=torch.float64, generator=generator)
    Y = torch.randn(n, *target_shape, dtype=torch.float64, generator=generator)
    Xq = torch.rand(2, 3, d, dtype=torch.float64, generator=generator)
    kwargs = {"latent_init": latent_init}
    if seed % 2:
        kwargs["outcome_transform"] = None
    torch.manual_seed(seed)
    bt = BtHigherOrderGP(X, Y, **kwargs)
    torch.manual_seed(seed)
    rs = HigherOrderGP(X, Y, **kwargs)
    mb, mr = BtMLL(bt.likelihood, bt), ExactMarginalLogLikelihood(rs.likelihood, rs)
    raw = []
    with torch.no_grad():
        for name, p in mb.named_parameters():
            noise = 0.07 * torch.randn(p.shape, generator=generator, dtype=torch.float64)
            p.add_(noise)
            if "raw_noise" in name or "raw_lengthscale" in name:
                p.abs_().add_(0.01)
            raw.extend(p.reshape(-1).tolist())
    rs._rs.set_raw_parameters(raw)
    with _fast_solves(False), max_cholesky_size(10000):
        mb.zero_grad()
        vb = mb(bt(X), bt.train_targets)
        vb.backward()
        gb = torch.cat([p.grad.reshape(-1) for p in mb.parameters()])
        vr, gr = mr.value_and_grad()
        torch.testing.assert_close(
            torch.tensor(vr, dtype=torch.float64), vb.detach(), rtol=1e-8, atol=1e-9
        )
        torch.testing.assert_close(torch.tensor(gr, dtype=torch.float64), gb, rtol=1e-7, atol=1e-8)
        pb, pr = bt.posterior(Xq), rs.posterior(Xq)
        torch.testing.assert_close(pr.mean, pb.mean.detach(), rtol=1e-8, atol=1e-9)
        torch.testing.assert_close(pr.variance, pb.variance.detach(), rtol=1e-7, atol=1e-8)
        # Official BoTorch 0.18.1 omits test noise from HOGP variances
        # (fixed in local fork commit 0629df4bc). Check the Gaussian formula
        # directly instead of reproducing that upstream bug.
        noisy = rs.posterior(Xq, observation_noise=True)
        scale = rs.outcome_transform._scalar_stats()[1] if hasattr(rs, "outcome_transform") else 1.0
        torch.testing.assert_close(
            noisy.variance - pr.variance,
            torch.ones_like(pr.variance) * rs.likelihood.noise[0] * scale**2,
        )
        # Small factors use deterministic Cholesky roots in both libraries.
        pb, pr = bt.posterior(Xq), rs.posterior(Xq)
        assert tuple(pb.base_sample_shape) == tuple(pr.base_sample_shape)
        base = torch.randn(5, *pb.base_sample_shape, dtype=torch.float64, generator=generator)
        sb = pb.rsample_from_base_samples(torch.Size([5]), base)
        sr = pr.rsample_from_base_samples((5,), base)
        torch.testing.assert_close(sr, sb.detach(), rtol=1e-6, atol=1e-7)


def test_multiple_latent_coordinates_and_input_transform() -> None:
    importorskip_botorch()
    import torch
    from botorch.models.higher_order_gp import HigherOrderGP as BtHigherOrderGP
    from botorch.models.transforms.input import Normalize as BtNormalize
    from botorch.models.utils.gpytorch_modules import get_covar_module_with_dim_scaled_prior
    from gpytorch.mlls import ExactMarginalLogLikelihood as BtMLL

    X, Y = data()
    X = torch.tensor(X, dtype=torch.float64) * 3 + 2
    Y = torch.tensor(Y, dtype=torch.float64)
    # Official 0.18.1 creates ARD=1 latent kernels even when there are
    # several coordinates, which raises during training. Supply the
    # dimension-correct kernels explicitly as the upstream workaround.
    kernels = [get_covar_module_with_dim_scaled_prior(d).to(X) for d in (2, 2, 1)]
    torch.manual_seed(9)
    bt = BtHigherOrderGP(
        X, Y, covar_modules=kernels, num_latent_dims=[2, 1], input_transform=BtNormalize(2)
    )
    torch.manual_seed(9)
    rs = HigherOrderGP(X, Y, num_latent_dims=[2, 1], input_transform=Normalize(2))
    mb, mr = BtMLL(bt.likelihood, bt), ExactMarginalLogLikelihood(rs.likelihood, rs)
    torch.testing.assert_close(
        torch.tensor(float(mr()), dtype=torch.float64),
        mb(bt(X), bt.train_targets).detach(),
        rtol=1e-8,
        atol=1e-9,
    )
    pb, pr = bt.posterior(X[:2]), rs.posterior(X[:2])
    torch.testing.assert_close(pr.mean, pb.mean.detach(), rtol=1e-8, atol=1e-9)
    torch.testing.assert_close(pr.variance, pb.variance.detach(), rtol=1e-7, atol=1e-8)

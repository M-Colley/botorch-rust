"""Sparse observation labels extend atomically with the actual Gaussian law."""

from unittest.mock import patch

import pytest
import torch

from botorch_rs.exceptions.errors import UnsupportedError
from botorch_rs.fit import fit_gpytorch_mll
from botorch_rs.mlls import ExactMarginalLogLikelihood
from botorch_rs.models._torch_backend_relevance import TorchSparseOutlierNoise
from botorch_rs.models.robust_relevance_pursuit_model import RobustRelevancePursuitSingleTaskGP
from botorch_rs.sampling.normal import IIDNormalSampler

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


def data(device):
    X = torch.tensor(
        [[0.05], [0.3], [0.6], [0.95]], dtype=torch.double, device=device
    ).requires_grad_()
    Y = (X.detach() * 2).sin().requires_grad_()
    Q = X.detach().new_tensor([[0.21], [0.74]]).requires_grad_()
    return X, Y, Q


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("convex", [True, False])
@pytest.mark.parametrize("dense", [False, True])
def test_exact_conditional_joint_law_and_shared_parameter_hessians(device, convex, dense):
    X, Y, Q = data(device)
    model = RobustRelevancePursuitSingleTaskGP(
        X, Y, outcome_transform=None, backend="torch", convex_parameterization=convex
    )
    sparse = model.likelihood.noise_covar
    sparse.expand_support([2, 0])
    if dense:
        sparse.to_dense()
    with torch.no_grad():
        sparse.raw_rho.fill_(0.16)
    new_X = X.detach().new_tensor([[0.4], [0.8]]).requires_grad_()
    new_Y = Y.detach().new_tensor([[0.3], [0.9]]).requires_grad_()
    noise = Y.detach().new_tensor([[0.04], [0.06]]).requires_grad_()
    original_rho = sparse.raw_rho
    conditional = model.condition_on_observations(new_X, new_Y, noise=noise)
    new_sparse = conditional.likelihood.noise_covar
    assert new_sparse.dim == 6 and sparse.dim == 4
    assert new_sparse.raw_rho is original_rho
    assert new_sparse.support == (list(range(4)) if dense else [2, 0])
    torch.testing.assert_close(new_sparse.rho, torch.cat((sparse.rho, sparse.rho.new_zeros(2))))
    joint = model.posterior(torch.cat((new_X, Q)))
    mean = joint.mean.squeeze(-1)
    covariance = joint.distribution.covariance_matrix
    observed = covariance[:2, :2] + noise.squeeze(-1).diag_embed()
    cross = covariance[2:, :2]
    expected_mean = mean[2:] + cross @ torch.linalg.solve(observed, new_Y.squeeze(-1) - mean[:2])
    expected_covariance = covariance[2:, 2:] - cross @ torch.linalg.solve(observed, cross.T)
    actual = conditional.posterior(Q)
    torch.testing.assert_close(actual.mean.squeeze(-1), expected_mean, rtol=1e-7, atol=1e-7)
    torch.testing.assert_close(
        actual.distribution.covariance_matrix, expected_covariance, rtol=1e-7, atol=1e-7
    )
    loss = actual.mean.square().sum() + actual.distribution.covariance_matrix.square().sum()
    expected = expected_mean.square().sum() + expected_covariance.square().sum()
    variables = (
        X,
        Y,
        Q,
        new_X,
        new_Y,
        noise,
        original_rho,
        *model.covar_module.parameters(),
        *model.mean_module.parameters(),
        *sparse.base_noise.parameters(),
    )
    ga = torch.autograd.grad(loss, variables, create_graph=True, retain_graph=True)
    gb = torch.autograd.grad(expected, variables, create_graph=True, retain_graph=True)
    for a, b in zip(ga, gb, strict=True):
        torch.testing.assert_close(a, b, rtol=2e-6, atol=2e-6)
    ha = torch.autograd.grad(sum(g.square().sum() for g in ga), variables)
    hb = torch.autograd.grad(sum(g.square().sum() for g in gb), variables)
    for a, b in zip(ha, hb, strict=True):
        torch.testing.assert_close(a, b, rtol=2e-5, atol=2e-5)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("fixed", [False, True])
def test_multioutput_model_noise_fantasy_batch_broadcast_and_frozen_transforms(device, fixed):
    from botorch_rs.models.transforms.input import Normalize

    X, Y, Q = data(device)
    X = X.expand(2, -1, -1).clone().detach().requires_grad_()
    Y = torch.cat((Y, Y + 0.2), -1).expand(2, -1, -1).clone().detach().requires_grad_()
    known = torch.full_like(Y, 0.03).requires_grad_() if fixed else None
    model = RobustRelevancePursuitSingleTaskGP(
        X, Y, known, backend="torch", input_transform=Normalize(1)
    )
    sparse = model.likelihood.noise_covar
    sparse.expand_support([2])
    with torch.no_grad():
        sparse.raw_rho.fill_(0.13)
    new_Y = Y.new_full((3, 2, 2, 2), 0.5).requires_grad_()
    # Noise contributes a leading fantasy axis even when targets and inputs
    # initially have no such axis.
    supplied_noise = Y.new_full((3, 2, 2, 2), 0.04).requires_grad_()
    conditioned = model.condition_on_observations(Q, new_Y[0], noise=supplied_noise)
    assert conditioned.batch_shape == torch.Size([3, 2])
    assert conditioned.likelihood.noise_covar.raw_rho is sparse.raw_rho
    posterior = conditioned.posterior(Q)
    assert posterior.mean.shape == (3, 2, 2, 2)
    variables = (X, Y, Q, supplied_noise, sparse.raw_rho) + (() if known is None else (known,))
    gradients = torch.autograd.grad(
        posterior.mean.square().sum() + posterior.variance.sum(), variables, create_graph=True
    )
    hessian = torch.autograd.grad(sum(g.square().sum() for g in gradients), variables)
    assert all(
        g.device.type == device and bool(torch.isfinite(g).all()) for g in (*gradients, *hessian)
    )
    before = conditioned.likelihood.noise_covar.raw_rho
    with pytest.raises(UnsupportedError, match="ranking"):
        fit_gpytorch_mll(ExactMarginalLogLikelihood(conditioned.likelihood, conditioned))
    assert conditioned.likelihood.noise_covar.raw_rho is before
    fantasy = model.fantasize(
        Q,
        IIDNormalSampler(torch.Size([2]), seed=321),
        observation_noise=Y.new_full((2, 2), 0.04),
        propagate_grads=True,
    )
    assert fantasy.posterior(Q).mean.shape == (2, 2, 2, 2)
    assert fantasy.likelihood.noise_covar.raw_rho is sparse.raw_rho
    gradient = torch.autograd.grad(fantasy.posterior(Q).mean.square().sum(), sparse.raw_rho)[0]
    assert bool(torch.isfinite(gradient).all())


@pytest.mark.parametrize("device", DEVICES)
def test_official_apportioned_timeout_reaches_every_actual_fit(device):
    X, Y, _ = data(device)
    model = RobustRelevancePursuitSingleTaskGP(
        X.detach(), Y.detach(), backend="torch", outcome_transform=None
    )
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    captured = []
    original = TorchSparseOutlierNoise.optimize_mll

    def record(self, *args, **kwargs):
        captured.append(kwargs["optimizer_kwargs"]["timeout_sec"])
        return original(self, *args, **kwargs)

    with patch.object(TorchSparseOutlierNoise, "optimize_mll", record):
        fit_gpytorch_mll(
            mll,
            numbers_of_outliers=[1, 0],
            timeout_sec=2.4,
            optimizer_kwargs={"options": {"maxiter": 1}},
        )
    assert captured == [1.2, 1.2, 1.2]
    assert mll.model is model and mll.likelihood is model.likelihood
    for invalid in (-1.0, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="timeout"):
            fit_gpytorch_mll(mll, numbers_of_outliers=[0], timeout_sec=invalid)
    assert mll.model is model and mll.likelihood is model.likelihood


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("fixed", [False, True])
def test_default_fantasies_are_new_observation_labels_with_base_noise(device, fixed):
    X, Y, Q = data(device)
    known = torch.full_like(Y, 0.03) if fixed else None
    model = RobustRelevancePursuitSingleTaskGP(X, Y, known, backend="torch", outcome_transform=None)
    sparse = model.likelihood.noise_covar
    sparse.expand_support([2])
    with torch.no_grad():
        sparse.raw_rho.fill_(0.2)
    captured = []

    def sampler(posterior):
        captured.append(posterior)
        return posterior.mean.unsqueeze(0)

    for locations in (X, Q):
        fantasy = model.fantasize(locations, sampler)
        latent = model.posterior(locations)
        base = known.mean() if fixed else sparse.base_noise.noise.squeeze()
        torch.testing.assert_close(
            captured[-1].variance - latent.variance, base.expand_as(latent.variance)
        )
        torch.testing.assert_close(
            fantasy.likelihood.noise_covar.rho[..., -locations.shape[-2] :],
            sparse.raw_rho.new_zeros(locations.shape[-2]),
        )
        assert fantasy.likelihood.noise_covar.raw_rho is sparse.raw_rho

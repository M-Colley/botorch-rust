# ruff: noqa: E402
"""Empirical-prior public helpers retain device, shared leaves and exact laws."""

from __future__ import annotations

import pytest

pytest.importorskip("gpytorch")
pytest.importorskip("torch")


import copy

import gpytorch
import torch
from test_torch_em_empirical_gp import DEVICES, data, modules

from botorch_rs.fit import fit_gpytorch_mll
from botorch_rs.models import SingleTaskGP
from botorch_rs.models.empirical_gps import build_shared_gp_model_list
from botorch_rs.models.empirical_gps.em_coordinate_ascent import (
    KLPriorFitMLL,
    coordinate_ascent_em,
    fit_kernel_to_em_prior,
    kl_divergence_mvn,
)
from botorch_rs.models.empirical_gps.em_empirical_gp import pretrain_em_prior


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("batched", [False, True])
def test_all_four_gaussian_kl_mixed_hessians(device, batched):
    torch.manual_seed(13)
    shape = (2, 3) if batched else (3,)
    p, q = [
        torch.randn(shape, device=device, dtype=torch.double).requires_grad_() for _ in range(2)
    ]
    factors = [
        torch.randn((1, 3, 3) if batched else (3, 3), device=device, dtype=torch.double)
        for _ in range(2)
    ]
    a, b = [
        (factor @ factor.mT + torch.eye(3, device=device, dtype=torch.double)).requires_grad_()
        for factor in factors
    ]
    actual = kl_divergence_mvn(p, a, q, b)
    expected = torch.distributions.kl_divergence(
        torch.distributions.MultivariateNormal(p, a), torch.distributions.MultivariateNormal(q, b)
    )
    torch.testing.assert_close(actual, expected, rtol=1e-11, atol=1e-11)
    variables = p, a, q, b
    ga = torch.autograd.grad(actual.sum(), variables, create_graph=True)
    ge = torch.autograd.grad(expected.sum(), variables, create_graph=True)
    for first, second in zip(ga, ge, strict=True):
        torch.testing.assert_close(first, second, rtol=1e-10, atol=1e-10)
    ha = torch.autograd.grad(sum(g.square().sum() for g in ga), variables)
    he = torch.autograd.grad(sum(g.square().sum() for g in ge), variables)
    for first, second in zip(ha, he, strict=True):
        torch.testing.assert_close(first, second, rtol=2e-10, atol=2e-10)


@pytest.mark.parametrize("device", DEVICES)
def test_kl_mll_live_reference_coords_detached_targets_and_actual_fit(device):
    mean, kernel, likelihood = modules(device, True)
    X = data(device)[0].X.clone().requires_grad_()
    model = SingleTaskGP(
        X[:1],
        X.new_zeros(1, 1),
        mean_module=mean,
        covar_module=kernel,
        backend="torch",
        outcome_transform=None,
    )
    target_mean = X.new_full((3,), 0.7, requires_grad=True)
    target_covariance = (
        dense(kernel, X, X).detach() + 0.3 * torch.eye(3, device=device, dtype=X.dtype)
    ).requires_grad_()
    mll = KLPriorFitMLL(model.likelihood, model, X, target_mean, target_covariance)
    actual = mll()
    expected = -torch.distributions.kl_divergence(
        torch.distributions.MultivariateNormal(target_mean.detach(), target_covariance.detach()),
        torch.distributions.MultivariateNormal(mean(X), dense(kernel, X, X)),
    )
    variables = X, mean.raw_constant, kernel.base_kernel.raw_lengthscale
    a = torch.autograd.grad(actual, variables, create_graph=True, retain_graph=True)
    b = torch.autograd.grad(expected, variables, create_graph=True)
    for first, second in zip(a, b, strict=True):
        torch.testing.assert_close(first, second)
    assert all(
        v is None
        for v in torch.autograd.grad(
            actual, (target_mean, target_covariance, model.likelihood.raw_noise), allow_unused=True
        )
    )
    fit_gpytorch_mll(mll, optimizer_kwargs={"options": {"maxiter": 4}})
    assert not mll.training
    assert mll() > actual.detach()


def dense(module, X, Z):
    value = module(X, Z)
    return value if isinstance(value, torch.Tensor) else value.to_dense()


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("native_descriptors", [False, True])
def test_standalone_kernel_fit_updates_caller_owned_parameters(device, native_descriptors):
    mean, kernel, _ = modules(device, not native_descriptors)
    X = data(device)[0].X
    expected_mean = X.new_full((3,), 1.1)
    desired = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel()).to(X)
    desired.base_kernel.lengthscale = 0.27
    desired.outputscale = 0.85
    target = dense(desired, X, X).detach()
    identities = tuple(map(id, kernel.parameters())) if not native_descriptors else ()
    fit_kernel_to_em_prior(mean, kernel, X, expected_mean, target, backend="torch")
    if not native_descriptors:
        assert tuple(map(id, kernel.parameters())) == identities
        value = -kl_divergence_mvn(expected_mean, target, mean(X), dense(kernel, X, X))
    else:
        # This explicitly inspects converted descriptor state after fitting;
        # numerical optimization used the live Tensor modules, on device.
        from botorch_rs.models._torch_backend_modules import build_kernel, build_mean

        converted_mean, converted_kernel = build_mean(mean, (), X), build_kernel(kernel, 1, (), X)
        value = -kl_divergence_mvn(expected_mean, target, converted_mean(X), converted_kernel(X, X))
    assert value.abs() < 2e-7


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("fixed", [False, True])
def test_shared_models_raw_target_joint_mll_gradients_and_leaf_identity(device, fixed):
    mean, kernel, _ = modules(device, True)
    histories = data(device)
    histories = [
        type(d)(d.X.clone().requires_grad_(), d.Y.clone().requires_grad_()) for d in histories
    ]
    model, mll = build_shared_gp_model_list(
        histories, mean, kernel, observation_noise=0.035 if fixed else None, backend="torch"
    )
    assert all(child.mean_module is mean and child.covar_module is kernel for child in model.models)
    assert all(child.outcome_transform is None for child in model.models)
    actual = mll()
    # An eval transition must not copy shared leaves into themselves and
    # invalidate the existing forward's autograd versions.
    mll.eval()
    expected = actual.new_zeros(())
    for child, dataset in zip(model.models, histories, strict=True):
        distribution = torch.distributions.MultivariateNormal(
            mean(dataset.X),
            dense(kernel, dataset.X, dataset.X)
            + child.likelihood.noise
            * torch.eye(dataset.X.shape[0], device=device, dtype=dataset.X.dtype),
        )
        expected += (
            distribution.log_prob(dataset.Y.squeeze(-1)) + child._log_prior()
        ) / dataset.X.shape[0]
    expected /= len(histories)
    torch.testing.assert_close(actual, expected)
    variables = (
        kernel.base_kernel.raw_lengthscale,
        mean.raw_constant,
        histories[1].X,
        histories[2].Y,
    )
    ga = torch.autograd.grad(actual, variables, create_graph=True)
    ge = torch.autograd.grad(expected, variables, create_graph=True)
    for first, second in zip(ga, ge, strict=True):
        torch.testing.assert_close(first, second, rtol=1e-9, atol=1e-9)
    ha = torch.autograd.grad(sum(g.square().sum() for g in ga), variables)
    he = torch.autograd.grad(sum(g.square().sum() for g in ge), variables)
    for first, second in zip(ha, he, strict=True):
        torch.testing.assert_close(first, second, rtol=1e-8, atol=1e-8)
    ids = tuple(map(id, kernel.parameters()))
    noise = [child.likelihood.raw_noise.detach().clone() for child in model.models]
    before = mll().detach()
    fit_gpytorch_mll(mll, optimizer_kwargs={"options": {"maxiter": 3}})
    assert tuple(map(id, kernel.parameters())) == ids
    assert mll() >= before - 1e-8
    if fixed:
        for child, old in zip(model.models, noise, strict=True):
            torch.testing.assert_close(child.likelihood.raw_noise, old)
    else:
        assert any(
            not torch.equal(child.likelihood.raw_noise, old)
            for child, old in zip(model.models, noise, strict=True)
        )


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize(
    "adam,optimize", [(False, False), (True, False), (None, True), (False, True)]
)
def test_coordinate_ascent_real_fits_locations_and_deep_history(device, adam, optimize):
    histories = data(device)
    mean, kernel, _ = modules(device, True)
    Z = histories[0].X.new_tensor([[0.05], [0.45], [0.95]])
    before = [p.detach().clone() for p in kernel.parameters()]
    history = []
    torch.manual_seed(84)
    prior = coordinate_ascent_em(
        histories,
        mean,
        kernel,
        likelihood_noise=0.045,
        num_em_iterations=2,
        n_coord_ascent_iterations=2,
        inducing_points=Z,
        optimize_inducing_points=optimize,
        mll_use_adam=adam,
        mll_adam_steps=3,
        mll_adam_lr=0.015,
        mll_max_iter=3,
        mll_dataset_subsample=2,
        container_history=history,
    )
    assert len(history) == 3
    assert any(not torch.equal(p, old) for p, old in zip(kernel.parameters(), before, strict=True))
    assert len(prior.datasets) == len(histories)
    assert all(len(snapshot.datasets) == len(histories) for snapshot in history)
    if optimize:
        assert not torch.equal(prior.X_inducing, Z)
    else:
        torch.testing.assert_close(prior.X_inducing, Z)
    saved = history[0].Sigma_inducing.clone()
    kernel.outputscale = 4.0
    torch.testing.assert_close(history[0].Sigma_inducing, saved)
    assert history[0].covar_module is not kernel
    torch.testing.assert_close(history[-1].Sigma_inducing, prior.Sigma_inducing)
    # The returned moments match a full-data EM run with the final fitted
    # snapshot; the numerical coordinate step used only its selected subset.
    rebuilt = pretrain_em_prior(
        histories,
        copy.deepcopy(prior.mean_module),
        copy.deepcopy(prior.covar_module),
        likelihood_noise=0.045,
        inducing_points=prior.X_inducing,
        num_em_iterations=2,
        backend="torch",
    )
    torch.testing.assert_close(prior.mu_inducing, rebuilt.mu_inducing)
    torch.testing.assert_close(prior.Sigma_inducing, rebuilt.Sigma_inducing)


@pytest.mark.parametrize("device", DEVICES)
def test_default_coordinate_noop_and_configuration_validation(device):
    mean, kernel, _ = modules(device, True)
    before = [p.detach().clone() for p in kernel.parameters()]
    history = []
    prior = coordinate_ascent_em(
        data(device),
        mean,
        kernel,
        likelihood_noise=0.045,
        num_em_iterations=2,
        n_coord_ascent_iterations=1,
        container_history=history,
    )
    assert len(history) == 2
    assert torch.isfinite(prior.Sigma_inducing).all()
    for parameter, old in zip(kernel.parameters(), before, strict=True):
        torch.testing.assert_close(parameter, old)
    with pytest.raises(ValueError, match="explicit"):
        coordinate_ascent_em(
            data(device), mean, kernel, optimize_inducing_points=True, backend="torch"
        )
    with pytest.raises(ValueError, match="positive integer"):
        coordinate_ascent_em(data(device), mean, kernel, mll_dataset_subsample=0, backend="torch")
    with pytest.raises(ValueError, match="lower bound"):
        build_shared_gp_model_list(
            data(device), mean, kernel, observation_noise=-0.1, backend="torch"
        )


@pytest.mark.parametrize("device", DEVICES)
def test_coordinate_native_descriptors_update_boundary_state_and_zero_cycles(device):
    mean, kernel, _ = modules(device)
    before = list(kernel._raw_flat())
    Z = data(device)[0].X.new_tensor([[0.05], [0.45], [0.95]])
    prior = coordinate_ascent_em(
        data(device),
        mean,
        kernel,
        likelihood_noise=0.045,
        inducing_points=Z,
        num_em_iterations=2,
        mll_use_adam=True,
        mll_adam_steps=3,
        backend="torch",
    )
    assert kernel._raw_flat() != before
    from botorch_rs.models._torch_backend_modules import build_kernel, build_mean

    torch.testing.assert_close(build_mean(mean, (), Z)(Z), prior.mean_module(Z))
    torch.testing.assert_close(build_kernel(kernel, 1, (), Z)(Z, Z), prior.covar_module(Z, Z))
    saved = list(kernel._raw_flat())
    history = []
    no_cycles = coordinate_ascent_em(
        data(device),
        mean,
        kernel,
        inducing_points=Z,
        num_em_iterations=2,
        n_coord_ascent_iterations=0,
        container_history=history,
        backend="torch",
    )
    assert saved == kernel._raw_flat()
    assert len(history) == 1
    torch.testing.assert_close(history[0].Sigma_inducing, no_cycles.Sigma_inducing)

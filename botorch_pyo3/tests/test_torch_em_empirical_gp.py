"""Device-local empirical EM laws and complete training graphs."""

from __future__ import annotations

import copy
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import gpytorch
import pytest
import test_em_empirical_gp as reference_helpers
import torch

from botorch_rs.fit import fit_gpytorch_mll
from botorch_rs.kernels import RBFKernel, ScaleKernel
from botorch_rs.means import ConstantMean
from botorch_rs.models.empirical_gps.em_empirical_gp import (
    EMEmpiricalGaussianProcess,
    EMEmpiricalMarginalLogLikelihood,
    EMPriorContainer,
    pretrain_em_prior,
)
from botorch_rs.models.empirical_gps.utils import ExperimentDataset
from botorch_rs.optim.fit import fit_gpytorch_mll_torch
from botorch_rs.sampling.normal import IIDNormalSampler


@pytest.fixture(scope="module")
def corrected_official():
    fixture = reference_helpers.official.__wrapped__()
    original = next(fixture)
    source = Path(original.em_empirical_gp.__file__).read_text(encoding="utf-8")
    missing = "Sigma_current=Sigma,\n            psd_stabilization=True,"
    repair = "Sigma_current=Sigma,\n            m_0=_evaluate_mean(mean_module, X_inducing),\n            psd_stabilization=True,"
    assert source.count(missing) == 1
    module = ModuleType("botorch_corrected_em_mean_prior_reference")
    sys.modules[module.__name__] = module
    exec(
        compile(source.replace(missing, repair), original.em_empirical_gp.__file__, "exec"),
        module.__dict__,
    )
    # The only changed line implements the documented prior mu ~ N(m(Z), K).
    # This known upstream defect is already repaired by BoTorch PR #3388.
    try:
        yield SimpleNamespace(utils=original.utils, em_empirical_gp=module)
    finally:
        sys.modules.pop(module.__name__, None)
        fixture.close()


DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


def data(device):
    grid = torch.tensor([[0.0], [0.2], [0.5], [0.8], [1.0]], device=device, dtype=torch.double)
    return [
        ExperimentDataset(grid[index], 0.3 * j + torch.sin((1.1 + j * 0.2) * grid[index]))
        for j, index in enumerate(([0, 1, 3], [0, 2, 4], [1, 2, 3, 4]))
    ]


def modules(device, reference=False):
    if reference:
        mean = gpytorch.means.ConstantMean().to(device=device, dtype=torch.double)
        kernel = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel()).to(
            device=device, dtype=torch.double
        )
        likelihood = gpytorch.likelihoods.GaussianLikelihood().to(device=device, dtype=torch.double)
    else:
        mean, kernel = ConstantMean(), ScaleKernel(RBFKernel())
        likelihood = gpytorch.likelihoods.GaussianLikelihood().to(device=device, dtype=torch.double)
    mean.constant = torch.tensor(0.23, device=device, dtype=torch.double)
    kernel.base_kernel.lengthscale = torch.tensor(0.38, device=device, dtype=torch.double)
    kernel.outputscale = torch.tensor(1.2, device=device, dtype=torch.double)
    likelihood.noise = torch.tensor(0.045, device=device, dtype=torch.double)
    return mean, kernel, likelihood


def assert_moments(a, b):
    for field in ("mean", "variance"):
        torch.testing.assert_close(getattr(a, field), getattr(b, field), rtol=2e-7, atol=3e-9)
    torch.testing.assert_close(
        a.distribution.covariance_matrix, b.distribution.covariance_matrix, rtol=2e-7, atol=3e-9
    )


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("inducing", [False, True])
@pytest.mark.parametrize("prior,init", [(False, "kernel"), (True, "kernel"), (True, "naive")])
def test_pretraining_official(device, inducing, prior, init, corrected_official):
    histories = data(device)
    mean, kernel, _ = modules(device)
    ref_mean, ref_kernel, _ = modules(device, True)
    Z = histories[0].X.new_tensor([[0.05], [0.45], [0.9]]) if inducing else None
    actual_history, expected_history = [], []
    options = dict(
        likelihood_noise=0.045,
        inducing_points=Z,
        num_em_iterations=3,
        use_mean_prior=prior,
        use_covar_prior=prior,
        init_mode=init,
        covariance_shrinkage=0.13,
        em_convergence_tol=None,
    )
    actual = pretrain_em_prior(
        histories, mean, kernel, backend="torch", iteration_history=actual_history, **options
    )
    expected = corrected_official.em_empirical_gp.pretrain_em_prior(
        histories, ref_mean, ref_kernel, iteration_history=expected_history, **options
    )
    for field in ("mu_inducing", "Sigma_inducing", "L_kernel_inducing", "delta_mu"):
        torch.testing.assert_close(
            getattr(actual, field), getattr(expected, field), rtol=2e-8, atol=3e-10
        )
        assert getattr(actual, field).device.type == device
        assert not getattr(actual, field).requires_grad
    assert len(actual_history) == len(expected_history) == 3
    for a, b in zip(actual_history, expected_history, strict=True):
        torch.testing.assert_close(a[0], b[0], rtol=2e-8, atol=3e-10)
        torch.testing.assert_close(a[1], b[1], rtol=2e-8, atol=3e-10)
    # The reusable prior owns a deep historical/module snapshot.
    before = actual.datasets[0].Y.clone()
    histories[0].Y.add_(3)
    torch.testing.assert_close(actual.datasets[0].Y, before)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("inducing,warm", [(False, False), (True, False), (True, True)])
def test_model_mll_and_warm_state_official(device, inducing, warm, corrected_official):
    histories = data(device)
    mean, kernel, likelihood = modules(device)
    ref_mean, ref_kernel, ref_likelihood = modules(device, True)
    X = histories[0].X.new_tensor([[0.12], [0.64]])
    Y = X.sin() + 0.1
    Z = X.new_tensor([[0.03], [0.43], [0.96]]) if inducing else None
    options = dict(
        datasets=histories,
        inducing_points=Z,
        num_em_iterations=2,
        use_mean_prior=True,
        use_covar_prior=True,
        covariance_shrinkage=0.15,
        warm_start_em=warm,
        em_convergence_tol=None,
    )
    model = EMEmpiricalGaussianProcess(
        X,
        Y,
        mean_module=mean,
        covar_module=kernel,
        likelihood=likelihood,
        backend="torch",
        **options,
    )
    ref = corrected_official.em_empirical_gp.EMEmpiricalGaussianProcess(
        X, Y, mean_module=ref_mean, covar_module=ref_kernel, likelihood=ref_likelihood, **options
    )
    mll = EMEmpiricalMarginalLogLikelihood(model.likelihood, model)
    ref_mll = corrected_official.em_empirical_gp.EMEmpiricalMarginalLogLikelihood(
        ref.likelihood, ref
    )
    for _ in range(2):
        model.train()
        ref.train()
        actual, expected = mll(), ref_mll(None, None)
        torch.testing.assert_close(actual, expected, rtol=3e-8, atol=3e-10)
        leaves = (
            model.covar_module.base_kernel.raw_lengthscale,
            model.covar_module.raw_outputscale,
            model.mean_module.raw_constant,
            model.likelihood.raw_noise,
        )
        ref_leaves = (
            ref_kernel.base_kernel.raw_lengthscale,
            ref_kernel.raw_outputscale,
            ref_mean.raw_constant,
            ref_likelihood.raw_noise,
        )
        for a, b in zip(
            torch.autograd.grad(actual, leaves, create_graph=True),
            torch.autograd.grad(expected, ref_leaves, create_graph=True),
            strict=True,
        ):
            torch.testing.assert_close(a, b, rtol=3e-7, atol=2e-9)
        torch.testing.assert_close(model._mu_inducing, ref._mu_inducing, rtol=3e-8, atol=3e-10)
        torch.testing.assert_close(
            model._Sigma_inducing, ref._Sigma_inducing, rtol=3e-8, atol=3e-10
        )
    Q = X.new_tensor([[[0.19], [0.71]], [[0.31], [0.84]]], requires_grad=True)
    # The source's cached interpolation matrices need current parameters.
    model.eval()
    ref.eval()
    assert_moments(model.posterior(Q), ref.posterior(Q))
    assert model._em_refreshes == 2


@pytest.mark.parametrize("device", DEVICES)
def test_cold_full_historical_and_inducing_mixed_hessians(device, corrected_official):
    histories = data(device)
    histories = [
        ExperimentDataset(d.X.clone().requires_grad_(), d.Y.clone().requires_grad_())
        for d in histories
    ]
    X = histories[0].X.new_tensor([[0.12], [0.64]], requires_grad=True)
    Y = X.detach().sin().requires_grad_()
    Z = X.new_tensor([[0.03], [0.43], [0.96]])
    mean, kernel, likelihood = modules(device)
    model = EMEmpiricalGaussianProcess(
        X,
        Y,
        datasets=histories,
        mean_module=mean,
        covar_module=kernel,
        likelihood=likelihood,
        inducing_points=Z,
        num_em_iterations=2,
        use_mean_prior=True,
        use_covar_prior=True,
        covariance_shrinkage=0.2,
        learnable_inducing_points=True,
        em_convergence_tol=None,
        backend="torch",
    )
    mll = EMEmpiricalMarginalLogLikelihood(model.likelihood, model)
    # Use the independent official Tensor loop with actual port parameter
    # leaves; no model delegate and no native float64 extraction.
    oracle = corrected_official.em_empirical_gp
    mu, cov = oracle._run_em_algorithm(
        histories,
        model.mean_module(model._X_inducing),
        model.covar_module(model._X_inducing, model._X_inducing),
        model._X_inducing,
        model.mean_module,
        model.covar_module,
        model.likelihood.noise,
        None,
        True,
        2,
        (model.iw_nu + 4) * model.covar_module(model._X_inducing, model._X_inducing),
        model.covar_module(model._X_inducing, model._X_inducing),
        model.iw_nu,
        None,
        3,
        covariance_shrinkage=0.2,
        shrinkage_target=model.covar_module(model._X_inducing, model._X_inducing),
    )
    factor = torch.linalg.cholesky(model.covar_module(model._X_inducing, model._X_inducing))
    loss = X.new_zeros(())
    for d in histories:
        m, c, _ = oracle._interpolate_prior(
            d.X,
            model.mean_module,
            model.covar_module,
            model._X_inducing,
            factor,
            mu - model.mean_module(model._X_inducing),
            cov,
        )
        loss = loss + torch.distributions.MultivariateNormal(
            m, c + model.likelihood.noise * torch.eye(d.X.shape[0], device=device, dtype=X.dtype)
        ).log_prob(d.Y.squeeze(-1))
    loss = loss / sum(d.X.shape[0] for d in histories)
    actual = mll()
    variables = (
        model._X_inducing,
        histories[0].X,
        histories[1].Y,
        model.covar_module.base_kernel.raw_lengthscale,
        model.likelihood.raw_noise,
    )
    a = torch.autograd.grad(actual, variables, create_graph=True)
    b = torch.autograd.grad(loss, variables, create_graph=True)
    for ag, bg in zip(a, b, strict=True):
        torch.testing.assert_close(ag, bg, rtol=2e-6, atol=2e-8)
    ah = torch.autograd.grad(sum(g.square().sum() for g in a), variables)
    bh = torch.autograd.grad(sum(g.square().sum() for g in b), variables)
    for ag, bg in zip(ah, bh, strict=True):
        torch.testing.assert_close(ag, bg, rtol=1e-5, atol=3e-7)


@pytest.mark.parametrize("device", DEVICES)
def test_pretrained_live_residual_condition_fantasy_fit_and_roundtrip(device, tmp_path):
    histories = data(device)
    mean, kernel, likelihood = modules(device)
    prior = pretrain_em_prior(
        histories,
        mean,
        kernel,
        inducing_points=histories[0].X.new_tensor([[0.05], [0.5], [0.95]]),
        likelihood_noise=0.045,
        num_em_iterations=2,
        backend="torch",
    )
    path = tmp_path / "tensor_prior.pkl"
    prior.save(str(path))
    prior = EMPriorContainer.load(str(path))
    X = histories[0].X.new_tensor([[0.13], [0.63]], requires_grad=True)
    Y = X.detach().sin().requires_grad_()
    model = EMEmpiricalGaussianProcess.from_pretrained(
        prior,
        X,
        Y,
        likelihood=likelihood,
        learnable_inducing_points=True,
        base_covar_module=ScaleKernel(RBFKernel()),
        backend="torch",
    )
    assert not any(p.requires_grad for p in model.initial_covar_module.parameters())
    assert model._X_inducing.requires_grad
    Q = X.new_tensor([[0.29], [0.83]], requires_grad=True)
    for _ in range(2):
        p = model.posterior(Q, observation_noise=Q.square() * 0.01)
        sample = p.rsample(torch.Size([3]))
        variables = (X, Y, Q, model._X_inducing, model._additive_base.raw_outputscale)
        gradients = torch.autograd.grad(
            sample.square().sum() + p.variance.sum(), variables, create_graph=True
        )
        hessian = torch.autograd.grad(sum(g.square().sum() for g in gradients), variables)
        assert all(torch.isfinite(g).all() for g in (*gradients, *hessian))
    extra = Q[:1]
    conditioned = model.condition_on_observations(extra, extra.sin())
    rebuilt = EMEmpiricalGaussianProcess.from_pretrained(
        prior,
        torch.cat((X, extra)),
        torch.cat((Y, extra.sin())),
        likelihood=copy.deepcopy(model.likelihood),
        base_covar_module=copy.deepcopy(model._additive_base),
        backend="torch",
    )
    assert_moments(conditioned.posterior(Q), rebuilt.posterior(Q))
    fantasy = model.fantasize(
        extra, IIDNormalSampler(torch.Size([2]), seed=3), propagate_grads=True
    )
    assert fantasy.posterior(Q).mean.shape == (2, 2, 1)
    mll = EMEmpiricalMarginalLogLikelihood(model.likelihood, model)
    frozen = model._Sigma_inducing.clone()
    fit_gpytorch_mll(mll, optimizer=fit_gpytorch_mll_torch, optimizer_kwargs={"step_limit": 2})
    torch.testing.assert_close(model._Sigma_inducing, frozen)
    assert model._em_refreshes == 0


@pytest.mark.parametrize("device", DEVICES)
def test_indexed_no_interpolation_and_repeated_positive_eigenvalue_hessian(device):
    from botorch_rs.models.empirical_gps._torch_em import psd_projection

    histories = data(device)
    mean, kernel, likelihood = modules(device)
    model = EMEmpiricalGaussianProcess(
        histories[0].X,
        histories[0].Y,
        datasets=histories,
        mean_module=mean,
        covar_module=kernel,
        likelihood=likelihood,
        enable_interpolation=False,
        num_em_iterations=2,
        backend="torch",
    )
    assert torch.isfinite(model.posterior(histories[1].X).variance).all()
    with pytest.raises(ValueError, match="historical subset"):
        model.posterior(histories[0].X.new_tensor([[0.12]]))
    matrix = torch.eye(3, dtype=torch.double, device=device).requires_grad_()
    gradient = torch.autograd.grad(
        psd_projection(matrix).square().sum(), matrix, create_graph=True
    )[0]
    hessian = torch.autograd.grad(gradient.sum(), matrix)[0]
    torch.testing.assert_close(hessian, 2 * torch.ones_like(matrix))
    clipped = torch.diag(matrix.new_tensor([-1.0, -1.0, 2.0])).requires_grad_()
    first = torch.autograd.grad(psd_projection(clipped).square().sum(), clipped, create_graph=True)[
        0
    ]
    second = torch.autograd.grad(first.square().sum(), clipped)[0]
    assert torch.isfinite(first).all() and torch.isfinite(second).all()
    with torch.no_grad():
        torch.testing.assert_close(
            psd_projection(clipped), torch.diag(matrix.new_tensor([1e-10, 1e-10, 2.0]))
        )


@pytest.mark.parametrize("device", DEVICES)
def test_update_snapshots_warm_setters_and_historical_objective(device):
    histories = data(device)
    mean, kernel, likelihood = modules(device)
    X, Y = histories[0].X[:2], histories[0].Y[:2]
    model = EMEmpiricalGaussianProcess(
        X,
        Y,
        datasets=histories,
        mean_module=mean,
        covar_module=kernel,
        likelihood=likelihood,
        warm_start_em=True,
        inducing_points=X.new_tensor([[0.05], [0.5], [0.95]]),
        num_em_iterations=2,
        backend="torch",
    )
    mll = EMEmpiricalMarginalLogLikelihood(model.likelihood, model)
    first = mll()
    first_mean = model._mu_inducing.clone()
    second = mll()
    assert model._em_refreshes == 2
    assert not torch.equal(model._mu_inducing, first_mean)
    # A newer warm update must not mutate tensors saved by an older forward.
    for value in (first, second):
        gradient = torch.autograd.grad(value, model.likelihood.raw_noise, create_graph=True)[0]
        assert torch.isfinite(
            torch.autograd.grad(gradient.sum(), model.likelihood.raw_noise)[0]
        ).all()
    before = model._mu_inducing.clone()
    model.initialize(**{"covar_module.base_kernel.lengthscale": X.new_tensor(0.42)})
    torch.testing.assert_close(model._mu_inducing, before)
    # Evaluating cached posterior after backward creates fresh interpolation
    # graphs, and does not rerun EM or reuse a consumed update graph.
    model.eval()
    for _ in range(2):
        torch.autograd.grad(
            model.posterior(X).mean.sum(), model.covar_module.base_kernel.raw_lengthscale
        )
    assert model._em_refreshes == 2
    # A historical MLL never substitutes the target conditioning rows.
    model._using_pretrained_prior = True
    fixed_value = mll()
    model.set_train_data(targets=Y + 100)
    torch.testing.assert_close(mll(), fixed_value)
    # Independent fitted models sharing a container do not share its modules.
    native_mean, native_kernel, _ = modules("cpu")
    prior = pretrain_em_prior(
        data("cpu"), native_mean, native_kernel, num_em_iterations=1, backend="native"
    )
    a = EMEmpiricalGaussianProcess.from_pretrained(prior, X, Y, backend="torch")
    b = EMEmpiricalGaussianProcess.from_pretrained(prior, X, Y, backend="torch")
    assert a.initial_covar_module is not b.initial_covar_module
    assert a.initial_covar_module is not prior.covar_module
    assert a.posterior(X).mean.device.type == device


@pytest.mark.parametrize("device", DEVICES)
def test_float32_safe_factor_and_validation(device):
    from botorch_rs.exceptions.errors import UnsupportedError

    histories = [ExperimentDataset(d.X.float(), d.Y.float()) for d in data(device)]
    mean, kernel, _ = modules(device)
    prior = pretrain_em_prior(
        histories, mean, kernel, num_em_iterations=3, likelihood_noise=0.0, backend="torch"
    )
    assert torch.isfinite(prior.Sigma_inducing).all()
    X, Y = histories[0].X, histories[0].Y
    model = EMEmpiricalGaussianProcess.from_pretrained(prior, X, Y, backend="torch")
    assert model.posterior(X).mean.dtype == torch.float32
    with pytest.raises(UnsupportedError, match="explicit inducing"):
        EMEmpiricalGaussianProcess(
            X, Y, em_prior=prior, learnable_inducing_points=True, backend="torch"
        )
    previous = model.datasets
    with pytest.raises(ValueError, match="estimation grid"):
        model.datasets = [ExperimentDataset(X + 0.01, Y)]
    torch.testing.assert_close(model.datasets[0].X, previous[0].X)
    with pytest.raises(ValueError, match="between zero and one"):
        pretrain_em_prior(
            histories, mean, kernel, covariance_shrinkage=float("nan"), backend="torch"
        )
    with pytest.raises(ValueError, match="iw_nu"):
        pretrain_em_prior(histories, mean, kernel, use_covar_prior=True, iw_nu=1, backend="torch")
    bad = copy.copy(prior)
    object.__setattr__(bad, "Sigma_inducing", -torch.eye(prior.X_inducing.shape[0], device=device))
    with pytest.raises(ValueError, match="positive semidefinite"):
        EMEmpiricalGaussianProcess(X, Y, em_prior=bad, backend="torch")


@pytest.mark.parametrize("device", DEVICES)
def test_training_forward_historical_fit_and_target_exact_mll(device):
    from botorch_rs.mlls import ExactMarginalLogLikelihood
    from botorch_rs.optim.fit import fit_gpytorch_mll_scipy

    histories = data(device)
    mean, kernel, likelihood = modules(device)
    X = histories[0].X[:2].clone().requires_grad_()
    Y = histories[0].Y[:2].clone().requires_grad_()
    model = EMEmpiricalGaussianProcess(
        X,
        Y,
        datasets=histories,
        mean_module=mean,
        covar_module=kernel,
        likelihood=likelihood,
        num_em_iterations=2,
        use_mean_prior=True,
        use_covar_prior=True,
        backend="torch",
    )
    output = model(X)
    assert model._em_refreshes == 1
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    actual = mll(output, Y.squeeze(-1))
    expected = (
        torch.distributions.MultivariateNormal(
            output.mean,
            output.covariance_matrix
            + model.likelihood.noise * torch.eye(X.shape[0], device=device, dtype=X.dtype),
        ).log_prob(Y.squeeze(-1))
        / X.shape[0]
    )
    torch.testing.assert_close(actual, expected)
    variables = (X, Y, model.mean_module.raw_constant, model.likelihood.raw_noise)
    grads = torch.autograd.grad(actual, variables, create_graph=True)
    assert all(
        torch.isfinite(g).all()
        for g in torch.autograd.grad(sum(v.square().sum() for v in grads), variables)
    )
    historical = EMEmpiricalMarginalLogLikelihood(model.likelihood, model)
    before = model._mu_inducing.clone()
    fit_gpytorch_mll(
        historical, optimizer=fit_gpytorch_mll_scipy, optimizer_kwargs={"options": {"maxiter": 2}}
    )
    assert not model.training
    assert model._em_refreshes > 1
    assert not torch.equal(model._mu_inducing, before)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("inducing", [False, True])
def test_map_mean_prior_offset_equivariance(device, inducing):
    histories = data(device)
    offset = 7.0
    shifted = [ExperimentDataset(d.X, d.Y + offset) for d in histories]
    mean, kernel, likelihood = modules(device)
    shifted_mean, shifted_kernel, shifted_likelihood = modules(device)
    shifted_mean.constant = histories[0].X.new_tensor(0.23 + offset)
    locations = histories[0].X.new_tensor([[0.05], [0.5], [0.95]]) if inducing else None
    settings = dict(
        num_em_iterations=3,
        use_mean_prior=True,
        use_covar_prior=True,
        inducing_points=locations,
        em_convergence_tol=None,
        backend="torch",
    )
    a = pretrain_em_prior(histories, mean, kernel, likelihood_noise=0.045, **settings)
    b = pretrain_em_prior(shifted, shifted_mean, shifted_kernel, likelihood_noise=0.045, **settings)
    torch.testing.assert_close(b.mu_inducing, a.mu_inducing + offset, rtol=1e-10, atol=1e-10)
    torch.testing.assert_close(b.Sigma_inducing, a.Sigma_inducing, rtol=1e-10, atol=1e-10)
    X, Y = histories[0].X[:2], histories[0].Y[:2]
    model = EMEmpiricalGaussianProcess(
        X,
        Y,
        datasets=histories,
        mean_module=mean,
        covar_module=kernel,
        likelihood=likelihood,
        **settings,
    )
    shifted_model = EMEmpiricalGaussianProcess(
        X,
        Y + offset,
        datasets=shifted,
        mean_module=shifted_mean,
        covar_module=shifted_kernel,
        likelihood=shifted_likelihood,
        **settings,
    )
    mll = EMEmpiricalMarginalLogLikelihood(model.likelihood, model)
    shifted_mll = EMEmpiricalMarginalLogLikelihood(shifted_model.likelihood, shifted_model)
    actual, translated = mll(), shifted_mll()
    torch.testing.assert_close(actual, translated, rtol=1e-10, atol=1e-10)
    a = torch.autograd.grad(actual, (model.mean_module.raw_constant, model.likelihood.raw_noise))
    b = torch.autograd.grad(
        translated, (shifted_model.mean_module.raw_constant, shifted_model.likelihood.raw_noise)
    )
    for first, second in zip(a, b, strict=True):
        torch.testing.assert_close(first, second, rtol=1e-9, atol=1e-10)

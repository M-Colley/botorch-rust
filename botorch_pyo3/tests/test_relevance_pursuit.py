# ruff: noqa: E402
"""Sparse-noise and pursuit regressions against official BoTorch."""

import pytest

pytest.importorskip("numpy")
pytest.importorskip("torch")
pytest.importorskip("botorch")
pytest.importorskip("gpytorch")


import copy

import numpy as np
import torch
from botorch.fit import fit_gpytorch_mll as ref_fit
from botorch.models import SingleTaskGP as RefGP
from botorch.models.likelihoods.sparse_outlier_noise import (
    SparseOutlierGaussianLikelihood as RefLikelihood,
)
from botorch.models.likelihoods.sparse_outlier_noise import SparseOutlierNoise as RefNoise
from botorch.models.relevance_pursuit import get_posterior_over_support as ref_support_posterior
from botorch.models.robust_relevance_pursuit_model import (
    RobustRelevancePursuitSingleTaskGP as RefRobustGP,
)
from botorch.models.utils.gpytorch_modules import (
    get_gaussian_likelihood_with_lognormal_prior as ref_base,
)
from gpytorch.mlls import ExactMarginalLogLikelihood as RefMLL
from gpytorch.priors import NormalPrior as RefNormalPrior

from botorch_rs.exceptions.warnings import InputDataWarning
from botorch_rs.fit import fit_gpytorch_mll
from botorch_rs.mlls import ExactMarginalLogLikelihood
from botorch_rs.models import SingleTaskGP
from botorch_rs.models.likelihoods.sparse_outlier_noise import (
    SparseOutlierGaussianLikelihood,
    SparseOutlierNoise,
)
from botorch_rs.models.relevance_pursuit import (
    backward_relevance_pursuit,
    forward_relevance_pursuit,
    get_posterior_over_support,
    initialize_dense_parameters,
)
from botorch_rs.models.robust_relevance_pursuit_model import RobustRelevancePursuitSingleTaskGP
from botorch_rs.models.utils.gpytorch_modules import get_gaussian_likelihood_with_lognormal_prior
from botorch_rs.priors import NormalPrior
from botorch_rs.utils.constraints import NonTransformedInterval

X = torch.tensor([[0.0], [0.2], [0.45], [0.7], [1.0]], dtype=torch.float64)
Y = torch.tensor([[0.0], [0.5], [4.0], [0.7], [-0.1]], dtype=torch.float64)


@pytest.fixture(autouse=True)
def canonical_factory_dtype():
    """Author BoTorch's default priors in float32, then construct double GPs."""
    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.float32)
    try:
        yield
    finally:
        torch.set_default_dtype(previous)


def pair(convex=True, kind="torch", support=None, rho_prior=False):
    support = [2, 0] if support is None else support
    reference_lik = RefLikelihood(
        ref_base().noise_covar,
        len(X),
        support,
        convex_parameterization=convex,
        loo=False,
        rho_prior=RefNormalPrior(0.0, 1.0) if rho_prior else None,
    )
    reference = RefGP(X, Y, likelihood=reference_lik, outcome_transform=None)
    lik = SparseOutlierGaussianLikelihood(
        get_gaussian_likelihood_with_lognormal_prior(),
        len(X),
        support,
        convex_parameterization=convex,
        loo=False,
        rho_prior=NormalPrior(0.0, 1.0) if rho_prior else None,
    )
    convert = {"torch": lambda x: x, "numpy": lambda x: x.numpy(), "list": lambda x: x.tolist()}[
        kind
    ]
    model = SingleTaskGP(convert(X), convert(Y), likelihood=lik, outcome_transform=None)
    with torch.no_grad():
        reference.likelihood.noise_covar.raw_rho.fill_(0.2)
        reference.likelihood.noise_covar.base_noise.noise = 0.05
        reference.mean_module.constant = 0.15
        reference.covar_module.lengthscale = 0.3
    for name, parameter in reference.named_parameters():
        model._set_raw_param(name, parameter)
    return model, reference


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_support_resize_preserves_other_leaves_and_sparse_trainability(dtype):
    likelihood = SparseOutlierGaussianLikelihood(
        get_gaussian_likelihood_with_lognormal_prior(), len(X), [], loo=False
    )
    model = SingleTaskGP(X.to(dtype), Y.to(dtype), likelihood=likelihood, outcome_transform=None)
    noise = model.likelihood.noise_covar
    before = dict(model.named_parameters())
    rho_name = "likelihood.noise_covar.raw_rho"
    before[rho_name].requires_grad_(False)
    mean = model.mean_module.raw_constant
    with torch.no_grad():
        mean.fill_(0.37)
    noise.expand_support([2, 0])
    rho = noise.raw_rho
    assert isinstance(rho, torch.nn.Parameter) and rho.shape == (2,)
    assert rho is not before[rho_name] and not rho.requires_grad
    with torch.no_grad():
        rho.fill_(0.2)
    noise.expand_support([4])
    torch.testing.assert_close(noise.raw_rho, torch.tensor([0.2, 0.2, 0.0], dtype=dtype))
    noise.contract_support([0])
    torch.testing.assert_close(noise.raw_rho, torch.tensor([0.2, 0.0], dtype=dtype))
    assert not noise.raw_rho.requires_grad
    after = dict(model.named_parameters())
    for name, leaf in before.items():
        if name != rho_name:
            assert leaf is after[name]
    torch.testing.assert_close(mean, torch.full_like(mean, 0.37), rtol=0, atol=0)


@pytest.mark.parametrize("convex", [False, True])
@pytest.mark.parametrize("kind", ["torch", "numpy", "list"])
def test_mll_all_gradients_and_candidate_posterior(convex, kind):
    model, reference = pair(convex, kind)
    actual, gradient = ExactMarginalLogLikelihood(model.likelihood, model).value_and_grad()
    reference.train()
    likelihood = reference.likelihood
    expected = RefMLL(likelihood, reference)(reference(X), reference.train_targets, X)
    expected.backward()
    gradients = torch.cat([p.grad.reshape(-1) for p in reference.parameters()])
    np.testing.assert_allclose(actual, expected.detach(), atol=2e-10, rtol=2e-10)
    np.testing.assert_allclose(gradient, gradients, atol=3e-9, rtol=3e-9)
    reference.eval()
    candidates = torch.tensor([[0.1], [0.6], [0.9]], dtype=torch.float64)
    expected_p = reference.posterior(candidates)
    actual_p = model.posterior(candidates)
    torch.testing.assert_close(actual_p.mean, expected_p.mean, atol=2e-9, rtol=2e-9)
    torch.testing.assert_close(
        actual_p.distribution.covariance_matrix,
        expected_p.distribution.covariance_matrix,
        atol=2e-9,
        rtol=2e-9,
    )


@pytest.mark.parametrize("convex", [False, True])
def test_rho_prior_includes_inactive_zeros_and_exact_chain_rule(convex):
    model, reference = pair(convex, rho_prior=True)
    value, gradient = ExactMarginalLogLikelihood(model.likelihood, model).value_and_grad()
    reference.train()
    expected = RefMLL(reference.likelihood, reference)(reference(X), reference.train_targets, X)
    expected.backward()
    expected_gradient = torch.cat([p.grad.reshape(-1) for p in reference.parameters()])
    np.testing.assert_allclose(value, expected.detach(), atol=2e-9, rtol=2e-9)
    np.testing.assert_allclose(gradient, expected_gradient, atol=3e-9, rtol=3e-9)


def test_nondefault_rho_constraint_initial_value_and_native_fit_bounds():
    likelihood = SparseOutlierGaussianLikelihood(
        get_gaussian_likelihood_with_lognormal_prior(),
        5,
        [2],
        rho_constraint=NonTransformedInterval(0.0, 0.7, initial_value=0.3),
    )
    model = SingleTaskGP(X, Y, likelihood=likelihood, outcome_transform=None)
    torch.testing.assert_close(
        likelihood.noise_covar.raw_rho,
        torch.tensor([0.3], dtype=torch.float64),
        atol=2e-8,
        rtol=2e-8,
    )
    mll = ExactMarginalLogLikelihood(likelihood, model)
    fit_gpytorch_mll(mll, optimizer_kwargs={"options": {"maxiter": 20}}, max_attempts=1)
    assert (
        0.0
        <= float(likelihood.noise_covar.raw_rho[0])
        <= likelihood.noise_covar.raw_rho_constraint.upper_bound
    )
    with pytest.raises(ValueError, match="constraint"):
        likelihood.noise_covar.raw_rho = torch.tensor([0.8], dtype=torch.float64)


def test_support_representation_copy_and_atomic_validation():
    model, _ = pair()
    noise = model.likelihood.noise_covar
    snapshot = copy.deepcopy(model)
    noise.to_dense()
    assert dict(model._rs.parameter_shapes())["likelihood.noise_covar.raw_rho"] == [5]
    torch.testing.assert_close(
        noise.raw_rho, torch.tensor([0.2, 0.0, 0.2, 0.0, 0.0], dtype=torch.float64)
    )
    noise.contract_support([0])
    noise.expand_support([4])
    noise.to_sparse()
    assert noise.support == [2, 4]
    torch.testing.assert_close(noise.raw_rho, torch.tensor([0.2, 0.0], dtype=torch.float64))
    assert snapshot.likelihood.noise_covar.support == [2, 0]
    for indices in ([4], [-1], [5], [1, 1]):
        with pytest.raises(ValueError):
            noise.expand_support(indices)
        assert noise.support == [2, 4]
    noise.full_support()
    assert len(noise.support) == len(X) and not noise.is_sparse
    noise.remove_support().to_sparse()
    assert noise.support == [] and noise.raw_rho.numel() == 0


@pytest.mark.parametrize("fixed", [False, True])
@pytest.mark.parametrize("convex", [False, True])
def test_observation_noise_matches_training_inputs_and_fixed_noise_shape(fixed, convex):
    variance = torch.full_like(Y, 0.05) if fixed else None
    model = RobustRelevancePursuitSingleTaskGP(
        X, Y, train_Yvar=variance, outcome_transform=None, convex_parameterization=convex
    )
    reference = RefRobustGP(
        X, Y, train_Yvar=variance, outcome_transform=None, convex_parameterization=convex
    )
    model.likelihood.noise_covar.expand_support([2])
    reference.likelihood.noise_covar.expand_support([2])
    with torch.no_grad():
        reference.likelihood.noise_covar.raw_rho.fill_(0.2)
    for name, parameter in reference.named_parameters():
        model._set_raw_param(name, parameter)
    reference.eval()
    for points in (X, X + 0.01, X[:3] + 0.01):
        actual = model.posterior(points, observation_noise=True)
        if fixed and len(points) != len(X):
            # FixedGaussianNoise returns a scalar ZeroLinearOperator for a
            # mismatched event size. Upstream indexes its absent final shape
            # and crashes; the correct added noise is zero, so compare to the
            # actual reference latent posterior instead of skipping this case.
            with pytest.raises(IndexError):
                reference.posterior(points, observation_noise=True)
            expected = reference.posterior(points, observation_noise=False)
        else:
            expected = reference.posterior(points, observation_noise=True)
        torch.testing.assert_close(actual.mean, expected.mean, atol=3e-9, rtol=3e-9)
        torch.testing.assert_close(
            actual.distribution.covariance_matrix,
            expected.distribution.covariance_matrix,
            atol=3e-9,
            rtol=3e-9,
        )


def test_standalone_noise_and_marginal():
    model, reference = pair()
    a = model.likelihood.noise_covar
    b = reference.likelihood.noise_covar
    diagonal = torch.linspace(0.8, 1.2, 5, dtype=torch.float64)
    expected = b(X=X, shape=torch.Size([5]), diag_K=diagonal).to_dense()
    actual = a(X=X, shape=(5,), diag_K=diagonal).to_dense()
    torch.testing.assert_close(actual, expected)
    a.eval()
    with pytest.warns(InputDataWarning, match="not applied"):
        outside = a(X=X + 0.01, shape=(5,), diag_K=diagonal).diagonal()
    torch.testing.assert_close(outside, torch.full((5,), 0.05, dtype=torch.float64))
    # Changing caller-owned inputs must not silently change the cached training inputs.
    inputs = X.clone()
    a.train()(X=inputs, shape=(5,), diag_K=diagonal)
    inputs.add_(0.01)
    a.eval()
    with pytest.warns(InputDataWarning, match="not applied"):
        a(X=inputs, shape=(5,), diag_K=diagonal)
    from gpytorch.distributions import MultivariateNormal

    from botorch_rs._arrays import kind_of, to_flat
    from botorch_rs.posteriors.gpytorch import _MomentPosterior

    covariance = torch.eye(5, dtype=torch.float64) * 1.2
    mean = torch.zeros(5, dtype=torch.float64)
    native_distribution = _MomentPosterior(
        to_flat(mean), to_flat(covariance), kind_of(mean)
    ).distribution
    a.train()
    expected_marginal = reference.likelihood.marginal(MultivariateNormal(mean, covariance), X=X)
    actual_marginal = model.likelihood.marginal(native_distribution, X=X)
    torch.testing.assert_close(actual_marginal.mean, expected_marginal.mean)
    torch.testing.assert_close(
        actual_marginal.covariance_matrix, expected_marginal.covariance_matrix
    )


@pytest.mark.parametrize("convex", [False, True])
def test_bounded_fit_matches_reference_fixed_support(convex):
    model, reference = pair(convex, support=[2])
    options = {"maxiter": 45, "ftol": 1e-10, "gtol": 1e-8}
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_mll(mll, optimizer_kwargs={"options": options}, max_attempts=1)
    ref_fit(
        RefMLL(reference.likelihood, reference),
        optimizer_kwargs={"options": options},
        max_attempts=1,
    )
    expected = torch.cat([p.detach().reshape(-1) for p in reference.parameters()])
    np.testing.assert_allclose(model._rs.raw_parameters(), expected, atol=3e-6, rtol=3e-6)
    assert model.likelihood.noise_covar.support == [2]
    assert not mll.training


def test_loo_expansion_centers_targets_at_nonzero_mean():
    model, reference = pair(support=[])
    reference.likelihood.noise_covar.loo = True
    model.likelihood.noise_covar.loo = True
    reference.train()
    marginal = reference.likelihood(reference(X), X)
    inverse = marginal.covariance_matrix.inverse()
    residual = reference.train_targets - marginal.mean
    alpha = inverse @ residual
    variance = inverse.diagonal().reciprocal()
    corrected = ((alpha * variance).square() - variance).clamp_min(0)
    actual = model.likelihood.noise_covar.expansion_objective(
        ExactMarginalLogLikelihood(model.likelihood, model)
    )
    torch.testing.assert_close(actual, corrected, atol=2e-9, rtol=2e-9)
    buggy = reference.likelihood.noise_covar._optimal_rhos(RefMLL(reference.likelihood, reference))
    assert not torch.allclose(buggy, corrected)


@pytest.mark.parametrize("pursuit", [forward_relevance_pursuit, backward_relevance_pursuit])
def test_pursuit_snapshots_support_posterior_and_input_validation(pursuit):
    model, _ = pair(support=[])
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    noise, trace = pursuit(
        model.likelihood.noise_covar,
        mll,
        sparsity_levels=[0, 1, 3, 5],
        optimizer_kwargs={"options": {"maxiter": 25}},
    )
    assert trace and trace[-1] is not model
    supports = [len(SparseOutlierNoise._from_model(m).support) for m in trace]
    assert supports == sorted(supports, reverse=pursuit is backward_relevance_pursuit)
    assert len(set(id(m._rs) for m in trace)) == len(trace)
    sizes, probabilities = get_posterior_over_support(
        SparseOutlierNoise, trace, prior_mean_of_support=1.0
    )
    torch.testing.assert_close(sizes, torch.tensor(supports, dtype=torch.float64))
    torch.testing.assert_close(probabilities.sum(), torch.tensor(1.0, dtype=torch.float64))
    old = copy.deepcopy(trace[0]._rs.raw_parameters())
    noise.raw_rho = torch.zeros_like(noise.raw_rho)
    assert trace[0]._rs.raw_parameters() == old
    with pytest.raises(ValueError, match="nonempty"):
        pursuit(noise, mll, sparsity_levels=[])


@pytest.mark.parametrize("custom_prior", [False, True])
def test_support_posterior_matches_official_normalized_evidence(custom_prior):
    models, references = zip(*(pair(support=support) for support in ([], [2], [0, 2])), strict=True)
    kwargs = (
        {"log_support_prior": lambda size: -2.0 * size}
        if custom_prior
        else {"prior_mean_of_support": 1.2}
    )
    actual_sizes, actual = get_posterior_over_support(SparseOutlierNoise, list(models), **kwargs)
    expected_sizes, expected = ref_support_posterior(RefNoise, list(references), **kwargs)
    torch.testing.assert_close(actual_sizes, expected_sizes)
    torch.testing.assert_close(actual, expected, atol=2e-9, rtol=2e-9)


def test_robust_map_fit_fixed_noise_and_exception_restoration():
    model = RobustRelevancePursuitSingleTaskGP(
        X, Y, train_Yvar=torch.full_like(Y, 0.05), outcome_transform=None, cache_model_trace=True
    )
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_mll(
        mll, numbers_of_outliers=[0, 1, 3, 5], optimizer_kwargs={"options": {"maxiter": 25}}
    )
    assert mll.model is model and mll.likelihood is model.likelihood
    assert model.model_trace and len(model.bmc_probabilities) == len(model.model_trace)
    selected = int(model.bmc_probabilities.argmax())
    expected = model.model_trace[selected]
    np.testing.assert_allclose(model._rs.raw_parameters(), expected._rs.raw_parameters())
    assert model.likelihood._model is model
    assert model.likelihood.noise_covar._model is model
    assert model.likelihood.noise_covar.base_noise._model is model
    before = model._rs.raw_parameters()

    def fail(**kwargs):
        raise RuntimeError("intentional failure")

    with pytest.raises(RuntimeError, match="intentional"):
        fit_gpytorch_mll(mll, relevance_pursuit_optimizer=fail)
    assert mll.model is model and mll.likelihood is model.likelihood
    assert model._rs.raw_parameters() == before


def test_small_robust_default_prior_is_positive_and_invalid_priors_rejected():
    small = RobustRelevancePursuitSingleTaskGP(X[:3], Y[:3], outcome_transform=None)
    assert small.prior_mean_of_support == 1.0
    # The official default truncates 0.2*n to zero and then fails in log(prior_mean).
    reference = RefRobustGP(X[:3], Y[:3], outcome_transform=None)
    assert reference.prior_mean_of_support == 0
    with pytest.raises(ValueError, match="positive"):
        RobustRelevancePursuitSingleTaskGP(X, Y, prior_mean_of_support=0)


def test_single_observation_fixed_noise_is_not_broadcast_to_different_event_size():
    model = RobustRelevancePursuitSingleTaskGP(
        X[:1], Y[:1], train_Yvar=torch.full_like(Y[:1], 0.05), outcome_transform=None
    )
    points = torch.tensor([[0.2], [0.8]], dtype=torch.float64)
    latent = model.posterior(points, observation_noise=False)
    observed = model.posterior(points, observation_noise=True)
    torch.testing.assert_close(
        observed.distribution.covariance_matrix,
        latent.distribution.covariance_matrix,
        atol=0.0,
        rtol=0.0,
    )


def test_initialize_dense_parameters_and_explicit_boundaries():
    model, _ = pair()
    _, initial = initialize_dense_parameters(model)
    assert set(initial) == {name for name, _ in model.named_parameters()}
    assert all(np.isfinite(model._rs.raw_parameters()))
    with pytest.raises(NotImplementedError, match="variational"):
        model.likelihood.expected_log_prob(None)
    model.eval()
    with pytest.raises(NotImplementedError, match="support extension"):
        model.condition_on_observations(X[:1], Y[:1])
    with pytest.raises(NotImplementedError, match="batch_shape"):
        SparseOutlierNoise(get_gaussian_likelihood_with_lognormal_prior(), 5, batch_shape=(2,))


@pytest.mark.parametrize("batched", [False, True])
def test_atomic_transformed_target_and_fixed_noise_replacement(batched):
    x = X.expand(2, 5, 1).clone() if batched else X
    y = Y.expand(2, 5, 1).clone() if batched else Y
    model = SingleTaskGP(x, y, train_Yvar=torch.full_like(y, 0.05))
    reference = RefGP(x, y, train_Yvar=torch.full_like(y, 0.05))
    old_raw = model._rs.raw_parameters()
    candidates = torch.tensor([[0.3], [0.8]], dtype=torch.float64)
    _ = model.posterior(candidates).mean  # populate the cache before replacement
    targets = model.train_targets + 0.4
    noise = torch.full_like(targets, 0.07)
    model._replace_transformed_train_data(targets, noise=noise)
    reference.set_train_data(targets=targets, strict=True)
    reference.likelihood.noise_covar.noise = noise
    reference.eval()
    torch.testing.assert_close(model.train_targets, targets)
    torch.testing.assert_close(model.likelihood.noise, noise)
    assert model._rs.raw_parameters() == old_raw
    torch.testing.assert_close(
        model.posterior(candidates).mean, reference.posterior(candidates).mean, atol=2e-9, rtol=2e-9
    )
    with pytest.raises(ValueError, match="nonnegative"):
        model._replace_transformed_train_data(targets + 1, noise=-noise)
    torch.testing.assert_close(model.train_targets, targets)
    torch.testing.assert_close(model.likelihood.noise, noise)
    with pytest.raises(ValueError, match="shape"):
        model.set_train_data(targets=targets[..., :-1])
    model.likelihood.noise = noise * 2
    torch.testing.assert_close(model.likelihood.noise, noise * 2)

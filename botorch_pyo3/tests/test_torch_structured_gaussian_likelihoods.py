# ruff: noqa: E402
"""Structured models use actual observation modules and their full Gaussian law."""

from unittest.mock import patch

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("gpytorch")

from gpytorch.distributions import MultitaskMultivariateNormal, MultivariateNormal
from gpytorch.likelihoods import (
    FixedNoiseGaussianLikelihood,
    GaussianLikelihood,
    MultitaskGaussianLikelihood,
)
from linear_operator.operators import DenseLinearOperator

from botorch_rs.mlls import ExactMarginalLogLikelihood
from botorch_rs.models import LatentKroneckerGP

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


class CorrelatedGaussian(GaussianLikelihood):
    def __init__(self):
        super().__init__()
        self.register_parameter("correlation", torch.nn.Parameter(torch.tensor(0.3)))
        self.register_parameter("offset", torch.nn.Parameter(torch.tensor(0.07)))

    def marginal(self, distribution, *params, multiplier=1.0, **kwargs):
        base = super().marginal(distribution, *params, **kwargs)
        rows = torch.linspace(0.2, 1.0, base.mean.shape[-1], device=base.mean.device).to(base.mean)
        covariance = base.covariance_matrix + multiplier * self.correlation.square() * (
            rows[:, None] * rows[None, :]
        )
        return MultivariateNormal(base.mean + self.offset, DenseLinearOperator(covariance))


class CorrelatedMultitask(MultitaskGaussianLikelihood):
    def __init__(self):
        super().__init__(num_tasks=2, rank=1)
        self.register_parameter("correlation", torch.nn.Parameter(torch.tensor(0.3)))
        self.register_parameter("offset", torch.nn.Parameter(torch.tensor(0.07)))

    def _shaped_noise_covar(self, *args, **kwargs):
        shape = args[0]
        factor = self.task_noise_covar_factor
        task = factor @ factor.T + self.noise * torch.eye(
            2, dtype=factor.dtype, device=factor.device
        )
        return DenseLinearOperator(
            3 * torch.kron(torch.eye(shape[-2], dtype=task.dtype, device=task.device), task)
        )

    def marginal(self, distribution, *params, multiplier=1.0, **kwargs):
        # Keep this supplied module's own law fully differentiable. Calling
        # the installed lazy Sum/Root wrapper here would inherit the separate
        # generic-operator higher-VJP defect tracked by linear_operator#155.
        mean = distribution.mean
        covariance = distribution.lazy_covariance_matrix.to_dense()
        covariance = covariance + self._shaped_noise_covar(mean.shape).to_dense()
        count = mean.shape[-2] * 2
        rows = torch.linspace(0.2, 1.0, count, device=mean.device).to(mean)
        covariance = covariance + multiplier * self.correlation.square() * (
            rows[:, None] * rows[None, :]
        )
        return MultitaskMultivariateNormal(
            mean + self.offset,
            DenseLinearOperator(covariance),
            interleaved=distribution._interleaved,
        )


def make_model(family, device, likelihood=None, reference=False):
    X = torch.tensor([[0.1], [0.45], [0.9]], dtype=torch.double, device=device)
    T = X.new_tensor([[0.0], [0.6]])
    Y = (4 * X + T.squeeze(-1)).sin()
    X.requires_grad_()
    Y.requires_grad_()
    if likelihood is None:
        likelihood = CorrelatedMultitask() if family == "kron" else CorrelatedGaussian()
    likelihood = likelihood.to(X)
    if reference:
        from botorch.models import HigherOrderGP, KroneckerMultiTaskGP
        from botorch.models.latent_kronecker_gp import LatentKroneckerGP
    else:
        from botorch_rs.models import HigherOrderGP, KroneckerMultiTaskGP, LatentKroneckerGP
    options = {} if reference else {"backend": "torch"}
    cls = {"kron": KroneckerMultiTaskGP, "higher": HigherOrderGP, "latent": LatentKroneckerGP}[
        family
    ]
    args = (X, T, Y) if family == "latent" else (X, Y)
    model = cls(*args, likelihood=likelihood, outcome_transform=None, **options).to(X)
    assert model.likelihood is likelihood
    return model, X, Y


def flatten_prior(model, family):
    prior = model(*model.train_inputs)
    mean = prior.mean.flatten(-2) if family == "kron" else prior.mean
    covariance = prior.covariance_matrix
    if family == "kron" and not prior._interleaved:
        q, m = prior.mean.shape[-2:]
        index = torch.arange(q * m, device=mean.device).reshape(m, q).T.flatten()
        covariance = covariance.index_select(-2, index).index_select(-1, index)
    return prior, mean, covariance


def independent_observed(model, family, mean, covariance, noise=None, multiplier=1.0):
    likelihood = model.likelihood
    if family == "kron":
        factor = likelihood.task_noise_covar_factor
        task = factor @ factor.T + likelihood.noise * torch.eye(
            2, device=mean.device, dtype=mean.dtype
        )
        noise_covariance = 3 * torch.kron(
            torch.eye(mean.shape[-1] // 2, device=mean.device, dtype=mean.dtype), task
        )
    else:
        variances = likelihood.noise if noise is None else noise
        noise_covariance = torch.diag_embed(variances.expand_as(mean))
    rows = torch.linspace(0.2, 1.0, mean.shape[-1], device=mean.device).to(mean)
    noise_covariance = noise_covariance + multiplier * likelihood.correlation.square() * (
        rows[:, None] * rows[None, :]
    )
    return mean + likelihood.offset, covariance + noise_covariance


def gradients(value, leaves, create_graph=False):
    result = torch.autograd.grad(
        value.sum(), leaves, create_graph=create_graph, retain_graph=True, allow_unused=True
    )
    return tuple(
        torch.zeros_like(leaf) if grad is None else grad
        for leaf, grad in zip(leaves, result, strict=True)
    )


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("family", ["kron", "higher", "latent"])
@pytest.mark.parametrize("overrides", [False, True])
def test_actual_marginal_mll_reference_and_independent_all_leaf_hessians(device, family, overrides):
    model, X, Y = make_model(family, device)
    reference, _, _ = make_model(family, device, reference=True)
    actual_parameters, expected_parameters = (
        dict(model.named_parameters()),
        dict(reference.named_parameters()),
    )
    assert actual_parameters.keys() == expected_parameters.keys()
    with torch.no_grad():
        for name, parameter in actual_parameters.items():
            expected_parameters[name].copy_(parameter)
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    prior, mean, covariance = flatten_prior(model, family)
    noise = (
        mean.new_full(mean.shape, 0.13, requires_grad=True)
        if overrides and family != "kron"
        else None
    )
    options = {"multiplier": 1.7} if overrides else {}
    if noise is not None:
        options["noise"] = noise
    arguments = model.train_inputs if overrides else ()
    with (
        patch.object(torch.Tensor, "cpu", side_effect=AssertionError("host copy")),
        patch.object(torch.Tensor, "tolist", side_effect=AssertionError("host copy")),
    ):
        actual = mll(prior, model.train_targets, *arguments, **options)
        expected_mean, expected_covariance = independent_observed(
            model, family, mean, covariance, noise, 1.7 if overrides else 1.0
        )
        target = model.train_targets.flatten(-2) if family == "kron" else model.train_targets
        independent = (
            torch.distributions.MultivariateNormal(expected_mean, expected_covariance).log_prob(
                target
            )
            + model._log_prior()
        ) / target.shape[-1]
        torch.testing.assert_close(actual, independent, atol=3e-10, rtol=3e-10)
        leaves = (X, Y, *model.parameters(), *((noise,) if noise is not None else ()))
        first = gradients(actual, leaves, create_graph=True)
        control = gradients(independent, leaves, create_graph=True)
        for left, right in zip(first, control, strict=True):
            torch.testing.assert_close(left, right, atol=3e-8, rtol=3e-8)
        direction = tuple(torch.full_like(value, 0.023) for value in leaves)
        second = gradients(
            sum((g * d).sum() for g, d in zip(first, direction, strict=True)), leaves
        )
        control_second = gradients(
            sum((g * d).sum() for g, d in zip(control, direction, strict=True)), leaves
        )
        for left, right in zip(second, control_second, strict=True):
            torch.testing.assert_close(left, right, atol=2e-7, rtol=2e-7)
    reference.train()
    reference_prior = reference(*reference.train_inputs)
    observed = reference.likelihood(reference_prior, *arguments, **options)
    expected_data = observed.log_prob(reference.train_targets) / target.shape[-1]
    torch.testing.assert_close(
        actual - model._log_prior() / target.shape[-1], expected_data, atol=3e-8, rtol=3e-8
    )


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("family", ["higher", "latent"])
def test_fixed_noise_constructor_training_keeps_identity_and_noise_graph(device, family):
    noise = torch.linspace(0.02, 0.09, 6, dtype=torch.double, device=device).requires_grad_()
    likelihood = FixedNoiseGaussianLikelihood(noise=noise)
    model, X, Y = make_model(family, device, likelihood)
    _, mean, covariance = flatten_prior(model, family)
    target = model.train_targets
    expected = (
        torch.distributions.MultivariateNormal(mean, covariance + torch.diag_embed(noise)).log_prob(
            target
        )
        + model._log_prior()
    ) / target.shape[-1]
    actual = ExactMarginalLogLikelihood(likelihood, model)()
    torch.testing.assert_close(actual, expected, atol=2e-9, rtol=2e-9)
    leaves = (X, Y, noise, *model.parameters())
    first = gradients(actual, leaves, True)
    control = gradients(expected, leaves, True)
    for left, right in zip(first, control, strict=True):
        torch.testing.assert_close(left, right, atol=2e-8, rtol=2e-8)
    assert all(
        bool(torch.isfinite(value).all())
        for value in gradients(sum(g.square().sum() for g in first), leaves)
    )


def prior_blocks(model, family, Q):
    from botorch_rs.models._torch_backend_kronecker import tensor_kron

    X = model._train_X
    if family == "kron":
        return (
            model.mean_module(X).flatten(-2),
            model.mean_module(Q).flatten(-2),
            model.covar_module(X, X),
            model.covar_module(X, Q),
            model.covar_module(Q, Q),
        )
    if family == "higher":
        latent = model._latent_covariance()
        module = model.covar_modules[0]

        def kernel(a, b):
            return tensor_kron(model._kernel(module, a, b), latent)

        return (
            X.new_zeros(X.shape[-2] * model._grid_size),
            Q.new_zeros(Q.shape[-2] * model._grid_size),
            kernel(X, X),
            kernel(X, Q),
            kernel(Q, Q),
        )
    T, mask = model._train_T, model.mask_valid
    return (
        model._get_mean(X, T, mask),
        model._get_mean(Q, T),
        model._covariance(X, T, X, T)[..., mask, :][..., :, mask],
        model._covariance(X, T, Q, T)[..., mask, :],
        model._covariance(Q, T, Q, T),
    )


def independent_posterior(model, family, Q, observed=False):
    train_mean, query_mean, train, cross, test = prior_blocks(model, family, Q)
    train_mean, train = independent_observed(model, family, train_mean, train)
    if observed:
        query_mean, test = independent_observed(model, family, query_mean, test)
    target = model.train_targets.flatten(-2) if family == "kron" else model.train_targets
    alpha = torch.linalg.solve(train, (target - train_mean).unsqueeze(-1))
    mean = query_mean + (cross.T @ alpha).squeeze(-1)
    return mean, test - cross.T @ torch.linalg.solve(train, cross)


def point_major_posterior(posterior, family):
    covariance = posterior.distribution.covariance_matrix
    if family == "kron":
        q, m = posterior.mean.shape[-2:]
        index = torch.arange(q * m, device=posterior.device).reshape(m, q).T.flatten()
        covariance = covariance.index_select(-2, index).index_select(-1, index)
    return posterior.mean.flatten(-2), covariance


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize(
    "family,observed",
    [("kron", False), ("kron", True), ("higher", False), ("higher", True), ("latent", False)],
)
def test_correlated_observation_posterior_complete_law_all_leaf_gradients_and_samples(
    device, family, observed
):
    model, X, Y = make_model(family, device)
    Q = X.new_tensor([[0.2], [0.7]], requires_grad=True)
    leaves = (X, Y, Q, *model.parameters())
    for _ in range(2):
        posterior = model.posterior(Q, observation_noise=observed)
        actual_mean, actual_covariance = point_major_posterior(posterior, family)
        mean, covariance = independent_posterior(model, family, Q, observed)
        torch.testing.assert_close(actual_mean, mean, atol=2e-9, rtol=2e-9)
        torch.testing.assert_close(actual_covariance, covariance, atol=2e-9, rtol=2e-9)
        first = gradients(
            actual_mean.square().sum() + actual_covariance.square().sum(), leaves, True
        )
        expected = gradients(mean.square().sum() + covariance.square().sum(), leaves, True)
        for left, right in zip(first, expected, strict=True):
            torch.testing.assert_close(left, right, atol=1e-7, rtol=1e-7)
        second = gradients(sum(g.square().sum() for g in first), leaves)
        expected_second = gradients(sum(g.square().sum() for g in expected), leaves)
        for left, right in zip(second, expected_second, strict=True):
            torch.testing.assert_close(left, right, atol=2e-6, rtol=2e-6)
    size = posterior.base_sample_shape[-1]
    zeros = X.new_zeros(1, size)
    basis = torch.eye(size, dtype=X.dtype, device=X.device)
    with (
        patch.object(torch.Tensor, "cpu", side_effect=AssertionError("host copy")),
        patch.object(torch.Tensor, "tolist", side_effect=AssertionError("host copy")),
    ):
        center = posterior.rsample_from_base_samples((1,), zeros).flatten(-2).squeeze(0)
        samples = posterior.rsample_from_base_samples((size,), basis).flatten(-2)
        centered = samples - center
        actual_sample_covariance = centered.T @ centered
        torch.testing.assert_close(center, mean, atol=2e-9, rtol=2e-9)
        torch.testing.assert_close(actual_sample_covariance, covariance, atol=2e-8, rtol=2e-8)
        first = gradients(samples.square().sum(), leaves, True)
        second = gradients(sum(g.square().sum() for g in first), leaves)
        assert all(bool(torch.isfinite(value).all()) for value in second)
    old_mean, old_covariance, old_samples = (
        actual_mean.detach().clone(),
        actual_covariance.detach().clone(),
        samples.detach().clone(),
    )
    with torch.no_grad():
        for parameter in model.likelihood.parameters():
            parameter.add_(0.13)
    torch.testing.assert_close(point_major_posterior(posterior, family)[0], old_mean)
    torch.testing.assert_close(point_major_posterior(posterior, family)[1], old_covariance)
    torch.testing.assert_close(
        posterior.rsample_from_base_samples((size,), basis).flatten(-2), old_samples
    )


@pytest.mark.parametrize("device", DEVICES)
def test_latent_fixed_noise_public_boundary_and_direct_evaluation(device):
    from botorch_rs.exceptions.errors import UnsupportedError

    noise = torch.linspace(0.02, 0.09, 6, dtype=torch.double, device=device).requires_grad_()
    model, X, _ = make_model("latent", device, FixedNoiseGaussianLikelihood(noise=noise))
    model.eval()
    Q = X.new_tensor([[0.2], [0.7]], requires_grad=True)
    with pytest.raises(UnsupportedError, match="fixed-noise modules support training"):
        model.posterior(Q)
    direct = model(Q)
    train_mean, query_mean, train, cross, test = prior_blocks(model, "latent", Q)
    train = train + torch.diag_embed(noise)
    alpha = torch.linalg.solve(train, (model.train_targets - train_mean).unsqueeze(-1))
    expected = query_mean + (cross.T @ alpha).squeeze(-1)
    torch.testing.assert_close(direct.mean, expected, atol=2e-9, rtol=2e-9)
    assert torch.isfinite(torch.autograd.grad(direct.mean.sum(), noise)[0]).all()


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("rank", [0, 1])
def test_actual_scaled_lazy_shaped_noise_preserves_all_leaf_higher_gradients(device, rank):
    class ScaledNoise(MultitaskGaussianLikelihood):
        def _shaped_noise_covar(self, *args, **kwargs):
            return 3 * super()._shaped_noise_covar(*args, **kwargs)

    likelihood = ScaledNoise(num_tasks=2, rank=rank)
    model, X, Y = make_model("kron", device, likelihood)
    Q = X.new_tensor([[0.2], [0.7]], requires_grad=True)
    train_mean, query_mean, train, cross, test = prior_blocks(model, "kron", Q)
    if rank:
        factor = likelihood.task_noise_covar_factor
        task = factor @ factor.T
    else:
        task = torch.diag_embed(likelihood.task_noises)
    task = task + likelihood.noise * torch.eye(2, dtype=X.dtype, device=X.device)
    noise = 3 * torch.kron(torch.eye(3, dtype=X.dtype, device=X.device), task)
    observed = train + noise
    alpha = torch.linalg.solve(observed, (Y.flatten(-2) - train_mean).unsqueeze(-1))
    expected_mean = query_mean + (cross.T @ alpha).squeeze(-1)
    expected_covariance = test - cross.T @ torch.linalg.solve(observed, cross)
    posterior = model.posterior(Q)
    actual_mean, actual_covariance = point_major_posterior(posterior, "kron")
    actual_loss = actual_mean.square().sum() + actual_covariance.square().sum()
    expected_loss = expected_mean.square().sum() + expected_covariance.square().sum()
    expected_mll = (
        torch.distributions.MultivariateNormal(train_mean, observed).log_prob(Y.flatten(-2))
        + model._log_prior()
    ) / Y.numel()
    actual_mll = model._exact_mll()
    torch.testing.assert_close(actual_mll, expected_mll, atol=2e-9, rtol=2e-9)
    leaves = (X, Y, Q, *model.parameters())
    first = gradients(actual_loss + actual_mll, leaves, True)
    expected = gradients(expected_loss + expected_mll, leaves, True)
    for left, right in zip(first, expected, strict=True):
        torch.testing.assert_close(left, right, atol=3e-8, rtol=3e-8)
    second = gradients(sum(g.square().sum() for g in first), leaves)
    expected_second = gradients(sum(g.square().sum() for g in expected), leaves)
    for left, right in zip(second, expected_second, strict=True):
        torch.testing.assert_close(left, right, atol=5e-7, rtol=5e-7)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("query_hook", [False, True])
def test_latent_virtual_training_prior_and_separate_query_hook_match_source(device, query_hook):
    from botorch.models.latent_kronecker_gp import LatentKroneckerGP as Reference
    from gpytorch.settings import detach_test_caches, fast_pred_var

    class PriorOverride:
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.register_parameter(
                "shift", torch.nn.Parameter(self.train_inputs[0].new_tensor(0.35))
            )

        def forward(self, *args, **kwargs):
            value = super().forward(*args, **kwargs)
            return MultivariateNormal(
                value.mean + self.shift,
                DenseLinearOperator(value.covariance_matrix * (1 + self.shift.square())),
            )

        def _get_test_prior_mean_and_covariances(self, *args, **kwargs):
            mean, covariance, cross, *metadata = super()._get_test_prior_mean_and_covariances(
                *args, **kwargs
            )
            if query_hook:
                mean = mean + 0.4 * self.shift
                covariance = covariance * (1 + 2 * self.shift.square())
                cross = cross * (1 + 0.1 * self.shift.square())
            return mean, covariance, cross, *metadata

    class Actual(PriorOverride, LatentKroneckerGP):
        pass

    class Expected(PriorOverride, Reference):
        pass

    X = torch.tensor([[0.1], [0.45], [0.9]], dtype=torch.double, device=device, requires_grad=True)
    T = X.new_tensor([[0.0], [0.6]])
    Y = (4 * X + T.squeeze(-1)).sin().detach().requires_grad_()
    actual = Actual(X, T, Y, outcome_transform=None, backend="torch")
    expected = Expected(X, T, Y, outcome_transform=None).to(X)
    aa, bb = dict(actual.named_parameters()), dict(expected.named_parameters())
    assert aa.keys() == bb.keys()
    with torch.no_grad():
        for name in aa:
            bb[name].copy_(aa[name])
    implicit = actual._exact_mll()
    explicit = actual._exact_mll(actual(*actual.train_inputs), actual.train_targets)
    torch.testing.assert_close(implicit, explicit, atol=2e-10, rtol=2e-10)
    Q = X.new_tensor([[0.2], [0.7]], requires_grad=True)
    actual.eval()
    expected.eval()
    posterior = actual.posterior(Q)
    result = actual(Q)
    with detach_test_caches(False), fast_pred_var(False):
        reference = expected(Q)
        torch.testing.assert_close(result.mean, reference.mean, atol=2e-9, rtol=2e-9)
        torch.testing.assert_close(
            result.covariance_matrix, reference.covariance_matrix, atol=2e-9, rtol=2e-9
        )
        first = gradients(
            result.mean.square().sum() + result.covariance_matrix.square().sum(),
            (Q, actual.shift),
            True,
        )
        source_first = gradients(
            reference.mean.square().sum() + reference.covariance_matrix.square().sum(),
            (Q, expected.shift),
        )
        for left, right in zip(first, source_first, strict=True):
            torch.testing.assert_close(left, right, atol=2e-8, rtol=2e-8)
    base_mean, query_mean, train, cross, test = prior_blocks(actual, "latent", Q)
    train = train * (1 + actual.shift.square()) + torch.diag_embed(
        actual.likelihood.noise.expand_as(base_mean)
    )
    base_mean = base_mean + actual.shift
    if query_hook:
        query_mean = query_mean + 0.4 * actual.shift
        cross = cross * (1 + 0.1 * actual.shift.square())
        test = test * (1 + 2 * actual.shift.square())
    expected_mean = query_mean + (
        cross.T @ torch.linalg.solve(train, (actual.train_targets - base_mean).unsqueeze(-1))
    ).squeeze(-1)
    expected_covariance = test - cross.T @ torch.linalg.solve(train, cross)
    leaves = (X, Y, Q, *actual.parameters())
    first = gradients(
        result.mean.square().sum() + result.covariance_matrix.square().sum(), leaves, True
    )
    control = gradients(
        expected_mean.square().sum() + expected_covariance.square().sum(), leaves, True
    )
    second = gradients(sum(g.square().sum() for g in first), leaves)
    control_second = gradients(sum(g.square().sum() for g in control), leaves)
    for left, right in zip(second, control_second, strict=True):
        torch.testing.assert_close(left, right, atol=2e-7, rtol=2e-7)
    size = posterior.base_sample_shape[-1]
    samples = posterior.rsample_from_base_samples(
        (size,), torch.eye(size, dtype=X.dtype, device=X.device)
    ).flatten(-2)
    center = posterior.mean.flatten(-2)
    torch.testing.assert_close(
        (samples - center).T @ (samples - center), expected_covariance, atol=2e-8, rtol=2e-8
    )

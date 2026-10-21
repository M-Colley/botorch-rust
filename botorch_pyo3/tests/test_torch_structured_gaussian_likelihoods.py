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

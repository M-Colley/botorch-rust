"""Probit ELBOs use exact differentiable log-CDF and device quadrature."""

# ruff: noqa: E402
import functools

import pytest

torch = pytest.importorskip("torch")
gpytorch = pytest.importorskip("gpytorch")
pytest.importorskip("botorch")

import gpytorch.likelihoods.bernoulli_likelihood as source_probit
from botorch.models import SingleTaskVariationalGP as ReferenceGP
from gpytorch.mlls import VariationalELBO as ReferenceELBO

from botorch_rs import variational
from botorch_rs.kernels import RBFKernel
from botorch_rs.likelihoods import BernoulliLikelihood
from botorch_rs.mlls import VariationalELBO
from botorch_rs.models.approximate_gp import SingleTaskVariationalGP
from botorch_rs.optim.fit import fit_gpytorch_mll_torch

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
FAMILIES = [
    "CholeskyVariationalDistribution",
    "MeanFieldVariationalDistribution",
    "DeltaVariationalDistribution",
]


def _pair(
    device, family="CholeskyVariationalDistribution", unwhitened=False, batch=False, own_spec=False
):
    pairs = []
    for reference in (False, True):
        x = torch.linspace(0.1, 0.9, 10, device=device, dtype=torch.double).reshape(5, 2)
        y = x.new_tensor([0.0, 1.0, 0.0, 1.0, 1.0]).unsqueeze(-1)
        if batch:
            x, y = x.expand(2, 5, 2).clone(), y.expand(2, 5, 1).clone()
            x[1] += 0.02
        x.requires_grad_()
        y.requires_grad_()
        batch_shape = x.shape[:-2]
        module = gpytorch.variational if reference else variational
        distribution = getattr(module, family)(
            3, batch_shape=torch.Size(batch_shape), mean_init_std=0
        )
        strategy = getattr(
            module, "UnwhitenedVariationalStrategy" if unwhitened else "VariationalStrategy"
        )
        likelihood = (
            BernoulliLikelihood()
            if own_spec and not reference
            else gpytorch.likelihoods.BernoulliLikelihood()
        )
        cls, kernel_class = (
            (ReferenceGP, gpytorch.kernels.RBFKernel)
            if reference
            else (SingleTaskVariationalGP, RBFKernel)
        )
        model = cls(
            x,
            y,
            likelihood=likelihood,
            covar_module=kernel_class(ard_num_dims=2, batch_shape=torch.Size(batch_shape)),
            inducing_points=x.new_tensor([[0.12, 0.2], [0.44, 0.51], [0.84, 0.92]]),
            variational_distribution=distribution,
            variational_strategy=strategy,
            outcome_transform=None,
        ).to(x)
        with torch.no_grad():
            model.model.covar_module.lengthscale = x.new_tensor([0.4, 0.7])
            model.model.mean_module.constant = x.new_tensor(-0.25)
            strategy_instance = model.model.variational_strategy
            strategy_instance.variational_params_initialized.fill_(1)
            q = strategy_instance._variational_distribution
            q.variational_mean.copy_(x.new_tensor([-0.3, 0.15, 0.5]))
            if family.startswith("MeanField"):
                q._variational_stddev.copy_(x.new_tensor([-0.7, 0.9, 1.2]))
            elif family.startswith("Cholesky"):
                q.chol_variational_covar.copy_(0.8 * torch.eye(3, device=x.device, dtype=x.dtype))
        pairs.append((model, x, y))
    return pairs


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("unwhitened", [False, True])
@pytest.mark.parametrize("batch", [False, True])
def test_probit_q_elbo_first_graphs_and_observed_marginal_match_corrected_reference(
    device, family, unwhitened, batch, monkeypatch
):
    # The independent reference likelihood's LogNormalCDF caches a detached
    # negative-tail rational backward and approximates its primal (e.g. -2 has
    # 4.8e-5 log error and a zero Hessian). Repair only that scalar primitive;
    # retain its GP/strategy/quadrature/likelihood/MLL implementations.
    monkeypatch.setattr(source_probit, "log_normal_cdf", torch.special.log_ndtr)
    pairs = _pair(device, family, unwhitened, batch)
    terms, gradients = [], []
    for index, (model, x, y) in enumerate(pairs):
        cls = ReferenceELBO if index else VariationalELBO
        mll = cls(model.likelihood, model.model, 11, beta=0.7, combine_terms=False)
        output = model.model(x)
        values = mll(output, y.squeeze(-1))
        terms.append(values)
        parameters = dict(model.named_parameters())
        leaves = tuple(parameters[name] for name in sorted(parameters)) + (x, y)
        gradients.append(
            torch.autograd.grad(sum(v.sum() for v in values), leaves, create_graph=True)
        )
    for a, b in zip(*terms, strict=True):
        torch.testing.assert_close(a, b, atol=3e-9, rtol=3e-9)
    for a, b in zip(*gradients, strict=True):
        torch.testing.assert_close(a, b, atol=3e-8, rtol=3e-8)
    for observed in (False, True):
        posteriors = [
            model.posterior(x[..., :2, :], observation_noise=observed) for model, x, _ in pairs
        ]
        torch.testing.assert_close(posteriors[0].mean, posteriors[1].mean, atol=2e-10, rtol=2e-10)
        torch.testing.assert_close(
            posteriors[0].variance, posteriors[1].variance, atol=2e-10, rtol=2e-10
        )
        if observed:
            assert not posteriors[0].distribution.has_rsample
            with pytest.raises(NotImplementedError):
                posteriors[0].rsample(torch.Size([2]))


@pytest.mark.parametrize("device", DEVICES)
def test_port_probit_spec_quadrature_snapshots_and_probability_higher_graphs(device, monkeypatch):
    monkeypatch.setattr(source_probit, "log_normal_cdf", torch.special.log_ndtr)
    (model, x, y), (reference, _, _) = _pair(device, own_spec=True)
    mll = VariationalELBO(model.likelihood, model.model, 5)
    reference_mll = ReferenceELBO(reference.likelihood, reference.model, 5)
    output = model.model(x)
    loss = mll(output, y.squeeze(-1))
    expected = reference_mll(reference.model(x), y.squeeze(-1))
    torch.testing.assert_close(loss, expected, atol=3e-9, rtol=3e-9)
    # This posterior snapshot retains its quadrature law after public buffer edits.
    model.likelihood.quadrature.weights.mul_(1.5)
    torch.testing.assert_close(mll(output, y.squeeze(-1)), loss, atol=0, rtol=0)
    leaves = tuple(model.parameters()) + (x, y)
    first = torch.autograd.grad(loss, leaves, create_graph=True)
    second = torch.autograd.grad(sum(g.square().sum() for g in first), leaves)
    assert all(bool(torch.isfinite(v).all()) for v in (*first, *second))
    logits = x.new_tensor([-8.0, -2.0, -0.3, 0.7, 4.0], requires_grad=True)
    likelihood = model.likelihood
    probabilities = likelihood(logits).probs
    expected = 0.5 * torch.erfc(-logits / 2**0.5)
    torch.testing.assert_close(probabilities, expected, atol=1e-16, rtol=1e-14)
    gradient = torch.autograd.grad(probabilities.sum(), logits, create_graph=True)[0]
    hessian = torch.autograd.grad(gradient.sum(), logits)[0]
    density = (-0.5 * logits.square()).exp() / (2 * torch.pi) ** 0.5
    torch.testing.assert_close(gradient, density, atol=1e-16, rtol=1e-14)
    torch.testing.assert_close(hessian, -logits * density, atol=1e-16, rtol=1e-14)


@pytest.mark.parametrize("device", DEVICES)
def test_external_gaussian_likelihood_auto_selection_retains_noise_leaves(device):
    x = torch.linspace(0.1, 0.9, 10, device=device, dtype=torch.double).reshape(5, 2)
    y = x[:, :1].sin()
    source = gpytorch.likelihoods.GaussianLikelihood().to(x)
    source.noise = x.new_tensor(0.08)
    source.noise_covar.raw_noise.requires_grad_(False)
    model = SingleTaskVariationalGP(
        x, y, likelihood=source, inducing_points=x[:3], outcome_transform=None
    )
    assert model._tensor_backend == "torch"
    assert model.likelihood.noise_covar.raw_noise is source.noise_covar.raw_noise
    assert not model.likelihood.noise_covar.raw_noise.requires_grad
    mll = VariationalELBO(model.likelihood, model.model, len(x))
    optimizer = torch.optim.Adam(mll.parameters(), lr=0.01)
    noise = source.noise.clone()
    for _ in range(3):
        optimizer.zero_grad()
        (-mll().sum()).backward()
        optimizer.step()
    torch.testing.assert_close(source.noise, noise, atol=0, rtol=0)
    assert torch.isfinite(mll()).all()


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("unwhitened", [False, True])
def test_complete_probit_hessians_match_independent_reference_gradient_differences(
    device, family, unwhitened, monkeypatch
):
    monkeypatch.setattr(source_probit, "log_normal_cdf", torch.special.log_ndtr)
    (model, x, y), _ = _pair(device, family, unwhitened)
    parameters = dict(model.named_parameters())
    variables = [*parameters.values(), x, y]
    directions = [
        torch.linspace(-0.3, 0.5, value.numel(), device=value.device, dtype=value.dtype).reshape_as(
            value
        )
        * (0.02 + 0.01 * i)
        for i, value in enumerate(variables)
    ]
    value = VariationalELBO(model.likelihood, model.model, 5)()
    gradients = torch.autograd.grad(value, variables, create_graph=True)
    hvp = torch.autograd.grad(
        sum(
            (gradient * direction).sum()
            for gradient, direction in zip(gradients, directions, strict=True)
        ),
        variables,
    )
    rows = []
    step = 1e-5
    for sign in (-1, 1):
        _, (reference, _, _) = _pair(device, family, unwhitened)
        ref_parameters = dict(reference.named_parameters())
        assert parameters.keys() == ref_parameters.keys()
        with torch.no_grad():
            for (name, parameter), direction in zip(
                parameters.items(), directions[:-2], strict=True
            ):
                ref_parameters[name].copy_(parameter + sign * step * direction)
        query = (x.detach() + sign * step * directions[-2]).requires_grad_()
        labels = (y.detach() + sign * step * directions[-1]).requires_grad_()
        loss = ReferenceELBO(reference.likelihood, reference.model, 5)(
            reference.model(query), labels.squeeze(-1)
        )
        rows.append(
            torch.autograd.grad(
                loss, [*(ref_parameters[name] for name in parameters), query, labels]
            )
        )
    for actual, before, after in zip(hvp, rows[0], rows[1], strict=True):
        torch.testing.assert_close(actual, (after - before) / (2 * step), atol=3e-6, rtol=3e-5)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("unwhitened", [False, True])
def test_public_probit_adam_fitter_matches_actual_reference_steps(device, unwhitened, monkeypatch):
    monkeypatch.setattr(source_probit, "log_normal_cdf", torch.special.log_ndtr)
    (model, _, _), (reference, rx, ry) = _pair(device, unwhitened=unwhitened)
    mll = VariationalELBO(model.likelihood, model.model, 5)
    expected_mll = ReferenceELBO(reference.likelihood, reference.model, 5)
    fit_gpytorch_mll_torch(
        mll, step_limit=4, optimizer=functools.partial(torch.optim.Adam, lr=0.008)
    )
    optimizer = torch.optim.Adam(expected_mll.parameters(), lr=0.008)
    for _ in range(4):
        optimizer.zero_grad()
        (-expected_mll(reference.model(rx), ry.squeeze(-1)).sum()).backward()
        optimizer.step()
    expected = dict(reference.named_parameters())
    for name, parameter in model.named_parameters():
        torch.testing.assert_close(parameter, expected[name], atol=3e-8, rtol=3e-8)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("unwhitened", [False, True])
def test_float32_probit_execution_and_higher_graphs(device, unwhitened, monkeypatch):
    monkeypatch.setattr(source_probit, "log_normal_cdf", torch.special.log_ndtr)
    pairs = _pair(device, "MeanFieldVariationalDistribution", unwhitened)
    values = []
    for index, (model, x, y) in enumerate(pairs):
        model.to(dtype=torch.float32)
        inputs = x.detach().float().requires_grad_()
        labels = y.detach().float().requires_grad_()
        cls = ReferenceELBO if index else VariationalELBO
        mll = cls(model.likelihood, model.model, 5)
        loss = mll(model.model(inputs), labels.squeeze(-1)).sum()
        values.append(loss)
        if not index:
            leaves = [*model.parameters(), inputs, labels]
            first = torch.autograd.grad(loss, leaves, create_graph=True)
            second = torch.autograd.grad(sum(g.square().sum() for g in first), leaves)
            assert all(bool(torch.isfinite(g).all()) for g in (*first, *second))
    torch.testing.assert_close(*values, atol=3e-4, rtol=3e-4)

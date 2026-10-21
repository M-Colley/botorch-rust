"""Unwhitened inducing variables retain their actual GP-coordinate law."""

# ruff: noqa: E402
import pytest

torch = pytest.importorskip("torch")
gpytorch = pytest.importorskip("gpytorch")
pytest.importorskip("botorch")

from botorch.sampling.pathwise import gaussian_update as reference_update
from gpytorch.mlls import VariationalELBO as ReferenceELBO
from test_torch_variational_distributions import DEVICES, _model

from botorch_rs.mlls import VariationalELBO
from botorch_rs.sampling.pathwise import gaussian_update

FAMILIES = [
    "CholeskyVariationalDistribution",
    "MeanFieldVariationalDistribution",
    "DeltaVariationalDistribution",
]


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("outputs,batch", [(1, False), (1, True), (2, True)])
@pytest.mark.parametrize("training", [False, True])
def test_unwhitened_conditionals_elbo_and_complete_first_graphs_match_reference(
    device, family, outputs, batch, training
):
    pairs = [
        _model(
            device,
            family,
            reference=index,
            outputs=outputs,
            batch=batch,
            graph=True,
            strategy="unwhitened",
        )
        for index in (False, True)
    ]
    values, gradients = [], []
    for index, (model, _, x, y) in enumerate(pairs):
        model.train(training)
        inputs = x.unsqueeze(-3) if index and outputs > 1 and batch else x
        distribution = model.model(inputs)
        cls = ReferenceELBO if index else VariationalELBO
        mll = cls(model.likelihood, model.model, 11, beta=0.7, combine_terms=False)
        terms = mll(distribution, y.squeeze(-1) if outputs == 1 else y)
        values.append(
            (distribution.mean, distribution.variance, distribution.covariance_matrix, *terms)
        )
        parameters = dict(model.named_parameters())
        leaves = tuple(parameters[name] for name in sorted(parameters)) + (x, y)
        gradients.append(torch.autograd.grad(sum(term.sum() for term in terms), leaves))
    for a, b in zip(*values, strict=True):
        torch.testing.assert_close(a, b, atol=3e-8, rtol=3e-8)
    for a, b in zip(*gradients, strict=True):
        torch.testing.assert_close(a, b, atol=2e-6, rtol=2e-6)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("training", [False, True])
def test_unwhitened_full_hessians_match_independent_gaussian_conditioning_formula(
    device, family, training
):
    model, _, x, y = _model(device, family, graph=True, strategy="unwhitened")
    model.train(training)
    mll = VariationalELBO(model.likelihood, model.model, len(x))
    actual = -mll().sum()
    strategy = model._strategy
    z, q = strategy.inducing_points, strategy._variational_distribution
    lengthscale = model.model.covar_module.lengthscale
    constant, noise = model.model.mean_module.constant, model.likelihood.noise

    def kernel(a, b):
        difference = (a.unsqueeze(-2) - b.unsqueeze(-3)) / lengthscale.unsqueeze(-2)
        return (-0.5 * difference.square().sum(-1)).exp()

    identity = torch.eye(3, dtype=x.dtype, device=x.device)
    kzz = kernel(z, z)
    root = torch.linalg.cholesky(kzz + 1e-6 * identity)
    cross = kernel(z, x)
    interpolation = torch.cholesky_solve(cross, root)
    covariance_q = (
        torch.diag_embed(q._variational_stddev.square())
        if family.startswith("MeanField")
        else torch.zeros_like(identity)
        if family.startswith("Delta")
        else q.chol_variational_covar.tril() @ q.chol_variational_covar.tril().mT
    )
    mean = constant + (interpolation.mT @ (q.variational_mean - constant).unsqueeze(-1)).squeeze(-1)
    residual = 1 - (cross * interpolation).sum(-2)
    variance = (residual.clamp_min(0) if training else residual) + (
        interpolation * (covariance_q @ interpolation)
    ).sum(-2)
    prior_root = torch.linalg.cholesky(kzz + (1e-6 if training else 1e-3) * identity)
    delta = q.variational_mean - constant
    quadratic = (delta * torch.cholesky_solve(delta.unsqueeze(-1), prior_root).squeeze(-1)).sum()
    logdet = 2 * prior_root.diagonal().log().sum()
    kl = (
        0.5 * (quadratic + logdet + 3 * torch.log(x.new_tensor(2 * torch.pi)))
        if family.startswith("Delta")
        else 0.5
        * (
            quadratic
            + logdet
            - 3
            - torch.linalg.slogdet(covariance_q).logabsdet
            + torch.cholesky_solve(covariance_q, prior_root).diagonal().sum()
        )
    )
    expected = 0.5 * (
        ((y.squeeze(-1) - mean).square() + variance) / noise
        + noise.log()
        + torch.log(x.new_tensor(2 * torch.pi))
    ).mean() + kl / len(x)
    torch.testing.assert_close(actual, expected, atol=3e-12, rtol=3e-12)
    leaves = tuple(model.parameters()) + (x, y)
    gradients = [
        torch.autograd.grad(loss, leaves, create_graph=True) for loss in (actual, expected)
    ]
    for a, b in zip(*gradients, strict=True):
        torch.testing.assert_close(a, b, atol=3e-10, rtol=3e-10)
    seconds = [
        torch.autograd.grad(sum(g.square().sum() for g in gradient), leaves)
        for gradient in gradients
    ]
    for a, b in zip(*seconds, strict=True):
        torch.testing.assert_close(a, b, atol=3e-8, rtol=3e-8)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("family", FAMILIES)
def test_unwhitened_prior_initialization_exact_inducing_case_and_source_update_boundary(
    device, family
):
    pairs = [
        _model(device, family, reference=index, strategy="unwhitened") for index in (False, True)
    ]
    outputs, priors = [], []
    for model, _, x, _ in pairs:
        model.train()
        strategy = model.model.variational_strategy
        strategy.variational_params_initialized.fill_(0)
        prior = strategy.prior_distribution
        priors.append((prior.mean, prior.covariance_matrix))
        outputs.append(model.model(x).mean)
    for a, b in zip(*priors, strict=True):
        torch.testing.assert_close(a, b, atol=2e-12, rtol=2e-12)
    torch.testing.assert_close(*outputs, atol=2e-12, rtol=2e-12)
    for index, (model, _, x, _) in enumerate(pairs):
        strategy = model.model.variational_strategy
        if family.startswith("Delta"):
            with pytest.raises(RuntimeError):
                model.model(strategy.inducing_points)
        else:
            output = model.model(strategy.inducing_points)
            torch.testing.assert_close(output.mean, strategy.variational_distribution.mean)
            torch.testing.assert_close(
                output.covariance_matrix, strategy.variational_distribution.covariance_matrix
            )
        function = reference_update if index else gaussian_update
        with pytest.raises(NotImplementedError):
            function(model, x.new_zeros(2, 3))


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("family", FAMILIES)
def test_unwhitened_actual_adam_training_matches_reference(device, family):
    models = [
        _model(device, family, reference=index, external=True, strategy="unwhitened")[0]
        for index in (False, True)
    ]
    mlls = [
        cls(model.likelihood, model.model, 5)
        for cls, model in zip((VariationalELBO, ReferenceELBO), models, strict=True)
    ]
    optimizers = [torch.optim.Adam(mll.parameters(), lr=0.008) for mll in mlls]
    for _ in range(4):
        losses = []
        for mll, optimizer in zip(mlls, optimizers, strict=True):
            optimizer.zero_grad()
            output = mll.model(mll.model.train_inputs[0])
            loss = -mll(output, mll.model.train_targets).sum()
            loss.backward()
            optimizer.step()
            losses.append(loss.detach())
        torch.testing.assert_close(*losses, atol=3e-8, rtol=3e-8)
    actual, expected = map(lambda model: dict(model.named_parameters()), models)
    assert actual.keys() == expected.keys()
    for name in actual:
        torch.testing.assert_close(actual[name], expected[name], atol=3e-8, rtol=3e-8)

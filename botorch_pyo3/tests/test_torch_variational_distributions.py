"""Alternative inducing-variable families use their actual device laws."""

# ruff: noqa: E402
import functools
from unittest.mock import patch

import pytest

torch = pytest.importorskip("torch")
gpytorch = pytest.importorskip("gpytorch")
pytest.importorskip("botorch")

from botorch.models import SingleTaskVariationalGP as ReferenceGP
from botorch.sampling.pathwise import draw_kernel_feature_paths as reference_prior
from botorch.sampling.pathwise import draw_matheron_paths as reference_paths
from gpytorch.mlls import VariationalELBO as ReferenceELBO

from botorch_rs import variational
from botorch_rs.kernels import RBFKernel
from botorch_rs.likelihoods import GaussianLikelihood, MultitaskGaussianLikelihood
from botorch_rs.mlls import VariationalELBO
from botorch_rs.models.approximate_gp import SingleTaskVariationalGP
from botorch_rs.sampling.pathwise import draw_kernel_feature_paths, draw_matheron_paths

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
FAMILIES = ["MeanFieldVariationalDistribution", "DeltaVariationalDistribution"]


def _model(
    device,
    family,
    reference=False,
    outputs=1,
    batch=False,
    external=False,
    graph=False,
    strategy="whitened",
):
    x = torch.linspace(0.1, 0.9, 10, dtype=torch.double, device=device).reshape(5, 2)
    y = x[:, :1].sin()
    if outputs > 1:
        y = torch.cat((y, 0.7 * y + 0.1), -1)
    if batch:
        x, y = x.expand(2, 5, 2).clone(), y.expand(2, 5, outputs).clone()
        x[1] += 0.03
        y[1] += 0.12
    x.requires_grad_(graph)
    y.requires_grad_(graph)
    aug = x.shape[:-2] + ((outputs,) if outputs > 1 else ())
    distribution_class = getattr(
        gpytorch.variational if reference or external else variational, family
    )
    source = distribution_class(3, batch_shape=torch.Size(aug), mean_init_std=0)
    cls = ReferenceGP if reference else SingleTaskVariationalGP
    kernel_class = gpytorch.kernels.RBFKernel if reference else RBFKernel
    likelihood_class = (
        (gpytorch.likelihoods.GaussianLikelihood if reference else GaussianLikelihood)
        if outputs == 1
        else (
            gpytorch.likelihoods.MultitaskGaussianLikelihood
            if reference
            else MultitaskGaussianLikelihood
        )
    )
    likelihood = (
        likelihood_class(batch_shape=torch.Size(aug)) if outputs == 1 else likelihood_class(outputs)
    )
    model = cls(
        x,
        y,
        covar_module=kernel_class(ard_num_dims=2, batch_shape=torch.Size(aug)),
        likelihood=likelihood,
        inducing_points=x.new_tensor([[0.12, 0.2], [0.44, 0.51], [0.84, 0.92]]),
        variational_distribution=source,
        outcome_transform=None,
        **({} if reference else {"backend": "torch"}),
        variational_strategy=getattr(
            gpytorch.variational if reference or external else variational,
            "VariationalStrategy" if strategy == "whitened" else "UnwhitenedVariationalStrategy",
        ),
    ).to(x)
    strategy = model.model.variational_strategy
    base = strategy.base_variational_strategy if outputs > 1 else strategy
    distribution = base._variational_distribution
    with torch.no_grad():
        distribution.variational_mean.copy_(x.new_tensor([-0.2, 0.15, 0.3]))
        if family.startswith("MeanField"):
            distribution._variational_stddev.copy_(x.new_tensor([-0.7, 0.9, 1.2]))
        base.variational_params_initialized.fill_(1)
        model.model.covar_module.lengthscale = x.new_tensor([0.4, 0.7])
        model.model.mean_module.constant = x.new_tensor(0.12)
        model.likelihood.noise = x.new_tensor(0.08)
    return model, source, x, y


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("outputs,batch", [(1, False), (1, True), (2, False), (2, True)])
def test_alternative_variational_q_elbo_kl_and_complete_first_gradients(
    device, family, outputs, batch
):
    models = [
        _model(device, family, reference, outputs, batch, graph=True) for reference in (False, True)
    ]
    values, gradients = [], []
    for index, (model, _, x, y) in enumerate(models):
        likelihood = ReferenceELBO if index else VariationalELBO
        mll = likelihood(model.likelihood, model.model, num_data=11, beta=0.7, combine_terms=False)
        # GPyTorch's inner ApproximateGP accepts output batches explicitly;
        # the port's model adapter inserts this axis for batched model inputs.
        inputs = x.unsqueeze(-3) if index and outputs > 1 and batch else x
        output = model.model(inputs)
        terms = mll(output, y.squeeze(-1) if outputs == 1 else y)
        values.append((output.mean, output.variance, output.covariance_matrix, *terms))
        parameters = dict(model.named_parameters())
        leaves = tuple(parameters[name] for name in sorted(parameters)) + (x, y)
        gradients.append(torch.autograd.grad(sum(v.sum() for v in terms), leaves))
    for actual, expected in zip(*values, strict=True):
        torch.testing.assert_close(actual, expected, atol=3e-8, rtol=3e-8)
    for actual, expected in zip(*gradients, strict=True):
        torch.testing.assert_close(actual, expected, atol=5e-7, rtol=5e-7)
    for reference, (model, _, x, _) in enumerate(models):
        model.eval()
        inputs = x.unsqueeze(-3) if reference and outputs > 1 and batch else x
        output = model.model(inputs)
        values[reference] = (output.mean, output.variance, output.covariance_matrix)
    for actual, expected in zip(*values, strict=True):
        torch.testing.assert_close(actual, expected, atol=3e-8, rtol=3e-8)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("family", [*FAMILIES, "CholeskyVariationalDistribution"])
def test_external_distribution_leaves_and_actual_adam_updates_match_reference(device, family):
    actual, source, x, _ = _model(device, family, external=True)
    expected, _, _, _ = _model(device, family, reference=True)
    assert actual._strategy._variational_distribution.variational_mean is source.variational_mean
    if family.startswith("MeanField"):
        assert (
            actual._strategy._variational_distribution._variational_stddev
            is source._variational_stddev
        )
        source._variational_stddev.requires_grad_(False)
        expected.model.variational_strategy._variational_distribution._variational_stddev.requires_grad_(
            False
        )
    mlls = [
        VariationalELBO(actual.likelihood, actual.model, 5),
        ReferenceELBO(expected.likelihood, expected.model, 5),
    ]
    optimizers = [torch.optim.Adam(mll.parameters(), lr=0.015) for mll in mlls]
    for _ in range(4):
        losses = []
        for mll, optimizer in zip(mlls, optimizers, strict=True):
            optimizer.zero_grad()
            loss = -mll(mll.model(x), mll.model.train_targets).sum()
            loss.backward()
            optimizer.step()
            losses.append(loss.detach())
        torch.testing.assert_close(*losses, atol=2e-8, rtol=2e-8)
    actual_parameters, expected_parameters = (
        dict(actual.named_parameters()),
        dict(expected.named_parameters()),
    )
    assert actual_parameters.keys() == expected_parameters.keys()
    for name, value in actual_parameters.items():
        torch.testing.assert_close(value, expected_parameters[name], atol=2e-8, rtol=2e-8)
    assert not torch.equal(source.variational_mean, x.new_tensor([-0.2, 0.15, 0.3]))


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("family", FAMILIES)
def test_all_second_derivatives_match_independent_dense_rbf_variational_formula(device, family):
    model, _, x, y = _model(device, family, graph=True)
    mll = VariationalELBO(model.likelihood, model.model, len(x))
    actual = -mll().sum()
    strategy = model._strategy
    z, distribution = strategy.inducing_points, strategy._variational_distribution
    lengthscale = model.model.covar_module.lengthscale
    constant, noise = model.model.mean_module.constant, model.likelihood.noise

    def kernel(a, b):
        return (
            -0.5
            * ((a.unsqueeze(-2) - b.unsqueeze(-3)) / lengthscale.unsqueeze(-2)).square().sum(-1)
        ).exp()

    eye = torch.eye(3, dtype=x.dtype, device=x.device)
    root = torch.linalg.cholesky(kernel(z, z) + 1e-6 * eye)
    interpolation = torch.linalg.solve_triangular(root, kernel(z, x), upper=False)
    covariance = (
        torch.diag_embed(distribution._variational_stddev.square())
        if family.startswith("MeanField")
        else torch.zeros_like(eye)
    )
    mean = constant + (interpolation.mT @ distribution.variational_mean.unsqueeze(-1)).squeeze(-1)
    variance = 1 + 1e-6 + (interpolation * ((covariance - eye) @ interpolation)).sum(-2)
    kl = (
        0.5
        * (
            distribution.variational_mean.square()
            + distribution._variational_stddev.square()
            - 1
            - distribution._variational_stddev.square().log()
        ).sum()
        if family.startswith("MeanField")
        else 0.5
        * (distribution.variational_mean.square() + torch.log(x.new_tensor(2 * torch.pi))).sum()
    )
    expected = 0.5 * (
        ((y.squeeze(-1) - mean).square() + variance) / noise
        + noise.log()
        + torch.log(x.new_tensor(2 * torch.pi))
    ).mean() + kl / len(x)
    torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)
    leaves = tuple(model.parameters()) + (x, y)
    gradients = [
        torch.autograd.grad(loss, leaves, create_graph=True) for loss in (actual, expected)
    ]
    for a, b in zip(*gradients, strict=True):
        torch.testing.assert_close(a, b, atol=2e-11, rtol=2e-11)
    second = [
        torch.autograd.grad(sum(g.square().sum() for g in gradient), leaves)
        for gradient in gradients
    ]
    for a, b in zip(*second, strict=True):
        torch.testing.assert_close(a, b, atol=2e-9, rtol=2e-9)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("family", FAMILIES)
def test_alternative_current_q_matheron_paths_match_independent_reference(device, family):
    model, _, x, _ = _model(device, family)
    reference, _, _, _ = _model(device, family, reference=True)
    reference.eval()
    model.eval()
    distribution = reference.model.variational_strategy.variational_distribution
    normal = x.new_tensor([[-0.3, 0.4, -0.2], [0.7, -0.6, 0.1]])
    original = distribution.rsample

    def sample(shape):
        return (
            original(shape, base_samples=normal)
            if family.startswith("MeanField")
            else original(shape)
        )

    with patch.object(distribution, "rsample", side_effect=sample):
        expected = reference_paths(
            reference, torch.Size([2]), functools.partial(reference_prior, num_features=16)
        )
    prior = expected["prior_paths"]
    draws = [(prior.feature_map.weight.detach(), prior.weight.detach(), normal)]
    actual = draw_matheron_paths(
        model, (2,), functools.partial(draw_kernel_feature_paths, num_features=16), draws=draws
    )
    query = x.new_tensor([[0.25, 0.32], [0.65, 0.73]], requires_grad=True)
    torch.testing.assert_close(actual(query), expected(query), atol=3e-10, rtol=3e-10)
    a = torch.autograd.grad(actual(query).sum(), query, create_graph=True)[0]
    b = torch.autograd.grad(expected(query).sum(), query)[0]
    torch.testing.assert_close(a, b, atol=2e-9, rtol=2e-9)
    assert bool(torch.isfinite(torch.autograd.grad(a.square().sum(), query)[0]).all())


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("family", FAMILIES)
def test_alternative_distribution_lazy_initialization_and_public_sampling(device, family):
    models = [_model(device, family, reference=index) for index in (False, True)]
    values, distributions = [], []
    for model, _, x, _ in models:
        strategy = model.model.variational_strategy
        strategy.variational_params_initialized.fill_(0)
        model.model.forward(x)
        assert not bool(strategy.variational_params_initialized)
        values.append(model.model(x).mean)
        assert bool(strategy.variational_params_initialized)
        distributions.append(strategy.variational_distribution)
    torch.testing.assert_close(*values, atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(distributions[0].mean, distributions[1].mean, atol=0, rtol=0)
    torch.testing.assert_close(distributions[0].variance, distributions[1].variance, atol=0, rtol=0)
    for distribution in distributions:
        samples = distribution.rsample(torch.Size([2, 4]))
        assert samples.shape == (2, 4, 3)
        assert samples.device.type == device
        assert torch.isfinite(samples).all()
        if family.startswith("Delta"):
            torch.testing.assert_close(samples, distribution.mean.expand(2, 4, 3), atol=0, rtol=0)

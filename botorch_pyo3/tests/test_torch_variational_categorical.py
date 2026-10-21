# ruff: noqa: E402
"""Multiple latent variational functions preserve categorical likelihood laws."""

import pytest

torch = pytest.importorskip("torch")
gpytorch = pytest.importorskip("gpytorch")
pytest.importorskip("botorch")

from botorch.models import SingleTaskVariationalGP as ReferenceGP
from gpytorch.mlls import VariationalELBO as ReferenceELBO
from test_torch_variational_external_likelihood import DenseRBF

from botorch_rs.mlls import VariationalELBO
from botorch_rs.models import SingleTaskVariationalGP

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


def pair(device, batch, mixing):
    models = []
    latent_count = 2 if mixing else 3
    for reference in (False, True):
        x = torch.tensor([[0.1], [0.3], [0.5], [0.7], [0.9]], device=device, dtype=torch.double)
        if batch:
            x = x.expand(2, 5, 1).clone()
            x[1] += 0.02
        x.requires_grad_()
        augmented = torch.Size((*x.shape[:-2], latent_count))
        likelihood = gpytorch.likelihoods.SoftmaxLikelihood(
            num_features=latent_count, num_classes=3, mixing_weights=mixing
        ).to(x)
        distribution = gpytorch.variational.CholeskyVariationalDistribution(
            3, batch_shape=augmented, mean_init_std=0
        ).to(x)
        kernel = DenseRBF(batch_shape=augmented).to(x)
        kernel.lengthscale = 0.5
        cls = ReferenceGP if reference else SingleTaskVariationalGP
        model = cls(
            x,
            num_outputs=latent_count,
            likelihood=likelihood,
            covar_module=kernel,
            mean_module=gpytorch.means.ZeroMean(batch_shape=augmented),
            variational_distribution=distribution,
            inducing_points=x.new_tensor([[0.1], [0.5], [0.9]]),
            **({} if reference else {"backend": "torch"}),
        ).to(x)
        assert model.likelihood is likelihood
        with torch.no_grad():
            strategy = model.model.variational_strategy.base_variational_strategy
            strategy.variational_params_initialized.fill_(1)
            distribution.variational_mean.copy_(
                x.new_tensor([-0.3, 0.1, 0.4])
                + torch.arange(latent_count, device=x.device).unsqueeze(-1) * 0.2
            )
            distribution.chol_variational_covar.copy_(
                torch.eye(3, device=x.device, dtype=x.dtype) * 0.8
            )
            if mixing:
                likelihood.mixing_weights.copy_(
                    x.new_tensor([[0.6, -0.2], [-0.1, 0.5], [0.3, 0.2]])
                )
        models.append((model, x))
    return models


def dense_moments(model, x):
    strategy = model.model.variational_strategy.base_variational_strategy
    points, kernel = strategy.inducing_points, model.model.covar_module
    eye = torch.eye(3, dtype=x.dtype, device=x.device)
    root = torch.linalg.cholesky(kernel.forward(points, points) + strategy.jitter_val * eye)
    interpolation = torch.linalg.solve_triangular(
        root, kernel.forward(points, x.unsqueeze(-3)), upper=False
    )
    distribution = strategy._variational_distribution
    mu, qroot = distribution.variational_mean, distribution.chol_variational_covar.tril()
    covariance = qroot @ qroot.mT
    mean = (interpolation.mT @ mu.unsqueeze(-1)).squeeze(-1)
    variance = (
        1 + strategy.jitter_val + (interpolation * ((covariance - eye) @ interpolation)).sum(-2)
    )
    kl = 0.5 * (
        mu.square().sum(-1)
        + covariance.diagonal(dim1=-2, dim2=-1).sum(-1)
        - 3
        - 2 * qroot.diagonal(dim1=-2, dim2=-1).log().sum(-1)
    )
    return mean.transpose(-1, -2), variance.transpose(-1, -2), kl.sum(-1)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("batch", [False, True])
@pytest.mark.parametrize("mixing", [False, True])
def test_categorical_elbo_values_first_derivatives_and_independent_dense_higher(
    device, batch, mixing
):
    devices = [torch.cuda.current_device()] if device == "cuda" else []
    results, gradients, higher = [], [], []
    target = torch.tensor([0, 1, 2, 1, 0], dtype=torch.long, device=device)
    if batch:
        target = target.expand(2, 5)
    for reference, (model, x) in enumerate(pair(device, batch, mixing)):
        output = model.model(x.unsqueeze(-3) if reference and batch else x)
        cls = ReferenceELBO if reference else VariationalELBO
        mll = cls(model.likelihood, model.model, num_data=11, beta=0.7)
        with torch.random.fork_rng(devices=devices), gpytorch.settings.num_likelihood_samples(9):
            torch.manual_seed(71)
            value = mll(output, target)
        results.append(value)
        parameters = dict(model.named_parameters())
        leaves = (x, *(parameters[name] for name in sorted(parameters)))
        first = torch.autograd.grad(value.sum(), leaves, create_graph=True, retain_graph=True)
        gradients.append(first)
        if reference:
            mean, variance, kl = dense_moments(model, x)
            with torch.random.fork_rng(devices=devices):
                torch.manual_seed(71)
                normals = torch.randn(9, *mean.shape, dtype=x.dtype, device=x.device)
            samples = mean + variance.sqrt() * normals
            logits = samples @ model.likelihood.mixing_weights.mT if mixing else samples
            log_probability = (
                logits.log_softmax(-1)
                .gather(-1, target.expand(9, *target.shape).unsqueeze(-1))
                .squeeze(-1)
            )
            dense = log_probability.mean((0, -1)) - 0.7 / 11 * kl
            torch.testing.assert_close(value, dense, rtol=2e-9, atol=2e-10)
            dense_first = torch.autograd.grad(dense.sum(), leaves, create_graph=True)
            for a, b in zip(first, dense_first, strict=True):
                torch.testing.assert_close(a, b, rtol=3e-7, atol=3e-8)
            higher.append(torch.autograd.grad(sum(g.square().sum() for g in dense_first), leaves))
        else:
            higher.append(torch.autograd.grad(sum(g.square().sum() for g in first), leaves))
    torch.testing.assert_close(*results, rtol=2e-9, atol=2e-10)
    for a, b in zip(*gradients, strict=True):
        torch.testing.assert_close(a, b, rtol=3e-7, atol=3e-8)
    for a, b in zip(*higher, strict=True):
        torch.testing.assert_close(a, b, rtol=3e-6, atol=3e-7)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("batch", [False, True])
@pytest.mark.parametrize("mixing", [False, True])
def test_categorical_observed_probabilities_keep_full_sampling_graph(device, batch, mixing):
    model, x = pair(device, batch, mixing)[0]
    query = x.new_tensor([[0.2], [0.55], [0.8], [0.95]], requires_grad=True)
    model.eval()
    latent = model.model(query)
    devices = [torch.cuda.current_device()] if device == "cuda" else []
    with torch.random.fork_rng(devices=devices), gpytorch.settings.num_likelihood_samples(7):
        torch.manual_seed(81)
        posterior = model.posterior(query, observation_noise=True)
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(81)
        normals = torch.randn(7, *latent.mean.shape, dtype=query.dtype, device=query.device)
    covariance = latent.covariance_matrix
    samples = latent.mean + (
        torch.linalg.cholesky(covariance) @ normals.flatten(-2).unsqueeze(-1)
    ).squeeze(-1).reshape(7, *latent.mean.shape)
    expected = model.likelihood.forward(samples)
    assert isinstance(posterior.distribution, torch.distributions.Categorical)
    torch.testing.assert_close(posterior.probs, expected.probs, rtol=2e-9, atol=2e-10)
    leaves = (query, *model.parameters())
    first = torch.autograd.grad(
        posterior.probs.square().sum(), leaves, create_graph=True, retain_graph=True
    )
    reference_first = torch.autograd.grad(
        expected.probs.square().sum(), leaves, create_graph=True, retain_graph=True
    )
    higher = torch.autograd.grad(sum(g.square().sum() for g in first), leaves, retain_graph=True)
    reference_higher = torch.autograd.grad(sum(g.square().sum() for g in reference_first), leaves)
    for a, b in zip((*first, *higher), (*reference_first, *reference_higher), strict=True):
        torch.testing.assert_close(a, b, rtol=3e-7, atol=3e-8)

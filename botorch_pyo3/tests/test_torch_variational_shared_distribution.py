"""A supplied q(u) may share parameters across model and output batches."""

# ruff: noqa: E402
from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")

from botorch_rs.mlls import VariationalELBO
from botorch_rs.models import SingleTaskVariationalGP
from botorch_rs.variational import CholeskyVariationalDistribution

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
CASES = [
    (1, (2,), ()),
    (2, (), ()),
    (2, (2,), ()),
    (2, (2,), (1, 2)),
    (1, (), (2,)),
    (2, (), (3, 1)),
]


class DenseKernel(torch.nn.Module):
    def __init__(self, batch, like):
        super().__init__()
        self.batch_shape = torch.Size(batch)
        self.raw_length = torch.nn.Parameter(like.new_full((*batch, 1, 2), -0.4))

    def forward(self, x, y=None, diag=False):
        y = x if y is None else y
        difference = (x - y) if diag else (x.unsqueeze(-2) - y.unsqueeze(-3))
        scale = self.raw_length.exp() if diag else self.raw_length.exp().unsqueeze(-2)
        return (-0.5 * (difference / scale).square().sum(-1)).exp()


class DenseMean(torch.nn.Module):
    def __init__(self, batch, like):
        super().__init__()
        self.constant = torch.nn.Parameter(like.new_full(batch, 0.12))

    def forward(self, x):
        batch = torch.broadcast_shapes(x.shape[:-2], self.constant.shape)
        return self.constant.unsqueeze(-1).expand(*batch, x.shape[-2])


def make_model(device, outputs, batch, distribution_batch, *, reference=False, external=False):
    from botorch_rs.likelihoods import GaussianLikelihood, MultitaskGaussianLikelihood

    x = torch.tensor(
        [[0.1, 0.2], [0.3, 0.7], [0.6, 0.4], [0.9, 0.8]],
        device=device,
        dtype=torch.double,
    )
    x = x.expand(*batch, 4, 2).clone().requires_grad_()
    y = torch.cat([x[..., :1].sin() + i * 0.2 for i in range(outputs)], -1)
    y = y.detach().clone().requires_grad_()
    aug = batch + ((outputs,) if outputs > 1 else ())
    distribution_class = CholeskyVariationalDistribution
    model_class = SingleTaskVariationalGP
    scalar, multitask = GaussianLikelihood, MultitaskGaussianLikelihood
    if reference or external:
        gpytorch = pytest.importorskip("gpytorch")
        distribution_class = gpytorch.variational.CholeskyVariationalDistribution
        scalar, multitask = (
            gpytorch.likelihoods.GaussianLikelihood,
            gpytorch.likelihoods.MultitaskGaussianLikelihood,
        )
    if reference:
        pytest.importorskip("botorch")
        from botorch.models import SingleTaskVariationalGP as model_class
    distribution = distribution_class(
        3, batch_shape=torch.Size(distribution_batch), mean_init_std=0
    )
    likelihood = scalar(batch_shape=torch.Size(batch)) if outputs == 1 else multitask(outputs)
    model = model_class(
        x,
        y,
        covar_module=DenseKernel(aug, x),
        mean_module=DenseMean(aug, x),
        likelihood=likelihood,
        inducing_points=x.new_tensor([[0.12, 0.2], [0.44, 0.51], [0.84, 0.92]]),
        variational_distribution=distribution,
        outcome_transform=None,
        **({} if reference else {"backend": "torch"}),
    ).to(x)
    strategy = model.model.variational_strategy
    if outputs > 1:
        strategy = strategy.base_variational_strategy
    with torch.no_grad():
        strategy.variational_params_initialized.fill_(1)
        strategy._variational_distribution.variational_mean.copy_(x.new_tensor([-0.2, 0.15, 0.3]))
        strategy._variational_distribution.chol_variational_covar.copy_(
            x.new_tensor([[0.8, 0, 0], [0.1, 0.9, 0], [-0.05, 0.12, 0.7]])
        )
        model.likelihood.noise = x.new_tensor(0.08)
        if outputs > 1:
            model.likelihood.task_noises = x.new_tensor([0.03, 0.04])
    return model, strategy, x, y


def dense_terms(model, strategy, x, y):
    outputs = model.num_outputs
    query = x.unsqueeze(-3) if outputs > 1 else x
    points = strategy.inducing_points
    if outputs > 1:
        points = points.unsqueeze(-3)
    kernel, mean_module = model.model.covar_module, model.model.mean_module
    eye = torch.eye(3, device=x.device, dtype=x.dtype)
    root = torch.linalg.cholesky(kernel(points, points) + strategy.jitter_val * eye)
    interpolation = torch.linalg.solve_triangular(root, kernel(points, query), upper=False)
    distribution = strategy._variational_distribution
    mu = distribution.variational_mean
    factor = distribution.chol_variational_covar.tril()
    q_covariance = factor @ factor.mT
    mean = mean_module(query) + (interpolation.mT @ mu.unsqueeze(-1)).squeeze(-1)
    covariance = (
        kernel(query, query)
        + strategy.jitter_val * torch.eye(x.shape[-2], device=x.device, dtype=x.dtype)
        + interpolation.mT @ ((q_covariance - eye) @ interpolation)
    )
    target = y.transpose(-1, -2) if outputs > 1 else y.squeeze(-1)
    noise = model.likelihood.noise
    if outputs > 1:
        noise = (noise + model.likelihood.task_noises).unsqueeze(-1)
    log_probability = -0.5 * (
        ((target - mean).square() + covariance.diagonal(dim1=-2, dim2=-1)) / noise
        + noise.log()
        + math.log(2 * math.pi)
    ).mean(-1)
    kl = 0.5 * (
        q_covariance.diagonal(dim1=-2, dim2=-1).sum(-1)
        + mu.square().sum(-1)
        - 3
        - 2 * factor.diagonal(dim1=-2, dim2=-1).log().sum(-1)
    )
    if outputs > 1:
        log_probability, kl = log_probability.sum(-1), kl.sum(-1)
    return mean, covariance, log_probability - 0.7 * kl / 11


def loss_terms(mean, covariance, elbo):
    weights = torch.linspace(0.7, 1.2, elbo.numel(), device=elbo.device, dtype=elbo.dtype)
    return (
        (elbo.reshape(-1) * weights).sum()
        + 0.03 * covariance.square().sum()
        + 0.07 * mean.sin().sum()
    )


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("outputs,batch,distribution_batch", CASES)
@pytest.mark.parametrize("external", [False, True])
def test_shared_distribution_matches_dense_values_all_gradients_and_hvp(
    device, outputs, batch, distribution_batch, external
):
    model, strategy, x, y = make_model(
        device, outputs, batch, distribution_batch, external=external
    )
    assert tuple(strategy._variational_distribution.batch_shape) == distribution_batch
    assert strategy._variational_distribution.variational_mean.shape[:-1] == distribution_batch
    model.eval()
    output = model.model(x)
    mll = VariationalELBO(model.likelihood, model.model, 11, beta=0.7)
    actual_elbo = mll(output, y.squeeze(-1) if outputs == 1 else y)
    mean, covariance, expected_elbo = dense_terms(model, strategy, x, y)
    torch.testing.assert_close(output._mean, mean, atol=2e-10, rtol=2e-10)
    torch.testing.assert_close(output._blocks(), covariance, atol=2e-10, rtol=2e-10)
    torch.testing.assert_close(actual_elbo, expected_elbo, atol=2e-10, rtol=2e-10)
    leaves = (*model.parameters(), x, y)
    gradients = [
        torch.autograd.grad(loss, leaves, create_graph=True)
        for loss in (
            loss_terms(output._mean, output._blocks(), actual_elbo),
            loss_terms(mean, covariance, expected_elbo),
        )
    ]
    for actual, expected in zip(*gradients, strict=True):
        torch.testing.assert_close(actual, expected, atol=3e-9, rtol=3e-9)
    hvps = [torch.autograd.grad(sum(g.square().sum() / 2 for g in gs), leaves) for gs in gradients]
    for actual, expected in zip(*hvps, strict=True):
        torch.testing.assert_close(actual, expected, atol=1e-8, rtol=1e-8)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("outputs,batch,distribution_batch", CASES)
def test_shared_distribution_source_values_and_all_first_gradients(
    device, outputs, batch, distribution_batch
):
    pytest.importorskip("botorch")
    from gpytorch.mlls import VariationalELBO as ReferenceELBO

    models = [
        make_model(device, outputs, batch, distribution_batch, reference=reference, external=True)
        for reference in (False, True)
    ]
    values, gradients = [], []
    for reference, (model, _, x, y) in enumerate(models):
        inputs = x.unsqueeze(-3) if reference and outputs > 1 and batch else x
        output = model.model(inputs)
        mll = (ReferenceELBO if reference else VariationalELBO)(
            model.likelihood, model.model, 11, beta=0.7
        )
        elbo = mll(output, y.squeeze(-1) if outputs == 1 else y)
        values.append((output.mean, output.variance, output.covariance_matrix, elbo))
        params = dict(model.named_parameters())
        gradients.append(
            torch.autograd.grad(
                loss_terms(output.mean, output.covariance_matrix, elbo),
                (*(params[name] for name in sorted(params)), x, y),
            )
        )
    for actual, expected in zip(*values, strict=True):
        torch.testing.assert_close(actual, expected, atol=3e-9, rtol=3e-9)
    for actual, expected in zip(*gradients, strict=True):
        torch.testing.assert_close(actual, expected, atol=1e-8, rtol=1e-8)


@pytest.mark.parametrize("device", DEVICES)
def test_shared_q_snapshot_survives_later_distribution_parameter_changes(device):
    model, strategy, x, _ = make_model(device, 2, (2,), (), external=True)
    old = model.posterior(x)
    saved = old.mean.clone(), old.variance.clone()
    with torch.no_grad():
        strategy._variational_distribution.variational_mean.add_(0.4)
        strategy._variational_distribution.chol_variational_covar.mul_(0.9)
    newer = model.posterior(x)
    assert not torch.allclose(newer.mean, saved[0])
    assert not torch.allclose(newer.variance, saved[1])
    torch.testing.assert_close(old.mean, saved[0])
    torch.testing.assert_close(old.variance, saved[1])


def test_incompatible_distribution_batches_have_clear_error():
    x = torch.rand(2, 4, 2, dtype=torch.double)
    with pytest.raises(ValueError, match="must broadcast"):
        SingleTaskVariationalGP(
            x,
            inducing_points=x[0, :3],
            variational_distribution=CholeskyVariationalDistribution(3, batch_shape=(3,)),
            backend="torch",
        )

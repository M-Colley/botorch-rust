# ruff: noqa: E402
"""Actual Gaussian observation modules retain their laws, identity and graphs."""

import copy
import math

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
FAMILIES = ["gaussian", "shifted", "multitask_diagonal", "multitask_correlated"]


class ShiftedGaussian(gpytorch.likelihoods.GaussianLikelihood):
    def __init__(self, batch_shape=()):
        super().__init__(batch_shape=batch_shape)
        self.offset = torch.nn.Parameter(torch.full((*batch_shape, 1), 0.15))

    def marginal(self, function, *args, **kwargs):
        result = super().marginal(function, *args, **kwargs)
        return type(result)(result.mean + self.offset, result.lazy_covariance_matrix)

    def expected_log_prob(self, target, function, likelihood_weight=1, **kwargs):
        return (
            super().expected_log_prob(target - self.offset, function, **kwargs) * likelihood_weight
        )


def pair(device, family, batch):
    models = []
    m = 2 if family.startswith("multitask") else 1
    for reference in (False, True):
        x = torch.tensor([[0.1], [0.3], [0.5], [0.7], [0.9]], dtype=torch.double, device=device)
        y = x.new_tensor([[0.2], [0.7], [0.4], [0.8], [0.3]]).expand(5, m).clone()
        if m > 1:
            y[:, 1] += 0.3
        if batch:
            x, y = x.expand(2, 5, 1).clone(), y.expand(2, 5, m).clone()
            x[1] += 0.02
            y[1] -= 0.1
        x.requires_grad_()
        y.requires_grad_()
        batch_shape = x.shape[:-2]
        augmented = torch.Size((*batch_shape, m)) if m > 1 else batch_shape
        if m > 1:
            likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(
                m, rank=int(family == "multitask_correlated"), batch_shape=batch_shape
            ).to(x)
        else:
            cls = (
                ShiftedGaussian if family == "shifted" else gpytorch.likelihoods.GaussianLikelihood
            )
            likelihood = cls(batch_shape=batch_shape).to(x)
        likelihood.noise = 0.06
        if m > 1:
            if likelihood.rank:
                with torch.no_grad():
                    likelihood.task_noise_covar_factor.copy_(x.new_tensor([[0.3], [-0.2]]))
            else:
                likelihood.task_noises = x.new_tensor([0.03, 0.09])
        distribution = gpytorch.variational.CholeskyVariationalDistribution(
            3, batch_shape=augmented, mean_init_std=0
        ).to(x)
        kernel = DenseRBF(batch_shape=augmented).to(x)
        kernel.lengthscale = 0.5
        cls = ReferenceGP if reference else SingleTaskVariationalGP
        model = cls(
            x,
            y,
            likelihood=likelihood,
            covar_module=kernel,
            mean_module=gpytorch.means.ZeroMean(batch_shape=augmented),
            inducing_points=x.new_tensor([[0.1], [0.5], [0.9]]),
            variational_distribution=distribution,
            outcome_transform=None,
            **({} if reference else {"backend": "torch"}),
        ).to(x)
        assert model.likelihood is likelihood
        with torch.no_grad():
            strategy = model.model.variational_strategy
            if m > 1:
                strategy = strategy.base_variational_strategy
            strategy.variational_params_initialized.fill_(1)
            distribution.variational_mean.copy_(x.new_tensor([-0.3, 0.1, 0.4]))
            distribution.chol_variational_covar.copy_(
                torch.eye(3, dtype=x.dtype, device=x.device) * 0.8
            )
        models.append((model, x, y))
    return models


def dense_moments(model, x, *, training):
    m = model.num_outputs
    strategy = model.model.variational_strategy
    if m > 1:
        strategy = strategy.base_variational_strategy
        x = x.unsqueeze(-3)
    points, kernel = strategy.inducing_points, model.model.covar_module
    eye = torch.eye(3, device=x.device, dtype=x.dtype)
    root = torch.linalg.cholesky(kernel.forward(points, points) + strategy.jitter_val * eye)
    interpolation = torch.linalg.solve_triangular(root, kernel.forward(points, x), upper=False)
    distribution = strategy._variational_distribution
    mu, qroot = distribution.variational_mean, distribution.chol_variational_covar.tril()
    covariance = qroot @ qroot.mT
    mean = (interpolation.mT @ mu.unsqueeze(-1)).squeeze(-1)
    update = interpolation.mT @ ((covariance - eye) @ interpolation)
    if training:
        # GPyTorch's training fast path retains only the update diagonal.
        qcov = kernel.forward(x, x) + torch.diag_embed(update.diagonal(dim1=-2, dim2=-1))
    else:
        qcov = kernel.forward(x, x) + update
    qcov = qcov + strategy.jitter_val * torch.eye(x.shape[-2], device=x.device, dtype=x.dtype)
    kl = 0.5 * (
        mu.square().sum(-1)
        + covariance.diagonal(dim1=-2, dim2=-1).sum(-1)
        - 3
        - 2 * qroot.diagonal(dim1=-2, dim2=-1).log().sum(-1)
    )
    if m > 1:
        mean = mean.transpose(-1, -2)
        qcov = torch.einsum(
            "...tij,tu->...tiuj", qcov, torch.eye(m, dtype=x.dtype, device=x.device)
        ).reshape(*qcov.shape[:-3], m * x.shape[-2], m * x.shape[-2])
        kl = kl.sum(-1)
    return mean, qcov, kl


def noise_covariance(likelihood, mean):
    if isinstance(likelihood, gpytorch.likelihoods.MultitaskGaussianLikelihood):
        n, m = mean.shape[-2:]
        task = (
            likelihood.task_noise_covar_factor @ likelihood.task_noise_covar_factor.mT
            if likelihood.rank
            else torch.diag_embed(likelihood.task_noises)
        )
        task = task + likelihood.noise.unsqueeze(-1) * torch.eye(
            m, dtype=mean.dtype, device=mean.device
        )
        return torch.einsum(
            "...tu,ij->...tiuj", task, torch.eye(n, dtype=mean.dtype, device=mean.device)
        ).reshape(*mean.shape[:-2], m * n, m * n)
    return torch.diag_embed(likelihood.noise.expand_as(mean))


def derivatives(value, leaves):
    first = torch.autograd.grad(value.sum(), leaves, create_graph=True, retain_graph=True)
    higher = torch.autograd.grad(sum(g.square().sum() for g in first), leaves, retain_graph=True)
    return (*first, *higher)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("batch", [False, True])
def test_actual_gaussian_elbo_source_first_and_independent_full_higher_graphs(
    device, family, batch
):
    values, first, higher = [], [], []
    for reference, (model, x, y) in enumerate(pair(device, family, batch)):
        output = model.model(
            x.unsqueeze(-3) if reference and batch and model.num_outputs > 1 else x
        )
        target = y if model.num_outputs > 1 else y.squeeze(-1)
        cls = ReferenceELBO if reference else VariationalELBO
        value = cls(model.likelihood, model.model, num_data=11, beta=0.7)(output, target)
        values.append(value)
        parameters = dict(model.named_parameters())
        leaves = (x, y, *(parameters[name] for name in sorted(parameters)))
        gradients = torch.autograd.grad(value.sum(), leaves, create_graph=True, retain_graph=True)
        first.append(gradients)
        mean, covariance, kl = dense_moments(model, x, training=True)
        if model.num_outputs > 1:
            variance = (
                covariance.diagonal(dim1=-2, dim2=-1)
                .reshape(*mean.shape[:-2], model.num_outputs, mean.shape[-2])
                .transpose(-1, -2)
            )
            noise = noise_covariance(model.likelihood, mean).diagonal(dim1=-2, dim2=-1)
            noise = noise.reshape(*mean.shape[:-2], model.num_outputs, mean.shape[-2]).transpose(
                -1, -2
            )
        else:
            variance, noise = covariance.diagonal(dim1=-2, dim2=-1), model.likelihood.noise
        if isinstance(model.likelihood, ShiftedGaussian):
            mean = mean + model.likelihood.offset
        likelihood = -0.5 * (
            ((target - mean).square() + variance) / noise + noise.log() + math.log(2 * math.pi)
        )
        expected = likelihood.sum(-1).mean(-1) if model.num_outputs > 1 else likelihood.mean(-1)
        expected = expected - 0.7 / 11 * kl
        torch.testing.assert_close(value, expected, rtol=2e-9, atol=2e-10)
        dense_first = torch.autograd.grad(
            expected.sum(), leaves, create_graph=True, retain_graph=True
        )
        for a, b in zip(gradients, dense_first, strict=True):
            torch.testing.assert_close(a, b, rtol=3e-7, atol=3e-8)
        higher.append(
            torch.autograd.grad(
                sum(g.square().sum() for g in (dense_first if reference else gradients)), leaves
            )
        )
    torch.testing.assert_close(*values, rtol=2e-9, atol=2e-10)
    for a, b in zip(*first, strict=True):
        torch.testing.assert_close(a, b, rtol=3e-7, atol=3e-8)
    for a, b in zip(*higher, strict=True):
        torch.testing.assert_close(a, b, rtol=3e-6, atol=3e-7)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("batch", [False, True])
def test_observed_gaussian_preserves_full_covariance_samples_and_parameter_graphs(
    device, family, batch
):
    (model, x, _), (reference, rx, _) = pair(device, family, batch)
    query = x.new_tensor([[0.24], [0.43], [0.66], [0.81]], requires_grad=True)
    actual = model.posterior(query, observation_noise=True)
    source = reference.posterior(query.detach(), observation_noise=True)
    torch.testing.assert_close(actual.mean, source.mean, rtol=3e-8, atol=3e-9)
    torch.testing.assert_close(actual.variance, source.variance, rtol=3e-8, atol=3e-9)
    mean, covariance, _ = dense_moments(model, query, training=False)
    covariance = covariance + noise_covariance(model.likelihood, mean)
    if isinstance(model.likelihood, ShiftedGaussian):
        mean = mean + model.likelihood.offset
    if model.num_outputs == 1:
        mean = mean.unsqueeze(-1)
    torch.testing.assert_close(actual.mean, mean, rtol=2e-9, atol=2e-10)
    torch.testing.assert_close(
        actual.distribution.covariance_matrix, covariance, rtol=2e-9, atol=2e-10
    )
    normals = torch.linspace(
        -0.8, 0.7, 3 * actual.base_sample_shape.numel(), device=x.device, dtype=x.dtype
    ).reshape(3, *actual.base_sample_shape)
    sample = actual.rsample_from_base_samples(torch.Size([3]), normals)
    flat_normals = normals.transpose(-1, -2).flatten(-2) if model.num_outputs > 1 else normals
    flat_mean = mean.transpose(-1, -2).flatten(-2)
    expected_sample = flat_mean + (
        torch.linalg.cholesky(covariance) @ flat_normals.unsqueeze(-1)
    ).squeeze(-1)
    expected_sample = expected_sample.reshape(
        3, *mean.shape[:-2], model.num_outputs, mean.shape[-2]
    ).transpose(-1, -2)
    torch.testing.assert_close(sample, expected_sample, rtol=2e-9, atol=2e-10)
    leaves = (query, *model.parameters())
    for a, b in zip(
        derivatives(sample.sin(), leaves), derivatives(expected_sample.sin(), leaves), strict=True
    ):
        torch.testing.assert_close(a, b, rtol=3e-6, atol=3e-7)


@pytest.mark.parametrize("device", DEVICES)
def test_actual_gaussian_saved_likelihood_state_kwargs_and_roundtrip(device):
    (model, x, y), _ = pair(device, "shifted", False)
    output = model.model(x)
    mll = VariationalELBO(model.likelihood, model.model, 11)
    weight = x.new_tensor(0.7, requires_grad=True)
    value = mll(output, y.squeeze(-1), likelihood_weight=weight)
    with torch.no_grad():
        model.likelihood.offset.add_(0.3)
        model.likelihood.noise = 0.1
    torch.testing.assert_close(
        mll(output, y.squeeze(-1), likelihood_weight=weight), value, atol=0, rtol=0
    )
    leaves = (x, y, weight, *model.parameters())
    assert all(torch.isfinite(v).all() for v in derivatives(value, leaves))
    clone = copy.deepcopy(model)
    assert clone.likelihood is not model.likelihood
    torch.testing.assert_close(clone.posterior(x).mean, model.posterior(x).mean)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("family", ["gaussian", "multitask_diagonal", "multitask_correlated"])
def test_standard_actual_gaussian_elbo_uses_marginals_without_training_square_covariance(
    device, family
):
    (model, x, y), _ = pair(device, family, False)
    output = model.model(x)

    def forbidden():
        raise AssertionError("Gaussian expected_log_prob only needs q(f) marginal variances")

    output._covariance = forbidden
    target = y if model.num_outputs > 1 else y.squeeze(-1)
    value = VariationalELBO(model.likelihood, model.model, 11)(output, target)
    assert torch.isfinite(value).all()


@pytest.mark.parametrize("device", DEVICES)
def test_actual_gaussian_module_prediction_and_training_stay_on_device(device, monkeypatch):
    (model, x, y), _ = pair(device, "multitask_correlated", True)
    query = x.new_tensor([[0.24], [0.43], [0.66], [0.81]], requires_grad=True)

    def forbidden(*args, **kwargs):
        raise AssertionError("device Gaussian evaluation must not copy numerical data to the host")

    with monkeypatch.context() as patch:
        for name in ("cpu", "numpy", "tolist"):
            patch.setattr(torch.Tensor, name, forbidden)
        for _ in range(2):
            output = model.model(x)
            value = VariationalELBO(model.likelihood, model.model, 11)(output, y)
            posterior = model.posterior(query, observation_noise=True)
            first = torch.autograd.grad(
                value.sum() + posterior.mean.square().sum() + posterior.variance.sum(),
                (x, y, query, *model.parameters()),
                create_graph=True,
            )
            higher = torch.autograd.grad(
                sum(g.square().sum() for g in first), (x, y, query, *model.parameters())
            )
            assert all(torch.isfinite(v).all() for v in (*first, *higher))

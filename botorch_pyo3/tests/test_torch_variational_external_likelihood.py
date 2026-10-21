# ruff: noqa: E402
"""Supplied scalar observation modules keep their actual law and leaf graphs."""

import pytest

torch = pytest.importorskip("torch")
gpytorch = pytest.importorskip("gpytorch")
pytest.importorskip("botorch")

from botorch.models import SingleTaskVariationalGP as ReferenceGP
from gpytorch.mlls import VariationalELBO as ReferenceELBO

from botorch_rs.mlls import VariationalELBO
from botorch_rs.models import SingleTaskVariationalGP

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
FAMILIES = [
    "LaplaceLikelihood",
    "StudentTLikelihood",
    "BetaLikelihood",
    "PoissonLikelihood",
    "ShiftedStudentTLikelihood",
    "JointLinearLikelihood",
]


class ShiftedStudentTLikelihood(gpytorch.likelihoods.StudentTLikelihood):
    def __init__(self, batch_shape=()):
        super().__init__(batch_shape=batch_shape)
        self.offset = torch.nn.Parameter(torch.full((*batch_shape, 1), 0.15))

    def forward(self, samples, **kwargs):
        return super().forward(samples + self.offset, **kwargs)

    def expected_log_prob(self, target, function, likelihood_weight=1, **kwargs):
        return super().expected_log_prob(target, function, **kwargs) * likelihood_weight


class JointLinearLikelihood(gpytorch.likelihoods.Likelihood):
    """A supplied likelihood whose expectation needs off-diagonal covariance."""

    def __init__(self, batch_shape=()):
        super().__init__()
        self.raw_scale = torch.nn.Parameter(torch.full((*batch_shape, 1), -0.5))

    @staticmethod
    def weights(value):
        n = value.shape[-1]
        return torch.eye(n, dtype=value.dtype, device=value.device) + 0.15

    def forward(self, samples, **kwargs):
        return torch.distributions.Normal(samples @ self.weights(samples).mT, self.raw_scale.exp())

    def expected_log_prob(self, target, function, **kwargs):
        weights = self.weights(function.mean)
        mean = function.mean @ weights.mT
        variance = (weights @ function.covariance_matrix @ weights.mT).diagonal(dim1=-2, dim2=-1)
        return (
            -0.5 * ((target - mean).square() + variance) / self.raw_scale.exp().square()
            - self.raw_scale
            - 0.5 * torch.log(target.new_tensor(2 * torch.pi))
        )


class DenseRBF(gpytorch.kernels.Kernel):
    has_lengthscale = True

    def forward(self, x1, x2, diag=False, **kwargs):
        if diag:
            return (-0.5 * ((x1 - x2) / self.lengthscale).square().sum(-1)).exp()
        distance = (x1.unsqueeze(-2) - x2.unsqueeze(-3)) / self.lengthscale.unsqueeze(-2)
        return (-0.5 * distance.square().sum(-1)).exp()


def pair(device, family, batch=False):
    result = []
    for reference in (False, True):
        x = torch.tensor([[0.1], [0.3], [0.5], [0.7], [0.9]], device=device, dtype=torch.double)
        y = x.new_tensor([0.2, 0.7, 0.4, 0.8, 0.3]).unsqueeze(-1)
        if family == "PoissonLikelihood":
            y = y.new_tensor([0.0, 2.0, 1.0, 3.0, 1.0]).unsqueeze(-1)
        if batch:
            x = x.expand(2, 5, 1).clone()
            y = y.expand(2, 5, 1).clone()
        x.requires_grad_()
        if family != "PoissonLikelihood":
            y.requires_grad_()
        likelihood_class = globals().get(family) or getattr(gpytorch.likelihoods, family)
        likelihood = likelihood_class(
            **({} if family == "PoissonLikelihood" else {"batch_shape": x.shape[:-2]})
        ).to(x)
        kernel = DenseRBF(batch_shape=x.shape[:-2]).to(x)
        kernel.lengthscale = 0.5
        distribution = gpytorch.variational.CholeskyVariationalDistribution(
            3, batch_shape=x.shape[:-2], mean_init_std=0
        ).to(x)
        cls = ReferenceGP if reference else SingleTaskVariationalGP
        model = cls(
            x,
            y,
            likelihood=likelihood,
            covar_module=kernel,
            mean_module=gpytorch.means.ZeroMean(batch_shape=x.shape[:-2]),
            inducing_points=x.detach()[..., [0, 2, 4], :],
            variational_distribution=distribution,
            outcome_transform=None,
            **({} if reference else {"backend": "torch"}),
        ).to(x)
        assert model.likelihood is likelihood
        with torch.no_grad():
            strategy = model.model.variational_strategy
            strategy.variational_params_initialized.fill_(1)
            distribution.variational_mean.copy_(x.new_tensor([-0.3, 0.1, 0.4]))
            distribution.chol_variational_covar.copy_(
                torch.eye(3, device=x.device, dtype=x.dtype) * 0.8
            )
        result.append((model, x, y))
    return result


def dense_elbo(model, x, y):
    """Independent dense Gaussian algebra and direct likelihood quadrature."""
    strategy = model.model.variational_strategy
    points = strategy.inducing_points
    kernel = model.model.covar_module
    kzz = kernel.forward(points, points) + strategy.jitter_val * torch.eye(
        points.shape[-2], device=x.device, dtype=x.dtype
    )
    interpolation = torch.linalg.solve_triangular(
        torch.linalg.cholesky(kzz), kernel.forward(points, x), upper=False
    )
    distribution = strategy._variational_distribution
    mu, root = distribution.variational_mean, distribution.chol_variational_covar.tril()
    eye = torch.eye(points.shape[-2], device=x.device, dtype=x.dtype)
    covariance = root @ root.mT
    mean = (interpolation.mT @ mu.unsqueeze(-1)).squeeze(-1)
    variance = (
        1 + strategy.jitter_val + (interpolation * ((covariance - eye) @ interpolation)).sum(-2)
    )
    if isinstance(model.likelihood, JointLinearLikelihood):
        update = (interpolation * ((covariance - eye) @ interpolation)).sum(-2)
        q_covariance = kernel.forward(x, x) + torch.diag_embed(update + strategy.jitter_val)
        weights = model.likelihood.weights(mean)
        noise = model.likelihood.raw_scale.exp()
        expected_mean = mean @ weights.mT
        expected_variance = (weights @ q_covariance @ weights.mT).diagonal(dim1=-2, dim2=-1)
        likelihood = (
            -0.5 * ((y.squeeze(-1) - expected_mean).square() + expected_variance) / noise.square()
            - noise.log()
            - 0.5 * torch.log(x.new_tensor(2 * torch.pi))
        ).mean(-1)
    else:
        nodes = model.likelihood.quadrature.locations
        weights = model.likelihood.quadrature.weights
        shape = (-1,) + (1,) * mean.ndim
        samples = mean + (2 * variance).sqrt() * nodes.reshape(shape)
        likelihood = (
            model.likelihood.forward(samples).log_prob(y.squeeze(-1)) * weights.reshape(shape)
        ).sum(0).mean(-1) / torch.pi**0.5
    kl = 0.5 * (
        mu.square().sum(-1)
        + covariance.diagonal(dim1=-2, dim2=-1).sum(-1)
        - mu.shape[-1]
        - 2 * root.diagonal(dim1=-2, dim2=-1).log().sum(-1)
    )
    return likelihood - 0.7 / 11 * kl


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("batch", [False, True])
def test_supplied_scalar_likelihood_values_and_all_leaf_graphs(device, family, batch):
    terms, first, higher = [], [], []
    for reference, (model, x, y) in enumerate(pair(device, family, batch)):
        output = model.model(x)
        cls = ReferenceELBO if reference else VariationalELBO
        values = cls(model.likelihood, model.model, num_data=11, beta=0.7, combine_terms=False)(
            output, y.squeeze(-1)
        )
        terms.append(values)
        parameters = dict(model.named_parameters())
        leaves = (
            x,
            *((y,) if y.requires_grad else ()),
            *(parameters[name] for name in sorted(parameters)),
        )
        gradients = torch.autograd.grad(
            (values[0] - values[1] + values[2]).sum(), leaves, create_graph=True, retain_graph=True
        )
        first.append(gradients)
        if reference:
            # The source model's solve/root custom backwards are not a full
            # second-order oracle. Its actual q/noise/kernel parameters enter
            # an independent dense Gaussian calculation with direct quadrature.
            dense_first = torch.autograd.grad(
                dense_elbo(model, x, y).sum(), leaves, create_graph=True
            )
            for a, b in zip(gradients, dense_first, strict=True):
                torch.testing.assert_close(a, b, rtol=3e-7, atol=3e-8)
            higher.append(torch.autograd.grad(sum(g.square().sum() for g in dense_first), leaves))
        else:
            higher.append(torch.autograd.grad(sum(g.square().sum() for g in gradients), leaves))
    for a, b in zip(*terms, strict=True):
        torch.testing.assert_close(a, b, rtol=2e-8, atol=2e-9)
    for actual, expected in zip(*first, strict=True):
        torch.testing.assert_close(actual, expected, rtol=3e-7, atol=3e-8)
    for actual, expected in zip(*higher, strict=True):
        torch.testing.assert_close(actual, expected, rtol=3e-6, atol=3e-7)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("family", FAMILIES)
def test_observed_marginal_preserves_likelihood_distribution_and_sampling_graph(device, family):
    from botorch_rs.models._torch_variational_external_likelihood import gaussian

    model, x, _ = pair(device, family)[0]
    query = x.new_tensor([[0.2], [0.6]], requires_grad=True)
    model.eval()
    latent = model.model(query)
    devices = [torch.cuda.current_device()] if device == "cuda" else []
    with torch.random.fork_rng(devices=devices), gpytorch.settings.num_likelihood_samples(4):
        torch.manual_seed(71)
        expected = model.likelihood(gaussian(latent))
        torch.manual_seed(71)
        actual = model.posterior(query, observation_noise=True)
    assert type(actual.distribution) is type(expected)
    assert actual.device == query.device and actual.dtype == query.dtype
    torch.testing.assert_close(actual.mean, expected.mean, rtol=2e-9, atol=2e-10)
    torch.testing.assert_close(actual.variance, expected.variance, rtol=2e-9, atol=2e-10)
    # An independent dense Cholesky sample verifies covariance-root Hessians
    # without relying on linear_operator's incomplete sampling double backward.
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(71)
        normals = torch.randn(4, 2, dtype=query.dtype, device=query.device)
    samples = latent.mean + (
        torch.linalg.cholesky(latent.covariance_matrix) @ normals.unsqueeze(-1)
    ).squeeze(-1)
    independent = model.likelihood.forward(samples)
    leaves = (query, *model.parameters())
    actual_first = torch.autograd.grad(
        actual.mean.square().sum() + 0.1 * actual.variance.square().sum(),
        leaves,
        create_graph=True,
        retain_graph=True,
    )
    expected_first = torch.autograd.grad(
        independent.mean.square().sum() + 0.1 * independent.variance.square().sum(),
        leaves,
        create_graph=True,
        retain_graph=True,
    )
    actual_higher = torch.autograd.grad(
        sum(g.square().sum() for g in actual_first), leaves, retain_graph=True
    )
    expected_higher = torch.autograd.grad(sum(g.square().sum() for g in expected_first), leaves)
    for a, b in zip(
        (*actual_first, *actual_higher), (*expected_first, *expected_higher), strict=True
    ):
        torch.testing.assert_close(a, b, rtol=3e-7, atol=3e-8)


@pytest.mark.parametrize("device", DEVICES)
def test_saved_output_keeps_likelihood_tensor_state_and_original_parameter_leaf(device):
    model, x, y = pair(device, "StudentTLikelihood")[0]
    mll = VariationalELBO(model.likelihood, model.model, num_data=11)
    output = model.model(x)
    before = mll(output, y.squeeze(-1))
    original = model.likelihood.raw_noise
    with torch.no_grad():
        original.add_(0.4)
    saved = mll(output, y.squeeze(-1))
    torch.testing.assert_close(saved, before, rtol=0, atol=0)
    fresh = mll(model.model(x), y.squeeze(-1))
    assert not torch.allclose(fresh, saved)
    gradient = torch.autograd.grad(saved.sum(), original, create_graph=True)[0]
    assert torch.isfinite(gradient).all() and gradient.abs().max() > 0
    assert model.likelihood.raw_noise is original


@pytest.mark.parametrize("device", DEVICES)
def test_scalar_quadrature_needs_no_full_training_covariance_or_host_copy(device, monkeypatch):
    from botorch_rs.models._torch_backend_variational import _TorchVariationalOutput

    model, x, y = pair(device, "BetaLikelihood")[0]

    def forbidden(*args, **kwargs):
        raise AssertionError(
            "Scalar likelihood inference materialized a full covariance or copied to host"
        )

    monkeypatch.setattr(_TorchVariationalOutput, "_blocks", forbidden)
    for name in ("cpu", "tolist", "numpy"):
        monkeypatch.setattr(torch.Tensor, name, forbidden)
    output = model.model(x)
    value = VariationalELBO(model.likelihood, model.model, num_data=11)(output, y.squeeze(-1))
    leaves = (x, y, *model.parameters())
    gradients = torch.autograd.grad(value.sum(), leaves, create_graph=True)
    higher = torch.autograd.grad(sum(g.square().sum() for g in gradients), leaves)
    assert all(torch.isfinite(v).all() for v in (*gradients, *higher))


@pytest.mark.parametrize("device", DEVICES)
def test_dense_higher_oracle_matches_convergent_unmodified_source_first_gradients(device):
    model, x, y = pair(device, "StudentTLikelihood")[1]
    names = sorted(dict(model.named_parameters()))
    leaves = (x, y, *(dict(model.named_parameters())[name] for name in names))
    mll = ReferenceELBO(model.likelihood, model.model, num_data=11, beta=0.7)
    first = torch.autograd.grad(mll(model.model(x), y.squeeze(-1)).sum(), leaves)
    dense_first = torch.autograd.grad(dense_elbo(model, x, y).sum(), leaves, create_graph=True)
    higher = torch.autograd.grad(sum(g.square().sum() for g in dense_first), leaves)
    state = tuple(value.detach().clone() for value in leaves)
    direction = tuple(2 * value.detach() for value in first)

    def source_gradient(step):
        candidate, xx, yy = pair(device, "StudentTLikelihood")[1]
        candidate_parameters = dict(candidate.named_parameters())
        values = (xx, yy, *(candidate_parameters[name] for name in names))
        with torch.no_grad():
            for value, previous, tangent in zip(values, state, direction, strict=True):
                value.copy_(previous + step * tangent)
        loss = ReferenceELBO(candidate.likelihood, candidate.model, num_data=11, beta=0.7)(
            candidate.model(xx), yy.squeeze(-1)
        )
        return torch.autograd.grad(loss.sum(), values)

    estimates = []
    for step in (2e-5, 1e-5):
        plus, minus = source_gradient(step), source_gradient(-step)
        estimates.append(tuple((a - b) / (2 * step) for a, b in zip(plus, minus, strict=True)))
    for actual, larger, smaller in zip(higher, *estimates, strict=True):
        torch.testing.assert_close(larger, smaller, rtol=2e-5, atol=2e-6)
        torch.testing.assert_close(actual, smaller, rtol=2e-5, atol=2e-6)


@pytest.mark.parametrize("device", DEVICES)
def test_elbo_likelihood_keyword_tensor_retains_mixed_graphs(device):
    actual_model, source_model = pair(device, "ShiftedStudentTLikelihood")
    model, x, y = actual_model
    reference, xx, yy = source_model
    weight = x.new_tensor(0.7, requires_grad=True)
    value = VariationalELBO(model.likelihood, model.model, num_data=11)(
        model.model(x), y.squeeze(-1), likelihood_weight=weight
    )
    output = reference.model(xx)
    source = ReferenceELBO(reference.likelihood, reference.model, num_data=11)(
        output, yy.squeeze(-1), likelihood_weight=weight
    )
    source_likelihood = ReferenceELBO(
        reference.likelihood, reference.model, num_data=11, combine_terms=False
    )(output, yy.squeeze(-1))[0]
    torch.testing.assert_close(value, source, rtol=2e-9, atol=2e-10)
    weight_gradient = torch.autograd.grad(value.sum(), weight, create_graph=True)[0]
    torch.testing.assert_close(weight_gradient, source_likelihood.sum(), rtol=2e-9, atol=2e-10)
    actual_mixed = torch.autograd.grad(weight_gradient, x)[0]
    source_mixed = torch.autograd.grad(source_likelihood.sum(), xx)[0]
    torch.testing.assert_close(actual_mixed, source_mixed, rtol=3e-7, atol=3e-8)

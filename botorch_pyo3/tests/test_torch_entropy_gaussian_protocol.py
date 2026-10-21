"""Entropy accepts caller Gaussian posteriors and their covariance ordering."""

# ruff: noqa: E402
from __future__ import annotations

import copy

import pytest

torch = pytest.importorskip("torch")

from botorch_rs.acquisition.multi_objective.joint_entropy_search import (
    qLowerBoundMultiObjectiveJointEntropySearch,
)
from botorch_rs.acquisition.multi_objective.max_value_entropy_search import (
    qLowerBoundMultiObjectiveMaxValueEntropySearch,
)
from botorch_rs.models.model import Model
from botorch_rs.posteriors._torch_gaussian import gaussian_posterior
from botorch_rs.posteriors.gpytorch import GPyTorchPosterior
from botorch_rs.sampling.normal import SobolQMCNormalSampler

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


class GaussianLaw:
    """Explicit independent-output GP with ordinary dense Gaussian conditioning."""

    _posterior_type = GPyTorchPosterior
    _torch_only = False

    def __init__(self, baseline, interleaved):
        super().__init__()
        self.register_buffer("baseline", baseline)
        self.register_buffer("condition_X", None)
        self.register_buffer("condition_Y", None)
        self.coefficients = torch.nn.Parameter(baseline.new_tensor([0.7, -0.4]))
        self.scales = torch.nn.Parameter(baseline.new_tensor([0.6, 0.9]))
        self.lengths = torch.nn.Parameter(baseline.new_tensor([0.3, 0.5]))
        self.interleaved = interleaved

    @property
    def train_inputs(self):
        return (self.baseline,)

    @property
    def num_outputs(self):
        return 2

    @property
    def batch_shape(self):
        return torch.Size() if self.condition_X is None else self.condition_X.shape[:-2]

    def condition_on_observations(self, X, Y, **kwargs):
        result = copy.deepcopy(self)
        result.condition_X, result.condition_Y = X, Y
        return result

    def _prior(self, X):
        x = X[..., 0]
        mean = self.coefficients[..., None] * x.unsqueeze(-2)
        delta = (x.unsqueeze(-1) - x.unsqueeze(-2)).unsqueeze(-3)
        covariance = (
            self.scales[..., None, None].square()
            * (-0.5 * (delta / self.lengths[..., None, None]).square()).exp()
        )
        return mean, covariance

    def posterior(self, X, observation_noise=False, posterior_transform=None, **kwargs):
        if self.condition_X is None:
            mean, covariance = self._prior(X)
        else:
            batch = torch.broadcast_shapes(X.shape[:-2], self.condition_X.shape[:-2])
            points = X.expand(*batch, *X.shape[-2:])
            conditions = self.condition_X.expand(*batch, *self.condition_X.shape[-2:])
            mean, covariance = self._prior(torch.cat((conditions, points), -2))
            n = conditions.shape[-2]
            observations = self.condition_Y.transpose(-1, -2)
            train = covariance[..., :n, :n] + 0.05 * torch.eye(n, dtype=X.dtype, device=X.device)
            cross = covariance[..., n:, :n]
            mean = mean[..., n:] + (
                cross @ torch.linalg.solve(train, (observations - mean[..., :n]).unsqueeze(-1))
            ).squeeze(-1)
            covariance = covariance[..., n:, n:] - cross @ torch.linalg.solve(train, cross.mT)
        q = X.shape[-2]
        if observation_noise:
            covariance = covariance + 0.05 * torch.eye(q, dtype=X.dtype, device=X.device)
        axes = "...mij,mn->...imjn" if self.interleaved else "...mij,mn->...minj"
        dense = torch.einsum(axes, covariance, torch.eye(2, dtype=X.dtype, device=X.device))
        dense = dense.reshape(*covariance.shape[:-3], q * 2, q * 2)
        public_mean = mean.transpose(-1, -2)
        if self._torch_only:
            flat_mean = public_mean.flatten(-2) if self.interleaved else mean.flatten(-2)
            distribution = torch.distributions.MultivariateNormal(flat_mean, dense)
            distribution._interleaved = self.interleaved
            result = PublicGaussianPosterior(public_mean, distribution)
        else:
            from gpytorch.distributions import MultitaskMultivariateNormal

            distribution = MultitaskMultivariateNormal(
                public_mean, dense, interleaved=self.interleaved
            )
            result = self._posterior_type(distribution)
        return result if posterior_transform is None else posterior_transform(result)


class CallerModel(GaussianLaw, Model):
    pass


class PublicGaussianPosterior:
    def __init__(self, mean, distribution):
        self.mean, self.distribution = mean, distribution


class TorchOnlyModel(CallerModel):
    _torch_only = True


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("interleaved", [False, True])
def test_gaussian_adapter_covariance_samples_and_all_leaf_hvps(device, interleaved, monkeypatch):
    baseline = torch.tensor([[0.1], [0.4], [0.8]], dtype=torch.double, device=device)
    model = TorchOnlyModel(baseline, interleaved)
    query = baseline.new_tensor(
        [[[0.17], [0.5], [0.7]], [[0.28], [0.6], [0.78]]], requires_grad=True
    )
    base = (
        torch.linspace(-0.8, 0.9, 48, dtype=baseline.dtype, device=device)
        .reshape(4, 2, 3, 2)
        .requires_grad_()
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("Gaussian posterior adaptation converted device data to host.")

    for method in ("cpu", "numpy", "tolist"):
        monkeypatch.setattr(torch.Tensor, method, forbidden)
    posterior = gaussian_posterior(model.posterior(query))
    prior_mean, prior_covariance = model._prior(query)
    expected_covariance = torch.stack(
        [
            torch.block_diag(prior_covariance[batch, 0], prior_covariance[batch, 1])
            for batch in range(2)
        ]
    )
    torch.testing.assert_close(posterior.mean, prior_mean.transpose(-1, -2))
    torch.testing.assert_close(posterior.distribution.covariance_matrix, expected_covariance)
    actual = posterior.rsample_from_base_samples((4,), base)
    expected = torch.stack(
        [
            prior_mean[..., task, :]
            + (
                torch.linalg.cholesky(prior_covariance[..., task, :, :])
                @ base[..., task].unsqueeze(-1)
            ).squeeze(-1)
            for task in range(2)
        ],
        -1,
    )
    torch.testing.assert_close(actual, expected)
    variables = (query, base, *model.parameters())
    own_first = torch.autograd.grad(actual.sin().sum(), variables, create_graph=True)
    expected_first = torch.autograd.grad(expected.sin().sum(), variables, create_graph=True)
    own_second = torch.autograd.grad(sum(item.square().sum() for item in own_first), variables)
    expected_second = torch.autograd.grad(
        sum(item.square().sum() for item in expected_first), variables
    )
    for own, reference in zip(
        (*own_first, *own_second), (*expected_first, *expected_second), strict=True
    ):
        torch.testing.assert_close(own, reference, rtol=3e-11, atol=3e-12)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("interleaved", [False, True])
@pytest.mark.parametrize("joint", [False, True])
@pytest.mark.parametrize("estimator", ["0", "LB", "LB2", "MC"])
def test_caller_public_gaussian_posterior_matches_source(device, interleaved, joint, estimator):
    pytest.importorskip("botorch")
    from botorch.acquisition.multi_objective.joint_entropy_search import (
        qLowerBoundMultiObjectiveJointEntropySearch as SourceJES,
    )
    from botorch.acquisition.multi_objective.max_value_entropy_search import (
        qLowerBoundMultiObjectiveMaxValueEntropySearch as SourceMES,
    )
    from botorch.models.model import Model as SourceModel
    from botorch.posteriors.gpytorch import GPyTorchPosterior as SourcePosterior
    from botorch.sampling.normal import SobolQMCNormalSampler as SourceSampler

    class SourceLaw(GaussianLaw, SourceModel):
        _posterior_type = SourcePosterior

    baseline = torch.tensor([[0.1], [0.4], [0.8]], dtype=torch.double, device=device)
    model, source_model = CallerModel(baseline, interleaved), SourceLaw(baseline, interleaved)
    cells = baseline.new_tensor([[[[-3.0, -3.0]], [[1.1, 1.2]]], [[[-3.0, -3.0]], [[1.0, 1.1]]]])
    options = dict(estimation_type=estimator, num_samples=8)
    if joint:
        sets = baseline.new_tensor([[[0.2], [0.5]], [[0.3], [0.6]]])
        fronts = baseline.new_tensor([[[0.8, 0.9], [0.9, 0.8]], [[0.85, 0.95], [0.95, 0.85]]])
        actual = qLowerBoundMultiObjectiveJointEntropySearch(model, sets, fronts, cells, **options)
        source = SourceJES(source_model, sets, fronts, cells, **options)
    else:
        actual = qLowerBoundMultiObjectiveMaxValueEntropySearch(model, cells, **options)
        source = SourceMES(source_model, cells, **options)
    actual.sampler = SobolQMCNormalSampler((8,), seed=43)
    source.sampler = SourceSampler(torch.Size((8,)), seed=43)
    query = baseline.new_tensor([[[0.17], [0.7]], [[0.28], [0.78]]], requires_grad=True)
    expected = source(query)
    assert torch.isfinite(expected).all()
    value = actual(query)
    torch.testing.assert_close(value, expected, rtol=3e-7, atol=3e-8)
    variables = (query, *model.parameters())
    expected_variables = (query, *source_model.parameters())
    if joint:
        # Initial Gaussian entropy is independent of this specific mean.
        variables = (query, model.scales, model.lengths, *actual.conditional_model.parameters())
        expected_variables = (
            query,
            source_model.scales,
            source_model.lengths,
            *source.conditional_model.parameters(),
        )
    gradient = torch.autograd.grad(value.sum(), variables, create_graph=True)
    expected_gradient = torch.autograd.grad(expected.sum(), expected_variables)
    for own, reference in zip(gradient, expected_gradient, strict=True):
        torch.testing.assert_close(own, reference, rtol=3e-5, atol=3e-6)
    second = torch.autograd.grad(sum(item.square().sum() for item in gradient), variables)
    assert all(torch.isfinite(item).all() for item in second)

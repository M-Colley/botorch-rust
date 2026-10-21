"""Public task-model subclasses retain their virtual Gaussian prior on device."""

# ruff: noqa: E402
from __future__ import annotations

import copy
import pickle

import pytest

torch = pytest.importorskip("torch")

from test_torch_fully_bayesian_multitask import DEVICES, _data, _samples

from botorch_rs.models.fully_bayesian_multitask import SaasFullyBayesianMultiTaskGP
from botorch_rs.posteriors._torch_backend import TorchGPyTorchPosterior


class TaskSubclass(SaasFullyBayesianMultiTaskGP):
    def __init__(self, *args, label="retained", **kwargs):
        super().__init__(*args, **kwargs)
        self.label = label
        self.prior_scale = torch.nn.Parameter(self._train_X.new_tensor(0.8))

    def forward(self, X):
        prior = super().forward(X)
        mean = prior.mean + 0.1 * X[..., 0].sin()
        return TorchGPyTorchPosterior(
            mean.unsqueeze(-1), prior.covariance_matrix * self.prior_scale.square()
        ).distribution

    def posterior(self, X, **kwargs):
        self.posterior_called = True
        return super().posterior(X, **kwargs)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("fixed", [False, True])
@pytest.mark.parametrize("backend", ["torch", "auto"])
def test_task_subclass_keeps_class_hooks_prior_and_all_graphs(device, fixed, backend):
    X, Y, noise = _data(device, 1, gradients=True)
    model = TaskSubclass(X, Y, 1, noise if fixed else None, backend=backend, label="custom")
    assert isinstance(model, TaskSubclass) and isinstance(model, torch.nn.Module)
    assert model._tensor_backend == "torch" and model.label == "custom"
    samples = _samples(X)
    model.load_mcmc_samples(samples)
    model.eval()
    query = X.new_tensor([[0.22, 3.0, 0.41], [0.63, 7.0, 0.71]], requires_grad=True)
    actual = model.posterior(query, observation_noise=True)
    assert model.posterior_called

    # Independently form a transformed joint Gaussian; bypass the model's
    # virtual forward to make omission of the custom mean/covariance visible.
    train, targets, fixed_noise = model._training_values()
    test = query.unsqueeze(-3).expand(model.num_mcmc_samples, *query.shape)
    mapped_train, mapped_test = model._mapped_inputs(train), model._mapped_inputs(test)
    mean_train = model._mean(mapped_train) + 0.1 * train[..., 0].sin()
    mean_test = model._mean(mapped_test) + 0.1 * test[..., 0].sin()
    scale = model.prior_scale.square()
    noise_diagonal = model._training_noise(targets, fixed_noise)
    observed = model._covariance(mapped_train, mapped_train) * scale + torch.diag_embed(
        noise_diagonal
    )
    cross = model._covariance(mapped_train, mapped_test) * scale
    expected_mean = mean_test + (
        cross.transpose(-1, -2) @ torch.linalg.solve(observed, (targets - mean_train).unsqueeze(-1))
    ).squeeze(-1)
    expected_cov = model._covariance(mapped_test, mapped_test) * scale - cross.transpose(
        -1, -2
    ) @ torch.linalg.solve(observed, cross)
    query_noise = (
        model._fixed_noise_at(mapped_test, fixed_noise) if fixed else model._noise_at(mapped_test)
    )
    expected_cov = expected_cov + torch.diag_embed(query_noise)
    torch.testing.assert_close(actual.mean.squeeze(-1), expected_mean, atol=3e-13, rtol=3e-12)
    torch.testing.assert_close(
        actual.distribution.covariance_matrix, expected_cov, atol=3e-13, rtol=3e-12
    )
    leaves = (X, Y, query, *((noise,) if fixed else ()), *model.parameters())

    def derivatives(mean, covariance):
        first = torch.autograd.grad(
            mean.sin().sum() + covariance.square().sum(), leaves, create_graph=True
        )
        return (*first, *torch.autograd.grad(sum(g.square().sum() for g in first), leaves))

    for a, e in zip(
        derivatives(actual.mean.squeeze(-1), actual.distribution.covariance_matrix),
        derivatives(expected_mean, expected_cov),
        strict=True,
    ):
        torch.testing.assert_close(a, e, atol=3e-10, rtol=3e-9)
    expected = model.posterior(query).mean
    for restored in (copy.deepcopy(model), pickle.loads(pickle.dumps(model))):
        assert type(restored) is type(model) and isinstance(restored, TaskSubclass)
        assert restored.label == "custom"
        torch.testing.assert_close(restored.posterior(query).mean, expected, atol=0, rtol=0)
    replacement = TaskSubclass(X, Y, 1, noise if fixed else None, backend="torch")
    replacement.load_state_dict(model.state_dict())
    torch.testing.assert_close(replacement.posterior(query).mean, expected, atol=0, rtol=0)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("transforms", [False, True])
def test_task_subclass_virtual_prior_matches_unmodified_source(device, transforms):
    gpytorch = pytest.importorskip("gpytorch")
    pytest.importorskip("botorch")
    from botorch.models.fully_bayesian_multitask import SaasFullyBayesianMultiTaskGP as Source
    from botorch.models.transforms.input import Normalize as SourceNormalize
    from botorch.models.transforms.outcome import Standardize as SourceStandardize
    from fully_bayesian_helpers import jax_x64

    from botorch_rs.models.transforms.input import Normalize
    from botorch_rs.models.transforms.outcome import Standardize

    class SourceSubclass(Source):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.prior_scale = torch.nn.Parameter(self.train_inputs[0].new_tensor(0.8))

        def forward(self, X):
            prior = super().forward(X)
            return gpytorch.distributions.MultivariateNormal(
                prior.mean + 0.1 * X[..., 0].sin(),
                prior.lazy_covariance_matrix * self.prior_scale.square(),
            )

    X, Y, _ = _data(device, 1)
    bounds = X.new_tensor([[0.0, 0.0, 0.0], [2.0, 1.0, 2.0]])
    model = TaskSubclass(
        X,
        Y,
        1,
        backend="torch",
        **(
            {
                "input_transform": Normalize(3, indices=[0, 2], bounds=bounds),
                "outcome_transform": Standardize(1),
            }
            if transforms
            else {}
        ),
    )
    with jax_x64():
        source = SourceSubclass(
            X,
            Y,
            1,
            **(
                {
                    "input_transform": SourceNormalize(3, indices=[0, 2], bounds=bounds),
                    "outcome_transform": SourceStandardize(1),
                }
                if transforms
                else {}
            ),
        )
    samples = _samples(X)
    model.load_mcmc_samples(samples)
    source.load_mcmc_samples(samples)
    query = X.new_tensor([[0.22, 0.41], [0.63, 0.71]], requires_grad=True)
    with gpytorch.settings.detach_test_caches(False):
        expected = source.posterior(query, observation_noise=True)
        reference_gradient = torch.autograd.grad(
            expected.mean.sum() + expected.variance.sum(), query
        )[0]
    actual = model.posterior(query, observation_noise=True)
    torch.testing.assert_close(actual.mean, expected.mean, atol=3e-13, rtol=3e-12)
    torch.testing.assert_close(
        actual.distribution.covariance_matrix,
        expected.distribution.covariance_matrix,
        atol=3e-13,
        rtol=3e-12,
    )
    first = torch.autograd.grad(actual.mean.sum() + actual.variance.sum(), query)[0]
    torch.testing.assert_close(first, reference_gradient, atol=3e-11, rtol=3e-10)

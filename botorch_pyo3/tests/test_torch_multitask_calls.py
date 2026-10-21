"""ExactGP call semantics, virtual multitask priors and actual Gaussian noise."""

# ruff: noqa: E402
from __future__ import annotations

import copy
import pickle

import pytest

torch = pytest.importorskip("torch")

from botorch_rs.mlls import ExactMarginalLogLikelihood
from botorch_rs.models import MultiTaskGP
from botorch_rs.models.contextual_multioutput import LCEMGP
from botorch_rs.models.transforms.input import Normalize
from botorch_rs.models.transforms.outcome import Standardize
from botorch_rs.posteriors._torch_backend import TorchGPyTorchPosterior

DEVICES = [
    "cpu",
    pytest.param(
        "cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
    ),
]


def _data(device, gradients=False):
    X = torch.tensor(
        [[0.1, 2, 0.3], [0.2, 5, 0.7], [0.4, 2, 0.2], [0.65, 5, 0.8], [0.7, 2, 0.5], [0.9, 5, 0.1]],
        dtype=torch.double,
        device=device,
        requires_grad=gradients,
    )
    Y = X.new_tensor([[-0.4], [0.8], [0.1], [-0.2], [0.9], [0.5]], requires_grad=gradients)
    noise = torch.full_like(Y, 0.04, requires_grad=gradients)
    query = X.new_tensor(
        [[[0.23, 5, 0.4], [0.57, 2, 0.6]], [[0.31, 2, 0.5], [0.69, 5, 0.3]]], requires_grad=True
    )
    return X, Y, noise, query


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("family", ["multitask", "lcem"])
@pytest.mark.parametrize("fixed", [False, True])
@pytest.mark.parametrize("transforms", [False, True])
def test_forward_is_prior_eval_call_conditions_in_transformed_units(
    device, family, fixed, transforms
):
    gpytorch = pytest.importorskip("gpytorch")
    from botorch.models import MultiTaskGP as SourceMT
    from botorch.models.contextual_multioutput import LCEMGP as SourceLCEM
    from botorch.models.transforms.input import Normalize as SourceNormalize
    from botorch.models.transforms.outcome import Standardize as SourceStandardize
    from botorch.settings import propagate_grads

    X, Y, noise, query = _data(device)
    # The source's separately reported inferred-noise remapping defect uses
    # raw labels in HadamardGaussianLikelihood. Keep this call-semantics
    # comparison contiguous; the independent Gaussian controls below retain
    # noncontiguous labels.
    X[:, 1] = (X[:, 1] == 5).to(X)
    with torch.no_grad():
        query[:, :, 1] = (query[:, :, 1] == 5).to(query)
    cls, source_cls = (MultiTaskGP, SourceMT) if family == "multitask" else (LCEMGP, SourceLCEM)
    # Supplied fixed embeddings are independently checked below: the pinned
    # LCEM source omits their width when allocating ARD lengthscales.
    options = {}
    transform = Normalize(3, indices=[0, 2]) if transforms else None
    outcome = Standardize(1) if transforms else None
    source_transform = SourceNormalize(3, indices=[0, 2]) if transforms else None
    source_outcome = SourceStandardize(1) if transforms else None
    torch.manual_seed(41)
    model = cls(
        X,
        Y,
        1,
        noise if fixed else None,
        backend="torch",
        input_transform=transform,
        outcome_transform=outcome,
        **options,
    )
    torch.manual_seed(41)
    source = source_cls(
        X,
        Y,
        1,
        noise if fixed else None,
        input_transform=source_transform,
        outcome_transform=source_outcome,
        **options,
    ).to(X)
    parameters = dict(source.named_parameters())
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            parameters[name].copy_(parameter)
    model.train()
    source.train()
    torch.testing.assert_close(model(X).mean, source(X).mean, atol=3e-11, rtol=3e-10)
    model.eval()
    source.eval()
    transformed = model.transform_inputs(query)
    source_transformed = source.transform_inputs(query)
    prior = model.forward(transformed)
    expected_prior = source.forward(source_transformed)
    torch.testing.assert_close(prior.mean, expected_prior.mean, atol=3e-11, rtol=3e-10)
    torch.testing.assert_close(
        prior.covariance_matrix, expected_prior.covariance_matrix, atol=3e-11, rtol=3e-10
    )
    actual = model(transformed)
    with propagate_grads(True), gpytorch.settings.detach_test_caches(False):
        expected = source(source_transformed)
        source_gradient = torch.autograd.grad(
            expected.mean.sin().sum() + expected.variance.sum(), query, retain_graph=True
        )[0]
    torch.testing.assert_close(actual.mean, expected.mean, atol=3e-11, rtol=3e-10)
    torch.testing.assert_close(
        actual.covariance_matrix, expected.covariance_matrix, atol=3e-11, rtol=3e-10
    )
    gradient = torch.autograd.grad(
        actual.mean.sin().sum() + actual.variance.sum(), query, create_graph=True
    )[0]
    torch.testing.assert_close(gradient, source_gradient, atol=3e-9, rtol=3e-8)
    assert torch.isfinite(torch.autograd.grad(gradient.square().sum(), query)[0]).all()
    assert actual.mean.shape == (2, 2)
    assert not torch.allclose(prior.covariance_matrix, actual.covariance_matrix)
    posterior = model.posterior(query)
    mean = actual.mean.unsqueeze(-1)
    if transforms:
        mean = mean * model.outcome_transform.stdvs + model.outcome_transform.means
    torch.testing.assert_close(posterior.mean, mean, atol=3e-11, rtol=3e-10)


class _Twisted:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.twist = torch.nn.Parameter(self._train_X.new_tensor(0.25))

    def forward(self, X):
        prior = super().forward(X)
        inputs = self.transform_inputs(X) if self.training else X
        feature = inputs[..., 0].square() + 0.3
        mean = prior.mean + self.twist * inputs[..., 0].sin()
        covariance = prior.covariance_matrix + self.twist.square() * feature.unsqueeze(
            -1
        ) * feature.unsqueeze(-2)
        return TorchGPyTorchPosterior(mean.unsqueeze(-1), covariance).distribution


class TwistedMT(_Twisted, MultiTaskGP):
    pass


class TwistedLCEM(_Twisted, LCEMGP):
    pass


def _prior(model, left, right):
    mapped_left, mapped_right = model._mapped_inputs(left), model._mapped_inputs(right)
    mean = model._mean(mapped_left) + model.twist * left[..., 0].sin()
    covariance = model._covariance(mapped_left, mapped_right)
    covariance = covariance + model.twist.square() * (left[..., 0].square() + 0.3).unsqueeze(-1) * (
        right[..., 0].square() + 0.3
    ).unsqueeze(-2)
    return mean, covariance


def _derivatives(loss, leaves):
    first = torch.autograd.grad(loss, leaves, create_graph=True, retain_graph=True)
    return (
        *first,
        *torch.autograd.grad(sum(g.square().sum() for g in first), leaves, retain_graph=True),
    )


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("family", ["multitask", "lcem"])
def test_virtual_prior_controls_posterior_mll_all_graphs_and_serialization(
    device, family, monkeypatch
):
    X, Y, noise, query = _data(device, gradients=True)
    cls = TwistedMT if family == "multitask" else TwistedLCEM
    options = (
        {"context_emb_feature": X.new_tensor([[0.2], [0.8]])}
        if family == "lcem"
        else {"task_covar_prior": None}
    )
    model = cls(
        X,
        Y,
        1,
        noise,
        backend="torch",
        input_transform=Normalize(3, indices=[0, 2]),
        outcome_transform=Standardize(1),
        **options,
    ).eval()
    train, targets, fixed = model._training_values()
    transformed = model.transform_inputs(query)
    mean_train, covariance_train = _prior(model, train, train)
    mean_query, covariance_query = _prior(model, transformed, transformed)
    cross = _prior(model, train, transformed)[1]
    observed = covariance_train + torch.diag_embed(fixed)
    residual = targets - mean_train
    solved = torch.linalg.solve(observed, residual.unsqueeze(-1))
    expected_mean = mean_query + (cross.mT @ solved).squeeze(-1)
    expected_covariance = covariance_query - cross.mT @ torch.linalg.solve(observed, cross)

    def forbidden(*args, **kwargs):
        raise AssertionError("Device multitask calls must not flatten data")

    with monkeypatch.context() as patch:
        for name in ("cpu", "numpy", "tolist"):
            patch.setattr(torch.Tensor, name, forbidden)
        actual = model(transformed)
        torch.testing.assert_close(actual.mean, expected_mean, atol=3e-12, rtol=3e-11)
        torch.testing.assert_close(
            actual.covariance_matrix, expected_covariance, atol=3e-12, rtol=3e-11
        )
        leaves = (X, Y, noise, query, *model.parameters())
        for a, e in zip(
            _derivatives(actual.mean.sin().sum() + actual.covariance_matrix.square().sum(), leaves),
            _derivatives(expected_mean.sin().sum() + expected_covariance.square().sum(), leaves),
            strict=True,
        ):
            torch.testing.assert_close(a, e, atol=3e-9, rtol=3e-8)
    expected_mll = (
        -0.5 * (residual.unsqueeze(-2) @ solved).squeeze()
        - 0.5 * torch.linalg.slogdet(observed)[1]
        - 0.5 * targets.shape[-1] * torch.log(targets.new_tensor(2 * torch.pi))
        + model._log_prior()
    ) / targets.shape[-1]
    model.train()
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    actual_mll = mll()
    torch.testing.assert_close(actual_mll, expected_mll, atol=3e-12, rtol=3e-11)
    torch.testing.assert_close(
        mll(model(X), model.train_targets), expected_mll, atol=3e-12, rtol=3e-11
    )
    leaves = (X, Y, noise, *model.parameters())
    for a, e in zip(
        _derivatives(actual_mll, leaves), _derivatives(expected_mll, leaves), strict=True
    ):
        torch.testing.assert_close(a, e, atol=3e-9, rtol=3e-8)
    model.eval()
    for duplicate in (copy.deepcopy(model), pickle.loads(pickle.dumps(model))):
        assert isinstance(duplicate, cls)
        torch.testing.assert_close(
            duplicate(transformed).mean, expected_mean, atol=3e-12, rtol=3e-11
        )


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("family", ["multitask", "lcem"])
def test_actual_correlated_likelihood_controls_observed_law_and_mll(device, family):
    gpytorch = pytest.importorskip("gpytorch")
    from test_torch_exact_gaussian_likelihood import InputNoise

    X, Y, _, query = _data(device, gradients=True)
    cls = MultiTaskGP if family == "multitask" else LCEMGP
    # A canonical Gaussian marginal need not expose a public .noise attribute.
    likelihood = gpytorch.likelihoods.gaussian_likelihood._GaussianLikelihoodBase(
        InputNoise(X, ())
    ).to(X)
    options = (
        {"context_emb_feature": X.new_tensor([[0.2], [0.8]])}
        if family == "lcem"
        else {"task_covar_prior": None}
    )
    model = cls(
        X,
        Y,
        1,
        torch.full_like(Y, 0.8),
        likelihood=likelihood,
        backend="torch",
        outcome_transform=None,
        **options,
    ).eval()
    assert model.likelihood is likelihood
    mapped_train, mapped_query = model._mapped_inputs(X), model._mapped_inputs(query)

    def noise(z):
        feature = (z[..., 0].sin() + 0.2) * likelihood.noise_covar.amplitude
        return torch.diag_embed(
            likelihood.noise_covar.noise.expand_as(feature)
        ) + feature.unsqueeze(-1) * feature.unsqueeze(-2)

    mean_train, mean_query = model._mean(mapped_train), model._mean(mapped_query)
    observed = model._covariance(mapped_train, mapped_train) + noise(mapped_train)
    cross = model._covariance(mapped_train, mapped_query)
    residual = Y.squeeze(-1) - mean_train
    solved = torch.linalg.solve(observed, residual.unsqueeze(-1))
    expected_mean = mean_query + (cross.mT @ solved).squeeze(-1)
    latent_covariance = model._covariance(
        mapped_query, mapped_query
    ) - cross.mT @ torch.linalg.solve(observed, cross)
    for include_noise in (False, True):
        expected_covariance = latent_covariance + (noise(mapped_query) if include_noise else 0)
        actual = model.posterior(query, observation_noise=include_noise)
        torch.testing.assert_close(actual.mean.squeeze(-1), expected_mean, atol=3e-12, rtol=3e-11)
        torch.testing.assert_close(
            actual.distribution.covariance_matrix, expected_covariance, atol=3e-12, rtol=3e-11
        )
        leaves = (X, Y, query, *model.parameters())
        for a, e in zip(
            _derivatives(
                actual.mean.sin().sum() + actual.distribution.covariance_matrix.square().sum(),
                leaves,
            ),
            _derivatives(expected_mean.sin().sum() + expected_covariance.square().sum(), leaves),
            strict=True,
        ):
            torch.testing.assert_close(a, e, atol=3e-9, rtol=3e-8)
    expected_mll = (
        -0.5 * (residual.unsqueeze(-2) @ solved).squeeze()
        - 0.5 * torch.linalg.slogdet(observed)[1]
        - 0.5 * X.shape[-2] * torch.log(X.new_tensor(2 * torch.pi))
        + model._log_prior()
    ) / X.shape[-2]
    model.train()
    mll = ExactMarginalLogLikelihood(likelihood, model)
    actual_mll = mll()
    torch.testing.assert_close(actual_mll, expected_mll, atol=3e-12, rtol=3e-11)
    torch.testing.assert_close(
        mll(model(X), model.train_targets, model._mapped_inputs(X)),
        expected_mll,
        atol=3e-12,
        rtol=3e-11,
    )
    leaves = (X, Y, *model.parameters())
    for a, e in zip(
        _derivatives(actual_mll, leaves), _derivatives(expected_mll, leaves), strict=True
    ):
        torch.testing.assert_close(a, e, atol=3e-9, rtol=3e-8)


@pytest.mark.parametrize("device", DEVICES)
def test_correlated_likelihood_matches_actual_source_values_and_first_derivatives(device):
    gpytorch = pytest.importorskip("gpytorch")
    from botorch.models import MultiTaskGP as Source
    from botorch.settings import propagate_grads
    from test_torch_exact_gaussian_likelihood import Gaussian, InputNoise

    X, Y, _, query = _data(device, gradients=True)
    likelihood = Gaussian(InputNoise(X, ())).to(X)
    model = MultiTaskGP(
        X,
        Y,
        1,
        likelihood=likelihood,
        backend="torch",
        outcome_transform=None,
        task_covar_prior=None,
    )
    source = Source(
        X, Y, 1, likelihood=copy.deepcopy(likelihood), outcome_transform=None, task_covar_prior=None
    ).to(X)
    with torch.no_grad():
        source_parameters = dict(source.named_parameters())
        for name, parameter in model.named_parameters():
            source_parameters[name].copy_(parameter)
    actual = model.posterior(query, observation_noise=True)
    with propagate_grads(True), gpytorch.settings.detach_test_caches(False):
        expected = source.posterior(query, observation_noise=True)
        parameters = dict(source.named_parameters())
        expected_gradients = torch.autograd.grad(
            expected.mean.sin().sum() + expected.variance.sum(),
            (X, Y, query, *(parameters[name] for name, _ in model.named_parameters())),
            retain_graph=True,
        )
    torch.testing.assert_close(actual.mean, expected.mean, atol=3e-12, rtol=3e-11)
    torch.testing.assert_close(
        actual.distribution.covariance_matrix,
        expected.distribution.covariance_matrix,
        atol=3e-12,
        rtol=3e-11,
    )
    actual_gradients = torch.autograd.grad(
        actual.mean.sin().sum() + actual.variance.sum(), (X, Y, query, *model.parameters())
    )
    for a, e in zip(actual_gradients, expected_gradients, strict=True):
        torch.testing.assert_close(a, e, atol=3e-9, rtol=3e-8)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("learned_noise", [False, True])
def test_actual_fixed_likelihood_conditioning_keeps_old_and_new_noise_graphs(device, learned_noise):
    gpytorch = pytest.importorskip("gpytorch")
    X, Y, noise, query = _data(device, gradients=True)
    likelihood = gpytorch.likelihoods.FixedNoiseGaussianLikelihood(
        noise=noise.squeeze(-1), learn_additional_noise=learned_noise
    ).to(X)
    model = MultiTaskGP(
        X,
        Y,
        1,
        likelihood=likelihood,
        backend="torch",
        input_transform=Normalize(3, indices=[0, 2]),
        outcome_transform=Standardize(1),
        task_covar_prior=None,
    ).eval()
    new_X = X.new_tensor([[0.3, 2, 0.4], [0.8, 5, 0.6]], requires_grad=True)
    new_Y = X.new_tensor([[0.7], [-0.2]], requires_grad=True)
    new_noise = X.new_full((2, 1), 0.07, requires_grad=True)
    with pytest.raises(ValueError, match="requires noise"):
        model.condition_on_observations(new_X, new_Y)
    before = likelihood.noise.clone()
    conditioned = model.condition_on_observations(new_X, new_Y, noise=new_noise)
    torch.testing.assert_close(likelihood.noise, before, atol=0, rtol=0)
    torch.testing.assert_close(
        conditioned.likelihood.noise_covar.noise,
        torch.cat((noise.squeeze(-1), new_noise.squeeze(-1))),
        atol=3e-12,
        rtol=3e-11,
    )
    if learned_noise:
        assert (
            conditioned.likelihood.second_noise_covar.raw_noise
            is not likelihood.second_noise_covar.raw_noise
        )
    train, targets, _ = conditioned._training_values()
    transformed = conditioned.transform_inputs(query)
    mapped_train, mapped_query = (
        conditioned._mapped_inputs(train),
        conditioned._mapped_inputs(transformed),
    )
    observed = conditioned._covariance(mapped_train, mapped_train) + torch.diag_embed(
        conditioned.likelihood.noise
    )
    cross = conditioned._covariance(mapped_train, mapped_query)
    mean_train, mean_query = conditioned._mean(mapped_train), conditioned._mean(mapped_query)
    expected_mean = mean_query + (
        cross.mT @ torch.linalg.solve(observed, (targets - mean_train).unsqueeze(-1))
    ).squeeze(-1)
    expected_covariance = conditioned._covariance(
        mapped_query, mapped_query
    ) - cross.mT @ torch.linalg.solve(observed, cross)
    actual = conditioned(transformed)
    leaves = (X, Y, noise, new_X, new_Y, new_noise, query, *conditioned.parameters())
    torch.testing.assert_close(actual.mean, expected_mean, atol=3e-12, rtol=3e-11)
    torch.testing.assert_close(
        actual.covariance_matrix, expected_covariance, atol=3e-12, rtol=3e-11
    )
    for a, e in zip(
        _derivatives(actual.mean.sin().sum() + actual.covariance_matrix.square().sum(), leaves),
        _derivatives(expected_mean.sin().sum() + expected_covariance.square().sum(), leaves),
        strict=True,
    ):
        torch.testing.assert_close(a, e, atol=3e-9, rtol=3e-8)


@pytest.mark.parametrize("device", DEVICES)
def test_custom_gaussian_conditioning_noise_override_defines_actual_mll(device):
    pytest.importorskip("gpytorch")
    from test_torch_exact_gaussian_likelihood import Gaussian, InputNoise

    X, Y, _, query = _data(device, gradients=True)
    likelihood = Gaussian(InputNoise(X, ())).to(X)
    model = MultiTaskGP(
        X,
        Y,
        1,
        likelihood=likelihood,
        backend="torch",
        outcome_transform=None,
        task_covar_prior=None,
    ).eval()
    new_X = X.new_tensor([[0.3, 2, 0.4], [0.8, 5, 0.6]], requires_grad=True)
    new_Y = X.new_tensor([[0.7], [-0.2]], requires_grad=True)
    noise = X.new_full((2, 1), 0.07, requires_grad=True)
    conditioned = model.condition_on_observations(new_X, new_Y, noise=noise)
    train, targets, _ = conditioned._training_values()
    mapped = conditioned._mapped_inputs(train)
    mapped_query = conditioned._mapped_inputs(query)
    mean_train, mean_query = conditioned._mean(mapped), conditioned._mean(mapped_query)
    future_noise = conditioned.likelihood.noise_covar
    feature = (X[..., 0].sin() + 0.2) * future_noise.amplitude
    old_diagonal = future_noise.noise.expand_as(feature) + feature.square()
    # This supplied noise module explicitly returns a diagonal covariance for
    # a noise= override. Its actual protocol determines the resulting law.
    diagonal = torch.cat((old_diagonal, noise.squeeze(-1)))
    observed = conditioned._covariance(mapped, mapped) + torch.diag_embed(diagonal)
    residual = targets - mean_train
    solved = torch.linalg.solve(observed, residual.unsqueeze(-1))
    cross = conditioned._covariance(mapped, mapped_query)
    expected_mean = mean_query + (cross.mT @ solved).squeeze(-1)
    expected_covariance = conditioned._covariance(
        mapped_query, mapped_query
    ) - cross.mT @ torch.linalg.solve(observed, cross)
    actual = conditioned(query)
    torch.testing.assert_close(actual.mean, expected_mean, atol=3e-12, rtol=3e-11)
    torch.testing.assert_close(
        actual.covariance_matrix, expected_covariance, atol=3e-12, rtol=3e-11
    )
    expected_mll = (
        -0.5 * (residual.unsqueeze(-2) @ solved).squeeze()
        - 0.5 * torch.linalg.slogdet(observed)[1]
        - 0.5 * targets.shape[-1] * torch.log(X.new_tensor(2 * torch.pi))
        + conditioned._log_prior()
    ) / targets.shape[-1]
    actual_mll = ExactMarginalLogLikelihood(conditioned.likelihood, conditioned)()
    torch.testing.assert_close(actual_mll, expected_mll, atol=3e-12, rtol=3e-11)
    leaves = (X, Y, new_X, new_Y, noise, *conditioned.parameters())
    for a, e in zip(
        _derivatives(actual_mll, leaves), _derivatives(expected_mll, leaves), strict=True
    ):
        torch.testing.assert_close(a, e, atol=3e-9, rtol=3e-8)

"""Stratified outcome context survives scalar and correlated multitask inference."""

# ruff: noqa: E402
from __future__ import annotations

import copy
from unittest.mock import patch

import pytest

torch = pytest.importorskip("torch")
gpytorch = pytest.importorskip("gpytorch")
pytest.importorskip("botorch")

from botorch.models import SingleTaskGP as RefGP
from botorch.models.transforms.input import Normalize as RefNormalize
from botorch.models.transforms.outcome import StratifiedStandardize as RefStratified
from test_torch_direct_standardize import _compare_graphs

from botorch_rs.kernels import RBFKernel
from botorch_rs.means import ZeroMean
from botorch_rs.models import MultiTaskGP, SingleTaskGP
from botorch_rs.models._torch_backend_modules import build_outcome_transform
from botorch_rs.models.transforms.input import Normalize
from botorch_rs.models.transforms.outcome import (
    ChainedOutcomeTransform,
    Log,
    Standardize,
    StratifiedStandardize,
)
from botorch_rs.posteriors._torch_backend import TorchGPyTorchPosterior
from botorch_rs.sampling.normal import IIDNormalSampler

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


def _data(device):
    X = torch.tensor(
        [[0.1, 0], [0.15, 1], [0.4, 0], [0.5, 1], [0.8, 0], [0.9, 1]],
        device=device,
        dtype=torch.double,
        requires_grad=True,
    )
    Y = X.new_tensor([[0.2], [1.1], [0.8], [1.7], [1.4], [0.9]], requires_grad=True)
    noise = torch.full_like(Y, 0.04, requires_grad=True)
    return X, Y, noise


def _standardized(X, Y, V, logarithm):
    values = Y.log() if logarithm else Y
    noise = V / Y.square() if logarithm else V
    groups = [values[X[:, -1] == task] for task in range(2)]
    means = torch.stack([group.mean() for group in groups])
    stds = torch.stack([group.std() for group in groups])
    indices = X[:, -1].long()
    return (
        (values - means[indices, None]) / stds[indices, None],
        noise / stds[indices, None].square(),
        means,
        stds,
    )


def _moments(mean, covariance, offset, scale, logarithm):
    gaussian_mean = mean * scale + offset
    gaussian_covariance = covariance * scale.unsqueeze(-1) * scale.unsqueeze(-2)
    variance = gaussian_covariance.diagonal(dim1=-2, dim2=-1)
    if logarithm:
        return (gaussian_mean + variance / 2).exp(), variance.expm1() * (
            2 * gaussian_mean + variance
        ).exp()
    return gaussian_mean, variance


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("supplied", ["port", "source"])
@pytest.mark.parametrize("logarithm", [False, True])
def test_scalar_stratified_builder_full_data_and_parameter_graphs(device, supplied, logarithm):
    X, Y, V = _data(device)
    stratum = (
        StratifiedStandardize(-1, [0, 1])
        if supplied == "port"
        else RefStratified(-1, torch.tensor([0, 1], device=device))
    )
    outcome = stratum
    # Source transforms are supplied directly only outside a port chain.
    if logarithm:
        if supplied == "source":
            from botorch.models.transforms.outcome import ChainedOutcomeTransform as RefChain
            from botorch.models.transforms.outcome import Log as RefLog

            outcome = RefChain(log=RefLog(), strata=stratum)
        else:
            outcome = ChainedOutcomeTransform(log=Log(), strata=stratum)
    # Calibrated eval descriptors are deliberately refitted by GP constructors.
    outcome(Y + 2, X=X)
    outcome.eval()
    bounds = X.new_tensor([[-0.2], [1.2]])
    model = SingleTaskGP(
        X,
        Y,
        V,
        covar_module=RBFKernel(ard_num_dims=2),
        mean_module=ZeroMean(),
        input_transform=Normalize(2, indices=[0], bounds=bounds),
        outcome_transform=outcome,
        backend="torch",
    )
    query = X.new_tensor([[0.3, 1], [0.6, 0]], requires_grad=True)
    leaves = (X, Y, V, query, *model.parameters())
    for _ in range(2):
        actual = model.posterior(query)
        targets, noise, means, stds = _standardized(X, Y, V, logarithm)
        train = torch.stack(((X[:, 0] + 0.2) / 1.4, X[:, 1]), -1)
        test = torch.stack(((query[:, 0] + 0.2) / 1.4, query[:, 1]), -1)
        lengthscale = model.covar_module.lengthscale

        def kernel(a, b, scale=lengthscale):
            return (-0.5 * ((a.unsqueeze(-2) - b.unsqueeze(-3)) / scale).square().sum(-1)).exp()

        covariance = kernel(train, train) + noise.squeeze(-1).diag_embed()
        cross = kernel(test, train)
        mean = (cross @ torch.linalg.solve(covariance, targets)).squeeze(-1)
        cov = kernel(test, test) - cross @ torch.linalg.solve(covariance, cross.mT)
        task = query[:, -1].long()
        expected_mean, expected_var = _moments(mean, cov, means[task], stds[task], logarithm)
        _compare_graphs(
            actual.mean.sin().sum() + actual.variance.square().sum(),
            expected_mean.sin().sum() + expected_var.square().sum(),
            leaves,
        )
    if not logarithm:
        source = RefGP(
            X,
            Y,
            V,
            covar_module=gpytorch.kernels.RBFKernel(ard_num_dims=2),
            mean_module=gpytorch.means.ZeroMean(),
            input_transform=RefNormalize(2, indices=torch.tensor([0]), bounds=bounds),
            outcome_transform=RefStratified(-1, torch.tensor([0, 1], device=device)),
        )
        source.covar_module.lengthscale = model.covar_module.lengthscale.detach()
        reference = source.posterior(query)
        torch.testing.assert_close(actual.mean, reference.mean, rtol=1e-9, atol=1e-10)
        torch.testing.assert_close(actual.variance, reference.variance, rtol=1e-9, atol=1e-10)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("selected", [[1, 0], [1]])
@pytest.mark.parametrize("logarithm", [False, True])
def test_multitask_stratified_full_query_context_and_correlated_graphs(device, selected, logarithm):
    X, Y, V = _data(device)
    outcome = StratifiedStandardize(-1, [0, 1])
    if logarithm:
        outcome = ChainedOutcomeTransform(log=Log(), strata=outcome)
    torch.manual_seed(23)
    model = MultiTaskGP(
        X,
        Y,
        -1,
        V,
        covar_module=RBFKernel(ard_num_dims=1, active_dims=[0]),
        mean_module=ZeroMean(),
        outcome_transform=outcome,
        task_covar_prior=None,
        backend="torch",
    )
    query = X.new_tensor([[[0.3], [0.6]], [[0.4], [0.7]], [[0.2], [0.5]]], requires_grad=True)
    leaves = (X, Y, V, query, *model.parameters())
    for _ in range(2):
        actual = model.posterior(query, output_indices=selected)
        targets, noise, means, stds = _standardized(X, Y, V, logarithm)
        full, m, q = model._query_inputs(query, selected)
        mapped_train, mapped_query = model._mapped_inputs(X), model._mapped_inputs(full)
        # This independent dense conditional reuses the tested physical kernel
        # while independently constructing all transform and solve operations.
        covariance = model._covariance(mapped_train, mapped_train) + noise.squeeze(-1).diag_embed()
        cross = model._covariance(mapped_query, mapped_train)
        mean = (cross @ torch.linalg.solve(covariance, targets)).squeeze(-1)
        cov = model._covariance(mapped_query, mapped_query) - cross @ torch.linalg.solve(
            covariance, cross.mT
        )
        task = full[..., -1].long()
        expected_mean, expected_var = _moments(mean, cov, means[task], stds[task], logarithm)
        expected_mean = expected_mean.reshape(*mean.shape[:-1], m, q).transpose(-1, -2)
        expected_var = expected_var.reshape(*mean.shape[:-1], m, q).transpose(-1, -2)
        _compare_graphs(
            actual.mean.sin().sum() + actual.variance.square().sum(),
            expected_mean.sin().sum() + expected_var.square().sum(),
            leaves,
        )


@pytest.mark.parametrize("device", DEVICES)
def test_public_chained_state_migration_and_immutable_sample_graph(device):
    X, Y, _ = _data(device)

    def make():
        return ChainedOutcomeTransform(log=Log(), strata=StratifiedStandardize(-1, [0, 1]))

    transform = make().to(Y)
    transformed, _ = transform(Y, X=X)
    restored, _ = transform.untransform(transformed, X=X)
    torch.testing.assert_close(restored, Y)
    state = copy.deepcopy({key: value.detach() for key, value in transform.state_dict().items()})
    assert set(state) == {"strata." + key for key in ("means", "stdvs", "_stdvs_sq", "_is_trained")}
    loaded = make().to(Y)
    loaded.load_state_dict(state)
    loaded.eval()
    torch.testing.assert_close(loaded(Y, X=X)[0], transform.eval()(Y, X=X)[0])
    assert dict(loaded.named_buffers())["strata.means"] is loaded["strata"].means
    loaded.float().cpu().double()
    assert loaded["strata"].means.device.type == "cpu"
    mean = Y.new_tensor([[0.2], [0.4]], requires_grad=True)
    covariance = Y.new_tensor([[0.6, 0.1], [0.1, 0.7]])
    posterior = transform.untransform_posterior(TorchGPyTorchPosterior(mean, covariance), X=X[:2])
    old_mean = posterior.mean.clone()
    transform.train()(Y + 3, X=X)
    torch.testing.assert_close(posterior.mean, old_mean)
    first = torch.autograd.grad(posterior.mean.square().sum(), (Y, mean), create_graph=True)
    second = torch.autograd.grad(sum(g.square().sum() for g in first), (Y, mean))
    assert all(bool(g.isfinite().all()) for g in (*first, *second))


@pytest.mark.parametrize("device", DEVICES)
def test_native_standardize_handle_exports_only_at_explicit_native_boundary(device):
    _, Y, _ = _data(device)
    transform = Standardize(1)
    transform(Y)
    with patch(
        "botorch_rs.models.transforms.outcome.to_flat", side_effect=AssertionError("host copy")
    ):
        assert transform.min_stdv == 1e-8
        torch.testing.assert_close(transform.means, Y.mean(-2, keepdim=True))
        transform.eval()(Y)
    native = transform._rs
    torch.testing.assert_close(Y.new_tensor(native.means[0]).reshape(1, 1), transform.means)
    torch.testing.assert_close(Y.new_tensor(native.stdvs[0]).reshape(1, 1), transform.stdvs)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("chained", [False, True])
def test_targetless_outcome_builder_retains_fitted_graph_without_fabricating_targets(
    device, chained
):
    X, Y, _ = _data(device)
    source = (
        ChainedOutcomeTransform(log=Log(), standardize=Standardize(1))
        if chained
        else Standardize(1)
    )
    source(Y)
    source.eval()
    transform = build_outcome_transform(source, 1, None, train_X=X)
    mean = Y.new_tensor([[0.2], [0.4]], requires_grad=True)
    covariance = Y.new_tensor([[0.6, 0.1], [0.1, 0.7]])
    posterior = transform.untransform_posterior(TorchGPyTorchPosterior(mean, covariance))
    values = Y.log() if chained else Y
    expected_mean, expected_var = _moments(
        mean.squeeze(-1), covariance, values.mean(), values.std().expand(2), chained
    )
    _compare_graphs(
        posterior.mean.square().sum() + posterior.variance.sin().sum(),
        expected_mean.square().sum() + expected_var.sin().sum(),
        (Y, mean),
    )
    untrained = build_outcome_transform(Standardize(1), 1, None, train_X=X)
    assert not untrained._is_trained
    with pytest.raises(RuntimeError, match="fitted"):
        untrained.untransform(mean)
    stateless = build_outcome_transform(Log(), 1, None, train_X=X)
    torch.testing.assert_close(stateless.untransform(mean)[0], mean.exp())


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("multitask", [False, True])
def test_stratified_conditioning_and_fantasies_use_task_specific_noise_context(device, multitask):
    X, Y, V = _data(device)
    options = dict(outcome_transform=StratifiedStandardize(-1, [0, 1]), backend="torch")
    model = MultiTaskGP(X, Y, -1, V, **options) if multitask else SingleTaskGP(X, Y, V, **options)
    query = X.new_tensor([[0.3, 1], [0.6, 0]], requires_grad=True)
    observed = X.new_tensor([[0.6], [0.4]], requires_grad=True)
    noise = X.new_tensor([[0.02], [0.03]], requires_grad=True)
    conditioned = model.condition_on_observations(query, observed, noise=noise)
    task = query[:, -1].long()
    _, _, _, stds = _standardized(X, Y, V, False)
    expected_raw = noise * stds[task, None].square()
    torch.testing.assert_close(conditioned._train_Yvar[-2:], expected_raw)
    _compare_graphs(
        conditioned._train_Yvar[-2:].square().sum(), expected_raw.square().sum(), (Y, noise)
    )
    fantasy = model.fantasize(
        query, IIDNormalSampler(torch.Size([2]), seed=11), propagate_grads=True
    )
    posterior = fantasy.posterior(query)
    first = torch.autograd.grad(
        posterior.mean.square().sum() + posterior.variance.sum(), (Y, query), create_graph=True
    )
    second = torch.autograd.grad(sum(g.square().sum() for g in first), (Y, query))
    assert all(bool(value.isfinite().all()) for value in (*first, *second))


@pytest.mark.parametrize("device", DEVICES)
def test_stratified_statistics_refresh_when_task_context_changes_without_target_edits(device):
    X, Y, V = (value.detach() for value in _data(device))
    model = SingleTaskGP(
        X, Y, V, outcome_transform=StratifiedStandardize(-1, [0, 1]), backend="torch"
    )
    model.posterior(X[:2])
    with torch.no_grad():
        X[:2, -1] = 1 - X[:2, -1]
    model.posterior(X[:2])
    expected = torch.stack([Y[X[:, -1] == task].mean() for task in range(2)]).unsqueeze(-1)
    torch.testing.assert_close(model.outcome_transform.means, expected)

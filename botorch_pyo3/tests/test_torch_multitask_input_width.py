"""Source-supported multitask training feature changes and native fallback."""

# ruff: noqa: E402
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from test_torch_backend_multitask import DEVICES, data
from test_torch_input_models import _derivatives

from botorch_rs.models import MultiTaskGP
from botorch_rs.models.transforms.input import AppendFeatures, FilterFeatures


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("family", ["filter", "append"])
@pytest.mark.parametrize("backend", ["auto", "torch"])
def test_multitask_transformed_training_width_matches_source(device, family, backend):
    pytest.importorskip("botorch")
    from botorch.models import MultiTaskGP as Source
    from botorch.models.transforms.input import AppendFeatures as SourceAppend
    from botorch.models.transforms.input import FilterFeatures as SourceFilter

    X, Y, Q = data(device)
    Q = Q[0]
    if family == "filter":
        X = torch.cat((X[:, :1], X.new_full((X.shape[0], 1), 0.43), X[:, 1:]), -1)
        actual_tf = FilterFeatures([0, 2], transform_on_eval=False)
        source_tf = SourceFilter(torch.tensor([0, 2], device=device), transform_on_eval=False)
        task_feature = -1
    else:
        features = X.new_tensor([[0.43]])
        actual_tf = AppendFeatures(features, transform_on_train=True, transform_on_eval=False)
        source_tf = SourceAppend(features, transform_on_train=True, transform_on_eval=False)
        Q = torch.cat((Q, Q.new_full((Q.shape[0], 1), 0.43)), -1)
        task_feature = 1
    noise = torch.full_like(Y, 0.03)
    torch.manual_seed(23)
    # Validate the unmodified source case before interpreting the port result.
    source = Source(
        X,
        Y,
        task_feature,
        noise,
        input_transform=source_tf,
        outcome_transform=None,
        task_covar_prior=None,
    ).to(X)
    expected = source.posterior(Q, observation_noise=True)
    torch.manual_seed(23)
    actual = MultiTaskGP(
        X,
        Y,
        task_feature,
        noise,
        input_transform=actual_tf,
        outcome_transform=None,
        task_covar_prior=None,
        backend=backend,
    )
    assert isinstance(actual, torch.nn.Module)
    assert actual.num_non_task_features == source.num_non_task_features
    with torch.no_grad():
        parameters = dict(source.named_parameters())
        for name, parameter in actual.named_parameters():
            parameters[name].copy_(parameter)
    source.prediction_strategy = None
    expected = source.posterior(Q, observation_noise=True)
    result = actual.posterior(Q, observation_noise=True)
    torch.testing.assert_close(result.mean, expected.mean, atol=3e-9, rtol=3e-8)
    torch.testing.assert_close(
        result.distribution.covariance_matrix,
        expected.distribution.covariance_matrix,
        atol=3e-9,
        rtol=3e-8,
    )


def test_multitask_native_only_input_limit_remains_explicit():
    from botorch_rs.exceptions.errors import UnsupportedError

    X, Y, _ = data("cpu")
    with pytest.raises(UnsupportedError, match="must be a botorch_rs Normalize"):
        MultiTaskGP(X, Y, -1, input_transform=FilterFeatures([0, 1]), backend="native")


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("family", ["filter", "append"])
def test_multitask_changed_training_width_full_dense_hessians(device, family):
    X, Y, Q = data(device)
    Q = Q[0]
    if family == "filter":
        X = torch.cat((X[:, :1], X.new_full((X.shape[0], 1), 0.43), X[:, 1:]), -1)
        transform = FilterFeatures([0, 2], transform_on_eval=False)
        task_feature = -1
    else:
        transform = AppendFeatures(
            X.new_tensor([[0.43]]), transform_on_train=True, transform_on_eval=False
        )
        Q = torch.cat((Q, Q.new_full((Q.shape[0], 1), 0.43)), -1)
        task_feature = 1
    X, Y, Q = (value.detach().requires_grad_() for value in (X, Y, Q))
    noise = torch.full_like(Y, 0.03)
    model = MultiTaskGP(
        X,
        Y,
        task_feature,
        noise,
        input_transform=transform,
        outcome_transform=None,
        task_covar_prior=None,
        backend="torch",
    )
    result = model.posterior(Q)
    train = (
        X[..., [0, 2]]
        if family == "filter"
        else torch.cat((X, X.new_full((X.shape[0], 1), 0.43)), -1)
    )
    task = model._task_feature
    query = torch.cat(
        tuple(
            torch.cat((Q[..., :task], Q.new_full((*Q.shape[:-1], 1), label), Q[..., task:]), -1)
            for label in (0, 1)
        ),
        -2,
    )
    train_mean, query_mean = model._mean(train), model._mean(query)
    matrix = model._covariance(train, train) + torch.diag_embed(noise.squeeze(-1))
    cross = model._covariance(train, query)
    mean = query_mean + (
        cross.transpose(-1, -2) @ torch.linalg.solve(matrix, Y - train_mean.unsqueeze(-1))
    ).squeeze(-1)
    covariance = model._covariance(query, query) - cross.transpose(-1, -2) @ torch.linalg.solve(
        matrix, cross
    )
    mean = mean.reshape(2, Q.shape[-2]).transpose(-1, -2)
    leaves = (X, Y, Q, *model.parameters())
    for values, controls in zip(
        _derivatives(result.mean, result.distribution.covariance_matrix, leaves),
        _derivatives(mean, covariance, leaves),
        strict=True,
    ):
        for value, expected in zip(values, controls, strict=True):
            torch.testing.assert_close(value, expected, atol=3e-7, rtol=3e-6)


def test_multitask_auto_uses_tensor_for_source_supported_nonlinear_outcome():
    from botorch_rs.models.transforms.outcome import Log

    X, Y, Q = data("cpu")
    actual = MultiTaskGP(X, Y.abs() + 1, -1, outcome_transform=Log())
    assert isinstance(actual, torch.nn.Module)
    assert torch.isfinite(actual.posterior(Q[0]).mean).all()

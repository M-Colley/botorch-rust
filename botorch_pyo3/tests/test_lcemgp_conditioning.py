"""Native LCEM updates, dense Gaussian identities and fantasy batch state."""

from __future__ import annotations

import copy
import math

import pytest

from botorch_rs import _ops
from botorch_rs._arrays import from_flat, kind_of, to_flat
from botorch_rs.acquisition import ExpectedImprovement
from botorch_rs.acquisition.objective import ScalarizedPosteriorTransform
from botorch_rs.exceptions.errors import (
    BotorchTensorDimensionError,
    DeprecationError,
    UnsupportedError,
)
from botorch_rs.mlls import ExactMarginalLogLikelihood
from botorch_rs.models.contextual_multioutput import LCEMGP, ProvidedEmbeddingLCEMGP
from botorch_rs.sampling import IIDNormalSampler
from helpers import as_kind, flatten, shape


def _model(kind, provided=False, standardized=True):
    x = [[0.1, 0], [0.23, 1], [0.48, 0], [0.66, 1], [0.82, 0], [0.92, 1]]
    y = [[1.1], [-0.1], [0.7], [0.3], [0.9], [-0.4]]
    options = dict(task_feature=1, seed=73)
    if not standardized:
        options["outcome_transform"] = None
    if provided:
        model = ProvidedEmbeddingLCEMGP(
            as_kind(x, kind),
            as_kind(y, kind),
            context_emb_feature=as_kind([[0.2], [0.7]], kind),
            **options,
        )
    else:
        model = LCEMGP(as_kind(x, kind), as_kind(y, kind), **options, backend="native")
        model.emb_layers[0].weight = as_kind([[0.2], [-0.3]], kind)
    model.likelihood.noise = as_kind([0.04, 0.07], kind)
    model.covar_module.lengthscale = 0.4
    model.eval()
    return model


def _assert_close(left, right, tolerance=2e-10):
    assert shape(left) == shape(right)
    assert (
        max((abs(a - b) for a, b in zip(flatten(left), flatten(right), strict=True)), default=0)
        < tolerance
    )


@pytest.mark.parametrize("provided", [False, True])
@pytest.mark.parametrize("standardized", [False, True])
@pytest.mark.parametrize("explicit", [False, True])
def test_native_conditioned_law_matches_independent_dense_gaussian(
    kind, provided, standardized, explicit
):
    model = _model(kind, provided, standardized)
    x = as_kind([[0.18, 0], [0.71, 1]], kind)
    y = as_kind([[0.5], [-0.2]], kind)
    test = as_kind([[0.3, 1], [0.61, 0]], kind)
    joint = model.posterior(_ops.concat([x, test], -2))
    mean = flatten(joint.mean)
    covariance = flatten(joint.distribution.covariance_matrix)
    standard = 1 if not standardized else flatten(model.outcome_transform.stdvs)[0]
    noise = [0.015, 0.08] if explicit else flatten(model.likelihood.noise)
    a, b, c = (
        covariance[0] + standard**2 * noise[0],
        covariance[1],
        covariance[5] + standard**2 * noise[1],
    )
    inverse = [
        [c / (a * c - b * b), -b / (a * c - b * b)],
        [-b / (a * c - b * b), a / (a * c - b * b)],
    ]
    residual = [flatten(y)[i] - mean[i] for i in range(2)]
    expected_mean, expected_covariance = [], []
    for i in range(2, 4):
        expected_mean.append(
            mean[i]
            + sum(
                covariance[i * 4 + j] * inverse[j][k] * residual[k]
                for j in range(2)
                for k in range(2)
            )
        )
        for column in range(2, 4):
            expected_covariance.append(
                covariance[i * 4 + column]
                - sum(
                    covariance[i * 4 + j] * inverse[j][k] * covariance[k * 4 + column]
                    for j in range(2)
                    for k in range(2)
                )
            )
    before = [(name, flatten(value)) for name, value in model.named_parameters()]
    conditioned = model.condition_on_observations(
        x, y, as_kind([[0.015], [0.08]], kind) if explicit else None
    )
    posterior = conditioned.posterior(test)
    assert flatten(posterior.mean) == pytest.approx(expected_mean, abs=3e-12, rel=3e-12)
    assert flatten(posterior.distribution.covariance_matrix) == pytest.approx(
        expected_covariance, abs=3e-12, rel=3e-12
    )
    assert [(name, flatten(value)) for name, value in conditioned.named_parameters()] == before
    assert shape(conditioned.train_inputs[0]) == (8, 2)
    assert shape(model.train_inputs[0]) == (6, 2)
    if standardized:
        _assert_close(conditioned.outcome_transform.means, model.outcome_transform.means)
        _assert_close(conditioned.outcome_transform.stdvs, model.outcome_transform.stdvs)
    assert isinstance(conditioned, type(model))


def test_native_broadcast_conditioning_nested_updates_shared_parameters_and_snapshots(kind):
    model = _model(kind)
    x = as_kind([[[0.15 + i * 0.04, 0], [0.7 - i * 0.03, 1]] for i in range(3)], kind)
    y = as_kind([[[[0.4]], [[-0.2]]], [[[0.8]], [[0.1]]]], kind)
    # Reshape the targets to two fantasies, singleton query batch and two observations.
    y = from_flat(to_flat(y)[0], (2, 1, 2, 1), kind_of(y))
    conditioned = model.condition_on_observations(x, y)
    assert conditioned.batch_shape == (2, 3)
    assert shape(conditioned.train_inputs[0]) == (2, 3, 8, 2)
    assert shape(conditioned.train_targets) == (2, 3, 8)
    test = as_kind([[0.32], [0.57]], kind)
    posterior = conditioned.posterior(test)
    assert shape(posterior.mean) == (2, 3, 2, 2)
    result = flatten(posterior.mean)
    covariance = flatten(posterior.distribution.covariance_matrix)
    for f in range(2):
        for b in range(3):
            one = model.condition_on_observations(_ops.getitem(x, b), _ops.getitem(y, (f, 0)))
            expected = one.posterior(test)
            index = f * 3 + b
            assert result[index * 4 : (index + 1) * 4] == pytest.approx(
                flatten(expected.mean), abs=3e-12
            )
            assert covariance[index * 16 : (index + 1) * 16] == pytest.approx(
                flatten(expected.distribution.covariance_matrix), abs=3e-12
            )
    saved = flatten(posterior.mean)
    cloned = copy.deepcopy(conditioned)
    conditioned.likelihood.noise = as_kind([0.08, 0.12], kind)
    assert flatten(posterior.mean) == saved
    assert flatten(cloned.posterior(test).mean) == saved
    assert (
        max(
            abs(a - b)
            for a, b in zip(flatten(conditioned.posterior(test).mean), saved, strict=True)
        )
        > 1e-4
    )
    nested = conditioned.condition_on_observations(
        as_kind([[0.44, 1]], kind), as_kind([[0.2]], kind)
    )
    assert nested.batch_shape == (2, 3)
    assert shape(nested.train_inputs[0]) == (2, 3, 9, 2)
    mll = ExactMarginalLogLikelihood(nested.likelihood, nested)
    values, gradient = mll.value_and_grad()
    assert len(values) == 6 and all(math.isfinite(value) for value in values)
    assert all(math.isfinite(value) for value in gradient)
    acquisition = ExpectedImprovement(
        model.condition_on_observations(as_kind([[0.44, 1]], kind), as_kind([[0.2]], kind)),
        best_f=0.0,
        posterior_transform=ScalarizedPosteriorTransform(as_kind([1, -0.2], kind)),
    )
    assert shape(acquisition(as_kind([[0.33]], kind))) == (1,)


def test_native_fantasies_replay_observed_samples_with_shared_batches(kind):
    model = _model(kind)
    x = as_kind([[[0.2, 0], [0.6, 1]], [[0.25, 0], [0.7, 1]]], kind)
    draws = IIDNormalSampler((3,), seed=19)(model.posterior(x, observation_noise=True))
    manual = model.condition_on_observations(x, draws)
    fantasy = model.fantasize(x, IIDNormalSampler((3,), seed=19))
    assert fantasy.batch_shape == (3, 2)
    _assert_close(fantasy.train_targets, manual.train_targets)
    test = as_kind([[0.3], [0.6]], kind)
    _assert_close(fantasy.posterior(test).mean, manual.posterior(test).mean)
    _assert_close(
        fantasy.posterior(test).distribution.covariance_matrix,
        manual.posterior(test).distribution.covariance_matrix,
    )


def test_native_lcem_update_validation_and_empty_fantasies(kind):
    model = _model(kind)
    x, y = as_kind([[0.3, 0]], kind), as_kind([[0.4]], kind)
    model.train()
    with pytest.raises(RuntimeError, match="after making predictions"):
        model.condition_on_observations(x, y)
    model.eval()
    with pytest.raises(BotorchTensorDimensionError, match="task feature"):
        model.condition_on_observations(as_kind([[0.3]], kind), y)
    with pytest.raises(ValueError, match="nonnegative"):
        model.condition_on_observations(x, y, as_kind([[-0.1]], kind))
    with pytest.raises(ValueError, match="finite"):
        model.condition_on_observations(x, as_kind([[float("nan")]], kind))
    with pytest.raises(ValueError, match="task"):
        model.condition_on_observations(as_kind([[0.3, 5]], kind), y)
    with pytest.raises(DeprecationError, match="boolean"):
        model.fantasize(x, IIDNormalSampler((2,)), observation_noise=False)
    with pytest.raises(NotImplementedError, match="tensor"):
        model.fantasize(x, IIDNormalSampler((2,)), observation_noise=as_kind([[0.1]], kind))
    if kind != "list":
        empty = from_flat([], (0, 2), kind_of(x))
        fantasy = model.fantasize(empty, IIDNormalSampler((2,)))
        assert fantasy.batch_shape == (2,)
        assert shape(fantasy.train_inputs[0]) == (2, 6, 2)
        _assert_close(
            fantasy.posterior(as_kind([[0.4]], kind)).mean,
            _ops.expand(model.posterior(as_kind([[0.4]], kind)).mean, (2, 1, 2)),
        )


def test_native_unobserved_context_metadata_and_shared_mean_updates_stay_consistent(kind):
    model = LCEMGP(
        as_kind([[0.1, 0], [0.4, 1], [0.6, 0], [0.9, 1]], kind),
        as_kind([[0.2], [-0.3], [0.6], [-0.1]], kind),
        task_feature=1,
        all_tasks=[0, 1, 2],
        backend="native",
        seed=4,
    )
    model.emb_layers[0].weight = as_kind([[0.1], [0.3], [0.5]], kind)
    model.eval()
    conditioned = model.condition_on_observations(
        as_kind([[[0.3, 0]], [[0.7, 2]]], kind), as_kind([[[0.4]], [[-0.2]]], kind)
    )
    conditioned.mean_module.base_means[0].constant = 0.4
    conditioned.mean_module.base_means[1].constant = -0.2
    conditioned.eval()
    # The source fantasy protocol retains the original task metadata, rather
    # than declaring different observed sets for each fantasy member.
    for member in conditioned._rs.models:
        assert member.observed_task_indices == [0, 1]
        assert member.constant == pytest.approx([0.4, -0.2, 0.1])
    if kind == "torch":
        torch = pytest.importorskip("torch")
        with pytest.raises(UnsupportedError, match="training-data gradients"):
            conditioned.condition_on_observations(
                torch.tensor([[0.3, 0]], dtype=torch.double, requires_grad=True),
                torch.tensor([[0.4]], dtype=torch.double),
            )


@pytest.mark.parametrize("standardized", [False, True])
def test_native_lcem_condition_and_fantasy_batches_match_official_source(kind, standardized):
    torch = pytest.importorskip("torch")
    pytest.importorskip("gpytorch")
    reference_module = pytest.importorskip("botorch.models.contextual_multioutput")
    model = _model(kind, standardized=standardized)
    xdata, xshape = to_flat(model.train_inputs[0])
    ydata, yshape = model._rs.train_targets()
    reference = reference_module.LCEMGP(
        torch.tensor(xdata, dtype=torch.double).reshape(xshape),
        torch.tensor(ydata, dtype=torch.double).reshape(yshape),
        task_feature=1,
        **({} if standardized else {"outcome_transform": None}),
    ).double()
    actual_parameters = dict(model.named_parameters())
    with torch.no_grad():
        for name, value in reference.named_parameters():
            value.copy_(
                torch.tensor(flatten(actual_parameters[name]), dtype=value.dtype).reshape_as(value)
            )
    x = torch.tensor([[0.18, 0.0], [0.71, 1.0]], dtype=torch.double)
    y = torch.tensor([[[0.4], [-0.2]], [[0.7], [0.1]]], dtype=torch.double)
    test = torch.tensor([[0.3], [0.61]], dtype=torch.double)
    reference.posterior(x)
    expected_model = reference.condition_on_observations(x, y)
    actual_model = model.condition_on_observations(
        as_kind(x.tolist(), kind), as_kind(y.tolist(), kind)
    )
    expected, actual = (
        expected_model.posterior(test),
        actual_model.posterior(as_kind(test.tolist(), kind)),
    )
    assert flatten(actual.mean) == pytest.approx(
        expected.mean.detach().flatten().tolist(), abs=2e-10
    )
    assert flatten(actual.variance) == pytest.approx(
        expected.variance.detach().flatten().tolist(), abs=2e-10
    )
    covariance = expected.distribution.covariance_matrix
    if expected.distribution._interleaved:
        order = torch.arange(4).reshape(2, 2).mT.flatten()
        covariance = covariance.index_select(-2, order).index_select(-1, order)
    assert flatten(actual.distribution.covariance_matrix) == pytest.approx(
        covariance.detach().flatten().tolist(), abs=3e-10
    )

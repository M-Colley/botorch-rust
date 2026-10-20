"""Historical-mean conditioning and actual native empirical fantasy batches."""

from __future__ import annotations

import math

import pytest
from test_empirical_gp_models import data

from botorch_rs import _ops
from botorch_rs.exceptions import UnsupportedError
from botorch_rs.models.empirical_gps.empirical_1d_gp import EmpiricalOneDimensionalGP
from botorch_rs.models.empirical_gps.multioutput_empirical_1d_gp import (
    MultiOutputEmpiricalOneDimensionalGP,
)
from botorch_rs.sampling.normal import IIDNormalSampler
from helpers import KINDS, as_kind, assert_close, to_list


@pytest.fixture(autouse=True)
def double_default():
    torch = pytest.importorskip("torch")
    before = torch.get_default_dtype()
    torch.set_default_dtype(torch.double)
    yield
    torch.set_default_dtype(before)


def pair(outputs=2, correlated=False, fixed=False, kind="torch"):
    torch = pytest.importorskip("torch")
    module = pytest.importorskip(
        "botorch.models.empirical_gps."
        + ("multioutput_empirical_1d_gp" if correlated else "empirical_1d_gp")
    )
    h, history, x, y, query = data(outputs)
    name = ("MultiOutput" if correlated else "") + "EmpiricalOneDimensionalGP"
    cls = MultiOutputEmpiricalOneDimensionalGP if correlated else EmpiricalOneDimensionalGP
    variance = [[0.02 + 0.01 * g for g in range(outputs)]] * len(x) if fixed else None
    actual = cls(
        *[as_kind(v, kind) for v in (x, y, h, history)],
        train_Yvar=None if variance is None else as_kind(variance, kind),
        ard=True,
    )
    reference = getattr(module, name)(
        *[torch.tensor(v) for v in (x, y, h, history)],
        train_Yvar=None if variance is None else torch.tensor(variance),
        ard=True,
    )
    reference.posterior(torch.tensor(query))
    return actual, reference, query


@pytest.mark.parametrize("correlated", [False, True])
@pytest.mark.parametrize("fixed", [False, True])
def test_empirical_conditioning_matches_official(correlated, fixed):
    torch = pytest.importorskip("torch")
    actual, reference, query = pair(correlated=correlated, fixed=fixed)
    x = torch.tensor([[0.25], [0.7]])
    y = torch.tensor([[2.4, 2.7], [2.0, 1.8]])
    noise = torch.tensor([[0.04, 0.02], [0.03, 0.05]]) if fixed else None
    native = actual.condition_on_observations(x, y, noise=noise)
    official = reference.condition_on_observations(x, y, noise=noise)
    q = torch.tensor(query)
    pa, pb = native.posterior(q), official.posterior(q)
    torch.testing.assert_close(pa.mean, pb.mean, rtol=4e-8, atol=4e-9)
    torch.testing.assert_close(
        pa.distribution.covariance_matrix, pb.distribution.covariance_matrix, rtol=4e-8, atol=4e-9
    )
    assert _ops.shape_of(actual.train_targets)[-1] == (8 if correlated else 4)
    assert _ops.shape_of(native.train_targets)[-1] == (12 if correlated else 6)


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("fixed", [False, True])
@pytest.mark.parametrize("sample_shape", [(3,), (2, 3)])
def test_empirical_fantasies_form_native_model_batches(kind, fixed, sample_shape):
    torch = pytest.importorskip("torch")
    actual, reference, query = pair(fixed=fixed, kind=kind)
    x = as_kind([[0.25], [0.7]], kind)
    fantasies = actual.fantasize(x, IIDNormalSampler(sample_shape, seed=61))
    assert fantasies.batch_shape == sample_shape
    assert _ops.shape_of(fantasies.train_targets) == (*sample_shape, 2, 6)
    y = torch.tensor(to_list(fantasies._train_Y))[..., -2:, :]
    noise = torch.tensor([[[0.02, 0.03]]]).expand(*sample_shape, 2, 2) if fixed else None
    # ExactGP accepts one new fantasy batch dimension; flatten multiple sample
    # dimensions for its reference and restore them after prediction.
    official = reference.condition_on_observations(
        torch.tensor([[0.25], [0.7]]),
        y.reshape(math.prod(sample_shape), 2, 2),
        noise=None if noise is None else noise.reshape(math.prod(sample_shape), 2, 2),
    )
    pa, pb = fantasies.posterior(as_kind(query, kind)), official.posterior(torch.tensor(query))
    assert_close(pa.mean, pb.mean.reshape(*sample_shape, 3, 2), rtol=4e-8, atol=4e-9)
    assert_close(
        pa.distribution.covariance_matrix,
        pb.distribution.covariance_matrix.reshape(*sample_shape, 6, 6),
        rtol=4e-8,
        atol=4e-9,
    )
    if fixed:
        assert _ops.shape_of(fantasies.likelihood.noise) == (*sample_shape, 2, 6)
    before = pa.mean
    fantasies.covar_module.curve_weights = as_kind([0.8] * 7, kind)
    assert_close(pa.mean, before, rtol=0, atol=0)
    assert _ops.shape_of(fantasies.posterior(as_kind(query, kind)).mean) == (*sample_shape, 3, 2)


def test_empirical_fantasy_query_gradients_and_empty_sampling():
    torch = pytest.importorskip("torch")
    model, reference, query = pair(outputs=1)
    fantasies = model.fantasize(torch.tensor([[0.4]]), IIDNormalSampler((3,), seed=8))
    y = fantasies._train_Y[..., -1:, :]
    expected_model = reference.condition_on_observations(torch.tensor([[0.4]]), y)
    x = torch.tensor(query, requires_grad=True)
    other = x.detach().clone().requires_grad_()
    actual = fantasies.posterior(x).mean
    expected = expected_model.posterior(other).mean
    torch.testing.assert_close(
        torch.autograd.grad(actual.sum(), x)[0],
        torch.autograd.grad(expected.sum(), other)[0],
        rtol=4e-8,
        atol=4e-9,
    )
    empty = model.fantasize(torch.empty(0, 1), IIDNormalSampler((2, 3), seed=8))
    assert empty.batch_shape == (2, 3)
    assert _ops.shape_of(empty.train_targets) == (2, 3, 4)
    with pytest.raises(UnsupportedError, match="gradients"):
        model.condition_on_observations(x, torch.ones(3, 1))

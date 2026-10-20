"""Constant mean initialization remains valid before and after native binding."""

from __future__ import annotations

import copy

import pytest

from botorch_rs.constraints import Interval
from botorch_rs.means import ConstantMean
from botorch_rs.models import SingleTaskGP
from helpers import as_kind, assert_close, check_kind, shape


def test_unbound_constant_mean(kind: str) -> None:
    mean = ConstantMean(batch_shape=(2, 3))
    assert_close(mean.constant, [[0.0] * 3] * 2)
    value = as_kind([[1.0, -2.0, 3.0]], kind)
    assert mean.initialize(raw_constant=value) is mean
    check_kind(mean.constant, kind)
    assert shape(mean.constant) == (2, 3)
    assert_close(mean.constant, [[1.0, -2.0, 3.0]] * 2)
    mean.constant = as_kind(1.75, kind)
    assert_close(mean.raw_constant, [[1.75] * 3] * 2)
    with pytest.raises(ValueError, match="broadcast"):
        mean.constant = [1.0, 2.0]
    with pytest.raises(ValueError, match="finite"):
        mean.constant = float("nan")
    with pytest.raises(AttributeError, match="no parameter"):
        mean.initialize(missing=1.0)


def test_initialized_constant_transfers_to_native_gp(kind: str) -> None:
    X = as_kind([[0.0], [0.4], [1.0]], kind)
    Y = as_kind([[0.1, 0.5], [0.4, 0.2], [-0.2, 0.7]], kind)
    mean = ConstantMean(batch_shape=(2,)).initialize(constant=as_kind([1.2, -0.8], kind))
    model = SingleTaskGP(X, Y, mean_module=mean, outcome_transform=None)
    check_kind(model.mean_module.constant, kind)
    assert_close(model.mean_module.constant, [1.2, -0.8])
    model.mean_module.constant = as_kind([-0.35, 0.55], kind)
    clone = copy.deepcopy(model)
    assert_close(clone.mean_module.constant, [-0.35, 0.55])
    subset = model.subset_output([1])
    assert_close(subset.mean_module.constant, 0.55)
    assert_close(model.mean_module.constant, [-0.35, 0.55])


def test_constant_constraint_initial_value() -> None:
    mean = ConstantMean(constant_constraint=Interval(-5.0, 5.0, transform=None, initial_value=2.25))
    assert_close(mean.constant, 2.25)
    model = SingleTaskGP([[0.0], [1.0]], [[0.1], [0.2]], mean_module=mean, outcome_transform=None)
    assert_close(model.mean_module.constant, 2.25)


def test_unbound_constant_matches_gpytorch() -> None:
    torch = pytest.importorskip("torch")
    gpytorch = pytest.importorskip("gpytorch")
    actual = ConstantMean(batch_shape=(2, 3))
    expected = gpytorch.means.ConstantMean(batch_shape=torch.Size([2, 3])).double()
    for value in (torch.tensor(-0.75, dtype=torch.double), torch.tensor([[1., 2., 3.]], dtype=torch.double)):
        actual.initialize(constant=value)
        expected.initialize(constant=value)
        assert_close(actual.constant, expected.constant)

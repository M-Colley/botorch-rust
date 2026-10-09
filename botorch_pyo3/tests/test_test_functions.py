"""Synthetic test functions."""

from __future__ import annotations

import pytest

from botorch_rs import test_functions as tf
from botorch_rs.exceptions import BotorchRsError
from helpers import HAS_NUMPY, HAS_TORCH, as_kind, check_kind, flatten, shape

ALL = [
    tf.Ackley,
    tf.Beale,
    tf.Branin,
    tf.Bukin,
    tf.Cosine8,
    tf.DixonPrice,
    tf.DropWave,
    tf.EggHolder,
    tf.Griewank,
    tf.Hartmann,
    tf.HolderTable,
    tf.Levy,
    tf.Michalewicz,
    tf.Powell,
    tf.Rastrigin,
    tf.Rosenbrock,
    tf.Shekel,
    tf.SixHumpCamel,
    tf.StyblinskiTang,
    tf.ThreeHumpCamel,
]


@pytest.mark.parametrize("cls", ALL, ids=[c.__name__ for c in ALL])
def test_optimizers_attain_the_optimal_value(cls: type) -> None:
    f = cls(dtype=None) if cls is not tf.Shekel else cls(m=10)
    bounds = f.bounds
    assert shape(bounds) == (2, f.dim)
    try:
        opt = f.optimizers
    except NotImplementedError:
        pytest.skip("no known optimizer")
    values = flatten(f.evaluate_true(opt))
    for v in values:
        assert v == pytest.approx(f.optimal_value, abs=1e-3 * (1 + abs(f.optimal_value)))
    negated = cls(negate=True) if cls is not tf.Shekel else cls(negate=True)
    assert negated.optimal_value == -f.optimal_value
    assert negated.is_minimization_problem != f.is_minimization_problem
    assert flatten(negated(opt, noise=False)) == [-v for v in values]


def test_hartmann(kind: str) -> None:
    h = tf.Hartmann(dim=6, negate=True)
    X = as_kind([[0.20169, 0.150011, 0.476874, 0.275332, 0.311652, 0.6573], [0.5] * 6], kind)
    y = h(X)
    check_kind(y, kind)
    assert shape(y) == (2,)
    assert flatten(y)[0] == pytest.approx(3.32237, abs=1e-5)
    assert shape(h(as_kind([[[0.5] * 6] * 3] * 2, kind))) == (2, 3)
    assert tf.Hartmann(dim=3).dim == 3
    with pytest.raises(NotImplementedError):
        _ = tf.Hartmann(dim=4).optimal_value
    with pytest.raises(NotImplementedError):
        _ = tf.Hartmann(dim=4).optimizers
    with pytest.raises(NotImplementedError):
        _ = tf.Michalewicz(dim=5).optimizers
    with pytest.raises(BotorchRsError):
        tf.Hartmann(dim=5)
    with pytest.raises(BotorchRsError):
        h.evaluate_true([[2.0] * 6])  # outside the bounds


def test_dimensions_and_bounds() -> None:
    assert tf.Ackley(dim=5).dim == 5
    assert tf.Cosine8().dim == 8 and tf.Branin().dim == 2  # fixed dimensions
    with pytest.raises(TypeError):  # like BoTorch, no `dim` argument
        tf.Branin(dim=2)
    assert tf.Branin(0.5).noise_std == 0.5  # BoTorch's positional order
    assert tf.Shekel(m=5).optimal_value == pytest.approx(-10.1532)
    custom = tf.Branin(bounds=[(-4.0, 4.0), (0.0, 15.0)], dtype=None)
    assert flatten(custom.bounds) == [-4.0, 0.0, 4.0, 15.0]
    with pytest.raises(BotorchRsError):
        tf.Branin(bounds=[(5.0, 10.0), (5.0, 10.0)])  # excludes every optimizer


def test_observation_noise(kind: str) -> None:
    f = tf.Rosenbrock(dim=2, noise_std=0.5, seed=1)
    X = as_kind([[1.0, 1.0]] * 200, kind)
    noisy = flatten(f(X))
    assert max(noisy) != min(noisy)
    assert sum(noisy) / len(noisy) == pytest.approx(0.0, abs=0.15)
    assert flatten(f(X, noise=False)) == [0.0] * 200
    assert flatten(f.evaluate_true(X)) == [0.0] * 200


@pytest.mark.skipif(not HAS_TORCH, reason="torch is not installed")
def test_torch_noise_follows_the_torch_seed() -> None:
    import torch

    f = tf.Branin(noise_std=1.0, negate=True)
    X = torch.rand(5, 2, dtype=torch.float64) * 10
    torch.manual_seed(3)
    a = f(X)
    torch.manual_seed(3)
    b = f(X)
    assert torch.equal(a, b)
    assert isinstance(f.bounds, torch.Tensor) and f.bounds.dtype == torch.float64
    assert tf.Branin(dtype=torch.float32).bounds.dtype == torch.float32


@pytest.mark.skipif(not HAS_NUMPY, reason="NumPy is not installed")
def test_numpy_dtype() -> None:
    import numpy as np

    f = tf.Hartmann(dtype=np.float64)
    assert isinstance(f.bounds, np.ndarray) and f.bounds.shape == (2, 6)
    assert isinstance(f.optimizers, np.ndarray)

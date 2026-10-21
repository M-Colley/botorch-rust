"""Model utility parity on lists, NumPy arrays and torch tensors."""

from __future__ import annotations

import math
import warnings
from types import SimpleNamespace

import pytest

from botorch_rs import settings
from botorch_rs.exceptions.errors import InputDataError
from botorch_rs.exceptions.warnings import InputDataWarning
from botorch_rs.models.utils import assorted as rs
from helpers import KINDS, as_kind, check_kind, to_list


@pytest.mark.parametrize("kind", KINDS)
def test_output_batch_shapes(kind: str) -> None:
    x = as_kind([[[0.0, 0.1], [0.2, 0.3], [0.4, 0.5]]] * 2, kind)
    y = as_kind([[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]] * 2, kind)
    X, Y, noise = rs.multioutput_to_batch_mode_transform(x, y, 2, y)
    check_kind(X, kind)
    assert to_list(X) == [[to_list(x)[b], to_list(x)[b]] for b in range(2)]
    assert to_list(Y) == [[[1.0, 3.0, 5.0], [2.0, 4.0, 6.0]]] * 2
    assert to_list(noise) == to_list(Y)
    expanded, index = rs.add_output_dim(x, (2,))
    assert index == 1 and to_list(expanded) == [[batch] for batch in to_list(x)]
    with pytest.raises(RuntimeError, match="trailing batch dimensions"):
        rs.add_output_dim(x, (3,))
    full = rs._make_X_full(x, [1, 3], tf=1)
    expected = [
        [[row[0], float(task), row[1]] for task in [1, 3] for row in batch] for batch in to_list(x)
    ]
    assert to_list(full) == expected


@pytest.mark.parametrize("kind", KINDS)
def test_data_validation(kind: str) -> None:
    X = as_kind([[0.0, 1.0], [1.0, 0.0]], kind)
    Y = as_kind([[-math.sqrt(0.5)], [math.sqrt(0.5)]], kind)
    with warnings.catch_warnings(record=True) as emitted:
        rs.check_min_max_scaling(X, strict=True)
        rs.check_standardization(Y)
        rs.validate_input_scaling(X, Y)
        rs.check_standardization(as_kind([[0.0]], kind))
    assert not emitted
    with pytest.warns(InputDataWarning, match="not scaled"):
        rs.check_min_max_scaling(as_kind([[0.2, 0.8]], kind), strict=True)
    with pytest.raises(InputDataError, match="not contained"):
        rs.check_min_max_scaling(as_kind([[0.0, 2.0]], kind), raise_on_fail=True)
    rs.check_min_max_scaling(as_kind([[0.0, 2.0]], kind), ignore_dims=[1])
    rs.check_min_max_scaling(as_kind([[2.0, 3.0]], kind), ignore_dims=[0, 1])
    with pytest.raises(InputDataError, match="not standardized"):
        rs.check_standardization(as_kind([[1.0], [1.0]], kind), raise_on_fail=True)
    with pytest.raises(InputDataError, match="NaN"):
        rs.check_no_nans(as_kind([[math.nan]], kind))
    variance = as_kind([[-0.1], [0.1]], kind)
    with pytest.raises(InputDataError, match="negative variances"):
        rs.validate_input_scaling(X, Y, variance, check_nans_only=True)
    with settings.validate_input_scaling(False):
        rs.validate_input_scaling(X, Y, variance)
    with pytest.raises(InputDataError, match="NaN"):
        rs.validate_input_scaling(X, as_kind([[math.nan]], kind), check_nans_only=True)


@pytest.mark.parametrize("kind", KINDS)
def test_duplicates_and_conservative_chains(kind: str) -> None:
    X = as_kind([[1.0, 2.0], [3.0, 4.0], [1.0, 2.0], [7.0, 8.0]], kind)
    Y = as_kind([[0, 1], [2, 3]], kind)
    assert list(rs.detect_duplicates(X)) == [(2, 0)]
    x, y, indices = rs.consolidate_duplicates(X, Y)
    assert to_list(x) == [[1.0, 2.0], [3.0, 4.0], [7.0, 8.0]]
    assert to_list(y) == [[0, 1], [0, 2]]
    assert to_list(indices) == [0, 1, 0, 2]
    # Every point in a chain is retained if removing it could change linkage.
    chain = as_kind([[0.0], [0.4], [0.8], [1.2]], kind)
    x, y, indices = rs.consolidate_duplicates(chain, Y, atol=0.5)
    assert x is chain and y is Y
    assert to_list(indices) == list(range(4))
    assert list(rs.detect_duplicates(as_kind([[0.0], [0.5]], kind), atol=0.5)) == []
    with pytest.raises(ValueError, match="2 dimensions"):
        rs.detect_duplicates(as_kind([[[0.0]]], kind))


@pytest.mark.parametrize("kind", KINDS)
def test_task_remapping(kind: str) -> None:
    values = as_kind([1, 3, 5], kind)
    dtype = values.dtype if kind != "list" else float
    mapping = rs.get_task_value_remapping(values, dtype)
    result = to_list(mapping)
    assert math.isnan(result[0]) and math.isnan(result[2]) and math.isnan(result[4])
    assert [result[i] for i in [1, 3, 5]] == [0, 1, 2]
    assert rs.get_task_value_remapping(as_kind([0, 1], kind), dtype) is None
    with pytest.raises(ValueError, match="nonnegative integers"):
        rs.get_task_value_remapping(as_kind([-1, 2], kind), dtype)
    with pytest.raises(ValueError, match="sorted and unique"):
        rs.get_task_value_remapping(as_kind([3, 1], kind), dtype)


def test_mod_batch_shape() -> None:
    leaf = SimpleNamespace(batch_shape=(2, 3))
    tree = SimpleNamespace(child=SimpleNamespace(leaf=leaf))
    rs.mod_batch_shape(tree, ["child", "leaf"], 4)
    assert leaf.batch_shape == (2, 4)
    rs.mod_batch_shape(tree, ["child", "leaf"], 0)
    assert leaf.batch_shape == (2,)


def test_live_botorch_assorted_parity() -> None:
    torch = pytest.importorskip("torch")
    bt = pytest.importorskip("botorch.models.utils.assorted")
    gen = torch.Generator().manual_seed(44)
    X = torch.rand(3, 4, 2, dtype=torch.double, generator=gen, requires_grad=True)
    Y = torch.randn(3, 4, 2, dtype=torch.double, generator=gen)
    for actual, expected in zip(
        rs.multioutput_to_batch_mode_transform(X, Y, 2, Y),
        bt.multioutput_to_batch_mode_transform(X, Y, 2, Y),
        strict=True,
    ):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    rs._make_X_full(X, [1, 3], 1).sum().backward()
    assert torch.equal(X.grad, torch.full_like(X, 2))
    for scale in (0.1, 1.0, 10.0):
        points = torch.randn(12, 3, dtype=torch.double, generator=gen) * scale
        points[4] = points[1]
        points[7] = points[6] + 1e-4
        comparisons = torch.randint(0, 12, (15, 3), generator=gen)
        assert list(rs.detect_duplicates(points, rtol=1e-4)) == list(
            bt.detect_duplicates(points, rtol=1e-4)
        )
        for actual, expected in zip(
            rs.consolidate_duplicates(points, comparisons, rtol=1e-4),
            bt.consolidate_duplicates(points, comparisons, rtol=1e-4),
            strict=True,
        ):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    for propagate in (False, True):
        pytest.importorskip("gpytorch")
        import gpytorch

        with settings.propagate_grads(propagate), rs.gpt_posterior_settings():
            assert gpytorch.settings.debug.off()
            assert gpytorch.settings.fast_pred_var.on()
            assert gpytorch.settings.detach_test_caches.on() != propagate


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("fixed_noise", [False, True])
def test_extract_native_training_data(kind: str, fixed_noise: bool) -> None:
    from botorch_rs.models import SingleTaskGP

    X = as_kind([[0.0], [0.5], [1.0]], kind)
    Y = as_kind([[0.0], [1.0], [0.0]], kind)
    variance = as_kind([[0.01], [0.02], [0.03]], kind) if fixed_noise else None
    model = SingleTaskGP(X, Y, train_Yvar=variance, outcome_transform=None)
    targets, noise = rs.extract_targets_and_noise_single_output(model)
    assert to_list(targets) == to_list(Y)
    if fixed_noise:
        assert to_list(noise) == to_list(variance)
    else:
        assert noise is None

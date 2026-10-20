"""Heterogeneous feature partitions and native exact binary kernels."""

from __future__ import annotations

import random

import pytest

from botorch_rs.models.kernels.heterogeneous_multitask import DeltaKernel, find_subsets, map_subsets
from helpers import KINDS, as_kind, to_list


def test_feature_partitions_against_official() -> None:
    bt = pytest.importorskip("botorch.models.kernels.heterogeneous_multitask")
    tasks = [[[1, 2, 3, 4], [1, 2, 5]], [[], [1, 2], [2, 3]], [[]], [[1, 2]], [[0, 2], [0, 2]]]
    rng = random.Random(17)
    for _ in range(30):
        tasks.append(
            [
                [index for index in range(12) if rng.random() < 0.5]
                for _ in range(rng.randrange(1, 7))
            ]
        )
    for feature_indices in tasks:
        actual = find_subsets(feature_indices)
        expected = bt.find_subsets(feature_indices)
        assert actual == expected
        assert map_subsets(actual, feature_indices) == bt.map_subsets(expected, feature_indices)


@pytest.mark.parametrize(
    "tasks, message", [([], "at least|At least"), ([[0, 0]], "unique"), ([[-1]], "nonnegative")]
)
def test_invalid_feature_lists(tasks, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        find_subsets(tasks)


@pytest.mark.parametrize(
    "subsets, message",
    [
        ([{0, 1}, {1, 2}], "disjoint"),
        ([{0}, {2}], "cover"),
        ([{0, 1}, {2, 3}], "absent"),
        ([{0, 2}, {1}], "fully active"),
        ([{0}, {1}, {2}, set(), set()], "unique"),
    ],
)
def test_invalid_feature_partition(subsets, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        map_subsets(subsets, [[0, 1], [1, 2]])


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("batched", [False, True])
def test_delta_kernel_exact_values(kind: str, batched: bool) -> None:
    torch = pytest.importorskip("torch")
    bt = pytest.importorskip("botorch.models.kernels.heterogeneous_multitask")
    X = [[-1.0], [0.0], [1.0], [1.0], [2.0]]
    Z = [[1.0], [2.0], [0.0]]
    if batched:
        X, Z = [X, list(reversed(X))], [Z]
    actual = DeltaKernel()(as_kind(X, kind), as_kind(Z, kind))
    expected = bt.DeltaKernel()(
        torch.tensor(X, dtype=torch.double), torch.tensor(Z, dtype=torch.double)
    ).to_dense()
    torch.testing.assert_close(
        torch.tensor(to_list(actual), dtype=torch.double), expected, rtol=0, atol=0
    )
    diagonal = DeltaKernel()(as_kind(X, kind), diag=True)
    expected_diag = bt.DeltaKernel()(torch.tensor(X, dtype=torch.double), diag=True)
    torch.testing.assert_close(
        torch.tensor(to_list(diagonal), dtype=torch.double), expected_diag, rtol=0, atol=0
    )
    with pytest.raises(AssertionError, match="1D"):
        DeltaKernel()(as_kind([[0.0, 1.0]], kind))
    assert to_list(DeltaKernel(active_dims=[1])(as_kind([[0.0, 1.0], [1.0, 0.0]], kind))) == [
        [1.0, 0.0],
        [0.0, 0.0],
    ]


def test_delta_as_native_model_kernel() -> None:
    from botorch_rs.models import SingleTaskGP

    model = SingleTaskGP(
        [[0.0], [1.0]], [[0.0], [0.7]], covar_module=DeltaKernel(), outcome_transform=None
    )
    posterior = model.posterior([[0.0], [1.0], [2.0]])
    means = to_list(posterior.mean)
    assert means[0] == means[2] == [0.0]
    assert means[1][0] > 0.6
    with pytest.raises(AssertionError, match="1D"):
        SingleTaskGP(
            [[0.0, 1.0], [1.0, 0.0]],
            [[0.0], [0.7]],
            covar_module=DeltaKernel(),
            outcome_transform=None,
        )

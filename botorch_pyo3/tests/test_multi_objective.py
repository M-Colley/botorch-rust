"""The multi-objective utilities (``botorch_rs.utils.multi_objective``) and
``sample_simplex`` / ``sample_hypersphere`` through the public API, for every
array type: result types and shapes, statefulness of the box
decompositions, errors, and properties that do not need BoTorch (the
hypervolumes of all methods agree, the cells tile the space, the
scalarization's gradient). The comparisons with BoTorch are in
``test_parity_mo_utils.py``.
"""

from __future__ import annotations

import math
import random
from typing import Any

import pytest

import botorch_rs
from botorch_rs.exceptions import BotorchError, BotorchTensorDimensionError, UnsupportedError
from botorch_rs.utils.multi_objective import (
    Hypervolume,
    get_chebyshev_scalarization,
    infer_reference_point,
    is_non_dominated,
)
from botorch_rs.utils.multi_objective.box_decompositions import (
    BoxDecompositionList,
    DominatedPartitioning,
    FastNondominatedPartitioning,
    NondominatedPartitioning,
)
from botorch_rs.utils.multi_objective.box_decompositions.box_decomposition import (
    BoxDecomposition,
    FastPartitioning,
)
from botorch_rs.utils.multi_objective.box_decompositions.utils import (
    _pad_batch_pareto_frontier,
    compute_local_upper_bounds,
    get_partition_bounds,
)
from botorch_rs.utils.multi_objective.hypervolume import get_hypervolume_maximizing_subset
from botorch_rs.utils.multi_objective.scalarization import ChebyshevScalarization
from botorch_rs.utils.sampling import sample_hypersphere, sample_simplex
from helpers import HAS_TORCH, as_kind, check_kind, flatten, shape, to_list

CLASSES = (NondominatedPartitioning, FastNondominatedPartitioning, DominatedPartitioning)


def front(n: int, m: int, seed: int) -> list[list[float]]:
    """Points near the positive unit sphere (many Pareto points)."""
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        u = [rng.random() + 1e-3 for _ in range(m)]
        s = (1.0 - 0.2 * rng.random()) / math.sqrt(sum(v * v for v in u))
        out.append([v * s for v in u])
    return out


def check_bool_kind(x: Any, kind: str) -> None:
    if kind == "list":
        assert isinstance(x, list) and all(isinstance(v, bool) for v in flatten_bool(x))
    elif kind == "numpy":
        import numpy as np

        assert isinstance(x, np.ndarray) and x.dtype == np.bool_
    else:
        import torch

        assert isinstance(x, torch.Tensor) and x.dtype == torch.bool


def flatten_bool(x: Any) -> list[bool]:
    x = to_list(x)
    if isinstance(x, list):
        return [v for item in x for v in flatten_bool(item)]
    return [x]


def test_is_non_dominated(kind: str) -> None:
    Y = [[1.0, 0.0], [0.0, 1.0], [0.5, 0.5], [1.0, 0.0], [0.2, 0.2]]
    mask = is_non_dominated(as_kind(Y, kind))
    check_bool_kind(mask, kind)
    assert flatten_bool(mask) == [True, True, True, False, False]
    assert flatten_bool(is_non_dominated(as_kind(Y, kind), deduplicate=False)) == [
        True,
        True,
        True,
        True,
        False,
    ]
    # Minimization: (0.2, 0.2) dominates (0.5, 0.5) only.
    assert flatten_bool(is_non_dominated(as_kind(Y, kind), maximize=False)) == [
        True,
        True,
        False,
        False,
        True,
    ]
    batch = is_non_dominated(as_kind([Y, Y[::-1]], kind))
    assert shape(batch) == (2, 5)
    assert flatten_bool(is_non_dominated(as_kind([[float("nan"), 2.0], [1.0, 1.0]], kind))) == [
        False,
        True,
    ]
    assert flatten_bool(is_non_dominated(as_kind([[[]]], kind))) == [True]  # m = 0
    with pytest.raises(IndexError):
        is_non_dominated(as_kind([1.0, 2.0], kind))


def test_hypervolume_and_reference_points(kind: str) -> None:
    Y = [[1.0, 3.0], [2.0, 2.0], [3.0, 1.0], [0.5, 0.5]]
    hv = Hypervolume(as_kind([0.0, 0.0], kind))
    assert hv.compute(as_kind(Y, kind)) == 6.0
    check_kind(hv.ref_point, kind)
    ref = infer_reference_point(as_kind(Y[:3], kind))
    check_kind(ref, kind)
    assert flatten(ref) == pytest.approx([0.8, 0.8], rel=1e-15)
    if kind != "list":  # nested lists cannot have the shape 0 x 2
        with pytest.raises(BotorchError, match="Empty pareto set"):
            infer_reference_point(as_kind([[0.0, 0.0]], kind)[:0])
    with pytest.raises(BotorchTensorDimensionError, match="same number of objectives"):
        hv.compute(as_kind([[1.0, 2.0, 3.0]], kind))
    sub, idx = get_hypervolume_maximizing_subset(2, as_kind(Y, kind), as_kind([0.0, 0.0], kind))
    check_kind(sub, kind)
    assert flatten(sub) == [2.0, 2.0, 1.0, 3.0]
    assert [int(i) for i in to_list(idx)] == [1, 0]
    if kind == "list":
        assert all(isinstance(i, int) for i in idx)
    else:
        assert str(idx.dtype) in ("int64", "torch.int64")


@pytest.mark.parametrize("cls", CLASSES)
def test_box_decompositions(kind: str, cls: type) -> None:
    Y = front(12, 3, 1)
    ref = [0.1, 0.1, 0.1]
    bd = cls(as_kind(ref, kind), Y=as_kind(Y, kind))
    assert isinstance(bd, BoxDecomposition)
    assert isinstance(bd, FastPartitioning) == (cls is not NondominatedPartitioning)
    assert bd.num_outcomes == 3 and tuple(bd.batch_shape) == ()
    for x in (bd.pareto_Y, bd.Y, bd.get_hypercell_bounds(), bd.ref_point):
        check_kind(x, kind)
    assert shape(bd.Y) == (12, 3)
    hv = bd.compute_hypervolume()
    check_kind(hv, kind)
    expected = Hypervolume(ref).compute(Y)
    assert float(to_list(hv)) == pytest.approx(expected, rel=1e-13)
    new = front(3, 3, 2)
    bd.update(as_kind(new, kind))
    assert shape(bd.Y) == (15, 3)
    assert float(to_list(bd.compute_hypervolume())) == pytest.approx(
        Hypervolume(ref).compute(Y + new), rel=1e-13
    )
    bd.reset()
    assert float(to_list(bd.compute_hypervolume())) == pytest.approx(
        Hypervolume(ref).compute(Y + new), rel=1e-13
    )
    if cls is not NondominatedPartitioning:
        assert shape(bd._U)[1] == 3 and shape(bd._Z)[1:] == (3, 3)


@pytest.mark.parametrize("cls", CLASSES)
def test_batched_box_decompositions(kind: str, cls: type) -> None:
    Ys = [front(8, 2, k) for k in range(3)]
    Ys[1] = [[-1.0, -1.0]] * 8  # no point better than the reference point
    bd = cls(as_kind([0.0, 0.0], kind), Y=as_kind(Ys, kind))
    assert tuple(bd.batch_shape) == (3,)
    hv = bd.compute_hypervolume()
    assert shape(hv) == (3,)
    for y, v in zip(Ys, flatten(hv), strict=True):
        assert v == pytest.approx(Hypervolume([0.0, 0.0]).compute(y), rel=1e-13, abs=1e-15)
    cells = bd.get_hypercell_bounds()
    assert shape(cells)[:2] == (2, 3)


def test_decompositions_without_outcomes() -> None:
    for cls in CLASSES:
        bd = cls([0.0, 0.0])
        assert bd.compute_hypervolume() == 0.0
        with pytest.raises(BotorchError, match="pareto_Y has not been initialized"):
            _ = bd.pareto_Y
        with pytest.raises(BotorchError, match="Y data has not been initialized"):
            bd.reset()
        with pytest.raises(AttributeError):
            _ = bd.batch_shape
        bd.update([[1.0, 2.0], [2.0, 1.0]])
        assert flatten(bd.compute_hypervolume()) == [3.0]
    with pytest.raises(TypeError):
        BoxDecomposition([0.0, 0.0], sort=True)  # abstract
    with pytest.raises(TypeError):
        FastPartitioning([0.0, 0.0])  # abstract


def test_decomposition_errors() -> None:
    with pytest.raises(ValueError, match="NaN inputs are not supported. Got Y with 1 NaN values."):
        FastNondominatedPartitioning([0.0, 0.0], Y=[[float("nan"), 1.0]])
    with pytest.raises(UnsupportedError, match="single batch dimension"):
        DominatedPartitioning([0.0, 0.0], Y=[[[[1.0, 1.0]]]])
    with pytest.raises(NotImplementedError, match="2-objective setting"):
        NondominatedPartitioning([0.0, 0.0, 0.0], Y=[[[1.0, 1.0, 1.0]]])
    with pytest.raises(IndexError, match="non-zero size"):
        DominatedPartitioning([0.0], Y=[[1.0], [2.0]])


def test_hypervolumes_agree_and_cells_tile() -> None:
    rng = random.Random(0)
    for m in (2, 3, 4):
        Y = front(15, m, m)
        ref = [0.05] * m
        expected = Hypervolume(ref).compute(Y)
        bds = [cls(ref, Y=Y) for cls in CLASSES]
        for bd in bds:
            assert flatten(bd.compute_hypervolume())[0] == pytest.approx(expected, rel=1e-13)
        # Monte Carlo in the box [0.05, 1]^m (all points are in [0, 1]^m).
        hits = 0
        draws = 10000
        for _ in range(draws):
            x = [0.05 + 0.95 * rng.random() for _ in range(m)]
            hits += any(all(p[j] >= x[j] for j in range(m)) for p in Y)
        volume = 0.95**m
        assert hits / draws * volume == pytest.approx(
            expected, abs=5 * volume * math.sqrt(0.25 / draws)
        )
        # Each point is in exactly one cell of the right kind.
        nd = [to_list(bds[k].get_hypercell_bounds()) for k in range(3)]
        for _ in range(300):
            x = [0.05 + 1.1 * rng.random() for _ in range(m)]
            dominated = any(all(p[j] > x[j] for j in range(m)) for p in Y)
            for k, cells in enumerate(nd):
                count = sum(
                    all(lo[j] <= x[j] < up[j] for j in range(m))
                    for lo, up in zip(cells[0], cells[1], strict=True)
                )
                assert count == (dominated if k == 2 else not dominated)


def test_box_decomposition_list(kind: str) -> None:
    ref = as_kind([0.1, 0.1, 0.1], kind)
    Ys = [front(5 + 3 * k, 3, 10 + k) for k in range(3)]
    bdl = BoxDecompositionList(*(FastNondominatedPartitioning(ref, Y=as_kind(Y, kind)) for Y in Ys))
    cells = bdl.get_hypercell_bounds()
    check_kind(cells, kind)
    sizes = [shape(bd.get_hypercell_bounds())[1] for bd in bdl.box_decompositions]
    assert shape(cells) == (2, 3, max(sizes), 3)
    hv = bdl.compute_hypervolume()
    assert shape(hv) == (3,)
    for y, v in zip(Ys, flatten(hv), strict=True):
        assert v == pytest.approx(Hypervolume([0.1] * 3).compute(y), rel=1e-13)
    bdl.update([as_kind(front(2, 3, 20 + k), kind) for k in range(3)])
    with pytest.raises(BotorchTensorDimensionError):
        bdl.update([as_kind(front(2, 3, 30), kind)])
    assert len(bdl.pareto_Y) == 3 and shape(bdl.ref_point) == (3, 3)


def test_box_decomposition_utils(kind: str) -> None:
    U = as_kind([[0.0, 0.0, 0.0]], kind)
    inf = float("inf")
    Z = as_kind([[[0.0, -inf, -inf], [-inf, 0.0, -inf], [-inf, -inf, 0.0]]], kind)
    U2, Z2 = compute_local_upper_bounds(U, Z, as_kind([-1.0, -2.0, -3.0], kind))
    check_kind(U2, kind)
    assert shape(U2) == (3, 3) and shape(Z2) == (3, 3, 3)
    bounds = get_partition_bounds(Z2, U2, as_kind([0.0, 0.0, 0.0], kind))
    assert shape(bounds) == (2, 1, 3)
    assert flatten(bounds) == [-1.0, -2.0, -3.0, 0.0, 0.0, 0.0]
    padded = _pad_batch_pareto_frontier(
        as_kind([[[1.0, 2.0], [2.0, 1.0]], [[-1.0, -1.0], [0.5, -2.0]]], kind),
        as_kind([[0.0, 0.0], [0.0, 0.0]], kind),
    )
    assert flatten(padded) == [1.0, 2.0, 2.0, 1.0, 0.0, 0.0, 0.0, 0.0]


def test_chebyshev_scalarization(kind: str) -> None:
    Y = [[0.0, 1.0], [1.0, 3.0], [0.5, 2.0]]
    f = get_chebyshev_scalarization(as_kind([0.3, -0.7], kind), as_kind(Y, kind))
    assert isinstance(f, ChebyshevScalarization)
    X = [[[0.25, 1.5], [1.0, 0.0]]]
    v = f(as_kind(X, kind))
    check_kind(v, kind)
    assert shape(v) == (1, 2)
    p = [0.3 * 0.75, -0.7 * -0.25]
    assert flatten(v)[0] == pytest.approx(-(max(p) + 0.05 * sum(p)), rel=1e-15)
    assert flatten(f(as_kind(X, kind), X=None)) == flatten(v)
    with pytest.raises(BotorchTensorDimensionError, match="weights must be an `m`-dim tensor"):
        get_chebyshev_scalarization(as_kind([0.3], kind), as_kind(Y, kind))
    if kind != "list":  # nested lists cannot have the shape 0 x 2
        with pytest.raises(UnsupportedError, match="negative weights"):
            get_chebyshev_scalarization(as_kind([0.3, -0.7], kind), as_kind([[0.0, 0.0]], kind)[:0])


@pytest.mark.skipif(not HAS_TORCH, reason="requires torch")
def test_chebyshev_scalarization_autograd() -> None:
    import torch

    f = get_chebyshev_scalarization(
        torch.tensor([0.4, 0.6], dtype=torch.float64), torch.rand(5, 2, dtype=torch.float64)
    )
    X = torch.rand(4, 3, 2, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(lambda x: f(x), (X,))


def test_samplers(kind: str) -> None:
    dtype = None
    if kind == "numpy":
        import numpy as np

        dtype = np.float64
    elif kind == "torch":
        import torch

        dtype = torch.float64
    x = sample_simplex(3, 50, seed=1, dtype=dtype)
    assert shape(x) == (50, 3)
    for row in to_list(x):
        assert sum(row) == pytest.approx(1.0, abs=1e-14) and min(row) >= 0.0
    assert flatten(sample_simplex(3, 5, qmc=True, seed=2, dtype=dtype)) == flatten(
        sample_simplex(3, 5, qmc=True, seed=2, dtype=dtype)
    )
    s = sample_hypersphere(4, 40, qmc=True, seed=3, dtype=dtype)
    for row in to_list(s):
        assert math.sqrt(sum(v * v for v in row)) == pytest.approx(1.0, abs=1e-14)
    assert set(flatten(sample_hypersphere(1, 20, seed=0, dtype=dtype))) <= {-1.0, 1.0}


def test_module_layout() -> None:
    assert botorch_rs.utils.multi_objective.is_non_dominated is is_non_dominated
    from botorch_rs.utils.multi_objective import box_decompositions as bd

    assert set(bd.__all__) >= {
        "BoxDecompositionList",
        "DominatedPartitioning",
        "FastNondominatedPartitioning",
        "NondominatedPartitioning",
        "compute_dominated_hypercell_bounds_2d",
        "compute_non_dominated_hypercell_bounds_2d",
    }


def test_respects_num_threads() -> None:
    Ys = [[front(10, 2, 40 + k) for k in range(64)]]
    threads = botorch_rs.get_num_threads()
    try:
        botorch_rs.set_num_threads(1)
        a = flatten(is_non_dominated(Ys))
        botorch_rs.set_num_threads(4)
        b = flatten(is_non_dominated(Ys))
    finally:
        botorch_rs.set_num_threads(threads)
    assert a == b

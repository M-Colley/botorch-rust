"""Parity of the multi-objective utilities with BoTorch, through the public
APIs of both libraries side by side:

* ``is_non_dominated`` (masks identical; m = 1..5, n = 0..300 and BoTorch's
  loop variant for n > 1000 or large batches, also in float32; ties,
  duplicates, infinities, ``maximize=False``, ``deduplicate=False``,
  batches); NaN outcomes against BoTorch with meta-pytorch/botorch#3392 if
  it is installed, else against BoTorch's mask of the points without NaNs
  (the corrected behaviour), checking that BoTorch's own mask differs;
* ``Hypervolume``, ``infer_reference_point`` and
  ``get_hypervolume_maximizing_subset`` (identical selections, including
  BoTorch's random choices under ``random.seed``);
* the three box decompositions after construction, incremental updates
  (with and without new Pareto points, with points worse than the reference
  point) and ``reset``: Pareto sets, cells and local upper bounds identical
  and in BoTorch's order, hypervolumes; batches of fronts of different
  sizes, batched reference points, ``alpha > 0``, empty fronts;
  ``BoxDecompositionList``; the helpers of ``box_decompositions.utils``;
* ``get_chebyshev_scalarization`` (values and autograd gradients);
* ``sample_simplex`` / ``sample_hypersphere`` (the transformations of given
  random numbers, and the moments of the samples, statistically);
* float32 inputs (computed in float64, returned in float32), error types and
  messages, and the absence of warnings.

Tolerances: discrete outputs identical; values relative 1e-12 (hypervolumes
are sums over many cells and reductions in torch sum in a different order);
float32 hypervolumes 1e-6 against BoTorch's float32 computation. The
randomized sweep (``test_equivalence_sweep``) is part of ``-m equivalence``.
Skipped unless BoTorch is importable (required in CI's parity job).
"""

from __future__ import annotations

import random
import warnings
from collections.abc import Iterator
from typing import Any

import pytest

from helpers import importorskip_botorch

importorskip_botorch()
torch = pytest.importorskip("torch")

import botorch.exceptions.errors as bt_errors  # noqa: E402
from botorch.utils import sampling as bt_sampling  # noqa: E402
from botorch.utils.multi_objective import box_decompositions as bt_bd  # noqa: E402
from botorch.utils.multi_objective import hypervolume as bt_hv  # noqa: E402
from botorch.utils.multi_objective import pareto as bt_pareto  # noqa: E402
from botorch.utils.multi_objective import scalarization as bt_scal  # noqa: E402
from botorch.utils.multi_objective.box_decompositions import utils as bt_bdu  # noqa: E402

import botorch_rs.exceptions.errors as rs_errors  # noqa: E402
from botorch_rs import _botorch_rs as _rs  # noqa: E402
from botorch_rs.utils import sampling as rs_sampling  # noqa: E402
from botorch_rs.utils.multi_objective import box_decompositions as rs_bd  # noqa: E402
from botorch_rs.utils.multi_objective import hypervolume as rs_hv  # noqa: E402
from botorch_rs.utils.multi_objective import pareto as rs_pareto  # noqa: E402
from botorch_rs.utils.multi_objective import scalarization as rs_scal  # noqa: E402
from botorch_rs.utils.multi_objective.box_decompositions import utils as rs_bdu  # noqa: E402
from parity_helpers import DT, Checker, rel_err  # noqa: E402

CHECK = Checker("multi-objective utilities")
VALUE_TOL = 1e-12
FLOAT32_HV_TOL = 1e-6
CLASSES = ("NondominatedPartitioning", "FastNondominatedPartitioning", "DominatedPartitioning")

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")


@pytest.fixture(autouse=True, scope="module")
def _single_threaded_torch() -> Iterator[None]:
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(threads)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


def points(n: int, m: int, kind: str, seed: int, batch: tuple[int, ...] = ()) -> Any:
    """Uniform points (small fronts), points near the positive unit sphere
    (large fronts) or grid points (ties); with an exact duplicate."""
    g = torch.Generator().manual_seed(seed)
    shape = (*batch, n, m)
    if kind == "uniform":
        Y = torch.rand(*shape, generator=g, dtype=DT)
    elif kind == "front":
        u = torch.rand(*shape, generator=g, dtype=DT) + 1e-3
        u = u / u.norm(dim=-1, keepdim=True)
        Y = u * (1.0 - 0.15 * torch.rand(*batch, n, 1, generator=g, dtype=DT))
    else:
        Y = torch.round(4.0 * torch.rand(*shape, generator=g, dtype=DT)) / 4.0
    if n >= 4:
        Y[..., n // 2, :] = Y[..., 0, :]
    return Y


def same(a: Any, e: Any) -> bool:
    """Identical tensors (shape, values, infinities and NaNs)."""
    a, e = torch.as_tensor(a), torch.as_tensor(e)
    if a.shape != e.shape:
        return False
    return bool(((a == e) | (a.isnan() & e.isnan())).all())


def bt_has_nan_fix() -> bool:
    Y = torch.tensor([[float("nan"), 0.0], [1.0, 1.0]], dtype=DT)
    return bt_pareto.is_non_dominated(Y, deduplicate=False).tolist() == [False, True]


def nan_reference(Y: Any, maximize: bool, deduplicate: bool) -> Any:
    """The mask with meta-pytorch/botorch#3392: BoTorch's mask of the points
    without NaNs, and ``False`` for the points with NaNs."""
    if bt_has_nan_fix():
        return bt_pareto.is_non_dominated(Y, maximize=maximize, deduplicate=deduplicate)
    flat = Y.reshape(-1, *Y.shape[-2:])
    out = torch.zeros(flat.shape[:-1], dtype=torch.bool)
    for b in range(flat.shape[0]):
        ok = ~flat[b].isnan().any(dim=-1)
        if ok.any():
            out[b, ok] = bt_pareto.is_non_dominated(
                flat[b, ok], maximize=maximize, deduplicate=deduplicate
            )
    return out.reshape(Y.shape[:-1])


# ---------------------------------------------------------------------------
# Pareto masks
# ---------------------------------------------------------------------------


def check_masks(label: str, Y: Any) -> None:
    for maximize in (True, False):
        for dedup in (True, False):
            a = rs_pareto.is_non_dominated(Y, maximize=maximize, deduplicate=dedup)
            e = bt_pareto.is_non_dominated(Y, maximize=maximize, deduplicate=dedup)
            assert a.dtype == torch.bool and torch.equal(a, e), (label, maximize, dedup)
    CHECK.record("is_non_dominated: masks (identical)", 0.0, 0.0)


@pytest.mark.parametrize("m", [1, 2, 3, 4, 5])
def test_is_non_dominated(m: int) -> None:
    for n in (0, 1, 2, 7, 50, 300):
        for k, kind in enumerate(("uniform", "front", "grid")):
            Y = points(n, m, kind, seed=100 * m + n + k)
            if n >= 7 and m >= 2:
                Y[3, 0] = float("inf")
                Y[5, 1] = float("-inf")
            check_masks(f"m{m} n{n} {kind}", Y)


def test_is_non_dominated_loop_variant_and_batches() -> None:
    # BoTorch's loop variant: n > 1000, and (in float32 with its smaller
    # elements) batches above 5 MB.
    for Y in (points(1200, 2, "front", 1), points(1100, 3, "grid", 2)):
        check_masks("loop", Y)
        check_masks("loop float32", Y.float())
    for batch, n, m in (((2, 3), 20, 2), ((10,), 300, 2), ((30,), 160, 3)):
        Y = points(n, m, "front", 3, batch=batch)
        check_masks(f"batch {batch}", Y)
        check_masks(f"batch {batch} float32", Y.float())


def test_is_non_dominated_nan() -> None:
    for n, m, rows in ((6, 2, [0]), (30, 3, [2, 7, 11]), (1200, 2, [0, 10, 500])):
        Y = points(n, m, "front", n)
        for r in rows:
            Y[r, r % m] = float("nan")
        Y[min(n - 1, 4)] = float("nan")  # an all-NaN point
        for maximize in (True, False):
            for dedup in (True, False):
                a = rs_pareto.is_non_dominated(Y, maximize=maximize, deduplicate=dedup)
                assert torch.equal(a, nan_reference(Y, maximize, dedup)), (n, maximize, dedup)
                assert not a[torch.as_tensor(rows)].any()
    if not bt_has_nan_fix():
        # BoTorch main keeps partially NaN points (the bug fixed by #3392).
        Y = torch.tensor([[float("nan"), 0.0], [1.0, 1.0]], dtype=DT)
        assert bt_pareto.is_non_dominated(Y, deduplicate=False).tolist() == [True, True]
        assert rs_pareto.is_non_dominated(Y, deduplicate=False).tolist() == [False, True]
    CHECK.record("is_non_dominated with NaNs: masks (identical, #3392)", 0.0, 0.0)


# ---------------------------------------------------------------------------
# Hypervolume
# ---------------------------------------------------------------------------


def test_hypervolume() -> None:
    sizes = {1: (0, 1, 30), 2: (0, 1, 5, 50, 300), 3: (2, 10, 60, 120), 4: (5, 20, 40), 5: (5, 12)}
    for m, ns in sizes.items():
        for n in ns:
            for k, kind in enumerate(("uniform", "front", "grid")):
                Y = points(n, m, kind, seed=1000 + 10 * m + n + k)
                ref = torch.full((m,), 0.1, dtype=DT)
                for pts in (Y, Y[bt_pareto.is_non_dominated(Y)] if n else Y):
                    a = rs_hv.Hypervolume(ref).compute(pts)
                    e = bt_hv.Hypervolume(ref).compute(pts)
                    assert isinstance(a, float)
                    CHECK.check("Hypervolume.compute", a, e, VALUE_TOL)
                    # float32: computed in float64 from the float32 values.
                    a32 = rs_hv.Hypervolume(ref.float()).compute(pts.float())
                    e64 = bt_hv.Hypervolume(ref.float().double()).compute(pts.float().double())
                    e32 = bt_hv.Hypervolume(ref.float()).compute(pts.float())
                    CHECK.check(
                        "Hypervolume.compute (float32 vs float64 BoTorch)", a32, e64, VALUE_TOL
                    )
                    CHECK.check(
                        "Hypervolume.compute (float32 vs float32 BoTorch)", a32, e32, FLOAT32_HV_TOL
                    )
    hv = rs_hv.Hypervolume([0.0, 0.0])
    assert hv.ref_point == [0.0, 0.0]
    hv.ref_point = torch.tensor([1.0, -1.0], dtype=DT)
    assert torch.equal(hv.ref_point, torch.tensor([1.0, -1.0], dtype=DT))


def test_infer_reference_point() -> None:
    g = torch.Generator().manual_seed(5)
    P2 = torch.rand(10, 2, generator=g, dtype=DT)
    P3 = torch.rand(10, 3, generator=g, dtype=DT)
    nan = float("nan")
    cases = [
        (P2, None),
        (P3, None),
        (P2, torch.tensor([0.1, 0.1], dtype=DT)),
        (P3, torch.tensor([0.9, 0.95, 0.99], dtype=DT)),
        (P3, torch.tensor([0.2, nan, 0.3], dtype=DT)),
        (P3, torch.tensor([2.0, nan, 0.3], dtype=DT)),
        (P2, torch.tensor([nan, nan], dtype=DT)),
        (P3[:1], None),
        (P3[:1], torch.zeros(3, dtype=DT)),
        (torch.tensor([[0.0, 2.0]], dtype=DT), None),
        (torch.tensor([[1.0, 0.5], [1.0, 0.2], [1.0, 0.9]], dtype=DT), None),
        (P2[:0], torch.tensor([0.5, -1.5], dtype=DT)),
    ]
    for P, max_ref in cases:
        for scale in (0.1, 0.3):
            for smr in (False, True):
                kw = {"max_ref_point": max_ref, "scale": scale, "scale_max_ref_point": smr}
                a = rs_hv.infer_reference_point(P, **kw)
                e = bt_hv.infer_reference_point(P, **kw)
                assert a.dtype == e.dtype
                CHECK.check("infer_reference_point", a, e, VALUE_TOL)


def test_get_hypervolume_maximizing_subset() -> None:
    for m, n, k, raw in ((2, 12, 4, False), (3, 15, 6, False), (2, 15, 15, True), (3, 10, 8, True)):
        Y = points(n, m, "front", 3000 + n * m)
        ref = torch.zeros(m, dtype=DT)
        if raw:
            Y[1] = -0.5  # does not dominate the reference point
            Y[2] = 0.5 * Y[0]  # dominated
        else:
            Y = Y[bt_pareto.is_non_dominated(Y)]
        for seed in (0, 1, 2):
            random.seed(seed)
            a_sub, a_idx = rs_hv.get_hypervolume_maximizing_subset(k, Y, ref)
            state = random.getstate()
            random.seed(seed)
            e_sub, e_idx = bt_hv.get_hypervolume_maximizing_subset(k, Y, ref)
            assert random.getstate() == state  # the same random numbers were drawn
            assert a_idx.dtype == torch.long and torch.equal(a_idx, e_idx), (m, n, seed)
            assert torch.equal(a_sub, e_sub)
    CHECK.record("get_hypervolume_maximizing_subset: selections (identical)", 0.0, 0.0)


# ---------------------------------------------------------------------------
# Box decompositions
# ---------------------------------------------------------------------------


def make_pair(name: str, ref: Any, Y: Any, alpha: float = 0.0) -> tuple[Any, Any]:
    kw = {"alpha": alpha} if alpha > 0 else {}
    return getattr(rs_bd, name)(ref_point=ref, Y=Y, **kw), getattr(bt_bd, name)(
        ref_point=ref, Y=Y, **kw
    )


def check_state(label: str, a: Any, e: Any, float32: bool = False) -> None:
    assert same(a.pareto_Y, e.pareto_Y), (label, "pareto_Y")
    assert same(a.Y, e.Y), (label, "Y")
    assert a.num_outcomes == e.num_outcomes and a.batch_shape == e.batch_shape, label
    ca, ce = a.get_hypercell_bounds(), e.get_hypercell_bounds()
    assert ca.shape == ce.shape, (label, ca.shape, ce.shape)
    assert ca.dtype == ce.dtype, label
    CHECK.check("decompositions: cell bounds (in BoTorch's order)", ca, ce, VALUE_TOL)
    ha, he = a.compute_hypervolume(), e.compute_hypervolume()
    assert ha.shape == he.shape and ha.dtype == he.dtype, label
    if float32:
        CHECK.check("decompositions: hypervolume (float32)", ha, he, FLOAT32_HV_TOL)
    else:
        CHECK.check("decompositions: hypervolume", ha, he, VALUE_TOL)
    if hasattr(e, "_U") and e.num_outcomes > 2:
        assert same(a._U, e._U) and same(a._Z, e._Z), (label, "local upper bounds")
        CHECK.record("decompositions: local upper bounds (identical, in order)", 0.0, 0.0)


def run_decomposition(name: str, ref: Any, Y: Any, updates: list[Any], alpha: float = 0.0) -> None:
    a, e = make_pair(name, ref, Y, alpha)
    label = f"{name} m{Y.shape[-1]} n{Y.shape[-2]} alpha={alpha}"
    check_state(label, a, e)
    for k, upd in enumerate(updates):
        a.update(upd)
        e.update(upd)
        check_state(f"{label} update {k}", a, e)
    a.reset()
    e.reset()
    check_state(f"{label} reset", a, e)


@pytest.mark.parametrize("name", CLASSES)
@pytest.mark.parametrize("m", [2, 3, 4, 5])
def test_box_decompositions(name: str, m: int) -> None:
    sizes = {2: (0, 1, 8, 60, 300), 3: (1, 10, 40, 100), 4: (6, 15, 30), 5: (5, 12)}[m]
    for n in sizes:
        if name == "NondominatedPartitioning" and n > (40 if m == 3 else 15 if m == 4 else 1000):
            continue  # exact binary partitionings get too large
        for k, kind in enumerate(("uniform", "front", "grid")):
            seed = 4000 + 100 * m + n + k
            Y = points(n, m, kind, seed)
            ref = torch.full((m,), 0.2, dtype=DT)
            g = torch.Generator().manual_seed(seed)
            new_pareto = 0.95 + 0.1 * torch.rand(3, m, generator=g, dtype=DT)
            dominated = 0.21 + 0.01 * torch.rand(2, m, generator=g, dtype=DT)
            worse = 0.1 * torch.rand(2, m, generator=g, dtype=DT)
            run_decomposition(name, ref, Y, [new_pareto, dominated, worse, Y[:2]])
            if name == "NondominatedPartitioning" and m > 2 and n >= 10:
                for alpha in (1e-3, 1e-2, 0.1):
                    run_decomposition(name, ref, Y, [new_pareto], alpha=alpha)


@pytest.mark.parametrize("name", CLASSES)
def test_batched_box_decompositions(name: str) -> None:
    for n, batched_ref in ((1, False), (12, False), (40, True), (150, False)):
        Y = points(n, 2, "front", 5000 + n, batch=(5,))
        Y[1] = Y[1] * 0.5  # a smaller front
        Y[3] = -1.0  # an empty front
        if n >= 4:
            Y[2, : n // 2] = Y[2, :1]
        ref = torch.zeros(2, dtype=DT)
        if batched_ref:
            ref = torch.tensor(
                [[0.0, 0.0], [0.1, 0.0], [0.0, 0.2], [0.0, 0.0], [-0.5, -0.5]], dtype=DT
            )
        upd = 1.2 * torch.rand(5, 3, 2, generator=torch.Generator().manual_seed(n), dtype=DT)
        run_decomposition(name, ref, Y, [upd])
    # All fronts empty.
    Y = -torch.rand(3, 4, 2, dtype=DT)
    run_decomposition(name, torch.zeros(2, dtype=DT), Y, [Y + 0.5])


def test_batched_reference_points_without_pareto_points() -> None:
    """Deviation: with a batched reference point and no Pareto point in any
    batch element, BoTorch's ``FastNondominatedPartitioning`` and
    ``DominatedPartitioning`` fail to broadcast the reference point; botorch_rs
    returns one cell per batch element, like with a shared reference point."""
    Y = torch.rand(3, 4, 2, generator=torch.Generator().manual_seed(0), dtype=DT)
    ref = torch.full((3, 2), 2.0, dtype=DT)
    ref[1] = 3.0
    for name, upper in (("FastNondominatedPartitioning", None), ("DominatedPartitioning", ref)):
        with pytest.raises(RuntimeError):
            getattr(bt_bd, name)(ref_point=ref, Y=Y)
        a = getattr(rs_bd, name)(ref_point=ref, Y=Y)
        cells = a.get_hypercell_bounds()
        assert cells.shape == (2, 3, 1, 2)
        assert torch.equal(cells[0, :, 0], ref)
        expected_upper = torch.full_like(ref, float("inf")) if upper is None else upper
        assert torch.equal(cells[1, :, 0], expected_upper)
        assert torch.equal(a.compute_hypervolume(), torch.zeros(3, dtype=DT))
        # NondominatedPartitioning supports this case in BoTorch.
    a, e = make_pair("NondominatedPartitioning", ref, Y)
    check_state("NondominatedPartitioning batched reference point", a, e)


def test_box_decompositions_float32() -> None:
    for name in CLASSES:
        for m in (2, 3):
            Y = points(12, m, "front", 6000 + m).float()
            ref = torch.full((m,), 0.2)
            a, e = make_pair(name, ref, Y)
            check_state(f"{name} float32", a, e, float32=True)
            # Against BoTorch in float64 on the same values: equal up to the
            # final rounding of the float64 result to float32.
            e64 = getattr(bt_bd, name)(ref_point=ref.double(), Y=Y.double())
            assert a.get_hypercell_bounds().dtype == torch.float32
            assert same(a.get_hypercell_bounds(), e64.get_hypercell_bounds().float())
            CHECK.check(
                "decompositions: hypervolume (float32 result vs float64 BoTorch)",
                a.compute_hypervolume(),
                e64.compute_hypervolume(),
                2.0**-24,
            )


def test_box_decompositions_without_outcomes() -> None:
    for name in CLASSES:
        ref = torch.zeros(3, dtype=DT)
        a, e = getattr(rs_bd, name)(ref_point=ref), getattr(bt_bd, name)(ref_point=ref)
        ha, he = a.compute_hypervolume(), e.compute_hypervolume()
        assert ha.dtype == he.dtype and ha.shape == he.shape and ha.item() == he.item() == 0.0
        assert a.num_outcomes == e.num_outcomes == 3
        for attr in ("pareto_Y", "Y"):
            with pytest.raises(rs_errors.BotorchError) as ea:
                getattr(a, attr)
            with pytest.raises(bt_errors.BotorchError) as eb:
                getattr(e, attr)
            assert str(ea.value) == str(eb.value)
        if name == "NondominatedPartitioning":
            with pytest.raises(AttributeError):
                a.get_hypercell_bounds()
            with pytest.raises(AttributeError):
                e.get_hypercell_bounds()
        else:
            assert a.get_hypercell_bounds() is None and e.get_hypercell_bounds() is None
        Y = points(9, 3, "front", 7)
        a.update(Y)
        e.update(Y)
        check_state(f"{name} first update", a, e)


def test_box_decomposition_list() -> None:
    ref = torch.full((3,), 0.1, dtype=DT)
    Ys = [points(6 + 3 * k, 3, "front", 7000 + k) for k in range(4)]
    a = rs_bd.BoxDecompositionList(*(rs_bd.FastNondominatedPartitioning(ref, Y=Y) for Y in Ys))
    e = bt_bd.BoxDecompositionList(*(bt_bd.FastNondominatedPartitioning(ref, Y=Y) for Y in Ys))
    for step in range(3):
        CHECK.check(
            "BoxDecompositionList: padded cell bounds",
            a.get_hypercell_bounds(),
            e.get_hypercell_bounds(),
            VALUE_TOL,
        )
        CHECK.check(
            "BoxDecompositionList: hypervolumes",
            a.compute_hypervolume(),
            e.compute_hypervolume(),
            VALUE_TOL,
        )
        assert all(same(x, y) for x, y in zip(a.pareto_Y, e.pareto_Y, strict=True))
        assert same(a.ref_point, e.ref_point)
        new = points(2, 3, "front", 7100 + step, batch=(4,)) * 1.1
        a.update(new if step == 0 else list(new))
        e.update(new if step == 0 else list(new))
    with pytest.raises(rs_errors.BotorchTensorDimensionError) as ea:
        a.update([new[0]])
    with pytest.raises(bt_errors.BotorchTensorDimensionError) as eb:
        e.update([new[0]])
    assert str(ea.value) == str(eb.value)


def test_box_decomposition_utils() -> None:
    for m, n in ((2, 8), (3, 12), (4, 10)):
        neg = -points(n, m, "front", 8000 + m)
        U = torch.zeros(1, m, dtype=DT)
        Z = torch.zeros(1, m, m, dtype=DT)
        for j in range(m):
            Z[0, j] = float("-inf")
            Z[0, j, j] = U[0, j]
        Ua, Za, Ue, Ze = U, Z, U, Z
        for i in range(n):
            Ua, Za = rs_bdu.compute_local_upper_bounds(Ua, Za, neg[i])
            Ue, Ze = bt_bdu.compute_local_upper_bounds(Ue, Ze, neg[i])
            assert same(Ua, Ue) and same(Za, Ze), (m, i)
        Ui, Zi = rs_bdu.update_local_upper_bounds_incremental(neg, U, Z)
        assert same(Ui, Ue) and same(Zi, Ze)
        ref = torch.zeros(m, dtype=DT)
        assert same(
            rs_bdu.get_partition_bounds(Za, Ua, ref), bt_bdu.get_partition_bounds(Ze, Ue, ref)
        )
    for batch, batched_ref in (((), False), ((3,), False), ((3,), True)):
        Y = points(6, 2, "front", 8100, batch=batch) + 0.5
        Y = Y.gather(-2, torch.argsort(Y[..., :1], dim=-2).expand(Y.shape))
        ref = torch.tensor([0.1, 0.2], dtype=DT)
        if batched_ref:
            ref = torch.tensor([[0.1, 0.2], [0.0, 0.0], [0.3, 0.1]], dtype=DT)
        for f in (
            "compute_non_dominated_hypercell_bounds_2d",
            "compute_dominated_hypercell_bounds_2d",
        ):
            assert same(getattr(rs_bdu, f)(Y, ref), getattr(bt_bdu, f)(Y, ref)), f
    Y = points(10, 2, "grid", 8200, batch=(4,))
    ref = torch.full((4, 2), 0.25, dtype=DT)
    feas = torch.rand(4, 10, generator=torch.Generator().manual_seed(1)) > 0.3
    for kw in ({}, {"feasibility_mask": feas}, {"is_pareto": True}):
        assert same(
            rs_bdu._pad_batch_pareto_frontier(Y, ref, **kw),
            bt_bdu._pad_batch_pareto_frontier(Y, ref, **kw),
        )
    r = torch.tensor([0.5, -1.0], dtype=DT)
    assert same(
        rs_bdu._expand_ref_point(r, torch.Size([3])), bt_bdu._expand_ref_point(r, torch.Size([3]))
    )
    CHECK.record("box decomposition utils (identical)", 0.0, 0.0)


# ---------------------------------------------------------------------------
# Scalarization and sampling
# ---------------------------------------------------------------------------


def test_chebyshev_scalarization() -> None:
    g = torch.Generator().manual_seed(9)
    for weights, n, alpha in (
        ([0.3, 0.7], 10, 0.05),
        ([0.5, -0.5], 10, 0.05),
        ([-0.2, 0.3, 0.5], 1, 0.0),
        ([0.1, 0.2, 0.3, 0.4], 25, 0.5),
        ([0.2, 0.3, 0.5], 0, 0.05),
    ):
        w = torch.tensor(weights, dtype=DT)
        Y = torch.randn(n, len(weights), generator=g, dtype=DT)
        if n > 1:
            Y[:, 0] = 1.5  # a constant outcome
        fa = rs_scal.get_chebyshev_scalarization(w, Y, alpha=alpha)
        fe = bt_scal.get_chebyshev_scalarization(w, Y, alpha=alpha)
        X = 2.0 * torch.randn(3, 7, len(weights), generator=g, dtype=DT)
        CHECK.check("Chebyshev scalarization: values", fa(X), fe(X), VALUE_TOL)
        Xa = X.clone().requires_grad_(True)
        Xe = X.clone().requires_grad_(True)
        (ga,) = torch.autograd.grad(fa(Xa).sum(), Xa)
        (ge,) = torch.autograd.grad(fe(Xe).sum(), Xe)
        CHECK.check("Chebyshev scalarization: gradients", ga, ge, VALUE_TOL)
        if n > 0:
            assert same(fa.Y_bounds, torch.stack([(-Y).min(0).values, (-Y).max(0).values]))


def test_sampling_transformations() -> None:
    for d, n, seed in ((2, 5, 0), (3, 7, 1), (5, 4, 2)):
        with bt_sampling.manual_seed(seed):
            u = torch.rand(n, d - 1, dtype=DT)
        expected = bt_sampling.sample_simplex(d=d, n=n, seed=seed, dtype=DT)
        data, shape = _rs.simplex_from_uniform((u.reshape(-1).tolist(), tuple(u.shape)))
        CHECK.check(
            "sample_simplex: transformation",
            torch.tensor(data, dtype=DT).reshape(shape),
            expected,
            0.0,
        )
        with bt_sampling.manual_seed(seed):
            z = torch.randn(n, d, dtype=DT)
        expected = bt_sampling.sample_hypersphere(d=d, n=n, seed=seed, dtype=DT)
        data, shape = _rs.hypersphere_from_normal((z.reshape(-1).tolist(), tuple(z.shape)))
        CHECK.check(
            "sample_hypersphere: transformation",
            torch.tensor(data, dtype=DT).reshape(shape),
            expected,
            1e-15,
        )


def sample_moments(x: Any) -> tuple[Any, Any]:
    return x.mean(dim=0), torch.cov(x.T)


@pytest.mark.parametrize("qmc", [False, True])
def test_sampling_distributions(qmc: bool) -> None:
    """The moments of 16384 samples agree with BoTorch's and with the
    distribution's (simplex: Dirichlet(1, ..., 1); sphere: mean 0,
    covariance I / d). The tolerances are about 5 standard errors of the
    i.i.d. estimates; quasi-random samples are far more accurate."""
    n = 16384
    mean_tol, cov_tol = (0.03, 0.015) if not qmc else (0.002, 0.002)
    tag = "QMC" if qmc else "i.i.d."
    for f, d in (("sample_simplex", 4), ("sample_hypersphere", 3)):
        a = getattr(rs_sampling, f)(d, n, qmc=qmc, seed=3, dtype=DT)
        e = getattr(bt_sampling, f)(d, n, qmc=qmc, seed=3, dtype=DT)
        assert a.shape == e.shape and a.dtype == e.dtype
        if f == "sample_simplex":
            mean = torch.full((d,), 1.0 / d, dtype=DT)
            cov = (torch.eye(d, dtype=DT) * d - 1.0) / (d * d * (d + 1))
            assert torch.allclose(a.sum(-1), torch.ones(n, dtype=DT)) and (a >= 0).all()
        else:
            mean = torch.zeros(d, dtype=DT)
            cov = torch.eye(d, dtype=DT) / d
            assert torch.allclose(a.norm(dim=-1), torch.ones(n, dtype=DT))
        (ma, ca), (me, ce) = sample_moments(a), sample_moments(e)
        CHECK.check_bound(
            f"{f} ({tag}): mean vs BoTorch (statistical)", (ma - me).abs().max().item(), mean_tol
        )
        CHECK.check_bound(
            f"{f} ({tag}): covariance vs BoTorch (statistical)",
            (ca - ce).abs().max().item(),
            cov_tol,
        )
        CHECK.check_bound(
            f"{f} ({tag}): mean vs exact (statistical)", (ma - mean).abs().max().item(), mean_tol
        )
        CHECK.check_bound(
            f"{f} ({tag}): covariance vs exact (statistical)",
            (ca - cov).abs().max().item(),
            cov_tol,
        )
    one = rs_sampling.sample_hypersphere(1, 64, seed=0, dtype=DT)
    assert set(one.reshape(-1).tolist()) <= {-1.0, 1.0}
    assert torch.equal(rs_sampling.sample_simplex(1, 3, dtype=DT), torch.ones(3, 1, dtype=DT))


# ---------------------------------------------------------------------------
# Errors and warnings
# ---------------------------------------------------------------------------

# BoTorch's exception type -> the botorch_rs type it must be an instance of.
ERROR_TYPES = {
    "BotorchError": rs_errors.BotorchError,
    "BotorchTensorDimensionError": rs_errors.BotorchTensorDimensionError,
    "UnsupportedError": rs_errors.UnsupportedError,
    "ValueError": ValueError,
    "NotImplementedError": NotImplementedError,
    "IndexError": IndexError,
}


def error_cases() -> list[tuple[str, Any, Any]]:
    nanY = torch.rand(10, 2, dtype=DT)
    nanY[8:, 1] = float("nan")
    Y3 = torch.rand(2, 4, 3, dtype=DT)
    z2 = torch.zeros(2, dtype=DT)
    cases: list[tuple[str, Any, Any]] = []
    for name in CLASSES:
        cases += [
            (f"{name} NaN", lambda c, n=name: getattr(c, n)(ref_point=z2, Y=nanY), None),
            (
                f"{name} NaN update",
                lambda c, n=name: getattr(c, n)(ref_point=z2, Y=nanY[:2]).update(nanY),
                None,
            ),
            (
                f"{name} two batch dims",
                lambda c, n=name: getattr(c, n)(ref_point=z2, Y=torch.rand(2, 2, 3, 2, dtype=DT)),
                None,
            ),
            (
                f"{name} batched m=3",
                lambda c, n=name: getattr(c, n)(ref_point=torch.zeros(3, dtype=DT), Y=Y3),
                None,
            ),
            (
                f"{name} batched ref point for unbatched Y",
                lambda c, n=name: getattr(c, n)(
                    ref_point=torch.zeros(1, 2, dtype=DT), Y=torch.rand(3, 2, dtype=DT)
                ),
                None,
            ),
            (
                f"{name} reset without Y",
                lambda c, n=name: getattr(c, n)(ref_point=z2).reset(),
                None,
            ),
        ]
    Y1 = torch.rand(3, 1, dtype=DT)
    cases += [
        (
            "DominatedPartitioning m=1",
            lambda c: c.DominatedPartitioning(ref_point=torch.zeros(1, dtype=DT), Y=Y1),
            None,
        ),
        (
            "Hypervolume objectives",
            lambda h: h.Hypervolume(z2).compute(torch.rand(3, 3, dtype=DT)),
            "hv",
        ),
        (
            "Hypervolume dims",
            lambda h: h.Hypervolume(z2).compute(torch.rand(2, 3, 2, dtype=DT)),
            "hv",
        ),
        ("infer empty", lambda h: h.infer_reference_point(torch.rand(0, 2, dtype=DT)), "hv"),
        (
            "infer empty NaN",
            lambda h: h.infer_reference_point(
                torch.rand(0, 2, dtype=DT), torch.tensor([float("nan"), 1.0], dtype=DT)
            ),
            "hv",
        ),
        (
            "subset batch",
            lambda h: h.get_hypervolume_maximizing_subset(2, torch.rand(2, 3, 2, dtype=DT), z2),
            "hv",
        ),
        (
            "subset too large",
            lambda h: h.get_hypervolume_maximizing_subset(5, torch.rand(3, 2, dtype=DT), z2),
            "hv",
        ),
        (
            "chebyshev weights",
            lambda s: s.get_chebyshev_scalarization(z2, torch.rand(3, 3, dtype=DT)),
            "scal",
        ),
        (
            "chebyshev batch",
            lambda s: s.get_chebyshev_scalarization(z2, torch.rand(2, 3, 2, dtype=DT)),
            "scal",
        ),
        (
            "chebyshev negative weights",
            lambda s: s.get_chebyshev_scalarization(
                torch.tensor([0.5, -0.5], dtype=DT), torch.rand(0, 2, dtype=DT)
            ),
            "scal",
        ),
        (
            "pad two batch dims",
            lambda u: u._pad_batch_pareto_frontier(torch.rand(2, 2, 3, 2, dtype=DT), z2),
            "utils",
        ),
        (
            "expand mismatch",
            lambda u: u._expand_ref_point(torch.zeros(2, 2, dtype=DT), torch.Size([3])),
            "utils",
        ),
    ]
    return cases


def test_errors_and_messages() -> None:
    modules = {
        None: (rs_bd, bt_bd),
        "hv": (rs_hv, bt_hv),
        "scal": (rs_scal, bt_scal),
        "utils": (rs_bdu, bt_bdu),
    }
    for label, call, mod in error_cases():
        rs_mod, bt_mod = modules[mod]
        with pytest.raises(Exception) as eb:
            call(bt_mod)
        with pytest.raises(Exception) as ea:
            call(rs_mod)
        expected = ERROR_TYPES[type(eb.value).__name__]
        assert isinstance(ea.value, expected), (label, type(ea.value), type(eb.value))
        assert str(ea.value) == str(eb.value), (label, str(ea.value), str(eb.value))
    CHECK.record("error types and messages (identical)", 0.0, 0.0)


def test_no_warnings() -> None:
    """Neither library warns, e.g. for an alpha outside [0, 1]."""
    Y = points(15, 3, "front", 11)
    ref = torch.zeros(3, dtype=DT)
    for lib in (rs_bd, bt_bd):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            for alpha in (-1.0, 0.0, 0.5, 2.0):
                p = lib.NondominatedPartitioning(ref_point=ref, Y=Y, alpha=alpha)
                p.compute_hypervolume()
                p.get_hypercell_bounds()
        assert not [w for w in caught if not issubclass(w.category, DeprecationWarning)], lib
    for alpha in (-1.0, 0.5, 2.0):
        a, e = make_pair("NondominatedPartitioning", ref, Y, alpha)
        check_state(f"alpha={alpha}", a, e)


def test_box_decompositions_with_infinite_outcomes() -> None:
    for m in (2, 3):
        Y = points(8, m, "front", 12 + m)
        Y[2, 0] = float("inf")
        ref = torch.zeros(m, dtype=DT)
        for name in CLASSES:
            a, e = make_pair(name, ref, Y)
            assert same(a.pareto_Y, e.pareto_Y), name
            assert same(a.get_hypercell_bounds(), e.get_hypercell_bounds()), name
            assert same(a.compute_hypervolume(), e.compute_hypervolume()), name


SIGNATURES = [
    (bt_pareto.is_non_dominated, rs_pareto.is_non_dominated),
    (bt_hv.Hypervolume, rs_hv.Hypervolume),
    (bt_hv.Hypervolume.compute, rs_hv.Hypervolume.compute),
    (bt_hv.infer_reference_point, rs_hv.infer_reference_point),
    (bt_hv.get_hypervolume_maximizing_subset, rs_hv.get_hypervolume_maximizing_subset),
    *[(getattr(bt_bd, n), getattr(rs_bd, n)) for n in (*CLASSES, "BoxDecompositionList")],
    *[
        (getattr(bt_bd.BoxDecompositionList, n), getattr(rs_bd.BoxDecompositionList, n))
        for n in ("update", "get_hypercell_bounds", "compute_hypervolume")
    ],
    *[
        (
            getattr(bt_bd.FastNondominatedPartitioning, n),
            getattr(rs_bd.FastNondominatedPartitioning, n),
        )
        for n in (
            "update",
            "reset",
            "partition_space",
            "compute_hypervolume",
            "get_hypercell_bounds",
        )
    ],
    *[
        (getattr(bt_bdu, n), getattr(rs_bdu, n))
        for n in (
            "_expand_ref_point",
            "_pad_batch_pareto_frontier",
            "compute_local_upper_bounds",
            "get_partition_bounds",
            "update_local_upper_bounds_incremental",
            "compute_non_dominated_hypercell_bounds_2d",
            "compute_dominated_hypercell_bounds_2d",
        )
    ],
    (bt_scal.get_chebyshev_scalarization, rs_scal.get_chebyshev_scalarization),
    (bt_sampling.sample_simplex, rs_sampling.sample_simplex),
    (bt_sampling.sample_hypersphere, rs_sampling.sample_hypersphere),
]


@pytest.mark.parametrize(
    ("bt_obj", "rs_obj"), SIGNATURES, ids=[r.__qualname__ for _, r in SIGNATURES]
)
def test_signatures_match_botorch(bt_obj: Any, rs_obj: Any) -> None:
    """BoTorch's parameters, in order, with the same plain defaults."""
    import inspect

    def params(obj: Any) -> list[tuple[str, Any, Any]]:
        sig = inspect.signature(obj.__init__ if inspect.isclass(obj) else obj)
        return [(p.name, p.kind, p.default) for p in sig.parameters.values() if p.name != "self"]

    assert params(rs_obj) == params(bt_obj)


# ---------------------------------------------------------------------------
# The randomized equivalence sweep
# ---------------------------------------------------------------------------


SWEEP = [(seed, m) for seed in range(6) for m in (2, 3, 4, 5)]


@pytest.mark.equivalence
@pytest.mark.parametrize(("seed", "m"), SWEEP)
def test_equivalence_sweep(seed: int, m: int) -> None:
    """Random outcome sets, reference points, updates and flags."""
    r = random.Random(1000 * seed + m)
    n = r.choice({2: [0, 3, 30, 120, 250], 3: [1, 10, 40, 90], 4: [4, 12, 25], 5: [3, 8, 12]}[m])
    kind = r.choice(["uniform", "front", "grid"])
    Y = points(n, m, kind, 9000 + 37 * seed + m)
    if n > 2 and r.random() < 0.3:
        Y[r.randrange(n), r.randrange(m)] = float("inf")
    ref = torch.tensor([r.uniform(-0.1, 0.4) for _ in range(m)], dtype=DT)
    check_masks("sweep", Y)
    finite = Y[torch.isfinite(Y).all(dim=-1)]
    pareto = finite[bt_pareto.is_non_dominated(finite)] if finite.shape[0] else finite
    CHECK.check(
        "sweep: Hypervolume.compute",
        rs_hv.Hypervolume(ref).compute(pareto),
        bt_hv.Hypervolume(ref).compute(pareto),
        VALUE_TOL,
    )
    if pareto.shape[0]:
        scale = r.uniform(0.05, 0.3)
        CHECK.check(
            "sweep: infer_reference_point",
            rs_hv.infer_reference_point(pareto, scale=scale),
            bt_hv.infer_reference_point(pareto, scale=scale),
            VALUE_TOL,
        )
    updates = [
        points(r.randint(1, 4), m, kind, 9500 + seed) * r.uniform(0.8, 1.3) for _ in range(2)
    ]
    for name in CLASSES:
        if name == "NondominatedPartitioning" and n > {2: 1000, 3: 40, 4: 12, 5: 8}[m]:
            continue
        alpha = r.choice([0.0, 0.0, 1e-3, 1e-2]) if name == "NondominatedPartitioning" else 0.0
        run_decomposition(name, ref, finite, updates, alpha=alpha)
    if m == 2:
        batch = points(max(n, 1), 2, kind, 9700 + seed, batch=(r.randint(2, 6),))
        batch[0] = -1.0
        for name in CLASSES:
            run_decomposition(name, ref, batch, [batch[:, :2] * 1.1])
    w = torch.tensor([r.uniform(-1, 1) for _ in range(m)], dtype=DT)
    if finite.shape[0] > 1:
        fa = rs_scal.get_chebyshev_scalarization(w, finite, alpha=r.uniform(0, 0.2))
        fe = bt_scal.get_chebyshev_scalarization(w, finite, alpha=fa.alpha)
        X = torch.randn(9, m, generator=torch.Generator().manual_seed(seed), dtype=DT)
        CHECK.check("sweep: Chebyshev scalarization", fa(X), fe(X), VALUE_TOL)


def test_rel_err_helper_is_strict() -> None:
    # The error measure used above treats infinities exactly.
    inf = float("inf")
    assert rel_err(torch.tensor([inf, 1.0]), torch.tensor([inf, 1.0])) == 0.0
    assert rel_err(torch.tensor([inf, 1.0]), torch.tensor([-inf, 1.0])) == inf

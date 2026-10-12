"""Multi-objective test problems through the public API (without BoTorch):
shapes and array types, batching, negation, noise, constraints, Pareto
fronts, dtypes and BoTorch's errors."""

from __future__ import annotations

import inspect
import math
from typing import Any

import pytest

import botorch_rs.test_functions as tf
from botorch_rs.exceptions import BotorchRsError, InputDataError, UnsupportedError
from botorch_rs.test_functions import base, multi_objective, multi_objective_multi_fidelity
from helpers import HAS_NUMPY, HAS_TORCH, as_kind, check_kind, flatten, shape

# (class, constructor arguments) with BoTorch's names.
PROBLEMS: list[tuple[type, dict[str, Any]]] = [
    (tf.BraninCurrin, {}),
    (multi_objective.DH1, {"dim": 3}),
    (multi_objective.DH2, {"dim": 2}),
    (multi_objective.DH3, {"dim": 4}),
    (multi_objective.DH4, {"dim": 3}),
    (tf.DTLZ1, {"dim": 4, "num_objectives": 3}),
    (tf.DTLZ2, {"dim": 3}),
    (tf.DTLZ3, {"dim": 5, "num_objectives": 3}),
    (tf.DTLZ4, {"dim": 4}),
    (tf.DTLZ5, {"dim": 5, "num_objectives": 3}),
    (tf.DTLZ7, {"dim": 6, "num_objectives": 4}),
    (tf.GMM, {"num_objectives": 3}),
    (tf.Penicillin, {}),
    (tf.ToyRobust, {}),
    (tf.VehicleSafety, {}),
    (tf.ZDT1, {"dim": 3}),
    (tf.ZDT2, {"dim": 2}),
    (tf.ZDT3, {"dim": 4}),
    (tf.CarSideImpact, {}),
    (tf.BNH, {}),
    (tf.CONSTR, {}),
    (tf.ConstrainedBraninCurrin, {}),
    (tf.C2DTLZ2, {"dim": 4, "num_objectives": 3}),
    (tf.DiscBrake, {}),
    (tf.MW7, {"dim": 3}),
    (tf.OSY, {}),
    (tf.SRN, {}),
    (tf.WeldedBeam, {}),
    (tf.MOMFBraninCurrin, {}),
    (tf.MOMFPark, {}),
]
IDS = [f"{c.__name__}-{i}" for i, (c, _) in enumerate(PROBLEMS)]


def points(problem: Any, n: int, seed: int = 0) -> list[list[float]]:
    """Deterministic points inside the bounds (nested lists)."""
    lo, hi = flatten(problem.bounds)[: problem.dim], flatten(problem.bounds)[problem.dim :]
    out = []
    for i in range(n):
        out.append(
            [
                lo[k] + (hi[k] - lo[k]) * ((0.37 * (i + 1) + 0.61 * (k + 1) * (seed + 1)) % 1.0)
                for k in range(problem.dim)
            ]
        )
    return out


@pytest.mark.parametrize("cls,kwargs", PROBLEMS, ids=IDS)
def test_shapes_and_array_types(cls: type, kwargs: dict, kind: str) -> None:
    p = cls(**kwargs)
    m = p.num_objectives
    X = points(p, 6)
    y = p.evaluate_true(as_kind(X, kind))
    check_kind(y, kind)
    assert shape(y) == (6, m)
    assert all(math.isfinite(v) for v in flatten(y))
    # Batched and one-dimensional inputs.
    Xb = as_kind([X[:3], X[3:]], kind)
    yb = p.evaluate_true(Xb)
    assert shape(yb) == (2, 3, m)
    assert flatten(yb) == flatten(y)
    assert shape(p(as_kind(X[0], kind), noise=False)) == (m,)
    if isinstance(p, base.ConstrainedBaseTestProblem):
        c = p.evaluate_slack_true(Xb)
        check_kind(c, kind)
        assert shape(c) == (2, 3, p.num_constraints)
        feasible = p.is_feasible(Xb, noise=False)
        assert shape(feasible) == (2, 3)
        expected = [all(v >= 0 for v in row) for row in _rows(flatten(c), p.num_constraints)]
        assert [bool(v) for v in flatten(feasible)] == expected
        assert p.worst_feasible_value == 0.0


def _rows(values: list[float], width: int) -> list[list[float]]:
    return [values[i : i + width] for i in range(0, len(values), width)]


@pytest.mark.parametrize("cls,kwargs", PROBLEMS, ids=IDS)
def test_negate_and_reference_point(cls: type, kwargs: dict) -> None:
    p = cls(**kwargs)
    q = cls(negate=True, **kwargs)
    X = points(p, 4, seed=1)
    assert flatten(q(X, noise=False)) == [-v for v in flatten(p.evaluate_true(X))]
    assert flatten(q.ref_point) == [-v for v in p._ref_point]
    assert flatten(p.ref_point) == p._ref_point
    if isinstance(p, base.ConstrainedBaseTestProblem):
        assert flatten(q.evaluate_slack(X, noise=False)) == flatten(p.evaluate_slack_true(X))
    try:
        hv = p.max_hv
    except NotImplementedError:
        assert p._max_hv is None
        with pytest.raises(NotImplementedError):
            _ = q.max_hv
    else:
        assert q.max_hv == hv == p._max_hv
    with pytest.raises(UnsupportedError, match="is not a valid property"):
        _ = p.is_minimization_problem
    corrected = {"BraninCurrin", "ConstrainedBraninCurrin", "DH4", "DTLZ1", "DTLZ4", "ZDT3"}
    corrected |= {"C2DTLZ2", "SRN", "WeldedBeam", "GMM"}
    assert (p.deviation is not None) == (cls.__name__ in corrected)
    assert p.continuous_inds == list(range(p.dim))
    assert p.discrete_inds == [] and p.categorical_inds == []
    assert len(p._bounds) == p.dim


def test_negate_set_later_keeps_the_reference_point() -> None:
    p = tf.DTLZ2(dim=3)
    p.negate = True  # BoTorch's ref_point buffer is created at construction
    assert flatten(p.ref_point) == [1.1, 1.1]
    X = [[0.2, 0.5, 0.5]]
    assert flatten(p(X, noise=False)) == [-v for v in flatten(p.evaluate_true(X))]


def _seed_global_generators(seed: int) -> None:
    if HAS_NUMPY:
        import numpy as np

        np.random.seed(seed)
    if HAS_TORCH:
        import torch

        torch.manual_seed(seed)


def test_noise(kind: str) -> None:
    _seed_global_generators(0)  # NumPy and torch inputs use the caller's generator
    p = tf.DTLZ2(dim=4, num_objectives=3, noise_std=[0.0, 0.5, 0.0], seed=3)
    X = as_kind([[0.3, 0.6, 0.5, 0.5]] * 400, kind)
    clean = _rows(flatten(p(X, noise=False)), 3)
    noisy = _rows(flatten(p(X)), 3)
    assert all(a[0] == b[0] and a[2] == b[2] for a, b in zip(clean, noisy, strict=True))
    diffs = [b[1] - a[1] for a, b in zip(clean, noisy, strict=True)]
    mean = sum(diffs) / len(diffs)
    std = math.sqrt(sum((d - mean) ** 2 for d in diffs) / len(diffs))
    assert abs(mean) < 0.1 and 0.4 < std < 0.6
    if kind == "list":  # the seeded Rust generator
        q = tf.DTLZ2(dim=4, num_objectives=3, noise_std=[0.0, 0.5, 0.0], seed=3)
        assert flatten(q(X)) == flatten(noisy)
    # Noise is added before negation; the slacks get their own noise.
    c = tf.ConstrainedBraninCurrin(noise_std=0.0, constraint_noise_std=2.0, negate=True)
    Xc = as_kind([[0.5, 0.5]] * 50, kind)
    assert flatten(c(Xc)) == flatten(c(Xc, noise=False))
    slack = flatten(c.evaluate_slack(Xc))
    assert len(set(slack)) > 1
    assert flatten(c.evaluate_slack(Xc, noise=False)) == flatten(c.evaluate_slack_true(Xc))


@pytest.mark.skipif(not HAS_TORCH, reason="torch is not installed")
def test_torch_noise_follows_the_torch_seed_and_keeps_the_dtype() -> None:
    import torch

    p = tf.VehicleSafety(noise_std=[1.0, 0.1, 0.01], negate=True)
    X = 1 + 2 * torch.rand(5, 5, dtype=torch.float32)
    torch.manual_seed(1)
    a = p(X)
    torch.manual_seed(1)
    b = p(X)
    assert torch.equal(a, b) and a.dtype == torch.float32
    c = tf.MW7(dim=4, constraint_noise_std=[0.1, 0.2])
    Xc = torch.rand(3, 2, 4, dtype=torch.float64)
    torch.manual_seed(2)
    s1 = c.evaluate_slack(Xc)
    torch.manual_seed(2)
    f1 = c.is_feasible(Xc)
    assert torch.equal(f1, (s1 >= 0).all(-1))
    assert f1.dtype == torch.bool and f1.shape == (3, 2)


@pytest.mark.skipif(not HAS_NUMPY, reason="NumPy is not installed")
def test_numpy_noise_keeps_the_dtype() -> None:
    import numpy as np

    p = tf.ZDT1(dim=3, noise_std=0.1)
    X = np.full((4, 3), 0.5, dtype=np.float32)
    y = p(X)
    assert y.dtype == np.float32 and y.shape == (4, 2)


def test_constraint_noise_can_be_set_later() -> None:
    p = tf.OSY(seed=0)
    X = [[5.0, 1.0, 2.0, 0.0, 5.0, 1.0]] * 20
    assert flatten(p.evaluate_slack(X)) == flatten(p.evaluate_slack_true(X))
    p.constraint_noise_std = 0.5  # an attribute in BoTorch
    assert flatten(p.evaluate_slack(X)) != flatten(p.evaluate_slack_true(X))
    p.noise_std = [0.0, 1.0]
    y = _rows(flatten(p(X)), 2)
    assert len({row[0] for row in y}) == 1 and len({row[1] for row in y}) > 1


def test_errors_like_botorch() -> None:
    with pytest.raises(ValueError, match=r"dim must be >= 2, but got dim=1!"):
        multi_objective.DH1(dim=1)
    with pytest.raises(ValueError, match=r"dim must be >= 3, but got dim=2!"):
        multi_objective.DH4(dim=2)
    with pytest.raises(ValueError, match=r"dim must be > num_objectives, but got 3 and 3\."):
        tf.DTLZ2(dim=3, num_objectives=3)
    with pytest.raises(ValueError, match=r"dim must be > num_objectives, but got -1 and 2\."):
        tf.C2DTLZ2(dim=-1)
    with pytest.raises(NotImplementedError, match="ZDT2 currently only supports 2 objectives."):
        tf.ZDT2(dim=4, num_objectives=3)
    with pytest.raises(ValueError, match="dim must be >= num_objectives, but got 1 and 2"):
        tf.ZDT1(dim=1)
    with pytest.raises(UnsupportedError, match="GMM only currently supports 2 to 4 objectives."):
        tf.GMM(num_objectives=5)
    with pytest.raises(ValueError, match="dim must be greater than or equal to 2."):
        tf.MW7(dim=1)
    with pytest.raises(InputDataError, match=r"length of noise_std \(3\) must match the number"):
        tf.BraninCurrin(noise_std=[0.1, 0.2, 0.3])
    with pytest.raises(InputDataError):
        tf.BraninCurrin(noise_std=[0.1])  # lists must have one entry per objective
    assert tf.BraninCurrin(noise_std=(0.1,)).noise_std == (0.1,)  # broadcast like torch
    with pytest.raises(TypeError, match="Can't instantiate abstract class DTLZ"):
        multi_objective.DTLZ(dim=3)
    with pytest.raises(TypeError):
        multi_objective.ZDT(dim=3)
    with pytest.raises(TypeError):
        tf.BraninCurrin(dim=2)  # like BoTorch, no `dim` argument
    p = tf.DTLZ1(dim=3)
    with pytest.raises(ValueError, match="within the bounds"):
        p.evaluate_true([[0.5, 0.5, 1.5]])
    with pytest.raises(ValueError, match=r"shape `\(batch_shape\) x n x d`"):
        p.evaluate_true([[0.5, 0.5]])
    with pytest.raises(BotorchRsError):
        p.evaluate_true([[float("nan"), 0.5, 0.5]])
    with pytest.raises(NotImplementedError, match="does not specify maximal hypervolume"):
        _ = tf.DTLZ5(dim=3).max_hv
    with pytest.raises(NotImplementedError):
        tf.BNH().gen_pareto_front(3)
    with pytest.raises(NotImplementedError):  # BoTorch's C2DTLZ2 has none for M > 2 either
        _ = tf.C2DTLZ2(dim=4, num_objectives=3).max_hv
    assert not hasattr(tf.DTLZ2(dim=3), "evaluate_slack")


def _dominated(F: list[list[float]], maximize: bool) -> list[bool]:
    sign = -1.0 if maximize else 1.0
    G = [[sign * v for v in row] for row in F]
    return [
        any(
            all(b <= a for a, b in zip(G[i], G[j], strict=True))
            and any(b < a for a, b in zip(G[i], G[j], strict=True))
            for j in range(len(G))
            if j != i
        )
        for i in range(len(G))
    ]


@pytest.mark.parametrize("negate", [False, True])
def test_pareto_fronts(negate: bool) -> None:
    sign = -1.0 if negate else 1.0
    for p in [tf.ZDT1(dim=3, negate=negate), tf.ZDT2(dim=2, negate=negate)]:
        F = _rows(flatten(p.gen_pareto_front(9)), 2)
        assert len(F) == 9 and not any(_dominated(F, negate))
        f0 = [sign * r[0] for r in F]
        assert f0[0] == 0.0 and f0[-1] == 1.0
    F3 = _rows(flatten(tf.ZDT3(dim=2, negate=negate).gen_pareto_front(23)), 2)
    assert len(F3) == 23 and not any(_dominated(F3, negate))
    for cls, m in [(tf.DTLZ1, 3), (tf.DTLZ2, 3), (tf.DTLZ3, 2), (tf.DTLZ4, 4)]:
        p = cls(dim=m + 2, num_objectives=m, negate=negate)
        F = _rows(flatten(p.gen_pareto_front(64, seed=0)), m)
        assert F == _rows(flatten(p.gen_pareto_front(64, seed=0)), m)
        assert F != _rows(flatten(p.gen_pareto_front(64, seed=1)), m)
        assert not any(_dominated(F, negate))
        for row in F:
            if cls is tf.DTLZ1:
                assert abs(sign * sum(row) - 0.5) < 1e-14
            else:
                assert abs(math.sqrt(sum(v * v for v in row)) - 1.0) < 1e-14
    # C2DTLZ2: only feasible points; checked through the problem's own
    # constraint at a pre-image on the Pareto set (x_i = 0.5 for i >= M - 1).
    p = tf.C2DTLZ2(dim=4, num_objectives=2, negate=negate)
    F = _rows(flatten(p.gen_pareto_front(200, seed=0)), 2)
    assert not any(_dominated(F, negate))
    X = [[math.atan2(sign * f1, sign * f0) * 2 / math.pi, 0.5, 0.5, 0.5] for f0, f1 in F]
    assert all(flatten(p.is_feasible(X, noise=False)))
    for row, y in zip(F, _rows(flatten(p(X, noise=False)), 2), strict=True):
        assert max(abs(a - b) for a, b in zip(row, y, strict=True)) < 1e-12


def test_gmm_buffers_and_class_attributes() -> None:
    g = tf.GMM(num_objectives=4, dtype=None)
    assert shape(g.gmm_pos) == (4, 3, 2)
    assert shape(g.gmm_covar) == (4, 3, 2, 2)
    assert flatten(g.gmm_norm)[0] == 2 * math.pi * 0.2**2 * 0.5
    assert tf.DTLZ7(dim=6, num_objectives=4).k == 3
    assert multi_objective.DH2(dim=2).beta == 10.0 and multi_objective.DH2(dim=2).alpha == 1.0
    assert tf.DTLZ4._alpha == 100.0 and len(tf.ZDT3._parts) == 5
    assert tf.Penicillin.K_2 == 7 * 10 ** (-5)
    assert flatten(tf.ConstrainedBraninCurrin(dtype=None).con_bounds) == [-5.0, 0.0, 10.0, 15.0]
    toy = tf.ToyRobust()
    X = [[0.1], [0.5]]
    y = flatten(toy.evaluate_true(X))
    assert flatten(toy.f_1(X)) == [-y[0], -y[2]] and flatten(toy.f_2(X)) == [-y[1], -y[3]]
    mw7 = tf.MW7(dim=2)
    assert mw7.LA2(0.4, 4.0, 1.0, 16.0, 0.3) == pytest.approx(0.4 * math.sin(1.2) ** 16)
    pen = tf.Penicillin()
    Xp = points(pen, 3)
    assert flatten(pen.penicillin_vectorized(Xp)) == flatten(pen.evaluate_true(Xp))


def test_large_batches_run_in_parallel_with_identical_values() -> None:
    p = tf.Penicillin()
    X = points(p, 80)
    assert flatten(p.evaluate_true(X)) == [v for x in X for v in flatten(p.evaluate_true([x]))]
    d = multi_objective.DH3(dim=3)  # not re-exported, as in BoTorch
    Xd = points(d, 5000)
    assert flatten(d.evaluate_true(Xd))[:10] == flatten(d.evaluate_true(Xd[:5]))


@pytest.mark.skipif(not HAS_TORCH, reason="torch is not installed")
def test_dtype_and_to() -> None:
    import torch

    p = tf.DTLZ2(dim=3, dtype=torch.float32)
    assert p.bounds.dtype == torch.float32 and p.ref_point.dtype == torch.float32
    assert p.gen_pareto_front(4, seed=0).dtype == torch.float32
    assert p.to(dtype=torch.float64) is p
    assert p.ref_point.dtype == torch.float64
    assert p.to(torch.zeros(1, dtype=torch.float32)).bounds.dtype == torch.float32
    assert tf.Branin().to(dtype=torch.float32).bounds.dtype == torch.float32


def test_signatures_follow_botorch() -> None:
    """BoTorch's positional parameters, in BoTorch's order; botorch_rs only
    adds the keyword-only ``seed``."""
    expected = {
        "BraninCurrin": ["noise_std", "negate", "dtype"],
        "DH1": ["dim", "noise_std", "negate", "dtype"],
        "DTLZ2": ["dim", "num_objectives", "noise_std", "negate", "dtype"],
        "GMM": ["noise_std", "negate", "num_objectives", "dtype"],
        "ZDT1": ["dim", "num_objectives", "noise_std", "negate", "dtype"],
        "ConstrainedBraninCurrin": ["noise_std", "constraint_noise_std", "negate", "dtype"],
        "C2DTLZ2": ["dim", "num_objectives", "noise_std", "negate", "dtype"],
        "MW7": ["dim", "noise_std", "constraint_noise_std", "negate", "dtype"],
        "OSY": ["noise_std", "negate", "dtype"],
        "MOMFPark": ["noise_std", "negate", "dtype"],
    }
    modules = [multi_objective, multi_objective_multi_fidelity]
    for name, params in expected.items():
        cls = next(getattr(m, name) for m in modules if hasattr(m, name))
        sig = inspect.signature(cls.__init__)
        names = [p.name for p in sig.parameters.values() if p.name != "self"]
        assert names == [*params, "seed"], name
        assert sig.parameters["seed"].kind == inspect.Parameter.KEYWORD_ONLY

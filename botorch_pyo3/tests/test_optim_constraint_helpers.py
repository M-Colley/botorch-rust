"""Constraint conversions against current upstream and the released API."""

from __future__ import annotations

import math

import pytest

from botorch_rs.optim import initializers as ri
from botorch_rs.optim import parameter_constraints as rs
from helpers import KINDS, as_kind, to_list


@pytest.mark.parametrize("kind", KINDS)
def test_constraint_transforms(kind: str) -> None:
    coefficients = as_kind([1.0, -2.0], kind)
    intra = (as_kind([0, 2], kind), coefficients, 0.5)
    actual = ri.transform_intra_point_constraint(intra, d=3, q=2)
    assert [to_list(row[0]) for row in actual] == [[0, 2], [3, 5]]
    assert all(row[1] is coefficients and row[2] == 0.5 for row in actual)
    inter = (as_kind([[0, 1], [1, 2]], kind), coefficients, 0.5)
    transformed = ri.transform_inter_point_constraint(inter, d=3)
    assert to_list(transformed[0]) == [1, 5]
    assert transformed[1] is coefficients
    combined = ri.transform_constraints([intra, inter], q=2, d=3)
    assert [to_list(row[0]) for row in combined] == [[0, 2], [3, 5], [1, 5]]
    with pytest.raises(ValueError, match="problem dimension"):
        ri.transform_intra_point_constraint((as_kind([3], kind), coefficients, 0), d=3, q=2)
    with pytest.raises(ValueError, match="one-dimensional integer"):
        ri.transform_intra_point_constraint((as_kind([0.2], kind), coefficients, 0), d=3, q=2)


@pytest.mark.parametrize("kind", KINDS)
def test_nonlinear_feasibility(kind: str) -> None:
    X = as_kind([[[0.1, 0.4], [0.3, 0.4]], [[0.2, 0.2], [-0.5, 0.4]]], kind)
    assert to_list(rs.nonlinear_constraint_is_feasible(lambda x: sum(x) - 0.2, True, X)) == [
        True,
        False,
    ]
    assert to_list(
        rs.nonlinear_constraint_is_feasible(lambda x: sum(sum(row) for row in x) - 0.5, False, X)
    ) == [True, False]
    assert (
        to_list(
            rs.nonlinear_constraint_is_feasible(lambda x: sum(x), True, as_kind([[1.0, 2.0]], kind))
        )
        is True
    )
    assert to_list(rs.nonlinear_constraint_is_feasible(lambda x: math.nan, True, X)) == [
        False,
        False,
    ]


def test_dtype_constraint_tolerance() -> None:
    np = pytest.importorskip("numpy")
    for dtype, expected in [
        (float, 1e-8),
        (np.float64, 1e-8),
        (np.dtype("float32"), 1e-6),
        (np.float16, 1e-4),
    ]:
        assert rs.get_constraint_tolerance(dtype) == expected
    with pytest.raises(ValueError, match="Unsupported dtype"):
        rs.get_constraint_tolerance(np.int32)
    X = np.asarray([[[-5e-7]]], dtype=np.float32)
    constraint = [(np.asarray([0]), np.asarray([1.0]), 0.0)]
    assert rs.evaluate_feasibility(X, inequality_constraints=constraint).item()
    assert not rs.evaluate_feasibility(X, inequality_constraints=constraint, tolerance=1e-8).item()


@pytest.mark.parametrize("kind", KINDS)
def test_scipy_bounds(kind: str) -> None:
    np = pytest.importorskip("numpy")
    pytest.importorskip("scipy")
    X = as_kind([[[0.2, 0.3], [0.4, 0.5]]] * 2, kind)
    assert rs.make_scipy_bounds(X) is None
    bounds = rs.make_scipy_bounds(X, as_kind([0.0, 0.1], kind), 1.0)
    np.testing.assert_array_equal(bounds.lb, [0.0, 0.1] * 4)
    np.testing.assert_array_equal(bounds.ub, [1.0] * 8)
    assert bounds.keep_feasible.all()
    assert np.isneginf(rs.make_scipy_bounds(X, upper_bounds=1.0).lb).all()
    assert np.isposinf(rs.make_scipy_bounds(X, lower_bounds=0.0).ub).all()
    with pytest.raises(ValueError):
        rs.make_scipy_bounds(X, lower_bounds=[0.0, 0.1, 0.2])


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("shape", [(2, 3), (3, 2, 3)])
def test_sparse_scipy_linear_constraints(kind: str, shape: tuple[int, ...]) -> None:
    np = pytest.importorskip("numpy")
    torch = pytest.importorskip("torch")
    pytest.importorskip("scipy")
    bt = pytest.importorskip("botorch.optim.parameter_constraints")
    inequalities = [
        (as_kind([0, 2], kind), as_kind([1.5, -0.5], kind), 0.2),
        (as_kind([[0, 1], [1, 2]], kind), as_kind([2.0, -1.0], kind), -0.1),
    ]
    equalities = [(as_kind([0, 1], kind), as_kind([1.0, 1.0], kind), 0.8)]
    actual = rs.make_scipy_linear_constraints(shape, inequalities, equalities)

    def convert(constraints):
        return [
            (
                torch.tensor(to_list(idx), dtype=torch.long),
                torch.tensor(to_list(coef), dtype=torch.double),
                rhs,
            )
            for idx, coef, rhs in constraints
        ]

    expected = bt.make_scipy_linear_constraints(
        torch.Size(shape), convert(inequalities), convert(equalities)
    )
    assert len(actual) == 2
    if isinstance(expected[0], dict):  # Released 0.18.1: compare the same numeric rows.
        for block, constraint_type in zip(actual, ["ineq", "eq"], strict=True):
            subset = [entry for entry in expected if entry["type"] == constraint_type]
            zeros = np.zeros(math.prod(shape))
            np.testing.assert_array_equal(
                block.A.toarray(), np.asarray([entry["jac"](zeros) for entry in subset])
            )
            np.testing.assert_array_equal(block.lb, [-entry["fun"](zeros) for entry in subset])
    else:
        for a, e in zip(actual, expected, strict=True):
            np.testing.assert_array_equal(a.A.toarray(), e.A.toarray())
            np.testing.assert_array_equal(a.lb, e.lb)
            np.testing.assert_array_equal(a.ub, e.ub)
    assert rs.make_scipy_linear_constraints(shape) == []


def test_nonlinear_scipy_callbacks_and_cache() -> None:
    np = pytest.importorskip("numpy")
    torch = pytest.importorskip("torch")
    bt = pytest.importorskip("botorch.optim.parameter_constraints")
    shape = (2, 2, 2)
    calls = []

    def wrapper(x, f):
        calls.append(1)
        tensor = torch.from_numpy(x).reshape(shape).requires_grad_(True)
        loss = f(tensor)
        gradient = torch.autograd.grad(loss, tensor)[0].reshape(-1).numpy()
        return loss.item(), gradient

    constraints = [(lambda x: x.sum() - 0.2, True), (lambda x: x.sum() - 0.5, False)]
    X = torch.ones(shape, dtype=torch.double)
    actual = rs.make_scipy_nonlinear_inequality_constraints(constraints, wrapper, X, shape)
    expected = bt.make_scipy_nonlinear_inequality_constraints(constraints, wrapper, X, shape)
    assert len(actual) == len(expected) == 6
    values = X.numpy().reshape(-1).copy()
    for a, e in zip(actual, expected, strict=True):
        assert a["type"] == e["type"] == "ineq"
        assert a["fun"](values) == e["fun"](values)
        np.testing.assert_array_equal(a["jac"](values), e["jac"](values))
    before = len(calls)
    actual[0]["fun"](values)
    actual[0]["jac"](values)
    assert len(calls) == before
    values[0] = 0.5
    actual[0]["fun"](values)
    assert len(calls) == before + 1
    with pytest.raises(ValueError, match="must satisfy"):
        rs.make_scipy_nonlinear_inequality_constraints(constraints, wrapper, -X, shape)
    assert (
        len(
            rs.make_scipy_nonlinear_inequality_constraints(
                constraints, wrapper, -X, shape, validate_feasibility=False
            )
        )
        == 6
    )


def test_live_dtype_and_bounds_parity() -> None:
    np = pytest.importorskip("numpy")
    torch = pytest.importorskip("torch")
    bt = pytest.importorskip("botorch.optim.parameter_constraints")
    for dtype in (torch.float16, torch.float32, torch.float64):
        assert rs.get_constraint_tolerance(dtype) == bt.get_constraint_tolerance(dtype)
        X = torch.rand(2, 3, dtype=dtype)
        for lower, upper in ((None, 0.8), (0.1, None), (0.1, 0.9)):
            actual, expected = (
                rs.make_scipy_bounds(X, lower, upper),
                bt.make_scipy_bounds(X, lower, upper),
            )
            np.testing.assert_array_equal(actual.lb, expected.lb)
            np.testing.assert_array_equal(actual.ub, expected.ub)

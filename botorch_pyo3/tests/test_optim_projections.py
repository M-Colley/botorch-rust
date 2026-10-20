"""Native equality and SLSQP projections against official BoTorch."""

from __future__ import annotations

import inspect

import pytest

from botorch_rs.optim import parameter_constraints as rs
from helpers import KINDS, as_kind, to_list


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("batch", [(), (2,), (2, 3)])
def test_equality_projection_values(kind: str, batch: tuple[int, ...]) -> None:
    torch = pytest.importorskip("torch")
    bt = pytest.importorskip("botorch.optim.parameter_constraints")
    X = torch.arange(12, dtype=torch.double).reshape(2, 2, 3) / 10
    if not batch:
        X = X[0]
    elif len(batch) == 2:
        X = X.unsqueeze(1).expand(2, 3, 2, 3)
    constraints = [
        (torch.tensor([0, 1]), torch.tensor([1.0, 2.0], dtype=torch.double), 0.7),
        (torch.tensor([1, 2]), torch.tensor([1.0, -1.0], dtype=torch.double), -0.1),
    ]
    converted = [
        (as_kind(i.tolist(), kind), as_kind(c.tolist(), kind), r) for i, c, r in constraints
    ]
    actual = rs.project_to_equality_constraints(as_kind(X.tolist(), kind), converted)
    expected = bt.project_to_equality_constraints(X, constraints)
    torch.testing.assert_close(torch.tensor(to_list(actual), dtype=torch.double), expected)


@pytest.mark.parametrize("dtype_name", ["float32", "float64"])
@pytest.mark.parametrize("scalar_coefficient", [False, True])
def test_equality_projection_all_derivatives(dtype_name: str, scalar_coefficient: bool) -> None:
    torch = pytest.importorskip("torch")
    bt = pytest.importorskip("botorch.optim.parameter_constraints")
    dtype = getattr(torch, dtype_name)
    seed = torch.arange(18, dtype=dtype).reshape(3, 2, 3) / 10
    adjoint = seed.square() + 0.2
    outputs, gradients = [], []
    for implementation in (rs, bt):
        X = seed.clone().requires_grad_()
        coefficient = torch.tensor(1.3 if scalar_coefficient else [1.3, -0.7], dtype=dtype)
        coefficient.requires_grad_()
        rhs = torch.tensor(0.8, dtype=dtype, requires_grad=True)
        projected = implementation.project_to_equality_constraints(
            X, [(torch.tensor([0, 2]), coefficient, rhs)]
        )
        gradients.append(torch.autograd.grad((projected * adjoint).sum(), (X, coefficient, rhs)))
        outputs.append(projected)
    tolerance = 2e-5 if dtype_name == "float32" else 1e-11
    torch.testing.assert_close(outputs[0], outputs[1], atol=tolerance, rtol=tolerance)
    for actual, expected in zip(*gradients, strict=True):
        torch.testing.assert_close(actual, expected, atol=tolerance, rtol=tolerance)


def test_equality_noop_and_invalid_constraints() -> None:
    torch = pytest.importorskip("torch")
    X = torch.tensor([[0.2, 0.4]], dtype=torch.double)
    assert rs.project_to_equality_constraints(X, []) is X
    assert (
        rs.project_to_equality_constraints(X, [(torch.tensor([[0, 0]]), torch.ones(1), 0.1)]) is X
    )
    with pytest.raises((RuntimeError, ValueError), match="[Ss]ingular"):
        rs.project_to_equality_constraints(
            X, [(torch.tensor([0]), torch.ones(1), 0.1), (torch.tensor([0]), torch.ones(1), 0.2)]
        )
    with pytest.raises(ValueError, match="nonnegative integers"):
        rs.project_to_equality_constraints(X, [([-1], [1.0], 0.0)])


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("batched", [False, True])
def test_slsqp_linear_projection(kind: str, batched: bool) -> None:
    torch = pytest.importorskip("torch")
    bt = pytest.importorskip("botorch.optim.parameter_constraints")
    X = torch.tensor([[[0.2, 0.2], [0.9, 0.1]], [[0.1, 0.8], [0.8, 0.8]]], dtype=torch.double)
    if not batched:
        X = X[0]
    bounds = torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.double)
    inequality = [(torch.tensor([0, 1]), torch.ones(2, dtype=torch.double), 1.2)]
    equality = [(torch.tensor([[0, 0], [1, 0]]), torch.ones(2, dtype=torch.double), 1.3)]

    def converted(constraints):
        return [
            (as_kind(i.tolist(), kind), as_kind(c.tolist(), kind), r) for i, c, r in constraints
        ]

    actual = rs.project_to_feasible_space_via_slsqp(
        as_kind(X.tolist(), kind),
        as_kind(bounds.tolist(), kind),
        converted(inequality),
        converted(equality),
    )
    expected = bt.project_to_feasible_space_via_slsqp(X, bounds, inequality, equality)
    torch.testing.assert_close(
        torch.tensor(to_list(actual), dtype=torch.double), expected, atol=2e-7, rtol=2e-7
    )


@pytest.mark.parametrize("dtype_name", ["float32", "float64"])
def test_slsqp_fixed_values_and_nonlinear(dtype_name: str) -> None:
    torch = pytest.importorskip("torch")
    bt = pytest.importorskip("botorch.optim.parameter_constraints")
    dtype = getattr(torch, dtype_name)
    X = torch.tensor([[[0.2, 0.2]], [[0.8, 0.1]]], dtype=dtype)
    bounds = torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=dtype)
    fixed = {0: torch.tensor([0.3, 0.7], dtype=dtype)}
    nonlinear = [(lambda x: x.sum() - 1.2, True), (lambda x: 1.5 - x.square().sum(), False)]
    actual = rs.project_to_feasible_space_via_slsqp(
        X, bounds, nonlinear_inequality_constraints=nonlinear, fixed_features=fixed
    )
    if (
        "nonlinear_inequality_constraints"
        in inspect.signature(bt.project_to_feasible_space_via_slsqp).parameters
    ):
        expected = bt.project_to_feasible_space_via_slsqp(
            X, bounds, nonlinear_inequality_constraints=nonlinear, fixed_features=fixed
        )
    else:
        # Released 0.18.1 lacks nonlinear projection. Here the first nonlinear
        # constraint is exactly x0+x1>=1.2, while the second is inactive.
        # Compare its released linear projection with the identical feasible set.
        expected = bt.project_to_feasible_space_via_slsqp(
            X,
            bounds,
            inequality_constraints=[(torch.tensor([0, 1]), torch.ones(2, dtype=dtype), 1.2)],
            fixed_features=fixed,
        )
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-6)
    torch.testing.assert_close(actual[:, 0], torch.tensor([[0.3, 0.9], [0.7, 0.5]], dtype=dtype))
    assert not actual.requires_grad


def test_slsqp_no_constraints_empty_constraints_and_failure() -> None:
    torch = pytest.importorskip("torch")
    X = torch.tensor([[0.2, 0.4]], dtype=torch.double, requires_grad=True)
    bounds = torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.double)
    assert rs.project_to_feasible_space_via_slsqp(X, bounds, fixed_features={0: 0.8}) is X
    actual = rs.project_to_feasible_space_via_slsqp(
        X, bounds, inequality_constraints=[], fixed_features={0: 0.8}
    )
    torch.testing.assert_close(actual, torch.tensor([[0.8, 0.4]], dtype=torch.double))
    with pytest.raises(Exception, match="Optimization failed"):
        rs.project_to_feasible_space_via_slsqp(
            X, bounds, inequality_constraints=[([0], [1.0], 2.0)]
        )

"""MC objectives, outcome constraints, parameter constraints, polytope
sampling and constrained acquisition optimization: array types, shapes,
feasibility, finite-difference gradients through Python callables, error
handling (no BoTorch needed; callables need torch)."""

from __future__ import annotations

import math
import sys
import warnings
from typing import Any

import pytest

from botorch_rs.acquisition import (
    ConstrainedMCObjective,
    GenericMCObjective,
    IdentityMCObjective,
    LinearMCObjective,
    MCAcquisitionObjective,
    prune_inferior_points,
    qExpectedImprovement,
    qLogExpectedImprovement,
    qLogNoisyExpectedImprovement,
    qLogProbabilityOfFeasibility,
    qProbabilityOfImprovement,
    qSimpleRegret,
    qUpperConfidenceBound,
)
from botorch_rs.acquisition.objective import ScalarizedPosteriorTransform
from botorch_rs.exceptions import (
    BotorchError,
    BotorchRuntimeError,
    InfeasibilityError,
    NumericsWarning,
    UnsupportedError,
)
from botorch_rs.models import ModelListGP, SingleTaskGP
from botorch_rs.optim import (
    evaluate_feasibility,
    gen_batch_initial_conditions,
    optimize_acqf,
    sample_q_batches_from_polytope,
    transform_constraints,
)
from botorch_rs.sampling import SobolQMCNormalSampler
from botorch_rs.utils.sampling import (
    HitAndRunPolytopeSampler,
    find_interior_point,
    get_polytope_samples,
    sample_polytope,
    sparse_to_dense_constraints,
)
from helpers import HAS_TORCH, as_kind, check_kind, flatten, make_data, shape, to_list

requires_torch = pytest.mark.skipif(not HAS_TORCH, reason="callables operate on torch tensors")

BOUNDS = [[0.0, 0.0], [1.0, 1.0]]


def multi_model(m: int = 2, kind: str = "list", seed: int = 0) -> SingleTaskGP:
    X, _ = make_data(n=12, d=2, seed=seed)
    Y = [[math.sin((3 + t) * x[0]) + (1 + t) * x[1] ** 2 + 0.3 * t for t in range(m)] for x in X]
    model = SingleTaskGP(as_kind(X, kind), as_kind(Y, kind))
    model.initialize(
        **{
            "covar_module.lengthscale": [[[0.4, 0.6]]] * m if m > 1 else [[0.4, 0.6]],
            "likelihood.noise": [[0.05]] * m if m > 1 else [0.05],
        }
    )
    return model


def train_x(seed: int = 0) -> Any:
    """The training inputs of ``multi_model`` as a float64 tensor."""
    import torch

    return torch.tensor(make_data(n=12, d=2, seed=seed)[0], dtype=torch.float64)


def sampler(n: int = 64, seed: int = 0) -> SobolQMCNormalSampler:
    return SobolQMCNormalSampler(sample_shape=[n], seed=seed)


def candidates(kind: str, b: int = 2, q: int = 2, seed: int = 3) -> Any:
    import random

    r = random.Random(seed)
    return as_kind([[[r.random(), r.random()] for _ in range(q)] for _ in range(b)], kind)


# ---------------------------------------------------------------------------
# Objectives without callables (any array type)
# ---------------------------------------------------------------------------


def test_linear_objective_matches_manual_computation(kind: str) -> None:
    """qSimpleRegret with a LinearMCObjective is the average over samples of
    the maximum over the q points of ``samples @ weights``, with the samples
    of ``sampler(posterior)`` (BoTorch's ``S x q x m`` base samples)."""
    model = multi_model(m=2, kind=kind)
    w = as_kind([0.7, -0.3], kind)
    s = sampler(32, seed=4)
    acqf = qSimpleRegret(model, sampler=s, objective=LinearMCObjective(w))
    X = candidates(kind, b=1, q=3)
    value = flatten(acqf(X))[0]
    samples = to_list(s(model.posterior(to_list(X)[0])))  # 32 x 3 x 2
    manual = sum(max(0.7 * y[0] - 0.3 * y[1] for y in sample) for sample in samples) / 32
    assert value == pytest.approx(manual, rel=1e-12)
    check_kind(acqf(X), kind)


def test_objective_and_transform_combinations(kind: str) -> None:
    model = multi_model(m=2, kind=kind)
    X = candidates(kind)
    w = as_kind([0.5, 0.5], kind)
    with pytest.raises(UnsupportedError, match="Must specify an objective"):
        qSimpleRegret(model)
    # A scalarizing transform alone, and a linear objective on the transformed
    # (single-output) samples.
    t = ScalarizedPosteriorTransform(w)
    a = qSimpleRegret(model, sampler=sampler(), posterior_transform=t)
    b = qSimpleRegret(
        model, sampler=sampler(), posterior_transform=t, objective=LinearMCObjective([2.0])
    )
    assert flatten(b(X)) == pytest.approx([2 * v for v in flatten(a(X))], rel=1e-12)
    with pytest.raises(BotorchRuntimeError, match="Output shape of samples"):
        qSimpleRegret(model, sampler=sampler(), objective=LinearMCObjective([1.0, 2.0, 3.0]))(X)
    with pytest.raises(BotorchRuntimeError):  # identity objective needs one output
        qSimpleRegret(model, sampler=sampler(), objective=IdentityMCObjective())(X)
    with pytest.raises(ValueError, match="one-dimensional"):
        LinearMCObjective([[1.0, 2.0]])
    with pytest.raises(UnsupportedError, match="MCAcquisitionObjective"):
        qSimpleRegret(model, objective=lambda Y, X=None: Y[..., 0])


def test_model_list_with_linear_objective(kind: str) -> None:
    X, _ = make_data(n=10, d=2, seed=2)
    Y1 = [[math.sin(4 * x[0])] for x in X]
    Y2 = [[x[1] ** 2] for x in X]
    lm = ModelListGP(
        SingleTaskGP(as_kind(X, kind), as_kind(Y1, kind)),
        SingleTaskGP(as_kind(X, kind), as_kind(Y2, kind)),
    )
    acqf = qUpperConfidenceBound(
        lm, beta=0.5, sampler=sampler(), objective=LinearMCObjective([1.0, -1.0])
    )
    values = acqf(candidates(kind))
    check_kind(values, kind)
    assert shape(values) == (2,) and all(math.isfinite(v) for v in flatten(values))


# ---------------------------------------------------------------------------
# Callables (torch)
# ---------------------------------------------------------------------------


def _fd_check(acqf: Any, X: Any, h: float = 1e-6, rtol: float = 1e-5) -> None:
    """Analytic gradients of the botorch_rs computation against central
    finite differences of its values."""
    import torch

    values, grads = acqf.value_and_grad(X)
    fd = torch.zeros_like(X)
    for idx in torch.cartesian_prod(*[torch.arange(s) for s in X.shape]):
        i = tuple(idx.tolist())
        Xp, Xm = X.clone(), X.clone()
        Xp[i] += h
        Xm[i] -= h
        fd[i] = (acqf(Xp).sum() - acqf(Xm).sum()) / (2 * h)
    scale = max(fd.abs().max().item(), 1e-8)
    assert ((grads - fd).abs() <= rtol * scale + 1e-9).all(), (grads, fd)
    # The autograd path gives the same gradient.
    Xg = X.clone().requires_grad_(True)
    (g,) = torch.autograd.grad(acqf(Xg).sum(), Xg)
    assert torch.allclose(g, grads, rtol=1e-12, atol=1e-14)
    assert torch.equal(acqf(X), values)


@requires_torch
def test_gradients_through_callables() -> None:
    import torch

    model = multi_model(m=2, kind="torch")
    X = candidates("torch", b=2, q=2, seed=8)

    def generic(Y: Any, X: Any = None) -> Any:
        out = Y[..., 0] + 0.3 * torch.sin(Y[..., 1])
        return out if X is None else out - 0.2 * (X**2).sum(-1)

    def c1(Y: Any) -> Any:
        return Y[..., 1] - 2.0

    def c2(Y: Any) -> Any:
        return (Y[..., 0] - 0.5) ** 2 - 1.0

    def prob(Y: Any) -> Any:
        return torch.sigmoid(2.0 * (2.2 - Y[..., 1]))

    eta = torch.tensor([0.1, 0.3], dtype=torch.float64)
    acqfs = [
        qLogExpectedImprovement(
            model,
            best_f=0.5,
            sampler=sampler(),
            objective=GenericMCObjective(generic),
            constraints=[c1, c2],
            eta=eta,
        ),
        qLogExpectedImprovement(
            model,
            best_f=0.5,
            sampler=sampler(),
            objective=GenericMCObjective(generic),
            constraints=[c1, prob],
            eta=eta,
            fat=[False, None],
        ),
        qProbabilityOfImprovement(
            model,
            best_f=0.5,
            sampler=sampler(),
            objective=LinearMCObjective([1.0, 0.2]),
            constraints=[c1],
            eta=0.2,
            tau=0.2,
        ),
        qSimpleRegret(
            model,
            sampler=sampler(),
            objective=ConstrainedMCObjective(generic, [c1, c2], infeasible_cost=3.0, eta=eta),
        ),
        qLogNoisyExpectedImprovement(
            model,
            X_baseline=train_x(),
            sampler=sampler(),
            objective=GenericMCObjective(generic),
            constraints=[c1],
            eta=0.2,
            prune_baseline=False,
        ),
        qLogProbabilityOfFeasibility(model, constraints=[c1, c2], sampler=sampler(), eta=eta),
    ]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", NumericsWarning)
        acqfs.append(
            qExpectedImprovement(
                model,
                best_f=0.0,
                sampler=sampler(),
                objective=GenericMCObjective(generic),
                constraints=[c2],
                eta=0.5,
            )
        )
    for acqf in acqfs:
        _fd_check(acqf, X)


@requires_torch
def test_objective_subclass_and_direct_calls() -> None:
    import torch

    class Weighted(MCAcquisitionObjective):
        def forward(self, samples: Any, X: Any = None) -> Any:
            return samples[..., 0] - 0.5 * samples[..., 1]

    model = multi_model(m=2, kind="torch")
    X = candidates("torch")
    a = qSimpleRegret(model, sampler=sampler(), objective=Weighted())
    b = qSimpleRegret(
        model,
        sampler=sampler(),
        objective=GenericMCObjective(lambda Y, X: Y[..., 0] - 0.5 * Y[..., 1]),
    )
    c = qSimpleRegret(model, sampler=sampler(), objective=LinearMCObjective([1.0, -0.5]))
    assert torch.allclose(a(X), b(X), rtol=0, atol=0)
    assert torch.allclose(a(X), c(X), rtol=1e-14)
    Y = torch.randn(4, 2, 3, 2, dtype=torch.float64)
    assert torch.equal(Weighted()(Y), Y[..., 0] - 0.5 * Y[..., 1])
    assert torch.allclose(LinearMCObjective([1.0, -0.5])(Y), Weighted()(Y))
    assert torch.equal(IdentityMCObjective()(Y[..., :1]), Y[..., 0])
    con = ConstrainedMCObjective(lambda Y, X=None: Y[..., 0], [lambda Y: Y[..., 1]], 2.0, 0.1)
    expected = (Y[..., 0] + 2.0).clamp_min(0) * torch.sigmoid(-Y[..., 1] / 0.1) - 2.0
    assert torch.allclose(con(Y), expected, rtol=1e-12)
    with pytest.raises(RuntimeError, match="q-batch shape"):
        GenericMCObjective(lambda Y, X: Y[..., 0, :])(Y, X=torch.zeros(2, 3, 2))


@requires_torch
def test_callable_errors_propagate() -> None:
    import torch

    model = multi_model(m=2, kind="torch")
    X = candidates("torch")

    def boom(Y: Any) -> Any:
        raise KeyError("boom")

    acqf = qLogExpectedImprovement(
        model,
        best_f=0.0,
        sampler=sampler(),
        objective=LinearMCObjective([1.0, 0.0]),
        constraints=[boom],
    )
    with pytest.raises(KeyError, match="boom"):
        acqf(X)
    with pytest.raises(KeyError, match="boom"):
        acqf.value_and_grad(X)
    with pytest.raises(KeyError, match="boom"):
        optimize_acqf(
            acqf, torch.tensor(BOUNDS, dtype=torch.float64), q=2, num_restarts=2, raw_samples=8
        )
    bad = qSimpleRegret(
        model,
        sampler=sampler(),
        objective=GenericMCObjective(lambda Y, X: Y.sum(-1).sum(-1, keepdim=True)),
    )
    with pytest.raises(RuntimeError, match="q-batch shape"):
        bad(X)
    # After an error, evaluations work again.
    ok = qSimpleRegret(model, sampler=sampler(), objective=LinearMCObjective([1.0, 0.0]))
    assert torch.isfinite(ok(X)).all()


def test_torch_is_required_for_callables(monkeypatch: pytest.MonkeyPatch) -> None:
    model = multi_model(m=2)
    monkeypatch.setitem(sys.modules, "torch", None)
    with pytest.raises(UnsupportedError, match="require torch"):
        qSimpleRegret(model, objective=GenericMCObjective(lambda Y, X: Y[..., 0]))
    with pytest.raises(UnsupportedError, match="require torch"):
        qLogExpectedImprovement(
            model,
            best_f=0.0,
            objective=LinearMCObjective([1.0, 0.0]),
            constraints=[lambda Y: Y[..., 1]],
        )
    acqf = qSimpleRegret(model, objective=LinearMCObjective([1.0, 0.0]))
    with pytest.raises(UnsupportedError, match="require torch"):
        optimize_acqf(
            acqf,
            BOUNDS,
            q=1,
            num_restarts=1,
            nonlinear_inequality_constraints=[(lambda x: x.sum(), True)],
            batch_initial_conditions=[[[0.5, 0.5]]],
        )


def test_argument_validation() -> None:
    model = multi_model(m=2)
    lin = LinearMCObjective([1.0, 0.0])
    con = ConstrainedMCObjective(lambda Y, X=None: Y[..., 0], [lambda Y: Y[..., 1]])
    cons = [lambda Y: Y[..., 1], lambda Y: Y[..., 0]]
    with pytest.raises(ValueError, match="ConstrainedMCObjective as well as constraints"):
        qExpectedImprovement(model, best_f=0.0, objective=con, constraints=cons)
    with pytest.raises(BotorchError, match="Log-Improvement should not be used"):
        qLogExpectedImprovement(model, best_f=0.0, objective=con)
    with pytest.raises(BotorchError, match="Log-Improvement should not be used"):
        qLogNoisyExpectedImprovement(model, X_baseline=make_data(n=12, d=2)[0], objective=con)
    if HAS_TORCH:
        with pytest.raises(ValueError, match="number of provided etas"):
            qLogExpectedImprovement(model, best_f=0.0, objective=lin, constraints=cons, eta=[0.1])
        with pytest.raises(ValueError, match="eta must be positive"):
            qLogExpectedImprovement(model, best_f=0.0, objective=lin, constraints=cons, eta=0.0)
        with pytest.raises(ValueError, match="number of provided fats"):
            qLogExpectedImprovement(model, best_f=0.0, objective=lin, constraints=cons, fat=[True])
    with pytest.raises(TypeError, match="list of callables"):
        qExpectedImprovement(model, best_f=0.0, objective=lin, constraints=lambda Y: Y)

    class MultiOutput(MCAcquisitionObjective):
        _is_mo = True

        def forward(self, samples: Any, X: Any = None) -> Any:
            return samples

    with pytest.raises(UnsupportedError, match="Multi-output"):
        qSimpleRegret(model, objective=MultiOutput())


@requires_torch
def test_qlogpf_and_constrained_qlognei_behavior() -> None:
    import torch

    model = multi_model(m=2, kind="torch")
    X = candidates("torch", b=3, q=1)
    loose = qLogProbabilityOfFeasibility(
        model, constraints=[lambda Y: Y[..., 1] - 100.0], sampler=sampler()
    )
    tight = qLogProbabilityOfFeasibility(
        model, constraints=[lambda Y: Y[..., 1] + 100.0], sampler=sampler(), fat=False
    )
    assert torch.allclose(loose(X), torch.zeros(3, dtype=torch.float64), atol=1e-6)
    assert (tight(X) < -100).all()
    # All baseline points infeasible: BoTorch's warning, and the lower bound.
    nei = qLogNoisyExpectedImprovement(
        model,
        X_baseline=train_x(),
        sampler=sampler(),
        objective=LinearMCObjective([1.0, 0.0]),
        constraints=[lambda Y: Y[..., 1] + 100.0],
        prune_baseline=False,
    )
    assert shape(nei.lower_bound_points) == (32, 2)
    with pytest.warns(Warning, match="q\\(Log\\)ProbabilityOfFeasibility"):
        nei(X)
    assert nei._rs.infeasible_lower_bound() is not None
    explicit = qLogNoisyExpectedImprovement(
        model,
        X_baseline=train_x(),
        sampler=sampler(),
        objective=LinearMCObjective([1.0, 0.0]),
        constraints=[lambda Y: Y[..., 1] + 100.0],
        prune_baseline=False,
        lower_bound_points=torch.rand(5, 2, dtype=torch.float64),
    )
    assert shape(explicit.lower_bound_points) == (5, 2)
    # Pruning keeps only feasible points if some are feasible.
    Xb = train_x()
    pruned = prune_inferior_points(
        model,
        Xb,
        objective=LinearMCObjective([1.0, 0.0]),
        constraints=[lambda Y: Y[..., 1] - 1.0],
        sampler=sampler(128),
    )
    assert 0 < pruned.shape[0] <= Xb.shape[0]
    none_feasible = prune_inferior_points(
        model,
        Xb,
        objective=LinearMCObjective([1.0, 0.0]),
        constraints=[lambda Y: Y[..., 1] + 100.0],
        sampler=sampler(128),
    )
    assert none_feasible.shape[0] >= 1


# ---------------------------------------------------------------------------
# Parameter constraints, polytope sampling, initial conditions
# ---------------------------------------------------------------------------

INEQ = [([0, 1], [-1.0, -1.0], -1.2)]  # x0 + x1 <= 1.2
EQ = [([0, 1], [1.0, -1.0], 0.1)]  # x0 - x1 = 0.1
INTER = [([[0, 0], [1, 0]], [1.0, -1.0], 0.2)]  # x[0, 0] - x[1, 0] >= 0.2


def constraints_of(kind: str, cs: list[tuple]) -> list[tuple]:
    """Constraints with index and coefficient arrays of the given kind."""
    out = []
    for idx, coef, rhs in cs:
        if kind == "torch":
            import torch

            out.append((torch.tensor(idx), torch.tensor(coef, dtype=torch.float64), rhs))
        elif kind == "numpy":
            import numpy as np

            out.append((np.array(idx), np.array(coef), rhs))
        else:
            out.append((idx, coef, rhs))
    return out


def satisfies(X: Any, ineq: list[tuple] = (), eq: list[tuple] = (), tol: float = 1e-8) -> bool:
    """Checks (nested-list) candidates ``q x d`` or ``b x q x d``."""
    X = to_list(X)
    sets = X if isinstance(X[0][0], list) else [X]
    for x in sets:
        for idx, coef, rhs in ineq:
            if isinstance(idx[0], list):
                if sum(x[p][f] * c for (p, f), c in zip(idx, coef, strict=True)) < rhs - tol:
                    return False
            else:
                for point in x:
                    if sum(point[i] * c for i, c in zip(idx, coef, strict=True)) < rhs - tol:
                        return False
        for idx, coef, rhs in eq:
            for point in x:
                if abs(sum(point[i] * c for i, c in zip(idx, coef, strict=True)) - rhs) > tol:
                    return False
        if any(not (-1e-12 <= v <= 1 + 1e-12) for point in x for v in point):
            return False
    return True


@pytest.fixture(scope="module")
def single_model() -> SingleTaskGP:
    X, Y = make_data(n=12, d=2, seed=5)
    model = SingleTaskGP(X, Y)
    model.initialize(**{"covar_module.lengthscale": [[0.3, 0.5]], "likelihood.noise": [0.01]})
    return model


def qlogei(model: SingleTaskGP) -> qLogExpectedImprovement:
    best = max(to_list(model.train_targets))
    return qLogExpectedImprovement(model, best_f=best, sampler=sampler(64, seed=1))


@pytest.mark.parametrize(
    "case",
    [
        {"ineq": INEQ, "q": 1},
        {"ineq": INEQ, "eq": EQ, "q": 2},
        {"ineq": INTER + INEQ, "q": 2},
        {"ineq": INEQ, "q": 2, "fixed": {1: 0.3}},
        {"eq": EQ, "q": 1, "options": {"method": "SLSQP", "maxiter": 50}},
    ],
    ids=["ineq", "eq_ineq_q2", "inter_point", "fixed", "eq_options"],
)
def test_optimize_with_linear_constraints(
    kind: str, single_model: SingleTaskGP, case: dict
) -> None:
    acqf = qlogei(single_model)
    q = case["q"]
    X, value = optimize_acqf(
        acqf,
        as_kind(BOUNDS, kind),
        q=q,
        num_restarts=4,
        raw_samples=32,
        inequality_constraints=constraints_of(kind, case.get("ineq", [])) or None,
        equality_constraints=constraints_of(kind, case.get("eq", [])) or None,
        fixed_features=case.get("fixed"),
        options={"seed": 0, **case.get("options", {})},
    )
    check_kind(X, kind)
    assert shape(X) == (q, 2) and shape(value) == ()
    assert satisfies(X, case.get("ineq", []), case.get("eq", []))
    for k, v in (case.get("fixed") or {}).items():
        assert all(x[k] == v for x in to_list(X))
    assert float(value) == pytest.approx(flatten(acqf(X))[0], rel=1e-12)
    # Reproducible, and all restarts are feasible.
    X2, _ = optimize_acqf(
        acqf,
        as_kind(BOUNDS, kind),
        q=q,
        num_restarts=4,
        raw_samples=32,
        inequality_constraints=constraints_of(kind, case.get("ineq", [])) or None,
        equality_constraints=constraints_of(kind, case.get("eq", [])) or None,
        fixed_features=case.get("fixed"),
        options={"seed": 0, **case.get("options", {})},
        return_best_only=False,
    )
    assert shape(X2) == (4, q, 2)
    assert satisfies(X2, case.get("ineq", []), case.get("eq", []))


@requires_torch
@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_optimize_with_nonlinear_constraints(single_model: SingleTaskGP) -> None:
    import torch

    acqf = qlogei(single_model)
    bounds = torch.tensor(BOUNDS, dtype=torch.float64)

    def ball(x: Any) -> Any:
        return 0.3**2 - ((x - 0.5) ** 2).sum(-1)

    def separation(x: Any) -> Any:
        return ((x[0] - x[1]) ** 2).sum(-1) - 0.05

    for nonlinear, q in (([(ball, True)], 1), ([(ball, True), (separation, False)], 2)):
        X, value = optimize_acqf(
            acqf,
            bounds,
            q=q,
            num_restarts=3,
            raw_samples=32,
            nonlinear_inequality_constraints=nonlinear,
            ic_generator=gen_batch_initial_conditions,
            options={"seed": 1},
        )
        assert X.shape == (q, 2)
        assert bool(
            evaluate_feasibility(X.unsqueeze(0), nonlinear_inequality_constraints=nonlinear).all()
        )
    # Explicit (possibly infeasible) initial conditions.
    ics = torch.tensor([[[0.5, 0.5]], [[0.95, 0.95]]], dtype=torch.float64)
    X, _ = optimize_acqf(
        acqf,
        bounds,
        q=1,
        num_restarts=2,
        nonlinear_inequality_constraints=[(ball, True)],
        batch_initial_conditions=ics,
    )
    assert ball(X[0]) >= -1e-8
    with pytest.raises(RuntimeError, match="`ic_generator` must be given"):
        optimize_acqf(
            acqf,
            bounds,
            q=1,
            num_restarts=2,
            raw_samples=8,
            nonlinear_inequality_constraints=[(ball, True)],
        )
    with pytest.raises(ValueError, match="`batch_limit` must be 1"):
        optimize_acqf(
            acqf,
            bounds,
            q=1,
            num_restarts=2,
            raw_samples=8,
            nonlinear_inequality_constraints=[(ball, True)],
            ic_generator=gen_batch_initial_conditions,
            options={"batch_limit": 2},
        )
    with pytest.raises(ValueError, match="has to be a tuple"):
        optimize_acqf(
            acqf,
            bounds,
            q=1,
            num_restarts=2,
            raw_samples=8,
            nonlinear_inequality_constraints=[[ball, True]],
            ic_generator=gen_batch_initial_conditions,
        )


def test_optimize_constraint_errors_and_options(kind: str, single_model: SingleTaskGP) -> None:
    acqf = qlogei(single_model)
    bounds = as_kind(BOUNDS, kind)
    with pytest.raises(RuntimeWarning, match="L-BFGS-B cannot handle constraints"):
        optimize_acqf(
            acqf,
            bounds,
            q=1,
            num_restarts=2,
            raw_samples=8,
            inequality_constraints=constraints_of(kind, INEQ),
            options={"method": "L-BFGS-B"},
        )
    with pytest.raises(UnsupportedError, match="method"):
        optimize_acqf(
            acqf, bounds, q=1, num_restarts=2, raw_samples=8, options={"method": "COBYLA"}
        )
    infeasible = constraints_of(kind, [([0], [1.0], 0.8), ([0], [-1.0], -0.2)])
    with pytest.raises(InfeasibilityError, match="polytope appears empty"):
        optimize_acqf(
            acqf, bounds, q=1, num_restarts=2, raw_samples=8, inequality_constraints=infeasible
        )
    with pytest.raises(UnsupportedError, match="Inter-point constraints are not supported"):
        optimize_acqf(
            acqf,
            bounds,
            q=2,
            num_restarts=2,
            raw_samples=8,
            sequential=True,
            inequality_constraints=constraints_of(kind, INTER),
        )
    with pytest.raises(ValueError, match="Constraint indices cannot exceed"):
        optimize_acqf(
            acqf,
            bounds,
            q=1,
            num_restarts=2,
            raw_samples=8,
            inequality_constraints=constraints_of(kind, [([0, 2], [1.0, 1.0], 0.0)]),
        )
    # Sequential optimization with intra-point constraints.
    X, values = optimize_acqf(
        acqf,
        bounds,
        q=2,
        num_restarts=2,
        raw_samples=16,
        sequential=True,
        inequality_constraints=constraints_of(kind, INEQ),
        options={"seed": 3},
    )
    assert shape(X) == (2, 2) and shape(values) == (2,) and satisfies(X, INEQ)
    # 2-dim initial conditions: BoTorch's warning about raw_samples.
    with pytest.warns(RuntimeWarning, match="2-dim `batch_initial_conditions`"):
        optimize_acqf(
            acqf,
            bounds,
            q=1,
            num_restarts=1,
            raw_samples=8,
            batch_initial_conditions=as_kind([[0.2, 0.3]], kind),
            inequality_constraints=constraints_of(kind, INEQ),
        )


def test_custom_ic_generator(kind: str, single_model: SingleTaskGP) -> None:
    acqf = qlogei(single_model)
    calls: list[dict] = []

    def generator(
        acq_function: Any, bounds: Any, q: int, num_restarts: int, raw_samples: int, **kwargs: Any
    ) -> Any:
        calls.append({"acq_function": acq_function, "q": q, "num_restarts": num_restarts, **kwargs})
        return as_kind(
            [[[0.2 + 0.1 * i, 0.3] for _ in range(q)] for i in range(num_restarts)], kind
        )

    X, _ = optimize_acqf(
        acqf,
        as_kind(BOUNDS, kind),
        q=1,
        num_restarts=3,
        raw_samples=16,
        inequality_constraints=constraints_of(kind, INEQ),
        ic_generator=generator,
        extra_kwarg=7,
        options={"seed": 0},
    )
    assert len(calls) == 1 and calls[0]["acq_function"] is acqf and calls[0]["extra_kwarg"] == 7
    assert calls[0]["num_restarts"] == 3 and calls[0]["fixed_features"] is None
    assert set(calls[0]) >= {"options", "inequality_constraints", "equality_constraints"}
    assert satisfies(X, INEQ)


def test_gen_batch_initial_conditions(kind: str, single_model: SingleTaskGP) -> None:
    acqf = qlogei(single_model)
    bounds = as_kind(BOUNDS, kind)
    ics = gen_batch_initial_conditions(
        acqf, bounds, q=2, num_restarts=5, raw_samples=32, options={"seed": 0}
    )
    check_kind(ics, kind)
    assert shape(ics) == (5, 2, 2) and satisfies(ics)
    again = gen_batch_initial_conditions(
        acqf, bounds, q=2, num_restarts=5, raw_samples=32, options={"seed": 0}
    )
    assert flatten(again) == flatten(ics)
    constrained = gen_batch_initial_conditions(
        acqf,
        bounds,
        q=2,
        num_restarts=5,
        raw_samples=32,
        options={"seed": 0, "n_burnin": 50},
        inequality_constraints=constraints_of(kind, INTER + INEQ),
        fixed_features={1: 0.6},
    )
    assert shape(constrained) == (5, 2, 2)
    assert all(p[1] == 0.6 for x in to_list(constrained) for p in x)
    eq_ics = gen_batch_initial_conditions(
        acqf,
        bounds,
        q=1,
        num_restarts=4,
        raw_samples=16,
        options={"seed": 1},
        equality_constraints=constraints_of(kind, EQ),
    )
    assert satisfies(eq_ics, eq=EQ, tol=1e-9)
    # sample_around_best is supported (see test_9_optim.py); with equality
    # constraints it is not, as in BoTorch.
    with pytest.raises(UnsupportedError):
        gen_batch_initial_conditions(
            acqf, bounds, 1, 2, 8, options={"sample_around_best": True}, equality_constraints=EQ
        )
    with pytest.raises(NotImplementedError, match="finite values"):
        gen_batch_initial_conditions(acqf, as_kind([[0.0, -math.inf], [1.0, 1.0]], kind), 1, 2, 8)


def test_polytope_sampling(kind: str) -> None:
    bounds = as_kind(BOUNDS, kind)
    x = get_polytope_samples(
        200,
        bounds,
        inequality_constraints=constraints_of(kind, INEQ),
        equality_constraints=constraints_of(kind, EQ),
        seed=0,
        n_burnin=100,
        n_thinning=2,
    )
    check_kind(x, kind)
    assert shape(x) == (200, 2) and satisfies([to_list(x)], INEQ, EQ, tol=1e-9)
    xq = sample_q_batches_from_polytope(
        50,
        2,
        bounds,
        n_burnin=100,
        n_thinning=2,
        seed=1,
        inequality_constraints=constraints_of(kind, INTER),
    )
    assert shape(xq) == (50, 2, 2) and satisfies(xq, INTER)
    A, b = sparse_to_dense_constraints(2, constraints_of(kind, INEQ))
    assert to_list(A) == [[-1.0, -1.0]] and to_list(b) == [[-1.2]]
    sampler_ = HitAndRunPolytopeSampler(
        inequality_constraints=(as_kind([[1.0, 1.0]], kind), as_kind([[1.2]], kind)),
        bounds=bounds,
        n_burnin=20,
        n_thinning=2,
        seed=3,
    )
    d1, d2 = sampler_.draw(10), sampler_.draw(10)
    assert shape(d1) == (10, 2) and flatten(d1) != flatten(d2)
    assert satisfies([to_list(d1) + to_list(d2)], INEQ)
    A2 = as_kind([[-1.0, 0.0], [0.0, -1.0], [1.0, 2.0]], kind)
    b2 = as_kind([[0.0], [0.0], [2.0]], kind)
    x0 = find_interior_point(A2, b2)
    assert flatten(x0) == pytest.approx([0.5, 0.5], abs=1e-9)
    xs = sample_polytope(A2, b2, as_kind([[0.5], [0.5]], kind), n=100, n0=10, seed=0)
    assert shape(xs) == (100, 2)
    assert all(-1e-9 <= p[0] and -1e-9 <= p[1] and p[0] + 2 * p[1] <= 2 + 1e-9 for p in to_list(xs))
    with pytest.raises(InfeasibilityError):
        find_interior_point(as_kind([[1.0], [-1.0]], kind), as_kind([[0.2], [-0.8]], kind))


def test_transform_constraints_and_feasibility(kind: str) -> None:
    t = transform_constraints(constraints_of(kind, INEQ + INTER), q=2, d=2)
    assert [to_list(i) for i, _, _ in t] == [[0, 1], [2, 3], [0, 2]]
    assert [r for _, _, r in t] == [-1.2, -1.2, 0.2]
    X = as_kind(
        [[[0.5, 0.5], [0.2, 0.3]], [[0.9, 0.9], [0.1, 0.1]], [[0.3, 0.1], [0.2, 0.2]]], kind
    )
    feas = evaluate_feasibility(X, inequality_constraints=constraints_of(kind, INEQ + INTER))
    assert flatten([float(v) for v in to_list(feas)]) == [1.0, 0.0, 0.0]
    loose = evaluate_feasibility(
        X, inequality_constraints=constraints_of(kind, INEQ), tolerance=0.7
    )
    assert all(bool(v) for v in to_list(loose))
    if kind == "torch":
        import torch

        assert feas.dtype == torch.bool
    elif kind == "numpy":
        assert feas.dtype == bool

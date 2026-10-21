"""Phase 9 API tests (optimization and candidate generation): array types
(nested lists, NumPy, torch), shapes, reproducibility with seeds, the
corrected behaviors (see ``docs/drafts/9-optim.md``) and BoTorch's
exceptions for invalid input. The comparisons with BoTorch are in
``test_parity_9_optim.py`` and ``test_equivalence_9_optim.py``."""

from __future__ import annotations

import math
import warnings
from typing import Any

import pytest

from botorch_rs.acquisition import (
    LogExpectedImprovement,
    UpperConfidenceBound,
    qLogExpectedImprovement,
)
from botorch_rs.cross_validation import CVFolds, CVResults, batch_cross_validation, gen_loo_cv_folds
from botorch_rs.exceptions import BotorchRsError, OptimizationWarning, UnsupportedError
from botorch_rs.exceptions.errors import (
    CandidateGenerationError,
    InputDataError,
    OptimizationGradientError,
    OptimizationTimeoutError,
)
from botorch_rs.fit import fit_gpytorch_mll
from botorch_rs.generation import (
    BoltzmannSampling,
    ConstrainedMaxPosteriorSampling,
    MaxPosteriorSampling,
    gen_candidates_scipy,
    gen_candidates_torch,
    get_best_candidates,
)
from botorch_rs.mlls import ExactMarginalLogLikelihood
from botorch_rs.models import ModelListGP, SingleTaskGP
from botorch_rs.models.deterministic import PosteriorMeanModel
from botorch_rs.optim import (
    ExpMAStoppingCriterion,
    FixedHomotopySchedule,
    Homotopy,
    HomotopyParameter,
    LinearHomotopySchedule,
    LogLinearHomotopySchedule,
    OptimizationStatus,
    initialize_q_batch,
    initialize_q_batch_nonneg,
    initialize_q_batch_topn,
    optimize_acqf,
    optimize_acqf_cyclic,
    optimize_acqf_discrete,
    optimize_acqf_discrete_local_search,
    optimize_acqf_homotopy,
    optimize_acqf_mixed,
    optimize_acqf_mixed_alternating,
)
from botorch_rs.optim import optimize_mixed as om
from botorch_rs.optim.fit import fit_gpytorch_mll_scipy, fit_gpytorch_mll_torch
from botorch_rs.optim.homotopy import HomotopyValue
from botorch_rs.optim.initializers import gen_batch_initial_conditions, sample_points_around_best
from botorch_rs.optim.optimize_homotopy import prune_candidates
from botorch_rs.optim.utils import (
    columnwise_clamp,
    fix_features,
    get_name_filter,
    get_parameters,
    get_parameters_and_bounds,
    get_X_baseline,
    sample_all_priors,
)
from botorch_rs.sampling import StochasticSampler
from botorch_rs.utils.sampling import (
    batched_multinomial,
    boltzmann_sample,
    sample_perturbed_subset_dims,
    sample_truncated_normal_perturbations,
)
from helpers import HAS_NUMPY, HAS_TORCH, as_kind, check_kind, flatten, make_data, shape, to_list

X_TRAIN, Y_TRAIN = make_data(n=12, d=3, seed=21)
BOUNDS = [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]


@pytest.fixture(autouse=True)
def _quiet() -> Any:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        yield


@pytest.fixture(scope="module")
def model() -> SingleTaskGP:
    m = SingleTaskGP(X_TRAIN, Y_TRAIN)
    m.initialize(**{"covar_module.lengthscale": 0.4, "likelihood.noise": 0.01})
    return m.eval()


def logei(model: SingleTaskGP) -> LogExpectedImprovement:
    return LogExpectedImprovement(model, best_f=1.0)


def qlogei(model: SingleTaskGP) -> qLogExpectedImprovement:
    return qLogExpectedImprovement(
        PosteriorMeanModel(model), best_f=1.0, sampler=StochasticSampler([1])
    )


def grid(n: int, d: int, seed: int = 0) -> list[list[float]]:
    import random

    r = random.Random(seed)
    return [[r.random() for _ in range(d)] for _ in range(n)]


# ---------------------------------------------------------------------------
# Discrete and mixed optimizers
# ---------------------------------------------------------------------------


def test_optimize_acqf_discrete(kind: str, model: SingleTaskGP) -> None:
    choices = as_kind(grid(40, 3), kind)
    X, v = optimize_acqf_discrete(logei(model), 1, choices)
    check_kind(X, kind)
    assert shape(X) == (1, 3) and shape(v) == ()
    values = flatten(logei(model)(as_kind([[c] for c in grid(40, 3)], kind)))
    assert float(v) == pytest.approx(max(values), rel=1e-12)
    X, v = optimize_acqf_discrete(qlogei(model), 3, choices)
    assert shape(X) == (3, 3) and shape(v) == (3,)
    assert len({tuple(r) for r in to_list(X)}) == 3  # unique
    X, v = optimize_acqf_discrete(qlogei(model), 3, choices, return_acq_values=False)
    assert v is None
    with pytest.raises(InputDataError, match="must be non-empty"):
        optimize_acqf_discrete(
            logei(model), 1, as_kind([[0.5] * 3], kind), X_avoid=as_kind([[0.5] * 3], kind)
        )
    with pytest.warns(OptimizationWarning, match="possible choices remain"):
        warnings.simplefilter("always")
        X, _ = optimize_acqf_discrete(qlogei(model), 5, as_kind(grid(2, 3), kind))
    assert shape(X) == (2, 3)


def test_optimize_acqf_mixed(kind: str, model: SingleTaskGP) -> None:
    bounds = as_kind(BOUNDS, kind)
    ffl = [{0: 0.0}, {0: 1.0}, {-1: 0.5}]
    X, v = optimize_acqf_mixed(logei(model), bounds, 1, 4, ffl, raw_samples=32, options={"seed": 0})
    check_kind(X, kind)
    assert shape(X) == (1, 3) and shape(v) == ()
    assert to_list(X)[0][0] in (0.0, 1.0) or to_list(X)[0][2] == 0.5
    X, v = optimize_acqf_mixed(
        logei(model), bounds, 1, 4, ffl, raw_samples=32, return_best_only=False, options={"seed": 0}
    )
    assert shape(X) == (4, 1, 3) and shape(v) == (4,)
    X, v = optimize_acqf_mixed(
        qlogei(model), bounds, 2, 4, ffl, raw_samples=32, options={"seed": 0}
    )
    assert shape(X) == (2, 3) and shape(v) == (1,)
    X2, v2 = optimize_acqf_mixed(
        qlogei(model), bounds, 2, 4, ffl, raw_samples=32, options={"seed": 0}
    )
    assert flatten(X2) == flatten(X)
    with pytest.raises(ValueError, match="must be non-empty"):
        optimize_acqf_mixed(logei(model), bounds, 1, 4, [], raw_samples=32)
    with pytest.raises(NotImplementedError):
        optimize_acqf_mixed(
            qlogei(model), bounds, 2, 4, ffl, raw_samples=32, return_best_only=False
        )
    cons = [([0, 1], [1.0, 1.0], 1.9)]
    with pytest.raises(CandidateGenerationError, match="failed for all"):
        optimize_acqf_mixed(
            logei(model), bounds, 1, 2, [{0: 0.0}], raw_samples=8, inequality_constraints=cons
        )


def test_local_search(kind: str, model: SingleTaskGP) -> None:
    levels = as_kind([0.0, 0.25, 0.5, 0.75, 1.0], kind)
    dc = [levels] * 3
    X, v = optimize_acqf_discrete_local_search(
        logei(model), dc, 1, num_restarts=3, raw_samples=32, seed=0
    )
    check_kind(X, kind)
    assert shape(X) == (1, 3) and shape(v) == (1,)
    assert all(x in (0.0, 0.25, 0.5, 0.75, 1.0) for x in flatten(X))
    X2, _ = optimize_acqf_discrete_local_search(
        logei(model), dc, 1, num_restarts=3, raw_samples=32, seed=0
    )
    assert flatten(X2) == flatten(X)
    acqf = qlogei(model)
    X, v = optimize_acqf_discrete_local_search(
        acqf, dc, 3, num_restarts=3, raw_samples=32, seed=1, return_acq_values=False
    )
    assert shape(X) == (3, 3) and v is None and acqf.X_pending is None  # restored (BoTorch: not)
    with pytest.raises(ValueError, match="n x 1 x d"):
        optimize_acqf_discrete_local_search(
            logei(model), dc, 1, batch_initial_conditions=as_kind([[0.0] * 3], kind)
        )


def test_mixed_alternating(kind: str, model: SingleTaskGP) -> None:
    bounds = as_kind(BOUNDS, kind)
    kw = dict(
        discrete_dims={0: [0.0, 0.5, 1.0]},
        cat_dims={2: [0.0, 0.5, 1.0]},
        num_restarts=4,
        raw_samples=32,
    )
    X, v = optimize_acqf_mixed_alternating(logei(model), bounds, seed=0, **kw)
    check_kind(X, kind)
    assert shape(X) == (1, 3) and shape(v) == (1,)
    assert flatten(X)[0] in (0.0, 0.5, 1.0) and flatten(X)[2] in (0.0, 0.5, 1.0)
    X2, v2 = optimize_acqf_mixed_alternating(logei(model), bounds, seed=0, **kw)
    assert flatten(X2) == flatten(X)
    X, v = optimize_acqf_mixed_alternating(
        qlogei(model), bounds, q=2, seed=0, fixed_features={1: 0.3}, **kw
    )
    assert shape(X) == (2, 3) and all(row[1] == 0.3 for row in to_list(X))
    # A discrete dimension with many values is relaxed and rounded.
    X, _ = optimize_acqf_mixed_alternating(
        logei(model),
        bounds,
        discrete_dims={0: [i / 40 for i in range(41)]},
        num_restarts=3,
        raw_samples=16,
        seed=1,
    )
    assert flatten(X)[0] * 40 == round(flatten(X)[0] * 40)
    # Fully continuous: plain optimize_acqf.
    X, _ = optimize_acqf_mixed_alternating(logei(model), bounds, num_restarts=3, raw_samples=16)
    assert shape(X) == (1, 3)
    for bad, err in (
        ({"discrete_dims": {0: [0.1, 1.0]}}, ValueError),
        ({"options": {"foo": 1}}, UnsupportedError),
        ({"options": {"initialization_strategy": "foo"}}, UnsupportedError),
        ({"sequential": False}, NotImplementedError),
    ):
        with pytest.raises(err):
            optimize_acqf_mixed_alternating(logei(model), bounds, **{**kw, **bad})


def test_mixed_building_blocks(kind: str) -> None:
    X = as_kind([[0.12, 0.9], [0.6, 0.2]], kind)
    dd = {0: [0.0, 0.5, 1.0]}
    R = om.round_discrete_dims(X, dd)
    check_kind(R, kind)
    assert to_list(R) == [[0.0, 0.9], [0.5, 0.2]]
    nb = om.get_nearest_neighbors(
        as_kind([0.5, 0.2], kind), as_kind([[0.0, 0.0], [1.0, 1.0]], kind), dd
    )
    assert to_list(nb) == [[0.0, 0.2], [1.0, 0.2]]
    cn = om.get_categorical_neighbors(as_kind([0.5, 1.0], kind), {1: [0.0, 1.0, 2.0]})
    assert to_list(cn) == [[0.5, 0.0], [0.5, 2.0]]
    many = {1: [float(i) for i in range(30)]}
    cn = om.get_categorical_neighbors(as_kind([0.5, 3.0], kind), many, seed=0)
    assert shape(cn) == (20, 2) and 3.0 not in [r[1] for r in to_list(cn)]
    sp = om.get_spray_points(
        as_kind([[0.5, 0.0, 0.3]], kind),
        [2],
        {0: [0.0, 0.5, 1.0]},
        {1: [0.0, 1.0]},
        as_kind([[0.0] * 3, [1.0] * 3], kind),
        5,
        seed=0,
    )
    assert shape(sp) == (5, 3)
    assert om.complement_indices([0, 2], 4) == [1, 3]
    assert om.should_use_mixed_alternating_optimizer({0: [0, 1, 2, 3]}, {1: list(range(3))})


def test_cyclic(kind: str, model: SingleTaskGP) -> None:
    bounds = as_kind(BOUNDS, kind)
    X, v = optimize_acqf_cyclic(
        qlogei(model),
        bounds,
        3,
        3,
        raw_samples=16,
        options={"seed": 0},
        cyclic_options={"maxiter": 3},
    )
    check_kind(X, kind)
    assert shape(X) == (3, 3) and shape(v) == (3,)
    X, v = optimize_acqf_cyclic(logei(model), bounds, 1, 3, raw_samples=16, options={"seed": 0})
    assert shape(X) == (1, 3) and shape(v) == ()
    with pytest.raises(ValueError, match="raw_samples"):
        optimize_acqf_cyclic(qlogei(model), bounds, 2, 3)


# ---------------------------------------------------------------------------
# Homotopy
# ---------------------------------------------------------------------------


def test_homotopy_mechanics() -> None:
    value = HomotopyValue(5.0)
    calls: list[float] = []
    h = Homotopy(
        [HomotopyParameter(value, LinearHomotopySchedule(1.0, 0.0, 3))],
        callbacks=[lambda: calls.append(value.value)],
    )
    assert value.value == 1.0 and not h.should_stop
    h.step()
    h.step()
    assert value.value == 0.0
    h.step()
    assert h.should_stop and value.value == 0.0
    h.reset()
    assert value.value == 5.0 and calls == [1.0, 0.5, 0.0, 0.0, 5.0]
    assert FixedHomotopySchedule([3.0, 2.0]).num_steps == 2
    assert LogLinearHomotopySchedule(1.0, 0.01, 3)._values == pytest.approx(
        [1.0, 0.1, 0.01], rel=1e-15
    )
    with pytest.raises(AssertionError):
        Homotopy(
            [
                HomotopyParameter(value, LinearHomotopySchedule(1, 0, 3)),
                HomotopyParameter(HomotopyValue(1), LinearHomotopySchedule(1, 0, 4)),
            ]
        )


@pytest.mark.skipif(not HAS_TORCH, reason="torch callables")
def test_optimize_acqf_homotopy(model: SingleTaskGP) -> None:
    import torch

    class Peak(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.s = torch.nn.Parameter(torch.tensor(1.0, dtype=torch.float64))

        def forward(self, X: Any) -> Any:
            return -((X - 0.3) / self.s).pow(2).sum(-1).sum(-1)

    acqf = Peak()
    h = Homotopy([HomotopyParameter(acqf.s, LinearHomotopySchedule(1.0, 0.1, 3))])
    bounds = torch.tensor(BOUNDS, dtype=torch.float64)
    X, v = optimize_acqf_homotopy(
        acqf, bounds, q=1, num_restarts=4, homotopy=h, raw_samples=16, options={"seed": 0}
    )
    assert X.shape == (1, 3) and v.shape == (1,)
    assert torch.allclose(X, torch.full((1, 3), 0.3, dtype=torch.float64), atol=1e-4)
    assert float(acqf.s) == 1.0  # reset
    pruned = prune_candidates(
        torch.tensor([[0.0], [1e-6], [1.0]], dtype=torch.float64),
        torch.tensor([1.0, 2.0, 0.0]),
        1e-3,
    )
    assert pruned.tolist() == [[1e-6], [1.0]]
    with pytest.raises(ValueError):
        prune_candidates(torch.zeros(3), torch.zeros(3), 0.1)


# ---------------------------------------------------------------------------
# Candidate generation and initializers
# ---------------------------------------------------------------------------


def test_gen_candidates(kind: str, model: SingleTaskGP) -> None:
    ics = as_kind([[[0.2, 0.3, 0.4]], [[0.7, 0.6, 0.5]]], kind)
    for gen in (gen_candidates_scipy, gen_candidates_torch):
        X, v = gen(ics, logei(model), lower_bounds=0.0, upper_bounds=1.0)
        check_kind(X, kind)
        assert shape(X) == (2, 1, 3) and shape(v) == (2,)
        best = get_best_candidates(X, v)
        assert shape(best) == (1, 3)
    X, v = gen_candidates_torch(
        ics, logei(model), lower_bounds=0.0, upper_bounds=1.0, fixed_features={1: 0.25}
    )
    assert all(r[0][1] == 0.25 for r in to_list(X))  # works where BoTorch fails
    steps: list[int] = []
    if HAS_TORCH:
        gen_candidates_torch(
            ics,
            logei(model),
            0.0,
            1.0,
            options={"stopping_criterion_options": {"maxiter": 7}},
            callback=lambda i, loss, g: steps.append(i),
        )
        assert steps == list(range(1, 8))
    with pytest.warns(DeprecationWarning):
        warnings.simplefilter("always")
        gen_candidates_torch(ics, logei(model), 0.0, 1.0, options={"maxiter": 3})
    X, v = optimize_acqf(
        logei(model),
        as_kind(BOUNDS, kind),
        1,
        2,
        raw_samples=16,
        gen_candidates=gen_candidates_torch,
        options={"seed": 0},
    )
    assert shape(X) == (1, 3)
    X, v = optimize_acqf(
        logei(model),
        as_kind(BOUNDS, kind),
        1,
        2,
        raw_samples=16,
        post_processing_func=lambda X: X,
        timeout_sec=10.0,
        options={"seed": 0},
    )
    assert shape(X) == (1, 3)


def test_initializers(kind: str, model: SingleTaskGP) -> None:
    X = as_kind([[[i / 20, 0.0]] for i in range(20)], kind)
    vals = as_kind([math.sin(i) for i in range(20)], kind)
    Xs, vs = initialize_q_batch(X, vals, 5)
    check_kind(Xs, kind)
    assert shape(Xs) == (5, 1, 2) and max(flatten(vals)) in flatten(vs)
    Xs, vs = initialize_q_batch_nonneg(
        X, as_kind([max(math.sin(i), 0.0) for i in range(20)], kind), 5
    )
    assert shape(Xs) == (5, 1, 2)
    Xs, vs = initialize_q_batch_topn(X, vals, 3)
    assert flatten(vs) == sorted(flatten(vals), reverse=True)[:3]
    with pytest.raises(RuntimeError, match="cannot be larger"):
        initialize_q_batch(X, vals, 30)
    bounds = as_kind(BOUNDS, kind)
    ics = gen_batch_initial_conditions(
        logei(model), bounds, 1, 4, 32, options={"seed": 0, "topn": True}
    )
    assert shape(ics) == (4, 1, 3)
    ics = gen_batch_initial_conditions(
        logei(model), bounds, 1, 4, 32, options={"seed": 0, "sample_around_best": True}
    )
    assert shape(ics) == (4, 1, 3)
    X, _ = optimize_acqf(
        logei(model),
        bounds,
        1,
        4,
        raw_samples=32,
        options={"seed": 0, "sample_around_best": True, "topn": True},
    )
    assert shape(X) == (1, 3)
    pts = sample_points_around_best(logei(model), 30, 1e-3, bounds, seed=0)
    assert shape(pts) == (30, 3) and all(0.0 <= v <= 1.0 for v in flatten(pts))
    pts = sample_truncated_normal_perturbations(
        as_kind([[0.5, 0.5, 0.5]], kind), 10, 0.01, bounds, seed=0
    )
    assert shape(pts) == (10, 3) and all(abs(v - 0.5) < 0.1 for v in flatten(pts))
    base = [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]
    pts = to_list(
        sample_perturbed_subset_dims(
            as_kind(base, kind), bounds, 200, sigma=1e-6, prob_perturb=0.5, seed=0
        )
    )
    # Every point is perturbed around one base point (BoTorch mixes two).
    assert all(max(r) - min(r) < 1e-3 for r in pts)


def test_sampling_strategies(kind: str, model: SingleTaskGP) -> None:
    Xc = as_kind(grid(60, 3, seed=4), kind)
    for repl in (True, False):
        S = MaxPosteriorSampling(model, replacement=repl)(Xc, num_samples=5)
        check_kind(S, kind)
        assert shape(S) == (5, 3)
        rows = [tuple(r) for r in to_list(S)]
        assert all(r in {tuple(c) for c in grid(60, 3, seed=4)} for r in rows)
        if not repl:
            assert len(set(rows)) == 5
    S = BoltzmannSampling(UpperConfidenceBound(model, beta=0.1), eta=1.0)(Xc, num_samples=4)
    assert shape(S) == (4, 3)
    S = ConstrainedMaxPosteriorSampling(model, ModelListGP(model))(Xc, num_samples=3)
    assert shape(S) == (3, 3)
    with pytest.raises(NotImplementedError):
        ConstrainedMaxPosteriorSampling(model, ModelListGP(model), objective=lambda Y, X=None: Y)
    idx = to_list(batched_multinomial(as_kind([[1.0, 0.0, 1.0]], kind), 2))
    assert sorted(idx[0]) == [0, 2]
    assert shape(boltzmann_sample(as_kind([0.0, 1.0, 2.0], kind), 2, eta=1.0)) == (2,)


# ---------------------------------------------------------------------------
# Fitting, stopping, closures, utilities
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("m", [1, 2])
def test_fit_routines(kind: str, m: int) -> None:
    Y = [[y[0]] + ([y[0] ** 2] if m == 2 else []) for y in Y_TRAIN]
    model = SingleTaskGP(as_kind(X_TRAIN, kind), as_kind(Y, kind))
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    r = fit_gpytorch_mll_torch(mll, step_limit=50)
    assert r.step == 50 and r.status == OptimizationStatus.STOPPED and "step_limit=50" in r.message
    r = fit_gpytorch_mll_scipy(mll)
    assert r.status in (OptimizationStatus.SUCCESS, OptimizationStatus.STOPPED)
    params = get_parameters(mll)
    assert list(params)[0].startswith("likelihood.")
    sub = {k: v for k, v in params.items() if "lengthscale" in k}
    before = {k: flatten(v) for k, v in get_parameters(mll).items()}
    fit_gpytorch_mll_torch(mll, parameters=sub, step_limit=20)
    after = {k: flatten(v) for k, v in get_parameters(mll).items()}
    assert all(after[k] == before[k] for k in before if "lengthscale" not in k)
    with pytest.raises(RuntimeError, match="No termination"):
        fit_gpytorch_mll_torch(mll, stopping_criterion=None)
    fit_gpytorch_mll(mll, optimizer=fit_gpytorch_mll_torch, optimizer_kwargs={"step_limit": 30})
    assert not mll.training
    _, bounds = get_parameters_and_bounds(mll)
    assert all(lo is not None for lo, _ in bounds.values())
    sample_all_priors(model, seed=0)
    f = get_name_filter(["likelihood.noise_covar.raw_noise"])
    assert not f("likelihood.noise_covar.raw_noise") and f("model.mean_module.raw_constant")


def test_stopping_criterion() -> None:
    c = ExpMAStoppingCriterion(maxiter=4)
    assert [c(v) for v in (3.0, 2.0, 1.0, 0.5)] == [False, False, False, True]
    c = ExpMAStoppingCriterion(n_window=2, rel_tol=1e-3)
    out = [c([1.0, 2.0]) for _ in range(4)]
    assert out == [False, False, True, True]
    assert len(ExpMAStoppingCriterion().weights) == 10
    assert sum(ExpMAStoppingCriterion(eta=0.3, n_window=4).weights) == pytest.approx(1.0, abs=1e-6)


@pytest.mark.skipif(not HAS_NUMPY, reason="NdarrayOptimizationClosure uses NumPy")
def test_closures(model: SingleTaskGP) -> None:
    from botorch_rs.optim.closures import (
        ForwardBackwardClosure,
        NdarrayOptimizationClosure,
        get_loss_closure,
        get_loss_closure_with_grads,
    )

    m = SingleTaskGP(X_TRAIN, Y_TRAIN)
    mll = ExactMarginalLogLikelihood(m.likelihood, m)
    loss = get_loss_closure(mll)
    assert float(loss()) == pytest.approx(-float(mll()), rel=1e-15)
    params = get_parameters(mll)
    closure = get_loss_closure_with_grads(mll, params)
    assert isinstance(closure, ForwardBackwardClosure)
    value, grads = closure()
    assert len(grads) == len(params)
    nd = NdarrayOptimizationClosure(closure, params)
    state = nd.state
    v1, g1 = nd(state)
    assert float(v1) == pytest.approx(float(value), rel=1e-14)
    nd.state = state + 0.01
    assert nd.state == pytest.approx(state + 0.01, rel=1e-15)


def test_utils(kind: str, model: SingleTaskGP) -> None:
    X = as_kind([[[0.5, -1.0, 2.0]]], kind)
    C = columnwise_clamp(X, as_kind([0.0, 0.0, 0.0], kind), 1.0)
    check_kind(C, kind)
    assert flatten(C) == [0.5, 0.0, 1.0]
    with pytest.raises(BotorchRsError, match="out of bounds"):
        columnwise_clamp(X, 0.0, 1.0, raise_on_violation=True)
    F = fix_features(as_kind([[[0.5, 0.6]]], kind), {1: 0.0}, replace_current_value=False)
    assert flatten(F) == [0.5, 0.0, 0.6]
    F = fix_features(as_kind([[[0.5, 0.6]], [[0.1, 0.2]]], kind), {0: as_kind([3.0, 4.0], kind)})
    assert flatten(F) == [3.0, 0.6, 4.0, 0.2]
    assert shape(get_X_baseline(logei(model))) == (12, 3)
    assert issubclass(OptimizationGradientError, BotorchRsError)
    e = OptimizationTimeoutError("t", current_x=[1.0], runtime=2.0)
    assert e.runtime == 2.0


def test_cross_validation(kind: str) -> None:
    folds = gen_loo_cv_folds(as_kind(X_TRAIN[:6], kind), as_kind([y[0] for y in Y_TRAIN[:6]], kind))
    assert isinstance(folds, CVFolds)
    assert shape(folds.train_X) == (6, 5, 3) and shape(folds.test_Y) == (6, 1, 1)
    assert folds.train_Yvar is None
    res = batch_cross_validation(SingleTaskGP, ExactMarginalLogLikelihood, folds)
    assert isinstance(res, CVResults)
    assert shape(res.posterior.mean) == (6, 1, 1) and res.observed_Y is folds.test_Y
    folds = gen_loo_cv_folds(
        as_kind(X_TRAIN[:6], kind), as_kind(Y_TRAIN[:6], kind), as_kind([[0.01]] * 6, kind)
    )
    assert shape(folds.test_Yvar) == (6, 1, 1)


def test_unsupported_and_errors(model: SingleTaskGP) -> None:
    with pytest.raises(ValueError, match="sequence requires sequential"):
        optimize_acqf(
            logei(model), BOUNDS, 1, 2, raw_samples=8, acq_function_sequence=[logei(model)]
        )
    with pytest.raises(UnsupportedError, match="equality"):
        gen_batch_initial_conditions(
            logei(model),
            BOUNDS,
            1,
            2,
            8,
            options={"sample_around_best": True},
            equality_constraints=[([0, 1], [1.0, 1.0], 1.0)],
        )


def test_one_shot_with_phase9_options(model: SingleTaskGP) -> None:
    """qKnowledgeGradient and qHypervolumeKnowledgeGradient (phases 8 and 5c)
    with the options of this phase: BoTorch hands ``options`` (``topn``,
    ``sample_around_best``) to ``gen_batch_initial_conditions`` of the
    one-shot initial conditions and supports ``post_processing_func``,
    ``timeout_sec`` and ``gen_candidates_torch`` on the augmented sets."""
    from botorch_rs.acquisition import qKnowledgeGradient
    from botorch_rs.acquisition.multi_objective import qHypervolumeKnowledgeGradient
    from botorch_rs.optim.initializers import (
        gen_one_shot_hvkg_initial_conditions,
        gen_one_shot_kg_initial_conditions,
    )
    from botorch_rs.sampling import ListSampler, SobolQMCNormalSampler

    kg = qKnowledgeGradient(model, num_fantasies=3, sampler=SobolQMCNormalSampler((3,), seed=0))
    # topn without value-function points (frac_random close to 1): the top
    # augmented raw samples, sorted.
    ics = gen_one_shot_kg_initial_conditions(
        kg, BOUNDS, 1, 4, 16, options={"topn": True, "frac_random": 0.99, "seed": 3}
    )
    assert shape(ics) == (4, 4, 3)
    values = flatten(kg(ics))
    assert values == sorted(values, reverse=True)
    plain = gen_one_shot_kg_initial_conditions(
        kg, BOUNDS, 1, 4, 16, options={"frac_random": 0.99, "seed": 3}
    )
    assert flatten(plain) != flatten(ics)
    # sample_around_best adds points around the best training points.
    around = gen_one_shot_kg_initial_conditions(
        kg, BOUNDS, 1, 4, 16, options={"sample_around_best": True, "seed": 3}
    )
    assert shape(around) == (4, 4, 3)
    assert all(0.0 <= v <= 1.0 for v in flatten(around))
    for kw in (
        {"options": {"topn": True, "seed": 1}},
        {"options": {"sample_around_best": True, "seed": 1}},
        {"options": {"seed": 1}, "gen_candidates": gen_candidates_torch},
        {"options": {"seed": 1}, "timeout_sec": 100.0},
    ):
        cand, value = optimize_acqf(kg, BOUNDS, 1, 2, raw_samples=16, **kw)
        assert shape(cand) == (1, 3)
        assert math.isfinite(flatten(value)[0])
    full, _ = optimize_acqf(
        kg,
        BOUNDS,
        1,
        2,
        raw_samples=16,
        options={"seed": 1},
        return_full_tree=True,
        post_processing_func=lambda X: X,
    )
    assert shape(full) == (4, 3)
    # BoTorch's error for sample_around_best with equality constraints.
    with pytest.raises(UnsupportedError, match="equalityconstraints"):
        optimize_acqf(
            kg,
            BOUNDS,
            1,
            2,
            raw_samples=16,
            options={"sample_around_best": True},
            equality_constraints=[([0, 1], [1.0, 1.0], 1.0)],
        )
    # The hypervolume knowledge gradient.
    X, Y = make_data(n=10, d=2, seed=4)
    Y2 = [[-((x[0] - 0.3) ** 2) - x[1]] for x in X]
    models = [SingleTaskGP(X, Y), SingleTaskGP(X, Y2)]
    hv = qHypervolumeKnowledgeGradient(
        ModelListGP(*models),
        ref_point=[-1.5, -1.0],
        num_fantasies=2,
        num_pareto=2,
        sampler=ListSampler(*[SobolQMCNormalSampler((2,), seed=k) for k in range(2)]),
    )
    bounds = [[0.0, 0.0], [1.0, 1.0]]
    opts = {"seed": 0, "num_inner_restarts": 2, "raw_inner_samples": 16, "maxiter": 30}
    for extra in ({"topn": True}, {"sample_around_best": True}):
        ics = gen_one_shot_hvkg_initial_conditions(hv, bounds, 1, 3, 16, options={**opts, **extra})
        assert shape(ics) == (3, 5, 2)
        cand, value = optimize_acqf(hv, bounds, 1, 2, raw_samples=16, options={**opts, **extra})
        assert shape(cand) == (1, 2)
        assert math.isfinite(flatten(value)[0])
    cand, _ = optimize_acqf(
        hv, bounds, 1, 2, raw_samples=16, options=opts, post_processing_func=lambda X: X
    )
    assert shape(cand) == (1, 2)

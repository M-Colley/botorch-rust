"""The multi-objective acquisition functions through the public API, for every
array type: result types and shapes, pending points and the state of the
noisy hypervolume acquisition functions, gradients (finite differences,
torch.autograd), objectives, pruning, ``optimize_acqf_list``, thread-count
independence and errors. The comparisons with BoTorch are in
``test_parity_mo_acquisition.py`` and ``test_equivalence_multi_objective.py``.
"""

from __future__ import annotations

import math
import random
import warnings
from typing import Any

import pytest

import botorch_rs
from botorch_rs.acquisition.multi_objective import (
    ExpectedHypervolumeImprovement,
    FeasibilityWeightedMCMultiOutputObjective,
    GenericMCMultiOutputObjective,
    IdentityMCMultiOutputObjective,
    WeightedMCMultiOutputObjective,
    get_default_partitioning_alpha,
    prune_inferior_points_multi_objective,
    qExpectedHypervolumeImprovement,
    qLogExpectedHypervolumeImprovement,
    qLogNoisyExpectedHypervolumeImprovement,
    qLogNParEGO,
    qNoisyExpectedHypervolumeImprovement,
)
from botorch_rs.exceptions import (
    BotorchError,
    BotorchTensorDimensionError,
    BotorchWarning,
    NumericsWarning,
    UnsupportedError,
)
from botorch_rs.fit import fit_gpytorch_mll
from botorch_rs.mlls import ExactMarginalLogLikelihood
from botorch_rs.models import ModelListGP, SingleTaskGP
from botorch_rs.optim.optimize import optimize_acqf_list
from botorch_rs.sampling import SobolQMCNormalSampler
from botorch_rs.utils.multi_objective.box_decompositions import FastNondominatedPartitioning
from helpers import HAS_TORCH, as_kind, assert_close, check_kind, flatten, shape


def _data(n: int, seed: int) -> tuple[list, list]:
    """Two conflicting smooth objectives on [0, 1]^2 plus noise."""
    rng = random.Random(seed)
    X = [[rng.random(), rng.random()] for _ in range(n)]
    Y = [
        [
            math.sin(3.0 * x[0]) + 0.5 * x[1] + 0.02 * rng.gauss(0.0, 1.0),
            math.cos(3.0 * x[0]) - 0.5 * x[1] ** 2 + 0.02 * rng.gauss(0.0, 1.0),
        ]
        for x in X
    ]
    return X, Y


# qEHVI and qNEHVI recommend their log variants (as in BoTorch); see
# test_legacy_functions_warn.
pytestmark = pytest.mark.filterwarnings("ignore::botorch_rs.exceptions.NumericsWarning")

X_TRAIN, Y_TRAIN = _data(12, seed=5)
REF = [min(y[j] for y in Y_TRAIN) - 0.2 for j in range(2)]
SETS = [[[0.1, 0.9], [0.4, 0.2]], [[0.5, 0.5], [0.8, 0.7]]]
POINTS = [[[0.1, 0.9]], [[0.5, 0.5]], [[0.95, 0.05]]]


@pytest.fixture(scope="module")
def model() -> SingleTaskGP:
    m = SingleTaskGP(X_TRAIN, Y_TRAIN)
    fit_gpytorch_mll(ExactMarginalLogLikelihood(m.likelihood, m))
    return m


def sampler(n: int = 64, seed: int = 3) -> SobolQMCNormalSampler:
    return SobolQMCNormalSampler(sample_shape=[n], seed=seed)


def partitioning(kind: str) -> FastNondominatedPartitioning:
    return FastNondominatedPartitioning(ref_point=as_kind(REF, kind), Y=as_kind(Y_TRAIN, kind))


def make(name: str, model: Any, kind: str = "list", **kwargs: Any) -> Any:
    """An acquisition function on lists or arrays of ``kind``."""
    ref = as_kind(REF, kind)
    kwargs.setdefault("sampler", sampler())
    if name == "qEHVI":
        return qExpectedHypervolumeImprovement(model, ref, partitioning(kind), **kwargs)
    if name == "qLogEHVI":
        return qLogExpectedHypervolumeImprovement(model, ref, partitioning(kind), **kwargs)
    baseline = as_kind(X_TRAIN, kind)
    if name == "qNEHVI":
        return qNoisyExpectedHypervolumeImprovement(model, ref, baseline, **kwargs)
    if name == "qLogNEHVI":
        return qLogNoisyExpectedHypervolumeImprovement(model, ref, baseline, **kwargs)
    if name == "qLogNParEGO":
        kwargs.setdefault("scalarization_weights", as_kind([0.3, 0.7], kind))
        return qLogNParEGO(model, baseline, **kwargs)
    raise ValueError(name)


MC = ["qEHVI", "qLogEHVI", "qNEHVI", "qLogNEHVI", "qLogNParEGO"]
NOISY = ["qNEHVI", "qLogNEHVI"]


@pytest.mark.parametrize("name", MC)
def test_shapes_and_pending_points(name: str, kind: str, model: SingleTaskGP) -> None:
    acqf = make(name, model, kind)
    values = acqf(as_kind(SETS, kind))
    check_kind(values, kind)
    assert shape(values) == (2,)
    assert all(math.isfinite(v) for v in flatten(values))
    assert shape(acqf(as_kind(SETS[0], kind))) == (1,)
    assert shape(acqf(as_kind([SETS] * 3, kind))) == (3, 2)
    assert acqf.X_pending is None
    pending = as_kind([[0.3, 0.3], [0.7, 0.1]], kind)
    acqf.set_X_pending(pending)
    check_kind(acqf.X_pending, kind)
    assert shape(acqf.X_pending) == (2, 2)
    with_pending = flatten(acqf(as_kind(SETS, kind)))
    assert with_pending != flatten(values)
    acqf.set_X_pending(None)
    assert acqf.X_pending is None
    assert_close(acqf(as_kind(SETS, kind)), values, rtol=1e-12, atol=1e-12)
    with pytest.raises(BotorchTensorDimensionError):
        acqf.set_X_pending(as_kind([[0.3, 0.3, 0.3]], kind))
    constructed = make(name, model, kind, X_pending=pending)
    assert_close(constructed(as_kind(SETS, kind)), with_pending, rtol=1e-10, atol=1e-12)


def test_log_variants_are_the_logarithms(model: SingleTaskGP) -> None:
    """With small temperatures, the log variants approach the logarithms of
    the plain ones where improvements are likely."""
    for plain, log in [("qEHVI", "qLogEHVI"), ("qNEHVI", "qLogNEHVI")]:
        v = flatten(make(plain, model)(SETS))
        lv = flatten(make(log, model, tau_relu=1e-6, tau_max=1e-6)(SETS))
        for a, b in zip(v, lv, strict=True):
            if a > 1e-3:
                assert math.exp(b) == pytest.approx(a, rel=1e-2)


@pytest.mark.parametrize("name", NOISY)
def test_noisy_state(name: str, kind: str, model: SingleTaskGP) -> None:
    acqf = make(name, model, kind)
    check_kind(acqf.X_baseline, kind)
    assert shape(acqf.X_baseline) == (12, 2)
    lower = acqf.cell_lower_bounds
    check_kind(lower, kind)
    assert shape(lower)[0] == 64 and shape(lower)[2] == 2
    assert shape(acqf.cell_upper_bounds) == shape(lower)
    assert shape(acqf._hypervolumes) == (64,)
    # Incremental NEHVI caches pending points in the decompositions.
    acqf.set_X_pending(as_kind([[0.3, 0.3], [0.7, 0.1]], kind))
    assert shape(acqf.X_baseline) == (14, 2)
    assert shape(acqf._X_baseline) == (12, 2)
    acqf.set_X_pending(as_kind([[0.2, 0.8]], kind))  # replaces them
    assert shape(acqf.X_baseline) == (13, 2)
    acqf.set_X_pending(None)
    assert shape(acqf.X_baseline) == (12, 2)
    # Up to max_iep pending points are evaluated jointly with the candidates.
    joint = make(name, model, kind, max_iep=3)
    joint.set_X_pending(as_kind([[0.3, 0.3], [0.7, 0.1]], kind))
    assert shape(joint.X_baseline) == (12, 2)
    # Non-incremental NEHVI keeps the improvement of cached pending points.
    full = make(name, model, kind, incremental_nehvi=False)
    with pytest.raises(AttributeError):
        make(name, model, kind)._baseline_hvs  # noqa: B018
    assert shape(full._baseline_hvs) == (64,)
    assert float(full._prev_nehvi) == 0.0
    full.set_X_pending(as_kind([[0.95, 0.05]], kind))
    assert float(full._prev_nehvi) >= 0.0


@pytest.mark.parametrize("name", NOISY)
def test_noisy_pruning_and_cache_root(name: str, model: SingleTaskGP) -> None:
    pruned = make(name, model, prune_baseline=True)
    kept = [list(x) for x in pruned.X_baseline]
    assert 1 <= len(kept) <= len(X_TRAIN) and all(x in X_TRAIN for x in kept)
    # Reproducible for a given sampler seed (pruning is seeded from it).
    again = make(name, model, prune_baseline=True, sampler=sampler())
    assert flatten(again(SETS)) == flatten(pruned(SETS))
    # The cached root decomposition and the joint posterior agree.
    a = flatten(make(name, model, cache_root=True)(SETS))
    b = flatten(make(name, model, cache_root=False)(SETS))
    assert_close(a, b, rtol=1e-9, atol=1e-12)


def test_analytic_ehvi(kind: str, model: SingleTaskGP) -> None:
    acqf = ExpectedHypervolumeImprovement(model, as_kind(REF, kind), partitioning(kind))
    values = acqf(as_kind(POINTS, kind))
    check_kind(values, kind)
    assert shape(values) == (3,)
    assert all(v >= 0.0 for v in flatten(values))
    with pytest.raises(BotorchTensorDimensionError):
        acqf(as_kind(SETS, kind))  # q = 2
    # The analytic value is the limit of qEHVI.
    mc = make("qEHVI", model, sampler=sampler(4096, 0))(POINTS)
    assert_close(mc, acqf(POINTS), rtol=2e-2, atol=1e-3 * max(flatten(values)))


@pytest.mark.parametrize("name", MC + ["EHVI"])
def test_gradients_match_finite_differences(name: str, model: SingleTaskGP) -> None:
    if name == "EHVI":
        acqf = ExpectedHypervolumeImprovement(model, REF, partitioning("list"))
        X = [[[0.3, 0.6]]]
    else:
        acqf = make(name, model)
        X = [[[0.3, 0.6], [0.7, 0.4]]]
    values, grads = acqf.value_and_grad(X)
    assert shape(grads) == shape(X)
    assert_close(values, acqf(X), rtol=1e-14)
    h = 1e-6
    for i in range(len(X[0])):
        for j in range(2):
            up = [[list(p) for p in X[0]]]
            down = [[list(p) for p in X[0]]]
            up[0][i][j] += h
            down[0][i][j] -= h
            fd = (flatten(acqf(up))[0] - flatten(acqf(down))[0]) / (2 * h)
            assert grads[0][i][j] == pytest.approx(fd, rel=1e-4, abs=1e-6)


@pytest.mark.skipif(not HAS_TORCH, reason="torch is not installed")
@pytest.mark.parametrize("name", MC)
def test_torch_autograd(name: str, model: SingleTaskGP) -> None:
    import torch

    acqf = make(name, model, "torch")
    X = torch.rand(3, 2, 2, dtype=torch.float64, generator=torch.Generator().manual_seed(0))
    X.requires_grad_(True)
    values = acqf(X)
    assert values.requires_grad and values.shape == (3,)
    (grad,) = torch.autograd.grad(values.sum(), X)
    _, expected = acqf.value_and_grad(X.detach())
    assert torch.allclose(grad, expected, rtol=1e-12, atol=0)
    with torch.no_grad():
        assert not acqf(X).requires_grad


def test_objectives(kind: str, model: SingleTaskGP) -> None:
    rng = random.Random(0)
    Y = [[[rng.gauss(0.0, 1.0) for _ in range(3)] for _ in range(2)] for _ in range(4)]
    Yk = as_kind(Y, kind)
    swapped = IdentityMCMultiOutputObjective(outcomes=[2, 0])(Yk)
    check_kind(swapped, kind)
    assert shape(swapped) == (4, 2, 2)
    assert flatten(swapped) == [v for s in Y for p in s for v in (p[2], p[0])]
    negative = IdentityMCMultiOutputObjective(outcomes=[-1, 0], num_outcomes=3)
    assert negative.outcomes == [2, 0]
    weighted = WeightedMCMultiOutputObjective(as_kind([2.0, -1.0, 0.5], kind))(Yk)
    check_kind(weighted, kind)
    assert_close(weighted, [[[2.0 * p[0], -p[1], 0.5 * p[2]] for p in s] for s in Y])
    assert IdentityMCMultiOutputObjective()(Yk) is Yk
    with pytest.raises(BotorchTensorDimensionError):
        IdentityMCMultiOutputObjective(outcomes=[0])
    with pytest.raises(BotorchError):
        IdentityMCMultiOutputObjective(outcomes=[0, -1])  # needs num_outcomes
    with pytest.raises(BotorchTensorDimensionError):
        WeightedMCMultiOutputObjective([1.0, 2.0], outcomes=[0, 1, 2])
    feas = FeasibilityWeightedMCMultiOutputObjective(model, X_TRAIN, constraint_idcs=[-1])
    assert feas.constraint_idcs == [1] and feas.objective_idcs == [0]
    assert shape(feas.infeasible_cost) == (1,)
    with pytest.raises(ValueError, match="duplicate"):
        FeasibilityWeightedMCMultiOutputObjective(model, X_TRAIN, constraint_idcs=[1, -1])


@pytest.mark.skipif(not HAS_TORCH, reason="torch is not installed")
def test_generic_and_feasibility_objectives_in_acquisition(model: SingleTaskGP) -> None:
    import torch

    def negate(Y: Any, X: Any = None) -> Any:
        return torch.stack([-Y[..., 1], -Y[..., 0]], dim=-1)

    swap = IdentityMCMultiOutputObjective(outcomes=[1, 0])
    generic = GenericMCMultiOutputObjective(negate)
    weighted = WeightedMCMultiOutputObjective([-1.0, -1.0], outcomes=[1, 0])
    ref = [-REF[1] + 3.0, -REF[0] + 3.0]  # below the negated outcomes
    a = qLogNoisyExpectedHypervolumeImprovement(
        model, ref, X_TRAIN, objective=generic, sampler=sampler()
    )
    b = qLogNoisyExpectedHypervolumeImprovement(
        model, ref, X_TRAIN, objective=weighted, sampler=sampler()
    )
    assert_close(a(SETS), b(SETS), rtol=1e-12, atol=1e-12)
    _, ga = a.value_and_grad(SETS)
    _, gb = b.value_and_grad(SETS)
    assert_close(ga, gb, rtol=1e-10, atol=1e-12)
    c = qLogNoisyExpectedHypervolumeImprovement(
        model, [REF[1], REF[0]], X_TRAIN, objective=swap, sampler=sampler()
    )
    assert all(math.isfinite(v) for v in flatten(c(SETS)))
    # A feasibility-weighted objective with the second output as constraint.
    feas = FeasibilityWeightedMCMultiOutputObjective(model, X_TRAIN, constraint_idcs=[1])
    with pytest.raises(UnsupportedError):
        feas(Y_TRAIN)  # torch tensors only
    out = feas(torch.tensor([[[0.5, 1.0], [0.5, -1.0]]], dtype=torch.float64))
    assert out.shape == (1, 2, 1)
    assert out[0, 0, 0].item() == pytest.approx(0.5)
    assert out[0, 1, 0].item() == pytest.approx(-float(feas.infeasible_cost[0]), abs=1e-6)


def test_prune_inferior_points_multi_objective(kind: str, model: SingleTaskGP) -> None:
    X = as_kind(X_TRAIN, kind)
    pruned = prune_inferior_points_multi_objective(model, X, as_kind(REF, kind), num_samples=256)
    check_kind(pruned, kind)
    kept = [list(x) for x in flatten_rows(pruned)]
    assert 1 <= len(kept) <= len(X_TRAIN) and all(x in X_TRAIN for x in kept)
    few = prune_inferior_points_multi_objective(model, X, REF, num_samples=256, max_frac=0.2)
    assert shape(few)[0] <= math.ceil(0.2 * len(X_TRAIN))
    # Explicit samplers (a botorch_rs extension) make the result reproducible.
    a = prune_inferior_points_multi_objective(model, X_TRAIN, REF, sampler=sampler(128, 1))
    b = prune_inferior_points_multi_objective(model, X_TRAIN, REF, sampler=sampler(128, 1))
    assert a == b


def flatten_rows(x: Any) -> list[list[float]]:
    return [[float(v) for v in row] for row in (x.tolist() if hasattr(x, "tolist") else x)]


def test_default_partitioning_alpha() -> None:
    assert get_default_partitioning_alpha(2) == 0.0
    assert get_default_partitioning_alpha(5) == 1e-3
    assert get_default_partitioning_alpha(6) == 1e-2
    with pytest.warns(BotorchWarning, match="less than 7 objectives"):
        assert get_default_partitioning_alpha(7) == 1e-2


def test_qlognparego_weights(model: SingleTaskGP) -> None:
    explicit = make("qLogNParEGO", model)
    assert_close(explicit.scalarization_weights, [0.3, 0.7], rtol=0, atol=0)
    assert shape(explicit.Y_baseline) == (12, 2)
    # Random weights on the simplex, reproducible for a given sampler seed.
    a = qLogNParEGO(model, X_TRAIN, sampler=sampler(64, 11))
    b = qLogNParEGO(model, X_TRAIN, sampler=sampler(64, 11))
    c = qLogNParEGO(model, X_TRAIN, sampler=sampler(64, 12))
    wa = flatten(a.scalarization_weights)
    assert sum(wa) == pytest.approx(1.0) and all(w >= 0 for w in wa)
    assert wa == flatten(b.scalarization_weights)
    assert wa != flatten(c.scalarization_weights)
    assert flatten(a(SETS)) == flatten(b(SETS))


def test_model_list(model: SingleTaskGP) -> None:
    """A ModelListGP of the two outputs gives the multi-output model's values
    (the outputs are independent and the hyperparameters are copied)."""
    subs = []
    for j in range(2):
        sub = SingleTaskGP(X_TRAIN, [[y[j]] for y in Y_TRAIN])
        params = {k: v for k, v in model.named_parameters()}
        sub.initialize(**{k: _element(v, j) for k, v in params.items()})
        subs.append(sub)
    lst = ModelListGP(*subs)
    for name in MC:
        a = flatten(make(name, model)(SETS))
        b = flatten(make(name, lst)(SETS))
        assert_close(a, b, rtol=1e-8, atol=1e-10)


def _element(value: Any, j: int) -> Any:
    """Element ``j`` of a batched (per-output) hyperparameter."""
    rows = value.tolist() if hasattr(value, "tolist") else value
    return rows[j]


def test_optimize_acqf_list(model: SingleTaskGP) -> None:
    acqfs = [make("qLogEHVI", model), make("qLogNParEGO", model)]
    bounds = [[0.0, 0.0], [1.0, 1.0]]
    candidates, values = optimize_acqf_list(
        acqfs, bounds, num_restarts=4, raw_samples=64, options={"seed": 0}
    )
    assert shape(candidates) == (2, 2) and shape(values) == (2,)
    assert all(0.0 <= v <= 1.0 for v in flatten(candidates))
    # Each value is conditional on the previous candidates.
    acqfs[1].set_X_pending([candidates[0]])
    assert flatten(acqfs[1]([[candidates[1]]]))[0] == pytest.approx(flatten(values)[1], rel=1e-9)


def test_results_do_not_depend_on_the_number_of_threads(model: SingleTaskGP) -> None:
    acqf = make("qLogNEHVI", model, sampler=sampler(256, 0))
    X = [[[0.1 * i, 1.0 - 0.1 * i], [0.05 * i, 0.5]] for i in range(10)]
    default = botorch_rs.get_num_threads()
    try:
        botorch_rs.set_num_threads(1)
        one = acqf.value_and_grad(X)
        botorch_rs.set_num_threads(4)
        four = acqf.value_and_grad(X)
    finally:
        botorch_rs.set_num_threads(default)
    assert flatten(one[0]) == flatten(four[0])
    assert flatten(one[1]) == flatten(four[1])


def test_legacy_functions_warn(model: SingleTaskGP) -> None:
    with pytest.warns(NumericsWarning, match="qLogExpectedHypervolumeImprovement"):
        make("qEHVI", model)
    with pytest.warns(NumericsWarning, match="qLogNoisyExpectedHypervolumeImprovement"):
        make("qNEHVI", model)
    with warnings.catch_warnings():
        warnings.simplefilter("error", NumericsWarning)
        make("qLogEHVI", model)
        make("qLogNEHVI", model)


def test_errors(model: SingleTaskGP) -> None:
    with pytest.raises(ValueError, match="length of the reference point"):
        qExpectedHypervolumeImprovement(model, [0.0, 0.0, 0.0], partitioning("list"))
    with pytest.raises(UnsupportedError, match="MCMultiOutputObjective"):
        make("qLogNEHVI", model, objective=lambda Y, X=None: Y.sum(-1))
    with pytest.raises(ValueError, match="m>=2"):
        qNoisyExpectedHypervolumeImprovement(model, [0.0], X_TRAIN)
    with pytest.raises(UnsupportedError):
        make("qLogEHVI", model, fat=[True])  # per-constraint fat flags
    with pytest.raises(UnsupportedError):
        qLogNoisyExpectedHypervolumeImprovement(model, REF, X_TRAIN, marginalize_dim=0)
    with pytest.raises(UnsupportedError):
        ExpectedHypervolumeImprovement(
            model, REF, partitioning("list"), posterior_transform=object()
        )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(BotorchTensorDimensionError):
            make("qNEHVI", model)(as_kind([[[0.1, 0.2, 0.3]]], "list"))

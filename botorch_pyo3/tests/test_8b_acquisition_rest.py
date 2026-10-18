"""Phase 8b through the public API, independently of BoTorch: the input
constructors and ``get_acquisition_function`` (result types for every array
type, BoTorch's errors), ``qMultiStepLookahead`` (shapes, the tree layout,
consistency over t-batches, gradients against finite differences, errors),
Python subclasses of ``SampleReducingMCAcquisitionFunction``, t-batched
fixed values and batched max-value samplers. The comparisons with BoTorch are
in ``test_parity_8b_acquisition_rest.py`` and
``test_equivalence_8b_acquisition_rest.py``.
"""

from __future__ import annotations

import functools
import math
import random
import warnings
from typing import Any

import pytest

from botorch_rs.acquisition import (
    ExpectedImprovement,
    FixedFeatureAcquisitionFunction,
    LogImprovementMCAcquisitionFunction,
    PosteriorMean,
    SampleReducingMCAcquisitionFunction,
    UpperConfidenceBound,
    get_acqf_input_constructor,
    get_acquisition_function,
    qExpectedImprovement,
    qKnowledgeGradient,
    qLogNoisyExpectedImprovement,
    qMaxValueEntropy,
    qMultiStepLookahead,
    qNegIntegratedPosteriorVariance,
    qNoisyExpectedImprovement,
    qSimpleRegret,
)
from botorch_rs.acquisition import input_constructors as ic
from botorch_rs.acquisition import utils as autils
from botorch_rs.acquisition.max_value_entropy_search import (
    _sample_max_value_Gumbel,
    _sample_max_value_Thompson,
)
from botorch_rs.acquisition.multi_step_lookahead import (
    _construct_inner_samplers,
    make_best_f,
    warmstart_multistep,
)
from botorch_rs.acquisition.objective import (
    ConstrainedMCObjective,
    IdentityMCObjective,
    LinearMCObjective,
    ScalarizedPosteriorTransform,
)
from botorch_rs.exceptions import UnsupportedError
from botorch_rs.exceptions.errors import BotorchError, DeprecationError
from botorch_rs.exceptions.warnings import BotorchWarning
from botorch_rs.models import ModelListGP, SingleTaskGP
from botorch_rs.optim import optimize_acqf
from botorch_rs.sampling import SobolQMCNormalSampler
from botorch_rs.utils.datasets import SupervisedDataset
from helpers import HAS_TORCH, as_kind, check_kind, flatten, make_data, shape, to_list

needs_torch = pytest.mark.skipif(not HAS_TORCH, reason="requires torch")


@functools.cache
def _data(seed: int = 0, n: int = 10, d: int = 2) -> tuple[list, list]:
    return make_data(n=n, d=d, seed=seed)


def _model(kind: str, seed: int = 0, m: int = 1) -> tuple[Any, Any, Any]:
    X, Y = _data(seed)
    if m == 2:
        Y = [[y[0], math.cos(3.0 * x[0]) + x[1]] for x, y in zip(X, Y, strict=True)]
    Xk, Yk = as_kind(X, kind), as_kind(Y, kind)
    return SingleTaskGP(Xk, Yk), Xk, Yk


def _dataset(X: Any, Y: Any) -> SupervisedDataset:
    names = [f"x{i}" for i in range(shape(X)[-1])]
    return SupervisedDataset(
        X, Y, feature_names=names, outcome_names=[f"y{i}" for i in range(shape(Y)[-1])]
    )


def _points(n: int, q: int, d: int = 2, seed: int = 1) -> list:
    rng = random.Random(seed)
    return [[[rng.random() for _ in range(d)] for _ in range(q)] for _ in range(n)]


def _fd_check(acqf: Any, X: list, h: float = 1e-6, tol: float = 1e-6) -> None:
    """``value_and_grad`` against central differences of the values."""
    _, g = acqf.value_and_grad(X)
    g = flatten(g)
    flat = flatten(X)
    shp = shape(X)
    for k in range(len(flat)):
        up = list(flat)
        dn = list(flat)
        up[k] += h
        dn[k] -= h
        fd = (sum(flatten(acqf(_unflat(up, shp)))) - sum(flatten(acqf(_unflat(dn, shp))))) / (2 * h)
        assert abs(fd - g[k]) <= tol * max(1.0, abs(fd)), (k, g[k], fd)


def _unflat(data: list, shp: tuple) -> Any:
    if len(shp) == 1:
        return list(data)
    step = len(data) // shp[0]
    return [_unflat(data[i * step : (i + 1) * step], shp[1:]) for i in range(shp[0])]


# ---------------------------------------------------------------------------
# Registry and helpers of the input constructors
# ---------------------------------------------------------------------------


def test_registry_and_errors() -> None:
    assert len(ic.ACQF_INPUT_CONSTRUCTOR_REGISTRY) == 33
    for cls in ic.ACQF_INPUT_CONSTRUCTOR_REGISTRY:
        assert callable(get_acqf_input_constructor(cls))

    class Dummy(ExpectedImprovement):
        pass

    with pytest.raises(RuntimeError, match="not registered"):
        get_acqf_input_constructor(Dummy)
    with pytest.raises(ValueError, match="duplicate"):
        ic.acqf_input_constructor(ExpectedImprovement)(lambda x: x)
    fn = lambda **kw: {"x": 1}  # noqa: E731
    ic._register_acqf_input_constructor(Dummy, fn)
    try:
        assert get_acqf_input_constructor(Dummy) is fn
    finally:
        ic.ACQF_INPUT_CONSTRUCTOR_REGISTRY.pop(Dummy)


def test_allow_only_specific_variable_kwargs() -> None:
    model, X, Y = _model("list")
    c = get_acqf_input_constructor(ExpectedImprovement)
    with pytest.raises(TypeError, match="Unexpected keyword argument `hat`"):
        c(model=model, training_data=_dataset(X, Y), hat="car")
    # Allowed extras are dropped (e.g. objective for analytic functions).
    out = get_acqf_input_constructor(PosteriorMean)(
        model=model, training_data=_dataset(X, Y), objective=None, bounds=[(0, 1)]
    )
    assert set(out) == {"model", "posterior_transform"}
    # Constructors with **kwargs receive every keyword argument (the
    # ``optimize_objective_kwargs`` of the knowledge gradient).
    out = get_acqf_input_constructor(qKnowledgeGradient)(
        model=model,
        training_data=_dataset(X, Y),
        bounds=[(0.0, 1.0)] * 2,
        num_fantasies=2,
        with_current_value=True,
        optimizer_options={"num_restarts": 2, "raw_samples": 8},
        batch_initial_conditions=_points(2, 1),
    )
    assert isinstance(out["current_value"], float)


def test_field_is_shared_and_block_designs(kind: str) -> None:
    X, Y = _data(0)
    X2, Y2 = _data(1)
    blocks = {
        0: _dataset(as_kind(X, kind), as_kind(Y, kind)),
        1: _dataset(as_kind(X, kind), as_kind(Y2, kind)),
    }
    assert ic._field_is_shared(blocks, "X")
    assert not ic._field_is_shared(blocks, "Y")
    with pytest.raises(AttributeError, match="has no field"):
        ic._field_is_shared(blocks, "foo")
    multi = {0: blocks[0], 1: _dataset(as_kind(X2, kind), as_kind(Y2, kind))}
    with pytest.raises(NotImplementedError, match="block designs"):
        ic.get_best_f_analytic(multi)
    with pytest.raises(NotImplementedError, match="block designs"):
        ic.get_best_f_mc(multi)


def test_get_best_f(kind: str) -> None:
    X, Y = _data(0)
    ds = _dataset(as_kind(X, kind), as_kind(Y, kind))
    best = ic.get_best_f_analytic(ds)
    check_kind(best, kind)
    assert float(to_list(best)) == max(y[0] for y in Y)
    assert float(to_list(ic.get_best_f_mc(ds))) == max(y[0] for y in Y)
    Y2 = [[y[0], -2.0 * y[0] + 1.0] for y in Y]
    blocks = {
        0: _dataset(as_kind(X, kind), as_kind([[y[0]] for y in Y2], kind)),
        1: _dataset(as_kind(X, kind), as_kind([[y[1]] for y in Y2], kind)),
    }
    with pytest.raises(NotImplementedError, match="ScalarizedObjective"):
        ic.get_best_f_analytic(blocks)
    with pytest.raises(UnsupportedError, match="require an objective"):
        ic.get_best_f_mc(blocks)
    tf = ScalarizedPosteriorTransform(as_kind([0.5, 0.25], kind))
    expected = max(0.5 * a + 0.25 * b for a, b in Y2)
    assert (
        abs(float(to_list(ic.get_best_f_analytic(blocks, posterior_transform=tf))) - expected)
        < 1e-15
    )
    assert abs(float(to_list(ic.get_best_f_mc(blocks, posterior_transform=tf))) - expected) < 1e-15
    if kind != "list":
        obj = LinearMCObjective(as_kind([0.5, 0.25], kind))
        assert abs(float(to_list(ic.get_best_f_mc(blocks, objective=obj))) - expected) < 1e-15


def test_constructors_without_torch() -> None:
    """The constructors that need no callables work with nested lists."""
    model, X, Y = _model("list")
    ds = _dataset(X, Y)
    out = get_acqf_input_constructor(ExpectedImprovement)(model=model, training_data=ds)
    assert out["best_f"] == max(y[0] for y in Y)
    ExpectedImprovement(**out)
    out = get_acqf_input_constructor(qNoisyExpectedImprovement)(model=model, training_data=ds)
    assert out["X_baseline"] is X
    out = get_acqf_input_constructor(qMaxValueEntropy)(
        model=model, training_data=ds, bounds=[(0.0, 1.0), (2.0, 3.0)], candidate_size=20
    )
    cs = out["candidate_set"]
    assert shape(cs) == (20, 2)
    assert all(0.0 <= r[0] <= 1.0 and 2.0 <= r[1] <= 3.0 for r in cs)
    qMaxValueEntropy(**out)
    out = get_acqf_input_constructor(qNegIntegratedPosteriorVariance)(
        model=model, bounds=[(0.0, 1.0)] * 2, num_mc_points=8
    )
    assert shape(out["mc_points"]) == (8, 2)
    qNegIntegratedPosteriorVariance(**out)


def test_multi_fidelity_and_reference_point_errors() -> None:
    with pytest.raises(RuntimeError, match="same indices"):
        ic.construct_inputs_mf_base({0: 1.0}, fidelity_weights={1: 1.0})
    out = ic.construct_inputs_mf_base({1: 1.0}, num_trace_observations=1)
    X = [[[0.2, 0.3]]]
    assert out["project"](X) == [[[0.2, 1.0]]]
    assert shape(out["expand"](X)) == (1, 2, 2)
    with pytest.raises(ValueError, match="Cannot specify both"):
        ic._get_ref_point(objective_thresholds=[0.0], ref_point=[0.0])
    with pytest.raises(ValueError, match="Either `ref_point`"):
        ic._get_ref_point()
    with pytest.warns(DeprecationWarning, match="objective_thresholds"):
        assert ic._get_ref_point(objective_thresholds=[1.0, 2.0]) == [1.0, 2.0]
    with pytest.raises(BotorchError, match="single outcomes"):
        ic._construct_constraint_dict_from_tuple(([[1.0, 1.0]], [[0.0]]), ExpectedImprovement)
    assert ic._construct_constraint_dict_from_tuple(
        ([[0.0, -2.0]], [[1.0]]), ExpectedImprovement
    ) == {1: (-0.5, None)}
    with pytest.raises(UnsupportedError, match="IIDNormalSampler"):
        ic._get_sampler(mc_samples=8, qmc=False)
    model, X, Y = _model("list", m=2)
    with pytest.raises(NotImplementedError, match="outcome constraints"):
        ic.construct_inputs_EHVI(
            model, _dataset(X, Y), ref_point=[0.0, 0.0], constraints=[lambda Z: Z[..., 0]]
        )


@needs_torch
def test_constraint_errors_of_the_mc_constructors() -> None:
    model, X, Y = _model("torch", m=2)
    for cls in (qSimpleRegret,):
        with pytest.raises(ValueError, match="Constraints require an X_baseline"):
            get_acqf_input_constructor(cls)(model=model, constraints=[lambda Z: Z[..., 0]])
    with pytest.raises(NotImplementedError, match="Trace observations"):
        ic.construct_inputs_qMFHVKG(
            model,
            _dataset(X, Y),
            bounds=[(0.0, 1.0)] * 2,
            target_fidelities={1: 1.0},
            ref_point=[0.0, 0.0],
            num_trace_observations=1,
        )


# ---------------------------------------------------------------------------
# get_acquisition_function
# ---------------------------------------------------------------------------


@needs_torch
def test_get_acquisition_function_errors_and_types() -> None:
    import torch

    model, X, Y = _model("torch")
    obj = IdentityMCObjective()
    with pytest.raises(NotImplementedError, match="Unknown acquisition function foo"):
        get_acquisition_function("foo", model, obj, X)
    with pytest.raises(ValueError, match="`beta` must be not be None"):
        get_acquisition_function("qUCB", model, obj, X)
    for name, cls in (
        ("qEI", qExpectedImprovement),
        ("qSR", qSimpleRegret),
        ("qLogNEI", qLogNoisyExpectedImprovement),
    ):
        acqf = get_acquisition_function(name, model, obj, X, mc_samples=8, seed=1)
        assert isinstance(acqf, cls)
        assert acqf.sampler.seed == 1 and acqf.sampler.num_samples == 8
    acqf = get_acquisition_function("qEI", model, obj, X)
    assert abs(float(acqf.best_f) - model.posterior(X).mean.max().item()) < 1e-15
    m2, X2, Y2 = _model("torch", m=2)
    mo = __import__(
        "botorch_rs.acquisition.multi_objective.objective", fromlist=["x"]
    ).IdentityMCMultiOutputObjective()
    with pytest.raises(ValueError, match="`Y` must not be None"):
        get_acquisition_function("qEHVI", m2, mo, X2, ref_point=[0.0, 0.0])
    with pytest.raises(ValueError, match="`ref_point` must not be None"):
        get_acquisition_function("qEHVI", m2, mo, X2, Y=Y2)
    with pytest.raises(ValueError, match="`ref_point` must not be None"):
        get_acquisition_function("qNEHVI", m2, mo, X2)
    tf = ScalarizedPosteriorTransform(torch.tensor([0.5, 0.5], dtype=torch.float64))
    with pytest.raises(NotImplementedError, match="PosteriorTransforms are not yet"):
        get_acquisition_function("qNEHVI", m2, mo, X2, posterior_transform=tf, ref_point=[0, 0])
    with pytest.raises(DeprecationError, match="factory"):
        autils.get_acquisition_function("qEI", model, obj, X)


# ---------------------------------------------------------------------------
# qMultiStepLookahead
# ---------------------------------------------------------------------------


def test_lookahead_shapes_and_tree_layout(kind: str) -> None:
    model, X, _ = _model(kind)
    acqf = qMultiStepLookahead(model, batch_sizes=[2, 1], num_fantasies=[2, 3])
    assert acqf._num_auxiliary == 2 * 2 + 1 * 6
    q_aug = acqf.get_augmented_q_batch_size(2)
    assert q_aug == 2 + 10
    Xt = as_kind(_points(3, q_aug), kind)
    v = acqf(Xt)
    check_kind(v, kind)
    assert shape(v) == (3,)
    batch, shapes, sizes = acqf.get_split_shapes(Xt)
    assert tuple(batch) == (3,)
    assert [tuple(s) for s in shapes] == [(3, 2, 2), (2, 3, 2, 2), (3, 2, 3, 1, 2)]
    assert list(sizes) == [2, 4, 6]
    reps = acqf.get_multi_step_tree_input_representation(Xt)
    assert [shape(r) for r in reps] == [tuple(s) for s in shapes]
    pts = to_list(Xt)
    r1 = to_list(reps[1])
    r2 = to_list(reps[2])
    for b in range(3):
        for i1 in range(2):
            # Node i1 at depth 1: rows q + 2 i1 of every candidate set.
            assert r1[i1][b] == pts[b][2 + 2 * i1 : 4 + 2 * i1]
            for i2 in range(3):
                # Node (i1, i2) at depth 2: row q + 4 + (i1 + 2 i2).
                assert r2[i2][i1][b] == pts[b][6 + i1 + 2 * i2 : 7 + i1 + 2 * i2]
    cand = acqf.extract_candidates(Xt)
    check_kind(cand, kind)
    assert to_list(cand) == [p[:2] for p in pts]


def test_lookahead_is_consistent_over_t_batches() -> None:
    model, X, _ = _model("list")
    for collapse in (True, False):
        acqf = qMultiStepLookahead(
            model, batch_sizes=[2, 1], num_fantasies=[2, 2], collapse_fantasy_base_samples=collapse
        )
        q_aug = acqf.get_augmented_q_batch_size(1)
        Xt = _points(4, q_aug, seed=3)
        many = acqf(Xt)
        for b in range(4):
            assert abs(acqf([Xt[b]])[0] - many[b]) <= 1e-14 * max(1.0, abs(many[b]))


def test_lookahead_gradients_match_finite_differences() -> None:
    model, X, _ = _model("list")
    stages = [qExpectedImprovement, None, PosteriorMean]
    acqf = qMultiStepLookahead(
        model,
        batch_sizes=[2, 1],
        num_fantasies=[2, 2],
        valfunc_cls=stages,
        valfunc_argfacs=[make_best_f, None, None],
        inner_mc_samples=[8, None, None],
        X_pending=[[0.4, 0.6]],
    )
    _fd_check(acqf, _points(2, acqf.get_augmented_q_batch_size(1), seed=4))


def test_lookahead_errors_and_warnings() -> None:
    model, X, _ = _model("list")
    with pytest.raises(UnsupportedError, match="exactly one of"):
        qMultiStepLookahead(model, batch_sizes=[1])
    with pytest.raises(UnsupportedError, match="exactly one of"):
        qMultiStepLookahead(
            model,
            batch_sizes=[1],
            num_fantasies=[2],
            samplers=[SobolQMCNormalSampler(sample_shape=(2,))],
        )
    with pytest.raises(UnsupportedError, match="nice try"):
        qMultiStepLookahead(
            model, batch_sizes=[1], num_fantasies=[2], valfunc_cls=[qMultiStepLookahead] * 2
        )
    with pytest.raises(UnsupportedError, match="non-MC `objective`"):
        qMultiStepLookahead(model, batch_sizes=[1], num_fantasies=[2], objective="cat")
    with pytest.raises(UnsupportedError, match="Only PosteriorTransforms"):
        qMultiStepLookahead(
            model,
            batch_sizes=[1],
            num_fantasies=[2],
            objective=IdentityMCObjective(),
            valfunc_cls=[ExpectedImprovement] * 2,
            valfunc_argfacs=[make_best_f] * 2,
        )
    with pytest.raises(UnsupportedError, match="q=1"):
        qMultiStepLookahead(
            model,
            batch_sizes=[2],
            num_fantasies=[2],
            valfunc_cls=[None, PosteriorMean],
        )
    with pytest.warns(BotorchWarning, match="inner_mc_samples is ignored"):
        qMultiStepLookahead(
            model,
            batch_sizes=[1],
            num_fantasies=[2],
            valfunc_cls=[ExpectedImprovement] * 2,
            valfunc_argfacs=[make_best_f] * 2,
            inner_mc_samples=[2, 2],
        )
    with pytest.raises(UnsupportedError, match="terminal value function"):
        qMultiStepLookahead(
            model, batch_sizes=[1], num_fantasies=[2], valfunc_cls=[PosteriorMean, None]
        )
    with pytest.raises(UnsupportedError, match="not supported by botorch_rs"):
        qMultiStepLookahead(
            model, batch_sizes=[1], num_fantasies=[2], valfunc_cls=[None, qNoisyExpectedImprovement]
        )
    m2, X2, _ = _model("list", m=2)
    tf = ScalarizedPosteriorTransform([0.5, 0.5])
    with pytest.raises(UnsupportedError, match="single-output"):
        qMultiStepLookahead(
            m2,
            batch_sizes=[1],
            num_fantasies=[2],
            posterior_transform=tf,
            valfunc_cls=[None, ExpectedImprovement],
            valfunc_argfacs=[None, make_best_f],
        )
    acqf = qMultiStepLookahead(model, batch_sizes=[1], num_fantasies=[3])
    with pytest.raises(Exception, match="needs more than 3 points"):
        acqf(_points(1, 3))
    # Inner samplers: 32 samples by default, BoTorch's checks.
    inner = _construct_inner_samplers([1], [qSimpleRegret, qSimpleRegret], [None, 4])
    assert [s.num_samples for s in inner if s is not None] == [32, 4]


def test_lookahead_argument_factories() -> None:
    model, X, Y = _model("list")
    calls: list[Any] = []

    def argfac(model: Any, X: Any) -> dict:
        calls.append(X)
        return {"beta": 0.5}

    acqf = qMultiStepLookahead(
        model,
        batch_sizes=[1],
        num_fantasies=[2],
        valfunc_cls=[UpperConfidenceBound, PosteriorMean],
        valfunc_argfacs=[argfac, None],
    )
    # Called once, with the current model and ``X=None``.
    assert calls == [None]
    acqf(_points(1, 3))
    assert calls == [None]
    # make_best_f: the best observed value (no outcome transform: the
    # targets themselves).
    assert make_best_f(model, X)["best_f"] == max(y[0] for y in Y)


def test_lookahead_pending_points_and_optimize_acqf() -> None:
    model, X, _ = _model("list")
    acqf = qMultiStepLookahead(model, batch_sizes=[1], num_fantasies=[2])
    q_aug = acqf.get_augmented_q_batch_size(1)
    Xt = _points(1, q_aug, seed=5)
    v0 = acqf(Xt)[0]
    acqf.set_X_pending([[0.2, 0.8]])
    assert acqf.X_pending == [[0.2, 0.8]]
    assert acqf(Xt)[0] != v0
    acqf.set_X_pending(None)
    assert acqf(Xt)[0] == v0
    bounds = [[0.0, 0.0], [1.0, 1.0]]
    # BoTorch's convention: optimize_acqf is called with the augmented size.
    cand, val = optimize_acqf(acqf, bounds, q=q_aug, num_restarts=2, raw_samples=16)
    assert shape(cand) == (1, 2)
    full, vals = optimize_acqf(
        acqf,
        bounds,
        q=q_aug,
        num_restarts=2,
        raw_samples=16,
        return_best_only=False,
        return_full_tree=True,
    )
    assert shape(full) == (2, q_aug, 2)
    init = warmstart_multistep(acqf, bounds, num_restarts=2, raw_samples=4, full_optimizer=full)
    assert shape(init) == (2, q_aug, 2)


@needs_torch
def test_lookahead_weights_and_induced_fantasy_model() -> None:
    import torch

    model, X, _ = _model("torch")
    samplers = [SobolQMCNormalSampler(sample_shape=(2,)), SobolQMCNormalSampler(sample_shape=(3,))]
    samplers[0].base_weights = torch.tensor([1.0, 3.0], dtype=torch.float64)
    acqf = qMultiStepLookahead(model, batch_sizes=[1, 1], samplers=samplers)
    q_aug = acqf.get_augmented_q_batch_size(1)
    Xt = torch.rand(2, q_aug, 2, dtype=torch.float64)
    fm = acqf.get_induced_fantasy_model(Xt)
    reps = acqf.get_multi_step_tree_input_representation(Xt)
    leaf = fm.posterior(reps[-1]).mean.squeeze(-1).squeeze(-1)
    # The value is the weighted average of the leaves' posterior means.
    w = torch.tensor([0.25, 0.75], dtype=torch.float64)
    expected = (leaf.mean(0) * w[:, None]).sum(0)
    assert torch.allclose(acqf(Xt), expected, rtol=1e-12, atol=1e-14)


# ---------------------------------------------------------------------------
# Python subclasses of the sample-reducing acquisition functions
# ---------------------------------------------------------------------------


@needs_torch
def test_sample_reducing_subclass(kind: str) -> None:
    import torch

    model, X, _ = _model(kind)

    class Mean(SampleReducingMCAcquisitionFunction):
        def _sample_forward(self, obj: Any) -> Any:
            return obj

    acqf = Mean(model, sampler=SobolQMCNormalSampler(sample_shape=(64,)))
    Xt = as_kind(_points(3, 2), kind)
    v = acqf(Xt)
    check_kind(v, kind)
    assert shape(v) == (3,)
    # q-reduction amax, sample reduction mean: the mean of the per-sample max.
    native = qSimpleRegret(model, sampler=acqf.sampler)
    for a, b in zip(flatten(v), flatten(native(Xt)), strict=True):
        assert abs(a - b) <= 1e-12 * max(1.0, abs(b))
    _fd_check(Mean(model, sampler=acqf.sampler), _points(2, 2))
    # optimize_acqf through torch.
    cand, _ = optimize_acqf(acqf, [[0.0, 0.0], [1.0, 1.0]], q=2, num_restarts=2, raw_samples=8)
    assert shape(cand) == (2, 2)
    del torch


@needs_torch
def test_sample_reducing_errors_and_native_classes() -> None:
    import torch

    model, X, _ = _model("torch", m=2)

    class Neg(SampleReducingMCAcquisitionFunction):
        def _sample_forward(self, obj: Any) -> Any:
            return obj - 100.0

    cons = [lambda Z: Z[..., 1]]
    obj = LinearMCObjective(torch.tensor([1.0, 0.0], dtype=torch.float64))
    with pytest.raises(ValueError, match="ConstrainedMCObjective as well as constraints"):
        Neg(
            model,
            objective=ConstrainedMCObjective(lambda Y, X=None: Y[..., 0], cons),
            constraints=cons,
        )
    acqf = Neg(model, objective=obj, constraints=cons)
    with pytest.raises(ValueError, match="non-negative"):
        acqf(torch.rand(1, 1, 2, dtype=torch.float64))

    class Log(LogImprovementMCAcquisitionFunction):
        def _sample_forward(self, obj: Any) -> Any:
            return obj

    with pytest.raises(BotorchError, match="Log-Improvement"):
        Log(model, objective=ConstrainedMCObjective(lambda Y, X=None: Y[..., 0], cons))
    # Pending points are appended to every candidate set.
    p = torch.tensor([[0.5, 0.5]], dtype=torch.float64)
    acqf = Neg(model, objective=obj, X_pending=p)
    assert torch.equal(acqf.X_pending, p)
    acqf.set_X_pending(None)
    assert acqf.X_pending is None
    # The native acquisition functions derive from the base classes.
    m1, X1, _ = _model("list")
    assert isinstance(qExpectedImprovement(m1, best_f=0.0), SampleReducingMCAcquisitionFunction)
    assert isinstance(qLogNoisyExpectedImprovement(m1, X1), LogImprovementMCAcquisitionFunction)


# ---------------------------------------------------------------------------
# t-batched fixed values and batched max-value samplers
# ---------------------------------------------------------------------------


def test_fixed_feature_with_t_batched_values(kind: str) -> None:
    model, X, _ = _model(kind)
    base = PosteriorMean(model)
    values = as_kind([[[0.1]], [[0.4]], [[0.9]]], kind)  # 3 x 1 x 1
    acqf = FixedFeatureAcquisitionFunction(base, 2, [1], values)
    Xt = as_kind([[[0.2]], [[0.3]], [[0.5]]], kind)
    full = acqf._construct_X_full(Xt)
    check_kind(full, kind)
    assert to_list(full) == [[[0.2, 0.1]], [[0.3, 0.4]], [[0.5, 0.9]]]
    v = acqf(Xt)
    check_kind(v, kind)
    expected = flatten(base(full))
    for a, b in zip(flatten(v), expected, strict=True):
        assert a == b
    with pytest.raises(ValueError, match="Feature dimension"):
        acqf._construct_X_full(as_kind([[[0.2, 0.3]]] * 3, kind))


def test_max_value_samplers_with_batched_candidate_sets(kind: str) -> None:
    model, X, _ = _model(kind)
    cs = as_kind(_points(6, 5, seed=7), kind)  # 6 x 5 x 2
    g = _sample_max_value_Gumbel(model, cs, 4, seed=3)
    check_kind(g, kind)
    assert shape(g) == (4, 6)
    # Every batch as one unbatched call (shared uniform numbers).
    pts = _points(6, 5, seed=7)
    for b in range(6):
        single = _sample_max_value_Gumbel(model, as_kind(pts[b], kind), 4, seed=3)
        assert [r[0] for r in to_list(single)] == [r[b] for r in to_list(g)]
    cs2 = as_kind([_points(3, 5, seed=8)] * 2, kind)  # 2 x 3 x 5 x 2
    assert shape(_sample_max_value_Thompson(model, cs2, 4, seed=1)) == (4, 2, 3)
    with pytest.raises(ValueError, match="base samples"):
        _sample_max_value_Thompson(model, cs2, 4, base_samples=[[0.0] * 5] * 4)
    with pytest.raises(ValueError, match="columns"):
        _sample_max_value_Gumbel(model, as_kind([[[0.1, 0.2, 0.3]]], kind), 2)
    # The acquisition functions keep requiring `n x d` candidate sets.
    with pytest.raises(UnsupportedError, match="batched candidate sets"):
        qMaxValueEntropy(model, cs)


def test_acquisition_utils(kind: str) -> None:
    t = as_kind([[1.0, 2.0], [3.0, 4.0]], kind)
    r = as_kind([0.0] * 6, kind)
    out = autils.repeat_to_match_aug_dim(t, r)
    check_kind(out, kind)
    assert to_list(out) == [[1.0, 2.0], [3.0, 4.0]] * 3
    with pytest.raises(ValueError, match="multiple"):
        autils.repeat_to_match_aug_dim(t, as_kind([0.0] * 3, kind))
    X = as_kind([[[0.1, 0.2, 0.3]]], kind)
    sp = as_kind([[0.5], [0.7]], kind)
    out = autils.project_to_sample_points(X, sp)
    assert to_list(out) == [[[0.1, 0.2, 0.5], [0.1, 0.2, 0.7]]]
    obj = as_kind([[1.0, 3.0, 2.0], [0.0, -1.0, 4.0]], kind)
    best = autils.compute_best_feasible_objective(obj, obj, None)
    assert to_list(best) == [3.0, 4.0]


def test_lookahead_rejects_models_without_fantasies() -> None:
    model, X, Y = _model("list")
    ml = ModelListGP(model, SingleTaskGP(*_data(2)))
    acqf = qMultiStepLookahead(
        ml,
        batch_sizes=[1],
        num_fantasies=[2],
        posterior_transform=ScalarizedPosteriorTransform([0.5, 0.5]),
    )
    assert shape(acqf(_points(1, 3))) == (1,)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(UnsupportedError):
            qMultiStepLookahead(ml, batch_sizes=[1], num_fantasies=[2])

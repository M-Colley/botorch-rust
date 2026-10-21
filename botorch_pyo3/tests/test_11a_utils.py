"""API tests for the utilities of phase 11a on every array type (nested
lists, NumPy, torch) and their error messages, against independent references
(``math``, closed forms, brute force); BoTorch is not needed."""

from __future__ import annotations

import math
import random
import subprocess
import sys
from typing import Any

import pytest

import botorch_rs.utils.probability.utils as pu
import botorch_rs.utils.safe_math as sm
from botorch_rs.utils.probability.bvn import bvn, bvnmom, bvnu
from helpers import HAS_NUMPY, HAS_TORCH, KINDS, as_kind, check_kind, to_list

TOL = 1e-12


def close(a: Any, b: Any, tol: float = TOL) -> None:
    a, b = to_list(a), to_list(b)
    if isinstance(b, list):
        assert isinstance(a, list) and len(a) == len(b), (a, b)
        for x, y in zip(a, b, strict=True):
            close(x, y, tol)
        return
    if math.isnan(b):
        assert math.isnan(a)
    elif math.isinf(b):
        assert a == b, (a, b)
    else:
        assert abs(a - b) <= tol * max(1.0, abs(b)), (a, b)


def phi(x: float) -> float:
    return math.exp(-0.5 * x * x) / math.sqrt(2 * math.pi)


def Phi(x: float) -> float:
    return 0.5 * math.erfc(-x / math.sqrt(2))


# ---------------------------------------------------------------------------
# safe_math and the standard-normal helpers
# ---------------------------------------------------------------------------

XS = [-40.0, -5.0, -1.0, -1e-3, 0.0, 0.5, 3.0, 30.0]


@pytest.mark.parametrize("kind", KINDS)
def test_standard_normal_helpers(kind: str) -> None:
    x = as_kind(XS, kind)
    for f, ref in (
        (pu.ndtr, Phi),
        (pu.phi, phi),
        (pu.log_phi, lambda v: -0.5 * v * v - 0.5 * math.log(2 * math.pi)),
        (pu.log_ndtr, lambda v: math.log(Phi(v)) if Phi(v) > 1e-300 else None),
        (pu.log_erfc, lambda v: math.log(math.erfc(v)) if math.erfc(v) > 1e-300 else None),
    ):
        out = f(x)
        check_kind(out, kind)
        for v, o in zip(XS, to_list(out), strict=True):
            r = ref(v)
            if r is not None and abs(r) < 700:
                close(o, r, 1e-13)
    # log_ndtr far in the tail: the asymptotic expansion.
    v = to_list(pu.log_ndtr(as_kind([-40.0], kind)))[0]
    close(
        v,
        -0.5 * 1600 - math.log(40 * math.sqrt(2 * math.pi)) + math.log1p(-1 / 1600 + 3 / 1600**2),
        1e-10,
    )
    close(
        pu.log_erfcx(as_kind([0.0, 10.0], kind)),
        [0.0, math.log(math.exp(100) * math.erfc(10))],
        1e-12,
    )


@pytest.mark.parametrize("kind", KINDS)
def test_safe_math_kinds(kind: str) -> None:
    x = as_kind([-3.0, -0.5, 0.0, 2.0], kind)
    out = sm.log1pexp(x)
    check_kind(out, kind)
    close(out, [math.log1p(math.exp(v)) for v in [-3.0, -0.5, 0.0, 2.0]])
    close(
        sm.log1mexp(as_kind([-1e-10, -1.0, -50.0], kind)),
        [math.log(-math.expm1(-1e-10)), math.log(-math.expm1(-1.0)), math.log1p(-math.exp(-50.0))],
    )
    close(
        sm.logmeanexp(as_kind([[0.0, 1.0], [2.0, 2.0]], kind), dim=-1),
        [math.log((1 + math.e) / 2), 2.0],
    )
    close(sm.logsumexp(as_kind([[0.0, -math.inf]], kind), dim=-1), [0.0])
    # fatplus(x) = tau (softplus(x / tau) + alpha cauchy(x / tau)), alpha = 0.1.
    close(sm.fatplus(as_kind([0.0], kind), tau=1.0), [math.log(2.0) + 0.1])
    close(sm.smooth_amax(as_kind([[1.0, 1.0]], kind), dim=-1, tau=0.5), [1.0 + 0.5 * math.log(2)])
    close(
        sm.logdiffexp(as_kind([0.0], kind), as_kind([1.0], kind)),
        [1.0 + math.log(-math.expm1(-1.0))],
    )
    with pytest.raises(IndexError):
        sm.logsumexp(as_kind([[0.0, 1.0]], kind), dim=5)


def test_safe_math_float32_numpy() -> None:
    np = pytest.importorskip("numpy")
    x = np.asarray([0.1, 2.0, -3.0], dtype=np.float32)
    out = sm.exp(x)
    assert out.dtype == np.float32
    assert np.allclose(out, np.exp(x), rtol=1e-6)


@pytest.mark.parametrize("kind", KINDS)
def test_log_prob_normal_in_and_feasibility(kind: str) -> None:
    a, b = [-1.0, -50.0, 3.0, -math.inf], [1.0, -49.0, 4.0, 0.0]
    out = pu.log_prob_normal_in(as_kind(a, kind), as_kind(b, kind))
    check_kind(out, kind)
    ref = [math.log(Phi(1) - Phi(-1)), None, math.log(Phi(-3) - Phi(-4)), math.log(0.5)]
    for o, r in zip(to_list(out), ref, strict=True):
        if r is not None:
            close(o, r, 1e-13)
    assert -1300 < to_list(out)[1] < -1200
    with pytest.raises(ValueError, match="a < b"):
        pu.log_prob_normal_in(as_kind([1.0], kind), as_kind([0.0], kind))
    # One constraint bound below and one above: logs of the normal tails.
    means = as_kind([[0.0, 1.0]], kind)
    sigmas = as_kind([[1.0, 2.0]], kind)
    lp = pu.compute_log_prob_feas_from_bounds(
        con_lower_inds=[0],
        con_upper_inds=[1],
        con_both_inds=[],
        con_lower=as_kind([0.5], kind),
        con_upper=as_kind([2.0], kind),
        con_both=as_kind([], kind) if kind != "list" else [],
        means=means,
        sigmas=sigmas,
    )
    close(lp, [math.log(1 - Phi(0.5)) + math.log(Phi(0.5))], 1e-13)


@pytest.mark.parametrize("kind", KINDS)
def test_percentile_case_dispatcher_and_indices(kind: str) -> None:
    scores = as_kind([[1.0, 2.0, 3.0, 4.0], [4.0, 4.0, 1.0, 0.0]], kind)
    close(pu.percentile_of_score(scores, as_kind([[2.5], [4.0]], kind), dim=-1), [50.0, 87.5])
    vals = [-1.0, 0.5, 2.0]
    x = as_kind(vals, kind)
    if kind == "list":
        neg = [v < 0 for v in vals]
        big = [v > 1 for v in vals]
    else:
        neg, big = x < 0, x > 1
    out = pu.case_dispatcher(
        as_kind([0.0, 0.0, 0.0], kind),
        cases=((lambda: neg, lambda mask: -1.0), (lambda: big, lambda mask: 1.0)),
        default=lambda mask: 7.0,
    )
    close(out, [-1.0, 7.0, 1.0])
    # Positional indices: dimension 1 (stride 1), then dimension 0 (stride 3).
    idx = list(pu.gen_positional_indices((2, 3), dim=-1))
    assert [to_list(v) for v in idx] == [[0, 1, 2], [[0], [3]]]
    with pytest.raises(ValueError, match="invalid"):
        list(pu.gen_positional_indices((2, 3), dim=2))


# ---------------------------------------------------------------------------
# bvn and MVNXPB
# ---------------------------------------------------------------------------


def _bvn_brute(a: float, b: float, r: float, n: int = 4000) -> float:
    """P(X > a, Y > b) for a standard bivariate normal with correlation r,
    by Simpson integration over x of phi(x) (1 - Phi((b - r x) / s))."""
    s = math.sqrt(1 - r * r)
    lo, hi = a, 9.0
    h = (hi - lo) / n
    total = 0.0
    for i in range(n + 1):
        x = lo + i * h
        w = 1 if i in (0, n) else (4 if i % 2 else 2)
        total += w * phi(x) * (1 - Phi((b - r * x) / s))
    return total * h / 3


@pytest.mark.parametrize("kind", KINDS)
def test_bvn_family(kind: str) -> None:
    cases = [(-0.5, 0.3, 0.2), (1.0, -1.0, -0.7), (0.0, 0.0, 0.95), (2.0, 1.5, 0.0)]
    h = as_kind([c[0] for c in cases], kind)
    k = as_kind([c[1] for c in cases], kind)
    r = as_kind([c[2] for c in cases], kind)
    # bvnu(r, h, k) = P(X > h, Y > k).
    out = bvnu(r, h, k)
    check_kind(out, kind)
    close(out, [_bvn_brute(*c) for c in cases], 1e-9)
    # bvn(a, b) over a box: inclusion-exclusion of bvnu.
    rr = as_kind([0.3], kind)
    # bvn(r, xl, yl, xu, yu) = P(xl < X < xu, yl < Y < yu).
    p = to_list(
        bvn(
            rr,
            as_kind([-1.0], kind),
            as_kind([-0.5], kind),
            as_kind([0.5], kind),
            as_kind([2.0], kind),
        )
    )[0]
    ref = (
        _bvn_brute(-1.0, -0.5, 0.3)
        - _bvn_brute(0.5, -0.5, 0.3)
        - _bvn_brute(-1.0, 2.0, 0.3)
        + _bvn_brute(0.5, 2.0, 0.3)
    )
    close(p, ref, 1e-9)
    m1, m2 = bvnmom(
        as_kind([0.3], kind),
        as_kind([-1.0], kind),
        as_kind([-0.5], kind),
        as_kind([0.5], kind),
        as_kind([2.0], kind),
    )
    check_kind(m1, kind)
    assert math.isfinite(to_list(m1)[0]) and math.isfinite(to_list(m2)[0])


@pytest.mark.parametrize("kind", KINDS)
def test_mvnxpb_against_bvn(kind: str) -> None:
    from botorch_rs.utils.probability.mvnxpb import MVNXPB

    cov = as_kind([[1.0, 0.3], [0.3, 1.0]], kind)
    bounds = as_kind([[-1.0, 0.5], [-0.5, 2.0]], kind)
    solver = MVNXPB(cov, bounds)
    solver.solve()
    log_prob = to_list(solver.log_prob)
    ref = (
        _bvn_brute(-1.0, -0.5, 0.3)
        - _bvn_brute(0.5, -0.5, 0.3)
        - _bvn_brute(-1.0, 2.0, 0.3)
        + _bvn_brute(0.5, 2.0, 0.3)
    )
    close(math.exp(log_prob if isinstance(log_prob, float) else log_prob), ref, 1e-9)
    # Solving twice on an odd dimension is a no-op (BoTorch: UnboundLocalError).
    s3 = MVNXPB(
        as_kind([[1.0, 0.2, 0.1], [0.2, 1.0, 0.3], [0.1, 0.3, 1.0]], kind),
        as_kind([[-1.0, 1.0]] * 3, kind),
    )
    s3.solve()
    first = to_list(s3.log_prob)
    s3.solve()
    assert to_list(s3.log_prob) == first


# ---------------------------------------------------------------------------
# Linear elliptical slice sampling
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("kind", KINDS)
def test_lin_ess_kinds_and_feasibility(kind: str) -> None:
    from botorch_rs.utils.probability.lin_ess import LinearEllipticalSliceSampler

    A = as_kind([[1.0, 1.0], [-1.0, 0.0]], kind)
    b = as_kind([[1.0], [0.5]], kind)
    random.seed(0)
    if kind == "numpy":
        import numpy as np

        np.random.seed(0)
    if kind == "torch":
        import torch

        torch.manual_seed(0)
    s = LinearEllipticalSliceSampler(inequality_constraints=(A, b), num_chains=2)
    X = s.draw(50)
    check_kind(X, kind)
    for x in to_list(X):
        assert x[0] + x[1] <= 1.0 + 1e-12 and -x[0] <= 0.5 + 1e-12
    assert s.lifetime_samples == 50
    with pytest.raises(ValueError):
        LinearEllipticalSliceSampler(
            inequality_constraints=(A, b), interior_point=as_kind([[5.0], [5.0]], kind)
        )


def test_lin_ess_seeded_runs_repeat() -> None:
    from botorch_rs.utils.probability.lin_ess import LinearEllipticalSliceSampler

    A = [[1.0, 1.0], [-1.0, 0.0]]
    b = [[1.0], [0.5]]
    random.seed(4)
    x1 = LinearEllipticalSliceSampler(inequality_constraints=(A, b)).draw(10)
    random.seed(4)
    x2 = LinearEllipticalSliceSampler(inequality_constraints=(A, b)).draw(10)
    assert x1 == x2


# ---------------------------------------------------------------------------
# Samplers and QMC engines
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("kind", KINDS)
def test_qmc_engines_kinds(kind: str) -> None:
    from botorch_rs.sampling.qmc import MultivariateNormalQMCEngine, NormalQMCEngine

    e = NormalQMCEngine(3, seed=1, inv_transform=True)
    dtype = None
    if kind == "numpy":
        import numpy as np

        dtype = np.float64
    elif kind == "torch":
        import torch

        dtype = torch.float64
    out = e.draw(16, dtype=dtype) if dtype is not None else e.draw(16)
    rows = to_list(out)
    assert len(rows) == 16 and all(len(r) == 3 for r in rows)
    mean = sum(r[0] for r in rows) / 16
    assert abs(mean) < 0.3
    m = MultivariateNormalQMCEngine(
        as_kind([1.0, -1.0], kind), as_kind([[2.0, 0.5], [0.5, 1.0]], kind), seed=2
    )
    x = m.draw(64)
    check_kind(x, kind)
    xs = to_list(x)
    assert abs(sum(r[0] for r in xs) / 64 - 1.0) < 0.2
    with pytest.raises(ValueError):
        MultivariateNormalQMCEngine(
            as_kind([0.0, 0.0], kind), as_kind([[1.0, 2.0], [2.0, 1.0]], kind)
        )


def test_iid_sampler_api() -> None:
    from botorch_rs.sampling import IIDNormalSampler, NormalMCSampler, get_sampler
    from botorch_rs.sampling.stochastic_samplers import ForkedRNGSampler

    s = IIDNormalSampler(sample_shape=[4], seed=3)
    assert isinstance(s, NormalMCSampler)
    assert s.seed == 3 and tuple(s.sample_shape) == (4,)
    assert ForkedRNGSampler(sample_shape=[2], seed=1).seed == 1
    assert callable(get_sampler)


@pytest.mark.parametrize("seed", [None, 1])
def test_native_qmc_continues_one_sequence(seed: int | None) -> None:
    from botorch_rs.sampling.qmc import _RustSobol

    engine = _RustSobol(3, seed)
    reference = _RustSobol(3, engine.seed)
    points = engine.draw(3) + engine.draw(5) + engine.draw(0) + engine.draw(7)
    assert points == reference.draw(15)
    assert engine.num_generated == 15
    # Reject unsupported lengths without advancing or overflowing the cursor.
    with pytest.raises(ValueError, match="65536"):
        engine.draw(65536)
    with pytest.raises(ValueError, match="65536"):
        engine.draw(2**63)
    assert engine.num_generated == 15
    assert engine.draw(2) == reference.draw(2)


def test_qmc_list_inputs_do_not_import_optional_libraries() -> None:
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; from botorch_rs.sampling.qmc import NormalQMCEngine, "
            "MultivariateNormalQMCEngine; "
            "engine = NormalQMCEngine(3); assert isinstance(engine.draw(4), list); "
            "multi = MultivariateNormalQMCEngine([0.0], [[1.0]]); "
            "assert isinstance(multi.draw(4), list); "
            "assert 'torch' not in sys.modules; assert 'numpy' not in sys.modules",
        ],
        check=True,
    )


@pytest.mark.parametrize("cov", [1.0, [1.0], [[[1.0]]], [[1.0, 0.0]]])
def test_qmc_rejects_nonmatrix_covariance(cov: Any) -> None:
    from botorch_rs.sampling.qmc import MultivariateNormalQMCEngine

    with pytest.raises(ValueError, match="not square"):
        MultivariateNormalQMCEngine([0.0], cov)


@pytest.mark.parametrize("mean", [0.0, [[0.0]], [0.0, 1.0]])
def test_qmc_rejects_nonvector_mean(mean: Any) -> None:
    from botorch_rs.sampling.qmc import MultivariateNormalQMCEngine

    with pytest.raises(ValueError, match="Dimension mismatch"):
        MultivariateNormalQMCEngine(mean, [[1.0]])


@pytest.mark.skipif(not HAS_TORCH, reason="torch")
def test_iid_sampler_draws_are_torch_draws() -> None:
    import torch

    from botorch_rs import _botorch_rs as _rs

    torch.manual_seed(9)
    assert torch.equal(
        torch.randn(40, dtype=torch.float64),
        torch.tensor(list(_rs.torch_randn(9, 40)[0]), dtype=torch.float64),
    )


def test_torch_rng_without_torch() -> None:
    from botorch_rs import _botorch_rs as _rs

    # torch.manual_seed(0); torch.randn(3) / torch.rand(2)
    close(
        list(_rs.torch_randn(0, 3)[0]),
        [1.5409960746765137, -0.293428897857666, -2.1787893772125244],
        1e-6,
    )
    v = list(_rs.torch_rand(0, 2, True)[0])
    close(v, [0.49625658988952637, 0.7682217955589294], 0.0)


# ---------------------------------------------------------------------------
# Test functions
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("kind", KINDS)
def test_test_functions_kinds(kind: str) -> None:
    from botorch_rs.test_functions.multi_fidelity import AugmentedBranin
    from botorch_rs.test_functions.sensitivity_analysis import Ishigami
    from botorch_rs.test_functions.synthetic import ConstrainedGramacy, PressureVessel

    x = as_kind([[0.0, 0.0, 0.0]], kind)
    close(Ishigami().evaluate_true(x), [0.0])
    close(Ishigami(b=0.1).evaluate_true(as_kind([[math.pi / 2, 0.0, 1.0]], kind)), [1.0 + 0.1])
    ab = AugmentedBranin()
    xb = as_kind([[-math.pi, 12.275, 1.0]], kind)
    close(ab.evaluate_true(xb), [0.397887357729738], 1e-12)
    cg = ConstrainedGramacy()
    xg = as_kind([[0.5, 0.25]], kind)
    out = cg.evaluate_slack_true(xg)
    check_kind(out, kind)
    x1, x2 = 0.5, 0.25
    c1 = 1.5 - x1 - 2 * x2 - 0.5 * math.sin(2 * math.pi * (x1**2 - 2 * x2))
    c2 = x1**2 + x2**2 - 1.5
    close(out, [[-c1, -c2]])
    pv = PressureVessel()
    with pytest.raises(ValueError):
        pv.evaluate_true(as_kind([[1.0, 1.0]], kind))


def test_round_nearest_and_validation() -> None:
    from botorch_rs.test_functions.base import validate_inputs, validate_parameter_indices
    from botorch_rs.test_functions.utils import round_nearest

    assert round_nearest([[0.5, 1.5, 2.5, -0.5]], 1.0, None) == [[0.0, 2.0, 2.0, -0.0]]
    # Multiples outside the bounds move one increment inwards.
    assert round_nearest([[0.26, 0.9]], 0.25, (0.1, 0.8)) == [[0.25, 0.75]]
    with pytest.raises(ValueError):
        validate_parameter_indices(3, [[0.0] * 3, [1.0] * 3], [0], [1], [1])
    with pytest.raises(ValueError):
        validate_inputs([[0.5, 1.5]], 2, [[0.0, 0.0], [1.0, 2.0]], [1], [])


# ---------------------------------------------------------------------------
# botorch.utils
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("kind", KINDS)
def test_transforms_kinds(kind: str) -> None:
    from botorch_rs.utils.transforms import normalize, standardize, unnormalize

    Y = as_kind([[1.0], [2.0], [4.0]], kind)
    out = standardize(Y)
    check_kind(out, kind)
    m, sd = 7 / 3, math.sqrt(((1 - 7 / 3) ** 2 + (2 - 7 / 3) ** 2 + (4 - 7 / 3) ** 2) / 2)
    close(out, [[(v - m) / sd] for v in (1.0, 2.0, 4.0)])
    b = as_kind([[0.0, 1.0], [2.0, 1.0]], kind)
    X = as_kind([[1.0, 1.0]], kind)
    # A constant column: its bounds become (lower, lower + 1).
    close(normalize(X, b), [[0.5, 0.0]])
    close(unnormalize(as_kind([[0.5, 0.5]], kind), b), [[1.0, 1.5]])
    out = to_list(normalize(X, b, update_constant_bounds=False))
    assert out[0][0] == 0.5 and math.isnan(out[0][1])


@pytest.mark.parametrize("kind", KINDS)
def test_objective_and_rounding_kinds(kind: str) -> None:
    from botorch_rs.utils.objective import compute_feasibility_indicator
    from botorch_rs.utils.rounding import approximate_round

    close(approximate_round(as_kind([0.0, 0.5, 1.2], kind), tau=1e-3), [0.0, 0.5, 1.0], 1e-6)
    samples = as_kind([[[1.0, -1.0]], [[-1.0, 1.0]]], kind)
    if kind == "list":
        pytest.skip("constraint callables need array arithmetic")
    feas = compute_feasibility_indicator([lambda Y: Y[..., 0]], samples)
    assert [[bool(v) for v in row] for row in to_list(feas)] == [[False], [True]]


def test_constants_dispatcher_settings_logging() -> None:
    from botorch_rs.logging import shape_to_str
    from botorch_rs.settings import log_level, propagate_grads, validate_input_scaling
    from botorch_rs.utils.dispatcher import Dispatcher, MDNotImplementedError

    d = Dispatcher("f")

    @d.register(int)
    def _i(x: int) -> str:
        return "int"

    @d.register(object)
    def _o(x: object) -> str:
        return "object"

    assert d(1) == "int" and d(1.5) == "object" and d(True) == "int"
    e = Dispatcher("g")

    @e.register(float)
    def _f(x: float) -> str:
        raise MDNotImplementedError

    @e.register(object)
    def _p(x: object) -> str:
        return "fallback"

    assert e(1.0) == "fallback"
    with pytest.raises(NotImplementedError):
        Dispatcher("h")(1)
    assert shape_to_str((2, 3)) == "`2 x 3`"
    with validate_input_scaling(False):
        assert not validate_input_scaling.on()
    assert validate_input_scaling.on()
    with propagate_grads(True):
        assert propagate_grads.on()
    with log_level(10):
        pass


def test_datasets_and_containers_lists() -> None:
    from botorch_rs.exceptions.errors import UnsupportedError
    from botorch_rs.utils.datasets import MultiTaskDataset, SupervisedDataset

    d0 = SupervisedDataset(
        [[0.1, 0.0], [0.2, 0.0]], [[1.0], [2.0]], feature_names=["x", "t"], outcome_names=["y0"]
    )
    d1 = SupervisedDataset([[0.3, 1.0]], [[3.0]], feature_names=["x", "t"], outcome_names=["y1"])
    mt = MultiTaskDataset([d0, d1], target_outcome_name="y1", task_feature_index=1)
    assert to_list(mt.X) == [[0.1, 0.0], [0.2, 0.0], [0.3, 1.0]]
    assert to_list(mt.Y) == [[1.0], [2.0], [3.0]]
    with pytest.raises(UnsupportedError, match="same outcome"):
        MultiTaskDataset([d0, d0], target_outcome_name="y0", task_feature_index=1)


@pytest.mark.skipif(not HAS_NUMPY, reason="numpy")
def test_polytope_sampler_numpy() -> None:
    import numpy as np

    from botorch_rs.utils.sampling import (
        HitAndRunPolytopeSampler,
        normalize_dense_linear_constraints,
    )

    bounds = np.array([[0.0, 0.0], [1.0, 2.0]])
    A = np.array([[1.0, 1.0]])
    b = np.array([[1.5]])
    s = HitAndRunPolytopeSampler(inequality_constraints=(A, b), bounds=bounds, seed=0, n_burnin=20)
    X = s.draw(30)
    assert isinstance(X, np.ndarray) and X.shape == (30, 2)
    assert (X @ A.T <= 1.5 + 1e-9).all() and (X >= 0).all() and (X[:, 1] <= 2).all()
    An, bn = normalize_dense_linear_constraints(bounds, (A, b))
    assert np.allclose(An, [[1.0, 2.0]]) and np.allclose(bn, [[1.5]])


def test_multi_objective_leftovers_lists() -> None:
    from botorch_rs.utils.multi_objective.hypervolume import MultiList, Node, sort_by_dimension

    nodes = [
        Node(m=2, dtype=None, device=None, data=[2.0, 1.0]),
        Node(m=2, dtype=None, device=None, data=[1.0, 3.0]),
    ]
    sort_by_dimension(nodes, 0)
    assert [to_list(n.data) for n in nodes] == [[1.0, 3.0], [2.0, 1.0]]
    ml = MultiList(m=2, dtype=None, device=None)
    ml.extend(nodes, 0)
    assert ml.sentinel.next[0] is nodes[0]

"""Parity with BoTorch exactly at clamp bounds, where PyTorch 2.14 changed
the subgradients (pytorch#191142): a scalar bound has the subgradient 0 at
equality (it was 1), and a tensor bound splits the gradient evenly with the
input at a tie. botorch_rs follows the ≥ 2.14 convention under every torch
version.

Under torch ≥ 2.14 these tests compare directly with BoTorch. Under torch
≤ 2.13 BoTorch's gradients are computed with the ≥ 2.14 rule
(``clamp_subgradients.torch214_clamp_subgradients``: the clamp family
replaced by an autograd function with the native values and the ≥ 2.14
backward), and the change against BoTorch's native (≤ 2.13) gradients is
recorded, and required to be nonzero, to show that the cases do sit at the
bounds. The cases:

* the emulation itself (against native torch ≥ 2.14 on random ties,
  infinite, inverted and degenerate bounds, broadcasting and in-place
  variants; against PyTorch's release-note examples under ≤ 2.13);
* ``SingleTaskMultiFidelityGP`` at fidelities 0 and 1: the
  ``LinearTruncatedFidelityKernel`` (one and two data fidelities, iteration
  and data fidelity), which clamps the fidelities to ``[0, 1]``, and the
  downsampling / exponential decay kernels (no clamp; a control). Values
  and input gradients of ``PosteriorMean``, ``PosteriorStandardDeviation``,
  ``ExpectedImprovement`` and ``qExpectedImprovement`` (``q = 2``, identical
  base samples); botorch_rs's gradient with respect to a fidelity at 0 or 1
  is exactly 0 for the linear truncated kernel;
* ``Warp`` at its ``eps`` bounds: ``x = 0`` maps to ``eps`` exactly, ``x = 1``
  to ``1 - eps`` exactly for ``eps = 2^-20`` but not for the default ``1e-7``;
* ``optimize_acqf`` on a multi-fidelity model from starts on the bounds of
  the fidelity: the projected gradient at the bound decides whether L-BFGS-B
  leaves it (all restarts compared, both L-BFGS-B paths);
* ``qExpectedImprovement``, ``qNoisyExpectedImprovement``, a
  ``ConstrainedMCObjective`` (in ``qSimpleRegret`` and its torch forward),
  ``FeasibilityWeightedMCMultiOutputObjective`` and
  ``qExpectedHypervolumeImprovement`` with samples exactly at the bound of
  their ``clamp_min(0)`` (deterministic models, whose samples are exact).

Tolerances: values 1e-9, gradients 1e-8 (relative to the largest component),
optimized candidates 1e-7 (``optim9_helpers.XTOL``).
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Iterator
from typing import Any

import pytest

from helpers import importorskip_botorch

importorskip_botorch()
torch = pytest.importorskip("torch")

from botorch.acquisition import ExpectedImprovement as BtEI  # noqa: E402
from botorch.acquisition import PosteriorMean as BtPM  # noqa: E402
from botorch.acquisition import PosteriorStandardDeviation as BtPSD  # noqa: E402
from botorch.acquisition import qExpectedImprovement as BtQEI  # noqa: E402
from botorch.acquisition import qNoisyExpectedImprovement as BtQNEI  # noqa: E402
from botorch.acquisition import qSimpleRegret as BtQSR  # noqa: E402
from botorch.acquisition.multi_objective import (  # noqa: E402
    qExpectedHypervolumeImprovement as BtQEHVI,
)
from botorch.acquisition.multi_objective.objective import (  # noqa: E402
    FeasibilityWeightedMCMultiOutputObjective as BtFeasibilityWeighted,
)
from botorch.acquisition.objective import ConstrainedMCObjective as BtConstrained  # noqa: E402
from botorch.models import SingleTaskGP as BtSingleTaskGP  # noqa: E402
from botorch.models import SingleTaskMultiFidelityGP as BtMF  # noqa: E402
from botorch.models.deterministic import AffineDeterministicModel as BtAffine  # noqa: E402
from botorch.models.transforms.input import Warp as BtWarp  # noqa: E402
from botorch.optim import optimize_acqf as bt_optimize_acqf  # noqa: E402
from botorch.sampling.normal import SobolQMCNormalSampler as BtSampler  # noqa: E402
from botorch.sampling.stochastic_samplers import StochasticSampler as BtStochastic  # noqa: E402
from botorch.utils.multi_objective.box_decompositions import (  # noqa: E402
    NondominatedPartitioning as BtPartitioning,
)
from clamp_subgradients import (  # noqa: E402
    TORCH_GE_214,
    clamp214,
    torch214_clamp_subgradients,
)
from models7b_helpers import compare_mll, randomize  # noqa: E402
from optim9_helpers import XTOL, lbfgsb_modes  # noqa: E402

from botorch_rs.acquisition import ExpectedImprovement as RsEI  # noqa: E402
from botorch_rs.acquisition import PosteriorMean as RsPM  # noqa: E402
from botorch_rs.acquisition import PosteriorStandardDeviation as RsPSD  # noqa: E402
from botorch_rs.acquisition import qExpectedImprovement as RsQEI  # noqa: E402
from botorch_rs.acquisition import qNoisyExpectedImprovement as RsQNEI  # noqa: E402
from botorch_rs.acquisition import qSimpleRegret as RsQSR  # noqa: E402
from botorch_rs.acquisition.multi_objective import (  # noqa: E402
    qExpectedHypervolumeImprovement as RsQEHVI,
)
from botorch_rs.acquisition.multi_objective.objective import (  # noqa: E402
    FeasibilityWeightedMCMultiOutputObjective as RsFeasibilityWeighted,
)
from botorch_rs.acquisition.objective import ConstrainedMCObjective as RsConstrained  # noqa: E402
from botorch_rs.models import SingleTaskGP as RsSingleTaskGP  # noqa: E402
from botorch_rs.models import SingleTaskMultiFidelityGP as RsMF  # noqa: E402
from botorch_rs.models.deterministic import AffineDeterministicModel as RsAffine  # noqa: E402
from botorch_rs.models.transforms.input import Warp as RsWarp  # noqa: E402
from botorch_rs.optim import optimize_acqf as rs_optimize_acqf  # noqa: E402
from botorch_rs.sampling import SobolQMCNormalSampler as RsSampler  # noqa: E402
from botorch_rs.sampling import StochasticSampler as RsStochastic  # noqa: E402
from botorch_rs.utils.multi_objective.box_decompositions import (  # noqa: E402
    NondominatedPartitioning as RsPartitioning,
)
from parity_helpers import Checker  # noqa: E402

pytestmark = pytest.mark.filterwarnings("ignore")

DT = torch.float64
CHECK = Checker("clamp bounds (torch ≥ 2.14 subgradients)")
VALUE_TOL = 1e-9
GRAD_TOL = 1e-8
#: A change of BoTorch's gradients from the torch ≤ 2.13 convention above
#: this is real (the emulation's autograd graph alone changes them by up to
#: ~1e-17, measured: 9.5e-18 for the control without clamps).
CHANGE_FLOOR = 1e-12


@pytest.fixture(autouse=True, scope="module")
def _single_threaded_torch() -> Iterator[None]:
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(threads)


def _gen(seed: int) -> Any:
    return torch.Generator().manual_seed(seed)


def grad_err(a: Any, e: Any) -> float:
    """Gradient error relative to the largest component."""
    a = torch.as_tensor(a, dtype=DT).reshape(-1)
    e = torch.as_tensor(e, dtype=DT).reshape(-1)
    assert a.shape == e.shape, (a.shape, e.shape)
    scale = max(e.abs().max().item(), 1e-300)
    return (a - e).abs().max().item() / scale


def values_and_grads(acqf: Any, X: Any) -> tuple[Any, Any]:
    Xg = X.detach().clone().requires_grad_(True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        v = acqf(Xg)
    if not v.requires_grad:
        return v.detach(), torch.zeros_like(X)
    (g,) = torch.autograd.grad(v.sum(), Xg, allow_unused=True)
    return v.detach(), torch.zeros_like(X) if g is None else g


def bt_values_and_grads(acqf: Any, X: Any) -> tuple[Any, Any, Any]:
    """BoTorch's values and gradients with the ≥ 2.14 subgradients, and its
    native gradients under torch ≤ 2.13 (``None`` under ≥ 2.14)."""
    with torch214_clamp_subgradients():
        v, g = values_and_grads(acqf, X)
    native = None if TORCH_GE_214 else values_and_grads(acqf, X)[1]
    return v, g, native


def compare(label: str, a_rs: Any, a_bt: Any, X: Any) -> tuple[Any, float]:
    """Values and input gradients of an acquisition-function pair; returns
    botorch_rs's gradient and the size of the change of BoTorch's gradient
    from the ≤ 2.13 convention (0 under torch ≥ 2.14)."""
    vb, gb, native = bt_values_and_grads(a_bt, X)
    vr, gr = values_and_grads(a_rs, X)
    CHECK.check(f"{label}: values", vr, vb, VALUE_TOL)
    err = grad_err(gr, gb)
    CHECK.record(f"{label}: gradients", err, GRAD_TOL)
    assert err <= GRAD_TOL, (label, err)
    change = 0.0 if native is None else grad_err(native, gb)
    if native is not None:
        CHECK.record(f"{label}: change from the torch ≤ 2.13 convention", change, None)
    return gr, change


# ---------------------------------------------------------------------------
# The emulation of the torch ≥ 2.14 subgradients
# ---------------------------------------------------------------------------


def _grads(fn: Any, args: tuple, w: Any) -> list[Any]:
    ts = [a.detach().clone().requires_grad_(True) if torch.is_tensor(a) else a for a in args]
    out = fn(*ts)
    tens = [t for t in ts if torch.is_tensor(t)]
    gs = torch.autograd.grad((out * w).sum(), tens, allow_unused=True)
    return [out.detach()] + [
        torch.zeros_like(t) if g is None else g for t, g in zip(tens, gs, strict=True)
    ]


def test_clamp_emulation_matches_torch_2_14() -> None:
    # PyTorch's release-note examples (any torch version).
    x = torch.tensor(0.0, dtype=DT, requires_grad=True)
    (g,) = torch.autograd.grad(clamp214(x, 0.0), x)
    assert g.item() == 0.0
    v, b = (torch.tensor(0.0, dtype=DT, requires_grad=True) for _ in range(2))
    gv, gb = torch.autograd.grad(clamp214(v, b), (v, b))
    assert (gv.item(), gb.item()) == (0.5, 0.5)
    # The patched functions and methods, including in-place variants, give
    # the native values and the ≥ 2.14 gradients.
    lo, hi = torch.zeros(3, dtype=DT), torch.ones(3, dtype=DT)
    cases = {
        "torch.clamp scalar": (lambda t: torch.clamp(t, 0.0, 1.0), [0.0, 1.0, 0.0]),
        "Tensor.clip min": (lambda t: t.clip(min=0.0), [0.0, 1.0, 1.0]),
        "torch.clamp_min": (lambda t: torch.clamp_min(t, 0.0), [0.0, 1.0, 1.0]),
        "Tensor.clamp_max": (lambda t: t.clamp_max(1.0), [1.0, 1.0, 0.0]),
        "Tensor.clamp_min_": (lambda t: (t * 1).clamp_min_(0.0), [0.0, 1.0, 1.0]),
        "Tensor.clamp tensor": (lambda t: t.clamp(lo, hi), [0.5, 1.0, 0.5]),
        "torch.clip tensor max": (lambda t: torch.clip(t, max=hi), [1.0, 1.0, 0.5]),
        "Tensor.clamp_ tensor": (lambda t: (t * 1).clamp_(min=lo), [0.5, 1.0, 1.0]),
    }
    x0 = torch.tensor([0.0, 0.5, 1.0], dtype=DT)
    for name, (fn, expected) in cases.items():
        with torch214_clamp_subgradients():
            out, gx = _grads(fn, (x0,), torch.ones(3, dtype=DT))
        assert gx.tolist() == expected, (name, gx.tolist())
        assert torch.equal(out, fn(x0)), name
    if not TORCH_GE_214:
        return
    # Under torch ≥ 2.14: the emulation equals the native backward on
    # random ties (finite, infinite, inverted and degenerate bounds,
    # broadcasting).
    g = _gen(0)
    grid = torch.tensor([-1.0, 0.0, 0.5, 1.0, 2.0, math.inf, -math.inf], dtype=DT)
    for trial in range(60):
        x = grid[torch.randint(0, len(grid), (5, 4), generator=g)]
        lo_t = grid[torch.randint(0, 5, (5, 4), generator=g)]
        hi_t = grid[torch.randint(0, 5, (5, 4), generator=g)]
        lo_s, hi_s = float(grid[trial % 5]), float(grid[(trial // 5) % 5])
        for args in (
            (x, lo_t, hi_t),
            (x, lo_t, None),
            (x, None, hi_t),
            (x, lo_s, hi_s),
            (x, lo_s, None),
            (x, None, hi_s),
            (x, lo_t[0], hi_t[0]),
        ):
            w = torch.rand(5, 4, generator=g, dtype=DT)
            native = _grads(lambda *a: torch.clamp(*a), args, w)
            emulated = _grads(clamp214, args, w)
            for n, e in zip(native, emulated, strict=True):
                assert torch.equal(n, e), (args, native, emulated)


# ---------------------------------------------------------------------------
# Multi-fidelity kernels at fidelities 0 and 1
# ---------------------------------------------------------------------------

#: name -> (keyword arguments, fidelity columns, linear truncated, whether the
#: fidelities enter only through the clamp). Two data fidelities give a
#: product of two ``LinearTruncatedFidelityKernel``s, each of which treats
#: the other fidelity as an ordinary input; an iteration and a data fidelity
#: give one kernel with two fidelity dimensions.
MF_CASES: dict[str, tuple[dict[str, Any], list[int], bool, bool]] = {
    "ltf_data": ({"data_fidelities": [2]}, [2], True, True),
    "ltf_two_data": ({"data_fidelities": [1, 2]}, [1, 2], True, False),
    "ltf_iteration_data": (
        {"iteration_fidelity": 1, "data_fidelities": [2]},
        [1, 2],
        True,
        True,
    ),
    "downsampling_exp_decay": (
        {"iteration_fidelity": 1, "data_fidelities": [2], "linear_truncated": False},
        [1, 2],
        False,
        False,
    ),
}


def _mf_data(seed: int) -> tuple[Any, Any, Any]:
    """Training data with fidelities in {0, 0.5, 1}, and query points with
    fidelities at 0 and 1 (and one interior row)."""
    g = _gen(seed)
    X = torch.rand(15, 3, generator=g, dtype=DT)
    X[:, 1] = torch.tensor([0.0, 0.5, 1.0] * 5, dtype=DT)
    X[:, 2] = torch.tensor([0.0, 1.0, 0.5] * 5, dtype=DT)
    Y = torch.sin(3 * X[:, :1]) + 0.5 * X[:, 1:2] + 0.8 * X[:, 2:3] ** 2
    Y = Y + 0.05 * torch.randn(15, 1, generator=g, dtype=DT)
    Xq = torch.rand(6, 3, generator=g, dtype=DT)
    Xq[:, 1] = torch.tensor([0.0, 1.0, 1.0, 0.0, 0.3, 0.5], dtype=DT)
    Xq[:, 2] = torch.tensor([0.0, 1.0, 0.0, 1.0, 0.5, 1.0], dtype=DT)
    return X, Y, Xq


def mf_pair(name: str, seed: int) -> tuple[Any, Any, Any, Any]:
    kw = MF_CASES[name][0]
    X, Y, Xq = _mf_data(seed)
    bt, rs = BtMF(X, Y, **kw), RsMF(X, Y, **kw)
    randomize(bt, rs, seed + 1)
    bt.eval()
    return bt, rs, X, Xq


def compare_mc(label: str, make_bt: Any, make_rs: Any, X: Any, num: int = 16) -> float:
    """A Monte-Carlo acquisition function with BoTorch's base samples."""
    sb = BtSampler(sample_shape=torch.Size([num]), seed=0)
    a_bt = make_bt(sb)
    with torch.no_grad():
        a_bt(X)
    sr = RsSampler(sample_shape=torch.Size([num]), seed=0)
    sr.base_samples = sb.base_samples.clone()
    return compare(label, make_rs(sr), a_bt, X)[1]


@pytest.mark.parametrize("name", list(MF_CASES))
def test_multi_fidelity_kernels_at_fidelity_bounds(name: str) -> None:
    _, fid, truncated, only_clamped = MF_CASES[name]
    seed = 8100 + list(MF_CASES).index(name)
    bt, rs, X, Xq = mf_pair(name, seed)
    compare_mll(CHECK, f"multi-fidelity ({name})", bt, rs)
    bt.eval()
    # A float64 tensor: BoTorch stores a Python float in the default dtype.
    best_f = bt.train_targets.max().detach()
    X1 = Xq.unsqueeze(-2)
    at_bound = torch.zeros(Xq.shape, dtype=torch.bool)
    for c in fid:
        at_bound[:, c] = (Xq[:, c] == 0.0) | (Xq[:, c] == 1.0)
    changes = []
    for label, a_bt, a_rs in (
        ("PosteriorMean", BtPM(bt), RsPM(rs)),
        ("PosteriorStandardDeviation", BtPSD(bt), RsPSD(rs)),
        ("ExpectedImprovement", BtEI(bt, best_f=best_f), RsEI(rs, best_f=best_f)),
    ):
        gr, change = compare(f"multi-fidelity ({name}): {label}", a_rs, a_bt, X1)
        changes.append(change)
        g = gr.reshape(Xq.shape)
        if only_clamped:
            # The fidelities enter only through the clamp to [0, 1].
            assert bool((g[at_bound] == 0.0).all()), (label, g)
        else:
            assert bool((g[at_bound] != 0.0).any()), (label, g)
    changes.append(
        compare_mc(
            f"multi-fidelity ({name}): qExpectedImprovement (q = 2)",
            lambda s: BtQEI(bt, best_f=best_f, sampler=s),
            lambda s: RsQEI(rs, best_f=best_f, sampler=s),
            Xq.reshape(3, 2, 3),
        )
    )
    if not TORCH_GE_214:
        # The query points do sit at the bounds of BoTorch's clamps (and
        # nothing changes without them).
        assert (max(changes) > CHANGE_FLOOR) == truncated, changes


# ---------------------------------------------------------------------------
# Warp at its eps bounds
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("eps", [1e-7, 2.0**-20])
def test_warp_at_eps_bounds(eps: float) -> None:
    g = _gen(8200 + int(eps < 1e-6))
    X = torch.rand(12, 3, generator=g, dtype=DT)
    X[:3, 0], X[3:6, 0], X[6:8, 1], X[8:10, 1] = 0.0, 1.0, 0.0, 1.0
    Y = torch.sin(4 * X[:, :1]) + X[:, 1:2] ** 2 + 0.3 * X[:, 2:3]
    Y = Y + 0.05 * torch.randn(12, 1, generator=g, dtype=DT)
    bt = BtSingleTaskGP(X, Y, input_transform=BtWarp(d=3, indices=[0, 1], eps=eps))
    rs = RsSingleTaskGP(X, Y, input_transform=RsWarp(d=3, indices=[0, 1], eps=eps))
    randomize(bt, rs, 8210)
    compare_mll(CHECK, f"Warp (eps = {eps:g})", bt, rs)
    bt.eval()
    Xq = torch.tensor(
        [[0.0, 0.3, 0.5], [1.0, 0.6, 0.2], [0.0, 1.0, 0.7], [1.0, 0.0, 0.1], [0.4, 0.7, 0.9]],
        dtype=DT,
    )
    # x = 0 maps to eps (a tie with the lower bound); x = 1 maps to
    # 1 · (1 - 2 eps) + eps, a tie with 1 - eps only for some eps.
    upper_tie = 1.0 * (1 - 2 * eps) + eps == 1.0 - eps
    assert upper_tie == (eps == 2.0**-20)
    tie = torch.zeros(Xq.shape, dtype=torch.bool)
    tie[:, :2] = (Xq[:, :2] == 0.0) | ((Xq[:, :2] == 1.0) & upper_tie)
    best_f = Y.max()
    changes = []
    for label, a_bt, a_rs in (
        ("PosteriorMean", BtPM(bt), RsPM(rs)),
        ("ExpectedImprovement", BtEI(bt, best_f=best_f), RsEI(rs, best_f=best_f)),
    ):
        gr, change = compare(f"Warp (eps = {eps:g}): {label}", a_rs, a_bt, Xq.unsqueeze(-2))
        changes.append(change)
        gr = gr.reshape(Xq.shape)
        assert bool((gr[tie] == 0.0).all()), (label, gr)
        assert bool((gr[:, :2][~tie[:, :2]] != 0.0).all()), (label, gr)
    if not TORCH_GE_214:
        assert max(changes) > CHANGE_FLOOR, changes


# ---------------------------------------------------------------------------
# optimize_acqf from starts on the bounds of the fidelity
# ---------------------------------------------------------------------------


def _one_sided_slopes(acqf: Any, ics: Any, col: int, h: float = 1e-6) -> Any:
    """``acqf``'s one-sided derivative along ``col`` from inside the unit
    interval at every start (the ≤ 2.13 gradient at the bound)."""
    with torch.no_grad():
        inward = torch.where(ics[..., col] == 0.0, h, -h)
        moved = ics.clone()
        moved[..., col] += inward
        return (acqf(moved) - acqf(ics)) / inward.squeeze(-1)


@pytest.mark.parametrize("acq", ["PosteriorMean", "ExpectedImprovement"])
def test_optimize_acqf_multi_fidelity_at_bounds(acq: str) -> None:
    bt, rs, X, _ = mf_pair("ltf_data", 8300)
    best_f = bt.train_targets.max().detach()
    a_bt = BtPM(bt) if acq == "PosteriorMean" else BtEI(bt, best_f=best_f)
    a_rs = RsPM(rs) if acq == "PosteriorMean" else RsEI(rs, best_f=best_f)
    g = _gen(8301)
    ics = 0.1 + 0.8 * torch.rand(8, 1, 3, generator=g, dtype=DT)
    ics[:, 0, 2] = torch.tensor([0.0, 1.0] * 4, dtype=DT)
    bounds = torch.stack([torch.zeros(3, dtype=DT), torch.ones(3, dtype=DT)])
    # Starts where the ≤ 2.13 gradient points into the interior (where the
    # convention decides whether L-BFGS-B leaves the bound).
    slopes = _one_sided_slopes(a_rs, ics, 2)
    deciding = torch.where(ics[:, 0, 2] == 0.0, slopes > 0, slopes < 0)
    assert bool(deciding.any()), slopes
    for options in lbfgsb_modes():
        tag = "serial" if options else "parallel"
        kw = dict(batch_initial_conditions=ics, options=options, return_best_only=False)
        with torch214_clamp_subgradients(), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cb, vb = bt_optimize_acqf(a_bt, bounds, 1, 8, **kw)
        cr, vr = rs_optimize_acqf(a_rs, bounds, 1, 8, **kw)
        label = f"optimize_acqf on a multi-fidelity GP ({acq}, {tag})"
        CHECK.check(f"{label}: candidates", cr, cb, XTOL)
        CHECK.check(f"{label}: values", vr, vb, VALUE_TOL)
        # The fidelity's subgradient at the bound is 0: no restart leaves it.
        cr = torch.as_tensor(cr, dtype=DT)
        assert torch.equal(cr[:, 0, 2], ics[:, 0, 2]), (cr[:, 0, 2], ics[:, 0, 2])
        if not TORCH_GE_214:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                cn, _ = bt_optimize_acqf(a_bt, bounds, 1, 8, **kw)
            moved = (cn[:, 0, 2] != ics[:, 0, 2]).sum().item()
            CHECK.record(f"{label}: restarts leaving the bound under torch ≤ 2.13", moved, None)
            assert moved > 0


# ---------------------------------------------------------------------------
# Monte-Carlo improvements with a sample exactly at the clamp bound
# ---------------------------------------------------------------------------


def _affine(a: list[list[float]]) -> tuple[Any, Any]:
    at = torch.tensor(a, dtype=DT)
    return BtAffine(a=at, b=0.0), RsAffine(a=at, b=0.0)


def test_mc_improvements_with_samples_at_the_bound() -> None:
    bt, rs = _affine([[1.0], [0.0]])
    # A float64 tensor: BoTorch stores a Python float in the default dtype.
    best_f = torch.tensor(0.3, dtype=DT)

    def samplers() -> tuple[Any, Any]:
        return BtStochastic(torch.Size([1])), RsStochastic(torch.Size([1]))

    changes = []
    # qEI: improvements of exactly 0 (ties with best_f), q = 1 and q = 2.
    X1 = torch.tensor([[[0.3, 0.7]], [[0.3, 0.2]], [[0.25, 0.1]], [[0.4, 0.5]]], dtype=DT)
    X2 = torch.tensor(
        [[[0.3, 0.7], [0.1, 0.4]], [[0.3, 0.2], [0.3, 0.9]], [[0.3, 0.1], [0.35, 0.0]]], dtype=DT
    )
    for X, ties in ((X1, [0, 1]), (X2, [0, 1])):
        sb, sr = samplers()
        gr, change = compare(
            f"qExpectedImprovement at best_f (q = {X.shape[-2]})",
            RsQEI(rs, best_f=best_f, sampler=sr),
            BtQEI(bt, best_f=best_f, sampler=sb),
            X,
        )
        changes.append(change)
        assert bool((gr[ties] == 0.0).all()), gr
    # qNEI: a candidate tying with the best baseline point.
    Xb = torch.tensor([[0.3, 0.1], [0.2, 0.5]], dtype=DT)
    sb, sr = samplers()
    gr, change = compare(
        "qNoisyExpectedImprovement at the best baseline value",
        RsQNEI(rs, Xb, sampler=sr, prune_baseline=False),
        BtQNEI(bt, Xb, sampler=sb, prune_baseline=False),
        torch.tensor([[[0.3, 0.9]], [[0.5, 0.5]], [[0.1, 0.0]]], dtype=DT),
    )
    changes.append(change)
    assert bool((gr[0] == 0.0).all()), gr
    # A ConstrainedMCObjective: obj + infeasible_cost == 0 at x0 = -0.5.
    sb, sr = samplers()
    cons = [lambda Y: Y[..., 0] - 1.0]
    gr, change = compare(
        "qSimpleRegret with a ConstrainedMCObjective at obj = -M",
        RsQSR(rs, sampler=sr, objective=RsConstrained(lambda Y, X=None: Y[..., 0], cons, 0.5)),
        BtQSR(bt, sampler=sb, objective=BtConstrained(lambda Y, X=None: Y[..., 0], cons, 0.5)),
        torch.tensor([[[-0.5, 0.3]], [[0.2, 0.3]]], dtype=DT),
    )
    changes.append(change)
    # qEHVI: objective values on the lower bounds of the cells.
    bt2, rs2 = _affine([[1.0, 0.0], [0.0, 1.0]])
    ref = [0.0, 0.0]
    front = torch.tensor([[1.0, 0.5], [0.5, 1.0]], dtype=DT)
    sb, sr = samplers()
    _, change = compare(
        "qExpectedHypervolumeImprovement on cell bounds",
        RsQEHVI(rs2, ref, RsPartitioning(torch.tensor(ref, dtype=DT), Y=front), sampler=sr),
        BtQEHVI(bt2, ref, BtPartitioning(torch.tensor(ref, dtype=DT), Y=front), sampler=sb),
        torch.tensor([[[1.0, 0.3]], [[0.5, 0.75]], [[1.2, 0.3]], [[0.7, 0.7]]], dtype=DT),
    )
    changes.append(change)
    # The torch forwards of the feasibility-weighted objectives (differentiated
    # by torch in both libraries), with samples at obj + infeasible_cost == 0.
    f = lambda Y, X=None: Y[..., 0]  # noqa: E731
    Y = torch.tensor([[[-0.5, 0.3]], [[0.2, 0.3]]], dtype=DT)
    gr, change = compare(
        "ConstrainedMCObjective at obj = -M",
        RsConstrained(f, cons, 0.5),
        BtConstrained(f, cons, 0.5),
        Y,
    )
    changes.append(change)
    assert bool((gr[0] == 0.0).all()), gr
    # Baseline objectives >= 0 and zero variances: the infeasible cost is 0.
    Xb2 = torch.tensor([[0.2, 0.4], [0.6, 0.1]], dtype=DT)
    gr, change = compare(
        "FeasibilityWeightedMCMultiOutputObjective at obj = -M",
        RsFeasibilityWeighted(rs2, Xb2, [1]),
        BtFeasibilityWeighted(bt2, Xb2, [1]),
        torch.tensor([[[0.0, 0.5]], [[0.3, 0.5]]], dtype=DT),
    )
    changes.append(change)
    assert bool((gr[0] == 0.0).all()), gr
    if not TORCH_GE_214:
        assert min(changes) > CHANGE_FLOOR, changes

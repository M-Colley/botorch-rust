"""Shared helpers of the parity tests of the multi-objective acquisition
functions (``test_parity_mo_acquisition.py``,
``test_equivalence_multi_objective.py``) and of the phase-5B fixtures.

Import only after ``importorskip_botorch()``: this module requires torch and
BoTorch.

Base samples. BoTorch's samplers draw base samples for ``D`` jointly sampled
values with ``draw_sobol_normal_samples(d=D, n=S, seed=seed)`` whenever the
required shape changes, and the noisy hypervolume acquisition functions
combine draws of several sizes (the baseline's and the joint draw with new
points). :func:`synced_draws` replaces BoTorch's draws by seeded Gaussian
matrices ``Z_D`` (one per ``D``), and :func:`rs_sampler` installs the same
matrices in a botorch_rs sampler for every ``D`` up to a bound, so that both
libraries use identical base samples in every evaluation.
"""

from __future__ import annotations

import contextlib
import inspect
from collections.abc import Iterator
from typing import Any

import botorch.sampling.normal as bt_normal
import torch
from botorch.acquisition.multi_objective import objective as bt_mo_objective
from botorch.sampling.normal import SobolQMCNormalSampler as BtSampler
from botorch.utils.multi_objective import box_decompositions as bt_bd
from botorch.utils.multi_objective import hypervolume as bt_hv

from botorch_rs.acquisition.multi_objective import objective as rs_mo_objective
from botorch_rs.sampling import SobolQMCNormalSampler as RsSampler
from botorch_rs.utils.multi_objective import box_decompositions as rs_bd

DT = torch.float64


def zdraw(d: int, n: int, seed: int) -> torch.Tensor:
    """The base samples ``Z_d`` (``n x d``) of a seed."""
    g = torch.Generator().manual_seed(seed * 7919 + d)
    return torch.randn(n, d, generator=g, dtype=DT)


@contextlib.contextmanager
def synced_draws(seed: int, record: dict[int, torch.Tensor] | None = None) -> Iterator[None]:
    """BoTorch's sampler draws ``Z_d`` (:func:`zdraw`) instead of Sobol
    samples; drawn matrices are recorded in ``record`` (if given)."""
    orig = bt_normal.draw_sobol_normal_samples

    def draw(d: int, n: int, device: Any = None, dtype: Any = None, seed: Any = None) -> Any:
        z = zdraw(d, n, _SEED[0])
        if record is not None:
            record[d] = z.clone()
        return z.to(dtype=dtype or DT)

    old = _SEED[0]
    _SEED[0] = seed
    bt_normal.draw_sobol_normal_samples = draw
    try:
        yield
    finally:
        bt_normal.draw_sobol_normal_samples = orig
        _SEED[0] = old


_SEED = [0]


def bt_sampler(num_samples: int) -> BtSampler:
    return BtSampler(sample_shape=torch.Size([num_samples]), seed=0)


def rs_sampler(num_samples: int, seed: int, max_dim: int) -> RsSampler:
    """A botorch_rs sampler with the base samples ``Z_d`` of ``seed`` for
    every ``d <= max_dim``."""
    s = RsSampler(sample_shape=torch.Size([num_samples]), seed=0)
    for d in range(1, max_dim + 1):
        s._rs.set_base_samples((zdraw(d, num_samples, seed).reshape(-1).tolist(), (num_samples, d)))
    return s


# ---------------------------------------------------------------------------
# BoTorch versions
# ---------------------------------------------------------------------------


def botorch_has_pending_semantics() -> bool:
    """Whether BoTorch's NEHVI replaces the complete pending set in
    ``set_X_pending`` (main since #3379; the released 0.18.1 only appends)."""
    return hasattr(bt_hv.NoisyExpectedHypervolumeMixin, "_is_prefix")


def botorch_keeps_baseline_base_samples() -> bool:
    """Whether BoTorch's NEHVI keeps the base samples of previously
    decomposed points when caching pending points (meta-pytorch/botorch#3392,
    which botorch_rs implements)."""
    src = inspect.getsource(bt_hv.NoisyExpectedHypervolumeMixin)
    return "_baseline_base_sampler" in src


def botorch_binds_constraint_indices() -> bool:
    """Whether BoTorch's ``FeasibilityWeightedMCMultiOutputObjective``
    evaluates every constraint index (meta-pytorch/botorch#3392)."""
    src = inspect.getsource(bt_mo_objective.FeasibilityWeightedMCMultiOutputObjective)
    return "lambda Y, i=i" in src


# ---------------------------------------------------------------------------
# Fronts, partitionings, objectives and constraints
# ---------------------------------------------------------------------------


def weak_front(bt_model: Any, X: torch.Tensor, outputs: list[int] | None = None) -> tuple[Any, Any]:
    """A reference point below the posterior means at ``X`` and a weakened
    front (the means pulled 40% of the way to the reference point), so that
    new points improve on it."""
    with torch.no_grad():
        Y = bt_model.posterior(X).mean
    if outputs is not None:
        Y = Y[..., outputs]
    lo, hi = Y.min(0).values, Y.max(0).values
    ref = lo - 0.1 * (hi - lo).clamp_min(1e-3)
    return ref, ref + 0.6 * (Y - ref)


def partitioning_pair(kind: str, ref: torch.Tensor, Y: torch.Tensor, alpha: float = 0.0) -> Any:
    """The same box decomposition in BoTorch and botorch_rs."""
    if kind == "nondominated":
        return (
            bt_bd.NondominatedPartitioning(ref_point=ref, Y=Y, alpha=alpha),
            rs_bd.NondominatedPartitioning(ref_point=ref, Y=Y, alpha=alpha),
        )
    return (
        bt_bd.FastNondominatedPartitioning(ref_point=ref, Y=Y),
        rs_bd.FastNondominatedPartitioning(ref_point=ref, Y=Y),
    )


def generic_mo(Y: Any, X: Any = None) -> Any:
    """A pointwise nonlinear multi-output objective (two objectives)."""
    return torch.stack([Y[..., 0] + 0.1 * Y[..., -1] ** 2, -((Y[..., 1] - 0.5) ** 2)], dim=-1)


def objective_pair(kind: str, m: int, bt_model: Any = None, rs_model: Any = None, X: Any = None):
    """The same multi-output objective in both libraries, and the number of
    objectives: ``identity``, ``outcomes`` (the first two), ``weighted``,
    ``generic`` or ``feasibility`` (the last output is a constraint)."""
    if kind == "identity":
        return None, None, m
    if kind == "outcomes":
        return (
            bt_mo_objective.IdentityMCMultiOutputObjective(outcomes=[0, 1]),
            rs_mo_objective.IdentityMCMultiOutputObjective(outcomes=[0, 1]),
            2,
        )
    if kind == "weighted":
        w = torch.tensor([1.0, -0.5] + [0.75] * (m - 2), dtype=DT)
        return (
            bt_mo_objective.WeightedMCMultiOutputObjective(w),
            rs_mo_objective.WeightedMCMultiOutputObjective(w),
            m,
        )
    if kind == "generic":
        return (
            bt_mo_objective.GenericMCMultiOutputObjective(generic_mo),
            rs_mo_objective.GenericMCMultiOutputObjective(generic_mo),
            2,
        )
    if kind == "feasibility":
        return (
            bt_mo_objective.FeasibilityWeightedMCMultiOutputObjective(
                model=bt_model, X_baseline=X, constraint_idcs=[-1]
            ),
            rs_mo_objective.FeasibilityWeightedMCMultiOutputObjective(
                model=rs_model, X_baseline=X, constraint_idcs=[-1]
            ),
            m - 1,
        )
    raise ValueError(kind)


def outcome_constraints(means: torch.Tensor, kind: str) -> tuple[list[Any], torch.Tensor]:
    """Outcome constraints on the model outputs (feasible iff ``<= 0``):
    ``one`` (the last output below its mean), ``two`` (also a quadratic
    one)."""
    c0 = means[:, -1].mean().item()
    cons = [lambda Y, c=c0: Y[..., -1] - c]
    etas = [0.05]
    if kind == "two":
        c1 = (means[:, 0] ** 2).mean().item()
        cons.append(lambda Y, c=c1: Y[..., 0] ** 2 - 1.5 * c)
        etas.append(0.2)
    return cons, torch.tensor(etas, dtype=DT)

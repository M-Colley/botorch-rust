"""Shared helpers of the phase 9 comparisons with BoTorch (optimization and
candidate generation): model and acquisition-function pairs, the L-BFGS-B
mode of the installed BoTorch, and replacements of BoTorch's random draws.

Import only after ``importorskip_botorch()``.
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterator
from typing import Any

import torch
from botorch.acquisition import LogExpectedImprovement as BtLogEI
from botorch.acquisition import qLogExpectedImprovement as BtqLogEI
from botorch.models.deterministic import PosteriorMeanModel as BtPM
from botorch.sampling.stochastic_samplers import StochasticSampler as BtStochastic

from botorch_rs.acquisition import LogExpectedImprovement as RsLogEI
from botorch_rs.acquisition import qLogExpectedImprovement as RsqLogEI
from botorch_rs.models.deterministic import PosteriorMeanModel as RsPM
from botorch_rs.sampling import StochasticSampler as RsStochastic
from parity_helpers import ModelSpec, model_pair

DT = torch.float64

#: Values from identical inputs.
VALUE_TOL = 1e-9
#: Optimized candidates: at a converged optimum the objective is flat, so
#: round-off differences of the gradients (~1e-15) move the final iterate by
#: up to ~sqrt(eps) ~ 3e-8; measured worst 2.0e-8 (sequential greedy q = 2 in
#: the randomized suite; the values agree to ~1e-14).
XTOL = 1e-7

#: Options that put BoTorch (main and 0.18.1) and botorch_rs on BoTorch's
#: serial L-BFGS-B path (one problem per chunk of restarts, SciPy's
#: ``maxfun``); 0.18.1 always takes it with SciPy 1.18.
JOINT = {"maxfun": 15000}


def botorch_parallel_lbfgsb() -> bool:
    """Whether BoTorch's parallel L-BFGS-B path (``fmin_l_bfgs_b_batched``)
    is available (BoTorch main with SciPy 1.13 - 1.18; not 0.18.1 with
    SciPy 1.18)."""
    from botorch.generation import gen

    return not gen.get_reasons_against_fast_path(
        method="L-BFGS-B", with_grad=True, minimize_options={"maxiter": 2000}, timeout_sec=None
    )


def lbfgsb_modes() -> list[dict[str, Any]]:
    """The ``options`` of the L-BFGS-B modes to compare: the default
    (parallel) path where BoTorch has it, and the serial path."""
    return ([{}] if botorch_parallel_lbfgsb() else []) + [dict(JOINT)]


def pairs(d: int = 3, n: int = 12, seed: int = 3) -> dict[str, Any]:
    """A SingleTaskGP pair with explicit hyperparameters and the acquisition
    functions of the comparisons: analytic LogEI, and qLogEI on the
    posterior mean (deterministic, so its values do not depend on the
    sampler; pending points and q > 1 work in both libraries)."""
    bt, rs, X = model_pair(ModelSpec(n=n, d=d), seed)
    return {
        "bt": bt,
        "rs": rs,
        "X": X,
        "d": d,
        "logei": (BtLogEI(bt, best_f=1.0), RsLogEI(rs, best_f=1.0)),
        "qlogei": (
            BtqLogEI(BtPM(bt), best_f=1.0, sampler=BtStochastic(torch.Size([1]))),
            RsqLogEI(RsPM(rs), best_f=1.0, sampler=RsStochastic(torch.Size([1]))),
        ),
        "bounds": torch.stack([torch.zeros(d, dtype=DT), torch.ones(d, dtype=DT)]),
    }


@contextlib.contextmanager
def patched(module: Any, name: str, value: Any) -> Iterator[None]:
    """Temporarily replaces ``module.name``."""
    orig = getattr(module, name)
    setattr(module, name, value)
    try:
        yield
    finally:
        setattr(module, name, orig)


def lattice_points(seed: int, n: int = 64) -> Any:
    """A replacement of ``_gen_batch_initial_conditions_local_search`` with
    seeded draws (the same points in both libraries)."""

    def generate(
        discrete_choices: Any,
        raw_samples: int,
        X_avoid: Any,
        inequality_constraints: Any,
        min_points: int,
        max_tries: int = 100,
    ) -> Any:
        avoid = torch.as_tensor(X_avoid, dtype=DT).reshape(-1, len(discrete_choices))
        g = torch.Generator().manual_seed(seed + len(avoid))
        choices = [torch.as_tensor(c, dtype=DT) for c in discrete_choices]
        pts = torch.stack([c[torch.randint(len(c), (n,), generator=g)] for c in choices], -1)
        pts = torch.unique(pts, dim=0)
        if len(avoid):
            pts = pts[~(pts == avoid.unsqueeze(-2)).all(-1).any(-2)]
        for indices, coefficients, rhs in inequality_constraints or []:
            idx = torch.as_tensor(indices)
            w = torch.as_tensor(coefficients, dtype=DT)
            pts = pts[(pts[:, idx] * w).sum(-1) >= rhs]
        return pts

    return generate


def starting_points(points: Any, acq_function: Any) -> Any:
    """A replacement of ``generate_starting_points`` returning ``points``
    (``b x d``) and their values under ``acq_function`` (BoTorch's)."""

    def generate(opt_inputs: Any, discrete_dims: Any, cat_dims: Any, cont_dims: Any) -> Any:
        with torch.no_grad():
            return points.clone(), acq_function(points.unsqueeze(-2))

    return generate


class SharpPeak(torch.nn.Module):
    """A torch acquisition function whose sharpness is a homotopy parameter:
    ``-sum_i log(1 + ((x_i - c_i) / s)^2) + 0.3 sum_i sin(9 x_i)`` per
    point, summed over the q-batch (with optional pending points penalized
    by proximity)."""

    def __init__(self, center: Any) -> None:
        super().__init__()
        self.center = center
        self.s = torch.nn.Parameter(torch.tensor(1.0, dtype=DT))
        self.X_pending = None

    def set_X_pending(self, X_pending: Any = None) -> None:
        self.X_pending = None if X_pending is None else torch.as_tensor(X_pending, dtype=DT)

    def forward(self, X: Any) -> Any:
        X = torch.as_tensor(X, dtype=DT)
        z = (X - self.center) / self.s
        v = -torch.log1p(z.pow(2)).sum(-1) + 0.3 * torch.sin(9 * X).sum(-1)
        if self.X_pending is not None:
            dist = (X.unsqueeze(-2) - self.X_pending).pow(2).sum(-1)
            v = v - torch.exp(-dist / 0.01).sum(-1)
        return v.sum(-1)

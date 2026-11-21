"""Array-independent result types for the decision extensions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class FPBIGeometry:
    """Normalized cost-space plane and the paper's asymmetric anchor grid.

    Array fields use the input array kind. Basis vectors are columns; flat
    objectives have zero entries. Utopia/nadir use oriented cost coordinates.
    """

    utopia: Any
    nadir: Any
    normalized_extremes: Any
    active_objectives: Any
    farthest_pair: Any
    center: Any
    focus: Any
    direction: Any
    basis: Any
    anchors: Any
    degenerate: Any
    used_fallback_basis: Any
    n_steps: int
    maximize: bool


@dataclass(frozen=True)
class SolverDiagnostic:
    """SLSQP status and independently checked constraint violation.

    The native solver reports the best attempt per scalar subproblem;
    ``restart`` is -1 for an aggregate over restarts. The Torch/SciPy adapter
    reports every attempt separately. ``accepted`` requires solver success
    and feasibility, rather than feasibility alone.
    """

    kind: str
    index: int
    restart: int
    solver_success: bool
    accepted: bool
    status: int
    message: str
    max_violation: float
    kappa: float | None = None


@dataclass(frozen=True)
class FPBIResult:
    """Unique nondominated samples, individual extrema, and solver diagnostics."""

    X: Any
    Y: Any
    extreme_X: Any
    extreme_Y: Any
    geometry: FPBIGeometry
    diagnostics: tuple[SolverDiagnostic, ...]
    ray_succeeded: Any

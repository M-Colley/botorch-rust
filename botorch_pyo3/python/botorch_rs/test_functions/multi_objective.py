"""Multi-objective test problems (``botorch.test_functions.multi_objective``).

``problem(X)`` (``forward``) evaluates the objectives at ``X``
(``batch_shape x d`` -> ``batch_shape x M``), adds observation noise (one
standard deviation, or one per objective) if ``noise=True`` and negates them
if ``negate=True``; ``evaluate_true`` returns the noiseless, un-negated
objectives. Constrained problems have ``evaluate_slack(X, noise)``,
``evaluate_slack_true`` and ``is_feasible`` (slack ``>= 0`` is feasible).
``ref_point`` (negated with ``negate``), ``max_hv`` and, for the DTLZ, ZDT
and C2DTLZ2 problems, ``gen_pareto_front(n)`` are available as in BoTorch.
See :mod:`botorch_rs.test_functions.base` for arrays, noise and dtypes.

The evaluations run in Rust and agree with BoTorch to rounding. Ten problems
follow corrected definitions (BoTorch pull requests meta-pytorch/botorch#3402
and #3405, and one fix without a pull request); ``problem.deviation``
describes them:

* ``BraninCurrin``, ``ConstrainedBraninCurrin``: ``max_hv`` 59.4066 and
  609.4038 (BoTorch's NSGA-II estimates are exceeded by evaluated points).
* ``DH4``: ``max_hv`` depends on ``dim``.
* ``DTLZ1``: ``max_hv`` subtracts the simplex volume ``0.5^M / M!``.
* ``DTLZ4``: the position variables are mapped to ``x_i^100`` (BoTorch's
  DTLZ4 equals DTLZ2).
* ``ZDT3``: the second objective is multiplied by ``g``.
* ``C2DTLZ2``: the constraint of Jain & Deb (2014) (``r^2`` subtracted once;
  ``r = 0.4`` for ``M = 3``, ``0.5`` for ``M > 3``), batched inputs, the exact
  ``max_hv`` for ``M = 2``; ``gen_pareto_front`` returns only feasible points.
* ``SRN``: Srinivas & Deb's first objective and first constraint.
* ``WeldedBeam``: the slack is negated (BoTorch inverts feasibility).
* ``GMM``: float64 parameters (BoTorch evaluates float64 inputs in float32).

``gen_pareto_front`` of the DTLZ problems draws quasi-random points from
botorch_rs's scrambled Sobol sequence (keyword-only ``seed``), so they differ
from BoTorch's points but have the same distribution.
"""

from __future__ import annotations

import math
from typing import Any

from botorch_rs._arrays import from_flat, is_torch_tensor
from botorch_rs.test_functions.base import (
    ConstrainedBaseTestProblem,
    MultiObjectiveTestProblem,
)
from botorch_rs.test_functions.synthetic import Levy


def _neg(x: Any) -> Any:
    """``-x`` for arrays and nested lists."""
    if isinstance(x, list):
        return [_neg(v) for v in x]
    return -x


def _column(x: Any, j: int) -> Any:
    """``x[..., j:j+1]`` for arrays and nested lists."""
    if isinstance(x, list):
        if x and isinstance(x[0], list):
            return [_column(v, j) for v in x]
        return [x[j]]
    return x[..., j : j + 1]


class BraninCurrin(MultiObjectiveTestProblem):
    """Two-objective problem on ``[0, 1]^2``: the rescaled Branin function
    and the Currin function (both minimized)."""

    _name = "BraninCurrin"


class DH(MultiObjectiveTestProblem):
    """Base class of the DH problems of Deb & Gupta (2005) for robust
    multi-objective optimization: ``f_0(x) = x_0`` and ``f_1(x) = h(x) + g(x)
    S(x)`` (DH1, DH2) or ``h(x) (g(x) + S(x))`` (DH3, DH4), minimized."""

    def __init__(
        self,
        dim: int,
        noise_std: None | float | list[float] = None,
        negate: bool = False,
        dtype: Any = None,
        *,
        seed: int | None = None,
    ) -> None:
        self._size_args = (dim, None)
        super().__init__(noise_std=noise_std, negate=negate, dtype=dtype, seed=seed)


class DH1(DH):
    """DH1 on ``[0, 1] x [-1, 1]^{d-1}`` (``d >= 2``), with ``alpha = beta = 1``."""

    _name = "DH1"
    alpha = 1.0
    beta = 1.0


class DH2(DH1):
    """DH2: DH1 with ``beta = 10``."""

    _name = "DH2"
    beta = 10.0


class DH3(DH):
    """DH3 on ``[0, 1]^2 x [-1, 1]^{d-2}`` (``d >= 3``) with a local and a
    global Pareto front."""

    _name = "DH3"


class DH4(DH3):
    """DH4 on ``[0, 1] x [-0.15, 1] x [-1, 1]^{d-2}`` (``d >= 3``); its
    ``max_hv`` depends on ``dim`` (BoTorch PR #3402)."""

    _name = "DH4"


class DTLZ(MultiObjectiveTestProblem):
    """Base class of the DTLZ problems of Deb et al. (2005) with ``M =
    num_objectives`` objectives and ``dim > M`` inputs on ``[0, 1]^d``."""

    def __init__(
        self,
        dim: int,
        num_objectives: int = 2,
        noise_std: None | float | list[float] = None,
        negate: bool = False,
        dtype: Any = None,
        *,
        seed: int | None = None,
    ) -> None:
        self._size_args = (dim, num_objectives)
        super().__init__(noise_std=noise_std, negate=negate, dtype=dtype, seed=seed)

    @property
    def k(self) -> int:
        """The number of distance variables, ``dim - num_objectives + 1``."""
        return self.dim - self.num_objectives + 1


class DTLZ1(DTLZ):
    """DTLZ1: linear Pareto front ``sum_i f_i = 0.5``."""

    _name = "DTLZ1"


class DTLZ2(DTLZ):
    """DTLZ2: spherical Pareto front ``sum_i f_i^2 = 1``."""

    _name = "DTLZ2"


class DTLZ3(DTLZ2):
    """DTLZ3: DTLZ2 with many local Pareto fronts."""

    _name = "DTLZ3"


class DTLZ4(DTLZ2):
    """DTLZ4: DTLZ2 with the position variables mapped to ``x_i^alpha``
    (``alpha = 100``; BoTorch PR #3402)."""

    _name = "DTLZ4"
    _alpha = 100.0


class DTLZ5(DTLZ):
    """DTLZ5: a degenerate (curve-shaped) Pareto front."""

    _name = "DTLZ5"


class DTLZ7(DTLZ):
    """DTLZ7: ``2^(M-1)`` disconnected Pareto-optimal regions."""

    _name = "DTLZ7"


#: GMM's component means, standard deviations and weights (BoTorch's values,
#: in float64 as in BoTorch PR #3405).
_GMM_POS = [
    [[0.2, 0.2], [0.8, 0.2], [0.5, 0.7]],
    [[0.07, 0.2], [0.4, 0.8], [0.85, 0.1]],
    [[0.08, 0.21], [0.45, 0.75], [0.86, 0.11]],
    [[0.09, 0.19], [0.44, 0.72], [0.89, 0.13]],
]
_GMM_STD = [[0.2, 0.1, 0.1], [0.2, 0.1, 0.05], [0.2, 0.1, 0.07], [0.15, 0.07, 0.09]]
_GMM_WEIGHT = [[0.5, 0.7, 0.7], [0.5, 0.7, 0.7], [0.5, 0.7, 0.9], [0.5, 0.7, 0.9]]


class GMM(MultiObjectiveTestProblem):
    """``num_objectives`` (2 to 4) negated Gaussian mixtures on ``[0, 1]^2``
    (Frohlich et al., 2020; Daulton et al., 2022)."""

    _name = "GMM"

    def __init__(
        self,
        noise_std: None | float | list[float] = None,
        negate: bool = False,
        num_objectives: int = 2,
        dtype: Any = None,
        *,
        seed: int | None = None,
    ) -> None:
        self._size_args = (None, num_objectives)
        super().__init__(noise_std=noise_std, negate=negate, dtype=dtype, seed=seed)

    @property
    def gmm_pos(self) -> Any:
        """The means of the components (``M x 3 x 2``)."""
        m = self.num_objectives
        flat = [v for obj in _GMM_POS[:m] for comp in obj for v in comp]
        return from_flat(flat, (m, 3, 2), self._kind)

    @property
    def gmm_covar(self) -> Any:
        """The covariance matrices ``var I`` of the components (``M x 3 x 2 x 2``)."""
        m = self.num_objectives
        flat = [v for row in _GMM_STD[:m] for s in row for v in (s**2, 0.0, 0.0, s**2)]
        return from_flat(flat, (m, 3, 2, 2), self._kind)

    @property
    def gmm_norm(self) -> Any:
        """The weights ``2 pi var w`` of the densities (``M x 3``)."""
        m = self.num_objectives
        flat = [
            2 * math.pi * s**2 * w
            for row, ws in zip(_GMM_STD[:m], _GMM_WEIGHT[:m], strict=True)
            for s, w in zip(row, ws, strict=True)
        ]
        return from_flat(flat, (m, 3), self._kind)


class Penicillin(MultiObjectiveTestProblem):
    """Penicillin production simulator (Liang & Lai, 2021): 7 inputs (culture
    volume, biomass concentration, temperature, glucose concentration,
    substrate feed rate, substrate feed concentration, H+ concentration
    exponent) and 3 objectives (negated penicillin yield, CO2, time), all
    minimized."""

    _name = "Penicillin"
    Y_xs = 0.45
    Y_ps = 0.90
    K_1 = 10 ** (-10)
    K_2 = 7 * 10 ** (-5)
    m_X = 0.014
    alpha_1 = 0.143
    alpha_2 = 4 * 10 ** (-7)
    alpha_3 = 10 ** (-4)
    mu_X = 0.092
    K_X = 0.15
    mu_p = 0.005
    K_p = 0.0002
    K_I = 0.10
    K = 0.04
    k_g = 7.0 * 10**3
    E_g = 5100.0
    k_d = 10.0**33
    E_d = 50000.0
    lambd = 2.5 * 10 ** (-4)
    T_v = 273.0
    T_o = 373.0
    R = 1.9872
    V_max = 180.0

    @classmethod
    def penicillin_vectorized(cls, X_input: Any) -> Any:
        """The simulation at ``X_input`` (``n x 7``; within the bounds) ->
        ``n x 3`` (negated penicillin, CO2, time)."""
        return cls().evaluate_true(X_input)


class ToyRobust(MultiObjectiveTestProblem):
    """A one-dimensional problem on ``[0, 0.7]`` whose Pareto front is
    sensitive to input noise (Daulton et al., 2022)."""

    _name = "ToyRobust"

    def __init__(
        self,
        noise_std: None | float | list[float] = None,
        negate: bool = False,
        dtype: Any = None,
        *,
        seed: int | None = None,
    ) -> None:
        super().__init__(noise_std=noise_std, negate=negate, dtype=dtype, seed=seed)
        # Each instance has its own Levy function (BoTorch PR #3405).
        self.levy = Levy(dtype=dtype)

    def f_1(self, X: Any) -> Any:
        """The first objective (un-negated) at ``X`` (``... x 1``)."""
        return _neg(_column(self.evaluate_true(X), 0))

    def f_2(self, X: Any) -> Any:
        """The second objective (un-negated) at ``X`` (``... x 1``)."""
        return _neg(_column(self.evaluate_true(X), 1))


class VehicleSafety(MultiObjectiveTestProblem):
    """Vehicle crash-worthiness (Tanabe & Ishibuchi, 2020): 5 inputs on
    ``[1, 3]^5``, 3 objectives (minimized)."""

    _name = "VehicleSafety"


class ZDT(MultiObjectiveTestProblem):
    """Base class of the ZDT problems of Zitzler et al. (2000): two
    objectives, ``dim >= 2`` inputs on ``[0, 1]^d``."""

    def __init__(
        self,
        dim: int,
        num_objectives: int = 2,
        noise_std: None | float | list[float] = None,
        negate: bool = False,
        dtype: Any = None,
        *,
        seed: int | None = None,
    ) -> None:
        self._size_args = (dim, num_objectives)
        super().__init__(noise_std=noise_std, negate=negate, dtype=dtype, seed=seed)


class ZDT1(ZDT):
    """ZDT1: convex Pareto front ``f_1 = 1 - sqrt(f_0)``."""

    _name = "ZDT1"


class ZDT2(ZDT):
    """ZDT2: concave Pareto front ``f_1 = 1 - f_0^2``."""

    _name = "ZDT2"


class ZDT3(ZDT):
    """ZDT3: Pareto front of five disconnected convex parts (with the factor
    ``g`` of the second objective, BoTorch PR #3402)."""

    _name = "ZDT3"
    _parts = [
        [0, 0.0830015349],
        [0.1822287280, 0.2577623634],
        [0.4093136748, 0.4538821041],
        [0.6183967944, 0.6525117038],
        [0.8233317983, 0.8518328654],
    ]
    _eps = 1e-6


class CarSideImpact(MultiObjectiveTestProblem):
    """Car side-impact design (Tanabe & Ishibuchi, 2020): 7 inputs, 4
    objectives (minimized; the fourth is the total constraint violation)."""

    _name = "CarSideImpact"


# ------ Constrained Multi-Objective Test Problems ----- #


class BNH(MultiObjectiveTestProblem, ConstrainedBaseTestProblem):
    """The constrained BNH problem: 2 inputs, 2 objectives (minimized), 2
    constraints."""

    _name = "BNH"


class CONSTR(MultiObjectiveTestProblem, ConstrainedBaseTestProblem):
    """The constrained CONSTR problem: 2 inputs, 2 objectives (minimized), 2
    constraints."""

    _name = "CONSTR"


class ConstrainedBraninCurrin(BraninCurrin, ConstrainedBaseTestProblem):
    """BraninCurrin with the disk constraint of Gelbart et al. (2014)."""

    _name = "ConstrainedBraninCurrin"
    _con_bounds = [(-5.0, 10.0), (0.0, 15.0)]

    def __init__(
        self,
        noise_std: None | float | list[float] = None,
        constraint_noise_std: None | float | list[float] = None,
        negate: bool = False,
        dtype: Any = None,
        *,
        seed: int | None = None,
    ) -> None:
        self._constraint_noise_std = constraint_noise_std
        super().__init__(noise_std=noise_std, negate=negate, dtype=dtype, seed=seed)

    @property
    def con_bounds(self) -> Any:
        """The ``2 x 2`` domain of the constraint (Branin's domain)."""
        lower = [lo for lo, _ in self._con_bounds]
        upper = [hi for _, hi in self._con_bounds]
        return from_flat(lower + upper, (2, 2), self._kind)


class C2DTLZ2(DTLZ2, ConstrainedBaseTestProblem):
    """C2-DTLZ2 (Jain & Deb, 2014): DTLZ2 with one constraint that only leaves
    regions near the unit vectors and near ``(1, ..., 1) / sqrt(M)`` feasible
    (BoTorch PR #3402)."""

    _name = "C2DTLZ2"


class DiscBrake(MultiObjectiveTestProblem, ConstrainedBaseTestProblem):
    """The disc brake design problem (Tanabe & Ishibuchi, 2020): 4 inputs, 2
    objectives (minimized), 4 constraints."""

    _name = "DiscBrake"


class MW7(MultiObjectiveTestProblem, ConstrainedBaseTestProblem):
    """The MW7 problem (Ma & Wang, 2019): ``dim >= 2`` inputs on ``[0,
    1]^d``, 2 objectives (minimized), 2 constraints, disconnected Pareto
    front."""

    _name = "MW7"

    def __init__(
        self,
        dim: int,
        noise_std: None | float | list[float] = None,
        constraint_noise_std: None | float | list[float] = None,
        negate: bool = False,
        dtype: Any = None,
        *,
        seed: int | None = None,
    ) -> None:
        self._size_args = (dim, None)
        self._constraint_noise_std = constraint_noise_std
        super().__init__(noise_std=noise_std, negate=negate, dtype=dtype, seed=seed)

    def LA2(self, A: Any, B: Any, C: Any, D: Any, theta: Any) -> Any:
        """pymoo's ``A sin(B theta^C)^D`` (for numbers, arrays and nested lists)."""
        if isinstance(theta, list):
            return [self.LA2(A, B, C, D, t) for t in theta]
        if is_torch_tensor(theta):
            return A * (B * theta.pow(C)).sin().pow(D)
        if hasattr(theta, "shape"):
            import numpy as np

            return A * np.sin(B * np.power(theta, C)) ** D
        return A * math.sin(B * theta**C) ** D


class OSY(MultiObjectiveTestProblem, ConstrainedBaseTestProblem):
    """The OSY problem (Osyczka & Kundu, 1995): 6 inputs, 2 objectives, 6
    constraints (minimization; choose ``negate=True`` for maximization)."""

    _name = "OSY"


class SRN(MultiObjectiveTestProblem, ConstrainedBaseTestProblem):
    """The constrained SRN problem (Srinivas & Deb, 1994) on ``[-20, 20]^2``:
    2 objectives (minimized), 2 constraints."""

    _name = "SRN"


class WeldedBeam(MultiObjectiveTestProblem, ConstrainedBaseTestProblem):
    """The welded beam design problem (from pymoo): 4 inputs, 2 objectives
    (minimized), 4 constraints."""

    _name = "WeldedBeam"


__all__ = [
    "BNH",
    "C2DTLZ2",
    "CONSTR",
    "DH",
    "DH1",
    "DH2",
    "DH3",
    "DH4",
    "DTLZ",
    "DTLZ1",
    "DTLZ2",
    "DTLZ3",
    "DTLZ4",
    "DTLZ5",
    "DTLZ7",
    "GMM",
    "MW7",
    "OSY",
    "SRN",
    "ZDT",
    "ZDT1",
    "ZDT2",
    "ZDT3",
    "BraninCurrin",
    "CarSideImpact",
    "ConstrainedBraninCurrin",
    "DiscBrake",
    "Penicillin",
    "ToyRobust",
    "VehicleSafety",
    "WeldedBeam",
]

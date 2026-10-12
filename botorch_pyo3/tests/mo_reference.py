"""BoTorch's multi-objective test problems as references for botorch_rs,
including the corrections that botorch_rs implements ahead of BoTorch.

botorch_rs follows the corrected definitions of two open BoTorch pull
requests, and of one BoTorch bug without a pull request (``DEVIATIONS``).
``reference_class(name)`` returns BoTorch's own class when it already
contains the fix (e.g. when testing against the branch of the pull request)
and otherwise a subclass of BoTorch's class that applies the fix with the code
of the pull request. The comparisons therefore pass against BoTorch main, the
released 0.18.1 and the branches of the pull requests alike, and
``reference_source(name)`` tells which reference was used.

Used by ``test_parity_mo_test_functions.py`` and by the fixture generator
(``scripts/fixtures_mo_test_functions.py``). Import only when BoTorch is
importable.
"""

from __future__ import annotations

import math
from functools import cache
from math import pi
from typing import Any, NamedTuple

import torch
from botorch.test_functions import multi_objective as bt_mo
from botorch.test_functions import multi_objective_multi_fidelity as bt_momf
from torch import Tensor
from torch.distributions import MultivariateNormal

PR3402 = "meta-pytorch/botorch#3402"
PR3405 = "meta-pytorch/botorch#3405"
NO_PR = "no pull request (reported)"

#: The public multi-objective test problems of BoTorch (``botorch.test_functions``).
MULTI_OBJECTIVE = [
    "BraninCurrin",
    "DH1",
    "DH2",
    "DH3",
    "DH4",
    "DTLZ1",
    "DTLZ2",
    "DTLZ3",
    "DTLZ4",
    "DTLZ5",
    "DTLZ7",
    "GMM",
    "Penicillin",
    "ToyRobust",
    "VehicleSafety",
    "ZDT1",
    "ZDT2",
    "ZDT3",
    "CarSideImpact",
    "BNH",
    "CONSTR",
    "ConstrainedBraninCurrin",
    "C2DTLZ2",
    "DiscBrake",
    "MW7",
    "OSY",
    "SRN",
    "WeldedBeam",
]
MULTI_FIDELITY = ["MOMFBraninCurrin", "MOMFPark"]
#: The abstract base classes of ``botorch.test_functions.multi_objective``.
ABSTRACT = ["DH", "DTLZ", "ZDT"]


class Deviation(NamedTuple):
    """A definition in which botorch_rs deviates from BoTorch main."""

    pr: str
    reason: str
    #: Which quantities differ: "objectives", "slack", "max_hv", "pareto_front", "dtype".
    quantities: tuple[str, ...]


DEVIATIONS: dict[str, Deviation] = {
    "BraninCurrin": Deviation(
        PR3402,
        "max_hv 59.4066 (BoTorch's NSGA-II estimate 59.3601 is exceeded by evaluated points)",
        ("max_hv",),
    ),
    "ConstrainedBraninCurrin": Deviation(
        PR3402,
        "max_hv 609.4038 (BoTorch's NSGA-II estimate 608.4004 is exceeded by evaluated points)",
        ("max_hv",),
    ),
    "ZDT3": Deviation(
        PR3402,
        "the second objective is multiplied by g, as in Zitzler et al. (2000)",
        ("objectives",),
    ),
    "DH4": Deviation(
        PR3402,
        "max_hv depends on dim: h < 0 for x_0 > 0.98534, where f_1 decreases with g",
        ("max_hv",),
    ),
    "C2DTLZ2": Deviation(
        PR3402,
        "constraint of Jain & Deb (2014): r^2 subtracted once, r = 0.4 (M = 3) and 0.5 (M > 3), "
        "batched and 1-d inputs; exact max_hv for M = 2 (none otherwise)",
        ("slack", "max_hv"),
    ),
    "DTLZ1": Deviation(
        PR3402, "max_hv subtracts the simplex volume 0.5^M / M! (not 1 / 2^M)", ("max_hv",)
    ),
    "DTLZ4": Deviation(
        PR3402,
        "the position variables are mapped to x_i^100 (BoTorch's DTLZ4 equals DTLZ2)",
        ("objectives",),
    ),
    "SRN": Deviation(
        PR3402,
        "f_0 uses (x_1 - 1)^2 and the first constraint is x_0^2 + x_1^2 <= 225 (Srinivas & Deb)",
        ("objectives", "slack"),
    ),
    "WeldedBeam": Deviation(
        PR3402, "the constraint slack is negated (pymoo's g <= 0 convention)", ("slack",)
    ),
    "GMM": Deviation(
        PR3405,
        "float64 parameters: BoTorch builds them in float32, so float64 inputs are evaluated "
        "in float32 and return float32 values",
        ("dtype",),
    ),
}

#: Deviations of ``gen_pareto_front`` that have no BoTorch reference at all.
FRONT_DEVIATIONS: dict[str, Deviation] = {
    "C2DTLZ2": Deviation(
        NO_PR,
        "only feasible points: BoTorch inherits DTLZ2's front, i.e. the whole unit sphere, "
        "whose points are partly infeasible",
        ("pareto_front",),
    ),
}

DT = torch.float64


def bt_class(name: str) -> type:
    """BoTorch's class ``name`` (from either multi-objective module)."""
    module = bt_momf if name in MULTI_FIDELITY else bt_mo
    return getattr(module, name)


# ---------------------------------------------------------------------------
# Replicas of the fixes (the code of the pull requests)
# ---------------------------------------------------------------------------


class BraninCurrinFixed(bt_mo.BraninCurrin):
    _max_hv = 59.4066


class ConstrainedBraninCurrinFixed(bt_mo.ConstrainedBraninCurrin):
    _max_hv = 609.4038


class ZDT3Fixed(bt_mo.ZDT3):
    def _evaluate_true(self, X: Tensor) -> Tensor:
        f_0 = X[..., 0]
        g = self._g(X=X)
        f_1 = g * (1 - (f_0 / g).sqrt() - f_0 / g * torch.sin(10 * math.pi * f_0))
        return torch.stack([f_0, f_1], dim=-1)


class DH4Fixed(bt_mo.DH4):
    @property  # type: ignore[override]
    def _area_under_curve(self) -> float:
        return 0.228445034673 - 0.001572780262 * 50 * (self.dim - 2)


class C2DTLZ2Fixed(bt_mo.C2DTLZ2):
    @property  # type: ignore[override]
    def _r(self) -> float:
        if self.num_objectives == 2:
            return 0.2
        return 0.4 if self.num_objectives == 3 else 0.5

    @property  # type: ignore[override]
    def _max_hv(self) -> float | None:
        if self.num_objectives != 2:
            return None
        a = 2 * math.asin(self._r / 2)

        def _area_under_circle(lo: float, hi: float) -> float:
            def F(t: float) -> float:
                return 0.5 * (t * math.sqrt(1 - t * t) + math.asin(t))

            return F(hi) - F(lo)

        lo, hi = a, pi / 4 - a
        gap = (math.cos(lo) - math.cos(hi)) * (math.sin(hi) - math.sin(lo)) - (
            _area_under_circle(math.cos(hi), math.cos(lo))
            - math.sin(lo) * (math.cos(lo) - math.cos(hi))
        )
        return self._ref_val**2 - pi / 4 - 2 * gap

    def _evaluate_slack_true(self, X: Tensor) -> Tensor:
        f_X = self.evaluate_true(X=X)
        f_X_sq = f_X.pow(2)
        dist_to_e_i = (f_X - 1).pow(2) + f_X_sq.sum(dim=-1, keepdim=True) - f_X_sq
        dist_to_center = (f_X - 1 / math.sqrt(self.num_objectives)).pow(2).sum(dim=-1)
        min_dist = torch.minimum(dist_to_e_i.min(dim=-1).values, dist_to_center)
        return -(min_dist - self._r**2).unsqueeze(-1)


class DTLZ1Fixed(bt_mo.DTLZ1):
    @property
    def _max_hv(self) -> float:  # type: ignore[override]
        M = self.num_objectives
        return self._ref_val**M - 0.5**M / math.factorial(M)


class DTLZ4Fixed(bt_mo.DTLZ4):
    def _evaluate_true(self, X: Tensor) -> Tensor:
        n_pos = self.num_objectives - 1
        X = torch.cat([X[..., :n_pos].pow(self._alpha), X[..., n_pos:]], dim=-1)
        # DTLZ2's objectives (not `super()`, which already maps X if BoTorch
        # contains the fix).
        return bt_mo.DTLZ2._evaluate_true(self, X)


class SRNFixed(bt_mo.SRN):
    def _evaluate_true(self, X: Tensor) -> Tensor:
        obj1 = 2.0 + (X[..., 0] - 2.0).pow(2) + (X[..., 1] - 1.0).pow(2)
        obj2 = 9.0 * X[..., 0] - (X[..., 1] - 1.0).pow(2)
        return torch.stack([obj1, obj2], dim=-1)

    def _evaluate_slack_true(self, X: Tensor) -> Tensor:
        c1 = 225.0 - X.pow(2).sum(dim=-1)
        c2 = -10.0 - X[..., 0] + 3 * X[..., 1]
        return torch.stack([c1, c2], dim=-1)


class WeldedBeamFixed(bt_mo.WeldedBeam):
    def _evaluate_slack_true(self, X: Tensor) -> Tensor:
        # BoTorch's (pymoo's) constraint values, negated (self-contained, so
        # that it does not compound with BoTorch's fix).
        x1, x2, x3, x4 = X.unbind(-1)
        P = 6000.0
        L = 14.0
        t_max = 13600.0
        s_max = 30000.0
        R = torch.sqrt(0.25 * (x2.pow(2) + (x1 + x3).pow(2)))
        M = P * (L + x2 / 2)
        J = 2 * math.sqrt(0.5) * x1 * x2 * (x2.pow(2) / 12 + 0.25 * (x1 + x3).pow(2))
        t1 = P / (math.sqrt(2) * x1 * x2)
        t2 = M * R / J
        t = torch.sqrt(t1.pow(2) + t1 * t2 * x2 / R + t2.pow(2))
        s = 6 * P * L / (x4 * x3.pow(2))
        P_c = 64746.022 * (1 - 0.0282346 * x3) * x3 * x4.pow(3)
        g1 = (t - t_max) / t_max
        g2 = (s - s_max) / s_max
        g3 = 1 / (5 - 0.125) * (x1 - x4)
        g4 = (P - P_c) / P
        return -torch.stack([g1, g2, g3, g4], dim=-1)


class GMMFixed(bt_mo.GMM):
    def __init__(
        self,
        noise_std: None | float | list[float] = None,
        negate: bool = False,
        num_objectives: int = 2,
        dtype: torch.dtype = torch.double,
    ) -> None:
        super().__init__(
            noise_std=noise_std, negate=negate, num_objectives=num_objectives, dtype=dtype
        )
        # The parameters of #3405, created with `dtype` instead of float32.
        tkwargs: dict[str, Any] = {"dtype": dtype}
        gmm_pos = torch.tensor(
            [[[0.2, 0.2], [0.8, 0.2], [0.5, 0.7]], [[0.07, 0.2], [0.4, 0.8], [0.85, 0.1]]],
            **tkwargs,
        )
        gmm_var = torch.tensor([[0.20, 0.10, 0.10], [0.2, 0.1, 0.05]], **tkwargs).pow(2)
        gmm_norm = 2 * pi * gmm_var * torch.tensor([0.5, 0.7, 0.7], **tkwargs)
        if num_objectives > 2:
            gmm_pos = torch.cat(
                [gmm_pos, torch.tensor([[[0.08, 0.21], [0.45, 0.75], [0.86, 0.11]]], **tkwargs)],
                dim=0,
            )
            gmm_var = torch.cat([gmm_var, torch.tensor([[0.2, 0.1, 0.07]], **tkwargs).pow(2)])
            gmm_norm = torch.cat(
                [gmm_norm, 2 * pi * gmm_var[2] * torch.tensor([[0.5, 0.7, 0.9]], **tkwargs)]
            )
        if num_objectives > 3:
            gmm_pos = torch.cat(
                [gmm_pos, torch.tensor([[[0.09, 0.19], [0.44, 0.72], [0.89, 0.13]]], **tkwargs)],
                dim=0,
            )
            gmm_var = torch.cat([gmm_var, torch.tensor([[0.15, 0.07, 0.09]], **tkwargs).pow(2)])
            gmm_norm = torch.cat(
                [gmm_norm, 2 * pi * gmm_var[3] * torch.tensor([[0.5, 0.7, 0.9]], **tkwargs)]
            )
        gmm_covar = gmm_var.view(*gmm_var.shape, 1, 1) * torch.eye(2, **tkwargs)
        self.gmm_pos = gmm_pos
        self.gmm_covar = gmm_covar
        self.gmm_norm = gmm_norm

    def _evaluate_true(self, X: Tensor) -> Tensor:
        mvn = MultivariateNormal(loc=self.gmm_pos.to(X), covariance_matrix=self.gmm_covar.to(X))
        view_shape = (
            X.shape[:-1] + torch.Size([1] * (self.gmm_pos.ndim - 1)) + self.gmm_pos.shape[-1:]
        )
        expand_shape = X.shape[:-1] + self.gmm_pos.shape
        pdf_X = mvn.log_prob(X.view(view_shape).expand(expand_shape)).exp()
        return -(self.gmm_norm.to(X) * pdf_X).sum(dim=-1)


FIXED: dict[str, type] = {
    "BraninCurrin": BraninCurrinFixed,
    "ConstrainedBraninCurrin": ConstrainedBraninCurrinFixed,
    "ZDT3": ZDT3Fixed,
    "DH4": DH4Fixed,
    "C2DTLZ2": C2DTLZ2Fixed,
    "DTLZ1": DTLZ1Fixed,
    "DTLZ4": DTLZ4Fixed,
    "SRN": SRNFixed,
    "WeldedBeam": WeldedBeamFixed,
    "GMM": GMMFixed,
}


# ---------------------------------------------------------------------------
# Detection of fixes that BoTorch already contains
# ---------------------------------------------------------------------------


def _t(rows: list[list[float]]) -> Tensor:
    return torch.tensor(rows, dtype=DT)


@cache
def botorch_has_fix(name: str) -> bool:
    """Whether the imported BoTorch already contains the fix of ``name``
    (probed on BoTorch's own class)."""
    cls = bt_class(name)
    if name == "BraninCurrin":
        return cls._max_hv == 59.4066
    if name == "ConstrainedBraninCurrin":
        return cls._max_hv == 609.4038
    if name == "ZDT3":  # f(0.5, 1, 1) = (0.5, 7.764) instead of (0.5, 0.776)
        return bool(cls(dim=3).evaluate_true(_t([[0.5, 1.0, 1.0]]))[0, 1] > 5.0)
    if name == "DH4":
        return cls(dim=4).max_hv != cls(dim=3).max_hv
    if name == "C2DTLZ2":
        return cls(dim=4, num_objectives=3)._r == 0.4
    if name == "DTLZ1":
        return cls(dim=3).max_hv == 400.0**2 - 0.125
    if name == "DTLZ4":
        X = _t([[0.5, 0.5, 0.5]])
        return not torch.equal(cls(dim=3).evaluate_true(X), bt_mo.DTLZ2(dim=3).evaluate_true(X))
    if name == "SRN":
        return bool(cls().evaluate_true(_t([[1.0, 3.0]]))[0, 0] == 7.0)
    if name == "WeldedBeam":
        return bool(cls().evaluate_slack_true(_t([[0.125, 0.1, 0.1, 0.125]]))[0, 0] < 0)
    if name == "GMM":
        return cls().gmm_pos.dtype == torch.float64
    return True


def reference_class(name: str) -> type:
    """The class that botorch_rs's ``name`` is compared with: BoTorch's own
    class, or the replica of the fix if BoTorch does not contain it yet."""
    if name in FIXED and not botorch_has_fix(name):
        return FIXED[name]
    return bt_class(name)


def reference_source(name: str) -> str:
    """ "BoTorch" or the pull request whose replica is used."""
    if name in FIXED and not botorch_has_fix(name):
        return f"replica of {DEVIATIONS[name].pr}"
    return "BoTorch"


def make(name: str, kwargs: dict[str, Any], corrected: bool = True) -> Any:
    """An instance of the reference class (``corrected``) or of BoTorch's
    own class."""
    cls = reference_class(name) if corrected else bt_class(name)
    return cls(**kwargs)


# ---------------------------------------------------------------------------
# Objective-space checks of Pareto fronts
# ---------------------------------------------------------------------------


def c2dtlz2_feasible(F: Tensor) -> Tensor:
    """Feasibility of objective vectors ``F`` (``n x M``, minimization) for the
    corrected C2-DTLZ2 constraint of Jain & Deb (2014)."""
    M = F.shape[-1]
    r = 0.2 if M == 2 else (0.4 if M == 3 else 0.5)
    sq = F.pow(2)
    dist_e = (F - 1).pow(2) + sq.sum(dim=-1, keepdim=True) - sq
    dist_c = (F - 1 / math.sqrt(M)).pow(2).sum(dim=-1)
    return -(torch.minimum(dist_e.min(dim=-1).values, dist_c) - r**2) >= 0


def dominated(F: Tensor, maximize: bool = False) -> Tensor:
    """For each row of ``F`` (``n x M``), whether another row dominates it."""
    G = -F if maximize else F
    le = (G.unsqueeze(0) <= G.unsqueeze(1)).all(dim=-1)  # [i, j]: G_j <= G_i
    lt = (G.unsqueeze(0) < G.unsqueeze(1)).any(dim=-1)
    return (le & lt).any(dim=-1)

r"""
Synthetic functions for multi-fidelity optimization benchmarks
(``botorch.test_functions.multi_fidelity``), evaluated in Rust.

.. [Chen2024]
    Multi-fidelity benchmark problems cited by BoTorch's
    ``WingWeightMultiFidelity`` and ``BoreholeMultiFidelity``.
"""

from __future__ import annotations

import math
from typing import Any

from botorch_rs._arrays import from_pair, kind_of, to_flat
from botorch_rs.test_functions.synthetic import _NativeProblem


class AugmentedBranin(_NativeProblem):
    r"""Augmented Branin test function for multi-fidelity optimization.

    3-dimensional function with domain ``[-5, 10] x [0, 15] * [0,1]``, where
    the last dimension of is the fidelity parameter:

        B(x) = (x_2 - (b - 0.1 * (1 - x_3))x_1^2 + c x_1 - r)^2 +
            10 (1-t) cos(x_1) + 10

    Here ``b``, ``c``, ``r`` and ``t`` are constants where
    ``b = 5.1 / (4 * math.pi ** 2)`` ``c = 5 / math.pi``, ``r = 6``,
    ``t = 1 / (8 * math.pi)``.
    B has infinitely many minimizers with ``x_1 = -pi, pi, 3pi``
    and ``B_min = 0.397887``
    """

    _name = "AugmentedBranin"
    _optimal_value = 0.397887
    _optimizers = [  # this is a subset, there are infinitely many optimizers
        (-math.pi, 12.275, 1),
        (math.pi, 1.3867356039019576, 0.1),
        (math.pi, 1.781519779945532, 0.5),
        (math.pi, 2.1763039559891064, 0.9),
    ]

    def __init__(
        self,
        noise_std: float | None = None,
        negate: bool = False,
        bounds: list[tuple[float, float]] | None = None,
        dtype: Any = None,
    ) -> None:
        r"""
        Args:
            noise_std: Standard deviation of the observation noise.
            negate: If True, negate the function.
            bounds: Custom bounds for the function specified as (lower, upper) pairs.
            dtype: The dtype that is used for the bounds of the function.
        """
        self.continuous_inds = list(range(3))
        self._setup(
            3,
            [(-5.0, 10.0), (0.0, 15.0), (0.0, 1.0)],
            noise_std,
            negate,
            dtype,
            custom_bounds=bounds,
        )


class AugmentedHartmann(_NativeProblem):
    r"""Augmented Hartmann synthetic test function.

    7-dimensional function (typically evaluated on ``[0, 1]^7``), where the last
    dimension is the fidelity parameter.

        H(x) = -(ALPHA_1 - 0.1 * (1-x_7)) * exp(- sum_{j=1}^6 A_1j (x_j - P_1j) ** 2) -
            sum_{i=2}^4 ALPHA_i exp( - sum_{j=1}^6 A_ij (x_j - P_ij) ** 2)

    H has a unique global minimizer
    ``x = [0.20169, 0.150011, 0.476874, 0.275332, 0.311652, 0.6573, 1.0]``

    with ``H_min = -3.32237``
    """

    _name = "AugmentedHartmann"
    _optimal_value = -3.32237
    _optimizers = [(0.20169, 0.150011, 0.476874, 0.275332, 0.311652, 0.6573, 1.0)]
    _check_grad_at_opt = False

    def __init__(
        self,
        noise_std: float | None = None,
        negate: bool = False,
        dtype: Any = None,
    ) -> None:
        r"""
        Args:
            noise_std: Standard deviation of the observation noise.
            negate: If True, negate the function.
            dtype: The dtype that is used for the bounds of the function.
        """
        self.continuous_inds = list(range(7))
        self._setup(7, [(0.0, 1.0)] * 7, noise_std, negate, dtype)


class AugmentedRosenbrock(_NativeProblem):
    r"""Augmented Rosenbrock synthetic test function for multi-fidelity optimization.

    d-dimensional function (usually evaluated on ``[-5, 10]^(d-2) * [0, 1]^2``),
    where the last two dimensions are the fidelity parameters:

        f(x) = sum_{i=1}^{d-1} (100 (x_{i+1} - x_i^2 + 0.1 * (1-x_{d-1}))^2 +
            (x_i - 1 + 0.1 * (1 - x_d)^2)^2)

    f has one minimizer for its global minimum at ``z_1 = (1, 1, ..., 1)`` with
    ``f(z_i) = 0.0``.
    """

    _name = "AugmentedRosenbrock"
    _optimal_value = 0.0

    def __init__(
        self,
        dim: int = 3,
        noise_std: float | None = None,
        negate: bool = False,
        dtype: Any = None,
    ) -> None:
        r"""
        Args:
            dim: The (input) dimension. Must be at least 3.
            noise_std: Standard deviation of the observation noise.
            negate: If True, negate the function.
            dtype: The dtype that is used for the bounds of the function.
        """
        if dim < 3:
            raise ValueError("AugmentedRosenbrock must be defined in at least 3 dimensions")
        self.continuous_inds = list(range(dim))
        self._optimizers = [tuple(1.0 for _ in range(dim))]
        bounds = [(-5.0, 10.0) for _ in range(dim - 2)] + [(0.0, 1.0) for _ in range(2)]
        self._setup(dim, bounds, noise_std, negate, dtype)


class _CostMixin:
    _rs: Any

    def cost(self, X: Any) -> Any:
        """The evaluation cost of the fidelity of every point of ``X`` (0
        for a fidelity value that is not one of ``fidelities``)."""
        data, shape = to_flat(X, "X")
        return from_pair(self._rs.cost((data, shape)), kind_of(X))


class WingWeightMultiFidelity(_CostMixin, _NativeProblem):
    """Wing Weight Design Problem from [Chen2024]_.

    Design variables (physical units):
       1. s_w        in [150,    200]  (wing area)
       2. w_fw       in [220,    300]  (fuel weight)
       3. A          in [6,       10]  (aspect ratio)
       4. Lambda_deg in [-10,     10]  (sweep angle, degrees)
       5. q          in [16,      45]  (dynamic pressure)
       6. lam        in [0.5,    1.0]  (taper ratio)
       7. t_c        in [0.08,  0.18]   (thickness-to-chord)
       8. N_z        in [2.5,    6.0]  (ultimate load factor)
       9. w_dg       in [1700,  2500]  (design gross weight)
       10. w_pp      in [0.025, 0.08]  (weight per unit area)

    Fidelity parameter (stored as the 11th input):
      0: High fidelity (HF)
      1: Low fidelity 1 (LF1)
      2: Low fidelity 2 (LF2)
      3: Low fidelity 3 (LF3)

    LF models use slightly altered exponents and additive biases. As in
    BoTorch, a fidelity value that is not one of ``fidelities`` gives 0.
    """

    _name = "WingWeightMultiFidelity"
    _num_fidelities = 1
    fidelities = [0, 1, 2, 3]
    _optimal_value = 123.25

    def __init__(
        self,
        noise_std: float | None = None,
        negate: bool = False,
        bounds: list[tuple[float, float]] | None = None,
        dtype: Any = None,
    ) -> None:
        self.continuous_inds = list(range(11))
        self._setup(
            11,
            [
                (150.0, 200.0),  # s_w
                (220.0, 300.0),  # w_fw
                (6.0, 10.0),  # A
                (-10.0, 10.0),  # Lambda_deg
                (16.0, 45.0),  # q
                (0.5, 1.0),  # lam
                (0.08, 0.18),  # t_c
                (2.5, 6.0),  # N_z
                (1700.0, 2500.0),  # w_dg
                (0.025, 0.08),  # w_pp
                (0.0, 3.0),  # fidelity
            ],
            noise_std,
            negate,
            dtype,
            custom_bounds=bounds,
        )


class BoreholeMultiFidelity(_CostMixin, _NativeProblem):
    """Borehole Problem from [Chen2024]_.

    This problem models water flow through a borehole with 8 design variables:
      1. r_w   in [0.05,   0.15]   (borehole radius)
      2. r     in [100,    50000]  (radius of influence)
      3. T_u   in [63070,  115600] (transmissivity of upper aquifer)
      4. T_l   in [63.1,   116]    (transmissivity of lower aquifer)
      5. H_u   in [990,    1110]   (potentiometric head of upper aquifer)
      6. H_l   in [700,    820]    (potentiometric head of lower aquifer)
      7. L     in [1120,   1680]   (length of borehole)
      8. K_w   in [9855,   12045]  (hydraulic conductivity)

    The fidelity index (9th input) is categorical:
      0: High fidelity (HF)
      1: Low fidelity 1 (LF1)
      2: Low fidelity 2 (LF2)
      3: Low fidelity 3 (LF3)
      4: Low fidelity 4 (LF4)


    The low-fidelity models modify exponents and add a bias.
    """

    _name = "BoreholeMultiFidelity"
    _num_fidelities = 1
    fidelities = [0, 1, 2, 3, 4]
    _optimal_value = 3.98

    def __init__(
        self,
        noise_std: float | None = None,
        negate: bool = False,
        bounds: list[tuple[float, float]] | None = None,
        dtype: Any = None,
    ) -> None:
        self.continuous_inds = list(range(9))
        self._setup(
            9,
            [
                (0.05, 0.15),  # r_w
                (100.0, 10000.0),  # r
                (100.0, 1000.0),  # T_u
                (10.0, 500.0),  # T_l
                (990.0, 1110.0),  # H_u
                (700.0, 820.0),  # H_l
                (1000.0, 2000.0),  # L
                (6000.0, 12000.0),  # K_w
                (0.0, 4.0),  # fidelity
            ],
            noise_std,
            negate,
            dtype,
            custom_bounds=bounds,
        )


__all__ = [
    "AugmentedBranin",
    "AugmentedHartmann",
    "AugmentedRosenbrock",
    "BoreholeMultiFidelity",
    "WingWeightMultiFidelity",
]

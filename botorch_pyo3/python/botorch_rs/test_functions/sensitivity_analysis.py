"""Test functions of global sensitivity analysis
(``botorch.test_functions.sensitivity_analysis``): ``Ishigami``, ``Gsobol``
and ``Morris``, evaluated in Rust."""

from __future__ import annotations

import math
import struct
import sys
from typing import Any

from botorch_rs._arrays import LIST, ArrayKind, from_flat, to_flat
from botorch_rs.test_functions.synthetic import _NativeProblem


def _f32(v: float) -> float:
    """``v`` rounded to float32."""
    return float(struct.unpack("f", struct.pack("f", v))[0])


def _float32_kind() -> ArrayKind:
    """The kind of BoTorch's ``Tensor(...)`` attributes: float32 tensors
    (torch's default dtype) if torch has been imported, else lists."""
    torch = sys.modules.get("torch")
    if torch is None:
        return LIST
    return ArrayKind("torch", torch.get_default_dtype())


class Ishigami(_NativeProblem):
    r"""Ishigami test function.

    three-dimensional function (usually evaluated on ``[-pi, pi]^3``):

        f(x) = sin(x_1) + a sin(x_2)^2 + b x_3^4 sin(x_1)

    Here ``a`` and ``b`` are constants where a=7 and b=0.1 or b=0.05
    Proposed to test sensitivity analysis methods because it exhibits strong
    nonlinearity and nonmonotonicity and a peculiar dependence on x_3.
    """

    _name = "Ishigami"

    def __init__(
        self,
        b: float = 0.1,
        noise_std: float | None = None,
        negate: bool = False,
        dtype: Any = None,
    ) -> None:
        r"""
        Args:
            b: the b constant, should be 0.1 or 0.05.
            noise_std: Standard deviation of the observation noise.
            negative: If True, negative the objective.
            dtype: The dtype that is used for the bounds of the function.
        """
        self._optimizers = None
        if b not in (0.1, 0.05):
            raise ValueError("b parameter should be 0.1 or 0.05")
        self.continuous_inds = list(range(3))
        if b == 0.1:
            self.si = [0.3138, 0.4424, 0]
            self.si_t = [0.558, 0.442, 0.244]
            self.s_ij = [0, 0.244, 0]
            self.dgsm_gradient = [-0.0004, -0.0004, -0.0004]
            self.dgsm_gradient_abs = [1.9, 4.45, 1.97]
            self.dgsm_gradient_square = [7.7, 24.5, 11]
        elif b == 0.05:
            self.si = [0.218, 0.687, 0]
            self.si_t = [0.3131, 0.6868, 0.095]
            self.s_ij = [0, 0.094, 0]
            self.dgsm_gradient = [-0.0002, -0.0002, -0.0002]
            self.dgsm_gradient_abs = [1.26, 4.45, 1.97]
            self.dgsm_gradient_square = [2.8, 24.5, 11]
        self.b = b
        self._setup(
            3, [(-math.pi, math.pi) for _ in range(3)], noise_std, negate, dtype, params=[b]
        )

    @property
    def _optimal_value(self) -> float:
        raise NotImplementedError

    @property
    def optimal_value(self) -> float:
        raise NotImplementedError

    def compute_dgsm(self, X: Any) -> tuple[list[float], list[float], list[float]]:
        r"""Compute derivative global sensitivity measures.

        This function can be called separately to estimate the dgsm measure
        The exact global integrals of these values are already added under
        as attributes dgsm_gradient, dgsm_gradient_abs, and dgsm_gradient_square.

        Args:
            X: Set of points at which to compute derivative measures.

        Returns: The average gradient, absolute gradient, and square gradients.
        """
        data, shape = to_flat(X, "X")
        n = len(data) // 3
        rows = [[float(data[3 * i + k]) for k in range(3)] for i in range(n)]
        dx_1 = [math.cos(x[0]) * (1 + self.b * (x[2] ** 4)) for x in rows]
        dx_2 = [14 * math.cos(x[1]) * math.sin(x[1]) for x in rows]
        dx_3 = [0.4 * (x[2] ** 3) * math.sin(x[0]) for x in rows]
        grads = (dx_1, dx_2, dx_3)
        gradient_measure = [math.fsum(g) / n for g in grads]
        gradient_absolute_measure = [math.fsum(abs(v) for v in g) / n for g in grads]
        gradient_square_measure = [math.fsum(v * v for v in g) / n for g in grads]
        return gradient_measure, gradient_absolute_measure, gradient_square_measure


class Gsobol(_NativeProblem):
    r"""Gsobol test function.

    d-dimensional function (usually evaluated on ``[0, 1]^d``):

        f(x) = Prod_{i=1}\^{d} ((\|4x_i-2\|+a_i)/(1+a_i)), a_i >=0

    common combinations of dimension and a vector:

        dim=8, a= [0, 1, 4.5, 9, 99, 99, 99, 99]
        dim=6, a=[0, 0.5, 3, 9, 99, 99]
        dim = 15, a= [1, 2, 5, 10, 20, 50, 100, 500, 1000, ..., 1000]

    Proposed to test sensitivity analysis methods
    First order Sobol indices have closed form expression S_i=V_i/V with :

        V_i= 1/(3(1+a_i)\^2)
        V= Prod_{i=1}\^{d} (1+V_i) - 1

    The attributes ``vi``, ``V``, ``si`` and ``si_t`` are computed in float32
    (torch's default dtype), operation by operation as BoTorch computes them.
    """

    _name = "Gsobol"

    def __init__(
        self,
        dim: int,
        a: list[float] | None = None,
        noise_std: float | None = None,
        negate: bool = False,
        dtype: Any = None,
    ) -> None:
        r"""
        Args:
            dim: Dimensionality of the problem. If 6, 8, or 15, will use standard a.
            a: a parameter, unless dim is 6, 8, or 15.
            noise_std: Standard deviation of observation noise.
            negate: Return negative of function.
            dtype: The dtype that is used for the bounds of the function.
        """
        self._optimizers = None
        self.continuous_inds = list(range(dim))
        if dim == 6:
            self.a = [0, 0.5, 3, 9, 99, 99]
        elif dim == 8:
            self.a = [0, 1, 4.5, 9, 99, 99, 99, 99]
        elif dim == 15:
            self.a = [1, 2, 5, 10, 20, 50, 100, 500, 1000] + [1000] * 6
        else:
            self.a = a  # type: ignore[assignment]
        self._dim_ = dim
        self.optimal_sobol_indicies()
        self._setup(
            dim,
            [(0.0, 1.0) for _ in range(dim)],
            noise_std,
            negate,
            dtype,
            params=[float(v) for v in self.a],
        )

    @property
    def _optimal_value(self) -> float:
        raise NotImplementedError

    @property
    def optimal_value(self) -> float:
        raise NotImplementedError

    def optimal_sobol_indicies(self) -> None:
        """Computes the Sobol indices ``vi``, ``V``, ``si`` and ``si_t``."""
        dim = self._dim_
        vi = [_f32(1 / (3 * ((1 + self.a[i]) ** 2))) for i in range(dim)]
        one_plus = [_f32(1 + v) for v in vi]
        prod = 1.0
        for v in one_plus:
            prod = _f32(prod * v)
        V = _f32(prod - 1)
        si = [_f32(v / V) for v in vi]
        si_t = []
        for i in range(dim):
            p_before = 1.0
            for v in one_plus[:i]:
                p_before = _f32(p_before * v)
            p_after = 1.0
            for v in one_plus[i + 1 :]:
                p_after = _f32(p_after * v)
            si_t.append(_f32(_f32(_f32(vi[i] * p_before) * p_after) / V))
        kind = _float32_kind()
        self.vi = from_flat(vi, (dim,), kind)
        self.V = from_flat([V], (), kind)
        self.si = from_flat(si, (dim,), kind)
        self.si_t = from_flat(si_t, (dim,), kind)


class Morris(_NativeProblem):
    r"""Morris test function.

    20-dimensional function (usually evaluated on ``[0, 1]^20``):

        f(x) = sum_{i=1}\^20 beta_i w_i + sum_{i<j}\^20 beta_ij w_i w_j
        + sum_{i<j<l}\^20 beta_ijl w_i w_j w_l + 5w_1 w_2 w_3 w_4

    Proposed to test sensitivity analysis methods
    """

    _name = "Morris"

    def __init__(
        self,
        noise_std: float | None = None,
        negate: bool = False,
        dtype: Any = None,
    ) -> None:
        r"""
        Args:
            noise_std: Standard deviation of observation noise.
            negate: Return negative of function.
            dtype: The dtype that is used for the bounds of the function.
        """
        self._optimizers = None
        self.continuous_inds = list(range(20))
        self.si = [
            0.005,
            0.008,
            0.017,
            0.009,
            0.016,
            0,
            0.069,
            0.1,
            0.15,
            0.1,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
        ]
        self._setup(20, [(0.0, 1.0) for _ in range(20)], noise_std, negate, dtype)

    @property
    def _optimal_value(self) -> float:
        raise NotImplementedError

    @property
    def optimal_value(self) -> float:
        raise NotImplementedError


__all__ = ["Gsobol", "Ishigami", "Morris"]

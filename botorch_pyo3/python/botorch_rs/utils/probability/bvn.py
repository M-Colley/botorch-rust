r"""
Methods for computing bivariate normal probabilities and statistics
(``botorch.utils.probability.bvn``), using device-resident Torch expressions
with higher derivatives for Tensor inputs and native Rust for NumPy/lists.
Both routes implement Genz's polar/Taylor algorithms.

One difference to BoTorch: the number of quadrature points of the polar
integration (6, 12 or 20) is chosen from the correlation of each pair; BoTorch
chooses it from the largest correlation of all pairs of a call, so its value
for one pair can depend on the others (at the level of the quadrature error,
~5e-16). BoTorch's NaN gradients where the probability is clipped (e.g.
``_bvnu_taylor`` with a lower bound of ``-1e6``) are finite here.
The Tensor route also retains the nonzero correlation derivative at r=0,
which the reference's independent-probability shortcut drops.

.. [Genz2004bvnt]
    A. Genz. Numerical computation of rectangular bivariate and trivariate normal and
    t probabilities. Statistics and Computing, 2004.

.. [Muthen1990moments]
    B. Muthen. Moments of the censored and truncated bivariate normal distribution.
    British Journal of Mathematical and Statistical Psychology, 1990.
"""

from __future__ import annotations

from typing import Any

from botorch_rs import _ops
from botorch_rs._arrays import first_kind
from botorch_rs.exceptions.errors import UnsupportedError


def _same_shape(name: str, *xs: Any) -> None:
    shapes = [_ops.shape_of(x) for x in xs]
    if any(s != shapes[0] for s in shapes):
        raise UnsupportedError(f"Arguments to `{name}` must have the same shape.")


def bvn(r: Any, xl: Any, yl: Any, xu: Any, yu: Any) -> Any:
    r"""A function for computing bivariate normal probabilities.

    Calculates ``P(xl < x < xu, yl < y < yu)`` where ``x`` and ``y`` are bivariate
    normal with unit variance and correlation coefficient ``r``. See Section 2.4 of
    [Genz2004bvnt]_.

    This method uses a sign flip trick to improve numerical performance. Many of
    ``bvnu``s internal branches rely on evaluations ``Phi(-bound)``. For
    ``a < b < 0``, the term ``Phi(-a) - Phi(-b)`` goes to zero faster than
    ``Phi(b) - Phi(a)`` because ``finfo(dtype).epsneg`` is typically much larger
    than ``finfo(dtype).tiny``. In these cases, flipping the sign can prevent
    situations where ``bvnu(...) - bvnu(...)`` would otherwise be zero due to
    round-off error.

    Args:
        r: Tensor of correlation coefficients.
        xl: Tensor of lower bounds for ``x``, same shape as ``r``.
        yl: Tensor of lower bounds for ``y``, same shape as ``r``.
        xu: Tensor of upper bounds for ``x``, same shape as ``r``.
        yu: Tensor of upper bounds for ``y``, same shape as ``r``.

    Returns:
        Tensor of probabilities ``P(xl < x < xu, yl < y < yu)``.

    """
    _same_shape("bvn", r, xl, xu, yl, yu)
    kind = first_kind(r, xl, yl, xu, yu)
    if kind.lib == "torch":
        from botorch_rs.utils.probability import _torch_bvn

        return _torch_bvn.bvn(*(_ops._as_array(x, kind) for x in (r, xl, yl, xu, yu)))
    return _ops.apply("bvn", r, xl, yl, xu, yu)


def bvnu(r: Any, h: Any, k: Any) -> Any:
    r"""Solves for ``P(x > h, y > k)`` where ``x`` and ``y`` are standard bivariate
    normal random variables with correlation coefficient ``r``. In [Genz2004bvnt]_,
    this is (1)

        `L(h, k, r) = P(x < -h, y < -k) \
        = 1/(a 2\pi) \int_{h}^{\infty} \int_{k}^{\infty} f(x, y, r) dy dx,`

    where ``f(x, y, r) = e^{-1/(2a^2) (x^2 - 2rxy + y^2)}`` and ``a = (1 - r^2)^{1/2}``.

    [Genz2004bvnt]_ report the following integation scheme incurs a maximum of 5e-16
    error when run in double precision: if ``|r| >= 0.925``, use a 20-point quadrature
    rule on a 5th order Taylor expansion; else, numerically integrate in polar
    coordinates using no more than 20 quadrature points.

    Args:
        r: Tensor of correlation coefficients.
        h: Tensor of negative upper bounds for ``x``, same shape as ``r``.
        k: Tensor of negative upper bounds for ``y``, same shape as ``r``.

    Returns:
        A tensor of probabilities ``P(x > h, y > k)``.
    """
    _same_shape("bvnu", r, h, k)
    kind = first_kind(r, h, k)
    if kind.lib == "torch":
        from botorch_rs.utils.probability import _torch_bvn

        return _torch_bvn.bvnu(*(_ops._as_array(x, kind) for x in (r, h, k)))
    return _ops.apply("bvnu", r, h, k)


def bvnmom(
    r: Any,
    xl: Any,
    yl: Any,
    xu: Any,
    yu: Any,
    p: Any = None,
) -> tuple[Any, Any]:
    r"""Computes the expected values of truncated, bivariate normal random variables.

    Let ``x`` and ``y`` be a pair of standard bivariate normal random variables having
    correlation ``r``. This function computes ``E([x,y] \| [xl,yl] < [x,y] < [xu,yu])``.

    Following [Muthen1990moments]_ equations (4) and (5), we have

        `E(x \| [xl, yl] < [x, y] < [xu, yu]) \
        = Z^{-1} \phi(xl) P(yl < y < yu \| x=xl) - \phi(xu) P(yl < y < yu \| x=xu),`

    where ``Z = P([xl, yl] < [x, y] < [xu, yu])`` and ``\phi`` is the standard
    normal PDF.

    Args:
        r: Tensor of correlation coefficients.
        xl: Tensor of lower bounds for ``x``, same shape as ``r``.
        xu: Tensor of upper bounds for ``x``, same shape as ``r``.
        yl: Tensor of lower bounds for ``y``, same shape as ``r``.
        yu: Tensor of upper bounds for ``y``, same shape as ``r``.
        p: Tensor of probabilities ``P(xl < x < xu, yl < y < yu)``, same shape as ``r``.

    Returns:
        ``E(x \| [xl, yl] < [x, y] < [xu, yu])`` and
        ``E(y \| [xl, yl] < [x, y] < [xu, yu])``.
    """
    _same_shape("bvn", r, xl, xu, yl, yu)
    if p is None:
        p = bvn(r=r, xl=xl, xu=xu, yl=yl, yu=yu)
    kind = first_kind(r, xl, yl, xu, yu, p)
    if kind.lib == "torch":
        from botorch_rs.utils.probability import _torch_bvn

        return _torch_bvn.bvnmom(*(_ops._as_array(x, kind) for x in (r, xl, yl, xu, yu, p)))
    out = _ops.apply("bvnmom", r, xl, yl, xu, yu, p)
    return out[0], out[1]


__all__ = ["bvn", "bvnmom", "bvnu"]

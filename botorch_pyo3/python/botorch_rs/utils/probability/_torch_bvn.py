"""Genz bivariate Gaussian quadrature with complete device-resident graphs.

Formulas follow Genz (2004) and the MIT-licensed BoTorch implementation.
Branches are evaluated on their active entries to avoid inactive singular
expressions contaminating higher derivatives.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor

from botorch_rs.utils.probability.utils import STANDARDIZED_RANGE, leggauss
from botorch_rs.utils.safe_math import exp as safe_exp
from botorch_rs.utils.safe_math import mul as safe_mul
from botorch_rs.utils.safe_math import sub as safe_sub


def _cdf(x: Tensor) -> Tensor:
    finite = x.masked_fill(x.isinf(), 0)
    value = 0.5 * torch.special.erfc(-(2**-0.5) * finite)
    return torch.where(x.isneginf(), 0.0, torch.where(x.isposinf(), 1.0, value))


def _density(x: Tensor) -> Tensor:
    infinite = x.isinf()
    finite = x.masked_fill(infinite, 0)
    return ((-0.5 * finite.square()).exp() / math.sqrt(2 * math.pi)).masked_fill(infinite, 0)


def _polar(r: Tensor, h: Tensor, k: Tensor, num_points: int) -> Tensor:
    nodes, weights = leggauss(num_points, dtype=r.dtype, device=r.device)
    angle = 0.5 * r.asin()
    sine = (angle.unsqueeze(-1) * (nodes + 1)).sin()
    hq, kq = h.unsqueeze(-1), k.unsqueeze(-1)
    exponent = (sine * hq * kq - 0.5 * (hq.square() + kq.square())) / (1 - sine.square())
    integral = angle * (safe_exp(exponent) @ weights) / (2 * math.pi)
    return (_cdf(-h) * _cdf(-k) + integral).clamp(0, 1)


def _taylor(r: Tensor, h: Tensor, k: Tensor) -> Tensor:
    nodes, weights = leggauss(20, dtype=r.dtype, device=r.device)
    nodes = nodes + 1
    sign = torch.where(r > 0, 1.0, -1.0)
    sk = sign * k
    skh = sk * h
    complement = 1 - r.square()
    a = complement.sqrt()
    b = h - sk
    b2 = b.square()
    c, d = (4 - skh) / 8, (12 - skh) / 80
    endpoint = torch.where(r > 0, _cdf(-torch.maximum(h, k)), (_cdf(sk) - _cdf(h)).clamp_min(0))
    a0 = -0.5 * (b2 / complement + skh)
    a1 = c * (1 - d * b2) / 3
    a2 = 1 - b2 * a1
    analytic = a * (a2 + complement * a1 + c * d * complement.square()) * safe_exp(a0) - math.sqrt(
        2 * math.pi
    ) * _cdf(-b.abs() / a) * b.abs() * a2 * safe_exp(-0.5 * skh)
    q0 = 0.25 * complement.unsqueeze(-1) * nodes.square()
    q1 = (1 - q0).sqrt()
    q2 = -0.5 * (b2.unsqueeze(-1) / q0 + skh.unsqueeze(-1))
    t1 = 1 + c.unsqueeze(-1) * q0 * (1 + 5 * d.unsqueeze(-1) * q0)
    t2 = safe_exp(-0.5 * q0 / (1 + q1).square() * skh.unsqueeze(-1)) / q1
    integrand = (safe_exp(q2.masked_fill(q2 <= -100, -100)) * (t1 - t2)).masked_fill(q2 <= -100, 0)
    quadrature = -0.5 * a * (integrand @ weights)
    return (endpoint - sign * (analytic + quadrature) / (2 * math.pi)).clamp(0, 1)


def bvnu(r: Tensor, h: Tensor, k: Tensor) -> Tensor:
    """P(X > h,Y > k), retaining correlation sensitivity at independence."""
    if bool((~r.isfinite() | (r.abs() > 1)).any()):
        raise ValueError("Correlation coefficients must be finite and lie in [-1, 1].")
    lower, upper = STANDARDIZED_RANGE
    x_free, y_free = h < lower, k < lower
    zero = (h > upper) | (k > upper)
    free = x_free | y_free
    out = torch.zeros_like(r)
    both = ~zero & x_free & y_free
    out = out.masked_fill(both, 1)
    mask = ~zero & x_free & ~y_free
    if bool(mask.any()):
        out[mask] = _cdf(-k[mask])
    mask = ~zero & y_free & ~x_free
    if bool(mask.any()):
        out[mask] = _cdf(-h[mask])
    active = ~zero & ~free
    singular = active & (r.abs() == 1)
    if bool(singular.any()):
        rr, hh, kk = r[singular], h[singular], k[singular]
        out[singular] = torch.where(
            rr > 0, _cdf(-torch.maximum(hh, kk)), (_cdf(-kk) - _cdf(hh)).clamp_min(0)
        )
    high = active & (r.abs() >= 0.925) & ~singular
    if bool(high.any()):
        out[high] = _taylor(r[high], h[high], k[high])
    for lo, hi, points in ((0.0, 0.3, 6), (0.3, 0.75, 12), (0.75, 0.925, 20)):
        mask = active & (r.abs() >= lo) & (r.abs() < hi)
        if bool(mask.any()):
            # In particular, evaluate this expression at r=0. The shortcut
            # Phi(-h)*Phi(-k) has the right value but drops the nonzero dr.
            out[mask] = _polar(r[mask], h[mask], k[mask], points)
    # Preserve zero first/second derivatives for entirely constant free-bound
    # cases without multiplying infinite bounds by zero.
    connected_zero = (
        r.square() + h.clamp(lower, upper).square() + k.clamp(lower, upper).square()
    ) * 0
    return out + connected_zero


def bvn(r: Tensor, xl: Tensor, yl: Tensor, xu: Tensor, yu: Tensor) -> Tensor:
    flip_x, flip_y = xl.abs() > xu, yl.abs() > yu
    flip = (flip_x & (~flip_y | yu.isinf())) | (flip_y & (~flip_x | xu.isinf()))
    sign = 1 - 2 * flip.to(r.dtype)
    xl, xu, yl, yu = sign * xl, sign * xu, sign * yl, sign * yu
    return (bvnu(r, xl, yl) - bvnu(r, xl, yu) - bvnu(r, xu, yl) + bvnu(r, xu, yu)).clamp(0, 1)


def bvnmom(
    r: Tensor, xl: Tensor, yl: Tensor, xu: Tensor, yu: Tensor, p: Tensor
) -> tuple[Tensor, Tensor]:
    singular = r.abs() == 1
    correlation = r.masked_fill(singular, 0)[..., None, None]
    inverse_std = (1 - correlation.square()).rsqrt()
    lower, upper = torch.stack((xl, yl), -1), torch.stack((xu, yu), -1)
    bounds = torch.stack((lower, upper), -1)
    deltas = safe_mul(correlation, bounds)
    conditional = _cdf(safe_mul(inverse_std, safe_sub(upper.flip(-1).unsqueeze(-1), deltas)))
    conditional = conditional - _cdf(
        safe_mul(inverse_std, safe_sub(lower.flip(-1).unsqueeze(-1), deltas))
    )
    differences = -(_density(bounds) * conditional).diff().squeeze(-1)
    means = (differences + r.unsqueeze(-1) * differences.flip(-1)) / p.unsqueeze(-1)
    if bool(singular.any()):
        positive = r > 0
        lo = torch.maximum(xl, torch.where(positive, yl, -yu))
        hi = torch.minimum(xu, torch.where(positive, yu, -yl))
        mean = (_density(lo) - _density(hi)) / p
        degenerate = torch.stack((mean, torch.where(positive, mean, -mean)), -1)
        means = torch.where(singular.unsqueeze(-1), degenerate, means)
    return means[..., 0], means[..., 1]

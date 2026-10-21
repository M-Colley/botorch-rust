"""Genz bivariate Gaussian quadrature with complete device-resident graphs.

Formulas follow Genz (2004) and the MIT-licensed BoTorch implementation.
Branches are evaluated on their active entries to avoid inactive singular
expressions contaminating higher derivatives.
"""

from __future__ import annotations

import math
from functools import cache
from typing import Any

import torch
from torch import Tensor

from botorch_rs.utils.probability.utils import STANDARDIZED_RANGE, leggauss
from botorch_rs.utils.safe_math import exp as safe_exp
from botorch_rs.utils.safe_math import mul as safe_mul
from botorch_rs.utils.safe_math import sub as safe_sub


@cache
def _quadrature(num_points: int, dtype: torch.dtype, device: torch.device) -> tuple[Tensor, Tensor]:
    nodes, weights = leggauss(num_points, dtype=dtype, device=device)
    return nodes, weights


def _cdf(x: Tensor) -> Tensor:
    finite = x.masked_fill(x.isinf(), 0)
    value = 0.5 * torch.special.erfc(-(2**-0.5) * finite)
    return torch.where(x.isneginf(), 0.0, torch.where(x.isposinf(), 1.0, value))


def _density(x: Tensor) -> Tensor:
    infinite = x.isinf()
    finite = x.masked_fill(infinite, 0)
    return ((-0.5 * finite.square()).exp() / math.sqrt(2 * math.pi)).masked_fill(infinite, 0)


def _polar(r: Tensor, h: Tensor, k: Tensor, num_points: int) -> Tensor:
    nodes, weights = _quadrature(num_points, r.dtype, r.device)
    angle = 0.5 * r.asin()
    sine = (angle.unsqueeze(-1) * (nodes + 1)).sin()
    hq, kq = h.unsqueeze(-1), k.unsqueeze(-1)
    exponent = (sine * hq * kq - 0.5 * (hq.square() + kq.square())) / (1 - sine.square())
    integral = angle * (safe_exp(exponent) @ weights) / (2 * math.pi)
    return (_cdf(-h) * _cdf(-k) + integral).clamp(0, 1)


def _taylor(r: Tensor, h: Tensor, k: Tensor) -> Tensor:
    nodes, weights = _quadrature(20, r.dtype, r.device)
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
    # If both analytic terms underflow, large polynomial factors still must
    # not be evaluated: their backward can overflow before multiplying by 0.
    analytic_active = a0 > (-105 if r.dtype == torch.float32 else -746)
    aa = a.masked_fill(~analytic_active, 1)
    bb = b.masked_fill(~analytic_active, 0)
    bb2 = bb.square()
    cc, dd = c.masked_fill(~analytic_active, 0), d.masked_fill(~analytic_active, 0)
    comp = complement.masked_fill(~analytic_active, 1)
    skew = skh.masked_fill(~analytic_active, 0)
    a1 = cc * (1 - dd * bb2) / 3
    a2 = 1 - bb2 * a1
    from botorch_rs._torch_math import log_ndtr

    # Combine the Gaussian tail and exponential before exponentiating.
    # Separately these can be zero and overflow while their product is finite.
    tail_exponential = safe_exp(log_ndtr(-bb.abs() / aa) - 0.5 * skew)
    analytic = (
        aa
        * (a2 + comp * a1 + cc * dd * comp.square())
        * safe_exp(a0.masked_fill(~analytic_active, 0))
        - math.sqrt(2 * math.pi) * bb.abs() * a2 * tail_exponential
    ).masked_fill(~analytic_active, 0)
    q0 = 0.25 * complement.unsqueeze(-1) * nodes.square()
    q2 = -0.5 * (b2.unsqueeze(-1) / q0 + skh.unsqueeze(-1))
    active = q2 > -100
    # Underflowed quadrature terms contribute zero. Mask their operands before
    # constructing the polynomial and exponential; masking only the product
    # can leave 0*inf in a backward pass for standardized infinite bounds.
    q0 = q0.masked_fill(~active, 0)
    q1 = (1 - q0).sqrt()
    cq = torch.where(active, c.unsqueeze(-1), 0)
    dq = torch.where(active, d.unsqueeze(-1), 0)
    skhq = torch.where(active, skh.unsqueeze(-1), 0)
    t1 = 1 + cq * q0 * (1 + 5 * dq * q0)
    t2 = safe_exp(-0.5 * q0 / (1 + q1).square() * skhq) / q1
    integrand = (safe_exp(q2.masked_fill(q2 <= -100, -100)) * (t1 - t2)).masked_fill(q2 <= -100, 0)
    quadrature = -0.5 * a * (integrand @ weights)
    return (endpoint - sign * (analytic + quadrature) / (2 * math.pi)).clamp(0, 1)


def _bvnu_value(r: Tensor, h: Tensor, k: Tensor) -> Tensor:
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


def _partials(r: Tensor, h: Tensor, k: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    interior = r.abs() < 1
    corr = r.masked_fill(~interior, 0)
    lower, upper = STANDARDIZED_RANGE
    hh, kk = h.clamp(lower, upper), k.clamp(lower, upper)
    std = (1 - corr.square()).sqrt()
    residual = (hh - corr * kk) / std
    log_density = -0.5 * (residual.square() + kk.square()) - math.log(2 * math.pi) - std.log()
    underflow = log_density < (-105 if r.dtype == torch.float32 else -746)
    safe_corr = corr.masked_fill(underflow, 0)
    safe_h, safe_k = hh.masked_fill(underflow, 0), kk.masked_fill(underflow, 0)
    safe_std = (1 - safe_corr.square()).sqrt()
    safe_residual = (safe_h - safe_corr * safe_k) / safe_std
    density = (
        (-0.5 * (safe_residual.square() + safe_k.square())).exp() / (2 * math.pi * safe_std)
    ).masked_fill(underflow | ~interior, 0)
    gh = -_density(hh) * _cdf((corr * hh - kk) / std)
    gk = -_density(kk) * _cdf((corr * kk - hh) / std)
    positive = r > 0
    positive_h = (h > k).to(r) + 0.5 * (h == k).to(r)
    positive_k = (k > h).to(r) + 0.5 * (h == k).to(r)
    negative_active = (h <= -k).to(r)
    singular_h = -_density(hh) * torch.where(positive, positive_h, negative_active)
    singular_k = -_density(kk) * torch.where(positive, positive_k, negative_active)
    return density, torch.where(interior, gh, singular_h), torch.where(interior, gk, singular_k)


class _BVNU(torch.autograd.Function):
    @staticmethod
    def forward(*args: Any, **kwargs: Any) -> Tensor:
        return _bvnu_value(*args)

    @staticmethod
    def setup_context(ctx: Any, inputs: Any, output: Any) -> None:
        ctx.save_for_backward(*inputs)
        ctx.save_for_forward(*inputs)

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Any) -> Any:
        return tuple(grad_outputs[0] * derivative for derivative in _partials(*ctx.saved_tensors))

    @staticmethod
    def jvp(ctx: Any, *grad_inputs: Any) -> Any:
        return sum(
            derivative * tangent
            for derivative, tangent in zip(_partials(*ctx.saved_tensors), grad_inputs, strict=True)
            if tangent is not None
        )

    @staticmethod
    def vmap(info: Any, in_dims: Any, *args: Any) -> Any:
        batched = [
            argument.unsqueeze(0).expand(info.batch_size, *argument.shape)
            if dim is None
            else argument.movedim(dim, 0)
            for dim, argument in zip(in_dims, args, strict=True)
        ]
        return _BVNU.apply(*batched), 0


def bvnu(r: Tensor, h: Tensor, k: Tensor) -> Tensor:
    """Genz forward with exact analytic partials, including at equal bounds."""
    return _BVNU.apply(r, h, k)


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

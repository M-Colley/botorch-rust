"""Device-resident scalar algebra for public array utilities.

Imported lazily for Tensor arguments. Numerical operations stay in the Torch
runtime and retain its complete graph; no upstream BoTorch code is called.
The native Rust array path remains available without Torch or NumPy.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import torch
from torch import Tensor

from botorch_rs.acquisition import _torch_reductions as reductions

ELEMENTWISE_NAMES = frozenset(
    {
        "ndtr",
        "ndtri",
        "sqrt",
        "sin",
        "cos",
        "log_raw",
        "beta_log_prob",
        "phi",
        "log_phi",
        "log_ndtr",
        "log_erfc",
        "log_erfcx",
        "standard_normal_log_hazard",
        "exp",
        "log",
        "log1mexp",
        "log1pexp",
        "logexpit",
        "sigmoid",
        "log_sigmoid",
        "fat_sigmoid",
        "log_fat_sigmoid",
        "cauchy",
        "add",
        "sub",
        "mul",
        "div",
        "add_raw",
        "sub_raw",
        "mul_raw",
        "div_raw",
        "logplusexp",
        "logdiffexp",
        "log_prob_normal_in",
        "fatmaximum",
        "fatminimum",
        "fatplus",
        "log_fatplus",
        "log_softplus",
        "fatmoid",
        "log_fatmoid",
        "approximate_round",
    }
)


def log1mexp(x: Tensor) -> Tensor:
    near = x > -math.log(2)
    # Mask before evaluating either branch, including at x=0 and -infinity.
    left = x.masked_fill(~near, -1)
    right = x.masked_fill(near, -1)
    return torch.where(near, (-left.expm1()).log(), (-right.exp()).log1p())


def log_erfcx(x: Tensor) -> Tensor:
    positive = x > 0
    # erfcx's direct backward subtracts nearly equal quantities in the tail.
    # The asymptotic expansion avoids that cancellation and its higher-order
    # amplification. All products in this branch are bounded by 1 / cutoff**2.
    cutoff = 8 if x.dtype == torch.float32 else 20
    tail = x > cutoff
    ordinary = x.masked_fill(~positive | tail, 0)
    large = x.masked_fill(~tail, cutoff)
    inverse_square = large.reciprocal().square()
    term = torch.ones_like(large)
    correction = torch.zeros_like(large)
    for order in range(1, 13):
        term = term * (-(2 * order - 1) / 2) * inverse_square
        correction = correction + term
    asymptotic = -large.log() - math.log(math.pi) / 2 + correction.log1p()
    pos = torch.where(tail, asymptotic, torch.special.erfcx(ordinary).log())
    negative = x.masked_fill(positive, 0)
    neg = torch.special.erfc(negative).log() + negative.square()
    return torch.where(positive, pos, neg)


def log_erfc(x: Tensor) -> Tensor:
    positive = x > 0
    pos = x.masked_fill(~positive, 0)
    neg = x.masked_fill(positive, 0)
    return torch.where(positive, log_erfcx(pos) - pos.square(), torch.special.erfc(neg).log())


def log_ndtr(x: Tensor) -> Tensor:
    return log_erfc(-(2**-0.5) * x) - math.log(2)


def log_prob_normal_in(a: Tensor, b: Tensor) -> Tensor:
    if not bool((a < b).all()):
        raise ValueError("Received input tensors a, b for which not all a < b.")
    lower_infinite, upper_infinite = a.isneginf(), b.isposinf()
    unbounded = lower_infinite | upper_infinite
    finite_a, finite_b = a.masked_fill(unbounded, -1), b.masked_fill(unbounded, 1)
    reverse = finite_b.abs() > finite_a.abs()
    lower = torch.where(reverse, -finite_b, finite_a)
    upper = torch.where(reverse, -finite_a, finite_b)
    ordinary = log_ndtr(upper) + log1mexp(log_ndtr(lower) - log_ndtr(upper))
    left = log_ndtr(b.masked_fill(upper_infinite, 0))
    right = log_ndtr(-a.masked_fill(lower_infinite, 0))
    return torch.where(
        lower_infinite & upper_infinite,
        0.0,
        torch.where(lower_infinite, left, torch.where(upper_infinite, right, ordinary)),
    )


def cauchy(x: Tensor) -> Tensor:
    infinite = x.isinf()
    ordinary = x.masked_fill(infinite, 0)
    return torch.where(infinite, 0.0, (1 + ordinary.square()).reciprocal())


def fatmoid(x: Tensor, tau: Tensor | float = 1.0) -> Tensor:
    scaled = x / tau
    center = math.sqrt(1 / 3)
    return torch.where(
        scaled < 0, (2 / 3) * cauchy(scaled - center), 1 - (2 / 3) * cauchy(scaled + center)
    )


def softplus(x: Tensor, tau: Tensor, threshold: float = 20) -> Tensor:
    scaled = x / tau
    high = scaled > threshold
    ordinary = x.to(scaled.dtype).masked_fill(high, 0) / tau.to(scaled.dtype)
    return torch.where(high, x, tau * torch.nn.functional.softplus(ordinary, threshold=threshold))


def log_softplus(x: Tensor, tau: Tensor) -> Tensor:
    tau = tau.to(x)
    upper, lower = (16, -15) if x.dtype == torch.float32 else (32, -35)
    regular = x / tau > lower
    ordinary = x.masked_fill(~regular, 0)
    value = softplus(ordinary, tau, upper).log()
    return torch.where(regular, value, x / tau + tau.log())


def fatmax(x: Tensor, dim: Any, keepdim: bool = False, tau: Any = 1.0, alpha: Any = 2.0) -> Tensor:
    def max_fun(value: Tensor, dim: Any, keepdim: bool = False) -> Tensor:
        infinite = value.isneginf()
        # A point at -infinity below a finite maximum contributes exactly zero.
        # Mask before division and the rational polynomial to keep both the
        # input and temperature derivatives finite, preserving the native fix.
        shifted = -value.masked_fill(infinite, 0) / tau
        weights = reductions._pareto(shifted, alpha=alpha).masked_fill(infinite, 0)
        return tau * weights.sum(dim=dim, keepdim=keepdim).log()

    return reductions._inf_max_helper(max_fun, x, dim, keepdim)


def temperature_reduction(
    name: str, x: Tensor, dim: Any, keepdim: bool, tau: Any, alpha: float = 2.0
) -> Tensor:
    if name == "smooth_amax":
        return reductions.smooth_amax(x, dim=dim, keepdim=keepdim, tau=tau)
    if name == "smooth_amin":
        return -reductions.smooth_amax(-x, dim=dim, keepdim=keepdim, tau=tau)
    if name == "fatmin":
        return -fatmax(-x, dim=dim, keepdim=keepdim, tau=tau, alpha=alpha)
    return fatmax(x, dim=dim, keepdim=keepdim, tau=tau, alpha=alpha)


def elementwise(name: str, xs: Sequence[Tensor], params: Sequence[float]) -> Tensor:
    x = xs[0]
    if name in {"sqrt", "sin", "cos"}:
        return getattr(torch, name)(x)
    if name == "ndtri":
        return torch.special.ndtri(x)
    if name == "ndtr":
        return 0.5 * torch.special.erfc(-(2**-0.5) * x)
    if name == "log_raw":
        return x.log()
    if name == "exp":
        bound = x.new_tensor(math.log(torch.finfo(x.dtype).max) - 1e-4)
        return x.clamp(max=bound).exp()
    if name == "log":
        return x.clamp_min(torch.finfo(x.dtype).tiny).log()
    if name == "phi":
        return (-0.5 * x.square()).exp() / math.sqrt(2 * math.pi)
    if name == "log_phi":
        return -0.5 * x.square() - math.log(2 * math.pi) / 2
    if name in {"log_erfc", "log_erfcx", "log_ndtr"}:
        return {"log_erfc": log_erfc, "log_erfcx": log_erfcx, "log_ndtr": log_ndtr}[name](x)
    if name == "standard_normal_log_hazard":
        return math.log(2 / math.sqrt(2 * math.pi)) - log_erfcx((2**-0.5) * x)
    if name == "log1mexp":
        return log1mexp(x)
    if name == "log1pexp":
        return reductions.log1pexp(x)
    if name in {"logexpit", "log_sigmoid"}:
        return reductions.logexpit(x)
    if name == "sigmoid":
        return torch.sigmoid(x)
    if name == "cauchy":
        return cauchy(x)
    if name in {"fat_sigmoid", "log_fat_sigmoid"}:
        value = fatmoid(x)
        return value.log() if name.startswith("log_") else value
    if name == "beta_log_prob":
        a, b = params
        return torch.distributions.Beta(x.new_tensor(a), x.new_tensor(b)).log_prob(x)
    y = xs[1]
    if name == "add_raw":
        return x + y
    if name == "sub_raw":
        return x - y
    if name == "mul_raw":
        return x * y
    if name == "div_raw":
        return x / y
    if name in {"add", "sub"}:
        dtype = torch.result_type(x, y)
        x, y = x.to(dtype), y.to(dtype)
        invalid = x.isinf() & y.isinf() & ((x != y) if name == "add" else (x == y))
        left, right = x.masked_fill(invalid, 0), y.masked_fill(invalid, 0)
        return left + right if name == "add" else left - right
    if name == "mul":
        # The reference's masked denominator/second factor is broadcast by
        # torch.where before arithmetic, including zero-dimensional tensors.
        dtype = torch.promote_types(x.dtype, y.dtype)
        x, y = x.to(dtype), y.to(dtype)
        invalid = (x.isinf() & (y == 0)) | (y.isinf() & (x == 0))
        return x.masked_fill(invalid, 0) * y.masked_fill(invalid, 0)
    if name == "div":
        dtype = torch.promote_types(x.dtype, y.dtype)
        x, y = x.to(dtype), y.to(dtype)
        invalid = ((x == 0) & (y == 0)) | (x.isinf() & y.isinf())
        result = x.masked_fill(invalid, 0) / y.masked_fill(invalid, 1)
        return torch.where(invalid, torch.where(x != y, -1.0, 1.0), result)
    if name == "logplusexp":
        return reductions.logsumexp(torch.stack(torch.broadcast_tensors(x, y), dim=-1), dim=-1)
    if name == "logdiffexp":
        return y + log1mexp(x - y.masked_fill(y == -torch.inf, 0))
    if name == "log_prob_normal_in":
        return log_prob_normal_in(x, y)
    if name in {"fatmaximum", "fatminimum"}:
        sign = -1 if name == "fatminimum" else 1
        return sign * fatmax(
            torch.stack(torch.broadcast_tensors(sign * x, sign * y), dim=-1),
            dim=-1,
            tau=params[0],
            alpha=params[1],
        )
    if name == "log_softplus":
        return log_softplus(x, y)
    if name in {"fatplus", "log_fatplus"}:
        value = softplus(x, y) + 0.1 * y * cauchy(x / y)
        return value.log() if name == "log_fatplus" else value
    if name in {"fatmoid", "log_fatmoid"}:
        value = fatmoid(x, y)
        return value.log() if name == "log_fatmoid" else value
    if name == "approximate_round":
        offset = x.floor()
        return offset + (((x - offset - 0.5) / y).tanh() + 1) / 2
    raise ValueError(f"Unknown Tensor elementwise operation {name!r}.")


def reduce_rows(name: str, x: Tensor, params: Sequence[float]) -> Tensor:
    if name == "sum":
        return x.sum(dim=-1)
    if name == "mean":
        return x.mean(dim=-1)
    if name == "min":
        return x.min(dim=-1).values
    if name in {"logsumexp", "logmeanexp"}:
        return getattr(reductions, name)(x, dim=-1)
    return temperature_reduction(name, x, -1, False, params[0], params[1] if len(params) > 1 else 2)

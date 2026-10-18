"""The torch operations of the sample-reducing MC acquisition functions
implemented in Python (subclasses of ``SampleReducingMCAcquisitionFunction``
that override ``_sample_forward``): BoTorch's smooth reductions
(``botorch.utils.safe_math``) and smoothed feasibility indicator
(``botorch.utils.objective``), ported from BoTorch (MIT license) so that
values and gradients are BoTorch's. Imported only when such a subclass is
evaluated (it requires torch).
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

import torch
from torch import Tensor

_inv_sqrt_3 = math.sqrt(1 / 3)
#: The default temperature of the smooth approximations.
TAU = 1.0
#: The default exponent of the power decay of ``_pareto``.
ALPHA = 2.0


def log1pexp(x: Tensor) -> Tensor:
    """Numerically accurate evaluation of log(1 + exp(x))."""
    mask = x <= 18
    return torch.where(
        mask,
        (lambda z: z.exp().log1p())(x.masked_fill(~mask, 0)),
        (lambda z: z + (-z).exp())(x.masked_fill(mask, 0)),
    )


def logexpit(X: Tensor) -> Tensor:
    """Computes the logarithm of the expit (a.k.a. sigmoid) function."""
    return -log1pexp(-X)


def _any(x: Tensor, dim: int | tuple[int, ...], keepdim: bool = False) -> Tensor:
    if isinstance(dim, tuple):
        for d in dim:
            x = x.any(dim=d, keepdim=True)
    else:
        x = x.any(dim, keepdim=True)
    return x if keepdim else x.squeeze(dim)


def _inf_max_helper(
    max_fun: Callable[..., Tensor],
    x: Tensor,
    dim: int | tuple[int, ...],
    keepdim: bool,
) -> Tensor:
    """BoTorch's treatment of infinities in smooth approximations of the
    maximum (``logsumexp``, ``fatmax``)."""
    dims = (dim,) if isinstance(dim, int) else tuple(dim)
    if any(x.shape[d] == 0 for d in dims):
        return x.sum(dim=dim, keepdim=keepdim) - torch.inf
    M = x.amax(dim=dim, keepdim=True)
    is_inf_max = torch.logical_and(*torch.broadcast_tensors(M.isinf(), x == M))
    has_inf_max = _any(is_inf_max, dim=dim, keepdim=True)

    y_inf = x.masked_fill(~is_inf_max, 0.0)
    M_no_inf = M.masked_fill(M.isinf(), 0.0)
    y_no_inf = x.masked_fill(has_inf_max, 0.0) - M_no_inf

    res = torch.where(
        has_inf_max,
        y_inf.sum(dim=dim, keepdim=True),
        M_no_inf + max_fun(y_no_inf, dim=dim, keepdim=True),
    )
    return res if keepdim else res.sum(dim=dim)


def logsumexp(x: Tensor, dim: int | tuple[int, ...], keepdim: bool = False) -> Tensor:
    """``logsumexp`` with a well-behaved backward pass at infinities."""
    return _inf_max_helper(torch.logsumexp, x=x, dim=dim, keepdim=keepdim)


def logmeanexp(X: Tensor, dim: int | tuple[int, ...], keepdim: bool = False) -> Tensor:
    """Computes ``log(mean(exp(X), dim=dim, keepdim=keepdim))``."""
    n = X.shape[dim] if isinstance(dim, int) else math.prod(X.shape[i] for i in dim)
    return logsumexp(X, dim=dim, keepdim=keepdim) - math.log(n)


def smooth_amax(
    X: Tensor,
    dim: int | tuple[int, ...] = -1,
    keepdim: bool = False,
    tau: float | Tensor = 1.0,
) -> Tensor:
    """A smooth approximation to ``max(X, dim=dim)`` (``logsumexp(X / tau) *
    tau``)."""
    return logsumexp(X / tau, dim=dim, keepdim=keepdim) * tau


def _pareto(x: Tensor, alpha: float, check: bool = True) -> Tensor:
    if check and (x < 0).any():
        raise ValueError("Argument `x` must be non-negative.")
    alpha = alpha / 2  # so that alpha stands for the power decay
    # choosing beta_0, beta_1 so that first and second derivatives at x = 0 are 1.
    beta_1 = 2 * alpha
    beta_0 = alpha * beta_1
    return (beta_0 / (beta_0 + beta_1 * x + x.square())).pow(alpha)


def fatmax(
    x: Tensor,
    dim: int | tuple[int, ...],
    keepdim: bool = False,
    tau: float | Tensor = TAU,
    alpha: float = ALPHA,
) -> Tensor:
    """A smooth approximation to ``amax(X, dim=dim)`` with a fat tail."""

    def max_fun(x: Tensor, dim: int | tuple[int, ...], keepdim: bool = False) -> Tensor:
        return tau * _pareto(-x / tau, alpha=alpha).sum(dim=dim, keepdim=keepdim).log()

    return _inf_max_helper(max_fun=max_fun, x=x, dim=dim, keepdim=keepdim)


def cauchy(x: Tensor) -> Tensor:
    """A Lorentzian, i.e. an un-normalized Cauchy density function."""
    return 1 / (1 + x.square())


def fatmoid(X: Tensor, tau: float | Tensor = 1.0) -> Tensor:
    """A twice continuously differentiable approximation to the Heaviside
    step function with a fat tail."""
    X = X / tau
    m = _inv_sqrt_3  # this defines the inflection point
    return torch.where(
        X < 0,
        2 / 3 * cauchy(X - m),
        1 - 2 / 3 * cauchy(X + m),
    )


def log_fatmoid(X: Tensor, tau: float | Tensor = 1.0) -> Tensor:
    """The logarithm of the fatmoid."""
    return fatmoid(X, tau=tau).log()


def compute_smoothed_feasibility_indicator(
    constraints: list[Callable[[Tensor], Tensor]],
    samples: Tensor,
    eta: Any,
    log: bool = False,
    fat: Any = False,
) -> Tensor:
    r"""The smoothed feasibility indicator of a list of constraints
    (``sample_shape x batch_shape x q``), BoTorch's
    ``compute_smoothed_feasibility_indicator`` (with a float ``eta`` in the
    samples' dtype): per constraint
    ``log_fatmoid(-c / eta)`` (``fat=True``), ``logexpit(-c / eta)``
    (``fat=False``) or ``log(c)`` (``fat=None``: the constraint returns
    probabilities), summed, and exponentiated unless ``log``."""
    if type(eta) is not Tensor:
        # float64 temperatures (BoTorch's ``torch.full`` rounds a Python float
        # to float32), like botorch_rs's native acquisition functions.
        eta = torch.full((len(constraints),), eta, dtype=samples.dtype)
    if type(fat) is not list:
        fat = [fat] * len(constraints)
    if len(eta) != len(constraints):
        raise ValueError("Number of provided constraints and number of provided etas do not match.")
    if len(fat) != len(constraints):
        raise ValueError("Number of provided constraints and number of provided fats do not match.")
    if not (eta > 0).all():
        raise ValueError("eta must be positive.")
    is_feasible = torch.zeros_like(samples[..., 0])

    for constraint, eta_, fat_ in zip(constraints, eta, fat, strict=True):
        if fat_ is None:
            is_feasible = is_feasible + constraint(samples).log()
        else:
            log_sigmoid = log_fatmoid if fat_ else logexpit
            is_feasible = is_feasible + log_sigmoid(-constraint(samples) / eta_)

    return is_feasible if log else is_feasible.exp()

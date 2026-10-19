r"""
Special implementations of mathematical functions that
solve numerical issues of naive implementations (``botorch.utils.safe_math``).

The functions are evaluated in Rust with BoTorch's formulas (operation by
operation, with float32's constants for float32 arrays) on torch tensors,
NumPy arrays or nested lists; results have the type of the first array
argument. For torch tensors that require gradients, the exact first
derivatives are computed with the values and enter the autograd graph (the
derivatives of PyTorch's autograd of BoTorch's expressions, including the
subgradients of PyTorch >= 2.14 at clamp bounds), also with respect to a
tensor ``tau`` (for ``log_softplus`` including the path through softplus's
``beta = 1 / tau``, which BoTorch's autograd drops).

.. [Maechler2012accurate]
    M. Mächler. Accurately Computing log (1 - exp (-| a|))
        Assessed by the Rmpfr package. Technical report, 2012.
"""

from __future__ import annotations

import importlib
import math
from typing import Any

from botorch_rs import _ops
from botorch_rs._arrays import first_kind, is_torch_tensor, to_flat
from botorch_rs.exceptions.errors import UnsupportedError

_log2 = math.log(2)
_inv_sqrt_3 = math.sqrt(1 / 3)

TAU = 1.0  # default temperature parameter for smooth approximations to non-linearities
ALPHA = 2.0  # default alpha parameter for the asymptotic power decay of _pareto


def _out(result: Any, kwargs: dict[str, Any]) -> Any:
    out = kwargs.pop("out", None)
    if kwargs:
        raise TypeError(f"Unexpected keyword arguments: {sorted(kwargs)}.")
    if out is None:
        return result
    if is_torch_tensor(out):
        out.copy_(result)
    else:
        out[...] = result
    return out


def _tau_requires_grad(tau: Any) -> bool:
    return is_torch_tensor(tau) and bool(tau.requires_grad)


def _scalar(value: Any, name: str) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    data, _ = to_flat(value, name)
    if len(data) != 1:
        raise UnsupportedError(f"botorch_rs supports a scalar `{name}` here.")
    return float(data[0])


def _reduce_tau(
    name: str, x: Any, dim: Any, keepdim: bool, tau: Any, rest: tuple[float, ...], empty: float
) -> Any:
    """The reduction ``name`` with a scalar temperature tensor ``tau`` that
    requires gradients. The reductions are positively homogeneous,
    ``F(c x, c tau) = c F(x, tau)``, so by Euler's relation
    ``dF/dtau = (F - sum_i x_i dF/dx_i) / tau`` (entries with a zero
    derivative, e.g. ``-inf``, contribute nothing; ``0`` for infinite
    ``F``); the value carries this derivative to ``tau`` to first order."""
    torch = importlib.import_module("torch")
    t = _scalar(tau, "tau")
    params = (t, *rest)
    xd = x.detach().requires_grad_(True)
    with torch.enable_grad():
        vd = _ops.reduce(name, xd, dim, True, params=params, empty=empty)
        (g,) = torch.autograd.grad(vd.sum(), xd, allow_unused=True)
    if g is None:
        g = torch.zeros_like(xd)
    dims = _ops.normalize_dims(dim, x.ndim) if x.ndim else ()
    gx = torch.where(g == 0, torch.zeros_like(g), g * xd.detach())
    s = gx.sum(dim=dims, keepdim=True) if dims else gx
    vdd = vd.detach()
    dtau = torch.where(torch.isfinite(vdd), (vdd - s) / t, torch.zeros_like(vdd))
    v = _ops.reduce(name, x, dim, True, params=params, empty=empty)
    tau0 = tau.reshape(())
    out = v + (tau0 - tau0.detach()) * dtau
    if not keepdim and dims:
        out = out.sum(dim=dims)
    return out


def exp(x: Any, **kwargs: Any) -> Any:
    """``exp(x)`` with the argument clipped at ``log(finfo.max) - 1e-4``."""
    return _out(_ops.apply("exp", x), kwargs)


def log(x: Any, **kwargs: Any) -> Any:
    """``log(x)`` with the argument clipped below at ``finfo.tiny``."""
    return _out(_ops.apply("log", x), kwargs)


def add(a: Any, b: Any, **kwargs: Any) -> Any:
    """``a + b`` with ``inf + (-inf) = 0``."""
    return _ops.apply("add", a, b)


def sub(a: Any, b: Any) -> Any:
    """``a - b`` with ``inf - inf = 0``."""
    return _ops.apply("sub", a, b)


def div(a: Any, b: Any) -> Any:
    """``a / b`` with ``0 / 0`` and ``inf / inf`` equal to ``+-1``."""
    return _ops.apply("div", a, b)


def mul(a: Any, b: Any) -> Any:
    """``a * b`` with ``0 * inf = 0``."""
    return _ops.apply("mul", a, b)


def log1mexp(x: Any) -> Any:
    """Numerically accurate evaluation of log(1 - exp(x)) for x < 0.
    See [Maechler2012accurate]_ for details.
    """
    return _ops.apply("log1mexp", x)


def log1pexp(x: Any) -> Any:
    """Numerically accurate evaluation of log(1 + exp(x)).
    See [Maechler2012accurate]_ for details.
    """
    return _ops.apply("log1pexp", x)


def logexpit(X: Any) -> Any:
    """Computes the logarithm of the expit (a.k.a. sigmoid) function."""
    return _ops.apply("logexpit", X)


def logplusexp(a: Any, b: Any) -> Any:
    """Computes log(exp(a) + exp(b)) similar to logsumexp."""
    return _ops.apply("logplusexp", a, b)


def logdiffexp(log_a: Any, log_b: Any) -> Any:
    """Computes log(b - a) accurately given log(a) and log(b).
    Assumes, log_b > log_a, i.e. b > a > 0.

    Args:
        log_a (Tensor): The logarithm of a, assumed to be less than log_b.
        log_b (Tensor): The logarithm of b, assumed to be larger than log_a.

    Returns:
        A Tensor of values corresponding to log(b - a).
    """
    return _ops.apply("logdiffexp", log_a, log_b)


def logsumexp(x: Any, dim: int | tuple[int, ...], keepdim: bool = False) -> Any:
    """Version of logsumexp that has a well-behaved backward pass when
    x contains infinities.

    In particular, the gradient of the standard torch version becomes NaN
    1) for any element that is positive infinity, and 2) for any slice that
    only contains negative infinities.

    This version returns a gradient of 1 for any positive infinities in case 1, and
    for all elements of the slice in case 2, in agreement with the asymptotic behavior
    of the function.

    Args:
        x: The Tensor to which to apply ``logsumexp``.
        dim: An integer or a tuple of integers, representing the dimensions to reduce.
        keepdim: Whether to keep the reduced dimensions. Defaults to False.

    Returns:
        A Tensor representing the log of the summed exponentials of ``x``.
    """
    return _ops.reduce("logsumexp", x, dim, keepdim)


def logmeanexp(X: Any, dim: int | tuple[int, ...], keepdim: bool = False) -> Any:
    """Computes ``log(mean(exp(X), dim=dim, keepdim=keepdim))``.

    Args:
        X: Values of which to compute the logmeanexp.
        dim: The dimension(s) over which to compute the mean.
        keepdim: If True, keeps the reduced dimensions.

    Returns:
        A Tensor of values corresponding to ``log(mean(exp(X), dim=dim))``.
    """
    shape = tuple(X.shape) if is_torch_tensor(X) else to_flat(X)[1]
    dims = _ops.normalize_dims(dim, len(shape))
    n = math.prod(shape[i] for i in dims) if shape else 1
    math.log(n)  # BoTorch's ValueError for an empty reduction
    return _ops.reduce("logmeanexp", X, dim, keepdim)


def log_softplus(x: Any, tau: Any = TAU) -> Any:
    """Computes the logarithm of the softplus function with high numerical accuracy.

    Args:
        x: Input tensor, should have single or double precision floats.
        tau: Decreasing tau increases the tightness of the
            approximation to ReLU. Non-negative and defaults to 1.0.

    Returns:
        Tensor corresponding to ``log(softplus(x))``.
    """
    check_dtype_float32_or_float64(x)
    return _ops.apply("log_softplus", x, tau, kind=first_kind(x, tau))


def smooth_amax(
    X: Any,
    dim: int | tuple[int, ...] = -1,
    keepdim: bool = False,
    tau: Any = 1.0,
) -> Any:
    """Computes a smooth approximation to ``max(X, dim=dim)``, i.e the maximum
    value of ``X`` over dimension ``dim``, using the logarithm of the
    ``l_(1/tau)`` norm of ``exp(X)``. Note that when ``X = log(U)`` is the
    *logarithm* of an acquisition utility ``U``,

    ``logsumexp(log(U) / tau) * tau = log(sum(U^(1/tau))^tau)``
    ``= log(norm(U, ord=(1/tau))``

    Args:
        X: A Tensor from which to compute the smoothed amax.
        dim: The dimensions to reduce over.
        keepdim: If True, keeps the reduced dimensions.
        tau: Temperature parameter controlling the smooth approximation
            to max operator, becomes tighter as tau goes to 0. Needs to be
            positive.

    Returns:
        A Tensor of smooth approximations to ``max(X, dim=dim)``.
    """
    if not isinstance(tau, (int, float)) and len(to_flat(tau, "tau")[0]) != 1:
        if first_kind(X, tau).lib == "list":
            raise UnsupportedError("A non-scalar `tau` requires torch tensors or NumPy arrays.")
        return logsumexp(X / tau, dim=dim, keepdim=keepdim) * tau
    if _tau_requires_grad(tau):
        return _reduce_tau("smooth_amax", X, dim, keepdim, tau, (), -math.inf)
    return _ops.reduce("smooth_amax", X, dim, keepdim, params=(_scalar(tau, "tau"),))


def smooth_amin(
    X: Any,
    dim: int | tuple[int, ...] = -1,
    keepdim: bool = False,
    tau: Any = 1.0,
) -> Any:
    """A smooth approximation to ``min(X, dim=dim)``, similar to ``smooth_amax``."""
    if not isinstance(tau, (int, float)) and len(to_flat(tau, "tau")[0]) != 1:
        return -smooth_amax(X=-X, dim=dim, keepdim=keepdim, tau=tau)
    if _tau_requires_grad(tau):
        return _reduce_tau("smooth_amin", X, dim, keepdim, tau, (), math.inf)
    return _ops.reduce(
        "smooth_amin", X, dim, keepdim, params=(_scalar(tau, "tau"),), empty=math.inf
    )


def check_dtype_float32_or_float64(X: Any) -> None:
    """Raises ``UnsupportedError`` unless ``X`` has a float32 or float64
    dtype (lists and numbers are float64)."""
    _ops.check_float_dtype(X, "X", UnsupportedError)


def log_fatplus(x: Any, tau: Any = TAU) -> Any:
    """Computes the logarithm of the fat-tailed softplus.

    NOTE: Separated out in case the complexity of the ``log`` implementation increases
    in the future.
    """
    return _ops.apply("log_fatplus", x, tau, kind=first_kind(x, tau))


def fatplus(x: Any, tau: Any = TAU) -> Any:
    """Computes a fat-tailed approximation to ``ReLU(x) = max(x, 0)`` by linearly
    combining a regular softplus function and the density function of a Cauchy
    distribution. The coefficient ``alpha`` of the Cauchy density is chosen to guarantee
    monotonicity and convexity.

    Args:
        x: A Tensor on whose values to compute the smoothed function.
        tau: Temperature parameter controlling the smoothness of the approximation.

    Returns:
        A Tensor of values of the fat-tailed softplus.
    """
    return _ops.apply("fatplus", x, tau, kind=first_kind(x, tau))


def fatmax(
    x: Any,
    dim: int | tuple[int, ...],
    keepdim: bool = False,
    tau: Any = TAU,
    alpha: float = ALPHA,
) -> Any:
    """Computes a smooth approximation to amax(X, dim=dim) with a fat tail.

    Args:
        X: A Tensor from which to compute the smoothed maximum.
        dim: The dimensions to reduce over.
        keepdim: If True, keeps the reduced dimensions.
        tau: Temperature parameter controlling the smooth approximation
            to max operator, becomes tighter as tau goes to 0. Needs to be positive.
        alpha: The exponent of the asymptotic power decay of the approximation. The
            default value is 2. Higher alpha parameters make the function behave more
            similarly to the standard logsumexp approximation to the max, so it is
            recommended to keep this value low or moderate, e.g. < 10.

    Returns:
        A Tensor of smooth approximations to ``amax(X, dim=dim)`` with a fat tail.
    """
    try:
        if _tau_requires_grad(tau):
            return _reduce_tau("fatmax", x, dim, keepdim, tau, (float(alpha),), -math.inf)
        return _ops.reduce("fatmax", x, dim, keepdim, params=(_scalar(tau, "tau"), float(alpha)))
    except ValueError as e:
        raise ValueError(str(e)) from None


def fatmin(
    x: Any,
    dim: int | tuple[int, ...],
    keepdim: bool = False,
    tau: Any = TAU,
    alpha: float = ALPHA,
) -> Any:
    """Computes a smooth approximation to amin(X, dim=dim) with a fat tail.

    Args:
        X: A Tensor from which to compute the smoothed minimum.
        dim: The dimensions to reduce over.
        keepdim: If True, keeps the reduced dimensions.
        tau: Temperature parameter controlling the smooth approximation
            to min operator, becomes tighter as tau goes to 0. Needs to be positive.
        alpha: The exponent of the asymptotic power decay of the approximation. The
            default value is 2. Higher alpha parameters make the function behave more
            similarly to the standard logsumexp approximation to the max, so it is
            recommended to keep this value low or moderate, e.g. < 10.

    Returns:
        A Tensor of smooth approximations to ``amin(X, dim=dim)`` with a fat tail.
    """
    try:
        if _tau_requires_grad(tau):
            return _reduce_tau("fatmin", x, dim, keepdim, tau, (float(alpha),), math.inf)
        return _ops.reduce(
            "fatmin",
            x,
            dim,
            keepdim,
            params=(_scalar(tau, "tau"), float(alpha)),
            empty=math.inf,
        )
    except ValueError as e:
        raise ValueError(str(e)) from None


def fatmaximum(a: Any, b: Any, tau: Any = TAU, alpha: float = ALPHA) -> Any:
    """Computes a smooth approximation to torch.maximum(a, b) with a fat tail.

    Args:
        a: The first Tensor from which to compute the smoothed component-wise maximum.
        b: The second Tensor from which to compute the smoothed component-wise maximum.
        tau: Temperature parameter controlling the smoothness of the approximation. A
            smaller tau corresponds to a tighter approximation that leads to a sharper
            objective landscape that might be more difficult to optimize.
        alpha: The exponent of the asymptotic power decay of the approximation. The
            default value is 2. Higher alpha parameters make the function behave more
            similarly to the standard logsumexp approximation to the max, so it is
            recommended to keep this value low or moderate, e.g. < 10.

    Returns:
        A smooth approximation of torch.maximum(a, b).
    """
    if _tau_requires_grad(tau):
        torch = importlib.import_module("torch")
        return fatmax(
            torch.stack(torch.broadcast_tensors(a, b), dim=-1), dim=-1, tau=tau, alpha=alpha
        )
    return _ops.apply("fatmaximum", a, b, params=(_scalar(tau, "tau"), float(alpha)))


def fatminimum(a: Any, b: Any, tau: Any = TAU, alpha: float = ALPHA) -> Any:
    """Computes a smooth approximation to torch.minimum(a, b) with a fat tail.

    Args:
        a: The first Tensor from which to compute the smoothed component-wise minimum.
        b: The second Tensor from which to compute the smoothed component-wise minimum.
        tau: Temperature parameter controlling the smoothness of the approximation. A
            smaller tau corresponds to a tighter approximation that leads to a sharper
            objective landscape that might be more difficult to optimize.
        alpha: The exponent of the asymptotic power decay of the approximation. The
            default value is 2. Higher alpha parameters make the function behave more
            similarly to the standard logsumexp approximation to the max, so it is
            recommended to keep this value low or moderate, e.g. < 10.

    Returns:
        A smooth approximation of torch.minimum(a, b).
    """
    if _tau_requires_grad(tau):
        return -fatmaximum(-a, -b, tau=tau, alpha=alpha)
    return _ops.apply("fatminimum", a, b, params=(_scalar(tau, "tau"), float(alpha)))


def log_fatmoid(X: Any, tau: Any = 1.0) -> Any:
    """Computes the logarithm of the fatmoid. Separated out in case the implementation
    of the logarithm becomes more complex in the future to ensure numerical stability.
    """
    return _ops.apply("log_fatmoid", X, tau, kind=first_kind(X, tau))


def fatmoid(X: Any, tau: Any = 1.0) -> Any:
    """Computes a twice continuously differentiable approximation to the Heaviside
    step function with a fat tail, i.e. ``O(1 / x^2)`` as ``x`` goes to -inf.

    Args:
        X: A Tensor from which to compute the smoothed step function.
        tau: Temperature parameter controlling the smoothness of the approximation.

    Returns:
        A tensor of fat-tailed approximations to the Heaviside step function.
    """
    return _ops.apply("fatmoid", X, tau, kind=first_kind(X, tau))


def cauchy(x: Any) -> Any:
    """Computes a Lorentzian, i.e. an un-normalized Cauchy density function."""
    return _ops.apply("cauchy", x)


def sigmoid(X: Any, log: bool = False, fat: bool = False) -> Any:
    """A sigmoid function with an optional fat tail and evaluation in log space for
    better numerical behavior. Notably, the fat-tailed sigmoid can be used to remedy
    numerical underflow problems in the value and gradient of the canonical sigmoid.

    Args:
        X: The Tensor on which to evaluate the sigmoid.
        log: Toggles the evaluation of the log sigmoid.
        fat: Toggles the evaluation of the fat-tailed sigmoid.

    Returns:
        A Tensor of (log-)sigmoid values.
    """
    name = ("log_" if log else "") + ("fat_sigmoid" if fat else "sigmoid")
    return _ops.apply(name, X)


__all__ = [
    "ALPHA",
    "TAU",
    "add",
    "cauchy",
    "check_dtype_float32_or_float64",
    "div",
    "exp",
    "fatmax",
    "fatmaximum",
    "fatmin",
    "fatminimum",
    "fatmoid",
    "fatplus",
    "log",
    "log1mexp",
    "log1pexp",
    "log_fatmoid",
    "log_fatplus",
    "log_softplus",
    "logdiffexp",
    "logexpit",
    "logmeanexp",
    "logplusexp",
    "logsumexp",
    "mul",
    "sigmoid",
    "smooth_amax",
    "smooth_amin",
    "sub",
]

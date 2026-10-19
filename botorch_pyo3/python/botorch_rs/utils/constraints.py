"""Helpers for handling input or outcome constraints
(``botorch.utils.constraints``): ``get_outcome_constraint_transforms``,
``get_monotonicity_constraints`` and the parameter constraints
``LogTransformedInterval`` and ``NonTransformedInterval``."""

from __future__ import annotations

import math
import sys
from collections.abc import Callable
from functools import partial
from typing import Any

from botorch_rs import _botorch_rs as _rs
from botorch_rs._arrays import (
    as_float,
    first_kind,
    from_flat,
    is_torch_tensor,
    kind_from_dtype,
    kind_of,
    to_flat,
)
from botorch_rs.constraints import Interval
from botorch_rs.priors import _float32


def _torch_log32(value: float) -> float | None:
    """torch's float32 ``log(value)`` (as GPyTorch computes the raw initial
    value of a constraint created from Python floats), or ``None`` without
    torch or in double-precision mode."""
    torch = sys.modules.get("torch")
    if torch is None:
        return None
    if not _float32():
        return None
    return float(torch.log(torch.tensor(value, dtype=torch.float32)).item())


class NonTransformedInterval(Interval):
    """Modification of the GPyTorch interval class that does not apply
    transformations: the bounds are passed to the optimizer (L-BFGS-B), so
    that parameters can reach the bounds exactly.
    """

    def __init__(
        self,
        lower_bound: Any,
        upper_bound: Any,
        initial_value: Any = None,
    ) -> None:
        """Constructor of the NonTransformedInterval class.

        Args:
            lower_bound: The lower bound of the interval.
            upper_bound: The upper bound of the interval.
            initial_value: The initial value of the parameter.
        """
        super().__init__(
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            transform=None,
            inv_transform=None,
            initial_value=initial_value,
        )


class LogTransformedInterval(Interval):
    """Modification of the GPyTorch interval class.

    The Interval class in GPyTorch will map the parameter to the range [0, 1] before
    applying the inverse transform. LogTransformedInterval skips this step to avoid
    numerical issues, and applies the log transform directly to the parameter values.
    GPyTorch automatically recognizes that the bound constraint have not been applied
    yet, and passes the bounds to the optimizer instead, which then optimizes
    log(parameter) under the constraints log(lower) <= log(parameter) <= log(upper).
    """

    def __init__(
        self,
        lower_bound: float,
        upper_bound: float,
        initial_value: float | None = None,
    ) -> None:
        """Constructor of the LogTransformedInterval class.

        Args:
            lower_bound: The lower bound of the interval.
            upper_bound: The upper bound of the interval.
            initial_value: The initial value of the parameter.
        """
        init = None if initial_value is None else as_float(initial_value, "initial_value")
        f32 = _float32()
        raw = _torch_log32(init) if init is not None and f32 else None
        try:
            self._rs = _rs.Constraint.log_transformed_interval(
                as_float(lower_bound, "lower_bound"),
                as_float(upper_bound, "upper_bound"),
                init,
                f32,
                raw,
            )
        except ValueError as e:
            raise ValueError(str(e)) from None

    @property
    def initial_value_untransformed(self) -> float | None:
        """The initial value in the constrained space."""
        return self._rs.initial_value


def _oc(a: Any, rhs: Any, Y: Any) -> Any:
    r"""Evaluate constraints.

    Args:
        a: ``m``-dim tensor of weights for the outcomes
        rhs: Singleton tensor containing the outcome constraint value
        Y: ``... x b x q x m`` tensor of function values

    Returns:
        A ``... x b x q``-dim tensor where negative values imply feasibility
    """
    if is_torch_tensor(Y):
        torch = sys.modules["torch"]
        a_t = torch.as_tensor(a).to(Y)
        lhs = torch.einsum("...m, m", [Y, a_t])
        return lhs - torch.as_tensor(rhs).to(Y)
    w = [float(v) for v in to_flat(a, "a")[0]]
    r = float(to_flat(rhs, "rhs")[0][0])
    data, shape = to_flat(Y, "Y")
    m = shape[-1]
    vals = [
        math.fsum(float(data[i + k]) * w[k] for k in range(m)) - r for i in range(0, len(data), m)
    ]
    return from_flat(vals, shape[:-1], first_kind(Y, a))


def get_outcome_constraint_transforms(
    outcome_constraints: tuple[Any, Any] | None,
) -> list[Callable[[Any], Any]] | None:
    r"""Create outcome constraint callables from outcome constraint tensors.

    Args:
        outcome_constraints: A tuple of ``(A, b)``. For ``k`` outcome constraints
            and ``m`` outputs at `f(x)``, ``A`` is ``k x m`` and ``b`` is ``k x 1`` such
            that ``A f(x) <= b``.

    Returns:
        A list of callables, each mapping a Tensor of size ``b x q x m`` to a
        tensor of size ``b x q``, where ``m`` is the number of outputs of the model.
        Negative values imply feasibility. The callables support broadcasting
        (e.g. for calling on a tensor of shape ``mc_samples x b x q x m``).

    Example:
        >>> # constrain ``f(x)[0] <= 0``
        >>> A = torch.tensor([[1., 0.]])
        >>> b = torch.tensor([[0.]])
        >>> outcome_constraints = get_outcome_constraint_transforms((A, b))
    """
    if outcome_constraints is None:
        return None
    A, b = outcome_constraints
    if is_torch_tensor(A):
        rows = list(A)
        rhs = list(b)
    else:
        adata, ashape = to_flat(A, "A")
        bdata, _ = to_flat(b, "b")
        k = ashape[0]
        m = ashape[-1]
        kind = kind_of(A)
        rows = [from_flat(list(adata[i * m : (i + 1) * m]), (m,), kind) for i in range(k)]
        rhs = [from_flat([float(bdata[i])], (1,), kind_of(b)) for i in range(k)]
    return [partial(_oc, a, r) for a, r in zip(rows, rhs, strict=True)]


def get_monotonicity_constraints(
    d: int,
    descending: bool = False,
    dtype: Any = None,
    device: Any = None,
) -> tuple[Any, Any]:
    """Returns a system of linear inequalities ``(A, b)`` that generically encodes
    order constraints on the elements of a ``d``-dimsensional space, i.e.
    ``A @ x < b`` implies ``x[i] < x[i + 1]`` for a ``d``-dimensional vector ``x``.

    Idea: Could encode ``A`` as sparse matrix, if it is supported well.

    Args:
        d: Dimensionality of the constraint space, i.e. number of monotonic parameters.
        descending: If True, forces the elements of a vector to be monotonically de-
            creasing and be monotonically increasing otherwise.
        dtype: The dtype of the returned Tensors.
        device: The device of the returned Tensors.

    Returns:
        A tuple of Tensors ``(A, b)`` representing the monotonicity constraint as a
        system of linear inequalities ``A @ x < b``. ``A`` is
        ``(d - 1) x d``-dimensional and ``b`` is ``(d - 1) x 1``-dimensional.

    Without a ``dtype``, the arrays are tensors of torch's default dtype if
    torch has been imported (as in BoTorch), else lists.
    """
    sign = -1.0 if descending else 1.0
    a = [0.0] * ((d - 1) * d)
    for i in range(d - 1):
        a[i * d + i] = sign * 1.0
        a[i * d + i + 1] = sign * -1.0
    torch = sys.modules.get("torch")
    if dtype is None and torch is not None:
        kind = kind_from_dtype(torch.get_default_dtype(), device)
    else:
        kind = kind_from_dtype(dtype, device)
    b = [0.0] * (d - 1)
    if descending:
        a = [v + 0.0 for v in a]  # -0.0 for the zeros of -A, as torch's negation
        a = [-0.0 if (v == 0.0) else v for v in a]
    return from_flat(a, (d - 1, d), kind), from_flat(b, (d - 1, 1), kind)


__all__ = [
    "LogTransformedInterval",
    "NonTransformedInterval",
    "get_monotonicity_constraints",
    "get_outcome_constraint_transforms",
]

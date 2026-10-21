r"""
Helpers for handling objectives (``botorch.utils.objective``).

The functions operate on torch tensors (with torch's operations and
botorch_rs's native ``logexpit`` / ``log_fatmoid``, so that gradients flow),
NumPy arrays and nested lists, with BoTorch's semantics. Constraint callables
receive the samples in the caller's array type.
"""

from __future__ import annotations

import math
import sys
from collections.abc import Callable
from typing import Any

from botorch_rs import _ops
from botorch_rs._arrays import (
    bool_array,
    first_kind,
    from_flat,
    is_torch_tensor,
    kind_of,
    to_flat,
)
from botorch_rs.utils.safe_math import log_fatmoid, logexpit
from botorch_rs.utils.transforms import normalize_indices


def _is_native(x: Any) -> bool:
    np = sys.modules.get("numpy")
    return is_torch_tensor(x) or (np is not None and isinstance(x, np.ndarray))


def get_objective_weights_transform(
    weights: Any,
) -> Callable[[Any, Any], Any]:
    r"""Create a linear objective callable from a set of weights.

    Create a callable mapping a Tensor of size ``b x q x m`` and an (optional)
    Tensor of size ``b x q x d`` to a Tensor of size ``b x q``, where ``m`` is the
    number of outputs of the model using scalarization via the objective weights.
    This callable supports broadcasting (e.g. for calling on a tensor of shape
    ``mc_samples x b x q x m``). For ``m = 1``, the objective weight is used to
    determine the optimization direction.

    Args:
        weights: a 1-dimensional Tensor containing a weight for each task.
            If not provided, the identity mapping is used.

    Returns:
        Transform function using the objective weights.

    Example:
        >>> weights = torch.tensor([0.75, 0.25])
        >>> transform = get_objective_weights_transform(weights)
    """

    def _objective(Y: Any, X: Any = None) -> Any:
        r"""Evaluate objective.

        Args:
            Y: A ``... x b x q x m`` tensor of function values.
            X: Ignored.

        Returns:
            A ``... x b x q``-dim tensor of objective values.
        """
        # if no weights provided, just extract the single output
        if weights is None:
            if _is_native(Y):
                return Y.squeeze(-1)
            data, shape = to_flat(Y, "Y")
            new = shape[:-1] if shape and shape[-1] == 1 else shape
            return from_flat(data, new, kind_of(Y))
        if is_torch_tensor(Y):
            torch = sys.modules["torch"]
            return torch.einsum("...m, m", [Y, torch.as_tensor(weights).to(Y)])
        w = [float(v) for v in to_flat(weights, "weights")[0]]
        data, shape = to_flat(Y, "Y")
        m = shape[-1]
        if m != len(w):
            raise RuntimeError(
                f"einsum(): the number of outputs ({m}) does not match the weights ({len(w)})."
            )
        vals = [
            math.fsum(float(data[i + k]) * w[k] for k in range(m)) for i in range(0, len(data), m)
        ]
        return from_flat(vals, shape[:-1], first_kind(Y, weights))

    return _objective


def _mul_last(w: Any, obj: Any, unsqueeze: bool) -> Any:
    if unsqueeze:
        if _is_native(w):
            w = w[..., None]
        else:
            data, shape = to_flat(w, "w")
            w = from_flat(data, (*shape, 1), kind_of(w))
    return w


def apply_constraints_nonnegative_soft(
    obj: Any,
    constraints: list[Callable[[Any], Any]],
    samples: Any,
    eta: Any,
) -> Any:
    r"""Applies constraints to a non-negative objective.

    This function uses a sigmoid approximation to an indicator function for
    each constraint.

    Args:
        obj: A ``n_samples x b x q (x m')``-dim Tensor of objective values.
        constraints: A list of callables, each mapping a Tensor of size ``b x q x m``
            to a Tensor of size ``b x q``, where negative values imply feasibility.
            This callable must support broadcasting. Only relevant for multi-
            output models (``m`` > 1).
        samples: A ``n_samples x b x q x m`` Tensor of samples drawn from the posterior.
        eta: The temperature parameter for the sigmoid function. Can be either a float
            or a 1-dim tensor. In case of a float the same eta is used for every
            constraint in constraints. In case of a tensor the length of the tensor
            must match the number of provided constraints. The i-th constraint is
            then estimated with the i-th eta value.

    Returns:
        A ``n_samples x b x q (x m')``-dim tensor of feasibility-weighted objectives.
    """
    w = compute_smoothed_feasibility_indicator(constraints=constraints, samples=samples, eta=eta)
    unsqueeze = len(_ops.shape_of(obj)) == len(_ops.shape_of(samples))
    w = _mul_last(w, obj, unsqueeze)
    if _is_native(obj):
        return obj.clip(min=0) * w
    data, shape = to_flat(obj, "obj")
    clamped = from_flat([max(float(v), 0.0) for v in data], shape, kind_of(obj))
    return _ops.arith("mul", clamped, w)


def compute_feasibility_indicator(
    constraints: list[Callable[[Any], Any]] | None,
    samples: Any,
    marginalize_dim: int | None = None,
) -> Any:
    r"""Computes the feasibility of a list of constraints given posterior samples.

    Args:
        constraints: A list of callables, each mapping a batch_shape x q x m`-dim
            Tensor to a ``batch_shape x q``-dim Tensor, where negative values imply
            feasibility.
        samples: A batch_shape x q x m`-dim Tensor of posterior samples.
        marginalize_dim: A batch dimension that should be marginalized.
            For example, this is useful when using a batched fully Bayesian
            model.

    Returns:
        A ``batch_shape x q``-dim tensor of Boolean feasibility values.
    """
    if is_torch_tensor(samples):
        torch = sys.modules["torch"]
        ind_tensor = torch.ones_like(samples[..., 0], dtype=torch.bool)
        for constraint in constraints or []:
            ind_tensor = ind_tensor.logical_and(constraint(samples) <= 0)
        if ind_tensor.ndim >= 3 and marginalize_dim is not None:
            if marginalize_dim < 0:
                normalized = normalize_indices([marginalize_dim], d=ind_tensor.ndim)
                assert normalized is not None
                marginalize_dim = 1 + normalized[0]
            ind_tensor = ind_tensor.float().mean(dim=marginalize_dim).round().bool()
        return ind_tensor
    shape = tuple(_ops.shape_of(samples))[:-1]
    n = math.prod(shape)
    ind = [True] * n
    if constraints is not None:
        for constraint in constraints:
            vals, vshape = to_flat(constraint(samples), "constraint")
            if tuple(vshape) != shape:
                from botorch_rs._arrays import broadcast_to

                vals = broadcast_to(vals, vshape, shape)
            ind = [a and float(v) <= 0 for a, v in zip(ind, vals, strict=True)]
    if len(shape) >= 3 and marginalize_dim is not None:
        # make sure marginalize_dim is not negative
        if marginalize_dim < 0:
            # add 1 to the normalize marginalize_dim since we have already
            # removed the output dim
            idx = normalize_indices([marginalize_dim], d=len(shape))
            assert idx is not None
            marginalize_dim = 1 + idx[0]
        d = marginalize_dim
        outer = math.prod(shape[:d])
        size = shape[d]
        inner = math.prod(shape[d + 1 :])
        new = []
        for o in range(outer):
            for i in range(inner):
                cnt = sum(ind[(o * size + k) * inner + i] for k in range(size))
                # torch's round (half to even) of the mean in float32.
                new.append(round(cnt / size) >= 1)
        ind = new
        shape = shape[:d] + shape[d + 1 :]
    return bool_array(ind, shape, kind_of(samples))


def compute_smoothed_feasibility_indicator(
    constraints: list[Callable[[Any], Any]],
    samples: Any,
    eta: Any,
    log: bool = False,
    fat: list[bool | None] | bool = False,
) -> Any:
    r"""Computes the smoothed feasibility indicator of a list of constraints.

    Given posterior samples, using a sigmoid to smoothly approximate the feasibility
    indicator of each individual constraint to ensure differentiability and high
    gradient signal. The ``fat`` and ``log`` options improve the numerical behavior of
    the smooth approximation.

    NOTE: *Negative* constraint values are associated with feasibility.

    Args:
        constraints: A list of callables, each mapping a Tensor of size ``b x q x m``
            to a Tensor of size ``b x q``. The ``fat`` keyword defines how the callable
            is further processed. By default a sigmoid or fatmoid transformation is
            applied where negative values imply feasibility.
            The applied transformation maps the feasibility indicator of the
            constraint from the interval [-inf, inf] to the interval [0, 1].
            If ``None`` is provided for ``fat``, no transformation is applied and it
            is expected that the constraint callable delivers values in the
            interval [0, 1] without further processing that can be interpreted as
            probabilities of feasibility directly. This is especially useful
            for using classifiers as constraints. The callable must support
            broadcasting. Only relevant for multi-output models (``m`` > 1).
        samples: A ``n_samples x b x q x m`` Tensor of samples drawn from the posterior.
        eta: The temperature parameter for the sigmoid/fatmoid function. Can be either
            a float or a 1-dim tensor. In case of a float the same eta is used for
            every constraint in constraints. In case of a tensor the length of the
            tensor must match the number of provided constraints. The i-th constraint
            is then estimated with the i-th eta value. In case no fatmoid/sigmoid is
            applied, eta is ignored.
        log: Toggles the computation of the log-feasibility indicator.
        fat: Toggles the computation of the fat-tailed feasibility indicator.
            Can be either a list or a boolean. If case of a boolean, the same
            feasibility indicator is used for all constraints. If a list is provided,
            the length of the list must match the number of provided constraints.
            The i-th constraint is then associated with the i-th fat value. In case,
            the i-th fat value is ``None``, no fatmoid/sigmoid transformation is
            applied to the i-th constraint and it is assumed that the constraint
            by itself delivers values in the interval [0, 1]. This is especially useful
            for using classifiers as constraints. If a boolean is provided and its
            value is ``True``, a fatmoid transformation is applied, if its value is
            ``False``, a sigmoid transformation is applied.


    Returns:
        A ``n_samples x b x q``-dim tensor of feasibility indicator values.
    """
    if is_torch_tensor(samples):
        torch = sys.modules["torch"]
        etas = (
            eta
            if is_torch_tensor(eta)
            else torch.full((len(constraints),), eta)
            if isinstance(eta, (int, float))
            else torch.as_tensor(eta)
        )
    elif isinstance(eta, (int, float)):
        etas = [float(eta)] * len(constraints)
    else:
        etas = [float(v) for v in to_flat(eta, "eta")[0]]
    fats = fat if isinstance(fat, list) else [fat] * len(constraints)
    if len(etas) != len(constraints):
        raise ValueError("Number of provided constraints and number of provided etas do not match.")
    if len(fats) != len(constraints):
        raise ValueError("Number of provided constraints and number of provided fats do not match.")
    if not all(e > 0 for e in etas):
        raise ValueError("eta must be positive.")
    shape = tuple(_ops.shape_of(samples))[:-1]
    kind = kind_of(samples)
    if is_torch_tensor(samples):
        is_feasible: Any = sys.modules["torch"].zeros_like(samples[..., 0])
    else:
        is_feasible = from_flat([0.0] * math.prod(shape), shape, kind)
    for constraint, eta_, fat_ in zip(constraints, etas, fats, strict=True):
        c = constraint(samples)
        if fat_ is None:
            term = _log_plain(c)
        else:
            log_sigmoid = log_fatmoid if fat_ else logexpit
            term = log_sigmoid(_ops.arith("div", _neg(c), eta_))
        is_feasible = _ops.arith("add", is_feasible, term)
    if log:
        return is_feasible
    if is_torch_tensor(is_feasible) or _is_native(is_feasible):
        return is_feasible.exp()
    data, shp = to_flat(is_feasible, "is_feasible")
    return from_flat([math.exp(v) for v in data], shp, kind_of(is_feasible))


def _neg(x: Any) -> Any:
    if _is_native(x):
        return -x
    data, shape = to_flat(x, "x")
    return from_flat([-float(v) for v in data], shape, kind_of(x))


def _log_plain(x: Any) -> Any:
    """``x.log()`` (``-inf`` at 0, NaN below)."""
    if is_torch_tensor(x):
        return x.log()
    if _is_native(x):
        return sys.modules["numpy"].log(x)
    data, shape = to_flat(x, "x")

    def lg(v: float) -> float:
        return math.log(v) if v > 0 else (-math.inf if v == 0 else math.nan)

    return from_flat([lg(float(v)) for v in data], shape, kind_of(x))


def apply_constraints(
    obj: Any,
    constraints: list[Callable[[Any], Any]],
    samples: Any,
    infeasible_cost: float,
    eta: Any = 1e-3,
) -> Any:
    r"""Apply constraints using an infeasible_cost ``M`` for negative objectives.

    This allows feasibility-weighting an objective for the case where the
    objective can be negative by using the following strategy:
    (1) Add ``M`` to make obj non-negative;
    (2) Apply constraints using the sigmoid approximation;
    (3) Shift by ``-M``.

    Args:
        obj: A ``n_samples x b x q (x m')``-dim Tensor of objective values.
        constraints: A list of callables, each mapping a Tensor of size ``b x q x m``
            to a Tensor of size ``b x q``, where negative values imply feasibility.
            This callable must support broadcasting. Only relevant for multi-
            output models (``m`` > 1).
        samples: A ``n_samples x b x q x m`` Tensor of samples drawn from the posterior.
        infeasible_cost: The infeasible value.
        eta: The temperature parameter of the sigmoid function. Can be either a float
            or a 1-dim tensor. In case of a float the same eta is used for every
            constraint in constraints. In case of a tensor the length of the tensor
            must match the number of provided constraints. The i-th constraint is
            then estimated with the i-th eta value.

    Returns:
        A ``n_samples x b x q (x m')``-dim tensor of feasibility-weighted objectives.
    """
    # obj has dimensions n_samples x b x q (x m')
    obj = _ops.arith("add", obj, infeasible_cost)  # now it is nonnegative
    obj = apply_constraints_nonnegative_soft(
        obj=obj,
        constraints=constraints,
        samples=samples,
        eta=eta,
    )
    return _ops.arith("add", obj, -infeasible_cost)


__all__ = [
    "apply_constraints",
    "apply_constraints_nonnegative_soft",
    "compute_feasibility_indicator",
    "compute_smoothed_feasibility_indicator",
    "get_objective_weights_transform",
]

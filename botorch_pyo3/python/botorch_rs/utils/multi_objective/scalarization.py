"""Scalarizations (``botorch.utils.multi_objective.scalarization``).

``get_chebyshev_scalarization`` returns a :class:`ChebyshevScalarization`, a
callable object with the semantics of the function BoTorch returns
(``obj(Y, X=None)`` on ``... x m`` outcomes). It is evaluated in Rust;
for torch tensors that require gradients the result is connected to the
autograd graph (with the exact gradient computed in Rust), so it can be used
inside ``GenericMCObjective``. Acquisition functions can detect it and use
the underlying Rust object (``scalarization._rs``) directly.

Reference:
    J. Knowles, "ParEGO: a hybrid algorithm with on-line landscape
    approximation for expensive multiobjective optimization problems," IEEE
    Transactions on Evolutionary Computation, vol. 10, no. 1, pp. 50-66, 2006.
"""

from __future__ import annotations

from typing import Any

from botorch_rs import _botorch_rs as _rs
from botorch_rs._arrays import (
    ArrayKind,
    first_kind,
    from_flat,
    from_pair,
    is_torch_tensor,
    kind_of,
    to_flat,
)
from botorch_rs.exceptions.errors import BotorchTensorDimensionError, UnsupportedError
from botorch_rs.utils.multi_objective._common import shape_repr

_AUTOGRAD_FUNCTION: Any = None


def _autograd_function() -> Any:
    """A ``torch.autograd.Function`` evaluating the scalarization in Rust
    (defined on first use, so that torch is only imported when needed)."""
    global _AUTOGRAD_FUNCTION
    if _AUTOGRAD_FUNCTION is None:
        import torch

        class _ChebyshevValue(torch.autograd.Function):
            @staticmethod
            def forward(ctx: Any, Y: Any, scalarization: Any) -> Any:  # type: ignore[override]
                kind = ArrayKind("torch", Y.dtype, Y.device)
                (values, vshape), (grads, _) = scalarization.evaluate_with_grad(to_flat(Y, "Y"))
                ctx.save_for_backward(from_flat(grads, tuple(Y.shape), kind))
                return from_flat(values, tuple(vshape), kind)

            @staticmethod
            def backward(ctx: Any, grad_output: Any) -> tuple[Any, None]:  # type: ignore[override]
                (grad,) = ctx.saved_tensors
                return grad_output.unsqueeze(-1) * grad, None

        _AUTOGRAD_FUNCTION = _ChebyshevValue
    return _AUTOGRAD_FUNCTION


class ChebyshevScalarization:
    r"""The augmented Chebyshev scalarization returned by
    :func:`get_chebyshev_scalarization`:
    ``-(max_i(w_i * y_i) + alpha * sum_i(w_i * y_i))`` of the outcomes
    negated and normalized with the observed outcomes (larger is better).

    Call it as ``scalarization(Y, X=None)`` with ``Y`` of shape ``... x m``;
    the result has shape ``...`` and the array type of ``Y``.
    """

    def __init__(self, weights: Any, Y: Any, alpha: float = 0.05) -> None:
        w, w_shape = to_flat(weights, "weights")
        y, y_shape = to_flat(Y, "Y")
        if tuple(w_shape) != tuple(y_shape[-1:]):
            raise BotorchTensorDimensionError(
                "weights must be an `m`-dim tensor where Y is `... x m`."
                f"Got shapes {shape_repr(weights, w_shape)} and {shape_repr(Y, y_shape)}."
            )
        elif len(y_shape) > 2:
            raise NotImplementedError("Batched Y is not currently supported.")
        elif len(y_shape) < 2:
            raise IndexError("tuple index out of range")
        if y_shape[0] == 0 and any(v < 0 for v in w):
            raise UnsupportedError(
                "negative weights (for minimization) are only supported if Y is provided."
            )
        self._rs = _rs.ChebyshevScalarization(w, (y, y_shape), float(alpha))
        self._kind = first_kind(weights, Y)

    @property
    def weights(self) -> Any:
        """The weights (``m``)."""
        w = self._rs.weights
        return from_pair((w, (len(w),)), self._kind)

    @property
    def alpha(self) -> float:
        """The weight of the sum term."""
        return self._rs.alpha

    @property
    def Y_bounds(self) -> Any:
        """BoTorch's ``Y_bounds``: the bounds (``2 x m``) of the negated
        observed outcomes used for the normalization (``None`` without
        observations)."""
        bounds = self._rs.bounds
        return None if bounds is None else from_pair(bounds, self._kind)

    def __call__(self, Y: Any, X: Any = None) -> Any:
        r"""The scalarized outcomes.

        Args:
            Y: A ``... x m``-dim array of outcomes.
            X: Ignored (BoTorch's objective signature).

        Returns:
            A ``...``-dim array of scalarized values.
        """
        if is_torch_tensor(Y) and Y.requires_grad:
            import torch

            if torch.is_grad_enabled():
                return _autograd_function().apply(Y, self._rs)
        return from_pair(self._rs.evaluate(to_flat(Y, "Y")), kind_of(Y))

    def __repr__(self) -> str:
        return f"ChebyshevScalarization(weights={self._rs.weights}, alpha={self._rs.alpha})"


def get_chebyshev_scalarization(
    weights: Any, Y: Any, alpha: float = 0.05
) -> ChebyshevScalarization:
    r"""Construct an augmented Chebyshev scalarization.

    The augmented Chebyshev scalarization is given by
        g(y) = max_i(w_i * y_i) + alpha * sum_i(w_i * y_i)

    where the goal is to minimize g(y) in the setting where all objectives
    y_i are to be minimized. Since the default in BoTorch is to maximize all
    objectives, this method constructs a Chebyshev scalarization where the
    inputs are first multiplied by -1, so that all objectives are to be
    minimized. Then, it computes g(y) (which should be minimized), and
    returns -g(y), which should be maximized.

    Minimizing an objective is supported by passing a negative weight for
    that objective. To make all w * y's have the same sign such that they are
    comparable when computing max(w * y), outcomes of minimization objectives
    are shifted from [0,1] to [-1,0].

    Args:
        weights: A ``m``-dim array of weights. Positive for maximization and
            negative for minimization.
        Y: A ``n x m``-dim array of observed outcomes, which are used for
            scaling the outcomes to [0,1] or [-1,0]. If ``n=0``, then outcomes
            are left unnormalized.
        alpha: Parameter governing the influence of the weighted sum term.

    Returns:
        A :class:`ChebyshevScalarization` (callable as ``obj(Y, X=None)``).
    """
    return ChebyshevScalarization(weights, Y, alpha)


__all__ = ["ChebyshevScalarization", "get_chebyshev_scalarization"]

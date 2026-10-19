r"""
Discretization (rounding) functions for acquisition optimization
(``botorch.utils.rounding``).

``approximate_round`` is evaluated in Rust (with its exact derivative for
torch tensors). The straight-through estimators ``RoundSTE`` and
``OneHotArgmaxSTE`` are ``torch.autograd.Function``\ s for torch tensors
(``Cls.apply(X)``, as in BoTorch); for NumPy arrays and nested lists
``apply`` computes the forward values.

.. [Daulton2022bopr]
    S. Daulton, X. Wan, D. Eriksson, M. Balandat, M. A. Osborne, E. Bakshy.
    Bayesian Optimization over Discrete and Mixed Spaces via Probabilistic
    Reparameterization. Advances in Neural Information Processing Systems
    35, 2022.
"""

from __future__ import annotations

import importlib
from typing import Any

from botorch_rs import _ops
from botorch_rs._arrays import from_flat, is_torch_tensor, kind_of, to_flat


def approximate_round(X: Any, tau: float = 1e-3) -> Any:
    r"""Differentiable approximate rounding function.

    This method is a piecewise approximation of a rounding function where
    each piece is a hyperbolic tangent function.

    Args:
        X: The tensor to round to the nearest integer (element-wise).
        tau: A temperature hyperparameter.

    Returns:
        The approximately rounded input tensor.
    """
    return _ops.apply("approximate_round", X, tau, kind=kind_of(X))


_TORCH_FUNCTIONS: dict[str, Any] = {}


def _torch_function(name: str) -> Any:
    fn = _TORCH_FUNCTIONS.get(name)
    if fn is not None:
        return fn
    torch = importlib.import_module("torch")

    class _Identity(torch.autograd.Function):
        @staticmethod
        def backward(ctx: Any, grad_output: Any) -> Any:  # type: ignore[override]
            return grad_output

    if name == "round":

        class _Round(_Identity):
            @staticmethod
            def forward(ctx: Any, X: Any) -> Any:  # type: ignore[override]
                return X.round()

        fn = _Round
    else:

        class _OneHot(_Identity):
            @staticmethod
            def forward(ctx: Any, X: Any) -> Any:  # type: ignore[override]
                return torch.nn.functional.one_hot(X.argmax(dim=-1), num_classes=X.shape[-1]).to(X)

        fn = _OneHot
    _TORCH_FUNCTIONS[name] = fn
    return fn


def _round_half_even(v: float) -> float:
    return float(round(v))


class IdentitySTEFunction:
    """Base class for functions using straight through gradient estimators.

    This class approximates the gradient with the identity function.
    """

    _name = ""

    @staticmethod
    def backward(ctx: Any, grad_output: Any) -> Any:
        r"""Use a straight-through estimator the gradient.

        This uses the identity function.

        Args:
            grad_output: A tensor of gradients.

        Returns:
            The provided tensor.
        """
        return grad_output

    @classmethod
    def apply(cls, X: Any) -> Any:
        """The forward values; for torch tensors through a
        ``torch.autograd.Function`` with the identity as its backward."""
        if is_torch_tensor(X):
            return _torch_function(cls._name).apply(X)
        return cls.forward(None, X)

    @staticmethod
    def forward(ctx: Any, X: Any) -> Any:  # pragma: no cover - abstract
        raise NotImplementedError


class RoundSTE(IdentitySTEFunction):
    r"""Round the input tensor and use a straight-through gradient estimator.

    [Daulton2022bopr]_ proposes using this in acquisition optimization.
    """

    _name = "round"

    @staticmethod
    def forward(ctx: Any, X: Any) -> Any:
        r"""Round the input tensor element-wise.

        Args:
            X: The tensor to be rounded.

        Returns:
            A tensor where each element is rounded to the nearest integer.
        """
        if is_torch_tensor(X):
            return X.round()
        data, shape = to_flat(X, "X")
        out = from_flat([_round_half_even(float(v)) for v in data], shape, kind_of(X))
        return out


class OneHotArgmaxSTE(IdentitySTEFunction):
    r"""Discretize a continuous relaxation of a one-hot encoded categorical.

    This returns a one-hot encoded categorical and use a straight-through
    gradient estimator via an identity function.

    [Daulton2022bopr]_ proposes using this in acquisition optimization.
    """

    _name = "one_hot"

    @staticmethod
    def forward(ctx: Any, X: Any) -> Any:
        r"""Discretize the input tensor.

        This applies a argmax along the last dimensions of the input tensor
        and one-hot encodes the result.

        Args:
            X: The tensor to be rounded.

        Returns:
            A tensor where each element is rounded to the nearest integer.
        """
        if is_torch_tensor(X):
            torch = importlib.import_module("torch")
            return torch.nn.functional.one_hot(X.argmax(dim=-1), num_classes=X.shape[-1]).to(X)
        data, shape = to_flat(X, "X")
        k = shape[-1]
        out: list[float] = []
        for i in range(0, len(data), k):
            row = [float(v) for v in data[i : i + k]]
            best = max(range(k), key=lambda j: (row[j], -j))
            out.extend(1.0 if j == best else 0.0 for j in range(k))
        return from_flat(out, shape, kind_of(X))


__all__ = ["IdentitySTEFunction", "OneHotArgmaxSTE", "RoundSTE", "approximate_round"]

"""Model-based closures (``botorch.optim.closures.model_closures``).

The loss closure of a botorch_rs marginal log likelihood evaluates the
negative MLL (and its gradient) in Rust; the closures with gradients return
the gradients of the selected parameters, computed analytically.
"""

from __future__ import annotations

from typing import Any

from botorch_rs._arrays import from_flat, kind_of, to_flat
from botorch_rs.exceptions.errors import UnsupportedError
from botorch_rs.optim.closures.core import ForwardBackwardClosure


class MllLossClosure:
    r"""The loss closure of a botorch_rs marginal log likelihood: calling it
    returns the negative MLL (one value per output and batch element, of the
    model's array type), like BoTorch's closures built by
    ``get_loss_closure``."""

    def __init__(self, mll: Any) -> None:
        self.mll = mll

    def __call__(self, **kwargs: Any) -> Any:
        if kwargs:
            raise UnsupportedError(
                "botorch_rs evaluates the MLL on the model's training data; closure keyword "
                f"arguments are not supported (got {sorted(kwargs)})."
            )
        values = self.mll()
        data, shape = to_flat(values, "mll")
        return from_flat([-v for v in data], shape, kind_of(values))

    def value_and_grad(self, names: list[str]) -> tuple[float, dict[str, list[float]]]:
        """The summed loss and its gradient with respect to the parameters
        ``names`` (flat values per parameter)."""
        from botorch_rs.optim.utils.model_utils import _entries

        entries = {e.name: e for e in _entries(self.mll)}
        total = 0.0
        grads: dict[str, list[float]] = {}
        cache: dict[int, list[float]] = {}
        for model in {id(e.model): e.model for e in entries.values()}.values():
            from botorch_rs.mlls import _forward_pass

            _forward_pass(model)
            (values, shape), grad = model._rs.mll_value_and_grad(None)
            total += -sum(values)
            cache[id(model)] = [-g for g in grad]
        k = len(getattr(self.mll, "mlls", [None]))
        for name in names:
            e = entries[name]
            grads[name] = cache[id(e.model)][e.offset : e.offset + e.size]
        if k > 1:
            # SumMarginalLogLikelihood divides the sum by the number of models.
            total /= k
            grads = {n: [g / k for g in v] for n, v in grads.items()}
        return total, grads


def get_loss_closure(mll: Any, data_loader: Any = None) -> Any:
    r"""Factory function for creating loss closures from marginal log
    likelihoods (BoTorch's ``get_loss_closure``).

    Args:
        mll: A botorch_rs ``MarginalLogLikelihood`` whose negative defines
            the loss (an MLL or model with ``compute_custom_loss`` uses that).
        data_loader: Not supported (botorch_rs evaluates the MLL on the
            model's training data).

    Returns:
        A closure that takes zero positional arguments and returns the
        negated value of ``mll``.
    """
    if hasattr(mll, "compute_custom_loss"):
        return mll.compute_custom_loss
    model = getattr(mll, "model", None)
    if model is not None and hasattr(model, "compute_custom_loss"):

        def custom(**kwargs: Any) -> Any:
            return model.compute_custom_loss(mll=mll, **kwargs)

        return custom
    if data_loader is not None:
        raise UnsupportedError("`data_loader` is not supported by botorch_rs.")
    return MllLossClosure(mll)


def get_loss_closure_with_grads(
    mll: Any,
    parameters: dict[str, Any],
    data_loader: Any = None,
) -> ForwardBackwardClosure:
    r"""A loss closure with gradients of ``parameters`` (BoTorch's
    ``get_loss_closure_with_grads``): a :class:`ForwardBackwardClosure`
    around :func:`get_loss_closure`.

    Returns:
        A closure that takes zero positional arguments and returns the reduced
        and negated value of ``mll`` along with the gradients of
        ``parameters``.
    """
    loss_closure = get_loss_closure(mll=mll, data_loader=data_loader)
    return ForwardBackwardClosure(forward=loss_closure, parameters=parameters)


def _grads_out(parameters: dict[str, Any], grads: dict[str, list[float]]) -> tuple[Any, ...]:
    """The gradients shaped like the parameter values."""
    out = []
    for name, value in parameters.items():
        shape = to_flat(value, name)[1]
        out.append(from_flat(grads[name], shape, kind_of(value)))
    return tuple(out)


__all__ = ["get_loss_closure", "get_loss_closure_with_grads"]

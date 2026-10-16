"""Core methods for building closures (``botorch.optim.closures.core``).

For the loss closures of botorch_rs marginal log likelihoods (see
:func:`~botorch_rs.optim.closures.get_loss_closure`), the gradients are
computed in Rust and the parameter state is read from and written to the
model; other closures follow BoTorch's semantics on torch tensors.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from botorch_rs._arrays import to_flat


def _native(forward: Any) -> Any:
    """The botorch_rs MLL loss closure behind ``forward``, if any."""
    from botorch_rs.optim.closures.model_closures import MllLossClosure

    return forward if isinstance(forward, MllLossClosure) else None


class ForwardBackwardClosure:
    r"""Wrapper for fused forward and backward closures (BoTorch's
    ``ForwardBackwardClosure``)."""

    def __init__(self, forward: Callable[..., Any], parameters: dict[str, Any]) -> None:
        r"""Initializes a ForwardBackwardClosure instance.

        Args:
            forward: Callable that returns an array (a botorch_rs loss
                closure, or a callable on torch tensors).
            parameters: A dictionary of parameters whose gradients are to be
                returned.
        """
        self.forward = forward
        self.parameters = parameters

    def __call__(self, **kwargs: Any) -> tuple[Any, tuple[Any, ...]]:
        native = _native(self.forward)
        if native is not None:
            from botorch_rs.optim.closures.model_closures import _grads_out

            value, grads = native.value_and_grad(list(self.parameters))
            return value, _grads_out(self.parameters, grads)
        import torch

        for p in self.parameters.values():
            p.grad = None
        value = self.forward(**kwargs).sum()
        value.backward()
        grads = tuple(param.grad for param in self.parameters.values())
        for p in self.parameters.values():
            p.grad = None
        return value, tuple(g if g is None else torch.as_tensor(g) for g in grads)


class NdarrayOptimizationClosure:
    r"""Adds stateful behavior and a ``numpy.ndarray``-typed API to a closure
    returning ``(value, gradients)`` (BoTorch's
    ``NdarrayOptimizationClosure``). NaN values are replaced with 0.0 in the
    returned gradient array.

    For a :class:`ForwardBackwardClosure` of a botorch_rs MLL, ``state`` is
    the flat vector of the selected raw hyperparameters of the model.
    """

    def __init__(
        self,
        closure: Callable[[], tuple[Any, Sequence[Any]]],
        parameters: dict[str, Any],
    ) -> None:
        self.closure = closure
        self.parameters = parameters
        self._native = (
            _native(closure.forward) if isinstance(closure, ForwardBackwardClosure) else None
        )

    def _sizes(self) -> list[int]:
        return [len(to_flat(p, n)[0]) for n, p in self.parameters.items()]

    def __call__(self, state: Any = None, **kwargs: Any) -> tuple[Any, Any]:
        import numpy as np

        if state is not None:
            self.state = state
        try:
            value, grads = self.closure(**kwargs)
            value = np.asarray(float(to_flat(value, "value")[0][0]), dtype=np.float64)
            out = np.zeros(sum(self._sizes()), dtype=np.float64)
            index = 0
            for (name, p), g in zip(self.parameters.items(), grads, strict=True):
                size = len(to_flat(p, name)[0])
                if g is not None:
                    out[index : index + size] = np.asarray(to_flat(g, name)[0], dtype=np.float64)
                index += size
            out[np.isnan(out)] = 0.0
            return value, out
        except RuntimeError as e:
            from botorch_rs.exceptions.errors import NanError, NotPSDError

            if isinstance(e, NotPSDError):
                raise
            if isinstance(e, NanError) or "singular" in str(e) or "positive-definite" in str(e):
                x = self.state
                return np.full((), np.nan), np.full_like(x, np.nan)
            raise

    @property
    def state(self) -> Any:
        import numpy as np

        if len(self.parameters) == 0:
            raise RuntimeError("No parameters to get state from.")
        if self._native is not None:
            from botorch_rs.optim.utils.model_utils import _entries

            entries = {e.name: e for e in _entries(self._native.mll)}
            values: list[float] = []
            for name in self.parameters:
                e = entries[name]
                raw = e.model._rs.raw_parameters()
                values.extend(raw[e.offset : e.offset + e.size])
            return np.asarray(values, dtype=np.float64)
        return np.concatenate(
            [np.asarray(to_flat(p, n)[0], dtype=np.float64) for n, p in self.parameters.items()]
        )

    @state.setter
    def state(self, state: Any) -> None:
        values = [float(v) for v in state]
        if self._native is not None:
            from botorch_rs.optim.utils.model_utils import _entries

            entries = {e.name: e for e in _entries(self._native.mll)}
            raws: dict[int, list[float]] = {}
            models: dict[int, Any] = {}
            index = 0
            for name in self.parameters:
                e = entries[name]
                raw = raws.setdefault(id(e.model), list(e.model._rs.raw_parameters()))
                models[id(e.model)] = e.model
                raw[e.offset : e.offset + e.size] = values[index : index + e.size]
                index += e.size
            for key, raw in raws.items():
                models[key]._rs.set_raw_parameters(raw)
            return
        import torch

        with torch.no_grad():
            index = 0
            for p in self.parameters.values():
                size = p.numel()
                vals = values[index : index + size] if p.ndim else values[index]
                p.copy_(torch.as_tensor(vals, device=p.device, dtype=p.dtype).view(p.shape))
                index += size


__all__ = ["ForwardBackwardClosure", "NdarrayOptimizationClosure"]

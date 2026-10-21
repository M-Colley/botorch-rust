"""Core methods for building closures (``botorch.optim.closures.core``).

For the loss closures of botorch_rs marginal log likelihoods (see
:func:`~botorch_rs.optim.closures.get_loss_closure`), the gradients are
computed in Rust and the parameter state is read from and written to the
model; other closures follow BoTorch's semantics on torch tensors.
"""

from __future__ import annotations

import math
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
            from botorch_rs.mlls import _persistent_parameter
            from botorch_rs.optim.closures.model_closures import _grads_out

            value, grads = native.value_and_grad(list(self.parameters), **kwargs)
            gradients = _grads_out(self.parameters, grads)
            out = []
            for parameter, gradient in zip(self.parameters.values(), gradients, strict=True):
                if _persistent_parameter(parameter):
                    if not hasattr(value, "detach"):
                        value = parameter.new_tensor(value)
                    if parameter.requires_grad:
                        parameter.grad = gradient
                    elif parameter.grad is not None:
                        parameter.grad.zero_()
                    gradient = parameter.grad
                out.append(gradient)
            return value, tuple(out)
        from botorch_rs.utils.context_managers import zero_grad_ctx

        with zero_grad_ctx(self.parameters):
            value = self.forward(**kwargs).sum()
            value.backward()
            return value, tuple(param.grad for param in self.parameters.values())


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
            out: Any = np.zeros(sum(self._sizes()), dtype=np.float64)
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


class BatchedNDarrayOptimizationClosure:
    """Expose independent batch losses and parameter gradients as NumPy rows.

    Native MLL closures retain their Rust parameter state and derivatives.
    User-provided torch closures use torch autograd, as in BoTorch.
    """

    def __init__(
        self,
        forward: Callable[..., Any],
        parameters: dict[str, Any],
        batch_shape: Any,
    ) -> None:
        self.forward = forward
        self.parameters = parameters
        self.batch_shape = batch_shape
        batch = tuple(int(value) for value in batch_shape)
        if any(value <= 0 for value in batch):
            raise ValueError("Batch dimensions must be positive.")
        self.batch_size = max(math.prod(batch), 1)
        self._trailing_sizes = {}
        self._per_element_size = 0
        for name, parameter in parameters.items():
            shape = to_flat(parameter, name)[1]
            if tuple(shape[: len(batch)]) != batch:
                raise ValueError(f"Parameter {name!r} must begin with batch_shape {batch}.")
            size = math.prod(shape[len(batch) :])
            self._trailing_sizes[name] = size
            self._per_element_size += size
        self._native = _native(forward)
        self._state_wrapper = (
            NdarrayOptimizationClosure(ForwardBackwardClosure(forward, parameters), parameters)
            if self._native is not None
            else None
        )

    @property
    def state(self) -> Any:
        import numpy as np

        output: Any = np.empty((self.batch_size, self._per_element_size), dtype=np.float64)
        flat = None if self._state_wrapper is None else self._state_wrapper.state
        column, offset = 0, 0
        for name, parameter in self.parameters.items():
            size = self._trailing_sizes[name]
            count = self.batch_size * size
            values = to_flat(parameter, name)[0] if flat is None else flat[offset : offset + count]
            output[:, column : column + size] = np.asarray(values).reshape(self.batch_size, size)
            column += size
            offset += count
        return output

    @state.setter
    def state(self, state: Any) -> None:
        import numpy as np

        state = np.asarray(state, dtype=np.float64)
        expected = (self.batch_size, self._per_element_size)
        if state.shape != expected:
            raise ValueError(f"Expected state shape {expected}, got {state.shape}.")
        column = 0
        if self._state_wrapper is not None:
            flattened = []
            for name in self.parameters:
                size = self._trailing_sizes[name]
                flattened.extend(state[:, column : column + size].reshape(-1))
                column += size
            self._state_wrapper.state = flattened
            return
        import torch

        with torch.no_grad():
            for name, parameter in self.parameters.items():
                size = self._trailing_sizes[name]
                values = state[:, column : column + size]
                parameter.copy_(
                    torch.as_tensor(values, dtype=parameter.dtype, device=parameter.device).reshape(
                        parameter.shape
                    )
                )
                column += size

    def __call__(
        self, state: Any = None, batch_indices: Any = None, **kwargs: Any
    ) -> tuple[Any, Any]:
        import numpy as np

        if state is not None:
            if batch_indices is None:
                self.state = state
            else:
                full = self.state
                full[batch_indices] = state
                self.state = full
        try:
            gradients: Any = np.zeros((self.batch_size, self._per_element_size), dtype=np.float64)
            if self._native is not None:
                per_batch = self.forward(**kwargs)
                _, native_gradients = self._native.value_and_grad(list(self.parameters))
                grad_values = [native_gradients[name] for name in self.parameters]
            else:
                from botorch_rs.utils.context_managers import zero_grad_ctx

                with zero_grad_ctx(self.parameters):
                    per_batch = self.forward(**kwargs)
                    per_batch.sum().backward()
                    grad_values = [parameter.grad for parameter in self.parameters.values()]
            values = np.asarray(to_flat(per_batch, "values")[0], dtype=np.float64).reshape(
                self.batch_size
            )
            column = 0
            for (name, _), gradient in zip(self.parameters.items(), grad_values, strict=True):
                size = self._trailing_sizes[name]
                if gradient is not None:
                    gradients[:, column : column + size] = np.asarray(
                        to_flat(gradient, name)[0], dtype=np.float64
                    ).reshape(self.batch_size, size)
                column += size
        except RuntimeError as error:
            from botorch_rs.exceptions.errors import NanError, NotPSDError

            if isinstance(error, NotPSDError) or type(error).__name__ == "NotPSDError":
                raise
            if not (
                isinstance(error, NanError)
                or type(error).__name__ == "NanError"
                or "singular" in str(error)
                or "input is not positive-definite" in str(error)
            ):
                raise
            values = np.full(self.batch_size, np.nan, dtype=np.float64)
            gradients = np.zeros((self.batch_size, self._per_element_size), dtype=np.float64)
        if batch_indices is not None:
            return values[batch_indices], gradients[batch_indices]
        return values, gradients


__all__ = [
    "BatchedNDarrayOptimizationClosure",
    "ForwardBackwardClosure",
    "NdarrayOptimizationClosure",
]

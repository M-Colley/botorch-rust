r"""
Utilities for optimization (``botorch.utils.context_managers``).

The context managers work on torch modules (state dicts, parameters and
their gradients, as in BoTorch) and on botorch_rs models, whose
hyperparameters live in Rust: ``module_rollback_ctx`` checkpoints their raw
hyperparameters (by BoTorch's names, ``model.named_parameters()``) and
restores them on exit.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Generator, Iterable
from contextlib import contextmanager
from typing import Any, NamedTuple

from botorch_rs._arrays import from_flat, is_torch_tensor, kind_of, to_flat


class TensorCheckpoint(NamedTuple):
    values: Any
    device: Any = None
    dtype: Any = None


@contextmanager
def delattr_ctx(
    instance: object, *attrs: str, enforce_hasattr: bool = False
) -> Generator[None, None, None]:
    r"""Contextmanager for temporarily deleting attributes."""
    cache = {}
    try:
        for key in attrs:
            if hasattr(instance, key):
                cache[key] = getattr(instance, key)
                delattr(instance, key)
            elif enforce_hasattr:
                raise ValueError(f"Attribute {key} missing from {type(instance)} instance.")
        yield
    finally:
        for key, cached_val in cache.items():
            setattr(instance, key, cached_val)


def _copy(values: Any, **tkwargs: Any) -> Any:
    if is_torch_tensor(values):
        return values.detach().to(**tkwargs).clone()
    data, shape = to_flat(values, "values")
    return from_flat(list(data), shape, kind_of(values))


def _assign(param: Any, values: Any, device: Any, dtype: Any) -> None:
    if is_torch_tensor(param):
        param.data.copy_(values.to(device=device, dtype=dtype))
    else:
        param[...] = values


@contextmanager
def parameter_rollback_ctx(
    parameters: dict[str, Any],
    checkpoint: dict[str, TensorCheckpoint] | None = None,
    **tkwargs: Any,
) -> Generator[dict[str, TensorCheckpoint], None, None]:
    r"""Contextmanager that exits by rolling back a module's state_dict.

    Args:
        module: Module instance.
        name_filter: Optional Boolean function used to filter items by name.
        checkpoint: Optional cache of values and tensor metadata specifying the rollback
            state for the module (or some subset thereof).
        **tkwargs: Keyword arguments passed to ``torch.Tensor.to`` when copying data
            from each tensor in ``module.state_dict()`` to the internally created
            checkpoint. Only adhered to when the ``checkpoint`` argument is None.

    Yields:
        A dictionary of TensorCheckpoints for the module's state_dict. Any in-places
        changes to the checkpoint will be observed at rollback time. If the checkpoint
        is cleared, no rollback will occur.

    The parameters are torch tensors (restored with ``param.data.copy_``) or
    writable NumPy arrays.
    """
    # Create copies of the original values
    if checkpoint is None:
        checkpoint = {
            name: TensorCheckpoint(
                values=_copy(param, **tkwargs),
                device=getattr(param, "device", None),
                dtype=getattr(param, "dtype", None),
            )
            for name, param in parameters.items()
        }

    try:  # yield the checkpoint dictionary to the user
        yield checkpoint
    finally:  # restore original values of tracked parameters
        if checkpoint:
            for name, param in parameters.items():
                if name in checkpoint:
                    values, device, dtype = checkpoint[name]
                    _assign(param, values, device, dtype)


def _is_rs_model(module: Any) -> bool:
    rs = getattr(module, "_rs", None)
    return rs is not None and hasattr(rs, "raw_parameters") and hasattr(rs, "parameter_shapes")


def _rs_layout(module: Any) -> list[tuple[str, tuple[int, ...]]]:
    return [(name, tuple(shape)) for name, shape in module._rs.parameter_shapes()]


@contextmanager
def module_rollback_ctx(
    module: Any,
    name_filter: Callable[[str], bool] | None = None,
    checkpoint: dict[str, TensorCheckpoint] | None = None,
    **tkwargs: Any,
) -> Generator[dict[str, TensorCheckpoint], None, None]:
    r"""Contextmanager that exits by rolling back a module's state_dict.

    Args:
        module: Module instance.
        name_filter: Optional Boolean function used to filter items by name.
        checkpoint: Optional cache of values and tensor metadata specifying the rollback
            state for the module (or some subset thereof).
        **tkwargs: Keyword arguments passed to ``torch.Tensor.to`` when copying data
            from each tensor in ``module.state_dict()`` to the internally created
            checkpoint. Only adhered to when the ``checkpoint`` argument is None.

    Yields:
        A dictionary of TensorCheckpoints for the module's state_dict. Any in-places
        changes to the checkpoint will be observed at rollback time. If the checkpoint
        is cleared, no rollback will occur.

    For a botorch_rs model the checkpoint holds its raw hyperparameters
    (``model.named_parameters()``: arrays of the model's array type), which
    are restored on exit.
    """
    if _is_rs_model(module):
        yield from _rs_rollback(module, name_filter, checkpoint)
        return
    # Create copies of the original values
    if checkpoint is None:
        checkpoint = {
            name: TensorCheckpoint(
                values=values.detach().to(**tkwargs).clone(),
                device=values.device,
                dtype=values.dtype,
            )
            for name, values in module.state_dict().items()
            if name_filter is None or name_filter(name)
        }

    try:  # yield the checkpoint dictionary to the user
        yield checkpoint
    finally:  # restore original values of tracked parameters
        if checkpoint:
            state_dict = module.state_dict()
            for key, (values, device, dtype) in checkpoint.items():
                tnsr = state_dict.get(key)
                if tnsr is None:
                    state_dict[key] = values.to(device=device, dtype=dtype)
                else:
                    tnsr[...] = values.to(device=device, dtype=dtype)

            module.load_state_dict(state_dict)


def _rs_rollback(
    module: Any,
    name_filter: Callable[[str], bool] | None,
    checkpoint: dict[str, TensorCheckpoint] | None,
) -> Generator[dict[str, TensorCheckpoint], None, None]:
    layout = _rs_layout(module)
    kind = getattr(module, "_kind", None) or kind_of([])
    if checkpoint is None:
        values = list(module._rs.raw_parameters())
        checkpoint = {}
        offset = 0
        for name, shape in layout:
            size = math.prod(shape)
            if name_filter is None or name_filter(name):
                arr = from_flat(values[offset : offset + size], shape, kind)
                checkpoint[name] = TensorCheckpoint(
                    values=arr,
                    device=getattr(arr, "device", None),
                    dtype=getattr(arr, "dtype", None),
                )
            offset += size
    try:
        yield checkpoint
    finally:
        if checkpoint:
            values = list(module._rs.raw_parameters())
            offset = 0
            for name, shape in layout:
                size = math.prod(shape)
                if name in checkpoint:
                    data, _ = to_flat(checkpoint[name].values, name)
                    values[offset : offset + size] = [float(v) for v in data]
                offset += size
            module._rs.set_raw_parameters(values)


@contextmanager
def zero_grad_ctx(
    parameters: dict[str, Any] | Iterable[Any],
    zero_on_enter: bool = True,
    zero_on_exit: bool = False,
) -> Generator[None, None, None]:
    """Zeroes the gradients of the (torch) parameters on entering and / or
    exiting the context."""

    def zero_() -> None:
        for param in parameters.values() if isinstance(parameters, dict) else parameters:
            if getattr(param, "grad", None) is not None:
                param.grad.zero_()

    if zero_on_enter:
        zero_()

    try:
        yield
    finally:
        if zero_on_exit:
            zero_()


__all__ = [
    "TensorCheckpoint",
    "delattr_ctx",
    "module_rollback_ctx",
    "parameter_rollback_ctx",
    "zero_grad_ctx",
]

"""Helpers of the pathwise sampling package (``botorch.sampling.pathwise.utils``).

BoTorch's ``utils`` module holds the transforms that its paths compose
(input transforms, ``OutcomeUntransformer``, the lengthscale, sine-cosine and
outputscale transforms of the Fourier features). In botorch_rs these are part
of the paths computed in Rust; this module arranges the values computed in
Rust into BoTorch's shapes.

A path evaluated at ``X`` (``batch_shape x n x d``) returns BoTorch's
broadcast of ``batch_shape`` with the batch shape of the path's weights
(``sample_shape`` for a single-output model, ``sample_shape x m`` for a
multi-output ``SingleTaskGP``), followed by ``n``: for every broadcast index,
the value of the indexed sample (and output) at the indexed point of ``X``.
"""

from __future__ import annotations

import itertools
from collections.abc import Sequence
from typing import Any, NamedTuple

from botorch_rs import _botorch_rs as _rs
from botorch_rs._arrays import (
    ArrayKind,
    broadcast_shapes,
    from_flat,
    is_torch_tensor,
    kind_of,
    to_flat,
)
from botorch_rs.exceptions.errors import UnsupportedError


def sample_shape_arg(sample_shape: Any) -> tuple[int, ...]:
    """``sample_shape`` (a ``torch.Size``, a sequence or an int) as a tuple."""
    if sample_shape is None:
        return ()
    if isinstance(sample_shape, int):
        return (int(sample_shape),)
    return tuple(int(s) for s in sample_shape)


def model_outputs(model: Any) -> tuple[_rs.Model, list[int]]:
    """The extension handle of a botorch_rs ``SingleTaskGP`` or ``ModelListGP``
    and the number of outputs of each of its models (one entry for a
    ``SingleTaskGP``)."""
    from botorch_rs.models.gp_regression import SingleTaskGP
    from botorch_rs.models.model_list_gp_regression import ModelListGP

    if isinstance(model, SingleTaskGP):
        sizes = [model.num_outputs]
    elif isinstance(model, ModelListGP):
        sizes = [m.num_outputs for m in model.models]
    else:
        raise UnsupportedError(
            "botorch_rs pathwise sampling requires a botorch_rs SingleTaskGP or ModelListGP, "
            f"got {type(model).__name__}."
        )
    if model.batch_shape:
        raise UnsupportedError(
            f"Pathwise sampling of batched models (batch shape {model.batch_shape}) is not "
            "supported by botorch_rs."
        )
    return model._acquisition_model(None), sizes


class PathValues(NamedTuple):
    """The ``m x S x N`` values of every output of a handle: a tensor (that
    may require gradients), a NumPy array, or a flat list (nested-list
    inputs)."""

    values: Any
    kind: ArrayKind


def input_shape(X: Any, d: int) -> tuple[int, ...]:
    """The shape of the inputs ``X`` (``batch_shape x n x d``)."""
    shape = tuple(X.shape) if hasattr(X, "shape") else to_flat(X, "X")[1]
    if len(shape) < 2 or shape[-1] != d:
        raise ValueError(
            f"Expected the inputs to have shape `batch_shape x n x {d}`, got {tuple(shape)}."
        )
    return shape


def path_values(handle: _rs.SamplePaths, X: Any, component: str | None = None) -> PathValues:
    """The values of every output of ``handle`` at the points of ``X``
    (connected to the autograd graph of ``X`` if it is a tensor that requires
    gradients)."""
    kind = kind_of(X)
    if is_torch_tensor(X):
        import torch

        if X.requires_grad and torch.is_grad_enabled():
            from botorch_rs.sampling.pathwise import _autograd

            return PathValues(_autograd.path_values(handle, X, component), kind)
    data, shape = handle.forward(to_flat(X, "X"), component)
    if kind.lib == "list":
        return PathValues(list(data), kind)
    return PathValues(from_flat(data, shape, kind), kind)


def _gather(pv: PathValues, index: list[int], shape: tuple[int, ...]) -> Any:
    """``values.reshape(-1)[index].reshape(shape)``."""
    values, kind = pv
    if kind.lib == "torch":
        import torch

        idx = torch.tensor(index, dtype=torch.long, device=values.device)
        return values.reshape(-1)[idx].reshape(shape)
    if kind.lib == "numpy":
        import numpy as np

        return values.reshape(-1)[np.asarray(index, dtype=np.int64)].reshape(shape)
    return from_flat([values[i] for i in index], shape, kind)


def _flat_index(idx: Sequence[int], dims: tuple[int, ...]) -> int:
    """The row-major index in an array of shape ``dims`` of the trailing
    entries of the broadcast index ``idx`` (size-1 dimensions broadcast)."""
    flat = 0
    offset = len(idx) - len(dims)
    for k, size in enumerate(dims):
        flat = flat * size + (0 if size == 1 else idx[offset + k])
    return flat


def arrange(
    pv: PathValues,
    num_samples: int,
    num_points: int,
    x_shape: tuple[int, ...],
    sample_shape: tuple[int, ...],
    outputs: Sequence[int],
    multi_output: bool,
    outputs_last: bool = False,
) -> Any:
    """BoTorch's values of the paths of ``outputs`` from the ``m_all x S x N``
    values of a handle (``S = num_samples``, ``N = num_points``) at inputs of
    shape ``x_shape``.

    ``multi_output``: the outputs form the last batch dimension of the
    weights (a multi-output ``SingleTaskGP``); ``outputs_last``: they form a
    trailing dimension instead (``MatheronPathModel``'s ``... x n x m``).
    """
    x_batch, n = x_shape[:-2], x_shape[-2]
    m = len(outputs)
    w_batch = sample_shape + ((m,) if multi_output and not outputs_last else ())
    out_batch = broadcast_shapes(x_batch, w_batch)
    if out_batch is None:
        raise RuntimeError(
            f"The inputs of batch shape {x_batch} cannot be broadcast with the batch shape "
            f"{w_batch} of the sample paths."
        )
    index: list[int] = []
    for idx in itertools.product(*(range(b) for b in out_batch)):
        xi = _flat_index(idx, x_batch) * n
        s = _flat_index(idx, sample_shape + ((1,) if w_batch != sample_shape else ()))
        if outputs_last:
            for i in range(n):
                index.extend((t * num_samples + s) * num_points + xi + i for t in outputs)
        else:
            t = outputs[idx[-1] if multi_output and m > 1 else 0]
            start = (t * num_samples + s) * num_points + xi
            index.extend(range(start, start + n))
    out_shape = out_batch + (n,) + ((m,) if outputs_last else ())
    return _gather(pv, index, out_shape)

"""Helpers of the pathwise sampling package (``botorch.sampling.pathwise.utils``).

Transforms preserve lists, NumPy arrays and torch tensors; elementary
functions and their first derivatives are evaluated in Rust. The module
also arranges native path values into BoTorch's shapes.

A path evaluated at ``X`` (``batch_shape x n x d``) returns BoTorch's
broadcast of ``batch_shape`` with the batch shape of the path's weights
(``sample_shape`` for a single-output model, ``sample_shape x m`` for a
multi-output ``SingleTaskGP``), followed by ``n``: for every broadcast index,
the value of the indexed sample (and output) at the indexed point of ``X``.
"""

from __future__ import annotations

import itertools
import math
import operator
from abc import ABC, abstractmethod
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
from botorch_rs._module import GPModule
from botorch_rs._ops import apply, concat, getitem, int_array, shape_of, transpose
from botorch_rs.exceptions.errors import UnsupportedError
from botorch_rs.models.approximate_gp import SingleTaskVariationalGP
from botorch_rs.models.model import Model, ModelList
from botorch_rs.models.model_list_gp_regression import ModelListGP
from botorch_rs.models.transforms.input import InputTransform
from botorch_rs.models.transforms.outcome import OutcomeTransform
from botorch_rs.utils.dispatcher import Dispatcher

GetTrainInputs = Dispatcher("get_train_inputs")
GetTrainTargets = Dispatcher("get_train_targets")


class TransformedModuleMixin:
    """Apply optional input and output transforms around a callable module."""

    def __call__(self, values: Any, *args: Any, **kwargs: Any) -> Any:
        transform = getattr(self, "input_transform", None)
        if transform is not None:
            values = (
                transform.forward(values)
                if isinstance(transform, InputTransform)
                else transform(values)
            )
        output = super().__call__(values, *args, **kwargs)  # type: ignore[misc]
        transform = getattr(self, "output_transform", None)
        if transform is None:
            return output
        return (
            transform.untransform(output)[0]
            if isinstance(transform, OutcomeTransform)
            else transform(output)
        )


class TensorTransform(ABC, GPModule):
    """Base of callable array transforms, independent of a torch runtime."""

    def __call__(self, values: Any, **kwargs: Any) -> Any:
        return self.forward(values, **kwargs)

    @abstractmethod
    def forward(self, values: Any, **kwargs: Any) -> Any:
        raise NotImplementedError


class ChainedTransform(TensorTransform):
    """Compose transforms from right to left."""

    def __init__(self, *transforms: TensorTransform) -> None:
        self.transforms = list(transforms)

    def forward(self, values: Any, **kwargs: Any) -> Any:
        for transform in reversed(self.transforms):
            values = transform(values)
        return values


class SineCosineTransform(TensorTransform):
    """Concatenate sine and cosine features, with optional scaling."""

    def __init__(self, scale: Any = None) -> None:
        self.scale = scale

    def forward(self, values: Any, **kwargs: Any) -> Any:
        sincos = concat([apply("sin", values), apply("cos", values)], dim=-1)
        return sincos if self.scale is None else apply("mul_raw", sincos, self.scale)


class InverseLengthscaleTransform(TensorTransform):
    """Divide features by a kernel's current lengthscales."""

    def __init__(self, kernel: Any) -> None:
        if not kernel.has_lengthscale:
            raise RuntimeError(f"{type(kernel)} does not implement `lengthscale`.")
        self.kernel = kernel

    def forward(self, values: Any, **kwargs: Any) -> Any:
        return apply("div_raw", values, self.kernel.lengthscale)


class OutputscaleTransform(TensorTransform):
    """Multiply features by the square root of a kernel's outputscale."""

    def __init__(self, kernel: Any) -> None:
        self.kernel = kernel

    def forward(self, values: Any, **kwargs: Any) -> Any:
        outputscale = self.kernel.outputscale
        if self.kernel.batch_shape:
            outputscale = getitem(outputscale, (..., None, None))
        return apply("mul_raw", values, apply("sqrt", outputscale))


class FeatureSelector(TensorTransform):
    """Select features along an axis, preserving repeated indices."""

    def __init__(self, indices: Any, dim: Any = -1) -> None:
        self.dim = operator.index(dim)
        self.indices = tuple(operator.index(i) for i in indices)

    def forward(self, values: Any, **kwargs: Any) -> Any:
        shape = shape_of(values)
        ndim = len(shape)
        if ndim == 0 or not -ndim <= self.dim < ndim:
            raise IndexError(f"Dimension out of range: {self.dim} for {ndim} dimensions.")
        axis = self.dim % ndim
        if any(i < 0 or i >= shape[axis] for i in self.indices):
            raise IndexError(f"Feature index out of range for axis of size {shape[axis]}.")
        kind = kind_of(values)
        if kind.lib == "torch":
            return values.index_select(axis, int_array(self.indices, (len(self.indices),), kind))
        if kind.lib == "numpy":
            return values.take(self.indices, axis=axis)
        data, _ = to_flat(values)
        inner, outer = math.prod(shape[axis + 1 :]), math.prod(shape[:axis])
        out = [
            data[(o * shape[axis] + i) * inner + j]
            for o in range(outer)
            for i in self.indices
            for j in range(inner)
        ]
        return from_flat(out, (*shape[:axis], len(self.indices), *shape[axis + 1 :]), kind)


class OutcomeUntransformer(TensorTransform):
    """Adapt an outcome transform to paths with output dimensions in a batch."""

    def __init__(self, transform: OutcomeTransform, num_outputs: Any) -> None:
        self.transform = transform
        self.num_outputs = operator.index(num_outputs)

    def forward(self, values: Any, **kwargs: Any) -> Any:
        if self.num_outputs == 1:
            untransformed = self.transform.untransform(getitem(values, (..., None)))[0]
            return getitem(untransformed, (..., 0))
        return transpose(self.transform.untransform(transpose(values))[0])


def get_input_transform(model: Any) -> InputTransform | None:
    """Return the model's input transform, when present."""
    return getattr(model, "input_transform", None)


def get_output_transform(model: Any) -> OutcomeUntransformer | None:
    """Return a path-compatible wrapper around the model's outcome transform."""
    transform = getattr(model, "outcome_transform", None)
    return None if transform is None else OutcomeUntransformer(transform, model.num_outputs)


def get_train_inputs(model: Any, transformed: bool = False) -> Any:
    """Return training inputs in the requested input space."""
    return GetTrainInputs(model, transformed=transformed)


@GetTrainInputs.register(Model)
def _get_train_inputs_model(model: Any, transformed: bool = False) -> tuple[Any]:
    # Native exact models expose original inputs in both modes; unlike GPyTorch
    # they do not replace train_inputs when toggling eval().
    original = getattr(model, "_original_train_inputs", None)
    (X,) = (original,) if original is not None else model.train_inputs
    transform = get_input_transform(model)
    return (transform.forward(X) if transformed and transform is not None else X,)


@GetTrainInputs.register(SingleTaskVariationalGP)
def _get_train_inputs_variational(model: Any, transformed: bool = False) -> tuple[Any]:
    (X,) = model.model.train_inputs
    transform = get_input_transform(model)
    return (X if transformed or transform is None else transform.untransform(X),)


@GetTrainInputs.register(
    ModelList,
)
@GetTrainInputs.register(ModelListGP)
def _get_train_inputs_list(model: Any, transformed: bool = False) -> list[Any]:
    return [get_train_inputs(m, transformed=transformed) for m in model.models]


def get_train_targets(model: Any, transformed: bool = False) -> Any:
    """Return training targets in the requested outcome space."""
    return GetTrainTargets(model, transformed=transformed)


@GetTrainTargets.register(Model)
def _get_train_targets_model(model: Any, transformed: bool = False) -> Any:
    Y = model.train_targets
    transform = getattr(model, "outcome_transform", None)
    if transformed or transform is None:
        return Y
    return OutcomeUntransformer(transform, model.num_outputs)(Y)


@GetTrainTargets.register(SingleTaskVariationalGP)
def _get_train_targets_variational(model: Any, transformed: bool = False) -> Any:
    Y = model.model.train_targets
    transform = getattr(model, "outcome_transform", None)
    if transformed or transform is None:
        return Y
    if model.num_outputs == 1:
        return OutcomeUntransformer(transform, 1)(Y)
    return transform.untransform(Y)[0]


@GetTrainTargets.register(ModelList)
@GetTrainTargets.register(ModelListGP)
def _get_train_targets_list(model: Any, transformed: bool = False) -> list[Any]:
    return [get_train_targets(m, transformed=transformed) for m in model.models]


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

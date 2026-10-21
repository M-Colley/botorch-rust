"""Callable feature maps with native arrays and graph-preserving Tensor algebra."""

from __future__ import annotations

from typing import Any

from botorch_rs import _botorch_rs as _rs
from botorch_rs._arrays import (
    broadcast_shapes,
    first_kind,
    from_flat,
    is_torch_tensor,
    kind_of,
    to_flat,
)
from botorch_rs._ops import _as_array, _requires_grad, apply, expand, full, shape_of, transpose
from botorch_rs.exceptions.errors import UnsupportedError
from botorch_rs.kernels import _batch_matrix, _flat_index, _indices
from botorch_rs.sampling.pathwise.utils import TensorTransform, TransformedModuleMixin

_KERNEL_FUNCTION: Any = None


def _native_matmul(a: Any, b: Any) -> Any:
    kind = first_kind(a, b)
    return from_flat(*_rs.feature_matmul(to_flat(a), to_flat(b)), kind)


def _matmul(a: Any, b: Any) -> Any:
    if is_torch_tensor(a) and is_torch_tensor(b):
        return a @ b
    kind = first_kind(a, b)
    a, b = _as_array(a, kind), _as_array(b, kind)
    if kind.lib == "torch":
        return a @ b
    sa, sb = shape_of(a), shape_of(b)
    if len(sa) < 2 or len(sb) < 2 or sa[-1] != sb[-2]:
        raise RuntimeError(f"Feature matrix shapes cannot be multiplied: {sa}, {sb}.")
    batch = broadcast_shapes(sa[:-2], sb[:-2])
    if batch is None:
        raise RuntimeError(f"Feature matrix batches do not broadcast: {sa}, {sb}.")
    a, b = expand(a, (*batch, *sa[-2:])), expand(b, (*batch, *sb[-2:]))
    return _native_matmul(a, b)


class FeatureMap(TransformedModuleMixin, TensorTransform):
    """A feature map with optional input and output transforms."""

    input_transform: Any = None
    output_transform: Any = None


class KernelEvaluationMap(FeatureMap):
    """Features obtained by evaluating a kernel against fixed points."""

    def __new__(cls, kernel: Any = None, points: Any = None, *args: Any, **kwargs: Any) -> Any:
        if cls is KernelEvaluationMap and is_torch_tensor(points):
            from botorch_rs.sampling.pathwise._torch_modules import TorchEvaluationFeatureMap

            return object.__new__(TorchEvaluationFeatureMap)
        return object.__new__(cls)

    def __init__(
        self, kernel: Any, points: Any, input_transform: Any = None, output_transform: Any = None
    ) -> None:
        shape = shape_of(points)
        if len(shape) < 2 or broadcast_shapes(shape[:-2], tuple(kernel.batch_shape)) is None:
            raise RuntimeError(
                f"Shape mismatch: points.shape={shape}, kernel.batch_shape={kernel.batch_shape}."
            )
        self.kernel, self.points = kernel, points
        self.input_transform, self.output_transform = input_transform, output_transform

    def forward(self, x: Any, **kwargs: Any) -> Any:  # type: ignore[override]
        return _kernel_values(self.kernel, x, self.points)

    @property
    def num_outputs(self) -> int:
        n = shape_of(self.points)[-2]
        if self.output_transform is None:
            return n
        return shape_of(self.output_transform(full((1, n), 0.0, first_kind(self.points))))[-1]

    @property
    def batch_shape(self) -> tuple[int, ...]:  # type: ignore[override]
        return tuple(self.kernel.batch_shape)


class KernelFeatureMap(FeatureMap):
    """An affine feature map ``x @ weight.T + bias``, with optional transforms."""

    kernel: Any

    def __new__(cls, kernel: Any = None, weight: Any = None, *args: Any, **kwargs: Any) -> Any:
        if cls is KernelFeatureMap and is_torch_tensor(weight):
            from botorch_rs.sampling.pathwise._torch_modules import TorchAffineFeatureMap

            return object.__new__(TorchAffineFeatureMap)
        return object.__new__(cls)

    def __init__(
        self,
        kernel: Any,
        weight: Any,
        bias: Any = None,
        input_transform: Any = None,
        output_transform: Any = None,
    ) -> None:
        # Existing path handles expose frequency metadata without a mutable
        # Python kernel. Preserve their constructor; standalone maps use
        # BoTorch's complete constructor.
        self._num_features = (
            weight if isinstance(weight, int) and not hasattr(kernel, "batch_shape") else None
        )
        self.kernel, self.weight = (
            (None, kernel) if self._num_features is not None else (kernel, weight)
        )
        self.bias = bias
        self.input_transform, self.output_transform = input_transform, output_transform
        if len(shape_of(self.weight)) < 2:
            raise ValueError("Feature weights must have at least two dimensions.")

    def forward(self, x: Any, **kwargs: Any) -> Any:  # type: ignore[override]
        if self.kernel is None:
            raise UnsupportedError(
                "A path handle's feature_map exposes frequency metadata; evaluate its path, or construct a standalone map with its kernel."
            )
        values = _matmul(x, transpose(self.weight))
        if is_torch_tensor(values):
            return values if self.bias is None else values + _as_array(self.bias, kind_of(values))
        return values if self.bias is None else apply("add_raw", values, self.bias)

    @property
    def num_outputs(self) -> int:
        if self._num_features is not None:
            return int(self._num_features)
        n = shape_of(self.weight)[-2]
        if self.output_transform is None:
            return n
        return shape_of(self.output_transform(full((n,), 0.0, first_kind(self.weight))))[-1]

    @property
    def batch_shape(self) -> tuple[int, ...]:  # type: ignore[override]
        return (
            tuple(self.kernel.batch_shape)
            if self.kernel is not None
            else shape_of(self.weight)[:-2]
        )

    def __repr__(self) -> str:
        return f"KernelFeatureMap(num_outputs={self.num_outputs})"


def _kernel_values(kernel: Any, x: Any, points: Any) -> Any:
    global _KERNEL_FUNCTION
    kind = first_kind(x, points)
    if kind.lib == "torch":
        import torch

        if isinstance(kernel, torch.nn.Module):
            value = kernel(x, points)
            return value.to_dense() if hasattr(value, "to_dense") else value
    if kind.lib != "torch" or not _requires_grad([x, points]):
        return kernel(x, points)
    import torch

    x, points = _as_array(x, kind), _as_array(points, kind)
    kbatch = kernel._aug if kernel._model is not None else tuple(kernel.batch_shape)
    batch = broadcast_shapes(tuple(x.shape[:-2]), tuple(points.shape[:-2]), kbatch)
    if batch is None:
        raise RuntimeError("Kernel and feature point batches do not broadcast.")
    x = x.expand(*batch, *x.shape[-2:])
    points = points.expand(*batch, *points.shape[-2:])
    if _KERNEL_FUNCTION is None:

        class _KernelEval(torch.autograd.Function):
            @staticmethod
            def forward(ctx: Any, kernel: Any, kbatch: tuple[int, ...], x: Any, points: Any) -> Any:
                ctx.save_for_backward(x, points)
                ctx.node = kernel._node()
                ctx.raws = kernel._raw_per_element(len(kbatch))
                ctx.kbatch = kbatch
                return kernel(x, points)

            @staticmethod
            def backward(ctx: Any, grad: Any) -> Any:
                x, points = ctx.saved_tensors
                ax, ap, ag = to_flat(x), to_flat(points), to_flat(grad)
                gx, gp = [], []
                for idx in _indices(tuple(x.shape[:-2])):
                    raw = ctx.raws[_flat_index(idx, ctx.kbatch)]
                    a, b = ctx.node.evaluate_vjp(
                        raw, _batch_matrix(ax, idx), _batch_matrix(ap, idx), _batch_matrix(ag, idx)
                    )
                    gx.extend(a[0])
                    gp.extend(b[0])
                kind = kind_of(x)
                return (
                    None,
                    None,
                    from_flat(gx, tuple(x.shape), kind),
                    from_flat(gp, tuple(points.shape), kind),
                )

        _KERNEL_FUNCTION = _KernelEval
    return _KERNEL_FUNCTION.apply(kernel, kbatch, x, points)


__all__ = ["FeatureMap", "KernelEvaluationMap", "KernelFeatureMap"]

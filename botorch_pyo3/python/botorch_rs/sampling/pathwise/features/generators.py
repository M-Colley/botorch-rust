"""Native random Fourier feature generation for stationary kernels."""

from __future__ import annotations

import math
import operator
import sys
from typing import Any

from botorch_rs import _botorch_rs as _rs
from botorch_rs._arrays import from_flat, kind_of
from botorch_rs.exceptions.errors import UnsupportedError
from botorch_rs.kernels import MaternKernel, RBFKernel, ScaleKernel
from botorch_rs.sampling.pathwise.features.maps import KernelFeatureMap
from botorch_rs.sampling.pathwise.utils import (
    ChainedTransform,
    FeatureSelector,
    InverseLengthscaleTransform,
    OutputscaleTransform,
    SineCosineTransform,
)
from botorch_rs.utils.dispatcher import Dispatcher

GenKernelFeatures = Dispatcher("gen_kernel_features")


def gen_kernel_features(
    kernel: Any, num_inputs: int, num_outputs: int, **kwargs: Any
) -> KernelFeatureMap:
    """Generate RBF or Matérn sine/cosine features (optional native ``seed``)."""
    torch = sys.modules.get("torch")
    if torch is not None and isinstance(kernel, torch.nn.Module):
        from botorch_rs.sampling.pathwise._torch_backend import tensor_kernel_features

        return tensor_kernel_features(kernel, num_inputs, num_outputs, **kwargs)
    return GenKernelFeatures(kernel, num_inputs=num_inputs, num_outputs=num_outputs, **kwargs)


@GenKernelFeatures.register(RBFKernel)
@GenKernelFeatures.register(MaternKernel)
def _stationary(
    kernel: Any, *, num_inputs: int, num_outputs: int, seed: int | None = None
) -> KernelFeatureMap:
    num_inputs, num_outputs = operator.index(num_inputs), operator.index(num_outputs)
    if num_outputs <= 0 or num_outputs % 2:
        raise UnsupportedError(
            f"Expected an even positive number of output features, but received num_outputs={num_outputs}."
        )
    if num_inputs <= 0:
        raise ValueError("num_inputs must be positive.")
    input_transform: Any = InverseLengthscaleTransform(kernel)
    if kernel.active_dims is not None:
        num_inputs = len(kernel.active_dims)
        input_transform = ChainedTransform(input_transform, FeatureSelector(kernel.active_dims))
    batch = tuple(kernel.batch_shape)
    count = math.prod(batch)
    if count == 0:
        raise ValueError("Kernel batch dimensions must be positive.")
    family = (
        "rbf"
        if isinstance(kernel, RBFKernel)
        else {0.5: "matern12", 1.5: "matern32", 2.5: "matern52"}[kernel.nu]
    )
    data, _ = _rs.feature_frequencies(family, num_inputs, count * num_outputs, seed)
    weight = from_flat(data, (*batch, num_outputs // 2, num_inputs), kind_of(kernel.lengthscale))
    return KernelFeatureMap(
        kernel,
        weight,
        input_transform=input_transform,
        output_transform=SineCosineTransform(math.sqrt(2 / num_outputs)),
    )


@GenKernelFeatures.register(ScaleKernel)
def _scale(
    kernel: Any, *, num_inputs: int, num_outputs: int, seed: int | None = None
) -> KernelFeatureMap:
    base_inputs = len(kernel.active_dims) if kernel.active_dims is not None else num_inputs
    feature_map = gen_kernel_features(
        kernel.base_kernel, num_inputs=base_inputs, num_outputs=num_outputs, seed=seed
    )
    if kernel.active_dims is not None and kernel.base_kernel.active_dims is None:
        feature_map.input_transform = ChainedTransform(
            feature_map.input_transform, FeatureSelector(kernel.active_dims)
        )
    feature_map.output_transform = ChainedTransform(
        OutputscaleTransform(kernel), feature_map.output_transform
    )
    return feature_map


__all__ = ["gen_kernel_features"]

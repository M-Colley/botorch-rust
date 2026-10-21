"""Lazy Torch module protocols for public paths and feature maps."""

from __future__ import annotations

from typing import Any

from torch import Tensor, nn

from botorch_rs.exceptions.errors import UnsupportedError
from botorch_rs.models.transforms.input import InputTransform
from botorch_rs.models.transforms.outcome import OutcomeTransform
from botorch_rs.sampling.pathwise.features.maps import KernelEvaluationMap, KernelFeatureMap
from botorch_rs.sampling.pathwise.paths import GeneralizedLinearPath, PathDict, PathList, SamplePath


class TorchTransformedModule(nn.Module):
    """Keep transforms outside the module's hooks, as in the source mixin."""

    _tensor_backend = "torch"

    @property
    def input_transform(self) -> Any:
        return self._modules.get("input_transform", self.__dict__.get("_input_transform"))

    @input_transform.setter
    def input_transform(self, value: Any) -> None:
        self.__dict__["_input_transform"] = value

    @property
    def output_transform(self) -> Any:
        return self._modules.get("output_transform", self.__dict__.get("_output_transform"))

    @output_transform.setter
    def output_transform(self, value: Any) -> None:
        self.__dict__["_output_transform"] = value

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        if args:
            values, args = args[0], args[1:]
        else:
            key = "values" if "values" in kwargs else "x"
            values = kwargs.pop(key)
        transform = getattr(self, "input_transform", None)
        if transform is not None:
            values = (
                transform.forward(values)
                if isinstance(transform, InputTransform)
                else transform(values)
            )
        result = nn.Module.__call__(self, values, *args, **kwargs)
        transform = getattr(self, "output_transform", None)
        if transform is None:
            return result
        return (
            transform.untransform(result)[0]
            if isinstance(transform, OutcomeTransform)
            else transform(result)
        )


def register_tensor(module: nn.Module, name: str, value: Tensor | None) -> None:
    if isinstance(value, nn.Parameter):
        module.register_parameter(name, value)
    else:
        module.register_buffer(name, value)


class TorchSamplePathModule(TorchTransformedModule, SamplePath):
    pass


class TorchLinearPathModule(TorchTransformedModule, GeneralizedLinearPath):
    _tensor_backend = "torch"

    def __init__(
        self,
        feature_map: Any,
        weight: Tensor,
        bias_module: Any = None,
        input_transform: Any = None,
        output_transform: Any = None,
        is_ensemble: bool = False,
        ensemble_as_batch: bool = False,
    ) -> None:
        nn.Module.__init__(self)
        if not callable(feature_map):
            raise TypeError("feature_map must be callable.")
        if weight.ndim == 0:
            raise ValueError("Path weights require a feature dimension.")
        if bias_module is not None and not callable(bias_module):
            raise TypeError("bias_module must be callable.")
        self.feature_map = feature_map
        register_tensor(self, "weight", weight)
        self.bias_module = bias_module
        self.input_transform, self.output_transform = input_transform, output_transform
        self.is_ensemble, self.ensemble_as_batch = bool(is_ensemble), bool(ensemble_as_batch)

    # Override the native snapshot descriptors while retaining their public names.
    @property
    def weight(self) -> Tensor:
        for state in (self._parameters, self._buffers):
            if "weight" in state:
                return state["weight"]  # type: ignore[return-value]
        raise AttributeError("weight")

    @weight.setter
    def weight(self, value: Tensor) -> None:
        self._buffers["weight"] = value

    @property
    def feature_map(self) -> Any:
        if "feature_map" in self._modules:
            return self._modules["feature_map"]
        return self.__dict__["_callable_feature_map"]

    @feature_map.setter
    def feature_map(self, value: Any) -> None:
        self.__dict__["_callable_feature_map"] = value


class TorchPathDictModule(TorchTransformedModule, PathDict):
    paths: Any
    forward = PathDict.forward

    def __init__(
        self,
        paths: Any = None,
        join: Any = None,
        input_transform: Any = None,
        output_transform: Any = None,
    ) -> None:
        nn.Module.__init__(self)
        if join is None and output_transform is not None:
            raise UnsupportedError("Output transforms must be preceded by a join rule.")
        self.join = join
        self.input_transform, self.output_transform = input_transform, output_transform
        self.paths = paths if isinstance(paths, nn.ModuleDict) else nn.ModuleDict(paths or {})


class TorchPathListModule(TorchTransformedModule, PathList):
    paths: Any
    forward = PathList.forward

    def __init__(
        self,
        paths: Any = None,
        join: Any = None,
        input_transform: Any = None,
        output_transform: Any = None,
    ) -> None:
        paths = self.__dict__.pop("_constructor_paths", paths)
        nn.Module.__init__(self)
        if join is None and output_transform is not None:
            raise UnsupportedError("Output transforms must be preceded by a join rule.")
        self.join = join
        self.input_transform, self.output_transform = input_transform, output_transform
        self.paths = paths if isinstance(paths, nn.ModuleList) else nn.ModuleList(paths or [])


class TorchAffineFeatureMap(TorchTransformedModule, KernelFeatureMap):
    forward = KernelFeatureMap.forward

    def __init__(
        self,
        kernel: Any,
        weight: Tensor,
        bias: Any = None,
        input_transform: Any = None,
        output_transform: Any = None,
    ) -> None:
        nn.Module.__init__(self)
        self._num_features = None
        self.kernel = kernel
        register_tensor(self, "weight", weight)
        if bias is None or isinstance(bias, Tensor):
            register_tensor(self, "bias", bias)
        else:
            self.bias = bias
        self.input_transform, self.output_transform = input_transform, output_transform
        if weight.ndim < 2:
            raise ValueError("Feature weights must have at least two dimensions.")


class TorchEvaluationFeatureMap(TorchTransformedModule, KernelEvaluationMap):
    forward = KernelEvaluationMap.forward

    def __init__(
        self, kernel: Any, points: Tensor, input_transform: Any = None, output_transform: Any = None
    ) -> None:
        nn.Module.__init__(self)
        # Preserve the public constructor's broadcast validation.
        KernelEvaluationMap.__init__(self, kernel, points, input_transform, output_transform)
        del self.points
        register_tensor(self, "points", points)

"""Sample paths (``botorch.sampling.pathwise.paths``).

``PathDict`` and ``PathList`` combine any sample paths as in BoTorch. The
paths drawn by botorch_rs (``draw_matheron_paths``,
``draw_kernel_feature_paths``, ``gaussian_update``) are computed in Rust:
evaluating one returns the values of all its samples in BoTorch's shapes,
and torch inputs that require gradients get exact gradients.
"""

from __future__ import annotations

import copy
import math
import sys
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from typing import Any

from botorch_rs import _botorch_rs as _rs
from botorch_rs._arrays import ArrayKind, from_flat, is_torch_tensor
from botorch_rs._ops import apply, getitem, shape_of
from botorch_rs.exceptions.errors import UnsupportedError
from botorch_rs.sampling.pathwise.features.maps import KernelFeatureMap, _matmul
from botorch_rs.sampling.pathwise.utils import arrange, input_shape, path_values


class SamplePath:
    """Base class of the sample paths: ``path(x)`` applies the optional
    ``input_transform``, evaluates ``forward`` and applies the optional
    ``output_transform``."""

    input_transform: Any = None
    output_transform: Any = None

    def __call__(self, x: Any, **kwargs: Any) -> Any:
        from botorch_rs.models.transforms.input import InputTransform
        from botorch_rs.models.transforms.outcome import OutcomeTransform

        if self.input_transform is not None:
            transform = self.input_transform
            x = transform.forward(x) if isinstance(transform, InputTransform) else transform(x)
        out = self.forward(x, **kwargs)
        if self.output_transform is None:
            return out
        transform = self.output_transform
        return (
            transform.untransform(out)[0]
            if isinstance(transform, OutcomeTransform)
            else transform(out)
        )

    def forward(self, x: Any, **kwargs: Any) -> Any:  # pragma: no cover - abstract
        raise NotImplementedError

    def set_ensemble_as_batch(self, ensemble_as_batch: bool) -> None:
        """Sets whether the ensemble dimension is a batch dimension (no effect:
        botorch_rs models are not ensembles)."""


class PathDict(SamplePath):
    """A dictionary of sample paths whose outputs are combined by ``join``
    (a dict of the outputs if ``join`` is ``None``)."""

    def __new__(cls, paths: Any = None, *args: Any, **kwargs: Any) -> Any:
        torch = sys.modules.get("torch")
        if cls is PathDict and torch is not None:
            if all(isinstance(path, torch.nn.Module) for path in (paths or {}).values()):
                from botorch_rs.sampling.pathwise._torch_modules import TorchPathDictModule

                return object.__new__(TorchPathDictModule)
        return object.__new__(cls)

    def __init__(
        self,
        paths: Mapping[str, SamplePath] | None = None,
        join: Callable[[list[Any]], Any] | None = None,
        input_transform: Any = None,
        output_transform: Any = None,
    ) -> None:
        if join is None and output_transform is not None:
            raise UnsupportedError("Output transforms must be preceded by a join rule.")
        self.join = join
        self.input_transform = input_transform
        self.output_transform = output_transform
        self.paths: dict[str, SamplePath] = dict(paths or {})

    def forward(self, x: Any, **kwargs: Any) -> Any:
        out = [path(x, **kwargs) for path in self.paths.values()]
        return dict(zip(self.paths, out, strict=True)) if self.join is None else self.join(out)

    def items(self) -> Iterable[tuple[str, SamplePath]]:
        return self.paths.items()

    def keys(self) -> Iterable[str]:
        return self.paths.keys()

    def values(self) -> Iterable[SamplePath]:
        return self.paths.values()

    def __len__(self) -> int:
        return len(self.paths)

    def __iter__(self) -> Iterator[str]:
        yield from self.paths

    def __delitem__(self, key: str) -> None:
        del self.paths[key]

    def __getitem__(self, key: str) -> SamplePath:
        return self.paths[key]

    def __setitem__(self, key: str, val: SamplePath) -> None:
        self.paths[key] = val

    def set_ensemble_as_batch(self, ensemble_as_batch: bool) -> None:
        for path in self.paths.values():
            path.set_ensemble_as_batch(ensemble_as_batch)


class PathList(SamplePath):
    """A list of sample paths whose outputs are combined by ``join`` (a list
    of the outputs if ``join`` is ``None``)."""

    def __new__(cls, paths: Any = None, *args: Any, **kwargs: Any) -> Any:
        torch = sys.modules.get("torch")
        if cls is PathList and torch is not None:
            paths = [] if paths is None else list(paths)
            if all(isinstance(path, torch.nn.Module) for path in paths):
                from botorch_rs.sampling.pathwise._torch_modules import TorchPathListModule

                result: Any = object.__new__(TorchPathListModule)
                result.__dict__["_constructor_paths"] = paths
                return result
            # A one-shot iterable must remain available to the public initializer.
            result = object.__new__(cls)
            result.__dict__["_constructor_paths"] = paths
            return result
        return object.__new__(cls)

    def __init__(
        self,
        paths: Iterable[SamplePath] | None = None,
        join: Callable[[list[Any]], Any] | None = None,
        input_transform: Any = None,
        output_transform: Any = None,
    ) -> None:
        if join is None and output_transform is not None:
            raise UnsupportedError("Output transforms must be preceded by a join rule.")
        self.join = join
        self.input_transform = input_transform
        self.output_transform = output_transform
        self.paths: list[SamplePath] = list(self.__dict__.pop("_constructor_paths", paths) or [])

    def forward(self, x: Any, **kwargs: Any) -> Any:
        out = [path(x, **kwargs) for path in self.paths]
        return out if self.join is None else self.join(out)

    def __len__(self) -> int:
        return len(self.paths)

    def __iter__(self) -> Iterator[SamplePath]:
        yield from self.paths

    def __delitem__(self, key: int) -> None:
        del self.paths[key]

    def __getitem__(self, key: int) -> SamplePath:
        return self.paths[key]

    def __setitem__(self, key: int, val: SamplePath) -> None:
        self.paths[key] = val

    def set_ensemble_as_batch(self, ensemble_as_batch: bool) -> None:
        for path in self.paths:
            path.set_ensemble_as_batch(ensemble_as_batch)


class _RsPaths:
    """The paths of the outputs ``outputs`` of a handle (all outputs of a
    ``SingleTaskGP``, or of one model of a ``ModelListGP``) with sample shape
    ``sample_shape``."""

    def __init__(
        self,
        handle: _rs.SamplePaths,
        sample_shape: tuple[int, ...],
        outputs: Sequence[int],
        kind: ArrayKind,
        component: str | None = None,
    ) -> None:
        self._handle = handle
        self._sample_shape = tuple(sample_shape)
        self._outputs = list(outputs)
        self._kind = kind
        self._component = component

    @property
    def _multi_output(self) -> bool:
        return len(self._outputs) > 1

    def _values(self, x: Any) -> Any:
        shape = input_shape(x, self._handle.dim)
        pv = path_values(self._handle, x, self._component)
        return arrange(
            pv,
            self._handle.num_samples,
            math.prod(shape[:-1]),
            shape,
            self._sample_shape,
            self._outputs,
            self._multi_output,
        )

    def __deepcopy__(self, memo: dict[int, Any]) -> Any:
        copied = self.__class__.__new__(self.__class__)
        memo[id(self)] = copied
        for key, value in self.__dict__.items():
            # The extension handle is immutable; sharing its fixed model and
            # random draws preserves the snapshot without serializing it.
            copied.__dict__[key] = value if key == "_handle" else copy.deepcopy(value, memo)
        return copied

    def _stacked(self, mats: Sequence[tuple[Sequence[float], Sequence[int]]]) -> Any:
        """Per-output ``S x k`` matrices as ``sample_shape x [m x] k``."""
        mats = [mats[t] for t in self._outputs]
        k = int(mats[0][1][1])
        s_total = int(mats[0][1][0])
        data: list[float] = []
        for s in range(s_total):
            for values, _ in mats:
                data.extend(values[s * k : (s + 1) * k])
        shape = self._sample_shape + ((len(mats),) if self._multi_output else ()) + (k,)
        return from_flat(data, shape, self._kind)

    def _per_output(self, mats: Sequence[tuple[Sequence[float], Sequence[int]]]) -> Any:
        """Per-output matrices of identical shape as ``[m x] r x c``."""
        mats = [mats[t] for t in self._outputs]
        data: list[float] = []
        for values, _ in mats:
            data.extend(values)
        shape = ((len(mats),) if self._multi_output else ()) + tuple(int(v) for v in mats[0][1])
        return from_flat(data, shape, self._kind)


class GeneralizedLinearPath(_RsPaths, SamplePath):
    """A sample path ``path(x) = bias(z) + feature_map(z)^T weight`` with ``z``
    the model's input transform of ``x`` (BoTorch's ``GeneralizedLinearPath``):
    the prior paths of ``draw_kernel_feature_paths`` (random Fourier
    features, with the model's outcome transform) and the updates of
    ``gaussian_update`` (the kernel evaluated at the training inputs, in the
    transformed outcome space), computed in Rust.

    ``weight`` has shape ``sample_shape x [m x] num_features`` (``m`` for a
    multi-output ``SingleTaskGP``).
    """

    is_ensemble = False
    ensemble_as_batch = False

    def __new__(cls, feature_map: Any = None, weight: Any = None, *args: Any, **kwargs: Any) -> Any:
        if cls is GeneralizedLinearPath and is_torch_tensor(weight):
            from botorch_rs.sampling.pathwise._torch_backend import TorchGeneralizedLinearPath

            return object.__new__(TorchGeneralizedLinearPath)
        return object.__new__(cls)

    def __init__(
        self,
        feature_map: Any,
        weight: Any,
        bias_module: Any = None,
        input_transform: Any = None,
        output_transform: Any = None,
        is_ensemble: bool = False,
        ensemble_as_batch: bool = False,
    ) -> None:
        if not callable(feature_map):
            raise TypeError("feature_map must be callable.")
        if not shape_of(weight):
            raise ValueError("Path weights require a feature dimension.")
        if bias_module is not None and not callable(bias_module):
            raise TypeError("bias_module must be callable.")
        self._explicit_feature_map = feature_map
        self._explicit_weight = weight
        self.bias_module = bias_module
        self.input_transform, self.output_transform = input_transform, output_transform
        self.is_ensemble, self.ensemble_as_batch = bool(is_ensemble), bool(ensemble_as_batch)

    @classmethod
    def _from_native(
        cls,
        handle: _rs.SamplePaths,
        sample_shape: tuple[int, ...],
        outputs: Sequence[int],
        kind: ArrayKind,
        component: str | None = None,
    ) -> GeneralizedLinearPath:
        path = cls.__new__(cls)
        _RsPaths.__init__(path, handle, sample_shape, outputs, kind, component)
        path.bias_module = None
        return path

    def forward(self, x: Any, **kwargs: Any) -> Any:
        if "_explicit_feature_map" not in self.__dict__:
            return self._values(x)
        if self.is_ensemble and not self.ensemble_as_batch:
            x = getitem(x, (..., None, slice(None), slice(None)))
        features = self.feature_map(x, **kwargs)
        out = getitem(_matmul(features, getitem(self.weight, (..., None))), (..., 0))
        return out if self.bias_module is None else apply("add_raw", out, self.bias_module(x))

    def set_ensemble_as_batch(self, ensemble_as_batch: bool) -> None:
        self.ensemble_as_batch = bool(ensemble_as_batch)

    @property
    def weight(self) -> Any:
        """The weights of the features (``sample_shape x [m x] num_features``)."""
        if "_explicit_weight" in self.__dict__:
            return self._explicit_weight
        prior = self._handle.kind == "prior" or self._component == "prior_paths"
        if prior:
            return self._stacked([(w[0], w[1]) for _, w, _ in self._handle.draws()])
        return self._stacked(self._handle.update_weights())

    @weight.setter
    def weight(self, value: Any) -> None:
        if "_explicit_feature_map" not in self.__dict__:
            raise UnsupportedError(
                "Native path weights are immutable snapshots; construct a GeneralizedLinearPath with explicit weights to modify them."
            )
        self._explicit_weight = value

    @property
    def feature_map(self) -> Any:
        """The random Fourier features of prior paths (``None`` for updates,
        whose features are kernel evaluations at the training inputs)."""
        if "_explicit_feature_map" in self.__dict__:
            return self._explicit_feature_map
        if not (self._handle.kind == "prior" or self._component == "prior_paths"):
            return None
        draws = self._handle.draws()
        freqs = self._per_output([(f[0], f[1]) for f, _, _ in draws])
        return KernelFeatureMap(freqs, 2 * int(draws[self._outputs[0]][0][1][0]))

    @feature_map.setter
    def feature_map(self, value: Any) -> None:
        if "_explicit_feature_map" not in self.__dict__:
            raise UnsupportedError("Native feature maps are immutable snapshots.")
        self._explicit_feature_map = value

    def __repr__(self) -> str:
        if "_explicit_feature_map" in self.__dict__:
            return f"GeneralizedLinearPath(weight_shape={shape_of(self.weight)}, is_ensemble={self.is_ensemble})"
        return (
            f"GeneralizedLinearPath(sample_shape={self._sample_shape}, "
            f"num_outputs={len(self._outputs)})"
        )


__all__ = ["GeneralizedLinearPath", "PathDict", "PathList", "SamplePath"]

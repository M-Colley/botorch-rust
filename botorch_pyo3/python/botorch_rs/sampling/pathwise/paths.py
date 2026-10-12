"""Sample paths (``botorch.sampling.pathwise.paths``).

``PathDict`` and ``PathList`` combine any sample paths as in BoTorch. The
paths drawn by botorch_rs (``draw_matheron_paths``,
``draw_kernel_feature_paths``, ``gaussian_update``) are computed in Rust:
evaluating one returns the values of all its samples in BoTorch's shapes,
and torch inputs that require gradients get exact gradients.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from typing import Any

from botorch_rs import _botorch_rs as _rs
from botorch_rs._arrays import ArrayKind, from_flat
from botorch_rs.exceptions.errors import UnsupportedError
from botorch_rs.sampling.pathwise.features.maps import KernelFeatureMap
from botorch_rs.sampling.pathwise.utils import arrange, input_shape, path_values


class SamplePath:
    """Base class of the sample paths: ``path(x)`` applies the optional
    ``input_transform``, evaluates ``forward`` and applies the optional
    ``output_transform``."""

    input_transform: Any = None
    output_transform: Any = None

    def __call__(self, x: Any, **kwargs: Any) -> Any:
        if self.input_transform is not None:
            x = self.input_transform(x)
        out = self.forward(x, **kwargs)
        return out if self.output_transform is None else self.output_transform(out)

    def forward(self, x: Any, **kwargs: Any) -> Any:  # pragma: no cover - abstract
        raise NotImplementedError

    def set_ensemble_as_batch(self, ensemble_as_batch: bool) -> None:
        """Sets whether the ensemble dimension is a batch dimension (no effect:
        botorch_rs models are not ensembles)."""


class PathDict(SamplePath):
    """A dictionary of sample paths whose outputs are combined by ``join``
    (a dict of the outputs if ``join`` is ``None``)."""

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
        self.paths: list[SamplePath] = list(paths or [])

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

    def forward(self, x: Any, **kwargs: Any) -> Any:
        return self._values(x)

    @property
    def weight(self) -> Any:
        """The weights of the features (``sample_shape x [m x] num_features``)."""
        prior = self._handle.kind == "prior" or self._component == "prior_paths"
        if prior:
            return self._stacked([(w[0], w[1]) for _, w, _ in self._handle.draws()])
        return self._stacked(self._handle.update_weights())

    @property
    def feature_map(self) -> KernelFeatureMap | None:
        """The random Fourier features of prior paths (``None`` for updates,
        whose features are kernel evaluations at the training inputs)."""
        if not (self._handle.kind == "prior" or self._component == "prior_paths"):
            return None
        draws = self._handle.draws()
        freqs = self._per_output([(f[0], f[1]) for f, _, _ in draws])
        return KernelFeatureMap(freqs, 2 * int(draws[self._outputs[0]][0][1][0]))

    def __repr__(self) -> str:
        return (
            f"GeneralizedLinearPath(sample_shape={self._sample_shape}, "
            f"num_outputs={len(self._outputs)})"
        )


__all__ = ["GeneralizedLinearPath", "PathDict", "PathList", "SamplePath"]

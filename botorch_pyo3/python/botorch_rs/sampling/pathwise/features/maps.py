"""Feature maps of sample paths (``botorch.sampling.pathwise.features.maps``).

botorch_rs computes the features of its paths in Rust; the feature maps
here describe the random Fourier features of prior paths (their
frequencies) and are not evaluated from Python.
"""

from __future__ import annotations

from typing import Any

from botorch_rs.exceptions.errors import UnsupportedError


class FeatureMap:
    """Base class of the feature maps."""

    input_transform: Any = None
    output_transform: Any = None

    @property
    def num_outputs(self) -> int:  # pragma: no cover - abstract
        raise NotImplementedError

    def __call__(self, x: Any, **kwargs: Any) -> Any:
        raise UnsupportedError(
            "botorch_rs feature maps are evaluated in Rust as part of their paths; evaluate the "
            "path instead."
        )


class KernelFeatureMap(FeatureMap):
    """The random Fourier features ``sqrt(2 / L) [sin(u), cos(u)]`` (times the
    square root of the outputscale) with ``u = (x / lengthscale) @ weight.T``
    of an RBF or Matérn-5/2 kernel: ``weight`` holds the ``L / 2``
    frequencies (``[m x] L/2 x d``)."""

    def __init__(self, weight: Any, num_features: int) -> None:
        self.weight = weight
        self.bias = None
        self._num_features = num_features

    @property
    def num_outputs(self) -> int:
        """The number of features ``L``."""
        return self._num_features

    def __repr__(self) -> str:
        return f"KernelFeatureMap(num_outputs={self._num_features})"


__all__ = ["FeatureMap", "KernelFeatureMap"]

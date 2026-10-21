"""Transformed posteriors (``botorch.posteriors.transformed``).

A ``TransformedPosterior`` is a posterior whose samples are the samples of
another posterior mapped by ``sample_transform``; its mean and variance are
given by ``mean_transform`` / ``variance_transform`` of the underlying mean
and variance, where defined (``NotImplementedError`` otherwise). Models with
a nonlinear outcome transform (``Log``, ``Power``, ``Bilog``) return one from
``posterior``; they compute its samples, moments and their derivatives
natively, with the same base samples as the underlying Gaussian posterior
(``_posterior``).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from botorch_rs._arrays import broadcast_to, first_kind, from_flat, to_flat
from botorch_rs.posteriors.gpytorch import GPyTorchPosterior, Posterior


def _affine(x: Any, offset: Any, scale: Any, power: int) -> Any:
    """``offset + scale^power * x`` with ``offset`` and ``scale`` given as
    ``(data, shape)`` pairs of shape ``batch_shape x 1 x m`` (broadcast over
    the trailing output dimension of ``x``)."""
    kind = first_kind(x)
    if kind.lib == "torch":
        from botorch_rs import _ops

        coefficient = from_flat([v**power for v in scale[0]], tuple(scale[1]), kind)
        result = _ops.apply("mul_raw", x, coefficient)
        if offset is not None:
            result = _ops.apply("add_raw", result, from_flat(offset[0], tuple(offset[1]), kind))
        return result
    data, shape = to_flat(x, "x")
    sdata = broadcast_to(scale[0], scale[1], shape)
    odata = broadcast_to(offset[0], offset[1], shape) if offset is not None else None
    out = []
    for i, v in enumerate(data):
        s = sdata[i] ** power
        o = odata[i] if odata is not None else 0.0
        out.append(o + s * v if odata is not None else s * v)
    return from_flat(out, shape, first_kind(x))


class TransformedPosterior(Posterior):
    r"""A generic transformation of a posterior (implicitly represented)."""

    def __init__(
        self,
        posterior: Any,
        sample_transform: Callable[[Any], Any],
        mean_transform: Callable[[Any, Any], Any] | None = None,
        variance_transform: Callable[[Any, Any], Any] | None = None,
    ) -> None:
        self._posterior = posterior
        self._sample_transform = sample_transform
        self._mean_transform = mean_transform
        self._variance_transform = variance_transform

    @property
    def _tensor_backend(self) -> Any:
        return getattr(self._posterior, "_tensor_backend", None)

    @property
    def _kind(self) -> Any:
        return getattr(self._posterior, "_kind", None)

    @property
    def batch_shape(self) -> Any:
        return self._posterior.batch_shape

    @property
    def base_sample_shape(self) -> Any:
        r"""The shape of a base sample used for constructing posterior samples."""
        return self._posterior.base_sample_shape

    @property
    def batch_range(self) -> tuple[int, int]:
        r"""The t-batch range (of the underlying posterior)."""
        return self._posterior.batch_range

    @property
    def device(self) -> Any:
        return getattr(self._posterior, "device", None)

    @property
    def dtype(self) -> Any:
        return getattr(self._posterior, "dtype", None)

    def _extended_shape(self, sample_shape: Any = ()) -> tuple[int, ...]:
        r"""The shape of the samples for ``sample_shape`` (the sample transform
        does not change it)."""
        return self._posterior._extended_shape(sample_shape)

    @property
    def mean(self) -> Any:
        r"""The mean of the posterior as a ``batch_shape x n x m`` array."""
        if self._mean_transform is None:
            raise NotImplementedError("No mean transform provided.")
        try:
            variance = self._posterior.variance
        except (NotImplementedError, AttributeError):
            variance = None
        return self._mean_transform(self._posterior.mean, variance)

    @property
    def variance(self) -> Any:
        r"""The variance of the posterior as a ``batch_shape x n x m`` array."""
        if self._variance_transform is None:
            raise NotImplementedError("No variance transform provided.")
        return self._variance_transform(self._posterior.mean, self._posterior.variance)

    def rsample_from_base_samples(self, sample_shape: Any, base_samples: Any) -> Any:
        r"""Samples from base samples (those of the underlying posterior)."""
        samples = self._posterior.rsample_from_base_samples(sample_shape, base_samples)
        return self._sample_transform(samples)

    def rsample(self, sample_shape: Any = None) -> Any:
        r"""Samples of shape ``sample_shape x batch_shape x q x m``."""
        return self._sample_transform(self._posterior.rsample(sample_shape))

    def __repr__(self) -> str:
        return f"TransformedPosterior({self._posterior!r})"


class _ModelTransformedPosterior(TransformedPosterior):
    """The ``TransformedPosterior`` of a botorch_rs model: samples, mean and
    variance are computed natively (``native``: the model's posterior with
    its outcome transform), ``_posterior`` is the Gaussian base posterior
    (the affine un-transforms applied, BoTorch's posterior before the first
    nonlinear un-transform)."""

    def __init__(self, native: GPyTorchPosterior, base: GPyTorchPosterior, moments: bool) -> None:
        self._native = native
        self._moments = moments
        super().__init__(
            posterior=base,
            sample_transform=self._transform_samples,
            mean_transform=(lambda m, v: native.mean) if moments else None,
            variance_transform=(lambda m, v: native.variance) if moments else None,
        )

    def _transform_samples(self, samples: Any) -> Any:  # pragma: no cover - not used natively
        raise NotImplementedError(
            "The sample transform of a model's TransformedPosterior is applied natively; use "
            "rsample or rsample_from_base_samples."
        )

    @property
    def mean(self) -> Any:
        if not self._moments:
            raise NotImplementedError("No mean transform provided.")
        return self._native.mean

    @property
    def variance(self) -> Any:
        if not self._moments:
            raise NotImplementedError("No variance transform provided.")
        return self._native.variance

    def rsample_from_base_samples(self, sample_shape: Any, base_samples: Any) -> Any:
        return self._native.rsample_from_base_samples(sample_shape, base_samples)

    def rsample(self, sample_shape: Any = None) -> Any:
        return self._native.rsample(sample_shape)

    @property
    def batch_shape(self) -> tuple[int, ...]:
        return self._native.batch_shape


__all__ = ["TransformedPosterior"]

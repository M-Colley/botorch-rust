"""Quasi-MC normal samplers (``botorch.sampling.normal``)."""

from __future__ import annotations

import math
from typing import Any

from botorch_rs import _botorch_rs as _rs
from botorch_rs._arrays import Pair, from_flat, to_flat
from botorch_rs.exceptions.errors import BotorchTensorDimensionError, UnsupportedError
from botorch_rs.posteriors.gpytorch import GPyTorchPosterior


def _sample_shape(sample_shape: Any) -> tuple[int, ...]:
    if isinstance(sample_shape, int):
        return (sample_shape,)
    try:
        if isinstance(sample_shape, (str, bytes)):
            raise TypeError
        shape = tuple(int(s) for s in sample_shape)
    except (TypeError, ValueError) as e:
        raise TypeError(
            f"Expected `sample_shape` to be a torch.Size, a tuple or an int, got {sample_shape!r}."
        ) from e
    if not shape:
        raise ValueError("`sample_shape` must contain at least one dimension.")
    return shape


class MCSampler:
    """Base class of the Monte-Carlo samplers."""

    sample_shape: tuple[int, ...]


class SobolQMCNormalSampler(MCSampler):
    r"""Draws standard-normal base samples from a scrambled Sobol sequence.

    For ``q`` jointly sampled points the base samples are a fixed
    ``num_samples x q`` matrix (``num_samples = prod(sample_shape)``), a
    deterministic function of the seed, so that MC acquisition functions are
    smooth, deterministic sample-average approximations.

    The Sobol sequence is Owen-scrambled (``sobol_burley``), so the samples
    differ from torch's ``SobolEngine`` for the same seed (with the same
    distribution). Explicit base samples can be assigned to ``base_samples``
    (shape ``num_samples x q``, or BoTorch's ``num_samples x 1 x q``).
    """

    def __init__(self, sample_shape: Any, seed: int | None = None, **kwargs: Any) -> None:
        if kwargs:
            raise UnsupportedError(f"Unsupported sampler arguments: {sorted(kwargs)}.")
        self.sample_shape = _sample_shape(sample_shape)
        self._rs = _rs.SobolQMCNormalSampler(
            math.prod(self.sample_shape), None if seed is None else int(seed)
        )
        self._base_samples: Any = None

    @property
    def seed(self) -> int:
        """The seed of the Sobol sequence (drawn at random if not given)."""
        return self._rs.seed

    @property
    def num_samples(self) -> int:
        return self._rs.num_samples

    @property
    def base_samples(self) -> Any:
        """The explicitly assigned base samples, or ``None``."""
        return self._base_samples

    @base_samples.setter
    def base_samples(self, value: Any) -> None:
        if value is None:
            raise UnsupportedError("Assigned base samples cannot be removed.")
        data, shape = to_flat(value, "base_samples")
        n = self._rs.num_samples
        if not shape or shape[0] != n:
            raise BotorchTensorDimensionError(
                f"Expected base samples with {n} rows (num_samples), got shape {shape}."
            )
        non_one = [s for s in shape[1:] if s != 1]
        if len(shape) == 1 or len(non_one) > 1:
            raise BotorchTensorDimensionError(
                f"Base samples of shape {shape} must be `num_samples x q` (or "
                "`num_samples x 1 x q`); batched base samples are not supported."
            )
        q = non_one[0] if non_one else 1
        self._rs.set_base_samples((data, (n, q)))
        self._base_samples = value

    def forward(self, posterior: GPyTorchPosterior) -> Any:
        """Posterior samples, shape ``sample_shape x batch_shape x q x 1``; the
        same base samples are used for every batch."""
        if not isinstance(posterior, GPyTorchPosterior):
            raise UnsupportedError("The sampler requires a botorch_rs GPyTorchPosterior.")
        q = posterior._q
        z: Pair = self._rs.base_samples(q)
        out = posterior._rs.posterior_samples(posterior._x, z, posterior._observation_noise)
        shape = (*self.sample_shape, *posterior._batch, q, 1)
        return from_flat(out[0], shape, posterior._kind)

    __call__ = forward

    def __repr__(self) -> str:
        return f"SobolQMCNormalSampler(sample_shape={self.sample_shape}, seed={self.seed})"


__all__ = ["MCSampler", "SobolQMCNormalSampler"]

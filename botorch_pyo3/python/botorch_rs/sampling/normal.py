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


class NormalMCSampler(MCSampler):
    r"""Base class for samplers producing (possibly QMC) N(0,1) samples
    (``botorch.sampling.normal.NormalMCSampler``): ``SobolQMCNormalSampler``
    and ``IIDNormalSampler``.

    Every botorch_rs normal sampler has a Rust sampler (``_rs``) that the
    Monte-Carlo acquisition functions, fantasy models and look-ahead
    acquisition functions draw their base samples from (``num_samples x
    dim`` for ``dim`` jointly sampled values, cached per ``dim``).
    """

    _rs: Any

    @property
    def num_samples(self) -> int:
        """The number of MC samples (``prod(sample_shape)``)."""
        return math.prod(self.sample_shape)

    @property
    def seed(self) -> int:
        """The seed of the base samples."""
        return int(self._rs.seed)


class SobolQMCNormalSampler(NormalMCSampler):
    r"""Draws standard-normal base samples from a scrambled Sobol sequence.

    For ``q`` jointly sampled points the base samples are a fixed
    ``num_samples x q`` matrix (``num_samples = prod(sample_shape)``), a
    deterministic function of the seed, so that MC acquisition functions are
    smooth, deterministic sample-average approximations; a posterior over
    ``m`` outputs uses ``num_samples x (q m)`` base samples (BoTorch's
    ``num_samples x q x m``, flattened).

    The Sobol sequence is Owen-scrambled (``sobol_burley``), so the samples
    differ from torch's ``SobolEngine`` for the same seed (with the same
    distribution). Explicit base samples can be assigned to ``base_samples``
    (shape ``num_samples x q``, BoTorch's ``num_samples x 1 x q``, or
    ``num_samples x [1 x] q x m`` for a multi-output posterior).
    """

    def __init__(self, sample_shape: Any, seed: int | None = None) -> None:
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
        if len(shape) == 1 or len(non_one) > 2:
            raise BotorchTensorDimensionError(
                f"Base samples of shape {shape} must be `num_samples x q` (or "
                "`num_samples x 1 x q`, `num_samples x q x m`); batched base samples are "
                "not supported."
            )
        self._rs.set_base_samples((data, (n, math.prod(non_one))))
        self._base_samples = value

    def forward(self, posterior: GPyTorchPosterior) -> Any:
        """Posterior samples, shape ``sample_shape x batch_shape x q x m``; the
        same base samples are used for every batch."""
        if not isinstance(posterior, GPyTorchPosterior):
            raise UnsupportedError("The sampler requires a botorch_rs GPyTorchPosterior.")
        q, m = posterior._q, posterior._m
        z: Pair = self._rs.base_samples(q * m)
        out = posterior._samples(z)
        shape = (*self.sample_shape, *posterior._batch, q, m)
        return from_flat(out, shape, posterior._kind)

    __call__ = forward

    def __repr__(self) -> str:
        return f"SobolQMCNormalSampler(sample_shape={self.sample_shape}, seed={self.seed})"


def _batch_range(posterior: Any, override: tuple[int, int] | None) -> tuple[int, int]:
    if override is not None:
        return override
    return posterior.batch_range


class IIDNormalSampler(NormalMCSampler):
    r"""Sampler for MC base samples using iid ``N(0,1)`` samples
    (``botorch.sampling.normal.IIDNormalSampler``).

    The base samples (``sample_shape x`` the posterior's ``base_sample_shape``
    with the dimensions of ``batch_range`` collapsed to 1) are BoTorch's
    draws for double-precision posteriors: ``torch.randn(shape)`` after
    ``torch.manual_seed(seed)``, from a native reproduction of torch's CPU
    generator (so torch is not needed for them), cached while their shape is
    unchanged. ``batch_range_override`` replaces the posterior's
    ``batch_range``.

    The sampler can be passed to the Monte-Carlo acquisition functions, the
    look-ahead acquisition functions and ``fantasize`` in place of a
    ``SobolQMCNormalSampler``: their base samples for ``q`` points and ``m``
    outputs are then BoTorch's (``torch.randn(n, 1, q, m)`` under
    ``manual_seed(seed)``), including the baseline rows that the cached
    Cholesky acquisition functions (``qNEI``, ``qLogNEI``, ``qNEHVI``)
    reuse.

    Example:
        >>> sampler = IIDNormalSampler(torch.Size([1000]), seed=1234)
        >>> posterior = model.posterior(test_X)
        >>> samples = sampler(posterior)
    """

    def __init__(self, sample_shape: Any, seed: int | None = None, **kwargs: Any) -> None:
        if kwargs:
            raise UnsupportedError(f"Unsupported keyword arguments: {sorted(kwargs)}.")
        self.sample_shape = _sample_shape(sample_shape)
        if seed is None:
            import importlib
            import random

            try:
                torch = importlib.import_module("torch")
            except ImportError:
                seed = random.randint(0, 999999)
            else:
                seed = int(torch.randint(0, 1000000, (1,)).item())
        self.seed = int(seed)
        self.base_samples: Any = None
        self._base_shape: tuple[int, ...] | None = None
        self.batch_range_override: tuple[int, int] | None = None

    @property
    def seed(self) -> int:
        """The seed of ``torch.manual_seed`` for the base samples."""
        return self._seed

    @seed.setter
    def seed(self, value: int) -> None:
        self._seed = int(value)

    def _draw(self, shape: tuple[int, ...], kind: Any) -> list[float]:
        # torch.randn(shape, dtype=torch.float64) after torch.manual_seed(seed).
        data, _ = _rs.torch_randn(self.seed, math.prod(shape))
        return list(data)

    @property
    def _rs(self) -> Any:  # type: ignore[override]
        """The Rust sampler of the acquisition functions (i.i.d. draws)."""
        rs = self.__dict__.get("_rs_sampler")
        if rs is None or rs.num_samples != self.num_samples or rs.seed != self.seed:
            rs = _rs.SobolQMCNormalSampler.iid(self.num_samples, self.seed)
            self.__dict__["_rs_sampler"] = rs
        return rs

    def _collapsed_shape(self, posterior: GPyTorchPosterior) -> tuple[int, ...]:
        shape = tuple(posterior.base_sample_shape)
        start, end = _batch_range(posterior, self.batch_range_override)
        end = len(shape) + end if end <= 0 else end
        start = len(shape) + start if start < 0 else start
        return (*shape[:start], *(1 for _ in shape[start:end]), *shape[end:])

    def forward(self, posterior: GPyTorchPosterior) -> Any:
        """Posterior samples, shape ``sample_shape x batch_shape x q x m``."""
        if not isinstance(posterior, GPyTorchPosterior):
            raise UnsupportedError("The sampler requires a botorch_rs GPyTorchPosterior.")
        collapsed = self._collapsed_shape(posterior)
        target = (*self.sample_shape, *collapsed)
        if self.base_samples is None or self._base_shape != target:
            draws = self._draw(target, posterior._kind)
            self.base_samples = from_flat(draws, target, posterior._kind)
            self._base_shape = target
        z, _ = to_flat(self.base_samples, "base_samples")
        # Broadcast the collapsed batch dimensions to the posterior's batch.
        full = tuple(posterior.base_sample_shape)
        n = math.prod(self.sample_shape)
        qm = posterior._q * posterior._m
        nb = math.prod(posterior._batch)
        batch_full = full[: len(full) - (2 if posterior._is_mt else 1)]
        batch_col = collapsed[: len(collapsed) - (2 if posterior._is_mt else 1)]
        if all(v == 1 for v in batch_col):
            pair: Pair = (list(z), (n, qm))
        else:
            per = math.prod(batch_col) * qm
            data: list[float] = []
            for i in range(n):
                for b in range(nb):
                    # Row-major index of b in the collapsed batch.
                    idx = 0
                    rem = b
                    stride_full = nb
                    stride_col = math.prod(batch_col)
                    for dim_f, dim_c in zip(batch_full, batch_col, strict=True):
                        stride_full //= dim_f
                        stride_col //= dim_c
                        k = rem // stride_full
                        rem %= stride_full
                        idx += (k if dim_c != 1 else 0) * stride_col
                    off = i * per + idx * qm
                    data.extend(z[off : off + qm])
            pair = (data, (n, nb, qm))
        out = posterior._samples(pair)
        shape = (*self.sample_shape, *posterior._batch, posterior._q, posterior._m)
        return from_flat(out, shape, posterior._kind)

    __call__ = forward

    def __repr__(self) -> str:
        return f"IIDNormalSampler(sample_shape={self.sample_shape}, seed={self.seed})"


__all__ = ["IIDNormalSampler", "MCSampler", "NormalMCSampler", "SobolQMCNormalSampler"]

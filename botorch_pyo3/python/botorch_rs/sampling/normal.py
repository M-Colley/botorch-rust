"""Quasi-MC normal samplers (``botorch.sampling.normal``)."""

from __future__ import annotations

import math
from typing import Any

from botorch_rs import _botorch_rs as _rs
from botorch_rs._arrays import Pair, broadcast_to, from_flat, kind_of, to_flat
from botorch_rs.exceptions.errors import BotorchTensorDimensionError, UnsupportedError
from botorch_rs.posteriors.gpytorch import GPyTorchPosterior, Posterior


def _posterior_kind(posterior: Any) -> Any:
    while not hasattr(posterior, "_kind") and hasattr(posterior, "_posterior"):
        posterior = posterior._posterior
    return getattr(posterior, "_kind", None)


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
    return shape


def _uses_tensor_protocol(posterior: Any) -> bool:
    """Recognize the public Torch posterior protocol without private markers.

    Native posteriors retain their native sampler cache even when their array
    kind is Torch. An explicit Tensor backend or a caller-supplied posterior
    with public Torch metadata uses device-resident normal generation.
    """
    if getattr(posterior, "_tensor_backend", None) == "torch":
        return True
    if _posterior_kind(posterior) is not None:
        return False
    try:
        import torch
    except ImportError:
        return False
    return isinstance(getattr(posterior, "dtype", None), torch.dtype) and isinstance(
        getattr(posterior, "device", None), torch.device
    )


class MCSampler:
    """Base class of the Monte-Carlo samplers."""

    sample_shape: tuple[int, ...]

    def forward(self, posterior: Any) -> Any:
        raise NotImplementedError

    def __call__(self, posterior: Any) -> Any:
        return self.forward(posterior=posterior)


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
    _base_shape: tuple[int, ...] | None
    _tensor_sampler_key: tuple[tuple[int, ...], Any, Any, int] | None
    base_samples: Any

    def _construct_tensor_base_samples(self, posterior: Any) -> None:
        """Construct device-resident normals without evaluating the posterior root."""
        import torch

        shape = self._get_collapsed_shape(posterior)
        current = self.base_samples
        explicit = getattr(self, "_base_samples_explicit", False)
        if (
            isinstance(self, IIDNormalSampler)
            and current is not None
            and getattr(self, "_tensor_sampler_key", None) is None
            and tuple(current.shape[: len(self.sample_shape)]) == self.sample_shape
        ):
            explicit = True
        if explicit and current is not None:
            current = torch.as_tensor(current, dtype=posterior.dtype, device=posterior.device)
        else:
            key = (shape, posterior.dtype, posterior.device, self.seed)
            if getattr(self, "_tensor_sampler_key", None) != key or current is None:
                if isinstance(self, IIDNormalSampler):
                    generator = torch.Generator(device=posterior.device).manual_seed(self.seed)
                    current = torch.randn(
                        shape, dtype=posterior.dtype, device=posterior.device, generator=generator
                    )
                else:
                    # Torch's Sobol engine maintains its integer sequence on
                    # CPU. Only newly generated uniforms are transferred;
                    # normal transforms and all GP algebra run on device.
                    dimension = math.prod(shape[len(self.sample_shape) :])
                    engine = torch.quasirandom.SobolEngine(dimension, scramble=True, seed=self.seed)
                    uniforms = engine.draw(self.num_samples, dtype=posterior.dtype).to(
                        posterior.device
                    )
                    epsilon = torch.finfo(posterior.dtype).eps
                    current = (
                        uniforms.clamp(epsilon, 1 - epsilon).mul(2).sub(1).erfinv() * math.sqrt(2)
                    ).reshape(shape)
                self._tensor_sampler_key = key
            elif isinstance(current, torch.Tensor):
                current = current.to(dtype=posterior.dtype, device=posterior.device)
        if isinstance(self, SobolQMCNormalSampler):
            self._base_samples = current
        else:
            self.base_samples = current
        self._base_shape = tuple(current.shape)

    def _tensor_samples(self, posterior: Any) -> Any:
        """Sample on the posterior device without flattening its graph."""
        from botorch_rs.posteriors._base_samples import expand_base_samples

        self._construct_tensor_base_samples(posterior)
        points = (
            getattr(posterior, "_q", None) if getattr(posterior, "_is_mt", True) is False else None
        )
        if points is not None and posterior.base_sample_shape[-1] != points:
            points = None  # Structured posterior layouts are not q-only events.
        return posterior.rsample_from_base_samples(
            sample_shape=self.sample_shape,
            base_samples=expand_base_samples(
                self.base_samples, self.sample_shape, posterior.base_sample_shape, points
            ),
        )

    @property
    def num_samples(self) -> int:
        """The number of MC samples (``prod(sample_shape)``)."""
        return math.prod(self.sample_shape)

    @property
    def seed(self) -> int:
        """The seed of the base samples."""
        return int(self._rs.seed)

    def _get_collapsed_shape(self, posterior: Any) -> tuple[int, ...]:
        shape = tuple(posterior.base_sample_shape)
        start, end = _batch_range(posterior, getattr(self, "batch_range_override", None))
        start, end, _ = slice(start, end).indices(len(shape))
        return (*self.sample_shape, *shape[:start], *(1 for _ in shape[start:end]), *shape[end:])

    def _get_extended_base_sample_shape(self, posterior: Any) -> tuple[int, ...]:
        return (*self.sample_shape, *posterior.base_sample_shape)

    def _store_base_samples(self, values: list[float], shape: tuple[int, ...], kind: Any) -> None:
        array = from_flat(values, shape, kind)
        if isinstance(self, SobolQMCNormalSampler):
            self._base_samples = array
            self._base_samples_explicit = False
        else:
            self.base_samples = array
        self._base_shape = shape
        self._rs.set_base_samples((values, (self.num_samples, len(values) // self.num_samples)))

    def _construct_base_samples(self, posterior: Any) -> None:
        if _uses_tensor_protocol(posterior):
            self._construct_tensor_base_samples(posterior)
            return
        kind = _posterior_kind(posterior)
        if kind is None:
            raise UnsupportedError("The sampler requires a botorch_rs posterior.")
        target = self._get_collapsed_shape(posterior)
        current = getattr(self, "base_samples", None)
        if (
            getattr(self, "_base_samples_explicit", False)
            and current is not None
            and len(to_flat(current)[0]) == math.prod(target)
        ):
            # Preserve the accepted public S x [1 x ...] q [x m] layout of
            # explicitly assigned Sobol normals, including singleton axes.
            return
        if current is None or to_flat(current)[1] != target:
            values, _ = self._rs.base_samples(math.prod(target[len(self.sample_shape) :]))
            self._store_base_samples(list(values), target, kind)
        elif kind_of(current) != kind:
            self._store_base_samples(list(to_flat(current)[0]), target, kind)

    def _update_base_samples(self, posterior: Any, base_sampler: Any) -> None:
        """Reuse baseline normals when extending an ordinary Gaussian posterior."""
        if not isinstance(base_sampler, self.__class__):
            raise RuntimeError(
                "Expected `base_sampler` to be an instance of "
                f"{self.__class__.__name__}. Got {base_sampler}."
            )
        underlying = posterior
        while hasattr(underlying, "_posterior"):
            underlying = underlying._posterior
        if not isinstance(underlying, GPyTorchPosterior) or getattr(
            underlying, "_uses_matheron", False
        ):
            raise UnsupportedError(
                "Base-sample updates require an ordinary Gaussian posterior; "
                "Matheron and structured posterior base layouts are unsupported."
            )
        if tuple(base_sampler.sample_shape) != tuple(self.sample_shape):
            raise ValueError("Base samplers must have the same sample_shape.")
        if getattr(posterior, "_tensor_backend", None) == "torch":
            self._update_tensor_base_samples(posterior, base_sampler, underlying)
            return
        self._construct_base_samples(posterior)
        if base_sampler.base_samples is None:
            return
        source, source_shape = to_flat(base_sampler.base_samples)
        target, target_shape = to_flat(self.base_samples)
        sample_dims = len(self.sample_shape)
        if len(source_shape) > len(target_shape):
            raise ValueError("Baseline base samples have more dimensions than the joint samples.")
        single_output = len(posterior.base_sample_shape) - len(underlying.batch_shape) == 1
        point_axis = len(target_shape) - (1 if single_output else 2)
        source_points = source_shape[-1 if single_output else -2]
        if source_points > target_shape[point_axis]:
            raise ValueError("Baseline base samples contain more points than the joint posterior.")
        view = (
            *self.sample_shape,
            *(1 for _ in range(len(target_shape) - len(source_shape))),
            *source_shape[sample_dims:],
        )
        expanded_shape = list(target_shape)
        expanded_shape[point_axis] = source_points
        expanded = broadcast_to(source, view, expanded_shape)
        data = list(target)
        trailing = math.prod(target_shape[point_axis + 1 :])
        width = source_points * trailing
        target_width = target_shape[point_axis] * trailing
        for row in range(math.prod(target_shape[:point_axis])):
            data[row * target_width : row * target_width + width] = expanded[
                row * width : (row + 1) * width
            ]
        self._store_base_samples(data, tuple(target_shape), _posterior_kind(posterior))

    def _update_tensor_base_samples(
        self, posterior: Any, base_sampler: Any, underlying: Any
    ) -> None:
        import torch

        self._construct_tensor_base_samples(posterior)
        if base_sampler.base_samples is None:
            return
        target = self.base_samples
        source = torch.as_tensor(
            base_sampler.base_samples, dtype=target.dtype, device=target.device
        )
        if source.ndim > target.ndim:
            raise ValueError("Baseline base samples have more dimensions than the joint samples.")
        point_axis = -2 if underlying._is_mt else -1
        source_points = source.shape[point_axis]
        if source_points > target.shape[point_axis]:
            raise ValueError("Baseline base samples contain more points than the joint posterior.")
        sample_dims = len(self.sample_shape)
        source = source.reshape(
            *self.sample_shape,
            *((1,) * (target.ndim - source.ndim)),
            *source.shape[sample_dims:],
        )
        prefix_shape = list(target.shape)
        prefix_shape[point_axis] = source_points
        prefix = source.expand(prefix_shape)
        suffix = target.narrow(point_axis, source_points, target.shape[point_axis] - source_points)
        updated = torch.cat((prefix, suffix), dim=point_axis)
        if isinstance(self, SobolQMCNormalSampler):
            self._base_samples = updated
        else:
            self.base_samples = updated
        self._base_shape = tuple(updated.shape)


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
        """Cached standard-normal base samples, or ``None`` before drawing."""
        return self._base_samples

    @base_samples.setter
    def base_samples(self, value: Any) -> None:
        if value is None:
            self._base_samples = None
            self._base_samples_explicit = False
            self._tensor_sampler_key = None
            # Explicit CPU normals are also cached by the native sampler.
            # Recreate that cache with the same seed when resetting it.
            self._rs = _rs.SobolQMCNormalSampler(self.num_samples, self.seed)
            return
        if getattr(getattr(value, "device", None), "type", "cpu") != "cpu" or getattr(
            value, "requires_grad", False
        ):
            if value.shape[: len(self.sample_shape)] != self.sample_shape:
                raise BotorchTensorDimensionError("Base samples must begin with sample_shape.")
            self._base_samples = value
            self._base_samples_explicit = True
            return
        data, shape = to_flat(value, "base_samples")
        n = self._rs.num_samples
        sample_dims = len(self.sample_shape)
        if tuple(shape[:sample_dims]) == self.sample_shape:
            event_shape = shape[sample_dims:]
        elif shape and shape[0] == n:
            # Preserve the native sampler's documented flattened sample axis.
            event_shape = shape[1:]
        else:
            raise BotorchTensorDimensionError(
                f"Expected base samples beginning with sample_shape {self.sample_shape}, got {shape}."
            )
        non_one = [s for s in event_shape if s != 1]
        if not event_shape or len(non_one) > 2:
            raise BotorchTensorDimensionError(
                f"Base samples of shape {shape} must be `num_samples x q` (or "
                "`num_samples x 1 x q`, `num_samples x q x m`); batched base samples are "
                "not supported."
            )
        self._rs.set_base_samples((data, (n, math.prod(non_one))))
        self._base_samples = value
        self._base_samples_explicit = True

    def forward(self, posterior: GPyTorchPosterior) -> Any:
        """Posterior samples, shape ``sample_shape x batch_shape x q x m``; the
        same base samples are used for every batch."""
        if _uses_tensor_protocol(posterior):
            return self._tensor_samples(posterior)
        self._construct_base_samples(posterior)
        if not isinstance(posterior, GPyTorchPosterior) or getattr(
            posterior, "_uses_matheron", False
        ):
            kind = _posterior_kind(posterior)
            if not isinstance(posterior, Posterior) or kind is None:
                raise UnsupportedError("The sampler requires a botorch_rs posterior.")
            shape = tuple(posterior.base_sample_shape)
            start, end = posterior.batch_range
            start, end, _ = slice(start, end).indices(len(shape))
            collapsed = (*shape[:start], *(1 for _ in shape[start:end]), *shape[end:])
            data, _ = self._rs.base_samples(math.prod(collapsed))
            base = from_flat(data, (*self.sample_shape, *collapsed), kind)
            return posterior.rsample_from_base_samples(self.sample_shape, base)
        q, m = posterior._q, posterior._m
        z: Pair = self._rs.base_samples(q * m)
        return posterior._samples_out(z, self.sample_shape)

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
        if _uses_tensor_protocol(posterior):
            return self._tensor_samples(posterior)
        self._construct_base_samples(posterior)
        if not isinstance(posterior, GPyTorchPosterior) or getattr(
            posterior, "_uses_matheron", False
        ):
            kind = _posterior_kind(posterior)
            if not isinstance(posterior, Posterior) or kind is None:
                raise UnsupportedError("The sampler requires a botorch_rs posterior.")
            collapsed = self._collapsed_shape(posterior)
            target = (*self.sample_shape, *collapsed)
            if self.base_samples is None or self._base_shape != target:
                self.base_samples = from_flat(self._draw(target, kind), target, kind)
                self._base_shape = target
            return posterior.rsample_from_base_samples(self.sample_shape, self.base_samples)
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
        return posterior._samples_out(pair, self.sample_shape)

    __call__ = forward

    def __repr__(self) -> str:
        return f"IIDNormalSampler(sample_shape={self.sample_shape}, seed={self.seed})"


__all__ = ["IIDNormalSampler", "MCSampler", "NormalMCSampler", "SobolQMCNormalSampler"]

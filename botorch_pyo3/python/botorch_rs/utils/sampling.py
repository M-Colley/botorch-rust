"""Quasi-random sampling (``botorch.utils.sampling``).

The Sobol points come from an Owen-scrambled Sobol sequence (Burley's
hash-based scrambling, Joe-Kuo direction numbers): for a given seed they
differ from torch's ``SobolEngine``, with the same distribution.
"""

from __future__ import annotations

import math
from typing import Any

from botorch_rs import _botorch_rs as _rs
from botorch_rs._arrays import first_kind, from_flat, from_pair, kind_from_dtype, to_flat


def draw_sobol_samples(
    bounds: Any,
    n: int,
    q: int,
    batch_shape: Any = None,
    seed: int | None = None,
) -> Any:
    r"""``n`` scrambled Sobol ``q``-batches in the box ``bounds`` (``2 x d``):
    an array of shape ``n x batch_shape x q x d`` of the type of ``bounds``."""
    batch = tuple(int(b) for b in batch_shape) if batch_shape is not None else ()
    nb = math.prod(batch)
    data, shape = _rs.draw_sobol_samples(
        to_flat(bounds, "bounds"), n * nb, int(q), None if seed is None else int(seed)
    )
    if batch:
        # Like BoTorch, consecutive draws fill `batch_shape x n` and are then
        # permuted to `n x batch_shape`.
        row = math.prod(shape[1:])
        out = [0.0] * len(data)
        for b in range(nb):
            for i in range(n):
                src, dst = (b * n + i) * row, (i * nb + b) * row
                out[dst : dst + row] = data[src : src + row]
        data = out
    return from_flat(data, (n, *batch, *shape[1:]), first_kind(bounds))


def draw_sobol_normal_samples(
    d: int,
    n: int,
    device: Any = None,
    dtype: Any = None,
    seed: int | None = None,
) -> Any:
    r"""``n x d`` quasi-random standard-normal samples (inverse transform of a
    scrambled Sobol sequence)."""
    return from_pair(
        _rs.draw_sobol_normal_samples(int(d), int(n), None if seed is None else int(seed)),
        kind_from_dtype(dtype, device),
    )


__all__ = ["draw_sobol_normal_samples", "draw_sobol_samples"]

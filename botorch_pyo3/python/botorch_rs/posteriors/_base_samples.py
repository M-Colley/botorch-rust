"""Shape-only expansion of Tensor normals, preserving their autograd graph."""

from __future__ import annotations

from typing import Any


def expand_base_samples(
    base_samples: Any,
    sample_shape: Any,
    base_sample_shape: Any,
    single_output_points: int | None = None,
) -> Any:
    """Insert omitted batch axes after MC axes and honor the advertised layout.

    A legacy single-output ``q x 1`` suffix is accepted only when the array
    cannot already represent the public base shape. In particular, a valid
    ``batch x 1`` suffix for q=1 must retain its batch axis.
    """
    sample_shape, base_shape = tuple(sample_shape), tuple(base_sample_shape)
    if tuple(base_samples.shape[: len(sample_shape)]) != sample_shape:
        raise RuntimeError("sample_shape disagrees with base_samples.shape.")
    tail = tuple(base_samples.shape[len(sample_shape) :])
    valid = len(tail) <= len(base_shape) and all(
        actual in (1, expected)
        for actual, expected in zip(reversed(tail), reversed(base_shape), strict=False)
    )
    if (
        not valid
        and single_output_points is not None
        and len(tail) >= 2
        and tail[-2:] == (single_output_points, 1)
    ):
        base_samples = base_samples.squeeze(-1)
        tail = tail[:-1]
    missing = len(base_shape) - len(tail)
    if missing > 0:
        base_samples = base_samples.reshape(*sample_shape, *((1,) * missing), *tail)
    return base_samples.expand((*sample_shape, *base_shape))

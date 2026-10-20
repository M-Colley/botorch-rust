"""Prior sample paths (``botorch.sampling.pathwise.prior_samplers``)."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

from botorch_rs import _botorch_rs as _rs
from botorch_rs._arrays import to_flat
from botorch_rs.exceptions.errors import UnsupportedError
from botorch_rs.sampling.pathwise.paths import GeneralizedLinearPath, PathList, SamplePath
from botorch_rs.sampling.pathwise.utils import model_outputs, sample_shape_arg

#: BoTorch's default number of random Fourier features.
DEFAULT_NUM_FEATURES = 1024


def seed_arg(seed: Any) -> int | None:
    """A seed as an unsigned 64-bit integer (``None`` draws a random seed)."""
    if seed is None:
        return None
    seed = int(seed)
    if seed < 0:
        raise ValueError(f"The seed must be non-negative, got {seed}.")
    return seed


def draws_arg(draws: Sequence[Sequence[Any]] | None, names: Sequence[str]) -> list[Any] | None:
    """Explicit draws (one tuple of arrays per output) as flat arrays."""
    if draws is None:
        return None
    out = []
    for d in draws:
        if len(d) != len(names):
            raise ValueError(f"Each output's draws must be a tuple ({', '.join(names)}).")
        out.append(tuple(to_flat(a, name) for a, name in zip(d, names, strict=True)))
    return out


def as_paths(
    model: Any,
    handle: _rs.SamplePaths,
    sizes: list[int],
    sample_shape: tuple[int, ...],
    make: Any,
) -> Any:
    """One path object per model (``make(handle, sample_shape, outputs,
    kind)``): the path of a ``SingleTaskGP``, or a ``PathList`` of the paths
    of the models of a ``ModelListGP``."""
    from botorch_rs.models.model_list_gp_regression import ModelListGP

    kind = model._kind
    if not isinstance(model, ModelListGP):
        path = make._from_native(handle, sample_shape, range(sizes[0]), kind)
        if hasattr(path, "_set_model_transform"):
            path._set_model_transform(model)
        return path
    paths: list[SamplePath] = []
    offset = 0
    for submodel, m in zip(model.models, sizes, strict=True):
        path = make._from_native(handle, sample_shape, range(offset, offset + m), kind)
        if hasattr(path, "_set_model_transform"):
            path._set_model_transform(submodel)
        paths.append(path)
        offset += m
    return PathList(paths)


def draw_kernel_feature_paths(
    model: Any,
    sample_shape: Any,
    *,
    num_features: int = DEFAULT_NUM_FEATURES,
    seed: int | None = None,
    draws: Sequence[Sequence[Any]] | None = None,
    **kwargs: Any,
) -> GeneralizedLinearPath | PathList:
    r"""Draws functions from a Bayesian-linear-model-based approximation to a
    GP prior: ``f(x) = c + sum_l w_l phi_l(x)`` with ``num_features`` random
    Fourier features ``phi`` of the model's kernel (an RBF or Matérn kernel
    with any ``nu``, ARD or a shared lengthscale and ``active_dims``, with or
    without a ``ScaleKernel``; BoTorch raises ``NotImplementedError`` for other
    kernels, and so does botorch_rs), its constant mean ``c`` and standard normal
    weights ``w``. The paths apply the model's input transform and return
    values in the model's original outcome space.

    Evaluated at ``X`` (``batch_shape x n x d``), the paths return
    ``sample_shape x batch_shape x [m x] n`` values (for a multi-output
    model, the outputs form the last batch dimension, which ``batch_shape``
    broadcasts with); the paths of a ``ModelListGP`` form a ``PathList``.

    botorch_rs extensions (keyword-only): ``seed`` (the random numbers of
    botorch_rs, which differ from torch's for the same seed) or ``draws``,
    one ``(frequencies, weights)`` tuple per output (``num_features / 2 x d``
    and ``prod(sample_shape) x num_features``), e.g. BoTorch's
    ``feature_map.weight`` and ``weight``.
    """
    if kwargs:
        raise UnsupportedError(
            f"Unsupported arguments of draw_kernel_feature_paths: {sorted(kwargs)}."
        )
    shape = sample_shape_arg(sample_shape)
    rs_model, sizes = model_outputs(model)
    handle = _rs.draw_kernel_feature_paths(
        rs_model,
        math.prod(shape),
        int(num_features),
        seed_arg(seed),
        draws_arg(draws, ("frequencies", "weights")),
    )
    return as_paths(model, handle, sizes, shape, GeneralizedLinearPath)


__all__ = ["draw_kernel_feature_paths"]

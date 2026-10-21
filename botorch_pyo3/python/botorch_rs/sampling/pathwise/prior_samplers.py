"""Prior sample paths (``botorch.sampling.pathwise.prior_samplers``)."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

from botorch_rs import _botorch_rs as _rs
from botorch_rs._arrays import broadcast_shapes, from_flat, is_torch_tensor, kind_of, to_flat
from botorch_rs._ops import expand, getitem, shape_of
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
    shape = sample_shape_arg(sample_shape)
    if kwargs or (
        getattr(model, "_tensor_backend", None) != "torch"
        and (
            bool(model.batch_shape)
            or hasattr(getattr(model, "model", None), "variational_strategy")
        )
    ):
        return _generic_prior_paths(model, shape, int(num_features), seed, draws, **kwargs)
    if getattr(model, "_tensor_backend", None) == "torch":
        from botorch_rs.sampling.pathwise._torch_backend import tensor_prior_paths

        return tensor_prior_paths(model, shape, int(num_features), seed, draws)
    rs_model, sizes = model_outputs(model)
    handle = _rs.draw_kernel_feature_paths(
        rs_model,
        math.prod(shape),
        int(num_features),
        seed_arg(seed),
        draws_arg(draws, ("frequencies", "weights")),
    )
    return as_paths(model, handle, sizes, shape, GeneralizedLinearPath)


def _generic_prior_paths(
    model: Any,
    shape: tuple[int, ...],
    num_features: int,
    seed: Any,
    draws: Any,
    **options: Any,
) -> Any:
    """The public feature-map/weight callback protocol, including model batches."""
    from botorch_rs.sampling.pathwise.features.generators import gen_kernel_features
    from botorch_rs.sampling.pathwise.utils import (
        get_input_transform,
        get_output_transform,
        get_train_inputs,
    )

    allowed = {"map_generator", "weight_generator", "input_transform", "output_transform", "join"}
    if options.keys() - allowed:
        raise UnsupportedError(
            f"Unsupported prior path options: {sorted(options.keys() - allowed)}."
        )
    if hasattr(model, "models"):
        children, start = [], 0
        for i, child in enumerate(model.models):
            stop = start + child.num_outputs
            children.append(
                draw_kernel_feature_paths(
                    child,
                    shape,
                    num_features=num_features,
                    seed=None if seed is None else seed + i,
                    draws=None if draws is None else draws[start:stop],
                    **{key: value for key, value in options.items() if key != "join"},
                )
            )
            start = stop
        return PathList(
            children,
            join=options.get("join"),
        )
    if "join" in options:
        raise TypeError("join is only defined for model-list prior paths.")
    if draws is not None:
        raise ValueError("Custom prior generators cannot be combined with explicit native draws.")
    model.eval()
    (train_x,) = get_train_inputs(model, transformed=False)
    modules = (
        model.model if hasattr(model, "model") and hasattr(model.model, "covar_module") else model
    )
    kernel, mean = modules.covar_module, modules.mean_module
    if not callable(mean):
        from botorch_rs.means import ConstantMean, ZeroMean

        if isinstance(mean, ConstantMean):
            mean = _NativeConstantMean(mean)
        elif isinstance(mean, ZeroMean):
            mean = None
        else:
            raise UnsupportedError("A custom path mean must be callable.")
    generator = options.get("map_generator", gen_kernel_features)
    feature_map = generator(
        kernel=kernel,
        num_inputs=to_flat(train_x)[1][-1] if not is_torch_tensor(train_x) else train_x.shape[-1],
        num_outputs=num_features,
        **({"seed": seed} if generator is gen_kernel_features and seed is not None else {}),
    )
    weights_shape = (*shape, *kernel.batch_shape, feature_map.num_outputs)
    weight_generator = options.get("weight_generator")
    kind = kind_of(train_x)
    if kind.lib == "torch":
        import torch

        if weight_generator is None:
            from botorch_rs.sampling.pathwise._torch_backend import _normals, _seed

            weights = _normals(weights_shape, train_x, _seed(seed))
        else:
            weights = weight_generator(torch.Size(weights_shape)).to(train_x)
    elif weight_generator is not None:
        weights = weight_generator(weights_shape)
    else:
        values, _ = _rs.draw_sobol_normal_samples(
            feature_map.num_outputs, math.prod(weights_shape[:-1]), seed_arg(seed)
        )
        weights = from_flat(values, weights_shape, kind)
    return GeneralizedLinearPath(
        feature_map=feature_map,
        weight=weights,
        bias_module=mean,
        input_transform=options.get("input_transform", get_input_transform(model)),
        output_transform=options.get("output_transform", get_output_transform(model)),
        is_ensemble=bool(getattr(model, "_is_ensemble", False)),
    )


class _NativeConstantMean:
    """Evaluate the native mean view using the array broadcast primitives."""

    def __init__(self, mean: Any) -> None:
        self.mean = mean

    def __call__(self, values: Any) -> Any:
        constant = self.mean.constant
        batch = broadcast_shapes(shape_of(values)[:-2], shape_of(constant))
        if batch is None:
            raise ValueError("Mean and input batches do not broadcast.")
        return expand(getitem(constant, (..., None)), (*batch, shape_of(values)[-2]))


__all__ = ["draw_kernel_feature_paths"]

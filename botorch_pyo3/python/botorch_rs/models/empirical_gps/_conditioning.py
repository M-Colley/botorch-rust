"""Native empirical conditioning and immutable batches of fantasy models."""

from __future__ import annotations

import copy
import math
from typing import Any

from botorch_rs import _botorch_rs as _rs
from botorch_rs import _ops, settings
from botorch_rs._arrays import broadcast_shapes, from_flat, kind_of, to_flat
from botorch_rs.exceptions.errors import (
    BotorchTensorDimensionError,
    DeprecationError,
    UnsupportedError,
)
from botorch_rs.models.utils.assorted import fantasize as fantasy_context


class _FantasyState:
    """Shared hyperparameters over native conditioned model snapshots.

    Predictions expose a real native model batch. Parameter replacements are
    validated on copies and committed to every leaf together. Refitting this
    shared-parameter fantasy batch is a separate, unsupported operation.
    """

    def __init__(self, models: list[Any], batch_shape: tuple[int, ...], fixed: bool) -> None:
        self.models, self.batch_shape, self.fixed = models, batch_shape, fixed

    def __getattr__(self, name: str) -> Any:
        if name.startswith("__"):
            raise AttributeError(name)
        if name in ("fit", "fit_torch", "mll", "mll_value_and_grad"):
            raise UnsupportedError("Refitting an empirical fantasy batch is not yet supported.")
        return getattr(self.models[0], name)

    @property
    def aug_batch_shape(self) -> tuple[int, ...]:
        shape = tuple(self.models[0].aug_batch_shape)
        return (*self.batch_shape, *shape) if self.fixed else shape

    @property
    def noise(self) -> list[float]:
        return (
            [v for model in self.models for v in model.noise]
            if self.fixed
            else list(self.models[0].noise)
        )

    def copy(self) -> _FantasyState:
        return _FantasyState([m.copy() for m in self.models], self.batch_shape, self.fixed)

    def model_handle(self) -> Any:
        return _rs.Model.from_batch_models(
            [m.model_handle() for m in self.models], self.batch_shape
        )

    def set_raw_parameters(self, values: Any) -> None:
        models = [model.copy() for model in self.models]
        for model in models:
            model.set_raw_parameters(values)
        self.models = models

    def set_noise(self, value: Any) -> None:
        models = [model.copy() for model in self.models]
        for model in models:
            model.set_noise(value)
        self.models = models

    def sample_all_priors(self, seed: Any = None) -> None:
        first = self.models[0].copy()
        first.sample_all_priors(seed)
        self.set_raw_parameters(first.raw_parameters())


def _source_index(index: int, output_shape: tuple[int, ...], source_shape: tuple[int, ...]) -> int:
    coordinates = []
    for size in reversed(output_shape):
        coordinates.append(index % size)
        index //= size
    coordinates.reverse()
    offset = len(output_shape) - len(source_shape)
    result = 0
    for i, size in enumerate(source_shape):
        result = result * size + (coordinates[offset + i] if size > 1 else 0)
    return result


def condition(model: Any, X: Any, Y: Any, noise: Any = None, **kwargs: Any) -> Any:
    if kwargs:
        raise UnsupportedError(f"Unsupported conditioning arguments: {sorted(kwargs)}.")
    if _ops._requires_grad((X, Y, noise)):
        raise UnsupportedError(
            "Empirical conditioning does not yet propagate gradients through new training data."
        )
    xs, ys = _ops.shape_of(X), _ops.shape_of(Y)
    if len(xs) < 2 or xs[-1] != 1 or len(ys) < 2 or ys[-2:] != (xs[-2], model.num_outputs):
        raise BotorchTensorDimensionError(
            "Empirical observations require X (..., n, 1), Y (..., n, m)."
        )
    batch = broadcast_shapes(tuple(model.batch_shape), xs[:-2], ys[:-2])
    if batch is None or math.prod(batch) == 0:
        raise BotorchTensorDimensionError("Empirical conditioning batch shapes must broadcast.")
    q, m = xs[-2], model.num_outputs
    xx = _ops.expand(X, (*batch, q, 1))
    yy = _ops.expand(Y, (*batch, q, m))
    nz = None if noise is None else _ops.expand(noise, (*batch, q, m))
    xflat, yflat = to_flat(xx)[0], to_flat(yy)[0]
    nflat = None if nz is None else to_flat(nz)[0]
    state = model._rs
    source_models = state.models if isinstance(state, _FantasyState) else [state]
    leaves = []
    for index in range(math.prod(batch)):
        source = source_models[_source_index(index, batch, tuple(model.batch_shape))]
        leaves.append(
            source.condition_on_observations(
                (list(xflat[index * q : (index + 1) * q]), (q, 1)),
                (list(yflat[index * q * m : (index + 1) * q * m]), (q, m)),
                None
                if nflat is None
                else (list(nflat[index * q * m : (index + 1) * q * m]), (q, m)),
            )
        )
    new = copy.deepcopy(model)
    new._rs = _FantasyState(leaves, batch, model._train_Yvar is not None) if batch else leaves[0]
    new._condition_batch_shape = batch
    old_n = _ops.shape_of(model._train_Y)[-2]
    old_x = _ops.expand(model._train_X, (*batch, old_n, 1))
    old_y = _ops.expand(model._train_Y, (*batch, old_n, m))
    new._train_X = from_flat(
        to_flat(_ops.concat([old_x, xx], dim=-2))[0], (*batch, old_n + q, 1), model._kind
    )
    new._train_Y = from_flat(
        to_flat(_ops.concat([old_y, yy], dim=-2))[0], (*batch, old_n + q, m), model._kind
    )
    if model._train_Yvar is not None:
        # Native fixed-noise conditioning already checked that nz is supplied.
        old_noise = _ops.expand(model._train_Yvar, (*batch, old_n, m))
        new._train_Yvar = from_flat(
            to_flat(_ops.concat([old_noise, nz], dim=-2))[0], (*batch, old_n + q, m), model._kind
        )
    new.training = False
    return new


def fantasize(
    model: Any, X: Any, sampler: Any, observation_noise: Any = None, **kwargs: Any
) -> Any:
    """Sample observed native posteriors, then condition residual GPs exactly."""
    if isinstance(observation_noise, bool):
        raise DeprecationError("`fantasize` no longer accepts a boolean for `observation_noise`.")
    if observation_noise is not None:
        raise UnsupportedError(
            "Empirical fantasy sampling with explicit noise is not yet supported."
        )
    if kwargs.pop("propagate_grads", False) or _ops._requires_grad((X,)):
        raise UnsupportedError("Empirical fantasies do not yet propagate training-data gradients.")
    noise = None
    if model._train_Yvar is not None:
        fixed = model.likelihood.noise
        if model._correlated:
            average = _ops.reduce("mean", fixed, dim=-1, keepdim=True)
            noise = _ops.expand(
                _ops.getitem(average, (Ellipsis, None)), (*model.batch_shape, 1, model.num_outputs)
            )
        else:
            noise = _ops.reduce("mean", _ops.transpose(fixed), dim=-2, keepdim=True)
    xs = _ops.shape_of(X)
    if len(xs) < 2:
        raise BotorchTensorDimensionError("Empirical fantasy inputs require (..., q, 1).")
    batch = broadcast_shapes(xs[:-2], model.batch_shape)
    if batch is None:
        raise BotorchTensorDimensionError(
            "Empirical fantasy input and model batches must broadcast."
        )
    if xs[-2] == 0:
        target = _ops.full((*sampler.sample_shape, *batch, 0, model.num_outputs), 0.0, kind_of(X))
        return condition(model, X, target, noise=noise, **kwargs)
    with fantasy_context(), settings.propagate_grads(False):
        posterior = model.posterior(X, observation_noise=True)
        targets = sampler(posterior)
        if noise is not None:
            noise = _ops.expand(noise, _ops.shape_of(targets)[len(sampler.sample_shape) :])
        return condition(model, X, targets, noise=noise, **kwargs)

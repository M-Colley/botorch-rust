"""Native LCEM observation updates and batches of immutable fantasy models."""

from __future__ import annotations

import copy
import math
from typing import Any

from botorch_rs import _botorch_rs as _rs
from botorch_rs import _ops
from botorch_rs._arrays import broadcast_shapes, from_flat, to_flat
from botorch_rs.exceptions.errors import (
    BotorchTensorDimensionError,
    DeprecationError,
    UnsupportedError,
)
from botorch_rs.models.utils.assorted import fantasize as fantasy_context


def _source_index(index: int, batch: tuple[int, ...], original: tuple[int, ...]) -> int:
    coordinates = []
    for size in reversed(batch):
        coordinates.append(index % size)
        index //= size
    coordinates.reverse()
    offset, result = len(batch) - len(original), 0
    for i, size in enumerate(original):
        result = result * size + (coordinates[offset + i] if size != 1 else 0)
    return result


class _LcemFantasyState:
    """Native members sharing hyperparameters, transforms and context tables."""

    _mutations = {
        "set_raw_parameters",
        "set_raw_parameter",
        "set_noise",
        "set_lengthscale_of",
        "set_outputscale_of",
        "set_task_covar_module",
        "renorm_embeddings",
        "eval_unobserved_means",
    }

    def __init__(self, models: list[Any], batch_shape: tuple[int, ...]) -> None:
        self.models, self.batch_shape = models, batch_shape

    def __getattr__(self, name: str) -> Any:
        if name.startswith("__"):
            raise AttributeError(name)
        if name == "fit":
            raise UnsupportedError(
                "Refitting shared-parameter native LCEM fantasy batches is not implemented."
            )
        if name in self._mutations:

            def update(*args: Any, **kwargs: Any) -> None:
                models = [model.copy() for model in self.models]
                for model in models:
                    getattr(model, name)(*args, **kwargs)
                self.models = models

            return update
        return getattr(self.models[0], name)

    def copy(self) -> _LcemFantasyState:
        return _LcemFantasyState([model.copy() for model in self.models], self.batch_shape)

    def model_handle(self, tasks: Any = None, full: bool = False, transform: Any = None) -> Any:
        handles = [
            _rs.Model.from_lcemgp_task_inputs(model)
            if full
            else _rs.Model.from_lcemgp(model, tasks, transform)
            for model in self.models
        ]
        return _rs.Model.from_batch_models(handles, self.batch_shape)

    def _stack_data(self, name: str) -> Any:
        values = [getattr(model, name)() for model in self.models]
        return [v for data, _ in values for v in data], (*self.batch_shape, *values[0][1])

    def train_inputs(self) -> Any:
        return self._stack_data("train_inputs")

    def train_targets(self) -> Any:
        return self._stack_data("train_targets")

    def transformed_train_targets(self) -> Any:
        return self._stack_data("transformed_train_targets")

    def mll(self, raw_parameters: Any = None) -> Any:
        return [model.mll(raw_parameters)[0][0] for model in self.models], self.batch_shape

    def mll_value_and_grad(self, raw_parameters: Any = None) -> Any:
        results = [model.mll_value_and_grad(raw_parameters) for model in self.models]
        gradients = [
            sum(entries) for entries in zip(*(result[1] for result in results), strict=True)
        ]
        return ([result[0][0][0] for result in results], self.batch_shape), gradients


def condition(model: Any, X: Any, Y: Any, noise: Any = None, **kwargs: Any) -> Any:
    if kwargs:
        raise UnsupportedError(f"Unsupported conditioning arguments: {sorted(kwargs)}.")
    if model.training:
        raise RuntimeError(
            "Fantasy observations can only be added after making predictions with a model."
        )
    if _ops._requires_grad((X, Y, noise)):
        raise UnsupportedError(
            "Native LCEM conditioning does not propagate training-data gradients; use backend='torch'."
        )
    xs, ys = _ops.shape_of(X), _ops.shape_of(Y)
    model._check_full_inputs(([], xs), "condition_on_observations")
    if len(ys) == len(xs) - 1:
        Y = _ops.getitem(Y, (Ellipsis, None))
        ys = (*ys, 1)
    if len(ys) < 2 or ys[-2:] != (xs[-2], 1):
        raise BotorchTensorDimensionError("LCEM observations require Y (..., n, 1).")
    batch = broadcast_shapes(tuple(model.batch_shape), xs[:-2], ys[:-2])
    if batch is None or math.prod(batch) == 0:
        raise BotorchTensorDimensionError(
            "LCEM conditioning batches must be nonempty and broadcast."
        )
    q, d = xs[-2:]
    xflat = to_flat(_ops.expand(X, (*batch, q, d)))[0]
    yflat = to_flat(_ops.expand(Y, (*batch, q, 1)))[0]
    if noise is not None:
        ns = _ops.shape_of(noise)
        if ns == ys[:-1]:
            noise = _ops.getitem(noise, (Ellipsis, None))
    nflat = None if noise is None else to_flat(_ops.expand(noise, (*batch, q, 1)))[0]
    original = model._rs.models if isinstance(model._rs, _LcemFantasyState) else [model._rs]
    # Group common input/noise blocks so all fantasy targets reuse one Rust factor.
    groups: dict[Any, list[tuple[int, Any]]] = {}
    for index in range(math.prod(batch)):
        source = _source_index(index, batch, tuple(model.batch_shape))
        block = tuple(xflat[index * q * d : (index + 1) * q * d])
        variance = None if nflat is None else tuple(nflat[index * q : (index + 1) * q])
        groups.setdefault((source, block, variance), []).append(
            (index, (list(yflat[index * q : (index + 1) * q]), (q, 1)))
        )
    members: list[Any] = [None] * math.prod(batch)
    for (source, block, variance), targets in groups.items():
        updates = original[source].condition_on_observations_many(
            (list(block), (q, d)),
            [value for _, value in targets],
            None if variance is None else (list(variance), (q, 1)),
        )
        for (index, _), updated in zip(targets, updates, strict=True):
            members[index] = updated
    new = copy.deepcopy(model)
    new._rs = _LcemFantasyState(members, batch) if batch else members[0]
    new.train(False)
    return new


def fantasize(
    model: Any, X: Any, sampler: Any, observation_noise: Any = None, **kwargs: Any
) -> Any:
    if isinstance(observation_noise, bool):
        raise DeprecationError("`fantasize` no longer accepts a boolean for `observation_noise`.")
    if observation_noise is not None:
        raise NotImplementedError(
            "Passing a tensor of observations is not supported by MultiTaskGP."
        )
    if kwargs.pop("propagate_grads", False) or _ops._requires_grad((X,)):
        raise UnsupportedError(
            "Native LCEM fantasies do not propagate training-data gradients; use backend='torch'."
        )
    if not callable(sampler) or not hasattr(sampler, "sample_shape"):
        raise TypeError("sampler must be callable and expose sample_shape.")
    xs = _ops.shape_of(X)
    model._check_full_inputs(([], xs), "fantasize")
    model.eval()
    if xs[-2] == 0:
        batch = broadcast_shapes(model.batch_shape, xs[:-2])
        if batch is None:
            raise BotorchTensorDimensionError("LCEM fantasy input/model batches must broadcast.")
        targets = from_flat([], (*sampler.sample_shape, *batch, 0, 1), model._kind)
    else:
        with fantasy_context():
            targets = sampler(model.posterior(X, observation_noise=True))
    return condition(model, X, targets, **kwargs)

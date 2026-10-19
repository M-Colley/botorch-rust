"""Array shapes, training-data checks and duplicate handling for models.

Arrays may be nested lists, NumPy arrays or torch tensors. Shape operations
preserve torch autograd; duplicate searches run in the Rust implementation
shared with ``PairwiseGP``. Optional array libraries are never imported for
list inputs.
"""

from __future__ import annotations

import math
import sys
import warnings
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from typing import Any

from botorch_rs import _botorch_rs as _rs
from botorch_rs import _ops, settings
from botorch_rs._arrays import ArrayKind, broadcast_shapes, from_flat, kind_of, to_flat
from botorch_rs.exceptions.errors import InputDataError
from botorch_rs.exceptions.warnings import InputDataWarning


class _Flag:
    """A global boolean flag that can be set in a ``with`` block (GPyTorch's
    ``settings._Flag``)."""

    _state: bool = False

    @classmethod
    def on(cls) -> bool:
        return cls._state

    @classmethod
    def off(cls) -> bool:
        return not cls._state

    @classmethod
    def _set_state(cls, state: bool) -> None:
        cls._state = state

    def __init__(self, state: bool = True) -> None:
        self.prev = self.__class__.on()
        self.state = state

    def __enter__(self) -> None:
        self.__class__._set_state(self.state)

    def __exit__(self, *args: Any) -> None:
        self.__class__._set_state(self.prev)


class fantasize(_Flag):
    """A flag denoting whether we are currently in a ``fantasize`` context:
    input transforms with ``transform_on_fantasize=False`` are skipped in eval
    mode while it is on."""

    _state: bool = False


def _make_X_full(X: Any, output_indices: list[int], tf: int) -> Any:
    """Insert a task feature and concatenate the observations for each task."""
    shape = _ops.shape_of(X)
    kind = kind_of(X)
    columns = (Ellipsis, slice(None, tf)), (Ellipsis, slice(tf, None))
    return _ops.concat(
        [
            _ops.concat(
                [
                    _ops.getitem(X, columns[0]),
                    _ops.full((*shape[:-1], 1), float(task), kind),
                    _ops.getitem(X, columns[1]),
                ],
                dim=-1,
            )
            for task in output_indices
        ],
        dim=-2,
    )


def multioutput_to_batch_mode_transform(
    train_X: Any, train_Y: Any, num_outputs: int, train_Yvar: Any = None
) -> tuple[Any, Any, Any]:
    """Represent independent output dimensions as batches of single outputs."""
    shape = _ops.shape_of(train_X)
    expanded = _ops.getitem(train_X, (Ellipsis, None, slice(None), slice(None)))
    X = _ops.expand(expanded, (*shape[:-2], num_outputs, *shape[-2:]))
    return X, _ops.transpose(train_Y), (None if train_Yvar is None else _ops.transpose(train_Yvar))


def add_output_dim(X: Any, original_batch_shape: Any) -> tuple[Any, int]:
    """Insert the output axis after the candidate batch dimensions."""
    shape = _ops.shape_of(X)
    batch = shape[:-2]
    original = tuple(original_batch_shape)
    if batch and original and broadcast_shapes(batch, original) is None:
        raise RuntimeError(
            "The trailing batch dimensions of X must match the trailing "
            f"batch dimensions of the training inputs. Got X.shape={shape} "
            f"and original_batch_shape={original}."
        )
    return (
        _ops.getitem(X, (Ellipsis, None, slice(None), slice(None))),
        max(len(original), len(batch)),
    )


def check_no_nans(Z: Any) -> None:
    """Raise ``InputDataError`` when an input contains NaN values."""
    if any(math.isnan(float(value)) for value in to_flat(Z, "Z")[0]):
        raise InputDataError("Input data contains NaN values.")


def _scaling_failure(message: str, raise_on_fail: bool) -> None:
    if raise_on_fail:
        raise InputDataError(message)
    warnings.warn(message, InputDataWarning, stacklevel=3)


def check_min_max_scaling(
    X: Any,
    strict: bool = False,
    atol: float = 1e-2,
    raise_on_fail: bool = False,
    ignore_dims: list[int] | None = None,
) -> None:
    """Check containment in the unit cube, optionally requiring both bounds."""
    values, shape = to_flat(X, "X")
    dimensions = sorted(set(range(shape[-1])) - set(ignore_dims or []))
    if not dimensions:
        return
    minima, maxima = [], []
    for start in range(0, len(values), shape[-1]):
        row = [float(values[start + j]) for j in dimensions]
        minima.append(math.nan if any(math.isnan(v) for v in row) else min(row))
        maxima.append(math.nan if any(math.isnan(v) for v in row) else max(row))
    msg = None
    if strict and not any(math.isnan(v) for v in minima + maxima):
        if max([abs(v) for v in minima] + [abs(v - 1) for v in maxima]) > atol:
            msg = "scaled"
    if any(v < -atol for v in minima) or any(v > 1 + atol for v in maxima):
        msg = "contained"
    if msg is not None:
        _scaling_failure(
            f"Data (input features) is not {msg} to the unit cube. "
            "Please consider min-max scaling the input data.",
            raise_on_fail,
        )


def _outcome_stats(Y: Any) -> tuple[Any, Any, list[float], list[float]]:
    kind = kind_of(Y)
    shape = _ops.shape_of(Y)
    n = shape[-2]
    if kind.lib == "torch":
        with sys.modules["torch"].no_grad():
            mean = Y.mean(dim=-2)
            std = Y.std(dim=-2) if n > 1 else None
    elif kind.lib == "numpy":
        np = sys.modules["numpy"]
        mean = np.mean(Y, axis=-2)
        std = np.std(Y, axis=-2, ddof=1) if n > 1 else None
    else:
        values, _ = to_flat(Y)
        m = shape[-1]
        means, stds = [], []
        for batch in range(math.prod(shape[:-2])):
            for output in range(m):
                column = [float(values[(batch * n + j) * m + output]) for j in range(n)]
                avg = sum(column) / n if n else math.nan
                means.append(avg)
                if n > 1:
                    stds.append(math.sqrt(sum((v - avg) ** 2 for v in column) / (n - 1)))
        mean = from_flat(means, (*shape[:-2], m), kind)
        std = from_flat(stds, (*shape[:-2], m), kind) if n > 1 else None
    return mean, std, list(to_flat(mean)[0]), [] if std is None else list(to_flat(std)[0])


def check_standardization(
    Y: Any,
    atol_mean: float = 1e-2,
    atol_std: float = 1e-2,
    raise_on_fail: bool = False,
) -> None:
    """Check zero mean and unbiased unit standard deviation across observations."""
    mean, std, means, stds = _outcome_stats(Y)

    def exceeds(values: list[float], target: float, tolerance: float) -> bool:
        return not any(math.isnan(v) for v in values) and any(
            abs(v - target) > tolerance for v in values
        )

    if exceeds(means, 0, atol_mean) or exceeds(stds, 1, atol_std):
        detail = f"mean = {mean}" if std is None else f"std = {std}, mean = {mean}"
        spacer = " " if std is None else ""
        _scaling_failure(
            f"Data (outcome observations) is not standardized ({detail}).{spacer}"
            "Please consider scaling the input to zero mean and unit variance.",
            raise_on_fail,
        )


def validate_input_scaling(
    train_X: Any,
    train_Y: Any,
    train_Yvar: Any = None,
    raise_on_fail: bool = False,
    ignore_X_dims: list[int] | None = None,
    check_nans_only: bool = False,
) -> None:
    """Validate NaNs, nonnegative observation noise and model data scaling."""
    if settings.validate_input_scaling.off():
        return
    check_no_nans(train_X)
    check_no_nans(train_Y)
    if train_Yvar is not None:
        check_no_nans(train_Yvar)
        if any(value < 0 for value in to_flat(train_Yvar)[0]):
            raise InputDataError("Input data contains negative variances.")
    if not check_nans_only:
        check_min_max_scaling(train_X, raise_on_fail=raise_on_fail, ignore_dims=ignore_X_dims)
        check_standardization(train_Y, raise_on_fail=raise_on_fail)


def mod_batch_shape(module: Any, names: list[str], b: int) -> None:
    """Change the last batch dimension of a named descendant module in place."""
    if not names:
        return
    child = getattr(module, names[0])
    if len(names) == 1 and hasattr(child, "batch_shape") and len(child.batch_shape) > 0:
        shape = (*child.batch_shape[:-1], *([b] if b > 0 else []))
        child.batch_shape = type(child.batch_shape)(shape)
    else:
        mod_batch_shape(child, names[1:], b)


@contextmanager
def gpt_posterior_settings() -> Iterator[None]:
    """Apply BoTorch's GPyTorch posterior settings when GPyTorch is in use.

    Native Rust posteriors do not consult GPyTorch's solver/cache settings.
    The context does not import GPyTorch when it has not been loaded.
    """
    with ExitStack() as stack:
        gpytorch = sys.modules.get("gpytorch")
        if gpytorch is not None:
            gpt_settings = gpytorch.settings
            if gpt_settings.debug.is_default():
                stack.enter_context(gpt_settings.debug(False))
            if gpt_settings.fast_pred_var.is_default():
                stack.enter_context(gpt_settings.fast_pred_var())
            stack.enter_context(gpt_settings.detach_test_caches(settings.propagate_grads.off()))
        yield


def detect_duplicates(X: Any, rtol: float = 0, atol: float = 1e-8) -> Iterator[tuple[int, int]]:
    """Return duplicate/original pairs using the strict infinity-norm tolerance."""
    if len(_ops.shape_of(X)) != 2:
        raise ValueError("X must have 2 dimensions.")
    check_no_nans(X)
    return iter(_rs.detect_duplicates(to_flat(X, "X"), float(rtol), float(atol)))


def consolidate_duplicates(
    X: Any, Y: Any, rtol: float = 0.0, atol: float = 1e-8
) -> tuple[Any, Any, Any]:
    """Remove duplicate rows and remap observation indices, retaining chain links."""
    n = _ops.shape_of(X)[-2] if len(_ops.shape_of(X)) == 2 else 0
    duplicates = dict(detect_duplicates(X, rtol=rtol, atol=atol))
    common = set(duplicates).intersection(duplicates.values())
    duplicates = {k: v for k, v in duplicates.items() if k not in common and v not in common}
    kind = kind_of(Y)
    if not duplicates:
        indices = _ops.int_array(range(n), (n,), kind)
        if kind.lib == "torch":
            indices = indices.to(dtype=Y.dtype)
        elif kind.lib == "numpy":
            indices = indices.astype(Y.dtype, copy=False)
        return X, Y, indices
    unique = sorted(set(range(n)) - set(duplicates))
    new = {old: i for i, old in enumerate(unique)}
    new.update({old: new[original] for old, original in duplicates.items()})
    values, shape = to_flat(Y, "Y")
    remapped = []
    for value in values:
        if int(value) != value or int(value) not in new:
            raise KeyError(value)
        remapped.append(new[int(value)])
    if kind_of(X).lib == "list":
        consolidated_X = [X[index][:] for index in unique]
    else:
        consolidated_X = X[unique, :]
    return (
        consolidated_X,
        _ops.int_array(remapped, shape, kind),
        _ops.int_array([new[index] for index in range(n)], (n,), kind),
    )


def get_task_value_remapping(all_task_values: Any, dtype: Any) -> Any:
    """Map sorted, nonnegative integer task labels to contiguous labels."""
    values, shape = to_flat(all_task_values, "all_task_values")
    kind = kind_of(all_task_values)
    if kind.lib == "torch":
        torch = sys.modules["torch"]
        valid_dtype = dtype in (torch.float32, torch.float64)
    elif kind.lib == "numpy":
        np = sys.modules["numpy"]
        dtype = np.dtype(dtype)
        valid_dtype = dtype in (np.dtype("float32"), np.dtype("float64"))
    else:
        valid_dtype = dtype in (float, "float32", "float64")
    if not valid_dtype:
        raise ValueError(f"dtype must be float32 or float64, but got {dtype}.")
    if len(shape) != 1 or any(v < 0 or not math.isfinite(v) or v != int(v) for v in values):
        raise ValueError("Task values must be a one-dimensional sequence of nonnegative integers.")
    tasks = [int(v) for v in values]
    if tasks != sorted(set(tasks)):
        raise ValueError("Task values must be sorted and unique.")
    if tasks == list(range(len(tasks))):
        return None
    mapping = [math.nan] * (tasks[-1] + 1)
    for index, task in enumerate(tasks):
        mapping[task] = float(index)
    return from_flat(mapping, (len(mapping),), ArrayKind(kind.lib, dtype, kind.device))


def extract_targets_and_noise_single_output(model: Any) -> tuple[Any, Any]:
    """Return single-output training targets and optional fixed noise as columns."""
    from botorch_rs.likelihoods import FixedNoiseGaussianLikelihood

    targets = _ops.getitem(model.train_targets, (Ellipsis, None))
    likelihood = model.likelihood
    fixed = isinstance(likelihood, FixedNoiseGaussianLikelihood) or any(
        base.__name__ == "FixedNoiseGaussianLikelihood" for base in type(likelihood).__mro__
    )
    noise = _ops.getitem(likelihood.noise_covar.noise, (Ellipsis, None)) if fixed else None
    return targets, noise


__all__ = [
    "add_output_dim",
    "check_min_max_scaling",
    "check_no_nans",
    "check_standardization",
    "consolidate_duplicates",
    "detect_duplicates",
    "extract_targets_and_noise_single_output",
    "fantasize",
    "get_task_value_remapping",
    "gpt_posterior_settings",
    "mod_batch_shape",
    "multioutput_to_batch_mode_transform",
    "validate_input_scaling",
]

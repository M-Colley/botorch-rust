"""Helpers for tensor transformations (``botorch.utils.transforms``).

``standardize``, ``normalize``, ``unnormalize`` and ``match_batch_shape``
operate on torch tensors (with torch's operations, so that gradients flow),
NumPy arrays and nested lists with BoTorch's semantics; the decorators
``t_batch_mode_transform``, ``concatenate_pending_points`` and
``average_over_ensemble_models`` work on botorch_rs and BoTorch acquisition
functions alike.
"""

from __future__ import annotations

import math
import sys
import warnings
from collections.abc import Callable, Iterable
from functools import wraps
from typing import Any

from botorch_rs import _ops
from botorch_rs._arrays import from_flat, is_torch_tensor, kind_of, to_flat


def normalize_indices(indices: list[int] | None, d: int) -> list[int] | None:
    r"""Normalize a list of indices to ensure that they are positive.

    Args:
        indices: A list of indices (may contain negative indices for indexing
            "from the back").
        d: The dimension of the tensor to index.

    Returns:
        A normalized list of indices such that each index is between ``0`` and
        ``d-1``, or None if indices is None.
    """
    if indices is None:
        return indices
    normalized_indices = []
    for i in indices:
        if i < 0:
            i = i + d
        if i < 0 or i > d - 1:
            raise ValueError(f"Index {i} out of bounds for tensor or length {d}.")
        normalized_indices.append(i)
    return normalized_indices


def is_ensemble(model: Any) -> bool:
    r"""Check if at least one model is an ensemble model.

    Args:
        model: A BoTorch model (may be a ``ModelList`` or ``ModelListGP``)

    Returns:
       True if at least one model is an ensemble model.
    """
    models = getattr(model, "models", None)
    if isinstance(models, Iterable):
        return any(is_ensemble(m) for m in models)
    return bool(getattr(model, "_is_ensemble", False))


def is_fully_bayesian(model: Any) -> bool:
    r"""Check if at least one model is a fully Bayesian model.

    Args:
        model: A BoTorch model (may be a ``ModelList`` or ``ModelListGP``)

    Returns:
        True if at least one model is a fully Bayesian model.
    """
    models = getattr(model, "models", None)
    if isinstance(models, Iterable):
        return any(is_fully_bayesian(m) for m in models)
    return bool(getattr(model, "_is_fully_bayesian", False))


def _is_numpy(x: Any) -> bool:
    np = sys.modules.get("numpy")
    return np is not None and isinstance(x, np.ndarray)


def standardize(Y: Any) -> Any:
    r"""Standardizes (zero mean, unit variance) a tensor by dim=-2.

    If the tensor is single-dimensional, simply standardizes the tensor.
    If for some batch index all elements are equal (or if there is only a single
    data point), this function will return 0 for that batch index.

    Args:
        Y: A ``batch_shape x n x m``-dim tensor.

    Returns:
        The standardized ``Y``.

    Example:
        >>> Y = torch.rand(4, 3)
        >>> Y_standardized = standardize(Y)
    """
    if is_torch_tensor(Y):
        torch = sys.modules["torch"]
        stddim = -1 if Y.dim() < 2 else -2
        Y_std = Y.std(dim=stddim, keepdim=True)
        Y_std = Y_std.where(Y_std >= 1e-9, torch.full_like(Y_std, 1.0))
        return (Y - Y.mean(dim=stddim, keepdim=True)) / Y_std
    data, shape = to_flat(Y, "Y")
    if not shape:
        raise IndexError("standardize requires at least one dimension")
    axis = len(shape) - 1 if len(shape) < 2 else len(shape) - 2
    n = shape[axis]
    inner = math.prod(shape[axis + 1 :])
    outer = math.prod(shape[:axis])
    out = [0.0] * len(data)
    for o in range(outer):
        for i in range(inner):
            idx = [(o * n + k) * inner + i for k in range(n)]
            vals = [float(data[j]) for j in idx]
            mean = math.fsum(vals) / n
            if n > 1:
                std = math.sqrt(math.fsum((v - mean) ** 2 for v in vals) / (n - 1))
            else:
                std = math.nan
            if not std >= 1e-9:
                std = 1.0
            for j, v in zip(idx, vals, strict=True):
                out[j] = (v - mean) / std
    return from_flat(out, shape, kind_of(Y))


def _update_constant_bounds(bounds: Any) -> Any:
    r"""If the lower and upper bounds are identical for a dimension, set
    the upper bound to lower bound + 1.

    If any modification is needed, this will return a clone of the original
    tensor to avoid in-place modification.

    Args:
        bounds: A ``2 x d``-dim tensor of lower and upper bounds.

    Returns:
        A ``2 x d``-dim tensor of updated lower and upper bounds.
    """
    if is_torch_tensor(bounds) or _is_numpy(bounds):
        constant_dims = bounds[1] == bounds[0]
        if constant_dims.any():
            bounds = bounds.clone() if is_torch_tensor(bounds) else bounds.copy()
            bounds[1, constant_dims] = bounds[0, constant_dims] + 1
        return bounds
    data, shape = to_flat(bounds, "bounds")
    d = shape[-1]
    lo, hi = list(data[:d]), list(data[d:])
    if any(a == b for a, b in zip(lo, hi, strict=True)):
        hi = [a + 1 if a == b else b for a, b in zip(lo, hi, strict=True)]
        return from_flat(lo + hi, shape, kind_of(bounds))
    return bounds


def _rows(bounds: Any) -> tuple[Any, Any]:
    return _ops.getitem(bounds, 0), _ops.getitem(bounds, 1)


def normalize(X: Any, bounds: Any, update_constant_bounds: bool = True) -> Any:
    r"""Min-max normalize X w.r.t. the provided bounds.

    Args:
        X: ``... x d`` tensor of data
        bounds: ``2 x d`` tensor of lower and upper bounds for each of the X's d
            columns.
        update_constant_bounds: If ``True``, update the constant bounds in order to
            avoid division by zero issues. When the upper and lower bounds are
            identical for a dimension, that dimension will not be scaled. Such
            dimensions will only be shifted as
            ``new_X[..., i] = X[..., i] - bounds[0, i]``.

    Returns:
        A ``... x d``-dim tensor of normalized data, given by
            ``(X - bounds[0]) / (bounds[1] - bounds[0])``. If all elements of ``X``
            are contained within ``bounds``, the normalized values will be
            contained within ``[0, 1]^d``.

    Example:
        >>> X = torch.rand(4, 3)
        >>> bounds = torch.stack([torch.zeros(3), 0.5 * torch.ones(3)])
        >>> X_normalized = normalize(X, bounds)
    """
    bounds = _update_constant_bounds(bounds=bounds) if update_constant_bounds else bounds
    lo, hi = _rows(bounds)
    return _ops.arith("div", _ops.arith("sub", X, lo), _ops.arith("sub", hi, lo))


def unnormalize(X: Any, bounds: Any, update_constant_bounds: bool = True) -> Any:
    r"""Un-normalizes X w.r.t. the provided bounds.

    Args:
        X: ``... x d`` tensor of data
        bounds: ``2 x d`` tensor of lower and upper bounds for each of the X's d
            columns.
        update_constant_bounds: If ``True``, update the constant bounds in order to
            avoid division by zero issues. When the upper and lower bounds are
            identical for a dimension, that dimension will not be scaled. Such
            dimensions will only be shifted as
            ``new_X[..., i] = X[..., i] + bounds[0, i]``. This is the inverse of
            the behavior of ``normalize`` when ``update_constant_bounds=True``.

    Returns:
        A ``... x d``-dim tensor of unnormalized data, given by
            ``X * (bounds[1] - bounds[0]) + bounds[0]``. If all elements of ``X``
            are contained in ``[0, 1]^d``, the un-normalized values will be
            contained within ``bounds``.

    Example:
        >>> X_normalized = torch.rand(4, 3)
        >>> bounds = torch.stack([torch.zeros(3), 0.5 * torch.ones(3)])
        >>> X = unnormalize(X_normalized, bounds)
    """
    bounds = _update_constant_bounds(bounds=bounds) if update_constant_bounds else bounds
    lo, hi = _rows(bounds)
    return _ops.arith("add", _ops.arith("mul", X, _ops.arith("sub", hi, lo)), lo)


def _verify_output_shape(acqf: Any, X: Any, output: Any) -> bool:
    r"""
    Performs the output shape checks for ``t_batch_mode_transform``. Output shape checks
    help in catching the errors due to AcquisitionFunction arguments with erroneous
    return shapes before these errors propagate further down the line.

    This method checks that the ``output`` shape matches either the t-batch shape of X
    or the ``batch_shape`` of ``acqf.model``.

    Args:
        acqf: The AcquisitionFunction object being evaluated.
        X: The ``... x q x d``-dim input tensor with an explicit t-batch.
        output: The return value of ``acqf.method(X, ...)``.

    Returns:
        True if ``output`` has the correct shape, False otherwise.
    """
    try:
        X_batch_shape = tuple(_ops.shape_of(X)[:-2])
        out_shape = tuple(_ops.shape_of(output))
        if out_shape == X_batch_shape:
            return True
        if out_shape == () and X_batch_shape == (1,):
            # X has a batch shape of [1] which gets squeezed.
            return True
        # Cases with model batch shape involved.
        model_b_shape = tuple(acqf.model.batch_shape)
        if out_shape == model_b_shape:
            # Simple inputs with batched model.
            return True
        model_b_dim = len(model_b_shape)
        if out_shape == X_batch_shape[:-model_b_dim] + model_b_shape and all(
            xs in [1, ms]
            for xs, ms in zip(X_batch_shape[-model_b_dim:], model_b_shape, strict=False)
        ):
            # X has additional batch dimensions beyond the model batch shape.
            return True
        return False
    except (AttributeError, NotImplementedError):
        # acqf does not have model or acqf.model does not define ``batch_shape``
        warnings.warn(
            "Output shape checks failed! Expected output shape to match t-batch shape"
            f"of X, but got output with shape {_ops.shape_of(output)} for X with shape "
            f"{_ops.shape_of(X)}. Make sure that this is the intended behavior!",
            RuntimeWarning,
            stacklevel=3,
        )
        return True


def _is_array(X: Any) -> bool:
    return is_torch_tensor(X) or _is_numpy(X)


def t_batch_mode_transform(
    expected_q: int | None = None,
    assert_output_shape: bool = True,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    r"""Factory for decorators enabling consistent t-batch behavior.

    This method creates decorators for instance methods to transform an input tensor
    ``X`` to t-batch mode (i.e. with at least 3 dimensions). This assumes the tensor
    has a q-batch dimension. The decorator also checks the q-batch size if
    ``expected_q`` is provided, and the output shape if ``assert_output_shape``
    is ``True``.

    Args:
        expected_q: The expected q-batch size of ``X``. If specified, this will raise an
            AssertionError if ``X``'s q-batch size does not equal expected_q.
        assert_output_shape: If ``True``, this will raise an AssertionError if the
            output shape does not match either the t-batch shape of ``X``,
            or the ``acqf.model.batch_shape`` for acquisition functions using
            batched models.

    Returns:
        The decorated instance method.

    Torch tensors and NumPy arrays are transformed; other inputs (e.g. nested
    lists, as BoTorch's non-tensor inputs) are passed through unchanged.

    Example:
        >>> class ExampleClass:
        >>>     @t_batch_mode_transform(expected_q=1)
        >>>     def single_q_method(self, X):
        >>>         ...
        >>>
        >>>     @t_batch_mode_transform()
        >>>     def arbitrary_q_method(self, X):
        >>>         ...
    """

    def decorator(method: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(method)
        def decorated(acqf: Any, X: Any, *args: Any, **kwargs: Any) -> Any:
            # Allow using acquisition functions for other inputs (e.g. lists of strings)
            if not _is_array(X):
                return method(acqf, X, *args, **kwargs)

            if X.ndim < 2:
                raise ValueError(
                    f"{type(acqf).__name__} requires X to have at least 2 dimensions,"
                    f" but received X with only {X.ndim} dimensions."
                )
            elif expected_q is not None and X.shape[-2] != expected_q:
                raise AssertionError(
                    f"Expected X to be `batch_shape x q={expected_q} x d`, but"
                    f" got X with shape {X.shape}."
                )
            X_original_shape = X.shape
            # add t-batch dim
            X = X if X.ndim > 2 else X[None]
            output = method(acqf, X, *args, **kwargs)
            if assert_output_shape and not _verify_output_shape(
                acqf=acqf,
                X=X,
                output=output,
            ):
                raise AssertionError(
                    "Expected the output shape to match either the t-batch shape of "
                    "X, or the `model.batch_shape` in the case of acquisition "
                    "functions using batch models; but got output with shape "
                    f"{output.shape} for X with shape {X.shape}."
                    + (
                        ""
                        if X_original_shape == X.shape
                        else f" Note that `X.shape` was originally {X_original_shape} "
                        "before the `t_batch_mode_transform` decorator added a batch "
                        "dimension."
                    )
                )
            return output

        return decorated

    return decorator


def average_over_ensemble_models(method: Callable[..., Any]) -> Callable[..., Any]:
    """Decorator for averaging acquisition values over ensemble models.

    For example, if the model is an ensemble, ``is_ensemble(model) == True``
    like for a SAAS model, the acquisition value is averaged over the samples
    in the ensemble.

    NOTE: If the class has a ``_log`` attribute, the acquisition value is averaged
    using logmeanexp instead of mean so that the log of the averaged acquisition value
    is averaged in a numerically stable way.

    Args:
        method: The method to be decorated, usually ``forward``.

    Returns:
        The decorated method.
    """

    def decorated(acqf: Any, X: Any, *args: Any, **kwargs: Any) -> Any:
        output = method(acqf, X, *args, **kwargs)
        if hasattr(acqf, "model") and is_ensemble(acqf.model):
            if getattr(acqf, "_log", False):
                from botorch_rs.utils.safe_math import logmeanexp

                output = logmeanexp(output, dim=-1)
            elif is_torch_tensor(output) or _is_numpy(output):
                output = output.mean(-1)
            else:
                data, shape = to_flat(output, "output")
                n = shape[-1]
                means = [math.fsum(data[i : i + n]) / n for i in range(0, len(data), n)]
                output = from_flat(means, shape[:-1], kind_of(output))
        return output

    return decorated


def concatenate_pending_points(method: Callable[..., Any]) -> Callable[..., Any]:
    r"""Decorator concatenating X_pending into an acquisition function's argument.

    This decorator works on the ``forward`` method of acquisition functions taking
    a tensor ``X`` as the argument. If the acquisition function has an ``X_pending``
    attribute (that is not ``None``), this is concatenated into the input ``X``,
    appropriately expanding the pending points to match the batch shape of ``X``.

    Example:
        >>> class ExampleAcquisitionFunction:
        >>>     @concatenate_pending_points
        >>>     @t_batch_mode_transform()
        >>>     def forward(self, X):
        >>>         ...
    """

    @wraps(method)
    def decorated(cls: Any, X: Any, **kwargs: Any) -> Any:
        if cls.X_pending is not None and getattr(cls, "_should_concatenate_pending_points", True):
            X = _ops.concat([X, match_batch_shape(cls.X_pending, X)], dim=-2)
        return method(cls, X, **kwargs)

    return decorated


def match_batch_shape(X: Any, Y: Any) -> Any:
    r"""Matches the batch dimension of a tensor to that of another tensor.

    Args:
        X: A ``batch_shape_X x q x d`` tensor, whose batch dimensions that
            correspond to batch dimensions of ``Y`` are to be matched to those
            (if compatible).
        Y: A ``batch_shape_Y x q' x d`` tensor.

    Returns:
        A ``batch_shape_Y x q x d`` tensor containing the data of ``X`` expanded to
        the batch dimensions of ``Y`` (if compatible). For instance, if ``X`` is
        ``b'' x b' x q x d`` and ``Y`` is ``b x q x d``, then the returned tensor is
        ``b'' x b x q x d``.

    Example:
        >>> X = torch.rand(2, 1, 5, 3)
        >>> Y = torch.rand(2, 6, 4, 3)
        >>> X_matched = match_batch_shape(X, Y)
        >>> X_matched.shape
        torch.Size([2, 6, 5, 3])

    """
    xs = tuple(_ops.shape_of(X))
    ys = tuple(_ops.shape_of(Y))
    lead = xs[: -len(ys)] if len(ys) < len(xs) else ()
    return _ops.expand(X, (*lead, *ys[:-2], *xs[-2:]))


__all__ = [
    "average_over_ensemble_models",
    "concatenate_pending_points",
    "is_ensemble",
    "is_fully_bayesian",
    "match_batch_shape",
    "normalize",
    "normalize_indices",
    "standardize",
    "t_batch_mode_transform",
    "unnormalize",
]

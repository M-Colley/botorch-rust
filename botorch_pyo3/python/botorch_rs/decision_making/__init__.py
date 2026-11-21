"""Preference-aware multiobjective decisions and FPBI front generation.

These extensions implement Schmitt et al. (2022), DOI 10.3390/inventions7030046.
They consume objective outcomes or model predictions and do not replace an
acquisition function. Maximization is the default, as in BoTorch.
"""

from __future__ import annotations

import importlib
import itertools
import math
from typing import Any

from botorch_rs import _botorch_rs as _rs
from botorch_rs._arrays import bool_array, is_torch_tensor, kind_of, to_flat


def _options(alpha: float, maximize: bool) -> None:
    if isinstance(alpha, bool) or not isinstance(alpha, (int, float)):
        raise TypeError("alpha must be a number in [0, 1].")
    if not math.isfinite(alpha) or not 0 <= alpha <= 1:
        raise ValueError("alpha must be in [0, 1].")
    if not isinstance(maximize, bool):
        raise TypeError("maximize must be a bool.")


def knee_region(Y: Any, *, alpha: float = 0.5, maximize: bool = True) -> Any:
    """Original-row mask of the knee region (paper equations 8--11).

    ``Y`` has shape ``(..., n, m)``. Dominated rows and repeated rows after
    their first occurrence are excluded before normalization. A flat objective
    contributes zero. The paper's signed distances are used: ``alpha=0``
    does not necessarily retain every Pareto row.

    Lists and NumPy arrays use native Rust. Tensor inputs stay on their Torch
    device. The output has the input's array kind and boolean dtype.
    """
    _options(alpha, maximize)
    if is_torch_tensor(Y):
        from ._selection import knee_region as torch_knee_region

        return torch_knee_region(Y, alpha=alpha, maximize=maximize)
    values, shape = to_flat(Y, "Y")
    mask, out_shape = _rs.decision_knee_region((values, shape), alpha, maximize)
    return bool_array(mask, out_shape, kind_of(Y))


def _reshape_indices(values: list[int], shape: tuple[int, ...]) -> Any:
    if not shape:
        return values[0]
    step = math.prod(shape[1:])
    return [_reshape_indices(values[i * step : (i + 1) * step], shape[1:]) for i in range(shape[0])]


def select_preferred(Y: Any, preferences: Any, *, alpha: float = 0.5, maximize: bool = True) -> Any:
    """Return the first preferred original row in each outcome batch.

    Nonnegative, nonzero preferences of shape ``(..., m)`` broadcast across
    ``Y``'s batch dimensions. Returns int64 Torch/NumPy indices, or Python
    integer(s) for lists. Relative preference scale is immaterial.
    """
    _options(alpha, maximize)
    if is_torch_tensor(Y):
        from ._selection import select_preferred as torch_select_preferred

        return torch_select_preferred(Y, preferences, alpha=alpha, maximize=maximize)
    values, shape = to_flat(Y, "Y")
    p, pshape = to_flat(preferences, "preferences")
    if len(shape) < 2 or not shape[-2] or not shape[-1] or not math.prod(shape[:-2]):
        raise ValueError("Y must have nonempty shape (..., n, m).")
    batch, n, m = shape[:-2], shape[-2], shape[-1]
    if not pshape or pshape[-1] != m:
        raise ValueError("preferences must have shape (..., m).")
    leading = pshape[:-1]
    if len(leading) > len(batch):
        raise ValueError("preferences do not broadcast to Y's batch shape.")
    padded = (1,) * (len(batch) - len(leading)) + leading
    if any(a not in (1, b) for a, b in zip(padded, batch, strict=True)):
        raise ValueError("preferences do not broadcast to Y's batch shape.")
    indices = []
    for flat_batch, coords in enumerate(itertools.product(*(range(b) for b in batch))):
        preference_row = 0
        for coordinate, size in zip(coords, padded, strict=True):
            preference_row = preference_row * size + (0 if size == 1 else coordinate)
        weights = p[preference_row * m : (preference_row + 1) * m]
        block = values[flat_batch * n * m : (flat_batch + 1) * n * m]
        result, _ = _rs.decision_select_preferred((block, (n, m)), weights, alpha, maximize)
        indices.append(result[0])
    kind = kind_of(Y)
    if kind.lib == "numpy":
        np = importlib.import_module("numpy")
        return np.asarray(indices, dtype=np.int64).reshape(batch)
    return _reshape_indices(indices, batch)


def fpbi_geometry(extreme_Y: Any, **kwargs: Any) -> Any:
    """Build the FPBI plane/grid; focus uses normalized cost coordinates.

    Accepts ``(..., m, m)`` objective extrema. See the matching OpenBO API
    for ``n_steps``, ``focus``, ``maximize`` and ``max_rays`` options.
    """
    backend = "_fpbi" if is_torch_tensor(extreme_Y) else "_native"
    module = importlib.import_module(f"botorch_rs.decision_making.{backend}")
    return module.fpbi_geometry(extreme_Y, **kwargs)


def sample_fpbi(objective: Any, bounds: Any, **kwargs: Any) -> Any:
    """Generate a constrained FPBI front with local multistart SLSQP.

    Bounds are ``(2,d)``; callbacks accept ``(n,d)`` inputs and return
    ``(n,m)`` outcomes of the bounds' array kind. Lists/NumPy use native
    Rust without SciPy. Torch uses the CPU SciPy adapter and evaluates
    callbacks on the original device. See ``docs/decision-making.md``.
    """
    backend = "_fpbi" if is_torch_tensor(bounds) else "_native"
    module = importlib.import_module(f"botorch_rs.decision_making.{backend}")
    return module.sample_fpbi(objective, bounds, **kwargs)


def sample_model_fpbi(
    model: Any,
    bounds: Any,
    *,
    model_batch_index: tuple[int, ...] | None = None,
    **kwargs: Any,
) -> Any:
    """Sample a front of posterior means of a BoTorch or botorch_rs model.

    These are surrogate predictions, not new observed evaluations. An
    explicitly chosen batch is required when a model returns batched means.
    """
    if is_torch_tensor(bounds):
        module = importlib.import_module("botorch_rs.decision_making._fpbi")
        return module.sample_model_fpbi(
            model, bounds, model_batch_index=model_batch_index, **kwargs
        )

    def objective(X):
        mean = model.posterior(X).mean
        if model_batch_index is not None:
            if kind_of(mean).lib == "numpy":
                mean = mean[model_batch_index]
            else:
                for i in model_batch_index:
                    mean = mean[i]
        _, shape = to_flat(mean, "posterior mean")
        if len(shape) != 2:
            raise ValueError("Choose model_batch_index for batched model posterior means.")
        return mean

    return sample_fpbi(objective, bounds, **kwargs)


def __getattr__(name: str) -> Any:
    if name in {"FPBIGeometry", "FPBIResult", "SolverDiagnostic"}:
        module = importlib.import_module("botorch_rs.decision_making._types")
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "FPBIGeometry",
    "FPBIResult",
    "SolverDiagnostic",
    "fpbi_geometry",
    "knee_region",
    "sample_fpbi",
    "sample_model_fpbi",
    "select_preferred",
]

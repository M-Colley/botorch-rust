"""Acquisition-function optimization (``botorch.optim.optimize``)."""

from __future__ import annotations

import warnings
from typing import Any

from botorch_rs import _botorch_rs as _rs
from botorch_rs._arrays import as_float, first_kind, from_flat, from_pair, to_flat
from botorch_rs.acquisition.acquisition import AcquisitionFunction, check_unsupported
from botorch_rs.exceptions.errors import BotorchTensorDimensionError, UnsupportedError
from botorch_rs.exceptions.warnings import BadInitialCandidatesWarning

_EPS = 2.220446049250313e-16

# ``options`` entries: initialization and L-BFGS-B settings.
_INT_OPTIONS = {"seed", "batch_limit", "init_batch_limit", "maxiter", "maxfun", "maxcor", "maxls"}
_FLOAT_OPTIONS = {"ftol", "gtol", "eta", "alpha"}
_IGNORED_OPTIONS = {"disp", "iprint"}

_WARNINGS: dict[str, type[Warning]] = {
    "BadInitialCandidatesWarning": BadInitialCandidatesWarning,
    "RuntimeWarning": RuntimeWarning,
}


def _parse_options(options: dict[str, Any] | None) -> dict[str, Any]:
    opts = dict(options or {})
    out: dict[str, Any] = {}
    method = opts.pop("method", "L-BFGS-B")
    if method != "L-BFGS-B":
        raise UnsupportedError(f"Only method='L-BFGS-B' is supported, got {method!r}.")
    if not opts.pop("with_grad", True):
        raise UnsupportedError("with_grad=False is not supported (exact gradients are used).")
    if "pgtol" in opts:
        opts.setdefault("gtol", opts.pop("pgtol"))
    if "max_cor" in opts:
        opts.setdefault("maxcor", opts.pop("max_cor"))
    if "factr" in opts:
        opts.setdefault("ftol", float(opts.pop("factr")) * _EPS)
    if opts.pop("nonnegative", False):
        out["nonnegative"] = True
    for key in list(opts):
        value = opts.pop(key)
        if key in _INT_OPTIONS:
            out[key] = None if value is None else int(value)
        elif key in _FLOAT_OPTIONS:
            out[key] = None if value is None else float(value)
        elif key not in _IGNORED_OPTIONS:
            raise UnsupportedError(f"optimize_acqf option {key!r} is not supported by botorch_rs.")
    return out


def optimize_acqf(
    acq_function: AcquisitionFunction,
    bounds: Any,
    q: int,
    num_restarts: int,
    raw_samples: int | None = None,
    options: dict[str, Any] | None = None,
    inequality_constraints: Any = None,
    equality_constraints: Any = None,
    nonlinear_inequality_constraints: Any = None,
    fixed_features: dict[int, Any] | None = None,
    post_processing_func: Any = None,
    batch_initial_conditions: Any = None,
    return_best_only: bool = True,
    gen_candidates: Any = None,
    sequential: bool = False,
    acq_function_sequence: Any = None,
    *,
    ic_generator: Any = None,
    timeout_sec: float | None = None,
    return_acq_values: bool = True,
    return_full_tree: bool = False,
    retry_on_optimization_warning: bool = True,
    **ic_gen_kwargs: Any,
) -> tuple[Any, Any]:
    r"""Optimizes ``acq_function`` over the box ``bounds`` (``2 x d``) with
    multi-start L-BFGS-B, like BoTorch's ``optimize_acqf``.

    ``raw_samples`` scrambled Sobol candidate sets are evaluated and
    ``num_restarts`` of them are selected as starting points (Boltzmann
    sampling, or the non-negative heuristic for non-negative acquisition
    functions); every start is optimized by an independent L-BFGS-B run with
    exact gradients (in parallel), and the best result is returned.

    Supported ``options``: ``seed`` (raw samples and initial-condition
    selection; BoTorch draws the latter from torch's global RNG),
    ``batch_limit``, ``init_batch_limit``, ``eta``, ``alpha``, ``nonnegative``
    and the L-BFGS-B settings ``maxiter`` (2000), ``maxfun``, ``ftol``,
    ``gtol``, ``maxcor``, ``maxls``.

    Returns:
        ``(candidates, acq_value)``: a ``q x d`` candidate set (``num_restarts x
        q x d`` with ``return_best_only=False``) and its acquisition value (a
        scalar; ``q`` values for ``sequential=True``; one per restart with
        ``return_best_only=False``), of the array type of ``bounds``.
    """
    if not isinstance(acq_function, AcquisitionFunction):
        raise UnsupportedError(
            "optimize_acqf requires a botorch_rs acquisition function (got "
            f"{type(acq_function).__name__})."
        )
    check_unsupported(
        inequality_constraints=inequality_constraints,
        equality_constraints=equality_constraints,
        nonlinear_inequality_constraints=nonlinear_inequality_constraints,
        post_processing_func=post_processing_func,
        gen_candidates=gen_candidates,
        acq_function_sequence=acq_function_sequence,
        ic_generator=ic_generator,
        timeout_sec=timeout_sec,
        return_full_tree=return_full_tree,
    )
    if ic_gen_kwargs:
        raise UnsupportedError(f"Unsupported keyword arguments: {sorted(ic_gen_kwargs)}.")
    if sequential and not return_best_only:
        raise UnsupportedError("`return_best_only=False` only supported for joint optimization.")
    kind = first_kind(bounds, batch_initial_conditions)
    bounds_flat = to_flat(bounds, "bounds")
    if len(bounds_flat[1]) != 2 or bounds_flat[1][0] != 2:
        raise BotorchTensorDimensionError(
            f"bounds should be a `2 x d` tensor, current shape: {bounds_flat[1]}."
        )
    d = bounds_flat[1][1]
    fixed = None
    if fixed_features:
        fixed = {int(k) % d: as_float(v, f"fixed_features[{k}]") for k, v in fixed_features.items()}
    ics = None
    if batch_initial_conditions is not None:
        ics = to_flat(batch_initial_conditions, "batch_initial_conditions")
    opts = _parse_options(options)
    candidates, acq_values, restart_x, restart_values, messages = _rs.optimize_acqf(
        acq_function._rs,
        bounds_flat,
        int(q),
        int(num_restarts),
        None if raw_samples is None else int(raw_samples),
        fixed_features=fixed,
        batch_initial_conditions=ics,
        sequential=bool(sequential),
        retry_on_optimization_warning=bool(retry_on_optimization_warning),
        **opts,
    )
    for category, message in messages:
        warnings.warn(message, _WARNINGS.get(category, RuntimeWarning), stacklevel=2)
    if return_best_only:
        X = from_pair(candidates, kind)
        if sequential:
            values = from_flat(acq_values, (len(acq_values),), kind)
        else:
            values = from_flat(acq_values[:1], (), kind)
    else:
        X = from_pair(restart_x, kind)
        values = from_flat(restart_values, (len(restart_values),), kind)
    return X, (values if return_acq_values else None)


__all__ = ["optimize_acqf"]

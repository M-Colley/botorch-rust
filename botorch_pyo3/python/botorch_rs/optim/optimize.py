"""Acquisition-function optimization (``botorch.optim.optimize``)."""

from __future__ import annotations

import warnings
from typing import Any

from botorch_rs import _botorch_rs as _rs
from botorch_rs._arrays import Flat, as_float, first_kind, from_flat, from_pair, to_flat
from botorch_rs.acquisition.acquisition import AcquisitionFunction, check_unsupported
from botorch_rs.exceptions.errors import BotorchTensorDimensionError, UnsupportedError
from botorch_rs.exceptions.warnings import BadInitialCandidatesWarning
from botorch_rs.optim.parameter_constraints import (
    linear_constraints_arg,
    nonlinear_constraints_arg,
)

_EPS = 2.220446049250313e-16

# ``options`` entries: initialization, L-BFGS-B and SLSQP settings.
_INT_OPTIONS = {
    "seed",
    "batch_limit",
    "init_batch_limit",
    "maxiter",
    "maxfun",
    "maxcor",
    "maxls",
    "n_burnin",
    "n_thinning",
}
_FLOAT_OPTIONS = {"ftol", "gtol", "eta", "alpha"}
_IGNORED_OPTIONS = {"disp", "iprint"}

_WARNINGS: dict[str, type[Warning]] = {
    "BadInitialCandidatesWarning": BadInitialCandidatesWarning,
    "RuntimeWarning": RuntimeWarning,
}


def _parse_options(options: dict[str, Any] | None) -> dict[str, Any]:
    opts = dict(options or {})
    out: dict[str, Any] = {}
    method = opts.pop("method", None)
    if method is not None:
        if method not in ("L-BFGS-B", "SLSQP"):
            raise UnsupportedError(
                f"Only method='L-BFGS-B' and method='SLSQP' are supported, got {method!r}."
            )
        out["method"] = method
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
    for key in ("sample_around_best", "topn"):
        if opts.pop(key, False):
            raise UnsupportedError(f"optimize_acqf option {key!r} is not supported by botorch_rs.")
    for key in list(opts):
        value = opts.pop(key)
        if key in _INT_OPTIONS:
            out[key] = None if value is None else int(value)
        elif key in _FLOAT_OPTIONS:
            out[key] = None if value is None else float(value)
        elif key not in _IGNORED_OPTIONS:
            raise UnsupportedError(f"optimize_acqf option {key!r} is not supported by botorch_rs.")
    return out


def _wrap_ic_generator(
    ic_generator: Any,
    acq_function: AcquisitionFunction,
    bounds: Any,
    fixed_features: dict[int, float] | None,
    options: dict[str, Any] | None,
    inequality_constraints: Any,
    equality_constraints: Any,
    ic_gen_kwargs: dict[str, Any],
) -> Any:
    """``generate(q, num_restarts, raw_samples) -> (data, shape)`` calling the
    user's ``ic_generator`` with BoTorch's keyword arguments."""

    def generate(q: int, num_restarts: int, raw_samples: int | None) -> Flat:
        X = ic_generator(
            acq_function=acq_function,
            bounds=bounds,
            q=q,
            num_restarts=num_restarts,
            raw_samples=raw_samples,
            fixed_features=fixed_features,
            options=options or {},
            inequality_constraints=inequality_constraints,
            equality_constraints=equality_constraints,
            **ic_gen_kwargs,
        )
        return to_flat(X, "initial conditions")

    return generate


def optimize_acqf(
    acq_function: AcquisitionFunction,
    bounds: Any,
    q: int,
    num_restarts: int,
    raw_samples: int | None = None,
    options: dict[str, Any] | None = None,
    inequality_constraints: list[tuple[Any, Any, float]] | None = None,
    equality_constraints: list[tuple[Any, Any, float]] | None = None,
    nonlinear_inequality_constraints: list[tuple[Any, bool]] | None = None,
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
    r"""Optimizes ``acq_function`` over the box ``bounds`` (``2 x d``), like
    BoTorch's ``optimize_acqf``.

    ``raw_samples`` scrambled Sobol candidate sets are evaluated and
    ``num_restarts`` of them are selected as starting points (Boltzmann
    sampling, or the non-negative heuristic for non-negative acquisition
    functions). Without parameter constraints, every start is optimized by an
    independent L-BFGS-B run with exact gradients (in parallel), and the best
    result is returned.

    With linear ``inequality_constraints`` / ``equality_constraints``
    (BoTorch's ``(indices, coefficients, rhs)`` tuples, intra- or
    inter-point) or ``nonlinear_inequality_constraints`` (``(callable,
    is_intrapoint)`` with ``callable(x) >= 0`` on torch tensors), the raw
    samples come from a hit-and-run sampler of the feasible polytope and
    chunks of ``batch_limit`` restarts are optimized jointly with SLSQP, as
    in BoTorch's ``gen_candidates_scipy``; infeasible results are projected
    onto the feasible set (``project_to_feasible_space_via_slsqp``) and a
    ``CandidateGenerationError`` is raised if no feasible candidate remains.
    Nonlinear constraints require ``batch_initial_conditions`` or an
    ``ic_generator`` (e.g. :func:`~botorch_rs.optim.initializers.gen_batch_initial_conditions`,
    which runs natively in Rust; any other generator is called with
    BoTorch's keyword arguments).

    Supported ``options``: ``seed`` (raw samples and initial-condition
    selection; BoTorch draws the latter from torch's global RNG),
    ``batch_limit``, ``init_batch_limit``, ``eta``, ``alpha``,
    ``nonnegative``, ``method`` (``"L-BFGS-B"`` or ``"SLSQP"``), ``maxiter``
    (2000), ``ftol``, the L-BFGS-B settings ``maxfun``, ``gtol``, ``maxcor``,
    ``maxls``, and the polytope sampler's ``n_burnin`` (10000) and
    ``n_thinning`` (32).

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
        post_processing_func=post_processing_func,
        gen_candidates=gen_candidates,
        acq_function_sequence=acq_function_sequence,
        timeout_sec=timeout_sec,
        return_full_tree=return_full_tree,
    )
    from botorch_rs.optim import initializers

    native_generator = ic_generator is initializers.gen_batch_initial_conditions
    if ic_gen_kwargs and (ic_generator is None or native_generator):
        raise UnsupportedError(f"Unsupported keyword arguments: {sorted(ic_gen_kwargs)}.")
    if sequential and not return_best_only:
        raise UnsupportedError("`return_best_only=False` only supported for joint optimization.")
    kind = first_kind(bounds, batch_initial_conditions)
    bounds_flat = to_flat(bounds, "bounds")
    if len(bounds_flat[1]) != 2 or bounds_flat[1][0] != 2:
        raise BotorchTensorDimensionError(
            f"bounds should be a `2 x d` tensor, current shape: {list(bounds_flat[1])}."
        )
    d = bounds_flat[1][1]
    fixed = None
    if fixed_features is not None:
        fixed = {int(k) % d: as_float(v, f"fixed_features[{k}]") for k, v in fixed_features.items()}
    ics = None
    if batch_initial_conditions is not None:
        ics = to_flat(batch_initial_conditions, "batch_initial_conditions")
        shape = ics[1]
        if len(shape) not in (2, 3):
            raise ValueError(
                "batch_initial_conditions must be 2-dimensional or 3-dimensional. "
                f"Its shape is {tuple(shape)}."
            )
        if shape[-1] != d:
            raise ValueError(
                f"batch_initial_conditions.shape[-1] must be {d}. The shape is {tuple(shape)}."
            )
        if len(shape) == 2 and raw_samples is not None:
            warnings.warn(
                "If using a 2-dim `batch_initial_conditions` botorch will "
                "default to old behavior of ignoring `num_restarts` and just "
                "use the given `batch_initial_conditions` by setting "
                "`raw_samples` to None.",
                RuntimeWarning,
                stacklevel=2,
            )
            raw_samples = None
        if (
            len(shape) == 3
            and shape[0] < num_restarts
            and shape[-2] != q
            and raw_samples is not None
        ):
            warnings.warn(
                "If using a 3-dim `batch_initial_conditions` where the "
                "first dimension is less than `num_restarts` and the second "
                "dimension is not equal to `q`, botorch will default to "
                "old behavior of ignoring `num_restarts` and just use the "
                "given `batch_initial_conditions` by setting `raw_samples` "
                "to None.",
                RuntimeWarning,
                stacklevel=2,
            )
            raw_samples = None
    opts = _parse_options(options)
    ineq = linear_constraints_arg(inequality_constraints, "inequality_constraints")
    eq = linear_constraints_arg(equality_constraints, "equality_constraints")
    nonlinear = nonlinear_constraints_arg(nonlinear_inequality_constraints)
    constrained = bool(ineq or eq or nonlinear)
    if opts.get("method") == "L-BFGS-B" and constrained:
        raise RuntimeWarning("Method L-BFGS-B cannot handle constraints.")
    generator = None
    if ic_generator is not None and not native_generator:
        generator = _wrap_ic_generator(
            ic_generator,
            acq_function,
            bounds,
            fixed,
            options,
            inequality_constraints,
            equality_constraints,
            ic_gen_kwargs,
        )
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
        inequality_constraints=ineq,
        equality_constraints=eq,
        nonlinear_inequality_constraints=nonlinear,
        ic_generator=generator,
        native_ic_generator=native_generator,
        **opts,
    )
    for category, message in messages:
        warnings.warn(message, _WARNINGS.get(category, RuntimeWarning), stacklevel=2)
    acq_function._after_evaluation()
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

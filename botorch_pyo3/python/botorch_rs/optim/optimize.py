"""Acquisition-function optimization (``botorch.optim.optimize``)."""

from __future__ import annotations

import warnings
from typing import Any

from botorch_rs import _botorch_rs as _rs
from botorch_rs._arrays import Flat, as_float, first_kind, from_flat, from_pair, to_flat
from botorch_rs.acquisition.acquisition import AcquisitionFunction, check_unsupported
from botorch_rs.exceptions.errors import (
    BotorchTensorDimensionError,
    CandidateGenerationError,
    UnsupportedError,
)
from botorch_rs.exceptions.warnings import BadInitialCandidatesWarning, OptimizationWarning
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
    # The one-shot initial conditions (gen_one_shot_(hv)kg_initial_conditions).
    "num_inner_restarts",
    "raw_inner_samples",
}
_FLOAT_OPTIONS = {"ftol", "gtol", "eta", "alpha", "frac_random"}
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
    if out.get("eta") is not None:
        # BoTorch's one-shot initial conditions read ``eta`` too (default 2).
        out["one_shot_eta"] = out["eta"]
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

    ``acq_function`` is a botorch_rs acquisition function, or (like in
    BoTorch) any callable on torch tensors, e.g. a ``torch.nn.Module`` whose
    ``forward(X)`` maps ``batch x q x d`` candidate sets to ``batch`` values:
    it is evaluated from Rust with gradients from ``torch.autograd`` (see
    :class:`~botorch_rs.acquisition.composite.CallableAcquisitionFunction`;
    :class:`~botorch_rs.acquisition.composite.WeightedLogMeanExp` evaluates
    weighted blends of log-scale acquisition functions natively).

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
    ``maxls``, the polytope sampler's ``n_burnin`` (10000) and
    ``n_thinning`` (32), and the one-shot initial conditions'
    ``frac_random``, ``num_inner_restarts`` and ``raw_inner_samples``.

    One-shot acquisition functions (``qKnowledgeGradient``,
    ``qHypervolumeKnowledgeGradient`` and their multi-fidelity variants) are
    optimized over augmented candidate sets (``get_augmented_q_batch_size``),
    starting by default from ``gen_one_shot_kg_initial_conditions`` /
    ``gen_one_shot_hvkg_initial_conditions``; the candidates are extracted
    (``extract_candidates``) unless ``return_full_tree=True``.

    Returns:
        ``(candidates, acq_value)``: a ``q x d`` candidate set (``num_restarts x
        q x d`` with ``return_best_only=False``) and its acquisition value (a
        scalar; ``q`` values for ``sequential=True``; one per restart with
        ``return_best_only=False``), of the array type of ``bounds``.
    """
    if not isinstance(acq_function, AcquisitionFunction) and not callable(acq_function):
        raise UnsupportedError(
            "optimize_acqf requires a botorch_rs acquisition function or a callable on torch "
            f"tensors (got {type(acq_function).__name__})."
        )
    check_unsupported(
        post_processing_func=post_processing_func,
        gen_candidates=gen_candidates,
        acq_function_sequence=acq_function_sequence,
        timeout_sec=timeout_sec,
    )
    from botorch_rs.acquisition.acquisition import OneShotAcquisitionFunction
    from botorch_rs.optim import initializers

    one_shot = isinstance(acq_function, OneShotAcquisitionFunction)
    if one_shot and ic_generator in (
        initializers.gen_one_shot_hvkg_initial_conditions,
        initializers.gen_one_shot_kg_initial_conditions,
    ):
        # The defaults of one-shot acquisition functions, generated natively.
        ic_generator = None
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
    user_acq_function = acq_function
    if not isinstance(acq_function, AcquisitionFunction):
        # A torch callable (e.g. a ``torch.nn.Module`` blending acquisition
        # functions): values from Python, gradients from torch.autograd.
        from botorch_rs.acquisition.composite import CallableAcquisitionFunction

        acq_function = CallableAcquisitionFunction(acq_function, d)
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
            user_acq_function,
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
    full_tree = return_full_tree and one_shot
    if return_best_only:
        if full_tree:
            X = _best_full_tree(candidates, acq_values[0], restart_x, restart_values, kind)
        else:
            X = from_pair(candidates, kind)
        if sequential:
            values = from_flat(acq_values, (len(acq_values),), kind)
        else:
            values = from_flat(acq_values[:1], (), kind)
    else:
        X = from_pair(restart_x, kind)
        if one_shot and not full_tree:
            X = acq_function.extract_candidates(X)  # type: ignore[attr-defined]
        values = from_flat(restart_values, (len(restart_values),), kind)
    return X, (values if return_acq_values else None)


def _best_full_tree(
    candidates: Any, value: float, restart_x: Any, restart_values: list[float], kind: Any
) -> Any:
    """The augmented candidate set of the best restart: the first restart
    with the returned value whose leading points are the returned
    candidates."""
    data, shape = restart_x
    per = shape[-2] * shape[-1]
    n = len(candidates[0])
    best = None
    for i, v in enumerate(restart_values):
        block = data[i * per : (i + 1) * per]
        if v == value and list(block[:n]) == list(candidates[0]):
            best = i
            break
    if best is None:
        finite = [(v, -i) for i, v in enumerate(restart_values) if v == v]
        best = -max(finite)[1] if finite else 0
    return from_flat(data[best * per : (best + 1) * per], tuple(shape[1:]), kind)


def _concat_rows(parts: list[Any], kind: Any) -> Any:
    """Arrays of shape ``k_i x d`` stacked to ``(sum k_i) x d`` (array type
    ``kind``)."""
    data: list[float] = []
    rows, d = 0, 0
    for p in parts:
        flat, shape = to_flat(p, "X")
        data.extend(flat)
        rows += shape[0]
        d = shape[1]
    return from_flat(data, (rows, d), kind)


def _optimize_acqf_mixed_q1(
    acq_function: AcquisitionFunction,
    bounds: Any,
    num_restarts: int,
    raw_samples: int | None,
    fixed_features_list: list[dict[int, Any]],
    options: dict[str, Any] | None,
    inequality_constraints: Any,
    equality_constraints: Any,
    nonlinear_inequality_constraints: Any,
    ic_generator: Any,
    ic_gen_kwargs: dict[str, Any],
    retry_on_optimization_warning: bool,
) -> tuple[Any, Any]:
    """BoTorch's ``optimize_acqf_mixed`` for ``q = 1``: the best restart over
    all ``fixed_features_list`` (fixed features for which candidate
    generation fails are skipped)."""
    candidates, values = [], []
    failures = 0
    for fixed_features in fixed_features_list:
        try:
            X, v = optimize_acqf(
                acq_function=acq_function,
                bounds=bounds,
                q=1,
                num_restarts=num_restarts,
                raw_samples=raw_samples,
                options=options or {},
                inequality_constraints=inequality_constraints,
                equality_constraints=equality_constraints,
                nonlinear_inequality_constraints=nonlinear_inequality_constraints,
                fixed_features=fixed_features,
                ic_generator=ic_generator,
                return_best_only=False,
                retry_on_optimization_warning=retry_on_optimization_warning,
                return_acq_values=True,
                **ic_gen_kwargs,
            )
        except CandidateGenerationError:
            failures += 1
            continue
        candidates.append(X)
        values.append(v)
    if not candidates:
        raise CandidateGenerationError("Candidate generation failed for all `fixed_features`.")
    if failures > 0:
        warnings.warn(
            f"Candidate generation failed for {failures} "
            "combinations of `fixed_features`. To suppress this warning, make "
            "sure all equality/inequality constraints can be satisfied by all "
            "`fixed_features` in `fixed_features_list`.",
            OptimizationWarning,
            stacklevel=3,
        )
    # torch.max(dim=-1) per fixed features, then torch.argmax over them
    # (the first maximum on ties).
    best_batch, best_idx, best_value = 0, 0, float("-inf")
    for b, v in enumerate(values):
        flat, _ = to_flat(v, "acquisition values")
        top = max(range(len(flat)), key=lambda i: (flat[i], -i))
        if b == 0 or flat[top] > best_value:
            best_batch, best_idx, best_value = b, top, flat[top]
    kind = first_kind(bounds)
    flat_x, shape = to_flat(candidates[best_batch], "X")
    d = shape[-1]
    candidate = from_flat(flat_x[best_idx * d : (best_idx + 1) * d], (1, d), kind)
    return candidate, from_flat([best_value], (), kind)


def optimize_acqf_list(
    acq_function_list: list[AcquisitionFunction],
    bounds: Any,
    num_restarts: int,
    raw_samples: int | None = None,
    options: dict[str, Any] | None = None,
    inequality_constraints: list[tuple[Any, Any, float]] | None = None,
    equality_constraints: list[tuple[Any, Any, float]] | None = None,
    nonlinear_inequality_constraints: list[tuple[Any, bool]] | None = None,
    fixed_features: dict[int, Any] | None = None,
    fixed_features_list: list[dict[int, Any]] | None = None,
    post_processing_func: Any = None,
    ic_generator: Any = None,
    ic_gen_kwargs: dict[str, Any] | None = None,
    return_acq_values: bool = True,
    retry_on_optimization_warning: bool = True,
) -> tuple[Any, Any]:
    r"""Generate a list of candidates from a list of acquisition functions
    (BoTorch's ``optimize_acqf_list``).

    The acquisition functions are optimized in sequence for ``q = 1`` with
    :func:`optimize_acqf`, each with the previous candidates (appended to the
    first acquisition function's pending points) set as ``X_pending``; this
    is also known as sequential greedy optimization. With
    ``fixed_features_list``, every step is BoTorch's ``optimize_acqf_mixed``
    for ``q = 1``: the best restart over all fixed features.

    Returns:
        ``(candidates, acq_values)``: a ``q x d`` array of candidates (one per
        acquisition function) and their ``q`` acquisition values, each
        conditional on the previous candidates (``None`` with
        ``return_acq_values=False``).
    """
    if fixed_features and fixed_features_list:
        raise ValueError(
            "\u00c8ither `fixed_feature` or `fixed_features_list` can be provided, not both."
        )
    if not acq_function_list:
        raise ValueError("acq_function_list must be non-empty.")
    check_unsupported(post_processing_func=post_processing_func)
    kind = first_kind(bounds)
    candidate_list: list[Any] = []
    acq_value_list: list[float] = []
    candidates: Any = None
    base_X_pending = acq_function_list[0].X_pending
    for acq_function in acq_function_list:
        if candidate_list:
            acq_function.set_X_pending(
                _concat_rows([base_X_pending, candidates], kind)
                if base_X_pending is not None
                else candidates
            )
        if fixed_features_list:
            candidate, acq_value = _optimize_acqf_mixed_q1(
                acq_function,
                bounds,
                num_restarts,
                raw_samples,
                fixed_features_list,
                options,
                inequality_constraints,
                equality_constraints,
                nonlinear_inequality_constraints,
                ic_generator,
                ic_gen_kwargs or {},
                retry_on_optimization_warning,
            )
        else:
            ic_gen_kwargs = ic_gen_kwargs or {}
            candidate, acq_value = optimize_acqf(
                acq_function=acq_function,
                bounds=bounds,
                q=1,
                num_restarts=num_restarts,
                raw_samples=raw_samples,
                options=options or {},
                inequality_constraints=inequality_constraints,
                equality_constraints=equality_constraints,
                nonlinear_inequality_constraints=nonlinear_inequality_constraints,
                fixed_features=fixed_features,
                return_best_only=True,
                sequential=False,
                ic_generator=ic_generator,
                retry_on_optimization_warning=retry_on_optimization_warning,
                **ic_gen_kwargs,
            )
        candidate_list.append(candidate)
        if return_acq_values:
            acq_value_list.append(float(to_flat(acq_value, "acq_value")[0][0]))
        candidates = _concat_rows(candidate_list, kind)
    if not return_acq_values:
        return candidates, None
    return candidates, from_flat(acq_value_list, (len(acq_value_list),), kind)


__all__ = ["optimize_acqf", "optimize_acqf_list"]

"""Candidate optimization against device-resident acquisition graphs.

Torch optimizers keep candidate search on the device. SciPy drivers transfer
only candidate vectors, acquisition scalars and their gradients at the solver
boundary; model data, posterior algebra and differentiation stay on the device.
"""

from __future__ import annotations

import functools
import time
import warnings
from collections.abc import Mapping
from typing import Any

import torch
from torch import Tensor

from botorch_rs.exceptions.errors import CandidateGenerationError, UnsupportedError
from botorch_rs.exceptions.warnings import OptimizationWarning


def _limit(value: Any, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ValueError(f"{name} must be a positive integer.")
    return value


def _generate_chunks(
    generator: Any, arguments: dict[str, Any], limit: int
) -> tuple[Tensor, Tensor]:
    """Independent groups of restarts, preserving fixed values and timeout budget."""
    limit = _limit(limit, "Candidate batch/aggregation limit")
    initial = arguments["initial_conditions"]
    if initial.ndim < 3 or initial.shape[0] <= limit:
        return generator(**arguments)
    fixed = _fixed_values(
        initial,
        arguments.get("fixed_features"),
        arguments.get("lower_bounds"),
        arguments.get("upper_bounds"),
    )
    candidates, values = [], []
    count = initial.shape[0]
    started = time.monotonic()
    timeout = arguments.get("timeout_sec")
    for offset in range(0, count, limit):
        part = dict(
            arguments,
            initial_conditions=initial[offset : offset + limit],
            fixed_features={k: v[offset : offset + limit] for k, v in fixed.items()} or None,
        )
        if timeout is not None:
            chunks_left = (count - offset + limit - 1) // limit
            part["timeout_sec"] = max(0.0, timeout - (time.monotonic() - started)) / chunks_left
        for key in ("lower_bounds", "upper_bounds"):
            bound = part.get(key)
            if isinstance(bound, Tensor) and bound.ndim >= initial.ndim - 1:
                part[key] = bound.expand_as(initial)[offset : offset + limit]
        candidate, value = generator(**part)
        candidates.append(candidate)
        values.append(value)
    return torch.cat(candidates, 0), torch.cat(values, 0)


def select_best(candidates: Tensor, values: Tensor) -> tuple[Tensor, Tensor]:
    if values.shape != candidates.shape[:-2]:
        raise ValueError("Candidate values must match their restart and model batch dimensions.")
    if values.ndim == 0:
        return candidates, values
    best = values.argmax(dim=0, keepdim=True)
    return candidates.gather(
        0, best[..., None, None].expand(*best.shape, *candidates.shape[-2:])
    ).squeeze(0), values.gather(0, best).squeeze(0)


def _join_pending(original: Any, points: list[Tensor]) -> Tensor:
    tensors = ([original] if original is not None else []) + points
    batch = torch.broadcast_shapes(*(t.shape[:-2] for t in tensors))
    return torch.cat([t.expand(*batch, *t.shape[-2:]) for t in tensors], -2)


def fix_features(X: Tensor, fixed: Mapping[int, Any] | None, *, insert: bool = False) -> Tensor:
    if not fixed:
        return X
    d = X.shape[-1] + (len(fixed) if insert else 0)
    normalized = {int(k) % d: value for k, value in fixed.items()}
    columns, free = [], 0
    for index in range(d):
        if index in normalized:
            value = normalized[index]
            if value is None:
                if insert:
                    raise ValueError("An inserted fixed feature requires an explicit value.")
                column = X[..., index].detach()
            else:
                column = torch.as_tensor(value, dtype=X.dtype, device=X.device)
                if column.ndim == X.ndim - 2:
                    column = column.unsqueeze(-1)
                column = column.expand(X.shape[:-1])
        else:
            column = X[..., free if insert else index]
            free += 1
        columns.append(column)
    return torch.stack(columns, dim=-1)


def clamp(X: Tensor, lower: Any, upper: Any) -> Tensor:
    lo = None if lower is None else torch.as_tensor(lower, dtype=X.dtype, device=X.device)
    hi = None if upper is None else torch.as_tensor(upper, dtype=X.dtype, device=X.device)
    if lo is not None and hi is not None and bool((lo > hi).any()):
        raise ValueError("Lower bounds cannot exceed upper bounds.")
    return X.clamp(lo, hi) if lo is not None or hi is not None else X


def _fixed_values(
    X: Tensor, fixed: Mapping[int, Any] | None, lower: Any, upper: Any
) -> dict[int, Tensor]:
    """Resolve per-restart fixed values before reducing optimization variables."""
    result: dict[int, Tensor] = {}
    d = X.shape[-1]
    for index, value in (fixed or {}).items():
        if not isinstance(index, int) or not -d <= index < d:
            raise ValueError("Fixed feature indices must lie within the input dimension.")
        index %= d
        if index in result:
            raise ValueError("Fixed feature indices must be unique.")
        column = (
            X[..., index].detach()
            if value is None
            else torch.as_tensor(value, dtype=X.dtype, device=X.device)
        )
        if column.ndim == X.ndim - 2:
            column = column.unsqueeze(-1)
        column = column.expand(X.shape[:-1])
        if not bool(torch.isfinite(column).all()):
            raise ValueError("Fixed feature values must be finite.")
        for bound, is_lower in ((lower, True), (upper, False)):
            if bound is not None:
                limit = torch.as_tensor(bound, dtype=X.dtype, device=X.device).expand_as(X)[
                    ..., index
                ]
                if bool((column < limit if is_lower else column > limit).any()):
                    raise ValueError("Fixed feature values must lie inside their bounds.")
        result[index] = column
    return result


def gen_torch(
    initial_conditions: Tensor,
    acquisition_function: Any,
    lower_bounds: Any = None,
    upper_bounds: Any = None,
    optimizer: Any = None,
    options: dict[str, Any] | None = None,
    callback: Any = None,
    fixed_features: Mapping[int, Any] | None = None,
    timeout_sec: float | None = None,
) -> tuple[Tensor, Tensor]:
    from botorch_rs.optim.stopping import ExpMAStoppingCriterion

    options = dict(options or {})
    optimizer_options = dict(options.pop("optimizer_options", {}) or {})
    stopping = dict(options.pop("stopping_criterion_options", {}) or {})
    if "maxiter" in options:
        warnings.warn(
            "Use stopping_criterion_options['maxiter'] instead of maxiter.",
            DeprecationWarning,
            stacklevel=3,
        )
        stopping.setdefault("maxiter", options.pop("maxiter"))
    if "max_optimization_problem_aggregation_size" in options:
        limit = options.pop("max_optimization_problem_aggregation_size")
        options.update(optimizer_options=optimizer_options, stopping_criterion_options=stopping)
        return _generate_chunks(
            gen_torch,
            dict(
                initial_conditions=initial_conditions,
                acquisition_function=acquisition_function,
                lower_bounds=lower_bounds,
                upper_bounds=upper_bounds,
                optimizer=optimizer,
                options=options,
                callback=callback,
                fixed_features=fixed_features,
                timeout_sec=timeout_sec,
            ),
            limit,
        )
    if options:
        raise UnsupportedError(f"Unsupported Torch candidate options: {sorted(options)}.")
    criterion = ExpMAStoppingCriterion(**stopping)
    weights = torch.as_tensor(
        criterion.weights, dtype=initial_conditions.dtype, device=initial_conditions.device
    )
    history: list[Tensor] = []
    d = initial_conditions.shape[-1]
    start = clamp(initial_conditions.detach(), lower_bounds, upper_bounds)
    fixed = _fixed_values(start, fixed_features, lower_bounds, upper_bounds)
    free = [i for i in range(d) if i not in fixed]
    if not free:
        final = fix_features(start, fixed)
        with torch.no_grad():
            return final, acquisition_function(final)
    parameter = start[..., free].clone().requires_grad_(True)
    optimizer_options.setdefault("lr", 0.025)
    opt = (torch.optim.Adam if optimizer is None else optimizer)([parameter], **optimizer_options)

    def candidates() -> Tensor:
        return clamp(fix_features(parameter, fixed, insert=True), lower_bounds, upper_bounds)

    started = time.monotonic()

    def evaluate() -> Tensor:
        opt.zero_grad(set_to_none=True)
        if isinstance(opt, torch.optim.LBFGS):
            # Line-search trials may leave the box. Project the state before
            # evaluating its physical query gradient: differentiating clamp
            # outside the box would report a false zero gradient at a point
            # from which moving inward still improves the acquisition.
            with torch.no_grad():
                parameter.copy_(candidates()[..., free])
            loss = -acquisition_function(candidates()).sum()
            gradient = torch.autograd.grad(loss, parameter)[0]
            # The projected gradient is zero for outward motion at an active
            # bound, the correct box-constrained first-order condition.
            for bound, lower in ((lower_bounds, True), (upper_bounds, False)):
                if bound is not None:
                    limit = torch.as_tensor(
                        bound, dtype=start.dtype, device=start.device
                    ).expand_as(start)[..., free]
                    outward = (
                        ((parameter <= limit) & (gradient > 0))
                        if lower
                        else ((parameter >= limit) & (gradient < 0))
                    )
                    gradient = gradient.masked_fill(outward, 0)
            parameter.grad = gradient
        else:
            with torch.no_grad():
                evaluated = candidates().requires_grad_(True)
            loss = -acquisition_function(evaluated).sum()
            parameter.grad = torch.autograd.grad(loss, evaluated)[0][..., free]
        return loss

    for iteration in range(1, criterion.maxiter + 1):
        loss = evaluate()
        if callback is not None:
            callback(iteration, loss, parameter.grad)
        if isinstance(opt, torch.optim.LBFGS):
            opt.step(evaluate)
        else:
            opt.step()
        history.append(loss.detach())
        if len(history) > criterion.n_window + 1:
            history.pop(0)
        if len(history) == criterion.n_window + 1:
            stacked = torch.stack(history)
            previous = (stacked[:-1] * weights).sum()
            current = (stacked[1:] * weights).sum()
            relative = (previous - current) / previous.abs()
            if not criterion.minimize:
                relative = -relative
            if bool(relative < criterion.rel_tol):
                break
        if timeout_sec is not None and time.monotonic() - started >= timeout_sec:
            break
    with torch.no_grad():
        final = candidates()
        values = acquisition_function(final)
    return final, values


def _constraints(
    X: Tensor, inequalities: Any, equalities: Any, nonlinear: Any
) -> tuple[Tensor, Tensor]:
    groups: list[list[Tensor]] = [[], []]
    for group, specifications in zip(groups, (inequalities, equalities), strict=True):
        for indices, coefficients, rhs in specifications or []:
            index = torch.as_tensor(indices, dtype=torch.long, device=X.device)
            coefficient = torch.as_tensor(coefficients, dtype=X.dtype, device=X.device)
            if index.ndim == 1:
                value = (X[..., index] * coefficient).sum(-1) - rhs
            elif index.ndim == 2 and index.shape[-1] == 2:
                value = (X[..., index[:, 0], index[:, 1]] * coefficient).sum(-1) - rhs
            else:
                raise ValueError("Constraint indices must have shape k or k x 2.")
            group.append(value.reshape(-1))
    for function, intra_point in nonlinear or []:
        points = X.reshape(-1, X.shape[-1]) if intra_point else X.reshape(-1, *X.shape[-2:])
        groups[0].extend(function(point).reshape(-1) for point in points)
    return tuple(torch.cat(group) if group else X.new_empty(0) for group in groups)  # type: ignore[return-value]


def gen_scipy(
    initial_conditions: Tensor,
    acquisition_function: Any,
    lower_bounds: Any = None,
    upper_bounds: Any = None,
    inequality_constraints: Any = None,
    equality_constraints: Any = None,
    nonlinear_inequality_constraints: Any = None,
    options: dict[str, Any] | None = None,
    fixed_features: Mapping[int, Any] | None = None,
    timeout_sec: float | None = None,
    use_parallel_mode: bool | None = None,
) -> tuple[Tensor, Tensor]:
    import numpy as np
    from scipy.optimize import minimize

    shape = initial_conditions.shape
    if len(shape) < 2:
        raise ValueError("initial_conditions must have at least two dimensions.")
    initial = clamp(initial_conditions.detach(), lower_bounds, upper_bounds)
    fixed = _fixed_values(initial, fixed_features, lower_bounds, upper_bounds)
    initial = fix_features(initial, fixed)
    lo = (
        torch.full_like(initial, -torch.inf)
        if lower_bounds is None
        else torch.as_tensor(lower_bounds, dtype=initial.dtype, device=initial.device).expand_as(
            initial
        )
    )
    hi = (
        torch.full_like(initial, torch.inf)
        if upper_bounds is None
        else torch.as_tensor(upper_bounds, dtype=initial.dtype, device=initial.device).expand_as(
            initial
        )
    )
    lo, hi = lo.clone(), hi.clone()
    for index, value in fixed.items():
        lo[..., index] = hi[..., index] = value
    options = dict(options or {})
    constrained = bool(
        inequality_constraints or equality_constraints or nonlinear_inequality_constraints
    )
    method = options.pop("method", "SLSQP" if constrained else "L-BFGS-B")
    with_grad = bool(options.pop("with_grad", True))
    callback = options.pop("callback", None)
    if "max_optimization_problem_aggregation_size" in options:
        limit = options.pop("max_optimization_problem_aggregation_size")
        options.update(method=method, callback=callback, with_grad=with_grad)
        return _generate_chunks(
            gen_scipy,
            dict(
                initial_conditions=initial_conditions,
                acquisition_function=acquisition_function,
                lower_bounds=lower_bounds,
                upper_bounds=upper_bounds,
                inequality_constraints=inequality_constraints,
                equality_constraints=equality_constraints,
                nonlinear_inequality_constraints=nonlinear_inequality_constraints,
                options=options,
                fixed_features=fixed_features,
                timeout_sec=timeout_sec,
                use_parallel_mode=use_parallel_mode,
            ),
            limit,
        )
    options.pop("disp", None)
    options.pop("iprint", None)
    for source, target in (("max_cor", "maxcor"), ("pgtol", "gtol"), ("tol", "gtol")):
        if source in options:
            options[target] = options.pop(source)
    if "factr" in options:
        options.setdefault("ftol", options.pop("factr") * np.finfo(float).eps)
    options.setdefault("maxiter", 2000)
    from botorch_rs.optim._torch_acq_parallel import eligible, gen_parallel

    parallel = eligible(
        initial, lo, hi, fixed, method, with_grad, constrained, timeout_sec, options
    )
    if use_parallel_mode is True and not parallel:
        raise NotImplementedError(
            "Parallel L-BFGS-B requires two/three-dimensional starts, analytic gradients, "
            "shared free-variable bounds, no constraints/timeout, and compatible options."
        )
    if parallel and use_parallel_mode is not False:
        return gen_parallel(initial, acquisition_function, lo, hi, fixed, options, callback)
    started = time.monotonic()
    last = initial

    def variable(vector: Any) -> Tensor:
        return (
            torch.as_tensor(vector, dtype=initial.dtype, device=initial.device)
            .reshape(shape)
            .requires_grad_(True)
        )

    def objective(vector: Any) -> Any:
        nonlocal last
        X = variable(vector)
        last = X.detach()
        if with_grad:
            loss = -acquisition_function(X).sum()
            gradient = torch.autograd.grad(loss, X)[0]
            return loss.detach().double().item(), gradient.detach().double().cpu().numpy().ravel()
        with torch.no_grad():
            return -acquisition_function(X).sum().double().item()

    constraints = []
    for index, specs in enumerate(
        (inequality_constraints or nonlinear_inequality_constraints, equality_constraints)
    ):
        if not specs:
            continue

        def constraint(vector: Any, index: int = index) -> Any:
            X = variable(vector)
            return (
                _constraints(
                    X,
                    inequality_constraints,
                    equality_constraints,
                    nonlinear_inequality_constraints,
                )[index]
                .detach()
                .double()
                .cpu()
                .numpy()
            )

        def jacobian(vector: Any, index: int = index) -> Any:
            X = variable(vector)

            def function(v: Tensor) -> Tensor:
                return _constraints(
                    v,
                    inequality_constraints,
                    equality_constraints,
                    nonlinear_inequality_constraints,
                )[index]

            jac = torch.autograd.functional.jacobian(function, X)
            return jac.reshape(jac.shape[0], -1).detach().double().cpu().numpy()

        constraints.append(
            {"type": "ineq" if index == 0 else "eq", "fun": constraint, "jac": jacobian}
        )

    class Timeout(Exception):
        pass

    def step(*args: Any, **kwargs: Any) -> None:
        if callback is not None:
            callback(*args, **kwargs)
        if timeout_sec is not None and time.monotonic() - started >= timeout_sec:
            raise Timeout

    if callback is not None:
        step = functools.wraps(callback)(step)

    try:
        result = minimize(
            objective,
            initial.double().cpu().numpy().ravel(),
            jac=with_grad,
            bounds=list(
                zip(
                    lo.double().cpu().numpy().ravel(),
                    hi.double().cpu().numpy().ravel(),
                    strict=True,
                )
            ),
            method=method,
            constraints=constraints,
            callback=step,
            options=options,
        )
        final = torch.as_tensor(result.x, dtype=initial.dtype, device=initial.device).reshape(shape)
        if not result.success:
            warnings.warn(
                f"SciPy candidate generation: {result.message}", OptimizationWarning, stacklevel=3
            )
    except Timeout:
        final = last
    final = clamp(fix_features(final, fixed), lower_bounds, upper_bounds)
    if constrained:
        from botorch_rs.optim.parameter_constraints import get_constraint_tolerance

        tolerance = get_constraint_tolerance(final.dtype)
        inequalities, equalities = _constraints(
            final, inequality_constraints, equality_constraints, nonlinear_inequality_constraints
        )
        if bool((inequalities < -tolerance).any()) or bool((equalities.abs() > tolerance).any()):
            raise CandidateGenerationError("SciPy did not produce feasible candidates.")
    with torch.no_grad():
        return final.detach(), acquisition_function(final)


def optimize(arguments: Mapping[str, Any]) -> tuple[Tensor, Tensor | None]:
    from botorch_rs.acquisition.acquisition import OneShotAcquisitionFunction
    from botorch_rs.generation.gen import gen_candidates_scipy

    args = dict(arguments)
    sequence = args["acq_function_sequence"]
    acquisition = args["acq_function"] if sequence is None else sequence[0] if sequence else None
    if acquisition is None:
        raise ValueError("An acquisition function or nonempty sequence is required.")
    bounds = args["bounds"]
    if not isinstance(bounds, Tensor) or bounds.ndim != 2 or bounds.shape[0] != 2:
        raise ValueError("Torch acquisition bounds must be a 2 x d Tensor.")
    if not bool(torch.isfinite(bounds).all()) or bool((bounds[0] > bounds[1]).any()):
        raise ValueError("Acquisition bounds must be finite and ordered.")
    q, restarts = args["q"], args["num_restarts"]
    if q < 1 or restarts < 1:
        raise ValueError("q and num_restarts must be positive.")
    if sequence is not None:
        if not args["sequential"] or q < 2 or len(sequence) != q:
            raise ValueError(
                "An acquisition sequence requires sequential=True, q>1 and q acquisitions."
            )
        if any(getattr(acq, "_tensor_backend", None) != "torch" for acq in sequence):
            raise UnsupportedError("All sequence acquisitions must use the Tensor backend.")
    if args["ic_generator"] is None and args["ic_gen_kwargs"]:
        raise UnsupportedError(
            "Additional initialization keywords require a custom ic_generator on the Torch backend."
        )
    if args["sequential"] and q > 1:
        if any(
            torch.as_tensor(indices).ndim != 1
            for specifications in (args["inequality_constraints"], args["equality_constraints"])
            for indices, _, _ in specifications or []
        ):
            raise UnsupportedError(
                "Sequential Torch optimization does not support inter-point linear constraints."
            )
        if any(
            not intra_point for _, intra_point in args["nonlinear_inequality_constraints"] or []
        ):
            raise UnsupportedError(
                "Sequential Torch optimization does not support inter-point nonlinear constraints."
            )
        if not args["return_best_only"] or args["batch_initial_conditions"] is not None:
            raise UnsupportedError(
                "Sequential optimization requires return_best_only and generated initial conditions."
            )
        acquisitions = [acquisition] * q if sequence is None else sequence
        if any(isinstance(acq, OneShotAcquisitionFunction) for acq in acquisitions):
            raise UnsupportedError("One-shot acquisitions require joint optimization.")
        originals = [getattr(acq, "X_pending", None) for acq in acquisitions]
        points: list[Tensor] = []
        sequential_values: list[Tensor] = []
        try:
            for i, acquisition in enumerate(acquisitions):
                if points:
                    acquisition.set_X_pending(_join_pending(originals[i], points))
                timeout = args["timeout_sec"]
                candidate, value = optimize(
                    dict(
                        args,
                        acq_function=acquisition,
                        acq_function_sequence=None,
                        q=1,
                        sequential=False,
                        timeout_sec=None if timeout is None else timeout / q,
                    )
                )
                points.append(candidate)
                if value is not None:
                    sequential_values.append(value)
        finally:
            for acquisition, original in zip(acquisitions, originals, strict=True):
                acquisition.set_X_pending(original)
        return torch.cat(points, dim=-2), torch.stack(sequential_values) if args[
            "return_acq_values"
        ] else None
    one_shot = isinstance(acquisition, OneShotAcquisitionFunction)
    augmented = acquisition.get_augmented_q_batch_size(q) if one_shot else q
    options = dict(args["options"] or {})
    batch_limit = options.get("batch_limit")
    if batch_limit is not None:
        _limit(batch_limit, "batch_limit")
    from botorch_rs.optim.optimize import _INIT_KEYS

    init_keys = {
        "seed",
        "batch_limit",
        "init_batch_limit",
        "topn",
        "nonnegative",
        "eta",
        "alpha",
        "largest",
        "sorted",
        "n_burnin",
        "n_thinning",
        "sample_around_best",
        "sample_around_best_sigma",
        "sample_around_best_subset_sigma",
        "sample_around_best_prob_perturb",
    }
    from botorch_rs.acquisition.knowledge_gradient import qKnowledgeGradient
    from botorch_rs.acquisition.multi_objective.hypervolume_knowledge_gradient import (
        qHypervolumeKnowledgeGradient,
    )

    specialized = isinstance(acquisition, (qKnowledgeGradient, qHypervolumeKnowledgeGradient))
    if specialized:
        init_keys |= {"frac_random", "num_inner_restarts", "raw_inner_samples"}
    if args["ic_generator"] is None:
        unsupported = (options.keys() & _INIT_KEYS) - init_keys
        if unsupported:
            raise UnsupportedError(
                f"Unsupported Torch initialization options: {sorted(unsupported)}."
            )
    initial = args["batch_initial_conditions"]
    if initial is None and args["ic_generator"] is not None:
        initial = args["ic_generator"](
            acq_function=acquisition,
            bounds=bounds,
            q=q,
            num_restarts=restarts,
            raw_samples=args["raw_samples"],
            fixed_features=args["fixed_features"],
            options=options,
            inequality_constraints=args["inequality_constraints"],
            equality_constraints=args["equality_constraints"],
            **args["ic_gen_kwargs"],
        )
    elif initial is None:
        if args["nonlinear_inequality_constraints"]:
            raise UnsupportedError(
                "Nonlinear constrained Torch optimization requires feasible batch_initial_conditions or an ic_generator."
            )
        raw = args["raw_samples"]
        if raw is None or raw < restarts:
            raise ValueError(
                "raw_samples must be at least num_restarts when no initial conditions are given."
            )
        if specialized:
            from botorch_rs.optim.initializers import (
                gen_one_shot_hvkg_initial_conditions,
                gen_one_shot_kg_initial_conditions,
            )

            initialize = (
                gen_one_shot_kg_initial_conditions
                if isinstance(acquisition, qKnowledgeGradient)
                else gen_one_shot_hvkg_initial_conditions
            )
            initial = initialize(
                acquisition,
                bounds,
                q,
                restarts,
                raw,
                args["fixed_features"],
                options,
                inequality_constraints=args["inequality_constraints"],
                equality_constraints=args["equality_constraints"],
            )
        else:
            from botorch_rs.optim._torch_acq_initializers import gen_initial_conditions

            initial = gen_initial_conditions(
                acquisition,
                bounds,
                augmented,
                restarts,
                raw,
                args["fixed_features"],
                options,
                inequality_constraints=args["inequality_constraints"],
                equality_constraints=args["equality_constraints"],
            )
    initial = initial.to(bounds)
    from botorch_rs.optim._torch_acq_initializers import acquisition_batch_shape

    model_batch = acquisition_batch_shape(acquisition)
    if initial.ndim == 3 and model_batch:
        initial = initial.reshape(
            initial.shape[0], *((1,) * len(model_batch)), *initial.shape[-2:]
        ).expand(initial.shape[0], *model_batch, *initial.shape[-2:])
    generator = args["gen_candidates"] or gen_candidates_scipy
    generator_base = generator.func if isinstance(generator, functools.partial) else generator
    torch_generator = getattr(generator_base, "__name__", "") == "gen_candidates_torch"
    gen_options = {key: value for key, value in options.items() if key not in _INIT_KEYS}
    generation_args = {
        "initial_conditions": initial,
        "acquisition_function": acquisition,
        "lower_bounds": bounds[0],
        "upper_bounds": bounds[1],
        "fixed_features": args["fixed_features"],
        "options": gen_options,
        "timeout_sec": args["timeout_sec"],
    }
    if not torch_generator:
        generation_args.update(
            inequality_constraints=args["inequality_constraints"],
            equality_constraints=args["equality_constraints"],
            nonlinear_inequality_constraints=args["nonlinear_inequality_constraints"],
        )
    elif (
        args["inequality_constraints"]
        or args["equality_constraints"]
        or args["nonlinear_inequality_constraints"]
    ):
        raise UnsupportedError(
            "Torch optimizers currently support box constraints and fixed features; use SciPy SLSQP for parameter constraints."
        )
    started = time.monotonic()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", OptimizationWarning)
        candidates, values = _generate_chunks(
            generator, generation_args, batch_limit or initial.shape[0]
        )
    failed = any(issubclass(w.category, OptimizationWarning) for w in caught)
    remaining = (
        None if args["timeout_sec"] is None else args["timeout_sec"] - (time.monotonic() - started)
    )
    if (
        failed
        and args["batch_initial_conditions"] is None
        and args.get("retry_on_optimization_warning", True)
        and (remaining is None or remaining > 0)
    ):
        for warning in caught:
            if not issubclass(warning.category, OptimizationWarning):
                warnings.warn(warning.message, warning.category, stacklevel=3)
        retry_options = dict(options)
        if retry_options.get("seed") is not None:
            retry_options["seed"] += 1
        return optimize(
            dict(
                args,
                options=retry_options,
                retry_on_optimization_warning=False,
                timeout_sec=remaining,
            )
        )
    for warning in caught:
        warnings.warn(warning.message, warning.category, stacklevel=3)
    if args["post_processing_func"] is not None:
        from botorch_rs.optim.parameter_constraints import get_constraint_tolerance

        tolerance = get_constraint_tolerance(bounds.dtype)
        candidates = args["post_processing_func"](candidates)
        if (
            not isinstance(candidates, Tensor)
            or candidates.shape != initial.shape
            or candidates.device != bounds.device
            or candidates.dtype != bounds.dtype
        ):
            raise CandidateGenerationError(
                "Post-processing must retain the candidate tensor shape, dtype and device."
            )
        if (
            not bool(torch.isfinite(candidates).all())
            or bool((candidates < bounds[0] - tolerance).any())
            or bool((candidates > bounds[1] + tolerance).any())
        ):
            raise CandidateGenerationError(
                "Post-processing produced nonfinite or out-of-bounds candidates."
            )
        for index, value in _fixed_values(
            clamp(initial, bounds[0], bounds[1]), args["fixed_features"], bounds[0], bounds[1]
        ).items():
            if not torch.allclose(candidates[..., index], value, rtol=0, atol=tolerance):
                raise CandidateGenerationError("Post-processing changed fixed feature values.")
        inequalities, equalities = _constraints(
            candidates,
            args["inequality_constraints"],
            args["equality_constraints"],
            args["nonlinear_inequality_constraints"],
        )
        if (
            not bool(torch.isfinite(inequalities).all() & torch.isfinite(equalities).all())
            or bool((inequalities < -tolerance).any())
            or bool((equalities.abs() > tolerance).any())
        ):
            raise CandidateGenerationError("Post-processing produced infeasible candidates.")
        with torch.no_grad():
            values = acquisition(candidates)
    if args["return_best_only"]:
        candidates, values = select_best(candidates, values)
    if one_shot and not args["return_full_tree"]:
        candidates = acquisition.extract_candidates(candidates)
    return candidates, values if args["return_acq_values"] else None

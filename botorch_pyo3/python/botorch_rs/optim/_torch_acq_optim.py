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
    options.pop("max_optimization_problem_aggregation_size", None)
    if options:
        raise UnsupportedError(f"Unsupported Torch candidate options: {sorted(options)}.")
    criterion = ExpMAStoppingCriterion(**stopping)
    weights = torch.as_tensor(
        criterion.weights, dtype=initial_conditions.dtype, device=initial_conditions.device
    )
    history: list[Tensor] = []
    d = initial_conditions.shape[-1]
    fixed = {} if not fixed_features else {int(i) % d: v for i, v in fixed_features.items()}
    free = [i for i in range(d) if i not in fixed]
    start = clamp(initial_conditions.detach(), lower_bounds, upper_bounds)
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
    for iteration in range(1, criterion.maxiter + 1):
        opt.zero_grad(set_to_none=True)
        with torch.no_grad():
            evaluated = candidates().requires_grad_(True)
        loss = -acquisition_function(evaluated).sum()
        gradient = torch.autograd.grad(loss, evaluated)[0][..., free]
        if callback is not None:
            callback(iteration, loss, gradient)
        parameter.grad = gradient
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
    initial = fix_features(initial, fixed_features)
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
    for index, value in (fixed_features or {}).items():
        if value is not None:
            lo[..., index] = hi[..., index] = torch.as_tensor(
                value, dtype=initial.dtype, device=initial.device
            )
    options = dict(options or {})
    constrained = bool(
        inequality_constraints or equality_constraints or nonlinear_inequality_constraints
    )
    method = options.pop("method", "SLSQP" if constrained else "L-BFGS-B")
    if method not in {"L-BFGS-B", "SLSQP"} or (constrained and method != "SLSQP"):
        raise UnsupportedError(
            "Torch SciPy candidate generation requires L-BFGS-B or constrained SLSQP."
        )
    if not options.pop("with_grad", True):
        raise UnsupportedError("Torch candidate generation requires with_grad=True.")
    callback = options.pop("callback", None)
    options.pop("max_optimization_problem_aggregation_size", None)
    options.pop("disp", None)
    options.pop("iprint", None)
    for source, target in (("max_cor", "maxcor"), ("pgtol", "gtol"), ("tol", "gtol")):
        if source in options:
            options[target] = options.pop(source)
    if "factr" in options:
        options.setdefault("ftol", options.pop("factr") * np.finfo(float).eps)
    options.setdefault("maxiter", 2000)
    started = time.monotonic()
    last = initial

    def variable(vector: Any) -> Tensor:
        return (
            torch.as_tensor(vector, dtype=initial.dtype, device=initial.device)
            .reshape(shape)
            .requires_grad_(True)
        )

    def objective(vector: Any) -> tuple[float, Any]:
        nonlocal last
        X = variable(vector)
        loss = -acquisition_function(X).sum()
        gradient = torch.autograd.grad(loss, X)[0]
        last = X.detach()
        return loss.detach().double().item(), gradient.detach().double().cpu().numpy().ravel()

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

    def step(vector: Any) -> None:
        if callback is not None:
            callback(vector)
        if timeout_sec is not None and time.monotonic() - started >= timeout_sec:
            raise Timeout

    try:
        result = minimize(
            objective,
            initial.double().cpu().numpy().ravel(),
            jac=True,
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
    final = clamp(fix_features(final, fixed_features), lower_bounds, upper_bounds)
    if constrained:
        inequalities, equalities = _constraints(
            final, inequality_constraints, equality_constraints, nonlinear_inequality_constraints
        )
        if bool((inequalities < -1e-6).any()) or bool((equalities.abs() > 1e-6).any()):
            raise CandidateGenerationError("SciPy did not produce feasible candidates.")
    with torch.no_grad():
        return final.detach(), acquisition_function(final)


def optimize(arguments: Mapping[str, Any]) -> tuple[Tensor, Tensor | None]:
    from botorch_rs.acquisition.acquisition import OneShotAcquisitionFunction
    from botorch_rs.generation.gen import gen_candidates_scipy

    args = dict(arguments)
    acquisition = args["acq_function"]
    bounds = args["bounds"]
    if not isinstance(bounds, Tensor) or bounds.ndim != 2 or bounds.shape[0] != 2:
        raise ValueError("Torch acquisition bounds must be a 2 x d Tensor.")
    if not bool(torch.isfinite(bounds).all()) or bool((bounds[0] > bounds[1]).any()):
        raise ValueError("Acquisition bounds must be finite and ordered.")
    q, restarts = args["q"], args["num_restarts"]
    if q < 1 or restarts < 1:
        raise ValueError("q and num_restarts must be positive.")
    if args["acq_function_sequence"] is not None:
        raise UnsupportedError("Torch acquisition sequences are not yet supported.")
    if args["sequential"] and q > 1:
        if not args["return_best_only"] or args["batch_initial_conditions"] is not None:
            raise UnsupportedError(
                "Sequential optimization requires return_best_only and generated initial conditions."
            )
        original = acquisition.X_pending
        points: list[Tensor] = []
        sequential_values: list[Tensor] = []
        try:
            for _ in range(q):
                if points:
                    acquisition.set_X_pending(
                        torch.cat(([original] if original is not None else []) + points, dim=-2)
                    )
                candidate, value = optimize(dict(args, q=1, sequential=False))
                points.append(candidate)
                if value is not None:
                    sequential_values.append(value)
        finally:
            acquisition.set_X_pending(original)
        return torch.cat(points, dim=-2), torch.stack(sequential_values) if args[
            "return_acq_values"
        ] else None
    one_shot = isinstance(acquisition, OneShotAcquisitionFunction)
    augmented = acquisition.get_augmented_q_batch_size(q) if one_shot else q
    options = dict(args["options"] or {})
    initial = args["batch_initial_conditions"]
    if initial is None and args["ic_generator"] is not None:
        initial = args["ic_generator"](
            acq_function=acquisition,
            bounds=bounds,
            q=q,
            num_restarts=restarts,
            raw_samples=args["raw_samples"],
            options=options,
            **args["ic_gen_kwargs"],
        )
    elif initial is None:
        if (
            args["inequality_constraints"]
            or args["equality_constraints"]
            or args["nonlinear_inequality_constraints"]
        ):
            raise UnsupportedError(
                "Constrained Torch optimization currently requires feasible batch_initial_conditions or an ic_generator."
            )
        raw = args["raw_samples"]
        if raw is None or raw < restarts:
            raise ValueError(
                "raw_samples must be at least num_restarts when no initial conditions are given."
            )
        engine = torch.quasirandom.SobolEngine(
            augmented * bounds.shape[-1], scramble=True, seed=options.get("seed")
        )
        unit = (
            engine.draw(raw, dtype=bounds.dtype)
            .to(bounds.device)
            .reshape(raw, augmented, bounds.shape[-1])
        )
        samples = fix_features(bounds[0] + (bounds[1] - bounds[0]) * unit, args["fixed_features"])
        with torch.no_grad():
            values = acquisition(samples)
            if values.shape != (raw,):
                raise UnsupportedError(
                    "Torch automatic initialization requires one scalar per raw candidate batch."
                )
            indices = values.topk(restarts).indices
            initial = samples.index_select(0, indices)
    initial = initial.to(bounds)
    generator = args["gen_candidates"] or gen_candidates_scipy
    generator_base = generator.func if isinstance(generator, functools.partial) else generator
    torch_generator = getattr(generator_base, "__name__", "") == "gen_candidates_torch"
    init_keys = {"seed", "batch_limit", "init_batch_limit", "topn", "nonnegative", "eta", "alpha"}
    gen_options = {key: value for key, value in options.items() if key not in init_keys}
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
    candidates, values = generator(**generation_args)
    if args["post_processing_func"] is not None:
        candidates = args["post_processing_func"](candidates)
        with torch.no_grad():
            values = acquisition(candidates)
    if args["return_best_only"]:
        if values.ndim != 1:
            raise UnsupportedError(
                "Torch best-restart selection currently requires scalar acquisition batches."
            )
        best = values.argmax()
        candidates, values = candidates[best], values[best]
    if one_shot and not args["return_full_tree"]:
        candidates = acquisition.extract_candidates(candidates)
    return candidates, values if args["return_acq_values"] else None

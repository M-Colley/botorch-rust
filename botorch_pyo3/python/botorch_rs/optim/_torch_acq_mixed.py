"""Device mixed ordinal/categorical search with real continuous subproblems."""

from __future__ import annotations

import dataclasses
import functools
import warnings
from collections.abc import Mapping
from typing import Any

import torch
from torch import Tensor

from botorch_rs.exceptions.errors import CandidateGenerationError, UnsupportedError
from botorch_rs.exceptions.warnings import OptimizationWarning
from botorch_rs.optim._torch_acq_discrete import _score
from botorch_rs.optim._torch_acq_initializers import acquisition_batch_shape, initialize_q_batch
from botorch_rs.optim._torch_acq_optim import _fixed_values, _join_pending, _limit, fix_features
from botorch_rs.optim._torch_initialization_sampling import feasible_batches, sample_polytope


def _stream(X: Tensor, seed: int | None) -> Any:
    return None if seed is None else torch.Generator(device=X.device).manual_seed(seed)


def _dims(dims: Any, like: Tensor) -> dict[int, Tensor]:
    return {
        int(index): torch.as_tensor(values, dtype=like.dtype, device=like.device)
        for index, values in (dims or {}).items()
    }


def round_dims(X: Tensor, dims: Any) -> Tensor:
    result = X.clone()
    for index, allowed in _dims(dims, X).items():
        if allowed.ndim != 1 or allowed.numel() == 0:
            raise ValueError("Discrete dimensions require nonempty one-dimensional allowed values.")
        result[..., index] = allowed[(result[..., index, None] - allowed).abs().argmin(-1)]
    return result


def nearest(current: Tensor, bounds: Tensor, dims: Any) -> Tensor:
    dims = _dims(dims, current)
    if not dims:
        return current.new_empty(0, current.numel())
    position = current.clone()
    for index, allowed in dims.items():
        position[index] = (allowed - current[index]).abs().argmin()
    rows = position.expand(2 * len(dims), -1).clone()
    for row, (index, allowed) in enumerate(dims.items()):
        rows[row, index] = (position[index] - 1).clamp_min(0)
        rows[len(dims) + row, index] = (position[index] + 1).clamp_max(allowed.numel() - 1)
    rows = torch.unique(rows, dim=0)
    rows = rows[~(rows == position).all(-1)]
    for index, allowed in dims.items():
        rows[:, index] = allowed[rows[:, index].long()]
    return rows


def categorical(current: Tensor, dims: Any, limit: int = 20, generator: Any = None) -> Tensor:
    rows = []
    for index, allowed in _dims(dims, current).items():
        if allowed.numel() > limit:
            allowed = allowed[allowed != current[index]]
            allowed = allowed[
                torch.randperm(allowed.numel(), device=current.device, generator=generator)[:limit]
            ]
        part = current.expand(allowed.numel(), -1).clone()
        part[:, index] = allowed
        rows.append(part)
    if not rows:
        return current.new_empty(0, current.numel())
    result = torch.unique(torch.cat(rows), dim=0)
    return result[~(result == current).all(-1)]


def spray(
    X: Tensor,
    continuous: Tensor,
    discrete: Any,
    categories: Any,
    bounds: Tensor,
    count: int,
    std: float = 0.1,
    generator: Any = None,
) -> Tensor:
    points = []
    continuous = torch.as_tensor(continuous, dtype=torch.long, device=X.device)
    for point in X:
        sample = point.expand(count, -1).clone()
        for dims, neighborhood in (
            (discrete, nearest(point, bounds, discrete)),
            (categories, categorical(point, categories, generator=generator)),
        ):
            if dims:
                if neighborhood.shape[0]:
                    indices = torch.randint(
                        neighborhood.shape[0], (count,), device=X.device, generator=generator
                    )
                    columns = list(dims)
                    sample[:, columns] = neighborhood[indices][:, columns]
        noise = torch.randn(
            (count, continuous.numel()), dtype=X.dtype, device=X.device, generator=generator
        )
        sample[:, continuous] = (point[continuous] + std * noise).clamp(
            bounds[0, continuous], bounds[1, continuous]
        )
        points.append(sample)
    return torch.cat(points) if points else X.new_empty(0, X.shape[-1])


def _feasible(points: Tensor, inputs: Any) -> Tensor:
    mask = feasible_batches(
        points.unsqueeze(-2),
        inputs.bounds,
        inputs.inequality_constraints,
        inputs.equality_constraints,
    )
    return points[mask]


def feasible_points(
    inputs: Any, discrete: Any, categories: Any, count: int, seed: int | None = None
) -> Tensor:
    bounds = inputs.bounds
    engine = torch.quasirandom.SobolEngine(bounds.shape[-1], scramble=True, seed=seed)
    points = bounds.new_empty(0, bounds.shape[-1])
    constrained = bool(inputs.inequality_constraints or inputs.equality_constraints)
    for attempt in range(10):
        remaining = count - points.shape[0]
        if remaining <= 0:
            break
        if constrained:
            sampled = sample_polytope(
                2 * remaining,
                1,
                bounds,
                seed=None if seed is None else seed + attempt,
                inequality_constraints=inputs.inequality_constraints,
                equality_constraints=inputs.equality_constraints,
                fixed_features=inputs.fixed_features,
            ).squeeze(-2)
        else:
            uniform = engine.draw(2 * remaining, dtype=bounds.dtype).to(bounds.device)
            sampled = bounds[0] + (bounds[1] - bounds[0]) * uniform
        sampled = round_dims(round_dims(sampled, discrete), categories)
        sampled = fix_features(
            sampled, _fixed_values(sampled, inputs.fixed_features, bounds[0], bounds[1])
        )
        points = torch.cat((points, _feasible(sampled, inputs)))
    if points.shape[0] == 0:
        raise CandidateGenerationError(
            "Could not generate any feasible starting points for mixed optimizer."
        )
    return points[:count]


def _continuous_optimize(
    inputs: Any, initial: Tensor | None, fixed: Any, maxiter: int, raw: int | None = None
) -> tuple[Tensor, Tensor]:
    from botorch_rs.generation.gen import gen_candidates_scipy, gen_candidates_torch
    from botorch_rs.optim.optimize import optimize_acqf

    options = inputs.options or {}
    if inputs.inequality_constraints or inputs.equality_constraints:
        generator = gen_candidates_scipy
        solver_options = {"maxiter": maxiter, "ftol": options.get("tol", 1e-8)}
    else:
        generator = functools.partial(gen_candidates_torch, optimizer=torch.optim.LBFGS)
        solver_options = {
            "optimizer_options": {
                "lr": 1.0,
                "max_iter": maxiter,
                "line_search_fn": "strong_wolfe",
                "tolerance_grad": options.get("tol", 1e-8),
                "tolerance_change": 1e-12,
            },
            "stopping_criterion_options": {"maxiter": 1},
        }
    return optimize_acqf(
        inputs.acq_function,
        inputs.bounds,
        1,
        inputs.num_restarts if initial is None else initial.shape[0],
        raw_samples=raw,
        batch_initial_conditions=initial,
        fixed_features=fixed,
        return_best_only=False,
        options={
            "seed": options.get("seed"),
            "max_optimization_problem_aggregation_size": 1,
            "batch_limit": options.get("batch_limit", 2048),
            "init_batch_limit": options.get("init_batch_limit", 2048),
            **solver_options,
        },
        gen_candidates=generator,
        inequality_constraints=inputs.inequality_constraints,
        equality_constraints=inputs.equality_constraints,
        post_processing_func=inputs.post_processing_func if initial is not None else None,
        retry_on_optimization_warning=inputs.retry_on_optimization_warning,
    )


def starting_points(
    inputs: Any, discrete: Any, categories: Any, continuous: Tensor, seed: int | None = None
) -> tuple[Tensor, Tensor]:
    from botorch_rs.optim.utils.acquisition_utils import get_X_baseline

    bounds = inputs.bounds
    options = inputs.options or {}
    binary = [i for i, values in discrete.items() if len(values) == 2]
    integer = len(discrete) - len(binary)
    strategy = options.get(
        "initialization_strategy",
        "equally_spaced" if integer == 0 and len(binary) >= 2 else "continuous_relaxation",
    )
    if strategy not in {"continuous_relaxation", "equally_spaced", "random"}:
        raise UnsupportedError(f"Unsupported initialization strategy: {strategy}.")
    generator = _stream(bounds, seed)
    points = bounds.new_empty(0, bounds.shape[-1])
    if strategy == "continuous_relaxation":
        try:
            points, _ = _continuous_optimize(
                dataclasses.replace(inputs, options={**options, "seed": seed}),
                None,
                inputs.fixed_features,
                options.get("maxiter_init", 100),
                inputs.raw_samples,
            )
            points = points.squeeze(-2)
        except Exception as error:
            warnings.warn(
                f"Failed to initialize using continuous relaxation. Using sample_feasible_points for initialization. Original error message: {error}",
                OptimizationWarning,
                stacklevel=3,
            )
    if points.shape[0] == 0:
        engine = torch.quasirandom.SobolEngine(bounds.shape[-1], scramble=True, seed=seed)
        points = bounds[0] + (bounds[1] - bounds[0]) * engine.draw(
            inputs.raw_samples, dtype=bounds.dtype
        ).to(bounds.device)
    if strategy == "equally_spaced":
        if integer or not binary:
            raise ValueError(
                "Equally spaced initialization requires binary discrete dimensions only."
            )
        counts = torch.randint(
            len(binary), (points.shape[0],), device=bounds.device, generator=generator
        )
        ranks = (
            torch.rand((points.shape[0], len(binary)), device=bounds.device, generator=generator)
            .argsort(-1)
            .argsort(-1)
        )
        points[:, binary] = torch.where(
            ranks < counts[:, None], bounds[1, binary], bounds[0, binary]
        )
    count = options.get("num_spray_points", 20 if integer == 0 else 0)
    if count > 0:
        baseline = get_X_baseline(inputs.acq_function)
        if baseline is not None:
            points = torch.cat(
                (
                    points,
                    spray(
                        baseline,
                        continuous,
                        discrete,
                        categories,
                        bounds,
                        count,
                        options.get("std_cont_perturbation", 0.1),
                        generator,
                    ),
                )
            )
    points = round_dims(round_dims(points, discrete), categories)
    points = fix_features(
        points, _fixed_values(points, inputs.fixed_features, bounds[0], bounds[1])
    )
    points = _feasible(points, inputs)
    if points.shape[0] < inputs.num_restarts:
        points = torch.cat(
            (
                points,
                feasible_points(
                    inputs, discrete, categories, inputs.num_restarts - points.shape[0], seed
                ),
            )
        )
    _, values = _score(inputs.acq_function, points, options.get("init_batch_limit", 2048))
    if points.shape[0] > inputs.num_restarts:
        selected, values = initialize_q_batch(
            points.unsqueeze(-2), values, inputs.num_restarts, generator=generator
        )
        points = selected.squeeze(-2)
    return points, values


def discrete_step(
    inputs: Any, discrete: Any, categories: Any, current: Tensor, seed: int | None = None
) -> tuple[Tensor, Tensor]:
    options = inputs.options or {}
    current = current.detach().clone()
    _, values = _score(inputs.acq_function, current, options.get("init_batch_limit", 2048))
    done = torch.zeros(current.shape[0], dtype=torch.bool, device=current.device)
    generator = _stream(current, seed)
    for _ in range(options.get("maxiter_discrete", 4)):
        neighborhoods = []
        for i in range(current.shape[0]):
            points = torch.cat(
                (
                    nearest(current[i], inputs.bounds, discrete),
                    categorical(current[i], categories, generator=generator),
                )
            )
            points = _feasible(points, inputs) if not bool(done[i]) else points[:0]
            neighborhoods.append(points)
        if sum(points.shape[0] for points in neighborhoods) == 0:
            break
        _, scores = _score(
            inputs.acq_function, torch.cat(neighborhoods), options.get("init_batch_limit", 2048)
        )
        offset = 0
        for i, points in enumerate(neighborhoods):
            if points.shape[0] == 0:
                done[i] = True
                continue
            score, index = scores[offset : offset + points.shape[0]].max(0)
            improvement = score - values[i]
            if bool(improvement > 0):
                current[i], values[i] = points[index], score
            done[i] = improvement <= options.get("tol", 1e-8)
            offset += points.shape[0]
        if bool(done.all()):
            break
    return current, values


def continuous_step(
    inputs: Any, discrete: Any, categories: Any, current: Tensor
) -> tuple[Tensor, Tensor]:
    if inputs.return_best_only:
        raise UnsupportedError("continuous_step does not support return_best_only=True.")
    if isinstance(discrete, Mapping):
        discrete = list(discrete)
    if isinstance(categories, Mapping):
        categories = list(categories)
    # The index tensors remain device tensors; feature number iteration is
    # static Python metadata and never extracts query values.
    noncontinuous = torch.cat(
        (
            torch.as_tensor(discrete, dtype=torch.long, device=current.device),
            torch.as_tensor(categories, dtype=torch.long, device=current.device),
        )
    )
    fixed = dict(inputs.fixed_features or {})
    for index in range(current.shape[-1]):
        if index not in fixed and bool((noncontinuous == index).any()):
            fixed[index] = current[:, index]
    candidates, values = _continuous_optimize(
        inputs, current.unsqueeze(-2), fixed, (inputs.options or {}).get("maxiter_continuous", 8)
    )
    return candidates.squeeze(-2), values


def optimize(arguments: Mapping[str, Any]) -> tuple[Tensor, Any]:
    from botorch_rs.acquisition.acquisition import OneShotAcquisitionFunction
    from botorch_rs.optim import optimize_mixed as public
    from botorch_rs.optim.optimize import OptimizeAcqfInputs

    args = dict(arguments)
    acquisition, bounds = args["acq_function"], args["bounds"]
    if not isinstance(bounds, Tensor) or bounds.ndim != 2 or bounds.shape[0] != 2:
        raise ValueError("Mixed search bounds must be 2 x d Tensor bounds.")
    if not bool(torch.isfinite(bounds).all()) or bool((bounds[0] > bounds[1]).any()):
        raise ValueError("Mixed search bounds must be finite and ordered.")
    for name in ("q", "num_restarts", "raw_samples"):
        _limit(args[name], name)
    if not args["sequential"]:
        raise NotImplementedError("Mixed alternating search supports sequential optimization only.")
    if isinstance(acquisition, OneShotAcquisitionFunction) or acquisition_batch_shape(acquisition):
        raise UnsupportedError(
            "Mixed alternating search requires an unbatched, non-one-shot acquisition, as in BoTorch."
        )
    options = dict(args["options"] or {})
    if set(options) - public.SUPPORTED_OPTIONS:
        raise UnsupportedError(
            f"Received unsupported options: {sorted(set(options) - public.SUPPORTED_OPTIONS)}."
        )
    discrete = _dims(args["discrete_dims"], bounds)
    categories = _dims(args["cat_dims"], bounds)
    keys = [*discrete, *categories]
    if len(set(keys)) != len(keys) or any(not 0 <= i < bounds.shape[-1] for i in keys):
        raise ValueError(
            "Discrete and categorical dimensions must be unique, disjoint and in range."
        )
    for i, values in discrete.items():
        values = values.sort().values
        discrete[i] = values
        if (
            values.numel() == 0
            or bool(values[0] != bounds[0, i])
            or bool(values[-1] != bounds[1, i])
        ):
            raise ValueError("Discrete values must start and end at their physical bounds.")
    for i, values in categories.items():
        categories[i] = values.sort().values
        if values.numel() == 0 or bool(((values < bounds[0, i]) | (values > bounds[1, i])).any()):
            raise ValueError("Categorical values must be nonempty and lie in the bounds.")
    constrained = set()
    for name in ("inequality_constraints", "equality_constraints"):
        for indices, _, _ in args[name] or []:
            indices = torch.as_tensor(indices, device=bounds.device)
            if indices.ndim != 1:
                raise UnsupportedError("Mixed alternating search requires intra-point constraints.")
            for i in range(bounds.shape[-1]):
                if bool((indices == i).any()):
                    constrained.add(i)
                    if name == "equality_constraints" and i in keys:
                        raise ValueError("Equality constraints can only use continuous dimensions.")
    relaxed = {
        i: values
        for i, values in discrete.items()
        if values.numel() > options.get("max_discrete_values", 20) and i not in constrained
    }
    discrete = {i: values for i, values in discrete.items() if i not in relaxed}
    fixed = args["fixed_features"] or {}
    if any(not -bounds.shape[-1] <= index < bounds.shape[-1] for index in fixed):
        raise ValueError("Fixed feature indices are out of range.")
    fixed = {index % bounds.shape[-1]: value for index, value in fixed.items()}
    discrete = {i: values for i, values in discrete.items() if i not in fixed}
    categories = {i: values for i, values in categories.items() if i not in fixed}
    continuous = torch.tensor(
        [
            i
            for i in range(bounds.shape[-1])
            if i not in discrete and i not in categories and i not in fixed
        ],
        device=bounds.device,
        dtype=torch.long,
    )

    def process(X):
        result = round_dims(X, relaxed)
        return (
            result if args["post_processing_func"] is None else args["post_processing_func"](result)
        )

    inputs = OptimizeAcqfInputs(
        acq_function=acquisition,
        bounds=bounds,
        q=1,
        num_restarts=args["num_restarts"],
        raw_samples=args["raw_samples"],
        options=options,
        inequality_constraints=args["inequality_constraints"],
        equality_constraints=args["equality_constraints"],
        nonlinear_inequality_constraints=None,
        fixed_features=fixed,
        post_processing_func=process,
        batch_initial_conditions=None,
        return_best_only=False,
        gen_candidates=None,
        sequential=False,
        return_acq_values=True,
        retry_on_optimization_warning=args["retry_on_optimization_warning"],
    )
    original = getattr(acquisition, "X_pending", None)
    candidates = []
    try:
        for step in range(args["q"]):
            current, values = public.generate_starting_points(
                inputs,
                discrete,
                categories,
                continuous,
                seed=None if args["seed"] is None else args["seed"] + step,
            )
            done = torch.zeros(current.shape[0], device=bounds.device, dtype=torch.bool)
            for _ in range(options.get("maxiter_alternating", 64)):
                previous = values.clone()
                updated, updated_values = public.discrete_step(
                    inputs, discrete, categories, current[~done], seed=args["seed"]
                )
                updated, updated_values = public.continuous_step(
                    inputs,
                    torch.tensor(list(discrete), device=bounds.device),
                    torch.tensor(list(categories), device=bounds.device),
                    updated,
                )
                current[~done], values[~done] = updated, updated_values
                done |= (values - previous) < options.get("tol", 1e-8)
                if bool(done.all()):
                    break
            candidates.append(current[values.argmax()].unsqueeze(-2))
            if args["q"] > 1:
                acquisition.set_X_pending(_join_pending(original, candidates))
    finally:
        if args["q"] > 1:
            acquisition.set_X_pending(original)
    joined = torch.cat(candidates)
    result = process(joined)
    if (
        not isinstance(result, Tensor)
        or result.shape != joined.shape
        or result.device != bounds.device
        or result.dtype != bounds.dtype
    ):
        raise CandidateGenerationError(
            "Post-processing must retain mixed candidate shape, dtype and device."
        )
    if not bool(
        feasible_batches(
            result.unsqueeze(0),
            bounds,
            args["inequality_constraints"],
            args["equality_constraints"],
        ).all()
    ):
        raise CandidateGenerationError("Post-processing produced infeasible mixed candidates.")
    if bool(((result < bounds[0]) | (result > bounds[1])).any()) or not bool(
        torch.isfinite(result).all()
    ):
        raise CandidateGenerationError(
            "Post-processing produced nonfinite or out-of-bounds candidates."
        )
    for index, value in _fixed_values(joined, fixed, bounds[0], bounds[1]).items():
        if not torch.allclose(result[..., index], value, rtol=0, atol=1e-8):
            raise CandidateGenerationError("Post-processing changed fixed feature values.")
    for i, values in {
        **_dims(args["discrete_dims"], bounds),
        **_dims(args["cat_dims"], bounds),
    }.items():
        if bool((result[:, i, None] != values).all(-1).any()):
            raise CandidateGenerationError(
                "Post-processing moved mixed candidates outside the discrete domain."
            )
    with torch.no_grad():
        return result, acquisition(result) if args["return_acq_values"] else None

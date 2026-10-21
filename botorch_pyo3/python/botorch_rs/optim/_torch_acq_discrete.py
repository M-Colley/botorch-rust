"""Device search over discrete, mixed and cyclic candidate domains.

The loops use the port's acquisitions and device candidate generation. Only
explicit SciPy candidate solvers cross the host/device solver boundary.
"""

from __future__ import annotations

import time
import warnings
from collections.abc import Mapping
from typing import Any

import torch
from torch import Tensor

from botorch_rs.acquisition.acquisition import OneShotAcquisitionFunction
from botorch_rs.exceptions.errors import CandidateGenerationError, InputDataError, UnsupportedError
from botorch_rs.exceptions.warnings import OptimizationWarning
from botorch_rs.optim._torch_acq_initializers import acquisition_batch_shape
from botorch_rs.optim._torch_acq_optim import _join_pending, _limit, select_best


def _score(acquisition: Any, points: Tensor, limit: int) -> tuple[Tensor, Tensor]:
    batch = acquisition_batch_shape(acquisition)
    inputs = points.reshape(points.shape[0], *((1,) * len(batch)), 1, points.shape[-1]).expand(
        points.shape[0], *batch, 1, points.shape[-1]
    )
    with torch.no_grad():
        values = torch.cat(
            [acquisition(part) for part in inputs.split(_limit(limit, "max_batch_size"))], 0
        )
    if values.shape != inputs.shape[:-2]:
        raise ValueError(
            "Discrete acquisition values must match choice and model batch dimensions."
        )
    return inputs, values


def _filter(points: Tensor, avoid: Any = None, constraints: Any = None) -> Tensor:
    keep = torch.ones(points.shape[0], dtype=torch.bool, device=points.device)
    if avoid is not None and avoid.numel():
        keep &= ~(points[:, None, :] == avoid[None, :, :]).all(-1).any(-1)
    for indices, coefficients, rhs in constraints or []:
        index = torch.as_tensor(indices, dtype=torch.long, device=points.device)
        if index.ndim != 1:
            raise UnsupportedError("Discrete search supports intra-point constraints only.")
        coefficients = torch.as_tensor(coefficients, dtype=points.dtype, device=points.device)
        keep &= (points[:, index] * coefficients).sum(-1) >= rhs
    return points[keep]


def discrete(arguments: Mapping[str, Any]) -> tuple[Tensor, Any]:
    args = dict(arguments)
    acquisition, q, choices = args["acq_function"], args["q"], args["choices"]
    if isinstance(acquisition, OneShotAcquisitionFunction):
        raise UnsupportedError("Discrete optimization does not support one-shot acquisitions.")
    _limit(q, "q")
    _limit(args["max_batch_size"], "max_batch_size")
    if not isinstance(choices, Tensor) or choices.ndim != 2:
        raise ValueError("choices must be a num_choices x d Tensor.")
    choices = _filter(
        choices, args["X_avoid"] if args["unique"] else None, args["inequality_constraints"]
    )
    if choices.shape[0] == 0:
        raise InputDataError("No feasible points remain in choices.")
    if args["unique"]:
        available_count = torch.unique(choices, dim=0).shape[0]
        if q > available_count:
            warnings.warn(
                f"Requested q={q} candidates but only {available_count} unique choices remain.",
                OptimizationWarning,
                stacklevel=3,
            )
            q = available_count
    original = getattr(acquisition, "X_pending", None)
    points, selected_values = [], []
    available: Tensor | None = None
    try:
        for _ in range(q):
            inputs, values = _score(acquisition, choices, args["max_batch_size"])
            if available is None:
                available = torch.ones_like(values, dtype=torch.bool)
            candidate, value = select_best(inputs, values.masked_fill(~available, -torch.inf))
            points.append(candidate)
            selected_values.append(value)
            if q > 1:
                acquisition.set_X_pending(_join_pending(original, points))
            if args["unique"]:
                available &= ~(inputs == candidate).all((-2, -1))
    finally:
        if q > 1:
            acquisition.set_X_pending(original)
    candidates = torch.cat(points, -2)
    values = selected_values[0] if q == 1 else torch.stack(selected_values)
    return candidates, values if args["return_acq_values"] else None


def _intrapoint(arguments: Mapping[str, Any]) -> None:
    for name in ("inequality_constraints", "equality_constraints"):
        if any(torch.as_tensor(index).ndim != 1 for index, _, _ in arguments.get(name) or []):
            raise UnsupportedError(
                "Mixed/sequential search does not support inter-point constraints."
            )
    if any(not intra for _, intra in arguments.get("nonlinear_inequality_constraints") or []):
        raise UnsupportedError(
            "Mixed/sequential search does not support inter-point nonlinear constraints."
        )


def mixed(arguments: Mapping[str, Any]) -> tuple[Tensor, Any]:
    from botorch_rs.optim.optimize import optimize_acqf

    args = dict(arguments)
    _intrapoint(args)
    acquisition, q = args["acq_function"], args["q"]
    _limit(q, "q")
    choices = args["fixed_features_list"]
    if not choices:
        raise ValueError("fixed_features_list must be non-empty.")
    if q > 1:
        if not args["return_best_only"]:
            raise NotImplementedError("return_best_only=False is supported for q=1 only.")
        if isinstance(acquisition, OneShotAcquisitionFunction) and not callable(
            getattr(acquisition, "evaluate", None)
        ):
            raise UnsupportedError("Sequential mixed one-shot search requires evaluate.")
        original = getattr(acquisition, "X_pending", None)
        points = []
        try:
            for _ in range(q):
                point, _ = mixed(
                    dict(
                        args,
                        q=1,
                        timeout_sec=None
                        if args["timeout_sec"] is None
                        else args["timeout_sec"] / q,
                        return_acq_values=False,
                    )
                )
                points.append(point)
                acquisition.set_X_pending(_join_pending(original, points))
        finally:
            acquisition.set_X_pending(original)
        candidate = torch.cat(points, -2)
        value = None
        if args["return_acq_values"]:
            with torch.no_grad():
                if isinstance(acquisition, OneShotAcquisitionFunction):
                    evaluate = getattr(acquisition, "evaluate", None)
                    assert callable(evaluate)
                    value = evaluate(X=candidate, bounds=args["bounds"])
                else:
                    value = acquisition(candidate)
        return candidate, value
    candidates, values = [], []
    failures = 0
    for fixed in choices:
        try:
            candidate, value = optimize_acqf(
                acquisition,
                args["bounds"],
                q=1,
                num_restarts=args["num_restarts"],
                raw_samples=args["raw_samples"],
                options=args["options"],
                inequality_constraints=args["inequality_constraints"],
                equality_constraints=args["equality_constraints"],
                nonlinear_inequality_constraints=args["nonlinear_inequality_constraints"],
                fixed_features=fixed,
                post_processing_func=args["post_processing_func"],
                batch_initial_conditions=args["batch_initial_conditions"],
                return_best_only=False,
                gen_candidates=args["gen_candidates"],
                ic_generator=args["ic_generator"],
                timeout_sec=None
                if args["timeout_sec"] is None
                else args["timeout_sec"] / len(choices),
                retry_on_optimization_warning=args["retry_on_optimization_warning"],
                **(args["ic_gen_kwargs"] or {}),
            )
        except CandidateGenerationError:
            failures += 1
            continue
        candidates.append(candidate)
        values.append(value)
    if not candidates:
        raise CandidateGenerationError("Candidate generation failed for all fixed_features.")
    if failures:
        warnings.warn(
            f"Candidate generation failed for {failures} fixed_features combinations.",
            OptimizationWarning,
            stacklevel=3,
        )
    all_candidates, all_values = torch.stack(candidates), torch.stack(values)
    if args["return_best_only"]:
        candidate, value = select_best(all_candidates.flatten(0, 1), all_values.flatten(0, 1))
    else:
        best = all_values.amax(1).argmax(0, keepdim=True)
        candidate = all_candidates.gather(
            0, best.unsqueeze(1)[..., None, None].expand(1, *all_candidates.shape[1:])
        ).squeeze(0)
        value = all_values.gather(0, best.unsqueeze(1).expand(1, *all_values.shape[1:])).squeeze(0)
    return candidate, value if args["return_acq_values"] else None


def acquisition_list(arguments: Mapping[str, Any]) -> tuple[Tensor, Any]:
    from botorch_rs.optim.optimize import optimize_acqf, optimize_acqf_mixed

    args = dict(arguments)
    args.pop("acquisition_list", None)
    acquisitions = args.pop("acq_function_list")
    if not acquisitions:
        raise ValueError("acq_function_list must be nonempty.")
    if args["fixed_features"] and args["fixed_features_list"]:
        raise ValueError("Specify fixed_features or fixed_features_list, not both.")
    if any(getattr(acq, "_tensor_backend", None) != "torch" for acq in acquisitions):
        raise UnsupportedError("All list acquisitions must use the Tensor backend.")
    originals = [getattr(acq, "X_pending", None) for acq in acquisitions]
    points: list[Tensor] = []
    values: list[Tensor] = []
    failed = False
    try:
        for i, acquisition in enumerate(acquisitions):
            if points:
                acquisition.set_X_pending(_join_pending(originals[i], points))
            kwargs = dict(args, acq_function=acquisition, q=1)
            fixed_list = kwargs.pop("fixed_features_list")
            if fixed_list:
                kwargs.pop("fixed_features")
                candidate, value = optimize_acqf_mixed(**kwargs, fixed_features_list=fixed_list)
            else:
                extra = kwargs.pop("ic_gen_kwargs") or {}
                candidate, value = optimize_acqf(**kwargs, **extra)
            points.append(candidate)
            if value is not None:
                values.append(value)
    except BaseException:
        failed = True
        raise
    finally:
        restore_error = None
        for acquisition, original in zip(acquisitions, originals, strict=True):
            try:
                acquisition.set_X_pending(original)
            except Exception as error:
                if restore_error is None:
                    restore_error = error
        if restore_error is not None and not failed:
            raise restore_error
    return torch.cat(points, -2), torch.stack(values) if args["return_acq_values"] else None


def cyclic(arguments: Mapping[str, Any]) -> tuple[Tensor, Any]:
    from botorch_rs.optim.optimize import optimize_acqf
    from botorch_rs.optim.stopping import ExpMAStoppingCriterion

    args = dict(arguments)
    args.pop("cyclic", None)
    criterion = ExpMAStoppingCriterion(**{"minimize": False, **(args.pop("cyclic_options") or {})})
    return_values = args.pop("return_acq_values")
    extra = args.pop("ic_gen_kwargs")
    acquisition, q = args["acq_function"], args["q"]
    original = getattr(acquisition, "X_pending", None)
    started = time.monotonic()
    timeout = args["timeout_sec"]
    candidates, values = optimize_acqf(**args, **extra, sequential=q > 1, return_acq_values=True)
    try:
        if q > 1:
            while not criterion(values):
                for i in range(q):
                    others = torch.cat((candidates[..., :i, :], candidates[..., i + 1 :, :]), -2)
                    acquisition.set_X_pending(_join_pending(original, [others]))
                    part = dict(
                        args,
                        q=1,
                        batch_initial_conditions=candidates[..., i : i + 1, :].unsqueeze(0),
                        raw_samples=None,
                        num_restarts=1,
                        timeout_sec=None
                        if timeout is None
                        else max(0.0, timeout - (time.monotonic() - started)),
                    )
                    point, value = optimize_acqf(**part, **extra, return_acq_values=True)
                    candidates = torch.cat(
                        (candidates[..., :i, :], point, candidates[..., i + 1 :, :]), -2
                    )
                    values = torch.cat((values[:i], value.unsqueeze(0), values[i + 1 :]), 0)
                if timeout is not None and time.monotonic() - started >= timeout:
                    break
    finally:
        if q > 1:
            acquisition.set_X_pending(original)
    return candidates, values if return_values else None


def lattice_points(
    discrete_choices: list[Tensor],
    raw_samples: int,
    avoid: Tensor,
    constraints: Any,
    min_points: int,
    max_tries: int,
    generator: Any = None,
) -> Tensor:
    if not discrete_choices or any(c.ndim != 1 or c.numel() == 0 for c in discrete_choices):
        raise ValueError("Every discrete feature must have nonempty one-dimensional choices.")
    _limit(raw_samples, "raw_samples")
    _limit(min_points, "min_points")
    points = avoid.new_empty(0, len(discrete_choices))
    for _ in range(_limit(max_tries, "max_tries")):
        sample = torch.stack(
            [
                c[torch.randint(c.numel(), (raw_samples,), device=c.device, generator=generator)]
                for c in discrete_choices
            ],
            -1,
        )
        points = _filter(torch.unique(torch.cat((points, sample), 0), dim=0), avoid, constraints)
        if points.shape[0] >= min_points:
            return points
    raise RuntimeError(f"Failed to generate at least {min_points} feasible lattice points.")


def local_search(arguments: Mapping[str, Any]) -> tuple[Tensor, Any]:
    args = dict(arguments)
    acquisition, choices = args["acq_function"], args["discrete_choices"]
    if acquisition_batch_shape(acquisition):
        raise UnsupportedError(
            "Discrete local search requires an unbatched acquisition, as in BoTorch."
        )
    if isinstance(acquisition, OneShotAcquisitionFunction):
        raise UnsupportedError("Discrete local search does not support one-shot acquisitions.")
    if not choices:
        raise ValueError("discrete_choices must be nonempty.")
    like = choices[0]
    choices = [torch.as_tensor(c, dtype=like.dtype, device=like.device) for c in choices]
    avoid = like.new_empty(0, len(choices)) if args["X_avoid"] is None else args["X_avoid"]
    original = getattr(acquisition, "X_pending", None)
    generator = (
        None
        if args["seed"] is None
        else torch.Generator(device=like.device).manual_seed(args["seed"])
    )
    points = []
    initial = args["batch_initial_conditions"]
    if initial is not None and (initial.ndim != 3 or initial.shape[-2:] != (1, len(choices))):
        raise ValueError("batch_initial_conditions must have shape n x 1 x d.")
    try:
        for _ in range(_limit(args["q"], "q")):
            starts = (
                like.new_empty(0, len(choices))
                if initial is None
                else _filter(initial.squeeze(-2), avoid, args["inequality_constraints"])
            )
            required = args["num_restarts"] - starts.shape[0]
            if required > 0:
                generated = lattice_points(
                    choices,
                    args["raw_samples"],
                    avoid,
                    args["inequality_constraints"],
                    args["num_restarts"],
                    args["max_tries"],
                    generator,
                )
                _, scores = _score(acquisition, generated, args["max_batch_size"])
                starts = torch.cat(
                    (starts, generated[scores.topk(args["num_restarts"]).indices]), 0
                )
            initial = None
            best_points, best_values = [], []
            for start in starts:
                point = start.unsqueeze(0)
                _, value = _score(acquisition, point, args["max_batch_size"])
                while True:
                    neighbors = []
                    for i, values in enumerate(choices):
                        perturb = point.expand(values.numel(), len(choices)).clone()
                        perturb[:, i] = values
                        neighbors.append(perturb)
                    neighborhood = _filter(
                        torch.cat(neighbors, 0),
                        torch.cat((avoid, point), 0),
                        args["inequality_constraints"],
                    )
                    if neighborhood.shape[0] == 0:
                        break
                    _, scores = _score(acquisition, neighborhood, args["max_batch_size"])
                    best = scores.argmax()
                    if bool(scores[best] <= value):
                        break
                    point, value = neighborhood[best].unsqueeze(0), scores[best]
                best_points.append(point)
                best_values.append(value.reshape(()))
            selected = torch.stack(best_values).argmax()
            points.append(torch.cat(best_points, 0)[selected].unsqueeze(0))
            if args["q"] > 1:
                acquisition.set_X_pending(_join_pending(original, points))
                if args["unique"]:
                    avoid = torch.cat(
                        ([args["X_avoid"]] if args["X_avoid"] is not None else []) + points, 0
                    )
    finally:
        if args["q"] > 1:
            acquisition.set_X_pending(original)
    candidate = torch.cat(points, 0)
    with torch.no_grad():
        joint_value = acquisition(candidate) if args["return_acq_values"] else None
    return candidate, joint_value

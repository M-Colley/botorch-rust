"""Homotopy optimization (``botorch.optim.optimize_homotopy``).

The homotopy loop runs in Python (its parameters are Python objects); every
step uses :func:`~botorch_rs.optim.optimize_acqf` (or
:func:`~botorch_rs.optim.optimize_acqf_mixed`,
:func:`~botorch_rs.optim.optimize_acqf_mixed_alternating`) and
:func:`prune_candidates` runs in Rust for array inputs and on-device for tensors.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from itertools import product
from typing import Any, cast

from botorch_rs import _botorch_rs as _rs
from botorch_rs._arrays import first_kind, from_flat, from_pair, is_torch_tensor, kind_of, to_flat
from botorch_rs.exceptions.errors import CandidateGenerationError, UnsupportedError
from botorch_rs.optim.homotopy import Homotopy
from botorch_rs.optim.optimize import _concat_rows, optimize_acqf, optimize_acqf_mixed
from botorch_rs.optim.optimize_mixed import (
    optimize_acqf_mixed_alternating,
    should_use_mixed_alternating_optimizer,
)


def prune_candidates(candidates: Any, acq_values: Any, prune_tolerance: float) -> Any:
    r"""Prune candidates based on their distance to other candidates
    (BoTorch's ``prune_candidates``): sorted by descending value, a candidate
    is kept if its Euclidean distance to all kept candidates exceeds
    ``prune_tolerance``.

    Args:
        candidates: An ``n x d`` array of candidates.
        acq_values: An ``n`` array of candidate values.
        prune_tolerance: The minimum distance to prune candidates.

    Returns:
        An ``m x d`` array of pruned candidates.
    """
    if is_torch_tensor(candidates):
        import torch

        if candidates.ndim != 2:
            raise ValueError("`candidates` must be of size `n x d`.")
        if acq_values.ndim != 1 or acq_values.shape[0] != candidates.shape[0]:
            raise ValueError("`acq_values` must be of size `n`.")
        if prune_tolerance < 0:
            raise ValueError("`prune_tolerance` must be >= 0.")
        ordered = candidates[acq_values.argsort(descending=True)]
        kept = ordered[:1]
        for i in range(1, ordered.shape[0]):
            if bool(torch.cdist(ordered[i : i + 1], kept).amin() > prune_tolerance):
                kept = torch.cat((kept, ordered[i : i + 1]), -2)
        return kept
    cflat = to_flat(candidates, "candidates")
    if len(cflat[1]) != 2:
        raise ValueError("`candidates` must be of size `n x d`.")
    vdata, vshape = to_flat(acq_values, "acq_values")
    if len(vshape) != 1 or vshape[0] != cflat[1][0]:
        raise ValueError("`acq_values` must be of size `n`.")
    if prune_tolerance < 0:
        raise ValueError("`prune_tolerance` must be >= 0.")
    out = _rs.prune_candidates_py(cflat, [float(v) for v in vdata], float(prune_tolerance))
    return from_pair(out, kind_of(candidates))


def _torch_argmax(values: list[float]) -> int:
    """``torch.argmax`` (the first NaN, else the first maximum)."""
    for i, v in enumerate(values):
        if v != v:
            return i
    return max(range(len(values)), key=lambda i: (values[i], -i))


def _unsqueeze1(X: Any) -> Any:
    """``X.unsqueeze(1)`` for an ``n x d`` array."""
    if is_torch_tensor(X):
        return X.unsqueeze(1)
    data, shape = to_flat(X, "X")
    return from_flat(data, (shape[0], 1, shape[1]), kind_of(X))


def _squeeze1(X: Any) -> Any:
    """``X.squeeze(1)`` for an ``n x 1 x d`` array."""
    if is_torch_tensor(X):
        return X.squeeze(1)
    data, shape = to_flat(X, "X")
    return from_flat(data, (shape[0], shape[-1]), kind_of(X))


def _check_postprocessed(
    X: Any,
    original: Any,
    bounds: Any,
    fixed: Any,
    inequalities: Any,
    equalities: Any,
    nonlinear: Any,
    domains: Any = None,
) -> None:
    """Postprocessing must preserve feasible physical candidate coordinates."""
    from botorch_rs.optim.parameter_constraints import (
        evaluate_feasibility,
        get_constraint_tolerance,
    )

    if is_torch_tensor(X):
        import torch

        from botorch_rs.optim._torch_acq_optim import _fixed_values

        if (
            not is_torch_tensor(original)
            or X.shape != original.shape
            or X.dtype != original.dtype
            or X.device != original.device
        ):
            raise CandidateGenerationError(
                "Postprocessing must retain candidate shape, dtype and device."
            )
        limits = torch.as_tensor(bounds, dtype=X.dtype, device=X.device)
        tolerance = get_constraint_tolerance(X.dtype)
        if not bool(torch.isfinite(X).all()) or bool(
            ((X < limits[0] - tolerance) | (X > limits[1] + tolerance)).any()
        ):
            raise CandidateGenerationError(
                "Homotopy postprocessing produced nonfinite or out-of-bounds candidates."
            )
        for index, value in _fixed_values(original, fixed, limits[0], limits[1]).items():
            if not torch.allclose(X[..., index], value, rtol=0, atol=tolerance):
                raise CandidateGenerationError(
                    "Homotopy postprocessing changed fixed feature values."
                )
        if not bool(evaluate_feasibility(X, inequalities, equalities, nonlinear).all()):
            raise CandidateGenerationError(
                "Homotopy postprocessing produced infeasible candidates."
            )
        for index, values in (domains or {}).items():
            allowed = torch.as_tensor(values, dtype=X.dtype, device=X.device)
            if not bool((X[..., index, None] - allowed).abs().amin(-1).le(tolerance).all()):
                raise CandidateGenerationError("Homotopy postprocessing left a discrete domain.")
        return
    import math

    data, shape = to_flat(X)
    previous, old_shape = to_flat(original)
    native_limits, limit_shape = to_flat(bounds)
    d = limit_shape[-1]
    if shape != old_shape or any(
        not math.isfinite(value)
        or value < native_limits[i % d] - 1e-8
        or value > native_limits[d + i % d] + 1e-8
        for i, value in enumerate(data)
    ):
        raise CandidateGenerationError("Homotopy postprocessing produced invalid candidates.")
    for index, value in (fixed or {}).items():
        for row in range(len(data) // d):
            target = previous[row * d + index] if value is None else float(to_flat(value)[0][0])
            if abs(data[row * d + index] - target) > 1e-8:
                raise CandidateGenerationError(
                    "Homotopy postprocessing changed fixed feature values."
                )
    if not all(to_flat(evaluate_feasibility(X, inequalities, equalities, nonlinear))[0]):
        raise CandidateGenerationError("Homotopy postprocessing produced infeasible candidates.")
    for index, values in (domains or {}).items():
        native_allowed = to_flat(values)[0]
        if any(
            min(abs(data[row * d + index] - value) for value in native_allowed) > 1e-8
            for row in range(len(data) // d)
        ):
            raise CandidateGenerationError("Homotopy postprocessing left a discrete domain.")


def optimize_acqf_homotopy(
    acq_function: Any,
    bounds: Any,
    q: int,
    num_restarts: int,
    homotopy: Homotopy,
    prune_tolerance: float = 1e-4,
    raw_samples: int | None = None,
    options: dict[str, Any] | None = None,
    final_options: dict[str, Any] | None = None,
    inequality_constraints: list[tuple[Any, Any, float]] | None = None,
    equality_constraints: list[tuple[Any, Any, float]] | None = None,
    nonlinear_inequality_constraints: list[tuple[Callable[..., Any], bool]] | None = None,
    fixed_features: dict[int, float] | None = None,
    discrete_dims: Mapping[int, Sequence[float]] | None = None,
    cat_dims: Mapping[int, Sequence[float]] | None = None,
    post_processing_func: Callable[[Any], Any] | None = None,
    batch_initial_conditions: Any = None,
    gen_candidates: Any = None,
    *,
    ic_generator: Any = None,
    timeout_sec: float | None = None,
    retry_on_optimization_warning: bool = True,
    return_acq_values: bool = True,
    **ic_gen_kwargs: Any,
) -> tuple[Any, Any]:
    r"""Generate a set of candidates via multi-start optimization along a
    homotopy (BoTorch's ``optimize_acqf_homotopy``).

    For each of the ``q`` candidates (generated sequentially with the
    previous ones as pending points), the homotopy is restarted and the
    acquisition function is optimized at every step of the schedule, starting
    from the pruned candidates of the previous step (``prune_tolerance``);
    a final optimization with ``final_options`` gives the candidate. With
    ``discrete_dims`` / ``cat_dims``, ``optimize_acqf_mixed_alternating`` is
    used for more than 10 discrete combinations, else
    ``optimize_acqf_mixed`` over all combinations.

    Returns:
        A ``q x d`` array of candidates and their ``q`` acquisition values
        (``None`` if ``return_acq_values=False``).
    """
    tensor_backend = getattr(acq_function, "_tensor_backend", None) == "torch"
    if tensor_backend:
        import torch

        from botorch_rs.acquisition.acquisition import OneShotAcquisitionFunction
        from botorch_rs.optim._torch_acq_initializers import acquisition_batch_shape
        from botorch_rs.optim._torch_acq_optim import _join_pending, _limit

        _limit(q, "q")
        if acquisition_batch_shape(acq_function) or isinstance(
            acq_function, OneShotAcquisitionFunction
        ):
            raise UnsupportedError(
                "Homotopy optimization requires an unbatched, non-one-shot acquisition."
            )
    kind = first_kind(bounds)
    use_mixed_alternating = should_use_mixed_alternating_optimizer(
        discrete_dims=discrete_dims, cat_dims=cat_dims
    )
    optimization_fn: Callable[..., tuple[Any, Any]]
    fixed_features_kwargs: dict[str, Any]
    if use_mixed_alternating:
        optimization_fn = cast("Callable[..., tuple[Any, Any]]", optimize_acqf_mixed_alternating)
        shared_optimize_acqf_kwargs: dict[str, Any] = {
            "num_restarts": num_restarts,
            "inequality_constraints": inequality_constraints,
            "equality_constraints": equality_constraints,
            "discrete_dims": discrete_dims,
            "cat_dims": cat_dims,
            "fixed_features": fixed_features,
        }
        fixed_features_kwargs = {}
    elif discrete_dims is not None or cat_dims is not None:
        all_discrete_dims = {**(discrete_dims or {}), **(cat_dims or {})}
        dim_indices = sorted(all_discrete_dims.keys())
        value_lists = [list(all_discrete_dims[idx]) for idx in dim_indices]
        fixed_features_list = [
            dict(zip(dim_indices, combo, strict=True)) for combo in product(*value_lists)
        ]
        optimization_fn = cast("Callable[..., tuple[Any, Any]]", optimize_acqf_mixed)
        shared_optimize_acqf_kwargs = {
            "num_restarts": num_restarts,
            "inequality_constraints": inequality_constraints,
            "equality_constraints": equality_constraints,
            "nonlinear_inequality_constraints": nonlinear_inequality_constraints,
            "return_best_only": False,
            "gen_candidates": gen_candidates,
            "ic_generator": ic_generator,
            "timeout_sec": timeout_sec,
            "retry_on_optimization_warning": retry_on_optimization_warning,
            "ic_gen_kwargs": ic_gen_kwargs or None,
        }
        fixed_features_kwargs = {"fixed_features_list": fixed_features_list}
    else:
        optimization_fn = optimize_acqf
        shared_optimize_acqf_kwargs = {
            "num_restarts": num_restarts,
            "inequality_constraints": inequality_constraints,
            "equality_constraints": equality_constraints,
            "nonlinear_inequality_constraints": nonlinear_inequality_constraints,
            "return_best_only": False,
            "gen_candidates": gen_candidates,
            "ic_generator": ic_generator,
            "timeout_sec": timeout_sec,
            "retry_on_optimization_warning": retry_on_optimization_warning,
            **ic_gen_kwargs,
        }
        fixed_features_kwargs = (
            {"fixed_features": fixed_features} if fixed_features is not None else {}
        )

    candidate_list: list[Any] = []
    acq_value_list: list[Any] = []
    base_X_pending = acq_function.X_pending if q > 1 else None
    selected_candidates: Any = None
    failed = False
    try:
        for _ in range(q):
            candidates = batch_initial_conditions
            q_raw_samples = raw_samples
            homotopy.restart()

            while not homotopy.should_stop:
                if use_mixed_alternating:
                    candidates, acq_values = optimization_fn(
                        acq_function=acq_function,
                        bounds=bounds,
                        q=1,
                        options=options,
                        raw_samples=q_raw_samples or raw_samples,
                        post_processing_func=post_processing_func,
                        **fixed_features_kwargs,
                        **shared_optimize_acqf_kwargs,
                    )
                    candidates = _unsqueeze1(candidates)
                    acq_values = (
                        acq_values.reshape(-1)
                        if tensor_backend
                        else from_flat(to_flat(acq_values)[0], (1,), kind_of(acq_values))
                    )
                else:
                    candidates, acq_values = optimization_fn(
                        acq_function=acq_function,
                        bounds=bounds,
                        q=1,
                        options=options,
                        batch_initial_conditions=candidates,
                        raw_samples=q_raw_samples,
                        **fixed_features_kwargs,
                        **shared_optimize_acqf_kwargs,
                    )
                homotopy.step()
                # Set raw_samples to None such that pruned restarts are not
                # repopulated at each step in the homotopy.
                q_raw_samples = None
                candidates = _unsqueeze1(
                    prune_candidates(
                        candidates=_squeeze1(candidates),
                        acq_values=acq_values,
                        prune_tolerance=prune_tolerance,
                    )
                )

            # Optimize one more time with the final options.
            if use_mixed_alternating:
                candidates, acq_values = optimization_fn(
                    acq_function=acq_function,
                    bounds=bounds,
                    q=1,
                    options=final_options,
                    raw_samples=raw_samples,
                    post_processing_func=post_processing_func,
                    **fixed_features_kwargs,
                    **shared_optimize_acqf_kwargs,
                )
                candidates = _unsqueeze1(candidates)
                acq_values = (
                    acq_values.reshape(-1)
                    if tensor_backend
                    else from_flat(to_flat(acq_values)[0], (1,), kind_of(acq_values))
                )
            else:
                candidates, acq_values = optimization_fn(
                    acq_function=acq_function,
                    bounds=bounds,
                    q=1,
                    options=final_options,
                    raw_samples=q_raw_samples,
                    batch_initial_conditions=candidates,
                    **fixed_features_kwargs,
                    **shared_optimize_acqf_kwargs,
                )

            # Post-process the candidates and grab the best candidate.
            if post_processing_func is not None and not use_mixed_alternating:
                before = candidates
                candidates = post_processing_func(candidates)
                _check_postprocessed(
                    candidates,
                    before,
                    bounds,
                    fixed_features,
                    inequality_constraints,
                    equality_constraints,
                    nonlinear_inequality_constraints,
                    {**(discrete_dims or {}), **(cat_dims or {})},
                )
                acq_values = acq_function(candidates)

            if tensor_backend:
                if acq_values.shape != candidates.shape[:1]:
                    raise UnsupportedError("Homotopy optimization requires one scalar per restart.")
                best = acq_values.argmax()
                candidate = candidates[best]
                value = acq_values[best]
            else:
                values = [float(v) for v in to_flat(acq_values, "acq_values")[0]]
                best = _torch_argmax(values)
                cdata, cshape = to_flat(candidates, "candidates")
                per = cshape[-2] * cshape[-1]
                candidate = from_flat(cdata[best * per : (best + 1) * per], tuple(cshape[1:]), kind)
                value = values[best]
            candidate_list.append(candidate)
            if return_acq_values:
                acq_value_list.append(value)
            selected_candidates = (
                torch.cat(candidate_list, -2)
                if tensor_backend
                else _concat_rows(candidate_list, kind)
            )
            if q > 1:
                acq_function.set_X_pending(
                    _join_pending(base_X_pending, candidate_list)
                    if tensor_backend
                    else (
                        _concat_rows([base_X_pending, selected_candidates], kind)
                        if base_X_pending is not None
                        else selected_candidates
                    )
                )

    except BaseException:
        failed = True
        raise
    finally:
        restore_error = None
        if q > 1:
            try:
                acq_function.set_X_pending(base_X_pending)
            except Exception as error:
                restore_error = error
        try:
            homotopy.reset()
        except Exception as error:
            if restore_error is None:
                restore_error = error
        if restore_error is not None and not failed:
            raise restore_error

    if not return_acq_values:
        return selected_candidates, None
    return selected_candidates, (
        torch.stack(acq_value_list)
        if tensor_backend
        else from_flat(acq_value_list, (len(acq_value_list),), kind)
    )


__all__ = ["optimize_acqf_homotopy", "prune_candidates"]

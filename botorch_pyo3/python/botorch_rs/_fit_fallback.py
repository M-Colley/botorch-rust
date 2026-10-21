"""Generic optimizer, retry and rollback semantics for high-level model fitting."""

from __future__ import annotations

import copy
import functools
import warnings
from contextlib import ExitStack
from typing import Any, NamedTuple

from botorch_rs.exceptions.errors import InvalidArgumentError, ModelFittingError, NotPSDError
from botorch_rs.exceptions.warnings import OptimizationWarning
from botorch_rs.optim.core import _LBFGSB_MAXITER_MAXFUN_REGEX


def default_warning_handler(message: warnings.WarningMessage) -> bool:
    if _LBFGSB_MAXITER_MAXFUN_REGEX.search(str(message.message)):
        return True
    if not issubclass(
        message.category, OptimizationWarning
    ) or "Optimization timed out after" in str(message.message):
        warnings.warn_explicit(
            str(message.message), message.category, message.filename, message.lineno
        )
        return True
    return False


def _roots(model: Any) -> list[Any]:
    if getattr(model, "_tensor_backend", None) == "torch":
        return [model]
    children = getattr(model, "models", None)
    roots = list(children) if children is not None else [getattr(model, "_outer", model)]
    return list({id(root): root for root in roots}.values())


class _FitCheckpoint(NamedTuple):
    state: Any


def _snapshots(roots: list[Any]) -> list[Any]:
    from botorch_rs.utils.context_managers import module_rollback_ctx

    snapshots: list[Any] = []
    for root in roots:
        capture = getattr(root, "_fit_snapshot", None)
        if callable(capture) and getattr(root, "_tensor_backend", None) != "torch":
            snapshots.append(_FitCheckpoint(capture()))
            continue
        with module_rollback_ctx(root, device="cpu") as checkpoint:
            snapshots.append(copy.deepcopy(checkpoint))
            checkpoint.clear()
    return snapshots


def _restore(roots: list[Any], snapshots: list[Any]) -> None:
    from botorch_rs.utils.context_managers import module_rollback_ctx

    with ExitStack() as stack:
        for root, checkpoint in zip(roots, snapshots, strict=True):
            if isinstance(checkpoint, _FitCheckpoint):
                root._fit_restore(checkpoint.state)
                continue
            stack.enter_context(module_rollback_ctx(root, checkpoint=checkpoint))


def fit_fallback(
    mll: Any,
    optimizer: Any,
    closure: Any = None,
    closure_kwargs: dict[str, Any] | None = None,
    optimizer_kwargs: dict[str, Any] | None = None,
    *,
    max_attempts: int = 5,
    pick_best_of_all_attempts: bool = False,
    warning_handler: Any = default_warning_handler,
    caught_exception_types: tuple[type[BaseException], ...] = (NotPSDError,),
    seed: int | None = None,
    **ignore: Any,
) -> Any:
    from botorch_rs.optim.closures import get_loss_closure_with_grads
    from botorch_rs.optim.utils.model_utils import get_parameters, sample_all_priors
    from botorch_rs.utils.context_managers import parameter_rollback_ctx

    if not isinstance(max_attempts, int) or max_attempts < 1:
        raise InvalidArgumentError("max_attempts must be a positive integer")
    if not callable(optimizer):
        raise TypeError("optimizer must be callable")
    mll.train()
    # A supplied approximate model need not register its separate likelihood.
    # Both are real children of this Tensor MLL and must roll back together.
    roots = [mll] if getattr(mll, "_external_model_protocol", False) else _roots(mll.model)
    for root in roots:
        initialize = getattr(root, "_ensure_initialized", None)
        if initialize is not None:
            initialize()
    initial = _snapshots(roots)
    frozen = get_parameters(mll, requires_grad=False)
    if closure is None and closure_kwargs is not None:
        closure = get_loss_closure_with_grads(mll, get_parameters(mll, requires_grad=True))
    if closure_kwargs is not None:
        closure = functools.partial(closure, **closure_kwargs)
    best = None
    summaries = []
    selected = 0
    for attempt in range(1, max_attempts + 1):
        _restore(roots, initial)
        mll.train()
        try:
            if attempt > 1:
                with parameter_rollback_ctx(frozen):
                    sample_all_priors(
                        mll.model, **({} if seed is None else {"seed": seed + attempt})
                    )
            with warnings.catch_warnings(record=True) as messages:
                warnings.simplefilter("always", OptimizationWarning)
                result = optimizer(mll, closure=closure, **dict(optimizer_kwargs or {}))
            success = True
            for message in messages:
                if not warning_handler(message):
                    warnings.warn_explicit(
                        str(message.message), message.category, message.filename, message.lineno
                    )
                    success = False
            summaries.append(result if success else None)
            if success and (
                not pick_best_of_all_attempts
                or float(result.fval) < (float("inf") if best is None else float(best[0].fval))
            ):
                best = (result, _snapshots(roots))
                selected = attempt
            if success and not pick_best_of_all_attempts:
                break
        except caught_exception_types:
            summaries.append(None)
        except BaseException:
            _restore(roots, initial)
            raise
        finally:
            _restore(roots, initial)
    if best is None:
        mll.train()
        raise ModelFittingError("All attempts to fit the model have failed.")
    result, state = best
    _restore(roots, state)
    mll._fit_result = {
        "fval": result.fval,
        "nit": result.step,
        "status": result.status.name,
        "message": result.message,
        "runtime": result.runtime,
        "attempts": len(summaries),
        "selected_attempt": selected,
    }
    mll._fit_attempt_results = summaries
    mll._fit_selected_attempt = selected
    return mll.eval()

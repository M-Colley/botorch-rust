"""Model fitting routines (``botorch.optim.fit``).

For an ``ExactMarginalLogLikelihood`` of a botorch_rs ``SingleTaskGP`` with
the default closure, both routines run natively: ``fit_gpytorch_mll_torch``
runs Adam (torch's update) with BoTorch's ``ExpMAStoppingCriterion``,
``fit_gpytorch_mll_scipy`` one L-BFGS-B run (a port of L-BFGS-B 3.0 that
reproduces SciPy's iterates; batched and multi-output models element by
element). Other optimizers, schedulers, callbacks, closures and models
use BoTorch's generic drivers (:func:`~botorch_rs.optim.core.torch_minimize`
with torch, :func:`~botorch_rs.optim.core.scipy_minimize` with SciPy) on
the natively computed loss and gradient.
"""

from __future__ import annotations

import functools
import math
import warnings
from collections.abc import Callable, Sequence
from typing import Any

from botorch_rs import _botorch_rs as _rs
from botorch_rs.exceptions.errors import UnsupportedError
from botorch_rs.exceptions.warnings import OptimizationWarning
from botorch_rs.optim.core import (
    OptimizationResult,
    OptimizationStatus,
    _result,
    scipy_minimize,
    torch_minimize,
)
from botorch_rs.optim.stopping import ExpMAStoppingCriterion
from botorch_rs.utils.types import DEFAULT

_EPS = 2.220446049250313e-16


def _native_model(mll: Any) -> Any:
    """The botorch_rs SingleTaskGP of an ExactMarginalLogLikelihood, or None."""
    from botorch_rs.mlls import ExactMarginalLogLikelihood
    from botorch_rs.models.gp_regression import SingleTaskGP

    if (
        isinstance(mll, ExactMarginalLogLikelihood)
        and isinstance(mll.model, SingleTaskGP)
        and getattr(mll.model, "_tensor_backend", None) != "torch"
    ):
        return mll.model
    return None


def _is_approximate(mll: Any) -> bool:
    """Whether ``mll`` is a botorch_rs variational ELBO."""
    from botorch_rs.mlls import _ApproximateMarginalLogLikelihood

    return isinstance(mll, _ApproximateMarginalLogLikelihood)


def _is_higher_order(mll: Any) -> bool:
    from botorch_rs.mlls import ExactMarginalLogLikelihood
    from botorch_rs.models.empirical_gps.em_empirical_gp import EMEmpiricalMarginalLogLikelihood
    from botorch_rs.models.empirical_gps.empirical_1d_gp import EmpiricalOneDimensionalGP
    from botorch_rs.models.empirical_gps.shared_gp import _SharedSumMarginalLogLikelihood
    from botorch_rs.models.higher_order_gp import HigherOrderGP
    from botorch_rs.models.latent_kronecker_gp import LatentKroneckerGP

    if isinstance(mll, (EMEmpiricalMarginalLogLikelihood, _SharedSumMarginalLogLikelihood)):
        mll._validate_fit()
        return True
    return isinstance(mll, ExactMarginalLogLikelihood) and isinstance(
        mll.model, (HigherOrderGP, LatentKroneckerGP, EmpiricalOneDimensionalGP)
    )


def _native_fit_backend(mll: Any) -> Any:
    return mll._rs if getattr(mll, "_rs_shared_fit", False) else mll.model._rs


def _fit_scipy_approximate(
    mll: Any,
    parameters: Any,
    bounds: Any,
    closure: Any,
    closure_kwargs: Any,
    method: str,
    options: dict[str, Any] | None,
    callback: Any,
    timeout_sec: float | None,
) -> OptimizationResult:
    """One native L-BFGS-B run on the selected variational parameters."""
    if (
        closure is not None
        or closure_kwargs is not None
        or method != "L-BFGS-B"
        or callback is not None
        or timeout_sec is not None
    ):
        raise UnsupportedError(
            "botorch_rs fits a variational ELBO with native L-BFGS-B (no "
            "custom closures, callbacks or timeouts)."
        )
    opts = dict(options or {})
    kwargs: dict[str, Any] = {}
    for key in ("maxiter", "maxfun", "maxcor", "maxls"):
        if key in opts:
            kwargs[key] = int(opts.pop(key))
    if "ftol" in opts:
        kwargs["ftol"] = float(opts.pop("ftol"))
    if "gtol" in opts:
        kwargs["pgtol"] = float(opts.pop("gtol"))
    for key in ("disp", "iprint"):
        opts.pop(key, None)
    if opts:
        warnings.warn(
            f"Unknown solver options: {', '.join(sorted(opts))}", UserWarning, stacklevel=3
        )
    mll.model._outer._ensure_initialized()
    mask, flat_bounds = _mask_and_bounds(mll, parameters, bounds)
    loss, nit, _, status, message, _ = mll._rs.fit(
        float(mll.num_data),
        float(mll.beta),
        max_attempts=1,
        mask=mask,
        bounds=flat_bounds,
        **kwargs,
    )
    mll.model._outer._sync_torch_parameters()
    result = OptimizationResult(
        step=int(nit),
        fval=float(loss),
        status=OptimizationStatus[status],
        message=message,
    )
    return result


def _mask_and_bounds(
    mll: Any,
    parameters: dict[str, Any] | None,
    bounds: dict[str, tuple[float | None, float | None]] | None,
) -> tuple[list[bool] | None, list[tuple[float, float]] | None]:
    """The flat mask of the optimized raw parameters and the flat bounds."""
    from botorch_rs.optim.utils.model_utils import _entries, _requires_grad

    entries = _entries(mll)
    if parameters is None and any(not _requires_grad(e.value) for e in entries):
        parameters = {e.name: e.value for e in entries if _requires_grad(e.value)}
    known = {e.name for e in entries}
    if parameters is not None:
        unknown = [n for n in parameters if n not in known]
        if unknown:
            raise ValueError(f"Unknown parameters: {unknown}.")
    mask: list[bool] | None = None
    flat_bounds: list[tuple[float, float]] | None = None
    model = entries[0].model if entries else None
    if model is None:
        return None, None
    n = sum(e.size for e in entries)
    if parameters is not None:
        mask = [False] * n
        for e in entries:
            if e.name in parameters:
                mask[e.offset : e.offset + e.size] = [True] * e.size
    if bounds:
        flat_bounds = list(model._rs.raw_parameter_bounds())
        for e in entries:
            if e.name in bounds:
                lo, hi = bounds[e.name]
                lo_f = -math.inf if lo is None else float(lo)
                hi_f = math.inf if hi is None else float(hi)
                flat_bounds[e.offset : e.offset + e.size] = [(lo_f, hi_f)] * e.size
    return mask, flat_bounds


def fit_gpytorch_mll_scipy(
    mll: Any,
    parameters: dict[str, Any] | None = None,
    bounds: dict[str, tuple[float | None, float | None]] | None = None,
    closure: Callable[[], tuple[Any, Sequence[Any]]] | None = None,
    closure_kwargs: dict[str, Any] | None = None,
    method: str = "L-BFGS-B",
    options: dict[str, Any] | None = None,
    callback: Callable[[dict[str, Any], OptimizationResult], None] | None = None,
    timeout_sec: float | None = None,
) -> OptimizationResult:
    r"""Generic scipy.optimize-based fitting routine for marginal log
    likelihoods (BoTorch's ``fit_gpytorch_mll_scipy``): one L-BFGS-B run, no
    retries.

    For models with several outputs or a batch shape, every element is
    fitted independently (BoTorch's batched independent fitting, which
    ignores ``timeout_sec`` and options other than ``maxiter``, ``maxcor``,
    ``ftol``, ``gtol`` / ``pgtol``, ``factr`` and ``maxls``).

    Args:
        mll: The marginal log likelihood to be maximized (in train mode).
        parameters: Optional dictionary of parameters to be optimized (by
            name, see ``get_parameters``). Defaults to all parameters.
        bounds: A dictionary of user-specified bounds for ``parameters``.
        closure: An optional closure returning the loss and gradients.
        closure_kwargs: Keyword arguments passed to ``closure``.
        method: Solver type (``"L-BFGS-B"`` runs natively).
        options: Solver options.
        callback: Optional callback taking ``parameters`` and an
            ``OptimizationResult``.
        timeout_sec: Timeout in seconds after which to terminate the fit.

    Returns:
        The final OptimizationResult.
    """
    if getattr(getattr(mll, "model", None), "_tensor_backend", None) == "torch":
        return _fit_generic_scipy(
            mll, parameters, bounds, closure, closure_kwargs, method, options, callback, timeout_sec
        )
    if _is_approximate(mll):
        return _fit_scipy_approximate(
            mll, parameters, bounds, closure, closure_kwargs, method, options, callback, timeout_sec
        )
    if getattr(mll, "_rs_shared_fit", False) and (
        parameters is not None
        or bounds is not None
        or closure is not None
        or closure_kwargs is not None
        or method != "L-BFGS-B"
        or callback is not None
        or timeout_sec is not None
    ):
        raise UnsupportedError(
            "Shared GP lists require the native L-BFGS-B closure over all parameters."
        )
    if getattr(mll, "_rs_em_fit", False) and (
        closure is not None
        or closure_kwargs is not None
        or method != "L-BFGS-B"
        or callback is not None
        or timeout_sec is not None
    ):
        raise UnsupportedError("EM likelihoods require the native L-BFGS-B closure.")
    if (
        _is_higher_order(mll)
        and ((parameters is None and bounds is None) or getattr(mll, "_rs_em_fit", False))
        and closure is None
        and closure_kwargs is None
        and method == "L-BFGS-B"
        and callback is None
        and timeout_sec is None
    ):
        opts = dict(options or {})
        hogp_kwargs: dict[str, Any] = {}
        for key in ("maxiter", "maxfun", "maxcor", "maxls", "ftol"):
            if key in opts:
                hogp_kwargs[key] = opts.pop(key)
        if "gtol" in opts:
            hogp_kwargs["pgtol"] = opts.pop("gtol")
        for key in ("disp", "iprint"):
            opts.pop(key, None)
        if opts:
            raise UnsupportedError(f"Unsupported L-BFGS-B options: {sorted(opts)}.")
        if getattr(mll, "_rs_em_fit", False):
            mask, flat_bounds = _mask_and_bounds(mll, parameters, bounds)
            hogp_kwargs.update(mask=mask, bounds=flat_bounds)
        loss, nit, _, status, message, _ = _native_fit_backend(mll).fit(
            max_attempts=1, **hogp_kwargs
        )
        if getattr(mll, "_rs_em_fit", False):
            mll.model._sync_em_state(force=True)
        return OptimizationResult(
            step=int(nit), fval=float(loss), status=OptimizationStatus[status], message=message
        )
    model = _native_model(mll)
    native = (
        model is not None
        and closure is None
        and closure_kwargs is None
        and method == "L-BFGS-B"
        and callback is None
    )
    batched = model is not None and math.prod(model._aug_batch_shape) > 1
    if native and batched:
        result = _fit_scipy_independent(mll, model, parameters, bounds, options, timeout_sec)
    elif native:
        opts = dict(options or {})
        kwargs: dict[str, Any] = {}
        for key in ("maxiter", "maxfun", "maxcor", "maxls"):
            if key in opts:
                kwargs[key] = int(opts.pop(key))
        for key in ("ftol", "gtol"):
            if key in opts:
                kwargs[key] = float(opts.pop(key))
        for key in ("disp", "iprint", "eps", "finite_diff_rel_step"):
            opts.pop(key, None)
        if opts:
            warnings.warn(
                f"Unknown solver options: {', '.join(sorted(opts))}", UserWarning, stacklevel=2
            )
        mask, flat_bounds = _mask_and_bounds(mll, parameters, bounds)
        result = _result(
            _rs.fit_gpytorch_mll_scipy_py(
                model._rs, mask=mask, bounds=flat_bounds, timeout_sec=timeout_sec, **kwargs
            )
        )
    else:
        result = _fit_generic_scipy(
            mll, parameters, bounds, closure, closure_kwargs, method, options, callback, timeout_sec
        )
    if result.status not in (OptimizationStatus.SUCCESS, OptimizationStatus.STOPPED):
        warnings.warn(
            f"`scipy_minimize` terminated with status {result.status}, displaying"
            f" original message from `scipy.optimize.minimize`: {result.message}",
            OptimizationWarning,
            stacklevel=2,
        )
    return result


def _fit_scipy_independent(
    mll: Any,
    model: Any,
    parameters: dict[str, Any] | None,
    bounds: dict[str, tuple[float | None, float | None]] | None,
    options: dict[str, Any] | None,
    timeout_sec: float | None,
) -> OptimizationResult:
    """BoTorch's ``_fit_gpytorch_mll_scipy_independent``."""
    if timeout_sec is not None:
        warnings.warn(
            "timeout_sec is not supported for batched independent fitting and will be ignored.",
            OptimizationWarning,
            stacklevel=3,
        )
    kwargs: dict[str, Any] = {}
    if options is not None:
        recognized = {"gtol", "maxiter", "maxcor", "ftol", "pgtol", "maxls", "factr"}
        for key, value in options.items():
            if key in ("gtol", "pgtol"):
                kwargs["gtol"] = float(value)
            elif key == "factr":
                kwargs.setdefault("ftol", float(value) * _EPS)
            elif key == "ftol":
                kwargs["ftol"] = float(value)
            elif key in ("maxiter", "maxcor", "maxls"):
                kwargs[key] = int(value)
        unrecognized = set(options) - recognized
        if unrecognized:
            warnings.warn(
                "Unrecognized options for batched independent fitting will be ignored: "
                f"{sorted(unrecognized)}.",
                OptimizationWarning,
                stacklevel=3,
            )
    mask, flat_bounds = _mask_and_bounds(mll, parameters, bounds)
    return _result(
        _rs.fit_gpytorch_mll_scipy_py(model._rs, mask=mask, bounds=flat_bounds, **kwargs)
    )


def _fit_generic_scipy(
    mll: Any,
    parameters: dict[str, Any] | None,
    bounds: dict[str, tuple[float | None, float | None]] | None,
    closure: Any,
    closure_kwargs: dict[str, Any] | None,
    method: str,
    options: dict[str, Any] | None,
    callback: Any,
    timeout_sec: float | None,
) -> OptimizationResult:
    """BoTorch's scipy fit with a closure (``scipy_minimize``)."""
    from botorch_rs.optim.closures import get_loss_closure_with_grads
    from botorch_rs.optim.utils.model_utils import get_parameters_and_bounds

    _parameters, _bounds = get_parameters_and_bounds(mll, requires_grad=True)
    bounds = _bounds if bounds is None else {**_bounds, **bounds}
    if parameters is None:
        parameters = dict(_parameters)
    if closure is None:
        closure = get_loss_closure_with_grads(mll, parameters=parameters)
    if closure_kwargs is not None:
        closure = functools.partial(closure, **closure_kwargs)
    return scipy_minimize(
        closure=closure,
        parameters=parameters,
        bounds=bounds,
        method=method,
        options=options,
        callback=callback,
        timeout_sec=timeout_sec,
    )


def _adam_kwargs(optimizer: Any) -> dict[str, Any] | None:
    """The settings of a native Adam (``None`` for other optimizers):
    ``torch.optim.Adam`` or a ``functools.partial`` of it."""
    if optimizer is None:
        return {}
    func, kwargs = optimizer, {}
    if isinstance(optimizer, functools.partial):
        if optimizer.args:
            return None
        func, kwargs = optimizer.func, dict(optimizer.keywords)
    if getattr(func, "__name__", None) != "Adam" or "torch.optim" not in getattr(
        func, "__module__", ""
    ):
        return None
    allowed = {"lr", "betas", "eps", "weight_decay", "amsgrad", "maximize"}
    if set(kwargs) - allowed:
        return None
    out: dict[str, Any] = {}
    for key, value in kwargs.items():
        if key == "betas":
            out[key] = (float(value[0]), float(value[1]))
        elif key in ("amsgrad", "maximize"):
            out[key] = bool(value)
        else:
            out[key] = float(value)
    return out


def fit_gpytorch_mll_torch(
    mll: Any,
    parameters: dict[str, Any] | None = None,
    bounds: dict[str, tuple[float | None, float | None]] | None = None,
    closure: Callable[[], tuple[Any, Sequence[Any]]] | None = None,
    closure_kwargs: dict[str, Any] | None = None,
    step_limit: int | None = None,
    stopping_criterion: Any = DEFAULT,
    optimizer: Any = None,
    scheduler: Any = None,
    callback: Callable[[dict[str, Any], OptimizationResult], None] | None = None,
    timeout_sec: float | None = None,
) -> OptimizationResult:
    r"""Generic torch.optim-based fitting routine for marginal log
    likelihoods (BoTorch's ``fit_gpytorch_mll_torch``).

    With Adam (``optimizer=None``, ``torch.optim.Adam`` or a
    ``functools.partial`` of it; torch's default learning rate 1e-3), the
    default closure, no scheduler and no callback, the loop runs natively in
    Rust with torch's Adam update and BoTorch's ``ExpMAStoppingCriterion``.

    Args:
        mll: The marginal log likelihood to be maximized.
        parameters: Optional dictionary of parameters to be optimized (by
            name). Defaults to all parameters.
        bounds: A dictionary of user-specified bounds for ``parameters``.
        closure: An optional closure returning the loss and gradients.
        closure_kwargs: Keyword arguments passed to ``closure``.
        step_limit: Optional upper bound on the number of optimization steps.
        stopping_criterion: A StoppingCriterion for the optimization loop
            (default ``ExpMAStoppingCriterion()``).
        optimizer: A torch optimizer class or factory (default Adam).
        scheduler: A learning-rate scheduler or a factory taking the
            optimizer.
        callback: Optional callback taking ``parameters`` and an
            ``OptimizationResult``.
        timeout_sec: Timeout in seconds after which to terminate the loop.

    Returns:
        The final OptimizationResult.
    """
    if stopping_criterion is DEFAULT:
        stopping_criterion = ExpMAStoppingCriterion()
    if getattr(getattr(mll, "model", None), "_tensor_backend", None) == "torch":
        from botorch_rs.optim._torch_backend import fit_torch

        return fit_torch(
            mll,
            parameters,
            bounds,
            closure,
            closure_kwargs,
            step_limit,
            stopping_criterion,
            optimizer,
            scheduler,
            callback,
            timeout_sec,
        )
    model = _native_model(mll)
    adam = _adam_kwargs(optimizer)
    if _is_higher_order(mll):
        native = (
            closure is None
            and closure_kwargs is None
            and adam is not None
            and scheduler is None
            and callback is None
            and (stopping_criterion is None or type(stopping_criterion) is ExpMAStoppingCriterion)
        )
        if native:
            assert adam is not None
            if step_limit is None and stopping_criterion is None and timeout_sec is None:
                raise RuntimeError("No termination conditions were given.")
            mask, flat_bounds = _mask_and_bounds(mll, parameters, bounds)
            if not getattr(mll.model, "_learn_latent_pars", True):
                if mask is None:
                    mask = [True] * len(mll.model._rs.raw_parameters())
                for name, (off, size) in mll.model._offsets().items():
                    if "latent_parameters" in name:
                        mask[off : off + size] = [False] * size
            stopping = None if stopping_criterion is None else stopping_criterion._native_args()
            result = _result(
                _native_fit_backend(mll).fit_torch(
                    mask=mask,
                    bounds=flat_bounds,
                    step_limit=step_limit,
                    stopping=stopping,
                    timeout_sec=timeout_sec,
                    **adam,
                )
            )
            if getattr(mll, "_rs_em_fit", False):
                mll.model._sync_em_state(force=True)
            return result
        if getattr(mll, "_rs_shared_fit", False) or getattr(mll, "_rs_em_fit", False):
            raise UnsupportedError(
                "EM and shared-GP Adam fitting require the native closure and stopping criterion, with no scheduler or callback."
            )
    if _is_approximate(mll):
        native = (
            closure is None
            and closure_kwargs is None
            and adam is not None
            and scheduler is None
            and callback is None
            and (stopping_criterion is None or type(stopping_criterion) is ExpMAStoppingCriterion)
        )
        if not native:
            raise UnsupportedError(
                "botorch_rs fits a variational ELBO with Adam (torch.optim.Adam or a "
                "functools.partial of it), the default closure and ExpMAStoppingCriterion."
            )
        assert adam is not None
        if step_limit is None and stopping_criterion is None and timeout_sec is None:
            raise RuntimeError("No termination conditions were given.")
        mll.model._outer._ensure_initialized()
        mask, flat_bounds = _mask_and_bounds(mll, parameters, bounds)
        stopping = None if stopping_criterion is None else stopping_criterion._native_args()
        result = _result(
            mll._rs.fit_torch(
                float(mll.num_data),
                float(mll.beta),
                mask=mask,
                bounds=flat_bounds,
                step_limit=step_limit,
                stopping=stopping,
                timeout_sec=timeout_sec,
                **adam,
            )
        )
        mll.model._outer._sync_torch_parameters()
        return result
    stopping_native = (
        stopping_criterion is None or type(stopping_criterion) is ExpMAStoppingCriterion
    )
    if (
        model is not None
        and closure is None
        and closure_kwargs is None
        and adam is not None
        and scheduler is None
        and callback is None
        and stopping_native
    ):
        if step_limit is None and stopping_criterion is None and timeout_sec is None:
            raise RuntimeError("No termination conditions were given.")
        mask, flat_bounds = _mask_and_bounds(mll, parameters, bounds)
        stopping = None if stopping_criterion is None else stopping_criterion._native_args()
        return _result(
            _rs.fit_gpytorch_mll_torch_py(
                model._rs,
                mask=mask,
                bounds=flat_bounds,
                step_limit=step_limit,
                stopping=stopping,
                timeout_sec=timeout_sec,
                **adam,
            )
        )
    return _fit_generic_torch(
        mll,
        parameters,
        bounds,
        closure,
        closure_kwargs,
        step_limit,
        stopping_criterion,
        optimizer,
        scheduler,
        callback,
        timeout_sec,
    )


def _fit_generic_torch(
    mll: Any,
    parameters: dict[str, Any] | None,
    bounds: dict[str, tuple[float | None, float | None]] | None,
    closure: Any,
    closure_kwargs: dict[str, Any] | None,
    step_limit: int | None,
    stopping_criterion: Any,
    optimizer: Any,
    scheduler: Any,
    callback: Any,
    timeout_sec: float | None,
) -> OptimizationResult:
    """BoTorch's torch fit on torch copies of the raw hyperparameters (the
    natively computed loss and gradient are written into ``grad``)."""
    import torch

    from botorch_rs.optim.closures import (
        NdarrayOptimizationClosure,
        get_loss_closure_with_grads,
    )
    from botorch_rs.optim.utils.model_utils import get_parameters_and_bounds

    param_dict, bounds_dict = get_parameters_and_bounds(mll)
    names = list(param_dict if parameters is None else parameters)
    state: Any = None
    if closure is None:
        native = get_loss_closure_with_grads(mll, {n: param_dict[n] for n in names})
        state = NdarrayOptimizationClosure(native, {n: param_dict[n] for n in names})
        tensors = {
            n: torch.tensor(param_dict[n], dtype=torch.float64).clone().requires_grad_(True)
            for n in names
        }

        def tensor_closure(**kwargs: Any) -> tuple[Any, tuple[Any, ...]]:
            flat = torch.cat([t.detach().reshape(-1) for t in tensors.values()])
            state.state = flat.tolist()
            value, grads = native(**kwargs)
            out = []
            for t, g in zip(tensors.values(), grads, strict=True):
                gt = torch.as_tensor(g, dtype=t.dtype).reshape(t.shape)
                t.grad = gt
                out.append(gt)
            return torch.as_tensor(value, dtype=torch.float64), tuple(out)

        closure = tensor_closure
        opt_parameters: dict[str, Any] = tensors
    else:
        opt_parameters = dict(parameters or {})
    if closure_kwargs is not None:
        closure = functools.partial(closure, **closure_kwargs)
    result = torch_minimize(
        closure=closure,
        parameters=opt_parameters,
        bounds=bounds_dict if bounds is None else {**bounds_dict, **bounds},
        optimizer=optimizer if optimizer is not None else torch.optim.Adam,
        scheduler=scheduler,
        step_limit=step_limit,
        stopping_criterion=stopping_criterion,
        callback=callback,
        timeout_sec=timeout_sec,
    )
    if state is not None:
        # Write the final values into the model.
        flat = torch.cat([t.detach().reshape(-1) for t in opt_parameters.values()])
        state.state = flat.tolist()
    return result


__all__ = ["fit_gpytorch_mll_scipy", "fit_gpytorch_mll_torch"]

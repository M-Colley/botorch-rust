"""Model fitting (``botorch.fit``)."""

from __future__ import annotations

import math
import warnings
from typing import Any

from botorch_rs.exceptions.errors import UnsupportedError
from botorch_rs.exceptions.warnings import OptimizationWarning
from botorch_rs.mlls import (
    ExactMarginalLogLikelihood,
    MarginalLogLikelihood,
    SumMarginalLogLikelihood,
)

# scipy L-BFGS-B options understood by the fit (``optimizer_kwargs["options"]``).
_LBFGSB_OPTIONS = {"maxiter", "maxfun", "ftol", "gtol", "maxcor", "maxls"}
_IGNORED_OPTIONS = {"disp", "iprint"}


def fit_gpytorch_mll(
    mll: MarginalLogLikelihood,
    closure: Any = None,
    optimizer: Any = None,
    closure_kwargs: dict[str, Any] | None = None,
    optimizer_kwargs: dict[str, Any] | None = None,
    **kwargs: Any,
) -> MarginalLogLikelihood:
    r"""Fits the hyperparameters of ``mll.model`` by maximizing the exact
    marginal log likelihood with L-BFGS-B (a port of L-BFGS-B 3.0 that
    reproduces scipy's iterates; BoTorch's ``fit_gpytorch_mll_scipy`` defaults).

    For a model with several outputs or a batch shape, every output and batch
    element is fitted as an independent L-BFGS-B problem (BoTorch's batched
    independent fitting with ``fmin_l_bfgs_b_batched``, which does not use
    ``maxfun``). For a ``SumMarginalLogLikelihood`` of a ``ModelListGP``, every
    sub-model is fitted on its own, like BoTorch.

    As in BoTorch, an attempt that fails (line-search failure, a covariance
    that is not positive definite) is retried from hyperparameters sampled from
    their priors, up to ``max_attempts`` times (default 5); if all attempts
    fail, ``ModelFittingError`` is raised and the model keeps its
    hyperparameters. Hitting the iteration limit is not a failure. A
    hyperparameter leaving the support of its prior (e.g. a ``UniformPrior``)
    raises ``ValueError``, as in BoTorch, which retries only ``NotPSDError``.

    Args:
        mll: An ``ExactMarginalLogLikelihood`` of a ``SingleTaskGP`` or a
            ``SumMarginalLogLikelihood`` of a ``ModelListGP``.
        closure, optimizer, closure_kwargs: Must be ``None`` (only the default
            L-BFGS-B optimizer is available).
        optimizer_kwargs: ``{"options": {...}}`` with any of scipy's L-BFGS-B
            options ``maxiter`` (default 15000), ``maxfun`` (15000), ``ftol``
            (2.22e-9), ``gtol`` (1e-5), ``maxcor`` (10) and ``maxls`` (20).
        **kwargs: ``max_attempts``; ``seed`` (botorch_rs only) seeds the prior
            samples of retries (default 0). Other keyword arguments are
            ignored, as by BoTorch's fallback fitting routine.

    Returns:
        ``mll``, in evaluation mode.
    """
    if isinstance(mll, SumMarginalLogLikelihood):
        mll.train()
        results = []
        for sub_mll in mll.mlls:
            fit_gpytorch_mll(
                sub_mll,
                closure=closure,
                optimizer=optimizer,
                closure_kwargs=closure_kwargs,
                optimizer_kwargs=optimizer_kwargs,
                **kwargs,
            )
            results.append(sub_mll._fit_result)
        mll._fit_result = results
        return mll.eval()
    if not isinstance(mll, ExactMarginalLogLikelihood) and not getattr(mll, "_rs_fit", False):
        raise UnsupportedError(
            "fit_gpytorch_mll requires a botorch_rs ExactMarginalLogLikelihood or "
            "SumMarginalLogLikelihood."
        )
    if closure is not None or closure_kwargs is not None:
        raise UnsupportedError("Custom closures are not supported by botorch_rs.")
    if optimizer is not None and getattr(optimizer, "__name__", None) == "fit_gpytorch_mll_torch":
        if not isinstance(mll, ExactMarginalLogLikelihood):
            raise UnsupportedError(
                "fit_gpytorch_mll_torch is supported for ExactMarginalLogLikelihood only in "
                "botorch_rs; fit a PairwiseGP with the default L-BFGS-B optimizer."
            )
        return _fit_with_optimizer(mll, optimizer, optimizer_kwargs, kwargs)
    if optimizer is not None and getattr(optimizer, "__name__", None) != "fit_gpytorch_mll_scipy":
        raise UnsupportedError(
            "botorch_rs fits with fit_gpytorch_mll_scipy (L-BFGS-B, the default) or "
            "fit_gpytorch_mll_torch."
        )
    opt_kwargs = dict(optimizer_kwargs or {})
    method = opt_kwargs.pop("method", "L-BFGS-B")
    if method != "L-BFGS-B":
        raise UnsupportedError(f"Only method='L-BFGS-B' is supported, got {method!r}.")
    options = dict(opt_kwargs.pop("options", None) or {})
    for key in ("bounds", "parameters", "callback", "timeout_sec", "closure_kwargs"):
        if opt_kwargs.pop(key, None) is not None:
            raise UnsupportedError(f"optimizer_kwargs[{key!r}] is not supported by botorch_rs.")
    if opt_kwargs:
        raise UnsupportedError(f"Unsupported optimizer_kwargs: {sorted(opt_kwargs)}.")
    unknown = set(options) - _LBFGSB_OPTIONS - _IGNORED_OPTIONS
    if unknown:
        raise UnsupportedError(f"Unsupported L-BFGS-B options: {sorted(unknown)}.")
    if kwargs.pop("pick_best_of_all_attempts", False):
        raise UnsupportedError("pick_best_of_all_attempts=True is not supported by botorch_rs.")
    batched = math.prod(getattr(mll.model, "_aug_batch_shape", ())) > 1
    if batched and "maxfun" in options:
        warnings.warn(
            "Unrecognized options for batched independent fitting will be ignored: ['maxfun'].",
            OptimizationWarning,
            stacklevel=2,
        )
    max_attempts = int(kwargs.pop("max_attempts", 5))
    seed = kwargs.pop("seed", None)
    result = mll.model._rs.fit(
        maxcor=options.get("maxcor"),
        pgtol=options.get("gtol"),
        ftol=options.get("ftol"),
        maxiter=options.get("maxiter"),
        maxfun=options.get("maxfun"),
        maxls=options.get("maxls"),
        max_attempts=max_attempts,
        seed=None if seed is None else int(seed),
    )
    loss, iterations, evaluations, status, message, attempts, elements = result
    mll._fit_result = {
        "fval": loss,
        "nit": iterations,
        "nfev": evaluations,
        "status": status,
        "message": message,
        "attempts": attempts,
        "elements": [
            {
                "fval": e[0],
                "nit": e[1],
                "nfev": e[2],
                "status": e[3],
                "message": e[4],
                "attempts": e[5],
            }
            for e in elements
        ],
    }
    return mll.eval()


def _fit_with_optimizer(
    mll: ExactMarginalLogLikelihood,
    optimizer: Any,
    optimizer_kwargs: dict[str, Any] | None,
    kwargs: dict[str, Any],
) -> MarginalLogLikelihood:
    """BoTorch's ``_fit_fallback`` with ``fit_gpytorch_mll_torch``: attempts
    that raise a numerical error are retried from hyperparameters sampled from
    their priors, up to ``max_attempts`` times."""
    from botorch_rs.exceptions.errors import ModelFittingError, NotPSDError
    from botorch_rs.optim.fit import fit_gpytorch_mll_torch
    from botorch_rs.optim.utils.model_utils import sample_all_priors

    max_attempts = int(kwargs.pop("max_attempts", 5))
    seed = kwargs.pop("seed", None)
    raw = list(mll.model._rs.raw_parameters())
    for attempt in range(1, max_attempts + 1):
        if attempt > 1:
            sample_all_priors(mll.model, seed=None if seed is None else int(seed) + attempt)
        mll.train()
        try:
            with warnings.catch_warnings(record=True) as ws:
                warnings.simplefilter("always", category=OptimizationWarning)
                result = fit_gpytorch_mll_torch(mll, **(optimizer_kwargs or {}))
        except NotPSDError:
            mll.model._rs.set_raw_parameters(raw)
            continue
        failed = [w for w in ws if issubclass(w.category, OptimizationWarning)]
        for w in ws:
            warnings.warn_explicit(str(w.message), w.category, w.filename, w.lineno)
        if not failed:
            mll._fit_result = result
            return mll.eval()
        mll.model._rs.set_raw_parameters(raw)
    raise ModelFittingError("All attempts to fit the model have failed.")


__all__ = ["fit_gpytorch_mll"]

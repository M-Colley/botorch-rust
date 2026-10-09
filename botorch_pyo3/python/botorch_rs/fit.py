"""Model fitting (``botorch.fit``)."""

from __future__ import annotations

from typing import Any

from botorch_rs.exceptions.errors import UnsupportedError
from botorch_rs.mlls import ExactMarginalLogLikelihood, MarginalLogLikelihood

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

    As in BoTorch, an attempt that fails (line-search failure, a covariance
    that is not positive definite) is retried from hyperparameters sampled from
    their priors, up to ``max_attempts`` times (default 5); if all attempts
    fail, ``ModelFittingError`` is raised and the model keeps its
    hyperparameters. Hitting the iteration limit is not a failure.

    Args:
        mll: An ``ExactMarginalLogLikelihood`` of a ``SingleTaskGP``.
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
    if not isinstance(mll, ExactMarginalLogLikelihood):
        raise UnsupportedError("fit_gpytorch_mll requires a botorch_rs ExactMarginalLogLikelihood.")
    if closure is not None or closure_kwargs is not None:
        raise UnsupportedError("Custom closures are not supported by botorch_rs.")
    if optimizer is not None:
        raise UnsupportedError(
            "Only the default L-BFGS-B optimizer is available in botorch_rs (optimizer=None)."
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
    loss, iterations, evaluations, status, message, attempts = result
    mll._fit_result = {
        "fval": loss,
        "nit": iterations,
        "nfev": evaluations,
        "status": status,
        "message": message,
        "attempts": attempts,
    }
    return mll.eval()


__all__ = ["fit_gpytorch_mll"]

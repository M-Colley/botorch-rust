"""Model fitting (``botorch.fit``)."""

from __future__ import annotations

import math
import warnings
from typing import Any, cast

from botorch_rs._arrays import to_flat
from botorch_rs.exceptions.errors import UnsupportedError
from botorch_rs.exceptions.warnings import OptimizationWarning
from botorch_rs.mlls import (
    ExactMarginalLogLikelihood,
    MarginalLogLikelihood,
    SumMarginalLogLikelihood,
    _ApproximateMarginalLogLikelihood,
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
    from botorch_rs.models.empirical_gps.em_coordinate_ascent import KLPriorFitMLL
    from botorch_rs.models.empirical_gps.em_empirical_gp import EMEmpiricalMarginalLogLikelihood
    from botorch_rs.models.empirical_gps.shared_gp import _SharedSumMarginalLogLikelihood
    from botorch_rs.models.robust_relevance_pursuit_model import RobustRelevancePursuitMixin

    if getattr(getattr(mll, "model", None), "_tensor_backend", None) == "torch" and isinstance(
        mll.model, RobustRelevancePursuitMixin
    ):
        return mll.model.custom_fit(
            mll,
            closure=closure,
            optimizer=optimizer,
            closure_kwargs=closure_kwargs,
            optimizer_kwargs=optimizer_kwargs,
            **kwargs,
        )

    if getattr(getattr(mll, "model", None), "_tensor_backend", None) == "torch":
        from botorch_rs.optim.fit import fit_gpytorch_mll_scipy, fit_gpytorch_mll_torch

        if getattr(mll.model, "_is_fully_bayesian", False):
            raise UnsupportedError(
                "Fully Bayesian Torch models use loaded MCMC samples, not MAP fitting."
            )
        selected = fit_gpytorch_mll_scipy if optimizer is None else optimizer
        if selected not in (fit_gpytorch_mll_scipy, fit_gpytorch_mll_torch):
            raise UnsupportedError(
                "Choose fit_gpytorch_mll_scipy or fit_gpytorch_mll_torch for the Torch backend."
            )
        mll.train()
        result = selected(
            mll, closure=closure, closure_kwargs=closure_kwargs, **dict(optimizer_kwargs or {})
        )
        mll._fit_result = {
            "fval": result.fval,
            "nit": result.step,
            "status": result.status.name,
            "message": result.message,
            "runtime": result.runtime,
        }
        return mll.eval()

    if isinstance(
        mll, (KLPriorFitMLL, EMEmpiricalMarginalLogLikelihood, _SharedSumMarginalLogLikelihood)
    ):
        return mll._fit(closure, optimizer, closure_kwargs, optimizer_kwargs, **kwargs)

    if isinstance(getattr(mll, "model", None), RobustRelevancePursuitMixin):
        return cast(
            MarginalLogLikelihood,
            mll.model.custom_fit(
                mll,
                closure=closure,
                optimizer=optimizer,
                closure_kwargs=closure_kwargs,
                optimizer_kwargs=optimizer_kwargs,
                **kwargs,
            ),
        )
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
    if isinstance(mll, _ApproximateMarginalLogLikelihood):
        return _fit_fallback_approximate(
            mll, closure, optimizer, closure_kwargs, optimizer_kwargs, kwargs
        )
    if not isinstance(mll, ExactMarginalLogLikelihood) and not getattr(mll, "_rs_fit", False):
        raise UnsupportedError(
            "fit_gpytorch_mll requires a botorch_rs ExactMarginalLogLikelihood or "
            "SumMarginalLogLikelihood."
        )
    if closure is not None or closure_kwargs is not None:
        raise UnsupportedError("Custom closures are not supported by botorch_rs.")
    if kwargs.pop("pick_best_of_all_attempts", False):
        from botorch_rs._fit_attempts import fit_all_attempts
        from botorch_rs.models.gp_regression import SingleTaskGP

        if not isinstance(mll.model, SingleTaskGP):
            raise UnsupportedError(
                "Best-of-all fitting currently requires a SingleTaskGP family model."
            )
        return fit_all_attempts(mll, optimizer, optimizer_kwargs, kwargs)
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
    from botorch_rs._fit_attempts import frozen_raw_ranges

    if frozen_raw_ranges(mll.model):
        from botorch_rs.optim.fit import fit_gpytorch_mll_scipy

        return _fit_with_optimizer(mll, fit_gpytorch_mll_scipy, optimizer_kwargs, kwargs)
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
    if len(result) == 6:
        loss, iterations, evaluations, status, message, attempts = result
        elements = []
    else:
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
    mll: Any,
    optimizer: Any,
    optimizer_kwargs: dict[str, Any] | None,
    kwargs: dict[str, Any],
) -> MarginalLogLikelihood:
    """BoTorch's ``_fit_fallback`` with a native optimizer: attempts
    that raise a numerical error are retried from hyperparameters sampled from
    their priors, up to ``max_attempts`` times."""
    from botorch_rs._fit_attempts import frozen_raw_ranges, restore_frozen
    from botorch_rs.exceptions.errors import ModelFittingError, NotPSDError
    from botorch_rs.optim.utils.model_utils import sample_all_priors

    max_attempts = int(kwargs.pop("max_attempts", 5))
    if max_attempts < 1:
        from botorch_rs.exceptions.errors import InvalidArgumentError

        raise InvalidArgumentError("max_attempts must be at least 1")
    seed = kwargs.pop("seed", None)
    frozen = frozen_raw_ranges(mll.model)
    raw = list(mll.model._rs.raw_parameters())
    if frozen and sum(end - start for start, end in frozen) == len(raw):
        from botorch_rs.optim.core import OptimizationResult, OptimizationStatus

        mll._fit_result = OptimizationResult(
            step=0,
            fval=-sum(to_flat(mll())[0]),
            status=OptimizationStatus.SUCCESS,
            message="All model parameters are fixed.",
        )
        return mll.eval()
    for attempt in range(1, max_attempts + 1):
        if attempt > 1:
            sample_all_priors(mll.model, seed=None if seed is None else int(seed) + attempt)
            restore_frozen(mll.model, raw, frozen)
        mll.train()
        try:
            with warnings.catch_warnings(record=True) as ws:
                warnings.simplefilter("always", category=OptimizationWarning)
                result = optimizer(mll, **(optimizer_kwargs or {}))
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


def fit_fully_bayesian_model_nuts(
    model: Any,
    max_tree_depth: int = 6,
    warmup_steps: int = 512,
    num_samples: int = 256,
    thinning: int = 16,
    disable_progbar: bool = False,
    jit_compile: bool = False,
    seed: int = 0,
    *,
    draws: Any = None,
    init_params: Any = None,
) -> None:
    r"""Fit a fully Bayesian model using the No-U-Turn-Sampler (NUTS)

    Runs botorch_rs's port of NumPyro's NUTS (as BoTorch: a dense mass
    matrix, step-size and mass-matrix adaptation during the warmup, initial
    values ``init_to_uniform(radius=2)``) on the model's potential energy,
    evaluated in Rust in double precision on the native backend, or using
    device-resident Tensor algebra and autograd on the Torch backend.

    Args:
        model: Fully Bayesian GP to be fitted.
        max_tree_depth: Maximum tree depth for NUTS
        warmup_steps: The number of burn-in steps for NUTS.
        num_samples:  The number of MCMC samples. Note that with thinning,
            num_samples / thinning samples are retained.
        thinning: The amount of thinning. Every nth sample is retained.
        disable_progbar: A boolean indicating whether to print the progress
            bar and diagnostics during MCMC (ignored: botorch_rs prints none).
        jit_compile: Whether to use jit (ignored on the compiled native backend;
            True is explicitly unsupported on the eager Torch backend).
        seed: Random seed. botorch_rs draws its random numbers from a seeded
            xoshiro256++ generator, not from JAX's PRNG, so the samples
            differ from BoTorch's for the same seed. The Torch backend uses a
            private, device-local Torch generator, preserving global RNG state.
        draws: (botorch_rs extension) the random draws of a NumPyro run to
            replay, ``("n", values)`` (standard-normal momenta) and
            ``("u", u, p)`` (the uniform draws of NumPyro's Bernoulli trials),
            with ``init_params``: then the chain is NumPyro's (see
            ``botorch_pyo3/tests/fully_bayesian_helpers.py``).
        init_params: (botorch_rs extension) the initial unconstrained
            parameters, in NumPyro's order (sites sorted by name).

    Example:
        >>> gp = SaasFullyBayesianSingleTaskGP(train_X, train_Y)
        >>> fit_fully_bayesian_model_nuts(gp)
    """
    from botorch_rs._arrays import from_flat

    if getattr(model, "_tensor_backend", None) == "torch":
        from botorch_rs.models._torch_backend_mcmc import fit_torch_fully_bayesian_nuts

        return fit_torch_fully_bayesian_nuts(
            model,
            max_tree_depth,
            warmup_steps,
            num_samples,
            thinning,
            jit_compile,
            seed,
            draws,
            init_params,
        )
    model.train()
    potential = model.pyro_model._potential()
    if (draws is None) != (init_params is None):
        raise ValueError("`draws` and `init_params` must be given together.")
    z0 = None if init_params is None else [float(v) for v in init_params]
    result = potential.sample_nuts(
        int(warmup_steps),
        int(num_samples),
        max_tree_depth=(int(max_tree_depth), int(max_tree_depth)),
        seed=int(seed) & 0xFFFFFFFFFFFFFFFF,
        z0=z0,
        draws=None if draws is None else list(draws),
    )
    samples = result["samples"]
    mcmc_samples = {}
    for name, shape, values in potential.postprocess(samples):
        size = math.prod(shape)
        rows = [values[i * size : (i + 1) * size] for i in range(len(samples))][::thinning]
        flat = [v for row in rows for v in row]
        mcmc_samples[name] = from_flat(flat, (len(rows), *shape), model._kind)
    model._nuts_result = result
    model.load_mcmc_samples(mcmc_samples)
    model.eval()


def get_fitted_map_saas_model(
    train_X: Any,
    train_Y: Any,
    train_Yvar: Any = None,
    input_transform: Any = None,
    outcome_transform: Any = None,
    tau: Any = None,
    optimizer_kwargs: dict[str, Any] | None = None,
    *,
    backend: str = "auto",
) -> Any:
    """Get a fitted MAP SAAS model with a Matern kernel.

    Args:
        train_X: Tensor of shape ``n x d`` with training inputs.
        train_Y: Tensor of shape ``n x 1`` with training targets.
        train_Yvar: Optional tensor of shape ``n x 1`` with observed noise,
            inferred if None.
        input_transform: An optional input transform.
        outcome_transform: An optional outcome transform.
        tau: Fixed value of the global shrinkage tau. If None, the model
            places a HC(0.1) prior on tau. Can be a tensor for batched models
            where each batch has a different sparsity prior.
        optimizer_kwargs: A dict of options for the optimizer passed
            to fit_gpytorch_mll.

    Returns:
        A fitted SingleTaskGP with a Matern kernel.
    """
    from botorch_rs.models.map_saas import get_map_saas_model

    # make sure optimizer_kwargs is a Dict
    optimizer_kwargs = optimizer_kwargs or {}
    model = get_map_saas_model(
        train_X=train_X,
        train_Y=train_Y,
        train_Yvar=train_Yvar,
        input_transform=(input_transform.train() if input_transform is not None else None),
        outcome_transform=outcome_transform,
        tau=tau,
        backend=backend,
    )
    mll = ExactMarginalLogLikelihood(model=model, likelihood=model.likelihood)
    fit_gpytorch_mll(mll, optimizer_kwargs=optimizer_kwargs)
    return model


def _scipy_options(optimizer_kwargs: dict[str, Any] | None) -> dict[str, Any]:
    """The L-BFGS-B options of ``optimizer_kwargs`` (BoTorch's
    ``fit_gpytorch_mll_scipy`` keyword arguments)."""
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
    return options


def _fit_fallback_approximate(
    mll: _ApproximateMarginalLogLikelihood,
    closure: Any,
    optimizer: Any,
    closure_kwargs: dict[str, Any] | None,
    optimizer_kwargs: dict[str, Any] | None,
    kwargs: dict[str, Any],
) -> MarginalLogLikelihood:
    """BoTorch's ``_fit_fallback_approximate``: L-BFGS-B
    (``fit_gpytorch_mll_scipy``) if the model has at most ``full_batch_limit``
    (default 1024) training targets, else Adam (``fit_gpytorch_mll_torch``),
    on the full training data, with ``_fit_fallback``'s retries from prior
    samples of the kernel and mean parameters."""
    from botorch_rs.exceptions.errors import ModelFittingError, NotPSDError

    data_loader = kwargs.pop("data_loader", None)
    full_batch_limit = int(kwargs.pop("full_batch_limit", 1024))
    if data_loader is not None:
        if closure is not None:
            raise UnsupportedError("Only one of `data_loader` or `closure` may be passed.")
        raise UnsupportedError(
            "Minibatch training with a `data_loader` is not supported by botorch_rs."
        )
    if closure is not None or closure_kwargs is not None:
        raise UnsupportedError("Custom closures are not supported by botorch_rs.")
    if kwargs.pop("pick_best_of_all_attempts", False):
        raise UnsupportedError("pick_best_of_all_attempts=True is not supported by botorch_rs.")
    # BoTorch's `len(mll.model.train_targets)` (AttributeError without targets).
    n = int(to_flat(mll.model.train_targets, "train_targets")[1][0])
    name = getattr(optimizer, "__name__", None)
    if optimizer is None:
        name = "fit_gpytorch_mll_scipy" if n <= full_batch_limit else "fit_gpytorch_mll_torch"
    max_attempts = int(kwargs.pop("max_attempts", 5))
    seed = kwargs.pop("seed", None)
    outer = mll.model._outer
    mll.train()
    outer._ensure_initialized()
    from botorch_rs.optim.fit import _mask_and_bounds

    mask, flat_bounds = _mask_and_bounds(mll, None, None)
    if name == "fit_gpytorch_mll_scipy":
        options = _scipy_options(optimizer_kwargs)
        loss, iterations, evaluations, status, message, attempts = mll._rs.fit(
            float(mll.num_data),
            float(mll.beta),
            maxcor=options.get("maxcor"),
            pgtol=options.get("gtol"),
            ftol=options.get("ftol"),
            maxiter=options.get("maxiter"),
            maxfun=options.get("maxfun"),
            maxls=options.get("maxls"),
            max_attempts=max_attempts,
            seed=None if seed is None else int(seed),
            mask=mask,
            bounds=flat_bounds,
        )
        outer._sync_torch_parameters()
        mll._fit_result = {
            "fval": loss,
            "nit": iterations,
            "nfev": evaluations,
            "status": status,
            "message": message,
            "attempts": attempts,
        }
        return mll.eval()
    if name != "fit_gpytorch_mll_torch":
        raise UnsupportedError(
            "botorch_rs fits with fit_gpytorch_mll_scipy (L-BFGS-B) or fit_gpytorch_mll_torch."
        )
    from botorch_rs.optim.fit import fit_gpytorch_mll_torch

    raw = list(mll._rs.raw_parameters())
    for attempt in range(1, max_attempts + 1):
        if attempt > 1:
            mll._rs.sample_all_priors(None if seed is None else int(seed) + attempt)
            if mask is not None:
                from botorch_rs._fit_attempts import restore_frozen

                restore_frozen(outer, raw, [(i, i + 1) for i, free in enumerate(mask) if not free])
        mll.train()
        try:
            with warnings.catch_warnings(record=True) as ws:
                warnings.simplefilter("always", category=OptimizationWarning)
                result = fit_gpytorch_mll_torch(mll, **(optimizer_kwargs or {}))
        except NotPSDError:
            mll._rs.set_raw_parameters(raw)
            outer._sync_torch_parameters()
            continue
        failed = [w for w in ws if issubclass(w.category, OptimizationWarning)]
        for w in ws:
            warnings.warn_explicit(str(w.message), w.category, w.filename, w.lineno)
        if not failed:
            mll._fit_result = result
            return mll.eval()
        mll._rs.set_raw_parameters(raw)
        outer._sync_torch_parameters()
    raise ModelFittingError("All attempts to fit the model have failed.")


__all__ = ["fit_fully_bayesian_model_nuts", "fit_gpytorch_mll", "get_fitted_map_saas_model"]

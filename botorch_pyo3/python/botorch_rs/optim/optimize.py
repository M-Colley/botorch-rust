"""Acquisition-function optimization (``botorch.optim.optimize``)."""

from __future__ import annotations

import dataclasses
import functools
import math
import warnings
from collections.abc import Callable, Mapping
from typing import Any

from botorch_rs import _botorch_rs as _rs
from botorch_rs._arrays import (
    ArrayKind,
    Flat,
    as_float,
    first_kind,
    from_flat,
    from_pair,
    is_torch_tensor,
    kind_of,
    to_flat,
)
from botorch_rs.acquisition.acquisition import AcquisitionFunction
from botorch_rs.exceptions.errors import (
    BotorchTensorDimensionError,
    InputDataError,
    UnsupportedError,
)
from botorch_rs.exceptions.warnings import BadInitialCandidatesWarning, OptimizationWarning
from botorch_rs.optim.parameter_constraints import (
    linear_constraints_arg,
    nonlinear_constraints_arg,
)

_EPS = 2.220446049250313e-16

#: ``options`` entries of the initialization (BoTorch's ``INIT_OPTION_KEYS``,
#: completed with the options that BoTorch reads in the initialization but
#: forgets to remove from the optimizer's options: ``n_thinning``, ``topn``,
#: ``sorted``, ``largest``, ``sample_around_best_subset_sigma`` and the
#: one-shot settings).
_INIT_KEYS = {
    "alpha",
    "batch_limit",
    "eta",
    "init_batch_limit",
    "nonnegative",
    "n_burnin",
    "n_thinning",
    "sample_around_best",
    "sample_around_best_sigma",
    "sample_around_best_subset_sigma",
    "sample_around_best_prob_perturb",
    "seed",
    "topn",
    "sorted",
    "largest",
    # The one-shot initial conditions (gen_one_shot_(hv)kg_initial_conditions).
    "frac_random",
    "num_inner_restarts",
    "raw_inner_samples",
}
#: Options accepted by BoTorch's parallel L-BFGS-B path
#: (``get_reasons_against_fast_path``).
_FAST_KEYS = {"maxiter", "disp", "iprint", "max_cor", "ftol", "pgtol", "factr", "tol", "maxls"}
#: scipy's L-BFGS-B options (the serial path).
_SCIPY_LBFGSB_KEYS = {"maxcor", "ftol", "gtol", "maxiter", "maxfun", "maxls"}
_IGNORED_KEYS = {"disp", "iprint", "eps", "finite_diff_rel_step"}
#: One L-BFGS-B problem per chunk in the serial path.
_ALL = 2**62

_WARNINGS: dict[str, type[Warning]] = {
    "BadInitialCandidatesWarning": BadInitialCandidatesWarning,
    "RuntimeWarning": RuntimeWarning,
    "OptimizationWarning": OptimizationWarning,
}


def _warning_class(message: str) -> type[Warning]:
    """The warning category of a message of the extension."""
    if message.startswith(
        (
            "Candidate generation failed for",
            "Requested q=",
            "Failed to initialize using continuous relaxation",
        )
    ):
        return OptimizationWarning
    if "selected randomly" in message or "Choosing initial conditions at random" in message:
        return BadInitialCandidatesWarning
    return RuntimeWarning


def _emit(messages: Any, stacklevel: int = 3) -> None:
    """Emits the extension's warnings (messages or ``(category, message)``)."""
    for m in messages:
        if isinstance(m, tuple):
            category, message = m
            cls = _WARNINGS.get(category, RuntimeWarning)
            if cls is RuntimeWarning:
                cls = _warning_class(message)
        else:
            message, cls = m, _warning_class(m)
        warnings.warn(message, cls, stacklevel=stacklevel)


def _is_gen(func: Any, name: str) -> bool:
    """Whether ``func`` is BoTorch's or botorch_rs's ``gen_candidates_*``."""
    if isinstance(func, functools.partial):
        func = func.func
    return getattr(func, "__name__", None) == name and getattr(func, "__module__", "") in (
        "botorch.generation.gen",
        "botorch_rs.generation.gen",
    )


def _torch_options(gen_options: dict[str, Any], partial_kw: dict[str, Any]) -> dict[str, Any]:
    """The native settings of ``gen_candidates_torch`` from BoTorch's
    ``options`` (``optimizer_options``, ``stopping_criterion_options`` and the
    deprecated ``maxiter``)."""
    out: dict[str, Any] = {}
    optimizer = partial_kw.get("optimizer")
    if optimizer is not None and getattr(optimizer, "__name__", None) != "Adam":
        raise UnsupportedError(
            "botorch_rs's native gen_candidates_torch supports the Adam optimizer only."
        )
    opt = dict(gen_options.get("optimizer_options", {}) or {})
    stop = dict(gen_options.get("stopping_criterion_options", {}) or {})
    if "maxiter" in gen_options:
        warnings.warn(
            "Passing 'maxiter' directly in options is deprecated. "
            "Please use options['stopping_criterion_options']['maxiter'] instead.",
            DeprecationWarning,
            stacklevel=4,
        )
        stop.setdefault("maxiter", gen_options["maxiter"])
    for key, value in opt.items():
        if key == "betas":
            out["betas"] = (float(value[0]), float(value[1]))
        elif key in ("amsgrad", "maximize"):
            out[key] = bool(value)
        elif key in ("lr", "eps", "weight_decay"):
            out[key] = float(value)
        else:
            raise UnsupportedError(f"Unsupported Adam option {key!r} in botorch_rs.")
    names = {"maxiter": "stop_maxiter", "minimize": "stop_minimize", "eta": "stop_eta"}
    for key, value in stop.items():
        if key not in ("maxiter", "minimize", "n_window", "eta", "rel_tol"):
            raise TypeError(f"ExpMAStoppingCriterion got an unexpected keyword argument {key!r}")
        target = names.get(key) or key
        if key in ("maxiter", "n_window"):
            out[target] = int(value)
        elif key == "minimize":
            out[target] = bool(value)
        else:
            out[target] = float(value)
    return out


def _parse_options(
    options: dict[str, Any] | None,
    *,
    constrained: bool,
    timeout_sec: float | None,
    torch_gen: bool,
    warn_unknown: bool = False,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """BoTorch's ``options`` as the extension's keyword arguments.

    Options that SciPy does not know are ignored; with ``warn_unknown`` (for
    ``gen_candidates_scipy``) with SciPy's warning (``optimize_acqf``
    swallows it, like BoTorch's, which records the warnings of the optimizer
    and only re-raises optimization warnings).

    Returns ``(kwargs, init, gen_options)``: the keyword arguments, the
    initialization entries (for ``sample_around_best`` and ``topn``, handled
    by the caller) and the optimizer's options (BoTorch's ``gen_options``).
    """
    opts = dict(options or {})
    init = {k: opts.pop(k) for k in list(opts) if k in _INIT_KEYS}
    out: dict[str, Any] = {}
    for key in ("seed", "batch_limit", "init_batch_limit", "n_burnin", "n_thinning"):
        if init.get(key) is not None:
            out[key] = int(init[key])
    for key in ("eta", "alpha", "frac_random"):
        if init.get(key) is not None:
            out[key] = float(init[key])
    for key in ("num_inner_restarts", "raw_inner_samples"):
        if init.get(key) is not None:
            out[key] = int(init[key])
    if init.get("nonnegative"):
        out["nonnegative"] = True
    if init.get("eta") is not None:
        # BoTorch's one-shot initial conditions read ``eta`` too (default 2).
        out["one_shot_eta"] = float(init["eta"])
    gen_options = dict(opts)
    if torch_gen:
        return out, init, gen_options
    method = opts.pop("method", None)
    if method is not None:
        if method not in ("L-BFGS-B", "SLSQP"):
            raise UnsupportedError(
                f"Only method='L-BFGS-B' and method='SLSQP' are supported, got {method!r}."
            )
        out["method"] = method
    if not opts.pop("with_grad", True):
        raise UnsupportedError("with_grad=False is not supported (exact gradients are used).")
    if opts.pop("callback", None) is not None:
        raise UnsupportedError("An optimizer `callback` is not supported by botorch_rs.")
    aggregation = opts.pop("max_optimization_problem_aggregation_size", None)
    if constrained or method == "SLSQP":
        for key in list(opts):
            value = opts.pop(key)
            if key == "maxiter":
                out["maxiter"] = int(value)
            elif key == "ftol":
                out["ftol"] = float(value)
            elif key not in _IGNORED_KEYS and warn_unknown:
                warnings.warn(f"Unknown solver options: {key}", UserWarning, stacklevel=4)
        return out, init, gen_options
    fast = set(opts) <= _FAST_KEYS and timeout_sec is None
    if fast:
        # BoTorch's parallel path (`fmin_l_bfgs_b_batched`); `ftol` and
        # `max_cor` work here, unlike in BoTorch (an AssertionError and a
        # TypeError).
        if "maxiter" in opts:
            out["maxiter"] = int(opts["maxiter"])
        if opts.get("ftol") is not None:
            out["ftol"] = float(opts["ftol"])
        elif opts.get("factr") is not None:
            out["ftol"] = float(opts["factr"]) * _EPS
        gtol = opts.get("tol", opts.get("pgtol"))
        if gtol is not None:
            out["gtol"] = float(gtol)
        if opts.get("max_cor") is not None:
            out["maxcor"] = int(opts["max_cor"])
        if opts.get("maxls") is not None:
            out["maxls"] = int(opts["maxls"])
        return out, init, gen_options
    # BoTorch's serial path: `scipy.optimize.minimize` on groups of restarts.
    out["lbfgsb_aggregation"] = _ALL if aggregation is None else int(aggregation)
    out["maxfun"] = 15000
    unknown = []
    for key, value in opts.items():
        if key in ("maxiter", "maxfun", "maxcor", "maxls"):
            out[key] = int(value)
        elif key in ("ftol", "gtol"):
            out[key] = float(value)
        elif key not in _IGNORED_KEYS:
            unknown.append(key)
    if unknown and warn_unknown:
        warnings.warn(
            f"Unknown solver options: {', '.join(sorted(unknown))}", UserWarning, stacklevel=4
        )
    if timeout_sec is not None:
        out["timeout_sec"] = float(timeout_sec)
    return out, init, gen_options


def _post_processing_arg(post_processing_func: Any, kind: ArrayKind) -> Any:
    """``post_processing_func`` as a callable on ``(data, shape)``."""
    if post_processing_func is None:
        return None

    def post_process(data: Any, shape: list[int]) -> Flat:
        X = from_flat(data, tuple(shape), kind)
        return to_flat(post_processing_func(X), "post_processing_func output")

    return post_process


def _custom_generator(
    gen_candidates: Any,
    acq_function: Any,
    bounds: Any,
    kind: ArrayKind,
    gen_options: dict[str, Any],
    constraint_kwargs: dict[str, Any],
) -> Any:
    """A user's ``gen_candidates`` as a callable on ``(data, shape)``."""
    bdata, bshape = to_flat(bounds, "bounds")
    d = bshape[1]
    lower = bdata[:d]
    upper = bdata[d:]
    lower_bounds = None if all(v == -math.inf for v in lower) else from_flat(lower, (d,), kind)
    upper_bounds = None if all(v == math.inf for v in upper) else from_flat(upper, (d,), kind)

    def generate(
        data: Any, shape: list[int], timeout_sec: float | None, fixed: dict[int, float]
    ) -> Any:
        ics = from_flat(data, tuple(shape), kind)
        candidates, values = gen_candidates(
            ics,
            acq_function,
            lower_bounds=lower_bounds,
            upper_bounds=upper_bounds,
            options=gen_options,
            fixed_features=fixed or None,
            timeout_sec=timeout_sec,
            **constraint_kwargs,
        )
        return to_flat(candidates, "candidates"), [float(v) for v in to_flat(values, "values")[0]]

    return generate


def _wrap_ic_generator(
    ic_generator: Any,
    acq_function: Any,
    bounds: Any,
    fixed_features: dict[int, float] | None,
    options: dict[str, Any] | None,
    inequality_constraints: Any,
    equality_constraints: Any,
    ic_gen_kwargs: dict[str, Any],
) -> Any:
    """``generate(q, num_restarts, raw_samples) -> (data, shape)`` calling the
    user's ``ic_generator`` with BoTorch's keyword arguments."""

    def generate(q: int, num_restarts: int, raw_samples: int | None) -> Flat:
        X = ic_generator(
            acq_function=acq_function,
            bounds=bounds,
            q=q,
            num_restarts=num_restarts,
            raw_samples=raw_samples,
            fixed_features=fixed_features,
            options=options or {},
            inequality_constraints=inequality_constraints,
            equality_constraints=equality_constraints,
            **ic_gen_kwargs,
        )
        return to_flat(X, "initial conditions")

    return generate


def _as_native(acq_function: Any, d: int) -> AcquisitionFunction:
    """A botorch_rs acquisition function, or a torch callable wrapped as one."""
    if isinstance(acq_function, AcquisitionFunction):
        return acq_function
    if not callable(acq_function):
        raise UnsupportedError(
            "optimize_acqf requires a botorch_rs acquisition function or a callable on torch "
            f"tensors (got {type(acq_function).__name__})."
        )
    from botorch_rs.acquisition.composite import CallableAcquisitionFunction

    return CallableAcquisitionFunction(acq_function, d)


def _check_bounds(bounds: Any) -> Flat:
    bounds_flat = to_flat(bounds, "bounds")
    if len(bounds_flat[1]) != 2 or bounds_flat[1][0] != 2:
        raise BotorchTensorDimensionError(
            f"bounds should be a `2 x d` tensor, current shape: {list(bounds_flat[1])}."
        )
    return bounds_flat


def _fixed_arg(fixed_features: Mapping[int, Any] | None, d: int) -> dict[int, float] | None:
    if fixed_features is None:
        return None
    return {int(k) % d: as_float(v, f"fixed_features[{k}]") for k, v in fixed_features.items()}


def _optimize_kwargs(
    acq_function: Any,
    native_acqf: AcquisitionFunction,
    bounds: Any,
    d: int,
    kind: ArrayKind,
    options: dict[str, Any] | None,
    inequality_constraints: Any,
    equality_constraints: Any,
    nonlinear_inequality_constraints: Any,
    fixed: dict[int, float] | None,
    post_processing_func: Any,
    gen_candidates: Any,
    ic_generator: Any,
    timeout_sec: float | None,
    retry_on_optimization_warning: bool,
    ic_gen_kwargs: dict[str, Any],
) -> dict[str, Any]:
    """The extension's keyword arguments of an ``optimize_acqf`` call."""
    from botorch_rs.acquisition.acquisition import OneShotAcquisitionFunction
    from botorch_rs.optim import initializers

    one_shot = isinstance(native_acqf, OneShotAcquisitionFunction)
    if one_shot and ic_generator in (
        initializers.gen_one_shot_hvkg_initial_conditions,
        initializers.gen_one_shot_kg_initial_conditions,
    ):
        # The defaults of one-shot acquisition functions, generated natively.
        ic_generator = None
    native_generator = ic_generator is initializers.gen_batch_initial_conditions
    if ic_gen_kwargs and (ic_generator is None or native_generator):
        raise UnsupportedError(f"Unsupported keyword arguments: {sorted(ic_gen_kwargs)}.")
    ineq = linear_constraints_arg(inequality_constraints, "inequality_constraints")
    eq = linear_constraints_arg(equality_constraints, "equality_constraints")
    nonlinear = nonlinear_constraints_arg(nonlinear_inequality_constraints)
    constrained = bool(ineq or eq or nonlinear)
    torch_gen = gen_candidates is not None and _is_gen(gen_candidates, "gen_candidates_torch")
    kwargs, init, gen_options = _parse_options(
        options, constrained=constrained, timeout_sec=timeout_sec, torch_gen=torch_gen
    )
    if kwargs.get("method") == "L-BFGS-B" and constrained:
        raise RuntimeWarning("Method L-BFGS-B cannot handle constraints.")
    if gen_candidates is not None and not _is_gen(gen_candidates, "gen_candidates_scipy"):
        if torch_gen:
            if constrained:
                raise TypeError(
                    "gen_candidates_torch() got an unexpected keyword argument "
                    "'inequality_constraints'"
                )
            partial_kw = (
                gen_candidates.keywords if isinstance(gen_candidates, functools.partial) else {}
            )
            kwargs["gen_candidates"] = "torch"
            kwargs.update(_torch_options(gen_options, partial_kw))
            if timeout_sec is not None:
                kwargs["timeout_sec"] = float(timeout_sec)
        else:
            constraint_kwargs = {
                k: v
                for k, v in (
                    ("inequality_constraints", inequality_constraints),
                    ("equality_constraints", equality_constraints),
                    ("nonlinear_inequality_constraints", nonlinear_inequality_constraints),
                )
                if v is not None
            }
            kwargs["gen_candidates"] = _custom_generator(
                gen_candidates, acq_function, bounds, kind, gen_options, constraint_kwargs
            )
            if timeout_sec is not None:
                kwargs["timeout_sec"] = float(timeout_sec)
    if init.get("topn"):
        kwargs["topn"] = True
        kwargs["largest"] = bool(init.get("largest", getattr(acq_function, "maximize", True)))
        kwargs["sorted"] = bool(init.get("sorted", True))
    if init.get("sample_around_best"):
        if equality_constraints:
            raise UnsupportedError(
                "Option 'sample_around_best' is not supported when equalityconstraints are present."
            )
        best = initializers._best_points(acq_function, d)
        if best is not None:
            kwargs["sample_around_best"] = best
            for key in (
                "sample_around_best_sigma",
                "sample_around_best_subset_sigma",
                "sample_around_best_prob_perturb",
            ):
                if init.get(key) is not None:
                    kwargs[key] = float(init[key])
    if ic_generator is not None and not native_generator:
        kwargs["ic_generator"] = _wrap_ic_generator(
            ic_generator,
            acq_function,
            bounds,
            fixed,
            options,
            inequality_constraints,
            equality_constraints,
            ic_gen_kwargs,
        )
    kwargs["native_ic_generator"] = native_generator
    kwargs["fixed_features"] = fixed
    kwargs["inequality_constraints"] = ineq
    kwargs["equality_constraints"] = eq
    kwargs["nonlinear_inequality_constraints"] = nonlinear
    kwargs["retry_on_optimization_warning"] = bool(retry_on_optimization_warning)
    kwargs["post_processing"] = _post_processing_arg(post_processing_func, kind)
    return kwargs


def optimize_acqf(
    acq_function: Any,
    bounds: Any,
    q: int,
    num_restarts: int,
    raw_samples: int | None = None,
    options: dict[str, Any] | None = None,
    inequality_constraints: list[tuple[Any, Any, float]] | None = None,
    equality_constraints: list[tuple[Any, Any, float]] | None = None,
    nonlinear_inequality_constraints: list[tuple[Any, bool]] | None = None,
    fixed_features: dict[int, Any] | None = None,
    post_processing_func: Callable[[Any], Any] | None = None,
    batch_initial_conditions: Any = None,
    return_best_only: bool = True,
    gen_candidates: Any = None,
    sequential: bool = False,
    acq_function_sequence: list[Any] | None = None,
    *,
    ic_generator: Any = None,
    timeout_sec: float | None = None,
    return_acq_values: bool = True,
    return_full_tree: bool = False,
    retry_on_optimization_warning: bool = True,
    **ic_gen_kwargs: Any,
) -> tuple[Any, Any]:
    r"""Optimizes ``acq_function`` over the box ``bounds`` (``2 x d``), like
    BoTorch's ``optimize_acqf``.

    ``acq_function`` is a botorch_rs acquisition function, or (like in
    BoTorch) any callable on torch tensors, e.g. a ``torch.nn.Module`` whose
    ``forward(X)`` maps ``batch x q x d`` candidate sets to ``batch`` values:
    it is evaluated from Rust with gradients from ``torch.autograd`` (see
    :class:`~botorch_rs.acquisition.composite.CallableAcquisitionFunction`;
    :class:`~botorch_rs.acquisition.composite.WeightedLogMeanExp` evaluates
    weighted blends of log-scale acquisition functions natively).

    ``raw_samples`` scrambled Sobol candidate sets are evaluated and
    ``num_restarts`` of them are selected as starting points (Boltzmann
    sampling, the non-negative heuristic for non-negative acquisition
    functions, or the top values with ``options["topn"]``; with
    ``options["sample_around_best"]`` points sampled around the best
    baseline points are added). Without parameter constraints, every start is
    optimized by an independent L-BFGS-B run with exact gradients (in
    parallel), BoTorch's parallel path; with ``timeout_sec`` or options
    outside that path's (``maxfun``, ``maxcor``, ``gtol``, ...), groups of
    restarts are optimized as one problem, as in BoTorch's serial path
    (``options["max_optimization_problem_aggregation_size"]``).
    ``gen_candidates=gen_candidates_torch`` optimizes with Adam natively.

    With linear ``inequality_constraints`` / ``equality_constraints``
    (BoTorch's ``(indices, coefficients, rhs)`` tuples, intra- or
    inter-point) or ``nonlinear_inequality_constraints`` (``(callable,
    is_intrapoint)`` with ``callable(x) >= 0`` on torch tensors), the raw
    samples come from a hit-and-run sampler of the feasible polytope and
    chunks of ``batch_limit`` restarts are optimized jointly with SLSQP, as
    in BoTorch's ``gen_candidates_scipy``; infeasible results are projected
    onto the feasible set (``project_to_feasible_space_via_slsqp``) and a
    ``CandidateGenerationError`` is raised if no feasible candidate remains.
    Nonlinear constraints require ``batch_initial_conditions`` or an
    ``ic_generator``.

    ``post_processing_func`` is applied to the candidates of all restarts
    (``b x q x d``, the array type of ``bounds``) before the best one is
    chosen; their values are recomputed.

    Supported ``options``: ``seed`` (raw samples and initial-condition
    selection; BoTorch draws the latter from torch's global RNG),
    ``batch_limit``, ``init_batch_limit``, ``eta``, ``alpha``,
    ``nonnegative``, ``topn`` (with ``largest`` and ``sorted``),
    ``sample_around_best`` (with ``sample_around_best_sigma``,
    ``sample_around_best_subset_sigma`` and
    ``sample_around_best_prob_perturb``), ``method`` (``"L-BFGS-B"`` or
    ``"SLSQP"``), ``maxiter`` (2000), ``ftol``, ``factr``, ``gtol`` /
    ``pgtol`` / ``tol``, ``maxcor`` / ``max_cor``, ``maxls``, ``maxfun``,
    ``max_optimization_problem_aggregation_size``, the polytope sampler's
    ``n_burnin`` (10000) and ``n_thinning`` (32), the one-shot initial
    conditions' ``frac_random``, ``num_inner_restarts`` and
    ``raw_inner_samples``, and for ``gen_candidates_torch``
    ``optimizer_options`` and ``stopping_criterion_options``.

    One-shot acquisition functions (``qKnowledgeGradient``,
    ``qHypervolumeKnowledgeGradient`` and their multi-fidelity variants) are
    optimized over augmented candidate sets (``get_augmented_q_batch_size``),
    starting by default from ``gen_one_shot_kg_initial_conditions`` /
    ``gen_one_shot_hvkg_initial_conditions``; the candidates are extracted
    (``extract_candidates``) unless ``return_full_tree=True``.

    ``acq_function_sequence`` selects a separate acquisition for each of the
    ``q`` sequential steps (requires ``sequential=True`` and ``q > 1``).
    Each retains its own base pending points, adds earlier candidates during
    its step, and has its original pending state restored, including errors.
    Supply per-step initial conditions with ``ic_generator``; a single
    ``batch_initial_conditions`` array is unsupported in sequential mode.

    Returns:
        ``(candidates, acq_value)``: a ``q x d`` candidate set (``num_restarts x
        q x d`` with ``return_best_only=False``) and its acquisition value (a
        scalar; ``q`` values for ``sequential=True``; one per restart with
        ``return_best_only=False``), of the array type of ``bounds``.
    """
    if getattr(acq_function, "_tensor_backend", None) == "torch" or (
        acq_function_sequence
        and getattr(acq_function_sequence[0], "_tensor_backend", None) == "torch"
    ):
        from botorch_rs.optim._torch_acq_optim import optimize

        return optimize(locals())
    if acq_function_sequence is not None:
        return _optimize_acqf_sequence(
            OptimizeAcqfInputs(
                acq_function=acq_function,
                bounds=bounds,
                q=q,
                num_restarts=num_restarts,
                raw_samples=raw_samples,
                options=options,
                inequality_constraints=inequality_constraints,
                equality_constraints=equality_constraints,
                nonlinear_inequality_constraints=nonlinear_inequality_constraints,
                fixed_features=fixed_features,
                post_processing_func=post_processing_func,
                batch_initial_conditions=batch_initial_conditions,
                return_best_only=return_best_only,
                gen_candidates=gen_candidates,
                sequential=sequential,
                acq_function_sequence=acq_function_sequence,
                ic_generator=ic_generator,
                timeout_sec=timeout_sec,
                return_acq_values=return_acq_values,
                return_full_tree=return_full_tree,
                retry_on_optimization_warning=retry_on_optimization_warning,
                ic_gen_kwargs=ic_gen_kwargs,
            )
        )
    if acq_function is None:
        raise ValueError("Either `acq_function` or `acq_function_sequence` must be specified.")
    if sequential and not return_best_only:
        raise UnsupportedError("`return_best_only=False` only supported for joint optimization.")
    kind = first_kind(bounds, batch_initial_conditions)
    bounds_flat = _check_bounds(bounds)
    d = bounds_flat[1][1]
    user_acq_function = acq_function
    native_acqf = _as_native(acq_function, d)
    from botorch_rs.acquisition.acquisition import OneShotAcquisitionFunction

    one_shot = isinstance(native_acqf, OneShotAcquisitionFunction)
    fixed = _fixed_arg(fixed_features, d)
    ics = None
    if batch_initial_conditions is not None:
        ics = to_flat(batch_initial_conditions, "batch_initial_conditions")
        shape = ics[1]
        if len(shape) not in (2, 3):
            raise ValueError(
                "batch_initial_conditions must be 2-dimensional or 3-dimensional. "
                f"Its shape is {tuple(shape)}."
            )
        if shape[-1] != d:
            raise ValueError(
                f"batch_initial_conditions.shape[-1] must be {d}. The shape is {tuple(shape)}."
            )
        if len(shape) == 2 and raw_samples is not None:
            warnings.warn(
                "If using a 2-dim `batch_initial_conditions` botorch will "
                "default to old behavior of ignoring `num_restarts` and just "
                "use the given `batch_initial_conditions` by setting "
                "`raw_samples` to None.",
                RuntimeWarning,
                stacklevel=2,
            )
            raw_samples = None
        if (
            len(shape) == 3
            and shape[0] < num_restarts
            and shape[-2] != q
            and raw_samples is not None
        ):
            warnings.warn(
                "If using a 3-dim `batch_initial_conditions` where the "
                "first dimension is less than `num_restarts` and the second "
                "dimension is not equal to `q`, botorch will default to "
                "old behavior of ignoring `num_restarts` and just use the "
                "given `batch_initial_conditions` by setting `raw_samples` "
                "to None.",
                RuntimeWarning,
                stacklevel=2,
            )
            raw_samples = None
    kwargs = _optimize_kwargs(
        user_acq_function,
        native_acqf,
        bounds,
        d,
        kind,
        options,
        inequality_constraints,
        equality_constraints,
        nonlinear_inequality_constraints,
        fixed,
        post_processing_func,
        gen_candidates,
        ic_generator,
        timeout_sec,
        retry_on_optimization_warning,
        ic_gen_kwargs,
    )
    candidates, acq_values, restart_x, restart_values, messages = _rs.optimize_acqf(
        native_acqf._rs,
        bounds_flat,
        int(q),
        int(num_restarts),
        None if raw_samples is None else int(raw_samples),
        batch_initial_conditions=ics,
        sequential=bool(sequential),
        **kwargs,
    )
    _emit(messages)
    native_acqf._after_evaluation()
    full_tree = return_full_tree and one_shot
    if return_best_only:
        if full_tree:
            X = _best_full_tree(candidates, acq_values[0], restart_x, restart_values, kind)
        else:
            X = from_pair(candidates, kind)
        if sequential:
            values = from_flat(acq_values, (len(acq_values),), kind)
        else:
            values = from_flat(acq_values[:1], (), kind)
    else:
        if not restart_values:
            # All features fixed: one "restart" (BoTorch's shapes).
            X = from_flat(candidates[0], (1, *candidates[1]), kind)
            values = from_flat(acq_values[:1], (1,), kind)
        else:
            X = from_pair(restart_x, kind)
            if one_shot and not full_tree:
                X = native_acqf.extract_candidates(X)  # type: ignore[attr-defined]
            values = from_flat(restart_values, (len(restart_values),), kind)
    return X, (values if return_acq_values else None)


def _optimize_acqf_sequence(inputs: OptimizeAcqfInputs) -> tuple[Any, Any]:
    """Optimize one real acquisition per step with independent pending state."""
    from botorch_rs.acquisition.acquisition import OneShotAcquisitionFunction

    for name, constraints in (
        ("linear inequality", inputs.inequality_constraints),
        ("linear equality", inputs.equality_constraints),
    ):
        for i, (indices, _, _) in enumerate(constraints or []):
            if len(to_flat(indices)[1]) > 1:
                raise UnsupportedError(
                    f"Inter-point constraints are not supported for sequential optimization. But the {i}th {name} constraint is defined as inter-point."
                )
    for i, (_, intra_point) in enumerate(inputs.nonlinear_inequality_constraints or []):
        if not intra_point:
            raise UnsupportedError(
                f"Inter-point constraints are not supported for sequential optimization. But the {i}th non-linear inequality constraint is defined as inter-point."
            )
    if inputs.batch_initial_conditions is not None:
        raise UnsupportedError(
            "`batch_initial_conditions` is not supported for sequential optimization. Use `ic_generator` to supply initial conditions for each q=1 step."
        )
    if not inputs.return_best_only:
        raise NotImplementedError("`return_best_only=False` only supported for joint optimization.")
    assert inputs.acq_function_sequence is not None
    sequence = inputs.acq_function_sequence
    if any(isinstance(acq, OneShotAcquisitionFunction) for acq in sequence):
        raise NotImplementedError(
            "sequential optimization currently not supported for one-shot acquisition functions. Must have `sequential=False`."
        )
    if any(not callable(getattr(acq, "set_X_pending", None)) for acq in sequence):
        raise UnsupportedError(
            "Every acquisition in acq_function_sequence must support set_X_pending."
        )
    pending = [getattr(acq, "X_pending", None) for acq in sequence]
    kind = first_kind(inputs.bounds)
    d = _check_bounds(inputs.bounds)[1][1]
    pending_kinds = [
        kind if isinstance(acq, AcquisitionFunction) else _as_native(acq, d)._pending_kind
        for acq in sequence
    ]
    candidates: list[Any] = []
    values: list[float] = []
    step_kwargs = {field.name: getattr(inputs, field.name) for field in dataclasses.fields(inputs)}
    step_kwargs.update(step_kwargs.pop("ic_gen_kwargs"))
    step_kwargs.update(
        q=1,
        sequential=False,
        acq_function_sequence=None,
        batch_initial_conditions=None,
        return_best_only=True,
        timeout_sec=None if inputs.timeout_sec is None else inputs.timeout_sec / inputs.q,
    )
    failed = False
    try:
        for i, acq in enumerate(sequence):
            if candidates:
                parts = ([pending[i]] if pending[i] is not None else []) + candidates
                acq.set_X_pending(_concat_rows(parts, pending_kinds[i]))
            step_kwargs["acq_function"] = acq
            candidate, value = optimize_acqf(**step_kwargs)
            candidates.append(candidate)
            if inputs.return_acq_values:
                values.append(float(to_flat(value)[0][0]))
    except BaseException:
        failed = True
        raise
    finally:
        # Official currently only restores after successful completion. Always
        # restore here, including exceptions in initialization or evaluation.
        restore_error = None
        for acq, original in zip(sequence, pending, strict=True):
            try:
                acq.set_X_pending(original)
            except Exception as error:
                if restore_error is None:
                    restore_error = error
        if restore_error is not None and not failed:
            raise restore_error
    return _concat_rows(candidates, kind), (
        from_flat(values, (len(values),), kind) if inputs.return_acq_values else None
    )


def _best_full_tree(
    candidates: Any, value: float, restart_x: Any, restart_values: list[float], kind: Any
) -> Any:
    """The augmented candidate set of the best restart: the first restart
    with the returned value whose leading points are the returned
    candidates."""
    data, shape = restart_x
    per = shape[-2] * shape[-1]
    n = len(candidates[0])
    best = None
    for i, v in enumerate(restart_values):
        block = data[i * per : (i + 1) * per]
        if v == value and list(block[:n]) == list(candidates[0]):
            best = i
            break
    if best is None:
        finite = [(v, -i) for i, v in enumerate(restart_values) if v == v]
        best = -max(finite)[1] if finite else 0
    return from_flat(data[best * per : (best + 1) * per], tuple(shape[1:]), kind)


def optimize_acqf_cyclic(
    acq_function: AcquisitionFunction,
    bounds: Any,
    q: int,
    num_restarts: int,
    raw_samples: int | None = None,
    options: dict[str, Any] | None = None,
    inequality_constraints: list[tuple[Any, Any, float]] | None = None,
    equality_constraints: list[tuple[Any, Any, float]] | None = None,
    fixed_features: dict[int, Any] | None = None,
    post_processing_func: Callable[[Any], Any] | None = None,
    batch_initial_conditions: Any = None,
    cyclic_options: dict[str, Any] | None = None,
    *,
    ic_generator: Any = None,
    timeout_sec: float | None = None,
    return_acq_values: bool = True,
    return_full_tree: bool = False,
    retry_on_optimization_warning: bool = True,
    **ic_gen_kwargs: Any,
) -> tuple[Any, Any]:
    r"""Generate a set of ``q`` candidates via cyclic optimization (BoTorch's
    ``optimize_acqf_cyclic``).

    The candidates are first generated sequentially; then, in cycles, every
    candidate is re-optimized from its current value with all other
    candidates as pending points, until BoTorch's
    ``ExpMAStoppingCriterion(minimize=False, **cyclic_options)`` holds on
    the ``q`` values. Maximization is the default, correcting the upstream
    minimization default (PR #3395). The CPU loop runs natively.

    Returns:
        A two-element tuple containing

        - a ``q x d`` array of generated candidates.
        - a ``q`` array of acquisition values, where the value at index ``i``
          is the acquisition value conditional on having observed all
          candidates except candidate ``i`` (``None`` if
          ``return_acq_values=False``).
    """
    if getattr(acq_function, "_tensor_backend", None) == "torch":
        from botorch_rs.optim._torch_acq_discrete import cyclic

        return cyclic(locals())
    from botorch_rs.optim.stopping import ExpMAStoppingCriterion

    kind = first_kind(bounds, batch_initial_conditions)
    bounds_flat = _check_bounds(bounds)
    d = bounds_flat[1][1]
    native_acqf = _as_native(acq_function, d)
    fixed = _fixed_arg(fixed_features, d)
    if batch_initial_conditions is None and ic_generator is None and raw_samples is None:
        raise ValueError("Must specify `raw_samples` when `batch_initial_conditions` is None`.")
    if batch_initial_conditions is not None and q > 1:
        raise UnsupportedError(
            "`batch_initial_conditions` is not supported for sequential "
            "optimization. Either avoid specifying "
            "`batch_initial_conditions` to use the custom initializer or "
            "use the `ic_generator` kwarg to generate initial conditions "
            "for the case of nonlinear inequality constraints."
        )
    criterion = ExpMAStoppingCriterion(**{"minimize": False, **(cyclic_options or {})})
    kwargs = _optimize_kwargs(
        acq_function,
        native_acqf,
        bounds,
        d,
        kind,
        options,
        inequality_constraints,
        equality_constraints,
        None,
        fixed,
        post_processing_func,
        None,
        ic_generator,
        timeout_sec,
        retry_on_optimization_warning,
        ic_gen_kwargs,
    )
    if batch_initial_conditions is not None:
        kwargs["batch_initial_conditions"] = to_flat(
            batch_initial_conditions, "batch_initial_conditions"
        )
    candidates, values, messages = _rs.optimize_acqf_cyclic_py(
        native_acqf._rs,
        bounds_flat,
        int(q),
        int(num_restarts),
        None if raw_samples is None else int(raw_samples),
        criterion._native_args(),
        **kwargs,
    )
    _emit(messages)
    native_acqf._after_evaluation()
    X = from_pair(candidates, kind)
    if not return_acq_values:
        return X, None
    if q == 1:
        return X, from_flat(values[:1], (), kind)
    return X, from_flat(values, (len(values),), kind)


def optimize_acqf_list(
    acq_function_list: list[AcquisitionFunction],
    bounds: Any,
    num_restarts: int,
    raw_samples: int | None = None,
    options: dict[str, Any] | None = None,
    inequality_constraints: list[tuple[Any, Any, float]] | None = None,
    equality_constraints: list[tuple[Any, Any, float]] | None = None,
    nonlinear_inequality_constraints: list[tuple[Any, bool]] | None = None,
    fixed_features: dict[int, Any] | None = None,
    fixed_features_list: list[dict[int, Any]] | None = None,
    post_processing_func: Callable[[Any], Any] | None = None,
    ic_generator: Any = None,
    ic_gen_kwargs: dict[str, Any] | None = None,
    return_acq_values: bool = True,
    retry_on_optimization_warning: bool = True,
) -> tuple[Any, Any]:
    r"""Generate a list of candidates from a list of acquisition functions
    (BoTorch's ``optimize_acqf_list``).

    The acquisition functions are optimized in sequence for ``q = 1`` with
    :func:`optimize_acqf`, each with the previous candidates (appended to the
    current acquisition function's pending points) set as ``X_pending``; this
    is also known as sequential greedy optimization. With
    ``fixed_features_list``, every step is :func:`optimize_acqf_mixed`.

    Returns:
        ``(candidates, acq_values)``: a ``q x d`` array of candidates (one per
        acquisition function) and their ``q`` acquisition values, each
        conditional on the previous candidates (``None`` with
        ``return_acq_values=False``).
    """
    if acq_function_list and getattr(acq_function_list[0], "_tensor_backend", None) == "torch":
        from botorch_rs.optim._torch_acq_discrete import acquisition_list

        return acquisition_list(locals())
    if fixed_features and fixed_features_list:
        raise ValueError(
            "Èither `fixed_feature` or `fixed_features_list` can be provided, not both."
        )
    if not acq_function_list:
        raise ValueError("acq_function_list must be non-empty.")
    kind = first_kind(bounds)
    candidate_list: list[Any] = []
    acq_value_list: list[float] = []
    candidates: Any = None
    originals = [getattr(acq, "X_pending", None) for acq in acq_function_list]
    failed = False
    try:
        for i, acq_function in enumerate(acq_function_list):
            if candidate_list:
                acq_function.set_X_pending(
                    _concat_rows([originals[i], candidates], kind)
                    if originals[i] is not None
                    else candidates
                )
            if fixed_features_list:
                candidate, acq_value = optimize_acqf_mixed(
                    acq_function=acq_function,
                    bounds=bounds,
                    q=1,
                    num_restarts=num_restarts,
                    raw_samples=raw_samples,
                    options=options or {},
                    inequality_constraints=inequality_constraints,
                    equality_constraints=equality_constraints,
                    nonlinear_inequality_constraints=nonlinear_inequality_constraints,
                    fixed_features_list=fixed_features_list,
                    post_processing_func=post_processing_func,
                    ic_generator=ic_generator,
                    ic_gen_kwargs=ic_gen_kwargs,
                    return_acq_values=True,
                    retry_on_optimization_warning=retry_on_optimization_warning,
                )
            else:
                candidate, acq_value = optimize_acqf(
                    acq_function=acq_function,
                    bounds=bounds,
                    q=1,
                    num_restarts=num_restarts,
                    raw_samples=raw_samples,
                    options=options or {},
                    inequality_constraints=inequality_constraints,
                    equality_constraints=equality_constraints,
                    nonlinear_inequality_constraints=nonlinear_inequality_constraints,
                    fixed_features=fixed_features,
                    post_processing_func=post_processing_func,
                    return_best_only=True,
                    sequential=False,
                    ic_generator=ic_generator,
                    retry_on_optimization_warning=retry_on_optimization_warning,
                    **(ic_gen_kwargs or {}),
                )
            candidate_list.append(candidate)
            if return_acq_values:
                acq_value_list.append(float(to_flat(acq_value, "acq_value")[0][0]))
            candidates = _concat_rows(candidate_list, kind)
    except BaseException:
        failed = True
        raise
    finally:
        restore_error = None
        for acquisition, original in zip(acq_function_list, originals, strict=True):
            try:
                acquisition.set_X_pending(original)
            except Exception as error:
                if restore_error is None:
                    restore_error = error
        if restore_error is not None and not failed:
            raise restore_error
    if not return_acq_values:
        return candidates, None
    return candidates, from_flat(acq_value_list, (len(acq_value_list),), kind)


def _concat_rows(parts: list[Any], kind: Any) -> Any:
    """Arrays of shape ``k_i x d`` stacked to ``(sum k_i) x d`` (array type
    ``kind``)."""
    data: list[float] = []
    rows, d = 0, 0
    for p in parts:
        flat, shape = to_flat(p, "X")
        data.extend(flat)
        rows += shape[0]
        d = shape[1]
    return from_flat(data, (rows, d), kind)


def optimize_acqf_mixed(
    acq_function: AcquisitionFunction,
    bounds: Any,
    q: int,
    num_restarts: int,
    fixed_features_list: list[dict[int, Any]],
    raw_samples: int | None = None,
    options: dict[str, Any] | None = None,
    inequality_constraints: list[tuple[Any, Any, float]] | None = None,
    equality_constraints: list[tuple[Any, Any, float]] | None = None,
    nonlinear_inequality_constraints: list[tuple[Any, bool]] | None = None,
    post_processing_func: Callable[[Any], Any] | None = None,
    batch_initial_conditions: Any = None,
    return_best_only: bool = True,
    gen_candidates: Any = None,
    ic_generator: Any = None,
    timeout_sec: float | None = None,
    retry_on_optimization_warning: bool = True,
    ic_gen_kwargs: dict[str, Any] | None = None,
    return_acq_values: bool = True,
) -> tuple[Any, Any]:
    r"""Optimize over a list of fixed_features and return the best solution
    (BoTorch's ``optimize_acqf_mixed``).

    This is useful for optimizing over mixed continuous and discrete domains.
    For ``q = 1`` every element of ``fixed_features_list`` is optimized with
    :func:`optimize_acqf` (natively and in parallel; combinations for which
    candidate generation fails are skipped with an ``OptimizationWarning``)
    and the best restart over all combinations is returned. For ``q > 1``
    this function performs sequential greedy optimization (with proper
    conditioning on generated candidates) and returns the joint value.

    NOTE: This method does not support the kind of "inter-point constraints"
    that are supported by ``optimize_acqf()``.

    Returns:
        A two-element tuple containing

        - an array of generated candidates: ``q x d`` if ``return_best_only``
          (default), else ``num_restarts x q x d`` (for ``q = 1``, the
          restarts of the best fixed features);
        - the associated acquisition value(s) (``None`` if
          ``return_acq_values=False``).
    """
    if getattr(acq_function, "_tensor_backend", None) == "torch":
        from botorch_rs.optim._torch_acq_discrete import mixed

        return mixed(locals())
    kind = first_kind(bounds, batch_initial_conditions)
    bounds_flat = _check_bounds(bounds)
    d = bounds_flat[1][1]
    native_acqf = _as_native(acq_function, d)
    for name, cs in (
        ("linear inequality", inequality_constraints),
        ("linear equality", equality_constraints),
    ):
        for i, (indices, _, _) in enumerate(cs or []):
            if len(to_flat(indices, "indices")[1]) > 1:
                raise UnsupportedError(
                    "Inter-point constraints are not supported for sequential optimization. "
                    f"But the {i}th {name} constraint is defined as inter-point."
                )
    for i, (_, intra_point) in enumerate(nonlinear_inequality_constraints or []):
        if not intra_point:
            raise UnsupportedError(
                "Inter-point constraints are not supported for sequential optimization. "
                f"But the {i}th non-linear inequality constraint is defined as inter-point."
            )
    if not return_best_only and q > 1:
        raise NotImplementedError("`return_best_only=False` is only supported for q=1.")
    if not fixed_features_list:
        raise ValueError("fixed_features_list must be non-empty.")
    ffl = [_fixed_arg(ff, d) or {} for ff in fixed_features_list]
    kwargs = _optimize_kwargs(
        acq_function,
        native_acqf,
        bounds,
        d,
        kind,
        options,
        inequality_constraints,
        equality_constraints,
        nonlinear_inequality_constraints,
        None,
        post_processing_func,
        gen_candidates,
        ic_generator,
        timeout_sec,
        retry_on_optimization_warning,
        ic_gen_kwargs or {},
    )
    kwargs.pop("fixed_features", None)
    if batch_initial_conditions is not None:
        bic = to_flat(batch_initial_conditions, "batch_initial_conditions")
        if len(bic[1]) == 2 and raw_samples is not None:
            warnings.warn(
                "If using a 2-dim `batch_initial_conditions` botorch will "
                "default to old behavior of ignoring `num_restarts` and just "
                "use the given `batch_initial_conditions` by setting "
                "`raw_samples` to None.",
                RuntimeWarning,
                stacklevel=2,
            )
            raw_samples = None
        kwargs["batch_initial_conditions"] = bic
    if timeout_sec is not None and "timeout_sec" not in kwargs:
        kwargs["timeout_sec"] = float(timeout_sec)
    candidates, value, restart_x, restart_values, messages = _rs.optimize_acqf_mixed_py(
        native_acqf._rs,
        bounds_flat,
        int(q),
        int(num_restarts),
        ffl,
        None if raw_samples is None else int(raw_samples),
        bool(return_best_only),
        **kwargs,
    )
    _emit(messages)
    native_acqf._after_evaluation()
    if not return_best_only:
        X = from_pair(restart_x, kind)
        values = from_flat(restart_values, (len(restart_values),), kind)
        return X, (values if return_acq_values else None)
    X = from_pair(candidates, kind)
    if not return_acq_values:
        return X, None
    # q = 1: the best restart's value; q > 1: `acq_function(candidates)` of
    # the `q x d` candidates (shape `(1,)`, as BoTorch's).
    return X, from_flat([value], () if q == 1 else (1,), kind)


def optimize_acqf_discrete(
    acq_function: AcquisitionFunction,
    q: int,
    choices: Any,
    max_batch_size: int = 2048,
    unique: bool = True,
    return_acq_values: bool = True,
    X_avoid: Any = None,
    inequality_constraints: list[tuple[Any, Any, float]] | None = None,
) -> tuple[Any, Any]:
    r"""Optimize over a discrete set of points using batch evaluation
    (BoTorch's ``optimize_acqf_discrete``).

    For ``q > 1`` this function generates candidates by means of sequential
    conditioning (rather than joint optimization), since for all but the
    smallest number of choices the set ``choices^q`` of discrete points to
    evaluate quickly explodes. The choices are evaluated in parallel
    (``max_batch_size`` only bounds BoTorch's memory use).

    Args:
        acq_function: An AcquisitionFunction.
        q: The number of candidates.
        choices: A ``num_choices x d`` array of possible choices.
        max_batch_size: The maximum number of choices to evaluate in batch.
        unique: If True return unique choices, o/w choices may be repeated
            (only relevant if ``q > 1``).
        return_acq_values: Return acquisition values.
        X_avoid: An ``n x d`` array of candidates that we aren't allowed to
            pick. These will be removed from the set of choices.
        inequality_constraints: A list of tuples (indices, coefficients, rhs),
            with each tuple encoding an inequality constraint of the form
            ``sum_i (X[indices[i]] * coefficients[i]) >= rhs``. Infeasible
            points will be removed from the set of choices.

    Returns:
        A two-element tuple containing

        - a ``q x d`` array of generated candidates.
        - an associated acquisition value (``q`` values for ``q > 1``).
    """
    from botorch_rs.acquisition.acquisition import OneShotAcquisitionFunction

    if getattr(acq_function, "_tensor_backend", None) == "torch":
        from botorch_rs.optim._torch_acq_discrete import discrete

        return discrete(locals())
    if isinstance(acq_function, OneShotAcquisitionFunction):
        raise UnsupportedError(
            "Discrete optimization is not supported for one-shot acquisition functions."
        )
    kind = first_kind(choices)
    cflat = to_flat(choices, "choices")
    if len(cflat[1]) != 2:
        raise BotorchTensorDimensionError(
            f"`choices` must be a `num_choices x d` tensor, got shape {list(cflat[1])}."
        )
    native_acqf = _as_native(acq_function, cflat[1][1])
    avoid = None
    if X_avoid is not None:
        avoid = to_flat(X_avoid, "X_avoid")
        if avoid[1] and avoid[1][0] == 0:
            avoid = ([], (0, cflat[1][1]))
    try:
        candidates, values, messages = _rs.optimize_acqf_discrete_py(
            native_acqf._rs,
            int(q),
            cflat,
            unique=bool(unique),
            x_avoid=avoid if avoid is None or avoid[1][0] > 0 else None,
            inequality_constraints=linear_constraints_arg(
                inequality_constraints, "inequality_constraints"
            ),
        )
    except InputDataError as e:
        if X_avoid is not None and "No feasible points remain" not in str(e):
            raise InputDataError(
                str(e) + " No feasible points remain after removing `X_avoid` and "
                "filtering out infeasible points."
            ) from None
        raise
    _emit(messages)
    native_acqf._after_evaluation()
    X = from_pair(candidates, kind)
    if not return_acq_values:
        return X, None
    if len(values) == 1 and candidates[1][0] == 1:
        return X, from_flat(values, (), kind)
    return X, from_flat(values, (len(values),), kind)


def _gen_batch_initial_conditions_local_search(
    discrete_choices: list[Any],
    raw_samples: int,
    X_avoid: Any,
    inequality_constraints: list[tuple[Any, Any, float]],
    min_points: int,
    max_tries: int = 100,
) -> Any:
    """Generate initial conditions for local search (BoTorch's private
    ``_gen_batch_initial_conditions_local_search``): random lattice points,
    deduplicated and sorted, without ``X_avoid`` and infeasible points.

    The native local search calls this function only when it is replaced
    (e.g. to reproduce given random draws); otherwise the points are drawn in
    Rust.
    """
    if discrete_choices and is_torch_tensor(discrete_choices[0]):
        from botorch_rs.optim._torch_acq_discrete import lattice_points

        avoid = (
            discrete_choices[0].new_empty(0, len(discrete_choices)) if X_avoid is None else X_avoid
        )
        return lattice_points(
            discrete_choices, raw_samples, avoid, inequality_constraints, min_points, max_tries
        )
    kind = kind_of(discrete_choices[0]) if discrete_choices else first_kind()
    choices = [[float(v) for v in to_flat(c, "discrete_choices")[0]] for c in discrete_choices]
    avoid = to_flat(X_avoid, "X_avoid") if X_avoid is not None else ([], (0, len(choices)))
    out = _rs.gen_lattice_points_py(
        choices,
        int(raw_samples),
        avoid,
        linear_constraints_arg(inequality_constraints, "inequality_constraints"),
        int(min_points),
        int(max_tries),
    )
    return from_pair(out, kind)


_NATIVE_LATTICE_POINTS = _gen_batch_initial_conditions_local_search


def optimize_acqf_discrete_local_search(
    acq_function: AcquisitionFunction,
    discrete_choices: list[Any],
    q: int,
    num_restarts: int = 20,
    raw_samples: int = 4096,
    inequality_constraints: list[tuple[Any, Any, float]] | None = None,
    X_avoid: Any = None,
    batch_initial_conditions: Any = None,
    max_batch_size: int = 2048,
    max_tries: int = 100,
    unique: bool = True,
    return_acq_values: bool = True,
    *,
    seed: int | None = None,
) -> tuple[Any, Any]:
    r"""Optimize the acquisition function over a lattice (BoTorch's
    ``optimize_acqf_discrete_local_search``).

    This is useful when d is large and enumeration of the search space isn't
    possible. For ``q > 1`` this function always performs sequential greedy
    optimization (with proper conditioning on generated candidates). The
    local searches from the ``num_restarts`` starting points run natively in
    parallel; ``seed`` (botorch_rs only) seeds the random lattice points
    (BoTorch draws them from torch's global generator).

    Unlike BoTorch, the pending points of the acquisition function are
    restored also with ``return_acq_values=False``.

    Returns:
        A two-element tuple containing

        - a ``q x d`` array of generated candidates.
        - an associated (joint) acquisition value.
    """
    if getattr(acq_function, "_tensor_backend", None) == "torch":
        from botorch_rs.optim._torch_acq_discrete import local_search

        return local_search(locals())
    if batch_initial_conditions is not None:
        bshape = to_flat(batch_initial_conditions, "batch_initial_conditions")[1]
        if not (len(bshape) == 3 and bshape[-2] == 1):
            raise ValueError(
                "batch_initial_conditions must have shape `n x 1 x d` if "
                f"given (received shape {tuple(bshape)})."
            )
    kind = first_kind(*discrete_choices) if discrete_choices else first_kind()
    choices = [[float(v) for v in to_flat(c, "discrete_choices")[0]] for c in discrete_choices]
    native_acqf = _as_native(acq_function, len(choices))
    hook = None
    if _gen_batch_initial_conditions_local_search is not _NATIVE_LATTICE_POINTS:
        # A replaced generator (e.g. reproducing given random draws).
        replaced = _gen_batch_initial_conditions_local_search

        def hook(data: Any, shape: list[int]) -> Flat:
            avoid = from_flat(data, tuple(shape), kind)
            X = replaced(
                discrete_choices=discrete_choices,
                raw_samples=raw_samples,
                X_avoid=avoid,
                inequality_constraints=inequality_constraints or [],
                min_points=num_restarts,
                max_tries=max_tries,
            )
            return to_flat(X, "lattice points")

    avoid = None
    if X_avoid is not None:
        avoid = to_flat(X_avoid, "X_avoid")
        if avoid[1][0] == 0:
            avoid = None
    candidates, values = _rs.optimize_acqf_discrete_local_search_py(
        native_acqf._rs,
        choices,
        int(q),
        num_restarts=int(num_restarts),
        raw_samples=int(raw_samples),
        inequality_constraints=linear_constraints_arg(
            inequality_constraints, "inequality_constraints"
        ),
        x_avoid=avoid,
        batch_initial_conditions=None
        if batch_initial_conditions is None
        else to_flat(batch_initial_conditions, "batch_initial_conditions"),
        max_tries=int(max_tries),
        unique=bool(unique),
        return_acq_values=bool(return_acq_values),
        lattice_points=hook,
        seed=None if seed is None else int(seed),
    )
    native_acqf._after_evaluation()
    X = from_pair(candidates, kind)
    if not return_acq_values:
        return X, None
    # The joint value `acq_function(candidates)` (shape `(1,)`, as BoTorch's).
    return X, from_flat(values[:1], (1,), kind)


@dataclasses.dataclass(frozen=True, kw_only=True)
class OptimizeAcqfInputs:
    """Container for inputs to ``optimize_acqf`` (BoTorch's
    ``OptimizeAcqfInputs``); see :func:`optimize_acqf` for the parameters."""

    acq_function: Any
    bounds: Any
    q: int
    num_restarts: int
    raw_samples: int | None
    options: dict[str, Any] | None
    inequality_constraints: list[tuple[Any, Any, float]] | None
    equality_constraints: list[tuple[Any, Any, float]] | None
    nonlinear_inequality_constraints: list[tuple[Any, bool]] | None
    fixed_features: Mapping[int, Any] | None
    post_processing_func: Callable[[Any], Any] | None
    batch_initial_conditions: Any
    return_best_only: bool
    gen_candidates: Any
    sequential: bool
    ic_generator: Any = None
    timeout_sec: float | None = None
    return_acq_values: bool = True
    return_full_tree: bool = False
    retry_on_optimization_warning: bool = True
    ic_gen_kwargs: dict[str, Any] = dataclasses.field(default_factory=dict)
    acq_function_sequence: list[Any] | None = None

    def __post_init__(self) -> None:
        if self.acq_function is None and self.acq_function_sequence is None:
            raise ValueError("Either `acq_function` or `acq_function_sequence` must be specified.")
        if self.acq_function_sequence is not None:
            if not self.sequential:
                raise ValueError("acq_function_sequence requires sequential optimization.")
            if len(self.acq_function_sequence) != self.q:
                raise ValueError("acq_function_sequence must have length q.")
            if self.q < 2:
                raise ValueError("acq_function_sequence requires q > 1.")
        shape = (
            tuple(self.bounds.shape)
            if is_torch_tensor(self.bounds)
            else to_flat(self.bounds, "bounds")[1]
        )
        if self.inequality_constraints is None and not (len(shape) == 2 and shape[0] == 2):
            raise ValueError(f"bounds should be a `2 x d` tensor, current shape: {list(shape)}.")
        if self.batch_initial_conditions is None and self.ic_generator is None:
            if self.nonlinear_inequality_constraints is not None:
                raise RuntimeError(
                    "`ic_generator` must be given if there are non-linear inequality constraints."
                )
            if self.raw_samples is None:
                raise ValueError(
                    "Must specify `raw_samples` when `batch_initial_conditions` is None`."
                )

    @property
    def full_tree(self) -> bool:
        from botorch_rs.acquisition.acquisition import OneShotAcquisitionFunction

        return self.return_full_tree or (
            not isinstance(self.acq_function, OneShotAcquisitionFunction)
        )

    def get_ic_generator(self) -> Callable[..., Any]:
        """Select the custom, KG, hypervolume-KG, or ordinary initializer."""
        from botorch_rs.acquisition.knowledge_gradient import qKnowledgeGradient
        from botorch_rs.acquisition.multi_objective.hypervolume_knowledge_gradient import (
            qHypervolumeKnowledgeGradient,
        )
        from botorch_rs.optim.initializers import (
            gen_batch_initial_conditions,
            gen_one_shot_hvkg_initial_conditions,
            gen_one_shot_kg_initial_conditions,
        )

        if self.ic_generator is not None:
            return self.ic_generator
        if isinstance(self.acq_function, qKnowledgeGradient):
            return gen_one_shot_kg_initial_conditions
        if isinstance(self.acq_function, qHypervolumeKnowledgeGradient):
            return gen_one_shot_hvkg_initial_conditions
        return gen_batch_initial_conditions


__all__ = [
    "OptimizeAcqfInputs",
    "optimize_acqf",
    "optimize_acqf_cyclic",
    "optimize_acqf_discrete",
    "optimize_acqf_discrete_local_search",
    "optimize_acqf_list",
    "optimize_acqf_mixed",
]

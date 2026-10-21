"""Optimization over mixed discrete and continuous domains
(``botorch.optim.optimize_mixed``).

:func:`optimize_acqf_mixed_alternating` alternates discrete local search and
continuous L-BFGS-B steps from many starting points, natively in Rust. The
building blocks (:func:`generate_starting_points`, :func:`discrete_step`,
:func:`continuous_step`, ...) are public, as in BoTorch; replacing
``botorch_rs.optim.optimize_mixed.generate_starting_points`` (e.g. to
reproduce given random starting points) makes the optimizer call the
replacement.

Random numbers (Sobol points, Boltzmann selection, spray points, random
categorical values) come from botorch_rs's generators; pass ``seed`` for
reproducibility (BoTorch uses torch's global generator).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from botorch_rs import _botorch_rs as _rs
from botorch_rs._arrays import (
    Flat,
    first_kind,
    from_flat,
    from_pair,
    is_torch_tensor,
    kind_of,
    to_flat,
)
from botorch_rs.exceptions.errors import UnsupportedError
from botorch_rs.optim.optimize import (
    OptimizeAcqfInputs,
    _as_native,
    _check_bounds,
    _emit,
    _post_processing_arg,
    optimize_acqf,
)
from botorch_rs.optim.parameter_constraints import linear_constraints_arg

# Default values (BoTorch's).
STD_CONT_PERTURBATION = 0.1
RAW_SAMPLES = 1024
NUM_RESTARTS = 20
MAX_BATCH_SIZE = 2048
MAX_ITER_ALTER = 64
MAX_ITER_DISCRETE = 4
MAX_ITER_CONT = 8
MAX_DISCRETE_VALUES = 20
MAX_ITER_INIT = 100
CONVERGENCE_TOL = 1e-8
DUPLICATE_TOL = 1e-6
STOP_AFTER_SHARE_CONVERGED = 1.0
ALTERNATING_OPTIMIZER_THRESHOLD = 10
MAX_CHOICES_ENUMERATE = 10_000
MAX_CARDINALITY_FOR_LOCAL_SEARCH = 100

SUPPORTED_OPTIONS = {
    "initialization_strategy",
    "tol",
    "maxiter_alternating",
    "maxiter_discrete",
    "maxiter_continuous",
    "maxiter_init",
    "max_discrete_values",
    "num_spray_points",
    "std_cont_perturbation",
    "batch_limit",
    "init_batch_limit",
}
SUPPORTED_INITIALIZATION = {"continuous_relaxation", "equally_spaced", "random"}

#: No continuous relaxation in the building blocks (BoTorch relaxes the
#: dimensions in `optimize_acqf_mixed_alternating` only).
_NO_RELAXATION = 2**62


def should_use_mixed_alternating_optimizer(
    discrete_dims: Mapping[int, Any] | None = None,
    cat_dims: Mapping[int, Any] | None = None,
) -> bool:
    r"""Whether to use ``optimize_acqf_mixed_alternating`` for a mixed (not
    fully discrete) search space: if the discrete and categorical dimensions
    have more than ``ALTERNATING_OPTIMIZER_THRESHOLD`` combinations, else
    ``optimize_acqf_mixed`` (BoTorch's
    ``should_use_mixed_alternating_optimizer``)."""
    if discrete_dims is None and cat_dims is None:
        return False
    n_combos = 1
    for values in (discrete_dims or {}).values():
        n_combos *= len(values)
    for values in (cat_dims or {}).values():
        n_combos *= len(values)
    return n_combos > ALTERNATING_OPTIMIZER_THRESHOLD


def _dims(dims: Mapping[int, Any] | None) -> dict[int, list[float]]:
    return {
        int(k): [float(v) for v in to_flat(vals, "values")[0]] for k, vals in (dims or {}).items()
    }


def _index_dims(indices: Any) -> dict[int, list[float]]:
    """Dimension indices (an integer array) as a map without values."""
    if isinstance(indices, Mapping):
        return {int(k): [] for k in indices}
    return {int(v): [] for v in to_flat(indices, "dims")[0]}


def _alternating_options(options: dict[str, Any] | None) -> dict[str, Any]:
    """The native options of the alternating optimizer."""
    out: dict[str, Any] = {}
    for key, value in (options or {}).items():
        if key == "initialization_strategy":
            out[key] = str(value)
        elif key in ("tol", "std_cont_perturbation"):
            out[key] = float(value)
        elif key in SUPPORTED_OPTIONS:
            out[key] = int(value)
    return out


def _problem(
    opt_inputs: OptimizeAcqfInputs,
    discrete_dims: dict[int, list[float]],
    cat_dims: dict[int, list[float]],
    relax: bool = False,
    x_baseline: Any = None,
) -> Any:
    """The native problem of ``opt_inputs`` and the dimension maps."""
    bounds_flat = _check_bounds(opt_inputs.bounds)
    d = bounds_flat[1][1]
    native = _as_native(opt_inputs.acq_function, d)
    kind = first_kind(opt_inputs.bounds)
    options = _alternating_options(opt_inputs.options)
    if not relax:
        options["max_discrete_values"] = _NO_RELAXATION
    fixed = {
        int(k) % d: float(to_flat(v, "fixed")[0][0])
        for k, v in (opt_inputs.fixed_features or {}).items()
    }
    return (
        _rs.MixedProblem(
            native._rs,
            bounds_flat,
            discrete_dims,
            cat_dims,
            int(opt_inputs.num_restarts),
            int(opt_inputs.raw_samples or 0),
            fixed_features=fixed,
            inequality_constraints=linear_constraints_arg(
                opt_inputs.inequality_constraints, "inequality_constraints"
            ),
            equality_constraints=linear_constraints_arg(
                opt_inputs.equality_constraints, "equality_constraints"
            ),
            post_processing=_post_processing_arg(opt_inputs.post_processing_func, kind),
            retry_on_optimization_warning=bool(opt_inputs.retry_on_optimization_warning),
            x_baseline=x_baseline,
            **options,
        ),
        native,
        kind,
    )


def round_discrete_dims(X: Any, discrete_dims: Mapping[int, Any]) -> Any:
    r"""Round the discrete dimensions of an array to the nearest allowed
    values (BoTorch's ``round_discrete_dims``; the first value on ties).

    Args:
        X: An array of shape ``n x d`` (or ``... x d``).
        discrete_dims: A dictionary mapping indices of discrete dimensions to
            a list of allowed values for that dimension.
    """
    if is_torch_tensor(X):
        from botorch_rs.optim._torch_acq_mixed import round_dims

        return round_dims(X, discrete_dims)
    out = _rs.round_discrete_dims_py(to_flat(X, "X"), _dims(discrete_dims))
    return from_pair(out, kind_of(X))


def get_nearest_neighbors(current_x: Any, bounds: Any, discrete_dims: Mapping[int, Any]) -> Any:
    r"""All 1-Manhattan distance neighbors of ``current_x`` (a ``d`` array)
    in the discrete dimensions (BoTorch's ``get_nearest_neighbors``), as a
    ``num_neighbors x d`` array, sorted like ``torch.unique(dim=0)`` in the
    index space of the allowed values."""
    if is_torch_tensor(current_x):
        from botorch_rs.optim._torch_acq_mixed import nearest

        return nearest(current_x, bounds, discrete_dims)
    x = [float(v) for v in to_flat(current_x, "current_x")[0]]
    rows = _rs.get_nearest_neighbors_py(x, to_flat(bounds, "bounds"), _dims(discrete_dims))
    return from_flat([v for r in rows for v in r], (len(rows), len(x)), kind_of(current_x))


def get_categorical_neighbors(
    current_x: Any,
    cat_dims: Mapping[int, Any],
    max_num_cat_values: int = MAX_DISCRETE_VALUES,
    *,
    seed: int | None = None,
) -> Any:
    r"""All 1-Hamming distance neighbors of ``current_x`` in the categorical
    dimensions (BoTorch's ``get_categorical_neighbors``); dimensions with
    more than ``max_num_cat_values`` values contribute a random sample of
    that many other values (``seed``, botorch_rs only, seeds it)."""
    if is_torch_tensor(current_x):
        from botorch_rs.optim._torch_acq_mixed import _stream, categorical

        return categorical(current_x, cat_dims, max_num_cat_values, _stream(current_x, seed))
    x = [float(v) for v in to_flat(current_x, "current_x")[0]]
    rows = _rs.get_categorical_neighbors_py(
        x, _dims(cat_dims), int(max_num_cat_values), None if seed is None else int(seed)
    )
    return from_flat([v for r in rows for v in r], (len(rows), len(x)), kind_of(current_x))


def get_spray_points(
    X_baseline: Any,
    cont_dims: Any,
    discrete_dims: Mapping[int, Any],
    cat_dims: Mapping[int, Any],
    bounds: Any,
    num_spray_points: int,
    std_cont_perturbation: float = STD_CONT_PERTURBATION,
    *,
    seed: int | None = None,
) -> Any:
    r"""Spray points around the baseline points (BoTorch's
    ``get_spray_points``): for every baseline point, ``num_spray_points``
    points with random discrete and categorical neighbors and Gaussian
    perturbations of the continuous dimensions (clamped to the bounds).

    Returns:
        A ``(n * num_spray_points) x d`` array.
    """
    if is_torch_tensor(X_baseline):
        from botorch_rs.optim._torch_acq_mixed import _stream, spray

        return spray(
            X_baseline,
            cont_dims,
            discrete_dims,
            cat_dims,
            bounds,
            num_spray_points,
            std_cont_perturbation,
            _stream(X_baseline, seed),
        )
    out = _rs.get_spray_points_py(
        to_flat(X_baseline, "X_baseline"),
        [int(v) for v in to_flat(cont_dims, "cont_dims")[0]],
        _dims(discrete_dims),
        _dims(cat_dims),
        to_flat(bounds, "bounds"),
        int(num_spray_points),
        float(std_cont_perturbation),
        None if seed is None else int(seed),
    )
    return from_pair(out, kind_of(X_baseline))


def sample_feasible_points(
    opt_inputs: OptimizeAcqfInputs,
    discrete_dims: Mapping[int, Any],
    cat_dims: Mapping[int, Any],
    num_points: int,
    *,
    seed: int | None = None,
) -> Any:
    r"""Sample feasible points from the optimization domain (BoTorch's
    ``sample_feasible_points``): Sobol points (or hit-and-run samples of the
    constraint polytope) with the discrete dimensions rounded and the fixed
    features set; infeasible points are removed (10 tries)."""
    if getattr(opt_inputs.acq_function, "_tensor_backend", None) == "torch":
        from botorch_rs.optim._torch_acq_mixed import feasible_points

        return feasible_points(opt_inputs, discrete_dims, cat_dims, num_points, seed)
    problem, _, kind = _problem(opt_inputs, _dims(discrete_dims), _dims(cat_dims))
    out = problem.sample_feasible_points(int(num_points), None if seed is None else int(seed))
    return from_pair(out, kind)


def _x_baseline(acq_function: Any) -> Flat | None:
    from botorch_rs.optim.utils.acquisition_utils import get_X_baseline

    X = get_X_baseline(acq_function)
    return None if X is None else to_flat(X, "X_baseline")


def _num_spray(options: dict[str, Any] | None, discrete_dims: dict[int, list[float]]) -> int:
    num_discrete = sum(1 for v in discrete_dims.values() if len(v) != 2)
    return int((options or {}).get("num_spray_points", 20 if num_discrete == 0 else 0))


def generate_starting_points(
    opt_inputs: OptimizeAcqfInputs,
    discrete_dims: Mapping[int, Any],
    cat_dims: Mapping[int, Any],
    cont_dims: Any,
    *,
    seed: int | None = None,
) -> tuple[Any, Any]:
    r"""Generate initial starting points for the alternating optimization
    (BoTorch's ``generate_starting_points``): candidates from the
    initialization strategy (the continuous relaxation optimized by
    ``optimize_acqf``, Sobol points, or equally spaced binary dimensions),
    spray points around the baseline points, rounded and with fixed features
    set, infeasible points removed and completed by
    ``sample_feasible_points``; ``num_restarts`` of them are selected by
    Boltzmann sampling.

    Returns:
        A ``num_restarts x d`` array of starting points and their
        ``num_restarts`` acquisition values.
    """
    if getattr(opt_inputs.acq_function, "_tensor_backend", None) == "torch":
        from botorch_rs.optim._torch_acq_mixed import starting_points

        return starting_points(opt_inputs, discrete_dims, cat_dims, cont_dims, seed)
    dd, cd = _dims(discrete_dims), _dims(cat_dims)
    xb = None
    if _num_spray(opt_inputs.options, dd) > 0:
        xb = _x_baseline(opt_inputs.acq_function)
    problem, _, kind = _problem(opt_inputs, dd, cd, x_baseline=xb)
    x, values, messages = problem.generate_starting_points(None if seed is None else int(seed))
    _emit(messages)
    return from_pair(x, kind), from_flat(values, (len(values),), kind)


def discrete_step(
    opt_inputs: OptimizeAcqfInputs,
    discrete_dims: Mapping[int, Any],
    cat_dims: Mapping[int, Any],
    current_x: Any,
    *,
    seed: int | None = None,
) -> tuple[Any, Any]:
    r"""Discrete nearest neighbour search (BoTorch's ``discrete_step``):
    every point of ``current_x`` (``b x d``) moves to its best feasible
    discrete or categorical neighbor while that improves its value, for at
    most ``maxiter_discrete`` iterations.

    Returns:
        The ``b x d`` points and their ``b`` acquisition values.
    """
    if getattr(opt_inputs.acq_function, "_tensor_backend", None) == "torch":
        from botorch_rs.optim._torch_acq_mixed import discrete_step as tensor_discrete_step

        return tensor_discrete_step(opt_inputs, discrete_dims, cat_dims, current_x, seed)
    problem, _, kind = _problem(opt_inputs, _dims(discrete_dims), _dims(cat_dims))
    x, values = problem.discrete_step(to_flat(current_x, "current_x"), seed)
    k = kind_of(current_x)
    return from_pair(x, k), from_flat(values, (len(values),), k)


def continuous_step(
    opt_inputs: OptimizeAcqfInputs,
    discrete_dims: Any,
    cat_dims: Any,
    current_x: Any,
) -> tuple[Any, Any]:
    """Continuous search using L-BFGS-B (BoTorch's ``continuous_step``): every
    point of ``current_x`` (``b x d``) is optimized over the continuous
    dimensions (``maxiter_continuous`` iterations, ``gtol = tol``; SLSQP
    with linear constraints) with the discrete and categorical dimensions
    (index arrays) fixed, then post-processed.

    Returns:
        The ``b x d`` points and their ``b`` acquisition values.
    """
    if getattr(opt_inputs.acq_function, "_tensor_backend", None) == "torch":
        from botorch_rs.optim._torch_acq_mixed import continuous_step as tensor_continuous_step

        return tensor_continuous_step(opt_inputs, discrete_dims, cat_dims, current_x)
    if opt_inputs.return_best_only:
        raise UnsupportedError("`continuous_step` does not support `return_best_only=True`.")
    problem, _, _ = _problem(opt_inputs, _index_dims(discrete_dims), _index_dims(cat_dims))
    x, values, messages = problem.continuous_step(to_flat(current_x, "current_x"))
    _emit(messages)
    k = kind_of(current_x)
    return from_pair(x, k), from_flat(values, (len(values),), k)


_NATIVE_GENERATE_STARTING_POINTS = generate_starting_points


def optimize_acqf_mixed_alternating(
    acq_function: Any,
    bounds: Any,
    discrete_dims: Mapping[int, Any] | None = None,
    cat_dims: Mapping[int, Any] | None = None,
    options: dict[str, Any] | None = None,
    q: int = 1,
    raw_samples: int = RAW_SAMPLES,
    num_restarts: int = NUM_RESTARTS,
    post_processing_func: Callable[[Any], Any] | None = None,
    sequential: bool = True,
    fixed_features: dict[int, float] | None = None,
    inequality_constraints: list[tuple[Any, Any, float]] | None = None,
    equality_constraints: list[tuple[Any, Any, float]] | None = None,
    return_acq_values: bool = True,
    retry_on_optimization_warning: bool = True,
    *,
    seed: int | None = None,
) -> tuple[Any, Any]:
    r"""Optimizes the acquisition function over mixed integer, categorical
    and continuous input spaces (BoTorch's
    ``optimize_acqf_mixed_alternating``). Multiple starting points are picked
    by evaluating a large set of initial candidates; from each, alternating
    discrete/categorical local search and continuous L-BFGS-B steps run until
    the improvement falls below ``tol`` (natively, the starting points in
    parallel).

    The discrete dimensions with more than ``options["max_discrete_values"]``
    (20) values that do not appear in constraints are optimized by continuous
    relaxation and rounded; the categorical dimensions with more than 20
    values use random subsets of their values in the discrete steps.

    Args:
        acq_function: The acquisition function.
        bounds: A ``2 x d`` array of lower and upper bounds.
        discrete_dims: A dictionary mapping indices of discrete and binary
            dimensions to a list of allowed values for that dimension.
        cat_dims: A dictionary mapping indices of categorical dimensions to a
            list of allowed values for that dimension.
        options: Options (``initialization_strategy``, ``tol``,
            ``maxiter_alternating``, ``maxiter_discrete``,
            ``maxiter_continuous``, ``maxiter_init``, ``max_discrete_values``,
            ``num_spray_points``, ``std_cont_perturbation``, ``batch_limit``,
            ``init_batch_limit``).
        q: Number of candidates (generated sequentially).
        raw_samples: Number of initial candidates used to select starting
            points from.
        num_restarts: Number of random restarts.
        post_processing_func: A function that post-processes the candidates.
        sequential: Must be True.
        fixed_features: A map ``{feature_index: value}`` of features fixed
            during the optimization.
        inequality_constraints: Intra-point linear inequality constraints.
        equality_constraints: Intra-point linear equality constraints (on
            continuous dimensions only).
        return_acq_values: Return the joint acquisition value.
        retry_on_optimization_warning: Retry the continuous relaxation of the
            initialization on optimization warnings.
        seed: Seed of the random numbers (botorch_rs only).

    Returns:
        A ``q x d`` array of candidates and their joint acquisition value
        (``None`` if ``return_acq_values=False``).
    """
    if getattr(acq_function, "_tensor_backend", None) == "torch":
        from botorch_rs.optim._torch_acq_mixed import optimize as tensor_optimize

        return tensor_optimize(locals())
    if sequential is False:
        raise NotImplementedError(
            "`optimize_acqf_mixed_alternating` only supports sequential optimization."
        )
    kind = first_kind(bounds)
    bounds_flat = _check_bounds(bounds)
    d = bounds_flat[1][1]
    dd = {k: sorted(v) for k, v in _dims(discrete_dims).items()}
    cd = {k: sorted(v) for k, v in _dims(cat_dims).items()}
    lower_b, upper_b = bounds_flat[0][:d], bounds_flat[0][d:]
    for dim, values in dd.items():
        lower_bnd, upper_bnd = float(lower_b[dim]), float(upper_b[dim])
        lower, upper = values[0], values[-1]
        if lower != lower_bnd:
            raise ValueError(
                f"Discrete dimension {dim} must start at the lower bound "
                f"{lower_bnd} but starts at {lower}."
            )
        if upper != upper_bnd:
            raise ValueError(
                f"Discrete dimension {dim} must end at the upper bound "
                f"{upper_bnd} but end at {upper}."
            )
    fixed_features = fixed_features or {}
    options = dict(options or {})
    if options.get("max_optimization_problem_aggregation_size", 1) != 1:
        raise UnsupportedError(
            "optimize_acqf_mixed_alternating does not support "
            "max_optimization_problem_aggregation_size != 1. "
            "You might leave this option empty, though."
        )
    options.setdefault("batch_limit", MAX_BATCH_SIZE)
    options.setdefault("init_batch_limit", options["batch_limit"])
    if not (keys := set(options.keys())).issubset(SUPPORTED_OPTIONS):
        unsupported_keys = keys.difference(SUPPORTED_OPTIONS)
        raise UnsupportedError(
            f"Received an unsupported option {unsupported_keys}. {SUPPORTED_OPTIONS=}."
        )
    strategy = options.get("initialization_strategy")
    if strategy is not None and strategy not in SUPPORTED_INITIALIZATION:
        raise UnsupportedError(
            f"Unsupported initialization strategy: {strategy}."
            f"Supported strategies are: {SUPPORTED_INITIALIZATION}."
        )
    for indices, _, _ in equality_constraints or []:
        idx = [int(v) for v in to_flat(indices, "indices")[0]]
        if any(i in dd or i in cd for i in idx):
            raise ValueError(
                "Equality constraints can only be used with continuous degrees of freedom."
            )
    opt_inputs = OptimizeAcqfInputs(
        acq_function=acq_function,
        bounds=bounds,
        q=q,
        num_restarts=num_restarts,
        raw_samples=raw_samples,
        options=options,
        inequality_constraints=inequality_constraints,
        equality_constraints=equality_constraints,
        nonlinear_inequality_constraints=None,
        fixed_features=fixed_features,
        post_processing_func=post_processing_func,
        batch_initial_conditions=None,
        return_best_only=False,
        gen_candidates=None,
        sequential=sequential,
        return_acq_values=True,
        retry_on_optimization_warning=retry_on_optimization_warning,
    )
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
    ff = {int(k) % d: v for k, v in fixed_features.items()}
    rem_d = {k: v for k, v in dd.items() if k not in ff}
    rem_c = {k: v for k, v in cd.items() if k not in ff}
    non_cont = [*rem_d.keys(), *rem_c.keys()]
    if not non_cont:
        # The fully continuous problem: plain (sequential) optimize_acqf.
        cont_options = {k: v for k, v in options.items()}
        return optimize_acqf(
            acq_function=acq_function,
            bounds=bounds,
            q=q,
            num_restarts=num_restarts,
            raw_samples=raw_samples,
            options=cont_options,
            inequality_constraints=inequality_constraints,
            equality_constraints=equality_constraints,
            fixed_features=fixed_features or None,
            post_processing_func=post_processing_func,
            sequential=True,
            return_acq_values=return_acq_values,
            retry_on_optimization_warning=retry_on_optimization_warning,
        )
    if len(set(non_cont)) != len(non_cont) or min(non_cont) < 0 or max(non_cont) > d - 1:
        raise ValueError(
            "`discrete_dims` and `cat_dims` must be dictionaries with unique, disjoint "
            "integers as keys between 0 and num_dims - 1."
        )
    problem, native, _ = _problem(opt_inputs, dd, cd, relax=True)
    xb = None
    if _num_spray(options, problem.discrete_dims) > 0:
        xb = _x_baseline(acq_function)
        if xb is not None:
            problem, native, _ = _problem(opt_inputs, dd, cd, relax=True, x_baseline=xb)
    hook = None
    replaced = generate_starting_points
    if replaced is not _NATIVE_GENERATE_STARTING_POINTS:
        cont_dims = from_flat(
            [float(v) for v in problem.cont_dims], (len(problem.cont_dims),), kind
        )

        def hook() -> tuple[Flat, list[float]]:
            X, values = replaced(
                opt_inputs, dict(problem.discrete_dims), dict(problem.cat_dims), cont_dims
            )
            return to_flat(X, "starting points"), [float(v) for v in to_flat(values, "values")[0]]

    candidates, value, messages = problem.optimize(
        int(q),
        None if seed is None else int(seed),
        hook,
        bool(return_acq_values),
    )
    _emit(messages)
    native._after_evaluation()
    X = from_pair(candidates, kind)
    if not return_acq_values:
        return X, None
    # The joint value `acq_function(candidates)` (shape `(1,)`, as BoTorch's).
    return X, from_flat([value], (1,), kind)


def complement_indices_like(indices: Any, d: int) -> Any:
    r"""The complement indices ``{range(d) \ indices}`` as an integer array
    like ``indices`` (BoTorch's ``complement_indices_like``)."""
    if is_torch_tensor(indices):
        import torch

        values = torch.arange(d, dtype=indices.dtype, device=indices.device)
        return values[(values[:, None] != indices).all(-1)]
    idx = [int(v) for v in to_flat(indices, "indices")[0]]
    out = complement_indices(idx, d)
    k = kind_of(indices)
    if k.lib == "torch":
        import torch

        return torch.tensor(out, dtype=indices.dtype, device=indices.device)
    if k.lib == "numpy":
        import numpy as np

        return np.asarray(out, dtype=indices.dtype)
    return out


def complement_indices(indices: list[int], d: int) -> list[int]:
    r"""The complement indices ``{range(d) \ indices}`` (BoTorch's
    ``complement_indices``)."""
    return sorted(set(range(d)).difference(indices))


__all__ = [
    "complement_indices",
    "complement_indices_like",
    "continuous_step",
    "discrete_step",
    "generate_starting_points",
    "get_categorical_neighbors",
    "get_nearest_neighbors",
    "get_spray_points",
    "optimize_acqf_mixed_alternating",
    "round_discrete_dims",
    "sample_feasible_points",
    "should_use_mixed_alternating_optimizer",
]

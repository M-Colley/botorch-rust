"""Initial conditions of the acquisition-function optimization
(``botorch.optim.initializers``)."""

from __future__ import annotations

import builtins
import math
import random
import sys
import warnings
from typing import Any

from botorch_rs import _botorch_rs as _rs
from botorch_rs import _ops
from botorch_rs._arrays import (
    Flat,
    as_float,
    first_kind,
    from_flat,
    from_pair,
    is_torch_tensor,
    kind_of,
    to_flat,
)
from botorch_rs.acquisition.acquisition import AcquisitionFunction, check_unsupported
from botorch_rs.exceptions.errors import UnsupportedError
from botorch_rs.exceptions.warnings import BadInitialCandidatesWarning, BotorchWarning
from botorch_rs.optim.parameter_constraints import linear_constraints_arg

_INT_KEYS = {"seed", "batch_limit", "init_batch_limit", "n_burnin", "n_thinning"}
_FLOAT_KEYS = {
    "eta",
    "alpha",
    "sample_around_best_sigma",
    "sample_around_best_subset_sigma",
    "sample_around_best_prob_perturb",
}
# Options of the one-shot initial conditions (ignored here).
_ONE_SHOT_KEYS = {"frac_random", "num_inner_restarts", "raw_inner_samples"}
# Options of the optimizer (ignored by the initialization).
_OPTIMIZER_KEYS = {
    "maxiter",
    "maxfun",
    "ftol",
    "gtol",
    "maxcor",
    "max_cor",
    "maxls",
    "pgtol",
    "factr",
    "disp",
    "iprint",
    "method",
    "with_grad",
}


def initializer_options(options: dict[str, Any] | None) -> dict[str, Any]:
    """The initialization entries of BoTorch's ``options`` dictionary."""
    out: dict[str, Any] = {}
    for key, value in (options or {}).items():
        if key in _INT_KEYS:
            out[key] = None if value is None else int(value)
        elif key in _FLOAT_KEYS:
            out[key] = None if value is None else float(value)
        elif key in ("nonnegative", "topn", "largest", "sorted", "sample_around_best"):
            out[key] = bool(value)
        elif key not in _OPTIMIZER_KEYS and key not in _ONE_SHOT_KEYS:
            raise UnsupportedError(f"Option {key!r} is not supported by botorch_rs.")
    if out.get("init_batch_limit") is not None:
        out["batch_limit"] = out["init_batch_limit"]
    out.pop("init_batch_limit", None)
    if not out.get("topn"):
        out.pop("largest", None)
        out.pop("sorted", None)
    return out


def _selection_options(
    acq_function: Any, opts: dict[str, Any], d: int, equality_constraints: Any
) -> dict[str, Any]:
    """The extension's ``nonnegative``, polytope-sampler, ``topn`` and
    ``sample_around_best`` arguments from the initialization options
    ``opts`` (of :func:`initializer_options`); the best points of
    ``sample_around_best`` are found here (``None`` if there are none)."""
    out: dict[str, Any] = {}
    for key in ("nonnegative", "n_burnin", "n_thinning"):
        if opts.get(key) is not None:
            out[key] = opts[key]
    if opts.get("topn"):
        out["topn"] = True
        out["largest"] = bool(opts.get("largest", getattr(acq_function, "maximize", True)))
        out["sorted"] = bool(opts.get("sorted", True))
    if opts.get("sample_around_best"):
        if equality_constraints:
            raise UnsupportedError(
                "Option 'sample_around_best' is not supported when equalityconstraints are present."
            )
        best = _best_points(acq_function, d)
        if best is not None:
            out["sample_around_best"] = best
            for key in (
                "sample_around_best_sigma",
                "sample_around_best_subset_sigma",
                "sample_around_best_prob_perturb",
            ):
                if opts.get(key) is not None:
                    out[key] = float(opts[key])
    return out


def gen_batch_initial_conditions(
    acq_function: AcquisitionFunction,
    bounds: Any,
    q: int,
    num_restarts: int,
    raw_samples: int,
    fixed_features: dict[int, float] | None = None,
    options: dict[str, Any] | None = None,
    inequality_constraints: list[tuple[Any, Any, float]] | None = None,
    equality_constraints: list[tuple[Any, Any, float]] | None = None,
    generator: Any = None,
    fixed_X_fantasies: Any = None,
) -> Any:
    r"""``num_restarts`` initial candidate sets (``num_restarts x q x d``) for
    the optimization of ``acq_function``: ``raw_samples`` scrambled Sobol
    q-batches in ``bounds`` (or, with linear constraints, q-batches from a
    hit-and-run sampler of the feasible polytope, BoTorch's
    ``sample_q_batches_from_polytope``) are evaluated and ``num_restarts`` of
    them selected by Boltzmann sampling (``initialize_q_batch``, or
    ``initialize_q_batch_nonneg`` for non-negative acquisition functions).
    If all raw samples have zero acquisition value, the selection is random
    and a ``BadInitialCandidatesWarning`` is raised.

    Supported ``options``: ``seed`` (the raw samples and the selection, which
    BoTorch draws from torch's global RNG), ``eta``, ``alpha``,
    ``nonnegative``, ``init_batch_limit`` / ``batch_limit`` and the polytope
    sampler's ``n_burnin`` (10000) and ``n_thinning`` (32).

    ``fixed_X_fantasies`` (``k x d``) are appended to every raw candidate set
    before it is evaluated (the fantasy designs of one-shot acquisition
    functions); the initial conditions are then ``(q + k) x d``.
    """
    check_unsupported(generator=generator)
    if not isinstance(acq_function, AcquisitionFunction):
        raise UnsupportedError(
            "gen_batch_initial_conditions requires a botorch_rs acquisition function."
        )
    kind = first_kind(bounds)
    bounds_flat = to_flat(bounds, "bounds")
    d = bounds_flat[1][-1] if bounds_flat[1] else 0
    if any(abs(v) == float("inf") for v in bounds_flat[0]):
        raise NotImplementedError(
            "Currently only finite values in `bounds` are supported "
            "for generating initial conditions for optimization."
        )
    fixed = None
    if fixed_features:
        fixed = {int(k) % d: as_float(v, f"fixed_features[{k}]") for k, v in fixed_features.items()}
    opts = initializer_options(options)
    selection = _selection_options(acq_function, opts, d, equality_constraints)
    for key in (
        "nonnegative",
        "n_burnin",
        "n_thinning",
        "topn",
        "largest",
        "sorted",
        "sample_around_best",
        "sample_around_best_sigma",
        "sample_around_best_subset_sigma",
        "sample_around_best_prob_perturb",
    ):
        opts.pop(key, None)
    opts.update(selection)
    ics, fallback = _rs.gen_batch_initial_conditions(
        acq_function._rs,
        bounds_flat,
        int(q),
        int(num_restarts),
        int(raw_samples),
        fixed_features=fixed,
        inequality_constraints=linear_constraints_arg(
            inequality_constraints, "inequality_constraints"
        ),
        equality_constraints=linear_constraints_arg(equality_constraints, "equality_constraints"),
        fixed_x_fantasies=None
        if fixed_X_fantasies is None
        else to_flat(fixed_X_fantasies, "fixed_X_fantasies"),
        **opts,
    )
    if fallback:
        warnings.warn(
            "Unable to find non-zero acquisition function values - initial conditions "
            "are being selected randomly.",
            BadInitialCandidatesWarning,
            stacklevel=2,
        )
    return from_pair(ics, kind)


def gen_one_shot_hvkg_initial_conditions(
    acq_function: Any,
    bounds: Any,
    q: int,
    num_restarts: int,
    raw_samples: int,
    fixed_features: dict[int, float] | None = None,
    options: dict[str, Any] | None = None,
    inequality_constraints: list[tuple[Any, Any, float]] | None = None,
    equality_constraints: list[tuple[Any, Any, float]] | None = None,
) -> Any:
    r"""Initial conditions (``num_restarts x (q + num_fantasies * num_pareto) x
    d``) for the one-shot optimization of a ``qHypervolumeKnowledgeGradient``
    (BoTorch's ``gen_one_shot_hvkg_initial_conditions``): the value function
    (the hypervolume of the posterior mean of ``num_pareto`` designs) is
    optimized from ``num_inner_restarts`` starts (``raw_inner_samples`` raw
    samples); for a fraction ``1 - frac_random`` of the restarts, the designs
    of every fantasy are drawn from the optimized sets by Boltzmann sampling
    (temperature ``eta``, default 2) and the ``q`` candidates are initialized
    by ``gen_batch_initial_conditions`` with these designs fixed; the other
    restarts come from ``gen_batch_initial_conditions`` entirely.

    ``options`` holds the initialization options of
    ``gen_batch_initial_conditions`` (``seed``, ``eta``, ``alpha``, batch
    limits, polytope sampler), ``maxiter`` of the inner optimization, and
    ``frac_random`` (0.1), ``num_inner_restarts`` (20) and
    ``raw_inner_samples`` (1024). The random choices use botorch_rs's
    generator, seeded by ``seed`` (BoTorch uses the global torch generator).
    """
    from botorch_rs.acquisition.multi_objective.hypervolume_knowledge_gradient import (
        qHypervolumeKnowledgeGradient,
    )

    if not isinstance(acq_function, qHypervolumeKnowledgeGradient):
        raise UnsupportedError(
            "gen_one_shot_hvkg_initial_conditions requires a botorch_rs "
            "qHypervolumeKnowledgeGradient."
        )
    opts = dict(options or {})
    frac_random = float(opts.pop("frac_random", 0.1))
    if not 0 < frac_random < 1:
        raise ValueError(f"frac_random must take on values in (0,1). Value: {frac_random}")
    num_inner = opts.pop("num_inner_restarts", None)
    raw_inner = opts.pop("raw_inner_samples", None)
    maxiter = opts.pop("maxiter", None)
    init = initializer_options(opts)
    eta = init.get("eta")
    kind = first_kind(bounds)
    bounds_flat = to_flat(bounds, "bounds")
    d = bounds_flat[1][-1] if bounds_flat[1] else 0
    fixed = None
    if fixed_features:
        fixed = {int(k) % d: as_float(v, f"fixed_features[{k}]") for k, v in fixed_features.items()}
    ics, messages = acq_function._state.gen_one_shot_initial_conditions(
        bounds_flat,
        int(q),
        int(num_restarts),
        int(raw_samples),
        fixed_features=fixed,
        inequality_constraints=linear_constraints_arg(
            inequality_constraints, "inequality_constraints"
        ),
        equality_constraints=linear_constraints_arg(equality_constraints, "equality_constraints"),
        seed=init.get("seed"),
        eta=eta,
        alpha=init.get("alpha"),
        batch_limit=init.get("batch_limit"),
        maxiter=None if maxiter is None else int(maxiter),
        frac_random=frac_random,
        num_inner_restarts=None if num_inner is None else int(num_inner),
        raw_inner_samples=None if raw_inner is None else int(raw_inner),
        one_shot_eta=eta,
        **_selection_options(acq_function, init, d, equality_constraints),
    )
    for message in messages:
        warnings.warn(message, BadInitialCandidatesWarning, stacklevel=2)
    return from_pair(ics, kind)


def gen_one_shot_kg_initial_conditions(
    acq_function: Any,
    bounds: Any,
    q: int,
    num_restarts: int,
    raw_samples: int,
    fixed_features: dict[int, float] | None = None,
    options: dict[str, Any] | None = None,
    inequality_constraints: list[tuple[Any, Any, float]] | None = None,
    equality_constraints: list[tuple[Any, Any, float]] | None = None,
) -> Any:
    r"""Generate a batch of smart initializations for qKnowledgeGradient
    (BoTorch's ``gen_one_shot_kg_initial_conditions``).

    The ``num_restarts x (q + num_fantasies) x d`` initial conditions come
    from ``gen_batch_initial_conditions`` of the knowledge gradient; the last
    ``int((1 - frac_random) * num_fantasies)`` fantasy points of every restart
    are replaced by maximizers of the value function of the current model
    (``optimize_acqf`` with ``num_inner_restarts`` (20) restarts from
    ``raw_inner_samples`` (1024) raw samples), drawn by Boltzmann sampling of
    their values (temperature ``eta``, default 2).

    ``options`` holds the initialization options of
    ``gen_batch_initial_conditions`` (``seed``, ``eta``, ``alpha``, batch
    limits, polytope sampler) and ``frac_random`` (0.1),
    ``num_inner_restarts`` and ``raw_inner_samples``. The random choices use
    botorch_rs's generator, seeded by ``seed`` (BoTorch uses the global torch
    generator).
    """
    from botorch_rs.acquisition.knowledge_gradient import qKnowledgeGradient

    if not isinstance(acq_function, qKnowledgeGradient):
        raise UnsupportedError(
            "gen_one_shot_kg_initial_conditions requires a botorch_rs qKnowledgeGradient."
        )
    opts = dict(options or {})
    frac_random = float(opts.pop("frac_random", 0.1))
    if not 0 < frac_random < 1:
        raise ValueError(f"frac_random must take on values in (0,1). Value: {frac_random}")
    for constraints, kind_ in (
        (inequality_constraints, "inequality"),
        (equality_constraints, "equality"),
    ):
        for indices, _, _ in constraints or []:
            if len(getattr(indices, "shape", ())) > 1:
                raise NotImplementedError(
                    "Indices must be one-dimensional "
                    "in gen_one_shot_kg_initial_conditions. "
                    f"Received indices {indices} "
                    f"for {kind_} constrained."
                )
    num_inner = opts.pop("num_inner_restarts", None)
    raw_inner = opts.pop("raw_inner_samples", None)
    init = initializer_options(opts)
    eta = init.get("eta")
    kind = first_kind(bounds)
    bounds_flat = to_flat(bounds, "bounds")
    d = bounds_flat[1][-1] if bounds_flat[1] else 0
    fixed = None
    if fixed_features:
        fixed = {int(k) % d: as_float(v, f"fixed_features[{k}]") for k, v in fixed_features.items()}
    ics, messages = acq_function._state.gen_one_shot_initial_conditions(
        bounds_flat,
        int(q),
        int(num_restarts),
        int(raw_samples),
        fixed_features=fixed,
        inequality_constraints=linear_constraints_arg(
            inequality_constraints, "inequality_constraints"
        ),
        equality_constraints=linear_constraints_arg(equality_constraints, "equality_constraints"),
        seed=init.get("seed"),
        eta=eta,
        alpha=init.get("alpha"),
        batch_limit=init.get("batch_limit"),
        frac_random=frac_random,
        num_inner_restarts=None if num_inner is None else int(num_inner),
        raw_inner_samples=None if raw_inner is None else int(raw_inner),
        one_shot_eta=eta,
        **_selection_options(acq_function, init, d, equality_constraints),
    )
    for message in messages:
        warnings.warn(message, BadInitialCandidatesWarning, stacklevel=2)
    return from_pair(ics, kind)


class _Draws:
    """Random draws with the random number generator of the caller's array
    library: torch's (so that ``torch.manual_seed`` reproduces BoTorch's
    choices exactly), else Python's ``random``."""

    def __init__(self, kind: Any) -> None:
        self.torch = kind.lib == "torch"

    def randperm(self, n: int) -> list[int]:
        if self.torch:
            import torch

            return [int(i) for i in torch.randperm(n).tolist()]
        perm = list(range(n))
        random.shuffle(perm)
        return perm

    def multinomial(self, weights: list[float], n: int) -> list[int]:
        """``n`` indices drawn without replacement with probabilities
        proportional to ``weights`` (``torch.multinomial``)."""
        if self.torch:
            import torch

            w = torch.tensor(weights, dtype=torch.float64)
            return [int(i) for i in torch.multinomial(w, n, replacement=False).tolist()]
        wl = list(weights)
        out: list[int] = []
        for _ in range(n):
            total = sum(wl)
            u = random.random() * total
            pick = None
            for i, wi in enumerate(wl):
                if wi > 0:
                    pick = i
                    if u < wi:
                        break
                    u -= wi
            if pick is None:
                pick = random.choice([i for i in range(len(wl)) if i not in out])
            out.append(pick)
            wl[pick] = 0.0
        return out


def _select(X: Any, acq_vals: Any, idcs: list[int]) -> tuple[Any, Any]:
    """``X[idcs]`` and ``acq_vals[idcs]`` (arrays of their own types)."""
    xdata, xshape = to_flat(X, "X")
    vdata, vshape = to_flat(acq_vals, "acq_vals")
    per = math.prod(xshape[1:])
    xs: list[float] = []
    for i in idcs:
        xs.extend(xdata[i * per : (i + 1) * per])
    vs = [vdata[i] for i in idcs]
    return (
        X[idcs] if is_torch_tensor(X) else from_flat(xs, (len(idcs), *xshape[1:]), kind_of(X)),
        acq_vals[idcs]
        if is_torch_tensor(acq_vals)
        else from_flat(vs, (len(idcs), *vshape[1:]), kind_of(acq_vals)),
    )


def _check_batch(X: Any, acq_vals: Any) -> tuple[int, list[float]]:
    xshape = to_flat(X, "X")[1]
    vdata, vshape = to_flat(acq_vals, "acq_vals")
    if len(xshape) > 3 or len(vshape) != 1:
        raise UnsupportedError(
            "botorch_rs supports `b x q x d` samples with `b` values (no batch shape)."
        )
    return xshape[0], [float(v) for v in vdata]


def _argmax(values: list[float]) -> int:
    """``torch.max(dim=0).indices`` (NaN is the largest value)."""
    for i, v in enumerate(values):
        if v != v:
            return i
    return max(range(len(values)), key=lambda i: (values[i], -i))


def _std(values: list[float]) -> float:
    n = len(values)
    if n < 2:
        return float("nan")
    mean = sum(values) / n
    return math.sqrt(sum((v - mean) ** 2 for v in values) / (n - 1))


def _descending(values: list[float]) -> list[int]:
    """Indices by descending value (NaN first, ties in index order)."""
    return builtins.sorted(
        range(len(values)), key=lambda i: (0 if values[i] != values[i] else 1, -values[i], i)
    )


def initialize_q_batch(X: Any, acq_vals: Any, n: int, eta: float = 1.0) -> tuple[Any, Any]:
    r"""Heuristic for selecting initial conditions for candidate generation
    (BoTorch's ``initialize_q_batch``).

    This heuristic selects points from ``X`` (without replacement) with
    probability proportional to ``exp(eta * Z)``, where ``Z = (acq_vals -
    mean(acq_vals)) / std(acq_vals)`` and ``eta`` is a temperature parameter
    (halved while a weight overflows); the best point is always included.
    The draws use the random number generator of the input arrays' library
    (torch's for tensors, so ``torch.manual_seed`` gives BoTorch's choices).

    Args:
        X: A ``b x q x d`` array of ``b`` samples of ``q``-batches.
        acq_vals: A ``b`` array of the corresponding acquisition values.
        n: The number of initial conditions to be generated.
        eta: Temperature parameter for weighting samples.

    Returns:
        The ``n x q x d`` initial conditions and their ``n`` values.
    """
    if len(_ops.shape_of(X)) > 3 or len(_ops.shape_of(acq_vals)) > 1:
        return _initialize_q_batch_batched(X, acq_vals, n=n, eta=eta)
    n_samples, values = _check_batch(X, acq_vals)
    if n > n_samples:
        raise RuntimeError(
            f"n ({n}) cannot be larger than the number of provided samples ({n_samples})"
        )
    if n == n_samples:
        return X, acq_vals
    draws = _Draws(kind_of(X))
    ystd = _std(values)
    if ystd == 0 or not math.isfinite(ystd):
        warnings.warn(
            "All acquisition values for raw samples points are the same or not finite for at "
            "least one batch. Choosing initial conditions at random.",
            BadInitialCandidatesWarning,
            stacklevel=2,
        )
        return _select(X, acq_vals, draws.randperm(n_samples)[:n])
    max_idx = _argmax(values)
    weights, _ = _rs.boltzmann_weights_py(values, float(eta))
    idcs = draws.multinomial(list(weights), n)
    if max_idx not in idcs:
        idcs[-1] = max_idx
    return _select(X, acq_vals, idcs)


def _initialize_q_batch_batched(X: Any, acq_vals: Any, n: int, eta: float) -> tuple[Any, Any]:
    from botorch_rs.utils.sampling import boltzmann_sample

    data, shape = to_flat(X, "X")
    values, vshape = to_flat(acq_vals, "acq_vals")
    count, batch, q, d = shape[0], tuple(shape[1:-2]), shape[-2], shape[-1]
    nb = math.prod(batch)
    if tuple(vshape) != (count, *batch):
        raise ValueError("Acquisition values must have shape samples x batch_shape.")
    if n > count:
        raise RuntimeError(
            f"n ({n}) cannot be larger than the number of provided samples ({count})"
        )
    if n == count:
        return X, acq_vals
    columns = [[float(values[sample * nb + b]) for sample in range(count)] for b in range(nb)]
    stds = []
    for column in columns:
        try:
            stds.append(_std(column))
        except OverflowError:
            stds.append(math.inf)
    if any(std == 0 or not math.isfinite(std) for std in stds):
        warnings.warn(
            "All acquisition values for raw samples points are the same or not finite for at least one batch. Choosing initial conditions at random.",
            BadInitialCandidatesWarning,
            stacklevel=3,
        )
        selected = [_Draws(kind_of(X)).randperm(count)[:n]] * nb
    else:
        value_array = from_flat(
            [value for column in columns for value in column], (*batch, count), kind_of(acq_vals)
        )
        if is_torch_tensor(acq_vals):
            # torch.multinomial's exponential draws follow the dense strides
            # of its weights. Preserve the layout of BoTorch's permuted
            # scores, otherwise the same seed assigns draws to other rows.
            template = acq_vals.permute(*range(1, len(batch) + 1), 0)
            value_array = sys.modules["torch"].empty_like(template).copy_(value_array)
        index_values, _ = to_flat(boltzmann_sample(value_array, n, eta))
        selected = [[int(index) for index in index_values[b * n : (b + 1) * n]] for b in range(nb)]
        for b, column in enumerate(columns):
            maximum = _argmax(column)
            if maximum not in selected[b]:
                selected[b][-1] = maximum
    points: list[float] = []
    selected_values: list[float] = []
    for sample in range(n):
        for b in range(nb):
            index = selected[b][sample]
            start = (index * nb + b) * q * d
            points.extend(data[start : start + q * d])
            selected_values.append(values[index * nb + b])
    selected_points = from_flat(points, (n, *batch, q, d), kind_of(X))
    selected_acq_vals = from_flat(selected_values, (n, *batch), kind_of(acq_vals))
    selected_indices = [selected[b][sample] for sample in range(n) for b in range(nb)]
    if is_torch_tensor(X):
        indices = _ops.int_array(selected_indices, (n, *batch), kind_of(X))
        selected_points = X.gather(0, indices.reshape(n, *batch, 1, 1).expand(n, *shape[1:]))
    if is_torch_tensor(acq_vals):
        indices = _ops.int_array(selected_indices, (n, *batch), kind_of(acq_vals))
        selected_acq_vals = acq_vals.gather(0, indices)
    return selected_points, selected_acq_vals


def gen_value_function_initial_conditions(
    acq_function: Any,
    bounds: Any,
    num_restarts: int,
    raw_samples: int,
    current_model: Any,
    fixed_features: dict[int, float] | None = None,
    options: dict[str, Any] | None = None,
) -> Any:
    """Initialize a KG value function from current-model optima and Sobol draws.

    The inner acquisition optimization and Boltzmann weights run in Rust.
    Fantasy/model batch dimensions are retained. Torch uses its Sobol and
    multinomial generators; other array types use the port's native Sobol
    stream and their usual sampling generators.
    """
    from botorch_rs.acquisition.knowledge_gradient import _get_value_function
    from botorch_rs.optim.optimize import optimize_acqf
    from botorch_rs.optim.utils.acquisition_utils import fix_features
    from botorch_rs.utils.sampling import boltzmann_sample, draw_sobol_samples

    opts = dict(options or {})
    if raw_samples <= 0 or num_restarts <= 0:
        raise ValueError("raw_samples and num_restarts must be positive.")
    if num_restarts > raw_samples:
        raise RuntimeError(
            f"n ({num_restarts}) cannot be larger than the number of provided samples ({raw_samples})"
        )
    frac_random = float(opts.pop("frac_random", 0.6))
    if not 0 < frac_random < 1:
        raise ValueError(f"frac_random must take on values in (0,1). Value: {frac_random}")
    inner_restarts = opts.pop("num_inner_restarts", 20)
    inner_samples = opts.pop("raw_inner_samples", 1024)
    eta = float(opts.pop("eta", 2.0))
    if not math.isfinite(eta):
        raise ValueError("eta must be finite.")
    value_function = _get_value_function(
        model=current_model,
        objective=getattr(acq_function, "objective", None),
        posterior_transform=acq_function.posterior_transform,
        sampler=getattr(acq_function, "sampler", None),
        project=getattr(acq_function, "project", None),
    )
    candidates, values = optimize_acqf(
        acq_function=value_function,
        bounds=bounds,
        q=1,
        num_restarts=inner_restarts,
        raw_samples=inner_samples,
        fixed_features=fixed_features,
        return_best_only=False,
        options=opts,
    )
    candidate_values, value_shape = to_flat(values)
    candidate_data, candidate_shape = to_flat(candidates)
    if len(value_shape) != 1 or len(candidate_shape) != 3 or candidate_shape[1] != 1:
        raise UnsupportedError(
            "Current-model optimization must return scalar values of single-point candidates."
        )
    batch = tuple(acq_function.model.batch_shape)
    nb, d = math.prod(batch), candidate_shape[-1]
    kind = kind_of(candidates)
    n_value = int((1.0 - frac_random) * raw_samples)
    n_random = raw_samples - n_value
    resampled = None
    if n_value:
        indices, _ = to_flat(
            boltzmann_sample(
                from_flat(
                    list(candidate_values) * nb, (*batch, len(candidate_values)), kind_of(values)
                ),
                n_value,
                eta,
                replacement=True,
            )
        )
        samples: list[float] = []
        for sample in range(n_value):
            for b in range(nb):
                index = int(indices[b * n_value + sample])
                samples.extend(candidate_data[index * d : (index + 1) * d])
        resampled = from_flat(samples, (n_value, *batch, 1, d), kind)
    randomized = None
    if n_random:
        if is_torch_tensor(bounds):
            torch = sys.modules["torch"]
            engine = torch.quasirandom.SobolEngine(d, scramble=True, seed=opts.get("seed"))
            uniform = engine.draw(n_random * nb, dtype=bounds.dtype).to(device=bounds.device)
            uniform = uniform.reshape(*batch, n_random, 1, d).permute(
                len(batch), *range(len(batch)), len(batch) + 1, len(batch) + 2
            )
            randomized = bounds[0] + (bounds[1] - bounds[0]) * uniform
        else:
            randomized = draw_sobol_samples(
                bounds, n=n_random, q=1, batch_shape=batch, seed=opts.get("seed")
            )
    raw = (
        randomized
        if resampled is None
        else resampled
        if randomized is None
        else _ops.concat([resampled, randomized], dim=0)
    )
    raw = fix_features(raw, fixed_features, replace_current_value=True)
    if is_torch_tensor(raw):
        with sys.modules["torch"].no_grad():
            acquisition_values = acq_function(raw)
    else:
        acquisition_values = acq_function(raw)
    return initialize_q_batch(raw, acquisition_values, n=num_restarts, eta=eta)[0]


def initialize_q_batch_nonneg(
    X: Any, acq_vals: Any, n: int, eta: float = 1.0, alpha: float = 1e-4
) -> tuple[Any, Any]:
    r"""Heuristic for selecting initial conditions for non-negative
    acquisition functions (BoTorch's ``initialize_q_batch_nonneg``).

    All samples for which ``acq_vals < alpha * max(acq_vals)`` are ignored
    (``alpha`` is divided by 10 until at least ``n`` remain) and the rest
    are drawn with weights ``exp(eta * (acq_vals / max - 1))``; if fewer
    than ``n`` values are positive, all positive samples are taken and the
    rest are filled randomly.

    Returns:
        The ``n x q x d`` initial conditions and their ``n`` values.
    """
    n_samples, values = _check_batch(X, acq_vals)
    if n > n_samples:
        raise RuntimeError("n cannot be larger than the number of provided samples")
    if n == n_samples:
        return X, acq_vals
    draws = _Draws(kind_of(X))
    max_idx = _argmax(values)
    max_val = values[max_idx]
    if not max_val > 0:
        warnings.warn(
            "All acquisition values for raw sampled points are nonpositive, so initial "
            "conditions are being selected randomly.",
            BadInitialCandidatesWarning,
            stacklevel=2,
        )
        return _select(X, acq_vals, draws.randperm(n_samples)[:n])
    pos = [v > 0 for v in values]
    num_pos = sum(pos)
    if num_pos < n:
        remaining = [i for i in range(n_samples) if not pos[i]]
        perm = draws.randperm(len(remaining))
        for j in perm[: n - num_pos]:
            pos[remaining[j]] = True
        return _select(X, acq_vals, [i for i in range(n_samples) if pos[i]])
    alpha_pos = [i for i in range(n_samples) if values[i] >= alpha * max_val]
    while len(alpha_pos) < n:
        alpha = 0.1 * alpha
        alpha_pos = [i for i in range(n_samples) if values[i] >= alpha * max_val]
    weights = [math.exp(eta * (values[i] / max_val - 1)) for i in alpha_pos]
    idcs = [alpha_pos[j] for j in draws.multinomial(weights, n)]
    if max_idx not in idcs:
        idcs[-1] = max_idx
    return _select(X, acq_vals, idcs)


def initialize_q_batch_topn(
    X: Any, acq_vals: Any, n: int, largest: bool = True, sorted: bool = True
) -> tuple[Any, Any]:
    r"""Take the top ``n`` initial conditions for candidate generation
    (BoTorch's ``initialize_q_batch_topn``).

    Returns:
        The ``n x q x d`` initial conditions and their ``n`` values.
    """
    n_samples, values = _check_batch(X, acq_vals)
    if n > n_samples:
        raise RuntimeError(
            f"n ({n}) cannot be larger than the number of provided samples ({n_samples})"
        )
    if n == n_samples:
        return X, acq_vals
    draws = _Draws(kind_of(X))
    if _std(values) == 0:
        warnings.warn(
            "All acquisition values for raw samples points are the same for at least one "
            "batch. Choosing initial conditions at random.",
            BadInitialCandidatesWarning,
            stacklevel=2,
        )
        return _select(X, acq_vals, draws.randperm(n_samples)[:n])
    if is_torch_tensor(acq_vals):
        top = acq_vals.topk(n, largest=largest, sorted=sorted).indices
        idcs = [int(i) for i in top.tolist()]
    else:
        idcs = _descending(values if largest else [-v for v in values])[:n]
        if not sorted:
            idcs = builtins.sorted(idcs)
    return _select(X, acq_vals, idcs)


def is_nonnegative(acq_function: Any) -> bool:
    r"""Whether an acquisition function is known to be non-negative
    (BoTorch's ``is_nonnegative``): expected and noisy expected improvement,
    probability of improvement and the expected hypervolume improvements
    (not their log versions)."""
    rs = getattr(acq_function, "_rs", None)
    return bool(getattr(rs, "is_nonnegative", False))


def _objective_values(acq_function: Any, mean_rows: list[list[float]]) -> list[Any]:
    """BoTorch's ``f_pred = acq_function.objective(mean)`` (rows ``n x m``):
    a list of values (one objective) or of rows (several objectives)."""
    from botorch_rs.acquisition.objective import IdentityMCObjective

    objective = getattr(acq_function, "objective", None)
    if objective is None:
        return list(mean_rows)
    if isinstance(objective, IdentityMCObjective):
        if mean_rows and len(mean_rows[0]) == 1:
            return [r[0] for r in mean_rows]
        return list(mean_rows)
    from botorch_rs._torch_bridge import require_torch

    require_torch()
    import torch

    out = objective(torch.tensor(mean_rows, dtype=torch.float64))
    data, shape = to_flat(out, "objective")
    if len(shape) == 1:
        return [float(v) for v in data]
    m = shape[-1]
    return [[float(v) for v in data[i * m : (i + 1) * m]] for i in range(len(data) // m)]


def _best_points(acq_function: Any, d: int, best_pct: float = 5.0) -> Flat | None:
    """The best baseline points of BoTorch's ``sample_points_around_best``
    (``best_pct`` percent of the baseline points by their predicted
    objective, or the Pareto set), or ``None``."""
    from botorch_rs.optim.utils.acquisition_utils import get_X_baseline

    X = get_X_baseline(acq_function)
    if X is None:
        return None
    model = getattr(acq_function, "model", None)
    if model is None or not hasattr(model, "posterior"):
        warnings.warn("Failed to sample around previous best points.", BotorchWarning, stacklevel=3)
        return None
    xdata, xshape = to_flat(X, "X_baseline")
    n = xshape[0]
    mean, mshape = to_flat(model.posterior(X).mean, "mean")
    m = mshape[-1]
    reps = len(mean) // (n * m)
    rows = [
        [sum(mean[(b * n + i) * m + j] for b in range(reps)) / reps for j in range(m)]
        for i in range(n)
    ]
    f_pred = _objective_values(acq_function, rows)
    if getattr(acq_function, "maximize", True) is False:
        f_pred = [[-v for v in r] if isinstance(r, list) else -r for r in f_pred]
    constraints = getattr(acq_function, "constraints", None)
    if constraints is not None and not isinstance(constraints, dict):
        import torch

        mean_t = torch.tensor(rows, dtype=torch.float64)
        viol = [0.0] * n
        for c in constraints:
            cv = to_flat(c(mean_t), "constraint")[0]
            for i in range(n):
                viol[i] -= max(float(cv[i]), 0.0)
        feas = [v == 0 for v in viol]
        if any(feas):
            f_pred = [
                fv if feas[i] else ([-math.inf] * len(fv) if isinstance(fv, list) else -math.inf)
                for i, fv in enumerate(f_pred)
            ]
        else:
            f_pred = list(viol)
    if f_pred and isinstance(f_pred[0], list) and len(f_pred[0]) > 1:
        from botorch_rs.utils.multi_objective.pareto import is_non_dominated

        mask = to_flat(is_non_dominated(f_pred), "pareto")[0]
        best = [i for i in range(n) if mask[i]]
    else:
        values = [r[0] if isinstance(r, list) else r for r in f_pred]
        n_best = max(1, round(n * best_pct / 100))
        best = _descending(values)[:n_best]
    data: list[float] = []
    for i in best:
        data.extend(xdata[i * d : (i + 1) * d])
    return data, (len(best), d)


def sample_points_around_best(
    acq_function: Any,
    n_discrete_points: int,
    sigma: float,
    bounds: Any,
    best_pct: float = 5.0,
    subset_sigma: float = 1e-1,
    prob_perturb: float | None = None,
    *,
    seed: int | None = None,
) -> Any:
    r"""Find the best points and sample nearby points (BoTorch's
    ``sample_points_around_best``).

    The best points are ``best_pct`` percent of the baseline points (the
    acquisition function's ``X_baseline`` or its model's training inputs)
    by the objective of the posterior mean, or the Pareto set for several
    objectives. They are perturbed with truncated normal noise
    (``sigma``, relative to the bounds); for ``d >= 20`` or with
    ``prob_perturb``, half of the points perturb only a subset of the
    dimensions (with ``subset_sigma``, which BoTorch ignores in favor of
    ``sigma``). ``seed`` (botorch_rs only) seeds the draws.

    Returns:
        An ``n_discrete_points x d`` array, or ``None`` if no baseline points
        are found.
    """
    kind = first_kind(bounds)
    bounds_flat = to_flat(bounds, "bounds")
    d = bounds_flat[1][-1]
    best = _best_points(acq_function, d, best_pct)
    if best is None:
        return None
    out = _rs.sample_points_around_py(
        best,
        int(n_discrete_points),
        float(sigma),
        bounds_flat,
        float(subset_sigma),
        None if prob_perturb is None else float(prob_perturb),
        None if seed is None else int(seed),
    )
    return from_pair(out, kind)


def sample_q_batches_from_polytope(
    n: int,
    q: int,
    bounds: Any,
    n_burnin: int,
    n_thinning: int,
    seed: int | None = None,
    inequality_constraints: list[tuple[Any, Any, float]] | None = None,
    equality_constraints: list[tuple[Any, Any, float]] | None = None,
) -> Any:
    r"""``n`` q-batches (``n x q x d``) from a hit-and-run sampler of the
    polytope defined by ``bounds`` and the linear constraints; with
    inter-point constraints the ``q x d`` space is sampled jointly (thinning
    ``n_thinning * q``), as in BoTorch. The chain differs from BoTorch's
    (torch RNG) for the same seed; the distribution is the same."""
    out = _rs.sample_q_batches_from_polytope(
        int(n),
        int(q),
        to_flat(bounds, "bounds"),
        int(n_burnin),
        int(n_thinning),
        None if seed is None else int(seed),
        linear_constraints_arg(inequality_constraints, "inequality_constraints"),
        linear_constraints_arg(equality_constraints, "equality_constraints"),
    )
    return from_pair(out, first_kind(bounds))


def transform_constraints(
    constraints: list[tuple[Any, Any, float]] | None, q: int, d: int
) -> list[tuple[Any, Any, float]] | None:
    r"""Intra- and inter-point constraints on a ``q x d`` q-batch as
    constraints on the flattened ``q * d`` vector (BoTorch's
    ``transform_constraints``): an intra-point constraint becomes ``q``
    constraints, one per point; the ``(point, feature)`` index pairs of an
    inter-point constraint become ``point * d + feature``."""
    if constraints is None:
        return None
    out: list[tuple[Any, Any, float]] = []
    for constraint in constraints:
        if len(to_flat(constraint[0], "indices")[1]) == 1:
            out.extend(transform_intra_point_constraint(constraint, d=d, q=q))
        else:
            out.append(transform_inter_point_constraint(constraint, d=d))
    return out


def transform_intra_point_constraint(
    constraint: tuple[Any, Any, float], d: int, q: int
) -> list[tuple[Any, Any, float]]:
    """Repeat a pointwise constraint for each point of a flattened q-batch."""
    indices, coefficients, rhs = constraint
    values, shape = to_flat(indices, "indices")
    if len(shape) != 1 or any(int(value) != value for value in values):
        raise ValueError("Intra-point constraint indices must be a one-dimensional integer array.")
    if any(not 0 <= value < d for value in values):
        raise ValueError(f"Constraint indices cannot exceed the problem dimension {d=}.")
    return [
        (
            _int_array([point * d + int(index) for index in values], kind_of(indices)),
            coefficients,
            rhs,
        )
        for point in range(q)
    ]


def transform_inter_point_constraint(
    constraint: tuple[Any, Any, float], d: int
) -> tuple[Any, Any, float]:
    """Map (point, feature) pairs to indices of the flattened q-batch."""
    indices, coefficients, rhs = constraint
    values, shape = to_flat(indices, "indices")
    if len(shape) != 2 or shape[1] != 2 or any(int(value) != value for value in values):
        raise ValueError("Inter-point constraint indices must be an integer array of shape (n, 2).")
    if any(value < 0 for value in values) or any(value >= d for value in values[1::2]):
        raise ValueError(f"Constraint indices cannot exceed the problem dimension {d=}.")
    transformed = [int(values[i]) * d + int(values[i + 1]) for i in range(0, len(values), 2)]
    return _int_array(transformed, kind_of(indices)), coefficients, rhs


def _int_array(values: list[int], kind: Any) -> Any:
    """An integer index array of the kind of the input indices."""
    if kind.lib == "torch":
        import torch

        return torch.tensor(values, dtype=torch.long, device=kind.device)
    if kind.lib == "numpy":
        import numpy as np

        return np.asarray(values, dtype=np.int64)
    return list(values)


__all__ = [
    "gen_batch_initial_conditions",
    "gen_one_shot_hvkg_initial_conditions",
    "gen_one_shot_kg_initial_conditions",
    "gen_value_function_initial_conditions",
    "initialize_q_batch",
    "initialize_q_batch_nonneg",
    "initialize_q_batch_topn",
    "is_nonnegative",
    "sample_points_around_best",
    "sample_q_batches_from_polytope",
    "transform_constraints",
    "transform_inter_point_constraint",
    "transform_intra_point_constraint",
]

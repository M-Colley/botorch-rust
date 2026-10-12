r"""Multi-objective optimization of multi-output acquisition functions with
NSGA-II (``botorch.utils.multi_objective.optimize``).

BoTorch runs pymoo's ``NSGA2``; botorch_rs implements NSGA-II (Deb, Pratap,
Agarwal and Meyarivan, *A fast and elitist multiobjective genetic algorithm:
NSGA-II*, IEEE TEC 6(2), 2002) in Rust with pymoo's defaults as BoTorch
configures them (pymoo 0.6.2: random sampling, binary tournament selection
by dominance and crowding distance, simulated binary crossover with
``eta = 15``, polynomial mutation with ``eta = 20``, elimination of duplicate
offspring, rank-and-crowding survival, and pymoo's default termination
unless ``max_gen`` is given), so pymoo is not needed. Random streams differ
from pymoo's: results agree with BoTorch + pymoo statistically (e.g. in the
hypervolume of the returned front), not bitwise.

Without an ``objective`` and outcome ``constraints``, a botorch_rs
``MultiOutputPosteriorMean`` or ``MultiOutputAcquisitionFunctionWrapper`` is
optimized entirely in Rust (with the GIL released); otherwise the population
is evaluated through Python (torch), as in BoTorch.
"""

from __future__ import annotations

import random
import warnings
from collections.abc import Callable
from typing import Any

from botorch_rs import _botorch_rs as _rs
from botorch_rs._arrays import (
    ArrayKind,
    first_kind,
    from_flat,
    from_pair,
    is_torch_tensor,
    kind_of,
    to_flat,
)
from botorch_rs.exceptions.errors import UnsupportedError
from botorch_rs.exceptions.warnings import BotorchWarning
from botorch_rs.utils.multi_objective.hypervolume import get_hypervolume_maximizing_subset
from botorch_rs.utils.multi_objective.pareto import is_non_dominated


def _floats(x: Any, name: str) -> list[float]:
    return [float(v) for v in to_flat(x, name)[0]]


def _ints(x: Any, name: str) -> list[int]:
    return [int(v) for v in to_flat(x, name)[0]]


def _evaluate_with(
    acq_function: Any,
    objective: Any,
    constraints: list[Callable[[Any], Any]] | None,
    kind: ArrayKind,
) -> Callable[[Any], Any]:
    """The population evaluation through Python: BoTorch's
    ``BotorchPymooProblem._evaluate`` (objectives and outcome constraints of
    the acquisition values of every point)."""

    def evaluate(pair: Any) -> tuple[Any, Any]:
        data, (n, d) = pair
        X = from_flat(data, (n, 1, d), kind)
        if is_torch_tensor(X):
            import torch

            with torch.no_grad():
                y = acq_function(X)
                obj = y if objective is None else objective(y)
                cons = None
                if constraints is not None:
                    cons = torch.stack([c(y) for c in constraints], dim=-1)
        else:
            y = acq_function(X)
            obj = y if objective is None else objective(y)
            cons = None
            if constraints is not None:
                raise UnsupportedError("Outcome constraints operate on torch tensors.")
        return to_flat(obj, "objectives"), None if cons is None else to_flat(cons, "constraints")

    return evaluate


def optimize_with_nsgaii(
    acq_function: Any,
    bounds: Any,
    num_objectives: int,
    q: int | None = None,
    ref_point: Any = None,
    objective: Any = None,
    constraints: list[Callable[[Any], Any]] | None = None,
    inequality_constraints: list[tuple[Any, Any, float]] | None = None,
    population_size: int = 250,
    max_gen: int | None = None,
    seed: int | None = None,
    fixed_features: dict[int, float] | None = None,
    max_attempts: int = 2,
    discrete_choices: dict[int, list[float]] | None = None,
    post_processing_func: Callable[[Any], Any] | None = None,
) -> tuple[Any, Any]:
    """Optimize a multi-output acquisition function with NSGA-II, returning
    the Pareto set and front (all objectives are maximized).

    Args:
        acq_function: The ``MultiOutputAcquisitionFunction`` to optimize.
        bounds: A ``2 x d`` array of lower and upper bounds of ``X``.
        num_objectives: The number of objectives.
        q: The number of candidates (a hypervolume-maximizing subset of the
            front relative to its nadir, or the front filled up with random
            dominated points). If None, return the whole front.
        ref_point: A lower bound on the objectives (after ``objective``).
        objective: An ``MCMultiOutputObjective`` (callable on torch tensors)
            applied to the acquisition values.
        constraints: Outcome constraints (callables on torch tensors,
            feasible iff negative) of the acquisition values.
        inequality_constraints: Parameter constraints ``(indices,
            coefficients, rhs)``: ``sum_i X[indices[i]] * coefficients[i] >=
            rhs``.
        population_size: The population size.
        max_gen: The number of generations (None: pymoo's default
            termination).
        seed: The random seed.
        fixed_features: ``{feature_index: value}`` for fixed inputs.
        max_attempts: The number of runs if NSGA-II finds no feasible point.
        discrete_choices: Allowed values of discrete inputs, enforced after
            every generation by rounding to the nearest allowed value.
        post_processing_func: A function applied to the result ``X`` (``n x
            d``), after which the objectives are re-evaluated.

    Returns:
        A two-element tuple containing the Pareto set ``X`` and front ``Y``.
    """
    kind = first_kind(bounds)
    bounds_flat = to_flat(bounds, "bounds")
    if len(bounds_flat[1]) != 2 or bounds_flat[1][0] != 2:
        raise ValueError(f"bounds must be a `2 x d` tensor, got shape {bounds_flat[1]}.")
    ref = None if ref_point is None else _floats(ref_point, "ref_point")
    ineq = [
        (_ints(idx, "indices"), _floats(coef, "coefficients"), float(rhs))
        for idx, coef, rhs in (inequality_constraints or [])
    ]
    fixed = [(int(i), float(v)) for i, v in (fixed_features or {}).items()]
    if any(i < 0 for i, _ in fixed):
        raise ValueError("All indices of `fixed_features` should be non-negative.")
    discrete = [(int(i), [float(v) for v in vals]) for i, vals in (discrete_choices or {}).items()]
    native = getattr(acq_function, "_rs_mo", None)
    use_native = native is not None and objective is None and constraints is None
    if not use_native:
        from botorch_rs._torch_bridge import require_torch

        if objective is not None or constraints is not None:
            require_torch()
    x_pair, y_pair, messages, _ = _rs.optimize_with_nsgaii(
        bounds_flat,
        int(num_objectives),
        acqf=native if use_native else None,
        evaluate=None if use_native else _evaluate_with(acq_function, objective, constraints, kind),
        q=None if post_processing_func is not None else q,
        ref_point=ref,
        inequality_constraints=ineq,
        population_size=int(population_size),
        max_gen=None if max_gen is None else int(max_gen),
        seed=None if seed is None else int(seed),
        fixed_features=fixed,
        max_attempts=int(max_attempts),
        discrete_choices=discrete,
    )
    X = from_pair(x_pair, kind)
    Y = from_pair(y_pair, kind)
    if post_processing_func is None:
        for message in messages:
            warnings.warn(message, BotorchWarning, stacklevel=2)
        return X, Y
    return _post_process(acq_function, objective, X, q, post_processing_func)


def _post_process(
    acq_function: Any,
    objective: Any,
    X: Any,
    q: int | None,
    post_processing_func: Callable[[Any], Any],
) -> tuple[Any, Any]:
    """BoTorch's handling of ``post_processing_func``: re-evaluates the
    objectives at the processed front and selects the Pareto set (and ``q``
    points) of the processed points."""
    X = post_processing_func(X)
    kind = kind_of(X)
    data, (n, d) = to_flat(X, "X")
    values = acq_function(from_flat(data, (n, 1, d), kind))
    Y = values if objective is None else objective(values)
    mask = [bool(v) for v in to_flat(is_non_dominated(Y, deduplicate=True), "mask")[0]]
    y_data, (_, m) = to_flat(Y, "Y")
    keep = [i for i in range(n) if mask[i]]
    if q is not None and len(keep) > q:
        y_p = from_flat([y_data[i * m + t] for i in keep for t in range(m)], (len(keep), m), kind)
        nadir = [min(y_data[i * m + t] for i in keep) for t in range(m)]
        _, idx = get_hypervolume_maximizing_subset(q, y_p, from_flat(nadir, (m,), kind))
        keep = [keep[int(i)] for i in to_flat(idx, "indices")[0]]
    elif q is not None and len(keep) < q:
        if n >= q:
            dominated = [i for i in range(n) if not mask[i]]
            extra = set(random.sample(dominated, q - len(keep)))
            keep = [i for i in range(n) if mask[i] or i in extra]
        else:
            warnings.warn(f"NSGA-II only returned {n} points.", BotorchWarning, stacklevel=3)
            return X, Y
    x_out = [data[i * d + j] for i in keep for j in range(d)]
    y_out = [y_data[i * m + t] for i in keep for t in range(m)]
    return from_flat(x_out, (len(keep), d), kind), from_flat(y_out, (len(keep), m), kind)


__all__ = ["optimize_with_nsgaii"]

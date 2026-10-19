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

import importlib
import random
import sys
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


def _pymoo_base(module: str, name: str) -> type:
    """pymoo's ``Problem`` / ``Repair`` base class if pymoo is installed (as
    in BoTorch), else ``object``."""
    try:
        mod = importlib.import_module(module)
    except ImportError:
        return object
    return getattr(mod, name)


def _build_pymoo_classes() -> dict[str, type]:
    """``DiscreteParameterRepair`` and ``BotorchPymooProblem``, subclasses of
    pymoo's ``Repair`` and ``Problem`` if pymoo is installed (imported on
    first access)."""
    _Repair: Any = _pymoo_base("pymoo.core.repair", "Repair")
    _Problem: Any = _pymoo_base("pymoo.core.problem", "Problem")


    class DiscreteParameterRepair(_Repair):  # type: ignore[misc]
        """Pymoo Repair operator that rounds discrete parameters to valid values.

        This repair operator is applied after each generation to ensure that
        discrete parameters are snapped to their nearest allowed values.

        A subclass of pymoo's ``Repair`` if pymoo is installed; botorch_rs's
        NSGA-II does not need it.
        """

        def __init__(
            self,
            discrete_choices: dict[int, list[float]],
        ) -> None:
            """Initialize the repair operator.

            Args:
                discrete_choices: A mapping from dimension index to allowed discrete
                    values. Only dimensions in this mapping will be rounded.
            """
            if _Repair is not object:
                super().__init__()
            self.discrete_choices = discrete_choices

        def _do(self, problem: Any, X: Any, **kwargs: Any) -> Any:
            """Round discrete dimensions to nearest allowed values.

            Args:
                problem: The pymoo Problem instance.
                X: A ``pop_size x n_var`` array of candidate solutions.
                **kwargs: Additional keyword arguments.

            Returns:
                The repaired array with discrete dimensions rounded.
            """
            np = sys.modules.get("numpy")
            if np is not None and isinstance(X, np.ndarray):
                for dim, allowed_values in self.discrete_choices.items():
                    allowed = np.array(allowed_values)
                    vals = X[:, dim]
                    distances = np.abs(vals[:, np.newaxis] - allowed[np.newaxis, :])
                    nearest_idx = np.argmin(distances, axis=1)
                    X[:, dim] = allowed[nearest_idx]
                return X
            for row in X:
                for dim, allowed_values in self.discrete_choices.items():
                    v = float(row[dim])
                    best = min(range(len(allowed_values)), key=lambda k: abs(v - allowed_values[k]))
                    row[dim] = allowed_values[best]
            return X

        if _Repair is object:

            def do(self, problem: Any, X: Any, **kwargs: Any) -> Any:
                """Applies the repair (pymoo's ``Repair.do`` without pymoo)."""
                return self._do(problem, X, **kwargs)


    class BotorchPymooProblem(_Problem):  # type: ignore[misc]
        def __init__(
            self,
            n_var: int,
            n_obj: int,
            xl: Any,
            xu: Any,
            acqf: Any,
            dtype: Any = None,
            device: Any = None,
            ref_point: Any = None,
            objective: Any = None,
            constraints: list[Callable[[Any], Any]] | None = None,
            inequality_constraints: list[tuple[Any, Any, float]] | None = None,
        ) -> None:
            """PyMOO problem for optimizing the model posterior mean using NSGA-II.

            This is instantiated and used within ``optimize_with_nsgaii`` to define
            the optimization problem to interface with pymoo.

            This assumes maximization of all objectives.

            A subclass of pymoo's ``Problem`` if pymoo is installed (else a plain
            class with ``_evaluate``); botorch_rs's ``optimize_with_nsgaii`` runs
            NSGA-II natively and does not use it.

            Args:
                n_var: The number of tunable parameters (``d``).
                n_obj: The number of objectives.
                xl: A ``d``-dim np.ndarray of lower bounds for each tunable parameter.
                xu: A ``d``-dim np.ndarray of upper bounds for each tunable parameter.
                acqf: A MultiOutputAcquisitionFunction.
                dtype: The torch dtype.
                device: The torch device.
                ref_point: A list or tensor with ``m`` elements representing the
                    reference point (in the outcome space), which is treated as a
                    lower bound on the objectives, after applying ``objective`` to
                    the samples.
                objective: The MCMultiOutputObjective under which the samples are
                    evaluated. Defaults to ``IdentityMultiOutputObjective()``.
                    This can be used to determine which outputs of the
                    MultiOutputAcquisitionFunction should be used as
                    objectives/constraints in NSGA-II.
                constraints: A list of callables, each mapping a Tensor of dimension
                    ``sample_shape x batch-shape x q x m`` to a Tensor of dimension
                    ``sample_shape x batch-shape x q``, where negative values imply
                    feasibility.
                inequality_constraints: A list of tuples (indices, coefficients, rhs),
                    representing inequality constraints of the form
                    ``sum_i (X[indices[i]] * coefficients[i]) >= rhs``. These are
                    parameter-space constraints (as opposed to outcome-space
                    constraints).
            """
            num_constraints = 0 if constraints is None else len(constraints)
            if ref_point is not None:
                num_constraints += len(to_flat(ref_point, "ref_point")[0])
            if inequality_constraints is not None:
                num_constraints += len(inequality_constraints)
            if _Problem is not object:
                np = importlib.import_module("numpy")
                super().__init__(
                    n_var=n_var,
                    n_obj=n_obj,
                    n_ieq_constr=num_constraints,
                    xl=xl,
                    xu=xu,
                    type_var=np.double,
                )
            else:
                self.n_var = n_var
                self.n_obj = n_obj
                self.n_ieq_constr = num_constraints
                self.xl = xl
                self.xu = xu
            from botorch_rs.acquisition.multi_objective.objective import (
                IdentityMCMultiOutputObjective,
            )

            self.botorch_acqf = acqf
            self.botorch_ref_point = ref_point
            self.botorch_objective = IdentityMCMultiOutputObjective() if objective is None else objective
            self.botorch_constraints = constraints
            self.botorch_inequality_constraints = inequality_constraints
            self.torch_dtype = dtype
            self.torch_device = device

        def _evaluate(self, x: Any, out: dict[str, Any], *args: Any, **kwargs: Any) -> None:
            """Evaluate x with respect to the objective/constraints."""
            torch = importlib.import_module("torch")
            X = torch.as_tensor(x).to(dtype=self.torch_dtype or torch.float64, device=self.torch_device)
            with torch.no_grad():
                # eval in batch mode, since all we need is the mean and this helps
                # avoid ill-conditioning
                y = self.botorch_acqf(X=X.unsqueeze(-2))
            obj = self.botorch_objective(y)
            # negate the objectives, since we want to maximize this function
            out["F"] = -obj.cpu().numpy()
            constraint_vals = None
            if self.botorch_constraints is not None:
                constraint_vals = torch.stack([c(y) for c in self.botorch_constraints], dim=-1)
            if self.botorch_ref_point is not None:
                # add constraints for the ref point
                ref_constraints = torch.as_tensor(self.botorch_ref_point).to(obj) - obj
                if constraint_vals is not None:
                    constraint_vals = torch.cat([constraint_vals, ref_constraints], dim=-1)
                else:
                    constraint_vals = ref_constraints
            if self.botorch_inequality_constraints is not None:
                ineq_constraint_vals = []
                for indices, coefficients, rhs in self.botorch_inequality_constraints:
                    lhs = (X[:, torch.as_tensor(indices)] * torch.as_tensor(coefficients).to(X)).sum(
                        dim=-1
                    )
                    # Transform to G(x) <= 0 form: rhs - lhs <= 0
                    ineq_constraint_vals.append(rhs - lhs)
                ineq_constraints = torch.stack(ineq_constraint_vals, dim=-1)
                if constraint_vals is not None:
                    constraint_vals = torch.cat([constraint_vals, ineq_constraints], dim=-1)
                else:
                    constraint_vals = ineq_constraints
            if constraint_vals is not None:
                out["G"] = constraint_vals.cpu().numpy()

    for cls in (DiscreteParameterRepair, BotorchPymooProblem):
        cls.__module__ = __name__
        cls.__qualname__ = cls.__name__
    return {
        "DiscreteParameterRepair": DiscreteParameterRepair,
        "BotorchPymooProblem": BotorchPymooProblem,
    }


def __getattr__(name: str) -> Any:
    if name in ("DiscreteParameterRepair", "BotorchPymooProblem"):
        classes = _build_pymoo_classes()
        globals().update(classes)
        return classes[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["BotorchPymooProblem", "DiscreteParameterRepair", "optimize_with_nsgaii"]

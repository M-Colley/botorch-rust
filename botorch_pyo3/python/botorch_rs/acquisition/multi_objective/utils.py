"""Utilities of the multi-objective acquisition functions
(``botorch.acquisition.multi_objective.utils``):
``get_default_partitioning_alpha``,
``prune_inferior_points_multi_objective`` and the sampling utilities of the
information-theoretic acquisition functions, ``sample_optimal_points``
(Pareto sets and fronts of Matheron paths of the posterior),
``random_search_optimizer`` and ``compute_sample_box_decomposition``.
"""

from __future__ import annotations

import warnings
from typing import Any

from botorch_rs import _botorch_rs as _rs
from botorch_rs._arrays import from_flat, from_pair, kind_of, to_flat
from botorch_rs.acquisition.acquisition import check_unsupported
from botorch_rs.acquisition.multi_objective.base import ref_point_arg
from botorch_rs.acquisition.multi_objective.objective import mo_objective_arg
from botorch_rs.acquisition.objective import constraints_arg
from botorch_rs.exceptions.errors import UnsupportedError
from botorch_rs.exceptions.warnings import BotorchWarning
from botorch_rs.models.gp_regression import SingleTaskGP
from botorch_rs.models.model_list_gp_regression import ModelListGP
from botorch_rs.sampling.normal import SobolQMCNormalSampler
from botorch_rs.utils.multi_objective.box_decompositions.dominated import DominatedPartitioning


def get_default_partitioning_alpha(num_objectives: int) -> float:
    r"""Determines an approximation level based on the number of objectives.

    If ``alpha`` is 0, ``FastNondominatedPartitioning`` should be used.
    Otherwise, an approximate ``NondominatedPartitioning`` should be used with
    approximation level ``alpha``.

    Args:
        num_objectives: the number of objectives.

    Returns:
        The approximation level ``alpha``.
    """
    if num_objectives <= 4:
        return 0.0
    elif num_objectives > 6:
        warnings.warn("EHVI works best for less than 7 objectives.", BotorchWarning, stacklevel=3)
    return 10 ** (-2 if num_objectives >= 6 else -3)


def prune_inferior_points_multi_objective(
    model: SingleTaskGP | ModelListGP,
    X: Any,
    ref_point: Any,
    objective: Any = None,
    constraints: Any = None,
    num_samples: int = 2048,
    max_frac: float = 1.0,
    marginalize_dim: int | None = None,
    *,
    sampler: SobolQMCNormalSampler | None = None,
) -> Any:
    r"""Prune points from an input array that are unlikely to be Pareto
    optimal.

    Returns the subset of points in ``X`` (``n x d``) that have a non-zero
    probability of being Pareto optimal, better than the reference point and
    feasible, estimated with ``num_samples`` posterior samples of the
    objectives (infeasible points are set to the reference point in each
    sample). At most ``ceil(max_frac * n)`` points are returned (the most
    probable ones).

    Without a ``sampler`` (a keyword-only botorch_rs extension providing the
    base samples), the base samples are drawn with a random seed, as in
    BoTorch.
    """
    check_unsupported(marginalize_dim=marginalize_dim)
    if not isinstance(model, (SingleTaskGP, ModelListGP)):
        raise UnsupportedError(
            "prune_inferior_points_multi_objective requires a botorch_rs SingleTaskGP or "
            "ModelListGP."
        )
    flat = to_flat(X, "X")
    if len(flat[1]) > 2:
        raise UnsupportedError(
            "Batched inputs `X` are currently unsupported by "
            "`prune_inferior_points_multi_objective`"
        )
    if model.batch_shape:
        raise UnsupportedError(
            "Models with multiple batch dims are currently unsupported by "
            "`prune_inferior_points_multi_objective`."
        )
    n = flat[1][0] if flat[1] else 0
    spec = mo_objective_arg(objective)
    cons, _ = constraints_arg(constraints, 1e-3)
    handle = model._acquisition_model(None)
    base_samples = None
    if sampler is not None:
        if not isinstance(sampler, SobolQMCNormalSampler):
            raise UnsupportedError("`sampler` must be a botorch_rs SobolQMCNormalSampler.")
        num_samples = sampler.num_samples
        base_samples = sampler._rs.base_samples(n * handle.num_outputs) if n > 0 else None
    pruned = _rs.prune_inferior_points_multi_objective(
        handle,
        flat,
        ref_point_arg(ref_point),
        objective=spec,
        constraints=cons,
        num_samples=int(num_samples),
        max_frac=float(max_frac),
        base_samples=base_samples,
    )
    return from_pair(pruned, kind_of(X))


#: The partitionings of ``compute_sample_box_decomposition`` by class name.
_PARTITIONINGS = {
    "DominatedPartitioning": "dominated",
    "FastNondominatedPartitioning": "fast_nondominated",
    "NondominatedPartitioning": "nondominated",
}


def compute_sample_box_decomposition(
    pareto_fronts: Any,
    partitioning: Any = DominatedPartitioning,
    maximize: bool = True,
    num_constraints: int = 0,
) -> Any:
    r"""The box decompositions of sampled Pareto fronts: the bounds of the
    hyper-rectangles of the dominated space of each front
    (``num_pareto_samples x num_pareto_points x M``), with a reference point
    of ``-1e10`` (minimization: of the negated fronts, negated and flipped
    back), padded with empty boxes to the largest number of boxes ``J``.
    With ``M = 1``, the single box ``[-1e10, max]``. With ``num_constraints =
    K > 0``, each box is extended by ``[-1e10, 0]`` in ``K`` constraint
    dimensions and the infeasible box ``[-1e10, 1e10]^M x [0, 1e10]^K`` is
    appended.

    ``partitioning`` is ``DominatedPartitioning`` (the default),
    ``FastNondominatedPartitioning`` or ``NondominatedPartitioning`` (BoTorch's
    or botorch_rs's class). Returns a ``num_pareto_samples x 2 x J x (M + K)``
    array.
    """
    name = getattr(partitioning, "__name__", "")
    if name not in _PARTITIONINGS:
        raise UnsupportedError(
            "compute_sample_box_decomposition supports DominatedPartitioning, "
            f"FastNondominatedPartitioning and NondominatedPartitioning, got {partitioning!r}."
        )
    flat = to_flat(pareto_fronts, "pareto_fronts")
    if len(flat[1]) != 3:
        raise UnsupportedError(
            "Currently this only supports Pareto fronts of the shape "
            "`num_pareto_samples x num_pareto_points x num_objectives`."
        )
    cells = _rs.compute_sample_box_decomposition(
        flat, _PARTITIONINGS[name], 0.0, bool(maximize), int(num_constraints)
    )
    return from_pair(cells, kind_of(pareto_fronts))


def _path_model_handle(model: Any) -> Any:
    """The extension handle of a single-path botorch_rs ``MatheronPathModel``
    (evaluated in Rust), else ``None``."""
    from botorch_rs.models.deterministic import MatheronPathModel

    if isinstance(model, MatheronPathModel) and not model.sample_shape:
        return model._handle
    return None


def random_search_optimizer(
    model: Any,
    bounds: Any,
    num_points: int,
    maximize: bool,
    pop_size: int = 1024,
    max_tries: int = 10,
    *,
    seed: int | None = None,
    candidates: Any = None,
) -> tuple[Any, Any]:
    r"""Optimizes a deterministic model by random search: each try evaluates
    ``model.posterior(X).mean`` at ``pop_size`` scrambled Sobol points ``X``
    in ``bounds`` (``2 x d``) and keeps the non-dominated points of the
    previous optima and the new points (in that order), until at least
    ``num_points`` points are non-dominated (``RuntimeError`` after
    ``max_tries`` tries). Returns the first ``num_points`` optimal inputs
    (``num_points x d``) and outputs (``num_points x M``).

    A botorch_rs ``MatheronPathModel`` is evaluated in Rust; any other model
    is called from Rust with arrays of the type of ``bounds``.

    botorch_rs extensions (keyword-only): ``seed`` of the Sobol points (try
    ``k`` uses stream ``k``; ``None`` draws a random seed) or explicit
    ``candidates``, one ``pop_size x d`` array per try (e.g. BoTorch's
    draws).
    """
    from botorch_rs.sampling.pathwise.prior_samplers import seed_arg

    kind = kind_of(bounds)
    handle = _path_model_handle(model)
    if handle is None:

        def evaluate(pair: Any) -> Any:
            X = from_flat(pair[0], pair[1], kind)
            return to_flat(model.posterior(X).mean, "Y")

        target: Any = evaluate
    else:
        target = handle
    cands = None if candidates is None else [to_flat(c, "candidates") for c in candidates]
    x, y = _rs.random_search_optimizer(
        target,
        to_flat(bounds, "bounds"),
        int(num_points),
        bool(maximize),
        int(pop_size),
        int(max_tries),
        seed_arg(seed),
        cands,
    )
    return from_pair(x, kind), from_pair(y, kind)


def sample_optimal_points(
    model: Any,
    bounds: Any,
    num_samples: int,
    num_points: int,
    optimizer: Any = random_search_optimizer,
    maximize: bool = True,
    optimizer_kwargs: dict[str, Any] | None = None,
    *,
    num_features: int = 1024,
    seed: int | None = None,
    draws: Any = None,
    candidates: Any = None,
) -> tuple[Any, Any]:
    r"""Optimal inputs and outputs of ``num_samples`` Matheron paths of the
    posterior of ``model`` (``get_matheron_path_model``), each optimized by
    ``optimizer`` (``optimizer(model=path_model, bounds=bounds,
    num_points=num_points, maximize=maximize, **optimizer_kwargs)``). Returns
    ``num_samples x num_points x d`` Pareto sets and ``num_samples x
    num_points x M`` Pareto fronts.

    With the default ``random_search_optimizer``, the paths are drawn and
    optimized in Rust, in parallel; other optimizers are called with one
    ``MatheronPathModel`` at a time.

    botorch_rs extensions (keyword-only): ``num_features`` of the paths
    (BoTorch's default 1024), ``seed`` (sample ``i`` draws its path from
    stream ``2i`` and its Sobol points from stream ``2i + 1``), or explicit
    ``draws`` (one list of per-output ``(frequencies, weights, noise)`` per
    sample, see ``draw_matheron_paths``) and ``candidates`` (one list of
    candidate sets per sample, see ``random_search_optimizer``).
    """
    from botorch_rs.models.deterministic import MatheronPathModel
    from botorch_rs.sampling.pathwise.prior_samplers import draws_arg, seed_arg

    if not isinstance(model, (SingleTaskGP, ModelListGP)):
        raise UnsupportedError(
            "sample_optimal_points requires a botorch_rs SingleTaskGP or ModelListGP."
        )
    if model.num_outputs == 1 and num_points > 1:
        raise UnsupportedError("For single-objective optimization `num_points` should be 1.")
    if isinstance(model, ModelListGP) and len(model.models) != model.num_outputs:
        # BoTorch's path models (``MatheronPathModel``) do not support them.
        raise UnsupportedError("A model-list of multi-output models is not supported.")
    kwargs = dict(optimizer_kwargs or {})
    kind = kind_of(bounds)
    rs_model = model._acquisition_model(None)
    path_draws: Any = (
        None
        if draws is None
        else [draws_arg(d, ("frequencies", "weights", "noise")) for d in draws]
    )
    if optimizer is random_search_optimizer:
        unknown = sorted(set(kwargs) - {"pop_size", "max_tries"})
        if unknown:
            raise TypeError(f"Unexpected arguments of random_search_optimizer: {unknown}.")
        cands = (
            None
            if candidates is None
            else [[to_flat(c, "candidates") for c in cs] for cs in candidates]
        )
        ps, pf = _rs.sample_optimal_points(
            rs_model,
            to_flat(bounds, "bounds"),
            int(num_samples),
            int(num_points),
            bool(maximize),
            int(kwargs.get("pop_size", 1024)),
            int(kwargs.get("max_tries", 10)),
            int(num_features),
            seed_arg(seed),
            path_draws,
            cands,
        )
        return from_pair(ps, kind), from_pair(pf, kind)
    if candidates is not None:
        raise UnsupportedError("`candidates` only applies to random_search_optimizer.")
    handles = _rs.sample_optimal_point_paths(
        rs_model, int(num_samples), int(num_features), seed_arg(seed), path_draws
    )
    sets: list[float] = []
    fronts: list[float] = []
    d = to_flat(bounds, "bounds")[1][-1]
    for handle in handles:
        path_model = MatheronPathModel._from_handle(model, handle)
        ps_i, pf_i = optimizer(
            model=path_model, bounds=bounds, num_points=num_points, maximize=maximize, **kwargs
        )
        sets.extend(to_flat(ps_i, "pareto_set")[0])
        fronts.extend(to_flat(pf_i, "pareto_front")[0])
    return (
        from_flat(sets, (num_samples, num_points, d), kind),
        from_flat(fronts, (num_samples, num_points, model.num_outputs), kind),
    )


__all__ = [
    "compute_sample_box_decomposition",
    "get_default_partitioning_alpha",
    "prune_inferior_points_multi_objective",
    "random_search_optimizer",
    "sample_optimal_points",
]

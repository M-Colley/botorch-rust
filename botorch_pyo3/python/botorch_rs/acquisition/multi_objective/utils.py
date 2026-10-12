"""Utilities of the multi-objective acquisition functions
(``botorch.acquisition.multi_objective.utils``):
``get_default_partitioning_alpha`` and
``prune_inferior_points_multi_objective``.

``compute_sample_box_decomposition``, ``random_search_optimizer`` and
``sample_optimal_points`` need pathwise posterior samples (not ported).
"""

from __future__ import annotations

import warnings
from typing import Any

from botorch_rs import _botorch_rs as _rs
from botorch_rs._arrays import from_pair, kind_of, to_flat
from botorch_rs.acquisition.acquisition import check_unsupported
from botorch_rs.acquisition.multi_objective.base import ref_point_arg
from botorch_rs.acquisition.multi_objective.objective import mo_objective_arg
from botorch_rs.acquisition.objective import constraints_arg
from botorch_rs.exceptions.errors import UnsupportedError
from botorch_rs.exceptions.warnings import BotorchWarning
from botorch_rs.models.gp_regression import SingleTaskGP
from botorch_rs.models.model_list_gp_regression import ModelListGP
from botorch_rs.sampling.normal import SobolQMCNormalSampler


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


__all__ = ["get_default_partitioning_alpha", "prune_inferior_points_multi_objective"]

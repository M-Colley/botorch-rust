"""Acquisition utilities (``botorch.acquisition.utils``)."""

from __future__ import annotations

from typing import Any

from botorch_rs import _botorch_rs as _rs
from botorch_rs._arrays import from_pair, kind_of, to_flat
from botorch_rs.acquisition._fidelity import (
    expand_trace_observations,
    project_to_target_fidelity,
)
from botorch_rs.acquisition.acquisition import check_unsupported, is_acquisition_model
from botorch_rs.acquisition.objective import constraints_arg, objective_arg
from botorch_rs.exceptions.errors import UnsupportedError
from botorch_rs.models.gp_regression import SingleTaskGP
from botorch_rs.models.model_list_gp_regression import ModelListGP
from botorch_rs.sampling.normal import SobolQMCNormalSampler


def prune_inferior_points(
    model: SingleTaskGP | ModelListGP,
    X: Any,
    objective: Any = None,
    posterior_transform: Any = None,
    constraints: Any = None,
    num_samples: int = 2048,
    max_frac: float = 1.0,
    sampler: SobolQMCNormalSampler | None = None,
    marginalize_dim: int | None = None,
) -> Any:
    r"""The points of ``X`` (``n x d``) that have a non-zero estimated
    probability of being the best point under the model's joint posterior.

    The objective values of the samples (``objective``, default: the single
    output) of points violating an outcome constraint (``constraints``,
    callables ``samples -> slack``, feasible iff ``<= 0``) are set below the
    worst value; if no point is feasible in any sample, the points most likely
    to be feasible (under a smoothed feasibility indicator) are kept, as in
    BoTorch.

    If more than ``ceil(max_frac * n)`` points qualify, the most frequent
    maximizers are kept (BoTorch's implementation selects wrong points in that
    case; with the default ``max_frac=1`` both agree). Multi-output models
    require an ``objective`` or a scalarizing ``posterior_transform``. Without
    a ``sampler``, the samples are drawn with a random seed (like BoTorch).
    """
    check_unsupported(marginalize_dim=marginalize_dim)
    if not is_acquisition_model(model):
        raise UnsupportedError(
            "prune_inferior_points requires a botorch_rs SingleTaskGP, ModelListGP or LCEMGP "
            "(or a PosteriorMeanModel or GenericDeterministicModel)."
        )
    if model.num_outputs > 1 and objective is None and posterior_transform is None:
        raise UnsupportedError(
            "Models with multiple outputs require an objective or posterior_transform."
        )
    spec = objective_arg(objective)
    cons, _ = constraints_arg(constraints, 1e-3)
    flat = to_flat(X, "X")
    n = flat[1][0] if flat[1] else 0
    handle = model._acquisition_model(posterior_transform)
    base_samples = None
    if sampler is not None:
        if not isinstance(sampler, SobolQMCNormalSampler):
            raise UnsupportedError("`sampler` must be a botorch_rs SobolQMCNormalSampler.")
        num_samples = sampler.num_samples
        base_samples = sampler._rs.base_samples(n * handle.num_outputs) if n > 0 else None
    pruned = _rs.prune_inferior_points(
        handle,
        flat,
        num_samples,
        float(max_frac),
        None,
        base_samples,
        objective=spec,
        constraints=cons,
    )
    return from_pair(pruned, kind_of(X))


def get_optimal_samples(
    model: SingleTaskGP | ModelListGP,
    bounds: Any,
    num_optima: int,
    raw_samples: int = 1024,
    num_restarts: int = 20,
    posterior_transform: Any = None,
    objective: Any = None,
    return_transformed: bool = False,
    *,
    num_features: int = 1024,
    seed: int | None = None,
    draws: Any = None,
    candidates: Any = None,
    maxiter: int | None = None,
) -> tuple[Any, Any]:
    r"""Draws ``num_optima`` sample paths (Matheron paths) from the posterior
    and maximizes them: every path is evaluated at ``raw_samples``
    quasi-random points, and its ``num_restarts`` best points are optimized
    with L-BFGS-B (one joint problem over all paths, as BoTorch's
    ``gen_candidates_scipy(..., use_parallel_mode=False)``). Returns the
    optimal inputs (``num_optima x d``) and outputs (``num_optima x m``, or
    the transformed values ``num_optima x 1`` with ``return_transformed``).

    botorch_rs extensions (keyword-only): ``num_features`` of the paths,
    ``seed`` of the paths and of the Sobol points, explicit ``draws`` of the
    paths (one ``(frequencies, weights, noise)`` per output with
    ``num_optima`` samples each, see ``draw_matheron_paths``) and
    ``candidates`` (``raw_samples x d``, BoTorch's Sobol points), and
    ``maxiter`` of L-BFGS-B (default 2000).
    """
    from botorch_rs.acquisition.objective import ScalarizedPosteriorTransform, transform_arg
    from botorch_rs.sampling.pathwise.prior_samplers import draws_arg, seed_arg

    if not is_acquisition_model(model) or not isinstance(model, (SingleTaskGP, ModelListGP)):
        raise UnsupportedError(
            "get_optimal_samples requires a botorch_rs SingleTaskGP or ModelListGP."
        )
    if posterior_transform is not None and not isinstance(
        posterior_transform, ScalarizedPosteriorTransform
    ):
        raise ValueError(
            "Only the ScalarizedPosteriorTransform is supported for get_optimal_samples."
        )
    if posterior_transform is not None and objective is not None:
        raise ValueError("Only one of `posterior_transform` and `objective` can be specified.")
    kind = kind_of(bounds)
    x, f = _rs.get_optimal_samples(
        model._acquisition_model(None),
        to_flat(bounds, "bounds"),
        int(num_optima),
        raw_samples=int(raw_samples),
        num_restarts=int(num_restarts),
        transform=transform_arg(posterior_transform, model.num_outputs),
        objective=objective_arg(objective),
        return_transformed=bool(return_transformed),
        num_features=int(num_features),
        seed=seed_arg(seed),
        draws=draws_arg(draws, ("frequencies", "weights", "noise")),
        candidates=None if candidates is None else to_flat(candidates, "candidates"),
        maxiter=None if maxiter is None else int(maxiter),
    )
    return from_pair(x, kind), from_pair(f, kind)


__all__ = [
    "expand_trace_observations",
    "get_optimal_samples",
    "project_to_target_fidelity",
    "prune_inferior_points",
]

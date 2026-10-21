"""Acquisition utilities (``botorch.acquisition.utils``)."""

from __future__ import annotations

import importlib
import warnings
from typing import Any

from botorch_rs import _botorch_rs as _rs
from botorch_rs._arrays import from_flat, from_pair, is_torch_tensor, kind_of, to_flat
from botorch_rs.acquisition._fidelity import (
    expand_trace_observations,
    project_to_target_fidelity,
)
from botorch_rs.acquisition.acquisition import check_unsupported, is_acquisition_model
from botorch_rs.acquisition.objective import constraints_arg, objective_arg
from botorch_rs.exceptions.errors import UnsupportedError
from botorch_rs.exceptions.warnings import BotorchWarning
from botorch_rs.models.gp_regression import SingleTaskGP
from botorch_rs.models.model_list_gp_regression import ModelListGP
from botorch_rs.sampling.normal import NormalMCSampler, SobolQMCNormalSampler


def get_acquisition_function(*args: Any, **kwargs: Any) -> None:
    """Moved to :func:`botorch_rs.acquisition.factory.get_acquisition_function`
    (BoTorch's ``DeprecationError``)."""
    from botorch_rs.exceptions.errors import DeprecationError

    raise DeprecationError(
        "`get_acquisition_function` has been moved to `botorch.acquisition.factory`."
    )


def _torch_module() -> Any:
    from botorch_rs._torch_bridge import require_torch

    require_torch()
    return importlib.import_module("torch")


def repeat_to_match_aug_dim(target_tensor: Any, reference_tensor: Any) -> Any:
    """Repeat target_tensor until it has the same first dimension as reference_tensor
    This works regardless of the batch shapes and q.
    This is useful as we sometimes modify sample shapes such as in LearnedObjective.

    Args:
        target_tensor: A ``sample_size x batch_shape x q x m``-dim Tensor
        reference_tensor: A ``(augmented_sample * sample_size) x batch_shape x q``-dim
            Tensor. ``augmented_sample`` could be 1.

    Returns:
        The content of ``target_tensor`` potentially repeated so that its first
        dimension matches that of ``reference_tensor``.
        The shape will be ``(augmented_sample * sample_size) x batch_shape x q x m``.
    """
    tdata, tshape = to_flat(target_tensor, "target_tensor")
    rshape = to_flat(reference_tensor, "reference_tensor")[1]
    augmented_sample_num, remainder = divmod(rshape[0], tshape[0])
    if remainder != 0:
        raise ValueError(
            "The first dimension of reference_tensor must be a multiple of target_tensor's."
        )
    if is_torch_tensor(target_tensor):
        repeat_size = (augmented_sample_num,) + (1,) * (target_tensor.ndim - 1)
        return target_tensor.repeat(*repeat_size)
    data = list(tdata) * augmented_sample_num
    return from_flat(data, (tshape[0] * augmented_sample_num, *tshape[1:]), kind_of(target_tensor))


def project_to_sample_points(X: Any, sample_points: Any) -> Any:
    r"""Augment ``X`` with sample points at which to take weighted average.

    Args:
        X: A ``batch_shape x 1 x d``-dim Tensor of with one d`-dim design points
            for each t-batch.
        sample_points: ``p x d'``-dim Tensor (``d' < d``) of ``d'``-dim sample points at
            which to compute the expectation. The ``d'``-dims refer to the trailing
            columns of X.
    Returns:
        A ``batch_shape x p x d`` Tensor where the q-batch includes the ``p`` sample points.
    """
    if is_torch_tensor(X):
        batch_shape = X.shape[:-2]
        p, d_prime = sample_points.shape
        X_new = X.repeat(*(1 for _ in batch_shape), p, 1)  # batch_shape x p x d
        X_new[..., -d_prime:] = sample_points
        return X_new
    xdata, xshape = to_flat(X, "X")
    sdata, sshape = to_flat(sample_points, "sample_points")
    p, d_prime = sshape
    d = xshape[-1]
    nb = len(xdata) // (xshape[-2] * d)
    out: list[float] = []
    for b in range(nb):
        row = list(xdata[b * xshape[-2] * d : b * xshape[-2] * d + d])
        for j in range(p):
            out.extend(row[: d - d_prime])
            out.extend(sdata[j * d_prime : (j + 1) * d_prime])
    return from_flat(out, (*xshape[:-2], p, d), kind_of(X))


def compute_best_feasible_objective(
    samples: Any,
    obj: Any,
    constraints: Any,
    model: Any = None,
    objective: Any = None,
    posterior_transform: Any = None,
    X_baseline: Any = None,
    infeasible_obj: Any = None,
) -> Any:
    """Computes the largest ``obj`` value that is feasible under the ``constraints``. If
    ``constraints`` is None, returns the best unconstrained objective value.

    When no feasible observations exist and ``infeasible_obj`` is not ``None``,
    returns ``infeasible_obj`` (potentially reshaped). When no feasible observations
    exist and ``infeasible_obj`` is ``None``, uses ``model``, ``objective``,
    ``posterior_transform``, and ``X_baseline`` to infer and return an
    ``infeasible_obj`` ``M`` s.t. ``M < min_x f(x)``.

    With constraints (callables on torch tensors), the computation uses
    torch; the lower bound of the objective is estimated at points drawn
    with torch's generator, as in BoTorch.

    Args:
        samples: ``(sample_shape) x batch_shape x q x m``-dim posterior samples.
        obj: A ``(sample_shape) x batch_shape x q``-dim Tensor of MC objective values.
        constraints: A list of constraint callables which map posterior samples to
            a scalar. The associated constraint is considered satisfied if this
            scalar is less than zero.
        model: A Model, only required when there are no feasible observations.
        objective: An MCAcquisitionObjective, only optionally used when there are no
            feasible observations.
        posterior_transform: A PosteriorTransform, only optionally used when there are
            no feasible observations.
        X_baseline: A ``batch_shape x d``-dim Tensor of baseline points, only required
            when there are no feasible observations.
        infeasible_obj: A Tensor to be returned when no feasible points exist.

    Returns:
        A ``(sample_shape) x batch_shape``-dim Tensor of best feasible objectives.
    """
    if constraints is None:  # unconstrained case
        if is_torch_tensor(obj):
            torch = _torch_module()
            with torch.no_grad():
                return obj.amax(dim=-1, keepdim=False)
        data, shape = to_flat(obj, "obj")
        n = shape[-1]
        best = [max(data[i : i + n]) for i in range(0, len(data), n)]
        return from_flat(best, tuple(shape[:-1]), kind_of(obj))
    torch = _torch_module()
    kind = kind_of(obj)
    samples_t = _as_tensor(samples)
    obj_t = _as_tensor(obj)
    is_feasible = torch.ones(samples_t.shape[:-1], dtype=torch.bool, device=samples_t.device)
    for constraint in constraints:
        is_feasible = is_feasible.logical_and(constraint(samples_t) <= 0)
    if is_feasible.any(dim=-1).all():
        infeasible_value = -torch.inf
    elif infeasible_obj is not None:
        infeasible_value = _as_tensor(infeasible_obj).item()
    else:
        if model is None:
            raise ValueError("Must specify `model` when no feasible observation exists.")
        if X_baseline is None:
            raise ValueError("Must specify `X_baseline` when no feasible observation exists.")
        warnings.warn(
            "When all training points are infeasible, it is better to use "
            "q(Log)ProbabilityOfFeasibility.",
            BotorchWarning,
            stacklevel=2,
        )
        infeasible_value = _estimate_objective_lower_bound(
            model=model,
            objective=objective,
            posterior_transform=posterior_transform,
            X=_as_tensor(X_baseline),
        ).item()
    is_feasible = repeat_to_match_aug_dim(target_tensor=is_feasible, reference_tensor=obj_t)
    obj_t = torch.where(is_feasible, obj_t, infeasible_value)
    with torch.no_grad():
        best = obj_t.amax(dim=-1, keepdim=False)
    if kind.lib == "torch":
        return best
    return from_flat(best.reshape(-1).tolist(), tuple(best.shape), kind)


def _as_tensor(x: Any) -> Any:
    """``x`` as a torch tensor (float64 unless it is one already)."""
    torch = _torch_module()
    if is_torch_tensor(x):
        return x
    data, shape = to_flat(x, "x")
    return torch.tensor(list(data), dtype=torch.float64).reshape(shape)


def _estimate_objective_lower_bound(
    model: Any,
    objective: Any,
    posterior_transform: Any,
    X: Any,
) -> Any:
    """Estimates a lower bound on the objective values by evaluating the at uniformly
    random points in the bounding box of ``X``, returning the 6-sigma lower bound of the
    computed statistics (the points are drawn with torch's generator, as in
    BoTorch).

    Args:
        model: A fitted model.
        objective: An MCAcquisitionObjective with ``m`` outputs.
        posterior_transform: A PosteriorTransform.
        X: A ``n x d``-dim Tensor of design points from which to draw convex
            combinations.

    Returns:
        A ``m``-dimensional Tensor of lower bounds of the objectives.
    """
    torch = _torch_module()
    # we do not have access to ``bounds`` here, so we infer the bounding box
    # from data, expanding by 10% in each direction
    X_lb = X.min(dim=-2, keepdim=True).values
    X_ub = X.max(dim=-2, keepdim=True).values
    X_range = X_ub - X_lb
    X_padding = 0.1 * X_range
    uniform_samples = torch.rand(*X.shape[:-2], 32, X.shape[-1], dtype=X.dtype, device=X.device)
    X_samples = X_lb - X_padding + uniform_samples * (X_range + 2 * X_padding)
    # infeasible cost M is such that -M < min_x f(x), thus
    # 0 < min_x f(x) - (-M), so we should take -M as a lower
    # bound on the best feasible objective
    return -get_infeasible_cost(
        X=X_samples,
        model=model,
        objective=objective,
        posterior_transform=posterior_transform,
    )


def get_infeasible_cost(
    X: Any,
    model: Any,
    objective: Any = None,
    posterior_transform: Any = None,
) -> Any:
    r"""Get infeasible cost for a model and objective.

    For each outcome, computes an infeasible cost ``M`` such that
    ``-M < min_x f(x)`` almost always, so that feasible points are preferred.

    Args:
        X: A ``n x d`` Tensor of ``n`` design points to use in evaluating the
            minimum. These points should cover the design space well. The more
            points the better the estimate, at the expense of added computation.
        model: A fitted botorch_rs model with ``m`` outcomes.
        objective: The objective with which to evaluate the model output (a
            callable on torch tensors).
        posterior_transform: A PosteriorTransform (optional).

    Returns:
        An ``m``-dim tensor of infeasible cost values.

    Example:
        >>> model = SingleTaskGP(train_X, train_Y)
        >>> objective = lambda Y: Y[..., -1] ** 2
        >>> M = get_infeasible_cost(train_X, model, obj)
    """
    torch = _torch_module()
    kind = kind_of(X)
    X_t = _as_tensor(X)
    if objective is None:

        def objective(Y: Any, X: Any = None) -> Any:
            return Y.squeeze(-1)

    posterior = model.posterior(X_t, posterior_transform=posterior_transform)
    mean = _as_tensor(posterior.mean)
    variance = _as_tensor(posterior.variance)
    # We check both the upper and lower bound of the posterior, since the objective
    # may be increasing or decreasing. For objectives that are neither (eg. absolute
    # distance from a target), this should still provide a good bound.
    six_stdv = 6 * variance.clamp_min(0).sqrt()
    lb = torch.stack(
        [
            objective(mean - six_stdv, X=X_t),
            objective(mean + six_stdv, X=X_t),
        ],
        dim=0,
    )
    lb = lb.min(dim=0).values
    if lb.ndim < mean.ndim:
        lb = lb.unsqueeze(-1)
    # Take outcome-wise min. Looping in to handle batched models.
    while lb.dim() > 1:
        lb = lb.min(dim=-2).values
    out = -(lb.clamp_max(0.0))
    if kind.lib == "torch":
        return out
    return from_flat(out.tolist(), tuple(out.shape), kind)


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
    if getattr(model, "_tensor_backend", None) == "torch":
        from botorch_rs.acquisition._torch_pruning import prune_indices

        return X.index_select(
            -2,
            prune_indices(
                model,
                X,
                objective,
                posterior_transform,
                constraints,
                num_samples,
                max_frac,
                sampler,
                marginalize_dim,
            ),
        )
    if hasattr(model, "_ensemble_acquisition"):
        # BoTorch marginalizes the ensemble dimension of an ensemble of GPs
        # (``marginalize_dim``) and samples members with an IndexSampler.
        if marginalize_dim not in (None, -3):
            raise UnsupportedError(
                "botorch_rs marginalizes the ensemble dimension of an ensemble model "
                "(marginalize_dim=-3)."
            )
        if objective is not None or posterior_transform is not None or constraints is not None:
            raise UnsupportedError(
                "prune_inferior_points of an ensemble model with an objective, a posterior "
                "transform or constraints is not supported by botorch_rs."
            )
        return _prune_ensemble(model, X, num_samples, max_frac, sampler)
    check_unsupported(marginalize_dim=marginalize_dim)
    if not is_acquisition_model(model):
        raise UnsupportedError(
            "prune_inferior_points requires a botorch_rs SingleTaskGP, ModelListGP, "
            "MultiTaskGP or LCEMGP (or a PosteriorMeanModel or GenericDeterministicModel)."
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
        if not isinstance(sampler, NormalMCSampler):
            raise UnsupportedError(
                "`sampler` must be a botorch_rs SobolQMCNormalSampler or IIDNormalSampler."
            )
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


def _prune_ensemble(
    model: Any,
    X: Any,
    num_samples: int = 2048,
    max_frac: float = 1.0,
    sampler: Any = None,
    members: list[Any] | None = None,
) -> Any:
    """``prune_inferior_points`` of an ensemble model (single output, no
    objective). For an ensemble of GPs the samples of every member (with the
    same base samples) are averaged over the members before the maximizers
    are counted (BoTorch's ``marginalize_dim``); for an ``EnsembleModel``
    every sample is the value of the member chosen by the index samples of
    an ``IndexSampler``."""
    import math

    from botorch_rs.sampling.index_sampler import IndexSampler

    data, shape = to_flat(X, "X")
    if len(shape) != 2:
        raise UnsupportedError(
            "Batched inputs `X` are currently unsupported by `prune_inferior_points`"
        )
    n = shape[0]
    if n == 0:
        raise ValueError("X must have at least one point.")
    if max_frac <= 0 or max_frac > 1.0:
        raise ValueError(f"max_frac must take values in (0, 1], is {max_frac}")
    max_points = math.ceil(max_frac * n)
    x3 = (data, (1, n, shape[1]))
    counts: dict[int, int] = {}

    def best(row: list[float]) -> int:
        return max(range(n), key=lambda i: (row[i], -i))

    if hasattr(model, "_ensemble_members"):
        gps = model._ensemble_members() if members is None else members
        if sampler is None:
            sampler = SobolQMCNormalSampler(sample_shape=(num_samples,))
        s = sampler.num_samples
        z = sampler._rs.base_samples(n)
        total = [0.0] * (s * n)
        for member in gps:
            samples = member._rs.posterior_samples(x3, z)[0]
            for i, v in enumerate(samples):
                total[i] += v
        k = len(gps)
        for j in range(s):
            b = best([v / k for v in total[j * n : (j + 1) * n]])
            counts[b] = counts.get(b, 0) + 1
    else:
        index_sampler = sampler if sampler is not None else IndexSampler((num_samples,))
        dets = members
        if dets is None:
            dets, _, _ = model._ensemble_acquisition(index_sampler)
        index_sampler._construct_base_samples(model.posterior(X))
        idx = [int(v) for v in to_flat(index_sampler.base_samples, "base_samples")[0]]
        member_best = [best(list(to_flat(m.forward(X), "values")[0])) for m in dets]
        for k in idx:
            counts[member_best[k]] = counts.get(member_best[k], 0) + 1
    keep = sorted(counts)
    if len(keep) > max_points:
        keep = sorted(sorted(keep, key=lambda i: -counts[i])[:max_points])
    rows = [list(data[i * shape[1] : (i + 1) * shape[1]]) for i in keep]
    return from_flat([v for r in rows for v in r], (len(keep), shape[1]), kind_of(X))


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
    "compute_best_feasible_objective",
    "expand_trace_observations",
    "get_acquisition_function",
    "get_infeasible_cost",
    "get_optimal_samples",
    "project_to_sample_points",
    "project_to_target_fidelity",
    "prune_inferior_points",
    "repeat_to_match_aug_dim",
]

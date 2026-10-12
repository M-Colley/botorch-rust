"""Initial conditions of the acquisition-function optimization
(``botorch.optim.initializers``)."""

from __future__ import annotations

import warnings
from typing import Any

from botorch_rs import _botorch_rs as _rs
from botorch_rs._arrays import as_float, first_kind, from_pair, kind_of, to_flat
from botorch_rs.acquisition.acquisition import AcquisitionFunction, check_unsupported
from botorch_rs.exceptions.errors import UnsupportedError
from botorch_rs.exceptions.warnings import BadInitialCandidatesWarning
from botorch_rs.optim.parameter_constraints import linear_constraints_arg

_INT_KEYS = {"seed", "batch_limit", "init_batch_limit", "n_burnin", "n_thinning"}
_FLOAT_KEYS = {"eta", "alpha"}
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
        elif key == "nonnegative":
            out[key] = bool(value)
        elif key in ("sample_around_best", "topn"):
            if value:
                raise UnsupportedError(f"Option {key!r} is not supported by botorch_rs.")
        elif key not in _OPTIMIZER_KEYS:
            raise UnsupportedError(f"Option {key!r} is not supported by botorch_rs.")
    if out.get("init_batch_limit") is not None:
        out["batch_limit"] = out["init_batch_limit"]
    out.pop("init_batch_limit", None)
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
    )
    for message in messages:
        warnings.warn(message, BadInitialCandidatesWarning, stacklevel=2)
    return from_pair(ics, kind)


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
    for indices, coefficients, rhs in constraints:
        idata, ishape = to_flat(indices, "indices")
        ints = [int(v) for v in idata]
        kind = kind_of(indices)
        if len(ishape) == 1:
            if max(ints) >= d:
                raise ValueError(f"Constraint indices cannot exceed the problem dimension {d=}.")
            for p in range(q):
                out.append((_int_array([p * d + j for j in ints], kind), coefficients, rhs))
        else:
            c = ishape[1]
            rows = [ints[r * c : (r + 1) * c] for r in range(ishape[0])]
            if max(r[1] for r in rows) >= d:
                raise ValueError(f"Constraint indices cannot exceed the problem dimension {d=}.")
            out.append((_int_array([r[0] * d + r[1] for r in rows], kind), coefficients, rhs))
    return out


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
    "sample_q_batches_from_polytope",
    "transform_constraints",
]

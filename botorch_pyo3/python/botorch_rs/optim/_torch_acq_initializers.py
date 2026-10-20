"""Restart selection and raw scoring on the acquisition's Torch device."""

from __future__ import annotations

import math
import warnings
from collections.abc import Mapping
from typing import Any

import torch
from torch import Tensor

from botorch_rs.exceptions.errors import UnsupportedError
from botorch_rs.exceptions.warnings import BadInitialCandidatesWarning, SamplingWarning


def _check(X: Tensor, acq_vals: Tensor, n: int) -> int:
    if X.ndim < 3 or acq_vals.shape != X.shape[:-2]:
        raise UnsupportedError(
            "Restart points and values must have shapes raw x batch_shape x q x d and raw x batch_shape."
        )
    count = X.shape[0]
    if n > count:
        raise RuntimeError(
            f"n ({n}) cannot be larger than the number of provided samples ({count})."
        )
    if not isinstance(n, int) or n < 1 or n > count:
        raise ValueError(
            "The restart count must be positive and no greater than the raw sample count."
        )
    return count


def _random(X: Tensor, values: Tensor, n: int, generator: Any) -> tuple[Tensor, Tensor]:
    warnings.warn(
        "Raw acquisition values are constant, nonpositive or nonfinite; choosing initial conditions at random.",
        BadInitialCandidatesWarning,
        stacklevel=3,
    )
    indices = torch.randperm(X.shape[0], device=X.device, generator=generator)[:n]
    return X[indices], values[indices]


def _ensure_max(indices: Tensor, values: Tensor) -> Tensor:
    maximum = values.argmax(dim=0)
    indices[-1] = torch.where((indices == maximum).any(dim=0), indices[-1], maximum)
    return indices


def _gather(X: Tensor, values: Tensor, indices: Tensor) -> tuple[Tensor, Tensor]:
    return X.gather(
        0, indices[..., None, None].expand(*indices.shape, *X.shape[-2:])
    ), values.gather(0, indices)


def acquisition_batch_shape(acquisition: Any) -> tuple[int, ...]:
    from botorch_rs.utils.transforms import is_ensemble

    model = getattr(acquisition, "model", None)
    shape = tuple(getattr(model, "batch_shape", ()))
    return shape[:-1] if model is not None and is_ensemble(model) else shape


def initialize_q_batch(
    X: Tensor,
    acq_vals: Tensor,
    n: int,
    eta: float = 1.0,
    *,
    generator: Any = None,
) -> tuple[Tensor, Tensor]:
    """Boltzmann restarts, including the maximum, as in the reference scalar case."""
    count = _check(X, acq_vals, n)
    if not math.isfinite(eta):
        raise ValueError("eta must be finite.")
    if n == count:
        return X, acq_vals
    std = acq_vals.std(dim=0)
    if bool(((std == 0) | ~torch.isfinite(std)).any()):
        return _random(X, acq_vals, n, generator)
    standardized = (acq_vals - acq_vals.mean(dim=0)) / std.where(std >= 1e-9, 1.0)
    weights = (eta * standardized).exp()
    while bool(torch.isinf(weights).any()):
        eta *= 0.5
        weights = (eta * standardized).exp()
    indices = (
        torch.multinomial(weights.movedim(0, -1).reshape(-1, count), n, generator=generator)
        .reshape(*acq_vals.shape[1:], n)
        .movedim(-1, 0)
    )
    return _gather(X, acq_vals, _ensure_max(indices, acq_vals))


def initialize_q_batch_nonneg(
    X: Tensor,
    acq_vals: Tensor,
    n: int,
    eta: float = 1.0,
    alpha: float = 1e-4,
    *,
    generator: Any = None,
) -> tuple[Tensor, Tensor]:
    """Positive quota, adaptive alpha threshold and relative-value sampling."""
    count = _check(X, acq_vals, n)
    if acq_vals.ndim > 1:
        batch = acq_vals.shape[1:]
        flat_X, flat_values = X.reshape(count, -1, *X.shape[-2:]), acq_vals.reshape(count, -1)
        selected = [
            initialize_q_batch_nonneg(
                flat_X[:, i], flat_values[:, i], n, eta, alpha, generator=generator
            )
            for i in range(flat_values.shape[-1])
        ]
        return torch.stack([s[0] for s in selected], 1).reshape(
            n, *batch, *X.shape[-2:]
        ), torch.stack([s[1] for s in selected], 1).reshape(n, *batch)
    if not math.isfinite(eta) or not math.isfinite(alpha) or alpha < 0:
        raise ValueError("eta must be finite and alpha must be finite and nonnegative.")
    if n == count:
        return X, acq_vals
    maximum = acq_vals.max()
    if not bool(torch.isfinite(acq_vals).all()) or bool(maximum <= 0):
        return _random(X, acq_vals, n, generator)
    positive = acq_vals > 0
    quota = int(positive.sum().item())
    if quota < n:
        remaining = (~positive).nonzero().flatten()
        chosen = torch.randperm(remaining.numel(), device=X.device, generator=generator)[
            : n - quota
        ]
        positive[remaining[chosen]] = True
        return X[positive], acq_vals[positive]
    eligible = acq_vals >= alpha * maximum
    while int(eligible.sum().item()) < n:
        alpha *= 0.1
        eligible = acq_vals >= alpha * maximum
    pool = eligible.nonzero().flatten()
    weights = (eta * (acq_vals[eligible] / maximum - 1)).exp()
    indices = _ensure_max(pool[torch.multinomial(weights, n, generator=generator)], acq_vals)
    return X[indices], acq_vals[indices]


def initialize_q_batch_topn(
    X: Tensor,
    acq_vals: Tensor,
    n: int,
    largest: bool = True,
    sorted: bool = True,
    *,
    generator: Any = None,
) -> tuple[Tensor, Tensor]:
    """Ordered or unordered top/bottom restarts, with the reference tie fallback."""
    count = _check(X, acq_vals, n)
    if n == count:
        return X, acq_vals
    if bool(acq_vals.std() == 0):
        return _random(X, acq_vals, n, generator)
    _, indices = acq_vals.topk(n, dim=0, largest=largest, sorted=sorted)
    return _gather(X, acq_vals, indices)


def _nonnegative(acquisition: Any) -> bool:
    # Tensor variants inherit the corresponding public acquisition class.
    from botorch_rs.acquisition import analytic, monte_carlo
    from botorch_rs.acquisition.multi_objective import analytic as mo_analytic
    from botorch_rs.acquisition.multi_objective import monte_carlo as mo_mc

    return isinstance(
        acquisition,
        (
            analytic.ExpectedImprovement,
            analytic.ConstrainedExpectedImprovement,
            analytic.ProbabilityOfImprovement,
            analytic.NoisyExpectedImprovement,
            monte_carlo.qExpectedImprovement,
            monte_carlo.qNoisyExpectedImprovement,
            monte_carlo.qProbabilityOfImprovement,
            mo_analytic.ExpectedHypervolumeImprovement,
            mo_mc.qExpectedHypervolumeImprovement,
            mo_mc.qNoisyExpectedHypervolumeImprovement,
        ),
    )


def gen_initial_conditions(
    acquisition: Any,
    bounds: Tensor,
    q: int,
    restarts: int,
    raw_samples: int,
    fixed_features: Mapping[int, Any] | None,
    options: Mapping[str, Any],
    raw_generator: Any = None,
    fixed_X_fantasies: Any = None,
    inequality_constraints: Any = None,
    equality_constraints: Any = None,
) -> Tensor:
    """Generate Sobol candidates and apply the reference scalar restart heuristic."""
    from botorch_rs.optim._torch_acq_optim import _fixed_values, fix_features

    limit = options.get("init_batch_limit")
    if limit is not None and (not isinstance(limit, int) or isinstance(limit, bool) or limit < 1):
        raise ValueError("init_batch_limit must be a positive integer.")
    seed = options.get("seed")
    if q < 1 or raw_samples < restarts or restarts < 1:
        raise ValueError(
            "Initialization requires positive q/restarts and raw_samples >= num_restarts."
        )
    if (
        bounds.ndim != 2
        or bounds.shape[0] != 2
        or not bool(torch.isfinite(bounds).all())
        or bool((bounds[0] > bounds[1]).any())
    ):
        raise ValueError("Initial condition bounds must be finite ordered 2 x d Tensors.")
    if options.get("sample_around_best") and equality_constraints:
        raise UnsupportedError("Sample-around-best is not supported with equality constraints.")
    if options.get("sample_around_best") and raw_generator is not None:
        raise UnsupportedError("Sample-around-best is not supported with a raw generator.")
    for attempt in range(4):
        current_seed = None if seed is None else int(seed) + attempt
        count = raw_samples * (attempt + 1)
        dimension = q * bounds.shape[-1]
        generator = (
            None
            if current_seed is None
            else torch.Generator(device=bounds.device).manual_seed(current_seed)
        )
        constrained = bool(inequality_constraints or equality_constraints)
        if raw_generator is not None:
            samples = raw_generator(count, q, current_seed)
            if not isinstance(samples, Tensor) or samples.shape != (count, q, bounds.shape[-1]):
                raise ValueError("A raw generator must return a raw_samples x q x d Tensor.")
            samples = samples.to(bounds)
            uniform = None
        elif constrained:
            from botorch_rs.optim._torch_initialization_sampling import sample_polytope

            samples = sample_polytope(
                count,
                q,
                bounds,
                n_burnin=options.get("n_burnin", 10000),
                n_thinning=options.get("n_thinning", 32),
                seed=current_seed,
                inequality_constraints=inequality_constraints,
                equality_constraints=equality_constraints,
                fixed_features=fixed_features,
            )
            uniform = None
        elif dimension <= torch.quasirandom.SobolEngine.MAXDIM:
            engine = torch.quasirandom.SobolEngine(dimension, scramble=True, seed=current_seed)
            uniform = engine.draw(count, dtype=bounds.dtype).to(bounds.device).reshape(count, q, -1)
        else:
            warnings.warn(
                "The Sobol dimension limit was exceeded; using device IID initial samples.",
                SamplingWarning,
                stacklevel=3,
            )
            uniform = torch.rand(
                (count, q, bounds.shape[-1]),
                dtype=bounds.dtype,
                device=bounds.device,
                generator=generator,
            )
        if raw_generator is None and not constrained:
            assert uniform is not None
            samples = bounds[0] + (bounds[1] - bounds[0]) * uniform
        if options.get("sample_around_best"):
            from botorch_rs.optim._torch_initialization_sampling import around_best

            extra = around_best(
                acquisition,
                count * q,
                options.get("sample_around_best_sigma", 1e-3),
                bounds,
                subset_sigma=options.get("sample_around_best_subset_sigma", 1e-1),
                prob_perturb=options.get("sample_around_best_prob_perturb"),
                seed=current_seed,
            )
            if extra is not None:
                samples = torch.cat((samples, extra.reshape(count, q, -1)), dim=0)
                count = samples.shape[0]
        batch = acquisition_batch_shape(acquisition)
        if batch:
            samples = samples.reshape(count, *((1,) * len(batch)), q, -1).expand(
                count, *batch, q, bounds.shape[-1]
            )
        samples = fix_features(
            samples, _fixed_values(samples, fixed_features, bounds[0], bounds[1])
        )
        if constrained:
            from botorch_rs.optim._torch_initialization_sampling import feasible_batches

            samples = samples[
                feasible_batches(samples, bounds, inequality_constraints, equality_constraints)
            ]
            count = samples.shape[0]
            if count < restarts:
                raise ValueError("Fewer feasible raw initial conditions than requested restarts.")
        if fixed_X_fantasies is not None:
            fantasies = torch.as_tensor(fixed_X_fantasies, dtype=bounds.dtype, device=bounds.device)
            if fantasies.ndim != 2 or fantasies.shape[-1] != bounds.shape[-1]:
                raise ValueError("fixed_X_fantasies must have shape k x d.")
            samples = torch.cat(
                (samples, fantasies.expand(*samples.shape[:-2], *fantasies.shape)), -2
            )
        with torch.no_grad():
            values = torch.cat(
                [acquisition(chunk) for chunk in samples.split(limit or count)], dim=0
            )
            if values.shape != samples.shape[:-2]:
                raise UnsupportedError(
                    "Automatic initialization requires one scalar per raw and model batch."
                )
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always", BadInitialCandidatesWarning)
                if options.get("topn"):
                    selected, _ = initialize_q_batch_topn(
                        samples,
                        values,
                        restarts,
                        largest=options.get("largest", getattr(acquisition, "maximize", True)),
                        sorted=options.get("sorted", True),
                        generator=generator,
                    )
                elif options.get("nonnegative") or _nonnegative(acquisition):
                    selected, _ = initialize_q_batch_nonneg(
                        samples,
                        values,
                        restarts,
                        eta=options.get("eta", 1.0),
                        alpha=options.get("alpha", 1e-4),
                        generator=generator,
                    )
                else:
                    selected, _ = initialize_q_batch(
                        samples, values, restarts, eta=options.get("eta", 1.0), generator=generator
                    )
            if not any(issubclass(w.category, BadInitialCandidatesWarning) for w in caught):
                return selected
    warnings.warn(
        "Unable to find useful acquisition values; initial conditions are selected randomly.",
        BadInitialCandidatesWarning,
        stacklevel=3,
    )
    return selected

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
    if X.ndim != 3 or acq_vals.shape != X.shape[:1]:
        raise UnsupportedError(
            "Torch restart selection requires raw x q x d points and raw scalar values."
        )
    count = X.shape[0]
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
    maximum = values.argmax()
    if not bool((indices == maximum).any()):
        indices[-1] = maximum
    return indices


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
    std = acq_vals.std()
    if bool((std == 0) | ~torch.isfinite(std)):
        return _random(X, acq_vals, n, generator)
    standardized = (acq_vals - acq_vals.mean()) / std.where(std >= 1e-9, 1.0)
    weights = (eta * standardized).exp()
    while bool(torch.isinf(weights).any()):
        eta *= 0.5
        weights = (eta * standardized).exp()
    indices = _ensure_max(torch.multinomial(weights, n, generator=generator), acq_vals)
    return X[indices], acq_vals[indices]


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
    values, indices = acq_vals.topk(n, largest=largest, sorted=sorted)
    return X[indices], values


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
) -> Tensor:
    """Generate Sobol candidates and apply the reference scalar restart heuristic."""
    from botorch_rs.optim._torch_acq_optim import _fixed_values, fix_features

    limit = options.get("init_batch_limit")
    if limit is not None and (not isinstance(limit, int) or isinstance(limit, bool) or limit < 1):
        raise ValueError("init_batch_limit must be a positive integer.")
    seed = options.get("seed")
    for attempt in range(4):
        current_seed = None if seed is None else int(seed) + attempt
        count = raw_samples * (attempt + 1)
        dimension = q * bounds.shape[-1]
        generator = (
            None
            if current_seed is None
            else torch.Generator(device=bounds.device).manual_seed(current_seed)
        )
        if dimension <= torch.quasirandom.SobolEngine.MAXDIM:
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
        samples = bounds[0] + (bounds[1] - bounds[0]) * uniform
        samples = fix_features(
            samples, _fixed_values(samples, fixed_features, bounds[0], bounds[1])
        )
        with torch.no_grad():
            values = torch.cat(
                [acquisition(chunk) for chunk in samples.split(limit or count)], dim=0
            )
            if values.shape != (count,):
                raise UnsupportedError(
                    "Torch automatic initialization requires one scalar per raw candidate batch."
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

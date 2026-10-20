"""Affine polytope sampling that keeps constraints, chains and points on device."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import Tensor

from botorch_rs.exceptions.errors import InfeasibilityError, UnsupportedError


def transform_constraint(constraint: Any, d: int, q: int | None = None) -> list[Any]:
    indices, coefficients, rhs = constraint
    index = torch.as_tensor(indices)
    if index.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
        raise ValueError("Constraint indices must be integer tensors.")
    if index.ndim == 1:
        if q is None:
            raise ValueError("Inter-point constraints must have shape n x 2.")
        if bool(((index < 0) | (index >= d)).any()):
            raise ValueError("Constraint feature indices are out of range.")
        return [(index + i * d, coefficients, rhs) for i in range(q)]
    if index.ndim != 2 or index.shape[-1] != 2:
        raise ValueError("Inter-point constraints must have shape n x 2.")
    if bool((index < 0).any()) or bool((index[:, 1] >= d).any()):
        raise ValueError("Constraint indices are out of range.")
    if q is not None and bool((index[:, 0] >= q).any()):
        raise ValueError("Constraint point indices are out of range.")
    return [(index[:, 0] * d + index[:, 1], coefficients, rhs)]


def _rows(constraints: Any, bounds: Tensor, q: int) -> tuple[Tensor, Tensor]:
    d = bounds.shape[-1]
    rows, rhs = [], []
    for constraint in constraints or []:
        for index, coefficients, value in transform_constraint(constraint, d, q):
            index = index.to(device=bounds.device, dtype=torch.long)
            coefficients = torch.as_tensor(coefficients, dtype=bounds.dtype, device=bounds.device)
            if coefficients.ndim != 1 or coefficients.numel() != index.numel():
                raise ValueError("Constraint coefficients must match the index vector.")
            row = bounds.new_zeros(q * d).scatter_add(0, index, coefficients)
            rows.append(row)
            rhs.append(torch.as_tensor(value, dtype=bounds.dtype, device=bounds.device).reshape(()))
    return (
        torch.stack(rows) if rows else bounds.new_empty(0, q * d),
        torch.stack(rhs) if rhs else bounds.new_empty(0),
    )


def _project(
    y: Tensor, matrix: Tensor, rhs: Tensor, tolerance: float, cycles: int = 2048
) -> Tensor | None:
    """Dykstra projection onto normalized halfspaces, entirely on the device."""
    if bool((matrix @ y >= rhs - tolerance).all()):
        return y
    corrections = torch.zeros_like(matrix)
    for cycle in range(cycles):
        for i in range(matrix.shape[0]):
            trial = y + corrections[i]
            updated = trial + (rhs[i] - matrix[i] @ trial).clamp_min(0) * matrix[i]
            corrections[i] = trial - updated
            y = updated
        if cycle % 8 == 0 and bool((matrix @ y >= rhs - tolerance).all()):
            return y
    return None


def feasible_batches(samples: Tensor, bounds: Tensor, inequalities: Any, equalities: Any) -> Tensor:
    """One feasibility flag per raw q-batch (including any model batches)."""
    if samples.shape[0] == 0:
        return torch.zeros(0, dtype=torch.bool, device=bounds.device)
    q = samples.shape[-2]
    flattened = samples.reshape(samples.shape[0], -1, q * samples.shape[-1])
    tolerance = 2e-6 if bounds.dtype == torch.float32 else 2e-10
    mask = torch.ones(samples.shape[0], dtype=torch.bool, device=bounds.device)
    for constraints, equality in ((inequalities, False), (equalities, True)):
        matrix, rhs = _rows(constraints, bounds, q)
        residual = flattened @ matrix.mT - rhs
        mask &= ((residual.abs() <= tolerance) if equality else (residual >= -tolerance)).all(
            (-2, -1)
        )
    return mask


def sample_polytope(
    n: int,
    q: int,
    bounds: Tensor,
    n_burnin: int = 10000,
    n_thinning: int = 32,
    seed: int | None = None,
    inequality_constraints: Any = None,
    equality_constraints: Any = None,
    fixed_features: Mapping[int, Any] | None = None,
) -> Tensor:
    """Uniform affine-polytope hit-and-run using independent device chains.

    Each returned q-batch has its own chain, with the requested burn-in and
    thinning transitions. This shares the uniform stationary law of the CPU
    serial chain; it deliberately does not share that chain's RNG sequence.
    """
    if any(
        not isinstance(value, int) or isinstance(value, bool) or value < 1
        for value in (n, q, n_thinning)
    ):
        raise ValueError("n, q and n_thinning must be positive integers.")
    if not isinstance(n_burnin, int) or isinstance(n_burnin, bool) or n_burnin < 0:
        raise ValueError("n_burnin must be a nonnegative integer.")
    if bounds.ndim != 2 or bounds.shape[0] != 2 or not bool(torch.isfinite(bounds).all()):
        raise ValueError("Polytope bounds must be finite 2 x d tensors.")
    if bool((bounds[0] > bounds[1]).any()):
        raise ValueError("Polytope bounds must be ordered.")
    d, dimension = bounds.shape[-1], q * bounds.shape[-1]
    lower, width = bounds[0].repeat(q), (bounds[1] - bounds[0]).repeat(q)
    inequality, irhs = _rows(inequality_constraints, bounds, q)
    equalities = list(equality_constraints or [])
    for index, value in (fixed_features or {}).items():
        if not -d <= index < d:
            raise ValueError("Fixed feature indices are out of range.")
        if value is None:
            continue
        value = torch.as_tensor(value, dtype=bounds.dtype, device=bounds.device)
        if value.numel() != 1:
            raise UnsupportedError(
                "Automatic constrained initialization requires scalar fixed values."
            )
        equalities.append(
            (torch.tensor([index % d], device=bounds.device), bounds.new_ones(1), value.reshape(()))
        )
    equality, erhs = _rows(equalities, bounds, q)
    irhs, inequality = irhs - inequality @ lower, inequality * width
    erhs, equality = erhs - equality @ lower, equality * width
    identity = torch.eye(dimension, dtype=bounds.dtype, device=bounds.device)
    inequality = torch.cat((inequality, identity, -identity))
    irhs = torch.cat((irhs, bounds.new_zeros(dimension), -bounds.new_ones(dimension)))
    if not bool(
        torch.isfinite(inequality).all()
        & torch.isfinite(irhs).all()
        & torch.isfinite(equality).all()
        & torch.isfinite(erhs).all()
    ):
        raise ValueError("Constraint coefficients and right hand sides must be finite.")
    tolerance = 2e-6 if bounds.dtype == torch.float32 else 2e-10
    if equality.shape[0]:
        _, singular, vh = torch.linalg.svd(equality, full_matrices=True)
        threshold = singular.amax() * max(equality.shape) * torch.finfo(bounds.dtype).eps
        rank = int((singular > threshold).sum())
        particular = torch.linalg.pinv(equality) @ erhs
        if bool((equality @ particular - erhs).abs().amax() > tolerance):
            raise InfeasibilityError("The equality constraints are inconsistent.")
        nullspace = vh[rank:].mT
    else:
        particular, nullspace = bounds.new_zeros(dimension), identity
    matrix, rhs = inequality @ nullspace, irhs - inequality @ particular
    norms = matrix.norm(dim=-1)
    zero = norms <= torch.finfo(bounds.dtype).eps * max(dimension, 1)
    if bool((rhs[zero] > tolerance).any()):
        raise InfeasibilityError("The affine constraints have no feasible point.")
    matrix, rhs, norms = matrix[~zero], rhs[~zero], norms[~zero]
    if nullspace.shape[-1] == 0:
        return (lower + width * particular).expand(n, dimension).reshape(n, q, d).clone()
    matrix, rhs = matrix / norms.unsqueeze(-1), rhs / norms
    center = nullspace.mT @ (bounds.new_full((dimension,), 0.5) - particular)
    feasible = _project(center, matrix, rhs, tolerance)
    if feasible is None:
        raise InfeasibilityError("Device projection could not find a feasible polytope point.")
    # Start in the relative interior where possible: a corner can trap a
    # high-dimensional line sampler. Explicit equalities already remove the
    # affine hull, so a strictly feasible point ordinarily exists here.
    if bool((matrix @ feasible - rhs).amin() <= tolerance):
        interior = None
        for margin in (1e-5, 1e-7, 1e-9):
            trial = _project(feasible.clone(), matrix, rhs + margin, tolerance, cycles=128)
            if trial is not None and bool((matrix @ trial > rhs).all()):
                interior = trial
                break
        if interior is not None:
            feasible = interior
        else:
            raise InfeasibilityError(
                "Represent the polytope's affine hull with equality constraints."
            )
    generator = None if seed is None else torch.Generator(device=bounds.device).manual_seed(seed)
    state = feasible.expand(n, -1).clone()
    interpoint = any(
        torch.as_tensor(indices).ndim == 2
        for indices, _, _ in (inequality_constraints or []) + (equality_constraints or [])
    )
    transitions = n_burnin + n_thinning * (q if interpoint else 1)
    for _ in range(transitions):
        direction = torch.randn(
            state.shape, dtype=bounds.dtype, device=bounds.device, generator=generator
        )
        projected = direction @ matrix.mT
        distances = rhs - state @ matrix.mT
        ratio = distances / torch.where(projected != 0, projected, torch.ones_like(projected))
        left = torch.where(projected > 0, ratio, -torch.inf).amax(-1)
        right = torch.where(projected < 0, ratio, torch.inf).amin(-1)
        uniform = torch.rand((n,), dtype=bounds.dtype, device=bounds.device, generator=generator)
        step = left + (right - left).clamp_min(0) * uniform
        state = state + step.unsqueeze(-1) * direction
    normalized = particular + state @ nullspace.mT
    return (lower + width * normalized).reshape(n, q, d)


def truncated_perturbations(
    X: Tensor,
    n: int,
    sigma: float,
    bounds: Tensor,
    qmc: bool = True,
    generator: Any = None,
    *,
    centers_selected: bool = False,
) -> Tensor:
    """Inverse-CDF Gaussian perturbations truncated to the physical box."""
    if X.ndim != 2 or bounds.shape != (2, X.shape[-1]) or X.shape[0] == 0:
        raise ValueError("Perturbations require nonempty n x d centers and 2 x d bounds.")
    if n < 0 or sigma <= 0:
        raise ValueError("The sample count must be nonnegative and sigma positive.")
    if n == 0:
        return X.new_empty(0, X.shape[-1])
    width = bounds[1] - bounds[0]
    safe_width = torch.where(width == 0, torch.ones_like(width), width)
    normalized = (X - bounds[0]) / safe_width
    if not centers_selected:
        normalized = (
            normalized[torch.randint(X.shape[0], (n,), device=X.device, generator=generator)]
            if X.shape[0] > 1
            else normalized.expand(n, -1)
        )
    if normalized.shape[0] != n:
        raise ValueError("The selected center count must match the sample count.")
    if qmc:
        seed = (
            None
            if generator is None
            else int(torch.randint(2**31 - 1, (), device=X.device, generator=generator))
        )
        uniform = (
            torch.quasirandom.SobolEngine(X.shape[-1], scramble=True, seed=seed)
            .draw(n, dtype=X.dtype)
            .to(X.device)
        )
    else:
        uniform = torch.rand((n, X.shape[-1]), dtype=X.dtype, device=X.device, generator=generator)
    import math

    alpha, beta = -normalized / sigma, (1 - normalized) / sigma
    lower_cdf = 0.5 * (1 + torch.erf(alpha / math.sqrt(2)))
    upper_cdf = 0.5 * (1 + torch.erf(beta / math.sqrt(2)))
    probability = lower_cdf + uniform * (upper_cdf - lower_cdf)
    perturbation = torch.erfinv(2 * probability - 1) * math.sqrt(2) * sigma
    return bounds[0] + width * (normalized + perturbation).clamp(0, 1)


def subset_perturbations(
    X: Tensor,
    bounds: Tensor,
    n: int,
    sigma: float = 0.1,
    qmc: bool = True,
    prob_perturb: float | None = None,
    generator: Any = None,
) -> Tensor:
    """Perturb selected dimensions around the same selected center."""
    import math

    d = X.shape[-1]
    probability = min(20 / d, 1.0) if prob_perturb is None else prob_perturb
    if not 0 <= probability <= 1:
        raise ValueError("prob_perturb must lie in [0, 1].")
    centers = (
        X[torch.randint(X.shape[0], (n,), device=X.device, generator=generator)]
        if X.shape[0] > 1
        else X.expand(n, -1)
    )
    perturbed = truncated_perturbations(
        centers, n, sigma, bounds, qmc, generator, centers_selected=True
    )
    mask = torch.rand((n, d), dtype=X.dtype, device=X.device, generator=generator) <= probability
    empty = ~mask.any(-1)
    number = math.ceil(d * probability)
    # Batched random permutations avoid one CUDA synchronization per row.
    ranks = (
        torch.rand((n, d), dtype=X.dtype, device=X.device, generator=generator)
        .argsort(-1)
        .argsort(-1)
    )
    mask = torch.where(empty.unsqueeze(-1), ranks < number, mask)
    return torch.where(mask, perturbed, centers)


def around_best(
    acquisition: Any,
    n: int,
    sigma: float,
    bounds: Tensor,
    best_pct: float = 5.0,
    subset_sigma: float = 0.1,
    prob_perturb: float | None = None,
    seed: int | None = None,
) -> Tensor | None:
    """Posterior-mean incumbent/Pareto selection followed by box perturbations."""
    import warnings
    from collections.abc import Mapping

    from botorch_rs.exceptions.warnings import BotorchWarning
    from botorch_rs.optim.utils.acquisition_utils import get_X_baseline

    X = get_X_baseline(acquisition)
    if X is None:
        return None
    with torch.no_grad():
        try:
            mean = acquisition.model.posterior(X).mean
        except AttributeError:
            warnings.warn(
                "Failed to sample around previous best points.", BotorchWarning, stacklevel=3
            )
            return None
        while mean.ndim > 2:
            mean = mean.mean(0)
        objective = getattr(acquisition, "objective", None)
        predicted = mean if objective is None else objective(mean)
        if getattr(acquisition, "maximize", True) is False:
            predicted = -predicted
        constraints = getattr(acquisition, "constraints", None)
        if constraints:
            if isinstance(constraints, Mapping):
                terms = [mean.new_zeros(mean.shape[:-1])]
                for index, (lower, upper) in constraints.items():
                    if lower is not None:
                        terms.append((lower - mean[..., index]).clamp_min(0))
                    if upper is not None:
                        terms.append((mean[..., index] - upper).clamp_min(0))
                violation = -torch.stack(terms, -1).sum(-1)
            else:
                violation = -torch.stack(
                    [constraint(mean).clamp_min(0) for constraint in constraints], -1
                ).sum(-1)
            feasible = violation == 0
            if bool(feasible.any()):
                predicted = predicted.masked_fill(
                    ~feasible.unsqueeze(-1) if predicted.ndim == 2 else ~feasible, -torch.inf
                )
            else:
                predicted = violation
        if predicted.ndim == 2 and predicted.shape[-1] > 1:
            dominated = (
                (predicted[None, :, :] >= predicted[:, None, :]).all(-1)
                & (predicted[None, :, :] > predicted[:, None, :]).any(-1)
            ).any(-1)
            duplicates = (predicted[:, None, :] == predicted[None, :, :]).all(-1).tril(-1).any(-1)
            best = X[~dominated & ~duplicates]
        else:
            count = min(X.shape[0], max(1, round(X.shape[0] * best_pct / 100)))
            best = X[predicted.reshape(-1).topk(count).indices]
        generator = (
            None if seed is None else torch.Generator(device=bounds.device).manual_seed(seed)
        )
        subset = bounds.shape[-1] >= 20 or prob_perturb is not None
        truncated_count = n // 2 if subset else n
        points = truncated_perturbations(best, truncated_count, sigma, bounds, generator=generator)
        if subset:
            # Keep the port's documented subset_sigma correction: upstream ignores it.
            other = subset_perturbations(
                best,
                bounds,
                n - truncated_count,
                subset_sigma,
                prob_perturb=prob_perturb,
                generator=generator,
            )
            points = torch.cat((points, other))
            points = points[torch.randperm(n, device=bounds.device, generator=generator)]
        return points

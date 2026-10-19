"""Monte-Carlo estimates of the feasible volume
(``botorch.utils.feasible_volume``).

The feature samples are uniform draws in the box (``torch.rand`` after
``torch.manual_seed(seed)``, BoTorch's draws, reproduced natively when the
bounds are not torch tensors); the outcome samples come from the model's
posterior with ``get_sampler`` (botorch_rs's Sobol sampler, whose points
differ from torch's ``SobolEngine``: the outcome probabilities agree with
BoTorch's statistically, not bitwise).
"""

from __future__ import annotations

import logging
import math
import random
import sys
from collections.abc import Callable
from typing import Any

from botorch_rs import _botorch_rs as _rs
from botorch_rs import _ops
from botorch_rs._arrays import from_flat, is_torch_tensor, kind_of, to_flat

logger = logging.getLogger("Feasibility")


def get_feasible_samples(
    samples: Any,
    inequality_constraints: list[tuple[Any, Any, float]] | None = None,
) -> tuple[Any, float]:
    r"""
    Checks which of the samples satisfy all of the inequality constraints.

    Args:
        samples: A ``sample size x d`` size tensor of feature samples,
            where d is a feature dimension.
        inequality constraints: A list of tuples (indices, coefficients, rhs),
            with each tuple encoding an inequality constraint of the form
            ``\sum_i (X[indices[i]] * coefficients[i]) >= rhs``.
    Returns:
        2-element tuple containing

        - Samples satisfying the linear constraints.
        - Estimated proportion of samples satisfying the linear constraints.
    """
    if inequality_constraints is None:
        return samples, 1.0
    data, shape = to_flat(samples, "samples")
    n, d = shape
    feasible = [True] * n
    for indices, coefficients, rhs in inequality_constraints:
        idx = _ops.ints(indices)
        coef = [float(v) for v in to_flat(coefficients, "coefficients")[0]]
        for r in range(n):
            lhs = 0.0
            for i, c in zip(idx, coef, strict=True):
                lhs += float(data[r * d + i]) * c
            feasible[r] = feasible[r] and lhs >= float(rhs)
    if is_torch_tensor(samples):
        torch = sys.modules["torch"]
        feasible_samples = samples[torch.tensor(feasible, dtype=torch.bool, device=samples.device)]
    else:
        keep = [r for r in range(n) if feasible[r]]
        feasible_samples = from_flat(
            [float(data[r * d + k]) for r in keep for k in range(d)],
            (len(keep), d),
            kind_of(samples),
        )
    p_linear = sum(feasible) / n
    return feasible_samples, p_linear


def get_outcome_feasibility_probability(
    model: Any,
    X: Any,
    outcome_constraints: list[Callable[[Any], Any]],
    threshold: float = 0.1,
    nsample_outcome: int = 1000,
    seed: int | None = None,
) -> float:
    r"""
    Monte Carlo estimate of the feasible volume with respect to the outcome constraints.

    Args:
        model: The model used for sampling the posterior.
        X: A tensor of dimension ``batch-shape x 1 x d``, where d is feature dimension.
        outcome_constraints: A list of callables, each mapping a Tensor of dimension
            ``sample_shape x batch-shape x q x m`` to a Tensor of dimension
            ``sample_shape x batch-shape x q``, where negative values imply feasibility.
        threshold: A lower limit for the probability of posterior samples feasibility.
        nsample_outcome: The number of samples from the model posterior.
        seed: The seed for the posterior sampler. If omitted, use a random seed.

    Returns:
        Estimated proportion of features for which posterior samples satisfy
        given outcome constraints with probability above or equal to
        the given threshold.
    """
    if outcome_constraints is None:
        return 1.0

    from botorch_rs.sampling.get_sampler import get_sampler

    if seed is None:
        torch = sys.modules.get("torch")
        seed = (
            int(torch.randint(0, 1000000, (1,)).item())
            if torch is not None
            else random.randint(0, 999999)
        )

    posterior = model.posterior(X)  # posterior consists of batch_shape marginals
    sampler: Any = get_sampler(posterior=posterior, sample_shape=(nsample_outcome,), seed=seed)
    # size of samples: (num outcome samples, batch_shape, 1, outcome dim)
    samples = sampler(posterior)
    sshape = tuple(_ops.shape_of(samples))[:-1]
    n_feas = math.prod(sshape)
    feasible = [True] * n_feas
    # a sample passes if each constraint applied to the sample
    # produces a non-negative tensor
    for oc in outcome_constraints:
        vals, _ = to_flat(oc(samples), "constraint")
        feasible = [f and float(v) <= 0 for f, v in zip(feasible, vals, strict=True)]
    # proportion of feasibile samples for each of the elements of X
    S = sshape[0]
    rest = n_feas // S
    p_feas = [sum(feasible[s * rest + j] for s in range(S)) / S for j in range(rest)]
    nX = _ops.shape_of(X)[0]
    # proportion of features leading to the posterior outcome satisfying the
    # given outcome constraints with at probability above a given threshold
    return sum(1 for p in p_feas if _f32(p) >= threshold) / nX


def _f32(v: float) -> float:
    import struct

    return float(struct.unpack("f", struct.pack("f", v))[0])


def estimate_feasible_volume(
    bounds: Any,
    model: Any,
    outcome_constraints: list[Callable[[Any], Any]],
    inequality_constraints: list[tuple[Any, Any, float]] | None = None,
    nsample_feature: int = 1000,
    nsample_outcome: int = 1000,
    threshold: float = 0.1,
    verbose: bool = False,
    seed: int | None = None,
    device: Any = None,
    dtype: Any = None,
) -> tuple[float, float]:
    r"""
    Monte Carlo estimate of the feasible volume with respect
    to feature constraints and outcome constraints.

    Args:
        bounds: A ``2 x d`` tensor of lower and upper bounds
            for each column of ``X``.
        model: The model used for sampling the outcomes.
        outcome_constraints: A list of callables, each mapping a Tensor of dimension
            ``sample_shape x batch-shape x q x m`` to a Tensor of dimension
            ``sample_shape x batch-shape x q``, where negative values imply
            feasibility.
        inequality constraints: A list of tuples (indices, coefficients, rhs),
            with each tuple encoding an inequality constraint of the form
            ``\sum_i (X[indices[i]] * coefficients[i]) >= rhs``.
        nsample_feature: The number of feature samples satisfying the bounds.
        nsample_outcome: The number of outcome samples from the model posterior.
        threshold: A lower limit for the probability of outcome feasibility
        seed: The seed for both feature and outcome samplers. If omitted,
            use a random seed.
        verbose: An indicator for whether to log the results.

    Returns:
        2-element tuple containing:

        - Estimated proportion of volume in feature space that is
            feasible wrt the bounds and the inequality constraints (linear).
        - Estimated proportion of feasible features for which
            posterior samples (outcome) satisfies the outcome constraints
            with probability above the given threshold.
    """
    torch = sys.modules.get("torch")
    if seed is None:
        seed = (
            int(torch.randint(0, 1000000, (1,)).item())
            if torch is not None
            else random.randint(0, 999999)
        )
    bdata, bshape = to_flat(bounds, "bounds")
    d = bshape[1]
    kind = kind_of(bounds)
    if torch is not None:
        from botorch_rs.utils.sampling import manual_seed

        with manual_seed(seed=seed):
            u = torch.rand((nsample_feature, d), dtype=dtype, device=device).reshape(-1).tolist()
    else:
        # BoTorch's torch.rand((n, d), dtype=dtype) after torch.manual_seed(seed)
        # (float32 for dtype=None), reproduced natively.
        single = dtype is None or str(dtype) in ("float32", "torch.float32")
        u = list(_rs.torch_rand(int(seed), nsample_feature * d, single)[0])
    lo, hi = [float(v) for v in bdata[:d]], [float(v) for v in bdata[d:]]
    box = [u[r * d + k] * (hi[k] - lo[k]) + lo[k] for r in range(nsample_feature) for k in range(d)]
    box_samples = from_flat(box, (nsample_feature, d), kind)

    features, p_feature = get_feasible_samples(
        samples=box_samples, inequality_constraints=inequality_constraints
    )  # each new feature sample is a row

    fshape = _ops.shape_of(features)
    X = (
        features.unsqueeze(-2)
        if is_torch_tensor(features)
        else from_flat(to_flat(features)[0], (fshape[0], 1, fshape[1]), kind_of(features))
    )
    p_outcome = get_outcome_feasibility_probability(
        model=model,
        X=X,
        outcome_constraints=outcome_constraints,
        threshold=threshold,
        nsample_outcome=nsample_outcome,
        seed=seed,
    )

    if verbose:  # pragma: no cover
        logger.info("Proportion of volume that satisfies linear constraints: " + f"{p_feature:.4e}")
        if p_feature <= 0.01:
            logger.warning(
                "The proportion of satisfying volume is very low and may lead to "
                + "very long run times. Consider making your constraints less "
                + "restrictive."
            )
        logger.info(
            "Proportion of linear-feasible volume that also satisfies each "
            + f"outcome constraint with probability > 0.1: {p_outcome:.4e}"
        )
        if p_outcome <= 0.001:
            logger.warning(
                "The proportion of volume that also satisfies the outcome constraint "
                + "is very low. Consider making your parameter and outcome constraints "
                + "less restrictive."
            )
    return p_feature, p_outcome


__all__ = [
    "estimate_feasible_volume",
    "get_feasible_samples",
    "get_outcome_feasibility_probability",
]

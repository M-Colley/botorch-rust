"""Acquisition utilities (``botorch.acquisition.utils``)."""

from __future__ import annotations

from typing import Any

from botorch_rs import _botorch_rs as _rs
from botorch_rs._arrays import from_pair, kind_of, to_flat
from botorch_rs.acquisition.acquisition import check_unsupported
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

    If more than ``ceil(max_frac * n)`` points qualify, the most frequent
    maximizers are kept (BoTorch's implementation selects wrong points in that
    case; with the default ``max_frac=1`` both agree). Multi-output models
    require a scalarizing ``posterior_transform``.
    """
    check_unsupported(
        objective=objective,
        constraints=constraints,
        marginalize_dim=marginalize_dim,
    )
    if not isinstance(model, (SingleTaskGP, ModelListGP)):
        raise UnsupportedError(
            "prune_inferior_points requires a botorch_rs SingleTaskGP or ModelListGP."
        )
    if model.num_outputs > 1 and posterior_transform is None:
        raise UnsupportedError(
            "Models with multiple outputs require an objective or posterior_transform."
        )
    flat = to_flat(X, "X")
    n = flat[1][0] if flat[1] else 0
    base_samples = None
    if sampler is not None:
        if not isinstance(sampler, SobolQMCNormalSampler):
            raise UnsupportedError("`sampler` must be a botorch_rs SobolQMCNormalSampler.")
        num_samples = sampler.num_samples
        base_samples = sampler._rs.base_samples(n) if n > 0 else None
    pruned = _rs.prune_inferior_points(
        model._acquisition_model(posterior_transform),
        flat,
        num_samples,
        float(max_frac),
        None,
        base_samples,
    )
    return from_pair(pruned, kind_of(X))


__all__ = ["prune_inferior_points"]

"""Probability utilities (``botorch.utils.probability``): bivariate normal
probabilities and moments, the bivariate conditioning algorithm ``MVNXPB``,
the linear elliptical slice sampler, the truncated multivariate normal and
unified skew normal distributions, and standard-normal helpers.

``TruncatedMultivariateNormal`` and ``UnifiedSkewNormal`` are torch
distributions (created on first access, which imports torch).
"""

from __future__ import annotations

from typing import Any

from botorch_rs.utils.probability.bvn import bvn, bvnmom
from botorch_rs.utils.probability.lin_ess import LinearEllipticalSliceSampler
from botorch_rs.utils.probability.mvnxpb import MVNXPB
from botorch_rs.utils.probability.utils import ndtr

__all__ = [
    "bvn",
    "bvnmom",
    "LinearEllipticalSliceSampler",
    "MVNXPB",
    "ndtr",
    "TruncatedMultivariateNormal",
    "UnifiedSkewNormal",
]


def __getattr__(name: str) -> Any:
    if name == "TruncatedMultivariateNormal":
        from botorch_rs.utils.probability import truncated_multivariate_normal as m
    elif name == "UnifiedSkewNormal":
        from botorch_rs.utils.probability import unified_skew_normal as m  # type: ignore[no-redef]
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(m, name)
    globals()[name] = value
    return value

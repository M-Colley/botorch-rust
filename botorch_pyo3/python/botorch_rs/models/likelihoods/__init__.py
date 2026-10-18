"""Likelihoods of BoTorch's models (``botorch.models.likelihoods``)."""

from botorch_rs.models.likelihoods.pairwise import (
    PairwiseLogitLikelihood,
    PairwiseProbitLikelihood,
)

__all__ = [
    "PairwiseProbitLikelihood",
    "PairwiseLogitLikelihood",
]

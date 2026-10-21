"""Likelihoods of BoTorch's models (``botorch.models.likelihoods``)."""

from botorch_rs.models.likelihoods.pairwise import (
    PairwiseLogitLikelihood,
    PairwiseProbitLikelihood,
)
from botorch_rs.models.likelihoods.sparse_outlier_noise import (
    SparseOutlierGaussianLikelihood,
    SparseOutlierNoise,
)

__all__ = [
    "SparseOutlierGaussianLikelihood",
    "SparseOutlierNoise",
    "PairwiseProbitLikelihood",
    "PairwiseLogitLikelihood",
]

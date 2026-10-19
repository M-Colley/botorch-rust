"""Models (``botorch.models``)."""

from botorch_rs.models.approximate_gp import ApproximateGPyTorchModel, SingleTaskVariationalGP
from botorch_rs.models.contextual import SACGP
from botorch_rs.models.contextual_multioutput import LCEMGP, ProvidedEmbeddingLCEMGP
from botorch_rs.models.cost import AffineFidelityCostModel
from botorch_rs.models.deterministic import (
    AffineDeterministicModel,
    GenericDeterministicModel,
    PosteriorMeanModel,
)
from botorch_rs.models.gp_regression import SingleTaskGP
from botorch_rs.models.gp_regression_fidelity import SingleTaskMultiFidelityGP
from botorch_rs.models.gp_regression_mixed import MixedSingleTaskGP
from botorch_rs.models.gpytorch import GPyTorchModel
from botorch_rs.models.higher_order_gp import HigherOrderGP
from botorch_rs.models.map_saas import (
    AdditiveMapSaasSingleTaskGP,
    EnsembleMapSaasSingleTaskGP,
    add_saas_prior,
)
from botorch_rs.models.model import Model, ModelList
from botorch_rs.models.model_list_gp_regression import ModelListGP
from botorch_rs.models.multitask import KroneckerMultiTaskGP, MultiTaskGP
from botorch_rs.models.pairwise_gp import PairwiseGP, PairwiseLaplaceMarginalLogLikelihood

__all__ = [
    "LCEMGP",
    "AdditiveMapSaasSingleTaskGP",
    "AffineDeterministicModel",
    "AffineFidelityCostModel",
    "ApproximateGPyTorchModel",
    "EnsembleMapSaasSingleTaskGP",
    "GenericDeterministicModel",
    "GPyTorchModel",
    "HigherOrderGP",
    "KroneckerMultiTaskGP",
    "MixedSingleTaskGP",
    "Model",
    "ModelList",
    "ModelListGP",
    "MultiTaskGP",
    "PairwiseGP",
    "PairwiseLaplaceMarginalLogLikelihood",
    "PosteriorMeanModel",
    "ProvidedEmbeddingLCEMGP",
    "SACGP",
    "SingleTaskGP",
    "SingleTaskMultiFidelityGP",
    "SingleTaskVariationalGP",
    "add_saas_prior",
]

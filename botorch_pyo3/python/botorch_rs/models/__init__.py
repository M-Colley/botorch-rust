"""Models (``botorch.models``)."""

from botorch_rs.models.approximate_gp import ApproximateGPyTorchModel, SingleTaskVariationalGP
from botorch_rs.models.contextual import LCEAGP, SACGP
from botorch_rs.models.contextual_multioutput import LCEMGP, ProvidedEmbeddingLCEMGP
from botorch_rs.models.cost import AffineFidelityCostModel
from botorch_rs.models.deterministic import (
    AffineDeterministicModel,
    GenericDeterministicModel,
    PosteriorMeanModel,
)
from botorch_rs.models.empirical_gps.empirical_1d_gp import EmpiricalOneDimensionalGP
from botorch_rs.models.empirical_gps.multioutput_empirical_1d_gp import (
    MultiOutputEmpiricalOneDimensionalGP,
)
from botorch_rs.models.gp_regression import SingleTaskGP
from botorch_rs.models.gp_regression_fidelity import SingleTaskMultiFidelityGP
from botorch_rs.models.gp_regression_mixed import MixedSingleTaskGP
from botorch_rs.models.gpytorch import (
    BatchedMultiOutputGPyTorchModel,
    GPyTorchModel,
    ModelListGPyTorchModel,
    MultiTaskGPyTorchModel,
)
from botorch_rs.models.heterogeneous_mtgp import HeterogeneousMTGP
from botorch_rs.models.higher_order_gp import HigherOrderGP
from botorch_rs.models.latent_kronecker_gp import LatentKroneckerGP
from botorch_rs.models.map_saas import (
    AdditiveMapSaasSingleTaskGP,
    EnsembleMapSaasSingleTaskGP,
    add_saas_prior,
)
from botorch_rs.models.model import Model, ModelList
from botorch_rs.models.model_list_gp_regression import ModelListGP
from botorch_rs.models.multitask import KroneckerMultiTaskGP, MultiTaskGP
from botorch_rs.models.pairwise_gp import PairwiseGP, PairwiseLaplaceMarginalLogLikelihood
from botorch_rs.models.robust_relevance_pursuit_model import RobustRelevancePursuitSingleTaskGP

__all__ = [
    "RobustRelevancePursuitSingleTaskGP",
    "LCEMGP",
    "AdditiveMapSaasSingleTaskGP",
    "AffineDeterministicModel",
    "AffineFidelityCostModel",
    "ApproximateGPyTorchModel",
    "EnsembleMapSaasSingleTaskGP",
    "EmpiricalOneDimensionalGP",
    "MultiOutputEmpiricalOneDimensionalGP",
    "GenericDeterministicModel",
    "GPyTorchModel",
    "BatchedMultiOutputGPyTorchModel",
    "ModelListGPyTorchModel",
    "MultiTaskGPyTorchModel",
    "HigherOrderGP",
    "HeterogeneousMTGP",
    "LatentKroneckerGP",
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
    "LCEAGP",
    "SingleTaskGP",
    "SingleTaskMultiFidelityGP",
    "SingleTaskVariationalGP",
    "add_saas_prior",
]

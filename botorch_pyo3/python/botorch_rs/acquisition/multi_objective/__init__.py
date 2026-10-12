"""Multi-objective acquisition functions (``botorch.acquisition.multi_objective``):
qLogNEHVI (BoTorch's recommended default), qLogEHVI, qNEHVI, qEHVI, the
analytic EHVI, qLogNParEGO, MOMF and the hypervolume knowledge gradient,
multi-output MC objectives (including the multi-output risk measures of
``multi_output_risk_measures``) and the multi-objective utilities, computed
in Rust with exact gradients."""

from botorch_rs.acquisition.multi_objective.analytic import ExpectedHypervolumeImprovement
from botorch_rs.acquisition.multi_objective.base import (
    MultiObjectiveAnalyticAcquisitionFunction,
    MultiObjectiveMCAcquisitionFunction,
)
from botorch_rs.acquisition.multi_objective.hypervolume_knowledge_gradient import (
    qHypervolumeKnowledgeGradient,
    qMultiFidelityHypervolumeKnowledgeGradient,
)
from botorch_rs.acquisition.multi_objective.logei import (
    qLogExpectedHypervolumeImprovement,
    qLogNoisyExpectedHypervolumeImprovement,
)
from botorch_rs.acquisition.multi_objective.monte_carlo import (
    qExpectedHypervolumeImprovement,
    qNoisyExpectedHypervolumeImprovement,
)
from botorch_rs.acquisition.multi_objective.multi_fidelity import MOMF
from botorch_rs.acquisition.multi_objective.objective import (
    FeasibilityWeightedMCMultiOutputObjective,
    GenericMCMultiOutputObjective,
    IdentityMCMultiOutputObjective,
    MCMultiOutputObjective,
    WeightedMCMultiOutputObjective,
)
from botorch_rs.acquisition.multi_objective.parego import qLogNParEGO
from botorch_rs.acquisition.multi_objective.utils import (
    get_default_partitioning_alpha,
    prune_inferior_points_multi_objective,
)

__all__ = [
    "MOMF",
    "ExpectedHypervolumeImprovement",
    "FeasibilityWeightedMCMultiOutputObjective",
    "GenericMCMultiOutputObjective",
    "IdentityMCMultiOutputObjective",
    "MCMultiOutputObjective",
    "MultiObjectiveAnalyticAcquisitionFunction",
    "MultiObjectiveMCAcquisitionFunction",
    "WeightedMCMultiOutputObjective",
    "get_default_partitioning_alpha",
    "prune_inferior_points_multi_objective",
    "qExpectedHypervolumeImprovement",
    "qHypervolumeKnowledgeGradient",
    "qLogExpectedHypervolumeImprovement",
    "qLogNParEGO",
    "qLogNoisyExpectedHypervolumeImprovement",
    "qMultiFidelityHypervolumeKnowledgeGradient",
    "qNoisyExpectedHypervolumeImprovement",
]

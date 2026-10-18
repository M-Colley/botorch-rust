"""Acquisition functions (``botorch.acquisition``)."""

from botorch_rs.acquisition.acquisition import AcquisitionFunction, OneShotAcquisitionFunction
from botorch_rs.acquisition.active_learning import qNegIntegratedPosteriorVariance
from botorch_rs.acquisition.analytic import (
    AnalyticAcquisitionFunction,
    ConstrainedExpectedImprovement,
    ExpectedImprovement,
    LogConstrainedExpectedImprovement,
    LogExpectedImprovement,
    LogNoisyExpectedImprovement,
    LogProbabilityOfFeasibility,
    LogProbabilityOfImprovement,
    NoisyExpectedImprovement,
    PosteriorMean,
    PosteriorStandardDeviation,
    ProbabilityOfImprovement,
    ScalarizedPosteriorMean,
    UpperConfidenceBound,
    qAnalyticProbabilityOfImprovement,
)
from botorch_rs.acquisition.composite import CallableAcquisitionFunction, WeightedLogMeanExp
from botorch_rs.acquisition.cost_aware import (
    CostAwareUtility,
    GenericCostAwareUtility,
    InverseCostWeightedUtility,
)
from botorch_rs.acquisition.decoupled import DecoupledAcquisitionFunction
from botorch_rs.acquisition.fixed_feature import FixedFeatureAcquisitionFunction
from botorch_rs.acquisition.joint_entropy_search import qJointEntropySearch
from botorch_rs.acquisition.knowledge_gradient import (
    ProjectedAcquisitionFunction,
    qKnowledgeGradient,
    qMultiFidelityKnowledgeGradient,
)
from botorch_rs.acquisition.logei import (
    LogImprovementMCAcquisitionFunction,
    qLogExpectedImprovement,
    qLogNoisyExpectedImprovement,
    qLogProbabilityOfFeasibility,
)
from botorch_rs.acquisition.max_value_entropy_search import (
    MaxValueBase,
    qLowerBoundMaxValueEntropy,
    qMaxValueEntropy,
    qMultiFidelityLowerBoundMaxValueEntropy,
    qMultiFidelityMaxValueEntropy,
)
from botorch_rs.acquisition.monte_carlo import (
    MCAcquisitionFunction,
    SampleReducingMCAcquisitionFunction,
    qExpectedImprovement,
    qLowerConfidenceBound,
    qNoisyExpectedImprovement,
    qPosteriorStandardDeviation,
    qProbabilityOfImprovement,
    qSimpleRegret,
    qUpperConfidenceBound,
)
from botorch_rs.acquisition.multi_step_lookahead import qMultiStepLookahead
from botorch_rs.acquisition.multioutput_acquisition import MultiOutputAcquisitionFunction
from botorch_rs.acquisition.objective import (
    ConstrainedMCObjective,
    GenericMCObjective,
    IdentityMCObjective,
    LinearMCObjective,
    MCAcquisitionObjective,
    PosteriorTransform,
    ScalarizedPosteriorTransform,
)
from botorch_rs.acquisition.penalized import (
    GaussianPenalty,
    GroupLassoPenalty,
    L0Approximation,
    L0PenaltyApprox,
    L0PenaltyApproxObjective,
    L1Penalty,
    L1PenaltyObjective,
    L2Penalty,
    PenalizedAcquisitionFunction,
    PenalizedMCObjective,
)
from botorch_rs.acquisition.predictive_entropy_search import qPredictiveEntropySearch
from botorch_rs.acquisition.prior_guided import PriorGuidedAcquisitionFunction
from botorch_rs.acquisition.proximal import ProximalAcquisitionFunction
from botorch_rs.acquisition.thompson_sampling import PathwiseThompsonSampling
from botorch_rs.acquisition.utils import get_optimal_samples, prune_inferior_points

# The factory and the input constructors import the acquisition functions above.
from botorch_rs.acquisition.factory import get_acquisition_function  # noqa: E402
from botorch_rs.acquisition.input_constructors import get_acqf_input_constructor  # noqa: E402

__all__ = [
    "AcquisitionFunction",
    "AnalyticAcquisitionFunction",
    "CallableAcquisitionFunction",
    "ConstrainedExpectedImprovement",
    "ConstrainedMCObjective",
    "CostAwareUtility",
    "DecoupledAcquisitionFunction",
    "ExpectedImprovement",
    "FixedFeatureAcquisitionFunction",
    "GaussianPenalty",
    "GenericCostAwareUtility",
    "GenericMCObjective",
    "GroupLassoPenalty",
    "IdentityMCObjective",
    "InverseCostWeightedUtility",
    "L0Approximation",
    "L0PenaltyApprox",
    "L0PenaltyApproxObjective",
    "L1Penalty",
    "L1PenaltyObjective",
    "L2Penalty",
    "LinearMCObjective",
    "LogImprovementMCAcquisitionFunction",
    "LogConstrainedExpectedImprovement",
    "LogExpectedImprovement",
    "LogNoisyExpectedImprovement",
    "LogProbabilityOfFeasibility",
    "LogProbabilityOfImprovement",
    "MCAcquisitionFunction",
    "MCAcquisitionObjective",
    "MaxValueBase",
    "MultiOutputAcquisitionFunction",
    "OneShotAcquisitionFunction",
    "NoisyExpectedImprovement",
    "PathwiseThompsonSampling",
    "PenalizedAcquisitionFunction",
    "PenalizedMCObjective",
    "PosteriorMean",
    "PosteriorStandardDeviation",
    "PosteriorTransform",
    "PriorGuidedAcquisitionFunction",
    "ProbabilityOfImprovement",
    "ProjectedAcquisitionFunction",
    "ProximalAcquisitionFunction",
    "SampleReducingMCAcquisitionFunction",
    "ScalarizedPosteriorMean",
    "ScalarizedPosteriorTransform",
    "UpperConfidenceBound",
    "WeightedLogMeanExp",
    "get_acqf_input_constructor",
    "get_acquisition_function",
    "get_optimal_samples",
    "prune_inferior_points",
    "qAnalyticProbabilityOfImprovement",
    "qExpectedImprovement",
    "qJointEntropySearch",
    "qKnowledgeGradient",
    "qLogExpectedImprovement",
    "qLogNoisyExpectedImprovement",
    "qLogProbabilityOfFeasibility",
    "qLowerBoundMaxValueEntropy",
    "qLowerConfidenceBound",
    "qMaxValueEntropy",
    "qMultiStepLookahead",
    "qMultiFidelityKnowledgeGradient",
    "qMultiFidelityLowerBoundMaxValueEntropy",
    "qMultiFidelityMaxValueEntropy",
    "qNegIntegratedPosteriorVariance",
    "qNoisyExpectedImprovement",
    "qPosteriorStandardDeviation",
    "qPredictiveEntropySearch",
    "qProbabilityOfImprovement",
    "qSimpleRegret",
    "qUpperConfidenceBound",
]

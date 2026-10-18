"""Acquisition functions (``botorch.acquisition``)."""

from botorch_rs.acquisition.acquisition import AcquisitionFunction, OneShotAcquisitionFunction
from botorch_rs.acquisition.active_learning import (
    PairwiseMCPosteriorVariance,
    qNegIntegratedPosteriorVariance,
)
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
    qExpectedImprovement,
    qLowerConfidenceBound,
    qNoisyExpectedImprovement,
    qPosteriorStandardDeviation,
    qProbabilityOfImprovement,
    qSimpleRegret,
    qUpperConfidenceBound,
)
from botorch_rs.acquisition.multioutput_acquisition import MultiOutputAcquisitionFunction
from botorch_rs.acquisition.objective import (
    ConstrainedMCObjective,
    GenericMCObjective,
    IdentityMCObjective,
    LearnedObjective,
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
from botorch_rs.acquisition.preference import (
    AnalyticExpectedUtilityOfBestOption,
    PairwiseBayesianActiveLearningByDisagreement,
    qExpectedUtilityOfBestOption,
)
from botorch_rs.acquisition.prior_guided import PriorGuidedAcquisitionFunction
from botorch_rs.acquisition.proximal import ProximalAcquisitionFunction
from botorch_rs.acquisition.thompson_sampling import PathwiseThompsonSampling
from botorch_rs.acquisition.utils import get_optimal_samples, prune_inferior_points

__all__ = [
    "AcquisitionFunction",
    "AnalyticAcquisitionFunction",
    "AnalyticExpectedUtilityOfBestOption",
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
    "LearnedObjective",
    "LinearMCObjective",
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
    "PairwiseBayesianActiveLearningByDisagreement",
    "PairwiseMCPosteriorVariance",
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
    "ScalarizedPosteriorMean",
    "ScalarizedPosteriorTransform",
    "UpperConfidenceBound",
    "WeightedLogMeanExp",
    "get_optimal_samples",
    "prune_inferior_points",
    "qAnalyticProbabilityOfImprovement",
    "qExpectedImprovement",
    "qExpectedUtilityOfBestOption",
    "qJointEntropySearch",
    "qKnowledgeGradient",
    "qLogExpectedImprovement",
    "qLogNoisyExpectedImprovement",
    "qLogProbabilityOfFeasibility",
    "qLowerBoundMaxValueEntropy",
    "qLowerConfidenceBound",
    "qMaxValueEntropy",
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

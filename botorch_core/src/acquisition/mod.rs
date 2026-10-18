//! Acquisition functions (`botorch.acquisition`).
//!
//! An [`AcquisitionFunction`] maps a candidate set — a `q × d` matrix of
//! raw-space points, one per row — to a scalar to be maximized. Values are
//! exactly those of BoTorch's `acqf(X)` for `X` of shape `1 × q × d`, and
//! [`AcquisitionFunction::forward_with_grad`] returns the exact gradient with
//! respect to `X` (what `torch.autograd` computes for the same expression).
//!
//! * [`analytic`] — `ExpectedImprovement`, `LogExpectedImprovement`,
//!   `ProbabilityOfImprovement`, `LogProbabilityOfImprovement`,
//!   `UpperConfidenceBound`, `PosteriorMean`, `PosteriorStandardDeviation`,
//!   `LogConstrainedExpectedImprovement`, `ConstrainedExpectedImprovement`
//!   (`q = 1`).
//! * [`monte_carlo`] — `qExpectedImprovement`, `qProbabilityOfImprovement`,
//!   `qUpperConfidenceBound`, `qSimpleRegret` (any `q`, pending points, MC
//!   objectives, outcome constraints).
//! * [`logei`] — `qLogExpectedImprovement`, `qLogNoisyExpectedImprovement`,
//!   `qLogProbabilityOfFeasibility`.
//! * [`objective`] — `ScalarizedPosteriorTransform`; acquisition functions
//!   evaluate a model through a posterior transform when given a
//!   [`ScalarizedModel`] (see [`ScalarizedModel::wrap`]).
//! * [`mc_objective`] — MC objectives (`IdentityMCObjective`,
//!   `LinearMCObjective`, `GenericMCObjective`, `ConstrainedMCObjective`) on
//!   posterior samples ([`McSamples`]); outcome constraints live in
//!   [`crate::utils::objective`].
//! * [`utils`] — `prune_inferior_points`, `compute_best_feasible_objective`,
//!   `get_infeasible_cost`.
//! * [`multioutput_acquisition`] — multi-output acquisition functions for
//!   NSGA-II (`MultiOutputPosteriorMean`, `MultiOutputAcquisitionFunctionWrapper`).
//! * [`risk_measures`] — risk measures as MC objectives (`CVaR`, `VaR`,
//!   `WorstCase`, `Expectation`) for models with an `InputPerturbation`.
//! * [`multi_objective`] — multi-objective acquisition functions (qLogNEHVI,
//!   qLogEHVI, qNEHVI, qEHVI, analytic EHVI, qLogNParEGO), multi-output MC
//!   objectives and `prune_inferior_points_multi_objective`.
//! * [`composite`] — [`WeightedLogMeanExp`], the weighted log-mean-exp blend
//!   of log-scale acquisition functions (a botorch-rust extension for
//!   transfer acquisition functions).
//! * [`ensemble`] — [`EnsembleAverage`], the average of the acquisition
//!   functions of the members of an ensemble model (BoTorch's
//!   `average_over_ensemble_models`).
//! * [`cost_aware`] — cost-aware utilities (`InverseCostWeightedUtility`,
//!   `GenericCostAwareUtility`); [`fidelity`] — `project_to_target_fidelity`
//!   and `expand_trace_observations`; [`one_shot`] — the interface of one-shot
//!   acquisition functions.
//!
//! Acquisition functions require unbatched models; multi-output models need
//! a posterior transform or (for the MC acquisition functions) an MC
//! objective, except for the constrained analytic acquisition functions and
//! `qLogProbabilityOfFeasibility`, which read several outputs.
//!
//! Acquisition functions hold their model as an `Arc<dyn Model>` and are
//! `Send + Sync`; pending points are stored behind a lock so that
//! [`AcquisitionFunction::set_x_pending`] takes `&self` (like BoTorch's
//! mutable `set_X_pending`).

use std::fmt::Debug;
use std::sync::Arc;

use rayon::prelude::*;

use crate::error::{BotorchError, Result};
use crate::models::Model;
use crate::Matrix;

pub mod active_learning;
pub mod analytic;
pub mod composite;
pub mod cost_aware;
pub(crate) mod dual;
pub mod ensemble;
pub mod fidelity;
pub mod joint_entropy_search;
pub mod knowledge_gradient;
pub mod logei;
pub mod max_value_entropy_search;
pub mod mc_objective;
pub mod monte_carlo;
pub mod multi_objective;
pub mod multioutput_acquisition;
pub mod objective;
pub mod one_shot;
pub mod optimal_samples;
pub mod penalized;
pub mod predictive_entropy_search;
pub mod preference;
pub mod risk_measures;
pub mod thompson_sampling;
pub mod utils;
pub mod wrappers;

pub use active_learning::QNegIntegratedPosteriorVariance;
pub use analytic::{
    ConstrainedExpectedImprovement, ExpectedImprovement, LogConstrainedExpectedImprovement,
    LogExpectedImprovement, LogNoisyExpectedImprovement, LogProbabilityOfFeasibility,
    LogProbabilityOfImprovement, NoisyExpectedImprovement, OutputConstraint, PosteriorMean,
    PosteriorStandardDeviation, ProbabilityOfImprovement, QAnalyticProbabilityOfImprovement,
    ScalarizedPosteriorMean, UpperConfidenceBound,
};
pub use composite::WeightedLogMeanExp;
pub use ensemble::EnsembleAverage;
pub use joint_entropy_search::{
    JesEstimationType, QJointEntropySearch, QJointEntropySearchOptions,
};
pub use knowledge_gradient::{
    KgValueFunction, ProjectedAcquisitionFunction, QKnowledgeGradient, QKnowledgeGradientOptions,
    QMultiFidelityKnowledgeGradient, QMultiFidelityKnowledgeGradientOptions,
};
pub use logei::{
    QLogExpectedImprovement, QLogExpectedImprovementOptions, QLogNoisyExpectedImprovement,
    QLogNoisyExpectedImprovementOptions, QLogProbabilityOfFeasibility,
    QLogProbabilityOfFeasibilityOptions,
};
pub use max_value_entropy_search::{
    MaxValueEntropyOptions, MultiFidelityMaxValueEntropyOptions, QLowerBoundMaxValueEntropy,
    QMaxValueEntropy, QMultiFidelityLowerBoundMaxValueEntropy, QMultiFidelityMaxValueEntropy,
};
pub use mc_objective::{
    ConstrainedMcObjective, GenericMcObjective, IdentityMcObjective, LinearMcObjective,
    McObjective, McSamples, SamplePullback, SampleValues,
};
pub use monte_carlo::{
    McAcquisitionOptions, QExpectedImprovement, QLowerConfidenceBound, QNoisyExpectedImprovement,
    QNoisyExpectedImprovementOptions, QPosteriorStandardDeviation, QProbabilityOfImprovement,
    QSimpleRegret, QUpperConfidenceBound, SampleUtility,
};
pub use objective::{ScalarizedModel, ScalarizedPosteriorTransform, SCALARIZE_TRANSFORMED_MESSAGE};
pub use optimal_samples::{get_optimal_samples, OptimalSamplesOptions, SampleTransform};
pub use penalized::{
    GaussianPenalty, GroupLassoPenalty, L0Approximation, L0PenaltyApprox, L0PenaltyApproxObjective,
    L1Penalty, L1PenaltyObjective, L2Penalty, PenalizedAcquisitionFunction, PenalizedMcObjective,
    Penalty, PointPenalty,
};
pub use predictive_entropy_search::QPredictiveEntropySearch;
pub use preference::{
    AnalyticExpectedUtilityOfBestOption, ComposedModel,
    PairwiseBayesianActiveLearningByDisagreement, PairwiseMcPosteriorVariance,
};
pub use thompson_sampling::{PathwiseThompsonSampling, PathwiseThompsonSamplingOptions};
pub use utils::{
    compute_best_feasible_objective, estimate_objective_lower_bound, get_infeasible_cost,
    prune_inferior_points, prune_inferior_points_indices, PruneOptions,
};
pub use wrappers::{
    FixedFeatureAcquisitionFunction, PriorGuidedAcquisitionFunction, PriorModule,
    ProximalAcquisitionFunction,
};

/// An acquisition function `α(X)` over candidate sets `X` (`q × d`, raw input
/// space), to be maximized (BoTorch's `AcquisitionFunction`).
pub trait AcquisitionFunction: Send + Sync + Debug {
    /// The model the acquisition function is defined on.
    fn model(&self) -> &Arc<dyn Model>;

    /// Input dimension `d`.
    fn dim(&self) -> usize {
        self.model().dim()
    }

    /// The required number of points per candidate set (`Some(1)` for analytic
    /// acquisition functions), or `None` if any `q` is supported.
    fn expected_q(&self) -> Option<usize> {
        None
    }

    /// The acquisition value of the candidate set `x` (`q × d`), i.e.
    /// BoTorch's `acqf(X)` for `X` of shape `1 × q × d`.
    ///
    /// # Errors
    /// [`BotorchError::DimensionMismatch`] for a wrong number of columns (or
    /// of rows when [`AcquisitionFunction::expected_q`] is set), and model
    /// errors.
    fn forward(&self, x: &Matrix) -> Result<f64>;

    /// The acquisition value and its gradient with respect to `x` (`q × d`).
    ///
    /// # Errors
    /// See [`AcquisitionFunction::forward`].
    fn forward_with_grad(&self, x: &Matrix) -> Result<(f64, Matrix)>;

    /// Values for a batch of candidate sets (BoTorch's `acqf(X)` for `X` of
    /// shape `b × q × d`), evaluated in parallel.
    ///
    /// # Errors
    /// The first error of [`AcquisitionFunction::forward`].
    fn forward_many(&self, xs: &[Matrix]) -> Result<Vec<f64>> {
        xs.par_iter().map(|x| self.forward(x)).collect()
    }

    /// The pending points (`m × d`) that are concatenated to every candidate
    /// set, if any.
    fn x_pending(&self) -> Option<Matrix> {
        None
    }

    /// Informs the acquisition function about pending points (BoTorch's
    /// `set_X_pending`); `None` clears them.
    ///
    /// # Errors
    /// [`BotorchError::Unsupported`] for acquisition functions that do not
    /// account for pending points (the analytic ones, as in BoTorch) and
    /// [`BotorchError::DimensionMismatch`] for a wrong number of columns.
    fn set_x_pending(&self, x_pending: Option<&Matrix>) -> Result<()> {
        let _ = x_pending;
        Err(BotorchError::Unsupported(
            "Analytic acquisition functions do not account for X_pending yet.".into(),
        ))
    }

    /// Whether the acquisition function is known to be non-negative (BoTorch's
    /// `is_nonnegative`), which selects `initialize_q_batch_nonneg` in
    /// [`crate::optim::optimize_acqf`].
    fn is_nonnegative(&self) -> bool {
        false
    }

    /// Whether the values are on a log scale (BoTorch's `_log` attribute).
    fn is_log(&self) -> bool {
        false
    }

    /// The one-shot interface of one-shot acquisition functions (BoTorch's
    /// `OneShotAcquisitionFunction`, e.g. the hypervolume knowledge
    /// gradient), which `optimize_acqf` uses to optimize augmented candidate
    /// sets; `None` for all others.
    fn as_one_shot(&self) -> Option<&dyn one_shot::OneShotAcquisitionFunction> {
        None
    }

    /// The interface of sample-reducing Monte-Carlo acquisition functions
    /// (BoTorch's `SampleReducingMCAcquisitionFunction`), which
    /// [`prior_guided::PriorGuidedAcquisitionFunction`] uses to weight the
    /// utilities of every point; `None` for all others.
    fn as_sample_reducing(&self) -> Option<&dyn SampleReducing> {
        None
    }
}

/// The value of a [`SampleReducing`] acquisition function with point
/// weights and, with gradients, its gradient with respect to the candidates
/// and the adjoints of the weights.
pub type WeightedEvaluation = (f64, Option<(Matrix, Vec<f64>)>);

/// A Monte-Carlo acquisition function whose value reduces per-sample,
/// per-point utilities over the points and the samples (BoTorch's
/// `SampleReducingMCAcquisitionFunction`).
pub trait SampleReducing: Send + Sync {
    /// The value with the utilities of every point (row of `[X; X_pending]`)
    /// multiplied by `weights` (added to them for log utilities) and, with
    /// gradients, the gradients with respect to `x` and to the weights.
    ///
    /// # Errors
    /// The errors of [`AcquisitionFunction::forward`] and a
    /// [`BotorchError::DimensionMismatch`] for a wrong number of weights.
    fn evaluate_weighted(
        &self,
        x: &Matrix,
        weights: &[f64],
        with_grad: bool,
    ) -> Result<WeightedEvaluation>;

    /// Whether the utilities are on a log scale (BoTorch's `_log`).
    fn log_utilities(&self) -> bool;
}

/// Checks that `model` is unbatched and has a single output (`message` is
/// the error for multi-output models).
pub(crate) fn check_model(model: &dyn Model, message: &str) -> Result<()> {
    check_unbatched(model)?;
    if model.num_outputs() != 1 {
        return Err(BotorchError::Unsupported(message.into()));
    }
    Ok(())
}

/// Checks that `model` is unbatched.
pub(crate) fn check_unbatched(model: &dyn Model) -> Result<()> {
    if !model.batch_shape().is_empty() {
        return Err(BotorchError::Unsupported(format!(
            "acquisition functions require unbatched models, got a model with batch shape {:?}",
            model.batch_shape()
        )));
    }
    Ok(())
}

/// Errors for models whose posterior is a BoTorch `TransformedPosterior`
/// (nonlinear outcome transforms), for acquisition functions (`name`) that
/// need the Gaussian posterior (BoTorch fails on its missing
/// `distribution`).
pub(crate) fn require_gaussian_model(model: &dyn Model, name: &str) -> Result<()> {
    if model.transformed_outputs().iter().any(|&t| t) {
        return Err(BotorchError::Unsupported(format!(
            "{name} requires a Gaussian posterior, but the model's nonlinear outcome transform \
             (e.g. Log, Power or Bilog) makes it a TransformedPosterior"
        )));
    }
    Ok(())
}

/// Checks the shape of a candidate set.
pub(crate) fn check_candidates(x: &Matrix, d: usize, expected_q: Option<usize>) -> Result<()> {
    if x.ncols() != d {
        return Err(BotorchError::DimensionMismatch(format!(
            "Expected X with {d} columns (d), got {}",
            x.ncols()
        )));
    }
    if x.nrows() == 0 {
        return Err(BotorchError::DimensionMismatch(
            "X must contain at least one point".into(),
        ));
    }
    if let Some(q) = expected_q {
        if x.nrows() != q {
            return Err(BotorchError::DimensionMismatch(format!(
                "Expected X to be `batch_shape x q={q} x d`, but got X with {} points.",
                x.nrows()
            )));
        }
    }
    Ok(())
}

/// Checks the shape of pending points.
pub(crate) fn check_pending(x_pending: &Matrix, d: usize) -> Result<()> {
    if x_pending.ncols() != d {
        return Err(BotorchError::DimensionMismatch(format!(
            "Expected X_pending with {d} columns, got {}",
            x_pending.ncols()
        )));
    }
    Ok(())
}

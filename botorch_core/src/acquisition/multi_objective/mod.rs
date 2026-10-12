//! Multi-objective acquisition functions (`botorch.acquisition.multi_objective`).
//!
//! * [`objective`] — multi-output MC objectives (`IdentityMCMultiOutputObjective`,
//!   `WeightedMCMultiOutputObjective`, `GenericMCMultiOutputObjective`,
//!   `FeasibilityWeightedMCMultiOutputObjective`).
//! * [`monte_carlo`] — [`QExpectedHypervolumeImprovement`] (qEHVI) and
//!   [`QNoisyExpectedHypervolumeImprovement`] (qNEHVI).
//! * [`logei`] — [`QLogExpectedHypervolumeImprovement`] (qLogEHVI) and
//!   [`QLogNoisyExpectedHypervolumeImprovement`] (qLogNEHVI, BoTorch's
//!   recommended default).
//! * [`analytic`] — [`ExpectedHypervolumeImprovement`] (`q = 1`).
//! * [`parego`] — [`QLogNParEGO`] (qLogNEI of a random augmented Chebyshev
//!   scalarization).
//! * [`utils`] — `get_default_partitioning_alpha`,
//!   `prune_inferior_points_multi_objective`.
//! * [`pareto_samples`] — the sampled Pareto sets and fronts of the
//!   information-theoretic acquisition functions (`sample_optimal_points`,
//!   `random_search_optimizer`) and their box decompositions
//!   (`compute_sample_box_decomposition`).
//! * [`max_value_entropy_search`] —
//!   [`QLowerBoundMultiObjectiveMaxValueEntropySearch`] (MESMO).
//! * [`joint_entropy_search`] —
//!   [`QLowerBoundMultiObjectiveJointEntropySearch`] and the entropy
//!   estimators of both (`"0"`, `"LB"`, `"LB2"`, `"MC"`).
//! * [`predictive_entropy_search`] —
//!   [`QMultiObjectivePredictiveEntropySearch`] (expectation propagation).
//! * [`hypervolume`] — the inclusion–exclusion over the subsets of the
//!   candidates (`compute_subset_indices`, `SubsetIndexCachingMixin`) and the
//!   hypervolume improvement of MC samples with exact gradients.
//! * [`hypervolume_knowledge_gradient`] — [`QHypervolumeKnowledgeGradient`]
//!   (qHVKG, one-shot, decoupled evaluations, cost-aware) and
//!   [`QMultiFidelityHypervolumeKnowledgeGradient`].
//!
//! The MC acquisition functions sample the joint posterior of the candidates
//! (and pending or baseline points) over all model outputs with BoTorch's
//! base samples (`S × n × m`; 128 MC samples by default, BoTorch's default
//! for the multi-objective acquisition functions), map the samples to `m'`
//! objectives with a multi-output objective, weight them with the smoothed
//! feasibility of outcome constraints, and compute the expected (log)
//! hypervolume improvement over a box decomposition of the space not
//! dominated by the current Pareto front: a fixed decomposition for qEHVI and
//! qLogEHVI, and one decomposition per MC sample of the baseline for qNEHVI
//! and qLogNEHVI. Gradients are computed in reverse mode.

use std::sync::Arc;

use crate::acquisition::check_unbatched;
use crate::acquisition::mc_objective::McSamples;
use crate::acquisition::multi_objective::hypervolume::{log_qehvi, qehvi, LogSettings};
use crate::acquisition::multi_objective::objective::check_values;
use crate::error::{BotorchError, Result};
use crate::models::Model;
use crate::sampling::SobolQMCNormalSampler;
use crate::utils::objective::{compute_smoothed_feasibility_indicator, OutcomeConstraints};
use crate::Matrix;

pub mod analytic;
mod entropy_linalg;
pub mod hypervolume;
pub mod hypervolume_knowledge_gradient;
pub mod joint_entropy_search;
pub mod logei;
pub mod max_value_entropy_search;
pub mod monte_carlo;
mod nehvi;
pub mod objective;
pub mod parego;
pub mod pareto_samples;
pub mod predictive_entropy_search;
pub mod utils;

pub use analytic::ExpectedHypervolumeImprovement;
pub use hypervolume::{compute_subset_indices, HypercellBounds, SubsetIndexCache, Subsets};
pub use hypervolume_knowledge_gradient::{
    split_hvkg_fantasy_points, HypervolumeValueFunction, QHypervolumeKnowledgeGradient,
    QHypervolumeKnowledgeGradientOptions, QMultiFidelityHypervolumeKnowledgeGradient,
    QMultiFidelityHypervolumeKnowledgeGradientOptions,
};
pub use joint_entropy_search::{
    compute_entropy_monte_carlo, compute_entropy_noiseless, compute_entropy_upper_bound,
    compute_entropy_with_grad, ConditionedOutput, EntropyGradient, EntropyStatistics,
    EstimationType, LowerBoundMultiObjectiveEntropySearch,
    LowerBoundMultiObjectiveEntropySearchOptions, QLowerBoundMultiObjectiveJointEntropySearch,
    QLowerBoundMultiObjectiveJointEntropySearchOptions,
};
pub use logei::{
    QLogExpectedHypervolumeImprovement, QLogExpectedHypervolumeImprovementOptions,
    QLogNoisyExpectedHypervolumeImprovement, QLogNoisyExpectedHypervolumeImprovementOptions,
};
pub use max_value_entropy_search::{
    QLowerBoundMultiObjectiveMaxValueEntropySearch,
    QLowerBoundMultiObjectiveMaxValueEntropySearchOptions,
};
pub use monte_carlo::{
    QExpectedHypervolumeImprovement, QExpectedHypervolumeImprovementOptions,
    QNoisyExpectedHypervolumeImprovement, QNoisyExpectedHypervolumeImprovementOptions,
};
pub use objective::{
    get_infeasible_cost_per_output, FeasibilityWeightedMcMultiOutputObjective,
    GenericMcMultiOutputObjective, IdentityMcMultiOutputObjective, McMultiOutputObjective,
    MultiOutputValues, WeightedMcMultiOutputObjective,
};
pub use parego::{QLogNParEGO, QLogNParEGOOptions};
pub use pareto_samples::{
    compute_sample_box_decomposition, random_search_optimizer, sample_optimal_point_paths,
    sample_optimal_points, sample_optimal_points_with, RandomSearchOptions,
    SampleOptimalPointsOptions, SamplePartitioning,
};
pub use predictive_entropy_search::{
    augment_factors_with_cached_factors, compute_log_determinant, initialize_predictive_matrices,
    log_cdf_robust, safe_update_omega, update_damping, update_damping_when_converged,
    update_marginals, update_omega, EpFactors, EpSummary, OmegaInputs, PredictiveMatrices,
    QMultiObjectivePredictiveEntropySearch, QMultiObjectivePredictiveEntropySearchOptions,
};
pub use utils::{
    get_default_partitioning_alpha, prune_inferior_points_multi_objective,
    prune_inferior_points_multi_objective_indices, PruneMultiObjectiveOptions,
};

/// The default number of MC samples of the multi-objective MC acquisition
/// functions (`MultiObjectiveMCAcquisitionFunction._default_sample_shape`).
pub const DEFAULT_MO_MC_SAMPLES: usize = 128;

/// The sampler of the options or a default one with
/// [`DEFAULT_MO_MC_SAMPLES`] samples and a random seed.
pub(crate) fn mo_sampler_or_default(
    sampler: Option<Arc<SobolQMCNormalSampler>>,
) -> Result<Arc<SobolQMCNormalSampler>> {
    match sampler {
        Some(s) => Ok(s),
        None => Ok(Arc::new(SobolQMCNormalSampler::new(
            DEFAULT_MO_MC_SAMPLES,
            None,
        )?)),
    }
}

/// How the hypervolume improvement of the samples is reduced: the mean
/// (qEHVI) or the log-space version (qLogEHVI).
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum Reduction {
    Linear,
    Log(LogSettings),
}

impl Reduction {
    fn is_log(&self) -> bool {
        matches!(self, Self::Log(_))
    }
}

/// The value of a sample-level evaluation and the adjoints of the samples
/// and (for objectives that depend on `X`) of the inputs.
pub(crate) struct SampleReduction {
    pub value: f64,
    pub samples_bar: Option<Matrix>,
    pub x_bar: Option<Matrix>,
}

/// Model, sampler, objective and constraints shared by the multi-objective
/// MC acquisition functions (BoTorch's `MultiObjectiveMCAcquisitionFunction`).
#[derive(Debug)]
pub(crate) struct MoMcCore {
    pub model: Arc<dyn Model>,
    pub sampler: Arc<SobolQMCNormalSampler>,
    pub objective: Arc<dyn McMultiOutputObjective>,
    pub constraints: Option<OutcomeConstraints>,
    /// Fat-tailed (`fatmoid`) smoothing of the feasibility indicator.
    pub fat: bool,
    pub subsets: SubsetIndexCache,
}

impl MoMcCore {
    pub fn new(
        model: Arc<dyn Model>,
        sampler: Option<Arc<SobolQMCNormalSampler>>,
        objective: Option<Arc<dyn McMultiOutputObjective>>,
        constraints: Option<OutcomeConstraints>,
        fat: bool,
    ) -> Result<Self> {
        check_unbatched(model.as_ref())?;
        Ok(Self {
            model,
            sampler: mo_sampler_or_default(sampler)?,
            objective: objective
                .unwrap_or_else(|| Arc::new(IdentityMcMultiOutputObjective::default())),
            constraints,
            fat,
            subsets: SubsetIndexCache::default(),
        })
    }

    /// Maps posterior samples of `q` points to objectives, weights them by
    /// the smoothed feasibility and reduces the hypervolume improvement over
    /// `cells`; with `with_grad`, backpropagates to the samples (and `x`).
    pub fn reduce_samples(
        &self,
        samples: &McSamples,
        x: &Matrix,
        cells: &HypercellBounds,
        reduction: Reduction,
        with_grad: bool,
    ) -> Result<SampleReduction> {
        let out = self.objective.evaluate(samples, Some(x), with_grad)?;
        check_values(&out.values, samples)?;
        if out.values.num_outputs() != cells.num_outcomes() {
            return Err(BotorchError::DimensionMismatch(format!(
                "The objective has {} outputs but the box decomposition (reference point) has \
                 {} outcomes.",
                out.values.num_outputs(),
                cells.num_outcomes()
            )));
        }
        let feas = match &self.constraints {
            Some(c) => Some(compute_smoothed_feasibility_indicator(
                c,
                &vec![Some(self.fat); c.len()],
                samples,
                reduction.is_log(),
                with_grad,
            )?),
            None => None,
        };
        let subsets = self.subsets.get(samples.num_points());
        let w = feas.as_ref().map(|f| &f.values);
        let r = match reduction {
            Reduction::Linear => qehvi(&out.values, w, cells, &subsets, with_grad),
            Reduction::Log(st) => log_qehvi(&out.values, w, cells, &subsets, st, with_grad),
        };
        if !with_grad {
            return Ok(SampleReduction {
                value: r.value,
                samples_bar: None,
                x_bar: None,
            });
        }
        let obj_bar = r.obj_bar.expect("gradient requested");
        let (mut sbar, x_bar) = (out.pullback.expect("pullback requested"))(&obj_bar)?;
        if let (Some(ind), Some(fb)) = (feas, r.feas_bar) {
            sbar += ind.backward(&fb)?;
        }
        Ok(SampleReduction {
            value: r.value,
            samples_bar: Some(sbar),
            x_bar,
        })
    }
}

/// BoTorch's check of the reference point against the box decomposition.
pub(crate) fn check_ref_point(ref_point: &[f64], num_outcomes: usize, log: bool) -> Result<()> {
    if ref_point.len() != num_outcomes {
        let msg = if log {
            format!(
                "The dimensionality of the reference point must match the number of outcomes. \
                 Got ref_point with {} elements, but expected {num_outcomes}.",
                ref_point.len()
            )
        } else {
            format!(
                "The length of the reference point must match the number of outcomes. Got \
                 ref_point with {} elements, but expected {num_outcomes}.",
                ref_point.len()
            )
        };
        return Err(BotorchError::InvalidArgument(msg));
    }
    Ok(())
}

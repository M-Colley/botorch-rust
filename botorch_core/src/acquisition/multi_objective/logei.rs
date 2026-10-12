//! Log-space hypervolume improvement (`botorch.acquisition.multi_objective.logei`;
//! Ament et al., *Unexpected Improvements to Expected Improvement for
//! Bayesian Optimization*, NeurIPS 2023): [`QLogExpectedHypervolumeImprovement`]
//! (qLogEHVI) and [`QLogNoisyExpectedHypervolumeImprovement`] (qLogNEHVI,
//! BoTorch's recommended multi-objective acquisition function).
//!
//! The value is BoTorch's `_compute_log_qehvi`: the inclusion–exclusion of
//! [`super::hypervolume`] in log space with `log_fatplus` / `fatmin` (or
//! `log_softplus` / `smooth_amin` with `fat = false`), `logsumexp`,
//! `logdiffexp` and `logmeanexp`. These are the formulas of BoTorch's
//! pure-Python path; BoTorch's optional fused C++ kernel (`_FusedLogAreas`,
//! used on CPUs when it compiles) evaluates the same formulas with
//! reassociated arithmetic (multiplication by `1/τ`, `exp(y)` for the
//! softplus below `-20`), which agrees with the Python path to rounding. The
//! smooth log-improvement of a point over a cell is computed once per cell
//! (as in the fused kernel), not once per subset containing the point.

use std::sync::Arc;

use crate::acquisition::logei::{TAU_MAX, TAU_RELU};
use crate::acquisition::multi_objective::hypervolume::{HypercellBounds, LogSettings};
use crate::acquisition::multi_objective::monte_carlo::{cells_of, ehvi_common, EhviCore};
use crate::acquisition::multi_objective::nehvi::{
    nehvi_accessors, nehvi_common, NehviConfig, NehviCore,
};
use crate::acquisition::multi_objective::objective::McMultiOutputObjective;
use crate::acquisition::multi_objective::utils::PruneMultiObjectiveOptions;
use crate::acquisition::multi_objective::{MoMcCore, Reduction};
use crate::acquisition::AcquisitionFunction;
use crate::error::{BotorchError, Result};
use crate::models::Model;
use crate::sampling::SobolQMCNormalSampler;
use crate::utils::multi_objective::box_decompositions::BoxDecomposition;
use crate::utils::objective::OutcomeConstraints;
use crate::Matrix;

/// The default `tau_max` of qLogNEHVI (`1e-3`, not `TAU_MAX`).
pub const NEHVI_TAU_MAX: f64 = 1e-3;

fn check_tau(tau: f64, name: &str) -> Result<f64> {
    if !tau.is_finite() || tau <= 0.0 {
        return Err(BotorchError::InvalidArgument(format!(
            "{name} must be positive and finite, got {tau}"
        )));
    }
    Ok(tau)
}

/// Options of [`QLogExpectedHypervolumeImprovement`] (BoTorch's keyword
/// arguments).
#[derive(Clone, Debug)]
pub struct QLogExpectedHypervolumeImprovementOptions {
    /// The sampler (default: 128 Sobol samples with a random seed).
    pub sampler: Option<Arc<SobolQMCNormalSampler>>,
    /// The multi-output objective (default: the identity on all outputs).
    pub objective: Option<Arc<dyn McMultiOutputObjective>>,
    /// Outcome constraints with their temperatures (BoTorch's default
    /// `eta = 1e-2` for qLogEHVI).
    pub constraints: Option<OutcomeConstraints>,
    /// Pending points concatenated to every candidate set.
    pub x_pending: Option<Matrix>,
    /// Fat-tailed approximations (default `true`).
    pub fat: bool,
    /// Temperature of the smooth ReLU (default `1e-6`).
    pub tau_relu: f64,
    /// Temperature of the smooth minima (default `1e-2`).
    pub tau_max: f64,
}

impl Default for QLogExpectedHypervolumeImprovementOptions {
    fn default() -> Self {
        Self {
            sampler: None,
            objective: None,
            constraints: None,
            x_pending: None,
            fat: true,
            tau_relu: TAU_RELU,
            tau_max: TAU_MAX,
        }
    }
}

/// MC-based batch Log Expected Hypervolume Improvement (BoTorch's
/// `qLogExpectedHypervolumeImprovement`): a smooth, numerically robust
/// approximation of the logarithm of qEHVI.
#[derive(Debug)]
pub struct QLogExpectedHypervolumeImprovement {
    core: EhviCore,
    settings: LogSettings,
}

impl QLogExpectedHypervolumeImprovement {
    /// `qLogExpectedHypervolumeImprovement(model, ref_point, partitioning,
    /// ...)` with an unbatched box decomposition of the current Pareto front.
    ///
    /// # Errors
    /// As [`super::QExpectedHypervolumeImprovement::new`], and
    /// [`BotorchError::InvalidArgument`] for non-positive temperatures.
    pub fn new(
        model: Arc<dyn Model>,
        ref_point: &[f64],
        partitioning: &dyn BoxDecomposition,
        options: QLogExpectedHypervolumeImprovementOptions,
    ) -> Result<Self> {
        Self::from_cell_bounds(model, ref_point, cells_of(partitioning)?, options)
    }

    /// Like [`QLogExpectedHypervolumeImprovement::new`] with the hypercell
    /// bounds of the decomposition.
    ///
    /// # Errors
    /// See [`QLogExpectedHypervolumeImprovement::new`].
    pub fn from_cell_bounds(
        model: Arc<dyn Model>,
        ref_point: &[f64],
        cells: HypercellBounds,
        options: QLogExpectedHypervolumeImprovementOptions,
    ) -> Result<Self> {
        let settings = LogSettings {
            fat: options.fat,
            tau_relu: check_tau(options.tau_relu, "tau_relu")?,
            tau_max: check_tau(options.tau_max, "tau_max")?,
        };
        Ok(Self {
            core: EhviCore::new(
                model,
                ref_point,
                cells,
                options.sampler,
                options.objective,
                options.constraints,
                options.x_pending,
                options.fat,
                true,
            )?,
            settings,
        })
    }

    /// The reference point.
    pub fn ref_point(&self) -> &[f64] {
        &self.core.ref_point
    }

    /// The hypercell bounds.
    pub fn cell_bounds(&self) -> &HypercellBounds {
        &self.core.cells
    }

    /// The sampler.
    pub fn sampler(&self) -> &Arc<SobolQMCNormalSampler> {
        &self.core.mc.sampler
    }
}

impl AcquisitionFunction for QLogExpectedHypervolumeImprovement {
    ehvi_common!(|a: &Self| Reduction::Log(a.settings));

    fn is_log(&self) -> bool {
        true
    }
}

/// Options of [`QLogNoisyExpectedHypervolumeImprovement`] (BoTorch's keyword
/// arguments).
#[derive(Clone, Debug)]
pub struct QLogNoisyExpectedHypervolumeImprovementOptions {
    /// The sampler (default: 128 Sobol samples with a random seed).
    pub sampler: Option<Arc<SobolQMCNormalSampler>>,
    /// The multi-output objective (default: the identity on all outputs).
    pub objective: Option<Arc<dyn McMultiOutputObjective>>,
    /// Outcome constraints with their temperatures (default `eta = 1e-3`).
    pub constraints: Option<OutcomeConstraints>,
    /// Pending points.
    pub x_pending: Option<Matrix>,
    /// Prune the baseline (default `false`).
    pub prune_baseline: bool,
    /// Settings of the pruning.
    pub prune_options: PruneMultiObjectiveOptions,
    /// Approximation level of the box decompositions (default `0`).
    pub alpha: f64,
    /// Cache box decompositions of pending points (default `true`).
    pub cache_pending: bool,
    /// Maximum number of uncached pending points (default `0`).
    pub max_iep: usize,
    /// Incremental NEHVI (default `true`).
    pub incremental_nehvi: bool,
    /// Low-rank updates of the cached baseline root (default `true`).
    pub cache_root: bool,
    /// Temperature of the smooth ReLU (default `1e-6`).
    pub tau_relu: f64,
    /// Temperature of the smooth minima (default `1e-3`).
    pub tau_max: f64,
    /// Fat-tailed approximations (default `true`).
    pub fat: bool,
}

impl Default for QLogNoisyExpectedHypervolumeImprovementOptions {
    fn default() -> Self {
        Self {
            sampler: None,
            objective: None,
            constraints: None,
            x_pending: None,
            prune_baseline: false,
            prune_options: PruneMultiObjectiveOptions::default(),
            alpha: 0.0,
            cache_pending: true,
            max_iep: 0,
            incremental_nehvi: true,
            cache_root: true,
            tau_relu: TAU_RELU,
            tau_max: NEHVI_TAU_MAX,
            fat: true,
        }
    }
}

/// MC-based batch Log Noisy Expected Hypervolume Improvement (BoTorch's
/// `qLogNoisyExpectedHypervolumeImprovement`): the log-space version of
/// qNEHVI. For non-incremental NEHVI, the improvement of cached pending
/// points is added in log space (`logplusexp(value, log(prev_nehvi))`).
#[derive(Debug)]
pub struct QLogNoisyExpectedHypervolumeImprovement {
    core: NehviCore,
    settings: LogSettings,
}

impl QLogNoisyExpectedHypervolumeImprovement {
    /// `qLogNoisyExpectedHypervolumeImprovement(model, ref_point,
    /// X_baseline, ...)`.
    ///
    /// # Errors
    /// See [`super::QNoisyExpectedHypervolumeImprovement::new`], and
    /// [`BotorchError::InvalidArgument`] for non-positive temperatures.
    pub fn new(
        model: Arc<dyn Model>,
        ref_point: &[f64],
        x_baseline: &Matrix,
        options: QLogNoisyExpectedHypervolumeImprovementOptions,
    ) -> Result<Self> {
        let settings = LogSettings {
            fat: options.fat,
            tau_relu: check_tau(options.tau_relu, "tau_relu")?,
            tau_max: check_tau(options.tau_max, "tau_max")?,
        };
        let mc = MoMcCore::new(
            model,
            options.sampler,
            options.objective,
            options.constraints,
            options.fat,
        )?;
        let cfg = NehviConfig {
            ref_point: ref_point.to_vec(),
            alpha: options.alpha,
            cache_pending: options.cache_pending,
            max_iep: options.max_iep,
            incremental: options.incremental_nehvi,
            cache_root: options.cache_root,
            prune_baseline: options.prune_baseline,
            prune_options: options.prune_options,
        };
        Ok(Self {
            core: NehviCore::new(mc, cfg, x_baseline, options.x_pending)?,
            settings,
        })
    }

    /// The reference point.
    pub fn ref_point(&self) -> &[f64] {
        &self.core.cfg.ref_point
    }

    nehvi_accessors!();
}

impl AcquisitionFunction for QLogNoisyExpectedHypervolumeImprovement {
    nehvi_common!(|a: &Self| Reduction::Log(a.settings));

    fn is_log(&self) -> bool {
        true
    }
}
